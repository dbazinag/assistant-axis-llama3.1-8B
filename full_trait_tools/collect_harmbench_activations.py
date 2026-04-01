#!/usr/bin/env python3
"""
collect_harmbench_activations.py

For each (jailbreak_template, behavior) pair from HarmBench:
  1. Format as a single user message (no system prompt)
  2. Forward pass  → extract hidden states at layers 16 & 28 at the last
                     prompt token (pre-generation position)
  3. Generate response
  4. Save metadata + response text to responses.jsonl
     Save activations to activations.pt

Supports multi-GPU via mp.Process (one model instance per GPU).
Skips the 2 broken HarmBench templates that have no {0} placeholder.

Usage:
  # Full run (20 jailbreak templates × 159 behaviors + DirectRequest baseline)
  uv run full_trait_tools/collect_harmbench_activations.py

  # Test run (3 templates × 3 behaviors, single GPU, ~2 min)
  uv run full_trait_tools/collect_harmbench_activations.py --test

  # Explicit GPU assignment
  CUDA_VISIBLE_DEVICES=0,1,2 uv run full_trait_tools/collect_harmbench_activations.py
"""

import argparse
import csv
import gc
import io
import json
import logging
import os
import random
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

JAILBREAKS_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
    "/main/baselines/human_jailbreaks/jailbreaks.py"
)
BEHAVIORS_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
    "/main/data/behavior_datasets/harmbench_behaviors_text_test.csv"
)

LAYERS_TO_SAVE       = [16, 28]
N_JAILBREAK_SAMPLES  = 20
RANDOM_SEED          = 42
MAX_NEW_TOKENS       = 300

# Templates confirmed to have no {0} placeholder
SKIP_JAILBREAK_INDICES = {16, 44}

# ── Data helpers ───────────────────────────────────────────────────────────────

def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")


def load_jailbreaks(source: str) -> List[Tuple[int, str]]:
    """
    Parse JAILBREAKS list from jailbreaks.py source.
    Returns list of (original_index, template_string) for valid templates only.
    """
    ns: dict = {}
    exec(source, ns)  # noqa: S102  — safe, file is a single list assignment
    raw = ns["JAILBREAKS"]
    valid = []
    for i, jb in enumerate(raw):
        if i in SKIP_JAILBREAK_INDICES:
            continue
        if "{0}" not in jb:
            logger.warning(f"Skipping template {i}: no {{0}} placeholder")
            continue
        valid.append((i, jb))
    return valid


def load_behaviors(csv_text: str) -> List[dict]:
    """Load standard (no-context) behaviors from the HarmBench CSV."""
    reader = csv.DictReader(io.StringIO(csv_text))
    return [
        row for row in reader
        if row.get("FunctionalCategory", "").lower() == "standard"
    ]


def build_pairs(
    jailbreaks: List[Tuple[int, str]],
    behaviors: List[dict],
    n_jailbreak_samples: int,
    include_direct_request: bool,
    seed: int,
) -> List[dict]:
    """
    Build the full list of (jailbreak, behavior) pairs to process.

    Each pair dict carries all metadata needed for later analysis:
      pair_id           — unique integer ID across the run
      behavior_id       — HarmBench BehaviorID string
      behavior_text     — the raw harmful behavior string
      semantic_category — HarmBench SemanticCategory
      attack_type       — "human_jailbreak" | "direct_request"
      jailbreak_idx     — original index in JAILBREAKS list (-1 for DirectRequest)
      formatted_prompt  — the full text sent to the model
    """
    rng = random.Random(seed)
    sampled = rng.sample(jailbreaks, min(n_jailbreak_samples, len(jailbreaks)))
    logger.info(
        f"Sampled jailbreak template indices: "
        f"{sorted(idx for idx, _ in sampled)}"
    )

    pairs: List[dict] = []
    pair_id = 0

    for bv in behaviors:
        behavior_text     = bv["Behavior"]
        behavior_id       = bv["BehaviorID"]
        semantic_category = bv.get("SemanticCategory", "")

        # DirectRequest baseline — no jailbreak wrapper
        if include_direct_request:
            pairs.append({
                "pair_id":           pair_id,
                "behavior_id":       behavior_id,
                "behavior_text":     behavior_text,
                "semantic_category": semantic_category,
                "attack_type":       "direct_request",
                "jailbreak_idx":     -1,
                "formatted_prompt":  behavior_text,
            })
            pair_id += 1

        # Human jailbreak templates
        for jb_idx, jb_template in sampled:
            try:
                formatted = jb_template.format(behavior_text)
            except (KeyError, IndexError) as e:
                logger.warning(
                    f"Format error for template {jb_idx} × {behavior_id}: {e}"
                )
                continue
            pairs.append({
                "pair_id":           pair_id,
                "behavior_id":       behavior_id,
                "behavior_text":     behavior_text,
                "semantic_category": semantic_category,
                "attack_type":       "human_jailbreak",
                "jailbreak_idx":     jb_idx,
                "formatted_prompt":  formatted,
            })
            pair_id += 1

    return pairs


# ── Model helpers ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    return model, tokenizer


def get_prompt_input_ids(
    tokenizer: AutoTokenizer,
    prompt_text: str,
) -> torch.Tensor:
    """
    Apply Llama chat template for a single user message with no system prompt.
    Returns input_ids of shape [1, seq_len].
    """
    conversation = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,  # adds the assistant header so model starts generating
    )
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


def process_pair(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pair: dict,
    layers: List[int],
    max_new_tokens: int,
    device: torch.device,
) -> Optional[dict]:
    """
    Run one forward pass (activations) + one generation pass for a single pair.

    Returns dict with:
      response_text  — decoded model response
      layer_acts     — {layer_idx (int): Tensor[hidden_dim]} at last prompt token
    Returns None on any error.
    """
    try:
        input_ids = get_prompt_input_ids(tokenizer, pair["formatted_prompt"])
        input_ids = input_ids.to(device)

        # ── Forward pass: extract pre-generation activations ──────────────────
        # hidden_states[0]     = embedding layer output
        # hidden_states[i + 1] = transformer layer i output  (i = 0..31)
        # We want the last token of the prompt (position -1) as the
        # pre-generation representation.
        with torch.no_grad():
            fwd_out = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )

        layer_acts: Dict[int, torch.Tensor] = {}
        for layer_idx in layers:
            hs = fwd_out.hidden_states[layer_idx + 1]  # [1, seq_len, hidden_dim]
            layer_acts[layer_idx] = hs[0, -1, :].detach().cpu()

        del fwd_out
        torch.cuda.empty_cache()

        # ── Generation pass ───────────────────────────────────────────────────
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,             # greedy — reproducible
                pad_token_id=tokenizer.eos_token_id,
            )

        # Slice off the prompt tokens to get only the response
        response_ids  = gen_ids[0, input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        del gen_ids
        torch.cuda.empty_cache()

        return {
            "response_text": response_text,
            "layer_acts":    layer_acts,
        }

    except Exception as e:
        logger.warning(f"Error processing pair {pair['pair_id']} "
                       f"({pair['behavior_id']}): {e}")
        return None


# ── Worker process ─────────────────────────────────────────────────────────────

def worker_fn(
    worker_id: int,
    gpu_id: int,
    pairs: List[dict],
    model_name: str,
    max_new_tokens: int,
    output_dir: Path,
) -> None:
    """Runs in a subprocess. Processes a subset of pairs on one GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    # Re-configure logging for this subprocess
    logging.basicConfig(
        level=logging.INFO,
        format=(
            f"%(asctime)s - Worker-{worker_id}[GPU:{gpu_id}]"
            " - %(levelname)s - %(message)s"
        ),
        force=True,
    )
    wlogger = logging.getLogger(f"worker_{worker_id}")
    wlogger.info(f"Starting — {len(pairs)} pairs assigned")

    model, tokenizer = load_model_and_tokenizer(model_name, device)

    responses:    List[dict]              = []
    activations:  Dict[int, dict]         = {}   # pair_id -> {layer_idx_str: tensor}
    n_errors = 0

    for pair in tqdm(pairs, desc=f"Worker-{worker_id}", position=worker_id):
        result = process_pair(
            model=model,
            tokenizer=tokenizer,
            pair=pair,
            layers=LAYERS_TO_SAVE,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        if result is None:
            n_errors += 1
            continue

        # Metadata row (everything except the full formatted prompt,
        # which can be reconstructed and is large)
        row = {k: v for k, v in pair.items() if k != "formatted_prompt"}
        row["response_text"] = result["response_text"]
        responses.append(row)

        # Activations keyed by pair_id
        # Store layer keys as strings for JSON-safe serialisation if needed later
        activations[pair["pair_id"]] = {
            str(layer_idx): tensor
            for layer_idx, tensor in result["layer_acts"].items()
        }

        if len(responses) % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # ── Save worker outputs ────────────────────────────────────────────────────
    resp_path = output_dir / f"worker_{worker_id}_responses.jsonl"
    acts_path  = output_dir / f"worker_{worker_id}_activations.pt"

    with open(resp_path, "w", encoding="utf-8") as f:
        for row in responses:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    torch.save(activations, acts_path)

    wlogger.info(
        f"Done — {len(responses)} OK, {n_errors} errors. "
        f"Saved to {resp_path.name} + {acts_path.name}"
    )


# ── Merge ──────────────────────────────────────────────────────────────────────

def merge_worker_outputs(output_dir: Path, n_workers: int) -> None:
    all_responses:   List[dict]      = []
    all_activations: Dict[int, dict] = {}

    for wid in range(n_workers):
        resp_path = output_dir / f"worker_{wid}_responses.jsonl"
        acts_path  = output_dir / f"worker_{wid}_activations.pt"

        if resp_path.exists():
            with open(resp_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_responses.append(json.loads(line))
            resp_path.unlink()

        if acts_path.exists():
            worker_acts = torch.load(acts_path, map_location="cpu", weights_only=False)
            all_activations.update(worker_acts)
            acts_path.unlink()

    # Sort by pair_id for deterministic output
    all_responses.sort(key=lambda x: x["pair_id"])

    final_resp = output_dir / "responses.jsonl"
    final_acts  = output_dir / "activations.pt"

    with open(final_resp, "w", encoding="utf-8") as f:
        for row in all_responses:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    torch.save(all_activations, final_acts)

    logger.info(
        f"Merged {len(all_responses)} responses and "
        f"{len(all_activations)} activation entries"
    )
    logger.info(f"  responses.jsonl  — metadata + response text per pair")
    logger.info(f"  activations.pt   — {{pair_id: {{'16': Tensor[4096], '28': Tensor[4096]}}}}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect HarmBench activations for jailbreak prediction"
    )
    parser.add_argument(
        "--model", type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="full_trait_output/harmbench_activations",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--n_jailbreak_samples", type=int, default=N_JAILBREAK_SAMPLES,
        help="How many jailbreak templates to sample per behavior",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
    )
    parser.add_argument(
        "--no_direct_request", action="store_true",
        help="Skip the DirectRequest (no jailbreak) baseline",
    )
    parser.add_argument(
        "--test", action="store_true",
        help=(
            "Run a small test: 3 behaviors × 3 jailbreaks + DirectRequest, "
            "single GPU, ~2 min"
        ),
    )
    args = parser.parse_args()

    # Redirect output dir for test runs
    output_dir = Path(args.output_dir)
    if args.test:
        output_dir = output_dir.parent / (output_dir.name + "_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Fetch data ─────────────────────────────────────────────────────────────
    logger.info("Fetching HarmBench data from GitHub...")
    jb_source    = fetch_text(JAILBREAKS_URL)
    behavior_csv = fetch_text(BEHAVIORS_URL)

    jailbreaks = load_jailbreaks(jb_source)
    behaviors  = load_behaviors(behavior_csv)
    logger.info(
        f"Loaded {len(jailbreaks)} valid jailbreak templates, "
        f"{len(behaviors)} standard behaviors"
    )

    # ── Test mode overrides ────────────────────────────────────────────────────
    n_jb = 3 if args.test else args.n_jailbreak_samples
    if args.test:
        behaviors = behaviors[:3]
        logger.info("TEST MODE — 3 behaviors × 3 jailbreaks + DirectRequest")

    # ── Build pairs ────────────────────────────────────────────────────────────
    pairs = build_pairs(
        jailbreaks=jailbreaks,
        behaviors=behaviors,
        n_jailbreak_samples=n_jb,
        include_direct_request=not args.no_direct_request,
        seed=args.seed,
    )
    logger.info(f"Built {len(pairs)} pairs total")

    # ── Save manifest ──────────────────────────────────────────────────────────
    manifest = {
        "created_at_utc":             datetime.now(timezone.utc).isoformat(),
        "model":                      args.model,
        "n_pairs":                    len(pairs),
        "n_behaviors":                len(behaviors),
        "n_jailbreak_templates":      n_jb,
        "layers_saved":               LAYERS_TO_SAVE,
        "seed":                       args.seed,
        "test_mode":                  args.test,
        "include_direct_request":     not args.no_direct_request,
        "skipped_template_indices":   sorted(SKIP_JAILBREAK_INDICES),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Save pairs metadata (no formatted_prompt — large and reconstructable)
    with open(output_dir / "pairs_metadata.jsonl", "w") as f:
        for p in pairs:
            row = {k: v for k, v in p.items() if k != "formatted_prompt"}
            f.write(json.dumps(row) + "\n")

    # ── GPU setup ──────────────────────────────────────────────────────────────
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_ids = [
            int(x.strip())
            for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            if x.strip()
        ]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    if not gpu_ids:
        logger.error("No GPUs found. Exiting.")
        sys.exit(1)

    if args.test:
        gpu_ids = gpu_ids[:1]

    n_workers = len(gpu_ids)
    logger.info(f"Using {n_workers} GPU(s): {gpu_ids}")

    # ── Distribute pairs evenly across workers ─────────────────────────────────
    chunks: List[List[dict]] = [[] for _ in range(n_workers)]
    for i, pair in enumerate(pairs):
        chunks[i % n_workers].append(pair)

    for wid in range(n_workers):
        logger.info(f"  Worker {wid} (GPU {gpu_ids[wid]}): {len(chunks[wid])} pairs")

    # ── Launch ─────────────────────────────────────────────────────────────────
    worker_kwargs = dict(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        output_dir=output_dir,
    )

    if n_workers == 1:
        # Single GPU — run in-process for easier debugging
        worker_fn(0, gpu_ids[0], chunks[0], **worker_kwargs)
    else:
        mp.set_start_method("spawn", force=True)
        processes = []
        for wid in range(n_workers):
            p = mp.Process(
                target=worker_fn,
                args=(wid, gpu_ids[wid], chunks[wid]),
                kwargs=worker_kwargs,
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    # ── Merge worker outputs ───────────────────────────────────────────────────
    if n_workers > 1:
        logger.info("Merging worker outputs...")
        merge_worker_outputs(output_dir, n_workers)
    else:
        # Single worker: rename to final filenames
        w0_resp = output_dir / "worker_0_responses.jsonl"
        w0_acts  = output_dir / "worker_0_activations.pt"
        if w0_resp.exists():
            w0_resp.rename(output_dir / "responses.jsonl")
        if w0_acts.exists():
            w0_acts.rename(output_dir / "activations.pt")

    logger.info(f"\nAll done. Outputs in: {output_dir}/")


if __name__ == "__main__":
    main()
