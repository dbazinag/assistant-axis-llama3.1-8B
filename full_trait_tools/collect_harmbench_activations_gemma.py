#!/usr/bin/env python3
"""
collect_harmbench_activations_gemma.py

Gemma-4-31B-it variant of collect_harmbench_activations.py.

For each (jailbreak_template, behavior) pair from HarmBench:
  1. Format as a single user message (no system prompt)
  2. Forward pass → extract hidden states at the requested layer(s) at the last
     prompt token (pre-generation position), under chunked_sdpa (Gemma head_dim=512)
  3. Generate response (greedy)
  4. Save metadata + response text to responses.jsonl
     Save activations to activations.pt

The pair construction / output schema are identical to the Llama collector so the
results are directly comparable. Only the model loading + forward/extraction are
swapped to the proven Gemma-4 recipe (AutoModelForCausalLM + trust_remote_code +
attn_implementation="sdpa" + chunked_sdpa_scope()).

Layer convention matches the trait pipeline: layer L = fwd_out.hidden_states[L + 1]
(hidden_states[0] is the embedding output), so the default layer 30 lives in the
same space as the layer-30 trait vectors.

Single GPU only: Gemma-4-31B in bf16 (~62 GB) needs one >=80 GB GPU.

Usage:
  # Test (3 templates x 3 behaviors + DirectRequest)
  .venv_gemma/bin/python full_trait_tools/collect_harmbench_activations_gemma.py \
      --model /path/to/gemma-4-31B-it --test

  # Full run (20 templates x 159 behaviors + DirectRequest = 3339 pairs)
  .venv_gemma/bin/python full_trait_tools/collect_harmbench_activations_gemma.py \
      --model /path/to/gemma-4-31B-it
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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from chunked_sdpa import chunked_sdpa_scope

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
# Local fallbacks (in case the GPU pod has no internet). Inside RunAI pods the PVC
# is mounted at /mnt, so try both prefixes.
JAILBREAKS_LOCAL = [
    "/mnt/dlabscratch1/bazina/HarmBench/baselines/human_jailbreaks/jailbreaks.py",
    "/dlabscratch1/bazina/HarmBench/baselines/human_jailbreaks/jailbreaks.py",
]
BEHAVIORS_LOCAL = [
    "/mnt/dlabscratch1/bazina/HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv",
    "/dlabscratch1/bazina/HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv",
]

LAYERS_TO_SAVE       = [30]      # Gemma-4-31B: layer 30 matches the kept trait vectors
N_JAILBREAK_SAMPLES  = 20
RANDOM_SEED          = 42
MAX_NEW_TOKENS       = 512       # Gemma-4 emits a "thought" channel before the answer

# Templates confirmed to have no {0} placeholder
SKIP_JAILBREAK_INDICES = {16, 44}

# ── Data helpers (identical to the Llama collector) ─────────────────────────────

def fetch_text(url: str, local_fallbacks: Optional[List[str]] = None) -> str:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        for path in (local_fallbacks or []):
            if os.path.exists(path):
                logger.warning(f"URL fetch failed ({e}); using local {path}")
                with open(path, encoding="utf-8") as f:
                    return f.read()
        raise


def load_jailbreaks(source: str) -> List[Tuple[int, str]]:
    """Parse the JAILBREAKS list; return (original_index, template) for valid templates."""
    ns: dict = {}
    exec(source, ns)  # noqa: S102 — file is a single list assignment
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
    """Build the full list of (jailbreak, behavior) pairs (same logic as the Llama run)."""
    rng = random.Random(seed)
    sampled = rng.sample(jailbreaks, min(n_jailbreak_samples, len(jailbreaks)))
    logger.info(f"Sampled jailbreak template indices: {sorted(idx for idx, _ in sampled)}")

    pairs: List[dict] = []
    pair_id = 0

    for bv in behaviors:
        behavior_text     = bv["Behavior"]
        behavior_id       = bv["BehaviorID"]
        semantic_category = bv.get("SemanticCategory", "")

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

        for jb_idx, jb_template in sampled:
            try:
                formatted = jb_template.format(behavior_text)
            except (KeyError, IndexError) as e:
                logger.warning(f"Format error for template {jb_idx} × {behavior_id}: {e}")
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


# ── Model helpers (Gemma-4 recipe) ──────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Proven Gemma-4-31B load (matches the hackathon HF extractor)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def sanitize_text(text: str) -> str:
    """Remove surrogate characters that break the Rust-based fast tokenizer."""
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def get_prompt_input_ids(tokenizer: AutoTokenizer, prompt_text: str) -> torch.Tensor:
    """Single user message, no system prompt; add_generation_prompt=True (pre-gen last token)."""
    prompt_text = sanitize_text(prompt_text)
    conversation = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
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
    """Forward pass (activations) + greedy generation for one pair. None on error."""
    try:
        input_ids = get_prompt_input_ids(tokenizer, pair["formatted_prompt"]).to(device)

        # Forward pass under chunked_sdpa (handles Gemma head_dim=512 prefill).
        # layer L -> hidden_states[L + 1]; take the last prompt token (pos -1).
        with torch.no_grad(), chunked_sdpa_scope():
            fwd_out = model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            layer_acts: Dict[int, torch.Tensor] = {}
            for layer_idx in layers:
                hs = fwd_out.hidden_states[layer_idx + 1]  # [1, seq_len, hidden]
                layer_acts[layer_idx] = hs[0, -1, :].detach().to(torch.float32).cpu()

        del fwd_out
        torch.cuda.empty_cache()

        # Generation (greedy). Decode steps are q_len=1 so chunked_sdpa early-returns;
        # keeping the scope here is safe and matches the hackathon generation path.
        with torch.no_grad(), chunked_sdpa_scope():
            gen_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_ids  = gen_ids[0, input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        del gen_ids
        torch.cuda.empty_cache()

        return {"response_text": response_text, "layer_acts": layer_acts}

    except Exception as e:
        logger.warning(f"Error processing pair {pair['pair_id']} ({pair['behavior_id']}): {e}")
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect HarmBench activations for Gemma-4-31B (single GPU)"
    )
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument(
        "--output_dir", type=str,
        default="full_trait_output/harmbench_activations_gemma",
    )
    parser.add_argument("--layers", type=str, default="30",
                        help="Comma-separated model layer indices to save (default: 30)")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--n_jailbreak_samples", type=int, default=N_JAILBREAK_SAMPLES,
                        help="How many jailbreak templates to sample per behavior")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no_direct_request", action="store_true",
                        help="Skip the DirectRequest (no jailbreak) baseline")
    parser.add_argument("--test", action="store_true",
                        help="Small test: 3 behaviors × 3 jailbreaks + DirectRequest")
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    output_dir = Path(args.output_dir)
    if args.test:
        output_dir = output_dir.parent / (output_dir.name + "_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Fetch HarmBench data ───────────────────────────────────────────────────
    logger.info("Fetching HarmBench data...")
    jb_source    = fetch_text(JAILBREAKS_URL, JAILBREAKS_LOCAL)
    behavior_csv = fetch_text(BEHAVIORS_URL, BEHAVIORS_LOCAL)
    jailbreaks = load_jailbreaks(jb_source)
    behaviors  = load_behaviors(behavior_csv)
    logger.info(f"Loaded {len(jailbreaks)} valid jailbreak templates, {len(behaviors)} standard behaviors")

    n_jb = 3 if args.test else args.n_jailbreak_samples
    if args.test:
        behaviors = behaviors[:3]
        logger.info("TEST MODE — 3 behaviors × 3 jailbreaks + DirectRequest")

    pairs = build_pairs(
        jailbreaks=jailbreaks,
        behaviors=behaviors,
        n_jailbreak_samples=n_jb,
        include_direct_request=not args.no_direct_request,
        seed=args.seed,
    )
    logger.info(f"Built {len(pairs)} pairs total")

    # ── Manifest + pairs metadata ──────────────────────────────────────────────
    manifest = {
        "created_at_utc":           datetime.now(timezone.utc).isoformat(),
        "model":                    args.model,
        "n_pairs":                  len(pairs),
        "n_behaviors":              len(behaviors),
        "n_jailbreak_templates":    n_jb,
        "layers_saved":             layers,
        "max_new_tokens":           args.max_new_tokens,
        "seed":                     args.seed,
        "test_mode":                args.test,
        "include_direct_request":   not args.no_direct_request,
        "skipped_template_indices": sorted(SKIP_JAILBREAK_INDICES),
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(output_dir / "pairs_metadata.jsonl", "w") as f:
        for p in pairs:
            row = {k: v for k, v in p.items() if k != "formatted_prompt"}
            f.write(json.dumps(row) + "\n")

    if not torch.cuda.is_available():
        logger.error("No GPU available. Exiting.")
        sys.exit(1)
    device = torch.device("cuda:0")

    # ── Load model + process (single GPU, single process) ──────────────────────
    logger.info(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    logger.info("Model loaded. Processing pairs...")

    responses:   List[dict]      = []
    activations: Dict[int, dict] = {}
    n_errors = 0

    for pair in tqdm(pairs, desc="pairs"):
        result = process_pair(model, tokenizer, pair, layers, args.max_new_tokens, device)
        if result is None:
            n_errors += 1
            continue
        row = {k: v for k, v in pair.items() if k != "formatted_prompt"}
        row["response_text"] = result["response_text"]
        responses.append(row)
        activations[pair["pair_id"]] = {
            str(layer_idx): tensor for layer_idx, tensor in result["layer_acts"].items()
        }
        if len(responses) % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # ── Save (final filenames directly; single process) ────────────────────────
    responses.sort(key=lambda x: x["pair_id"])
    with open(output_dir / "responses.jsonl", "w", encoding="utf-8") as f:
        for row in responses:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    torch.save(activations, output_dir / "activations.pt")

    logger.info(f"Done — {len(responses)} OK, {n_errors} errors.")
    logger.info(f"  responses.jsonl  — metadata + response text per pair")
    logger.info(f"  activations.pt   — {{pair_id: {{'{layers[0]}': Tensor[5376]}}}}")
    logger.info(f"All outputs in: {output_dir}/")


if __name__ == "__main__":
    main()
