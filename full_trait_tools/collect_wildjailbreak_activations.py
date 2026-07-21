#!/usr/bin/env python3
"""
collect_wildjailbreak_activations.py

For each harmful pair in the WildJailbreak eval split:
  1. Format as a single user message (no system prompt)
  2. Forward pass  → extract hidden states at layers 16 & 28 at the last
                     prompt token (pre-generation position)
  3. Generate response
  4. Save metadata + response text to responses.jsonl
     Save activations to activations.pt

Two attack_type values:
  wjb_adversarial — adversarial column (jailbreak-wrapped harmful request)
  wjb_direct      — vanilla column    (raw direct harmful request, baseline)

Usage:
  uv run full_trait_tools/collect_wildjailbreak_activations.py
  uv run full_trait_tools/collect_wildjailbreak_activations.py --test
"""

import argparse
import gc
import json
import logging
import os
import sys
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

LAYERS_TO_SAVE  = [16, 28]
MAX_NEW_TOKENS  = 300
RANDOM_SEED     = 42

# ── Data loading ───────────────────────────────────────────────────────────────

def load_wildjailbreak_pairs(split: str = "eval", test_mode: bool = False) -> List[dict]:
    """
    Load adversarial_harmful rows from the WildJailbreak eval config.

    The eval config schema is: adversarial, label, data_type.
    There is no vanilla field — only jailbreak-wrapped prompts.
    We collect rows where data_type == "adversarial_harmful".

    pair dict keys:
      pair_id, behavior_id, behavior_text, semantic_category,
      attack_type, jailbreak_idx, formatted_prompt
    """
    logger.info(f"Loading WildJailbreak ({split} config) from HuggingFace...")
    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        for p in [
            Path.home() / ".cache" / "huggingface" / "token",
            Path("/dlabscratch1/bazina/.cache/huggingface/token"),
            Path("/mnt/dlabscratch1/bazina/.cache/huggingface/token"),
        ]:
            if p.exists():
                token = p.read_text().strip()
                break
    logger.info(f"HF token: {'found (%d chars)' % len(token) if token else 'not found'}")
    # WildJailbreak uses config names ("train"/"eval") not HF split names.
    # Each config has a single "train" split internally.
    ds = load_dataset("allenai/wildjailbreak", name=split, split="train", token=token or None)
    logger.info(f"Dataset size: {len(ds)} rows")

    pairs: List[dict] = []
    pair_id = 0

    for i, row in enumerate(ds):
        dtype = (row.get("data_type") or "").lower()
        # eval config uses underscore format: "adversarial_harmful"
        if dtype != "adversarial_harmful":
            continue

        adversarial = (row.get("adversarial") or "").strip()
        if not adversarial:
            continue

        pairs.append({
            "pair_id":           pair_id,
            "behavior_id":       f"wjb_{i:05d}",
            "behavior_text":     adversarial,
            "semantic_category": "wildjailbreak",
            "attack_type":       "wjb_adversarial",
            "jailbreak_idx":     i,
            "formatted_prompt":  adversarial,
        })
        pair_id += 1

        if test_mode and pair_id >= 10:
            break

    logger.info(f"Built {len(pairs)} adversarial_harmful pairs")
    return pairs


# ── Model helpers ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    trust_remote_code: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def sanitize_text(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def get_prompt_input_ids(tokenizer: AutoTokenizer, prompt_text: str) -> torch.Tensor:
    prompt_text  = sanitize_text(prompt_text)
    conversation = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


def process_pair(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pair: dict,
    layers: List[int],
    max_new_tokens: int,
    device: torch.device,
) -> Optional[dict]:
    try:
        input_ids = get_prompt_input_ids(tokenizer, pair["formatted_prompt"]).to(device)

        with torch.no_grad():
            fwd_out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

        layer_acts: Dict[int, torch.Tensor] = {}
        for layer_idx in layers:
            hs = fwd_out.hidden_states[layer_idx + 1]
            layer_acts[layer_idx] = hs[0, -1, :].detach().cpu()

        del fwd_out
        torch.cuda.empty_cache()

        with torch.no_grad():
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


# ── Worker process ─────────────────────────────────────────────────────────────

def worker_fn(
    worker_id: int,
    gpu_id: int,
    pairs: List[dict],
    model_name: str,
    max_new_tokens: int,
    output_dir: Path,
    trust_remote_code: bool = False,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    logging.basicConfig(
        level=logging.INFO,
        format=(f"%(asctime)s - Worker-{worker_id}[GPU:{gpu_id}] - %(levelname)s - %(message)s"),
        force=True,
    )
    wlogger = logging.getLogger(f"worker_{worker_id}")
    wlogger.info(f"Starting — {len(pairs)} pairs assigned")

    model, tokenizer = load_model_and_tokenizer(model_name, device, trust_remote_code)

    responses:   List[dict]      = []
    activations: Dict[int, dict] = {}
    n_errors = 0

    for pair in tqdm(pairs, desc=f"Worker-{worker_id}", position=worker_id):
        result = process_pair(
            model=model, tokenizer=tokenizer, pair=pair,
            layers=LAYERS_TO_SAVE, max_new_tokens=max_new_tokens, device=device,
        )

        if result is None:
            n_errors += 1
            continue

        row = {k: v for k, v in pair.items() if k != "formatted_prompt"}
        row["response_text"] = result["response_text"]
        responses.append(row)

        activations[pair["pair_id"]] = {
            str(layer_idx): tensor
            for layer_idx, tensor in result["layer_acts"].items()
        }

        if len(responses) % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    resp_path = output_dir / f"worker_{worker_id}_responses.jsonl"
    acts_path  = output_dir / f"worker_{worker_id}_activations.pt"

    with open(resp_path, "w", encoding="utf-8") as f:
        for row in responses:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    torch.save(activations, acts_path)
    wlogger.info(f"Done — {len(responses)} OK, {n_errors} errors.")


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

    all_responses.sort(key=lambda x: x["pair_id"])

    with open(output_dir / "responses.jsonl", "w", encoding="utf-8") as f:
        for row in all_responses:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    torch.save(all_activations, output_dir / "activations.pt")
    logger.info(f"Merged {len(all_responses)} responses, {len(all_activations)} activation entries")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir",     type=str, default="full_trait_output/wildjailbreak_activations")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--seed",           type=int, default=RANDOM_SEED)
    parser.add_argument("--test", action="store_true",
                        help="First 10 pairs only, single GPU")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Pass trust_remote_code=True to AutoModel/AutoTokenizer (needed for OLMo-3)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.test:
        output_dir = output_dir.parent / (output_dir.name + "_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_wildjailbreak_pairs(split="eval", test_mode=args.test)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model":          args.model,
        "dataset":        "allenai/wildjailbreak",
        "split":          "eval",
        "n_pairs":        len(pairs),
        "n_adversarial":  sum(1 for p in pairs if p["attack_type"] == "wjb_adversarial"),
        "layers_saved":   LAYERS_TO_SAVE,
        "seed":           args.seed,
        "test_mode":      args.test,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(output_dir / "pairs_metadata.jsonl", "w") as f:
        for p in pairs:
            row = {k: v for k, v in p.items() if k != "formatted_prompt"}
            f.write(json.dumps(row) + "\n")

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_ids = [int(x.strip()) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    if not gpu_ids:
        logger.error("No GPUs found. Exiting.")
        sys.exit(1)

    if args.test:
        gpu_ids = gpu_ids[:1]

    n_workers = len(gpu_ids)
    logger.info(f"Using {n_workers} GPU(s): {gpu_ids}")

    chunks: List[List[dict]] = [[] for _ in range(n_workers)]
    for i, pair in enumerate(pairs):
        chunks[i % n_workers].append(pair)

    worker_kwargs = dict(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        output_dir=output_dir,
        trust_remote_code=args.trust_remote_code,
    )

    if n_workers == 1:
        worker_fn(0, gpu_ids[0], chunks[0], **worker_kwargs)
    else:
        mp.set_start_method("spawn", force=True)
        processes = []
        for wid in range(n_workers):
            p = mp.Process(target=worker_fn, args=(wid, gpu_ids[wid], chunks[wid]), kwargs=worker_kwargs)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    if n_workers > 1:
        merge_worker_outputs(output_dir, n_workers)
    else:
        w0_resp = output_dir / "worker_0_responses.jsonl"
        w0_acts  = output_dir / "worker_0_activations.pt"
        if w0_resp.exists():
            w0_resp.rename(output_dir / "responses.jsonl")
        if w0_acts.exists():
            w0_acts.rename(output_dir / "activations.pt")

    logger.info(f"\nAll done. Outputs in: {output_dir}/")


if __name__ == "__main__":
    main()
