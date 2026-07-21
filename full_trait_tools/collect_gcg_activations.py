#!/usr/bin/env python3
"""
collect_gcg_activations.py

Collects pre-generation activations for GCG attack test cases from HarmBench.
This is the transfer experiment counterpart to collect_harmbench_activations.py.

Reads GCG test cases from HarmBench merged test_cases.json, runs Llama-3.1-8B
on each, extracts pre-generation activations at layers 16 & 28, and saves
responses + activations in the same format as collect_harmbench_activations.py
so all downstream scripts (judge, transfer classifier) work identically.

GCG test case format (from HarmBench merge_test_cases.py):
  {behavior_id: ["behavior text + adversarial suffix"]}

Output format (identical to human jailbreak collection):
  responses.jsonl  — one row per pair with metadata + response_text
  activations.pt   — {pair_id: {"16": Tensor[4096], "28": Tensor[4096]}}

Usage:
  uv run full_trait_tools/collect_gcg_activations.py \\
    --test_cases_path /dlabscratch1/bazina/HarmBench/results/GCG/llama3_1_8b/test_cases/test_cases.json

  # Test run (first 5 behaviors)
  uv run full_trait_tools/collect_gcg_activations.py \\
    --test_cases_path /dlabscratch1/bazina/HarmBench/results/GCG/llama3_1_8b/test_cases/test_cases.json \\
    --test
"""

import argparse
import csv
import gc
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LAYERS_TO_SAVE = [16, 28]
MAX_NEW_TOKENS = 300
MODEL_NAME     = "meta-llama/Llama-3.1-8B-Instruct"


def load_test_cases(path: Path) -> Dict[str, List[str]]:
    with open(path, encoding="utf-8") as f:
        test_cases = json.load(f)
    logger.info(f"Loaded test cases for {len(test_cases)} behaviors")
    return test_cases


def load_behaviors(csv_path: Path) -> Dict[str, dict]:
    behaviors = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if True:  # accept all functional categories
                behaviors[row["BehaviorID"]] = row
    logger.info(f"Loaded {len(behaviors)} standard behaviors")
    return behaviors


def build_pairs(test_cases, behaviors):
    pairs = []
    pair_id = 0
    skipped = 0
    for behavior_id, prompts in test_cases.items():
        if behavior_id not in behaviors:
            skipped += 1
            continue
        bv = behaviors[behavior_id]
        for prompt in prompts:
            pairs.append({
                "pair_id":           pair_id,
                "behavior_id":       behavior_id,
                "behavior_text":     bv["Behavior"],
                "semantic_category": bv.get("SemanticCategory", ""),
                "attack_type":       "GCG",
                "jailbreak_idx":     -1,
                "formatted_prompt":  prompt,
            })
            pair_id += 1
    if skipped:
        logger.warning(f"Skipped {skipped} behaviors not in behaviors CSV")
    logger.info(f"Built {len(pairs)} pairs total")
    return pairs


def sanitize_text(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map={"": device},
    )
    model.eval()
    return model, tokenizer


def process_pair(model, tokenizer, pair, layers, max_new_tokens, device):
    try:
        prompt = sanitize_text(pair["formatted_prompt"])
        conversation = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )
        input_ids = tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        attn_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        layer_acts = {}
        for layer_idx in layers:
            hidden = outputs.hidden_states[layer_idx + 1]
            layer_acts[layer_idx] = hidden[0, -1, :].cpu()

        del outputs
        torch.cuda.empty_cache()

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_ids = gen_ids[0, input_ids.shape[1]:]
        response_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        return {"response_text": response_text, "layer_acts": layer_acts}

    except Exception as e:
        logger.warning(f"Error pair {pair['pair_id']} ({pair['behavior_id']}): {e}")
        return None


def worker_fn(worker_id, gpu_id, pairs, model_name, max_new_tokens, output_dir):
    device = torch.device(f"cuda:{gpu_id}")
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    responses = []
    activations = {}
    n_errors = 0

    for pair in tqdm(pairs, desc=f"Worker-{worker_id}", position=worker_id):
        result = process_pair(model, tokenizer, pair, LAYERS_TO_SAVE, max_new_tokens, device)
        if result is None:
            n_errors += 1
            continue

        responses.append({
            "pair_id":           pair["pair_id"],
            "behavior_id":       pair["behavior_id"],
            "behavior_text":     pair["behavior_text"],
            "semantic_category": pair["semantic_category"],
            "attack_type":       pair["attack_type"],
            "jailbreak_idx":     pair["jailbreak_idx"],
            "response_text":     result["response_text"],
        })
        activations[pair["pair_id"]] = {
            str(k): v for k, v in result["layer_acts"].items()
        }

        if len(responses) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    resp_path = output_dir / f"worker_{worker_id}_responses.jsonl"
    acts_path = output_dir / f"worker_{worker_id}_activations.pt"

    with open(resp_path, "w", encoding="utf-8") as f:
        for row in responses:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    torch.save(activations, acts_path)
    logger.info(f"Worker {worker_id}: {len(responses)} OK, {n_errors} errors")


def merge_worker_outputs(output_dir, n_workers):
    all_responses = []
    all_activations = {}
    for wid in range(n_workers):
        rp = output_dir / f"worker_{wid}_responses.jsonl"
        ap = output_dir / f"worker_{wid}_activations.pt"
        if rp.exists():
            with open(rp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_responses.append(json.loads(line))
            rp.unlink()
        if ap.exists():
            all_activations.update(torch.load(ap, map_location="cpu", weights_only=False))
            ap.unlink()

    all_responses.sort(key=lambda x: x["pair_id"])
    with open(output_dir / "responses.jsonl", "w", encoding="utf-8") as f:
        for row in all_responses:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    torch.save(all_activations, output_dir / "activations.pt")
    logger.info(f"Merged {len(all_responses)} responses, {len(all_activations)} activations")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_cases_path", type=str, required=True)
    parser.add_argument("--behaviors_path", type=str,
        default="/dlabscratch1/bazina/HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv")
    parser.add_argument("--output_dir", type=str,
        default="full_trait_output/gcg_activations")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--test", action="store_true", help="First 5 behaviors only")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.test:
        output_dir = output_dir.parent / (output_dir.name + "_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_cases = load_test_cases(Path(args.test_cases_path))
    behaviors  = load_behaviors(Path(args.behaviors_path))

    if args.test:
        bids = list(test_cases.keys())[:5]
        test_cases = {bid: test_cases[bid] for bid in bids}

    pairs = build_pairs(test_cases, behaviors)
    if not pairs:
        logger.error("No pairs built")
        sys.exit(1)

    with open(output_dir / "manifest.json", "w") as f:
        json.dump({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model, "attack_type": "GCG",
            "n_pairs": len(pairs), "layers_saved": LAYERS_TO_SAVE,
            "test_mode": args.test,
        }, f, indent=2)

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_ids = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    if not gpu_ids:
        logger.error("No GPUs found")
        sys.exit(1)

    if args.test:
        gpu_ids = gpu_ids[:1]

    n_workers = len(gpu_ids)
    logger.info(f"Using {n_workers} GPU(s): {gpu_ids}")

    chunks = [[] for _ in range(n_workers)]
    for i, pair in enumerate(pairs):
        chunks[i % n_workers].append(pair)

    worker_kwargs = dict(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        output_dir=output_dir,
    )

    if n_workers == 1:
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

    if n_workers > 1:
        merge_worker_outputs(output_dir, n_workers)
    else:
        for suffix, name in [("responses.jsonl", "responses.jsonl"), ("activations.pt", "activations.pt")]:
            w0 = output_dir / f"worker_0_{suffix}"
            if w0.exists():
                w0.rename(output_dir / name)

    logger.info(f"Done. Outputs: {output_dir}/")
    logger.info("Next steps:")
    logger.info("  1. Judge: uv run full_trait_tools/judge_harmbench_responses.py --classified_path full_trait_output/gcg_activations/responses.jsonl")
    logger.info("  2. Transfer classifier: uv run full_trait_tools/transfer_classifier.py")


if __name__ == "__main__":
    main()
