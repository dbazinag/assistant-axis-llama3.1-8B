#!/usr/bin/env python3
"""
collect_attack_activations.py

Generic activation collector for new attack families (PAP, GPTFuzz, PEZ).
Mirrors collect_gcg_activations.py but works with all HarmBench behaviors
(not just standard) since PAP/GPTFuzz/PEZ ran on harmbench_behaviors_text_all.csv.

Usage:
  uv run full_trait_tools/collect_attack_activations.py --attack_type PAP
  uv run full_trait_tools/collect_attack_activations.py --attack_type GPTFuzz
  uv run full_trait_tools/collect_attack_activations.py --attack_type PEZ
  uv run full_trait_tools/collect_attack_activations.py --attack_type PAP --test
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
from typing import Dict, List

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LAYERS_TO_SAVE = [16, 28]
MAX_NEW_TOKENS = 300
MODEL_NAME     = "meta-llama/Llama-3.1-8B-Instruct"
HARMBENCH_ROOT = "/dlabscratch1/bazina/HarmBench"


ATTACK_CONFIGS = {
    "PAP": {
        "test_cases_path": f"{HARMBENCH_ROOT}/results/PAP/llama3_1_8b/test_cases/test_cases.json",
        "output_dir":      "full_trait_output/pap_activations",
    },
    "GPTFuzz": {
        "test_cases_path": f"{HARMBENCH_ROOT}/results/GPTFuzz/llama3_1_8b/test_cases/test_cases.json",
        "output_dir":      "full_trait_output/gptfuzz_activations",
    },
    "PEZ": {
        "test_cases_path": f"{HARMBENCH_ROOT}/results/PEZ/llama3_1_8b/test_cases/test_cases.json",
        "output_dir":      "full_trait_output/pez_activations",
    },
}


def load_test_cases(path: Path) -> Dict[str, List[str]]:
    with open(path, encoding="utf-8") as f:
        test_cases = json.load(f)
    logger.info(f"Loaded test cases for {len(test_cases)} behaviors")
    return test_cases


def load_behaviors(csv_path: Path) -> Dict[str, dict]:
    """Load ALL behaviors (not just standard) since new attacks ran on full CSV."""
    behaviors = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            behaviors[row["BehaviorID"]] = row
    logger.info(f"Loaded {len(behaviors)} behaviors")
    return behaviors


def build_pairs(test_cases, behaviors, attack_type):
    pairs = []
    pair_id = 0
    skipped = 0
    for behavior_id, prompts in test_cases.items():
        if behavior_id not in behaviors:
            skipped += 1
            continue
        bv = behaviors[behavior_id]

        # Flatten nested lists — PAP saves [[p1, p2, p3]] instead of [p1, p2, p3]
        flat_prompts = []
        for p in prompts:
            if isinstance(p, list):
                flat_prompts.extend(p)
            elif isinstance(p, str):
                flat_prompts.append(p)

        for prompt in flat_prompts:
            if not prompt or not isinstance(prompt, str) or not prompt.strip():
                continue
            pairs.append({
                "pair_id":             pair_id,
                "behavior_id":         behavior_id,
                "behavior_text":       bv["Behavior"],
                "semantic_category":   bv.get("SemanticCategory", ""),
                "functional_category": bv.get("FunctionalCategory", ""),
                "attack_type":         attack_type,
                "jailbreak_idx":       -1,
                "formatted_prompt":    prompt,
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


def worker_fn(worker_id, gpu_id, pairs, model_name, max_new_tokens, output_dir, attack_type):
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
            "pair_id":             pair["pair_id"],
            "behavior_id":         pair["behavior_id"],
            "behavior_text":       pair["behavior_text"],
            "semantic_category":   pair["semantic_category"],
            "functional_category": pair["functional_category"],
            "attack_type":         pair["attack_type"],
            "jailbreak_idx":       pair["jailbreak_idx"],
            "response_text":       result["response_text"],
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
    parser.add_argument("--attack_type", type=str, required=True,
                        choices=["PAP", "GPTFuzz", "PEZ"],
                        help="Attack family to collect activations for")
    parser.add_argument("--behaviors_path", type=str,
                        default=f"{HARMBENCH_ROOT}/data/behavior_datasets/harmbench_behaviors_text_all.csv")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--test", action="store_true", help="First 5 behaviors only")
    args = parser.parse_args()

    cfg = ATTACK_CONFIGS[args.attack_type]
    output_dir = Path(cfg["output_dir"])
    if args.test:
        output_dir = output_dir.parent / (output_dir.name + "_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_cases = load_test_cases(Path(cfg["test_cases_path"]))
    behaviors  = load_behaviors(Path(args.behaviors_path))

    if args.test:
        bids = list(test_cases.keys())[:5]
        test_cases = {bid: test_cases[bid] for bid in bids}

    pairs = build_pairs(test_cases, behaviors, args.attack_type)
    if not pairs:
        logger.error("No pairs built — check test_cases.json and behaviors CSV")
        sys.exit(1)

    with open(output_dir / "manifest.json", "w") as f:
        json.dump({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "attack_type": args.attack_type,
            "n_pairs": len(pairs),
            "layers_saved": LAYERS_TO_SAVE,
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
        attack_type=args.attack_type,
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
        for suffix in ["responses.jsonl", "activations.pt"]:
            w0 = output_dir / f"worker_0_{suffix}"
            if w0.exists():
                w0.rename(output_dir / suffix)

    logger.info(f"Done. Outputs in {output_dir}/")
    logger.info(f"Next: judge responses then run transfer classifier on {args.attack_type}")


if __name__ == "__main__":
    main()
