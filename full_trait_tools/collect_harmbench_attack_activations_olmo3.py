#!/usr/bin/env python3
"""
collect_harmbench_attack_activations_olmo3.py

Run OLMo-3-7B-Instruct on HarmBench attack test cases (PAIR/PAP/GPTFuzz/PEZ)
and collect layer-16 & 28 hidden states + responses.

Reads from HarmBench/results/HarmBench_{attack_type}/olmo3_*/test_cases/
test_cases_individual_behaviors/*/test_cases.json (one subdir per behavior).

Output format matches collect_attack_activations.py so judge_harmbench_responses.py
and run_all_traits_sweep_v2.py can consume it directly.

Usage:
  uv run python full_trait_tools/collect_harmbench_attack_activations_olmo3.py --attack_type PAIR
  uv run python full_trait_tools/collect_harmbench_attack_activations_olmo3.py --attack_type PAP
  uv run python full_trait_tools/collect_harmbench_attack_activations_olmo3.py --attack_type GPTFuzz
  uv run python full_trait_tools/collect_harmbench_attack_activations_olmo3.py --attack_type PEZ
  uv run python full_trait_tools/collect_harmbench_attack_activations_olmo3.py --attack_type PAIR --test
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

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME     = "allenai/OLMo-3-7B-Instruct"
LAYERS_TO_SAVE = [16, 28]
MAX_NEW_TOKENS = 300
HARMBENCH_ROOT = Path("/dlabscratch1/bazina/HarmBench")

ATTACK_CONFIGS = {
    "PAIR":     {"exp": "olmo3_7b_harmbench"},
    "PAP":      {"exp": "olmo3_7b_top5"},
    "GPTFuzz":  {"exp": "olmo3_7b"},
    "PEZ":      {"exp": "olmo3_7b"},
}


def load_individual_test_cases(attack_type: str) -> dict[str, list]:
    """Merge per-behavior test_cases.json files into {behavior_id: [prompt, ...]}."""
    exp = ATTACK_CONFIGS[attack_type]["exp"]
    ind_dir = (HARMBENCH_ROOT / "results" / f"HarmBench_{attack_type}" / exp
               / "test_cases" / "test_cases_individual_behaviors")
    if not ind_dir.exists():
        raise FileNotFoundError(f"Not found: {ind_dir}")

    merged: dict[str, list] = {}
    for bdir in sorted(ind_dir.iterdir()):
        if not bdir.is_dir():
            continue
        tc_file = bdir / "test_cases.json"
        if not tc_file.exists():
            continue
        with tc_file.open() as f:
            d = json.load(f)
        for bid, prompts in d.items():
            # Flatten nested lists (PAP/GPTFuzz store [[p1, p2], ...])
            flat: list[str] = []
            for p in prompts:
                if isinstance(p, list):
                    flat.extend(p)
                elif isinstance(p, str):
                    flat.append(p)
            if flat:
                merged[bid] = merged.get(bid, []) + flat

    logger.info(f"Merged {len(merged)} behaviors for {attack_type}")
    return merged


def load_behaviors(csv_path: Path) -> dict[str, dict]:
    behaviors = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            behaviors[row["BehaviorID"]] = row
    logger.info(f"Loaded {len(behaviors)} behaviors from CSV")
    return behaviors


def build_pairs(test_cases: dict, behaviors: dict, attack_type: str) -> list[dict]:
    pairs = []
    pair_id = 0
    skipped = 0
    for behavior_id, prompts in test_cases.items():
        bv = behaviors.get(behavior_id)
        if bv is None:
            skipped += 1
            continue
        for prompt in prompts:
            if not prompt or not isinstance(prompt, str) or not prompt.strip():
                continue
            pairs.append({
                "pair_id":           pair_id,
                "behavior_id":       behavior_id,
                "behavior_text":     bv["Behavior"],
                "semantic_category": bv.get("SemanticCategory", ""),
                "attack_type":       attack_type.lower(),
                "jailbreak_idx":     -1,
                "formatted_prompt":  prompt.strip(),
            })
            pair_id += 1
    if skipped:
        logger.warning(f"Skipped {skipped} behaviors not in behaviors CSV")
    logger.info(f"Built {len(pairs)} pairs")
    return pairs


def load_model_and_tokenizer(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def process_pair(model, tokenizer, pair, device):
    try:
        prompt = pair["formatted_prompt"].encode("utf-8", errors="replace").decode("utf-8")
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
        for layer_idx in LAYERS_TO_SAVE:
            hidden = outputs.hidden_states[layer_idx + 1]
            layer_acts[layer_idx] = hidden[0, -1, :].cpu()

        del outputs
        torch.cuda.empty_cache()

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_ids = gen_ids[0, input_ids.shape[1]:]
        response_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        return {"response_text": response_text, "layer_acts": layer_acts}

    except Exception as e:
        logger.warning(f"Error pair {pair['pair_id']} ({pair['behavior_id']}): {e}")
        return None


def worker_fn(worker_id, gpu_id, pairs, model_name, output_dir):
    device = torch.device(f"cuda:{gpu_id}")
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    responses = []
    activations = {}
    n_errors = 0

    for pair in tqdm(pairs, desc=f"Worker-{worker_id}", position=worker_id):
        result = process_pair(model, tokenizer, pair, device)
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
        activations[pair["pair_id"]] = {str(k): v for k, v in result["layer_acts"].items()}

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


def merge_worker_outputs(output_dir: Path, n_workers: int):
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
    parser.add_argument("--attack_type", required=True, choices=["PAIR", "PAP", "GPTFuzz", "PEZ"])
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--behaviors_path",
                        default=str(HARMBENCH_ROOT / "data/behavior_datasets/harmbench_behaviors_text_all.csv"))
    parser.add_argument("--output_base", default="full_trait_output",
                        help="Parent dir; output goes to {output_base}/{attack_type.lower()}_activations_olmo3_hb")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--test", action="store_true", help="First 5 behaviors only")
    args = parser.parse_args()

    out_name = f"{args.attack_type.lower()}_activations_olmo3_hb"
    if args.test:
        out_name += "_test"
    output_dir = Path(args.output_base) / out_name
    output_dir.mkdir(parents=True, exist_ok=True)

    test_cases = load_individual_test_cases(args.attack_type)
    behaviors  = load_behaviors(Path(args.behaviors_path))

    if args.test:
        bids = list(test_cases.keys())[:5]
        test_cases = {bid: test_cases[bid] for bid in bids}

    pairs = build_pairs(test_cases, behaviors, args.attack_type)
    if not pairs:
        logger.error("No pairs built — check test case dirs and behaviors CSV")
        sys.exit(1)

    with open(output_dir / "manifest.json", "w") as f:
        json.dump({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model":          args.model,
            "attack_type":    args.attack_type,
            "n_behaviors":    len(test_cases),
            "n_pairs":        len(pairs),
            "layers_saved":   LAYERS_TO_SAVE,
            "test_mode":      args.test,
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
    logger.info(f"Attack: {args.attack_type} | Pairs: {len(pairs)} | GPUs: {gpu_ids}")

    chunks = [[] for _ in range(n_workers)]
    for i, pair in enumerate(pairs):
        chunks[i % n_workers].append(pair)

    procs = []
    for wid, (gid, chunk) in enumerate(zip(gpu_ids, chunks)):
        p = mp.Process(target=worker_fn, args=(wid, gid, chunk, args.model, output_dir))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    merge_worker_outputs(output_dir, n_workers)

    with open(output_dir / "manifest.json", "w") as f:
        json.dump({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model":          args.model,
            "attack_type":    args.attack_type,
            "n_behaviors":    len(test_cases),
            "n_pairs":        len(pairs),
            "layers_saved":   LAYERS_TO_SAVE,
            "test_mode":      args.test,
        }, f, indent=2)

    logger.info(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
