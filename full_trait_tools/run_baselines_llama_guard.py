#!/usr/bin/env python3
"""
run_baselines_llama_guard.py

Runs Llama Guard 3 on all attack families and reports balanced direct AUC.
Evaluates both input-only and input+output modes.

No training needed — Llama Guard outputs calibrated P(unsafe) directly.
Balanced AUC: undersample majority class to 50/50 on each test set.

Attack families: HarmBench (held-out), GCG, PAIR, PAP, GPTFuzz, PEZ

Usage:
  uv run python full_trait_tools/run_baselines_llama_guard.py
  uv run python full_trait_tools/run_baselines_llama_guard.py --test
"""

import argparse
import json
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GUARD_MODEL            = "meta-llama/Llama-Guard-3-8B"
JAILBREAKS_URL         = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench"
                          "/main/baselines/human_jailbreaks/jailbreaks.py")
SKIP_JAILBREAK_INDICES = {16, 44}
HARMBENCH_ROOT         = "/dlabscratch1/bazina/HarmBench"
RANDOM_SEED            = 42

DATASETS = {
    "HarmBench": {
        "responses":   "full_trait_output/harmbench_activations/classified_responses.jsonl",
        "metadata":    "full_trait_output/harmbench_activations/pairs_metadata.jsonl",
        "test_cases":  None,
        "attack_type": "human_jailbreak",
    },
    "GCG": {
        "responses":  "full_trait_output/gcg_activations/responses.jsonl",
        "test_cases": f"{HARMBENCH_ROOT}/results/GCG/llama3_1_8b/test_cases/test_cases.json",
    },
    "PAIR": {
        "responses":  "full_trait_output/pair_activations/responses.jsonl",
        "test_cases": f"{HARMBENCH_ROOT}/results/PAIR/llama3_1_8b/test_cases/test_cases.json",
    },
    "PAP": {
        "responses":  "full_trait_output/pap_activations/responses.jsonl",
        "test_cases": f"{HARMBENCH_ROOT}/results/PAP/llama3_1_8b/test_cases/test_cases.json",
    },
    "GPTFuzz": {
        "responses":  "full_trait_output/gptfuzz_activations/responses.jsonl",
        "test_cases": f"{HARMBENCH_ROOT}/results/GPTFuzz/llama3_1_8b/test_cases/test_cases.json",
    },
    "PEZ": {
        "responses":  "full_trait_output/pez_activations/responses.jsonl",
        "test_cases": f"{HARMBENCH_ROOT}/results/PEZ/llama3_1_8b/test_cases/test_cases.json",
    },
}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fetch_jailbreaks():
    with urllib.request.urlopen(JAILBREAKS_URL) as resp:
        source = resp.read().decode("utf-8")
    ns = {}
    exec(source, ns)
    raw = ns["JAILBREAKS"]
    valid = {}
    for i, jb in enumerate(raw):
        if i in SKIP_JAILBREAK_INDICES or "{0}" not in jb:
            continue
        valid[i] = jb
    return valid


def build_prompt_map_harmbench(metadata_path, responses_path, jailbreak_templates):
    meta = load_jsonl(metadata_path)
    resp = load_jsonl(responses_path)
    btext = {r["pair_id"]: r["behavior_text"] for r in resp}
    pm = {}
    for row in meta:
        pid = row["pair_id"]
        jb_idx = row["jailbreak_idx"]
        bt = row.get("behavior_text") or btext.get(pid, "")
        if jb_idx == -1:
            pm[pid] = bt
        elif jb_idx in jailbreak_templates:
            try:
                pm[pid] = jailbreak_templates[jb_idx].format(bt)
            except Exception:
                pm[pid] = bt
        else:
            pm[pid] = bt
    return pm


def build_prompt_map_generic(responses, test_cases_path):
    tc = json.load(open(test_cases_path))
    pm = {}
    # Track index per behavior for PAP (which has 5 prompts per behavior)
    behavior_counts = {}
    for row in responses:
        pid = row["pair_id"]
        bid = row["behavior_id"]
        attack_type = row.get("attack_type", "")

        if bid not in behavior_counts:
            behavior_counts[bid] = 0

        prompts = tc.get(bid, [])
        # Flatten nested lists (PAP format)
        flat = []
        for p in prompts:
            if isinstance(p, list):
                flat.extend(p)
            elif isinstance(p, str):
                flat.append(p)

        if flat:
            idx = behavior_counts[bid] % len(flat)
            pm[pid] = flat[idx]
        else:
            pm[pid] = row.get("behavior_text", "")

        behavior_counts[bid] += 1
    return pm


def get_token_ids(tokenizer):
    safe_id   = tokenizer.encode("safe",   add_special_tokens=False)[-1]
    unsafe_id = tokenizer.encode("unsafe", add_special_tokens=False)[-1]
    logger.info(f"  safe={safe_id} ('{tokenizer.decode([safe_id])}'), unsafe={unsafe_id} ('{tokenizer.decode([unsafe_id])}')")
    return safe_id, unsafe_id


def guard_score(model, tokenizer, prompt, device, safe_id, unsafe_id, response=None):
    try:
        prompt = prompt.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        if not prompt.strip():
            return None
        if response:
            conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        else:
            conv = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(formatted, return_tensors="pt", truncation=True,
                        max_length=2048, add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits[0, -1, :]
        probs = torch.softmax(torch.stack([logits[safe_id].float(), logits[unsafe_id].float()]), dim=0)
        return float(probs[1].item())
    except Exception as e:
        logger.warning(f"Guard error: {e}")
        return None


def balanced_auc(scores, y):
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n = min(len(idx_pos), len(idx_neg))
    if n == 0 or len(set(y)) < 2:
        return float("nan")
    rng = np.random.RandomState(RANDOM_SEED)
    idx = np.concatenate([rng.choice(idx_pos, n, replace=False),
                          rng.choice(idx_neg, n, replace=False)])
    s, yb = np.array(scores)[idx], y[idx]
    auc = roc_auc_score(yb, s)
    return max(auc, 1 - auc)


def score_dataset(name, rows, prompt_map, model, tokenizer, device, safe_id, unsafe_id, test_mode):
    scores_input, scores_full, y_list = [], [], []
    filter_type = DATASETS[name].get("attack_type")
    if filter_type:
        rows = [r for r in rows if r.get("attack_type") == filter_type]
    if test_mode:
        rows = rows[:20]

    for i, row in enumerate(rows):
        if i % 100 == 0:
            logger.info(f"  {name} {i}/{len(rows)}")
        pid = row["pair_id"]
        jb  = row.get("jailbroken")
        if jb is None:
            continue
        prompt = prompt_map.get(pid, "")
        if not prompt:
            continue
        response = row.get("response_text", "")

        s_in  = guard_score(model, tokenizer, prompt, device, safe_id, unsafe_id)
        s_out = guard_score(model, tokenizer, prompt, device, safe_id, unsafe_id, response)

        if s_in is None or s_out is None:
            continue

        scores_input.append(s_in)
        scores_full.append(s_out)
        y_list.append(1 if jb else 0)

    y = np.array(y_list)
    auc_in  = balanced_auc(scores_input, y)
    auc_out = balanced_auc(scores_full,  y)
    logger.info(f"  {name}: {len(y)} pairs, {y.sum():.0f} jb | input={auc_in:.4f}, full={auc_out:.4f}")
    return auc_in, auc_out, len(y), int(y.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guard_model", default=GUARD_MODEL)
    parser.add_argument("--output_dir",  default="full_trait_output/baselines_all_attacks")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--test",        action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading {args.guard_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.guard_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.guard_model, dtype=torch.bfloat16, device_map={"": device})
    model.eval()
    safe_id, unsafe_id = get_token_ids(tokenizer)

    logger.info("Fetching jailbreak templates...")
    jailbreak_templates = fetch_jailbreaks()

    results = {}
    for name, cfg in DATASETS.items():
        if not Path(cfg["responses"]).exists():
            logger.warning(f"Skipping {name} — responses not found")
            continue

        logger.info(f"\n=== {name} ===")
        rows = load_jsonl(cfg["responses"])

        if name == "HarmBench":
            prompt_map = build_prompt_map_harmbench(
                cfg["metadata"], cfg["responses"], jailbreak_templates)
        else:
            prompt_map = build_prompt_map_generic(rows, cfg["test_cases"])

        auc_in, auc_out, n, n_jb = score_dataset(
            name, rows, prompt_map, model, tokenizer, device, safe_id, unsafe_id, args.test)

        results[name] = {
            "n": n, "n_jb": n_jb,
            "input_only_auc":   auc_in,
            "input_output_auc": auc_out,
        }

    # Summary
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  LLAMA GUARD 3 — ALL ATTACK FAMILIES  (balanced AUC, best direction)")
    print(sep)
    print(f"  {'Family':12s}  {'N':>6}  {'JB%':>6}  {'Input-only':>12}  {'Input+Output':>13}")
    print("  " + "─" * 55)
    for name, r in results.items():
        jb_rate = r["n_jb"] / r["n"] if r["n"] > 0 else 0
        print(f"  {name:12s}  {r['n']:>6}  {jb_rate:>6.1%}  {r['input_only_auc']:>12.4f}  {r['input_output_auc']:>13.4f}")
    print(sep)

    out_path = output_dir / "llama_guard_all_attacks.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
