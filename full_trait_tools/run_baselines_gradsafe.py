#!/usr/bin/env python3
"""
run_baselines_gradsafe.py

Runs GradSafe on all attack families using the full HarmBench training split
as reference gradients. Reports balanced direct AUC (50/50 test set).

GradSafe: cosine similarity between gradient of loss on "Sure, here is"
given candidate prompt, vs reference gradients from known jailbreak/benign prompts.

Reference set: full HarmBench strict pool split training rows (seed 0).
This matches the training set size used by our activation classifier.

Attack families: HarmBench (held-out), GCG, PAIR, PAP, GPTFuzz, PEZ

WARNING: Very slow (~7s per prompt per backward pass).
Expected runtime: ~3h reference + ~5h scoring = ~8h total.

Usage:
  uv run python full_trait_tools/run_baselines_gradsafe.py
  uv run python full_trait_tools/run_baselines_gradsafe.py --test
"""

import argparse
import json
import logging
import random
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME             = "meta-llama/Llama-3.1-8B-Instruct"
COMPLIANCE_TARGET      = "Sure, here is"
GRAD_LAYERS            = list(range(28, 32))   # last 4 MLP layers
TRAIN_FRAC             = 0.7
RANDOM_SEED            = 42
JAILBREAKS_URL         = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench"
                          "/main/baselines/human_jailbreaks/jailbreaks.py")
SKIP_JAILBREAK_INDICES = {16, 44}
HARMBENCH_ROOT         = "/dlabscratch1/bazina/HarmBench"

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


# ── Data loading ───────────────────────────────────────────────────────────────

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
    behavior_counts = {}
    for row in responses:
        pid = row["pair_id"]
        bid = row["behavior_id"]
        if bid not in behavior_counts:
            behavior_counts[bid] = 0
        prompts = tc.get(bid, [])
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


# ── Pool split (strict, same as classifier) ────────────────────────────────────

def get_pool_split(rows, train_frac, seed):
    rng = random.Random(seed)
    all_behaviors = sorted({r["behavior_id"]  for r in rows})
    all_templates = sorted({r["jailbreak_idx"] for r in rows})
    rng.shuffle(all_behaviors)
    rng.shuffle(all_templates)
    n_beh = max(1, int(len(all_behaviors) * train_frac))
    n_tpl = max(1, int(len(all_templates) * train_frac))
    train_beh = set(all_behaviors[:n_beh])
    train_tpl = set(all_templates[:n_tpl])
    test_beh  = set(all_behaviors[n_beh:])
    test_tpl  = set(all_templates[n_tpl:])
    return train_beh, train_tpl, test_beh, test_tpl


# ── GradSafe core ──────────────────────────────────────────────────────────────

def get_target_params(model):
    params = []
    for layer_idx in GRAD_LAYERS:
        layer = model.model.layers[layer_idx]
        for _, param in layer.mlp.named_parameters():
            if param.requires_grad:
                params.append(param)
    return params


def compute_gradient_vector(model, tokenizer, prompt_text, target_params, device):
    try:
        prompt_text = prompt_text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        if not prompt_text.strip():
            return None
        conversation = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True)
        full_text = formatted + COMPLIANCE_TARGET

        full_enc  = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        prompt_enc = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        prompt_len = prompt_enc.shape[1]

        if full_enc.shape[1] <= prompt_len:
            return None

        labels = full_enc.clone()
        labels[0, :prompt_len] = -100
        attn = torch.ones_like(full_enc)

        model.zero_grad()
        outputs = model(input_ids=full_enc, attention_mask=attn, labels=labels)
        outputs.loss.backward()

        grad_parts = []
        for param in target_params:
            if param.grad is not None:
                grad_parts.append(param.grad.detach().float().cpu().flatten())

        model.zero_grad()
        torch.cuda.empty_cache()

        if not grad_parts:
            return None

        grad_vec = torch.cat(grad_parts).numpy()
        norm = np.linalg.norm(grad_vec)
        if norm < 1e-10:
            return None
        return grad_vec / norm

    except Exception as e:
        logger.warning(f"Gradient error: {e}")
        model.zero_grad()
        torch.cuda.empty_cache()
        return None


def gradsafe_score(grad, jb_refs, benign_refs):
    jb_sim     = float(np.mean([np.dot(grad, r) for r in jb_refs]))
    benign_sim = float(np.mean([np.dot(grad, r) for r in benign_refs]))
    return jb_sim - benign_sim


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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=MODEL_NAME)
    parser.add_argument("--output_dir", default="full_trait_output/baselines_all_attacks")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--test",       action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": device})
    model.eval()
    for param in model.parameters():
        param.requires_grad_(True)

    target_params = get_target_params(model)
    grad_dim = sum(p.numel() for p in target_params)
    logger.info(f"  Gradient params: {len(target_params)} tensors, {grad_dim:,} dims")

    logger.info("Fetching jailbreak templates...")
    jailbreak_templates = fetch_jailbreaks()

    # ── Build reference set from full HarmBench training split (seed 0) ────────
    logger.info("\n=== Building reference set (full HarmBench training split, seed=0) ===")
    hb_cfg  = DATASETS["HarmBench"]
    hb_rows = load_jsonl(hb_cfg["responses"])
    hb_rows = [r for r in hb_rows if r.get("attack_type") == "human_jailbreak"]
    hb_pm   = build_prompt_map_harmbench(hb_cfg["metadata"], hb_cfg["responses"], jailbreak_templates)

    train_beh, train_tpl, test_beh, test_tpl = get_pool_split(hb_rows, TRAIN_FRAC, seed=0)
    train_rows = [r for r in hb_rows
                  if r["behavior_id"] in train_beh and r["jailbreak_idx"] in train_tpl]
    test_rows  = [r for r in hb_rows
                  if r["behavior_id"] in test_beh  and r["jailbreak_idx"] in test_tpl]

    jb_train     = [r for r in train_rows if r.get("jailbroken")]
    benign_train = [r for r in train_rows if not r.get("jailbroken")]
    logger.info(f"  Training split: {len(train_rows)} rows ({len(jb_train)} jb, {len(benign_train)} benign)")

    if args.test:
        jb_train     = jb_train[:5]
        benign_train = benign_train[:5]

    logger.info(f"  Computing {len(jb_train)} jailbreak reference gradients...")
    jb_refs = []
    for i, row in enumerate(jb_train):
        if i % 50 == 0:
            logger.info(f"    JB ref {i}/{len(jb_train)}")
        prompt = hb_pm.get(row["pair_id"], "")
        if not prompt:
            continue
        g = compute_gradient_vector(model, tokenizer, prompt, target_params, device)
        if g is not None:
            jb_refs.append(g)

    logger.info(f"  Computing {len(benign_train)} benign reference gradients...")
    benign_refs = []
    for i, row in enumerate(benign_train):
        if i % 50 == 0:
            logger.info(f"    Benign ref {i}/{len(benign_train)}")
        prompt = hb_pm.get(row["pair_id"], "")
        if not prompt:
            continue
        g = compute_gradient_vector(model, tokenizer, prompt, target_params, device)
        if g is not None:
            benign_refs.append(g)

    logger.info(f"  Got {len(jb_refs)} jb refs, {len(benign_refs)} benign refs")
    if not jb_refs or not benign_refs:
        logger.error("No reference gradients computed — exiting")
        return

    # Save references for potential reuse
    np.save(output_dir / "gradsafe_jb_refs.npy",     np.stack(jb_refs))
    np.save(output_dir / "gradsafe_benign_refs.npy",  np.stack(benign_refs))
    logger.info("  Reference gradients saved.")

    # ── Score all datasets ─────────────────────────────────────────────────────
    results = {}

    def score_rows(name, rows, prompt_map, filter_type=None):
        if filter_type:
            rows = [r for r in rows if r.get("attack_type") == filter_type]
        # For HarmBench use only held-out test split
        if name == "HarmBench":
            rows = [r for r in rows if r["behavior_id"] in test_beh and r["jailbreak_idx"] in test_tpl]
        if args.test:
            rows = rows[:10]

        scores_list, y_list = [], []
        for i, row in enumerate(rows):
            if i % 50 == 0:
                logger.info(f"  {name} {i}/{len(rows)}")
            pid = row["pair_id"]
            jb  = row.get("jailbroken")
            if jb is None:
                continue
            prompt = prompt_map.get(pid, "")
            if not prompt:
                continue
            grad = compute_gradient_vector(model, tokenizer, prompt, target_params, device)
            if grad is None:
                continue
            scores_list.append(gradsafe_score(grad, jb_refs, benign_refs))
            y_list.append(1 if jb else 0)

        y = np.array(y_list)
        auc = balanced_auc(scores_list, y)
        logger.info(f"  {name}: {len(y)} pairs, {y.sum():.0f} jb, AUC={auc:.4f}")
        return auc, len(y), int(y.sum())

    for name, cfg in DATASETS.items():
        if not Path(cfg["responses"]).exists():
            logger.warning(f"Skipping {name} — not found")
            continue
        logger.info(f"\n=== Scoring {name} ===")
        rows = load_jsonl(cfg["responses"])
        if name == "HarmBench":
            pm = hb_pm
            filter_type = "human_jailbreak"
        else:
            pm = build_prompt_map_generic(rows, cfg["test_cases"])
            filter_type = None

        auc, n, n_jb = score_rows(name, rows, pm, filter_type)
        results[name] = {"n": n, "n_jb": n_jb, "auc": auc}

    # ── Summary ────────────────────────────────────────────────────────────────
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  GRADSAFE — ALL ATTACK FAMILIES  (balanced AUC, best direction)")
    print(f"  Reference: {len(jb_refs)} jb + {len(benign_refs)} benign (full HarmBench train split)")
    print(sep)
    print(f"  {'Family':12s}  {'N':>6}  {'JB%':>6}  {'AUC':>8}")
    print("  " + "─" * 38)
    for name, r in results.items():
        jb_rate = r["n_jb"] / r["n"] if r["n"] > 0 else 0
        print(f"  {name:12s}  {r['n']:>6}  {jb_rate:>6.1%}  {r['auc']:>8.4f}")
    print(sep)

    out_path = output_dir / "gradsafe_all_attacks.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_jb_refs": len(jb_refs),
            "n_benign_refs": len(benign_refs),
            "results": results,
        }, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
