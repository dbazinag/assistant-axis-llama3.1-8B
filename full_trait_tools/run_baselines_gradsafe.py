#!/usr/bin/env python3
"""
run_baselines_gradsafe.py

Full GradSafe implementation (Xie et al., ACL 2024) with proper safety-critical
parameter selection and strict train/calibration/test splits.

Data splits (all from HarmBench strict pool split, seed=0):
  - Calibration set (first 50% of train): used to identify safety-critical parameters
    via gradient norm accumulation across known jailbreak prompts
  - Reference set (second 50% of train): used to compute jailbreak/benign reference
    gradient vectors for cosine similarity scoring
  - Test set (held-out pool split): HarmBench evaluation
  - Transfer sets: GCG, PAIR, PAP, GPTFuzz, PEZ (no overlap with any train data)

Parameter selection: top-k parameters by accumulated squared gradient norm
across calibration jailbreak prompts (top 20% of last 16 MLP layers).

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
TRAIN_FRAC             = 0.7
RANDOM_SEED            = 42
TOP_PARAM_FRAC         = 0.20
CANDIDATE_LAYERS       = list(range(16, 32))
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


# ── Pool split ─────────────────────────────────────────────────────────────────

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

def get_candidate_params(model):
    params = []
    for layer_idx in CANDIDATE_LAYERS:
        layer = model.model.layers[layer_idx]
        for name, param in layer.mlp.named_parameters():
            if param.requires_grad:
                params.append((f"layer{layer_idx}.{name}", param))
    total = sum(p.numel() for _, p in params)
    logger.info(f"  Candidate params: {len(params)} tensors, {total:,} total dims")
    return params


def accumulate_gradient_norms(model, tokenizer, prompts, candidate_params, device):
    norm_accum = {name: 0.0 for name, _ in candidate_params}
    n_ok = 0
    for i, prompt in enumerate(prompts):
        if i % 10 == 0:
            logger.info(f"    Calibration {i}/{len(prompts)}")
        try:
            prompt = prompt.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            if not prompt.strip():
                continue
            conversation = [{"role": "user", "content": prompt}]
            formatted  = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True)
            full_text  = formatted + COMPLIANCE_TARGET
            full_enc   = tokenizer(full_text, return_tensors="pt",
                                   add_special_tokens=False).input_ids.to(device)
            prompt_enc = tokenizer(formatted, return_tensors="pt",
                                   add_special_tokens=False).input_ids.to(device)
            prompt_len = prompt_enc.shape[1]
            if full_enc.shape[1] <= prompt_len:
                continue
            labels = full_enc.clone()
            labels[0, :prompt_len] = -100
            attn = torch.ones_like(full_enc)
            model.zero_grad()
            outputs = model(input_ids=full_enc, attention_mask=attn, labels=labels)
            outputs.loss.backward()
            for name, param in candidate_params:
                if param.grad is not None:
                    norm_accum[name] += float(
                        param.grad.detach().float().pow(2).sum().cpu())
            model.zero_grad()
            torch.cuda.empty_cache()
            n_ok += 1
        except Exception as e:
            logger.warning(f"Calibration error {i}: {e}")
            model.zero_grad()
            torch.cuda.empty_cache()
    logger.info(f"  Accumulated norms over {n_ok} prompts")
    return norm_accum


def select_safety_critical_params(norm_accum, candidate_params, top_frac):
    sorted_names = sorted(norm_accum.keys(), key=lambda n: norm_accum[n], reverse=True)
    n_select = max(1, int(len(sorted_names) * top_frac))
    selected_names = set(sorted_names[:n_select])
    selected = [(name, param) for name, param in candidate_params
                if name in selected_names]
    total_dims = sum(p.numel() for _, p in selected)
    logger.info(f"  Selected {len(selected)} params ({total_dims:,} dims) "
                f"— top {top_frac:.0%} by gradient norm")
    layer_counts = {}
    for name, _ in selected:
        layer = name.split('.')[0]
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    top_layers = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"  Dominant layers: {top_layers}")
    return selected


def compute_gradient_vector(model, tokenizer, prompt_text, target_params, device):
    try:
        prompt_text = prompt_text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        if not prompt_text.strip():
            return None
        conversation = [{"role": "user", "content": prompt_text}]
        formatted  = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True)
        full_text  = formatted + COMPLIANCE_TARGET
        full_enc   = tokenizer(full_text, return_tensors="pt",
                               add_special_tokens=False).input_ids.to(device)
        prompt_enc = tokenizer(formatted, return_tensors="pt",
                               add_special_tokens=False).input_ids.to(device)
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
        for _, param in target_params:
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
    parser.add_argument("--model",          default=MODEL_NAME)
    parser.add_argument("--output_dir",     default="full_trait_output/baselines_all_attacks")
    parser.add_argument("--top_param_frac", type=float, default=TOP_PARAM_FRAC)
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--test",           action="store_true")
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

    logger.info("Fetching jailbreak templates...")
    jailbreak_templates = fetch_jailbreaks()

    # ── Build splits ──────────────────────────────────────────────────────────
    hb_cfg  = DATASETS["HarmBench"]
    hb_rows = load_jsonl(hb_cfg["responses"])
    hb_rows = [r for r in hb_rows if r.get("attack_type") == "human_jailbreak"]
    hb_pm   = build_prompt_map_harmbench(
        hb_cfg["metadata"], hb_cfg["responses"], jailbreak_templates)

    train_beh, train_tpl, test_beh, test_tpl = get_pool_split(hb_rows, TRAIN_FRAC, seed=0)
    train_rows = [r for r in hb_rows
                  if r["behavior_id"] in train_beh and r["jailbreak_idx"] in train_tpl]
    test_rows  = [r for r in hb_rows
                  if r["behavior_id"] in test_beh  and r["jailbreak_idx"] in test_tpl]

    # Split train 50/50 into calibration and reference
    rng_split = random.Random(RANDOM_SEED)
    train_shuffled = list(train_rows)
    rng_split.shuffle(train_shuffled)
    n_calib    = len(train_shuffled) // 2
    calib_rows = train_shuffled[:n_calib]
    ref_rows   = train_shuffled[n_calib:]

    calib_jb   = [r for r in calib_rows if r.get("jailbroken")]
    ref_jb     = [r for r in ref_rows   if r.get("jailbroken")]
    ref_benign = [r for r in ref_rows   if not r.get("jailbroken")]

    logger.info(f"\n=== Data splits ===")
    logger.info(f"  Calibration: {len(calib_rows)} rows ({len(calib_jb)} jb) → param selection")
    logger.info(f"  Reference:   {len(ref_rows)} rows ({len(ref_jb)} jb, {len(ref_benign)} benign)")
    logger.info(f"  Test:        {len(test_rows)} rows")

    if args.test:
        calib_jb   = calib_jb[:5]
        ref_jb     = ref_jb[:5]
        ref_benign = ref_benign[:5]
        test_rows  = test_rows[:10]

    # ── Step 1: Parameter selection ───────────────────────────────────────────
    logger.info(f"\n=== Step 1: Safety-critical parameter selection ===")
    candidate_params = get_candidate_params(model)
    calib_prompts = [hb_pm[r["pair_id"]] for r in calib_jb if r["pair_id"] in hb_pm]
    norm_accum = accumulate_gradient_norms(
        model, tokenizer, calib_prompts, candidate_params, device)
    target_params = select_safety_critical_params(
        norm_accum, candidate_params, args.top_param_frac)

    with open(output_dir / "gradsafe_selected_params.json", "w") as f:
        json.dump({
            "n_candidate":  len(candidate_params),
            "n_selected":   len(target_params),
            "top_frac":     args.top_param_frac,
            "selected_names": [name for name, _ in target_params],
        }, f, indent=2)

    # ── Step 2: Reference gradients ───────────────────────────────────────────
    logger.info(f"\n=== Step 2: Reference gradients ===")
    jb_refs, benign_refs = [], []

    for i, row in enumerate(ref_jb):
        if i % 50 == 0:
            logger.info(f"  JB ref {i}/{len(ref_jb)}")
        g = compute_gradient_vector(model, tokenizer, hb_pm.get(row["pair_id"], ""),
                                    target_params, device)
        if g is not None:
            jb_refs.append(g)

    for i, row in enumerate(ref_benign):
        if i % 50 == 0:
            logger.info(f"  Benign ref {i}/{len(ref_benign)}")
        g = compute_gradient_vector(model, tokenizer, hb_pm.get(row["pair_id"], ""),
                                    target_params, device)
        if g is not None:
            benign_refs.append(g)

    logger.info(f"  Got {len(jb_refs)} jb refs, {len(benign_refs)} benign refs")
    if not jb_refs or not benign_refs:
        logger.error("No reference gradients — exiting")
        return

    np.save(output_dir / "gradsafe_jb_refs.npy",    np.stack(jb_refs))
    np.save(output_dir / "gradsafe_benign_refs.npy", np.stack(benign_refs))

    # ── Step 3: Score all datasets ─────────────────────────────────────────────
    logger.info(f"\n=== Step 3: Scoring ===")

    def score_rows(name, rows, prompt_map, filter_type=None):
        if filter_type:
            rows = [r for r in rows if r.get("attack_type") == filter_type]
        if name == "HarmBench":
            rows = [r for r in rows
                    if r["behavior_id"] in test_beh and r["jailbreak_idx"] in test_tpl]
        if args.test:
            rows = rows[:10]
        scores_list, y_list = [], []
        for i, row in enumerate(rows):
            if i % 50 == 0:
                logger.info(f"  {name} {i}/{len(rows)}")
            jb = row.get("jailbroken")
            if jb is None:
                continue
            prompt = prompt_map.get(row["pair_id"], "")
            if not prompt:
                continue
            grad = compute_gradient_vector(model, tokenizer, prompt, target_params, device)
            if grad is None:
                continue
            scores_list.append(gradsafe_score(grad, jb_refs, benign_refs))
            y_list.append(1 if jb else 0)
        y   = np.array(y_list)
        auc = balanced_auc(scores_list, y)
        logger.info(f"  {name}: {len(y)} pairs, {y.sum():.0f} jb, AUC={auc:.4f}")
        return auc, len(y), int(y.sum())

    results = {}
    for name, cfg in DATASETS.items():
        if not Path(cfg["responses"]).exists():
            logger.warning(f"Skipping {name} — not found")
            continue
        logger.info(f"\n--- {name} ---")
        rows = load_jsonl(cfg["responses"])
        pm   = hb_pm if name == "HarmBench" else build_prompt_map_generic(rows, cfg["test_cases"])
        ft   = "human_jailbreak" if name == "HarmBench" else None
        auc, n, n_jb = score_rows(name, rows, pm, ft)
        results[name] = {"n": n, "n_jb": n_jb, "auc": auc}

    # ── Summary ────────────────────────────────────────────────────────────────
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  GRADSAFE (full) — ALL ATTACK FAMILIES")
    print(f"  Calib: {len(calib_jb)} jb → top {args.top_param_frac:.0%} params")
    print(f"  Refs:  {len(jb_refs)} jb + {len(benign_refs)} benign")
    print(f"  Balanced AUC (50/50, best direction)")
    print(sep)
    print(f"  {'Family':12s}  {'N':>6}  {'JB%':>6}  {'AUC':>8}")
    print("  " + "─" * 38)
    for name, r in results.items():
        rate = r["n_jb"] / r["n"] if r["n"] > 0 else 0
        print(f"  {name:12s}  {r['n']:>6}  {rate:>6.1%}  {r['auc']:>8.4f}")
    print(sep)

    with open(output_dir / "gradsafe_all_attacks.json", "w") as f:
        json.dump({"n_calib_jb": len(calib_jb), "n_ref_jb": len(jb_refs),
                   "n_ref_benign": len(benign_refs),
                   "top_param_frac": args.top_param_frac,
                   "n_params_selected": len(target_params),
                   "results": results}, f, indent=2)
    logger.info(f"Saved to {output_dir}/gradsafe_all_attacks.json")


if __name__ == "__main__":
    main()
