#!/usr/bin/env python3
"""
run_baselines_gradsafe.py

GradSafe implementation faithful to Xie et al. (ACL 2024).

Key differences from naive gradient-norm approach:
  - Reference gradient = average gradient over a small calibration jailbreak set
  - Critical slice selection = row/column unsafe-safe cosine gap > 1
  - Scoring = mean candidate/reference cosine over those critical row/column slices
  - Gradients kept as 2D matrices, NOT flattened — much more memory efficient
  - Calibration set: ~10 jailbreak + ~10 benign prompts (paper uses 2+2)

Splits:
  - Calibration (first 20 from train): find reference gradient + critical params
  - Test (held-out pool split): HarmBench evaluation
  - Transfer: GCG, PAIR, PAP, GPTFuzz, PEZ

Balanced AUC (50/50, best direction) throughout.

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

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME             = "meta-llama/Llama-3.1-8B-Instruct"
COMPLIANCE_TARGET      = "Sure"
TRAIN_FRAC             = 0.7
RANDOM_SEED            = 42
N_CALIB                = 20   # jailbreak + benign prompts each for calibration
CRITICAL_GAP_THRESHOLD = 1.0  # official GradSafe code uses row/col gap > 1
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
    meta  = load_jsonl(metadata_path)
    resp  = load_jsonl(responses_path)
    btext = {r["pair_id"]: r["behavior_text"] for r in resp}
    pm = {}
    for row in meta:
        pid    = row["pair_id"]
        jb_idx = row["jailbreak_idx"]
        bt     = row.get("behavior_text") or btext.get(pid, "")
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
    all_beh = sorted({r["behavior_id"]  for r in rows})
    all_tpl = sorted({r["jailbreak_idx"] for r in rows})
    rng.shuffle(all_beh); rng.shuffle(all_tpl)
    n_beh = max(1, int(len(all_beh) * train_frac))
    n_tpl = max(1, int(len(all_tpl) * train_frac))
    return (set(all_beh[:n_beh]), set(all_tpl[:n_tpl]),
            set(all_beh[n_beh:]),  set(all_tpl[n_tpl:]))


# ── Gradient computation ───────────────────────────────────────────────────────

def compute_gradient(model, tokenizer, prompt_text, device):
    """
    Compute per-parameter gradient matrices for a prompt paired with
    compliance target "Sure". Returns {param_name: grad_tensor (2D)} or None.
    Gradients kept as 2D matrices — NOT flattened.
    """
    try:
        prompt_text = prompt_text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        if not prompt_text.strip():
            return None

        conversation = [{"role": "user", "content": prompt_text}]
        formatted    = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True)
        full_text    = formatted + COMPLIANCE_TARGET

        full_enc    = tokenizer(full_text, return_tensors="pt",
                                add_special_tokens=False).input_ids.to(device)
        prompt_enc  = tokenizer(formatted,  return_tensors="pt",
                                add_special_tokens=False).input_ids.to(device)
        prompt_len  = prompt_enc.shape[1]

        if full_enc.shape[1] <= prompt_len:
            return None

        labels = full_enc.clone()
        labels[0, :prompt_len] = -100
        attn = torch.ones_like(full_enc)

        model.zero_grad()
        outputs = model(input_ids=full_enc, attention_mask=attn, labels=labels)
        outputs.loss.backward()

        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None and ("mlp" in name or "self" in name):
                # Keep as 2D — same as paper
                g = param.grad.detach().float()
                if g.dim() == 1:
                    g = g.unsqueeze(0)  # ensure 2D
                grads[name] = g.cpu()

        model.zero_grad()
        torch.cuda.empty_cache()
        return grads

    except Exception as e:
        logger.warning(f"Gradient error: {e}")
        model.zero_grad()
        torch.cuda.empty_cache()
        return None


def average_gradients(grad_list):
    """Average a list of gradient dicts."""
    avg = {}
    for grads in grad_list:
        for name, g in grads.items():
            if name not in avg:
                avg[name] = g.clone()
            else:
                avg[name] = avg[name] + g.to(avg[name].device)
    for name in avg:
        avg[name] = avg[name] / len(grad_list)
    return avg


def add_to_gradient_sum(grad_sum, grads):
    """Accumulate gradients without retaining every calibration example."""
    for name, g in grads.items():
        if name not in grad_sum:
            grad_sum[name] = g.clone()
        else:
            grad_sum[name].add_(g.to(grad_sum[name].device))


def divide_gradient_sum(grad_sum, count):
    return {name: g.div(count) for name, g in grad_sum.items()}


def cosine_similarity_score(grads_candidate, grads_reference, critical_masks):
    """
    Score a candidate prompt by averaging row/column cosine similarities over
    safety-critical row/column slices, matching the official GradSafe code.
    """
    cos_values = []
    for name, masks in critical_masks.items():
        if name not in grads_candidate or name not in grads_reference:
            continue
        g_cand = grads_candidate[name].to(grads_reference[name].device)
        g_ref  = grads_reference[name]

        row_cos = torch.nan_to_num(F.cosine_similarity(g_cand, g_ref, dim=1))
        col_cos = torch.nan_to_num(F.cosine_similarity(g_cand, g_ref, dim=0))
        row_mask = masks["row"].to(row_cos.device)
        col_mask = masks["col"].to(col_cos.device)

        if row_mask.any():
            cos_values.extend(row_cos[row_mask].cpu().tolist())
        if col_mask.any():
            cos_values.extend(col_cos[col_mask].cpu().tolist())

    if not cos_values:
        return float("nan")
    return float(np.mean(cos_values))


# ── AUC ────────────────────────────────────────────────────────────────────────

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
    auc   = roc_auc_score(yb, s)
    return max(auc, 1 - auc)


def average_precision(scores, y):
    if len(scores) == 0 or len(set(y)) < 2:
        return float("nan")
    return average_precision_score(y, scores)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=MODEL_NAME)
    parser.add_argument("--output_dir", default="full_trait_output/baselines_all_attacks")
    parser.add_argument("--n_calib",    type=int, default=N_CALIB,
                        help="Number of jailbreak + benign prompts for calibration each")
    parser.add_argument("--critical_gap_threshold", type=float, default=CRITICAL_GAP_THRESHOLD,
                        help="Row/column unsafe-safe cosine gap threshold for safety-critical slices")
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
        args.model, dtype=torch.float16, device_map={"": device})
    model.eval()
    for param in model.parameters():
        param.requires_grad_(True)
    logger.info("Model loaded.")

    logger.info("Fetching jailbreak templates...")
    jailbreak_templates = fetch_jailbreaks()

    # ── Load HarmBench and build splits ───────────────────────────────────────
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

    # Calibration: first n_calib jailbreak + n_calib benign from training split
    calib_jb     = [r for r in train_rows if r.get("jailbroken")][:args.n_calib]
    calib_benign = [r for r in train_rows if not r.get("jailbroken")][:args.n_calib]

    logger.info(f"\n=== Data splits ===")
    logger.info(f"  Calibration: {len(calib_jb)} jb + {len(calib_benign)} benign")
    logger.info(f"  Test: {len(test_rows)} rows")

    if args.test:
        calib_jb     = calib_jb[:3]
        calib_benign = calib_benign[:3]

    # ── Step 1: Compute reference gradient (avg over calibration jailbreaks) ──
    logger.info(f"\n=== Step 1: Reference gradient ({len(calib_jb)} jb prompts) ===")
    jb_grad_sum = {}
    n_jb_grads = 0
    for i, row in enumerate(calib_jb):
        logger.info(f"  JB calib {i+1}/{len(calib_jb)}")
        prompt = hb_pm.get(row["pair_id"], "")
        if not prompt:
            continue
        g = compute_gradient(model, tokenizer, prompt, device)
        if g is not None:
            add_to_gradient_sum(jb_grad_sum, g)
            n_jb_grads += 1
            del g
            torch.cuda.empty_cache()

    if not n_jb_grads:
        logger.error("No calibration gradients computed — exiting")
        return

    reference_grad = divide_gradient_sum(jb_grad_sum, n_jb_grads)
    del jb_grad_sum
    logger.info(f"  Reference gradient: {len(reference_grad)} parameters")

    # ── Step 2: Find critical params via unsafe-safe cosine gap ───────────────
    logger.info(f"\n=== Step 2: Critical parameter selection ({len(calib_benign)} benign) ===")

    # Average row/column cosine of jb with reference
    jb_row_cos = {}
    jb_col_cos = {}
    for i, row in enumerate(calib_jb):
        logger.info(f"  JB gap calib {i+1}/{len(calib_jb)}")
        prompt = hb_pm.get(row["pair_id"], "")
        if not prompt:
            continue
        g = compute_gradient(model, tokenizer, prompt, device)
        if g is None:
            continue
        for name in reference_grad:
            if name not in g:
                continue
            gc = g[name].to(reference_grad[name].device)
            gr = reference_grad[name]
            row_cos = torch.nan_to_num(F.cosine_similarity(gc, gr, dim=1)).cpu()
            col_cos = torch.nan_to_num(F.cosine_similarity(gc, gr, dim=0)).cpu()
            if name not in jb_row_cos:
                jb_row_cos[name] = row_cos
                jb_col_cos[name] = col_cos
            else:
                jb_row_cos[name].add_(row_cos)
                jb_col_cos[name].add_(col_cos)
        del g
        torch.cuda.empty_cache()

    for name in jb_row_cos:
        jb_row_cos[name].div_(n_jb_grads)
        jb_col_cos[name].div_(n_jb_grads)

    # Average row/column cosine of benign with reference
    benign_row_cos = {}
    benign_col_cos = {}
    n_benign_grads = 0
    for i, row in enumerate(calib_benign):
        logger.info(f"  Benign calib {i+1}/{len(calib_benign)}")
        prompt = hb_pm.get(row["pair_id"], "")
        if not prompt:
            continue
        g = compute_gradient(model, tokenizer, prompt, device)
        if g is not None:
            n_benign_grads += 1
            for name in reference_grad:
                if name not in g:
                    continue
                gc = g[name].to(reference_grad[name].device)
                gr = reference_grad[name]
                row_cos = torch.nan_to_num(F.cosine_similarity(gc, gr, dim=1)).cpu()
                col_cos = torch.nan_to_num(F.cosine_similarity(gc, gr, dim=0)).cpu()
                if name not in benign_row_cos:
                    benign_row_cos[name] = row_cos
                    benign_col_cos[name] = col_cos
                else:
                    benign_row_cos[name].add_(row_cos)
                    benign_col_cos[name].add_(col_cos)
            del g
            torch.cuda.empty_cache()

    for name in benign_row_cos:
        if n_benign_grads:
            benign_row_cos[name].div_(n_benign_grads)
            benign_col_cos[name].div_(n_benign_grads)

    # Safety-critical slices: official GradSafe selects row/column entries where
    # unsafe-safe cosine gap exceeds 1.
    critical_masks = {}
    n_critical_rows = 0
    n_critical_cols = 0
    for name in jb_row_cos:
        if name not in benign_row_cos:
            continue
        row_gap = jb_row_cos[name] - benign_row_cos[name]
        col_gap = jb_col_cos[name] - benign_col_cos[name]
        row_mask = row_gap > args.critical_gap_threshold
        col_mask = col_gap > args.critical_gap_threshold
        if row_mask.any() or col_mask.any():
            critical_masks[name] = {"row": row_mask, "col": col_mask}
            n_critical_rows += int(row_mask.sum().item())
            n_critical_cols += int(col_mask.sum().item())

    if not critical_masks:
        logger.error("No safety-critical row/column slices selected — exiting")
        return

    logger.info(f"  Critical params: {len(critical_masks)}/{len(jb_row_cos)}")
    logger.info(f"  Critical slices: {n_critical_rows} rows + {n_critical_cols} cols")
    logger.info(f"  Gap threshold: {args.critical_gap_threshold}")

    # Save calibration results
    with open(output_dir / "gradsafe_calibration.json", "w") as f:
        json.dump({
            "n_jb_calib": n_jb_grads,
            "n_benign_calib": n_benign_grads,
            "critical_gap_threshold": args.critical_gap_threshold,
            "n_critical_params": len(critical_masks),
            "n_critical_rows": n_critical_rows,
            "n_critical_cols": n_critical_cols,
            "critical_params": list(critical_masks),
        }, f, indent=2)

    # ── Step 3: Score all datasets ─────────────────────────────────────────────
    logger.info(f"\n=== Step 3: Scoring all attack families ===")

    def score_dataset(name, rows, prompt_map, filter_type=None):
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
            g = compute_gradient(model, tokenizer, prompt, device)
            if g is None:
                continue
            score = cosine_similarity_score(g, reference_grad, critical_masks)
            if np.isnan(score):
                continue
            scores_list.append(score)
            y_list.append(1 if jb else 0)

        y   = np.array(y_list)
        auc = balanced_auc(scores_list, y)
        ap = average_precision(scores_list, y)
        logger.info(f"  {name}: {len(y)} pairs, {y.sum():.0f} jb, AUC={auc:.4f}, AP={ap:.4f}")
        return auc, ap, len(y), int(y.sum())

    results = {}
    for name, cfg in DATASETS.items():
        if not Path(cfg["responses"]).exists():
            logger.warning(f"Skipping {name} — not found")
            continue
        logger.info(f"\n--- {name} ---")
        rows = load_jsonl(cfg["responses"])
        pm   = hb_pm if name == "HarmBench" else build_prompt_map_generic(rows, cfg["test_cases"])
        ft   = "human_jailbreak" if name == "HarmBench" else None
        auc, ap, n, n_jb = score_dataset(name, rows, pm, ft)
        results[name] = {"n": n, "n_jb": n_jb, "auc": auc, "average_precision": ap}

    # ── Summary ────────────────────────────────────────────────────────────────
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  GRADSAFE (paper implementation) — ALL ATTACK FAMILIES")
    print(f"  Calib: {n_jb_grads} jb + {n_benign_grads} benign | "
          f"{len(critical_masks)} critical params | "
          f"{n_critical_rows + n_critical_cols} critical slices")
    print(f"  Balanced AUC (50/50, best direction) + AP")
    print(sep)
    print(f"  {'Family':12s}  {'N':>6}  {'JB%':>6}  {'AUC':>8}  {'AP':>8}")
    print("  " + "─" * 48)
    for name, r in results.items():
        rate = r["n_jb"] / r["n"] if r["n"] > 0 else 0
        print(f"  {name:12s}  {r['n']:>6}  {rate:>6.1%}  "
              f"{r['auc']:>8.4f}  {r['average_precision']:>8.4f}")
    print(sep)

    with open(output_dir / "gradsafe_all_attacks.json", "w") as f:
        json.dump({
            "n_jb_calib": n_jb_grads,
            "n_benign_calib": n_benign_grads,
            "critical_gap_threshold": args.critical_gap_threshold,
            "n_critical_params": len(critical_masks),
            "n_critical_rows": n_critical_rows,
            "n_critical_cols": n_critical_cols,
            "results": results,
        }, f, indent=2)
    logger.info(f"Saved to {output_dir}/gradsafe_all_attacks.json")


if __name__ == "__main__":
    main()
