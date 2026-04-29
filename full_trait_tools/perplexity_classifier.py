#!/usr/bin/env python3
"""
perplexity_classifier.py

Baseline jailbreak detector using Llama-3.1-8B perplexity of the input prompt.

KEY DESIGN DECISIONS:
  - Uses log(perplexity) as feature to handle the massive scale difference
    between attack families (GCG: ~1000-1500, HarmBench/PAIR: ~50-100)
  - Reports BOTH:
      (a) Direct within-dataset AUC — no training, just threshold the score.
          This is the honest measure of what perplexity can do per family.
      (b) Cross-family logreg AUC — train on HarmBench, test on GCG/PAIR.
          Expected to be near-chance due to distribution shift. This is itself
          the key finding: perplexity requires per-family calibration.

Same strict pool split regime as fast_transfer_classifier.py.

Usage:
  uv run python full_trait_tools/perplexity_classifier.py
  uv run python full_trait_tools/perplexity_classifier.py --test
"""

import argparse
import json
import logging
import random
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
print("USING LOG PPL...")

# ── Constants ──────────────────────────────────────────────────────────────────
JAILBREAKS_URL         = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
    "/main/baselines/human_jailbreaks/jailbreaks.py"
)
SKIP_JAILBREAK_INDICES = {16, 44}
HARMBENCH_SEED         = 42
TRAIN_FRAC             = 0.7
N_SEEDS                = 50
RANDOM_SEED            = 42
MODEL_NAME             = "meta-llama/Llama-3.1-8B-Instruct"


# ── Jailbreak template reconstruction ─────────────────────────────────────────

def fetch_jailbreaks() -> Dict[int, str]:
    logger.info("Fetching jailbreaks.py from GitHub...")
    with urllib.request.urlopen(JAILBREAKS_URL) as resp:
        source = resp.read().decode("utf-8")
    ns = {}
    exec(source, ns)  # noqa: S102
    raw = ns["JAILBREAKS"]
    valid = {}
    for i, jb in enumerate(raw):
        if i in SKIP_JAILBREAK_INDICES:
            continue
        if "{0}" not in jb:
            continue
        valid[i] = jb
    logger.info(f"  Loaded {len(valid)} valid jailbreak templates")
    return valid


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_harmbench_prompt_map(
    metadata_path: Path,
    classified_path: Path,
    jailbreak_templates: Dict[int, str],
) -> Dict[int, str]:
    meta_rows = load_jsonl(metadata_path)
    classified_rows = load_jsonl(classified_path)
    behavior_lookup = {r["pair_id"]: r["behavior_text"] for r in classified_rows}
    prompt_map = {}
    for row in meta_rows:
        pid = row["pair_id"]
        jb_idx = row["jailbreak_idx"]
        behavior_text = row.get("behavior_text") or behavior_lookup.get(pid, "")
        if jb_idx == -1:
            prompt_map[pid] = behavior_text
        elif jb_idx in jailbreak_templates:
            try:
                prompt_map[pid] = jailbreak_templates[jb_idx].format(behavior_text)
            except Exception:
                prompt_map[pid] = behavior_text
        else:
            prompt_map[pid] = behavior_text
    logger.info(f"  Reconstructed {len(prompt_map)} HarmBench prompts")
    return prompt_map


def build_gcg_prompt_map(classified_path: Path, test_cases_path: Path) -> Dict[int, str]:
    test_cases = json.load(open(test_cases_path))
    rows = load_jsonl(classified_path)
    prompt_map = {}
    for row in rows:
        pid = row["pair_id"]
        bid = row["behavior_id"]
        if bid in test_cases and test_cases[bid]:
            prompt_map[pid] = test_cases[bid][0]
        else:
            prompt_map[pid] = row.get("behavior_text", "")
    return prompt_map


def build_pair_prompt_map(classified_path: Path, test_cases_path: Path) -> Dict[int, str]:
    test_cases = json.load(open(test_cases_path))
    rows = load_jsonl(classified_path)
    prompt_map = {}
    for row in rows:
        pid = row["pair_id"]
        bid = row["behavior_id"]
        attack_type = row.get("attack_type", "")
        if attack_type == "PAIR" and bid in test_cases and test_cases[bid]:
            prompt_map[pid] = test_cases[bid][0]
        else:
            prompt_map[pid] = row.get("behavior_text", "")
    return prompt_map


# ── Perplexity computation ─────────────────────────────────────────────────────

def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
    max_length: int = 2048,
) -> float:
    try:
        text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        if not text.strip():
            return float("nan")
        conversation = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
        return float(torch.exp(outputs.loss).item())
    except Exception as e:
        logger.warning(f"Perplexity error: {e}")
        return float("nan")


def compute_all_perplexities(
    rows: List[dict],
    prompt_map: Dict[int, str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    desc: str = "",
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """Returns (X [N x 2], y [N], valid_rows) where X = [log_ppl, ppl]."""
    X_list, y_list, valid_rows = [], [], []
    for i, row in enumerate(rows):
        if i % 100 == 0:
            logger.info(f"  {desc} {i}/{len(rows)}")
        pid = row["pair_id"]
        jb = row.get("jailbroken")
        if jb is None:
            continue
        prompt = prompt_map.get(pid)
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            continue
        ppl = compute_perplexity(model, tokenizer, prompt, device)
        if np.isnan(ppl) or np.isinf(ppl):
            continue
        log_ppl = float(np.log(ppl + 1e-8))
        X_list.append([log_ppl, ppl])
        y_list.append(1 if jb else 0)
        valid_rows.append(row)
    if not X_list:
        return np.array([]), np.array([]), []
    return np.stack(X_list), np.array(y_list), valid_rows


# ── Pool split (identical to fast_transfer_classifier.py) ─────────────────────

def get_pool_split(rows, train_frac, seed):
    rng = random.Random(seed)
    all_behaviors = sorted({r["behavior_id"]  for r in rows})
    all_templates = sorted({r["jailbreak_idx"] for r in rows})
    rng.shuffle(all_behaviors)
    rng.shuffle(all_templates)
    n_beh = max(1, int(len(all_behaviors) * train_frac))
    n_tpl = max(1, int(len(all_templates) * train_frac))
    return (set(all_behaviors[:n_beh]), set(all_templates[:n_tpl]),
            set(all_behaviors[n_beh:]),  set(all_templates[n_tpl:]))


def split_by_pool(rows, train_beh, train_tpl, test_beh, test_tpl):
    train_idx = [i for i, r in enumerate(rows)
                 if r["behavior_id"] in train_beh and r["jailbreak_idx"] in train_tpl]
    test_idx  = [i for i, r in enumerate(rows)
                 if r["behavior_id"] in test_beh  and r["jailbreak_idx"] in test_tpl]
    return train_idx, test_idx


def fit_eval_logreg(X_tr, y_tr, X_te, y_te):
    """Logistic regression: train on HarmBench, test on target."""
    if len(set(y_te)) < 2:
        return float("nan")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, C=1.0)
    clf.fit(X_tr_s, y_tr)
    probs = clf.predict_proba(X_te_s)[:, 1]
    return float(roc_auc_score(y_te, probs))


def direct_auc(X, y, feature_idx=0):
    """
    Direct AUC: use log_ppl score directly as classifier (no training).
    Higher perplexity → more likely jailbroken.
    We try both directions and return the better one (since within HarmBench
    jailbroken prompts are actually slightly lower perplexity).
    """
    if len(set(y)) < 2 or len(X) == 0:
        return float("nan"), float("nan")
    scores = X[:, feature_idx]
    auc_pos = float(roc_auc_score(y, scores))
    auc_neg = float(roc_auc_score(y, -scores))
    return max(auc_pos, auc_neg), auc_pos  # (best_auc, raw_auc)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_classified_path", default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--human_metadata_path",   default="full_trait_output/harmbench_activations/pairs_metadata.jsonl")
    parser.add_argument("--gcg_classified_path",   default="full_trait_output/gcg_activations/responses.jsonl")
    parser.add_argument("--gcg_test_cases_path",   default="/dlabscratch1/bazina/HarmBench/results/GCG/llama3_1_8b/test_cases/test_cases.json")
    parser.add_argument("--pair_classified_path",  default="full_trait_output/pair_activations/responses.jsonl")
    parser.add_argument("--pair_test_cases_path",  default="/dlabscratch1/bazina/HarmBench/results/PAIR/llama3_1_8b/test_cases/test_cases.json")
    parser.add_argument("--output_dir",            default="full_trait_output/perplexity_baseline")
    parser.add_argument("--model",                 default=MODEL_NAME)
    parser.add_argument("--n_seeds",  type=int,    default=N_SEEDS)
    parser.add_argument("--device",                default="cuda")
    parser.add_argument("--test",     action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.test:
        output_dir = output_dir.parent / (output_dir.name + "_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load model ─────────────────────────────────────────────────────────────
    logger.info(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()
    logger.info("Model loaded.")

    # ── Reconstruct prompts ────────────────────────────────────────────────────
    logger.info("\n=== Reconstructing prompts ===")
    jailbreak_templates = fetch_jailbreaks()
    human_prompt_map = build_harmbench_prompt_map(
        Path(args.human_metadata_path), Path(args.human_classified_path), jailbreak_templates)
    gcg_prompt_map  = build_gcg_prompt_map(Path(args.gcg_classified_path),  Path(args.gcg_test_cases_path))
    pair_prompt_map = build_pair_prompt_map(Path(args.pair_classified_path), Path(args.pair_test_cases_path))

    # ── Load metadata ──────────────────────────────────────────────────────────
    logger.info("\n=== Loading metadata ===")
    human_rows_all = load_jsonl(Path(args.human_classified_path))
    human_rows = [r for r in human_rows_all if r.get("attack_type") == "human_jailbreak"]
    gcg_rows   = load_jsonl(Path(args.gcg_classified_path))
    pair_rows  = load_jsonl(Path(args.pair_classified_path))

    if args.test:
        bids = list({r["behavior_id"] for r in human_rows})[:3]
        human_rows = [r for r in human_rows if r["behavior_id"] in bids]
        gcg_rows   = gcg_rows[:20]
        pair_rows  = pair_rows[:20]

    # ── Compute perplexities ───────────────────────────────────────────────────
    logger.info("\n=== Computing perplexities ===")
    logger.info("HarmBench...")
    X_human, y_human, human_valid = compute_all_perplexities(
        human_rows, human_prompt_map, model, tokenizer, device, "HarmBench")
    logger.info("GCG...")
    X_gcg, y_gcg, _ = compute_all_perplexities(
        gcg_rows, gcg_prompt_map, model, tokenizer, device, "GCG")
    logger.info("PAIR...")
    X_pair, y_pair, _ = compute_all_perplexities(
        pair_rows, pair_prompt_map, model, tokenizer, device, "PAIR")

    # Log perplexity stats
    def ppl_stats(X, y, name):
        if len(X) == 0: return
        ppl = np.exp(X[:, 0])  # X[:,0] is log_ppl
        jb = y == 1
        logger.info(f"  {name}: mean={ppl.mean():.1f}, jb={ppl[jb].mean():.1f}, no_jb={ppl[~jb].mean():.1f}")
    ppl_stats(X_human, y_human, "HarmBench")
    ppl_stats(X_gcg,   y_gcg,   "GCG")
    ppl_stats(X_pair,  y_pair,  "PAIR")

    # ── Direct AUC (no training) ───────────────────────────────────────────────
    logger.info("\n=== Direct AUC (no logreg, threshold log_ppl directly) ===")
    direct_human_best, direct_human_raw = direct_auc(X_human, y_human)
    direct_gcg_best,   direct_gcg_raw   = direct_auc(X_gcg,   y_gcg)
    direct_pair_best,  direct_pair_raw  = direct_auc(X_pair,  y_pair)
    logger.info(f"  HarmBench: {direct_human_best:.4f} (raw: {direct_human_raw:.4f})")
    logger.info(f"  GCG:       {direct_gcg_best:.4f}   (raw: {direct_gcg_raw:.4f})")
    logger.info(f"  PAIR:      {direct_pair_best:.4f}  (raw: {direct_pair_raw:.4f})")

    # ── Cross-family logreg (train HarmBench → test GCG/PAIR) ─────────────────
    logger.info(f"\n=== Cross-family logreg ({args.n_seeds} seeds) ===")
    gcg_aucs, pair_aucs, human_test_aucs = [], [], []

    for seed in range(args.n_seeds):
        train_beh, train_tpl, test_beh, test_tpl = get_pool_split(human_valid, TRAIN_FRAC, seed)
        train_idx, test_idx = split_by_pool(human_valid, train_beh, train_tpl, test_beh, test_tpl)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        # Use only log_ppl column (index 0) — more stable across families than raw ppl
        X_tr = X_human[train_idx, :1]; y_tr = y_human[train_idx]
        X_te = X_human[test_idx,  :1]; y_te = y_human[test_idx]

        gcg_aucs.append(fit_eval_logreg(X_tr, y_tr, X_gcg[:, :1],  y_gcg))
        pair_aucs.append(fit_eval_logreg(X_tr, y_tr, X_pair[:, :1], y_pair))
        human_test_aucs.append(fit_eval_logreg(X_tr, y_tr, X_te, y_te))

        if seed % 10 == 0:
            logger.info(f"  Seed {seed} done")

    # ── Summary ────────────────────────────────────────────────────────────────
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  PERPLEXITY BASELINE  |  Llama-3.1-8B log(ppl)  |  {args.n_seeds} seeds")
    print(sep)
    print(f"\n  DIRECT AUC (within-dataset, no training):")
    print(f"  {'HarmBench':30s}  {direct_human_best:>8.4f}")
    print(f"  {'GCG':30s}  {direct_gcg_best:>8.4f}")
    print(f"  {'PAIR':30s}  {direct_pair_best:>8.4f}")
    print(f"\n  CROSS-FAMILY LOGREG (train HarmBench → test *):")
    print(f"  {'Metric':30s}  {'Mean':>8}  {'Std':>8}")
    print("  " + "─" * 50)
    print(f"  {'→ HarmBench test':30s}  {np.mean(human_test_aucs):>8.4f}  {np.std(human_test_aucs):>8.4f}")
    print(f"  {'→ GCG transfer':30s}  {np.mean(gcg_aucs):>8.4f}  {np.std(gcg_aucs):>8.4f}")
    print(f"  {'→ PAIR transfer':30s}  {np.mean(pair_aucs):>8.4f}  {np.std(pair_aucs):>8.4f}")
    print(f"\n  GCG chance:  {y_gcg.mean():.4f}")
    print(f"  PAIR chance: {y_pair.mean():.4f}")
    print(sep)
    print("\n  NOTE: Cross-family logreg failure is expected and is itself")
    print("  a finding — perplexity requires per-family calibration.")
    print("  Use DIRECT AUC for honest within-family comparison.")

    out = {
        "model": args.model,
        "n_seeds": args.n_seeds,
        "gcg_chance":  float(y_gcg.mean()),
        "pair_chance": float(y_pair.mean()),
        "direct_auc": {
            "human": {"best": direct_human_best, "raw": direct_human_raw},
            "gcg":   {"best": direct_gcg_best,   "raw": direct_gcg_raw},
            "pair":  {"best": direct_pair_best,  "raw": direct_pair_raw},
        },
        "cross_family_logreg": {
            "human_test": {"mean": float(np.mean(human_test_aucs)), "std": float(np.std(human_test_aucs))},
            "transfer_gcg":  {"mean": float(np.mean(gcg_aucs)),  "std": float(np.std(gcg_aucs))},
            "transfer_pair": {"mean": float(np.mean(pair_aucs)), "std": float(np.std(pair_aucs))},
        }
    }
    out_path = output_dir / "perplexity_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
