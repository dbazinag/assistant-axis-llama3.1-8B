#!/usr/bin/env python3
"""
perplexity_classifier.py

Baseline jailbreak detector using Llama-3.1-8B perplexity of the input prompt.
Trains a logistic regression on perplexity scores from the HarmBench training set,
then tests on HarmBench held-out, GCG, and PAIR — same regime as fast_transfer_classifier.py.

Perplexity = exp(mean NLL) of the formatted prompt under Llama-3.1-8B.
High perplexity → prompt looks unusual to the model (e.g. GCG suffix).
Low perplexity → fluent natural language (PAIR, human jailbreaks).

HarmBench formatted prompts are reconstructed from:
  - pairs_metadata.jsonl (has jailbreak_idx per pair)
  - jailbreaks.py fetched from HarmBench GitHub (same seed/sample as collection)

GCG/PAIR formatted prompts come from their test_cases.json files.

Usage:
  uv run python full_trait_tools/perplexity_classifier.py
  uv run python full_trait_tools/perplexity_classifier.py --test  # 3 behaviors only
"""

import argparse
import json
import logging
import random
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Constants (must match collect_harmbench_activations.py) ───────────────────
JAILBREAKS_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
    "/main/baselines/human_jailbreaks/jailbreaks.py"
)
SKIP_JAILBREAK_INDICES = {16, 44}
N_JAILBREAK_SAMPLES    = 20
HARMBENCH_SEED         = 42
TRAIN_FRAC             = 0.7
N_SEEDS                = 50
RANDOM_SEED            = 42
MODEL_NAME             = "meta-llama/Llama-3.1-8B-Instruct"


# ── Jailbreak template reconstruction ─────────────────────────────────────────

def fetch_jailbreaks() -> Dict[int, str]:
    """Fetch and parse jailbreaks.py, return {original_idx: template}."""
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


def get_sampled_jailbreak_indices(all_indices: List[int]) -> List[int]:
    """Reproduce the same random sample used during collection."""
    rng = random.Random(HARMBENCH_SEED)
    return sorted(rng.sample(all_indices, min(N_JAILBREAK_SAMPLES, len(all_indices))))


# ── Data loading ───────────────────────────────────────────────────────────────

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
    """
    Reconstruct formatted_prompt for each HarmBench pair_id.
    Returns {pair_id: formatted_prompt}.
    """
    # Load metadata to get jailbreak_idx and behavior_text per pair
    meta_rows = load_jsonl(metadata_path)
    classified_rows = load_jsonl(classified_path)

    # Build behavior_text lookup from classified responses
    behavior_lookup = {r["pair_id"]: r["behavior_text"] for r in classified_rows}

    prompt_map = {}
    for row in meta_rows:
        pid = row["pair_id"]
        jb_idx = row["jailbreak_idx"]
        behavior_text = row.get("behavior_text") or behavior_lookup.get(pid, "")

        if jb_idx == -1:
            # DirectRequest — plain behavior text
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


def build_gcg_prompt_map(
    classified_path: Path,
    test_cases_path: Path,
) -> Dict[int, str]:
    """
    Build {pair_id: formatted_prompt} for GCG.
    GCG prompt = the adversarial suffix prompt from test_cases.json.
    """
    test_cases = json.load(open(test_cases_path))
    rows = load_jsonl(classified_path)

    prompt_map = {}
    for row in rows:
        pid = row["pair_id"]
        bid = row["behavior_id"]
        if bid in test_cases and test_cases[bid]:
            prompt_map[pid] = test_cases[bid][0]
        else:
            # Fallback to behavior text if no test case found
            prompt_map[pid] = row.get("behavior_text", "")
    return prompt_map


def build_pair_prompt_map(
    classified_path: Path,
    test_cases_path: Path,
) -> Dict[int, str]:
    """
    Build {pair_id: formatted_prompt} for PAIR.
    For PAIR jailbreaks: prompt from test_cases.json.
    For plain prompts: behavior_text.
    """
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
            # plain prompt — just behavior text
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
    """
    Compute perplexity of `text` under the model.
    Applies Llama chat template (same as during collection).
    Returns perplexity as a float (lower = more fluent/natural).
    """
    try:
        # Apply chat template — same format as collection
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
        # outputs.loss = mean NLL per token
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
    """
    Compute perplexity for all rows that have a label and a prompt.
    Returns (X [N x 1], y [N], valid_rows).
    """
    X_list, y_list, valid_rows = [], [], []

    for i, row in enumerate(rows):
        if i % 100 == 0:
            logger.info(f"  {desc} {i}/{len(rows)}")

        pid = row["pair_id"]
        jb = row.get("jailbroken")
        if jb is None:
            continue
        prompt = prompt_map.get(pid)
        if not prompt:
            continue

        ppl = compute_perplexity(model, tokenizer, prompt, device)
        if np.isnan(ppl):
            continue

        X_list.append([ppl])
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
    train_beh = set(all_behaviors[:n_beh])
    train_tpl = set(all_templates[:n_tpl])
    test_beh  = set(all_behaviors[n_beh:])
    test_tpl  = set(all_templates[n_tpl:])
    return train_beh, train_tpl, test_beh, test_tpl


def split_by_pool(rows, train_beh, train_tpl, test_beh, test_tpl):
    train_idx = [i for i, r in enumerate(rows)
                 if r["behavior_id"] in train_beh and r["jailbreak_idx"] in train_tpl]
    test_idx  = [i for i, r in enumerate(rows)
                 if r["behavior_id"] in test_beh  and r["jailbreak_idx"] in test_tpl]
    return train_idx, test_idx


def fit_eval(X_tr, y_tr, X_te, y_te):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, C=1.0)
    clf.fit(X_tr_s, y_tr)
    if len(set(y_te)) < 2:
        return float("nan")
    probs = clf.predict_proba(X_te_s)[:, 1]
    return float(roc_auc_score(y_te, probs))


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
    parser.add_argument("--test",     action="store_true", help="First 3 behaviors only")
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
        Path(args.human_metadata_path),
        Path(args.human_classified_path),
        jailbreak_templates,
    )
    gcg_prompt_map  = build_gcg_prompt_map(
        Path(args.gcg_classified_path),
        Path(args.gcg_test_cases_path),
    )
    pair_prompt_map = build_pair_prompt_map(
        Path(args.pair_classified_path),
        Path(args.pair_test_cases_path),
    )

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

    logger.info("HarmBench human jailbreaks...")
    X_human, y_human, human_valid = compute_all_perplexities(
        human_rows, human_prompt_map, model, tokenizer, device, "HarmBench"
    )
    logger.info(f"  {len(y_human)} pairs, {y_human.sum():.0f} jailbroken")

    logger.info("GCG...")
    X_gcg, y_gcg, gcg_valid = compute_all_perplexities(
        gcg_rows, gcg_prompt_map, model, tokenizer, device, "GCG"
    )
    logger.info(f"  {len(y_gcg)} pairs, {y_gcg.sum():.0f} jailbroken")

    logger.info("PAIR...")
    X_pair, y_pair, pair_valid = compute_all_perplexities(
        pair_rows, pair_prompt_map, model, tokenizer, device, "PAIR"
    )
    logger.info(f"  {len(y_pair)} pairs, {y_pair.sum():.0f} jailbroken")

    # Save perplexities for inspection
    ppl_out = {
        "human_ppl_mean": float(X_human.mean()) if len(X_human) else None,
        "gcg_ppl_mean":   float(X_gcg.mean())   if len(X_gcg) else None,
        "pair_ppl_mean":  float(X_pair.mean())  if len(X_pair) else None,
        "human_ppl_jb_mean":  float(X_human[y_human==1].mean()) if y_human.sum() > 0 else None,
        "human_ppl_nojb_mean": float(X_human[y_human==0].mean()) if (y_human==0).sum() > 0 else None,
        "gcg_ppl_jb_mean":    float(X_gcg[y_gcg==1].mean()) if y_gcg.sum() > 0 else None,
        "gcg_ppl_nojb_mean":  float(X_gcg[y_gcg==0].mean()) if (y_gcg==0).sum() > 0 else None,
        "pair_ppl_jb_mean":   float(X_pair[y_pair==1].mean()) if y_pair.sum() > 0 else None,
        "pair_ppl_nojb_mean": float(X_pair[y_pair==0].mean()) if (y_pair==0).sum() > 0 else None,
    }
    logger.info(f"\nPerplexity summary: {json.dumps(ppl_out, indent=2)}")

    # ── Multi-seed classification ──────────────────────────────────────────────
    logger.info(f"\n=== Classification ({args.n_seeds} seeds) ===")

    gcg_aucs, pair_aucs, human_test_aucs = [], [], []

    for seed in range(args.n_seeds):
        train_beh, train_tpl, test_beh, test_tpl = get_pool_split(
            human_valid, TRAIN_FRAC, seed
        )
        train_idx, test_idx = split_by_pool(
            human_valid, train_beh, train_tpl, test_beh, test_tpl
        )
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        X_tr = X_human[train_idx]; y_tr = y_human[train_idx]
        X_te = X_human[test_idx];  y_te = y_human[test_idx]

        gcg_aucs.append(fit_eval(X_tr, y_tr, X_gcg,  y_gcg))
        pair_aucs.append(fit_eval(X_tr, y_tr, X_pair, y_pair))
        human_test_aucs.append(fit_eval(X_tr, y_tr, X_te, y_te))

        if seed % 10 == 0:
            logger.info(f"  Seed {seed} done")

    # ── Summary ────────────────────────────────────────────────────────────────
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  PERPLEXITY BASELINE  |  Llama-3.1-8B  |  {args.n_seeds} seeds")
    print(sep)
    print(f"\n  {'Metric':30s}  {'Mean':>8}  {'Std':>8}")
    print("  " + "─" * 50)
    print(f"  {'→ HarmBench test AUC':30s}  {np.mean(human_test_aucs):>8.4f}  {np.std(human_test_aucs):>8.4f}")
    print(f"  {'→ GCG transfer AUC':30s}  {np.mean(gcg_aucs):>8.4f}  {np.std(gcg_aucs):>8.4f}")
    print(f"  {'→ PAIR transfer AUC':30s}  {np.mean(pair_aucs):>8.4f}  {np.std(pair_aucs):>8.4f}")
    print(f"\n  GCG chance:  {y_gcg.mean():.4f}")
    print(f"  PAIR chance: {y_pair.mean():.4f}")
    print(f"\n  Perplexity means:")
    print(f"    HarmBench jailbroken:     {ppl_out['human_ppl_jb_mean']:.2f}")
    print(f"    HarmBench not jailbroken: {ppl_out['human_ppl_nojb_mean']:.2f}")
    print(f"    GCG jailbroken:           {ppl_out['gcg_ppl_jb_mean']:.2f}")
    print(f"    GCG not jailbroken:       {ppl_out['gcg_ppl_nojb_mean']:.2f}")
    print(f"    PAIR jailbroken:          {ppl_out['pair_ppl_jb_mean']:.2f}")
    print(f"    PAIR not jailbroken:      {ppl_out['pair_ppl_nojb_mean']:.2f}")
    print(sep)

    # Save results
    out = {
        "model": args.model,
        "n_seeds": args.n_seeds,
        "gcg_chance": float(y_gcg.mean()),
        "pair_chance": float(y_pair.mean()),
        "perplexity_stats": ppl_out,
        "results": {
            "human_test": {"mean": float(np.mean(human_test_aucs)), "std": float(np.std(human_test_aucs)), "all": human_test_aucs},
            "transfer_gcg":  {"mean": float(np.mean(gcg_aucs)),  "std": float(np.std(gcg_aucs)),  "all": gcg_aucs},
            "transfer_pair": {"mean": float(np.mean(pair_aucs)), "std": float(np.std(pair_aucs)), "all": pair_aucs},
        }
    }
    out_path = output_dir / "perplexity_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
