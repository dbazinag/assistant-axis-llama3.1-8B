#!/usr/bin/env python3
"""
run_transfer_classifier.py

Generic wrapper around fast_transfer_classifier logic that accepts any
attack family as the transfer target. Replaces the hardcoded GCG/PAIR
paths with configurable --transfer1 and --transfer2 arguments.

Usage:
  # PAP as primary transfer, PAIR as secondary
  uv run python full_trait_tools/run_transfer_classifier.py \\
    --transfer1_name PAP \\
    --transfer1_classified_path full_trait_output/pap_activations/responses.jsonl \\
    --transfer1_activations_path full_trait_output/pap_activations/activations.pt \\
    --output_dir full_trait_output/transfer_results_pap

  # GPTFuzz only (no second transfer target)
  uv run python full_trait_tools/run_transfer_classifier.py \\
    --transfer1_name GPTFuzz \\
    --transfer1_classified_path full_trait_output/gptfuzz_activations/responses.jsonl \\
    --transfer1_activations_path full_trait_output/gptfuzz_activations/activations.pt \\
    --output_dir full_trait_output/transfer_results_gptfuzz

  # Original GCG + PAIR setup
  uv run python full_trait_tools/run_transfer_classifier.py \\
    --transfer1_name GCG \\
    --transfer1_classified_path full_trait_output/gcg_activations/responses.jsonl \\
    --transfer1_activations_path full_trait_output/gcg_activations/activations.pt \\
    --transfer2_name PAIR \\
    --transfer2_classified_path full_trait_output/pair_activations/responses.jsonl \\
    --transfer2_activations_path full_trait_output/pair_activations/activations.pt \\
    --output_dir full_trait_output/transfer_results_gcg_pair
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────────────
LAYER       = 16
N_PCA       = 4
TOP_K       = 20
TRAIN_FRAC  = 0.7
N_SEEDS     = 50
RANDOM_SEED = 42


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_activations(path: Path) -> dict:
    print(f"  Loading {path} ({path.stat().st_size / 1e6:.1f} MB)...")
    t0 = time.time()
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return data


def load_trait_matrix(vectors_dir: Path, layer: int) -> Tuple[np.ndarray, List[str]]:
    vecs, names = [], []
    for pt_file in sorted(vectors_dir.glob("*.pt")):
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            v = data["vector"][layer].float().numpy()
            norm = np.linalg.norm(v)
            if norm > 1e-8:
                vecs.append(v / norm)
                names.append(pt_file.stem)
        except Exception:
            pass
    matrix = np.stack(vecs)
    print(f"  Loaded {len(names)} trait vectors → matrix {matrix.shape}")
    return matrix, names


def load_hyperplane(path: Path) -> np.ndarray:
    data = torch.load(path, map_location="cpu", weights_only=False)
    v = data["vector"].float().numpy()
    return v / (np.linalg.norm(v) + 1e-12)


# ── Feature extraction ─────────────────────────────────────────────────────────

def build_activation_matrix(rows, activations, layer):
    layer_key = str(layer)
    X_list, y_list, valid_rows = [], [], []
    for row in rows:
        pid = row["pair_id"]
        jb = row.get("jailbroken")
        if jb is None:
            continue
        if pid not in activations or layer_key not in activations[pid]:
            continue
        X_list.append(activations[pid][layer_key].float().numpy())
        y_list.append(1 if jb else 0)
        valid_rows.append(row)
    if not X_list:
        return np.array([]), np.array([]), []
    return np.stack(X_list), np.array(y_list), valid_rows


def project_traits(X_raw: np.ndarray, trait_matrix: np.ndarray) -> np.ndarray:
    return X_raw @ trait_matrix.T


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


def split_by_pool(rows, train_beh, train_tpl, test_beh, test_tpl):
    train_idx = [i for i, r in enumerate(rows)
                 if r["behavior_id"] in train_beh and r["jailbreak_idx"] in train_tpl]
    test_idx  = [i for i, r in enumerate(rows)
                 if r["behavior_id"] in test_beh  and r["jailbreak_idx"] in test_tpl]
    return train_idx, test_idx


# ── Classifier ─────────────────────────────────────────────────────────────────

def undersample_to_balanced(X, y, seed=RANDOM_SEED):
    """Undersample majority class to get 50/50 balance (test set only)."""
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n = min(len(idx_pos), len(idx_neg))
    rng = np.random.RandomState(seed)
    idx_pos_s = rng.choice(idx_pos, n, replace=False)
    idx_neg_s = rng.choice(idx_neg, n, replace=False)
    idx = np.concatenate([idx_pos_s, idx_neg_s])
    return X[idx], y[idx]


def fit_eval(X_tr, y_tr, X_te, y_te, mode, n_pca):
    if len(X_te) == 0 or len(set(y_te)) < 2:
        return float("nan")
    # Undersample test set to 50/50 — training set untouched
    X_te_b, y_te_b = undersample_to_balanced(X_te, y_te)
    if len(set(y_te_b)) < 2:
        return float("nan")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te_b)
    if mode == "pca":
        pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
        X_tr_s = pca.fit_transform(X_tr_s)
        X_te_s = pca.transform(X_te_s)
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, C=1.0)
    clf.fit(X_tr_s, y_tr)
    probs = clf.predict_proba(X_te_s)[:, 1]
    auc = float(roc_auc_score(y_te_b, probs))
    # Best direction — handles inverted signal across attack families
    return max(auc, 1 - auc)


def cv_auc(X, y, mode, n_pca, n_splits=5):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    if mode == "pca":
        pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
        X_s = pca.fit_transform(X_s)
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, C=1.0)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(clf, X_s, y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


def get_features(mode, X_raw, X_traits, X_top):
    if mode in ("raw", "pca"):
        return X_raw
    elif mode == "all_traits":
        return X_traits
    elif mode == "top_traits":
        return X_top


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generic transfer classifier for any attack family")
    # Human jailbreak (training source — always the same)
    parser.add_argument("--human_classified_path", default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--human_activations_path", default="full_trait_output/harmbench_activations/activations.pt")
    # Transfer target 1 (required)
    parser.add_argument("--transfer1_name",              required=True,  help="Column header name e.g. PAP, GPTFuzz, GCG")
    parser.add_argument("--transfer1_classified_path",   required=True)
    parser.add_argument("--transfer1_activations_path",  required=True)
    # Transfer target 2 (optional)
    parser.add_argument("--transfer2_name",              default=None,   help="Optional second transfer target name")
    parser.add_argument("--transfer2_classified_path",   default=None)
    parser.add_argument("--transfer2_activations_path",  default=None)
    # Model / feature config
    parser.add_argument("--trait_vectors_dir", default="full_trait_output/traits40_vectors/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total")
    parser.add_argument("--hyperplane_path",   default="full_trait_output/harmbench_logreg/stable_hyperplane_layer16.pt")
    parser.add_argument("--output_dir",        default="full_trait_output/transfer_results")
    parser.add_argument("--layer",    type=int, default=LAYER)
    parser.add_argument("--n_pca",    type=int, default=N_PCA)
    parser.add_argument("--top_k",    type=int, default=TOP_K)
    parser.add_argument("--n_seeds",  type=int, default=N_SEEDS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n=== Loading data ===")
    human_rows_all = load_jsonl(Path(args.human_classified_path))
    human_rows = [r for r in human_rows_all if r.get("attack_type") == "human_jailbreak"]
    human_acts = load_activations(Path(args.human_activations_path))
    print(f"  Human: {len(human_rows)} rows")

    t1_rows = load_jsonl(Path(args.transfer1_classified_path))
    t1_acts = load_activations(Path(args.transfer1_activations_path))
    print(f"  {args.transfer1_name}: {len(t1_rows)} rows")

    t2_rows, t2_acts = None, None
    if args.transfer2_name:
        t2_rows = load_jsonl(Path(args.transfer2_classified_path))
        t2_acts = load_activations(Path(args.transfer2_activations_path))
        print(f"  {args.transfer2_name}: {len(t2_rows)} rows")

    print("Trait vectors...")
    trait_matrix, trait_names = load_trait_matrix(Path(args.trait_vectors_dir), args.layer)
    w_vec = load_hyperplane(Path(args.hyperplane_path))
    cos_w = np.abs(trait_matrix @ w_vec)
    top_k_idx = np.argsort(cos_w)[::-1][:args.top_k]
    top_trait_names = [trait_names[i] for i in top_k_idx]
    print(f"  Top {args.top_k} traits by |cos(w)|: {top_trait_names[:5]}...")

    # ── Build activation matrices ─────────────────────────────────────────────
    print("\n=== Building activation matrices ===")
    X_human_raw, y_human, human_valid = build_activation_matrix(human_rows, human_acts, args.layer)
    X_t1_raw, y_t1, _ = build_activation_matrix(t1_rows, t1_acts, args.layer)
    print(f"  Human: {len(y_human)} pairs ({y_human.sum():.0f} jb)")
    print(f"  {args.transfer1_name}: {len(y_t1)} pairs ({y_t1.sum():.0f} jb), chance={y_t1.mean():.3f}")

    X_t2_raw, y_t2 = None, None
    if t2_rows is not None:
        X_t2_raw, y_t2, _ = build_activation_matrix(t2_rows, t2_acts, args.layer)
        print(f"  {args.transfer2_name}: {len(y_t2)} pairs ({y_t2.sum():.0f} jb), chance={y_t2.mean():.3f}")

    # ── Trait projections ─────────────────────────────────────────────────────
    print("\n=== Pre-computing trait projections ===")
    t0 = time.time()
    X_human_traits = project_traits(X_human_raw, trait_matrix)
    X_t1_traits    = project_traits(X_t1_raw,    trait_matrix)
    X_human_top    = X_human_traits[:, top_k_idx]
    X_t1_top       = X_t1_traits[:,   top_k_idx]
    if X_t2_raw is not None:
        X_t2_traits = project_traits(X_t2_raw, trait_matrix)
        X_t2_top    = X_t2_traits[:, top_k_idx]
    print(f"  Done in {time.time()-t0:.2f}s")

    modes = ["raw", "pca", "all_traits", "top_traits"]

    # ── Within-human CV ───────────────────────────────────────────────────────
    print("\n=== Within-human cross-validation ===")
    human_cv = {}
    for mode in modes:
        X = get_features(mode, X_human_raw, X_human_traits, X_human_top)
        auc, std = cv_auc(X, y_human, mode, args.n_pca)
        human_cv[mode] = (auc, std)
        print(f"  {mode:12s}: {auc:.4f} ± {std:.4f}")

    # ── Multi-seed transfer ───────────────────────────────────────────────────
    print(f"\n=== Transfer experiment ({args.n_seeds} seeds) ===")
    results = {mode: {"t1": [], "t2": [], "human_test": []} for mode in modes}

    for seed in range(args.n_seeds):
        train_beh, train_tpl, test_beh, test_tpl = get_pool_split(human_valid, TRAIN_FRAC, seed)
        train_idx, test_idx = split_by_pool(human_valid, train_beh, train_tpl, test_beh, test_tpl)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        for mode in modes:
            X_h   = get_features(mode, X_human_raw, X_human_traits, X_human_top)
            X_t1  = get_features(mode, X_t1_raw,    X_t1_traits,    X_t1_top)
            X_tr  = X_h[train_idx]; y_tr = y_human[train_idx]
            X_te  = X_h[test_idx];  y_te = y_human[test_idx]

            results[mode]["t1"].append(fit_eval(X_tr, y_tr, X_t1, y_t1, mode, args.n_pca))
            results[mode]["human_test"].append(fit_eval(X_tr, y_tr, X_te, y_te, mode, args.n_pca))

            if X_t2_raw is not None:
                X_t2 = get_features(mode, X_t2_raw, X_t2_traits, X_t2_top)
                results[mode]["t2"].append(fit_eval(X_tr, y_tr, X_t2, y_t2, mode, args.n_pca))

        if seed % 10 == 0:
            print(f"  Seed {seed} done")

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "=" * 100
    print(f"\n\n{sep}")
    print(f"  TRANSFER SUMMARY  |  Train: HarmBench → Test: {args.transfer1_name}"
          + (f" + {args.transfer2_name}" if args.transfer2_name else "")
          + f"  |  Layer {args.layer}  |  {args.n_seeds} seeds")
    print(sep)

    t1n = args.transfer1_name
    t2n = args.transfer2_name or ""

    header = f"  {'Mode':12s}  {'Human CV':>10}  {'→'+t1n+' mean':>14}  {'→'+t1n+' std':>12}"
    if args.transfer2_name:
        header += f"  {'→'+t2n+' mean':>14}  {'→'+t2n+' std':>12}"
    header += f"  {'Human test':>12}"
    print(f"\n{header}")
    print("  " + "─" * 96)

    summary = {}
    for mode in modes:
        h_cv, h_cv_std = human_cv[mode]
        t1_mean = float(np.nanmean(results[mode]["t1"]))
        t1_std  = float(np.nanstd(results[mode]["t1"]))
        ht_mean = float(np.nanmean(results[mode]["human_test"]))

        row = f"  {mode:12s}  {h_cv:>10.4f}  {t1_mean:>14.4f}  {t1_std:>12.4f}"
        s = {
            "human_cv": {"mean": h_cv, "std": h_cv_std},
            f"transfer_{t1n}": {"mean": t1_mean, "std": t1_std, "all": results[mode]["t1"]},
            "human_test": {"mean": ht_mean, "all": results[mode]["human_test"]},
        }

        if args.transfer2_name and results[mode]["t2"]:
            t2_mean = float(np.nanmean(results[mode]["t2"]))
            t2_std  = float(np.nanstd(results[mode]["t2"]))
            row += f"  {t2_mean:>14.4f}  {t2_std:>12.4f}"
            s[f"transfer_{t2n}"] = {"mean": t2_mean, "std": t2_std, "all": results[mode]["t2"]}

        row += f"  {ht_mean:>12.4f}"
        print(row)
        summary[mode] = s

    print(f"\n  {t1n} chance: {y_t1.mean():.4f}")
    if y_t2 is not None:
        print(f"  {t2n} chance: {y_t2.mean():.4f}")
    print(sep)

    out = {
        "layer": args.layer, "n_pca": args.n_pca, "top_k": args.top_k,
        "n_seeds": args.n_seeds,
        "transfer1_name": t1n, "transfer2_name": t2n,
        f"{t1n}_chance": float(y_t1.mean()),
        "top_trait_names": top_trait_names,
        "modes": summary,
    }
    if y_t2 is not None:
        out[f"{t2n}_chance"] = float(y_t2.mean())

    out_path = output_dir / "transfer_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
