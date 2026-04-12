#!/usr/bin/env python3
"""
pc_ablation_study.py

Ablation study over PCA components for jailbreak classification.

For each subset of PCs, trains logistic regression using only those
PC projections as features and reports ROC-AUC. This reveals:
  - Which PCs are individually useful (size-1 subsets)
  - Which PCs are necessary (removing them causes big AUC drop)
  - Which PCs interact (pairs/triples that outperform their individuals)
  - The full model AUC (all PCs) as baseline

Results are printed as a clean table sorted by AUC.

Usage:
  uv run full_trait_tools/pc_ablation_study.py
  uv run full_trait_tools/pc_ablation_study.py --n_pca 16:4 28:4
"""

import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

RANDOM_SEED      = 42
TRAIN_FRAC       = 0.7
MIN_SUCCESS_RATE = 0.20
MAX_SUCCESS_RATE = 0.80
N_SEEDS          = 8
DEFAULT_N_PCA    = {16: 4, 28: 4}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_classified(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_activations(path: Path) -> Dict[int, Dict[str, torch.Tensor]]:
    return torch.load(path, map_location="cpu", weights_only=False)


# ── Filtering + splitting ──────────────────────────────────────────────────────

def filter_human_jailbreak(rows):
    return [r for r in rows if r.get("attack_type") == "human_jailbreak"]


def compute_behavior_success_rates(rows):
    counts: dict = defaultdict(lambda: {"total": 0, "jailbroken": 0})
    for r in rows:
        counts[r["behavior_id"]]["total"] += 1
        if r["jailbroken"]:
            counts[r["behavior_id"]]["jailbroken"] += 1
    return {
        bid: c["jailbroken"] / c["total"]
        for bid, c in counts.items() if c["total"] > 0
    }


def filter_by_variance(rows, success_rates, min_rate, max_rate):
    kept = {bid for bid, r in success_rates.items() if min_rate <= r <= max_rate}
    return [r for r in rows if r["behavior_id"] in kept]


def split_pools(rows, train_frac, seed):
    rng = random.Random(seed)
    all_behaviors = sorted({r["behavior_id"]   for r in rows})
    all_templates = sorted({r["jailbreak_idx"]  for r in rows})
    rng.shuffle(all_behaviors)
    rng.shuffle(all_templates)
    n_train_beh = max(1, int(len(all_behaviors) * train_frac))
    n_train_tpl = max(1, int(len(all_templates) * train_frac))
    train_beh = set(all_behaviors[:n_train_beh])
    test_beh  = set(all_behaviors[n_train_beh:])
    train_tpl = set(all_templates[:n_train_tpl])
    test_tpl  = set(all_templates[n_train_tpl:])
    return train_beh, test_beh, train_tpl, test_tpl


def get_activations(rows, activations, layer, beh_pool=None, tpl_pool=None):
    layer_key = str(layer)
    X_list, y_list = [], []
    for row in rows:
        if beh_pool is not None and row["behavior_id"]   not in beh_pool: continue
        if tpl_pool is not None and row["jailbreak_idx"] not in tpl_pool: continue
        pid = row["pair_id"]
        if pid not in activations or layer_key not in activations[pid]: continue
        X_list.append(activations[pid][layer_key].float().numpy())
        y_list.append(int(row["jailbroken"]))
    if not X_list:
        return np.array([]), np.array([])
    return np.stack(X_list), np.array(y_list)


# ── Ablation ───────────────────────────────────────────────────────────────────

def run_ablation(
    rows_filtered: List[dict],
    activations: Dict,
    layer: int,
    pca: PCA,
    scaler: StandardScaler,
    n_pca: int,
    n_seeds: int,
    train_frac: float,
) -> List[dict]:
    """
    For every non-empty subset of {0, ..., n_pca-1}, train logistic regression
    using only those PC projections as features. Returns list of results.
    """
    # Generate all non-empty subsets
    all_subsets = []
    for size in range(1, n_pca + 1):
        for combo in itertools.combinations(range(n_pca), size):
            all_subsets.append(combo)

    print(f"  Testing {len(all_subsets)} subsets across {n_seeds} seeds...")

    # Collect AUCs per subset across seeds
    subset_aucs: Dict[tuple, List[float]] = {s: [] for s in all_subsets}

    for seed in range(n_seeds):
        train_beh, test_beh, train_tpl, test_tpl = split_pools(
            rows_filtered, train_frac, seed
        )
        X_train_raw, y_train = get_activations(
            rows_filtered, activations, layer, train_beh, train_tpl
        )
        X_test_raw, y_test = get_activations(
            rows_filtered, activations, layer, test_beh, test_tpl
        )
        if len(X_train_raw) == 0 or len(X_test_raw) == 0:
            continue

        # Project all data into full PCA space once
        X_train_pca = pca.transform(scaler.transform(X_train_raw))
        X_test_pca  = pca.transform(scaler.transform(X_test_raw))

        for subset in all_subsets:
            X_tr = X_train_pca[:, list(subset)]
            X_te = X_test_pca[:,  list(subset)]

            clf = LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=500,
                random_state=seed, class_weight="balanced",
            )
            clf.fit(X_tr, y_train)
            try:
                auc = float(roc_auc_score(y_test, clf.predict_proba(X_te)[:, 1]))
            except ValueError:
                auc = float("nan")
            subset_aucs[subset].append(auc)

    # Aggregate results
    results = []
    for subset, aucs in subset_aucs.items():
        valid = [a for a in aucs if not np.isnan(a)]
        results.append({
            "subset":      subset,
            "subset_str":  "+".join([f"PC{i+1}" for i in subset]),
            "n_pcs":       len(subset),
            "mean_auc":    float(np.mean(valid)) if valid else float("nan"),
            "std_auc":     float(np.std(valid))  if valid else float("nan"),
        })

    results.sort(key=lambda x: x["mean_auc"], reverse=True)
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    return results


# ── Printing ───────────────────────────────────────────────────────────────────

def print_results(layer: int, results: List[dict], n_pca: int) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  LAYER {layer} — PC ABLATION STUDY  ({n_pca} PCs, all subsets)")
    print(sep)

    # Full model baseline
    full = next(r for r in results if r["n_pcs"] == n_pca)
    print(f"\n  Full model (all {n_pca} PCs): AUC = {full['mean_auc']:.4f} ± {full['std_auc']:.4f}")

    # Print by subset size
    for size in range(1, n_pca + 1):
        size_results = [r for r in results if r["n_pcs"] == size]
        size_results.sort(key=lambda x: x["mean_auc"], reverse=True)

        label = {1: "Individual PCs", 2: "Pairs", 3: "Triples", 4: "All 4"}.get(size, f"Size {size}")
        print(f"\n  ── {label} ──")
        print(f"  {'Rank':>4}  {'Subset':20s}  {'AUC':>8}  {'±':>6}  {'vs full':>8}")
        print("  " + "─" * 55)

        for r in size_results:
            delta = r["mean_auc"] - full["mean_auc"]
            delta_str = f"{delta:+.4f}"
            print(f"  {r['rank']:>4}  {r['subset_str']:20s}  "
                  f"{r['mean_auc']:>8.4f}  {r['std_auc']:>6.4f}  {delta_str:>8}")

    # Highlight interactions: pairs that beat both their individual components
    print(f"\n  ── Synergistic pairs (pair AUC > both individual AUCs) ──")
    found_any = False
    individual_aucs = {
        r["subset"][0]: r["mean_auc"]
        for r in results if r["n_pcs"] == 1
    }
    for r in results:
        if r["n_pcs"] != 2:
            continue
        i, j = r["subset"]
        auc_i = individual_aucs.get(i, 0)
        auc_j = individual_aucs.get(j, 0)
        if r["mean_auc"] > max(auc_i, auc_j):
            found_any = True
            print(f"    {r['subset_str']:20s}  AUC={r['mean_auc']:.4f}  "
                  f"(PC{i+1}={auc_i:.4f}, PC{j+1}={auc_j:.4f}, "
                  f"gain={r['mean_auc']-max(auc_i,auc_j):+.4f})")
    if not found_any:
        print("    None found — no pair outperforms its best individual component")

    # Necessity: which PC causes the biggest drop when removed
    print(f"\n  ── Necessity: AUC drop when each PC is removed from full model ──")
    print(f"  {'PC removed':>12}  {'AUC without':>12}  {'Drop':>8}")
    print("  " + "─" * 36)
    for pc_idx in range(n_pca):
        remaining = tuple(i for i in range(n_pca) if i != pc_idx)
        r_without = next((r for r in results if r["subset"] == remaining), None)
        if r_without:
            drop = full["mean_auc"] - r_without["mean_auc"]
            print(f"  {'PC'+str(pc_idx+1)+' removed':>12}  "
                  f"{r_without['mean_auc']:>12.4f}  {drop:>+8.4f}")

    print(sep)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classified_path", type=str,
        default="full_trait_output/harmbench_activations/classified_responses.jsonl",
    )
    parser.add_argument(
        "--activations_path", type=str,
        default="full_trait_output/harmbench_activations/activations.pt",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="full_trait_output/harmbench_logreg",
    )
    parser.add_argument(
        "--n_pca", nargs="+", type=str, default=None,
        help="Per-layer PCA dims as 'layer:n' e.g. '16:4 28:4'",
    )
    parser.add_argument("--n_seeds",          type=int,   default=N_SEEDS)
    parser.add_argument("--min_success_rate", type=float, default=MIN_SUCCESS_RATE)
    parser.add_argument("--max_success_rate", type=float, default=MAX_SUCCESS_RATE)
    parser.add_argument("--train_frac",       type=float, default=TRAIN_FRAC)
    parser.add_argument("--layers", nargs="+", type=int,  default=[16, 28])
    args = parser.parse_args()

    n_pca_map = dict(DEFAULT_N_PCA)
    if args.n_pca:
        for item in args.n_pca:
            layer_str, n_str = item.split(":")
            n_pca_map[int(layer_str)] = int(n_str)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    rows        = load_classified(Path(args.classified_path))
    activations = load_activations(Path(args.activations_path))
    rows        = filter_human_jailbreak(rows)
    success_rates = compute_behavior_success_rates(rows)
    rows_filtered = filter_by_variance(
        rows, success_rates,
        min_rate=args.min_success_rate,
        max_rate=args.max_success_rate,
    )
    n_jb = sum(r["jailbroken"] for r in rows_filtered)
    print(f"  {len(rows_filtered)} pairs ({n_jb} jailbroken, {len(rows_filtered)-n_jb} not)")

    all_results = {}

    for layer in args.layers:
        n_pca = n_pca_map.get(layer, 4)
        print(f"\n{'='*65}")
        print(f"  Layer {layer} | n_pca={n_pca}")
        print(f"{'='*65}")

        # Fit PCA
        X_all, _ = get_activations(rows_filtered, activations, layer)
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        pca      = PCA(n_components=n_pca, random_state=RANDOM_SEED)
        pca.fit(X_scaled)
        var = pca.explained_variance_ratio_
        print(f"  Variance explained: {[f'{100*v:.1f}%' for v in var]}")

        results = run_ablation(
            rows_filtered, activations, layer,
            pca, scaler, n_pca,
            n_seeds=args.n_seeds,
            train_frac=args.train_frac,
        )

        print_results(layer, results, n_pca)
        all_results[f"layer_{layer}"] = results

    # Save
    out_path = output_dir / "pc_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {k: v for k, v in all_results.items()},
            f, indent=2,
        )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
