#!/usr/bin/env python3
"""
classify_jailbreak_per_trait.py

Trains a separate logistic regression classifier for each trait vector
(and the assistant axis) individually, using only that single projection
as the feature. Ranks all traits by ROC-AUC.

This is a univariate analysis — it answers: which individual persona
directions are most predictive of jailbreak success?

Same filtering and train/test split as classify_jailbreak_logreg.py so
results are directly comparable.

Runs two passes: layer 16 and layer 28.

Usage:
  uv run full_trait_tools/classify_jailbreak_per_trait.py
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

RANDOM_SEED         = 42
TRAIN_BEHAVIOR_FRAC = 0.8
MIN_SUCCESS_RATE    = 0.20
MAX_SUCCESS_RATE    = 0.80
LAYERS              = [16, 28]
N_TOP_BOTTOM        = 20  # how many best/worst to print


# ── Vector loading ─────────────────────────────────────────────────────────────

def load_trait_vectors(
    vectors_dir: Path,
    layer: int,
) -> Tuple[List[str], np.ndarray]:
    """
    Returns:
        trait_names: list of trait name strings
        vectors:     np.ndarray [n_traits, 4096]
    """
    pt_files = sorted(vectors_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {vectors_dir}")

    trait_names, vectors = [], []
    for pt_file in pt_files:
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        trait_names.append(pt_file.stem)
        vectors.append(data["vector"][layer].float().numpy())

    return trait_names, np.stack(vectors)  # [n_traits, 4096]


def load_axis_vector(axis_path: Path, layer: int) -> np.ndarray:
    data = torch.load(axis_path, map_location="cpu", weights_only=False)
    return data["axis"][layer].float().numpy()


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


# ── Filtering ──────────────────────────────────────────────────────────────────

def filter_human_jailbreak(rows: List[dict]) -> List[dict]:
    return [r for r in rows if r.get("attack_type") == "human_jailbreak"]


def compute_behavior_success_rates(rows: List[dict]) -> Dict[str, float]:
    counts: dict = defaultdict(lambda: {"total": 0, "jailbroken": 0})
    for r in rows:
        bid = r["behavior_id"]
        counts[bid]["total"] += 1
        if r["jailbroken"]:
            counts[bid]["jailbroken"] += 1
    return {
        bid: c["jailbroken"] / c["total"]
        for bid, c in counts.items()
        if c["total"] > 0
    }


def filter_by_variance(
    rows: List[dict],
    success_rates: Dict[str, float],
    min_rate: float,
    max_rate: float,
) -> Tuple[List[dict], List[str]]:
    kept = {
        bid for bid, rate in success_rates.items()
        if min_rate <= rate <= max_rate
    }
    return [r for r in rows if r["behavior_id"] in kept], sorted(kept)


def split_behaviors_by_success_rate(
    behaviors: List[str],
    success_rates: Dict[str, float],
    train_frac: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    sorted_bids = sorted(behaviors, key=lambda b: success_rates[b])
    n       = len(sorted_bids)
    tertile = n // 3
    buckets = [
        sorted_bids[:tertile],
        sorted_bids[tertile : 2 * tertile],
        sorted_bids[2 * tertile :],
    ]
    train_behaviors, test_behaviors = [], []
    for bucket in buckets:
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * train_frac))
        train_behaviors.extend(shuffled[:n_train])
        test_behaviors.extend(shuffled[n_train:])
    return sorted(train_behaviors), sorted(test_behaviors)


# ── Projection ─────────────────────────────────────────────────────────────────

def build_projection_arrays(
    rows: List[dict],
    activations: Dict[int, Dict[str, torch.Tensor]],
    all_vectors: np.ndarray,  # [n_vectors, 4096]
    layer: int,
    behavior_set: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project all activations in behavior_set onto all vectors at once.

    Returns:
        X: np.ndarray [n_samples, n_vectors] — all projections
        y: np.ndarray [n_samples]            — labels
    """
    behavior_set_s = set(behavior_set)
    layer_key      = str(layer)

    X_list, y_list = [], []

    for row in rows:
        if row["behavior_id"] not in behavior_set_s:
            continue
        pid = row["pair_id"]
        if pid not in activations or layer_key not in activations[pid]:
            continue

        act = activations[pid][layer_key].float().numpy()  # [4096]
        projections = all_vectors @ act                     # [n_vectors]
        X_list.append(projections)
        y_list.append(int(row["jailbroken"]))

    if not X_list:
        return np.array([]), np.array([])

    return np.stack(X_list), np.array(y_list)


# ── Per-trait classifier ───────────────────────────────────────────────────────

def run_single_feature_logreg(
    X_train_col: np.ndarray,  # [n_train, 1]
    y_train: np.ndarray,
    X_test_col: np.ndarray,   # [n_test, 1]
    y_test: np.ndarray,
) -> dict:
    """
    Fit logistic regression with a single feature (one trait projection).
    Returns metrics dict.
    """
    scaler      = StandardScaler()
    X_train_s   = scaler.fit_transform(X_train_col)
    X_test_s    = scaler.transform(X_test_col)

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=500,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    clf.fit(X_train_s, y_train)

    y_pred  = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1]

    # Handle edge case where test set has only one class
    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except ValueError:
        auc = float("nan")

    return {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   auc,
        "coef":      float(clf.coef_[0, 0]),
        "direction": "→ jailbroken" if clf.coef_[0, 0] > 0 else "→ not jailbroken",
    }


# ── Printing ───────────────────────────────────────────────────────────────────

def print_ranking(
    layer: int,
    ranked: List[dict],
    n_top_bottom: int,
) -> None:
    sep = "=" * 75
    print(f"\n{sep}")
    print(f"  LAYER {layer} — PER-TRAIT RANKING  ({len(ranked)} traits total)")
    print(sep)

    header = f"  {'Rank':>4}  {'Trait':40s}  {'ROC-AUC':>8}  {'F1':>8}  {'Acc':>8}  Direction"
    divider = "  " + "-" * 80

    print(f"\n  ── TOP {n_top_bottom} (most predictive) ──")
    print(header)
    print(divider)
    for entry in ranked[:n_top_bottom]:
        print(
            f"  {entry['rank']:>4}  {entry['trait']:40s}  "
            f"{entry['roc_auc']:>8.4f}  {entry['f1']:>8.4f}  "
            f"{entry['accuracy']:>8.4f}  {entry['direction']}"
        )

    print(f"\n  ── BOTTOM {n_top_bottom} (least predictive) ──")
    print(header)
    print(divider)
    for entry in ranked[-n_top_bottom:]:
        print(
            f"  {entry['rank']:>4}  {entry['trait']:40s}  "
            f"{entry['roc_auc']:>8.4f}  {entry['f1']:>8.4f}  "
            f"{entry['accuracy']:>8.4f}  {entry['direction']}"
        )

    print(sep)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-trait univariate jailbreak classifier"
    )
    parser.add_argument(
        "--classified_path", type=str,
        default="full_trait_output/harmbench_activations/classified_responses.jsonl",
    )
    parser.add_argument(
        "--activations_path", type=str,
        default="full_trait_output/harmbench_activations/activations.pt",
    )
    parser.add_argument(
        "--trait_vectors_dir", type=str,
        default="full_trait_output/traits40_vectors/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total",
    )
    parser.add_argument(
        "--axis_path", type=str,
        default="full_trait_output/traits40_axes/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total/assistant_axis_pc1.pt",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="full_trait_output/harmbench_logreg",
    )
    parser.add_argument("--min_success_rate", type=float, default=MIN_SUCCESS_RATE)
    parser.add_argument("--max_success_rate", type=float, default=MAX_SUCCESS_RATE)
    parser.add_argument("--train_frac",        type=float, default=TRAIN_BEHAVIOR_FRAC)
    parser.add_argument("--seed",              type=int,   default=RANDOM_SEED)
    parser.add_argument("--n_top_bottom",      type=int,   default=N_TOP_BOTTOM)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and filter data ───────────────────────────────────────────────────
    print("Loading data...")
    rows        = load_classified(Path(args.classified_path))
    activations = load_activations(Path(args.activations_path))
    rows        = filter_human_jailbreak(rows)
    print(f"  {len(rows)} human_jailbreak rows, {len(activations)} activations")

    success_rates = compute_behavior_success_rates(rows)
    rows_filtered, kept_behaviors = filter_by_variance(
        rows, success_rates,
        min_rate=args.min_success_rate,
        max_rate=args.max_success_rate,
    )

    rate_distribution = {
        "always_refused   (0%)":    sum(1 for r in success_rates.values() if r == 0.0),
        "low             (0-20%)":  sum(1 for r in success_rates.values() if 0.0 < r < 0.2),
        "variable       (20-80%)":  sum(1 for r in success_rates.values() if 0.2 <= r <= 0.8),
        "high           (80-100%)": sum(1 for r in success_rates.values() if 0.8 < r < 1.0),
        "always_succeeded(100%)":   sum(1 for r in success_rates.values() if r == 1.0),
    }
    print(f"\n  Behavior success rate distribution:")
    for label, count in rate_distribution.items():
        print(f"    {label}: {count}")

    print(f"\n  After variance filter: {len(kept_behaviors)} behaviors, "
          f"{len(rows_filtered)} rows")

    train_behaviors, test_behaviors = split_behaviors_by_success_rate(
        kept_behaviors, success_rates,
        train_frac=args.train_frac,
        seed=args.seed,
    )
    print(f"  Train: {len(train_behaviors)} behaviors | "
          f"Test: {len(test_behaviors)} behaviors")

    # ── Run per layer ──────────────────────────────────────────────────────────
    all_layer_results = {}

    for layer in LAYERS:
        print(f"\n{'─' * 65}")
        print(f"  Layer {layer}: loading vectors...")

        trait_names, trait_vectors = load_trait_vectors(
            Path(args.trait_vectors_dir), layer
        )
        axis_vector   = load_axis_vector(Path(args.axis_path), layer)

        # Combine into one matrix: [n_traits+1, 4096]
        all_vectors   = np.vstack([trait_vectors, axis_vector[np.newaxis, :]])
        feature_names = trait_names + ["assistant_axis"]
        n_features    = len(feature_names)

        print(f"  {n_features} features (traits + axis)")
        print(f"  Building projection matrices...")

        # Project all samples at once — [n_samples, n_features]
        X_train_full, y_train = build_projection_arrays(
            rows_filtered, activations, all_vectors, layer, train_behaviors
        )
        X_test_full, y_test = build_projection_arrays(
            rows_filtered, activations, all_vectors, layer, test_behaviors
        )

        if len(X_train_full) == 0 or len(X_test_full) == 0:
            print(f"  Skipping layer {layer}: no data found")
            continue

        print(f"  Train: {X_train_full.shape} | Test: {X_test_full.shape}")
        print(f"  Running {n_features} univariate classifiers...")

        results = []
        for i, feat_name in enumerate(tqdm(feature_names, desc=f"Layer {layer}")):
            X_train_col = X_train_full[:, i : i + 1]
            X_test_col  = X_test_full[:, i : i + 1]

            metrics = run_single_feature_logreg(
                X_train_col, y_train,
                X_test_col,  y_test,
            )
            metrics["trait"] = feat_name
            results.append(metrics)

        # Rank by ROC-AUC descending, handle NaN
        results.sort(
            key=lambda x: x["roc_auc"] if not np.isnan(x["roc_auc"]) else -1,
            reverse=True,
        )
        for rank, entry in enumerate(results, 1):
            entry["rank"] = rank

        all_layer_results[f"layer_{layer}"] = results
        print_ranking(layer, results, args.n_top_bottom)

    # ── Save full results ──────────────────────────────────────────────────────
    out_path = output_dir / "per_trait_ranking.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": {
                    "min_success_rate":  args.min_success_rate,
                    "max_success_rate":  args.max_success_rate,
                    "train_frac":        args.train_frac,
                    "seed":              args.seed,
                    "layers":            LAYERS,
                    "n_behaviors_kept":  len(kept_behaviors),
                    "n_train_behaviors": len(train_behaviors),
                    "n_test_behaviors":  len(test_behaviors),
                    "rate_distribution": rate_distribution,
                },
                "results": all_layer_results,
            },
            f, indent=2,
        )
    print(f"\nFull per-trait ranking saved to {out_path}")


if __name__ == "__main__":
    main()
