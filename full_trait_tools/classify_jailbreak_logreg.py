#!/usr/bin/env python3
"""
classify_jailbreak_logreg.py

Trains logistic regression classifiers to predict jailbreak success from
pre-generation activations at layers 16 and 28.

Key design decisions:
  - Human jailbreak pairs only (DirectRequest excluded)
  - Behavior-level variance filter: keep only behaviors with 20%-80% jailbreak
    success rate across jailbreak templates (forces classifier to learn from
    activation signal, not just topic difficulty)
  - Behavior-level train/test split: train on some behaviors, test on held-out
    behaviors — tests genuine generalization, not memorization
  - Two separate runs: layer 16 vs layer 28

Outputs (per layer):
  - Accuracy, Precision, Recall, F1, ROC-AUC
  - Confusion matrix
  - Top 20 most predictive activation dimensions (by coefficient magnitude)
  - Summary JSON saved to output_dir

Usage:
  uv run full_trait_tools/classify_jailbreak_logreg.py

  uv run full_trait_tools/classify_jailbreak_logreg.py \\
    --classified_path full_trait_output/harmbench_activations/classified_responses.jsonl \\
    --activations_path full_trait_output/harmbench_activations/activations.pt \\
    --output_dir full_trait_output/harmbench_logreg
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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

RANDOM_SEED         = 42
TRAIN_BEHAVIOR_FRAC = 0.8
MIN_SUCCESS_RATE    = 0.20
MAX_SUCCESS_RATE    = 0.80
LAYERS              = [16, 28]


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
    """
    Returns {pair_id: {"16": Tensor[4096], "28": Tensor[4096]}}
    """
    return torch.load(path, map_location="cpu", weights_only=False)


# ── Filtering ──────────────────────────────────────────────────────────────────

def filter_human_jailbreak(rows: List[dict]) -> List[dict]:
    return [r for r in rows if r.get("attack_type") == "human_jailbreak"]


def compute_behavior_success_rates(rows: List[dict]) -> Dict[str, float]:
    counts = defaultdict(lambda: {"total": 0, "jailbroken": 0})
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
    """Keep only rows whose behavior has success rate in [min_rate, max_rate]."""
    kept_behaviors = {
        bid for bid, rate in success_rates.items()
        if min_rate <= rate <= max_rate
    }
    filtered = [r for r in rows if r["behavior_id"] in kept_behaviors]
    return filtered, sorted(kept_behaviors)


# ── Train/test split ───────────────────────────────────────────────────────────

def split_behaviors_by_success_rate(
    behaviors: List[str],
    success_rates: Dict[str, float],
    train_frac: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Split behaviors into train/test, stratified by success rate bucket
    (low / mid / high) so both sets have similar distributions.
    """
    rng = random.Random(seed)

    # Sort into tertiles
    sorted_bids = sorted(behaviors, key=lambda b: success_rates[b])
    n = len(sorted_bids)
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


# ── Feature extraction ─────────────────────────────────────────────────────────

def build_feature_matrix(
    rows: List[dict],
    activations: Dict[int, Dict[str, torch.Tensor]],
    layer: int,
    behavior_set: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Build X [n_samples, 4096] and y [n_samples] for rows belonging to
    behavior_set and having activations at the given layer.

    Returns (X, y, valid_pair_ids).
    """
    behavior_set_s = set(behavior_set)
    layer_key      = str(layer)

    X_list, y_list, pair_ids = [], [], []

    for row in rows:
        if row["behavior_id"] not in behavior_set_s:
            continue
        pid = row["pair_id"]
        if pid not in activations:
            continue
        if layer_key not in activations[pid]:
            continue

        vec = activations[pid][layer_key]  # Tensor[4096]
        X_list.append(vec.float().numpy())
        y_list.append(int(row["jailbroken"]))
        pair_ids.append(pid)

    if not X_list:
        return np.array([]), np.array([]), []

    return np.stack(X_list), np.array(y_list), pair_ids


# ── Classifier ─────────────────────────────────────────────────────────────────

def run_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Fit logistic regression, evaluate on test set.
    Returns dict of metrics + coefficients.
    """
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "n_train":        int(len(y_train)),
        "n_test":         int(len(y_test)),
        "n_train_pos":    int(y_train.sum()),
        "n_train_neg":    int((1 - y_train).sum()),
        "n_test_pos":     int(y_test.sum()),
        "n_test_neg":     int((1 - y_test).sum()),
        "accuracy":       float(accuracy_score(y_test, y_pred)),
        "precision":      float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":         float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":             float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":        float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred),
    }

    # Top predictive dimensions by |coefficient|
    coefs     = clf.coef_[0]  # [4096]
    top_idx   = np.argsort(np.abs(coefs))[::-1][:20]
    top_coefs = [
        {"dim": int(idx), "coef": float(coefs[idx])}
        for idx in top_idx
    ]
    metrics["top_20_dimensions"] = top_coefs

    return metrics


# ── Printing ───────────────────────────────────────────────────────────────────

def print_results(layer: int, metrics: dict) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  LAYER {layer} RESULTS")
    print(sep)
    print(f"  Train samples : {metrics['n_train']} "
          f"(pos={metrics['n_train_pos']}, neg={metrics['n_train_neg']})")
    print(f"  Test samples  : {metrics['n_test']} "
          f"(pos={metrics['n_test_pos']}, neg={metrics['n_test_neg']})")
    print()
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print()
    print("  Confusion matrix (rows=actual, cols=predicted):")
    cm = metrics["confusion_matrix"]
    print(f"              Pred 0   Pred 1")
    print(f"    Actual 0  {cm[0][0]:6d}   {cm[0][1]:6d}")
    print(f"    Actual 1  {cm[1][0]:6d}   {cm[1][1]:6d}")
    print()
    print("  Classification report:")
    for line in metrics["classification_report"].splitlines():
        print(f"    {line}")
    print()
    print("  Top 20 most predictive activation dimensions:")
    print(f"  {'Rank':>4}  {'Dim':>6}  {'Coef':>10}  Direction")
    print("  " + "-" * 40)
    for rank, entry in enumerate(metrics["top_20_dimensions"], 1):
        direction = "→ jailbroken" if entry["coef"] > 0 else "→ not jailbroken"
        print(f"  {rank:>4}  {entry['dim']:>6}  {entry['coef']:>10.4f}  {direction}")
    print(sep)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logistic regression classifier on pre-generation activations"
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
        "--output_dir", type=str,
        default="full_trait_output/harmbench_logreg",
    )
    parser.add_argument(
        "--min_success_rate", type=float, default=MIN_SUCCESS_RATE,
    )
    parser.add_argument(
        "--max_success_rate", type=float, default=MAX_SUCCESS_RATE,
    )
    parser.add_argument(
        "--train_frac", type=float, default=TRAIN_BEHAVIOR_FRAC,
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data...")
    rows        = load_classified(Path(args.classified_path))
    activations = load_activations(Path(args.activations_path))
    print(f"  {len(rows)} classified rows, {len(activations)} activation entries")

    # ── Filter to human jailbreak only ────────────────────────────────────────
    rows = filter_human_jailbreak(rows)
    print(f"  {len(rows)} rows after filtering to human_jailbreak only")

    # ── Compute per-behavior success rates ────────────────────────────────────
    success_rates = compute_behavior_success_rates(rows)
    print(f"\n  {len(success_rates)} behaviors total")

    rate_distribution = {
        "always_refused (0%)":    sum(1 for r in success_rates.values() if r == 0.0),
        "low (0-20%)":            sum(1 for r in success_rates.values() if 0.0 < r < 0.2),
        "variable (20-80%)":      sum(1 for r in success_rates.values() if 0.2 <= r <= 0.8),
        "high (80-100%)":         sum(1 for r in success_rates.values() if 0.8 < r < 1.0),
        "always_succeeded (100%)":sum(1 for r in success_rates.values() if r == 1.0),
    }
    print("\n  Behavior success rate distribution:")
    for label, count in rate_distribution.items():
        print(f"    {label:30s}: {count}")

    # ── Apply variance filter ─────────────────────────────────────────────────
    rows_filtered, kept_behaviors = filter_by_variance(
        rows, success_rates,
        min_rate=args.min_success_rate,
        max_rate=args.max_success_rate,
    )
    print(f"\n  After variance filter ({args.min_success_rate:.0%}–{args.max_success_rate:.0%}):")
    print(f"    {len(kept_behaviors)} behaviors kept")
    print(f"    {len(rows_filtered)} rows kept")
    print(f"    {sum(r['jailbroken'] for r in rows_filtered)} jailbroken")
    print(f"    {sum(not r['jailbroken'] for r in rows_filtered)} not jailbroken")

    if len(kept_behaviors) < 10:
        print("\nWARNING: Very few behaviors kept — consider loosening the filter.")

    # ── Behavior-level train/test split ───────────────────────────────────────
    train_behaviors, test_behaviors = split_behaviors_by_success_rate(
        kept_behaviors, success_rates,
        train_frac=args.train_frac,
        seed=args.seed,
    )
    print(f"\n  Behavior split:")
    print(f"    Train: {len(train_behaviors)} behaviors")
    print(f"    Test : {len(test_behaviors)} behaviors")

    # ── Run classifier for each layer ─────────────────────────────────────────
    all_results = {}

    for layer in LAYERS:
        print(f"\nBuilding feature matrices for layer {layer}...")

        X_train, y_train, _ = build_feature_matrix(
            rows_filtered, activations, layer, train_behaviors
        )
        X_test, y_test, _ = build_feature_matrix(
            rows_filtered, activations, layer, test_behaviors
        )

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  Skipping layer {layer}: no data found in activations")
            continue

        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Fitting logistic regression...")

        metrics = run_logreg(X_train, y_train, X_test, y_test)
        all_results[f"layer_{layer}"] = metrics
        print_results(layer, metrics)

    # ── Save summary ───────────────────────────────────────────────────────────
    summary = {
        "config": {
            "classified_path":  args.classified_path,
            "activations_path": args.activations_path,
            "min_success_rate": args.min_success_rate,
            "max_success_rate": args.max_success_rate,
            "train_frac":       args.train_frac,
            "seed":             args.seed,
            "layers":           LAYERS,
        },
        "data": {
            "n_behaviors_total":    len(success_rates),
            "n_behaviors_kept":     len(kept_behaviors),
            "n_train_behaviors":    len(train_behaviors),
            "n_test_behaviors":     len(test_behaviors),
            "train_behaviors":      train_behaviors,
            "test_behaviors":       test_behaviors,
            "behavior_success_rates": success_rates,
            "rate_distribution":    rate_distribution,
        },
        "results": all_results,
    }

    out_path = output_dir / "logreg_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
