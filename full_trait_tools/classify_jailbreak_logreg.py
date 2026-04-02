#!/usr/bin/env python3
"""
classify_jailbreak_logreg.py

Predicts jailbreak success from persona vector projections of pre-generation
activations.

Pipeline:
  1. Load 229 trait vectors (pre_generation_last_token) + assistant axis
  2. For each (pair_id, layer) activation: project onto all trait vectors
     and the assistant axis → 230-dim feature vector
  3. Filter to human_jailbreak only
  4. Filter to behaviors with 20%-80% jailbreak success rate (variance filter)
  5. Behavior-level train/test split (80/20, stratified by success rate)
  6. Two separate logistic regression runs: layer 16 vs layer 28
  7. Report accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
     top most predictive trait dimensions

This tests the hypothesis: do persona vector projections at the pre-generation
token predict whether a jailbreak attempt will succeed?

Usage:
  uv run full_trait_tools/classify_jailbreak_logreg.py
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


# ── Vector loading ─────────────────────────────────────────────────────────────

def load_trait_vectors(
    vectors_dir: Path,
    layer: int,
) -> Tuple[List[str], np.ndarray]:
    """
    Load all trait vectors from vectors_dir.
    Each .pt file has key "vector": Tensor[32, 4096].
    Extracts the vector at the given layer index.

    Returns:
        trait_names: list of trait names (from filename stems)
        vectors:     np.ndarray [n_traits, 4096]
    """
    pt_files = sorted(vectors_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {vectors_dir}")

    trait_names = []
    vectors     = []

    for pt_file in pt_files:
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        vec  = data["vector"]  # Tensor[32, 4096]
        trait_names.append(pt_file.stem)
        vectors.append(vec[layer].float().numpy())  # [4096]

    return trait_names, np.stack(vectors)  # [n_traits, 4096]


def load_axis_vector(axis_path: Path, layer: int) -> np.ndarray:
    """
    Load the assistant axis vector at the given layer.
    File has key "axis": Tensor[32, 4096].
    Returns np.ndarray [4096].
    """
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
    """Returns {pair_id: {"16": Tensor[4096], "28": Tensor[4096]}}"""
    return torch.load(path, map_location="cpu", weights_only=False)


# ── Projection ─────────────────────────────────────────────────────────────────

def project_activation(
    activation: np.ndarray,    # [4096]
    trait_vectors: np.ndarray, # [n_traits, 4096]
    axis_vector: np.ndarray,   # [4096]
) -> np.ndarray:
    """
    Project a single activation onto all trait vectors and the assistant axis.
    Uses raw dot product to preserve magnitude information.
    Returns np.ndarray [n_traits + 1] — traits first, axis last.
    """
    trait_projections = trait_vectors @ activation           # [n_traits]
    axis_projection   = np.array([axis_vector @ activation]) # [1]
    return np.concatenate([trait_projections, axis_projection])


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
    Stratified behavior-level split by success rate tertile so both
    train and test sets have similar distributions.
    """
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


# ── Feature matrix ─────────────────────────────────────────────────────────────

def build_feature_matrix(
    rows: List[dict],
    activations: Dict[int, Dict[str, torch.Tensor]],
    trait_vectors: np.ndarray,  # [n_traits, 4096]
    axis_vector: np.ndarray,    # [4096]
    layer: int,
    behavior_set: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build X [n_samples, n_traits+1] and y [n_samples] by projecting
    each activation onto trait vectors + assistant axis.
    """
    behavior_set_s = set(behavior_set)
    layer_key      = str(layer)

    X_list, y_list = [], []

    for row in rows:
        if row["behavior_id"] not in behavior_set_s:
            continue
        pid = row["pair_id"]
        if pid not in activations:
            continue
        if layer_key not in activations[pid]:
            continue

        act      = activations[pid][layer_key].float().numpy()  # [4096]
        features = project_activation(act, trait_vectors, axis_vector)
        X_list.append(features)
        y_list.append(int(row["jailbroken"]))

    if not X_list:
        return np.array([]), np.array([])

    return np.stack(X_list), np.array(y_list)


# ── Classifier ─────────────────────────────────────────────────────────────────

def run_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
) -> dict:
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
        "n_train":               int(len(y_train)),
        "n_test":                int(len(y_test)),
        "n_train_pos":           int(y_train.sum()),
        "n_train_neg":           int((1 - y_train).sum()),
        "n_test_pos":            int(y_test.sum()),
        "n_test_neg":            int((1 - y_test).sum()),
        "accuracy":              float(accuracy_score(y_test, y_pred)),
        "precision":             float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":                float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":                    float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":               float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix":      cm.tolist(),
        "classification_report": classification_report(y_test, y_pred),
    }

    # Top predictive traits by |coefficient|
    coefs    = clf.coef_[0]  # [n_traits + 1]
    top_idx  = np.argsort(np.abs(coefs))[::-1][:20]
    top_traits = [
        {
            "rank":      int(rank),
            "feature":   feature_names[idx],
            "coef":      float(coefs[idx]),
            "direction": "→ jailbroken" if coefs[idx] > 0 else "→ not jailbroken",
        }
        for rank, idx in enumerate(top_idx, 1)
    ]
    metrics["top_20_traits"] = top_traits

    return metrics


# ── Printing ───────────────────────────────────────────────────────────────────

def print_results(layer: int, metrics: dict) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  LAYER {layer} RESULTS")
    print(sep)
    print(f"  Train: {metrics['n_train']} samples "
          f"(pos={metrics['n_train_pos']}, neg={metrics['n_train_neg']})")
    print(f"  Test : {metrics['n_test']} samples "
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
    print("  Top 20 most predictive trait projections:")
    print(f"  {'Rank':>4}  {'Trait':40s}  {'Coef':>10}  Direction")
    print("  " + "-" * 75)
    for entry in metrics["top_20_traits"]:
        print(f"  {entry['rank']:>4}  {entry['feature']:40s}  "
              f"{entry['coef']:>10.4f}  {entry['direction']}")
    print(sep)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logistic regression on persona vector projections for jailbreak prediction"
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading classified responses and activations...")
    rows        = load_classified(Path(args.classified_path))
    activations = load_activations(Path(args.activations_path))
    print(f"  {len(rows)} classified rows")
    print(f"  {len(activations)} activation entries")

    # ── Filter to human jailbreak ──────────────────────────────────────────────
    rows = filter_human_jailbreak(rows)
    print(f"  {len(rows)} rows after filtering to human_jailbreak only")

    # ── Per-behavior success rates ─────────────────────────────────────────────
    success_rates = compute_behavior_success_rates(rows)
    print(f"\n  {len(success_rates)} unique behaviors")

    rate_distribution = {
        "always_refused   (0%)":    sum(1 for r in success_rates.values() if r == 0.0),
        "low             (0-20%)":  sum(1 for r in success_rates.values() if 0.0 < r < 0.2),
        "variable       (20-80%)":  sum(1 for r in success_rates.values() if 0.2 <= r <= 0.8),
        "high           (80-100%)": sum(1 for r in success_rates.values() if 0.8 < r < 1.0),
        "always_succeeded(100%)":   sum(1 for r in success_rates.values() if r == 1.0),
    }
    print("\n  Behavior success rate distribution:")
    for label, count in rate_distribution.items():
        print(f"    {label}: {count}")

    # ── Variance filter ────────────────────────────────────────────────────────
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
        print("\n  WARNING: Very few behaviors kept — consider loosening the filter.")

    # ── Behavior-level train/test split ───────────────────────────────────────
    train_behaviors, test_behaviors = split_behaviors_by_success_rate(
        kept_behaviors, success_rates,
        train_frac=args.train_frac,
        seed=args.seed,
    )
    print(f"\n  Behavior split:")
    print(f"    Train : {len(train_behaviors)} behaviors")
    print(f"    Test  : {len(test_behaviors)} behaviors")

    # ── Run classifier per layer ───────────────────────────────────────────────
    all_results = {}

    for layer in LAYERS:
        print(f"\n{'─' * 65}")
        print(f"  Loading trait vectors for layer {layer}...")

        trait_names, trait_vectors = load_trait_vectors(
            Path(args.trait_vectors_dir), layer
        )
        axis_vector   = load_axis_vector(Path(args.axis_path), layer)
        feature_names = trait_names + ["assistant_axis"]

        print(f"  {len(trait_names)} trait vectors + 1 axis = {len(feature_names)} features")
        print(f"  Building feature matrices...")

        X_train, y_train = build_feature_matrix(
            rows_filtered, activations, trait_vectors, axis_vector,
            layer, train_behaviors,
        )
        X_test, y_test = build_feature_matrix(
            rows_filtered, activations, trait_vectors, axis_vector,
            layer, test_behaviors,
        )

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  Skipping layer {layer}: no data found in activations")
            continue

        print(f"  Train: {X_train.shape}  Test: {X_test.shape}")
        print(f"  Fitting logistic regression...")

        metrics = run_logreg(X_train, y_train, X_test, y_test, feature_names)
        all_results[f"layer_{layer}"] = metrics
        print_results(layer, metrics)

    # ── Save summary ───────────────────────────────────────────────────────────
    summary = {
        "config": {
            "classified_path":   args.classified_path,
            "activations_path":  args.activations_path,
            "trait_vectors_dir": args.trait_vectors_dir,
            "axis_path":         args.axis_path,
            "min_success_rate":  args.min_success_rate,
            "max_success_rate":  args.max_success_rate,
            "train_frac":        args.train_frac,
            "seed":              args.seed,
            "layers":            LAYERS,
        },
        "data": {
            "n_behaviors_total":      len(success_rates),
            "n_behaviors_kept":       len(kept_behaviors),
            "n_train_behaviors":      len(train_behaviors),
            "n_test_behaviors":       len(test_behaviors),
            "train_behaviors":        train_behaviors,
            "test_behaviors":         test_behaviors,
            "behavior_success_rates": success_rates,
            "rate_distribution":      rate_distribution,
        },
        "results": all_results,
    }

    out_path = output_dir / "logreg_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
