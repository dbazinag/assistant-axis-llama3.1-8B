#!/usr/bin/env python3
"""
classify_jailbreak_logreg_pairs.py

Logistic regression classifier predicting jailbreak success from persona
vector projections of pre-generation activations.

Key design:
  - Uses LLM judge labels (jailbroken field, post judge_harmbench_responses.py)
  - Human jailbreak pairs only (DirectRequest excluded)
  - Variance filter: behaviors with 20-80% success rate across templates
  - Strict pair-level train/test split: every (behavior, jailbreak_template)
    pair is assigned to exactly one set. No behavior AND no jailbreak template
    index seen in training appears in test.
  - Features: dot product projection of pre-generation activation onto each
    trait vector + assistant axis (~230 features)
  - Two separate runs: layer 16 and layer 28
  - Balanced classes via class_weight="balanced"

Usage:
  uv run full_trait_tools/classify_jailbreak_logreg_pairs.py
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

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

RANDOM_SEED         = 43
TRAIN_PAIR_FRAC     = 0.7
MIN_SUCCESS_RATE    = 0.20
MAX_SUCCESS_RATE    = 0.80
LAYERS              = [16, 28]


# ── Vector loading ─────────────────────────────────────────────────────────────

def load_trait_vectors(
    vectors_dir: Path,
    layer: int,
) -> Tuple[List[str], np.ndarray]:
    pt_files = sorted(vectors_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {vectors_dir}")
    trait_names, vectors = [], []
    for pt_file in pt_files:
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        trait_names.append(pt_file.stem)
        vectors.append(data["vector"][layer].float().numpy())
    return trait_names, np.stack(vectors)


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


# ── Pair-level train/test split ────────────────────────────────────────────────

def split_pools(
    rows: List[dict],
    train_frac: float,
    seed: int,
) -> Tuple[Set[str], Set[str], Set[int], Set[int]]:
    """
    Splits behaviors and jailbreak templates into completely separate pools.
    Train set = (train_behavior, train_template) pairs only.
    Test set  = (test_behavior,  test_template)  pairs only.
    Mixed pairs are dropped entirely.
    """
    rng = random.Random(seed)
    all_behaviors = sorted({r["behavior_id"]   for r in rows})
    all_templates = sorted({r["jailbreak_idx"]  for r in rows})
    rng.shuffle(all_behaviors)
    rng.shuffle(all_templates)
    n_train_beh = max(1, int(len(all_behaviors) * train_frac))
    n_train_tpl = max(1, int(len(all_templates) * train_frac))
    train_behaviors = set(all_behaviors[:n_train_beh])
    test_behaviors  = set(all_behaviors[n_train_beh:])
    train_templates = set(all_templates[:n_train_tpl])
    test_templates  = set(all_templates[n_train_tpl:])
    return train_behaviors, test_behaviors, train_templates, test_templates


# ── Feature matrix ─────────────────────────────────────────────────────────────

def project_activation(
    activation: np.ndarray,    # [4096]
    trait_vectors: np.ndarray, # [n_traits, 4096]
    axis_vector: np.ndarray,   # [4096]
) -> np.ndarray:
    """Dot product projection onto all trait vectors + axis."""
    trait_proj = trait_vectors @ activation
    axis_proj  = np.array([axis_vector @ activation])
    return np.concatenate([trait_proj, axis_proj])


def build_feature_matrix(
    rows: List[dict],
    activations: Dict[int, Dict[str, torch.Tensor]],
    trait_vectors: np.ndarray,
    axis_vector: np.ndarray,
    layer: int,
    behavior_pool: Set[str],
    template_pool: Set[int],
) -> Tuple[np.ndarray, np.ndarray]:
    layer_key = str(layer)
    X_list, y_list = [], []

    for row in rows:
        if row["behavior_id"]   not in behavior_pool:
            continue
        if row["jailbreak_idx"] not in template_pool:
            continue
        pid = row["pair_id"]
        if pid not in activations or layer_key not in activations[pid]:
            continue

        act      = activations[pid][layer_key].float().numpy()
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
    coefs    = clf.coef_[0]
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
    print(f"  Train: {metrics['n_train']} pairs "
          f"(pos={metrics['n_train_pos']}, neg={metrics['n_train_neg']})")
    print(f"  Test : {metrics['n_test']} pairs "
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
        description="Logistic regression on persona projections with strict pair-level split"
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
    parser.add_argument("--train_frac",        type=float, default=TRAIN_PAIR_FRAC)
    parser.add_argument("--seed",              type=int,   default=RANDOM_SEED)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data...")
    rows        = load_classified(Path(args.classified_path))
    activations = load_activations(Path(args.activations_path))

    # Check for LLM judge labels
    has_llm_labels = any("llm_judge_raw" in r for r in rows)
    label_source   = "LLM judge (gpt-4.1-mini)" if has_llm_labels else "HarmBench classifier"
    print(f"  Label source: {label_source}")
    print(f"  {len(rows)} total rows, {len(activations)} activations")

    # ── Filter to human jailbreak ──────────────────────────────────────────────
    rows = filter_human_jailbreak(rows)
    print(f"  {len(rows)} rows after filtering to human_jailbreak only")

    # ── Per-behavior success rates ─────────────────────────────────────────────
    success_rates = compute_behavior_success_rates(rows)

    rate_distribution = {
        "always_refused   (0%)":    sum(1 for r in success_rates.values() if r == 0.0),
        "low             (0-20%)":  sum(1 for r in success_rates.values() if 0.0 < r < 0.2),
        "variable       (20-80%)":  sum(1 for r in success_rates.values() if 0.2 <= r <= 0.8),
        "high           (80-100%)": sum(1 for r in success_rates.values() if 0.8 < r < 1.0),
        "always_succeeded(100%)":   sum(1 for r in success_rates.values() if r == 1.0),
    }
    print(f"\n  Behavior success rate distribution ({len(success_rates)} behaviors):")
    for label, count in rate_distribution.items():
        print(f"    {label}: {count}")

    # ── Variance filter ────────────────────────────────────────────────────────
    rows_filtered, kept_behaviors = filter_by_variance(
        rows, success_rates,
        min_rate=args.min_success_rate,
        max_rate=args.max_success_rate,
    )
    n_jb     = sum(r["jailbroken"] for r in rows_filtered)
    n_not_jb = sum(not r["jailbroken"] for r in rows_filtered)
    print(f"\n  After variance filter ({args.min_success_rate:.0%}–{args.max_success_rate:.0%}):")
    print(f"    {len(kept_behaviors)} behaviors kept")
    print(f"    {len(rows_filtered)} pairs kept")
    print(f"    {n_jb} jailbroken ({100*n_jb/len(rows_filtered):.1f}%)")
    print(f"    {n_not_jb} not jailbroken ({100*n_not_jb/len(rows_filtered):.1f}%)")

    if len(kept_behaviors) < 10:
        print("\n  WARNING: Very few behaviors kept — consider loosening the filter.")

    # ── Strict pool split ─────────────────────────────────────────────────────
    train_behaviors, test_behaviors, train_templates, test_templates = split_pools(
        rows_filtered,
        train_frac=args.train_frac,
        seed=args.seed,
    )
    train_rows = [
        r for r in rows_filtered
        if r["behavior_id"] in train_behaviors and r["jailbreak_idx"] in train_templates
    ]
    test_rows = [
        r for r in rows_filtered
        if r["behavior_id"] in test_behaviors and r["jailbreak_idx"] in test_templates
    ]
    dropped   = len(rows_filtered) - len(train_rows) - len(test_rows)
    train_pos = sum(r["jailbroken"] for r in train_rows)
    test_pos  = sum(r["jailbroken"] for r in test_rows)
    print(f"\n  Strict pool split:")
    print(f"    Train behaviors : {len(train_behaviors)} | Train templates: {len(train_templates)}")
    print(f"    Test behaviors  : {len(test_behaviors)}  | Test templates : {len(test_templates)}")
    print(f"    Train pairs     : {len(train_rows)} "
          f"(pos={train_pos}, neg={len(train_rows)-train_pos}, "
          f"rate={100*train_pos/max(1,len(train_rows)):.1f}%)")
    print(f"    Test pairs      : {len(test_rows)} "
          f"(pos={test_pos}, neg={len(test_rows)-test_pos}, "
          f"rate={100*test_pos/max(1,len(test_rows)):.1f}%)")
    print(f"    Dropped (mixed) : {dropped} pairs ({100*dropped/max(1,len(rows_filtered)):.1f}%)")

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

        print(f"  {len(feature_names)} features (traits + axis)")
        print(f"  Building feature matrices...")

        X_train, y_train = build_feature_matrix(
            rows_filtered, activations, trait_vectors, axis_vector,
            layer, train_behaviors, train_templates,
        )
        X_test, y_test = build_feature_matrix(
            rows_filtered, activations, trait_vectors, axis_vector,
            layer, test_behaviors, test_templates,
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
        "label_source": label_source,
        "split_strategy": "strict_behavior_and_template_pool_split",
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
            "n_behaviors_total":     len(success_rates),
            "n_behaviors_kept":      len(kept_behaviors),
            "n_pairs_total":         len(rows_filtered),
            "n_train_behaviors":     len(train_behaviors),
            "n_test_behaviors":      len(test_behaviors),
            "n_train_templates":     len(train_templates),
            "n_test_templates":      len(test_templates),
            "n_train_pairs":         len(train_rows),
            "n_test_pairs":          len(test_rows),
            "n_dropped_pairs":       dropped,
            "rate_distribution":     rate_distribution,
            "behavior_success_rates": success_rates,
        },
        "results": all_results,
    }

    out_path = output_dir / "logreg_pairs_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
