#!/usr/bin/env python3
"""
assistant_axis_predictor.py

Directly tests the central hypothesis of the Assistant Axis project:
does the model's pre-generation activation along the assistant axis
predict whether a jailbreak attempt will succeed?

No classifier is trained. For each pair:
  - Take the pre-generation activation at layers 16 and 28
  - Project it onto the assistant axis (dot product)
  - Use that single scalar as a jailbreak success predictor
  - Compute ROC-AUC

This is the cleanest possible test of the hypothesis — a single number
that says how much the assistant axis direction predicts jailbreak outcome.

Also reports:
  - Distribution of projections split by jailbroken/not (mean, std, overlap)
  - Mann-Whitney U test (non-parametric, tests if distributions differ)
  - ROC-AUC with 95% bootstrap CI
  - Comparison: axis projection AUC vs random baseline

Uses the same variance filter (20-80% success rate) and includes ALL
pairs (no train/test split needed since nothing is trained).

Usage:
  uv run full_trait_tools/assistant_axis_predictor.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

RANDOM_SEED      = 42
MIN_SUCCESS_RATE = 0.20
MAX_SUCCESS_RATE = 0.80
LAYERS           = [16, 28]
N_BOOTSTRAP      = 2000


# ── Loading ────────────────────────────────────────────────────────────────────

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


def load_axis(axis_path: Path, layer: int) -> np.ndarray:
    data = torch.load(axis_path, map_location="cpu", weights_only=False)
    v = data["axis"][layer].float().numpy()
    return v / (np.linalg.norm(v) + 1e-12)


# ── Filtering ──────────────────────────────────────────────────────────────────

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
    return [r for r in rows if r["behavior_id"] in kept], sorted(kept)


# ── Core ───────────────────────────────────────────────────────────────────────

def get_projections_and_labels(
    rows: List[dict],
    activations: Dict,
    axis: np.ndarray,
    layer: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project each activation onto the axis, return (projections, labels)."""
    layer_key = str(layer)
    projs, labels = [], []
    for row in rows:
        pid = row["pair_id"]
        if pid not in activations or layer_key not in activations[pid]:
            continue
        act  = activations[pid][layer_key].float().numpy()
        proj = float(np.dot(act, axis))
        projs.append(proj)
        labels.append(int(row["jailbroken"]))
    return np.array(projs), np.array(labels)


def bootstrap_auc_ci(
    projections: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float, float]:
    """Returns (auc, ci_lo, ci_hi) via bootstrap."""
    rng  = np.random.default_rng(seed)
    n    = len(projections)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            aucs.append(float(roc_auc_score(labels[idx], projections[idx])))
        except ValueError:
            pass
    aucs = np.array(aucs)
    auc  = float(roc_auc_score(labels, projections))
    return auc, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ── Printing ───────────────────────────────────────────────────────────────────

def print_layer_results(layer: int, result: dict) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  LAYER {layer} — Assistant Axis as Direct Jailbreak Predictor")
    print(sep)
    print(f"  Samples : {result['n_total']} "
          f"({result['n_jailbroken']} jailbroken, {result['n_not_jailbroken']} not)")
    print()
    print(f"  ── Projection distributions ──")
    print(f"  {'':30s}  {'Jailbroken':>12}  {'Not jailbroken':>15}")
    print(f"  {'Mean projection':30s}  "
          f"{result['mean_jailbroken']:>12.4f}  {result['mean_not_jailbroken']:>15.4f}")
    print(f"  {'Std projection':30s}  "
          f"{result['std_jailbroken']:>12.4f}  {result['std_not_jailbroken']:>15.4f}")
    print(f"  {'Median projection':30s}  "
          f"{result['median_jailbroken']:>12.4f}  {result['median_not_jailbroken']:>15.4f}")
    print()
    print(f"  ── Statistical tests ──")
    print(f"  Mann-Whitney U p-value : {result['mannwhitney_p']:.6f} "
          f"({'significant' if result['mannwhitney_p'] < 0.05 else 'not significant'})")
    print(f"  Effect size (r)        : {result['effect_size_r']:.4f}")
    print()
    print(f"  ── ROC-AUC ──")
    print(f"  AUC                    : {result['auc']:.4f}")
    print(f"  95% bootstrap CI       : [{result['auc_ci_lo']:.4f}, {result['auc_ci_hi']:.4f}]")
    print(f"  Baseline (chance)      : 0.5000")
    print()

    # Interpret
    auc = result["auc"]
    if result["auc_ci_lo"] > 0.5:
        if auc > 0.65:
            verdict = "MEANINGFUL signal — axis projection predicts jailbreak outcome"
        else:
            verdict = "WEAK but significant signal — axis has some predictive value"
    else:
        verdict = "NO significant signal — axis projection does not predict jailbreak outcome"

    sign = "positive" if result["mean_jailbroken"] > result["mean_not_jailbroken"] else "negative"
    direction = (
        "Higher axis projection → more likely jailbroken"
        if result["mean_jailbroken"] > result["mean_not_jailbroken"]
        else "Lower axis projection → more likely jailbroken"
    )

    print(f"  Verdict  : {verdict}")
    print(f"  Direction: {direction}")
    print(f"  (Note: PCA sign of assistant axis is arbitrary — direction may be flipped)")
    print(sep)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test assistant axis as direct jailbreak predictor"
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
        "--axis_path", type=str,
        default="full_trait_output/traits40_axes/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total/assistant_axis_pc1.pt",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="full_trait_output/harmbench_logreg",
    )
    parser.add_argument("--min_success_rate", type=float, default=MIN_SUCCESS_RATE)
    parser.add_argument("--max_success_rate", type=float, default=MAX_SUCCESS_RATE)
    parser.add_argument("--n_bootstrap",      type=int,   default=N_BOOTSTRAP)
    parser.add_argument("--layers", nargs="+", type=int,  default=LAYERS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and filter ────────────────────────────────────────────────────────
    print("Loading data...")
    rows        = load_classified(Path(args.classified_path))
    activations = load_activations(Path(args.activations_path))
    rows        = filter_human_jailbreak(rows)
    success_rates = compute_behavior_success_rates(rows)
    rows_filtered, kept_behaviors = filter_by_variance(
        rows, success_rates,
        min_rate=args.min_success_rate,
        max_rate=args.max_success_rate,
    )
    n_jb = sum(r["jailbroken"] for r in rows_filtered)
    print(f"  {len(kept_behaviors)} behaviors, {len(rows_filtered)} pairs "
          f"({n_jb} jailbroken, {len(rows_filtered)-n_jb} not)")

    all_results = {}

    for layer in args.layers:
        axis = load_axis(Path(args.axis_path), layer)
        print(f"\n  Computing projections for layer {layer}...")

        projections, labels = get_projections_and_labels(
            rows_filtered, activations, axis, layer
        )

        jb_proj     = projections[labels == 1]
        not_jb_proj = projections[labels == 0]

        # Mann-Whitney U test
        u_stat, p_val = stats.mannwhitneyu(
            jb_proj, not_jb_proj, alternative="two-sided"
        )
        # Effect size r = z / sqrt(n)
        n_total   = len(projections)
        z_score   = (u_stat - (len(jb_proj) * len(not_jb_proj) / 2)) / np.sqrt(
            len(jb_proj) * len(not_jb_proj) * (n_total + 1) / 12
        )
        effect_r  = float(abs(z_score) / np.sqrt(n_total))

        # ROC-AUC with bootstrap CI
        auc, ci_lo, ci_hi = bootstrap_auc_ci(
            projections, labels, args.n_bootstrap, seed=RANDOM_SEED
        )

        result = {
            "layer":               layer,
            "n_total":             int(n_total),
            "n_jailbroken":        int(labels.sum()),
            "n_not_jailbroken":    int((1 - labels).sum()),
            "mean_jailbroken":     float(jb_proj.mean()),
            "mean_not_jailbroken": float(not_jb_proj.mean()),
            "std_jailbroken":      float(jb_proj.std()),
            "std_not_jailbroken":  float(not_jb_proj.std()),
            "median_jailbroken":   float(np.median(jb_proj)),
            "median_not_jailbroken": float(np.median(not_jb_proj)),
            "mannwhitney_p":       float(p_val),
            "effect_size_r":       effect_r,
            "auc":                 auc,
            "auc_ci_lo":           ci_lo,
            "auc_ci_hi":           ci_hi,
        }

        all_results[f"layer_{layer}"] = result
        print_layer_results(layer, result)

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = output_dir / "assistant_axis_predictor.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
