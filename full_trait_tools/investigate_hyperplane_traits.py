#!/usr/bin/env python3
"""
investigate_hyperplane_traits.py

Two analyses:

1. STABILITY TEST
   Retrain the raw-activation logistic regression with N different seeds.
   Compute pairwise cosine similarities between the resulting hyperplane
   normals. If they cluster near 1.0, w is a stable real direction.
   If they cluster near 0.0, the classifier is fitting noise despite
   high AUC (many equivalent hyperplanes in high-dimensional space).

2. POINT-BISERIAL CORRELATION
   For each trait vector (and the assistant axis), compute the projection
   of every pre-generation activation onto that trait direction. Then
   correlate those scalar projection values with the jailbreak label (0/1)
   using point-biserial correlation.

   This directly answers: when the model's pre-generation state is high
   on a given trait direction, does jailbreak success go up or down?

   This bypasses the high-dimensionality problem of cosine similarity
   between unit vectors. We're correlating scalars, not comparing vectors.

   Bootstrap confidence intervals (n=1000) determine which correlations
   are reliably non-zero.

Usage:
  uv run full_trait_tools/investigate_hyperplane_traits.py

  # Skip stability test if already confident w is stable
  uv run full_trait_tools/investigate_hyperplane_traits.py --skip_stability

  # Run stability test only
  uv run full_trait_tools/investigate_hyperplane_traits.py --skip_correlation
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

RANDOM_SEED      = 42
TRAIN_PAIR_FRAC  = 0.7
MIN_SUCCESS_RATE = 0.20
MAX_SUCCESS_RATE = 0.80
LAYERS           = [16, 28]
N_STABILITY_SEEDS = 8
N_BOOTSTRAP       = 1000
N_TOP             = 20


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


def load_trait_vectors(
    vectors_dir: Path, layer: int
) -> Tuple[List[str], np.ndarray]:
    pt_files = sorted(vectors_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {vectors_dir}")
    trait_names, vectors = [], []
    for pt_file in pt_files:
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        trait_names.append(pt_file.stem)
        vectors.append(data["vector"][layer].float().numpy())
    return trait_names, np.stack(vectors)


def load_axis_vector(axis_path: Path, layer: int) -> np.ndarray:
    data = torch.load(axis_path, map_location="cpu", weights_only=False)
    return data["axis"][layer].float().numpy()


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
    return [r for r in rows if r["behavior_id"] in kept], sorted(kept)


def split_pools(rows, train_frac, seed):
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


def get_pairs(rows, behavior_pool, template_pool):
    return [
        r for r in rows
        if r["behavior_id"] in behavior_pool and r["jailbreak_idx"] in template_pool
    ]


# ── Feature extraction ─────────────────────────────────────────────────────────

def get_raw_activations(
    rows: List[dict],
    activations: Dict[int, Dict[str, torch.Tensor]],
    layer: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns X [n, 4096] and y [n] for all rows that have activations."""
    layer_key = str(layer)
    X_list, y_list = [], []
    for row in rows:
        pid = row["pair_id"]
        if pid not in activations or layer_key not in activations[pid]:
            continue
        X_list.append(activations[pid][layer_key].float().numpy())
        y_list.append(int(row["jailbroken"]))
    if not X_list:
        return np.array([]), np.array([])
    return np.stack(X_list), np.array(y_list)


# ── Hyperplane normal ──────────────────────────────────────────────────────────

def learn_hyperplane_normal(X_train, y_train, seed):
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000,
        random_state=seed, class_weight="balanced",
    )
    clf.fit(X_train_s, y_train)
    w_scaled   = clf.coef_[0]
    w_original = w_scaled / (scaler.scale_ + 1e-12)
    w_unit     = w_original / (np.linalg.norm(w_original) + 1e-12)
    return w_unit


def cosine_sim(a, b):
    return float(np.dot(a, b) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ── Analysis 1: Stability test ─────────────────────────────────────────────────

def run_stability_test(
    rows_filtered: List[dict],
    activations: Dict[int, Dict[str, torch.Tensor]],
    layer: int,
    train_frac: float,
    n_seeds: int,
) -> dict:
    """
    Train the classifier with n_seeds different random seeds.
    Compute pairwise cosine similarities between the resulting hyperplane normals.
    """
    print(f"\n  Training {n_seeds} classifiers with different seeds...")
    normals = []

    for seed in tqdm(range(n_seeds), desc=f"  Stability[layer {layer}]"):
        train_behaviors, test_behaviors, train_templates, test_templates = \
            split_pools(rows_filtered, train_frac, seed)

        train_rows = get_pairs(rows_filtered, train_behaviors, train_templates)
        X_train, y_train = get_raw_activations(train_rows, activations, layer)

        if len(X_train) < 10:
            continue

        w = learn_hyperplane_normal(X_train, y_train, seed)
        normals.append((seed, w))

    n = len(normals)
    cos_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_sims.append(cosine_sim(normals[i][1], normals[j][1]))

    cos_sims = np.array(cos_sims)

    result = {
        "n_seeds":       n,
        "n_pairs":       len(cos_sims),
        "mean_cos_sim":  float(cos_sims.mean()),
        "std_cos_sim":   float(cos_sims.std()),
        "min_cos_sim":   float(cos_sims.min()),
        "max_cos_sim":   float(cos_sims.max()),
        "mean_angle_deg": float(np.degrees(np.arccos(np.clip(cos_sims.mean(), -1, 1)))),
        "verdict": (
            "STABLE — w is a real direction" if cos_sims.mean() > 0.5
            else "MODERATE — some consistency but noisy" if cos_sims.mean() > 0.1
            else "UNSTABLE — w is likely noise"
        ),
    }
    return result, cos_sims


def print_stability_results(layer: int, result: dict, cos_sims: np.ndarray) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  LAYER {layer} — STABILITY TEST")
    print(sep)
    print(f"  Seeds tested      : {result['n_seeds']}")
    print(f"  Pairs compared    : {result['n_pairs']}")
    print(f"  Mean cos_sim      : {result['mean_cos_sim']:.4f}")
    print(f"  Std  cos_sim      : {result['std_cos_sim']:.4f}")
    print(f"  Min  cos_sim      : {result['min_cos_sim']:.4f}")
    print(f"  Max  cos_sim      : {result['max_cos_sim']:.4f}")
    print(f"  Mean angle        : {result['mean_angle_deg']:.2f}°")
    print(f"\n  Verdict: {result['verdict']}")
    print()
    print(f"  Interpretation:")
    print(f"    cos_sim ~1.0  → all normals point same direction → stable")
    print(f"    cos_sim ~0.0  → normals are random → fitting noise")
    print(f"    cos_sim ~-1.0 → sign flip only → still stable (sign ambiguity)")
    print(sep)


# ── Analysis 2: Point-biserial correlation ─────────────────────────────────────

def point_biserial_correlation(projections: np.ndarray, labels: np.ndarray) -> float:
    """
    Point-biserial correlation between continuous projections and binary labels.
    Equivalent to Pearson r between a continuous and a binary variable.
    """
    r, p = stats.pointbiserialr(labels, projections)
    return float(r)


def bootstrap_ci(
    projections: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int,
    seed: int,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for point-biserial correlation."""
    rng = np.random.default_rng(seed)
    n   = len(projections)
    boot_rs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        r = point_biserial_correlation(projections[idx], labels[idx])
        boot_rs.append(r)
    boot_rs = np.array(boot_rs)
    lo = np.percentile(boot_rs, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_rs, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def run_correlation_analysis(
    rows_filtered: List[dict],
    activations: Dict[int, Dict[str, torch.Tensor]],
    trait_names: List[str],
    trait_vectors: np.ndarray,
    axis_vector: np.ndarray,
    layer: int,
    n_bootstrap: int,
) -> List[dict]:
    """
    For every trait vector + axis, compute:
      - projection of each activation onto the trait direction
      - point-biserial correlation with jailbreak label
      - bootstrap 95% CI
    """
    layer_key = str(layer)

    # Collect all activations and labels
    acts, labels = [], []
    for row in rows_filtered:
        pid = row["pair_id"]
        if pid not in activations or layer_key not in activations[pid]:
            continue
        acts.append(activations[pid][layer_key].float().numpy())
        labels.append(int(row["jailbroken"]))

    if not acts:
        return []

    X      = np.stack(acts)    # [n, 4096]
    y      = np.array(labels)  # [n]

    print(f"  {len(y)} samples | {y.sum()} jailbroken | {(1-y).sum()} not")

    # All vectors: traits + axis
    all_names   = trait_names + ["assistant_axis"]
    all_vectors = np.vstack([trait_vectors, axis_vector[np.newaxis, :]])

    # Project all at once: [n, n_traits+1]
    projections = X @ all_vectors.T

    results = []
    for i, name in enumerate(tqdm(all_names, desc=f"  Correlating[layer {layer}]")):
        proj = projections[:, i]
        r    = point_biserial_correlation(proj, y)
        lo, hi = bootstrap_ci(proj, y, n_bootstrap, seed=i)

        results.append({
            "trait":       name,
            "correlation": r,
            "ci_lo":       lo,
            "ci_hi":       hi,
            "significant": lo > 0 or hi < 0,  # CI doesn't cross zero
            "direction":   (
                "→ jailbreak more likely" if r > 0
                else "→ jailbreak less likely"
            ),
            "abs_r":       abs(r),
        })

    return results


def print_correlation_results(
    layer: int,
    results: List[dict],
    n_top: int,
) -> None:
    sep = "=" * 75
    print(f"\n{sep}")
    print(f"  LAYER {layer} — POINT-BISERIAL CORRELATION: Trait Projections vs Jailbreak")
    print(sep)
    print(f"  r > 0: high projection on this trait → jailbreak more likely")
    print(f"  r < 0: high projection on this trait → jailbreak less likely")
    print(f"  *: 95% bootstrap CI does not cross zero (reliable signal)\n")

    header  = (f"  {'Rank':>4}  {'Trait':40s}  {'r':>7}  "
               f"{'CI_lo':>7}  {'CI_hi':>7}  {'Sig':>4}  Direction")
    divider = "  " + "─" * 85

    # Sort by correlation descending (most positive)
    pos_sorted = sorted(results, key=lambda x: x["correlation"], reverse=True)
    neg_sorted = sorted(results, key=lambda x: x["correlation"])
    abs_sorted = sorted(results, key=lambda x: x["abs_r"], reverse=True)

    print(f"  ── TOP {n_top} most positively correlated with jailbreak success ──")
    print(header)
    print(divider)
    for rank, entry in enumerate(pos_sorted[:n_top], 1):
        sig = " *" if entry["significant"] else "  "
        print(f"  {rank:>4}  {entry['trait']:40s}  "
              f"{entry['correlation']:>7.4f}  "
              f"{entry['ci_lo']:>7.4f}  {entry['ci_hi']:>7.4f}  "
              f"{sig:>4}  {entry['direction']}")

    print(f"\n  ── TOP {n_top} most negatively correlated (trait protects against jailbreak) ──")
    print(header)
    print(divider)
    for rank, entry in enumerate(neg_sorted[:n_top], 1):
        sig = " *" if entry["significant"] else "  "
        print(f"  {rank:>4}  {entry['trait']:40s}  "
              f"{entry['correlation']:>7.4f}  "
              f"{entry['ci_lo']:>7.4f}  {entry['ci_hi']:>7.4f}  "
              f"{sig:>4}  {entry['direction']}")

    print(f"\n  ── TOP {n_top} by absolute correlation (most predictive regardless of sign) ──")
    print(header)
    print(divider)
    for rank, entry in enumerate(abs_sorted[:n_top], 1):
        sig = " *" if entry["significant"] else "  "
        print(f"  {rank:>4}  {entry['trait']:40s}  "
              f"{entry['correlation']:>7.4f}  "
              f"{entry['ci_lo']:>7.4f}  {entry['ci_hi']:>7.4f}  "
              f"{sig:>4}  {entry['direction']}")

    # Always print assistant axis
    axis_entry = next(r for r in results if r["trait"] == "assistant_axis")
    axis_pos_rank = next(i+1 for i, r in enumerate(pos_sorted)
                         if r["trait"] == "assistant_axis")
    axis_abs_rank = next(i+1 for i, r in enumerate(abs_sorted)
                         if r["trait"] == "assistant_axis")
    sig = "*" if axis_entry["significant"] else "not significant"

    print(f"\n  ── Assistant Axis ──")
    print(f"  r           : {axis_entry['correlation']:.4f}")
    print(f"  95% CI      : [{axis_entry['ci_lo']:.4f}, {axis_entry['ci_hi']:.4f}]")
    print(f"  Significant : {sig}")
    print(f"  Direction   : {axis_entry['direction']}")
    print(f"  Rank (pos)  : {axis_pos_rank} of {len(results)}")
    print(f"  Rank (abs)  : {axis_abs_rank} of {len(results)}")

    n_sig = sum(1 for r in results if r["significant"])
    print(f"\n  {n_sig} of {len(results)} traits have significant correlations (CI ≠ 0)")
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
    parser.add_argument("--n_stability_seeds", type=int, default=N_STABILITY_SEEDS)
    parser.add_argument("--n_bootstrap",       type=int, default=N_BOOTSTRAP)
    parser.add_argument("--n_top",             type=int, default=N_TOP)
    parser.add_argument("--min_success_rate",  type=float, default=MIN_SUCCESS_RATE)
    parser.add_argument("--max_success_rate",  type=float, default=MAX_SUCCESS_RATE)
    parser.add_argument("--train_frac",        type=float, default=TRAIN_PAIR_FRAC)
    parser.add_argument("--skip_stability",    action="store_true")
    parser.add_argument("--skip_correlation",  action="store_true")
    parser.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and filter data ───────────────────────────────────────────────────
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
        print(f"\n{'#' * 70}")
        print(f"  LAYER {layer}")
        print(f"{'#' * 70}")

        layer_results = {}

        # ── 1. Stability test ──────────────────────────────────────────────────
        if not args.skip_stability:
            result, cos_sims = run_stability_test(
                rows_filtered, activations, layer,
                args.train_frac, args.n_stability_seeds,
            )
            print_stability_results(layer, result, cos_sims)
            layer_results["stability"] = result
        else:
            print("  [Skipping stability test]")

        # ── 2. Correlation analysis ────────────────────────────────────────────
        if not args.skip_correlation:
            print(f"\n  Loading trait vectors for layer {layer}...")
            trait_names, trait_vectors = load_trait_vectors(
                Path(args.trait_vectors_dir), layer
            )
            axis_vector = load_axis_vector(Path(args.axis_path), layer)
            print(f"  {len(trait_names)} traits + assistant axis")
            print(f"  Computing point-biserial correlations "
                  f"(bootstrap n={args.n_bootstrap})...")

            corr_results = run_correlation_analysis(
                rows_filtered, activations,
                trait_names, trait_vectors, axis_vector,
                layer, args.n_bootstrap,
            )
            print_correlation_results(layer, corr_results, args.n_top)
            layer_results["correlation"] = corr_results
        else:
            print("  [Skipping correlation analysis]")

        all_results[f"layer_{layer}"] = layer_results

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = output_dir / "hyperplane_investigation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
