#!/usr/bin/env python3
"""
stable_hyperplane_analysis.py

Uses the sweet-spot PCA dimensionality found by pca_sweep_stability.py
to produce a stable jailbreak hyperplane normal and compare it to trait
vectors in a space where cosine similarities are geometrically meaningful.

Sweet spots (from sweep):
  Layer 16: n=5  PCs (stability=0.936, AUC=0.802, 46.6% var)
  Layer 28: n=10 PCs (stability=0.755, AUC=0.795, 64.5% var)

Pipeline:
  1. Fit PCA on all activations at the sweet-spot dimensionality
  2. Train logistic regression in that PCA space across N seeds
  3. Average the w vectors to get a stable direction
  4. Project all trait vectors + assistant axis into the same PCA space
  5. Compare cosine similarities — now meaningful in low-dim space
  6. Report top/bottom trait rankings + assistant axis result

Outputs:
  - stable_hyperplane_layer{N}.pt
  - stable_hyperplane_analysis.json
  - printed rankings

Usage:
  uv run full_trait_tools/stable_hyperplane_analysis.py

  # Custom dimensionalities
  uv run full_trait_tools/stable_hyperplane_analysis.py --n_pca 16:5 28:10
"""

import argparse
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
from tqdm import tqdm

RANDOM_SEED      = 42
TRAIN_FRAC       = 0.7
MIN_SUCCESS_RATE = 0.20
MAX_SUCCESS_RATE = 0.80
N_SEEDS          = 10
N_TOP            = 20

# Sweet-spot dimensionalities from pca_sweep_stability.py
DEFAULT_N_PCA = {16: 4, 28: 4}


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


def load_trait_vectors(vectors_dir: Path, layer: int) -> Tuple[List[str], np.ndarray]:
    pt_files = sorted(vectors_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {vectors_dir}")
    trait_names, vectors = [], []
    for pt_file in pt_files:
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        trait_names.append(pt_file.stem)
        vectors.append(data["vector"][layer].float().numpy())
    return trait_names, np.stack(vectors)  # [n_traits, 4096]


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


def get_activations_and_labels(
    rows, activations, layer,
    behavior_pool=None, template_pool=None,
):
    layer_key = str(layer)
    X_list, y_list = [], []
    for row in rows:
        if behavior_pool is not None and row["behavior_id"] not in behavior_pool:
            continue
        if template_pool is not None and row["jailbreak_idx"] not in template_pool:
            continue
        pid = row["pair_id"]
        if pid not in activations or layer_key not in activations[pid]:
            continue
        X_list.append(activations[pid][layer_key].float().numpy())
        y_list.append(int(row["jailbroken"]))
    if not X_list:
        return np.array([]), np.array([])
    return np.stack(X_list), np.array(y_list)


# ── Helpers ────────────────────────────────────────────────────────────────────

def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def angle_deg(cos: float) -> float:
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def pairwise_cosine_stats(vectors: List[np.ndarray]) -> dict:
    n = len(vectors)
    cos_sims = [
        cosine_sim(vectors[i], vectors[j])
        for i in range(n) for j in range(i + 1, n)
    ]
    cos_sims = np.array(cos_sims)
    return {
        "mean":           float(cos_sims.mean()),
        "std":            float(cos_sims.std()),
        "min":            float(cos_sims.min()),
        "max":            float(cos_sims.max()),
        "mean_angle_deg": float(angle_deg(float(cos_sims.mean()))),
    }


# ── Main analysis per layer ────────────────────────────────────────────────────

def run_layer(
    layer: int,
    n_pca: int,
    rows_filtered: List[dict],
    activations: Dict,
    trait_names: List[str],
    trait_vectors: np.ndarray,   # [n_traits, 4096]
    axis_vector: np.ndarray,     # [4096]
    n_seeds: int,
    train_frac: float,
    n_top: int,
    output_dir: Path,
) -> dict:

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  LAYER {layer}  |  n_pca={n_pca}")
    print(sep)

    # ── Get all activations ────────────────────────────────────────────────────
    X_all, y_all = get_activations_and_labels(rows_filtered, activations, layer)
    print(f"\n  {X_all.shape[0]} samples, {X_all.shape[1]} dimensions")

    # ── Fit scaler + PCA on all data ───────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
    pca.fit(X_scaled)
    var_explained = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA {n_pca} components: {100*var_explained:.1f}% variance explained")

    # ── Train N classifiers, collect w vectors in PCA space ───────────────────
    print(f"\n  Training {n_seeds} classifiers across seeds...")
    ws_pca   = []   # [n_seeds, n_pca]
    aucs     = []

    for seed in tqdm(range(n_seeds), desc=f"  Seeds[layer {layer}]"):
        train_beh, test_beh, train_tpl, test_tpl = split_pools(
            rows_filtered, train_frac, seed
        )
        X_train, y_train = get_activations_and_labels(
            rows_filtered, activations, layer, train_beh, train_tpl
        )
        X_test, y_test = get_activations_and_labels(
            rows_filtered, activations, layer, test_beh, test_tpl
        )
        if len(X_train) < 20 or len(X_test) < 5:
            continue

        X_train_pca = pca.transform(scaler.transform(X_train))
        X_test_pca  = pca.transform(scaler.transform(X_test))

        clf = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000,
            random_state=seed, class_weight="balanced",
        )
        clf.fit(X_train_pca, y_train)
        ws_pca.append(unit(clf.coef_[0]))

        try:
            aucs.append(float(roc_auc_score(
                y_test, clf.predict_proba(X_test_pca)[:, 1]
            )))
        except Exception:
            pass

    # ── Stability ──────────────────────────────────────────────────────────────
    stats = pairwise_cosine_stats(ws_pca)
    print(f"\n  Stability across {len(ws_pca)} seeds:")
    print(f"    Mean cos_sim : {stats['mean']:.4f}")
    print(f"    Std  cos_sim : {stats['std']:.4f}")
    print(f"    Min  cos_sim : {stats['min']:.4f}")
    print(f"    Max  cos_sim : {stats['max']:.4f}")
    print(f"    Mean angle   : {stats['mean_angle_deg']:.2f}°")
    print(f"    Mean AUC     : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # ── Stable averaged w ──────────────────────────────────────────────────────
    # Average in PCA space then unit normalise
    w_avg_pca  = unit(np.stack(ws_pca).mean(axis=0))   # [n_pca]
    # Map back to 4096-dim for saving
    w_avg_4096 = unit(w_avg_pca @ pca.components_)      # [4096]

    # ── Project trait vectors into PCA space ───────────────────────────────────
    # Each trait vector t [4096] → PCA space: pca.components_ @ t → [n_pca]
    # Note: pca.components_ is [n_pca, 4096]
    print(f"\n  Projecting {len(trait_names)} trait vectors + axis into {n_pca}-dim PCA space...")

    all_names    = trait_names + ["assistant_axis"]
    all_vecs_4096 = np.vstack([trait_vectors, axis_vector[np.newaxis, :]])  # [n+1, 4096]

    # Centre using PCA mean before projecting
    # pca.transform does: (X - mean) @ components_.T
    # For a single vector v: (v - mean) @ components_.T
    all_vecs_centred = all_vecs_4096 - pca.mean_       # [n+1, 4096]
    all_vecs_pca = all_vecs_centred @ pca.components_.T  # [n+1, n_pca]

    # Unit normalise each projected vector
    norms = np.linalg.norm(all_vecs_pca, axis=1, keepdims=True)
    all_vecs_pca_unit = all_vecs_pca / (norms + 1e-12)  # [n+1, n_pca]

    # Cosine similarities with stable w
    cos_sims = all_vecs_pca_unit @ w_avg_pca            # [n+1]

    # Build results list
    results = []
    for i, name in enumerate(all_names):
        cos = float(cos_sims[i])
        results.append({
            "trait":       name,
            "cos_sim_pca": cos,
            "angle_deg":   angle_deg(cos),
            "abs_cos":     abs(cos),
            "direction":   "→ jailbreak more likely" if cos > 0
                           else "→ jailbreak less likely",
        })

    # Add ranks
    pos_sorted  = sorted(results, key=lambda x: x["cos_sim_pca"], reverse=True)
    neg_sorted  = sorted(results, key=lambda x: x["cos_sim_pca"])
    abs_sorted  = sorted(results, key=lambda x: x["abs_cos"], reverse=True)
    for rank, e in enumerate(pos_sorted,  1): e["rank_pos"] = rank
    for rank, e in enumerate(neg_sorted,  1): e["rank_neg"] = rank
    for rank, e in enumerate(abs_sorted,  1): e["rank_abs"] = rank

    # ── Print rankings ─────────────────────────────────────────────────────────
    header  = (f"  {'Rank':>4}  {'Trait':40s}  "
               f"{'cos_sim':>8}  {'angle':>8}  Direction")
    divider = "  " + "─" * 82

    print(f"\n{sep}")
    print(f"  LAYER {layer} — Trait Cosine Similarities in {n_pca}-dim PCA Space")
    print(f"  (Stable w averaged over {len(ws_pca)} seeds)")
    print(sep)

    print(f"\n  ── TOP {n_top} most aligned with jailbreak direction ──")
    print(header)
    print(divider)
    for entry in pos_sorted[:n_top]:
        print(f"  {entry['rank_pos']:>4}  {entry['trait']:40s}  "
              f"{entry['cos_sim_pca']:>8.4f}  {entry['angle_deg']:>7.2f}°  "
              f"{entry['direction']}")

    print(f"\n  ── TOP {n_top} most aligned against jailbreak direction ──")
    print(header)
    print(divider)
    for entry in neg_sorted[:n_top]:
        print(f"  {entry['rank_neg']:>4}  {entry['trait']:40s}  "
              f"{entry['cos_sim_pca']:>8.4f}  {entry['angle_deg']:>7.2f}°  "
              f"{entry['direction']}")

    print(f"\n  ── TOP {n_top} by |cos_sim| (most predictive regardless of sign) ──")
    print(header)
    print(divider)
    for entry in abs_sorted[:n_top]:
        print(f"  {entry['rank_abs']:>4}  {entry['trait']:40s}  "
              f"{entry['cos_sim_pca']:>8.4f}  {entry['angle_deg']:>7.2f}°  "
              f"{entry['direction']}")

    # ── Assistant axis ─────────────────────────────────────────────────────────
    axis_entry = next(r for r in results if r["trait"] == "assistant_axis")
    print(f"\n  ── Assistant Axis ──")
    print(f"  cos_sim ({n_pca}-dim PCA) : {axis_entry['cos_sim_pca']:.4f}")
    print(f"  angle                  : {axis_entry['angle_deg']:.2f}°")
    print(f"  direction              : {axis_entry['direction']}")
    print(f"  rank (pos)             : {axis_entry['rank_pos']} of {len(results)}")
    print(f"  rank (abs)             : {axis_entry['rank_abs']} of {len(results)}")

    # ── How much of w_avg_pca is explained by trait vocabulary ────────────────
    # Each trait projection squared = variance of w along that direction
    # Sum of squared cosines = total fraction explained (if traits orthogonal)
    sum_cos_sq = float(np.sum(cos_sims ** 2))
    top_cos_sq = float(np.sum(np.sort(cos_sims ** 2)[::-1][:n_top]))
    print(f"\n  Sum of squared cosines (all {len(results)} traits)  : {sum_cos_sq:.4f}")
    print(f"  Sum of squared cosines (top {n_top})               : {top_cos_sq:.4f}")
    print(f"  (Upper bound on variance of w explained by trait vocabulary)")

    # ── Save ───────────────────────────────────────────────────────────────────
    out_pt = output_dir / f"stable_hyperplane_layer{layer}.pt"
    torch.save({
        "vector":           torch.from_numpy(w_avg_4096).float(),
        "vector_pca":       torch.from_numpy(w_avg_pca).float(),
        "layer":            layer,
        "n_pca_components": n_pca,
        "n_seeds":          len(ws_pca),
        "var_explained":    var_explained,
        "stability":        stats,
        "mean_auc":         float(np.mean(aucs)),
        "description": (
            f"Stable hyperplane normal: averaged over {len(ws_pca)} seeds "
            f"in {n_pca}-dim PCA space (sweet spot from pca_sweep). "
            f"Positive direction = jailbreak success."
        ),
    }, out_pt)
    print(f"\n  Saved to {out_pt.name}")

    return {
        "layer":            layer,
        "n_pca":            n_pca,
        "var_explained":    var_explained,
        "stability":        stats,
        "mean_auc":         float(np.mean(aucs)),
        "std_auc":          float(np.std(aucs)),
        "n_seeds":          len(ws_pca),
        "cosine_ranking":   results,
        "assistant_axis":   axis_entry,
        "sum_cos_sq_all":   sum_cos_sq,
        "sum_cos_sq_top20": top_cos_sq,
        "top_positive":     pos_sorted[:n_top],
        "top_negative":     neg_sorted[:n_top],
        "top_abs":          abs_sorted[:n_top],
    }


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
    parser.add_argument(
        "--n_pca", nargs="+", type=str, default=None,
        help="Per-layer PCA components as 'layer:n' pairs e.g. '16:5 28:10'. "
             "Defaults to sweet-spot values from pca_sweep.",
    )
    parser.add_argument("--n_seeds",          type=int,   default=N_SEEDS)
    parser.add_argument("--n_top",            type=int,   default=N_TOP)
    parser.add_argument("--min_success_rate", type=float, default=MIN_SUCCESS_RATE)
    parser.add_argument("--max_success_rate", type=float, default=MAX_SUCCESS_RATE)
    parser.add_argument("--train_frac",       type=float, default=TRAIN_FRAC)
    parser.add_argument("--layers", nargs="+", type=int,  default=[16, 28])
    args = parser.parse_args()

    # Parse n_pca overrides
    n_pca_map = dict(DEFAULT_N_PCA)
    if args.n_pca:
        for item in args.n_pca:
            layer_str, n_str = item.split(":")
            n_pca_map[int(layer_str)] = int(n_str)

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
        n_pca = n_pca_map.get(layer, 10)
        print(f"\n  Using n_pca={n_pca} for layer {layer} "
              f"(from {'sweep sweet spot' if layer in DEFAULT_N_PCA else 'default'})")

        trait_names, trait_vectors = load_trait_vectors(
            Path(args.trait_vectors_dir), layer
        )
        axis_vector = load_axis_vector(Path(args.axis_path), layer)

        result = run_layer(
            layer=layer,
            n_pca=n_pca,
            rows_filtered=rows_filtered,
            activations=activations,
            trait_names=trait_names,
            trait_vectors=trait_vectors,
            axis_vector=axis_vector,
            n_seeds=args.n_seeds,
            train_frac=args.train_frac,
            n_top=args.n_top,
            output_dir=output_dir,
        )
        all_results[f"layer_{layer}"] = result

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = output_dir / "stable_hyperplane_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
