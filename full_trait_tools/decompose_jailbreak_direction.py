#!/usr/bin/env python3
"""
decompose_jailbreak_direction.py

Full two-level decomposition of the jailbreak direction:

  Level 1: w = c1*PC1 + c2*PC2 + ... + cn*PCn
    - Coefficient ci = how much PC_i contributes to the jailbreak boundary
    - ci^2 = fraction of w explained by PC_i (sums to 1.0)
    - Standalone AUC = how predictive PC_i is alone (logistic regression)

  Level 2: PCi ~ weighted sum of trait vectors
    - Cosine similarity between PC_i and each trait vector
    - Shows what each jailbreak-relevant PC "means" in persona space

Together gives the full chain:
  trait vectors -> PCA components -> jailbreak direction

Usage:
  uv run full_trait_tools/decompose_jailbreak_direction.py
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

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
N_TOP_TRAITS     = 10  # traits to show per PC
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


def load_trait_vectors(vectors_dir: Path, layer: int) -> Tuple[List[str], np.ndarray]:
    pt_files = sorted(vectors_dir.glob("*.pt"))
    trait_names, vectors = [], []
    for pt_file in pt_files:
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        trait_names.append(pt_file.stem)
        vectors.append(data["vector"][layer].float().numpy())
    return trait_names, np.stack(vectors)


def load_axis(axis_path: Path, layer: int) -> np.ndarray:
    data = torch.load(axis_path, map_location="cpu", weights_only=False)
    return data["axis"][layer].float().numpy()


def load_stable_hyperplane(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (w_4096, w_pca)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return (
        data["vector"].float().numpy(),
        data["vector_pca"].float().numpy(),
    )


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


# ── Standalone AUC per PC ──────────────────────────────────────────────────────

def standalone_auc_per_pc(
    rows_filtered: List[dict],
    activations: Dict,
    layer: int,
    pca: PCA,
    scaler: StandardScaler,
    n_pca: int,
    n_seeds: int,
    train_frac: float,
) -> List[float]:
    """
    For each PC, train logistic regression using only that PC's projection
    as the feature. Returns mean AUC across seeds for each PC.
    """
    auc_per_pc = [[] for _ in range(n_pca)]

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

        X_train_pca = pca.transform(scaler.transform(X_train_raw))
        X_test_pca  = pca.transform(scaler.transform(X_test_raw))

        for pc_idx in range(n_pca):
            X_tr = X_train_pca[:, pc_idx:pc_idx+1]
            X_te = X_test_pca[:,  pc_idx:pc_idx+1]

            clf = LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=500,
                random_state=seed, class_weight="balanced",
            )
            clf.fit(X_tr, y_train)
            try:
                auc = float(roc_auc_score(y_test, clf.predict_proba(X_te)[:, 1]))
            except ValueError:
                auc = float("nan")
            auc_per_pc[pc_idx].append(auc)

    return [float(np.nanmean(aucs)) if aucs else float("nan")
            for aucs in auc_per_pc]


# ── Trait alignment per PC ─────────────────────────────────────────────────────

def trait_alignment_per_pc(
    pca: PCA,
    trait_names: List[str],
    trait_vectors: np.ndarray,
    axis_vector: np.ndarray,
    n_pca: int,
    n_top: int,
) -> List[dict]:
    """
    For each PC, compute cosine similarity with all trait vectors + axis.
    Returns list of dicts with top/bottom traits per PC.
    """
    all_names = trait_names + ["assistant_axis"]
    all_vecs  = np.vstack([trait_vectors, axis_vector[np.newaxis, :]])

    results = []
    for pc_idx in range(n_pca):
        pc_vec = pca.components_[pc_idx]  # [4096], unit norm

        cos_sims = []
        for name, vec in zip(all_names, all_vecs):
            norm = np.linalg.norm(vec)
            cos  = float(np.dot(pc_vec, vec) / (norm + 1e-12))
            cos_sims.append((name, cos))

        cos_sims.sort(key=lambda x: x[1], reverse=True)

        results.append({
            "top_positive": cos_sims[:n_top],
            "top_negative": cos_sims[-n_top:][::-1],
            "assistant_axis_cos": next(
                cos for name, cos in cos_sims if name == "assistant_axis"
            ),
            "assistant_axis_rank": next(
                i+1 for i, (name, _) in enumerate(cos_sims)
                if name == "assistant_axis"
            ),
        })

    return results


# ── Printing ───────────────────────────────────────────────────────────────────

def print_decomposition(
    layer: int,
    n_pca: int,
    w_pca: np.ndarray,
    standalone_aucs: List[float],
    trait_alignments: List[dict],
    var_ratios: np.ndarray,
    n_top: int,
) -> None:

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  LAYER {layer} — JAILBREAK DIRECTION DECOMPOSITION ({n_pca} PCs)")
    print(sep)

    print(f"\n  {'PC':>4}  {'Var%':>6}  {'Coef in w':>10}  {'Coef²':>7}  "
          f"{'StandaloneAUC':>14}  {'Persona profile'}")
    print("  " + "─" * 75)

    # Sort by |coef| for display
    order = np.argsort(np.abs(w_pca))[::-1]

    for rank, pc_idx in enumerate(order):
        coef      = float(w_pca[pc_idx])
        coef_sq   = coef ** 2
        auc       = standalone_aucs[pc_idx]
        var_pct   = 100 * var_ratios[pc_idx]
        ta        = trait_alignments[pc_idx]
        top_pos   = ta["top_positive"][0][0]
        top_neg   = ta["top_negative"][0][0]
        direction = "→ jailbreak" if coef > 0 else "→ refusal"

        print(f"  PC{pc_idx+1:>2}  {var_pct:>5.1f}%  {coef:>+10.4f}  {coef_sq:>7.4f}  "
              f"{auc:>14.4f}  {top_pos} ↔ {top_neg}  ({direction})")

    print(f"\n  Sum of coef² = {float(np.sum(w_pca**2)):.4f} (should be 1.0)")

    print(f"\n{sep}")
    print(f"  TOP 10 TRAIT COMPONENTS PER PC  (ranked by |cosine similarity|)")
    print(sep)

    for pc_idx in order:
        coef = float(w_pca[pc_idx])
        ta   = trait_alignments[pc_idx]
        auc  = standalone_aucs[pc_idx]
        var  = 100 * var_ratios[pc_idx]

        direction_str = "→ jailbreak" if coef > 0 else "→ refusal"
        print(f"\n  PC{pc_idx+1}  |  var={var:.1f}%  |  coef in w={coef:+.4f} ({direction_str})"
              f"  |  standalone AUC={auc:.4f}"
              f"  |  assistant_axis cos={ta['assistant_axis_cos']:+.3f} "
              f"(rank {ta['assistant_axis_rank']}/{229+1})")

        # Combine positive and negative, sort by |cos|
        all_traits = ta["top_positive"] + ta["top_negative"]
        all_traits_sorted = sorted(all_traits, key=lambda x: abs(x[1]), reverse=True)[:10]

        print(f"  {'Rank':>4}  {'Trait':40s}  {'cos_sim':>8}  Direction")
        print("  " + "─" * 65)
        for rank, (name, cos) in enumerate(all_traits_sorted, 1):
            direction = "→ jailbreak" if cos > 0 else "→ refusal"
            print(f"  {rank:>4}  {name:40s}  {cos:>+8.4f}  {direction}")

    print(f"\n{sep}")
    print(f"  SUMMARY: What drives jailbreak success at layer {layer}?")
    print(sep)
    top_pc = int(np.argmax(np.abs(w_pca)))
    print(f"\n  The dominant PC is PC{top_pc+1} (coef²={w_pca[top_pc]**2:.4f}, "
          f"{100*var_ratios[top_pc]:.1f}% activation variance)")
    print(f"  It represents: '{trait_alignments[top_pc]['top_positive'][0][0]}' "
          f"↔ '{trait_alignments[top_pc]['top_negative'][0][0]}'")
    print(f"  Standalone AUC: {standalone_aucs[top_pc]:.4f}")


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
        "--hyperplane_dir", type=str,
        default="full_trait_output/harmbench_logreg",
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
    parser.add_argument("--n_top_traits",     type=int,   default=N_TOP_TRAITS)
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

    # ── Load and filter data ───────────────────────────────────────────────────
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
    print(f"  {len(rows_filtered)} pairs after variance filter "
          f"({n_jb} jailbroken, {len(rows_filtered)-n_jb} not)")

    all_results = {}

    for layer in args.layers:
        n_pca = n_pca_map.get(layer, 4)
        print(f"\n{'='*70}")
        print(f"  Processing layer {layer} with n_pca={n_pca}")
        print(f"{'='*70}")

        # ── Load saved hyperplane normal ───────────────────────────────────────
        hp_path = Path(args.hyperplane_dir) / f"stable_hyperplane_layer{layer}.pt"
        if not hp_path.exists():
            print(f"  Skipping layer {layer}: {hp_path} not found")
            print(f"  Run stable_hyperplane_analysis.py --n_pca {layer}:{n_pca} first")
            continue

        w_4096, w_pca = load_stable_hyperplane(hp_path)
        print(f"  Loaded w_pca: shape={w_pca.shape}, norm={np.linalg.norm(w_pca):.4f}")

        # ── Refit PCA (same as stable_hyperplane_analysis.py) ─────────────────
        print(f"  Refitting PCA ({n_pca} components) on all activations...")
        X_all, y_all = get_activations(rows_filtered, activations, layer)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
        pca.fit(X_scaled)
        var_ratios = pca.explained_variance_ratio_
        print(f"  Variance explained: {[f'{100*v:.1f}%' for v in var_ratios]}")

        # ── Load trait vectors ─────────────────────────────────────────────────
        trait_names, trait_vectors = load_trait_vectors(
            Path(args.trait_vectors_dir), layer
        )
        axis_vector = load_axis(Path(args.axis_path), layer)

        # ── Level 1: standalone AUC per PC ────────────────────────────────────
        print(f"  Computing standalone AUC per PC ({args.n_seeds} seeds)...")
        standalone_aucs = standalone_auc_per_pc(
            rows_filtered, activations, layer,
            pca, scaler, n_pca,
            n_seeds=args.n_seeds,
            train_frac=args.train_frac,
        )

        # ── Level 2: trait alignment per PC ───────────────────────────────────
        print(f"  Computing trait alignments per PC...")
        trait_alignments = trait_alignment_per_pc(
            pca, trait_names, trait_vectors, axis_vector,
            n_pca, args.n_top_traits,
        )

        # ── Print full decomposition ───────────────────────────────────────────
        print_decomposition(
            layer=layer,
            n_pca=n_pca,
            w_pca=w_pca,
            standalone_aucs=standalone_aucs,
            trait_alignments=trait_alignments,
            var_ratios=var_ratios,
            n_top=args.n_top_traits,
        )

        # ── Save ───────────────────────────────────────────────────────────────
        result = {
            "layer":           layer,
            "n_pca":           n_pca,
            "var_ratios":      [float(v) for v in var_ratios],
            "w_pca":           w_pca.tolist(),
            "w_coef_squared":  [float(c**2) for c in w_pca],
            "standalone_aucs": standalone_aucs,
            "trait_alignments": [
                {
                    "pc_index":           pc_idx + 1,
                    "coef_in_w":          float(w_pca[pc_idx]),
                    "coef_sq":            float(w_pca[pc_idx]**2),
                    "standalone_auc":     standalone_aucs[pc_idx],
                    "assistant_axis_cos": ta["assistant_axis_cos"],
                    "assistant_axis_rank": ta["assistant_axis_rank"],
                    "top_positive":       ta["top_positive"],
                    "top_negative":       ta["top_negative"],
                }
                for pc_idx, ta in enumerate(trait_alignments)
            ],
        }
        all_results[f"layer_{layer}"] = result

    out_path = output_dir / "jailbreak_direction_decomposition.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
