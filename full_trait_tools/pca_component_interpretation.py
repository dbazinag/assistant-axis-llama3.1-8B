#!/usr/bin/env python3
"""
pca_component_interpretation.py

Interprets the top PCA components of the jailbreak activation space
by comparing each component to all trait vectors and the assistant axis.

The 4 PCA components are directions in 4096-dim activation space that
capture the most variance in the harmbench pre-generation activations.
This script asks: what do those directions mean in terms of persona traits?

For each PC:
  - Computes cosine similarity with all 229 trait vectors + assistant axis
  - Prints top 10 most aligned traits (positive and negative)
  - Shows where the assistant axis falls

This tells you what the "jailbreak-relevant" dimensions of the activation
space actually represent in terms of your persona vocabulary, giving a
more interpretable picture than comparing w directly.

Also reports:
  - How much variance each PC explains
  - Whether the assistant axis has meaningful alignment with any PC
  - A summary "persona profile" of each PC

Usage:
  uv run full_trait_tools/pca_component_interpretation.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RANDOM_SEED      = 42
MIN_SUCCESS_RATE = 0.20
MAX_SUCCESS_RATE = 0.80
LAYERS           = [16, 28]
N_TOP            = 10

# Sweet-spot n_pca from pca_sweep_stability.py
DEFAULT_N_PCA = {16: 5, 28: 10}


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


def load_trait_vectors(vectors_dir: Path, layer: int) -> Tuple[List[str], np.ndarray]:
    pt_files = sorted(vectors_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {vectors_dir}")
    trait_names, vectors = [], []
    for pt_file in pt_files:
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        trait_names.append(pt_file.stem)
        vectors.append(data["vector"][layer].float().numpy())
    return trait_names, np.stack(vectors)


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


def get_all_activations(rows, activations, layer):
    layer_key = str(layer)
    X_list = []
    for row in rows:
        pid = row["pair_id"]
        if pid not in activations or layer_key not in activations[pid]:
            continue
        X_list.append(activations[pid][layer_key].float().numpy())
    return np.stack(X_list) if X_list else np.array([])


# ── Helpers ────────────────────────────────────────────────────────────────────

def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def angle_deg(cos: float) -> float:
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# ── Main analysis ──────────────────────────────────────────────────────────────

def interpret_pcs(
    layer: int,
    n_pca: int,
    rows_filtered: List[dict],
    activations: Dict,
    trait_names: List[str],
    trait_vectors: np.ndarray,
    axis_vector: np.ndarray,
    n_top: int,
    output_dir: Path,
) -> dict:

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  LAYER {layer}  |  n_pca={n_pca}")
    print(sep)

    # ── Fit PCA on all activations ─────────────────────────────────────────────
    X_all = get_all_activations(rows_filtered, activations, layer)
    print(f"\n  {X_all.shape[0]} samples, fitting PCA ({n_pca} components)...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
    pca.fit(X_scaled)

    var_ratios = pca.explained_variance_ratio_
    print(f"  Variance explained per PC:")
    for i, v in enumerate(var_ratios):
        print(f"    PC{i+1}: {100*v:.1f}%")
    print(f"  Total: {100*var_ratios.sum():.1f}%")

    # ── Build all vectors to compare: traits + axis ────────────────────────────
    all_names    = trait_names + ["assistant_axis"]
    all_vecs     = np.vstack([trait_vectors, axis_vector[np.newaxis, :]])  # [n+1, 4096]

    # ── For each PC, compare to all trait vectors ──────────────────────────────
    # pca.components_[i] is the i-th PC direction in 4096-dim space (unit vector)
    all_pc_results = []

    for pc_idx in range(n_pca):
        pc_vec = pca.components_[pc_idx]  # [4096], already unit norm from sklearn

        # Cosine similarity with each trait + axis
        cos_sims = []
        for i, (name, vec) in enumerate(zip(all_names, all_vecs)):
            cos = cosine_sim(pc_vec, vec)
            cos_sims.append({
                "trait":     name,
                "cos_sim":   cos,
                "angle_deg": angle_deg(cos),
                "abs_cos":   abs(cos),
                "direction": "positive" if cos > 0 else "negative",
            })

        pos_sorted = sorted(cos_sims, key=lambda x: x["cos_sim"], reverse=True)
        neg_sorted = sorted(cos_sims, key=lambda x: x["cos_sim"])
        abs_sorted = sorted(cos_sims, key=lambda x: x["abs_cos"], reverse=True)

        axis_result = next(r for r in cos_sims if r["trait"] == "assistant_axis")
        axis_abs_rank = next(i+1 for i, r in enumerate(abs_sorted)
                             if r["trait"] == "assistant_axis")

        # Print this PC
        print(f"\n{'─'*70}")
        print(f"  PC{pc_idx+1}  |  Variance explained: {100*var_ratios[pc_idx]:.1f}%")
        print(f"{'─'*70}")

        header  = f"  {'Rank':>4}  {'Trait':40s}  {'cos_sim':>8}  {'angle':>8}"
        divider = "  " + "─" * 65

        print(f"\n  Top {n_top} traits aligned with PC{pc_idx+1} (positive direction):")
        print(header)
        print(divider)
        for rank, entry in enumerate(pos_sorted[:n_top], 1):
            print(f"  {rank:>4}  {entry['trait']:40s}  "
                  f"{entry['cos_sim']:>8.4f}  {entry['angle_deg']:>7.2f}°")

        print(f"\n  Top {n_top} traits aligned against PC{pc_idx+1} (negative direction):")
        print(header)
        print(divider)
        for rank, entry in enumerate(neg_sorted[:n_top], 1):
            print(f"  {rank:>4}  {entry['trait']:40s}  "
                  f"{entry['cos_sim']:>8.4f}  {entry['angle_deg']:>7.2f}°")

        print(f"\n  Assistant axis:")
        print(f"    cos_sim  : {axis_result['cos_sim']:.4f}")
        print(f"    angle    : {axis_result['angle_deg']:.2f}°")
        print(f"    abs rank : {axis_abs_rank} of {len(cos_sims)}")

        # Summary label for this PC
        top_pos = pos_sorted[0]["trait"]
        top_neg = neg_sorted[0]["trait"]
        print(f"\n  PC{pc_idx+1} summary: '{top_pos}' ↔ '{top_neg}'")

        all_pc_results.append({
            "pc_index":         pc_idx + 1,
            "var_explained":    float(var_ratios[pc_idx]),
            "top_positive":     pos_sorted[:n_top],
            "top_negative":     neg_sorted[:n_top],
            "top_abs":          abs_sorted[:n_top],
            "assistant_axis":   axis_result,
            "axis_abs_rank":    axis_abs_rank,
            "summary":          f"PC{pc_idx+1}: '{top_pos}' ↔ '{top_neg}'",
        })

    # ── Summary across all PCs ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  LAYER {layer} — PC SUMMARY")
    print(sep)
    print(f"\n  Each PC's persona profile (top positive ↔ top negative trait):")
    for r in all_pc_results:
        print(f"    {r['summary']}  ({100*r['var_explained']:.1f}% var)  "
              f"| assistant axis: cos={r['assistant_axis']['cos_sim']:.3f}, "
              f"abs_rank={r['axis_abs_rank']}/{len(all_names)}")

    # Save
    out_path = output_dir / f"pca_interpretation_layer{layer}.json"
    result = {
        "layer":          layer,
        "n_pca":          n_pca,
        "var_per_pc":     [float(v) for v in var_ratios],
        "total_var":      float(var_ratios.sum()),
        "pc_results":     all_pc_results,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {out_path.name}")

    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interpret PCA components of jailbreak activations via trait vectors"
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
    parser.add_argument(
        "--n_pca", nargs="+", type=str, default=None,
        help="Per-layer PCA dims as 'layer:n' e.g. '16:5 28:10'",
    )
    parser.add_argument("--n_top",            type=int,   default=N_TOP)
    parser.add_argument("--min_success_rate", type=float, default=MIN_SUCCESS_RATE)
    parser.add_argument("--max_success_rate", type=float, default=MAX_SUCCESS_RATE)
    parser.add_argument("--layers", nargs="+", type=int,  default=LAYERS)
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
        n_pca = n_pca_map.get(layer, 5)
        trait_names, trait_vectors = load_trait_vectors(
            Path(args.trait_vectors_dir), layer
        )
        axis_vector = load_axis(Path(args.axis_path), layer)

        result = interpret_pcs(
            layer=layer,
            n_pca=n_pca,
            rows_filtered=rows_filtered,
            activations=activations,
            trait_names=trait_names,
            trait_vectors=trait_vectors,
            axis_vector=axis_vector,
            n_top=args.n_top,
            output_dir=output_dir,
        )
        all_results[f"layer_{layer}"] = result

    # Combined output
    out_path = output_dir / "pca_interpretation_all_layers.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {out_path}")


if __name__ == "__main__":
    main()
