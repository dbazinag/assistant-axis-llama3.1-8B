#!/usr/bin/env python3
"""
compare_hyperplane_to_personas.py

Interprets the jailbreak decision hyperplane normals (from
classify_jailbreak_raw_activations.py) by comparing them to persona vectors.

Three analyses:

1. COSINE SIMILARITY RANKING
   For each trait vector and the assistant axis, compute cosine similarity
   with the hyperplane normal w. Print top 20 most similar and top 20 most
   dissimilar. Also print angle in degrees (0° = identical, 90° = orthogonal,
   180° = opposite). This shows which persona directions are most aligned
   with the jailbreak decision boundary.

2. ASSISTANT AXIS (always printed)
   Regardless of rank, always report the assistant axis cosine similarity
   and angle since it is the central hypothesis of the project.

3. SUBSPACE DECOMPOSITION
   Project w onto the subspace spanned by all trait vectors using least
   squares. Reports:
   - R² (fraction of w explained by the trait vocabulary)
   - Top contributing traits in the reconstruction (by coefficient magnitude)
   - Residual norm (the part of w unexplained by any trait)
   This answers: do our persona vectors collectively explain the jailbreak
   boundary, and if so, which ones contribute most?

Runs for both layer 16 and layer 28.

Usage:
  uv run full_trait_tools/compare_hyperplane_to_personas.py
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

N_TOP_BOTTOM = 20


# ── Loading ────────────────────────────────────────────────────────────────────

def load_hyperplane_normal(path: Path) -> Tuple[np.ndarray, dict]:
    """Load hyperplane normal vector. Returns (w [4096], metadata)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    w    = data["vector"].float().numpy()
    meta = {k: v for k, v in data.items() if k != "vector"}
    return w, meta


def load_trait_vectors(
    vectors_dir: Path,
    layer: int,
) -> Tuple[List[str], np.ndarray]:
    """
    Load all trait vectors at a given layer.
    Returns (trait_names, vectors [n_traits, 4096]).
    """
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


# ── Cosine similarity ──────────────────────────────────────────────────────────

def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def angle_degrees(cos_sim: float) -> float:
    """Convert cosine similarity to angle in degrees [0, 180]."""
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_sim)))


# ── Subspace decomposition ─────────────────────────────────────────────────────

def decompose_in_trait_subspace(
    w: np.ndarray,           # [4096] unit vector to decompose
    trait_vectors: np.ndarray, # [n_traits, 4096]
    trait_names: List[str],
) -> dict:
    """
    Find the least-squares best approximation of w in the subspace spanned
    by the trait vectors.

    Solves: min ||w - V^T c||^2  where V is [n_traits, 4096]

    Returns a dict with:
      r_squared       — fraction of w explained by the trait subspace
      residual_norm   — ||w - w_hat|| (unexplained component)
      reconstruction  — top contributing traits with their coefficients
      w_hat_norm      — norm of the reconstruction
    """
    # V: [n_traits, 4096], w: [4096]
    # Least squares: c = (V V^T)^{-1} V w
    # Use numpy lstsq for numerical stability
    V = trait_vectors  # [n_traits, 4096]

    # lstsq expects A x = b where A is [4096, n_traits], b is [4096]
    coeffs, residuals, rank, sv = np.linalg.lstsq(V.T, w, rcond=None)

    w_hat     = V.T @ coeffs           # [4096] reconstruction
    residual  = w - w_hat
    r_squared = 1.0 - (np.dot(residual, residual) / (np.dot(w, w) + 1e-12))
    r_squared = float(np.clip(r_squared, 0.0, 1.0))

    # Rank traits by |coefficient|
    abs_coeffs = np.abs(coeffs)
    top_idx    = np.argsort(abs_coeffs)[::-1]

    reconstruction = [
        {
            "rank":      int(rank_),
            "trait":     trait_names[i],
            "coef":      float(coeffs[i]),
            "abs_coef":  float(abs_coeffs[i]),
            "direction": "adds to jailbreak direction" if coeffs[i] > 0
                         else "subtracts from jailbreak direction",
        }
        for rank_, i in enumerate(top_idx, 1)
    ]

    return {
        "r_squared":      r_squared,
        "residual_norm":  float(np.linalg.norm(residual)),
        "w_hat_norm":     float(np.linalg.norm(w_hat)),
        "matrix_rank":    int(rank),
        "reconstruction": reconstruction,
    }


# ── Printing ───────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_cosine_ranking(
    layer: int,
    w: np.ndarray,
    trait_names: List[str],
    trait_vectors: np.ndarray,
    axis_vector: np.ndarray,
    n_top_bottom: int,
) -> List[dict]:
    """
    Compute and print cosine similarities. Returns full ranked list.
    """
    results = []
    for name, vec in zip(trait_names, trait_vectors):
        cos = cosine_similarity(w, vec)
        results.append({
            "trait":     name,
            "cos_sim":   cos,
            "angle_deg": angle_degrees(cos),
            "direction": "aligned with jailbreak" if cos > 0
                         else "aligned against jailbreak",
        })

    # Add axis
    axis_cos = cosine_similarity(w, axis_vector)
    axis_entry = {
        "trait":     "assistant_axis",
        "cos_sim":   axis_cos,
        "angle_deg": angle_degrees(axis_cos),
        "direction": "aligned with jailbreak" if axis_cos > 0
                     else "aligned against jailbreak",
    }

    # Sort by cosine similarity descending
    results.sort(key=lambda x: x["cos_sim"], reverse=True)
    for rank, entry in enumerate(results, 1):
        entry["rank"] = rank

    print_section(f"LAYER {layer} — Cosine Similarity: Trait Vectors vs Hyperplane Normal")
    print(f"  w direction: positive cos_sim = aligned with jailbreak success")
    print(f"  Angle 0°   = identical direction")
    print(f"  Angle 90°  = orthogonal (irrelevant to jailbreak prediction)")
    print(f"  Angle 180° = opposite direction\n")

    header  = f"  {'Rank':>4}  {'Trait':40s}  {'cos_sim':>8}  {'angle':>8}  Direction"
    divider = "  " + "─" * 75

    print(f"  ── TOP {n_top_bottom} most aligned with jailbreak direction ──")
    print(header)
    print(divider)
    for entry in results[:n_top_bottom]:
        print(f"  {entry['rank']:>4}  {entry['trait']:40s}  "
              f"{entry['cos_sim']:>8.4f}  {entry['angle_deg']:>7.2f}°  "
              f"{entry['direction']}")

    print(f"\n  ── BOTTOM {n_top_bottom} least aligned (most orthogonal/opposite) ──")
    print(header)
    print(divider)
    for entry in results[-n_top_bottom:]:
        print(f"  {entry['rank']:>4}  {entry['trait']:40s}  "
              f"{entry['cos_sim']:>8.4f}  {entry['angle_deg']:>7.2f}°  "
              f"{entry['direction']}")

    # Always print assistant axis regardless of rank
    axis_rank = next(
        (i + 1 for i, r in enumerate(results) if r["trait"] == "assistant_axis"),
        None
    )
    # assistant_axis was not added to results list yet — compute separately
    print(f"\n  ── Assistant Axis (rank {axis_rank if axis_rank else '?'} of {len(results)}) ──")
    print(header)
    print(divider)
    print(f"  {'–':>4}  {'assistant_axis':40s}  "
          f"{axis_entry['cos_sim']:>8.4f}  {axis_entry['angle_deg']:>7.2f}°  "
          f"{axis_entry['direction']}")

    return results, axis_entry


def print_decomposition(layer: int, decomp: dict, n_top: int = 20) -> None:
    print_section(f"LAYER {layer} — Subspace Decomposition")
    print(f"  How much of the hyperplane normal w can be explained by")
    print(f"  a linear combination of the {len(decomp['reconstruction'])} trait vectors?\n")
    print(f"  R²              : {decomp['r_squared']:.4f}  "
          f"({100*decomp['r_squared']:.1f}% of w explained by trait subspace)")
    print(f"  Residual norm   : {decomp['residual_norm']:.4f}  "
          f"(unexplained component of w)")
    print(f"  Reconstruction  : {decomp['w_hat_norm']:.4f}  (norm of trait approximation)")
    print(f"  Matrix rank     : {decomp['matrix_rank']}")

    print(f"\n  Top {n_top} traits by contribution to reconstruction:")
    print(f"  {'Rank':>4}  {'Trait':40s}  {'Coef':>10}  Role")
    print("  " + "─" * 70)
    for entry in decomp["reconstruction"][:n_top]:
        print(f"  {entry['rank']:>4}  {entry['trait']:40s}  "
              f"{entry['coef']:>10.4f}  {entry['direction']}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare hyperplane normals to persona vectors"
    )
    parser.add_argument(
        "--hyperplane_dir", type=str,
        default="full_trait_output/harmbench_logreg",
        help="Directory containing hyperplane_normal_layer{N}.pt files",
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
    parser.add_argument("--n_top_bottom", type=int, default=N_TOP_BOTTOM)
    parser.add_argument("--layers", nargs="+", type=int, default=[16, 28])
    args = parser.parse_args()

    hyperplane_dir = Path(args.hyperplane_dir)
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for layer in args.layers:
        normal_path = hyperplane_dir / f"hyperplane_normal_layer{layer}.pt"
        if not normal_path.exists():
            print(f"Skipping layer {layer}: {normal_path} not found")
            continue

        print(f"\n{'#' * 70}")
        print(f"  LAYER {layer}")
        print(f"{'#' * 70}")

        # Load hyperplane normal
        w, meta = load_hyperplane_normal(normal_path)
        print(f"\n  Hyperplane normal loaded: shape={w.shape}, norm={np.linalg.norm(w):.4f}")
        print(f"  ROC-AUC when learned: {meta.get('roc_auc', 'N/A')}")

        # Load trait vectors + axis at this layer
        trait_names, trait_vectors = load_trait_vectors(
            Path(args.trait_vectors_dir), layer
        )
        axis_vector = load_axis_vector(Path(args.axis_path), layer)
        print(f"  Loaded {len(trait_names)} trait vectors + assistant axis")

        # ── 1. Cosine ranking ──────────────────────────────────────────────────
        ranked, axis_entry = print_cosine_ranking(
            layer=layer,
            w=w,
            trait_names=trait_names,
            trait_vectors=trait_vectors,
            axis_vector=axis_vector,
            n_top_bottom=args.n_top_bottom,
        )

        # ── 2. Subspace decomposition ──────────────────────────────────────────
        print(f"\n  Running subspace decomposition "
              f"({len(trait_names)} trait vectors)...")
        decomp = decompose_in_trait_subspace(w, trait_vectors, trait_names)
        print_decomposition(layer, decomp, n_top=args.n_top_bottom)

        # ── Store results ──────────────────────────────────────────────────────
        all_results[f"layer_{layer}"] = {
            "roc_auc_when_learned": meta.get("roc_auc"),
            "cosine_ranking":       ranked,
            "assistant_axis":       axis_entry,
            "decomposition": {
                "r_squared":      decomp["r_squared"],
                "residual_norm":  decomp["residual_norm"],
                "w_hat_norm":     decomp["w_hat_norm"],
                "top_20_reconstruction": decomp["reconstruction"][:20],
            },
        }

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = output_dir / "hyperplane_persona_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
