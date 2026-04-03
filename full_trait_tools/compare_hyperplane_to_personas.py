#!/usr/bin/env python3
"""
compare_hyperplane_to_personas.py

Interprets the jailbreak decision hyperplane normals (from
classify_jailbreak_raw_activations.py) by comparing them to persona vectors.

Three analyses:

1. COSINE SIMILARITY RANKING
   For each trait vector and the assistant axis, compute cosine similarity
   with the hyperplane normal w. Print:
   - top N most positively aligned with jailbreak (largest cosine)
   - top N most negatively aligned with jailbreak (smallest cosine)
   - top N closest to orthogonal / 90 degrees (least useful by cosine)
   Also print angle in degrees (0° = identical, 90° = orthogonal,
   180° = opposite).

2. ASSISTANT AXIS (always printed)
   Regardless of rank, always report the assistant axis cosine similarity,
   angle, and its rank in each of the three views above.

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
from typing import List, Tuple

import numpy as np
import torch

N_TOP = 10


# ── Loading ────────────────────────────────────────────────────────────────────

def load_hyperplane_normal(path: Path) -> Tuple[np.ndarray, dict]:
    """Load hyperplane normal vector. Returns (w [4096], metadata)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    w = data["vector"].float().numpy()
    meta = {k: v for k, v in data.items() if k != "vector"}
    return w, meta


def load_trait_vectors(vectors_dir: Path, layer: int) -> Tuple[List[str], np.ndarray]:
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

    return trait_names, np.stack(vectors)


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
    w: np.ndarray,
    trait_vectors: np.ndarray,
    trait_names: List[str],
) -> dict:
    """
    Find the least-squares best approximation of w in the subspace spanned
    by the trait vectors.

    Solves: min ||w - V^T c||^2 where V is [n_traits, 4096]

    Returns a dict with:
      r_squared       — fraction of w explained by the trait subspace
      residual_norm   — ||w - w_hat|| (unexplained component)
      reconstruction  — top contributing traits with their coefficients
      w_hat_norm      — norm of the reconstruction

    Note:
      The least-squares coefficients depend on vector scale and collinearity,
      so coefficient rankings should be interpreted cautiously.
    """
    V = trait_vectors  # [n_traits, 4096]

    # Solve V.T @ coeffs ≈ w
    coeffs, residuals, rank, sv = np.linalg.lstsq(V.T, w, rcond=None)

    w_hat = V.T @ coeffs
    residual = w - w_hat
    r_squared = 1.0 - (np.dot(residual, residual) / (np.dot(w, w) + 1e-12))
    r_squared = float(np.clip(r_squared, 0.0, 1.0))

    abs_coeffs = np.abs(coeffs)
    top_idx = np.argsort(abs_coeffs)[::-1]

    reconstruction = [
        {
            "rank": int(rank_),
            "trait": trait_names[i],
            "coef": float(coeffs[i]),
            "abs_coef": float(abs_coeffs[i]),
            "direction": (
                "adds to jailbreak direction"
                if coeffs[i] > 0
                else "subtracts from jailbreak direction"
            ),
        }
        for rank_, i in enumerate(top_idx, 1)
    ]

    return {
        "r_squared": r_squared,
        "residual_norm": float(np.linalg.norm(residual)),
        "w_hat_norm": float(np.linalg.norm(w_hat)),
        "matrix_rank": int(rank),
        "reconstruction": reconstruction,
    }


# ── Printing ───────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def make_entry(name: str, vec: np.ndarray, w: np.ndarray) -> dict:
    cos = cosine_similarity(w, vec)
    ang = angle_degrees(cos)
    return {
        "trait": name,
        "cos_sim": cos,
        "angle_deg": ang,
        "abs_cos_sim": abs(cos),
        "orthogonality_gap": abs(ang - 90.0),
        "direction": (
            "aligned with jailbreak" if cos > 0
            else "aligned against jailbreak" if cos < 0
            else "orthogonal to jailbreak"
        ),
    }


def add_ranks(results: List[dict]) -> None:
    pos_sorted = sorted(results, key=lambda x: x["cos_sim"], reverse=True)
    neg_sorted = sorted(results, key=lambda x: x["cos_sim"])
    ortho_sorted = sorted(results, key=lambda x: x["orthogonality_gap"])

    for rank, entry in enumerate(pos_sorted, 1):
        entry["rank_positive"] = rank
    for rank, entry in enumerate(neg_sorted, 1):
        entry["rank_negative"] = rank
    for rank, entry in enumerate(ortho_sorted, 1):
        entry["rank_orthogonal"] = rank


def print_ranked_block(title: str, entries: List[dict], rank_key: str) -> None:
    print(f"\n  ── {title} ──")
    header = f"  {'Rank':>4}  {'Trait':40s}  {'cos_sim':>8}  {'angle':>8}  Direction"
    divider = "  " + "─" * 88
    print(header)
    print(divider)
    for entry in entries:
        print(
            f"  {entry[rank_key]:>4}  {entry['trait']:40s}  "
            f"{entry['cos_sim']:>8.4f}  {entry['angle_deg']:>7.2f}°  "
            f"{entry['direction']}"
        )


def print_assistant_axis_summary(axis_entry: dict, total_count: int) -> None:
    print(f"\n  ── Assistant Axis Summary ──")
    print(f"  Trait        : assistant_axis")
    print(f"  cos_sim      : {axis_entry['cos_sim']:.4f}")
    print(f"  angle        : {axis_entry['angle_deg']:.2f}°")
    print(f"  direction    : {axis_entry['direction']}")
    print(f"  positive rank: {axis_entry['rank_positive']} of {total_count}")
    print(f"  negative rank: {axis_entry['rank_negative']} of {total_count}")
    print(f"  90° rank     : {axis_entry['rank_orthogonal']} of {total_count}")


def print_cosine_ranking(
    layer: int,
    w: np.ndarray,
    trait_names: List[str],
    trait_vectors: np.ndarray,
    axis_vector: np.ndarray,
    n_top: int,
) -> Tuple[List[dict], dict]:
    """
    Compute and print cosine similarities.

    Returns:
      - full_results: full ranked list including assistant axis
      - axis_entry: the assistant axis entry
    """
    results = [make_entry(name, vec, w) for name, vec in zip(trait_names, trait_vectors)]

    axis_entry = make_entry("assistant_axis", axis_vector, w)
    results.append(axis_entry)

    add_ranks(results)

    positive_sorted = sorted(results, key=lambda x: x["cos_sim"], reverse=True)
    negative_sorted = sorted(results, key=lambda x: x["cos_sim"])
    orthogonal_sorted = sorted(results, key=lambda x: x["orthogonality_gap"])

    print_section(f"LAYER {layer} — Cosine Similarity: Trait Vectors vs Hyperplane Normal")
    print(f"  w direction: positive cos_sim = aligned with jailbreak success")
    print(f"  Angle 0°   = identical direction")
    print(f"  Angle 90°  = orthogonal (least useful by cosine)")
    print(f"  Angle 180° = opposite direction\n")

    print_ranked_block(
        title=f"TOP {n_top} most positively aligned with jailbreak direction",
        entries=positive_sorted[:n_top],
        rank_key="rank_positive",
    )

    print_ranked_block(
        title=f"TOP {n_top} most negatively aligned with jailbreak direction",
        entries=negative_sorted[:n_top],
        rank_key="rank_negative",
    )

    print_ranked_block(
        title=f"TOP {n_top} closest to 90° (most orthogonal / least useful by cosine)",
        entries=orthogonal_sorted[:n_top],
        rank_key="rank_orthogonal",
    )

    print_assistant_axis_summary(axis_entry, total_count=len(results))

    return results, axis_entry


def print_decomposition(layer: int, decomp: dict, n_top: int = 20) -> None:
    print_section(f"LAYER {layer} — Subspace Decomposition")
    print(f"  How much of the hyperplane normal w can be explained by")
    print(f"  a linear combination of the {len(decomp['reconstruction'])} trait vectors?\n")
    print(
        f"  R²              : {decomp['r_squared']:.4f}  "
        f"({100 * decomp['r_squared']:.1f}% of w explained by trait subspace)"
    )
    print(
        f"  Residual norm   : {decomp['residual_norm']:.4f}  "
        f"(unexplained component of w)"
    )
    print(
        f"  Reconstruction  : {decomp['w_hat_norm']:.4f}  "
        f"(norm of trait approximation)"
    )
    print(f"  Matrix rank     : {decomp['matrix_rank']}")
    print(f"  Note            : coefficient magnitudes depend on vector scale/correlation")

    print(f"\n  Top {n_top} traits by |least-squares coefficient|:")
    print(f"  {'Rank':>4}  {'Trait':40s}  {'Coef':>10}  Role")
    print("  " + "─" * 82)
    for entry in decomp["reconstruction"][:n_top]:
        print(
            f"  {entry['rank']:>4}  {entry['trait']:40s}  "
            f"{entry['coef']:>10.4f}  {entry['direction']}"
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare hyperplane normals to persona vectors"
    )
    parser.add_argument(
        "--hyperplane_dir",
        type=str,
        default="full_trait_output/harmbench_logreg",
        help="Directory containing hyperplane_normal_layer{N}.pt files",
    )
    parser.add_argument(
        "--trait_vectors_dir",
        type=str,
        default="full_trait_output/traits40_vectors/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total",
    )
    parser.add_argument(
        "--axis_path",
        type=str,
        default="full_trait_output/traits40_axes/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total/assistant_axis_pc1.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="full_trait_output/harmbench_logreg",
    )
    parser.add_argument("--n_top", type=int, default=N_TOP)
    parser.add_argument("--layers", nargs="+", type=int, default=[16, 28])
    args = parser.parse_args()

    hyperplane_dir = Path(args.hyperplane_dir)
    output_dir = Path(args.output_dir)
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

        w, meta = load_hyperplane_normal(normal_path)
        print(f"\n  Hyperplane normal loaded: shape={w.shape}, norm={np.linalg.norm(w):.4f}")
        print(f"  ROC-AUC when learned: {meta.get('roc_auc', 'N/A')}")

        trait_names, trait_vectors = load_trait_vectors(Path(args.trait_vectors_dir), layer)
        axis_vector = load_axis_vector(Path(args.axis_path), layer)
        print(f"  Loaded {len(trait_names)} trait vectors + assistant axis")

        ranked, axis_entry = print_cosine_ranking(
            layer=layer,
            w=w,
            trait_names=trait_names,
            trait_vectors=trait_vectors,
            axis_vector=axis_vector,
            n_top=args.n_top,
        )

        print(f"\n  Running subspace decomposition ({len(trait_names)} trait vectors)...")
        decomp = decompose_in_trait_subspace(w, trait_vectors, trait_names)
        print_decomposition(layer, decomp, n_top=20)

        all_results[f"layer_{layer}"] = {
            "roc_auc_when_learned": meta.get("roc_auc"),
            "cosine_ranking_all": ranked,
            "assistant_axis": axis_entry,
            "top_positive_alignment": sorted(
                ranked, key=lambda x: x["cos_sim"], reverse=True
            )[:args.n_top],
            "top_negative_alignment": sorted(
                ranked, key=lambda x: x["cos_sim"]
            )[:args.n_top],
            "top_closest_to_90deg": sorted(
                ranked, key=lambda x: x["orthogonality_gap"]
            )[:args.n_top],
            "decomposition": {
                "r_squared": decomp["r_squared"],
                "residual_norm": decomp["residual_norm"],
                "w_hat_norm": decomp["w_hat_norm"],
                "matrix_rank": decomp["matrix_rank"],
                "top_20_reconstruction": decomp["reconstruction"][:20],
            },
        }

    out_path = output_dir / "hyperplane_persona_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()