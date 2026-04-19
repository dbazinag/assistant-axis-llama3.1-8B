#!/usr/bin/env python3
"""
plot_activation_scatter.py

Visualizes the jailbreak activation space at layer 16 as a 2D scatter plot.
Each point is one (behavior, jailbreak template) pair's pre-generation activation,
colored by whether the model was jailbroken or refused.

Overlays trait vector arrows showing where key persona directions point
in the same 2D space.

Produces TWO plots side by side:
  Left:  Jailbreak PCA axes (PC1/PC2 from PCA fit on harmbench activations)
  Right: Fresh PCA axes (PCA fit fresh for visualization, same data)

Usage:
  uv run full_trait_tools/plot_activation_scatter.py
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

LAYER = 16
RANDOM_SEED = 42

# Traits to show as arrows
TRAITS_TO_SHOW = [
    "naive",
    "essentialist",
    "utilitarian",
    "absolutist",
    "progressive",
]
AXIS_NAME = "assistant_axis"

# Colors
COLOR_JAILBROKEN = "#e63946"   # red
COLOR_REFUSED    = "#457b9d"   # blue
COLOR_DEGENERATE = "#adb5bd"   # grey

# Arrow colors per trait
TRAIT_COLORS = {
    "naive":         "#f4a261",
    "essentialist":  "#e76f51",
    "utilitarian":   "#2a9d8f",
    "absolutist":    "#e9c46a",
    "progressive":   "#264653",
    "assistant_axis": "#6a0dad",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_trait_vector(vectors_dir: Path, trait: str, layer: int) -> np.ndarray:
    path = vectors_dir / f"{trait}.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    v = data["vector"][layer].float().numpy()
    return v / (np.linalg.norm(v) + 1e-12)


def load_axis(axis_path: Path, layer: int) -> np.ndarray:
    data = torch.load(axis_path, map_location="cpu", weights_only=False)
    v = data["axis"][layer].float().numpy()
    return v / (np.linalg.norm(v) + 1e-12)


def load_hyperplane(path: Path) -> np.ndarray:
    data = torch.load(path, map_location="cpu", weights_only=False)
    v = data["vector"].float().numpy()
    return v / (np.linalg.norm(v) + 1e-12)


# ── Plot one scatter panel ─────────────────────────────────────────────────────

def plot_panel(
    ax,
    X_2d: np.ndarray,
    labels: np.ndarray,      # 0=refused, 1=jailbroken, 2=degenerate
    trait_arrows: Dict[str, np.ndarray],  # name -> 2D projected vector
    w_2d: np.ndarray,
    title: str,
    pca_var: List[float],
) -> None:

    # Scatter
    mask_jb  = labels == 1
    mask_ref = labels == 0
    mask_deg = labels == 2

    ax.scatter(X_2d[mask_ref, 0], X_2d[mask_ref, 1],
               c=COLOR_REFUSED, alpha=0.35, s=8, linewidths=0, label="Refused")
    ax.scatter(X_2d[mask_jb, 0],  X_2d[mask_jb, 1],
               c=COLOR_JAILBROKEN, alpha=0.35, s=8, linewidths=0, label="Jailbroken")
    if mask_deg.any():
        ax.scatter(X_2d[mask_deg, 0], X_2d[mask_deg, 1],
                   c=COLOR_DEGENERATE, alpha=0.2, s=6, linewidths=0, label="Degenerate")

    # Classifier boundary line (w projected to 2D)
    # The boundary is w·x = 0, so in 2D: w[0]*x + w[1]*y = 0 → y = -(w[0]/w[1])*x
    xlims = ax.get_xlim()
    if abs(w_2d[1]) > 1e-6:
        x_range = np.linspace(X_2d[:, 0].min() * 1.2, X_2d[:, 0].max() * 1.2, 100)
        y_range = -(w_2d[0] / w_2d[1]) * x_range
        # Only draw within plot bounds
        ax.plot(x_range, y_range, "k--", linewidth=1.0, alpha=0.5, label="Classifier boundary (w)")

    # Scale arrows to ~20% of axis range
    x_range_size = X_2d[:, 0].max() - X_2d[:, 0].min()
    y_range_size = X_2d[:, 1].max() - X_2d[:, 1].min()
    scale = 0.22 * max(x_range_size, y_range_size)

    for name, vec_2d in trait_arrows.items():
        norm = np.linalg.norm(vec_2d) + 1e-12
        direction = vec_2d / norm
        color = TRAIT_COLORS.get(name, "#333333")
        ax.annotate(
            "", xy=(direction[0] * scale, direction[1] * scale),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=2.0,
            ),
        )
        # Label at arrow tip with slight offset
        ax.text(
            direction[0] * scale * 1.12,
            direction[1] * scale * 1.12,
            name,
            fontsize=7.5,
            color=color,
            ha="center", va="center",
            fontweight="bold",
        )

    ax.set_xlabel(f"PC1 ({100*pca_var[0]:.1f}% var)", fontsize=10)
    ax.set_ylabel(f"PC2 ({100*pca_var[1]:.1f}% var)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axhline(0, color="grey", linewidth=0.4, alpha=0.4)
    ax.axvline(0, color="grey", linewidth=0.4, alpha=0.4)
    ax.set_xlim(X_2d[:, 0].min() * 1.25, X_2d[:, 0].max() * 1.25)
    ax.set_ylim(X_2d[:, 1].min() * 1.25, X_2d[:, 1].max() * 1.25)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLOR_JAILBROKEN, alpha=0.7, label=f"Jailbroken (n={mask_jb.sum()})"),
        mpatches.Patch(color=COLOR_REFUSED,    alpha=0.7, label=f"Refused (n={mask_ref.sum()})"),
        Line2D([0], [0], color="k", linestyle="--", linewidth=1, label="Classifier boundary"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classified_path", type=str,
        default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--activations_path", type=str,
        default="full_trait_output/harmbench_activations/activations.pt")
    parser.add_argument("--hyperplane_path", type=str,
        default="full_trait_output/harmbench_logreg/stable_hyperplane_layer16.pt")
    parser.add_argument("--trait_vectors_dir", type=str,
        default="full_trait_output/traits40_vectors/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total")
    parser.add_argument("--axis_path", type=str,
        default="full_trait_output/traits40_axes/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total/assistant_axis_pc1.pt")
    parser.add_argument("--output_path", type=str,
        default="full_trait_output/plots/activation_scatter_layer16.png")
    parser.add_argument("--layer", type=int, default=LAYER)
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading classified responses...")
    rows = load_jsonl(Path(args.classified_path))
    rows = [r for r in rows if r.get("attack_type") == "human_jailbreak"]
    print(f"  {len(rows)} human jailbreak pairs")

    print("Loading activations...")
    activations = torch.load(
        Path(args.activations_path), map_location="cpu", weights_only=False
    )

    layer_key = str(args.layer)
    X_list, labels_list = [], []
    for row in rows:
        pid = row["pair_id"]
        if pid not in activations or layer_key not in activations[pid]:
            continue
        act = activations[pid][layer_key].float().numpy()
        X_list.append(act)
        jb = row.get("jailbroken")
        if jb is True:
            labels_list.append(1)
        elif jb is False:
            labels_list.append(0)
        else:
            labels_list.append(2)  # unknown/degenerate

    X_all    = np.stack(X_list)
    labels   = np.array(labels_list)
    print(f"  Matrix shape: {X_all.shape}")
    print(f"  Jailbroken: {(labels==1).sum()}, Refused: {(labels==0).sum()}, Other: {(labels==2).sum()}")

    # ── Standardize ────────────────────────────────────────────────────────────
    print("Standardizing...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # ── Fit jailbreak PCA (4 components, use first 2 for plotting) ─────────────
    print("Fitting jailbreak PCA (n=4)...")
    pca_jb = PCA(n_components=4, random_state=RANDOM_SEED)
    pca_jb.fit(X_scaled)
    X_jb_4d = pca_jb.transform(X_scaled)
    X_jb_2d = X_jb_4d[:, :2]
    print(f"  Var explained: {[f'{100*v:.1f}%' for v in pca_jb.explained_variance_ratio_]}")

    # ── Fit fresh PCA (2 components for visualization) ────────────────────────
    print("Fitting fresh PCA (n=2)...")
    pca_fresh = PCA(n_components=2, random_state=RANDOM_SEED)
    X_fresh_2d = pca_fresh.fit_transform(X_scaled)
    print(f"  Var explained: {[f'{100*v:.1f}%' for v in pca_fresh.explained_variance_ratio_]}")

    # ── Load trait vectors + project to 2D ────────────────────────────────────
    print("Loading trait vectors...")
    vectors_dir = Path(args.trait_vectors_dir)

    raw_vecs = {}
    for trait in TRAITS_TO_SHOW:
        try:
            raw_vecs[trait] = load_trait_vector(vectors_dir, trait, args.layer)
            print(f"  Loaded: {trait}")
        except FileNotFoundError:
            print(f"  WARNING: {trait} not found")

    try:
        raw_vecs[AXIS_NAME] = load_axis(Path(args.axis_path), args.layer)
        print(f"  Loaded: {AXIS_NAME}")
    except FileNotFoundError:
        print(f"  WARNING: {AXIS_NAME} not found")

    # Project each trait vector into PCA space
    # For a unit vector v in original space, its projection onto PC_i is:
    # v_projected[i] = pca.components_[i] · v  (since scaler mean ~ 0 for unit vecs)
    def project_to_pca_2d(vec, pca):
        return np.array([
            np.dot(pca.components_[0], vec),
            np.dot(pca.components_[1], vec),
        ])

    trait_arrows_jb = {
        name: project_to_pca_2d(vec, pca_jb)
        for name, vec in raw_vecs.items()
    }
    trait_arrows_fresh = {
        name: project_to_pca_2d(vec, pca_fresh)
        for name, vec in raw_vecs.items()
    }

    # ── Load + project hyperplane normal ───────────────────────────────────────
    print("Loading hyperplane...")
    w_vec = load_hyperplane(Path(args.hyperplane_path))
    w_jb_2d    = project_to_pca_2d(w_vec, pca_jb)
    w_fresh_2d = project_to_pca_2d(w_vec, pca_fresh)

    # ── Plot ───────────────────────────────────────────────────────────────────
    print("Plotting...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"Pre-generation Activation Space — Layer {args.layer}\n"
        f"Human jailbreak pairs (n={len(X_list)}), colored by jailbreak outcome",
        fontsize=12, fontweight="bold", y=1.01
    )

    plot_panel(
        axes[0], X_jb_2d, labels,
        trait_arrows_jb, w_jb_2d,
        title="Jailbreak PCA\n(PCA fit on jailbreak activations)",
        pca_var=pca_jb.explained_variance_ratio_[:2],
    )

    plot_panel(
        axes[1], X_fresh_2d, labels,
        trait_arrows_fresh, w_fresh_2d,
        title="Fresh PCA\n(PCA fit fresh for visualization)",
        pca_var=pca_fresh.explained_variance_ratio_,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
