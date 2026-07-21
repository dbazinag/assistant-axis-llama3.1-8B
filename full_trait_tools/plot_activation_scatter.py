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
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
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

# ── Palette (Assistant-Axis-paper inspired) ────────────────────────────────────
FIG_BG    = "#F1EDE3"   # warm cream page
PANEL_BG  = "#FBFAF6"   # panel card
C_REFUSED = "#7F9EC0"   # soft blue  (assistant-like)
C_JAILBRK = "#CE8A8A"   # soft rose  (role-playing / jailbroken)
C_DEGEN   = "#BFBAB0"   # muted grey
AXIS_BLUE = "#37618E"   # the assistant axis
ARROW_RED = "#B15E6C"   # persona-direction arrows
TEXT_DARK = "#3D3B37"
TEXT_MUT  = "#6E6A62"
GRID      = "#E4DFD4"
SPINE     = "#D8D3C7"
BOUNDARY  = "#8A857A"

# soft blue -> cream -> rose band aligned with the classifier normal
GRAD_CMAP = LinearSegmentedColormap.from_list(
    "assist_grad", ["#AAC1D9", "#F5F2EA", "#E4B4B4"])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": FIG_BG,
    "savefig.facecolor": FIG_BG,
})


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

def _panel_limits(X_2d: np.ndarray):
    """Robust limits from the 1st/99th percentile so outliers don't shrink the
    cluster; keeps the bulk of the data big and visible."""
    lo = np.percentile(X_2d, 1, axis=0)
    hi = np.percentile(X_2d, 99, axis=0)
    pad = 0.12 * (hi - lo)
    return (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])


def plot_panel(
    ax,
    X_2d: np.ndarray,
    labels: np.ndarray,      # 0=refused, 1=jailbroken, 2=degenerate
    trait_arrows: Dict[str, np.ndarray],  # name -> 2D projected vector
    w_2d: np.ndarray,
    title: str,
    pca_var: List[float],
) -> None:

    ax.set_facecolor(PANEL_BG)
    xlim, ylim = _panel_limits(X_2d)

    # 1. soft assistant/role-play gradient aligned with the classifier normal
    wn = w_2d / (np.linalg.norm(w_2d) + 1e-12)
    gx = np.linspace(*xlim, 240)
    gy = np.linspace(*ylim, 240)
    XX, YY = np.meshgrid(gx, gy)
    proj = XX * wn[0] + YY * wn[1]
    mx = np.abs(proj).max() + 1e-9
    ax.imshow(proj, extent=[*xlim, *ylim], origin="lower", aspect="auto",
              cmap=GRAD_CMAP, vmin=-mx, vmax=mx, alpha=0.16,
              interpolation="bilinear", zorder=0)

    # 2. grid + frame
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=1)
    for s in ax.spines.values():
        s.set_color(SPINE)
        s.set_linewidth(0.9)
    ax.tick_params(colors=TEXT_MUT, labelsize=9, length=0)

    # 3. scatter
    m_ref = labels == 0
    m_jb  = labels == 1
    m_deg = labels == 2
    if m_deg.any():
        ax.scatter(X_2d[m_deg, 0], X_2d[m_deg, 1], c=C_DEGEN, s=14,
                   alpha=0.35, linewidths=0, zorder=2)
    ax.scatter(X_2d[m_ref, 0], X_2d[m_ref, 1], c=C_REFUSED, s=26, alpha=0.6,
               edgecolors="white", linewidths=0.3, zorder=3)
    ax.scatter(X_2d[m_jb, 0], X_2d[m_jb, 1], c=C_JAILBRK, s=26, alpha=0.6,
               edgecolors="white", linewidths=0.3, zorder=3)

    # 4. classifier boundary (w·x = 0  ->  y = -(w0/w1) x)
    if abs(w_2d[1]) > 1e-6:
        xr = np.array(xlim)
        ax.plot(xr, -(w_2d[0] / w_2d[1]) * xr, ls=(0, (6, 4)),
                color=BOUNDARY, lw=1.4, alpha=0.75, zorder=4)

    span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])

    cx = 0.5 * (xlim[0] + xlim[1])
    cy = 0.5 * (ylim[0] + ylim[1])

    def draw_arrow(vec, length, color):
        d = vec / (np.linalg.norm(vec) + 1e-12)
        tip = d * length
        ax.annotate("", xy=(tip[0], tip[1]), xytext=(0, 0), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=2.4, mutation_scale=18,
                                    shrinkA=0, shrinkB=0))
        return d, tip

    def label_ha(lx, text, dx):
        # extend outward from the origin, but flip inward near a panel edge
        tw = 0.011 * span * len(text)
        if dx >= 0:
            return "right" if lx + tw > xlim[1] else "left"
        return "left" if lx - tw < xlim[0] else "right"

    def draw_label(lx, ly, text, color, fontsize, ha, va):
        t = ax.text(lx, ly, text, color=color, fontsize=fontsize,
                    fontweight="bold", ha=ha, va=va, zorder=8, clip_on=True)
        t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    # 5. persona direction arrows (rose), lightly staggered. Labels hug each
    #    arrowhead, then get spread apart vertically so none overlap.
    persona = [(n, v) for n, v in trait_arrows.items() if n != AXIS_NAME]
    n_p = len(persona)
    plabels = []
    for i, (name, vec_2d) in enumerate(persona):
        L = (0.42 - 0.02 * i) * span if n_p > 1 else 0.36 * span
        d, tip = draw_arrow(vec_2d, L, ARROW_RED)
        lx = tip[0] + d[0] * 0.03 * span
        ly = tip[1] + d[1] * 0.03 * span
        plabels.append({"x": lx, "y": ly, "name": name,
                        "ha": label_ha(lx, name, d[0])})
    gap = 0.05 * span
    order = sorted(range(n_p), key=lambda k: -plabels[k]["y"])  # top first
    for j in range(1, n_p):
        a, b = order[j - 1], order[j]
        if plabels[b]["y"] > plabels[a]["y"] - gap:
            plabels[b]["y"] = plabels[a]["y"] - gap
    for pl in plabels:
        draw_label(pl["x"], pl["y"], pl["name"], ARROW_RED, 9.5, pl["ha"],
                   "center")

    # 6. assistant axis: same arrow style as the others, distinct blue.
    if AXIS_NAME in trait_arrows:
        d, tip = draw_arrow(trait_arrows[AXIS_NAME], 0.46 * span, AXIS_BLUE)
        lx, ly = tip[0] + d[0] * 0.03 * span, tip[1] + d[1] * 0.03 * span
        draw_label(lx, ly, AXIS_NAME, AXIS_BLUE, 10,
                   label_ha(lx, AXIS_NAME, d[0]),
                   "top" if d[1] < 0 else "bottom")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(f"PC1  ({100*pca_var[0]:.1f}% variance)", fontsize=10,
                  color=TEXT_MUT)
    ax.set_ylabel(f"PC2  ({100*pca_var[1]:.1f}% variance)", fontsize=10,
                  color=TEXT_MUT)
    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT_DARK, pad=10)

    handles = [
        Line2D([0], [0], marker="o", ls="", ms=8, mec="white", mew=0.4,
               color=C_JAILBRK, label=f"Jailbroken (n={m_jb.sum()})"),
        Line2D([0], [0], marker="o", ls="", ms=8, mec="white", mew=0.4,
               color=C_REFUSED, label=f"Refused (n={m_ref.sum()})"),
        Line2D([0], [0], color=BOUNDARY, ls=(0, (6, 4)), lw=1.4,
               label="Classifier boundary"),
    ]
    leg = ax.legend(handles=handles, fontsize=9, loc="upper right",
                    framealpha=0.9, edgecolor=SPINE)
    leg.get_frame().set_facecolor("white")


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
    parser.add_argument("--single", action="store_true",
        help="Render only the Jailbreak-PCA panel (single-panel paper figure).")
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
    if args.single:
        fig, ax = plt.subplots(figsize=(8.4, 7.2), dpi=200)
        plot_panel(
            ax, X_jb_2d, labels,
            trait_arrows_jb, w_jb_2d,
            title=f"Pre-generation activation space  ·  Layer {args.layer}",
            pca_var=pca_jb.explained_variance_ratio_[:2],
        )
        fig.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"\nSaved to {output_path}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), dpi=200)
    fig.suptitle(
        f"Pre-generation activation space  ·  Layer {args.layer}",
        fontsize=15, fontweight="bold", color=TEXT_DARK, y=1.02,
    )

    plot_panel(
        axes[0], X_jb_2d, labels,
        trait_arrows_jb, w_jb_2d,
        title="Jailbreak PCA",
        pca_var=pca_jb.explained_variance_ratio_[:2],
    )

    plot_panel(
        axes[1], X_fresh_2d, labels,
        trait_arrows_fresh, w_fresh_2d,
        title="Fresh PCA",
        pca_var=pca_fresh.explained_variance_ratio_,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
