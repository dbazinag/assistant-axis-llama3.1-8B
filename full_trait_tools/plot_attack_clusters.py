#!/usr/bin/env python3
"""
plot_attack_clusters.py

PCA scatter plot of layer-16 activations for all attack families,
colored by attack type and jailbreak status.

Usage:
  uv run python full_trait_tools/plot_attack_clusters.py
"""

import json
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LAYER = "16"
DATASETS = [
    ("HarmBench", "full_trait_output/harmbench_activations/activations.pt",
                  "full_trait_output/harmbench_activations/classified_responses.jsonl", "human_jailbreak"),
    ("GCG",       "full_trait_output/gcg_activations/activations.pt",
                  "full_trait_output/gcg_activations/responses.jsonl", None),
    ("PAIR",      "full_trait_output/pair_activations/activations.pt",
                  "full_trait_output/pair_activations/responses.jsonl", None),
    ("PAP",       "full_trait_output/pap_activations/activations.pt",
                  "full_trait_output/pap_activations/responses.jsonl", None),
    ("GPTFuzz",   "full_trait_output/gptfuzz_activations/activations.pt",
                  "full_trait_output/gptfuzz_activations/responses.jsonl", None),
    ("PEZ",       "full_trait_output/pez_activations/activations.pt",
                  "full_trait_output/pez_activations/responses.jsonl", None),
]

# Colors per attack family
FAMILY_COLORS = {
    "HarmBench": "#4e79a7",
    "GCG":       "#f28e2b",
    "PAIR":      "#e15759",
    "PAP":       "#76b7b2",
    "GPTFuzz":   "#59a14f",
    "PEZ":       "#b07aa1",
}

MAX_PER_FAMILY = 300  # subsample for readability


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    all_vecs, all_labels, all_families, all_jailbroken = [], [], [], []

    for name, act_path, resp_path, filter_type in DATASETS:
        print(f"Loading {name}...")
        acts = torch.load(act_path, map_location="cpu", weights_only=False)
        rows = load_jsonl(resp_path)

        if filter_type:
            rows = [r for r in rows if r.get("attack_type") == filter_type]

        # Subsample for readability
        rng = np.random.RandomState(42)
        if len(rows) > MAX_PER_FAMILY:
            idx = rng.choice(len(rows), MAX_PER_FAMILY, replace=False)
            rows = [rows[i] for i in idx]

        for row in rows:
            pid = row["pair_id"]
            jb  = row.get("jailbroken")
            if jb is None:
                continue
            if pid not in acts or LAYER not in acts[pid]:
                continue
            vec = acts[pid][LAYER].float().numpy()
            all_vecs.append(vec)
            all_families.append(name)
            all_jailbroken.append(bool(jb))

        print(f"  {name}: {sum(1 for f in all_families if f == name)} pairs")

    X = np.stack(all_vecs)
    families  = np.array(all_families)
    jailbroken = np.array(all_jailbroken)

    print(f"\nRunning PCA on {len(X)} points...")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_s)
    var = pca.explained_variance_ratio_
    print(f"  PC1: {var[0]:.1%}, PC2: {var[1]:.1%}")

    # ── Plot 1: colored by attack family ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Layer-16 Activation Space — All Attack Families", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.set_title("By Attack Family", fontsize=12)
    for name in FAMILIES_ORDER := ["HarmBench", "GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]:
        mask = families == name
        if not mask.any():
            continue
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=FAMILY_COLORS[name], label=name,
                   alpha=0.4, s=8, linewidths=0)
    ax.set_xlabel(f"PC1 ({var[0]:.1%} var)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]:.1%} var)", fontsize=10)
    ax.legend(markerscale=3, fontsize=9, loc="best")
    ax.grid(True, alpha=0.2)

    # ── Plot 2: colored by jailbreak status ───────────────────────────────────
    ax = axes[1]
    ax.set_title("By Jailbreak Outcome", fontsize=12)
    colors = np.where(jailbroken, "#e15759", "#4e79a7")
    ax.scatter(X_2d[~jailbroken, 0], X_2d[~jailbroken, 1],
               c="#4e79a7", alpha=0.3, s=8, linewidths=0, label="Not jailbroken")
    ax.scatter(X_2d[jailbroken, 0],  X_2d[jailbroken, 1],
               c="#e15759", alpha=0.4, s=8, linewidths=0, label="Jailbroken")
    ax.set_xlabel(f"PC1 ({var[0]:.1%} var)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]:.1%} var)", fontsize=10)
    ax.legend(markerscale=3, fontsize=9, loc="best")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = Path("full_trait_output/plots/attack_clusters.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out}")

    # ── Plot 3: per-family jailbreak split (small multiples) ─────────────────
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    fig2.suptitle("Jailbreak vs Non-Jailbreak by Attack Family (Layer 16 PCA)", fontsize=13, fontweight="bold")
    axes2 = axes2.flatten()

    for i, name in enumerate(["HarmBench", "GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]):
        ax = axes2[i]
        mask = families == name
        X_fam = X_2d[mask]
        jb_fam = jailbroken[mask]
        ax.scatter(X_fam[~jb_fam, 0], X_fam[~jb_fam, 1],
                   c="#4e79a7", alpha=0.4, s=10, linewidths=0, label="Not jailbroken")
        ax.scatter(X_fam[jb_fam, 0],  X_fam[jb_fam, 1],
                   c="#e15759", alpha=0.5, s=10, linewidths=0, label="Jailbroken")
        n_jb = jb_fam.sum()
        n_tot = len(jb_fam)
        ax.set_title(f"{name}  ({n_jb}/{n_tot} jb, {n_jb/n_tot:.0%})", fontsize=10)
        ax.set_xlabel(f"PC1", fontsize=8)
        ax.set_ylabel(f"PC2", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, markerscale=2)

    plt.tight_layout()
    out2 = Path("full_trait_output/plots/attack_clusters_per_family.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved to {out2}")


if __name__ == "__main__":
    main()
