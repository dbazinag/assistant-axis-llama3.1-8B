#!/usr/bin/env python3
"""
pca_direction_decomposition.py

Decompose the jailbreak-detection direction learned on raw (unprojected) layer-16
activations into its top PCA components. For each PC: variance explained (unsupervised,
no labels), its standalone AUC (same method as persona_individual_auc.py), the weight
it receives in a small logistic regression fit on the PCA scores, and its nearest
persona trait vectors by cosine similarity (interpretability).

PCA is fit once on all HarmBench raw activations (unsupervised -- no label leakage,
same status as the fixed persona vectors). Standalone AUC and classifier coefficients
are averaged over the same held-out seeds used elsewhere in this project.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_all_traits_sweep_v2 import (
    build_activation_matrix,
    get_pool_split,
    load_jsonl,
    load_trait_matrix,
    safe_auc,
    split_by_pool,
)

LAYER = 16
TRAIN_FRAC = 0.7
N_SEEDS = 101
N_COMPONENTS = 4
PCA_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_classified_path",
                        default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--human_activations_path",
                        default="full_trait_output/harmbench_activations/activations.pt")
    parser.add_argument("--output_dir", default="full_trait_output/pca_direction_decomposition")
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--n_components", type=int, default=N_COMPONENTS)
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS)
    parser.add_argument("--train_frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--n_top_personas", type=int, default=5,
                        help="Number of nearest personas to record per pole (by cosine similarity).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading data ===", flush=True)
    human_rows = [
        r for r in load_jsonl(Path(args.human_classified_path))
        if r.get("attack_type") == "human_jailbreak"
    ]
    human_acts = torch.load(Path(args.human_activations_path), map_location="cpu", weights_only=False)
    x_raw_h, y_h, human_valid = build_activation_matrix(human_rows, human_acts, args.layer)
    print(f"  HarmBench: {x_raw_h.shape}, jb={y_h.mean():.3f}", flush=True)

    trait_matrix, trait_names = load_trait_matrix(args.layer)
    print(f"  Trait matrix: {trait_matrix.shape}", flush=True)

    # Unsupervised PCA on all raw activations -- no labels used, same status as fixed persona vectors.
    pca = PCA(n_components=args.n_components, random_state=PCA_SEED)
    pcs = pca.fit_transform(x_raw_h).astype(np.float32)
    var_ratio = pca.explained_variance_ratio_
    print(f"  PCA: {args.n_components} components, "
          f"cumulative var={var_ratio.sum():.3f}", flush=True)

    # --- Standalone AUC per PC + classifier weights, averaged over held-out seeds ---
    per_pc_aucs = [[] for _ in range(args.n_components)]
    per_pc_coefs = [[] for _ in range(args.n_components)]
    n_used_seeds = 0
    for seed in range(args.n_seeds):
        train_beh, train_tpl, test_beh, test_tpl = get_pool_split(human_valid, args.train_frac, seed)
        train_idx, test_idx = split_by_pool(human_valid, train_beh, train_tpl, test_beh, test_tpl)
        if not train_idx or not test_idx or len(np.unique(y_h[train_idx])) < 2:
            continue
        n_used_seeds += 1

        for i in range(args.n_components):
            per_pc_aucs[i].append(safe_auc(y_h[test_idx], pcs[test_idx, i]))

        m = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, random_state=seed)),
        ])
        m.fit(pcs[train_idx], y_h[train_idx])
        w = m.named_steps["clf"].coef_[0]
        w_unit = w / np.linalg.norm(w)
        for i in range(args.n_components):
            per_pc_coefs[i].append(w_unit[i])

    print(f"  Used {n_used_seeds}/{args.n_seeds} seeds", flush=True)

    # --- Nearest persona alignment per PC (cosine similarity, unsupervised) ---
    # trait_matrix rows and pca.components_ rows are both unit-norm.
    cos_sim = trait_matrix @ pca.components_.T  # [n_traits, n_components]

    n_top = args.n_top_personas
    results = []
    for i in range(args.n_components):
        coefs = np.array(per_pc_coefs[i])
        mean_coef = float(np.mean(coefs))
        order = np.argsort(cos_sim[:, i])  # ascending
        pos_order = order[::-1][:n_top]   # most positive first
        neg_order = order[:n_top]         # most negative first
        direction = "jailbreak" if mean_coef > 0 else "refusal"
        results.append({
            "pc": i + 1,
            "var_pct": float(var_ratio[i] * 100),
            "coef_in_w_mean": mean_coef,
            "coef_in_w_std": float(np.std(coefs)),
            "coef_sq": float(mean_coef ** 2),
            "standalone_auc_mean": float(np.nanmean(per_pc_aucs[i])),
            "standalone_auc_std": float(np.nanstd(per_pc_aucs[i])),
            "positive_pole_top": [
                {"persona": trait_names[j], "cos_sim": float(cos_sim[j, i])} for j in pos_order
            ],
            "negative_pole_top": [
                {"persona": trait_names[j], "cos_sim": float(cos_sim[j, i])} for j in neg_order
            ],
            "direction": direction,
        })

    # Sort by |coef_in_w| descending, matching the reference table's presentation order.
    results_sorted = sorted(results, key=lambda r: abs(r["coef_in_w_mean"]), reverse=True)

    coef_sq_sum = sum(r["coef_sq"] for r in results)
    print("\n" + "=" * 100)
    print(f"  LAYER {args.layer} -- JAILBREAK DIRECTION DECOMPOSITION ({args.n_components} PCs) "
          f"| {n_used_seeds} seeds")
    print("=" * 100)
    print(f"  {'PC':4s}{'Var%':>8s}{'CoefInW':>10s}{'Coef2':>9s}{'StandaloneAUC':>15s}  Persona profile (top-1 each pole)")
    print("  " + "-" * 96)
    for r in results_sorted:
        neg1 = r["negative_pole_top"][0]["persona"]
        pos1 = r["positive_pole_top"][0]["persona"]
        profile = f"{neg1} <-> {pos1}  (-> {r['direction']})"
        print(f"  PC{r['pc']:<3d}{r['var_pct']:8.1f}{r['coef_in_w_mean']:10.4f}"
              f"{r['coef_sq']:9.4f}{r['standalone_auc_mean']:15.4f}  {profile}")
    print(f"\n  Sum of coef^2 = {coef_sq_sum:.4f} (should be 1.0)")
    print("=" * 100)

    print(f"\n  Top-{n_top} aligned personas per pole:")
    for r in results_sorted:
        pos_str = ", ".join(f"{p['persona']}({p['cos_sim']:.2f})" for p in r["positive_pole_top"])
        neg_str = ", ".join(f"{p['persona']}({p['cos_sim']:.2f})" for p in r["negative_pole_top"])
        print(f"  PC{r['pc']} positive: {pos_str}")
        print(f"  PC{r['pc']} negative: {neg_str}")

    out = {
        "method": "pca_direction_decomposition",
        "layer": args.layer,
        "n_components": args.n_components,
        "n_seeds": args.n_seeds,
        "n_seeds_used": n_used_seeds,
        "cumulative_var_pct": float(var_ratio.sum() * 100),
        "results": results_sorted,
    }
    out_path = out_dir / "results.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
