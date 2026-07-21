#!/usr/bin/env python3
"""
raw_vs_trait_classifier_alignment.py

Direct comparison of two already-saved, already-fitted vectors -- no refitting:

  w_raw   = hyperplane normal from the raw-activation classifier
            (full_trait_output/harmbench_logreg/hyperplane_normal_layer16.pt)
  w_trait = weight vector of the all-traits classifier trained on the 229
            persona-projected features (full_trait_output/all_traits_sweep_v2/best_model.pkl,
            field 'coef_proj'), mapped back into raw 4096-dim space via that
            same file's 'trait_matrix' so it's directly comparable to w_raw.

Reports the single overall cosine similarity between w_raw and w_trait, plus,
for each of the two directions, the top-N persona vectors most aligned with it
(by |cos_sim|, sign-agnostic) and the top-N most orthogonal to it.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def top_bottom(align: np.ndarray, trait_names: list[str], k: int):
    order = np.argsort(-np.abs(align))
    top = [(trait_names[i], float(align[i])) for i in order[:k]]
    bottom = [(trait_names[i], float(align[i])) for i in order[::-1][:k]]
    return top, bottom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperplane_path",
                        default="full_trait_output/harmbench_logreg/hyperplane_normal_layer16.pt")
    parser.add_argument("--trait_model_path",
                        default="full_trait_output/all_traits_sweep_v2/best_model.pkl")
    parser.add_argument("--output_dir", default="full_trait_output/raw_vs_trait_alignment")
    parser.add_argument("--n_top", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hp = torch.load(Path(args.hyperplane_path), map_location="cpu", weights_only=False)
    w_raw = unit(hp["vector"].float().numpy())
    print(f"  Loaded w_raw: shape={w_raw.shape}, saved ROC-AUC={hp.get('roc_auc')}", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(args.trait_model_path, "rb") as f:
            trait_model = pickle.load(f)
    coef_proj = trait_model["coef_proj"].astype(np.float64)     # [229] persona-classifier weights
    trait_matrix = trait_model["trait_matrix"].astype(np.float64)  # [229, 4096] unit-norm rows
    trait_names = trait_model["trait_names"]
    print(f"  Loaded w_trait: {trait_model['meta']['model_name']}, "
          f"{len(trait_names)} traits", flush=True)

    w_trait_raw_space = unit(trait_matrix.T @ coef_proj)

    overall_sim = float(np.dot(w_raw, w_trait_raw_space))

    raw_align = trait_matrix @ w_raw               # [229] cos_sim(w_raw, persona_i)
    trait_align = trait_matrix @ w_trait_raw_space  # [229] cos_sim(w_trait_raw_space, persona_i)

    raw_top, raw_bottom = top_bottom(raw_align, trait_names, args.n_top)
    trait_top, trait_bottom = top_bottom(trait_align, trait_names, args.n_top)

    print("\n" + "=" * 90)
    print("  W_RAW vs W_TRAIT (both already-saved, already-fitted classifiers -- no refitting)")
    print("=" * 90)
    print(f"  Overall cos_sim(w_raw, w_trait_in_raw_space) = {overall_sim:.4f}")

    print(f"\n  Top-{args.n_top} personas aligned with w_raw (|cos_sim|):")
    for name, v in raw_top:
        print(f"    {name:20s} {v:+.4f}")
    print(f"  Most orthogonal to w_raw:")
    for name, v in raw_bottom:
        print(f"    {name:20s} {v:+.4f}")

    print(f"\n  Top-{args.n_top} personas aligned with w_trait's raw-space direction:")
    for name, v in trait_top:
        print(f"    {name:20s} {v:+.4f}")
    print(f"  Most orthogonal to w_trait's raw-space direction:")
    for name, v in trait_bottom:
        print(f"    {name:20s} {v:+.4f}")
    print("=" * 90)

    out = {
        "method": "raw_vs_trait_classifier_alignment",
        "w_raw_source": args.hyperplane_path,
        "w_trait_source": args.trait_model_path,
        "w_trait_model_name": trait_model["meta"]["model_name"],
        "overall_cos_sim": overall_sim,
        "w_raw_top_aligned": raw_top,
        "w_raw_most_orthogonal": raw_bottom,
        "w_trait_top_aligned": trait_top,
        "w_trait_most_orthogonal": trait_bottom,
        "all_personas": {
            name: {"w_raw_align": float(raw_align[i]), "w_trait_align": float(trait_align[i])}
            for i, name in enumerate(trait_names)
        },
    }
    out_path = out_dir / "results.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
