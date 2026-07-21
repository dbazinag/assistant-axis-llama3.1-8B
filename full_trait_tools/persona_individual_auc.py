#!/usr/bin/env python3
"""
persona_individual_auc.py

For each persona/trait vector individually, measure its standalone predictive
power (AUC) for jailbreak detection on HarmBench, averaged over the same 50
seeds/held-out pool splits used by run_all_traits_sweep_v2.py.

Since AUC is invariant to monotonic rescaling of a single scalar feature, this
skips fitting a classifier per persona per seed and instead computes safe_auc
directly on each persona's raw projection score within the held-out test pool.
This is numerically identical to fitting a 1-feature LogisticRegression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import roc_auc_score

from run_all_traits_sweep_v2 import (
    build_activation_matrix,
    get_pool_split,
    load_jsonl,
    load_trait_matrix,
    project_all_traits,
    safe_auc,
    split_by_pool,
)

LAYER = 16
TRAIN_FRAC = 0.7
N_SEEDS = 50


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_classified_path",
                        default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--human_activations_path",
                        default="full_trait_output/harmbench_activations/activations.pt")
    parser.add_argument("--output_dir", default="full_trait_output/persona_individual_auc")
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS)
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="Added to each seed in range(n_seeds), so a different draw of "
                             "held-out splits can be evaluated without reusing seeds 0..49.")
    parser.add_argument("--train_frac", type=float, default=TRAIN_FRAC)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading data ===", flush=True)
    human_rows = [
        r for r in load_jsonl(Path(args.human_classified_path))
        if r.get("attack_type") == "human_jailbreak"
    ]
    human_acts = torch.load(Path(args.human_activations_path), map_location="cpu", weights_only=False)

    trait_matrix, trait_names = load_trait_matrix(args.layer)
    print(f"  Trait matrix: {trait_matrix.shape}", flush=True)

    x_raw_h, y_h, human_valid = build_activation_matrix(human_rows, human_acts, args.layer)
    x_h = project_all_traits(x_raw_h, trait_matrix)
    print(f"  HarmBench: {x_h.shape}, jb={y_h.mean():.3f}", flush=True)

    # Per-persona AUC per seed, using the identical held-out pool split as the main sweep.
    per_persona_aucs: list[list[float]] = [[] for _ in trait_names]
    per_persona_raw_aucs: list[list[float]] = [[] for _ in trait_names]
    n_used_seeds = 0
    for seed in range(args.seed_offset, args.seed_offset + args.n_seeds):
        train_beh, train_tpl, test_beh, test_tpl = get_pool_split(
            human_valid, args.train_frac, seed
        )
        train_idx, test_idx = split_by_pool(
            human_valid, train_beh, train_tpl, test_beh, test_tpl
        )
        if not train_idx or not test_idx or len(np.unique(y_h[train_idx])) < 2:
            continue
        n_used_seeds += 1
        y_test = y_h[test_idx]
        x_test = x_h[test_idx]
        for i in range(len(trait_names)):
            per_persona_aucs[i].append(safe_auc(y_test, x_test[:, i]))
            per_persona_raw_aucs[i].append(float(roc_auc_score(y_test, x_test[:, i])))

        if seed % 10 == 0:
            print(f"  Seed {seed}/{args.n_seeds}", flush=True)

    print(f"\n  Used {n_used_seeds}/{args.n_seeds} seeds", flush=True)

    results = []
    for name, aucs, raw_aucs in zip(trait_names, per_persona_aucs, per_persona_raw_aucs):
        arr = np.array(aucs, dtype=float)
        raw_arr = np.array(raw_aucs, dtype=float)
        raw_mean = float(np.nanmean(raw_arr))
        results.append({
            "trait": name,
            "mean_auc": float(np.nanmean(arr)),
            "std_auc": float(np.nanstd(arr)),
            "n_seeds": int(np.sum(~np.isnan(arr))),
            "raw_auc_mean": raw_mean,
            "direction": "high_score_predicts_jailbreak" if raw_mean >= 0.5 else "high_score_predicts_refusal",
        })

    results.sort(key=lambda r: r["mean_auc"], reverse=True)

    print("\n" + "=" * 60)
    print(f"  PER-PERSONA STANDALONE AUC | layer {args.layer} | {n_used_seeds} seeds")
    print("=" * 60)
    print(f"  {'Trait':45s}{'MeanAUC':>10s}{'StdAUC':>10s}")
    print("  " + "-" * 56)
    for r in results[:30]:
        print(f"  {r['trait']:45s}{r['mean_auc']:10.4f}{r['std_auc']:10.4f}")
    print("=" * 60)

    out = {
        "method": "persona_individual_auc",
        "layer": args.layer,
        "n_seeds": args.n_seeds,
        "n_seeds_used": n_used_seeds,
        "n_traits": len(trait_names),
        "ranked": results,
    }
    out_path = out_dir / "results.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
