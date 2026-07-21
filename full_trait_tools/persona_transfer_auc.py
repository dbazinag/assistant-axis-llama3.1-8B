#!/usr/bin/env python3
"""
persona_transfer_auc.py

For selected persona/trait vectors, measure standalone AUC on transfer attack
families (GCG, PAIR, PAP, GPTFuzz, PEZ, WJB) in addition to the in-distribution
HarmBench human_jailbreak AUC. Since this is a raw single-feature projection
(no classifier fitting), transfer AUC is deterministic per persona -- it does
not vary with the HarmBench train/test seed, unlike a fitted multi-feature
model whose learned weights shift seed to seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

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
N_SEEDS = 101


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_classified_path",
                        default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--human_activations_path",
                        default="full_trait_output/harmbench_activations/activations.pt")
    parser.add_argument("--gcg_classified_path", default="full_trait_output/gcg_activations/responses.jsonl")
    parser.add_argument("--gcg_activations_path", default="full_trait_output/gcg_activations/activations.pt")
    parser.add_argument("--pair_classified_path", default="full_trait_output/pair_activations/responses.jsonl")
    parser.add_argument("--pair_activations_path", default="full_trait_output/pair_activations/activations.pt")
    parser.add_argument("--pap_classified_path", default="full_trait_output/pap_activations/responses.jsonl")
    parser.add_argument("--pap_activations_path", default="full_trait_output/pap_activations/activations.pt")
    parser.add_argument("--gptfuzz_classified_path", default="full_trait_output/gptfuzz_activations/responses.jsonl")
    parser.add_argument("--gptfuzz_activations_path", default="full_trait_output/gptfuzz_activations/activations.pt")
    parser.add_argument("--pez_classified_path", default="full_trait_output/pez_activations/responses.jsonl")
    parser.add_argument("--pez_activations_path", default="full_trait_output/pez_activations/activations.pt")
    parser.add_argument("--wjb_classified_path",
                        default="full_trait_output/wildjailbreak_activations/classified_responses.jsonl")
    parser.add_argument("--wjb_activations_path", default="full_trait_output/wildjailbreak_activations/activations.pt")
    parser.add_argument("--output_dir", default="full_trait_output/persona_transfer_auc")
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS)
    parser.add_argument("--train_frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--traits", default="egalitarian,universalist,principled,progressive,deontological",
                        help="Comma-separated persona names to evaluate.")
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

    transfer_inputs = [
        ("GCG", args.gcg_classified_path, args.gcg_activations_path),
        ("PAIR", args.pair_classified_path, args.pair_activations_path),
        ("PAP", args.pap_classified_path, args.pap_activations_path),
        ("GPTFuzz", args.gptfuzz_classified_path, args.gptfuzz_activations_path),
        ("PEZ", args.pez_classified_path, args.pez_activations_path),
        ("WJB", args.wjb_classified_path, args.wjb_activations_path),
    ]
    transfer_proj: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, rp, ap in transfer_inputs:
        if Path(rp).exists() and Path(ap).exists():
            rows_ = load_jsonl(Path(rp))
            acts_ = torch.load(Path(ap), map_location="cpu", weights_only=False)
            x_raw_t, y_t, _ = build_activation_matrix(rows_, acts_, args.layer)
            if len(x_raw_t) > 0:
                x_t = project_all_traits(x_raw_t, trait_matrix)
                transfer_proj[name] = (x_t, y_t)
                print(f"  {name}: {len(x_raw_t)} rows, jb={y_t.mean():.3f}", flush=True)

    wanted_traits = [t.strip() for t in args.traits.split(",") if t.strip()]
    results = []
    for tname in wanted_traits:
        if tname not in trait_names:
            print(f"  WARNING: trait '{tname}' not found, skipping", flush=True)
            continue
        idx = trait_names.index(tname)

        # In-distribution HarmBench AUC, averaged over held-out test splits (seed-dependent).
        harmbench_aucs = []
        for seed in range(args.n_seeds):
            train_beh, train_tpl, test_beh, test_tpl = get_pool_split(human_valid, args.train_frac, seed)
            train_idx, test_idx = split_by_pool(human_valid, train_beh, train_tpl, test_beh, test_tpl)
            if not train_idx or not test_idx or len(np.unique(y_h[train_idx])) < 2:
                continue
            harmbench_aucs.append(safe_auc(y_h[test_idx], x_h[test_idx, idx]))

        row = {
            "trait": tname,
            "harmbench_test_auc_mean": float(np.nanmean(harmbench_aucs)),
            "harmbench_test_auc_std": float(np.nanstd(harmbench_aucs)),
            "n_seeds": len(harmbench_aucs),
        }
        # Transfer families: deterministic (no fitting => no seed dependence).
        for name, (x_t, y_t) in transfer_proj.items():
            row[f"{name}_auc"] = safe_auc(y_t, x_t[:, idx])
        results.append(row)

    families = list(transfer_proj.keys())
    col_w = 10
    header = f"  {'Trait':16s}{'HB_test':>{col_w}s}" + "".join(f"{f:>{col_w}s}" for f in families)
    print("\n" + "=" * len(header))
    print(f"  PERSONA TRANSFER AUC | layer {args.layer} | HB over {args.n_seeds} seeds, transfer deterministic")
    print("=" * len(header))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        row_str = f"  {r['trait']:16s}{r['harmbench_test_auc_mean']:{col_w}.4f}"
        row_str += "".join(f"{r[f'{f}_auc']:{col_w}.4f}" for f in families)
        print(row_str)
    print("=" * len(header))

    out = {
        "method": "persona_transfer_auc",
        "layer": args.layer,
        "n_seeds": args.n_seeds,
        "families": families,
        "results": results,
    }
    out_path = out_dir / "results.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
