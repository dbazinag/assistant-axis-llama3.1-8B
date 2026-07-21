#!/usr/bin/env python3
"""HarmBench-only transfer: RAW-5376 vs TRAIT-240 features, all 4 held-out attacks.

Companion to eval_gemma_attack_transfer_hb_only.py. Same train set (HarmBench
only) and same logreg pipeline, but compares the raw 5376-dim layer-30 activation
against the 240-dim trait projection (P = x @ T.T) head-to-head on transfer to
GPTFuzz / PAIR / PAP / PEZ.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_gemma_cross_attack_transfer import load_trait_matrix
from eval_gemma_raw_vs_traits_transfer import fit_eval, load_raw_xy


def main() -> None:
    ap = argparse.ArgumentParser()
    base = "/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B"
    fo = f"{base}/full_trait_output"
    ap.add_argument("--hb_acts", default=f"{fo}/harmbench_activations_gemma/activations.pt")
    ap.add_argument("--hb_clf", default=f"{fo}/harmbench_activations_gemma/classified_responses.jsonl")
    ap.add_argument("--gptfuzz_acts", default=f"{fo}/gptfuzz_activations_gemma_hb/activations.pt")
    ap.add_argument("--gptfuzz_clf", default=f"{fo}/gptfuzz_activations_gemma_hb/classified_responses.jsonl")
    ap.add_argument("--pair_acts", default=f"{fo}/pair_activations_gemma_hb/activations.pt")
    ap.add_argument("--pair_clf", default=f"{fo}/pair_activations_gemma_hb/classified_responses.jsonl")
    ap.add_argument("--pap_acts", default=f"{fo}/pap_activations_gemma_hb/activations.pt")
    ap.add_argument("--pap_clf", default=f"{fo}/pap_activations_gemma_hb/classified_responses.jsonl")
    ap.add_argument("--pez_acts", default=f"{fo}/pez_activations_gemma_hb/activations.pt")
    ap.add_argument("--pez_clf", default=f"{fo}/pez_activations_gemma_hb/classified_responses.jsonl")
    ap.add_argument("--vectors_dir", default=f"{base}/gemma_trait_output/traits40_vectors_layer30_only/pre_generation_last_token/all_traits_no_filter")
    ap.add_argument("--out", default=f"{fo}/attack_transfer_hb_only_rawcmp_results.json")
    args = ap.parse_args()

    T = load_trait_matrix(args.vectors_dir)
    print(f"trait matrix {T.shape}", flush=True)

    X_hb, y_hb = load_raw_xy(args.hb_acts, args.hb_clf)
    print(f"TRAIN HarmBench-only raw X {X_hb.shape}  pos={int(y_hb.sum())}  base={y_hb.mean():.3f}", flush=True)

    tests = [
        ("HB->GPTFuzz", args.gptfuzz_acts, args.gptfuzz_clf),
        ("HB->PAIR", args.pair_acts, args.pair_clf),
        ("HB->PAP", args.pap_acts, args.pap_clf),
        ("HB->PEZ", args.pez_acts, args.pez_clf),
    ]

    results = {}
    header = f"{'transfer':14s} {'base':>6s} | {'RAW-5376 AUC':>12s} {'AP':>6s} | {'TRAIT-240 AUC':>13s} {'AP':>6s}"
    print("\n" + header, flush=True)
    print("-" * len(header), flush=True)
    P_hb = X_hb @ T.T
    for name, a, c in tests:
        X_te, y_te = load_raw_xy(a, c)
        raw_auc, raw_ap = fit_eval(X_hb, y_hb, X_te, y_te)
        tr_auc, tr_ap = fit_eval(P_hb, y_hb, X_te @ T.T, y_te)
        base = float(y_te.mean())
        results[name] = {
            "test_n": int(len(y_te)), "test_pos": int(y_te.sum()), "base_rate": base,
            "raw5376": {"auc": raw_auc, "ap": raw_ap},
            "trait240": {"auc": tr_auc, "ap": tr_ap},
        }
        print(f"{name:14s} {base:6.3f} | {raw_auc:12.3f} {raw_ap:6.3f} | {tr_auc:13.3f} {tr_ap:6.3f}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
