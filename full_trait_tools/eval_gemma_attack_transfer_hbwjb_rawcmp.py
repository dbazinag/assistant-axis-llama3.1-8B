#!/usr/bin/env python3
"""HB+WJB-pooled transfer: RAW-5376 vs TRAIT-240, all 4 held-out attacks.

Same as eval_gemma_attack_transfer_hb_only_rawcmp.py but trains on the pooled
HarmBench + WildJailbreak in-distribution set (the design's intended training
mix). Lets us see how much the WJB positives help transfer over HB-only.
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
    ap.add_argument("--wjb_acts", default=f"{fo}/wildjailbreak_activations_gemma/activations.pt")
    ap.add_argument("--wjb_clf", default=f"{fo}/wildjailbreak_activations_gemma/classified_responses.jsonl")
    ap.add_argument("--gptfuzz_acts", default=f"{fo}/gptfuzz_activations_gemma_hb/activations.pt")
    ap.add_argument("--gptfuzz_clf", default=f"{fo}/gptfuzz_activations_gemma_hb/classified_responses.jsonl")
    ap.add_argument("--pair_acts", default=f"{fo}/pair_activations_gemma_hb/activations.pt")
    ap.add_argument("--pair_clf", default=f"{fo}/pair_activations_gemma_hb/classified_responses.jsonl")
    ap.add_argument("--pap_acts", default=f"{fo}/pap_activations_gemma_hb/activations.pt")
    ap.add_argument("--pap_clf", default=f"{fo}/pap_activations_gemma_hb/classified_responses.jsonl")
    ap.add_argument("--pez_acts", default=f"{fo}/pez_activations_gemma_hb/activations.pt")
    ap.add_argument("--pez_clf", default=f"{fo}/pez_activations_gemma_hb/classified_responses.jsonl")
    ap.add_argument("--vectors_dir", default=f"{base}/gemma_trait_output/traits40_vectors_layer30_only/pre_generation_last_token/all_traits_no_filter")
    ap.add_argument("--out", default=f"{fo}/attack_transfer_hbwjb_rawcmp_results.json")
    args = ap.parse_args()

    T = load_trait_matrix(args.vectors_dir)
    print(f"trait matrix {T.shape}", flush=True)

    X_hb, y_hb = load_raw_xy(args.hb_acts, args.hb_clf)
    X_wjb, y_wjb = load_raw_xy(args.wjb_acts, args.wjb_clf)
    X_tr = np.concatenate([X_hb, X_wjb], 0)
    y_tr = np.concatenate([y_hb, y_wjb], 0)
    print(f"TRAIN HB+WJB raw X {X_tr.shape}  pos={int(y_tr.sum())}  base={y_tr.mean():.3f} "
          f"(HB pos={int(y_hb.sum())} | WJB pos={int(y_wjb.sum())})", flush=True)

    tests = [
        ("HB+WJB->GPTFuzz", args.gptfuzz_acts, args.gptfuzz_clf),
        ("HB+WJB->PAIR", args.pair_acts, args.pair_clf),
        ("HB+WJB->PAP", args.pap_acts, args.pap_clf),
        ("HB+WJB->PEZ", args.pez_acts, args.pez_clf),
    ]

    results = {}
    header = f"{'transfer':17s} {'base':>6s} | {'RAW-5376 AUC':>12s} {'AP':>6s} | {'TRAIT-240 AUC':>13s} {'AP':>6s}"
    print("\n" + header, flush=True)
    print("-" * len(header), flush=True)
    P_tr = X_tr @ T.T
    for name, a, c in tests:
        X_te, y_te = load_raw_xy(a, c)
        raw_auc, raw_ap = fit_eval(X_tr, y_tr, X_te, y_te)
        tr_auc, tr_ap = fit_eval(P_tr, y_tr, X_te @ T.T, y_te)
        b = float(y_te.mean())
        results[name] = {
            "test_n": int(len(y_te)), "test_pos": int(y_te.sum()), "base_rate": b,
            "raw5376": {"auc": raw_auc, "ap": raw_ap},
            "trait240": {"auc": tr_auc, "ap": tr_ap},
        }
        print(f"{name:17s} {b:6.3f} | {raw_auc:12.3f} {raw_ap:6.3f} | {tr_auc:13.3f} {tr_ap:6.3f}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
