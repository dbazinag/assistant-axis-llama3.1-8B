#!/usr/bin/env python3
"""HarmBench-only attack-transfer eval for the Gemma-4-31B trait detector.

Train the multi-trait logreg on HarmBench alone (no WildJailbreak) and test
honest transfer to each held-out attack family the detector never saw:

  HarmBench -> GPTFuzz | PAIR | PAP | PEZ

Each test reports AUC + AP (vs the test set's base rate). The assistant axis is
a fixed direction (no training); its per-set unsigned AUC is shown for reference.
Reuses the projection + model recipe from eval_gemma_cross_attack_transfer.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from eval_gemma_cross_attack_transfer import (
    build_xy, fit_predict, load_axis, load_trait_matrix,
)


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
    ap.add_argument("--axis_path", default=f"{base}/gemma_trait_output/traits40_axes_layer30_only/pre_generation_last_token/all_traits_no_filter/assistant_axis_pc1.pt")
    ap.add_argument("--out", default=f"{fo}/attack_transfer_hb_only_results.json")
    args = ap.parse_args()

    T = load_trait_matrix(args.vectors_dir)
    axis = load_axis(args.axis_path)
    print(f"trait matrix {T.shape}, axis {axis.shape}", flush=True)

    P_hb, _, y_hb = build_xy(args.hb_acts, args.hb_clf, T, axis)
    P_gf, ax_gf, y_gf = build_xy(args.gptfuzz_acts, args.gptfuzz_clf, T, axis)
    P_pair, ax_pair, y_pair = build_xy(args.pair_acts, args.pair_clf, T, axis)
    P_pap, ax_pap, y_pap = build_xy(args.pap_acts, args.pap_clf, T, axis)
    P_pez, ax_pez, y_pez = build_xy(args.pez_acts, args.pez_clf, T, axis)

    print(f"TRAIN HarmBench-only: n={len(y_hb)}, pos={int(y_hb.sum())}, base={y_hb.mean():.3f}", flush=True)

    results = {"train": {"set": "HarmBench", "n": len(y_hb),
                         "pos": int(y_hb.sum()), "base_rate": float(y_hb.mean())}}
    for name, P_te, y_te, ax_te in [
        ("HB->GPTFuzz", P_gf, y_gf, ax_gf),
        ("HB->PAIR", P_pair, y_pair, ax_pair),
        ("HB->PAP", P_pap, y_pap, ax_pap),
        ("HB->PEZ", P_pez, y_pez, ax_pez),
    ]:
        proba = fit_predict(P_hb, y_hb, P_te)
        auc = float(roc_auc_score(y_te, proba))
        apv = float(average_precision_score(y_te, proba))
        a = float(roc_auc_score(y_te, ax_te))
        ax_auc = max(a, 1.0 - a)
        base = float(y_te.mean())
        results[name] = {
            "test_n": len(y_te), "test_pos": int(y_te.sum()), "test_base_rate": base,
            "transfer_logreg_auc": auc, "transfer_logreg_ap": apv,
            "axis_unsigned_auc_on_test": ax_auc,
        }
        print(f"\n=== {name} ===", flush=True)
        print(f"  test: n={len(y_te)}, pos={int(y_te.sum())}, base={base:.3f}", flush=True)
        print(f"  TRANSFER logreg : AUC={auc:.3f}  AP={apv:.3f}  (chance AP={base:.3f})", flush=True)
        print(f"  (ref) assistant axis on test : unsigned AUC={ax_auc:.3f}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
