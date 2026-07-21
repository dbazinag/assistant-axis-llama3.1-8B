#!/usr/bin/env python3
"""Config test: re-run all three training mixes with a plain logreg.

LogisticRegression(C=1.0, max_iter=2000, random_state=42), L2/lbfgs, NO class
weighting (vs the project default C=0.1, class_weight="balanced"). Raw-5376 vs
trait-240, for HB-only / HB+WJB / HB+WJB+GPTFuzz. One load, all mixes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eval_gemma_cross_attack_transfer import load_trait_matrix
from eval_gemma_raw_vs_traits_transfer import load_raw_xy

BASE = "/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B"
FO = f"{BASE}/full_trait_output"
VEC = f"{BASE}/gemma_trait_output/traits40_vectors_layer30_only/pre_generation_last_token/all_traits_no_filter"

SETS = {
    "HB":      (f"{FO}/harmbench_activations_gemma/activations.pt",     f"{FO}/harmbench_activations_gemma/classified_responses.jsonl"),
    "WJB":     (f"{FO}/wildjailbreak_activations_gemma/activations.pt", f"{FO}/wildjailbreak_activations_gemma/classified_responses.jsonl"),
    "GPTFuzz": (f"{FO}/gptfuzz_activations_gemma_hb/activations.pt",    f"{FO}/gptfuzz_activations_gemma_hb/classified_responses.jsonl"),
    "PAIR":    (f"{FO}/pair_activations_gemma_hb/activations.pt",       f"{FO}/pair_activations_gemma_hb/classified_responses.jsonl"),
    "PAP":     (f"{FO}/pap_activations_gemma_hb/activations.pt",        f"{FO}/pap_activations_gemma_hb/classified_responses.jsonl"),
    "PEZ":     (f"{FO}/pez_activations_gemma_hb/activations.pt",        f"{FO}/pez_activations_gemma_hb/classified_responses.jsonl"),
}


def fit_eval(X_tr, y_tr, X_te, y_te):
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
    ])
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, proba)), float(average_precision_score(y_te, proba))


def main() -> None:
    T = load_trait_matrix(VEC)
    print(f"trait matrix {T.shape}  |  MODEL: LogReg(C=1.0, max_iter=2000, no class_weight)", flush=True)

    Xy = {name: load_raw_xy(a, c) for name, (a, c) in SETS.items()}
    for name, (X, y) in Xy.items():
        print(f"  {name:8s} n={len(y):5d} pos={int(y.sum()):4d} base={y.mean():.3f}", flush=True)

    mixes = [
        ("HB",          ["HB"],                  ["GPTFuzz", "PAIR", "PAP", "PEZ"]),
        ("HB+WJB",       ["HB", "WJB"],           ["GPTFuzz", "PAIR", "PAP", "PEZ"]),
        ("HB+WJB+GPTFuzz", ["HB", "WJB", "GPTFuzz"], ["PAIR", "PAP", "PEZ"]),
    ]

    out = {}
    for mix_name, train_keys, test_keys in mixes:
        X_tr = np.concatenate([Xy[k][0] for k in train_keys], 0)
        y_tr = np.concatenate([Xy[k][1] for k in train_keys], 0)
        P_tr = X_tr @ T.T
        print(f"\n===== TRAIN {mix_name}  n={len(y_tr)} pos={int(y_tr.sum())} base={y_tr.mean():.3f} =====", flush=True)
        header = f"{'-> test':10s} {'base':>6s} | {'RAW AUC':>8s} {'AP':>6s} | {'TRAIT AUC':>9s} {'AP':>6s}"
        print(header, flush=True)
        print("-" * len(header), flush=True)
        rows = {}
        for tk in test_keys:
            X_te, y_te = Xy[tk]
            r_auc, r_ap = fit_eval(X_tr, y_tr, X_te, y_te)
            t_auc, t_ap = fit_eval(P_tr, y_tr, X_te @ T.T, y_te)
            rows[tk] = {"base_rate": float(y_te.mean()),
                        "raw5376": {"auc": r_auc, "ap": r_ap},
                        "trait240": {"auc": t_auc, "ap": t_ap}}
            print(f"{tk:10s} {y_te.mean():6.3f} | {r_auc:8.3f} {r_ap:6.3f} | {t_auc:9.3f} {t_ap:6.3f}", flush=True)
        avg = {fs: {m: float(np.mean([rows[k][fs][m] for k in rows])) for m in ("auc", "ap")}
               for fs in ("raw5376", "trait240")}
        avg_base = float(np.mean([rows[k]["base_rate"] for k in rows]))
        print(f"{'AVERAGE':10s} {avg_base:6.3f} | {avg['raw5376']['auc']:8.3f} {avg['raw5376']['ap']:6.3f} | "
              f"{avg['trait240']['auc']:9.3f} {avg['trait240']['ap']:6.3f}", flush=True)
        rows["AVERAGE"] = {"base_rate": avg_base, **avg}
        out[mix_name] = rows

    Path(f"{FO}/attack_transfer_cfgtest_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {FO}/attack_transfer_cfgtest_results.json", flush=True)


if __name__ == "__main__":
    main()
