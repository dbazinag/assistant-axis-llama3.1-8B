#!/usr/bin/env python3
"""Raw activations vs trait-projection detector, compared on transfer.

The Gemma detector so far trains a logreg on the 240-dim trait projections
(P = x @ T.T). This asks the missing baseline question: does projecting onto the
240 trait/persona directions HELP, or does it discard signal a logreg on the raw
5376-dim layer-30 activation would keep?

Same pipeline for both feature sets (StandardScaler + LogisticRegression C=0.1,
class_weight=balanced) so the only difference is raw-5376 vs trait-240. We judge
on TRANSFER (train pooled HB+WJB, test a held-out attack family), because raw
features have 22x more dimensions and will look better in-distribution by
overfitting (261 positives, p >> n) — transfer is the honest test.

  train HB+WJB  ->  GPTFuzz
  train HB+WJB  ->  PAP
  HarmBench     ->  WJB        (reference cross-transfer)
  WJB           ->  HarmBench  (reference cross-transfer)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eval_gemma_cross_attack_transfer import (
    LAYER_KEY, load_axis, load_jsonl, load_trait_matrix,
)


def load_raw_xy(acts_path: str, clf_path: str):
    """Raw layer-30 activations X (N,5376) and labels y (N,)."""
    acts = torch.load(acts_path, map_location="cpu", weights_only=False)
    rows = load_jsonl(clf_path)
    X, y = [], []
    for r in rows:
        item = acts.get(r.get("pair_id"))
        lab = r.get("jailbroken")
        if item is None or LAYER_KEY not in item or lab is None:
            continue
        X.append(item[LAYER_KEY].float().numpy())
        y.append(1 if lab else 0)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)


def fit_eval(X_tr, y_tr, X_te, y_te, seed=42):
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, class_weight="balanced",
                                   max_iter=5000, random_state=seed)),
    ])
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, proba)), float(average_precision_score(y_te, proba))


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
    ap.add_argument("--pap_acts", default=f"{fo}/pap_activations_gemma_hb/activations.pt")
    ap.add_argument("--pap_clf", default=f"{fo}/pap_activations_gemma_hb/classified_responses.jsonl")
    ap.add_argument("--vectors_dir", default=f"{base}/gemma_trait_output/traits40_vectors_layer30_only/pre_generation_last_token/all_traits_no_filter")
    ap.add_argument("--out", default=f"{fo}/raw_vs_traits_transfer_results.json")
    args = ap.parse_args()

    T = load_trait_matrix(args.vectors_dir)          # (240, 5376)
    print(f"trait matrix {T.shape}", flush=True)

    sets = {}
    for name, a, c in [
        ("HB", args.hb_acts, args.hb_clf),
        ("WJB", args.wjb_acts, args.wjb_clf),
        ("GPTFuzz", args.gptfuzz_acts, args.gptfuzz_clf),
        ("PAP", args.pap_acts, args.pap_clf),
    ]:
        X, y = load_raw_xy(a, c)
        sets[name] = (X, y)
        print(f"{name:8s} raw X {X.shape}  pos={int(y.sum())}  base={y.mean():.3f}", flush=True)

    X_hb, y_hb = sets["HB"]
    X_wjb, y_wjb = sets["WJB"]
    X_pool = np.concatenate([X_hb, X_wjb], 0)
    y_pool = np.concatenate([y_hb, y_wjb], 0)

    transfers = [
        ("HB+WJB->GPTFuzz", X_pool, y_pool, *sets["GPTFuzz"]),
        ("HB+WJB->PAP", X_pool, y_pool, *sets["PAP"]),
        ("HB->WJB", X_hb, y_hb, *sets["WJB"]),
        ("WJB->HB", X_wjb, y_wjb, X_hb, y_hb),
    ]

    results = {}
    header = f"{'transfer':18s} {'base':>6s} | {'RAW-5376 AUC':>12s} {'AP':>6s} | {'TRAIT-240 AUC':>13s} {'AP':>6s}"
    print("\n" + header, flush=True)
    print("-" * len(header), flush=True)
    for name, Xtr, ytr, Xte, yte in transfers:
        raw_auc, raw_ap = fit_eval(Xtr, ytr, Xte, yte)
        Ptr, Pte = Xtr @ T.T, Xte @ T.T
        tr_auc, tr_ap = fit_eval(Ptr, ytr, Pte, yte)
        base = float(yte.mean())
        results[name] = {
            "test_n": int(len(yte)), "test_pos": int(yte.sum()), "base_rate": base,
            "raw5376": {"auc": raw_auc, "ap": raw_ap},
            "trait240": {"auc": tr_auc, "ap": tr_ap},
        }
        print(f"{name:18s} {base:6.3f} | {raw_auc:12.3f} {raw_ap:6.3f} | {tr_auc:13.3f} {tr_ap:6.3f}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
