#!/usr/bin/env python3
"""Cross-attack transfer for the Gemma-4-31B trait detector.

Within-attack AUCs (HarmBench leak-free ~0.94, WJB leak-free ~0.89) can't tell
whether the detector learns attack-general jailbreak structure or attack-specific
artifacts. This script trains the multi-trait logreg on ONE attack's pooled
labels and tests on the OTHER attack (fully disjoint data = honest transfer):

  - HarmBench-trained  -> WildJailbreak-tested
  - WildJailbreak-trained -> HarmBench-tested

Both directions report AUC + AP (vs the test set's base rate). The assistant
axis is a fixed direction (no training), so its per-set AUC is shown for
reference but doesn't "transfer" in the trained sense.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LAYER_KEY = "30"


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_trait_matrix(vec_dir: str) -> np.ndarray:
    vecs = []
    for f in sorted(glob.glob(os.path.join(vec_dir, "*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        v = d["vector"][0].float().numpy()  # row 0 == model layer 30
        n = np.linalg.norm(v)
        if n > 1e-8:
            vecs.append(v / n)
    return np.stack(vecs).astype(np.float32)


def load_axis(axis_path: str) -> np.ndarray:
    d = torch.load(axis_path, map_location="cpu", weights_only=False)
    v = d["axis"][0].float().numpy()
    return (v / np.linalg.norm(v)).astype(np.float32)


def build_xy(acts_path: str, clf_path: str, T: np.ndarray, axis: np.ndarray):
    """Return trait projections P (N,240), axis projection (N,), labels y (N,)."""
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
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    P = (X @ T.T).astype(np.float32)
    axproj = (X @ axis).astype(np.float32)
    return P, axproj, y


def fit_predict(P_tr, y_tr, P_te, seed=42):
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, class_weight="balanced",
                                   max_iter=4000, random_state=seed)),
    ])
    pipe.fit(P_tr, y_tr)
    return pipe.predict_proba(P_te)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    base = "/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B"
    ap.add_argument("--hb_acts", default=f"{base}/full_trait_output/harmbench_activations_gemma/activations.pt")
    ap.add_argument("--hb_clf", default=f"{base}/full_trait_output/harmbench_activations_gemma/classified_responses.jsonl")
    ap.add_argument("--wjb_acts", default=f"{base}/full_trait_output/wildjailbreak_activations_gemma/activations.pt")
    ap.add_argument("--wjb_clf", default=f"{base}/full_trait_output/wildjailbreak_activations_gemma/classified_responses.jsonl")
    ap.add_argument("--vectors_dir", default=f"{base}/gemma_trait_output/traits40_vectors_layer30_only/pre_generation_last_token/all_traits_no_filter")
    ap.add_argument("--axis_path", default=f"{base}/gemma_trait_output/traits40_axes_layer30_only/pre_generation_last_token/all_traits_no_filter/assistant_axis_pc1.pt")
    ap.add_argument("--out", default=f"{base}/full_trait_output/cross_attack_transfer_results.json")
    args = ap.parse_args()

    T = load_trait_matrix(args.vectors_dir)
    axis = load_axis(args.axis_path)
    print(f"trait matrix {T.shape}, axis {axis.shape}", flush=True)

    P_hb, ax_hb, y_hb = build_xy(args.hb_acts, args.hb_clf, T, axis)
    P_wjb, ax_wjb, y_wjb = build_xy(args.wjb_acts, args.wjb_clf, T, axis)
    print(f"HarmBench:    n={len(y_hb)}, pos={int(y_hb.sum())}, base={y_hb.mean():.3f}", flush=True)
    print(f"WildJailbreak: n={len(y_wjb)}, pos={int(y_wjb.sum())}, base={y_wjb.mean():.3f}", flush=True)

    results = {}
    transfers = [
        ("HarmBench->WJB", P_hb, y_hb, P_wjb, y_wjb, ax_wjb),
        ("WJB->HarmBench", P_wjb, y_wjb, P_hb, y_hb, ax_hb),
    ]
    for name, P_tr, y_tr, P_te, y_te, ax_te in transfers:
        proba = fit_predict(P_tr, y_tr, P_te)
        auc = float(roc_auc_score(y_te, proba))
        apv = float(average_precision_score(y_te, proba))
        # axis on the test set (fixed direction, sign oriented so higher=positive)
        a = float(roc_auc_score(y_te, ax_te))
        ax_auc = max(a, 1.0 - a)
        base = float(y_te.mean())
        results[name] = {
            "train_n": len(y_tr), "train_pos": int(y_tr.sum()),
            "test_n": len(y_te), "test_pos": int(y_te.sum()), "test_base_rate": base,
            "transfer_logreg_auc": auc, "transfer_logreg_ap": apv,
            "axis_unsigned_auc_on_test": ax_auc,
        }
        print(f"\n=== {name} ===", flush=True)
        print(f"  train: n={len(y_tr)}, pos={int(y_tr.sum())}   test: n={len(y_te)}, pos={int(y_te.sum())}, base={base:.3f}", flush=True)
        print(f"  TRANSFER logreg : AUC={auc:.3f}  AP={apv:.3f}  (chance AP={base:.3f})", flush=True)
        print(f"  (ref) assistant axis on test : unsigned AUC={ax_auc:.3f}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
