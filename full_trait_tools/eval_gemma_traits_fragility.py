#!/usr/bin/env python3
"""Fragility / honesty check for the Gemma Option-A separation result.

For each subset (human_jailbreak / direct_request / pooled):
  - Assistant axis (single fixed direction, no training):
      * point AUC + AP,
      * bootstrap 95% CI (resample pairs with replacement),
      * a permutation null (shuffle labels) to show the chance AUC spread,
      * the ACTUAL precision you'd get at recall=0.5 and the absolute false-positive
        count -- the deployment number ROC hides under imbalance.
  - Multi-trait logreg: repeated stratified CV over many fold seeds, to show how
    much the 0.96 wobbles with fold assignment.
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
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LAYER_KEY = "30"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_trait_matrix(vec_dir):
    vecs, names = [], []
    for f in sorted(glob.glob(os.path.join(vec_dir, "*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        v = d["vector"][0].float().numpy()
        n = np.linalg.norm(v)
        if n > 1e-8:
            vecs.append(v / n)
            names.append(Path(f).stem)
    return np.stack(vecs).astype(np.float32), names


def load_axis(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    v = d["axis"][0].float().numpy()
    return (v / np.linalg.norm(v)).astype(np.float32)


def orient(y, s):
    """Flip score so higher = more positive (signed AUC >= 0.5)."""
    return -s if roc_auc_score(y, s) < 0.5 else s


def bootstrap_ci(y, s, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs, aps = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() < 2 or yb.sum() > len(yb) - 1:
            continue
        sb = s[idx]
        aucs.append(roc_auc_score(yb, sb))
        aps.append(average_precision_score(yb, sb))
    pct = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return pct(aucs), pct(aps), len(aucs)


def perm_null(y, s, n_perm=4000, seed=0):
    rng = np.random.default_rng(seed)
    yc = y.copy()
    null = []
    for _ in range(n_perm):
        rng.shuffle(yc)
        null.append(roc_auc_score(yc, s))
    null = np.array(null)
    obs = roc_auc_score(y, s)
    p = float((np.sum(null >= obs) + 1) / (len(null) + 1))
    return (float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))), p


def precision_at_recall(y, s, target=0.5):
    prec, rec, thr = precision_recall_curve(y, s)
    # points with recall >= target; pick the one with highest precision
    ok = rec >= target
    if not ok.any():
        return float("nan"), float("nan"), -1
    best = np.argmax(np.where(ok, prec, -1))
    p = float(prec[best])
    r = float(rec[best])
    n_pos = int(y.sum())
    tp = r * n_pos
    fp = tp * (1.0 / p - 1.0) if p > 0 else float("nan")
    return p, r, int(round(fp))


def repeated_cv(X, y, n_seeds=25):
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    k = min(5, n_pos, n_neg)
    if k < 2:
        return None
    aucs, aps = [], []
    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        pipe = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, class_weight="balanced",
                                       max_iter=4000, random_state=seed)),
        ])
        oof = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, oof))
        aps.append(average_precision_score(y, oof))
    return {
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "auc_min": float(np.min(aucs)), "auc_max": float(np.max(aucs)),
        "ap_mean": float(np.mean(aps)), "ap_std": float(np.std(aps)),
        "ap_min": float(np.min(aps)), "ap_max": float(np.max(aps)),
        "k": k, "n_seeds": n_seeds,
    }


def main():
    ap = argparse.ArgumentParser()
    base = "/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B"
    ap.add_argument("--activations", default=f"{base}/full_trait_output/harmbench_activations_gemma/activations.pt")
    ap.add_argument("--classified", default=f"{base}/full_trait_output/harmbench_activations_gemma/classified_responses.jsonl")
    ap.add_argument("--vectors_dir", default=f"{base}/gemma_trait_output/traits40_vectors_layer30_only/pre_generation_last_token/all_traits_no_filter")
    ap.add_argument("--axis_path", default=f"{base}/gemma_trait_output/traits40_axes_layer30_only/pre_generation_last_token/all_traits_no_filter/assistant_axis_pc1.pt")
    ap.add_argument("--out", default=f"{base}/full_trait_output/harmbench_activations_gemma/fragility_results.json")
    args = ap.parse_args()

    acts = torch.load(args.activations, map_location="cpu", weights_only=False)
    rows = load_jsonl(args.classified)
    T, _ = load_trait_matrix(args.vectors_dir)
    axis = load_axis(args.axis_path)

    X, y, atk = [], [], []
    for r in rows:
        item = acts.get(r.get("pair_id"))
        lab = r.get("jailbroken")
        if item is None or LAYER_KEY not in item or lab is None:
            continue
        X.append(item[LAYER_KEY].float().numpy())
        y.append(1 if lab else 0)
        atk.append(r.get("attack_type"))
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    atk = np.array(atk)
    P = (X @ T.T).astype(np.float32)
    axproj = (X @ axis).astype(np.float32)

    subsets = {
        "human_jailbreak": atk == "human_jailbreak",
        "direct_request": atk == "direct_request",
        "pooled": np.ones(len(y), dtype=bool),
    }

    results = {}
    for sname, mask in subsets.items():
        yy = y[mask]
        n, npos = int(mask.sum()), int(yy.sum())
        if npos < 1 or npos >= n:           # degenerate subset (e.g. WJB has one attack_type)
            print(f"\n=== {sname}  (n={n}, pos={npos}) — skipped (single class) ===", flush=True)
            continue
        s = orient(yy, axproj[mask])
        auc = float(roc_auc_score(yy, s))
        apv = float(average_precision_score(yy, s))
        (auc_lo, auc_hi), (ap_lo, ap_hi), nb = bootstrap_ci(yy, s)
        (null_lo, null_hi), pval = perm_null(yy, s)
        prec, rec, fp = precision_at_recall(yy, s, 0.5)
        cv = repeated_cv(P[mask], yy)

        results[sname] = {
            "n": n, "n_pos": npos, "base_rate": npos / n,
            "axis_auc": auc, "axis_auc_ci95": [auc_lo, auc_hi],
            "axis_ap": apv, "axis_ap_ci95": [ap_lo, ap_hi],
            "perm_null_auc_ci95": [null_lo, null_hi], "perm_pvalue": pval,
            "axis_precision_at_recall0.5": prec, "axis_actual_recall": rec,
            "axis_false_positives_at_recall0.5": fp,
            "multitrait_cv": cv,
        }

        print(f"\n=== {sname}  (n={n}, pos={npos}, base rate={npos/n:.3f}) ===", flush=True)
        print(f"  assistant axis AUC = {auc:.3f}  95% CI [{auc_lo:.3f}, {auc_hi:.3f}]", flush=True)
        print(f"  assistant axis AP  = {apv:.3f}  95% CI [{ap_lo:.3f}, {ap_hi:.3f}]   (chance AP = {npos/n:.3f})", flush=True)
        print(f"  permutation null AUC 95% band [{null_lo:.3f}, {null_hi:.3f}]   p = {pval:.4f}", flush=True)
        print(f"  DEPLOYMENT: at recall {rec:.2f} (catch ~half the jailbreaks) -> precision {prec:.3f}, {fp} false positives", flush=True)
        if cv:
            print(f"  multi-trait CV logreg over {cv['n_seeds']} fold-seeds: "
                  f"AUC {cv['auc_mean']:.3f}±{cv['auc_std']:.3f} (range {cv['auc_min']:.3f}-{cv['auc_max']:.3f}), "
                  f"AP {cv['ap_mean']:.3f}±{cv['ap_std']:.3f} (range {cv['ap_min']:.3f}-{cv['ap_max']:.3f})", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
