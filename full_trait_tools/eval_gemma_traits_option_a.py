#!/usr/bin/env python3
"""Option A: separation smoke test for Gemma-4-31B.

Project the layer-30 HarmBench activations onto the layer-30 trait vectors and the
assistant axis (PC1), then report how well those directions separate jailbroken
from refused. No heavy training: positives are scarce (~63 human_jailbreak, ~14
direct_request, ~77 pooled), so we report

  - single-direction separation AUC/AP for the assistant axis (signed + unsigned),
  - the best individual trait directions (unsigned AUC), and
  - a stratified-CV multi-trait logistic-regression AUC/AP (out-of-fold),

for each of human_jailbreak / direct_request / pooled.

Unsigned AUC = max(auc, 1-auc): the trait/axis sign is arbitrary for detection, so
this measures separability. AP is reported next to the base rate so it's readable.
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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


def load_trait_matrix(vec_dir: str) -> tuple[np.ndarray, list[str]]:
    vecs, names = [], []
    for f in sorted(glob.glob(os.path.join(vec_dir, "*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        v = d["vector"][0].float().numpy()  # row 0 == model layer 30
        n = np.linalg.norm(v)
        if n > 1e-8:
            vecs.append(v / n)
            names.append(Path(f).stem)
    if not vecs:
        raise ValueError(f"No trait vectors in {vec_dir}")
    return np.stack(vecs).astype(np.float32), names


def load_axis(axis_path: str) -> np.ndarray:
    d = torch.load(axis_path, map_location="cpu", weights_only=False)
    v = d["axis"][0].float().numpy()
    return (v / np.linalg.norm(v)).astype(np.float32)


def safe_auc(y, s) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    a = float(roc_auc_score(y, s))
    return max(a, 1.0 - a)


def safe_ap(y, s) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return max(float(average_precision_score(y, s)),
               float(average_precision_score(y, -s)))


def cv_logreg(X, y, seed=42) -> tuple[float, float]:
    """Out-of-fold stratified-CV logreg AUC/AP on the trait-projection features."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    k = min(5, n_pos, n_neg)
    if k < 2:
        return float("nan"), float("nan")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, class_weight="balanced",
                                   max_iter=4000, random_state=seed)),
    ])
    oof = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, oof)), float(average_precision_score(y, oof))


def main() -> None:
    ap = argparse.ArgumentParser()
    base = "/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B"
    ap.add_argument("--activations", default=f"{base}/full_trait_output/harmbench_activations_gemma/activations.pt")
    ap.add_argument("--classified", default=f"{base}/full_trait_output/harmbench_activations_gemma/classified_responses.jsonl")
    ap.add_argument("--vectors_dir", default=f"{base}/gemma_trait_output/traits40_vectors_layer30_only/pre_generation_last_token/all_traits_no_filter")
    ap.add_argument("--axis_path", default=f"{base}/gemma_trait_output/traits40_axes_layer30_only/pre_generation_last_token/all_traits_no_filter/assistant_axis_pc1.pt")
    ap.add_argument("--out", default=f"{base}/full_trait_output/harmbench_activations_gemma/option_a_results.json")
    args = ap.parse_args()

    print("Loading activations ...", flush=True)
    acts = torch.load(args.activations, map_location="cpu", weights_only=False)
    rows = load_jsonl(args.classified)
    T, names = load_trait_matrix(args.vectors_dir)
    axis = load_axis(args.axis_path)
    print(f"  trait matrix {T.shape}, axis {axis.shape}, {len(rows)} labelled rows", flush=True)

    X, y, atk = [], [], []
    for r in rows:
        pid = r.get("pair_id")
        lab = r.get("jailbroken")
        item = acts.get(pid)
        if item is None or LAYER_KEY not in item or lab is None:
            continue
        X.append(item[LAYER_KEY].float().numpy())
        y.append(1 if lab else 0)
        atk.append(r.get("attack_type"))
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    atk = np.array(atk)
    print(f"  matched {len(y)} pairs, {int(y.sum())} jailbroken", flush=True)

    P = (X @ T.T).astype(np.float32)        # (N, 240) trait projections
    axproj = (X @ axis).astype(np.float32)  # (N,) assistant-axis projection

    subsets = {
        "human_jailbreak": atk == "human_jailbreak",
        "direct_request":  atk == "direct_request",
        "pooled":          np.ones(len(y), dtype=bool),
    }

    results = {}
    for sname, mask in subsets.items():
        yy = y[mask]
        n = int(mask.sum())
        npos = int(yy.sum())
        base_rate = npos / n if n else float("nan")

        s_ax = axproj[mask]
        signed_ax = (float(roc_auc_score(yy, s_ax))
                     if 0 < npos < n else float("nan"))
        ax_auc = safe_auc(yy, s_ax)
        ax_ap = safe_ap(yy, s_ax)

        trait_aucs = np.array([safe_auc(yy, P[mask][:, j]) for j in range(P.shape[1])])
        order = np.argsort(np.nan_to_num(trait_aucs, nan=-1))[::-1]
        top = [(names[j], float(trait_aucs[j])) for j in order[:5]]

        mt_auc, mt_ap = cv_logreg(P[mask], yy)

        results[sname] = {
            "n": n, "n_pos": npos, "base_rate": base_rate,
            "assistant_axis_signed_auc": signed_ax,
            "assistant_axis_unsigned_auc": ax_auc,
            "assistant_axis_ap": ax_ap,
            "best_traits_unsigned_auc": top,
            "multitrait_cv_logreg_auc": mt_auc,
            "multitrait_cv_logreg_ap": mt_ap,
        }

        print(f"\n=== {sname}  (n={n}, pos={npos}, base rate={base_rate:.3f}) ===", flush=True)
        print(f"  assistant axis : signed AUC={signed_ax:.3f}  unsigned AUC={ax_auc:.3f}  AP={ax_ap:.3f}", flush=True)
        print(f"  best traits (unsigned AUC):", flush=True)
        for tn, a in top:
            print(f"      {a:.3f}  {tn}", flush=True)
        print(f"  multi-trait CV logreg : AUC={mt_auc:.3f}  AP={mt_ap:.3f}  (chance AP={base_rate:.3f})", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
