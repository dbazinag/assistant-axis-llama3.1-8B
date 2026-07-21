#!/usr/bin/env python3
"""Leakage-proof multi-trait evaluation for Gemma + empirical metric baselines.

Closes the leakage loophole left by random CV (same behavior/template in train+test):
  1. GroupKFold by behavior_id  -> no harmful behavior seen in both train and test
     (leak-free, uses all positives) -> out-of-fold AUC/AP.
  2. Strict behavior x template cross-product split (the run_all_traits_sweep_v2
     protocol) over many seeds -> mean held-out AUC + how many test positives each
     seed actually gets (shows the scarcity starvation).

Also empirically measures the metric floor on the real labels:
  - random-uniform scores, constant all-negative score, constant all-positive score
  -> their AUC and AP, so we can see exactly what "guessing" scores.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
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
    vecs = []
    for f in sorted(glob.glob(os.path.join(vec_dir, "*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        v = d["vector"][0].float().numpy()
        n = np.linalg.norm(v)
        if n > 1e-8:
            vecs.append(v / n)
    return np.stack(vecs).astype(np.float32)


def make_pipe(seed):
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, class_weight="balanced",
                                   max_iter=4000, random_state=seed)),
    ])


# ── strict behavior x template split (matches run_all_traits_sweep_v2) ──────────

def strict_split_eval(P, y, beh, tpl, n_seeds=50, train_frac=0.7):
    aucs, aps, test_pos, valid = [], [], [], 0
    behaviors = sorted(set(beh))
    templates = sorted(set(tpl))
    for seed in range(n_seeds):
        rng = random.Random(seed)
        b = behaviors[:]; t = templates[:]
        rng.shuffle(b); rng.shuffle(t)
        nb = max(1, int(len(b) * train_frac))
        nt = max(1, int(len(t) * train_frac))
        train_b, test_b = set(b[:nb]), set(b[nb:])
        train_t, test_t = set(t[:nt]), set(t[nt:])
        tr = [i for i in range(len(y)) if beh[i] in train_b and tpl[i] in train_t]
        te = [i for i in range(len(y)) if beh[i] in test_b and tpl[i] in test_t]
        if not tr or not te:
            continue
        ytr, yte = y[tr], y[te]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        m = make_pipe(seed)
        m.fit(P[tr], ytr)
        score = m.predict_proba(P[te])[:, 1]
        aucs.append(roc_auc_score(yte, score))
        aps.append(average_precision_score(yte, score))
        test_pos.append(int(yte.sum()))
        valid += 1
    if not aucs:
        return None
    return {
        "valid_seeds": valid, "n_seeds": n_seeds,
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "ap_mean": float(np.mean(aps)), "ap_std": float(np.std(aps)),
        "test_pos_mean": float(np.mean(test_pos)), "test_pos_min": int(np.min(test_pos)),
        "test_pos_max": int(np.max(test_pos)),
    }


def main():
    ap = argparse.ArgumentParser()
    base = "/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B"
    ap.add_argument("--activations", default=f"{base}/full_trait_output/harmbench_activations_gemma/activations.pt")
    ap.add_argument("--classified", default=f"{base}/full_trait_output/harmbench_activations_gemma/classified_responses.jsonl")
    ap.add_argument("--vectors_dir", default=f"{base}/gemma_trait_output/traits40_vectors_layer30_only/pre_generation_last_token/all_traits_no_filter")
    ap.add_argument("--out", default=f"{base}/full_trait_output/harmbench_activations_gemma/strict_split_results.json")
    args = ap.parse_args()

    acts = torch.load(args.activations, map_location="cpu", weights_only=False)
    rows = load_jsonl(args.classified)
    T = load_trait_matrix(args.vectors_dir)

    X, y, atk, beh, tpl = [], [], [], [], []
    for r in rows:
        item = acts.get(r.get("pair_id"))
        lab = r.get("jailbroken")
        if item is None or LAYER_KEY not in item or lab is None:
            continue
        X.append(item[LAYER_KEY].float().numpy())
        y.append(1 if lab else 0)
        atk.append(r.get("attack_type"))
        beh.append(r.get("behavior_id"))
        tpl.append(r.get("jailbreak_idx"))
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    atk = np.array(atk)
    beh = np.array(beh)
    tpl = np.array(tpl)
    P = (X @ T.T).astype(np.float32)

    # ── empirical metric baselines on the REAL human_jailbreak labels ──────────
    mask_h = atk == "human_jailbreak"
    yh = y[mask_h]
    base_rate = float(yh.mean())
    print(f"=== METRIC BASELINES on human_jailbreak labels (n={len(yh)}, pos={int(yh.sum())}, base rate={base_rate:.3f}) ===", flush=True)
    rng = np.random.default_rng(0)
    rand_aucs, rand_aps = [], []
    for _ in range(2000):
        s = rng.random(len(yh))
        rand_aucs.append(roc_auc_score(yh, s))
        rand_aps.append(average_precision_score(yh, s))
    const0 = np.zeros(len(yh))
    const1 = np.ones(len(yh))
    print(f"  random-uniform score : AUC {np.mean(rand_aucs):.3f} (95% [{np.percentile(rand_aucs,2.5):.3f},{np.percentile(rand_aucs,97.5):.3f}])  AP {np.mean(rand_aps):.3f}", flush=True)
    print(f"  constant all-NEGATIVE : AUC {roc_auc_score(yh, const0):.3f}  AP {average_precision_score(yh, const0):.3f}", flush=True)
    print(f"  constant all-POSITIVE : AUC {roc_auc_score(yh, const1):.3f}  AP {average_precision_score(yh, const1):.3f}", flush=True)
    print(f"  (AUC floor = 0.5 for any non-informative score; AP floor = base rate = {base_rate:.3f})", flush=True)

    # ── leak-free evaluations ──────────────────────────────────────────────────
    results = {"baselines": {"base_rate": base_rate,
                             "random_auc": float(np.mean(rand_aucs)),
                             "random_ap": float(np.mean(rand_aps)),
                             "allneg_auc": float(roc_auc_score(yh, const0)),
                             "allneg_ap": float(average_precision_score(yh, const0)),
                             "allpos_auc": float(roc_auc_score(yh, const1)),
                             "allpos_ap": float(average_precision_score(yh, const1))}}

    for sname, mask in [("human_jailbreak", atk == "human_jailbreak"),
                        ("pooled", np.ones(len(y), dtype=bool))]:
        yy = y[mask]
        Pm = P[mask]
        behm = beh[mask]
        tplm = tpl[mask]
        npos = int(yy.sum())

        # 1. GroupKFold by behavior_id (leak-free on behavior, all positives used)
        n_groups = len(set(behm))
        k = min(5, n_groups)
        gkf = GroupKFold(n_splits=k)
        oof = cross_val_predict(make_pipe(0), Pm, yy, cv=gkf, groups=behm,
                                method="predict_proba")[:, 1]
        gk_auc = float(roc_auc_score(yy, oof))
        gk_ap = float(average_precision_score(yy, oof))

        # 2. strict behavior x template cross-product split
        strict = strict_split_eval(Pm, yy, behm, tplm)

        results[sname] = {"n": int(mask.sum()), "n_pos": npos,
                          "groupkfold_behavior_auc": gk_auc, "groupkfold_behavior_ap": gk_ap,
                          "groupkfold_k": k, "strict_xprod": strict}

        print(f"\n=== {sname}  (n={int(mask.sum())}, pos={npos}) ===", flush=True)
        print(f"  GroupKFold-by-behavior ({k} folds, leak-free on behavior): AUC {gk_auc:.3f}  AP {gk_ap:.3f}", flush=True)
        if strict:
            print(f"  strict behavior x template split ({strict['valid_seeds']}/{strict['n_seeds']} seeds usable): "
                  f"AUC {strict['auc_mean']:.3f}±{strict['auc_std']:.3f}  AP {strict['ap_mean']:.3f}±{strict['ap_std']:.3f}", flush=True)
            print(f"     test positives per seed: mean {strict['test_pos_mean']:.1f} (range {strict['test_pos_min']}-{strict['test_pos_max']})", flush=True)
        else:
            print(f"  strict behavior x template split: NO usable seeds (too few positives survive the split)", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
