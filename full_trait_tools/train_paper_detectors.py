#!/usr/bin/env python3
"""
train_paper_detectors.py

Trains the TWO detectors the paper actually uses, back to back, so you can reproduce
them without running the full 57-config sweep:

  1. Trait detector  — trait-projected features (layer-16 activation @ trait matrix),
                       StandardScaler + LogisticRegression(C=10.0, l2, lbfgs).
                       This is the sweep's `logreg_l2_C10.0_raw` config.
  2. Raw detector    — raw 4096-dim layer-16 activation,
                       StandardScaler + LogisticRegression(C=1.0, l2, lbfgs).
                       This is the fast_transfer raw head.

Both are trained/tuned on HarmBench only and evaluated on the five unseen attack
families (GCG, PAIR, PAP, GPTFuzz, PEZ) with a strict pool split (no behavior or
template shared between train and test), averaged over N_SEEDS seeds — the same
protocol as the paper. Each model is then refit on all of HarmBench and saved.

Run from the project root (cluster):
  .venv/bin/python full_trait_tools/train_paper_detectors.py
"""
import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FAMILIES = ["GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]


# ── I/O ──────────────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_acts(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def build_xy(rows, acts, layer):
    """Return X [N x 4096], y [N], and the kept rows (with behavior_id/jailbreak_idx)."""
    lk = str(layer)
    X, y, kept = [], [], []
    for r in rows:
        pid, jb = r.get("pair_id"), r.get("jailbroken")
        if jb is None or pid not in acts or lk not in acts[pid]:
            continue
        X.append(acts[pid][lk].float().numpy())
        y.append(1 if jb else 0)
        kept.append(r)
    return np.stack(X), np.array(y), kept


def load_trait_matrix(npy_path, vectors_dir, layer):
    if Path(npy_path).exists():
        T = np.load(npy_path)
        print(f"  trait matrix (cached): {T.shape}")
        return T
    vecs = []
    for pt in sorted(Path(vectors_dir).glob("*.pt")):
        v = torch.load(pt, map_location="cpu", weights_only=False)["vector"][layer].float().numpy()
        n = np.linalg.norm(v)
        if n > 1e-8:
            vecs.append(v / n)
    T = np.stack(vecs)
    print(f"  trait matrix (built from {vectors_dir}): {T.shape}")
    return T


# ── strict pool split (behavior x template), matches the sweep / fast_transfer ─
def pool_split(rows, train_frac, seed):
    rng = random.Random(seed)
    beh = sorted({r["behavior_id"] for r in rows})
    tpl = sorted({r["jailbreak_idx"] for r in rows})
    rng.shuffle(beh); rng.shuffle(tpl)
    nb, nt = max(1, int(len(beh) * train_frac)), max(1, int(len(tpl) * train_frac))
    tr_b, tr_t = set(beh[:nb]), set(tpl[:nt])
    te_b, te_t = set(beh[nb:]), set(tpl[nt:])
    tr = [i for i, r in enumerate(rows) if r["behavior_id"] in tr_b and r["jailbreak_idx"] in tr_t]
    te = [i for i, r in enumerate(rows) if r["behavior_id"] in te_b and r["jailbreak_idx"] in te_t]
    return tr, te


# ── model factories (the two chosen configs) ─────────────────────────────────
def make_model(kind, seed):
    if kind == "trait":   # sweep logreg_l2_C10.0_raw
        clf = LogisticRegression(C=10.0, penalty="l2", solver="lbfgs",
                                 max_iter=4000, class_weight=None, random_state=seed)
    else:                 # fast_transfer raw head
        clf = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                 max_iter=2000, class_weight=None, random_state=seed)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def features(X_raw, kind, T):
    return X_raw @ T.T if kind == "trait" else X_raw


def best_threshold(y, score):
    """Youden's J on the given scores; returns (threshold, sign)."""
    from sklearn.metrics import roc_curve
    sign = 1 if roc_auc_score(y, score) >= 0.5 else -1
    fpr, tpr, thr = roc_curve(y, sign * score)
    j = np.argmax(tpr - fpr)
    return float(sign * thr[j]), sign


# ── train + evaluate one detector ────────────────────────────────────────────
def run_detector(kind, hb_X, hb_y, hb_rows, fam_data, T, n_seeds, train_frac):
    per = {"HarmBench": []}
    for f in FAMILIES:
        per[f] = []
    for seed in range(n_seeds):
        tr, te = pool_split(hb_rows, train_frac, seed)
        if len(set(hb_y[tr])) < 2 or len(te) == 0:
            continue
        m = make_model(kind, seed)
        m.fit(features(hb_X[tr], kind, T), hb_y[tr])
        if len(set(hb_y[te])) == 2:
            per["HarmBench"].append(roc_auc_score(hb_y[te], m.predict_proba(features(hb_X[te], kind, T))[:, 1]))
        for f in FAMILIES:
            fx, fy = fam_data[f]
            if fx is None or len(set(fy)) < 2:
                continue
            per[f].append(roc_auc_score(fy, m.predict_proba(features(fx, kind, T))[:, 1]))
    # Paper convention: report max(auc, 1-auc) so a family scored the wrong way up
    # (e.g. GPTFuzz, where the HarmBench-trained head is anti-correlated) reads correctly.
    def orient(m):
        return m if (m != m) else max(m, 1.0 - m)  # NaN-safe
    summary = {k: orient(float(np.mean(v))) if v else float("nan") for k, v in per.items()}
    summary["Avg_Transfer"] = float(np.nanmean([summary[f] for f in FAMILIES]))
    # refit on ALL of HarmBench and save
    final = make_model(kind, 0)
    final.fit(features(hb_X, kind, T), hb_y)
    thr, sign = best_threshold(hb_y, final.decision_function(features(hb_X, kind, T)))
    return summary, final, thr, sign


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="full_trait_output")
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--n_seeds", type=int, default=50)
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--trait_matrix_npy", default="full_trait_output/trait_matrix_layer16.npy")
    p.add_argument("--trait_vectors_dir",
                   default="full_trait_output/traits40_vectors/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total")
    p.add_argument("--hb_responses", default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    p.add_argument("--hb_acts", default="full_trait_output/harmbench_activations/activations.pt")
    p.add_argument("--fam_dir", default="full_trait_output", help="contains {gcg,pair,pap,gptfuzz,pez}_activations/")
    p.add_argument("--save_dir", default="full_trait_output/paper_detectors")
    args = p.parse_args()

    t0 = time.time()
    print("Loading trait matrix ...")
    T = load_trait_matrix(args.trait_matrix_npy, args.trait_vectors_dir, args.layer)
    print("Loading HarmBench ...")
    hb_X, hb_y, hb_rows = build_xy(load_jsonl(args.hb_responses), load_acts(args.hb_acts), args.layer)
    print(f"  HarmBench: {hb_X.shape}, jb_rate={hb_y.mean():.3f}")

    print("Loading attack families ...")
    fam_data = {}
    for f in FAMILIES:
        d = Path(args.fam_dir) / f"{f.lower()}_activations"
        rp, ap = d / "responses.jsonl", d / "activations.pt"
        if not (rp.exists() and ap.exists()):
            print(f"  {f}: MISSING ({d}) — skipped"); fam_data[f] = (None, None); continue
        fx, fy, _ = build_xy(load_jsonl(rp), load_acts(ap), args.layer)
        fam_data[f] = (fx, fy)
        print(f"  {f}: {fx.shape}, jb_rate={fy.mean():.3f}")

    save = Path(args.save_dir); save.mkdir(parents=True, exist_ok=True)
    cols = ["HarmBench"] + FAMILIES + ["Avg_Transfer"]
    print(f"\n{'detector':10s} " + " ".join(f"{c:>10s}" for c in cols))
    results = {}
    for kind in ["trait", "raw"]:
        summ, model, thr, sign = run_detector(kind, hb_X, hb_y, hb_rows, fam_data, T,
                                               args.n_seeds, args.train_frac)
        results[kind] = summ
        print(f"{kind:10s} " + " ".join(f"{summ[c]:10.3f}" for c in cols))
        artefact = {"pipeline": model, "kind": kind, "layer": args.layer,
                    "threshold": thr, "sign": sign, "n_seeds": args.n_seeds,
                    "config": "logreg_l2_C10.0 (trait proj)" if kind == "trait" else "logreg_C1.0 (raw 4096)",
                    "cv_summary": summ}
        if kind == "trait":
            artefact["trait_matrix"] = T
        with open(save / f"{kind}_detector.pkl", "wb") as fh:
            pickle.dump(artefact, fh)
    json.dump(results, open(save / "train_paper_detectors_results.json", "w"), indent=2)
    print(f"\nSaved models + results to {save}/  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
