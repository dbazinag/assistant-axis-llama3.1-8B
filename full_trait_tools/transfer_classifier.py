#!/usr/bin/env python3
"""
transfer_classifier.py

Transfer experiment: trains a logistic regression classifier on human jailbreak
activations at layer 16, then evaluates it on GCG attack activations.

If the jailbreak signal generalizes across attack families, we expect high AUC
on GCG even though the classifier was trained only on human jailbreak templates.

Also runs:
  - Within-human cross-validated AUC (upper bound reference)
  - Within-GCG cross-validated AUC (how well GCG is classifiable at all)
  - Chance baseline
  - Per-category breakdown

Usage:
  uv run full_trait_tools/transfer_classifier.py
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

LAYER       = 16
RANDOM_SEED = 42
TRAIN_FRAC  = 0.7
SPLIT_SEED  = 0
N_PCA       = 4


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_activations(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def get_test_pool(rows, train_frac, seed):
    rng = random.Random(seed)
    all_behaviors = sorted({r["behavior_id"]   for r in rows})
    all_templates = sorted({r["jailbreak_idx"]  for r in rows})
    rng.shuffle(all_behaviors)
    rng.shuffle(all_templates)
    n_train_beh = max(1, int(len(all_behaviors) * train_frac))
    n_train_tpl = max(1, int(len(all_templates) * train_frac))
    return (set(all_behaviors[:n_train_beh]), set(all_templates[:n_train_tpl]),
            set(all_behaviors[n_train_beh:]),  set(all_templates[n_train_tpl:]))


def build_xy(rows, activations, layer, pair_ids=None):
    layer_key = str(layer)
    X_list, y_list = [], []
    for row in rows:
        pid = row["pair_id"]
        if pair_ids is not None and pid not in pair_ids:
            continue
        jb = row.get("jailbroken")
        if jb is None:
            continue
        if pid not in activations or layer_key not in activations[pid]:
            continue
        X_list.append(activations[pid][layer_key].float().numpy())
        y_list.append(1 if jb else 0)
    if not X_list:
        return np.array([]), np.array([])
    return np.stack(X_list), np.array(y_list)


def train_clf(X_train, y_train, n_pca=None):
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train)
    pca = None
    if n_pca is not None:
        pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
        X_sc = pca.fit_transform(X_sc)
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
    clf.fit(X_sc, y_train)
    return clf, scaler, pca


def evaluate(clf, scaler, pca, X_test, y_test):
    X_sc = scaler.transform(X_test)
    if pca is not None:
        X_sc = pca.transform(X_sc)
    probs = clf.predict_proba(X_sc)[:, 1]
    preds = clf.predict(X_sc)
    auc = roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else float("nan")
    acc = accuracy_score(y_test, preds)
    return {"auc": auc, "acc": acc, "n": len(y_test),
            "n_pos": int(y_test.sum()), "n_neg": int((1-y_test).sum())}


def cross_val_auc(X, y, n_pca=None, n_splits=5):
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    if n_pca is not None:
        pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
        X_sc = pca.fit_transform(X_sc)
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(clf, X_sc, y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_classified_path", type=str,
        default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--human_activations_path", type=str,
        default="full_trait_output/harmbench_activations/activations.pt")
    parser.add_argument("--gcg_classified_path", type=str,
        default="full_trait_output/gcg_activations/responses.jsonl")
    parser.add_argument("--gcg_activations_path", type=str,
        default="full_trait_output/gcg_activations/activations.pt")
    parser.add_argument("--output_dir", type=str,
        default="full_trait_output/gcg_transfer")
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--n_pca", type=int, default=N_PCA)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load human jailbreak data ──────────────────────────────────────────────
    print("Loading human jailbreak data...")
    human_rows = load_jsonl(Path(args.human_classified_path))
    human_rows = [r for r in human_rows if r.get("attack_type") == "human_jailbreak"]
    human_acts = load_activations(Path(args.human_activations_path))
    print(f"  {len(human_rows)} rows")

    train_beh, train_tpl, test_beh, test_tpl = get_test_pool(
        human_rows, TRAIN_FRAC, SPLIT_SEED
    )
    train_pids = {r["pair_id"] for r in human_rows
                  if r["behavior_id"] in train_beh and r["jailbreak_idx"] in train_tpl}
    test_pids  = {r["pair_id"] for r in human_rows
                  if r["behavior_id"] in test_beh  and r["jailbreak_idx"] in test_tpl}

    X_h_train, y_h_train = build_xy(human_rows, human_acts, args.layer, train_pids)
    X_h_test,  y_h_test  = build_xy(human_rows, human_acts, args.layer, test_pids)
    X_h_all,   y_h_all   = build_xy(human_rows, human_acts, args.layer)

    print(f"  Train: {len(X_h_train)} ({y_h_train.sum()} jb / {(1-y_h_train).sum()} ref)")
    print(f"  Test:  {len(X_h_test)}  ({y_h_test.sum()} jb / {(1-y_h_test).sum()} ref)")

    # ── Load GCG data ──────────────────────────────────────────────────────────
    print("\nLoading GCG data...")
    gcg_rows = load_jsonl(Path(args.gcg_classified_path))
    gcg_acts = load_activations(Path(args.gcg_activations_path))
    X_gcg, y_gcg = build_xy(gcg_rows, gcg_acts, args.layer)
    print(f"  {len(X_gcg)} pairs ({y_gcg.sum()} jb / {(1-y_gcg).sum()} ref)")

    if len(X_gcg) == 0:
        print("ERROR: No GCG activations found.")
        return

    # ── Experiment 1: Within-human CV ─────────────────────────────────────────
    print("\nExperiment 1: Within-human cross-validated AUC...")
    h_cv_pca, h_cv_pca_std = cross_val_auc(X_h_all, y_h_all, n_pca=args.n_pca)
    h_cv_raw, h_cv_raw_std = cross_val_auc(X_h_all, y_h_all, n_pca=None)
    print(f"  PCA: {h_cv_pca:.4f} ± {h_cv_pca_std:.4f}")
    print(f"  Raw: {h_cv_raw:.4f} ± {h_cv_raw_std:.4f}")

    # ── Experiment 2: Within-GCG CV ───────────────────────────────────────────
    print("\nExperiment 2: Within-GCG cross-validated AUC...")
    if len(set(y_gcg)) < 2:
        gcg_cv_pca, gcg_cv_pca_std = float("nan"), float("nan")
        gcg_cv_raw, gcg_cv_raw_std = float("nan"), float("nan")
        print("  Cannot compute — only one class")
    else:
        n_splits = max(2, min(5, int(min(y_gcg.sum(), (1-y_gcg).sum()))))
        gcg_cv_pca, gcg_cv_pca_std = cross_val_auc(X_gcg, y_gcg, n_pca=args.n_pca, n_splits=n_splits)
        gcg_cv_raw, gcg_cv_raw_std = cross_val_auc(X_gcg, y_gcg, n_pca=None, n_splits=n_splits)
        print(f"  PCA: {gcg_cv_pca:.4f} ± {gcg_cv_pca_std:.4f}")
        print(f"  Raw: {gcg_cv_raw:.4f} ± {gcg_cv_raw_std:.4f}")

    # ── Experiment 3: Transfer human → GCG ────────────────────────────────────
    print("\nExperiment 3: Transfer — train on human, test on GCG...")
    clf_pca, scaler_pca, pca_obj = train_clf(X_h_train, y_h_train, n_pca=args.n_pca)
    clf_raw, scaler_raw, _       = train_clf(X_h_train, y_h_train, n_pca=None)

    t_pca = evaluate(clf_pca, scaler_pca, pca_obj, X_gcg, y_gcg)
    t_raw = evaluate(clf_raw, scaler_raw, None,    X_gcg, y_gcg)
    print(f"  PCA: AUC={t_pca['auc']:.4f} acc={100*t_pca['acc']:.1f}%")
    print(f"  Raw: AUC={t_raw['auc']:.4f} acc={100*t_raw['acc']:.1f}%")

    # ── Experiment 4: Human test set sanity check ──────────────────────────────
    print("\nExperiment 4: Human test set (sanity check)...")
    h_test_pca = evaluate(clf_pca, scaler_pca, pca_obj, X_h_test, y_h_test)
    h_test_raw = evaluate(clf_raw, scaler_raw, None,    X_h_test, y_h_test)
    print(f"  PCA: AUC={h_test_pca['auc']:.4f}")
    print(f"  Raw: AUC={h_test_raw['auc']:.4f}")

    # ── Per-category ───────────────────────────────────────────────────────────
    print("\nPer-category transfer AUC:")
    cat_results = {}
    for cat in sorted({r.get("semantic_category", "unknown") for r in gcg_rows}):
        cat_rows = [r for r in gcg_rows if r.get("semantic_category", "unknown") == cat]
        X_cat, y_cat = build_xy(cat_rows, gcg_acts, args.layer)
        if len(X_cat) < 4 or len(set(y_cat)) < 2:
            print(f"  {cat:40s} — skipped (n={len(X_cat)})")
            continue
        r = evaluate(clf_pca, scaler_pca, pca_obj, X_cat, y_cat)
        cat_results[cat] = r
        print(f"  {cat:40s} AUC={r['auc']:.3f} n={r['n']} "
              f"({r['n_pos']} jb / {r['n_neg']} ref)")

    # ── Summary ────────────────────────────────────────────────────────────────
    chance = float(y_gcg.mean())
    sep = "=" * 72
    print(f"\n\n{sep}")
    print(f"  TRANSFER EXPERIMENT SUMMARY  |  Layer {args.layer}  |  n_pca={args.n_pca}")
    print(sep)
    print(f"\n  {'Experiment':48s}  {'AUC':>8}")
    print("  " + "─" * 60)
    print(f"  {'Within-human CV (PCA)':48s}  {h_cv_pca:>8.4f}  ← upper bound")
    print(f"  {'Within-human CV (raw)':48s}  {h_cv_raw:>8.4f}")
    print(f"  {'Within-GCG CV (PCA)':48s}  {gcg_cv_pca:>8.4f}  ← GCG ceiling")
    print(f"  {'Within-GCG CV (raw)':48s}  {gcg_cv_raw:>8.4f}")
    print(f"  {'Transfer human→GCG (PCA)  *** KEY RESULT ***':48s}  {t_pca['auc']:>8.4f}")
    print(f"  {'Transfer human→GCG (raw)  *** KEY RESULT ***':48s}  {t_raw['auc']:>8.4f}")
    print(f"  {'Human test set (PCA)':48s}  {h_test_pca['auc']:>8.4f}  ← sanity check")
    print(f"  {'Chance (GCG positive rate)':48s}  {chance:>8.4f}")
    print(f"\n  Interpretation:")
    if not np.isnan(t_pca['auc']):
        gap_from_chance = t_pca['auc'] - chance
        gap_from_ceiling = h_cv_pca - t_pca['auc']
        print(f"    Transfer AUC is {gap_from_chance:+.3f} above chance")
        print(f"    Transfer AUC is {gap_from_ceiling:.3f} below within-human ceiling")
        if t_pca['auc'] > 0.65:
            print(f"    → Strong generalization: jailbreak signal transfers across attack families")
        elif t_pca['auc'] > 0.55:
            print(f"    → Moderate generalization: partial transfer across attack families")
        else:
            print(f"    → Weak generalization: signal may be template-specific")
    print(sep)

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = output_dir / "transfer_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "layer": args.layer, "n_pca": args.n_pca,
            "human_train_n": len(X_h_train),
            "human_test_n":  len(X_h_test),
            "gcg_n":         len(X_gcg),
            "within_human_cv_pca": {"auc": h_cv_pca, "std": h_cv_pca_std},
            "within_human_cv_raw": {"auc": h_cv_raw, "std": h_cv_raw_std},
            "within_gcg_cv_pca":   {"auc": gcg_cv_pca, "std": gcg_cv_pca_std},
            "within_gcg_cv_raw":   {"auc": gcg_cv_raw, "std": gcg_cv_raw_std},
            "transfer_pca":        t_pca,
            "transfer_raw":        t_raw,
            "human_test_pca":      h_test_pca,
            "human_test_raw":      h_test_raw,
            "chance_baseline":     chance,
            "per_category":        cat_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
