#!/usr/bin/env python3
"""
transfer_classifier.py

Transfer experiment: trains a logistic regression classifier on human jailbreak
activations at layer 16, then evaluates it on GCG attack activations.

Supports four feature modes:
  raw        — full 4096-dim activations
  pca        — PCA-compressed activations (n_pca components)
  all_traits — project activations onto all available trait vectors (229-dim)
  top_traits — project onto top K traits by alignment with w (default K=20)

For each mode runs:
  - Within-human cross-validated AUC (upper bound reference)
  - Within-GCG cross-validated AUC (GCG ceiling)
  - Transfer: train on human jailbreak, test on GCG (KEY RESULT)
  - Human test set AUC (sanity check)
  - Per-category breakdown

Usage:
  # All modes
  uv run full_trait_tools/transfer_classifier.py

  # Single mode
  uv run full_trait_tools/transfer_classifier.py --feature_modes raw
  uv run full_trait_tools/transfer_classifier.py --feature_modes top_traits --top_k 10
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
TOP_K       = 20


# ── Data loading ───────────────────────────────────────────────────────────────

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


def load_trait_vectors(vectors_dir: Path, layer: int) -> Dict[str, np.ndarray]:
    """Load all available trait vectors at given layer. Returns {trait: unit_vec}."""
    trait_vecs = {}
    for pt_file in sorted(vectors_dir.glob("*.pt")):
        trait = pt_file.stem
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            v = data["vector"][layer].float().numpy()
            norm = np.linalg.norm(v)
            if norm > 1e-8:
                trait_vecs[trait] = v / norm
        except Exception:
            pass
    return trait_vecs


def load_hyperplane(path: Path) -> np.ndarray:
    data = torch.load(path, map_location="cpu", weights_only=False)
    v = data["vector"].float().numpy()
    return v / (np.linalg.norm(v) + 1e-12)


def get_top_traits_by_w(
    trait_vecs: Dict[str, np.ndarray],
    w_vec: np.ndarray,
    top_k: int,
) -> List[str]:
    """Return top_k traits by |cosine similarity| with w."""
    cosines = {t: abs(float(np.dot(v, w_vec))) for t, v in trait_vecs.items()}
    ranked = sorted(cosines.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in ranked[:top_k]]


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(
    act: np.ndarray,
    mode: str,
    trait_vecs: Optional[Dict[str, np.ndarray]] = None,
    trait_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Extract feature vector from a single activation vector."""
    if mode in ("raw", "pca"):
        return act
    elif mode in ("all_traits", "top_traits"):
        names = trait_names if trait_names is not None else list(trait_vecs.keys())
        return np.array([float(np.dot(act, trait_vecs[t])) for t in names if t in trait_vecs])
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ── Train/test split ───────────────────────────────────────────────────────────

def get_test_pool(rows, train_frac, seed):
    rng = random.Random(seed)
    all_behaviors = sorted({r["behavior_id"]   for r in rows})
    all_templates = sorted({r["jailbreak_idx"]  for r in rows})
    rng.shuffle(all_behaviors)
    rng.shuffle(all_templates)
    n_beh = max(1, int(len(all_behaviors) * train_frac))
    n_tpl = max(1, int(len(all_templates) * train_frac))
    return (set(all_behaviors[:n_beh]), set(all_templates[:n_tpl]),
            set(all_behaviors[n_beh:]),  set(all_templates[n_tpl:]))


def build_xy(
    rows, activations, layer,
    mode, trait_vecs=None, trait_names=None,
    pair_ids=None,
):
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
        act = activations[pid][layer_key].float().numpy()
        feat = extract_features(act, mode, trait_vecs, trait_names)
        X_list.append(feat)
        y_list.append(1 if jb else 0)
    if not X_list:
        return np.array([]), np.array([])
    return np.stack(X_list), np.array(y_list)


# ── Classifier ─────────────────────────────────────────────────────────────────

def train_clf(X_train, y_train, mode, n_pca=None):
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train)
    pca = None
    if mode == "pca" and n_pca is not None:
        pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
        X_sc = pca.fit_transform(X_sc)
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, C=1.0)
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
    return {
        "auc": auc, "acc": acc, "n": len(y_test),
        "n_pos": int(y_test.sum()), "n_neg": int((1-y_test).sum()),
    }


def cross_val_auc(X, y, mode, n_pca=None, n_splits=5):
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    if mode == "pca" and n_pca is not None:
        pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
        X_sc = pca.fit_transform(X_sc)
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, C=1.0)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(clf, X_sc, y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


# ── Run one mode ───────────────────────────────────────────────────────────────

def run_mode(
    mode: str,
    X_h_train, y_h_train,
    X_h_test,  y_h_test,
    X_h_all,   y_h_all,
    X_gcg,     y_gcg,
    gcg_rows,  gcg_acts,
    layer, n_pca,
    trait_vecs, trait_names,
) -> dict:
    print(f"\n{'─'*60}")
    print(f"  MODE: {mode.upper()}")
    print(f"  Feature dim: {X_h_train.shape[1]}")
    print(f"{'─'*60}")

    # Within-human CV
    h_cv, h_cv_std = cross_val_auc(X_h_all, y_h_all, mode, n_pca)
    print(f"  Within-human CV AUC:     {h_cv:.4f} ± {h_cv_std:.4f}")

    # Within-GCG CV
    if len(set(y_gcg)) < 2:
        gcg_cv, gcg_cv_std = float("nan"), float("nan")
        print(f"  Within-GCG CV AUC:       nan (only one class)")
    else:
        n_splits = max(2, min(5, int(min(y_gcg.sum(), (1-y_gcg).sum()))))
        gcg_cv, gcg_cv_std = cross_val_auc(X_gcg, y_gcg, mode, n_pca, n_splits)
        print(f"  Within-GCG CV AUC:       {gcg_cv:.4f} ± {gcg_cv_std:.4f}")

    # Train on human, test on GCG
    clf, scaler, pca_obj = train_clf(X_h_train, y_h_train, mode, n_pca)
    t_gcg  = evaluate(clf, scaler, pca_obj, X_gcg,   y_gcg)
    t_h    = evaluate(clf, scaler, pca_obj, X_h_test, y_h_test)
    print(f"  Transfer human→GCG AUC:  {t_gcg['auc']:.4f}  ← KEY RESULT")
    print(f"  Human test set AUC:      {t_h['auc']:.4f}  (sanity check)")

    # Per-category
    cat_results = {}
    for cat in sorted({r.get("semantic_category", "unknown") for r in gcg_rows}):
        cat_rows = [r for r in gcg_rows if r.get("semantic_category", "unknown") == cat]
        # Re-extract features for this category
        layer_key = str(layer)
        X_cat_list, y_cat_list = [], []
        for row in cat_rows:
            pid = row["pair_id"]
            jb  = row.get("jailbroken")
            if jb is None:
                continue
            if pid not in gcg_acts or layer_key not in gcg_acts[pid]:
                continue
            act = gcg_acts[pid][layer_key].float().numpy()
            feat = extract_features(act, mode, trait_vecs, trait_names)
            X_cat_list.append(feat)
            y_cat_list.append(1 if jb else 0)
        if len(X_cat_list) < 4:
            continue
        X_cat = np.stack(X_cat_list)
        y_cat = np.array(y_cat_list)
        if len(set(y_cat)) < 2:
            continue
        r = evaluate(clf, scaler, pca_obj, X_cat, y_cat)
        cat_results[cat] = r

    return {
        "within_human_cv":  {"auc": h_cv,            "std": h_cv_std},
        "within_gcg_cv":    {"auc": gcg_cv,           "std": gcg_cv_std},
        "transfer_gcg":     t_gcg,
        "human_test":       t_h,
        "per_category":     cat_results,
        "feature_dim":      int(X_h_train.shape[1]),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

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
    parser.add_argument("--trait_vectors_dir", type=str,
        default="full_trait_output/traits40_vectors/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total")
    parser.add_argument("--hyperplane_path", type=str,
        default="full_trait_output/harmbench_logreg/stable_hyperplane_layer16.pt")
    parser.add_argument("--output_dir", type=str,
        default="full_trait_output/gcg_transfer")
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--n_pca", type=int, default=N_PCA)
    parser.add_argument("--top_k", type=int, default=TOP_K,
        help="Number of top traits to use in top_traits mode")
    parser.add_argument("--feature_modes", type=str,
        default="raw,pca,all_traits,top_traits",
        help="Comma-separated list of modes to run")
    args = parser.parse_args()

    modes = [m.strip() for m in args.feature_modes.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading human jailbreak data...")
    human_rows = load_jsonl(Path(args.human_classified_path))
    human_rows = [r for r in human_rows if r.get("attack_type") == "human_jailbreak"]
    human_acts = load_activations(Path(args.human_activations_path))
    print(f"  {len(human_rows)} rows")

    print("Loading GCG data...")
    gcg_rows = load_jsonl(Path(args.gcg_classified_path))
    gcg_acts = load_activations(Path(args.gcg_activations_path))
    print(f"  {len(gcg_rows)} rows")

    print("Loading trait vectors...")
    trait_vecs = load_trait_vectors(Path(args.trait_vectors_dir), args.layer)
    print(f"  {len(trait_vecs)} traits loaded")

    print("Loading hyperplane...")
    w_vec = load_hyperplane(Path(args.hyperplane_path))
    top_trait_names = get_top_traits_by_w(trait_vecs, w_vec, args.top_k)
    print(f"  Top {args.top_k} traits by |cos w|: {top_trait_names[:5]}...")

    # ── Pool split ─────────────────────────────────────────────────────────────
    train_beh, train_tpl, test_beh, test_tpl = get_test_pool(
        human_rows, TRAIN_FRAC, SPLIT_SEED
    )
    train_pids = {r["pair_id"] for r in human_rows
                  if r["behavior_id"] in train_beh and r["jailbreak_idx"] in train_tpl}
    test_pids  = {r["pair_id"] for r in human_rows
                  if r["behavior_id"] in test_beh  and r["jailbreak_idx"] in test_tpl}

    # ── Run each mode ──────────────────────────────────────────────────────────
    all_results = {}
    chance = None

    for mode in modes:
        trait_names = (
            top_trait_names if mode == "top_traits"
            else list(trait_vecs.keys()) if mode == "all_traits"
            else None
        )

        X_h_train, y_h_train = build_xy(human_rows, human_acts, args.layer,
                                         mode, trait_vecs, trait_names, train_pids)
        X_h_test,  y_h_test  = build_xy(human_rows, human_acts, args.layer,
                                         mode, trait_vecs, trait_names, test_pids)
        X_h_all,   y_h_all   = build_xy(human_rows, human_acts, args.layer,
                                         mode, trait_vecs, trait_names)
        X_gcg,     y_gcg     = build_xy(gcg_rows, gcg_acts, args.layer,
                                         mode, trait_vecs, trait_names)

        if chance is None:
            chance = float(y_gcg.mean()) if len(y_gcg) > 0 else float("nan")

        if len(X_h_train) == 0 or len(X_gcg) == 0:
            print(f"\nSkipping mode {mode} — no data")
            continue

        print(f"\n  Human train: {len(X_h_train)} "
              f"({y_h_train.sum()} jb / {(1-y_h_train).sum()} ref)")
        print(f"  Human test:  {len(X_h_test)}")
        print(f"  GCG:         {len(X_gcg)} ({y_gcg.sum()} jb / {(1-y_gcg).sum()} ref)")

        result = run_mode(
            mode,
            X_h_train, y_h_train,
            X_h_test,  y_h_test,
            X_h_all,   y_h_all,
            X_gcg,     y_gcg,
            gcg_rows,  gcg_acts,
            args.layer, args.n_pca,
            trait_vecs, trait_names,
        )
        if mode == "top_traits":
            result["trait_names"] = top_trait_names
        all_results[mode] = result

    # ── Summary table ──────────────────────────────────────────────────────────
    sep = "=" * 90
    print(f"\n\n{sep}")
    print(f"  TRANSFER EXPERIMENT SUMMARY  |  Layer {args.layer}  |  n_pca={args.n_pca}  |  top_k={args.top_k}")
    print(sep)
    print(f"\n  {'Mode':15s}  {'Feat dim':>9}  {'Human CV':>10}  {'GCG CV':>8}  "
          f"{'Transfer':>10}  {'Human test':>11}  {'vs chance':>10}")
    print("  " + "─" * 86)

    for mode, r in all_results.items():
        h_cv   = r["within_human_cv"]["auc"]
        gcg_cv = r["within_gcg_cv"]["auc"]
        t_auc  = r["transfer_gcg"]["auc"]
        h_test = r["human_test"]["auc"]
        fdim   = r["feature_dim"]
        delta  = t_auc - chance if not np.isnan(t_auc) else float("nan")
        print(f"  {mode:15s}  {fdim:>9}  {h_cv:>10.4f}  {gcg_cv:>8.4f}  "
              f"{t_auc:>10.4f}  {h_test:>11.4f}  {delta:>+10.4f}")

    print(f"\n  Chance baseline (GCG positive rate): {chance:.4f}")
    print(f"\n  Transfer AUC interpretation:")
    print(f"    > 0.70 = strong generalization across attack families")
    print(f"    0.55–0.70 = moderate generalization")
    print(f"    < 0.55 = weak / template-specific signal")

    # Per-category for best mode
    best_mode = max(all_results.items(), key=lambda x: x[1]["transfer_gcg"]["auc"])[0]
    print(f"\n  Per-category breakdown (best mode: {best_mode}):")
    print(f"  {'Category':42s}  {'Transfer AUC':>13}  {'n':>5}  {'jb%':>6}")
    print("  " + "─" * 70)
    for cat, r in sorted(all_results[best_mode]["per_category"].items(),
                          key=lambda x: x[1]["auc"]):
        jb_pct = 100 * r["n_pos"] / r["n"] if r["n"] > 0 else 0
        print(f"  {cat:42s}  {r['auc']:>13.3f}  {r['n']:>5}  {jb_pct:>5.0f}%")

    print(sep)

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = output_dir / "transfer_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "layer": args.layer, "n_pca": args.n_pca, "top_k": args.top_k,
            "chance_baseline": chance,
            "modes": all_results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
