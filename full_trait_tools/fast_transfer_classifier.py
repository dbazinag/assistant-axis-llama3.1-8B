#!/usr/bin/env python3
"""
fast_transfer_classifier.py

Runs transfer classification experiments for all attack families in one shot.
Key speedup over transfer_classifier.py: pre-computes all trait projections
as a single matrix multiply (N x 4096) @ (4096 x 229) instead of 229 loops.

Evaluates:
  - Transfer: train on human jailbreak → test on GCG
  - Transfer: train on human jailbreak → test on PAIR
  - Within-human: cross-validated (base reference)

All with strict pool split, averaged over N_SEEDS seeds.

Feature modes: raw, pca, all_traits, top_traits

Usage:
  uv run python full_trait_tools/fast_transfer_classifier.py

  # Custom paths
  uv run python full_trait_tools/fast_transfer_classifier.py \\
    --gcg_classified_path full_trait_output/gcg_activations/responses.jsonl \\
    --pair_classified_path full_trait_output/pair_activations/responses.jsonl
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────────────
LAYER       = 16
N_PCA       = 4
TOP_K       = 20
TRAIN_FRAC  = 0.7
N_SEEDS     = 10
RANDOM_SEED = 42


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_activations(path: Path) -> dict:
    print(f"  Loading {path} ({path.stat().st_size / 1e6:.1f} MB)...")
    t0 = time.time()
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return data


def load_trait_matrix(vectors_dir: Path, layer: int) -> Tuple[np.ndarray, List[str]]:
    """
    Load all trait vectors at given layer into a single matrix.
    Returns (matrix [n_traits x 4096], trait_names).
    Key speedup: projects all traits at once via matmul.
    """
    vecs, names = [], []
    for pt_file in sorted(vectors_dir.glob("*.pt")):
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            v = data["vector"][layer].float().numpy()
            norm = np.linalg.norm(v)
            if norm > 1e-8:
                vecs.append(v / norm)
                names.append(pt_file.stem)
        except Exception:
            pass
    matrix = np.stack(vecs)  # [n_traits x 4096]
    print(f"  Loaded {len(names)} trait vectors → matrix {matrix.shape}")
    return matrix, names


def load_hyperplane(path: Path) -> np.ndarray:
    data = torch.load(path, map_location="cpu", weights_only=False)
    v = data["vector"].float().numpy()
    return v / (np.linalg.norm(v) + 1e-12)


# ── Feature extraction ─────────────────────────────────────────────────────────

def build_activation_matrix(rows, activations, layer):
    """Extract activation vectors and labels for a list of rows."""
    layer_key = str(layer)
    X_list, y_list, valid_rows = [], [], []
    for row in rows:
        pid = row["pair_id"]
        jb = row.get("jailbroken")
        if jb is None:
            continue
        if pid not in activations or layer_key not in activations[pid]:
            continue
        X_list.append(activations[pid][layer_key].float().numpy())
        y_list.append(1 if jb else 0)
        valid_rows.append(row)
    if not X_list:
        return np.array([]), np.array([]), []
    return np.stack(X_list), np.array(y_list), valid_rows


def project_traits(X_raw: np.ndarray, trait_matrix: np.ndarray) -> np.ndarray:
    """Project activations onto all trait vectors at once: (N x 4096) @ (4096 x T) = (N x T)."""
    return X_raw @ trait_matrix.T


# ── Pool split ─────────────────────────────────────────────────────────────────

def get_pool_split(rows, train_frac, seed):
    rng = random.Random(seed)
    all_behaviors = sorted({r["behavior_id"]  for r in rows})
    all_templates = sorted({r["jailbreak_idx"] for r in rows})
    rng.shuffle(all_behaviors)
    rng.shuffle(all_templates)
    n_beh = max(1, int(len(all_behaviors) * train_frac))
    n_tpl = max(1, int(len(all_templates) * train_frac))
    train_beh = set(all_behaviors[:n_beh])
    train_tpl = set(all_templates[:n_tpl])
    test_beh  = set(all_behaviors[n_beh:])
    test_tpl  = set(all_templates[n_tpl:])
    return train_beh, train_tpl, test_beh, test_tpl


def split_by_pool(rows, train_beh, train_tpl, test_beh, test_tpl):
    train_idx = [i for i, r in enumerate(rows)
                 if r["behavior_id"] in train_beh and r["jailbreak_idx"] in train_tpl]
    test_idx  = [i for i, r in enumerate(rows)
                 if r["behavior_id"] in test_beh  and r["jailbreak_idx"] in test_tpl]
    return train_idx, test_idx


# ── Classifier ─────────────────────────────────────────────────────────────────

def fit_eval(X_tr, y_tr, X_te, y_te, mode, n_pca):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    pca = None
    if mode == "pca":
        pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
        X_tr_s = pca.fit_transform(X_tr_s)
        X_te_s = pca.transform(X_te_s)
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, C=1.0)
    clf.fit(X_tr_s, y_tr)
    if len(set(y_te)) < 2:
        return float("nan")
    probs = clf.predict_proba(X_te_s)[:, 1]
    return float(roc_auc_score(y_te, probs))


def cv_auc(X, y, mode, n_pca, n_splits=5):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    if mode == "pca":
        pca = PCA(n_components=n_pca, random_state=RANDOM_SEED)
        X_s = pca.fit_transform(X_s)
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, C=1.0)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(clf, X_s, y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_classified_path", default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--human_activations_path", default="full_trait_output/harmbench_activations/activations.pt")
    parser.add_argument("--gcg_classified_path",   default="full_trait_output/gcg_activations/responses.jsonl")
    parser.add_argument("--gcg_activations_path",  default="full_trait_output/gcg_activations/activations.pt")
    parser.add_argument("--pair_classified_path",  default="full_trait_output/pair_activations/responses.jsonl")
    parser.add_argument("--pair_activations_path", default="full_trait_output/pair_activations/activations.pt")
    parser.add_argument("--pap_classified_path",   default="full_trait_output/pap_activations/responses.jsonl")
    parser.add_argument("--pap_activations_path",  default="full_trait_output/pap_activations/activations.pt")
    parser.add_argument("--augment_classified_path",  default="", help="comma-separated classified jsonl paths folded into the TRAIN pool each seed")
    parser.add_argument("--augment_activations_path", default="", help="comma-separated activations.pt paths matching --augment_classified_path")
    parser.add_argument("--trait_vectors_dir",     default="full_trait_output/traits40_vectors/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total")
    parser.add_argument("--hyperplane_path",       default="full_trait_output/harmbench_logreg/stable_hyperplane_layer16.pt")
    parser.add_argument("--output_dir",            default="full_trait_output/transfer_results_all")
    parser.add_argument("--layer",    type=int, default=LAYER)
    parser.add_argument("--n_pca",    type=int, default=N_PCA)
    parser.add_argument("--top_k",    type=int, default=TOP_K)
    parser.add_argument("--n_seeds",  type=int, default=N_SEEDS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load everything ────────────────────────────────────────────────────────
    print("\n=== Loading data ===")

    print("Human jailbreak data...")
    human_rows_all = load_jsonl(Path(args.human_classified_path))
    human_rows = [r for r in human_rows_all if r.get("attack_type") == "human_jailbreak"]
    human_acts = load_activations(Path(args.human_activations_path))
    print(f"  {len(human_rows)} human rows")

    print("GCG data...")
    gcg_rows = load_jsonl(Path(args.gcg_classified_path))
    gcg_acts = load_activations(Path(args.gcg_activations_path))
    print(f"  {len(gcg_rows)} GCG rows")

    print("PAIR data...")
    pair_rows = load_jsonl(Path(args.pair_classified_path))
    pair_acts = load_activations(Path(args.pair_activations_path))
    print(f"  {len(pair_rows)} PAIR rows")

    print("PAP data...")
    pap_rows = load_jsonl(Path(args.pap_classified_path))
    pap_acts = load_activations(Path(args.pap_activations_path))
    print(f"  {len(pap_rows)} PAP rows")

    # Augment sources folded into the TRAIN pool each seed (eval stays pure)
    aug_X_raw_list, aug_y_list = [], []
    if args.augment_classified_path:
        aug_cls = [p for p in args.augment_classified_path.split(",") if p]
        aug_act = [p for p in args.augment_activations_path.split(",") if p]
        assert len(aug_cls) == len(aug_act), "augment classified/activations path counts differ"
        for cpath, apath in zip(aug_cls, aug_act):
            print(f"Augment source: {cpath}")
            a_rows = load_jsonl(Path(cpath))
            a_acts = load_activations(Path(apath))
            Xa, ya, _ = build_activation_matrix(a_rows, a_acts, args.layer)
            print(f"  {len(ya)} augment rows ({int(ya.sum()) if len(ya) else 0} jb)")
            if len(ya):
                aug_X_raw_list.append(Xa)
                aug_y_list.append(ya)

    print("Trait vectors...")
    trait_matrix, trait_names = load_trait_matrix(Path(args.trait_vectors_dir), args.layer)

    print("Hyperplane...")
    w_vec = load_hyperplane(Path(args.hyperplane_path))
    cos_w = np.abs(trait_matrix @ w_vec)
    top_k_idx = np.argsort(cos_w)[::-1][:args.top_k]
    top_trait_names = [trait_names[i] for i in top_k_idx]
    print(f"  Top {args.top_k} traits: {top_trait_names[:5]}...")

    # ── Pre-compute activation matrices (do this ONCE) ─────────────────────────
    print("\n=== Pre-computing activation matrices ===")

    X_human_raw, y_human, human_valid = build_activation_matrix(human_rows, human_acts, args.layer)
    X_gcg_raw,   y_gcg,   gcg_valid   = build_activation_matrix(gcg_rows,   gcg_acts,   args.layer)
    X_pair_raw,  y_pair,  pair_valid  = build_activation_matrix(pair_rows,  pair_acts,  args.layer)
    X_pap_raw,   y_pap,   pap_valid   = build_activation_matrix(pap_rows,   pap_acts,   args.layer)

    print(f"  Human: {len(y_human)} pairs ({y_human.sum():.0f} jb)")
    print(f"  GCG:   {len(y_gcg)} pairs ({y_gcg.sum():.0f} jb), chance={y_gcg.mean():.3f}")
    print(f"  PAIR:  {len(y_pair)} pairs ({y_pair.sum():.0f} jb), chance={y_pair.mean():.3f}")
    print(f"  PAP:   {len(y_pap)} pairs ({y_pap.sum():.0f} jb), chance={y_pap.mean():.3f}")

    X_aug_raw = np.vstack(aug_X_raw_list) if aug_X_raw_list else None
    y_aug     = np.concatenate(aug_y_list) if aug_y_list else None
    if X_aug_raw is not None:
        print(f"  Augment (train-only): {len(y_aug)} pairs ({y_aug.sum():.0f} jb)")

    # ── Pre-compute trait projections (single matmul each) ─────────────────────
    print("\n=== Pre-computing trait projections (fast matmul) ===")
    t0 = time.time()
    X_human_traits = project_traits(X_human_raw, trait_matrix)   # [N_h x 229]
    X_gcg_traits   = project_traits(X_gcg_raw,   trait_matrix)   # [N_gcg x 229]
    X_pair_traits  = project_traits(X_pair_raw,  trait_matrix)   # [N_pair x 229]
    X_pap_traits   = project_traits(X_pap_raw,   trait_matrix)   # [N_pap x 229]
    X_aug_traits   = project_traits(X_aug_raw,   trait_matrix) if X_aug_raw is not None else None
    print(f"  Done in {time.time()-t0:.2f}s")

    # top_k slice
    X_human_top = X_human_traits[:, top_k_idx]
    X_gcg_top   = X_gcg_traits[:,   top_k_idx]
    X_pair_top  = X_pair_traits[:,   top_k_idx]
    X_pap_top   = X_pap_traits[:,    top_k_idx]
    X_aug_top   = X_aug_traits[:,    top_k_idx] if X_aug_traits is not None else None

    # feature sets per mode
    def get_features(mode, X_raw, X_traits, X_top):
        if mode in ("raw", "pca"):
            return X_raw
        elif mode == "all_traits":
            return X_traits
        elif mode == "top_traits":
            return X_top

    modes = ["raw", "pca", "all_traits", "top_traits"]

    # ── Within-human CV (once per mode, not per seed) ─────────────────────────
    print("\n=== Within-human cross-validation ===")
    human_cv = {}
    for mode in modes:
        X = get_features(mode, X_human_raw, X_human_traits, X_human_top)
        auc, std = cv_auc(X, y_human, mode, args.n_pca)
        human_cv[mode] = (auc, std)
        print(f"  {mode:12s}: {auc:.4f} ± {std:.4f}")

    # ── Multi-seed transfer experiment ─────────────────────────────────────────
    print(f"\n=== Transfer experiment ({args.n_seeds} seeds) ===")

    # Accumulators: {mode: {dataset: [auc_per_seed]}}
    results = {mode: {"gcg": [], "pair": [], "pap": [], "human_test": []} for mode in modes}

    for seed in range(args.n_seeds):
        train_beh, train_tpl, test_beh, test_tpl = get_pool_split(
            human_valid, TRAIN_FRAC, seed
        )
        train_idx, test_idx = split_by_pool(
            human_valid, train_beh, train_tpl, test_beh, test_tpl
        )

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        for mode in modes:
            X_h  = get_features(mode, X_human_raw, X_human_traits, X_human_top)
            X_g  = get_features(mode, X_gcg_raw,   X_gcg_traits,   X_gcg_top)
            X_p  = get_features(mode, X_pair_raw,  X_pair_traits,  X_pair_top)
            X_pp = get_features(mode, X_pap_raw,   X_pap_traits,   X_pap_top)

            X_tr = X_h[train_idx];  y_tr = y_human[train_idx]
            X_te = X_h[test_idx];   y_te = y_human[test_idx]

            # fold augment families into the training pool (eval stays pure)
            if X_aug_raw is not None:
                X_a = get_features(mode, X_aug_raw, X_aug_traits, X_aug_top)
                X_tr = np.vstack([X_tr, X_a])
                y_tr = np.concatenate([y_tr, y_aug])

            auc_gcg       = fit_eval(X_tr, y_tr, X_g,  y_gcg,  mode, args.n_pca)
            auc_pair      = fit_eval(X_tr, y_tr, X_p,  y_pair, mode, args.n_pca)
            auc_pap       = fit_eval(X_tr, y_tr, X_pp, y_pap,  mode, args.n_pca)
            auc_human_test = fit_eval(X_tr, y_tr, X_te, y_te,  mode, args.n_pca)

            results[mode]["gcg"].append(auc_gcg)
            results[mode]["pair"].append(auc_pair)
            results[mode]["pap"].append(auc_pap)
            results[mode]["human_test"].append(auc_human_test)

        print(f"  Seed {seed} done")

    # ── Summary ────────────────────────────────────────────────────────────────
    sep = "=" * 100
    print(f"\n\n{sep}")
    print(f"  FULL TRANSFER SUMMARY  |  Layer {args.layer}  |  n_pca={args.n_pca}  |  top_k={args.top_k}  |  {args.n_seeds} seeds")
    print(sep)

    header = f"  {'Mode':12s}  {'Human CV':>12}  {'→GCG (mean)':>13}  {'→GCG (std)':>12}  {'→PAIR (mean)':>14}  {'→PAIR (std)':>12}  {'→PAP (mean)':>13}  {'Human test':>12}"
    print(f"\n{header}")
    print("  " + "─" * 96)

    summary = {}
    for mode in modes:
        h_cv, h_cv_std     = human_cv[mode]
        gcg_aucs           = results[mode]["gcg"]
        pair_aucs          = results[mode]["pair"]
        pap_aucs           = results[mode]["pap"]
        ht_aucs            = results[mode]["human_test"]

        gcg_mean  = float(np.mean(gcg_aucs))
        gcg_std   = float(np.std(gcg_aucs))
        pair_mean = float(np.mean(pair_aucs))
        pair_std  = float(np.std(pair_aucs))
        pap_mean  = float(np.mean(pap_aucs))
        pap_std   = float(np.std(pap_aucs))
        ht_mean   = float(np.mean(ht_aucs))

        print(f"  {mode:12s}  {h_cv:>12.4f}  {gcg_mean:>13.4f}  {gcg_std:>12.4f}  {pair_mean:>14.4f}  {pair_std:>12.4f}  {pap_mean:>13.4f}  {ht_mean:>12.4f}")

        summary[mode] = {
            "human_cv": {"mean": h_cv, "std": h_cv_std},
            "transfer_gcg":  {"mean": gcg_mean,  "std": gcg_std,  "all": gcg_aucs},
            "transfer_pair": {"mean": pair_mean, "std": pair_std, "all": pair_aucs},
            "transfer_pap":  {"mean": pap_mean,  "std": pap_std,  "all": pap_aucs},
            "human_test":    {"mean": ht_mean,   "all": ht_aucs},
        }

    print(f"\n  GCG chance baseline:  {y_gcg.mean():.4f}")
    print(f"  PAIR chance baseline: {y_pair.mean():.4f}")
    print(f"  PAP chance baseline:  {y_pap.mean():.4f}")
    print(sep)

    # Save
    out = {
        "layer": args.layer, "n_pca": args.n_pca, "top_k": args.top_k,
        "n_seeds": args.n_seeds,
        "augment_classified_path": args.augment_classified_path,
        "gcg_chance":  float(y_gcg.mean()),
        "pair_chance": float(y_pair.mean()),
        "pap_chance":  float(y_pap.mean()),
        "top_trait_names": top_trait_names,
        "modes": summary,
    }
    out_path = output_dir / "transfer_results_all.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
