#!/usr/bin/env python3
"""
save_best_wjb_clf.py

Sweep LogisticRegression variants (L1 / L2 / ElasticNet, various C / l1_ratio)
over WildJailbreak Llama-3.1-8B activations (layer 16, trait-projected).

Label: binary_jailbroken from rescored_responses.jsonl (~51% positive).
Data:  wildjailbreak_activations/activations.pt  (2000 Llama unsteered pairs)

Saves the best-AUC model to:
    full_trait_output/wildjailbreak_clf/best_model.pkl

Artefact keys: pipeline, trait_matrix, trait_names, layer, threshold, sign, meta
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LAYER       = 16
N_CV_SEEDS  = 20          # seeds for AUC comparison sweep
VAL_FRAC    = 0.15        # held-out fraction for threshold calibration
RANDOM_SEED = 42


# ── helpers ────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_trait_matrix(output_dir: Path) -> tuple[np.ndarray, list[str]]:
    mat_path   = output_dir / f"trait_matrix_layer{LAYER}.npy"
    names_path = output_dir / f"trait_names_layer{LAYER}.json"
    if not mat_path.exists() or not names_path.exists():
        raise FileNotFoundError(f"Missing trait matrix files in {output_dir}")
    matrix = np.load(mat_path).astype(np.float32)
    names  = json.loads(names_path.read_text())
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    print(f"  Trait matrix: {matrix.shape}", flush=True)
    return matrix, names


def load_data(acts_path: Path, labels_path: Path,
              trait_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    print(f"  Loading activations ({acts_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    acts = torch.load(acts_path, map_location="cpu", weights_only=False)

    label_rows = load_jsonl(labels_path)
    label_map  = {int(r["pair_id"]): r.get("binary_jailbroken", r.get("jailbroken")) for r in label_rows
                  if "pair_id" in r and ("binary_jailbroken" in r or "jailbroken" in r)}

    layer_key = str(LAYER)
    xs, ys = [], []
    for pid, item in acts.items():
        if layer_key not in item:
            continue
        label = label_map.get(pid)
        if label is None:
            continue
        xs.append(item[layer_key].float().numpy())
        ys.append(1 if label else 0)

    X_raw = np.stack(xs).astype(np.float32)
    X     = (X_raw @ trait_matrix.T).astype(np.float32)
    y     = np.array(ys, dtype=np.int64)
    return X, y


def best_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, int]:
    candidates = np.unique(score)
    if len(candidates) > 500:
        candidates = np.quantile(candidates, np.linspace(0, 1, 500))
    best_bacc, best_thr, best_sign = -1.0, 0.0, 1
    for sign in (1, -1):
        for thr in candidates:
            pred = (sign * score >= sign * thr).astype(int)
            bacc = float(balanced_accuracy_score(y_true, pred))
            if bacc > best_bacc:
                best_bacc, best_thr, best_sign = bacc, float(thr), sign
    return best_thr, best_sign


def make_configs() -> list[dict]:
    configs = []
    Cs = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

    for C in Cs:
        configs.append({"name": f"logreg_l2_C{C}",
                        "penalty": "l2", "solver": "lbfgs",
                        "C": C, "l1_ratio": None})
    for C in Cs:
        configs.append({"name": f"logreg_l1_C{C}",
                        "penalty": "l1", "solver": "saga",
                        "C": C, "l1_ratio": None})
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        for l1r in [0.1, 0.3, 0.5, 0.7, 0.9]:
            configs.append({"name": f"logreg_en_C{C}_l1r{l1r}",
                            "penalty": "elasticnet", "solver": "saga",
                            "C": C, "l1_ratio": l1r})
    return configs


def build_pipeline(cfg: dict) -> Pipeline:
    kwargs = dict(
        C=cfg["C"], penalty=cfg["penalty"], solver=cfg["solver"],
        max_iter=4000, class_weight="balanced", random_state=RANDOM_SEED,
    )
    if cfg["l1_ratio"] is not None:
        kwargs["l1_ratio"] = cfg["l1_ratio"]
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(**kwargs))])


def cv_auc(X: np.ndarray, y: np.ndarray, cfg: dict) -> float:
    aucs = []
    for seed in range(N_CV_SEEDS):
        tr, te = train_test_split(np.arange(len(y)), test_size=0.20,
                                  random_state=seed, stratify=y)
        pipe = build_pipeline(cfg)
        pipe.fit(X[tr], y[tr])
        score = pipe.predict_proba(X[te])[:, 1]
        aucs.append(float(roc_auc_score(y[te], score)))
    return float(np.mean(aucs))


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_path",
                        default="full_trait_output/wildjailbreak_activations/activations.pt")
    parser.add_argument("--labels_path",
                        default="full_trait_output/wildjailbreak_activations/rescored_responses.jsonl")
    parser.add_argument("--trait_matrix_dir",
                        default="full_trait_output",
                        help="Dir containing trait_matrix_layer16.npy (Llama)")
    parser.add_argument("--output_path",
                        default="full_trait_output/wildjailbreak_clf/best_model.pkl")
    args = parser.parse_args()

    print("=== WildJailbreak Llama classifier sweep ===", flush=True)

    print("\n[1] Loading trait matrix ...", flush=True)
    trait_matrix, trait_names = load_trait_matrix(Path(args.trait_matrix_dir))

    print("\n[2] Loading WJB data ...", flush=True)
    X, y = load_data(Path(args.activations_path), Path(args.labels_path), trait_matrix)
    print(f"  X: {X.shape}, jb_rate={y.mean():.3f} "
          f"({y.sum()} pos / {len(y)-y.sum()} neg)", flush=True)

    # Hold out a val set now for threshold calibration (stratified)
    tr_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=VAL_FRAC,
        random_state=RANDOM_SEED, stratify=y,
    )
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"  Train pool: {len(tr_idx)}, Val: {len(val_idx)}", flush=True)

    print(f"\n[3] Sweeping {len(make_configs())} configs "
          f"({N_CV_SEEDS}-seed CV each) ...", flush=True)
    configs  = make_configs()
    results  = []
    for i, cfg in enumerate(configs):
        auc = cv_auc(X_tr, y_tr, cfg)
        results.append((auc, cfg))
        if (i + 1) % 10 == 0 or (i + 1) == len(configs):
            print(f"  [{i+1}/{len(configs)}] {cfg['name']:35s}  AUC={auc:.4f}", flush=True)

    results.sort(key=lambda x: -x[0])
    best_auc, best_cfg = results[0]
    print(f"\n  Best: {best_cfg['name']}  AUC={best_auc:.4f}", flush=True)
    print(f"  Top 5:", flush=True)
    for auc, cfg in results[:5]:
        print(f"    {cfg['name']:35s}  AUC={auc:.4f}", flush=True)

    print("\n[4] Fitting best config on train pool, calibrating threshold on val ...",
          flush=True)
    pipe = build_pipeline(best_cfg)
    pipe.fit(X_tr, y_tr)
    val_scores = pipe.predict_proba(X_val)[:, 1]
    val_auc    = float(roc_auc_score(y_val, val_scores))
    threshold, sign = best_threshold(y_val, val_scores)
    val_pred   = (sign * val_scores >= sign * threshold).astype(int)
    val_bacc   = float(balanced_accuracy_score(y_val, val_pred))
    print(f"  Val AUC: {val_auc:.4f}  Val balanced_acc: {val_bacc:.4f}", flush=True)
    print(f"  Threshold: {threshold:.4f}  Sign: {sign}", flush=True)

    print("\n[5] Refitting on full dataset ...", flush=True)
    pipe.fit(X, y)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artefact = {
        "pipeline":     pipe,
        "trait_matrix": trait_matrix,
        "trait_names":  trait_names,
        "layer":        LAYER,
        "threshold":    threshold,
        "sign":         sign,
        "meta": {
            "model_name":      best_cfg["name"],
            "cv_auc":          best_auc,
            "val_auc":         val_auc,
            "val_bacc":        val_bacc,
            "n_train":         int(len(y)),
            "jb_rate_train":   float(y.mean()),
            "activations_path": args.activations_path,
            "label":           "binary_jailbroken",
        },
    }
    with open(out_path, "wb") as fh:
        pickle.dump(artefact, fh, protocol=4)
    print(f"\nSaved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
