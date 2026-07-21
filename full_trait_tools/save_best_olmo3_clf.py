#!/usr/bin/env python3
"""
save_best_olmo3_clf.py

Train the best OLMo-3 classifier (logreg_en_C0.03_l1r0.3, as determined by
all_traits_sweep_v2_olmo3) on all available HarmBench OLMo-3 human_jailbreak
data and save it to disk for reuse.

The saved artefact is a dict with:
  - 'pipeline':      fitted sklearn Pipeline (StandardScaler → LogisticRegression)
  - 'trait_matrix':  float32 ndarray (n_traits × 4096)
  - 'trait_names':   list[str]
  - 'layer':         int (16)
  - 'threshold':     float (balanced-accuracy-optimal, calibrated on heldout val)
  - 'sign':          int  (+1 or -1)
  - 'meta':          dict with training stats
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

LAYER = 16
VAL_FRAC = 0.15
RANDOM_SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_activations(path: Path) -> dict:
    print(f"  Loading {path.name} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_trait_matrix(cache_dir: Path) -> tuple[np.ndarray, list[str]]:
    mat_path   = cache_dir / f"trait_matrix_layer{LAYER}.npy"
    names_path = cache_dir / f"trait_names_layer{LAYER}.json"
    if not mat_path.exists() or not names_path.exists():
        raise FileNotFoundError(f"Missing trait matrix in {cache_dir}")
    matrix = np.load(mat_path).astype(np.float32)
    names  = json.loads(names_path.read_text())
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    print(f"  Trait matrix: {matrix.shape}", flush=True)
    return matrix, names


def build_features(rows: list[dict], acts: dict) -> tuple[np.ndarray, np.ndarray]:
    layer_key = str(LAYER)
    xs, ys = [], []
    for row in rows:
        pid   = row.get("pair_id")
        label = row.get("jailbroken")
        if pid is None or label is None:
            continue
        item = acts.get(pid)
        if item is None or layer_key not in item:
            continue
        xs.append(item[layer_key].float().numpy())
        ys.append(1 if label else 0)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_classified_path",
                        default="full_trait_output/harmbench_activations_olmo3/classified_responses.jsonl")
    parser.add_argument("--human_activations_path",
                        default="full_trait_output/harmbench_activations_olmo3/activations.pt")
    parser.add_argument("--sweep_dir",
                        default="full_trait_output/all_traits_sweep_v2_olmo3",
                        help="Dir containing trait_matrix_layer16.npy from the sweep.")
    parser.add_argument("--output_path",
                        default="full_trait_output/all_traits_sweep_v2_olmo3/best_model.pkl")
    args = parser.parse_args()

    print("=== Saving best OLMo-3 classifier ===", flush=True)

    # Load data
    print("\n[1] Loading data ...", flush=True)
    all_rows = [
        r for r in load_jsonl(Path(args.human_classified_path))
        if r.get("attack_type") == "human_jailbreak"
    ]
    acts = load_activations(Path(args.human_activations_path))
    print(f"  HarmBench human_jailbreak rows: {len(all_rows)}", flush=True)

    # Load trait matrix
    trait_matrix, trait_names = load_trait_matrix(Path(args.sweep_dir))

    # Build features
    print("\n[2] Building features ...", flush=True)
    x_raw, y = build_features(all_rows, acts)
    x = (x_raw @ trait_matrix.T).astype(np.float32)
    jb_rate = float(y.mean())
    print(f"  X: {x.shape}, jb_rate={jb_rate:.3f}", flush=True)

    # Hold out a small val set to calibrate threshold
    tr_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=VAL_FRAC,
        random_state=RANDOM_SEED, stratify=y,
    )
    print(f"  Train: {len(tr_idx)}, Val: {len(val_idx)}", flush=True)

    # Train best model: logreg_en_C0.03_l1r0.3
    print("\n[3] Training logreg_en_C0.03_l1r0.3 ...", flush=True)
    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(
            C=0.03, penalty="elasticnet", solver="saga", l1_ratio=0.3,
            max_iter=4000, class_weight="balanced", random_state=RANDOM_SEED,
        )),
    ])
    # Fit on train only to calibrate threshold on val
    pipe.fit(x[tr_idx], y[tr_idx])
    val_score = pipe.predict_proba(x[val_idx])[:, 1]
    threshold, sign = best_threshold(y[val_idx], val_score)
    val_auc = float(roc_auc_score(y[val_idx], val_score))
    val_pred = (sign * val_score >= sign * threshold).astype(int)
    val_bacc = float(balanced_accuracy_score(y[val_idx], val_pred))
    print(f"  Val AUC: {val_auc:.4f}, Val balanced_acc: {val_bacc:.4f}", flush=True)
    print(f"  Threshold: {threshold:.4f}, Sign: {sign}", flush=True)

    # Refit on ALL data (threshold already calibrated)
    print("\n[4] Refitting on full dataset ...", flush=True)
    pipe.fit(x, y)

    # Save
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
            "model_name":    "logreg_en_C0.03_l1r0.3",
            "n_train":       int(len(y)),
            "jb_rate_train": jb_rate,
            "val_auc":       val_auc,
            "val_bacc":      val_bacc,
            "human_classified_path": args.human_classified_path,
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(artefact, f, protocol=4)
    print(f"\nSaved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
