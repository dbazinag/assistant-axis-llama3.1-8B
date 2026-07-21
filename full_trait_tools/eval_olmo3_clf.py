#!/usr/bin/env python3
"""
eval_olmo3_clf.py

Evaluate the saved OLMo-3 best classifier against any new attack dataset.
Prints per-family AUC and saves a JSON result file.

Usage:
  uv run python full_trait_tools/eval_olmo3_clf.py \
    --model_path full_trait_output/all_traits_sweep_v2_olmo3/best_model.pkl \
    --classified_path full_trait_output/jbb_attack_activations_olmo3/classified_responses.jsonl \
    --activations_path full_trait_output/jbb_attack_activations_olmo3/activations.pt \
    --output_json full_trait_output/all_traits_sweep_v2_olmo3/eval_jbb_attacks.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    auc = float(roc_auc_score(y_true, score))
    return max(auc, 1.0 - auc)


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return max(
        float(average_precision_score(y_true, score)),
        float(average_precision_score(y_true, -score)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        default="full_trait_output/all_traits_sweep_v2_olmo3/best_model.pkl")
    parser.add_argument("--classified_path", required=True,
                        help="JSONL with pair_id, attack_type, jailbroken fields.")
    parser.add_argument("--activations_path", required=True,
                        help="activations.pt with {pair_id: {'16': Tensor, '28': Tensor}}.")
    parser.add_argument("--output_json", default=None,
                        help="Where to save JSON results (default: next to classified_path).")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path} ...", flush=True)
    with open(args.model_path, "rb") as f:
        artefact = pickle.load(f)
    pipe          = artefact["pipeline"]
    trait_matrix  = artefact["trait_matrix"]   # (n_traits, 4096) float32
    layer         = artefact["layer"]
    threshold     = artefact["threshold"]
    sign          = artefact["sign"]
    layer_key     = str(layer)
    print(f"  Model: {artefact['meta']['model_name']}", flush=True)
    print(f"  Trained on {artefact['meta']['n_train']} samples "
          f"(jb={artefact['meta']['jb_rate_train']:.3f})", flush=True)
    print(f"  Val AUC: {artefact['meta']['val_auc']:.4f}", flush=True)

    # Load attack data
    print(f"\nLoading {args.classified_path} ...", flush=True)
    rows = load_jsonl(Path(args.classified_path))
    print(f"  {len(rows)} rows", flush=True)

    print(f"Loading {args.activations_path} ...", flush=True)
    acts = torch.load(args.activations_path, map_location="cpu", weights_only=False)
    print(f"  {len(acts)} activations", flush=True)

    # Build feature matrix
    xs, ys, attack_types = [], [], []
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
        attack_types.append(row.get("attack_type", "unknown"))

    x_raw = np.stack(xs).astype(np.float32)
    y     = np.array(ys, dtype=np.int64)
    x     = (x_raw @ trait_matrix.T).astype(np.float32)
    score = pipe.predict_proba(x)[:, 1]
    print(f"\n  Matched {len(y)} rows, jb_rate={y.mean():.3f}", flush=True)

    # Overall metrics
    overall_auc  = safe_auc(y, score)
    overall_ap   = safe_ap(y, score)
    pred         = (sign * score >= sign * threshold).astype(int)
    overall_bacc = float(balanced_accuracy_score(y, pred))
    print(f"\n=== Overall ===")
    print(f"  AUC:           {overall_auc:.4f}")
    print(f"  AP:            {overall_ap:.4f}")
    print(f"  Balanced acc:  {overall_bacc:.4f}  (threshold={threshold:.4f}, sign={sign:+d})")

    # Per-family metrics
    families = sorted(set(attack_types))
    per_family: dict[str, dict] = {}
    print(f"\n=== Per attack family ===")
    for fam in families:
        mask = np.array([at == fam for at in attack_types])
        y_f  = y[mask]
        s_f  = score[mask]
        auc  = safe_auc(y_f, s_f)
        ap   = safe_ap(y_f, s_f)
        pred_f = (sign * s_f >= sign * threshold).astype(int)
        bacc = float(balanced_accuracy_score(y_f, pred_f)) if len(np.unique(y_f)) >= 2 else float("nan")
        per_family[fam] = {
            "n": int(mask.sum()),
            "n_pos": int(y_f.sum()),
            "auc":  auc,
            "ap":   ap,
            "balanced_acc": bacc,
        }
        print(f"  {fam:<20}  n={mask.sum():>5}  jb={y_f.mean():.3f}  "
              f"AUC={auc:.4f}  AP={ap:.4f}  bacc={bacc:.4f}")

    # Save JSON
    out = {
        "model_path":       args.model_path,
        "classified_path":  args.classified_path,
        "activations_path": args.activations_path,
        "model_meta":       artefact["meta"],
        "overall": {
            "n":            int(len(y)),
            "n_pos":        int(y.sum()),
            "auc":          overall_auc,
            "ap":           overall_ap,
            "balanced_acc": overall_bacc,
        },
        "per_family": per_family,
    }
    if args.output_json:
        out_path = Path(args.output_json)
    else:
        cp = Path(args.classified_path)
        out_path = cp.parent / "clf_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
