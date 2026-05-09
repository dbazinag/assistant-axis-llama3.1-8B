#!/usr/bin/env python3
"""
Train Gemma-only classifiers using projections onto learned trait/persona vectors.

Method:
  residuals [n_layers, n_tokens, d_model]
  -> final valid prompt token per layer [n_layers, d_model]
  -> project onto unit trait vectors [n_layers, n_traits]
  -> optional assistant-axis projections [n_layers]
  -> logistic regression classifiers
  -> report held-out validation AUCs

This version only worries about Gemma right now:
  - Refusal-Gemma if is_refusal labels are present
  - Cyber Probe-1 if cyber category labels are present
  - Cyber Probe-2 if cyber category labels are present
  - Cyber Probe-3 if cyber category labels are present
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_VECTOR_LAYERS = [15, 20, 25, 30, 35, 40, 45]


def load_manifest_samples(manifest_path: Path) -> List[Dict[str, Any]]:
    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
    found = []

    def visit(obj):
        if isinstance(obj, list):
            if obj and all(isinstance(x, dict) for x in obj):
                if any("sample_id" in x for x in obj):
                    found.extend([x for x in obj if "sample_id" in x])
            for x in obj:
                visit(x)
        elif isinstance(obj, dict):
            for v in obj.values():
                visit(v)

    visit(manifest)

    dedup = {}
    for s in found:
        dedup.setdefault(s["sample_id"], s)

    return list(dedup.values())


def load_extract(extracts_dir: Path, sample_id: str) -> Optional[Dict[str, Any]]:
    path = extracts_dir / f"{sample_id}.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def get_final_token_by_layer(ex: Dict[str, Any]) -> torch.Tensor:
    if "residuals" not in ex:
        raise KeyError("extract missing key 'residuals'")

    residuals = ex["residuals"].float()

    if residuals.ndim == 2:
        residuals = residuals.unsqueeze(0)

    if residuals.ndim != 3:
        raise ValueError(f"Expected residuals [L,T,D], got {tuple(residuals.shape)}")

    attention_mask = ex["attention_mask"]
    if isinstance(attention_mask, torch.Tensor):
        mask = attention_mask.bool()
    else:
        mask = torch.tensor(attention_mask).bool()

    if not mask.any():
        raise ValueError("empty attention mask")

    last_idx = int(mask.nonzero().max().item())
    return residuals[:, last_idx, :]


def load_trait_matrix(trait_vector_dir: Path) -> Tuple[torch.Tensor, List[str]]:
    files = sorted(trait_vector_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt trait vectors found in {trait_vector_dir}")

    vectors = []
    names = []

    for p in files:
        d = torch.load(p, map_location="cpu", weights_only=False)

        if isinstance(d, dict):
            if "vector" not in d:
                raise KeyError(f"{p} missing key 'vector'")
            vec = d["vector"].float()
            name = d.get("trait", p.stem)
        else:
            vec = d.float()
            name = p.stem

        if vec.ndim != 2:
            raise ValueError(f"{p} vector must be 2D, got {tuple(vec.shape)}")

        vectors.append(vec)
        names.append(name)

    mat = torch.stack(vectors, dim=1)  # [L, n_traits, D]
    mat = mat / (mat.norm(dim=-1, keepdim=True) + 1e-8)

    return mat, names


def load_axis(axis_path: Optional[Path]) -> Optional[torch.Tensor]:
    if axis_path is None:
        return None

    if not axis_path.exists():
        print(f"[warn] assistant axis not found: {axis_path}")
        return None

    d = torch.load(axis_path, map_location="cpu", weights_only=False)

    if isinstance(d, dict):
        if "axis" in d:
            axis = d["axis"].float()
        elif "vector" in d:
            axis = d["vector"].float()
        else:
            raise KeyError(f"Axis file missing axis/vector keys: {list(d.keys())}")
    else:
        axis = d.float()

    if axis.ndim != 2:
        raise ValueError(f"Axis must be 2D, got {tuple(axis.shape)}")

    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
    return axis


def projection_features(
    hidden_by_layer: torch.Tensor,
    trait_mat: torch.Tensor,
    axis: Optional[torch.Tensor],
) -> np.ndarray:
    """
    hidden_by_layer: [L, D]
    trait_mat:       [L, n_traits, D]
    axis:            [L, D] or None
    """
    hidden = hidden_by_layer.float()

    if hidden.shape[0] != trait_mat.shape[0]:
        raise ValueError(f"Layer mismatch: hidden {tuple(hidden.shape)}, trait_mat {tuple(trait_mat.shape)}")

    if hidden.shape[-1] != trait_mat.shape[-1]:
        raise ValueError(f"Hidden-size mismatch: hidden {tuple(hidden.shape)}, trait_mat {tuple(trait_mat.shape)}")

    # [L, n_traits]
    trait_proj = torch.einsum("ld,ltd->lt", hidden, trait_mat)
    features = [trait_proj.flatten()]

    if axis is not None:
        if axis.shape != hidden.shape:
            raise ValueError(f"Axis shape mismatch: hidden {tuple(hidden.shape)}, axis {tuple(axis.shape)}")
        axis_proj = torch.einsum("ld,ld->l", hidden, axis)
        features.append(axis_proj.flatten())

    return torch.cat(features).cpu().numpy().astype(np.float32)


def cyber_probe_label(category: str, probe_id: int) -> Optional[int]:
    pos = {
        1: "dual_use",
        2: "high_risk_dual_use",
        3: "prohibited",
    }[probe_id]

    neg = {
        1: {"benign"},
        2: {"dual_use", "benign"},
        3: {"high_risk_dual_use", "dual_use", "benign"},
    }[probe_id]

    if category == pos:
        return 1
    if category in neg:
        return 0
    return None


def infer_category(sample: Dict[str, Any]) -> Optional[str]:
    return sample.get("category") or sample.get("label")


def infer_is_refusal(sample: Dict[str, Any]) -> Optional[int]:
    if "is_refusal" in sample:
        return int(sample["is_refusal"])
    if "refusal" in sample:
        return int(sample["refusal"])
    return None


def merge_extract_metadata(sample: Dict[str, Any], ex: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(sample)

    for key in [
        "category",
        "label",
        "is_refusal",
        "refusal",
        "model_key",
        "sample_id",
        "prompt",
        "attack_prompt",
    ]:
        if key in ex and key not in merged:
            merged[key] = ex[key]

    return merged


def build_feature_table(
    samples: List[Dict[str, Any]],
    extracts_dir: Path,
    trait_mat: torch.Tensor,
    axis: Optional[torch.Tensor],
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    rows = []
    feats = []
    skipped = 0

    for s in samples:
        sample_id = s["sample_id"]
        ex = load_extract(extracts_dir, sample_id)

        if ex is None:
            skipped += 1
            continue

        try:
            hidden_by_layer = get_final_token_by_layer(ex)
            feat = projection_features(hidden_by_layer, trait_mat, axis)
        except Exception:
            skipped += 1
            continue

        rows.append(merge_extract_metadata(s, ex))
        feats.append(feat)

    if skipped:
        print(f"[warn] skipped {skipped} samples")

    if not feats:
        raise RuntimeError("No usable features built")

    return rows, np.stack(feats, axis=0)


def train_eval_task(
    task_name: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    y = y.astype(int)

    if len(set(y.tolist())) < 2:
        raise ValueError(f"{task_name}: only one class present")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=5000,
        class_weight="balanced",
        random_state=seed,
    )

    clf.fit(X_train_s, y_train)

    decision = clf.decision_function(X_test_s)
    prob = 1.0 / (1.0 + np.exp(-decision))
    pred = (prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, decision)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, zero_division=0)

    return {
        "task": task_name,
        "auc": float(auc),
        "acc": float(acc),
        "f1": float(f1),
        "n": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_rate": float(y.mean()),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "coef": clf.coef_[0].astype(np.float32),
        "intercept": float(clf.intercept_[0]),
    }


def json_safe_task_summary(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v for k, v in task.items()
        if k not in {"scaler_mean", "scaler_scale", "coef"}
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracts_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--trait_vector_dir", required=True)
    ap.add_argument("--assistant_axis_path", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    extracts_dir = Path(args.extracts_dir)
    manifest_path = Path(args.manifest)
    trait_vector_dir = Path(args.trait_vector_dir)
    axis_path = Path(args.assistant_axis_path) if args.assistant_axis_path else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading trait vectors...")
    trait_mat, trait_names = load_trait_matrix(trait_vector_dir)

    print("Loading assistant axis...")
    axis = load_axis(axis_path)

    print(f"trait_mat shape: {tuple(trait_mat.shape)}")
    print(f"n_traits: {len(trait_names)}")
    print(f"axis: {'yes' if axis is not None else 'no'}")

    print("Loading manifest samples...")
    samples = load_manifest_samples(manifest_path)
    print(f"manifest samples: {len(samples)}")

    print("Building projection features...")
    rows, X = build_feature_table(samples, extracts_dir, trait_mat, axis)
    print(f"feature table: X={X.shape}, rows={len(rows)}")

    tasks = {}

    # Refusal-Gemma
    idxs = []
    ys = []

    for i, row in enumerate(rows):
        label = infer_is_refusal(row)
        if label is not None:
            idxs.append(i)
            ys.append(label)

    if ys and len(set(ys)) == 2:
        idx = np.array(idxs)
        task_name = "refusal_gemma4_31b"
        tasks[task_name] = train_eval_task(task_name, X[idx], np.array(ys), args.seed)
        print(f"{task_name}: AUC={tasks[task_name]['auc']:.4f}, n={tasks[task_name]['n']}")
    else:
        print("[warn] no usable refusal_gemma4_31b task found")

    # Cyber Probe 1/2/3, only if cyber category labels exist in this manifest/extract set.
    for probe_id in [1, 2, 3]:
        idxs = []
        ys = []

        for i, row in enumerate(rows):
            category = infer_category(row)
            if category is None:
                continue

            label = cyber_probe_label(category, probe_id)
            if label is None:
                continue

            idxs.append(i)
            ys.append(label)

        if ys and len(set(ys)) == 2:
            task_name = {
                1: "cyber_probe_1_dual_use_vs_benign",
                2: "cyber_probe_2_hdu_vs_du_benign",
                3: "cyber_probe_3_prohibited_vs_rest",
            }[probe_id]
            idx = np.array(idxs)
            tasks[task_name] = train_eval_task(task_name, X[idx], np.array(ys), args.seed)
            print(f"{task_name}: AUC={tasks[task_name]['auc']:.4f}, n={tasks[task_name]['n']}")
        else:
            print(f"[warn] no usable cyber probe {probe_id} task found")

    if not tasks:
        raise RuntimeError("No tasks were trained")

    summary = {
        name: json_safe_task_summary(task)
        for name, task in tasks.items()
    }

    mean_auc = float(np.mean([task["auc"] for task in tasks.values()]))
    summary["_mean_auc_over_trained_tasks"] = mean_auc

    package = {
        "method": "gemma_trait_projection_logistic_regression",
        "vector_layers": DEFAULT_VECTOR_LAYERS,
        "trait_names": trait_names,
        "feature_dim": int(X.shape[1]),
        "uses_axis": axis is not None,
        "tasks": tasks,
        "summary": summary,
        "extracts_dir": str(extracts_dir),
        "manifest": str(manifest_path),
        "trait_vector_dir": str(trait_vector_dir),
        "assistant_axis_path": str(axis_path) if axis_path else None,
    }

    torch.save(package, out_dir / "gemma_trait_projection_probe.pt")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(out_dir / "gemma_trait_projection_probe.pt")
    print(out_dir / "summary.json")

    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
