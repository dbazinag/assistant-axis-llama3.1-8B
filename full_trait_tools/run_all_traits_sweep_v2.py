#!/usr/bin/env python3
"""
run_all_traits_sweep_v2.py

Wide hyperparameter sweep over all-traits projection features.

Key fixes over v1:
  - Threshold calibrated on REFIT model's val predictions (not stale inner model).
  - AUC reported as primary metric alongside balanced_acc.
  - Table sorted by mean transfer AUC.
  - Much wider model/hyperparameter grid.

Protocol:
  - Outer pool split: HarmBench human_jailbreak behavior/template strict split.
  - Inner val fraction: drawn from outer train pool (for threshold only).
  - Model selection: by inner-val AUC.
  - All transfer families: evaluation only, never touch training.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


LAYER      = 16
TRAIN_FRAC = 0.7
VAL_FRAC   = 0.15
N_SEEDS    = 50
RANDOM_SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable[[int], object]


# ── Data loading ───────────────────────────────────────────────────────────────

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


def load_trait_matrix(layer: int) -> tuple[np.ndarray, list[str]]:
    mat_path   = Path(f"full_trait_output/trait_matrix_layer{layer}.npy")
    names_path = Path(f"full_trait_output/trait_names_layer{layer}.json")
    if not mat_path.exists() or not names_path.exists():
        raise FileNotFoundError(f"Missing cached trait matrix for layer {layer}")
    matrix = np.load(mat_path).astype(np.float32)
    names  = json.loads(names_path.read_text())
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    return matrix, names


def build_activation_matrix(
    rows: list[dict], activations: dict, layer: int
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    layer_key = str(layer)
    xs, ys, valid = [], [], []
    for row in rows:
        pid   = row.get("pair_id")
        label = row.get("jailbroken")
        if pid is None or label is None:
            continue
        item = activations.get(pid)
        if item is None or layer_key not in item:
            continue
        xs.append(item[layer_key].float().numpy())
        ys.append(1 if label else 0)
        valid.append(row)
    if not xs:
        return np.empty((0, 0)), np.empty((0,)), []
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64), valid


def project_all_traits(x: np.ndarray, trait_matrix: np.ndarray) -> np.ndarray:
    return (x @ trait_matrix.T).astype(np.float32)


# ── Pool split ─────────────────────────────────────────────────────────────────

def get_pool_split(rows: list[dict], train_frac: float, seed: int):
    rng       = random.Random(seed)
    behaviors = sorted({r["behavior_id"] for r in rows})
    templates = sorted({r["jailbreak_idx"] for r in rows})
    rng.shuffle(behaviors)
    rng.shuffle(templates)
    n_beh = max(1, int(len(behaviors) * train_frac))
    n_tpl = max(1, int(len(templates) * train_frac))
    return (
        set(behaviors[:n_beh]),
        set(templates[:n_tpl]),
        set(behaviors[n_beh:]),
        set(templates[n_tpl:]),
    )


def split_by_pool(rows, train_beh, train_tpl, test_beh, test_tpl):
    train_idx = [
        i for i, r in enumerate(rows)
        if r["behavior_id"] in train_beh and r["jailbreak_idx"] in train_tpl
    ]
    test_idx = [
        i for i, r in enumerate(rows)
        if r["behavior_id"] in test_beh and r["jailbreak_idx"] in test_tpl
    ]
    return train_idx, test_idx


# ── Scoring ────────────────────────────────────────────────────────────────────

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


def best_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, int]:
    """Find threshold + sign maximising balanced accuracy."""
    candidates = np.unique(score)
    if len(candidates) > 500:
        candidates = np.quantile(candidates, np.linspace(0, 1, 500))
    best_bacc, best_thr, best_sign = -1.0, 0.0, 1
    for sign in (1, -1):
        s = sign * score
        for thr in candidates:
            bacc = float(balanced_accuracy_score(y_true, (s >= thr * sign).astype(int)
                         if sign == 1 else (score <= thr).astype(int)))
            if bacc > best_bacc:
                best_bacc, best_thr, best_sign = bacc, float(thr), sign
    return best_thr, best_sign


def get_score(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def eval_set(
    y_true: np.ndarray, score: np.ndarray, threshold: float, sign: int
) -> dict:
    pred = ((sign * score) >= (sign * threshold)).astype(int)
    return {
        "auc":          safe_auc(y_true, score),
        "ap":           safe_ap(y_true, score),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "n":            int(len(y_true)),
        "n_pos":        int(y_true.sum()),
    }


# ── Model grid ─────────────────────────────────────────────────────────────────

def build_specs(include_slow: bool) -> list[ModelSpec]:
    specs: list[ModelSpec] = []

    # Logistic regression — L2, wide C range, balanced and unbalanced
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        for cw in ["balanced", None]:
            tag = f"logreg_l2_C{C}_{'bal' if cw else 'raw'}"
            _C, _cw = C, cw
            specs.append(ModelSpec(tag, lambda seed, C=_C, cw=_cw: Pipeline([
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(
                    C=C, penalty="l2", solver="lbfgs",
                    max_iter=4000, class_weight=cw, random_state=seed,
                )),
            ])))

    # Logistic regression — L1
    for C in [0.01, 0.03, 0.1, 0.3, 1.0]:
        for cw in ["balanced", None]:
            tag = f"logreg_l1_C{C}_{'bal' if cw else 'raw'}"
            _C, _cw = C, cw
            specs.append(ModelSpec(tag, lambda seed, C=_C, cw=_cw: Pipeline([
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(
                    C=C, penalty="l1", solver="liblinear",
                    max_iter=4000, class_weight=cw, random_state=seed,
                )),
            ])))

    # Logistic regression — ElasticNet
    for C in [0.03, 0.1, 0.3]:
        for l1r in [0.3, 0.5, 0.7]:
            tag = f"logreg_en_C{C}_l1r{l1r}"
            _C, _l1r = C, l1r
            specs.append(ModelSpec(tag, lambda seed, C=_C, l1r=_l1r: Pipeline([
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(
                    C=C, penalty="elasticnet", solver="saga", l1_ratio=l1r,
                    max_iter=4000, class_weight="balanced", random_state=seed,
                )),
            ])))

    # Linear SVM
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        for cw in ["balanced", None]:
            tag = f"linsvm_C{C}_{'bal' if cw else 'raw'}"
            _C, _cw = C, cw
            specs.append(ModelSpec(tag, lambda seed, C=_C, cw=_cw: Pipeline([
                ("sc", StandardScaler()),
                ("clf", LinearSVC(
                    C=C, class_weight=cw, max_iter=10000, random_state=seed,
                )),
            ])))

    # Extra Trees
    for n_est, min_leaf in [(300, 2), (300, 5), (500, 3)]:
        tag = f"extratrees_{n_est}_{min_leaf}"
        _n, _m = n_est, min_leaf
        specs.append(ModelSpec(tag, lambda seed, n=_n, m=_m: ExtraTreesClassifier(
            n_estimators=n, max_features="sqrt", min_samples_leaf=m,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )))

    # MLP
    for arch, alpha in [((128,), 0.001), ((128, 64), 0.001), ((256, 128), 0.0003), ((64,), 0.01)]:
        tag = f"mlp_{'x'.join(str(a) for a in arch)}_a{alpha}"
        _arch, _alpha = arch, alpha
        specs.append(ModelSpec(tag, lambda seed, arch=_arch, alpha=_alpha: Pipeline([
            ("sc", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=arch, alpha=alpha,
                learning_rate_init=1e-3, max_iter=400,
                early_stopping=True, n_iter_no_change=20,
                random_state=seed,
            )),
        ])))

    # Gradient Boosting (sklearn — no GPU needed, usually strong on tabular)
    for n_est, lr, max_d in [(200, 0.05, 3), (300, 0.03, 4), (200, 0.1, 3)]:
        tag = f"gb_{n_est}_lr{lr}_d{max_d}"
        _n, _lr, _d = n_est, lr, max_d
        specs.append(ModelSpec(tag, lambda seed, n=_n, lr=_lr, d=_d: GradientBoostingClassifier(
            n_estimators=n, learning_rate=lr, max_depth=d,
            subsample=0.8, random_state=seed,
        )))

    # RBF SVM — slow but often best on moderate-dim tabular
    if include_slow:
        for C, gamma in [(1.0, "scale"), (3.0, "scale"), (10.0, "scale"), (1.0, "auto")]:
            tag = f"rbfsvm_C{C}_{gamma}"
            _C, _g = C, gamma
            specs.append(ModelSpec(tag, lambda seed, C=_C, g=_g: Pipeline([
                ("sc", StandardScaler()),
                ("clf", SVC(C=C, gamma=g, kernel="rbf",
                            class_weight="balanced", probability=False,
                            random_state=seed)),
            ])))

    return specs


# ── Main ───────────────────────────────────────────────────────────────────────

def summarize(vals: list[float]) -> dict:
    arr = np.array(vals, dtype=float)
    return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)), "all": vals}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_classified_path",
                        default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--human_activations_path",
                        default="full_trait_output/harmbench_activations/activations.pt")
    parser.add_argument("--gcg_classified_path",
                        default="full_trait_output/gcg_activations/responses.jsonl")
    parser.add_argument("--gcg_activations_path",
                        default="full_trait_output/gcg_activations/activations.pt")
    parser.add_argument("--pair_classified_path",
                        default="full_trait_output/pair_activations/responses.jsonl")
    parser.add_argument("--pair_activations_path",
                        default="full_trait_output/pair_activations/activations.pt")
    parser.add_argument("--pap_classified_path",
                        default="full_trait_output/pap_activations/responses.jsonl")
    parser.add_argument("--pap_activations_path",
                        default="full_trait_output/pap_activations/activations.pt")
    parser.add_argument("--gptfuzz_classified_path",
                        default="full_trait_output/gptfuzz_activations/responses.jsonl")
    parser.add_argument("--gptfuzz_activations_path",
                        default="full_trait_output/gptfuzz_activations/activations.pt")
    parser.add_argument("--pez_classified_path",
                        default="full_trait_output/pez_activations/responses.jsonl")
    parser.add_argument("--pez_activations_path",
                        default="full_trait_output/pez_activations/activations.pt")
    parser.add_argument("--output_dir",   default="full_trait_output/all_traits_sweep_v2")
    parser.add_argument("--layer",        type=int,   default=LAYER)
    parser.add_argument("--n_seeds",      type=int,   default=N_SEEDS)
    parser.add_argument("--train_frac",   type=float, default=TRAIN_FRAC)
    parser.add_argument("--val_frac",     type=float, default=VAL_FRAC)
    parser.add_argument("--include_slow", action="store_true",
                        help="Include RBF SVM (much slower).")
    parser.add_argument("--only_models",  default="",
                        help="Comma-separated subset of model names to run.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n=== Loading data ===", flush=True)
    human_rows = [
        r for r in load_jsonl(Path(args.human_classified_path))
        if r.get("attack_type") == "human_jailbreak"
    ]
    human_acts = load_activations(Path(args.human_activations_path))

    transfer_inputs = [
        ("GCG",     args.gcg_classified_path,     args.gcg_activations_path),
        ("PAIR",    args.pair_classified_path,     args.pair_activations_path),
        ("PAP",     args.pap_classified_path,      args.pap_activations_path),
        ("GPTFuzz", args.gptfuzz_classified_path,  args.gptfuzz_activations_path),
        ("PEZ",     args.pez_classified_path,      args.pez_activations_path),
    ]
    transfer_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, rp, ap in transfer_inputs:
        if Path(rp).exists() and Path(ap).exists():
            rows_ = load_jsonl(Path(rp))
            acts_ = load_activations(Path(ap))
            x_raw, y_, _ = build_activation_matrix(rows_, acts_, args.layer)
            if len(x_raw) > 0:
                print(f"  {name}: {len(x_raw)} rows, jb={y_.mean():.3f}", flush=True)
                transfer_data[name] = (x_raw, y_)

    trait_matrix, trait_names = load_trait_matrix(args.layer)
    print(f"  Trait matrix: {trait_matrix.shape}", flush=True)

    # ── Build features ─────────────────────────────────────────────────────────
    print("\n=== Building features ===", flush=True)
    x_raw_h, y_h, human_valid = build_activation_matrix(human_rows, human_acts, args.layer)
    x_h = project_all_traits(x_raw_h, trait_matrix)
    print(f"  HarmBench: {x_h.shape}, jb={y_h.mean():.3f}", flush=True)

    transfer: dict[str, tuple[np.ndarray, np.ndarray]] = {
        name: (project_all_traits(xr, trait_matrix), y)
        for name, (xr, y) in transfer_data.items()
    }

    # ── Model specs ────────────────────────────────────────────────────────────
    specs = build_specs(args.include_slow)
    if args.only_models:
        wanted = {s.strip() for s in args.only_models.split(",") if s.strip()}
        specs = [s for s in specs if s.name in wanted]
        if not specs:
            raise ValueError(f"No specs matched --only_models={args.only_models!r}")
    families = list(transfer.keys())
    print(f"\n=== {len(specs)} model configs × {args.n_seeds} seeds ===", flush=True)

    # ── Storage ────────────────────────────────────────────────────────────────
    raw: dict[str, dict] = {
        s.name: {
            "val_auc":    [],
            "human_test": [],
            **{name: [] for name in families},
        }
        for s in specs
    }
    selected_counts: dict[str, int] = {}

    t0 = time.time()
    for seed in range(args.n_seeds):
        train_beh, train_tpl, test_beh, test_tpl = get_pool_split(
            human_valid, args.train_frac, seed
        )
        train_idx, test_idx = split_by_pool(
            human_valid, train_beh, train_tpl, test_beh, test_tpl
        )
        if not train_idx or not test_idx or len(np.unique(y_h[train_idx])) < 2:
            continue

        # Inner val split — for model selection AND threshold calibration on refit model
        tr_idx, val_idx = train_test_split(
            np.array(train_idx), test_size=args.val_frac,
            random_state=seed, stratify=y_h[train_idx],
        )

        # --- Step 1: select best model by inner-val AUC ---
        val_aucs: dict[str, float] = {}
        for spec in specs:
            m = spec.builder(seed)
            m.fit(x_h[tr_idx], y_h[tr_idx])
            val_score = get_score(m, x_h[val_idx])
            val_aucs[spec.name] = safe_auc(y_h[val_idx], val_score)

        best_name = max(val_aucs, key=val_aucs.get)
        selected_counts[best_name] = selected_counts.get(best_name, 0) + 1

        # --- Step 2: refit every model on full train pool, calibrate threshold
        #             on val_idx predictions from the REFIT model (fixes staleness) ---
        for spec in specs:
            raw[spec.name]["val_auc"].append(val_aucs[spec.name])

            m_full = spec.builder(seed)
            m_full.fit(x_h[train_idx], y_h[train_idx])

            # Threshold from refit model's val predictions — not stale
            val_score_full = get_score(m_full, x_h[val_idx])
            threshold, sign = best_threshold(y_h[val_idx], val_score_full)

            # Human test
            human_score = get_score(m_full, x_h[test_idx])
            raw[spec.name]["human_test"].append(
                eval_set(y_h[test_idx], human_score, threshold, sign)
            )

            # Transfer families
            for name, (x_t, y_t) in transfer.items():
                t_score = get_score(m_full, x_t)
                raw[spec.name][name].append(
                    eval_set(y_t, t_score, threshold, sign)
                )

        if seed % 10 == 0:
            print(f"  Seed {seed}/{args.n_seeds} ({time.time()-t0:.1f}s)", flush=True)

    # ── Summarise ──────────────────────────────────────────────────────────────
    summary: dict[str, dict] = {}
    for spec in specs:
        r = raw[spec.name]
        summary[spec.name] = {
            "val_auc": summarize(r["val_auc"]),
            "human_test": {
                m: summarize([x[m] for x in r["human_test"]])
                for m in ("auc", "ap", "balanced_acc")
            },
        }
        for name in families:
            summary[spec.name][name] = {
                m: summarize([x[m] for x in r[name]])
                for m in ("auc", "ap", "balanced_acc")
            }

    # ── Print AUC table ────────────────────────────────────────────────────────
    col_w = 9
    header_cols = ["ValAUC", "Human"] + families + ["AvgXfer"]
    header = f"  {'Model':36s}" + "".join(f"{c:>{col_w}}" for c in header_cols)

    def avg_transfer_auc(name: str) -> float:
        vals = [summary[name][f]["auc"]["mean"] for f in families]
        return float(np.nanmean(vals))

    ranked = sorted([s.name for s in specs], key=avg_transfer_auc, reverse=True)

    print("\n" + "=" * (len(header) + 2))
    print(f"  ALL-TRAITS SWEEP v2 | layer {args.layer} | {args.n_seeds} seeds | PRIMARY METRIC: AUC")
    print("=" * (len(header) + 2))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in ranked[:30]:   # cap at 30 rows
        s = summary[name]
        val_auc   = s["val_auc"]["mean"]
        human_auc = s["human_test"]["auc"]["mean"]
        xfer_aucs = [s[f]["auc"]["mean"] for f in families]
        avg_xfer  = float(np.nanmean(xfer_aucs))
        row = (
            f"  {name:36s}"
            f"{val_auc:{col_w}.4f}"
            f"{human_auc:{col_w}.4f}"
            + "".join(f"{v:{col_w}.4f}" for v in xfer_aucs)
            + f"{avg_xfer:{col_w}.4f}"
        )
        print(row)
    print("=" * (len(header) + 2))

    # Also print balanced_acc table for reference
    print("\n" + "=" * (len(header) + 2))
    print("  SAME MODELS | BALANCED_ACC (threshold calibrated on refit model's val predictions)")
    print("=" * (len(header) + 2))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in ranked[:30]:
        s = summary[name]
        val_auc    = s["val_auc"]["mean"]
        human_bacc = s["human_test"]["balanced_acc"]["mean"]
        xfer_baccs = [s[f]["balanced_acc"]["mean"] for f in families]
        avg_xfer   = float(np.nanmean(xfer_baccs))
        row = (
            f"  {name:36s}"
            f"{val_auc:{col_w}.4f}"
            f"{human_bacc:{col_w}.4f}"
            + "".join(f"{v:{col_w}.4f}" for v in xfer_baccs)
            + f"{avg_xfer:{col_w}.4f}"
        )
        print(row)
    print(f"\n  Selected by val AUC: {dict(sorted(selected_counts.items(), key=lambda x: -x[1]))}")
    print("=" * (len(header) + 2))

    # ── Save ───────────────────────────────────────────────────────────────────
    out = {
        "method": "all_traits_sweep_v2",
        "layer":  args.layer,
        "n_seeds": args.n_seeds,
        "n_traits": len(trait_names),
        "families": families,
        "ranked_by_transfer_auc": ranked,
        "selected_by_val_auc": selected_counts,
        "summary": summary,
    }
    out_path = out_dir / "results.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
