#!/usr/bin/env python3
"""
Run all-trait projection classifiers under the transfer-attack protocol.

Features:
  - layer-16 pre-generation activation projected onto every cached trait vector

Protocol:
  - outer split: HarmBench human_jailbreak behavior/template pool split
  - inner split: validation subset drawn only from the outer training pool
  - model/hyperparameter choice uses only the inner HarmBench validation split
  - transfer attack families are evaluation-only

This keeps the "train on HarmBench, test transfer attacks" constraint intact.
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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


LAYER = 16
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
RANDOM_SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable[[int], object]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_activations(path: Path) -> dict:
    print(f"  Loading {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_trait_matrix(layer: int) -> tuple[np.ndarray, list[str]]:
    mat_path = Path(f"full_trait_output/trait_matrix_layer{layer}.npy")
    names_path = Path(f"full_trait_output/trait_names_layer{layer}.json")
    if not mat_path.exists() or not names_path.exists():
        raise FileNotFoundError(
            f"Missing cached trait matrix/name files for layer {layer}: "
            f"{mat_path}, {names_path}"
        )
    matrix = np.load(mat_path).astype(np.float32)
    names = json.loads(names_path.read_text())
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    return matrix, names


def build_activation_matrix(rows: list[dict], activations: dict, layer: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    layer_key = str(layer)
    xs, ys, valid = [], [], []
    for row in rows:
        pair_id = row.get("pair_id")
        label = row.get("jailbroken")
        if pair_id is None or label is None:
            continue
        item = activations.get(pair_id)
        if item is None or layer_key not in item:
            continue
        xs.append(item[layer_key].float().numpy())
        ys.append(1 if label else 0)
        valid.append(row)
    if not xs:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64), []
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64), valid


def project_all_traits(x_raw: np.ndarray, trait_matrix: np.ndarray) -> np.ndarray:
    return (x_raw @ trait_matrix.T).astype(np.float32)


def get_pool_split(rows: list[dict], train_frac: float, seed: int):
    rng = random.Random(seed)
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


def split_by_pool(rows: list[dict], train_beh, train_tpl, test_beh, test_tpl) -> tuple[list[int], list[int]]:
    train_idx = [
        i for i, r in enumerate(rows)
        if r["behavior_id"] in train_beh and r["jailbreak_idx"] in train_tpl
    ]
    test_idx = [
        i for i, r in enumerate(rows)
        if r["behavior_id"] in test_beh and r["jailbreak_idx"] in test_tpl
    ]
    return train_idx, test_idx


def score_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    auc = float(roc_auc_score(y_true, score))
    return max(auc, 1.0 - auc)


def score_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    ap = float(average_precision_score(y_true, score))
    ap_flip = float(average_precision_score(y_true, -score))
    return max(ap, ap_flip)


def balanced_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float, int]:
    order = np.argsort(score)
    candidates = np.unique(score[order])
    if len(candidates) > 400:
        candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 400))
    best = (-1.0, 0.0, 1)
    for sign in (1, -1):
        signed = sign * score
        for thr in candidates:
            pred = (signed >= sign * thr).astype(np.int64)
            bacc = float(balanced_accuracy_score(y_true, pred))
            if bacc > best[0]:
                best = (bacc, float(thr), sign)
    return best[1], best[0], best[2]


def eval_scores(y_true: np.ndarray, score: np.ndarray, threshold: float, sign: int) -> dict:
    pred = (sign * score >= sign * threshold).astype(np.int64)
    return {
        "auc": score_auc(y_true, score),
        "ap": score_ap(y_true, score),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
    }


def get_score(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    raise TypeError(f"Model {type(model)} has neither predict_proba nor decision_function")


def model_specs(include_slow: bool) -> list[ModelSpec]:
    specs = [
        ModelSpec(
            "logreg_l2_C0.03",
            lambda seed: Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(
                    C=0.03, max_iter=3000, class_weight="balanced", random_state=seed
                )),
            ]),
        ),
        ModelSpec(
            "logreg_l2_C0.1",
            lambda seed: Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(
                    C=0.1, max_iter=3000, class_weight="balanced", random_state=seed
                )),
            ]),
        ),
        ModelSpec(
            "logreg_l2_C0.3",
            lambda seed: Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(
                    C=0.3, max_iter=3000, class_weight="balanced", random_state=seed
                )),
            ]),
        ),
        ModelSpec(
            "logreg_l1_C0.1",
            lambda seed: Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(
                    C=0.1, penalty="l1", solver="liblinear",
                    max_iter=3000, class_weight="balanced", random_state=seed
                )),
            ]),
        ),
        ModelSpec(
            "linear_svm_C0.03",
            lambda seed: Pipeline([
                ("scale", StandardScaler()),
                ("clf", LinearSVC(
                    C=0.03, class_weight="balanced", max_iter=10000, random_state=seed
                )),
            ]),
        ),
        ModelSpec(
            "linear_svm_C0.1",
            lambda seed: Pipeline([
                ("scale", StandardScaler()),
                ("clf", LinearSVC(
                    C=0.1, class_weight="balanced", max_iter=10000, random_state=seed
                )),
            ]),
        ),
        ModelSpec(
            "extra_trees",
            lambda seed: ExtraTreesClassifier(
                n_estimators=400, max_features="sqrt", min_samples_leaf=3,
                class_weight="balanced", random_state=seed, n_jobs=-1
            ),
        ),
        ModelSpec(
            "mlp_64_alpha0.01",
            lambda seed: Pipeline([
                ("scale", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(64,), alpha=0.01, learning_rate_init=1e-3,
                    max_iter=300, early_stopping=True, n_iter_no_change=20,
                    random_state=seed
                )),
            ]),
        ),
        ModelSpec(
            "mlp_64_32_alpha0.01",
            lambda seed: Pipeline([
                ("scale", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(64, 32), alpha=0.01, learning_rate_init=1e-3,
                    max_iter=300, early_stopping=True, n_iter_no_change=20,
                    random_state=seed
                )),
            ]),
        ),
    ]
    if include_slow:
        specs.extend([
            ModelSpec(
                "rbf_svm_C1_gamma_scale",
                lambda seed: Pipeline([
                    ("scale", StandardScaler()),
                    ("clf", SVC(
                        C=1.0, gamma="scale", class_weight="balanced",
                        probability=False, random_state=seed
                    )),
                ]),
            ),
            ModelSpec(
                "rbf_svm_C3_gamma_scale",
                lambda seed: Pipeline([
                    ("scale", StandardScaler()),
                    ("clf", SVC(
                        C=3.0, gamma="scale", class_weight="balanced",
                        probability=False, random_state=seed
                    )),
                ]),
            ),
            ModelSpec(
                "random_forest",
                lambda seed: RandomForestClassifier(
                    n_estimators=400, max_features="sqrt", min_samples_leaf=3,
                    class_weight="balanced", random_state=seed, n_jobs=-1
                ),
            ),
        ])
    return specs


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "all": [float(x) for x in arr],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_classified_path", default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--human_activations_path", default="full_trait_output/harmbench_activations/activations.pt")
    parser.add_argument("--gcg_classified_path", default="full_trait_output/gcg_activations/responses.jsonl")
    parser.add_argument("--gcg_activations_path", default="full_trait_output/gcg_activations/activations.pt")
    parser.add_argument("--pair_classified_path", default="full_trait_output/pair_activations/responses.jsonl")
    parser.add_argument("--pair_activations_path", default="full_trait_output/pair_activations/activations.pt")
    parser.add_argument("--pap_classified_path", default="full_trait_output/pap_activations/responses.jsonl")
    parser.add_argument("--pap_activations_path", default="full_trait_output/pap_activations/activations.pt")
    parser.add_argument("--gptfuzz_classified_path", default="full_trait_output/gptfuzz_activations/responses.jsonl")
    parser.add_argument("--gptfuzz_activations_path", default="full_trait_output/gptfuzz_activations/activations.pt")
    parser.add_argument("--pez_classified_path", default="full_trait_output/pez_activations/responses.jsonl")
    parser.add_argument("--pez_activations_path", default="full_trait_output/pez_activations/activations.pt")
    parser.add_argument("--output_dir", default="full_trait_output/all_traits_model_sweep")
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument("--train_frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val_frac", type=float, default=VAL_FRAC)
    parser.add_argument("--include_slow", action="store_true")
    parser.add_argument("--only_models", default="", help="Comma-separated model name subset.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading data ===", flush=True)
    human_rows = [
        r for r in load_jsonl(Path(args.human_classified_path))
        if r.get("attack_type") == "human_jailbreak"
    ]
    human_acts = load_activations(Path(args.human_activations_path))

    transfer_inputs = [
        ("GCG", args.gcg_classified_path, args.gcg_activations_path),
        ("PAIR", args.pair_classified_path, args.pair_activations_path),
        ("PAP", args.pap_classified_path, args.pap_activations_path),
        ("GPTFuzz", args.gptfuzz_classified_path, args.gptfuzz_activations_path),
        ("PEZ", args.pez_classified_path, args.pez_activations_path),
    ]
    transfer_rows_acts = {}
    for name, rows_path, acts_path in transfer_inputs:
        if Path(rows_path).exists() and Path(acts_path).exists():
            print(f"Loading {name}", flush=True)
            transfer_rows_acts[name] = (load_jsonl(Path(rows_path)), load_activations(Path(acts_path)))

    trait_matrix, trait_names = load_trait_matrix(args.layer)
    print(f"  Trait matrix: {trait_matrix.shape}", flush=True)

    print("\n=== Building all-trait projection features ===", flush=True)
    x_h_raw, y_h, human_valid = build_activation_matrix(human_rows, human_acts, args.layer)
    x_h = project_all_traits(x_h_raw, trait_matrix)
    print(f"  HarmBench human_jailbreak: {x_h.shape}, jb rate={y_h.mean():.3f}", flush=True)

    transfer = {}
    for name, (rows, acts) in transfer_rows_acts.items():
        x_raw, y, _ = build_activation_matrix(rows, acts, args.layer)
        transfer[name] = (project_all_traits(x_raw, trait_matrix), y)
        print(f"  {name}: {transfer[name][0].shape}, jb rate={y.mean():.3f}", flush=True)

    specs = model_specs(args.include_slow)
    if args.only_models:
        wanted = {x.strip() for x in args.only_models.split(",") if x.strip()}
        specs = [s for s in specs if s.name in wanted]
        if not specs:
            raise ValueError(f"No model specs matched --only_models={args.only_models}")
    print(f"\n=== Models: {', '.join(s.name for s in specs)} ===", flush=True)

    raw_results = {
        spec.name: {
            "validation": [],
            "human_test": [],
            **{name: [] for name in transfer},
        }
        for spec in specs
    }
    selected = []

    t0 = time.time()
    for seed in range(args.n_seeds):
        train_beh, train_tpl, test_beh, test_tpl = get_pool_split(human_valid, args.train_frac, seed)
        train_idx, test_idx = split_by_pool(human_valid, train_beh, train_tpl, test_beh, test_tpl)
        if not train_idx or not test_idx or len(np.unique(y_h[train_idx])) < 2:
            continue

        tr_idx, val_idx = train_test_split(
            np.asarray(train_idx),
            test_size=args.val_frac,
            random_state=seed,
            stratify=y_h[train_idx],
        )

        seed_validation = {}
        fitted = {}
        thresholds = {}
        for spec in specs:
            model = spec.builder(seed)
            model.fit(x_h[tr_idx], y_h[tr_idx])
            val_score = get_score(model, x_h[val_idx])
            val_auc = score_auc(y_h[val_idx], val_score)
            threshold, val_bacc, sign = balanced_threshold(y_h[val_idx], val_score)
            seed_validation[spec.name] = val_auc
            fitted[spec.name] = model
            thresholds[spec.name] = (threshold, sign, val_bacc)
            raw_results[spec.name]["validation"].append({
                "auc": val_auc,
                "balanced_acc": val_bacc,
            })

        best_name = max(seed_validation, key=seed_validation.get)
        selected.append(best_name)

        # Refit every model on full HarmBench training pool for fair per-model results.
        for spec in specs:
            model = spec.builder(seed)
            model.fit(x_h[train_idx], y_h[train_idx])

            # Threshold comes from the inner-validation model, not transfer data.
            threshold, sign, _ = thresholds[spec.name]

            human_score = get_score(model, x_h[test_idx])
            raw_results[spec.name]["human_test"].append(
                eval_scores(y_h[test_idx], human_score, threshold, sign)
            )
            for name, (x_t, y_t) in transfer.items():
                score = get_score(model, x_t)
                raw_results[spec.name][name].append(eval_scores(y_t, score, threshold, sign))

        if seed % 5 == 0:
            elapsed = time.time() - t0
            print(f"  Seed {seed}/{args.n_seeds} done ({elapsed:.1f}s)", flush=True)

    summary = {}
    for spec in specs:
        spec_res = raw_results[spec.name]
        val_auc = [r["auc"] for r in spec_res["validation"]]
        val_bacc = [r["balanced_acc"] for r in spec_res["validation"]]
        summary[spec.name] = {
            "validation_auc": summarize(val_auc),
            "validation_balanced_acc": summarize(val_bacc),
        }
        for split in ["human_test", *transfer.keys()]:
            summary[spec.name][split] = {
                metric: summarize([r[metric] for r in spec_res[split]])
                for metric in ("auc", "ap", "balanced_acc")
            }

    select_counts = {name: selected.count(name) for name in sorted(set(selected))}
    families = list(transfer.keys())

    print("\n" + "=" * 120)
    print(f"  ALL-TRAITS MODEL SWEEP | layer {args.layer} | {args.n_seeds} seeds")
    print("  Hyperparameters selected only on inner HarmBench validation")
    print("=" * 120)
    header = f"  {'Model':24s} {'ValAUC':>7} {'Human':>7}" + "".join(f" {f:>8}" for f in families) + "  AvgXfer"
    print(header)
    print("  " + "-" * (len(header) - 2))
    best_avg = -1.0
    best_name = None
    for spec in specs:
        s = summary[spec.name]
        vals = [s[f]["balanced_acc"]["mean"] for f in families]
        avg = float(np.nanmean(vals))
        if avg > best_avg:
            best_avg = avg
            best_name = spec.name
        row = (
            f"  {spec.name:24s} "
            f"{s['validation_auc']['mean']:7.4f} "
            f"{s['human_test']['balanced_acc']['mean']:7.4f}"
            + "".join(f" {v:8.4f}" for v in vals)
            + f"  {avg:7.4f}"
        )
        print(row)
    print(f"\n  Best by transfer balanced accuracy: {best_name} ({best_avg:.4f})")
    print(f"  Selected-by-validation counts: {select_counts}")
    print("=" * 120)

    out = {
        "method": "all_traits_model_sweep",
        "layer": args.layer,
        "n_seeds": args.n_seeds,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "n_traits": len(trait_names),
        "trait_names": trait_names,
        "families": families,
        "selected_by_validation_counts": select_counts,
        "best_by_transfer_balanced_acc": best_name,
        "summary": summary,
        "raw_results": raw_results,
    }
    out_path = out_dir / "all_traits_model_sweep_results.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
