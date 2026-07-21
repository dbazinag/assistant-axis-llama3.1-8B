#!/usr/bin/env python3
"""
run_baseline_jlt_hidden_tensor.py

Jailbreaking-Leaves-a-Trace-style hidden-state tensor baseline.

This is not a full reproduction of JLT's MHA/LO tensor pipeline. It is the
closest honest baseline supported by the activations already collected for RTV:

  X: samples × 3 RTV layers × 5 prompt-token positions × hidden_dim

For each HarmBench train split, this script:
  1. flattens the hidden-state tensor,
  2. standardizes using HarmBench train statistics,
  3. applies PCA to obtain low-dimensional latent tensor factors,
  4. trains supervised classifiers on successful-vs-failed HarmBench jailbreaks,
  5. evaluates unchanged on held-out HarmBench and transfer attack families.

The classifier set mirrors the JLT paper family: SVM-RBF, Random Forest, and
Logistic Regression. The experimental split mirrors this repo: train/calibrate
only on HarmBench human_jailbreak train pools, then transfer to GCG/PAIR/PAP/
GPTFuzz/PEZ.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PAPER_LAYERS = [18, 25, 32]
N_LAST_TOKENS = 5
TRAIN_FRAC = 0.7
N_SEEDS = 20
RANDOM_SEED = 42
N_COMPONENTS = 128

DATASETS = {
    "HarmBench": "harmbench",
    "GCG": "gcg",
    "PAIR": "pair",
    "PAP": "pap",
    "GPTFuzz": "gptfuzz",
    "PEZ": "pez",
}


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bool_label(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return int(value.lower() == "true")
    return int(bool(value))


def load_dataset_tensor(root: Path, dataset: str):
    rows = load_jsonl(root / f"{dataset}_rows.jsonl")
    acts = torch.load(root / f"{dataset}_activations.pt", map_location="cpu", weights_only=False)
    X, valid_rows = [], []
    for row in rows:
        pid = row["pair_id"]
        if pid not in acts:
            continue
        try:
            sample = np.stack([acts[pid][str(layer)].float().numpy() for layer in PAPER_LAYERS])
        except KeyError:
            continue
        if sample.shape[0] != len(PAPER_LAYERS) or sample.shape[1] != N_LAST_TOKENS:
            continue
        X.append(sample.reshape(-1))
        valid_rows.append(row)
    if not X:
        raise ValueError(f"No valid activations for {dataset}")
    return np.stack(X).astype(np.float32), valid_rows


def labels_for_rows(rows: List[dict]) -> np.ndarray:
    return np.array([bool_label(row.get("jailbroken", False)) for row in rows], dtype=int)


def get_pool_split(rows: List[dict], seed: int):
    rng = random.Random(seed)
    behaviors = sorted({r["behavior_id"] for r in rows})
    templates = sorted({r["jailbreak_idx"] for r in rows})
    rng.shuffle(behaviors)
    rng.shuffle(templates)
    n_beh = max(1, int(len(behaviors) * TRAIN_FRAC))
    n_tpl = max(1, int(len(templates) * TRAIN_FRAC))
    return (
        set(behaviors[:n_beh]),
        set(templates[:n_tpl]),
        set(behaviors[n_beh:]),
        set(templates[n_tpl:]),
    )


def split_human_rows(rows: List[dict], train_beh, train_tpl, val_beh, val_tpl):
    train_idx = [
        i for i, r in enumerate(rows)
        if r.get("attack_type") == "human_jailbreak"
        and r["behavior_id"] in train_beh
        and r["jailbreak_idx"] in train_tpl
    ]
    val_idx = [
        i for i, r in enumerate(rows)
        if r.get("attack_type") == "human_jailbreak"
        and r["behavior_id"] in val_beh
        and r["jailbreak_idx"] in val_tpl
    ]
    return train_idx, val_idx


def best_threshold(scores: np.ndarray, y: np.ndarray):
    values = np.unique(scores)
    if len(values) > 200:
        values = np.quantile(scores, np.linspace(0.01, 0.99, 199))
    best_t, best_bacc = float(values[0]), -1.0
    for t in values:
        pred = (scores >= t).astype(int)
        bacc = balanced_accuracy_score(y, pred)
        if bacc > best_bacc:
            best_t, best_bacc = float(t), float(bacc)
    return best_t, best_bacc


def metric_dict(scores: np.ndarray, y: np.ndarray, threshold: float):
    if len(scores) == 0 or len(set(y.tolist())) < 2:
        return {"auc": float("nan"), "ap": float("nan"), "balanced_acc": float("nan")}
    pred = (scores >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y, scores)),
        "ap": float(average_precision_score(y, scores)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
    }


def summarize(values: Iterable[float]):
    arr = np.array(list(values), dtype=float)
    return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)), "all": arr.tolist()}


def make_models(seed: int, fast: bool):
    rf_trees = 150 if fast else 300
    return {
        "pca_svm_rbf": make_pipeline(
            StandardScaler(),
            SVC(C=3.0, gamma="scale", kernel="rbf", probability=False, class_weight="balanced"),
        ),
        "pca_random_forest": RandomForestClassifier(
            n_estimators=rf_trees,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "pca_logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed),
        ),
    }


def model_scores(model, X: np.ndarray):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # Pipeline exposes these methods through its final estimator.
    if hasattr(model, "named_steps"):
        final = list(model.named_steps.values())[-1]
        if hasattr(model, "decision_function"):
            return model.decision_function(X)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        raise TypeError(f"Unsupported pipeline final estimator: {type(final)}")
    raise TypeError(f"Unsupported model type: {type(model)}")


def print_table(final: dict):
    print("\n" + "=" * 100)
    print("  JLT-STYLE HIDDEN-STATE TENSOR TRANSFER BASELINE")
    print("  Train/calibrate: HarmBench human_jailbreak train pools only")
    print("=" * 100)
    for model_name, result in final["models"].items():
        print(f"\n  {model_name}")
        print(f"  Train balanced acc: {result['train_balanced_acc']['mean']:.4f} ± {result['train_balanced_acc']['std']:.4f}")
        print(f"  {'Family':12s} {'AUC':>9s} {'AP':>9s} {'BAcc':>9s}")
        print("  " + "-" * 40)
        for family, vals in result["datasets"].items():
            print(
                f"  {family:12s} "
                f"{vals['auc']['mean']:9.4f} "
                f"{vals['ap']['mean']:9.4f} "
                f"{vals['balanced_acc']['mean']:9.4f}"
            )
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtv_dir", default="full_trait_output/rtv_activations")
    parser.add_argument("--output_dir", default="full_trait_output/baselines_jlt_hidden_tensor")
    parser.add_argument("--n_components", type=int, default=N_COMPONENTS)
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--olmo3", action="store_true",
                        help="Use OLMo-3 RTV caches (GCG excluded; not yet collected)")
    parser.add_argument("--gemma", action="store_true",
                        help="Use Gemma-4-31B RTV caches (GCG excluded; not collected)")
    parser.add_argument("--augment_wjb", action="store_true",
                        help="Fold the wildjailbreak RTV cache into the TRAINING set each seed "
                             "(HB+WJB); eval families unchanged. Output dir gets a _hbwjb suffix.")
    args = parser.parse_args()

    if args.olmo3:
        if args.rtv_dir == "full_trait_output/rtv_activations":
            args.rtv_dir = "full_trait_output/rtv_activations_olmo3"
        if args.output_dir == "full_trait_output/baselines_jlt_hidden_tensor":
            args.output_dir = "full_trait_output/baselines_jlt_hidden_tensor_olmo3"
    if args.gemma:
        if args.rtv_dir == "full_trait_output/rtv_activations":
            args.rtv_dir = "full_trait_output/rtv_activations_gemma"
        if args.output_dir == "full_trait_output/baselines_jlt_hidden_tensor":
            args.output_dir = "full_trait_output/baselines_jlt_hidden_tensor_gemma"
    datasets = {k: v for k, v in DATASETS.items() if not ((args.olmo3 or args.gemma) and k == "GCG")}

    rtv_dir = Path(args.rtv_dir)
    if args.test:
        rtv_dir = rtv_dir.parent / f"{rtv_dir.name}_test"
    output_dir = Path(args.output_dir)
    if args.augment_wjb:
        output_dir = output_dir.parent / f"{output_dir.name}_hbwjb"
    if args.test:
        output_dir = output_dir.parent / f"{output_dir.name}_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading hidden-state tensor caches ===", flush=True)
    X_hb_all, hb_rows_all = load_dataset_tensor(rtv_dir, "harmbench")
    y_hb_all = labels_for_rows(hb_rows_all)
    human_rows = [r for r in hb_rows_all if r.get("attack_type") == "human_jailbreak"]
    all_idx_by_pid = {r["pair_id"]: i for i, r in enumerate(hb_rows_all)}

    X_wj = y_wj = None
    if args.augment_wjb:
        X_wj, wj_rows = load_dataset_tensor(rtv_dir, "wildjailbreak")
        y_wj = labels_for_rows(wj_rows)
        print(f"  WJB train augment: {len(wj_rows)} rows, positive={y_wj.mean():.3f}", flush=True)

    transfer = {}
    for display, dataset in datasets.items():
        if dataset == "harmbench":
            continue
        X, rows = load_dataset_tensor(rtv_dir, dataset)
        y = labels_for_rows(rows)
        transfer[display] = (X, y)
        print(f"  {display}: {len(rows)} rows, positive={y.mean():.3f}", flush=True)

    model_names = ["pca_svm_rbf", "pca_random_forest", "pca_logreg"]
    accum = {
        name: {
            "train_balanced_acc": [],
            "datasets": {family: {"auc": [], "ap": [], "balanced_acc": []} for family in ["HarmBench", *transfer.keys()]},
        }
        for name in model_names
    }

    for seed in range(args.n_seeds):
        train_beh, train_tpl, val_beh, val_tpl = get_pool_split(human_rows, seed)
        train_local, val_local = split_human_rows(human_rows, train_beh, train_tpl, val_beh, val_tpl)
        train_idx = [all_idx_by_pid[human_rows[i]["pair_id"]] for i in train_local]
        val_idx = [all_idx_by_pid[human_rows[i]["pair_id"]] for i in val_local]
        if not train_idx or not val_idx or len(set(y_hb_all[train_idx].tolist())) < 2:
            continue

        X_train = X_hb_all[train_idx]
        y_train = y_hb_all[train_idx]
        X_val = X_hb_all[val_idx]
        y_val = y_hb_all[val_idx]

        if X_wj is not None:  # fold WJB into training only (PCA below is fit on it too)
            X_train = np.vstack([X_train, X_wj])
            y_train = np.concatenate([y_train, y_wj])

        n_comp = min(args.n_components, X_train.shape[0] - 1, X_train.shape[1])
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
        X_train_p = pca.fit_transform(X_train)
        X_val_p = pca.transform(X_val)
        transfer_p = {family: (pca.transform(X), y) for family, (X, y) in transfer.items()}

        for model_name, model in make_models(seed, args.test).items():
            model.fit(X_train_p, y_train)
            train_scores = model_scores(model, X_train_p)
            threshold, train_bacc = best_threshold(train_scores, y_train)
            accum[model_name]["train_balanced_acc"].append(train_bacc)

            scores = model_scores(model, X_val_p)
            vals = metric_dict(scores, y_val, threshold)
            for k, v in vals.items():
                accum[model_name]["datasets"]["HarmBench"][k].append(v)

            for family, (X_p, y) in transfer_p.items():
                scores = model_scores(model, X_p)
                vals = metric_dict(scores, y, threshold)
                for k, v in vals.items():
                    accum[model_name]["datasets"][family][k].append(v)

        if seed % 5 == 0:
            print(f"  seed {seed} done", flush=True)

    final = {
        "method": "jlt_style_hidden_state_tensor",
        "paper_layers": PAPER_LAYERS,
        "n_last_tokens": N_LAST_TOKENS,
        "n_components": args.n_components,
        "n_seeds": args.n_seeds,
        "models": {
            name: {
                "train_balanced_acc": summarize(vals["train_balanced_acc"]),
                "datasets": {
                    family: {metric: summarize(v) for metric, v in metrics.items()}
                    for family, metrics in vals["datasets"].items()
                },
            }
            for name, vals in accum.items()
        },
    }

    with open(output_dir / "jlt_hidden_tensor_results.json", "w") as f:
        json.dump(final, f, indent=2)
    print_table(final)
    print(f"\nSaved to {output_dir / 'jlt_hidden_tensor_results.json'}", flush=True)


if __name__ == "__main__":
    main()
