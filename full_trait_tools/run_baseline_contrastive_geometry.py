#!/usr/bin/env python3
"""
run_baseline_contrastive_geometry.py

Contrastive representation-space baselines for jailbreak success prediction.

This is the task-matched version of the Mahalanobis idea:
  - Train only on HarmBench human_jailbreak train pools.
  - Classes are failed jailbreak attempts vs successful jailbreak attempts.
  - Evaluate unchanged on held-out HarmBench and transfer attacks.

Methods:
  MCD: Mahalanobis Contrastive Detection
       score = distance_to_failed_class - distance_to_success_class

  KCD: K-nearest Contrastive Detection
       score = mean_k_distance_to_failed_class - mean_k_distance_to_success_class

Features use the already-collected RTV hidden tensor cache:
  samples × layers(18,25,32) × last_5_prompt_tokens × hidden_dim
flattened, standardized on the train split, and PCA-reduced before scoring.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PAPER_LAYERS = [18, 25, 32]
TRAIN_FRAC = 0.7
N_SEEDS = 50
N_COMPONENTS = 128
K_NEIGHBORS = 15

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


def load_dataset_matrix(root: Path, dataset: str):
    rows = load_jsonl(root / f"{dataset}_rows.jsonl")
    activations = torch.load(root / f"{dataset}_activations.pt", map_location="cpu", weights_only=False)
    X, valid_rows = [], []
    for row in rows:
        pid = row["pair_id"]
        if pid not in activations:
            continue
        try:
            sample = np.stack([activations[pid][str(layer)].float().numpy() for layer in PAPER_LAYERS])
        except KeyError:
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
    train = [
        i for i, r in enumerate(rows)
        if r.get("attack_type") == "human_jailbreak"
        and r["behavior_id"] in train_beh
        and r["jailbreak_idx"] in train_tpl
    ]
    val = [
        i for i, r in enumerate(rows)
        if r.get("attack_type") == "human_jailbreak"
        and r["behavior_id"] in val_beh
        and r["jailbreak_idx"] in val_tpl
    ]
    return train, val


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


class MCD:
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mu_failed = X[y == 0].mean(axis=0)
        self.mu_success = X[y == 1].mean(axis=0)
        self.cov_failed = LedoitWolf().fit(X[y == 0])
        self.cov_success = LedoitWolf().fit(X[y == 1])
        return self

    @staticmethod
    def _dist(X: np.ndarray, mean: np.ndarray, precision: np.ndarray):
        diff = X - mean
        d2 = np.einsum("ij,jk,ik->i", diff, precision, diff)
        return np.sqrt(np.maximum(d2, 0.0))

    def score(self, X: np.ndarray):
        d_failed = self._dist(X, self.mu_failed, self.cov_failed.precision_)
        d_success = self._dist(X, self.mu_success, self.cov_success.precision_)
        return d_failed - d_success


class KCD:
    def __init__(self, k: int):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_failed = X[y == 0]
        self.X_success = X[y == 1]
        self.k_failed = min(self.k, len(self.X_failed))
        self.k_success = min(self.k, len(self.X_success))
        self.nn_failed = NearestNeighbors(n_neighbors=self.k_failed, metric="euclidean").fit(self.X_failed)
        self.nn_success = NearestNeighbors(n_neighbors=self.k_success, metric="euclidean").fit(self.X_success)
        return self

    def score(self, X: np.ndarray):
        d_failed = self.nn_failed.kneighbors(X, return_distance=True)[0].mean(axis=1)
        d_success = self.nn_success.kneighbors(X, return_distance=True)[0].mean(axis=1)
        return d_failed - d_success


def print_table(final: dict):
    print("\n" + "=" * 100)
    print("  CONTRASTIVE GEOMETRY TRANSFER BASELINES")
    print("  Train/calibrate: HarmBench successful vs failed human_jailbreak train pools only")
    print("=" * 100)
    for method, result in final["methods"].items():
        print(f"\n  {method}")
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
    parser.add_argument("--output_dir", default="full_trait_output/baselines_contrastive_geometry")
    parser.add_argument("--n_components", type=int, default=N_COMPONENTS)
    parser.add_argument("--k_neighbors", type=int, default=K_NEIGHBORS)
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
        if args.output_dir == "full_trait_output/baselines_contrastive_geometry":
            args.output_dir = "full_trait_output/baselines_contrastive_geometry_olmo3"
    if args.gemma:
        if args.rtv_dir == "full_trait_output/rtv_activations":
            args.rtv_dir = "full_trait_output/rtv_activations_gemma"
        if args.output_dir == "full_trait_output/baselines_contrastive_geometry":
            args.output_dir = "full_trait_output/baselines_contrastive_geometry_gemma"
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

    print("\n=== Loading hidden tensor caches ===", flush=True)
    X_hb_all, hb_rows_all = load_dataset_matrix(rtv_dir, "harmbench")
    y_hb_all = labels_for_rows(hb_rows_all)
    human_rows = [r for r in hb_rows_all if r.get("attack_type") == "human_jailbreak"]
    idx_by_pid = {r["pair_id"]: i for i, r in enumerate(hb_rows_all)}

    X_wj = y_wj = None
    if args.augment_wjb:
        X_wj, wj_rows = load_dataset_matrix(rtv_dir, "wildjailbreak")
        y_wj = labels_for_rows(wj_rows)
        print(f"  WJB train augment: {len(wj_rows)} rows, positive={y_wj.mean():.3f}", flush=True)

    transfer = {}
    for family, dataset in datasets.items():
        if dataset == "harmbench":
            continue
        X, rows = load_dataset_matrix(rtv_dir, dataset)
        y = labels_for_rows(rows)
        transfer[family] = (X, y)
        print(f"  {family}: {len(rows)} rows, positive={y.mean():.3f}", flush=True)

    methods = ["mcd", "kcd"]
    accum = {
        method: {
            "train_balanced_acc": [],
            "datasets": {family: {"auc": [], "ap": [], "balanced_acc": []} for family in ["HarmBench", *transfer.keys()]},
        }
        for method in methods
    }

    for seed in range(args.n_seeds):
        train_beh, train_tpl, val_beh, val_tpl = get_pool_split(human_rows, seed)
        train_local, val_local = split_human_rows(human_rows, train_beh, train_tpl, val_beh, val_tpl)
        train_idx = [idx_by_pid[human_rows[i]["pair_id"]] for i in train_local]
        val_idx = [idx_by_pid[human_rows[i]["pair_id"]] for i in val_local]
        if not train_idx or not val_idx or len(set(y_hb_all[train_idx].tolist())) < 2:
            continue

        X_train = X_hb_all[train_idx]
        y_train = y_hb_all[train_idx]
        X_val = X_hb_all[val_idx]
        y_val = y_hb_all[val_idx]

        if X_wj is not None:  # fold WJB into training only (scaler/PCA below see it too)
            X_train = np.vstack([X_train, X_wj])
            y_train = np.concatenate([y_train, y_wj])

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        transfer_s = {family: (scaler.transform(X), y) for family, (X, y) in transfer.items()}

        n_comp = min(args.n_components, X_train_s.shape[0] - 1, X_train_s.shape[1])
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
        X_train_p = pca.fit_transform(X_train_s)
        X_val_p = pca.transform(X_val_s)
        transfer_p = {family: (pca.transform(X), y) for family, (X, y) in transfer_s.items()}

        fitted = {
            "mcd": MCD().fit(X_train_p, y_train),
            "kcd": KCD(args.k_neighbors).fit(X_train_p, y_train),
        }

        for method, detector in fitted.items():
            train_scores = detector.score(X_train_p)
            threshold, train_bacc = best_threshold(train_scores, y_train)
            accum[method]["train_balanced_acc"].append(train_bacc)

            scores = detector.score(X_val_p)
            vals = metric_dict(scores, y_val, threshold)
            for k, v in vals.items():
                accum[method]["datasets"]["HarmBench"][k].append(v)

            for family, (X_p, y) in transfer_p.items():
                scores = detector.score(X_p)
                vals = metric_dict(scores, y, threshold)
                for k, v in vals.items():
                    accum[method]["datasets"][family][k].append(v)

        if seed % 10 == 0:
            print(f"  seed {seed} done", flush=True)

    final = {
        "method": "contrastive_geometry",
        "paper_layers": PAPER_LAYERS,
        "n_components": args.n_components,
        "k_neighbors": args.k_neighbors,
        "n_seeds": args.n_seeds,
        "methods": {
            method: {
                "train_balanced_acc": summarize(vals["train_balanced_acc"]),
                "datasets": {
                    family: {metric: summarize(v) for metric, v in metrics.items()}
                    for family, metrics in vals["datasets"].items()
                },
            }
            for method, vals in accum.items()
        },
    }

    with open(output_dir / "contrastive_geometry_results.json", "w") as f:
        json.dump(final, f, indent=2)
    print_table(final)
    print(f"\nSaved to {output_dir / 'contrastive_geometry_results.json'}", flush=True)


if __name__ == "__main__":
    main()
