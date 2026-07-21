#!/usr/bin/env python3
"""
run_baseline_rtv_mahalanobis.py

RTV-style Mahalanobis outlier detector for transfer jailbreak detection.

Paper-faithful core:
  - refusal direction per layer from harmful - harmless calibration means
  - layers 18, 25, 32 and last five prompt tokens
  - 15D fingerprint of cosine(hidden_state, refusal_direction)
  - Ledoit-Wolf covariance for harmless/harmful calibration clusters
  - score M(x) = min(distance_to_harmless, distance_to_harmful)
  - threshold = 95th percentile of legitimate calibration scores

Adaptation for this project:
  - harmless calibration prompts come from Alpaca, as in the RTV paper
  - harmful calibration prompts are HarmBench direct requests
  - adversarial evaluation uses held-out HarmBench human jailbreaks and
    transfer families GCG, PAIR, PAP, GPTFuzz, PEZ
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

PAPER_LAYERS = [18, 25, 32]
N_LAST_TOKENS = 5
TRAIN_FRAC = 0.7
N_SEEDS = 50
CALIBRATION_N = 100
RANDOM_SEED = 42

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


def load_activations(path: Path) -> dict:
    print(f"  Loading {path.name}...", flush=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def bool_label(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return int(value.lower() == "true")
    return int(bool(value))


def load_dataset_matrix(root: Path, dataset: str):
    rows = load_jsonl(root / f"{dataset}_rows.jsonl")
    activations = load_activations(root / f"{dataset}_activations.pt")
    X, valid_rows = [], []
    for row in rows:
        pid = row["pair_id"]
        if pid not in activations:
            continue
        try:
            arr = np.stack([
                activations[pid][str(layer)].float().numpy()
                for layer in PAPER_LAYERS
            ])  # [3, 5, hidden_dim]
        except KeyError:
            continue
        if arr.shape[0] != len(PAPER_LAYERS) or arr.shape[1] != N_LAST_TOKENS:
            continue
        X.append(arr)
        valid_rows.append(row)
    if not X:
        raise ValueError(f"No valid activations for {dataset}")
    return np.stack(X).astype(np.float32), valid_rows


def get_pool_split(rows: List[dict], train_frac: float, seed: int):
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


def human_split_indices(rows: List[dict], train_beh, train_tpl, val_beh, val_tpl):
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


def direct_request_indices(rows: List[dict], behaviors: set):
    return [
        i for i, r in enumerate(rows)
        if r.get("attack_type") == "direct_request" and r["behavior_id"] in behaviors
    ]


def sample_indices(indices: List[int], n: int, rng: random.Random):
    if len(indices) <= n:
        return list(indices)
    return rng.sample(indices, n)


def compute_refusal_dirs(X_harmless: np.ndarray, X_harmful: np.ndarray) -> np.ndarray:
    # Direction uses final prompt token, matching Eq. 6 in the paper.
    mu_harmless = X_harmless[:, :, -1, :].mean(axis=0)  # [3, hidden]
    mu_harmful = X_harmful[:, :, -1, :].mean(axis=0)
    dirs = mu_harmful - mu_harmless
    norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    return (dirs / norms).astype(np.float32)


def fingerprints(X: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    # X: [N, 3, 5, hidden], dirs: [3, hidden] -> [N, 15]
    x_norm = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)
    feats = (x_norm * dirs[None, :, None, :]).sum(axis=-1)
    return feats.reshape(X.shape[0], -1).astype(np.float32)


class MahalanobisCluster:
    def fit(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.cov = LedoitWolf().fit(X)
        return self

    def distance(self, X: np.ndarray) -> np.ndarray:
        diff = X - self.mean
        precision = self.cov.precision_
        d2 = np.einsum("ij,jk,ik->i", diff, precision, diff)
        return np.sqrt(np.maximum(d2, 0.0))


def fit_rtv(X_harmless: np.ndarray, X_harmful: np.ndarray):
    dirs = compute_refusal_dirs(X_harmless, X_harmful)
    F_harmless = fingerprints(X_harmless, dirs)
    F_harmful = fingerprints(X_harmful, dirs)
    harmless_cluster = MahalanobisCluster().fit(F_harmless)
    harmful_cluster = MahalanobisCluster().fit(F_harmful)

    calib_scores = np.minimum(
        harmless_cluster.distance(np.vstack([F_harmless, F_harmful])),
        harmful_cluster.distance(np.vstack([F_harmless, F_harmful])),
    )
    threshold = float(np.quantile(calib_scores, 0.95))
    return dirs, harmless_cluster, harmful_cluster, threshold


def score_rtv(X: np.ndarray, dirs: np.ndarray, harmless_cluster, harmful_cluster) -> np.ndarray:
    F = fingerprints(X, dirs)
    return np.minimum(harmless_cluster.distance(F), harmful_cluster.distance(F))


def metrics(scores: np.ndarray, y: np.ndarray, threshold: float):
    if len(scores) == 0 or len(set(y.tolist())) < 2:
        return {"auc": float("nan"), "ap": float("nan"), "balanced_acc": float("nan")}
    pred = (scores > threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y, scores)),
        "ap": float(average_precision_score(y, scores)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
    }


def summarize(values: Iterable[float]):
    arr = np.array(list(values), dtype=float)
    return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)), "all": arr.tolist()}


def labels_for_rows(rows: List[dict]) -> np.ndarray:
    return np.array([bool_label(r.get("jailbroken", False)) for r in rows], dtype=int)


def print_table(results: dict) -> None:
    print("\n" + "=" * 100)
    print("  RTV-STYLE MAHALANOBIS TRANSFER BASELINE")
    print("  Calibration: Alpaca harmless + HarmBench direct harmful prompts only")
    print("=" * 100)
    print(f"  Legitimate calibration FPR: {results['calibration_fpr']['mean']:.4f} ± {results['calibration_fpr']['std']:.4f}")
    print(f"  Harmful calibration n:      {results['harmful_calibration_n']['mean']:.1f}")
    print(f"  Harmless calibration n:     {results['harmless_calibration_n']['mean']:.1f}\n")
    print(f"  {'Family':12s} {'AUC':>9s} {'AP':>9s} {'BAcc@5FPR':>11s}")
    print("  " + "-" * 43)
    for name, vals in results["datasets"].items():
        print(
            f"  {name:12s} "
            f"{vals['auc']['mean']:9.4f} "
            f"{vals['ap']['mean']:9.4f} "
            f"{vals['balanced_acc']['mean']:11.4f}"
        )
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtv_dir", default="full_trait_output/rtv_activations")
    parser.add_argument("--output_dir", default="full_trait_output/baselines_rtv_mahalanobis")
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS)
    parser.add_argument("--calibration_n", type=int, default=CALIBRATION_N)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--olmo3", action="store_true",
                        help="Use OLMo-3 RTV caches (GCG excluded; not yet collected)")
    parser.add_argument("--gemma", action="store_true",
                        help="Use Gemma-4-31B RTV caches (GCG excluded; not collected)")
    parser.add_argument("--augment_wjb", action="store_true",
                        help="Fold a wildjailbreak subsample into the HARMFUL calibration cluster "
                             "each seed (HB+WJB); eval unchanged. Output dir gets a _hbwjb suffix.")
    args = parser.parse_args()

    if args.olmo3:
        if args.rtv_dir == "full_trait_output/rtv_activations":
            args.rtv_dir = "full_trait_output/rtv_activations_olmo3"
        if args.output_dir == "full_trait_output/baselines_rtv_mahalanobis":
            args.output_dir = "full_trait_output/baselines_rtv_mahalanobis_olmo3"
    if args.gemma:
        if args.rtv_dir == "full_trait_output/rtv_activations":
            args.rtv_dir = "full_trait_output/rtv_activations_gemma"
        if args.output_dir == "full_trait_output/baselines_rtv_mahalanobis":
            args.output_dir = "full_trait_output/baselines_rtv_mahalanobis_gemma"
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

    print("\n=== Loading RTV activation caches ===", flush=True)
    X_harmless_all, harmless_rows = load_dataset_matrix(rtv_dir, "alpaca_harmless")
    X_harmbench_all, harmbench_rows = load_dataset_matrix(rtv_dir, "harmbench")
    human_rows = [r for r in harmbench_rows if r.get("attack_type") == "human_jailbreak"]

    X_wj_all = None
    if args.augment_wjb:
        X_wj_all, wj_rows = load_dataset_matrix(rtv_dir, "wildjailbreak")
        print(f"  WJB harmful-cluster augment: {len(wj_rows)} rows available", flush=True)

    transfer = {}
    for display, dataset in datasets.items():
        if dataset == "harmbench":
            continue
        rows_path = rtv_dir / f"{dataset}_rows.jsonl"
        acts_path = rtv_dir / f"{dataset}_activations.pt"
        if rows_path.exists() and acts_path.exists():
            X, rows = load_dataset_matrix(rtv_dir, dataset)
            transfer[display] = (X, rows, labels_for_rows(rows))
            print(f"  {display}: {len(rows)} rows, positive={labels_for_rows(rows).mean():.3f}")

    y_harmbench_all = labels_for_rows(harmbench_rows)
    index_by_pair_id = {r["pair_id"]: i for i, r in enumerate(harmbench_rows)}
    human_index_rows = [r for r in harmbench_rows if r.get("attack_type") == "human_jailbreak"]

    results = {name: {"auc": [], "ap": [], "balanced_acc": []} for name in ["HarmBench", *transfer.keys()]}
    calibration_fprs = []
    harmful_ns = []
    harmless_ns = []

    for seed in range(args.n_seeds):
        rng = random.Random(seed)
        train_beh, train_tpl, val_beh, val_tpl = get_pool_split(human_index_rows, TRAIN_FRAC, seed)
        _, val_human_local = human_split_indices(human_index_rows, train_beh, train_tpl, val_beh, val_tpl)

        # val_human_local indexes human_index_rows; convert to all-HarmBench indexes.
        val_human_idx = [index_by_pair_id[human_index_rows[i]["pair_id"]] for i in val_human_local]
        harmful_idx = direct_request_indices(harmbench_rows, train_beh)
        harmful_idx = sample_indices(harmful_idx, args.calibration_n, rng)
        harmless_idx = sample_indices(list(range(len(harmless_rows))), args.calibration_n, rng)
        if len(harmful_idx) < 20 or len(harmless_idx) < 20 or not val_human_idx:
            continue

        X_harmful = X_harmbench_all[harmful_idx]
        X_harmless = X_harmless_all[harmless_idx]
        if X_wj_all is not None:  # add a balanced WJB subsample to the harmful cluster
            wj_idx = sample_indices(list(range(len(X_wj_all))), args.calibration_n, rng)
            X_harmful = np.concatenate([X_harmful, X_wj_all[wj_idx]], axis=0)
        dirs, harmless_cluster, harmful_cluster, threshold = fit_rtv(X_harmless, X_harmful)

        cal_scores = score_rtv(np.vstack([X_harmless, X_harmful]), dirs, harmless_cluster, harmful_cluster)
        calibration_fprs.append(float((cal_scores > threshold).mean()))
        harmful_ns.append(len(harmful_idx))
        harmless_ns.append(len(harmless_idx))

        scores_hb = score_rtv(X_harmbench_all[val_human_idx], dirs, harmless_cluster, harmful_cluster)
        y_hb = y_harmbench_all[val_human_idx]
        m = metrics(scores_hb, y_hb, threshold)
        for k, v in m.items():
            results["HarmBench"][k].append(v)

        for name, (X, _rows, y) in transfer.items():
            scores = score_rtv(X, dirs, harmless_cluster, harmful_cluster)
            m = metrics(scores, y, threshold)
            for k, v in m.items():
                results[name][k].append(v)

        if seed % 10 == 0:
            print(f"  seed {seed} done", flush=True)

    final = {
        "method": "rtv_mahalanobis",
        "paper_layers": PAPER_LAYERS,
        "n_last_tokens": N_LAST_TOKENS,
        "calibration_n_requested": args.calibration_n,
        "n_seeds": args.n_seeds,
        "calibration_fpr": summarize(calibration_fprs),
        "harmful_calibration_n": summarize(harmful_ns),
        "harmless_calibration_n": summarize(harmless_ns),
        "datasets": {name: {k: summarize(v) for k, v in vals.items()} for name, vals in results.items()},
    }

    with open(output_dir / "rtv_mahalanobis_results.json", "w") as f:
        json.dump(final, f, indent=2)
    print_table(final)
    print(f"\nSaved to {output_dir / 'rtv_mahalanobis_results.json'}", flush=True)


if __name__ == "__main__":
    main()
