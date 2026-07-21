#!/usr/bin/env python3
"""
Small transfer sweep using concatenated trait/persona projections from layers 16 and 28.

Feature vector:
  [activation_layer16 @ vectors_layer16.T, activation_layer28 @ vectors_layer28.T]

The script preserves the transfer constraint:
  - train/tune only on HarmBench human_jailbreak strict pool splits
  - evaluate GCG, PAIR, PAP, GPTFuzz, PEZ only after fitting

This is intentionally a small/fast sweep for quick feedback.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


LAYERS = (16, 28)
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable[[int], object]


class AverageScoreEnsemble:
    def __init__(self, builders: list[Callable[[int], object]], seed: int):
        self.builders = builders
        self.seed = seed
        self.models = []
        self.means = []
        self.stds = []

    @staticmethod
    def _score(model, x):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x)[:, 1]
        return model.decision_function(x)

    def fit(self, x, y):
        self.models = []
        self.means = []
        self.stds = []
        for offset, builder in enumerate(self.builders):
            model = builder(self.seed + offset)
            model.fit(x, y)
            score = self._score(model, x)
            self.models.append(model)
            self.means.append(float(np.mean(score)))
            self.stds.append(float(np.std(score) + 1e-8))
        return self

    def decision_function(self, x):
        scores = [
            (self._score(model, x) - mean) / std
            for model, mean, std in zip(self.models, self.means, self.stds)
        ]
        return np.mean(np.stack(scores, axis=0), axis=0)


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


def load_layer_trait_matrices(vectors_dir: Path, layers: tuple[int, ...]) -> tuple[dict[int, np.ndarray], list[str]]:
    names = []
    by_layer = {layer: [] for layer in layers}
    for pt_file in sorted(vectors_dir.glob("*.pt")):
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            vec = data["vector"].float()
            if max(layers) >= vec.shape[0]:
                continue
            layer_vecs = {}
            ok = True
            for layer in layers:
                v = vec[layer].numpy().astype(np.float32)
                norm = np.linalg.norm(v)
                if norm <= 1e-8:
                    ok = False
                    break
                layer_vecs[layer] = v / norm
            if ok:
                names.append(pt_file.stem)
                for layer in layers:
                    by_layer[layer].append(layer_vecs[layer])
        except Exception:
            continue
    if not names:
        raise RuntimeError(f"No usable vector files found in {vectors_dir}")
    matrices = {layer: np.stack(by_layer[layer]).astype(np.float32) for layer in layers}
    return matrices, names


def build_projection_matrix(rows, activations, matrices: dict[int, np.ndarray], layers: tuple[int, ...]):
    xs, ys, valid = [], [], []
    for row in rows:
        pair_id = row.get("pair_id")
        label = row.get("jailbroken")
        item = activations.get(pair_id)
        if pair_id is None or label is None or item is None:
            continue
        parts = []
        ok = True
        for layer in layers:
            key = str(layer)
            if key not in item:
                ok = False
                break
            act = item[key].float().numpy().astype(np.float32)
            parts.append(act @ matrices[layer].T)
        if not ok:
            continue
        xs.append(np.concatenate(parts).astype(np.float32))
        ys.append(1 if label else 0)
        valid.append(row)
    if not xs:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64), []
    return np.stack(xs), np.asarray(ys, dtype=np.int64), valid


def get_pool_split(rows, train_frac, seed):
    rng = random.Random(seed)
    behaviors = sorted({r["behavior_id"] for r in rows})
    templates = sorted({r["jailbreak_idx"] for r in rows})
    rng.shuffle(behaviors)
    rng.shuffle(templates)
    n_beh = max(1, int(len(behaviors) * train_frac))
    n_tpl = max(1, int(len(templates) * train_frac))
    return set(behaviors[:n_beh]), set(templates[:n_tpl]), set(behaviors[n_beh:]), set(templates[n_tpl:])


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


def safe_auc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    auc = float(roc_auc_score(y_true, score))
    return max(auc, 1.0 - auc)


def safe_ap(y_true, score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return max(float(average_precision_score(y_true, score)), float(average_precision_score(y_true, -score)))


def best_threshold(y_true, score):
    candidates = np.unique(score)
    if len(candidates) > 400:
        candidates = np.quantile(candidates, np.linspace(0, 1, 400))
    best_bacc, best_thr, best_sign = -1.0, 0.0, 1
    for sign in (1, -1):
        for thr in candidates:
            pred = ((sign * score) >= (sign * thr)).astype(int)
            bacc = float(balanced_accuracy_score(y_true, pred))
            if bacc > best_bacc:
                best_bacc, best_thr, best_sign = bacc, float(thr), sign
    return best_thr, best_sign


def get_score(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def eval_set(y_true, score, threshold, sign):
    pred = ((sign * score) >= (sign * threshold)).astype(int)
    return {
        "auc": safe_auc(y_true, score),
        "ap": safe_ap(y_true, score),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }


def summarize(vals):
    arr = np.asarray(vals, dtype=float)
    return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)), "all": [float(x) for x in arr]}


def build_specs():
    def logreg(C, class_weight):
        return lambda seed: Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=4000, class_weight=class_weight, random_state=seed)),
        ])

    def linsvm(C, class_weight):
        return lambda seed: Pipeline([
            ("sc", StandardScaler()),
            ("clf", LinearSVC(C=C, class_weight=class_weight, max_iter=10000, random_state=seed)),
        ])

    base = [
        ModelSpec("logreg_l2_C1_raw", logreg(1.0, None)),
        ModelSpec("logreg_l2_C3_raw", logreg(3.0, None)),
        ModelSpec("logreg_l2_C10_raw", logreg(10.0, None)),
        ModelSpec("linsvm_C0.1_bal", linsvm(0.1, "balanced")),
        ModelSpec("linsvm_C0.3_bal", linsvm(0.3, "balanced")),
        ModelSpec("linsvm_C1_bal", linsvm(1.0, "balanced")),
    ]
    ensemble_builders = [
        logreg(10.0, None),
        logreg(3.0, None),
        linsvm(0.3, "balanced"),
        linsvm(1.0, "balanced"),
    ]
    base.append(ModelSpec("ensemble_linear_top", lambda seed: AverageScoreEnsemble(ensemble_builders, seed)))
    return base


def main():
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
    parser.add_argument("--vectors_dir", default="full_trait_output/traits40_vectors/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total")
    parser.add_argument("--output_dir", default="full_trait_output/all_traits_layers16_28_sweep_small")
    parser.add_argument("--n_seeds", type=int, default=15)
    parser.add_argument("--train_frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val_frac", type=float, default=VAL_FRAC)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading vectors ===", flush=True)
    matrices, trait_names = load_layer_trait_matrices(Path(args.vectors_dir), LAYERS)
    print(f"  {len(trait_names)} vectors, feature_dim={sum(m.shape[0] for m in matrices.values())}", flush=True)

    print("\n=== Loading data ===", flush=True)
    human_rows = [r for r in load_jsonl(Path(args.human_classified_path)) if r.get("attack_type") == "human_jailbreak"]
    human_acts = load_activations(Path(args.human_activations_path))
    x_h, y_h, human_valid = build_projection_matrix(human_rows, human_acts, matrices, LAYERS)
    print(f"  HarmBench: {x_h.shape}, jb={y_h.mean():.3f}", flush=True)

    transfer_inputs = [
        ("GCG", args.gcg_classified_path, args.gcg_activations_path),
        ("PAIR", args.pair_classified_path, args.pair_activations_path),
        ("PAP", args.pap_classified_path, args.pap_activations_path),
        ("GPTFuzz", args.gptfuzz_classified_path, args.gptfuzz_activations_path),
        ("PEZ", args.pez_classified_path, args.pez_activations_path),
    ]
    transfer = {}
    for name, rows_path, acts_path in transfer_inputs:
        if Path(rows_path).exists() and Path(acts_path).exists():
            rows = load_jsonl(Path(rows_path))
            acts = load_activations(Path(acts_path))
            x, y, _ = build_projection_matrix(rows, acts, matrices, LAYERS)
            transfer[name] = (x, y)
            print(f"  {name}: {x.shape}, jb={y.mean():.3f}", flush=True)

    specs = build_specs()
    families = list(transfer)
    raw = {s.name: {"val_auc": [], "human_test": [], **{f: [] for f in families}} for s in specs}

    t0 = time.time()
    for seed in range(args.n_seeds):
        train_beh, train_tpl, test_beh, test_tpl = get_pool_split(human_valid, args.train_frac, seed)
        train_idx, test_idx = split_by_pool(human_valid, train_beh, train_tpl, test_beh, test_tpl)
        if not train_idx or not test_idx:
            continue
        tr_idx, val_idx = train_test_split(
            np.asarray(train_idx),
            test_size=args.val_frac,
            random_state=seed,
            stratify=y_h[train_idx],
        )
        for spec in specs:
            m_val = spec.builder(seed)
            m_val.fit(x_h[tr_idx], y_h[tr_idx])
            val_score = get_score(m_val, x_h[val_idx])
            raw[spec.name]["val_auc"].append(safe_auc(y_h[val_idx], val_score))

            m = spec.builder(seed)
            m.fit(x_h[train_idx], y_h[train_idx])
            val_score_full = get_score(m, x_h[val_idx])
            threshold, sign = best_threshold(y_h[val_idx], val_score_full)

            human_score = get_score(m, x_h[test_idx])
            raw[spec.name]["human_test"].append(eval_set(y_h[test_idx], human_score, threshold, sign))
            for fam, (x_t, y_t) in transfer.items():
                score = get_score(m, x_t)
                raw[spec.name][fam].append(eval_set(y_t, score, threshold, sign))
        print(f"  Seed {seed}/{args.n_seeds} ({time.time()-t0:.1f}s)", flush=True)

    summary = {}
    for spec in specs:
        r = raw[spec.name]
        summary[spec.name] = {
            "val_auc": summarize(r["val_auc"]),
            "human_test": {m: summarize([x[m] for x in r["human_test"]]) for m in ("auc", "ap", "balanced_acc")},
        }
        for fam in families:
            summary[spec.name][fam] = {
                m: summarize([x[m] for x in r[fam]]) for m in ("auc", "ap", "balanced_acc")
            }

    def avg_auc(name):
        return float(np.nanmean([summary[name][f]["auc"]["mean"] for f in families]))

    ranked = sorted([s.name for s in specs], key=avg_auc, reverse=True)
    print("\n" + "=" * 110)
    print(f"  LAYERS 16+28 ALL-TRAITS SMALL SWEEP | {args.n_seeds} seeds | PRIMARY: AUC")
    print("=" * 110)
    header = f"  {'Model':24s} {'ValAUC':>7} {'Human':>7}" + "".join(f" {f:>8}" for f in families) + "  AvgXfer"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in ranked:
        s = summary[name]
        vals = [s[f]["auc"]["mean"] for f in families]
        row = (
            f"  {name:24s} {s['val_auc']['mean']:7.4f} {s['human_test']['auc']['mean']:7.4f}"
            + "".join(f" {v:8.4f}" for v in vals)
            + f"  {np.nanmean(vals):7.4f}"
        )
        print(row)
    print("=" * 110)

    out = {
        "method": "all_traits_layers16_28_sweep_small",
        "layers": list(LAYERS),
        "n_seeds": args.n_seeds,
        "n_traits": len(trait_names),
        "feature_dim": int(sum(m.shape[0] for m in matrices.values())),
        "families": families,
        "ranked_by_transfer_auc": ranked,
        "summary": summary,
    }
    out_path = out_dir / "results.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
