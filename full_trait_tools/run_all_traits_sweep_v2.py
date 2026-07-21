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
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
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


class AverageScoreEnsemble:
    """Average standardized decision scores from a small fixed model set."""

    def __init__(self, builders: list[Callable[[int], object]], seed: int):
        self.builders = builders
        self.seed = seed
        self.models = []
        self.means = []
        self.stds = []

    @staticmethod
    def _score(model, x: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x)[:, 1]
        return model.decision_function(x)

    def fit(self, x: np.ndarray, y: np.ndarray):
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

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        scores = []
        for model, mean, std in zip(self.models, self.means, self.stds):
            scores.append((self._score(model, x) - mean) / std)
        return np.mean(np.stack(scores, axis=0), axis=0)


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


def load_trait_matrix(
    layer: int,
    vectors_dir: Path | None = None,
    cache_dir: Path | None = None,
    row_index: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    # The trait .pt files store `vector` as a [n_saved_layers x hidden] tensor; `layer`
    # selects the row. When the saved tensor isn't indexed by absolute layer number
    # (e.g. Gemma layer30_only vectors store a single row 0 == model layer 30), pass
    # row_index to pick the row directly while `layer` still keys the activations.
    idx = row_index if row_index is not None else layer
    if vectors_dir is not None:
        # Build from .pt trait vector files; cache in output dir for speed.
        cache_dir = cache_dir or Path("full_trait_output")
        mat_path   = cache_dir / f"trait_matrix_layer{layer}.npy"
        names_path = cache_dir / f"trait_names_layer{layer}.json"
        if mat_path.exists() and names_path.exists():
            print(f"  Loading cached OLMo trait matrix from {mat_path}", flush=True)
            matrix = np.load(mat_path).astype(np.float32)
            names  = json.loads(names_path.read_text())
        else:
            print(f"  Building trait matrix from {vectors_dir} ...", flush=True)
            vecs, names = [], []
            for pt_file in sorted(Path(vectors_dir).glob("*.pt")):
                try:
                    data = torch.load(pt_file, map_location="cpu", weights_only=False)
                    v = data["vector"][idx].float().numpy()
                    norm = np.linalg.norm(v)
                    if norm > 1e-8:
                        vecs.append(v / norm)
                        names.append(pt_file.stem)
                except Exception as e:
                    print(f"  Warning: skipping {pt_file.name}: {e}", flush=True)
            if not vecs:
                raise ValueError(f"No valid trait vectors found in {vectors_dir}")
            matrix = np.stack(vecs).astype(np.float32)
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(mat_path, matrix)
            names_path.write_text(json.dumps(names))
            print(f"  Cached {len(names)} trait vectors → {mat_path}", flush=True)
        norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        return matrix, names

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

    def logreg_l2(C: float, class_weight):
        return lambda seed: Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                C=C, penalty="l2", solver="lbfgs",
                max_iter=4000, class_weight=class_weight, random_state=seed,
            )),
        ])

    def logreg_l1(C: float, class_weight):
        return lambda seed: Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                C=C, penalty="l1", solver="liblinear",
                max_iter=4000, class_weight=class_weight, random_state=seed,
            )),
        ])

    def linsvm(C: float, class_weight):
        return lambda seed: Pipeline([
            ("sc", StandardScaler()),
            ("clf", LinearSVC(
                C=C, class_weight=class_weight, max_iter=10000, random_state=seed,
            )),
        ])

    # Logistic regression — L2, wide C range, balanced and unbalanced
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        for cw in ["balanced", None]:
            tag = f"logreg_l2_C{C}_{'bal' if cw else 'raw'}"
            specs.append(ModelSpec(tag, logreg_l2(C, cw)))

    # Logistic regression — L1
    for C in [0.01, 0.03, 0.1, 0.3, 1.0]:
        for cw in ["balanced", None]:
            tag = f"logreg_l1_C{C}_{'bal' if cw else 'raw'}"
            specs.append(ModelSpec(tag, logreg_l1(C, cw)))

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
            specs.append(ModelSpec(tag, linsvm(C, cw)))

    linear_auc_builders = [
        logreg_l2(10.0, None),
        logreg_l2(3.0, None),
        linsvm(0.3, "balanced"),
        linsvm(1.0, "balanced"),
        logreg_l1(1.0, None),
    ]
    linear_balanced_builders = [
        logreg_l2(3.0, None),
        logreg_l2(3.0, "balanced"),
        linsvm(0.3, "balanced"),
        linsvm(0.3, None),
        logreg_l1(1.0, None),
    ]
    specs.append(ModelSpec(
        "ensemble_linear_auc_top",
        lambda seed: AverageScoreEnsemble(linear_auc_builders, seed),
    ))
    specs.append(ModelSpec(
        "ensemble_linear_bacc_top",
        lambda seed: AverageScoreEnsemble(linear_balanced_builders, seed),
    ))

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

    for max_iter, lr, max_leaf in [(200, 0.04, 15), (300, 0.03, 31), (200, 0.06, 31)]:
        tag = f"histgb_{max_iter}_lr{lr}_leaf{max_leaf}"
        _it, _lr, _leaf = max_iter, lr, max_leaf
        specs.append(ModelSpec(tag, lambda seed, it=_it, lr=_lr, leaf=_leaf: HistGradientBoostingClassifier(
            max_iter=it, learning_rate=lr, max_leaf_nodes=leaf,
            l2_regularization=0.05, random_state=seed,
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
    parser.add_argument("--wjb_classified_path",
                        default="full_trait_output/wildjailbreak_activations/classified_responses.jsonl")
    parser.add_argument("--wjb_activations_path",
                        default="full_trait_output/wildjailbreak_activations/activations.pt")
    parser.add_argument("--augment_classified_path", default=None,
                        help="If set, fold these labelled rows into the TRAINING pool each seed "
                             "(held-out eval families unchanged). Used for WJB training-augmentation.")
    parser.add_argument("--augment_activations_path", default=None)
    parser.add_argument("--output_dir",        default="full_trait_output/all_traits_sweep_v2")
    parser.add_argument("--trait_vectors_dir", default=None,
                        help="If set, build trait matrix from .pt files in this dir "
                             "(for OLMo-3 or other non-default models). "
                             "Cached in --output_dir between runs.")
    parser.add_argument("--layer",        type=int,   default=LAYER)
    parser.add_argument("--trait_row_index", type=int, default=None,
                        help="Row to select from each trait .pt 'vector' tensor. "
                             "Defaults to --layer. Set when trait vectors aren't indexed "
                             "by absolute layer (e.g. Gemma layer30_only: --layer 30 "
                             "--trait_row_index 0).")
    parser.add_argument("--n_seeds",      type=int,   default=N_SEEDS)
    parser.add_argument("--train_frac",   type=float, default=TRAIN_FRAC)
    parser.add_argument("--val_frac",     type=float, default=VAL_FRAC)
    parser.add_argument("--include_slow", action="store_true",
                        help="Include RBF SVM (much slower).")
    parser.add_argument("--only_models",  default="",
                        help="Comma-separated subset of model names to run.")
    parser.add_argument("--feature_type", choices=["trait", "raw", "pca", "random"], default="trait",
                        help="Feature space: trait-projected, raw 4096-dim, PCA-reduced, or random projection.")
    parser.add_argument("--pca_n_components", type=int, default=256,
                        help="PCA output dimensionality (only used when --feature_type pca).")
    parser.add_argument("--random_n_components", type=int, default=228,
                        help="Random projection dimensionality (only used when --feature_type random).")
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
        ("WJB",     args.wjb_classified_path,      args.wjb_activations_path),
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

    aug_raw_x, aug_raw_y = None, None
    if args.augment_classified_path and args.augment_activations_path:
        # Comma-separated lists let us fold in multiple sources (e.g. WJB + GPTFuzz);
        # each is loaded independently so colliding pair_ids across files don't clash.
        cls_paths = [p.strip() for p in args.augment_classified_path.split(",") if p.strip()]
        act_paths = [p.strip() for p in args.augment_activations_path.split(",") if p.strip()]
        aug_xs, aug_ys = [], []
        for cp, ap in zip(cls_paths, act_paths):
            if not (Path(cp).exists() and Path(ap).exists()):
                continue
            ax, ay, _ = build_activation_matrix(load_jsonl(Path(cp)), load_activations(Path(ap)), args.layer)
            if len(ax) > 0:
                print(f"  AUGMENT(train) {Path(cp).parent.name}: {len(ax)} rows, jb={ay.mean():.3f}", flush=True)
                aug_xs.append(ax); aug_ys.append(ay)
        if aug_xs:
            aug_raw_x = np.vstack(aug_xs)
            aug_raw_y = np.concatenate(aug_ys)
            print(f"  AUGMENT(train) TOTAL: {len(aug_raw_x)} rows, jb={aug_raw_y.mean():.3f}", flush=True)

    if args.feature_type == "trait":
        vectors_dir = Path(args.trait_vectors_dir) if args.trait_vectors_dir else None
        trait_matrix, trait_names = load_trait_matrix(
            args.layer,
            vectors_dir=vectors_dir,
            cache_dir=out_dir if vectors_dir is not None else None,
            row_index=args.trait_row_index,
        )
        print(f"  Trait matrix: {trait_matrix.shape}", flush=True)
    else:
        trait_matrix, trait_names = None, []
        print(f"  Feature type: {args.feature_type} (no trait matrix needed)", flush=True)

    # ── Build features ─────────────────────────────────────────────────────────
    print("\n=== Building features ===", flush=True)
    x_raw_h, y_h, human_valid = build_activation_matrix(human_rows, human_acts, args.layer)

    if args.feature_type == "trait":
        x_h = project_all_traits(x_raw_h, trait_matrix)
        transfer: dict[str, tuple[np.ndarray, np.ndarray]] = {
            name: (project_all_traits(xr, trait_matrix), y)
            for name, (xr, y) in transfer_data.items()
        }
    elif args.feature_type == "raw":
        x_h = x_raw_h
        transfer = {name: (xr, y) for name, (xr, y) in transfer_data.items()}
    elif args.feature_type == "pca":
        n_components = min(args.pca_n_components, x_raw_h.shape[0] - 1, x_raw_h.shape[1])
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        x_h = pca.fit_transform(x_raw_h).astype(np.float32)
        transfer = {
            name: (pca.transform(xr).astype(np.float32), y)
            for name, (xr, y) in transfer_data.items()
        }
        print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}", flush=True)
    else:  # random
        n_components = min(args.random_n_components, x_raw_h.shape[1])
        rng = np.random.default_rng(RANDOM_SEED)
        R = rng.standard_normal((x_raw_h.shape[1], n_components)).astype(np.float32)
        R, _ = np.linalg.qr(R)  # orthonormal columns
        R = R[:, :n_components].astype(np.float32)
        x_h = (x_raw_h @ R).astype(np.float32)
        transfer = {
            name: ((xr @ R).astype(np.float32), y)
            for name, (xr, y) in transfer_data.items()
        }
        print(f"  Random projection: {x_raw_h.shape[1]} → {n_components} dims (fixed seed, orthonormal)", flush=True)

    x_aug = y_aug = None
    if aug_raw_x is not None and len(aug_raw_x) > 0:
        if args.feature_type == "trait":
            x_aug = project_all_traits(aug_raw_x, trait_matrix)
        elif args.feature_type == "raw":
            x_aug = aug_raw_x
        elif args.feature_type == "pca":
            # fold augment into training in the SAME (HB-fit) PCA basis used for transfer
            x_aug = pca.transform(aug_raw_x).astype(np.float32)
        elif args.feature_type == "random":
            x_aug = (aug_raw_x @ R).astype(np.float32)
        else:
            raise NotImplementedError(f"--augment_* not supported for feature_type {args.feature_type}")
        y_aug = aug_raw_y
        print(f"  Train pool augmented with {len(x_aug)} rows "
              f"({int(y_aug.sum())} pos / {int((1 - y_aug).sum())} neg)", flush=True)

    print(f"  HarmBench: {x_h.shape}, jb={y_h.mean():.3f}", flush=True)

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

        # Optional training-pool augmentation (val/test/transfer eval stay pure HarmBench).
        if x_aug is not None:
            x_tr_fit   = np.vstack([x_h[tr_idx], x_aug])
            y_tr_fit   = np.concatenate([y_h[tr_idx], y_aug])
            x_full_fit = np.vstack([x_h[train_idx], x_aug])
            y_full_fit = np.concatenate([y_h[train_idx], y_aug])
        else:
            x_tr_fit,   y_tr_fit   = x_h[tr_idx],    y_h[tr_idx]
            x_full_fit, y_full_fit = x_h[train_idx], y_h[train_idx]

        # --- Step 1: select best model by inner-val AUC ---
        val_aucs: dict[str, float] = {}
        for spec in specs:
            m = spec.builder(seed)
            m.fit(x_tr_fit, y_tr_fit)
            val_score = get_score(m, x_h[val_idx])
            val_aucs[spec.name] = safe_auc(y_h[val_idx], val_score)

        best_name = max(val_aucs, key=val_aucs.get)
        selected_counts[best_name] = selected_counts.get(best_name, 0) + 1

        # --- Step 2: refit every model on full train pool, calibrate threshold
        #             on val_idx predictions from the REFIT model (fixes staleness) ---
        for spec in specs:
            raw[spec.name]["val_auc"].append(val_aucs[spec.name])

            m_full = spec.builder(seed)
            m_full.fit(x_full_fit, y_full_fit)

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
    print(f"  ALL-TRAITS SWEEP v2 | layer {args.layer} | {args.n_seeds} seeds | feature={args.feature_type} | PRIMARY METRIC: AUC")
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
        "feature_type": args.feature_type,
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
