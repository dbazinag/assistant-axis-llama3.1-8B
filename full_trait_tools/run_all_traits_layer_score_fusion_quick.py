#!/usr/bin/env python3
"""
Quick layer-score fusion sweep for all-traits projections.

This tests whether layer 28 helps when used as a score-level residual instead of
concatenating all layer-16 and layer-28 projection features.

Protocol:
  - train/tune only on HarmBench human_jailbreak strict pool splits
  - evaluate transfer families only after fitting
  - keep models linear and small for fast preliminary feedback
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LAYERS = (16, 28)
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
ALPHAS = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]


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


def load_layer_trait_matrices(vectors_dir: Path) -> tuple[dict[int, np.ndarray], list[str]]:
    names = []
    by_layer = {layer: [] for layer in LAYERS}
    for pt_file in sorted(vectors_dir.glob("*.pt")):
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            vec = data["vector"].float()
            if max(LAYERS) >= vec.shape[0]:
                continue
            layer_vecs = {}
            for layer in LAYERS:
                v = vec[layer].numpy().astype(np.float32)
                norm = np.linalg.norm(v)
                if norm <= 1e-8:
                    layer_vecs = None
                    break
                layer_vecs[layer] = v / norm
            if layer_vecs is None:
                continue
            names.append(pt_file.stem)
            for layer in LAYERS:
                by_layer[layer].append(layer_vecs[layer])
        except Exception:
            continue
    if not names:
        raise RuntimeError(f"No usable vector files found in {vectors_dir}")
    return {layer: np.stack(by_layer[layer]).astype(np.float32) for layer in LAYERS}, names


def build_projection_matrices(rows, activations, matrices):
    xs = {layer: [] for layer in LAYERS}
    ys, valid = [], []
    for row in rows:
        pair_id = row.get("pair_id")
        label = row.get("jailbroken")
        item = activations.get(pair_id)
        if pair_id is None or label is None or item is None:
            continue
        projected = {}
        ok = True
        for layer in LAYERS:
            key = str(layer)
            if key not in item:
                ok = False
                break
            act = item[key].float().numpy().astype(np.float32)
            projected[layer] = act @ matrices[layer].T
        if not ok:
            continue
        for layer in LAYERS:
            xs[layer].append(projected[layer].astype(np.float32))
        ys.append(1 if label else 0)
        valid.append(row)
    if not ys:
        return {layer: np.empty((0, 0), dtype=np.float32) for layer in LAYERS}, np.empty((0,), dtype=np.int64), []
    return {layer: np.stack(xs[layer]) for layer in LAYERS}, np.asarray(ys, dtype=np.int64), valid


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


def build_model(C: float):
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=4000)),
    ])


def score_model(model, x):
    return model.predict_proba(x)[:, 1]


def zfit(score):
    return float(np.mean(score)), float(np.std(score) + 1e-8)


def zapply(score, mean, std):
    return (score - mean) / std


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
    parser.add_argument("--output_dir", default="full_trait_output/all_traits_layer_score_fusion_quick")
    parser.add_argument("--n_seeds", type=int, default=12)
    parser.add_argument("--train_frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val_frac", type=float, default=VAL_FRAC)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading vectors ===", flush=True)
    matrices, trait_names = load_layer_trait_matrices(Path(args.vectors_dir))
    print(f"  {len(trait_names)} vectors per layer", flush=True)

    print("\n=== Loading data ===", flush=True)
    human_rows = [r for r in load_jsonl(Path(args.human_classified_path)) if r.get("attack_type") == "human_jailbreak"]
    human_acts = load_activations(Path(args.human_activations_path))
    x_h, y_h, human_valid = build_projection_matrices(human_rows, human_acts, matrices)
    print(f"  HarmBench: {x_h[16].shape}, jb={y_h.mean():.3f}", flush=True)

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
            x, y, _ = build_projection_matrices(rows, acts, matrices)
            transfer[name] = (x, y)
            print(f"  {name}: {x[16].shape}, jb={y.mean():.3f}", flush=True)

    families = list(transfer)
    methods = ["layer16", "layer28", "avg_16_28", "val_alpha_fusion", "meta_logreg_scores"]
    raw = {m: {"val_auc": [], "human_test": [], **{f: [] for f in families}} for m in methods}
    chosen_alphas = []

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

        models = {}
        stats = {}
        val_scores = {}
        full_val_scores = {}
        for layer, C in [(16, 10.0), (28, 3.0)]:
            mv = build_model(C)
            mv.fit(x_h[layer][tr_idx], y_h[tr_idx])
            val_scores[layer] = score_model(mv, x_h[layer][val_idx])

            m = build_model(C)
            m.fit(x_h[layer][train_idx], y_h[train_idx])
            models[layer] = m
            train_score = score_model(m, x_h[layer][train_idx])
            stats[layer] = zfit(train_score)
            full_val_scores[layer] = zapply(score_model(m, x_h[layer][val_idx]), *stats[layer])

        method_scores_val = {
            "layer16": full_val_scores[16],
            "layer28": full_val_scores[28],
            "avg_16_28": 0.5 * (full_val_scores[16] + full_val_scores[28]),
        }
        best_alpha = max(
            ALPHAS,
            key=lambda a: safe_auc(y_h[val_idx], full_val_scores[16] + a * full_val_scores[28]),
        )
        chosen_alphas.append(best_alpha)
        method_scores_val["val_alpha_fusion"] = full_val_scores[16] + best_alpha * full_val_scores[28]

        meta = Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000)),
        ])
        meta_train = np.stack([
            zapply(score_model(models[16], x_h[16][train_idx]), *stats[16]),
            zapply(score_model(models[28], x_h[28][train_idx]), *stats[28]),
        ], axis=1)
        meta.fit(meta_train, y_h[train_idx])
        method_scores_val["meta_logreg_scores"] = meta.predict_proba(
            np.stack([full_val_scores[16], full_val_scores[28]], axis=1)
        )[:, 1]

        thresholds = {m: best_threshold(y_h[val_idx], s) for m, s in method_scores_val.items()}
        for method, val_score in method_scores_val.items():
            raw[method]["val_auc"].append(safe_auc(y_h[val_idx], val_score))

        def method_scores_for(x_by_layer):
            s16 = zapply(score_model(models[16], x_by_layer[16]), *stats[16])
            s28 = zapply(score_model(models[28], x_by_layer[28]), *stats[28])
            return {
                "layer16": s16,
                "layer28": s28,
                "avg_16_28": 0.5 * (s16 + s28),
                "val_alpha_fusion": s16 + best_alpha * s28,
                "meta_logreg_scores": meta.predict_proba(np.stack([s16, s28], axis=1))[:, 1],
            }

        human_scores = method_scores_for({layer: x_h[layer][test_idx] for layer in LAYERS})
        for method, score in human_scores.items():
            threshold, sign = thresholds[method]
            raw[method]["human_test"].append(eval_set(y_h[test_idx], score, threshold, sign))
        for fam, (x_t, y_t) in transfer.items():
            fam_scores = method_scores_for(x_t)
            for method, score in fam_scores.items():
                threshold, sign = thresholds[method]
                raw[method][fam].append(eval_set(y_t, score, threshold, sign))

        print(f"  Seed {seed}/{args.n_seeds} alpha={best_alpha} ({time.time()-t0:.1f}s)", flush=True)

    summary = {}
    for method in methods:
        r = raw[method]
        summary[method] = {
            "val_auc": summarize(r["val_auc"]),
            "human_test": {m: summarize([x[m] for x in r["human_test"]]) for m in ("auc", "ap", "balanced_acc")},
        }
        for fam in families:
            summary[method][fam] = {m: summarize([x[m] for x in r[fam]]) for m in ("auc", "ap", "balanced_acc")}

    def avg_auc(method):
        return float(np.nanmean([summary[method][f]["auc"]["mean"] for f in families]))

    ranked = sorted(methods, key=avg_auc, reverse=True)
    print("\n" + "=" * 110)
    print(f"  LAYER SCORE FUSION QUICK | {args.n_seeds} seeds | PRIMARY: AUC")
    print("=" * 110)
    header = f"  {'Method':22s} {'ValAUC':>7} {'Human':>7}" + "".join(f" {f:>8}" for f in families) + "  AvgXfer  AvgBAcc"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for method in ranked:
        s = summary[method]
        aucs = [s[f]["auc"]["mean"] for f in families]
        baccs = [s[f]["balanced_acc"]["mean"] for f in families]
        print(
            f"  {method:22s} {s['val_auc']['mean']:7.4f} {s['human_test']['auc']['mean']:7.4f}"
            + "".join(f" {v:8.4f}" for v in aucs)
            + f"  {np.nanmean(aucs):7.4f}  {np.nanmean(baccs):7.4f}"
        )
    print("=" * 110)
    print(f"  Chosen alpha counts: {dict(sorted((a, chosen_alphas.count(a)) for a in set(chosen_alphas)))}")

    out = {
        "method": "all_traits_layer_score_fusion_quick",
        "layers": list(LAYERS),
        "n_seeds": args.n_seeds,
        "n_traits": len(trait_names),
        "families": families,
        "alphas": ALPHAS,
        "chosen_alphas": chosen_alphas,
        "ranked_by_transfer_auc": ranked,
        "summary": summary,
    }
    out_path = out_dir / "results.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
