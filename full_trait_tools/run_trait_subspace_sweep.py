#!/usr/bin/env python3
"""
run_trait_subspace_sweep.py

Uses top-k PCA components of the TRAIT VECTOR MATRIX as classifier features.

The trait vectors form a 229×4096 matrix. PC1 of this matrix is the assistant
axis — the dominant direction of the trait subspace in activation space. This
script sweeps k = 1...N, projecting raw activations onto the top-k eigenvectors
and classifying with logistic regression.

This is different from run_trait_pca_sweep.py which does PCA on the
229-dimensional projection space. This does PCA on the trait vectors themselves
to find the k most important directions in activation space.

k=1 → assistant axis (current baseline)
k>1 → richer representation of the trait subspace

Sweeps k = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50]
50 seeds, balanced AUC, strict pool split.

Usage:
  uv run python full_trait_tools/run_trait_subspace_sweep.py
  uv run python full_trait_tools/run_trait_subspace_sweep.py --output_dir full_trait_output/trait_subspace_sweep
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

LAYER       = 16
TRAIN_FRAC  = 0.7
N_SEEDS     = 50
RANDOM_SEED = 42

K_VALUES = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50]


# ── I/O ────────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_activations(path):
    print(f"  Loading {Path(path).name}...", flush=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_trait_matrix(layer):
    cache_mat   = Path(f"full_trait_output/trait_matrix_layer{layer}.npy")
    cache_names = Path(f"full_trait_output/trait_names_layer{layer}.json")
    if not cache_mat.exists():
        raise FileNotFoundError("Run cache script first to generate trait_matrix_layer16.npy")
    print("  Loading cached trait matrix...", flush=True)
    matrix = np.load(str(cache_mat))   # [229, 4096]
    names  = json.load(open(cache_names))
    return matrix, names


def build_activation_matrix(rows, activations, layer):
    layer_key = str(layer)
    X_list, y_list, valid_rows = [], [], []
    for row in rows:
        pid = row["pair_id"]
        jb  = row.get("jailbroken")
        if jb is None:
            continue
        if pid not in activations or layer_key not in activations[pid]:
            continue
        X_list.append(activations[pid][layer_key].float().numpy())
        y_list.append(1 if jb else 0)
        valid_rows.append(row)
    if not X_list:
        return np.array([]), np.array([]), []
    return np.stack(X_list), np.array(y_list), valid_rows


# ── Pool split ─────────────────────────────────────────────────────────────────

def get_pool_split(rows, train_frac, seed):
    rng = random.Random(seed)
    all_beh = sorted({r["behavior_id"]  for r in rows})
    all_tpl = sorted({r["jailbreak_idx"] for r in rows})
    rng.shuffle(all_beh); rng.shuffle(all_tpl)
    n_beh = max(1, int(len(all_beh) * train_frac))
    n_tpl = max(1, int(len(all_tpl) * train_frac))
    return (set(all_beh[:n_beh]), set(all_tpl[:n_tpl]),
            set(all_beh[n_beh:]),  set(all_tpl[n_tpl:]))


def split_by_pool(rows, train_beh, train_tpl, test_beh, test_tpl):
    tr = [i for i, r in enumerate(rows)
          if r["behavior_id"] in train_beh and r["jailbreak_idx"] in train_tpl]
    te = [i for i, r in enumerate(rows)
          if r["behavior_id"] in test_beh  and r["jailbreak_idx"] in test_tpl]
    return tr, te


# ── Classifier ─────────────────────────────────────────────────────────────────

def fit_eval(X_tr, y_tr, X_te, y_te):
    idx_pos = np.where(y_te == 1)[0]
    idx_neg = np.where(y_te == 0)[0]
    n = min(len(idx_pos), len(idx_neg))
    if n == 0 or len(set(y_te)) < 2:
        return float("nan")
    rng = np.random.RandomState(RANDOM_SEED)
    idx = np.concatenate([rng.choice(idx_pos, n, replace=False),
                          rng.choice(idx_neg, n, replace=False)])
    X_te_b, y_te_b = X_te[idx], y_te[idx]
    if len(set(y_te_b)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED, C=1.0)
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te_b)[:, 1]
    auc   = roc_auc_score(y_te_b, probs)
    return max(auc, 1 - auc)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_classified_path",   default="full_trait_output/harmbench_activations/classified_responses.jsonl")
    parser.add_argument("--human_activations_path",  default="full_trait_output/harmbench_activations/activations.pt")
    parser.add_argument("--gcg_classified_path",     default="full_trait_output/gcg_activations/responses.jsonl")
    parser.add_argument("--gcg_activations_path",    default="full_trait_output/gcg_activations/activations.pt")
    parser.add_argument("--pair_classified_path",    default="full_trait_output/pair_activations/responses.jsonl")
    parser.add_argument("--pair_activations_path",   default="full_trait_output/pair_activations/activations.pt")
    parser.add_argument("--pap_classified_path",     default="full_trait_output/pap_activations/responses.jsonl")
    parser.add_argument("--pap_activations_path",    default="full_trait_output/pap_activations/activations.pt")
    parser.add_argument("--gptfuzz_classified_path", default="full_trait_output/gptfuzz_activations/responses.jsonl")
    parser.add_argument("--gptfuzz_activations_path",default="full_trait_output/gptfuzz_activations/activations.pt")
    parser.add_argument("--pez_classified_path",     default="full_trait_output/pez_activations/responses.jsonl")
    parser.add_argument("--pez_activations_path",    default="full_trait_output/pez_activations/activations.pt")
    parser.add_argument("--output_dir",  default="full_trait_output/trait_subspace_sweep")
    parser.add_argument("--layer",   type=int, default=LAYER)
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n=== Loading ===", flush=True)
    human_rows = [r for r in load_jsonl(args.human_classified_path)
                  if r.get("attack_type") == "human_jailbreak"]
    human_acts = load_activations(args.human_activations_path)

    transfer_datasets = {}
    for name, cp, ap in [
        ("GCG",     args.gcg_classified_path,     args.gcg_activations_path),
        ("PAIR",    args.pair_classified_path,    args.pair_activations_path),
        ("PAP",     args.pap_classified_path,     args.pap_activations_path),
        ("GPTFuzz", args.gptfuzz_classified_path, args.gptfuzz_activations_path),
        ("PEZ",     args.pez_classified_path,     args.pez_activations_path),
    ]:
        if Path(cp).exists() and Path(ap).exists():
            transfer_datasets[name] = (load_jsonl(cp), load_activations(ap))
            print(f"  {name} loaded", flush=True)

    trait_matrix, trait_names = load_trait_matrix(args.layer)  # [229, 4096]
    n_traits = trait_matrix.shape[0]
    print(f"  Trait matrix: {trait_matrix.shape}", flush=True)

    # ── Compute PCA on trait vectors ───────────────────────────────────────────
    # PCA on the 229×4096 trait vector matrix
    # Each PC is a direction in 4096-dim activation space
    print("\n=== PCA on trait vector matrix ===", flush=True)
    print(f"  Running PCA on {trait_matrix.shape} trait matrix...", flush=True)

    max_k    = min(max(K_VALUES), n_traits, 4096)
    pca_on_traits = PCA(n_components=max_k, random_state=RANDOM_SEED)
    pca_on_traits.fit(trait_matrix)  # fit on [229, 4096] — PCs are in activation space

    # Components: [max_k, 4096] — each row is a direction in activation space
    components = pca_on_traits.components_  # [max_k, 4096]
    var_ratio  = pca_on_traits.explained_variance_ratio_

    print(f"  Variance explained:", flush=True)
    cumvar = 0.0
    for i, v in enumerate(var_ratio[:10]):
        cumvar += v
        print(f"    PC{i+1}: {v:.3f} (cumulative: {cumvar:.3f})", flush=True)

    # Save PCA components for inspection
    np.save(output_dir / "trait_subspace_components.npy", components)
    with open(output_dir / "trait_subspace_variance.json", "w") as f:
        json.dump({
            "explained_variance_ratio": var_ratio.tolist(),
            "n_components": int(max_k),
        }, f, indent=2)

    # ── Build raw activation matrices ─────────────────────────────────────────
    print("\n=== Building activation matrices ===", flush=True)
    X_human_raw, y_human, human_valid = build_activation_matrix(
        human_rows, human_acts, args.layer)
    print(f"  Human: {len(y_human)} ({y_human.sum():.0f} jb)", flush=True)

    transfer_raw = {}
    families = []
    for name, (rows, acts) in transfer_datasets.items():
        X, y, _ = build_activation_matrix(rows, acts, args.layer)
        transfer_raw[name] = (X, y)
        families.append(name)
        print(f"  {name}: {len(y)} pairs, chance={y.mean():.3f}", flush=True)

    # ── Project onto top-k trait subspace PCs ────────────────────────────────
    # X_raw: [N, 4096], components[:k]: [k, 4096]
    # Projection: X_raw @ components[:k].T → [N, k]
    print("\n=== Pre-projecting onto trait subspace PCs ===", flush=True)
    X_human_proj = X_human_raw @ components.T   # [N_human, max_k]
    transfer_proj = {
        name: (X @ components.T, y)             # [N_transfer, max_k]
        for name, (X, y) in transfer_raw.items()
    }
    print(f"  Projections done: {X_human_proj.shape}", flush=True)

    # Clamp k values
    k_values = sorted(set(min(k, max_k) for k in K_VALUES))
    print(f"\n  Sweeping k = {k_values}", flush=True)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    print(f"\n=== Trait subspace sweep ({len(k_values)} values × {args.n_seeds} seeds) ===",
          flush=True)

    results = {k: {f: [] for f in families + ["human_test"]} for k in k_values}

    for seed in range(args.n_seeds):
        tr_beh, tr_tpl, te_beh, te_tpl = get_pool_split(human_valid, TRAIN_FRAC, seed)
        tr_idx, te_idx = split_by_pool(human_valid, tr_beh, tr_tpl, te_beh, te_tpl)
        if not tr_idx or not te_idx:
            continue

        # Scale on the full max_k projection (fit once, slice for each k)
        scaler = StandardScaler()
        X_tr_full = scaler.fit_transform(X_human_proj[tr_idx])
        X_te_full = scaler.transform(X_human_proj[te_idx])
        y_tr = y_human[tr_idx]
        y_te = y_human[te_idx]

        transfer_scaled = {
            name: scaler.transform(Xp)
            for name, (Xp, _) in transfer_proj.items()
        }

        for k in k_values:
            # Slice to first k components
            X_tr_k = X_tr_full[:, :k]
            X_te_k = X_te_full[:, :k]

            results[k]["human_test"].append(fit_eval(X_tr_k, y_tr, X_te_k, y_te))

            for name, (_, y_t) in transfer_proj.items():
                X_t_k = transfer_scaled[name][:, :k]
                results[k][name].append(fit_eval(X_tr_k, y_tr, X_t_k, y_t))

        if seed % 10 == 0:
            print(f"  Seed {seed} done", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "=" * 110
    print(f"\n\n{sep}")
    print(f"  TRAIT SUBSPACE SWEEP  |  Layer {args.layer}  |  {args.n_seeds} seeds")
    print(f"  PCA on 229×4096 trait vector matrix → top-k directions in activation space")
    print(f"  k=1 = assistant axis (PC1 of trait matrix)")
    print(sep)

    header = f"  {'k':>5}  {'VarExpl':>8}  {'HumanTest':>10}  {'AvgXfer':>10}" + \
             "".join(f"  {f:>10}" for f in families)
    print(f"\n{header}")
    print("  " + "─" * (5 + 8 + 10 + 10 + 12 * len(families)))

    best_k   = None
    best_avg = 0.0
    summary  = {}

    for k in k_values:
        cumvar = float(var_ratio[:k].sum())
        ht     = float(np.nanmean(results[k]["human_test"]))
        fam    = [float(np.nanmean(results[k][f])) for f in families]
        avg    = float(np.nanmean(fam))

        marker = ""
        if avg > best_avg:
            best_avg = avg
            best_k   = k
            marker   = " ← best"
        if k == 1:
            marker += " (assistant axis)"

        row  = f"  {k:>5}  {cumvar:>8.3f}  {ht:>10.4f}  {avg:>10.4f}"
        row += "".join(f"  {v:>10.4f}" for v in fam)
        print(row + marker)

        summary[k] = {
            "cumulative_variance": cumvar,
            "human_test": {"mean": ht, "std": float(np.nanstd(results[k]["human_test"]))},
            "avg_transfer": avg,
            **{f: {
                "mean": float(np.nanmean(results[k][f])),
                "std":  float(np.nanstd(results[k][f])),
            } for f in families},
        }

    print(f"\n  Best k by avg transfer: k={best_k} (avg={best_avg:.4f})")
    print(f"\n  Per-family best k:")
    for f in families:
        best_k_f   = max(k_values, key=lambda k: np.nanmean(results[k][f]))
        best_auc_f = float(np.nanmean(results[best_k_f][f]))
        print(f"    {f:10s}: k={best_k_f} (AUC={best_auc_f:.4f})")

    print(f"\n{sep}")

    out_path = output_dir / "trait_subspace_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "layer": args.layer,
            "n_seeds": args.n_seeds,
            "k_values": k_values,
            "n_traits": n_traits,
            "families": families,
            "best_k_avg_transfer": best_k,
            "variance_explained": var_ratio[:max(k_values)].tolist(),
            "results": {str(k): v for k, v in summary.items()},
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
