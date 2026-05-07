#!/usr/bin/env python3
"""
run_trait_pca_sweep.py

Sweeps over number of PCA components applied to the 229-dimensional trait
projection space. Finds the optimal k for transfer classification.

The idea: instead of using all 229 trait projections or top-k by cosine
similarity to w, apply PCA to find the k directions of maximum variance
in trait space that are predictive of jailbreak behavior.

Tests k = [2, 4, 6, 8, 10, 15, 20, 30, 50, 100, 229(full)]
50 seeds, balanced AUC, strict pool split — same as run_transfer_classifier.py

Also reports logistic regression on full traits (229) as the baseline
so you can see clearly when PCA helps vs hurts.

Usage:
  uv run python full_trait_tools/run_trait_pca_sweep.py
  uv run python full_trait_tools/run_trait_pca_sweep.py --output_dir full_trait_output/trait_pca_sweep
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

# k values to sweep — includes full 229 as upper bound
K_VALUES = [2, 4, 6, 8, 10, 15, 20, 30, 50, 100, 229]


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
        raise FileNotFoundError(
            "Trait matrix cache not found. Generate it first with the cache script.")
    print("  Loading cached trait matrix...", flush=True)
    return np.load(str(cache_mat)), json.load(open(cache_names))


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


def project_traits(X_raw, trait_matrix):
    return X_raw @ trait_matrix.T


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
    """Logistic regression with balanced 50/50 test set and best-direction AUC."""
    # Balance test set
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
    parser.add_argument("--output_dir",  default="full_trait_output/trait_pca_sweep")
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
    print(f"  Human: {len(human_rows)} rows", flush=True)

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

    trait_matrix, trait_names = load_trait_matrix(args.layer)
    n_traits = trait_matrix.shape[0]
    print(f"  {n_traits} trait vectors", flush=True)

    # ── Build trait projection matrices ───────────────────────────────────────
    print("\n=== Building trait projections ===", flush=True)
    X_human_raw, y_human, human_valid = build_activation_matrix(
        human_rows, human_acts, args.layer)
    X_human_traits = project_traits(X_human_raw, trait_matrix)
    print(f"  Human traits: {X_human_traits.shape}", flush=True)

    transfer_traits = {}
    families = []
    for name, (rows, acts) in transfer_datasets.items():
        X, y, _ = build_activation_matrix(rows, acts, args.layer)
        X_tr = project_traits(X, trait_matrix)
        transfer_traits[name] = (X_tr, y)
        families.append(name)
        print(f"  {name}: {len(y)} pairs, chance={y.mean():.3f}", flush=True)

    # Clamp K_VALUES to valid range
    k_values = [k if k < n_traits else n_traits for k in K_VALUES]
    k_values = sorted(set(k_values))
    print(f"\n  Sweeping k = {k_values}", flush=True)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    print(f"\n=== Trait PCA sweep ({len(k_values)} values × {args.n_seeds} seeds) ===",
          flush=True)

    # results[k] = {family: [auc_per_seed], "human_test": [auc_per_seed]}
    results = {k: {f: [] for f in families + ["human_test"]} for k in k_values}

    for seed in range(args.n_seeds):
        tr_beh, tr_tpl, te_beh, te_tpl = get_pool_split(human_valid, TRAIN_FRAC, seed)
        tr_idx, te_idx = split_by_pool(human_valid, tr_beh, tr_tpl, te_beh, te_tpl)
        if not tr_idx or not te_idx:
            continue

        # Scale trait projections (fit on train, apply to all)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_human_traits[tr_idx])
        X_te_scaled = scaler.transform(X_human_traits[te_idx])
        y_tr = y_human[tr_idx]
        y_te = y_human[te_idx]

        transfer_scaled = {
            name: scaler.transform(Xt)
            for name, (Xt, _) in transfer_traits.items()
        }

        for k in k_values:
            if k >= n_traits:
                # Full trait space — no PCA
                X_tr_k = X_tr_scaled
                X_te_k = X_te_scaled
                transfer_k = transfer_scaled
            else:
                pca = PCA(n_components=k, random_state=RANDOM_SEED)
                X_tr_k = pca.fit_transform(X_tr_scaled)
                X_te_k = pca.transform(X_te_scaled)
                transfer_k = {
                    name: pca.transform(Xs)
                    for name, Xs in transfer_scaled.items()
                }

            # Human test AUC
            results[k]["human_test"].append(
                fit_eval(X_tr_k, y_tr, X_te_k, y_te))

            # Transfer AUC for each family
            for name, (_, y_t) in transfer_traits.items():
                results[k][name].append(
                    fit_eval(X_tr_k, y_tr, transfer_k[name], y_t))

        if seed % 10 == 0:
            print(f"  Seed {seed} done", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = "=" * 110
    print(f"\n\n{sep}")
    print(f"  TRAIT PCA SWEEP  |  Layer {args.layer}  |  {args.n_seeds} seeds")
    print(f"  Logistic regression on PCA-reduced trait projections (229 = no PCA)")
    print(sep)

    header = f"  {'k':>6}  {'HumanTest':>10}  {'AvgXfer':>10}" + \
             "".join(f"  {f:>10}" for f in families)
    print(f"\n{header}")
    print("  " + "─" * (6 + 10 + 10 + 12 * len(families)))

    best_k_by_avg = None
    best_avg      = 0.0
    summary       = {}

    for k in k_values:
        ht  = float(np.nanmean(results[k]["human_test"]))
        fam = [float(np.nanmean(results[k][f])) for f in families]
        avg = float(np.nanmean(fam))

        label = " ← best" if avg > best_avg else ""
        if avg > best_avg:
            best_avg = avg
            best_k_by_avg = k

        row = f"  {k:>6}  {ht:>10.4f}  {avg:>10.4f}"
        row += "".join(f"  {v:>10.4f}" for v in fam)
        if k == n_traits:
            row += "  (no PCA)"
        print(row + label)

        summary[k] = {
            "human_test": {"mean": ht, "std": float(np.nanstd(results[k]["human_test"]))},
            "avg_transfer": avg,
            **{f: {
                "mean": float(np.nanmean(results[k][f])),
                "std":  float(np.nanstd(results[k][f])),
            } for f in families},
        }

    print(f"\n  Best k by avg transfer AUC: k={best_k_by_avg} (avg={best_avg:.4f})")

    # Per-family best k
    print(f"\n  Best k per family:")
    for f in families:
        best_k_f   = max(k_values, key=lambda k: np.nanmean(results[k][f]))
        best_auc_f = float(np.nanmean(results[best_k_f][f]))
        print(f"    {f:10s}: k={best_k_f} (AUC={best_auc_f:.4f})")

    print(f"\n{sep}")

    # Save
    out_path = output_dir / "trait_pca_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "layer": args.layer,
            "n_seeds": args.n_seeds,
            "k_values": k_values,
            "n_traits": n_traits,
            "families": families,
            "best_k_avg_transfer": best_k_by_avg,
            "results": {str(k): v for k, v in summary.items()},
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
