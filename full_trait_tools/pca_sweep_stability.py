#!/usr/bin/env python3
"""
pca_sweep_stability.py

Sweeps PCA dimensionality from low to high, measuring at each level:
  - Mean pairwise cosine similarity between hyperplane normals (stability)
  - Mean ROC-AUC across seeds (predictive power)

Plots both curves together to find the sweet spot where stability is
high AND AUC hasn't collapsed. Saves a PDF with the plots.

If no sweet spot exists (stability only improves when AUC collapses),
this definitively shows the jailbreak signal is too high-dimensional
to be captured stably with this dataset size.

Usage:
  uv run full_trait_tools/pca_sweep_stability.py
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RANDOM_SEED      = 42
TRAIN_FRAC       = 0.7
MIN_SUCCESS_RATE = 0.20
MAX_SUCCESS_RATE = 0.80
LAYERS           = [16, 28]
N_SEEDS          = 8

# PCA component counts to sweep
PCA_SWEEP = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_classified(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_activations(path: Path) -> Dict[int, Dict[str, torch.Tensor]]:
    return torch.load(path, map_location="cpu", weights_only=False)


# ── Filtering + splitting ──────────────────────────────────────────────────────

def filter_human_jailbreak(rows):
    return [r for r in rows if r.get("attack_type") == "human_jailbreak"]


def compute_behavior_success_rates(rows):
    counts: dict = defaultdict(lambda: {"total": 0, "jailbroken": 0})
    for r in rows:
        counts[r["behavior_id"]]["total"] += 1
        if r["jailbroken"]:
            counts[r["behavior_id"]]["jailbroken"] += 1
    return {
        bid: c["jailbroken"] / c["total"]
        for bid, c in counts.items() if c["total"] > 0
    }


def filter_by_variance(rows, success_rates, min_rate, max_rate):
    kept = {bid for bid, r in success_rates.items() if min_rate <= r <= max_rate}
    return [r for r in rows if r["behavior_id"] in kept], sorted(kept)


def split_pools(rows, train_frac, seed):
    rng = random.Random(seed)
    all_behaviors = sorted({r["behavior_id"]   for r in rows})
    all_templates = sorted({r["jailbreak_idx"]  for r in rows})
    rng.shuffle(all_behaviors)
    rng.shuffle(all_templates)
    n_train_beh = max(1, int(len(all_behaviors) * train_frac))
    n_train_tpl = max(1, int(len(all_templates) * train_frac))
    train_behaviors = set(all_behaviors[:n_train_beh])
    test_behaviors  = set(all_behaviors[n_train_beh:])
    train_templates = set(all_templates[:n_train_tpl])
    test_templates  = set(all_templates[n_train_tpl:])
    return train_behaviors, test_behaviors, train_templates, test_templates


def get_activations_and_labels(rows, activations, layer, behavior_pool=None, template_pool=None):
    layer_key = str(layer)
    X_list, y_list = [], []
    for row in rows:
        if behavior_pool is not None and row["behavior_id"] not in behavior_pool:
            continue
        if template_pool is not None and row["jailbreak_idx"] not in template_pool:
            continue
        pid = row["pair_id"]
        if pid not in activations or layer_key not in activations[pid]:
            continue
        X_list.append(activations[pid][layer_key].float().numpy())
        y_list.append(int(row["jailbroken"]))
    if not X_list:
        return np.array([]), np.array([])
    return np.stack(X_list), np.array(y_list)


# ── Core ───────────────────────────────────────────────────────────────────────

def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def cosine_sim(a, b):
    return float(np.dot(unit(a), unit(b)))


def pairwise_mean_cosine(vectors):
    n = len(vectors)
    if n < 2:
        return float("nan")
    cos_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_sims.append(cosine_sim(vectors[i], vectors[j]))
    return float(np.mean(cos_sims)), float(np.std(cos_sims))


def run_sweep_for_layer(
    layer: int,
    rows_filtered: List[dict],
    activations: Dict,
    n_seeds: int,
    train_frac: float,
    pca_sweep: List[int],
) -> List[dict]:
    """
    For each n_components in pca_sweep, train N_SEEDS classifiers and measure
    mean stability (pairwise cosine) and mean AUC.
    """
    # Get all activations once
    X_all, y_all = get_activations_and_labels(rows_filtered, activations, layer)
    print(f"  Layer {layer}: {X_all.shape[0]} samples, {X_all.shape[1]} dims")

    # Fit scaler once on all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # Pre-generate splits for all seeds
    splits = []
    for seed in range(n_seeds):
        train_beh, test_beh, train_tpl, test_tpl = split_pools(
            rows_filtered, train_frac, seed
        )
        X_train, y_train = get_activations_and_labels(
            rows_filtered, activations, layer, train_beh, train_tpl
        )
        X_test, y_test = get_activations_and_labels(
            rows_filtered, activations, layer, test_beh, test_tpl
        )
        if len(X_train) < 20 or len(X_test) < 5:
            continue
        # Scale
        X_train_s = scaler.transform(X_train)
        X_test_s  = scaler.transform(X_test)
        splits.append((X_train_s, y_train, X_test_s, y_test, seed))

    print(f"  {len(splits)} valid splits")

    # Also measure raw (no PCA) stability for reference
    raw_ws = []
    for X_train_s, y_train, X_test_s, y_test, seed in splits:
        clf = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000,
            random_state=seed, class_weight="balanced",
        )
        clf.fit(X_train_s, y_train)
        raw_ws.append(unit(clf.coef_[0]))

    raw_mean_cos, raw_std_cos = pairwise_mean_cosine(raw_ws)
    raw_aucs = []
    for (X_train_s, y_train, X_test_s, y_test, seed), w in zip(splits, raw_ws):
        clf = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000,
            random_state=seed, class_weight="balanced",
        )
        clf.fit(X_train_s, y_train)
        try:
            raw_aucs.append(float(roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])))
        except Exception:
            pass

    print(f"  Raw (no PCA): stability={raw_mean_cos:.3f}, AUC={np.mean(raw_aucs):.3f}")

    results = [{
        "n_components":   "raw",
        "var_explained":  1.0,
        "mean_cos_sim":   raw_mean_cos,
        "std_cos_sim":    raw_std_cos,
        "mean_auc":       float(np.mean(raw_aucs)),
        "std_auc":        float(np.std(raw_aucs)),
    }]

    # Sweep PCA components
    for n_components in tqdm(pca_sweep, desc=f"  PCA sweep [layer {layer}]"):
        # Cap at available dimensions
        n_comp = min(n_components, X_scaled.shape[1], X_scaled.shape[0] - 1)
        if n_comp < 2:
            continue

        pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
        pca.fit(X_scaled)
        var_explained = float(pca.explained_variance_ratio_.sum())

        ws_pca  = []   # w in PCA space
        aucs    = []

        for X_train_s, y_train, X_test_s, y_test, seed in splits:
            X_train_pca = pca.transform(X_train_s)
            X_test_pca  = pca.transform(X_test_s)

            clf = LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=1000,
                random_state=seed, class_weight="balanced",
            )
            clf.fit(X_train_pca, y_train)
            ws_pca.append(unit(clf.coef_[0]))

            try:
                aucs.append(float(roc_auc_score(
                    y_test, clf.predict_proba(X_test_pca)[:, 1]
                )))
            except Exception:
                pass

        mean_cos, std_cos = pairwise_mean_cosine(ws_pca)

        results.append({
            "n_components":  n_comp,
            "var_explained": var_explained,
            "mean_cos_sim":  mean_cos,
            "std_cos_sim":   std_cos,
            "mean_auc":      float(np.mean(aucs)) if aucs else float("nan"),
            "std_auc":       float(np.std(aucs))  if aucs else float("nan"),
        })

        print(f"    n={n_comp:4d} ({100*var_explained:.1f}% var)  "
              f"stability={mean_cos:.3f}±{std_cos:.3f}  "
              f"AUC={np.mean(aucs) if aucs else float('nan'):.3f}")

    return results


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_sweep(layer_results: Dict[int, List[dict]], output_path: Path) -> None:
    with PdfPages(str(output_path)) as pdf:
        for layer, results in layer_results.items():

            # Separate raw from PCA results
            raw = next((r for r in results if r["n_components"] == "raw"), None)
            pca_results = [r for r in results if r["n_components"] != "raw"]

            if not pca_results:
                continue

            n_comps      = [r["n_components"]  for r in pca_results]
            mean_cos     = [r["mean_cos_sim"]  for r in pca_results]
            std_cos      = [r["std_cos_sim"]   for r in pca_results]
            mean_auc     = [r["mean_auc"]      for r in pca_results]
            std_auc      = [r["std_auc"]       for r in pca_results]
            var_exp      = [r["var_explained"] for r in pca_results]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            fig.suptitle(
                f"Layer {layer}: Stability vs AUC across PCA dimensionalities\n"
                f"(n_seeds={N_SEEDS}, strict pool split)",
                fontsize=13,
            )

            # ── Top: stability ────────────────────────────────────────────────
            ax1.plot(n_comps, mean_cos, "o-", color="#2563eb",
                     linewidth=2, markersize=6, label="Mean pairwise cos_sim")
            ax1.fill_between(
                n_comps,
                [m - s for m, s in zip(mean_cos, std_cos)],
                [m + s for m, s in zip(mean_cos, std_cos)],
                alpha=0.2, color="#2563eb",
            )
            if raw:
                ax1.axhline(raw["mean_cos_sim"], color="#2563eb",
                            linestyle="--", linewidth=1.5, alpha=0.6,
                            label=f"Raw (no PCA): {raw['mean_cos_sim']:.3f}")
            ax1.axhline(0.7, color="green", linestyle=":", linewidth=1,
                        label="Stability threshold (0.7)")
            ax1.axhline(0.0, color="gray", linestyle="-", linewidth=0.5)
            ax1.set_ylabel("Mean pairwise cosine similarity\n(higher = more stable)", fontsize=11)
            ax1.set_ylim(-0.1, 1.05)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.set_title("Stability of hyperplane normal w", fontsize=11)

            # ── Bottom: AUC ───────────────────────────────────────────────────
            ax2.plot(n_comps, mean_auc, "o-", color="#dc2626",
                     linewidth=2, markersize=6, label="Mean ROC-AUC")
            ax2.fill_between(
                n_comps,
                [m - s for m, s in zip(mean_auc, std_auc)],
                [m + s for m, s in zip(mean_auc, std_auc)],
                alpha=0.2, color="#dc2626",
            )
            if raw:
                ax2.axhline(raw["mean_auc"], color="#dc2626",
                            linestyle="--", linewidth=1.5, alpha=0.6,
                            label=f"Raw (no PCA): {raw['mean_auc']:.3f}")
            ax2.axhline(0.7, color="green", linestyle=":", linewidth=1,
                        label="AUC threshold (0.7)")
            ax2.axhline(0.5, color="gray", linestyle="-", linewidth=0.5,
                        label="Chance (0.5)")

            # Add variance explained as secondary axis
            ax2b = ax2.twinx()
            ax2b.plot(n_comps, [100 * v for v in var_exp],
                      "s--", color="#7c3aed", linewidth=1, markersize=4,
                      alpha=0.6, label="Variance explained (%)")
            ax2b.set_ylabel("Variance explained (%)", color="#7c3aed", fontsize=10)
            ax2b.tick_params(axis="y", labelcolor="#7c3aed")
            ax2b.set_ylim(0, 105)

            ax2.set_xlabel("Number of PCA components", fontsize=11)
            ax2.set_ylabel("Mean ROC-AUC\n(higher = more predictive)", fontsize=11)
            ax2.set_ylim(0.4, 1.05)
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2b.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.set_title("Predictive power (AUC) + variance explained", fontsize=11)

            # Shade the sweet spot: AUC > 0.7 AND stability > 0.7
            sweet_x = [
                n for n, c, a in zip(n_comps, mean_cos, mean_auc)
                if c > 0.7 and a > 0.7
            ]
            if sweet_x:
                for ax in [ax1, ax2]:
                    ax.axvspan(min(sweet_x), max(sweet_x),
                               alpha=0.12, color="green",
                               label="Sweet spot" if ax == ax1 else "")
                print(f"  Layer {layer}: sweet spot at n_components = {sweet_x}")
            else:
                print(f"  Layer {layer}: no sweet spot found (stability & AUC never both > 0.7)")

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"  PDF saved to {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classified_path", type=str,
        default="full_trait_output/harmbench_activations/classified_responses.jsonl",
    )
    parser.add_argument(
        "--activations_path", type=str,
        default="full_trait_output/harmbench_activations/activations.pt",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="full_trait_output/harmbench_logreg",
    )
    parser.add_argument("--n_seeds",          type=int,   default=N_SEEDS)
    parser.add_argument("--min_success_rate", type=float, default=MIN_SUCCESS_RATE)
    parser.add_argument("--max_success_rate", type=float, default=MAX_SUCCESS_RATE)
    parser.add_argument("--train_frac",       type=float, default=TRAIN_FRAC)
    parser.add_argument("--layers", nargs="+", type=int,  default=LAYERS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    rows        = load_classified(Path(args.classified_path))
    activations = load_activations(Path(args.activations_path))
    rows        = filter_human_jailbreak(rows)
    success_rates = compute_behavior_success_rates(rows)
    rows_filtered, kept_behaviors = filter_by_variance(
        rows, success_rates,
        min_rate=args.min_success_rate,
        max_rate=args.max_success_rate,
    )
    n_jb = sum(r["jailbroken"] for r in rows_filtered)
    print(f"  {len(kept_behaviors)} behaviors, {len(rows_filtered)} pairs "
          f"({n_jb} jailbroken, {len(rows_filtered)-n_jb} not)\n")

    all_results = {}

    for layer in args.layers:
        print(f"\n{'='*60}")
        print(f"  LAYER {layer}")
        print(f"{'='*60}")
        results = run_sweep_for_layer(
            layer=layer,
            rows_filtered=rows_filtered,
            activations=activations,
            n_seeds=args.n_seeds,
            train_frac=args.train_frac,
            pca_sweep=PCA_SWEEP,
        )
        all_results[layer] = results

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_json = output_dir / "pca_sweep_results.json"
    with open(out_json, "w") as f:
        json.dump(
            {str(layer): results for layer, results in all_results.items()},
            f, indent=2,
        )
    print(f"\nResults saved to {out_json}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    print("\nGenerating PDF...")
    plot_sweep(all_results, output_dir / "pca_sweep_stability.pdf")

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for layer, results in all_results.items():
        print(f"\n  Layer {layer}:")
        pca_results = [r for r in results if r["n_components"] != "raw"]
        for r in pca_results:
            sweet = (
                " ← SWEET SPOT"
                if r["mean_cos_sim"] > 0.7 and r["mean_auc"] > 0.7
                else ""
            )
            print(f"    n={r['n_components']:4}  "
                  f"stability={r['mean_cos_sim']:.3f}  "
                  f"AUC={r['mean_auc']:.3f}{sweet}")


if __name__ == "__main__":
    main()
