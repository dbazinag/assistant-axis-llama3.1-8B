#!/usr/bin/env python3
"""
stable_hyperplane_analysis.py

Produces a stable jailbreak hyperplane normal by:
  1. Running PCA on all harmbench activations to reduce dimensionality
  2. Fitting logistic regression in PCA space across N seeds
  3. Averaging the resulting weight vectors (cancels noise, reinforces signal)
  4. Mapping the averaged w back to 4096-dim space
  5. Comparing w to all trait vectors and assistant axis in PCA space
     where geometric comparisons are meaningful

This solves two problems simultaneously:
  - Instability: fewer dimensions = fewer degrees of freedom = more stable w
  - Dimensionality: comparison happens in reduced space where cosine
    similarities are meaningful rather than near-zero by construction

Outputs:
  - Stability comparison: before vs after PCA reduction
  - Cosine similarities in PCA space (interpretable)
  - Top/bottom trait rankings
  - Assistant axis result highlighted
  - stable_hyperplane_layer{N}.pt — the averaged unit-norm w in 4096-dim space

Usage:
  uv run full_trait_tools/stable_hyperplane_analysis.py
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

RANDOM_SEED      = 42
TRAIN_FRAC       = 0.7
MIN_SUCCESS_RATE = 0.20
MAX_SUCCESS_RATE = 0.80
LAYERS           = [16, 28]
N_SEEDS          = 10
N_PCA_COMPONENTS = 15   # keep top 100 PCs — adjust if variance explained is too low
N_TOP            = 20


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


def load_trait_vectors(
    vectors_dir: Path, layer: int
) -> Tuple[List[str], np.ndarray]:
    pt_files = sorted(vectors_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {vectors_dir}")
    trait_names, vectors = [], []
    for pt_file in pt_files:
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        trait_names.append(pt_file.stem)
        vectors.append(data["vector"][layer].float().numpy())
    return trait_names, np.stack(vectors)  # [n_traits, 4096]


def load_axis_vector(axis_path: Path, layer: int) -> np.ndarray:
    data = torch.load(axis_path, map_location="cpu", weights_only=False)
    return data["axis"][layer].float().numpy()  # [4096]


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


def get_activations_and_labels(
    rows: List[dict],
    activations: Dict[int, Dict[str, torch.Tensor]],
    layer: int,
    behavior_pool: Set[str] = None,
    template_pool: Set[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get raw activations and labels for given rows.
    If behavior_pool and template_pool are None, use all rows.
    """
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


# ── Core computation ───────────────────────────────────────────────────────────

def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def angle_deg(cos: float) -> float:
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def pairwise_cosine_stats(vectors: List[np.ndarray]) -> dict:
    """Compute pairwise cosine similarities between a list of unit vectors."""
    n = len(vectors)
    cos_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_sims.append(cosine_sim(vectors[i], vectors[j]))
    cos_sims = np.array(cos_sims)
    return {
        "mean":      float(cos_sims.mean()),
        "std":       float(cos_sims.std()),
        "min":       float(cos_sims.min()),
        "max":       float(cos_sims.max()),
        "mean_angle_deg": float(angle_deg(cos_sims.mean())),
    }


def learn_w_raw(X_train, y_train, seed, C=1.0):
    """Learn hyperplane normal in raw 4096-dim space."""
    scaler    = StandardScaler()
    Xtr       = scaler.fit_transform(X_train)
    clf = LogisticRegression(
        C=C, solver="lbfgs", max_iter=1000,
        random_state=seed, class_weight="balanced",
    )
    clf.fit(Xtr, y_train)
    w = clf.coef_[0] / (scaler.scale_ + 1e-12)
    return unit(w)


def learn_w_pca(X_train, y_train, pca, seed, C=1.0):
    """
    Learn hyperplane normal in PCA-reduced space,
    then map back to 4096-dim space.
    """
    # Project training data into PCA space
    X_pca = pca.transform(X_train)   # [n_train, n_components]

    clf = LogisticRegression(
        C=C, solver="lbfgs", max_iter=1000,
        random_state=seed, class_weight="balanced",
    )
    clf.fit(X_pca, y_train)

    # w_pca is in PCA space: [n_components]
    w_pca = clf.coef_[0]

    # Map back to original 4096-dim space via PCA components
    # pca.components_ is [n_components, 4096]
    # w_4096 = w_pca @ pca.components_
    w_4096 = w_pca @ pca.components_   # [4096]

    return unit(w_pca), unit(w_4096)


def evaluate_auc(X_test, y_test, X_train, y_train, pca=None, C=1.0, seed=42):
    """Quick AUC evaluation for a train/test split."""
    if pca is not None:
        X_tr = pca.transform(X_train)
        X_te = pca.transform(X_test)
    else:
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_train)
        X_te   = scaler.transform(X_test)

    clf = LogisticRegression(
        C=C, solver="lbfgs", max_iter=1000,
        random_state=seed, class_weight="balanced",
    )
    clf.fit(X_tr, y_tr := y_train)
    y_prob = clf.predict_proba(X_te)[:, 1]
    try:
        return float(roc_auc_score(y_test, y_prob))
    except ValueError:
        return float("nan")


# ── Main analysis ──────────────────────────────────────────────────────────────

def run_layer_analysis(
    layer: int,
    rows_filtered: List[dict],
    activations: Dict[int, Dict[str, torch.Tensor]],
    trait_names: List[str],
    trait_vectors: np.ndarray,   # [n_traits, 4096]
    axis_vector: np.ndarray,     # [4096]
    n_seeds: int,
    n_pca_components: int,
    train_frac: float,
    n_top: int,
    output_dir: Path,
) -> dict:

    sep = "=" * 70

    print(f"\n{sep}")
    print(f"  LAYER {layer}")
    print(sep)

    # ── Get all activations for PCA fitting ────────────────────────────────────
    print(f"\n  Loading all activations for layer {layer}...")
    X_all, y_all = get_activations_and_labels(rows_filtered, activations, layer)
    print(f"  {X_all.shape[0]} samples, {X_all.shape[1]} dimensions")

    # ── Fit PCA on all data ────────────────────────────────────────────────────
    print(f"\n  Fitting PCA ({n_pca_components} components)...")
    scaler_pca = StandardScaler()
    X_scaled   = scaler_pca.fit_transform(X_all)

    pca = PCA(n_components=n_pca_components, random_state=RANDOM_SEED)
    pca.fit(X_scaled)

    var_explained = float(pca.explained_variance_ratio_.sum())
    print(f"  Variance explained by {n_pca_components} PCs: {100*var_explained:.1f}%")

    # ── Collect w vectors from N seeds: raw and PCA ────────────────────────────
    print(f"\n  Training {n_seeds} classifiers (raw + PCA) across seeds...")

    raw_ws  = []   # [n_seeds, 4096]
    pca_ws_pca  = []   # [n_seeds, n_components]  — w in PCA space
    pca_ws_4096 = []   # [n_seeds, 4096]           — w mapped back
    auc_raw_list  = []
    auc_pca_list  = []

    for seed in tqdm(range(n_seeds), desc=f"  Seeds[layer {layer}]"):
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

        # Raw
        w_raw = learn_w_raw(X_train, y_train, seed)
        raw_ws.append(w_raw)

        # PCA — use the PCA fitted on all data
        # (In practice fitting PCA on train-only is cleaner but since
        #  PCA is unsupervised and we're using it for w comparison,
        #  fitting on all data is standard and avoids instability)
        X_train_pca = pca.transform(scaler_pca.transform(X_train))
        X_test_pca  = pca.transform(scaler_pca.transform(X_test))

        clf_pca = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000,
            random_state=seed, class_weight="balanced",
        )
        clf_pca.fit(X_train_pca, y_train)

        w_pca_space = unit(clf_pca.coef_[0])             # [n_components]
        w_4096      = unit(w_pca_space @ pca.components_) # [4096]
        pca_ws_pca.append(w_pca_space)
        pca_ws_4096.append(w_4096)

        # AUC
        try:
            y_prob_raw = LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=1000,
                random_state=seed, class_weight="balanced"
            ).fit(
                StandardScaler().fit_transform(X_train), y_train
            ).predict_proba(
                StandardScaler().fit(X_train).transform(X_test)
            )[:, 1]
            auc_raw_list.append(float(roc_auc_score(y_test, y_prob_raw)))
        except Exception:
            pass

        try:
            auc_pca_list.append(
                float(roc_auc_score(
                    y_test, clf_pca.predict_proba(X_test_pca)[:, 1]
                ))
            )
        except Exception:
            pass

    # ── Stability comparison ───────────────────────────────────────────────────
    raw_stats = pairwise_cosine_stats(raw_ws)
    pca_stats = pairwise_cosine_stats(pca_ws_4096)

    print(f"\n  Stability comparison (pairwise cosine sim between w vectors):")
    print(f"  {'':20s}  {'Raw 4096-dim':>14}  {'PCA {}-dim'.format(n_pca_components):>14}")
    print(f"  {'Mean cos_sim':20s}  {raw_stats['mean']:>14.4f}  {pca_stats['mean']:>14.4f}")
    print(f"  {'Std  cos_sim':20s}  {raw_stats['std']:>14.4f}  {pca_stats['std']:>14.4f}")
    print(f"  {'Min  cos_sim':20s}  {raw_stats['min']:>14.4f}  {pca_stats['min']:>14.4f}")
    print(f"  {'Max  cos_sim':20s}  {raw_stats['max']:>14.4f}  {pca_stats['max']:>14.4f}")
    print(f"  {'Mean angle':20s}  {raw_stats['mean_angle_deg']:>13.2f}°  "
          f"{pca_stats['mean_angle_deg']:>13.2f}°")
    print(f"\n  Mean AUC (raw)  : {np.mean(auc_raw_list):.4f} ± {np.std(auc_raw_list):.4f}")
    print(f"  Mean AUC (PCA)  : {np.mean(auc_pca_list):.4f} ± {np.std(auc_pca_list):.4f}")

    # ── Compute stable averaged w ──────────────────────────────────────────────
    # Average in PCA space (where the geometry is clean), then map back
    w_avg_pca  = unit(np.stack(pca_ws_pca).mean(axis=0))   # [n_components]
    w_avg_4096 = unit(w_avg_pca @ pca.components_)           # [4096]

    print(f"\n  Stable w computed by averaging {len(pca_ws_pca)} seed vectors")
    print(f"  w_avg norm (PCA space)  : {np.linalg.norm(w_avg_pca):.4f}")
    print(f"  w_avg norm (4096 space) : {np.linalg.norm(w_avg_4096):.4f}")

    # ── Project trait vectors into PCA space ───────────────────────────────────
    # pca.components_ is [n_components, 4096]
    # trait vector t [4096] → PCA space: pca.components_ @ t = [n_components]
    # Then unit-normalise for cosine comparison

    print(f"\n  Projecting trait vectors into PCA space...")
    all_names   = trait_names + ["assistant_axis"]
    all_vecs_4096 = np.vstack([trait_vectors, axis_vector[np.newaxis, :]])  # [n+1, 4096]

    # Project each trait vector into PCA space
    # [n+1, 4096] @ [4096, n_components] = [n+1, n_components]
    all_vecs_pca = all_vecs_4096 @ pca.components_.T  # [n+1, n_components]

    # Unit normalise each
    norms = np.linalg.norm(all_vecs_pca, axis=1, keepdims=True)
    all_vecs_pca_unit = all_vecs_pca / (norms + 1e-12)

    # Cosine similarities with w_avg_pca
    cos_sims = all_vecs_pca_unit @ w_avg_pca   # [n+1]

    results = []
    for i, name in enumerate(all_names):
        cos = float(cos_sims[i])
        results.append({
            "trait":       name,
            "cos_sim_pca": cos,
            "angle_deg":   angle_deg(cos),
            "abs_cos":     abs(cos),
            "direction":   "→ jailbreak more likely" if cos > 0
                           else "→ jailbreak less likely",
        })

    # Rank
    pos_sorted   = sorted(results, key=lambda x: x["cos_sim_pca"], reverse=True)
    neg_sorted   = sorted(results, key=lambda x: x["cos_sim_pca"])
    abs_sorted   = sorted(results, key=lambda x: x["abs_cos"], reverse=True)

    for rank, entry in enumerate(pos_sorted, 1):
        entry["rank_pos"] = rank
    for rank, entry in enumerate(neg_sorted, 1):
        entry["rank_neg"] = rank
    for rank, entry in enumerate(abs_sorted, 1):
        entry["rank_abs"] = rank

    # ── Print results ──────────────────────────────────────────────────────────
    header  = (f"  {'Rank':>4}  {'Trait':40s}  "
               f"{'cos_sim':>8}  {'angle':>8}  Direction")
    divider = "  " + "─" * 80

    print(f"\n{sep}")
    print(f"  LAYER {layer} — Cosine Similarities in PCA Space ({n_pca_components}-dim)")
    print(f"  (Trait vectors projected into PCA space before comparison)")
    print(sep)

    print(f"\n  ── TOP {n_top} most aligned with jailbreak direction ──")
    print(header)
    print(divider)
    for entry in pos_sorted[:n_top]:
        print(f"  {entry['rank_pos']:>4}  {entry['trait']:40s}  "
              f"{entry['cos_sim_pca']:>8.4f}  {entry['angle_deg']:>7.2f}°  "
              f"{entry['direction']}")

    print(f"\n  ── TOP {n_top} most aligned against jailbreak direction ──")
    print(header)
    print(divider)
    for entry in neg_sorted[:n_top]:
        print(f"  {entry['rank_neg']:>4}  {entry['trait']:40s}  "
              f"{entry['cos_sim_pca']:>8.4f}  {entry['angle_deg']:>7.2f}°  "
              f"{entry['direction']}")

    print(f"\n  ── TOP {n_top} by absolute cosine (most predictive) ──")
    print(header)
    print(divider)
    for entry in abs_sorted[:n_top]:
        print(f"  {entry['rank_abs']:>4}  {entry['trait']:40s}  "
              f"{entry['cos_sim_pca']:>8.4f}  {entry['angle_deg']:>7.2f}°  "
              f"{entry['direction']}")

    # Assistant axis
    axis_entry = next(r for r in results if r["trait"] == "assistant_axis")
    print(f"\n  ── Assistant Axis ──")
    print(f"  cos_sim (PCA space) : {axis_entry['cos_sim_pca']:.4f}")
    print(f"  angle               : {axis_entry['angle_deg']:.2f}°")
    print(f"  direction           : {axis_entry['direction']}")
    print(f"  rank (pos)          : {axis_entry['rank_pos']} of {len(results)}")
    print(f"  rank (abs)          : {axis_entry['rank_abs']} of {len(results)}")

    # Variance explained by top traits
    total_cos_sq = float(np.sum(cos_sims ** 2))
    print(f"\n  Sum of squared cosines (all traits): {total_cos_sq:.4f}")
    print(f"  This is the fraction of w_avg_pca explained by the trait vocabulary")
    print(f"  if traits were orthogonal (upper bound): {100*total_cos_sq:.1f}%")

    # Save stable w
    out_path = output_dir / f"stable_hyperplane_layer{layer}.pt"
    torch.save({
        "vector":          torch.from_numpy(w_avg_4096).float(),
        "vector_pca":      torch.from_numpy(w_avg_pca).float(),
        "layer":           layer,
        "n_seeds":         len(pca_ws_pca),
        "n_pca_components": n_pca_components,
        "var_explained":   var_explained,
        "stability_raw":   raw_stats,
        "stability_pca":   pca_stats,
        "mean_auc_raw":    float(np.mean(auc_raw_list)),
        "mean_auc_pca":    float(np.mean(auc_pca_list)),
        "description": (
            "Stable hyperplane normal: averaged over N seeds in PCA space, "
            "mapped back to 4096-dim. Positive direction = jailbreak success."
        ),
    }, out_path)
    print(f"\n  Saved stable hyperplane normal to {out_path.name}")

    return {
        "stability_raw":   raw_stats,
        "stability_pca":   pca_stats,
        "var_explained":   var_explained,
        "mean_auc_raw":    float(np.mean(auc_raw_list)),
        "mean_auc_pca":    float(np.mean(auc_pca_list)),
        "cosine_ranking":  results,
        "assistant_axis":  axis_entry,
        "sum_cos_sq":      total_cos_sq,
    }


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
        "--trait_vectors_dir", type=str,
        default="full_trait_output/traits40_vectors/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total",
    )
    parser.add_argument(
        "--axis_path", type=str,
        default="full_trait_output/traits40_axes/pre_generation_last_token/filter_matched_pairs_ge_50_count_ge_10_total/assistant_axis_pc1.pt",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="full_trait_output/harmbench_logreg",
    )
    parser.add_argument("--n_seeds",          type=int,   default=N_SEEDS)
    parser.add_argument("--n_pca_components", type=int,   default=N_PCA_COMPONENTS)
    parser.add_argument("--n_top",            type=int,   default=N_TOP)
    parser.add_argument("--min_success_rate", type=float, default=MIN_SUCCESS_RATE)
    parser.add_argument("--max_success_rate", type=float, default=MAX_SUCCESS_RATE)
    parser.add_argument("--train_frac",       type=float, default=TRAIN_FRAC)
    parser.add_argument("--layers", nargs="+", type=int,  default=LAYERS)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and filter ────────────────────────────────────────────────────────
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
          f"({n_jb} jailbroken, {len(rows_filtered)-n_jb} not)")

    all_results = {}

    for layer in args.layers:
        # Load trait vectors for this layer
        trait_names, trait_vectors = load_trait_vectors(
            Path(args.trait_vectors_dir), layer
        )
        axis_vector = load_axis_vector(Path(args.axis_path), layer)

        result = run_layer_analysis(
            layer=layer,
            rows_filtered=rows_filtered,
            activations=activations,
            trait_names=trait_names,
            trait_vectors=trait_vectors,
            axis_vector=axis_vector,
            n_seeds=args.n_seeds,
            n_pca_components=args.n_pca_components,
            train_frac=args.train_frac,
            n_top=args.n_top,
            output_dir=output_dir,
        )
        all_results[f"layer_{layer}"] = result

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = output_dir / "stable_hyperplane_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
