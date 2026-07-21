#!/usr/bin/env python3
"""
save_best_llama_clf.py

Refit and persist the best Llama-3.1-8B trait-detection classifier
(logreg_l2_C10.0_raw, the AvgXfer~0.76 model from all_traits_sweep_v2) and
extract per-trait importance for the report.

The sweep script (run_all_traits_sweep_v2.py) only ever wrote metrics to
results.json — it never saved fitted coefficients. This script:

  1. Reproduces the sweep's 50-seed transfer protocol for THIS one config
     (importing the sweep's own helpers, so the logic is byte-identical) and
     prints the mean AUCs to confirm they match the logged numbers.
  2. Refits the pipeline on ALL HarmBench human_jailbreak data, calibrates a
     threshold+sign on a held-out val split, and saves the artefact.
  3. Extracts the logistic-regression weight vector (in standardized feature
     space) and ranks the 229 traits by it, so the report can name the traits
     most aligned with the jailbreak vs safe direction.

Artefact (pickle) fields mirror save_best_olmo3_clf.py, plus:
  - 'coef_std':         (n_traits,) logreg weights in standardized feature space
  - 'coef_proj':        (n_traits,) weights mapped back to raw projection space
  - 'intercept':        float
  - 'trait_importance': list[dict] sorted by |coef_std| desc
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

# Allow importing the sweep module's helpers when run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Reuse the sweep's exact data-loading / split / scoring helpers.
from run_all_traits_sweep_v2 import (  # type: ignore
    load_jsonl,
    load_activations,
    load_trait_matrix,
    build_activation_matrix,
    project_all_traits,
    get_pool_split,
    split_by_pool,
    best_threshold,
    get_score,
    eval_set,
    safe_auc,
)

LAYER = 16
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
N_SEEDS = 50
RANDOM_SEED = 42
MODEL_NAME = "logreg_l2_C10.0_raw"


def build_pipeline(seed: int) -> Pipeline:
    """Exactly logreg_l2(C=10.0, class_weight=None) from run_all_traits_sweep_v2."""
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(
            C=10.0, penalty="l2", solver="lbfgs",
            max_iter=4000, class_weight=None, random_state=seed,
        )),
    ])


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
    parser.add_argument("--output_path",
                        default="full_trait_output/all_traits_sweep_v2/best_model.pkl")
    parser.add_argument("--importance_csv",
                        default="full_trait_output/all_traits_sweep_v2/trait_importance.csv")
    parser.add_argument("--n_seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    print("=== Saving best Llama trait classifier (logreg_l2_C10.0_raw) ===", flush=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1] Loading data ...", flush=True)
    human_rows = [
        r for r in load_jsonl(Path(args.human_classified_path))
        if r.get("attack_type") == "human_jailbreak"
    ]
    human_acts = load_activations(Path(args.human_activations_path))
    print(f"  HarmBench human_jailbreak rows: {len(human_rows)}", flush=True)

    trait_matrix, trait_names = load_trait_matrix(LAYER)  # default Llama paths
    print(f"  Trait matrix: {trait_matrix.shape}  ({len(trait_names)} names)", flush=True)

    transfer_inputs = [
        ("GCG",     args.gcg_classified_path,     args.gcg_activations_path),
        ("PAIR",    args.pair_classified_path,    args.pair_activations_path),
        ("PAP",     args.pap_classified_path,     args.pap_activations_path),
        ("GPTFuzz", args.gptfuzz_classified_path, args.gptfuzz_activations_path),
        ("PEZ",     args.pez_classified_path,     args.pez_activations_path),
    ]

    # ── Build features ───────────────────────────────────────────────────────
    print("\n[2] Building trait-projected features ...", flush=True)
    x_raw_h, y_h, human_valid = build_activation_matrix(human_rows, human_acts, LAYER)
    x_h = project_all_traits(x_raw_h, trait_matrix)
    print(f"  HarmBench: X={x_h.shape}, jb_rate={y_h.mean():.3f}", flush=True)

    transfer: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, rp, ap in transfer_inputs:
        if Path(rp).exists() and Path(ap).exists():
            rows_ = load_jsonl(Path(rp))
            acts_ = load_activations(Path(ap))
            xr, y_, _ = build_activation_matrix(rows_, acts_, LAYER)
            if len(xr) > 0:
                transfer[name] = (project_all_traits(xr, trait_matrix), y_)
                print(f"  {name}: {len(xr)} rows, jb={y_.mean():.3f}", flush=True)
    families = list(transfer.keys())

    # ── Step 1: reproduce the 50-seed protocol for this config ────────────────
    print(f"\n[3] Reproducing {args.n_seeds}-seed transfer protocol ...", flush=True)
    val_aucs, human_aucs = [], []
    fam_aucs: dict[str, list[float]] = {f: [] for f in families}
    for seed in range(args.n_seeds):
        train_beh, train_tpl, test_beh, test_tpl = get_pool_split(
            human_valid, TRAIN_FRAC, seed
        )
        train_idx, test_idx = split_by_pool(
            human_valid, train_beh, train_tpl, test_beh, test_tpl
        )
        if not train_idx or not test_idx or len(np.unique(y_h[train_idx])) < 2:
            continue
        tr_idx, vl_idx = train_test_split(
            np.array(train_idx), test_size=VAL_FRAC,
            random_state=seed, stratify=y_h[train_idx],
        )
        m_inner = build_pipeline(seed)
        m_inner.fit(x_h[tr_idx], y_h[tr_idx])
        val_aucs.append(safe_auc(y_h[vl_idx], get_score(m_inner, x_h[vl_idx])))

        m_full = build_pipeline(seed)
        m_full.fit(x_h[train_idx], y_h[train_idx])
        thr, sgn = best_threshold(y_h[vl_idx], get_score(m_full, x_h[vl_idx]))
        human_aucs.append(
            eval_set(y_h[test_idx], get_score(m_full, x_h[test_idx]), thr, sgn)["auc"]
        )
        for name, (x_t, y_t) in transfer.items():
            fam_aucs[name].append(
                eval_set(y_t, get_score(m_full, x_t), thr, sgn)["auc"]
            )

    repro = {
        "val_auc":    float(np.nanmean(val_aucs)),
        "human_test": float(np.nanmean(human_aucs)),
        **{f: float(np.nanmean(fam_aucs[f])) for f in families},
    }
    repro["avg_xfer"] = float(np.nanmean([repro[f] for f in families]))
    print(f"  val_auc   = {repro['val_auc']:.4f}", flush=True)
    print(f"  human_test= {repro['human_test']:.4f}", flush=True)
    for f in families:
        print(f"  {f:8s}  = {repro[f]:.4f}", flush=True)
    print(f"  AvgXfer   = {repro['avg_xfer']:.4f}   "
          f"(sweep logged: AvgXfer=0.7596, test=0.8538, val=0.9665)", flush=True)

    # ── Step 2: calibrate threshold, refit on all data ────────────────────────
    print("\n[4] Calibrating threshold + refitting on full dataset ...", flush=True)
    tr_idx, vl_idx = train_test_split(
        np.arange(len(y_h)), test_size=VAL_FRAC,
        random_state=RANDOM_SEED, stratify=y_h,
    )
    pipe = build_pipeline(RANDOM_SEED)
    pipe.fit(x_h[tr_idx], y_h[tr_idx])
    val_score = get_score(pipe, x_h[vl_idx])
    threshold, sign = best_threshold(y_h[vl_idx], val_score)
    val_auc = float(roc_auc_score(y_h[vl_idx], val_score))
    val_pred = (sign * val_score >= sign * threshold).astype(int)
    val_bacc = float(balanced_accuracy_score(y_h[vl_idx], val_pred))
    print(f"  calib val AUC={val_auc:.4f} bacc={val_bacc:.4f} thr={threshold:.4f} sign={sign}", flush=True)

    pipe.fit(x_h, y_h)  # final fit on everything

    # ── Step 3: trait importance from logreg weights ──────────────────────────
    print("\n[5] Extracting trait importance ...", flush=True)
    scaler: StandardScaler = pipe.named_steps["sc"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    coef_std = clf.coef_.ravel().astype(np.float64)          # standardized space
    coef_proj = coef_std / scaler.scale_                     # raw projection space
    intercept = float(clf.intercept_[0])

    order = np.argsort(-np.abs(coef_std))
    trait_importance = [
        {
            "rank":      int(i + 1),
            "trait":     trait_names[j],
            "coef_std":  float(coef_std[j]),
            "abs_std":   float(abs(coef_std[j])),
            "coef_proj": float(coef_proj[j]),
            "direction": "jailbroken" if coef_std[j] > 0 else "safe",
        }
        for i, j in enumerate(order)
    ]

    print("  Top 15 traits by |weight| (standardized space):", flush=True)
    for d in trait_importance[:15]:
        print(f"    {d['rank']:3d}. {d['trait']:40s} "
              f"coef={d['coef_std']:+.4f}  → {d['direction']}", flush=True)

    # ── Save artefact + CSV ───────────────────────────────────────────────────
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artefact = {
        "pipeline":     pipe,
        "trait_matrix": trait_matrix,
        "trait_names":  trait_names,
        "layer":        LAYER,
        "threshold":    threshold,
        "sign":         sign,
        "coef_std":     coef_std.astype(np.float32),
        "coef_proj":    coef_proj.astype(np.float32),
        "intercept":    intercept,
        "trait_importance": trait_importance,
        "meta": {
            "model_name":     MODEL_NAME,
            "n_train":        int(len(y_h)),
            "jb_rate_train":  float(y_h.mean()),
            "calib_val_auc":  val_auc,
            "calib_val_bacc": val_bacc,
            "n_seeds_repro":  args.n_seeds,
            "reproduced":     repro,
            "sweep_logged":   {"avg_xfer": 0.7596, "human_test": 0.8538, "val_auc": 0.9665},
            "human_classified_path": args.human_classified_path,
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(artefact, f, protocol=4)
    print(f"\nSaved artefact → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    csv_path = Path(args.importance_csv)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "trait", "coef_std", "abs_std", "coef_proj", "direction"])
        w.writeheader()
        w.writerows(trait_importance)
    print(f"Saved trait importance → {csv_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
