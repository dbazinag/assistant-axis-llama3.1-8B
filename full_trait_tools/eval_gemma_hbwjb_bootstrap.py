#!/usr/bin/env python3
"""Bootstrap 95% CIs for Table 2 (train HB+WJB, untuned logreg).

Classifier is deterministic (lbfgs, convex), so the uncertainty that matters is
which test pairs we happened to draw. We fit ONCE on HB+WJB, then resample the
test set with replacement (1000x) and recompute AUC/AP to get percentile CIs.
Config matches the report: LogReg(C=1.0, max_iter=2000), L2/lbfgs, no weighting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eval_gemma_cross_attack_transfer import load_trait_matrix
from eval_gemma_raw_vs_traits_transfer import load_raw_xy

BASE = "/mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B"
FO = f"{BASE}/full_trait_output"
VEC = f"{BASE}/gemma_trait_output/traits40_vectors_layer30_only/pre_generation_last_token/all_traits_no_filter"
N_BOOT = 1000
BOOT_SEED = 12345

SETS = {
    "HB":      (f"{FO}/harmbench_activations_gemma/activations.pt",     f"{FO}/harmbench_activations_gemma/classified_responses.jsonl"),
    "WJB":     (f"{FO}/wildjailbreak_activations_gemma/activations.pt", f"{FO}/wildjailbreak_activations_gemma/classified_responses.jsonl"),
    "GPTFuzz": (f"{FO}/gptfuzz_activations_gemma_hb/activations.pt",    f"{FO}/gptfuzz_activations_gemma_hb/classified_responses.jsonl"),
    "PAIR":    (f"{FO}/pair_activations_gemma_hb/activations.pt",       f"{FO}/pair_activations_gemma_hb/classified_responses.jsonl"),
    "PAP":     (f"{FO}/pap_activations_gemma_hb/activations.pt",        f"{FO}/pap_activations_gemma_hb/classified_responses.jsonl"),
    "PEZ":     (f"{FO}/pez_activations_gemma_hb/activations.pt",        f"{FO}/pez_activations_gemma_hb/classified_responses.jsonl"),
}
TEST_KEYS = ["GPTFuzz", "PAIR", "PAP", "PEZ"]


def new_pipe():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=42))])


def boot_ci(y, score, rng):
    n = len(y)
    aucs, aps = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], score[idx]
        if yb.min() == yb.max():       # degenerate resample (one class) -> skip
            continue
        aucs.append(roc_auc_score(yb, sb))
        aps.append(average_precision_score(yb, sb))
    f = lambda xs: (float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5)))
    return f(aucs), f(aps), len(aucs)


def main() -> None:
    T = load_trait_matrix(VEC)
    rng = np.random.default_rng(BOOT_SEED)
    print(f"trait matrix {T.shape} | MODEL C=1.0 unweighted | {N_BOOT} bootstraps seed={BOOT_SEED}", flush=True)

    Xy = {k: load_raw_xy(*SETS[k]) for k in SETS}
    X_tr = np.concatenate([Xy["HB"][0], Xy["WJB"][0]], 0)
    y_tr = np.concatenate([Xy["HB"][1], Xy["WJB"][1]], 0)
    P_tr = X_tr @ T.T
    print(f"TRAIN HB+WJB n={len(y_tr)} pos={int(y_tr.sum())} base={y_tr.mean():.3f}\n", flush=True)

    raw_pipe = new_pipe().fit(X_tr, y_tr)
    tr_pipe = new_pipe().fit(P_tr, y_tr)

    out = {}
    hdr = f"{'-> test':9s} {'n':>5s} {'base':>6s} | {'RAW AUC [95% CI]':>24s} {'RAW AP [95% CI]':>24s} | {'TRAIT AUC [95% CI]':>24s} {'TRAIT AP [95% CI]':>24s}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for tk in TEST_KEYS:
        X_te, y_te = Xy[tk]
        s_raw = raw_pipe.predict_proba(X_te)[:, 1]
        s_tr = tr_pipe.predict_proba(X_te @ T.T)[:, 1]
        pt = lambda y, s: (float(roc_auc_score(y, s)), float(average_precision_score(y, s)))
        raw_auc, raw_ap = pt(y_te, s_raw)
        tr_auc, tr_ap = pt(y_te, s_tr)
        # bootstrap CIs (fresh rng per feature set, same seed for comparability)
        (rac_lo_hi, rapci, nr) = boot_ci(y_te, s_raw, np.random.default_rng(BOOT_SEED))
        (tac_lo_hi, tapci, nt) = boot_ci(y_te, s_tr, np.random.default_rng(BOOT_SEED))
        out[tk] = {
            "test_n": int(len(y_te)), "test_pos": int(y_te.sum()), "base_rate": float(y_te.mean()),
            "raw5376":  {"auc": raw_auc, "auc_ci": rac_lo_hi, "ap": raw_ap, "ap_ci": rapci},
            "trait240": {"auc": tr_auc,  "auc_ci": tac_lo_hi, "ap": tr_ap,  "ap_ci": tapci},
        }
        fmt = lambda v, ci: f"{v:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"
        print(f"{tk:9s} {len(y_te):5d} {y_te.mean():6.3f} | "
              f"{fmt(raw_auc, rac_lo_hi):>24s} {fmt(raw_ap, rapci):>24s} | "
              f"{fmt(tr_auc, tac_lo_hi):>24s} {fmt(tr_ap, tapci):>24s}", flush=True)

    # ---- macro-average row: resample all 4 test sets jointly per iteration ----
    scored = {}
    for tk in TEST_KEYS:
        X_te, y_te = Xy[tk]
        scored[tk] = (y_te,
                      raw_pipe.predict_proba(X_te)[:, 1],
                      tr_pipe.predict_proba(X_te @ T.T)[:, 1])
    rng_avg = np.random.default_rng(BOOT_SEED)
    avg_raw_auc, avg_raw_ap, avg_tr_auc, avg_tr_ap = [], [], [], []
    for _ in range(N_BOOT):
        ra, rp, ta, tp = [], [], [], []
        for tk in TEST_KEYS:
            y_te, s_raw, s_tr = scored[tk]
            n = len(y_te); idx = rng_avg.integers(0, n, n)
            yb = y_te[idx]
            if yb.min() == yb.max():
                ra = None; break
            ra.append(roc_auc_score(yb, s_raw[idx])); rp.append(average_precision_score(yb, s_raw[idx]))
            ta.append(roc_auc_score(yb, s_tr[idx]));  tp.append(average_precision_score(yb, s_tr[idx]))
        if ra is None:
            continue
        avg_raw_auc.append(np.mean(ra)); avg_raw_ap.append(np.mean(rp))
        avg_tr_auc.append(np.mean(ta));  avg_tr_ap.append(np.mean(tp))
    ci = lambda xs: (float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5)))
    pt_avg = lambda fs, m: float(np.mean([out[tk][fs][m] for tk in TEST_KEYS]))
    out["AVERAGE"] = {
        "base_rate": float(np.mean([out[tk]["base_rate"] for tk in TEST_KEYS])),
        "raw5376":  {"auc": pt_avg("raw5376", "auc"),  "auc_ci": ci(avg_raw_auc),
                     "ap": pt_avg("raw5376", "ap"),    "ap_ci": ci(avg_raw_ap)},
        "trait240": {"auc": pt_avg("trait240", "auc"), "auc_ci": ci(avg_tr_auc),
                     "ap": pt_avg("trait240", "ap"),   "ap_ci": ci(avg_tr_ap)},
    }
    a = out["AVERAGE"]
    fmt = lambda v, c: f"{v:.3f} [{c[0]:.3f},{c[1]:.3f}]"
    print(f"{'AVERAGE':9s} {'':5s} {a['base_rate']:6.3f} | "
          f"{fmt(a['raw5376']['auc'], a['raw5376']['auc_ci']):>24s} {fmt(a['raw5376']['ap'], a['raw5376']['ap_ci']):>24s} | "
          f"{fmt(a['trait240']['auc'], a['trait240']['auc_ci']):>24s} {fmt(a['trait240']['ap'], a['trait240']['ap_ci']):>24s}", flush=True)

    Path(f"{FO}/attack_transfer_hbwjb_bootstrap_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {FO}/attack_transfer_hbwjb_bootstrap_results.json", flush=True)


if __name__ == "__main__":
    main()
