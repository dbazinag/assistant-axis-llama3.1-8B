"""Time the marginal DETECTION-HEAD latency of trait vs raw (no model forward pass).

Both heads start from the same already-extracted 4096-dim layer activation (a free
byproduct of the production forward pass). We time only the work added on top:
  raw   : logreg on the 4096-dim activation
  trait : project 4096 -> 240 (matmul w/ trait matrix), then logreg on 240-dim
StandardScaler is folded into each fitted Pipeline (a per-feature subtract+divide).
"""
import sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "full_trait_tools")
from run_all_traits_sweep_v2 import (
    load_jsonl, load_activations, build_activation_matrix,
    load_trait_matrix, project_all_traits,
)

LAYER = 16
HUMAN_CLS = "full_trait_output/harmbench_activations_olmo3/classified_responses.jsonl"
HUMAN_ACT = "full_trait_output/harmbench_activations_olmo3/activations.pt"
TRAIT_DIR = "full_trait_output/traits40_vectors_olmo3_7b/pre_generation_last_token/all_traits_no_filter"


def mk():
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                   max_iter=4000, random_state=0)),
    ])


def bench(fn, X, n_warm, n_rep):
    n = len(X)
    for i in range(n_warm):
        fn(X[i % n])
    t0 = time.perf_counter()
    for i in range(n_rep):
        fn(X[i % n])
    return (time.perf_counter() - t0) / n_rep * 1e6  # microseconds / sample


def bench_batched(fn, X, n_rep):
    fn(X)  # warm
    t0 = time.perf_counter()
    for _ in range(n_rep):
        fn(X)
    return (time.perf_counter() - t0) / n_rep / len(X) * 1e6  # us / sample (amortized)


def main():
    import os
    print(f"threads: OMP={os.environ.get('OMP_NUM_THREADS','?')} "
          f"MKL={os.environ.get('MKL_NUM_THREADS','?')}", flush=True)
    rows = load_jsonl(Path(HUMAN_CLS))
    acts = load_activations(Path(HUMAN_ACT))
    X_raw, y, _ = build_activation_matrix(rows, acts, LAYER)
    X_raw = np.ascontiguousarray(X_raw, dtype=np.float32)
    tm, names = load_trait_matrix(LAYER, vectors_dir=Path(TRAIT_DIR),
                                  cache_dir=Path("full_trait_output"))
    tmT = np.ascontiguousarray(tm.T, dtype=np.float32)  # (4096, 240)
    X_trait = project_all_traits(X_raw, tm)
    print(f"N={len(X_raw)}  raw_dim={X_raw.shape[1]}  trait_dim={X_trait.shape[1]}", flush=True)

    raw_m = mk().fit(X_raw, y)
    trait_m = mk().fit(X_trait, y)

    # Extract fitted params for the DEPLOYABLE pure-numpy head (standardize + dot):
    def params(m):
        sc, clf = m.named_steps["sc"], m.named_steps["clf"]
        return (sc.mean_.astype(np.float32), sc.scale_.astype(np.float32),
                clf.coef_[0].astype(np.float32), np.float32(clf.intercept_[0]))
    rmu, rsd, rw, rb = params(raw_m)
    tmu, tsd, tw, tb = params(trait_m)

    # ── deployable heads, both fed the SAME raw 4096-dim activation ──
    raw_np   = lambda x: float(((x - rmu) / rsd) @ rw + rb)
    trait_np = lambda x: float((((x @ tmT) - tmu) / tsd) @ tw + tb)  # project, then logit
    proj_np  = lambda x: x @ tmT  # projection cost alone
    raw_batch_np   = lambda X: ((X - rmu) / rsd) @ rw + rb
    trait_batch_np = lambda X: (((X @ tmT) - tmu) / tsd) @ tw + tb
    # sklearn-call variants (per-request framework overhead you'd pay if you ship sklearn)
    raw_sk   = lambda x: raw_m.decision_function(x.reshape(1, -1))
    trait_sk = lambda x: trait_m.decision_function((x @ tmT).reshape(1, -1))

    print("\n=== DEPLOYABLE head, per-request (batch=1, raw numpy standardize+dot) ===", flush=True)
    for name, fn in [("raw", raw_np), ("trait", trait_np), ("  trait-proj only", proj_np)]:
        us = bench(fn, X_raw, n_warm=500, n_rep=20000)
        print(f"  {name:18s} {us:8.3f} us/sample", flush=True)

    print("\n=== DEPLOYABLE head, amortized (batched over full set) ===", flush=True)
    for name, fn in [("raw", raw_batch_np), ("trait", trait_batch_np)]:
        us = bench_batched(fn, X_raw, n_rep=500)
        print(f"  {name:18s} {us:8.4f} us/sample", flush=True)

    print("\n=== sklearn Pipeline.decision_function, per-request (incl. framework overhead) ===", flush=True)
    for name, fn in [("raw", raw_sk), ("trait", trait_sk)]:
        us = bench(fn, X_raw, n_warm=200, n_rep=5000)
        print(f"  {name:18s} {us:8.2f} us/sample", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
