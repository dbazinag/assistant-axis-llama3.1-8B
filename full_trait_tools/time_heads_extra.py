"""Time the marginal DETECTION-HEAD latency of the remaining Class-1 (free-activation)
methods on OLMo-3-7B activations, to sit alongside raw/trait in time_detection_heads.py.

All five consume an activation that is a free byproduct of the production forward pass;
we time ONLY the head op (project / kernel / tree / Mahalanobis / direction-dot):

  pca3      : layer-16 (4096) -> PCA(3) -> logreg                       [sweep feature_type=pca]
  jbshield  : layer-16 (4096) -> StandardScaler -> dot(unit direction)  [MeanDirectionDetector]
  jlt_svm   : layers[18,25,32]x5tok (61440) -> PCA(128) -> SVC-RBF
  jlt_rf    : layers[18,25,32]x5tok (61440) -> PCA(128) -> RandomForest
  mcd       : layers[18,25,32]   (12288) -> Std -> PCA(128) -> Mahalanobis(failed)-Mahalanobis(success)

Each method's real fitted estimator is timed (imported from the baseline scripts, not
reimplemented). Fitting is untimed; we time per-request (batch=1) and batched throughput.
"""
import os, sys, time
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "full_trait_tools")
from run_all_traits_sweep_v2 import load_jsonl, load_activations, build_activation_matrix
from run_baselines_jbshield_fjd import MeanDirectionDetector
import run_baseline_jlt_hidden_tensor as JLT
import run_baseline_contrastive_geometry as MCDmod

LAYER16_CLS = "full_trait_output/harmbench_activations_olmo3/classified_responses.jsonl"
LAYER16_ACT = "full_trait_output/harmbench_activations_olmo3/activations.pt"
RTV_ROOT = Path("full_trait_output/rtv_activations_olmo3")
LAYER = 16
SEED = 0


def bench1(fn, X, n_warm=300, n_rep=10000):
    n = len(X)
    for i in range(n_warm):
        fn(X[i % n])
    t0 = time.perf_counter()
    for i in range(n_rep):
        fn(X[i % n])
    return (time.perf_counter() - t0) / n_rep * 1e6  # us/sample


def benchN(fn, X, n_rep=200):
    fn(X)  # warm
    t0 = time.perf_counter()
    for _ in range(n_rep):
        fn(X)
    return (time.perf_counter() - t0) / n_rep / len(X) * 1e6  # us/sample amortized


def report(name, us1, usN):
    print(f"  {name:12s} batch1 {us1:10.3f} us | batched {usN:10.4f} us/sample", flush=True)


def main():
    print(f"threads OMP={os.environ.get('OMP_NUM_THREADS','?')} "
          f"MKL={os.environ.get('MKL_NUM_THREADS','?')}", flush=True)

    # ---- layer-16 features (jbshield, pca3) ----
    rows = load_jsonl(Path(LAYER16_CLS))
    acts = load_activations(Path(LAYER16_ACT))
    X16, y16, _ = build_activation_matrix(rows, acts, LAYER)
    X16 = np.ascontiguousarray(X16, dtype=np.float32)
    print(f"layer16: N={len(X16)} dim={X16.shape[1]} jb={y16.mean():.3f}", flush=True)

    # ---- rtv multi-layer features (jlt 61440, mcd 12288) ----
    X_jlt, jlt_rows = JLT.load_dataset_tensor(RTV_ROOT, "harmbench")
    y_jlt = JLT.labels_for_rows(jlt_rows)
    X_jlt = np.ascontiguousarray(X_jlt, dtype=np.float32)
    X_mcd, mcd_rows = MCDmod.load_dataset_matrix(RTV_ROOT, "harmbench")
    y_mcd = MCDmod.labels_for_rows(mcd_rows)
    X_mcd = np.ascontiguousarray(X_mcd, dtype=np.float32)
    print(f"jlt:  N={len(X_jlt)} dim={X_jlt.shape[1]}", flush=True)
    print(f"mcd:  N={len(X_mcd)} dim={X_mcd.shape[1]}", flush=True)

    # ===== fit each head (UNTIMED) =====
    # jbshield: scaler + unit mean-difference direction
    jb = MeanDirectionDetector().fit(X16, y16)

    # pca3: PCA(3) on raw, then StandardScaler+LogReg (matches sweep feature_type=pca)
    pca3 = PCA(n_components=3, random_state=SEED).fit(X16)
    pca3_clf = Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=4000, random_state=SEED))])
    pca3_clf.fit(pca3.transform(X16), y16)

    # jlt: PCA(128) on raw 61440, then SVC-RBF and RandomForest (real make_models)
    n_comp_j = min(JLT.N_COMPONENTS, X_jlt.shape[0] - 1, X_jlt.shape[1])
    pca_j = PCA(n_components=n_comp_j, svd_solver="randomized", random_state=SEED)
    Xjp = pca_j.fit_transform(X_jlt)
    models = JLT.make_models(SEED, fast=False)
    svm = models["pca_svm_rbf"]; svm.fit(Xjp, y_jlt)
    rf = models["pca_random_forest"]; rf.fit(Xjp, y_jlt)
    rf.n_jobs = 1  # single-request latency: avoid n_jobs=-1 threadpool spin-up per batch-1 call

    # mcd: StandardScaler -> PCA(128) -> MCD (real class)
    n_comp_m = min(MCDmod.N_COMPONENTS, X_mcd.shape[0] - 1, X_mcd.shape[1])
    sc_m = StandardScaler().fit(X_mcd)
    pca_m = PCA(n_components=n_comp_m, svd_solver="randomized", random_state=SEED)
    Xmp = pca_m.fit_transform(sc_m.transform(X_mcd))
    mcd = MCDmod.MCD().fit(Xmp, y_mcd)

    # ===== deployable per-request callables (fed the raw activation for that method) =====
    jb_fn   = lambda x: float(jb.score(x.reshape(1, -1))[0])
    pca3_fn = lambda x: float(pca3_clf.decision_function(pca3.transform(x.reshape(1, -1)))[0])
    svm_fn  = lambda x: float(svm.decision_function(pca_j.transform(x.reshape(1, -1)))[0])
    rf_fn   = lambda x: float(rf.predict_proba(pca_j.transform(x.reshape(1, -1)))[0, 1])
    mcd_fn  = lambda x: float(mcd.score(pca_m.transform(sc_m.transform(x.reshape(1, -1))))[0])

    jb_b   = lambda X: jb.score(X)
    pca3_b = lambda X: pca3_clf.decision_function(pca3.transform(X))
    svm_b  = lambda X: svm.decision_function(pca_j.transform(X))
    rf_b   = lambda X: rf.predict_proba(pca_j.transform(X))[:, 1]
    mcd_b  = lambda X: mcd.score(pca_m.transform(sc_m.transform(X)))

    print("\n=== Class-1 extra heads (OLMo activations, CPU) ===", flush=True)
    report("jbshield", bench1(jb_fn, X16),  benchN(jb_b, X16))
    report("pca3",     bench1(pca3_fn, X16), benchN(pca3_b, X16))
    report("jlt_svm",  bench1(svm_fn, X_jlt), benchN(svm_b, X_jlt))
    report("jlt_rf",   bench1(rf_fn, X_jlt),  benchN(rf_b, X_jlt))
    report("mcd",      bench1(mcd_fn, X_mcd), benchN(mcd_b, X_mcd))
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
