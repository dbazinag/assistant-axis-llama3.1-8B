#!/usr/bin/env python3
"""
Pairwise p-value table: traits (all_traits) vs every other method, per attack.

Reference = all_traits model `linsvm_C0.3_bal` (the row used in the final report),
per-seed AUC arrays from the all_traits_sweep_v2 sweep.

Test:
  - Seeded baselines (have per-seed AUC arrays): one-tailed Welch's t-test,
    H0: AUC_traits == AUC_base, H1: AUC_traits > AUC_base   (ttest_ind, equal_var=False).
  - Scalar baselines (single point AUC, no per-seed data): one-tailed one-sample
    t-test of the traits per-seed array against the baseline's fixed AUC
    (ttest_1samp, popmean=base_auc, alternative='greater').

Sign-correction: every method's AUC is oriented by its mean direction
(if mean < 0.5, the whole per-seed array / point is flipped to 1-auc), matching
the sign-corrected AUCs reported in the paper.

Multiple comparisons: Benjamini-Hochberg FDR across all tested cells.
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import ttest_ind, ttest_1samp

OUT = Path("/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output")
ATTACKS = ["HarmBench", "GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]

seeded = {}  # (label, attack) -> per-seed AUC array (sign-corrected)
point = {}   # (label, attack) -> scalar AUC (sign-corrected)


def orient(arr):
    """Sign-correct by mean direction: flip if anti-correlated (mean < 0.5)."""
    a = np.asarray(arr, dtype=float)
    return 1.0 - a if a.mean() < 0.5 else a


def add_seeded(label, attack, auc_dict):
    all_vals = auc_dict.get("all", [])
    if len(all_vals) > 1:
        seeded[(label, attack)] = orient(all_vals)
    else:
        point[(label, attack)] = float(orient([auc_dict.get("mean", np.nan)])[0])


def add_scalar(label, attack, auc_val):
    point[(label, attack)] = float(orient([auc_val])[0])


def try_load(fn):
    try:
        fn()
    except (FileNotFoundError, KeyError) as e:
        print(f"  [skip] {fn.__name__}: {e}")


# ── reference: traits = linsvm_C0.3_bal ─────────────────────────────────────
REF = "all_traits"
_traits = json.load(open(OUT / "all_traits_sweep_v2/results.json"))["summary"]["linsvm_C0.3_bal"]
TRAITS = {}
for atk in ATTACKS:
    key = "human_test" if atk == "HarmBench" else atk
    TRAITS[atk] = orient(_traits[key]["auc"]["all"])

# ── other "our" feature variants: raw, pca, top_traits ──────────────────────
def load_transfer(path, key_to_attack):
    d = json.load(open(path))
    for mode, mode_data in d["modes"].items():
        if mode == "all_traits":
            continue  # that's the reference family; we use the sweep model instead
        for key, attack in key_to_attack.items():
            if key in mode_data:
                add_seeded(mode, attack, mode_data[key])

try_load(lambda: load_transfer(OUT / "transfer_results_gcg_pair/transfer_results.json",
                               {"transfer_GCG": "GCG", "transfer_PAIR": "PAIR"}))
try_load(lambda: load_transfer(OUT / "transfer_results_pap/transfer_results.json",
                               {"transfer_PAP": "PAP"}))
try_load(lambda: load_transfer(OUT / "transfer_results_gptfuzz/transfer_results.json",
                               {"transfer_GPTFuzz": "GPTFuzz"}))
try_load(lambda: load_transfer(OUT / "transfer_results_pez/transfer_results.json",
                               {"transfer_PEZ": "PEZ"}))
try_load(lambda: load_transfer(OUT / "transfer_results_all/transfer_results_all.json",
                               {"human_test": "HarmBench"}))

# ── seeded baselines ────────────────────────────────────────────────────────
def _jbshield():
    d = json.load(open(OUT / "baselines_jbshield_fjd/jbshield_fjd_results.json"))
    for a in ATTACKS:
        add_seeded("jbshield", a, d["results"]["jbshield_mean_direction"]["datasets"][a]["auc"])
try_load(_jbshield)

def _contrastive():
    d = json.load(open(OUT / "baselines_contrastive_geometry/contrastive_geometry_results.json"))
    for a in ATTACKS:
        add_seeded("contrastive_mcd", a, d["methods"]["mcd"]["datasets"][a]["auc"])
        add_seeded("contrastive_kcd", a, d["methods"]["kcd"]["datasets"][a]["auc"])
try_load(_contrastive)

def _ppl():
    d = json.load(open(OUT / "baselines_harmbench_ppl_smoothllm/harmbench_ppl_smoothllm_results.json"))
    for a in ATTACKS:
        add_seeded("perplexity",    a, d["results"]["perplexity_nll"]["datasets"][a]["auc"])
        add_seeded("smoothllm",     a, d["results"]["smoothllm_refusal_majority"]["datasets"][a]["auc"])
        add_seeded("harmbench_cls", a, d["results"]["harmbench_classifier"]["datasets"][a]["auc"])
try_load(_ppl)

def _fjd():
    d = json.load(open(OUT / "baselines_fjd_paper/fjd_paper_results.json"))
    for a in ATTACKS:
        add_seeded("fjd", a, d["results"]["fjd_paper_first_token_confidence"]["datasets"][a]["auc"])
try_load(_fjd)

def _maha():
    d = json.load(open(OUT / "baselines_rtv_mahalanobis/rtv_mahalanobis_results.json"))
    for a in ATTACKS:
        add_seeded("mahalanobis", a, d["datasets"][a]["auc"])
try_load(_maha)

def _jlt():
    d = json.load(open(OUT / "baselines_jlt_hidden_tensor/jlt_hidden_tensor_results.json"))
    for a in ATTACKS:
        add_seeded("jlt_svm",    a, d["models"]["pca_svm_rbf"]["datasets"][a]["auc"])
        add_seeded("jlt_rf",     a, d["models"]["pca_random_forest"]["datasets"][a]["auc"])
        add_seeded("jlt_logreg", a, d["models"]["pca_logreg"]["datasets"][a]["auc"])
try_load(_jlt)

# ── scalar baselines (no per-seed data) ─────────────────────────────────────
def _llama_guard():
    d = json.load(open(OUT / "baselines_all_attacks/llama_guard_all_attacks.json"))
    for a in ATTACKS:
        add_scalar("llama_guard_input", a, d[a]["input_only_auc"])
        add_scalar("llama_guard_i/o",   a, d[a]["input_output_auc"])
try_load(_llama_guard)

def _self_exam():
    d = json.load(open(OUT / "baselines_all_attacks/self_exam_all_attacks.json"))
    for a in ATTACKS:
        add_scalar("verbalized_direct", a, d[a]["direct_auc"])
        add_scalar("verbalized_cot",    a, d[a]["cot_auc"])
try_load(_self_exam)

def _gradsafe():
    d = json.load(open(OUT / "baselines_all_attacks/gradsafe_all_attacks.json"))
    for a in ATTACKS:
        add_scalar("gradsafe", a, d["results"][a]["auc"])
try_load(_gradsafe)

def _wildguard():
    d = json.load(open(OUT / "baselines_wildguard/wildguard_results.json"))
    for a in ATTACKS:
        add_scalar("wildguard_harm",   a, d[a]["response_harmfulness"]["auc"])
        add_scalar("wildguard_comply", a, d[a]["response_compliance"]["auc"])
try_load(_wildguard)

# ── tests ───────────────────────────────────────────────────────────────────
# raw[(label, attack)] = (base_auc, delta, p_raw, test_type)
raw = {}
for (label, atk), arr in seeded.items():
    _, p = ttest_ind(TRAITS[atk], arr, equal_var=False, alternative="greater")
    raw[(label, atk)] = (float(arr.mean()), float(TRAITS[atk].mean() - arr.mean()), float(p), "welch")
for (label, atk), val in point.items():
    _, p = ttest_1samp(TRAITS[atk], popmean=val, alternative="greater")
    raw[(label, atk)] = (float(val), float(TRAITS[atk].mean() - val), float(p), "1samp")

# ── BH FDR across all tested cells ──────────────────────────────────────────
keys = sorted(raw.keys())
pv = np.array([raw[k][2] for k in keys])
n = len(pv)
order = np.argsort(pv)
ranks = np.empty(n, dtype=int)
ranks[order] = np.arange(1, n + 1)
adj = np.minimum(1.0, pv * n / ranks)
for i in range(n - 2, -1, -1):
    adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])
padj = {k: float(adj[i]) for i, k in enumerate(keys)}


def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

# ── assemble + write CSV ────────────────────────────────────────────────────
METHODS = ["raw", "pca", "top_traits",
           "jbshield", "contrastive_mcd", "contrastive_kcd",
           "perplexity", "smoothllm", "harmbench_cls", "fjd", "mahalanobis",
           "jlt_svm", "jlt_rf", "jlt_logreg",
           "llama_guard_input", "llama_guard_i/o",
           "verbalized_direct", "verbalized_cot", "gradsafe",
           "wildguard_harm", "wildguard_comply"]

cols = ["method", "test"]
for a in ATTACKS:
    cols += [f"{a}_auc", f"{a}_dAUC", f"{a}_p_adj", f"{a}_sig"]

lines = [",".join(cols)]
# reference row
ref_cells = []
for a in ATTACKS:
    ref_cells += [f"{TRAITS[a].mean():.4f}", "0.0", "", ""]
lines.append(",".join([f"all_traits(linsvm_C0.3_bal)", "reference"] + ref_cells))

for m in METHODS:
    test_types = {raw[(m, a)][3] for a in ATTACKS if (m, a) in raw}
    tt = "/".join(sorted(test_types)) if test_types else "n/a"
    cells = []
    for a in ATTACKS:
        if (m, a) in raw:
            base, d, _, _ = raw[(m, a)]
            cells += [f"{base:.4f}", f"{d:+.4f}", f"{padj[(m,a)]:.3e}", sig(padj[(m, a)])]
        else:
            cells += ["", "", "", ""]
    lines.append(",".join([m, tt] + cells))

csv_text = "\n".join(lines) + "\n"
out_csv = OUT / "pairwise_pvalue_table_traits.csv"
open(out_csv, "w").write(csv_text)

# ── console pretty-print ────────────────────────────────────────────────────
print("\nReference: all_traits = linsvm_C0.3_bal (50 seeds). One-tailed test H1: traits > method.")
print("Welch two-sample for seeded methods; one-sample t (traits vs constant) for scalar baselines.")
print("AUCs sign-corrected by mean direction. p_adj = BH-FDR.\n")
w = 20
print(f"{'method':<26}{'test':<7}" + "".join(f"{a:>{w}}" for a in ATTACKS))
for m in ["all_traits(linsvm_C0.3_bal)"] + METHODS:
    if m.startswith("all_traits"):
        row = f"{m:<26}{'ref':<7}" + "".join(f"{TRAITS[a].mean():>{w}.3f}" for a in ATTACKS)
        print(row); continue
    test_types = {raw[(m, a)][3] for a in ATTACKS if (m, a) in raw}
    tt = "/".join(sorted(test_types)) if test_types else "-"
    row = f"{m:<26}{tt:<7}"
    for a in ATTACKS:
        if (m, a) in raw:
            base, d, _, _ = raw[(m, a)]
            row += f"{f'{base:.2f} {d:+.2f}{sig(padj[(m,a)])}':>{w}}"
        else:
            row += f"{'—':>{w}}"
    print(row)
print(f"\nSaved CSV -> {out_csv}")
