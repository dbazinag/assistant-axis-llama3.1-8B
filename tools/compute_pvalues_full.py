#!/usr/bin/env python3
"""
Comprehensive p-value table for all methods x all attacks.
One-sample one-tailed t-test: H0: AUC=0.5, H1: AUC>0.5, per-seed arrays.
Benjamini-Hochberg FDR correction across all cells with per-seed data.
Methods without per-seed arrays get AUC printed but no p-value.
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import ttest_1samp

OUT = Path("/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output")
ATTACKS = ["HarmBench", "GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]

# cells[(method_label, attack)] = list of per-seed AUCs  (for t-test)
# point[(method_label, attack)] = scalar AUC             (no t-test available)
cells = {}
point = {}

# ── helpers ────────────────────────────────────────────────────────────────

def add_seeded(label, attack, auc_dict):
    """auc_dict is {'mean':..,'std':..,'all':[...]}"""
    all_vals = auc_dict.get("all", [])
    if len(all_vals) > 1:
        cells[(label, attack)] = all_vals
    else:
        point[(label, attack)] = auc_dict.get("mean", float("nan"))

def add_scalar(label, attack, auc_val):
    point[(label, attack)] = float(auc_val)

def load_transfer(path, key_to_attack):
    """Load from transfer_results files (modes->mode->key->{'mean','all'})"""
    d = json.load(open(path))
    for mode, mode_data in d["modes"].items():
        for key, attack in key_to_attack.items():
            if key in mode_data:
                add_seeded(mode, attack, mode_data[key])

# ── 1. our methods: all_traits, top_traits, raw, pca ───────────────────────

load_transfer(OUT / "transfer_results_gcg_pair/transfer_results.json",
              {"transfer_GCG": "GCG", "transfer_PAIR": "PAIR"})
load_transfer(OUT / "transfer_results_pap/transfer_results.json",
              {"transfer_PAP": "PAP"})
load_transfer(OUT / "transfer_results_gptfuzz/transfer_results.json",
              {"transfer_GPTFuzz": "GPTFuzz"})
load_transfer(OUT / "transfer_results_pez/transfer_results.json",
              {"transfer_PEZ": "PEZ"})
load_transfer(OUT / "transfer_results_all/transfer_results_all.json",
              {"human_test": "HarmBench"})

# ── 2. JBShield ─────────────────────────────────────────────────────────────

d = json.load(open(OUT / "baselines_jbshield_fjd/jbshield_fjd_results.json"))
for atk in ATTACKS:
    add_seeded("jbshield", atk,
               d["results"]["jbshield_mean_direction"]["datasets"][atk]["auc"])

# ── 3. Contrastive Geometry (MCD / KCD) ─────────────────────────────────────

d = json.load(open(OUT / "baselines_contrastive_geometry/contrastive_geometry_results.json"))
for atk in ATTACKS:
    add_seeded("contrastive_mcd", atk, d["methods"]["mcd"]["datasets"][atk]["auc"])
    add_seeded("contrastive_kcd", atk, d["methods"]["kcd"]["datasets"][atk]["auc"])

# ── 4. HarmBench perplexity / SmoothLLM / HB-classifier ───────────────────

d = json.load(open(OUT / "baselines_harmbench_ppl_smoothllm/harmbench_ppl_smoothllm_results.json"))
for atk in ATTACKS:
    add_seeded("perplexity",       atk, d["results"]["perplexity_nll"]["datasets"][atk]["auc"])
    add_seeded("smoothllm",        atk, d["results"]["smoothllm_refusal_majority"]["datasets"][atk]["auc"])
    add_seeded("harmbench_cls",    atk, d["results"]["harmbench_classifier"]["datasets"][atk]["auc"])

# ── 5. FJD (first-token confidence) ─────────────────────────────────────────

d = json.load(open(OUT / "baselines_fjd_paper/fjd_paper_results.json"))
for atk in ATTACKS:
    add_seeded("fjd", atk,
               d["results"]["fjd_paper_first_token_confidence"]["datasets"][atk]["auc"])

# ── 6. Mahalanobis ───────────────────────────────────────────────────────────

d = json.load(open(OUT / "baselines_rtv_mahalanobis/rtv_mahalanobis_results.json"))
for atk in ATTACKS:
    add_seeded("mahalanobis", atk, d["datasets"][atk]["auc"])

# ── 7. JLT hidden tensor (SVM / RF / LogReg) ─────────────────────────────────

d = json.load(open(OUT / "baselines_jlt_hidden_tensor/jlt_hidden_tensor_results.json"))
for atk in ATTACKS:
    add_seeded("jlt_svm",    atk, d["models"]["pca_svm_rbf"]["datasets"][atk]["auc"])
    add_seeded("jlt_rf",     atk, d["models"]["pca_random_forest"]["datasets"][atk]["auc"])
    add_seeded("jlt_logreg", atk, d["models"]["pca_logreg"]["datasets"][atk]["auc"])

# ── 8. Llama Guard (no per-seed) ─────────────────────────────────────────────

d = json.load(open(OUT / "baselines_all_attacks/llama_guard_all_attacks.json"))
for atk in ATTACKS:
    add_scalar("llama_guard_input",        atk, d[atk]["input_only_auc"])
    add_scalar("llama_guard_input_output", atk, d[atk]["input_output_auc"])

# ── 9. Self-exam / Verbalized (no per-seed) ───────────────────────────────────

d = json.load(open(OUT / "baselines_all_attacks/self_exam_all_attacks.json"))
for atk in ATTACKS:
    add_scalar("verbalized_direct", atk, d[atk]["direct_auc"])
    add_scalar("verbalized_cot",    atk, d[atk]["cot_auc"])

# ── 10. GradSafe (no per-seed) ───────────────────────────────────────────────

d = json.load(open(OUT / "baselines_all_attacks/gradsafe_all_attacks.json"))
for atk in ATTACKS:
    add_scalar("gradsafe", atk, d["results"][atk]["auc"])

# ── 11. WildGuard (no per-seed) ──────────────────────────────────────────────

d = json.load(open(OUT / "baselines_wildguard/wildguard_results.json"))
for atk in ATTACKS:
    add_scalar("wildguard_response_harm",    atk, d[atk]["response_harmfulness"]["auc"])
    add_scalar("wildguard_response_comply",  atk, d[atk]["response_compliance"]["auc"])

# ── t-tests + BH correction ──────────────────────────────────────────────────

ttest_results = {}  # key -> (mean, std, p_raw)
for (label, atk), aucs in cells.items():
    a = np.array(aucs)
    _, p = ttest_1samp(a, popmean=0.5, alternative="greater")
    ttest_results[(label, atk)] = (float(np.mean(a)), float(np.std(a)), float(p))

keys_bh = sorted(ttest_results.keys())
pvals   = np.array([ttest_results[k][2] for k in keys_bh])
n       = len(pvals)
order   = np.argsort(pvals)
ranks   = np.empty(n, dtype=int)
ranks[order] = np.arange(1, n + 1)
adj     = np.minimum(1.0, pvals * n / ranks)
for i in range(n - 2, -1, -1):
    adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])

bh = {k: (ttest_results[k][0], ttest_results[k][1], ttest_results[k][2], float(adj[i]))
      for i, k in enumerate(keys_bh)}

def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

# ── Print table ──────────────────────────────────────────────────────────────

# Ordered method list for display
METHODS = [
    # ours
    ("all_traits",          "all_traits"),
    ("top_traits",          "top_traits"),
    ("raw",                 "raw"),
    ("pca",                 "pca"),
    # seeded baselines
    ("jbshield",            "jbshield"),
    ("contrastive_mcd",     "contrastive_mcd"),
    ("contrastive_kcd",     "contrastive_kcd"),
    ("perplexity",          "perplexity"),
    ("smoothllm",           "smoothllm"),
    ("harmbench_cls",       "harmbench_cls"),
    ("fjd",                 "fjd"),
    ("mahalanobis",         "mahalanobis"),
    ("jlt_svm",             "jlt_svm"),
    ("jlt_rf",              "jlt_rf"),
    ("jlt_logreg",          "jlt_logreg"),
    # scalar baselines
    ("llama_guard_input",        "llama_guard_input"),
    ("llama_guard_input_output", "llama_guard_i/o"),
    ("verbalized_direct",        "verbalized_direct"),
    ("verbalized_cot",           "verbalized_cot"),
    ("gradsafe",                 "gradsafe"),
    ("wildguard_response_harm",  "wildguard_harm"),
    ("wildguard_response_comply","wildguard_comply"),
]

col_w = 26
print("\n" + "=" * (22 + col_w * len(ATTACKS)))
print("AUC  (p_adj)  [*** p<.001  ** p<.01  * p<.05]  — no p-val for non-seeded methods")
print("=" * (22 + col_w * len(ATTACKS)))
header = f"{'Method':<22}" + "".join(f"{a:>{col_w}}" for a in ATTACKS)
print(header)
print("-" * (22 + col_w * len(ATTACKS)))

all_data = {}
for (method_key, display_name) in METHODS:
    row = f"{display_name:<22}"
    for atk in ATTACKS:
        k = (method_key, atk)
        if k in bh:
            mean, std, p_raw, p_adj = bh[k]
            cell = f"{mean:.3f}({p_adj:.2e}){sig(p_adj)}"
        elif k in point:
            mean = point[k]
            cell = f"{mean:.3f}"
        else:
            cell = "—"
        row += f"{cell:>{col_w}}"
    print(row)

print("=" * (22 + col_w * len(ATTACKS)))

# ── Save ─────────────────────────────────────────────────────────────────────

out = {}
for (method_key, display_name) in METHODS:
    out[display_name] = {}
    for atk in ATTACKS:
        k = (method_key, atk)
        if k in bh:
            mean, std, p_raw, p_adj = bh[k]
            out[display_name][atk] = {"mean": mean, "std": std, "p_raw": p_raw,
                                       "p_adj_bh": p_adj, "sig": sig(p_adj)}
        elif k in point:
            out[display_name][atk] = {"mean": point[k]}

out_path = OUT / "pvalue_table_full.json"
json.dump(out, open(out_path, "w"), indent=2)
print(f"\nSaved to {out_path}")
