#!/usr/bin/env python3
"""
Compute p-values for jailbreak detection AUC results.

For each (method, attack) cell:
  - One-sample one-tailed t-test: H0: mean_AUC = 0.5, H1: mean_AUC > 0.5
  - Uses per-seed AUC arrays (n=50) stored in the result JSONs

Applies Benjamini-Hochberg FDR correction across all cells.
Prints a table with AUC, p-value, and significance markers.
"""

import json
from pathlib import Path
import numpy as np
from scipy.stats import ttest_1samp

OUT = Path("/dlabscratch1/bazina/assistant-axis-llama3.1-8B/full_trait_output")

# --------------------------------------------------------------------------
# Load per-seed AUC arrays for each (mode, attack)
# --------------------------------------------------------------------------
# Structure: cells[(mode, attack)] = list of AUC floats (n_seeds=50)

cells = {}

def load_file(path, key_map):
    """Load per-seed AUC arrays. key_map: {json_key: canonical_attack_name}"""
    d = json.load(open(path))
    for mode, mode_data in d["modes"].items():
        for json_key, attack_name in key_map.items():
            if json_key in mode_data and "all" in mode_data[json_key]:
                cells[(mode, attack_name)] = mode_data[json_key]["all"]

load_file(OUT / "transfer_results_gcg_pair/transfer_results.json",
          {"transfer_GCG": "GCG", "transfer_PAIR": "PAIR"})
load_file(OUT / "transfer_results_pap/transfer_results.json",
          {"transfer_PAP": "PAP"})
load_file(OUT / "transfer_results_gptfuzz/transfer_results.json",
          {"transfer_GPTFuzz": "GPTFuzz"})
load_file(OUT / "transfer_results_pez/transfer_results.json",
          {"transfer_PEZ": "PEZ"})

# --------------------------------------------------------------------------
# Compute raw p-values
# --------------------------------------------------------------------------
results = {}  # (mode, attack) -> (mean, std, p_raw)
for (mode, attack), aucs in cells.items():
    a = np.array(aucs)
    mean = float(np.mean(a))
    std  = float(np.std(a))
    _, p = ttest_1samp(a, popmean=0.5, alternative="greater")
    results[(mode, attack)] = (mean, std, float(p))

# --------------------------------------------------------------------------
# Benjamini-Hochberg FDR correction
# --------------------------------------------------------------------------
keys   = sorted(results.keys())
pvals  = np.array([results[k][2] for k in keys])
n      = len(pvals)
order  = np.argsort(pvals)
ranks  = np.empty(n, dtype=int)
ranks[order] = np.arange(1, n + 1)
bh_thresh = ranks / n * 0.05  # FDR q = 0.05
reject = pvals <= bh_thresh
# For display, compute BH-adjusted p-values
adj_pvals = np.minimum(1.0, pvals * n / ranks)
# Make monotone
for i in range(n - 2, -1, -1):
    adj_pvals[order[i]] = min(adj_pvals[order[i]], adj_pvals[order[i + 1]])

fdr_results = {k: (results[k][0], results[k][1], results[k][2], float(adj_pvals[i]))
               for i, k in enumerate(keys)}

# --------------------------------------------------------------------------
# Print table
# --------------------------------------------------------------------------
MODES   = ["all_traits", "top_traits", "raw", "pca"]
ATTACKS = ["GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]

def sig(p_adj):
    if p_adj < 0.001: return "***"
    if p_adj < 0.01:  return "**"
    if p_adj < 0.05:  return "*"
    return ""

col_w = 22
header = f"{'Mode':<14}" + "".join(f"{a:>{col_w}}" for a in ATTACKS)
print("\n" + "=" * (14 + col_w * len(ATTACKS)))
print("AUC  (p_adj)  [*** p<.001  ** p<.01  * p<.05]")
print("=" * (14 + col_w * len(ATTACKS)))
print(header)
print("-" * (14 + col_w * len(ATTACKS)))

summary = {}
for mode in MODES:
    row = f"{mode:<14}"
    transfer_aucs = []
    for attack in ATTACKS:
        key = (mode, attack)
        if key in fdr_results:
            mean, std, p_raw, p_adj = fdr_results[key]
            s = sig(p_adj)
            cell = f"{mean:.3f} ({p_adj:.3f}){s}"
            transfer_aucs.append(mean)
        else:
            cell = "—"
        row += f"{cell:>{col_w}}"
    avg = np.mean(transfer_aucs) if transfer_aucs else float("nan")
    row += f"   avg={avg:.3f}"
    print(row)
    summary[mode] = {"attacks": {}, "avg_transfer": float(avg)}
    for attack in ATTACKS:
        key = (mode, attack)
        if key in fdr_results:
            mean, std, p_raw, p_adj = fdr_results[key]
            summary[mode]["attacks"][attack] = {
                "mean": mean, "std": std, "p_raw": p_raw, "p_adj_bh": p_adj, "sig": sig(p_adj)
            }

print("=" * (14 + col_w * len(ATTACKS)))

# --------------------------------------------------------------------------
# Also print clean summary for all_traits (the main claim)
# --------------------------------------------------------------------------
print("\n── all_traits detail ──────────────────────────────────────")
print(f"  {'Attack':<10} {'AUC':>6} {'±std':>6} {'p_raw':>8} {'p_adj':>8} {'sig':>4}")
print("  " + "─" * 46)
for attack in ATTACKS:
    key = ("all_traits", attack)
    if key in fdr_results:
        mean, std, p_raw, p_adj = fdr_results[key]
        print(f"  {attack:<10} {mean:>6.3f} {std:>6.3f} {p_raw:>8.4f} {p_adj:>8.4f} {sig(p_adj):>4}")

# Save
out_path = OUT / "pvalue_table.json"
json.dump(summary, open(out_path, "w"), indent=2)
print(f"\nSaved to {out_path}")
