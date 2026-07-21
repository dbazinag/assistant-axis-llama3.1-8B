#!/usr/bin/env python3
"""Plot per-layer in-domain (HarmBench test) AUC vs. per-layer average transfer AUC."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

CSV_PATH = Path("full_trait_output/all_layers_sweep_summary.csv")
OUT_PATH = Path("full_trait_output/all_layers_sweep_plot.png")

TRANSFER_FAMILIES = ["GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]

layers = []
in_domain_auc = []
transfer_auc = []

with CSV_PATH.open() as f:
    for row in csv.DictReader(f):
        layers.append(int(row["layer"]))
        in_domain_auc.append(float(row["best_test_human_test_auc"]))
        vals = [float(row[f"best_transfer_{fam}"]) for fam in TRANSFER_FAMILIES if row[f"best_transfer_{fam}"]]
        transfer_auc.append(sum(vals) / len(vals))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(layers, in_domain_auc, marker="o", label="In-domain (HarmBench test) AUC")
ax.plot(layers, transfer_auc, marker="s", label="Avg transfer AUC (GCG/PAIR/PAP/GPTFuzz/PEZ)")
ax.set_xlabel("Layer")
ax.set_ylabel("AUC")
ax.set_title("Trait-vector jailbreak detection: AUC by layer")
ax.set_xticks(layers)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"Wrote {OUT_PATH}")
