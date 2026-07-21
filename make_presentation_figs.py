"""Generate presentation figures from the paper's result tables + the latency benchmark.
All numbers are hardcoded from report/neurips_2025.tex (detection + attack tables) and the
inference-latency benchmark (job time-allmethods, 2026-06-15). Run: python3 make_presentation_figs.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = "presentation_figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})

OURS = "#c0392b"      # highlight colour for our trait method
OURS2 = "#e67e22"     # our raw head
BASE = "#5b6b7b"      # baselines

# ----------------------------------------------------------------------------- #
# Detection table (Llama-3.1-8B), from tab:detection. (ID = HarmBench test, Xfer = avg transfer)
# name: (ID_auc, transfer_auc)
det = {
    "Trait projections":      (0.86, 0.76),
    "Raw activations":        (0.81, 0.71),
    "PCA (3)":                (0.88, 0.68),
    "Mahalanobis contrastive":(0.92, 0.71),
    "JB-Leaves-a-Trace (SVM)":(0.91, 0.70),
    "JBShield":               (0.85, 0.69),
    "JB-Leaves-a-Trace (RF)": (0.89, 0.64),
    "GradSafe":               (0.95, 0.62),
    "Llama Guard (input)":    (0.69, 0.61),
    "Verbalized (no CoT)":    (0.58, 0.55),
    "Verbalized (CoT)":       (0.58, 0.52),
    "Perplexity":             (0.65, 0.51),
}

# ---- FIG 1: in-distribution vs transfer ("who overfits") -------------------- #
fig, ax = plt.subplots(figsize=(10.0, 6.4))
lo, hi = 0.45, 1.0
ax.plot([lo, hi], [lo, hi], ls="--", color="#999", lw=1.2, zorder=1,
        label="transfer = in-distribution")
for name, (idv, tr) in det.items():
    ours = name in ("Trait projections", "Raw activations")
    c = OURS if name == "Trait projections" else (OURS2 if name == "Raw activations" else BASE)
    ax.scatter(idv, tr, s=240 if ours else 90, color=c, zorder=3,
               edgecolor="black", linewidth=1.1 if ours else 0.4,
               marker="*" if name == "Trait projections" else "o")
# label placement (dx, dy in points; ha)
offs = {
    "Trait projections": (8, 9, "left"), "Raw activations": (-8, -16, "right"),
    "PCA (3)": (6, -14, "left"),
    "Mahalanobis contrastive": (4, 11, "center"), "JB-Leaves-a-Trace (SVM)": (8, -13, "left"),
    "JBShield": (-7, -15, "right"), "JB-Leaves-a-Trace (RF)": (9, 3, "left"),
    "GradSafe": (6, -15, "left"),
    "Llama Guard (input)": (9, -2, "left"), "Verbalized (no CoT)": (9, 4, "left"),
    "Verbalized (CoT)": (-9, -12, "right"), "Perplexity": (9, -2, "left"),
}
for name, (idv, tr) in det.items():
    dx, dy, ha = offs[name]
    fw = "bold" if name in ("Trait projections", "Raw activations") else "normal"
    cc = OURS if name == "Trait projections" else (OURS2 if name == "Raw activations" else "#333")
    ax.annotate(name, (idv, tr), textcoords="offset points", xytext=(dx, dy),
                ha=ha, fontsize=10.5, color=cc, fontweight=fw)
ax.set_xlabel("In-distribution AUC  (HarmBench, trained here)")
ax.set_ylabel("Transfer AUC  (5 unseen attacks)")
ax.set_title("Detectors that win in-distribution overfit; trait projections transfer best",
             fontsize=14, pad=12)
ax.set_xlim(lo, hi); ax.set_ylim(lo, 0.84)
ax.legend(loc="upper left", frameon=False, fontsize=10.5)
fig.savefig(f"{OUT}/fig_id_vs_transfer.png")
plt.close(fig)

# ---- FIG 2: accuracy vs cost (efficiency) ----------------------------------- #
# batch=1 per-request latency (ms) + transfer AUC, for every method with both.
HEAD = "#3b5b8c"; NEURAL = "#5b6b7b"
# name: (latency_ms, transfer_auc, colour, marker, dx, dy, ha)
cost = {
    "Trait projection (ours)": (0.067, 0.76, OURS,   "*",  9,  9, "left"),
    "Raw":                     (0.0042, 0.71, OURS2, "*",  6,-16, "left"),
    "JBShield":                (0.071, 0.69, HEAD,   "o",  0,-15, "center"),
    "PCA (3)":                 (0.242, 0.68, HEAD,   "o",  8, -3, "left"),
    "Mahalanobis":             (2.54, 0.71, HEAD,    "o", -4, 12, "right"),
    "JLT (SVM)":               (2.64, 0.70, HEAD,    "o",  6,-15, "left"),
    "JLT (RF)":                (31.4, 0.64, HEAD,    "o", -6, 12, "right"),
    "Perplexity":              (33.0, 0.51, NEURAL,  "o",  0,-15, "center"),
    "Llama Guard":             (43.0, 0.61, NEURAL,  "o",  8,  6, "left"),
    "Self-Exam":               (3094.0, 0.55, NEURAL,"o", -6, 12, "right"),
    "Self-Exam +CoT":          (3748.0, 0.52, NEURAL,"o",  6,-14, "left"),
    "GradSafe":                (20440.0, 0.62, NEURAL,"o", -4, 12, "right"),
}
fig, ax = plt.subplots(figsize=(9.6, 6.0))
for name, (lat, auc, c, m, dx, dy, ha) in cost.items():
    big = m == "*"
    ax.scatter(lat, auc, s=300 if big else 95, color=c, marker=m, zorder=4 if big else 3,
               edgecolor="black", linewidth=1.1 if big else 0.4)
for name, (lat, auc, c, m, dx, dy, ha) in cost.items():
    ax.annotate(name, (lat, auc), textcoords="offset points", xytext=(dx, dy),
                ha=ha, fontsize=10, color="#1a2b4a" if m == "*" else "#333",
                fontweight="bold" if m == "*" else "normal")
ax.set_xscale("log")
ax.set_xlabel("Added detection latency per request (ms, log scale)  —  lower is better")
ax.set_ylabel("Transfer AUC  —  higher is better")
ax.set_title("Best transfer AND near-zero cost  (top-left is best)", fontsize=14)
ax.set_xlim(1.5e-3, 6e4); ax.set_ylim(0.45, 0.82)
from matplotlib.lines import Line2D
handles = [
    Line2D([0],[0], marker="*", color="w", markerfacecolor=OURS, markeredgecolor="black",
           markersize=15, label="ours (trait / raw)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=HEAD, markeredgecolor="black",
           markersize=10, label="other activation heads"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=NEURAL, markeredgecolor="black",
           markersize=10, label="neural baselines"),
]
ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=10.5)
fig.savefig(f"{OUT}/fig_accuracy_vs_cost.png")
plt.close(fig)

# ---- FIG 3: attack ASR, HarmBench(refused) vs JBB --------------------------- #
# (name, harmbench_refused, jbb)  jbb None = not run
atk = [
    ("Steer-$w$\n(persona)", 0.747, 0.63, True),
    ("Steer-$w$\n(raw)",     0.76,  0.68, False),
    ("Steer-traits",         0.666, 0.16, True),
    ("GPTFuzz",              0.59,  0.77, False),
    ("GCG",                  0.44,  None, False),
    ("PAIR",                 0.39,  0.28, False),
    ("PAP",                  0.28,  0.21, False),
    ("PEZ",                  0.15,  0.02, False),
]
names = [a[0] for a in atk]
hb = [a[1] for a in atk]
jbb = [a[2] for a in atk]
ours_mask = [a[3] for a in atk]
x = np.arange(len(atk)); w = 0.4
fig, ax = plt.subplots(figsize=(11, 5.6))
for i in range(len(atk)):
    chb = OURS if ours_mask[i] else BASE
    ax.bar(x[i] - w/2, hb[i], w, color=chb, edgecolor="black", linewidth=0.5,
           label="HarmBench (originally refused)" if i == 0 else None)
    if jbb[i] is not None:
        ax.bar(x[i] + w/2, jbb[i], w, color=chb, alpha=0.45, edgecolor="black", linewidth=0.5,
               hatch="//", label="JBB (always refused)" if i == 0 else None)
    else:
        ax.text(x[i] + w/2, 0.01, "n/a", ha="center", va="bottom", fontsize=8, color="#888", rotation=90)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10.5)
ax.set_ylabel("Attack success rate")
ax.set_ylim(0, 0.85)
ax.set_title("Steering flips most refusals — and the dense direction transfers, top-$k$ doesn't",
             fontsize=13.5)
ax.legend(loc="upper right", frameon=False, fontsize=10.5)
ax.text(0.0, -0.16, "Red = our trait-based steering  ·  grey = raw-activation control + standard attacks",
        transform=ax.transAxes, fontsize=9.5, color="#555")
fig.savefig(f"{OUT}/fig_attack_asr.png")
plt.close(fig)

# ---- FIG 5: detection cost, ALL timed methods (from the latency benchmark) --- #
# (name, batch=1 ms, class)
costrows = [
    ("raw (ours)",   0.0042, "probe"),
    ("trait (ours)", 0.067,  "probe"),
    ("Perplexity",   33.0,   "1 forward"),
    ("Llama Guard",  48.0,   "1 forward"),
    ("WildGuard",    510.0,  "generate"),
    ("FJD",          1094.0, "generate"),
    ("Self-Exam",    7496.0, "generate"),
    ("GradSafe",     20440.0,"fwd + backward"),
]
classcolor = {"probe": OURS, "1 forward": "#3b5b8c", "generate": BASE, "fwd + backward": "#2f3e52"}
def fmt(ms):
    return f"{ms:g} ms" if ms < 1 else f"{ms:,.0f} ms"
fig, ax = plt.subplots(figsize=(9.6, 5.8))
ypos = list(range(len(costrows)))[::-1]   # first row at top
for y, (name, ms, cls) in zip(ypos, costrows):
    ax.barh(y, ms, color=classcolor[cls], edgecolor="black", linewidth=0.5, height=0.62, zorder=3)
    ax.text(ms * 1.35, y, fmt(ms), va="center", ha="left", fontsize=11,
            fontweight="bold" if cls == "probe" else "normal", color="#222")
ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in costrows], fontsize=12.5)
ax.set_xscale("log"); ax.set_xlim(2e-3, 3e5)
ax.set_xlabel("Added detection latency per request (ms, log scale)  —  lower is better")
ax.set_title("Detection cost per request, across method classes", fontsize=14)
from matplotlib.patches import Patch
handles = [Patch(facecolor=classcolor[c], edgecolor="black", label=c)
           for c in ["probe", "1 forward", "generate", "fwd + backward"]]
ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=10.5, title="what it runs")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
fig.savefig(f"{OUT}/fig_cost_bars.png")
plt.close(fig)

# ---- FIG 4: causal / dimension-drop -- persona(229) vs raw(4096) normal ----- #
groups = ["HarmBench\n(originally refused)", "JBB\n(always refused)"]
persona = [0.747, 0.63]   # ~229-dim trait subspace
raw = [0.76, 0.68]        # full 4096-dim
xg = np.arange(2); bw = 0.34
fig, ax = plt.subplots(figsize=(8.6, 5.6))
b1 = ax.bar(xg - bw/2, persona, bw, color=OURS, edgecolor="black", linewidth=0.6,
            label="persona normal  (~229 dims)")
b2 = ax.bar(xg + bw/2, raw, bw, color=BASE, edgecolor="black", linewidth=0.6,
            label="raw normal  (4096 dims)")
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.012, f"{r.get_height():.2f}",
                ha="center", fontsize=12, fontweight="bold")
ax.set_xticks(xg); ax.set_xticklabels(groups, fontsize=12.5)
ax.set_ylabel("Attack success rate")
ax.set_ylim(0, 0.9)
ax.legend(loc="upper right", frameon=False, fontsize=12)
ax.annotate("drop ~3,900 dims\n→ lose ~1 pt", xy=(xg[0] + bw/2, 0.745), xytext=(0.5, 0.40),
            fontsize=11, color="#444", ha="center",
            arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
fig.savefig(f"{OUT}/fig_causal_dims.png")
plt.close(fig)

print("wrote:")
for f in sorted(os.listdir(OUT)):
    print(" ", os.path.join(OUT, f))
