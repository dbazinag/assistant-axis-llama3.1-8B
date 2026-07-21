"""Render one full 16:9 PNG per slide into ./presentation/ — FIGURE-FORWARD style:
each slide is a title + one dominant visual + a single takeaway caption to talk over.
Detailed bullets live in the speaker notes (presentation_final.md).
Run: python3 make_slides.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

OUT = "presentation"
os.makedirs(OUT, exist_ok=True)

NAVY = "#1a2b4a"
RED = "#c0392b"
ORANGE = "#e67e22"
TEXT = "#222222"
GREY = "#5b6b7b"
LBLUE = "#dde6f0"
LRED = "#f6dcd6"
LGREY = "#e8ecf0"
W, H = 13.333, 7.5
DPI = 144
plt.rcParams.update({"font.family": "DejaVu Sans"})


def new_slide():
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return fig, ax


def title(ax, text, y=0.915):
    ax.text(0.045, y, text, fontsize=28, fontweight="bold", color=NAVY,
            va="center", ha="left", transform=ax.transAxes)
    ax.add_patch(plt.Rectangle((0.045, y - 0.052), 0.15, 0.007, color=RED,
                               transform=ax.transAxes, clip_on=False))


def caption(ax, text, color=RED):
    ax.text(0.5, 0.055, text, fontsize=21, color=color, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)


def add_image(fig, path, rect):
    img = plt.imread(path)
    ar = img.shape[1] / img.shape[0]
    l, b, rw, rh = rect
    rect_ar = (rw * W) / (rh * H)
    if ar > rect_ar:
        nw = rw; nh = rw * W / ar / H
    else:
        nh = rh; nw = rh * H * ar / W
    iax = fig.add_axes([l + (rw - nw) / 2, b + (rh - nh) / 2, nw, nh]); iax.axis("off")
    iax.imshow(img)


def box(ax, cx, cy, w, h, text, fc, ec, tc, fs=17, bold=True):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                 boxstyle="round,pad=0.012,rounding_size=0.02",
                 fc=fc, ec=ec, lw=2, transform=ax.transAxes))
    ax.text(cx, cy, text, fontsize=fs, color=tc, ha="center", va="center",
            fontweight="bold" if bold else "normal", transform=ax.transAxes)


def arrow(ax, x1, y1, x2, y2, color=GREY, lw=2.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), transform=ax.transAxes,
                 arrowstyle="-|>", mutation_scale=22, color=color, lw=lw))


def fig_slide(fname, title_text, image_path, cap, cap_color=RED, rect=(0.06, 0.16, 0.88, 0.60)):
    fig, ax = new_slide(); title(ax, title_text)
    add_image(fig, image_path, list(rect))
    caption(ax, cap, cap_color)
    fig.savefig(f"{OUT}/{fname}", dpi=DPI); plt.close(fig)


# ----------------------------------------------------------------------------- #
# Slide 1 — title
fig, ax = new_slide()
ax.add_patch(plt.Rectangle((0, 0.62), 1, 0.012, color=RED, transform=ax.transAxes))
ax.text(0.5, 0.665, "Detecting and Inducing Jailbreaks", fontsize=40, fontweight="bold",
        color=NAVY, ha="center", va="center", transform=ax.transAxes)
ax.text(0.5, 0.56, "with Persona-Trait Directions", fontsize=40, fontweight="bold",
        color=NAVY, ha="center", va="center", transform=ax.transAxes)
ax.text(0.5, 0.41, "Dominic Bazina-Grolinger", fontsize=22, color=TEXT,
        ha="center", transform=ax.transAxes)
ax.text(0.5, 0.355, "with Viktor Moskvoretskii · Robert West   —   EPFL DLab",
        fontsize=17, color=GREY, ha="center", transform=ax.transAxes)
ax.text(0.5, 0.26, "Llama-3.1-8B-Instruct  ·  HarmBench", fontsize=18, color=RED,
        ha="center", style="italic", transform=ax.transAxes)
fig.savefig(f"{OUT}/slide_01_title.png", dpi=DPI); plt.close(fig)

# Slide 2 — the question (concept diagram)
fig, ax = new_slide(); title(ax, "The question")
ax.text(0.5, 0.80, "Are trait directions tied to jailbreaking — and is the link causal?",
        fontsize=20, color=TEXT, ha="center", va="center", transform=ax.transAxes)
box(ax, 0.50, 0.50, 0.24, 0.20, "Persona-trait\nsubspace\n(~229 directions)", LGREY, NAVY, NAVY, fs=17)
box(ax, 0.17, 0.50, 0.22, 0.18, "DETECT\ndo trait scores\nflag jailbreaks?", LBLUE, "#3b5b8c", NAVY, fs=15)
box(ax, 0.83, 0.50, 0.22, 0.18, "INDUCE\ndoes steering\ncause them?", LRED, RED, "#7a1f12", fs=15)
arrow(ax, 0.385, 0.50, 0.285, 0.50, color="#3b5b8c")
arrow(ax, 0.615, 0.50, 0.715, 0.50, color=RED)
ax.text(0.17, 0.375, "correlation", fontsize=15, color="#3b5b8c", ha="center", style="italic",
        transform=ax.transAxes)
ax.text(0.83, 0.375, "causation", fontsize=15, color=RED, ha="center", style="italic",
        transform=ax.transAxes)
caption(ax, "Same subspace, two sides — that pairing is what lets us argue causality.", RED)
fig.savefig(f"{OUT}/slide_02_question.png", dpi=DPI); plt.close(fig)

# Slide 3 — pipeline figure
fig_slide("slide_03_setup.png", "One trait subspace, used two ways",
          "report/figures/fig_pipeline.png",
          "The same directions feed a detector and a steering attack.",
          rect=(0.06, 0.18, 0.88, 0.56))

# Slide 4 — detection protocol (concept flow)
fig, ax = new_slide(); title(ax, "Detection: trained on one, tested on five")
box(ax, 0.16, 0.55, 0.20, 0.22, "TRAIN + TUNE\nHarmBench only\n(HumanJailbreaks)", LBLUE, "#3b5b8c", NAVY, fs=14)
box(ax, 0.45, 0.55, 0.18, 0.16, "Trait-score\nlogistic regression", LGREY, NAVY, NAVY, fs=14)
arrow(ax, 0.265, 0.55, 0.355, 0.55, color="#3b5b8c")
arrow(ax, 0.545, 0.55, 0.66, 0.55, color=RED)
ax.text(0.81, 0.79, "5 unseen attack families", fontsize=15, color=RED, fontweight="bold",
        ha="center", transform=ax.transAxes)
fam = ["GCG", "PAIR", "PAP", "GPTFuzz", "PEZ"]
for i, f in enumerate(fam):
    box(ax, 0.81, 0.71 - i * 0.115, 0.20, 0.085, f, LRED, RED, "#7a1f12", fs=14)
caption(ax, "Never trained on the attacks we evaluate → it measures transfer.", RED)
fig.savefig(f"{OUT}/slide_04_detection_setup.png", dpi=DPI); plt.close(fig)

# Slide 5 — detection result figure
fig_slide("slide_05_detection_result.png", "Trait projections transfer best",
          "presentation_figures/fig_id_vs_transfer.png",
          "In-distribution winners overfit; trait features generalize (best transfer, 0.76).")

# Slide 6 — which traits (minimal table)
fig, ax = new_slide(); title(ax, "We can read which traits drive it")
ax.text(0.5, 0.79, "Named traits → an interpretable detector", fontsize=19, color=TEXT,
        ha="center", va="center", transform=ax.transAxes)
cols = [("toward REFUSAL", ["paranoid", "confrontational", "analytical", "circumspect"], "#3b5b8c"),
        ("toward JAILBREAK", ["risk_taking", "blunt", "impulsive", "sassy"], RED),
        ("aligned with axis w", ["sardonic", "wry", "sarcastic", "acerbic"], ORANGE)]
xs = [0.20, 0.50, 0.80]
for (head, vals, c), x in zip(cols, xs):
    ax.text(x, 0.64, head, fontsize=18, fontweight="bold", color=c, ha="center",
            va="top", transform=ax.transAxes)
    ax.add_patch(plt.Rectangle((x - 0.11, 0.605), 0.22, 0.005, color=c, transform=ax.transAxes))
    for i, v in enumerate(vals):
        ax.text(x, 0.55 - i * 0.078, v, fontsize=19, color=TEXT, ha="center", va="top",
                transform=ax.transAxes)
caption(ax, "A jailbreak looks like the model shifting its assistant persona.", GREY)
fig.savefig(f"{OUT}/slide_06_traits.png", dpi=DPI); plt.close(fig)

# Slide 7 — efficiency figure
fig_slide("slide_07_efficiency.png", "...and detection is essentially free",
          "presentation_figures/fig_accuracy_vs_cost.png",
          "Best transfer AND ~700× cheaper than Llama Guard.")

# Slide 8 — attack figure
fig_slide("slide_08_attack.png", "Steering induces jailbreaks",
          "presentation_figures/fig_attack_asr.png",
          "Flips ~75% of refusals; the dense direction transfers, top-k doesn't.")

# Slide 9 — causal figure
fig_slide("slide_09_causal.png", "Why this is causal",
          "presentation_figures/fig_causal_dims.png",
          "Drop ~3,900 dimensions, lose ~1 point — the signal is in the trait subspace.")

# Slide 10 — takeaway (minimal)
fig, ax = new_slide(); title(ax, "Takeaway")
ax.text(0.5, 0.66, "The same persona-trait subspace gives", fontsize=23, color=TEXT,
        ha="center", transform=ax.transAxes)
ax.text(0.5, 0.575, "the best-transferring detector", fontsize=25, fontweight="bold",
        color="#3b5b8c", ha="center", transform=ax.transAxes)
ax.text(0.5, 0.495, "AND the strongest refusal-flipping attack.", fontsize=25,
        fontweight="bold", color=RED, ha="center", transform=ax.transAxes)
ax.text(0.5, 0.37, "→ traits are tied to jailbreaking, and the link is partly causal.",
        fontsize=20, color=NAVY, ha="center", style="italic", transform=ax.transAxes)
ax.text(0.5, 0.16, "In progress: OLMo-3, Gemma      ·      Caveats: single model · judge-based · PAP hard",
        fontsize=15, color=GREY, ha="center", transform=ax.transAxes)
fig.savefig(f"{OUT}/slide_10_takeaway.png", dpi=DPI); plt.close(fig)

print("wrote slides:")
for f in sorted(os.listdir(OUT)):
    print(" ", os.path.join(OUT, f))
