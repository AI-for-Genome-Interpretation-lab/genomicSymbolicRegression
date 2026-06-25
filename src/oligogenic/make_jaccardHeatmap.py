#!/usr/bin/env python3
"""
Left: bar chart describing each phenotype (n_samples, n_pairs, n_genes).
Right: heatmap Feyn vs PLINK1.9 Jaccard.
Only phenotypes where Feyn > 0 with f100 + 200 epochs (permpair-FIXED).
Output: jaccardHeatmap.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# (name, n_samples, n_pairs (=K truth pairs), n_genes, Feyn_J, PLINK_J)
DATA = [
    ("Hypodontia",        40,  2,  4, 0.500, 0.002),
    ("LongQT",            73,  9, 10, 0.333, 0.333),
    ("FEVR",             140,  7, 14, 0.143, 0.085),
    ("Hypothyroidism",     420, 28, 40, 0.025, 0.007),
]

names    = [d[0] for d in DATA]
samples  = np.array([d[1] for d in DATA])
pairs    = np.array([d[2] for d in DATA])
genes    = np.array([d[3] for d in DATA])
feyn_J   = np.array([d[4] for d in DATA])
plink_J  = np.array([d[5] for d in DATA])
mat      = np.vstack([feyn_J, plink_J])

fig = plt.figure(figsize=(14, 4.5))
gs  = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.30)

# ── LEFT: descriptor barplot (3 grouped bars per phenotype, log y) ──────
ax1 = fig.add_subplot(gs[0, 0])
x   = np.arange(len(names))
w   = 0.27
ax1.bar(x - w, samples, width=w, color="#A8D8EA", label="samples")
ax1.bar(x,     pairs,   width=w, color="#FFAAA5", label="pairs")
ax1.bar(x + w, genes,   width=w, color="#B8E0D2", label="genes")
ax1.set_yscale("log")
ax1.set_xticks(x)
ax1.set_xticklabels(names, fontsize=14, fontname="Arial",
                    rotation=30, ha="right")
ax1.set_ylabel("count (log scale)", fontsize=15, fontname="Arial")
ax1.tick_params(axis="y", labelsize=13)
ax1.legend(fontsize=13, frameon=False, loc="upper left")
ax1.set_title("Dataset descriptors", fontsize=16, fontname="Arial", pad=10)
ax1.grid(axis="y", linestyle=":", alpha=0.4, which="both")
ax1.set_ylim(top=1500)

# Annotate exact values above each bar
for xi, vals in zip(x, zip(samples, pairs, genes)):
    for off, v in zip([-w, 0, w], vals):
        ax1.text(xi + off, v * 1.08, str(int(v)),
                 ha="center", va="bottom", fontsize=10,
                 fontname="Arial", color="#222222")

# ── RIGHT: heatmap ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
cmap = mcolors.LinearSegmentedColormap.from_list("wb", ["white", "#1155CC"])
im = ax2.imshow(mat, cmap=cmap, vmin=0, vmax=0.55, aspect="equal")

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        color = "white" if v > 0.30 else "black"
        ax2.text(j, i, f"{v:.3f}", ha="center", va="center",
                 fontsize=17, fontname="Arial", color=color, fontweight="bold")

ax2.set_xticks(range(len(names)))
ax2.set_xticklabels(names, fontsize=14, fontname="Arial",
                    rotation=30, ha="right")
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["Feyn", "PLINK1.9"], fontsize=16,
                    fontname="Arial", fontweight="bold")
ax2.set_title("Pair-detection Jaccard", fontsize=16, fontname="Arial", pad=10)

cb = fig.colorbar(im, ax=ax2, fraction=0.05, pad=0.02, shrink=0.75)
cb.set_label("Jaccard", fontsize=15, fontname="Arial")
cb.ax.tick_params(labelsize=13)

fig.suptitle("Feyn vs PLINK1.9 pair detection on OLIDA datasets",
             fontsize=18, fontname="Arial", y=1.00)

plt.savefig("jaccardHeatmap.png", dpi=200, bbox_inches="tight")
print("Saved → jaccardHeatmap.png")
