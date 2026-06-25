#!/usr/bin/env python3
"""
Two-panel surface plot of pair-detection Jaccard on the longqt_uniform grid:
  left  = Feyn Jaccard (multiplicative pairs)
  right = PLINK1.9 Jaccard

Overlay colored dots for the 3 real datasets at their (K, n_per_pair) with the
real Jaccard value annotated next to each dot.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import RegularGridInterpolator

CSV       = "uniform_surface.csv"
OUT_PNG   = "uniform_surface.png"
FEYN_COL  = "feyn_J_combo"    # mul ∪ hessian pairs (combined detector)

# Real datasets and their pair Jaccards.
# feyn_strict = multiplicative pairs from sympify (combo_jaccard_feyn_plink.csv)
# feyn_loose  = truth pairs whose both SNPs are in the Feyn formula features
# npp        = mean pair occurrences per pair (total pair-occurrences / K).
# npp_min/max = range of pair occurrences across the K pairs (uniform if equal).
REAL = {
    "LongQT":     {"K": 9, "npp": 133/9, "npp_min": 7,  "npp_max": 27,
                   "color": "#1F77B4",
                   "feyn_strict": 0.200, "feyn_combo": 0.200,
                   "feyn_loose": 0.333, "plink": 0.083},
    "FEVR":       {"K": 7, "npp": 20.0,  "npp_min": 20, "npp_max": 20,
                   "color": "#FF7F0E",
                   "feyn_strict": 0.143, "feyn_combo": 0.143,
                   "feyn_loose": 0.143, "plink": 0.113},
    "Hypodontia": {"K": 2, "npp": 20.0,  "npp_min": 20, "npp_max": 20,
                   "color": "#2CA02C",
                   "feyn_strict": 0.500, "feyn_combo": 0.500,
                   "feyn_loose": 0.500, "plink": 0.002},
}


def make_matrix(df, value_col, K_vals, npp_vals):
    M = np.full((len(npp_vals), len(K_vals)), np.nan)
    for _, r in df.iterrows():
        if int(r["K"]) in K_vals and int(r["n_per_pair"]) in npp_vals:
            i = npp_vals.index(int(r["n_per_pair"]))
            j = K_vals.index(int(r["K"]))
            M[i, j] = r[value_col]
    return M


def overlay_dots(ax, K_vals, npp_vals, value_key):
    """Plot the 3 real datasets. Dot = mean npp per pair; vertical bar = range
    of pair occurrences across the K pairs (non-uniform datasets only)."""
    K_arr   = np.array(K_vals,   dtype=float)
    npp_arr = np.array(npp_vals, dtype=float)
    for name, d in REAL.items():
        x  = float(np.interp(d["K"],   K_arr,   np.arange(len(K_arr))))
        y  = float(np.interp(d["npp"], npp_arr, np.arange(len(npp_arr))))
        y0 = float(np.interp(d["npp_min"], npp_arr, np.arange(len(npp_arr))))
        y1 = float(np.interp(d["npp_max"], npp_arr, np.arange(len(npp_arr))))
        # Range bar shows pair-count variability (collapses to a dot when uniform).
        if d["npp_min"] != d["npp_max"]:
            ax.plot([x, x], [y0, y1], color=d["color"], lw=2.5,
                    solid_capstyle="round", alpha=0.45, zorder=4)
            for yt in (y0, y1):
                ax.plot([x - 0.15, x + 0.15], [yt, yt],
                        color=d["color"], lw=1.8, alpha=0.45, zorder=4)
        ax.scatter([x], [y], s=180, c=d["color"], edgecolors="black",
                   linewidths=1.2, alpha=0.55, zorder=5)
        # Place annotation to the LEFT for dots near the right edge of the panel.
        on_right_edge = x >= len(K_arr) - 1.5
        offset = (4, 8) if on_right_edge else (10, 8)
        ha     = "right" if on_right_edge else "left"
        ax.annotate(
            f"{name}\nJ={d[value_key]:.2f}",
            xy=(x, y), xytext=offset,
            textcoords="offset points",
            fontsize=9, fontweight="bold", ha=ha,
            color="black", alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.25",
                      fc="white", ec=d["color"], lw=1.0, alpha=0.55),
            zorder=6)


def draw_heatmap(ax, M, K_vals, npp_vals, cmap, title):
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1,
                   aspect="auto", origin="lower", interpolation="nearest")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v): continue
            color = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color=color)
    ax.set_xticks(range(len(K_vals)))
    ax.set_xticklabels(K_vals, fontsize=12)
    ax.set_yticks(range(len(npp_vals)))
    ax.set_yticklabels(npp_vals, fontsize=12)
    ax.set_xlabel("K (number of causative pairs)", fontsize=13)
    ax.set_ylabel("Samples per pair", fontsize=13)
    ax.set_title(title, fontsize=14, fontname="Arial")
    return im


def main():
    df = pd.read_csv(CSV)
    K_vals   = sorted(df["K"].astype(int).unique().tolist())
    npp_vals = sorted(df["n_per_pair"].astype(int).unique().tolist())

    M_feyn = make_matrix(df, FEYN_COL, K_vals, npp_vals)
    M_plk  = make_matrix(df, "plink_J", K_vals, npp_vals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"wspace": 0.30})
    cmap_b = mcolors.LinearSegmentedColormap.from_list("wb", ["white", "#1155CC"])
    cmap_r = mcolors.LinearSegmentedColormap.from_list("wr", ["white", "#CC0000"])

    feyn_label = "Feyn Jaccard"

    if FEYN_COL == "feyn_J_loose":
        feyn_dot_key = "feyn_loose"
    elif FEYN_COL == "feyn_J_combo":
        feyn_dot_key = "feyn_combo"
    else:
        feyn_dot_key = "feyn_strict"
    im_f = draw_heatmap(axes[0], M_feyn, K_vals, npp_vals, cmap_b, feyn_label)
    overlay_dots(axes[0], K_vals, npp_vals, feyn_dot_key)

    im_p = draw_heatmap(axes[1], M_plk, K_vals, npp_vals, cmap_r,
                        "PLINK1.9 Jaccard")
    overlay_dots(axes[1], K_vals, npp_vals, "plink")

    fig.colorbar(im_f, ax=axes[0], fraction=0.045, pad=0.02, label="Jaccard")
    fig.colorbar(im_p, ax=axes[1], fraction=0.045, pad=0.02, label="Jaccard")

    fig.suptitle(
        "Pair detection on LongQT-uniform synthetic  (real datasets overlaid)",
        fontsize=15, fontname="Arial", y=0.98)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
