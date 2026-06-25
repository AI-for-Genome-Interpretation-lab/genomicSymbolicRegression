#!/usr/bin/env python3
"""
Filled-contour surface plot of Feyn pair-detection Jaccard on the dense
longqt_uniform grid (K=2..50, n_per_pair=2..100 step 2).

Two side-by-side panels (strict + loose metric) with overlay dots for the
three real datasets (LongQT, FEVR, Hypodontia).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

CSV       = "uniform_surface_dense.csv"
OUT_PNG   = "uniform_surface_dense.png"
LEVELS    = np.linspace(0.0, 1.0, 11)   # 0.0, 0.1, ..., 1.0

REAL = {
    "LongQT":     {"K": 9, "npp": 133/9, "npp_min": 7,  "npp_max": 27,
                   "color": "#1F77B4",
                   "feyn_strict": 0.200, "feyn_loose": 0.333},
    "FEVR":       {"K": 7, "npp": 20.0,  "npp_min": 20, "npp_max": 20,
                   "color": "#FF7F0E",
                   "feyn_strict": 0.143, "feyn_loose": 0.143},
    "Hypodontia": {"K": 2, "npp": 20.0,  "npp_min": 20, "npp_max": 20,
                   "color": "#2CA02C",
                   "feyn_strict": 0.500, "feyn_loose": 0.500},
}


def make_grid(df, value_col):
    K_vals   = sorted(df["K"].astype(int).unique().tolist())
    npp_vals = sorted(df["n_per_pair"].astype(int).unique().tolist())
    Z = np.full((len(npp_vals), len(K_vals)), np.nan)
    for _, r in df.iterrows():
        i = npp_vals.index(int(r["n_per_pair"]))
        j = K_vals.index(int(r["K"]))
        Z[i, j] = r[value_col]
    return np.array(K_vals), np.array(npp_vals), Z


def overlay_dots(ax, value_key):
    for name, d in REAL.items():
        x  = d["K"]
        y  = d["npp"]
        if d["npp_min"] != d["npp_max"]:
            ax.plot([x, x], [d["npp_min"], d["npp_max"]],
                    color=d["color"], lw=2.5, alpha=0.55,
                    solid_capstyle="round", zorder=5)
            for yt in (d["npp_min"], d["npp_max"]):
                ax.plot([x - 0.4, x + 0.4], [yt, yt],
                        color=d["color"], lw=1.8, alpha=0.55, zorder=5)
        ax.scatter([x], [y], s=140, c=d["color"], edgecolors="black",
                   linewidths=1.2, alpha=0.7, zorder=6)
        ax.annotate(
            f"{name}\nJ={d[value_key]:.2f}",
            xy=(x, y), xytext=(10, 8),
            textcoords="offset points",
            fontsize=9, fontweight="bold",
            color="black", alpha=0.9,
            bbox=dict(boxstyle="round,pad=0.25",
                      fc="white", ec=d["color"], lw=1.0, alpha=0.7),
            zorder=7)


def draw_contour(ax, K, npp, Z, cmap, title):
    # Fill NaN cells with 0 for contouring (treat unsampled as 'no detection')
    Zf = np.where(np.isnan(Z), 0.0, Z)
    KK, NN = np.meshgrid(K, npp)
    cf = ax.contourf(KK, NN, Zf, levels=LEVELS, cmap=cmap, extend="neither")
    ax.contour(KK, NN, Zf, levels=LEVELS, colors="white",
               linewidths=0.4, alpha=0.45)
    ax.set_xlabel("K (number of causative pairs)", fontsize=13)
    ax.set_ylabel("Samples per pair", fontsize=13)
    ax.set_title(title, fontsize=14, fontname="Arial")
    ax.set_xlim(K.min(), K.max())
    ax.set_ylim(npp.min(), npp.max())
    return cf


def main():
    df = pd.read_csv(CSV)
    K_arr, npp_arr, Z_strict = make_grid(df, "feyn_J_strict")

    cmap = mcolors.LinearSegmentedColormap.from_list("wb", ["white", "#1155CC"])

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cf_s = draw_contour(ax, K_arr, npp_arr, Z_strict, cmap,
                        "Feyn Jaccard — multiplicative pairs")
    overlay_dots(ax, "feyn_strict")

    fig.colorbar(cf_s, ax=ax, fraction=0.045, pad=0.02, label="Jaccard")

    fig.suptitle(
        "Pair detection on LongQT-uniform synthetic — dense grid",
        fontsize=15, fontname="Arial", y=0.98)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}  ({len(df)} cells)")


if __name__ == "__main__":
    main()
