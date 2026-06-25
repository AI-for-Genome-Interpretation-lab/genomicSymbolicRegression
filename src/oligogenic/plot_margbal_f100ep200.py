#!/usr/bin/env python3
"""Single-panel contour plot for margbal-uniform f100 ep200 sweep."""
import importlib, sys
sys.path.insert(0, ".")
mod = importlib.import_module("plot_uniform_surface")
mod.CSV     = "uniform_surface_margbal_f100ep200.csv"
mod.OUT_PNG = "uniform_surface_margbal_f100ep200.png"
mod.FEYN_COL = "feyn_J_strict"


def main():
    import matplotlib.pyplot as plt
    import numpy as np, pandas as pd
    import matplotlib.colors as mcolors

    df = pd.read_csv(mod.CSV)
    K_vals   = sorted(df["K"].astype(int).unique().tolist())
    npp_vals = sorted(df["n_per_pair"].astype(int).unique().tolist())
    M_feyn = mod.make_matrix(df, mod.FEYN_COL, K_vals, npp_vals)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cmap_b = mcolors.LinearSegmentedColormap.from_list("wb", ["white", "#1155CC"])
    im_f = mod.draw_heatmap(ax, M_feyn, K_vals, npp_vals, cmap_b,
                            "Feyn J_MUL — margbal-uniform f100 ep200")
    mod.overlay_dots(ax, K_vals, npp_vals, "feyn_strict")
    fig.colorbar(im_f, ax=ax, fraction=0.045, pad=0.02, label="Jaccard")
    fig.suptitle("Pair detection — margbal-uniform / f100 / 200 epochs",
                 fontsize=15, fontname="Arial", y=0.98)
    plt.savefig(mod.OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Saved → {mod.OUT_PNG}")


if __name__ == "__main__":
    main()
