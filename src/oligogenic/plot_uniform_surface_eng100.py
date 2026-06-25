#!/usr/bin/env python3
"""Top-100 engineered variant: reads uniform_surface_eng100.csv."""
import importlib, sys
sys.path.insert(0, ".")
mod = importlib.import_module("plot_uniform_surface")

mod.CSV     = "uniform_surface_eng100.csv"
mod.OUT_PNG = "uniform_surface_eng100.png"
mod.FEYN_COL = "feyn_J_eng"

for d in mod.REAL.values():
    d["feyn_eng"] = d.get("feyn_strict", 0.0)


def main():
    import matplotlib.pyplot as plt
    import numpy as np, pandas as pd
    import matplotlib.colors as mcolors

    df = pd.read_csv(mod.CSV)
    K_vals   = sorted(df["K"].astype(int).unique().tolist())
    npp_vals = sorted(df["n_per_pair"].astype(int).unique().tolist())

    M_feyn = mod.make_matrix(df, mod.FEYN_COL, K_vals, npp_vals)
    M_plk  = mod.make_matrix(df, "plink_J",   K_vals, npp_vals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"wspace": 0.30})
    cmap_b = mcolors.LinearSegmentedColormap.from_list("wb", ["white", "#1155CC"])
    cmap_r = mcolors.LinearSegmentedColormap.from_list("wr", ["white", "#CC0000"])

    im_f = mod.draw_heatmap(axes[0], M_feyn, K_vals, npp_vals, cmap_b,
                            "Feyn Jaccard (engineered v_a·v_b on top-100 by |corr|)")
    mod.overlay_dots(axes[0], K_vals, npp_vals, "feyn_eng")

    im_p = mod.draw_heatmap(axes[1], M_plk, K_vals, npp_vals, cmap_r,
                            "PLINK1.9 Jaccard (top-100 SNPs)")
    mod.overlay_dots(axes[1], K_vals, npp_vals, "plink")

    fig.colorbar(im_f, ax=axes[0], fraction=0.045, pad=0.02, label="Jaccard")
    fig.colorbar(im_p, ax=axes[1], fraction=0.045, pad=0.02, label="Jaccard")

    fig.suptitle(
        "Pair detection — engineered interactions over top-100 SNPs by |corr|",
        fontsize=15, fontname="Arial", y=0.98)
    plt.savefig(mod.OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Saved → {mod.OUT_PNG}")


if __name__ == "__main__":
    main()
