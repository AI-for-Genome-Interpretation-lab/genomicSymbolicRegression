#!/usr/bin/env python3
"""
Composite figure: disease statistics barplots (left) + Feyn vs PLINK Jaccard heatmap (right).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

CSV = "combo_jaccard_feyn_plink.csv"

DISEASE_ORDER = ["LongQT", "FEVR", "Hypodontia"]
RENAME = {
    "LongQT":     "LongQT",
    "FEVR":       "FEVR",
    "Hypodontia": "Hypodontia",
}

STATS = {
    "LongQT":     {"Patients": 191, "Genes": 4, "Caus. SNPs": 10, "K pairs": 9},
    "FEVR":       {"Patients": 260, "Genes": 4, "Caus. SNPs": 14, "K pairs": 7},
    "Hypodontia": {"Patients": 240, "Genes": 2, "Caus. SNPs": 4,  "K pairs": 2},
}

STAT_COLS = ["Patients", "Genes", "Caus. SNPs", "K pairs"]

DISEASE_COLORS = {
    "LongQT":     "#4472C4",
    "FEVR":       "#ED7D31",
    "Hypodontia": "#70AD47",
}


def draw_stat_bars(axes):
    """One horizontal bar subplot per statistic."""
    for ax, stat in zip(axes, STAT_COLS):
        values = [STATS[d][stat] for d in DISEASE_ORDER]
        colors = [DISEASE_COLORS[d] for d in DISEASE_ORDER]
        bars = ax.barh(DISEASE_ORDER, values, color=colors, height=0.55, edgecolor="none")

        # value labels at the end of each bar
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                    str(v), va="center", ha="left", fontsize=9, fontweight="bold")

        ax.set_xlim(0, max(values) * 1.25)
        ax.set_xlabel(stat, fontsize=9, labelpad=3)
        ax.tick_params(axis="y", labelsize=9, left=False)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
        ax.set_facecolor("none")
        ax.xaxis.set_label_position("bottom")


def main():
    df = pd.read_csv(CSV)
    df["disease"] = df["dataset"].map(RENAME)
    df = df.dropna(subset=["disease"])
    df = df.drop_duplicates("disease")
    df = df.set_index("disease").reindex(DISEASE_ORDER)

    nat = df[["Feyn_J", "PLINK_J"]].values.astype(float)
    nat_plot = np.nan_to_num(nat, nan=0.0)

    cmap = mcolors.LinearSegmentedColormap.from_list("wb", ["white", "#1155CC"])

    fig = plt.figure(figsize=(11, 5))
    gs_outer = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1.0], wspace=0.45, figure=fig)

    gs_left = gridspec.GridSpecFromSubplotSpec(
        len(STAT_COLS), 1, subplot_spec=gs_outer[0], hspace=0.55)
    axes_stats = [fig.add_subplot(gs_left[i]) for i in range(len(STAT_COLS))]

    ax_heat = fig.add_subplot(gs_outer[1])

    draw_stat_bars(axes_stats)

    # invisible full-height axes for the left panel — used only for the A label
    ax_left_bg = fig.add_subplot(gs_outer[0])
    ax_left_bg.axis("off")
    ax_left_bg.patch.set_visible(False)

    # panel labels at the same transAxes y on both full-height axes
    ax_left_bg.text(-0.08, 1.12, "A", transform=ax_left_bg.transAxes,
                    fontsize=17, fontweight="bold", va="top")
    ax_heat.text(-0.18, 1.12, "B", transform=ax_heat.transAxes,
                 fontsize=17, fontweight="bold", va="top")

    im = ax_heat.imshow(nat_plot, cmap=cmap, vmin=0, vmax=1,
                        aspect="auto", interpolation="none")

    for ri in range(nat_plot.shape[0]):
        for ci in range(nat_plot.shape[1]):
            v = nat[ri, ci]
            txt = "—" if np.isnan(v) else f"{v:.2f}"
            color = "white" if nat_plot[ri, ci] > 0.55 else "black"
            ax_heat.text(ci, ri, txt, ha="center", va="center",
                         fontsize=10, color=color, fontweight="bold")

    ax_heat.set_xticks([0, 1])
    ax_heat.set_xticklabels(["Feyn", "PLINK"], fontsize=11)
    ax_heat.set_yticks(range(len(DISEASE_ORDER)))
    ax_heat.set_yticklabels(
        [f"{d}  (K={int(df.loc[d,'K'])})" for d in DISEASE_ORDER], fontsize=9)
    ax_heat.tick_params(left=False)
    ax_heat.set_title("Pair detection Jaccard\nFeyn vs PLINK1.9", fontsize=11, pad=6)

    fig.colorbar(im, ax=ax_heat, orientation="vertical",
                 fraction=0.05, pad=0.03, label="Jaccard")

    fname = "jaccardHeatmap.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    print("Saved:", fname)
    plt.close()


if __name__ == "__main__":
    main()
