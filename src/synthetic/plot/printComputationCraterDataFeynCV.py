#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Re-draw Fig 5 (craterHeatmap) from the 5-fold CV bundles.

Produces 3 versions:
  craterHeatmap_CV_main.png         MAIN  : RF replaces MLP, no Comparison
                                            (Feyn(Gauss), Feyn, RF, Ridge, Lasso, GB)
  craterHeatmap_CV_suppl_mean.png   SUPPL : full mean
                                            (+ MLP + Comparison), square cells
  craterHeatmap_CV_suppl_std.png    SUPPL : full std (+ Pooled std), square cells
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- config ---------------------------------------------------------------
SAMPLES = [100, 500, 1000, 2000, 3000, 4000]
QTL     = [2, 4, 8, 16, 32, 50, 100]

METRIC_NAME = "Pearson"
METRIC_IDX  = 1

GAUSS_BUNDLE   = "results/run_CraterGaussCV/runCraterGaussCV_FINAL.pickle"
SIGMOID_BUNDLE = "results/run_CraterSigmoidCV/runCraterSigmoidCV_FINAL.pickle"

# All output PNGs go here (override with FIGOUT). Never the repo root.
OUTDIR = os.environ.get("FIGOUT", "figures/_build")
os.makedirs(OUTDIR, exist_ok=True)

# Column orderings ----------------------------------------------------------
# Full (supplementary): Feyn first, then conventional, then GB
FULL_KEYS  = ["FeynGauss Test", "FeynBIC Test", "MLP Test",
              "Ridge Test", "Lasso Test", "RF Test", "GB Test"]
FULL_NAMES = ["FeynGauss",      "Feyn",         "MLP",
              "Ridge",          "Lasso",        "RF",      "GB"]

# Main paper: no Comparison, no Lasso (always ~0 on crater), include MLP and RF
MAIN_KEYS  = ["FeynGauss Test", "FeynBIC Test", "RF Test", "MLP Test",
              "Ridge Test",     "GB Test"]
MAIN_NAMES = ["FeynGauss",      "Feyn",         "RF",      "MLP",
              "Ridge",          "GB"]

# Full set (supplementary): includes Lasso
ALL_KEYS  = ["FeynGauss Test", "FeynBIC Test", "RF Test", "MLP Test",
             "Ridge Test",     "Lasso Test",   "GB Test"]
ALL_NAMES = ["FeynGauss",      "Feyn",         "RF",      "MLP",
             "Ridge",          "Lasso",        "GB"]


# ---- helpers --------------------------------------------------------------
def load_bundle(path):
    return pickle.load(open(path, "rb"))


def build_matrix(bundle, model_key, std=False, metric_idx=METRIC_IDX):
    mat = np.zeros((len(SAMPLES), len(QTL)))
    suffix = "_std" if std else ""
    look_key = model_key + suffix
    for j, q in enumerate(QTL):
        for i, s in enumerate(SAMPLES):
            tag = f"S{s}_Q{q}"
            try:
                tup = bundle[tag][look_key]
                v = tup[metric_idx]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    v = 0.0
                mat[i, j] = v
            except (KeyError, IndexError, TypeError):
                mat[i, j] = 0.0
    return mat


def diff_mat(mat_list):
    """Feyn (best of FeynGauss, FeynBIC) minus max of conventional methods."""
    feyn_best = np.maximum(mat_list[0], mat_list[1])
    rest = np.max(np.stack(mat_list[2:], axis=2), axis=2)
    return feyn_best - rest


# ---- plot -----------------------------------------------------------------
def plot(bundle_gauss, bundle_sigmoid, std=False, vmax_override=None,
         vmin_override=None,
         out_name=None, keys=None, lbls=None,
         metric_idx=METRIC_IDX, metric_name=METRIC_NAME):
    """Two landscape rows (Gaussian on top, Sigmoid below) × one column per
       algorithm.
       std=False → mean (paper), std=True → std (supplementary).
       keys/lbls default to MAIN_KEYS/MAIN_NAMES (no Lasso).
       metric_idx selects which entry of the score tuple to plot.
    """
    cmap_main = "inferno"
    if std:
        vmin = vmin_override if vmin_override is not None else 0.0
        vmax = vmax_override if vmax_override is not None else 0.35
        title_kind = f"{metric_name} std (5-fold CV)"
    else:
        vmin = vmin_override if vmin_override is not None else 0.0
        vmax = vmax_override if vmax_override is not None else 1.0
        title_kind = f"{metric_name} (5-fold CV mean)"

    if keys is None:
        keys, lbls = MAIN_KEYS, MAIN_NAMES
    n_cols = len(keys)

    fig, axs = plt.subplots(figsize=(2.0 + 1.7 * n_cols, 4.2),
                            nrows=2, ncols=n_cols,
                            sharey="row", sharex="col", layout="constrained")
    fig.get_layout_engine().set(w_pad=0.0, h_pad=0.0, wspace=0.10, hspace=0.0,
                                rect=(0.02, 0, 0.98, 1))
    fig.suptitle(title_kind, fontsize=13)

    im_ref = None
    for row, (bundle, row_label) in enumerate([(bundle_gauss,   "Gaussian"),
                                               (bundle_sigmoid, "Sigmoid")]):
        for col in range(n_cols):
            ax = axs[row, col]
            m = build_matrix(bundle, keys[col], std=std, metric_idx=metric_idx)
            im = ax.imshow(m, cmap=cmap_main, vmin=vmin, vmax=vmax,
                           aspect="equal")
            if row == 0:
                ax.set_title(lbls[col], fontsize=10)
            ax.set_yticks(range(len(SAMPLES)))
            ax.set_yticklabels(SAMPLES, fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{row_label} landscape\nSamples", fontsize=9)
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.set_xticks(range(len(QTL)))
            ax.set_xticklabels(QTL, rotation=45, fontsize=7)
            if row == 1:
                ax.set_xlabel("QTLs", fontsize=9)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            im_ref = im

        # one colorbar per row, matched to the heatmap (square) height
        cbar = fig.colorbar(im_ref, ax=axs[row, :], orientation="vertical",
                            fraction=0.04, pad=0.02, shrink=0.7, aspect=12)
        cbar.set_label(metric_name, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    if out_name is None:
        out_name = f"craterHeatmap_CV{'_std' if std else ''}.png"
    plt.savefig(os.path.join(OUTDIR, out_name), dpi=400)
    print("Saved:", out_name)
    plt.close()


def _rmse_vmax(bg, bs, rmse_keys, std=False):
    """95th-pctl across all RMSE entries (mean or std) for the supplied keys."""
    vals = []
    for bundle in (bg, bs):
        for k in rmse_keys:
            look = k + ("_std" if std else "")
            for s in SAMPLES:
                for q in QTL:
                    tag = f"S{s}_Q{q}"
                    try:
                        v = bundle[tag][look][1]
                        if v is not None and not np.isnan(v):
                            vals.append(float(v))
                    except (KeyError, IndexError, TypeError):
                        pass
    if not vals:
        return 1.0
    return float(np.percentile(vals, 95))


def estimate_std_vmax(bg, bs, metric_idx=METRIC_IDX):
    vals = []
    for bundle in (bg, bs):
        for k in FULL_KEYS:
            for s in SAMPLES:
                for q in QTL:
                    tag = f"S{s}_Q{q}"
                    try:
                        v = bundle[tag][k + "_std"][metric_idx]
                        if v is not None and not np.isnan(v):
                            vals.append(float(v))
                    except (KeyError, IndexError, TypeError):
                        pass
    if not vals:
        return 0.35
    return float(min(np.percentile(vals, 95), 0.5))


def plot_lasso(bundle_gauss, bundle_sigmoid, out_name=None):
    """Supplementary: Lasso only, Pearson (square cells).
       2 rows (Gaussian / Sigmoid) × 1 col (Pearson). R² is shown in the
       full R² figure."""
    cmap_main = "inferno"
    vmin, vmax = 0.0, 1.0

    fig, axs = plt.subplots(figsize=(3.5, 4.2),
                            nrows=2, ncols=1, sharex="col",
                            layout="constrained")
    fig.get_layout_engine().set(w_pad=0.0, h_pad=0.0, wspace=0.0, hspace=0.0,
                                rect=(0.02, 0, 0.98, 1))
    fig.suptitle("Lasso Pearson (5-fold CV mean)", fontsize=13)

    for row, (bundle, row_label) in enumerate([(bundle_gauss,   "Gaussian"),
                                               (bundle_sigmoid, "Sigmoid")]):
        ax = axs[row]
        m = build_matrix(bundle, "Lasso Test", std=False, metric_idx=1)
        im = ax.imshow(m, cmap=cmap_main, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_yticks(range(len(SAMPLES)))
        ax.set_yticklabels(SAMPLES, fontsize=7)
        ax.set_ylabel(f"{row_label} landscape\nSamples", fontsize=9)
        ax.set_xticks(range(len(QTL)))
        ax.set_xticklabels(QTL, rotation=45, fontsize=7)
        if row == 1:
            ax.set_xlabel("QTLs", fontsize=9)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                            fraction=0.04, pad=0.02, shrink=0.7, aspect=12)
        cbar.set_label("Pearson", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    if out_name is None:
        out_name = "craterHeatmap_CV_lasso.png"
    plt.savefig(os.path.join(OUTDIR, out_name), dpi=400)
    print("Saved:", out_name)
    plt.close()


def main():
    bg = load_bundle(GAUSS_BUNDLE)
    bs = load_bundle(SIGMOID_BUNDLE)
    print("Gaussian   settings:", len(bg))
    print("Sigmoid    settings:", len(bs))

    # Main: mean, no Lasso
    plot(bg, bs, std=False)

    vmax_std = estimate_std_vmax(bg, bs)
    print(f"std vmax (95th pctl): {vmax_std:.3f}")
    # Suppl: std with Lasso included (full set)
    plot(bg, bs, std=True, vmax_override=vmax_std,
         out_name="craterHeatmap_CV_std.png",
         keys=ALL_KEYS, lbls=ALL_NAMES)

    # Suppl: R² mean with full set (Lasso included)
    plot(bg, bs, std=False,
         out_name="craterHeatmap_CV_R2.png",
         keys=ALL_KEYS, lbls=ALL_NAMES,
         metric_idx=2, metric_name="R²")

    # Suppl: R² std with full set
    vmax_std_r2 = estimate_std_vmax(bg, bs, metric_idx=2)
    print(f"R² std vmax (95th pctl): {vmax_std_r2:.3f}")
    plot(bg, bs, std=True, vmax_override=vmax_std_r2,
         out_name="craterHeatmap_CV_R2_std.png",
         keys=ALL_KEYS, lbls=ALL_NAMES,
         metric_idx=2, metric_name="R²")

    # Suppl: RMSE mean and std with full set
    rmse_keys = [k + "_RMSE" for k in ALL_KEYS]
    rmse_vmax = _rmse_vmax(bg, bs, rmse_keys, std=False)
    print(f"RMSE mean vmax (95th pctl): {rmse_vmax:.4f}")
    plot(bg, bs, std=False, vmax_override=rmse_vmax,
         out_name="craterHeatmap_CV_RMSE.png",
         keys=rmse_keys, lbls=ALL_NAMES,
         metric_idx=1, metric_name="RMSE")
    rmse_std_vmax = _rmse_vmax(bg, bs, rmse_keys, std=True)
    print(f"RMSE std  vmax (95th pctl): {rmse_std_vmax:.4f}")
    plot(bg, bs, std=True, vmax_override=rmse_std_vmax,
         out_name="craterHeatmap_CV_RMSE_std.png",
         keys=rmse_keys, lbls=ALL_NAMES,
         metric_idx=1, metric_name="RMSE")

    # Suppl: Lasso only, Pearson
    plot_lasso(bg, bs)


if __name__ == "__main__":
    main()
