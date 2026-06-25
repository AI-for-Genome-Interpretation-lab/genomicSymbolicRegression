#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Re-draw Fig 2 heatmaps from the 5-fold CV bundle.

Produces three versions of each (mode, H, metric) plot:
  *_CV_main.png       MAIN  : RF replaces MLP, no Comparison column (5 cols)
  *_CV_suppl.png      SUPPL : full mean (Feyn, MLP, Lasso, Ridge, RF, GB,
                              Comparison), square cells
  *_CV_suppl_std.png  SUPPL : full std + Pooled std, square cells

PLINK2 is excluded (all-NaN in the CV bundle: predictive run only).
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FOLDER   = "results/run100_synthLargeCV/"
BUNDLE   = FOLDER + "run100CV_FINAL.pickle"

# All output PNGs go here (override with FIGOUT). Never the repo root.
OUTDIR = os.environ.get("FIGOUT", "figures/_build")
os.makedirs(OUTDIR, exist_ok=True)

qtl_vals    = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 26, 30, 34, 38, 40, 46, 50]
decoy_vals  = [0, 100, 500, 1000, 1900]
sample_vals = [500, 1000, 2000]

metricsHT  = ["", "Pearson", "R2", "NDCG"]

# Column orderings ---------------------------------------------------------
# Full (supplementary) order: Feyn first (for Comparison), then alphabetical-ish
FULL_KEYS  = ["FeynBIC Test", "MLP Test", "Lasso Test", "Ridge Test", "RF Test", "GB Test"]
FULL_NAMES = ["Feyn",         "MLP",      "Lasso",      "Ridge",      "RF",      "GB"]

# Main paper order: Feyn first, then conventional baselines
MAIN_KEYS  = ["FeynBIC Test", "RF Test", "Lasso Test", "Ridge Test"]
MAIN_NAMES = ["Feyn",         "RF",      "Lasso",      "Ridge"]

# Supplementary version: include MLP and GB
SUPPL_KEYS  = ["FeynBIC Test", "RF Test", "MLP Test", "Lasso Test", "Ridge Test", "GB Test"]
SUPPL_NAMES = ["Feyn",         "RF",      "MLP",      "Lasso",      "Ridge",      "GB"]


# ---- accessors ------------------------------------------------------------
def tag(h, qtl, dom, epi, dec, smp, ep=10):
    return f"h{h}_q{qtl}_D{dom}_E{epi}_dec{dec}_N{smp}_ep{ep}"


def get_val(bundle, h, qtl, dom, epi, dec, smp, algo_key, metric, std=False):
    t = tag(h, qtl, dom, epi, dec, smp)
    if t not in bundle:
        return np.nan
    look = algo_key + ("_std" if std else "")
    v = bundle[t].get(look)
    if v is None or not isinstance(v, (tuple, list)) or len(v) <= metric:
        return np.nan
    val = v[metric]
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    return float(val)


def build_decoys(bundle, h, dom, epi, smp, algo_key, metric, std=False):
    mat = np.zeros((len(qtl_vals), len(decoy_vals)))
    for i, q in enumerate(qtl_vals):
        for j, d in enumerate(decoy_vals):
            mat[i, j] = get_val(bundle, h, q, dom, epi, d, smp, algo_key, metric, std)
    return mat


def build_samples(bundle, h, dom, epi, dec, algo_key, metric, std=False):
    mat = np.zeros((len(qtl_vals), len(sample_vals)))
    for i, q in enumerate(qtl_vals):
        for j, s in enumerate(sample_vals):
            mat[i, j] = get_val(bundle, h, q, dom, epi, dec, s, algo_key, metric, std)
    return mat


def diff_mat(mats):
    """Feyn (mats[0]) minus max of all others."""
    feyn   = mats[0]
    others = np.max(np.stack(mats[1:], axis=2), axis=2)
    return feyn - others


# ---- plot -----------------------------------------------------------------
def plot_heatmaps(bundle, mode, h, metric,
                  version="main",      # "main", "suppl_mean", "suppl_std"
                  std_vmax=None):
    dom, epi = '0', '0'
    mname = metricsHT[metric]
    cmap_main = "inferno"
    std = (version == "suppl_std")

    if std:
        vmin, vmax = 0.0, std_vmax if std_vmax is not None else 0.3
    else:
        vmin, vmax = 0.0, 1.0

    # column set
    if version == "main":
        keys, lbls = MAIN_KEYS, MAIN_NAMES
        with_extra = False                # no Comparison column
    elif version == "suppl_mean":
        keys, lbls = FULL_KEYS, FULL_NAMES
        with_extra = True                 # Comparison column
    else:                                  # suppl_std: no Pooled std column
        keys, lbls = FULL_KEYS, FULL_NAMES
        with_extra = False

    # axis layout
    if mode == 'decoys':
        fixed_samples = 1000
        other_label = "Decoys"
        other_ticks = decoy_vals
        fname_tpl = "qtlVdecoys_H%s_Metric%s_CV_%s.png"
        mats = [build_decoys(bundle, h, dom, epi, fixed_samples, k, metric, std)
                for k in keys]
        suptitle_extra = f"fix samples={fixed_samples}"
    else:
        fixed_decoys = 100
        other_label = "Samples"
        other_ticks = sample_vals
        fname_tpl = "qtlVsamples_H%s_Metric%s_CV_%s.png"
        mats = [build_samples(bundle, h, dom, epi, fixed_decoys, k, metric, std)
                for k in keys]
        suptitle_extra = f"fix decoys={fixed_decoys}"

    # Transpose for MAIN: QTL on X, decoys/samples on Y
    if version == "main":
        mats = [m.T for m in mats]
        xlabel, xticks = "QTLs", qtl_vals
        ylabel, yticks = other_label, other_ticks
    else:
        xlabel, xticks = other_label, other_ticks
        ylabel, yticks = "QTLs", qtl_vals

    n_algo = len(keys)
    n_cols = n_algo + (1 if with_extra else 0)

    # square cells for supplementary; auto for main
    aspect_pol = 'equal' if version != "main" else 'auto'

    # figsize and layout:
    #   main  = 4 rows × 1 col, split into group A (top 3) and group B (4th)
    #           with a visual gap between them and A) / B) panel labels.
    #   suppl = 1 row × n_cols with square cells
    if version == "main":
        figsize = (10, 1.8 * n_cols + 1.2)
        fig = plt.figure(figsize=figsize, layout="constrained")
        # Insert a small empty row between the A and B groups
        hr = [1.0] * MAIN_SPLIT + [0.25] + [1.0] * (n_cols - MAIN_SPLIT)
        gs = fig.add_gridspec(len(hr), 1, height_ratios=hr)
        axs = []
        for i in range(n_cols):
            gs_row = i if i < MAIN_SPLIT else i + 1   # skip the gap row
            axs.append(fig.add_subplot(gs[gs_row]))
        # Share x across all panels (independent y because they all share decoys/samples)
        for a in axs[1:]:
            a.sharex(axs[0])
    else:
        figsize = (3 + 2.4 * n_cols, 7)
        fig, axs = plt.subplots(1, n_cols, figsize=figsize, sharey=True,
                                layout="constrained")
    if version == "main":
        kind = "mean (5-fold CV)"
    elif version == "suppl_mean":
        kind = "mean (5-fold CV) — full"
    else:
        kind = "std (5-fold CV) — full"
    fig.suptitle(f"{mname} {kind}  |  H={h:.1f}  |  dom=0, epi=0  |  {suptitle_extra}",
                 fontsize=12)

    im_ref = None
    for A in range(n_algo):
        ax = axs[A]
        ax.set_title(lbls[A], fontsize=10)
        im = ax.imshow(mats[A], cmap=cmap_main, vmin=vmin, vmax=vmax,
                       aspect=aspect_pol, origin='upper')
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, rotation=45, fontsize=7)
        ax.set_yticks(range(len(yticks)))
        ax.set_yticklabels(yticks, fontsize=7)
        if version == "main":
            ax.set_ylabel(ylabel, fontsize=8)
            if A == n_algo - 1:
                ax.set_xlabel(xlabel, fontsize=8)
        else:
            if A == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel, fontsize=8)
        im_ref = im

    if with_extra:
        ax = axs[n_cols - 1]
        if std:
            pooled = np.sqrt(np.mean(np.stack(mats, axis=2) ** 2, axis=2))
            im_cmp = ax.imshow(pooled, cmap=cmap_main, vmin=vmin, vmax=vmax,
                               aspect=aspect_pol, origin='upper')
            ax.set_title("Pooled std", fontsize=10)
        else:
            dm = diff_mat(mats)
            im_cmp = ax.imshow(dm, cmap="seismic", vmin=-1, vmax=1,
                               aspect=aspect_pol, origin='upper')
            ax.set_title("Feyn - Best", fontsize=10)
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, rotation=45, fontsize=7)
        ax.set_yticks(range(len(yticks)))
        ax.set_yticklabels(yticks, fontsize=7)
        ax.set_xlabel(xlabel, fontsize=8)
        fig.colorbar(im_ref, ax=axs[:n_algo], orientation='vertical',
                     fraction=0.015, pad=0.02, label=mname)
        fig.colorbar(im_cmp, ax=axs[n_algo], orientation='vertical',
                     fraction=0.04,  pad=0.04)
    else:
        fig.colorbar(im_ref, ax=axs[:n_algo], orientation='vertical',
                     fraction=0.015, pad=0.02, label=mname)

    # A) / B) panel labels for the main version
    if version == "main":
        axs[0].text(-0.07, 1.15, 'A', transform=axs[0].transAxes,
                    fontsize=14, fontweight='bold', va='bottom', ha='right')
        axs[MAIN_SPLIT].text(-0.07, 1.15, 'B',
                             transform=axs[MAIN_SPLIT].transAxes,
                             fontsize=14, fontweight='bold',
                             va='bottom', ha='right')

    h_str = str(h).replace('.', '')
    fname = fname_tpl % (h_str, mname, version)
    plt.savefig(os.path.join(OUTDIR, fname), dpi=300)
    print("Saved:", fname)
    plt.close()


def estimate_std_vmax(bundle, metric):
    """Auto-pick the 95th-percentile std across all settings/methods/configs (cap 0.5)."""
    vals = []
    for h in [0.6, 0.3]:
        for q in qtl_vals:
            for d in decoy_vals:
                for s in sample_vals:
                    for nk in FULL_KEYS:
                        v = get_val(bundle, h, q, '0', '0', d, s, nk, metric, std=True)
                        if not np.isnan(v):
                            vals.append(v)
    if not vals:
        return 0.3
    return float(min(np.percentile(vals, 95), 0.5))


def estimate_rmse_vmax(bundle, std=False):
    """Auto-pick 95th-percentile RMSE (or RMSE std) across all settings/methods."""
    vals = []
    for h in [0.6, 0.3]:
        for q in qtl_vals:
            for d in decoy_vals:
                for s in sample_vals:
                    for nk in FULL_KEYS:
                        v = get_val(bundle, h, q, '0', '0', d, s,
                                    nk + "_RMSE", metric=1, std=std)
                        if not np.isnan(v):
                            vals.append(v)
    if not vals:
        return 1.0
    return float(np.percentile(vals, 95))


def plot_main_combined(bundle, metric, version="main", std=False, std_vmax=None,
                       key_suffix="", mname_override=None, vmax_override=None):
    """Combined figure per metric, H=0.6 fixed.
       Two sections, both with rows = (dom, epi) combinations:
         A) decoys=1000 fixed, heatmap Y axis = samples (3 vals)
         B) samples=1000 fixed, heatmap Y axis = decoys (5 vals)
       Columns depend on `version`:
         "main"  → Feyn, RF, Lasso, Ridge (4 cols)
         "suppl" → Feyn, RF, MLP, Lasso, Ridge, GB (6 cols)

       For RMSE: pass key_suffix='_RMSE', metric=1, mname_override='RMSE',
       and an appropriate vmax_override (since RMSE is not in [0,1]).
    """
    mname = mname_override if mname_override else metricsHT[metric]
    cmap_main = "inferno"
    if std:
        vmin, vmax = 0.0, std_vmax if std_vmax is not None else 0.30
    else:
        vmin, vmax = 0.0, vmax_override if vmax_override is not None else 1.0
    if version == "suppl":
        keys, lbls = SUPPL_KEYS, SUPPL_NAMES
    else:
        keys, lbls = MAIN_KEYS, MAIN_NAMES
    keys = [k + key_suffix for k in keys]
    n_cols = len(keys)

    H = 0.6
    DOM_EPI = [('0', '0'), ('0', '1'), ('1', '0'), ('1', '1')]
    A_DECOYS  = 1000      # section A: decoys fixed
    B_SAMPLES = 1000      # section B: samples fixed, decoys vary on heatmap Y

    n_rows_sec = len(DOM_EPI)

    # Layout: n_rows section A + spacer + n_rows section B
    fig = plt.figure(figsize=(3 + 3.0 * n_cols,
                              1.0 * 2 * n_rows_sec + 2.0),
                     layout="constrained")
    height_ratios = ([1.0] * n_rows_sec) + [0.35] + ([1.0] * n_rows_sec)
    gs = fig.add_gridspec(len(height_ratios), n_cols, height_ratios=height_ratios)

    im_ref = None
    all_axes_A = []
    all_axes_B = []

    kind = "std (5-fold CV)" if std else "comparison CV"
    sections = [
        ("A", f"{mname} {kind}, herit={H}, decoys={A_DECOYS}", 0),
        ("B", f"{mname} {kind}, herit={H}, feats=2000",        n_rows_sec + 1),
    ]

    # ---- Section A: heatmap Y = samples, decoys fixed ----------------------
    sec_label, sec_title, sec_offset = sections[0]
    for r, (dom, epi) in enumerate(DOM_EPI):
        gs_row = sec_offset + r
        for c in range(n_cols):
            ax = fig.add_subplot(gs[gs_row, c])
            all_axes_A.append(ax)
            mat = build_samples(bundle, H, dom, epi, A_DECOYS, keys[c],
                                metric, std)
            mat = mat.T                       # samples × qtl
            im = ax.imshow(mat, cmap=cmap_main, vmin=vmin, vmax=vmax,
                           aspect='equal', origin='upper')
            ax.set_xticks(range(len(qtl_vals)))
            ax.set_xticklabels(qtl_vals, rotation=45, fontsize=6)
            ax.set_yticks(range(len(sample_vals)))
            ax.set_yticklabels(sample_vals, fontsize=7)
            if r == 0:
                ax.set_title(lbls[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"dom={dom}, epi={epi}\nSamples", fontsize=8)
            if r == n_rows_sec - 1:
                ax.set_xlabel("QTLs", fontsize=8)
            else:
                ax.set_xticklabels([])
            im_ref = im

    # ---- Section B: heatmap Y = decoys, X = QTL, samples fixed -------------
    sec_label, sec_title, sec_offset = sections[1]
    for r, (dom, epi) in enumerate(DOM_EPI):
        gs_row = sec_offset + r
        for c in range(n_cols):
            ax = fig.add_subplot(gs[gs_row, c])
            all_axes_B.append(ax)
            mat = build_decoys(bundle, H, dom, epi, B_SAMPLES, keys[c],
                               metric, std)
            mat = mat.T                       # decoys × qtl
            im = ax.imshow(mat, cmap=cmap_main, vmin=vmin, vmax=vmax,
                           aspect='equal', origin='upper')
            ax.set_xticks(range(len(qtl_vals)))
            ax.set_xticklabels(qtl_vals, rotation=45, fontsize=6)
            ax.set_yticks(range(len(decoy_vals)))
            ax.set_yticklabels(decoy_vals, fontsize=7)
            if r == 0:
                ax.set_title(lbls[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"dom={dom}, epi={epi}\nDecoys", fontsize=8)
            if r == n_rows_sec - 1:
                ax.set_xlabel("QTLs", fontsize=8)
            else:
                ax.set_xticklabels([])
            im_ref = im

    # Single colorbar
    fig.colorbar(im_ref, ax=all_axes_A + all_axes_B,
                 orientation='vertical', fraction=0.015, pad=0.02, label=mname)

    # Section A title goes in fig.suptitle (reserved top space).
    # Section B title goes in fig.text at the gap between sections.
    fig.suptitle(sections[0][1], fontsize=13, fontweight='bold')

    fig.canvas.draw()
    yA_top   = all_axes_A[0].get_position(fig).y1
    yB_top   = all_axes_B[0].get_position(fig).y1
    yA_bot_A = all_axes_A[-1].get_position(fig).y0

    fig.text(0.03, yA_top + 0.003, 'A',
             ha='left', va='bottom', fontsize=16, fontweight='bold')
    fig.text(0.5, yB_top + 0.012, sections[1][1],
             ha='center', va='bottom', fontsize=13, fontweight='bold')
    fig.text(0.03, yB_top + 0.012, 'B',
             ha='left', va='bottom', fontsize=16, fontweight='bold')

    safe_mname = mname.replace("²", "2").replace(" ", "")
    fname = f"qtl_{version}_Metric{safe_mname}_CV{'_std' if std else ''}.png"
    plt.savefig(os.path.join(OUTDIR, fname), dpi=300)
    print("Saved:", fname)
    plt.close()


def main():
    bundle = pickle.load(open(BUNDLE, "rb"))
    print(f"Loaded {len(bundle)} settings from {BUNDLE}")

    for metric in [1, 2, 3]:
        std_vmax = estimate_std_vmax(bundle, metric)
        print(f"  metric {metricsHT[metric]} std vmax = {std_vmax:.3f}")

        # Combined main figure (4 algos) + combined suppl figure (6 algos)
        # × mean and std for each.
        plot_main_combined(bundle, metric, version="main",  std=False)
        plot_main_combined(bundle, metric, version="main",  std=True,  std_vmax=std_vmax)
        plot_main_combined(bundle, metric, version="suppl", std=False)
        plot_main_combined(bundle, metric, version="suppl", std=True,  std_vmax=std_vmax)

        # Supplementary still per (H, mode)
        for h in [0.6, 0.3]:
            for mode in ('decoys', 'samples'):
                plot_heatmaps(bundle, mode, h, metric, version="suppl_mean")
                plot_heatmaps(bundle, mode, h, metric, version="suppl_std",
                              std_vmax=std_vmax)

    # RMSE combined figures (main + suppl), mean and std
    rmse_vmax_mean = estimate_rmse_vmax(bundle, std=False)
    rmse_vmax_std  = estimate_rmse_vmax(bundle, std=True)
    print(f"  metric RMSE mean vmax = {rmse_vmax_mean:.3f}")
    print(f"  metric RMSE std  vmax = {rmse_vmax_std:.3f}")
    for version in ("main", "suppl"):
        plot_main_combined(bundle, metric=1, version=version, std=False,
                           key_suffix="_RMSE", mname_override="RMSE",
                           vmax_override=rmse_vmax_mean)
        plot_main_combined(bundle, metric=1, version=version, std=True,
                           key_suffix="_RMSE", mname_override="RMSE",
                           std_vmax=rmse_vmax_std)

    print("All plots saved.")


if __name__ == '__main__':
    main()
