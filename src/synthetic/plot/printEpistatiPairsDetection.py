#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Generate Fig 3: epistatic SNP detection Jaccard heatmaps.
# For Feyn: Jaccard(FEATURES, CAUSATIVE) in column space.
# For PLINK2: Jaccard(PLINK2Features, CAUSATIVE) in column space.
# Both measure: fraction of causative (epistatic) SNPs found by each method.
# Produces: epiPairsDetectionJaccard.png  (vs decoys)
#           epiPairsDetectionJaccardSamples.png  (vs samples)
#
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FOLDER = "results/run100_synthLarge/"
RUN_NAME = "run100FINAL100_withRF_PLINK2"

import os
# All output PNGs go here (override with FIGOUT). Never the repo root.
OUTDIR = os.environ.get("FIGOUT", "figures/_build")
os.makedirs(OUTDIR, exist_ok=True)

qtl_vals    = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 26, 30, 34, 38, 40, 46, 50]
decoy_vals  = [0, 100, 500, 1000, 1900]
sample_vals = [500, 1000, 2000]


def jaccard(f1, f2):
    f1, f2 = set(f1), set(f2)
    if not f1 and not f2:
        return 1.0
    if not f1 or not f2:
        return 0.0
    return len(f1 & f2) / len(f1 | f2)


def get_jaccard(results, h, qtl, dom, epi, decoys, samples, method):
    k = (h, qtl, dom, epi, decoys, samples, 10)
    if k not in results:
        return np.nan
    v = results[k]
    causative = v.get('CAUSATIVE', [])
    if method == 'feyn':
        features = v.get('FEATURES', [])
    elif method == 'plink2':
        features = v.get('PLINK2Features', [])
    else:
        return np.nan
    if not causative:
        return np.nan
    return jaccard(features, causative)


def build_matrix_decoys(results, h, dom, epi, samples, method):
    mat = np.zeros((len(qtl_vals), len(decoy_vals)))
    for i, qtl in enumerate(qtl_vals):
        for j, dec in enumerate(decoy_vals):
            v = get_jaccard(results, h, qtl, dom, epi, dec, samples, method)
            mat[i, j] = v if not np.isnan(v) else 0.0
    return mat


def build_matrix_samples(results, h, dom, epi, decoys, method):
    mat = np.zeros((len(qtl_vals), len(sample_vals)))
    for i, qtl in enumerate(qtl_vals):
        for j, smp in enumerate(sample_vals):
            v = get_jaccard(results, h, qtl, dom, epi, decoys, smp, method)
            mat[i, j] = v if not np.isnan(v) else 0.0
    return mat


def plot_jaccard(results, mode, h, dom, epi, fname):
    if mode == 'decoys':
        fixed = 1000   # fix samples
        xticks = decoy_vals
        xlabel = "Decoys"
        mat_feyn   = build_matrix_decoys(results, h, dom, epi, fixed, 'feyn')
        mat_plink2 = build_matrix_decoys(results, h, dom, epi, fixed, 'plink2')
    else:
        fixed = 100    # fix decoys
        xticks = sample_vals
        xlabel = "Samples"
        mat_feyn   = build_matrix_samples(results, h, dom, epi, fixed, 'feyn')
        mat_plink2 = build_matrix_samples(results, h, dom, epi, fixed, 'plink2')

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    fig.suptitle("Causative SNP Detection Jaccard  |  H=%.1f  dom=%s  epi=%s" % (h, dom, epi),
                 fontsize=11)

    for ax, mat, title in zip(axs, [mat_feyn, mat_plink2], ["Feyn", "PLINK1.9"]):
        im = ax.imshow(mat, cmap="inferno", vmin=0, vmax=1, aspect='auto', origin='upper')
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, rotation=45, fontsize=8)
        ax.set_yticks(range(len(qtl_vals)))
        ax.set_yticklabels(qtl_vals, fontsize=8)
        ax.set_xlabel(xlabel)
        if ax is axs[0]:
            ax.set_ylabel("QTLs")

    fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.02, label="Jaccard")

    plt.savefig(os.path.join(OUTDIR, fname), dpi=300)
    print("Saved:", fname)
    plt.close()


def main():
    results = pickle.load(open(FOLDER + RUN_NAME + ".pickle", "rb"))
    print("Loaded %d entries" % len(results))

    h   = 0.6
    dom = '0'
    epi = '1'   # focus on epistatic settings

    plot_jaccard(results, 'decoys',  h, dom, epi, "epiPairsDetectionJaccard.png")
    plot_jaccard(results, 'samples', h, dom, epi, "epiPairsDetectionJaccardSamples.png")

    print("Done.")


if __name__ == '__main__':
    import sys
    sys.exit(main())
