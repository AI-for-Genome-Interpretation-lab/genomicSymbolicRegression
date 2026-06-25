#!/usr/bin/env python3
"""
2x2 heatmap: Combo Jaccard, dom=0 vs dom=1, Feyn vs PLINK1.9 (epi=1 fixed).
Feyn: pairs from collect_multiplied_pairs on the sympify formula.
PLINK1.9: pairs from PLINK2EpiPairs.
Colormap: white -> red.
Saves: epiJaccardDomComparison.png
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sympy import simplify, expand, Mul, Symbol

FOLDER   = "results/run100_synthLarge/"
RUN_NAME = "run100FINAL100_withRF_PLINK2"

import os
# All output PNGs go here (override with FIGOUT). Never the repo root.
OUTDIR = os.environ.get("FIGOUT", "figures/_build")
os.makedirs(OUTDIR, exist_ok=True)

qtl_vals      = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 26, 30, 34, 38, 40, 46, 50]
decoy_vals    = [0, 100, 500, 1000, 1900]
FIXED_SAMPLES = 1000
H, EPI        = 0.6, '1'
DOM_VALS      = ['0', '1']


def jaccard(s1, s2):
    s1, s2 = set(s1), set(s2)
    if not s1 and not s2: return 1.0
    if not s1 or not s2:  return 0.0
    return len(s1 & s2) / len(s1 | s2)


def collect_multiplied_pairs(expr):
    expr = simplify(expand(expr))
    pairs = set()
    for term in expr.as_ordered_terms():
        factors = Mul.make_args(term)
        syms = [f for f in factors if isinstance(f, Symbol)]
        if len(syms) > 1:
            for i in range(len(syms)):
                for j in range(i + 1, len(syms)):
                    pairs.add(frozenset([syms[i], syms[j]]))
    return [tuple(p) for p in pairs]


def convert_pairs(pairs, causative, simphe_causative):
    corresp = {}
    for i, c in enumerate(causative):
        corresp[c] = "SNP_" + str(simphe_causative[i] + 1)
    result = []
    for p in pairs:
        try:
            a = corresp[str(p[0]).replace("SNP", "SNP_")]
            b = corresp[str(p[1]).replace("SNP", "SNP_")]
            result.append(tuple(sorted([a, b])))
        except KeyError:
            pass
    return result


def get_entry(results, qtl, decoys, dom):
    return results.get((H, qtl, dom, EPI, decoys, FIXED_SAMPLES, 10))


def combo_jaccard(v, method):
    epi_pairs_orig = v.get('epiPairs', [])
    if not epi_pairs_orig:
        return np.nan
    truth_pairs = {frozenset(p) for p in epi_pairs_orig}

    if method == 'feyn':
        best = v.get('best')
        if best is None:
            return np.nan
        try:
            expr = best.sympify(signif=3)
        except Exception:
            expr = best.sympify()
        pred_pairs = convert_pairs(
            collect_multiplied_pairs(expr),
            v.get('CAUSATIVE', []),
            v.get('simpheCausative', []))
        detected = {frozenset(p) for p in pred_pairs}
        return jaccard(detected, truth_pairs)

    else:  # plink2
        detected = {frozenset(p) for p in v.get('PLINK2EpiPairs', [])}
        return jaccard(detected, truth_pairs)


def build_matrix(results, method, dom):
    mat = np.full((len(qtl_vals), len(decoy_vals)), 0.0)
    for i, qtl in enumerate(qtl_vals):
        for j, dec in enumerate(decoy_vals):
            v = get_entry(results, qtl, dec, dom=dom)
            if v is None:
                continue
            val = combo_jaccard(v, method)
            mat[i, j] = val if not np.isnan(val) else 0.0
    return mat


def main():
    results = pickle.load(open(FOLDER + RUN_NAME + ".pickle", "rb"))
    print("Loaded %d entries" % len(results))

    cmap = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "#CC0000"])

    methods    = ['feyn', 'plink2']
    col_labels = ['Feyn', 'PLINK1.9']
    row_labels = ['dom=0, epi=1', 'dom=1, epi=1']

    fig, axs = plt.subplots(2, 2, figsize=(14, 6), sharey=True, sharex='col',
                            layout="constrained")
    fig.suptitle(
        "Jaccard score  |  H=%.1f  epi=%s  n=%d" % (
            H, EPI, FIXED_SAMPLES),
        fontsize=18, fontname="Arial", y=0.98)

    im = None
    for ri, dom in enumerate(DOM_VALS):
        for ci, (method, cl) in enumerate(zip(methods, col_labels)):
            ax = axs[ri, ci]
            mat = build_matrix(results, method, dom)
            im = ax.imshow(mat.T, cmap=cmap, vmin=0, vmax=1,
                           aspect='equal', origin='upper')
            if ri == 0:
                ax.set_title(cl, fontsize=16)
            if ci == 0:
                ax.set_ylabel(row_labels[ri] + "\nDecoys", fontsize=15)
                ax.set_yticks(range(len(decoy_vals)))
                ax.set_yticklabels(decoy_vals, fontsize=13)
            if ri == 1:
                ax.set_xlabel("QTLs", fontsize=15)
                ax.set_xticks(range(len(qtl_vals)))
                ax.set_xticklabels(qtl_vals, rotation=45, fontsize=13)

    cb = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02,
                      pad=0.02, label="Jaccard")
    cb.ax.tick_params(labelsize=13)
    cb.set_label("Jaccard", fontsize=15)

    fname = "epiJaccardDomComparison.png"
    plt.savefig(os.path.join(OUTDIR, fname), dpi=300)
    print("Saved:", fname)
    plt.close()


if __name__ == '__main__':
    main()
