#!/usr/bin/env python3
"""
3 heatmaps (FHL, LongQT, FEVR margbal) — one per disease.

Axes: causative loci × causative loci  (symmetric matrix).
Cell value: number of Feyn top-8 models (max 8) for which
            ∂²f / ∂x_i ∂x_j ≠ 0  in the pre-activation inner formula,
            i.e. the pair (x_i, x_j) appears as a product term.

Truth pairs are indicated with a green square outline.
"""

import os, itertools, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from sympy import expand, Mul, Symbol
from sympy.core.function import AppliedUndef

BASE     = "dataset"
DISEASES = [
    ("fhl_margbal",    "FHL"),
    ("longqt_margbal", "LongQT"),
    ("fevr_margbal",   "FEVR"),
]
N_MODELS = 8


def strip_logreg(expr):
    if isinstance(expr, AppliedUndef) and len(expr.args) == 1:
        return expr.args[0]
    return expr


def multiply_pairs_orig(inner, pfx):
    """Return detected pairs as frozensets of ORIGINAL column indices."""
    inner_ex = expand(inner)
    pairs = set()
    for term in inner_ex.as_ordered_terms():
        factors = Mul.make_args(term)
        syms = [f for f in factors if isinstance(f, Symbol)]
        if len(syms) > 1:
            for i in range(len(syms)):
                for j in range(i + 1, len(syms)):
                    pairs.add(frozenset([syms[i], syms[j]]))
    result = set()
    for sp in pairs:
        names = [str(s) for s in sp]
        if all(n.startswith("v") and n[1:].isdigit() for n in names):
            idxs = [int(n[1:]) for n in names]
            if all(i < len(pfx) for i in idxs):
                result.add(frozenset(int(pfx[i]) for i in idxs))
    return result


def load_truth_pairs(base_short, caus_set):
    truth = set()
    for split in ["train", "val", "test"]:
        d = np.load(os.path.join(BASE, base_short, f"genotype_{split}.npz"), allow_pickle=True)
        X_s, y_s = d["X"].astype(np.float32), d["y"].astype(int)
        for i in range(len(y_s)):
            if y_s[i] == 1:
                fs = frozenset(j for j in caus_set if X_s[i, j] > 0)
                if len(fs) >= 2:
                    for a, b in itertools.combinations(sorted(fs), 2):
                        truth.add(frozenset([a, b]))
    return truth


def pos_label(var_id):
    parts = str(var_id).split(":")
    chrom, pos = parts[0], int(parts[1])
    return f"{chrom}:{pos/1e6:.1f}M"


def build_count_matrix(short):
    data_dir  = os.path.join(BASE, short)
    d         = np.load(os.path.join(data_dir, "genotype_train.npz"), allow_pickle=True)
    ns, nn    = int(d["novel_start"]), int(d["n_novel"])
    var_ids   = [str(x) for x in d["variant_ids"]]
    caus_cols = list(range(ns, ns + nn))
    caus_set  = set(caus_cols)

    pfx = np.load(os.path.join(data_dir, "feyn_prefilter_idx_c30.npy"))

    with open(os.path.join(data_dir, "feyn_top8_c30.pickle"), "rb") as f:
        models = pickle.load(f)
    models = models[:N_MODELS]

    # count matrix: n_caus × n_caus
    n = len(caus_cols)
    col2idx = {c: i for i, c in enumerate(caus_cols)}
    count = np.zeros((n, n), dtype=int)

    for m in models:
        inner = strip_logreg(m.sympify(signif=3))
        pairs = multiply_pairs_orig(inner, pfx)
        for pair in pairs:
            pair_l = sorted(pair)
            if len(pair_l) == 2:
                a, b = pair_l
                if a in col2idx and b in col2idx:
                    i, j = col2idx[a], col2idx[b]
                    count[i, j] += 1
                    count[j, i] += 1

    base_short  = short.replace("_margbal", "")
    truth_pairs = load_truth_pairs(base_short, caus_set)

    return dict(
        count=count,
        caus_cols=caus_cols,
        var_ids=var_ids,
        truth_pairs=truth_pairs,
        col2idx=col2idx,
    )


def draw_heatmap(ax, data, title):
    count    = data["count"]
    caus_cols = data["caus_cols"]
    var_ids   = data["var_ids"]
    truth_pairs = data["truth_pairs"]
    col2idx   = data["col2idx"]
    n = len(caus_cols)

    cmap = mcolors.LinearSegmentedColormap.from_list("wr", ["white", "#CC0000"])
    im = ax.imshow(count, cmap=cmap, vmin=0, vmax=N_MODELS,
                   aspect="equal", interpolation="none")

    # mark truth pairs with a green square
    for pair in truth_pairs:
        pair_l = sorted(pair)
        if len(pair_l) == 2:
            a, b = pair_l
            if a in col2idx and b in col2idx:
                i, j = col2idx[a], col2idx[b]
                for (ri, ci) in [(i, j), (j, i)]:
                    ax.add_patch(mpatches.Rectangle(
                        (ci - 0.5, ri - 0.5), 1, 1,
                        fill=False, edgecolor="#2CA02C", lw=1.5, zorder=3))

    # annotate non-zero cells
    for i in range(n):
        for j in range(n):
            v = count[i, j]
            if v > 0:
                tc = "white" if v > N_MODELS * 0.6 else "black"
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=7 if n > 10 else 9, color=tc, fontweight="bold")

    labels = [pos_label(var_ids[c]) for c in caus_cols]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=6 if n > 10 else 8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=6 if n > 10 else 8)
    ax.set_xlabel("Locus  j", fontsize=9)
    ax.set_ylabel("Locus  i", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    return im


def main():
    fig, axes = plt.subplots(1, 3, figsize=(18, 7),
                             gridspec_kw={"wspace": 0.35})

    last_im = None
    for ci, (short, title) in enumerate(DISEASES):
        data = build_count_matrix(short)
        im   = draw_heatmap(axes[ci], data, title)
        last_im = im

    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
    cb = fig.colorbar(last_im, cax=cbar_ax)
    cb.set_label(f"# Feyn models detecting pair  (max {N_MODELS})", fontsize=9)
    cb.set_ticks(range(N_MODELS + 1))

    legend_elems = [
        mpatches.Patch(fc="white", ec="#2CA02C", lw=1.5, label="Truth pair (OLIDA)"),
        mpatches.Patch(fc="#CC0000", label=f"Detected in all {N_MODELS} models"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.46, 0.01))

    fig.suptitle(
        f"Pairwise interaction consensus across top-{N_MODELS} Feyn models\n"
        r"Cell = # models with $\partial^2 f / \partial x_i \partial x_j \neq 0$  "
        "(margbal datasets)",
        fontsize=11, y=1.01)

    fname = "feynTop8Heatmap.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    print("Saved:", fname)
    plt.close()


if __name__ == "__main__":
    main()
