#!/usr/bin/env python3
"""
3 heatmaps (one per disease: FHL, LongQT, FEVR margbal).

For each disease, strip the outer logistic and take the inner algebraic formula f.
y-axis : all C(n_causative, 2) pairs of causative loci  (or pairs involving ≥1 formula locus
         when n_causative is large, to keep the plot readable)
x-axis : loci that appear in the formula (causative shown in orange, background in blue)
cell   : the coefficient of that x-locus in  ∂²f / ∂x_i ∂x_j
         → non-zero  ⟹  x_k modulates the i-j interaction  (higher-order interaction)
         A row that has a solid non-zero BACKGROUND but all-zero cells means a PURE
         pairwise interaction (∂²f = constant, independent of all other variables).

Row label colour key:
  green  – pair detected by Feyn AND is a truth pair  (TP)
  red    – pair detected but NOT a truth pair          (FP)
  gold   – truth pair NOT detected                     (FN)
  gray   – neither detected nor truth
"""

import os, itertools, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sympy as sp
from sympy.core.function import AppliedUndef

BASE     = "dataset"
DISEASES = [
    ("fhl_margbal",    "FHL"),
    ("longqt_margbal", "LongQT"),
    ("fevr_margbal",   "FEVR"),
]

# ── helpers ───────────────────────────────────────────────────────────────────

def strip_logreg(expr):
    if isinstance(expr, AppliedUndef) and len(expr.args) == 1:
        return expr.args[0]
    return expr


def collect_multiply_pairs(inner):
    inner_ex = sp.expand(inner)
    pairs = set()
    for term in inner_ex.as_ordered_terms():
        factors = sp.Mul.make_args(term)
        syms = [f for f in factors if isinstance(f, sp.Symbol)]
        if len(syms) > 1:
            for i in range(len(syms)):
                for j in range(i + 1, len(syms)):
                    pairs.add(frozenset([syms[i], syms[j]]))
    return pairs


def pos_label(var_id):
    parts = str(var_id).split(":")
    chrom, pos = parts[0], int(parts[1])
    return f"chr{chrom}:{pos/1e6:.1f}M"


def load_disease(short):
    data_dir = os.path.join(BASE, short)
    d = np.load(os.path.join(data_dir, "genotype_train.npz"), allow_pickle=True)
    ns, nn   = int(d["novel_start"]), int(d["n_novel"])
    var_ids  = [str(x) for x in d["variant_ids"]]
    caus_cols = list(range(ns, ns + nn))
    caus_set  = set(caus_cols)

    mp  = os.path.join(data_dir, "feyn_model_c30.pickle")
    pp  = os.path.join(data_dir, "feyn_prefilter_idx_c30.npy")
    with open(mp, "rb") as f: best = pickle.load(f)
    pfx = np.load(pp)
    expr  = best.sympify(signif=3)
    inner = strip_logreg(expr)

    # map symbol → original column
    sym_to_col = {}
    for sym in inner.free_symbols:
        n = str(sym)
        if n.startswith("v") and n[1:].isdigit():
            i = int(n[1:])
            if i < len(pfx):
                sym_to_col[sym] = int(pfx[i])

    formula_syms  = sorted(sym_to_col, key=str)   # all symbols in formula
    formula_cols  = [sym_to_col[s] for s in formula_syms]
    is_causative  = [c in caus_set for c in formula_cols]

    # detected pairs (multiply pairs or fallback)
    sym_pairs  = collect_multiply_pairs(inner)
    detected   = set()
    for sp_pair in sym_pairs:
        cols = [sym_to_col[s] for s in sp_pair if s in sym_to_col]
        if len(cols) == 2:
            detected.add(frozenset(cols))
    if not detected:
        caus_in_f = sorted(c for c in formula_cols if c in caus_set)
        detected  = {frozenset([a, b]) for a, b in itertools.combinations(caus_in_f, 2)}

    # truth pairs (baseline)
    base_short  = short.replace("_margbal", "")
    truth_pairs = set()
    for split in ["train", "val", "test"]:
        d2   = np.load(os.path.join(BASE, base_short, f"genotype_{split}.npz"), allow_pickle=True)
        X_s  = d2["X"].astype(np.float32)
        y_s  = d2["y"].astype(int)
        for i in range(len(y_s)):
            if y_s[i] == 1:
                fs = frozenset(j for j in caus_set if X_s[i, j] > 0)
                if len(fs) >= 2:
                    for a, b in itertools.combinations(sorted(fs), 2):
                        truth_pairs.add(frozenset([a, b]))

    # which pairs to include on y-axis
    # for large datasets limit to pairs involving ≥1 formula locus
    formula_col_set = set(formula_cols)
    if nn <= 10:
        row_pairs = [(a, b) for a, b in itertools.combinations(caus_cols, 2)]
    else:
        row_pairs = [(a, b) for a, b in itertools.combinations(caus_cols, 2)
                     if a in formula_col_set or b in formula_col_set]

    # compute ∂²inner / ∂x_i ∂x_j for each row pair × formula symbol
    # matrix rows = row_pairs, cols = formula_syms
    col_to_sym = {v: k for k, v in sym_to_col.items()}
    inner_ex   = sp.expand(inner)

    nrows = len(row_pairs)
    ncols = len(formula_syms)
    mat_coeff = np.zeros((nrows, ncols))  # coefficient of formula_sym[j] in ∂²
    mat_const = np.zeros(nrows)           # constant part of ∂²

    for ri, (ci, cj) in enumerate(row_pairs):
        si = col_to_sym.get(ci)
        sj = col_to_sym.get(cj)
        if si is None or sj is None:
            continue
        d2 = sp.diff(sp.diff(inner_ex, si), sj)
        d2 = sp.expand(d2)
        # constant part
        mat_const[ri] = float(d2.subs({s: 0 for s in inner.free_symbols}))
        # coefficient of each formula symbol in d2
        for ci2, fs in enumerate(formula_syms):
            mat_coeff[ri, ci2] = float(sp.diff(d2, fs))

    return dict(
        caus_cols=caus_cols,
        var_ids=var_ids,
        formula_syms=formula_syms,
        formula_cols=formula_cols,
        is_causative=is_causative,
        row_pairs=row_pairs,
        mat_coeff=mat_coeff,
        mat_const=mat_const,
        detected=detected,
        truth_pairs=truth_pairs,
    )


# ── plot ──────────────────────────────────────────────────────────────────────

ROW_COLORS = {
    "TP": "#2CA02C",
    "FP": "#D62728",
    "FN": "#FF7F0E",
    "none": "#AAAAAA",
}


def draw_heatmap(ax, data, title):
    row_pairs   = data["row_pairs"]
    mat_coeff   = data["mat_coeff"]   # (n_pairs, n_formula_syms)
    mat_const   = data["mat_const"]   # (n_pairs,)
    detected    = data["detected"]
    truth_pairs = data["truth_pairs"]
    formula_syms = data["formula_syms"]
    formula_cols = data["formula_cols"]
    is_causative = data["is_causative"]
    var_ids      = data["var_ids"]

    nrows, ncols = mat_coeff.shape
    vmax = max(np.abs(mat_coeff).max(), 1e-6)

    # background image
    im = ax.imshow(mat_coeff, aspect="auto",
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="none")

    # highlight rows where ∂² is non-zero constant (pairwise interaction, no HO)
    for ri, (ci, cj) in enumerate(row_pairs):
        pair_fs = frozenset([ci, cj])
        is_det  = pair_fs in detected
        is_tr   = pair_fs in truth_pairs
        if is_det and is_tr:
            tag = "TP"
        elif is_det and not is_tr:
            tag = "FP"
        elif not is_det and is_tr:
            tag = "FN"
        else:
            tag = "none"

        # row background tint for non-zero constant partials
        if abs(mat_const[ri]) > 1e-9:
            ax.axhspan(ri - 0.5, ri + 0.5,
                       facecolor=ROW_COLORS[tag], alpha=0.25, zorder=0)

        # left-edge colour bar
        ax.add_patch(plt.Rectangle((-0.5, ri - 0.5), 0.25, 1.0,
                                    fc=ROW_COLORS[tag], ec="none",
                                    transform=ax.transData, zorder=3, clip_on=False))

    # x-tick labels
    xlabels = []
    for sym, col, caus in zip(formula_syms, formula_cols, is_causative):
        lbl = pos_label(var_ids[col]) if col < len(var_ids) else str(sym)
        xlabels.append(lbl)

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=7)
    for tk, caus in zip(ax.get_xticklabels(), is_causative):
        tk.set_color("#CC5500" if caus else "#2255AA")

    # y-tick labels (every row label for small, sampled for large)
    if nrows <= 30:
        yticks = list(range(nrows))
        ylabels = [f"({pos_label(var_ids[a])}, {pos_label(var_ids[b])})"
                   for a, b in row_pairs]
    else:
        step  = max(1, nrows // 20)
        yticks = list(range(0, nrows, step))
        ylabels = [f"({pos_label(var_ids[row_pairs[i][0]])}, "
                   f"{pos_label(var_ids[row_pairs[i][1]])})" for i in yticks]

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=6)

    # colour y-tick labels by row type
    ytick_pairs = [row_pairs[i] for i in yticks]
    for tk, (ci, cj) in zip(ax.get_yticklabels(), ytick_pairs):
        pair_fs = frozenset([ci, cj])
        is_det  = pair_fs in detected
        is_tr   = pair_fs in truth_pairs
        if is_det and is_tr:   tk.set_color(ROW_COLORS["TP"])
        elif is_det:           tk.set_color(ROW_COLORS["FP"])
        elif is_tr:            tk.set_color(ROW_COLORS["FN"])
        else:                  tk.set_color(ROW_COLORS["none"])

    ax.set_xlabel("Formula loci  (causative = orange, background = blue)", fontsize=8)
    ax.set_ylabel("Locus pairs", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")

    return im


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 9))

    last_im = None
    for ci, (short, title) in enumerate(DISEASES):
        data = load_disease(short)
        im   = draw_heatmap(axes[ci], data, title)
        last_im = im

    # colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(last_im, cax=cbar_ax,
                 label=r"Coeff. of $x_k$ in $\partial^2 f / \partial x_i \partial x_j$")

    legend_elems = [
        Patch(fc=ROW_COLORS["TP"], alpha=0.7, label="Detected pair — TP"),
        Patch(fc=ROW_COLORS["FP"], alpha=0.7, label="Detected pair — FP"),
        Patch(fc=ROW_COLORS["FN"], alpha=0.7, label="Truth pair — not detected (FN)"),
        Patch(fc=ROW_COLORS["none"], alpha=0.7, label="Neither detected nor truth"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.46, 0.01))

    fig.suptitle(
        r"Mixed partial derivatives  $\partial^2 f / \partial x_i \partial x_j$  — Feyn (margbal)"
        "\nColoured row = non-zero pairwise interaction;  "
        "coloured cell = higher-order interaction with $x_k$",
        fontsize=10, y=0.99)

    plt.tight_layout(rect=[0, 0.07, 0.91, 0.97])
    fname = "feynMixedPartials.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    print("Saved:", fname)
    plt.close()


if __name__ == "__main__":
    main()
