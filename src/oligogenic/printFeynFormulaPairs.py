#!/usr/bin/env python3
"""
3-panel figure: Feyn formula + pair detection graph for FHL, LongQT, FEVR margbal.

For each disease:
  - Top: Feyn formula with causative SNPs labelled s_k, background SNPs as b
  - Bottom: circular graph of causative SNPs; edges coloured by detection status
      green  = detected pair that is a truth pair  (TP)
      red    = detected pair NOT in truth           (FP)
      gray   = truth pair not detected              (FN, dashed)
    Nodes highlighted in orange if they appear in the formula.
"""

import os, itertools, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from sympy import expand, Mul, Symbol, latex
from sympy.core.function import AppliedUndef

BASE     = "dataset"
DISEASES = [
    ("fhl_margbal",    "FHL"),
    ("longqt_margbal", "LongQT"),
    ("fevr_margbal",   "FEVR"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def collect_multiply_pairs(expr):
    if isinstance(expr, AppliedUndef) and len(expr.args) == 1:
        expr = expr.args[0]
    expr = expand(expr)
    pairs = set()
    for term in expr.as_ordered_terms():
        factors = Mul.make_args(term)
        syms = [f for f in factors if isinstance(f, Symbol)]
        if len(syms) > 1:
            for i in range(len(syms)):
                for j in range(i + 1, len(syms)):
                    pairs.add(frozenset([syms[i], syms[j]]))
    return pairs


def pos_label(var_id):
    parts = str(var_id).split(":")
    chrom, pos = parts[0], int(parts[1])
    mb = pos / 1_000_000
    return f"chr{chrom}:{mb:.1f}M"


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
    expr = best.sympify(signif=3)

    inner = expr.args[0] if isinstance(expr, AppliedUndef) and len(expr.args) == 1 else expr

    # Map each symbol → original column
    sym_to_col = {}
    for sym in inner.free_symbols:
        n = str(sym)
        if n.startswith("v") and n[1:].isdigit():
            i = int(n[1:])
            if i < len(pfx):
                sym_to_col[sym] = int(pfx[i])

    formula_cols = set(sym_to_col.values())

    # Detected pairs: multiply pairs → map to original cols
    sym_pairs = collect_multiply_pairs(expr)
    detected  = []
    for sp in sym_pairs:
        cols = [sym_to_col[s] for s in sp if s in sym_to_col]
        if len(cols) == 2:
            detected.append(frozenset(cols))

    # Fallback (no multiply pairs): pairs among causative formula SNPs
    if not detected:
        caus_in_formula = sorted(c for c in formula_cols if c in caus_set)
        detected = [frozenset([a, b]) for a, b in itertools.combinations(caus_in_formula, 2)]

    # Truth pairs from all splits (canonical baseline)
    base_short = short.replace("_margbal", "")
    truth_pairs = set()
    for split in ["train", "val", "test"]:
        d2 = np.load(os.path.join(BASE, base_short, f"genotype_{split}.npz"), allow_pickle=True)
        X_s, y_s = d2["X"].astype(np.float32), d2["y"].astype(int)
        for i in range(len(y_s)):
            if y_s[i] == 1:
                fs = frozenset(j for j in caus_set if X_s[i, j] > 0)
                if len(fs) >= 2:
                    for a, b in itertools.combinations(sorted(fs), 2):
                        truth_pairs.add(frozenset([a, b]))

    # LaTeX formula with readable labels
    subs = {}
    for sym, col in sym_to_col.items():
        if col in caus_set:
            k = caus_cols.index(col) + 1
            subs[sym] = Symbol(f"s_{{{k}}}")
        else:
            subs[sym] = Symbol("b")
    formula_latex = latex(inner.subs(subs))

    return dict(
        short=short,
        caus_cols=caus_cols,
        var_ids=var_ids,
        formula_cols=formula_cols,
        detected=detected,
        truth_pairs=truth_pairs,
        formula_latex=formula_latex,
    )


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_formula(ax, formula_latex, title):
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold")
    try:
        ax.text(0.5, 0.5, r"$\sigma\!\left(" + formula_latex + r"\right)$",
                ha="center", va="center", fontsize=8.5,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F5", ec="#AAAAAA", lw=0.8))
    except Exception:
        ax.text(0.5, 0.5, formula_latex, ha="center", va="center", fontsize=7,
                transform=ax.transAxes, wrap=True)


def draw_graph(ax, data, subtitle):
    caus_cols   = data["caus_cols"]
    var_ids     = data["var_ids"]
    formula_cols = data["formula_cols"]
    detected    = [frozenset(p) for p in data["detected"]]
    truth_pairs = data["truth_pairs"]
    det_set     = set(detected)

    n      = len(caus_cols)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    pos    = {c: (np.cos(a), np.sin(a)) for c, a in zip(caus_cols, angles)}

    # truth FN edges
    for pair in truth_pairs:
        if pair not in det_set:
            a, b = sorted(pair)
            if a in pos and b in pos:
                ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                        color="#CCCCCC", lw=0.9, ls="--", zorder=1, alpha=0.8)

    # detected edges
    for pair in detected:
        a, b  = sorted(pair)
        is_tp = pair in truth_pairs
        col   = "#2CA02C" if is_tp else "#D62728"
        if a in pos and b in pos:
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                    color=col, lw=2.8, zorder=2)

    # nodes
    for c in caus_cols:
        x, y = pos[c]
        fc   = "#FF7F0E" if c in formula_cols else "#AEC7E8"
        ax.scatter(x, y, s=300, c=fc, zorder=3, edgecolors="k", linewidths=0.7)

    # labels — nudge outward
    for c in caus_cols:
        x, y = pos[c]
        lbl  = pos_label(var_ids[c])
        ax.text(x * 1.55, y * 1.55, lbl,
                fontsize=6, ha="center", va="center", zorder=4)

    n_tp = sum(1 for p in detected if p in truth_pairs)
    n_fp = len(detected) - n_tp
    n_fn = len(truth_pairs) - n_tp
    ax.set_title(f"{subtitle}  (TP={n_tp}  FP={n_fp}  FN={n_fn}  K={len(truth_pairs)})",
                 fontsize=9)
    ax.set_xlim(-2.1, 2.1); ax.set_ylim(-2.1, 2.1)
    ax.set_aspect("equal"); ax.axis("off")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for ci, (short, title) in enumerate(DISEASES):
        data = load_disease(short)
        draw_formula(axes[0, ci], data["formula_latex"], title)
        draw_graph(axes[1, ci], data, title)

    # legend
    legend_elems = [
        Line2D([0], [0], color="#2CA02C", lw=2.5, label="Detected pair — TP"),
        Line2D([0], [0], color="#D62728", lw=2.5, label="Detected pair — FP"),
        Line2D([0], [0], color="#CCCCCC", lw=1.0, ls="--", label="Truth pair — FN"),
        mpatches.Patch(fc="#FF7F0E", ec="k", label="SNP in formula"),
        mpatches.Patch(fc="#AEC7E8", ec="k", label="SNP not in formula"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("Feyn formula pair detection — margbal datasets", fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    fname = "feynFormulaPairs.png"
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    print("Saved:", fname)
    plt.close()


if __name__ == "__main__":
    main()
