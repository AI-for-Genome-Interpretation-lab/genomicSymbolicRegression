#!/usr/bin/env python3
"""
Compute the combo (mul ∪ hessian) Feyn pair Jaccard for the 3 real OLIDA datasets
(LongQT, FEVR, Hypodontia) using the existing pickled Feyn models in each
dataset directory. Prints values ready to paste into plot_uniform_surface.py REAL dict.
"""
import os, pickle, itertools
import numpy as np
from sympy import expand, Mul, Symbol, diff, simplify
from sympy.core.function import AppliedUndef


DATASETS = [
    ("LongQT",     "dataset/longqt"),
    ("FEVR",       "dataset/fevr"),
    ("Hypodontia", "dataset/hypodontia"),
]


def _strip(expr):
    if isinstance(expr, AppliedUndef) and len(expr.args) == 1:
        return expr.args[0]
    return expr


def collect_mul(expr):
    expr = expand(_strip(expr))
    pairs = set()
    for term in expr.as_ordered_terms():
        factors = Mul.make_args(term)
        syms = [f for f in factors if isinstance(f, Symbol)]
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                pairs.add(frozenset([syms[i], syms[j]]))
    return pairs


def collect_hess(expr):
    expr = _strip(expr)
    syms = sorted(expr.free_symbols, key=lambda s: str(s))
    pairs = set()
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            try:
                d2 = simplify(diff(expr, syms[i], syms[j]))
            except Exception:
                continue
            if d2 != 0:
                pairs.add(frozenset([syms[i], syms[j]]))
    return pairs


def map_to_orig(sym_pairs, prefilter_idx):
    out = set()
    for sp in sym_pairs:
        names = [str(s) for s in sp]
        if all(n.startswith("v") and n[1:].isdigit() for n in names):
            sub = [int(n[1:]) for n in names]
            if all(i < len(prefilter_idx) for i in sub):
                out.add(frozenset(int(prefilter_idx[i]) for i in sub))
    return out


def load_model(dd):
    for cx in ("_c30", "_c10", ""):
        mp = os.path.join(dd, f"feyn_model{cx}.pickle")
        pp = os.path.join(dd, f"feyn_prefilter_idx{cx}.npy")
        if os.path.exists(mp) and os.path.exists(pp):
            with open(mp, "rb") as f:
                return pickle.load(f), np.load(pp), cx
    return None, None, None


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)


def load_truth(data_dir):
    """Truth pairs = all pairs of causative SNPs co-occurring in positives."""
    tr = np.load(os.path.join(data_dir, "genotype_train.npz"), allow_pickle=True)
    novel_start = int(tr["novel_start"]); n_novel = int(tr["n_novel"])
    causative   = set(range(novel_start, novel_start + n_novel))
    seen, combos = set(), []
    for split in ["train", "val", "test"]:
        d = np.load(os.path.join(data_dir, f"genotype_{split}.npz"), allow_pickle=True)
        X_s, y_s = d["X"].astype(np.float32), d["y"].astype(int)
        for i in range(len(y_s)):
            if y_s[i] == 1:
                fs = frozenset(j for j in causative if X_s[i, j] > 0)
                if fs and fs not in seen:
                    seen.add(fs); combos.append(fs)
    pairs = set()
    for c in combos:
        for a, b in itertools.combinations(sorted(c), 2):
            pairs.add(frozenset([a, b]))
    return pairs


def main():
    print(f"{'dataset':<12}  {'cx':<5}  {'mul':>4}  {'hess':>4}  {'combo':>5}  "
          f"{'J_mul':>6}  {'J_hess':>7}  {'J_combo':>7}")
    print("-" * 78)
    for label, dd in DATASETS:
        if not os.path.isdir(dd):
            print(f"{label:<12}  (no dir {dd})")
            continue
        best, pref, cx = load_model(dd)
        if best is None:
            print(f"{label:<12}  (no pickled model)")
            continue
        try:
            expr = best.sympify(signif=3)
        except Exception:
            expr = best.sympify()

        mul_p   = map_to_orig(collect_mul(expr),  pref)
        hess_p  = map_to_orig(collect_hess(expr), pref)
        combo_p = mul_p | hess_p

        truth = load_truth(dd)
        if truth is None:
            print(f"{label:<12}  {cx:<5}  {len(mul_p):4d}  {len(hess_p):4d}  "
                  f"{len(combo_p):5d}  (no truth file — pairs only)")
            print(f"   mul={[sorted(p) for p in mul_p]}")
            print(f"   hess={[sorted(p) for p in hess_p]}")
            continue

        print(f"{label:<12}  {cx:<5}  {len(mul_p):4d}  {len(hess_p):4d}  "
              f"{len(combo_p):5d}  "
              f"{jaccard(mul_p, truth):6.3f}  "
              f"{jaccard(hess_p, truth):7.3f}  "
              f"{jaccard(combo_p, truth):7.3f}")


if __name__ == "__main__":
    main()
