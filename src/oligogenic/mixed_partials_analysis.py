#!/usr/bin/env python3
"""
Mixed-partial-derivative analysis of Feyn formulas, following Estimating
fitness-landscapes literature (Aguilar-Rodriguez et al.; eLife protein G1 SR paper).

For each model:
  1. f = sympify(best), strip outer logreg wrapper → polynomial z(v_i)
  2. For every pair (i,j) of features in formula, compute D²_ij = ∂²z/∂v_i∂v_j
  3. If D²_ij ≠ 0 → pair is epistatically interacting
     - Identify the symbolic form (constant + linear in other loci + higher order)
     - Sign of constant term → positive / negative epistasis at baseline
     - Other loci appearing → "third-locus modifiers" (higher-order epistasis)
     - Multiplicative terms (e.g. v_k·v_l) → triple/quadruple-order interactions
  4. Produce a per-pair table.
"""
import os, pickle, itertools, sys
import numpy as np
from sympy import expand, Mul, Pow, Symbol, diff, simplify
from sympy.core.function import AppliedUndef
sys.path.insert(0, ".")
from compute_combo_jaccard_feyn_plink import combos_to_pairs, load_olida_combos


DATASETS = [
    ("LongQT margbal",     "dataset/longqt_margbal"),
    ("FEVR margbal",       "dataset/fevr_margbal"),
    ("Hypodontia margbal", "dataset/hypodontia_margbal"),
    ("FHL margbal",        "dataset/fhl_margbal"),
    ("Alport margbal",     "dataset/alport_margbal"),
    ("Hypo margbal",       "dataset/hypo_margbal"),
]


def strip_wrapper(expr):
    if isinstance(expr, AppliedUndef) and len(expr.args) == 1:
        return expr.args[0]
    return expr


def load_model(dd):
    for cx in ("_c30", "_c10", ""):
        mp = os.path.join(dd, f"feyn_model{cx}.pickle")
        pp = os.path.join(dd, f"feyn_prefilter_idx{cx}.npy")
        if os.path.exists(mp) and os.path.exists(pp):
            with open(mp, "rb") as f:
                return pickle.load(f), np.load(pp), cx
    return None, None, None


def analyze_pair(z, vi, vj, all_syms):
    """Compute D²_ij and extract structural features."""
    d2 = simplify(diff(z, vi, vj))
    if d2 == 0:
        return None

    d2_expanded = expand(d2)
    # Split into constant + variable parts
    constant_part = 0
    other_loci   = set()
    higher_terms = []   # list of (coeff, frozenset_of_other_loci, sympy_term)
    for term in d2_expanded.as_ordered_terms():
        # Constant?
        if not term.free_symbols:
            constant_part += term
            continue
        # Collect other variables in this term (excluding vi, vj themselves)
        term_vars = (term.free_symbols & all_syms) - {vi, vj}
        if term_vars:
            other_loci |= term_vars
            higher_terms.append((term, frozenset(term_vars)))
        else:
            # Has vi or vj as factor (e.g. v_i term)
            higher_terms.append((term, frozenset()))

    return {
        "d2":            d2_expanded,
        "constant":      constant_part,
        "sign":          ("+" if constant_part > 0 else
                          "-" if constant_part < 0 else "0"),
        "other_loci":    other_loci,
        "higher_terms":  higher_terms,
    }


def main():
    print("=" * 100)
    for label, dd in DATASETS:
        if not os.path.isdir(dd):
            print(f"\n{label}: dir not found — skipping")
            continue
        best, pref, cx = load_model(dd)
        if best is None:
            print(f"\n{label}: no pickled model")
            continue

        try:
            expr_full = best.sympify(signif=3)
        except Exception:
            expr_full = best.sympify()
        z = strip_wrapper(expr_full)
        z_exp = expand(z)
        syms = sorted(z_exp.free_symbols, key=lambda s: str(s))

        # Map v_k → original SNP index
        def to_orig(sym):
            name = str(sym)
            if name.startswith("v") and name[1:].isdigit():
                i = int(name[1:])
                if i < len(pref):
                    return int(pref[i])
            return None

        # Load truth pairs
        tr = np.load(os.path.join(dd, "genotype_train.npz"), allow_pickle=True)
        ns, nn = int(tr["novel_start"]), int(tr["n_novel"])
        truth_pairs = combos_to_pairs(load_olida_combos(dd, ns, nn))

        print(f"\n{label}  ({cx})  truth_pairs={len(truth_pairs)}")
        print(f"  features in z: {len(syms)}")
        print(f"  z = {z_exp}")

        all_syms_set = set(syms)
        epi_pairs = []
        for vi, vj in itertools.combinations(syms, 2):
            res = analyze_pair(z_exp, vi, vj, all_syms_set)
            if res is None:
                continue
            oi, oj = to_orig(vi), to_orig(vj)
            third_orig = sorted({to_orig(s) for s in res["other_loci"]} - {None})
            in_truth = (oi is not None and oj is not None and
                        frozenset([oi, oj]) in truth_pairs)
            epi_pairs.append((vi, vj, oi, oj, res, third_orig, in_truth))

        if not epi_pairs:
            print("  No epistatic pairs detected (all D²_ij = 0 — purely additive z)")
            print(f"  J_strict = 0/{len(truth_pairs)} = 0.000")
            continue

        n_in_truth = sum(1 for _, _, _, _, _, _, t in epi_pairs if t)
        n_det = len(epi_pairs)
        J = n_in_truth / (n_det + len(truth_pairs) - n_in_truth)

        print(f"  → {n_det} epistatic pair(s)  ({n_in_truth} in truth)  J_strict={J:.3f}")
        for vi, vj, oi, oj, res, third_orig, in_truth in epi_pairs:
            mark = " ★ TRUTH" if in_truth else ""
            print(f"     ({vi}={oi}, {vj}={oj}){mark}  sign={res['sign']}  D² = {res['d2']}")
            if third_orig:
                print(f"        third-locus modifiers (orig SNP idx): {third_orig}")
    print()


if __name__ == "__main__":
    main()
