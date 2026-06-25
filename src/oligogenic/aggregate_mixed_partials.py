#!/usr/bin/env python3
"""
Aggregate mixed-partials analysis over all margbal-uniform cells.
For each (K, npp) cell:
  1. Load Feyn pickle + prefilter + truth_pairs
  2. Sympify, strip logreg → z
  3. Detect pairs by BOTH methods:
       - mul:  collect_multiplied_pairs(expand(z))
       - D²:   pairs (i,j) where simplify(diff(z, vi, vj)) != 0
  4. Union = mul ∪ D². For polynomial z these coincide; for nonlinear cells they may differ.
  5. For each detected pair, identify:
       - which method(s) detected it
       - in truth?  (truth = LONGQT_PAIRS[:K])
       - sign of D²_ij (constant term)
       - third-locus modifier variables in D²_ij
"""
import os, pickle, itertools, glob
import numpy as np
import pandas as pd
from sympy import expand, Mul, Symbol, diff, simplify
from sympy.core.function import AppliedUndef

PICKLE_ROOT = "pickles_margbal_uniform"


def strip_wrapper(expr):
    if isinstance(expr, AppliedUndef) and len(expr.args) == 1:
        return expr.args[0]
    return expr


def collect_mul_sym_pairs(z):
    z = expand(z)
    pairs = set()
    for term in z.as_ordered_terms():
        factors = Mul.make_args(term)
        syms = [f for f in factors if isinstance(f, Symbol)]
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                pairs.add(frozenset([syms[i], syms[j]]))
    return pairs


def analyze_cell(cell_dir):
    with open(os.path.join(cell_dir, "feyn_model.pickle"), "rb") as f:
        best = pickle.load(f)
    pref  = np.load(os.path.join(cell_dir, "feyn_prefilter.npy"))
    truth = {frozenset([int(a), int(b)])
             for a, b in np.load(os.path.join(cell_dir, "truth_pairs.npy"))}

    try:
        expr = best.sympify(signif=3)
    except Exception:
        expr = best.sympify()
    z = strip_wrapper(expr)
    z_exp = expand(z)
    syms = sorted(z_exp.free_symbols, key=lambda s: str(s))

    def to_orig(s):
        n = str(s)
        if n.startswith("v") and n[1:].isdigit():
            i = int(n[1:])
            if i < len(pref):
                return int(pref[i])
        return None

    # MUL
    mul_sym = collect_mul_sym_pairs(z_exp)
    mul_orig = {frozenset(to_orig(s) for s in p) for p in mul_sym
                if all(to_orig(s) is not None for s in p)}

    # D²
    d2_results = {}   # frozenset(sym_pair) -> {d2 expr, constant, third_loci_orig}
    for vi, vj in itertools.combinations(syms, 2):
        d2 = simplify(diff(z_exp, vi, vj))
        if d2 == 0:
            continue
        d2e = expand(d2)
        const = 0
        third_syms = set()
        for term in d2e.as_ordered_terms():
            if not term.free_symbols:
                const += term
            else:
                third_syms |= (term.free_symbols & set(syms)) - {vi, vj}
        third_orig = {to_orig(s) for s in third_syms if to_orig(s) is not None}
        d2_results[frozenset([vi, vj])] = {
            "d2": d2e, "constant": const, "third_orig": third_orig}

    d2_orig = {frozenset(to_orig(s) for s in p) for p in d2_results
               if all(to_orig(s) is not None for s in p)}

    return {
        "n_features":     len(syms),
        "truth":          truth,
        "mul_orig":       mul_orig,
        "d2_orig":        d2_orig,
        "d2_results":     d2_results,
        "to_orig":        to_orig,
    }


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)


def main():
    cell_dirs = sorted(glob.glob(os.path.join(PICKLE_ROOT, "K*_npp*")))
    if not cell_dirs:
        print(f"No cells found in {PICKLE_ROOT}/ — has the sweep finished?")
        return

    rows = []
    third_locus_freq = {}     # locus -> count of D² where it appears as modifier
    sign_counts = {"+": 0, "-": 0, "0": 0}
    for cd in cell_dirs:
        tag = os.path.basename(cd)
        K   = int(tag.split("_")[0][1:])
        npp = int(tag.split("npp")[1])
        try:
            r = analyze_cell(cd)
        except Exception as e:
            print(f"{tag}: error — {e}")
            continue

        union   = r["mul_orig"] | r["d2_orig"]
        truth   = r["truth"]
        n_only_mul = len(r["mul_orig"] - r["d2_orig"])
        n_only_d2  = len(r["d2_orig"]  - r["mul_orig"])
        n_both     = len(r["mul_orig"] & r["d2_orig"])
        in_truth   = union & truth

        for pair_sym, info in r["d2_results"].items():
            c = info["constant"]
            sign_counts["+" if c > 0 else "-" if c < 0 else "0"] += 1
            for t in info["third_orig"]:
                third_locus_freq[t] = third_locus_freq.get(t, 0) + 1

        rows.append({
            "K": K, "n_per_pair": npp,
            "n_features": r["n_features"],
            "n_mul": len(r["mul_orig"]),
            "n_d2":  len(r["d2_orig"]),
            "n_only_mul": n_only_mul,
            "n_only_d2":  n_only_d2,
            "n_both":     n_both,
            "n_union":    len(union),
            "n_in_truth": len(in_truth),
            "J_mul":   jaccard(r["mul_orig"], truth),
            "J_d2":    jaccard(r["d2_orig"],  truth),
            "J_union": jaccard(union,         truth),
        })

    df = pd.DataFrame(rows)
    df.to_csv("mixed_partials_margbal_uniform.csv", index=False)
    print(df.to_string(index=False))
    print()
    print(f"Saved → mixed_partials_margbal_uniform.csv ({len(df)} cells)")
    print()
    print(f"Sign distribution of D²_ij constants across all pairs:")
    print(f"  positive epistasis: {sign_counts['+']}")
    print(f"  negative epistasis: {sign_counts['-']}")
    print(f"  zero constant:      {sign_counts['0']}")
    print()
    print(f"Third-locus modifiers (orig SNP idx → count of D² where it appears):")
    for t, n in sorted(third_locus_freq.items(), key=lambda kv: -kv[1])[:20]:
        causative_marker = " (CAUSATIVE)" if 17654 <= t <= 17673 else ""
        print(f"  {t:6d}: {n}{causative_marker}")


if __name__ == "__main__":
    main()
