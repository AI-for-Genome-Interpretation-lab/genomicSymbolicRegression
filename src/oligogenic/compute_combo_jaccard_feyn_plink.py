#!/usr/bin/env python3
"""
Pair-level Combo Jaccard: Feyn vs PLINK1.9.

Two comparison modes:
  NATURAL: each method produces its own set of pairs; Jaccard and Recall are
            computed over those sets vs truth.
    Feyn    : take top-n_novel SNPs from feyn_raw_rank, report ALL C(n_novel,2) pairs.
    PLINK1.9: run --fast-epistasis; report all pairs with p < EPI_P_THRESH.

  FAIR@K : both methods output exactly K = n_truth_pairs pairs.
    Feyn    : rank pairs by (rank_i + rank_j); take top-K.
    PLINK1.9: rank pairs by p-value ascending; take top-K.

Datasets with only singleton combos (SCA17) are skipped (K=0).
"""

import os, subprocess, tempfile, shutil, itertools, warnings, pickle
import numpy as np
import pandas as pd
from sympy import simplify, expand, Mul, Symbol
warnings.filterwarnings("ignore")

PLINK19      = "plink1.9"
EPI_P_THRESH = 0.05         # threshold for PLINK natural-output mode
LARGE_THR    = 3000         # datasets wider than this get marginal pre-filter
LARGE_K      = 500          # pre-filter size for large datasets

DATASETS = [
    ("fhl_margbal",       "FHL"),
    ("alport_margbal",    "Alport"),
    ("hypo_margbal",      "Hypo"),
    ("longqt_margbal",    "LongQT"),
    ("fevr_margbal",      "FEVR"),
    ("hypodontia_margbal","Hypodontia"),
]

ALLELES = {0: ('A','A'), 1: ('A','C'), 2: ('C','C')}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_tv(data_dir):
    tr = np.load(os.path.join(data_dir, "genotype_train.npz"), allow_pickle=True)
    va = np.load(os.path.join(data_dir, "genotype_val.npz"),   allow_pickle=True)
    X  = np.concatenate([tr["X"], va["X"]]).astype(np.float32)
    y  = np.concatenate([tr["y"], va["y"]]).astype(int)
    return X, y, int(tr["n_novel"]), int(tr["novel_start"])


def load_olida_combos(data_dir, novel_start, n_novel):
    causative = set(range(novel_start, novel_start + n_novel))
    seen, combos = set(), []
    for split in ["train", "val", "test"]:
        d = np.load(os.path.join(data_dir, f"genotype_{split}.npz"), allow_pickle=True)
        X_s, y_s = d["X"].astype(np.float32), d["y"].astype(int)
        for i in range(len(y_s)):
            if y_s[i] == 1:
                fs = frozenset(j for j in causative if X_s[i, j] > 0)
                if fs and fs not in seen:
                    seen.add(fs); combos.append(fs)
    return combos


def combos_to_pairs(combos):
    pairs = set()
    for c in combos:
        for a, b in itertools.combinations(sorted(c), 2):
            pairs.add(frozenset([a, b]))
    return pairs


def jaccard_sets(detected, truth):
    detected, truth = set(detected), set(truth)
    if not detected and not truth: return 1.0
    if not detected or not truth:  return 0.0
    return len(detected & truth) / len(detected | truth)


def recall_sets(detected, truth):
    if not truth: return float("nan")
    return len(set(detected) & set(truth)) / len(truth)


def jaccard_at_k(ranked_pairs, truth_pairs, k):
    return jaccard_sets(ranked_pairs[:k], truth_pairs)


def recall_at_k(ranked_pairs, truth_pairs, k):
    return recall_sets(ranked_pairs[:k], truth_pairs)


# ── Feyn ──────────────────────────────────────────────────────────────────────

def collect_multiplied_pairs(expr):
    """Return frozensets of symbol pairs that appear as co-factors in the formula.
    Strips outer function wrappers (e.g. logreg from classification) before expanding."""
    from sympy.core.function import AppliedUndef
    # Strip outer wrapper like logreg(...) so expand can see inside
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


def feyn_pairs_from_model(data_dir):
    """
    Load pickled Feyn model (c30 preferred), call sympify, extract multiply pairs.
    Maps symbol names (v0, v1, ...) back to original SNP indices via prefilter_idx.
    Returns list of frozenset([orig_i, orig_j]) or [] if not available.
    """
    model_file = prefilter_file = None
    for cx in ("_c30", "_c10", ""):
        mp = os.path.join(data_dir, f"feyn_model{cx}.pickle")
        pp = os.path.join(data_dir, f"feyn_prefilter_idx{cx}.npy")
        if os.path.exists(mp) and os.path.exists(pp):
            model_file, prefilter_file = mp, pp
            break
    if model_file is None:
        return []

    try:
        with open(model_file, "rb") as f:
            best = pickle.load(f)
        prefilter_idx = np.load(prefilter_file)
    except Exception:
        return []

    try:
        expr = best.sympify(signif=3)
    except Exception:
        try:
            expr = best.sympify()
        except Exception:
            return []

    sym_pairs = collect_multiplied_pairs(expr)
    result = []
    for sp in sym_pairs:
        names = [str(s) for s in sp]
        if all(n.startswith("v") and n[1:].isdigit() for n in names):
            sub_idx = [int(n[1:]) for n in names]
            if all(i < len(prefilter_idx) for i in sub_idx):
                orig = [int(prefilter_idx[i]) for i in sub_idx]
                result.append(frozenset(orig))
    return result


def feyn_formula_snps(data_dir):
    """Return SNP indices in the Feyn formula (fallback when no model pickle)."""
    for fname in ("feyn_model_features_c30.npy", "feyn_model_features_c10.npy",
                  "feyn_model_features.npy"):
        p = os.path.join(data_dir, fname)
        if os.path.exists(p):
            return [int(x) for x in np.load(p)]
    return []


def feyn_ranked_pairs(data_dir, top_cands=200):
    """Pairs sorted by (rank_i + rank_j) among top-top_cands features."""
    for fname in ("feyn_raw_rank_c10.npy", "feyn_raw_rank.npy"):
        p = os.path.join(data_dir, fname)
        if os.path.exists(p):
            rank = np.load(p)
            break
    else:
        return []

    cands = int(min(top_cands, len(rank)))
    top_feats = rank[:cands]
    pos = {int(snp): i for i, snp in enumerate(rank)}
    scored = []
    for i in range(cands):
        for j in range(i+1, cands):
            a, b = int(top_feats[i]), int(top_feats[j])
            scored.append((pos[a] + pos[b], frozenset([a, b])))
    scored.sort(key=lambda x: x[0])
    return [p for _, p in scored]


# ── PLINK1.9 ──────────────────────────────────────────────────────────────────

def write_ped_map(X, y, tmpdir, name="data"):
    n, p = X.shape
    with open(os.path.join(tmpdir, name + ".map"), "w") as f:
        for i in range(p):
            f.write(f"1\tSNP_{i}\t0\t{i+1}\n")
    with open(os.path.join(tmpdir, name + ".ped"), "w") as f:
        for i in range(n):
            pheno = "2" if y[i] == 1 else "1"
            row = [f"FAM{i}", f"IND{i}", "0", "0", "0", pheno]
            for j in range(p):
                a1, a2 = ALLELES[int(np.clip(round(float(X[i, j])), 0, 2))]
                row += [a1, a2]
            f.write(" ".join(row) + "\n")


def plink_run(X, y, n_novel, novel_start):
    """
    Run --fast-epistasis; return list of (pvalue, frozenset) sorted by p asc.
    For large datasets pre-filter to top-LARGE_K by |corr| + all causative SNPs.
    """
    p = X.shape[1]
    tmpdir = tempfile.mkdtemp(prefix="olida_epi_")
    try:
        if p <= LARGE_THR:
            snp_idx = list(range(p))
        else:
            causative = set(range(novel_start, novel_start + n_novel))
            corr = np.array([
                abs(np.corrcoef(X[:, i] / 2.0, y)[0, 1]) if X[:, i].std() > 0 else 0.0
                for i in range(p)
            ])
            top_corr = np.argsort(corr)[::-1][:LARGE_K].tolist()
            snp_idx = list(dict.fromkeys(top_corr + sorted(causative)))
            print(f"    pre-filter {p}→{len(snp_idx)} SNPs (incl. {n_novel} causative)")

        X_sub = X[:, snp_idx]
        write_ped_map(X_sub, y, tmpdir, "epi")
        r = subprocess.run(
            [PLINK19, "--ped", "epi.ped", "--map", "epi.map",
             "--fast-epistasis", "--epi1", "1",   # collect ALL, filter in Python
             "--allow-no-sex", "--out", "epi"],
            capture_output=True, text=True, cwd=tmpdir)

        cc = os.path.join(tmpdir, "epi.epi.cc")
        if r.returncode != 0 or not os.path.exists(cc):
            return []

        rows = []
        with open(cc) as f:
            for line in f:
                pts = line.split()
                if not pts or pts[0] == "CHR1": continue
                try:
                    s1 = int(pts[1].replace("SNP_", ""))
                    s2 = int(pts[3].replace("SNP_", ""))
                    pv = float(pts[-1]) if pts[-1] != "NA" else 1.0
                    o1 = snp_idx[s1] if s1 < len(snp_idx) else s1
                    o2 = snp_idx[s2] if s2 < len(snp_idx) else s2
                    rows.append((pv, frozenset([o1, o2])))
                except: pass

        rows.sort(key=lambda x: x[0])
        return rows

    except Exception as e:
        print(f"    PLINK1.9 error: {e}")
        return []
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    base    = "dataset"
    rows    = []
    K_CANDS = 200   # candidate pool for fair@K Feyn pair generation

    for short, label in DATASETS:
        data_dir   = os.path.join(base, short)
        print(f"\n{'='*55}\n{label}")

        X, y, n_novel, novel_start = load_tv(data_dir)

        # Truth pairs always come from the canonical baseline disease dataset
        # (strip _margbal, _mb_*, _fp* suffixes so variants compare against
        # the full original truth, not just what's in the resampled dataset).
        import re as _re
        truth_short = _re.sub(r'(_margbal|_mb_\w+|_fp\w+|_fpbal)$', '', short)
        truth_dir = os.path.join(base, truth_short)
        combos      = load_olida_combos(truth_dir, novel_start, n_novel)
        truth_pairs = combos_to_pairs(combos)
        K           = len(truth_pairs)

        print(f"  p={X.shape[1]}, n_novel={n_novel}, combos={len(combos)}, "
              f"truth_pairs={K}")

        row = {"dataset": label, "n_novel": n_novel,
               "n_combos": len(combos), "K": K}

        if K == 0:
            print("  All singleton combos — pair metric N/A")
            for m in ["Feyn_J","Feyn_R","Feyn_n",
                      "PLINK_J","PLINK_R","PLINK_n",
                      "FeynK_J","FeynK_R","PLINKK_J","PLINKK_R"]:
                row[m] = float("nan")
            rows.append(row); continue

        # ── Feyn: pairs from collect_multiplied_pairs on sympify formula ──
        # Falls back to "both SNPs in formula" if no model pickle available.
        mul_pairs = feyn_pairs_from_model(data_dir)
        if mul_pairs:
            detected_feyn = set(mul_pairs)
            row["Feyn_J"] = jaccard_sets(detected_feyn, truth_pairs)
            row["Feyn_R"] = recall_sets(detected_feyn, truth_pairs)
            row["Feyn_n"] = len(detected_feyn)
            print(f"  Feyn (multiply pairs={row['Feyn_n']}): "
                  f"J={row['Feyn_J']:.3f}  R={row['Feyn_R']:.3f}  pairs={[sorted(p) for p in detected_feyn]}")
        else:
            feyn_formula = set(feyn_formula_snps(data_dir))
            if feyn_formula:
                detected_feyn = {p for p in truth_pairs if p.issubset(feyn_formula)}
                row["Feyn_J"] = jaccard_sets(detected_feyn, truth_pairs)
                row["Feyn_R"] = recall_sets(detected_feyn, truth_pairs)
                row["Feyn_n"] = len(feyn_formula)
            else:
                row["Feyn_J"] = row["Feyn_R"] = row["Feyn_n"] = float("nan")
            print(f"  Feyn (formula feats={row['Feyn_n']}): "
                  f"J={row['Feyn_J']:.3f}  formula_snps={sorted(feyn_formula)}")

        # ── NATURAL: PLINK = all pairs with p < EPI_P_THRESH ──
        plink_rows = plink_run(X, y, n_novel, novel_start)
        plink_nat  = {pair for pv, pair in plink_rows if pv < EPI_P_THRESH}
        if plink_nat:
            row["PLINK_J"] = jaccard_sets(plink_nat, truth_pairs)
            row["PLINK_R"] = recall_sets(plink_nat, truth_pairs)
            row["PLINK_n"] = len(plink_nat)
        else:
            row["PLINK_J"] = row["PLINK_R"] = float("nan")
            row["PLINK_n"] = 0
        print(f"  PLINK (nat p<{EPI_P_THRESH}, n={row['PLINK_n']}): J={row['PLINK_J']:.3f}, R={row['PLINK_R']:.3f}")

        # ── FAIR@K: both report exactly K pairs ──
        feyn_ranked = feyn_ranked_pairs(data_dir, top_cands=K_CANDS)
        plink_ranked = [pair for _, pair in plink_rows]

        row["FeynK_J"]  = jaccard_at_k(feyn_ranked, truth_pairs, K)
        row["FeynK_R"]  = recall_at_k(feyn_ranked, truth_pairs, K)
        row["PLINKK_J"] = jaccard_at_k(plink_ranked, truth_pairs, K)
        row["PLINKK_R"] = recall_at_k(plink_ranked, truth_pairs, K)
        print(f"  Fair@K={K}: Feyn J={row['FeynK_J']:.3f} R={row['FeynK_R']:.3f} | "
              f"PLINK J={row['PLINKK_J']:.3f} R={row['PLINKK_R']:.3f}")

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("combo_jaccard_feyn_plink.csv", index=False)
    print(f"\nSaved → combo_jaccard_feyn_plink.csv")

    def fmt(x):
        return f"{x:.3f}" if not (isinstance(x, float) and np.isnan(x)) else "  N/A"

    print("\n" + "="*90)
    print(f"NATURAL output  (Feyn: all C(n_novel,2) pairs  |  PLINK: p < {EPI_P_THRESH})")
    print(f"{'Dataset':<22} {'K':>4}  {'Feyn n':>7} {'Feyn J':>7} {'Feyn R':>7}  "
          f"{'PLINK n':>8} {'PLINK J':>8} {'PLINK R':>8}")
    print("-"*80)
    for _, r in df.iterrows():
        k = int(r['K']) if not np.isnan(r['K']) else 0
        fn = int(r['Feyn_n']) if not np.isnan(r['Feyn_n']) else 0
        pn = int(r['PLINK_n']) if not np.isnan(r['PLINK_n']) else 0
        print(f"{r['dataset']:<22} {k:>4}  {fn:>7} {fmt(r['Feyn_J']):>7} {fmt(r['Feyn_R']):>7}  "
              f"{pn:>8} {fmt(r['PLINK_J']):>8} {fmt(r['PLINK_R']):>8}")

    print("\n" + "="*90)
    print(f"FAIR@K  (both methods report exactly K = n_truth_pairs pairs)")
    print(f"{'Dataset':<22} {'K':>4}  {'Feyn J':>8} {'Feyn R':>8}  {'PLINK J':>8} {'PLINK R':>8}")
    print("-"*70)
    for _, r in df.iterrows():
        k = int(r['K']) if not np.isnan(r['K']) else 0
        print(f"{r['dataset']:<22} {k:>4}  {fmt(r['FeynK_J']):>8} {fmt(r['FeynK_R']):>8}  "
              f"{fmt(r['PLINKK_J']):>8} {fmt(r['PLINKK_R']):>8}")


if __name__ == "__main__":
    main()
