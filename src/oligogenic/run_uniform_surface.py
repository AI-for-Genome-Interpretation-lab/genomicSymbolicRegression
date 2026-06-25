#!/usr/bin/env python3
"""
Sweep Feyn + PLINK1.9 pair-detection Jaccard across a K x n_per_pair grid using
the longqt_uniform construction (uniformly balanced pairs) generated in memory.

For each (K, n_per_pair) cell:
  1. Build a synthetic dataset (K causative pairs, n_per_pair positives each,
     2 pseudo-controls per positive, margbal style) on the FULL LongQT margbal
     background (17 674 features).
  2. Prefilter features: top-2500 by |corr(X,y)| ∪ all 20 causative cols.
  3. Run Feyn (classification, BIC, add+multiply only) — extract multiplicative
     pairs from sympify formula; compute pair Jaccard vs the K truth pairs.
  4. Run PLINK1.9 --fast-epistasis on the same prefiltered subset; collect
     pairs with p<0.05; compute Jaccard.
  5. Append a row to uniform_surface.csv.
"""

import os, warnings, time, subprocess, tempfile, shutil
import numpy as np
import pandas as pd
from sympy import expand, Mul, Symbol, diff, simplify
from sympy.core.function import AppliedUndef
warnings.filterwarnings("ignore")

import feyn  # noqa: E402

SRC                = "dataset/longqt_margbal"
OUT_CSV            = "uniform_surface_dense.csv"
SEED               = 42
N_EPOCHS           = 30
MAX_COMPLEXITY     = 30
MAX_FEYN_FEATURES  = 2500
EPI_P_THRESH       = 0.05
PLINK19            = "plink1.9"
ALLELES            = {0: ('A','A'), 1: ('A','C'), 2: ('C','C')}
RUN_PLINK          = False   # Feyn-only dense sweep

K_VALS             = list(range(2, 51))         # 2..50 step 1  (49 values)
NPP_VALS           = list(range(2, 101, 2))     # 2..100 step 2 (50 values)

# 9 LongQT margbal truth pairs as a stable prefix, then extend with all other
# C(20,2) pairs (sorted lex) so K up to 190 is supported.
_LONGQT_REAL = [
    # Disjoint (no shared SNPs) — K=2..5 use only these:
    (17659, 17660),
    (17654, 17655),
    (17657, 17658),
    (17663, 17664),
    (17661, 17662),
    # Shared-SNP pairs (the v17661/v17662/v17663/v17664 clique extras):
    (17661, 17664),
    (17662, 17664),
    (17661, 17663),
    (17662, 17663),
]
_CAUS_COLS = list(range(17654, 17654 + 20))  # 20 LongQT causative SNPs
import itertools as _it
_seen = set(frozenset(p) for p in _LONGQT_REAL)
_extra = [p for p in _it.combinations(_CAUS_COLS, 2)
          if frozenset(p) not in _seen]
_extra.sort()
LONGQT_PAIRS = _LONGQT_REAL + _extra
assert len(LONGQT_PAIRS) >= max(K_VALS), f"need {max(K_VALS)} pairs, have {len(LONGQT_PAIRS)}"


# ── Data ──────────────────────────────────────────────────────────────────────

def load_src():
    Xs, ys = [], []
    novel_start = n_novel = None
    for sp in ("train", "val", "test"):
        d = np.load(os.path.join(SRC, f"genotype_{sp}.npz"), allow_pickle=True)
        Xs.append(d["X"]); ys.append(d["y"])
        novel_start = int(d["novel_start"]); n_novel = int(d["n_novel"])
    return np.concatenate(Xs), np.concatenate(ys), novel_start, n_novel


def build_uniform(X_src, y_src, novel_start, n_novel, K, npp, rng):
    pairs           = LONGQT_PAIRS[:K]
    causative_cols  = list(range(novel_start, novel_start + n_novel))
    neg_pool        = X_src[y_src == 0]

    pos_list, pseudo_list = [], []
    for a, b in pairs:
        for _ in range(npp):
            bg = neg_pool[rng.integers(0, len(neg_pool))].copy()
            for c in causative_cols: bg[c] = 0
            bg[a] = 1; bg[b] = 1
            pos_list.append(bg)
            for snp in (a, b):
                pc = neg_pool[rng.integers(0, len(neg_pool))].copy()
                for c in causative_cols: pc[c] = 0
                pc[snp] = 1
                pseudo_list.append(pc)

    X_out = np.concatenate(
        [np.array(pos_list,    dtype=np.int8),
         np.array(pseudo_list, dtype=np.int8)], axis=0)
    y_out = np.concatenate(
        [np.ones(len(pos_list),    dtype=int),
         np.zeros(len(pseudo_list), dtype=int)])
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm], [frozenset(p) for p in pairs]


def prefilter(X_norm, y, causative_cols, max_feats):
    corr = np.array([
        abs(np.corrcoef(X_norm[:, i], y)[0, 1]) if X_norm[:, i].std() > 0 else 0.0
        for i in range(X_norm.shape[1])
    ])
    top = np.argsort(corr)[::-1][:max_feats]
    return np.array(sorted(set(top.tolist()) | set(causative_cols)))


# ── Feyn ──────────────────────────────────────────────────────────────────────

def collect_multiplied_pairs(expr):
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


def collect_hessian_pairs(expr):
    """Pairs (s_i, s_j) where d^2/dx_i dx_j (expr) != 0 symbolically.
    Catches interactions through nonlinear wrappers (e.g. sigmoid for
    classification) that collect_multiplied_pairs would miss."""
    if isinstance(expr, AppliedUndef) and len(expr.args) == 1:
        expr = expr.args[0]
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


def _map_sym_pairs(sym_pairs, prefilter_idx):
    out = set()
    for sp in sym_pairs:
        names = [str(s) for s in sp]
        if all(n.startswith("v") and n[1:].isdigit() for n in names):
            sub_idx = [int(n[1:]) for n in names]
            if all(i < len(prefilter_idx) for i in sub_idx):
                orig = tuple(int(prefilter_idx[i]) for i in sub_idx)
                out.add(frozenset(orig))
    return out


def feyn_extract(best, prefilter_idx):
    """Return (mul_pairs, hess_pairs, formula_feature_cols).
    mul_pairs  = pairs that appear multiplied together in expand(expr)
    hess_pairs = pairs where d^2/dx_i dx_j (expr) != 0 (incl. nonlinear interactions)
    """
    try:
        expr = best.sympify(signif=3)
    except Exception:
        try:
            expr = best.sympify()
        except Exception:
            return set(), set(), set()
    mul_pairs  = _map_sym_pairs(collect_multiplied_pairs(expr), prefilter_idx)
    hess_pairs = _map_sym_pairs(collect_hessian_pairs(expr),    prefilter_idx)
    formula_cols = set()
    for feat in best.features:
        if feat.startswith("v") and feat[1:].isdigit():
            i = int(feat[1:])
            if i < len(prefilter_idx):
                formula_cols.add(int(prefilter_idx[i]))
    return mul_pairs, hess_pairs, formula_cols


def run_feyn(X_sub, y, prefilter_idx):
    cols = [f"v{i}" for i in range(X_sub.shape[1])]
    df = pd.DataFrame(X_sub, columns=cols); df["y"] = y
    ql = feyn.QLattice(random_seed=42)
    models = ql.auto_run(
        df, output_name="y", kind="classification",
        n_epochs=N_EPOCHS, criterion="bic",
        max_complexity=MAX_COMPLEXITY,
        function_names=["add", "multiply"])
    best = models[0]
    mul_pairs, hess_pairs, formula_cols = feyn_extract(best, prefilter_idx)
    return mul_pairs, hess_pairs, formula_cols


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


def run_plink(X_sub_dosage, y, prefilter_idx, p_thresh=EPI_P_THRESH):
    """X_sub_dosage is in [0, 2] integer space (not normalized)."""
    tmpdir = tempfile.mkdtemp(prefix="uni_epi_")
    try:
        write_ped_map(X_sub_dosage, y, tmpdir, "epi")
        r = subprocess.run(
            [PLINK19, "--ped", "epi.ped", "--map", "epi.map",
             "--fast-epistasis", "--epi1", "1",
             "--allow-no-sex", "--out", "epi"],
            capture_output=True, text=True, cwd=tmpdir)
        cc = os.path.join(tmpdir, "epi.epi.cc")
        if r.returncode != 0 or not os.path.exists(cc):
            return set()
        pairs = set()
        with open(cc) as f:
            for line in f:
                pts = line.split()
                if not pts or pts[0] == "CHR1": continue
                try:
                    s1 = int(pts[1].replace("SNP_", ""))
                    s2 = int(pts[3].replace("SNP_", ""))
                    pv = float(pts[-1]) if pts[-1] != "NA" else 1.0
                    if pv < p_thresh:
                        o1 = int(prefilter_idx[s1])
                        o2 = int(prefilter_idx[s2])
                        pairs.add(frozenset([o1, o2]))
                except Exception:
                    pass
        return pairs
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_src, y_src, novel_start, n_novel = load_src()
    causative_cols = list(range(novel_start, novel_start + n_novel))

    rows = []
    done = set()
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        rows = prev.to_dict("records")
        done = {(int(r["K"]), int(r["n_per_pair"])) for r in rows}
        print(f"Resuming: {len(done)} cells already in {OUT_CSV}")

    for K in K_VALS:
        for npp in NPP_VALS:
            if (K, npp) in done:
                continue
            tag = f"K={K} npp={npp}"
            t0  = time.time()
            rng = np.random.default_rng(SEED + K * 100 + npp)
            X, y, truth_pairs = build_uniform(
                X_src, y_src, novel_start, n_novel, K, npp, rng)
            n_pos = int(y.sum())

            X_norm = X.astype(np.float32) / 2.0
            pre    = prefilter(X_norm, y, causative_cols, MAX_FEYN_FEATURES)
            X_sub  = X_norm[:, pre]
            X_sub_dosage = X.astype(np.int8)[:, pre]

            # Feyn
            t_feyn = time.time()
            mul_pairs, hess_pairs, formula_cols = run_feyn(X_sub, y, pre)
            combo_pairs = mul_pairs | hess_pairs
            j_feyn_strict = jaccard(mul_pairs,   set(truth_pairs))
            j_feyn_hess   = jaccard(hess_pairs,  set(truth_pairs))
            j_feyn_combo  = jaccard(combo_pairs, set(truth_pairs))
            # loose: truth pair detected if BOTH SNPs are in the Feyn formula
            detected_loose = {p for p in truth_pairs if p.issubset(formula_cols)}
            j_feyn_loose  = jaccard(detected_loose, set(truth_pairs))
            t_feyn = time.time() - t_feyn

            # PLINK1.9 (skipped in dense sweep — too slow at high npp/K)
            if RUN_PLINK:
                t_plk = time.time()
                detected_plk = run_plink(X_sub_dosage, y, pre)
                j_plk = jaccard(detected_plk, set(truth_pairs))
                t_plk = time.time() - t_plk
            else:
                detected_plk = set()
                j_plk = float("nan")
                t_plk = 0.0

            elapsed = time.time() - t0
            row = dict(
                K=K, n_per_pair=npp, n_pos=n_pos, n_total=len(y),
                feyn_J_strict=j_feyn_strict,
                feyn_J_hess=j_feyn_hess,
                feyn_J_combo=j_feyn_combo,
                feyn_J_loose=j_feyn_loose,
                plink_J=j_plk,
                feyn_n_pairs=len(mul_pairs),
                feyn_n_hess=len(hess_pairs),
                feyn_n_combo=len(combo_pairs),
                feyn_n_feats=len(formula_cols),
                plink_n=len(detected_plk),
                feyn_s=round(t_feyn,1), plink_s=round(t_plk,1),
                elapsed_s=round(elapsed, 1))
            rows.append(row)
            print(f"[{tag:<12}]  pos={n_pos:4d}  Feyn(strict={j_feyn_strict:.2f} "
                  f"hess={j_feyn_hess:.2f} combo={j_feyn_combo:.2f} "
                  f"loose={j_feyn_loose:.2f} mul={len(mul_pairs)} h={len(hess_pairs)} "
                  f"feats={len(formula_cols)}, {t_feyn:.0f}s)  "
                  f"PLINK(n={len(detected_plk):3d} J={j_plk:.2f}, {t_plk:.0f}s)  "
                  f"tot={elapsed:.0f}s")

            # save incrementally
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(f"\nSaved → {OUT_CSV}")


if __name__ == "__main__":
    main()
