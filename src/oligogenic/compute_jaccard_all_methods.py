#!/usr/bin/env python3
"""
Compute SNP Jaccard@K for Feyn, Feyn rank→Logistic, Ridge, Lasso, MLP
across all 12 oligogenic datasets (baseline, fp001, fp005, fpbal).

Rankings:
  Feyn / Feyn rank→Logistic : feyn_raw_rank_c10.npy or feyn_raw_rank.npy
  Lasso (L1)                : |coef_| from LogisticRegression(penalty='l1')
  Ridge (L2)                : |coef_| from LogisticRegression(penalty='l2')
  MLP                       : sum of |first-layer weights| per feature
                              (after pre-filtering to top-2500 by |corr|)
"""

import os, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

MAX_FEYN_FEATURES = 2500

DATASETS = [
    ("sca17",        "SCA17 baseline"),
    ("sca17_fp001",  "SCA17 fp001"),
    ("sca17_fp005",  "SCA17 fp005"),
    ("sca17_fpbal",  "SCA17 fpbal"),
    ("fhl",          "FHL baseline"),
    ("fhl_fp001",    "FHL fp001"),
    ("fhl_fp005",    "FHL fp005"),
    ("fhl_fpbal",    "FHL fpbal"),
    ("alport",       "Alport baseline"),
    ("alport_fp001", "Alport fp001"),
    ("alport_fp005", "Alport fp005"),
    ("alport_fpbal", "Alport fpbal"),
]


def load_tv(data_dir):
    tr = np.load(os.path.join(data_dir, "genotype_train.npz"), allow_pickle=True)
    va = np.load(os.path.join(data_dir, "genotype_val.npz"),   allow_pickle=True)
    X = np.concatenate([tr["X"], va["X"]]).astype(np.float32)
    y = np.concatenate([tr["y"], va["y"]]).astype(int)
    n_novel     = int(tr["n_novel"])
    novel_start = int(tr["novel_start"])
    return X, y, n_novel, novel_start


def jaccard(s1, s2):
    s1, s2 = set(s1), set(s2)
    if not s1 and not s2: return 1.0
    if not s1 or not s2:  return 0.0
    return len(s1 & s2) / len(s1 | s2)


def load_olida_combos(data_dir, novel_start, n_novel):
    causative_set = set(range(novel_start, novel_start + n_novel))
    seen, combos = set(), []
    for split in ["train", "val", "test"]:
        d = np.load(os.path.join(data_dir, f"genotype_{split}.npz"), allow_pickle=True)
        X, y = d["X"].astype(np.float32), d["y"].astype(int)
        for i in range(len(y)):
            if y[i] == 1:
                fs = frozenset(j for j in causative_set if X[i, j] > 0)
                if fs and fs not in seen:
                    seen.add(fs); combos.append(fs)
    return combos


def combo_jaccard(ranked, olida_combos, k):
    top_k = set(ranked[:k])
    rec = sum(1 for c in olida_combos if c.issubset(top_k))
    return rec / len(olida_combos) if olida_combos else 0.0


def prefilter(X, y, max_feats):
    if X.shape[1] <= max_feats:
        return np.arange(X.shape[1])
    corr = np.array([
        abs(np.corrcoef(X[:, i], y)[0, 1]) if X[:, i].std() > 0 else 0.0
        for i in range(X.shape[1])
    ])
    return np.argsort(corr)[::-1][:max_feats]


def feyn_rank(data_dir):
    for fname in ("feyn_raw_rank_c10.npy", "feyn_raw_rank.npy"):
        p = os.path.join(data_dir, fname)
        if os.path.exists(p):
            return np.load(p)
    return None


def lasso_rank(X, y):
    Xn = X / 2.0
    clf = LogisticRegression(penalty="l1", solver="saga", C=0.1,
                             max_iter=3000, random_state=42)
    clf.fit(Xn, y)
    return np.argsort(np.abs(clf.coef_[0]))[::-1]


def ridge_rank(X, y):
    Xn = X / 2.0
    clf = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0,
                             max_iter=1000, random_state=42)
    clf.fit(Xn, y)
    return np.argsort(np.abs(clf.coef_[0]))[::-1]


def mlp_rank(X, y, max_feats=MAX_FEYN_FEATURES):
    pre_idx = prefilter(X / 2.0, y, max_feats)
    Xs = X[:, pre_idx] / 2.0
    sc = StandardScaler()
    Xs = sc.fit_transform(Xs)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                        random_state=42, early_stopping=True)
    mlp.fit(Xs, y)
    # first-layer |weights| summed over hidden units → importance per input feature
    imp_sub = np.abs(mlp.coefs_[0]).sum(axis=1)
    # fill full-space importance (0 for features not in pre-filter)
    imp_full = np.zeros(X.shape[1])
    imp_full[pre_idx] = imp_sub
    # rank: pre-filter features first (by imp), then rest
    ranked_pre  = pre_idx[np.argsort(imp_sub)[::-1]]
    rest = np.array([i for i in np.argsort(imp_full)[::-1] if i not in set(pre_idx.tolist())])
    return np.concatenate([ranked_pre, rest])


def main():
    base = "dataset"
    rows = []

    for short, label in DATASETS:
        data_dir = os.path.join(base, short)
        print(f"\n{'='*55}")
        print(f"{label}")

        X, y, n_novel, novel_start = load_tv(data_dir)
        causative = set(range(novel_start, novel_start + n_novel))
        combos    = load_olida_combos(data_dir, novel_start, n_novel)
        k         = n_novel
        print(f"  p={X.shape[1]}, n_novel={n_novel}, n_combos={len(combos)}")

        row = {"dataset": label, "n_novel": n_novel, "n_combos": len(combos)}

        # Feyn / Feyn rank→Logistic (same ranking)
        fr = feyn_rank(data_dir)
        if fr is not None:
            row["Feyn_snp"]   = jaccard(set(fr[:k]), causative)
            row["Feyn_combo"] = combo_jaccard(fr, combos, k)
        else:
            row["Feyn_snp"] = row["Feyn_combo"] = float("nan")
            print("  Feyn rank not found")
        print(f"  Feyn   SNP={row['Feyn_snp']:.3f}  Combo={row['Feyn_combo']:.3f}")

        # Lasso
        lr1 = lasso_rank(X, y)
        row["Lasso_snp"]   = jaccard(set(lr1[:k]), causative)
        row["Lasso_combo"] = combo_jaccard(lr1, combos, k)
        print(f"  Lasso  SNP={row['Lasso_snp']:.3f}  Combo={row['Lasso_combo']:.3f}")

        # Ridge
        lr2 = ridge_rank(X, y)
        row["Ridge_snp"]   = jaccard(set(lr2[:k]), causative)
        row["Ridge_combo"] = combo_jaccard(lr2, combos, k)
        print(f"  Ridge  SNP={row['Ridge_snp']:.3f}  Combo={row['Ridge_combo']:.3f}")

        # MLP
        mr = mlp_rank(X, y)
        row["MLP_snp"]   = jaccard(set(mr[:k]), causative)
        row["MLP_combo"] = combo_jaccard(mr, combos, k)
        print(f"  MLP    SNP={row['MLP_snp']:.3f}  Combo={row['MLP_combo']:.3f}")

        rows.append(row)

    df = pd.DataFrame(rows)
    out = "jaccard_all_methods.csv"
    df.to_csv(out, index=False)
    print(f"\n\nSaved → {out}")

    print("\n\n" + "="*90)
    print("SNP Jaccard@K")
    print("="*90)
    print(f"{'Dataset':<22} {'Feyn':>8} {'Lasso':>8} {'Ridge':>8} {'MLP':>8}")
    print("-"*60)
    for _, r in df.iterrows():
        print(f"{r['dataset']:<22} {r['Feyn_snp']:>8.3f} {r['Lasso_snp']:>8.3f} "
              f"{r['Ridge_snp']:>8.3f} {r['MLP_snp']:>8.3f}")

    print("\n\n" + "="*90)
    print("Combo Jaccard@K")
    print("="*90)
    print(f"{'Dataset':<22} {'Feyn':>8} {'Lasso':>8} {'Ridge':>8} {'MLP':>8}")
    print("-"*60)
    for _, r in df.iterrows():
        print(f"{r['dataset']:<22} {r['Feyn_combo']:>8.3f} {r['Lasso_combo']:>8.3f} "
              f"{r['Ridge_combo']:>8.3f} {r['MLP_combo']:>8.3f}")


if __name__ == "__main__":
    main()
