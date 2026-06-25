#!/usr/bin/env python3
"""
Full benchmark (Task 1 + Task 2) for any OLIDA disease dataset.

Usage:
  python run_disease_comparison.py alport
  python run_disease_comparison.py sca17
  python run_disease_comparison.py fhl

Task 1: classification AUROC/AUPRC (RF, PLINK2, Logistic L1/L2, MLP, Feyn, Jaccard)
Task 2: Jaccard(top-K features, OLIDA-causative) for RF, PLINK2, Logistic L1
"""

import os, sys, subprocess, tempfile, shutil, warnings, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, precision_recall_curve)
warnings.filterwarnings("ignore")

PLINK2  = "plink2"
TOP_K   = 100
JAC_K   = 10
N_SVD   = 200
K_DETECT = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def load_split(data_dir, name):
    d = np.load(os.path.join(data_dir, f"genotype_{name}.npz"), allow_pickle=True)
    novel_start = int(d["novel_start"]) if "novel_start" in d else None
    n_novel     = int(d["n_novel"])     if "n_novel"     in d else None
    return (d["X"].astype(np.float32), d["y"].astype(int),
            d["variant_ids"], d["sample_ids"], novel_start, n_novel)


def svd_reduce(X_tr, X_va, X_te, n=N_SVD):
    n = min(n, X_tr.shape[1] - 1, X_tr.shape[0] - 1)
    svd = TruncatedSVD(n_components=n, random_state=42)
    sc  = StandardScaler()
    Xr_tr = sc.fit_transform(svd.fit_transform(X_tr))
    Xr_va = sc.transform(svd.transform(X_va))
    Xr_te = sc.transform(svd.transform(X_te))
    return Xr_tr, Xr_va, Xr_te


def jaccard_knn(X_query, X_pos, k=JAC_K):
    Bq = (X_query > 0).astype(np.float32)
    Bp = (X_pos   > 0).astype(np.float32)
    inter = Bq @ Bp.T
    union = Bq.sum(1, keepdims=True) + Bp.sum(1, keepdims=True).T - inter
    jac   = np.where(union > 0, inter / union, 0.0)
    k = min(k, jac.shape[1])
    return np.partition(jac, -k, axis=1)[:, -k:].mean(axis=1)


def jaccard_vs_set(X, ref):
    B     = (X > 0).astype(np.float32)
    inter = B @ ref
    union = B.sum(1) + ref.sum() - inter
    return np.where(union > 0, inter / union, 0.0)


def jaccard(s1, s2):
    s1, s2 = set(s1), set(s2)
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)


# ── PLINK2 BED writer ──────────────────────────────────────────────────────────

def write_bed(X, y, sample_ids, variant_ids, tmpdir, prefix="data"):
    n, p = X.shape
    with open(os.path.join(tmpdir, prefix + ".fam"), "w") as f:
        for i, sid in enumerate(sample_ids):
            f.write(f"FAM{i}\t{sid}\t0\t0\t0\t{2 if y[i]==1 else 1}\n")

    parsed = []
    for j, vid in enumerate(variant_ids):
        parts = str(vid).split(":")
        chrom = parts[0] if parts else "1"
        pos   = int(parts[1]) if len(parts) > 1 else j + 1
        cn    = chrom.replace("chr","").replace("X","23").replace("Y","24")
        try:    cn = int(cn)
        except: cn = 99
        parsed.append((cn, pos, j))

    order = [t[2] for t in sorted(parsed, key=lambda t: (t[0], t[1]))]
    X_sorted = X[:, order].astype(np.int32)
    parsed_s  = [parsed[i] for i in order]

    with open(os.path.join(tmpdir, prefix + ".bim"), "w") as f:
        for j, (cn, pos, _) in enumerate(parsed_s):
            f.write(f"{cn}\tSNP{j}\t0\t{pos}\tA\tT\n")

    X_clipped = np.clip(X_sorted, 0, 2)
    bed_enc   = np.where(X_clipped == 0, 0, np.where(X_clipped == 1, 2, 3))
    pad = (4 - n % 4) % 4
    if pad:
        bed_enc = np.concatenate([bed_enc, np.zeros((pad, p), dtype=np.int32)])
    bed_enc  = bed_enc.reshape((n + pad) // 4, 4, p)
    mult     = np.array([1, 4, 16, 64], dtype=np.int32)
    byte_mat = (bed_enc * mult[:, None]).sum(axis=1).astype(np.uint8)

    with open(os.path.join(tmpdir, prefix + ".bed"), "wb") as f:
        f.write(bytes([0x6c, 0x1b, 0x01]))
        f.write(byte_mat.tobytes(order='F'))

    return order


def run_plink2_classify(X_tv, y_tv, ids_tv, X_te, var_ids):
    tmpdir = tempfile.mkdtemp()
    try:
        order = write_bed(X_tv.astype(np.int8), y_tv, ids_tv, var_ids, tmpdir, "data")
        cmd = [PLINK2, "--bfile", "data",
               "--logistic", "hide-covar", "allow-no-covars",
               "--out", "data_glm", "--maf", "0.0001", "--threads", "4"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir)
        if r.returncode != 0: return None

        glm = (glob.glob(os.path.join(tmpdir, "*.hybrid")) +
               glob.glob(os.path.join(tmpdir, "*.logistic*")))
        if not glm: return None

        df = pd.read_csv(glm[0], sep="\t")
        df.columns = [c.lstrip("#").strip() for c in df.columns]
        if "TEST" in df.columns: df = df[df["TEST"] == "ADD"]
        p_col = next((c for c in df.columns if c.upper() in ("P","P_LOGISTIC","PVAL")), None)
        if p_col is None: return None
        df["P"] = pd.to_numeric(df[p_col], errors="coerce")
        df = df.dropna(subset=["P"]).sort_values("P")

        top_idx = df.head(TOP_K)["ID"].str.replace("SNP","").astype(int).values
        top_idx = top_idx[top_idx < len(order)]
        orig_cols = np.array(order)[top_idx]
        orig_cols = orig_cols[orig_cols < X_tv.shape[1]]

        print(f"  PLINK2: {len(df)} tested, top {len(orig_cols)} selected")
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        clf.fit(X_tv[:, orig_cols], y_tv)
        return clf.predict_proba(X_te[:, orig_cols])[:, 1]
    finally:
        shutil.rmtree(tmpdir)


def run_plink2_rank(X_tv, y_tv, ids_tv, var_ids):
    """Return ranked original indices by p-value for Task 2."""
    tmpdir = tempfile.mkdtemp()
    try:
        order = write_bed(X_tv.astype(np.int8), y_tv, ids_tv, var_ids, tmpdir, "data")
        cmd = [PLINK2, "--bfile", "data",
               "--logistic", "hide-covar", "allow-no-covars",
               "--out", "data_glm", "--maf", "0.0001", "--threads", "4"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir)
        if r.returncode != 0: return None

        glm = (glob.glob(os.path.join(tmpdir, "*.hybrid")) +
               glob.glob(os.path.join(tmpdir, "*.logistic*")))
        if not glm: return None

        df = pd.read_csv(glm[0], sep="\t")
        df.columns = [c.lstrip("#").strip() for c in df.columns]
        if "TEST" in df.columns: df = df[df["TEST"] == "ADD"]
        p_col = next((c for c in df.columns if c.upper() in ("P","P_LOGISTIC","PVAL")), None)
        if p_col is None: return None
        df["P"] = pd.to_numeric(df[p_col], errors="coerce")
        df = df.dropna(subset=["P"]).sort_values("P")

        snp_sorted = df["ID"].str.replace("SNP","").astype(int).values
        snp_orig   = np.array(order)[snp_sorted[snp_sorted < len(order)]]
        tested     = set(snp_orig.tolist())
        remaining  = [i for i in range(X_tv.shape[1]) if i not in tested]
        return np.concatenate([snp_orig, remaining])
    finally:
        shutil.rmtree(tmpdir)


def main():
    if len(sys.argv) < 2:
        print("Usage: run_disease_comparison.py <short_name>")
        sys.exit(1)

    SHORT    = sys.argv[1]
    DATA_DIR = f"dataset/{SHORT}"
    OUT_DIR  = DATA_DIR

    print(f"Loading data from {DATA_DIR}...")
    X_tr, y_tr, var_ids, ids_tr, novel_start, n_novel = load_split(DATA_DIR, "train")
    X_va, y_va, _,       ids_va, _,            _      = load_split(DATA_DIR, "val")
    X_te, y_te, _,       _,      _,            _      = load_split(DATA_DIR, "test")

    CAUSATIVE = set(range(novel_start, novel_start + n_novel))
    print(f"Features: {X_tr.shape[1]} total, {n_novel} OLIDA-causative (idx {novel_start}–{novel_start+n_novel-1})")
    print(f"Samples: train={len(y_tr)}, val={len(y_va)}, test={len(y_te)}")

    X_tv   = np.concatenate([X_tr, X_va])
    y_tv   = np.concatenate([y_tr, y_va])
    ids_tv = np.concatenate([ids_tr, ids_va])

    Xr_tv, _, Xr_te = svd_reduce(X_tv, X_va, X_te)

    X_pos_tv   = X_tv[y_tv == 1]
    olida_union = (X_pos_tv > 0).any(axis=0).astype(np.float32)

    # ── TASK 1: Classification ─────────────────────────────────────────────────
    print("\n=== TASK 1: Classification ===")
    all_scores = {}

    print("Logistic L2...")
    m = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(Xr_tv, y_tv)
    all_scores["Logistic L2"] = (m.predict_proba(Xr_te)[:,1], "#4C72B0", "-")

    print("Logistic L1...")
    m = LogisticRegression(penalty="l1", solver="saga", C=1.0,
                           max_iter=1000, random_state=42).fit(Xr_tv, y_tv)
    all_scores["Logistic L1"] = (m.predict_proba(Xr_te)[:,1], "#6EA6CD", "--")

    print("MLP...")
    m = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300, random_state=42).fit(Xr_tv, y_tv)
    all_scores["MLP"] = (m.predict_proba(Xr_te)[:,1], "#55A868", "-")

    print("RF...")
    m = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                               n_jobs=-1, random_state=42).fit(X_tv, y_tv)
    all_scores["Random Forest"] = (m.predict_proba(X_te)[:,1], "#C44E52", "-")
    rf_importances = m.feature_importances_

    print("Jaccard vs OLIDA...")
    all_scores["Jaccard vs OLIDA"] = (jaccard_vs_set(X_te, olida_union),    "#917BB2", ":")

    print("PLINK2...")
    sc_plink = run_plink2_classify(X_tv, y_tv, ids_tv, X_te, var_ids)
    if sc_plink is not None:
        all_scores["PLINK2 → Logistic"] = (sc_plink, "#DA8BC3", "-")

    # Metrics
    print(f"\n{'Method':<24} {'AUROC':>7}  {'AUPRC':>7}")
    print("=" * 42)
    rows_csv = []
    for name, (sc, col, ls) in sorted(all_scores.items(),
                                       key=lambda x: -roc_auc_score(y_te, x[1][0])):
        auroc = roc_auc_score(y_te, sc)
        auprc = average_precision_score(y_te, sc)
        print(f"  {name:<22} {auroc:>7.4f}  {auprc:>7.4f}")
        rows_csv.append({"method": name, "auroc": auroc, "auprc": auprc})

    pd.DataFrame(rows_csv).to_csv(os.path.join(OUT_DIR, "comparison_results.csv"), index=False)

    # ROC + PR plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for name, (sc, col, ls) in all_scores.items():
        auroc = roc_auc_score(y_te, sc)
        auprc = average_precision_score(y_te, sc)
        fpr, tpr, _ = roc_curve(y_te, sc)
        prec, rec, _ = precision_recall_curve(y_te, sc)
        lw = 2.5 if name in ("Random Forest", "PLINK2 → Logistic") else 1.8
        axes[0].plot(fpr, tpr, color=col, ls=ls, lw=lw, label=f"{name}  (AUC={auroc:.3f})")
        axes[1].plot(rec, prec, color=col, ls=ls, lw=lw, label=f"{name}  (AP={auprc:.3f})")

    axes[0].plot([0,1],[0,1], "k--", lw=1, alpha=0.4)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title(f"ROC — {SHORT.upper()}\n(test set)")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].axhline(y_te.mean(), color="gray", ls="--", lw=1)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall — {SHORT.upper()}\n(test set)")
    axes[1].legend(fontsize=8, loc="upper right")
    for ax in axes: ax.grid(alpha=0.3); ax.set_xlim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "full_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── TASK 2: Causative variant detection ────────────────────────────────────
    print("\n=== TASK 2: Causative variant detection ===")

    def jaccard_curve(ranked_indices, k_vals):
        return [jaccard(set(ranked_indices[:k]), CAUSATIVE) for k in k_vals]

    k_vals_detect = [k for k in K_DETECT if k <= X_tv.shape[1]]

    curves = {}

    rf_rank = np.argsort(rf_importances)[::-1]
    curves["Random Forest"] = jaccard_curve(rf_rank, k_vals_detect)

    print("  Logistic L1 ranking...")
    X_sc = X_tv.astype(np.float32) / 2.0
    m_l1 = LogisticRegression(penalty="l1", solver="saga", C=0.1,
                               max_iter=2000, random_state=42)
    m_l1.fit(X_sc, y_tv)
    l1_rank = np.argsort(np.abs(m_l1.coef_[0]))[::-1]
    curves["Logistic L1"] = jaccard_curve(l1_rank, k_vals_detect)

    print("  PLINK2 ranking...")
    p2_rank = run_plink2_rank(X_tv, y_tv, ids_tv, var_ids)
    if p2_rank is not None:
        curves["PLINK2"] = jaccard_curve(p2_rank, k_vals_detect)

    # Print table
    print(f"\n{'K':>7}  " + "  ".join(f"{n:>14}" for n in curves))
    print("-" * (9 + 16 * len(curves)))
    for i, k in enumerate(k_vals_detect):
        row = f"{k:>7}  " + "  ".join(f"{curves[n][i]:>14.4f}" for n in curves)
        print(row)

    # Detection plot
    colors = {"Random Forest": "#C44E52", "Logistic L1": "#6EA6CD", "PLINK2": "#DA8BC3"}
    styles = {"Random Forest": "-",       "Logistic L1": "--",       "PLINK2": ":"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, vals in curves.items():
        ax.plot(range(len(k_vals_detect)), vals,
                color=colors.get(name, "gray"), ls=styles.get(name, "-"),
                lw=2.2, marker="o", ms=5, label=name)

    k_exact_idx = next((i for i, k in enumerate(k_vals_detect) if k >= n_novel), len(k_vals_detect)-1)
    ax.axvline(k_exact_idx, color="gray", ls=":", lw=1.2, alpha=0.7,
               label=f"K = {n_novel} (# causative)")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)

    ax.set_xticks(range(len(k_vals_detect)))
    ax.set_xticklabels([str(k) for k in k_vals_detect], rotation=45, fontsize=8)
    ax.set_xlabel("Top-K features selected", fontsize=11)
    ax.set_ylabel("Jaccard (selected ∩ causative / selected ∪ causative)", fontsize=10)
    ax.set_title(f"Causative variant detection — {SHORT.upper()}\n"
                 f"Jaccard(top-K features, {n_novel} OLIDA variants)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "jaccard_detection.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nOutputs → {OUT_DIR}/")
    print(f"  full_comparison.png, jaccard_detection.png, comparison_results.csv")


if __name__ == "__main__":
    main()
