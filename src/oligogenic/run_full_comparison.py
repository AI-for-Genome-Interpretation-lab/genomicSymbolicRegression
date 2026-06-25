#!/usr/bin/env python3
"""
Full comparison: tutti i metodi sullo stesso grafico ROC/PR.
Logistic L1/L2, MLP, RF, Feyn, PLINK2→Logistic, Jaccard kNN, Jaccard vs OLIDA.
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

DATA_DIR  = "dataset/kallmann"
OUT_DIR   = DATA_DIR
PLINK2    = "plink2"
TOP_K     = 100
JAC_K     = 10
N_SVD     = 200


def load_split(name):
    d = np.load(os.path.join(DATA_DIR, f"genotype_{name}.npz"), allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(int), d["variant_ids"], d["sample_ids"]


# ── SVD reduction ─────────────────────────────────────────────────────────────
def svd_reduce(X_tr, X_va, X_te, n=N_SVD):
    n = min(n, X_tr.shape[1] - 1, X_tr.shape[0] - 1)
    svd = TruncatedSVD(n_components=n, random_state=42)
    sc  = StandardScaler()
    Xr_tr = sc.fit_transform(svd.fit_transform(X_tr))
    Xr_va = sc.transform(svd.transform(X_va))
    Xr_te = sc.transform(svd.transform(X_te))
    return Xr_tr, Xr_va, Xr_te


# ── Jaccard ───────────────────────────────────────────────────────────────────
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


# ── PLINK2 ────────────────────────────────────────────────────────────────────
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
        parsed.append((cn, pos, j, chrom))

    order = [t[2] for t in sorted(parsed, key=lambda t: (t[0], t[1]))]
    X = X[:, order]
    parsed_sorted = [parsed[i] for i in order]

    with open(os.path.join(tmpdir, prefix + ".bim"), "w") as f:
        for j, (cn, pos, _, _chrom) in enumerate(parsed_sorted):
            f.write(f"{cn}\tSNP{j}\t0\t{pos}\tA\tT\n")

    X_clipped = np.clip(X.astype(np.int32), 0, 2)
    bed_enc   = np.where(X_clipped == 0, 0, np.where(X_clipped == 1, 2, 3))
    pad = (4 - n % 4) % 4
    if pad:
        bed_enc = np.concatenate([bed_enc, np.zeros((pad, p), dtype=np.int32)])
    bed_enc   = bed_enc.reshape((n + pad) // 4, 4, p)
    mult      = np.array([1, 4, 16, 64], dtype=np.int32)
    byte_mat  = (bed_enc * mult[:, None]).sum(axis=1).astype(np.uint8)

    with open(os.path.join(tmpdir, prefix + ".bed"), "wb") as f:
        f.write(bytes([0x6c, 0x1b, 0x01]))
        f.write(byte_mat.tobytes(order='F'))

    return order   # variant reordering


def run_plink2(X_tv, y_tv, ids_tv, X_te, var_ids):
    tmpdir = tempfile.mkdtemp()
    try:
        order = write_bed(X_tv.astype(np.int8), y_tv, ids_tv, var_ids, tmpdir, "data")
        cmd = [PLINK2, "--bfile", "data",
               "--logistic", "hide-covar", "allow-no-covars",
               "--out", "data_glm", "--maf", "0.0001", "--threads", "4"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir)
        if r.returncode != 0:
            log = open(os.path.join(tmpdir,"data_glm.log")).read() if \
                  os.path.exists(os.path.join(tmpdir,"data_glm.log")) else r.stdout+r.stderr
            print(f"  PLINK2 failed:\n{log}")
            return None

        glm = (glob.glob(os.path.join(tmpdir,"*.hybrid")) +
               glob.glob(os.path.join(tmpdir,"*.logistic*")))
        if not glm:
            print("  PLINK2: no output file"); return None

        df = pd.read_csv(glm[0], sep="\t")
        df.columns = [c.lstrip("#").strip() for c in df.columns]
        if "TEST" in df.columns:
            df = df[df["TEST"] == "ADD"]
        p_col = next((c for c in df.columns if c.upper() in ("P","P_LOGISTIC","PVAL")), None)
        if p_col is None:
            print(f"  PLINK2: no p-value col in {list(df.columns)}"); return None
        df["P"] = pd.to_numeric(df[p_col], errors="coerce")
        df = df.dropna(subset=["P"]).sort_values("P")

        top_idx = df.head(TOP_K)["ID"].str.replace("SNP","").astype(int).values
        top_idx = top_idx[top_idx < len(order)]

        # map back to original column space via order
        orig_cols = np.array(order)[top_idx]
        orig_cols = orig_cols[orig_cols < X_tv.shape[1]]

        print(f"  PLINK2: {len(df)} variants tested, top {len(orig_cols)} selected")
        print(f"  Smallest p={df['P'].iloc[0]:.2e}")

        clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        clf.fit(X_tv[:, orig_cols], y_tv)
        return clf.predict_proba(X_te[:, orig_cols])[:, 1]
    finally:
        shutil.rmtree(tmpdir)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    X_tr, y_tr, var_ids, ids_tr = load_split("train")
    X_va, y_va, _,       ids_va = load_split("val")
    X_te, y_te, _,       _      = load_split("test")

    X_tv  = np.concatenate([X_tr, X_va])
    y_tv  = np.concatenate([y_tr, y_va])
    ids_tv = np.concatenate([ids_tr, ids_va])

    Xr_tr, Xr_va, Xr_te = svd_reduce(X_tr, X_va, X_te)
    Xr_tv, _, Xr_te2     = svd_reduce(X_tv, X_va, X_te)

    X_pos_tv = X_tv[y_tv == 1]
    olida_union = (X_pos_tv > 0).any(axis=0).astype(np.float32)

    all_scores = {}   # name → (score_te, color, ls)

    # ── Linear models (SVD features) ──────────────────────────────────────────
    print("\nLogistic L2...")
    m = LogisticRegression(C=1.0, max_iter=500, random_state=42).fit(Xr_tv, y_tv)
    all_scores["Logistic L2"]  = (m.predict_proba(Xr_te2)[:,1], "#4C72B0", "-")

    print("Logistic L1...")
    m = LogisticRegression(penalty="l1", solver="saga", C=1.0,
                           max_iter=1000, random_state=42).fit(Xr_tv, y_tv)
    all_scores["Logistic L1"]  = (m.predict_proba(Xr_te2)[:,1], "#6EA6CD", "--")

    # ── MLP ───────────────────────────────────────────────────────────────────
    print("MLP...")
    m = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300, random_state=42).fit(Xr_tv, y_tv)
    all_scores["MLP"]          = (m.predict_proba(Xr_te2)[:,1], "#55A868", "-")

    # ── RF (raw dosage) ───────────────────────────────────────────────────────
    print("RF...")
    m = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                               n_jobs=-1, random_state=42).fit(X_tv, y_tv)
    all_scores["Random Forest"] = (m.predict_proba(X_te)[:,1], "#C44E52", "-")

    # ── Feyn ──────────────────────────────────────────────────────────────────
    try:
        import feyn
        print("Feyn...")
        ql = feyn.QLattice(random_seed=42)
        cols = [f"pc{i}" for i in range(Xr_tv.shape[1])]
        df_tv = pd.DataFrame(Xr_tv, columns=cols); df_tv["y"] = y_tv
        df_te = pd.DataFrame(Xr_te2, columns=cols)
        models = ql.auto_run(df_tv, output_name="y", kind="classification",
                             n_epochs=15, criterion="bic")
        sc = models[0].predict(df_te)
        all_scores["Feyn"] = (sc, "#DD8452", "-")
    except Exception as e:
        print(f"  Feyn skipped: {e}")

    # ── Jaccard ───────────────────────────────────────────────────────────────
    print("Jaccard kNN...")
    all_scores["Jaccard kNN"]      = (jaccard_knn(X_te, X_pos_tv, JAC_K),  "#8172B2", "--")
    print("Jaccard vs OLIDA union...")
    all_scores["Jaccard vs OLIDA"] = (jaccard_vs_set(X_te, olida_union),    "#917BB2", ":")

    # ── PLINK2 ────────────────────────────────────────────────────────────────
    print("PLINK2...")
    sc_plink = run_plink2(X_tv, y_tv, ids_tv, X_te, var_ids)
    if sc_plink is not None:
        all_scores["PLINK2 → Logistic"] = (sc_plink, "#DA8BC3", "-")

    # ── Metrics table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print(f"{'Method':<22} {'AUROC':>7}  {'AUPRC':>7}")
    print("=" * 58)
    rows = []
    for name, (sc, col, ls) in sorted(all_scores.items(),
                                       key=lambda x: -roc_auc_score(y_te, x[1][0])):
        auroc = roc_auc_score(y_te, sc)
        auprc = average_precision_score(y_te, sc)
        print(f"  {name:<20} {auroc:>7.4f}  {auprc:>7.4f}")
        rows.append({"method": name, "auroc": auroc, "auprc": auprc})
    print("=" * 58)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "comparison_results.csv"), index=False)

    # ── Combined ROC + PR plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for name, (sc, col, ls) in all_scores.items():
        auroc = roc_auc_score(y_te, sc)
        auprc = average_precision_score(y_te, sc)
        fpr, tpr, _ = roc_curve(y_te, sc)
        prec, rec, _ = precision_recall_curve(y_te, sc)
        lw = 2.5 if name in ("Random Forest", "Feyn") else 1.8
        axes[0].plot(fpr, tpr, color=col, ls=ls, lw=lw,
                     label=f"{name}  (AUC={auroc:.3f})")
        axes[1].plot(rec, prec, color=col, ls=ls, lw=lw,
                     label=f"{name}  (AP={auprc:.3f})")

    axes[0].plot([0,1],[0,1], "k--", lw=1, alpha=0.4)
    axes[0].set_xlabel("False Positive Rate", fontsize=12)
    axes[0].set_ylabel("True Positive Rate", fontsize=12)
    axes[0].set_title("ROC — Kallmann syndrome\ngenotype dosage matrix (test set)", fontsize=12)
    axes[0].legend(fontsize=9, loc="lower right")

    axes[1].axhline(y_te.mean(), color="gray", ls="--", lw=1, label="Baseline")
    axes[1].set_xlabel("Recall", fontsize=12)
    axes[1].set_ylabel("Precision", fontsize=12)
    axes[1].set_title("Precision-Recall — Kallmann syndrome\ngenotype dosage matrix (test set)", fontsize=12)
    axes[1].legend(fontsize=9, loc="upper right")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.02, 1.02)

    plt.tight_layout()
    out_fig = os.path.join(OUT_DIR, "full_comparison.png")
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot → {out_fig}")
    print(f"CSV  → {OUT_DIR}/comparison_results.csv")


if __name__ == "__main__":
    main()
