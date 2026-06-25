#!/usr/bin/env python3
"""
PLINK2 association test + Jaccard similarity classifier on the Kallmann genotype matrix.

PLINK2: exports the dosage matrix to BED/BIM/FAM, runs --glm logistic,
        uses top-k associated variants as Ridge logistic features.

Jaccard: for each sample, score = mean Jaccard similarity to training positives
         Jaccard(i, j) = |vars(i) ∩ vars(j)| / |vars(i) ∪ vars(j)|
         where vars(x) = set of variant indices with dosage > 0.
"""

import os, sys, subprocess, tempfile, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

DATA_DIR = "dataset/kallmann"
OUT_DIR  = DATA_DIR

PLINK2_BIN = "plink2"
TOP_K_VARIANTS = 100   # top associated variants to keep for Ridge
JACCARD_K      = 10    # kNN: mean Jaccard to top-K training positives


# ── Load data ─────────────────────────────────────────────────────────────────

def load_split(name):
    d = np.load(os.path.join(DATA_DIR, f"genotype_{name}.npz"), allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(int), d["variant_ids"], d["sample_ids"]


# ── Jaccard ───────────────────────────────────────────────────────────────────

def jaccard_score_vs_positives(X_query, X_pos_train, k=JACCARD_K):
    """
    For each query sample, compute Jaccard similarity to each training positive,
    return mean of top-k values as the score.
    Vectorized: O(n_query × n_pos) matrix ops instead of triple loop.
    """
    Bq = (X_query > 0).astype(np.float32)       # (n_query, n_var)
    Bp = (X_pos_train > 0).astype(np.float32)   # (n_pos, n_var)

    # |intersection| via dot product (binary)
    inter = Bq @ Bp.T                            # (n_query, n_pos)

    # |A| and |B| row sums
    sum_q = Bq.sum(axis=1, keepdims=True)        # (n_query, 1)
    sum_p = Bp.sum(axis=1, keepdims=True).T      # (1, n_pos)

    union = sum_q + sum_p - inter                # (n_query, n_pos)
    jac = np.where(union > 0, inter / union, 0.0)  # (n_query, n_pos)

    # mean of top-k per query
    if k >= jac.shape[1]:
        return jac.mean(axis=1)
    # partial sort for top-k
    scores = np.partition(jac, -k, axis=1)[:, -k:].mean(axis=1)
    return scores


# ── PLINK2 ────────────────────────────────────────────────────────────────────

def write_bed(X, y, sample_ids, variant_ids, tmpdir, prefix="data"):
    """Write minimal BED/BIM/FAM for PLINK2. Vectorized BED encoding."""
    n, p = X.shape

    # FAM
    with open(os.path.join(tmpdir, prefix + ".fam"), "w") as f:
        for i, sid in enumerate(sample_ids):
            pheno = 2 if y[i] == 1 else 1
            f.write(f"FAM{i}\t{sid}\t0\t0\t0\t{pheno}\n")

    # Parse and sort variants by chrom, pos
    parsed = []
    for j, vid in enumerate(variant_ids):
        parts = str(vid).split(":")
        chrom = parts[0] if len(parts) > 0 else "1"
        pos   = int(parts[1]) if len(parts) > 1 else j + 1
        chrom_n = chrom.replace("chr", "").replace("X", "23").replace("Y", "24")
        try: chrom_n = int(chrom_n)
        except: chrom_n = 99
        parsed.append((chrom_n, pos, j, chrom))

    sort_order = [t[2] for t in sorted(parsed, key=lambda t: (t[0], t[1]))]
    X = X[:, sort_order]
    variant_ids = [variant_ids[i] for i in sort_order]
    parsed = [parsed[i] for i in sort_order]

    # BIM
    with open(os.path.join(tmpdir, prefix + ".bim"), "w") as f:
        for j, (chrom_n, pos, orig_j, chrom) in enumerate(parsed):
            f.write(f"{chrom_n}\tSNP{j}\t0\t{pos}\tA\tT\n")

    # BED — vectorized: encode all samples for each variant at once
    # PLINK BED bit pairs per sample: 00=hom_ref, 10=het, 11=hom_alt, 01=missing
    # dosage 0→0b00=0, 1→0b10=2, 2→0b11=3
    X_clipped = np.clip(X.astype(np.int32), 0, 2)
    bed_enc = np.where(X_clipped == 0, 0, np.where(X_clipped == 1, 2, 3))  # (n, p)

    # Pad n to multiple of 4
    pad = (4 - n % 4) % 4
    if pad:
        bed_enc = np.concatenate([bed_enc, np.zeros((pad, p), dtype=np.int32)], axis=0)
    n_padded = n + pad

    # Pack 4 samples per byte: byte = g0*(1<<0) | g1*(1<<2) | g2*(1<<4) | g3*(1<<6)
    bed_enc = bed_enc.reshape(n_padded // 4, 4, p)  # (n_bytes, 4, p)
    multipliers = np.array([1, 4, 16, 64], dtype=np.int32)  # 1<<[0,2,4,6]
    byte_matrix = (bed_enc * multipliers[:, None]).sum(axis=1).astype(np.uint8)  # (n_bytes, p)

    with open(os.path.join(tmpdir, prefix + ".bed"), "wb") as f:
        f.write(bytes([0x6c, 0x1b, 0x01]))
        f.write(byte_matrix.tobytes(order='F'))  # column-major = variant-major

    return prefix


def run_plink2_glm(tmpdir, prefix, n_samples):
    """Run PLINK2 --glm logistic on the BED files."""
    cmd = [
        PLINK2_BIN,
        "--bfile", prefix,
        "--logistic", "hide-covar", "allow-no-covars",
        "--out", prefix + "_glm",
        "--maf", "0.0001",
        "--threads", "4",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir)
    return r.returncode, r.stdout + r.stderr


def parse_plink2_glm(tmpdir, prefix):
    """Parse PLINK2 .PHENO1.glm.logistic output. Returns DataFrame sorted by p-value."""
    # PLINK2 names the output file based on the phenotype column name
    import glob
    glm_files = glob.glob(os.path.join(tmpdir, f"{prefix}_glm*.hybrid")) + \
                glob.glob(os.path.join(tmpdir, f"{prefix}_glm*.logistic*"))
    if not glm_files:
        return None
    df = pd.read_csv(glm_files[0], sep="\t")
    df.columns = [c.lstrip("#").strip() for c in df.columns]
    print(f"  PLINK2 GLM columns: {list(df.columns)}")
    # keep only ADD test (marginal effect)
    if "TEST" in df.columns:
        df = df[df["TEST"] == "ADD"]
    # p-value column name varies between plink2 versions
    p_col = next((c for c in df.columns if c.upper() in ("P", "P_LOGISTIC", "PVAL")), None)
    if p_col is None:
        print(f"  No p-value column found in: {list(df.columns)}")
        return None
    df = df.rename(columns={p_col: "P"})
    df["P"] = pd.to_numeric(df["P"], errors="coerce")
    df = df.dropna(subset=["P"])
    df = df.sort_values("P")
    return df


def plink2_predict(X_tr, y_tr, X_va, y_va, X_te, y_te, variant_ids, sample_ids_tr,
                   sample_ids_va, sample_ids_te):
    """Full PLINK2 pipeline: write BED, run --glm, top-k → Logistic."""
    # combine train+val for fitting, test for eval
    X_tv = np.concatenate([X_tr, X_va])
    y_tv = np.concatenate([y_tr, y_va])
    ids_tv = np.concatenate([sample_ids_tr, sample_ids_va])

    tmpdir = tempfile.mkdtemp()
    try:
        write_bed(X_tv, y_tv, ids_tv, variant_ids, tmpdir, "data")
        rc, log = run_plink2_glm(tmpdir, "data", len(X_tv))
        if rc != 0:
            # also try to read the .log file
            log_file = os.path.join(tmpdir, "data_glm.log")
            if os.path.exists(log_file):
                with open(log_file) as lf:
                    log = lf.read()
            print(f"  PLINK2 failed (rc={rc}):\n{log}")
            return None, None

        glm_df = parse_plink2_glm(tmpdir, "data")
        if glm_df is None or len(glm_df) == 0:
            print("  PLINK2: no GLM results parsed")
            return None, None

        # top-k SNP indices
        top_snps = glm_df.head(TOP_K_VARIANTS)["ID"].str.replace("SNP", "").astype(int).values
        top_snps = top_snps[top_snps < X_tr.shape[1]]

        print(f"  Top {len(top_snps)} variants selected by PLINK2 p-value")
        print(f"  Smallest p: {glm_df['P'].iloc[0]:.2e}, median p: {glm_df['P'].median():.2e}")

        X_tv_sub = X_tv[:, top_snps]
        X_te_sub = X_te[:, top_snps]

        clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        clf.fit(X_tv_sub, y_tv)
        sc_te = clf.predict_proba(X_te_sub)[:, 1]
        return sc_te, glm_df

    finally:
        shutil.rmtree(tmpdir)


# ── Main ──────────────────────────────────────────────────────────────────────

def metrics(y_true, y_score, name):
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    print(f"  {name:35s}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
    return auroc, auprc


def main():
    print("Loading genotype matrices...")
    X_tr, y_tr, var_ids, ids_tr = load_split("train")
    X_va, y_va, _,       ids_va = load_split("val")
    X_te, y_te, _,       ids_te = load_split("test")

    results = {}

    # ── Jaccard ───────────────────────────────────────────────────────────────
    # Two variants:
    # 1. kNN: similarity to each training positive individually (mean top-k)
    # 2. vs OLIDA set: Jaccard between sample's variant set and UNION of all training positive variants
    print(f"\n=== Jaccard (kNN k={JACCARD_K} + vs OLIDA union) ===")
    X_tv = np.concatenate([X_tr, X_va])
    y_tv = np.concatenate([y_tr, y_va])
    X_pos_tr = X_tr[y_tr == 1]
    X_pos_tv = X_tv[y_tv == 1]

    # OLIDA union: set of variant positions present in ANY training positive
    olida_union_tr = (X_pos_tr > 0).any(axis=0).astype(np.float32)  # (n_var,)
    olida_union_tv = (X_pos_tv > 0).any(axis=0).astype(np.float32)

    def jaccard_vs_set(X, ref_set):
        """Jaccard between each sample's variant set and a reference set."""
        B = (X > 0).astype(np.float32)
        inter = B @ ref_set                          # (n_samples,)
        sum_b = B.sum(axis=1)                        # (n_samples,)
        sum_r = ref_set.sum()
        union = sum_b + sum_r - inter
        return np.where(union > 0, inter / union, 0.0)

    # kNN val/test
    sc_va_knn = jaccard_score_vs_positives(X_va, X_pos_tr, k=JACCARD_K)
    results["Jaccard_kNN_val"] = metrics(y_va, sc_va_knn, f"Jaccard kNN val")
    sc_te_jac_knn = jaccard_score_vs_positives(X_te, X_pos_tv, k=JACCARD_K)
    results["Jaccard_kNN_test"] = metrics(y_te, sc_te_jac_knn, f"Jaccard kNN test")

    # vs OLIDA union val/test
    sc_va_set = jaccard_vs_set(X_va, olida_union_tr)
    results["Jaccard_set_val"] = metrics(y_va, sc_va_set, f"Jaccard vs OLIDA union val")
    sc_te_jac = jaccard_vs_set(X_te, olida_union_tv)
    results["Jaccard_set_test"] = metrics(y_te, sc_te_jac, f"Jaccard vs OLIDA union test")

    # ── PLINK2 ────────────────────────────────────────────────────────────────
    print(f"\n=== PLINK2 --glm logistic → top-{TOP_K_VARIANTS} → Logistic ===")
    sc_te_plink, glm_df = plink2_predict(
        X_tr.astype(np.int8), y_tr,
        X_va.astype(np.int8), y_va,
        X_te.astype(np.int8), y_te,
        var_ids, ids_tr, ids_va, ids_te
    )
    if sc_te_plink is not None:
        results["PLINK2_test"] = metrics(y_te, sc_te_plink, "PLINK2 → Logistic (test)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    model_scores = [("Jaccard kNN", sc_te_jac_knn, "mediumpurple"),
                    ("Jaccard vs OLIDA", sc_te_jac, "purple")]
    if sc_te_plink is not None:
        model_scores.append(("PLINK2 → Logistic", sc_te_plink, "darkorange"))

    # Load previous results for comparison
    prev_scores = {}
    try:
        prev = np.load(os.path.join(DATA_DIR, "genotype_train.npz"), allow_pickle=True)
        # just load RF scores from the previous benchmark if saved
    except: pass

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, sc, col in model_scores:
        fpr, tpr, _ = roc_curve(y_te, sc)
        axes[0].plot(fpr, tpr, color=col, lw=2,
                     label=f"{name} AUC={roc_auc_score(y_te, sc):.3f}")
        prec, rec, _ = precision_recall_curve(y_te, sc)
        axes[1].plot(rec, prec, color=col, lw=2,
                     label=f"{name} AP={average_precision_score(y_te, sc):.3f}")

    for ax, title, xlabel, ylabel in [
        (axes[0], "ROC", "FPR", "TPR"),
        (axes[1], "Precision-Recall", "Recall", "Precision"),
    ]:
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — Kallmann (test)")
        ax.legend(fontsize=9)
    axes[0].plot([0,1],[0,1],'k--', lw=1)
    axes[1].axhline(y_te.mean(), color='gray', ls='--', lw=1, label="Baseline")

    plt.tight_layout()
    out_fig = os.path.join(OUT_DIR, "plink2_jaccard_roc_pr.png")
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"\nPlot saved → {out_fig}")

    # ── PLINK2 top hits ───────────────────────────────────────────────────────
    if glm_df is not None:
        print("\nTop 10 variants by PLINK2 p-value:")
        print(glm_df[["ID", "A1", "OBS_CT", "OR", "P"]].head(10).to_string(index=False))

        # map SNP index back to variant id
        top10_idx = glm_df.head(10)["ID"].str.replace("SNP","").astype(int).values
        print("\nVariant IDs:")
        for i in top10_idx:
            if i < len(var_ids):
                print(f"  SNP{i}: {var_ids[i]}")


if __name__ == "__main__":
    main()
