#!/usr/bin/env python3
"""
Fig 3 style: Jaccard(selected_features, causative_variants) per Kallmann.

Causative = le 72 colonne OLIDA-novel (indici 17480-17551).
Ogni metodo seleziona le sue top-K feature; misuriamo Jaccard vs causative
al variare di K (quante feature tieni).

Metodi:
  - RF: top-K per feature importance
  - PLINK2: top-K per p-value (--glm logistic)
  - Logistic L1 (raw): coefficienti non-zero sulla matrice dosage senza SVD

Output: jaccard_detection.png
"""

import os, sys, subprocess, tempfile, shutil, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

DATA_DIR   = "dataset/kallmann"
OUT_DIR    = DATA_DIR
PLINK2_BIN = "plink2"
OLIDA_START = 17480   # le 72 colonne OLIDA-novel iniziano qui
N_OLIDA     = 72
CAUSATIVE   = set(range(OLIDA_START, OLIDA_START + N_OLIDA))

K_VALUES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 17552]


def load_split(name):
    d = np.load(os.path.join(DATA_DIR, f"genotype_{name}.npz"), allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(int), d["variant_ids"], d["sample_ids"]


def jaccard(set1, set2):
    s1, s2 = set(set1), set(set2)
    if not s1 and not s2: return 1.0
    if not s1 or  not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)


def jaccard_curve(ranked_indices):
    """Jaccard vs CAUSATIVE as we include top-1, top-2, ... top-K features."""
    vals = []
    for k in K_VALUES:
        top = set(ranked_indices[:k])
        vals.append(jaccard(top, CAUSATIVE))
    return vals


# ── RF ────────────────────────────────────────────────────────────────────────
def rf_ranking(X_tv, y_tv):
    print("  RF fitting...")
    rf = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                n_jobs=-1, random_state=42)
    rf.fit(X_tv, y_tv)
    return np.argsort(rf.feature_importances_)[::-1]


# ── Logistic L1 (no SVD — direct on dosage) ───────────────────────────────────
def lasso_ranking(X_tv, y_tv):
    print("  Logistic L1 fitting (raw dosage)...")
    # Scale to [0,1] per variant (max=2), then L1
    X_sc = X_tv / 2.0
    m = LogisticRegression(penalty="l1", solver="saga", C=0.1,
                           max_iter=2000, random_state=42)
    m.fit(X_sc, y_tv)
    coef = np.abs(m.coef_[0])
    return np.argsort(coef)[::-1]


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
        parsed.append((cn, pos, j))

    order = [t[2] for t in sorted(parsed, key=lambda t: (t[0], t[1]))]
    X_sorted = X[:, order]
    parsed_s  = [parsed[i] for i in order]

    with open(os.path.join(tmpdir, prefix + ".bim"), "w") as f:
        for j, (cn, pos, _) in enumerate(parsed_s):
            f.write(f"{cn}\tSNP{j}\t0\t{pos}\tA\tT\n")

    X_clipped = np.clip(X_sorted.astype(np.int32), 0, 2)
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

    return order  # original column indices in sorted order


def plink2_ranking(X_tv, y_tv, ids_tv, var_ids):
    print("  PLINK2 running...")
    tmpdir = tempfile.mkdtemp()
    try:
        order = write_bed(X_tv.astype(np.int8), y_tv, ids_tv, var_ids, tmpdir, "data")
        cmd = [PLINK2_BIN, "--bfile", "data",
               "--logistic", "hide-covar", "allow-no-covars",
               "--out", "data_glm", "--maf", "0.0001", "--threads", "4"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir)
        if r.returncode != 0:
            print("  PLINK2 failed"); return None

        glm = (glob.glob(os.path.join(tmpdir, "*.hybrid")) +
               glob.glob(os.path.join(tmpdir, "*.logistic*")))
        if not glm: return None

        df = pd.read_csv(glm[0], sep="\t")
        df.columns = [c.lstrip("#").strip() for c in df.columns]
        if "TEST" in df.columns:
            df = df[df["TEST"] == "ADD"]
        p_col = next((c for c in df.columns if c.upper() in ("P","P_LOGISTIC","PVAL")), None)
        if p_col is None: return None

        df["P"] = pd.to_numeric(df[p_col], errors="coerce")
        df = df.dropna(subset=["P"]).sort_values("P")

        # SNP index in sorted space → back to original column index
        snp_sorted_idx = df["ID"].str.replace("SNP", "").astype(int).values
        snp_orig_idx   = np.array(order)[snp_sorted_idx[snp_sorted_idx < len(order)]]

        # variants filtered by MAF are missing — append the rest at the end
        tested = set(snp_orig_idx.tolist())
        remaining = [i for i in range(X_tv.shape[1]) if i not in tested]
        return np.concatenate([snp_orig_idx, remaining])
    finally:
        shutil.rmtree(tmpdir)


# ── Feyn ──────────────────────────────────────────────────────────────────────
def feyn_ranking(X_tv, y_tv, n_svd=200):
    """
    1. SVD riduce X_tv → PC space
    2. Feyn trova il modello simbolico (usa alcuni PC)
    3. Per ogni PC usato da Feyn, prendiamo il loading vector (riga di svd.components_)
    4. Ranchiamo le varianti originali per contributo totale ai PC usati
    """
    import feyn

    n_svd = min(n_svd, X_tv.shape[1] - 1, X_tv.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    sc  = StandardScaler()
    Xr  = sc.fit_transform(svd.fit_transform(X_tv))  # (n, n_svd)

    cols = [f"pc{i}" for i in range(n_svd)]
    df_tv = pd.DataFrame(Xr, columns=cols)
    df_tv["y"] = y_tv

    print("  Feyn fitting...")
    ql = feyn.QLattice(random_seed=42)
    models = ql.auto_run(df_tv, output_name="y", kind="classification",
                         n_epochs=15, criterion="bic")
    best = models[0]

    # Quali PC usa il modello Feyn?
    # feyn model ha .graph con i nodi; i nodi input hanno .name = "pc{i}"
    used_pcs = []
    try:
        for node in best:
            name = getattr(node, "name", "")
            if name.startswith("pc"):
                idx = int(name[2:])
                if idx not in used_pcs:
                    used_pcs.append(idx)
    except Exception:
        pass

    print(f"  Feyn used PCs: {used_pcs}")

    if not used_pcs:
        # fallback: rank by abs weight in logistic on Xr
        return np.arange(X_tv.shape[1])

    # Loading matrix: svd.components_ shape (n_svd, n_features)
    # Per ogni PC usato, prendi il loading vector (importanza per variante originale)
    loadings = np.abs(svd.components_[used_pcs, :])  # (n_used, n_features)
    # Score per variante = somma dei loading assoluti sui PC usati
    scores = loadings.sum(axis=0)  # (n_features,)
    return np.argsort(scores)[::-1]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    X_tr, y_tr, var_ids, ids_tr = load_split("train")
    X_va, y_va, _,       ids_va = load_split("val")
    X_tv  = np.concatenate([X_tr, X_va]).astype(np.int8)
    y_tv  = np.concatenate([y_tr, y_va])
    ids_tv = np.concatenate([ids_tr, ids_va])

    print(f"Causative set: {len(CAUSATIVE)} OLIDA-novel columns ({OLIDA_START}-{OLIDA_START+N_OLIDA-1})")

    curves = {}

    rf_rank   = rf_ranking(X_tv.astype(np.float32), y_tv)
    curves["Random Forest"] = jaccard_curve(rf_rank)

    l1_rank   = lasso_ranking(X_tv.astype(np.float32), y_tv)
    curves["Logistic L1"] = jaccard_curve(l1_rank)

    p2_rank = plink2_ranking(X_tv, y_tv, ids_tv, var_ids)
    if p2_rank is not None:
        curves["PLINK2"] = jaccard_curve(p2_rank)

    try:
        feyn_rank = feyn_ranking(X_tv.astype(np.float32), y_tv)
        curves["Feyn"] = jaccard_curve(feyn_rank)
    except Exception as e:
        print(f"  Feyn skipped: {e}")

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'K':>7}  " + "  ".join(f"{n:>14}" for n in curves))
    print("-" * (9 + 16 * len(curves)))
    for i, k in enumerate(K_VALUES):
        row = f"{k:>7}  " + "  ".join(f"{curves[n][i]:>14.4f}" for n in curves)
        print(row)

    # ── Plot ──────────────────────────────────────────────────────────────────
    colors = {"Random Forest": "#C44E52", "Logistic L1": "#6EA6CD",
              "PLINK2": "#DA8BC3",       "Feyn": "#DD8452"}
    styles = {"Random Forest": "-",       "Logistic L1": "--",
              "PLINK2": ":",             "Feyn": "-"}

    fig, ax = plt.subplots(figsize=(9, 5))

    for name, vals in curves.items():
        ax.plot(range(len(K_VALUES)), vals, color=colors.get(name, "gray"),
                ls=styles.get(name, "-"), lw=2.2, marker="o", ms=5, label=name)

    # mark where K = 72 (the exact number of causative variants)
    k72_idx = next(i for i, k in enumerate(K_VALUES) if k >= N_OLIDA)
    ax.axvline(k72_idx, color="gray", ls=":", lw=1.2, alpha=0.7,
               label=f"K = {N_OLIDA} (# causative)")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)

    ax.set_xticks(range(len(K_VALUES)))
    ax.set_xticklabels([str(k) for k in K_VALUES], rotation=45, fontsize=8)
    ax.set_xlabel("Top-K features selected", fontsize=11)
    ax.set_ylabel("Jaccard (selected ∩ causative / selected ∪ causative)", fontsize=10)
    ax.set_title("Causative variant detection — Kallmann syndrome\n"
                 "Jaccard(top-K features, 72 OLIDA disease variants)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "jaccard_detection.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot → {out}")


if __name__ == "__main__":
    main()
