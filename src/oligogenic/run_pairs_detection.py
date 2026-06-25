#!/usr/bin/env python3
"""
Task 2 — Pair-level detection.

For each OLIDA combination (which involves 2+ variants), a combination is
"detected" at rank K if ALL its variant columns appear in the top-K features.

Metric: Jaccard(detected_combos, causative_combos)
  = |detected| / |causative|   (since detected ⊆ causative, this equals recall)

Methods ranked: RF, PLINK2, Logistic L1, Feyn (raw, if results saved).

Usage:
  python run_pairs_detection.py sca17
  python run_pairs_detection.py alport
  python run_pairs_detection.py fhl
"""

import os, sys, subprocess, tempfile, shutil, warnings, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

POSITIVES_TSV = "dataset/processed/olida_positives.tsv"
PLINK2        = "plink2"
K_VALUES      = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]


def load_split(data_dir, name):
    d = np.load(os.path.join(data_dir, f"genotype_{name}.npz"), allow_pickle=True)
    return (d["X"].astype(np.float32), d["y"].astype(int),
            d["variant_ids"], d["sample_ids"],
            int(d["novel_start"]), int(d["n_novel"]))


def build_causative_pairs(disease_name, variant_ids_array):
    """
    For each OLIDA combination of the disease, collect the frozenset of
    column indices corresponding to its variants.
    Returns list of (combo_id, frozenset_of_col_indices).
    """
    pos_all = pd.read_csv(POSITIVES_TSV, sep="\t")
    disease_df = pos_all[pos_all["disease_name"] == disease_name].copy()

    # Build lookup: "chrom:pos:ref:alt" → column index
    var_lookup = {str(vid): i for i, vid in enumerate(variant_ids_array)}

    combos = []
    for _, row in disease_df.iterrows():
        indices = set()
        for i in range(1, 5):
            p = f"var{i}_"
            chrom = str(row.get(p+"chrom", "N.A."))
            pos   = str(row.get(p+"pos_hg19", "N.A."))
            ref   = str(row.get(p+"ref", "N.A."))
            alt   = str(row.get(p+"alt", "N.A."))
            if chrom in ("N.A.", "nan", "") or pos in ("N.A.", "nan", ""):
                continue
            key  = f"{chrom}:{pos}:{ref}:{alt}"
            pos38 = str(row.get(p+"pos_hg38", "N.A."))
            key38 = f"{chrom}:{pos38}:{ref}:{alt}"
            for k in [key, key38]:
                if k in var_lookup:
                    indices.add(var_lookup[k])
                    break
        if len(indices) >= 2:
            combos.append((str(row["combination_id"]), frozenset(indices)))
        elif len(indices) == 1:
            combos.append((str(row["combination_id"]), frozenset(indices)))
    return combos  # list of (combo_id, frozenset)


def pair_jaccard_curve(ranked_indices, causative_combos, k_vals):
    """
    At each K: fraction of causative combos whose ALL variants are in top-K.
    Jaccard = |detected| / |causative|  (recall on combos).
    """
    n_causative = len(causative_combos)
    if n_causative == 0:
        return [0.0] * len(k_vals)

    vals = []
    for k in k_vals:
        top_set = set(ranked_indices[:k])
        detected = sum(1 for _, combo in causative_combos if combo.issubset(top_set))
        vals.append(detected / n_causative)
    return vals


# ── PLINK2 helpers (same as run_disease_comparison.py) ────────────────────────

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


def plink2_rank(X_tv, y_tv, ids_tv, var_ids):
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
    if len(sys.argv) < 3:
        print("Usage: run_pairs_detection.py <short_name> <disease_name>")
        sys.exit(1)

    SHORT       = sys.argv[1]
    DISEASE     = sys.argv[2]
    DATA_DIR    = f"dataset/{SHORT}"
    OUT_DIR     = DATA_DIR

    print(f"Loading {SHORT}...")
    X_tr, y_tr, var_ids, ids_tr, novel_start, n_novel = load_split(DATA_DIR, "train")
    X_va, y_va, _,       ids_va, _,            _      = load_split(DATA_DIR, "val")
    X_te, y_te, _,       _,      _,            _      = load_split(DATA_DIR, "test")

    X_tv   = np.concatenate([X_tr, X_va])
    y_tv   = np.concatenate([y_tr, y_va])
    ids_tv = np.concatenate([ids_tr, ids_va])

    # Build causative pairs (combination-level)
    print(f"Building causative pairs for '{DISEASE}'...")
    causative_combos = build_causative_pairs(DISEASE, var_ids)
    print(f"  {len(causative_combos)} combinations, "
          f"{len(set(frozenset(c) for _,c in causative_combos))} unique variant-sets")

    # Show combo sizes
    from collections import Counter
    sizes = Counter(len(c) for _, c in causative_combos)
    print(f"  Combo sizes: {dict(sorted(sizes.items()))}")

    # Show how many variant indices are in each combo
    all_causative_vars = set()
    for _, c in causative_combos:
        all_causative_vars |= c
    print(f"  Total unique causative variant indices: {len(all_causative_vars)}")
    print(f"  Expected novel_start..novel_start+n_novel: {novel_start}..{novel_start+n_novel-1}")
    in_novel = sum(1 for v in all_causative_vars if novel_start <= v < novel_start+n_novel)
    print(f"  Causative variants in OLIDA-novel range: {in_novel}/{len(all_causative_vars)}")

    k_vals = [k for k in K_VALUES if k <= X_tv.shape[1]]

    curves = {}

    # RF
    print("\nRF fitting...")
    rf = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                n_jobs=-1, random_state=42)
    rf.fit(X_tv, y_tv)
    rf_rank = np.argsort(rf.feature_importances_)[::-1]
    curves["Random Forest"] = pair_jaccard_curve(rf_rank, causative_combos, k_vals)

    # Logistic L1
    print("Logistic L1 fitting...")
    X_sc = X_tv.astype(np.float32) / 2.0
    m_l1 = LogisticRegression(penalty="l1", solver="saga", C=0.1,
                               max_iter=2000, random_state=42)
    m_l1.fit(X_sc, y_tv)
    l1_rank = np.argsort(np.abs(m_l1.coef_[0]))[::-1]
    curves["Logistic L1"] = pair_jaccard_curve(l1_rank, causative_combos, k_vals)

    # PLINK2
    print("PLINK2 ranking...")
    p2_rank = plink2_rank(X_tv, y_tv, ids_tv, var_ids)
    if p2_rank is not None:
        curves["PLINK2"] = pair_jaccard_curve(p2_rank, causative_combos, k_vals)

    # Feyn (raw) — load saved ranking from run_feyn_raw.py
    feyn_rank_path = os.path.join(DATA_DIR, "feyn_raw_rank.npy")
    if os.path.exists(feyn_rank_path):
        print("Loading Feyn ranking from file...")
        feyn_rank = np.load(feyn_rank_path)
        curves["Feyn (raw)"] = pair_jaccard_curve(feyn_rank, causative_combos, k_vals)
    else:
        print("Feyn ranking not found — run run_feyn_raw.py first")

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'K':>7}  " + "  ".join(f"{n:>18}" for n in curves))
    print("-" * (9 + 20 * len(curves)))
    for i, k in enumerate(k_vals):
        row = f"{k:>7}  " + "  ".join(f"{curves[n][i]:>18.4f}" for n in curves)
        print(row)

    # ── Plot ──────────────────────────────────────────────────────────────────
    colors = {"Random Forest": "#C44E52", "Logistic L1": "#6EA6CD",
              "PLINK2": "#DA8BC3", "Feyn (raw)": "#DD8452"}
    styles = {"Random Forest": "-", "Logistic L1": "--",
              "PLINK2": ":", "Feyn (raw)": "-"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, vals in curves.items():
        ax.plot(range(len(k_vals)), vals,
                color=colors.get(name, "gray"), ls=styles.get(name, "-"),
                lw=2.2, marker="o", ms=5, label=name)

    # Mark K = total unique causative variants
    n_causative_vars = len(all_causative_vars)
    k_exact_idx = next((i for i, k in enumerate(k_vals) if k >= n_causative_vars),
                        len(k_vals)-1)
    ax.axvline(k_exact_idx, color="gray", ls=":", lw=1.2, alpha=0.7,
               label=f"K = {n_causative_vars} (# causative vars)")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)

    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels([str(k) for k in k_vals], rotation=45, fontsize=8)
    ax.set_xlabel("Top-K features selected", fontsize=11)
    ax.set_ylabel("Fraction of OLIDA combos fully detected", fontsize=10)
    ax.set_title(f"Pair-level detection — {SHORT.upper()}\n"
                 f"Recall: fraction of {len(causative_combos)} combos with ALL variants in top-K",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "pairs_detection.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot → {out}")


if __name__ == "__main__":
    main()
