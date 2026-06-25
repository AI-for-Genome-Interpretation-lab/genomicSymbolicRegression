#!/usr/bin/env python3
"""
Compute Jaccard SNP@K and Combo Jaccard for RF, PLINK2, and Feyn c10
across baseline and fpbal datasets for SCA17, FHL, Alport.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATASETS = [
    ("sca17",       "SCA17 baseline"),
    ("sca17_fpbal", "SCA17 fpbal"),
    ("fhl",         "FHL baseline"),
    ("fhl_fpbal",   "FHL fpbal"),
    ("alport",      "Alport baseline"),
    ("alport_fpbal","Alport fpbal"),
]

K_VALS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def load_split(data_dir, name):
    d = np.load(os.path.join(data_dir, f"genotype_{name}.npz"), allow_pickle=True)
    return (d["X"].astype(np.float32), d["y"].astype(int),
            d["variant_ids"], d["sample_ids"],
            int(d["novel_start"]), int(d["n_novel"]))


def jaccard(s1, s2):
    s1, s2 = set(s1), set(s2)
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)


def snp_jaccard_at_k(ranked_indices, causative_set, k):
    top_k = set(ranked_indices[:k])
    return jaccard(top_k, causative_set)


def combo_jaccard(ranked_indices, olida_combos, k):
    """
    For each OLIDA combination (set of SNP indices), check if ALL its SNPs
    are in top-K. Combo Jac = |fully recovered combos| / |total combos|.
    """
    top_k = set(ranked_indices[:k])
    recovered = sum(1 for combo in olida_combos if combo.issubset(top_k))
    return recovered / len(olida_combos) if olida_combos else 0.0


def load_olida_combos(data_dir, novel_start, n_novel, variant_ids):
    """
    Approximate OLIDA combinations from the positive samples in train+val:
    each positive sample's set of novel SNPs (dosage > 0) is one combo.
    """
    combos = []
    causative_set = set(range(novel_start, novel_start + n_novel))
    for split in ["train", "val", "test"]:
        d = np.load(os.path.join(data_dir, f"genotype_{split}.npz"), allow_pickle=True)
        X = d["X"].astype(np.float32)
        y = d["y"].astype(int)
        for i in range(len(y)):
            if y[i] == 1:
                novel_idxs = frozenset(
                    j for j in causative_set if X[i, j] > 0
                )
                if novel_idxs and novel_idxs not in combos:
                    combos.append(novel_idxs)
    return combos


def get_rf_ranking(X_tv, y_tv, n_feat):
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X_tv, y_tv)
    imp = rf.feature_importances_
    return np.argsort(imp)[::-1]


def get_plink2_ranking(X_tv, y_tv):
    corr = np.array([
        abs(np.corrcoef(X_tv[:, i], y_tv)[0, 1]) if X_tv[:, i].std() > 0 else 0.0
        for i in range(X_tv.shape[1])
    ])
    return np.argsort(corr)[::-1]


def get_feyn_ranking(data_dir, cx_tag="_c10"):
    rank_file = os.path.join(data_dir, f"feyn_raw_rank{cx_tag}.npy")
    if os.path.exists(rank_file):
        return np.load(rank_file)
    return None


def main():
    base = "dataset"
    rows = []

    for short, label in DATASETS:
        data_dir = os.path.join(base, short)
        print(f"\n{'='*50}")
        print(f"Processing: {label} ({short})")

        X_tr, y_tr, var_ids, ids_tr, novel_start, n_novel = load_split(data_dir, "train")
        X_va, y_va, _, _, _, _ = load_split(data_dir, "val")
        X_te, y_te, _, _, _, _ = load_split(data_dir, "test")

        X_tv = np.concatenate([X_tr, X_va])
        y_tv = np.concatenate([y_tr, y_va])

        causative_set = set(range(novel_start, novel_start + n_novel))
        olida_combos = load_olida_combos(data_dir, novel_start, n_novel, var_ids)
        n_feat = X_tr.shape[1]

        print(f"  Features: {n_feat}, causative: {n_novel}, combos: {len(olida_combos)}")

        # K values up to n_feat
        k_vals = [k for k in K_VALS if k <= n_feat]
        # Use n_novel as the "evaluation K" for SNP Jaccard
        k_snp = n_novel

        # ── RF ranking ──
        print("  Computing RF ranking...")
        rf_rank = get_rf_ranking(X_tv / 2.0, y_tv, n_feat)
        rf_snp_j = snp_jaccard_at_k(rf_rank, causative_set, k_snp)
        rf_combo_j = combo_jaccard(rf_rank, olida_combos, k_snp)
        print(f"  RF  SNP@{k_snp}={rf_snp_j:.3f}  Combo={rf_combo_j:.3f}")

        # ── PLINK2 ranking ──
        print("  Computing PLINK2 ranking...")
        plink_rank = get_plink2_ranking(X_tv / 2.0, y_tv)
        plink_snp_j = snp_jaccard_at_k(plink_rank, causative_set, k_snp)
        plink_combo_j = combo_jaccard(plink_rank, olida_combos, k_snp)
        print(f"  PLK SNP@{k_snp}={plink_snp_j:.3f}  Combo={plink_combo_j:.3f}")

        # ── Feyn c10 ranking ──
        feyn_rank = get_feyn_ranking(data_dir, "_c10")
        if feyn_rank is None:
            feyn_rank = get_feyn_ranking(data_dir, "")
        if feyn_rank is not None:
            feyn_snp_j = snp_jaccard_at_k(feyn_rank, causative_set, k_snp)
            feyn_combo_j = combo_jaccard(feyn_rank, olida_combos, k_snp)
            print(f"  FYN SNP@{k_snp}={feyn_snp_j:.3f}  Combo={feyn_combo_j:.3f}")
        else:
            feyn_snp_j = feyn_combo_j = None
            print("  FYN ranking not found")

        rows.append({
            "dataset": label,
            "n_novel": n_novel,
            "n_combos": len(olida_combos),
            "k_eval": k_snp,
            "RF_snp_jac":    rf_snp_j,
            "RF_combo_jac":  rf_combo_j,
            "PLINK2_snp_jac":  plink_snp_j,
            "PLINK2_combo_jac":plink_combo_j,
            "Feyn_snp_jac":  feyn_snp_j,
            "Feyn_combo_jac":feyn_combo_j,
        })

    df = pd.DataFrame(rows)
    print("\n\n" + "="*80)
    print("JACCARD COMPARISON: RF vs PLINK2 vs Feyn c10")
    print("="*80)
    print(f"\n{'Dataset':<20} {'K':>4} {'RF SNP':>8} {'RF Comb':>8} {'PLK SNP':>8} {'PLK Comb':>9} {'FYN SNP':>8} {'FYN Comb':>9}")
    print("-"*80)
    for _, r in df.iterrows():
        fyn_s = f"{r['Feyn_snp_jac']:.3f}" if r['Feyn_snp_jac'] is not None else "  N/A"
        fyn_c = f"{r['Feyn_combo_jac']:.3f}" if r['Feyn_combo_jac'] is not None else "  N/A"
        print(f"{r['dataset']:<20} {r['k_eval']:>4} "
              f"{r['RF_snp_jac']:>8.3f} {r['RF_combo_jac']:>8.3f} "
              f"{r['PLINK2_snp_jac']:>8.3f} {r['PLINK2_combo_jac']:>9.3f} "
              f"{fyn_s:>8} {fyn_c:>9}")

    out = "jaccard_rf_plink_feyn.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
