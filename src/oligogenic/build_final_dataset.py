#!/usr/bin/env python3
"""
Step 4 — Build final dataset from positives + negatives.
Step 5 — Validate and produce stats report.

Input:
  dataset/processed/olida_positives.tsv
  dataset/processed/negatives.tsv

Output:
  dataset/processed/final_dataset.tsv
  dataset/splits/train.tsv
  dataset/splits/val.tsv
  dataset/splits/test.tsv
  dataset/stats/dataset_report.txt
"""

import os, sys
import pandas as pd
import numpy as np
from collections import defaultdict

POS_FILE     = "dataset/processed/olida_positives.tsv"
NEG_FILE     = "dataset/processed/negatives.tsv"
FINAL_FILE   = "dataset/processed/final_dataset.tsv"
SPLITS_DIR   = "dataset/splits"
STATS_DIR    = "dataset/stats"

os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15


def stratified_split_by_disease(df, train_f, val_f, test_f, seed=42):
    """
    Split stratified by disease_name so that all combinations of the same
    disease end up in the same fold (avoids data leakage).
    Negatives (label=0) are split randomly to match the split fractions.
    """
    rng = np.random.default_rng(seed)

    # Split positives by disease group
    pos = df[df["label"] == 1].copy()
    neg = df[df["label"] == 0].copy()

    diseases = pos["disease_name"].unique()
    rng.shuffle(diseases)
    n = len(diseases)
    n_train = int(n * train_f)
    n_val   = int(n * val_f)

    train_dis = set(diseases[:n_train])
    val_dis   = set(diseases[n_train:n_train + n_val])
    test_dis  = set(diseases[n_train + n_val:])

    pos_train = pos[pos["disease_name"].isin(train_dis)]
    pos_val   = pos[pos["disease_name"].isin(val_dis)]
    pos_test  = pos[pos["disease_name"].isin(test_dis)]

    # Split negatives randomly (proportional) — gene-pair leakage is documented
    neg = neg.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_neg = len(neg)
    n_neg_train = int(n_neg * train_f)
    n_neg_val   = int(n_neg * val_f)
    neg_train = neg.iloc[:n_neg_train]
    neg_val   = neg.iloc[n_neg_train:n_neg_train + n_neg_val]
    neg_test  = neg.iloc[n_neg_train + n_neg_val:]

    train = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=seed).reset_index(drop=True)
    val   = pd.concat([pos_val,   neg_val  ]).sample(frac=1, random_state=seed).reset_index(drop=True)
    test  = pd.concat([pos_test,  neg_test ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    return train, val, test


def check_leakage(train, val, test):
    """Check that gene pairs from train don't appear in test."""
    def gene_pairs(df):
        pairs = set()
        for _, row in df.iterrows():
            genes = [g.strip() for g in str(row["genes"]).split(";") if g.strip()]
            if len(genes) >= 2:
                pairs.add(tuple(sorted([genes[0], genes[1]])))
        return pairs

    train_pairs = gene_pairs(train)
    test_pairs  = gene_pairs(test)
    leakage = train_pairs & test_pairs
    return leakage


def write_report(positives, negatives, final, train, val, test, leakage, path):
    lines = []
    lines.append("=" * 60)
    lines.append("DATASET REPORT")
    lines.append("=" * 60)
    lines.append(f"\nTotal samples:    {len(final)}")
    lines.append(f"  Positives:      {(final['label']==1).sum()}")
    lines.append(f"  Negatives:      {(final['label']==0).sum()}")
    lines.append(f"\nSplit sizes:")
    lines.append(f"  Train:          {len(train)} ({100*len(train)/len(final):.1f}%)")
    lines.append(f"  Val:            {len(val)}   ({100*len(val)/len(final):.1f}%)")
    lines.append(f"  Test:           {len(test)}  ({100*len(test)/len(final):.1f}%)")
    lines.append(f"\nClass balance in train set:")
    lines.append(f"  Positives:      {(train['label']==1).sum()}")
    lines.append(f"  Negatives:      {(train['label']==0).sum()}")

    lines.append(f"\nAllelic state distribution (positives):")
    for k, v in positives["allelic_state"].value_counts().items():
        lines.append(f"  {k}: {v}")

    lines.append(f"\nFINALmeta distribution (positives):")
    for k, v in sorted(positives["finalMeta_score"].value_counts().items()):
        lines.append(f"  score={k}: {v}")

    lines.append(f"\nTop 15 diseases by n combinations:")
    for k, v in positives["disease_name"].value_counts().head(15).items():
        lines.append(f"  {k}: {v}")

    lines.append(f"\nOligogenic effect distribution (positives):")
    for k, v in positives["oligogenic_effect"].value_counts().items():
        lines.append(f"  {k}: {v}")

    lines.append(f"\nSuperpopulation (negatives):")
    for k, v in negatives["population"].value_counts().items():
        lines.append(f"  {k}: {v}")

    lines.append(f"\nData leakage check (gene pairs in both train and test):")
    if leakage:
        lines.append(f"  WARNING: {len(leakage)} overlapping gene pairs found:")
        for p in sorted(leakage)[:10]:
            lines.append(f"    {p[0]} x {p[1]}")
    else:
        lines.append("  OK — no gene pair appears in both train and test")

    lines.append(f"\n{'=' * 60}")
    lines.append("BIAS NOTES")
    lines.append("=" * 60)
    lines.append("""
- Ascertainment bias: OLIDA contains only published cases.
- Population bias: 1000G 'negatives' may include undiagnosed affected.
- 50% prevalence is artificial (real: 1/2500 - 1/160000).
- For clean benchmark: use only combinations added to OLIDA post-2019
  (to avoid overlap with VarCoPP/DIDA training sets).
""")

    report = "\n".join(lines)
    with open(path, "w") as f:
        f.write(report)
    print(report)


def main():
    # ── Load ──────────────────────────────────────────────────────────────────
    if not os.path.exists(POS_FILE):
        print(f"ERROR: {POS_FILE} not found. Run parse_olida.py first.")
        sys.exit(1)
    if not os.path.exists(NEG_FILE):
        print(f"ERROR: {NEG_FILE} not found. Run build_negatives.py first.")
        sys.exit(1)

    positives = pd.read_csv(POS_FILE, sep="\t")
    negatives = pd.read_csv(NEG_FILE, sep="\t")
    print(f"Positives: {len(positives)}, Negatives: {len(negatives)}")

    # ── Validate no variant overlap ───────────────────────────────────────────
    def var_keys(df, label):
        keys = set()
        for i in range(1, 5):
            pfx = f"var{i}_"
            for _, row in df.iterrows():
                chrom = str(row.get(pfx + "chrom", "N.A."))
                pos   = str(row.get(pfx + "pos_hg19", "N.A."))
                if chrom not in ("N.A.", "nan", "") and pos not in ("N.A.", "nan", ""):
                    keys.add((chrom, pos))
        return keys

    pos_variants = var_keys(positives, 1)
    neg_variants = var_keys(negatives, 0)
    overlap = pos_variants & neg_variants
    if overlap:
        print(f"WARNING: {len(overlap)} positions appear in both positives and negatives!")
    else:
        print("OK: no variant position overlap between positives and negatives")

    # ── Balance at 50/50 ─────────────────────────────────────────────────────
    n = min(len(positives), len(negatives))
    pos_balanced = positives.sample(n=n, random_state=42) if len(positives) > n else positives
    neg_balanced = negatives.sample(n=n, random_state=42) if len(negatives) > n else negatives
    print(f"Balanced dataset: {n} positives + {n} negatives = {2*n} total")

    # ── Merge and shuffle ─────────────────────────────────────────────────────
    final = pd.concat([pos_balanced, neg_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)
    final["sample_id"] = [f"SIM_{i:05d}" for i in range(len(final))]
    final.to_csv(FINAL_FILE, sep="\t", index=False)
    print(f"Final dataset saved → {FINAL_FILE}")

    # ── Split ─────────────────────────────────────────────────────────────────
    train, val, test = stratified_split_by_disease(final, TRAIN_FRAC, VAL_FRAC, TEST_FRAC)
    train.to_csv(os.path.join(SPLITS_DIR, "train.tsv"), sep="\t", index=False)
    val.to_csv(os.path.join(SPLITS_DIR, "val.tsv"),   sep="\t", index=False)
    test.to_csv(os.path.join(SPLITS_DIR, "test.tsv"),  sep="\t", index=False)
    print(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")

    # ── Leakage check ─────────────────────────────────────────────────────────
    leakage = check_leakage(train, val, test)

    # ── Report ────────────────────────────────────────────────────────────────
    report_path = os.path.join(STATS_DIR, "dataset_report.txt")
    write_report(positives, negatives, final, train, val, test, leakage, report_path)
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()
