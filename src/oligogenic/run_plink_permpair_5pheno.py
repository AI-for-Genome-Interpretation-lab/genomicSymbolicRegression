#!/usr/bin/env python3
"""Run PLINK1.9 --fast-epistasis on the 5 permpair-FIXED margbal datasets
that appear in jaccardHeatmap.png, so the Feyn vs PLINK comparison is
apples-to-apples (same controls).
"""
import os, sys
import pandas as pd

sys.path.insert(0, ".")
from compute_combo_jaccard_feyn_plink import (
    load_tv, load_olida_combos, combos_to_pairs, jaccard_sets,
    plink_run, EPI_P_THRESH,
)

PHENOS = [
    ("hypodontia_margbal_permpair",       "Hypodontia"),
    ("longqt_margbal_permpair",           "LongQT"),
    ("autoinflammatory_margbal_permpair", "Autoinflammatory"),
    ("fevr_margbal_permpair",             "FEVR"),
    ("hypo_margbal_permpair",             "Hypogonadism"),
]

rows = []
for short, label in PHENOS:
    dd = f"dataset/{short}"
    X, y, n_novel, novel_start = load_tv(dd)
    combos = load_olida_combos(dd, novel_start, n_novel)
    truth = combos_to_pairs(combos)
    K = len(truth)
    print(f"\n=== {label}  K={K}  n_pos={int(y.sum())}  n_tot={len(y)} ===")
    plink_rows = plink_run(X, y, n_novel, novel_start)
    detected = {pair for pv, pair in plink_rows if pv < EPI_P_THRESH}
    J = jaccard_sets(detected, truth) if detected else 0.0
    print(f"  PLINK: detected={len(detected)}, J={J:.3f}")
    rows.append(dict(dataset=label, K=K, PLINK_n=len(detected), PLINK_J=J))

pd.DataFrame(rows).to_csv("combo_jaccard_plink_permpair_5pheno.csv", index=False)
print("\nSaved → combo_jaccard_plink_permpair_5pheno.csv")
