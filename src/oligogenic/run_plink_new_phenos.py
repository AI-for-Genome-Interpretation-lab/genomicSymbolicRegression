#!/usr/bin/env python3
"""Run PLINK epistasis for the 6 new phenotypes margbal datasets.
Writes combo_jaccard_plink_new.csv with K, PLINK_n, PLINK_J columns.
"""
import os, sys, subprocess, tempfile, shutil, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from compute_combo_jaccard_feyn_plink import (
    load_tv, load_olida_combos, combos_to_pairs, jaccard_sets,
    write_ped_map, plink_run, EPI_P_THRESH, LARGE_THR, LARGE_K,
)

PHENOS = [
    ("autoinflammatory_margbal", "Autoinflammatory"),
    ("hypoleft_margbal",         "Hypoleft"),
    ("ocalbinism_margbal",       "Ocalbinism"),
    ("cystinuria_margbal",       "Cystinuria"),
    ("glaucoma_margbal",         "Glaucoma"),
    ("deafness_margbal",         "Deafness"),
]

rows = []
for short, label in PHENOS:
    dd = f"dataset/{short}"
    if not os.path.isdir(dd):
        print(f"{label}: dir missing — skip"); continue
    X, y, n_novel, novel_start = load_tv(dd)
    causative = set(range(novel_start, novel_start + n_novel))
    combos = load_olida_combos(dd, novel_start, n_novel)
    truth = combos_to_pairs(combos)
    K = len(truth)
    print(f"\n=== {label}  K={K} ===")
    if K == 0:
        rows.append(dict(dataset=label, K=K, PLINK_n=0, PLINK_J=float("nan")))
        continue
    plink_rows = plink_run(X, y, n_novel, novel_start)
    detected = {pair for pv, pair in plink_rows if pv < EPI_P_THRESH}
    J = jaccard_sets(detected, truth) if detected else 0.0
    print(f"  PLINK: detected={len(detected)}, J={J:.3f}")
    rows.append(dict(dataset=label, K=K, PLINK_n=len(detected), PLINK_J=J))

pd.DataFrame(rows).to_csv("combo_jaccard_plink_new.csv", index=False)
print("\nSaved → combo_jaccard_plink_new.csv")
