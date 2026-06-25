#!/usr/bin/env python3
"""
Margbal-faithful uniform sweep:
  - For each truth pair (a,b), positives use REAL margbal positives carrying
    that pair as background (sampled with replacement to reach npp). Only
    OTHER causative cols are zeroed; non-causative background = real patient.
  - Pseudo-controls = 1000G-background (margbal neg pool) + one focal SNP,
    all causative zeroed otherwise (same as margbal construction).

For K > available real pairs (>9 in LongQT margbal), falls back to neg-pool
background (matches the previous build_uniform construction for those).

K=2..9 x npp=5..30 step 5 (48 cells). Writes to uniform_surface_margbal.csv.
"""
import os, time
import numpy as np
import pandas as pd

import run_uniform_surface as R

R.OUT_CSV    = "uniform_surface_margbal.csv"
R.K_VALS     = [2, 3, 4, 5, 6, 7, 8, 9]
R.NPP_VALS   = [5, 10, 15, 20, 25, 30]
R.RUN_PLINK  = True


def build_uniform_margbal(X_src, y_src, novel_start, n_novel, K, npp, rng):
    """Margbal-style construction with uniform pair counts.
    - positives: real margbal cases sampled WITH replacement so each focal pair
      has exactly npp positives. Full genotype KEPT (nothing zeroed).
    - pseudo-controls: 1000G background + ONE causative SNP (margbal-style),
      one per causative SNP active in the sampled positive.
    """
    pairs           = R.LONGQT_PAIRS[:K]
    causative_cols  = list(range(novel_start, novel_start + n_novel))

    pos_pool = X_src[y_src == 1]
    neg_pool = X_src[y_src == 0]

    pos_list, pseudo_list = [], []
    for a, b in pairs:
        avail = np.where((pos_pool[:, a] > 0) & (pos_pool[:, b] > 0))[0]
        if len(avail) == 0:
            # Fallback: synthetic positive from neg-pool (only K>9 hits this)
            for _ in range(npp):
                bg = neg_pool[rng.integers(0, len(neg_pool))].copy()
                for c in causative_cols:
                    bg[c] = 0
                bg[a] = 1; bg[b] = 1
                pos_list.append(bg)
                for snp in (a, b):
                    pc = neg_pool[rng.integers(0, len(neg_pool))].copy()
                    for c in causative_cols:
                        pc[c] = 0
                    pc[snp] = 1
                    pseudo_list.append(pc)
        else:
            picks = rng.choice(avail, size=npp, replace=True)
            for pi in picks:
                pos_instance = pos_pool[pi].copy()
                pos_list.append(pos_instance)
                active = [c for c in causative_cols if pos_instance[c] > 0]
                for snp in active:
                    pc = neg_pool[rng.integers(0, len(neg_pool))].copy()
                    for c in causative_cols:
                        pc[c] = 0
                    pc[snp] = 1
                    pseudo_list.append(pc)

    X_out = np.concatenate(
        [np.array(pos_list,    dtype=np.int8),
         np.array(pseudo_list, dtype=np.int8)], axis=0)
    y_out = np.concatenate(
        [np.ones(len(pos_list),    dtype=int),
         np.zeros(len(pseudo_list), dtype=int)])
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm], [frozenset(p) for p in pairs]


# Patch R.build_uniform so R.main uses our margbal-faithful builder
R.build_uniform = build_uniform_margbal


if __name__ == "__main__":
    if os.path.exists(R.OUT_CSV):
        os.remove(R.OUT_CSV)
        print(f"Removed stale {R.OUT_CSV}")
    R.main()
