#!/usr/bin/env python3
"""
Build margbal-AND versions of the real datasets:
  - positives: real OLIDA multi-SNP cases (kept as-is)
  - pseudo-controls: for each positive carrying combo C of size k:
      → generate k pseudo-controls, each with ONE SNP from C active +
        (k-1) random non-pair causative SNPs active (decoys), others zeroed
      → matches the count of active causatives in pos/pseudo
      → breaks linear separability via sum-of-causatives

Usage: python build_margbal_AND.py <short_name>  (e.g. longqt -> dataset/longqt_margbal_AND/)
"""
import os, sys
import numpy as np

SEED = 42


def load_split(data_dir, split):
    d = np.load(os.path.join(data_dir, f"genotype_{split}.npz"), allow_pickle=True)
    return d["X"], d["y"], d["variant_ids"], d["sample_ids"], int(d["novel_start"]), int(d["n_novel"])


def load_olida_combos_per_sample(X, y, novel_start, n_novel):
    causative = set(range(novel_start, novel_start + n_novel))
    combos = []
    for i in range(len(y)):
        if y[i] == 1:
            fs = frozenset(j for j in causative if X[i, j] > 0)
            combos.append((i, fs))
    return combos


def build_AND_split(X, y, novel_start, n_novel, rng):
    causative_cols = list(range(novel_start, novel_start + n_novel))
    neg_idx = np.where(y == 0)[0]
    X_neg   = X[neg_idx]

    combos = load_olida_combos_per_sample(X, y, novel_start, n_novel)
    multi  = [(i, c) for i, c in combos if len(c) >= 2]
    if not multi:
        return X, y

    multi_case_idx = [i for i, _ in multi]
    X_cases = X[multi_case_idx]
    y_cases = np.ones(len(multi_case_idx), dtype=int)

    pseudo_X = []
    for _, combo in multi:
        combo_list = sorted(combo)
        k = len(combo_list)
        for snp in combo_list:
            bg_i = rng.integers(0, len(X_neg))
            pc   = X_neg[bg_i].copy()
            for c in causative_cols:
                pc[c] = 0
            pc[snp] = 1
            # Add k-1 decoy causatives outside the combo
            decoys_pool = [c for c in causative_cols if c not in combo]
            if len(decoys_pool) >= k - 1:
                decoys = rng.choice(decoys_pool, size=k - 1, replace=False)
            elif decoys_pool:
                decoys = rng.choice(decoys_pool, size=k - 1, replace=True)
            else:
                decoys = []
            for dc in decoys:
                pc[dc] = 1
            pseudo_X.append(pc)

    X_pseudo = np.array(pseudo_X, dtype=np.int8)
    y_pseudo = np.zeros(len(X_pseudo), dtype=int)
    return (np.concatenate([X_cases, X_pseudo], axis=0),
            np.concatenate([y_cases, y_pseudo]))


def main():
    SHORT = sys.argv[1]
    SRC = f"dataset/{SHORT}"
    DST = f"dataset/{SHORT}_margbal_AND"
    os.makedirs(DST, exist_ok=True)
    rng = np.random.default_rng(SEED)

    for split in ("train", "val", "test"):
        X, y, var_ids, sample_ids, novel_start, n_novel = load_split(SRC, split)
        X_out, y_out = build_AND_split(X, y, novel_start, n_novel, rng)
        sample_ids_out = np.array([f"{SHORT}AND_{split}_{i:06d}" for i in range(len(y_out))])
        np.savez_compressed(
            os.path.join(DST, f"genotype_{split}.npz"),
            X=X_out, y=y_out,
            variant_ids=var_ids, sample_ids=sample_ids_out,
            novel_start=np.array(novel_start),
            n_novel=np.array(n_novel))
        print(f"  {split}: {(y_out==1).sum()} pos / {(y_out==0).sum()} neg → {DST}/")


if __name__ == "__main__":
    main()
