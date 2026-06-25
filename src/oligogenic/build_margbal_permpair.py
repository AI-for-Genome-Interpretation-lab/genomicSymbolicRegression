#!/usr/bin/env python3
"""
Build permutation-pair margbal: pseudo-controls have a RANDOM non-truth pair
of causative SNPs active (instead of one focal SNP). Both pos and pseudo have
sum_causative = combo_size, but pseudo never reproduces a truth pair.

For each pos carrying combo C (size k):
  - Compute pseudo_per_pos = k (matches margbal cardinality)
  - For each pseudo: pick a random pair (c, d) from C(causatives,2) \\ truth_pair_set
  - Activate v_c=v_d=1, all other causative=0

Usage: python build_margbal_permpair.py <short>  → dataset/<short>_margbal_permpair/
"""
import os, sys, itertools
import numpy as np

SEED = 42


def load_split(data_dir, split):
    d = np.load(os.path.join(data_dir, f"genotype_{split}.npz"), allow_pickle=True)
    return d["X"], d["y"], d["variant_ids"], d["sample_ids"], int(d["novel_start"]), int(d["n_novel"])


def load_olida_combos_per_sample(X, y, novel_start, n_novel):
    causative = set(range(novel_start, novel_start + n_novel))
    return [(i, frozenset(j for j in causative if X[i, j] > 0))
            for i in range(len(y)) if y[i] == 1]


def truth_pairs_from(combos):
    out = set()
    for _, c in combos:
        for a, b in itertools.combinations(sorted(c), 2):
            out.add(frozenset([a, b]))
    return out


def build_perm_split(X, y, novel_start, n_novel, rng, truth_pair_set, truth_snps):
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

    # Restrict the non-truth pair pool to ONLY the SNPs that appear in truth.
    # This avoids leaking "never-in-pos" SNPs that Feyn would use as additive
    # neg-only markers.
    truth_snps_sorted = sorted(truth_snps)
    all_pairs = list(itertools.combinations(truth_snps_sorted, 2))
    non_truth_pairs = [(a, b) for a, b in all_pairs
                       if frozenset([a, b]) not in truth_pair_set]
    if not non_truth_pairs:
        print(f"  WARNING: all C(truth_snps, 2) are truth pairs — fallback")
        non_truth_pairs = all_pairs

    pseudo_X = []
    for _, combo in multi:
        k = len(combo)
        for _ in range(k):
            bg_i = rng.integers(0, len(X_neg))
            pc   = X_neg[bg_i].copy()
            for c in causative_cols:
                pc[c] = 0
            idx = rng.integers(0, len(non_truth_pairs))
            a, b = non_truth_pairs[idx]
            pc[a] = 1; pc[b] = 1
            pseudo_X.append(pc)

    X_pseudo = np.array(pseudo_X, dtype=np.int8)
    y_pseudo = np.zeros(len(X_pseudo), dtype=int)
    return (np.concatenate([X_cases, X_pseudo], axis=0),
            np.concatenate([y_cases, y_pseudo]))


def main():
    SHORT = sys.argv[1]
    SRC = f"dataset/{SHORT}"
    DST = f"dataset/{SHORT}_margbal_permpair"
    os.makedirs(DST, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Compute truth pairs from all splits combined (so we exclude all real combos)
    Xs, ys = [], []
    for sp in ("train", "val", "test"):
        d = np.load(os.path.join(SRC, f"genotype_{sp}.npz"), allow_pickle=True)
        Xs.append(d["X"]); ys.append(d["y"])
        novel_start = int(d["novel_start"]); n_novel = int(d["n_novel"])
    X_all = np.concatenate(Xs); y_all = np.concatenate(ys)
    truth_pair_set = truth_pairs_from(load_olida_combos_per_sample(X_all, y_all, novel_start, n_novel))
    truth_snps = set()
    for p in truth_pair_set:
        truth_snps |= p
    print(f"  Truth pair set size: {len(truth_pair_set)}")
    print(f"  Truth SNPs (used in any truth pair): {len(truth_snps)} of {n_novel} causative")

    for split in ("train", "val", "test"):
        X, y, var_ids, sample_ids, novel_start, n_novel = load_split(SRC, split)
        X_out, y_out = build_perm_split(X, y, novel_start, n_novel, rng, truth_pair_set, truth_snps)
        sample_ids_out = np.array([f"{SHORT}PP_{split}_{i:06d}" for i in range(len(y_out))])
        np.savez_compressed(
            os.path.join(DST, f"genotype_{split}.npz"),
            X=X_out, y=y_out,
            variant_ids=var_ids, sample_ids=sample_ids_out,
            novel_start=np.array(novel_start),
            n_novel=np.array(n_novel))
        print(f"  {split}: {(y_out==1).sum()} pos / {(y_out==0).sum()} neg → {DST}/")


if __name__ == "__main__":
    main()
