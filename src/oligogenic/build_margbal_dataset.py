#!/usr/bin/env python3
"""
Build a marginal-balanced (margbal) version of an OLIDA disease dataset.

Problem: in the standard dataset every positive has causative SNPs implanted,
every negative does not → P(SNP_a=1|y=1) ≈ 1, P(SNP_a=1|y=0) ≈ 0.
Feyn can fit the data using individual marginal effects and ignores interactions.

Fix: for every positive case carrying combo {A, B} add pseudo-controls that
carry A alone (B=0) and B alone (A=0), matching 1000G background elsewhere.
After balancing: P(A=1|y) is equal in cases and controls, so A has no marginal
signal. Only the co-occurrence P(A=1 AND B=1|y=1) >> P(A=1 AND B=1|y=0)
remains — forcing Feyn to discover the pair.

Steps:
  1. Keep only cases with multi-SNP combos (size ≥ 2); drop singletons.
  2. For every case with combo C = {s1, s2, ...}:
       for each si in C:
         clone a random negative background, set si=1, all other causatifs=0
         → pseudo-control (y=0)
  3. Merge original negatives + pseudo-controls → new negatives pool.
  4. Save as dataset/<short>_margbal/genotype_{train,val,test}.npz

Usage:
  python build_margbal_dataset.py alport
  python build_margbal_dataset.py fhl
"""

import os, sys, itertools
import numpy as np

SEED = 42

def load_split(data_dir, split):
    d = np.load(os.path.join(data_dir, f"genotype_{split}.npz"), allow_pickle=True)
    return (d["X"].astype(np.int8), d["y"].astype(int),
            d["variant_ids"], d["sample_ids"],
            int(d["novel_start"]), int(d["n_novel"]))


def load_olida_combos_per_sample(X, y, novel_start, n_novel):
    """Return list of (sample_idx, frozenset_of_causative_snp_cols) for each positive."""
    causative = set(range(novel_start, novel_start + n_novel))
    result = []
    for i in range(len(y)):
        if y[i] == 1:
            snps = frozenset(j for j in causative if X[i, j] > 0)
            result.append((i, snps))
    return result


def build_margbal_split(X, y, novel_start, n_novel, rng):
    """
    Returns new (X_out, y_out) with:
      - only multi-SNP cases kept as positives
      - ONLY pseudo-controls as negatives (original 1000G negatives discarded)
        → each case {A,B} generates: pseudo(A=1,B=0) and pseudo(A=0,B=1)
        → P(A=1|neg) = 0.5 independent of case count,  max marginal Δ = 0.5
    """
    causative_cols = list(range(novel_start, novel_start + n_novel))

    neg_idx = np.where(y == 0)[0]
    X_neg   = X[neg_idx]               # background pool for non-causative features

    combos = load_olida_combos_per_sample(X, y, novel_start, n_novel)
    multi  = [(i, c) for i, c in combos if len(c) >= 2]

    if not multi:
        print("  No multi-SNP combos found — margbal dataset is empty of positives")
        return X, y

    # Keep only multi-SNP cases
    multi_case_idx = [i for i, _ in multi]
    X_cases = X[multi_case_idx]
    y_cases = np.ones(len(multi_case_idx), dtype=int)

    # Build pseudo-controls (one per SNP per case, no original 1000G negatives)
    pseudo_X = []
    for _, combo in multi:
        for snp in combo:
            bg_i = rng.integers(0, len(X_neg))
            pc   = X_neg[bg_i].copy()
            for c in causative_cols:     # zero ALL causative, then set only this SNP
                pc[c] = 0
            pc[snp] = 1
            pseudo_X.append(pc)

    X_pseudo = np.array(pseudo_X, dtype=np.int8)
    y_pseudo = np.zeros(len(X_pseudo), dtype=int)

    print(f"  Multi-SNP cases (pos):   {len(X_cases)}")
    print(f"  Pseudo-controls (neg):   {len(X_pseudo)}  (original 1000G negatives dropped)")

    X_out = np.concatenate([X_cases, X_pseudo], axis=0)
    y_out = np.concatenate([y_cases, y_pseudo])

    perm  = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def main():
    short    = sys.argv[1] if len(sys.argv) > 1 else "alport"
    src_dir  = f"dataset/{short}"
    dst_dir  = f"dataset/{short}_margbal"
    os.makedirs(dst_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)

    print(f"\nBuilding margbal dataset: {short} → {dst_dir}")

    # Load one split to get metadata
    X_tr, y_tr, var_ids, sids_tr, novel_start, n_novel = load_split(src_dir, "train")
    X_va, y_va, _,       sids_va, _,            _      = load_split(src_dir, "val")
    X_te, y_te, _,       sids_te, _,            _      = load_split(src_dir, "test")

    causative = set(range(novel_start, novel_start + n_novel))
    print(f"p={X_tr.shape[1]}, n_novel={n_novel}, novel_start={novel_start}")

    # Report original combo sizes
    for split_name, X, y in [("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)]:
        combos = load_olida_combos_per_sample(X, y, novel_start, n_novel)
        from collections import Counter
        sizes = Counter(len(c) for _, c in combos)
        print(f"  {split_name}: {len(combos)} positive samples, combo sizes={dict(sorted(sizes.items()))}")

    print()
    for split_name, X, y, sids in [("train", X_tr, y_tr, sids_tr),
                                    ("val",   X_va, y_va, sids_va),
                                    ("test",  X_te, y_te, sids_te)]:
        print(f"Processing {split_name}...")
        X_new, y_new = build_margbal_split(X, y, novel_start, n_novel, rng)
        print(f"  → shape={X_new.shape}, pos={y_new.sum()}, neg={(y_new==0).sum()}")

        # Verify marginal balancing on train
        if split_name == "train":
            causative_cols = list(range(novel_start, novel_start + n_novel))
            pos_mask = y_new == 1
            neg_mask = y_new == 0
            marginals_pos = X_new[pos_mask][:, causative_cols].astype(float).mean(axis=0)
            marginals_neg = X_new[neg_mask][:, causative_cols].astype(float).mean(axis=0)
            max_diff = np.max(np.abs(marginals_pos - marginals_neg))
            print(f"  Marginal balance check (train): max |P(snp|pos)-P(snp|neg)| = {max_diff:.3f}")

        new_sids = np.array([f"MB_{i:06d}" for i in range(len(y_new))])
        np.savez_compressed(
            os.path.join(dst_dir, f"genotype_{split_name}.npz"),
            X=X_new, y=y_new,
            variant_ids=var_ids,
            sample_ids=new_sids,
            novel_start=np.array(novel_start),
            n_novel=np.array(n_novel)
        )

    print(f"\nSaved → {dst_dir}/genotype_{{train,val,test}}.npz")

    # Summary of truth pairs
    X_all = np.concatenate([X_tr, X_va, X_te])
    y_all = np.concatenate([y_tr, y_va, y_te])
    combos_all = load_olida_combos_per_sample(X_all, y_all, novel_start, n_novel)
    seen, unique_combos = set(), []
    for _, c in combos_all:
        if len(c) >= 2 and c not in seen:
            seen.add(c); unique_combos.append(c)

    truth_pairs = set()
    for c in unique_combos:
        for a, b in itertools.combinations(sorted(c), 2):
            truth_pairs.add(frozenset([a, b]))
    print(f"\nMulti-SNP combos: {len(unique_combos)}, truth pairs: {len(truth_pairs)}")
    for p in sorted(truth_pairs, key=lambda x: min(x)):
        print(f"  {sorted(p)}")


if __name__ == "__main__":
    main()
