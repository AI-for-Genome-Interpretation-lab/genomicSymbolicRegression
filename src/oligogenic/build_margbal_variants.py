#!/usr/bin/env python3
"""
Build margbal dataset variants to test different negative generation strategies.

Variants:
  k5        : K_PSEUDO=5  (5 pseudo-controls per SNP per case, instead of 1)
  wrongpair : add wrong-pair negatives (2 causative SNPs from different pairs)
  samebg    : pseudo-controls use same background as the corresponding positive

Usage:
  python build_margbal_variants.py longqt k5
  python build_margbal_variants.py alport wrongpair
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
    causative = set(range(novel_start, novel_start + n_novel))
    result = []
    for i in range(len(y)):
        if y[i] == 1:
            snps = frozenset(j for j in causative if X[i, j] > 0)
            result.append((i, snps))
    return result


def get_truth_pairs(combos):
    pairs = set()
    for _, combo in combos:
        if len(combo) >= 2:
            for a, b in itertools.combinations(sorted(combo), 2):
                pairs.add(frozenset([a, b]))
    return pairs


def build_variant(X, y, novel_start, n_novel, rng, variant, k_pseudo=5):
    causative_cols = list(range(novel_start, novel_start + n_novel))
    neg_idx  = np.where(y == 0)[0]
    pos_idx  = np.where(y == 1)[0]
    X_neg    = X[neg_idx]
    combos   = load_olida_combos_per_sample(X, y, novel_start, n_novel)
    multi    = [(i, c) for i, c in combos if len(c) >= 2]

    if not multi:
        print("  No multi-SNP combos — empty")
        return X, y

    multi_case_idx = [i for i, _ in multi]
    X_cases = X[multi_case_idx]
    y_cases = np.ones(len(multi_case_idx), dtype=int)

    # truth pairs (for wrong-pair variant)
    truth_pairs = list(get_truth_pairs(multi))

    pseudo_X = []

    if variant == "k5":
        # K_PSEUDO=k_pseudo pseudo-controls per SNP per case
        for _, combo in multi:
            for snp in combo:
                for _ in range(k_pseudo):
                    bg_i = rng.integers(0, len(X_neg))
                    pc   = X_neg[bg_i].copy()
                    for c in causative_cols:
                        pc[c] = 0
                    pc[snp] = 1
                    pseudo_X.append(pc)

    elif variant == "samebg":
        # Use SAME background as the corresponding positive case
        for case_i, combo in multi:
            bg = X[case_i].copy()
            for c in causative_cols:
                bg[c] = 0          # strip all causative from the positive background
            for snp in combo:
                pc = bg.copy()
                pc[snp] = 1
                pseudo_X.append(pc)

    elif variant == "wrongpair":
        # Standard pseudo-controls + wrong-pair negatives
        # Standard
        for _, combo in multi:
            for snp in combo:
                bg_i = rng.integers(0, len(X_neg))
                pc   = X_neg[bg_i].copy()
                for c in causative_cols:
                    pc[c] = 0
                pc[snp] = 1
                pseudo_X.append(pc)
        # Wrong-pair: pick 2 SNPs from DIFFERENT truth pairs
        all_pair_snps = list(set(s for p in truth_pairs for s in p))
        if len(truth_pairs) >= 2:
            n_wrong = len(multi_case_idx)   # match number of positives
            for _ in range(n_wrong):
                # pick two truth pairs at random, take one SNP from each
                p1, p2 = rng.choice(len(truth_pairs), 2, replace=False)
                snp1 = rng.choice(sorted(truth_pairs[p1]))
                snp2 = rng.choice(sorted(truth_pairs[p2]))
                if snp1 == snp2:
                    continue
                bg_i = rng.integers(0, len(X_neg))
                pc   = X_neg[bg_i].copy()
                for c in causative_cols:
                    pc[c] = 0
                pc[snp1] = 1
                pc[snp2] = 1
                pseudo_X.append(pc)
        else:
            print("  Not enough truth pairs for wrong-pair negatives")

    else:
        raise ValueError(f"Unknown variant: {variant}")

    X_pseudo = np.array(pseudo_X, dtype=np.int8)
    y_pseudo = np.zeros(len(X_pseudo), dtype=int)

    print(f"  Variant={variant}: pos={len(X_cases)}, neg={len(X_pseudo)}")

    X_out = np.concatenate([X_cases, X_pseudo], axis=0)
    y_out = np.concatenate([y_cases, y_pseudo])
    perm  = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def main():
    short   = sys.argv[1] if len(sys.argv) > 1 else "longqt"
    variant = sys.argv[2] if len(sys.argv) > 2 else "k5"

    src_dir = f"dataset/{short}"
    dst_dir = f"dataset/{short}_mb_{variant}"
    os.makedirs(dst_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    print(f"\nBuilding margbal variant '{variant}': {short} → {dst_dir}")

    X_tr, y_tr, var_ids, sids_tr, novel_start, n_novel = load_split(src_dir, "train")
    X_va, y_va, _,       sids_va, _,            _      = load_split(src_dir, "val")
    X_te, y_te, _,       sids_te, _,            _      = load_split(src_dir, "test")

    print(f"p={X_tr.shape[1]}, n_novel={n_novel}")

    for split_name, X, y, sids in [("train", X_tr, y_tr, sids_tr),
                                    ("val",   X_va, y_va, sids_va),
                                    ("test",  X_te, y_te, sids_te)]:
        print(f"Processing {split_name}...")
        X_new, y_new = build_variant(X, y, novel_start, n_novel, rng, variant)
        print(f"  → shape={X_new.shape}, pos={y_new.sum()}, neg={(y_new==0).sum()}")

        if split_name == "train":
            caus = list(range(novel_start, novel_start + n_novel))
            pm, nm = y_new==1, y_new==0
            diff = np.abs(X_new[pm][:,caus].astype(float).mean(0) -
                          X_new[nm][:,caus].astype(float).mean(0))
            print(f"  Marginal balance: max Δ = {diff.max():.3f}")

        new_sids = np.array([f"MBV_{i:06d}" for i in range(len(y_new))])
        np.savez_compressed(
            os.path.join(dst_dir, f"genotype_{split_name}.npz"),
            X=X_new, y=y_new,
            variant_ids=var_ids,
            sample_ids=new_sids,
            novel_start=np.array(novel_start),
            n_novel=np.array(n_novel)
        )
    print(f"Saved → {dst_dir}/")


if __name__ == "__main__":
    main()
