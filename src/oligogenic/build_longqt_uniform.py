#!/usr/bin/env python3
"""
LongQT-based semi-synthetic dataset with UNIFORM pair balancing.
Each of the K causative pairs appears in exactly N_PER_PAIR positive cases.

Negatives are margbal-style pseudo-controls (2 per case, one per SNP of the
pair, with all causative columns zeroed except the focal one).

Writes to dataset/longqt_uniform/ (independent of longqt, longqt_margbal,
longqt_mb_*). Truth pairs are the K=9 pairs observed in longqt_margbal.
"""

import os, sys
import numpy as np

SEED         = 42
SRC          = "dataset/longqt_margbal"
DST          = "dataset/longqt_uniform"
DEFAULT_K    = 9
DEFAULT_NPP  = 20
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15

# 9 truth pairs observed in LongQT positives (column indices in margbal feature space)
LONGQT_PAIRS = [
    (17661, 17662),
    (17662, 17664),
    (17659, 17660),
    (17654, 17655),
    (17661, 17663),
    (17661, 17664),
    (17662, 17663),
    (17663, 17664),
    (17657, 17658),
]


def load_all_margbal():
    Xs, ys = [], []
    novel_start = n_novel = None
    var_ids = None
    for sp in ("train", "val", "test"):
        d = np.load(os.path.join(SRC, f"genotype_{sp}.npz"), allow_pickle=True)
        Xs.append(d["X"]); ys.append(d["y"])
        novel_start = int(d["novel_start"]); n_novel = int(d["n_novel"])
        var_ids = d["variant_ids"]
    return (np.concatenate(Xs), np.concatenate(ys),
            novel_start, n_novel, var_ids)


def build_uniform(X_src, y_src, novel_start, n_novel,
                  K, n_per_pair, rng):
    """Generate uniform-pair positives + margbal pseudo-controls."""
    assert K <= len(LONGQT_PAIRS), f"K={K} > available pairs ({len(LONGQT_PAIRS)})"
    pairs           = LONGQT_PAIRS[:K]
    causative_cols  = list(range(novel_start, novel_start + n_novel))
    neg_pool        = X_src[y_src == 0]

    pos_list, pseudo_list = [], []
    for a, b in pairs:
        for _ in range(n_per_pair):
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

    X_pos  = np.array(pos_list,    dtype=np.int8)
    X_neg  = np.array(pseudo_list, dtype=np.int8)
    X_out  = np.concatenate([X_pos, X_neg], axis=0)
    y_out  = np.concatenate([np.ones(len(X_pos), dtype=int),
                             np.zeros(len(X_neg), dtype=int)])
    perm   = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def save_splits(X, y, var_ids, novel_start, n_novel, out_dir, rng_split):
    os.makedirs(out_dir, exist_ok=True)
    n     = len(y)
    idx   = rng_split.permutation(n)
    n_tr  = int(n * TRAIN_FRAC)
    n_va  = int(n * VAL_FRAC)
    sids  = np.array([f"U_{i:06d}" for i in range(n)])
    parts = [("train", idx[:n_tr]),
             ("val",   idx[n_tr:n_tr + n_va]),
             ("test",  idx[n_tr + n_va:])]
    for split, sl in parts:
        np.savez_compressed(
            os.path.join(out_dir, f"genotype_{split}.npz"),
            X=X[sl], y=y[sl],
            variant_ids=var_ids, sample_ids=sids[sl],
            novel_start=np.array(novel_start),
            n_novel=np.array(n_novel))
    print(f"  → wrote {n_tr} train / {n_va} val / {n - n_tr - n_va} test  to {out_dir}")


def main():
    K   = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_K
    npp = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_NPP

    print(f"Building longqt_uniform with K={K}, n_per_pair={npp}")
    print(f"  source: {SRC}")
    print(f"  truth pairs: {LONGQT_PAIRS[:K]}")

    X, y, novel_start, n_novel, var_ids = load_all_margbal()
    print(f"  source shape: {X.shape}, pos={int(y.sum())}, neg={int((y==0).sum())}")

    rng = np.random.default_rng(SEED)
    X_out, y_out = build_uniform(X, y, novel_start, n_novel, K, npp, rng)
    n_pos = int(y_out.sum()); n_neg = int((y_out == 0).sum())
    print(f"  uniform output: shape={X_out.shape}, pos={n_pos}, neg={n_neg}  "
          f"(per pair: pos={n_pos // K}, neg/pair={n_neg // K})")

    save_splits(X_out, y_out, var_ids, novel_start, n_novel, DST,
                np.random.default_rng(SEED + 1))


if __name__ == "__main__":
    main()
