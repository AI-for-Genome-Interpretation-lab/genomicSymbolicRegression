#!/usr/bin/env python3
"""
Margbal-AND uniform sweep: pseudo-controls keep the SAME count of active
causative SNPs as the positive, but replace one of the pair-SNPs with a
random non-pair causative SNP. Breaks linear separability:
  pos:    v_a=v_b=1 (+ others from real combo)
  pseudo: v_a=1 + v_c=1  (or v_b=1 + v_d=1)   — c, d random non-pair causative

Sum(causative)=count is identical in pos and pseudo → only v_a·v_b discriminates.
"""
import os, pickle, time
import numpy as np
import pandas as pd
import feyn

import run_uniform_surface as R

R.OUT_CSV    = "uniform_surface_margbal_AND.csv"
R.K_VALS     = [2, 3, 4, 5, 6, 7, 8, 9]
R.NPP_VALS   = [5, 10, 15, 20, 25, 30]
R.RUN_PLINK  = False  # skip to save time

PICKLE_ROOT = "pickles_margbal_uniform_AND"


def build_uniform_margbal_AND(X_src, y_src, novel_start, n_novel, K, npp, rng):
    pairs           = R.LONGQT_PAIRS[:K]
    causative_cols  = list(range(novel_start, novel_start + n_novel))
    pos_pool = X_src[y_src == 1]
    neg_pool = X_src[y_src == 0]

    pos_list, pseudo_list = [], []
    for a, b in pairs:
        avail = np.where((pos_pool[:, a] > 0) & (pos_pool[:, b] > 0))[0]
        if len(avail) == 0:
            # synthetic fallback for K>9
            for _ in range(npp):
                bg = neg_pool[rng.integers(0, len(neg_pool))].copy()
                for c in causative_cols: bg[c] = 0
                bg[a] = 1; bg[b] = 1
                pos_list.append(bg)
                for snp in (a, b):
                    pc = neg_pool[rng.integers(0, len(neg_pool))].copy()
                    for c in causative_cols: pc[c] = 0
                    pc[snp] = 1
                    # replace the partner with a random non-pair causative
                    others = [c for c in causative_cols if c not in (a, b)]
                    decoy = others[rng.integers(0, len(others))]
                    pc[decoy] = 1
                    pseudo_list.append(pc)
            continue

        picks = rng.choice(avail, size=npp, replace=True)
        for pi in picks:
            pos_instance = pos_pool[pi].copy()
            pos_list.append(pos_instance)
            active = [c for c in causative_cols if pos_instance[c] > 0]
            for snp in active:
                pc = neg_pool[rng.integers(0, len(neg_pool))].copy()
                for c in causative_cols: pc[c] = 0
                pc[snp] = 1
                # add a decoy causative SNP to match the count of active causatives in pos
                # (pos has len(active), pseudo would have 1 → add len(active)-1 decoys)
                n_decoys = len(active) - 1
                others = [c for c in causative_cols if c != snp]
                if n_decoys > 0:
                    decoys = rng.choice(others, size=n_decoys, replace=False)
                    for dc in decoys:
                        pc[dc] = 1
                pseudo_list.append(pc)

    X_out = np.concatenate(
        [np.array(pos_list, dtype=np.int8),
         np.array(pseudo_list, dtype=np.int8)], axis=0)
    y_out = np.concatenate(
        [np.ones(len(pos_list), dtype=int),
         np.zeros(len(pseudo_list), dtype=int)])
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm], [frozenset(p) for p in pairs]


R.build_uniform = build_uniform_margbal_AND


def main():
    X_src, y_src, novel_start, n_novel = R.load_src()
    causative_cols = list(range(novel_start, novel_start + n_novel))
    os.makedirs(PICKLE_ROOT, exist_ok=True)

    rows = []
    for K in R.K_VALS:
        for npp in R.NPP_VALS:
            tag = f"K{K}_npp{npp}"
            t0  = time.time()
            rng = np.random.default_rng(R.SEED + K * 100 + npp)
            X, y, truth_pairs = R.build_uniform(
                X_src, y_src, novel_start, n_novel, K, npp, rng)
            n_pos = int(y.sum())

            X_norm = X.astype(np.float32) / 2.0
            pre    = R.prefilter(X_norm, y, causative_cols, R.MAX_FEYN_FEATURES)
            X_sub  = X_norm[:, pre]

            t_feyn = time.time()
            cols = [f"v{i}" for i in range(X_sub.shape[1])]
            df = pd.DataFrame(X_sub, columns=cols); df["y"] = y
            ql = feyn.QLattice(random_seed=42)
            models = ql.auto_run(
                df, output_name="y", kind="classification",
                n_epochs=R.N_EPOCHS, criterion="bic",
                max_complexity=R.MAX_COMPLEXITY,
                function_names=["add", "multiply"])
            best = models[0]
            t_feyn = time.time() - t_feyn

            cell_dir = os.path.join(PICKLE_ROOT, tag)
            os.makedirs(cell_dir, exist_ok=True)
            with open(os.path.join(cell_dir, "feyn_model.pickle"), "wb") as f:
                pickle.dump(best, f)
            np.save(os.path.join(cell_dir, "feyn_prefilter.npy"), pre)
            np.save(os.path.join(cell_dir, "truth_pairs.npy"),
                    np.array([sorted(p) for p in truth_pairs]))

            mul_pairs, hess_pairs, formula_cols = R.feyn_extract(best, pre)
            truth_set = set(truth_pairs)
            j_strict = R.jaccard(mul_pairs, truth_set)
            j_loose  = R.jaccard({p for p in truth_pairs if p.issubset(formula_cols)}, truth_set)

            elapsed = time.time() - t0
            try: expr = best.sympify(signif=3)
            except: expr = best.sympify()
            row = dict(
                K=K, n_per_pair=npp, n_pos=n_pos, n_total=len(y),
                feyn_J_strict=j_strict, feyn_J_loose=j_loose,
                feyn_n_pairs=len(mul_pairs), feyn_n_feats=len(formula_cols),
                feyn_s=round(t_feyn,1), elapsed_s=round(elapsed,1))
            rows.append(row)
            print(f"[{tag:<14}] pos={n_pos:4d}  strict={j_strict:.2f} loose={j_loose:.2f}  "
                  f"formula={str(expr)[:100]}")
            pd.DataFrame(rows).to_csv(R.OUT_CSV, index=False)

    print(f"\nSaved → {R.OUT_CSV} and pickles in {PICKLE_ROOT}/")


if __name__ == "__main__":
    if os.path.exists(R.OUT_CSV):
        os.remove(R.OUT_CSV)
    main()
