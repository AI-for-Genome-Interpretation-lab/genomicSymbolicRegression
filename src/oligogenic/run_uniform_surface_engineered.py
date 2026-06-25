#!/usr/bin/env python3
"""
Small-grid wrapper with PRE-ENGINEERED INTERACTION features:
for every pair (a,b) of the 20 causative columns we append a new column
v_ab = v_a * v_b. Feyn can then pick it additively → pair detection by
"engineered feature present in formula" (no need for Feyn to learn multiplication).

K=2..9 x npp=5..30 step 5 (48 cells).
Writes to uniform_surface_eng.csv.
"""
import os, itertools, time
import numpy as np
import pandas as pd

import run_uniform_surface as R

R.OUT_CSV    = "uniform_surface_eng.csv"
R.K_VALS     = [2, 3, 4, 5, 6, 7, 8, 9]
R.NPP_VALS   = [5, 10, 15, 20, 25, 30]
R.RUN_PLINK  = True

# All C(20,2) = 190 interaction features over the 20 causative cols.
_CAUS_COLS = list(range(17654, 17654 + 20))
INTERACTION_PAIRS = list(itertools.combinations(_CAUS_COLS, 2))   # length 190


def main_engineered():
    X_src, y_src, novel_start, n_novel = R.load_src()
    causative_cols = list(range(novel_start, novel_start + n_novel))
    n_orig = X_src.shape[1]

    rows = []
    done = set()
    if os.path.exists(R.OUT_CSV):
        prev = pd.read_csv(R.OUT_CSV)
        rows = prev.to_dict("records")
        done = {(int(r["K"]), int(r["n_per_pair"])) for r in rows}
        print(f"Resuming: {len(done)} cells already in {R.OUT_CSV}")

    for K in R.K_VALS:
        for npp in R.NPP_VALS:
            if (K, npp) in done:
                continue
            tag = f"K={K} npp={npp}"
            t0  = time.time()
            rng = np.random.default_rng(R.SEED + K * 100 + npp)

            X, y, truth_pairs = R.build_uniform(
                X_src, y_src, novel_start, n_novel, K, npp, rng)
            n_pos = int(y.sum())

            # Append engineered interactions: 190 cols at indices n_orig..n_orig+189
            X_eng = np.zeros((X.shape[0], len(INTERACTION_PAIRS)), dtype=np.int8)
            for k, (a, b) in enumerate(INTERACTION_PAIRS):
                X_eng[:, k] = (X[:, a] > 0) & (X[:, b] > 0)
            X_full = np.concatenate([X, X_eng], axis=1)

            X_norm = X_full.astype(np.float32) / 2.0
            # Force-include all causative singletons AND all 190 engineered pairs
            extra_cols = list(range(n_orig, n_orig + len(INTERACTION_PAIRS)))
            pre = R.prefilter(X_norm, y,
                              causative_cols + extra_cols,
                              R.MAX_FEYN_FEATURES)
            X_sub  = X_norm[:, pre]
            X_sub_dosage = X_full.astype(np.int8)[:, pre]

            # Feyn
            t_feyn = time.time()
            mul_pairs, hess_pairs, formula_cols = R.run_feyn(X_sub, y, pre)
            t_feyn = time.time() - t_feyn

            # Engineered-pair detection: which interaction cols are in formula?
            eng_detected = set()
            for k, (a, b) in enumerate(INTERACTION_PAIRS):
                if (n_orig + k) in formula_cols:
                    eng_detected.add(frozenset([a, b]))

            # Also map mul/hess back to original SNP space (filter out engineered ones)
            def _to_orig_only(pairs):
                out = set()
                for p in pairs:
                    if all(c < n_orig for c in p):
                        out.add(p)
                return out
            mul_orig  = _to_orig_only(mul_pairs)
            hess_orig = _to_orig_only(hess_pairs)

            truth_set = set(truth_pairs)
            j_eng    = R.jaccard(eng_detected, truth_set)
            j_strict = R.jaccard(mul_orig,     truth_set)
            j_hess   = R.jaccard(hess_orig,    truth_set)
            j_combo  = R.jaccard(mul_orig | hess_orig | eng_detected, truth_set)
            detected_loose = {p for p in truth_pairs
                              if p.issubset({c for c in formula_cols if c < n_orig})}
            j_loose  = R.jaccard(detected_loose, truth_set)

            # PLINK (on the original feature subset only — exclude engineered cols)
            orig_mask = np.array([int(pre[i]) < n_orig for i in range(len(pre))])
            X_sub_dosage_orig = X_sub_dosage[:, orig_mask]
            pre_orig = pre[orig_mask]
            if R.RUN_PLINK:
                t_plk = time.time()
                detected_plk = R.run_plink(X_sub_dosage_orig, y, pre_orig)
                j_plk = R.jaccard(detected_plk, truth_set)
                t_plk = time.time() - t_plk
            else:
                detected_plk = set()
                j_plk = float("nan")
                t_plk = 0.0

            elapsed = time.time() - t0
            row = dict(
                K=K, n_per_pair=npp, n_pos=n_pos, n_total=len(y),
                feyn_J_eng=j_eng,
                feyn_J_strict=j_strict,
                feyn_J_hess=j_hess,
                feyn_J_combo=j_combo,
                feyn_J_loose=j_loose,
                plink_J=j_plk,
                feyn_n_eng=len(eng_detected),
                feyn_n_pairs=len(mul_orig),
                feyn_n_hess=len(hess_orig),
                feyn_n_feats=len([c for c in formula_cols if c < n_orig]),
                plink_n=len(detected_plk),
                feyn_s=round(t_feyn,1), plink_s=round(t_plk,1),
                elapsed_s=round(elapsed, 1))
            rows.append(row)
            print(f"[{tag:<12}]  pos={n_pos:4d}  "
                  f"eng={j_eng:.2f}(n={len(eng_detected)}) "
                  f"strict={j_strict:.2f} loose={j_loose:.2f} "
                  f"PLINK={j_plk:.2f} "
                  f"feyn={t_feyn:.0f}s plk={t_plk:.0f}s tot={elapsed:.0f}s")

            pd.DataFrame(rows).to_csv(R.OUT_CSV, index=False)

    print(f"\nSaved → {R.OUT_CSV}")


if __name__ == "__main__":
    if os.path.exists(R.OUT_CSV):
        os.remove(R.OUT_CSV)
        print(f"Removed stale {R.OUT_CSV}")
    main_engineered()
