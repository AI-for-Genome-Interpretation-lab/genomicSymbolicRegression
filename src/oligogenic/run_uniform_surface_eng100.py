#!/usr/bin/env python3
"""
Small-grid wrapper with PRE-ENGINEERED INTERACTIONS over TOP-100 SNPs by |corr|
(no privileged knowledge of causative cols).

For each cell:
  1. Build the synthetic as usual.
  2. Compute |corr(X, y)| → take top-100 SNPs.
  3. Build C(100,2) = 4 950 interaction columns v_a*v_b over those 100.
  4. Feed Feyn: just the 100 singletons + 4 950 engineered = 5 050 features.
  5. Pair detected = engineered feature present in Feyn formula.
K=2..9 x npp=5..30 step 5 (48 cells). Writes to uniform_surface_eng100.csv.
"""
import os, itertools, time
import numpy as np
import pandas as pd

import run_uniform_surface as R

R.OUT_CSV    = "uniform_surface_eng100.csv"
R.K_VALS     = [2, 3, 4, 5, 6, 7, 8, 9]
R.NPP_VALS   = [5, 10, 15, 20, 25, 30]
R.RUN_PLINK  = True

TOP_N = 100


def main_engineered():
    X_src, y_src, novel_start, n_novel = R.load_src()

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

            # Top-N by |corr|
            X_norm = X.astype(np.float32) / 2.0
            corr = np.array([
                abs(np.corrcoef(X_norm[:, i], y)[0, 1]) if X_norm[:, i].std() > 0 else 0.0
                for i in range(X_norm.shape[1])
            ])
            top_idx = np.argsort(corr)[::-1][:TOP_N]
            top_idx = np.array(sorted(top_idx.tolist()))   # ascending for stable order

            # Build engineered pairs over the top-N
            pairs_idx = list(itertools.combinations(range(TOP_N), 2))    # 4 950 pairs (positions in top_idx)
            X_eng = np.zeros((X.shape[0], len(pairs_idx)), dtype=np.int8)
            for k, (i, j) in enumerate(pairs_idx):
                a, b = int(top_idx[i]), int(top_idx[j])
                X_eng[:, k] = (X[:, a] > 0) & (X[:, b] > 0)

            # Feed Feyn the 100 singletons + 4 950 engineered = 5 050 features
            X_singletons = X_norm[:, top_idx]
            X_full = np.concatenate([X_singletons, X_eng.astype(np.float32)], axis=1)

            # Map prefilter-style index back to original space:
            # positions 0..99  → original top_idx[i]
            # positions 100..  → engineered pair (top_idx[i], top_idx[j])
            pre_map_orig_singleton = list(map(int, top_idx))
            pre_map_eng_pairs      = [(int(top_idx[i]), int(top_idx[j])) for i, j in pairs_idx]

            # Run Feyn directly on the 5 050 features
            t_feyn = time.time()
            cols = [f"v{k}" for k in range(X_full.shape[1])]
            df = pd.DataFrame(X_full, columns=cols); df["y"] = y
            import feyn
            ql = feyn.QLattice(random_seed=42)
            models = ql.auto_run(
                df, output_name="y", kind="classification",
                n_epochs=R.N_EPOCHS, criterion="bic",
                max_complexity=R.MAX_COMPLEXITY,
                function_names=["add", "multiply"])
            best = models[0]
            t_feyn = time.time() - t_feyn

            # Which formula features did Feyn pick?
            formula_idx = set()
            for f in best.features:
                if f.startswith("v") and f[1:].isdigit():
                    formula_idx.add(int(f[1:]))

            # Engineered-pair detection
            n_top = TOP_N
            eng_detected = set()
            for k, (a, b) in enumerate(pre_map_eng_pairs):
                if (n_top + k) in formula_idx:
                    eng_detected.add(frozenset([a, b]))
            singletons_in_formula = {pre_map_orig_singleton[k] for k in formula_idx if k < n_top}

            # Coverage check: how many truth pairs even have BOTH SNPs in top-100?
            top_set = set(top_idx.tolist())
            truth_coverable = {p for p in truth_pairs if p.issubset(top_set)}

            truth_set = set(truth_pairs)
            j_eng     = R.jaccard(eng_detected, truth_set)
            j_eng_cov = R.jaccard(eng_detected, truth_coverable) if truth_coverable else float("nan")
            # loose on the singletons used by Feyn (still bound to top-N space)
            detected_loose = {p for p in truth_pairs if p.issubset(singletons_in_formula)}
            j_loose  = R.jaccard(detected_loose, truth_set)

            # PLINK on the same top-N singletons (no engineered cols)
            X_sub_dosage_orig = X.astype(np.int8)[:, top_idx]
            if R.RUN_PLINK:
                t_plk = time.time()
                detected_plk = R.run_plink(X_sub_dosage_orig, y, top_idx)
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
                feyn_J_eng_cov=j_eng_cov,
                truth_coverable=len(truth_coverable),
                feyn_J_loose=j_loose,
                plink_J=j_plk,
                feyn_n_eng=len(eng_detected),
                feyn_n_singletons=len(singletons_in_formula),
                plink_n=len(detected_plk),
                feyn_s=round(t_feyn,1), plink_s=round(t_plk,1),
                elapsed_s=round(elapsed, 1))
            rows.append(row)
            print(f"[{tag:<12}]  pos={n_pos:4d}  "
                  f"eng={j_eng:.2f}(n={len(eng_detected)}) "
                  f"eng|cov={j_eng_cov:.2f}({len(truth_coverable)}/{len(truth_pairs)}) "
                  f"loose={j_loose:.2f} PLINK={j_plk:.2f} "
                  f"feyn={t_feyn:.0f}s plk={t_plk:.0f}s tot={elapsed:.0f}s")

            pd.DataFrame(rows).to_csv(R.OUT_CSV, index=False)

    print(f"\nSaved → {R.OUT_CSV}")


if __name__ == "__main__":
    if os.path.exists(R.OUT_CSV):
        os.remove(R.OUT_CSV)
        print(f"Removed stale {R.OUT_CSV}")
    main_engineered()
