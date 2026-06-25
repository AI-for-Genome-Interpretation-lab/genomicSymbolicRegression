#!/usr/bin/env python3
"""
Run margbal-uniform sweep saving the Feyn pickle for every cell,
so mixed-partials analysis can be applied later. K=2..9 x npp=5..30 step 5.
Writes:
  - uniform_surface_margbal.csv  (metrics, same schema as before)
  - pickles_margbal_uniform/K{K}_npp{npp}/feyn_model.pickle  (Feyn model)
  - pickles_margbal_uniform/K{K}_npp{npp}/feyn_prefilter.npy (prefilter idx)
"""
import os, pickle, time
import numpy as np
import pandas as pd
import feyn

import run_uniform_surface as R
# Override build with margbal-faithful uniform (reuse the existing patch)
import run_uniform_surface_margbal as MB
R.build_uniform = MB.build_uniform_margbal

R.OUT_CSV    = "uniform_surface_margbal.csv"
R.K_VALS     = [2, 3, 4, 5, 6, 7, 8, 9]
R.NPP_VALS   = [5, 10, 15, 20, 25, 30]
R.RUN_PLINK  = True

PICKLE_ROOT = "pickles_margbal_uniform"


def main():
    X_src, y_src, novel_start, n_novel = R.load_src()
    causative_cols = list(range(novel_start, novel_start + n_novel))
    os.makedirs(PICKLE_ROOT, exist_ok=True)

    rows = []
    done = set()
    if os.path.exists(R.OUT_CSV):
        prev = pd.read_csv(R.OUT_CSV)
        rows = prev.to_dict("records")
        done = {(int(r["K"]), int(r["n_per_pair"])) for r in rows}
        print(f"Resuming: {len(done)} cells already done")

    for K in R.K_VALS:
        for npp in R.NPP_VALS:
            if (K, npp) in done:
                continue
            tag = f"K{K}_npp{npp}"
            t0  = time.time()
            rng = np.random.default_rng(R.SEED + K * 100 + npp)
            X, y, truth_pairs = R.build_uniform(
                X_src, y_src, novel_start, n_novel, K, npp, rng)
            n_pos = int(y.sum())

            X_norm = X.astype(np.float32) / 2.0
            pre    = R.prefilter(X_norm, y, causative_cols, R.MAX_FEYN_FEATURES)
            X_sub  = X_norm[:, pre]
            X_sub_dosage = X.astype(np.int8)[:, pre]

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

            # Save pickle + prefilter for this cell
            cell_dir = os.path.join(PICKLE_ROOT, tag)
            os.makedirs(cell_dir, exist_ok=True)
            with open(os.path.join(cell_dir, "feyn_model.pickle"), "wb") as f:
                pickle.dump(best, f)
            np.save(os.path.join(cell_dir, "feyn_prefilter.npy"), pre)
            np.save(os.path.join(cell_dir, "truth_pairs.npy"),
                    np.array([sorted(p) for p in truth_pairs]))

            mul_pairs, hess_pairs, formula_cols = R.feyn_extract(best, pre)
            combo_pairs = mul_pairs | hess_pairs
            truth_set = set(truth_pairs)
            j_strict = R.jaccard(mul_pairs,   truth_set)
            j_hess   = R.jaccard(hess_pairs,  truth_set)
            j_combo  = R.jaccard(combo_pairs, truth_set)
            detected_loose = {p for p in truth_pairs if p.issubset(formula_cols)}
            j_loose  = R.jaccard(detected_loose, truth_set)

            if R.RUN_PLINK:
                t_plk = time.time()
                detected_plk = R.run_plink(X_sub_dosage, y, pre)
                j_plk = R.jaccard(detected_plk, truth_set)
                t_plk = time.time() - t_plk
            else:
                detected_plk = set()
                j_plk = float("nan")
                t_plk = 0.0

            elapsed = time.time() - t0
            row = dict(
                K=K, n_per_pair=npp, n_pos=n_pos, n_total=len(y),
                feyn_J_strict=j_strict, feyn_J_hess=j_hess,
                feyn_J_combo=j_combo,   feyn_J_loose=j_loose,
                plink_J=j_plk,
                feyn_n_pairs=len(mul_pairs), feyn_n_hess=len(hess_pairs),
                feyn_n_combo=len(combo_pairs), feyn_n_feats=len(formula_cols),
                plink_n=len(detected_plk),
                feyn_s=round(t_feyn,1), plink_s=round(t_plk,1),
                elapsed_s=round(elapsed,1))
            rows.append(row)
            print(f"[{tag:<14}] pos={n_pos:4d}  strict={j_strict:.2f} loose={j_loose:.2f} "
                  f"plink={j_plk:.2f}  feyn={t_feyn:.0f}s  pickle→{cell_dir}")
            pd.DataFrame(rows).to_csv(R.OUT_CSV, index=False)

    print(f"\nSaved → {R.OUT_CSV} and pickles in {PICKLE_ROOT}/")


if __name__ == "__main__":
    if os.path.exists(R.OUT_CSV):
        os.remove(R.OUT_CSV)
        print(f"Removed stale {R.OUT_CSV}")
    main()
