#!/usr/bin/env python3
"""
Small-grid wrapper with AIC criterion (instead of BIC) to push Feyn toward
multiplicative formulas. K=2..9 x npp=5..30 step 5 (48 cells).
Writes to uniform_surface_aic.csv.
"""
import os
import run_uniform_surface as R

R.OUT_CSV    = "uniform_surface_aic.csv"
R.K_VALS     = [2, 3, 4, 5, 6, 7, 8, 9]
R.NPP_VALS   = [5, 10, 15, 20, 25, 30]
R.RUN_PLINK  = True

# Monkey-patch run_feyn to use AIC instead of BIC
import feyn, pandas as pd

def run_feyn_aic(X_sub, y, prefilter_idx):
    cols = [f"v{i}" for i in range(X_sub.shape[1])]
    df = pd.DataFrame(X_sub, columns=cols); df["y"] = y
    ql = feyn.QLattice(random_seed=42)
    models = ql.auto_run(
        df, output_name="y", kind="classification",
        n_epochs=R.N_EPOCHS, criterion="aic",
        max_complexity=R.MAX_COMPLEXITY,
        function_names=["add", "multiply"])
    best = models[0]
    return R.feyn_extract(best, prefilter_idx)

R.run_feyn = run_feyn_aic


if __name__ == "__main__":
    if os.path.exists(R.OUT_CSV):
        os.remove(R.OUT_CSV)
        print(f"Removed stale {R.OUT_CSV}")
    R.main()
