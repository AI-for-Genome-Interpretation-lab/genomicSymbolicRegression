#!/usr/bin/env python3
"""
Small-grid wrapper for run_uniform_surface: K=2..9 x npp=5..30 step 5 (48 cells),
with the new combo (mul ∪ hessian) Feyn metric. PLINK enabled (small grid).
Writes to uniform_surface.csv.
"""
import os, sys

# Force small-grid params before importing the runner.
import run_uniform_surface as R
R.OUT_CSV    = "uniform_surface.csv"
R.K_VALS     = [2, 3, 4, 5, 6, 7, 8, 9]
R.NPP_VALS   = [5, 10, 15, 20, 25, 30]
R.RUN_PLINK  = True

if __name__ == "__main__":
    if os.path.exists(R.OUT_CSV):
        bak = R.OUT_CSV + ".pre_combo.bak"
        if not os.path.exists(bak):
            os.replace(R.OUT_CSV, bak)
            print(f"Backed up old {R.OUT_CSV} → {bak}")
        else:
            os.remove(R.OUT_CSV)
            print(f"Removed stale {R.OUT_CSV} (backup {bak} already exists)")
    R.main()
