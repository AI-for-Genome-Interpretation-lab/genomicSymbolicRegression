#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Incremental add-on: compute per-fold RMSE for every algorithm in the crater
CV bundles, derived from the stored R² and the test-fold variance — no model
re-fitting required.

For each model key (e.g. 'Feyn Test'), appends:
  '<key>_RMSE'         → ("<key>_RMSE", rmse_mean)
  '<key>_RMSE_std'     → ("<key>_RMSE_std", rmse_std)
  '<key>_RMSE_folds'   → list of per-fold rmse floats

Usage: python addRMSECraterFeynCV.py {gaussian|sigmoid}
"""
import os, pickle, sys, time
import numpy as np
from sklearn.model_selection import KFold

N_FOLDS = 5
CV_SEED = 42
R2_IDX  = 2     # index of R² in score tuples (works for both synth + crater)


ALGO_KEYS = [
    "Ridge Test", "Lasso Test", "MLP Test", "RF Test",
    "FeynBIC Test", "FeynGauss Test", "GB Test",
]


def fold_variances(phenotypes):
    """Return list of per-fold population variance of y_test."""
    n = len(phenotypes)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    out = []
    for _, te in kf.split(np.zeros(n)):
        y_test = phenotypes[te]
        out.append(float(np.var(y_test)))
    return out


def rmse_for_algo(fold_tuples, fold_vars):
    """Compute per-fold RMSE from stored R² and fold variances."""
    rmse_folds = []
    for ft, vy in zip(fold_tuples, fold_vars):
        r2 = ft[R2_IDX]
        if r2 is None or (isinstance(r2, float) and np.isnan(r2)):
            rmse_folds.append(float("nan"))
        else:
            mse = max((1.0 - float(r2)) * vy, 0.0)
            rmse_folds.append(float(np.sqrt(mse)))
    return rmse_folds


def main(argv):
    if len(argv) < 2 or argv[1] not in ("gaussian", "sigmoid"):
        print("usage: addRMSECraterFeynCV.py {gaussian|sigmoid}")
        return 1
    func = argv[1]
    OUT = "results/run_CraterGaussCV/" if func == "gaussian" else "results/run_CraterSigmoidCV/"
    DATASETS_PATH = (f"data/synthetic/craterModel/"
                     f"dist_{'gaussianCrater' if func=='gaussian' else 'sigmoidCrater'}.pickle")

    DATASETS = pickle.load(open(DATASETS_PATH, "rb"))
    by_tag = {f"S{ds[0]}_Q{ds[2]}": ds for ds in DATASETS}

    files = sorted(fn for fn in os.listdir(OUT)
                   if fn.startswith("setting_") and fn.endswith(".pickle"))
    print(f"{func}: {len(files)} setting pickles, {len(by_tag)} landscape datasets")

    t0 = time.time()
    done = skipped = failed = 0
    for i, fn in enumerate(files):
        tag = fn[len("setting_"):-len(".pickle")]
        path = os.path.join(OUT, fn)
        try:
            d = pickle.load(open(path, "rb"))
        except Exception as e:
            print(f"  load fail {fn}: {e}"); failed += 1; continue

        if tag not in by_tag:
            print(f"  no landscape data for {tag}"); failed += 1; continue

        # Always recompute so an earlier (deprecated) format is replaced.
        ds = by_tag[tag]
        phenotypes = np.asarray(ds[4])
        fvars = fold_variances(phenotypes)

        # Clean any prior 'X Test RMSE' (space) format
        for old in [k for k in list(d.keys()) if " RMSE" in k]:
            del d[old]

        for algo_key in ALGO_KEYS:
            fold_key = algo_key + "_folds"
            if fold_key not in d:
                continue
            fold_tuples = d[fold_key]
            rmse_folds = rmse_for_algo(fold_tuples, fvars)
            with np.errstate(all="ignore"):
                rmse_arr = np.asarray(rmse_folds, dtype=float)
                m = float(np.nanmean(rmse_arr))
                s = float(np.nanstd(rmse_arr))
            d[algo_key + "_RMSE"]       = (algo_key + "_RMSE", m)
            d[algo_key + "_RMSE_std"]   = (algo_key + "_RMSE_std", s)
            d[algo_key + "_RMSE_folds"] = rmse_folds

        pickle.dump(d, open(path, "wb"))
        done += 1

    print(f"\n{func} FINAL: done={done} skipped={skipped} failed={failed}  "
          f"({time.time()-t0:.1f}s)")

    # Re-bundle
    bundle = {}
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith("setting_") and fn.endswith(".pickle"):
            key = fn[len("setting_"):-len(".pickle")]
            try:
                bundle[key] = pickle.load(open(os.path.join(OUT, fn), "rb"))
            except Exception as e:
                print("bundle load fail", fn, e)
    final = "runCrater" + ("Gauss" if func == "gaussian" else "Sigmoid") + "CV_FINAL.pickle"
    pickle.dump(bundle, open(os.path.join(OUT, final), "wb"))
    print(f"Re-bundled: {OUT}{final} (settings: {len(bundle)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)
