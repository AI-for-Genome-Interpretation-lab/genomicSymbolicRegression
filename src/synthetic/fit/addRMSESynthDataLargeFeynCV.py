#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Incremental add-on: compute per-fold RMSE for every algorithm in the synth CV
bundles, derived from stored R² and the test-fold variance — no model re-fit.

For each model key 'X Test' present in a setting pickle, appends:
  'X Test_RMSE'         → ("X Test_RMSE", rmse_mean)
  'X Test_RMSE_std'     → ("X Test_RMSE_std", rmse_std)
  'X Test_RMSE_folds'   → list of per-fold rmse floats

Phenotypes are derived using the original logic of
computePredsSynthDataLargeFeynCV.mainExecuteCV (row shuffle seed=42, take first
numSamples). KFold(seed=42) gives the same train/test indices as the original
run, so fold variances are exact.

The SimPhe phenotype list per (h, qtl, dom, epi) is cached in memory.
"""
import os, pickle, sys, time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from computePredsSynthDataLargeFeyn import readSimulations

OUT_FOLDER = "results/run100_synthLargeCV/"
N_FOLDS    = 5
CV_SEED    = 42
R2_IDX     = 2     # tuple layout: (name, pearson, r2, ndcg)

ALGO_KEYS = ["Ridge Test", "Lasso Test", "MLP Test", "RF Test",
             "FeynBIC Test", "GB Test", "PLINK2 Test"]


_PHEN_CACHE = {}


def get_phenotypes(h, qtl, dom, epi):
    """Cached SimPhe phenotype list for the given setting."""
    k = (h, qtl, dom, epi)
    if k not in _PHEN_CACHE:
        f = ("data/synthetic/synthPhenotypesLargeFewQTL/"
             "herit_ %1.1f numQTL %d _Dfract %s _EpiAddOv %s/") % (h, qtl, dom, epi)
        _, phenotypes, _, _ = readSimulations(f)
        _PHEN_CACHE[k] = phenotypes
    return _PHEN_CACHE[k]


def fold_variances(phenotypes, numSamples):
    """Reproduce row-shuffle, subsampling, and KFold splits of the CV run,
    then return per-fold variance of y_test."""
    n_full = len(phenotypes)
    idx = np.arange(n_full)
    # mirror shuffle(SNPDATA, phenotypes, random_state=CV_SEED) — same seed
    rng = np.random.RandomState(CV_SEED)
    perm = rng.permutation(n_full)
    idx = idx[perm]
    idx = idx[:numSamples]
    phen = np.asarray(phenotypes)[idx] * -1
    phen = phen.astype(np.float32).reshape(-1)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    out = []
    for _, te in kf.split(np.zeros(len(phen))):
        out.append(float(np.var(phen[te])))
    return out


def rmse_for_algo(fold_tuples, fold_vars):
    rmse_folds = []
    for ft, vy in zip(fold_tuples, fold_vars):
        try:
            r2 = ft[R2_IDX]
        except (IndexError, TypeError):
            r2 = None
        if r2 is None or (isinstance(r2, float) and np.isnan(r2)):
            rmse_folds.append(float("nan"))
        else:
            mse = max((1.0 - float(r2)) * vy, 0.0)
            rmse_folds.append(float(np.sqrt(mse)))
    return rmse_folds


def parse_tag(tag):
    """setting_h{H}_q{Q}_D{D}_E{E}_dec{DEC}_N{N}_ep{EP} → dict."""
    parts = tag.split("_")
    kv = {}
    for p in parts:
        if p.startswith("h"):     kv["h"]   = float(p[1:])
        elif p.startswith("q"):   kv["q"]   = int(p[1:])
        elif p.startswith("D"):   kv["D"]   = p[1:]
        elif p.startswith("E"):   kv["E"]   = p[1:]
        elif p.startswith("dec"): kv["dec"] = int(p[3:])
        elif p.startswith("N"):   kv["N"]   = int(p[1:])
        elif p.startswith("ep"):  kv["ep"]  = int(p[2:])
    return kv


def main():
    files = sorted(fn for fn in os.listdir(OUT_FOLDER)
                   if fn.startswith("setting_") and fn.endswith(".pickle"))
    print(f"Found {len(files)} setting pickles in {OUT_FOLDER}")

    t0 = time.time()
    done = skipped = failed = 0
    for i, fn in enumerate(files):
        path = os.path.join(OUT_FOLDER, fn)
        try:
            d = pickle.load(open(path, "rb"))
        except Exception as e:
            print(f"[{i+1}/{len(files)}] load fail {fn}: {e}")
            failed += 1; continue

        present_algos = [k for k in ALGO_KEYS if (k + "_folds") in d]
        if not present_algos:
            skipped += 1
            continue

        tag = fn[len("setting_"):-len(".pickle")]
        try:
            kv = parse_tag(tag)
        except Exception as e:
            print(f"[{i+1}/{len(files)}] tag parse fail {tag}: {e}")
            failed += 1; continue

        try:
            phen = get_phenotypes(kv["h"], kv["q"], kv["D"], kv["E"])
            fvars = fold_variances(phen, kv["N"])
        except Exception as e:
            print(f"[{i+1}/{len(files)}] phen/var fail {tag}: {e}")
            failed += 1; continue

        # Clean any prior 'X Test RMSE' (space) format
        for old in [k for k in list(d.keys()) if " RMSE" in k]:
            del d[old]

        for algo_key in present_algos:
            fold_tuples = d[algo_key + "_folds"]
            rmse_folds = rmse_for_algo(fold_tuples, fvars)
            with np.errstate(all="ignore"):
                arr = np.asarray(rmse_folds, dtype=float)
                m = float(np.nanmean(arr))
                s = float(np.nanstd(arr))
            d[algo_key + "_RMSE"]       = (algo_key + "_RMSE", m)
            d[algo_key + "_RMSE_std"]   = (algo_key + "_RMSE_std", s)
            d[algo_key + "_RMSE_folds"] = rmse_folds

        pickle.dump(d, open(path, "wb"))
        done += 1
        if done % 200 == 0 or i == len(files) - 1:
            print(f"[{i+1}/{len(files)}] done={done} skipped={skipped} fail={failed} "
                  f"elapsed={time.time()-t0:.0f}s")

    print(f"\nFINAL: done={done} skipped={skipped} failed={failed}  "
          f"({time.time()-t0:.1f}s)")

    # Re-bundle
    bundle = {}
    for fn in sorted(os.listdir(OUT_FOLDER)):
        if fn.startswith("setting_") and fn.endswith(".pickle"):
            key = fn[len("setting_"):-len(".pickle")]
            try:
                bundle[key] = pickle.load(open(os.path.join(OUT_FOLDER, fn), "rb"))
            except Exception as e:
                print("bundle load fail", fn, e)
    out = os.path.join(OUT_FOLDER, "run100CV_FINAL.pickle")
    pickle.dump(bundle, open(out, "wb"))
    print(f"Re-bundled: {out} (settings: {len(bundle)})")


if __name__ == "__main__":
    sys.exit(main() or 0)
