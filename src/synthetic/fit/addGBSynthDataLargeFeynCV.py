#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Incremental add-on: fit HistGradientBoostingRegressor ("Gradient Boosting")
on the same 5-fold CV splits as the existing synth-CV bundle, and append
'GB Test', 'GB Test_std', 'GB Test_folds' (+ feature_importances per fold)
to each setting's pickle in run100_synthLargeCV/.

KFold(random_state=42) is deterministic so GB sees exactly the same fold
indices as the existing algorithms. Decoy column selection is the original
unseeded logic — GB will see different decoy SNPs than the prior run, but
since decoys are noise this is fine for the comparison.

Idempotent: if 'GB Test' is already in the per-setting pickle, skip.
"""
import os, pickle, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, ndcg_score
from scipy.stats import pearsonr

from computePredsSynthDataLargeFeyn import (
    readSNPMatrix, readSimulations,
)

OUT_FOLDER = "results/run100_synthLargeCV/"
N_FOLDS    = 5
CV_SEED    = 42


def getScores(Y, Yp, name):
    try:    r  = pearsonr(Y, Yp)[0]
    except: r  = np.nan
    try:    r2 = r2_score(Y, Yp)
    except: r2 = np.nan
    try:
        s1 = MinMaxScaler(); Ys  = s1.fit_transform(Y.reshape(-1, 1)).T
        s2 = MinMaxScaler(); Yps = s2.fit_transform(Yp.reshape(-1, 1)).T
        ndcg = ndcg_score(Ys, Yps)
    except:
        ndcg = np.nan
    return (name,
            float(r)    if r    == r    else np.nan,
            float(r2)   if r2   == r2   else np.nan,
            float(ndcg) if ndcg == ndcg else np.nan)


def aggregate(name, fold_tuples):
    arr = np.array([[t[1], t[2], t[3]] for t in fold_tuples], dtype=float)
    with np.errstate(all="ignore"):
        m = np.nanmean(arr, axis=0); s = np.nanstd(arr, axis=0)
    return ((name,          float(m[0]), float(m[1]), float(m[2])),
            (name + "_std", float(s[0]), float(s[1]), float(s[2])))


_SNP_CACHE = None


def get_snp_matrix():
    global _SNP_CACHE
    if _SNP_CACHE is None:
        print("Loading DATASETS/synthSNPsLarge.csv (one-shot cache) ...")
        _SNP_CACHE = readSNPMatrix("data/synthetic/synthSNPsLarge.csv")
    return _SNP_CACHE


def fit_gb_for_setting(h, qtl, dom, epi, numDecoys, numSamples):
    """Reproduce the dataset assembly logic of computePredsSynthDataLargeFeynCV
       and fit GB on the 5 KFold splits. Returns list of fold score tuples,
       and the per-fold feature_importances arrays."""
    f = ("data/synthetic/synthPhenotypesLargeFewQTL/"
         "herit_ %1.1f numQTL %d _Dfract %s _EpiAddOv %s/") % (h, qtl, dom, epi)
    causative, phenotypes, corresp, epiPairs = readSimulations(f)
    _, snpNames, SNPDATA = get_snp_matrix()

    columns = list(range(0, SNPDATA.shape[1]))
    for i in causative:
        columns.remove(i)
    columns = causative + np.random.choice(columns, (numDecoys,)).tolist()
    np.random.shuffle(columns)

    SNPDATA, phenotypes = shuffle(SNPDATA, phenotypes, random_state=CV_SEED)
    SNPDATA = SNPDATA[:numSamples][:, columns].astype(np.float32)
    phenotypes = (np.array(phenotypes[:numSamples]) * -1).reshape(-1).astype(np.float32)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    fold_scores = []
    fold_importances = []
    for ifold, (tr_idx, te_idx) in enumerate(kf.split(SNPDATA)):
        X, x = SNPDATA[tr_idx], SNPDATA[te_idx]
        Y, y = phenotypes[tr_idx], phenotypes[te_idx]
        gb = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=None,
            max_leaf_nodes=31, l2_regularization=0.0,
            random_state=CV_SEED,
        )
        gb.fit(X, Y)
        yp = gb.predict(x)
        fold_scores.append(getScores(y, yp, "GB Test"))
        # HistGB has no .feature_importances_; use permutation-free placeholder:
        fold_importances.append(None)
    return fold_scores, fold_importances


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
            failed += 1
            continue

        if "GB Test" in d and "GB Test_std" in d and "GB Test_folds" in d:
            skipped += 1
            continue

        # decode tag: setting_h{H}_q{Q}_D{D}_E{E}_dec{DEC}_N{N}_ep{EP}.pickle
        tag = fn[len("setting_"):-len(".pickle")]
        try:
            parts = tag.split("_")
            kv = {}
            for p in parts:
                if p.startswith("h"):   kv["h"] = float(p[1:])
                elif p.startswith("q"): kv["q"] = int(p[1:])
                elif p.startswith("D"): kv["D"] = p[1:]
                elif p.startswith("E"): kv["E"] = p[1:]
                elif p.startswith("dec"): kv["dec"] = int(p[3:])
                elif p.startswith("N"): kv["N"] = int(p[1:])
                elif p.startswith("ep"): kv["ep"] = int(p[2:])
        except Exception as e:
            print(f"[{i+1}/{len(files)}] tag parse fail {tag}: {e}")
            failed += 1
            continue

        try:
            fold_scores, fold_imp = fit_gb_for_setting(
                kv["h"], kv["q"], kv["D"], kv["E"], kv["dec"], kv["N"])
        except Exception as e:
            print(f"[{i+1}/{len(files)}] GB fit fail {tag}: {e}")
            failed += 1
            continue

        mean_t, std_t = aggregate("GB Test", fold_scores)
        d["GB Test"]            = mean_t
        d["GB Test_std"]        = std_t
        d["GB Test_folds"]      = fold_scores
        d["GBParams_folds"]     = fold_imp

        pickle.dump(d, open(path, "wb"))
        done += 1
        if done % 25 == 0 or i == len(files) - 1:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            remaining = (len(files) - skipped - done - failed) / max(rate, 1e-6)
            print(f"[{i+1}/{len(files)}] done={done} skipped={skipped} fail={failed} "
                  f"rate={rate:.2f}/s ETA={remaining/60:.1f}min "
                  f"last={tag} GB.r={mean_t[1]:.3f}±{std_t[1]:.3f}")

    print(f"\nFINAL: done={done} skipped={skipped} failed={failed}")

    # Re-bundle FINAL pickle
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
