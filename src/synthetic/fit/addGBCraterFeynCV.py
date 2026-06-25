#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Incremental add-on: fit HistGradientBoostingRegressor on the same 5-fold CV
splits as the existing crater-CV bundles (gaussian + sigmoid), and append
'GB Test', 'GB Test_std', 'GB Test_folds' to each setting's pickle.

Reloads the landscape data from DATASETS/craterModel/dist_{gaussian,sigmoid}Crater.pickle
and rebuilds the KFold splits with the same seed.

Usage: python addGBCraterFeynCV.py {gaussian|sigmoid}
"""
import os, pickle, sys, time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, ndcg_score
from scipy.stats import pearsonr, spearmanr

N_FOLDS = 5
CV_SEED = 42


def getScores(Y, Yp, name):
    try:    r  = pearsonr(Y, Yp)[0]
    except: r  = np.nan
    try:    r2 = r2_score(Y, Yp)
    except: r2 = np.nan
    try:    sp = spearmanr(Y, Yp)[0]
    except: sp = np.nan
    try:
        s1 = MinMaxScaler(); Ys  = s1.fit_transform(Y.reshape(-1, 1)).T
        s2 = MinMaxScaler(); Yps = s2.fit_transform(Yp.reshape(-1, 1)).T
        ndcg = ndcg_score(Ys, Yps)
    except:
        ndcg = np.nan
    return (name,
            float(r)    if r    == r    else np.nan,
            float(r2)   if r2   == r2   else np.nan,
            float(ndcg) if ndcg == ndcg else np.nan,
            float(sp)   if sp   == sp   else np.nan)


def aggregate(name, fold_tuples):
    arr = np.array([[t[1], t[2], t[3], t[4]] for t in fold_tuples], dtype=float)
    with np.errstate(all="ignore"):
        m = np.nanmean(arr, axis=0); s = np.nanstd(arr, axis=0)
    return ((name,          float(m[0]), float(m[1]), float(m[2]), float(m[3])),
            (name + "_std", float(s[0]), float(s[1]), float(s[2]), float(s[3])))


def fit_gb(ds):
    SNPDATA    = ds[5]
    phenotypes = ds[4]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    folds = []
    for tr, te in kf.split(SNPDATA):
        Xraw, xraw = SNPDATA[tr], SNPDATA[te]
        Y, y       = phenotypes[tr], phenotypes[te]
        scaler = StandardScaler()
        X = scaler.fit_transform(Xraw)
        x = scaler.transform(xraw)
        gb = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05,
            max_leaf_nodes=31, random_state=CV_SEED)
        gb.fit(X, Y)
        yp = gb.predict(x)
        folds.append(getScores(y, yp, "GB Test"))
    return folds


def main(argv):
    if len(argv) < 2 or argv[1] not in ("gaussian", "sigmoid"):
        print("usage: addGBCraterFeynCV.py {gaussian|sigmoid}")
        return 1
    func = argv[1]
    if func == "gaussian":
        OUT = "results/run_CraterGaussCV/"
        DATASETS_PATH = "data/synthetic/craterModel/dist_gaussianCrater.pickle"
    else:
        OUT = "results/run_CraterSigmoidCV/"
        DATASETS_PATH = "data/synthetic/craterModel/dist_sigmoidCrater.pickle"

    DATASETS = pickle.load(open(DATASETS_PATH, "rb"))
    by_tag = {}
    for ds in DATASETS:
        s_samples, _, q_qtl = ds[0], ds[1], ds[2]
        by_tag[f"S{s_samples}_Q{q_qtl}"] = ds

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

        if "GB Test" in d and "GB Test_std" in d and "GB Test_folds" in d:
            skipped += 1; continue

        if tag not in by_tag:
            print(f"  no landscape data for {tag}"); failed += 1; continue

        try:
            folds = fit_gb(by_tag[tag])
        except Exception as e:
            print(f"  GB fit fail {tag}: {e}"); failed += 1; continue

        mean_t, std_t = aggregate("GB Test", folds)
        d["GB Test"]       = mean_t
        d["GB Test_std"]   = std_t
        d["GB Test_folds"] = folds
        pickle.dump(d, open(path, "wb"))
        done += 1
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-6)
        print(f"  [{i+1}/{len(files)}] {tag} GB.r={mean_t[1]:.3f}±{std_t[1]:.3f} "
              f"rate={rate:.2f}/s")

    print(f"\n{func} FINAL: done={done} skipped={skipped} failed={failed}")

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
