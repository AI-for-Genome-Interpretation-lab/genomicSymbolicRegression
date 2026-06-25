#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5-fold CV variant of computePredsSCraterFeyn.py for Fig 5 (crater + mesa).

For each (samples, qtl) in the loaded crater/mesa pickle:
- 5-fold CV on the (X, y) of that dataset
- per fold: Ridge, Lasso, MLP, RF, Feyn(+, *), Feyn(+, *, Gauss/exp)
- store per-fold scores + mean + std under each model key
- per-setting pickle in run_Crater{Gauss,Sigmoid}CV/, resumable

Usage: python computePredsSCraterFeynCV.py {gaussian|sigmoid}
"""
import numpy as np
import os, pickle, sys
import matplotlib
matplotlib.use("Agg")
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, ndcg_score
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import feyn

from computePredsSCraterFeyn import nameColumns, sortPairs

N_FOLDS = 5
CV_SEED = 42
EPOCHS  = 25


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
    return ((name,           float(m[0]), float(m[1]), float(m[2]), float(m[3])),
            (name + "_std",  float(s[0]), float(s[1]), float(s[2]), float(s[3])))


def runCV(s_samples, q_qtl, ds, func):
    """One (samples, qtl) → 5 fold CV → aggregated dict."""
    SNPDATA    = ds[5]
    phenotypes = ds[4]
    genoDist   = ds[3]
    numVars    = SNPDATA.shape[1]

    print(f"\n===== {func} samples={s_samples} qtl={q_qtl} shape={SNPDATA.shape} =====")

    results = {
        "n_folds": N_FOLDS, "cv_seed": CV_SEED,
        "samples": s_samples, "qtl": q_qtl, "func": func,
    }
    cpos = ["SNP_" + str(c) for c in range(numVars)]
    results["CAUSATIVE"] = cpos

    fold_acc = {k: [] for k in
                ["Ridge Test", "Lasso Test", "MLP Test", "RF Test",
                 "FeynBIC Test", "FeynGauss Test"]}
    extra = {k: [] for k in
             ["ridgeParams", "LassoParams", "RFParams",
              "FEATURESFeynBic", "FEATURESFeynGauss",
              "bestFeynBicQuery", "bestFeynGaussQuery",
              "feynBicLinePlot", "feynGaussLinePlot",
              "labelLinePlot",
              "ridgeLinePlot", "LassoLinePlot", "MLPLinePlot", "RFLinePlot"]}

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    for ifold, (tr_idx, te_idx) in enumerate(kf.split(SNPDATA)):
        print(f"  -- fold {ifold + 1}/{N_FOLDS} (train={len(tr_idx)} test={len(te_idx)})")
        Xraw, xraw = SNPDATA[tr_idx], SNPDATA[te_idx]
        Y, y       = phenotypes[tr_idx], phenotypes[te_idx]
        genoTest   = genoDist[te_idx]

        scaler = StandardScaler()
        X = scaler.fit_transform(Xraw)
        x = scaler.transform(xraw)

        # Ridge
        m = Ridge(); m.fit(X, Y)
        ypR = m.predict(x)
        fold_acc["Ridge Test"].append(getScores(y, ypR, "Ridge Test"))
        extra["ridgeParams"].append(m.coef_.copy())
        extra["ridgeLinePlot"].append(sortPairs(genoTest, ypR))

        # Lasso
        m = Lasso(max_iter=2000); m.fit(X, Y)
        ypL = m.predict(x)
        fold_acc["Lasso Test"].append(getScores(y, ypL, "Lasso Test"))
        extra["LassoParams"].append(m.coef_.copy())
        extra["LassoLinePlot"].append(sortPairs(genoTest, ypL))

        # MLP
        m = MLPRegressor(activation="tanh", learning_rate="adaptive",
                         max_iter=400, learning_rate_init=1e-2)
        m.fit(X, Y)
        ypM = m.predict(x)
        fold_acc["MLP Test"].append(getScores(y, ypM, "MLP Test"))
        extra["MLPLinePlot"].append(sortPairs(genoTest, ypM))

        # RF
        m = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        m.fit(X, Y)
        ypF = m.predict(x)
        fold_acc["RF Test"].append(getScores(y, ypF, "RF Test"))
        extra["RFParams"].append(m.feature_importances_.copy())
        extra["RFLinePlot"].append(sortPairs(genoTest, ypF))

        # Feyn (+, *)
        TRAIN = pd.DataFrame(X); TRAIN.columns = nameColumns(TRAIN.columns); TRAIN["label"] = Y
        TEST  = pd.DataFrame(x); TEST .columns = nameColumns(TEST .columns); TEST ["label"] = y
        ql = feyn.QLattice()
        models = ql.auto_run(TRAIN, output_name="label", kind="regression",
                             n_epochs=EPOCHS, max_complexity=2 * numVars,
                             function_names=["add", "multiply"],
                             threads=16, criterion="bic")
        best = models[0]
        ypFB = best.predict(TEST)
        fold_acc["FeynBIC Test"].append(getScores(y, ypFB, "FeynBIC Test"))
        extra["FEATURESFeynBic"].append(list(best.features))
        extra["bestFeynBicQuery"].append(best.to_query_string())
        extra["feynBicLinePlot"].append(sortPairs(genoTest, ypFB))

        # Feyn (+, *, gaussian|exp)
        ql = feyn.QLattice()
        if func == "gaussian":
            fnames = ["add", "multiply", "gaussian"]
        elif func == "sigmoid":
            fnames = ["add", "multiply", "exp"]
        else:
            raise Exception("unknown func: " + str(func))
        models = ql.auto_run(TRAIN, output_name="label", kind="regression",
                             n_epochs=EPOCHS, max_complexity=2 * numVars,
                             function_names=fnames,
                             threads=16, criterion="bic")
        best = models[0]
        ypFG = best.predict(TEST)
        fold_acc["FeynGauss Test"].append(getScores(y, ypFG, "FeynGauss Test"))
        extra["FEATURESFeynGauss"].append(list(best.features))
        extra["bestFeynGaussQuery"].append(best.to_query_string())
        extra["feynGaussLinePlot"].append(sortPairs(genoTest, ypFG))

        extra["labelLinePlot"].append(sortPairs(genoTest, y))

    # Aggregate
    for key, folds in fold_acc.items():
        mean_t, std_t = aggregate(key, folds)
        results[key]            = mean_t
        results[key + "_std"]   = std_t
        results[key + "_folds"] = folds
    for key, vals in extra.items():
        results[key + "_folds"] = vals
    return results


def main(argv):
    if len(argv) < 2 or argv[1] not in ("gaussian", "sigmoid"):
        print("usage: computePredsSCraterFeynCV.py {gaussian|sigmoid}")
        return 1
    func = argv[1]
    if func == "gaussian":
        OUT = "results/run_CraterGaussCV/"
        DATASETS = pickle.load(open("data/synthetic/craterModel/dist_gaussianCrater.pickle", "rb"))
    else:
        OUT = "results/run_CraterSigmoidCV/"
        DATASETS = pickle.load(open("data/synthetic/craterModel/dist_sigmoidCrater.pickle", "rb"))
    os.makedirs(OUT, exist_ok=True)

    tot = len(DATASETS); a = 0
    for ds in DATASETS:
        a += 1
        s_samples, _, q_qtl = ds[0], ds[1], ds[2]
        tag = f"S{s_samples}_Q{q_qtl}"
        out_pickle = os.path.join(OUT, f"setting_{tag}.pickle")
        if os.path.exists(out_pickle):
            print(f"[skip] {a}/{tot} {tag} already done"); continue
        print(f"\n#### {a}/{tot} ({100*a/tot:.2f}%) {func} {tag}")
        try:
            res = runCV(s_samples, q_qtl, ds, func)
            pickle.dump(res, open(out_pickle, "wb"))
        except Exception as exc:
            print(f"  !! FAIL {tag}: {exc}")
            with open(out_pickle + ".err", "w") as fp:
                fp.write(str(exc))

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
    print(f"\nDone. Bundle: {OUT}{final} (settings: {len(bundle)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)
