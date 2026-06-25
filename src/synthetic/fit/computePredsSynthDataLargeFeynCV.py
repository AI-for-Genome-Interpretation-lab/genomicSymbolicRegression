#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5-fold CV variant of computePredsSynthDataLargeFeyn.py.

For each (heritability, qtl, dom, epi, decoys, samples, epochs) setting:
- subsample as in the original (decoys + samples)
- run KFold(n_splits=5, shuffle=True, random_state=42)
- inside each fold: fit Ridge, Lasso, MLP, RF, Feyn(+, *), PLINK2 (optional)
- store per-fold scores + mean + std under each model key
- write one pickle per setting in run100_synthLargeCV/, resumable by skipping
  settings whose pickle already exists.
"""
import numpy as np
import os, pickle, math, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, ndcg_score
from scipy.stats import pearsonr
import pandas as pd
import feyn

# Reuse parsers from the original script
from computePredsSynthDataLargeFeyn import (
    readSNPMatrix, readCausativeSNPs, readPhenotypes,
    readEpistaticPairsSNPs, readSimulations, nameColumns,
)

# PLINK2 disabled: this CV run targets predictive performance only, not Jaccard.
HAVE_PLINK = False


N_FOLDS = 5
CV_SEED = 42
OUT_FOLDER = "results/run100_synthLargeCV/"
PERC_TRAIN = 0.7   # only used to size MLP iters; not for split


def getScores(Y, Yp, name):
    try:
        r = pearsonr(Y, Yp)[0]
    except Exception:
        r = np.nan
    try:
        r2 = r2_score(Y, Yp)
    except Exception:
        r2 = np.nan
    try:
        s1 = MinMaxScaler(); Ys  = s1.fit_transform(Y.reshape(-1, 1)).T
        s2 = MinMaxScaler(); Yps = s2.fit_transform(Yp.reshape(-1, 1)).T
        ndcg = ndcg_score(Ys, Yps)
    except Exception:
        ndcg = np.nan
    return (name, float(r) if r == r else np.nan,
                  float(r2) if r2 == r2 else np.nan,
                  float(ndcg) if ndcg == ndcg else np.nan)


def aggregate_folds(name, fold_tuples):
    arr = np.array([[t[1], t[2], t[3]] for t in fold_tuples], dtype=float)
    with np.errstate(all="ignore"):
        mean = np.nanmean(arr, axis=0)
        std  = np.nanstd (arr, axis=0)
    return ((name, float(mean[0]), float(mean[1]), float(mean[2])),
            (name + "_std", float(std[0]), float(std[1]), float(std[2])))


def mainExecuteCV(h, DOM, EPI, numDecoys, numSamples, EPOCHS, qtl):
    """One setting → 5 fold CV → aggregated results dict."""
    f = "data/synthetic/synthPhenotypesLargeFewQTL/herit_ %1.1f numQTL %d _Dfract %s _EpiAddOv %s/" \
        % (h, qtl, DOM, EPI)
    print(f"\n===== SETTING h={h} qtl={qtl} D={DOM} E={EPI} dec={numDecoys} N={numSamples} ep={EPOCHS} =====")
    print(f)

    causative, phenotypes, corresp, epiPairs = readSimulations(f)
    _, snpNames, SNPDATA = readSNPMatrix("data/synthetic/synthSNPsLarge.csv")

    # Decoy + column selection (same as original)
    columns = list(range(0, SNPDATA.shape[1]))
    for i in causative:
        columns.remove(i)
    columns = causative + np.random.choice(columns, (numDecoys)).tolist()
    np.random.shuffle(columns)

    # Subsample rows BEFORE CV split, matching original behaviour
    SNPDATA, phenotypes = shuffle(SNPDATA, phenotypes, random_state=CV_SEED)
    SNPDATA = SNPDATA[:numSamples][:, columns].astype(np.float32)
    phenotypes = np.array(phenotypes[:numSamples]) * -1
    phenotypes = phenotypes.reshape(-1).astype(np.float32)

    results = {
        "simpheCausative": causative,
        "epiPairs":        epiPairs,
        "n_folds":         N_FOLDS,
        "cv_seed":         CV_SEED,
    }
    cpos = ["SNP_" + str(columns.index(c)) for c in causative]
    results["CAUSATIVE"] = cpos

    fold_acc = {k: [] for k in
                ["Ridge Test", "Lasso Test", "MLP Test", "RF Test", "FeynBIC Test", "PLINK2 Test"]}
    extra_per_fold = {k: [] for k in
                      ["ridgeParams", "LassoParams", "RFParams", "FEATURES",
                       "best_query", "PLINK2EpiPairs", "PLINK2Features"]}

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    for ifold, (tr_idx, te_idx) in enumerate(kf.split(SNPDATA)):
        print(f"  -- fold {ifold + 1}/{N_FOLDS} (train={len(tr_idx)} test={len(te_idx)})")
        X, x = SNPDATA[tr_idx], SNPDATA[te_idx]
        Y, y = phenotypes[tr_idx], phenotypes[te_idx]

        # Ridge
        m = Ridge(); m.fit(X, Y)
        fold_acc["Ridge Test"].append(getScores(y, m.predict(x), "Ridge Test"))
        extra_per_fold["ridgeParams"].append(m.coef_.copy())

        # Lasso
        m = Lasso(max_iter=2000); m.fit(X, Y)
        fold_acc["Lasso Test"].append(getScores(y, m.predict(x), "Lasso Test"))
        extra_per_fold["LassoParams"].append(m.coef_.copy())

        # MLP
        m = MLPRegressor(activation="tanh", learning_rate="adaptive",
                         max_iter=400, learning_rate_init=1e-2)
        m.fit(X, Y)
        fold_acc["MLP Test"].append(getScores(y, m.predict(x), "MLP Test"))

        # RF
        m = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        m.fit(X, Y)
        fold_acc["RF Test"].append(getScores(y, m.predict(x), "RF Test"))
        extra_per_fold["RFParams"].append(m.feature_importances_.copy())

        # Feyn (+, *)
        TRAIN = pd.DataFrame(X); TRAIN.columns = nameColumns(TRAIN.columns); TRAIN["label"] = Y
        TEST  = pd.DataFrame(x); TEST.columns  = nameColumns(TEST.columns ); TEST ["label"] = y
        ql = feyn.QLattice()
        models = ql.auto_run(TRAIN, output_name="label", kind="regression",
                             n_epochs=EPOCHS, max_complexity=99,
                             function_names=["add", "multiply"], threads=16)
        best = models[0]
        fold_acc["FeynBIC Test"].append(getScores(y, best.predict(TEST), "FeynBIC Test"))
        extra_per_fold["FEATURES"].append(list(best.features))
        extra_per_fold["best_query"].append(best.to_query_string())

        # PLINK2 (optional)
        if HAVE_PLINK:
            try:
                yp_plink, plink_pairs = plink2EpistasisPredict(
                    X, Y, x, n_top_pairs=max(1, len(epiPairs)))
                fold_acc["PLINK2 Test"].append(getScores(y, yp_plink, "PLINK2 Test"))
                pairs_orig = []
                for pair in plink_pairs:
                    n1 = "SNP_" + str(columns[pair['snp1']] + 1)
                    n2 = "SNP_" + str(columns[pair['snp2']] + 1)
                    pairs_orig.append(tuple(sorted([n1, n2])))
                extra_per_fold["PLINK2EpiPairs"].append(pairs_orig)
                detected = set()
                for pair in plink_pairs:
                    detected.add("SNP_%d" % pair['snp1'])
                    detected.add("SNP_%d" % pair['snp2'])
                extra_per_fold["PLINK2Features"].append(list(detected))
            except Exception as e:
                print("    PLINK2 error:", e)
                fold_acc["PLINK2 Test"].append(("PLINK2 Test", np.nan, np.nan, np.nan))
                extra_per_fold["PLINK2EpiPairs"].append([])
                extra_per_fold["PLINK2Features"].append([])
        else:
            fold_acc["PLINK2 Test"].append(("PLINK2 Test", np.nan, np.nan, np.nan))
            extra_per_fold["PLINK2EpiPairs"].append([])
            extra_per_fold["PLINK2Features"].append([])

    # Aggregate
    for key, folds in fold_acc.items():
        mean_t, std_t = aggregate_folds(key, folds)
        results[key]            = mean_t
        results[key + "_std"]   = std_t
        results[key + "_folds"] = folds
    for key, vals in extra_per_fold.items():
        results[key + "_folds"] = vals
    return results


def main(argv):
    os.makedirs(OUT_FOLDER, exist_ok=True)

    heritability = [0.6, 0.3]
    qtl          = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 26, 30, 34, 38, 40, 46, 50]
    epi          = ["0", "1"]
    dom          = ["0", "1"]
    numDecoys    = [0, 100, 500, 1000, 1900]
    numSamples   = [500, 1000, 2000]
    EPOCHS_LIST  = [10]

    totIter = (len(heritability) * len(qtl) * len(epi) * len(dom)
               * len(numDecoys) * len(numSamples) * len(EPOCHS_LIST))
    s = 0
    for ep in EPOCHS_LIST:
        for h in heritability:
            for c in qtl:
                for E in epi:
                    for D in dom:
                        for d in numDecoys:
                            for n in numSamples:
                                s += 1
                                tag = f"h{h}_q{c}_D{D}_E{E}_dec{d}_N{n}_ep{ep}"
                                out_pickle = os.path.join(OUT_FOLDER, f"setting_{tag}.pickle")
                                if os.path.exists(out_pickle):
                                    print(f"[skip] {s}/{totIter} {tag} already done")
                                    continue
                                print(f"\n#### {s}/{totIter} ({100*s/totIter:.2f}%) {tag}")
                                try:
                                    res = mainExecuteCV(h, D, E, d, n, ep, c)
                                    pickle.dump(res, open(out_pickle, "wb"))
                                except Exception as exc:
                                    print(f"  !! FAIL on {tag}: {exc}")
                                    with open(out_pickle + ".err", "w") as fp:
                                        fp.write(str(exc))

    # Bundle into a single FINAL pickle for plotting compatibility
    bundle = {}
    for fn in sorted(os.listdir(OUT_FOLDER)):
        if fn.startswith("setting_") and fn.endswith(".pickle"):
            key = fn[len("setting_"):-len(".pickle")]
            try:
                bundle[key] = pickle.load(open(os.path.join(OUT_FOLDER, fn), "rb"))
            except Exception as e:
                print("bundle load fail", fn, e)
    pickle.dump(bundle, open(os.path.join(OUT_FOLDER, "run100CV_FINAL.pickle"), "wb"))
    print(f"\nDone. Bundle: {OUT_FOLDER}run100CV_FINAL.pickle (settings: {len(bundle)})")


if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)
