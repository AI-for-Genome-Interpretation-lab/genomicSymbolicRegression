#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Benchmark run times for each algorithm vs numSamples and numDecoys.
# Saves results to benchmark_times.pickle and generates plots.
#
import numpy as np
import pickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fit"))
from plinkEpistasis import plink2EpistasisPredict
import feyn
import pandas as pd

# All output PNGs go here (override with FIGOUT). Never the repo root.
OUTDIR = os.environ.get("FIGOUT", "figures/_build")
os.makedirs(OUTDIR, exist_ok=True)

PERC_TRAIN = 0.7
N_SNPS_TOTAL = 2000
N_CAUSATIVE = 4        # fixed: 4 QTL, 2 epistatic pairs
N_EPI_PAIRS = 2
N_REPEATS = 3          # repeat each timing N times, take median
FEYN_EPOCHS = 10       # same as main codebase

def nameColumns(cols):
    return ["SNP_" + str(i) for i in cols]

def makeSyntheticData(numSamples, numDecoys):
    """Generate random SNP data and phenotype."""
    n = numSamples
    p_total = N_CAUSATIVE + numDecoys
    X = np.random.randint(0, 3, size=(n, p_total)).astype(np.float32)
    # simple additive phenotype from first N_CAUSATIVE columns
    coef = np.random.randn(N_CAUSATIVE)
    Y = X[:, :N_CAUSATIVE] @ coef + 0.3 * np.random.randn(n)
    Y = Y.astype(np.float32)
    X, Y = shuffle(X, Y)
    return X, Y

def timeAlgo(fn, *args, **kwargs):
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))

def benchmarkSamples():
    """Fix numDecoys=100, vary numSamples."""
    numDecoys = 100
    sample_values = [1000, 2500, 5000, 10000, 20000]
    results = {algo: [] for algo in ["Ridge", "Lasso", "MLP", "RF", "PLINK2", "Feyn"]}

    for n in sample_values:
        print("numSamples=%d" % n)
        X, Y = makeSyntheticData(n, numDecoys)
        train = int(n * PERC_TRAIN)
        Xtrain, Ytrain = X[:train], Y[:train]
        Xtest, Ytest = X[train:], Y[train:]

        # Ridge
        t = timeAlgo(lambda: Ridge().fit(Xtrain, Ytrain).predict(Xtest))
        results["Ridge"].append(t)
        print("  Ridge: %.3fs" % t)

        # Lasso
        t = timeAlgo(lambda: Lasso(max_iter=2000).fit(Xtrain, Ytrain).predict(Xtest))
        results["Lasso"].append(t)
        print("  Lasso: %.3fs" % t)

        # MLP
        t = timeAlgo(lambda: MLPRegressor(activation="tanh", learning_rate="adaptive",
                                           max_iter=400, learning_rate_init=1e-2).fit(Xtrain, Ytrain).predict(Xtest))
        results["MLP"].append(t)
        print("  MLP: %.3fs" % t)

        # RF
        t = timeAlgo(lambda: RandomForestRegressor(n_estimators=100, n_jobs=-1).fit(Xtrain, Ytrain).predict(Xtest))
        results["RF"].append(t)
        print("  RF: %.3fs" % t)

        # PLINK2
        t = timeAlgo(lambda: plink2EpistasisPredict(Xtrain, Ytrain, Xtest, n_top_pairs=N_EPI_PAIRS))
        results["PLINK2"].append(t)
        print("  PLINK2: %.3fs" % t)

        # Feyn
        ql = feyn.QLattice()
        TRAIN = pd.DataFrame(Xtrain)
        TRAIN.columns = nameColumns(TRAIN.columns)
        TRAIN["label"] = Ytrain
        TEST = pd.DataFrame(Xtest)
        TEST.columns = nameColumns(TEST.columns)
        TEST["label"] = Ytest
        def run_feyn():
            models = ql.auto_run(TRAIN, output_name="label", kind="regression",
                                  n_epochs=FEYN_EPOCHS, max_complexity=99,
                                  function_names=["add", "multiply"], threads=16)
            models[0].predict(TEST)
        t = timeAlgo(run_feyn)
        results["Feyn"].append(t)
        print("  Feyn: %.3fs" % t)

    return sample_values, results

def benchmarkDecoys():
    """Fix numSamples=1000, vary numDecoys."""
    numSamples = 1000
    decoy_values = [0, 500, 1000, 2500, 5000, 10000, 19000]
    results = {algo: [] for algo in ["Ridge", "Lasso", "MLP", "RF", "PLINK2", "Feyn"]}

    for d in decoy_values:
        print("numDecoys=%d" % d)
        X, Y = makeSyntheticData(numSamples, d)
        train = int(numSamples * PERC_TRAIN)
        Xtrain, Ytrain = X[:train], Y[:train]
        Xtest, Ytest = X[train:], Y[train:]

        t = timeAlgo(lambda: Ridge().fit(Xtrain, Ytrain).predict(Xtest))
        results["Ridge"].append(t)
        print("  Ridge: %.3fs" % t)

        t = timeAlgo(lambda: Lasso(max_iter=2000).fit(Xtrain, Ytrain).predict(Xtest))
        results["Lasso"].append(t)
        print("  Lasso: %.3fs" % t)

        t = timeAlgo(lambda: MLPRegressor(activation="tanh", learning_rate="adaptive",
                                           max_iter=400, learning_rate_init=1e-2).fit(Xtrain, Ytrain).predict(Xtest))
        results["MLP"].append(t)
        print("  MLP: %.3fs" % t)

        t = timeAlgo(lambda: RandomForestRegressor(n_estimators=100, n_jobs=-1).fit(Xtrain, Ytrain).predict(Xtest))
        results["RF"].append(t)
        print("  RF: %.3fs" % t)

        t = timeAlgo(lambda: plink2EpistasisPredict(Xtrain, Ytrain, Xtest, n_top_pairs=N_EPI_PAIRS))
        results["PLINK2"].append(t)
        print("  PLINK2: %.3fs" % t)

        ql = feyn.QLattice()
        TRAIN = pd.DataFrame(Xtrain)
        TRAIN.columns = nameColumns(TRAIN.columns)
        TRAIN["label"] = Ytrain
        TEST = pd.DataFrame(Xtest)
        TEST.columns = nameColumns(TEST.columns)
        TEST["label"] = Ytest
        def run_feyn():
            models = ql.auto_run(TRAIN, output_name="label", kind="regression",
                                  n_epochs=FEYN_EPOCHS, max_complexity=99,
                                  function_names=["add", "multiply"], threads=16)
            models[0].predict(TEST)
        t = timeAlgo(run_feyn)
        results["Feyn"].append(t)
        print("  Feyn: %.3fs" % t)

    return decoy_values, results

def plot_results(sample_values, results_samples, decoy_values, results_decoys):
    colors = {
        "Ridge":  "#2196F3",
        "Lasso":  "#4CAF50",
        "MLP":    "#FF9800",
        "RF":     "#9C27B0",
        "PLINK2": "#F44336",
        "Feyn":   "#009688",
    }
    markers = {
        "Ridge": "o", "Lasso": "s", "MLP": "^",
        "RF": "D", "PLINK2": "X", "Feyn": "*"
    }

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    fig.suptitle("Algorithm runtime comparison", fontsize=13)

    # --- samples plot ---
    ax = axs[0]
    for algo, times in results_samples.items():
        ax.plot(sample_values, times, label=algo, color=colors[algo],
                marker=markers[algo], linewidth=2, markersize=7)
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Time (seconds, median of %d runs)" % N_REPEATS)
    ax.set_title("Runtime vs Samples\n(decoys=100, QTL=4)")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- decoys plot ---
    ax = axs[1]
    for algo, times in results_decoys.items():
        ax.plot(decoy_values, times, label=algo, color=colors[algo],
                marker=markers[algo], linewidth=2, markersize=7)
    ax.set_xlabel("Number of decoy SNPs")
    ax.set_ylabel("Time (seconds, median of %d runs)" % N_REPEATS)
    ax.set_title("Runtime vs Decoy SNPs\n(samples=1000, QTL=4)")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.savefig(os.path.join(OUTDIR, "benchmark_times.png"), dpi=300)
    print("Saved: benchmark_times.png")
    plt.show()


def main():
    print("=== Benchmarking vs numSamples ===")
    sample_values, results_samples = benchmarkSamples()

    print("\n=== Benchmarking vs numDecoys ===")
    decoy_values, results_decoys = benchmarkDecoys()

    data = {
        "sample_values": sample_values,
        "results_samples": results_samples,
        "decoy_values": decoy_values,
        "results_decoys": results_decoys,
    }
    pickle.dump(data, open("results/elife/benchmark_times.pickle", "wb"))
    print("Saved: benchmark_times.pickle")

    plot_results(sample_values, results_samples, decoy_values, results_decoys)


if __name__ == "__main__":
    main()
