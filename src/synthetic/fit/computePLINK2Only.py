#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Compute PLINK2 epistasis predictions on the synthetic large dataset.
# Saves results compatible with the main pickle (run100FINAL100_withRF.pickle).
# Run with: python computePLINK2Only.py
#
import numpy as np
import os, pickle
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, ndcg_score
from plinkEpistasis import plink2EpistasisPredict


def readSNPMatrix(f, header=True):
    row, cols, data = [], [], []
    ifp = open(f, "rb")
    lines = ifp.readlines()
    ifp.close()
    if header:
        cols = lines.pop(0).decode('utf8').strip().split(",")[1:]
    for l in lines:
        tmp = l.decode('utf8').strip().split(",")
        if header:
            row.append(tmp[0])
            data.append(np.array(tmp[1:], dtype=np.int8))
        else:
            data.append(np.array(tmp[:], dtype=np.int8))
    data = np.array(data).T
    print(len(row), len(cols), data.shape)
    return row, cols, data


def readCausativeSNPs(f):
    ifp = open(f, "r")
    l = ifp.readlines()
    rnames, num = [], []
    for i in l:
        rnames.append(i.strip())
        num.append(int(i.strip()[4:]) - 1)
    return rnames, num


def readPhenotypes(f):
    ifp = open(f, "r")
    l = ifp.readlines()
    l.pop(0)
    phen, corresp = [], []
    for i, tmp in enumerate(l):
        p, c = tmp.strip().replace("sample_", "").split(",")
        assert int(c) == i + 1
        phen.append(float(p))
        corresp.append(c)
    return phen, corresp


def readEpistaticPairsSNPs(f):
    ifp = open(f, "r")
    l = ifp.readlines()
    epiPairs = []
    i = 0
    while i < len(l):
        if "[P1epistasis]" in l[i]:
            break
        i += 1
    i += 1
    while i < len(l):
        if "[P1heritability]" in l[i]:
            break
        tmp = l[i].strip().split(" ")
        epiPairs.append((tmp[0], tmp[1]))
        i += 1
    return epiPairs


def readSimulations(f):
    causativeNames, causative = readCausativeSNPs(f + "SNPs_causative.list.txt")
    phenotypes, corresp = readPhenotypes(f + "simulated_phenotype.csv")
    epiPairs = readEpistaticPairsSNPs(f + "usedpars.txt")
    assert len(phenotypes) == len(corresp)
    return causative, phenotypes, corresp, epiPairs


def getScores(Y, Yp, modelName):
    r = pearsonr(Y, Yp)[0]
    r2 = r2_score(Y, Yp)
    print(modelName + " r: ", r)
    print(modelName + " R²: ", r2)
    s = MinMaxScaler()
    Y = s.fit_transform(Y.reshape(-1, 1)).T
    s = MinMaxScaler()
    Yp = s.fit_transform(Yp.reshape(-1, 1)).T
    ndcg = ndcg_score(Y, Yp)
    print(modelName + " NDCG: ", ndcg)
    return modelName, r, r2, ndcg


def mainExecute(args):
    results = {}
    heritability, DOM, EPI, numDecoys, numSamples, PERC_TRAIN, EPOCHS, qtl = args
    TRAIN_SAMPLES = int(numSamples * PERC_TRAIN)
    print("RUN ARGS *************************************************************")
    print(args, TRAIN_SAMPLES)
    f = "data/synthetic/synthPhenotypesLargeFewQTL/herit_ %1.1f numQTL %d _Dfract %s _EpiAddOv %s/" % (heritability, qtl, DOM, EPI)
    print(f)

    causative, phenotypes, corresp, epiPairs = readSimulations(f)
    _, snpNames, SNPDATA = readSNPMatrix("data/synthetic/synthSNPsLarge.csv")
    results["simpheCausative"] = causative
    results["epiPairs"] = epiPairs
    columns = list(range(0, SNPDATA.shape[1]))
    for i in causative:
        columns.remove(i)
    columns = causative + np.random.choice(columns, (numDecoys)).tolist()
    np.random.shuffle(columns)
    SNPDATA, phenotypes = shuffle(SNPDATA, phenotypes)
    SNPDATA = SNPDATA[:numSamples]
    SNPDATA = SNPDATA[:, columns]
    SNPDATA = SNPDATA.astype(np.float32)
    phenotypes = phenotypes[:numSamples]
    phenotypes = (np.array(phenotypes) * -1).reshape(-1)

    X = SNPDATA[:TRAIN_SAMPLES, :]
    Y = phenotypes[:TRAIN_SAMPLES]
    x = SNPDATA[TRAIN_SAMPLES:, :]
    y = phenotypes[TRAIN_SAMPLES:]

    cpos = []
    for c in causative:
        cpos.append("SNP_" + str(columns.index(c)))
    results["CAUSATIVE"] = cpos

    try:
        yp_plink, plink_pairs = plink2EpistasisPredict(X, Y, x, n_top_pairs=max(1, len(epiPairs)))
        results["PLINK2 Test"] = getScores(y, yp_plink, "PLINK2 Test")
        results["PLINK2Pairs"] = plink_pairs
        # Detected SNPs for Jaccard (column-index names, matching CAUSATIVE format)
        detected_snps = set()
        for pair in plink_pairs:
            detected_snps.add("SNP_%d" % pair['snp1'])
            detected_snps.add("SNP_%d" % pair['snp2'])
        results["PLINK2Features"] = list(detected_snps)
        # Detected pairs in original SNP name space (1-indexed), for Fig 3 Jaccard
        epi_pairs_orig = []
        for pair in plink_pairs:
            name1 = "SNP_" + str(columns[pair['snp1']] + 1)
            name2 = "SNP_" + str(columns[pair['snp2']] + 1)
            epi_pairs_orig.append(tuple(sorted([name1, name2])))
        results["PLINK2EpiPairs"] = epi_pairs_orig
    except Exception as e:
        print("PLINK2 error:", e)
        results["PLINK2 Test"] = ("PLINK2 Test", np.nan, np.nan, np.nan)
        results["PLINK2Pairs"] = []
        results["PLINK2Features"] = []
        results["PLINK2EpiPairs"] = []

    return results


def main():
    FOLDER = "results/run100_synthLarge/"
    RUN_NAME = "run100_PLINK2only"
    os.makedirs(FOLDER, exist_ok=True)

    heritability = [0.6, 0.3]
    qtl = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 26, 30, 34, 38, 40, 46, 50]
    epi = ["0", "1"]
    dom = ["0", "1"]
    numDecoys = [0, 100, 500, 1000, 1900]
    numSamples = [500, 1000, 2000]
    EPOCHS = [10]
    PERC_TRAIN = 0.7

    totIter = len(heritability) * len(qtl) * len(epi) * len(dom) * len(numDecoys) * len(numSamples) * len(EPOCHS)
    CONTINUE = -1
    if CONTINUE > 0:
        results = pickle.load(open(FOLDER + RUN_NAME + "_%d.pickle" % CONTINUE, "rb"))
    else:
        results = {}
    s = 0
    for ep in EPOCHS:
        for h in heritability:
            for c in qtl:
                for E in epi:
                    for D in dom:
                        for d in numDecoys:
                            for n in numSamples:
                                if s <= CONTINUE:
                                    s += 1
                                    continue
                                print("########################ITER: %d/%d (%.1f%%)" % (s, totIter, 100 * s / float(totIter)))
                                try:
                                    results[(h, c, D, E, d, n, ep)] = mainExecute((h, D, E, d, n, PERC_TRAIN, ep, c))
                                except Exception as e:
                                    print("ERROR: ", e)
                                    results[(h, c, D, E, d, n, ep)] = {}
                                s += 1
                                if s % 10 == 0:
                                    pickle.dump(results, open(FOLDER + RUN_NAME + "_%d.pickle" % s, "wb"))

    pickle.dump(results, open(FOLDER + RUN_NAME + "FINAL.pickle", "wb"))
    print("Done. Saved to", FOLDER + RUN_NAME + "FINAL.pickle")


if __name__ == '__main__':
    main()
