#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Merge PLINK2-only results into the main withRF pickle.
# Run after computePLINK2Only.py completes.
#
import pickle
import numpy as np

FOLDER = "results/run100_synthLarge/"

print("Loading main pickle...")
main = pickle.load(open(FOLDER + "run100FINAL100_withRF.pickle", "rb"))

print("Loading PLINK2 pickle...")
plink2 = pickle.load(open(FOLDER + "run100_PLINK2onlyFINAL.pickle", "rb"))

print("Main keys:", len(main))
print("PLINK2 keys:", len(plink2))

merged = 0
missing = 0
for k, v in plink2.items():
    if k in main and len(v) > 0:
        main[k]["PLINK2 Test"] = v.get("PLINK2 Test", ("PLINK2 Test", np.nan, np.nan, np.nan))
        main[k]["PLINK2Pairs"] = v.get("PLINK2Pairs", [])
        main[k]["PLINK2Features"] = v.get("PLINK2Features", [])
        main[k]["PLINK2EpiPairs"] = v.get("PLINK2EpiPairs", [])
        merged += 1
    else:
        missing += 1

print("Merged: %d, missing/empty: %d" % (merged, missing))
pickle.dump(main, open(FOLDER + "run100FINAL100_withRF_PLINK2.pickle", "wb"))
print("Saved to", FOLDER + "run100FINAL100_withRF_PLINK2.pickle")
