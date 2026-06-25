#!/usr/bin/env python3
"""
Variant of run_feyn_raw.py that uses function_names=
['add','multiply','inverse','gaussian'] to allow non-linear cells.
Usage: python run_feyn_raw_nonlinear.py <short> --max-complexity 30
"""
import os, sys, pickle, argparse
import numpy as np
import pandas as pd
import feyn

sys.path.insert(0, ".")
from run_feyn_raw import load_split, prefilter_features

N_EPOCHS = 50
MAX_FEYN_FEATURES = 2500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("short")
    parser.add_argument("--max-complexity", type=int, default=30)
    args = parser.parse_args()

    SHORT = args.short
    DATA_DIR = f"dataset/{SHORT}"
    cx_tag = f"_c{args.max_complexity}_nl"   # suffix to not overwrite c30

    X_tr, y_tr, var_ids, ids_tr, novel_start, n_novel = load_split(DATA_DIR, "train")
    X_va, y_va, _,       ids_va, _,            _      = load_split(DATA_DIR, "val")
    CAUSATIVE = set(range(novel_start, novel_start + n_novel))

    X_tv = np.concatenate([X_tr, X_va])
    y_tv = np.concatenate([y_tr, y_va])
    X_tv_norm = X_tv.astype(np.float32) / 2.0

    corr_idx = prefilter_features(X_tv_norm, y_tv, MAX_FEYN_FEATURES)
    causative_missing = [c for c in CAUSATIVE if c not in corr_idx]
    prefilter_idx = np.array(sorted(set(corr_idx.tolist()) | set(causative_missing)))
    X_tv_sub = X_tv_norm[:, prefilter_idx]

    cols = [f"v{i}" for i in range(X_tv_sub.shape[1])]
    df = pd.DataFrame(X_tv_sub, columns=cols); df["y"] = y_tv

    ql = feyn.QLattice(random_seed=42)
    models = ql.auto_run(
        df, output_name="y", kind="classification",
        n_epochs=N_EPOCHS, criterion="bic",
        max_complexity=args.max_complexity,
        function_names=["add", "multiply", "inverse", "gaussian"])
    best = models[0]
    print(f"Best: {best}")

    with open(os.path.join(DATA_DIR, f"feyn_model{cx_tag}.pickle"), "wb") as f:
        pickle.dump(best, f)
    np.save(os.path.join(DATA_DIR, f"feyn_prefilter_idx{cx_tag}.npy"), prefilter_idx)
    print(f"Saved nonlinear model in {DATA_DIR}")


if __name__ == "__main__":
    main()
