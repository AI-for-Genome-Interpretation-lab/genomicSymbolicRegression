#!/usr/bin/env python3
"""
Run Feyn on raw dosage matrix (no dimensionality reduction) for a given disease.
Also computes Jaccard detection curve for Feyn features.

Improvements over v1:
  - Pre-filter features to top MAX_FEYN_FEATURES by |correlation| with y
    (reduces Feyn's search space for large feature matrices like Alport)
  - "Feyn → Logistic" pipeline: use Feyn-identified features as input to
    LogisticRegression for well-calibrated classification scores
  - More epochs (N_EPOCHS=50)

Usage:
  python run_feyn_raw.py <short_name>
  e.g.: python run_feyn_raw.py sca17
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

N_EPOCHS          = 50
MAX_FEYN_FEATURES = 2500  # pre-filter to this many features before Feyn
K_DETECT          = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
DEFAULT_MAX_COMPLEXITY = None  # None = Feyn default


def load_split(data_dir, name):
    d = np.load(os.path.join(data_dir, f"genotype_{name}.npz"), allow_pickle=True)
    return (d["X"].astype(np.float32), d["y"].astype(int),
            d["variant_ids"], d["sample_ids"],
            int(d["novel_start"]), int(d["n_novel"]))


def jaccard(s1, s2):
    s1, s2 = set(s1), set(s2)
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    return len(s1 & s2) / len(s1 | s2)


def jaccard_curve(ranked_indices, causative, k_vals):
    return [jaccard(set(ranked_indices[:k]), causative) for k in k_vals]


def prefilter_features(X, y, max_feats):
    """Return indices of top-max_feats features by |Pearson correlation| with y."""
    if X.shape[1] <= max_feats:
        return np.arange(X.shape[1])
    corr = np.array([
        abs(np.corrcoef(X[:, i], y)[0, 1]) if X[:, i].std() > 0 else 0.0
        for i in range(X.shape[1])
    ])
    return np.argsort(corr)[::-1][:max_feats]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("short_name")
    parser.add_argument("--max-complexity", type=int, default=None,
                        help="Feyn max_complexity (default: Feyn built-in default)")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="override N_EPOCHS (default 50)")
    parser.add_argument("--max-feyn-features", type=int, default=None,
                        help="override MAX_FEYN_FEATURES (default 2500)")
    args = parser.parse_args()
    if args.n_epochs is not None:
        global N_EPOCHS
        N_EPOCHS = args.n_epochs
    if args.max_feyn_features is not None:
        global MAX_FEYN_FEATURES
        MAX_FEYN_FEATURES = args.max_feyn_features

    SHORT        = args.short_name
    MAX_COMPLEX  = args.max_complexity
    DATA_DIR     = f"dataset/{SHORT}"
    OUT_DIR      = DATA_DIR

    # Method name suffix for CSV (so multiple complexity runs don't overwrite each other)
    cx_tag = f"_c{MAX_COMPLEX}" if MAX_COMPLEX is not None else ""
    if MAX_FEYN_FEATURES != 2500:
        cx_tag = cx_tag + f"_f{MAX_FEYN_FEATURES}"
    if N_EPOCHS != 50:
        cx_tag = cx_tag + f"_ep{N_EPOCHS}"

    print(f"Loading {SHORT}...")
    X_tr, y_tr, var_ids, ids_tr, novel_start, n_novel = load_split(DATA_DIR, "train")
    X_va, y_va, _,       ids_va, _,            _      = load_split(DATA_DIR, "val")
    X_te, y_te, _,       _,      _,            _      = load_split(DATA_DIR, "test")

    CAUSATIVE = set(range(novel_start, novel_start + n_novel))
    n_feat    = X_tr.shape[1]
    print(f"Features: {n_feat} total, {n_novel} OLIDA-causative")
    print(f"Samples: train={len(y_tr)}, val={len(y_va)}, test={len(y_te)}")

    X_tv = np.concatenate([X_tr, X_va])
    y_tv = np.concatenate([y_tr, y_va])

    import feyn

    # ── Pre-filter features ────────────────────────────────────────────────────
    X_tv_norm = X_tv / 2.0
    X_te_norm = X_te / 2.0

    corr_idx = prefilter_features(X_tv_norm, y_tv, MAX_FEYN_FEATURES)
    # Always include all OLIDA-causative SNPs regardless of their marginal correlation
    # (important for margbal datasets where causative SNPs have near-zero marginal corr)
    causative_missing = sorted(CAUSATIVE - set(corr_idx.tolist()))
    if causative_missing:
        prefilter_idx = np.array(sorted(set(corr_idx.tolist()) | set(causative_missing)))
    else:
        prefilter_idx = corr_idx
    n_pre = len(prefilter_idx)
    if n_pre < n_feat:
        print(f"\nPre-filtering: {n_feat} → {n_pre} features (top by |corr| with y + all causative)")
        olida_in_prefilter = sum(1 for i in prefilter_idx if i in CAUSATIVE)
        print(f"  OLIDA-causative retained: {olida_in_prefilter}/{n_novel}  "
              f"(force-added: {len(causative_missing)})")
    else:
        print(f"\nNo pre-filtering needed ({n_feat} features)")

    X_tv_sub = X_tv_norm[:, prefilter_idx]
    X_te_sub = X_te_norm[:, prefilter_idx]

    cols = [f"v{i}" for i in range(n_pre)]
    df_tv = pd.DataFrame(X_tv_sub, columns=cols)
    df_tv["y"] = y_tv
    df_te = pd.DataFrame(X_te_sub, columns=cols)

    # ── Feyn on pre-filtered dosage ────────────────────────────────────────────
    cx_str = f", max_complexity={MAX_COMPLEX}" if MAX_COMPLEX is not None else ""
    print(f"\nRunning Feyn ({n_pre} features, {N_EPOCHS} epochs{cx_str})...")
    ql = feyn.QLattice(random_seed=42)
    run_kwargs = dict(
        output_name="y",
        kind="classification",
        n_epochs=N_EPOCHS,
        criterion="bic",
    )
    if MAX_COMPLEX is not None:
        run_kwargs["max_complexity"] = MAX_COMPLEX
    models = ql.auto_run(df_tv, **run_kwargs)
    best = models[0]
    print(f"Best model: {best}")

    # ── Extract Feyn-selected features (indices in prefilter_idx space) ────────
    used_sub_names = []
    try:
        for node in best:
            name = getattr(node, "name", "")
            if name.startswith("v") and name[1:].isdigit():
                if name not in used_sub_names:
                    used_sub_names.append(name)
    except Exception:
        pass

    used_sub_idx = [int(n[1:]) for n in used_sub_names]          # indices in sub-space
    used_orig_idx = [int(prefilter_idx[i]) for i in used_sub_idx] # indices in original space
    print(f"Feyn selected {len(used_orig_idx)} features: {used_sub_names}")
    print(f"  Original indices: {used_orig_idx}")
    print(f"  OLIDA-causative used: {[i for i in used_orig_idx if i in CAUSATIVE]}")

    # ── Task 1a: Feyn raw score ────────────────────────────────────────────────
    sc_feyn = best.predict(df_te)
    auroc_feyn = roc_auc_score(y_te, sc_feyn)
    auprc_feyn = average_precision_score(y_te, sc_feyn)
    print(f"\nFeyn (raw score) — AUROC: {auroc_feyn:.4f}  AUPRC: {auprc_feyn:.4f}")

    # ── Task 1b: Feyn → Logistic (Feyn-selected features only) ───────────────
    if len(used_orig_idx) >= 1:
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        clf.fit(X_tv[:, used_orig_idx], y_tv)
        sc_fl = clf.predict_proba(X_te[:, used_orig_idx])[:, 1]
        auroc_fl = roc_auc_score(y_te, sc_fl)
        auprc_fl = average_precision_score(y_te, sc_fl)
        print(f"Feyn → Logistic     — AUROC: {auroc_fl:.4f}  AUPRC: {auprc_fl:.4f}")
    else:
        sc_fl, auroc_fl, auprc_fl = None, None, None
        print("Feyn → Logistic: no features selected by Feyn")

    # ── Build feature ranking (needed for both Task 1c and Task 2) ────────────
    # Rank: Feyn-selected features first, then pre-filtered by |corr|, then rest
    corr_full = np.array([
        abs(np.corrcoef(X_tv_norm[:, i], y_tv)[0, 1]) if X_tv_norm[:, i].std() > 0 else 0.0
        for i in range(n_feat)
    ])
    rest_pre  = [int(prefilter_idx[i]) for i in np.argsort(
                    corr_full[prefilter_idx])[::-1] if int(prefilter_idx[i]) not in set(used_orig_idx)]
    rest_out  = [i for i in np.argsort(corr_full)[::-1]
                 if i not in set(prefilter_idx.tolist()) and i not in set(used_orig_idx)]
    feyn_rank = np.array(used_orig_idx + rest_pre + rest_out)

    # ── Task 1c: Feyn rank → Logistic (top-n_novel features from ranking) ─────
    # Feyn's BIC model is intentionally sparse (~5 features). For multi-variant
    # diseases we need to capture ALL causative variants. Use the ranking to
    # select top-N features (N = n_novel), then fit LogisticRegression — same
    # strategy as PLINK2 → Logistic.
    top_n = min(n_novel, len(feyn_rank))
    top_feats = feyn_rank[:top_n]
    clf_rank = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf_rank.fit(X_tv[:, top_feats], y_tv)
    sc_flr = clf_rank.predict_proba(X_te[:, top_feats])[:, 1]
    auroc_flr = roc_auc_score(y_te, sc_flr)
    auprc_flr = average_precision_score(y_te, sc_flr)
    print(f"Feyn rank→Logistic  — AUROC: {auroc_flr:.4f}  AUPRC: {auprc_flr:.4f}  (top-{top_n} features)")

    rank_file = os.path.join(OUT_DIR, f"feyn_raw_rank{cx_tag}.npy")
    np.save(rank_file, feyn_rank)
    print(f"\nRanking saved → {rank_file}")

    # Save the actual formula features separately (needed for epiPairs-style Jaccard)
    model_feats_file = os.path.join(OUT_DIR, f"feyn_model_features{cx_tag}.npy")
    np.save(model_feats_file, np.array(used_orig_idx, dtype=np.int64))
    print(f"Model features saved → {model_feats_file}  ({len(used_orig_idx)} features: {used_orig_idx})")

    # Save prefilter_idx (maps v_i → original column index, needed for sympify pair extraction)
    prefilter_file = os.path.join(OUT_DIR, f"feyn_prefilter_idx{cx_tag}.npy")
    np.save(prefilter_file, prefilter_idx)
    print(f"Prefilter idx saved → {prefilter_file}  ({len(prefilter_idx)} features)")

    # Save best model + top-8 list (for multi-model second-derivative heatmap)
    import pickle as _pickle
    model_pickle_file = os.path.join(OUT_DIR, f"feyn_model{cx_tag}.pickle")
    with open(model_pickle_file, "wb") as _f:
        _pickle.dump(best, _f)
    print(f"Model pickle saved  → {model_pickle_file}")

    top8 = models[:8]
    top8_pickle_file = os.path.join(OUT_DIR, f"feyn_top8{cx_tag}.pickle")
    with open(top8_pickle_file, "wb") as _f:
        _pickle.dump(top8, _f)
    print(f"Top-8 models saved  → {top8_pickle_file}  ({len(top8)} models)")

    # ── Jaccard detection curve ────────────────────────────────────────────────
    k_vals = [k for k in K_DETECT if k <= n_feat]
    curve  = jaccard_curve(feyn_rank, CAUSATIVE, k_vals)

    method_raw  = f"Feyn raw{cx_tag}"
    method_fl   = f"Feyn→Logistic{cx_tag}"
    method_flr  = f"Feyn rank→Logistic{cx_tag}"

    print(f"\n{'K':>7}  {method_raw:>20}")
    print("-" * 30)
    for k, j in zip(k_vals, curve):
        print(f"{k:>7}  {j:>20.4f}")

    # ── Update comparison_results.csv ─────────────────────────────────────────
    existing_csv = os.path.join(OUT_DIR, "comparison_results.csv")
    if os.path.exists(existing_csv):
        df_res = pd.read_csv(existing_csv)
        df_res = df_res[~df_res["method"].isin([method_raw, method_fl, method_flr])]
    else:
        df_res = pd.DataFrame()

    new_rows = [{"method": method_raw, "auroc": auroc_feyn, "auprc": auprc_feyn}]
    if auroc_fl is not None:
        new_rows.append({"method": method_fl, "auroc": auroc_fl, "auprc": auprc_fl})
    new_rows.append({"method": method_flr, "auroc": auroc_flr, "auprc": auprc_flr})
    df_res = pd.concat([df_res, pd.DataFrame(new_rows)], ignore_index=True)
    df_res.to_csv(existing_csv, index=False)
    print(f"\nUpdated {existing_csv}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    # Detection curve
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(len(k_vals)), curve, color="#DD8452", ls="-", lw=2.2,
            marker="o", ms=5, label="Feyn (raw)")
    k_exact_idx = next((i for i, k in enumerate(k_vals) if k >= n_novel), len(k_vals)-1)
    ax.axvline(k_exact_idx, color="gray", ls=":", lw=1.2, alpha=0.7,
               label=f"K = {n_novel} (# causative)")
    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels([str(k) for k in k_vals], rotation=45, fontsize=8)
    ax.set_xlabel("Top-K features selected", fontsize=11)
    ax.set_ylabel("Jaccard", fontsize=10)
    ax.set_title(f"Feyn (raw) causative detection — {SHORT.upper()}", fontsize=11)
    ax.legend(); ax.set_ylim(-0.02, 1.05); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feyn_raw_detection.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ROC + PR
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(*roc_curve(y_te, sc_feyn)[:2],
                 color="#DD8452", lw=2.2, label=f"Feyn raw (AUC={auroc_feyn:.3f})")
    if sc_fl is not None:
        axes[0].plot(*roc_curve(y_te, sc_fl)[:2],
                     color="#8B4513", lw=2.2, ls="--", label=f"Feyn→Logistic (AUC={auroc_fl:.3f})")
    axes[0].plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title(f"ROC — {SHORT.upper()} Feyn")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    prec, rec, _ = precision_recall_curve(y_te, sc_feyn)
    axes[1].plot(rec, prec, color="#DD8452", lw=2.2, label=f"Feyn raw (AP={auprc_feyn:.3f})")
    if sc_fl is not None:
        prec2, rec2, _ = precision_recall_curve(y_te, sc_fl)
        axes[1].plot(rec2, prec2, color="#8B4513", lw=2.2, ls="--",
                     label=f"Feyn→Logistic (AP={auprc_fl:.3f})")
    axes[1].axhline(y_te.mean(), color="gray", ls="--", lw=1)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR — {SHORT.upper()} Feyn")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feyn_raw_roc.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
