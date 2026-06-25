#!/usr/bin/env python3
"""
Benchmark on Kallmann genotype dosage matrix.

Features: 17552-column sparse 0/1/2 dosage matrix
  - 17480 columns: rare background variants from 1000G in Kallmann gene regions
  - 72 columns:    OLIDA Kallmann disease-causing variants (implanted in positives)

Models: Logistic (L2), Lasso (L1), MLP, Random Forest, Feyn QLattice
Dimensionality reduction via TruncatedSVD (LSA) before dense models.

Output: dataset/kallmann/genotype_benchmark_results.txt
        dataset/kallmann/genotype_roc.png
        dataset/kallmann/genotype_pr.png
        dataset/kallmann/genotype_feature_importance.png
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, precision_recall_curve)
warnings.filterwarnings("ignore")

DATA_DIR = "dataset/kallmann"
OUT_DIR  = DATA_DIR


def load_split(name):
    d = np.load(os.path.join(DATA_DIR, f"genotype_{name}.npz"), allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(int), d["variant_ids"], d["sample_ids"]


def reduce_dims(X_tr, X_va, X_te, n_components=200):
    """TruncatedSVD (works on sparse-like dense) → StandardScaler."""
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Xr_tr = svd.fit_transform(X_tr)
    Xr_va = svd.transform(X_va)
    Xr_te = svd.transform(X_te)
    sc = StandardScaler()
    Xr_tr = sc.fit_transform(Xr_tr)
    Xr_va = sc.transform(Xr_va)
    Xr_te = sc.transform(Xr_te)
    return Xr_tr, Xr_va, Xr_te, svd


def metrics(y_true, y_score, name):
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    print(f"  {name:30s}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
    return auroc, auprc


def main():
    print("Loading genotype matrices...")
    X_tr, y_tr, var_ids, _ = load_split("train")
    X_va, y_va, _,       _ = load_split("val")
    X_te, y_te, _,       _ = load_split("test")
    n_var = X_tr.shape[1]

    print(f"Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")
    print(f"Class balance train: {y_tr.mean():.3f} positive fraction")

    # Combine train+val for final evaluation
    X_tv = np.concatenate([X_tr, X_va])
    y_tv = np.concatenate([y_tr, y_va])

    # ── Dimensionality reduction ─────────────────────────────────────────────
    n_comp = min(200, X_tr.shape[1] - 1, X_tr.shape[0] - 1)
    print(f"\nApplying TruncatedSVD (n_components={n_comp})...")
    Xr_tr, Xr_va, Xr_te, svd = reduce_dims(X_tr, X_va, X_te, n_comp)
    Xr_tv = np.concatenate([Xr_tr, Xr_va])

    # Also reduce combined train+val (for RF/MLP final eval)
    svd2 = TruncatedSVD(n_components=n_comp, random_state=42)
    Xr_tv2 = svd2.fit_transform(X_tv)
    sc2 = StandardScaler()
    Xr_tv2 = sc2.fit_transform(Xr_tv2)
    Xr_te2 = sc2.transform(svd2.transform(X_te))

    results = {}

    print("\n=== Val-set evaluation (train on train, predict val) ===")
    # Logistic L2
    lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    lr.fit(Xr_tr, y_tr)
    sc_va = lr.predict_proba(Xr_va)[:,1]
    results["Logistic_L2_val"] = metrics(y_va, sc_va, "Logistic L2 (val)")

    # Lasso (L1 logistic)
    ll = LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=1000, random_state=42)
    ll.fit(Xr_tr, y_tr)
    sc_va_l1 = ll.predict_proba(Xr_va)[:,1]
    results["Logistic_L1_val"] = metrics(y_va, sc_va_l1, "Logistic L1 (val)")

    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
    mlp.fit(Xr_tr, y_tr)
    sc_va_mlp = mlp.predict_proba(Xr_va)[:,1]
    results["MLP_val"] = metrics(y_va, sc_va_mlp, "MLP (val)")

    # Random Forest (directly on sparse dosage — no SVD needed)
    print("  Training RF on raw dosage matrix...")
    rf = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    sc_va_rf = rf.predict_proba(X_va)[:,1]
    results["RF_val"] = metrics(y_va, sc_va_rf, "RF (val)")

    # ── Test-set evaluation ───────────────────────────────────────────────────
    print("\n=== Test-set evaluation (train on train+val, predict test) ===")

    lr2 = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    lr2.fit(Xr_tv, y_tv)
    sc_te = lr2.predict_proba(Xr_te)[:,1]
    results["Logistic_L2_test"] = metrics(y_te, sc_te, "Logistic L2 (test)")

    ll2 = LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=1000, random_state=42)
    ll2.fit(Xr_tv, y_tv)
    sc_te_l1 = ll2.predict_proba(Xr_te)[:,1]
    results["Logistic_L1_test"] = metrics(y_te, sc_te_l1, "Logistic L1 (test)")

    mlp2 = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
    mlp2.fit(Xr_tv, y_tv)
    sc_te_mlp = mlp2.predict_proba(Xr_te)[:,1]
    results["MLP_test"] = metrics(y_te, sc_te_mlp, "MLP (test)")

    rf2 = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                 n_jobs=-1, random_state=42)
    rf2.fit(X_tv, y_tv)
    sc_te_rf = rf2.predict_proba(X_te)[:,1]
    results["RF_test"] = metrics(y_te, sc_te_rf, "RF (test)")

    # ── Feyn ─────────────────────────────────────────────────────────────────
    try:
        import feyn
        print("\n  Training Feyn QLattice on SVD-reduced features (train+val → test)...")
        ql = feyn.QLattice(random_seed=42)
        df_tv = pd.DataFrame(Xr_tv2, columns=[f"pc{i}" for i in range(Xr_tv2.shape[1])])
        df_tv["y"] = y_tv
        df_te = pd.DataFrame(Xr_te2, columns=[f"pc{i}" for i in range(Xr_te2.shape[1])])

        models = ql.auto_run(df_tv, output_name="y", kind="classification",
                             n_epochs=15, criterion="bic")
        best = models[0]
        sc_te_feyn = best.predict(df_te)
        results["Feyn_test"] = metrics(y_te, sc_te_feyn, "Feyn (test)")
    except Exception as e:
        print(f"  Feyn skipped: {e}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    model_scores = [
        ("Logistic L2", sc_te, "steelblue"),
        ("Logistic L1", sc_te_l1, "royalblue"),
        ("MLP",         sc_te_mlp, "seagreen"),
        ("RF",          sc_te_rf, "tomato"),
    ]
    if "Feyn_test" in results:
        model_scores.append(("Feyn", sc_te_feyn, "darkorange"))

    # ROC
    ax = axes[0]
    for name, sc, col in model_scores:
        fpr, tpr, _ = roc_curve(y_te, sc)
        auc = roc_auc_score(y_te, sc)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} AUC={auc:.3f}")
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC — Kallmann genotype matrix (test)")
    ax.legend(fontsize=9)

    # PR
    ax = axes[1]
    for name, sc, col in model_scores:
        prec, rec, _ = precision_recall_curve(y_te, sc)
        ap = average_precision_score(y_te, sc)
        ax.plot(rec, prec, color=col, lw=2, label=f"{name} AP={ap:.3f}")
    ax.axhline(y_te.mean(), color="gray", ls="--", lw=1, label="Baseline")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall — Kallmann genotype matrix (test)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "genotype_roc_pr.png"), dpi=150)
    plt.close()
    print(f"\nPlot saved → {OUT_DIR}/genotype_roc_pr.png")

    # RF feature importance — top 30 variants
    importances = rf2.feature_importances_
    top_idx = np.argsort(importances)[::-1][:30]

    fig, ax = plt.subplots(figsize=(10, 6))
    vals = importances[top_idx]
    labels = [str(var_ids[i]) for i in top_idx]
    # Mark OLIDA-novel features
    colors = ["tomato" if i >= 17480 else "steelblue" for i in top_idx]
    ax.barh(range(len(vals)), vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels[::-1], fontsize=7)
    ax.set_xlabel("RF Feature Importance")
    ax.set_title("Top 30 Variants — RF\n(red = OLIDA disease variant, blue = 1000G background)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "genotype_feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot saved → {OUT_DIR}/genotype_feature_importance.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    lines = ["=" * 60, "GENOTYPE MATRIX BENCHMARK — Kallmann Syndrome",
             "=" * 60,
             f"Features: {n_var} variants ({n_var - 72} background + 72 OLIDA-novel)",
             f"Train: {X_tr.shape[0]}, Val: {X_va.shape[0]}, Test: {X_te.shape[0]}",
             f"Positive fraction (test): {y_te.mean():.3f}",
             "",
             "Test-set results:"]
    for k, (auroc, auprc) in results.items():
        if "test" in k:
            lines.append(f"  {k:25s}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUT_DIR, "genotype_benchmark_results.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\nReport saved → {OUT_DIR}/genotype_benchmark_results.txt")


if __name__ == "__main__":
    main()
