#!/usr/bin/env python3
"""
Benchmark pipeline on the oligogenic dataset.

Steps:
  1. Annotate negatives from 1000G VCF (gnomAD AF proxy, variant effect)
  2. Build feature matrix (consistent for pos and neg)
  3. Train: Ridge, Lasso, MLP, RF, Feyn
  4. Evaluate: ROC, PR curve, feature importance
  5. Save figures to dataset/stats/

Features used:
  var1/var2: gnomad_maf, cadd, is_lof, is_missense, is_synonymous
  combination: ppi_direct, same_pathway, n_variants
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cyvcf2 import VCF
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, precision_recall_curve,
                              classification_report)
import feyn
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
FINAL_DS   = "dataset/processed/final_dataset.tsv"
NEG_FILE   = "dataset/processed/negatives.tsv"
VCF_FILE   = "dataset/raw/1kg/1kg_olida_regions_all.vcf.gz"
COORDS_TSV = "dataset/processed/gene_coords.tsv"
GC_TSV     = "dataset/raw/olida/GeneCombination.tsv"
TRAIN_FILE = "dataset/splits/train.tsv"
VAL_FILE   = "dataset/splits/val.tsv"
TEST_FILE  = "dataset/splits/test.tsv"
STATS_DIR  = "dataset/stats"
os.makedirs(STATS_DIR, exist_ok=True)

LOF_EFFECTS = {"frameshift", "nonsense", "stop_gained", "splice_donor",
               "splice_acceptor", "start_lost"}


# ── 1. Annotate negatives ─────────────────────────────────────────────────────

def load_gene_regions(coords_tsv):
    df = pd.read_csv(coords_tsv, sep="\t")
    return {r["gene"]: (str(r["chrom"]), int(r["start"]), int(r["end"]))
            for _, r in df.iterrows()}


def annotate_sample_in_gene(vcf_path, sample_name, region):
    """Return (min_af, max_cadd_proxy, is_lof, is_missense) for rarest rare variant
    carried by sample in given gene region. Uses 1000G AF as MAF proxy.
    CADD not in 1000G VCF → return NaN.
    """
    chrom, start, end = region
    region_str = f"{chrom}:{start}-{end}"
    best_af = 1.0
    is_lof = 0
    is_missense = 0

    vcf = VCF(vcf_path, samples=[sample_name])
    sample_idx = 0  # only one sample loaded
    try:
        for v in vcf(region_str):
            info = dict(v.INFO)
            af_raw = info.get("AF")
            try:
                af = float(str(af_raw).split(",")[0])
            except (TypeError, ValueError):
                continue
            if af >= 0.01:
                continue  # not rare
            gt = v.genotypes[sample_idx]
            if gt[0] <= 0 and gt[1] <= 0:
                continue  # sample doesn't carry it
            if af < best_af:
                best_af = af
            # VEP consequence not in 1000G VCF — infer from ID/REF/ALT length
            ref, alt = v.REF, v.ALT[0] if v.ALT else ""
            if len(ref) != len(alt):
                is_lof = 1  # indel → frameshift proxy
            else:
                is_missense = 1
    except Exception:
        pass
    vcf.close()

    maf = best_af if best_af < 1.0 else np.nan
    return maf, np.nan, is_lof, is_missense  # (maf, cadd, is_lof, is_missense)


def annotate_negatives(neg_df, vcf_path, regions):
    """Add var1/var2 annotations to negatives from 1000G VCF."""
    print(f"Annotating {len(neg_df)} negatives from VCF (this may take a few minutes)...")
    records = []
    for i, (_, row) in enumerate(neg_df.iterrows()):
        if i % 50 == 0:
            print(f"  {i}/{len(neg_df)}...")
        sample = str(row["combination_id"]).split("_")[-1]  # extract 1000G sample ID
        geneA = str(row.get("var1_gene", "N.A.")).strip()
        geneB = str(row.get("var2_gene", "N.A.")).strip()

        maf1, cadd1, lof1, mis1 = (np.nan, np.nan, 0, 0)
        maf2, cadd2, lof2, mis2 = (np.nan, np.nan, 0, 0)

        if geneA in regions:
            maf1, cadd1, lof1, mis1 = annotate_sample_in_gene(vcf_path, sample, regions[geneA])
        if geneB in regions:
            maf2, cadd2, lof2, mis2 = annotate_sample_in_gene(vcf_path, sample, regions[geneB])

        records.append({
            "var1_gnomad_maf": maf1, "var1_cadd": cadd1,
            "var1_is_lof": lof1,    "var1_is_missense": mis1,
            "var2_gnomad_maf": maf2, "var2_cadd": cadd2,
            "var2_is_lof": lof2,    "var2_is_missense": mis2,
        })

    ann = pd.DataFrame(records, index=neg_df.index)
    for col in ann.columns:
        neg_df[col] = ann[col]
    return neg_df


# ── 2. Feature matrix ─────────────────────────────────────────────────────────

def encode_effect(effect_str):
    e = str(effect_str).lower()
    is_lof = int(any(x in e for x in ["frameshift", "nonsense", "stop", "splice"]))
    is_mis = int("missense" in e)
    return is_lof, is_mis


def parse_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def build_features(df):
    """Build numeric feature matrix from combined pos+neg dataframe."""
    rows = []
    for _, r in df.iterrows():
        # per-variant features (var1, var2)
        f = {}
        for i in [1, 2]:
            p = f"var{i}_"
            maf  = parse_float(r.get(p + "gnomad_maf"))
            cadd = parse_float(r.get(p + "cadd"))
            # log-transform MAF (rarity signal)
            f[p + "log_maf"] = np.log10(maf + 1e-6) if not np.isnan(maf) else np.nan
            f[p + "cadd"]    = cadd

            # variant effect
            lof_raw = r.get(p + "is_lof", None)
            mis_raw = r.get(p + "is_missense", None)
            if lof_raw is not None and str(lof_raw) not in ("N.A.", "nan", ""):
                f[p + "is_lof"]      = float(lof_raw)
                f[p + "is_missense"] = float(mis_raw) if mis_raw is not None else 0.0
            else:
                # encode from effect string
                lof, mis = encode_effect(r.get(p + "effect", "N.A."))
                f[p + "is_lof"]      = float(lof)
                f[p + "is_missense"] = float(mis)

        # combination-level features
        ppi = str(r.get("ppi_direct", "N.A."))
        f["ppi_direct"]   = 1.0 if ppi == "Yes" else (0.0 if ppi == "No" else np.nan)
        pw = str(r.get("same_pathway", "N.A."))
        f["same_pathway"] = 1.0 if pw == "Yes" else (0.0 if pw == "No" else np.nan)
        f["n_variants"]   = float(r.get("n_variants", 2))

        rows.append(f)

    feat_df = pd.DataFrame(rows)
    return feat_df


# ── 3. Train and evaluate ─────────────────────────────────────────────────────

def evaluate(model, X, y, model_name, method="sklearn"):
    if method == "feyn":
        df_test = pd.DataFrame(X, columns=FEAT_COLS)
        df_test["label"] = y.values
        probs = model.predict(df_test)
    else:
        probs = model.predict_proba(X)[:, 1]
    auroc = roc_auc_score(y, probs)
    auprc = average_precision_score(y, probs)
    return probs, auroc, auprc


def plot_roc_pr(results, y_test, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["steelblue", "darkorange", "green", "red", "purple"]
    for (name, probs), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, probs)
        auroc = roc_auc_score(y_test, probs)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auroc:.3f})", color=color)

        prec, rec, _ = precision_recall_curve(y_test, probs)
        auprc = average_precision_score(y_test, probs)
        axes[1].plot(rec, prec, label=f"{name} (AP={auprc:.3f})", color=color)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curve"); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0.5, color="k", ls="--", alpha=0.4)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_roc_pr.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_feature_importance(rf_model, feat_cols, out_dir):
    imp = rf_model.feature_importances_
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(imp)), imp[idx], color="steelblue")
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([feat_cols[i] for i in idx], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_rf_importance.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_auroc_bar(scores, out_dir):
    names = list(scores.keys())
    aurocs = [scores[n]["auroc"] for n in names]
    auprcs = [scores[n]["auprc"] for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, aurocs, 0.35, label="AUROC", color="steelblue")
    ax.bar(x + 0.2, auprcs, 0.35, label="AUPRC", color="darkorange")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1); ax.axhline(0.5, color="k", ls="--", alpha=0.4)
    ax.set_ylabel("Score"); ax.set_title("Model Benchmark — Test Set")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(out_dir, "benchmark_summary.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global FEAT_COLS

    # ── Load splits ───────────────────────────────────────────────────────────
    train = pd.read_csv(TRAIN_FILE, sep="\t")
    val   = pd.read_csv(VAL_FILE,   sep="\t")
    test  = pd.read_csv(TEST_FILE,  sep="\t")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # ── Annotate negatives (cached) ───────────────────────────────────────────
    ann_train = "dataset/splits/train_annotated.tsv"
    ann_val   = "dataset/splits/val_annotated.tsv"
    ann_test  = "dataset/splits/test_annotated.tsv"

    if all(os.path.exists(p) for p in [ann_train, ann_val, ann_test]):
        print("Loading cached annotated splits...")
        train = pd.read_csv(ann_train, sep="\t")
        val   = pd.read_csv(ann_val,   sep="\t")
        test  = pd.read_csv(ann_test,  sep="\t")
    else:
        regions = load_gene_regions(COORDS_TSV)

        def annotate_split(df):
            neg_mask = df["label"] == 0
            if neg_mask.sum() == 0:
                return df
            neg = df[neg_mask].copy()
            neg = annotate_negatives(neg, VCF_FILE, regions)
            df = df.copy()
            df.loc[neg_mask] = neg
            return df

        train = annotate_split(train)
        val   = annotate_split(val)
        test  = annotate_split(test)
        train.to_csv(ann_train, sep="\t", index=False)
        val.to_csv(ann_val,     sep="\t", index=False)
        test.to_csv(ann_test,   sep="\t", index=False)
        print("Annotated splits cached.")

    # ── Build feature matrices ────────────────────────────────────────────────
    X_train_df = build_features(train)
    X_val_df   = build_features(val)
    X_test_df  = build_features(test)
    FEAT_COLS  = list(X_train_df.columns)

    y_train = train["label"].values
    y_val   = val["label"].values
    y_test  = test["label"].values

    # Impute + scale
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_df)
    X_val   = imputer.transform(X_val_df)
    X_test  = imputer.transform(X_test_df)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    print(f"\nFeature matrix: {X_train.shape[1]} features")
    print("Features:", FEAT_COLS)

    # ── Train sklearn models ──────────────────────────────────────────────────
    models = {
        "Logistic": LogisticRegression(max_iter=500),
        "Lasso":    LogisticRegression(penalty="l1", solver="saga", max_iter=500, C=0.1),
        "MLP":      MLPClassifier(hidden_layer_sizes=(64, 32), activation="tanh",
                                  max_iter=400, learning_rate_init=1e-3),
        "RF":       RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    }

    scores = {}
    test_probs = {}
    print("\nTraining models...")
    for name, model in models.items():
        if name in ("Logistic", "Lasso", "MLP"):
            model.fit(X_train_s, y_train)
            probs = model.predict_proba(X_test_s)[:, 1]
        else:
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, probs)
        auprc = average_precision_score(y_test, probs)
        scores[name] = {"auroc": auroc, "auprc": auprc}
        test_probs[name] = probs
        print(f"  {name:12s}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}")

    # ── Train Feyn ────────────────────────────────────────────────────────────
    print("\nTraining Feyn QLattice...")
    train_feyn = pd.DataFrame(X_train, columns=FEAT_COLS)
    train_feyn["label"] = y_train
    test_feyn  = pd.DataFrame(X_test,  columns=FEAT_COLS)
    test_feyn["label"] = y_test

    ql = feyn.QLattice()
    feyn_models = ql.auto_run(
        train_feyn, output_name="label", kind="classification",
        n_epochs=10, max_complexity=8,
        function_names=["add", "multiply"],
        threads=8
    )
    best_feyn = feyn_models[0]
    feyn_probs = best_feyn.predict(test_feyn)
    feyn_auroc = roc_auc_score(y_test, feyn_probs)
    feyn_auprc = average_precision_score(y_test, feyn_probs)
    scores["Feyn"] = {"auroc": feyn_auroc, "auprc": feyn_auprc}
    test_probs["Feyn"] = feyn_probs
    print(f"  {'Feyn':12s}  AUROC={feyn_auroc:.3f}  AUPRC={feyn_auprc:.3f}")
    print(f"  Feyn model: {best_feyn.to_query_string()}")

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_roc_pr(test_probs, y_test, STATS_DIR)
    plot_feature_importance(models["RF"], FEAT_COLS, STATS_DIR)
    plot_auroc_bar(scores, STATS_DIR)

    # ── Text summary ──────────────────────────────────────────────────────────
    report_lines = ["\n=== BENCHMARK RESULTS (Test Set) ==="]
    for name, s in scores.items():
        report_lines.append(f"  {name:12s}  AUROC={s['auroc']:.3f}  AUPRC={s['auprc']:.3f}")
    report_lines.append(f"\nFeyn model: {best_feyn.to_query_string()}")
    report = "\n".join(report_lines)
    print(report)

    report_path = os.path.join(STATS_DIR, "benchmark_results.txt")
    with open(report_path, "a") as f:
        f.write(report + "\n")
    print(f"\nReport saved → {report_path}")
    print("Figures saved → dataset/stats/benchmark_*.png")


if __name__ == "__main__":
    main()
