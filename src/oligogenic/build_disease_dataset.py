#!/usr/bin/env python3
"""
Generic simulated-patient dataset builder for any OLIDA disease.

Usage:
  python build_disease_dataset.py "Alport syndrome" alport
  python build_disease_dataset.py "Spinocerebellar ataxia type 17" sca17
  python build_disease_dataset.py "Familial hemophagocytic lymphohistiocytosis" fhl
"""

import os, sys
import numpy as np
import pandas as pd
from cyvcf2 import VCF
from sklearn.utils import shuffle as sk_shuffle

POSITIVES_TSV = "dataset/processed/olida_positives.tsv"
VCF_FILE      = "dataset/raw/1kg/1kg_olida_regions_all.vcf.gz"
PANEL_FILE    = "dataset/raw/1kg/integrated_call_samples_v3.20130502.ALL.panel"
COORDS_TSV    = "dataset/processed/gene_coords.tsv"
SPLITS_DIR    = "dataset/splits"
STATS_DIR     = "dataset/stats"
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

K_SIM = 20
MAF_RARE = 0.01
TRAIN_F, VAL_F, TEST_F = 0.70, 0.15, 0.15
SEED = 42


def load_panel(panel_file):
    df = pd.read_csv(panel_file, sep="\t")
    return dict(zip(df["sample"], df["super_pop"]))


def load_gene_regions(coords_tsv):
    df = pd.read_csv(coords_tsv, sep="\t")
    return {r["gene"]: (str(r["chrom"]), int(r["start"]), int(r["end"]))
            for _, r in df.iterrows()}


def get_vcf_samples(vcf_file):
    v = VCF(vcf_file); s = list(v.samples); v.close(); return s


def build_exclusion_set(combinations):
    excl = set()
    for _, row in combinations.iterrows():
        for i in range(1, 5):
            p = f"var{i}_"
            ch = str(row.get(p+"chrom", "N.A."))
            po = str(row.get(p+"pos_hg19", "N.A."))
            re = str(row.get(p+"ref", "N.A."))
            al = str(row.get(p+"alt", "N.A."))
            if ch not in ("N.A.", "nan") and po not in ("N.A.", "nan"):
                excl.add((ch, po, re, al))
    return excl


def find_olida_carriers(vcf_file, sample_names, excl_set, gene_regions):
    excluded = set()
    for gene, (chrom, start, end) in gene_regions.items():
        vcf = VCF(vcf_file, samples=sample_names)
        try:
            for v in vcf(f"{chrom}:{start}-{end}"):
                key = (str(v.CHROM), str(v.POS), v.REF, v.ALT[0] if v.ALT else "")
                if key in excl_set:
                    for idx, gt in enumerate(v.genotypes):
                        if gt[0] > 0 or gt[1] > 0:
                            excluded.add(idx)
        except Exception:
            pass
        vcf.close()
    return excluded


def build_positives(disease_combs, vcf_samples, panel, rng):
    rows = []
    for _, combo in disease_combs.iterrows():
        chosen = rng.choice(vcf_samples, size=K_SIM, replace=False)
        for sample in chosen:
            row = {
                "sample_id":       f"SIM_POS_{combo['combination_id']}_{sample}",
                "label":           1,
                "combination_id":  combo["combination_id"],
                "base_individual": sample,
                "population":      panel.get(sample, "N.A."),
                "allelic_state":   combo.get("allelic_state", "N.A."),
                "n_variants":      combo.get("n_variants", "N.A."),
                "genes":           combo.get("genes", "N.A."),
                "ppi_direct":      combo.get("ppi_direct", "N.A."),
                "same_pathway":    combo.get("same_pathway", "N.A."),
            }
            for i in range(1, 5):
                p = f"var{i}_"
                for c in ["gene", "gnomad_maf", "cadd", "effect", "sift", "polyphen", "zygosity",
                          "chrom", "pos_hg19", "pos_hg38", "ref", "alt"]:
                    row[p+c] = combo.get(p+c, "N.A.")
            rows.append(row)
    return pd.DataFrame(rows)


def build_negatives(disease_combs, vcf_samples, panel, excl_set, gene_regions, rng):
    disease_genes = set()
    for _, row in disease_combs.iterrows():
        for g in str(row["genes"]).split(";"):
            disease_genes.add(g.strip())
    d_regions = {g: r for g, r in gene_regions.items() if g in disease_genes}

    print(f"  Scanning {len(d_regions)} gene regions for OLIDA variant carriers...")
    excl_indices = find_olida_carriers(VCF_FILE, vcf_samples, excl_set, d_regions)
    valid = [s for i, s in enumerate(vcf_samples) if i not in excl_indices]
    print(f"  {len(excl_indices)} carriers excluded, {len(valid)} valid negatives")

    n_target = len(disease_combs) * K_SIM
    chosen = rng.choice(valid, size=min(n_target, len(valid)), replace=False)

    rows = []
    for sample in chosen:
        rows.append({
            "sample_id":       f"NEG_RAND_{sample}",
            "label":           0,
            "combination_id":  "N.A.",
            "base_individual": sample,
            "population":      panel.get(sample, "N.A."),
            **{k: "N.A." for k in
               ["allelic_state","n_variants","genes","ppi_direct","same_pathway"]},
            **{f"var{i}_{c}": "N.A." for i in range(1, 5)
               for c in ["gene","gnomad_maf","cadd","effect","sift","polyphen",
                         "zygosity","chrom","pos_hg19","pos_hg38","ref","alt"]},
        })
    print(f"  Sampled {len(rows)} negatives")
    return pd.DataFrame(rows)


def split_dataset(pos, neg, rng):
    """Stratify by combination_id: each combination contributes samples to all splits."""
    pos_train_rows, pos_val_rows, pos_test_rows = [], [], []

    for combo_id, group in pos.groupby("combination_id"):
        idx = group.index.tolist()
        rng_local = np.random.default_rng(abs(hash(combo_id)) % (2**31))
        rng_local.shuffle(idx)
        n = len(idx)
        n_train = max(1, int(n * TRAIN_F))
        n_val   = max(1, int(n * VAL_F))
        pos_train_rows.extend(idx[:n_train])
        pos_val_rows.extend(idx[n_train:n_train+n_val])
        pos_test_rows.extend(idx[n_train+n_val:])

    pos_train = pos.loc[pos_train_rows]
    pos_val   = pos.loc[pos_val_rows]
    pos_test  = pos.loc[pos_test_rows]

    neg = neg.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_neg = len(neg)
    neg_train = neg.iloc[:int(n_neg*TRAIN_F)]
    neg_val   = neg.iloc[int(n_neg*TRAIN_F):int(n_neg*(TRAIN_F+VAL_F))]
    neg_test  = neg.iloc[int(n_neg*(TRAIN_F+VAL_F)):]

    train = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    val   = pd.concat([pos_val,   neg_val  ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    test  = pd.concat([pos_test,  neg_test ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return train, val, test


def main():
    if len(sys.argv) < 3:
        print("Usage: build_disease_dataset.py <disease_name> <short_name>")
        sys.exit(1)

    DISEASE    = sys.argv[1]
    SHORT      = sys.argv[2]
    OUT_DIR    = f"dataset/{SHORT}"
    os.makedirs(OUT_DIR, exist_ok=True)

    rng = np.random.default_rng(SEED)

    print(f"Disease: {DISEASE}  →  short={SHORT}")
    pos_all = pd.read_csv(POSITIVES_TSV, sep="\t")
    disease_combs = pos_all[pos_all["disease_name"] == DISEASE].copy()
    print(f"OLIDA combinations: {len(disease_combs)}")

    panel       = load_panel(PANEL_FILE)
    gene_regions = load_gene_regions(COORDS_TSV)
    vcf_samples  = get_vcf_samples(VCF_FILE)
    vcf_samples  = [s for s in vcf_samples if s in panel]
    excl_set     = build_exclusion_set(disease_combs)
    print(f"1000G samples: {len(vcf_samples)}, exclusion variants: {len(excl_set)}")

    print(f"\nBuilding positives (K={K_SIM})...")
    pos_sim = build_positives(disease_combs, vcf_samples, panel, rng)
    pos_sim.to_csv(os.path.join(OUT_DIR, f"{SHORT}_positives.tsv"), sep="\t", index=False)
    print(f"Simulated positives: {len(pos_sim)}")

    print("\nBuilding negatives...")
    neg_sim = build_negatives(disease_combs, vcf_samples, panel, excl_set, gene_regions, rng)
    neg_sim.to_csv(os.path.join(OUT_DIR, f"{SHORT}_negatives.tsv"), sep="\t", index=False)

    n = min(len(pos_sim), len(neg_sim))
    pos_bal = pos_sim.sample(n=n, random_state=SEED) if len(pos_sim) > n else pos_sim
    neg_bal = neg_sim.sample(n=n, random_state=SEED) if len(neg_sim) > n else neg_sim
    print(f"Balanced: {n} pos + {n} neg = {2*n}")

    final = pd.concat([pos_bal, neg_bal]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    final.to_csv(os.path.join(OUT_DIR, f"{SHORT}_final.tsv"), sep="\t", index=False)

    train, val, test = split_dataset(pos_bal, neg_bal, rng)
    train.to_csv(os.path.join(SPLITS_DIR, f"{SHORT}_train.tsv"), sep="\t", index=False)
    val.to_csv(  os.path.join(SPLITS_DIR, f"{SHORT}_val.tsv"),   sep="\t", index=False)
    test.to_csv( os.path.join(SPLITS_DIR, f"{SHORT}_test.tsv"),  sep="\t", index=False)
    print(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")

    genes_involved = sorted(set(g.strip() for gs in disease_combs["genes"] for g in str(gs).split(";")))
    report = f"""=== {DISEASE} Dataset ===
Combinations: {len(disease_combs)}
Positives (K={K_SIM}): {len(pos_sim)}
Negatives: {len(neg_sim)}
Balanced total: {2*n}
Splits: train={len(train)}, val={len(val)}, test={len(test)}
Genes: {genes_involved}
Gene combos:
{disease_combs['genes'].value_counts().to_string()}
"""
    print(report)
    with open(os.path.join(STATS_DIR, f"{SHORT}_report.txt"), "w") as f:
        f.write(report)
    print(f"Done → {OUT_DIR}/")


if __name__ == "__main__":
    main()
