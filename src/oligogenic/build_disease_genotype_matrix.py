#!/usr/bin/env python3
"""
Generic genotype matrix builder for any OLIDA disease.

Usage:
  python build_disease_genotype_matrix.py "Alport syndrome" alport
  python build_disease_genotype_matrix.py "Spinocerebellar ataxia type 17" sca17
  python build_disease_genotype_matrix.py "Familial hemophagocytic lymphohistiocytosis" fhl

Output: dataset/<short>/genotype_{train,val,test}.npz
"""

import os, sys
import numpy as np
import pandas as pd
from cyvcf2 import VCF

POSITIVES_TSV = "dataset/processed/olida_positives.tsv"
VCF_FILE      = "dataset/raw/1kg/1kg_olida_regions_all.vcf.gz"
COORDS_TSV    = "dataset/processed/gene_coords.tsv"
MAF_RARE      = 0.01
SEED          = 42

# Noise defaults (overridable via CLI)
DEFAULT_NOISE_FN = 0.0   # prob of NOT implanting a causative variant in a positive
DEFAULT_NOISE_FP = 0.0   # prob that each OLIDA-novel SNP is set to 1 in a negative


def load_gene_regions(coords_tsv):
    df = pd.read_csv(coords_tsv, sep="\t")
    return {r["gene"]: (str(r["chrom"]), int(r["start"]), int(r["end"]))
            for _, r in df.iterrows()}


def get_vcf_samples(vcf_file):
    v = VCF(vcf_file); s = list(v.samples); v.close(); return s


def get_disease_genes(disease_combs):
    genes = set()
    for gs in disease_combs["genes"]:
        for g in str(gs).split(";"):
            genes.add(g.strip())
    return genes


def collect_olida_variants(disease_df):
    olida_vars = []
    seen = set()
    for _, row in disease_df.iterrows():
        for i in range(1, 5):
            p = f"var{i}_"
            chrom = str(row.get(p+"chrom", "N.A."))
            pos   = str(row.get(p+"pos_hg19", "N.A."))
            ref   = str(row.get(p+"ref", "N.A."))
            alt   = str(row.get(p+"alt", "N.A."))
            if chrom in ("N.A.", "nan", "") or pos in ("N.A.", "nan", ""):
                continue
            key = f"{chrom}:{pos}:{ref}:{alt}"
            if key not in seen:
                seen.add(key)
                olida_vars.append(key)
    return olida_vars


def scan_rare_variants(vcf_file, gene_regions, maf_max=MAF_RARE):
    print(f"Scanning VCF for rare variants in {len(gene_regions)} gene regions...")
    vcf = VCF(vcf_file)
    sample_names = list(vcf.samples)
    vcf.close()

    variant_list = []
    seen = set()
    for gene, (chrom, start, end) in gene_regions.items():
        vcf = VCF(vcf_file)
        try:
            for v in vcf(f"{chrom}:{start}-{end}"):
                info = dict(v.INFO)
                af_raw = info.get("AF")
                try:   af = float(str(af_raw).split(",")[0])
                except: continue
                if af >= maf_max: continue
                alt = v.ALT[0] if v.ALT else ""
                key = (str(v.CHROM), v.POS, v.REF, alt)
                if key not in seen:
                    seen.add(key)
                    variant_list.append(key)
        except Exception:
            pass
        vcf.close()

    print(f"  Found {len(variant_list)} rare 1000G variant positions")

    var_index = {v: i for i, v in enumerate(variant_list)}
    n_samples  = len(sample_names)
    n_variants = len(variant_list)
    dosage = np.zeros((n_samples, n_variants), dtype=np.int8)

    for gene, (chrom, start, end) in gene_regions.items():
        vcf = VCF(vcf_file)
        try:
            for v in vcf(f"{chrom}:{start}-{end}"):
                info = dict(v.INFO)
                af_raw = info.get("AF")
                try:   af = float(str(af_raw).split(",")[0])
                except: continue
                if af >= maf_max: continue
                alt = v.ALT[0] if v.ALT else ""
                key = (str(v.CHROM), v.POS, v.REF, alt)
                if key not in var_index: continue
                vi = var_index[key]
                for si, gt in enumerate(v.genotypes):
                    dose = int(gt[0] > 0) + int(gt[1] > 0)
                    if dose > 0:
                        dosage[si, vi] = dose
        except Exception:
            pass
        vcf.close()

    variant_ids = [f"{c}:{p}:{r}:{a}" for c, p, r, a in variant_list]
    return variant_ids, sample_names, dosage


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("disease_name")
    parser.add_argument("short_name")
    parser.add_argument("--noise-fn", type=float, default=DEFAULT_NOISE_FN,
                        help="Prob of NOT implanting each causative variant in a positive (false negative rate)")
    parser.add_argument("--noise-fp", type=float, default=DEFAULT_NOISE_FP,
                        help="Prob of setting each OLIDA-novel SNP to 1 in a negative (false positive rate)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override output directory (default: dataset/<short_name>)")
    args = parser.parse_args()

    DISEASE  = args.disease_name
    SHORT    = args.short_name
    NOISE_FN = args.noise_fn
    NOISE_FP = args.noise_fp
    OUT_DIR  = args.out_dir if args.out_dir else f"dataset/{SHORT}"
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"Disease: {DISEASE}  →  short={SHORT}")
    print(f"Noise: FN={NOISE_FN:.3f}  FP={NOISE_FP:.4f}")
    pos_all  = pd.read_csv(POSITIVES_TSV, sep="\t")
    disease_df = pos_all[pos_all["disease_name"] == DISEASE].copy()
    print(f"OLIDA combinations: {len(disease_df)}")

    regions     = load_gene_regions(COORDS_TSV)
    d_genes     = get_disease_genes(disease_df)
    d_regions   = {g: r for g, r in regions.items() if g in d_genes}
    print(f"Gene regions found: {len(d_regions)} / {len(d_genes)} genes")
    if len(d_regions) < len(d_genes):
        missing = d_genes - set(d_regions.keys())
        print(f"  Missing coords for: {missing}")

    variant_ids_1kg, sample_names, dosage_1kg = scan_rare_variants(VCF_FILE, d_regions)
    sample_index = {s: i for i, s in enumerate(sample_names)}

    olida_vars  = collect_olida_variants(disease_df)
    existing    = set(variant_ids_1kg)
    novel_vars  = [v for v in olida_vars if v not in existing]
    print(f"OLIDA variants total: {len(olida_vars)}, novel (not in 1000G): {len(novel_vars)}")

    all_variant_ids = variant_ids_1kg + novel_vars
    n_var       = len(all_variant_ids)
    novel_start = len(variant_ids_1kg)
    full_var_index = {v: i for i, v in enumerate(all_variant_ids)}
    print(f"Total features: {n_var}  ({len(variant_ids_1kg)} background + {len(novel_vars)} OLIDA-novel)")

    train_df = pd.read_csv(f"dataset/splits/{SHORT}_train.tsv", sep="\t")
    val_df   = pd.read_csv(f"dataset/splits/{SHORT}_val.tsv",   sep="\t")
    test_df  = pd.read_csv(f"dataset/splits/{SHORT}_test.tsv",  sep="\t")

    def encode_olida_variants(pos_row):
        vec = np.zeros(n_var, dtype=np.int8)
        for i in range(1, 5):
            p = f"var{i}_"
            chrom = str(pos_row.get(p+"chrom", "N.A."))
            pos   = str(pos_row.get(p+"pos_hg19", "N.A."))
            ref   = str(pos_row.get(p+"ref", "N.A."))
            alt   = str(pos_row.get(p+"alt", "N.A."))
            if chrom in ("N.A.", "nan", ""): continue
            zyg  = str(pos_row.get(p+"zygosity", "Heterozygous")).lower()
            dose = 2 if "homo" in zyg else 1
            key  = f"{chrom}:{pos}:{ref}:{alt}"
            pos38 = str(pos_row.get(p+"pos_hg38", "N.A."))
            key38 = f"{chrom}:{pos38}:{ref}:{alt}"
            for k in [key, key38]:
                if k in full_var_index:
                    vec[full_var_index[k]] = dose
                    break
        return vec

    def build_split_matrix(df):
        X_rows, y_rows, ids = [], [], []
        n_novel_total = n_var - novel_start
        for _, row in df.iterrows():
            label = int(row["label"])
            sid   = str(row["sample_id"])
            vec   = np.zeros(n_var, dtype=np.int8)

            if label == 1:
                combo_id = str(row.get("combination_id", "N.A."))
                pos_rows = disease_df[disease_df["combination_id"] == combo_id]
                if len(pos_rows) > 0:
                    vec = encode_olida_variants(pos_rows.iloc[0])
                base_indiv = str(row.get("base_individual", ""))
                if base_indiv in sample_index:
                    bg = dosage_1kg[sample_index[base_indiv], :]
                    vec[:novel_start] = np.maximum(vec[:novel_start], bg)
                # False-negative noise: randomly zero out implanted causative variants
                if NOISE_FN > 0:
                    for j in range(novel_start, n_var):
                        if vec[j] > 0 and rng.random() < NOISE_FN:
                            vec[j] = 0
            else:
                base_indiv = str(row.get("base_individual", ""))
                if base_indiv in sample_index:
                    bg = dosage_1kg[sample_index[base_indiv], :]
                    vec[:novel_start] = bg
                # False-positive noise: randomly set OLIDA-novel SNPs to 1
                if NOISE_FP > 0 and n_novel_total > 0:
                    fp_mask = rng.random(n_novel_total) < NOISE_FP
                    vec[novel_start:][fp_mask] = 1

            X_rows.append(vec)
            y_rows.append(label)
            ids.append(sid)

        return np.array(X_rows, dtype=np.int8), np.array(y_rows), ids

    print("\nBuilding genotype matrices...")
    X_tr, y_tr, ids_tr = build_split_matrix(train_df)
    X_va, y_va, ids_va = build_split_matrix(val_df)
    X_te, y_te, ids_te = build_split_matrix(test_df)

    print(f"Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")
    sparsity = 100*(X_tr==0).mean()
    print(f"Sparsity: {sparsity:.1f}% zeros")

    n_pos = (y_tr==1).sum()
    pos_with_olida = (X_tr[y_tr==1, novel_start:].sum(axis=1) > 0).sum()
    print(f"Positives with OLIDA signal in train: {pos_with_olida}/{n_pos} ({100*pos_with_olida/n_pos:.1f}%)")

    for split_name, (X, y, ids) in [("train", (X_tr, y_tr, ids_tr)),
                                      ("val",   (X_va, y_va, ids_va)),
                                      ("test",  (X_te, y_te, ids_te))]:
        np.savez_compressed(
            os.path.join(OUT_DIR, f"genotype_{split_name}.npz"),
            X=X, y=y,
            variant_ids=np.array(all_variant_ids),
            sample_ids=np.array(ids),
            novel_start=np.array(novel_start),
            n_novel=np.array(len(novel_vars))
        )

    print(f"\nSaved → {OUT_DIR}/genotype_*.npz")
    print(f"  background: {len(variant_ids_1kg)}, OLIDA-novel: {len(novel_vars)}")


if __name__ == "__main__":
    main()
