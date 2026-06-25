#!/usr/bin/env python3
"""
Build a genotype matrix for Kallmann syndrome.

For each sample (positive or negative):
  - Extract rare variant positions (MAF < 1%) in Kallmann gene regions
  - Encode genotype as dosage: 0=ref/ref, 1=het, 2=hom_alt

Columns = (a) rare variants found in 1000G VCF scan of Kallmann genes
        + (b) all OLIDA Kallmann variant positions (not present in 1000G by definition)

Positives (simulated): their 1000G background dosage + OLIDA variants implanted (dosage=1 or 2)
Negatives:             their 1000G dosage in Kallmann gene regions; OLIDA columns = 0

Output:
  dataset/kallmann/genotype_train/val/test.npz   (X, y, variant_ids, sample_ids)
"""

import os, sys
import numpy as np
import pandas as pd
from cyvcf2 import VCF

POSITIVES_TSV = "dataset/processed/olida_positives.tsv"
VCF_FILE      = "dataset/raw/1kg/1kg_olida_regions_all.vcf.gz"
COORDS_TSV    = "dataset/processed/gene_coords.tsv"
TRAIN_TSV     = "dataset/splits/kallmann_train.tsv"
VAL_TSV       = "dataset/splits/kallmann_val.tsv"
TEST_TSV      = "dataset/splits/kallmann_test.tsv"
OUT_DIR       = "dataset/kallmann"
os.makedirs(OUT_DIR, exist_ok=True)

DISEASE   = "Kallmann syndrome"
MAF_RARE  = 0.01
SEED      = 42


def load_gene_regions(coords_tsv):
    df = pd.read_csv(coords_tsv, sep="\t")
    return {r["gene"]: (str(r["chrom"]), int(r["start"]), int(r["end"]))
            for _, r in df.iterrows()}


def get_vcf_samples(vcf_file):
    v = VCF(vcf_file); s = list(v.samples); v.close(); return s


def get_kallmann_genes(pos_all):
    kal = pos_all[pos_all["disease_name"] == DISEASE]
    genes = set()
    for gs in kal["genes"]:
        for g in str(gs).split(";"):
            genes.add(g.strip())
    return genes


def collect_olida_kallmann_variants(kal_df):
    """
    Collect all unique variant positions from OLIDA Kallmann combinations.
    Returns list of "chrom:pos:ref:alt" strings (hg19 coordinates).
    """
    olida_vars = []
    seen = set()
    for _, row in kal_df.iterrows():
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
    """
    Scan VCF for all rare variants in given gene regions.
    Returns:
      variant_ids: list of "chrom:pos:ref:alt" strings
      sample_names: list of sample names in VCF order
      dosage_matrix: np.array (n_samples, n_variants) with 0/1/2
    """
    print(f"Scanning VCF for rare variants in {len(gene_regions)} gene regions...")
    vcf = VCF(vcf_file)
    sample_names = list(vcf.samples)
    vcf.close()

    # First pass: collect variant positions
    variant_list = []  # (chrom, pos, ref, alt)
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
                if key not in variant_list:
                    variant_list.append(key)
        except Exception:
            pass
        vcf.close()

    print(f"  Found {len(variant_list)} rare variant positions in 1000G")

    # Build variant index
    var_index = {v: i for i, v in enumerate(variant_list)}
    n_samples  = len(sample_names)
    n_variants = len(variant_list)

    # Second pass: fill dosage matrix
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
    print("Loading data...")
    pos_all = pd.read_csv(POSITIVES_TSV, sep="\t")
    kal     = pos_all[pos_all["disease_name"] == DISEASE].copy()
    regions = load_gene_regions(COORDS_TSV)

    kal_genes   = get_kallmann_genes(pos_all)
    kal_regions = {g: r for g, r in regions.items() if g in kal_genes}
    print(f"Kallmann genes with regions: {len(kal_regions)}")

    # Scan all rare 1000G variants in Kallmann gene regions
    variant_ids_1kg, sample_names, dosage_1kg = scan_rare_variants(VCF_FILE, kal_regions)
    sample_index = {s: i for i, s in enumerate(sample_names)}

    # Collect all OLIDA Kallmann variant positions (not in 1000G — ultra-rare disease variants)
    olida_vars = collect_olida_kallmann_variants(kal)
    # Only add positions NOT already in 1kg scan
    existing = set(variant_ids_1kg)
    novel_vars = [v for v in olida_vars if v not in existing]
    print(f"  OLIDA Kallmann variants total: {len(olida_vars)}, novel (not in 1000G): {len(novel_vars)}")

    # Full variant set: 1kg background + OLIDA novel positions
    all_variant_ids = variant_ids_1kg + novel_vars
    n_var = len(all_variant_ids)
    n_novel = len(novel_vars)
    novel_start = len(variant_ids_1kg)  # index where OLIDA-novel columns begin
    print(f"Total feature columns: {n_var} ({len(variant_ids_1kg)} from 1000G + {n_novel} OLIDA-novel)")

    # Variant index for full set
    full_var_index = {v: i for i, v in enumerate(all_variant_ids)}

    # Load splits
    train_df = pd.read_csv(TRAIN_TSV, sep="\t")
    val_df   = pd.read_csv(VAL_TSV,   sep="\t")
    test_df  = pd.read_csv(TEST_TSV,  sep="\t")

    def encode_olida_variants(pos_row):
        """Return dosage vector (length = n_var) for a positive sample."""
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
            # try hg19 first, then hg38
            pos38 = str(pos_row.get(p+"pos_hg38", "N.A."))
            key38 = f"{chrom}:{pos38}:{ref}:{alt}"
            for k in [key, key38]:
                if k in full_var_index:
                    vec[full_var_index[k]] = dose
                    break
        return vec

    def build_split_matrix(df):
        X_rows, y_rows, ids = [], [], []
        for _, row in df.iterrows():
            label = int(row["label"])
            sid   = str(row["sample_id"])

            if label == 1:
                combo_id = str(row.get("combination_id", "N.A."))
                pos_row = kal[kal["combination_id"] == combo_id]
                if len(pos_row) == 0:
                    vec = np.zeros(n_var, dtype=np.int8)
                else:
                    vec = encode_olida_variants(pos_row.iloc[0])
                    # Add background rare variants from the 1000G individual
                    base_indiv = str(row.get("base_individual", ""))
                    if base_indiv in sample_index:
                        bg = dosage_1kg[sample_index[base_indiv], :]
                        # Merge: take max of background and implanted variant
                        vec[:len(variant_ids_1kg)] = np.maximum(
                            vec[:len(variant_ids_1kg)], bg
                        )
            else:
                base_indiv = str(row.get("base_individual", ""))
                if base_indiv in sample_index:
                    bg = dosage_1kg[sample_index[base_indiv], :]
                    vec = np.zeros(n_var, dtype=np.int8)
                    vec[:len(variant_ids_1kg)] = bg
                else:
                    vec = np.zeros(n_var, dtype=np.int8)

            X_rows.append(vec)
            y_rows.append(label)
            ids.append(sid)

        return np.array(X_rows, dtype=np.int8), np.array(y_rows), ids

    print("\nBuilding genotype matrices for splits...")
    X_tr, y_tr, ids_tr = build_split_matrix(train_df)
    X_va, y_va, ids_va = build_split_matrix(val_df)
    X_te, y_te, ids_te = build_split_matrix(test_df)

    print(f"Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}")
    print(f"Sparsity: {100*(X_tr==0).mean():.1f}% zeros")
    n_pos_tr = (y_tr==1).sum()
    pos_with_vars = (X_tr[y_tr==1].sum(axis=1) > 0).sum()
    print(f"Positives in train with at least one variant: {pos_with_vars}/{n_pos_tr} ({100*pos_with_vars/n_pos_tr:.1f}%)")
    print(f"  - of which in OLIDA-novel columns: "
          f"{(X_tr[y_tr==1, novel_start:].sum(axis=1)>0).sum()}/{n_pos_tr}")

    # Save
    for split_name, (X, y, ids) in [("train", (X_tr, y_tr, ids_tr)),
                                      ("val",   (X_va, y_va, ids_va)),
                                      ("test",  (X_te, y_te, ids_te))]:
        np.savez_compressed(
            os.path.join(OUT_DIR, f"genotype_{split_name}.npz"),
            X=X, y=y,
            variant_ids=np.array(all_variant_ids),
            sample_ids=np.array(ids)
        )
    print(f"\nSaved genotype matrices → {OUT_DIR}/genotype_*.npz")
    print(f"n_variants (features): {n_var}")
    print(f"  - background (1000G): {len(variant_ids_1kg)}")
    print(f"  - OLIDA-novel:        {n_novel}")


if __name__ == "__main__":
    main()
