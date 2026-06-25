#!/usr/bin/env python3
"""
Step 3 — Build negative set from 1000 Genomes.

For each gene pair (geneA, geneB) in OLIDA positives:
  - Find 1000G individuals with at least one rare variant (gnomAD MAF < 1%)
    in BOTH genes simultaneously
  - Exclude individuals carrying exactly the same variants as a known OLIDA positive
  - Label = 0, hpo_terms = []

Input:
  dataset/processed/olida_positives.tsv
  dataset/raw/1kg/1kg_olida_regions_all.vcf.gz
  dataset/raw/1kg/integrated_call_samples_v3.20130502.ALL.panel
  dataset/processed/gene_coords.tsv  (GRCh37 gene regions)

Output:
  dataset/processed/negatives.tsv
"""

import os, sys
import pandas as pd
import numpy as np
from cyvcf2 import VCF

POSITIVES   = "dataset/processed/olida_positives.tsv"
VCF_FILE    = "dataset/raw/1kg/1kg_olida_regions_all.vcf.gz"
PANEL_FILE  = "dataset/raw/1kg/integrated_call_samples_v3.20130502.ALL.panel"
COORDS_TSV  = "dataset/processed/gene_coords.tsv"   # GRCh37
OUT_NEG     = "dataset/processed/negatives.tsv"
OUT_DIR     = "dataset/processed"
os.makedirs(OUT_DIR, exist_ok=True)

MAF_THRESHOLD = 0.01  # rare = gnomAD MAF < 1%


def load_gene_regions_grch37(coords_tsv):
    """Return dict: gene → (chrom, start, end) in GRCh37."""
    df = pd.read_csv(coords_tsv, sep="\t")
    regions = {}
    for _, r in df.iterrows():
        chrom = str(r["chrom"])
        if chrom.isdigit() or chrom in ("X", "Y"):
            regions[r["gene"]] = (chrom, int(r["start"]), int(r["end"]))
    return regions


def load_panel(panel_file):
    """Return dict: sample_id → superpopulation."""
    df = pd.read_csv(panel_file, sep="\t")
    return dict(zip(df["sample"], df["super_pop"]))


def parse_maf(info_dict, key="AF"):
    """Parse AF field from VCF INFO; return float or None."""
    val = info_dict.get(key)
    if val is None:
        return None
    try:
        return float(str(val).split(",")[0])
    except (ValueError, TypeError):
        return None


def get_rare_variant_carriers(vcf_path, gene, region, sample_names):
    """
    Return set of sample indices that carry at least one rare variant
    in the given gene region.
    rare = AF < MAF_THRESHOLD in 1000G global AF (INFO/AF).
    """
    chrom, start, end = region
    region_str = f"{chrom}:{start}-{end}"
    carriers = set()

    vcf = VCF(vcf_path, samples=sample_names)
    try:
        for variant in vcf(region_str):
            # Parse allele frequency from INFO
            info = dict(variant.INFO)
            af = parse_maf(info, "AF")
            if af is None:
                continue
            if af >= MAF_THRESHOLD:
                continue  # not rare
            # Check which samples carry this alt allele
            gts = variant.genotypes  # list of [allele1, allele2, phased]
            for idx, gt in enumerate(gts):
                a1, a2 = gt[0], gt[1]
                if a1 > 0 or a2 > 0:  # at least one alt allele
                    carriers.add(idx)
    except Exception as e:
        pass
    vcf.close()
    return carriers


def build_olida_exclusion_set(positives):
    """
    Build set of (chrom, pos, ref, alt) tuples from OLIDA positives
    to exclude exact matches from negatives.
    """
    exclusion = set()
    for i in range(1, 5):
        pfx = f"var{i}_"
        cols = [pfx + c for c in ["chrom", "pos_hg19", "ref", "alt"]]
        if all(c in positives.columns for c in cols):
            for _, row in positives.iterrows():
                chrom = str(row[pfx + "chrom"])
                pos   = str(row[pfx + "pos_hg19"])
                ref   = str(row[pfx + "ref"])
                alt   = str(row[pfx + "alt"])
                if chrom not in ("N.A.", "nan") and pos not in ("N.A.", "nan"):
                    exclusion.add((chrom, pos, ref, alt))
    return exclusion


def check_sample_exclusion(vcf_path, sample_names, exclusion_set, chrom, start, end):
    """Return set of sample indices that carry any OLIDA-exact variant."""
    region_str = f"{chrom}:{start}-{end}"
    excluded = set()
    vcf = VCF(vcf_path, samples=sample_names)
    try:
        for variant in vcf(region_str):
            key = (str(variant.CHROM), str(variant.POS), variant.REF, variant.ALT[0])
            if key in exclusion_set:
                gts = variant.genotypes
                for idx, gt in enumerate(gts):
                    if gt[0] > 0 or gt[1] > 0:
                        excluded.add(idx)
    except Exception:
        pass
    vcf.close()
    return excluded


def main():
    print("Loading positives...")
    positives = pd.read_csv(POSITIVES, sep="\t")
    print(f"  {len(positives)} positive combinations")

    # Extract unique gene pairs (only digenic for simplicity)
    gene_pairs = set()
    for _, row in positives.iterrows():
        genes = [g.strip() for g in str(row["genes"]).split(";") if g.strip()]
        if len(genes) >= 2:
            gene_pairs.add(tuple(sorted([genes[0], genes[1]])))
    print(f"  {len(gene_pairs)} unique gene pairs")

    print("Loading gene regions (GRCh37)...")
    regions = load_gene_regions_grch37(COORDS_TSV)

    print("Loading 1000G panel...")
    panel = load_panel(PANEL_FILE)
    all_samples = list(panel.keys())
    print(f"  {len(all_samples)} individuals")

    print("Building OLIDA exclusion set...")
    exclusion_set = build_olida_exclusion_set(positives)
    print(f"  {len(exclusion_set)} exact positions to exclude")

    if not os.path.exists(VCF_FILE):
        print(f"ERROR: VCF file not found: {VCF_FILE}")
        print("Run extract_1kg_variants.py first.")
        sys.exit(1)

    # Get sample names from VCF
    vcf_tmp = VCF(VCF_FILE)
    vcf_samples = list(vcf_tmp.samples)
    vcf_tmp.close()
    # Filter to only samples in panel
    sample_names = [s for s in vcf_samples if s in panel]
    sample_to_idx = {s: i for i, s in enumerate(sample_names)}
    print(f"  {len(sample_names)} samples in VCF + panel")

    # ── For each gene pair, find carriers in both genes ──────────────────────
    rows = []
    n_target = len(positives)  # aim for ~same number as positives

    print(f"\nSearching for negative candidates across {len(gene_pairs)} gene pairs...")
    for geneA, geneB in sorted(gene_pairs):

        regionA = regions.get(geneA)
        regionB = regions.get(geneB)
        if regionA is None or regionB is None:
            continue

        carriersA = get_rare_variant_carriers(VCF_FILE, geneA, regionA, sample_names)
        carriersB = get_rare_variant_carriers(VCF_FILE, geneB, regionB, sample_names)
        both = carriersA & carriersB

        if not both:
            continue

        # Exclude samples carrying exact OLIDA variants
        excl_A = check_sample_exclusion(VCF_FILE, sample_names, exclusion_set,
                                         regionA[0], regionA[1], regionA[2])
        excl_B = check_sample_exclusion(VCF_FILE, sample_names, exclusion_set,
                                         regionB[0], regionB[1], regionB[2])
        both -= (excl_A | excl_B)

        # Cap per gene pair to keep uniform distribution across pairs
        max_per_pair = 30
        both_list = list(both)
        if len(both_list) > max_per_pair:
            rng = np.random.default_rng(42)
            both_list = rng.choice(both_list, max_per_pair, replace=False).tolist()

        for idx in both_list:
            sample = sample_names[idx]
            rows.append({
                "combination_id": f"NEG_{geneA}_{geneB}_{sample}",
                "label": 0,
                "allelic_state": "digenic",
                "n_variants": 2,
                "oligogenic_effect": "N.A.",
                "ethnicity": panel.get(sample, "N.A."),
                "disease_name": "N.A.",
                "orphanet_id": "N.A.",
                "omim_id": "N.A.",
                "hpo_terms": "",
                "finalMeta_score": 0,
                "genes": f"{geneA}; {geneB}",
                "genes_relationship": "N.A.",
                "ppi_direct": "N.A.",
                "same_pathway": "N.A.",
                "population": panel.get(sample, "N.A."),
                "var1_gene": geneA,
                "var2_gene": geneB,
                # Remaining variant fields filled with N.A. (positions not extracted here)
                **{f"var{i}_{col}": "N.A."
                   for i in range(1, 5)
                   for col in ["chrom","pos_hg38","pos_hg19","ref","alt","cdna",
                               "protein_change","zygosity","dbsnp","effect",
                               "gnomad_maf","cadd","sift","polyphen"]},
            })

        print(f"  {geneA} x {geneB}: {len(both)} candidates (total so far: {len(rows)})")

    print(f"\nTotal candidates before balancing: {len(rows)}")

    if not rows:
        print("No negatives found. Check that the VCF file is complete.")
        sys.exit(1)

    negatives = pd.DataFrame(rows)

    # Downsample to match positives
    if len(negatives) > n_target:
        negatives = negatives.sample(n=n_target, random_state=42).reset_index(drop=True)
        print(f"Downsampled to {len(negatives)} to match positives")

    negatives.to_csv(OUT_NEG, sep="\t", index=False)
    print(f"Saved → {OUT_NEG}")
    print(f"Superpopulation distribution:")
    print(negatives["population"].value_counts().to_string())


if __name__ == "__main__":
    main()
