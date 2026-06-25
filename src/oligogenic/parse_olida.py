#!/usr/bin/env python3
"""
Step 1 — Parse OLIDA data and produce olida_positives.tsv.

Input files (dataset/raw/olida/):
  Combination.tsv, SMALLVARIANT.tsv, GeneCombination.tsv, Disease.tsv

Output files (dataset/processed/):
  olida_positives.tsv   — filtered combinations (FINALmeta >= 1)
  olida_genes.txt       — unique gene symbols (one per line, for Step 2)
  olida_chroms.txt      — unique chromosomes needed (for 1000G download)

Note: Disease.tsv does not contain HPO terms — only Orphanet ID, OMIM ID,
disease name. HPO column is left empty; add from external source if needed.
"""

import pandas as pd
import numpy as np
import os
import re

RAW_DIR = "dataset/raw/olida"
OUT_DIR = "dataset/processed"
os.makedirs(OUT_DIR, exist_ok=True)


def parse_associated_variants(av_str):
    """Parse 'Associated Variants' field.
    Format: 'EntryId:Gene:pos_hg19,pos_hg38:cdna:zygosity; ...'
    Returns list of dicts with entry_id, gene, pos_hg19, pos_hg38, cdna, zygosity.
    """
    variants = []
    for part in av_str.split(";"):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) < 5:
            continue
        entry_id = int(fields[0].strip())
        gene = fields[1].strip()
        positions = fields[2].strip().split(",")
        pos_hg19 = positions[0].strip() if len(positions) > 0 else "N.A."
        pos_hg38 = positions[1].strip() if len(positions) > 1 else "N.A."
        cdna = fields[3].strip()
        zygosity = fields[4].strip()
        variants.append({
            "entry_id": entry_id,
            "gene": gene,
            "pos_hg19": pos_hg19,
            "pos_hg38": pos_hg38,
            "cdna": cdna,
            "zygosity": zygosity,
        })
    return variants


def allelic_state_label(n_variants):
    return {2: "digenic", 3: "trigenic", 4: "tetraallelic"}.get(n_variants, f"{n_variants}-allelic")


def main():
    # ── Load tables ──────────────────────────────────────────────────────────
    comb = pd.read_csv(os.path.join(RAW_DIR, "Combination.tsv"), sep="\t")
    sv   = pd.read_csv(os.path.join(RAW_DIR, "SMALLVARIANT.tsv"), sep="\t")
    gc   = pd.read_csv(os.path.join(RAW_DIR, "GeneCombination.tsv"), sep="\t")
    dis  = pd.read_csv(os.path.join(RAW_DIR, "Disease.tsv"), sep="\t")

    print(f"Combinations total: {len(comb)}")

    # ── Filter FINALmeta >= 1 ────────────────────────────────────────────────
    comb = comb[comb["FINALmeta"] >= 1].copy()
    print(f"Combinations FINALmeta >= 1: {len(comb)}")

    # ── Build SMALLVARIANT lookup: Entry Id → row ────────────────────────────
    sv_idx = sv.set_index("Entry Id")

    # ── Build GeneCombination lookup: OLIDA ID → row ─────────────────────────
    # gc['Oligogenic variant combinations'] can be "OLI001; OLI002"
    gc_expanded = []
    for _, row in gc.iterrows():
        for oid in row["Oligogenic variant combinations"].split(";"):
            oid = oid.strip()
            if oid:
                gc_expanded.append({
                    "olida_id": oid,
                    "genes": row["Genes"],
                    "genes_relationship": row["Genes Relationship"],
                    "ppi": row["Protein Interactions"],
                    "common_pathways": row["Common Pathways"],
                    "gene_meta": row["GENEmeta"],
                })
    gc_df = pd.DataFrame(gc_expanded).set_index("olida_id")

    # ── Build Disease lookup: OLIDA ID → disease row ─────────────────────────
    dis_expanded = []
    for _, row in dis.iterrows():
        for oid in str(row["Combinations"]).split(";"):
            oid = oid.strip()
            if oid:
                dis_expanded.append({
                    "olida_id": oid,
                    "orphanet_id": row["Orphanet ID"],
                    "disease_name": row["Disease Name"],
                    "omim_id": row["Omim Id"],
                    "icd10": row["ICD-10 ID"],
                })
    dis_df = pd.DataFrame(dis_expanded).drop_duplicates("olida_id").set_index("olida_id")

    # ── Build output rows ────────────────────────────────────────────────────
    rows = []
    all_genes = set()
    all_chroms = set()

    for _, combo in comb.iterrows():
        oid = combo["OLIDA ID"]
        av_str = combo["Associated Variants"]
        parsed_vars = parse_associated_variants(av_str)
        n_vars = len(parsed_vars)

        # Look up variant details from SMALLVARIANT
        var_details = []
        for pv in parsed_vars:
            eid = pv["entry_id"]
            if eid in sv_idx.index:
                sv_row = sv_idx.loc[eid]
                chrom = str(sv_row["Chromosome"]).strip()
                all_chroms.add(chrom)
                all_genes.add(pv["gene"])
                var_details.append({
                    "gene": pv["gene"],
                    "chrom": chrom,
                    "pos_hg19": sv_row["Genomic Position Hg19"],
                    "pos_hg38": sv_row["Genomic Position Hg38"],
                    "ref": sv_row["Ref Allele"],
                    "alt": sv_row["Alt Allele"],
                    "cdna": pv["cdna"],
                    "protein_change": sv_row["Protein Change"],
                    "zygosity": pv["zygosity"],
                    "dbsnp": sv_row["Dbsnp Id"],
                    "variant_effect": sv_row["Variant Effect"],
                    "gnomad_maf": sv_row["Gnomad Maf"],
                    "cadd": sv_row["CADD"],
                    "sift": sv_row["SIFT"],
                    "polyphen": sv_row["PP2 HVAR"],
                })
            else:
                var_details.append({k: "N.A." for k in [
                    "gene", "chrom", "pos_hg19", "pos_hg38", "ref", "alt",
                    "cdna", "protein_change", "zygosity", "dbsnp",
                    "variant_effect", "gnomad_maf", "cadd", "sift", "polyphen"
                ]})
                var_details[-1]["gene"] = pv["gene"]
                all_genes.add(pv["gene"])

        # Gene relationship info
        gc_info = gc_df.loc[oid] if oid in gc_df.index else {}
        genes_str = gc_info.get("genes", "; ".join(v["gene"] for v in var_details)) if isinstance(gc_info, pd.Series) else "; ".join(v["gene"] for v in var_details)
        ppi = gc_info.get("ppi", "N.A.") if isinstance(gc_info, pd.Series) else "N.A."
        pathways = gc_info.get("common_pathways", "N.A.") if isinstance(gc_info, pd.Series) else "N.A."
        genes_rel = gc_info.get("genes_relationship", "N.A.") if isinstance(gc_info, pd.Series) else "N.A."

        # Disease info
        dis_info = dis_df.loc[oid] if oid in dis_df.index else {}
        orphanet_id = dis_info.get("orphanet_id", "N.A.") if isinstance(dis_info, pd.Series) else "N.A."
        disease_name = dis_info.get("disease_name", combo["Diseases"]) if isinstance(dis_info, pd.Series) else combo["Diseases"]
        omim_id = dis_info.get("omim_id", combo["Omim Id"]) if isinstance(dis_info, pd.Series) else combo["Omim Id"]

        # Build flat row — up to 4 variants, pad with N.A.
        row = {
            "combination_id": oid,
            "label": 1,
            "allelic_state": allelic_state_label(n_vars),
            "n_variants": n_vars,
            "oligogenic_effect": combo["Oligogenic Effect"],
            "ethnicity": combo["Ethnicity"],
            "disease_name": disease_name,
            "orphanet_id": orphanet_id,
            "omim_id": omim_id,
            "hpo_terms": "",  # not in OLIDA files — add from external source
            "finalMeta_score": combo["FINALmeta"],
            "genes": genes_str,
            "genes_relationship": genes_rel,
            "ppi_direct": "Yes" if ppi not in ["N.A.", "", None] and ppi == ppi else "No",
            "same_pathway": "Yes" if pathways not in ["N.A.", "", None] and pathways == pathways else "No",
            "population": "OLIDA_case",
        }
        for i in range(4):
            prefix = f"var{i+1}_"
            if i < len(var_details):
                vd = var_details[i]
                row[prefix + "gene"] = vd["gene"]
                row[prefix + "chrom"] = vd["chrom"]
                row[prefix + "pos_hg38"] = vd["pos_hg38"]
                row[prefix + "pos_hg19"] = vd["pos_hg19"]
                row[prefix + "ref"] = vd["ref"]
                row[prefix + "alt"] = vd["alt"]
                row[prefix + "cdna"] = vd["cdna"]
                row[prefix + "protein_change"] = vd["protein_change"]
                row[prefix + "zygosity"] = vd["zygosity"]
                row[prefix + "dbsnp"] = vd["dbsnp"]
                row[prefix + "effect"] = vd["variant_effect"]
                row[prefix + "gnomad_maf"] = vd["gnomad_maf"]
                row[prefix + "cadd"] = vd["cadd"]
                row[prefix + "sift"] = vd["sift"]
                row[prefix + "polyphen"] = vd["polyphen"]
            else:
                for col in ["gene","chrom","pos_hg38","pos_hg19","ref","alt","cdna",
                            "protein_change","zygosity","dbsnp","effect",
                            "gnomad_maf","cadd","sift","polyphen"]:
                    row[prefix + col] = "N.A."

        rows.append(row)

    positives = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "olida_positives.tsv")
    positives.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved {len(positives)} positive combinations → {out_path}")

    # ── Save gene and chromosome lists ────────────────────────────────────────
    # Filter out invalid chroms
    valid_chroms = sorted(
        [c for c in all_chroms if c.isdigit() or c in ("X", "Y", "MT")],
        key=lambda x: int(x) if x.isdigit() else {"X": 23, "Y": 24, "MT": 25}[x]
    )
    genes_path = os.path.join(OUT_DIR, "olida_genes.txt")
    chroms_path = os.path.join(OUT_DIR, "olida_chroms.txt")
    with open(genes_path, "w") as f:
        f.write("\n".join(sorted(all_genes)) + "\n")
    with open(chroms_path, "w") as f:
        f.write("\n".join(valid_chroms) + "\n")
    print(f"Unique genes: {len(all_genes)} → {genes_path}")
    print(f"Unique chromosomes: {len(valid_chroms)} → {chroms_path}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("\n── Stats ──────────────────────────────────────────────────────────")
    print("Allelic state distribution:")
    print(positives["allelic_state"].value_counts().to_string())
    print("\nOligogenic effect distribution:")
    print(positives["oligogenic_effect"].value_counts().to_string())
    print("\nFINALmeta distribution:")
    print(positives["finalMeta_score"].value_counts().sort_index().to_string())
    print("\nTop 10 diseases:")
    print(positives["disease_name"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
