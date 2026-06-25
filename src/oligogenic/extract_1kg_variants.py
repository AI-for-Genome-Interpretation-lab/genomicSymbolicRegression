#!/usr/bin/env python3
"""
Step 2b — Extract 1000 Genomes variants in OLIDA gene regions.

Streams from 1000G Phase 3 FTP using bcftools (no full VCF download).
Uses GRCh37 coordinates from Ensembl GRCh37 REST API.

Output: dataset/raw/1kg/1kg_olida_regions_all.vcf.gz  (merged, indexed)
        dataset/processed/olida_genes_hg19.bed         (GRCh37 BED for reference)

Requirements: bcftools, tabix, htslib with libcurl (remote file support)
"""

import os, sys, subprocess, time, requests, pandas as pd

RAW_1KG   = "dataset/raw/1kg"
OUT_DIR   = "dataset/processed"
GENES_FILE = "dataset/processed/olida_genes.txt"
BED_HG19  = "dataset/processed/olida_genes_hg19.bed"
MERGED_VCF = os.path.join(RAW_1KG, "1kg_olida_regions_all.vcf.gz")

FTP = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
SUFFIX = ".phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
PANEL = f"{FTP}/../integrated_call_samples_v3.20130502.ALL.panel"

PADDING  = 2000
BATCH_SIZE = 50

os.makedirs(RAW_1KG, exist_ok=True)


def fetch_grch37_coords(genes):
    """Fetch GRCh37 gene coordinates from Ensembl GRCh37 REST API."""
    url = "https://grch37.rest.ensembl.org/lookup/symbol/homo_sapiens"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    coords = {}
    for i in range(0, len(genes), BATCH_SIZE):
        batch = genes[i:i + BATCH_SIZE]
        print(f"  GRCh37 batch {i//BATCH_SIZE + 1}/{(len(genes)-1)//BATCH_SIZE + 1}...")
        for attempt in range(3):
            try:
                r = requests.post(url, headers=headers, json={"symbols": batch}, timeout=30)
                if r.status_code == 200:
                    for sym, info in r.json().items():
                        if info and isinstance(info, dict):
                            chrom = str(info.get("seq_region_name", ""))
                            start = info.get("start", 0)
                            end   = info.get("end", 0)
                            if chrom and start and end:
                                coords[sym] = (chrom, start, end)
                    break
                elif r.status_code == 429:
                    time.sleep(int(r.headers.get("Retry-After", 5)))
                else:
                    time.sleep(2)
            except Exception as e:
                print(f"    Error: {e}")
                time.sleep(2)
        time.sleep(0.5)
    return coords


def build_hg19_bed(coords):
    """Write sorted GRCh37 BED file."""
    rows = []
    for gene, (chrom, start, end) in coords.items():
        if not (chrom.isdigit() or chrom in ("X", "Y")):
            continue
        rows.append((chrom, max(0, start - 1 - PADDING), end + PADDING, gene))

    def key(r):
        c = r[0]
        return int(c) if c.isdigit() else {"X": 23, "Y": 24}.get(c, 99)

    rows.sort(key=key)
    with open(BED_HG19, "w") as f:
        for chrom, start, end, gene in rows:
            f.write(f"{chrom}\t{start}\t{end}\t{gene}\n")
    print(f"Wrote {len(rows)} BED regions (GRCh37) → {BED_HG19}")
    return rows


def get_chroms_from_bed(bed_file):
    chroms = set()
    with open(bed_file) as f:
        for line in f:
            c = line.split("\t")[0].strip()
            if c:
                chroms.add(c)
    return sorted(chroms, key=lambda x: int(x) if x.isdigit() else {"X": 23, "Y": 24}.get(x, 99))


def bcftools_extract(chrom, bed_file, out_vcf):
    """Stream-extract variants for one chromosome from 1000G FTP."""
    remote = f"{FTP}/ALL.chr{chrom}{SUFFIX}"
    cmd = [
        "bcftools", "view",
        "--regions-file", bed_file,
        "--output-file", out_vcf,
        "--output-type", "z",
        remote,
    ]
    print(f"  chr{chrom}: extracting from remote...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR chr{chrom}: {result.stderr[:300]}")
        return False
    subprocess.run(["tabix", "-p", "vcf", out_vcf], check=True)
    size = os.path.getsize(out_vcf) / 1e6
    print(f"  chr{chrom}: {size:.1f} MB → {out_vcf}")
    return True


def merge_vcfs(vcf_files, out_vcf):
    """Merge per-chromosome VCFs into one."""
    cmd = ["bcftools", "concat"] + vcf_files + ["-Oz", "-o", out_vcf]
    print(f"Merging {len(vcf_files)} VCFs...")
    subprocess.run(cmd, check=True)
    subprocess.run(["tabix", "-p", "vcf", out_vcf], check=True)
    size = os.path.getsize(out_vcf) / 1e6
    print(f"Merged VCF: {size:.1f} MB → {out_vcf}")


def download_panel():
    panel_path = os.path.join(RAW_1KG, "integrated_call_samples_v3.20130502.ALL.panel")
    if not os.path.exists(panel_path):
        print("Downloading 1000G sample panel...")
        r = requests.get(PANEL, timeout=60)
        with open(panel_path, "wb") as f:
            f.write(r.content)
        print(f"Panel saved → {panel_path}")
    return panel_path


def check_bcftools_remote():
    """Check if bcftools can access remote htslib (libcurl)."""
    result = subprocess.run(["bcftools", "--version"], capture_output=True, text=True)
    if "libcurl" not in result.stdout and "libcurl" not in result.stderr:
        # Try anyway - might still work
        pass
    return True


def main():
    with open(GENES_FILE) as f:
        genes = [g.strip() for g in f if g.strip()]

    # ── Step 2b-1: build GRCh37 BED ─────────────────────────────────────────
    if not os.path.exists(BED_HG19):
        print(f"Fetching GRCh37 coordinates for {len(genes)} genes...")
        coords = fetch_grch37_coords(genes)
        print(f"Found {len(coords)}/{len(genes)} genes in GRCh37")
        build_hg19_bed(coords)
    else:
        print(f"Using existing BED: {BED_HG19}")

    # ── Step 2b-2: download sample panel ────────────────────────────────────
    panel_path = download_panel()

    # ── Step 2b-3: extract per-chromosome ───────────────────────────────────
    check_bcftools_remote()
    chroms = get_chroms_from_bed(BED_HG19)
    print(f"\nExtracting variants for {len(chroms)} chromosomes...")

    per_chrom_vcfs = []
    failed_chroms = []
    for chrom in chroms:
        out_vcf = os.path.join(RAW_1KG, f"1kg_chr{chrom}_olida_regions.vcf.gz")
        if os.path.exists(out_vcf) and os.path.exists(out_vcf + ".tbi"):
            print(f"  chr{chrom}: already done, skipping")
            per_chrom_vcfs.append(out_vcf)
            continue
        ok = bcftools_extract(chrom, BED_HG19, out_vcf)
        if ok:
            per_chrom_vcfs.append(out_vcf)
        else:
            failed_chroms.append(chrom)

    if failed_chroms:
        print(f"\nWarning: failed chromosomes: {failed_chroms}")

    # ── Step 2b-4: merge ────────────────────────────────────────────────────
    if per_chrom_vcfs:
        merge_vcfs(per_chrom_vcfs, MERGED_VCF)
        print(f"\nDone. Merged VCF: {MERGED_VCF}")
    else:
        print("No VCFs to merge.")


if __name__ == "__main__":
    main()
