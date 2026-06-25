#!/usr/bin/env python3
"""
Step 2a — Fetch gene coordinates from Ensembl REST API and build BED file.

Input:  dataset/processed/olida_genes.txt
Output: dataset/processed/olida_genes.bed  (0-based, ±2kb padding, GRCh38)
        dataset/processed/gene_coords.tsv  (gene → chrom, start, end, strand)
"""

import requests, time, json, os, sys
import pandas as pd

GENES_FILE = "dataset/processed/olida_genes.txt"
BED_FILE   = "dataset/processed/olida_genes.bed"
COORDS_TSV = "dataset/processed/gene_coords.tsv"
ENSEMBL_URL = "https://rest.ensembl.org/lookup/symbol/homo_sapiens"
BATCH_SIZE = 50
PADDING = 2000


def fetch_batch(symbols):
    """POST batch lookup to Ensembl REST."""
    url = f"{ENSEMBL_URL}"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    body = {"symbols": symbols}
    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=30)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 5))
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP {r.status_code} for batch, retrying...")
                time.sleep(2)
        except Exception as e:
            print(f"  Error: {e}, retrying...")
            time.sleep(2)
    return {}


def main():
    with open(GENES_FILE) as f:
        genes = [g.strip() for g in f if g.strip()]
    print(f"Fetching coordinates for {len(genes)} genes from Ensembl (GRCh38)...")

    coords = {}
    for i in range(0, len(genes), BATCH_SIZE):
        batch = genes[i:i + BATCH_SIZE]
        print(f"  Batch {i//BATCH_SIZE + 1}/{(len(genes)-1)//BATCH_SIZE + 1} ({len(batch)} genes)...")
        result = fetch_batch(batch)
        for sym, info in result.items():
            if info and isinstance(info, dict):
                chrom = str(info.get("seq_region_name", ""))
                start = info.get("start", 0)
                end   = info.get("end", 0)
                strand = info.get("strand", 1)
                if chrom and start and end:
                    coords[sym] = {"chrom": chrom, "start": start, "end": end, "strand": strand}
        time.sleep(0.5)  # polite pause between batches

    found = len(coords)
    missing = [g for g in genes if g not in coords]
    print(f"\nFound: {found}/{len(genes)}")
    if missing:
        print(f"Missing ({len(missing)}): {', '.join(missing[:20])}" + ("..." if len(missing) > 20 else ""))

    # Save coords TSV
    rows = [{"gene": g, **v} for g, v in coords.items()]
    df = pd.DataFrame(rows)
    df.to_csv(COORDS_TSV, sep="\t", index=False)
    print(f"Saved gene coordinates → {COORDS_TSV}")

    # Write BED file (0-based half-open, sorted by chrom/start)
    bed_rows = []
    for gene, c in coords.items():
        chrom = c["chrom"]
        # Keep only standard chromosomes (1-22, X, Y)
        if not (chrom.isdigit() or chrom in ("X", "Y")):
            continue
        start = max(0, c["start"] - 1 - PADDING)  # convert to 0-based
        end   = c["end"] + PADDING
        bed_rows.append((chrom, start, end, gene))

    # Sort: numeric chroms first, then X/Y
    def chrom_key(row):
        c = row[0]
        return int(c) if c.isdigit() else {"X": 23, "Y": 24}.get(c, 99)

    bed_rows.sort(key=chrom_key)

    with open(BED_FILE, "w") as f:
        for chrom, start, end, gene in bed_rows:
            f.write(f"{chrom}\t{start}\t{end}\t{gene}\n")

    print(f"Saved {len(bed_rows)} BED regions → {BED_FILE}")
    print(f"Padding: ±{PADDING}bp")


if __name__ == "__main__":
    main()
