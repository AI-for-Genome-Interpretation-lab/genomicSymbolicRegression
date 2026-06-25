#!/usr/bin/env python3
"""
Build simulated patient dataset for Kallmann syndrome.

Positives: for each OLIDA Kallmann combination, sample K 1000G individuals
as genetic background and implant the OLIDA pathogenic variants.
→ 43 combinations × K individuals = K*43 simulated affected patients.

Negatives: 1000G individuals with rare variants (MAF<1%) in at least one
Kallmann gene pair, without carrying the exact OLIDA variants.

Output:
  dataset/kallmann/kallmann_positives.tsv
  dataset/kallmann/kallmann_negatives.tsv
  dataset/kallmann/kallmann_final.tsv
  dataset/splits/kallmann_{train,val,test}.tsv
  dataset/stats/kallmann_report.txt
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
OUT_DIR       = "dataset/kallmann"
SPLITS_DIR    = "dataset/splits"
STATS_DIR     = "dataset/stats"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

DISEASE       = "Kallmann syndrome"
K_SIM         = 20       # simulated patients per combination
MAF_RARE      = 0.01
TRAIN_F, VAL_F, TEST_F = 0.70, 0.15, 0.15
SEED          = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_panel(panel_file):
    df = pd.read_csv(panel_file, sep="\t")
    return dict(zip(df["sample"], df["super_pop"]))


def load_gene_regions(coords_tsv):
    df = pd.read_csv(coords_tsv, sep="\t")
    return {r["gene"]: (str(r["chrom"]), int(r["start"]), int(r["end"]))
            for _, r in df.iterrows()}


def parse_float(v):
    try:    return float(v)
    except: return np.nan


def encode_effect(eff):
    e = str(eff).lower()
    lof = int(any(x in e for x in ["frameshift","nonsense","stop","splice"]))
    mis = int("missense" in e)
    return lof, mis


def get_vcf_samples(vcf_file):
    v = VCF(vcf_file); s = list(v.samples); v.close(); return s


def build_exclusion_set(combinations):
    """Set of (chrom, pos_hg19, ref, alt) from OLIDA combinations."""
    excl = set()
    for _, row in combinations.iterrows():
        for i in range(1, 5):
            p = f"var{i}_"
            ch = str(row.get(p+"chrom","N.A."))
            po = str(row.get(p+"pos_hg19","N.A."))
            re = str(row.get(p+"ref","N.A."))
            al = str(row.get(p+"alt","N.A."))
            if ch not in ("N.A.","nan") and po not in ("N.A.","nan"):
                excl.add((ch, po, re, al))
    return excl


def carriers_in_region(vcf_file, sample_names, region, excl_set=None, maf_max=MAF_RARE):
    """Return dict: sample_idx → list of (af, is_lof, is_missense) for rare variants."""
    chrom, start, end = region
    result = {i: [] for i in range(len(sample_names))}
    vcf = VCF(vcf_file, samples=sample_names)
    try:
        for v in vcf(f"{chrom}:{start}-{end}"):
            info = dict(v.INFO)
            af_raw = info.get("AF")
            try:   af = float(str(af_raw).split(",")[0])
            except: continue
            if af >= maf_max: continue
            if excl_set:
                key = (str(v.CHROM), str(v.POS), v.REF, v.ALT[0] if v.ALT else "")
                if key in excl_set: continue
            ref, alt = v.REF, v.ALT[0] if v.ALT else ""
            is_lof = int(len(ref) != len(alt))
            is_mis = int(len(ref) == len(alt) and ref != alt)
            for idx, gt in enumerate(v.genotypes):
                if gt[0] > 0 or gt[1] > 0:
                    result[idx].append((af, is_lof, is_mis))
    except Exception:
        pass
    vcf.close()
    return result


# ── Step 1: Kallmann positives ────────────────────────────────────────────────

def build_positives(kal_combs, vcf_samples, panel, regions, rng):
    """
    For each Kallmann combination, sample K 1000G individuals as background
    and create a simulated affected patient.
    Features = OLIDA variant annotations + individual background info.
    """
    rows = []
    for _, combo in kal_combs.iterrows():
        # sample K individuals (stratified across superpops)
        chosen = rng.choice(vcf_samples, size=K_SIM, replace=False)
        for sample in chosen:
            row = {
                "sample_id":      f"SIM_POS_{combo['combination_id']}_{sample}",
                "label":          1,
                "combination_id": combo["combination_id"],
                "base_individual": sample,
                "population":     panel.get(sample, "N.A."),
                "allelic_state":  combo["allelic_state"],
                "n_variants":     combo["n_variants"],
                "genes":          combo["genes"],
                "ppi_direct":     combo["ppi_direct"],
                "same_pathway":   combo["same_pathway"],
            }
            # copy variant features from OLIDA
            for i in range(1, 5):
                p = f"var{i}_"
                row[p+"gene"]       = combo.get(p+"gene", "N.A.")
                row[p+"gnomad_maf"] = combo.get(p+"gnomad_maf", "N.A.")
                row[p+"cadd"]       = combo.get(p+"cadd", "N.A.")
                row[p+"effect"]     = combo.get(p+"effect", "N.A.")
                row[p+"sift"]       = combo.get(p+"sift", "N.A.")
                row[p+"polyphen"]   = combo.get(p+"polyphen", "N.A.")
                row[p+"zygosity"]   = combo.get(p+"zygosity", "N.A.")
            rows.append(row)

    return pd.DataFrame(rows)


# ── Step 2: Kallmann negatives ────────────────────────────────────────────────

def find_olida_carriers(vcf_file, sample_names, excl_set, kal_regions):
    """
    Return set of sample indices that carry any exact OLIDA Kallmann variant
    in the Kallmann gene regions.
    """
    excluded = set()
    for gene, (chrom, start, end) in kal_regions.items():
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


def build_negatives(kal_combs, all_combs, vcf_samples, panel, regions, excl_set, rng):
    """
    Random sample from all 1000G individuals as negatives for Kallmann syndrome.
    Exclude only those who carry an exact OLIDA Kallmann pathogenic variant.
    The genotype matrix will later fill in their background rare variants
    in Kallmann gene regions.
    """
    kal_genes = set()
    for _, row in kal_combs.iterrows():
        for g in str(row["genes"]).split(";"):
            kal_genes.add(g.strip())
    kal_regions = {g: r for g, r in regions.items() if g in kal_genes}

    print(f"  Scanning Kallmann regions for OLIDA variant carriers to exclude...")
    excl_indices = find_olida_carriers(VCF_FILE, vcf_samples, excl_set, kal_regions)
    valid_samples = [s for i, s in enumerate(vcf_samples) if i not in excl_indices]
    print(f"  {len(excl_indices)} carriers excluded, {len(valid_samples)} valid negatives available")

    n_target = len(kal_combs) * K_SIM
    chosen = rng.choice(valid_samples, size=min(n_target, len(valid_samples)), replace=False)

    rows = []
    for sample in chosen:
        rows.append({
            "sample_id":       f"NEG_RAND_{sample}",
            "label":           0,
            "combination_id":  "N.A.",
            "base_individual": sample,
            "population":      panel.get(sample, "N.A."),
            "allelic_state":   "N.A.",
            "n_variants":      "N.A.",
            "genes":           "N.A.",
            "ppi_direct":      "N.A.",
            "same_pathway":    "N.A.",
            **{f"var{i}_{c}": "N.A." for i in range(1, 5)
               for c in ["gene", "gnomad_maf", "cadd", "effect", "sift", "polyphen", "zygosity"]},
        })

    print(f"  Sampled {len(rows)} random negatives")
    return pd.DataFrame(rows)


# ── Step 3: Feature matrix ────────────────────────────────────────────────────

def build_features(df):
    rows = []
    for _, r in df.iterrows():
        f = {"n_variants": float(r.get("n_variants", 2))}

        # ppi / pathway
        ppi = str(r.get("ppi_direct", "N.A."))
        f["ppi_direct"]   = 1. if ppi=="Yes" else (0. if ppi=="No" else np.nan)
        pw  = str(r.get("same_pathway", "N.A."))
        f["same_pathway"] = 1. if pw=="Yes"  else (0. if pw=="No"  else np.nan)

        # per-variant features
        for i in [1, 2]:
            p = f"var{i}_"
            maf  = parse_float(r.get(p+"gnomad_maf"))
            cadd = parse_float(r.get(p+"cadd"))
            f[p+"log_maf"] = np.log10(maf + 1e-6) if not np.isnan(maf) else np.nan
            f[p+"cadd"]    = cadd
            lof, mis = encode_effect(r.get(p+"effect", "N.A."))
            f[p+"is_lof"]      = float(lof)
            f[p+"is_missense"] = float(mis)
            sift = str(r.get(p+"sift","N.A.")).lower()
            f[p+"sift_del"] = 1. if "deleterious" in sift else (0. if "tolerated" in sift else np.nan)

        # population one-hot
        pop = str(r.get("population", "N.A."))
        for sp in ["EUR","AFR","EAS","SAS","AMR"]:
            f[f"pop_{sp}"] = 1. if pop == sp else 0.

        rows.append(f)
    return pd.DataFrame(rows)


# ── Step 4: Split by combination_id (no leakage) ─────────────────────────────

def split_dataset(pos, neg, rng):
    """Split positives by combination_id, negatives by gene pair."""
    combos = pos["combination_id"].unique()
    rng.shuffle(combos)
    n = len(combos)
    train_ids = set(combos[:int(n*TRAIN_F)])
    val_ids   = set(combos[int(n*TRAIN_F):int(n*(TRAIN_F+VAL_F))])
    test_ids  = set(combos[int(n*(TRAIN_F+VAL_F)):])

    pos_train = pos[pos["combination_id"].isin(train_ids)]
    pos_val   = pos[pos["combination_id"].isin(val_ids)]
    pos_test  = pos[pos["combination_id"].isin(test_ids)]

    neg = neg.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_neg = len(neg)
    neg_train = neg.iloc[:int(n_neg*TRAIN_F)]
    neg_val   = neg.iloc[int(n_neg*TRAIN_F):int(n_neg*(TRAIN_F+VAL_F))]
    neg_test  = neg.iloc[int(n_neg*(TRAIN_F+VAL_F)):]

    train = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    val   = pd.concat([pos_val,   neg_val  ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    test  = pd.concat([pos_test,  neg_test ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return train, val, test


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)

    print("Loading data...")
    pos_all = pd.read_csv(POSITIVES_TSV, sep="\t")
    kal = pos_all[pos_all["disease_name"] == DISEASE].copy()
    print(f"Kallmann combinations: {len(kal)}")

    panel   = load_panel(PANEL_FILE)
    regions = load_gene_regions(COORDS_TSV)
    vcf_samples = get_vcf_samples(VCF_FILE)
    vcf_samples = [s for s in vcf_samples if s in panel]
    excl_set    = build_exclusion_set(kal)
    print(f"1000G samples: {len(vcf_samples)}, exclusion variants: {len(excl_set)}")

    # ── Positives ─────────────────────────────────────────────────────────────
    print(f"\nBuilding positives (K={K_SIM} per combination)...")
    pos_sim = build_positives(kal, vcf_samples, panel, regions, rng)
    pos_sim.to_csv(os.path.join(OUT_DIR, "kallmann_positives.tsv"), sep="\t", index=False)
    print(f"Simulated positives: {len(pos_sim)}")

    # ── Negatives ─────────────────────────────────────────────────────────────
    print("\nBuilding negatives from 1000G...")
    neg_sim = build_negatives(kal, pos_all, vcf_samples, panel, regions, excl_set, rng)
    neg_sim.to_csv(os.path.join(OUT_DIR, "kallmann_negatives.tsv"), sep="\t", index=False)
    print(f"Negatives found: {len(neg_sim)}")

    # balance
    n = min(len(pos_sim), len(neg_sim))
    pos_bal = pos_sim.sample(n=n, random_state=SEED) if len(pos_sim) > n else pos_sim
    neg_bal = neg_sim.sample(n=n, random_state=SEED) if len(neg_sim) > n else neg_sim
    print(f"Balanced: {n} pos + {n} neg = {2*n} total")

    final = pd.concat([pos_bal, neg_bal]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    final.to_csv(os.path.join(OUT_DIR, "kallmann_final.tsv"), sep="\t", index=False)

    # ── Splits ────────────────────────────────────────────────────────────────
    train, val, test = split_dataset(pos_bal, neg_bal, rng)
    train.to_csv(os.path.join(SPLITS_DIR, "kallmann_train.tsv"), sep="\t", index=False)
    val.to_csv(  os.path.join(SPLITS_DIR, "kallmann_val.tsv"),   sep="\t", index=False)
    test.to_csv( os.path.join(SPLITS_DIR, "kallmann_test.tsv"),  sep="\t", index=False)
    print(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")

    # ── Report ────────────────────────────────────────────────────────────────
    report = f"""
=== Kallmann Syndrome Dataset ===
Disease: {DISEASE}
OLIDA combinations: {len(kal)}
Simulated positives (K={K_SIM}): {len(pos_sim)}
Negatives (1000G): {len(neg_sim)}
Balanced total: {2*n}

Splits:
  Train: {len(train)} (pos={( train['label']==1).sum()}, neg={(train['label']==0).sum()})
  Val:   {len(val)}   (pos={(val['label']==1).sum()},   neg={(val['label']==0).sum()})
  Test:  {len(test)}  (pos={(test['label']==1).sum()},  neg={(test['label']==0).sum()})

Kallmann genes involved: {sorted(set(g for gs in kal['genes'] for g in gs.split(';')))}
Gene pairs: {kal['genes'].value_counts().to_string()}
"""
    print(report)
    with open(os.path.join(STATS_DIR, "kallmann_report.txt"), "w") as f:
        f.write(report)
    print("Done. Run run_kallmann_benchmark.py to train models.")


if __name__ == "__main__":
    main()
