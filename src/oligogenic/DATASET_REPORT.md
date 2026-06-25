# Kallmann Syndrome Oligogenic Dataset — Report

## 1. Obiettivo

Costruire un dataset di pazienti **simulati** per la sindrome di Kallmann (malattia oligogenica)
a partire da due fonti pubbliche:

- **OLIDA** (Oligogenic Diseases Database) — combinazioni patogene validate con metadati variante
- **1000 Genomes Project Phase 3** — 2504 individui sani con genotipo intero (GRCh37)

---

## 2. Fonti dati

### 2.1 OLIDA
- URL: `https://olida.ibsquare.be/`
- Download via Zenodo API: `https://zenodo.org/api/records/10732286/files/`
- File scaricati: `Combination.tsv`, `GeneCombination.tsv`, `SMALLVARIANT.tsv`, `Disease.tsv`
- Script: `parse_olida.py`
- Filtro: `FINALmeta ≥ 1` (confidenza minima) → **579 combinazioni positive** su 142 malattie

### 2.2 1000 Genomes Phase 3
- URL FTP: `https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/`
- Streaming remoto via `bcftools view --regions-file` su tutti i 22 autosomi + X
- Script: `extract_1kg_variants.py`
- Output: `dataset/raw/1kg/1kg_olida_regions_all.vcf.gz` (288 MB, regioni OLIDA intere)
- Campioni: **2504 individui**, 5 superpopolazioni (EUR, AFR, EAS, SAS, AMR)

---

## 3. Scelta del fenotipo: Sindrome di Kallmann

Tra le 142 malattie in OLIDA, Kallmann è la più rappresentata:
- **43 combinazioni oligogeniche** validate
- Geni coinvolti (30 con coordinate): FGFR1, PROKR2, ANOS1, CHD7, PROK2, SEMA3A, ...
- Combinazioni prevalenti: `ANOS1; PROKR2` (4×), `FGFR1; IL17RD` (4×), `FGF8; FGFR1` (3×)

---

## 4. Costruzione del dataset

### 4.1 Positivi simulati — `build_kallmann_dataset.py`

Per ogni combinazione OLIDA Kallmann:
1. Si campionano **K = 20 individui 1000G** come background genomico
2. Si crea un "paziente simulato": individuo reale + varianti patogene OLIDA **implantate**
3. Le varianti OLIDA vengono codificate con la loro zigosità (eterozigosi → dosage 1, omozigosi → dosage 2)

**Totale positivi**: 43 combinazioni × 20 individui = **860 campioni**

### 4.2 Negativi — `build_kallmann_dataset.py`

Strategia: **random sampling** da tutti i 2504 individui 1000G.

- Si esclude l'unico individuo trovato portatore di una variante OLIDA Kallmann esatta (VCF scan)
- Si campionano casualmente **860 individui** dai 2503 rimanenti
- I negativi sono biologicamente sani per Kallmann e hanno il loro background reale nei geni Kallmann

**Razionale**: entrambi i gruppi (pos/neg) hanno background rare variants nelle stesse regioni geniche → confronto simmetrico.

### 4.3 Split train/val/test — `build_kallmann_dataset.py`

- **Positivi**: split per `combination_id` (nessun leakage tra combinazioni)
  - Train: 30 combinazioni (600 campioni), Val: 7 (120), Test: 6 (140)
- **Negativi**: split random stratificato
- **Totale**: 1720 campioni → Train=1202, Val=249, Test=269

---

## 5. Feature matrix genotipica — `build_genotype_matrix.py`

### 5.1 Colonne background (17.480)
Scan del VCF 1000G nelle regioni dei 30 geni Kallmann (±2000 bp):
- Filtro: **MAF < 1%** (varianti rare)
- Encoding dosage: 0=ref/ref, 1=eterozigote, 2=omozigote alt
- Matrice densa: `(2504, 17480)` int8

### 5.2 Colonne OLIDA-novel (72)
Le varianti OLIDA Kallmann (73 totali) sono **ultra-rare**: solo 1 trovata nel VCF 1000G.
Le restanti 72 vengono aggiunte come colonne esplicite:
- Positivi: dosage = 1 (het) o 2 (hom) secondo la zigosità OLIDA
- Negativi: dosage = 0 (non portatori per definizione)

### 5.3 Matrice finale
| Split | Campioni | Feature |
|-------|----------|---------|
| Train | 1202 | 17552 |
| Val   | 249  | 17552 |
| Test  | 269  | 17552 |

- Sparsità: **99.8%** di zeri
- Positivi con almeno 1 variante: **100%** (tutte nelle colonne OLIDA-novel)

---

## 6. Benchmark — `run_full_comparison.py`

### Metodi
| Metodo | Preprocessing |
|--------|--------------|
| Logistic L1/L2 | TruncatedSVD (200 comp) + StandardScaler |
| MLP (128→64) | TruncatedSVD (200 comp) + StandardScaler |
| Random Forest (300 alberi) | Matrice dosage raw (sparse) |
| Feyn QLattice | TruncatedSVD (200 comp) + StandardScaler |
| PLINK2 → Logistic | Top-100 varianti per p-value (--glm logistic) |
| Jaccard kNN (k=10) | Similarità Jaccard vs training positives |
| Jaccard vs OLIDA | Jaccard vs unione varianti OLIDA training |

### Risultati (test set)
| Metodo | AUROC | AUPRC |
|--------|-------|-------|
| **PLINK2 → Logistic** | **0.711** | **0.754** |
| Random Forest | 0.612 | 0.732 |
| Logistic L1 | 0.599 | 0.649 |
| Logistic L2 | 0.597 | 0.641 |
| Feyn | 0.591 | 0.657 |
| MLP | 0.530 | 0.554 |
| Jaccard vs OLIDA | 0.481 | 0.494 |
| Jaccard kNN | 0.464 | 0.493 |

### Interpretazione
- **PLINK2 vince** perché seleziona direttamente le varianti più associate (feature selection supervisionato su 17K colonne) — trova le 72 colonne OLIDA senza diluirle con SVD
- **RF** è secondo grazie alla capacità di trovare feature sparse senza riduzione dimensionale
- **SVD diluisce** le 72 colonne OLIDA nelle 17K background → i modelli lineari e MLP perdono segnale
- **Jaccard è sotto random**: il denominatore (17K colonne sparse) è troppo grande rispetto alle 2-3 varianti OLIDA per campione

---

## 7. Limitazioni note

1. **Segnale dominato dalle 72 colonne OLIDA**: il task è in parte "riconosci le varianti patogene implantate", non solo il pattern oligogenico
2. **Varianti OLIDA non presenti in 1000G**: per costruzione, sono ultra-rare → separazione artificiale pos/neg
3. **Split per combination_id (positivi)**: garantisce no leakage, ma i test set sono piccoli (140 pos)
4. **Background non specifico per Kallmann**: i negativi sono individui random, non portatori asintomatici di varianti Kallmann

---

## 8. File prodotti

```
oligogenic_dataset/
├── dataset/
│   ├── raw/1kg/1kg_olida_regions_all.vcf.gz     # VCF 1000G (288 MB)
│   ├── processed/
│   │   ├── olida_positives.tsv                   # 579 combinazioni OLIDA (FINALmeta≥1)
│   │   ├── gene_coords.tsv                       # coordinate GRCh37 di 465 geni
│   │   └── olida_genes_hg19.bed                  # BED regioni geniche
│   ├── splits/
│   │   ├── kallmann_train.tsv                    # 1202 campioni
│   │   ├── kallmann_val.tsv                      # 249 campioni
│   │   └── kallmann_test.tsv                     # 269 campioni
│   └── kallmann/
│       ├── genotype_train/val/test.npz           # matrice dosage (X, y, variant_ids)
│       ├── full_comparison.png                   # ROC + PR tutti i metodi
│       └── comparison_results.csv               # tabella AUROC/AUPRC
├── parse_olida.py
├── make_gene_bed.py
├── extract_1kg_variants.py
├── build_kallmann_dataset.py
├── build_genotype_matrix.py
├── run_full_comparison.py
└── run_plink2_jaccard.py
```
