# KANGengi — Oligogenic Dataset: Risultati

## Dataset: Kallmann Syndrome

| | |
|---|---|
| Malattia | Sindrome di Kallmann (più rappresentata in OLIDA) |
| Combinazioni OLIDA | 43 |
| Positivi simulati | 860 (43 combinazioni × 20 individui 1000G) |
| Negativi | 860 (random sampling da 2504 individui 1000G) |
| Totale | 1720 campioni |
| Split | Train=1202 / Val=249 / Test=269 |

**Feature matrix**: 17.552 varianti × campione (dosage 0/1/2)
- 17.480 colonne: varianti rare (MAF<1%) nei 30 geni Kallmann da 1000G
- 72 colonne: varianti patogene OLIDA (ultra-rare, non presenti in 1000G → aggiunte come colonne esplicite)
- Sparsità: 99.8% zeri
- Background medio per campione: ~33 varianti rare (identico tra positivi e negativi)

---

## Task 1 — Classificazione caso/controllo

> Dato il genotipo di un individuo nei geni Kallmann, distingui positivi (portatori di una combinazione OLIDA patogena) da negativi (individui sani random).

| Metodo | AUROC | AUPRC | Note |
|--------|-------|-------|------|
| **PLINK2 → Logistic** | **0.711** | **0.754** | Feature selection supervisionato: top-100 varianti per p-value |
| Random Forest | 0.612 | 0.732 | Trova direttamente colonne OLIDA sparse |
| Logistic L1 | 0.599 | 0.649 | SVD diluisce il segnale OLIDA |
| Logistic L2 | 0.597 | 0.641 | SVD diluisce il segnale OLIDA |
| Feyn | 0.591 | 0.657 | 3 PC usati; segnale OLIDA disperso nell'SVD |
| MLP | 0.530 | 0.554 | Non generalizza su features così sparse |
| Jaccard vs OLIDA | 0.481 | 0.494 | Denominatore troppo grande (17K colonne) |
| Jaccard kNN | 0.464 | 0.493 | Test combos diverse da train → Jaccard basso |

**Interpretazione**: PLINK2 vince perché fa feature selection direttamente sul segnale genotipico grezzo. RF secondo per lo stesso motivo. I metodi basati su SVD (Feyn, Logistic, MLP) perdono perché SVD mescola le 72 colonne OLIDA nelle 17K background. Jaccard è inefficace in questo spazio ad alta sparsità.

---

## Task 2 — Identificazione varianti causative (Fig 3 style)

> Dato il ranking di feature di ogni metodo, quante delle 72 varianti OLIDA causative riesce a recuperare nelle sue top-K scelte?

**Metrica**: Jaccard(top-K feature del metodo, 72 colonne OLIDA-causative)

| K selezionato | RF | PLINK2 | Feyn | Logistic L1 |
|---|---|---|---|---|
| 10 | 0.139 | 0.139 | 0.038 | 0.079 |
| 20 | 0.260 | **0.278** | 0.082 | 0.070 |
| **50** | **0.671** | **0.671** | 0.080 | 0.052 |
| 100 | 0.564 | 0.564 | 0.068 | 0.036 |
| 200 | 0.295 | 0.295 | 0.058 | 0.023 |
| 1000 | 0.061 | 0.061 | 0.051 | 0.006 |

**Risultati chiave**:
- **RF e PLINK2 sono equivalenti**: picco Jaccard=0.67 a K=50 → selezionando le 50 varianti più importanti, recuperano ~48 delle 72 varianti causative
- Il picco a K=50 (non K=72) indica che alcune varianti OLIDA sono meno discriminative (condivise tra poche combinazioni, segnale debole)
- **Feyn** rimane basso (max ~0.08) su tutto il range: usare solo 3 PC non permette di risalire alle varianti originali specifiche
- **Logistic L1** fallisce: la penalizzazione su 17K colonne sparse non seleziona le colonne OLIDA

---

## Considerazioni sul dataset

### Punti di forza
- Negativi simmetrici: stesso background genotipico nei geni Kallmann (no leakage strutturale)
- Split per combination_id: no leakage tra combinazioni nei positivi
- Segnale biologicamente corretto: le varianti patogene OLIDA sono effettivamente discriminative

### Limitazioni
- Il segnale è quasi interamente nelle 72 colonne OLIDA → il problema è in parte "riconosci varianti note"
- OLIDA variants ultra-rare: separazione artificiale (0 vs non-0) tra pos/neg
- Background non Kallmann-specifico: i negativi non sono portatori asintomatici di varianti Kallmann
- Test set piccolo per i positivi (140 campioni, 6 combinazioni)

---

## File principali

| File | Descrizione |
|------|-------------|
| `build_kallmann_dataset.py` | Costruisce pos/neg, split |
| `build_genotype_matrix.py` | Matrice dosage 17552 varianti |
| `run_full_comparison.py` | Benchmark classificazione (Task 1) |
| `printJaccardDetection.py` | Jaccard feature detection (Task 2) |
| `dataset/kallmann/full_comparison.png` | ROC + PR tutti i metodi |
| `dataset/kallmann/jaccard_detection.png` | Jaccard vs K (Fig 3 style) |
| `dataset/kallmann/comparison_results.csv` | Tabella AUROC/AUPRC |
| `DATASET_REPORT.md` | Dettagli costruzione dataset |
