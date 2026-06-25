# Symbolic Regression for Epistatic Oligogenic Traits

Code and figure pipeline for the paper
**"Insights into epistatic oligogenic traits using interpretable symbolic regression models"**
(F. Codicè, D. Raimondi).

Symbolic Regression (Feyn / QLattice) is benchmarked against linear (Lasso, Ridge)
and non-linear (MLP, Random Forest, Gradient Boosting) models on synthetic SNP-array
phenotypes, biophysical fitness landscapes, the GB1 protein landscape, and a
semi-synthetic OLIDA oligogenic-disease benchmark.

---

## 1. Layout

```
.
├── figures/            # ← ALL paper figures land here
│   ├── *.png           #     main-text figures
│   └── suppl_figs/     #     supplementary figures
├── src/                # all code
│   ├── generate/       #   synthetic dataset generators
│   ├── synthetic/
│   │   ├── fit/        #   model fitting   → results/*.pickle   (slow)
│   │   └── plot/       #   plotting        → figures/           (fast)
│   ├── elife/          #   GB1 protein analysis
│   └── oligogenic/     #   OLIDA benchmark (self-contained)
├── data/               # input datasets        (large files git-ignored)
├── results/            # precomputed model bundles (git-ignored, ~7 GB)
├── paper_assets/       # slide decks / text used to assemble composite figures
├── archive/            # superseded scripts & outputs (git-ignored, nothing deleted)
├── Makefile            # regenerate the figures
├── requirements.txt
└── README.md
```

**Golden rule: run everything from the repository root.** Scripts resolve
`data/` and `results/` with paths relative to the root.

The plot scripts **never write to the repo root**: they write into a scratch
folder (`figures/_build/`, override with `FIGOUT`) and the Makefile installs the
final PNGs into `figures/` and `figures/suppl_figs/`, then deletes the scratch.

---

## 2. Quick start

```bash
conda activate kangengi          # env with feyn installed
pip install -r requirements.txt  # if needed

make all                         # regenerate every "direct" figure into figures/
```

`make all` runs only the plotting scripts (seconds–minutes); it does **not** refit
models — the fitted bundles ship under `results/`. It produces:

| figures/ | figures/suppl_figs/ |
|---|---|
| `qtl_main_MetricPearson_CV.png` (Fig 2) | `qtl_suppl_MetricPearson_CV[_std].png`, `qtl_suppl_MetricR2_CV[_std].png`, `qtl_main_MetricRMSE_CV[_std].png` |
| `craterHeatmap_CV.png` (Fig 4) | `craterHeatmap_CV_std/_R2/_R2_std/_RMSE/_RMSE_std.png` |
| `epiJaccardDomComparison.png` | `benchmark_times.png` |
| `jaccardHeatmap.png` (OLIDA) | |

### Composite figures (assembled by hand)

Four figures are montages of several sub-panels arranged in slides/Inkscape, so
`make all` does not rebuild them — their final versions already sit in `figures/`.
To regenerate the **panels** (into `figures/_build/`):

```bash
make fig1-toy        # Fig 1  toy phenotypes      → figures/toyPheno1.png
make fig3-epipairs   # Fig 3  epistatic pairs     → figures/epiPairs.png
make fig5-lineplots  # Fig 5  crater/mesa lines   → figures/lineplots.png
make figelife        # GB1 (Untitled.png) panels  → figures/Untitled.png
```
(`paper_assets/Fig5_elife_Untitled_presentation.pdf` is the GB1 assembly source.)

---

## 3. Figure → script map

| Figure | `figures/` file | Make target | Script | Reads |
|---|---|---|---|---|
| Fig 1 | `toyPheno1.png` * | `fig1-toy` | `src/synthetic/plot/printToyPhenotypes.py` | `data/synthetic/toy*` |
| Fig 2 | `qtl_main_MetricPearson_CV.png` | `fig2-synth` | `src/synthetic/plot/printComputationSynthDataLargeFeynCV.py` | `results/run100_synthLargeCV/run100CV_FINAL.pickle` |
| Fig 3 | `epiPairs.png` * | `fig3-epipairs` | `src/synthetic/plot/printEpistatiPairsDetection.py` | `results/run100_synthLarge/run100FINAL100_withRF_PLINK2.pickle` |
| Fig (Jaccard) | `epiJaccardDomComparison.png` | `figjaccard` | `src/synthetic/plot/printEpiJaccardDomComparison.py` | same as Fig 3 |
| Fig 4 | `craterHeatmap_CV.png` | `fig4-crater` | `src/synthetic/plot/printComputationCraterDataFeynCV.py` | `results/run_Crater{Gauss,Sigmoid}CV/*_FINAL.pickle` |
| Fig 5 | `lineplots.png` * | `fig5-lineplots` | `src/synthetic/plot/printSingleCratersFeyn.py {gaussian,sigmoid}` | `results/run_Crater{Gauss,Sigmoid}/` |
| Fig GB1 | `Untitled.png` * | `figelife` | `src/elife/*` (see §4) | `results/elife/*.pickle` |
| Fig OLIDA | `jaccardHeatmap.png` | `figolida` | `src/oligogenic/make_jaccardHeatmap.py` | files in `src/oligogenic/` |

`*` = composite (hand-assembled from panels).

### Two figures not present under their LaTeX name
- `figures/Untitled.png` — GB1 composite, assembled in slides (`paper_assets/`).
- `figures/suppl_figs/craterHeatmap_Pearson_suppl.png` — the 6-method "suppl mean"
  crater panel; regenerate from `printComputationCraterDataFeynCV.py` (suppl mode).

---

## 4. GB1 / eLife pipeline (Fig "Untitled")

```bash
python src/elife/computePredsElife.py      # fit SR models  → results/elife/elifeResults_7vars_ADDMUL.pickle
python src/elife/elifeAnalysis.py          # symbolic mixed-partial derivatives → results/elife/derivativesResults.pickle
python src/elife/elifeDerivativeAnalysis.py# interaction data → results/elife/catIntPlotdata.pickle
python src/elife/mulBarchart.py            # panel: multiplication barchart
python src/elife/plotCatPairsAtoms.py      # panel: derivative heatmap
```
`derivativesResults.pickle` and `catIntPlotdata.pickle` ship precomputed; step 1's
model pickle is regenerated from `data/elife/elife-16965-supp1-v4.csv`.

---

## 5. Refitting the model bundles (slow, optional)

Only needed to recompute `results/`. Typical synthetic-benchmark chain:

```bash
python src/synthetic/fit/computePredsSynthDataLargeFeynCV.py   # Fig 2 bundle
python src/synthetic/fit/computePredsSynthDataLargeFeyn.py     # Fig 3 base bundle
python src/synthetic/fit/computeRFOnly.py                      # + RF column
python src/synthetic/fit/computePLINK2Only.py                  # + PLINK pairwise
python src/synthetic/fit/mergePLINK2Pickle.py                  # → *_withRF_PLINK2 (Fig 3 / Jaccard)
python src/synthetic/fit/computePredsSCraterFeynCV.py gaussian # Fig 4 bundles
python src/synthetic/fit/computePredsSCraterFeynCV.py sigmoid
```
`addGB*` / `addRMSE*` augment existing bundles with Gradient-Boosting and RMSE.

---

## 6. Notes
- **Dependencies**: `requirements.txt`. `feyn` is required even just to *plot*,
  because the bundles in `results/` unpickle into Feyn model objects. PLINK1.9/PLINK2
  must be on `$PATH` for the epistasis baselines.
- **Git**: the working tree's `.git` is not a valid repository; version control is
  left untouched. `.gitignore` already excludes `data/` blobs, `results/`, `archive/`,
  and any stray root images.
- **archive/** holds superseded scripts (non-CV variants, `*BUTTARE*`, KAN
  experiments), old run dirs, and intermediate PNGs — nothing was deleted.
