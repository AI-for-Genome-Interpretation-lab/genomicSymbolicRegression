# =============================================================================
# Makefile - regenerate the paper figures into figures/
#
# RUN FROM THE REPOSITORY ROOT:  `make all`
#
# How it works
#   * Plot scripts write every PNG into a scratch dir ($(BUILD), default
#     figures/_build) via the FIGOUT env var - never into the repo root.
#   * Each target then installs ONLY the paper-named files into figures/
#     (main) or figures/suppl_figs/ (supplement), and wipes the scratch dir.
#   * Result: the repo root stays clean and figures/ holds exactly the paper
#     figures.
#
# "direct"    = produced ready-to-use by one script (handled by `make all`).
# "composite" = final figure is assembled by hand from sub-panels; those
#               targets only (re)generate the panels into $(BUILD).
#
# Model fitting (src/synthetic/fit, src/elife/computePredsElife) is NOT run
# here: it takes hours and its outputs ship under results/.
# =============================================================================

PY    := python
PLOT  := src/synthetic/plot
ELIFE := src/elife
OLIDA := src/oligogenic
FIG   := figures
SUP   := figures/suppl_figs
BUILD := figures/_build

export FIGOUT := $(BUILD)

.PHONY: all clean clean-figs \
        fig2-synth fig4-crater figjaccard figolida supp-benchmark \
        fig1-toy fig3-epipairs fig5-lineplots figelife

# ---- direct figures --------------------------------------------------------
all: fig2-synth fig4-crater figjaccard figolida supp-benchmark
	@rm -rf $(BUILD)
	@echo ""
	@echo ">> Direct figures written to $(FIG)/ and $(SUP)/."
	@echo ">> Composite figures (toyPheno1, epiPairs, lineplots, Untitled) are"
	@echo "   assembled by hand - regenerate their panels with:"
	@echo "       make fig1-toy fig3-epipairs fig5-lineplots figelife"

# Fig 2 (main) + synthetic supplement
fig2-synth:
	@mkdir -p $(FIG) $(SUP)
	$(PY) $(PLOT)/printComputationSynthDataLargeFeynCV.py
	mv -f $(BUILD)/qtl_main_MetricPearson_CV.png $(FIG)/
	mv -f $(BUILD)/qtl_suppl_MetricPearson_CV.png     $(SUP)/
	mv -f $(BUILD)/qtl_suppl_MetricPearson_CV_std.png $(SUP)/
	mv -f $(BUILD)/qtl_suppl_MetricR2_CV.png          $(SUP)/
	mv -f $(BUILD)/qtl_suppl_MetricR2_CV_std.png      $(SUP)/
	mv -f $(BUILD)/qtl_main_MetricRMSE_CV.png         $(SUP)/
	mv -f $(BUILD)/qtl_main_MetricRMSE_CV_std.png     $(SUP)/
	rm -rf $(BUILD)

# Fig 4 (main) + crater supplement
fig4-crater:
	@mkdir -p $(FIG) $(SUP)
	$(PY) $(PLOT)/printComputationCraterDataFeynCV.py
	mv -f $(BUILD)/craterHeatmap_CV.png $(FIG)/
	mv -f $(BUILD)/craterHeatmap_CV_std.png    $(SUP)/
	mv -f $(BUILD)/craterHeatmap_CV_R2.png     $(SUP)/
	mv -f $(BUILD)/craterHeatmap_CV_R2_std.png $(SUP)/
	mv -f $(BUILD)/craterHeatmap_CV_RMSE.png     $(SUP)/
	mv -f $(BUILD)/craterHeatmap_CV_RMSE_std.png $(SUP)/
	rm -rf $(BUILD)

# Fig "SR vs PLINK Jaccard" (main)
figjaccard:
	@mkdir -p $(FIG)
	$(PY) $(PLOT)/printEpiJaccardDomComparison.py
	mv -f $(BUILD)/epiJaccardDomComparison.png $(FIG)/
	rm -rf $(BUILD)

# Fig OLIDA (main) - script reads local files, so run it from its own dir
figolida:
	@mkdir -p $(FIG)
	cd $(OLIDA) && $(PY) make_jaccardHeatmap.py
	mv -f $(OLIDA)/jaccardHeatmap.png $(FIG)/

# Suppl runtime benchmark
supp-benchmark:
	@mkdir -p $(SUP)
	$(PY) $(PLOT)/benchmarkTimes.py
	mv -f $(BUILD)/benchmark_times.png $(SUP)/
	rm -rf $(BUILD)

# ---- composite figures: regenerate panels into figures/_build -------------
fig1-toy:
	$(PY) $(PLOT)/printToyPhenotypes.py
	@echo ">> Panel(s) in $(BUILD)/ - assemble into $(FIG)/toyPheno1.png"

fig3-epipairs:
	$(PY) $(PLOT)/printEpistatiPairsDetection.py
	@echo ">> Panels in $(BUILD)/ - assemble into $(FIG)/epiPairs.png"

fig5-lineplots:
	$(PY) $(PLOT)/printSingleCratersFeyn.py gaussian
	$(PY) $(PLOT)/printSingleCratersFeyn.py sigmoid
	@echo ">> Panels in $(BUILD)/ - assemble into $(FIG)/lineplots.png"

figelife:
	$(PY) $(ELIFE)/elifeAnalysis.py
	$(PY) $(ELIFE)/elifeDerivativeAnalysis.py
	$(PY) $(ELIFE)/mulBarchart.py
	$(PY) $(ELIFE)/plotCatPairsAtoms.py
	@echo ">> Panels in $(BUILD)/ - assemble into $(FIG)/Untitled.png"

# ---- housekeeping ----------------------------------------------------------
clean:
	rm -rf $(BUILD)

# Danger: removes the regenerable figures (composites stay).
clean-figs:
	rm -f $(FIG)/qtl_main_MetricPearson_CV.png $(FIG)/craterHeatmap_CV.png \
	      $(FIG)/epiJaccardDomComparison.png $(FIG)/jaccardHeatmap.png \
	      $(SUP)/*.png
