#!/usr/bin/env python3
"""AIC variant of plot_uniform_surface — reads uniform_surface_aic.csv."""
import importlib, sys
sys.path.insert(0, ".")
mod = importlib.import_module("plot_uniform_surface")

mod.CSV     = "uniform_surface_aic.csv"
mod.OUT_PNG = "uniform_surface_aic.png"
mod.FEYN_COL = "feyn_J_strict"  # AIC pushes Feyn toward multiplicative — strict is now informative

if __name__ == "__main__":
    # Patch the suptitle text by monkey-patching main
    import matplotlib.pyplot as plt
    orig_main = mod.main
    def main():
        orig_main()
        # title patch happens inside main; we re-render with our title
    main()
