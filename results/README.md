# Experimental results

This directory contains the compact result artifacts used by the ETG manuscript tables and figures. The files are grouped by their role in the paper:

- `main/`: five backbone/ETG pairs on Amazon, Douban, and Epinions. The JSON files contain the detailed evaluation payloads; the CSV files are compact summaries.
- `ablation/`: the Amazon core ablation and pseudo-label-design comparison.
- `fairness/`: the VUG versus ETG-VUG summary used by the fairness comparison plots. The QB CDFA values are recorded directly in `scripts/plot_etg_bar_charts.py`.
- `sensitivity/`: the source data and report for the four Amazon sensitivity plots.
- `runtime/`: the combined 20-epoch Epinions timing measurements and parameter summary.
- `diagnostics/`: the Amazon uncertainty-bucket diagnostic reported in the appendix.

All paths used by the plotting scripts are repository-relative. Run plotting commands from the repository root. Generated paper figures are written to `figures/`.

The files intentionally exclude model checkpoints, temporary logs, TensorBoard runs, and intermediate sweep outputs that are not used by the manuscript.
