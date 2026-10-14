# Paper figures

This directory contains the figure files referenced by the current ETG manuscript. The empirical plots can be regenerated from the committed files under `results/`:

```bash
python scripts/plot_etg_intro_single_panel_variants.py
python scripts/plot_etg_bar_charts.py
python scripts/run_etrcdtcdrlogit_amazon_sensitivity.py --replot_only
python scripts/plot_epinions_20e_train_time_stacked.py
```

Additional exploratory figure generators are also available in `scripts/`. The corrected overview figure, `ETG_main_figure_ai_v4_corrected.pdf`, is the manuscript asset; the two academic framework drawing scripts generate earlier editable alternatives rather than that exact AI-assisted version.
