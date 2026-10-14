# ETG

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

**ETG** is the code repository for our fairness-aware cross-domain recommendation framework based on evidential transfer gating. The repository contains the core `recbole_cdr` codebase, the processed benchmark datasets used in our experiments, reusable shared splits, and the main training and analysis scripts.

## What Is Included

- `recbole_cdr/`: the main RecBole-CDR-based implementation, including backbone models and ETG variants.
- `recbole_cdr/dataset/`: processed Amazon, Douban, and Epinions benchmark files used by the current project.
- `scripts/prepare_shared_cross_domain_dataloaders.py`: generates fixed shared dataloader splits locally under the ignored `checkpoint/` directory.
- `scripts/`: experiment runners, analysis scripts, and figure-generation utilities.
- `results/`: compact CSV/JSON artifacts used by the manuscript tables and empirical figures.
- `figures/`: the figure assets referenced by the current manuscript.
- `tests/`: lightweight smoke tests.

## Requirements

Install the dependencies listed in [requirements.txt](./requirements.txt). The project was developed around:

```text
recbole==1.0.1
torch>=1.7.0
python>=3.7.0
```

## Quick Start

The general entry point is:

```bash
python run_recbole_cdr.py --model MODEL_NAME --config_files "CONFIG_1 CONFIG_2 ..."
```

Example:

```bash
python run_recbole_cdr.py \
  --model ETRCDTCDRLogit \
  --config_files "./recbole_cdr/properties/dataset/Amazon.yaml ./recbole_cdr/properties/model/ETRCDTCDRLogit.yaml"
```

For repeatable cross-dataset experiments, use the scripts in [scripts](./scripts), for example:

```bash
python scripts/run_etrcdtcdrlogit_cross_dataset_repeatable.py
```

The manuscript plots can be regenerated from the committed result artifacts. For example:

```bash
python scripts/plot_etg_intro_single_panel_variants.py
python scripts/plot_etg_bar_charts.py
python scripts/run_etrcdtcdrlogit_amazon_sensitivity.py --replot_only
python scripts/plot_epinions_20e_train_time_stacked.py
```

These scripts use repository-relative paths and write their outputs to `figures/`.

## Dataset Paths

The dataset configuration files already use repository-relative paths such as:

```yaml
source_domain:
  data_path: './recbole_cdr/dataset'

target_domain:
  data_path: './recbole_cdr/dataset'
```

So after cloning the repository, no machine-specific dataset path replacement should be needed.

## Key Implementation Files

- `recbole_cdr/model/cross_domain_recommender/etrc_modules.py`: shared ETG transfer-state heads, evidential utilities, and gating-related components.
- `recbole_cdr/model/cross_domain_recommender/etrcdtcdrlogit.py`: one representative ETG-enhanced backbone implementation.
- `recbole_cdr/model/cross_domain_recommender/etrcclfmlogit.py`: ETG integration on a matrix-factorization-style backbone.
- `recbole_cdr/properties/model/`: model-specific configuration files.
- `recbole_cdr/properties/dataset/`: dataset-specific experiment settings.

## Naming Note

Some internal classes, scripts, and configuration files still use the historical `ETRC*` prefix. These are the current ETG implementations kept as-is for compatibility with the existing experiment pipeline.

## Citation

Citation information for the ETG paper can be added here once the manuscript is finalized or published.

## Acknowledgement

This repository builds on the open-source recommendation framework [RecBole-CDR](https://github.com/RUCAIBox/RecBole-CDR).
