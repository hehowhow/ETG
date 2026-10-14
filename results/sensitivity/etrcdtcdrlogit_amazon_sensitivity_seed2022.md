# DTCDR / Amazon Hyperparameter Sensitivity

- Date: `2026-06-11`
- Dataset: `amazon`
- Seed: `2022`
- Split: `checkpoint/shared_repeatable_splits/amazon_repeatable_seed2022_fullsort_dataloaders.pth`
- Model: `ETRCDTCDRLogit`
- Plot metric pairing: `NDCG@10` and `TGF(NDCG)@10`

## $\lambda_{edl}$ (`etrc_edl_weight`)

![etrc_edl_weight](etrcdtcdrlogit_amazon_sensitivity_etrc_edl_weight_seed2022.png)

| Value | NDCG@10 | TGF(NDCG)@10 | NDCG@20 | HR@10 | HR@20 | Runtime(s) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0146 | 0.0025 | 0.0202 | 0.0462 | 0.0796 | 209.89 |
| 0.25 | 0.0141 | 0.0006 | 0.0190 | 0.0455 | 0.0764 | 174.83 |
| 0.5 | 0.0136 | 0.0003 | 0.0187 | 0.0444 | 0.0761 | 154.98 |
| 0.75 | 0.0125 | 0.0002 | 0.0175 | 0.0413 | 0.0723 | 151.85 |
| 1.0 | 0.0140 | 0.0017 | 0.0192 | 0.0433 | 0.0763 | 162.18 |

## $\lambda_{tgf}$ (`etrc_tgf_weight`)

![etrc_tgf_weight](etrcdtcdrlogit_amazon_sensitivity_etrc_tgf_weight_seed2022.png)

| Value | NDCG@10 | TGF(NDCG)@10 | NDCG@20 | HR@10 | HR@20 | Runtime(s) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0134 | 0.0018 | 0.0188 | 0.0443 | 0.0786 | 174.71 |
| 0.25 | 0.0130 | 0.0004 | 0.0184 | 0.0425 | 0.0750 | 156.77 |
| 0.5 | 0.0136 | 0.0026 | 0.0191 | 0.0447 | 0.0790 | 205.24 |
| 0.75 | 0.0134 | 0.0010 | 0.0184 | 0.0444 | 0.0756 | 133.88 |
| 1.0 | 0.0143 | 0.0001 | 0.0192 | 0.0454 | 0.0754 | 144.36 |

## $\gamma$ (`etrc_harmful_penalty`)

![etrc_harmful_penalty](etrcdtcdrlogit_amazon_sensitivity_etrc_harmful_penalty_seed2022.png)

| Value | NDCG@10 | TGF(NDCG)@10 | NDCG@20 | HR@10 | HR@20 | Runtime(s) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0123 | 0.0031 | 0.0176 | 0.0397 | 0.0733 | 163.67 |
| 0.25 | 0.0138 | 0.0003 | 0.0187 | 0.0463 | 0.0761 | 146.02 |
| **0.5** | 0.0158 | 0.0004 | 0.0217 | 0.0483 | 0.0839 | 208.99 |
| 0.75 | 0.0125 | 0.0008 | 0.0181 | 0.0394 | 0.0738 | 173.75 |
| 1.0 | 0.0134 | 0.0001 | 0.0181 | 0.0447 | 0.0739 | 213.03 |

## $\tau$ (`etrc_delta_tau`)

![etrc_delta_tau](etrcdtcdrlogit_amazon_sensitivity_etrc_delta_tau_seed2022.png)

| Value | NDCG@10 | TGF(NDCG)@10 | NDCG@20 | HR@10 | HR@20 | Runtime(s) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0005 | 0.0143 | 0.0002 | 0.0198 | 0.0464 | 0.0794 | 160.13 |
| **0.001** | 0.0158 | 0.0004 | 0.0217 | 0.0483 | 0.0839 | 212.00 |
| 0.002 | 0.0141 | 0.0017 | 0.0194 | 0.0446 | 0.0778 | 181.97 |
| 0.005 | 0.0136 | 0.0010 | 0.0188 | 0.0447 | 0.0768 | 145.11 |
| 0.01 | 0.0136 | 0.0011 | 0.0184 | 0.0434 | 0.0716 | 169.45 |
