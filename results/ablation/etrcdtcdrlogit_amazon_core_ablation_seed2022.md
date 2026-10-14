# ETG Core Ablation Results on DTCDR / Amazon

- Date: `2026-05-19`
- Seed: `2022`
- Split: `checkpoint/shared_repeatable_splits/amazon_repeatable_seed2022_fullsort_dataloaders.pth`
- Model: `ETRCDTCDRLogit`
- Training setup: same as the repeatable backbone run (`BOTH:300`, fixed split, `train_batch_size=4096`)

## Metrics

| Variant | Description | NDCG@10 | NDCG@20 | HR@10 | HR@20 | TGF(NDCG)@10 | TGF(NDCG)@20 | TGF(HR)@10 | TGF(HR)@20 | Train Time (s) | Runtime (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `V0_full_etg` | 完整 ETG | 0.0158 | 0.0217 | 0.0483 | 0.0839 | 0.0004 | 0.0015 | 0.0002 | 0.0038 | 204.00 | 207.15 |
| `V1_no_uncertainty` | 去掉 gate 中的 `(1-U)` | 0.0142 | 0.0191 | 0.0471 | 0.0779 | 0.0012 | 0.0003 | 0.0101 | 0.0003 | 177.94 | 180.92 |
| `V2_no_tgf_loss` | 关闭 TGF loss | 0.0134 | 0.0188 | 0.0443 | 0.0786 | 0.0018 | 0.0018 | 0.0050 | 0.0077 | 222.55 | 225.67 |
| `V3_plain_softmax_head` | evidential head 改为 plain softmax head | 0.0122 | 0.0169 | 0.0386 | 0.0687 | 0.0000 | 0.0009 | 0.0003 | 0.0060 | 242.48 | 245.43 |
| `V4_transfer_only_features` | 可靠性特征只保留 transfer-side 信息 | 0.0143 | 0.0201 | 0.0444 | 0.0792 | 0.0039 | 0.0028 | 0.0169 | 0.0140 | 203.68 | 206.91 |

## Quick Read

- `V0_full_etg` is the strongest overall setting on accuracy and keeps very low `TGF(HR)@10`.
- Removing uncertainty (`V1`) hurts both accuracy and fairness stability, especially `TGF(HR)@10`.
- Removing the fairness objective (`V2`) hurts both accuracy and `TGF`.
- Replacing the evidential head with plain softmax (`V3`) causes the largest accuracy drop.
- Removing target-only information from reliability features (`V4`) keeps moderate accuracy but sharply worsens `TGF`, especially on `HR`.

## Files

- CSV: [etrcdtcdrlogit_amazon_core_ablation_seed2022.csv](./etrcdtcdrlogit_amazon_core_ablation_seed2022.csv)
- JSON: [etrcdtcdrlogit_amazon_core_ablation_seed2022.json](./etrcdtcdrlogit_amazon_core_ablation_seed2022.json)
