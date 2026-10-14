# DTCDR / Amazon Uncertainty Diagnostic

- Date: `2026-05-20`
- Dataset: `amazon`
- Seed: `2022`
- Split: `checkpoint/shared_repeatable_splits/amazon_repeatable_seed2022_fullsort_dataloaders.pth`
- Best Valid Score: `0.0150`
- Train Time: `224.92s`
- Eval Time: `3.20s`
- Tertile thresholds: `q1=0.846542`, `q2=0.865435`

## Bucket Summary

| Bucket | Count | U Mean | U Range | State Acc | Gate Acc (Non-Neutral) | Mean TG | Helpful Ratio | Neutral Ratio | Harmful Ratio | Mean Gate | Mean Reliability |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `low` | 6697 | 0.8247 | [0.7079, 0.8465] | 0.3760 | 0.6892 | -0.661374 | 0.3106 | 0.0007 | 0.6887 | 0.3193 | 0.0018 |
| `mid` | 6696 | 0.8557 | [0.8466, 0.8654] | 0.8674 | 0.4400 | -0.042667 | 0.5596 | 0.0007 | 0.4397 | 0.3245 | 0.0016 |
| `high` | 6696 | 0.8942 | [0.8654, 1.0000] | 0.7116 | 0.4753 | 0.043623 | 0.5158 | 0.0169 | 0.4673 | 0.3253 | 0.0003 |

## Notes

- Diagnostic is computed on held-out positive target interactions from the full-sort test split.
- `Mean TG` uses the same loss-space transfer gain as the main ETG formulation: `BCE(target-only) - BCE(transfer-on)`.
- `State Acc` compares `argmax(probs)` against the oracle `helpful / neutral / harmful` label induced by `tau`.
- `Gate Acc (Non-Neutral)` evaluates whether `gate >= 0.5` agrees with the helpful-vs-harmful oracle on non-neutral pairs only.
