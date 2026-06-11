import csv
from pathlib import Path

METRIC_COLUMNS = [
    "seed",
    "best_valid_score",
    "runtime_sec",
    "ndcg10",
    "ndcg20",
    "hit10",
    "hit20",
    "overlap_ndcg10",
    "overlap_ndcg20",
    "non_overlap_ndcg10",
    "non_overlap_ndcg20",
    "ugf_ndcg10",
    "ugf_ndcg20",
    "ugf_hit10",
    "ugf_hit20",
    "tgf_ndcg10",
    "tgf_ndcg20",
    "tgf_hit10",
    "tgf_hit20",
]

SUMMARY_COLUMNS = ["dataset", "model", *METRIC_COLUMNS]

ROOT = Path("etrc_backbone_compare_results")
INPUTS = [
    ROOT / "mfgslaetgf_cross_dataset_repeatable_seed2022.csv",
    ROOT / "etrcmfgslae_userlevel_cross_dataset_repeatable_seed2022.csv",
]
OUTPUT = ROOT / "mfgslaetgf_vs_etrcmfgslae_repeatable_seed2022.csv"

rows = []
for path in INPUTS:
    with open(path, newline="", encoding="utf-8") as fp:
        rows.extend(list(csv.DictReader(fp)))

rows.sort(key=lambda row: (row["dataset"], row["model"]))

with open(OUTPUT, "w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=SUMMARY_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

print(OUTPUT)
