import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        nargs="+",
        type=Path,
        default=[
            Path("etrc_backbone_compare_results/conettgf_cross_dataset_repeatable_seed2022.csv"),
            Path("etrc_backbone_compare_results/etrcconetlogit_cross_dataset_repeatable_seed2022.csv"),
        ],
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("etrc_backbone_compare_results/conettgf_vs_etrcconetlogit_repeatable_seed2022.csv"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["amazon", "douban"],
    )
    args = parser.parse_args()

    rows = []
    for path in args.input_csv:
        with open(path, newline="", encoding="utf-8") as fp:
            rows.extend(list(csv.DictReader(fp)))

    dataset_filter = set(args.datasets)
    rows = [row for row in rows if row["dataset"] in dataset_filter]
    rows.sort(key=lambda row: (row["dataset"], row["model"]))

    with open(args.output_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(args.output_csv)


if __name__ == "__main__":
    main()
