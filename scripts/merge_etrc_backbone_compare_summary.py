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


def read_rows(path, dataset):
    with open(path, "r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = []
        for row in reader:
            rows.append({"dataset": dataset, "model": row["model"], **{k: row[k] for k in METRIC_COLUMNS}})
        return rows


def main():
    output_dir = Path("etrc_backbone_compare_results")
    seed = 2022
    datasets = ["amazon", "douban", "epinions"]
    merged_rows = []
    for dataset in datasets:
        merged_rows.extend(read_rows(output_dir / f"etrc_backbones_{dataset}_seed{seed}.csv", dataset))

    out_path = output_dir / f"etrc_backbones_cross_dataset_summary_seed{seed}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["dataset", "model", *METRIC_COLUMNS])
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)

    print(out_path)


if __name__ == "__main__":
    main()
