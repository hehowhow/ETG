import argparse
import csv
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from recbole_cdr.quick_start import objective_function


DATASET_CONFIG_MAP = {
    "amazon": "./recbole_cdr/properties/dataset/Amazon.yaml",
    "douban": "./recbole_cdr/properties/dataset/Douban.yaml",
    "epinions": "./recbole_cdr/properties/dataset/Epinions.yaml",
}

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

COMMON_CONFIG = {
    "seed": 2022,
    "gpu_id": 0,
    "state": "ERROR",
    "show_progress": False,
    "save_dataset": False,
    "save_dataloaders": False,
    "delete_checkpoint_after_eval": True,
    "compute_tgf": True,
    "repeatable": True,
    "train_epochs": ["BOTH:300"],
    "train_batch_size": 4096,
    "eval_step": 1,
    "stopping_step": 10,
    "valid_metric": "NDCG@10",
    "model": "CMFTGF",
}


def to_builtin(value):
    if isinstance(value, (dict, OrderedDict)):
        return {key: to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(seed, result, runtime_sec):
    test_result = result["test_result"]
    overall = test_result["overall"]
    overlapped = test_result["overlapped"]
    non_overlapped = test_result["non_overlapped"]
    ugf = test_result["ugf"]
    tgf = test_result.get("tgf", {})

    return {
        "seed": seed,
        "best_valid_score": round(float(result["best_valid_score"]), 4),
        "runtime_sec": round(runtime_sec, 2),
        "ndcg10": overall.get("ndcg@10"),
        "ndcg20": overall.get("ndcg@20"),
        "hit10": overall.get("hit@10"),
        "hit20": overall.get("hit@20"),
        "overlap_ndcg10": overlapped.get("ndcg@10"),
        "overlap_ndcg20": overlapped.get("ndcg@20"),
        "non_overlap_ndcg10": non_overlapped.get("ndcg@10"),
        "non_overlap_ndcg20": non_overlapped.get("ndcg@20"),
        "ugf_ndcg10": ugf.get("ndcg@10"),
        "ugf_ndcg20": ugf.get("ndcg@20"),
        "ugf_hit10": ugf.get("hit@10"),
        "ugf_hit20": ugf.get("hit@20"),
        "tgf_ndcg10": tgf.get("ndcg@10"),
        "tgf_ndcg20": tgf.get("ndcg@20"),
        "tgf_hit10": tgf.get("hit@10"),
        "tgf_hit20": tgf.get("hit@20"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["amazon", "douban", "epinions"],
        choices=sorted(DATASET_CONFIG_MAP.keys()),
    )
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("etrc_backbone_compare_results"),
    )
    args = parser.parse_args()

    merged_rows = []
    payload = {}

    for dataset_slug in args.datasets:
        config = dict(COMMON_CONFIG)
        config["seed"] = args.seed
        config["gpu_id"] = args.gpu_id
        start_time = time.time()
        result = objective_function(
            config_dict=config,
            config_file_list=[
                DATASET_CONFIG_MAP[dataset_slug],
                "./recbole_cdr/properties/experiment/no_progress.yaml",
                "./recbole_cdr/properties/experiment/compute_tgf.yaml",
            ],
            saved=True,
        )
        runtime_sec = time.time() - start_time
        summary = summarize(args.seed, result, runtime_sec)
        row = {"dataset": dataset_slug, "model": "CMFTGF", **summary}
        merged_rows.append(row)
        payload[dataset_slug] = to_builtin(result)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    merged_csv_path = args.output_dir / f"cmftgf_cross_dataset_repeatable_seed{args.seed}.csv"
    merged_json_path = args.output_dir / f"cmftgf_cross_dataset_repeatable_seed{args.seed}.json"
    save_csv(merged_csv_path, merged_rows, fieldnames=SUMMARY_COLUMNS)
    with open(merged_json_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "merged_csv_path": str(merged_csv_path),
                "merged_json_path": str(merged_json_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
