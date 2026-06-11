import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from recbole_cdr.quick_start.quick_start import run_recbole_cdr


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

DATASET_CONFIG_MAP = {
    "amazon": "./recbole_cdr/properties/dataset/Amazon.yaml",
    "douban": "./recbole_cdr/properties/dataset/Douban.yaml",
    "epinions": "./recbole_cdr/properties/dataset/Epinions.yaml",
}


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_result(seed, result, runtime_sec):
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
    parser.add_argument("--output_dir", type=Path, default=Path("etrc_vug_compare_results"))
    args = parser.parse_args()

    seed = args.seed
    noharm_weight = 0.1
    tgf_weight = 0.1
    output_dir = args.output_dir

    common_config = {
        "seed": seed,
        "gpu_id": args.gpu_id,
        "state": "ERROR",
        "show_progress": False,
        "save_dataset": False,
        "save_dataloaders": False,
        "delete_checkpoint_after_eval": True,
        "compute_tgf": True,
        "train_epochs": ["BOTH:300"],
        "train_batch_size": 4096,
        "eval_step": 1,
        "stopping_step": 10,
        "valid_metric": "NDCG@10",
    }

    merged_rows = []

    for dataset_slug in args.datasets:
        dataset_config = DATASET_CONFIG_MAP[dataset_slug]
        rows = []
        payload = {}
        for model_name in ("VUG", "ETRCVUG"):
            model_config = dict(common_config)
            model_config["model"] = model_name
            if model_name == "ETRCVUG":
                model_config["etrc_noharm_weight"] = noharm_weight
                model_config["etrc_tgf_weight"] = tgf_weight

            start_time = time.time()
            result = run_recbole_cdr(
                model=model_name,
                config_file_list=[
                    dataset_config,
                    "./recbole_cdr/properties/experiment/no_progress.yaml",
                    "./recbole_cdr/properties/experiment/compute_tgf.yaml",
                ],
                config_dict=model_config,
                saved=True,
            )
            runtime_sec = time.time() - start_time
            summary = summarize_result(seed, result, runtime_sec)
            row = {"model": model_name, **summary}
            rows.append(row)
            payload[model_name] = result
            merged_rows.append({"dataset": dataset_slug, "model": model_name, **summary})

        csv_path = output_dir / f"etrc_vug_{dataset_slug}_noharm_tgf_0_1_seed{seed}.csv"
        json_path = output_dir / f"etrc_vug_{dataset_slug}_noharm_tgf_0_1_seed{seed}.json"
        save_csv(csv_path, rows, fieldnames=["model", *METRIC_COLUMNS])
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

        print(
            json.dumps(
                {
                    "dataset": dataset_slug,
                    "csv_path": str(csv_path),
                    "json_path": str(json_path),
                    "rows": rows,
                },
                indent=2,
            ),
            flush=True,
        )

    if len(args.datasets) > 1:
        merged_csv_path = output_dir / f"etrc_vug_cross_dataset_summary_seed{seed}.csv"
        save_csv(merged_csv_path, merged_rows, fieldnames=SUMMARY_COLUMNS)
        print(json.dumps({"merged_csv_path": str(merged_csv_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
