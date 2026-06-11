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

from recbole_cdr.quick_start.quick_start import run_recbole_cdr


DATASET_CONFIG_MAP = {
    "amazon": "./recbole_cdr/properties/dataset/Amazon.yaml",
    "epinions": "./recbole_cdr/properties/dataset/Epinions.yaml",
}

VARIANTS = OrderedDict(
    [
        (
            "etg_full",
            {
                "model": "ETRCDTCDRLogit",
                "description": "three-state pseudo labels with evidential uncertainty",
            },
        ),
        (
            "regression_only",
            {
                "model": "ETRCDTCDRLogitReg",
                "description": "replace three-state pseudo labels with direct gain regression",
            },
        ),
        (
            "no_neutral",
            {
                "model": "ETRCDTCDRLogitBinary",
                "description": "remove the centered neutral state and keep only helpful/harmful",
            },
        ),
    ]
)

METRIC_COLUMNS = [
    "seed",
    "best_valid_score",
    "runtime_sec",
    "ndcg10",
    "ndcg20",
    "hit10",
    "hit20",
    "tgf_ndcg10",
    "tgf_ndcg20",
    "tgf_hit10",
    "tgf_hit20",
]

SUMMARY_COLUMNS = ["dataset", "variant", "model", "description", *METRIC_COLUMNS]

COMMON_CONFIG = {
    "seed": 2022,
    "gpu_id": 0,
    "state": "INFO",
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
    "etrc_noharm_weight": 0.1,
    "etrc_tgf_weight": 0.1,
    "etrc_lambda_max": 1.0,
    "etrc_lambda_base": 0.0,
    "etrc_harmful_penalty": 0.5,
    "etrc_delta_tau": 0.001,
}


def shared_split_path(shared_split_dir, dataset_slug, seed):
    return shared_split_dir / f"{dataset_slug}_repeatable_seed{seed}_fullsort_dataloaders.pth"


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
    tgf = test_result.get("tgf", {})

    return {
        "seed": seed,
        "best_valid_score": round(float(result["best_valid_score"]), 4),
        "runtime_sec": round(runtime_sec, 2),
        "ndcg10": overall.get("ndcg@10"),
        "ndcg20": overall.get("ndcg@20"),
        "hit10": overall.get("hit@10"),
        "hit20": overall.get("hit@20"),
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
        default=["amazon", "epinions"],
        choices=sorted(DATASET_CONFIG_MAP.keys()),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
    )
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--train_epochs", nargs="+", default=["BOTH:300"])
    parser.add_argument("--shared_split_dir", type=Path, default=Path("checkpoint/shared_repeatable_splits"))
    parser.add_argument("--output_stem", type=str, default="etrcdtcdr_pseudolabel_variants")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("etrc_ablation_results"),
    )
    args = parser.parse_args()

    merged_rows = []
    payload = {}

    for dataset_slug in args.datasets:
        split_path = shared_split_path(args.shared_split_dir, dataset_slug, args.seed)
        if not split_path.exists():
            raise FileNotFoundError(
                f"Shared dataloader split not found: {split_path}. "
                "Run scripts/prepare_shared_cross_domain_dataloaders.py first."
            )

        payload[dataset_slug] = {}
        for variant_name in args.variants:
            variant_spec = dict(VARIANTS[variant_name])
            model_name = variant_spec.pop("model")
            description = variant_spec.pop("description")

            config = dict(COMMON_CONFIG)
            config["seed"] = args.seed
            config["gpu_id"] = args.gpu_id
            config["train_epochs"] = args.train_epochs
            config["model"] = model_name
            config["checkpoint_dir"] = str(
                (args.output_dir / "_saved" / args.output_stem / dataset_slug / variant_name).resolve()
            )
            config["dataloaders_save_path"] = str(split_path.resolve())
            config.update(variant_spec)

            start_time = time.time()
            result = run_recbole_cdr(
                model=model_name,
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
            row = {
                "dataset": dataset_slug,
                "variant": variant_name,
                "model": model_name,
                "description": description,
                **summary,
            }
            merged_rows.append(row)
            payload[dataset_slug][variant_name] = to_builtin(result)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    merged_csv_path = args.output_dir / f"{args.output_stem}_seed{args.seed}.csv"
    merged_json_path = args.output_dir / f"{args.output_stem}_seed{args.seed}.json"
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
