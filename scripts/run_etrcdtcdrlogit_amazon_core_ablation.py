import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from recbole.utils import init_logger, init_seed

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from recbole_cdr.config import CDRConfig
from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.utils import get_model, get_trainer


DATASET_CONFIG_MAP = {
    "amazon": "./recbole_cdr/properties/dataset/Amazon.yaml",
}

VARIANTS = OrderedDict(
    [
        (
            "V0_full_etg",
            {
                "description": "完整 ETG",
            },
        ),
        (
            "V1_no_uncertainty",
            {
                "description": "去掉 gate 中的 (1-U)",
                "etrc_disable_uncertainty": True,
            },
        ),
        (
            "V2_no_tgf_loss",
            {
                "description": "关闭 TGF loss",
                "etrc_tgf_weight": 0.0,
            },
        ),
        (
            "V2b_no_edl_loss",
            {
                "description": "关闭 EDL loss，保留 evidential state estimator",
                "etrc_edl_weight": 0.0,
            },
        ),
        (
            "V3_plain_softmax_head",
            {
                "description": "evidential head 改为 plain softmax head",
                "etrc_plain_softmax_head": True,
                "etrc_disable_uncertainty": True,
            },
        ),
        (
            "V4_transfer_only_features",
            {
                "description": "可靠性特征只保留 transfer-side 信息",
                "etrc_use_target_only_feature": False,
                "etrc_use_delta_feature": False,
            },
        ),
    ]
)

SUMMARY_COLUMNS = [
    "variant",
    "description",
    "dataset",
    "model",
    "seed",
    "best_valid_score",
    "train_time_sec",
    "eval_time_sec",
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
    "model": "ETRCDTCDRLogit",
    "etrc_lambda_max": 1.0,
    "etrc_lambda_base": 0.0,
    "etrc_harmful_penalty": 0.5,
    "etrc_delta_tau": 0.001,
    "etrc_noharm_weight": 0.1,
    "etrc_tgf_weight": 0.1,
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


def summarize_row(variant_name, description, dataset_slug, seed, best_valid_score, test_result, train_time_sec, eval_time_sec):
    overall = test_result["overall"]
    tgf = test_result.get("tgf", {})
    runtime_sec = train_time_sec + eval_time_sec
    return {
        "variant": variant_name,
        "description": description,
        "dataset": dataset_slug,
        "model": "ETRCDTCDRLogit",
        "seed": seed,
        "best_valid_score": round(float(best_valid_score), 4),
        "train_time_sec": round(train_time_sec, 2),
        "eval_time_sec": round(eval_time_sec, 2),
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


def run_one(config_dict, dataset_slug):
    warnings.filterwarnings("ignore")
    config = CDRConfig(
        model=config_dict["model"],
        config_file_list=[
            DATASET_CONFIG_MAP[dataset_slug],
            "./recbole_cdr/properties/experiment/no_progress.yaml",
            "./recbole_cdr/properties/experiment/compute_tgf.yaml",
        ],
        config_dict=config_dict,
    )

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    if test_data.overlap_users is None:
        test_data.overlap_users = train_data.overlap_users

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    train_start = time.time()
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )
    train_time_sec = time.time() - train_start

    eval_start = time.time()
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config["show_progress"])
    eval_time_sec = time.time() - eval_start

    if config["delete_checkpoint_after_eval"] and os.path.exists(trainer.saved_model_file):
        os.remove(trainer.saved_model_file)

    return {
        "best_valid_score": best_valid_score,
        "best_valid_result": best_valid_result,
        "test_result": test_result,
        "train_time_sec": train_time_sec,
        "eval_time_sec": eval_time_sec,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
    )
    parser.add_argument("--dataset", choices=["amazon"], default="amazon")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--train_epochs", nargs="+", default=["BOTH:300"])
    parser.add_argument(
        "--shared_split_dir",
        type=Path,
        default=Path("checkpoint/shared_repeatable_splits"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("etrc_ablation_results"),
    )
    parser.add_argument(
        "--output_stem",
        type=str,
        default="etrcdtcdrlogit_amazon_core_ablation",
    )
    args = parser.parse_args()

    dataset_slug = args.dataset
    seed = args.seed
    split_path = shared_split_path(args.shared_split_dir, dataset_slug, seed)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Shared dataloader split not found: {split_path}. "
            "Run scripts/prepare_shared_cross_domain_dataloaders.py first."
        )

    rows = []
    payload = {}

    for variant_name in args.variants:
        variant_spec = dict(VARIANTS[variant_name])
        description = variant_spec.pop("description")

        config = dict(COMMON_CONFIG)
        config["seed"] = seed
        config["gpu_id"] = args.gpu_id
        config["train_epochs"] = args.train_epochs
        config["checkpoint_dir"] = str(
            (args.output_dir / "_saved" / args.output_stem / variant_name / dataset_slug).resolve()
        )
        config["dataloaders_save_path"] = str(split_path.resolve())
        config.update(variant_spec)

        result = run_one(config, dataset_slug)
        row = summarize_row(
            variant_name,
            description,
            dataset_slug,
            seed,
            result["best_valid_score"],
            result["test_result"],
            result["train_time_sec"],
            result["eval_time_sec"],
        )
        rows.append(row)
        payload[variant_name] = to_builtin(result)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    csv_path = args.output_dir / f"{args.output_stem}_seed{seed}.csv"
    json_path = args.output_dir / f"{args.output_stem}_seed{seed}.json"
    save_csv(csv_path, rows, fieldnames=SUMMARY_COLUMNS)
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "csv_path": str(csv_path),
                "json_path": str(json_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
