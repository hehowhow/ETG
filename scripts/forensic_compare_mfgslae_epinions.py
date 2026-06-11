import argparse
import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path
import sys

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from recbole.utils import init_seed
from recbole_cdr.config import CDRConfig
from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.quick_start import objective_function


CONFIG_FILES = [
    "./recbole_cdr/properties/dataset/Epinions.yaml",
    "./recbole_cdr/properties/experiment/no_progress.yaml",
]

COMMON_CONFIG = {
    "seed": 2022,
    "gpu_id": 0,
    "state": "ERROR",
    "show_progress": False,
    "save_dataset": False,
    "save_dataloaders": False,
    "delete_checkpoint_after_eval": True,
    "train_epochs": ["BOTH:300"],
    "train_batch_size": 4096,
    "eval_step": 1,
    "stopping_step": 10,
    "valid_metric": "NDCG@10",
    "repeatable": True,
}

MODEL_SPECS = [
    ("native", "MFGSLAE", False),
    ("native_compute_tgf", "MFGSLAE", True),
    ("wrapper_compute_tgf", "MFGSLAETGF", True),
]


def to_builtin(value):
    if isinstance(value, (dict, OrderedDict)):
        return {key: to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def sha256_json(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def extract_pairs(dataset):
    inter_feat = dataset.inter_feat
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    users = inter_feat[uid_field].numpy().tolist()
    items = inter_feat[iid_field].numpy().tolist()
    pairs = sorted((int(u), int(i)) for u, i in zip(users, items))
    return pairs


def summarize_pairs(pairs):
    user_counts = {}
    item_ids = set()
    for uid, iid in pairs:
        item_ids.add(iid)
        user_counts[uid] = user_counts.get(uid, 0) + 1
    per_user = sorted((uid, cnt) for uid, cnt in user_counts.items())
    counts = [cnt for _, cnt in per_user]
    return {
        "interaction_count": len(pairs),
        "unique_user_count": len(user_counts),
        "unique_item_count": len(item_ids),
        "pair_hash": sha256_json(pairs),
        "user_count_hash": sha256_json(per_user),
        "user_count_stats": {
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "mean": round(float(sum(counts) / len(counts)), 6) if counts else 0.0,
        },
    }


def summarize_eval_loader(eval_data):
    overlap_users = sorted(int(uid) for uid in (eval_data.overlap_users or []))
    uid2positive = []
    if hasattr(eval_data, "uid2positive_item"):
        for uid, items in enumerate(eval_data.uid2positive_item):
            if items is None:
                continue
            uid2positive.append((int(uid), [int(i) for i in items.tolist()]))
    return {
        "overlap_user_count": len(overlap_users),
        "overlap_user_hash": sha256_json(overlap_users),
        "uid2positive_hash": sha256_json(uid2positive),
    }


def filter_pairs_by_users(pairs, allowed_users):
    allowed = set(int(u) for u in allowed_users)
    return sorted((uid, iid) for uid, iid in pairs if uid in allowed)


def intersection_summary(left_pairs, right_pairs):
    overlap = sorted(set(left_pairs) & set(right_pairs))
    return {
        "count": len(overlap),
        "hash": sha256_json(overlap),
        "sample": overlap[:10],
    }


def build_split_report(config_dict):
    config = CDRConfig(config_dict=config_dict, config_file_list=CONFIG_FILES)
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    target_train_dataset = train_data.target_dataloader.dataset
    target_valid_dataset = valid_data.dataset
    target_test_dataset = test_data.dataset

    train_pairs = extract_pairs(target_train_dataset)
    valid_pairs = extract_pairs(target_valid_dataset)
    test_pairs = extract_pairs(target_test_dataset)

    overlap_users = sorted(int(uid) for uid in dataset.overlap_users)
    non_overlap_users = sorted(int(uid) for uid in dataset.non_overlap_users)

    return {
        "dataset_summary": {
            "num_overlap_user": int(dataset.num_overlap_user),
            "num_target_only_user": int(dataset.num_target_only_user),
            "num_overlap_item": int(dataset.num_overlap_item),
            "num_target_only_item": int(dataset.num_target_only_item),
            "overlap_user_hash": sha256_json(overlap_users),
            "non_overlap_user_hash": sha256_json(non_overlap_users),
        },
        "config_fingerprint": sha256_json({
            "model": config_dict["model"],
            "compute_tgf": config_dict.get("compute_tgf", False),
            "seed": config_dict["seed"],
            "repeatable": config_dict["repeatable"],
            "train_epochs": config_dict["train_epochs"],
            "train_batch_size": config_dict["train_batch_size"],
        }),
        "target_train": summarize_pairs(train_pairs),
        "target_valid": summarize_pairs(valid_pairs),
        "target_test": summarize_pairs(test_pairs),
        "eval_loader": summarize_eval_loader(test_data),
        "leakage_checks": {
            "train_valid": intersection_summary(train_pairs, valid_pairs),
            "train_test": intersection_summary(train_pairs, test_pairs),
            "valid_test": intersection_summary(valid_pairs, test_pairs),
            "overlap_train_valid": intersection_summary(
                filter_pairs_by_users(train_pairs, overlap_users),
                filter_pairs_by_users(valid_pairs, overlap_users),
            ),
            "overlap_train_test": intersection_summary(
                filter_pairs_by_users(train_pairs, overlap_users),
                filter_pairs_by_users(test_pairs, overlap_users),
            ),
            "overlap_valid_test": intersection_summary(
                filter_pairs_by_users(valid_pairs, overlap_users),
                filter_pairs_by_users(test_pairs, overlap_users),
            ),
            "non_overlap_train_valid": intersection_summary(
                filter_pairs_by_users(train_pairs, non_overlap_users),
                filter_pairs_by_users(valid_pairs, non_overlap_users),
            ),
            "non_overlap_train_test": intersection_summary(
                filter_pairs_by_users(train_pairs, non_overlap_users),
                filter_pairs_by_users(test_pairs, non_overlap_users),
            ),
            "non_overlap_valid_test": intersection_summary(
                filter_pairs_by_users(valid_pairs, non_overlap_users),
                filter_pairs_by_users(test_pairs, non_overlap_users),
            ),
        },
    }


def summarize_result(result, runtime_sec):
    test_result = result["test_result"]
    payload = {
        "runtime_sec": round(runtime_sec, 2),
        "best_valid_score": result["best_valid_score"],
        "overall": test_result["overall"],
        "overlapped": test_result["overlapped"],
        "non_overlapped": test_result["non_overlapped"],
        "ugf": test_result["ugf"],
    }
    if "target_only" in test_result:
        payload["target_only"] = test_result["target_only"]
        payload["transfer_gain"] = test_result["transfer_gain"]
        payload["tgf"] = test_result["tgf"]
    return to_builtin(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument(
        "--output",
        default="etrc_backbone_compare_results/mfgslae_forensic_compare_seed2022.json",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "seed": args.seed,
        "config_files": CONFIG_FILES,
        "runs": {},
    }

    reference_split = None
    for run_name, model_name, compute_tgf in MODEL_SPECS:
        config = dict(COMMON_CONFIG)
        config["seed"] = args.seed
        config["model"] = model_name
        if compute_tgf:
            config["compute_tgf"] = True

        split_report = build_split_report(config)
        run_record = {
            "model": model_name,
            "compute_tgf": compute_tgf,
            "split_report": split_report,
        }

        if reference_split is None:
            reference_split = split_report
            run_record["split_matches_reference"] = True
        else:
            comparable_keys = ["target_train", "target_valid", "target_test", "eval_loader", "leakage_checks"]
            run_record["split_matches_reference"] = all(
                split_report[key] == reference_split[key] for key in comparable_keys
            )

        if not args.split_only:
            start_time = time.time()
            result = objective_function(config_dict=config, config_file_list=CONFIG_FILES, saved=True)
            runtime_sec = time.time() - start_time
            run_record["result"] = summarize_result(result, runtime_sec)

        summary["runs"][run_name] = run_record
        print(json.dumps({run_name: run_record}, ensure_ascii=False), flush=True)

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(to_builtin(summary), fp, indent=2, ensure_ascii=False)
    print(json.dumps({"output_path": str(output_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
