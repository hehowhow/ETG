import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from recbole_cdr.quick_start import objective_function


CONFIG_FILES = [
    "./recbole_cdr/properties/dataset/Epinions.yaml",
    "./recbole_cdr/properties/experiment/no_progress.yaml",
    "./recbole_cdr/properties/experiment/compute_tgf.yaml",
]

COMMON_CONFIG = {
    "seed": 2022,
    "gpu_id": 0,
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

MODEL_SPECS = [
    ("MFGSLAE", {}),
    ("MFGSLAETGF", {}),
    ("ETRCMFGSLAE", {"etrc_noharm_weight": 0.1, "etrc_tgf_weight": 0.1}),
]


def to_builtin(value):
    if isinstance(value, (dict, OrderedDict)):
        return {key: to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize(result, runtime_sec):
    test_result = result["test_result"]
    return to_builtin(
        {
            "runtime_sec": round(runtime_sec, 2),
            "best_valid_score": result["best_valid_score"],
            "overall": test_result["overall"],
            "overlapped": test_result["overlapped"],
            "non_overlapped": test_result["non_overlapped"],
            "ugf": test_result["ugf"],
            "target_only_overall": test_result.get("target_only", {}).get("overall"),
            "transfer_gain": test_result.get("transfer_gain"),
            "tgf": test_result.get("tgf"),
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument(
        "--output",
        default="etrc_backbone_compare_results/epinions_full_targetonly_trial_seed2022.json",
    )
    args = parser.parse_args()

    summary = {}
    for model, extra_config in MODEL_SPECS:
        config = dict(COMMON_CONFIG)
        config["seed"] = args.seed
        config["model"] = model
        config.update(extra_config)

        start_time = time.time()
        result = objective_function(config_dict=config, config_file_list=CONFIG_FILES, saved=True)
        runtime_sec = time.time() - start_time
        summary[model] = summarize(result, runtime_sec)
        print(json.dumps({model: summary[model]}, ensure_ascii=False), flush=True)

    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    print(json.dumps({"output_path": args.output}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
