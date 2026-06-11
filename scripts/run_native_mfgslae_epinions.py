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


def to_builtin(value):
    if isinstance(value, (dict, OrderedDict)):
        return {key: to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    config = {
        "model": "MFGSLAE",
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
    }
    start_time = time.time()
    result = objective_function(
        config_dict=config,
        config_file_list=[
            "./recbole_cdr/properties/dataset/Epinions.yaml",
            "./recbole_cdr/properties/experiment/no_progress.yaml",
        ],
        saved=True,
    )
    runtime_sec = time.time() - start_time
    overall = result["test_result"]["overall"]
    summary = to_builtin(
        {
            "seed": 2022,
            "runtime_sec": round(runtime_sec, 2),
            "best_valid_score": result["best_valid_score"],
            "ndcg10": overall.get("ndcg@10"),
            "ndcg20": overall.get("ndcg@20"),
            "hit10": overall.get("hit@10"),
            "hit20": overall.get("hit@20"),
        }
    )

    output_path = Path("etrc_backbone_compare_results/native_mfgslae_epinions_seed2022.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(json.dumps({"output_path": str(output_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
