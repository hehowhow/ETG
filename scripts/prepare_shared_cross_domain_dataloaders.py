import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from recbole_cdr.config import CDRConfig
from recbole_cdr.data import create_dataset, data_preparation
from recbole.utils import init_seed


DATASET_CONFIG_MAP = {
    "amazon": "./recbole_cdr/properties/dataset/Amazon.yaml",
    "douban": "./recbole_cdr/properties/dataset/Douban.yaml",
    "epinions": "./recbole_cdr/properties/dataset/Epinions.yaml",
}

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
    "model": "CoNetTGF",
}


def shared_split_path(shared_split_dir, dataset_slug, seed):
    return shared_split_dir / f"{dataset_slug}_repeatable_seed{seed}_fullsort_dataloaders.pth"


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
        "--shared_split_dir",
        type=Path,
        default=Path("checkpoint/shared_repeatable_splits"),
    )
    args = parser.parse_args()

    saved_paths = {}
    for dataset_slug in args.datasets:
        dataloader_path = shared_split_path(args.shared_split_dir, dataset_slug, args.seed)
        dataloader_path.parent.mkdir(parents=True, exist_ok=True)
        if dataloader_path.exists():
            saved_paths[dataset_slug] = str(dataloader_path)
            print(json.dumps({"dataset": dataset_slug, "dataloader_path": str(dataloader_path), "status": "exists"}))
            continue

        config_dict = dict(COMMON_CONFIG)
        config_dict["seed"] = args.seed
        config_dict["gpu_id"] = args.gpu_id
        config_dict["dataloaders_save_path"] = str(dataloader_path)

        config = CDRConfig(
            config_file_list=[
                DATASET_CONFIG_MAP[dataset_slug],
                "./recbole_cdr/properties/experiment/no_progress.yaml",
                "./recbole_cdr/properties/experiment/compute_tgf.yaml",
            ],
            config_dict=config_dict,
        )
        init_seed(config["seed"], config["reproducibility"])
        dataset = create_dataset(config)
        dataloaders = data_preparation(config, dataset)
        with open(dataloader_path, "wb") as fp:
            pickle.dump(dataloaders, fp)
        saved_paths[dataset_slug] = str(dataloader_path)
        print(json.dumps({"dataset": dataset_slug, "dataloader_path": str(dataloader_path), "status": "created"}))

    print(json.dumps({"saved_paths": saved_paths}, ensure_ascii=False))


if __name__ == "__main__":
    main()
