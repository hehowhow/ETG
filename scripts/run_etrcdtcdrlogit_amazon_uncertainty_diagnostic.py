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
import torch.nn.functional as F
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

COMMON_CONFIG = {
    "seed": 2022,
    "gpu_id": 0,
    "state": "INFO",
    "show_progress": False,
    "save_dataset": False,
    "save_dataloaders": False,
    "delete_checkpoint_after_eval": False,
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

BUCKET_NAMES = ["low", "mid", "high"]
SUMMARY_COLUMNS = [
    "bucket",
    "count",
    "uncertainty_min",
    "uncertainty_max",
    "uncertainty_mean",
    "state_accuracy",
    "gate_accuracy_non_neutral",
    "mean_transfer_gain",
    "helpful_ratio",
    "neutral_ratio",
    "harmful_ratio",
    "mean_gate",
    "mean_reliability",
]


def shared_split_path(shared_split_dir, dataset_slug, seed):
    return shared_split_dir / f"{dataset_slug}_repeatable_seed{seed}_fullsort_dataloaders.pth"


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_builtin(value):
    if isinstance(value, (dict, OrderedDict)):
        return {key: to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def fit_and_eval(config_dict, dataset_slug):
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

    return {
        "config": config,
        "model": trainer.model,
        "trainer": trainer,
        "test_data": test_data,
        "best_valid_score": float(best_valid_score),
        "best_valid_result": best_valid_result,
        "test_result": test_result,
        "train_time_sec": train_time_sec,
        "eval_time_sec": eval_time_sec,
    }


def collect_positive_pair_diagnostics(model, test_data):
    device = next(model.parameters()).device
    tau = float(model.etrc_delta_tau)
    rows = []

    model.eval()
    with torch.no_grad():
        for interaction, _, positive_u, positive_i in test_data:
            batch_users = interaction[model.TARGET_USER_ID][positive_u].to(device)
            batch_items = positive_i.to(device)
            labels = torch.ones(batch_users.size(0), device=device)

            transfer_logits = model.target_transfer_score(batch_users, batch_items)
            target_only_logits = model.target_only_score(batch_users, batch_items)
            state_output, probs, uncertainty, reliability = model.transfer_evidence(
                transfer_logits, target_only_logits
            )
            gate = model._gate_from_probs(probs, reliability)

            target_only_loss_each = F.binary_cross_entropy_with_logits(
                target_only_logits, labels, reduction="none"
            )
            transfer_loss_each = F.binary_cross_entropy_with_logits(
                transfer_logits, labels, reduction="none"
            )
            delta = target_only_loss_each - transfer_loss_each
            oracle_state = model._make_transfer_labels(delta)
            pred_state = probs.argmax(dim=-1)

            non_neutral_mask = oracle_state != model.NEUTRAL
            gate_binary = gate >= 0.5
            gate_correct = torch.zeros_like(non_neutral_mask, dtype=torch.float)
            gate_correct[non_neutral_mask] = (
                gate_binary[non_neutral_mask]
                == (oracle_state[non_neutral_mask] == model.HELPFUL)
            ).float()

            overlap_mask = model._overlap_mask(batch_users, batch_items)

            for i in range(batch_users.size(0)):
                rows.append(
                    {
                        "uncertainty": float(uncertainty[i].item()),
                        "reliability": float(reliability[i].item()),
                        "gate": float(gate[i].item()),
                        "delta": float(delta[i].item()),
                        "oracle_state": int(oracle_state[i].item()),
                        "pred_state": int(pred_state[i].item()),
                        "state_correct": float((pred_state[i] == oracle_state[i]).item()),
                        "non_neutral": bool(non_neutral_mask[i].item()),
                        "gate_correct": float(gate_correct[i].item()),
                        "overlap": bool(overlap_mask[i].item()),
                    }
                )

    if not rows:
        raise RuntimeError("No positive test pairs were collected for uncertainty diagnostic.")

    return rows, tau


def bucketize_rows(rows):
    uncertainties = np.array([row["uncertainty"] for row in rows], dtype=float)
    q1, q2 = np.quantile(uncertainties, [1.0 / 3.0, 2.0 / 3.0])

    bucketed = {name: [] for name in BUCKET_NAMES}
    for row in rows:
        u = row["uncertainty"]
        if u <= q1:
            bucketed["low"].append(row)
        elif u <= q2:
            bucketed["mid"].append(row)
        else:
            bucketed["high"].append(row)
    return bucketed, {"q1": float(q1), "q2": float(q2)}


def summarize_bucket(rows, bucket_name):
    uncertainty = np.array([row["uncertainty"] for row in rows], dtype=float)
    delta = np.array([row["delta"] for row in rows], dtype=float)
    reliability = np.array([row["reliability"] for row in rows], dtype=float)
    gate = np.array([row["gate"] for row in rows], dtype=float)
    oracle_state = np.array([row["oracle_state"] for row in rows], dtype=int)
    state_correct = np.array([row["state_correct"] for row in rows], dtype=float)
    non_neutral = np.array([row["non_neutral"] for row in rows], dtype=bool)
    gate_correct = np.array([row["gate_correct"] for row in rows], dtype=float)

    if non_neutral.any():
        gate_acc = float(gate_correct[non_neutral].mean())
    else:
        gate_acc = float("nan")

    return {
        "bucket": bucket_name,
        "count": int(len(rows)),
        "uncertainty_min": float(uncertainty.min()),
        "uncertainty_max": float(uncertainty.max()),
        "uncertainty_mean": float(uncertainty.mean()),
        "state_accuracy": float(state_correct.mean()),
        "gate_accuracy_non_neutral": gate_acc,
        "mean_transfer_gain": float(delta.mean()),
        "helpful_ratio": float((oracle_state == 0).mean()),
        "neutral_ratio": float((oracle_state == 1).mean()),
        "harmful_ratio": float((oracle_state == 2).mean()),
        "mean_gate": float(gate.mean()),
        "mean_reliability": float(reliability.mean()),
    }


def build_markdown_report(metadata, summary_rows, thresholds):
    lines = [
        "# DTCDR / Amazon Uncertainty Diagnostic",
        "",
        f"- Date: `{metadata['date']}`",
        f"- Dataset: `{metadata['dataset']}`",
        f"- Seed: `{metadata['seed']}`",
        f"- Split: `{metadata['split_path']}`",
        f"- Best Valid Score: `{metadata['best_valid_score']:.4f}`",
        f"- Train Time: `{metadata['train_time_sec']:.2f}s`",
        f"- Eval Time: `{metadata['eval_time_sec']:.2f}s`",
        f"- Tertile thresholds: `q1={thresholds['q1']:.6f}`, `q2={thresholds['q2']:.6f}`",
        "",
        "## Bucket Summary",
        "",
        "| Bucket | Count | U Mean | U Range | State Acc | Gate Acc (Non-Neutral) | Mean TG | Helpful Ratio | Neutral Ratio | Harmful Ratio | Mean Gate | Mean Reliability |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        u_range = f"[{row['uncertainty_min']:.4f}, {row['uncertainty_max']:.4f}]"
        gate_acc = "nan" if np.isnan(row["gate_accuracy_non_neutral"]) else f"{row['gate_accuracy_non_neutral']:.4f}"
        lines.append(
            f"| `{row['bucket']}` | {row['count']} | {row['uncertainty_mean']:.4f} | {u_range} | "
            f"{row['state_accuracy']:.4f} | {gate_acc} | {row['mean_transfer_gain']:.6f} | "
            f"{row['helpful_ratio']:.4f} | {row['neutral_ratio']:.4f} | {row['harmful_ratio']:.4f} | "
            f"{row['mean_gate']:.4f} | {row['mean_reliability']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Diagnostic is computed on held-out positive target interactions from the full-sort test split.",
            "- `Mean TG` uses the same loss-space transfer gain as the main ETG formulation: `BCE(target-only) - BCE(transfer-on)`.",
            "- `State Acc` compares `argmax(probs)` against the oracle `helpful / neutral / harmful` label induced by `tau`.",
            "- `Gate Acc (Non-Neutral)` evaluates whether `gate >= 0.5` agrees with the helpful-vs-harmful oracle on non-neutral pairs only.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
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
        default=Path("etrc_uncertainty_results"),
    )
    parser.add_argument(
        "--output_stem",
        type=str,
        default="etrcdtcdrlogit_amazon_uncertainty_diagnostic",
    )
    args = parser.parse_args()

    split_path = shared_split_path(args.shared_split_dir, args.dataset, args.seed)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Shared dataloader split not found: {split_path}. "
            "Run scripts/prepare_shared_cross_domain_dataloaders.py first."
        )

    config = dict(COMMON_CONFIG)
    config["seed"] = args.seed
    config["gpu_id"] = args.gpu_id
    config["train_epochs"] = args.train_epochs
    config["dataloaders_save_path"] = str(split_path.resolve())
    config["checkpoint_dir"] = str((args.output_dir / "_saved" / args.output_stem).resolve())

    result = fit_and_eval(config, args.dataset)
    pair_rows, tau = collect_positive_pair_diagnostics(result["model"], result["test_data"])
    bucketed, thresholds = bucketize_rows(pair_rows)

    summary_rows = [summarize_bucket(bucketed[name], name) for name in BUCKET_NAMES]
    metadata = {
        "date": time.strftime("%Y-%m-%d"),
        "dataset": args.dataset,
        "seed": args.seed,
        "split_path": str(split_path),
        "best_valid_score": result["best_valid_score"],
        "train_time_sec": result["train_time_sec"],
        "eval_time_sec": result["eval_time_sec"],
        "tau": tau,
        "pair_count": len(pair_rows),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.output_stem}_seed{args.seed}.csv"
    json_path = args.output_dir / f"{args.output_stem}_seed{args.seed}.json"
    md_path = args.output_dir / f"{args.output_stem}_seed{args.seed}.md"

    save_csv(csv_path, summary_rows, SUMMARY_COLUMNS)
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "metadata": to_builtin(metadata),
                "thresholds": to_builtin(thresholds),
                "summary_rows": to_builtin(summary_rows),
            },
            fp,
            indent=2,
            ensure_ascii=False,
        )
    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write(build_markdown_report(metadata, summary_rows, thresholds))

    if os.path.exists(result["trainer"].saved_model_file):
        os.remove(result["trainer"].saved_model_file)

    print(
        json.dumps(
            {
                "csv_path": str(csv_path),
                "json_path": str(json_path),
                "md_path": str(md_path),
                "pair_count": len(pair_rows),
                "best_valid_score": result["best_valid_score"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
