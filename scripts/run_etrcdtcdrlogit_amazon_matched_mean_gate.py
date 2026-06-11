import argparse
import csv
import json
import os
import sys
import time
import warnings
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

SUMMARY_COLUMNS = [
    "variant",
    "description",
    "dataset",
    "seed",
    "etrc_disable_uncertainty",
    "etrc_lambda_base",
    "best_valid_score",
    "train_time_sec",
    "eval_time_sec",
    "runtime_sec",
    "mean_gate",
    "gate_pair_count",
    "ndcg10",
    "ndcg20",
    "hit10",
    "hit20",
    "tgf_ndcg10",
    "tgf_ndcg20",
    "tgf_hit10",
    "tgf_hit20",
]

DEFAULT_LAMBDA_BASE_GRID = [-0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10]


def shared_split_path(shared_split_dir, dataset_slug, seed):
    return shared_split_dir / f"{dataset_slug}_repeatable_seed{seed}_fullsort_dataloaders.pth"


def to_builtin(value):
    if isinstance(value, dict):
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


def collect_mean_gate(model, test_data):
    device = next(model.parameters()).device
    gate_values = []
    model.eval()
    with torch.no_grad():
        for interaction, _, positive_u, positive_i in test_data:
            batch_users = interaction[model.TARGET_USER_ID][positive_u].to(device)
            batch_items = positive_i.to(device)
            transfer_logits = model.target_transfer_score(batch_users, batch_items)
            target_only_logits = model.target_only_score(batch_users, batch_items)
            gate, _, _ = model.transfer_gate(transfer_logits, target_only_logits)
            gate_values.append(gate.detach().cpu())

    if not gate_values:
        raise RuntimeError("No held-out positive target pairs were collected for gate matching.")

    all_gates = torch.cat(gate_values, dim=0).float()
    return float(all_gates.mean().item()), int(all_gates.numel())


def summarize_row(variant_name, description, dataset_slug, seed, config_dict, run_output, mean_gate, gate_pair_count):
    overall = run_output["test_result"]["overall"]
    tgf = run_output["test_result"].get("tgf", {})
    return {
        "variant": variant_name,
        "description": description,
        "dataset": dataset_slug,
        "seed": seed,
        "etrc_disable_uncertainty": bool(config_dict.get("etrc_disable_uncertainty", False)),
        "etrc_lambda_base": float(config_dict.get("etrc_lambda_base", 0.0)),
        "best_valid_score": round(run_output["best_valid_score"], 4),
        "train_time_sec": round(run_output["train_time_sec"], 2),
        "eval_time_sec": round(run_output["eval_time_sec"], 2),
        "runtime_sec": round(run_output["train_time_sec"] + run_output["eval_time_sec"], 2),
        "mean_gate": round(mean_gate, 6),
        "gate_pair_count": gate_pair_count,
        "ndcg10": overall.get("ndcg@10"),
        "ndcg20": overall.get("ndcg@20"),
        "hit10": overall.get("hit@10"),
        "hit20": overall.get("hit@20"),
        "tgf_ndcg10": tgf.get("ndcg@10"),
        "tgf_ndcg20": tgf.get("ndcg@20"),
        "tgf_hit10": tgf.get("hit@10"),
        "tgf_hit20": tgf.get("hit@20"),
    }


def build_markdown_report(metadata, rows, target_row, matched_row):
    lines = [
        "# DTCDR / Amazon Matched Mean Gate Fairness Test",
        "",
        f"- Date: `{metadata['date']}`",
        f"- Dataset: `{metadata['dataset']}`",
        f"- Seed: `{metadata['seed']}`",
        f"- Split: `{metadata['split_path']}`",
        f"- Matching population: `held-out positive target interactions`",
        f"- Target mean gate (Full ETG): `{target_row['mean_gate']:.6f}`",
        f"- Selected matched variant: `{matched_row['variant']}` with `lambda_base={matched_row['etrc_lambda_base']:.2f}`",
        f"- Absolute mean-gate gap: `{abs(matched_row['mean_gate'] - target_row['mean_gate']):.6f}`",
        "",
        "## Full Sweep",
        "",
        "| Variant | Lambda Base | Disable U | Mean Gate | Gate Gap | NDCG@10 | NDCG@20 | HR@10 | HR@20 | TGF(NDCG)@10 | TGF(NDCG)@20 | TGF(HR)@10 | TGF(HR)@20 | Train(s) | Runtime(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    target_mean_gate = target_row["mean_gate"]
    for row in rows:
        lines.append(
            f"| `{row['variant']}` | {row['etrc_lambda_base']:.2f} | {int(row['etrc_disable_uncertainty'])} | "
            f"{row['mean_gate']:.6f} | {abs(row['mean_gate'] - target_mean_gate):.6f} | "
            f"{row['ndcg10']:.4f} | {row['ndcg20']:.4f} | {row['hit10']:.4f} | {row['hit20']:.4f} | "
            f"{row['tgf_ndcg10']:.4f} | {row['tgf_ndcg20']:.4f} | {row['tgf_hit10']:.4f} | {row['tgf_hit20']:.4f} | "
            f"{row['train_time_sec']:.2f} | {row['runtime_sec']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Matched Comparison",
            "",
            "| Metric | Full ETG | Matched No-Uncertainty | Better |",
            "|---|---:|---:|---|",
        ]
    )

    metric_labels = [
        ("mean_gate", "Mean Gate", False),
        ("ndcg10", "NDCG@10", True),
        ("ndcg20", "NDCG@20", True),
        ("hit10", "HR@10", True),
        ("hit20", "HR@20", True),
        ("tgf_ndcg10", "TGF(NDCG)@10", False),
        ("tgf_ndcg20", "TGF(NDCG)@20", False),
        ("tgf_hit10", "TGF(HR)@10", False),
        ("tgf_hit20", "TGF(HR)@20", False),
    ]
    for key, label, higher_is_better in metric_labels:
        lhs = target_row[key]
        rhs = matched_row[key]
        if abs(lhs - rhs) < 1e-12:
            better = "Tie"
        elif (lhs > rhs and higher_is_better) or (lhs < rhs and not higher_is_better):
            better = "Full ETG"
        else:
            better = "Matched No-U"
        lines.append(f"| {label} | {lhs:.6f} | {rhs:.6f} | {better} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This test isolates whether uncertainty helps beyond simply reducing or increasing overall transfer strength.",
            "- Matching is done by minimizing the absolute gap in mean gate over the same held-out positive target pairs.",
            "- If Full ETG still achieves better fairness/accuracy under matched mean gate, the benefit is less likely to come from a trivial global gate shrinkage effect.",
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
        "--lambda_base_grid",
        type=float,
        nargs="+",
        default=DEFAULT_LAMBDA_BASE_GRID,
    )
    parser.add_argument(
        "--shared_split_dir",
        type=Path,
        default=Path("checkpoint/shared_repeatable_splits"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("etrc_gate_matching_results"),
    )
    parser.add_argument(
        "--output_stem",
        type=str,
        default="etrcdtcdrlogit_amazon_matched_mean_gate",
    )
    args = parser.parse_args()

    split_path = shared_split_path(args.shared_split_dir, args.dataset, args.seed)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Shared dataloader split not found: {split_path}. "
            "Run scripts/prepare_shared_cross_domain_dataloaders.py first."
        )

    output_root = args.output_dir
    rows = []
    payload = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "dataset": args.dataset,
            "seed": args.seed,
            "split_path": str(split_path.resolve()),
            "lambda_base_grid": args.lambda_base_grid,
        },
        "runs": {},
    }

    experiment_specs = [("full_etg", "完整 ETG", {"etrc_disable_uncertainty": False, "etrc_lambda_base": 0.0})]
    for lambda_base in args.lambda_base_grid:
        experiment_specs.append(
            (
                f"no_uncertainty_lb_{lambda_base:+.2f}".replace("+", "p").replace("-", "m"),
                f"No Uncertainty (lambda_base={lambda_base:.2f})",
                {"etrc_disable_uncertainty": True, "etrc_lambda_base": float(lambda_base)},
            )
        )

    for variant_name, description, variant_updates in experiment_specs:
        config = dict(COMMON_CONFIG)
        config["seed"] = args.seed
        config["gpu_id"] = args.gpu_id
        config["train_epochs"] = args.train_epochs
        config["dataloaders_save_path"] = str(split_path.resolve())
        config["checkpoint_dir"] = str((output_root / "_saved" / args.output_stem / variant_name).resolve())
        config.update(variant_updates)

        run_output = fit_and_eval(config, args.dataset)
        mean_gate, gate_pair_count = collect_mean_gate(run_output["model"], run_output["test_data"])

        row = summarize_row(
            variant_name,
            description,
            args.dataset,
            args.seed,
            config,
            run_output,
            mean_gate,
            gate_pair_count,
        )
        rows.append(row)
        payload["runs"][variant_name] = {
            "description": description,
            "config": to_builtin(config),
            "summary": to_builtin(row),
            "best_valid_result": to_builtin(run_output["best_valid_result"]),
            "test_result": to_builtin(run_output["test_result"]),
        }

        if config["delete_checkpoint_after_eval"] and os.path.exists(run_output["trainer"].saved_model_file):
            os.remove(run_output["trainer"].saved_model_file)

    target_row = next(row for row in rows if row["variant"] == "full_etg")
    candidates = [row for row in rows if row["variant"] != "full_etg"]
    matched_row = min(
        candidates,
        key=lambda row: (abs(row["mean_gate"] - target_row["mean_gate"]), -row["ndcg10"]),
    )

    payload["matched_summary"] = {
        "target_variant": "full_etg",
        "target_mean_gate": target_row["mean_gate"],
        "matched_variant": matched_row["variant"],
        "matched_lambda_base": matched_row["etrc_lambda_base"],
        "matched_mean_gate": matched_row["mean_gate"],
        "absolute_gate_gap": abs(matched_row["mean_gate"] - target_row["mean_gate"]),
    }

    csv_path = output_root / f"{args.output_stem}_seed{args.seed}.csv"
    json_path = output_root / f"{args.output_stem}_seed{args.seed}.json"
    md_path = output_root / f"{args.output_stem}_seed{args.seed}.md"

    save_csv(csv_path, rows, SUMMARY_COLUMNS)
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(to_builtin(payload), fp, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write(build_markdown_report(payload["metadata"], rows, target_row, matched_row))

    print(f"Saved CSV to {csv_path}")
    print(f"Saved JSON to {json_path}")
    print(f"Saved Markdown report to {md_path}")
    print(
        "Matched mean gate result:",
        json.dumps(payload["matched_summary"], ensure_ascii=False),
    )


if __name__ == "__main__":
    main()
