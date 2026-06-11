import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    "model": "ETRCDTCDRLogit",
    "etrc_lambda_max": 1.0,
    "etrc_lambda_base": 0.0,
    "etrc_harmful_penalty": 0.5,
    "etrc_delta_tau": 0.001,
    "etrc_noharm_weight": 0.1,
    "etrc_tgf_weight": 0.1,
}

SENSITIVITY_SPECS = OrderedDict(
    [
        (
            "etrc_edl_weight",
            {
                "label": r"$\lambda_{edl}$",
                "values": [0.00, 0.25, 0.50, 0.75, 1.00],
                "base": 0.05,
            },
        ),
        (
            "etrc_tgf_weight",
            {
                "label": r"$\lambda_{tgf}$",
                "values": [0.00, 0.25, 0.50, 0.75, 1.00],
                "base": 0.10,
            },
        ),
        (
            "etrc_gate_temperature",
            {
                "label": r"$T$",
                "values": [0.50, 0.70, 1.00, 1.50, 2.00],
                "base": 1.00,
            },
        ),
        (
            "etrc_harmful_penalty",
            {
                "label": r"$\gamma$",
                "values": [0.00, 0.25, 0.50, 0.75, 1.00],
                "base": 0.50,
            },
        ),
        (
            "etrc_delta_tau",
            {
                "label": r"$\tau$",
                "values": [0.0005, 0.0010, 0.0020, 0.0050, 0.0100],
                "base": 0.0010,
            },
        ),
    ]
)

SUMMARY_COLUMNS = [
    "parameter",
    "parameter_label",
    "value",
    "is_base_value",
    "dataset",
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


def load_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = []
        for row in reader:
            cast_row = dict(row)
            for key in [
                "value",
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
            ]:
                cast_row[key] = float(cast_row[key])
            cast_row["is_base_value"] = cast_row["is_base_value"] == "True"
            cast_row["seed"] = int(cast_row["seed"])
            rows.append(cast_row)
        return rows


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
        "trainer": trainer,
        "best_valid_score": float(best_valid_score),
        "best_valid_result": best_valid_result,
        "test_result": test_result,
        "train_time_sec": train_time_sec,
        "eval_time_sec": eval_time_sec,
    }


def summarize_row(parameter_name, label, value, base_value, dataset_slug, seed, run_output):
    overall = run_output["test_result"]["overall"]
    tgf = run_output["test_result"].get("tgf", {})
    return {
        "parameter": parameter_name,
        "parameter_label": label,
        "value": float(value),
        "is_base_value": bool(abs(float(value) - float(base_value)) < 1e-12),
        "dataset": dataset_slug,
        "seed": seed,
        "best_valid_score": round(run_output["best_valid_score"], 4),
        "train_time_sec": round(run_output["train_time_sec"], 2),
        "eval_time_sec": round(run_output["eval_time_sec"], 2),
        "runtime_sec": round(run_output["train_time_sec"] + run_output["eval_time_sec"], 2),
        "ndcg10": overall.get("ndcg@10"),
        "ndcg20": overall.get("ndcg@20"),
        "hit10": overall.get("hit@10"),
        "hit20": overall.get("hit@20"),
        "tgf_ndcg10": tgf.get("ndcg@10"),
        "tgf_ndcg20": tgf.get("ndcg@20"),
        "tgf_hit10": tgf.get("hit@10"),
        "tgf_hit20": tgf.get("hit@20"),
    }


def format_x_value(value, parameter_name=None):
    if parameter_name != "etrc_delta_tau":
        return f"{value:.2f}"
    if abs(value) >= 0.001:
        return f"{value:.4f}"
    return f"{value:.4f}"


def compute_band_limits(values, lower_band, upper_band, min_padding_ratio=0.12):
    values = np.asarray(values, dtype=float)
    v_min = float(values.min())
    v_max = float(values.max())
    span = v_max - v_min
    if span < 1e-12:
        base = max(abs(v_max), 1.0)
        span = base * min_padding_ratio
        v_min -= span / 2.0
        v_max += span / 2.0
    else:
        pad = span * min_padding_ratio
        v_min -= pad
        v_max += pad
        span = v_max - v_min

    band_width = max(upper_band - lower_band, 1e-6)
    full_range = span / band_width
    y_min = v_min - lower_band * full_range
    y_max = y_min + full_range
    return y_min, y_max


def compute_gentle_limits(ndcg, tgf):
    ndcg = np.asarray(ndcg, dtype=float)
    tgf = np.asarray(tgf, dtype=float)

    ndcg_min = float(ndcg.min())
    ndcg_max = float(ndcg.max())
    ndcg_span = max(ndcg_max - ndcg_min, max(abs(ndcg_max), 1.0) * 1e-4)
    ndcg_ylim = (
        ndcg_min - 1.25 * ndcg_span,
        ndcg_max + 0.45 * ndcg_span,
    )

    tgf_min = float(tgf.min())
    tgf_max = float(tgf.max())
    tgf_span = max(tgf_max - tgf_min, max(abs(tgf_max), 1.0) * 1e-4)
    tgf_lower = max(0.0, tgf_min - 0.15 * tgf_span)
    tgf_upper = tgf_max + 2.20 * tgf_span
    tgf_ylim = (tgf_lower, tgf_upper)
    return ndcg_ylim, tgf_ylim


def make_plot(rows, parameter_name, spec, out_path, plot_style="banded"):
    rows = sorted(rows, key=lambda x: x["value"])
    xs = [row["value"] for row in rows]
    positions = list(range(len(xs)))
    ndcg = [row["ndcg10"] for row in rows]
    tgf = [row["tgf_ndcg10"] for row in rows]
    tick_labels = [format_x_value(x, parameter_name=parameter_name) for x in xs]
    if plot_style == "gentle":
        ndcg_ylim, tgf_ylim = compute_gentle_limits(ndcg, tgf)
    else:
        ndcg_ylim = compute_band_limits(ndcg, lower_band=0.58, upper_band=0.93)
        tgf_ylim = compute_band_limits(tgf, lower_band=0.07, upper_band=0.42)

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, ax1 = plt.subplots(figsize=(8.0, 5.4))
    ax2 = ax1.twinx()

    line1 = ax1.plot(positions, ndcg, marker="o", linewidth=2.2, markersize=6.5, color="#1f77b4", label="NDCG@10")
    line2 = ax2.plot(positions, tgf, marker="s", linewidth=2.2, markersize=6.5, color="#d62728", label="TGF(NDCG)@10")

    ax1.set_xlabel(spec["label"], fontsize=16)
    ax1.set_ylabel("NDCG@10", color="#1f77b4", fontsize=15)
    ax2.set_ylabel("TGF(NDCG)@10", color="#d62728", fontsize=15)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(tick_labels, fontsize=13)
    ax1.set_xlim(-0.5, len(positions) - 0.5)
    ax1.set_ylim(*ndcg_ylim)
    ax2.set_ylim(*tgf_ylim)
    ax1.tick_params(axis="y", labelcolor="#1f77b4", labelsize=13)
    ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=13)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.set_title(spec["label"], fontsize=19, pad=12)

    handles = line1 + line2
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="best", fontsize=13)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_markdown_report(metadata, rows_by_param, plot_paths):
    lines = [
        "# DTCDR / Amazon Hyperparameter Sensitivity",
        "",
        f"- Date: `{metadata['date']}`",
        f"- Dataset: `{metadata['dataset']}`",
        f"- Seed: `{metadata['seed']}`",
        f"- Split: `{metadata['split_path']}`",
        "- Model: `ETRCDTCDRLogit`",
        "- Plot metric pairing: `NDCG@10` and `TGF(NDCG)@10`",
        "",
    ]

    for parameter_name, rows in rows_by_param.items():
        spec = SENSITIVITY_SPECS[parameter_name]
        lines.extend(
            [
                f"## {spec['label']} (`{parameter_name}`)",
                "",
                f"![{parameter_name}]({plot_paths[parameter_name].name})",
                "",
                "| Value | NDCG@10 | TGF(NDCG)@10 | NDCG@20 | HR@10 | HR@20 | Runtime(s) |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(rows, key=lambda x: x["value"]):
            value_str = f"**{row['value']}**" if row["is_base_value"] else f"{row['value']}"
            lines.append(
                f"| {value_str} | {row['ndcg10']:.4f} | {row['tgf_ndcg10']:.4f} | "
                f"{row['ndcg20']:.4f} | {row['hit10']:.4f} | {row['hit20']:.4f} | {row['runtime_sec']:.2f} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["amazon"], default="amazon")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--train_epochs", nargs="+", default=["BOTH:300"])
    parser.add_argument(
        "--parameters",
        nargs="+",
        choices=list(SENSITIVITY_SPECS.keys()),
        default=list(SENSITIVITY_SPECS.keys()),
    )
    parser.add_argument(
        "--shared_split_dir",
        type=Path,
        default=Path("checkpoint/shared_repeatable_splits"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("etrc_sensitivity_results"),
    )
    parser.add_argument(
        "--output_stem",
        type=str,
        default="etrcdtcdrlogit_amazon_sensitivity",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Optional suffix appended to plot filenames before the seed marker.",
    )
    parser.add_argument(
        "--plot_style",
        choices=["banded", "gentle"],
        default="banded",
        help="banded keeps the two lines separated; gentle adds more whitespace to reduce visual sensitivity.",
    )
    parser.add_argument(
        "--replot_only",
        action="store_true",
        help="Only regenerate plots and markdown from an existing CSV, without rerunning training.",
    )
    args = parser.parse_args()

    split_path = shared_split_path(args.shared_split_dir, args.dataset, args.seed)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Shared dataloader split not found: {split_path}. "
            "Run scripts/prepare_shared_cross_domain_dataloaders.py first."
        )

    output_root = args.output_dir
    csv_path = output_root / f"{args.output_stem}_seed{args.seed}.csv"
    json_path = output_root / f"{args.output_stem}_seed{args.seed}.json"
    report_suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    md_path = output_root / f"{args.output_stem}{report_suffix}_seed{args.seed}.md"

    rows = []
    existing_rows = []
    rows_by_param = OrderedDict()
    payload = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "dataset": args.dataset,
            "seed": args.seed,
            "split_path": str(split_path.resolve()),
            "parameters": args.parameters,
        },
        "runs": OrderedDict(),
    }

    if args.replot_only:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found for replot: {csv_path}")
        rows = load_csv(csv_path)
        for parameter_name in args.parameters:
            rows_by_param[parameter_name] = [
                row for row in rows if row["parameter"] == parameter_name
            ]
    else:
        if csv_path.exists():
            existing_rows = load_csv(csv_path)
        for parameter_name in args.parameters:
            spec = SENSITIVITY_SPECS[parameter_name]
            rows_by_param[parameter_name] = []
            payload["runs"][parameter_name] = []

            for value in spec["values"]:
                config = dict(COMMON_CONFIG)
                config["seed"] = args.seed
                config["gpu_id"] = args.gpu_id
                config["train_epochs"] = args.train_epochs
                config["dataloaders_save_path"] = str(split_path.resolve())
                config["checkpoint_dir"] = str(
                    (output_root / "_saved" / args.output_stem / parameter_name / str(value).replace(".", "p")).resolve()
                )
                config[parameter_name] = value

                run_output = fit_and_eval(config, args.dataset)
                row = summarize_row(
                    parameter_name,
                    spec["label"],
                    value,
                    spec["base"],
                    args.dataset,
                    args.seed,
                    run_output,
                )
                rows.append(row)
                rows_by_param[parameter_name].append(row)
                payload["runs"][parameter_name].append(
                    {
                        "value": value,
                        "summary": to_builtin(row),
                        "best_valid_result": to_builtin(run_output["best_valid_result"]),
                        "test_result": to_builtin(run_output["test_result"]),
                    }
                )

                if config["delete_checkpoint_after_eval"] and os.path.exists(run_output["trainer"].saved_model_file):
                    os.remove(run_output["trainer"].saved_model_file)

    plot_paths = {}
    for parameter_name, param_rows in rows_by_param.items():
        suffix = f"_{args.output_suffix}" if args.output_suffix else ""
        plot_path = output_root / f"{args.output_stem}_{parameter_name}{suffix}_seed{args.seed}.png"
        make_plot(param_rows, parameter_name, SENSITIVITY_SPECS[parameter_name], plot_path, plot_style=args.plot_style)
        plot_paths[parameter_name] = plot_path

    if not args.replot_only:
        preserved_rows = [row for row in existing_rows if row["parameter"] not in set(args.parameters)]
        merged_rows = preserved_rows + rows
        param_order = {name: idx for idx, name in enumerate(SENSITIVITY_SPECS.keys())}
        merged_rows.sort(key=lambda row: (param_order.get(row["parameter"], 999), row["value"]))
        save_csv(csv_path, merged_rows, SUMMARY_COLUMNS)
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(to_builtin(payload), fp, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write(build_markdown_report(payload["metadata"], rows_by_param, plot_paths))

    if not args.replot_only:
        print(f"Saved CSV to {csv_path}")
        print(f"Saved JSON to {json_path}")
    print(f"Saved Markdown report to {md_path}")
    for parameter_name, plot_path in plot_paths.items():
        print(f"Saved plot for {parameter_name} to {plot_path}")


if __name__ == "__main__":
    main()
