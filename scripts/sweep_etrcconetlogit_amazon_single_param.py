import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT_DIR / "scripts" / "run_etrcconetlogit_cross_dataset_repeatable.py"

DEFAULTS = {
    "etrc_lambda_max": 1.0,
    "etrc_lambda_base": 0.0,
    "etrc_harmful_penalty": 0.5,
    "etrc_delta_tau": 0.001,
}

SWEEP_VALUES = {
    "etrc_lambda_max": [0.4, 0.6, 0.8, 1.0],
    "etrc_lambda_base": [-0.2, -0.1, 0.0, 0.05],
    "etrc_harmful_penalty": [0.25, 0.5, 0.75, 1.0],
    "etrc_delta_tau": [0.0005, 0.001, 0.002, 0.005],
}

PARAM_ORDER = [
    "etrc_lambda_max",
    "etrc_lambda_base",
    "etrc_harmful_penalty",
    "etrc_delta_tau",
]

SUMMARY_COLUMNS = [
    "dataset",
    "model",
    "param_name",
    "param_value",
    "etrc_lambda_max",
    "etrc_lambda_base",
    "etrc_harmful_penalty",
    "etrc_delta_tau",
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
    "run_csv_path",
    "run_json_path",
]


@dataclass
class Experiment:
    param_name: str
    param_value: float
    etrc_lambda_max: float
    etrc_lambda_base: float
    etrc_harmful_penalty: float
    etrc_delta_tau: float

    @property
    def output_stem(self):
        value = format(self.param_value, ".10g").replace("-", "m").replace(".", "p")
        return f"etrcconetlogit_amazon_{self.param_name}_{value}"


def build_experiments():
    experiments = []
    for param_name in PARAM_ORDER:
        for value in SWEEP_VALUES[param_name]:
            cfg = dict(DEFAULTS)
            cfg[param_name] = value
            experiments.append(
                Experiment(
                    param_name=param_name,
                    param_value=value,
                    etrc_lambda_max=cfg["etrc_lambda_max"],
                    etrc_lambda_base=cfg["etrc_lambda_base"],
                    etrc_harmful_penalty=cfg["etrc_harmful_penalty"],
                    etrc_delta_tau=cfg["etrc_delta_tau"],
                )
            )
    return experiments


def run_one(exp: Experiment, gpu_id: int, seed: int, output_dir: Path, shared_split_dir: Path):
    cmd = [
        "conda",
        "run",
        "-n",
        "AHBF",
        "python",
        str(RUN_SCRIPT),
        "--datasets",
        "amazon",
        "--seed",
        str(seed),
        "--gpu_id",
        str(gpu_id),
        "--shared_split_dir",
        str(shared_split_dir),
        "--output_dir",
        str(output_dir),
        "--output_stem",
        exp.output_stem,
        "--etrc_lambda_max",
        str(exp.etrc_lambda_max),
        "--etrc_lambda_base",
        str(exp.etrc_lambda_base),
        "--etrc_harmful_penalty",
        str(exp.etrc_harmful_penalty),
        "--etrc_delta_tau",
        str(exp.etrc_delta_tau),
    ]
    completed = subprocess.run(
        cmd,
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        check=True,
    )

    run_csv_path = output_dir / f"{exp.output_stem}_seed{seed}.csv"
    run_json_path = output_dir / f"{exp.output_stem}_seed{seed}.json"
    with open(run_csv_path, newline="", encoding="utf-8") as fp:
        row = next(csv.DictReader(fp))

    merged = {
        "param_name": exp.param_name,
        "param_value": exp.param_value,
        "etrc_lambda_max": exp.etrc_lambda_max,
        "etrc_lambda_base": exp.etrc_lambda_base,
        "etrc_harmful_penalty": exp.etrc_harmful_penalty,
        "etrc_delta_tau": exp.etrc_delta_tau,
        "run_csv_path": str(run_csv_path),
        "run_json_path": str(run_json_path),
        **row,
    }
    return merged, completed.stdout


def gpu_worker(gpu_id: int, experiments, seed: int, output_dir: Path, shared_split_dir: Path):
    results = []
    stdout_chunks = []
    for exp in experiments:
        result, stdout = run_one(exp, gpu_id, seed, output_dir, shared_split_dir)
        results.append(result)
        stdout_chunks.append(stdout)
    return results, stdout_chunks


def save_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    parser.add_argument(
        "--shared_split_dir",
        type=Path,
        default=Path("checkpoint/shared_repeatable_splits"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("etrc_backbone_compare_results/etrcconetlogit_amazon_single_param_sweep"),
    )
    args = parser.parse_args()

    split_path = args.shared_split_dir / f"amazon_repeatable_seed{args.seed}_fullsort_dataloaders.pth"
    if not split_path.exists():
        raise FileNotFoundError(f"Shared split not found: {split_path}")

    experiments = build_experiments()
    buckets = {gpu_id: [] for gpu_id in args.gpus}
    for idx, exp in enumerate(experiments):
        gpu_id = args.gpus[idx % len(args.gpus)]
        buckets[gpu_id].append(exp)

    all_results = []
    all_stdout = {}
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        future_map = {
            executor.submit(
                gpu_worker,
                gpu_id,
                buckets[gpu_id],
                args.seed,
                args.output_dir,
                args.shared_split_dir,
            ): gpu_id
            for gpu_id in args.gpus
        }
        for future in as_completed(future_map):
            gpu_id = future_map[future]
            results, stdout_chunks = future.result()
            all_results.extend(results)
            all_stdout[gpu_id] = stdout_chunks

    all_results.sort(
        key=lambda row: (
            PARAM_ORDER.index(row["param_name"]),
            SWEEP_VALUES[row["param_name"]].index(float(row["param_value"])),
        )
    )

    summary_csv = args.output_dir / f"etrcconetlogit_amazon_single_param_sweep_seed{args.seed}.csv"
    summary_json = args.output_dir / f"etrcconetlogit_amazon_single_param_sweep_seed{args.seed}.json"
    save_csv(summary_csv, all_results, SUMMARY_COLUMNS)
    with open(summary_json, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "defaults": DEFAULTS,
                "sweep_values": SWEEP_VALUES,
                "results": all_results,
            },
            fp,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps({"summary_csv": str(summary_csv), "summary_json": str(summary_json)}, ensure_ascii=False))
    for gpu_id in sorted(all_stdout):
        print(f"===== GPU {gpu_id} =====")
        for chunk in all_stdout[gpu_id]:
            sys.stdout.write(chunk)


if __name__ == "__main__":
    main()
