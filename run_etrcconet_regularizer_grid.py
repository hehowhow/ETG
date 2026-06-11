import csv
import glob
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path("/home/howhow/miniconda3/envs/AHBF/bin/python")
VALUES = [0.005, 0.01, 0.05, 0.1]

BASE_CONFIGS = [
    ROOT / "recbole_cdr/properties/dataset/Epinions.yaml",
    ROOT / "recbole_cdr/properties/experiment/no_progress.yaml",
]
TMP_CONFIG_DIR = ROOT / "tmp_etrc_grid_configs"
RUN_LOG_DIR = ROOT / "etrcconet_regularizer_grid_run_logs"
RESULT_JSON = ROOT / "etrcconet_regularizer_grid_results.json"
RESULT_CSV = ROOT / "etrcconet_regularizer_grid_results.csv"


def parse_metric_line(line):
    return {
        key.lower(): float(value)
        for key, value in re.findall(r"([a-z]+@\d+)\s*:\s*([-+]?\d+(?:\.\d+)?)", line, re.IGNORECASE)
    }


def parse_section(log_text, section_name):
    match = re.search(rf"test result {re.escape(section_name)}:\n([^\n]+)", log_text)
    return parse_metric_line(match.group(1)) if match else {}


def parse_best_epoch(log_text):
    match = re.search(r"Finished training, best eval result in epoch (\d+)", log_text)
    if match:
        return int(match.group(1))

    best_epoch = None
    best_score = -float("inf")
    for epoch, score in re.findall(r"epoch (\d+) evaluating .*?valid_score: ([0-9.]+)", log_text):
        score_value = float(score)
        if score_value > best_score:
            best_score = score_value
            best_epoch = int(epoch)
    return best_epoch


def latest_model_log():
    logs = [Path(path) for path in glob.glob(str(ROOT / "log/ETRCCoNet/*.log"))]
    if not logs:
        return None
    return max(logs, key=lambda path: path.stat().st_mtime)


def run_one(index, noharm_weight, tgf_weight):
    override = TMP_CONFIG_DIR / f"etrcconet_noharm_{noharm_weight:g}_tgf_{tgf_weight:g}.yaml"
    override.write_text(
        f"etrc_noharm_weight: {noharm_weight}\n"
        f"etrc_tgf_weight: {tgf_weight}\n",
        encoding="utf-8",
    )

    config_files = " ".join(str(path.relative_to(ROOT)) for path in [*BASE_CONFIGS, override])
    command = [
        str(PYTHON),
        "run_recbole_cdr.py",
        "--model=ETRCCoNet",
        f"--config_files={config_files}",
    ]

    print(
        f"[{index:02d}/16] noharm={noharm_weight:g}, tgf={tgf_weight:g} running...",
        flush=True,
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    captured_log = RUN_LOG_DIR / f"run_{index:02d}_noharm_{noharm_weight:g}_tgf_{tgf_weight:g}.log"
    captured_log.write_text(completed.stdout, encoding="utf-8")

    recbole_log = latest_model_log()
    log_text = recbole_log.read_text(encoding="utf-8") if recbole_log else completed.stdout

    result = {
        "index": index,
        "noharm_weight": noharm_weight,
        "tgf_weight": tgf_weight,
        "returncode": completed.returncode,
        "captured_log": str(captured_log.relative_to(ROOT)),
        "recbole_log": str(recbole_log.relative_to(ROOT)) if recbole_log else None,
        "best_epoch": parse_best_epoch(log_text),
        "overall": parse_section(log_text, "overall"),
        "overlapped": parse_section(log_text, "overlapped"),
        "non_overlapped": parse_section(log_text, "non_overlapped"),
        "ugf": parse_section(log_text, "ugf"),
        "tgf": parse_section(log_text, "tgf"),
    }

    overall = result["overall"]
    ugf = result["ugf"]
    tgf = result["tgf"]
    print(
        "[{idx:02d}/16] done rc={rc} best={best} "
        "NDCG@10={ndcg:.4f} HR@10={hit:.4f} UGF_NDCG@10={ugf_ndcg:.4f} "
        "TGF_NDCG@10={tgf_ndcg:.4f}".format(
            idx=index,
            rc=completed.returncode,
            best=result["best_epoch"] if result["best_epoch"] is not None else -1,
            ndcg=overall.get("ndcg@10", float("nan")),
            hit=overall.get("hit@10", float("nan")),
            ugf_ndcg=ugf.get("ndcg@10", float("nan")),
            tgf_ndcg=tgf.get("ndcg@10", float("nan")),
        ),
        flush=True,
    )
    return result


def write_results(results):
    RESULT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    fields = [
        "index",
        "noharm_weight",
        "tgf_weight",
        "returncode",
        "best_epoch",
        "overall_ndcg@10",
        "overall_ndcg@20",
        "overall_hit@10",
        "overall_hit@20",
        "ugf_ndcg@10",
        "ugf_hit@10",
        "tgf_ndcg@10",
        "tgf_hit@10",
        "recbole_log",
        "captured_log",
    ]
    with RESULT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "index": result["index"],
                    "noharm_weight": result["noharm_weight"],
                    "tgf_weight": result["tgf_weight"],
                    "returncode": result["returncode"],
                    "best_epoch": result["best_epoch"],
                    "overall_ndcg@10": result["overall"].get("ndcg@10"),
                    "overall_ndcg@20": result["overall"].get("ndcg@20"),
                    "overall_hit@10": result["overall"].get("hit@10"),
                    "overall_hit@20": result["overall"].get("hit@20"),
                    "ugf_ndcg@10": result["ugf"].get("ndcg@10"),
                    "ugf_hit@10": result["ugf"].get("hit@10"),
                    "tgf_ndcg@10": result["tgf"].get("ndcg@10"),
                    "tgf_hit@10": result["tgf"].get("hit@10"),
                    "recbole_log": result["recbole_log"],
                    "captured_log": result["captured_log"],
                }
            )


def main():
    TMP_CONFIG_DIR.mkdir(exist_ok=True)
    RUN_LOG_DIR.mkdir(exist_ok=True)

    results = []
    index = 1
    for noharm_weight in VALUES:
        for tgf_weight in VALUES:
            results.append(run_one(index, noharm_weight, tgf_weight))
            write_results(results)
            index += 1

    ok_results = [result for result in results if result["returncode"] == 0]
    if ok_results:
        best = max(ok_results, key=lambda item: item["overall"].get("ndcg@10", -1))
        print(
            "BEST overall NDCG@10: noharm={noharm:g}, tgf={tgf:g}, "
            "best_epoch={epoch}, NDCG@10={ndcg:.4f}, HR@10={hit:.4f}".format(
                noharm=best["noharm_weight"],
                tgf=best["tgf_weight"],
                epoch=best["best_epoch"],
                ndcg=best["overall"].get("ndcg@10", float("nan")),
                hit=best["overall"].get("hit@10", float("nan")),
            ),
            flush=True,
        )

    print(f"Wrote {RESULT_JSON.relative_to(ROOT)}", flush=True)
    print(f"Wrote {RESULT_CSV.relative_to(ROOT)}", flush=True)
    return 0 if all(result["returncode"] == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
