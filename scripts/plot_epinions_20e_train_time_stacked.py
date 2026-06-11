import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "etrc_runtime_epinions_20e_combined" / "epinions_20e_no_earlystop_seed2022_combined.csv"
OUTPUT_DIR = PROJECT_ROOT / "generated" / "runtime_figures"

BASE_METHODS = ["CLMF", "CoNet", "MFGSLAE", "DTCDR", "BiTGCF"]
ETG_METHOD_MAP = {
    "CLMF": "ETRC-CLMF-Logit",
    "CoNet": "ETRC-CoNet-Logit",
    "MFGSLAE": "ETRC-MFGSLAE",
    "DTCDR": "ETRC-DTCDR-Logit",
    "BiTGCF": "ETRC-BiTGCF-StrictLogit",
}

BASE_COLOR = "#456990"
OVERHEAD_COLOR = "#EF767A"
EDGE_COLOR = "#4A4A4A"


def read_rows(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows[row["label"]] = row
    return rows


def to_float(row, key):
    return float(row[key])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows(CSV_PATH)

    base_train = []
    etg_overhead = []
    total_train = []

    for method in BASE_METHODS:
        base_label = method
        etg_label = ETG_METHOD_MAP[method]
        base_time = to_float(rows[base_label], "train_time_sec")
        etg_time = to_float(rows[etg_label], "train_time_sec")
        overhead = max(etg_time - base_time, 0.0)

        base_train.append(base_time)
        etg_overhead.append(overhead)
        total_train.append(etg_time)

    x = np.arange(len(BASE_METHODS))
    width = 0.62

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 15,
            "legend.fontsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    ax.bar(
        x,
        base_train,
        width=width,
        color=BASE_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.8,
        label="Backbone Method",
    )
    ax.bar(
        x,
        etg_overhead,
        width=width,
        bottom=base_train,
        color=OVERHEAD_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.8,
        label="w ETG",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(BASE_METHODS, rotation=0, ha="center", fontsize=13)
    ax.set_xlabel("Method", fontsize=15)
    ax.set_ylabel("Train Time (s)", fontsize=15)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    legend = ax.legend(
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#B8BEC8",
        loc="upper right",
        fontsize=13,
    )
    legend.get_frame().set_linewidth(0.8)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ymax = max(total_train) * 1.14
    ax.set_ylim(0, ymax)

    png_path = OUTPUT_DIR / "epinions_20e_train_time_stacked_bar.png"
    pdf_path = OUTPUT_DIR / "epinions_20e_train_time_stacked_bar.pdf"

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
