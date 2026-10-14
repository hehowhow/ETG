import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VUG_CSV_PATH = PROJECT_ROOT / "results" / "fairness" / "etrc_vug_cross_dataset_summary_seed2022.csv"
OUTPUT_DIR = PROJECT_ROOT / "figures"

ACCURACY_METRICS = [
    ("ndcg10", "NDCG@10"),
    ("ndcg20", "NDCG@20"),
    ("hit10", "HR@10"),
    ("hit20", "HR@20"),
]

FAIRNESS_METRICS = [
    ("tgf_ndcg10", "TGF(NDCG@10)"),
    ("tgf_ndcg20", "TGF(NDCG@20)"),
    ("tgf_hit10", "TGF(HR@10)"),
    ("tgf_hit20", "TGF(HR@20)"),
]

DATASET_LABELS = {
    "amazon": "Amazon",
    "douban": "Douban",
    "epinions": "Epinions",
}

NATIVE_COLOR = "#74A9FF"
ETG_COLOR = "#FFD65A"
EDGE_COLOR = "#4A4A4A"


CDFA_QB_RESULTS = {
    "native": {
        "ndcg10": 0.1448,
        "ndcg20": 0.1777,
        "hit10": 0.2803,
        "hit20": 0.4117,
        "tgf_ndcg10": 0.0012,
        "tgf_ndcg20": 0.0074,
        "tgf_hit10": 0.0105,
        "tgf_hit20": 0.0348,
    },
    "etg": {
        "ndcg10": 0.1462,
        "ndcg20": 0.1816,
        "hit10": 0.2869,
        "hit20": 0.4281,
        "tgf_ndcg10": 0.0009,
        "tgf_ndcg20": 0.0023,
        "tgf_hit10": 0.0028,
        "tgf_hit20": 0.0074,
    },
}


def read_vug_results(csv_path: Path):
    rows = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            dataset = row["dataset"].strip().lower()
            model = row["model"].strip()
            rows[(dataset, model)] = {key: _to_float(value) for key, value in row.items()}
    return rows


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def choose_headroom(y_max: float) -> float:
    if y_max <= 0.01:
        return max(0.0008, y_max * 0.35)
    if y_max <= 0.05:
        return max(0.002, y_max * 0.25)
    if y_max <= 0.15:
        return y_max * 0.18
    return y_max * 0.12


def format_value(value: float) -> str:
    return f"{value:.4f}"


def make_zero_visible(values, zero_stub):
    visible = []
    for value in values:
        if abs(value) < 1e-12:
            visible.append(zero_stub)
        else:
            visible.append(value)
    return visible


def plot_grouped_bar_chart(
    values_native,
    values_etg,
    x_labels,
    title,
    legend_labels,
    output_stem: str,
):
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    x = np.arange(len(x_labels))
    width = 0.34
    y_max = max(max(values_native), max(values_etg))
    zero_stub = max(y_max * 0.015, 0.00015)
    plot_values_native = make_zero_visible(values_native, zero_stub)
    plot_values_etg = make_zero_visible(values_etg, zero_stub)

    native_bars = ax.bar(
        x - width / 2,
        plot_values_native,
        width,
        label=legend_labels[0],
        color=NATIVE_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.8,
    )
    etg_bars = ax.bar(
        x + width / 2,
        plot_values_etg,
        width,
        label=legend_labels[1],
        color=ETG_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.8,
    )

    ax.set_ylim(0, y_max + choose_headroom(y_max))
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=13)
    ax.set_ylabel("Score", fontsize=15)
    ax.set_title(title, pad=14, fontsize=18)
    ax.tick_params(axis="y", labelsize=13)
    legend = ax.legend(
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#B8BEC8",
        loc="upper left",
        fontsize=13,
    )
    legend.get_frame().set_linewidth(0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    png_path = OUTPUT_DIR / f"{output_stem}.png"
    pdf_path = OUTPUT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def plot_vug_charts(vug_rows):
    created_paths = []
    for dataset_key, dataset_label in DATASET_LABELS.items():
        native = vug_rows[(dataset_key, "VUG")]
        etg = vug_rows[(dataset_key, "ETRCVUG")]

        acc_native = [native[key] for key, _ in ACCURACY_METRICS]
        acc_etg = [etg[key] for key, _ in ACCURACY_METRICS]
        acc_labels = [label for _, label in ACCURACY_METRICS]
        created_paths.extend(
            plot_grouped_bar_chart(
                acc_native,
                acc_etg,
                acc_labels,
                title="Accuracy",
                legend_labels=("VUG", "ETG-VUG"),
                output_stem=f"{dataset_key}_vug_accuracy_bar",
            )
        )

        fair_native = [native[key] for key, _ in FAIRNESS_METRICS]
        fair_etg = [etg[key] for key, _ in FAIRNESS_METRICS]
        fair_labels = [label for _, label in FAIRNESS_METRICS]
        created_paths.extend(
            plot_grouped_bar_chart(
                fair_native,
                fair_etg,
                fair_labels,
                title="Fairness",
                legend_labels=("VUG", "ETG-VUG"),
                output_stem=f"{dataset_key}_vug_fairness_bar",
            )
        )
    return created_paths


def plot_cdfa_charts():
    created_paths = []

    acc_native = [CDFA_QB_RESULTS["native"][key] for key, _ in ACCURACY_METRICS]
    acc_etg = [CDFA_QB_RESULTS["etg"][key] for key, _ in ACCURACY_METRICS]
    acc_labels = [label for _, label in ACCURACY_METRICS]
    created_paths.extend(
        plot_grouped_bar_chart(
            acc_native,
            acc_etg,
            acc_labels,
            title="Accuracy",
            legend_labels=("CDFA", "ETG-CDFA"),
            output_stem="qb_cdfa_accuracy_bar",
        )
    )

    fair_native = [CDFA_QB_RESULTS["native"][key] for key, _ in FAIRNESS_METRICS]
    fair_etg = [CDFA_QB_RESULTS["etg"][key] for key, _ in FAIRNESS_METRICS]
    fair_labels = [label for _, label in FAIRNESS_METRICS]
    created_paths.extend(
        plot_grouped_bar_chart(
            fair_native,
            fair_etg,
            fair_labels,
            title="Fairness",
            legend_labels=("CDFA", "ETG-CDFA"),
            output_stem="qb_cdfa_fairness_bar",
        )
    )

    return created_paths


def write_manifest(paths):
    manifest_path = OUTPUT_DIR / "bar_chart_manifest.md"
    lines = [
        "# ETG Bar Charts",
        "",
        f"Source CSV: `{VUG_CSV_PATH}`",
        "",
        "Generated files:",
        "",
    ]
    lines.extend(f"- `{path.name}`" for path in paths)
    manifest_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "legend.fontsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    vug_rows = read_vug_results(VUG_CSV_PATH)
    created_paths = []
    created_paths.extend(plot_vug_charts(vug_rows))
    created_paths.extend(plot_cdfa_charts())
    write_manifest(created_paths)

    for path in created_paths:
        print(path)


if __name__ == "__main__":
    main()
