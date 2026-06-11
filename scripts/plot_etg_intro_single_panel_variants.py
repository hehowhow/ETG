import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "generated" / "figures"

TARGET_ONLY_COLOR = "#C9D7EA"
TRANSFER_GAIN_COLOR = "#5CB8A8"
HARMFUL_GAIN_COLOR = "#E8A3A3"
OVERALL_EDGE = "#6C7A89"
OVERLAP_EDGE = "#456990"
NON_OVERLAP_EDGE = "#6C7A89"
TEXT_COLOR = "#2F3640"
GRID_COLOR = "#D9DEE7"

PLOT_CONFIGS = {
    "dtcdr_amazon": {
        "data_json": ROOT / "etrc_backbone_compare_results" / "dtcdrtgf_cross_dataset_repeatable_seed2022.json",
        "dataset": "amazon",
        "panel_title": "Standard CDR",
        "subtitle": "DTCDR / Amazon",
        "output_stem": "etg_intro_single_panel_dtcdr_amazon",
        "ylim": (0.0, 0.0195),
    },
    "etg_dtcdr_amazon": {
        "data_json": ROOT / "etrc_backbone_compare_results" / "etrcdtcdrlogit_cross_dataset_repeatable_seed2022.json",
        "dataset": "amazon",
        "panel_title": "ETG-augmented CDR",
        "subtitle": "ETG-DTCDR / Amazon",
        "output_stem": "etg_intro_single_panel_etg_dtcdr_amazon",
        "ylim": (0.0, 0.0202),
    },
    "mfgslae_amazon": {
        "data_json": ROOT / "etrc_backbone_compare_results" / "mfgslaetgf_cross_dataset_repeatable_seed2022.json",
        "dataset": "amazon",
        "panel_title": "Standard CDR",
        "subtitle": "MFGSLAE / Amazon",
        "output_stem": "etg_intro_single_panel_mfgslae_amazon",
        "ylim": (0.01, 0.0258),
    },
    "etg_mfgslae_amazon": {
        "data_json": ROOT / "etrc_backbone_compare_results" / "etrcmfgslae_userlevel_cross_dataset_repeatable_seed2022.json",
        "dataset": "amazon",
        "panel_title": "ETG-augmented CDR",
        "subtitle": "ETG-MFGSLAE / Amazon",
        "output_stem": "etg_intro_single_panel_etg_mfgslae_amazon",
        "ylim": (0.015, 0.0249),
    },
}


def load_stats(data_json: Path, dataset: str):
    data = json.loads(data_json.read_text(encoding="utf-8"))[dataset]["test_result"]
    return {
        "overall_target_only": data["target_only"]["overall"]["ndcg@10"],
        "overlap_target_only": data["target_only"]["overlapped"]["ndcg@10"],
        "non_overlap_target_only": data["target_only"]["non_overlapped"]["ndcg@10"],
        "overall_gain": data["transfer_gain"]["overall"]["ndcg@10"],
        "overlap_gain": data["transfer_gain"]["overlapped"]["ndcg@10"],
        "non_overlap_gain": data["transfer_gain"]["non_overlapped"]["ndcg@10"],
        "overall_final": data["overall"]["ndcg@10"],
        "overlap_final": data["overlapped"]["ndcg@10"],
        "non_overlap_final": data["non_overlapped"]["ndcg@10"],
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def draw_single_panel(cfg):
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 15,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    stats = load_stats(cfg["data_json"], cfg["dataset"])
    labels = ["Overall", "Overlap", "Non-overlap"]
    target_only = [
        stats["overall_target_only"],
        stats["overlap_target_only"],
        stats["non_overlap_target_only"],
    ]
    gain = [
        stats["overall_gain"],
        stats["overlap_gain"],
        stats["non_overlap_gain"],
    ]
    final_scores = [
        stats["overall_final"],
        stats["overall_final"] if False else stats["overall_final"],
        stats["non_overlap_final"],
    ]
    final_scores = [
        stats["overall_final"],
        stats["overlap_final"],
        stats["non_overlap_final"],
    ]
    edge_colors = [OVERALL_EDGE, OVERLAP_EDGE, NON_OVERLAP_EDGE]
    has_positive = any(value > 0 for value in gain)
    has_harmful = any(value < 0 for value in gain)

    # Keep a wider fixed canvas so legend-heavy panels are not clipped while
    # preserving the same exported width across all four figures.
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    x = np.arange(len(labels))
    width = 0.62

    for i in range(len(labels)):
        gain_color = TRANSFER_GAIN_COLOR if gain[i] >= 0 else HARMFUL_GAIN_COLOR
        ax.bar(
            x[i],
            target_only[i],
            width=width,
            color=TARGET_ONLY_COLOR,
            edgecolor=edge_colors[i],
            linewidth=1.8,
            zorder=3,
        )
        ax.bar(
            x[i],
            gain[i],
            width=width,
            bottom=target_only[i],
            color=gain_color,
            edgecolor=edge_colors[i],
            linewidth=1.8,
            zorder=3,
        )

    ymin, ymax = cfg["ylim"]
    ax.set_title(cfg["panel_title"], pad=10, fontsize=19, fontweight="bold", color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    for tick in ax.get_xticklabels():
        tick.set_color(TEXT_COLOR)
        tick.set_fontweight("bold")

    ax.set_ylabel("Group-level NDCG@10", fontsize=16)
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", linestyle="--", color=GRID_COLOR, linewidth=0.8, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#98A2B3")
    ax.spines["bottom"].set_color("#98A2B3")
    ax.tick_params(axis="y", labelsize=14)

    y_span = ymax - ymin
    tiny_gain_threshold = y_span * 0.018
    label_offset = y_span * 0.02

    for xi, base, gain_value, final in zip(x, target_only, gain, final_scores):
        visible_base_bottom = max(0.0, ymin)
        if gain_value < 0:
            base_label_y = (visible_base_bottom + final) / 2
        else:
            base_label_y = (visible_base_bottom + base) / 2
        ax.text(
            xi,
            base_label_y,
            fmt(base),
            ha="center",
            va="center",
            fontsize=13.5,
            color="#31445E",
            fontweight="bold",
        )

        if gain_value > 0:
            gain_label_y = base + gain_value / 2
            gain_label_color = TEXT_COLOR if gain_value < tiny_gain_threshold else "white"
            if gain_value < tiny_gain_threshold:
                gain_label_y = base + max(gain_value / 2, label_offset * 0.18)
            ax.text(
                xi,
                gain_label_y,
                fmt(gain_value),
                ha="center",
                va="center",
                fontsize=12.8,
                color=gain_label_color,
                fontweight="bold",
            )
            final_y = min(ymax - label_offset * 0.25, final + label_offset)
            final_va = "bottom"
        else:
            if abs(gain_value) < tiny_gain_threshold:
                gain_label_y = base + label_offset * 0.15
            else:
                gain_label_y = base + gain_value / 2
            ax.text(
                xi,
                gain_label_y,
                fmt(gain_value),
                ha="center",
                va="center",
                fontsize=12.8,
                color=TEXT_COLOR,
                fontweight="bold",
            )
            if abs(gain_value) < tiny_gain_threshold:
                final_y = max(ymin + label_offset * 0.25, final - label_offset * 1.6)
            else:
                final_y = max(ymin + label_offset * 0.25, final - label_offset)
            final_va = "top"

        ax.text(
            xi,
            final_y,
            fmt(final),
            ha="center",
            va=final_va,
            fontsize=14,
            color=TEXT_COLOR,
            fontweight="bold",
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=TARGET_ONLY_COLOR, edgecolor="#8AA0BE", linewidth=1.0)
    ]
    legend_labels = ["Target-only performance"]
    if has_positive:
        handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=TRANSFER_GAIN_COLOR, edgecolor="#4F9E92", linewidth=1.0)
        )
        legend_labels.append("Transfer gain")
    if has_harmful:
        handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=HARMFUL_GAIN_COLOR, edgecolor="#C27B7B", linewidth=1.0)
        )
        legend_labels.append("Harmful transfer")

    ax.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#C7CED8",
        fontsize=13,
        bbox_to_anchor=(0.5, 1.28),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_out = OUTPUT_DIR / f"{cfg['output_stem']}.png"
    pdf_out = OUTPUT_DIR / f"{cfg['output_stem']}.pdf"
    # Use fixed margins instead of tight cropping so all panels keep the same
    # exported canvas size and therefore the same visual bar width.
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.12, top=0.78)
    fig.savefig(png_out, dpi=300)
    fig.savefig(pdf_out)
    plt.close(fig)
    print(png_out)
    print(pdf_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot",
        choices=["all", *sorted(PLOT_CONFIGS.keys())],
        default="all",
    )
    args = parser.parse_args()

    if args.plot == "all":
        for key in sorted(PLOT_CONFIGS.keys()):
            draw_single_panel(PLOT_CONFIGS[key])
    else:
        draw_single_panel(PLOT_CONFIGS[args.plot])


if __name__ == "__main__":
    main()
