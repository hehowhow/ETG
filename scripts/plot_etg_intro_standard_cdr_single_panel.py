import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "generated" / "figures"

TARGET_ONLY_COLOR = "#C9D7EA"
TRANSFER_GAIN_COLOR = "#5CB8A8"
HARMFUL_GAIN_COLOR = "#E8A3A3"
OVERALL_EDGE = "#6C7A89"
OVERLAP_EDGE = "#456990"
NON_OVERLAP_EDGE = "#E39A3B"
TEXT_COLOR = "#2F3640"
GRID_COLOR = "#D9DEE7"
TIMES_FONT_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")

PLOT_CONFIGS = {
    ("dtcdr", "amazon"): {
        "data_json": ROOT / "etrc_backbone_compare_results" / "dtcdrtgf_cross_dataset_repeatable_seed2022.json",
        "dataset": "amazon",
        "subtitle": "DTCDR / Amazon",
        "output_stem": "etg_intro_standard_cdr_single_panel",
        "ylim": 0.0228,
    },
    ("dtcdr", "douban"): {
        "data_json": ROOT / "etrc_backbone_compare_results" / "dtcdrtgf_cross_dataset_repeatable_seed2022.json",
        "dataset": "douban",
        "subtitle": "DTCDR / Douban",
        "output_stem": "etg_intro_standard_cdr_single_panel_douban",
        "ylim": 0.055,
    },
    ("mfgslae", "amazon"): {
        "data_json": ROOT / "etrc_backbone_compare_results" / "mfgslaetgf_cross_dataset_repeatable_seed2022.json",
        "dataset": "amazon",
        "subtitle": "MFGSLAE / Amazon",
        "output_stem": "etg_intro_standard_cdr_single_panel_mfgslae_amazon",
        "ylim": 0.027,
    },
    ("mfgslae", "douban"): {
        "data_json": ROOT / "etrc_backbone_compare_results" / "mfgslaetgf_cross_dataset_repeatable_seed2022.json",
        "dataset": "douban",
        "subtitle": "MFGSLAE / Douban",
        "output_stem": "etg_intro_standard_cdr_single_panel_mfgslae_douban",
        "ylim": 0.058,
    },
}


def load_stats(data_json, dataset):
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


def fmt(value):
    return f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["dtcdr", "mfgslae"],
        default="dtcdr",
    )
    parser.add_argument(
        "--dataset",
        choices=["amazon", "douban"],
        default="amazon",
    )
    args = parser.parse_args()

    cfg = PLOT_CONFIGS[(args.model, args.dataset)]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_out = OUTPUT_DIR / f"{cfg['output_stem']}.png"
    pdf_out = OUTPUT_DIR / f"{cfg['output_stem']}.pdf"

    if TIMES_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(TIMES_FONT_PATH))

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 16,
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
        stats["overlap_final"],
        stats["non_overlap_final"],
    ]
    edge_colors = [OVERALL_EDGE, OVERLAP_EDGE, NON_OVERLAP_EDGE]
    has_harmful = any(value < 0 for value in gain)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))

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

    ax.set_title("Standard CDR", pad=18, fontsize=17, fontweight="bold", color=TEXT_COLOR)
    ax.text(
        0.5,
        1.02,
        cfg["subtitle"],
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12.5,
        color="#667085",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    for tick, color in zip(ax.get_xticklabels(), edge_colors):
        tick.set_color(color)
        tick.set_fontweight("bold")

    ax.set_ylabel("Group-level NDCG@10", fontsize=13)
    ax.set_ylim(0, cfg["ylim"])
    ax.grid(axis="y", linestyle="--", color=GRID_COLOR, linewidth=0.8, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#98A2B3")
    ax.spines["bottom"].set_color("#98A2B3")
    ax.tick_params(axis="y", labelsize=11)

    y_max = ax.get_ylim()[1]
    tiny_gain_threshold = y_max * 0.012
    label_offset = y_max * 0.014

    for xi, base, gain_value, final in zip(x, target_only, gain, final_scores):
        ax.text(
            xi,
            base / 2,
            fmt(base),
            ha="center",
            va="center",
            fontsize=12,
            color="#31445E",
            fontweight="bold",
        )
        if gain_value > 0:
            gain_color = TEXT_COLOR if gain_value < tiny_gain_threshold else "white"
            ax.text(
                xi,
                base + gain_value / 2,
                fmt(gain_value),
                ha="center",
                va="center",
                fontsize=11.5,
                color=gain_color,
                fontweight="bold",
            )
        else:
            if abs(gain_value) < tiny_gain_threshold:
                gain_y = base + label_offset * 0.28
            else:
                gain_y = base + gain_value / 2
            ax.text(
                xi,
                gain_y,
                fmt(gain_value),
                ha="center",
                va="center",
                fontsize=11,
                color=TEXT_COLOR,
                fontweight="bold",
            )
        if gain_value >= 0:
            final_y = final + label_offset
            final_va = "bottom"
        else:
            final_y = final - label_offset
            final_va = "top"
        ax.text(
            xi,
            final_y,
            fmt(final),
            ha="center",
            va=final_va,
            fontsize=12.5,
            color=TEXT_COLOR,
            fontweight="bold",
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=TARGET_ONLY_COLOR, edgecolor="#8AA0BE", linewidth=1.0),
        plt.Rectangle((0, 0), 1, 1, facecolor=TRANSFER_GAIN_COLOR, edgecolor="#4F9E92", linewidth=1.0),
    ]
    legend_labels = ["Target-only performance", "Transfer gain"]
    if has_harmful:
        handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=HARMFUL_GAIN_COLOR, edgecolor="#C27B7B", linewidth=1.0)
        )
        legend_labels.append("Harmful transfer")
    ax.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3 if has_harmful else 2,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#C7CED8",
        fontsize=10.8,
        bbox_to_anchor=(0.5, 1.30),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.82])
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    print(png_out)
    print(pdf_out)


if __name__ == "__main__":
    main()
