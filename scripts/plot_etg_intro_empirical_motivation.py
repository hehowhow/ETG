import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT
RESULTS_DIR = PROJECT_ROOT / "results" / "main"
OUTPUT_DIR = PROJECT_ROOT / "figures"

TARGET_ONLY_COLOR = "#C9D7EA"
TRANSFER_GAIN_COLOR = "#5CB8A8"
HARMFUL_GAIN_COLOR = "#E8A3A3"
OVERLAP_EDGE = "#456990"
NON_OVERLAP_EDGE = "#E39A3B"
TEXT_COLOR = "#2F3640"
GRID_COLOR = "#D9DEE7"
TIMES_FONT_PATH = Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf")

PLOT_CONFIGS = {
    "dtcdr_amazon": {
        "dataset": "amazon",
        "std_json": RESULTS_DIR / "dtcdrtgf_cross_dataset_repeatable_seed2022.json",
        "etg_json": RESULTS_DIR / "etrcdtcdrlogit_cross_dataset_repeatable_seed2022.json",
        "subtitle_std": "DTCDR / Amazon",
        "subtitle_etg": "ETG-DTCDR / Amazon",
        "output_stem": "etg_intro_motivation_empirical",
        "ylim": 0.0228,
    },
    "dtcdr_douban": {
        "dataset": "douban",
        "std_json": RESULTS_DIR / "dtcdrtgf_cross_dataset_repeatable_seed2022.json",
        "etg_json": RESULTS_DIR / "etrcdtcdrlogit_cross_dataset_repeatable_seed2022.json",
        "subtitle_std": "DTCDR / Douban",
        "subtitle_etg": "ETG-DTCDR / Douban",
        "output_stem": "etg_intro_motivation_empirical_dtcdr_douban",
        "ylim": 0.055,
    },
    "conet_epinions": {
        "dataset": "epinions",
        "std_json": RESULTS_DIR / "conettgf_cross_dataset_repeatable_seed2022.json",
        "etg_json": RESULTS_DIR / "etrcconet_cross_dataset_repeatable_seed2022.json",
        "subtitle_std": "CoNet / Epinions",
        "subtitle_etg": "ETG-CoNet / Epinions",
        "output_stem": "etg_intro_motivation_empirical_conet_epinions",
        "ylim": 0.062,
    },
    "mfgslae_amazon": {
        "dataset": "amazon",
        "std_json": RESULTS_DIR / "mfgslaetgf_cross_dataset_repeatable_seed2022.json",
        "etg_json": RESULTS_DIR / "etrcmfgslae_userlevel_cross_dataset_repeatable_seed2022.json",
        "subtitle_std": "MFGSLAE / Amazon",
        "subtitle_etg": "ETG-MFGSLAE / Amazon",
        "output_stem": "etg_intro_motivation_empirical_mfgslae_amazon",
        "ylim": 0.027,
    },
}


def load_group_ndcg10(path: Path, dataset: str):
    data = json.loads(path.read_text(encoding="utf-8"))[dataset]["test_result"]
    return {
        "overlap_target_only": data["target_only"]["overlapped"]["ndcg@10"],
        "non_overlap_target_only": data["target_only"]["non_overlapped"]["ndcg@10"],
        "overlap_gain": data["transfer_gain"]["overlapped"]["ndcg@10"],
        "non_overlap_gain": data["transfer_gain"]["non_overlapped"]["ndcg@10"],
        "overlap_final": data["overlapped"]["ndcg@10"],
        "non_overlap_final": data["non_overlapped"]["ndcg@10"],
        "tgf": data["tgf"]["ndcg@10"],
        "ugf": data["ugf"]["ndcg@10"],
    }


def _format_value(value):
    return f"{value:.4f}"


def plot_panel(ax, stats, panel_tag, title, subtitle):
    labels = ["Overlap", "Non-overlap"]
    x = np.arange(2)
    width = 0.56

    target_only = [stats["overlap_target_only"], stats["non_overlap_target_only"]]
    gain = [stats["overlap_gain"], stats["non_overlap_gain"]]
    final_scores = [stats["overlap_final"], stats["non_overlap_final"]]
    edge_colors = [OVERLAP_EDGE, NON_OVERLAP_EDGE]

    for i in range(2):
        gain_color = TRANSFER_GAIN_COLOR if gain[i] >= 0 else HARMFUL_GAIN_COLOR
        ax.bar(
            x[i],
            target_only[i],
            width=width,
            color=TARGET_ONLY_COLOR,
            edgecolor=edge_colors[i],
            linewidth=1.6,
            zorder=3,
        )
        ax.bar(
            x[i],
            gain[i],
            width=width,
            bottom=target_only[i],
            color=gain_color,
            edgecolor=edge_colors[i],
            linewidth=1.6,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    for tick, color in zip(ax.get_xticklabels(), edge_colors):
        tick.set_color(color)
        tick.set_fontweight("bold")

    ax.text(0.02, 1.08, panel_tag, transform=ax.transAxes, ha="left", va="bottom", fontsize=12.5, fontweight="bold", color=TEXT_COLOR)
    ax.text(0.5, 1.08, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=15.5, fontweight="bold", color=TEXT_COLOR)
    ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10.8, color="#667085")

    ax.grid(axis="y", linestyle="--", color=GRID_COLOR, linewidth=0.8, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#98A2B3")
    ax.spines["bottom"].set_color("#98A2B3")
    ax.tick_params(axis="y", labelsize=10)

    y_max = ax.get_ylim()[1]
    tiny_gain_threshold = y_max * 0.012
    label_offset = y_max * 0.014

    for xi, base, gain_value, final in zip(x, target_only, gain, final_scores):
        ax.text(
            xi,
            base / 2,
            _format_value(base),
            ha="center",
            va="center",
            fontsize=10.4,
            color="#31445E",
            fontweight="bold",
        )
        if gain_value > 0:
            if gain_value < tiny_gain_threshold:
                gain_y = base + gain_value / 2
                gain_color = TEXT_COLOR
            else:
                gain_y = base + gain_value / 2
                gain_color = "white"
            ax.text(
                xi,
                gain_y,
                _format_value(gain_value),
                ha="center",
                va="center",
                fontsize=10.2,
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
                _format_value(gain_value),
                ha="center",
                va="center",
                fontsize=10.0,
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
            _format_value(final),
            ha="center",
            va=final_va,
            fontsize=10.8,
            color=TEXT_COLOR,
            fontweight="bold",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot",
        choices=sorted(PLOT_CONFIGS.keys()),
        default="dtcdr_amazon",
    )
    args = parser.parse_args()

    cfg = PLOT_CONFIGS[args.plot]
    std_stats = load_group_ndcg10(cfg["std_json"], cfg["dataset"])
    etg_stats = load_group_ndcg10(cfg["etg_json"], cfg["dataset"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_out = OUTPUT_DIR / f"{cfg['output_stem']}.png"
    pdf_out = OUTPUT_DIR / f"{cfg['output_stem']}.pdf"
    has_harmful = any(
        value < 0
        for value in [
            std_stats["overlap_gain"],
            std_stats["non_overlap_gain"],
            etg_stats["overlap_gain"],
            etg_stats["non_overlap_gain"],
        ]
    )
    has_positive = any(
        value > 0
        for value in [
            std_stats["overlap_gain"],
            std_stats["non_overlap_gain"],
            etg_stats["overlap_gain"],
            etg_stats["non_overlap_gain"],
        ]
    )

    if TIMES_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(TIMES_FONT_PATH))

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 14,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), sharey=True)

    plot_panel(
        axes[0],
        std_stats,
        "(a)",
        "Standard CDR",
        cfg["subtitle_std"],
    )
    plot_panel(
        axes[1],
        etg_stats,
        "(b)",
        "ETG-augmented CDR",
        cfg["subtitle_etg"],
    )

    axes[0].set_ylabel("Group-level NDCG@10", fontsize=11.5)
    axes[0].set_ylim(0, cfg["ylim"])

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=TARGET_ONLY_COLOR, edgecolor="#8AA0BE", linewidth=1.0),
    ]
    labels = ["Target-only performance"]
    if has_positive:
        handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=TRANSFER_GAIN_COLOR, edgecolor="#4F9E92", linewidth=1.0)
        )
        labels.append("Transfer gain")
    if has_harmful:
        handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=HARMFUL_GAIN_COLOR, edgecolor="#C27B7B", linewidth=1.0)
        )
        labels.append("Harmful transfer")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#C7CED8",
        bbox_to_anchor=(0.5, 1.05),
        fontsize=10.6,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.9], w_pad=2.0)
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    print(png_out)
    print(pdf_out)


if __name__ == "__main__":
    main()
