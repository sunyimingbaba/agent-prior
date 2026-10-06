"""Render the completed word-drop robustness evaluations consistently.

Only the three measured corruption levels (0, 20 and 40%) are shown.  The
script intentionally does not extrapolate the curve to 60/80%.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "noise_robustness_summary"
NOISE = np.array([0, 20, 40])
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#7A7A7A"

# Values transcribed from the completed figures. Replace only with re-run
# measurements; never infer missing higher-noise points.
RESULTS = {
    "UCM": {
        "T2I R@1": {"Agent prior": [18.6, 14.8, 14.3], "CCB prior": [15.2, 12.9, 9.5], "CLIP zero-shot": [10.0, 11.4, 7.1]},
        "T2I R@10": {"Agent prior": [97.6, 84.7, 73.8], "CCB prior": [90.0, 79.5, 67.5], "CLIP zero-shot": [73.0, 66.4, 57.0]},
    },
    "RSITMD": {
        "T2I R@1": {"Agent prior": [39.9, 25.6, 13.1], "CLIP zero-shot": [11.9, 10.0, 6.2]},
        "T2I R@10": {"Agent prior": [81.6, 67.4, 47.8], "CLIP zero-shot": [47.5, 40.6, 32.5]},
    },
}
STYLE = {
    "Agent prior": (BLUE, "o", "-"),
    "CCB prior": (ORANGE, "s", "--"),
    "CLIP zero-shot": (GREY, "^", ":"),
}


def plot_panel(ax, title, series):
    for name, values in series.items():
        color, marker, linestyle = STYLE[name]
        ax.plot(NOISE, values, label=name, color=color, marker=marker,
                linestyle=linestyle, linewidth=1.9, markersize=5, zorder=3)
        # Directly label terminal degradation; it is more useful than a dense legend.
        ax.annotate(f"{values[-1]:.1f}", (NOISE[-1], values[-1]),
                    xytext=(5, 0), textcoords="offset points", va="center",
                    color=color, fontsize=8, weight="medium")
    ax.set_title(title, pad=5)
    ax.set_xlim(-2, 44)
    ax.set_xticks(NOISE)
    ax.set_xlabel("Word-drop noise ratio (%)")
    ax.set_ylabel("Text-to-image recall (%)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6), layout="constrained")
    for row, dataset in enumerate(["UCM", "RSITMD"]):
        for col, metric in enumerate(["T2I R@1", "T2I R@10"]):
            plot_panel(axes[row, col], f"{dataset} — {metric}", RESULTS[dataset][metric])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.suptitle("Robustness to query word-drop noise", fontsize=12, weight="medium")
    fig.savefig(OUT.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {OUT}.png and {OUT}.pdf")


if __name__ == "__main__":
    main()
