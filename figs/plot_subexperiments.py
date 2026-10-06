"""Paper-ready summary of the completed CUP agent sub-experiments.

The source values are deliberately kept in this file (rather than copied from
rendered PNGs) so the figure is reproducible and individual experiments can
be updated without hand-editing a chart.  All values use the ordinary
six-recall mR protocol recorded in ``experiment_board.html``; results are
single-run (seed=0) and should not be read as estimates with uncertainty.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CURVE_CSV = ROOT / "val_curves.csv"
OUT = ROOT / "subexperiment_summary"

# Okabe--Ito palette: color-blind-safe and print-friendly.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GREY = "#7A7A7A"
LIGHT_GREY = "#D9D9D9"


def _style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
    ax.tick_params(direction="out", length=3, width=0.8)


def learning_curves(ax, curves, dataset):
    """Show only the clean, matched prior-source comparison."""
    subset = curves[(curves.dataset == dataset) & curves.variant.isin(
        ["Agent (inference)", "CCB (original)"]
    )]
    labels = {
        "Agent (inference)": ("Agent prior", BLUE, "o"),
        "CCB (original)": ("CCB prior", ORANGE, "s"),
    }
    for variant, (label, color, marker) in labels.items():
        part = subset[subset.variant == variant]
        ax.plot(part.epoch, part.val_mR, label=label, color=color,
                marker=marker, markersize=3.2, linewidth=1.8, zorder=3)
        # Directly label the final observed point; no misleading interpolation.
        final = part.iloc[-1]
        ax.annotate(f"{final.val_mR:.1f}", (final.epoch, final.val_mR),
                    xytext=(5, 2), textcoords="offset points", color=color,
                    fontsize=8, weight="medium")
    ax.set_title(dataset, pad=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation mR (%)")
    _style_axis(ax)


def reward_tradeoff(ax):
    """Make reward-hacking visible without conflating it with mR."""
    names = ["format\nonly", "hard\nrank", r"$\lambda=0.5$", r"$\lambda=1.0$", r"$\lambda=2.0$"]
    mr = np.array([63.66, 61.64, 57.94, 57.97, 56.97])
    injection = np.array([0.0, 0.0, 98.9, 90.3, 75.4])
    x = np.arange(len(names))

    bars = ax.bar(x, mr, width=0.64, color=BLUE, zorder=3, label="Test mR")
    ax.set_ylim(54, 66)
    ax.set_xticks(x, names)
    ax.set_ylabel("UCM test mR (%)")
    ax.set_title("Reward design: performance vs. tool use", pad=5)
    _style_axis(ax)
    for bar, value in zip(bars, mr):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.20, f"{value:.2f}",
                ha="center", va="bottom", fontsize=7.5, color=BLUE)

    right = ax.twinx()
    right.plot(x, injection, color=ORANGE, marker="D", linewidth=1.8,
               markersize=4, label="Tool injection rate", zorder=4)
    right.set_ylim(-8, 108)
    right.set_ylabel("Tool injection rate (%)", color=ORANGE)
    right.tick_params(axis="y", colors=ORANGE, length=3)
    right.spines["top"].set_visible(False)
    for xi, value in zip(x, injection):
        # Zero-rate labels add no information and collide with the x-axis.
        if value:
            right.annotate(f"{value:.1f}%", (xi, value), xytext=(0, 6),
                           textcoords="offset points", ha="center", fontsize=7.5,
                           color=ORANGE)
    handles = [bars, right.lines[0]]
    ax.legend(handles, ["Test mR", "Tool injection rate"], frameon=False,
              loc="upper left", fontsize=8)


def difficulty_gain(ax):
    """Paired dots preserve the absolute values and foreground the gain."""
    labels = ["Hard", "Medium", "Easy"]
    ccb = np.array([2.9, 5.7, 37.1])
    agent = np.array([7.1, 11.4, 37.1])
    x = np.arange(len(labels))
    for i, (base, ours) in enumerate(zip(ccb, agent)):
        ax.plot([i, i], [base, ours], color=LIGHT_GREY, linewidth=2.2, zorder=1)
        ax.scatter(i, base, marker="s", s=38, color=ORANGE, zorder=3)
        ax.scatter(i, ours, marker="o", s=42, color=BLUE, zorder=4)
        ax.annotate(f"{ours - base:+.1f}", (i, max(base, ours)), xytext=(0, 8),
                    textcoords="offset points", ha="center", color=GREEN,
                    fontsize=8.5, weight="medium")
    ax.scatter([], [], marker="s", color=ORANGE, label="CCB prior")
    ax.scatter([], [], marker="o", color=BLUE, label="Agent prior")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 42)
    ax.set_ylabel("T2I R@1 (%)")
    ax.set_title("Where the prior helps (UCM test)", pad=5)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    _style_axis(ax)


def data_efficiency(ax):
    ratios = np.array([25, 50, 100])
    agent = np.array([45.9, 49.6, 58.5])
    ccb_full = 50.85
    ax.plot(ratios, agent, color=BLUE, marker="o", linewidth=1.8,
            markersize=5, label="Agent prior")
    ax.axhline(ccb_full, color=ORANGE, linewidth=1.3, linestyle="--",
               label="CCB prior (100% data)")
    ax.scatter([50], [49.6], color=BLUE, s=45, zorder=4)
    ax.annotate("50%: 49.6\n(1.3 below CCB @100%)", xy=(50, 49.6),
                xytext=(57, 48.3), textcoords="data", fontsize=7.6,
                arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.8})
    ax.set_xlim(20, 105)
    ax.set_xticks(ratios, ["25", "50", "100"])
    ax.set_xlabel("Training-data ratio (%)")
    ax.set_ylabel("UCM test mR (%)")
    ax.set_title("Data efficiency", pad=5)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    _style_axis(ax)


def main():
    curves = pd.read_csv(CURVE_CSV)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(10.4, 7.2), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05])
    curve_axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    for ax, dataset in zip(curve_axes, ["UCM", "RSICD", "RSITMD"]):
        learning_curves(ax, curves, dataset)
    curve_axes[0].legend(frameon=False, loc="lower right", fontsize=8)

    reward_tradeoff(fig.add_subplot(grid[1, 0]))
    difficulty_gain(fig.add_subplot(grid[1, 1]))
    data_efficiency(fig.add_subplot(grid[1, 2]))
    fig.suptitle("Agent-prior sub-experiments  |  ordinary six-recall protocol, seed=0",
                 fontsize=12, weight="medium")
    fig.savefig(OUT.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {OUT}.png and {OUT}.pdf")


if __name__ == "__main__":
    main()
