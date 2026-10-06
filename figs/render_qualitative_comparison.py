"""Render matched CCB-versus-Agent text-to-image retrieval examples.

Input is a JSON list.  Each record represents one query and supplies the same
ground-truth image plus the top-5 ranked results of both methods::

  {
    "query": "There are two airplanes at the airport.",
    "ground_truth": "/absolute/path/to/gt.jpg",
    "ccb": [{"path": "/.../rank1.jpg", "score": 0.271}, ...],
    "agent": [{"path": "/.../rank1.jpg", "score": 0.288}, ...]
  }

The first matching ground-truth image in either ranked list is highlighted in
green.  Absolute paths make the manifest portable across experiment folders.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"


def load_records(path):
    records = json.loads(Path(path).read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("manifest must be a non-empty JSON list")
    for record in records:
        for key in ("query", "ground_truth", "ccb", "agent"):
            if key not in record:
                raise ValueError(f"missing {key!r} in one manifest record")
        if len(record["ccb"]) != 5 or len(record["agent"]) != 5:
            raise ValueError("each method must provide exactly five ranked images")
    return records


def image_tile(ax, item, rank, ground_truth):
    path = Path(item["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    ax.imshow(Image.open(path).convert("RGB"))
    correct = path.resolve() == Path(ground_truth).resolve()
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(2.1 if correct else 0.55)
        spine.set_edgecolor(GREEN if correct else "#CFCFCF")
    score = item.get("score")
    label = f"#{rank}" + (f"  {score:.3f}" if score is not None else "")
    if correct:
        label += "  ✓"
    ax.set_title(label, fontsize=7.4, color=GREEN if correct else GREY, pad=3)


def draw(records, output):
    # Each case is: query label + GT / CCB Top-5 / Agent Top-5.
    fig = plt.figure(figsize=(11.0, 3.05 * len(records)), layout="constrained")
    outer = fig.add_gridspec(len(records), 1)
    for row, record in enumerate(records):
        grid = outer[row].subgridspec(3, 6, height_ratios=[0.35, 1, 1], hspace=0.08, wspace=0.08)
        title_ax = fig.add_subplot(grid[0, :])
        title_ax.axis("off")
        title_ax.text(0, 0.60, f"Query {row + 1}: {record['query']}", color="#222222", fontsize=9, weight="medium")
        title_ax.text(0, 0.10, "Text-to-image retrieval. Green frame marks a ground-truth image.", color=GREY, fontsize=7.5)

        gt_ax = fig.add_subplot(grid[1, 0])
        gt_ax.imshow(Image.open(record["ground_truth"]).convert("RGB"))
        gt_ax.set_title("Ground truth\nCCB Top-5 →", fontsize=7.6, pad=3)
        gt_ax.set_xticks([]); gt_ax.set_yticks([])
        for spine in gt_ax.spines.values():
            spine.set_linewidth(2.1); spine.set_edgecolor(GREEN)
        # CCB row uses cells 1–5; Agent row has its label in cell 0 and uses 1–5.
        for col, item in enumerate(record["ccb"], start=1):
            image_tile(fig.add_subplot(grid[1, col]), item, col, record["ground_truth"])
        agent_label = fig.add_subplot(grid[2, 0])
        agent_label.axis("off")
        agent_label.text(0.5, 0.5, "Agent prior", ha="center", va="center", rotation=90,
                            fontsize=8.2, color=BLUE, weight="medium")
        for col, item in enumerate(record["agent"], start=1):
            image_tile(fig.add_subplot(grid[2, col]), item, col, record["ground_truth"])
    fig.savefig(output.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {output}.png and {output}.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSON ranking manifest")
    parser.add_argument("--output", default="qualitative_ccb_vs_agent")
    args = parser.parse_args()
    draw(load_records(args.manifest), Path(args.output))


if __name__ == "__main__":
    main()
