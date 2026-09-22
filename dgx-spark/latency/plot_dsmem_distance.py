#!/usr/bin/env python3
"""DSMEM latency vs. rank distance, one panel per cluster size.

    ~/tools/SbatchMan/.venv/bin/python plot_dsmem_distance.py [--out ../slides/images/dsmem-distance.png]

Numbers are the medians printed by latency-original (cycles per dependent
load, one thread, 16 384 steps, 101 reps). Distance 0 is the block's own
shared memory through map_shared_rank.
"""

import argparse

import plotstyle                      # noqa: F401  (house style, must precede pyplot use)
import matplotlib.pyplot as plt

CYCLES = {
    2:  [36.9, 210.9],
    4:  [36.9, 210.9, 198.9, 198.9],
    8:  [36.9, 210.9, 198.9, 198.9, 187.9, 187.9, 203.9, 203.9],
    12: [36.9, 210.9, 198.9, 198.9, 187.9, 187.9, 203.9, 203.9, 191.9, 191.9, 179.9, 179.9],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../slides/images/dsmem-distance.png")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    sizes = sorted(CYCLES)
    fig, axes = plt.subplots(1, len(sizes), figsize=(17, 5.2), sharey=True,
                             gridspec_kw={"width_ratios": [len(CYCLES[s]) + 1 for s in sizes]})
    # Let the panels breathe: a shared y-axis otherwise butts them together.
    fig.get_layout_engine().set(wspace=0.07, w_pad=0.10)
    local, remote = "#9e9e9e", "#1f77b4"
    for ax, cs in zip(axes, sizes):
        cy = CYCLES[cs]
        colors = [local] + [remote] * (len(cy) - 1)
        bars = ax.bar(range(len(cy)), cy, color=colors, width=0.75)
        for b, v in zip(bars, cy):
            ax.text(b.get_x() + b.get_width() / 2, v + 6, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=16)
        ax.set_title(f"cluster size {cs}")
        ax.set_xticks(range(len(cy)))
        ax.set_xticklabels([str(d) for d in range(len(cy))])
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("latency (cycles per load)")
    axes[0].set_ylim(0, 260)
    # One x-label for all four panels instead of the same words four times.
    fig.supxlabel("rank distance")
    fig.suptitle(args.title)

    for p in plotstyle.save(fig, args.out):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
