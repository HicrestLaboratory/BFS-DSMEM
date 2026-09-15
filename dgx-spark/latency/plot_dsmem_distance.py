#!/usr/bin/env python3
"""DSMEM latency vs. rank distance, one panel per cluster size.

    ~/tools/SbatchMan/.venv/bin/python plot_dsmem_distance.py [--out ../slides/images/dsmem-distance.png]

Numbers are the medians printed by latency-original (cycles per dependent
load, one thread, 16 384 steps, 101 reps). Distance 0 is the block's own
shared memory through map_shared_rank.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CYCLES = {
    2:  [36.9, 210.9],
    4:  [36.9, 210.9, 198.9, 198.9],
    8:  [36.9, 210.9, 198.9, 198.9, 187.9, 187.9, 203.9, 203.9],
    12: [36.9, 210.9, 198.9, 198.9, 187.9, 187.9, 203.9, 203.9, 191.9, 191.9, 179.9, 179.9],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../slides/images/dsmem-distance.png")
    args = ap.parse_args()

    sizes = sorted(CYCLES)
    fig, axes = plt.subplots(1, len(sizes), figsize=(13, 2.7), sharey=True,
                             gridspec_kw={"width_ratios": [len(CYCLES[s]) + 1 for s in sizes]})
    local, remote = "#9e9e9e", "#1f77b4"
    for ax, cs in zip(axes, sizes):
        cy = CYCLES[cs]
        colors = [local] + [remote] * (len(cy) - 1)
        bars = ax.bar(range(len(cy)), cy, color=colors, width=0.75)
        for b, v in zip(bars, cy):
            ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_title(f"cluster size {cs}", fontsize=10)
        ax.set_xticks(range(len(cy)))
        ax.set_xticklabels([str(d) for d in range(len(cy))], fontsize=8)
        ax.set_xlabel("rank distance", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("latency (cycles per load)", fontsize=9)
    axes[0].set_ylim(0, 240)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
