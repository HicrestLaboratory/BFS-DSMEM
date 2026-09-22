#!/usr/bin/env python3
"""The full rank-to-rank DSMEM latency matrix, and the model that explains it.

    python plot_dsmem_matrix.py [--out ../../slides/images/dsmem-matrix.png]

Left panel: median cycles per dependent load for every ordered pair of ranks,
one reader active at a time (so every cell is an unloaded latency). Right
panel: the same 36 TPC pairs plotted against w[reader] + w[target]; they fall
on one straight line, which is what "each endpoint pays its own cost, and the
distance between them is irrelevant" looks like.

Numbers are the medians printed by ./dsmem_matrix (cluster size 12, one
thread, 16 384 steps). Re-run it and paste a new MATRIX if the hardware or
the method changes.
"""

import argparse

import numpy as np

import plotstyle                      # noqa: F401  (house style, must precede pyplot use)
import matplotlib.pyplot as plt

# ./dsmem_matrix --reps 11   (rows = reader rank, cols = target rank)
MATRIX = np.array([
    [36.9, 210.9, 198.9, 198.9, 187.9, 187.9, 203.9, 203.9, 191.9, 191.9, 179.9, 179.9],
    [210.9, 36.9, 198.9, 198.9, 187.9, 187.9, 203.9, 203.9, 191.9, 191.9, 179.9, 179.9],
    [198.9, 198.9, 36.9, 186.9, 175.9, 175.9, 191.9, 191.9, 179.9, 179.9, 167.9, 167.9],
    [198.9, 198.9, 186.9, 36.9, 175.9, 175.9, 191.9, 191.9, 179.9, 179.9, 167.9, 167.9],
    [187.9, 187.9, 175.9, 175.9, 36.9, 164.9, 180.9, 180.9, 168.9, 168.9, 156.9, 156.9],
    [187.9, 187.9, 175.9, 175.9, 164.9, 36.9, 180.9, 180.9, 168.9, 168.9, 156.9, 156.9],
    [203.9, 203.9, 191.9, 191.9, 180.9, 180.9, 36.9, 196.9, 184.9, 184.9, 172.9, 172.9],
    [203.9, 203.9, 191.9, 191.9, 180.9, 180.9, 196.9, 36.9, 184.9, 184.9, 172.9, 172.9],
    [191.9, 191.9, 179.9, 179.9, 168.9, 168.9, 184.9, 184.9, 36.9, 172.9, 160.9, 160.9],
    [191.9, 191.9, 179.9, 179.9, 168.9, 168.9, 184.9, 184.9, 172.9, 36.9, 160.9, 160.9],
    [179.9, 179.9, 167.9, 167.9, 156.9, 156.9, 172.9, 172.9, 160.9, 160.9, 36.9, 148.9],
    [179.9, 179.9, 167.9, 167.9, 156.9, 156.9, 172.9, 172.9, 160.9, 160.9, 148.9, 36.9],
])

# rank -> SM, measured by the same program and identical across every launch.
SMID = [0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41]
TPC = [s // 2 for s in SMID]                 # 0 0 4 4 8 8 12 12 16 16 20 20
BASE = 148.9
W = {0: 31.0, 4: 19.0, 8: 8.0, 12: 24.0, 16: 12.0, 20: 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../slides/images/dsmem-matrix.png")
    ap.add_argument("--fit", action="store_true",
                    help="add a second panel: the matrix collapsed onto the line 148.9 + w_k + w_d")
    args = ap.parse_args()

    n = len(MATRIX)
    off = MATRIX.copy()
    np.fill_diagonal(off, np.nan)            # the diagonal is a local access, not a hop

    if args.fit:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.4),
                                       gridspec_kw={"width_ratios": [1.25, 1]})
    else:
        fig, ax1 = plt.subplots(figsize=(8.5, 6.4))

    # ---- left: the matrix itself ----
    im = ax1.imshow(off, cmap="viridis_r", vmin=148.9, vmax=211)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax1.text(j, i, "self", ha="center", va="center", fontsize=8, color="0.4")
            else:
                v = MATRIX[i, j]
                ax1.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                         color="white" if v > 185 else "black")
    ax1.set_xticks(range(n)); ax1.set_xticklabels(range(n), fontsize=13)
    ax1.set_yticks(range(n)); ax1.set_yticklabels(range(n), fontsize=13)
    ax1.set_xlabel("target rank"); ax1.set_ylabel("reader rank")
    ax1.set_title("every pair, one reader at a time")
    ax1.grid(False)
    for s in ax1.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label("cycles per load", fontsize=15)
    cb.ax.tick_params(labelsize=13)

    # ---- right (optional): the whole matrix collapsed onto one line ----
    xs, ys = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xs.append(W[TPC[i]] + W[TPC[j]])
            ys.append(MATRIX[i, j])
    if args.fit:
        ax2.scatter(xs, ys, s=90, zorder=3, edgecolor="white", linewidth=0.8)
        grid = np.array([min(xs), max(xs)])
        ax2.plot(grid, BASE + grid, color="0.35", ls="--", zorder=2,
                 label=f"$\\mathrm{{latency}} = {BASE} + w_k + w_d$")
        ax2.set_xlabel("$w_k + w_d$   (cycles)")
        ax2.set_ylabel("measured latency (cycles)")
        ax2.set_title("each endpoint pays its own cost")
        ax2.legend(frameon=False, fontsize=15, loc="upper left")

    for p in plotstyle.save(fig, args.out):
        print(f"wrote {p}")

    # the residual, so the claim in the slide is checkable
    resid = max(abs(y - (BASE + x)) for x, y in zip(xs, ys))
    print(f"max |measured - model| over {len(xs)} ordered pairs: {resid:.2f} cycles")


if __name__ == "__main__":
    main()
