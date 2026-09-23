#!/usr/bin/env python3
"""all-vs-1 versus all-vs-all: does it matter WHO you read from?

    python plot_pattern.py [--out ../../slides/images/dsmem-pattern.png]

Left:  aggregate DSMEM throughput against offered load, one line per pattern.
Right: throughput against the number of owners the traffic is spread over,
       which separates the per-owner limit from the fabric-wide one.

Numbers from ./latency-many-threads (cluster 12, dependent chase, 4 B loads,
--bank-conflict 1). Re-run and paste if the method changes.
"""
import argparse
import numpy as np
import plotstyle                      # noqa: F401
import matplotlib.pyplot as plt

THREADS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
GBPS = {                              # aggregate GB/s, cluster size 12
    "hotspot": [0.50, 0.64, 0.64, 0.65, 0.67, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71],
    "random":  [0.60, 1.19, 2.24, 2.58, 2.94, 3.34, 3.57, 3.85, 3.59, 3.46, 3.62],
    "ring":    [0.55, 1.08, 2.06, 3.07, 3.68, 4.35, 4.46, 4.45, 4.47, 4.46, 4.43],
}
READERS  = [1, 2, 3, 4, 6, 8, 10, 12]          # cluster size sweep, 128 threads
BY_OWNERS = {
    "hotspot": [0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71, 0.71],
    "ring":    [1.41, 2.12, 2.81, None, 4.16, 4.43, 4.45, 4.45],
}
LABEL = {"hotspot": "all vs 1  (one owner)",
         "ring":    "all vs all  (one owner each)",
         "random":  "random  (per-warp target)"}
COLOR = {"hotspot": "#d62728", "ring": "#2ca02c", "random": "#1f77b4"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../slides/images/dsmem-pattern.png")
    args = ap.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6))

    for p in ("hotspot", "random", "ring"):
        ax1.plot(THREADS, GBPS[p], marker="o", color=COLOR[p], label=LABEL[p])
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([1, 4, 16, 64, 256, 1024])
    ax1.set_xticklabels([1, 4, 16, 64, 256, 1024])
    ax1.set_xlabel("threads per SM")
    ax1.set_ylabel("throughput (GB/s)")
    ax1.set_title("spreading only pays once you are loaded")
    ax1.legend(frameon=False, fontsize=13, loc="upper left")
    # the four points the professor asked for
    ax1.set_ylim(0.0, 5.1)
    for p, t, dx, dy in (("hotspot", 1, 16, -6), ("hotspot", 128, 0, -24),
                         ("ring", 1, 8, 10), ("ring", 128, 0, 12)):
        i = THREADS.index(t)
        ax1.annotate(f"{GBPS[p][i]:.2f}", (t, GBPS[p][i]), textcoords="offset points",
                     xytext=(dx, dy), ha="center", fontsize=14,
                     color=COLOR[p], fontweight="bold")

    for p in ("ring", "hotspot"):
        xs = [r for r, v in zip(READERS, BY_OWNERS[p]) if v is not None]
        ys = [v for v in BY_OWNERS[p] if v is not None]
        ax2.plot(xs, ys, marker="o", color=COLOR[p], label=LABEL[p])
    ax2.axhline(4.45, color="0.4", ls="--", lw=1.5)
    ax2.set_ylim(0.0, 5.1)
    ax2.text(6.3, 4.05, "fabric ceiling  4.45 GB/s", color="0.35", fontsize=13)
    ax2.plot([1, 6], [0.71, 4.26], ls=":", color="0.55", lw=1.5)
    ax2.text(4.4, 1.35, "linear:\n0.71 GB/s per owner", color="0.45", fontsize=13, ha="center")
    ax2.set_xlabel("owners the traffic is spread over")
    ax2.set_ylabel("throughput (GB/s)")
    ax2.set_title("two ceilings: per owner, then the fabric")
    ax2.set_xticks(READERS)
    ax2.legend(frameon=False, fontsize=13, loc="lower right")

    for p in plotstyle.save(fig, args.out):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
