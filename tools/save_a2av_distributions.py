# save_a2av_distributions.py — one PNG per axccl distribution: the size of every
# message, source -> destination, at 256 B (left, from the trace), and the time
# per exchange and the throughput (right), with the x axis limited to the sizes
# that distribution ran. Same data, medians and style as the "a2av: time per
# exchange" and "a2av: throughput" tabs of dgx-spark/throughput/plots.json.
#
# Run with the Python of sbatchman (it has pandas and matplotlib):
#     $PY tools/save_a2av_distributions.py SbatchMan/data.sqlite dgx-spark/throughput/traces slides/images/throughput
import csv, os, sqlite3, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter, NullLocator

HOUSE = {
    "font.size": 16, "axes.titlesize": 17, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 2.2, "lines.markersize": 6,
}
METHODS = {
    "dsmem-push": ("#1f77b4", "o"), "dsmem-pull": ("#17becf", "o"), "dsmem-bulk": ("#9467bd", "o"),
    "gmem-ldst": ("#ff7f0e", "s"), "gmem-tma": ("#d62728", "s"),
}
DISTRIBUTIONS = ["UNIMODALFLAT", "UNIMODALPEAK", "MONOTONEDECAY", "HEAVYTAIL"]
MATRIX_SIZE = 256   # the average message size of the matrix


def matrix(traces_dir, dist, avg_msgsize):
    """bytes[src][dst] of one trace at one average message size."""
    rows = [r for r in csv.DictReader(open(f"{traces_dir}/ALL2ALLV_{dist}_pairs.log"))
            if int(r["avg_msgsize"]) == avg_msgsize]
    p = int(rows[0]["nranks"])
    m = np.zeros((p, p))
    for r in rows:
        m[int(r["src_rank"]), int(r["dst_rank"])] = int(r["sendsize"])
    return m


db_path, traces_dir, out_dir = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)
# The exchange's columns are the same in every block's row: take rank 0's.
df = pd.read_sql("SELECT distribution, method, avg_msgsize, cycles_per_exchange, gbps FROM a2av "
                 "WHERE method != 'cluster-sync' AND rank = 0", sqlite3.connect(db_path))
df = df.groupby(["distribution", "method", "avg_msgsize"], as_index=False).median()

mpl.rcParams.update(HOUSE)
for dist in DISTRIBUTIONS:
    part = df[df["distribution"] == dist]
    sizes = sorted(part["avg_msgsize"].unique())
    fig = plt.figure(figsize=(20, 6.4), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1.15, 2])

    # ---- left: the size of every message, source -> destination
    m = matrix(traces_dir, dist, MATRIX_SIZE)
    ax = left.subplots()
    image = ax.imshow(m, cmap="viridis", vmin=0)
    ax.grid(False)
    for src in range(len(m)):
        for dst in range(len(m)):
            light = m[src, dst] < 0.6 * m.max()   # white text on the dark cells
            ax.text(dst, src, f"{m[src, dst]:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if light else "black")
    ax.set_xticks(range(len(m)))
    ax.set_yticks(range(len(m)))
    ax.tick_params(labelsize=11)
    ax.set_xlabel("destination block")
    ax.set_ylabel("source block")
    ax.set_title(f"bytes per message, average {MATRIX_SIZE} B")
    left.colorbar(image, ax=ax, label="bytes", format=FormatStrFormatter("%d"))

    # ---- right: time per exchange and throughput
    time_ax, gbps_ax = right.subplots(1, 2)
    for method, (color, marker) in METHODS.items():
        line = part[part["method"] == method].sort_values("avg_msgsize")
        time_ax.plot(line["avg_msgsize"], line["cycles_per_exchange"], color=color, marker=marker,
                     label=method)
        gbps_ax.plot(line["avg_msgsize"], line["gbps"], color=color, marker=marker)
    for ax in (time_ax, gbps_ax):
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(sizes))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{s}" if s < 1024 else f"{s // 1024} Ki" for s in sizes])
        ax.set_xlabel("average message size (bytes)")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    time_ax.yaxis.set_major_locator(FixedLocator([2000, 3000, 5000, 10000, 20000, 30000]))
    time_ax.set_ylabel("cycles per exchange")
    time_ax.set_title("time per exchange")
    gbps_ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100]))
    gbps_ax.set_ylabel("GB/s")
    gbps_ax.set_title("throughput")
    right.legend(loc="outside upper center", ncol=len(METHODS), frameon=False)

    path = os.path.join(out_dir, f"a2av-{dist.lower()}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(path)
