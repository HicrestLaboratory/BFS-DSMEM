#!/usr/bin/env python3
"""Aggregate DSMEM throughput vs cluster size, from the `tp_*` SbatchMan jobs.

Run from latency/ after `sbatchman launch -f experiments.yaml -t 'tp_*'`:

    ~/tools/SbatchMan/.venv/bin/python plot_throughput.py

Writes one grouped bar chart per pattern (broadcast, ring) and scope into
../../slides/images/, two bars per cluster size: random vs coalesced access.
    throughput-<pattern>.png       one cluster        (tp_<pattern>_...,     dsmem_many_threads)
    throughput-<pattern>-gpu.png   48/cs clusters     (tp_gpu_<pattern>_..., dsmem_many_threads_gpu)

One cluster: throughput is recomputed from the per-warp CSV rows. Each row is
one warp of 32 lanes, each lane made `steps` 4-byte loads, and every warp
started at the same cluster.sync(), so a repetition's wall time is its
slowest warp's `ns`. Bytes / wall time = GB/s; the bar is the median over
repetitions.

Whole GPU: different clusters can start microseconds apart, so the slowest
warp's `ns` is not the wall time. The program measures it from absolute
timestamps (first start to last end) and prints the median over repetitions
on stderr; that number is used as is.
"""

import argparse
import glob
import io
import os
import re
import sys

import plotstyle                      # noqa: F401  (house style, must precede pyplot use)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TAG = re.compile(r"^tp_(?:(?P<gpu>gpu)_)?(?P<pattern>broadcast|ring)_(?P<access>random|coalesced)"
                 r"_cs(?P<cs>\d+)$")
STDERR_GBPS = re.compile(r"GB/s total .*median=([0-9.]+)")
ACCESS = ["random", "coalesced"]
COLORS = {"random": "#4FC3F7", "coalesced": "#EF5350"}   # myblue / myred in the slides
EDGES = {"random": "#0288D1", "coalesced": "#B71C1C"}


def latest_run(tag_dir: str) -> str | None:
    runs = sorted(glob.glob(os.path.join(tag_dir, "*", "stdout.log")))
    return runs[-1] if runs else None


def gbps(log: str) -> float:
    df = pd.read_csv(log, comment="#")
    per_rep = df.groupby("rep").agg(rows=("ns", "size"), wall=("ns", "max"),
                                    steps=("steps", "first"))
    bytes_ = per_rep["rows"] * 32 * per_rep["steps"] * 4
    return float((bytes_ / per_rep["wall"]).median())       # bytes/ns = GB/s


def gbps_gpu(log: str) -> float:
    stderr = os.path.join(os.path.dirname(log), "stderr.log")
    m = STDERR_GBPS.search(open(stderr).read())
    return float(m.group(1)) if m else float("nan")


def load() -> pd.DataFrame:
    rows = []
    for tag_dir in glob.glob("SbatchMan/experiments/*/*/tp_*"):
        m = TAG.match(os.path.basename(tag_dir))
        log = latest_run(tag_dir)
        if not m or not log or os.path.getsize(log) == 0:
            continue
        scope = "gpu" if m["gpu"] else "cluster"
        rows.append({"scope": scope, "pattern": m["pattern"], "access": m["access"],
                     "cluster_size": int(m["cs"]),
                     "gbps": gbps_gpu(log) if scope == "gpu" else gbps(log)})
    if not rows:
        sys.exit("no tp_* runs found — run: sbatchman launch -f experiments.yaml -t 'tp_*'")
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, scope: str, pattern: str, sizes: list[int]):
    plt.rcParams.update({"font.size": 22, "axes.labelsize": 22,
                         "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 18})
    fig, ax = plt.subplots(figsize=(10, 5.2))
    d = df[(df["scope"] == scope) & (df["pattern"] == pattern) & (df["cluster_size"].isin(sizes))]
    x = np.arange(len(sizes))
    width = 0.36
    for k, acc in enumerate(ACCESS):
        vals = [d[(d["access"] == acc) & (d["cluster_size"] == cs)]["gbps"].median()
                for cs in sizes]
        bars = ax.bar(x + (k - 0.5) * width, vals, width, label=acc,
                      color=COLORS[acc], edgecolor=EDGES[acc], linewidth=1.2)
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}" if v >= 1 else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(cs) for cs in sizes])
    ax.set_xlabel("cluster size", labelpad=8)
    ax.set_ylabel("throughput (GB/s)", labelpad=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)                 # room for the value labels
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster-sizes", type=int, nargs="+", default=[2, 4, 6, 12])
    ap.add_argument("--outdir", default="../../slides/images")
    args = ap.parse_args()

    df = load()
    pd.set_option("display.width", 120)
    print(df.pivot_table(index=["scope", "pattern", "access"], columns="cluster_size",
                         values="gbps").round(2).to_string())
    for scope, suffix in (("cluster", ""), ("gpu", "-gpu")):
        if not (df["scope"] == scope).any():
            continue
        for pattern in ["broadcast", "ring"]:
            fig = plot(df, scope, pattern, args.cluster_sizes)
            out = os.path.join(args.outdir, f"throughput-{pattern}{suffix}.png")
            for p in plotstyle.save(fig, out):
                print(f"wrote {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
