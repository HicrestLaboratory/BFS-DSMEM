#!/usr/bin/env python3
"""Latency vs. load for DSMEM, from the `loaded_*` SbatchMan jobs.

Run from latency/ after `sbatchman launch -f experiments.yaml -t 'loaded*'`:

    python plot_loaded.py            # writes loaded_latency.png and .pdf

Left panel: per-warp latency (mean over warps and repetitions, p95 as a thin
line) against chasing threads per requester, one line per number of
requesters. Right panel: aggregate throughput of the cluster. Together they
are the two halves of Little's law.
"""

import argparse
import sys

import plotstyle                      # noqa: F401  (house style, must precede pyplot use)
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from sbatchman.config.project_config import get_experiments_dir
from sbatchman.core.jobs_manager import jobs_df


def load_runs() -> pd.DataFrame:
    jobs = jobs_df()
    jobs = jobs[jobs["tag"].str.startswith("loaded_")] if not jobs.empty else jobs
    if jobs.empty:
        sys.exit("no loaded_* jobs — run: sbatchman launch -f experiments.yaml -t 'loaded*'")
    frames = []
    for job in jobs.itertuples():
        log = get_experiments_dir() / job.exp_dir / "stdout.log"
        if job.status != "COMPLETED" or not log.exists():
            print(f"skipping {job.tag}: status={job.status}", file=sys.stderr)
            continue
        df = pd.read_csv(log, comment="#")
        v = job.variables or {}
        df["requesters"] = int(v.get("requesters", df["cluster_size"].iloc[0] - 1))
        df["target"] = v.get("target", "remote")
        df["bank_aligned"] = int(v.get("bank_aligned", 1))
        df["tag"] = job.tag
        frames.append(df)
    if not frames:
        sys.exit("no completed loaded jobs")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="loaded_latency.png")
    ap.add_argument("--title", default="DSMEM under load: many readers, one owner")
    args = ap.parse_args()

    runs = load_runs()
    keys = ["target", "bank_aligned", "requesters", "chasers"]

    # Latency: every warp of every repetition is one sample.
    runs["total"] = runs["requesters"] * runs["chasers"]  # concurrent chasers, whole cluster
    keys = keys + ["total"]
    lat = (runs.groupby(keys)["cycles_per_load"]
               .agg(mean="mean", p95=lambda s: s.quantile(0.95), max="max")
               .reset_index())

    # Throughput per repetition: all loads over the slowest warp's wall time,
    # then the median over repetitions.
    per_rep = (runs.groupby(keys + ["tag", "rep"])
                   .agg(ns=("ns", "max"), steps=("steps", "first"), n=("warp", "count"))
                   .reset_index())
    per_rep["gbps"] = per_rep["requesters"] * per_rep["chasers"] * per_rep["steps"] * 4 / per_rep["ns"]
    thr = per_rep.groupby(keys)["gbps"].median().reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    styles = {("remote", 1): "-", ("local", 1): "--", ("remote", 0): ":"}
    for (tgt, ba, req), g in lat.groupby(["target", "bank_aligned", "requesters"]):
        g = g.sort_values("chasers")
        ls = styles.get((tgt, ba), "-.")
        lbl = f"{tgt}, {req} requester{'s' if req > 1 else ''}" + ("" if ba else ", random banks")
        line, = ax1.plot(g["total"], g["mean"], ls, marker="o", label=lbl)
        ax1.plot(g["total"], g["p95"], ls, color=line.get_color(), lw=0.8, alpha=0.5)
        t = thr[(thr.target == tgt) & (thr.bank_aligned == ba) & (thr.requesters == req)].sort_values("total")
        ax2.plot(t["total"], t["gbps"], ls, marker="o", color=line.get_color(), label=lbl)

    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("concurrent threads")
        ax.grid(True, which="both", alpha=0.3)
    ax1.set_ylabel("latency (cycles)")
    ax2.set_ylabel("throughput (GB/s)")
    ax1.set_title("Latency")
    ax2.set_title("Throughput")
    fig.suptitle(args.title, fontweight="bold")
    # Thin companion lines are the p95; say so once, in the legend.
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.35", lw=0.8, alpha=0.6))
    labels.append("thin line = p95")
    ax1.legend(handles, labels, frameon=False)

    for p in plotstyle.save(fig, args.out):
        print(f"wrote {p}")

    pd.set_option("display.width", 160)
    table = lat.merge(thr, on=keys)
    table["in_flight"] = table["gbps"] / 4 / 2.4 * table["mean"]   # loads/cycle x cycles
    print(table.sort_values(keys).to_string(index=False, float_format=lambda x: f"{x:8.1f}"))


if __name__ == "__main__":
    main()
