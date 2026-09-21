#!/usr/bin/env python3
"""Collect the latency results launched through SbatchMan and summarise them.

Run from the latency/ directory (SbatchMan looks for its project folder
upwards from the current directory):

    python analyze.py              # summary table
    python analyze.py --csv all.csv  # also dump every repetition to one CSV

Each job's stdout.log is a CSV with '#' metadata lines on top; all programs
use the same columns, so the logs concatenate into one DataFrame.
"""

import argparse
import sys

import pandas as pd
from sbatchman.config.project_config import get_experiments_dir
from sbatchman.core.jobs_manager import jobs_df


def load_all_runs() -> pd.DataFrame:
    jobs = jobs_df()
    if jobs.empty:
        sys.exit("no SbatchMan jobs found — did you run `sbatchman launch -f experiments.yaml`?")

    frames = []
    for job in jobs.itertuples():
        log = get_experiments_dir() / job.exp_dir / "stdout.log"
        if job.status != "COMPLETED" or not log.exists():
            print(f"skipping {job.tag}: status={job.status}", file=sys.stderr)
            continue
        df = pd.read_csv(log, comment="#")
        df["tag"] = job.tag
        frames.append(df)

    if not frames:
        sys.exit("no completed jobs with output")
    runs = pd.concat(frames, ignore_index=True)
    # Logs written before --stride existed have no stride column: they were
    # random, which is stride 0.
    if "stride_bytes" not in runs:
        runs["stride_bytes"] = 0
    runs["stride_bytes"] = runs["stride_bytes"].fillna(0).astype(int)
    if "chasers" not in runs:
        runs["chasers"] = 1
    runs["chasers"] = runs["chasers"].fillna(1).astype(int)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="also write every repetition to this file")
    args = ap.parse_args()

    runs = load_all_runs()
    if args.csv:
        runs.to_csv(args.csv, index=False)
        print(f"wrote {len(runs)} rows to {args.csv}", file=sys.stderr)

    keys = ["benchmark", "cluster_size", "distance", "mapped", "stride_bytes", "buffer_bytes", "chasers"]
    summary = (
        runs.groupby(keys)["cycles_per_load"]
        .agg(n="count", median="median", mean="mean", std="std",
             p95=lambda s: s.quantile(0.95))
        .reset_index()
    )
    # Present in the order a reader expects: cheapest memory first.
    order = {"smem_local": 0, "smem_cluster_local": 1, "dsmem_remote": 2, "l1": 3, "l2": 4, "dram": 5, "chain": 6,
             "smem_loaded": 6, "dsmem_loaded": 7}
    summary["_o"] = summary["benchmark"].map(order)
    summary = summary.sort_values(["_o", "cluster_size", "distance", "mapped", "stride_bytes", "buffer_bytes", "chasers"]).drop(columns="_o")

    pd.set_option("display.width", 120)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:8.2f}"))


if __name__ == "__main__":
    main()
