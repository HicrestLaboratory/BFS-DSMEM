#!/usr/bin/env python3
"""Latency vs. chain data volume, from the `chain_*` SbatchMan jobs.

Run from the latency/ directory after `sbatchman launch -f experiments.yaml -t 'chain*'`:

    python plot_chain.py                 # writes chain_latency.png and .pdf
    python plot_chain.py --out fig.png   # custom output name (extension picks the format)

The flat stretches are cache levels; the steps between them are the cache
sizes, measured.
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sbatchman.config.project_config import get_experiments_dir
from sbatchman.core.jobs_manager import jobs_df

L1_SRAM_KIB = 128          # unified L1 + shared memory per SM
L2_KIB = 24 * 1024         # 24 MiB


def load_runs() -> pd.DataFrame:
    jobs = jobs_df()
    jobs = jobs[jobs["tag"].str.startswith("chain_")] if not jobs.empty else jobs
    if jobs.empty:
        sys.exit("no chain_* jobs found — run: sbatchman launch -f experiments.yaml -t 'chain*'")
    frames = []
    for job in jobs.itertuples():
        log = get_experiments_dir() / job.exp_dir / "stdout.log"
        if job.status != "COMPLETED" or not log.exists():
            print(f"skipping {job.tag}: status={job.status}", file=sys.stderr)
            continue
        df = pd.read_csv(log, comment="#")
        frames.append(df)
    if not frames:
        sys.exit("no completed chain jobs")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="chain_latency.png")
    args = ap.parse_args()

    runs = load_runs()
    runs["kib"] = runs["buffer_bytes"] / 1024
    med = (runs.groupby("kib")["cycles_per_load"]
               .median().reset_index().sort_values("kib"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(med["kib"], med["cycles_per_load"], marker="o", ms=3.5, lw=1.4)

    ymax = med["cycles_per_load"].max()
    for x, name, yf in [(L1_SRAM_KIB, " 128 KiB\n L1+SMEM SRAM", 0.62), (L2_KIB, " 24 MiB\n L2", 0.27)]:
        ax.axvline(x, color="gray", ls=":", lw=1)
        ax.text(x, ymax * yf, name, ha="left", va="top", fontsize=8, color="gray")

    ax.set_xscale("log", base=2)
    ticks = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
    ax.set_xticks([t for t in ticks if med["kib"].min() <= t <= med["kib"].max()])
    ax.set_xticklabels([f"{t} KiB" if t < 1024 else f"{t // 1024} MiB" for t in ax.get_xticks()],
                       rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("chain data volume (KB)")
    ax.set_ylabel("latency (cycles per load)")
    ax.set_title("")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    fig.savefig(args.out, dpi=160)
    stem = args.out.rsplit(".", 1)[0]
    fig.savefig(stem + ".pdf")
    print(f"wrote {args.out} and {stem}.pdf")

    # The numbers behind the picture, for the text.
    pd.set_option("display.width", 140)
    print(med.set_index("kib")["cycles_per_load"].round(1).to_string())


if __name__ == "__main__":
    main()
