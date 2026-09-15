#!/usr/bin/env python3
"""Plot latency vs stride straight from the SbatchMan jobs, without the web UI.

Run from latency/ with the interpreter that has sbatchman installed:

    ~/tools/SbatchMan/.venv/bin/python plot_stride.py                 # all components
    ~/tools/SbatchMan/.venv/bin/python plot_stride.py dram            # dram_s4 ... dram_s4096
    ~/tools/SbatchMan/.venv/bin/python plot_stride.py dram l2 --metric ns_per_load -o dram_l2.png

Each positional argument is a variant, i.e. the part of the tag before
`_s<stride>` (dram, l2, smem_local, smem_cluster_local_cs2_m1, dsmem_cs2_d1).
Jobs are selected through SbatchMan's own tag glob and reduced with the
parse() from parser.py, the same function `sbatchman visualize` runs, so the
two views always agree.

The random-stride run has no place on a log x-axis; it is drawn as a dashed
horizontal line in the series' colour instead.
"""

import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import pandas as pd
import sbatchman as sbm

from parser import STRIDE_TAG, parse

LABELS = {
    "cycles_per_load": "latency per load (cycles)",
    "ns_per_load": "latency per load (ns)",
}


def stride_summary(job: sbm.Job) -> dict:
    result = parse(job)
    return result["stride_summary"] if result else {}


def load_summary(variants: list[str]) -> pd.DataFrame:
    df = sbm.jobs_to_dataframe(
        tag="*_s*",
        job_filter=lambda job: bool(STRIDE_TAG.match(job.tag or "")),
        extractors=[stride_summary],
        include_job_variables=False,
    )
    # Non-COMPLETED jobs make parse() return None, which leaves an empty row.
    if df.empty or "tag" not in df:
        sys.exit("no completed stride jobs found — run `sbatchman launch -f experiments.yaml` first")
    df = df.dropna(subset=["tag"])
    if variants:
        missing = set(variants) - set(df["variant"])
        if missing:
            sys.exit(f"no jobs for variant(s): {', '.join(sorted(missing))}\n"
                     f"available: {', '.join(sorted(df['variant'].unique()))}")
        df = df[df["variant"].isin(variants)]
    return df.sort_values(["component_order", "is_random", "stride_bytes"])


def plot(df: pd.DataFrame, metric: str, logy: bool):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    median = f"median_{metric}"

    for variant, g in df.groupby("variant", sort=False):
        strided = g[g["is_random"] == 0]
        (line,) = ax.plot(strided["stride_bytes"], strided[median],
                          marker="o", markersize=4, label=variant)
        random = g[g["is_random"] == 1]
        if not random.empty:
            ax.axhline(random[median].iloc[0], linestyle="--", linewidth=1,
                       color=line.get_color(), alpha=0.6)

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(df.loc[df["is_random"] == 0, "stride_bytes"].unique()))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("stride (bytes)")
    ax.set_ylabel(LABELS.get(metric, metric))
    ax.set_title("median over reps; dashed = random stride")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variants", nargs="*", help="variants to plot (default: all)")
    ap.add_argument("--metric", default="cycles_per_load", choices=sorted(LABELS))
    ap.add_argument("--logy", action="store_true", help="log scale on the y axis too")
    ap.add_argument("-o", "--out", default="stride.png", help="output image (default: stride.png)")
    ap.add_argument("--show", action="store_true", help="open a window instead of only saving")
    args = ap.parse_args()

    df = load_summary(args.variants)

    table = df[df["is_random"] == 0].pivot(index="variant", columns="stride_bytes", values=f"median_{args.metric}")
    table = table.loc[df["variant"].unique()]  # keep component order
    pd.set_option("display.width", 160)
    print(f"median {args.metric}:")
    print(table.round(1).to_string())

    fig = plot(df, args.metric, args.logy)
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}", file=sys.stderr)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
