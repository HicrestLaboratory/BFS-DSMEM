#!/usr/bin/env python3
"""DSMEM against the traditional copy-back, from the `transfer_*` and `kboundary_*` jobs.

Run from latency/ after
    sbatchman launch -f experiments.yaml -t 'transfer*'
    sbatchman launch -f experiments.yaml -t 'kboundary*'

    python plot_transfer.py                   # pair 10 -> 11, writes transfer.png and .pdf
    python plot_transfer.py --pair 0-1        # the slowest DSMEM pair instead

Left:  one-way latency of an S-byte message against S (ping-pong), one line per
       method, with the kernel-boundary baseline (one launch per BFS level)
       and one cluster-wide barrier for scale.
Right: sustained bandwidth (stream mode) against S.
The summary table and the crossover sizes are printed to stdout.
"""

import argparse
import sys

import plotstyle                      # noqa: F401  (house style, must precede pyplot use)
import matplotlib.pyplot as plt
import pandas as pd
import sbatchman as sbm

METHODS = ["dsmem-pull", "dsmem-push", "dsmem-bulk", "gmem-ldst", "gmem-tma"]
LABEL = {"dsmem-pull": "DSMEM pull (loads)",
         "dsmem-push": "DSMEM push (stores)",
         "dsmem-bulk": "DSMEM bulk copy",
         "gmem-ldst": "GMEM copy-back (ld/st)",
         "gmem-tma": "GMEM copy-back (TMA)"}
COLOR = {"dsmem-pull": "#1f77b4", "dsmem-push": "#17becf", "dsmem-bulk": "#2ca02c",
         "gmem-ldst": "#d62728", "gmem-tma": "#ff7f0e"}
STYLE = {m: ("-" if m.startswith("dsmem") else "--") for m in METHODS}


def load(prefix: str) -> pd.DataFrame:
    frames = []
    for job in sbm.jobs_list(tag=prefix + "*"):
        status = getattr(job.status, "value", job.status)
        log = job.get_stdout_path()
        if status != "COMPLETED" or not log.exists():
            print(f"skipping {job.tag}: status={status}", file=sys.stderr)
            continue
        frames.append(pd.read_csv(log, comment="#"))
    if not frames:
        sys.exit(f"no completed {prefix}* jobs -- see the docstring for how to run them")
    return pd.concat(frames, ignore_index=True)


def crossover(med: pd.DataFrame, col: str, better) -> str:
    """Smallest size at which the best GMEM method beats the best DSMEM method."""
    dsm = med[med.method.str.startswith("dsmem")].groupby("bytes")[col]
    gm = med[med.method.str.startswith("gmem")].groupby("bytes")[col]
    best_d = dsm.min() if better == "low" else dsm.max()
    best_g = gm.min() if better == "low" else gm.max()
    wins = (best_g < best_d) if better == "low" else (best_g > best_d)
    sizes = [b for b in sorted(wins.index) if b > 0 and wins[b]]
    return f"{sizes[0]} B" if sizes else "never (up to 32 KiB)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="10-11", help="src-dst ranks, e.g. 10-11 or 0-1")
    ap.add_argument("--out", default="transfer.png")
    args = ap.parse_args()
    src, dst = (int(x) for x in args.pair.split("-"))

    t = load("transfer_")
    t = t[(t.src == src) & (t.dst == dst)]
    med = (t.groupby(["method", "mode", "bytes"])
             .agg(ns=("ns_per_msg", "median"), cy=("cycles_per_msg", "median"),
                  gbps=("gbps", "median"))
             .reset_index())
    kb = load("kboundary_")
    kbm = kb.groupby(["variant", "bytes"])["ns_per_level"].median().reset_index()
    csync = med[(med.method == "cluster-sync")]["ns"]

    pp = med[(med["mode"] == "pingpong") & (med.bytes > 0)]
    sm = med[(med["mode"] == "stream") & (med.bytes > 0)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.4))
    for m in METHODS:
        g = pp[pp.method == m].sort_values("bytes")
        ax1.plot(g.bytes, g.ns, STYLE[m], marker="o", color=COLOR[m], label=LABEL[m])
        g = sm[sm.method == m].sort_values("bytes")
        ax2.plot(g.bytes, g.gbps, STYLE[m], marker="o", color=COLOR[m], label=LABEL[m])
    for v, ls in (("graph", ":"), ("stream", "-.")):
        g = kbm[kbm.variant == v].sort_values("bytes")
        ax1.plot(g.bytes, g.ns_per_level, ls, color="0.35", lw=1.8,
                 label=f"kernel boundary ({'CUDA graph' if v == 'graph' else 'launches'})")
    if len(csync):
        y = float(csync.iloc[0])
        ax1.axhline(y, color="0.6", lw=1.2, ls=(0, (1, 3)))
        ax1.text(4096, y * 1.07, f"one cluster.sync(): {y:.0f} ns", color="0.45", fontsize=13,
                 ha="center")
        ax1.set_ylim(bottom=y * 0.8)

    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 64, 256, 1024, 4096, 16384])
        ax.set_xticklabels(["16 B", "64 B", "256 B", "1 KiB", "4 KiB", "16 KiB"])
        ax.set_xlabel("message size")
    ax1.set_yscale("log")
    ax1.set_ylabel("one-way time (ns)")
    ax1.set_title("latency: synchronization dominates")
    ax2.set_ylabel("sustained bandwidth (GB/s)")
    ax2.set_title("bandwidth: copy-back wins large messages")
    # One legend for both panels, below them: the curves fill the top-left
    # corner of the latency panel, where a legend would hide the baselines.
    h1, l1 = ax1.get_legend_handles_labels()
    fig.legend(h1, l1, loc="outside lower center", ncol=4, frameon=False, fontsize=14)

    for p in plotstyle.save(fig, args.out):
        print(f"wrote {p}")

    # ---- the numbers behind the figure ----
    print(f"\npair {src} -> {dst}: one-way ns (ping-pong) / peak GB/s (stream)")
    cols = [16, 128, 1024, 32768]
    print(f"{'method':<14}" + "".join(f"{str(c) + ' B':>10}" for c in cols) + f"{'peak GB/s':>11}")
    for m in METHODS:
        row = pp[pp.method == m].set_index("bytes").ns
        peak = sm[sm.method == m].gbps.max()
        print(f"{m:<14}" + "".join(f"{row.get(c, float('nan')):>10.0f}" for c in cols)
              + f"{peak:>11.2f}")
    for v in ("stream", "graph", "host-check"):
        row = kbm[kbm.variant == v].set_index("bytes").ns_per_level
        print(f"{'kb-' + v:<14}" + "".join(f"{row.get(c, float('nan')):>10.0f}" for c in cols))
    flag = med[(med.bytes == 0) & (med["mode"] == "pingpong")].set_index("method").ns
    print("flag only (0 B) one-way ns: " + ", ".join(f"{k} {v:.0f}" for k, v in flag.items()))
    print(f"GMEM beats DSMEM in latency from:   {crossover(pp, 'ns', 'low')}")
    print(f"GMEM beats DSMEM in bandwidth from: {crossover(sm, 'gbps', 'high')}")


if __name__ == "__main__":
    main()
