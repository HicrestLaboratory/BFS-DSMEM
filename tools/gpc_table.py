# gpc_table.py — the table "cost by TPC position, for each GPC" of the slides,
# as LaTeX, from the database: cluster.sync(), relaxed barrier and fence from
# the fence_placement table (build/fence placement --cluster-size 1) and the L2
# load from the l2_topology table, for every TPC position of every GPC. Every
# number is the mean of the TPC's two SMs.
#
# GPCs and positions come from the measurement (topology.cuh), so it works for
# any number of GPCs. If the barrier and fence columns are the same in every GPC,
# as on GB10, it prints one compact table with one L2 column per GPC; otherwise
# one row per (GPC, position). Rows go from the most expensive cluster.sync() to
# the cheapest.
#
# Run with the Python of sbatchman (it has pandas):
#     $PY tools/gpc_table.py SbatchMan/data.sqlite
import sqlite3
import sys

import pandas as pd

PIECES = [("cluster.sync()", "sync"), ("relaxed arrive + wait", "relaxed"),
          ("fence.acq_rel.gpu", "fence")]

db = sqlite3.connect(sys.argv[1])
sm = pd.read_sql("SELECT smid AS sm, gpc, tpc_pos AS pos, median AS l2 FROM l2_topology", db)
for piece, column in PIECES:
    part = pd.read_sql(f"SELECT sm, cycles AS {column} FROM fence_placement WHERE piece = '{piece}'", db)
    sm = sm.merge(part, on="sm")
# One row per TPC: the mean of its two SMs.
tpc = sm.groupby(["gpc", "pos"], as_index=False)[["l2", "sync", "relaxed", "fence"]].mean().round().astype(int)
gpcs = sorted(tpc["gpc"].unique())

# Are the barrier and fence columns the same in every GPC?
same = all(tpc.groupby("pos")[c].nunique().max() == 1 for c in ("sync", "relaxed", "fence")) \
       and tpc.groupby("pos")["gpc"].nunique().min() == len(gpcs)

if same:
    rows = tpc[tpc["gpc"] == gpcs[0]].sort_values("sync", ascending=False)
    print(r"\begin{tabular}{@{}r@{\hspace{1.8em}}rrr@{\hspace{1.8em}}" + "r" * len(gpcs) + "@{}}")
    print(r"  \toprule")
    print(r"  & \multicolumn{3}{c}{\textbf{same in all %d GPCs}} & \multicolumn{%d}{c}{\textbf{L2 load, depends on the GPC}} \\"
          % (len(gpcs), len(gpcs)))
    print(r"  \cmidrule(r{1.2em}){2-4}\cmidrule{5-%d}" % (4 + len(gpcs)))
    print(r"  TPC pos. & \texttt{cluster.sync()} & relaxed & fence & "
          + " & ".join(f"GPC {g}" for g in gpcs) + r" \\")
    print(r"  \midrule")
    for _, r in rows.iterrows():
        l2 = [str(tpc[(tpc["gpc"] == g) & (tpc["pos"] == r["pos"])]["l2"].iloc[0]) for g in gpcs]
        print(f"  {r['pos']} & {r['sync']} & {r['relaxed']} & {r['fence']} & " + " & ".join(l2) + r" \\")
    print(r"  \bottomrule")
    print(r"\end{tabular}")
else:
    print("% barrier and fence differ between GPCs: one row per (GPC, TPC position)")
    print(r"\begin{tabular}{@{}rrrrrr@{}}")
    print(r"  \toprule")
    print(r"  GPC & TPC pos. & \texttt{cluster.sync()} & relaxed & fence & L2 load \\")
    print(r"  \midrule")
    for g in gpcs:
        for _, r in tpc[tpc["gpc"] == g].sort_values("sync", ascending=False).iterrows():
            print(f"  {g} & {r['pos']} & {r['sync']} & {r['relaxed']} & {r['fence']} & {r['l2']} \\\\")
        print(r"  \midrule" if g != gpcs[-1] else r"  \bottomrule")
    print(r"\end{tabular}")
