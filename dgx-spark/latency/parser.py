#!/usr/bin/env python3
"""Parser for `sbatchman visualize`: the stride sweep, one series per component.

`visualize` calls parse() once per job and collects the returned rows into a
SQLite database that the web UI plots. Run it from this directory:

    sbatchman visualize

Only the stride-sweep jobs are picked up, i.e. the ones whose tag ends in
`_s<stride>` (dram_s64, dsmem_cs2_d1_s4096, l2_srandom, ...). Everything else
(the cluster-size and distance sweeps) is skipped; widen STRIDE_TAG if you
want those in the same database.

Two tables come out:

  stride_runs     one row per repetition, for box/violin plots of the spread
                  across reps.
  stride_summary  one row per job: median/mean/std/min/p05/p95 across the reps.
                  This is the one to plot -- x = stride_bytes (log scale),
                  y = median_cycles_per_load, group by `variant`.

The random-stride jobs have no position on a log x-axis, so they carry
is_random = 1: filter them out with `WHERE is_random = 0` for the line plot
and show them separately as a bar chart.
"""

import io
import re

import pandas as pd

import sbatchman as sbm

# Tags look like <variant>_s<stride>: dsmem_cs2_d1_s4096, dram_srandom, l2_s8.
STRIDE_TAG = re.compile(r"^(?P<variant>.+)_s(?P<stride>\d+|random)$")

# Cheapest memory first, so the legend reads in the order the thesis does.
COMPONENT_ORDER = {
  "smem_local": 0,
  "smem_cluster_local": 1,
  "dsmem_remote": 2,
  "l2": 3,
  "dram": 4,
}

# Aggregated in stride_summary as <stat>_<metric>, e.g. median_cycles_per_load.
METRICS = ["cycles_per_load", "ns_per_load", "cycles", "ns"]

# Copied verbatim onto every stride_runs row.
PER_REP_COLUMNS = [
  "cluster_size", "distance", "mapped", "block_size", "steps", "buffer_bytes",
  "seed", "rep", "cycles", "ns", "cycles_per_load", "ns_per_load", "ghz",
]

# Constant within a job, so stride_summary keeps a single copy of each.
PER_JOB_COLUMNS = [
  "cluster_size", "distance", "mapped", "block_size", "steps", "buffer_bytes",
]


def _status(job) -> str:
  """job.status is a plain string in the metadata but a Status enum in memory."""
  return getattr(job.status, "value", job.status)


def _header(text: str) -> dict:
  """The `# key: value` preamble the benchmarks print above the CSV."""
  meta = {}
  for line in text.splitlines():
    if not line.startswith("#"):
      break
    key, _, value = line[1:].partition(":")
    if value:
      meta[key.strip()] = value.strip()
  return meta


def _int(meta: dict, key: str):
  try:
    return int(meta[key])
  except (KeyError, TypeError, ValueError):
    return None


def parse(job: sbm.Job):
  match = STRIDE_TAG.match(job.tag or "")
  if not match or _status(job) != "COMPLETED":
    return None

  text = job.get_stdout()
  if not text:
    return None

  runs = pd.read_csv(io.StringIO(text), comment="#")
  if runs.empty:
    return None

  meta = _header(text)
  stride = match.group("stride")
  is_random = stride == "random"
  first = runs.iloc[0]

  # Identifies the point on the plot; shared by both tables so they join.
  common = {
    "tag": job.tag,
    "variant": match.group("variant"),
    "component": str(first["benchmark"]),
    "stride_label": stride,
    # The CSV reports stride_bytes = 0 for the random walk; is_random is what
    # actually tells a random run apart from a (nonexistent) zero stride.
    "stride_bytes": 0 if is_random else int(stride),
    "is_random": int(is_random),
    "job_id": job.job_id,
  }
  common["component_order"] = COMPONENT_ORDER.get(common["component"], 99)

  per_rep = []
  for record in runs.to_dict("records"):
    row = dict(common)
    row.update({c: record[c] for c in PER_REP_COLUMNS if c in record})
    per_rep.append(row)

  summary = dict(common)
  summary.update({c: int(first[c]) for c in PER_JOB_COLUMNS if c in runs})
  summary.update({
    "reps": len(runs),
    "ghz": float(runs["ghz"].median()),
    # cycle_length and buffer_elems say how long the pointer chase actually
    # was, which is what makes an L2 or DRAM number believable.
    "cycle_length": _int(meta, "cycle_length"),
    "buffer_elems": _int(meta, "buffer_elems"),
    "gpu": meta.get("gpu"),
  })
  for metric in METRICS:
    values = runs[metric]
    summary[f"median_{metric}"] = float(values.median())
    summary[f"mean_{metric}"] = float(values.mean())
    summary[f"std_{metric}"] = float(values.std())
    summary[f"min_{metric}"] = float(values.min())
    # A few reps finish in half the time (timer glitch); p05 hides them.
    summary[f"p05_{metric}"] = float(values.quantile(0.05))
    summary[f"p95_{metric}"] = float(values.quantile(0.95))

  return {"stride_runs": per_rep, "stride_summary": summary}
