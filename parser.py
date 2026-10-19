#!/usr/bin/env python3
"""Parser for `sbatchman visualize`: every latency job, as raw rows.

Run from this directory (it picks up ./parser.py by default):

    sbatchman visualize

One table comes out, `latency`: one row per repetition of every completed
latency job (smem, l1, l2, dram, dsmem), holding exactly what the program
measured. There are no statistics and no filtering here on purpose: medians,
means, selecting a sweep, dropping or keeping dsmem's self-accesses, ... are
all done in the pandas transforms of the visualize tabs, where they are
visible and easy to change.

Columns
  tag            SbatchMan tag of the job, e.g. l1/buffer_kib/32 or
                 dsmem/cluster_size_2/stride/64: tells which sweep a row
                 belongs to, so a transform can select it.
  component      what was measured: smem, l1, l2, dram or dsmem (the
                 `benchmark` column the programs write).
  stride_bytes   0 = random cycle, otherwise the fixed stride in bytes.
  buffer_kib     size of the chase buffer in KiB.
  cluster_size, reader, target
                 dsmem: the cluster size, the rank that read and the rank
                 whose shared memory it read (reader == target is a rank
                 reading its own shared memory).
  block_size, steps, seed, rep
  cycles, ns, cycles_per_load, ns_per_load, ghz

Every value is copied from the program's CSV output. A column a program does
not write (e.g. buffer_kib for dsmem) is left empty.
"""

import io

import pandas as pd

import sbatchman as sbm

# The `benchmark` names written by the latency programs. Jobs of the other
# experiments (throughput, transfer, ...) write different CSVs and are skipped.
LATENCY_COMPONENTS = {"smem", "l1", "l2", "dram", "dsmem"}

# Copied from the CSV, in this order, after `tag` and `component`.
COLUMNS = [
  "stride_bytes", "buffer_kib", "cluster_size", "reader", "target",
  "block_size", "steps", "seed", "rep",
  "cycles", "ns", "cycles_per_load", "ns_per_load", "ghz",
]


def _status(job) -> str:
  """job.status is a plain string in the metadata but a Status enum in memory."""
  return getattr(job.status, "value", job.status)


def parse(job: sbm.Job):
  if _status(job) != "COMPLETED":
    return None
  text = job.get_stdout()
  if not text:
    return None

  # The programs print '#' metadata lines, then a CSV with one row per repetition.
  runs = pd.read_csv(io.StringIO(text), comment="#")
  if runs.empty or "benchmark" not in runs:
    return None
  if runs["benchmark"].iloc[0] not in LATENCY_COMPONENTS:
    return None

  rows = []
  for record in runs.to_dict("records"):
    row = {"tag": job.tag, "component": record["benchmark"]}
    for column in COLUMNS:
      row[column] = record.get(column)   # None if this program does not write it
    rows.append(row)
  return {"latency": rows}
