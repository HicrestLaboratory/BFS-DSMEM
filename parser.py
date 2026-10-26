#!/usr/bin/env python3
"""Parser for `sbatchman visualize`: every job, as raw rows.

Run from this directory (it picks up ./parser.py by default):

    sbatchman visualize

One row per repetition of every completed job, holding exactly what the
program measured, in one table per kind of experiment. There are no
statistics and no filtering here on purpose: medians, means, selecting a
sweep, ... are all done in the pandas transforms of the visualize tabs, where
they are visible and easy to change.

Every value is copied from the program's CSV output. A column a program does
not write (e.g. buffer_kib for dsmem) is left empty.

Table `latency` (dgx-spark/latency/: smem, l1, l2, dram, dsmem)
  tag            SbatchMan tag of the job, e.g. l1_buffer_kib_32.
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

Table `a2av` (dgx-spark/throughput/a2av.cu: an axccl all-to-all-v pattern
replayed among the blocks of one cluster)
  tag            SbatchMan tag of the job.
  method         how the bytes move: dsmem-push, dsmem-pull, dsmem-bulk,
                 gmem-ldst, gmem-tma, or cluster-sync (no data: the floor).
  distribution   the axccl imbalance: UNIMODALFLAT, UNIMODALPEAK,
                 HEAVYTAIL or MONOTONEDECAY.
  nranks         blocks in the cluster (= MPI ranks of the trace).
  avg_msgsize    average message size of the replayed pattern, in bytes.
  total_bytes, max_sent_bytes, max_received_bytes
                 bytes of one exchange in total, and of the block that sends
                 / receives the most.
  block_size, rounds, seed, rep
  ns, cycles     whole repetition (`rounds` exchanges), timed by rank 0.
  ns_per_exchange, cycles_per_exchange, gbps
                 one exchange: time, and total_bytes / time.
  busy_max_ns, busy_mean_ns
                 time the blocks spent moving bytes in one exchange: the
                 slowest block, and the mean over blocks.

Table `smem_bandwidth` (dgx-spark/throughput/smem_bandwidth.cu: bandwidth of
an SM's own shared memory, one block per SM on every SM)
  tag, op (read | write), width (bytes per access), stride_bytes (between
  consecutive threads; = width means no bank conflicts), block_size, blocks
  (= SMs), steps (accesses per thread), seed, rep
  bytes_per_block, cycles, ns   the median block
  bytes_per_cycle_per_sm, gbps_per_sm   the median block
  gbps_chip      all blocks' bytes over the slowest block's time.

Table `barrier` (dgx-spark/barriers/barrier.cu: the cost of one barrier)
  One row per BARRIER, not per repetition: the program times each one.
  tag, barrier (syncthreads | cluster | cluster-relaxed | grid), block_size,
  blocks, cluster_size, rounds (barriers per repetition), seed, rep,
  round (which barrier of the repetition)
  cycles         that one barrier, timed by thread 0 of block 0

Table `atomic` (dgx-spark/atomics/atomic.cu: atomicOr / atomicAdd / atomicCAS)
  tag, memory (throughput: smem | dsmem-rank0 | dsmem-rank10 |
  dsmem-rank10-spread | global; latency: smem | dsmem | global),
  mode (latency | throughput), op (or | add | cas), contention (threads
  per word), layout (consecutive | scattered; "-" for latency), distance
  (dsmem latency: target rank), block_size, workers, steps, seed, rep
  atomics        how many were done in total
  ns, cycles     latency: the chain; throughput: first start to last end
  ns_per_atomic, cycles_per_atomic (latency only), gatomics_per_s
"""

import io

import pandas as pd

import sbatchman as sbm

# The `benchmark` names written by the latency programs.
LATENCY_COMPONENTS = {"smem", "l1", "l2", "dram", "dsmem"}

# Copied from the CSV, in this order, after `tag` and `component`.
LATENCY_COLUMNS = [
  "stride_bytes", "buffer_kib", "cluster_size", "reader", "target",
  "block_size", "steps", "seed", "rep",
  "cycles", "ns", "cycles_per_load", "ns_per_load", "ghz",
]

# The other programs: their `benchmark` name -> (table, the columns copied
# from their CSV, in this order, after `tag`).
TABLES = {
  "a2av": ("a2av", [
    "method", "distribution", "nranks", "avg_msgsize",
    "total_bytes", "max_sent_bytes", "max_received_bytes",
    "block_size", "rounds", "seed", "rep",
    "ns", "cycles", "ns_per_exchange", "cycles_per_exchange", "gbps",
    "busy_max_ns", "busy_mean_ns",
  ]),
  "smem_bandwidth": ("smem_bandwidth", [
    "op", "width", "stride_bytes", "block_size", "blocks", "steps", "seed", "rep",
    "bytes_per_block", "cycles", "ns", "bytes_per_cycle_per_sm", "gbps_per_sm", "gbps_chip",
  ]),
  "barrier": ("barrier", [
    "barrier", "block_size", "blocks", "cluster_size", "rounds", "seed", "rep", "round",
    "cycles",
  ]),
  "atomic": ("atomic", [
    "memory", "mode", "op", "contention", "layout", "distance",
    "block_size", "workers", "steps", "seed", "rep",
    "atomics", "ns", "cycles", "ns_per_atomic", "cycles_per_atomic", "gatomics_per_s",
  ]),
}


def _status(job) -> str:
  """job.status is a plain string in the metadata but a Status enum in memory."""
  return getattr(job.status, "value", job.status)


def _rows(job, runs, columns, first=None):
  """One dict per CSV row: `tag`, the `first` columns, then `columns`."""
  rows = []
  for record in runs.to_dict("records"):
    row = {"tag": job.tag, **(first(record) if first else {})}
    for column in columns:
      row[column] = record.get(column)   # None if this program does not write it
    rows.append(row)
  return rows


def parse(job: sbm.Job):
  # sbatchman visualize also passes the archived jobs: archiving a run is how
  # it is retired (e.g. measured by an older version of a program), so skip them.
  if job.archive_name:
    return None
  if _status(job) != "COMPLETED":
    return None
  text = job.get_stdout()
  if not text:
    return None

  # The programs print '#' metadata lines, then a CSV with one row per repetition.
  runs = pd.read_csv(io.StringIO(text), comment="#")
  if runs.empty or "benchmark" not in runs:
    return None
  benchmark = runs["benchmark"].iloc[0]

  if benchmark in LATENCY_COMPONENTS:
    return {"latency": _rows(job, runs, LATENCY_COLUMNS,
                             first=lambda r: {"component": r["benchmark"]})}
  if benchmark in TABLES:
    table, columns = TABLES[benchmark]
    return {table: _rows(job, runs, columns)}
  # Jobs of the other experiments (transfer, ...) have no table yet.
  return None
