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

One machine per database: only the jobs of SbatchMan cluster $MACHINE are
parsed (default dgx-spark), so GB10 and H100 rows are never mixed. On the H100:

    MACHINE=h100-pcie sbatchman visualize

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
  rank, sm, sent_bytes, received_bytes, busy_cycles
                 ONE ROW PER BLOCK: which block, on which SM, the bytes it
                 sends and receives in one exchange, and the cycles it spent
                 moving bytes in one exchange (until the final barrier).

Table `smem_bandwidth` (dgx-spark/throughput/smem_bandwidth.cu: bandwidth of
an SM's own shared memory, one block per SM on every SM)
  tag, op (read | write), width (bytes per access), stride_bytes (between
  consecutive threads; = width means no bank conflicts), block_size, blocks
  (= SMs), steps (accesses per thread), seed, rep
  bytes_per_block, cycles, ns   the median block
  bytes_per_cycle_per_sm, gbps_per_sm   the median block
  gbps_chip      all blocks' bytes over the slowest block's time.

Tables `barriers`, `barriers_shared`, `barriers_l2` and `barriers_gmem`
(dgx-spark/barriers/barriers.cu, barriers-shared.cu, barriers-l2.cu and
barriers-gmem.cu: how long each thread waits in ONE barrier)
  tag, barrier (syncthreads | cluster | cluster-relaxed | grid), block_size,
  blocks (cluster: the cluster size), seed, rep (the launch)
  block, thread  which thread of the launch this row is; thread < 32 is warp 0,
                 the only warp whose clock read is surely not early
  cycles         end - start around the barrier; except in barriers it also
                 holds the load after the barrier (from shared memory, L2 or DRAM)

Table `fence` (dgx-spark/barriers/fence.cu parts: what each fence and each piece
of cluster.sync() costs, the mean of 1000 back-to-back executions)
  tag, piece (the PTX or the barrier timed), cluster_size, block_size
  median_cycles, min_cycles, max_cycles   over the blocks of the cluster
  median_wait    only for the split piece: the wait on a completed barrier

Table `fence_placement` (dgx-spark/barriers/fence.cu placement: one row per block)
  tag, piece, cluster_size, block_size, block, cluster, sm, cycles

Table `l2_topology` (dgx-spark/extra/l2_topology.cu: L2 latency from every SM)
  tag, smid, gpc, tpc, tpc_pos (the measured GPC and the TPC's position in it),
  buffer_kib, steps, seed, p25, median, p75   cycles per L2 load

Table `histogram_dsmem` (dgx-spark/extra/histogram-dsmem.cu: the same histogram
built with global atomics, a private copy per block, or split over a cluster)
  tag, method (global | private | dsmem), cluster_size (1 without clusters),
  bins, hist_kib, elements, grid, block_size, rep, ms (one timed run)

Table `cluster_residency` (dgx-spark/extra/cluster_residency.cu: how many blocks
run at the same time, with and without clusters)
  tag, block_size, smem_bytes, cluster_size (1 = no clusters), grid (blocks
  launched), api_blocks (occupancy API), measured_blocks (most blocks alive at
  once on the GPU), max_blocks_per_sm (most alive at once on one SM)

Table `p2p` (dgx-spark/throughput/p2p.cu: the paper's copy.cu, --bytes from
rank src to rank dst through DSMEM or global memory)
  tag, method (dsmem-push | dsmem-pull | gmem-ldst), src, dst (1 -> 0 in a
  2-block cluster; 11 -> 10 in a 12-block one), bytes, block_size, seed, rep
  cycles_receiver, cycles_sender   the time of each block, in SM cycles
  cycles         max of the two: the transfer

Table `atomic` (dgx-spark/atomics/atomic.cu: atomicOr / atomicAdd / atomicCAS)
  tag, memory (throughput: smem | dsmem-rank0 | dsmem-rank10 | global;
  latency: smem | dsmem | global), mode (latency | throughput),
  op (or | add | cas), contention (threads per word), scope (intra | inter:
  the threads of a word in one block or over several; "-" for latency), distance
  (dsmem latency: target rank), block_size, workers, steps, seed, rep
  atomics        how many were done in total
  ns, cycles     latency: the chain; throughput: first start to last end
  ns_per_atomic, cycles_per_atomic (latency only), gatomics_per_s
"""

import io
import os

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
    "rank", "sm", "sent_bytes", "received_bytes", "busy_cycles",
  ]),
  "smem_bandwidth": ("smem_bandwidth", [
    "op", "width", "stride_bytes", "block_size", "blocks", "steps", "seed", "rep",
    "bytes_per_block", "cycles", "ns", "bytes_per_cycle_per_sm", "gbps_per_sm", "gbps_chip",
  ]),
  "barriers": ("barriers", [
    "barrier", "block_size", "blocks", "seed", "rep", "block", "thread", "cycles",
  ]),
  "barriers-shared": ("barriers_shared", [
    "barrier", "block_size", "blocks", "seed", "rep", "block", "thread", "cycles",
  ]),
  "barriers-l2": ("barriers_l2", [
    "barrier", "block_size", "blocks", "seed", "rep", "block", "thread", "cycles",
  ]),
  "barriers-gmem": ("barriers_gmem", [
    "barrier", "block_size", "blocks", "seed", "rep", "block", "thread", "cycles",
  ]),
  "fence": ("fence", [
    "piece", "cluster_size", "block_size", "median_cycles", "min_cycles", "max_cycles", "median_wait",
  ]),
  "l2_topology": ("l2_topology", [
    "smid", "gpc", "tpc", "tpc_pos", "buffer_kib", "steps", "seed", "p25", "median", "p75",
  ]),
  "histogram-dsmem": ("histogram_dsmem", [
    "method", "cluster_size", "bins", "hist_kib", "elements", "grid", "block_size", "rep", "ms",
  ]),
  "cluster_residency": ("cluster_residency", [
    "block_size", "smem_bytes", "cluster_size", "grid", "api_blocks", "measured_blocks",
    "max_blocks_per_sm",
  ]),
  "p2p": ("p2p", [
    "method", "src", "dst", "bytes", "block_size", "seed", "rep",
    "cycles_receiver", "cycles_sender", "cycles",
  ]),
  "atomic": ("atomic", [
    "memory", "mode", "op", "contention", "scope", "distance",
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
  if job.cluster_name != os.environ.get("MACHINE", "dgx-spark"):
    return None   # another machine's job: it goes in that machine's database
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
  # fence.cu writes the same benchmark name for its two experiments: only
  # `placement` has one row per SM.
  if benchmark == "fence" and "sm" in runs:
    return {"fence_placement": _rows(job, runs, ["piece", "cluster_size", "block_size", "block",
                                                 "cluster", "sm", "cycles"])}
  if benchmark in TABLES:
    table, columns = TABLES[benchmark]
    return {table: _rows(job, runs, columns)}
  # Jobs of the other experiments (transfer, ...) have no table yet.
  return None
