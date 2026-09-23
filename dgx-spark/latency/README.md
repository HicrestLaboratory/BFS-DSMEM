# Latency microbenchmarks

Pointer-chase latency of every memory a thread on the DGX Spark (GB10) can
reach, **one program per scenario**, driven by
[SbatchMan](https://sbatchman.readthedocs.io/). This directory supersedes
`../latency.cu`, which measured everything in a single binary and reported
only medians; that file is kept untouched as the reference implementation.

| program | measures |
|---|---|
| `smem_local` | own shared memory, plain block — the baseline |
| `smem_cluster_local` | own shared memory from inside a cluster (`--mapped 0` direct pointer, `--mapped 1` via `map_shared_rank` to self) |
| `dsmem_remote` | another block's shared memory at `--distance` ranks (DSMEM) |
| `l1` | global memory resident in the SM's L1 (warmed in-kernel: L1 is per-SM and does not survive a launch) |
| `l2` | global memory resident in L2 |
| `dram` | global memory not in any cache |
| `chain` | the same chase at any buffer size (`--buffer-kib`), for the latency-vs-working-set curve |
| `latency-many-threads` | DSMEM **under load**: `--requesters` blocks × `--chasers` threads all chasing rank 0's buffer; one row per warp; `--target local` is the own-SRAM control |
| `dsmem_bandwidth` | DSMEM **bytes/s**: `--width 4\|8\|16` bytes per lane × `--access chase\|random\|coalesced` × `--ilp` loads in flight × `--pattern`; own CSV schema (one row per warp, with start/end on a shared clock) |
| `transfer` | S bytes from one SM's SMEM into another's, synchronization included: `--method dsmem-pull\|dsmem-push\|dsmem-bulk\|gmem-ldst\|gmem-tma\|cluster-sync` × `--mode pingpong` (one-way latency) `\|stream` (bandwidth) × `--bytes`, on the rank pair `--src`/`--dst`; every run is verified before it is timed |
| `kernel_boundary` | the level-synchronous baseline: one kernel launch per level handing S bytes through global memory (back-to-back launches, CUDA graph, and host check per level) |

All shared code (chase kernel, Sattolo cycle, CLI parsing, CSV output) is in
`common.cuh`; each `.cu` is only its kernel and a `main()`.

## Build

```sh
make            # all five, nvcc -O3 -arch=sm_121
make dsmem_remote
make clean
```

## Run one program by hand

```sh
./dsmem_remote --cluster-size 4 --distance 1 --reps 11
./l2 --help
```

Every program accepts `--reps`, `--steps`, `--seed`, `--warmup` and
`--stride`; see `--help` for its own flags.

`--stride` selects the access pattern. `random` (the default) is a Sattolo
cycle; a byte count is a fixed stride, so `--stride 4` is the linear walk and
`--stride 128` touches a new cache line every load. The pattern lives in the
buffer *contents*: the kernel always runs the same `idx = buf[idx]` loop, so
the only thing that differs between a random and a stride run is the address
stream. That is what makes the comparison a fair prefetcher test.

Output:

- **stdout** — machine-readable only: `#` metadata lines (exact command, GPU,
  driver, seed, …), then a CSV header, then **one row per repetition**. A
  stdout log is enough on its own to reproduce the run.
- **stderr** — a human summary: n, min, p25, median, mean, sd, p95, max.

All five programs emit the same CSV columns, so any set of logs can be
loaded into one table:

```
benchmark,cluster_size,distance,mapped,block_size,steps,buffer_bytes,
stride_bytes,seed,rep,cycles,ns,cycles_per_load,ns_per_load,ghz
```

Fields that do not apply to a program are `0`; `stride_bytes` is `0` for the
random pattern. `ghz` is `cycles/ns` for that repetition — the clock the SM
actually ran at (a DVFS check).

The `#` metadata block also reports `cycle_length`: how many distinct
elements the chase visits before it repeats. A stride that divides the buffer
size visits only every *n*-th element. `dram` refuses to run if the walk would
wrap, because the second pass would then hit L2 and be reported as DRAM.

## Run the full sweep with SbatchMan

One-time setup, from this directory:

```sh
sbatchman set-cluster-name dgx-spark     # any name; becomes a directory level
sbatchman init                           # creates ./SbatchMan/
sbatchman configure local --name local   # the scheduler config used by experiments.yaml
```

Launch (jobs run one after another, from this directory):

```sh
sbatchman launch -f experiments.yaml
sbatchman launch -f experiments.yaml -t 'dsmem*'   # a subset, by tag glob
sbatchman status                                    # interactive TUI
```

`experiments.yaml` sweeps `dsmem_remote` over cluster sizes 2/4/8/12 at every
rank distance, `smem_cluster_local` over the same sizes × `mapped`, the three
single-scenario programs once each, and then `--stride` from `random` through
4 B to 4 KiB at one representative configuration per placement. Every job's output lands in
`SbatchMan/experiments/<cluster>/local/<tag>/<timestamp>/{stdout.log,stderr.log,metadata.yaml}`.

SbatchMan refuses to re-launch a job identical to one that already exists;
pass `--force` to run it again anyway, or `sbatchman archive` the old ones.

## Read the results

```sh
python analyze.py                # summary per (benchmark, cluster_size, distance, mapped)
python analyze.py --csv all.csv  # plus every repetition, all jobs, one CSV
```

`analyze.py` uses SbatchMan's Python API (`jobs_df()`) to find the jobs, reads
each `stdout.log` with `pandas.read_csv(..., comment='#')`, concatenates, and
prints median / mean / std / p95 of `cycles_per_load`.

```sh
sbatchman launch -f experiments.yaml -t 'chain*'   # 1 KiB .. 2 GiB
python plot_chain.py                                # -> chain_latency.png / .pdf
```

`plot_chain.py` draws latency against chain data volume on a log axis, the
figure of Luo et al. ("Dissecting the NVIDIA Hopper Architecture", Fig. 2)
for this machine: flat while the buffer fits a cache level, a step where it
outgrows one.

```sh
sbatchman launch -f experiments.yaml -t 'loaded*'  # requesters x chasers, plus controls
python plot_loaded.py                               # -> loaded_latency.png / .pdf
```

`plot_loaded.py` is the latency-versus-load curve: per-warp latency (left)
and aggregate throughput (right) against how many threads are chasing at
once. The single-thread programs give the unloaded end of this curve and
`../throughput.cu` the saturated end; this is the middle, where a kernel
actually runs. Little's law (loads in flight = throughput × latency) is
printed alongside as a consistency check.

## Why it is done this way

- **Dependent chase** (`idx = buf[idx]`) — the next address is unknown until
  the current load returns, so nothing overlaps and the time per step is the
  true latency.
- **Sattolo's algorithm** — one random cycle through every element: no short
  loops that would shrink the working set, no order a prefetcher can guess.
- **Untimed warm-up pass** — absorbs one-off costs (instruction cache, DSMEM
  address mapping). The timed pass continues along the cycle, so it walks
  different addresses and the warm-up cannot turn a DRAM chase into an L2 one.
- **`cluster.sync()` after the chase** — a block's shared memory is freed when
  it exits; the barrier keeps the target block alive while its memory is read.
- **`sink`** — the final index is written out, or the compiler deletes the loop.
- **Cycles and ns both** — `clock64()` is DVFS-insensitive; `%globaltimer`
  gives wall time; their ratio is the sustained clock.
- **Every repetition emitted** — the spread is where scheduling and contention
  effects show up; a median alone hides them.
- **Stride encoded in the data, not the loop** — `idx += n` would make every
  address known in advance, the loads would overlap, and the program would
  measure throughput instead of latency. Keeping `idx = buf[idx]` and putting
  the stride into the buffer keeps the dependency (the core cannot run ahead)
  while making the address stream predictable (a prefetcher can). If latency
  drops, only a prefetcher can be the cause.
- **Dummy launch before the L2 warm sweep** — CUDA loads a kernel lazily on
  its first launch (`CUDA_MODULE_LOADING=LAZY` is the default) and that load
  flushes L2. Without the dummy launch the chase kernel's first launch undoes
  the sweep, and L2 reads start at DRAM latency (~930 cy) and drift down to
  the true 340 cy over ~50 repetitions. `../latency.cu` has this bug; its
  reported 349 cy was the median of that drift. Per-rep output made it visible.
