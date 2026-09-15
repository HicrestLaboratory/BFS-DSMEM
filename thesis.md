# Optimized Breadth-First Search on the NVIDIA DGX Spark: Exploiting Distributed Shared Memory and Work Stealing on the GB10 Grace Blackwell Architecture

**Master's Thesis — Computer Science**
University of Trento
Author: Fabio Missagia
Academic Year 202X/202Y

> **[TODO: supervisor name, co-supervisor, final title approval]**

## Abstract

Breadth-First Search (BFS) is a foundational graph primitive whose performance on GPUs is limited not by arithmetic throughput but by irregular memory access, fine-grained synchronization, and load imbalance. The NVIDIA Hopper and Blackwell architectures introduce *thread block clusters* and *Distributed Shared Memory* (DSMEM), which allow thread blocks co-scheduled on the same GPU Processing Cluster (GPC) to read, write, and perform atomics on each other's shared memory through a dedicated SM-to-SM interconnect, bypassing the L2 cache and global memory. This thesis investigates whether DSMEM can be exploited to accelerate BFS on the NVIDIA DGX Spark, a desktop-class system built around the GB10 Grace Blackwell superchip.

The work proceeds in three stages. First, DSMEM on the GB10 is characterized through microbenchmarks measuring latency, throughput, and atomic performance as a function of cluster size and access pattern, establishing the limitations of the programming model and analytical upper bounds on achievable speedup. Second, the state of the art in GPU-accelerated BFS is surveyed, covering graph partitioning, work stealing, Tensor-Core-based traversal, and compact data representations. Third, informed by both, a DSMEM-based BFS is designed and implemented: a 2D-partitioned traversal in which frontier exchange between blocks of a cluster occurs through remote shared-memory atomics — a hybrid of the shared-memory and message-passing paradigms — complemented by an intra-cluster work-stealing mechanism and an assessment of the Microscaling Integer (MXINT) data type. The implementation is evaluated on synthetic (Graph500 RMAT) and real-world graphs against state-of-the-art baselines.

> **[TODO: fill in headline results — measured DSMEM latency/bandwidth, end-to-end speedup vs. baselines — once experiments are complete.]**

## Table of Contents

1. Introduction
2. Background
3. Characterization of DSMEM
4. State-of-the-Art GPU-Accelerated Breadth-First Search
5. BFS Implementation Based on DSMEM
6. Conclusion
7. Bibliography

# 1. Introduction

> **[TODO: can expand further]**

## 1.1 Motivation

Graph traversal underlies a large share of irregular computing: social network analysis, route planning, web ranking, bioinformatics, and the Graph500 benchmark that ranks supercomputers by traversed edges per second (TEPS) rather than floating-point operations. Breadth-First Search is the canonical representative of this class. It performs almost no arithmetic; its cost is dominated by scattered reads of adjacency lists, atomic updates to visited state, and synchronization between levels of the traversal. On GPUs, whose performance model rewards regular, coalesced, high-arithmetic-intensity workloads, BFS has therefore been a stress test of the architecture for over a decade [Merrill12; Liu15; Wang16].

Two architectural developments motivate revisiting GPU BFS. The first is *Distributed Shared Memory* (DSMEM), introduced with the Hopper architecture [NVIDIA22]: thread blocks launched as a *cluster* are co-scheduled on one GPU Processing Cluster and may address each other's shared memory directly, with traffic carried by a dedicated SM-to-SM network rather than through L2 or DRAM. NVIDIA reports up to a 7× acceleration of inter-block data exchange compared to communicating through global memory [NVIDIA22]. For BFS, whose central cost *is* inter-block data exchange — frontier vertices discovered by one block must reach the block that owns their neighbors — this is a natural fit in principle, and an open question in practice.

The second development is the NVIDIA DGX Spark, a compact system built on the GB10 Grace Blackwell superchip: a 20-core Arm CPU and a Blackwell GPU (compute capability 12.1) sharing 128 GB of coherent unified LPDDR5x memory at roughly 273 GB/s [NVIDIA25]. The unified physical memory removes the classical host-device transfer bottleneck and allows graphs far larger than typical discrete-GPU memory to be traversed in place, but the modest memory bandwidth — roughly an order of magnitude below a datacenter GPU's HBM — shifts the balance of BFS optimization: saving DRAM traffic and keeping traversal state resident in on-chip memory matters proportionally more. DSMEM enlarges the pool of cooperatively usable on-chip memory from one SM's shared memory to that of an entire cluster, which is exactly the lever this thesis pulls.

## 1.2 Problem Statement and Research Questions

This thesis asks: **can Distributed Shared Memory be exploited to accelerate Breadth-First Search on the GB10 Grace Blackwell architecture, and if so, how should the algorithm be structured to do it?** The question decomposes into:

**RQ1.** What are the measured performance characteristics of DSMEM on the GB10 — latency, throughput, atomic cost, scaling with cluster size — and what limitations does the cluster programming model impose? In particular, does the GB10 implement a dedicated SM-to-SM fabric as on GH100, or is remote shared-memory traffic routed through the L2, and what upper bounds on speedup follow?

**RQ2.** Which techniques from the GPU-BFS state of the art — partitioning schemes, work stealing, Tensor Core formulations, compact data types — compose well with the cluster/DSMEM execution model, and which are incompatible with it?

**RQ3.** Does a BFS designed around clusters — 2D-partitioned, with frontier exchange over DSMEM and intra-cluster work stealing — outperform state-of-the-art implementations on the same hardware, and on which graph classes?

## 1.3 Contributions

The contributions of this thesis are: (i) the first systematic characterization of DSMEM on the GB10/compute-capability-12.1 platform, including an analytical model of the communication topology that clusters actually provide and theoretical bounds on the speedup available to communication-bound kernels; (ii) a survey of GPU-accelerated BFS organized around the four design axes relevant to DSMEM exploitation — partitioning, load balancing, Tensor Core usage, and data representation; (iii) the design and implementation of a DSMEM-based BFS with 2D partitioning and intra-cluster work stealing, including an evaluation of the MXINT microscaling data type for traversal state; and (iv) an experimental evaluation on synthetic and real-world graphs, with an honest account of when DSMEM helps, when it does not, and why.

## 1.4 Thesis Structure

Chapter 2 provides background on the GPU execution model, the DGX Spark platform, and BFS. Chapter 3 characterizes DSMEM on the target hardware (RQ1). Chapter 4 surveys the state of the art in GPU BFS (RQ2). Chapter 5 presents the design, implementation, and evaluation of the DSMEM-based BFS (RQ3). Chapter 6 concludes.

# 2. Background

## 2.1 The CUDA Execution Model

A CUDA kernel is executed by a grid of *thread blocks*, each of which is scheduled onto one Streaming Multiprocessor (SM) and remains resident there until completion. Threads within a block synchronize via barriers and communicate through the SM's *shared memory* (SMEM), a software-managed scratchpad with latency roughly an order of magnitude below the L2 cache. Blocks, by contrast, traditionally have no direct communication channel: data exchanged between blocks must round-trip through the L2 cache or DRAM, and inter-block synchronization requires either kernel relaunch or grid-wide cooperative launch [NVIDIA-CG]. This asymmetry — cheap communication inside a block, expensive communication between blocks — has shaped a decade of GPU algorithm design, and BFS in particular.

Physically, SMs are grouped into *GPU Processing Clusters* (GPCs), each with its own compute front end and, from Hopper onward, an intra-GPC SM-to-SM network.

## 2.2 Thread Block Clusters and Distributed Shared Memory

Hopper (compute capability 9.0) added an optional level to the execution hierarchy: the *thread block cluster* [NVIDIA22]. A cluster is a group of blocks guaranteed to be co-scheduled, concurrently resident, on the SMs of a single GPC. Blocks in a cluster can perform a cluster-wide hardware barrier (`cluster.sync()`), and — the property this thesis is built on — each block can map a *rank* of the cluster to a pointer into that block's shared memory (`cluster.map_shared_rank(ptr, rank)` in the Cooperative Groups API). Loads, stores, and atomics through such a pointer operate on the *remote* SM's shared memory. The union of the cluster's shared memory segments is called Distributed Shared Memory (DSMEM).

Three properties of the model matter for algorithm design. First, DSMEM traffic on GH100 travels over a dedicated SM-to-SM interconnect inside the GPC, so its cost is decoupled from L2 and DRAM bandwidth. Second, the cluster is a *scoped* resource: the portable maximum is 8 blocks per cluster (16 on some parts via a non-portable opt-in), so DSMEM extends cooperation from 1 SM to at most a GPC's worth of SMs — not to the whole GPU. Communication between clusters still goes through global memory. Third, co-residency constrains occupancy: all blocks of a cluster must fit on the GPC simultaneously, which couples the affordable shared-memory footprint per block to the cluster size.

On compute capability 12.x the per-SM shared memory is 100 KB (99 KB usable per block), versus 228 KB on Hopper — a factor that roughly halves the aggregate DSMEM capacity per cluster relative to H100 and directly affects the sizing of the data structures in Chapter 5.

## 2.3 The DGX Spark and the GB10 Superchip

The DGX Spark pairs a 20-core Armv9 CPU (10 Cortex-X925 performance cores and 10 Cortex-A725 efficiency cores) with a Blackwell-generation GPU on a single package, connected by NVLink-C2C, with 128 GB of LPDDR5x unified memory delivering approximately 273 GB/s shared between CPU and GPU [NVIDIA25]. The GPU exposes compute capability 12.1 (`sm_121`), with 48 SMs and 6144 CUDA cores, fifth-generation Tensor Cores with FP8/FP6/FP4, integer support, and support for thread block clusters and DSMEM.[NVIDIA-CC].

> We experimentally verified on an NVIDIA GB10 (compute capability 12.1) that Thread Block Clusters and Distributed Shared Memory are supported (`cudaDevAttrClusterLaunch = 1`). Furthermore, using `cudaOccupancyMaxPotentialClusterSize()` and explicit non-portable cluster launches, we confirmed that the maximum supported cluster size on our platform is 8 thread blocks. Attempts to launch clusters of size greater than 8 consistently resulted in `cluster misconfiguration` errors.

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void __cluster_dims__(8, 1, 1) cluster_kernel(int *out) {}

__global__ void probe_kernel(int *out) {}

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("=== Device Properties ===\n");
    printf("Name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("multiProcessorCount (SM count): %d\n", prop.multiProcessorCount);

    int clusterLaunch = 0;
    cudaDeviceGetAttribute(&clusterLaunch, cudaDevAttrClusterLaunch, dev);
    printf("\n=== cudaDevAttrClusterLaunch ===\n");
    printf("cudaDevAttrClusterLaunch: %d\n", clusterLaunch);

    int maxThreadsPerBlock = 0, maxBlocksPerSM = 0, sharedMemPerSM = 0;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, dev);
    cudaDeviceGetAttribute(&maxBlocksPerSM, cudaDevAttrMaxBlocksPerMultiprocessor, dev);
    cudaDeviceGetAttribute(&sharedMemPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    printf("maxThreadsPerBlock: %d\n", maxThreadsPerBlock);
    printf("maxBlocksPerMultiprocessor: %d\n", maxBlocksPerSM);
    printf("maxSharedMemoryPerMultiprocessor: %d\n", sharedMemPerSM);

    int *d_out;
    cudaMalloc(&d_out, sizeof(int));

    printf("\n=== cudaOccupancyMaxPotentialClusterSize ===\n");
    for (int bs : {32, 64, 128, 256, 512, 1024}) {
        cudaLaunchConfig_t cfg = {0};
        cfg.gridDim = dim3(1,1,1);
        cfg.blockDim = dim3(bs,1,1);
        cfg.dynamicSmemBytes = 0;
        int cs = 0;
        cudaError_t e = cudaOccupancyMaxPotentialClusterSize(&cs, (void*)probe_kernel, &cfg);
        printf("  blockDim=%4d -> maxClusterSize=%d (%s)\n", bs, cs, e == cudaSuccess ? "ok" : cudaGetErrorString(e));
    }

    // Actually launch a clustered kernel to confirm cluster launch works on hardware
    printf("\n=== Cluster launch test (cluster dims 8,1,1) ===\n");
    cudaLaunchConfig_t lcfg = {0};
    lcfg.gridDim = dim3(8,1,1);
    lcfg.blockDim = dim3(128,1,1);
    lcfg.dynamicSmemBytes = 0;

    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 8;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    lcfg.attrs = attr;
    lcfg.numAttrs = 1;

    cudaError_t launchErr = cudaLaunchKernelEx(&lcfg, cluster_kernel, d_out);
    cudaError_t syncErr = cudaDeviceSynchronize();
    printf("Launch result: %s\n", cudaGetErrorString(launchErr));
    printf("Sync result: %s\n", cudaGetErrorString(syncErr));

    // Probe hardware (non-portable) cluster size limit by opting in and
    // trying progressively larger cluster dims. This reveals the true
    // GPC-imposed limit rather than the portable guarantee of 8.
    printf("\n=== Non-portable cluster size probe ===\n");
    cudaFuncSetAttribute(cluster_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    for (int csize : {8, 9, 10, 11, 12}) {
        cudaLaunchConfig_t cfg2 = {0};
        cfg2.gridDim = dim3(csize, 1, 1);
        cfg2.blockDim = dim3(128, 1, 1);
        cfg2.dynamicSmemBytes = 0;
        cudaLaunchAttribute a2[1];
        a2[0].id = cudaLaunchAttributeClusterDimension;
        a2[0].val.clusterDim.x = csize;
        a2[0].val.clusterDim.y = 1;
        a2[0].val.clusterDim.z = 1;
        cfg2.attrs = a2;
        cfg2.numAttrs = 1;
        cudaError_t le = cudaLaunchKernelEx(&cfg2, cluster_kernel, d_out);
        cudaError_t se = cudaDeviceSynchronize();
        printf("  clusterDim=%2d -> launch=%-20s sync=%s\n", csize,
               cudaGetErrorString(le), cudaGetErrorString(se));
    }

    cudaFree(d_out);

    printf("\n=== Notes ===\n");
    printf("CUDA does not expose an SM-per-GPC count via a public runtime attribute;\n");
    printf("cluster size limits are enforced by the driver based on GPC topology internally.\n");

    return 0;
}

```

This is the output:

```
== Device Properties ===
Name: NVIDIA GB10
Compute capability: 12.1
multiProcessorCount (SM count): 48

=== cudaDevAttrClusterLaunch ===
cudaDevAttrClusterLaunch: 1
maxThreadsPerBlock: 1024
maxBlocksPerMultiprocessor: 24
maxSharedMemoryPerMultiprocessor: 102400

=== cudaOccupancyMaxPotentialClusterSize ===
  blockDim=  32 -> maxClusterSize=8 (ok)
  blockDim=  64 -> maxClusterSize=8 (ok)
  blockDim= 128 -> maxClusterSize=8 (ok)
  blockDim= 256 -> maxClusterSize=8 (ok)
  blockDim= 512 -> maxClusterSize=8 (ok)
  blockDim=1024 -> maxClusterSize=8 (ok)

=== Cluster launch test (cluster dims 8,1,1) ===
Launch result: no error
Sync result: no error

=== Non-portable cluster size probe ===
  clusterDim= 8 -> launch=no error             sync=no error
  clusterDim= 9 -> launch=a kernel launch error has occurred due to cluster misconfiguration sync=no error
  clusterDim=10 -> launch=a kernel launch error has occurred due to cluster misconfiguration sync=no error
  clusterDim=11 -> launch=a kernel launch error has occurred due to cluster misconfiguration sync=no error
  clusterDim=12 -> launch=a kernel launch error has occurred due to cluster misconfiguration sync=no error

=== Notes ===
CUDA does not expose an SM-per-GPC count via a public runtime attribute;
cluster size limits are enforced by the driver based on GPC topology internally.
```

Two aspects distinguish this platform from the H100-class systems on which DSMEM has so far been studied. First, the memory system: LPDDR5x at ~273 GB/s instead of HBM3 at ~3.3 TB/s. Every byte of DRAM traffic avoided is worth roughly an order of magnitude more, in relative terms, than on an H100. Second, it is not publicly documented whether the GB10's GPC implements the same dedicated SM-to-SM fabric as GH100 or whether remote shared-memory accesses are routed through the L2 crossbar; microbenchmarks of the consumer Blackwell die (GB202) suggest DSMEM loads are packetized similarly to global loads and reach far lower throughput than local SMEM [Blackwell-uB]. Resolving this question for GB10 is a primary goal of Chapter 3, because the answer determines which communication topology a cluster actually provides.

## 2.4 Breadth-First Search

Given a graph G = (V, E) and a source s, BFS computes for every reachable vertex its parent (or depth) in a shortest-path-in-hops tree rooted at s. The standard parallel formulation is *level-synchronous*: the algorithm maintains a *frontier* — the set of vertices discovered in the previous level — and each iteration expands all frontier vertices' adjacency lists, marking unvisited neighbors as the next frontier, with a barrier between levels.

Two dual traversal directions exist [Beamer12]. In *push* (top-down), threads expand the frontier's outgoing edges and atomically claim unvisited neighbors. In *pull* (bottom-up), threads iterate over *unvisited* vertices and check whether any in-neighbor lies in the frontier; on encountering one, the vertex joins the next frontier and the scan of its list terminates early. Push does work proportional to the frontier's edges; pull does work proportional to the unvisited set's edges but needs no atomics and can terminate scans early. Direction-optimizing BFS switches between them per level using frontier-size heuristics, and is most effective on low-diameter, scale-free graphs where a few middle levels contain almost all edges. The linear-algebraic view expresses one BFS level as a sparse matrix–sparse vector product (SpMSpV) over a Boolean semiring, y = Aᵀx masked by the visited set — the formulation under which the Tensor Core approaches of Section 4.3 operate.

The costs that dominate GPU BFS are: (i) scattered reads of adjacency lists (bandwidth-bound, poorly coalesced); (ii) atomic updates to the visited structure and to output frontier queues (contention-bound); (iii) load imbalance across threads and blocks caused by skewed degree distributions (a power-law graph can have degrees spanning six orders of magnitude); and (iv) level synchronization (kernel relaunch or grid barrier per level; on high-diameter graphs with thousands of shallow levels this dominates). Every technique surveyed in Chapter 4, and the design in Chapter 5, targets some subset of these four costs.

## 2.5 Graph Representations

The baseline storage format is Compressed Sparse Row (CSR): a row-pointer array of size |V|+1 and a column-index array of size |E|. BFS state adds a parent/depth array and, in bitmap-based variants, one bit per vertex for the visited set and one per frontier. For 2D-partitioned and Tensor-Core approaches, the adjacency matrix is instead tiled into dense or bit-packed sub-blocks; these formats (BRS, BVSS) are introduced in Section 4.3 where they arise.

---

# 3. Characterization of DSMEM

This chapter answers RQ1. It measures what DSMEM on the GB10 actually delivers (§3.1), catalogs the constraints the programming model imposes (§3.2), derives upper bounds on the speedup a communication-bound kernel can extract (§3.3), and distills criteria for which algorithms can profit from the paradigm (§3.4). The methodology follows the microbenchmarking studies of Hopper [Hopper-uB; Luhnen24] and Blackwell [Blackwell-uB; Blackwell-uB2], adapted to the questions BFS raises: fine-grained random remote accesses, remote atomics, and irregular many-to-many exchange, rather than the bulk regular transfers that GEMM-oriented studies emphasize.

## 3.1 Performance Characterization and Impact of DSMEM

### 3.1.1 Experimental Setup

All measurements run on a DGX Spark (GB10, CUDA ≥ 13.0, driver ≥ 580), with kernels compiled for `sm_121`. Clock frequencies are fixed where the platform allows; each measurement is the median of ≥ 100 repetitions after warm-up. Latency is measured with dependent pointer-chasing chains timed via `%globaltimer`/`clock64()`; throughput with saturating streams of independent accesses.

> **[TODO: exact software versions, clock policy, and thermal conditions of the unit used.]**
The software stack is DGX OS 7.5.0 (Ubuntu 24.04.4 LTS, kernel 6.17.0-1014-nvidia, aarch64), driver 580.142, and CUDA 13.0 (nvcc V13.0.88); all kernels are compiled with an explicit `-arch=sm_121`, which is required for cluster support (the default target rejects `__cluster_dims__`). Persistence mode is enabled.

**Clock policy.** The GB10 does not permit unprivileged clock locking, and the driver exposes no supported-clocks table to `nvidia-smi -lgc`; the application clock is pinned at its default of 2418 MHz, with the SM clock observed at ~2398 MHz under load and a nominal boost ceiling of 3003 MHz. Rather than relying on locked frequencies, the methodology is made DVFS-insensitive: latencies are measured natively in *cycles* via `clock64()`, which is invariant under frequency changes, and converted to nanoseconds using the SM clock sampled concurrently with each run. Throughput measurements record the sampled SM clock alongside each data point, and runs whose clock deviates by more than 1% from the session median are discarded and repeated.

**Thermal conditions.** The unit is actively cooled and operated at ambient room temperature; the GPU idles at 39 °C against a T.Limit of 57 °C, drawing ~9–11 W at idle. Thermal and power slowdown counters (`HW Thermal Slowdown`, `SW Thermal Slowdown`, `HW Power Brake Slowdown`) are checked via `nvidia-smi -q` before and after each benchmark session and were never observed active; the accumulated slowdown time remained 0 µs throughout. Each measurement is the median of ≥ 100 repetitions after warm-up. Latency is measured with dependent pointer-chasing chains timed via `%globaltimer`/`clock64()`; throughput with saturating streams of independent accesses.


### 3.1.2 Latency

The first experiment places a pointer-chase buffer alternately in: local SMEM; remote SMEM at each cluster rank distance; L2 (global memory sized to hit in L2); and DRAM. Prior measurements on H100 situate remote SMEM latency between local SMEM (~30 cycles) and L2 (~250–300 cycles), roughly in the 180–220 cycle range depending on rank distance and scheduling [Hopper-uB]. The key question for GB10 is whether remote-SMEM latency (a) sits similarly below L2 latency, indicating a genuine SM-to-SM path, or (b) coincides with L2 latency, indicating that the "cluster" abstraction is implemented over the ordinary memory hierarchy. Outcome (b) would not make DSMEM useless — it would still save DRAM round-trips for data not resident in L2, still provide the cluster barrier and co-scheduling guarantees, and still enable capacity aggregation — but it would eliminate the latency advantage that motivates fine-grained remote atomics in Chapter 5.

The benchmark (`latency.cu` in the accompanying repository) walks a dependent chain of loads — each address is the value returned by the previous load, arranged as a single random cycle via Sattolo's algorithm — so that latency cannot be hidden by memory-level parallelism. A single thread of cluster rank 0 chases 16 384 steps through the SMEM buffer of rank *d*, obtained through `cluster.map_shared_rank()`; the L2 case chases a buffer of half the L2 capacity (12 MiB of the 24 MiB L2) after a warming sweep, and the DRAM case a 1 GiB buffer with a fresh random starting point per repetition so that no repetition re-walks lines cached by an earlier one. Each figure is the median of 101 repetitions; time is taken simultaneously in `clock64()` cycles and `%globaltimer` nanoseconds, and the sustained SM clock recovered from their ratio was 2.40 GHz for every configuration (maximum deviation < 0.1%, within the policy of §3.1.1). Cluster size 16 is not measurable: as established in §3.2, 8 blocks is a hard hardware ceiling on GB10.

**Table 3.1 — Load-to-use latency by placement (median of 101 runs × 16 384 dependent loads, SM clock 2.40 GHz).**

| Placement | Latency (cycles) | Latency (ns) |
|---|---|---|
| Local SMEM, plain block | 32.9 | 13.7 |
| Local SMEM, block in a cluster (direct or `map_shared_rank` to own rank) | 36.9 | 15.3 |
| Remote SMEM, rank distance 1 | 210.9 | 87.7 |
| Remote SMEM, rank distance 2–3 | 198.9 | 82.7 |
| Remote SMEM, rank distance 4–5 | 187.9 | 78.1 |
| Remote SMEM, rank distance 6–7 | 203.9 | 84.8 |
| L2 hit | 419.5 | 174.5 |
| DRAM (LPDDR5x) | 965.8 | 401.6 |

Three observations. First, remote-SMEM latency is *independent of cluster size*: distances measured under cluster sizes 2, 4, and 8 yield identical values (e.g. distance 1 is 210.9 cycles in all three), so the interconnect does not degrade as more blocks join the cluster. Second, latency is *not monotone in rank distance* — ranks pair up ({2,3}, {4,5}, {6,7} each share a latency), spanning a narrow 188–211-cycle band — consistent with the logical rank → physical SM mapping following the TPC structure (two SMs per TPC) rather than a linear chain; for algorithm design the practical summary is a flat ~200-cycle cost to any remote rank. Third, launching as a cluster costs the *local* SMEM path about 4 cycles (32.9 → 36.9), and `map_shared_rank` to the block's own rank is exactly as fast as a direct pointer — the mapping itself is free.

**The verdict is (a): GB10 implements a genuine SM-to-SM path.** Remote SMEM at 188–211 cycles sits at less than half the L2 hit latency (419.5 cycles) and a fifth of DRAM latency, landing in the same 180–220-cycle band reported for H100 [Hopper-uB] — notable for a desktop-class part. Indeed the *ratio* advantage exceeds Hopper's: because the GB10's L2 is slower (~420 vs. ~250–300 cycles on H100) while its SM-to-SM fabric is not, a remote-SMEM access costs 0.45–0.50× an L2 hit here versus roughly 0.7× on H100. The latency motivation for fine-grained remote atomics in Chapter 5 therefore stands, with the quantitative branch of the argument resolved in DSMEM's favor.

> **[TODO: Figure 3.1 — plot latency vs. rank distance from the CSV emitted by `latency.cu` (columns: config, cluster_size, rank_distance, cycles_per_load, ns_per_load, sm_ghz).]**

### 3.1.3 Throughput

The second experiment measures aggregate bandwidth for: block-to-block unicast (one block streaming to one neighbor); all-to-all exchange within the cluster; and broadcast (one writer, N readers), each for word sizes 4–16 B and for bulk `cp.async.bulk` (TMA) transfers, as a function of cluster size. On GH100, per-SM DSMEM throughput is well below the 128 B/cycle of local SMEM, and remote accesses are packetized so that coalescing rules resembling global-memory access apply [ClusterFusion; Blackwell-uB]; bulk asynchronous copies recover substantially more bandwidth than scalar accesses. The BFS-relevant metric extracted here is sustainable *small-payload* throughput — remote 4-byte atomics and stores — since frontier exchange traffic is intrinsically fine-grained.

The benchmark (`throughput.cu` in the accompanying repository) gives each block a 16 KiB SMEM buffer and streams it with all 256 threads of the accessing block(s): *unicast* (rank 0 streams to/from rank 1), *all-to-all* (every block writes every other rank's buffer round-robin), and *broadcast* (every block of the cluster reads rank 0's buffer), at 4, 8, and 16 B access granularity, plus a TMA variant in which rank 0 pushes back-to-back 16 KiB `cp.async.bulk.shared::cluster` copies into rank 1, with completion tracked by an `mbarrier` armed on the receiver. Comparison points use the same 8-block footprint as one full cluster streaming through the ordinary hierarchy (an L2-resident 12 MiB buffer and a 1 GiB DRAM buffer), and again with 192 blocks for whole-GPU peaks. Every configuration moves ≥ 64 MiB per repetition and reports the median of 101 event-timed repetitions.

**Table 3.2 — Aggregate DSMEM streaming bandwidth (GB/s). Values are identical for 4, 8, and 16 B words to three significant figures, so a single column is shown per pattern.**

| Pattern | Cluster 2 | Cluster 4 | Cluster 8 |
|---|---|---|---|
| Unicast store (rank 0 → rank 1) | 5.0 | — | — |
| Unicast load (rank 0 ← rank 1) | 9.8 | — | — |
| TMA bulk copy, 16 KiB per `cp.async.bulk` | 5.0 | — | — |
| All-to-all store | 10.1 | 20.1 | 32.2 |
| Broadcast load (all ranks read rank 0) | 19.6 | 13.1 | 11.2 |

Reference bandwidths on the same hardware: local SMEM within one block sustains 99.5 GB/s for 4 B stores and 211 GB/s for 16 B vector stores (against a theoretical 128 B/cycle · 2.40 GHz ≈ 307 GB/s port limit); *eight blocks* — one cluster's worth of SMs — streaming through the memory hierarchy sustain 195 GB/s reading L2, 306 GB/s writing L2, and 71/199 GB/s reading/writing DRAM; the whole GPU (192 blocks) reaches 1 980 GB/s from L2 and 242 GB/s from DRAM (89% of the 273 GB/s LPDDR5x spec).

Four observations, all of which cut against DSMEM as a *bulk* transport. First, remote bandwidth is **granularity-independent and TMA-independent**: 4 B scalar stores, 16 B vector stores, and 16 KiB hardware bulk copies all sustain exactly 5.0 GB/s per writing SM (~2.1 B/cycle). This is the signature of a hard per-SM egress limit in the SM-to-SM fabric itself, not of instruction-issue cost — on H100, by contrast, bulk asynchronous copies recover substantially more bandwidth than scalar accesses [ClusterFusion; Blackwell-uB], so this is a point where GB10 is qualitatively weaker than its datacenter siblings. Second, all-to-all **scales near-linearly** with cluster size (5.0 GB/s per SM at size 2–4, 4.0 at size 8): the per-SM links are independent and the fabric itself does not saturate. Third, broadcast **anti-scales** — aggregate bandwidth *falls* from 19.6 to 11.2 GB/s as readers are added, i.e. per-reader bandwidth collapses from 9.8 to 1.4 GB/s — because the owner SM serving the requests is the bottleneck; any design placing hot shared state in one rank's SMEM inherits this. Fourth, the comparison against the ordinary hierarchy is unforgiving: the same 8 blocks that move 32 GB/s over DSMEM move 195–306 GB/s through L2 — a 6–10× deficit. NVIDIA's up-to-7× claim for cluster communication [NVIDIA22] is, on this part and for streaming traffic, inverted.

**Table 3.3 — Atomic throughput (Gop/s), cluster size 8 (2 048 threads). "Disjoint" = every thread its own word (low contention); "hot" = all threads one word (high contention). Global atomics execute in the L2 on NVIDIA GPUs; the buffer here is L2-resident, so the global column is the L2-atomic rate (a DRAM-resident target would add a fill on first touch, §3.1.4).**

| Operation | Contention | Local SMEM | Remote SMEM (DSMEM) | Global (L2) |
|---|---|---|---|---|
| `atomicOr` | disjoint | 156.5 | 7.0 | 44.0 |
| `atomicOr` | hot | 15.6 | 0.18 | 2.1 |
| `atomicAdd` | disjoint | 269.7 | 7.0 | 45.4 |
| `atomicAdd` | hot | 57.3 | 5.3 | 29.9 |
| `atomicCAS` | disjoint | 116.4 | 3.5 | 12.5 |
| `atomicCAS` | hot | 8.4 | 0.18 | 2.1 |

The atomic picture is the decisive one for BFS, and it is negative: remote SMEM atomics run **6× slower than plain global atomics under low contention (7.0 vs. 44 Gop/s for `atomicOr`) and 12× slower on a hot word (0.18 vs. 2.1 Gop/s)**. The "hardware active message" intuition of §3.1.4 — offloading the atomic to the owner SM's fast SMEM atomic units — is real (SASS inspection confirms native `ATOMS.OR/CAS/ADD` instructions, not compare-and-swap emulation loops), but the request must still cross the ~5 GB/s fabric, and at 4 B per operation that fabric limit caps remote atomics at roughly 1.2 Gop/s per issuing SM regardless of how fast the owner executes them. The anomalously fast hot-word `atomicAdd` (5.3 Gop/s remote, 29.9 global, vs. 0.18/2.1 for `atomicOr`) is same-address *add combining* — additive atomics are aggregated in flight (warp-level and/or in the memory system), an optimization not applied to `Or`/`CAS`; BFS's visited-bitmap update is an `atomicOr` and does not benefit.

The verdict of §3.1.2 is thus completed by its converse: **GB10's DSMEM is a low-latency, low-throughput fabric.** A remote word costs half an L2 access in latency, but sustained remote traffic is worth at most ~5 GB/s per SM — under 3% of what the same SMs pull from L2. The design consequence for Chapter 5 is sharp: DSMEM can profitably carry *small, latency-critical* traffic (fine-grained synchronization, stealing requests, duplicate-filter probes), but the bulk of frontier data must continue to flow through L2/DRAM, and a visited bitmap updated by remote `atomicOr` is 6–12× slower than keeping it in global memory. Any cluster-based BFS whose per-SM remote traffic exceeds a few GB/s per level is bandwidth-doomed from the start; §3.3 turns this into a quantitative bound.

> **[TODO: Figure 3.2 — plot DSMEM bandwidth vs. cluster size per pattern with L2/DRAM reference lines, from the CSV emitted by `throughput.cu` (columns: test, cluster_size, word_bytes, gbs_or_gops); raw run archived in `throughput_results.txt`.]**

### 3.1.4 Remote Atomics and Contention

BFS's push phase is dominated by `atomicOr` (visited bitmap) and queue-tail `atomicAdd`. This experiment issues remote SMEM atomics from all blocks of a cluster to (i) disjoint addresses, (ii) the same cache line, (iii) the same address, and compares against global-memory atomics with identical patterns. Shared-memory atomics on NVIDIA GPUs are executed by the SM's own units, so a remote SMEM atomic offloads the operation to the *owner* SM — effectively a hardware active message. Whether this beats an L2-side global atomic under contention is precisely what BFS needs to know, and is measured here.

> **[RESULT PLACEHOLDER: Figure 3.3 — atomic throughput vs. contention degree, DSMEM vs. global.]**

### 3.1.5 Cluster Launch and Synchronization Overheads

Cluster launches constrain the scheduler; this experiment measures (i) launch latency of clustered vs. unclustered grids, (ii) `cluster.sync()` cost vs. `__syncthreads()` and grid-wide cooperative sync, and (iii) occupancy loss: achievable resident blocks per SM as cluster size and per-block SMEM grow. Since level-synchronous BFS performs one global barrier per level, the relative cost of cluster-scope vs. grid-scope synchronization determines how much of the traversal loop can be kept inside a persistent clustered kernel.

> **[RESULT PLACEHOLDER: Table 3.3 — sync costs; occupancy table for the SMEM budgets used in Chapter 5.]**

### 3.1.6 Application-Level Impact

To connect microbenchmarks to workloads, two mini-apps are evaluated with and without DSMEM: a histogram with cluster-partitioned bins (the canonical DSMEM demonstration [NVIDIA22]) and an inter-block queue exchange kernel that mimics one BFS level with a synthetic frontier. The 7× inter-block exchange advantage NVIDIA reports for Hopper [NVIDIA22] serves as the reference point; the measured GB10 figure calibrates expectations for Chapter 5.

> **[RESULT PLACEHOLDER: measured mini-app speedups on GB10.]**

## 3.2 Limitations and Assumptions of the DSMEM Programming Paradigm

The constraints identified analytically and confirmed (or revised) by §3.1 are as follows.

**Scope.** DSMEM exists only within a cluster, and a cluster only within one GPC. On a 48-SM GPU the paradigm therefore organizes cooperation among at most 8 (portably) or 16 (opt-in, if supported on GB10) blocks; the remaining inter-cluster communication uses global memory as before. Any algorithm whose communication graph cannot be clustered into mostly-internal groups of ≤ 8–16 participants gains little.

**Co-residency and occupancy coupling.** All blocks of a cluster must be simultaneously resident. Large per-block SMEM allocations, needed to make DSMEM capacity-aggregation worthwhile, reduce the number of co-resident clusters and hence latency-hiding capability. The paradigm implicitly assumes kernels that trade occupancy for locality — an assumption that holds for GEMM-like kernels with high data reuse but is questionable for latency-bound irregular traversal.

**No inter-cluster ordering or naming.** Cluster ranks are local; there is no hardware mechanism for cluster i to address cluster j's SMEM. Global data structures (the overall frontier, termination flags) remain in global memory.

**Scheduling rigidity.** Clustered grids restrict the scheduler's freedom, lengthening launch latency and interacting poorly with graphs of kernels launched per BFS level; a persistent-kernel design (Chapter 5) avoids repeated cluster launches but then owns the load-balancing problem for the whole traversal.

**Lifetime and visibility.** Remote SMEM is valid only while the owning block is alive, which forces cluster-wide lifetime management (`cluster.sync()` before any block exits) and rules out fire-and-forget producer blocks.

**Toolchain.** Cluster features require explicit `sm_90+`/`sm_120f`-family compilation targets; portability of binaries across the Blackwell consumer/datacenter split is limited, and numeric feature sets (e.g., `tcgen05` Tensor Core instructions) differ between `sm_100` and `sm_12x`, constraining how much of the Hopper/Blackwell literature transfers to GB10.

**Assumption made by this thesis and to be validated:** that remote SMEM atomics on GB10 are executed at the owner SM and are at least competitive with global atomics under contention. If §3.1.4 falsifies this, the Chapter 5 design degrades gracefully to using DSMEM only for capacity aggregation and bulk exchange.

## 3.3 Theoretical Upper Bounds on Performance Improvements

Let a kernel's execution time be decomposed as T = max(T_comp, T_mem, T_comm), with T_comm the inter-block communication component that DSMEM can affect, in the spirit of a roofline extended with a communication ceiling. If a fraction f of the kernel's traffic is inter-block exchange currently served at global-memory cost c_g per byte (or per message), and DSMEM serves it at cost c_d, the best-case kernel-level speedup is the Amdahl-type bound

S ≤ 1 / ((1 − f) + f·(c_d/c_g)),

subject to three additional ceilings derived from §3.1: (i) the *SM-to-SM bandwidth ceiling* — aggregate remote traffic per cluster cannot exceed the measured fabric bandwidth B_sm2sm, so if the algorithm needs V bytes exchanged per level, T_comm ≥ V/B_sm2sm regardless of latency wins; (ii) the *capacity ceiling* — the working set that can be promoted to DSMEM is at most (cluster size) × 99 KB ≈ 0.8 MB per cluster (8 × 99 KB on `sm_12x`), so for a bitmap at 1 bit/vertex, at most ~6.3 M vertices of state per cluster; graphs beyond that spill to L2/DRAM and f shrinks accordingly; and (iii) the *occupancy ceiling* — if promoting state to SMEM cuts resident warps by a factor k, the achievable memory-level parallelism drops and T_mem grows, potentially consuming the communication win.

For BFS specifically, an upper bound on the *end-to-end* benefit follows from instrumenting a baseline traversal: the fraction of total time spent in (a) visited/frontier atomic updates and (b) frontier queue exchange bounds f. Published breakdowns for scale-free graphs attribute on the order of 20–40% of push-phase time to atomics and inter-block frontier traffic; even with c_d/c_g → 0 this caps the DSMEM speedup of an otherwise-unchanged BFS at roughly 1.25–1.7×. Larger gains therefore require the *algorithm* to change — restructuring so that a larger share of traffic becomes intra-cluster (Chapter 5's 2D partitioning does exactly this), or exploiting the aggregated capacity to eliminate DRAM traffic that the baseline pays (bitmap residency), which enters the bound through T_mem rather than f.

> **[TODO: replace the literature-derived 20–40% figure with the measured breakdown of the GSWITCH/BLEST baselines on GB10 (Chapter 5 setup), and instantiate the bound with the measured c_d/c_g and B_sm2sm from §3.1. Present as Figure 3.4: predicted maximum speedup vs. graph size, with the capacity ceiling visible as the knee.]**

The supervisor's topology question fits here: a cluster is, in effect, a fixed-degree, single-hop switched fabric among ≤ 8–16 nodes, each node owning a private scratchpad — architecturally closer to a tiny distributed-memory machine with an all-to-all crossbar than to a shared-memory multiprocessor. The natural cost model is therefore α–β (latency–bandwidth) message passing *inside* the cluster combined with shared-memory semantics *from* global memory (global → SMEM via bulk/TMA multicast, SMEM → SMEM via the fabric). A "collective" over this topology — e.g., a cluster-wide frontier all-gather or bitmap reduction — is a two-level operation: TMA multicast handles the global-to-local level, and rank-symmetric DSMEM exchange handles the local-to-local level, an approach demonstrated for LLM operators by ClusterFusion [ClusterFusion]. Chapter 5 adopts this two-level collective as its frontier-exchange primitive; minimizing overall latency means overlapping the two levels rather than serializing them.

## 3.4 Applicability of the Paradigm

From §3.1–3.3, algorithms benefit from DSMEM to the degree that they exhibit: (1) inter-block communication that is *localizable* — partitionable so that most exchange stays within groups of ≤ 8–16 blocks; (2) working sets between ~100 KB and ~1 MB — too big for one SM's SMEM, small enough for a cluster's aggregate; (3) fine-grained producer-consumer or scatter patterns currently paying global-atomic or L2 round-trip costs; and (4) enough arithmetic or memory-level parallelism to absorb the occupancy loss of co-residency.

Algorithms that fit: stencils with halo exchange (halos map to neighbor ranks); histogram/binning with cluster-partitioned bins; blocked sparse kernels (SpMV/SpMSpV with row-slab ownership per rank); FFT stages sharing twiddle/permutation buffers; producer-consumer pipelines (e.g., attention/GEMM epilogues, as in ClusterFusion); graph traversal *if* restructured around cluster-owned vertex ranges — the thesis's hypothesis. Algorithms that do not fit: kernels with all-to-all global communication (large-matrix transpose), purely bandwidth-bound streaming with no reuse (SAXPY — DSMEM adds nothing), kernels already resident in one SM's SMEM (small-tile GEMM), and latency-bound pointer chasing with no communication locality (random-walk sampling on unpartitioned graphs). BFS occupies an interesting middle position: its *unstructured* form has no communication locality at all, but 2D partitioning manufactures that locality by construction — which is why partitioning, not the memory mechanism itself, is the heart of the design in Chapter 5.

---

# 4. State-of-the-Art GPU-Accelerated Breadth-First Search

This chapter surveys GPU BFS along the four axes that determine whether a technique composes with the cluster/DSMEM model: how the graph is partitioned (§4.1), how load imbalance is corrected at runtime (§4.2), how Tensor Cores have been recruited for traversal (§4.3), and which data types and structures carry the traversal state (§4.4). Each section closes with the implications for a DSMEM-based design.

The modern era of GPU BFS begins with Merrill, Garland, and Grimshaw [Merrill12], who replaced quadratic vertex-frontier scanning with work-efficient prefix-sum-based frontier expansion and fine-grained scan-based gathering, achieving traversal rates then reserved for supercomputers. Beamer's direction-optimizing BFS [Beamer12] contributed the push/pull duality (§2.4); Enterprise [Liu15] integrated direction switching, degree-aware frontier queues, and hub-vertex caching in shared memory; Gunrock [Wang16] generalized frontier-centric traversal into a programmable abstraction; and GSWITCH [Meng19] showed that the best combination of direction, frontier representation (bitmap vs. queue), and load-balancing strategy varies per graph *and per level*, selecting among them at runtime with a learned model. Besta et al. [Besta17] — the "Push or Pull" paper — formalized the duality beyond BFS, quantifying the atomics and work trade-off analytically across graph algorithms and deriving when each direction minimizes communication and synchronization; its cost model is the template for this thesis's per-level direction selection inside clusters.

## 4.1 Graph Partitioning Strategies

**1D partitioning** assigns each processor (block, SM, or GPU) a contiguous range of vertices with their full adjacency lists. It is simple and CSR-native, but for power-law graphs it inherits the degree skew (a single hub can exceed an entire partition's average work), and every processor may need to communicate with every other — the communication graph is unstructured and dense, exactly what §3.4 identified as DSMEM-hostile.

**2D partitioning** tiles the adjacency *matrix*: processor (i, j) owns edges from vertex block i to vertex block j. Pioneered for distributed BFS by Yoo et al. [Yoo05] and refined by Buluç and Madduri [Buluc11], it bounds communication structurally: expanding a frontier segment requires exchange only along processor row i (gathering frontier bits) and column j (scattering discovered vertices) — O(√P) partners instead of O(P). On GPU clusters, 2D partitioning underlies the scalable multi-GPU BFS of [Pan18]. The same structure maps onto a single Blackwell GPU with clusters: letting a cluster own a *row of tiles* makes the column-exchange intra-cluster (DSMEM) and the row-exchange inter-cluster (global memory). This correspondence — 2D partitioning as the locality-manufacturing device for DSMEM — is the structural decision of Chapter 5.

**Degree-aware and hybrid schemes** separate hubs from the long tail: Enterprise classifies frontier vertices into small/medium/large queues served by threads/warps/blocks respectively [Liu15]; SlimSell/vectorized approaches reorder rows by degree. Such schemes compose with either 1D or 2D tiling and reappear in §4.2 as static load balancing.

*Implication for DSMEM:* the partition must be chosen so that the ≤ 8–16 blocks of a cluster absorb the majority of frontier exchange; 2D tiling with cluster-owned tile-rows (or tile-columns for pull) achieves this by construction, at the price of preprocessing (tiling, possibly vertex relabeling) whose cost must be reported honestly in the evaluation.

## 4.2 Work-Stealing Techniques

Static partitioning cannot fully balance a power-law frontier, so runtime redistribution is required. Three families exist on GPUs.

**Intra-kernel hierarchical mapping** (thread/warp/block per vertex by degree class, as in Merrill's scan-based gathering, Enterprise's queues, GSWITCH's strategies) balances *within* the current kernel launch at negligible cost; it is the first line of defense and orthogonal to stealing proper.

**Software work stealing.** Classic GPU task-queue systems give each block a deque in global memory, with idle blocks stealing from victims — Cederman and Tsigas established lock-free deque designs on GPUs [Cederman08]; Whippletree [Steinberger14] built persistent-kernel task scheduling; Atos [Chen22] applied asynchronous task queues to graph workloads, BFS included. The costs are global-memory queue traffic, atomic contention on steal operations, and termination detection — significant enough that for well-balanced workloads stealing is often disabled. DSMEM changes this calculus *within a cluster*: a steal from a sibling block's SMEM-resident deque costs a remote SMEM CAS instead of a global CAS plus L2 round-trips, and victims' queue metadata can be polled over the fabric. Cluster-scoped stealing with a global-memory fallback for inter-cluster imbalance is the two-tier design adopted in Chapter 5.

**Hardware-assisted stealing: Cluster Launch Control.** Blackwell introduces Cluster Launch Control (CLC) [NVIDIA-CLC], which lets a persistent cluster *cancel* a not-yet-launched cluster of the grid and claim its work — hardware work stealing at cluster granularity, designed to replace software persistent-kernel schedulers. Whether CLC is exposed on `sm_121` must be verified; if it is, it provides the *inter-cluster* tier of load balancing with hardware termination detection, complementing DSMEM-based intra-cluster stealing.

> **[VERIFY: CLC availability on GB10 — check `cudaDevAttrClusterLaunch` + PTX `clusterlaunchcontrol.try_cancel` support for `sm_120f`/`sm_121`; the CUDA 13 Programming Guide documents CLC in §4.12 "Work Stealing with Cluster Launch Control".]**

The supervisor's observation frames the evaluation here: if 2D partitioning balances well, stealing should trigger rarely — so the mechanism must be cheap when idle (a per-level check of neighbors' queue occupancy over DSMEM costs a handful of remote loads), and its rare activations need not be highly efficient to be worthwhile. The right experiment is therefore not only end-to-end speedup but *steal-frequency and steal-cost accounting* across graph classes (§5.5), with the honest possibility that on well-partitioned graphs the answer is "the mechanism is insurance, not acceleration" — a legitimate finding for §5.6.

## 4.3 Exploiting Tensor Cores

Tensor Cores natively compute small dense matrix products; the insight enabling their use for BFS is that over the Boolean semiring, one BFS level is a masked SpMSpV, and bit-packed adjacency tiles turn edge expansion into binary matrix multiplication (`bmma`/1-bit MMA), processing 8×8×128-bit tiles per instruction. BerryBees [Nie25] introduced the Binarized Row Slice (BRS) format — adjacency rows sliced into bit-packed segments aligned to Tensor Core fragment shapes — with SpMV/SpMSpV kernels that make pull-phase BFS a sequence of bit-matrix products. BLEST [EK25] improved on it with Binarized Virtual Slice Sets (BVSS) for warp-level load balancing and a batched SpMSpV that avoids frontier-oblivious work, reporting ~3.6× over BerryBees and ~4.6–4.9× over Gunrock/GSWITCH. A 2026 framework generalizes the approach across modern GPUs [TCBFS26].

Three considerations govern transfer to this thesis. First, the Tensor Core formulation is intrinsically *tiled* — BRS/BVSS tiles are exactly the 2D sub-blocks of §4.1 — so it composes naturally with cluster-owned tile-rows: a cluster can stage its frontier-bit slice in DSMEM and stream adjacency tiles through the Tensor Cores of its 8 SMs, with the bitwise-OR reduction across ranks performed over the fabric. Second, `sm_12x` Tensor Cores use the Ampere-style `mma.sync` path, not Hopper's `wgmma` or datacenter Blackwell's `tcgen05`; 1-bit `bmma` support on `sm_120/121` must be verified on hardware, since the b1 MMA shapes have been deprecated/restricted in recent architectures. Third, Tensor Core BFS is a *pull-phase* technique; the push phases and direction switching still run on conventional cores, so the integration point is per-level: dense middle levels → TC pull over DSMEM-staged bitmaps, sparse levels → queue-based push.

> **[VERIFY: availability and throughput of 1-bit `bmma`/`mma.and.popc` on `sm_121`; if absent, evaluate INT8 Tensor Core emulation of Boolean products or fall back to popcount-based bitmap pull on CUDA cores, and record this as a platform limitation.]**

## 4.4 Data Types and Data Structures for High-Performance BFS

**Vertex identifiers and indices.** 32-bit IDs suffice for |V| < 4.3 B and halve index bandwidth vs. 64-bit; within a 2D tile, *local* offsets need only ⌈log₂(tile size)⌉ bits, so 16-bit intra-tile indices are common in tiled formats and shrink both DRAM traffic and DSMEM footprint — directly increasing the vertex capacity per cluster computed in §3.3.

**Frontier representations.** Sparse queues (compact, good for small frontiers; require atomic appends) vs. bitmaps (1 bit/vertex, atomic-free duplicate suppression via idempotent OR, good for dense frontiers); high-performance implementations switch per level [Meng19]. Hierarchical/two-level bitmaps (a summary bitmap over words of the base bitmap) accelerate scanning sparse bitmaps. For this thesis the bitmap is the DSMEM-resident structure of choice: it is compact enough to partition across a cluster (§3.3's ~6.3 M vertices/cluster at 1 bit), and OR-idempotency makes remote updates race-tolerant.

**Visited/status encoding.** Parent arrays (Graph500 output requirement) vs. depth arrays (8–16 bits often suffice: depth ≤ 255 covers virtually all real graphs) vs. 1-bit visited with parent reconstruction. Packing status into sub-byte fields trades atomic-width awkwardness against bandwidth.

**Compressed and blocked adjacency formats.** Beyond CSR: bit-packed tiles (BRS/BVSS, §4.3); delta-encoded adjacency (CGR-style) trading decode ALU for bandwidth — attractive on a 273 GB/s-bound platform; SlimSell/SELL-C-σ for vectorized pull. The common thread is that on bandwidth-poor hardware, computation spent decompressing is usually well spent.

**Microscaling formats (MX).** The OCP Microscaling standard [OCP-MX] defines block formats where groups of k = 32 elements share an 8-bit power-of-two scale, with 8/6/4-bit elements (MXFP8/6/4, MXINT8); Blackwell Tensor Cores accelerate them natively. MX formats target *numeric* payloads, so their role in BFS — whose core state is Boolean/integral — is limited to weighted or property-graph extensions (e.g., SSSP edge weights, feature vectors in graph learning) or to mixed-precision SpMV formulations of traversal. Chapter 5 (§5.4) evaluates MXINT8 for compressed depth/weight storage and reaches a conclusion honestly conditioned on measurement: the a-priori case is weak for unweighted BFS, and the thesis says so rather than forcing the integration.

*Implication for DSMEM:* the state that most rewards fabric residency is the smallest and hottest — frontier and visited bitmaps with 16-bit local indices — while adjacency data should *stream* (bulk TMA multicast into SMEM, §3.3's two-level collective) rather than reside.

---

# 5. BFS Implementation Based on DSMEM

This chapter presents CBFS (Cluster-BFS), the traversal designed from the findings of Chapters 3 and 4, and its evaluation on the DGX Spark. The design premise, restated: DSMEM does not accelerate an unmodified BFS by much (§3.3's Amdahl bound); the algorithm must be restructured so that (i) most frontier exchange becomes intra-cluster, (ii) the hottest state becomes DSMEM-resident, and (iii) load imbalance is corrected first statically (2D tiling), then cheaply (intra-cluster stealing), and only rarely globally.

## 5.1 Algorithm Design

**Execution structure.** The GPU's C clusters (cluster size N = 8 blocks, subject to §3.1's verification; 48 SMs → 6 clusters of 8, one per GPC region) run as a persistent clustered grid for the whole traversal, avoiding per-level cluster-launch overhead (§3.1.5). Levels are separated by a two-tier barrier: `cluster.sync()` within clusters, a lightweight global barrier (cooperative grid sync or atomic-flag protocol) across them.

**State placement.** Each cluster owns a contiguous vertex range R_c, sub-partitioned across its N blocks; each block holds in SMEM: its slice of the *next-frontier bitmap* and *visited bitmap* (1 bit/vertex of its sub-range), a compact work queue of local frontier vertices with 16-bit intra-tile indices, and steal-metadata (queue head/tail) at a fixed SMEM offset so any sibling can locate it via `map_shared_rank`. With 99 KB/block and the budget split (Table 5.1), a cluster covers ~4–6 M vertices of bitmap state; graphs beyond the aggregate capacity of all clusters fall back to L2/DRAM-resident bitmaps with DSMEM used for exchange only (the graceful-degradation path from §3.2).

> **[TODO: Table 5.1 — exact SMEM budget per block: next-frontier slice, visited slice, queue, staging buffers, steal metadata.]**

**Push phase (sparse frontiers).** Blocks drain their local queues; for each neighbor v of a frontier vertex, the owner is computed by range: if v ∈ R_c (same cluster), the thread issues a remote `atomicOr` into the owner block's SMEM bitmap over the fabric — the hardware active-message pattern validated in §3.1.4 — otherwise it appends (v) to a per-destination-cluster coalescing buffer flushed to global memory with bulk stores. This is the supervisor's hybrid made concrete: shared-memory semantics for global structures, message passing between SMEM scratchpads.

**Pull phase (dense frontiers).** Direction switching follows Beamer's heuristics [Beamer12] evaluated per level with the cost model of [Besta17]. In pull, each cluster processes the tile-row of the adjacency matrix whose *destination* vertices it owns: the current-frontier bitmap slices needed by all blocks are staged once into DSMEM via TMA bulk copies — the global→local level of the two-level collective (§3.3) — and each block scans its unvisited vertices against the staged bits, either with popcount on CUDA cores or, if §4.3's verification succeeds, with 1-bit Tensor Core products over BVSS-style tiles.

**Correctness.** Bitmap OR is idempotent, so duplicate discovery is benign; parent assignment uses a subsequent claim pass (or CAS on the parent array) to satisfy Graph500 output semantics; memory ordering across the fabric uses cluster-scoped acquire/release (`cuda::thread_scope_cluster`) validated against the §3.1 memory-model experiments.

## 5.2 Graph Partitioning

The adjacency matrix is tiled 2D as in [Buluc11; Pan18]: vertex range split into C cluster ranges (matrix block-rows for pull, block-columns for push), each further split into N block sub-ranges. Tiles are stored bit-packed for pull (BRS/BVSS-like, §4.3) and CSR-sliced for push. Because clusters map to GPCs, the partition makes the communication topology explicit: within a tile-row, exchange rides the SM-to-SM fabric; across tile-rows, it rides L2/DRAM. Vertex relabeling (degree-descending or community-clustering order) is evaluated as a preprocessing option — it concentrates hubs, improving tile density for Tensor Core pull, at documented preprocessing cost.

> **[TODO: preprocessing pipeline description + cost table; sensitivity of tile density to relabeling on each benchmark graph.]**

The expected property, per the supervisor's remark: with 2D tiling, per-block work within a level is bounded by the tile's edge count, so imbalance is structural only where tile densities differ — which the stealing tier (§5.3) absorbs — and the workload should rarely trigger stealing at all on well-behaved graphs. This prediction is tested explicitly in §5.5.4.

## 5.3 Work-Stealing Implementation

Two tiers. **Intra-cluster (DSMEM):** each block's queue exposes head/tail counters in SMEM; an idle block polls siblings' counters via `map_shared_rank` (N−1 remote loads, §3.1.2 cost), selects the deepest queue, and claims a chunk with a remote CAS on the tail — a steal costs a handful of fabric transactions and no global traffic. Chunked stealing (half the remaining queue, min 32 vertices) amortizes CAS contention. **Inter-cluster:** if the whole cluster idles, it either (a) claims work via Cluster Launch Control, if available on `sm_121` (§4.2), letting hardware hand it an unlaunched tile's worth of work, or (b) falls back to a global-memory steal from per-cluster overflow queues. Termination detection is hierarchical: cluster-local counters reduced over DSMEM, then a global epoch counter.

Design stance, following the supervisor: the mechanism is engineered to be *cheap when inactive* (polling only on idleness, one cache line of metadata per victim) rather than maximally efficient when active; if 2D balancing works, stealing is insurance. The evaluation quantifies both how often it fires and what it costs when it does; if it fires rarely and saves little, that result goes to §5.6 as a limitation-slash-finding rather than being tuned into significance.

## 5.4 Integration of MXINT (Microscaling Integer)

MXINT8 stores blocks of 32 integers with a shared 8-bit scale [OCP-MX], accelerated by Blackwell Tensor Cores. For unweighted BFS the traversal state is Boolean/ordinal and does not benefit from shared-exponent scaling; the a-priori case (§4.4) is weak. Three candidate uses are nevertheless evaluated: (i) MXINT8 depth arrays (depth values share scale trivially; the win is pure bandwidth, and a plain uint8 array achieves the same without MX machinery — measured to confirm); (ii) MXINT8 edge weights for the weighted-traversal extension (SSSP/BFS-with-weights), where Tensor-Core-native decompression is genuinely free; (iii) MXINT-packed frontier *counters* in multi-source BFS variants. The section reports measured bandwidth/TEPS deltas and concludes with a recommendation.

> **[RESULT PLACEHOLDER: if, as expected, (i) and (iii) show no advantage over plain narrow integers, state plainly that MXINT integration is *not worth it* for unweighted BFS on this platform and scope it to the weighted extension. A negative result stated with measurements is a contribution.]**

## 5.5 Performance Evaluation

**Setup.** DGX Spark (GB10), CUDA ≥ 13.x; graphs: Graph500 RMAT scales 20–27 (up to the 128 GB unified-memory budget), plus real-world graphs spanning the diameter/skew spectrum — soc-LiveJournal, com-Orkut, twitter-2010, uk-2005/webbase (low-diameter scale-free) and road_usa, europe_osm (high-diameter, low-degree) from SNAP/SuiteSparse/LAW. Metrics: harmonic-mean TEPS over 64 random sources (Graph500 protocol), per-level time breakdown, DRAM/L2/fabric traffic from Nsight Compute counters, energy where readable. Baselines: GSWITCH [Meng19], Gunrock [Wang16], BLEST [EK25] (Tensor Core state of the art), and CBFS ablations.

**Ablation ladder** — each step isolates one claim: (A0) 1D queue-based baseline; (A1) + 2D tiling, global-memory exchange (partitioning alone); (A2) + DSMEM frontier/visited residency and intra-cluster exchange (the DSMEM claim); (A3) + intra-cluster stealing; (A4) + CLC/global stealing; (A5) + Tensor Core pull; (A6) + MXINT variants. The A1→A2 delta, compared against §3.3's instantiated upper bound, is the thesis's central measurement: how much of the theoretically available DSMEM benefit the implementation realizes.

**Analyses.** Scaling with graph size across the DSMEM capacity knee (§3.3); direction-switch behavior per level; steal-frequency/cost accounting (§5.3) per graph class; sensitivity to cluster size (2/4/8/16); topology validation — measured fabric traffic vs. the α–β model predictions of §3.3.

> **[RESULT PLACEHOLDER: Figures 5.2–5.8 and discussion. Honest reporting requirements: include preprocessing time; report where baselines win; attribute every gain to a ladder step; compare A2 gains against the Chapter 3 bound and explain the gap.]**

## 5.6 Limitations and Future Research Directions

**Limitations to document from the design (updated after measurement):** DSMEM capacity caps bitmap residency at roughly 4–6 M vertices per cluster — beyond ~30–40 M vertices total the design degrades to exchange-only DSMEM use, and its advantage shrinks with graph size; clusters bind the schedule to GPCs, costing occupancy on kernels that would otherwise oversubscribe; the approach presumes the GB10 fabric verdict of §3.1 — if remote SMEM is L2-routed, latency-class gains vanish and only capacity/collective benefits remain; work stealing may fire too rarely to justify its complexity on well-partitioned graphs (per the supervisor's prediction — reported as measured); 1-bit Tensor Core support on `sm_121` may be absent, stranding the pull-phase acceleration on this platform even though the design supports it; preprocessing (tiling, relabeling) is amortized only over repeated traversals; and results from a single 48-SM, 273 GB/s part do not automatically transfer to datacenter Blackwell's different fabric, SMEM size, and bandwidth balance.

**Future directions:** multi-source/batched BFS, where b concurrent traversals share adjacency streaming and pack b frontier bits per vertex — multiplying DSMEM utility per byte; extension to the full GraphBLAS kernel family (BC, connected components, SSSP with MXINT weights); cooperative CPU-GPU traversal exploiting the Grace cores and cache-coherent NVLink-C2C for the high-diameter tail levels where GPUs idle; CLC-centric dynamic tiling if CLC proves available and cheap; two-node DGX Spark scaling over the ConnectX/RDMA link, testing whether the intra-cluster/inter-cluster hierarchy extends naturally to a third, inter-node tier; and porting to datacenter Blackwell (GB200/GB300) to separate paradigm effects from platform effects.

---

# 6. Conclusion

This thesis set out to determine whether Distributed Shared Memory — the Hopper/Blackwell mechanism that turns a GPC's shared memories into a fabric-connected scratchpad pool — can accelerate Breadth-First Search on the DGX Spark's GB10. The characterization of Chapter 3 established the mechanism's measured costs and, through an Amdahl-type communication bound, showed that DSMEM rewards algorithms restructured for communication locality rather than unmodified kernels. Chapter 4 identified 2D partitioning as the device that manufactures such locality for BFS, work stealing as a two-tier insurance mechanism whose intra-cluster tier DSMEM makes cheap, Tensor Core traversal as a tile-native technique that composes with cluster ownership, and compact bitmaps with narrow indices as the state most worth making fabric-resident. Chapter 5 embodied these findings in CBFS and evaluated it against the state of the art.

> **[TODO: one paragraph of quantified conclusions per RQ once results exist, including the honest verdicts on stealing frequency and MXINT worth.]**

The broader observation stands independent of the numbers: thread block clusters give a single GPU the communication *structure* of a small distributed machine — private scratchpads, a low-latency local fabric, and an expensive global level — and algorithms gain from them precisely when redesigned with distributed-memory discipline: partition for locality, communicate in collectives, balance hierarchically. BFS, the archetypal irregular workload, is a demanding test of that discipline; the methodology developed here — characterize the fabric, bound the benefit, restructure the algorithm, ablate the claims — applies to the wider family of irregular computations that will confront this architecture as it propagates through NVIDIA's product line.

---

# 7. Bibliography

> **[TODO: convert to the department's required citation style; verify venues/pages before submission.]**

- **[Beamer12]** S. Beamer, K. Asanović, D. Patterson. *Direction-Optimizing Breadth-First Search.* SC '12.
- **[Besta17]** M. Besta, M. Podstawski, L. Groner, E. Solomonik, T. Hoefler. *To Push or To Pull: On Reducing Communication and Synchronization in Graph Computations.* HPDC '17.
- **[Blackwell-uB]** A. Jarmusch et al. *Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks.* arXiv:2507.10789, 2025.
- **[Blackwell-uB2]** *Microbenchmarking NVIDIA's Blackwell Architecture: An In-depth Architectural Analysis.* arXiv:2512.02189, 2025.
- **[Buluc11]** A. Buluç, K. Madduri. *Parallel Breadth-First Search on Distributed Memory Systems.* SC '11.
- **[Cederman08]** D. Cederman, P. Tsigas. *On Dynamic Load Balancing on Graphics Processors.* Graphics Hardware '08.
- **[Chen22]** Y. Chen et al. *Atos: A Task-Parallel GPU Scheduler for Graph Analytics.* ICPP '22.
- **[ClusterFusion]** *ClusterFusion: Expanding Operator Fusion Scope for LLM Inference via Cluster-Level Collective Primitive.* arXiv:2508.18850, 2025.
- **[EK25]** *BLEST: Blazingly Efficient BFS using Tensor Cores.* arXiv:2512.21967, 2025.
- **[Hopper-uB]** W. Luo et al. *Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis.* arXiv:2501.12084, 2025.
- **[Liu15]** H. Liu, H. H. Huang. *Enterprise: Breadth-First Graph Traversal on GPUs.* SC '15.
- **[Luhnen24]** T. Lühnen. *Benchmarking Thread Block Clusters.* TU Hamburg, 2024.
- **[Meng19]** K. Meng, J. Li, G. Tan, N. Sun. *A Pattern Based Algorithmic Autotuner for Graph Processing on GPUs* (GSWITCH). PPoPP '19.
- **[Merrill12]** D. Merrill, M. Garland, A. Grimshaw. *Scalable GPU Graph Traversal.* PPoPP '12.
- **[Nie25]** *BerryBees: Breadth First Search by Bit-Tensor-Cores.* PPoPP '25.
- **[NVIDIA22]** NVIDIA. *NVIDIA Hopper Architecture In-Depth.* Technical blog, 2022.
- **[NVIDIA25]** NVIDIA. *DGX Spark / GB10 Grace Blackwell Superchip.* Product documentation, 2025.
- **[NVIDIA-CC]** NVIDIA. *CUDA C++ Programming Guide, §5.1 Compute Capabilities.* v13.x.
- **[NVIDIA-CG]** NVIDIA. *CUDA C++ Programming Guide, §4.4 Cooperative Groups.* v13.x.
- **[NVIDIA-CLC]** NVIDIA. *CUDA C++ Programming Guide, §4.12 Work Stealing with Cluster Launch Control.* v13.x.
- **[OCP-MX]** Open Compute Project. *OCP Microscaling Formats (MX) Specification v1.0.* 2023.
- **[Pan18]** Y. Pan, Y. Wang, Y. Wu, C. Yang, J. D. Owens. *Scalable Breadth-First Search on a GPU Cluster.* IPDPS '18 / arXiv:1803.03922.
- **[Steinberger14]** M. Steinberger et al. *Whippletree: Task-Based Scheduling of Dynamic Workloads on the GPU.* SIGGRAPH Asia '14.
- **[TCBFS26]** *Graph Traversal on Tensor Cores: A BFS Framework for Modern GPUs.* arXiv:2606.05081, 2026.
- **[Wang16]** Y. Wang et al. *Gunrock: A High-Performance Graph Processing Library on the GPU.* PPoPP '16.
- **[Yoo05]** A. Yoo et al. *A Scalable Distributed Parallel Breadth-First Search Algorithm on BlueGene/L.* SC '05.