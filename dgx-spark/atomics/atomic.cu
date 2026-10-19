// atomic.cu — what an atomic operation costs, on each kind of memory.
//
// THE QUESTION
//   A BFS uses atomics everywhere: to claim a slot in the next frontier, to
//   mark a vertex visited, to count. Where the counter lives decides the
//   price: in my own shared memory, in another block's (DSMEM), or in global
//   memory (the atomic is executed in L2). And it depends on how many threads
//   fight over the same address.
//
// THE MEMORIES (--memory)
//   smem     counters in my own shared memory.
//   dsmem    counters in another block's shared memory (DSMEM).
//   global   counters in global memory.
//
// THE MODES (--mode)
//   latency     ONE thread does --steps atomics, each on the address the
//               previous one returned (the counter holds 0 and every atomic
//               adds 0, so the chain stays on counter 0). A thread cannot
//               start an atomic before the previous one has answered, so
//               time / steps is the latency of one atomic.
//               dsmem: rank 0 works on rank --distance's counters.
//   throughput  every thread of every block does --steps atomicAdd(+1), as
//               fast as it can. Result: atomics per second, all blocks
//               together, timed from the first block's start to the last
//               block's end.
//
// THROUGHPUT KNOBS
//   --op add          the result is not used (counting). What the compiler
//                     emits depends on the memory (checked in the SASS):
//                       smem    ATOMS.POPC.INC: ONE instruction adds, to each
//                               address, how many lanes of the warp target it;
//                               32 lanes on the same counter cost one update.
//                       global  REDG, a "reduction": fired and forgotten.
//                       dsmem   a generic ATOM, the same as fetch-add (the
//                               pointer from map_shared_rank() is generic).
//   --op fetch-add    the result is used (claiming a frontier slot): ATOMS.ADD,
//                     ATOMG, or ATOM; the thread waits for the old value.
//   --counters K      thread g of the grid always updates counter g % K.
//                     K = 1: everybody on ONE address (the worst contention);
//                     K >= threads: every thread on its own (no contention).
//                     smem: each block has its own K counters. dsmem: the
//                     target block's K counters. global: K counters shared by
//                     the whole grid.
//   --pattern P       dsmem only: broadcast = every block updates rank 0's
//                     counters (rank 0 itself does nothing); ring = rank k
//                     updates rank k+1's.
//   --clusters N      clusters running at once (0 = fill the chip); every
//                     launch is one or more clusters of --cluster-size blocks.
//
// Build: make atomic
// Run:   build/atomic --memory dsmem --mode throughput --counters 1 --reps 11

#include <cooperative_groups.h>

#include "../common.cuh"   // CUDA_CHECK, Args, globaltimer(), smid(), print_header, ...

namespace cg = cooperative_groups;

enum { SMEM, DSMEM, GLOBAL };
static const char* MEMORY_NAME[] = {"smem", "dsmem", "global"};
enum { BROADCAST = 0, RING = 1 };   // Args::pattern
static const char* PATTERN_NAME[] = {"broadcast", "ring"};

constexpr int MAX_COUNTERS = 16384;   // 64 KiB of shared memory

// What one block measured.
struct BlockTime {
    long long cycles;
    unsigned long long start_ns, end_ns;   // %globaltimer: one clock for the whole GPU
    unsigned sink;                          // touched only if the results add up to a magic value
};

// The counters this block works on: mine, another block's, or the global ones.
// MEMORY is a template parameter so that for smem the compiler sees a shared
// memory pointer and emits a shared-memory atomic.
template <int MEMORY>
__device__ __forceinline__ unsigned* counters_of(unsigned* my_smem, unsigned* global_counters,
                                                 cg::cluster_group& cluster, int target) {
    if (MEMORY == SMEM) return my_smem;
    if (MEMORY == DSMEM) return cluster.map_shared_rank(my_smem, target);
    return global_counters;
}

// --mode latency: thread 0 of rank 0 runs a chain of dependent atomics.
template <int MEMORY>
__global__ void atomic_latency(unsigned* global_counters, int target, unsigned zero, int steps,
                               int warmup, BlockTime* out) {
    extern __shared__ unsigned my_counters[];
    cg::cluster_group cluster = cg::this_cluster();
    if (threadIdx.x == 0) my_counters[0] = 0;
    cluster.sync();   // every block's counter is ready before anyone uses it

    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        unsigned* c = counters_of<MEMORY>(my_counters, global_counters, cluster, target);
        unsigned i = 0;   // counter 0 holds 0, and adding `zero` keeps it so
        for (int s = 0; s < warmup * steps; s++) i = atomicAdd(&c[i], zero);
        const long long c0 = clock64();
        const unsigned long long t0 = globaltimer();
        for (int s = 0; s < steps; s++) i = atomicAdd(&c[i], zero);
        out[0].cycles = clock64() - c0;
        out[0].end_ns = globaltimer();
        out[0].start_ns = t0;
        out[0].sink = i;
    }
    cluster.sync();   // my shared memory must outlive the peer's last atomic
}

// --mode throughput: every thread does `steps` atomicAdd(+1) on its counter.
template <int MEMORY, bool FETCH>
__global__ void atomic_throughput(unsigned* global_counters, int counters, int pattern, int steps,
                                  int warmup, BlockTime* out) {
    extern __shared__ unsigned my_counters[];
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = (int)cluster.block_rank();
    const int cluster_size = (int)cluster.num_blocks();
    for (int i = threadIdx.x; i < counters; i += blockDim.x) my_counters[i] = 0;
    cluster.sync();

    const int target = (pattern == BROADCAST) ? 0 : (rank + 1) % cluster_size;
    const bool works = !(MEMORY == DSMEM && pattern == BROADCAST && rank == 0);
    unsigned* c = counters_of<MEMORY>(my_counters, global_counters, cluster, target);
    const unsigned g = blockIdx.x * blockDim.x + threadIdx.x;   // my number in the grid
    unsigned* mine = &c[g % counters];

    unsigned sum = 0;
    if (works)
        for (int s = 0; s < warmup * steps; s++) {
            if (FETCH) sum += atomicAdd(mine, 1u);
            else       atomicAdd(mine, 1u);
        }
    cluster.sync();   // the whole cluster starts together
    const long long c0 = clock64();
    const unsigned long long t0 = globaltimer();
    if (works)
        for (int s = 0; s < steps; s++) {
            if (FETCH) sum += atomicAdd(mine, 1u);
            else       atomicAdd(mine, 1u);
        }
    __syncthreads();   // the block is done when its slowest thread is
    if (threadIdx.x == 0) {
        out[blockIdx.x].cycles = clock64() - c0;
        out[blockIdx.x].start_ns = t0;
        out[blockIdx.x].end_ns = globaltimer();
    }
    if (FETCH && sum == 0x9e3779b9u) atomicAdd(&out[blockIdx.x].sink, 1u);   // keeps `sum` alive
    cluster.sync();   // my shared memory must outlive the peers' last atomics
}

int main(int argc, char** argv) {
    Args a;
    a.cluster_size = 12;
    a.steps = 1024;
    parse_args(argc, argv, &a, "atomic",
               "  --memory M        smem | dsmem | global                 (default smem)\n"
               "  --mode M          latency | throughput                  (default latency)\n"
               "  --op OP           throughput: add | fetch-add           (default add)\n"
               "  --counters K      throughput: distinct addresses        (default 1)\n"
               "  --pattern P       dsmem throughput: broadcast | ring    (default broadcast)\n"
               "  --distance D      dsmem latency: target rank            (default 1)\n"
               "  --steps N         atomics per thread                    (default 1024)\n"
               "  --block-size N    threads per block                     (default 128)\n"
               "  --cluster-size N  blocks per cluster, 2..12             (default 12)\n"
               "  --clusters N      clusters at once, 0 = fill the chip   (default 1)\n");

    // ---- check the arguments
    int memory = -1;
    for (int m = 0; m < 3; m++)
        if (strcmp(a.memory ? a.memory : "smem", MEMORY_NAME[m]) == 0) memory = m;
    if (memory < 0) { fprintf(stderr, "atomic: --memory must be smem, dsmem or global\n"); return 1; }
    const bool throughput = (a.mode == 1);
    const char* op = a.op ? a.op : "add";
    const bool fetch = strcmp(op, "fetch-add") == 0;
    if (!fetch && strcmp(op, "add") != 0) {
        fprintf(stderr, "atomic: --op must be add or fetch-add\n");
        return 1;
    }
    if (!throughput) op = "fetch-add";   // a latency chain always uses the result
    if (a.pattern != BROADCAST && a.pattern != RING) {
        fprintf(stderr, "atomic: --pattern must be broadcast or ring\n");
        return 1;
    }
    if (a.cluster_size < 2 || a.cluster_size > 12) {
        fprintf(stderr, "atomic: --cluster-size must be 2..12\n");
        return 1;
    }
    if (a.distance < 1 || a.distance >= a.cluster_size) {
        fprintf(stderr, "atomic: --distance must be 1..--cluster-size - 1\n");
        return 1;
    }
    if (a.counters < 1 || a.counters > MAX_COUNTERS) {
        fprintf(stderr, "atomic: --counters must be 1..%d\n", MAX_COUNTERS);
        return 1;
    }
    if (a.block_size < 32 || a.block_size > 1024 || a.block_size % 32) {
        fprintf(stderr, "atomic: --block-size must be a multiple of 32, 32..1024\n");
        return 1;
    }

    // ---- the kernel for this memory / mode / op
    void (*latency_kernel)(unsigned*, int, unsigned, int, int, BlockTime*) =
        memory == SMEM ? atomic_latency<SMEM> : memory == DSMEM ? atomic_latency<DSMEM>
                                                                : atomic_latency<GLOBAL>;
    void (*throughput_kernel)(unsigned*, int, int, int, int, BlockTime*) =
        memory == SMEM  ? (fetch ? atomic_throughput<SMEM, true> : atomic_throughput<SMEM, false>)
      : memory == DSMEM ? (fetch ? atomic_throughput<DSMEM, true> : atomic_throughput<DSMEM, false>)
                        : (fetch ? atomic_throughput<GLOBAL, true> : atomic_throughput<GLOBAL, false>);
    const void* kernel = throughput ? (const void*)throughput_kernel : (const void*)latency_kernel;

    // ---- launch configuration: clusters of cluster_size blocks
    const int counters = throughput ? a.counters : 1;
    const int smem_bytes = counters * sizeof(unsigned);
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    std::max(smem_bytes, 4)));
    if (a.cluster_size > 8) allow_big_clusters(kernel);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(a.cluster_size);
    cfg.blockDim = dim3(throughput ? a.block_size : 32);   // latency: one warp is enough
    cfg.dynamicSmemBytes = smem_bytes;
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = a.cluster_size;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    int fit = 0;   // clusters of this shape that can be resident at once
    CUDA_CHECK(cudaOccupancyMaxActiveClusters(&fit, kernel, &cfg));
    const int clusters = throughput ? (a.clusters == 0 ? fit : a.clusters) : 1;
    if (clusters < 1 || clusters > fit) {
        fprintf(stderr, "atomic: --clusters must be 1..%d (what fits at once)\n", fit);
        return 1;
    }
    const int blocks = clusters * a.cluster_size;
    cfg.gridDim = dim3(blocks);

    // Threads that do atomics: in dsmem broadcast, the owner blocks (rank 0 of
    // each cluster) only receive them.
    const int working_blocks =
        (memory == DSMEM && a.pattern == BROADCAST) ? blocks - clusters : blocks;
    const double atomics = throughput ? (double)working_blocks * a.block_size * a.steps : a.steps;

    char extra[300];
    snprintf(extra, sizeof extra,
             "# memory: %s\n# mode: %s\n# op: %s\n# counters: %d\n# pattern: %s\n# blocks: %d\n"
             "# clusters: %d\n",
             MEMORY_NAME[memory], throughput ? "throughput" : "latency", op, counters,
             PATTERN_NAME[a.pattern], blocks, clusters);
    print_header(argc, argv, a, 0, extra,
                 "benchmark,memory,mode,op,counters,pattern,distance,block_size,cluster_size,"
                 "clusters,steps,seed,rep,atomics,ns,cycles,ns_per_atomic,cycles_per_atomic,"
                 "gatomics_per_s\n");

    // ---- memory the kernel needs
    unsigned* global_counters;
    CUDA_CHECK(cudaMalloc(&global_counters, MAX_COUNTERS * sizeof(unsigned)));
    BlockTime* out;
    CUDA_CHECK(cudaMallocManaged(&out, blocks * sizeof(BlockTime)));

    std::vector<double> per_atomic, rate;
    for (int rep = 0; rep < a.reps; rep++) {
        CUDA_CHECK(cudaMemset(global_counters, 0, MAX_COUNTERS * sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(out, 0, blocks * sizeof(BlockTime)));
        if (throughput)
            CUDA_CHECK(cudaLaunchKernelEx(&cfg, throughput_kernel, global_counters, counters,
                                          a.pattern, a.steps, a.warmup, out));
        else
            CUDA_CHECK(cudaLaunchKernelEx(&cfg, latency_kernel, global_counters, a.distance, 0u,
                                          a.steps, a.warmup, out));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Wall time: from the earliest start to the latest end over the blocks
        // that measured (latency: block 0 only). Cycles: block 0's.
        unsigned long long first = ~0ull, last = 0;
        for (int b = 0; b < (throughput ? blocks : 1); b++) {
            first = std::min(first, out[b].start_ns);
            last = std::max(last, out[b].end_ns);
        }
        const double ns = (double)(last - first);
        const double ns_per = ns / atomics;
        // cycles per atomic is a latency: only meaningful for the chain
        const double cycles_per = throughput ? 0.0 : (double)out[0].cycles / atomics;
        printf("atomic,%s,%s,%s,%d,%s,%d,%d,%d,%d,%d,%d,%d,%.0f,%.0f,%lld,%.4f,%.2f,%.4f\n",
               MEMORY_NAME[memory], throughput ? "throughput" : "latency", op, counters,
               PATTERN_NAME[a.pattern], a.distance, (int)cfg.blockDim.x, a.cluster_size, clusters,
               a.steps, a.seed, rep, atomics, ns, out[0].cycles, ns_per, cycles_per,
               atomics / ns);
        per_atomic.push_back(throughput ? ns_per : cycles_per);
        rate.push_back(atomics / ns);
    }

    fprintf(stderr, "atomic %s %s, %s", MEMORY_NAME[memory], throughput ? "throughput" : "latency",
            op);
    if (throughput)
        fprintf(stderr, ", %d counters, %d blocks x %d threads (%d clusters of %d)%s%s", counters,
                blocks, a.block_size, clusters, a.cluster_size,
                memory == DSMEM ? ", pattern " : "", memory == DSMEM ? PATTERN_NAME[a.pattern] : "");
    fprintf(stderr, "\n");
    print_stats_line(throughput ? "ns per atomic" : "cycles/atomic", per_atomic);
    print_stats_line("G atomics/s", rate);
    CUDA_CHECK(cudaFree(global_counters));
    CUDA_CHECK(cudaFree(out));
    return 0;
}
