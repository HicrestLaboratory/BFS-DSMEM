// barrier.cu — what one barrier costs, at each level of the GPU.
//
// THE QUESTION
//   A BFS level ends with a barrier: every block must know the others are
//   done before it reads what they produced. Depending on who has to agree,
//   that barrier is a block barrier, a cluster barrier, or a barrier over the
//   whole grid. What does each one cost, and how does it vary?
//
// THE BARRIERS (--barrier)
//   syncthreads      __syncthreads(): the threads of ONE block (one SM).
//   cluster          cluster.sync(): the blocks of one cluster
//                    (--cluster-size, 1..12). It is a barrier.cluster.arrive
//                    with RELEASE (every write before it is visible to every
//                    block of the cluster after it) + barrier.cluster.wait
//                    (acquire). On GB10 the release compiles to MEMBAR.ALL.GPU.
//   cluster-relaxed  the same barrier with a RELAXED arrive: the blocks wait for
//                    each other but no write is made visible. The difference
//                    to `cluster` is the price of the release.
//   grid             grid.sync() of cooperative groups: every block of the
//                    grid (--blocks). It is built from atomics in global
//                    memory plus fences.
//
// THE MEASUREMENT
//   Every thread of every block runs --warmup x --rounds untimed barriers,
//   then --rounds timed ones. Thread 0 of block 0 times EACH timed barrier
//   on its own:
//       c0 = clock64();  barrier;  c1 = clock64();  sample[r] = c1 - c0;
//   and the program writes one CSV row per barrier, so the distribution can
//   be plotted, not only an average. A sample includes the two clock reads
//   and one shared-memory store. The samples are kept in shared memory and
//   copied out at the end: a store to global memory would be waited for by
//   the fence inside cluster.sync() and grid.sync(), and counted.
//
// Build: make barrier
// Run:   build/barrier --barrier cluster --cluster-size 12 --block-size 1 --reps 5

#include <cooperative_groups.h>

#include "../common.cuh"   // CUDA_CHECK, Args, print_header, print_stats_line, ...

namespace cg = cooperative_groups;

enum { SYNCTHREADS, CLUSTER, CLUSTER_RELAXED, GRID };
static const char* BARRIER_NAME[] = {"syncthreads", "cluster", "cluster-relaxed", "grid"};

constexpr int MAX_ROUNDS = 1024;   // timed barriers per launch: 4 KiB of samples

// cluster.sync() with a relaxed arrive: execution only, no memory ordering.
__device__ __forceinline__ void cluster_sync_relaxed() {
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\t"
                 "barrier.cluster.wait.aligned;" ::: "memory");
}

// One barrier of kind WHICH. A template parameter, not a runtime switch, so
// the timed loop contains the barrier and nothing else.
template <int WHICH>
__device__ __forceinline__ void one_barrier() {
    if (WHICH == SYNCTHREADS)     __syncthreads();
    if (WHICH == CLUSTER)         cg::this_cluster().sync();
    if (WHICH == CLUSTER_RELAXED) cluster_sync_relaxed();
    if (WHICH == GRID)            cg::this_grid().sync();
}

template <int WHICH>
__global__ void barriers(int rounds, int warmup_rounds, unsigned* samples) {
    __shared__ unsigned sample[MAX_ROUNDS];
    const bool timer = (blockIdx.x == 0 && threadIdx.x == 0);

    for (int r = 0; r < warmup_rounds; r++) one_barrier<WHICH>();
    for (int r = 0; r < rounds; r++) {
        long long c0 = 0;
        if (timer) c0 = clock64();
        one_barrier<WHICH>();
        if (timer) sample[r] = (unsigned)(clock64() - c0); // da verificare con gli altri threads e con ncu
    }
    if (timer)
        for (int r = 0; r < rounds; r++) samples[r] = sample[r];
}

int main(int argc, char** argv) {
    Args a;
    a.cluster_size = 12;
    a.reps = 5;
    parse_args(argc, argv, &a, "barrier",
               "  --barrier B       syncthreads | cluster | cluster-relaxed | grid\n"
               "                                                          (default cluster)\n"
               "  --rounds N        timed barriers per repetition, 1..1024 (default 1000)\n"
               "  --block-size N    threads per block, 1..1024            (default 128)\n"
               "  --cluster-size N  cluster barriers: blocks per cluster, 1..12 (default 12)\n"
               "  --blocks N        grid: blocks, 0 = one per SM          (default 0)\n"
               "  --reps N          launches; each writes --rounds rows   (default 5)\n");

    // ---- check the arguments
    int which = -1;
    for (int b = 0; b < 4; b++)
        if (strcmp(a.barrier ? a.barrier : "cluster", BARRIER_NAME[b]) == 0) which = b;
    if (which < 0) {
        fprintf(stderr, "barrier: --barrier must be syncthreads, cluster, cluster-relaxed or grid\n");
        return 1;
    }
    if (a.block_size < 1 || a.block_size > 1024) {
        fprintf(stderr, "barrier: --block-size must be 1..1024\n");
        return 1;
    }
    if (a.rounds < 1 || a.rounds > MAX_ROUNDS) {
        fprintf(stderr, "barrier: --rounds must be 1..%d\n", MAX_ROUNDS);
        return 1;
    }
    const bool clustered = (which == CLUSTER || which == CLUSTER_RELAXED);
    if (clustered && (a.cluster_size < 1 || a.cluster_size > 12)) {
        fprintf(stderr, "barrier: --cluster-size must be 1..12\n");
        return 1;
    }

    unsigned* samples;
    CUDA_CHECK(cudaMallocManaged(&samples, a.rounds * sizeof(unsigned)));
    void (*kernel)(int, int, unsigned*) = (which == SYNCTHREADS) ? barriers<SYNCTHREADS>
                                        : (which == CLUSTER)     ? barriers<CLUSTER>
                                        : (which == GRID)        ? barriers<GRID>
                                                                 : barriers<CLUSTER_RELAXED>;
    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));

    // ---- launch configuration: who takes part in the barrier
    int blocks = 1, cluster_size = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.blockDim = dim3(a.block_size);
    cudaLaunchAttribute attr = {};
    if (clustered) {
        cluster_size = a.cluster_size;
        blocks = cluster_size;   // one cluster
        if (cluster_size > 8) allow_big_clusters((const void*)kernel);
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim.x = cluster_size;
        attr.val.clusterDim.y = 1;
        attr.val.clusterDim.z = 1;
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
    } else if (which == GRID) {
        // grid.sync() deadlocks unless every block is resident at once: a
        // cooperative launch checks that and fails otherwise.
        blocks = a.blocks == 0 ? sms : a.blocks;
        attr.id = cudaLaunchAttributeCooperative;
        attr.val.cooperative = 1;
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
    }
    cfg.gridDim = dim3(blocks);

    char extra[200];
    snprintf(extra, sizeof extra, "# barrier: %s\n# blocks: %d\n# cluster_size: %d\n",
             BARRIER_NAME[which], blocks, cluster_size);
    print_header(argc, argv, a, 0, extra,
                 "benchmark,barrier,block_size,blocks,cluster_size,rounds,seed,rep,round,cycles\n");

    std::vector<double> cycles;
    for (int rep = 0; rep < a.reps; rep++) {
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, a.rounds, a.warmup * a.rounds, samples));
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int r = 0; r < a.rounds; r++) {
            printf("barrier,%s,%d,%d,%d,%d,%d,%d,%d,%u\n", BARRIER_NAME[which], a.block_size,
                   blocks, cluster_size, a.rounds, a.seed, rep, r, samples[r]);
            cycles.push_back(samples[r]);
        }
    }

    fprintf(stderr, "barrier %s: %d blocks x %d threads", BARRIER_NAME[which], blocks, a.block_size);
    if (clustered) fprintf(stderr, " (one cluster of %d)", cluster_size);
    fprintf(stderr, "\n");
    print_stats_line("cycles/barrier", cycles);
    CUDA_CHECK(cudaFree(samples));
    return 0;
}
