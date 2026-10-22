// barrier.cu — what one barrier costs, at each level of the GPU.
//
// THE QUESTION
//   A BFS level ends with a barrier: every block must know the others are
//   done before it reads what they produced. Depending on who has to agree,
//   that barrier is a block barrier, a cluster barrier, a barrier over the
//   whole grid, or a kernel launch. What does each one cost?
//
// THE BARRIERS (--barrier)
//   syncthreads      __syncthreads(): the threads of ONE block (one SM).
//   cluster          cluster.sync(): the blocks of one cluster. It is a
//                    barrier.cluster.arrive with RELEASE (every write before
//                    it is visible to every block of the cluster after it) +
//                    barrier.cluster.wait (acquire). On GB10 the release
//                    compiles to MEMBAR.ALL.GPU.
//   cluster-relaxed  the same barrier with a RELAXED arrive: the blocks wait for
//                    each other but no write is made visible. The difference
//                    to `cluster` is the price of the release.
//   grid             grid.sync() of cooperative groups: every block of the
//                    grid (--blocks, default one per SM). It is built from
//                    atomics in global memory plus fences.
//   The kernel launch, the barrier of a level-synchronous BFS, is not here:
//   it is measured from the host and has its own program.
//
// THE MEASUREMENT
//   Every thread of every block runs --rounds barriers in a row, after
//   --warmup x --rounds untimed ones. Thread 0 of block 0 times them, so the
//   result is the time from one barrier to the next when nobody has anything
//   else to do: the bare cost.
//   --fill-gpu 1 runs as many clusters as fit on the GPU at once, each with
//   its own barrier (a cluster.sync() waits only for its own blocks), to see
//   whether they slow each other down. The time is still cluster 0's,
//   measured while all the others are synchronizing too.
//
// Build: make barrier
// Run:   build/barrier --barrier cluster --cluster-size 12 --reps 11

#include <cooperative_groups.h>

#include "../common.cuh"   // CUDA_CHECK, Args, globaltimer(), print_header, ...

namespace cg = cooperative_groups;

enum { SYNCTHREADS, CLUSTER, CLUSTER_RELAXED, GRID };
static const char* BARRIER_NAME[] = {"syncthreads", "cluster", "cluster-relaxed", "grid"};

// What block 0 measured.
struct Timing {
    long long cycles;
    unsigned long long ns;
};

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
__global__ void barriers(int rounds, int warmup_rounds, Timing* out) {
    for (int r = 0; r < warmup_rounds; r++) one_barrier<WHICH>();
    const long long c0 = clock64();
    const unsigned long long t0 = globaltimer();
    for (int r = 0; r < rounds; r++) one_barrier<WHICH>();
    const long long c1 = clock64();
    const unsigned long long t1 = globaltimer();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out->cycles = c1 - c0;
        out->ns = t1 - t0;
    }
}

int main(int argc, char** argv) {
    Args a;
    a.cluster_size = 12;
    parse_args(argc, argv, &a, "barrier",
               "  --barrier B       syncthreads | cluster | cluster-relaxed | grid\n"
               "                                                          (default cluster)\n"
               "  --rounds N        barriers per repetition               (default 1000)\n"
               "  --block-size N    threads per block                     (default 128)\n"
               "  --cluster-size N  cluster barriers: blocks per cluster, 2..12 (default 12)\n"
               "  --fill-gpu 0|1    cluster barriers: 0 = one cluster, 1 = as many clusters\n"
               "                    as fit on the GPU at once           (default 0)\n"
               "  --blocks N        grid: blocks, 0 = one per SM          (default 0)\n");

    // ---- check the arguments
    int which = -1;
    for (int b = 0; b < 4; b++)
        if (strcmp(a.barrier ? a.barrier : "cluster", BARRIER_NAME[b]) == 0) which = b;
    if (which < 0) {
        fprintf(stderr, "barrier: --barrier must be syncthreads, cluster, cluster-relaxed or grid\n");
        return 1;
    }
    if (a.block_size < 32 || a.block_size > 1024 || a.block_size % 32) {
        fprintf(stderr, "barrier: --block-size must be a multiple of 32, 32..1024\n");
        return 1;
    }
    if (a.rounds < 1) { fprintf(stderr, "barrier: --rounds must be >= 1\n"); return 1; }
    const bool clustered = (which == CLUSTER || which == CLUSTER_RELAXED);
    if (clustered && (a.cluster_size < 2 || a.cluster_size > 12)) {
        fprintf(stderr, "barrier: --cluster-size must be 2..12\n");
        return 1;
    }

    Timing* out;
    CUDA_CHECK(cudaMallocManaged(&out, sizeof(Timing)));
    void (*kernel)(int, int, Timing*) = (which == SYNCTHREADS) ? barriers<SYNCTHREADS>
                                      : (which == CLUSTER)     ? barriers<CLUSTER>
                                      : (which == GRID)        ? barriers<GRID>
                                                               : barriers<CLUSTER_RELAXED>;
    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));

    // ---- launch configuration: who takes part in the barrier
    int blocks = 1, cluster_size = 1, clusters = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.blockDim = dim3(a.block_size);
    cudaLaunchAttribute attr = {};
    if (clustered) {
        cluster_size = a.cluster_size;
        if (cluster_size > 8) allow_big_clusters((const void*)kernel);
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim.x = cluster_size;
        attr.val.clusterDim.y = 1;
        attr.val.clusterDim.z = 1;
        cfg.gridDim = dim3(cluster_size);
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
        // Filling the GPU means exactly as many clusters as can be resident at
        // once: any more would run in a second wave, after the first ones have
        // finished, not at the same time.
        int fit = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveClusters(&fit, (const void*)kernel, &cfg));
        clusters = a.fill_gpu ? fit : 1;
        blocks = clusters * cluster_size;
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

    // fill_gpu is what was asked; clusters is how many that turned out to be.
    const int fill_gpu = clustered ? a.fill_gpu : 0;
    char extra[200];
    snprintf(extra, sizeof extra,
             "# barrier: %s\n# blocks: %d\n# cluster_size: %d\n# fill_gpu: %d\n# clusters: %d\n",
             BARRIER_NAME[which], blocks, cluster_size, fill_gpu, clusters);
    print_header(argc, argv, a, 0, extra,
                 "benchmark,barrier,block_size,blocks,cluster_size,fill_gpu,clusters,rounds,seed,"
                 "rep,cycles,ns,cycles_per_barrier,ns_per_barrier\n");

    std::vector<double> cycles, ns;
    for (int rep = 0; rep < a.reps; rep++) {
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, a.rounds, a.warmup * a.rounds, out));
        CUDA_CHECK(cudaDeviceSynchronize());
        const double cy = (double)out->cycles / a.rounds, t = (double)out->ns / a.rounds;
        printf("barrier,%s,%d,%d,%d,%d,%d,%d,%d,%d,%lld,%llu,%.2f,%.2f\n", BARRIER_NAME[which],
               a.block_size, blocks, cluster_size, fill_gpu, clusters, a.rounds, a.seed, rep,
               out->cycles, out->ns, cy, t);
        cycles.push_back(cy);
        ns.push_back(t);
    }

    fprintf(stderr, "barrier %s: %d blocks x %d threads", BARRIER_NAME[which], blocks, a.block_size);
    if (clustered) fprintf(stderr, " (%d clusters of %d)", clusters, cluster_size);
    fprintf(stderr, "\n");
    print_stats_line("cycles/barrier", cycles);
    print_stats_line("ns/barrier", ns);
    CUDA_CHECK(cudaFree(out));
    return 0;
}
