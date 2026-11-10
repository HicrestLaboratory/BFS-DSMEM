// fence.cu — take cluster.sync() apart: what each piece costs and where the
// barrier signal goes. (Whether it travels through L2: l2-flooding.cu.)
//
// WHAT cluster.sync() IS
//   PTX:  barrier.cluster.arrive (release) + barrier.cluster.wait (acquire).
//   SASS: MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR   the release: a GPU-scope fence
//         UCGABAR_ARV                         the arrive (dedicated instruction)
//         UCGABAR_WAIT                        the wait
//         CCTL.IVALL                          the acquire: invalidate L1
//   fence.acq_rel.cluster compiles to the same SASS as fence.acq_rel.gpu:
//   there is no cluster-scope fence in the instruction set.
//
// THE EXPERIMENTS (first argument)
//   parts            every piece below (barriers, and fences of every scope,
//                    including the shared-memory-only ones), on one cluster of
//                    --cluster-size blocks of --block-size threads. One row per
//                    piece: the median over the blocks.
//   placement        48 blocks at once, in clusters of --cluster-size: one row
//                    per block with its SM. With --cluster-size 1 every block
//                    measures alone, so the cost of each SM's own path is
//                    visible.
//
// THE MEASUREMENT
//   Every number is the mean of ITERS back-to-back executions of one piece,
//   after WARM untimed ones, with one clock pair around the whole loop:
//       t0 = clock64();  ITERS x piece;  t1 = clock64();
//   Each piece is its own __device__ function, passed to the kernel as a
//   template argument, so the timed loop has no branch. A barrier in this
//   steady state is cheaper than the first barrier of a kernel, which is what
//   barriers.cu measures.
//
//   WARNING: the SASS of a barrier depends on the code around it. With a
//   printf in the same kernel, ptxas adds a MEMBAR.ALL.CTA before every
//   MEMBAR.ALL.GPU, and cluster.sync() costs ~60 cycles more. Keep the
//   kernels free of anything but the measurement (check with
//   cuobjdump -sass build/fence).
//
// Build: make fence
// Run:   build/fence parts --cluster-size 12 --block-size 256
//        build/fence placement --cluster-size 1

#include <cooperative_groups.h>

#include "../common.cuh"   // CUDA_CHECK, Args, parse_args, smid(), allow_big_clusters()

constexpr int ITERS = 1000;   // timed executions of a piece
constexpr int WARM = 100;     // untimed executions before them
constexpr int SMS = 48;       // GB10

// What one block measured.
struct Out {
    long long cycles;   // mean cycles of the piece (split: of the arrive)
    long long wait;     // split only: mean cycles of the wait
    unsigned sm;
};

__device__ __forceinline__ void spin(long long cycles) {
    long long t = clock64();
    while (clock64() - t < cycles) {}
}

// ---------------------------------------------------------------- the pieces

// Barriers and fences: nothing in, nothing out.
using Sync = void (*)();

__device__ __forceinline__ void full_sync() {   // = cluster.sync()
    asm volatile("barrier.cluster.arrive.release.aligned;\n\t"
                 "barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void relaxed_sync() {   // no release fence
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\t"
                 "barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void fence_cta() {
    asm volatile("fence.acq_rel.cta;" ::: "memory");
}
__device__ __forceinline__ void fence_cluster() {
    asm volatile("fence.acq_rel.cluster;" ::: "memory");
}
__device__ __forceinline__ void fence_gpu() {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
}
__device__ __forceinline__ void fence_release_cluster() {
    asm volatile("fence.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void fence_acquire_cluster() {
    asm volatile("fence.acquire.cluster;" ::: "memory");
}
// Fences restricted to shared memory (PTX 8.6): they promise to order only
// shared-memory accesses, so the release needs no GPU-scope fence.
__device__ __forceinline__ void fence_release_restricted() {
    asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
}
__device__ __forceinline__ void fence_acquire_restricted() {
    asm volatile("fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
}
// Orders ordinary accesses against the async proxy (TMA, bulk copies); not a
// cheaper cluster fence.
__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cluster;" ::: "memory");
}
// The release done by hand: a cluster fence, then the relaxed barrier.
__device__ __forceinline__ void fence_then_relaxed_sync() {
    asm volatile("fence.acq_rel.cluster;\n\t"
                 "barrier.cluster.arrive.relaxed.aligned;\n\t"
                 "barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// WARM untimed executions of SYNC, then the mean of ITERS timed ones. One clock
// pair around the whole loop, not one per execution: a wait can finish after
// the clock read that follows it (the early-read problem of barriers.cu), and
// with a pair per execution that time could fall between two windows and be
// lost.
template <Sync SYNC>
__device__ __forceinline__ long long mean_cycles() {
    for (int it = 0; it < WARM; it++) SYNC();
    long long t0 = clock64();
    for (int it = 0; it < ITERS; it++) SYNC();
    long long t1 = clock64();
    return (t1 - t0) / ITERS;
}

// ---------------------------------------------------------------- kernels

// parts / placement: every block times SYNC.
template <Sync SYNC>
__global__ void time_sync(Out* out) {
    long long cycles = mean_cycles<SYNC>();
    if (threadIdx.x == 0) out[blockIdx.x] = {cycles, 0, smid()};
}

// The relaxed barrier split in two: the arrive alone, then a spin long enough
// for every block to have arrived, then the wait on a barrier already complete.
__global__ void time_split(Out* out) {
    long long arrive = 0, wait = 0;
    for (int it = 0; it < WARM + ITERS; it++) {
        long long t0 = clock64();
        asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
        long long t1 = clock64();
        spin(5000);
        long long t2 = clock64();
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        long long t3 = clock64();
        arrive += (it >= WARM) * (t1 - t0);   // multiply instead of branching
        wait += (it >= WARM) * (t3 - t2);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = {arrive / ITERS, wait / ITERS, smid()};
}

// ---------------------------------------------------------------- host

// One launch of `blocks` blocks in clusters of `cluster_size`; false if the
// cluster cannot be scheduled.
bool launch(const void* kernel, int blocks, int threads, int cluster_size, void** args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = blocks;
    cfg.blockDim = threads;
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = cluster_size;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    allow_big_clusters(kernel);
    if (cudaLaunchKernelExC(&cfg, kernel, args) != cudaSuccess) {
        cudaGetLastError();   // not sticky: clear it
        return false;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

long long median(std::vector<long long> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Named {
    const char* name;
    const void* kernel;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: fence parts | placement [--cluster-size N] [--block-size N]\n");
        return 1;
    }
    const char* experiment = argv[1];
    Args a;
    a.cluster_size = 12;
    a.block_size = 32;
    parse_args(argc - 1, argv + 1, &a, "fence",
               "  --cluster-size N  blocks per cluster, 1..12              (default 12)\n"
               "  --block-size N    threads per block, 32..1024            (default 32)\n");
    Out* out;
    CUDA_CHECK(cudaMallocManaged(&out, SMS * sizeof(Out)));

    const Named pieces[] = {
        {"cluster.sync()", (const void*)time_sync<full_sync>},
        {"relaxed arrive + wait", (const void*)time_sync<relaxed_sync>},
        {"fence.acq_rel.cta", (const void*)time_sync<fence_cta>},
        {"fence.acq_rel.cluster", (const void*)time_sync<fence_cluster>},
        {"fence.acq_rel.gpu", (const void*)time_sync<fence_gpu>},
        {"fence.release.cluster", (const void*)time_sync<fence_release_cluster>},
        {"fence.acquire.cluster", (const void*)time_sync<fence_acquire_cluster>},
        {"fence.acq_rel.cluster + relaxed arrive + wait", (const void*)time_sync<fence_then_relaxed_sync>},
        {"fence.release.sync_restrict::shared::cta.cluster", (const void*)time_sync<fence_release_restricted>},
        {"fence.acquire.sync_restrict::shared::cluster.cluster", (const void*)time_sync<fence_acquire_restricted>},
        {"fence.proxy.async.shared::cluster", (const void*)time_sync<fence_proxy_async>},
    };

    if (strcmp(experiment, "parts") == 0) {
        printf("benchmark,piece,cluster_size,block_size,median_cycles,min_cycles,max_cycles,median_wait\n");
        void* args[] = {&out};
        for (const Named& p : pieces) {
            if (!launch(p.kernel, a.cluster_size, a.block_size, a.cluster_size, args)) {
                fprintf(stderr, "fence: a cluster of %d cannot be scheduled\n", a.cluster_size);
                return 1;
            }
            std::vector<long long> v;
            for (int b = 0; b < a.cluster_size; b++) v.push_back(out[b].cycles);
            printf("fence,%s,%d,%d,%lld,%lld,%lld,\n", p.name, a.cluster_size, a.block_size, median(v),
                   *std::min_element(v.begin(), v.end()), *std::max_element(v.begin(), v.end()));
        }
        // arrive alone | wait on a completed barrier
        if (!launch((const void*)time_split, a.cluster_size, a.block_size, a.cluster_size, args)) {
            fprintf(stderr, "fence: a cluster of %d cannot be scheduled\n", a.cluster_size);
            return 1;
        }
        std::vector<long long> arrive, wait;
        for (int b = 0; b < a.cluster_size; b++) {
            arrive.push_back(out[b].cycles);
            wait.push_back(out[b].wait);
        }
        printf("fence,split: arrive alone | wait on a completed barrier,%d,%d,%lld,%lld,%lld,%lld\n",
               a.cluster_size, a.block_size, median(arrive),
               *std::min_element(arrive.begin(), arrive.end()),
               *std::max_element(arrive.begin(), arrive.end()), median(wait));
    } else if (strcmp(experiment, "placement") == 0) {
        printf("benchmark,piece,cluster_size,block_size,block,cluster,sm,cycles\n");
        for (int i : {0, 1, 4}) {   // cluster.sync(), relaxed, the GPU fence
            void* args[] = {&out};
            if (!launch(pieces[i].kernel, SMS, a.block_size, a.cluster_size, args)) {
                fprintf(stderr, "fence: clusters of %d cannot be scheduled\n", a.cluster_size);
                return 1;
            }
            for (int b = 0; b < SMS; b++)
                printf("fence,%s,%d,%d,%d,%d,%u,%lld\n", pieces[i].name, a.cluster_size, a.block_size, b,
                       b / a.cluster_size, out[b].sm, out[b].cycles);
        }
    } else {
        fprintf(stderr, "usage: fence parts | placement [--cluster-size N] [--block-size N]\n");
        return 1;
    }
    return 0;
}
