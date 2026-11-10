// l2-flooding.cu — do DSMEM and the cluster barrier go through L2? One cluster
// of --cluster-size blocks measures while every other block of the GPU floods
// L2. If what is measured shared L2 with the flood, it would slow down.
//
// THE FLOODS (one row each)
//   idle        only the measuring cluster is launched
//   L2 loads    every other block reads 16 B at a time, straight from L2
//   L2 atomics  every other block does atomicAdd, executed in L2
//
// WHAT IS MEASURED
//   L2 load (control)      a dependent chain of L2 loads: it must get slower,
//                          otherwise the flood is not flooding anything
//   DSMEM load             a dependent chain of loads from the next block's
//                          shared memory
//   cluster.sync()         the full barrier
//   relaxed arrive + wait  the barrier without its release fence
//   fence.acq_rel.gpu      the fence alone
//
//   With --cluster-size 2 the flooding blocks include SMs of the same GPC; with
//   --cluster-size 12 the measuring cluster owns its whole GPC.
//
// THE MEASUREMENT
//   As in fence.cu: the mean of ITERS back-to-back executions after WARM untimed
//   ones, with one clock pair around the whole loop. Each piece and each flood
//   is its own __device__ function, passed to the kernel as a template argument,
//   so the timed loop has no branch. How hard the flood hits varies from run to
//   run; look at each row's change from idle.
//
// Build: make l2-flooding
// Run:   build/l2-flooding --cluster-size 2

#include <cooperative_groups.h>

#include "../common.cuh"   // CUDA_CHECK, Args, parse_args, smid(), allow_big_clusters()

namespace cg = cooperative_groups;

constexpr int ITERS = 1000;   // timed executions of a piece
constexpr int WARM = 100;     // untimed executions before them
constexpr int SMS = 48;       // GB10

// What the two control loads need: a word that holds 0, so that each load can
// use the value of the previous one as its address offset.
struct Ctx {
    const unsigned* l2;         // a zeroed word in global memory (read with ld.cg: L2)
    const unsigned* remote;     // a zeroed word in the next block's shared memory
};

// What one block measured.
struct Out {
    long long cycles;   // mean cycles of the piece
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
__device__ __forceinline__ void fence_gpu() {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

// The control loads: they take the previous value and return the next one,
// so each load waits for the one before (latency, not throughput).
using Load = unsigned (*)(const Ctx&, unsigned);

__device__ __forceinline__ unsigned l2_load(const Ctx& c, unsigned v) {
    return __ldcg(c.l2 + v);   // v is always 0: the address depends on the last load
}
__device__ __forceinline__ unsigned dsmem_load(const Ctx& c, unsigned v) {
    return *((volatile const unsigned*)c.remote + v);
}

// One clock pair around the whole loop, not one per execution: a wait can
// finish after the clock read that follows it (the early-read problem of
// barriers.cu), and with a pair per execution that time could fall between
// two windows and be lost.

// WARM untimed executions of SYNC, then the mean of ITERS timed ones.
template <Sync SYNC>
__device__ __forceinline__ long long mean_cycles() {
    for (int it = 0; it < WARM; it++) SYNC();
    long long t0 = clock64();
    for (int it = 0; it < ITERS; it++) SYNC();
    long long t1 = clock64();
    return (t1 - t0) / ITERS;
}

// The same for a chain of dependent loads.
template <Load LOAD>
__device__ __forceinline__ long long mean_load_cycles(const Ctx& c, unsigned* sink) {
    unsigned v = 0;
    for (int it = 0; it < WARM; it++) v = LOAD(c, v);
    long long t0 = clock64();
    for (int it = 0; it < ITERS; it++) v = LOAD(c, v);
    long long t1 = clock64();
    if (v == 12345) *sink = v;   // keeps the loads alive
    return (t1 - t0) / ITERS;
}

// ---------------------------------------------------------------- kernels

// The first `cluster_size` blocks (one cluster) measure while every other block
// runs HAMMER on L2 until they are done.
__device__ volatile int stop_flag;

__device__ __forceinline__ void hammer_loads(const uint4* buf, unsigned* abuf, size_t n) {
    uint4 acc = {0, 0, 0, 0};
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    while (!stop_flag)
        for (int k = 0; k < 64; k++) {
            uint4 x = __ldcg(&buf[i]);   // 16 B, straight from L2
            acc.x ^= x.x;
            acc.y ^= x.w;
            i = (i + gridDim.x * blockDim.x) & (n - 1);
        }
    if (acc.x == 0x12345678) abuf[0] = acc.y;   // keeps the loads alive
}
__device__ __forceinline__ void hammer_atomics(const uint4* buf, unsigned* abuf, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    while (!stop_flag)
        for (int k = 0; k < 64; k++) {
            atomicAdd(&abuf[(i * 33) & (n - 1)], 1u);   // executed in L2
            i = (i + gridDim.x * blockDim.x) & (n - 1);
        }
}
using Hammer = void (*)(const uint4*, unsigned*, size_t);

// The end of both kernels: wait for the whole measuring cluster,
// record, then let the hammers stop.
__device__ __forceinline__ void report_and_stop(long long cycles, Out* out) {
    cg::this_cluster().sync();   // every block of the cluster is done (and its
                                 // shared memory outlives the neighbour's loads)
    if (threadIdx.x == 0) out[blockIdx.x] = {cycles, smid()};
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        spin(1000);
        stop_flag = 1;   // the hammers may stop
    }
}

template <Sync SYNC, Hammer HAMMER>
__global__ void flood_sync(const uint4* buf, unsigned* abuf, size_t n, int cluster_size,
                               Out* out) {
    if (blockIdx.x >= (unsigned)cluster_size) {   // a hammer block
        HAMMER(buf, abuf, n);
        return;
    }
    spin(200000);   // let the hammers get going
    long long cycles = mean_cycles<SYNC>();
    report_and_stop(cycles, out);
}

template <Load LOAD, Hammer HAMMER>
__global__ void flood_load(const uint4* buf, unsigned* abuf, size_t n, int cluster_size,
                               Out* out, const unsigned* zero, unsigned* sink) {
    if (blockIdx.x >= (unsigned)cluster_size) {   // a hammer block
        HAMMER(buf, abuf, n);
        return;
    }
    // The word of the DSMEM control holds 0 in every block of the cluster.
    __shared__ unsigned word;
    cg::cluster_group cluster = cg::this_cluster();
    if (threadIdx.x == 0) word = 0;
    cluster.sync();
    Ctx c;
    c.l2 = zero;
    c.remote = cluster.map_shared_rank(&word, ((int)cluster.block_rank() + 1) % cluster_size);
    spin(200000);   // let the hammers get going
    long long cycles = mean_load_cycles<LOAD>(c, sink);
    report_and_stop(cycles, out);
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

// One row per flood. kernels[0] floods L2 with loads, kernels[1]
// with atomics; "idle" launches kernels[0] with the measuring cluster only.
bool flood_rows(const Named* kernels, void** args, int cluster_size, Out* out) {
    const char* flood[] = {"idle", "L2 loads", "L2 atomics"};
    for (int f = 0; f < 3; f++) {
        int stop = 0;
        CUDA_CHECK(cudaMemcpyToSymbol(stop_flag, &stop, sizeof stop));
        const int blocks = (f == 0) ? cluster_size : SMS;
        if (!launch(kernels[f == 2].kernel, blocks, 256, cluster_size, args)) return false;
        std::vector<long long> v;
        for (int b = 0; b < cluster_size; b++) v.push_back(out[b].cycles);
        printf("l2-flooding,%s,%s,%d,%lld,%lld,%lld\n", kernels[0].name, flood[f], cluster_size, median(v),
               *std::min_element(v.begin(), v.end()), *std::max_element(v.begin(), v.end()));
    }
    return true;
}

int main(int argc, char** argv) {
    Args a;
    parse_args(argc, argv, &a, "l2-flooding",
               "  --cluster-size N  blocks of the measuring cluster, 2..12  (default 2)\n");
    Out* out;
    CUDA_CHECK(cudaMallocManaged(&out, SMS * sizeof(Out)));

    const size_t n = (8u << 20) / sizeof(uint4);   // 8 MiB: stays in L2
    uint4* buf;
    unsigned *abuf, *zero;
    CUDA_CHECK(cudaMalloc(&buf, n * sizeof(uint4)));
    CUDA_CHECK(cudaMemset(buf, 0, n * sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&abuf, n * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&zero, sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(zero, 0, sizeof(unsigned)));
    unsigned* sink;
    CUDA_CHECK(cudaMalloc(&sink, sizeof(unsigned)));
    size_t nn = n;
    int cs = a.cluster_size;
    // [piece][0 = flood with loads, 1 = flood with atomics]
    const Named loads[][2] = {
        {{"L2 load (control)", (const void*)flood_load<l2_load, hammer_loads>},
         {"L2 load (control)", (const void*)flood_load<l2_load, hammer_atomics>}},
        {{"DSMEM load", (const void*)flood_load<dsmem_load, hammer_loads>},
         {"DSMEM load", (const void*)flood_load<dsmem_load, hammer_atomics>}},
    };
    const Named syncs[][2] = {
        {{"cluster.sync()", (const void*)flood_sync<full_sync, hammer_loads>},
         {"cluster.sync()", (const void*)flood_sync<full_sync, hammer_atomics>}},
        {{"relaxed arrive + wait", (const void*)flood_sync<relaxed_sync, hammer_loads>},
         {"relaxed arrive + wait", (const void*)flood_sync<relaxed_sync, hammer_atomics>}},
        {{"fence.acq_rel.gpu", (const void*)flood_sync<fence_gpu, hammer_loads>},
         {"fence.acq_rel.gpu", (const void*)flood_sync<fence_gpu, hammer_atomics>}},
    };
    void* load_args[] = {&buf, &abuf, &nn, &cs, &out, &zero, &sink};
    void* sync_args[] = {&buf, &abuf, &nn, &cs, &out};
    printf("benchmark,piece,flood,cluster_size,median_cycles,min_cycles,max_cycles\n");
    bool ok = true;
    for (const Named* k : loads) ok = ok && flood_rows(k, load_args, a.cluster_size, out);
    for (const Named* k : syncs) ok = ok && flood_rows(k, sync_args, a.cluster_size, out);
    if (!ok) {
        fprintf(stderr, "l2-flooding: clusters of %d cannot be scheduled\n", a.cluster_size);
        return 1;
    }
    return 0;
}
