// histogram-proof.cu — does the 96-block cluster residency cap show up in a
// REAL workload, and what does it cost?
//
// Companion to histogram.cu. Three things are demonstrated here:
//
//  1. THE CAP IS NOT A LAUNCH LIMIT. A clustered histogram over a grid of
//     3072 blocks launches and completes perfectly well. The cap governs how
//     many blocks are resident *at one instant*; the rest run in later waves.
//     Any attempt to "prove" the limit by watching a big launch fail will
//     find nothing.
//
//  2. THE CAP IS DIRECTLY OBSERVABLE FROM INSIDE THE KERNEL. Each block
//     increments a live counter on entry and decrements it on exit, tracking
//     the running maximum with atomicMax. That peak is the number of blocks
//     the hardware actually had resident simultaneously. Run the same
//     histogram clustered and unclustered and compare. Note this measures an
//     observed peak, so it is a LOWER bound on the true residency: blocks
//     must overlap in time for the counter to see them, which is why each
//     block is given a substantial amount of work.
//
//  3. THE CAP HAS A PRICE, AND BLOCK SIZE IS THE LEVER. Fewer resident blocks
//     means fewer resident warps to hide memory latency. The block-size sweep
//     shows the clustered configuration recovering as blocks grow, exactly as
//     the microbenchmarks predict.
//
// The two kernels do the same total work. The unclustered one keeps a private
// full histogram in its own shared memory; the clustered one splits the bins
// across the ranks of its cluster and uses map_shared_rank + remote atomicAdd
// (the DSMEM pattern of histogram.cu). Both then merge into global memory.
// Timing runs are uninstrumented (the counters are a compile-time template
// parameter) so the residency probe cannot perturb the measured times.
//
// Build: nvcc -O3 -arch=sm_121 histogram-proof.cu -o histogram-proof
// Run:   ./histogram-proof

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

#define CK(c) do { cudaError_t e_ = (c); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

// nbins is divisible by every cluster size tested (2,4,6,8,12) so the bins
// split evenly across ranks.
static const int NBINS      = 1536;
static const int ARRAY_SIZE = 32 * 1024 * 1024;
static const int GRID       = 3072;    // divisible by 2,4,6,8,12

// ---------------------------------------------------------------- residency probe
struct Probe { int live; int peak; };

template <bool INSTRUMENT>
__device__ __forceinline__ void probe_enter(Probe* p) {
    if (INSTRUMENT && threadIdx.x == 0) {
        int live = atomicAdd(&p->live, 1) + 1;
        atomicMax(&p->peak, live);
    }
}
template <bool INSTRUMENT>
__device__ __forceinline__ void probe_exit(Probe* p) {
    if (INSTRUMENT && threadIdx.x == 0) atomicSub(&p->live, 1);
}

// ---------------------------------------------------------------- unclustered
// Private full histogram in this block's own shared memory, then global merge.
template <bool INSTRUMENT>
__global__ void hist_plain(int* __restrict__ bins, const int* __restrict__ input,
                           int arraySize, Probe* probe) {
    extern __shared__ int smem[];
    probe_enter<INSTRUMENT>(probe);

    for (int i = threadIdx.x; i < NBINS; i += blockDim.x) smem[i] = 0;
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < arraySize; i += stride)
        atomicAdd(&smem[input[i]], 1);          // local shared-memory atomic

    __syncthreads();
    for (int i = threadIdx.x; i < NBINS; i += blockDim.x)
        if (smem[i]) atomicAdd(&bins[i], smem[i]);

    probe_exit<INSTRUMENT>(probe);
}

// ---------------------------------------------------------------- clustered
// Bins are split across the cluster's ranks; a value whose bin lives on
// another rank is accumulated with a REMOTE shared-memory atomic through
// map_shared_rank — the DSMEM pattern from histogram.cu.
template <bool INSTRUMENT>
__global__ void hist_cluster(int* __restrict__ bins, const int* __restrict__ input,
                             int arraySize, int binsPerBlock, Probe* probe) {
    extern __shared__ int smem[];
    cg::cluster_group cluster = cg::this_cluster();
    probe_enter<INSTRUMENT>(probe);

    for (int i = threadIdx.x; i < binsPerBlock; i += blockDim.x) smem[i] = 0;
    cluster.sync();                              // every rank's bins are zeroed

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < arraySize; i += stride) {
        int binid = input[i];
        int dstBlock  = binid / binsPerBlock;    // which rank owns this bin
        int dstOffset = binid % binsPerBlock;
        int* dst = cluster.map_shared_rank(smem, dstBlock);
        atomicAdd(dst + dstOffset, 1);           // remote (or local) SMEM atomic
    }

    cluster.sync();                              // all remote writes landed
    int* out = bins + cluster.block_rank() * binsPerBlock;
    for (int i = threadIdx.x; i < binsPerBlock; i += blockDim.x)
        if (smem[i]) atomicAdd(&out[i], smem[i]);

    probe_exit<INSTRUMENT>(probe);
}

// ---------------------------------------------------------------- host helpers
struct Result { int peak; double ms; bool correct; };

// mode 0 = plain algorithm, plain launch
// mode 1 = plain algorithm, CLUSTERED launch (same work, same memory pattern;
//          the only difference is the residency cap -> isolates its cost)
// mode 2 = DSMEM algorithm (bins split across ranks, remote atomics)
static Result run(int mode, int clusterSize, int blockDim_,
                  int* d_bins, const int* d_input, Probe* d_probe, int reps) {
    const bool clustered = (mode == 2);
    const int binsPerBlock = clustered ? NBINS / clusterSize : NBINS;
    const size_t smemBytes = binsPerBlock * sizeof(int);
    Result r{};

    auto launch = [&](bool instrument) {
        CK(cudaMemset(d_bins, 0, NBINS * sizeof(int)));
        CK(cudaMemset(d_probe, 0, sizeof(Probe)));
        if (mode == 0) {
            void* k = instrument ? (void*)hist_plain<true> : (void*)hist_plain<false>;
            CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes));
            if (instrument) hist_plain<true><<<GRID, blockDim_, smemBytes>>>(d_bins, d_input, ARRAY_SIZE, d_probe);
            else            hist_plain<false><<<GRID, blockDim_, smemBytes>>>(d_bins, d_input, ARRAY_SIZE, d_probe);
        } else if (mode == 1) {
            // identical kernel to mode 0, but launched with a cluster dimension
            void* k = instrument ? (void*)hist_plain<true> : (void*)hist_plain<false>;
            CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes));
            CK(cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            cudaLaunchConfig_t cfg = {};
            cfg.gridDim = dim3(GRID, 1, 1);
            cfg.blockDim = dim3(blockDim_, 1, 1);
            cfg.dynamicSmemBytes = smemBytes;
            cudaLaunchAttribute attr;
            attr.id = cudaLaunchAttributeClusterDimension;
            attr.val.clusterDim.x = clusterSize;
            attr.val.clusterDim.y = 1; attr.val.clusterDim.z = 1;
            cfg.attrs = &attr; cfg.numAttrs = 1;
            if (instrument) CK(cudaLaunchKernelEx(&cfg, hist_plain<true>,  d_bins, d_input, ARRAY_SIZE, d_probe));
            else            CK(cudaLaunchKernelEx(&cfg, hist_plain<false>, d_bins, d_input, ARRAY_SIZE, d_probe));
        } else {
            void* k = instrument ? (void*)hist_cluster<true> : (void*)hist_cluster<false>;
            CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemBytes));
            // cluster sizes above the portable maximum of 8 need this opt-in
            CK(cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            cudaLaunchConfig_t cfg = {};
            cfg.gridDim = dim3(GRID, 1, 1);
            cfg.blockDim = dim3(blockDim_, 1, 1);
            cfg.dynamicSmemBytes = smemBytes;
            cudaLaunchAttribute attr;
            attr.id = cudaLaunchAttributeClusterDimension;
            attr.val.clusterDim.x = clusterSize;
            attr.val.clusterDim.y = 1;
            attr.val.clusterDim.z = 1;
            cfg.attrs = &attr; cfg.numAttrs = 1;
            if (instrument) CK(cudaLaunchKernelEx(&cfg, hist_cluster<true>,  d_bins, d_input, ARRAY_SIZE, binsPerBlock, d_probe));
            else            CK(cudaLaunchKernelEx(&cfg, hist_cluster<false>, d_bins, d_input, ARRAY_SIZE, binsPerBlock, d_probe));
        }
    };

    // (a) instrumented run -> peak resident blocks
    launch(true);
    CK(cudaDeviceSynchronize());
    Probe hp; CK(cudaMemcpy(&hp, d_probe, sizeof(Probe), cudaMemcpyDeviceToHost));
    r.peak = hp.peak;

    // correctness: every input element must land in exactly one bin
    std::vector<int> hb(NBINS);
    CK(cudaMemcpy(hb.data(), d_bins, NBINS * sizeof(int), cudaMemcpyDeviceToHost));
    long long total = 0; for (int v : hb) total += v;
    r.correct = (total == ARRAY_SIZE);

    // (b) uninstrumented timing runs
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<double> t;
    launch(false); CK(cudaDeviceSynchronize());               // warm-up
    for (int i = 0; i < reps; i++) {
        CK(cudaEventRecord(e0));
        launch(false);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        t.push_back(ms);
    }
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    std::sort(t.begin(), t.end());
    r.ms = t[t.size() / 2];
    return r;
}

int main() {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("# %s, %d SMs | %d elements, %d bins, grid %d blocks\n\n",
           prop.name, prop.multiProcessorCount, ARRAY_SIZE, NBINS, GRID);

    // random input spread over all bins
    std::vector<int> h_input(ARRAY_SIZE);
    unsigned seed = 12345;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        seed = seed * 1664525u + 1013904223u;
        h_input[i] = (int)((seed >> 8) % NBINS);
    }
    int *d_input, *d_bins; Probe* d_probe;
    CK(cudaMalloc(&d_input, ARRAY_SIZE * sizeof(int)));
    CK(cudaMalloc(&d_bins,  NBINS * sizeof(int)));
    CK(cudaMalloc(&d_probe, sizeof(Probe)));
    CK(cudaMemcpy(d_input, h_input.data(), ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    const int reps = 11;

    printf("=== 1. Peak SIMULTANEOUSLY RESIDENT blocks, measured inside the kernel ===\n");
    printf("(grid is %d blocks in every case; all of them run, in waves)\n\n", GRID);
    printf("%-34s %9s %14s %8s %10s %8s\n",
           "configuration", "blockDim", "peak resident", "waves", "time ms", "correct");
    for (int bd : {128, 256}) {
        Result p = run(0, 1, bd, d_bins, d_input, d_probe, reps);
        printf("%-34s %9d %14d %8.1f %10.3f %8s\n", "plain algo, plain launch", bd,
               p.peak, (double)GRID / p.peak, p.ms, p.correct ? "yes" : "NO");
        for (int cs : {2, 4, 8, 12}) {
            Result q = run(1, cs, bd, d_bins, d_input, d_probe, reps);
            char nm[80]; snprintf(nm, sizeof nm, "plain algo, CLUSTERED launch cs=%d", cs);
            printf("%-34s %9d %14d %8.1f %10.3f %8s\n", nm, bd, q.peak,
                   (double)GRID / q.peak, q.ms, q.correct ? "yes" : "NO");
        }
        for (int cs : {2, 4, 6, 8, 12}) {
            Result c = run(2, cs, bd, d_bins, d_input, d_probe, reps);
            char nm[80]; snprintf(nm, sizeof nm, "DSMEM algo, cluster size %d", cs);
            printf("%-34s %9d %14d %8.1f %10.3f %8s\n", nm, bd, c.peak,
                   (double)GRID / c.peak, c.ms, c.correct ? "yes" : "NO");
        }
        printf("\n");
    }

    printf("=== 2. Isolating the two costs (blockDim 256) ===\n");
    printf("Same algorithm under plain vs clustered launch isolates the residency\n");
    printf("cap; the DSMEM row adds remote atomics on top of it.\n\n");
    {
        Result a = run(0, 1, 256, d_bins, d_input, d_probe, reps);
        Result b = run(1, 4, 256, d_bins, d_input, d_probe, reps);
        Result c = run(2, 4, 256, d_bins, d_input, d_probe, reps);
        printf("  plain algo, plain launch      : %7.3f ms  (%d resident)\n", a.ms, a.peak);
        printf("  plain algo, clustered launch  : %7.3f ms  (%d resident)  -> residency cap costs %.2fx\n",
               b.ms, b.peak, b.ms / a.ms);
        printf("  DSMEM algo, cluster size 4    : %7.3f ms  (%d resident)  -> remote atomics add %.2fx more\n",
               c.ms, c.peak, c.ms / b.ms);
        printf("  total clustered-DSMEM penalty : %.2fx\n", c.ms / a.ms);
    }

    printf("\n=== 3. Block size as a lever (DSMEM algo, cluster size 4) ===\n\n");
    printf("%-34s %9s %14s %12s %10s\n",
           "configuration", "blockDim", "peak resident", "warps/SM", "time ms");
    for (int bd : {64, 128, 256, 512}) {
        Result c = run(2, 4, bd, d_bins, d_input, d_probe, reps);
        Result p = run(0, 1, bd, d_bins, d_input, d_probe, reps);
        printf("%-34s %9d %14d %12.1f %10.3f\n", "DSMEM algo, cluster size 4", bd, c.peak,
               (double)c.peak / prop.multiProcessorCount * (bd / 32.0), c.ms);
        printf("%-34s %9d %14d %12.1f %10.3f\n", "  plain launch (reference)", bd, p.peak,
               (double)p.peak / prop.multiProcessorCount * (bd / 32.0), p.ms);
    }

    CK(cudaFree(d_input)); CK(cudaFree(d_bins)); CK(cudaFree(d_probe));
    return 0;
}
