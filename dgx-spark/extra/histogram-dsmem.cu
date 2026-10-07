// histogram-dsmem.cu — is a DSMEM histogram faster than the alternatives?
//
// NVIDIA's motivating example for distributed shared memory is a histogram
// with more bins than one block's shared memory can hold: a cluster keeps the
// histogram in shared memory, split across its blocks, instead of falling
// back to atomics on global memory. Three ways to build the same histogram:
//
//   global   every element is an atomicAdd on the histogram in global memory
//            (served by L2). Works for any number of bins.
//   private  every block keeps a full copy of the histogram in its own shared
//            memory (fast local atomics), then adds it to global memory.
//            Only possible while all the bins fit in one block.
//   DSMEM    the bins are split across the blocks of a cluster; an element
//            whose bin lives on another block is counted with an atomicAdd on
//            that block's shared memory (map_shared_rank).
//
// Result on GB10: when the bins fit, the private copy is fastest; when they
// do not, global atomics beat DSMEM by 4-7x. Every DSMEM atomic is a separate
// 4-byte request through the SM-to-SM network, which handles only a few
// billion such requests per second for the whole chip, while L2 atomics run
// at ~27 billion per second.
//
// Build: make histogram-dsmem   (or: nvcc -O3 -arch=sm_121 histogram-dsmem.cu -o histogram-dsmem)
// Run:   ./histogram-dsmem

#include <cstdio>
#include <vector>
#include <algorithm>
#include <cooperative_groups.h>
#include "../common.cuh"   // CUDA_CHECK

namespace cg = cooperative_groups;

constexpr int ARRAY_SIZE = 32 * 1024 * 1024;   // elements to count
constexpr int GRID = 3072;                     // blocks: divisible by every cluster size used
constexpr int BLOCK = 512;                     // threads per block
constexpr int REPS = 11;                       // timed runs; the median is reported

// Special "times" for a table cell that holds no measurement.
constexpr double WRONG = -1;          // the histogram came out incorrect
constexpr double DOES_NOT_FIT = -2;   // the version cannot run with this many bins

// ---------------------------------------------------------------- kernels
// All three use a grid-stride loop: thread t handles elements t, t + stride,
// t + 2*stride, ... so any grid size covers the whole input.

__global__ void hist_global(int* bins, const int* input, int n) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&bins[input[i]], 1);
}

__global__ void hist_private(int* bins, const int* input, int n, int nbins) {
    extern __shared__ int local[];             // this block's copy of all the bins
    for (int b = threadIdx.x; b < nbins; b += blockDim.x) local[b] = 0;
    __syncthreads();

    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&local[input[i]], 1);        // shared-memory atomic, on this SM
    __syncthreads();

    for (int b = threadIdx.x; b < nbins; b += blockDim.x)
        if (local[b]) atomicAdd(&bins[b], local[b]);
}

__global__ void hist_dsmem(int* bins, const int* input, int n, int bins_per_block) {
    extern __shared__ int local[];             // this block's share of the bins
    cg::cluster_group cluster = cg::this_cluster();
    for (int b = threadIdx.x; b < bins_per_block; b += blockDim.x) local[b] = 0;
    cluster.sync();                            // every block's share is zeroed

    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        int bin = input[i];
        int owner = bin / bins_per_block;      // which block of the cluster holds this bin
        int* owner_bins = cluster.map_shared_rank(local, owner);
        atomicAdd(&owner_bins[bin % bins_per_block], 1);   // usually on ANOTHER SM
    }
    cluster.sync();                            // all remote atomics have landed

    // Bins of rank r are bins r*bins_per_block ... (r+1)*bins_per_block - 1.
    int* my_bins = bins + cluster.block_rank() * bins_per_block;
    for (int b = threadIdx.x; b < bins_per_block; b += blockDim.x)
        if (local[b]) atomicAdd(&my_bins[b], local[b]);
}

// ---------------------------------------------------------------- host

// Random input spread uniformly over `nbins` bins, copied to the GPU.
void fill_input(int* d_input, int nbins) {
    std::vector<int> h(ARRAY_SIZE);
    unsigned seed = 12345;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        seed = seed * 1664525u + 1013904223u;  // simple random generator
        h[i] = (int)((seed >> 8) % nbins);
    }
    CUDA_CHECK(cudaMemcpy(d_input, h.data(), ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice));
}

// Runs `launch` (which must clear and fill d_bins) REPS times and returns the
// median time in ms, or WRONG if the histogram is wrong: every element must be
// counted exactly once, so the bins must add up to ARRAY_SIZE.
template <class Launch>
double time_ms(Launch launch, int* d_bins, int nbins) {
    launch();                                  // warm-up, and the correctness check
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> h(nbins);
    CUDA_CHECK(cudaMemcpy(h.data(), d_bins, nbins * sizeof(int), cudaMemcpyDeviceToHost));
    long long total = 0;
    for (int v : h) total += v;
    if (total != ARRAY_SIZE) return WRONG;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> times;
    for (int r = 0; r < REPS; r++) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(times.begin(), times.end());
    return times[REPS / 2];
}

// One table cell: the time, '-' if the version cannot run, 'WRONG' if the
// histogram was incorrect.
void cell(double ms) {
    if (ms == DOES_NOT_FIT) printf(" %8s", "-");
    else if (ms == WRONG)   printf(" %8s", "WRONG");
    else                    printf(" %8.3f", ms);
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int max_smem = (int)prop.sharedMemPerBlockOptin;   // most shared memory per block

    // Every bin count is divisible by 2, 4, 8 and 12, so the bins split
    // evenly across the blocks of any cluster tested.
    const std::vector<int> bin_counts = {1536, 6144, 24576, 49152, 98304, 196608, 393216};
    const int max_bins = bin_counts.back();

    int *d_input, *d_bins;
    CUDA_CHECK(cudaMalloc(&d_input, ARRAY_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bins, max_bins * sizeof(int)));

    // Opt-ins: clusters bigger than 8, and shared memory above 48 KiB.
    CUDA_CHECK(cudaFuncSetAttribute(hist_dsmem, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CUDA_CHECK(cudaFuncSetAttribute(hist_dsmem, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem));
    CUDA_CHECK(cudaFuncSetAttribute(hist_private, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem));

    printf("%s: %d elements, %d blocks of %d threads. Time in ms, '-' = does not fit.\n\n",
           prop.name, ARRAY_SIZE, GRID, BLOCK);
    printf("%8s %6s | %8s %8s | %8s %8s %8s %8s\n", "bins", "KiB", "global", "private",
           "DSMEM 2", "DSMEM 4", "DSMEM 8", "DSMEM 12");

    for (int nbins : bin_counts) {
        fill_input(d_input, nbins);
        const size_t hist_bytes = nbins * sizeof(int);
        printf("%8d %6d |", nbins, (int)(hist_bytes / 1024));

        // global atomics: no shared memory, always possible
        cell(time_ms([&] {
            CUDA_CHECK(cudaMemset(d_bins, 0, hist_bytes));
            hist_global<<<GRID, BLOCK>>>(d_bins, d_input, ARRAY_SIZE);
        }, d_bins, nbins));

        // private copy: only if the whole histogram fits in one block
        if ((int)hist_bytes > max_smem) cell(DOES_NOT_FIT);
        else cell(time_ms([&] {
            CUDA_CHECK(cudaMemset(d_bins, 0, hist_bytes));
            hist_private<<<GRID, BLOCK, hist_bytes>>>(d_bins, d_input, ARRAY_SIZE, nbins);
        }, d_bins, nbins));
        printf(" |");

        // DSMEM: only if one block's share of the bins fits in its shared memory
        for (int cluster_size : {2, 4, 8, 12}) {
            const int bins_per_block = nbins / cluster_size;
            const size_t share_bytes = bins_per_block * sizeof(int);
            if ((int)share_bytes > max_smem) { cell(DOES_NOT_FIT); continue; }

            cudaLaunchConfig_t cfg = {};
            cfg.gridDim = dim3(GRID);
            cfg.blockDim = dim3(BLOCK);
            cfg.dynamicSmemBytes = share_bytes;
            cudaLaunchAttribute attr;
            attr.id = cudaLaunchAttributeClusterDimension;
            attr.val.clusterDim.x = cluster_size;
            attr.val.clusterDim.y = 1;
            attr.val.clusterDim.z = 1;
            cfg.attrs = &attr;
            cfg.numAttrs = 1;
            cell(time_ms([&] {
                CUDA_CHECK(cudaMemset(d_bins, 0, hist_bytes));
                CUDA_CHECK(cudaLaunchKernelEx(&cfg, hist_dsmem, d_bins, (const int*)d_input,
                                              ARRAY_SIZE, bins_per_block));
            }, d_bins, nbins));
        }
        printf("\n");
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_bins));
    return 0;
}
