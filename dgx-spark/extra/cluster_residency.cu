// cluster_residency.cu — how many blocks run AT THE SAME TIME, with and
// without thread block clusters?
//
// For every case (block size, shared memory per block, cluster size) we
// collect the result in two ways:
//   API: what the CUDA occupancy API says can be resident at once;
//   measured:  what really happens. We launch the hardware maximum of blocks
//              (48 SMs x 24 blocks = 1152), every block records its SM and
//              its start/end time, and we count how many blocks were alive
//              at the same instant.
//
// Result on GB10: without clusters, small blocks fit 24 per SM (1152 on the
// GPU). With clusters, at most 96 blocks run at once (2 per SM), whatever the
// cluster size and however small the blocks. The other blocks wait and run
// in later "waves".
//
// Build: make cluster_residency   (or: nvcc -O3 -arch=sm_121 cluster_residency.cu -o cluster_residency)
// Run:   ./cluster_residency

#include <cstdio>
#include <vector>
#include <algorithm>
#include <string>
#include <cooperative_groups.h>
#include "../common.cuh"   // CUDA_CHECK, smid(), globaltimer()

namespace cg = cooperative_groups;

// How long every block stays alive: ~4 ms at 2.4 GHz. Starting a block takes
// microseconds, so every block that CAN run at the same time as the others
// will have started long before the first one finishes.
constexpr long long WAIT_CYCLES = 10000000;

// Records the SM id and start/end times of each block, to check how many
// blocks are alive on each SM at once.
__global__ void record(unsigned* smids, unsigned long long* t_start, unsigned long long* t_end) {
    cg::cluster_group c = cg::this_cluster();   // without clusters: a cluster of 1 block
    unsigned long long t0 = globaltimer();
    long long s = clock64();
    // wait a bit to be sure all blocks are alive at the same time.
    while (clock64() - s < WAIT_CYCLES);
    c.sync();
    if (threadIdx.x == 0) {
        smids[blockIdx.x] = smid();
        t_start[blockIdx.x] = t0;
        t_end[blockIdx.x] = globaltimer();
    }
}

// Launch configuration for `grid` blocks of `block_size` threads with
// `smem` bytes of shared memory each. cluster_size = 1 means no clusters.
cudaLaunchConfig_t make_config(int grid, int block_size, int smem, int cluster_size,
                               cudaLaunchAttribute* attr) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(block_size);
    cfg.dynamicSmemBytes = smem;
    attr->id = cudaLaunchAttributeClusterDimension;
    attr->val.clusterDim.x = cluster_size;
    attr->val.clusterDim.y = 1;
    attr->val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = (cluster_size > 1) ? 1 : 0;   // no attribute = normal launch
    return cfg;
}

// What the occupancy API predicts: blocks resident on the whole GPU at once.
int predicted(int block_size, int smem, int cluster_size, int num_sms) {
    if (cluster_size == 1) {
        int per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, record, block_size, smem));
        return per_sm * num_sms;
    }
    // for clusters the API answers "how many whole clusters fit on the GPU"
    cudaLaunchAttribute attr;
    cudaLaunchConfig_t cfg = make_config(cluster_size, block_size, smem, cluster_size, &attr);
    int clusters = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveClusters(&clusters, (void*)record, &cfg));
    return clusters * cluster_size;
}

// What really happens: launch `grid` blocks and find the most blocks alive at
// the same instant, on the whole GPU and on a single SM.
void measured(int grid, int block_size, int smem, int cluster_size, int* max_gpu, int* max_sm) {
    // smids contains the SM id of block i
    // ts contains the start time of block i, te its end time
    unsigned* smids;
    unsigned long long *ts, *te;
    CUDA_CHECK(cudaMallocManaged(&smids, grid * sizeof(unsigned)));
    CUDA_CHECK(cudaMallocManaged(&ts, grid * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMallocManaged(&te, grid * sizeof(unsigned long long)));

    cudaLaunchAttribute attr;
    cudaLaunchConfig_t cfg = make_config(grid, block_size, smem, cluster_size, &attr);
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, record, smids, ts, te));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Take a snapshot at the start of every block i and count who was alive.
    // The number of alive blocks only grows when a block starts, so the peak
    // is always at one of these instants.
    *max_gpu = 0;
    *max_sm = 0;
    for (int i = 0; i < grid; i++) {
        int gpu = 0, same_sm = 0;
        for (int j = 0; j < grid; j++) {
            // block j had started and not yet finished when block i started:
            // |---j---|
            //    |---i---|
            if (ts[j] <= ts[i] && ts[i] < te[j]) {
                gpu++;
                if (smids[j] == smids[i]) same_sm++;
            }
        }
        *max_gpu = std::max(*max_gpu, gpu);
        *max_sm = std::max(*max_sm, same_sm);
    }
    CUDA_CHECK(cudaFree(smids));
    CUDA_CHECK(cudaFree(ts));
    CUDA_CHECK(cudaFree(te));
}

// A horizontal line of `width` dashes, to separate groups of rows.
void separator(int width) {
    printf("%s\n", std::string(width, '-').c_str());
}

// One table row: prediction and measurement for one case.
void row(int grid, int block_size, int smem, int cluster_size, int num_sms) {
    int max_gpu, max_sm;
    measured(grid, block_size, smem, cluster_size, &max_gpu, &max_sm);
    char cluster[8];
    if (cluster_size == 1) snprintf(cluster, sizeof cluster, "none");
    else                   snprintf(cluster, sizeof cluster, "%d", cluster_size);
    printf("%10d | %10d | %7s | %6d | %8d | %12d\n", block_size, smem, cluster,
           predicted(block_size, smem, cluster_size, num_sms), max_gpu, max_sm);
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int num_sms = prop.multiProcessorCount;
    int max_blocks_per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, 0));
    // The most blocks the GPU could ever hold at once (1152 on GB10). It is a
    // multiple of every cluster size used below, so the grid always splits
    // into whole clusters.
    const int grid = num_sms * max_blocks_per_sm;

    // Clusters bigger than 8 and shared memory above 48 KiB need an opt-in.
    CUDA_CHECK(cudaFuncSetAttribute((void*)record, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CUDA_CHECK(cudaFuncSetAttribute((void*)record, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)prop.sharedMemPerBlockOptin));

    printf("%s: %d SMs, at most %d blocks per SM. Every case launches %d blocks.\n\n",
           prop.name, num_sms, max_blocks_per_sm, grid);
    // printf returns how many characters it printed: minus the newline, that
    // is the width of the table, so the separator lines always match it.
    const int width = printf("%10s | %10s | %7s | %6s | %8s | %12s\n", "block size",
                             "smem bytes", "cluster", "API", "measured", "max block/SM") - 1;
    separator(width);

    // 1. Block size: small blocks fit many per SM, unless they are clustered.
    for (int block_size : {64, 256, 1024}) {
        for (int cluster_size : {1, 2, 4, 12})
            row(grid, block_size, 0, cluster_size, num_sms);
        separator(width);
    }
    separator(width);
    // 2. Shared memory: above ~50 KiB per block only one block fits per SM,
    //    and a cluster size that does not divide the 12 SMs of a GPC (8)
    //    leaves SMs empty, because a cluster cannot span two GPCs.
    for (int smem : {32 * 1024, 50 * 1024, (int)prop.sharedMemPerBlockOptin}) {
        for (int cluster_size : {1, 4, 8, 12})
            row(grid, 128, smem, cluster_size, num_sms);
        separator(width);
    }
    return 0;
}
