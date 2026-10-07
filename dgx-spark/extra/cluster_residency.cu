#include <cstdio>
#include <vector>
#include <algorithm>
#include <cooperative_groups.h>
#include "../common.cuh"

namespace cg = cooperative_groups;

__device__ const int WAIT_CYCLES = 100000000;

// records the SM id and start/end times of each block, to check how many blocks are alive on each SM at once.
__global__ void record(unsigned* smids, unsigned long long* t_start, unsigned long long* t_end) {
    cg::cluster_group c = cg::this_cluster();
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
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int numSMs = prop.multiProcessorCount;
    int maxBlocksPerSM = 0;
    cudaDeviceGetAttribute(&maxBlocksPerSM, cudaDevAttrMaxBlocksPerMultiprocessor, 0);

    const int cs = 12;
    const int grid = numSMs * maxBlocksPerSM; // number of blocks to launch (max for DGX Spark is 1152, 48 SMs * 24 blocks per SM)
    //smids contains the SM id of block i
    unsigned* smids; 
    //ts contains the start time of block i
    //te contains the end time of block i
    unsigned long long *ts, *te;
    
    cudaFuncSetAttribute((void*)record, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    
    cudaMallocManaged(&smids, grid * 4);
    cudaMallocManaged(&ts, grid * 8);
    cudaMallocManaged(&te, grid * 8);
    
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(64);
    cudaLaunchAttribute a[1];
    a[0].id = cudaLaunchAttributeClusterDimension;
    a[0].val.clusterDim.x = cs;
    a[0].val.clusterDim.y = 1;
    a[0].val.clusterDim.z = 1;
    cfg.attrs = a;
    cfg.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, record, smids, ts, te));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // find the earliest start time
    unsigned long long t_first = *std::min_element(ts, ts + grid);
    // blocks per SM over the whole kernel
    std::vector<int> total(numSMs, 0);
    for (int i = 0; i < grid; i++)
        total[smids[i]]++;
    // most blocks alive on one SM at the same instant (check at every block's start time)
    int max_together = 0, max_gpu = 0;
    for (int i = 0; i < grid; i++) {
        int gpu = 0, same_sm = 0;
        for (int j = 0; j < grid; j++) {
            // if block j started before block i and block i started before block j ended, then they were alive at the same time!
            // |---j---|
            //    |---i---|
            if (ts[j] <= ts[i] && ts[i] < te[j]) {
                gpu++;
                if (smids[j] == smids[i]) same_sm++;
            }
        }
        max_together = std::max(max_together, same_sm);
        max_gpu = std::max(max_gpu, gpu);
    }
    std::vector<unsigned long long> starts(ts, ts + grid);
    std::sort(starts.begin(), starts.end());
    
    int waves = 1;
    for (int i = 1; i < grid; i++)
        if (starts[i] - starts[i-1] > WAIT_CYCLES)
            waves++;   // gap > WAIT_CYCLES = new wave
    
    printf("blocks per SM over the whole kernel : min %d, max %d\n", *std::min_element(total.begin(), total.end()), *std::max_element(total.begin(), total.end()));
    printf("most blocks alive on one SM at once : %d\n", max_together);
    printf("most blocks alive on the GPU at once: %d\n", max_gpu);
    printf("waves of block starts               : %d\n", waves);
    printf("kernel duration                     : %.1f ms  (one block alone: ~%.1f ms)\n",
           (*std::max_element(te, te + grid) - t_first) / 1e6, (te[0] - ts[0]) / 1e6);
}