// cluster_residency.cu — how many blocks stay simultaneously resident when a
// kernel is launched as clusters, versus normally?
//
// Two independent methods that must agree:
//   A. The occupancy API. cudaOccupancyMaxActiveBlocksPerMultiprocessor for
//      the unclustered launch; cudaOccupancyMaxActiveClusters (how many
//      clusters of a given size fit on the whole device at once) for the
//      clustered one.
//   B. A rendezvous test that measures reality instead of trusting the API:
//      every block atomically registers its arrival, then spins until it has
//      seen ALL blocks of the grid arrive. That can only happen if the whole
//      grid is resident at the same time — if it is not, the blocks that did
//      launch wait in vain for blocks that cannot start, and the spin times
//      out. Stepping the grid size up until the first timeout finds the true
//      co-residency limit.
// The distinction matters for persistent kernels — kernels sized to have
// every block resident for their whole lifetime so they can synchronize
// grid-wide. Launch one block more than the residency limit and such a kernel
// deadlocks.
//
// Build: nvcc -O3 -arch=sm_121 cluster_residency.cu -o cluster_residency
// Run:   ./cluster_residency        (a few minutes: failed rendezvous = timeout)

#include <cstdio>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "../common.cuh"  // CUDA_CHECK

namespace cg = cooperative_groups;

// Every block signs in on a shared counter, then waits until ALL blocks have
// signed in. That is only possible if the whole grid runs at the same time:
// otherwise the running blocks wait for blocks that cannot start until they
// themselves finish. The timeout breaks that wait.
__global__ void rendezvous(int* arrived, int* all_met, int total_blocks, long long timeout_cycles) {
    if (threadIdx.x == 0) {                  // one thread represents the whole block
        // "I am here". atomicAdd makes read-add-write one indivisible step, so
        // no increment is lost when many blocks add at the same moment.
        atomicAdd(arrived, 1);
        long long start = clock64();         // stopwatch, in SM clock cycles
        // Wait until every block has signed in. Adding 0 changes nothing, but
        // atomicAdd returns the CURRENT value from memory; a plain read could
        // return an old copy cached in this SM and never see the others.
        while (atomicAdd(arrived, 0) < total_blocks)
            if (clock64() - start > timeout_cycles) return;     // gave up: not co-resident
        // Only blocks that saw everyone get here. The host checks
        // all_met == total_blocks to know whether the whole grid fit at once.
        atomicAdd(all_met, 1);
    }
}

// True if all `blocks` blocks were resident at the same time.
static bool all_resident(int blocks, int block_size, int cluster_size) {
    int *arrived, *all_met;
    CUDA_CHECK(cudaMallocManaged(&arrived, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&all_met, sizeof(int)));
    *arrived = 0; *all_met = 0;
    // How long a block waits for the others before giving up: ~0.8 s at
    // 2.4 GHz, far longer than any block needs to start if it can start at all.
    const long long timeout_cycles = 2000000000LL;
    cudaError_t launch_error;
    if (cluster_size <= 1) {
        rendezvous<<<blocks, block_size>>>(arrived, all_met, blocks, timeout_cycles);
        launch_error = cudaGetLastError();
    } else {
        cudaLaunchConfig_t config = {};
        config.gridDim = dim3(blocks, 1, 1); config.blockDim = dim3(block_size, 1, 1);
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = cluster_size;
        attribute[0].val.clusterDim.y = 1; attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute; config.numAttrs = 1;
        launch_error = cudaLaunchKernelEx(&config, rendezvous, arrived, all_met, blocks, timeout_cycles);
    }
    if (launch_error != cudaSuccess) {
        cudaGetLastError(); cudaFree(arrived); cudaFree(all_met);
        return false;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    bool ok = (*all_met == blocks);
    cudaFree(arrived); cudaFree(all_met);
    return ok;
}

// Largest grid (multiple of the cluster size) whose rendezvous succeeds.
static int max_resident_blocks(int block_size, int cluster_size, int max_blocks) {
    int step = (cluster_size <= 1) ? 1 : cluster_size, best = 0;
    for (int blocks = step; blocks <= max_blocks; blocks += step) {
        if (all_resident(blocks, block_size, cluster_size)) best = blocks;
        else break;
    }
    return best;
}

int main() {
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaFuncSetAttribute((void*)rendezvous,
                            cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

    printf("=== A. Occupancy API: unclustered vs clustered, by block size ===\n");
    printf("%10s | %-22s | %s\n", "block size", "unclustered",
           "clustered: device-wide blocks by cluster size");
    printf("%10s | %-22s | %6s %6s %6s %6s %6s\n", "",
           "blocks/SM  device-wide", "1", "2", "4", "6", "12");
    for (int block_size : {32, 64, 128, 256, 512, 1024}) {
        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, (void*)rendezvous,
                                                                 block_size, 0));
        printf("%10d | %9d %12d |", block_size, blocks_per_sm,
               blocks_per_sm * prop.multiProcessorCount);
        for (int cluster_size : {1, 2, 4, 6, 12}) {
            cudaLaunchConfig_t config = {};
            config.gridDim = dim3(cluster_size, 1, 1); config.blockDim = dim3(block_size, 1, 1);
            config.dynamicSmemBytes = 1024;
            cudaLaunchAttribute attribute[1];
            attribute[0].id = cudaLaunchAttributeClusterDimension;
            attribute[0].val.clusterDim.x = cluster_size;
            attribute[0].val.clusterDim.y = 1; attribute[0].val.clusterDim.z = 1;
            config.attrs = attribute; config.numAttrs = 1;
            int clusters = 0;
            if (cudaOccupancyMaxActiveClusters(&clusters, (void*)rendezvous, &config) != cudaSuccess)
                { printf("%6s", "err"); cudaGetLastError(); }
            else printf("%6d", clusters * cluster_size);
        }
        printf("\n");
    }

    printf("\n=== B. Rendezvous: measured largest co-resident grid ===\n");
    printf("%10s %12s %10s\n", "block size", "cluster size", "measured");
    for (int block_size : {64, 128}) {
        for (int cluster_size : {1, 2, 4, 12}) {
            int measured = max_resident_blocks(block_size, cluster_size, 1400);
            printf("%10d %12s %10d\n", block_size,
                   cluster_size <= 1 ? "none" : std::to_string(cluster_size).c_str(), measured);
        }
    }

    // Shared memory shrinks the budget: past ~50 KiB/block only one block
    // fits per SM, and cluster sizes that do not divide the 12 SMs of a GPC
    // strand the remainder (a cluster cannot span GPCs).
    printf("\n=== C. Occupancy API: clustered blocks vs shared memory (block size 128) ===\n");
    CUDA_CHECK(cudaFuncSetAttribute((void*)rendezvous,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));
    printf("%14s | %6s %6s %6s %6s %6s   device-wide blocks by cluster size\n",
           "smem per block", "1", "2", "4", "8", "12");
    for (int smem_bytes : {0, 1024, 32 * 1024, 50 * 1024, 101376}) {
        printf("%14d |", smem_bytes);
        for (int cluster_size : {1, 2, 4, 8, 12}) {
            cudaLaunchConfig_t config = {};
            config.gridDim = dim3(1152, 1, 1); config.blockDim = dim3(128, 1, 1);
            config.dynamicSmemBytes = smem_bytes;
            cudaLaunchAttribute attribute[1];
            attribute[0].id = cudaLaunchAttributeClusterDimension;
            attribute[0].val.clusterDim.x = cluster_size;
            attribute[0].val.clusterDim.y = 1; attribute[0].val.clusterDim.z = 1;
            config.attrs = attribute; config.numAttrs = 1;
            int clusters = 0;
            if (cudaOccupancyMaxActiveClusters(&clusters, (void*)rendezvous, &config) != cudaSuccess)
                { printf("%6s", "err"); cudaGetLastError(); }
            else printf("%6d", clusters * cluster_size);
        }
        printf("\n");
    }
    return 0;
}
