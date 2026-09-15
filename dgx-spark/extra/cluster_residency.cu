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
namespace cg = cooperative_groups;

#define CK(c) do { cudaError_t e_ = (c); if (e_ != cudaSuccess) { \
    printf("ERR %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

__global__ void rendezvous(int* arrived, int* allSaw, int target, long long budget) {
    if (threadIdx.x == 0) {
        atomicAdd(arrived, 1);
        long long t0 = clock64();
        while (atomicAdd(arrived, 0) < target)      // spin until everyone is in
            if (clock64() - t0 > budget) return;    // gave up: not co-resident
        atomicAdd(allSaw, 1);
    }
}

static bool test(int blocks, int bd, int clusterSize) {
    int *arr, *saw;
    CK(cudaMallocManaged(&arr, sizeof(int))); CK(cudaMallocManaged(&saw, sizeof(int)));
    *arr = 0; *saw = 0;
    const long long budget = 2000000000LL;          // ~0.8 s at 2.4 GHz
    cudaError_t le;
    if (clusterSize <= 1) {
        rendezvous<<<blocks, bd>>>(arr, saw, blocks, budget);
        le = cudaGetLastError();
    } else {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(blocks, 1, 1); cfg.blockDim = dim3(bd, 1, 1);
        cudaLaunchAttribute a[1];
        a[0].id = cudaLaunchAttributeClusterDimension;
        a[0].val.clusterDim.x = clusterSize; a[0].val.clusterDim.y = 1; a[0].val.clusterDim.z = 1;
        cfg.attrs = a; cfg.numAttrs = 1;
        le = cudaLaunchKernelEx(&cfg, rendezvous, arr, saw, blocks, budget);
    }
    if (le != cudaSuccess) { cudaGetLastError(); cudaFree(arr); cudaFree(saw); return false; }
    CK(cudaDeviceSynchronize());
    bool ok = (*saw == blocks);
    cudaFree(arr); cudaFree(saw);
    return ok;
}

// Largest grid (multiple of the cluster size) whose rendezvous succeeds.
static int maxResident(int bd, int cs, int hi) {
    int step = (cs <= 1) ? 1 : cs, best = 0;
    for (int b = step; b <= hi; b += step) { if (test(b, bd, cs)) best = b; else break; }
    return best;
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    CK(cudaFuncSetAttribute((void*)rendezvous,
                            cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

    printf("=== A. Occupancy API: unclustered vs clustered, by block size ===\n");
    printf("%9s | %-22s | %s\n", "blockDim", "unclustered",
           "clustered: device-wide blocks by cluster size");
    printf("%9s | %-22s | %6s %6s %6s %6s %6s\n", "",
           "blocks/SM  device-wide", "cs=1", "cs=2", "cs=4", "cs=6", "cs=12");
    for (int bd : {32, 64, 128, 256, 512, 1024}) {
        int b = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, (void*)rendezvous, bd, 0));
        printf("%9d | %9d %12d |", bd, b, b * p.multiProcessorCount);
        for (int cs : {1, 2, 4, 6, 12}) {
            cudaLaunchConfig_t cfg = {};
            cfg.gridDim = dim3(cs, 1, 1); cfg.blockDim = dim3(bd, 1, 1);
            cudaLaunchAttribute a[1];
            a[0].id = cudaLaunchAttributeClusterDimension;
            a[0].val.clusterDim.x = cs; a[0].val.clusterDim.y = 1; a[0].val.clusterDim.z = 1;
            cfg.attrs = a; cfg.numAttrs = 1;
            int n = 0;
            if (cudaOccupancyMaxActiveClusters(&n, (void*)rendezvous, &cfg) != cudaSuccess)
                { printf("%6s", "err"); cudaGetLastError(); }
            else printf("%6d", n * cs);
        }
        printf("\n");
    }

    printf("\n=== B. Rendezvous: measured largest co-resident grid ===\n");
    printf("%9s %10s %12s %14s\n", "blockDim", "cluster", "measured", "occupancy API");
    for (int bd : {64, 128}) {
        for (int cs : {1, 2, 4, 12}) {
            int meas = maxResident(bd, cs, 1400);
            int api;
            if (cs <= 1) {
                int b; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, (void*)rendezvous, bd, 0));
                api = b * p.multiProcessorCount;
            } else {
                cudaLaunchConfig_t cfg = {};
                cfg.gridDim = dim3(cs, 1, 1); cfg.blockDim = dim3(bd, 1, 1);
                cudaLaunchAttribute a[1];
                a[0].id = cudaLaunchAttributeClusterDimension;
                a[0].val.clusterDim.x = cs; a[0].val.clusterDim.y = 1; a[0].val.clusterDim.z = 1;
                cfg.attrs = a; cfg.numAttrs = 1;
                int n; CK(cudaOccupancyMaxActiveClusters(&n, (void*)rendezvous, &cfg));
                api = n * cs;
            }
            printf("%9d %10s %12d %14d\n", bd,
                   cs <= 1 ? "none" : std::to_string(cs).c_str(), meas, api);
        }
    }

    // Shared memory shrinks the budget: past ~50 KiB/block only one block
    // fits per SM, and cluster sizes that do not divide the 12 SMs of a GPC
    // strand the remainder (a cluster cannot span GPCs).
    printf("\n=== C. Occupancy API: clustered blocks vs shared memory (blockDim 128) ===\n");
    CK(cudaFuncSetAttribute((void*)rendezvous,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));
    printf("%12s | %6s %6s %6s %6s %6s   device-wide blocks\n",
           "smem/block", "cs=1", "cs=2", "cs=4", "cs=8", "cs=12");
    for (int smem : {0, 32 * 1024, 50 * 1024, 101376}) {
        printf("%12d |", smem);
        for (int cs : {1, 2, 4, 8, 12}) {
            cudaLaunchConfig_t cfg = {};
            cfg.gridDim = dim3(cs, 1, 1); cfg.blockDim = dim3(128, 1, 1);
            cfg.dynamicSmemBytes = smem;
            cudaLaunchAttribute a[1];
            a[0].id = cudaLaunchAttributeClusterDimension;
            a[0].val.clusterDim.x = cs; a[0].val.clusterDim.y = 1; a[0].val.clusterDim.z = 1;
            cfg.attrs = a; cfg.numAttrs = 1;
            int n = 0;
            if (cudaOccupancyMaxActiveClusters(&n, (void*)rendezvous, &cfg) != cudaSuccess)
                { printf("%6s", "err"); cudaGetLastError(); }
            else printf("%6d", n * cs);
        }
        printf("\n");
    }
    return 0;
}
