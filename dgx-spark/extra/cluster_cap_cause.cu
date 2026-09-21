// cluster_cap_cause.cu — WHY are only 96 blocks resident under a clustered
// launch? Elimination experiments: each section tests one candidate cause.
// (Spoiler: all of them are ruled out. The cap appears only when a launch
// carries a cluster dimension, pointing at a dedicated, undocumented hardware
// resource for tracking resident clusters.)
//
//   1. A limit on the NUMBER of clusters?   -> no: the cluster count varies
//      (96/48/32/24/16/8) but the block total is always 96.
//   2. A hidden shared-memory reservation for DSMEM? -> no: the cap is 96 at
//      every L1/SMEM carveout setting from 0 to 100%.
//   3. The cluster CODE itself? -> no: kernels using this_cluster(),
//      map_shared_rank and cluster.sync(), launched WITHOUT a cluster
//      dimension, reach the full 24 blocks/SM like a plain kernel.
//   4. The NonPortableClusterSizeAllowed attribute? -> no: 24 blocks/SM
//      before and after setting it.
//
// Sections 3 and 4 deliberately use fresh kernels: per-kernel attributes
// persist, and reusing a kernel whose carveout was changed in section 2 would
// contaminate the result (that exact mistake produced a wrong number once).
//
// Build: nvcc -O3 -arch=sm_121 cluster_cap_cause.cu -o cluster_cap_cause
// Run:   ./cluster_cap_cause

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "../common.cuh"  // CUDA_CHECK

namespace cg = cooperative_groups;

// The clustered kernel under test (sections 1-2).
__global__ void clus(int* o) { cg::cluster_group c = cg::this_cluster();
    if (!threadIdx.x && c.block_rank() == 0) o[0] = c.num_blocks(); c.sync(); }

// Fresh kernels for sections 3-4 (no attribute ever set on them).
__global__ void k_plain(int* o)       { if (!threadIdx.x) o[0] = 1; }
__global__ void k_syncthreads(int* o) { __syncthreads(); if (!threadIdx.x) o[0] = 1; }
__global__ void k_rank(int* o)        { cg::cluster_group c = cg::this_cluster();
                                        if (!threadIdx.x) o[0] = c.block_rank(); }
__global__ void k_map(int* o)         { __shared__ int s[4]; cg::cluster_group c = cg::this_cluster();
                                        s[0] = 1; int* p = c.map_shared_rank(s, 0);
                                        if (!threadIdx.x) o[0] = p[0]; }
__global__ void k_sync(int* o)        { cg::cluster_group c = cg::this_cluster(); c.sync();
                                        if (!threadIdx.x) o[0] = c.block_rank(); }
__global__ void k_attr(int* o)        { cg::cluster_group c = cg::this_cluster(); c.sync();
                                        if (!threadIdx.x) o[0] = c.block_rank(); }

static int clusterBlocks(int bd, int cs) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(cs, 1, 1); cfg.blockDim = dim3(bd, 1, 1);
    cudaLaunchAttribute a[1];
    a[0].id = cudaLaunchAttributeClusterDimension;
    a[0].val.clusterDim.x = cs; a[0].val.clusterDim.y = 1; a[0].val.clusterDim.z = 1;
    cfg.attrs = a; cfg.numAttrs = 1;
    int n = 0; CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, (void*)clus, &cfg));
    return n * cs;
}

int main() {
    CUDA_CHECK(cudaFuncSetAttribute((void*)clus, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    cudaFuncAttributes fa; CUDA_CHECK(cudaFuncGetAttributes(&fa, (void*)clus));
    printf("cluster kernel: %d regs, %zu B static shared memory\n",
           fa.numRegs, fa.sharedSizeBytes);

    printf("\n=== 1. A limit on the NUMBER of clusters? ===\n");
    printf("%12s %10s %14s\n", "cluster size", "clusters", "blocks");
    for (int cs : {1, 2, 3, 4, 6, 12}) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(cs, 1, 1); cfg.blockDim = dim3(64, 1, 1);
        cudaLaunchAttribute a[1];
        a[0].id = cudaLaunchAttributeClusterDimension;
        a[0].val.clusterDim.x = cs; a[0].val.clusterDim.y = 1; a[0].val.clusterDim.z = 1;
        cfg.attrs = a; cfg.numAttrs = 1;
        int n = 0; CUDA_CHECK(cudaOccupancyMaxActiveClusters(&n, (void*)clus, &cfg));
        printf("%12d %10d %14d\n", cs, n, n * cs);
    }

    printf("\n=== 2. A hidden shared-memory reservation? (carveout sweep) ===\n");
    printf("%28s %16s\n", "carveout (% toward shared)", "clustered blocks");
    for (int pc : {0, 10, 25, 50, 100}) {
        CUDA_CHECK(cudaFuncSetAttribute((void*)clus, cudaFuncAttributePreferredSharedMemoryCarveout, pc));
        printf("%28d %16d\n", pc, clusterBlocks(64, 4));
    }

    printf("\n=== 3. The cluster code itself? (unclustered launch, blockDim 64) ===\n");
    struct { const char* name; void* fn; } ks[] = {
        {"plain", (void*)k_plain}, {"__syncthreads only", (void*)k_syncthreads},
        {"this_cluster().block_rank()", (void*)k_rank}, {"+ map_shared_rank", (void*)k_map},
        {"+ cluster.sync()", (void*)k_sync}};
    printf("%-30s %10s %12s\n", "kernel", "blocks/SM", "device-wide");
    for (auto& en : ks) {
        int b = 0; CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, en.fn, 64, 0));
        printf("%-30s %10d %12d\n", en.name, b, b * 48);
    }

    printf("\n=== 4. The NonPortableClusterSizeAllowed attribute? ===\n");
    int b = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, (void*)k_attr, 64, 0));
    printf("before opt-in: %2d blocks/SM (%d device-wide)\n", b, b * 48);
    CUDA_CHECK(cudaFuncSetAttribute((void*)k_attr, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, (void*)k_attr, 64, 0));
    printf("after  opt-in: %2d blocks/SM (%d device-wide)\n", b, b * 48);
    return 0;
}
