#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// No __cluster_dims__: the cluster size is chosen at launch time, so this is
// the kernel to use when probing which sizes the hardware actually accepts.
// It reports the cluster size it really ran with.
__global__ void dyn_cluster_kernel(int *out) {
    cg::cluster_group c = cg::this_cluster();
    if (threadIdx.x == 0 && c.block_rank() == 0) out[0] = c.num_blocks();
}

int main() {
    printf("=== CPU Properties ===\n");
    printf("20 cores: 10 x Cortex-X925 + 10 x Cortex-A725 (aarch64)\n");
    
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("\n=== NVIDIA DGX Spark Properties ===\n");
    printf("Name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Total global memory: %zu GB\n", prop.totalGlobalMem / (1024 * 1024 * 1024));
    printf("L2 cache size: %i MB\n", prop.l2CacheSize >> 20);
    printf("SMs count: %d\n", prop.multiProcessorCount);
    printf("Max grid size per SM: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    int maxThreadsPerBlock = 0, maxThreadsPerSM = 0, maxBlocksPerSM = 0, regsPerSM = 0;
    int sharedMemPerSM = 0, sharedMemPerBlock = 0, sharedMemPerBlockOptin = 0, reservedSharedMem = 0;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, dev);
    cudaDeviceGetAttribute(&maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    cudaDeviceGetAttribute(&maxBlocksPerSM, cudaDevAttrMaxBlocksPerMultiprocessor, dev);
    cudaDeviceGetAttribute(&regsPerSM, cudaDevAttrMaxRegistersPerMultiprocessor, dev);
    cudaDeviceGetAttribute(&sharedMemPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    // MaxSharedMemoryPerBlock is the limit a kernel gets WITHOUT asking (48 KiB);
    cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, dev);
    // MaxSharedMemoryPerBlockOptin is what cudaFuncAttributeMaxDynamicSharedMemorySize can raise it to.
    cudaDeviceGetAttribute(&sharedMemPerBlockOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaDeviceGetAttribute(&reservedSharedMem, cudaDevAttrReservedSharedMemoryPerBlock, dev);
    
    printf("\n=== NVIDIA DGX Spark Per-SM Properties ===\n");
    printf("Max threads / block: %d\n", maxThreadsPerBlock);
    printf("Max threads / SM: %d (%d warps)\n", maxThreadsPerSM, maxThreadsPerSM / 32);
    printf("Max blocks / SM: %d\n", maxBlocksPerSM);
    printf("Registers / SM: %d\n", regsPerSM);
    // No attribute exposes the physical SRAM, but documentations says that the
    // unified L1 + shared memory per SM is 128 KiB, split by a configurable carveout.
    printf("Unified L1 + shared memory / SM: 128 KiB\n");
    printf("Max shared memory / SM: %d KiB\n", sharedMemPerSM >> 10);
    printf("Default shared memory / block: %d KiB\n", sharedMemPerBlock >> 10);
    printf("Max shared memory / block (opt-in): %d KiB\n", sharedMemPerBlockOptin >> 10);
    printf("Reserved shared memory / block: %d B\n", reservedSharedMem);
    
    int clusterLaunch = 0;
    cudaDeviceGetAttribute(&clusterLaunch, cudaDevAttrClusterLaunch, dev);
    printf("\n=== NVIDIA DGX Spark Thread Block Cluster Properties ===\n");
    printf("Thread Block Cluster Support: %d\n", clusterLaunch);

    int *d_out;
    cudaMalloc(&d_out, sizeof(int));

    // Probe which cluster sizes the hardware actually accepts. This must use
    // a kernel WITHOUT __cluster_dims__, otherwise every size except the
    // compile-time one fails and the result says nothing about the hardware.
    // Run once before opting in (portable limit) and once after (real limit).
    printf("\n--- Max Cluster Size ---\n");
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 1) {
            // Opt in to non-portable cluster sizes
            cudaError_t at = cudaFuncSetAttribute(
                (void *)dyn_cluster_kernel,
                cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
            printf("-- after opting in to non-portable sizes (%s) --\n",
                   cudaGetErrorString(at));
        } else {
            printf("-- portable, no opt-in --\n");
        }

        int cluster_size = 0;
        cudaLaunchConfig_t ocfg = {0};
        ocfg.gridDim = dim3(1, 1, 1);
        ocfg.blockDim = dim3(128, 1, 1);
        cudaOccupancyMaxPotentialClusterSize(&cluster_size, (void *)dyn_cluster_kernel, &ocfg);
        printf("  cudaOccupancyMaxPotentialClusterSize = %d\n", cluster_size);

        int maxOk = 0;
        for (int csize = 1; csize <= 16; csize++) {
            cudaLaunchConfig_t cfg2 = {0};
            cfg2.gridDim = dim3(csize, 1, 1);
            cfg2.blockDim = dim3(128, 1, 1);
            cudaLaunchAttribute a2[1];
            a2[0].id = cudaLaunchAttributeClusterDimension;
            a2[0].val.clusterDim.x = csize;
            a2[0].val.clusterDim.y = 1;
            a2[0].val.clusterDim.z = 1;
            cfg2.attrs = a2;
            cfg2.numAttrs = 1;
            cudaMemset(d_out, 0, sizeof(int));
            cudaError_t le = cudaLaunchKernelEx(&cfg2, dyn_cluster_kernel, d_out);
            cudaError_t se = cudaDeviceSynchronize();
            if (le == cudaSuccess && se == cudaSuccess) {
                int observed = 0;
                cudaMemcpy(&observed, d_out, sizeof(int), cudaMemcpyDeviceToHost);
                printf("  clusterDim=%2d -> ok (cluster.num_blocks()=%d)\n", csize, observed);
                maxOk = csize;
            } else {
                printf("  clusterDim=%2d -> %s\n", csize,
                       cudaGetErrorString(le != cudaSuccess ? le : se));
            }
        }
        printf("  => largest accepted cluster size: %d\n\n", maxOk);
    }

    cudaFree(d_out);
    return 0;
}
