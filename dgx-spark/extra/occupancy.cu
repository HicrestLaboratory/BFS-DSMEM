// occupancy.cu — how many thread blocks can a GPU hold, and what limits them?
//
// Two different questions hide behind "max blocks":
//   1. How many blocks may be LAUNCHED in one grid? A launch-configuration
//      limit (maxGridSize), essentially unbounded — excess blocks just wait
//      and run in later waves.
//   2. How many blocks are RESIDENT simultaneously? A resource question,
//      bounded per SM by four caps: a hard block count, a warp/thread count,
//      the register file, and shared memory. Whichever binds first wins.
// This program prints the caps, then uses the occupancy API
// (cudaOccupancyMaxActiveBlocksPerMultiprocessor: given a kernel, block size
// and dynamic shared memory, how many blocks fit on one SM) to show which cap
// binds as the block size, the shared memory, and the register pressure vary.
//
// Build: nvcc -O3 -arch=sm_121 occupancy.cu -o occupancy
// Run:   ./occupancy

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(c) do { cudaError_t e_ = (c); if (e_ != cudaSuccess) { \
    printf("ERR %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

// Tiny kernel: minimal registers, no shared memory. The guard on
// threadIdx.x == 1023 is never true at the block sizes used for the sweep but
// keeps the compiler from optimising the body away.
__global__ void tiny(int* o) { if (threadIdx.x == 1023) o[0] = 1; }

// Register-hungry kernel: a live array of 40 floats forces a high per-thread
// register count, so the 64 K-register file becomes the binding limit.
__global__ void fat(float* o, int n) {
    float a[40];
    #pragma unroll
    for (int i = 0; i < 40; i++) a[i] = o[i] * (i + n);
    for (int k = 0; k < n; k++) {
        #pragma unroll
        for (int i = 0; i < 39; i++) a[i] = fmaf(a[i], a[i + 1], a[39 - i]);
    }
    float s = 0; for (int i = 0; i < 40; i++) s += a[i];
    if (threadIdx.x == 1023) o[0] = s;
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("=== Hardware caps (per SM) ===\n");
    printf("  maxThreadsPerMultiProcessor : %d  (= %d warps)\n",
           p.maxThreadsPerMultiProcessor, p.maxThreadsPerMultiProcessor / 32);
    int mb = 0; CK(cudaDeviceGetAttribute(&mb, cudaDevAttrMaxBlocksPerMultiprocessor, 0));
    printf("  maxBlocksPerMultiprocessor  : %d\n", mb);
    printf("  regsPerMultiprocessor       : %d\n", p.regsPerMultiprocessor);
    printf("  sharedMemPerMultiprocessor  : %zu B\n", p.sharedMemPerMultiprocessor);
    printf("  SMs                         : %d\n", p.multiProcessorCount);

    printf("\n=== Grid limit (how many blocks may be *launched*) ===\n");
    printf("  maxGridSize = %d x %d x %d  -> %.2e blocks in x alone\n",
           p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2], (double)p.maxGridSize[0]);

    printf("\n=== Resident blocks: tiny kernel, no shared memory ===\n");
    printf("%10s %8s %10s %12s %14s %s\n",
           "blockDim", "warps", "blocks/SM", "warps/SM", "occupancy", "device-wide blocks");
    for (int bd : {32, 64, 128, 256, 512, 1024}) {
        int n = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, (void*)tiny, bd, 0));
        int warps = bd / 32;
        printf("%10d %8d %10d %12d %13.0f%% %d\n", bd, warps, n, n * warps,
               100.0 * n * warps * 32 / p.maxThreadsPerMultiProcessor,
               n * p.multiProcessorCount);
    }

    printf("\n=== Same kernel, varying dynamic shared memory (blockDim 128) ===\n");
    printf("%14s %10s %s\n", "smem/block", "blocks/SM", "device-wide blocks");
    for (int sm : {0, 8 * 1024, 16 * 1024, 32 * 1024, 50 * 1024}) {
        int n = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, (void*)tiny, 128, sm));
        printf("%14d %10d %d\n", sm, n, n * p.multiProcessorCount);
    }
    printf("(50 KiB gives 0: the default per-block shared-memory limit is 48 KiB"
           " unless the kernel opts in)\n");

    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, (void*)fat));
    printf("\n=== Register pressure (fat kernel uses %d regs/thread) ===\n", fa.numRegs);
    printf("%10s %10s %s\n", "blockDim", "blocks/SM", "device-wide blocks");
    for (int bd : {32, 128, 256}) {
        int n = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, (void*)fat, bd, 0));
        printf("%10d %10d %d\n", bd, n, n * p.multiProcessorCount);
    }
    return 0;
}
