// carveout_occupancy.cu — a low shared-memory carveout costs occupancy, even
// for kernels that use no shared memory at all.
//
// The reason is cudaDevAttrReservedSharedMemoryPerBlock: every resident block
// reserves this much shared memory (1024 B here) for the system, regardless
// of what the kernel declares. If the carveout (the split of the SM's unified
// 128 KiB SRAM between shared memory and L1 cache) gives shared memory only a
// tiny partition, that partition holds only a few 1 KiB reservations, and the
// number of resident blocks per SM is capped by it.
//
// cudaFuncAttributePreferredSharedMemoryCarveout sets the preferred split as
// a percentage of the maximum shared memory (0 = favour L1, 100 = favour
// shared memory); the driver rounds it to a supported configuration.
//
// Build: nvcc -O3 -arch=sm_121 carveout_occupancy.cu -o carveout_occupancy
// Run:   ./carveout_occupancy

#include <cstdio>
#include <cuda_runtime.h>

// Uses no shared memory whatsoever — any occupancy change comes from the
// carveout and the per-block reservation alone.
__global__ void k(int* o) { if (!threadIdx.x) o[0] = 1; }

int main() {
    int res = 0; cudaDeviceGetAttribute(&res, cudaDevAttrReservedSharedMemoryPerBlock, 0);
    printf("reservedSharedMemoryPerBlock = %d B\n\n", res);
    printf("%10s %12s %12s %12s\n", "carveout", "bd=64", "bd=128", "bd=256");
    for (int pc : {0, 10, 25, 50, 75, 100}) {
        cudaFuncSetAttribute((void*)k, cudaFuncAttributePreferredSharedMemoryCarveout, pc);
        printf("%9d%%", pc);
        for (int bd : {64, 128, 256}) {
            int b = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, (void*)k, bd, 0);
            printf(" %11d", b);
        }
        printf("   blocks/SM\n");
    }
    printf("\nAt 0%% the shared partition is smallest, holds the fewest 1 KiB\n"
           "reservations, and blocks/SM drops although the kernel uses no SMEM.\n");
    return 0;
}
