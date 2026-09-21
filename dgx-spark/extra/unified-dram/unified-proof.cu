// unified.cu — is the DGX Spark's 128 GB the CPU's memory, the GPU's memory,
// or one shared pool?
//
// Four independent tests, each sufficient on its own:
//   1. The device property `integrated` (CUDA's own classification: an
//      integrated GPU shares system memory; a discrete one has its own VRAM),
//      plus the properties that describe how far the sharing goes.
//   2. Size comparison: the GPU's totalGlobalMem vs the host's MemTotal from
//      /proc/meminfo. Two separate pools would have two different sizes.
//   3. A cudaMalloc of 8 GiB, watching /proc/meminfo MemAvailable: if GPU
//      allocations consume *system* RAM, there is only one pool.
//   4. A kernel that dereferences a plain host malloc() pointer — no
//      cudaMemcpy, no cudaHostRegister — and increments it in place. If the
//      CPU then reads the new value, CPU and GPU are hardware-coherent.
//
// Build: nvcc -O3 -arch=sm_121 unified.cu -o unified
// Run:   ./unified

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "../../common.cuh"  // CUDA_CHECK

// Reads and writes a plain host malloc() pointer directly.
__global__ void touch_host_ptr(int* p, int* out) {
    if (!threadIdx.x) { p[0] += 41; *out = p[0]; }
}

// One value out of /proc/meminfo, in kB.
static long meminfo(const char* key) {
    FILE* f = fopen("/proc/meminfo", "r");
    char k[64]; long v;
    while (fscanf(f, "%63s %ld kB\n", k, &v) == 2)
        if (!strncmp(k, key, strlen(key))) { fclose(f); return v; }
    fclose(f); return -1;
}

int main() {
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("=== 1. Is the GPU integrated (shares host memory) or discrete? ===\n");
    printf("  integrated                   : %d   %s\n", p.integrated,
           p.integrated ? "<- shares system memory with the CPU" : "<- has its own VRAM");
    printf("  unifiedAddressing            : %d\n", p.unifiedAddressing);
    printf("  managedMemory                : %d\n", p.managedMemory);
    printf("  concurrentManagedAccess      : %d\n", p.concurrentManagedAccess);
    printf("  pageableMemoryAccess         : %d   %s\n", p.pageableMemoryAccess,
           p.pageableMemoryAccess ? "<- GPU can read plain malloc() pointers" : "");
    printf("  hostNativeAtomicSupported    : %d\n", p.hostNativeAtomicSupported);
    printf("  canMapHostMemory             : %d\n", p.canMapHostMemory);

    printf("\n=== 2. One pool or two? ===\n");
    printf("  GPU totalGlobalMem   : %.1f GiB\n", p.totalGlobalMem / 1073741824.0);
    printf("  Host MemTotal        : %.1f GiB\n", meminfo("MemTotal") / 1048576.0);
    printf("  -> %s\n", (long)(p.totalGlobalMem / 1048576) == meminfo("MemTotal") / 1024
           ? "identical: a single physical pool" : "different sizes");

    printf("\n=== 3. Does a cudaMalloc consume *system* RAM? ===\n");
    long before = meminfo("MemAvailable");
    size_t bytes = 8ull << 30;
    void* d; CUDA_CHECK(cudaMalloc(&d, bytes));
    CUDA_CHECK(cudaMemset(d, 1, bytes));                    // fault the pages in
    long after = meminfo("MemAvailable");
    printf("  MemAvailable before cudaMalloc(8 GiB): %.1f GiB\n", before / 1048576.0);
    printf("  MemAvailable after                   : %.1f GiB\n", after / 1048576.0);
    printf("  -> system RAM consumed by the GPU allocation: %.1f GiB\n",
           (before - after) / 1048576.0);
    CUDA_CHECK(cudaFree(d));

    printf("\n=== 4. Can the GPU dereference a plain malloc() pointer? ===\n");
    int* hp = (int*)malloc(sizeof(int));            // ordinary host heap, never registered
    *hp = 1;
    int* out; CUDA_CHECK(cudaMallocManaged(&out, sizeof(int)));
    touch_host_ptr<<<1, 32>>>(hp, out);
    cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess)
        printf("  kernel read/wrote host heap directly: *hp = %d (expected 42)"
               " -> COHERENT SHARED MEMORY\n", *hp);
    else
        printf("  failed: %s\n", cudaGetErrorString(e));
    free(hp);
    CUDA_CHECK(cudaFree(out));
    return 0;
}
