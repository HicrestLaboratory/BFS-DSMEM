// dsmem_capacity.cu — how much DSMEM can one cluster actually address?
//
// The naive answer would be clusterSize x 100 KiB (the per-SM shared memory),
// but a block cannot opt into the full per-SM amount: 1 KiB per SM is
// reserved, so the per-block ceiling is 99 KiB. This program
//   1. prints the three shared-memory limits from the device attributes,
//   2. for each cluster size, binary-searches the largest dynamic
//      shared-memory request that still launches, and
//   3. VERIFIES the aggregate is genuinely addressable, not merely
//      allocatable: every rank fills its whole extent with a rank-dependent
//      pattern, and rank 0 reads every peer's entire extent back through
//      map_shared_rank and checks every word.
// It also reports which SMs the 12-block cluster landed on (via the %smid
// special register — a hardware register holding the id of the SM the code is
// running on), which independently confirms the GPC topology.
// Finally it cross-checks cudaOccupancyMaxPotentialClusterSize at several
// shared-memory sizes: maximising DSMEM costs no cluster size.
//
// Build: nvcc -O3 -arch=sm_121 dsmem_capacity.cu -o dsmem_capacity
// Run:   ./dsmem_capacity

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CK(c) do { cudaError_t e_ = (c); if (e_ != cudaSuccess) { \
    printf("ERR %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

// Each rank fills its whole dynamic SMEM extent with a rank-dependent pattern,
// then rank 0 reads every remote extent back and validates it.
__global__ void fill_and_check(unsigned nWords, int* ok, unsigned* smids) {
    extern __shared__ unsigned buf[];
    cg::cluster_group c = cg::this_cluster();
    unsigned r = c.block_rank(), n = c.num_blocks();
    for (unsigned i = threadIdx.x; i < nWords; i += blockDim.x)
        buf[i] = r * 1000003u + i;
    if (threadIdx.x == 0) {
        unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
        smids[r] = s;
    }
    __syncthreads();
    c.sync();
    if (r == 0) {
        int bad = 0;
        for (unsigned peer = 0; peer < n; peer++) {
            unsigned* p = c.map_shared_rank(buf, peer);
            for (unsigned i = threadIdx.x; i < nWords; i += blockDim.x)
                if (p[i] != peer * 1000003u + i) bad = 1;
        }
        if (bad) atomicOr(ok, 1);
    }
    c.sync();
}

static bool tryLaunch(int cs, int smemBytes, int* d_ok, unsigned* d_smids, bool verify) {
    CK(cudaFuncSetAttribute((void*)fill_and_check,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
    CK(cudaMemset(d_ok, 0, sizeof(int)));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(cs, 1, 1);
    cfg.blockDim = dim3(256, 1, 1);
    cfg.dynamicSmemBytes = smemBytes;
    cudaLaunchAttribute a[1];
    a[0].id = cudaLaunchAttributeClusterDimension;
    a[0].val.clusterDim.x = cs; a[0].val.clusterDim.y = 1; a[0].val.clusterDim.z = 1;
    cfg.attrs = a; cfg.numAttrs = 1;
    cudaError_t le = cudaLaunchKernelEx(&cfg, fill_and_check,
                                        (unsigned)(smemBytes / sizeof(unsigned)), d_ok, d_smids);
    cudaError_t se = cudaDeviceSynchronize();
    if (le != cudaSuccess || se != cudaSuccess) { cudaGetLastError(); return false; }
    if (verify) { int h = 0; CK(cudaMemcpy(&h, d_ok, sizeof(int), cudaMemcpyDeviceToHost)); return h == 0; }
    return true;
}

int main() {
    int perSM = 0, perBlockOptin = 0, perBlockDefault = 0;
    CK(cudaDeviceGetAttribute(&perSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    CK(cudaDeviceGetAttribute(&perBlockOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    CK(cudaDeviceGetAttribute(&perBlockDefault, cudaDevAttrMaxSharedMemoryPerBlock, 0));
    printf("=== Shared memory limits ===\n");
    printf("  per SM              : %6d B (%.1f KiB)\n", perSM, perSM / 1024.0);
    printf("  per block, default  : %6d B (%.1f KiB)\n", perBlockDefault, perBlockDefault / 1024.0);
    printf("  per block, opt-in   : %6d B (%.1f KiB)\n", perBlockOptin, perBlockOptin / 1024.0);

    CK(cudaFuncSetAttribute((void*)fill_and_check,
                            cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    int* d_ok; unsigned* d_smids;
    CK(cudaMalloc(&d_ok, sizeof(int)));
    CK(cudaMallocManaged(&d_smids, 16 * sizeof(unsigned)));

    printf("\n=== Max dynamic SMEM/block that still launches, per cluster size ===\n");
    printf("(binary search; VERIFIED = rank 0 read back every peer's full extent)\n");
    for (int cs : {1, 2, 4, 8, 12}) {
        int lo = 0, hi = perBlockOptin, best = -1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2 / 256 * 256;        // 256 B granularity
            if (mid <= 0) break;
            if (tryLaunch(cs, mid, d_ok, d_smids, false)) { best = mid; lo = mid + 256; }
            else hi = mid - 256;
        }
        if (best < 0) { printf("  cluster %2d: no size launched\n", cs); continue; }
        bool v = tryLaunch(cs, best, d_ok, d_smids, true);
        printf("  cluster %2d: %6d B/block (%.1f KiB) -> aggregate %8.1f KiB  [%s]",
               cs, best, best / 1024.0, (double)best * cs / 1024.0,
               v ? "VERIFIED" : "MISMATCH");
        if (v && cs > 1) { printf("  SMs:"); for (int i = 0; i < cs; i++) printf(" %u", d_smids[i]); }
        printf("\n");
    }

    printf("\n=== Does maximising SMEM cost cluster size? ===\n");
    for (int smem : {0, 32 * 1024, 50 * 1024, 64 * 1024, perBlockOptin}) {
        CK(cudaFuncSetAttribute((void*)fill_and_check,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(1, 1, 1);
        cfg.blockDim = dim3(256, 1, 1);
        cfg.dynamicSmemBytes = smem;
        int occ = 0;
        CK(cudaOccupancyMaxPotentialClusterSize(&occ, (void*)fill_and_check, &cfg));
        int maxOk = 0;
        for (int cs = 1; cs <= 12; cs++)
            if (tryLaunch(cs, smem, d_ok, d_smids, false)) maxOk = cs;
        printf("  smem/block=%6d B -> occupancy says %2d, actually launches up to %2d"
               " (aggregate %.1f KiB)\n", smem, occ, maxOk, (double)smem * maxOk / 1024.0);
    }
    CK(cudaFree(d_ok)); CK(cudaFree(d_smids));
    return 0;
}
