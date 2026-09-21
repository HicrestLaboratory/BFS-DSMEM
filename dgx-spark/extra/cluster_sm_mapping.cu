// cluster_sm_mapping.cu — are the blocks of one cluster guaranteed to land on
// DISTINCT SMs, or can two ranks of the same cluster share an SM?
//
// The CUDA programming guide promises only that a cluster's blocks are
// "scheduled simultaneously on the SMs of a single GPC" — it never says one
// block per SM. This program checks directly: every block records the SM it
// runs on (%smid) together with its cluster id and rank; the host then asks,
// for every cluster, whether any two of its ranks share an SM.
//
// The conditions are chosen to give sharing the best possible chance:
//   - tiny blocks (64 threads) and no shared memory, so up to 24 blocks fit
//     on one SM and the scheduler has every opportunity to double up;
//   - grids both at and well beyond the residency limit, so placement is
//     observed in the first wave and in later ones;
//   - several cluster sizes, including sizes far below the 12 SMs of a GPC.
//
// Build: nvcc -O3 -arch=sm_121 cluster_sm_mapping.cu -o cluster_sm_mapping
// Run:   ./cluster_sm_mapping

#include <cstdio>
#include <cstdlib>
#include <set>
#include <map>
#include <vector>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "../common.cuh"  // CUDA_CHECK

namespace cg = cooperative_groups;

// Each block reports the SM it is executing on, indexed by its global block id.
__global__ void record(unsigned* smids, unsigned* ranks, int spin) {
    cg::cluster_group c = cg::this_cluster();
    if (threadIdx.x == 0) {
        unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
        smids[blockIdx.x] = s;
        ranks[blockIdx.x] = c.block_rank();
    }
    // Hold the block alive briefly so co-residency is real, not sequential reuse.
    if (spin) { long long t0 = clock64(); while (clock64() - t0 < spin) ; }
    c.sync();
}

struct Verdict { int clusters; int sharing; int minDistinct; int maxDistinct; };

static Verdict probe(int clusterSize, int grid, int blockDim_, int smem, int spin) {
    unsigned *smids, *ranks;
    CUDA_CHECK(cudaMallocManaged(&smids, grid * sizeof(unsigned)));
    CUDA_CHECK(cudaMallocManaged(&ranks, grid * sizeof(unsigned)));
    for (int i = 0; i < grid; i++) { smids[i] = 0xFFFFFFFFu; ranks[i] = 0xFFFFFFFFu; }

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid, 1, 1);
    cfg.blockDim = dim3(blockDim_, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cudaLaunchAttribute a[1];
    a[0].id = cudaLaunchAttributeClusterDimension;
    a[0].val.clusterDim.x = clusterSize; a[0].val.clusterDim.y = 1; a[0].val.clusterDim.z = 1;
    cfg.attrs = a; cfg.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, record, smids, ranks, spin));
    CUDA_CHECK(cudaDeviceSynchronize());

    // group blocks by cluster id and look for repeated SMs inside a cluster
    Verdict v{0, 0, 1 << 30, 0};
    std::map<int, std::vector<unsigned>> byCluster;
    for (int i = 0; i < grid; i++)
        if (smids[i] != 0xFFFFFFFFu) byCluster[i / clusterSize].push_back(smids[i]);
    for (auto& kv : byCluster) {
        std::set<unsigned> distinct(kv.second.begin(), kv.second.end());
        v.clusters++;
        if ((int)distinct.size() < (int)kv.second.size()) v.sharing++;
        v.minDistinct = std::min(v.minDistinct, (int)distinct.size());
        v.maxDistinct = std::max(v.maxDistinct, (int)distinct.size());
    }
    CUDA_CHECK(cudaFree(smids)); CUDA_CHECK(cudaFree(ranks));
    return v;
}

int main() {
    CUDA_CHECK(cudaFuncSetAttribute((void*)record, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("# %s, %d SMs, 12 SMs per GPC\n", p.name, p.multiProcessorCount);
    printf("# A cluster's ranks share an SM only if 'distinct SMs' < cluster size.\n\n");

    printf("=== Best chance of sharing: 64-thread blocks, no shared memory ===\n");
    printf("(up to 24 such blocks fit on one SM, so the scheduler is free to double up)\n\n");
    printf("%12s %8s %10s %12s %16s %s\n",
           "cluster size", "grid", "clusters", "distinct SMs", "clusters sharing", "verdict");
    for (int cs : {2, 3, 4, 6, 8, 12}) {
        for (int mult : {1, 4}) {                 // at, and 4x beyond, the residency limit
            int grid = 96 * mult;
            if (grid % cs) grid = (grid / cs) * cs;
            Verdict v = probe(cs, grid, 64, 0, 2000000);
            printf("%12d %8d %10d %12s %16d %s\n", cs, grid, v.clusters,
                   (v.minDistinct == v.maxDistinct
                        ? (std::to_string(v.minDistinct)).c_str()
                        : (std::to_string(v.minDistinct) + "-" + std::to_string(v.maxDistinct)).c_str()),
                   v.sharing, v.sharing ? "SHARES AN SM" : "all distinct");
        }
    }

    printf("\n=== Same question with larger blocks / shared memory ===\n\n");
    printf("%12s %8s %8s %10s %12s %16s\n",
           "cluster size", "blockDim", "smem", "clusters", "distinct SMs", "clusters sharing");
    struct { int cs, bd, smem; } cases[] = {
        {4, 256, 0}, {4, 256, 32768}, {12, 128, 0}, {12, 256, 51200}, {2, 1024, 0}};
    for (auto& c : cases) {
        int grid = 96; if (grid % c.cs) grid = (grid / c.cs) * c.cs;
        if (c.smem) CUDA_CHECK(cudaFuncSetAttribute((void*)record,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, c.smem));
        Verdict v = probe(c.cs, grid, c.bd, c.smem, 2000000);
        printf("%12d %8d %8d %10d %12s %16d\n", c.cs, c.bd, c.smem, v.clusters,
               (v.minDistinct == v.maxDistinct
                    ? (std::to_string(v.minDistinct)).c_str()
                    : (std::to_string(v.minDistinct) + "-" + std::to_string(v.maxDistinct)).c_str()),
               v.sharing);
    }
    return 0;
}
