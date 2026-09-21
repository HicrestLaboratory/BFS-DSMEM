// gpc.cu — measure the number of GPCs (GPU Processing Clusters) and the
// SM-to-GPC mapping of the installed GPU.
//
// NVIDIA exposes neither through any public CUDA attribute. Both can be
// measured, because thread block clusters carry a hardware guarantee: all
// blocks of a cluster are co-scheduled on SMs of a SINGLE GPC.
//
// That guarantee becomes a direct read-out once two conditions hold:
//
//   * the cluster size equals a GPC's SM count (12 here, the largest size
//     this GPU accepts — see properties.cu), so one cluster fills one GPC
//     exactly; and
//   * each block requests more than half the SM's shared memory, which
//     forces one block per SM, so a cluster of 12 blocks occupies 12
//     DISTINCT SMs.
//
// Then a single launch of 48 blocks is 4 clusters covering all 48 SMs exactly
// once, and the clusters ARE the GPCs. Group the blocks by cluster and the
// mapping falls out. No inference, no merging, no repetition.
//
// A note on what this file used to do: an earlier version accumulated
// "these two SMs were seen in the same cluster" facts across hundreds of
// trials and merged them with a union-find (disjoint-set) structure. That is
// the right approach when clusters are smaller than a GPC and each launch
// only samples part of the device. Here it was doing no work: 30 independent
// runs of a single launch produced the identical partition, with all 48 SMs
// observed every time. The union-find and the trial loop are gone.
//
// Build: nvcc -O3 -arch=sm_121 gpc.cu -o gpc
// Run:   ./gpc

#include <cooperative_groups.h>
#include <map>
#include <set>
#include <vector>

#include "common.cuh"

namespace cg = cooperative_groups;

// Every block reports the SM it landed on and the cluster it belongs to.
// The dynamic shared memory request (sized by the host) is what forces one
// block per SM.
__global__ void record_placement(unsigned* smids, unsigned* clusterIds) {
    extern __shared__ char scratch[];
    cg::cluster_group c = cg::this_cluster();
    if (threadIdx.x == 0) {
        scratch[0] = 1;  // touch the allocation so it is not optimized away
        smids[blockIdx.x] = smid();
        clusterIds[blockIdx.x] = (unsigned)cg::this_grid().cluster_rank();
    }
    c.sync();  // hold every block resident until the whole cluster has reported
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int numSM = prop.multiProcessorCount;
    const int clusterSize = 12;  // the largest this GPU accepts; see properties.cu

    // One block per SM: request more than half of the SM's shared memory so a
    // second block cannot fit alongside.
    int smemPerSM = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smemPerSM,
                                      cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    int smemPerBlock = smemPerSM / 2 + 1024;
    CUDA_CHECK(cudaFuncSetAttribute((void*)record_placement,
                                    cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CUDA_CHECK(cudaFuncSetAttribute((void*)record_placement,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smemPerBlock));

    const int numClusters = numSM / clusterSize;
    const int numBlocks = numClusters * clusterSize;

    printf("=== Placement ===\n");
    printf("Forcing 1 block/SM with %d B dynamic shared memory per block\n", smemPerBlock);
    printf("One launch: %d clusters x %d blocks = %d blocks over %d SMs\n\n",
           numClusters, clusterSize, numBlocks, numSM);
    if (numBlocks != numSM)
        printf("NOTE: %d SM(s) are not covered because %d does not divide %d.\n\n",
               numSM - numBlocks, clusterSize, numSM);

    unsigned *dSmid, *dCid;
    CUDA_CHECK(cudaMallocManaged(&dSmid, numBlocks * sizeof(unsigned)));
    CUDA_CHECK(cudaMallocManaged(&dCid, numBlocks * sizeof(unsigned)));
    for (int i = 0; i < numBlocks; i++) dSmid[i] = 0xFFFFFFFFu;  // detect blocks that never ran

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(numBlocks, 1, 1);
    cfg.blockDim = dim3(32, 1, 1);
    cfg.dynamicSmemBytes = smemPerBlock;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = clusterSize;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, record_placement, dSmid, dCid));
    CUDA_CHECK(cudaDeviceSynchronize());

    // A cluster is confined to one GPC, so the blocks of cluster c name the
    // SMs of one GPC. Group by cluster and sort for readability.
    std::map<unsigned, std::vector<int>> byCluster;
    std::set<int> seenSMs;
    for (int i = 0; i < numBlocks; i++) {
        if (dSmid[i] == 0xFFFFFFFFu) continue;
        byCluster[dCid[i]].push_back((int)dSmid[i]);
        seenSMs.insert((int)dSmid[i]);
    }
    for (auto& kv : byCluster) std::sort(kv.second.begin(), kv.second.end());

    printf("=== SM -> GPC mapping ===\n");
    printf("SMs observed: %zu of %d%s\n", seenSMs.size(), numSM,
           (int)seenSMs.size() == numBlocks ? " (every block on a distinct SM)"
                                            : "  <- two blocks shared an SM!");

    int gpc = 0;
    bool uniform = true;
    size_t firstSize = byCluster.empty() ? 0 : byCluster.begin()->second.size();
    std::map<int, int> gpcOf;  // smid -> printed GPC index
    for (auto& kv : byCluster) {
        printf("  GPC %d (%zu SMs): ", gpc, kv.second.size());
        for (int sm : kv.second) { printf("%d ", sm); gpcOf[sm] = gpc; }
        printf("\n");
        if (kv.second.size() != firstSize) uniform = false;
        gpc++;
    }

    printf("\n=== Result ===\n");
    printf("GPC count (measured by cluster co-scheduling): %d\n", gpc);
    if (uniform && gpc > 0) {
        printf("SMs per GPC: %zu (uniform)\n", firstSize);
        printf("Cross-check: %d GPCs x %zu SMs = %zu (device reports %d SMs) -> %s\n",
               gpc, firstSize, gpc * firstSize, numSM,
               (int)(gpc * firstSize) == numSM ? "consistent" : "MISMATCH");
    } else {
        printf("SMs per GPC: non-uniform (partially disabled GPCs are normal on\n");
        printf("harvested dies; the sizes above are the usable SMs per GPC)\n");
    }
    printf("Cross-check: max cluster size %d %s largest GPC size %zu\n", clusterSize,
           (size_t)clusterSize == firstSize ? "==" : "!=", firstSize);

    // SM ids are interleaved across GPCs rather than assigned in contiguous
    // ranges. Test the round-robin-by-TPC layout, in which consecutive SMs
    // form a TPC pair and consecutive TPCs cycle over GPCs:
    //     TPC = smid / 2,  GPC = TPC % numGPC
    if (gpc > 0 && uniform) {
        bool formulaHolds = true;
        for (int sm : seenSMs)
            if ((sm / 2) % gpc != gpcOf[sm]) formulaHolds = false;
        printf("\n=== Derived SM addressing ===\n");
        if (formulaHolds) {
            printf("Layout confirmed: TPC = smid/2, GPC = (smid/2) %% %d\n", gpc);
            printf("  -> %zu TPCs of 2 SMs, %zu TPCs per GPC, round-robin across GPCs\n",
                   seenSMs.size() / 2, seenSMs.size() / 2 / gpc);
            printf("  -> consecutive SM ids are TPC siblings, NOT GPC neighbours\n");
        } else {
            printf("SM ids do not follow TPC = smid/2, GPC = TPC %% %d;\n", gpc);
            printf("use the measured mapping above.\n");
        }
    }

    CUDA_CHECK(cudaFree(dSmid));
    CUDA_CHECK(cudaFree(dCid));
    return 0;
}