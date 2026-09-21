// gpc.cu — empirically determine the number of GPCs (GPU Processing Clusters)
// and the SM-to-GPC mapping of the installed GPU.
//
// NVIDIA does not expose GPC count or SM-per-GPC topology through any public
// CUDA runtime attribute. It can nevertheless be *measured*, because the
// thread-block-cluster feature carries a hardware guarantee: all blocks of a
// cluster are co-scheduled on SMs of a single GPC. So:
//
//   1. Launch a grid of clusters; every block reports the SM it landed on
//      (the %smid special register).
//   2. Two SMs observed in the same cluster must belong to the same GPC.
//      Union-find over those co-occurrences builds the GPC partition.
//   3. Repeat over many trials so the scheduler exercises different
//      placements and every SM is eventually observed.
//
// The number of connected components is the GPC count. A cross-check comes
// for free: the largest launchable cluster cannot exceed one GPC's SM count.
//
// Build: nvcc -O3 -arch=sm_121 gpc.cu -o gpc
// Run:   ./gpc [trials=200]

#include <cooperative_groups.h>
#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "common.cuh"

namespace cg = cooperative_groups;

// Every block reports its SM id and its cluster id. The dynamic shared memory
// request (sized by the host) is what forces one block per SM, so that a
// cluster of N blocks necessarily occupies N distinct SMs.
__global__ void record_placement(unsigned* smids, unsigned* clusterIds) {
    extern __shared__ char scratch[];
    cg::cluster_group c = cg::this_cluster();
    if (threadIdx.x == 0) {
        scratch[0] = 1;  // touch the allocation so it is not optimized away
        smids[blockIdx.x] = smid();
        clusterIds[blockIdx.x] = blockIdx.x / c.num_blocks();
    }
    c.sync();  // hold every block resident until the whole cluster has reported
}

// ---------------- union-find over SM ids ----------------
struct DSU {
    std::vector<int> p;
    explicit DSU(int n) : p(n) {
        for (int i = 0; i < n; i++) p[i] = i;
    }
    int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b) p[a] = b;
    }
};

int main(int argc, char** argv) {
    int trials = argc > 1 ? atoi(argv[1]) : 300;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int numSM = prop.multiProcessorCount;
    const int clusterSize = 12; // see properties.cu

    // One block per SM: request more than half of the SM's shared memory so a
    // second block cannot fit alongside. Without this, several blocks of a
    // cluster could share an SM and the co-occurrence data would be sparser.
    int smemPerSM = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smemPerSM,
                                      cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    int smemPerBlock = smemPerSM / 2 + 1024;
    CUDA_CHECK(cudaFuncSetAttribute((void*)record_placement,
                                    cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CUDA_CHECK(cudaFuncSetAttribute((void*)record_placement,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smemPerBlock));
    printf("=== Placement sampling ===\n");
    printf("Forcing 1 block/SM with %d B dynamic shared memory per block\n",
           smemPerBlock);

    // Enough clusters to cover the device, rounded down to whole clusters.
    const int numClusters = numSM / clusterSize;
    const int numBlocks = numClusters * clusterSize;
    printf("Launching %d clusters x %d blocks = %d blocks, %d trials\n\n",
           numClusters, clusterSize, numBlocks, trials);

    // array of 48 unsigned, one per block, filled by the kernel: dSmid[i] = the SM id block i ran on
    unsigned *dSmid;
    // dCid[i] = which cluster block i belongs to (blockIdx.x / 12, so 0–3)
    unsigned *dCid;
    CUDA_CHECK(cudaMallocManaged(&dSmid, numBlocks * sizeof(unsigned)));
    CUDA_CHECK(cudaMallocManaged(&dCid, numBlocks * sizeof(unsigned)));

    // the union-find structure over the 48 SM ids.
    // It persists across all trials and accumulates the "same-GPC" facts:
    // for each list in byCluster, every SM is united with the first one.
    DSU dsu(numSM);
    // a set of every SM id observed in any trial. Persists across trials.
    std::set<int> seenSMs;
    int usableTrials = 0;

    for (int t = 0; t < trials; t++) {
        // Reset the SM id array to an impossible value so we can detect which blocks ran and which did not.
        for (int i = 0; i < numBlocks; i++) dSmid[i] = 0xFFFFFFFFu;

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
        usableTrials++;

        // Merge every SM seen in the same cluster.
        std::map<unsigned, std::vector<unsigned>> byCluster;
        for (int i = 0; i < numBlocks; i++) {
            if (dSmid[i] == 0xFFFFFFFFu) continue;
            byCluster[dCid[i]].push_back(dSmid[i]);
            seenSMs.insert((int)dSmid[i]);
        }
        for (auto& kv : byCluster)
            for (size_t j = 1; j < kv.second.size(); j++)
                dsu.unite((int)kv.second[0], (int)kv.second[j]);
    }

    // ---------------- results ----------------
    std::map<int, std::vector<int>> groups;
    for (int sm : seenSMs) groups[dsu.find(sm)].push_back(sm);

    printf("=== SM -> GPC mapping ===\n");
    printf("SMs observed: %zu of %d\n", seenSMs.size(), numSM);
    if ((int)seenSMs.size() < numSM)
        printf("WARNING: %d SM(s) never sampled; raise the trial count.\n",
               numSM - (int)seenSMs.size());

    int gpc = 0;
    bool uniform = true;
    size_t firstSize = groups.empty() ? 0 : groups.begin()->second.size();
    for (auto& kv : groups) {
        printf("  GPC %d (%zu SMs): ", gpc++, kv.second.size());
        for (int sm : kv.second) printf("%d ", sm);
        printf("\n");
        if (kv.second.size() != firstSize) uniform = false;
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

    // SM ids are typically interleaved across GPCs rather than assigned in
    // contiguous ranges. Test the common round-robin-by-TPC layout, in which
    // consecutive SMs form a TPC pair and consecutive TPCs cycle over GPCs:
    //     TPC  = smid / 2,  GPC = TPC % numGPC
    if (gpc > 0 && uniform) {
        std::map<int, int> gpcIndex;  // dsu root -> printed GPC index
        int idx = 0;
        for (auto& kv : groups) gpcIndex[kv.first] = idx++;
        bool formulaHolds = true;
        for (int sm : seenSMs)
            if ((sm / 2) % gpc != gpcIndex[dsu.find(sm)]) formulaHolds = false;
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