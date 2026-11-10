// topology.cuh — which SMs belong to which GPC, measured at run time.
//
// CUDA exposes neither the number of GPCs nor the SM-to-GPC mapping, but
// thread block clusters carry a hardware guarantee: all blocks of a cluster
// run on SMs of ONE GPC. So two SMs seen in the same cluster are in the same
// GPC. discover_topology() launches clusters of every size from the largest
// the GPU accepts down to 2, with one block per SM (each block asks for more
// than half of the SM's shared memory) and with 1, 2, ... clusters per launch
// so that the scheduler places them differently, and merges the "same cluster"
// facts with a union-find (disjoint-set) structure. The groups are the GPCs.
//
// On GB10 one launch of clusters of 12 already covers every GPC exactly (4 GPCs
// x 12 SMs). On GPUs whose GPCs are larger than the largest cluster, or of
// unequal size (H100: 14-18 SMs per GPC), the merging over many launches is
// what joins all the SMs of a GPC: check the result with build/gpc.
//
// GPCs are numbered by their smallest SM id. A TPC is the pair of SMs
// smid / 2 (checked: both SMs of a TPC must be in the same GPC), and a TPC's
// position is its index among the TPCs of its GPC, by TPC id. On GB10 this
// gives exactly GPC = (smid / 2) % 4 and position = (smid / 2) / 4.

#pragma once

#include <cooperative_groups.h>

#include <algorithm>
#include <map>
#include <numeric>
#include <vector>

#include "common.cuh"   // CUDA_CHECK, smid()

struct Topology {
    int num_sms = 0;
    int num_gpcs = 0;
    std::vector<int> gpc_of_sm;            // GPC of every SM
    std::vector<int> pos_of_sm;            // position of the SM's TPC inside its GPC
    std::vector<std::vector<int>> sms;     // the SMs of every GPC, sorted
    bool tpc_pairs_ok = true;              // both SMs of every TPC in the same GPC
};

// Every block records its SM and its cluster; the dynamic shared memory request
// (sized by the host) forces one block per SM.
__global__ void topology_probe(unsigned* sm_of_block, unsigned* cluster_of_block) {
    extern __shared__ char scratch[];
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    if (threadIdx.x == 0) {
        scratch[0] = 1;   // touch the allocation so it is not optimized away
        sm_of_block[blockIdx.x] = smid();
        cluster_of_block[blockIdx.x] = (unsigned)cooperative_groups::this_grid().cluster_rank();
    }
    cluster.sync();   // keep the whole cluster resident until every block has reported
}

// Union-find: the representative of SM x's group.
inline int topology_find(std::vector<int>& parent, int x) {
    while (parent[x] != x) x = parent[x] = parent[parent[x]];
    return x;
}

// GPCs are numbered by their smallest SM id.
inline bool topology_by_first_sm(const std::vector<int>& x, const std::vector<int>& y) {
    return x[0] < y[0];
}

inline Topology discover_topology() {
    Topology t;
    CUDA_CHECK(cudaDeviceGetAttribute(&t.num_sms, cudaDevAttrMultiProcessorCount, 0));
    int smem_per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    const int smem_per_block = smem_per_sm / 2 + 1024;   // more than half: one block per SM
    CUDA_CHECK(cudaFuncSetAttribute((const void*)topology_probe,
                                    cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CUDA_CHECK(cudaFuncSetAttribute((const void*)topology_probe,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_per_block));

    unsigned *sm_of_block, *cluster_of_block;
    CUDA_CHECK(cudaMallocManaged(&sm_of_block, t.num_sms * sizeof(unsigned)));
    CUDA_CHECK(cudaMallocManaged(&cluster_of_block, t.num_sms * sizeof(unsigned)));
    std::vector<int> parent(t.num_sms);
    std::iota(parent.begin(), parent.end(), 0);

    for (int cs = 16; cs >= 2; cs--) {
        cudaLaunchConfig_t cfg = {};
        cfg.blockDim = dim3(32, 1, 1);
        cfg.dynamicSmemBytes = smem_per_block;
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim.x = cs;
        attr.val.clusterDim.y = 1;
        attr.val.clusterDim.z = 1;
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
        cfg.gridDim = dim3(cs, 1, 1);
        int max_clusters = 0;
        if (cudaOccupancyMaxActiveClusters(&max_clusters, (const void*)topology_probe, &cfg) != cudaSuccess) {
            cudaGetLastError();   // this cluster size is not supported: try the next one
            continue;
        }
        // 1, 2, ... clusters per launch: the scheduler places them differently.
        for (int clusters = 1; clusters <= max_clusters; clusters++) {
            cfg.gridDim = dim3(clusters * cs, 1, 1);
            for (int b = 0; b < clusters * cs; b++) sm_of_block[b] = 0xFFFFFFFFu;
            CUDA_CHECK(cudaLaunchKernelEx(&cfg, topology_probe, sm_of_block, cluster_of_block));
            CUDA_CHECK(cudaDeviceSynchronize());
            std::map<unsigned, int> first_sm_of_cluster;
            for (int b = 0; b < clusters * cs; b++) {
                if (sm_of_block[b] == 0xFFFFFFFFu) continue;
                const int sm = (int)sm_of_block[b];
                std::map<unsigned, int>::iterator it = first_sm_of_cluster.find(cluster_of_block[b]);
                if (it == first_sm_of_cluster.end()) first_sm_of_cluster[cluster_of_block[b]] = sm;
                else parent[topology_find(parent, sm)] = topology_find(parent, it->second);
            }
        }
    }
    CUDA_CHECK(cudaFree(sm_of_block));
    CUDA_CHECK(cudaFree(cluster_of_block));

    // The groups, numbered by their smallest SM id.
    std::map<int, std::vector<int>> by_root;
    for (int sm = 0; sm < t.num_sms; sm++) by_root[topology_find(parent, sm)].push_back(sm);
    for (std::map<int, std::vector<int>>::iterator it = by_root.begin(); it != by_root.end(); ++it)
        t.sms.push_back(it->second);
    std::sort(t.sms.begin(), t.sms.end(), topology_by_first_sm);
    t.num_gpcs = (int)t.sms.size();
    t.gpc_of_sm.assign(t.num_sms, -1);
    t.pos_of_sm.assign(t.num_sms, -1);
    for (int g = 0; g < t.num_gpcs; g++) {
        std::vector<int> tpcs;
        for (int sm : t.sms[g]) {
            t.gpc_of_sm[sm] = g;
            if (tpcs.empty() || tpcs.back() != sm / 2) tpcs.push_back(sm / 2);
        }
        for (int sm : t.sms[g])
            t.pos_of_sm[sm] = (int)(std::find(tpcs.begin(), tpcs.end(), sm / 2) - tpcs.begin());
    }
    for (int sm = 0; sm + 1 < t.num_sms; sm += 2)
        if (t.gpc_of_sm[sm] != t.gpc_of_sm[sm + 1]) t.tpc_pairs_ok = false;
    return t;
}
