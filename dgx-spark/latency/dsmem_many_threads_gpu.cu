// dsmem_many_threads_glob.cu — dsmem_many_threads.cu over the WHOLE chip.
//
// dsmem_many_threads.cu launches one cluster, so its cluster size is also the
// number of SMs doing the work: a bigger cluster means more SMs and more total
// throughput. Luo et al. ("Dissecting the NVIDIA Hopper Architecture through
// Microbenchmarking", 2025) instead fill the whole GPU with clusters and vary
// only how the SMs are grouped; on H800 they report 3.28 TB/s at cluster size
// 2 and 2.78 TB/s at 4. This program is that experiment on GB10: --clusters
// independent clusters run the same pattern side by side, and the reported
// throughput is the total over all of them.
//
// Three things differ from dsmem_many_threads.cu:
//
//   ONE BLOCK PER SM. With a 32 KiB buffer, two or three blocks fit on one SM,
//   and the scheduler may stack them, leaving SMs idle while the output
//   claims 48 of them busy. Every block therefore also requests enough unused
//   dynamic shared memory to take more than half an SM (the same trick as
//   ../extra/l2_topology.cu), and the SMs the blocks landed on are checked
//   after every launch.
//
//   ALL CLUSTERS AT ONCE. If more clusters are launched than can be resident,
//   the extra ones run after the first wave and the total would be an average
//   over two waves. cudaOccupancyMaxActiveClusters is checked before launching.
//
//   WALL TIME ACROSS CLUSTERS. Inside one cluster every warp leaves the same
//   cluster.sync(), so the longest warp is the wall time. Different clusters
//   can start microseconds apart, so each warp records its absolute
//   %globaltimer start and end, and throughput = bytes / (last end - first start).
//
//   --cluster-size N   blocks per cluster, 2..12                  (default 2)
//   --clusters N       clusters to launch; 0 = fill every SM once (default 0)
//   --block-size N     threads per block, multiple of 32          (default 128)
//   --pattern P        hotspot | ring | random, inside each cluster (default ring)
//   --access A         pchase | coalesced                         (default coalesced)
//
// A GB10 GPC has 12 SMs and a cluster never spans two GPCs, so only cluster
// sizes that divide 12 (2, 3, 4, 6, 12) can use all 48 SMs.
//
// CSV: one row per warp, the shared schema of common.cuh. `reader` and
// `target` are GLOBAL block indices (cluster * cluster_size + rank), so rows
// from different clusters stay distinct; `distance` is the rank distance
// inside the cluster.
//
// Build: make dsmem_many_threads_glob
// Run:   ./dsmem_many_threads_glob --cluster-size 2 --pattern ring --access coalesced

#include <cooperative_groups.h>

#include <set>

#include "../common.cuh"

namespace cg = cooperative_groups;

enum { PAT_HOTSPOT = 0, PAT_RING = 1, PAT_RANDOM = 2 };
static const char* PATTERN_NAME[] = {"hotspot", "ring", "random"};

__device__ __forceinline__ unsigned mix(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// out[] and span[] are indexed by (global block, warp); smids[] holds the SM
// of every block, then the target rank of every warp.
__global__ void loaded_chase(const unsigned* __restrict__ perm, int pattern,
                             int coalesced, int chasers, int seed, int warmup,
                             int steps, Result* out, unsigned long long* span,
                             unsigned* smids) {
    __shared__ unsigned sbuf[SBUF];
    cg::cluster_group cluster = cg::this_cluster();
    const int cs = (int)cluster.num_blocks();
    const int rank = (int)cluster.block_rank();
    const int warps = blockDim.x / 32;
    const int blocks = (int)gridDim.x;
    const int gb = (int)blockIdx.x;   // with a 1-D grid: cluster = gb / cs, rank = gb % cs

    for (int i = threadIdx.x; i < SBUF; i += blockDim.x) sbuf[i] = perm[i];
    __syncthreads();
    if (threadIdx.x == 0) smids[gb] = smid();
    cluster.sync();

    const int lane = threadIdx.x & 31, warp = (int)threadIdx.x >> 5;

    int target = -1;
    if (pattern == PAT_HOTSPOT) {
        if (rank != 0) target = 0;
    } else if (pattern == PAT_RING) {
        target = (rank + 1) % cs;
    } else {
        target = (int)(mix((unsigned)(gb * 977 + warp * 31 + seed)) % (unsigned)(cs - 1));
        if (target >= rank) target++;
    }

    if (target >= 0 && (int)threadIdx.x < chasers) {
        unsigned idx = coalesced
            ? (unsigned)((32 * (rank * warps + warp) + lane) % SBUF)
            : (mix((unsigned)(threadIdx.x + 1024 * gb + 7 * seed)) % SBUF);
        // map_shared_rank always addresses a block of MY cluster: every
        // cluster runs its own pattern, independently of the others.
        const unsigned* buf = cluster.map_shared_rank((const unsigned*)sbuf, target);
        for (int w = 0; w < warmup; w++)
            for (int i = 0; i < steps; i++) idx = buf[idx];
        Result r;
        const unsigned long long g0 = globaltimer();
        chase(buf, idx, steps, &r);
        const unsigned long long g1 = globaltimer();
        if (lane == 0) {
            const int row = gb * warps + warp;
            out[row] = r;
            span[2 * row] = g0;
            span[2 * row + 1] = g1;
            smids[blocks + row] = (unsigned)target;
        }
    }
    cluster.sync();
}

int main(int argc, char** argv) {
    Args a;
    a.cluster_size = 2;
    a.clusters = 0;           // 0 = fill the chip
    a.pattern = PAT_RING;
    a.access = 2;             // coalesced
    parse_args(argc, argv, &a, "dsmem_many_threads_glob",
               "  --cluster-size N  blocks per cluster, 2..12              (default 2)\n"
               "  --clusters N      clusters to launch; 0 = fill every SM   (default 0)\n"
               "  --block-size N    threads per block, multiple of 32      (default 128)\n"
               "  --pattern P       hotspot | ring | random                (default ring)\n"
               "  --access A        pchase | coalesced                     (default coalesced)\n"
               "  --chasers N       chasing threads per block; 0 = all     (default 0)\n");
    const int cs = a.cluster_size;
    if (cs < 2 || cs > 12) {
        fprintf(stderr, "dsmem_many_threads_glob: --cluster-size must be 2..12 on GB10\n");
        return 1;
    }
    if (a.block_size < 32 || a.block_size > 1024 || a.block_size % 32) {
        fprintf(stderr, "dsmem_many_threads_glob: --block-size must be a multiple of 32, 32..1024\n");
        return 1;
    }
    if (a.chasers < 0 || a.chasers > a.block_size) {
        fprintf(stderr, "dsmem_many_threads_glob: --chasers must be 0..--block-size\n");
        return 1;
    }
    if (a.access == 1) {
        fprintf(stderr, "dsmem_many_threads_glob: --access must be pchase or coalesced\n");
        return 1;
    }
    const bool coalesced = (a.access == 2);
    if (coalesced && a.stride_bytes) {
        fprintf(stderr, "dsmem_many_threads_glob: --stride only applies to --access pchase\n");
        return 1;
    }
    const char* access_name = coalesced ? "coalesced" : "pchase";
    if (a.chasers == 0) a.chasers = a.block_size;

    int nsm = 0, smem_per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    if (a.clusters == 0) a.clusters = nsm / cs;
    if (a.clusters < 1) {
        fprintf(stderr, "dsmem_many_threads_glob: --clusters must be >= 1\n");
        return 1;
    }
    const int clusters = a.clusters, blocks = clusters * cs;
    const int warps = a.block_size / 32;
    const int readers_per_cluster = (a.pattern == PAT_HOTSPOT) ? cs - 1 : cs;
    const int readers = clusters * readers_per_cluster;
    const int rows = blocks * warps;

    // One block per SM: static buffer + unused dynamic padding > half an SM.
    const int static_bytes = SBUF * (int)sizeof(unsigned);
    const int pad_bytes = std::max(0, smem_per_sm / 2 + 1024 - static_bytes);
    CUDA_CHECK(cudaFuncSetAttribute((const void*)loaded_chase,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, pad_bytes));
    if (cs > 8) allow_big_clusters((const void*)loaded_chase);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(blocks, 1, 1);
    cfg.blockDim = dim3(a.block_size, 1, 1);
    cfg.dynamicSmemBytes = pad_bytes;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = cs;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;

    int max_clusters = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveClusters(&max_clusters, (void*)loaded_chase, &cfg));
    if (clusters > max_clusters) {
        fprintf(stderr, "dsmem_many_threads_glob: %d clusters of %d requested, but only %d fit at "
                        "once (one block per SM); use --clusters %d or a cluster size that "
                        "divides 12\n", clusters, cs, max_clusters, max_clusters);
        return 1;
    }

    char extra[400];
    snprintf(extra, sizeof extra,
             "# pattern: %s\n# access: %s\n# cluster_size: %d\n# clusters: %d\n# blocks: %d\n"
             "# readers: %d\n# chasers_per_block: %d\n# pad_dynamic_smem_bytes: %d\n",
             PATTERN_NAME[a.pattern], access_name, cs, clusters, blocks, readers, a.chasers,
             pad_bytes);
    print_header(argc, argv, a, SBUF, extra);

    std::mt19937 rng(a.seed);
    std::vector<unsigned> hperm(SBUF);
    if (coalesced) for (int i = 0; i < SBUF; i++) hperm[i] = (i + 32) % SBUF;  // one line per step
    else           make_pattern(hperm, rng, a.stride_bytes);
    unsigned* dperm;
    CUDA_CHECK(cudaMalloc(&dperm, SBUF * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(dperm, hperm.data(), SBUF * sizeof(unsigned), cudaMemcpyHostToDevice));

    Result* out;             CUDA_CHECK(cudaMallocManaged(&out, rows * sizeof(Result)));
    unsigned long long* span; CUDA_CHECK(cudaMallocManaged(&span, 2 * rows * sizeof(unsigned long long)));
    unsigned* smids;         CUDA_CHECK(cudaMallocManaged(&smids, (blocks + rows) * sizeof(unsigned)));

    char name[64];
    snprintf(name, sizeof name, coalesced ? "dsmem_glob_%s_coalesced" : "dsmem_glob_%s",
             PATTERN_NAME[a.pattern]);
    const double total_bytes = (double)readers * a.chasers * a.steps * sizeof(unsigned);

    std::vector<double> mean_cpl, gbps;
    int distinct_sms = 0;
    for (int rep = 0; rep < a.reps; rep++) {
        for (int i = 0; i < rows; i++) out[i].cycles = 0;
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, loaded_chase, (const unsigned*)dperm, a.pattern,
                                      (int)coalesced, a.chasers, a.seed, a.warmup, a.steps,
                                      out, span, smids));
        CUDA_CHECK(cudaDeviceSynchronize());

        std::set<unsigned> sms(smids, smids + blocks);
        distinct_sms = (int)sms.size();
        if (distinct_sms != blocks) {
            fprintf(stderr, "dsmem_many_threads_glob: %d blocks landed on only %d SMs; "
                            "the one-block-per-SM padding did not hold\n", blocks, distinct_sms);
            return 1;
        }

        unsigned long long first = ~0ull, last = 0;
        double sum_cpl = 0;
        int n = 0;
        for (int i = 0; i < rows; i++) {
            if (out[i].cycles == 0) continue;           // hotspot owners: no data
            const int gb = i / warps, warp = i % warps;
            const int rank = gb % cs, cl = gb / cs;
            const int target = (int)smids[blocks + i];  // rank inside the cluster
            const int tgb = cl * cs + target;
            print_row(name, cs, (target - rank + cs) % cs, 1, a.block_size, a.steps, 0,
                      a.stride_bytes, a.seed, rep, out[i], a.block_size, warp, gb, tgb,
                      (int)smids[gb], (int)smids[tgb]);
            first = std::min(first, span[2 * i]);
            last = std::max(last, span[2 * i + 1]);
            sum_cpl += (double)out[i].cycles / a.steps;
            n++;
        }
        mean_cpl.push_back(sum_cpl / n);
        gbps.push_back(total_bytes / (double)(last - first));   // bytes/ns = GB/s
    }

    fprintf(stderr, "%s: %d clusters x %d blocks on %d distinct SMs, %d readers x %d threads, "
                    "access %s\n", name, clusters, cs, distinct_sms, readers, a.chasers,
            access_name);
    print_stats_line("mean cy/load", mean_cpl);
    print_stats_line("GB/s total", gbps);
    {
        std::vector<double> g = gbps; std::sort(g.begin(), g.end());
        const double thr = g[g.size() / 2];
        fprintf(stderr, "  per cluster  : %.2f GB/s\n", thr / clusters);
        fprintf(stderr, "  per reader   : %.3f GB/s   (%d readers)\n", thr / readers, readers);
    }

    cudaFree(dperm); cudaFree(out); cudaFree(span); cudaFree(smids);
    return 0;
}
