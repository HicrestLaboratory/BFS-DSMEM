// latency-many-threads.cu — DSMEM latency UNDER LOAD.
//
// dsmem_remote.cu measures ONE thread chasing alone. That is the unloaded
// latency: the raw cost of the path when nothing else is in flight. It is a
// real number and a lower bound, but no kernel ever runs like that. Here
// R requester blocks each run N chasing threads at the same time, all
// against a single owner block's shared memory (rank 0, the hot spot).
// Requests now queue: at the issuing SM, in the SM-to-SM fabric, and at the
// owner's SRAM port. What comes out is the latency-versus-load curve whose
// two endpoints were already known — ~200 cycles unloaded (dsmem_remote) and
// ~5 GB/s per SM saturated (throughput.cu) — and whose middle is where a real
// algorithm lives. Little's law ties the three together:
//     loads in flight  =  throughput (loads/cycle)  x  latency (cycles)
//
// Every WARP times its own chase (a warp's 32 lanes issue one load
// instruction and finish together, so per-lane rows would be 32 copies), and
// every warp is emitted as its own row: the spread across warps is the
// fairness of the arbitration, which is exactly the "scheduling" question.
//
//   --chasers N          chasing threads per requester block, 1..1024
//   --requesters R       requester blocks, 1..11; the cluster is R+1 blocks
//   --target remote|local
//                        remote: rank 0's buffer through map_shared_rank (LD)
//                        local:  each block's own buffer (LDS) — the control
//                        that tells a fabric bottleneck from an SRAM-port one
//   --bank-aligned 1|0   1: lane l walks a cycle inside bank l, so a warp
//                        never conflicts with itself (default);
//                        0: all lanes walk one random cycle — random banks
//
// Build: make latency-many-threads
// Run:   ./latency-many-threads --target remote --requesters 4 --chasers 256

#include <cooperative_groups.h>

#include "common.cuh"

namespace cg = cooperative_groups;

// 32 independent random cycles, one per shared-memory bank (bank = element
// index mod 32). Lane l of every warp walks bank l's cycle, so the 32 lanes
// of one load instruction always hit 32 different banks.
void make_bank_cycles(std::vector<unsigned>& p, std::mt19937& rng) {
    const int per_bank = SBUF / 32;  // 128 elements in each bank
    std::vector<unsigned> c(per_bank);
    for (int b = 0; b < 32; b++) {
        make_cycle(c, rng);  // c[k] = successor of k inside this bank's cycle
        for (int k = 0; k < per_bank; k++) p[b + 32 * k] = b + 32 * c[k];
    }
}

__global__ void loaded_chase(const unsigned* __restrict__ perm, int chasers,
                             int target_remote, int bank_aligned, int warmup,
                             int steps, Result* out) {
    __shared__ unsigned sbuf[SBUF];
    cg::cluster_group cluster = cg::this_cluster();

    for (int i = threadIdx.x; i < SBUF; i += blockDim.x) sbuf[i] = perm[i];
    __syncthreads();
    cluster.sync();  // every buffer filled; every chaser starts from here

    unsigned rank = cluster.block_rank();
    if (rank != 0 && (int)threadIdx.x < chasers) {  // rank 0 only owns memory
        int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        // Distinct starting points. Bank-aligned: lane l must start on an
        // element of bank l (index = l mod 32); warp w takes the w-th one.
        // Random cycle: any hash that spreads the threads out will do.
        unsigned start = bank_aligned
            ? (unsigned)(lane + 32 * warp)
            : (unsigned)(((threadIdx.x + 1024u * (rank - 1)) * 2654435761u) % SBUF);
        Result r;
        if (target_remote) {
            const unsigned* buf = cluster.map_shared_rank((const unsigned*)sbuf, 0);
            warm_then_chase(buf, start, warmup, steps, &r);
        } else {
            warm_then_chase(sbuf, start, warmup, steps, &r);  // provably shared: LDS
        }
        if (lane == 0) out[(rank - 1) * (blockDim.x / 32) + warp] = r;
    }

    // The owner's shared memory is freed when the owner exits: keep every
    // block alive until the last chaser is done.
    cluster.sync();
}

int main(int argc, char** argv) {
    Args a;
    parse_args(argc, argv, &a, "latency-many-threads",
               "  --chasers N       chasing threads per requester block, 1..1024 (default 1)\n"
               "  --requesters R    requester blocks, 1..11; cluster is R+1     (default 1)\n"
               "  --target remote|local  rank 0's buffer via map_shared_rank, or own buffer (default remote)\n"
               "  --bank-aligned 0|1  1 = lane l walks bank l, no intra-warp conflicts (default 1)\n");
    if (a.chasers < 1 || a.chasers > 1024) {
        fprintf(stderr, "latency-many-threads: --chasers must be 1..1024\n");
        return 1;
    }
    if (a.requesters < 1 || a.requesters > 11) {
        fprintf(stderr, "latency-many-threads: --requesters must be 1..11 (cluster size %d > 12)\n",
                a.requesters + 1);
        return 1;
    }
    int block_size = ((a.chasers + 31) / 32) * 32;  // whole warps only
    int warps = block_size / 32;                     // chasing warps per block
    int cluster_size = a.requesters + 1;
    int rows = a.requesters * warps;                 // one Result per chasing warp

    char extra[256];
    snprintf(extra, sizeof extra,
             "# chasers: %d\n# requesters: %d\n# target: %s\n# bank_aligned: %d\n"
             "# warps_per_requester: %d\n",
             a.chasers, a.requesters, a.target_remote ? "remote" : "local",
             a.bank_aligned, warps);
    print_header(argc, argv, a, SBUF, extra);

    std::mt19937 rng(a.seed);
    std::vector<unsigned> hperm(SBUF);
    if (a.bank_aligned) make_bank_cycles(hperm, rng);
    else                make_pattern(hperm, rng, a.stride_bytes);
    unsigned* dperm;
    CUDA_CHECK(cudaMalloc(&dperm, SBUF * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(dperm, hperm.data(), SBUF * sizeof(unsigned),
                          cudaMemcpyHostToDevice));

    Result* out;
    CUDA_CHECK(cudaMallocManaged(&out, rows * sizeof(Result)));

    if (cluster_size > 8) allow_big_clusters((const void*)loaded_chase);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(cluster_size, 1, 1);
    cfg.blockDim = dim3(block_size, 1, 1);
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = cluster_size;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;

    const char* name = a.target_remote ? "dsmem_loaded" : "smem_loaded";
    double total_loads = (double)a.requesters * a.chasers * a.steps;

    std::vector<double> mean_cpl, gbps;  // one entry per repetition
    for (int rep = 0; rep < a.reps; rep++) {
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, loaded_chase, (const unsigned*)dperm,
                                      a.chasers, a.target_remote, a.bank_aligned,
                                      a.warmup, a.steps, out));
        CUDA_CHECK(cudaDeviceSynchronize());

        double sum_cpl = 0, max_ns = 0;
        for (int i = 0; i < rows; i++) {
            print_row(name, cluster_size, 0, a.target_remote, block_size, a.steps,
                      0, a.stride_bytes, a.seed, rep, out[i], a.chasers, i);
            sum_cpl += (double)out[i].cycles / a.steps;
            if ((double)out[i].ns > max_ns) max_ns = (double)out[i].ns;
        }
        mean_cpl.push_back(sum_cpl / rows);
        // Aggregate throughput of the whole cluster: all loads, over the
        // slowest warp's wall time (every warp started at the same barrier).
        gbps.push_back(total_loads * sizeof(unsigned) / max_ns);  // bytes/ns = GB/s
    }

    fprintf(stderr, "%s: %d requester(s) x %d chaser(s) = %d chasing warps, target %s\n",
            name, a.requesters, a.chasers, rows, a.target_remote ? "rank 0" : "own");
    print_stats_line("mean cy/load", mean_cpl);
    print_stats_line("GB/s", gbps);
    {
        // Little's law check: loads in flight = throughput x latency.
        std::vector<double> s = mean_cpl; std::sort(s.begin(), s.end());
        std::vector<double> g = gbps;     std::sort(g.begin(), g.end());
        double lat = s[s.size() / 2], thr = g[g.size() / 2];
        double loads_per_cycle = thr / sizeof(unsigned) / 2.4;  // GB/s -> loads/ns -> loads/cycle @2.4 GHz
        fprintf(stderr, "  little's law: %.2f loads/cycle x %.1f cy = %.0f loads in flight "
                        "(%d chasing threads issued)\n",
                loads_per_cycle, lat, loads_per_cycle * lat, a.requesters * a.chasers);
    }

    cudaFree(dperm);
    cudaFree(out);
    return 0;
}
