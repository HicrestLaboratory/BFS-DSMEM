// latency-many-threads.cu — DSMEM under load, and whether it matters WHO you read.
//
// dsmem_remote.cu measures ONE thread chasing alone: the unloaded latency, a
// real number but a lower bound no kernel ever sees. Here every reader block
// runs a full complement of threads, so requests queue: at the issuing SM, in
// the SM-to-SM fabric, and at the owner's SRAM port. Little's law ties the
// three quantities together, and because the chase is dependent each thread
// has exactly ONE load outstanding, so
//     loads in flight  =  throughput (loads/cycle) x latency (cycles)
//                      =  number of chasing threads, by construction.
// That identity is printed as a self-check: if the timing or the throughput
// arithmetic were wrong, it would not come out.
//
// THE PATTERN. --pattern decides whose shared memory each block reads, and
// nothing else changes between the three:
//
//   broadcast   every block reads rank 0; rank 0 itself only owns memory.
//             Maximum concentration: cluster_size-1 readers, ONE owner.
//   ring      rank k reads rank k+1 (mod cluster_size). Maximum spread:
//             every rank is a reader AND an owner, exactly one reader each.
//   random    each WARP draws its own target rank (never its own). Realistic
//             imbalance: about one reader per owner, but clumped by luck.
//
// broadcast vs ring at the same thread count is the experiment that matters. If
// ring delivers ~N times the aggregate throughput, the saturation point is a
// PER-OWNER limit and spreading ownership multiplies bandwidth. If ring
// delivers the same, the fabric itself is the ceiling and no partitioning
// scheme can help.
//
// WHY THE TARGET IS DRAWN PER WARP. A warp issues one load instruction for
// all 32 lanes. If every lane addresses the same peer, that is one coherent
// request to one remote SM, exactly like broadcast but aimed elsewhere. If
// lanes addressed different peers, one instruction would have to fan out to
// up to 32 SMs, and a bad result could not be attributed to spreading rather
// than to fan-out. Per-warp changes exactly one variable versus broadcast.
// (Per-thread fan-out is a separate experiment, deliberately not done here.)
//
//   --cluster-size N   blocks in the cluster, 2..12              (default 12)
//   --block-size N     threads per block, multiple of 32         (default 128 = 4 warps)
//   --pattern P        broadcast | ring | random                   (default broadcast)
//   --access A         random | coalesced                        (default random)
//
// THE ACCESS. Both are dependent chases (one load in flight per thread); they
// differ only in where the 32 lanes of a warp read at the same step:
//
//   random     one random cycle, every lane at its own random entry point: an
//              instruction touches ~32 different lines, ~32 separate requests.
//   coalesced  buf[i] = i + 32 and lane l starts at base + l, so at every step
//              the lanes read 32 consecutive words, ONE 128 B line: the
//              hardware merges them into a single request.
//
// Every WARP times its own chase and is emitted as its own CSV row: the 32
// lanes of a warp run in lockstep and hold the same measurement, so per-lane
// rows would be 32 copies. The spread ACROSS warps is the fairness of the
// arbitration.
//
// Build: make latency-many-threads
// Run:   ./latency-many-threads --pattern ring --block-size 128

#include <cooperative_groups.h>

#include "../common.cuh"

namespace cg = cooperative_groups;

enum { PAT_broadcast = 0, PAT_RING = 1, PAT_RANDOM = 2 };
static const char* PATTERN_NAME[] = {"broadcast", "ring", "random"};

// A cheap, well-mixed hash: used both to pick random targets and to scatter
// starting offsets. Deterministic, so a run is reproducible from --seed.
__device__ __forceinline__ unsigned mix(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__global__ void loaded_chase(const unsigned* __restrict__ perm, int pattern,
                             int coalesced, int chasers, int seed, int warmup,
                             int steps, Result* out, unsigned* smids) {
    __shared__ unsigned sbuf[SBUF];
    cg::cluster_group cluster = cg::this_cluster();
    const int cs = (int)cluster.num_blocks();
    const int rank = (int)cluster.block_rank();
    const int warps = blockDim.x / 32;

    for (int i = threadIdx.x; i < SBUF; i += blockDim.x) sbuf[i] = perm[i];
    __syncthreads();
    if (threadIdx.x == 0) smids[rank] = smid();
    cluster.sync();  // every buffer filled before anyone reads a peer's

    const int lane = threadIdx.x & 31, warp = (int)threadIdx.x >> 5;

    // Who does this warp read from? -1 means "this block does not read".
    int target = -1;
    if (pattern == PAT_broadcast) {
        if (rank != 0) target = 0;                  // rank 0 only owns memory
    } else if (pattern == PAT_RING) {
        target = (rank + 1) % cs;
    } else {                                        // PAT_RANDOM, per warp
        // uniform over the cs-1 ranks that are not me, then skip self
        target = (int)(mix((unsigned)(rank * 977 + warp * 31 + seed)) % (unsigned)(cs - 1));
        if (target >= rank) target++;
    }

    if (target >= 0 && (int)threadIdx.x < chasers) {
        // pchase: distinct random entry points, or the lanes would read one
        // address and the hardware would broadcast instead of making
        // independent requests. coalesced: lane l at base + l, each warp on
        // its own line.
        unsigned start = coalesced
            ? (unsigned)((32 * (rank * warps + warp) + lane) % SBUF)
            : (mix((unsigned)(threadIdx.x + 1024 * rank + 7 * seed)) % SBUF);
        Result r;
        const unsigned* buf = cluster.map_shared_rank((const unsigned*)sbuf, target);
        warm_then_chase(buf, start, warmup, steps, &r);
        if (lane == 0) {
            out[rank * warps + warp] = r;
            smids[cs + rank * warps + warp] = (unsigned)target;   // who I read
        }
    }

    // A block's shared memory is released when it exits: hold every block
    // alive until the last reader has finished with it.
    cluster.sync();
}

int main(int argc, char** argv) {
    Args a;
    a.cluster_size = 12;   // Args' default of 2 means "unset" for this program
    parse_args(argc, argv, &a, "latency-many-threads",
               "  --cluster-size N  blocks in the cluster, 2..12          (default 12)\n"
               "  --block-size N    threads per block, multiple of 32     (default 128)\n"
               "  --pattern P       broadcast | ring | random               (default broadcast)\n"
               "  --access A        random | coalesced                    (default random)\n"
               "  --chasers N       chasing threads per block; 0 = all      (default 0)\n");
    const int cs = a.cluster_size;
    if (cs < 2 || cs > 12) {
        fprintf(stderr, "latency-many-threads: --cluster-size must be 2..12 on GB10\n");
        return 1;
    }
    if (a.block_size < 32 || a.block_size > 1024 || a.block_size % 32) {
        fprintf(stderr, "latency-many-threads: --block-size must be a multiple of 32, 32..1024\n");
        return 1;
    }
    if (a.chasers < 0 || a.chasers > a.block_size) {
        fprintf(stderr, "latency-many-threads: --chasers must be 0..--block-size\n");
        return 1;
    }
    if (a.access == 1) a.access = 0;   // "random" and "pchase" name the same chase here
    const bool coalesced = (a.access == 2);
    if (coalesced && a.stride_bytes) {
        fprintf(stderr, "latency-many-threads: --stride only applies to --access random\n");
        return 1;
    }
    const char* access_name = coalesced ? "coalesced" : "random";
    if (a.chasers == 0) a.chasers = a.block_size;     // 0 means "every thread"
    const int warps = a.block_size / 32;
    const int active_warps = (a.chasers + 31) / 32;   // warps that hold a result
    // broadcast keeps rank 0 passive; ring and random make every rank a reader.
    const int readers = (a.pattern == PAT_broadcast) ? cs - 1 : cs;
    const int rows = cs * warps;           // rank 0's slots stay empty in broadcast

    char extra[320];
    snprintf(extra, sizeof extra,
             "# pattern: %s\n# cluster_size: %d\n# readers: %d\n# chasers_per_block: %d\n"
             "# active_warps: %d\n# access: %s\n",
             PATTERN_NAME[a.pattern], cs, readers, a.chasers, active_warps, access_name);
    print_header(argc, argv, a, SBUF, extra);

    std::mt19937 rng(a.seed);
    std::vector<unsigned> hperm(SBUF);
    if (coalesced) for (int i = 0; i < SBUF; i++) hperm[i] = (i + 32) % SBUF;  // one line per step
    else           make_pattern(hperm, rng, a.stride_bytes);
    unsigned* dperm;
    CUDA_CHECK(cudaMalloc(&dperm, SBUF * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(dperm, hperm.data(), SBUF * sizeof(unsigned),
                          cudaMemcpyHostToDevice));

    Result* out;     CUDA_CHECK(cudaMallocManaged(&out, rows * sizeof(Result)));
    unsigned* smids; CUDA_CHECK(cudaMallocManaged(&smids, (cs + rows) * sizeof(unsigned)));

    if (cs > 8) allow_big_clusters((const void*)loaded_chase);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(cs, 1, 1);
    cfg.blockDim = dim3(a.block_size, 1, 1);
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = cs;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;

    char name[48];
    // pchase keeps the historical name, so old and new logs stay comparable
    snprintf(name, sizeof name, coalesced ? "dsmem_%s_coalesced" : "dsmem_%s",
             PATTERN_NAME[a.pattern]);
    const double total_loads = (double)readers * a.chasers * a.steps;

    std::vector<double> mean_cpl, gbps;   // one entry per repetition
    for (int rep = 0; rep < a.reps; rep++) {
        for (int i = 0; i < rows; i++) out[i].cycles = 0;
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, loaded_chase, (const unsigned*)dperm,
                                      a.pattern, (int)coalesced, a.chasers,
                                      a.seed, a.warmup, a.steps, out, smids));
        CUDA_CHECK(cudaDeviceSynchronize());

        double sum_cpl = 0, max_ns = 0;
        int n = 0;
        for (int i = 0; i < rows; i++) {
            if (out[i].cycles == 0) continue;       // rank 0 in broadcast: no data
            int rank = i / warps, warp = i % warps;
            int target = (int)smids[cs + i];
            print_row(name, cs, (target - rank + cs) % cs, 1, a.block_size, a.steps,
                      0, a.stride_bytes, a.seed, rep, out[i], a.block_size, warp,
                      rank, target, (int)smids[rank], (int)smids[target]);
            sum_cpl += (double)out[i].cycles / a.steps;
            if ((double)out[i].ns > max_ns) max_ns = (double)out[i].ns;
            n++;
        }
        mean_cpl.push_back(sum_cpl / n);
        // Aggregate throughput: all the loads, over the SLOWEST warp's wall
        // time. Every warp started at the same cluster.sync(), so the slowest
        // one's elapsed time is the wall time of the whole experiment.
        gbps.push_back(total_loads * sizeof(unsigned) / max_ns);  // bytes/ns = GB/s
    }

    fprintf(stderr, "%s: cluster %d, %d readers x %d chasing threads, access %s\n",
            name, cs, readers, a.chasers, access_name);
    print_stats_line("mean cy/load", mean_cpl);
    print_stats_line("GB/s total", gbps);
    {
        std::vector<double> g = gbps;     std::sort(g.begin(), g.end());
        std::vector<double> l = mean_cpl; std::sort(l.begin(), l.end());
        const double thr = g[g.size() / 2], lat = l[l.size() / 2];
        // Patterns differ in reader count (broadcast has one fewer), so the
        // per-reader rate is what makes them comparable at a glance.
        fprintf(stderr, "  per reader   : %.3f GB/s   (%d readers)\n", thr / readers, readers);
        const double loads_per_cycle = thr / sizeof(unsigned) / 2.4;
        fprintf(stderr, "  little's law : %.3f loads/cycle x %.1f cy = %.0f in flight "
                        "(%d threads issued)\n",
                loads_per_cycle, lat, loads_per_cycle * lat, readers * a.chasers);
    }

    cudaFree(dperm); cudaFree(out); cudaFree(smids);
    return 0;
}
