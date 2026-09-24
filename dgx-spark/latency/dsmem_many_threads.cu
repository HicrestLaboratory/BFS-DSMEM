// dsmem_many_threads.cu — DSMEM under load, and whether it matters WHO you read.
//
// dsmem_remote.cu measures ONE thread alone: the unloaded latency. Here every
// thread of every reader block keeps reading a peer's shared memory, so
// requests queue at the issuing SM, in the SM-to-SM fabric and at the owner's
// SRAM port, and what comes out is the throughput the cluster sustains.
//
// THE PATTERN. --pattern decides whose shared memory each block reads:
//
//   broadcast every block reads rank 0; rank 0 itself only owns memory.
//             Maximum concentration: cluster_size-1 readers, ONE owner.
//   ring      rank k reads rank k+1 (mod cluster_size). Maximum spread:
//             every rank is a reader AND an owner, exactly one reader each.
//   random    each WARP draws its own target rank (never its own). Realistic
//             imbalance: about one reader per owner, but clumped by luck.
//
// The target is drawn per warp, not per thread, so that one load instruction
// always goes to one remote SM: a bad result then cannot come from a single
// instruction fanning out to up to 32 SMs.
//
// THE ACCESS. Every thread issues --steps independent loads, one at a time
// (each load's value is consumed before the next load is issued). --access
// decides what the 32 lanes of a warp read:
//
//   random     each lane a random word: an instruction touches ~32 different
//              lines, ~32 separate requests. BFS checking visited[v] for its
//              neighbours.
//   coalesced  lane l reads word l of a line, and each load moves to the next
//              line: ONE 128 B request per instruction. A block copying a
//              frontier segment out of another block.
//
// More loads in flight means more threads: --block-size.
//
//   --cluster-size N   blocks in the cluster, 2..12              (default 12)
//   --block-size N     threads per block, multiple of 32         (default 128 = 4 warps)
//   --pattern P        broadcast | ring | random                 (default broadcast)
//   --access A         random | coalesced                        (default random)
//
// Every WARP times its own loop and is emitted as its own CSV row: the 32
// lanes of a warp run in lockstep and hold the same measurement. The spread
// ACROSS warps is the fairness of the arbitration.
//
// Build: make dsmem_many_threads
// Run:   ./dsmem_many_threads --pattern ring --access coalesced --block-size 256

#include <cooperative_groups.h>

#include "../common.cuh"

namespace cg = cooperative_groups;

enum { PAT_BROADCAST = 0, PAT_RING = 1, PAT_RANDOM = 2 };
enum { ACC_RANDOM = 1, ACC_COALESCED = 2 };   // values of Args::access
static const char* PATTERN_NAME[] = {"broadcast", "ring", "random"};

// A cheap, well-mixed hash: used to pick random targets and to seed each
// thread's random addresses. Deterministic, so a run is reproducible from --seed.
__device__ __forceinline__ unsigned mix(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// `steps` loads by one thread. The loop is kept rolled so each load's value
// is xor-ed into `sink` before the next load issues: one load in flight per
// thread. `pos` carries on from the warm-up into the timed pass.
__device__ __forceinline__ unsigned read_loop(const unsigned* buf, int access,
                                              unsigned& pos, int steps) {
    unsigned sink = 0;
#pragma unroll 1
    for (int i = 0; i < steps; i++) {
        if (access == ACC_COALESCED) {
            sink ^= buf[pos % SBUF];
            pos += 32;                                   // next line, same word
        } else {
            pos = pos * 1664525u + 1013904223u;          // LCG: arithmetic, no memory
            sink ^= buf[(pos >> 8) % SBUF];              // low LCG bits are weak
        }
    }
    return sink;
}

__global__ void loaded(int pattern, int access, int seed, int warmup, int steps,
                       Result* out, unsigned* smids) {
    __shared__ unsigned sbuf[SBUF];
    cg::cluster_group cluster = cg::this_cluster();
    const int cs = (int)cluster.num_blocks();
    const int rank = (int)cluster.block_rank();
    const int warps = blockDim.x / 32;

    for (int i = threadIdx.x; i < SBUF; i += blockDim.x) sbuf[i] = i;   // any contents do
    __syncthreads();
    if (threadIdx.x == 0) smids[rank] = smid();
    cluster.sync();  // every buffer filled before anyone reads a peer's

    const int lane = threadIdx.x & 31, warp = (int)threadIdx.x >> 5;

    // Who does this warp read from? -1 means "this block does not read".
    int target = -1;
    if (pattern == PAT_BROADCAST) {
        if (rank != 0) target = 0;                  // rank 0 only owns memory
    } else if (pattern == PAT_RING) {
        target = (rank + 1) % cs;
    } else {                                        // PAT_RANDOM, per warp
        // uniform over the cs-1 ranks that are not me, then skip self
        target = (int)(mix((unsigned)(rank * 977 + warp * 31 + seed)) % (unsigned)(cs - 1));
        if (target >= rank) target++;
    }

    if (target >= 0) {
        const unsigned* buf = cluster.map_shared_rank((const unsigned*)sbuf, target);
        // coalesced: lane l on word l, each warp on its own line.
        // random: the per-thread generator seed.
        unsigned pos = access == ACC_COALESCED
            ? (unsigned)(32 * (rank * warps + warp) + lane)
            : mix((unsigned)(threadIdx.x + 1024 * rank + 7 * seed));
        unsigned sink = 0;
        for (int w = 0; w < warmup; w++) sink ^= read_loop(buf, access, pos, steps);

        const long long c0 = clock64();
        const unsigned long long g0 = globaltimer();
        sink ^= read_loop(buf, access, pos, steps);
        Result r;
        r.cycles = clock64() - c0;
        r.ns = globaltimer() - g0;
        r.sink = sink;   // stored, so the loads cannot be optimized away
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
    a.access = ACC_COALESCED;   // Args' default of 0 means "unset" for this program
    parse_args(argc, argv, &a, "dsmem_many_threads",
               "  --cluster-size N  blocks in the cluster, 2..12          (default 12)\n"
               "  --block-size N    threads per block, multiple of 32     (default 128)\n"
               "  --pattern P       broadcast | ring | random             (default broadcast)\n"
               "  --access A        random | coalesced                    (default coalesced)\n");
    const int cs = a.cluster_size;
    if (cs < 2 || cs > 12) {
        fprintf(stderr, "dsmem_many_threads: --cluster-size must be 2..12 on GB10\n");
        return 1;
    }
    if (a.block_size < 32 || a.block_size > 1024 || a.block_size % 32) {
        fprintf(stderr, "dsmem_many_threads: --block-size must be a multiple of 32, 32..1024\n");
        return 1;
    }
    if (a.access != ACC_RANDOM && a.access != ACC_COALESCED) {
        fprintf(stderr, "dsmem_many_threads: --access must be random or coalesced\n");
        return 1;
    }
    const char* access_name = (a.access == ACC_COALESCED) ? "coalesced" : "random";
    const int warps = a.block_size / 32;
    // broadcast keeps rank 0 passive; ring and random make every rank a reader.
    const int readers = (a.pattern == PAT_BROADCAST) ? cs - 1 : cs;
    const int rows = cs * warps;           // rank 0's slots stay empty in broadcast

    char extra[200];
    snprintf(extra, sizeof extra, "# pattern: %s\n# access: %s\n# cluster_size: %d\n# readers: %d\n",
             PATTERN_NAME[a.pattern], access_name, cs, readers);
    print_header(argc, argv, a, SBUF, extra);

    Result* out;     CUDA_CHECK(cudaMallocManaged(&out, rows * sizeof(Result)));
    unsigned* smids; CUDA_CHECK(cudaMallocManaged(&smids, (cs + rows) * sizeof(unsigned)));

    if (cs > 8) allow_big_clusters((const void*)loaded);

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
    snprintf(name, sizeof name, "dsmem_%s_%s", PATTERN_NAME[a.pattern], access_name);
    const double total_loads = (double)readers * a.block_size * a.steps;

    std::vector<double> mean_cpl, gbps;   // one entry per repetition
    for (int rep = 0; rep < a.reps; rep++) {
        for (int i = 0; i < rows; i++) out[i].cycles = 0;
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, loaded, a.pattern, a.access, a.seed, a.warmup,
                                      a.steps, out, smids));
        CUDA_CHECK(cudaDeviceSynchronize());

        double sum_cpl = 0, max_ns = 0;
        int n = 0;
        for (int i = 0; i < rows; i++) {
            if (out[i].cycles == 0) continue;       // rank 0 in broadcast: no data
            int rank = i / warps, warp = i % warps;
            int target = (int)smids[cs + i];
            print_row(name, cs, (target - rank + cs) % cs, 1, a.block_size, a.steps,
                      0, 0, a.seed, rep, out[i], a.block_size, warp,
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

    fprintf(stderr, "%s: cluster %d, %d readers x %d threads\n", name, cs, readers, a.block_size);
    print_stats_line("mean cy/load", mean_cpl);
    print_stats_line("GB/s total", gbps);
    {
        std::vector<double> g = gbps; std::sort(g.begin(), g.end());
        const double thr = g[g.size() / 2];
        // Patterns differ in reader count (broadcast has one fewer), so the
        // per-reader rate is what makes them comparable at a glance.
        fprintf(stderr, "  per reader   : %.3f GB/s   (%d readers)\n", thr / readers, readers);
    }

    cudaFree(out); cudaFree(smids);
    return 0;
}
