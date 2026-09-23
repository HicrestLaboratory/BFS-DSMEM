// dsmem_bandwidth.cu — how many BYTES per second can DSMEM actually move?
//
// latency-many-threads.cu found two ceilings, 0.71 GB/s into one owner and
// 4.45 GB/s across the GPC, but with the least favourable traffic there is:
// random 4-byte words, each thread waiting for one load before issuing the
// next. Such a path is limited by REQUESTS, not bytes, so its GB/s says little
// about what a kernel moving real data would get. This program changes the
// three things that decide how many bytes a request carries and how many
// requests are in flight, one at a time:
//
//   --width W       bytes each lane loads per instruction: 4, 8 or 16
//                   (unsigned, uint2, uint4 -> LD.32 / LD.64 / LD.128).
//                   If the cost is per request, 16 B moves ~4x the bytes of 4 B.
//
//   --access A      what the lanes of a warp read, and whether they wait:
//     chase         dependent random walk, idx = buf[idx]: ONE load in flight
//                   per thread. W = 4 reproduces latency-many-threads.cu.
//     random        independent random addresses, --ilp loads in flight per
//                   thread. Removes the dependency, keeps the scatter.
//     coalesced     lane l reads element base+l: one instruction covers
//                   32*W contiguous bytes (128/256/512 B), --ilp in flight.
//                   Removes the scatter too: the best case for the fabric.
//
//   --ilp N         independent loads each thread issues before it waits for
//                   the first: 1, 2, 4, 8 or 16. Ignored by chase.
//
//   --pattern P     hotspot | ring | random, exactly as in latency-many-threads.cu:
//                   one owner for everybody, one owner each, or a per-warp draw.
//
// The ladder chase -> random -> coalesced, each at W = 4, 8, 16, tells apart
// "each request costs a fixed time" (GB/s scales with W), "the reader cannot
// keep enough requests in flight" (random beats chase) and "the fabric is
// charged per address, not per instruction" (coalesced beats random).
//
// The buffer is the same 16 KiB of shared memory in every configuration, so a
// wider element means fewer of them: 4096 x 4 B, 2048 x 8 B, 1024 x 16 B.
//
// Every warp times itself and is emitted as one CSV row, with its start and
// end on the wall clock relative to the first warp to start. The aggregate
// throughput of a repetition is then  total bytes / (last end - first start),
// computable from the rows alone.
//
// Build: make dsmem_bandwidth
// Run:   ./dsmem_bandwidth --pattern ring --access coalesced --width 16 --ilp 8 --reps 11

#include <cooperative_groups.h>

#include <set>

#include "../common.cuh"

namespace cg = cooperative_groups;

enum { PAT_HOTSPOT = 0, PAT_RING = 1, PAT_RANDOM = 2 };
enum { ACC_CHASE = 0, ACC_RANDOM = 1, ACC_COALESCED = 2 };
static const char* PATTERN_NAME[] = {"hotspot", "ring", "random"};
static const char* ACCESS_NAME[] = {"chase", "random", "coalesced"};

// One warp's measurement. t0/t1 are absolute %globaltimer readings, so the
// host can line the warps up on one clock.
struct Span {
    long long cycles;
    unsigned long long t0, t1;
    unsigned sink;
};

__device__ __forceinline__ unsigned mix(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Consume EVERY word of the element. If only .x were used, the compiler would
// narrow the 16-byte load to a 4-byte one and the width would not be tested.
// The host writes 0 into all words but the first, so for a chase the fold is
// exactly the successor index.
__device__ __forceinline__ unsigned fold(unsigned v) { return v; }
__device__ __forceinline__ unsigned fold(uint2 v) { return v.x ^ v.y; }
__device__ __forceinline__ unsigned fold(uint4 v) { return v.x ^ v.y ^ v.z ^ v.w; }

// `steps` loads by one thread, in the chosen style. `state` carries the chase
// position (or the random generator) from the warm-up into the timed pass.
template <class V, int ILP>
__device__ __forceinline__ unsigned read_loop(const V* buf, unsigned mask, int access,
                                              unsigned& state, int steps) {
    if (access == ACC_CHASE) {
        unsigned idx = state;
        for (int i = 0; i < steps; i++) idx = fold(buf[idx]);
        state = idx;
        return idx;
    }
    // ILP separate accumulators, so load k of one iteration never waits for
    // load k-1. The outer loop is kept rolled: each iteration issues ILP
    // loads, then stalls on the first xor, which caps the loads in flight at
    // ILP per thread.
    unsigned acc[ILP];
#pragma unroll
    for (int k = 0; k < ILP; k++) acc[k] = 0;
    if (access == ACC_RANDOM) {
#pragma unroll 1
        for (int i = 0; i < steps; i += ILP) {
#pragma unroll
            for (int k = 0; k < ILP; k++) {
                // LCG: two ALU ops, no memory. The low bits of an LCG are
                // weak, so the index comes from bits 16 and up.
                state = state * 1664525u + 1013904223u;
                acc[k] ^= fold(buf[(state >> 16) & mask]);
            }
        }
    } else {  // ACC_COALESCED: state is this lane's position, lane-contiguous
#pragma unroll 1
        for (int i = 0; i < steps; i += ILP) {
#pragma unroll
            for (int k = 0; k < ILP; k++) acc[k] ^= fold(buf[(state + 32 * k) & mask]);
            state += 32 * ILP;
        }
    }
    unsigned s = 0;
#pragma unroll
    for (int k = 0; k < ILP; k++) s ^= acc[k];
    return s;
}

template <class V, int ILP>
__global__ void bandwidth(const unsigned* __restrict__ init, int pattern, int access,
                          int chasers, int seed, int warmup, int steps, Span* out,
                          unsigned* smids) {
    // 16 KiB, aligned for the 16-byte loads.
    __shared__ __align__(16) unsigned sbuf[SBUF];
    cg::cluster_group cluster = cg::this_cluster();
    const int cs = (int)cluster.num_blocks();
    const int rank = (int)cluster.block_rank();
    const int warps = blockDim.x / 32;
    const unsigned n = SBUF * sizeof(unsigned) / sizeof(V);   // elements of width W
    const unsigned mask = n - 1;                               // n is a power of two

    for (int i = threadIdx.x; i < SBUF; i += blockDim.x) sbuf[i] = init[i];
    __syncthreads();
    if (threadIdx.x == 0) smids[rank] = smid();
    cluster.sync();  // every buffer filled before anyone reads a peer's

    const int lane = threadIdx.x & 31, warp = (int)threadIdx.x >> 5;

    int target = -1;                                // -1: this block does not read
    if (pattern == PAT_HOTSPOT) {
        if (rank != 0) target = 0;
    } else if (pattern == PAT_RING) {
        target = (rank + 1) % cs;
    } else {
        target = (int)(mix((unsigned)(rank * 977 + warp * 31 + seed)) % (unsigned)(cs - 1));
        if (target >= rank) target++;
    }

    if (target >= 0 && (int)threadIdx.x < chasers) {
        // The lanes that took this branch: all 32, or fewer when --chasers
        // is not a multiple of 32. A full-mask __syncwarp would wait forever
        // for the lanes that did not.
        const unsigned lanes = __activemask();
        const V* buf = (const V*)cluster.map_shared_rank(sbuf, target);
        // chase: a random entry point on the cycle. random: a per-thread
        // generator seed. coalesced: lane-contiguous, each warp at its own
        // offset so the warps of a block do not all start on the same line.
        unsigned state;
        if (access == ACC_COALESCED)
            state = ((unsigned)(rank * warps + warp) * 32u * 4u + lane) & mask;
        else if (access == ACC_CHASE)
            state = mix((unsigned)(threadIdx.x + 1024 * rank + 7 * seed)) & mask;
        else
            state = mix((unsigned)(threadIdx.x + 1024 * rank + 7919 * seed));

        unsigned sink = 0;
        for (int w = 0; w < warmup; w++) sink ^= read_loop<V, ILP>(buf, mask, access, state, steps);

        __syncwarp(lanes);
        long long c0 = clock64();
        unsigned long long g0 = globaltimer();
        sink ^= read_loop<V, ILP>(buf, mask, access, state, steps);
        __syncwarp(lanes);   // the warp's time ends when its LAST lane is done
        long long c1 = clock64();
        unsigned long long g1 = globaltimer();

        if (lane == 0) {
            Span& s = out[rank * warps + warp];
            s.cycles = c1 - c0;
            s.t0 = g0;
            s.t1 = g1;
            s.sink = sink;
            smids[cs + rank * warps + warp] = (unsigned)target;
        }
    }

    // Shared memory is released when its block exits: keep every block
    // alive until the last reader is done with it.
    cluster.sync();
}

using Kernel = void (*)(const unsigned*, int, int, int, int, int, int, Span*, unsigned*);

template <class V>
Kernel pick_ilp(int ilp) {
    switch (ilp) {
        case 1:  return bandwidth<V, 1>;
        case 2:  return bandwidth<V, 2>;
        case 4:  return bandwidth<V, 4>;
        case 8:  return bandwidth<V, 8>;
        case 16: return bandwidth<V, 16>;
    }
    return nullptr;
}

Kernel pick(int width, int ilp) {
    switch (width) {
        case 4:  return pick_ilp<unsigned>(ilp);
        case 8:  return pick_ilp<uint2>(ilp);
        case 16: return pick_ilp<uint4>(ilp);
    }
    return nullptr;
}

int main(int argc, char** argv) {
    Args a;
    a.cluster_size = 12;
    parse_args(argc, argv, &a, "dsmem_bandwidth",
               "  --cluster-size N  blocks in the cluster, 2..12          (default 12)\n"
               "  --block-size N    threads per block, multiple of 32     (default 128)\n"
               "  --chasers N       reading threads per block; 0 = all    (default 0)\n"
               "  --pattern P       hotspot | ring | random               (default hotspot)\n"
               "  --access A        chase | random | coalesced            (default chase)\n"
               "  --width W         bytes per lane per load: 4, 8, 16     (default 4)\n"
               "  --ilp N           loads in flight per thread: 1,2,4,8,16 (default 1;\n"
               "                    chase is always 1)\n");
    const int cs = a.cluster_size;
    if (cs < 2 || cs > 12) {
        fprintf(stderr, "dsmem_bandwidth: --cluster-size must be 2..12 on GB10\n");
        return 1;
    }
    if (a.block_size < 32 || a.block_size > 1024 || a.block_size % 32) {
        fprintf(stderr, "dsmem_bandwidth: --block-size must be a multiple of 32, 32..1024\n");
        return 1;
    }
    if (a.chasers < 0 || a.chasers > a.block_size) {
        fprintf(stderr, "dsmem_bandwidth: --chasers must be 0..--block-size\n");
        return 1;
    }
    if (a.access == ACC_CHASE) a.ilp = 1;             // a dependent walk has one in flight
    Kernel kernel = pick(a.width, a.ilp);
    if (!kernel) {
        fprintf(stderr, "dsmem_bandwidth: --width must be 4, 8 or 16 and --ilp 1, 2, 4, 8 or 16\n");
        return 1;
    }
    if (a.steps % a.ilp) {
        fprintf(stderr, "dsmem_bandwidth: --steps must be a multiple of --ilp\n");
        return 1;
    }
    if (a.stride_bytes) {
        fprintf(stderr, "dsmem_bandwidth: --stride is not used here; see --access\n");
        return 1;
    }
    if (a.chasers == 0) a.chasers = a.block_size;
    const int warps = a.block_size / 32;
    const int readers = (a.pattern == PAT_HOTSPOT) ? cs - 1 : cs;
    const int rows = cs * warps;
    const int words = a.width / (int)sizeof(unsigned);       // 32-bit words per element
    const size_t n = SBUF / words;                            // elements in the buffer

    char extra[400];
    snprintf(extra, sizeof extra,
             "# pattern: %s\n# access: %s\n# width_bytes: %d\n# ilp: %d\n# cluster_size: %d\n"
             "# readers: %d\n# chasers_per_block: %d\n# elements: %zu\n",
             PATTERN_NAME[a.pattern], ACCESS_NAME[a.access], a.width, a.ilp, cs, readers,
             a.chasers, n);
    print_header(argc, argv, a, SBUF, extra,
                 "benchmark,cluster_size,block_size,chasers,pattern,access,width_bytes,ilp,"
                 "steps,seed,rep,warp,reader,target,smid_reader,smid_target,cycles,ns,"
                 "start_ns,end_ns,bytes,gbps\n");

    // Element i's first word is its successor on one random cycle, the other
    // words are 0 (see fold()). Only chase follows the links; the other two
    // access styles read the same buffer for its bytes.
    std::mt19937 rng(a.seed);
    std::vector<unsigned> perm(n), h(SBUF, 0u);
    make_cycle(perm, rng);
    for (size_t i = 0; i < n; i++) h[i * words] = perm[i];
    unsigned* dinit;
    CUDA_CHECK(cudaMalloc(&dinit, SBUF * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(dinit, h.data(), SBUF * sizeof(unsigned), cudaMemcpyHostToDevice));

    Span* out;       CUDA_CHECK(cudaMallocManaged(&out, rows * sizeof(Span)));
    unsigned* smids; CUDA_CHECK(cudaMallocManaged(&smids, (cs + rows) * sizeof(unsigned)));

    if (cs > 8) allow_big_clusters((const void*)kernel);

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

    char name[64];
    snprintf(name, sizeof name, "dsmem_bw_%s_%s", PATTERN_NAME[a.pattern], ACCESS_NAME[a.access]);

    std::vector<double> gbps, per_owner, cyc_per_load, ghz_v;
    int owners = 0;
    for (int rep = 0; rep < a.reps; rep++) {
        for (int i = 0; i < rows; i++) out[i].cycles = 0;
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, (const unsigned*)dinit, a.pattern, a.access,
                                      a.chasers, a.seed, a.warmup, a.steps, out, smids));
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned long long first = ~0ull, last = 0;
        for (int i = 0; i < rows; i++) {
            if (out[i].cycles == 0) continue;
            first = std::min(first, out[i].t0);
            last = std::max(last, out[i].t1);
        }
        double bytes_total = 0, sum_cpl = 0, sum_cycles = 0, sum_ns = 0;
        int n_warps = 0;
        std::set<int> targets;
        for (int i = 0; i < rows; i++) {
            if (out[i].cycles == 0) continue;           // rank 0 in hotspot: no data
            const int rank = i / warps, warp = i % warps;
            const int target = (int)smids[cs + i];
            const int lanes = std::min(32, a.chasers - 32 * warp);
            const double bytes = (double)lanes * a.steps * a.width;
            const unsigned long long ns = out[i].t1 - out[i].t0;
            printf("%s,%d,%d,%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%lld,%llu,%llu,%llu,%.0f,%.4f\n",
                   name, cs, a.block_size, a.chasers, PATTERN_NAME[a.pattern],
                   ACCESS_NAME[a.access], a.width, a.ilp, a.steps, a.seed, rep, warp, rank,
                   target, (int)smids[rank], (int)smids[target], out[i].cycles, ns,
                   out[i].t0 - first, out[i].t1 - first, bytes, bytes / ns);
            bytes_total += bytes;
            sum_cpl += (double)out[i].cycles / a.steps;
            sum_cycles += (double)out[i].cycles;
            sum_ns += (double)ns;
            targets.insert(target);
            n_warps++;
        }
        owners = (int)targets.size();
        gbps.push_back(bytes_total / (double)(last - first));   // bytes/ns = GB/s
        per_owner.push_back(gbps.back() / owners);
        cyc_per_load.push_back(sum_cpl / n_warps);
        ghz_v.push_back(sum_cycles / sum_ns);
    }

    fprintf(stderr, "%s: cluster %d, %d readers x %d threads, %d B/lane, ilp %d, %d owners\n",
            name, cs, readers, a.chasers, a.width, a.ilp, owners);
    print_stats_line("GB/s total", gbps);
    print_stats_line("GB/s per owner", per_owner);
    print_stats_line("cy per load", cyc_per_load);
    {
        std::vector<double> g = per_owner; std::sort(g.begin(), g.end());
        std::vector<double> c = ghz_v;     std::sort(c.begin(), c.end());
        const double bw = g[g.size() / 2], clk = c[c.size() / 2];
        // The two ways to read the same number. Bytes per cycle is what a
        // kernel designer budgets; lane-loads per cycle is what the fabric
        // actually serves if it is charged per request. If W = 16 gives 4x
        // the bytes/cycle of W = 4 at the same lane-loads/cycle, the cost is
        // per request and wide loads are free bandwidth.
        fprintf(stderr, "  per owner    : %.2f B/cycle = %.4f lane-loads/cycle "
                        "(one every %.1f cy) at %.2f GHz\n",
                bw / clk, bw / clk / a.width, clk * a.width / bw, clk);
    }

    cudaFree(dinit); cudaFree(out); cudaFree(smids);
    return 0;
}
