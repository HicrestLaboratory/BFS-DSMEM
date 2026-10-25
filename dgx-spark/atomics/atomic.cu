// atomic.cu — atomicOr, atomicAdd and atomicCAS: throughput under contention,
// and latency, on each kind of memory.
//
// THE QUESTION
//   A BFS uses atomics to mark vertices visited (Or, CAS) and to count or
//   claim slots (Add). Where the word lives decides the price: my own shared
//   memory, another block's (DSMEM), or global memory (executed in L2). And
//   it depends on how many threads hit the same word at the same time.
//
// THROUGHPUT (--mode throughput)
//   One cluster of 12 blocks. Ranks 1..8 are the WORKERS: 8 blocks x 512
//   threads = 4096 threads, each doing --steps atomics on one word, with the
//   result never used. --contention x makes x threads share each word:
//   threads 0..x-1 on one word, x..2x-1 on another, ... (x = 1: nobody
//   collides; x = 4096: everybody on the same word). The groups use different
//   words of a 16 KiB buffer (SBUF = 4096 words), so two groups never meet,
//   laid out by --layout:
//     consecutive   group g on word g: a warp's lanes on neighbouring words,
//                   ONE cache line (the best case, like coalesced accesses)
//     scattered     group g on a word spread by a permutation: a warp's lanes
//                   on 32 different lines (vertices spread over the owner's
//                   range, as in a BFS). Atomics of a warp on one line travel
//                   as one request, so this is several times slower.
//   Where the buffer is (--memory):
//     smem                  each worker on its OWN shared memory (so at most
//                           one block, 512 threads, can share a word)
//     dsmem-rank0           every worker on rank 0's shared memory: rank 0
//                           is the hotspot (it sits on SM 0, the worst TPC)
//     dsmem-rank10          the same, with rank 10 as the hotspot (SM 40)
//     dsmem-rank10-spread   only the first x threads hit one word of rank
//                           10; every other thread updates its own word in
//                           the NEXT worker's shared memory (rank 1 -> 2, ...,
//                           8 -> 1), so the rest of the traffic is spread
//     global                every worker on one 16 KiB buffer in global memory
//   Result: atomics per second, all workers together, from the first
//   worker's start to the last one's end.
//
// THE OPERATIONS (--op)
//   or    atomicOr(word, 1 << (thread % 32))
//   add   atomicAdd(word, 1)
//   cas   atomicCAS(word, 0, my id + 1): one attempt to claim the word, as a
//         BFS claims an unvisited vertex; after the first success the others
//         fail, and every attempt is counted.
//   In shared memory, an atomicAdd(word, 1) whose result is unused compiles
//   to ATOMS.POPC.INC: one instruction adds, to each word, how many lanes of
//   the warp hit it, so lanes of a warp colliding on a word cost one update.
//
// LATENCY (--mode latency)
//   One thread of rank 0 runs a chain of --steps atomics, each on the word
//   the previous one returned (the word holds 0 and stays 0:
//   atomicOr(w, 0), atomicAdd(w, 0), atomicCAS(w, 0, 0)), so time / steps is
//   one atomic. --memory smem | dsmem | global; dsmem works on rank
//   --distance's shared memory.
//
// CORRECTNESS
//   After every throughput run the host replays which thread hit which word
//   and checks every word of every buffer: add = the number of +1s aimed at
//   it, or = the OR of the bits aimed at it, cas = claimed (non-zero, by one
//   of the threads aimed at it) exactly when somebody aimed at it.
//
// Build: make atomic
// Run:   build/atomic --mode throughput --memory dsmem-rank10 --op add --contention 64

#include <cooperative_groups.h>

#include "../common.cuh"   // CUDA_CHECK, Args, SBUF, globaltimer(), print_header, ...

namespace cg = cooperative_groups;

constexpr int CLUSTER_SIZE = 12;
constexpr int FIRST_WORKER = 1, WORKERS = 8;   // ranks 1..8 do the atomics

enum { OR, ADD, CAS };
static const char* OP_NAME[] = {"or", "add", "cas"};
// Throughput targets, then the latency ones.
enum { SMEM, DSMEM_RANK0, DSMEM_RANK10, DSMEM_RANK10_SPREAD, GLOBAL };
static const char* TARGET_NAME[] = {"smem", "dsmem-rank0", "dsmem-rank10",
                                    "dsmem-rank10-spread", "global"};

enum { CONSECUTIVE, SCATTERED };   // --layout
static const char* LAYOUT_NAME[] = {"consecutive", "scattered"};

// The word of group g. consecutive: word g. scattered: a permutation of
// 0..SBUF-1 (an odd multiplier is a bijection modulo a power of two), so
// different groups still never share a word, but neighbouring groups land
// far apart.
__host__ __device__ __forceinline__ unsigned word_of(unsigned group, int layout) {
    return layout == SCATTERED ? (group * 2654435761u) & (SBUF - 1) : group & (SBUF - 1);
}

// Where worker thread `w` (0..4095; for smem its thread index) of rank `rank`
// sends its atomics: which buffer (a rank's shared memory, or -1 = the global
// buffer) and which word. Used by the kernel and by the host check.
struct Target { int rank; unsigned word; };
__host__ __device__ __forceinline__ Target target_of(int target, int rank, int w, int tid,
                                                     int contention, int layout) {
    switch (target) {
        case SMEM:         return {rank, word_of(tid / contention, layout)};
        case DSMEM_RANK0:  return {0, word_of(w / contention, layout)};
        case DSMEM_RANK10: return {10, word_of(w / contention, layout)};
        case DSMEM_RANK10_SPREAD:
            if (w < contention) return {10, word_of(0, layout)};
            return {rank == FIRST_WORKER + WORKERS - 1 ? FIRST_WORKER : rank + 1,
                    word_of(w, layout)};
        default:           return {-1, word_of(w / contention, layout)};
    }
}

__device__ __forceinline__ void one_atomic(int op, unsigned* p, unsigned id) {
    if (op == OR)       atomicOr(p, 1u << (id & 31));
    else if (op == ADD) atomicAdd(p, 1u);
    else                atomicCAS(p, 0u, id + 1);
}

// What one block measured.
struct BlockTime {
    long long cycles;
    unsigned long long start_ns, end_ns;   // %globaltimer: one clock for the whole GPU
};

template <int TARGET, int OP>
__global__ void atomic_throughput(unsigned* global_buf, int contention, int layout, int steps,
                                  int warmup, BlockTime* out, unsigned* dump) {
    __shared__ unsigned buf[SBUF];
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = (int)cluster.block_rank();
    for (int i = threadIdx.x; i < SBUF; i += blockDim.x) buf[i] = 0;
    cluster.sync();   // every buffer is zero before anyone uses it

    const bool worker = rank >= FIRST_WORKER && rank < FIRST_WORKER + WORKERS;
    const int w = (rank - FIRST_WORKER) * blockDim.x + threadIdx.x;   // my number among the workers
    const Target t = target_of(TARGET, rank, w, threadIdx.x, contention, layout);
    unsigned* p;
    if (TARGET == SMEM)        p = &buf[t.word];   // a shared-memory pointer: ATOMS
    else if (TARGET == GLOBAL) p = &global_buf[t.word];
    else                       p = cluster.map_shared_rank(buf, t.rank) + t.word;

    if (worker)
        for (int s = 0; s < warmup * steps; s++) one_atomic(OP, p, w);
    cluster.sync();   // the workers start together
    const long long c0 = clock64();
    const unsigned long long t0 = globaltimer();
    if (worker)
        for (int s = 0; s < steps; s++) one_atomic(OP, p, w);
    __syncthreads();   // the block is done when its slowest thread is
    if (threadIdx.x == 0) {
        out[rank].cycles = clock64() - c0;
        out[rank].start_ns = t0;
        out[rank].end_ns = globaltimer();
    }
    cluster.sync();   // every atomic on every buffer is done
    for (int i = threadIdx.x; i < SBUF; i += blockDim.x) dump[rank * SBUF + i] = buf[i];
}

// Latency: thread 0 of rank 0 runs a chain on word 0 of its own shared memory
// (MEMORY 0), of rank `rank_target`'s (1, DSMEM) or of the global buffer (2).
// MEMORY is a template parameter so that for smem the compiler sees a
// shared-memory pointer and emits ATOMS.
template <int MEMORY, int OP>
__global__ void atomic_latency(unsigned* global_buf, int rank_target, int steps, int warmup,
                               BlockTime* out) {
    __shared__ unsigned buf[SBUF];
    cg::cluster_group cluster = cg::this_cluster();
    if (threadIdx.x == 0) buf[0] = 0;
    cluster.sync();
    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        unsigned* c;
        if (MEMORY == 0)      c = buf;
        else if (MEMORY == 1) c = cluster.map_shared_rank(buf, rank_target);
        else                  c = global_buf;
        unsigned i = 0;   // word 0 holds 0, and every atomic leaves it 0
        long long c0 = 0;
        unsigned long long t0 = 0;
        for (int s = 0; s < (warmup + 1) * steps; s++) {
            if (s == warmup * steps) { c0 = clock64(); t0 = globaltimer(); }
            if (OP == OR)       i = atomicOr(&c[i], 0u);
            else if (OP == ADD) i = atomicAdd(&c[i], 0u);
            else                i = atomicCAS(&c[i], 0u, 0u);
        }
        out[0].cycles = clock64() - c0;
        out[0].start_ns = t0;
        out[0].end_ns = globaltimer();
    }
    cluster.sync();   // my shared memory must outlive the last remote atomic
}

// ------------------------------------------------------------ host

// The host replay of one throughput run: what every word must hold.
static bool check(int target, int op, int contention, int layout, int block_size,
                  int atomics_per_thread,
                  const std::vector<unsigned>& dump, const std::vector<unsigned>& global) {
    // expected value per (buffer, word); buffer CLUSTER_SIZE = the global one
    std::vector<unsigned long long> add((CLUSTER_SIZE + 1) * SBUF, 0);
    std::vector<unsigned> orv((CLUSTER_SIZE + 1) * SBUF, 0);
    std::vector<std::vector<unsigned>> claimers((CLUSTER_SIZE + 1) * SBUF);
    for (int r = FIRST_WORKER; r < FIRST_WORKER + WORKERS; r++)
        for (int tid = 0; tid < block_size; tid++) {
            const int w = (r - FIRST_WORKER) * block_size + tid;
            const Target t = target_of(target, r, w, tid, contention, layout);
            const int k = (t.rank < 0 ? CLUSTER_SIZE : t.rank) * SBUF + t.word;
            add[k] += atomics_per_thread;
            orv[k] |= 1u << (w & 31);
            claimers[k].push_back(w + 1);
        }
    for (int k = 0; k < (CLUSTER_SIZE + 1) * SBUF; k++) {
        const unsigned got = k < CLUSTER_SIZE * SBUF ? dump[k] : global[k - CLUSTER_SIZE * SBUF];
        bool ok;
        if (op == ADD)     ok = got == (unsigned)add[k];
        else if (op == OR) ok = got == orv[k];
        else ok = claimers[k].empty() ? got == 0
                : std::find(claimers[k].begin(), claimers[k].end(), got) != claimers[k].end();
        if (!ok) {
            fprintf(stderr, "atomic: WRONG word %d of buffer %d: %u\n", k % SBUF, k / SBUF, got);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    Args a;
    a.block_size = 512;
    a.steps = 1024;
    parse_args(argc, argv, &a, "atomic",
               "  --mode M          throughput | latency                  (default latency)\n"
               "  --op OP           or | add | cas                        (default add)\n"
               "  --memory M        throughput: smem | dsmem-rank0 | dsmem-rank10 |\n"
               "                    dsmem-rank10-spread | global;  latency: smem | dsmem | global\n"
               "  --contention x    throughput: threads per word, 1..4096 (default 1)\n"
               "  --layout L        throughput: consecutive | scattered   (default consecutive)\n"
               "  --distance D      dsmem latency: target rank            (default 1)\n"
               "  --steps N         atomics per thread                    (default 1024)\n"
               "  --block-size N    throughput: threads per worker block  (default 512)\n");

    // ---- check the arguments
    const bool throughput = (a.mode == 1);
    int op = -1;
    for (int o = 0; o < 3; o++)
        if (strcmp(a.op ? a.op : "add", OP_NAME[o]) == 0) op = o;
    if (op < 0) { fprintf(stderr, "atomic: --op must be or, add or cas\n"); return 1; }
    const char* memory_name = a.memory ? a.memory : "smem";
    int target = -1;   // throughput target, or latency memory 0 smem / 1 dsmem / 2 global
    if (throughput) {
        for (int t = 0; t < 5; t++)
            if (strcmp(memory_name, TARGET_NAME[t]) == 0) target = t;
    } else {
        const char* lat[] = {"smem", "dsmem", "global"};
        for (int t = 0; t < 3; t++)
            if (strcmp(memory_name, lat[t]) == 0) target = t;
    }
    if (target < 0) {
        fprintf(stderr, "atomic: --memory must be smem, dsmem-rank0, dsmem-rank10, "
                        "dsmem-rank10-spread or global (throughput); smem, dsmem or global (latency)\n");
        return 1;
    }
    int layout = -1;
    for (int l = 0; l < 2; l++)
        if (strcmp(a.layout ? a.layout : "consecutive", LAYOUT_NAME[l]) == 0) layout = l;
    if (layout < 0) { fprintf(stderr, "atomic: --layout must be consecutive or scattered\n"); return 1; }
    const int threads = WORKERS * a.block_size;   // all the workers' threads
    if (throughput) {
        if (a.block_size < 32 || a.block_size > 1024 || a.block_size % 32 ||
            threads > SBUF) {
            fprintf(stderr, "atomic: --block-size must be a multiple of 32 with %d x it <= %d "
                            "(every thread needs its own word at --contention 1)\n", WORKERS, SBUF);
            return 1;
        }
        const int most = target == SMEM ? a.block_size : threads;   // who can share a word
        if (a.contention < 1 || a.contention > most) {
            fprintf(stderr, "atomic: --contention must be 1..%d for --memory %s\n", most,
                    memory_name);
            return 1;
        }
    } else if (target == 1 && (a.distance < 1 || a.distance >= CLUSTER_SIZE)) {
        fprintf(stderr, "atomic: --distance must be 1..%d\n", CLUSTER_SIZE - 1);
        return 1;
    }

    // ---- the kernel for this target / op
    using ThroughputKernel = void (*)(unsigned*, int, int, int, int, BlockTime*, unsigned*);
    static const ThroughputKernel throughput_kernels[5][3] = {
        {atomic_throughput<SMEM, OR>, atomic_throughput<SMEM, ADD>, atomic_throughput<SMEM, CAS>},
        {atomic_throughput<DSMEM_RANK0, OR>, atomic_throughput<DSMEM_RANK0, ADD>,
         atomic_throughput<DSMEM_RANK0, CAS>},
        {atomic_throughput<DSMEM_RANK10, OR>, atomic_throughput<DSMEM_RANK10, ADD>,
         atomic_throughput<DSMEM_RANK10, CAS>},
        {atomic_throughput<DSMEM_RANK10_SPREAD, OR>, atomic_throughput<DSMEM_RANK10_SPREAD, ADD>,
         atomic_throughput<DSMEM_RANK10_SPREAD, CAS>},
        {atomic_throughput<GLOBAL, OR>, atomic_throughput<GLOBAL, ADD>, atomic_throughput<GLOBAL, CAS>},
    };
    using LatencyKernel = void (*)(unsigned*, int, int, int, BlockTime*);
    static const LatencyKernel latency_kernels[3][3] = {
        {atomic_latency<0, OR>, atomic_latency<0, ADD>, atomic_latency<0, CAS>},
        {atomic_latency<1, OR>, atomic_latency<1, ADD>, atomic_latency<1, CAS>},
        {atomic_latency<2, OR>, atomic_latency<2, ADD>, atomic_latency<2, CAS>},
    };
    const void* kernel = throughput ? (const void*)throughput_kernels[target][op]
                                    : (const void*)latency_kernels[target][op];

    // ---- launch configuration: one cluster of 12 blocks
    allow_big_clusters(kernel);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CLUSTER_SIZE);
    cfg.blockDim = dim3(throughput ? a.block_size : 32);   // latency: one warp is enough
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = CLUSTER_SIZE;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;

    const int contention = throughput ? a.contention : 1;
    const int workers = throughput ? WORKERS : 1;
    const double atomics = throughput ? (double)threads * a.steps : a.steps;
    char extra[300];
    snprintf(extra, sizeof extra,
             "# mode: %s\n# op: %s\n# memory: %s\n# contention: %d\n# layout: %s\n"
             "# workers: %d (ranks %d..%d)\n# buffer_words: %d\n",
             throughput ? "throughput" : "latency", OP_NAME[op], memory_name, contention,
             LAYOUT_NAME[layout], workers, FIRST_WORKER, FIRST_WORKER + workers - 1, SBUF);
    print_header(argc, argv, a, 0, extra,
                 "benchmark,memory,mode,op,contention,layout,distance,block_size,workers,steps,seed,rep,"
                 "atomics,ns,cycles,ns_per_atomic,cycles_per_atomic,gatomics_per_s\n");

    // ---- memory the kernel needs
    unsigned *global_buf, *dump;
    CUDA_CHECK(cudaMalloc(&global_buf, SBUF * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&dump, CLUSTER_SIZE * SBUF * sizeof(unsigned)));
    std::vector<unsigned> host_dump(CLUSTER_SIZE * SBUF), host_global(SBUF);
    BlockTime* out;
    CUDA_CHECK(cudaMallocManaged(&out, CLUSTER_SIZE * sizeof(BlockTime)));

    std::vector<double> per_atomic, rate;
    for (int rep = 0; rep < a.reps; rep++) {
        CUDA_CHECK(cudaMemset(global_buf, 0, SBUF * sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(out, 0, CLUSTER_SIZE * sizeof(BlockTime)));
        if (throughput)
            CUDA_CHECK(cudaLaunchKernelEx(&cfg, throughput_kernels[target][op], global_buf,
                                          contention, layout, a.steps, a.warmup, out, dump));
        else
            CUDA_CHECK(cudaLaunchKernelEx(&cfg, latency_kernels[target][op], global_buf,
                                          a.distance, a.steps, a.warmup, out));
        CUDA_CHECK(cudaDeviceSynchronize());

        if (throughput) {   // ---- correctness, then the time over the workers
            CUDA_CHECK(cudaMemcpy(host_dump.data(), dump, host_dump.size() * sizeof(unsigned),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(host_global.data(), global_buf, SBUF * sizeof(unsigned),
                                  cudaMemcpyDeviceToHost));
            if (!check(target, op, contention, layout, a.block_size, a.steps * (a.warmup + 1),
                       host_dump, host_global))
                return 1;
        }
        unsigned long long first = ~0ull, last = 0;
        const int from = throughput ? FIRST_WORKER : 0, to = throughput ? FIRST_WORKER + WORKERS : 1;
        for (int b = from; b < to; b++) {
            first = std::min(first, out[b].start_ns);
            last = std::max(last, out[b].end_ns);
        }
        const double ns = (double)(last - first);
        const double cycles_per = throughput ? 0.0 : (double)out[0].cycles / atomics;
        printf("atomic,%s,%s,%s,%d,%s,%d,%d,%d,%d,%d,%d,%.0f,%.0f,%lld,%.4f,%.2f,%.4f\n",
               memory_name, throughput ? "throughput" : "latency", OP_NAME[op], contention,
               throughput ? LAYOUT_NAME[layout] : "-",
               throughput ? 0 : a.distance, (int)cfg.blockDim.x, workers, a.steps, a.seed, rep,
               atomics, ns, out[from].cycles, ns / atomics, cycles_per, atomics / ns);
        per_atomic.push_back(throughput ? ns / atomics : cycles_per);
        rate.push_back(atomics / ns);
    }

    fprintf(stderr, "atomic %s %s on %s", throughput ? "throughput" : "latency", OP_NAME[op],
            memory_name);
    if (throughput)
        fprintf(stderr, ", %d threads per word, %s words, %d workers x %d threads, verified",
                contention, LAYOUT_NAME[layout], WORKERS, a.block_size);
    fprintf(stderr, "\n");
    print_stats_line(throughput ? "ns per atomic" : "cycles/atomic", per_atomic);
    print_stats_line("G atomics/s", rate);
    CUDA_CHECK(cudaFree(global_buf));
    CUDA_CHECK(cudaFree(dump));
    CUDA_CHECK(cudaFree(out));
    return 0;
}
