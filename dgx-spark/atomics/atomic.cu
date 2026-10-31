// atomic.cu — atomicOr, atomicAdd and atomicCAS: throughput under contention,
// and latency, on each kind of memory.
//
// THE QUESTION
//   A BFS uses atomics to mark vertices visited (Or, CAS) and to count or
//   claim slots (Add). Where the word lives decides the price: my own shared
//   memory, another block's (DSMEM), or global memory (executed in L2). And
//   it depends on how many threads hit the same word at the same time, and
//   on WHERE those threads are: in my block, or in other blocks.
//
// THROUGHPUT (--mode throughput)
//   One cluster of 12 blocks. Ranks 1..8 are the WORKERS: 8 blocks x 512
//   threads = 4096 threads, each doing --steps atomics on one word, with the
//   result never used. --contention x makes x threads share each word (x = 1:
//   nobody collides). There are always 4096 / x words in use, neighbouring
//   words of a 16 KiB buffer (SBUF = 4096 words), and a warp's lanes always
//   hit neighbouring words, ONE cache line. --scope says which threads share
//   a word:
//     intra   intra-block contention: the x threads of a word are neighbours
//             in ONE block (x <= 512), so up to 32 of them are lanes of the
//             same warp. In shared memory the hardware merges those
//             (ATOMS.POPC.INC: one update per word per warp); in DSMEM and
//             global memory it does not, they are applied one by one.
//     inter   inter-block contention: the x threads of a word are spread over
//             min(x, 8) different blocks, x / min(x, 8) neighbours from each
//             (x = 2: the same thread of blocks 0 and 1; x = 8: the same
//             thread of all 8 blocks; x = 4096: everybody). The collisions
//             come from different SMs; within a warp only x / min(x, 8)
//             lanes share a word.
//   Where the buffer is (--memory):
//     smem          each worker on its OWN shared memory (intra only: no other
//                   block can see it)
//     dsmem-rank0   every worker on rank 0's shared memory: rank 0 is the
//                   hotspot (it sits on SM 0, the worst TPC)
//     dsmem-rank10  the same, with rank 10 as the hotspot (SM 40)
//     global        every worker on one 16 KiB buffer in global memory
//   Result: atomics per second, all workers together, from the first
//   worker's start to the last one's end.
//
// THE OPERATIONS (--op)
//   or    atomicOr(word, 1 << (thread % 32))
//   add   atomicAdd(word, 1)
//   cas   atomicCAS(word, 0, my id + 1): one attempt to claim the word, as a
//         BFS claims an unvisited vertex; after the first success the others
//         fail, and every attempt is counted.
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
// Run:   build/atomic --mode throughput --memory dsmem-rank10 --op add --scope inter --contention 64

#include <cooperative_groups.h>

#include "../common.cuh"   // CUDA_CHECK, Args, SBUF, globaltimer(), print_header, ...

namespace cg = cooperative_groups;

constexpr int CLUSTER_SIZE = 12;
constexpr int FIRST_WORKER = 1, WORKERS = 8;   // ranks 1..8 do the atomics

enum { OR, ADD, CAS };
static const char* OP_NAME[] = {"or", "add", "cas"};
enum { SMEM, DSMEM_RANK0, DSMEM_RANK10, GLOBAL };   // throughput --memory
static const char* TARGET_NAME[] = {"smem", "dsmem-rank0", "dsmem-rank10", "global"};
enum { INTRA, INTER };                              // --scope
static const char* SCOPE_NAME[] = {"intra", "inter"};

// The word of thread `tid` of worker block `r` (0..7), with x threads per word.
// intra: the x threads of a word are neighbours in one block.
__host__ __device__ __forceinline__ unsigned word_intra(int r, int tid, int block_size, int x) {
    return (unsigned)(r * block_size + tid) / x;
}
// inter: the x threads of a word are spread over B = min(x, 8) blocks, k = x / B
// neighbouring threads from each. Blocks 0..B-1 use the first words, blocks
// B..2B-1 the next ones, ...
__host__ __device__ __forceinline__ unsigned word_inter(int r, int tid, int block_size, int x) {
    const int B = x < WORKERS ? x : WORKERS;
    const int k = x / B;
    return (unsigned)((r / B) * (block_size / k) + tid / k);
}

// ---- the operations: one function each, chosen at compile time (template
// argument), so the timed loop has no branch.
__device__ __forceinline__ void atomic_or(unsigned* p, unsigned id) { atomicOr(p, 1u << (id & 31)); }
__device__ __forceinline__ void atomic_add(unsigned* p, unsigned id) { atomicAdd(p, 1u); }
__device__ __forceinline__ void atomic_cas(unsigned* p, unsigned id) { atomicCAS(p, 0u, id + 1); }
using Atomic = void (*)(unsigned*, unsigned);

// Latency: the same operations, returning the word's old value (always 0).
__device__ __forceinline__ unsigned chain_or(unsigned* p) { return atomicOr(p, 0u); }
__device__ __forceinline__ unsigned chain_add(unsigned* p) { return atomicAdd(p, 0u); }
__device__ __forceinline__ unsigned chain_cas(unsigned* p) { return atomicCAS(p, 0u, 0u); }
using Chain = unsigned (*)(unsigned*);

// ---- the memories: where the buffer is. `buf` is my block's shared buffer;
// after inlining the compiler still sees it is shared memory and emits ATOMS.
__device__ __forceinline__ unsigned* own_smem(unsigned* buf, unsigned* global_buf,
                                              cg::cluster_group& cluster, int rank) {
    return buf;
}
__device__ __forceinline__ unsigned* remote_smem(unsigned* buf, unsigned* global_buf,
                                                 cg::cluster_group& cluster, int rank) {
    return cluster.map_shared_rank(buf, rank);
}
__device__ __forceinline__ unsigned* global_mem(unsigned* buf, unsigned* global_buf,
                                                cg::cluster_group& cluster, int rank) {
    return global_buf;
}
using Memory = unsigned* (*)(unsigned*, unsigned*, cg::cluster_group&, int);

// What one block measured.
struct BlockTime {
    long long cycles;
    unsigned long long start_ns, end_ns;   // %globaltimer: one clock for the whole GPU
};

// Everything but the choice of the word: zero the buffers, run the atomics
// on `word` of the buffer MEMORY gives (remote_smem: rank `target_rank`'s),
// time them, and dump every shared buffer for the host check.
template <Memory MEMORY, Atomic ATOMIC>
__device__ __forceinline__ void throughput(unsigned* global_buf, int target_rank, unsigned word,
                                           int steps, int warmup, BlockTime* out, unsigned* dump) {
    __shared__ unsigned buf[SBUF];
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = (int)cluster.block_rank();
    for (int i = threadIdx.x; i < SBUF; i += blockDim.x) buf[i] = 0;
    cluster.sync();   // every buffer is zero before anyone uses it

    const bool worker = rank >= FIRST_WORKER && rank < FIRST_WORKER + WORKERS;
    const unsigned id = (rank - FIRST_WORKER) * blockDim.x + threadIdx.x;   // my number among the workers
    unsigned* p = MEMORY(buf, global_buf, cluster, target_rank) + word;

    if (worker)
        for (int s = 0; s < warmup * steps; s++) ATOMIC(p, id);
    cluster.sync();   // the workers start together
    const long long c0 = clock64();
    const unsigned long long t0 = globaltimer();
    if (worker)
        for (int s = 0; s < steps; s++) ATOMIC(p, id);
    __syncthreads();   // the block is done when its slowest thread is
    if (threadIdx.x == 0) {
        out[rank].cycles = clock64() - c0;
        out[rank].start_ns = t0;
        out[rank].end_ns = globaltimer();
    }
    cluster.sync();   // every atomic on every buffer is done
    for (int i = threadIdx.x; i < SBUF; i += blockDim.x) dump[rank * SBUF + i] = buf[i];
}

// Each worker on its own shared memory: the word only depends on my thread.
template <Atomic ATOMIC>
__global__ void atomic_throughput_smem(unsigned* global_buf, int target_rank, int contention,
                                       int steps, int warmup, BlockTime* out, unsigned* dump) {
    const unsigned word = threadIdx.x / contention;
    throughput<own_smem, ATOMIC>(global_buf, target_rank, word, steps, warmup, out, dump);
}

template <Memory MEMORY, Atomic ATOMIC>
__global__ void atomic_throughput_intra(unsigned* global_buf, int target_rank, int contention,
                                        int steps, int warmup, BlockTime* out, unsigned* dump) {
    const int r = (int)cg::this_cluster().block_rank() - FIRST_WORKER;
    const unsigned word = word_intra(r, threadIdx.x, blockDim.x, contention);
    throughput<MEMORY, ATOMIC>(global_buf, target_rank, word, steps, warmup, out, dump);
}

template <Memory MEMORY, Atomic ATOMIC>
__global__ void atomic_throughput_inter(unsigned* global_buf, int target_rank, int contention,
                                        int steps, int warmup, BlockTime* out, unsigned* dump) {
    const int r = (int)cg::this_cluster().block_rank() - FIRST_WORKER;
    const unsigned word = word_inter(r, threadIdx.x, blockDim.x, contention);
    throughput<MEMORY, ATOMIC>(global_buf, target_rank, word, steps, warmup, out, dump);
}

// Latency: thread 0 of rank 0 runs a chain on word 0 of the buffer MEMORY
// gives (remote_smem: rank `target_rank`'s).
template <Memory MEMORY, Chain CHAIN>
__global__ void atomic_latency(unsigned* global_buf, int target_rank, int steps, int warmup,
                               BlockTime* out) {
    __shared__ unsigned buf[SBUF];
    cg::cluster_group cluster = cg::this_cluster();
    if (threadIdx.x == 0) buf[0] = 0;
    cluster.sync();
    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        unsigned* c = MEMORY(buf, global_buf, cluster, target_rank);
        unsigned i = 0;   // word 0 holds 0, and every atomic leaves it 0
        long long c0 = 0;
        unsigned long long t0 = 0;
        for (int s = 0; s < (warmup + 1) * steps; s++) {
            if (s == warmup * steps) { c0 = clock64(); t0 = globaltimer(); }
            i = CHAIN(&c[i]);
        }
        out[0].cycles = clock64() - c0;
        out[0].start_ns = t0;
        out[0].end_ns = globaltimer();
    }
    cluster.sync();   // my shared memory must outlive the last remote atomic
}

// ------------------------------------------------------------ host

// The buffer the atomics of `target` go to: a rank's shared memory, or -1 =
// the global buffer. smem: each worker its own.
static int buffer_of(int target, int rank) {
    if (target == SMEM)         return rank;
    if (target == DSMEM_RANK0)  return 0;
    if (target == DSMEM_RANK10) return 10;
    return -1;
}

// The host replay of one throughput run: what every word must hold.
static bool check(int target, int op, int scope, int contention, int block_size,
                  int atomics_per_thread,
                  const std::vector<unsigned>& dump, const std::vector<unsigned>& global) {
    // expected value per (buffer, word); buffer CLUSTER_SIZE = the global one
    std::vector<unsigned long long> add((CLUSTER_SIZE + 1) * SBUF, 0);
    std::vector<unsigned> orv((CLUSTER_SIZE + 1) * SBUF, 0);
    std::vector<std::vector<unsigned>> claimers((CLUSTER_SIZE + 1) * SBUF);
    for (int rank = FIRST_WORKER; rank < FIRST_WORKER + WORKERS; rank++)
        for (int tid = 0; tid < block_size; tid++) {
            const int r = rank - FIRST_WORKER;
            const unsigned id = r * block_size + tid;
            const unsigned word = scope == INTER ? word_inter(r, tid, block_size, contention)
                                : target == SMEM ? tid / contention
                                                 : word_intra(r, tid, block_size, contention);
            const int b = buffer_of(target, rank);
            const int k = (b < 0 ? CLUSTER_SIZE : b) * SBUF + word;
            add[k] += atomics_per_thread;
            orv[k] |= 1u << (id & 31);
            claimers[k].push_back(id + 1);
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
               "  --memory M        throughput: smem | dsmem-rank0 | dsmem-rank10 | global;\n"
               "                    latency: smem | dsmem | global\n"
               "  --scope S         throughput: intra | inter (-block contention) (default intra)\n"
               "  --contention x    throughput: threads per word, a power of two (default 1)\n"
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
        for (int t = 0; t < 4; t++)
            if (strcmp(memory_name, TARGET_NAME[t]) == 0) target = t;
    } else {
        const char* lat[] = {"smem", "dsmem", "global"};
        for (int t = 0; t < 3; t++)
            if (strcmp(memory_name, lat[t]) == 0) target = t;
    }
    if (target < 0) {
        fprintf(stderr, "atomic: --memory must be smem, dsmem-rank0, dsmem-rank10 or global "
                        "(throughput); smem, dsmem or global (latency)\n");
        return 1;
    }
    int scope = -1;
    for (int s = 0; s < 2; s++)
        if (strcmp(a.scope ? a.scope : "intra", SCOPE_NAME[s]) == 0) scope = s;
    if (scope < 0) { fprintf(stderr, "atomic: --scope must be intra or inter\n"); return 1; }
    const int threads = WORKERS * a.block_size;   // all the workers' threads
    if (throughput) {
        if (a.block_size < 32 || a.block_size > 1024 || (a.block_size & (a.block_size - 1)) ||
            threads > SBUF) {
            fprintf(stderr, "atomic: --block-size must be a power of two >= 32 with %d x it <= %d "
                            "(every thread needs its own word at --contention 1)\n", WORKERS, SBUF);
            return 1;
        }
        if (scope == INTER && target == SMEM) {
            fprintf(stderr, "atomic: --scope inter needs dsmem or global: no other block can "
                            "see a block's own shared memory\n");
            return 1;
        }
        const int most = scope == INTRA ? a.block_size : threads;   // who can share a word
        if (a.contention < 1 || a.contention > most || (a.contention & (a.contention - 1))) {
            fprintf(stderr, "atomic: --contention must be a power of two, 1..%d for --scope %s\n",
                    most, SCOPE_NAME[scope]);
            return 1;
        }
    } else if (target == 1 && (a.distance < 1 || a.distance >= CLUSTER_SIZE)) {
        fprintf(stderr, "atomic: --distance must be 1..%d\n", CLUSTER_SIZE - 1);
        return 1;
    }

    // ---- the kernel for this scope / target / op
    using ThroughputKernel = void (*)(unsigned*, int, int, int, int, BlockTime*, unsigned*);
    static const ThroughputKernel smem_kernels[3] = {
        atomic_throughput_smem<atomic_or>, atomic_throughput_smem<atomic_add>,
        atomic_throughput_smem<atomic_cas>};
    static const ThroughputKernel intra_kernels[2][3] = {   // [dsmem, global][op]
        {atomic_throughput_intra<remote_smem, atomic_or>, atomic_throughput_intra<remote_smem, atomic_add>,
         atomic_throughput_intra<remote_smem, atomic_cas>},
        {atomic_throughput_intra<global_mem, atomic_or>, atomic_throughput_intra<global_mem, atomic_add>,
         atomic_throughput_intra<global_mem, atomic_cas>},
    };
    static const ThroughputKernel inter_kernels[2][3] = {   // [dsmem, global][op]
        {atomic_throughput_inter<remote_smem, atomic_or>, atomic_throughput_inter<remote_smem, atomic_add>,
         atomic_throughput_inter<remote_smem, atomic_cas>},
        {atomic_throughput_inter<global_mem, atomic_or>, atomic_throughput_inter<global_mem, atomic_add>,
         atomic_throughput_inter<global_mem, atomic_cas>},
    };
    using LatencyKernel = void (*)(unsigned*, int, int, int, BlockTime*);
    static const LatencyKernel latency_kernels[3][3] = {   // [smem, dsmem, global][op]
        {atomic_latency<own_smem, chain_or>, atomic_latency<own_smem, chain_add>,
         atomic_latency<own_smem, chain_cas>},
        {atomic_latency<remote_smem, chain_or>, atomic_latency<remote_smem, chain_add>,
         atomic_latency<remote_smem, chain_cas>},
        {atomic_latency<global_mem, chain_or>, atomic_latency<global_mem, chain_add>,
         atomic_latency<global_mem, chain_cas>},
    };
    ThroughputKernel throughput_kernel = nullptr;
    int target_rank = a.distance;   // latency: the dsmem rank
    if (throughput) {
        const int memory = (target == GLOBAL) ? 1 : 0;   // row of intra_kernels / inter_kernels
        target_rank = (target == DSMEM_RANK10) ? 10 : 0;
        throughput_kernel = (target == SMEM)   ? smem_kernels[op]
                          : (scope == INTRA)   ? intra_kernels[memory][op]
                                               : inter_kernels[memory][op];
    }
    const void* kernel = throughput ? (const void*)throughput_kernel
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
             "# mode: %s\n# op: %s\n# memory: %s\n# contention: %d\n# scope: %s\n"
             "# workers: %d (ranks %d..%d)\n# buffer_words: %d\n",
             throughput ? "throughput" : "latency", OP_NAME[op], memory_name, contention,
             SCOPE_NAME[scope], workers, FIRST_WORKER, FIRST_WORKER + workers - 1, SBUF);
    print_header(argc, argv, a, 0, extra,
                 "benchmark,memory,mode,op,contention,scope,distance,block_size,workers,steps,seed,rep,"
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
            CUDA_CHECK(cudaLaunchKernelEx(&cfg, throughput_kernel, global_buf, target_rank, contention,
                                          a.steps, a.warmup, out, dump));
        else
            CUDA_CHECK(cudaLaunchKernelEx(&cfg, latency_kernels[target][op], global_buf,
                                          target_rank, a.steps, a.warmup, out));
        CUDA_CHECK(cudaDeviceSynchronize());

        if (throughput) {   // ---- correctness, then the time over the workers
            CUDA_CHECK(cudaMemcpy(host_dump.data(), dump, host_dump.size() * sizeof(unsigned),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(host_global.data(), global_buf, SBUF * sizeof(unsigned),
                                  cudaMemcpyDeviceToHost));
            if (!check(target, op, scope, contention, a.block_size, a.steps * (a.warmup + 1),
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
               throughput ? SCOPE_NAME[scope] : "-",
               throughput ? 0 : a.distance, (int)cfg.blockDim.x, workers, a.steps, a.seed, rep,
               atomics, ns, out[from].cycles, ns / atomics, cycles_per, atomics / ns);
        per_atomic.push_back(throughput ? ns / atomics : cycles_per);
        rate.push_back(atomics / ns);
    }

    fprintf(stderr, "atomic %s %s on %s", throughput ? "throughput" : "latency", OP_NAME[op],
            memory_name);
    if (throughput)
        fprintf(stderr, ", %s-block contention, %d threads per word, %d workers x %d threads, "
                        "verified", SCOPE_NAME[scope], contention, WORKERS, a.block_size);
    fprintf(stderr, "\n");
    print_stats_line(throughput ? "ns per atomic" : "cycles/atomic", per_atomic);
    print_stats_line("G atomics/s", rate);
    CUDA_CHECK(cudaFree(global_buf));
    CUDA_CHECK(cudaFree(dump));
    CUDA_CHECK(cudaFree(out));
    return 0;
}
