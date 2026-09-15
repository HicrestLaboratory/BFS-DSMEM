// latency-original-malloc.cu — does it matter HOW global memory is allocated?
//
// On the DGX Spark CPU and GPU share one physical LPDDR5x pool, and the GPU
// can dereference a plain malloc() pointer (pageableMemoryAccess = 1). So
// the question is not "does it work" but "is it as fast". This program runs
// the same dependent pointer chase (see latency.cu) over four buffers that
// hold the same permutation but were allocated differently:
//
//   cudaMalloc          GPU-side allocation, 2 MiB GPU pages
//   cudaMallocManaged   unified allocation, migrates/maps on demand
//   malloc              plain host allocation, 4 KiB pages
//   malloc + THP        plain host allocation, madvise(MADV_HUGEPAGE) -> 2 MiB
//
// The last two differ only in page size. That isolates the confound: a random
// chase over 1 GiB in 4 KiB pages misses the GPU TLB on nearly every load,
// and that would show up as "malloc is slow" even if the memory path were
// identical. DGX OS sets transparent huge pages to "madvise", so a plain
// malloc never gets 2 MiB pages unless asked.
//
// Each buffer is measured L2-resident (small buffer, warmed) and cold (1 GiB,
// never warmed), exactly as in latency.cu.
//
// Build: nvcc -O3 -arch=sm_121 latency-original-malloc.cu -o latency-original-malloc
// Run:   ./latency-original-malloc [reps=101] [steps=16384]

#include <cuda_runtime.h>
#include <sys/mman.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, \
                    __LINE__, cudaGetErrorString(err_));                     \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

struct Result {
    long long cycles;
    unsigned long long ns;
    unsigned sink;  // defeats dead-code elimination of the chase
};

__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Dependent chase: each load's address is the previous load's value.
__device__ __forceinline__ void chase(const unsigned* buf, unsigned start,
                                      int steps, Result* out) {
    unsigned idx = start;
    long long c0 = clock64();
    unsigned long long g0 = globaltimer();
    for (int i = 0; i < steps; i++) idx = buf[idx];
    long long c1 = clock64();
    unsigned long long g1 = globaltimer();
    out->cycles = c1 - c0;
    out->ns = g1 - g0;
    out->sink = idx;
}

__global__ void gmem_chase(const unsigned* __restrict__ buf, unsigned start,
                           int steps, Result* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) chase(buf, start, steps, out);
}

// Coalesced sweep to pull a buffer into L2.
__global__ void warm_up(const unsigned* __restrict__ buf, size_t n,
                        unsigned long long* sink) {
    unsigned long long s = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        s += buf[i];
    if (s == 0xDEADBEEFull) *sink = s;  // never true; keeps the sweep alive
}

// Sattolo's algorithm: a single cycle visiting every element exactly once,
// in random order — defeats any prefetcher and guarantees no short cycles.
void make_cycle(std::vector<unsigned>& p, std::mt19937& rng) {
    size_t n = p.size();
    std::iota(p.begin(), p.end(), 0u);
    for (size_t i = n - 1; i > 0; i--) {
        std::uniform_int_distribution<size_t> d(0, i - 1);
        std::swap(p[i], p[d(rng)]);
    }
}

struct Stats {
    double cyc, ns, ghz;
};

double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

Stats median_of(std::vector<double>& cyc, std::vector<double>& ns) {
    Stats s;
    s.cyc = median(cyc);
    s.ns = median(ns);
    s.ghz = s.cyc / s.ns;
    return s;
}

void print_benchmark(const char* placement, const char* alloc, Stats s) {
    printf("%-10s %-22s : %7.1f cy  %7.2f ns  (%.2f GHz)\n", placement, alloc,
           s.cyc, s.ns, s.ghz);
}

Stats run_reps(int reps, int steps, const Result* out,
               const std::function<void()>& launch) {
    std::vector<double> vc(reps), vn(reps);
    for (int r = 0; r < reps; r++) {
        launch();
        CUDA_CHECK(cudaDeviceSynchronize());
        vc[r] = (double)out->cycles / steps;
        vn[r] = (double)out->ns / steps;
    }
    return median_of(vc, vn);
}

// ------------------------------------------------------------ allocation

// Plain malloc, but 2 MiB-aligned and with a request for transparent huge
// pages. Without the madvise, THP in "madvise" mode gives 4 KiB pages.
unsigned* malloc_huge(size_t bytes) {
    void* p = aligned_alloc(2u << 20, bytes);
    if (!p) { perror("aligned_alloc"); exit(1); }
    if (madvise(p, bytes, MADV_HUGEPAGE) != 0) perror("madvise(MADV_HUGEPAGE)");
    return (unsigned*)p;
}

// How much of this process's anonymous memory is backed by huge pages right
// now — the check that the madvise above actually took effect.
void print_thp_usage(const char* when) {
    FILE* f = fopen("/proc/self/smaps_rollup", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof line, f))
        if (strncmp(line, "AnonHugePages:", 14) == 0)
            printf("# %s: %s", when, line);  // line already ends in '\n'
    fclose(f);
}

// The same permutation in four differently-allocated buffers.
struct Buffers {
    const char* name[4] = {"cudaMalloc", "cudaMallocManaged",
                           "malloc (4 KiB pages)", "malloc + THP (2 MiB)"};
    unsigned* ptr[4];
};

Buffers make_buffers(const std::vector<unsigned>& h) {
    size_t bytes = h.size() * sizeof(unsigned);
    Buffers b;
    CUDA_CHECK(cudaMalloc(&b.ptr[0], bytes));
    CUDA_CHECK(cudaMemcpy(b.ptr[0], h.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMallocManaged(&b.ptr[1], bytes));
    memcpy(b.ptr[1], h.data(), bytes);
    b.ptr[2] = (unsigned*)malloc(bytes);
    memcpy(b.ptr[2], h.data(), bytes);  // also faults the pages in
    b.ptr[3] = malloc_huge(bytes);
    memcpy(b.ptr[3], h.data(), bytes);
    return b;
}

void free_buffers(Buffers& b) {
    cudaFree(b.ptr[0]);
    cudaFree(b.ptr[1]);
    free(b.ptr[2]);
    free(b.ptr[3]);
}

int main(int argc, char** argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 101;
    int steps = argc > 2 ? atoi(argv[2]) : 16384;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("# %s  CC %d.%d  SMs %d  L2 %d KiB  reps %d  steps %d\n", prop.name,
           prop.major, prop.minor, prop.multiProcessorCount,
           prop.l2CacheSize >> 10, reps, steps);
    printf("# pageableMemoryAccess=%d  concurrentManagedAccess=%d\n",
           prop.pageableMemoryAccess, prop.concurrentManagedAccess);
    if (!prop.pageableMemoryAccess) {
        fprintf(stderr, "this GPU cannot read malloc() memory directly\n");
        return 1;
    }

    std::mt19937 rng(42);
    Result* out;
    CUDA_CHECK(cudaMallocManaged(&out, sizeof(Result)));
    unsigned long long* dsink;
    CUDA_CHECK(cudaMalloc(&dsink, sizeof(unsigned long long)));

    // --- L2 buffers: L2/4 so the chase is fully cache-resident ---
    size_t l2n = (size_t)prop.l2CacheSize / 4 / sizeof(unsigned);
    std::vector<unsigned> hl2(l2n);
    make_cycle(hl2, rng);
    Buffers l2 = make_buffers(hl2);

    // --- DRAM buffers: 1 GiB, far larger than L2, never warmed ---
    size_t drn = (1ull << 30) / sizeof(unsigned);
    Buffers dr;
    {
        std::vector<unsigned> hdr(drn);
        make_cycle(hdr, rng);
        dr = make_buffers(hdr);
    }  // drop the 1 GiB host copy before measuring
    print_thp_usage("AnonHugePages after allocation");

    // CUDA loads a kernel lazily on its first launch, and that load flushes
    // L2. Launch the chase kernel once, untimed, so the first L2 warm-up
    // below is not undone by it (this is the bug in latency.cu).
    gmem_chase<<<1, 32>>>(l2.ptr[0], 0, 0, out);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- L2-resident: warm, then chase; the reps keep it resident ---
    std::uniform_int_distribution<unsigned> l2start(0, (unsigned)l2n - 1);
    for (int k = 0; k < 4; k++) {
        warm_up<<<prop.multiProcessorCount * 4, 256>>>(l2.ptr[k], l2n, dsink);
        CUDA_CHECK(cudaDeviceSynchronize());
        print_benchmark("L2 CACHE", l2.name[k], run_reps(reps, steps, out, [&] {
            gmem_chase<<<1, 32>>>(l2.ptr[k], l2start(rng), steps, out);
        }));
    }

    // --- DRAM: fresh random start each rep, no warming ---
    std::uniform_int_distribution<unsigned> drstart(0, (unsigned)drn - 1);
    for (int k = 0; k < 4; k++) {
        print_benchmark("DRAM", dr.name[k], run_reps(reps, steps, out, [&] {
            gmem_chase<<<1, 32>>>(dr.ptr[k], drstart(rng), steps, out);
        }));
    }

    free_buffers(l2);
    free_buffers(dr);
    cudaFree(dsink);
    cudaFree(out);
    return 0;
}
