// common.cuh — code shared by the five latency microbenchmarks.
//
// Everything that is identical across the programs lives here, so that each
// .cu file contains only the kernel it measures plus a short main(). Read this
// file once; after that every program is a few dozen lines.
//
// The measurement technique is a DEPENDENT POINTER CHASE: a single thread
// executes `idx = buf[idx]` in a loop. Each load's address is the previous
// load's value, so the hardware cannot issue the next load before the current
// one returns. The time per step is therefore the true load-to-use latency,
// not hidden by memory-level parallelism.

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Size of the shared-memory chase buffer, in 4-byte elements (16 KiB).
// Small enough that any block can allocate it statically; large enough that
// a chase over it is not a handful of addresses.
constexpr int SBUF = 4096;

// One measurement, written by the kernel into managed memory.
struct Result {
    long long cycles;       // SM clock cycles, from clock64()
    unsigned long long ns;  // wall-clock nanoseconds, from %globaltimer
    unsigned sink;          // final index; written out so the compiler
                            // cannot delete the chase as dead code
};

// ---------------------------------------------------------------- device side

// %globaltimer is a 64-bit wall-clock nanosecond counter (Ampere and later).
// We time in BOTH cycles and ns: cycles are insensitive to clock changes
// (DVFS), and cycles/ns tells us the clock the SM actually ran at.
__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// The timed chase itself. Executed by ONE thread.
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

// `warmup` untimed passes, then the timed one, all by the same thread.
// Without the warm-up the first pass pays one-off costs (instruction cache,
// the DSMEM address mapping, TLB fills) that are not the latency we want.
// The timed pass CONTINUES along the cycle from where the warm-up stopped,
// so it walks different addresses: warming up does not turn a DRAM chase
// into an L2 chase.
__device__ __forceinline__ void warm_then_chase(const unsigned* buf,
                                                unsigned start, int warmup,
                                                int steps, Result* out) {
    unsigned idx = start;
    for (int w = 0; w < warmup; w++)
        for (int i = 0; i < steps; i++) idx = buf[idx];
    chase(buf, idx, steps, out);
}

// ------------------------------------------------------------------ host side

// Sattolo's algorithm: a random permutation that is ONE cycle visiting every
// element exactly once. Two properties matter:
//   * no short cycles — a plain shuffle could contain a 3-element loop, and a
//     chase that fell into it would measure a 3-element working set;
//   * random order — the next address is unpredictable, so no prefetcher can
//     fetch it ahead of time. That is exactly what makes this a latency test.
void make_cycle(std::vector<unsigned>& p, std::mt19937& rng) {
    size_t n = p.size();
    std::iota(p.begin(), p.end(), 0u);
    for (size_t i = n - 1; i > 0; i--) {
        std::uniform_int_distribution<size_t> d(0, i - 1);
        std::swap(p[i], p[d(rng)]);
    }
}

// Fixed stride: p[i] = (i + s) mod n. The chase then visits i, i+s, i+2s, ...
// a perfectly regular address stream. If any prefetcher watches the addresses
// going out, this is the pattern it can predict. Stride 4 B is the "linear"
// walk, one element to the next.
void make_stride(std::vector<unsigned>& p, size_t stride_elems) {
    size_t n = p.size();
    for (size_t i = 0; i < n; i++) p[i] = (unsigned)((i + stride_elems) % n);
}

// Fill the buffer with the pattern selected by --stride. Whatever the
// pattern, the kernel runs the same `idx = buf[idx]` loop: only the VALUES
// in the buffer change, so the address stream is the one thing that differs
// between a random run and a stride run.
void make_pattern(std::vector<unsigned>& p, std::mt19937& rng, int stride_bytes) {
    if (stride_bytes == 0) { make_cycle(p, rng); return; }
    size_t s = stride_bytes / sizeof(unsigned);
    if (s >= p.size()) {
        fprintf(stderr, "stride of %d B is not smaller than the buffer (%zu B)\n",
                stride_bytes, p.size() * sizeof(unsigned));
        exit(1);
    }
    make_stride(p, s);
}

// Distinct elements a chase visits before it returns to its start.
// Random: all n. Stride s: n / gcd(n, s) -- a stride that divides n visits
// only every s-th element, then repeats. A walk longer than this re-reads
// addresses it has just touched, which turns a DRAM test into an L2 test.
size_t cycle_length(size_t n, int stride_bytes) {
    if (stride_bytes == 0) return n;
    return n / std::gcd(n, (size_t)(stride_bytes / sizeof(unsigned)));
}

// Every flag every program understands. A program simply ignores the ones it
// does not use, and writes a 0 for them in its CSV rows.
struct Args {
    int reps = 101;         // repetitions; one CSV row each
    int steps = 16384;      // dependent loads per repetition
    int seed = 42;          // RNG seed for the permutation
    int warmup = 1;         // untimed passes before the timed one
    int block_size = 128;   // threads per block
    int cluster_size = 2;   // blocks per cluster
    int distance = 1;       // rank distance from rank 0 to the target block
    int mapped = 1;         // 1 = go through map_shared_rank, 0 = direct pointer
    size_t buffer_bytes = 0;  // 0 = "use the program's default"
    int stride_bytes = 0;   // 0 = random (Sattolo); > 0 = fixed stride in bytes
    // latency-many-threads.cu only:
    int chasers = 1;        // chasing threads per requester block
    int requesters = 1;     // requester blocks (cluster size = requesters + 1)
    int target_remote = 1;  // 1 = chase rank 0's buffer, 0 = chase own buffer
    int bank_aligned = 1;   // 1 = each lane on its own bank, 0 = random banks
};

// The flags common to every program, for --help.
const char* COMMON_USAGE =
    "  --reps N          repetitions, one CSV row each        (default 101)\n"
    "  --steps N         dependent loads per repetition       (default 16384)\n"
    "  --seed N          RNG seed for the permutation         (default 42)\n"
    "  --warmup N        untimed passes before the timed one  (default 1)\n"
    "  --stride random|B access pattern: random cycle, or a fixed\n"
    "                    stride of B bytes (multiple of 4)     (default random)\n";

// A plain loop over argv: `--flag value` pairs. Unknown flags are an error,
// so a typo cannot silently run the wrong experiment.
void parse_args(int argc, char** argv, Args* a, const char* prog,
                const char* extra_usage) {
    for (int i = 1; i < argc; i++) {
        const char* f = argv[i];
        if (strcmp(f, "--help") == 0 || strcmp(f, "-h") == 0) {
            printf("usage: %s [flags]\n%s%s", prog, COMMON_USAGE, extra_usage);
            exit(0);
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "%s: flag %s needs a value\n", prog, f);
            exit(1);
        }
        const char* v = argv[++i];
        if (strcmp(f, "--reps") == 0)               a->reps = atoi(v);
        else if (strcmp(f, "--steps") == 0)         a->steps = atoi(v);
        else if (strcmp(f, "--seed") == 0)          a->seed = atoi(v);
        else if (strcmp(f, "--warmup") == 0)        a->warmup = atoi(v);
        else if (strcmp(f, "--block-size") == 0)    a->block_size = atoi(v);
        else if (strcmp(f, "--cluster-size") == 0)  a->cluster_size = atoi(v);
        else if (strcmp(f, "--distance") == 0)      a->distance = atoi(v);
        else if (strcmp(f, "--mapped") == 0)        a->mapped = atoi(v);
        else if (strcmp(f, "--buffer-bytes") == 0)  a->buffer_bytes = strtoull(v, nullptr, 10);
        else if (strcmp(f, "--buffer-kib") == 0)    a->buffer_bytes = strtoull(v, nullptr, 10) * 1024;
        else if (strcmp(f, "--chasers") == 0)       a->chasers = atoi(v);
        else if (strcmp(f, "--requesters") == 0)    a->requesters = atoi(v);
        else if (strcmp(f, "--bank-aligned") == 0)  a->bank_aligned = atoi(v);
        else if (strcmp(f, "--target") == 0) {
            if (strcmp(v, "remote") == 0)      a->target_remote = 1;
            else if (strcmp(v, "local") == 0)  a->target_remote = 0;
            else { fprintf(stderr, "%s: --target must be remote or local\n", prog); exit(1); }
        }
        else if (strcmp(f, "--stride") == 0) {
            if (strcmp(v, "random") == 0) a->stride_bytes = 0;
            else {
                a->stride_bytes = atoi(v);
                if (a->stride_bytes <= 0 || a->stride_bytes % 4 != 0) {
                    fprintf(stderr, "%s: --stride must be 'random' or a positive "
                                    "multiple of 4 bytes\n", prog);
                    exit(1);
                }
            }
        }
        else {
            fprintf(stderr, "%s: unknown flag %s (try --help)\n", prog, f);
            exit(1);
        }
    }
}

// Clusters larger than 8 are "non-portable": CUDA only guarantees 8 on every
// cluster-capable GPU. GB10 accepts up to 12, but only if the kernel opts in.
void allow_big_clusters(const void* kernel) {
    CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
}

// ------------------------------------------------------------------- output

// Reject a stride that does not fit the buffer BEFORE anything is printed,
// so a bad run leaves no half-written CSV behind.
void check_pattern(size_t n_elems, int stride_bytes) {
    if (stride_bytes > 0 && (size_t)stride_bytes / sizeof(unsigned) >= n_elems) {
        fprintf(stderr, "stride of %d B is not smaller than the buffer (%zu B)\n",
                stride_bytes, n_elems * sizeof(unsigned));
        exit(1);
    }
}

// stdout is MACHINE-READABLE ONLY: a few '#' metadata lines, then a CSV
// header, then one row per repetition. pandas.read_csv(path, comment='#')
// parses it directly. The metadata block records everything needed to
// reproduce the run from the log alone.
// `extra` lets a program add its own '#' metadata lines before the CSV header.
void print_header(int argc, char** argv, const Args& a, size_t n_elems,
                  const char* extra = nullptr) {
    check_pattern(n_elems, a.stride_bytes);
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int drv = 0, rt = 0, khz = 0;
    CUDA_CHECK(cudaDriverGetVersion(&drv));
    CUDA_CHECK(cudaRuntimeGetVersion(&rt));
    CUDA_CHECK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));

    printf("# command:");
    for (int i = 0; i < argc; i++) printf(" %s", argv[i]);
    if (extra) fputs(extra, stdout);
    printf("benchmark,cluster_size,distance,mapped,block_size,steps,"
           "buffer_bytes,stride_bytes,seed,rep,cycles,ns,cycles_per_load,"
           "ns_per_load,ghz,chasers,warp\n");
}

// One CSV row. The schema is identical for all programs; a program passes 0
// for the fields that do not apply to it, so every output file has the same
// columns and they can all be loaded into one table. The last two columns
// (chasing threads per block, and which warp this row is) matter only for
// latency-many-threads.cu; single-thread programs leave them at 1 and 0.
void print_row(const char* benchmark, int cluster_size, int distance,
               int mapped, int block_size, int steps, size_t buffer_bytes,
               int stride_bytes, int seed, int rep, const Result& r,
               int chasers = 1, int warp = 0) {
    double cpl = (double)r.cycles / steps;
    double npl = (double)r.ns / steps;
    printf("%s,%d,%d,%d,%d,%d,%zu,%d,%d,%d,%lld,%llu,%.4f,%.4f,%.4f,%d,%d\n",
           benchmark, cluster_size, distance, mapped, block_size, steps,
           buffer_bytes, stride_bytes, seed, rep, r.cycles, r.ns, cpl, npl,
           cpl / npl, chasers, warp);
}

// Value below which a fraction `q` of the (sorted) samples lie.
double percentile(const std::vector<double>& sorted, double q) {
    size_t i = (size_t)(q * (sorted.size() - 1));
    return sorted[i];
}

void print_stats_line(const char* label, std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double var = 0;
    for (double x : v) var += (x - mean) * (x - mean);
    double sd = std::sqrt(var / v.size());
    fprintf(stderr,
            "  %-15s n=%zu  min=%.2f  p25=%.2f  median=%.2f  mean=%.2f  "
            "sd=%.2f  p95=%.2f  max=%.2f\n",
            label, v.size(), v.front(), percentile(v, 0.25),
            percentile(v, 0.5), mean, sd, percentile(v, 0.95), v.back());
}

// The human-readable summary goes to STDERR so it never mixes with the CSV.
// It is a convenience: the per-repetition rows on stdout are the real record.
void print_summary(const char* benchmark, const std::vector<double>& cpl,
                   const std::vector<double>& npl) {
    fprintf(stderr, "%s summary (per load):\n", benchmark);
    print_stats_line("cycles", cpl);
    print_stats_line("ns", npl);
}
