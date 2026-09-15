// l1_carveout.cu — how big is the L1 data cache, and what do shared memory
// requests and the carveout setting do to it?
//
// Background: since Volta, an SM's L1 cache and its shared memory are the same
// physical SRAM (128 KiB here), split by a configurable "carveout". The split
// is chosen per kernel launch by the driver;
// cudaFuncAttributePreferredSharedMemoryCarveout states a preference as a
// percentage of the maximum shared memory (0 = favour L1, 100 = favour SMEM).
//
// Method: pointer-chase a random cycle of increasing size. While the working
// set fits in L1, every load is an L1 hit at flat latency; once it exceeds L1,
// loads spill to L2 and the cycles/load figure jumps. The size where the jump
// happens ("the knee") measures the *effective* L1 capacity. A random cycle is
// used so the prefetcher cannot help and each load's address depends on the
// previous load's value — no memory-level parallelism hides the latency.
//
// Three experiments:
//   A. Working-set sweep with 0 vs 99 KiB of dynamic shared memory requested:
//      the raw latency table, showing the L1 collapse at max SMEM.
//   B. Carveout sweep with NO shared memory: isolates the carveout as the
//      variable that controls L1 size.
//   C. The trap: request just 8 KiB of dynamic SMEM and compare the default
//      carveout against explicitly set ones. The driver's default favours
//      shared memory however little was asked for, and the L1 collapses
//      unless the carveout is stated explicitly.
// Each experiment uses its own kernel: per-kernel attributes persist, and
// reusing one kernel across experiments would contaminate the results.
//
// Build: nvcc -O3 -arch=sm_121 l1_carveout.cu -o l1_carveout
// Run:   ./l1_carveout             (about a minute)

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#define CK(c) do { cudaError_t e_ = (c); if (e_ != cudaSuccess) { \
    printf("ERR %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

// A: dynamic-SMEM chase. useSmem touches the allocation so it is real.
__global__ void chaseA(const unsigned* b, int steps, long long* o, int useSmem) {
    extern __shared__ unsigned smA[];
    if (useSmem && threadIdx.x == 0) smA[0] = 1;
    if (threadIdx.x) return;
    unsigned i = 0;
    for (int k = 0; k < 4096; k++) i = b[i];         // warm the cache
    long long t0 = clock64();
    for (int k = 0; k < steps; k++) i = b[i];
    *o = clock64() - t0;
    if (i == 0xFFFFFFFFu) *o = 0;                    // keep the chase alive
}
// B: plain chase, no shared memory at all.
__global__ void chaseB(const unsigned* b, int steps, long long* o) {
    if (threadIdx.x) return;
    unsigned i = 0;
    for (int k = 0; k < 8192; k++) i = b[i];
    long long t0 = clock64();
    for (int k = 0; k < steps; k++) i = b[i];
    *o = clock64() - t0;
    if (i == 0xFFFFFFFFu) *o = 0;
}
// C: chase with 8 KiB of dynamic SMEM requested.
__global__ void chaseC(const unsigned* b, int steps, long long* o) {
    extern __shared__ unsigned smC[];
    if (threadIdx.x == 0) smC[0] = 1;
    if (threadIdx.x) return;
    unsigned i = 0;
    for (int k = 0; k < 8192; k++) i = b[i];
    long long t0 = clock64();
    for (int k = 0; k < steps; k++) i = b[i];
    *o = clock64() - t0;
    if (i == 0xFFFFFFFFu) *o = 0;
}

// One random cycle over n words (Sattolo's algorithm: a single cycle visiting
// every element once, so there are no short cycles to get stuck in).
static unsigned* make_buf(size_t kib, std::mt19937& rng) {
    size_t n = kib * 1024 / sizeof(unsigned);
    std::vector<unsigned> h(n);
    std::iota(h.begin(), h.end(), 0u);
    for (size_t i = n - 1; i > 0; i--)
        std::swap(h[i], h[std::uniform_int_distribution<size_t>(0, i - 1)(rng)]);
    unsigned* d; CK(cudaMalloc(&d, n * 4));
    CK(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice));
    return d;
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end()); return v[v.size() / 2];
}

int main() {
    const int steps = 8192;
    std::mt19937 rng(11);
    long long* cyc; CK(cudaMallocManaged(&cyc, sizeof(long long)));

    // -------- A. working-set sweep, smem = 0 vs 99 KiB --------
    {
        const int reps = 21;
        std::vector<int> ws = {4, 8, 16, 24, 28, 32, 40, 48, 56, 64,
                               80, 96, 112, 128, 160, 192, 256, 384};
        std::vector<unsigned*> bufs;
        for (int k : ws) bufs.push_back(make_buf(k, rng));
        int optin = 0; CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
        printf("=== A. cycles/load vs working set, smem request 0 vs %d B ===\n", optin);
        printf("%-10s %18s %20s\n", "working", "smem=0 (max L1)", "smem=99KiB (min L1)");
        printf("%-10s %18s %20s\n", "set (KiB)", "cycles/load", "cycles/load");
        std::vector<double> resA[2];
        int cases[2] = {0, optin};
        for (int c = 0; c < 2; c++) {
            CK(cudaFuncSetAttribute((void*)chaseA, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    std::max(cases[c], 1)));
            for (size_t k = 0; k < ws.size(); k++) {
                std::vector<double> v;
                for (int r = 0; r < reps; r++) {
                    cudaLaunchConfig_t cfg = {};
                    cfg.gridDim = dim3(1, 1, 1); cfg.blockDim = dim3(32, 1, 1);
                    cfg.dynamicSmemBytes = std::max(cases[c], 1);
                    CK(cudaLaunchKernelEx(&cfg, chaseA, (const unsigned*)bufs[k], steps, cyc,
                                          cases[c] > 0 ? 1 : 0));
                    CK(cudaDeviceSynchronize());
                    v.push_back((double)*cyc / steps);
                }
                resA[c].push_back(median(v));
            }
        }
        for (size_t k = 0; k < ws.size(); k++)
            printf("%-10d %18.1f %20.1f\n", ws[k], resA[0][k], resA[1][k]);
        for (auto b : bufs) cudaFree(b);
    }

    // -------- B. carveout sweep, no shared memory --------
    {
        const int reps = 15;
        std::vector<int> ws; for (int k = 4; k <= 152; k += 4) ws.push_back(k);
        std::vector<unsigned*> bufs;
        for (int k : ws) bufs.push_back(make_buf(k, rng));
        printf("\n=== B. effective L1 vs carveout (no dynamic smem) ===\n");
        printf("(plateau = largest set still within 15%% of base latency;"
               " knee = first set above 2x base)\n");
        printf("%-22s %-14s %-12s\n", "carveout (% shared)", "L1 plateau KiB", "knee KiB");
        for (int pc : {0, 10, 25, 50, 75, 100}) {
            CK(cudaFuncSetAttribute((void*)chaseB,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, pc));
            double base = 0; int knee = -1, plateau = 0;
            for (size_t k = 0; k < ws.size(); k++) {
                std::vector<double> v;
                for (int r = 0; r < reps; r++) {
                    chaseB<<<1, 32>>>(bufs[k], steps, cyc);
                    CK(cudaDeviceSynchronize());
                    v.push_back((double)*cyc / steps);
                }
                double m = median(v);
                if (!k) base = m;
                if (m < base * 1.15) plateau = ws[k];
                if (knee < 0 && m > base * 2.0) knee = ws[k];
            }
            printf("%-22d %-14d %-12d\n", pc, plateau, knee);
        }
        for (auto b : bufs) cudaFree(b);
    }

    // -------- C. the trap: 8 KiB dynamic smem, default vs explicit carveout --------
    {
        const int reps = 15;
        std::vector<int> ws; for (int k = 4; k <= 120; k += 4) ws.push_back(k);
        std::vector<unsigned*> bufs;
        for (int k : ws) bufs.push_back(make_buf(k, rng));
        printf("\n=== C. 8 KiB dynamic smem: default carveout vs explicit ===\n");
        printf("%-34s %-12s\n", "configuration", "L1 plateau KiB");
        CK(cudaFuncSetAttribute((void*)chaseC, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 1024));
        for (int pc : {-1, 10, 25, 50}) {                 // -1 = leave the default
            if (pc >= 0)
                CK(cudaFuncSetAttribute((void*)chaseC,
                                        cudaFuncAttributePreferredSharedMemoryCarveout, pc));
            double base = 0; int plateau = 0;
            for (size_t k = 0; k < ws.size(); k++) {
                std::vector<double> v;
                for (int r = 0; r < reps; r++) {
                    cudaLaunchConfig_t cfg = {};
                    cfg.gridDim = dim3(1, 1, 1); cfg.blockDim = dim3(32, 1, 1);
                    cfg.dynamicSmemBytes = 8 * 1024;
                    CK(cudaLaunchKernelEx(&cfg, chaseC, (const unsigned*)bufs[k], steps, cyc));
                    CK(cudaDeviceSynchronize());
                    v.push_back((double)*cyc / steps);
                }
                double m = median(v);
                if (!k) base = m;
                if (m < base * 1.15) plateau = ws[k];
            }
            if (pc < 0) printf("%-34s %-12d\n", "carveout left at default", plateau);
            else { char nm[40]; snprintf(nm, sizeof nm, "carveout set to %d%%", pc);
                   printf("%-34s %-12d\n", nm, plateau); }
        }
        for (auto b : bufs) cudaFree(b);
    }
    return 0;
}
