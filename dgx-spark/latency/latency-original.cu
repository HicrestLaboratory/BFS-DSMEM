// latency.cu — pointer-chase latency microbenchmark for thesis §3.1.2.
//
// Measures load-to-use latency of: local SMEM, remote SMEM (DSMEM) at each
// cluster rank distance (cluster sizes 2/4/8), L2-resident global memory,
// and DRAM. A single thread walks a dependent chain of loads (each address
// depends on the previous load), so latency is not hidden by ILP/MLP.
// Time is taken simultaneously in SM cycles (clock64) and wall-clock ns
// (%globaltimer); the cycle count is DVFS-insensitive, and the ratio of the
// two gives the SM clock actually sustained during each run.
//
// Build: nvcc -O3 -arch=sm_121 latency.cu -o latency   (sm_121 is required:
//        the default target lacks thread-block-cluster support)
// Run:   ./latency [reps=101] [steps=16384]
// Output: human-readable table of per-load latency by placement.

#include <cooperative_groups.h>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include "../common.cuh"  // CUDA_CHECK, Result, SBUF, globaltimer(), chase()

namespace cg = cooperative_groups;

// Cluster kernel: every block fills its SMEM with the same permutation;
// the thread 0 of rank 0 chases the buffer of rank `target` through
// map_shared_rank (or its own buffer directly if useMapped == 0).
__global__ void cluster_chase(const unsigned* __restrict__ perm, int steps,
                              int target, int useMapped, Result* out) {
    __shared__ unsigned sbuf[SBUF];
    cg::cluster_group cluster = cg::this_cluster();
    for (int i = threadIdx.x; i < SBUF; i += blockDim.x) sbuf[i] = perm[i];
    __syncthreads();
    cluster.sync();  // all buffers initialized before any remote access
    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        const unsigned* buf =
            useMapped ? cluster.map_shared_rank((const unsigned*)sbuf, target)
                      : sbuf;
        unsigned idx = 0;
        // untimed warm-up lap (pipelines, mapping)
        for (int i = 0; i < SBUF; i++) idx = buf[idx];
        chase(buf, idx, steps, out);
    }
    cluster.sync();  // keep all blocks resident while their SMEM is chased
}

// Plain (non-cluster) local SMEM baseline.
__global__ void smem_plain_chase(const unsigned* __restrict__ perm, int steps,
                                 Result* out) {
    __shared__ unsigned sbuf[SBUF];
    for (int i = threadIdx.x; i < SBUF; i += blockDim.x) sbuf[i] = perm[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned idx = 0;
        // untimed warm-up 1 lap
        for (int i = 0; i < SBUF; i++) idx = sbuf[idx];
        chase(sbuf, idx, steps, out);
    }
}

// Global-memory chase; L2 vs DRAM is determined by the buffer size and
// warming policy chosen on the host.
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
    if (s == 0xDEADBEEFull) *sink = s;  // never true; keeps the sweep alive avoiding dead-code elimination
}

struct Stats {
    double cyc, ns, ghz;
};

// Sorts v in place and returns its median.
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

// Print one measurement as a table row.
void print_benchmark(const char* cfg, int cs, int d, Stats s) {
    printf("%-24s cluster_size=%d distance=%2d : %7.1f cy  %7.2f ns  (%.2f GHz)\n", cfg, cs,
           d, s.cyc, s.ns, s.ghz);
}

// Run `launch` reps times, collecting per-load cycles and ns from *out
// (mapped managed memory written by the kernel), and return their medians.
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

// Launch cluster_chase as a `clusterSize`-block cluster.
void launch_cluster_chase(int clusterSize, const unsigned* perm,
                                 int steps, int target, int useMapped,
                                 Result* out) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(clusterSize, 1, 1);
    cfg.blockDim = dim3(128, 1, 1);
    cudaLaunchAttribute a[1];
    a[0].id = cudaLaunchAttributeClusterDimension;
    a[0].val.clusterDim.x = clusterSize;
    a[0].val.clusterDim.y = 1;
    a[0].val.clusterDim.z = 1;
    cfg.attrs = a;
    cfg.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, cluster_chase, perm, steps, target,
                                  useMapped, out));
}

int main(int argc, char** argv) {
    // how many reps of the benchmark to run
    int reps = argc > 1 ? atoi(argv[1]) : 101;
    // how many steps per rep
    int steps = argc > 2 ? atoi(argv[2]) : 16384;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("# %s  CC %d.%d  SMs %d  L2 %d KiB reps %d  steps %d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.l2CacheSize >> 10, reps, steps);

    std::mt19937 rng(42);
    Result* out;
    CUDA_CHECK(cudaMallocManaged(&out, sizeof(Result)));

    // --- SMEM permutation (shared by all SMEM cases) ---
    std::vector<unsigned> hperm(SBUF);
    make_cycle(hperm, rng);
    unsigned* dperm;
    CUDA_CHECK(cudaMalloc(&dperm, SBUF * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(dperm, hperm.data(), SBUF * sizeof(unsigned),
                          cudaMemcpyHostToDevice));

    // --- L2 buffer: L2 / 4 to increase cache hit rate, cycle permutation ---
    size_t l2n = (size_t)prop.l2CacheSize / 4 / sizeof(unsigned);
    std::vector<unsigned> hl2(l2n);
    make_cycle(hl2, rng);
    unsigned* dl2;
    CUDA_CHECK(cudaMalloc(&dl2, l2n * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(dl2, hl2.data(), l2n * sizeof(unsigned),
                          cudaMemcpyHostToDevice));

    // --- DRAM buffer: 1 GiB, cycle permutation. Each rep chases a
    //     different segment of the cycle (random start) so that lines
    //     cached by earlier reps are not re-walked. ---
    size_t drn = (1ull << 30) / sizeof(unsigned);
    unsigned* ddr;
    CUDA_CHECK(cudaMalloc(&ddr, drn * sizeof(unsigned)));
    {
        std::vector<unsigned> hdr(drn);
        make_cycle(hdr, rng);
        CUDA_CHECK(cudaMemcpy(ddr, hdr.data(), drn * sizeof(unsigned),
                              cudaMemcpyHostToDevice));
    }
    // used to keep the warm-up of L2 alive (avoid dead-code elimination)
    unsigned long long* dsink;
    CUDA_CHECK(cudaMalloc(&dsink, sizeof(unsigned long long)));

    // 1) plain local shared memory (no cluster)
    print_benchmark("SHARED MEMORY", 1, 0, run_reps(reps, steps, out, [&] {
        smem_plain_chase<<<1, 128>>>(dperm, steps, out);
    }));

    // 2) cluster cases: local-direct, and mapped at every rank distance.
    //    Sizes above the portable maximum of 8 (GB10 accepts up to 12) are
    //    rejected as "cluster misconfiguration" unless the kernel opts in.
    CUDA_CHECK(cudaFuncSetAttribute(
        (void*)cluster_chase, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    for (int cs : {2, 4, 8, 12}) {
        for (int d = 0; d < cs; d++)
            print_benchmark("DSMEM", cs, d, run_reps(reps, steps, out, [&] {
                     launch_cluster_chase(cs, dperm, steps, d, 1, out);
                 }));
    }

    // 3) L2: warm once, then chase (reps keep it resident)
    warm_up<<<prop.multiProcessorCount * 4, 256>>>(dl2, l2n, dsink);
    CUDA_CHECK(cudaDeviceSynchronize());
    {
        std::uniform_int_distribution<unsigned> d(0, (unsigned)l2n - 1);
        print_benchmark("L2 CACHE", 0, 0, run_reps(reps, steps, out, [&] {
                 gmem_chase<<<1, 32>>>(dl2, d(rng), steps, out);
             }));
    }

    // 4) DRAM: fresh random start each rep, no warming
    {
        std::uniform_int_distribution<unsigned> d(0, (unsigned)drn - 1);
        print_benchmark("DRAM", 0, 0, run_reps(reps, steps, out, [&] {
                 gmem_chase<<<1, 32>>>(ddr, d(rng), steps, out);
             }));
    }

    cudaFree(dperm);
    cudaFree(dl2);
    cudaFree(ddr);
    cudaFree(dsink);
    cudaFree(out);
    return 0;
}
// quanto si fa fatica a fare lo scheduling in concorrenza con altri processi (// che impatto ha lo scheduler)
// un altra versione del benchmark in cui non è solo il thread 0 a fare il chase, ma tutti i thread del cluster fanno il chase in concorrenza, e si misura quanto aumenta la latenza per via dello scheduling.
// cosa succede se invece di una random chase usiamo una linear chase, cioè un buffer in cui ogni elemento punta al successivo, e l'ultimo punta al primo. In questo caso il prefetcher dovrebbe essere in grado di anticipare le richieste e ridurre la latenza.
// provare a usare stride-n
// tenersi la varianza, prendi le run, emetti tutti i numeri, passare i parametri da cli, ogni programma una casistica

// slides con disegnini, come funziona l'accesso, il warmup
// fare grafici
// usare spatzman riproducibilità, come si fa a garantire che il benchmark sia riproducibile
