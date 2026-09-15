// l2.cu — latency of global memory that is resident in the L2 cache.
//
// The chase buffer is much smaller than L2 (default: a quarter of it), it is
// pulled into L2 once with a coalesced sweep, and every repetition then
// walks a random segment of it. Nothing else runs, so the buffer stays
// resident and every load is an L2 hit.
//
// Check it yourself: cycles_per_load must be FLAT across repetitions from
// rep 0. A downward trend means the cache was cold at the start and the
// chase is warming it — see the note on lazy module loading in main().
//
// Build: make l2
// Run:   ./l2 [--buffer-bytes N (default l2CacheSize/4)]
//             [--reps 101] [--steps 16384] [--seed 42] [--warmup 1]

#include "common.cuh"

// A single thread chases; the block is one warp so nothing else competes.
__global__ void gmem_chase(const unsigned* __restrict__ buf, unsigned start,
                           int warmup, int steps, Result* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        warm_then_chase(buf, start, warmup, steps, out);
}

// Coalesced read of the whole buffer by many threads: the fastest way to
// bring it into L2. The impossible comparison keeps the loads alive — the
// compiler would otherwise delete a loop whose result is never used.
__global__ void warm_l2(const unsigned* __restrict__ buf, size_t n,
                        unsigned long long* sink) {
    unsigned long long s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        s += buf[i];
    if (s == 0xDEADBEEFull) *sink = s;
}

int main(int argc, char** argv) {
    Args a;
    parse_args(argc, argv, &a, "l2",
               "  --buffer-bytes N  chase buffer size                   (default l2CacheSize/4)\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (a.buffer_bytes == 0) a.buffer_bytes = (size_t)prop.l2CacheSize / 4;
    size_t n = a.buffer_bytes / sizeof(unsigned);
    print_header(argc, argv, a, n);

    // With a large stride the walk repeats after cycle_length elements (see
    // the '#' metadata line). Repeating is harmless here -- the buffer is
    // meant to be resident -- but a very short cycle fits in L1, and then
    // this measures L1, not L2. Watch the cycle_length line.
    std::mt19937 rng(a.seed);
    std::vector<unsigned> hbuf(n);
    make_pattern(hbuf, rng, a.stride_bytes);
    unsigned* dbuf;
    CUDA_CHECK(cudaMalloc(&dbuf, n * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(dbuf, hbuf.data(), n * sizeof(unsigned),
                          cudaMemcpyHostToDevice));

    unsigned long long* dsink;
    CUDA_CHECK(cudaMalloc(&dsink, sizeof(unsigned long long)));
    Result* out;
    CUDA_CHECK(cudaMallocManaged(&out, sizeof(Result)));

    // CUDA loads a kernel lazily the first time it is launched
    // (CUDA_MODULE_LOADING=LAZY is the default), and that first load flushes
    // L2. If the chase kernel's first launch came AFTER the warm sweep, the
    // sweep would be undone and the first ~50 repetitions would read DRAM
    // while the cache refilled — a cold start that a median silently hides.
    // So: launch the chase kernel once, untimed, BEFORE warming.
    gmem_chase<<<1, 32>>>(dbuf, 0, 0, 0, out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Warm once; the repetitions keep the buffer resident.
    warm_l2<<<prop.multiProcessorCount * 4, 256>>>(dbuf, n, dsink);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // A fresh random start each repetition, so consecutive reps do not walk
    // the identical address sequence.
    std::uniform_int_distribution<unsigned> start(0, (unsigned)n - 1);

    std::vector<double> cpl, npl;
    for (int rep = 0; rep < a.reps; rep++) {
        gmem_chase<<<1, 32>>>(dbuf, start(rng), a.warmup, a.steps, out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        print_row("l2", 0, 0, 0, 32, a.steps, a.buffer_bytes, a.stride_bytes,
                  a.seed, rep, *out);
        cpl.push_back((double)out->cycles / a.steps);
        npl.push_back((double)out->ns / a.steps);
    }
    print_summary("l2", cpl, npl);

    cudaFree(dbuf);
    cudaFree(dsink);
    cudaFree(out);
    return 0;
}
