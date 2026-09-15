// dram.cu — latency of global memory that is NOT in any cache.
//
// The chase buffer is far larger than L2 (default 1 GiB against 24 MiB) and
// is never warmed. Each repetition starts at a fresh random point of the
// cycle, so it walks addresses no earlier repetition has touched: every load
// misses L2 and goes to DRAM.
//
// Build: make dram
// Run:   ./dram [--buffer-bytes N (default 1073741824)]
//               [--reps 101] [--steps 16384] [--seed 42] [--warmup 1]
//
// Note on --warmup: the untimed pass and the timed pass walk DIFFERENT
// segments of the cycle (the timed pass continues where the warm-up stops),
// so warming up does not turn this into an L2 measurement.

#include "common.cuh"

__global__ void gmem_chase(const unsigned* __restrict__ buf, unsigned start,
                           int warmup, int steps, Result* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        warm_then_chase(buf, start, warmup, steps, out);
}

int main(int argc, char** argv) {
    Args a;
    parse_args(argc, argv, &a, "dram",
               "  --buffer-bytes N  chase buffer size                   (default 1073741824 = 1 GiB)\n");
    if (a.buffer_bytes == 0) a.buffer_bytes = 1ull << 30;  // 1 GiB
    size_t n = a.buffer_bytes / sizeof(unsigned);

    // A stride walk repeats after cycle_length elements. If the warm-up plus
    // the timed walk do not fit inside ONE cycle, the timed pass re-reads
    // addresses the warm-up just pulled into L2, and this program would
    // silently report an L2 number under the name "dram". Refuse instead.
    size_t walk = (size_t)(a.warmup + 1) * a.steps;
    size_t cyc = cycle_length(n, a.stride_bytes);
    if (walk > cyc) {
        fprintf(stderr,
                "dram: a walk of %zu loads would wrap a cycle of only %zu elements "
                "(stride %d B over %zu B). Lower --steps/--warmup or raise "
                "--buffer-bytes.\n",
                walk, cyc, a.stride_bytes, a.buffer_bytes);
        return 1;
    }
    print_header(argc, argv, a, n);

    // Building a 256M-element Sattolo cycle on the host takes a few seconds;
    // that is the price of a buffer large enough to defeat a 24 MiB L2.
    std::mt19937 rng(a.seed);
    unsigned* dbuf;
    CUDA_CHECK(cudaMalloc(&dbuf, n * sizeof(unsigned)));
    {
        std::vector<unsigned> hbuf(n);
        make_pattern(hbuf, rng, a.stride_bytes);
        CUDA_CHECK(cudaMemcpy(dbuf, hbuf.data(), n * sizeof(unsigned),
                              cudaMemcpyHostToDevice));
    }  // free the 1 GiB host copy before measuring

    Result* out;
    CUDA_CHECK(cudaMallocManaged(&out, sizeof(Result)));

    std::uniform_int_distribution<unsigned> start(0, (unsigned)n - 1);

    std::vector<double> cpl, npl;
    for (int rep = 0; rep < a.reps; rep++) {
        gmem_chase<<<1, 32>>>(dbuf, start(rng), a.warmup, a.steps, out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        print_row("dram", 0, 0, 0, 32, a.steps, a.buffer_bytes, a.stride_bytes,
                  a.seed, rep, *out);
        cpl.push_back((double)out->cycles / a.steps);
        npl.push_back((double)out->ns / a.steps);
    }
    print_summary("dram", cpl, npl);

    cudaFree(dbuf);
    cudaFree(out);
    return 0;
}
