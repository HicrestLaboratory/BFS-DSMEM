// l1.cu — latency of global memory that is resident in the SM's L1 cache.
//
// The chase buffer is far smaller than L1 (default 32 KiB against an
// effective ~88 KiB for a kernel that uses no shared memory — see
// ../extra/l1_carveout.cu and the chain curve for where that number comes
// from), so once it has been touched once, every load is an L1 hit.
//
// The one thing that differs from l2.cu: L1 cannot be warmed from the host.
// It is private to one SM and it is invalidated between kernel launches, so
// a separate warm-up kernel would warm some other SM's L1 and then lose it
// anyway. The only warm-up that works is the in-kernel untimed pass, by the
// SAME thread on the SAME SM, immediately before the timed one — which is
// exactly what warm_then_chase() does. That pass must cover the whole buffer
// at least once, so the program raises --warmup on its own when
// warmup x steps < elements (and says so on stderr).
//
// Check it yourself: the number must be flat across repetitions AND the
// buffer must sit well below the L1 knee; a buffer above ~88 KiB measures L2
// under the name "l1". `chain` sweeps that boundary explicitly.
//
// Build: make l1
// Run:   ./l1 [--buffer-bytes N (default 32768)]
//             [--reps 101] [--steps 16384] [--seed 42] [--warmup 1]

#include "../common.cuh"

// A single thread chases; the block is one warp so nothing else competes.
__global__ void gmem_chase(const unsigned* __restrict__ buf, unsigned start,
                           int warmup, int steps, Result* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        warm_then_chase(buf, start, warmup, steps, out);
}

int main(int argc, char** argv) {
    Args a;
    parse_args(argc, argv, &a, "l1",
               "  --buffer-bytes N  chase buffer size                   (default 32768 = 32 KiB)\n");
    if (a.buffer_bytes == 0) a.buffer_bytes = 32 * 1024;
    size_t n = a.buffer_bytes / sizeof(unsigned);

    // The in-kernel warm-up is the ONLY thing that puts the buffer in L1, so
    // it has to walk every element at least once before the timed pass. One
    // lap of `steps` loads covers steps elements; a bigger buffer needs more
    // laps, and the program adds them itself rather than start cold.
    size_t elems = cycle_length(n, a.stride_bytes);
    int laps_needed = (int)((elems + a.steps - 1) / a.steps);
    if (a.warmup < laps_needed) {
        fprintf(stderr, "l1: note: raised --warmup from %d to %d so the untimed pass "
                        "covers all %zu elements before timing.\n",
                a.warmup, laps_needed, elems);
        a.warmup = laps_needed;
    }
    if (a.buffer_bytes > 64 * 1024)
        fprintf(stderr, "l1: warning: %zu KiB is close to or above the ~88 KiB effective L1; "
                        "the result may be L2 latency. Use ./chain to see the knee.\n",
                a.buffer_bytes >> 10);
    print_header(argc, argv, a, n);

    std::mt19937 rng(a.seed);
    std::vector<unsigned> hbuf(n);
    make_pattern(hbuf, rng, a.stride_bytes);
    unsigned* dbuf;
    CUDA_CHECK(cudaMalloc(&dbuf, n * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(dbuf, hbuf.data(), n * sizeof(unsigned),
                          cudaMemcpyHostToDevice));

    Result* out;
    CUDA_CHECK(cudaMallocManaged(&out, sizeof(Result)));

    // No lazy-loading dummy launch here (compare l2.cu): the first launch's
    // L2 flush is irrelevant, because every repetition warms its own L1 from
    // scratch anyway. Whether the buffer arrives from L2 or DRAM during the
    // untimed pass does not change the timed one.

    // A fresh random start each repetition, so consecutive reps do not walk
    // the identical address sequence.
    std::uniform_int_distribution<unsigned> start(0, (unsigned)n - 1);

    std::vector<double> cpl, npl;
    for (int rep = 0; rep < a.reps; rep++) {
        gmem_chase<<<1, 32>>>(dbuf, start(rng), a.warmup, a.steps, out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        print_row("l1", 0, 0, 0, 32, a.steps, a.buffer_bytes, a.stride_bytes,
                  a.seed, rep, *out);
        cpl.push_back((double)out->cycles / a.steps);
        npl.push_back((double)out->ns / a.steps);
    }
    print_summary("l1", cpl, npl);

    cudaFree(dbuf);
    cudaFree(out);
    return 0;
}
