// chain.cu — latency as a function of chain data volume.
//
// The same dependent chase as l2.cu / dram.cu, but meant to be run at MANY
// buffer sizes. Plotted against buffer size on a log axis, the latency is
// flat while the buffer fits in a cache level and steps up each time it
// outgrows one: L1, then L2, then DRAM. The positions of the steps ARE the
// cache sizes, measured rather than quoted. (Luo et al., "Dissecting the
// NVIDIA Hopper Architecture", Fig. 2, does this for A100 / H800 / RTX 4090.)
//
// --carveout sets the kernel's preferred L1/shared split (0 = maximise L1,
// 100 = maximise shared memory, -1 = leave the driver's default). L1 and
// shared memory are the same 128 KiB SRAM, so this moves the first step.
//
// Build: make chain
// Run:   ./chain --buffer-kib 64 [--carveout -1] [--reps 101] [--steps 16384]
//               [--seed 42] [--warmup 1]

#include "common.cuh"

__global__ void gmem_chase(const unsigned* __restrict__ buf, unsigned start,
                           int warmup, int steps, Result* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        warm_then_chase(buf, start, warmup, steps, out);
}

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
    parse_args(argc, argv, &a, "chain",
               "  --buffer-kib N    chase buffer size in KiB             (default 64)\n"
               "  --buffer-bytes N  ... or in bytes\n"
               "  --carveout N      preferred L1/shared split, -1 = driver default,\n"
               "                    0 = max L1, 100 = max shared memory (default -1)\n");
    if (a.buffer_bytes == 0) a.buffer_bytes = 64 * 1024;
    size_t n = a.buffer_bytes / sizeof(unsigned);
    print_header(argc, argv, a, n);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::mt19937 rng(a.seed);
    unsigned* dbuf;
    CUDA_CHECK(cudaMalloc(&dbuf, n * sizeof(unsigned)));
    {
        std::vector<unsigned> hbuf(n);
        make_pattern(hbuf, rng, a.stride_bytes);
        CUDA_CHECK(cudaMemcpy(dbuf, hbuf.data(), n * sizeof(unsigned),
                              cudaMemcpyHostToDevice));
    }  // free the host copy (up to 2 GiB) before measuring

    unsigned long long* dsink;
    CUDA_CHECK(cudaMalloc(&dsink, sizeof(unsigned long long)));
    Result* out;
    CUDA_CHECK(cudaMallocManaged(&out, sizeof(Result)));

    // The carveout is a per-kernel attribute; set it before the first launch.
    if (a.carveout >= 0)
        CUDA_CHECK(cudaFuncSetAttribute(
            (const void*)gmem_chase,
            cudaFuncAttributePreferredSharedMemoryCarveout, a.carveout));

    // Lazy module loading flushes L2 on a kernel's first launch (see l2.cu):
    // launch once, untimed, before warming.
    gmem_chase<<<1, 32>>>(dbuf, 0, 0, 0, out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // A buffer that fits in L2 is pulled in once by a coalesced sweep, then
    // stays resident across repetitions. A buffer larger than L2 cannot be
    // made resident, so there is nothing to warm; the in-kernel warm-up pass
    // still covers the one-off costs.
    if (a.buffer_bytes <= (size_t)prop.l2CacheSize) {
        warm_l2<<<prop.multiProcessorCount * 4, 256>>>(dbuf, n, dsink);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::uniform_int_distribution<unsigned> start(0, (unsigned)n - 1);

    std::vector<double> cpl, npl;
    for (int rep = 0; rep < a.reps; rep++) {
        gmem_chase<<<1, 32>>>(dbuf, start(rng), a.warmup, a.steps, out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        print_row("chain", 0, 0, 0, 32, a.steps, a.buffer_bytes, a.stride_bytes,
                  a.seed, rep, *out);
        cpl.push_back((double)out->cycles / a.steps);
        npl.push_back((double)out->ns / a.steps);
    }
    print_summary("chain", cpl, npl);

    cudaFree(dbuf);
    cudaFree(dsink);
    cudaFree(out);
    return 0;
}
