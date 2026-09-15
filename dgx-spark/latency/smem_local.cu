// smem_local.cu — latency of a block's OWN shared memory, no cluster.
//
// This is the baseline every other number is compared against: the fastest
// memory an SM has, accessed the simplest way.
//
// Build: make smem_local        (nvcc -O3 -arch=sm_121)
// Run:   ./smem_local [--reps 101] [--steps 16384] [--seed 42] [--warmup 1]
//                     [--block-size 128]

#include "common.cuh"

// Every thread helps copy the permutation into shared memory; then thread 0
// alone chases it. The other threads just wait at the end of the block.
__global__ void smem_local_chase(const unsigned* __restrict__ perm, int warmup,
                                 int steps, Result* out) {
    __shared__ unsigned sbuf[SBUF];
    for (int i = threadIdx.x; i < SBUF; i += blockDim.x) sbuf[i] = perm[i];
    __syncthreads();  // the whole buffer must be in place before the chase
    if (threadIdx.x == 0) warm_then_chase(sbuf, 0, warmup, steps, out);
}

int main(int argc, char** argv) {
    Args a;
    parse_args(argc, argv, &a, "smem_local",
               "  --block-size N    threads per block                   (default 128)\n");
    print_header(argc, argv, a, SBUF);

    // The permutation is built on the host and copied to global memory;
    // the kernel loads it into shared memory itself.
    std::mt19937 rng(a.seed);
    std::vector<unsigned> hperm(SBUF);
    make_pattern(hperm, rng, a.stride_bytes);
    unsigned* dperm;
    CUDA_CHECK(cudaMalloc(&dperm, SBUF * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(dperm, hperm.data(), SBUF * sizeof(unsigned),
                          cudaMemcpyHostToDevice));

    // Managed memory: the kernel writes the Result, the host reads it back
    // after cudaDeviceSynchronize() without an explicit copy.
    Result* out;
    CUDA_CHECK(cudaMallocManaged(&out, sizeof(Result)));

    std::vector<double> cpl, npl;  // per-load cycles / ns, one entry per rep
    for (int rep = 0; rep < a.reps; rep++) {
        smem_local_chase<<<1, a.block_size>>>(dperm, a.warmup, a.steps, out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        print_row("smem_local", 0, 0, 0, a.block_size, a.steps, 0,
                  a.stride_bytes, a.seed, rep, *out);
        cpl.push_back((double)out->cycles / a.steps);
        npl.push_back((double)out->ns / a.steps);
    }
    print_summary("smem_local", cpl, npl);

    cudaFree(dperm);
    cudaFree(out);
    return 0;
}
