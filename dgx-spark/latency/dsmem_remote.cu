// dsmem_remote.cu — latency of ANOTHER block's shared memory (DSMEM).
//
// Rank 0 of a cluster chases the shared-memory buffer of rank `distance`,
// obtained with map_shared_rank. Every load crosses the SM-to-SM network.
// --distance 0 is allowed as a control (it is the self-rank case of
// smem_cluster_local, measured by the same binary).
//
// Build: make dsmem_remote
// Run:   ./dsmem_remote [--cluster-size 2] [--distance 1] [--block-size 128]
//                       [--reps 101] [--steps 16384] [--seed 42] [--warmup 1]

#include <cooperative_groups.h>

#include "common.cuh"

namespace cg = cooperative_groups;

__global__ void dsmem_remote_chase(const unsigned* __restrict__ perm, int target,
                                   int warmup, int steps, Result* out) {
    __shared__ unsigned sbuf[SBUF];
    cg::cluster_group cluster = cg::this_cluster();

    // Every block fills ITS OWN buffer with the same permutation, so rank 0
    // can chase any of them and see the same cycle.
    for (int i = threadIdx.x; i < SBUF; i += blockDim.x) sbuf[i] = perm[i];
    __syncthreads();
    cluster.sync();  // the target's buffer must be filled before we read it

    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        const unsigned* buf = cluster.map_shared_rank((const unsigned*)sbuf, target);
        warm_then_chase(buf, 0, warmup, steps, out);
    }

    // ESSENTIAL: a block's shared memory is released when the block exits.
    // Without this barrier the target block would finish immediately, and
    // rank 0 would be chasing freed memory — a wrong result or a fault.
    cluster.sync();
}

int main(int argc, char** argv) {
    Args a;
    parse_args(argc, argv, &a, "dsmem_remote",
               "  --block-size N    threads per block                   (default 128)\n"
               "  --cluster-size N  blocks per cluster, 2..12           (default 2)\n"
               "  --distance N      target rank, 0..cluster-size-1     (default 1)\n");
    if (a.cluster_size < 2 || a.cluster_size > 12) {
        fprintf(stderr, "dsmem_remote: --cluster-size must be 2..12 on GB10\n");
        return 1;
    }
    if (a.distance < 0 || a.distance >= a.cluster_size) {
        fprintf(stderr, "dsmem_remote: --distance must be 0..%d for cluster size %d\n",
                a.cluster_size - 1, a.cluster_size);
        return 1;
    }
    print_header(argc, argv, a, SBUF);

    std::mt19937 rng(a.seed);
    std::vector<unsigned> hperm(SBUF);
    make_pattern(hperm, rng, a.stride_bytes);
    unsigned* dperm;
    CUDA_CHECK(cudaMalloc(&dperm, SBUF * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(dperm, hperm.data(), SBUF * sizeof(unsigned),
                          cudaMemcpyHostToDevice));

    Result* out;
    CUDA_CHECK(cudaMallocManaged(&out, sizeof(Result)));

    if (a.cluster_size > 8) allow_big_clusters((const void*)dsmem_remote_chase);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(a.cluster_size, 1, 1);
    cfg.blockDim = dim3(a.block_size, 1, 1);
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = a.cluster_size;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;

    std::vector<double> cpl, npl;
    for (int rep = 0; rep < a.reps; rep++) {
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, dsmem_remote_chase, dperm,
                                      a.distance, a.warmup, a.steps, out));
        CUDA_CHECK(cudaDeviceSynchronize());
        print_row("dsmem_remote", a.cluster_size, a.distance, 1, a.block_size,
                  a.steps, 0, a.stride_bytes, a.seed, rep, *out);
        cpl.push_back((double)out->cycles / a.steps);
        npl.push_back((double)out->ns / a.steps);
    }
    print_summary("dsmem_remote", cpl, npl);

    cudaFree(dperm);
    cudaFree(out);
    return 0;
}
