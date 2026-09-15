// smem_cluster_local.cu — latency of a block's OWN shared memory when the
// block is part of a thread block cluster.
//
// Compared with smem_local, this asks whether merely being in a cluster slows
// down local shared memory. Two variants:
//   --mapped 0   rank 0 chases its buffer through the __shared__ array itself
//                (compiles to LDS, a shared-memory load)
//   --mapped 1   rank 0 chases its buffer through map_shared_rank(sbuf, 0),
//                the DSMEM interface pointed at itself (compiles to LD, a
//                generic load whose address space is resolved at run time)
// Result: the cluster costs nothing; the generic load costs ~4 cycles.
//
// Build: make smem_cluster_local
// Run:   ./smem_cluster_local [--cluster-size 2] [--mapped 1] [--block-size 128]
//                             [--reps 101] [--steps 16384] [--seed 42] [--warmup 1]

#include <cooperative_groups.h>

#include "common.cuh"

namespace cg = cooperative_groups;

__global__ void cluster_local_chase(const unsigned* __restrict__ perm,
                                    int mapped, int warmup, int steps,
                                    Result* out) {
    __shared__ unsigned sbuf[SBUF];
    cg::cluster_group cluster = cg::this_cluster();

    for (int i = threadIdx.x; i < SBUF; i += blockDim.x) sbuf[i] = perm[i];
    __syncthreads();
    cluster.sync();  // every block's buffer is ready before anyone chases

    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        if (mapped) {
            // map_shared_rank returns a GENERIC pointer into the cluster's
            // shared-memory window, so this load is an LD: the hardware must
            // work out which address space it belongs to on every access.
            const unsigned* buf = cluster.map_shared_rank((const unsigned*)sbuf, 0);
            warm_then_chase(buf, 0, warmup, steps, out);
        } else {
            // Two separate calls ON PURPOSE. Choosing the pointer with `?:`
            // would leave the compiler unable to prove it is shared memory,
            // the load would become a generic LD as well, and this path
            // would silently measure the same thing as the one above.
            warm_then_chase(sbuf, 0, warmup, steps, out);
        }
    }

    // Without this barrier the other blocks would exit as soon as they had
    // filled their buffer. Here rank 0 only reads its own memory, so it would
    // not actually break — but the same kernel shape is used in
    // dsmem_remote.cu where it is essential, and keeping it here makes the
    // two programs differ only in the target rank.
    cluster.sync();
}

int main(int argc, char** argv) {
    Args a;
    parse_args(argc, argv, &a, "smem_cluster_local",
               "  --block-size N    threads per block                   (default 128)\n"
               "  --cluster-size N  blocks per cluster, 1..12           (default 2)\n"
               "  --mapped 0|1      0 = direct pointer, 1 = map_shared_rank to self (default 1)\n");
    if (a.cluster_size < 1 || a.cluster_size > 12) {
        fprintf(stderr, "smem_cluster_local: --cluster-size must be 1..12 on GB10\n");
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

    if (a.cluster_size > 8) allow_big_clusters((const void*)cluster_local_chase);

    // A cluster launch needs cudaLaunchKernelEx with a cluster-dimension
    // attribute; the <<<>>> syntax cannot express it. The grid is exactly one
    // cluster.
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
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, cluster_local_chase, dperm,
                                      a.mapped, a.warmup, a.steps, out));
        CUDA_CHECK(cudaDeviceSynchronize());
        print_row("smem_cluster_local", a.cluster_size, 0, a.mapped,
                  a.block_size, a.steps, 0, a.stride_bytes, a.seed, rep, *out);
        cpl.push_back((double)out->cycles / a.steps);
        npl.push_back((double)out->ns / a.steps);
    }
    print_summary("smem_cluster_local", cpl, npl);

    cudaFree(dperm);
    cudaFree(out);
    return 0;
}
