#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>

namespace cg = cooperative_groups;

// ------------------------------
// KERNEL
// ------------------------------
__global__ void clusterHist_kernel(
    int *bins,
    int nbins,
    int bins_per_block,
    const int *__restrict__ input,
    int array_size)
{
    extern __shared__ int smem[];

    cg::cluster_group cluster = cg::this_cluster();

    int tid = cg::this_grid().thread_rank();

    int cluster_size = cluster.dim_blocks().x;
    int cluster_rank = cluster.block_rank();

    // init shared memory
    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
        smem[i] = 0;

    cluster.sync();

    // histogram accumulation
    for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
    {
        int val = input[i];

        int binid = val;
        if (binid < 0) binid = 0;
        if (binid >= nbins) binid = nbins - 1;

        int dst_block = binid / bins_per_block;
        int dst_offset = binid % bins_per_block;

        int *dst_smem = cluster.map_shared_rank(smem, dst_block);

        atomicAdd(dst_smem + dst_offset, 1);
    }

    cluster.sync();

    // merge into global memory
    int *out = bins + cluster_rank * bins_per_block;

    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
    {
        atomicAdd(&out[i], smem[i]);
    }
}

// ------------------------------
// HOST CODE
// ------------------------------
int main()
{
    const int array_size = 1024 * 1024;
    const int nbins = 1024;
    const int threads_per_block = 256;
    const int cluster_size = 2;

    const int bins_per_block = nbins / cluster_size;

    // input: all 4
    int h_input[array_size];
    for (int i = 0; i < array_size; i++)
        h_input[i] = 4;

    int h_bins[nbins] = {0};

    int *d_input, *d_bins;

    cudaMalloc(&d_input, array_size * sizeof(int));
    cudaMalloc(&d_bins, nbins * sizeof(int));

    cudaMemcpy(d_input, h_input, array_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_bins, 0, nbins * sizeof(int));

    // launch config
    cudaLaunchConfig_t config = {0};

    config.gridDim = array_size / threads_per_block;
    config.blockDim = threads_per_block;

    config.dynamicSmemBytes = bins_per_block * sizeof(int);

    cudaFuncSetAttribute(
        clusterHist_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        config.dynamicSmemBytes
    );

    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = cluster_size;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;

    config.numAttrs = 1;
    config.attrs = &attr;

    cudaLaunchKernelEx(
        &config,
        clusterHist_kernel,
        d_bins,
        nbins,
        bins_per_block,
        d_input,
        array_size
    );

    cudaDeviceSynchronize();

    cudaMemcpy(h_bins, d_bins, nbins * sizeof(int), cudaMemcpyDeviceToHost);

    // print result
    std::cout << "Histogram:\n";
    for (int i = 0; i < nbins; i++)
        std::cout << i << ": " << h_bins[i] << "\n";

    cudaFree(d_input);
    cudaFree(d_bins);

    return 0;
}