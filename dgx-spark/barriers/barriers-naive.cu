// barriers-naive.cu — how long each thread waits in ONE barrier, timed the
// naive way. The kernels are those of paper-benchmark/paper-barriers-naive.cu
// (plus cluster-relaxed); only the host side changes: flags instead of a
// fixed sweep, and one CSV row per thread per launch.
//
// THE BARRIERS (--barrier)
//   syncthreads  __syncthreads(): the threads of one block.
//   cluster      cluster.sync(): the blocks of ONE cluster of --blocks blocks.
//   cluster-relaxed  the same barrier with a relaxed arrive: no write is made
//                visible, so the difference to `cluster` is the price of the
//                release.
//   grid         grid.sync() of cooperative groups: the --blocks blocks of a
//                cooperative launch.
//
// THE MEASUREMENT
//   Every thread of every block, once, at the start of the kernel:
//       start = clock64();  barrier;  end = clock64();
//   --warmup untimed launches come first: without them the first launches
//   are off (cluster.sync(): ~3850, then ~840, then from the 8th launch on
//   ~470 cycles).
//
//   WARNING: this does not always measure the barrier. A barrier compiles to
//   BAR.SYNC.DEFER_BLOCKING: the warp is not stopped AT the barrier but a
//   few instructions later, and a clock read is not a memory access, so
//   nothing makes it wait. The second clock64() can execute before the barrier
//   is over: e.g. grid.sync() shows ~1250 cycles on warp 0 and ~130 on the
//   other warps, which in fact wait just as long. barriers-shared.cu adds a
//   shared-memory load after the barrier to fix this.
//
// Build: make barriers-naive
// Run:   build/barriers-naive --barrier grid --blocks 12 --block-size 128 --reps 11

#include <cooperative_groups.h>

#include "../common.cuh"   // CUDA_CHECK, Args, print_header, print_stats_line, ...

__managed__ long long arrive_time[12][1024];

__global__ void barrier_kernel() {
  long long start = clock64();
  __syncthreads();
  long long end = clock64();
  long long duration = end - start;

  arrive_time[blockIdx.x][threadIdx.x] = duration;
}

__global__ void cluster_sync_kernel() {
  cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
  long long start = clock64();
  cluster.sync();
  long long end = clock64();
  long long duration = end - start;

  arrive_time[blockIdx.x][threadIdx.x] = duration;
}

// cluster.sync() with a RELAXED arrive: the blocks wait for each other, but no
// write is made visible (no MEMBAR). The difference to cluster_sync_kernel is
// the price of the release.
__global__ void cluster_relaxed_kernel() {
  long long start = clock64();
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\t"
               "barrier.cluster.wait.aligned;" ::: "memory");
  long long end = clock64();
  long long duration = end - start;

  arrive_time[blockIdx.x][threadIdx.x] = duration;
}

__global__ void grid_sync_kernel() {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  long long start = clock64();
  grid.sync();
  long long end = clock64();
  arrive_time[blockIdx.x][threadIdx.x] = end - start;
}

// One launch: cooperative for grid.sync(), one cluster of --blocks otherwise.
void launch(void (*kernel)(), bool grid, const Args& a, const cudaLaunchConfig_t& config) {
  if (grid) CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel, a.blocks, a.block_size, nullptr));
  else      CUDA_CHECK(cudaLaunchKernelEx(&config, kernel));
  CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc, char** argv) {
  Args a;
  a.blocks = 1;
  a.reps = 11;
  a.warmup = 10;
  parse_args(argc, argv, &a, "barriers-naive",
             "  --barrier B       syncthreads | cluster | cluster-relaxed | grid\n"
             "                                                          (default cluster)\n"
             "  --block-size N    threads per block, 1..1024            (default 128)\n"
             "  --blocks N        blocks, 1..12 (cluster: one cluster of N) (default 1)\n"
             "  --reps N          timed launches                        (default 11)\n"
             "  --warmup N        untimed launches before them          (default 10)\n");

  const char* barrier = a.barrier ? a.barrier : "cluster";
  void (*kernel)() = nullptr;
  if (strcmp(barrier, "syncthreads") == 0) kernel = barrier_kernel;
  if (strcmp(barrier, "cluster") == 0)     kernel = cluster_sync_kernel;
  if (strcmp(barrier, "cluster-relaxed") == 0) kernel = cluster_relaxed_kernel;
  if (strcmp(barrier, "grid") == 0)        kernel = grid_sync_kernel;
  if (!kernel) {
    fprintf(stderr, "barriers-naive: --barrier must be syncthreads, cluster, cluster-relaxed or grid\n");
    return 1;
  }
  if (a.block_size < 1 || a.block_size > 1024 || a.blocks < 1 || a.blocks > 12) {
    fprintf(stderr, "barriers-naive: --block-size must be 1..1024 and --blocks 1..12\n");
    return 1;
  }
  const bool grid = (kernel == grid_sync_kernel);

  // As in the paper: every kernel but grid runs as ONE cluster of --blocks
  // blocks; grid.sync() needs a cooperative launch instead.
  CUDA_CHECK(cudaFuncSetAttribute(barrier_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  CUDA_CHECK(cudaFuncSetAttribute(cluster_sync_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  CUDA_CHECK(cudaFuncSetAttribute(cluster_relaxed_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  cudaLaunchConfig_t config = {0};
  config.gridDim = a.blocks;
  config.blockDim = a.block_size;
  config.dynamicSmemBytes = 0;
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = a.blocks;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;
  config.attrs = attribute;
  config.numAttrs = 1;

  char extra[200];
  snprintf(extra, sizeof extra, "# barrier: %s\n# blocks: %d\n", barrier, a.blocks);
  print_header(argc, argv, a, 0, extra,
               "benchmark,barrier,block_size,blocks,seed,rep,block,thread,cycles\n");

  for (int w = 0; w < a.warmup; w++) launch(kernel, grid, a, config);
  std::vector<double> all;
  for (int rep = 0; rep < a.reps; rep++) {
    launch(kernel, grid, a, config);
    for (int i = 0; i < a.blocks; i++) {
      for (int j = 0; j < a.block_size; j++) {
        printf("barriers-naive,%s,%d,%d,%d,%d,%d,%d,%lld\n", barrier, a.block_size, a.blocks, a.seed,
               rep, i, j, arrive_time[i][j]);
        all.push_back(arrive_time[i][j]);
      }
    }
  }

  fprintf(stderr, "barriers-naive %s: %d blocks x %d threads\n", barrier, a.blocks, a.block_size);
  print_stats_line("cycles", all);
  return 0;
}
