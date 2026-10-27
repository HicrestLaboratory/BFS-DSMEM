// barriers.cu — how long ONE barrier really takes, from thread 0 of each
// block. The kernels are those of barriers.cu (the paper's naive ones,
// plus cluster-relaxed); only the output changes: one CSV row per block per
// launch, for its thread 0.
//
// Why thread 0? Warp 0 does the work of the barrier (e.g. in grid.sync() it
// is warp 0 that does the atomic in global memory and spins on it), so its
// clock read comes after the barrier is really over. The other warps may
// read the clock early (see the WARNING below), and barriers-shared.cu's fix,
// a load after the barrier, adds the load's own time, which grows when many
// warps load at once. Thread 0 needs neither.
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
//   Every thread of every block, once, at the start of the kernel (only
//   thread 0 of each block is printed):
//       start = clock();  barrier;  end = clock();
//   --warmup untimed launches come first: without them the first launches
//   are off (cluster.sync(): ~3850, then ~840, then from the 8th launch on
//   ~470 cycles).
//
//   WARNING: this does not always measure the barrier. A barrier compiles to
//   BAR.SYNC.DEFER_BLOCKING: the warp is not stopped AT the barrier but a
//   few instructions later, and a clock read is not a memory access, so
//   nothing makes it wait. The second clock() can execute before the barrier
//   is over: e.g. grid.sync() shows ~1250 cycles on warp 0 and ~130 on the
//   other warps, which in fact wait just as long. barriers-shared.cu adds a
//   shared-memory load after the barrier to fix this.
//
// Build: make barriers
// Run:   build/barriers --barrier grid --blocks 12 --block-size 128 --reps 11

#include <cooperative_groups.h>

#include "../common.cuh"   // CUDA_CHECK, Args, print_header, print_stats_line, ...

__managed__ long arrive_time[12][1024];

__global__ void barrier_kernel() {
  auto start = clock();
  __syncthreads();
  auto end = clock();
  auto duration = end - start;

  arrive_time[blockIdx.x][threadIdx.x] = duration;
}

__global__ void cluster_sync_kernel() {
  auto cluster = cooperative_groups::this_cluster();
  auto start = clock();
  cluster.sync();
  auto end = clock();
  auto duration = end - start;

  arrive_time[blockIdx.x][threadIdx.x] = duration;
}

// cluster.sync() with a RELAXED arrive: the blocks wait for each other, but no
// write is made visible (no MEMBAR). The difference to cluster_sync_kernel is
// the price of the release.
__global__ void cluster_relaxed_kernel() {
  auto start = clock();
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\t"
               "barrier.cluster.wait.aligned;" ::: "memory");
  auto end = clock();
  auto duration = end - start;

  arrive_time[blockIdx.x][threadIdx.x] = duration;
}

__global__ void grid_sync_kernel() {
  auto grid = cooperative_groups::this_grid();
  auto start = clock();
  grid.sync();
  auto end = clock();
  arrive_time[blockIdx.x][threadIdx.x] = end - start;
}

int main(int argc, char** argv) {
  Args a;
  a.blocks = 1;
  a.reps = 11;
  a.warmup = 10;
  parse_args(argc, argv, &a, "barriers",
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
    fprintf(stderr, "barriers: --barrier must be syncthreads, cluster, cluster-relaxed or grid\n");
    return 1;
  }
  if (a.block_size < 1 || a.block_size > 1024 || a.blocks < 1 || a.blocks > 12) {
    fprintf(stderr, "barriers: --block-size must be 1..1024 and --blocks 1..12\n");
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
  auto launch = [&]() {
    if (grid) CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel, a.blocks, a.block_size, nullptr));
    else      CUDA_CHECK(cudaLaunchKernelEx(&config, kernel));
    CUDA_CHECK(cudaDeviceSynchronize());
  };

  char extra[200];
  snprintf(extra, sizeof extra, "# barrier: %s\n# blocks: %d\n", barrier, a.blocks);
  print_header(argc, argv, a, 0, extra,
               "benchmark,barrier,block_size,blocks,seed,rep,block,cycles\n");

  for (int w = 0; w < a.warmup; w++) launch();
  std::vector<double> all;
  for (int rep = 0; rep < a.reps; rep++) {
    launch();
    for (int i = 0; i < a.blocks; i++) {   // thread 0 of every block only
      printf("barriers,%s,%d,%d,%d,%d,%d,%ld\n", barrier, a.block_size, a.blocks, a.seed,
             rep, i, arrive_time[i][0]);
      all.push_back(arrive_time[i][0]);
    }
  }

  fprintf(stderr, "barriers %s: %d blocks x %d threads\n", barrier, a.blocks, a.block_size);
  print_stats_line("cycles", all);
  return 0;
}
