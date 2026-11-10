// barriers-gmem.cu — how long each thread waits in ONE barrier, with a load from
// global memory that misses L2 after the barrier, so that the clock read cannot
// run early. The same as barriers-shared.cu, but every thread of every launch
// loads a different random word of a 1 GiB buffer, much larger than L2, with .cg
// so that it skips L1: almost every load goes to DRAM. It can not be one word for
// everybody as in barriers-shared.cu, because after the first load that word
// would be in L2.
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
//       start = clock64();  barrier;  v = load from DRAM;  end = clock64();
//   --warmup untimed launches come first: without them the first launches
//   are off (cluster.sync(): ~3850, then ~840, then from the 8th launch on
//   ~470 cycles).
//
//   Why the load? See barriers-shared.cu: the clock read behind a load cannot
//   run before the barrier is over. The price is the load itself, here a DRAM
//   access, more when many warps load at once.
//
//   The value must be used BEFORE the second clock read. A load from global
//   memory (LDG) only issues the request: the warp goes on, and a clock read
//   right behind it does not wait for the data. Using the value makes it wait.
//
// Build: make barriers-gmem
// Run:   build/barriers-gmem --barrier grid --blocks 12 --block-size 128 --reps 11

#include <cooperative_groups.h>

#include "../common.cuh"   // CUDA_CHECK, Args, print_header, print_stats_line, ...

constexpr size_t BUFFER_WORDS = size_t(1) << 28;   // 1 GiB of 4-byte words; L2: 24 MiB on GB10, 50 MB on H100

// A word of the buffer, different for every thread of every launch. Computed
// before the first clock read, so it is not timed.
__device__ __forceinline__ unsigned random_word(unsigned seed) {
  const unsigned id = blockIdx.x * 1024 + threadIdx.x + seed * 12288;
  return (id * 2654435761u) >> 4;   // Knuth's multiplicative hash, 28 bits
}

__managed__ long long arrive_time[12][1024];

__global__ void barrier_kernel(const unsigned* buf, unsigned seed) {
  const unsigned* p = buf + random_word(seed);
  unsigned v = 0;
  long long start = clock64();
  __syncthreads();
  asm volatile("ld.global.cg.u32 %0, [%1];"            // a load that misses L2...
              : "=r"(v)
              : "l"(p)
              : "memory");
  if (v == 12345) arrive_time[0][0] = 0;   // never true: makes the warp wait for the load
  long long end = clock64();
  long long duration = end - start + (v == 12345);   // ...whose value is used, so it isn't removed

  arrive_time[blockIdx.x][threadIdx.x] = duration;
}

__global__ void cluster_sync_kernel(const unsigned* buf, unsigned seed) {
  const unsigned* p = buf + random_word(seed);
  cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
  unsigned v = 0;
  long long start = clock64();
  cluster.sync();
  asm volatile("ld.global.cg.u32 %0, [%1];"            // a load that misses L2...
              : "=r"(v)
              : "l"(p)
              : "memory");
  if (v == 12345) arrive_time[0][0] = 0;   // never true: makes the warp wait for the load
  long long end = clock64();
  long long duration = end - start + (v == 12345);   // ...whose value is used, so it isn't removed

  arrive_time[blockIdx.x][threadIdx.x] = duration;
}

// cluster.sync() with a RELAXED arrive: the blocks wait for each other, but no
// write is made visible (no MEMBAR). The difference to cluster_sync_kernel is
// the price of the release.
__global__ void cluster_relaxed_kernel(const unsigned* buf, unsigned seed) {
  const unsigned* p = buf + random_word(seed);
  unsigned v = 0;
  long long start = clock64();
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\t"
               "barrier.cluster.wait.aligned;" ::: "memory");
  asm volatile("ld.global.cg.u32 %0, [%1];"            // a load that misses L2...
              : "=r"(v)
              : "l"(p)
              : "memory");
  if (v == 12345) arrive_time[0][0] = 0;   // never true: makes the warp wait for the load
  long long end = clock64();
  long long duration = end - start + (v == 12345);   // ...whose value is used, so it isn't removed

  arrive_time[blockIdx.x][threadIdx.x] = duration;
}

__global__ void grid_sync_kernel(const unsigned* buf, unsigned seed) {
  const unsigned* p = buf + random_word(seed);
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  unsigned v = 0;

  long long start = clock64();
  grid.sync();
  asm volatile("ld.global.cg.u32 %0, [%1];"            // a load that misses L2...
              : "=r"(v)
              : "l"(p)
              : "memory");
  if (v == 12345) arrive_time[0][0] = 0;   // never true: makes the warp wait for the load
  long long end = clock64();
  arrive_time[blockIdx.x][threadIdx.x] = end - start + (v == 12345);   // ...whose value is used, so it isn't removed
}

// One launch: cooperative for grid.sync(), one cluster of --blocks otherwise.
void launch(void (*kernel)(const unsigned*, unsigned), bool grid, const Args& a,
            const cudaLaunchConfig_t& config, const unsigned* buf, unsigned seed) {
  void* args[] = {&buf, &seed};
  if (grid) CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel, a.blocks, a.block_size, args));
  else      CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, buf, seed));
  CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc, char** argv) {
  Args a;
  a.blocks = 1;
  a.reps = 11;
  a.warmup = 10;
  parse_args(argc, argv, &a, "barriers-gmem",
             "  --barrier B       syncthreads | cluster | cluster-relaxed | grid\n"
             "                                                          (default cluster)\n"
             "  --block-size N    threads per block, 1..1024            (default 128)\n"
             "  --blocks N        blocks, 1..12 (cluster: one cluster of N) (default 1)\n"
             "  --reps N          timed launches                        (default 11)\n"
             "  --warmup N        untimed launches before them          (default 10)\n");

  const char* barrier = a.barrier ? a.barrier : "cluster";
  void (*kernel)(const unsigned*, unsigned) = nullptr;
  if (strcmp(barrier, "syncthreads") == 0) kernel = barrier_kernel;
  if (strcmp(barrier, "cluster") == 0)     kernel = cluster_sync_kernel;
  if (strcmp(barrier, "cluster-relaxed") == 0) kernel = cluster_relaxed_kernel;
  if (strcmp(barrier, "grid") == 0)        kernel = grid_sync_kernel;
  if (!kernel) {
    fprintf(stderr, "barriers-gmem: --barrier must be syncthreads, cluster, cluster-relaxed or grid\n");
    return 1;
  }
  if (a.block_size < 1 || a.block_size > 1024 || a.blocks < 1 || a.blocks > 12) {
    fprintf(stderr, "barriers-gmem: --block-size must be 1..1024 and --blocks 1..12\n");
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

  unsigned* buf;
  CUDA_CHECK(cudaMalloc(&buf, BUFFER_WORDS * sizeof(unsigned)));
  CUDA_CHECK(cudaMemset(buf, 0, BUFFER_WORDS * sizeof(unsigned)));
  unsigned seed = 0;   // every launch loads other words
  for (int w = 0; w < a.warmup; w++) launch(kernel, grid, a, config, buf, seed++);
  std::vector<double> all;
  for (int rep = 0; rep < a.reps; rep++) {
    launch(kernel, grid, a, config, buf, seed++);
    for (int i = 0; i < a.blocks; i++) {
      for (int j = 0; j < a.block_size; j++) {
        printf("barriers-gmem,%s,%d,%d,%d,%d,%d,%d,%lld\n", barrier, a.block_size, a.blocks, a.seed,
               rep, i, j, arrive_time[i][j]);
        all.push_back(arrive_time[i][j]);
      }
    }
  }

  fprintf(stderr, "barriers-gmem %s: %d blocks x %d threads\n", barrier, a.blocks, a.block_size);
  print_stats_line("cycles", all);
  return 0;
}
