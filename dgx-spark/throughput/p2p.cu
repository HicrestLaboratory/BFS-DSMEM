// p2p.cu — moving `--bytes` bytes from one block to another of the same
// cluster: through DSMEM (push or pull) or through global memory. The kernels
// are those of the paper's copy.cu (paper-benchmark/copy.cu); only the host
// side changes: flags instead of a fixed sweep, and one CSV row per launch.
//
// THE METHODS (--method, --src S --dst D: the data goes from rank S to rank D)
//   dsmem-push  rank 1 writes into rank 0's shared memory, rank 0 reads it
//               locally                                        (--src 1 --dst 0)
//   dsmem-pull  rank 1 writes its own shared memory, rank 0 reads it
//               remotely                                       (--src 1 --dst 0)
//               or the same between ranks 11 and 10 of a 12-block cluster,
//               the fastest DSMEM pair                         (--src 11 --dst 10)
//   gmem-ldst   rank 1 writes a global buffer, rank 0 reads it (--src 1 --dst 0)
//
// THE MEASUREMENT
//   Every block reads clock64() after a first cluster.sync() and again at
//   the end: the sender writes, cluster.sync(), the receiver reads. The CSV
//   gives the time of the receiver and of the sender, and their max: the
//   transfer. 4-byte (int) accesses; --bytes is a multiple of 4, up to
//   32768. --warmup untimed launches come first.
//
// Build: make p2p
// Run:   build/p2p --method dsmem-pull --src 11 --dst 10 --bytes 4096 --block-size 256

#include <array>
#include <cooperative_groups.h>
#include <iostream>
#include <vector>
#include <fstream>

#include "../common.cuh"

// The largest transfer, and the size of every buffer: 8192 ints, as in the
// paper.
constexpr int MAX_BYTES = 32768;

namespace cg = cooperative_groups;

// Small global buffer used to keep the computed results observable.
__device__ int d_data[100];

// Global-memory buffer used by the Global kernel.
__device__ int d_p_global_share[MAX_BYTES / sizeof(int)];

// Managed array storing the elapsed time measured by each block/rank, in SM
// clock cycles.
__managed__ long long d_time[2];


// -----------------------------------------------------------------------------
// PUSH: remote writes through Distributed Shared Memory (DSMEM)
//
// Block 1 writes data directly into block 0's shared memory.
// Block 0 then reads the data from its own local shared memory.
//
// This measures a push-style transfer: the sender chooses the destination
// memory location and writes the data there.
// -----------------------------------------------------------------------------
template <typename T>
__global__ __cluster_dims__(2, 1, 1)
void push(int bytes) {
    cg::cluster_group cluster = cg::this_cluster();

    // Per-thread accumulator. The register qualifier requests register storage.
    int register counter = 0;

    // Each block owns its own shared-memory buffer.
    __shared__ int p_sdata[MAX_BYTES / sizeof(int)];

    // Interpret the shared-memory buffer as an array of T elements.
    T* sdata = (T*)p_sdata;

    // Map the shared-memory buffer of rank 0.
    // Block 1 can use this pointer to write into block 0's shared memory.
    T* cluster_smem = cluster.map_shared_rank(sdata, 0);

    // Convert the requested transfer size from bytes to elements.
    int max_index = bytes / sizeof(T);

    // Ensure that all blocks have initialized their shared-memory buffers
    // before the timed transfer begins.
    cluster.sync();

    // Start timing. Each block records its own timestamp.
    long long start = clock64();

    // Block 1 is the sender: write directly into block 0's shared memory.
    if (blockIdx.x == 1) {
        for (int i = threadIdx.x; i < max_index; i += blockDim.x) {
            cluster_smem[i] = i;
        }
    }

    // Ensure that all remote writes are complete and visible before block 0
    // starts reading the received data.
    cluster.sync();

    // Block 0 is the receiver: read from its own local shared memory.
    if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < max_index; i += blockDim.x) {
            counter += sdata[i];
        }
    }

    // Ensure that all threads in each block have finished their work
    // before recording the end timestamp.
    __syncthreads();

    long long end = clock64();
    long long duration = end - start;

    // Store the elapsed time measured by this block.
    if (threadIdx.x == 0) {
        d_time[blockIdx.x] = duration;
    }

    // Consume the accumulated result so that the compiler cannot simply
    // eliminate the data-reading loop as dead code.
    atomicAdd(&d_data[0], counter);

    // Ensure that all blocks finish before the kernel exits.
    cluster.sync();
}


// -----------------------------------------------------------------------------
// PULL: remote reads through Distributed Shared Memory (DSMEM)
//
// Block 1 writes data into its own shared memory.
// Block 0 reads that data directly from block 1's shared memory.
//
// This measures a pull-style transfer: the receiver accesses the sender's
// memory rather than having the sender write into the receiver's memory.
// -----------------------------------------------------------------------------
template <typename T>
__global__ __cluster_dims__(2, 1, 1)
void pull(int bytes) {
    cg::cluster_group cluster = cg::this_cluster();

    int register counter = 0;

    // Local shared memory owned by each block.
    __shared__ int p_sdata[MAX_BYTES / sizeof(int)];

    T* sdata = (T*)p_sdata;
    int max_index = bytes / sizeof(T);

    // Map block 1's shared memory.
    // Block 0 can read remote_smem to access data stored by block 1.
    T* cluster_smem = cluster.map_shared_rank(sdata, 1);

    // Synchronize the cluster before starting the transfer.
    cluster.sync();

    long long start = clock64();

    // Block 1 is the sender: populate its own local shared memory.
    if (blockIdx.x == 1) {
        for (int i = threadIdx.x; i < max_index; i += blockDim.x) {
            sdata[i] = i;
        }
    }

    // Make sure the sender's writes are complete and visible before
    // the receiver begins accessing the remote shared memory.
    cluster.sync();

    // Block 0 is the receiver: read directly from block 1's shared memory.
    if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < max_index; i += blockDim.x) {
            counter += cluster_smem[i];
        }
    }

    // Wait for all threads in the block to finish their reads.
    __syncthreads();

    long long end = clock64();
    long long duration = end - start;

    if (threadIdx.x == 0) {
        d_time[blockIdx.x] = duration;
    }

    // Keep the computed result observable to prevent dead-code elimination.
    atomicAdd(&d_data[0], counter);

    cluster.sync();
}


// -----------------------------------------------------------------------------
// PULL V2: remote reads between ranks 11 and 10 in a 12-block cluster
//
// Intended access pattern:
//   - Rank 11 writes data into its own shared memory.
//   - Rank 10 reads that data from rank 11's shared memory.
//
// -----------------------------------------------------------------------------
template <typename T>
__global__ __cluster_dims__(12, 1, 1)
void pull_v2(int bytes) {
    cg::cluster_group cluster = cg::this_cluster();

    int counter = 0;

    // Each block has its own shared-memory buffer.
    __shared__ int p_sdata[MAX_BYTES / sizeof(int)];

    T* sdata = reinterpret_cast<T*>(p_sdata);
    const int max_index = bytes / sizeof(T);

    // Obtain the block's rank within the cluster.
    const int rank = cluster.block_rank();

    // Map rank 11's shared memory: rank 10 reads it remotely. (The paper
    // mapped rank 10, i.e. rank 10's OWN memory: no DSMEM access at all.)
    T* remote_sdata = cluster.map_shared_rank(sdata, 11);

    // Synchronize all blocks before starting the measurement.
    cluster.sync();

    const long long start = clock64();

    // Rank 11 initializes its own shared-memory buffer.
    if (rank == 11) {
        for (int i = threadIdx.x; i < max_index; i += blockDim.x) {
            sdata[i] = static_cast<T>(i);
        }
    }

    // Ensure that rank 11's writes are complete before the receiver reads.
    cluster.sync();

    // Rank 10 reads rank 11's shared memory through the mapped pointer.
    if (rank == 10) {
        for (int i = threadIdx.x; i < max_index; i += blockDim.x) {
            counter += remote_sdata[i];
        }
    }

    // Wait until all threads in each block finish before recording the time.
    __syncthreads();

    const long long end = clock64();

    // Store timing for the two ranks of interest.
    if (threadIdx.x == 0) {
        if (rank == 10) d_time[0] = end - start;
        if (rank == 11) d_time[1] = end - start;
    }

    // Prevent the read results from being optimized away.
    atomicAdd(&d_data[0], counter);

    cluster.sync();
}


// -----------------------------------------------------------------------------
// GLOBAL: transfer through global memory
//
// Block 1 writes the data into a global-memory buffer.
// After synchronization, block 0 reads the data from global memory.
//
// This provides a baseline for comparison with the DSMEM push and pull
// implementations.
// -----------------------------------------------------------------------------
template <typename T>
__global__ __cluster_dims__(2, 1, 1)
void global(int bytes) {
    cg::cluster_group cluster = cg::this_cluster();

    int register counter = 0;

    // Unused shared-memory buffer, retained from the original implementation.
    [[maybe_unused]] __shared__ int sdata[MAX_BYTES / sizeof(int)];

    // Interpret the global buffer as an array of T elements.
    T* d_global_share = (T*)d_p_global_share;

    int max_index = bytes / sizeof(T);

    cluster.sync();

    long long start = clock64();

    // Block 1 writes the data into global memory.
    if (blockIdx.x == 1) {
        for (int i = threadIdx.x; i < max_index; i += blockDim.x) {
            d_global_share[i] = i;
        }
    }

    // Ensure that the global-memory writes are complete before block 0 reads.
    cluster.sync();

    // Block 0 reads the data from global memory.
    if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < max_index; i += blockDim.x) {
            counter += d_global_share[i];
        }
    }

    __syncthreads();

    long long end = clock64();
    long long duration = end - start;

    if (threadIdx.x == 0) {
        d_time[blockIdx.x] = duration;
    }

    // Prevent the result from being optimized away.
    atomicAdd(&d_data[0], counter);

    cluster.sync();
}


// -----------------------------------------------------------------------------
// HOST
// -----------------------------------------------------------------------------
// One launch of `blocks` blocks (one cluster), waited for.
void launch(void (*kernel)(int), int blocks, const Args& a) {
    kernel<<<blocks, a.block_size>>>(a.bytes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc, char** argv) {
    Args a;
    a.method = 0;   // dsmem-pull
    a.src = 1;
    a.dst = 0;
    a.block_size = 256;
    a.reps = 21;
    a.warmup = 10;
    parse_args(argc, argv, &a, "p2p",
               "  --method M        dsmem-push | dsmem-pull | gmem-ldst   (default dsmem-pull)\n"
               "  --src S --dst D   1 0, or (dsmem-pull only) 11 10       (default 1 0)\n"
               "  --bytes N         bytes moved, a multiple of 4, 4..32768 (default 128)\n"
               "  --block-size N    threads per block, 1..1024           (default 256)\n"
               "  --reps N          timed launches, one CSV row each     (default 21)\n"
               "  --warmup N        untimed launches before them         (default 10)\n");

    // ---- check the arguments, pick the kernel
    const char* method = TRANSFER_METHOD[a.method];
    const bool pair_0_1 = a.src == 1 && a.dst == 0, pair_10_11 = a.src == 11 && a.dst == 10;
    void (*kernel)(int) = nullptr;
    int blocks = 2;
    if (strcmp(method, "dsmem-push") == 0 && pair_0_1) kernel = push<int>;
    if (strcmp(method, "dsmem-pull") == 0 && pair_0_1) kernel = pull<int>;
    if (strcmp(method, "gmem-ldst") == 0 && pair_0_1)  kernel = global<int>;
    if (strcmp(method, "dsmem-pull") == 0 && pair_10_11) { kernel = pull_v2<int>; blocks = 12; }
    if (!kernel) {
        fprintf(stderr, "p2p: --method must be dsmem-push, dsmem-pull or gmem-ldst with --src 1 "
                        "--dst 0, or dsmem-pull with --src 11 --dst 10\n");
        return 1;
    }
    if (a.bytes < 4 || a.bytes > MAX_BYTES || a.bytes % 4) {
        fprintf(stderr, "p2p: --bytes must be a multiple of 4, 4..%d\n", MAX_BYTES);
        return 1;
    }
    if (a.block_size < 1 || a.block_size > 1024) {
        fprintf(stderr, "p2p: --block-size must be 1..1024\n");
        return 1;
    }
    // PullV2's cluster of 12 is larger than the portable 8.
    if (blocks == 12) allow_big_clusters((const void*)kernel);

    char extra[200];
    snprintf(extra, sizeof extra, "# method: %s\n# src: %d\n# dst: %d\n# bytes: %d\n# cluster_size: %d\n",
             method, a.src, a.dst, a.bytes, blocks);
    print_header(argc, argv, a, 0, extra,
                 "benchmark,method,src,dst,bytes,block_size,seed,rep,cycles_receiver,cycles_sender,cycles\n");

    for (int w = 0; w < a.warmup; w++) launch(kernel, blocks, a);
    std::vector<double> cycles;
    for (int rep = 0; rep < a.reps; rep++) {
        launch(kernel, blocks, a);
        // d_time[0]: the receiver (block 0, or rank 10); d_time[1]: the sender
        const long long transfer = std::max(d_time[0], d_time[1]);
        printf("p2p,%s,%d,%d,%d,%d,%d,%d,%lld,%lld,%lld\n", method, a.src, a.dst, a.bytes, a.block_size,
               a.seed, rep, d_time[0], d_time[1], transfer);
        cycles.push_back(transfer);
    }

    fprintf(stderr, "p2p %s %d -> %d: %d bytes, %d threads per block\n", method, a.src, a.dst,
            a.bytes, a.block_size);
    print_stats_line("cycles", cycles);
    return 0;
}
