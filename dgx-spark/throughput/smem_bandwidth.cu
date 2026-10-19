// smem_bandwidth.cu — the bandwidth of an SM's OWN shared memory.
//
// THE QUESTION
//   DSMEM moves ~10-17 GB/s into or out of one SM (cluster.cu, transfer.cu).
//   What does the same SM get from its own shared memory? That is the number
//   DSMEM has to be compared with: the speed of the memory the data comes
//   from and goes to.
//
// THE MEASUREMENT
//   One block per SM, on every SM at once (the 64 KiB buffer leaves no room
//   for a second block). Every thread does --steps accesses of --width bytes
//   to the block's own buffer, back to back, and thread 0 times the whole
//   block between two __syncthreads(). Each SM is independent, so
//     per SM:   bytes / cycles   (the hardware number: 32 banks x 4 bytes =
//               128 bytes per cycle is the most one SM's shared memory can do)
//     chip:     all bytes / the slowest block's time
//
// THE ACCESS (--op, --width, --stride)
//   read    acc ^= buf[i]   (the XOR keeps the loads from being optimized away)
//   write   buf[i] = i
//   Thread t of the block touches element t, then t + blockDim, t + 2 blockDim,
//   ... (mod the buffer): consecutive threads, consecutive elements.
//   --stride B puts consecutive threads B bytes apart instead (B a multiple of
//   --width), to see bank conflicts: with --width 4, stride 8 makes two
//   threads of a warp share each bank (a 2-way conflict), stride 128 makes all
//   32 share one bank (32-way).
//
// Build: make smem_bandwidth
// Run:   build/smem_bandwidth --op read --width 16 --reps 11

#include "../common.cuh"   // CUDA_CHECK, Args, smid(), globaltimer(), print_header, ...

constexpr int BUFFER_BYTES = 64 * 1024;   // > half of the 100 KiB per SM: one block per SM

// What one block measured.
struct BlockTime {
    long long cycles;
    unsigned long long ns;
    unsigned sm;
    unsigned sink;   // touched only if the XOR of everything read is a magic value
};

// One shared-memory access of W = 4, 8 or 16 bytes at a 32-bit shared
// address, written in PTX: `asm volatile` and `.volatile` make every access a
// real one, which the compiler can neither skip, nor keep in a register
// across passes, nor merge with another.
template <int W> __device__ __forceinline__ unsigned load_shared(uint32_t addr);
template <> __device__ __forceinline__ unsigned load_shared<4>(uint32_t addr) {
    unsigned x;
    asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(x) : "r"(addr));
    return x;
}
template <> __device__ __forceinline__ unsigned load_shared<8>(uint32_t addr) {
    unsigned x, y;
    asm volatile("ld.volatile.shared.v2.u32 {%0, %1}, [%2];" : "=r"(x), "=r"(y) : "r"(addr));
    return x ^ y;
}
template <> __device__ __forceinline__ unsigned load_shared<16>(uint32_t addr) {
    unsigned x, y, z, w;
    asm volatile("ld.volatile.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
    return x ^ y ^ z ^ w;
}
template <int W> __device__ __forceinline__ void store_shared(uint32_t addr, unsigned v);
template <> __device__ __forceinline__ void store_shared<4>(uint32_t addr, unsigned v) {
    asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"(addr), "r"(v));
}
template <> __device__ __forceinline__ void store_shared<8>(uint32_t addr, unsigned v) {
    asm volatile("st.volatile.shared.v2.u32 [%0], {%1, %1};" ::"r"(addr), "r"(v));
}
template <> __device__ __forceinline__ void store_shared<16>(uint32_t addr, unsigned v) {
    asm volatile("st.volatile.shared.v4.u32 [%0], {%1, %1, %1, %1};" ::"r"(addr), "r"(v));
}

template <int W>
__global__ void smem_bandwidth(int write, int stride_elems, int steps, int warmup, BlockTime* out) {
    extern __shared__ __align__(16) unsigned char smem[];
    const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem);
    const unsigned mask = BUFFER_BYTES / W - 1;   // elements in the buffer, minus 1

    for (int i = threadIdx.x; i < BUFFER_BYTES / 4; i += blockDim.x) ((unsigned*)smem)[i] = i;
    __syncthreads();

    // Address of access s of this thread: consecutive threads stride_elems
    // elements apart.
    auto address = [&](int s) {
        return base + (((unsigned)(s * blockDim.x + threadIdx.x) * stride_elems) & mask) * W;
    };

    unsigned acc = 0;
    long long c0 = 0;
    unsigned long long t0 = 0;
    for (int pass = 0; pass <= warmup; pass++) {   // `warmup` untimed passes, then the timed one
        __syncthreads();
        if (pass == warmup) { c0 = clock64(); t0 = globaltimer(); }
        if (write) {
#pragma unroll 8
            for (int s = 0; s < steps; s++) store_shared<W>(address(s), (unsigned)s);
        } else {
#pragma unroll 8
            for (int s = 0; s < steps; s++) acc ^= load_shared<W>(address(s));
        }
    }
    __syncthreads();   // the block is done when its slowest thread is
    const long long c1 = clock64();
    const unsigned long long t1 = globaltimer();

    if (acc == 0x9e3779b9u) atomicAdd(&out[blockIdx.x].sink, 1u);   // never true in practice
    if (threadIdx.x == 0) {
        out[blockIdx.x].cycles = c1 - c0;
        out[blockIdx.x].ns = t1 - t0;
        out[blockIdx.x].sm = smid();
    }
}

int main(int argc, char** argv) {
    Args a;
    a.block_size = 1024;
    a.steps = 4096;
    parse_args(argc, argv, &a, "smem_bandwidth",
               "  --op OP           read | write                          (default read)\n"
               "  --width B         bytes per access: 4 | 8 | 16          (default 16)\n"
               "  --stride B        bytes between consecutive threads, a multiple of\n"
               "                    --width (default --width: no bank conflicts)\n"
               "  --block-size N    threads per block                     (default 1024)\n"
               "  --steps N         accesses per thread                   (default 4096)\n");

    // ---- check the arguments
    const char* op = a.op ? a.op : "read";
    const bool write = strcmp(op, "write") == 0;
    if (!write && strcmp(op, "read") != 0) {
        fprintf(stderr, "smem_bandwidth: --op must be read or write\n");
        return 1;
    }
    if (a.width != 4 && a.width != 8 && a.width != 16) {
        fprintf(stderr, "smem_bandwidth: --width must be 4, 8 or 16\n");
        return 1;
    }
    const int stride_bytes = a.stride_bytes ? a.stride_bytes : a.width;
    if (stride_bytes % a.width || stride_bytes > BUFFER_BYTES / 32) {
        fprintf(stderr, "smem_bandwidth: --stride must be a multiple of --width, at most %d\n",
                BUFFER_BYTES / 32);
        return 1;
    }
    if (a.block_size < 32 || a.block_size > 1024 || a.block_size % 32) {
        fprintf(stderr, "smem_bandwidth: --block-size must be a multiple of 32, 32..1024\n");
        return 1;
    }
    a.stride_bytes = stride_bytes;   // so the header records what ran

    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));

    char extra[200];
    snprintf(extra, sizeof extra, "# op: %s\n# width: %d\n# buffer_bytes: %d\n# blocks: %d\n",
             op, a.width, BUFFER_BYTES, sms);
    print_header(argc, argv, a, BUFFER_BYTES / 4, extra,
                 "benchmark,op,width,stride_bytes,block_size,blocks,steps,seed,rep,"
                 "bytes_per_block,cycles,ns,bytes_per_cycle_per_sm,gbps_per_sm,gbps_chip\n");

    // ---- one block per SM
    BlockTime* out;
    CUDA_CHECK(cudaMallocManaged(&out, sms * sizeof(BlockTime)));
    auto kernel = (a.width == 4) ? smem_bandwidth<4>
                : (a.width == 8) ? smem_bandwidth<8> : smem_bandwidth<16>;
    CUDA_CHECK(cudaFuncSetAttribute((const void*)kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, BUFFER_BYTES));
    const int stride_elems = stride_bytes / a.width;
    const double bytes_per_block = (double)a.block_size * a.steps * a.width;

    std::vector<double> per_sm, chip;
    for (int rep = 0; rep < a.reps; rep++) {
        CUDA_CHECK(cudaMemset(out, 0, sms * sizeof(BlockTime)));
        kernel<<<sms, a.block_size, BUFFER_BYTES>>>(write, stride_elems, a.steps, a.warmup, out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Every block on its own SM: otherwise "per SM" would be wrong.
        std::vector<unsigned> seen;
        for (int b = 0; b < sms; b++) seen.push_back(out[b].sm);
        std::sort(seen.begin(), seen.end());
        if (std::unique(seen.begin(), seen.end()) != seen.end()) {
            fprintf(stderr, "smem_bandwidth: two blocks shared an SM\n");
            return 1;
        }

        // Per SM: the median block. Chip: everything over the slowest block.
        std::vector<double> bpc;
        unsigned long long slowest = 0;
        long long median_cycles = 0;
        unsigned long long median_ns = 0;
        for (int b = 0; b < sms; b++) {
            bpc.push_back(bytes_per_block / out[b].cycles);
            slowest = std::max(slowest, out[b].ns);
        }
        std::vector<double> sorted = bpc;
        std::sort(sorted.begin(), sorted.end());
        const double median_bpc = sorted[sms / 2];
        for (int b = 0; b < sms; b++)
            if (bpc[b] == median_bpc) { median_cycles = out[b].cycles; median_ns = out[b].ns; }
        const double gbps_sm = bytes_per_block / median_ns;   // bytes per ns = GB/s
        const double gbps_chip = bytes_per_block * sms / slowest;
        printf("smem_bandwidth,%s,%d,%d,%d,%d,%d,%d,%d,%.0f,%lld,%llu,%.3f,%.2f,%.1f\n", op, a.width,
               stride_bytes, a.block_size, sms, a.steps, a.seed, rep, bytes_per_block,
               median_cycles, median_ns, median_bpc, gbps_sm, gbps_chip);
        per_sm.push_back(median_bpc);
        chip.push_back(gbps_chip);
    }

    fprintf(stderr, "smem_bandwidth %s, %d B per access, threads %d B apart, %d blocks x %d threads\n", op,
            a.width, stride_bytes, sms, a.block_size);
    print_stats_line("B/cycle per SM", per_sm);
    print_stats_line("GB/s chip", chip);
    CUDA_CHECK(cudaFree(out));
    return 0;
}
