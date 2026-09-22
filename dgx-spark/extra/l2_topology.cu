// l2_topology.cu — how far is each SM, physically, from the L2?
//
// L2 latency is not uniform: it varies by 19% across the 48 SMs, in a fixed,
// reproducible pattern that depends only on which TPC the SM belongs to.
// Subtracting each GPC's minimum cancels the constant leg from the GPC
// outward and leaves each TPC's own distance to its GPC's interconnect point.
//
// Those distances are the CROSS-CHECK for ../latency/dsmem_matrix.cu, which
// fits the whole 12x12 DSMEM latency matrix with
//     latency(k -> d) = 148.9 + w[TPC of k] + w[TPC of d]
// and recovers, from a completely different experiment, the same six numbers
// this program prints: 31 19 8 24 12 0. Run both and compare the last block
// of output.
//
// METHOD. A *fine-grained* pointer chase: unlike ../latency/l2.cu, which
// times 16384 loads and divides, this times every load on its own.
//     t0 = clock64();  next = buf[idx];  sink[..] = next;  t1 = clock64();
// The store to `sink` is essential: loads do not block, so without something
// that consumes the value the second clock read can happen before the data
// arrives.
//
// You cannot ask CUDA to place a block on a chosen SM, but you can force the
// placement: each block requests more than half an SM's shared memory, so only
// one block fits per SM, and a grid of exactly 48 blocks therefore covers all
// 48 SMs one apiece. The block that finds itself on the target SM chases; the
// other 47 return immediately. The target is swept over all 48 SMs, every run
// walking the identical address sequence through the same warmed, L2-resident
// buffer.
//
// Build: make l2_topology      (or: nvcc -O3 -arch=sm_121 l2_topology.cu -o l2_topology)
// Run:   ./l2_topology [--buffer-bytes N (default 6 MiB)] [--steps N] [--seed N]
// Output: stdout = one CSV row per SM; stderr = the human-readable report.

#include <cuda_runtime.h>

#include <vector>

#include "../common.cuh"  // CUDA_CHECK, smid()

// One block per SM (the host requests >50% of an SM's shared memory), so
// exactly one block sees smid() == target_sm. Times every load individually.
__global__ void fine_chase(const unsigned* buf, unsigned start, int steps,
                           unsigned* lat, unsigned target_sm, unsigned* ran_on) {
    extern __shared__ char scratch[];
    if (threadIdx.x != 0) return;
    if (smid() != target_sm) return;              // not my SM
    *ran_on = smid();

    unsigned idx = start;
    for (int i = 0; i < steps; i++) idx = buf[idx];   // untimed warm lap

    for (int i = 0; i < steps; i++) {
        long long t0 = clock64();
        unsigned next = buf[idx];
        scratch[i % 32] = next;        // forces the load to have completed
        long long t1 = clock64();
        lat[i] = (unsigned)(t1 - t0);
        idx = next;
    }
    if (scratch[0] == 0xFFFFFFFFu) *ran_on = 999;    // never true, avoid optimizations
}

// Coalesced sweep: pulls the buffer into L2 before the timed runs.
__global__ void warm_l2(const unsigned* buf, size_t n, unsigned long long* s) {
    unsigned long long acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        acc += buf[i];
    if (acc == 0xDEADBEEFull) *s = acc;
}

int main(int argc, char** argv) {
    Args a;
    parse_args(argc, argv, &a, "l2_topology",
               "  --buffer-bytes N  chase buffer size                   (default 6291456 = 6 MiB)\n");
    if (a.buffer_bytes == 0) a.buffer_bytes = 6u << 20;
    int steps = a.steps;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    size_t n = a.buffer_bytes / sizeof(unsigned);
    int nsm = prop.multiProcessorCount;

    std::mt19937 rng(a.seed);
    std::vector<unsigned> h(n);
    make_cycle(h, rng);                            // one random cycle: no prefetch, no short loops
    unsigned* d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice));

    unsigned long long* s;  CUDA_CHECK(cudaMalloc(&s, sizeof(unsigned long long)));
    unsigned* ran_on;       CUDA_CHECK(cudaMallocManaged(&ran_on, sizeof(unsigned)));

    // Force one block per SM: ask for more than half of an SM's shared memory.
    // A grid of exactly `nsm` blocks then lands one block on every SM.
    int smemPerSM = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smemPerSM,
                                      cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
    int smemPerBlock = smemPerSM / 2 + 1024;
    CUDA_CHECK(cudaFuncSetAttribute((void*)fine_chase,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smemPerBlock));
    unsigned* lat;          CUDA_CHECK(cudaMallocManaged(&lat, steps * sizeof(unsigned)));

    // ONE starting point, drawn once from the seeded RNG and reused by every SM.
    // All 48 therefore walk the identical address sequence through the identical
    // warmed buffer, so any difference between them is the SM's position and
    // nothing else. Drawn rather than hardcoded so --seed still varies it, and
    // so no particular offset is baked into the result.
    const unsigned shared_start =
        std::uniform_int_distribution<unsigned>(0, (unsigned)n - 1)(rng);

    // Lazy module loading flushes L2 on a kernel's first launch, so load the
    // chase kernel BEFORE warming (same trap as ../latency/l2.cu).
    fine_chase<<<nsm, 32, smemPerBlock>>>(d, 1, steps, lat, 0, ran_on);
    CUDA_CHECK(cudaDeviceSynchronize());
    warm_l2<<<48, 256>>>(d, n, s);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto run = [&](unsigned sm, unsigned start, std::vector<unsigned>& outv) {
        *ran_on = 0xFFFF;
        fine_chase<<<nsm, 32, smemPerBlock>>>(d, start, steps, lat, sm, ran_on);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (*ran_on != sm) return false;
        outv.assign(lat, lat + steps);
        return true;
    };

    fprintf(stderr, "buffer %zu MiB (L2-resident, warmed), %d timed loads per SM\n",
            a.buffer_bytes >> 20, steps);
    fprintf(stderr, "start index %u, identical for every SM\n", shared_start);
    fprintf(stderr, "placement: %d blocks x %d B smem -> one block per SM\n\n",
            nsm, smemPerBlock);

    // ---------------- Part 1: every SM ----------------
    printf("benchmark,smid,gpc,tpc,tpc_pos,buffer_bytes,steps,seed,p25,median,p75\n");
    std::vector<unsigned> med(nsm, 0);
    for (int sm = 0; sm < nsm; sm++) {
        std::vector<unsigned> v;
        if (!run((unsigned)sm, shared_start, v)) continue;
        std::sort(v.begin(), v.end());
        med[sm] = v[steps / 2];
        // TPC = smid/2; the GPC it belongs to is TPC % 4 (measured in gpc.cu);
        // its position inside that GPC is TPC / 4.
        int tpc = sm / 2;
        printf("l2_topology,%d,%d,%d,%d,%zu,%d,%d,%u,%u,%u\n", sm, tpc % 4, tpc, tpc / 4,
               a.buffer_bytes, steps, a.seed, v[steps / 4], v[steps / 2], v[3 * steps / 4]);
    }

    fprintf(stderr, "=== 1. median latency from every SM (cell = the TPC's two SMs) ===\n");
    fprintf(stderr, "        pos0     pos1     pos2     pos3     pos4     pos5\n");
    for (int g = 0; g < 4; g++) {
        fprintf(stderr, "GPC %d ", g);
        for (int p = 0; p < 6; p++) {
            int tpc = p * 4 + g;
            fprintf(stderr, " %3u/%-3u ", med[2 * tpc], med[2 * tpc + 1]);
        }
        fprintf(stderr, "\n");
    }

    // ---------------- Part 2: the cross-check ----------------
    // Time to L2 = (this TPC -> its GPC's interconnect point) + (that point -> L2).
    // The second leg is the same for every TPC in a GPC, so subtracting the
    // row minimum cancels it and leaves the per-TPC distances.
    fprintf(stderr, "\n=== 2. CROSS-CHECK: each row minus its own minimum = per-TPC distance ===\n");
    fprintf(stderr, "        pos0     pos1     pos2     pos3     pos4     pos5    (TPC id = pos*4 + GPC)\n");
    for (int g = 0; g < 4; g++) {
        unsigned lo = 0xFFFFFFFFu;
        for (int p = 0; p < 6; p++) lo = std::min(lo, med[2 * (p * 4 + g)]);
        fprintf(stderr, "GPC %d ", g);
        for (int p = 0; p < 6; p++)
            fprintf(stderr, " %7u ", med[2 * (p * 4 + g)] - lo);
        fprintf(stderr, "\n");
    }
    fprintf(stderr,
            "\nCompare the GPC 0 row with the w[] vector printed by\n"
            "../latency/dsmem_matrix.cu, which measures SM-to-SM hops instead of\n"
            "L2 reads. Two unrelated experiments, the same six numbers.\n");

    cudaFree(d); cudaFree(s); cudaFree(ran_on); cudaFree(lat);
    return 0;
}
