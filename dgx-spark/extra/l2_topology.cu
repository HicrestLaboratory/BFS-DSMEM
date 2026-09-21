// l2_topology.cu — where does each SM sit, physically, relative to the L2?
//
// Two questions, one instrument.
//
// 1. IS GB10's L2 SPLIT INTO PARTITIONS? On A100 and H800 the L2 is two
//    halves joined by a link, and Luo et al. ("Dissecting the NVIDIA Hopper
//    Architecture", arXiv:2501.12084) see it as two latency classes roughly
//    2x apart, plus an extra step at half the L2 size in the working-set
//    curve. Answer here: no. One peak per SM, and the per-address latency is
//    the same seen from every SM (correlation 0.86-0.99 between SMs in
//    different GPCs, as high as the same SM repeated).
//
// 2. HOW FAR IS EACH SM FROM THE L2? This is the useful part. Latency is not
//    uniform: it varies by 19% across the 48 SMs in a fixed, reproducible
//    pattern that depends only on which TPC the SM belongs to. Subtracting
//    each GPC's minimum cancels the constant leg from the GPC outward and
//    leaves each TPC's own distance to its GPC's interconnect point.
//
//    Those distances are the CROSS-CHECK for ../latency/dsmem_matrix.cu,
//    which fits the whole 12x12 DSMEM latency matrix with
//        latency(k -> d) = 148.9 + w[TPC of k] + w[TPC of d]
//    and recovers, from a completely different experiment, the same six
//    numbers this program prints: 31 19 8 24 12 0. Run both and compare the
//    last block of output.
//
// METHOD. A *fine-grained* pointer chase: unlike ../latency/l2.cu, which
// times 16384 loads and divides, this times every load on its own.
//     t0 = clock64();  next = buf[idx];  sink[..] = next;  t1 = clock64();
// The store to `sink` is essential: loads do not block, so without something
// that consumes the value the second clock read can happen before the data
// arrives. You cannot ask CUDA to place a block on a chosen SM, so the kernel
// is launched with 192 blocks, every block reads %smid, all but those on the
// target SM exit, and one atomicCAS picks a single winner among those. The
// target is swept over all 48 SMs, every run walking the identical address
// sequence through the same warmed, L2-resident buffer.
//
// Build: make l2_topology      (or: nvcc -O3 -arch=sm_121 l2_topology.cu -o l2_topology)
// Run:   ./l2_topology [--buffer-bytes N (default 6 MiB)] [--steps N] [--seed N]
// Output: stdout = one CSV row per SM; stderr = the human-readable report.

#include <cuda_runtime.h>

#include <cmath>
#include <map>
#include <vector>

#include "../common.cuh"  // CUDA_CHECK, smid()

// Runs on the first block that lands on `target_sm`, and times each load.
__global__ void fine_chase(const unsigned* buf, unsigned start, int steps,
                           unsigned* lat, unsigned target_sm, int* claimed,
                           unsigned* ran_on) {
    __shared__ unsigned sink[32];
    if (threadIdx.x != 0) return;
    if (smid() != target_sm) return;              // not my SM
    if (atomicCAS(claimed, 0, 1) != 0) return;    // someone else got here first
    *ran_on = smid();

    unsigned idx = start;
    for (int i = 0; i < steps; i++) idx = buf[idx];   // untimed warm lap

    for (int i = 0; i < steps; i++) {
        long long t0 = clock64();
        unsigned next = buf[idx];
        sink[i & 31] = next;        // forces the load to have completed
        long long t1 = clock64();
        lat[i] = (unsigned)(t1 - t0);
        idx = next;
    }
    if (sink[0] == 0xFFFFFFFFu) *ran_on = 999;    // never true; keeps sink alive
}

// Coalesced sweep: pulls the buffer into L2 before the timed runs.
__global__ void warm_l2(const unsigned* buf, size_t n, unsigned long long* s) {
    unsigned long long acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        acc += buf[i];
    if (acc == 0xDEADBEEFull) *s = acc;
}

static double corr(const std::vector<unsigned>& a, const std::vector<unsigned>& b) {
    size_t n = a.size();
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    double sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < n; i++) {
        double x = a[i] - ma, y = b[i] - mb;
        sab += x * y; saa += x * x; sbb += y * y;
    }
    return sab / std::sqrt(saa * sbb);
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
    bool resident = a.buffer_bytes <= (size_t)prop.l2CacheSize / 2;
    int nsm = prop.multiProcessorCount;

    std::mt19937 rng(a.seed);
    std::vector<unsigned> h(n);
    make_cycle(h, rng);                            // one random cycle: no prefetch, no short loops
    unsigned* d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice));

    unsigned long long* s;  CUDA_CHECK(cudaMalloc(&s, sizeof(unsigned long long)));
    int* claimed;           CUDA_CHECK(cudaMallocManaged(&claimed, sizeof(int)));
    unsigned* ran_on;       CUDA_CHECK(cudaMallocManaged(&ran_on, sizeof(unsigned)));
    unsigned* lat;          CUDA_CHECK(cudaMallocManaged(&lat, steps * sizeof(unsigned)));
    std::uniform_int_distribution<unsigned> rstart(0, (unsigned)n - 1);

    // Lazy module loading flushes L2 on a kernel's first launch, so load the
    // chase kernel BEFORE warming (same trap as ../latency/l2.cu).
    *claimed = 0;
    fine_chase<<<192, 32>>>(d, 1, steps, lat, 0, claimed, ran_on);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (resident) {
        warm_l2<<<192, 256>>>(d, n, s);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto run = [&](unsigned sm, unsigned start, std::vector<unsigned>& outv) {
        *claimed = 0; *ran_on = 0xFFFF;
        fine_chase<<<192, 32>>>(d, start, steps, lat, sm, claimed, ran_on);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (*ran_on != sm) return false;
        outv.assign(lat, lat + steps);
        return true;
    };

    fprintf(stderr, "buffer %zu MiB (%s), %d timed loads per SM\n\n",
            a.buffer_bytes >> 20,
            resident ? "L2-resident, warmed" : "larger than L2/2: NOT resident", steps);

    // ---------------- Part 1: is the L2 partitioned? ----------------
    // Two peaks in the histogram, or SMs that disagree about which addresses
    // are fast, would mean partitions. One SM per GPC, each sampled twice.
    fprintf(stderr, "=== 1. partition check: one SM per GPC, each run twice ===\n");
    unsigned probe[] = {0, 0, 2, 2, 4, 4, 6, 6};
    std::vector<std::vector<unsigned>> R;
    std::vector<unsigned> Rsm;
    for (unsigned sm : probe) {
        std::vector<unsigned> v;
        if (!run(sm, resident ? 12345u : rstart(rng), v)) {
            fprintf(stderr, "  SM %u: no block landed there\n", sm);
            continue;
        }
        std::vector<unsigned> srt = v;
        std::sort(srt.begin(), srt.end());
        fprintf(stderr, "  SM %2u (GPC %u): p5 %u  p50 %u  p95 %u   ", sm, (sm / 2) % 4,
                srt[steps / 20], srt[steps / 2], srt[19 * steps / 20]);
        std::map<int, int> hist;
        for (unsigned x : v) hist[(x / 10) * 10]++;
        for (auto& kv : hist) if (kv.second >= steps / 100)
            fprintf(stderr, "[%d:%.0f%%] ", kv.first, 100.0 * kv.second / steps);
        fprintf(stderr, "\n");
        R.push_back(v); Rsm.push_back(sm);
    }
    if (resident && R.size() > 1) {
        fprintf(stderr, "\n  per-load latency correlation (identical address sequence):\n        ");
        for (unsigned sm : Rsm) fprintf(stderr, "SM%-2u  ", sm);
        fprintf(stderr, "\n");
        for (size_t i = 0; i < R.size(); i++) {
            fprintf(stderr, "  SM%-2u  ", Rsm[i]);
            for (size_t j = 0; j < R.size(); j++) fprintf(stderr, "%5.2f ", corr(R[i], R[j]));
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "  (all high => every SM agrees which addresses are fast => ONE L2, not two partitions)\n");
    }

    // ---------------- Part 2: every SM ----------------
    printf("benchmark,smid,gpc,tpc,tpc_pos,buffer_bytes,steps,seed,p25,median,p75\n");
    std::vector<unsigned> med(nsm, 0);
    for (int sm = 0; sm < nsm; sm++) {
        std::vector<unsigned> v;
        if (!run((unsigned)sm, resident ? 12345u : rstart(rng), v)) continue;
        std::sort(v.begin(), v.end());
        med[sm] = v[steps / 2];
        // TPC = smid/2; the GPC it belongs to is TPC % 4 (measured in gpc.cu);
        // its position inside that GPC is TPC / 4.
        int tpc = sm / 2;
        printf("l2_topology,%d,%d,%d,%d,%zu,%d,%d,%u,%u,%u\n", sm, tpc % 4, tpc, tpc / 4,
               a.buffer_bytes, steps, a.seed, v[steps / 4], v[steps / 2], v[3 * steps / 4]);
    }

    fprintf(stderr, "\n=== 2. median latency from every SM (cell = the TPC's two SMs) ===\n");
    fprintf(stderr, "        pos0     pos1     pos2     pos3     pos4     pos5\n");
    for (int g = 0; g < 4; g++) {
        fprintf(stderr, "GPC %d ", g);
        for (int p = 0; p < 6; p++) {
            int tpc = p * 4 + g;
            fprintf(stderr, " %3u/%-3u ", med[2 * tpc], med[2 * tpc + 1]);
        }
        fprintf(stderr, "\n");
    }

    // ---------------- Part 3: the cross-check ----------------
    // Time to L2 = (this TPC -> its GPC's interconnect point) + (that point -> L2).
    // The second leg is the same for every TPC in a GPC, so subtracting the
    // row minimum cancels it and leaves the per-TPC distances.
    fprintf(stderr, "\n=== 3. CROSS-CHECK: each row minus its own minimum = per-TPC distance ===\n");
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

    cudaFree(d); cudaFree(s); cudaFree(claimed); cudaFree(ran_on); cudaFree(lat);
    return 0;
}
