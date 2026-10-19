// transfer.cu — what does it cost to get S bytes from one SM's shared memory
// into another's? DSMEM against the traditional copy-back through global memory.
//
// Every method starts with S bytes in the SENDER's shared memory and ends when
// they are in the RECEIVER's shared memory AND the receiver knows it. The
// synchronization is part of the cost: a copy nobody can safely read yet has
// not transferred anything.
//
//   --method
//     dsmem-pull   receiver's threads read the sender's SMEM (map_shared_rank,
//                  16 B coalesced loads) into their own. The sender only
//                  signals that its data is ready.
//     dsmem-push   sender's threads write 16 B stores straight into the
//                  receiver's SMEM, then signal.
//     dsmem-bulk   one sender thread issues cp.async.bulk SMEM -> peer SMEM;
//                  the copy engine signals an mbarrier in the RECEIVER's SMEM.
//     gmem-ldst    the traditional copy-back: sender stores SMEM -> global,
//                  fences, raises a flag in global memory; receiver acquires
//                  the flag and loads global -> SMEM.
//     gmem-tma     the same route with the copy engine doing both copies:
//                  cp.async.bulk SMEM -> global, then global -> SMEM.
//     cluster-sync no data: the cost of one cluster-wide barrier, the other
//                  way DSMEM kernels commonly synchronize.
//
//   --mode
//     pingpong     latency. A sends S bytes to B, B sends S bytes back,
//                  --rounds times; one-way time = round trip / 2, timed on
//                  A's clock only (SM clocks are not synchronized).
//     stream       bandwidth. A sends --rounds messages back-to-back into a
//                  ring of slots in B; B hands back per-message credits so a
//                  slot is never overwritten before it is consumed. Timed on
//                  B's clock, from the last warm-up arrival to the last one.
//
// SYNCHRONIZATION. Each side uses its cheapest correct mechanism. DSMEM
// methods signal with mbarriers in shared memory, the cluster-native tool
// (CUTLASS's full/empty pipelines): the sender arrives remotely on the
// receiver's "full" barrier of a slot, the receiver arrives remotely on the
// sender's "empty" barrier once the slot is consumed. The GMEM methods cannot
// use mbarriers across SMs outside a cluster, so they use a flag pair in
// global memory, the traditional way. A one-thread scratch test on the
// 10 -> 11 pair measured the DSMEM options, one-way:
//     remote mbarrier arrive 270 ns < st.release/ld.acquire 292 ns
//     < fence + relaxed 433 ns;   unordered volatile store/poll (a floor, not
//     correct code) 182 ns.
//
// FENCES ARE THE EXPENSIVE PART. On sm_121 a release at CLUSTER scope
// compiles to the same MEMBAR.ALL.GPU as one at GPU scope and costs ~420
// cycles per issue, against ~114 for the same remote arrive relaxed. So every
// method fences exactly where the memory model requires it, and nowhere else:
//   * publishing data: release at the consumer's scope (cluster for DSMEM,
//     gpu for GMEM), acquire on the other side. Both pay one MEMBAR.ALL.GPU.
//   * bulk copies: no release. The copy engine itself completes the
//     receiver's mbarrier; the sender only needs fence.proxy.async so the
//     engine sees what ordinary stores wrote.
//   * credits ("slot consumed"): relaxed. The receiver has finished its reads
//     of the slot (their values were used before the __syncthreads() that
//     precedes the credit), so there is nothing left to order.
//
// PLACEMENT. The launch is always one cluster (default 12 blocks), even for
// the GMEM methods, so every method runs on the SAME physical SM pair,
// --src -> --dst. 10 -> 11 is the fastest DSMEM pair measured by
// dsmem_matrix.cu (148.9 cy), 0 -> 1 the slowest (210.9 cy). The other blocks
// only take part in the opening and closing cluster.sync().
//
// CORRECTNESS. Before any timing, one untimed launch runs the same code with
// every message stamped (word i of round r = r * 1000003 + i) and every
// received word checked, with a window of one message so a stale slot always
// holds a different round's stamp. A memory-ordering bug shows up as a fast,
// wrong result; any mismatch aborts the program. Timed runs skip the stamping
// and the check and are otherwise identical.
//
// Build: make transfer
// Run:   ./transfer --method dsmem-bulk --mode pingpong --bytes 1024 --reps 11

#include <cooperative_groups.h>

#include <cstdint>

#include "../common.cuh"

namespace cg = cooperative_groups;

enum { M_DSMEM_PULL = 0, M_DSMEM_PUSH, M_DSMEM_BULK, M_GMEM_LDST, M_GMEM_TMA, M_CLUSTER_SYNC };
enum { MODE_PINGPONG = 0, MODE_STREAM = 1 };

// Shared memory layout of every block: a 256 B control area, then the
// sender's S-byte source buffer, then the receiver's ring of nslot S-byte
// slots. Each part starts on a 128 B boundary (TMA needs 16).
constexpr int OFF_FULL = 0;       // u64[MAX_SLOTS]: slot k of MY ring holds a message
constexpr int OFF_EMPTY = 64;     // u64[MAX_SLOTS]: the peer has consumed my message in slot k
constexpr int CTRL_BYTES = 256;
constexpr int MAX_SLOTS = 4;

// Global flags: one 256 B block per direction (0 = src -> dst, 1 = dst -> src),
// arrive and credit on separate 128 B lines.
constexpr int GF_STRIDE = 64, GF_ARRIVE = 0, GF_CREDIT = 32;

constexpr long long TIMEOUT_CYCLES = 5000000000ll;   // ~2 s: a deadlock, not a slow run

struct Timing {
    long long cycles;
    unsigned long long ns;
};

struct Params {
    int method, mode, bytes, slot_bytes, nslot, src, dst, rounds, warm, verify;
    unsigned char* gstage;   // 2 directions x nslot x slot_bytes, global
    unsigned* gflags;        // 2 x GF_STRIDE
    Timing* out;
    unsigned* smids;
    unsigned* err;           // [0] mismatches, [1..2] first bad (round, word), [3] abort
};

// ------------------------------------------------------------ PTX wrappers

__device__ __forceinline__ uint32_t cta_addr(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}
// My shared::cta address -> the same offset in block `rank`'s shared memory,
// as a shared::cluster address (what cp.async.bulk and remote mbarriers take).
__device__ __forceinline__ uint32_t mapa(uint32_t a, uint32_t rank) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
    return r;
}
__device__ __forceinline__ void st_relaxed_gpu(unsigned* p, unsigned v) {
    asm volatile("st.relaxed.gpu.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_relaxed_gpu(const unsigned* p) {
    unsigned v;
    asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release_gpu(unsigned* p, unsigned v) {
    asm volatile("st.release.gpu.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void mbar_init(uint32_t a, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count) : "memory");
}
// One arrival (the receiver's) plus the bytes the copy engine will deliver.
// The phase completes when both have happened, in either order.
__device__ __forceinline__ void mbar_expect(uint32_t a, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 ::"r"(a), "r"(bytes) : "memory");
}
__device__ __forceinline__ bool mbar_try_wait_relaxed(uint32_t a, unsigned parity) {
    uint32_t ok;
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 p, [%1], %2;\n\t"
                 "selp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok) : "r"(a), "r"(parity) : "memory");
    return ok;
}
__device__ __forceinline__ bool mbar_try_wait(uint32_t a, unsigned parity) {
    uint32_t ok;
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n\t"
                 "selp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok) : "r"(a), "r"(parity) : "memory");
    return ok;
}
// One arrival on an mbarrier in ANOTHER block's shared memory. Release at
// cluster scope: everything this block wrote before it is visible to whoever
// sees the phase complete.
__device__ __forceinline__ void mbar_remote_arrive(uint32_t a_cluster) {
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
                 ::"r"(a_cluster) : "memory");
}
// The same arrival with no ordering: for credits, which publish nothing.
__device__ __forceinline__ void mbar_remote_arrive_relaxed(uint32_t a_cluster) {
    asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];"
                 ::"r"(a_cluster) : "memory");
}
// SMEM -> peer SMEM, completion signalled on the peer's mbarrier.
__device__ __forceinline__ void bulk_s2s(uint32_t dst_cluster, uint32_t src_cta, unsigned bytes,
                                         uint32_t mbar_cluster) {
    asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes "
                 "[%0], [%1], %2, [%3];"
                 ::"r"(dst_cluster), "r"(src_cta), "r"(bytes), "r"(mbar_cluster) : "memory");
}
// global -> my SMEM, completion on my mbarrier.
__device__ __forceinline__ void bulk_g2s(uint32_t dst_cta, const void* gsrc, unsigned bytes,
                                         uint32_t mbar_cta) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1], %2, [%3];"
                 ::"r"(dst_cta), "l"(__cvta_generic_to_global(gsrc)), "r"(bytes), "r"(mbar_cta)
                 : "memory");
}
// my SMEM -> global, completion tracked by bulk groups; waits until written.
__device__ __forceinline__ void bulk_s2g_and_wait(void* gdst, uint32_t src_cta, unsigned bytes) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                 ::"l"(__cvta_generic_to_global(gdst)), "r"(src_cta), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}
// The copy engine is a different "proxy" from ordinary loads and stores:
// data handed between the two needs an explicit proxy fence.
__device__ __forceinline__ void fence_proxy_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_global() {
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

// ------------------------------------------------------------ helpers

// Spin until cond() holds. Gives up (and tells every other waiter to give up)
// after TIMEOUT_CYCLES, so a synchronization bug ends the run instead of
// hanging the GPU.
template <class F>
__device__ __forceinline__ bool spin(F cond, unsigned* abortp) {
    long long t0 = clock64();
    unsigned n = 0;
    while (!cond()) {
        if ((++n & 255) == 0) {
            if (*(volatile unsigned*)abortp) return false;
            if (clock64() - t0 > TIMEOUT_CYCLES) { atomicExch(abortp, 1u); return false; }
        }
    }
    return true;
}

__device__ __forceinline__ void copy16(void* dst, const void* src, int bytes) {
    uint4* d = (uint4*)dst;
    const uint4* s = (const uint4*)src;
    for (int i = threadIdx.x; i < bytes / 16; i += blockDim.x) d[i] = s[i];
}

__device__ __forceinline__ unsigned stamp_of(int r, int i) { return (unsigned)r * 1000003u + i; }

__device__ void stamp(unsigned char* buf, int bytes, int r) {
    unsigned* w = (unsigned*)buf;
    for (int i = threadIdx.x; i < bytes / 4; i += blockDim.x) w[i] = stamp_of(r, i);
}

__device__ void check(const unsigned char* buf, int bytes, int r, unsigned* err) {
    const unsigned* w = (const unsigned*)buf;
    for (int i = threadIdx.x; i < bytes / 4; i += blockDim.x)
        if (w[i] != stamp_of(r, i) && atomicAdd(&err[0], 1u) == 0) {
            err[1] = (unsigned)r;
            err[2] = (unsigned)i;
        }
}

// Everything one block of the pair needs to talk to the other.
struct Ctx {
    const Params* p;
    int peer, dir_out, dir_in, window;
    unsigned char* smem;       // mine
    unsigned char* peer_smem;  // the peer's, generic (DSMEM) pointer
    uint32_t smem_cta;         // mine, shared::cta address
    uint32_t peer_cluster;     // the peer's, shared::cluster address
};

__device__ __forceinline__ bool is_gmem(int m) { return m == M_GMEM_LDST || m == M_GMEM_TMA; }
__device__ __forceinline__ int src_off() { return CTRL_BYTES; }
__device__ __forceinline__ uint32_t full_bar(int slot) { return OFF_FULL + 8 * slot; }
__device__ __forceinline__ uint32_t empty_bar(int slot) { return OFF_EMPTY + 8 * slot; }
// Message q (1-based) is the ((q-1)/nslot)-th use of slot (q-1)%nslot, so the
// barrier phase to wait on has parity ((q-1)/nslot) & 1.
__device__ __forceinline__ unsigned phase_of(const Params& p, int q) {
    return (unsigned)((q - 1) / p.nslot) & 1u;
}
__device__ __forceinline__ int slot_off(const Params& p, int slot) {
    return CTRL_BYTES + p.slot_bytes * (1 + slot);
}
__device__ __forceinline__ unsigned char* gslot(const Params& p, int dir, int slot) {
    return p.gstage + (size_t)(dir * p.nslot + slot) * p.slot_bytes;
}

// Deliver message r (S bytes of my source buffer) into the peer's slot.
__device__ void send(const Ctx& c, int r) {
    const Params& p = *c.p;
    const int S = p.bytes, slot = (r - 1) % p.nslot;
    unsigned* abortp = &p.err[3];

    // Stream: wait until the peer has consumed the message that last used
    // this slot. (Ping-pong needs no credit: the reply is the credit.)
    if (p.mode == MODE_STREAM && r > c.window && threadIdx.x == 0) {
        const int q = r - c.window;   // this message must have been consumed
        if (is_gmem(p.method)) {
            const unsigned* cr = &p.gflags[c.dir_out * GF_STRIDE + GF_CREDIT];
            spin([&] { return ld_relaxed_gpu(cr) >= (unsigned)q; }, abortp);
        } else {
            const uint32_t e = c.smem_cta + empty_bar((q - 1) % p.nslot);
            spin([&] { return mbar_try_wait_relaxed(e, phase_of(p, q)); }, abortp);
        }
    }
    if (p.verify) {
        __syncthreads();
        stamp(c.smem + src_off(), S, r);
    }
    __syncthreads();   // the source is complete: "the block has produced its data"

    const uint32_t peer_full = c.peer_cluster + full_bar(slot);
    switch (p.method) {
    case M_DSMEM_PULL:   // the data stays here; just say it is ready
        if (threadIdx.x == 0) mbar_remote_arrive(peer_full);
        break;
    case M_DSMEM_PUSH:
        copy16(c.peer_smem + slot_off(p, slot), c.smem + src_off(), S);
        __syncthreads();
        if (threadIdx.x == 0) mbar_remote_arrive(peer_full);
        break;
    case M_DSMEM_BULK:
        if (threadIdx.x == 0) {
            fence_proxy_shared();   // our generic writes -> visible to the copy engine
            bulk_s2s(c.peer_cluster + slot_off(p, slot), c.smem_cta + src_off(), S, peer_full);
        }
        break;
    case M_GMEM_LDST:
        copy16(gslot(p, c.dir_out, slot), c.smem + src_off(), S);
        __syncthreads();
        if (threadIdx.x == 0) {   // release: our stores, and (via the barrier) everyone's
            st_release_gpu(&p.gflags[c.dir_out * GF_STRIDE + GF_ARRIVE], (unsigned)r);
        }
        break;
    case M_GMEM_TMA:
        if (threadIdx.x == 0) {
            fence_proxy_shared();
            bulk_s2g_and_wait(gslot(p, c.dir_out, slot), c.smem_cta + src_off(), S);
            fence_proxy_global();   // copy-engine writes -> ordered before the flag
            st_release_gpu(&p.gflags[c.dir_out * GF_STRIDE + GF_ARRIVE], (unsigned)r);
        }
        break;
    }
}

// Wait for message r, and make sure it sits in my slot, readable by the
// whole block.
__device__ void recv(const Ctx& c, int r) {
    const Params& p = *c.p;
    const int S = p.bytes, slot = (r - 1) % p.nslot;
    const unsigned ph = phase_of(p, r);
    unsigned* abortp = &p.err[3];
    unsigned char* dst = c.smem + slot_off(p, slot);
    const uint32_t mbar = c.smem_cta + full_bar(slot);
    const unsigned* g_arrive = &p.gflags[c.dir_in * GF_STRIDE + GF_ARRIVE];

    switch (p.method) {
    case M_DSMEM_PULL:
        spin([&] { return mbar_try_wait(mbar, ph); }, abortp);
        copy16(dst, c.peer_smem + src_off(), S);
        __syncthreads();
        break;
    case M_DSMEM_PUSH:   // every thread sees the phase complete: no barrier needed
        spin([&] { return mbar_try_wait(mbar, ph); }, abortp);
        break;
    case M_DSMEM_BULK:
        if (threadIdx.x == 0) mbar_expect(mbar, (unsigned)S);
        spin([&] { return mbar_try_wait(mbar, ph); }, abortp);
        break;
    case M_GMEM_LDST:
        if (threadIdx.x == 0) spin([&] { return ld_acquire_gpu(g_arrive) >= (unsigned)r; }, abortp);
        __syncthreads();
        copy16(dst, gslot(p, c.dir_in, slot), S);
        __syncthreads();
        break;
    case M_GMEM_TMA:
        if (threadIdx.x == 0) {
            spin([&] { return ld_acquire_gpu(g_arrive) >= (unsigned)r; }, abortp);
            fence_proxy_global();
            mbar_expect(mbar, (unsigned)S);
            bulk_g2s(c.smem_cta + slot_off(p, slot), gslot(p, c.dir_in, slot), S, mbar);
        }
        spin([&] { return mbar_try_wait(mbar, ph); }, abortp);
        break;
    }
    if (p.verify) {
        check(dst, S, r, p.err);
        __syncthreads();
    }
}

// Stream mode: tell the sender that message r has been consumed.
__device__ void give_credit(const Ctx& c, int r) {
    const Params& p = *c.p;
    __syncthreads();   // every thread is done with the slot
    if (threadIdx.x != 0) return;
    if (is_gmem(p.method)) st_relaxed_gpu(&p.gflags[c.dir_in * GF_STRIDE + GF_CREDIT], (unsigned)r);
    else                   mbar_remote_arrive_relaxed(c.peer_cluster + empty_bar((r - 1) % p.nslot));
}

__global__ void transfer(Params p) {
    extern __shared__ __align__(128) unsigned char smem[];
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = (int)cluster.block_rank();
    const int total = p.warm + p.rounds;

    if (threadIdx.x == 0) {
        // count 1: one arrival per phase (the sender's remote arrive, or for
        // the bulk methods the receiver's own arrive.expect_tx)
        for (int k = 0; k < MAX_SLOTS; k++) {
            mbar_init(cta_addr(smem + full_bar(k)), 1);
            mbar_init(cta_addr(smem + empty_bar(k)), 1);
        }
        // make the initialized mbarriers visible to the peer's copy engine
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        p.smids[rank] = smid();
    }
    if (p.bytes) stamp(smem + src_off(), p.bytes, 0);   // defined contents for timed runs
    __syncthreads();
    cluster.sync();   // flags and barriers ready everywhere before the first message

    long long c0 = 0, c1 = 0;
    unsigned long long g0 = 0, g1 = 0;
    bool timer = false;   // does this block own the measurement?

    if (p.method == M_CLUSTER_SYNC) {
        timer = (rank == p.src);
        for (int r = 1; r <= total; r++) {
            if (r == p.warm + 1) { c0 = clock64(); g0 = globaltimer(); }
            cluster.sync();
        }
        c1 = clock64(); g1 = globaltimer();
    } else if (rank == p.src || rank == p.dst) {
        const bool isA = (rank == p.src);
        Ctx c;
        c.p = &p;
        c.peer = isA ? p.dst : p.src;
        c.dir_out = isA ? 0 : 1;
        c.dir_in = 1 - c.dir_out;
        c.window = p.verify ? 1 : p.nslot;
        c.smem = smem;
        c.peer_smem = (unsigned char*)cluster.map_shared_rank((void*)smem, c.peer);
        c.smem_cta = cta_addr(smem);
        c.peer_cluster = mapa(c.smem_cta, (uint32_t)c.peer);

        if (p.mode == MODE_PINGPONG) {
            timer = isA;
            for (int r = 1; r <= total; r++) {
                if (isA && r == p.warm + 1) { c0 = clock64(); g0 = globaltimer(); }
                if (isA) { send(c, r); recv(c, r); }
                else     { recv(c, r); send(c, r); }
            }
            c1 = clock64(); g1 = globaltimer();
        } else {   // stream: A only sends, B only receives
            timer = !isA;
            if (isA) {
                for (int r = 1; r <= total; r++) send(c, r);
            } else {
                if (p.warm == 0) { c0 = clock64(); g0 = globaltimer(); }
                for (int r = 1; r <= total; r++) {
                    recv(c, r);
                    give_credit(c, r);
                    if (r == p.warm) { c0 = clock64(); g0 = globaltimer(); }
                }
                c1 = clock64(); g1 = globaltimer();
            }
        }
    }

    if (timer && threadIdx.x == 0) {
        p.out->cycles = c1 - c0;
        p.out->ns = g1 - g0;
    }
    // Shared memory disappears when its block exits: nobody leaves while a
    // peer might still be reading or writing it.
    cluster.sync();
}

int main(int argc, char** argv) {
    Args a;
    a.cluster_size = 12;
    parse_args(argc, argv, &a, "transfer",
               "  --method M        dsmem-pull | dsmem-push | dsmem-bulk | gmem-ldst |\n"
               "                    gmem-tma | cluster-sync                (default dsmem-pull)\n"
               "  --mode M          pingpong (latency) | stream (bandwidth) (default pingpong)\n"
               "  --bytes N         message size, multiple of 16; 0 = flag only (default 128)\n"
               "  --rounds N        messages per repetition                (default 1000)\n"
               "  --warmup N        untimed passes of --rounds messages    (default 1)\n"
               "  --src R --dst R   the sending and receiving ranks        (default 10 11)\n"
               "  --cluster-size N  blocks in the cluster, 2..12           (default 12)\n"
               "  --block-size N    threads per block                      (default 128)\n");
    const int cs = a.cluster_size, S = a.bytes;
    const bool bulk = a.method == M_DSMEM_BULK || a.method == M_GMEM_TMA;
    if (cs < 2 || cs > 12) { fprintf(stderr, "transfer: --cluster-size must be 2..12\n"); return 1; }
    if (a.src < 0 || a.dst < 0 || a.src >= cs || a.dst >= cs || a.src == a.dst) {
        fprintf(stderr, "transfer: --src and --dst must be two different ranks < --cluster-size\n");
        return 1;
    }
    if (S < 0 || S % 16) { fprintf(stderr, "transfer: --bytes must be a multiple of 16\n"); return 1; }
    if (bulk && S == 0) { fprintf(stderr, "transfer: the bulk methods need --bytes >= 16\n"); return 1; }
    if (a.block_size < 32 || a.block_size > 1024 || a.block_size % 32) {
        fprintf(stderr, "transfer: --block-size must be a multiple of 32, 32..1024\n");
        return 1;
    }
    if (a.rounds < 1) { fprintf(stderr, "transfer: --rounds must be >= 1\n"); return 1; }
    if (a.method == M_CLUSTER_SYNC) a.bytes = 0;

    int optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    const int slot_bytes = (a.bytes + 127) / 128 * 128;
    int nslot = 1;
    if (a.mode == MODE_STREAM)
        nslot = slot_bytes ? std::min(MAX_SLOTS, (optin - CTRL_BYTES - slot_bytes) / slot_bytes)
                           : MAX_SLOTS;
    const int smem = CTRL_BYTES + slot_bytes * (1 + nslot);
    if (nslot < 1 || smem > optin) {
        fprintf(stderr, "transfer: %d B does not fit in %d B of shared memory\n", a.bytes, optin);
        return 1;
    }

    char extra[400];
    snprintf(extra, sizeof extra,
             "# method: %s\n# mode: %s\n# bytes: %d\n# rounds: %d\n# cluster_size: %d\n"
             "# src: %d\n# dst: %d\n# slots: %d\n# dynamic_smem_bytes: %d\n",
             TRANSFER_METHOD[a.method], TRANSFER_MODE[a.mode], a.bytes, a.rounds, cs, a.src,
             a.dst, nslot, smem);
    print_header(argc, argv, a, a.bytes / 4, extra,
                 "benchmark,method,mode,bytes,block_size,cluster_size,src,dst,smid_src,smid_dst,"
                 "rounds,seed,rep,cycles,ns,ns_per_msg,cycles_per_msg,gbps\n");

    Params p = {};
    p.method = a.method; p.mode = a.mode; p.bytes = a.bytes; p.slot_bytes = slot_bytes;
    p.nslot = nslot; p.src = a.src; p.dst = a.dst;
    const size_t gbytes = 2 * (size_t)nslot * std::max(slot_bytes, 128);
    CUDA_CHECK(cudaMalloc(&p.gstage, gbytes));
    CUDA_CHECK(cudaMemset(p.gstage, 0, gbytes));
    CUDA_CHECK(cudaMalloc(&p.gflags, 2 * GF_STRIDE * sizeof(unsigned)));
    CUDA_CHECK(cudaMallocManaged(&p.out, sizeof(Timing)));
    CUDA_CHECK(cudaMallocManaged(&p.smids, cs * sizeof(unsigned)));
    CUDA_CHECK(cudaMallocManaged(&p.err, 4 * sizeof(unsigned)));

    CUDA_CHECK(cudaFuncSetAttribute((const void*)transfer,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    if (cs > 8) allow_big_clusters((const void*)transfer);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(cs, 1, 1);
    cfg.blockDim = dim3(a.block_size, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = cs;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;

    auto launch = [&](int rounds, int warm, int verify) {
        p.rounds = rounds; p.warm = warm; p.verify = verify;
        CUDA_CHECK(cudaMemset(p.gflags, 0, 2 * GF_STRIDE * sizeof(unsigned)));
        for (int i = 0; i < 4; i++) p.err[i] = 0;
        p.out->cycles = 0; p.out->ns = 0;
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, transfer, p));
        CUDA_CHECK(cudaDeviceSynchronize());
        if (p.err[3]) {
            fprintf(stderr, "transfer: %s %s timed out (deadlock?)\n",
                    TRANSFER_METHOD[a.method], TRANSFER_MODE[a.mode]);
            exit(1);
        }
    };

    // Correctness first: every message stamped and checked, window of one.
    launch(std::min(a.rounds, 256), 0, 1);
    if (p.err[0]) {
        fprintf(stderr, "transfer: %s %s %d B: %u WRONG words, first in round %u word %u\n",
                TRANSFER_METHOD[a.method], TRANSFER_MODE[a.mode], a.bytes, p.err[0], p.err[1],
                p.err[2]);
        return 1;
    }

    // Messages per timed interval: a ping-pong round is two one-way trips.
    const double msgs = (a.mode == MODE_PINGPONG && a.method != M_CLUSTER_SYNC) ? 2.0 * a.rounds
                                                                                : (double)a.rounds;
    std::vector<double> cpm, npm, gbps;
    for (int rep = 0; rep < a.reps; rep++) {
        launch(a.rounds, a.warmup * a.rounds, 0);
        const double ns = (double)p.out->ns, cyc = (double)p.out->cycles;
        const double ns_msg = ns / msgs, cyc_msg = cyc / msgs, g = a.bytes / ns_msg;
        printf("transfer,%s,%s,%d,%d,%d,%d,%d,%u,%u,%d,%d,%d,%lld,%llu,%.3f,%.3f,%.4f\n",
               TRANSFER_METHOD[a.method], TRANSFER_MODE[a.mode], a.bytes, a.block_size, cs,
               a.src, a.dst, p.smids[a.src], p.smids[a.dst], a.rounds, a.seed, rep,
               p.out->cycles, p.out->ns, ns_msg, cyc_msg, g);
        cpm.push_back(cyc_msg); npm.push_back(ns_msg); gbps.push_back(g);
    }

    fprintf(stderr, "transfer %s %s, %d B, rank %d (SM %u) -> rank %d (SM %u), %d slot(s), verified\n",
            TRANSFER_METHOD[a.method], TRANSFER_MODE[a.mode], a.bytes, a.src, p.smids[a.src],
            a.dst, p.smids[a.dst], nslot);
    print_stats_line(a.mode == MODE_PINGPONG ? "cy one-way" : "cy per msg", cpm);
    print_stats_line(a.mode == MODE_PINGPONG ? "ns one-way" : "ns per msg", npm);
    if (a.bytes) print_stats_line("GB/s", gbps);

    cudaFree(p.gstage); cudaFree(p.gflags); cudaFree(p.out); cudaFree(p.smids); cudaFree(p.err);
    return 0;
}
