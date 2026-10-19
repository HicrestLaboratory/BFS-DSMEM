// transfer.cu — how long does it take to move S bytes from the shared memory
// of one SM to the shared memory of another SM?
//
// THE QUESTION
//   Two thread blocks, A and B, run on two different SMs of the same cluster.
//   A has S bytes in its shared memory and B needs them in ITS shared memory.
//   There are several ways to get them there; this program times each one.
//   A transfer counts as finished only when the bytes are in B's shared
//   memory AND B knows they have arrived, so the signalling is always part of
//   the measured cost.
//
// THE METHODS (--method)
//   Directly from SM to SM through DSMEM (only possible inside a cluster):
//     dsmem-pull    B's threads read A's shared memory and copy it into their own.
//     dsmem-push    A's threads write straight into B's shared memory.
//     dsmem-bulk    one thread of A asks the copy engine (the "TMA" unit) to
//                   copy the bytes into B's shared memory.
//   Through global memory, the traditional way (works between any two SMs):
//     gmem-ldst     A's threads copy shared -> global, then B's threads copy
//                   global -> shared.
//     gmem-tma      the same two copies, done by the copy engine.
//   For reference:
//     cluster-sync  no data at all: the time of one cluster.sync().
//
// THE MODES (--mode)
//     pingpong      latency. A sends a message to B, B sends one back, and so
//                   on for --rounds rounds. One-way time = total / (2 x rounds).
//                   Timed on A's clock (each SM has its own clock).
//     stream        bandwidth. A sends --rounds messages one after another
//                   and B receives them. Throughput = bytes / time, timed on B.
//
// HOW B KNOWS THAT A MESSAGE HAS ARRIVED
//   DSMEM methods use an mbarrier ("memory barrier object"): a small counter
//   in B's shared memory. Each use of it is a "phase" that completes when the
//   expected arrivals (and, for the copy engine, the expected bytes) have
//   come in. A signals by "arriving" on it from its own SM; B waits for the
//   phase to complete. Phases alternate between parity 0 and 1, so B waits
//   for "the phase with parity p" to finish.
//   GMEM methods cannot do that between SMs, so they use the traditional
//   flag: A writes the message number into a counter in global memory, B
//   polls the counter until it reaches that number.
//
// SLOTS AND CREDITS (stream mode only)
//   B has a ring of up to 4 slots for incoming messages, so A can be up to 4
//   messages ahead. Message r goes into slot (r-1) % slots. Before A reuses a
//   slot it waits for a "credit": B's signal that it has finished with the
//   previous message in that slot. (In ping-pong mode B's reply is the credit.)
//
// CORRECTNESS FIRST
//   Before timing, one untimed launch runs the same code with every message
//   filled with a pattern that depends on the round (word i of round r holds
//   r * 1000003 + i), and B checks every word it receives. A synchronization
//   bug would show up as a fast but wrong result; any mismatch stops the
//   program. The timed runs skip the filling and the checking.
//
// PLACEMENT
//   Every method runs inside one cluster (default 12 blocks), even the GMEM
//   ones, so all of them use the SAME two SMs: ranks --src and --dst.
//   10 -> 11 is the fastest DSMEM pair on GB10 (148.9 cycles per load),
//   0 -> 1 the slowest (210.9). The other blocks only join the cluster.sync()
//   at the start and at the end.
//
// WHY THE FENCES ARE WHERE THEY ARE
//   A "release" (make my earlier writes visible before this signal) costs
//   ~420 cycles on GB10, about four times a plain signal, so each one is used
//   only where it is required:
//     - announcing data: release on the signal, acquire on the wait;
//     - copy-engine transfers: no release, the engine itself completes B's
//       mbarrier when the bytes have landed;
//     - credits: relaxed (no release), because B has already finished
//       reading the slot when it gives the credit.
//   Measured one-way on the 10 -> 11 pair, one thread: mbarrier arrive
//   270 ns < flag with release/acquire 292 ns < fence + relaxed flag 433 ns.
//
// Build: make transfer
// Run:   build/transfer --method dsmem-bulk --mode pingpong --bytes 1024 --reps 11

#include <cooperative_groups.h>

#include <cstdint>

#include "../common.cuh"   // CUDA_CHECK, Args, smid(), globaltimer(), print_header, ...

namespace cg = cooperative_groups;

enum { DSMEM_PULL, DSMEM_PUSH, DSMEM_BULK, GMEM_LDST, GMEM_TMA, CLUSTER_SYNC };   // --method
enum { PINGPONG, STREAM };                                                     // --mode

// ------------------------------------------------------------ memory layout

// Shared memory of each block (A and B use the same layout):
//
//   offset   0   full[0..3]   mbarriers: "a message has arrived in my slot k"
//   offset  64   empty[0..3]  mbarriers: "the peer has used my message in its slot k"
//   offset 256   source       the S bytes this block sends
//   then         slot 0, slot 1, ...   where the incoming messages land
//
// The source and every slot take slot_bytes = S rounded up to 128 bytes, so
// each starts on a 128-byte boundary (the copy engine needs 16).
constexpr int FULL_OFFSET = 0;
constexpr int EMPTY_OFFSET = 64;
constexpr int SOURCE_OFFSET = 256;
constexpr int MAX_SLOTS = 4;

// For the GMEM methods: two counters per direction (0 = A -> B, 1 = B -> A).
// alignas(128) puts each counter on its own 128-byte line, so writing one
// does not slow down the SM that is polling the other.
struct alignas(128) Counter { unsigned value; };
struct GlobalFlags {
    Counter arrived;    // number of the last message the sender has delivered
    Counter consumed;   // number of the last message the receiver has finished with
};

struct Timing {
    long long cycles;
    unsigned long long ns;
};

// What went wrong, if anything (managed memory, read by the host).
struct Errors {
    unsigned wrong_words;        // words that did not hold the expected pattern
    unsigned first_wrong_round;
    unsigned first_wrong_word;
    unsigned timed_out;          // 1 = some wait gave up: a deadlock, not a slow run
};

// Everything the kernel needs, passed by value.
struct Params {
    int method, mode;
    int bytes;           // S, the message size
    int slot_bytes;      // S rounded up to 128
    int slots;           // slots in the receive ring (1 in ping-pong mode)
    int src, dst;        // ranks of A and B
    int rounds;          // timed messages (stream) or round trips (ping-pong)
    int warmup_msgs;     // untimed messages before them
    int verify;          // 1 = fill and check every message (the correctness run)
    unsigned char* staging;   // GMEM methods: 2 directions x slots x slot_bytes
    GlobalFlags* flags;       // GMEM methods: flags[0] for A -> B, flags[1] for B -> A
    Timing* out;
    unsigned* sm_of_rank;     // which SM each rank ran on
    Errors* errors;
};

// ------------------------------------------------------------ hardware operations
//
// Glossary:
//   shared address   a block's shared memory has two kinds of address: a
//                    normal pointer, and a 32-bit offset into shared memory
//                    ("shared::cta"). The mbarrier and copy-engine
//                    instructions want the 32-bit form.
//   cluster address  the 32-bit address of ANOTHER block's shared memory in
//                    the cluster-wide shared window ("shared::cluster").
//   release/acquire  a release store makes all my earlier writes visible
//                    before the stored value; an acquire load that sees that
//                    value also sees those writes.
//   proxy fence      the copy engine reads and writes memory separately from
//                    the threads (a different "proxy"); a proxy fence makes
//                    writes of one visible to the other.

// My shared memory as a 32-bit shared address.
__device__ __forceinline__ uint32_t shared_addr(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}
// The same offset in block `rank`'s shared memory, as a cluster address.
__device__ __forceinline__ uint32_t peer_shared_addr(uint32_t my_addr, uint32_t rank) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(my_addr), "r"(rank));
    return r;
}

// Flags in global memory (GMEM methods).
__device__ __forceinline__ void store_release(unsigned* p, unsigned v) {
    asm volatile("st.release.gpu.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned load_acquire(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void store_relaxed(unsigned* p, unsigned v) {
    asm volatile("st.relaxed.gpu.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned load_relaxed(const unsigned* p) {
    unsigned v;
    asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// mbarriers (DSMEM methods, and the receive side of gmem-tma).
// Set up an mbarrier whose phases complete after `count` arrivals.
__device__ __forceinline__ void barrier_init(uint32_t bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
// Arrive on my own mbarrier and announce `bytes` that the copy engine will
// deliver: the phase completes when both have happened, in either order.
__device__ __forceinline__ void barrier_expect_bytes(uint32_t bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 ::"r"(bar), "r"(bytes) : "memory");
}
// Arrive on an mbarrier in ANOTHER block, with release: everything this block
// wrote before is visible to whoever sees the phase complete.
__device__ __forceinline__ void barrier_arrive_remote(uint32_t peer_bar) {
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
                 ::"r"(peer_bar) : "memory");
}
// The same arrival without release: for credits, which announce no data.
__device__ __forceinline__ void barrier_arrive_remote_relaxed(uint32_t peer_bar) {
    asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];"
                 ::"r"(peer_bar) : "memory");
}
// Has the phase with this parity completed? With acquire: the data it
// announces is then visible.
__device__ __forceinline__ bool barrier_phase_done(uint32_t bar, unsigned parity) {
    uint32_t done;
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n\t"
                 "selp.u32 %0, 1, 0, p;\n}"
                 : "=r"(done) : "r"(bar), "r"(parity) : "memory");
    return done;
}
// The same check without acquire: for credits.
__device__ __forceinline__ bool barrier_phase_done_relaxed(uint32_t bar, unsigned parity) {
    uint32_t done;
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 p, [%1], %2;\n\t"
                 "selp.u32 %0, 1, 0, p;\n}"
                 : "=r"(done) : "r"(bar), "r"(parity) : "memory");
    return done;
}

// The copy engine (TMA). It works in the background; completion is reported
// on an mbarrier, or (for copies to global memory) by waiting for the group.
// My shared memory -> the peer's shared memory; completes the peer's mbarrier.
__device__ __forceinline__ void copy_engine_smem_to_peer(uint32_t peer_dst, uint32_t my_src,
                                                         unsigned bytes, uint32_t peer_bar) {
    asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes "
                 "[%0], [%1], %2, [%3];"
                 ::"r"(peer_dst), "r"(my_src), "r"(bytes), "r"(peer_bar) : "memory");
}
// Global memory -> my shared memory; completes my mbarrier.
__device__ __forceinline__ void copy_engine_global_to_smem(uint32_t my_dst, const void* src,
                                                           unsigned bytes, uint32_t my_bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1], %2, [%3];"
                 ::"r"(my_dst), "l"(__cvta_generic_to_global(src)), "r"(bytes), "r"(my_bar)
                 : "memory");
}
// My shared memory -> global memory, and wait until the bytes are written.
__device__ __forceinline__ void copy_engine_smem_to_global_and_wait(void* dst, uint32_t my_src,
                                                                    unsigned bytes) {
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                 ::"l"(__cvta_generic_to_global(dst)), "r"(my_src), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}
// Proxy fences: what the threads wrote becomes visible to the copy engine
// (shared memory), or what the copy engine wrote to the threads (global).
__device__ __forceinline__ void fence_threads_to_copy_engine() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void fence_copy_engine_global() {
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

// ------------------------------------------------------------ small helpers

// Wait until cond() is true. After ~2 s (5e9 cycles) give up and tell every
// other waiter to give up too, so a synchronization bug ends the run instead
// of hanging the GPU.
template <class Cond>
__device__ __forceinline__ void wait_until(Cond cond, unsigned* timed_out) {
    const long long start = clock64();
    unsigned tries = 0;
    while (!cond()) {
        if ((++tries & 255) == 0) {                       // check rarely: it is slow
            if (*(volatile unsigned*)timed_out) return;   // someone else gave up
            if (clock64() - start > 5000000000ll) { atomicExch(timed_out, 1u); return; }
        }
    }
}

// Copy `bytes` with all threads of the block, 16 bytes per thread per step.
__device__ __forceinline__ void block_copy(void* dst, const void* src, int bytes) {
    uint4* d = (uint4*)dst;
    const uint4* s = (const uint4*)src;
    for (int i = threadIdx.x; i < bytes / 16; i += blockDim.x) d[i] = s[i];
}

// The verification pattern: word i of round r.
__device__ __forceinline__ unsigned pattern(int round, int i) {
    return (unsigned)round * 1000003u + i;
}
__device__ void fill_pattern(unsigned char* buf, int bytes, int round) {
    unsigned* w = (unsigned*)buf;
    for (int i = threadIdx.x; i < bytes / 4; i += blockDim.x) w[i] = pattern(round, i);
}
__device__ void check_pattern(const unsigned char* buf, int bytes, int round, Errors* e) {
    const unsigned* w = (const unsigned*)buf;
    for (int i = threadIdx.x; i < bytes / 4; i += blockDim.x)
        if (w[i] != pattern(round, i) && atomicAdd(&e->wrong_words, 1u) == 0) {
            e->first_wrong_round = (unsigned)round;   // the first one to see it records it
            e->first_wrong_word = (unsigned)i;
        }
}

// ------------------------------------------------------------ one side of the pair

// What one block (A or B) knows about the connection to the other.
struct Link {
    const Params* p;
    int peer;                  // rank of the other block
    int out_dir, in_dir;       // flags[] index of what I send / of what I receive
    int window;                // stream: how many messages A may be ahead of B
    unsigned char* my_smem;    // my shared memory
    unsigned char* peer_smem;  // the peer's shared memory (a DSMEM pointer)
    uint32_t my_addr;          // my shared memory, 32-bit shared address
    uint32_t peer_addr;        // the peer's shared memory, cluster address

    __device__ unsigned* timed_out() const { return &p->errors->timed_out; }
    __device__ unsigned char* source() const { return my_smem + SOURCE_OFFSET; }
    __device__ int slot_offset(int slot) const { return SOURCE_OFFSET + p->slot_bytes * (1 + slot); }
    __device__ unsigned char* my_slot(int slot) const { return my_smem + slot_offset(slot); }
    // A slot of the global staging area (GMEM methods), one ring per direction.
    __device__ unsigned char* staging(int dir, int slot) const {
        return p->staging + (size_t)(dir * p->slots + slot) * p->slot_bytes;
    }
};

// Message r (counting from 1) is the ((r-1) / slots)-th use of its slot, and
// the phases of that slot's mbarrier alternate 0, 1, 0, 1...
__device__ __forceinline__ int slot_of(const Params& p, int r) { return (r - 1) % p.slots; }
__device__ __forceinline__ unsigned parity_of(const Params& p, int r) {
    return (unsigned)((r - 1) / p.slots) & 1u;
}
__device__ __forceinline__ uint32_t full_barrier(uint32_t base, int slot) {
    return base + FULL_OFFSET + 8 * slot;
}
__device__ __forceinline__ uint32_t empty_barrier(uint32_t base, int slot) {
    return base + EMPTY_OFFSET + 8 * slot;
}

// ------------------------------------------------------------ the five methods
// For each method: what the sender does with message r, and what the
// receiver does to get it into its slot. send() and recv() below add the
// steps all methods share.

// dsmem-pull: the data stays in the sender; the receiver copies it over.
__device__ void pull_send(const Link& l, int r, int slot) {
    if (threadIdx.x == 0) barrier_arrive_remote(full_barrier(l.peer_addr, slot));   // "ready"
}
__device__ void pull_recv(const Link& l, int r, int slot) {
    wait_until([&] { return barrier_phase_done(full_barrier(l.my_addr, slot), parity_of(*l.p, r)); },
               l.timed_out());
    block_copy(l.my_slot(slot), l.peer_smem + SOURCE_OFFSET, l.p->bytes);
    __syncthreads();
}

// dsmem-push: the sender writes into the receiver's slot, then signals.
__device__ void push_send(const Link& l, int r, int slot) {
    block_copy(l.peer_smem + l.slot_offset(slot), l.source(), l.p->bytes);
    __syncthreads();   // every thread's writes are done before the signal
    if (threadIdx.x == 0) barrier_arrive_remote(full_barrier(l.peer_addr, slot));
}
__device__ void push_recv(const Link& l, int r, int slot) {
    // every thread waits on the mbarrier itself, so no __syncthreads() needed
    wait_until([&] { return barrier_phase_done(full_barrier(l.my_addr, slot), parity_of(*l.p, r)); },
               l.timed_out());
}

// dsmem-bulk: the copy engine copies and completes the receiver's mbarrier.
__device__ void bulk_send(const Link& l, int r, int slot) {
    if (threadIdx.x == 0) {
        fence_threads_to_copy_engine();   // the engine must see what the threads wrote
        copy_engine_smem_to_peer(l.peer_addr + l.slot_offset(slot), l.my_addr + SOURCE_OFFSET,
                                 l.p->bytes, full_barrier(l.peer_addr, slot));
    }
}
__device__ void bulk_recv(const Link& l, int r, int slot) {
    const uint32_t bar = full_barrier(l.my_addr, slot);
    if (threadIdx.x == 0) barrier_expect_bytes(bar, l.p->bytes);   // "S bytes are coming"
    wait_until([&] { return barrier_phase_done(bar, parity_of(*l.p, r)); }, l.timed_out());
}

// gmem-ldst: shared -> global by the sender's threads, a flag, global ->
// shared by the receiver's threads.
__device__ void ldst_send(const Link& l, int r, int slot) {
    block_copy(l.staging(l.out_dir, slot), l.source(), l.p->bytes);
    __syncthreads();   // the release below then covers every thread's stores
    if (threadIdx.x == 0) store_release(&l.p->flags[l.out_dir].arrived.value, (unsigned)r);
}
__device__ void ldst_recv(const Link& l, int r, int slot) {
    if (threadIdx.x == 0)
        wait_until([&] { return load_acquire(&l.p->flags[l.in_dir].arrived.value) >= (unsigned)r; },
                   l.timed_out());
    __syncthreads();
    block_copy(l.my_slot(slot), l.staging(l.in_dir, slot), l.p->bytes);
    __syncthreads();
}

// gmem-tma: the same route, with the copy engine doing both copies.
__device__ void tma_send(const Link& l, int r, int slot) {
    if (threadIdx.x == 0) {
        fence_threads_to_copy_engine();
        copy_engine_smem_to_global_and_wait(l.staging(l.out_dir, slot), l.my_addr + SOURCE_OFFSET,
                                            l.p->bytes);
        fence_copy_engine_global();   // the engine's writes come before the flag
        store_release(&l.p->flags[l.out_dir].arrived.value, (unsigned)r);
    }
}
__device__ void tma_recv(const Link& l, int r, int slot) {
    const uint32_t bar = full_barrier(l.my_addr, slot);
    if (threadIdx.x == 0) {
        wait_until([&] { return load_acquire(&l.p->flags[l.in_dir].arrived.value) >= (unsigned)r; },
                   l.timed_out());
        fence_copy_engine_global();
        barrier_expect_bytes(bar, l.p->bytes);
        copy_engine_global_to_smem(l.my_addr + l.slot_offset(slot), l.staging(l.in_dir, slot),
                                   l.p->bytes, bar);
    }
    wait_until([&] { return barrier_phase_done(bar, parity_of(*l.p, r)); }, l.timed_out());
}

// ------------------------------------------------------------ the shared steps

// Stream mode: wait until message q has been consumed by the receiver (its
// slot is free again). Message q's credit arrives on my `empty` mbarrier
// (DSMEM) or in the `consumed` counter (GMEM).
__device__ void wait_for_credit(const Link& l, int q) {
    const Params& p = *l.p;
    if (p.method == GMEM_LDST || p.method == GMEM_TMA)
        wait_until([&] { return load_relaxed(&p.flags[l.out_dir].consumed.value) >= (unsigned)q; },
                   l.timed_out());
    else
        wait_until([&] { return barrier_phase_done_relaxed(empty_barrier(l.my_addr, slot_of(p, q)),
                                                           parity_of(p, q)); },
                   l.timed_out());
}

// Stream mode, receiver: message r is consumed, its slot may be reused.
__device__ void give_credit(const Link& l, int r) {
    const Params& p = *l.p;
    __syncthreads();   // every thread is done with the slot
    if (threadIdx.x != 0) return;
    if (p.method == GMEM_LDST || p.method == GMEM_TMA)
        store_relaxed(&p.flags[l.in_dir].consumed.value, (unsigned)r);
    else
        barrier_arrive_remote_relaxed(empty_barrier(l.peer_addr, slot_of(p, r)));
}

// Send message r: my source buffer -> the peer's slot.
__device__ void send(const Link& l, int r) {
    const Params& p = *l.p;
    const int slot = slot_of(p, r);
    // Stream: the slot must be free. (Ping-pong: the peer's reply was the credit.)
    if (p.mode == STREAM && r > l.window && threadIdx.x == 0) wait_for_credit(l, r - l.window);
    if (p.verify) {
        __syncthreads();
        fill_pattern(l.source(), p.bytes, r);
    }
    __syncthreads();   // the block has "produced" its data
    switch (p.method) {
        case DSMEM_PULL: pull_send(l, r, slot); break;
        case DSMEM_PUSH: push_send(l, r, slot); break;
        case DSMEM_BULK: bulk_send(l, r, slot); break;
        case GMEM_LDST:  ldst_send(l, r, slot); break;
        case GMEM_TMA:   tma_send(l, r, slot);  break;
    }
}

// Receive message r: wait until it is in my slot, readable by every thread.
__device__ void recv(const Link& l, int r) {
    const Params& p = *l.p;
    const int slot = slot_of(p, r);
    switch (p.method) {
        case DSMEM_PULL: pull_recv(l, r, slot); break;
        case DSMEM_PUSH: push_recv(l, r, slot); break;
        case DSMEM_BULK: bulk_recv(l, r, slot); break;
        case GMEM_LDST:  ldst_recv(l, r, slot); break;
        case GMEM_TMA:   tma_recv(l, r, slot);  break;
    }
    if (p.verify) {
        check_pattern(l.my_slot(slot), p.bytes, r, p.errors);
        __syncthreads();
    }
}

// ------------------------------------------------------------ the kernel

__global__ void transfer(Params p) {
    extern __shared__ __align__(128) unsigned char smem[];
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = (int)cluster.block_rank();
    const int total = p.warmup_msgs + p.rounds;   // messages are numbered 1 .. total

    // Set up the mbarriers: each phase completes after one arrival (the
    // sender's signal, or for the copy-engine methods the receiver's own
    // barrier_expect_bytes, plus the bytes).
    if (threadIdx.x == 0) {
        for (int k = 0; k < MAX_SLOTS; k++) {
            barrier_init(shared_addr(smem + FULL_OFFSET + 8 * k), 1);
            barrier_init(shared_addr(smem + EMPTY_OFFSET + 8 * k), 1);
        }
        // make the initialized mbarriers visible to the other blocks
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        p.sm_of_rank[rank] = smid();
    }
    if (p.bytes) fill_pattern(smem + SOURCE_OFFSET, p.bytes, 0);   // defined contents
    __syncthreads();
    cluster.sync();   // everyone is set up before the first message

    long long c0 = 0, c1 = 0;
    unsigned long long t0 = 0, t1 = 0;
    bool i_time_it = false;   // which block's clock measures the run

    if (p.method == CLUSTER_SYNC) {
        // Every block takes part; rank src times the last `rounds` barriers.
        i_time_it = (rank == p.src);
        for (int r = 1; r <= total; r++) {
            if (r == p.warmup_msgs + 1) { c0 = clock64(); t0 = globaltimer(); }
            cluster.sync();
        }
        c1 = clock64(); t1 = globaltimer();
    } else if (rank == p.src || rank == p.dst) {
        const bool is_a = (rank == p.src);
        Link l;
        l.p = &p;
        l.peer = is_a ? p.dst : p.src;
        l.out_dir = is_a ? 0 : 1;    // flags[0]: A -> B, flags[1]: B -> A
        l.in_dir = 1 - l.out_dir;
        // The correctness run allows one message in flight, so a stale slot
        // always holds a different round's pattern.
        l.window = p.verify ? 1 : p.slots;
        l.my_smem = smem;
        l.peer_smem = (unsigned char*)cluster.map_shared_rank((void*)smem, l.peer);
        l.my_addr = shared_addr(smem);
        l.peer_addr = peer_shared_addr(l.my_addr, (uint32_t)l.peer);

        if (p.mode == PINGPONG) {
            // A: send, wait for the reply. B: wait, reply. Timed on A.
            i_time_it = is_a;
            for (int r = 1; r <= total; r++) {
                if (is_a && r == p.warmup_msgs + 1) { c0 = clock64(); t0 = globaltimer(); }
                if (is_a) { send(l, r); recv(l, r); }
                else      { recv(l, r); send(l, r); }
            }
            c1 = clock64(); t1 = globaltimer();
        } else {
            // A only sends, B only receives. Timed on B, from the arrival of
            // the last warm-up message to the arrival of the last message.
            i_time_it = !is_a;
            if (is_a) {
                for (int r = 1; r <= total; r++) send(l, r);
            } else {
                if (p.warmup_msgs == 0) { c0 = clock64(); t0 = globaltimer(); }
                for (int r = 1; r <= total; r++) {
                    recv(l, r);
                    give_credit(l, r);
                    if (r == p.warmup_msgs) { c0 = clock64(); t0 = globaltimer(); }
                }
                c1 = clock64(); t1 = globaltimer();
            }
        }
    }

    if (i_time_it && threadIdx.x == 0) {
        p.out->cycles = c1 - c0;
        p.out->ns = t1 - t0;
    }
    // A block's shared memory disappears when it exits: nobody leaves while
    // the other block might still be reading or writing it.
    cluster.sync();
}

// ------------------------------------------------------------ host

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

    // ---- check the arguments
    const int cluster_size = a.cluster_size;
    const bool uses_copy_engine = (a.method == DSMEM_BULK || a.method == GMEM_TMA);
    if (cluster_size < 2 || cluster_size > 12) {
        fprintf(stderr, "transfer: --cluster-size must be 2..12\n");
        return 1;
    }
    if (a.src < 0 || a.dst < 0 || a.src >= cluster_size || a.dst >= cluster_size || a.src == a.dst) {
        fprintf(stderr, "transfer: --src and --dst must be two different ranks < --cluster-size\n");
        return 1;
    }
    if (a.bytes < 0 || a.bytes % 16) {
        fprintf(stderr, "transfer: --bytes must be a multiple of 16\n");
        return 1;
    }
    if (uses_copy_engine && a.bytes == 0) {
        fprintf(stderr, "transfer: the copy-engine methods need --bytes >= 16\n");
        return 1;
    }
    if (a.block_size < 32 || a.block_size > 1024 || a.block_size % 32) {
        fprintf(stderr, "transfer: --block-size must be a multiple of 32, 32..1024\n");
        return 1;
    }
    if (a.rounds < 1) {
        fprintf(stderr, "transfer: --rounds must be >= 1\n");
        return 1;
    }
    if (a.method == CLUSTER_SYNC) a.bytes = 0;

    // ---- shared memory: control area + source + receive slots
    int max_smem = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    const int slot_bytes = (a.bytes + 127) / 128 * 128;   // S rounded up to 128
    int slots = 1;                                        // ping-pong: one at a time
    if (a.mode == STREAM)   // as many as fit, up to MAX_SLOTS
        slots = slot_bytes ? std::min(MAX_SLOTS, (max_smem - SOURCE_OFFSET - slot_bytes) / slot_bytes)
                           : MAX_SLOTS;
    const int smem_bytes = SOURCE_OFFSET + slot_bytes * (1 + slots);
    if (slots < 1 || smem_bytes > max_smem) {
        fprintf(stderr, "transfer: %d B does not fit in %d B of shared memory\n", a.bytes, max_smem);
        return 1;
    }

    char extra[400];
    snprintf(extra, sizeof extra,
             "# method: %s\n# mode: %s\n# bytes: %d\n# rounds: %d\n# cluster_size: %d\n"
             "# src: %d\n# dst: %d\n# slots: %d\n# dynamic_smem_bytes: %d\n",
             TRANSFER_METHOD[a.method], TRANSFER_MODE[a.mode], a.bytes, a.rounds, cluster_size,
             a.src, a.dst, slots, smem_bytes);
    print_header(argc, argv, a, a.bytes / 4, extra,
                 "benchmark,method,mode,bytes,block_size,cluster_size,src,dst,smid_src,smid_dst,"
                 "rounds,seed,rep,cycles,ns,ns_per_msg,cycles_per_msg,gbps\n");

    // ---- memory the kernel needs
    Params p = {};
    p.method = a.method;
    p.mode = a.mode;
    p.bytes = a.bytes;
    p.slot_bytes = slot_bytes;
    p.slots = slots;
    p.src = a.src;
    p.dst = a.dst;
    const size_t staging_bytes = 2 * (size_t)slots * std::max(slot_bytes, 128);
    CUDA_CHECK(cudaMalloc(&p.staging, staging_bytes));
    CUDA_CHECK(cudaMemset(p.staging, 0, staging_bytes));
    CUDA_CHECK(cudaMalloc(&p.flags, 2 * sizeof(GlobalFlags)));
    CUDA_CHECK(cudaMallocManaged(&p.out, sizeof(Timing)));
    CUDA_CHECK(cudaMallocManaged(&p.sm_of_rank, cluster_size * sizeof(unsigned)));
    CUDA_CHECK(cudaMallocManaged(&p.errors, sizeof(Errors)));

    // ---- launch configuration: one cluster of cluster_size blocks
    CUDA_CHECK(cudaFuncSetAttribute((const void*)transfer,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    if (cluster_size > 8) allow_big_clusters((const void*)transfer);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(cluster_size);
    cfg.blockDim = dim3(a.block_size);
    cfg.dynamicSmemBytes = smem_bytes;
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = cluster_size;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;

    // One launch: `rounds` timed messages after `warmup_msgs` untimed ones.
    auto launch = [&](int rounds, int warmup_msgs, int verify) {
        p.rounds = rounds;
        p.warmup_msgs = warmup_msgs;
        p.verify = verify;
        CUDA_CHECK(cudaMemset(p.flags, 0, 2 * sizeof(GlobalFlags)));
        *p.errors = Errors{};
        *p.out = Timing{};
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, transfer, p));
        CUDA_CHECK(cudaDeviceSynchronize());
        if (p.errors->timed_out) {
            fprintf(stderr, "transfer: %s %s timed out (deadlock?)\n",
                    TRANSFER_METHOD[a.method], TRANSFER_MODE[a.mode]);
            exit(1);
        }
    };

    // ---- 1. correctness: every message filled and checked
    launch(std::min(a.rounds, 256), 0, 1);
    if (p.errors->wrong_words) {
        fprintf(stderr, "transfer: %s %s %d B: %u WRONG words, first in round %u word %u\n",
                TRANSFER_METHOD[a.method], TRANSFER_MODE[a.mode], a.bytes, p.errors->wrong_words,
                p.errors->first_wrong_round, p.errors->first_wrong_word);
        return 1;
    }

    // ---- 2. timing: one CSV row per repetition
    // A ping-pong round is two one-way trips.
    const double msgs = (a.mode == PINGPONG && a.method != CLUSTER_SYNC) ? 2.0 * a.rounds
                                                                         : (double)a.rounds;
    std::vector<double> cycles_per_msg, ns_per_msg, gbps;
    for (int rep = 0; rep < a.reps; rep++) {
        launch(a.rounds, a.warmup * a.rounds, 0);
        const double ns = (double)p.out->ns / msgs, cy = (double)p.out->cycles / msgs;
        const double g = a.bytes / ns;   // bytes per ns = GB/s
        printf("transfer,%s,%s,%d,%d,%d,%d,%d,%u,%u,%d,%d,%d,%lld,%llu,%.3f,%.3f,%.4f\n",
               TRANSFER_METHOD[a.method], TRANSFER_MODE[a.mode], a.bytes, a.block_size,
               cluster_size, a.src, a.dst, p.sm_of_rank[a.src], p.sm_of_rank[a.dst], a.rounds,
               a.seed, rep, p.out->cycles, p.out->ns, ns, cy, g);
        cycles_per_msg.push_back(cy);
        ns_per_msg.push_back(ns);
        gbps.push_back(g);
    }

    fprintf(stderr, "transfer %s %s, %d B, rank %d (SM %u) -> rank %d (SM %u), %d slot(s), verified\n",
            TRANSFER_METHOD[a.method], TRANSFER_MODE[a.mode], a.bytes, a.src, p.sm_of_rank[a.src],
            a.dst, p.sm_of_rank[a.dst], slots);
    print_stats_line(a.mode == PINGPONG ? "cy one-way" : "cy per msg", cycles_per_msg);
    print_stats_line(a.mode == PINGPONG ? "ns one-way" : "ns per msg", ns_per_msg);
    if (a.bytes) print_stats_line("GB/s", gbps);

    CUDA_CHECK(cudaFree(p.staging));
    CUDA_CHECK(cudaFree(p.flags));
    CUDA_CHECK(cudaFree(p.out));
    CUDA_CHECK(cudaFree(p.sm_of_rank));
    CUDA_CHECK(cudaFree(p.errors));
    return 0;
}
