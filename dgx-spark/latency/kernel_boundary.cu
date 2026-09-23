// kernel_boundary.cu — the cost of handing data to the next BFS level the way a
// level-synchronous GPU BFS does it: through global memory and a kernel launch.
//
// transfer.cu measures every IN-KERNEL way to move S bytes between two SMs.
// Most GPU BFS implementations never do that: each level is one kernel that
// reads the previous level's output from global memory and writes its own,
// and the kernel boundary is the synchronization. This program measures that
// boundary, so the in-kernel numbers have the baseline they replace.
//
// One "level" is one launch of a single block that loads S bytes from buffer
// A into shared memory and stores them to buffer B; the next level reads B and
// writes A. Per-level time = total / --rounds, over three ways of driving the
// launches from the host:
//
//   stream       --rounds launches back-to-back on one stream, one sync at the
//                end. The lower bound for launch-driven levels.
//   graph        the same launches captured once in a CUDA graph (capture and
//                instantiation untimed), then launched as one graph.
//   host-check   after every level, a 4-byte device-to-host copy and a stream
//                sync: what a BFS does to learn whether the frontier is empty.
//
// All times are host wall-clock (std::chrono), in ns, so they compare directly
// with the ns_per_msg of transfer.cu.
//
// Build: make kernel_boundary
// Run:   ./kernel_boundary --bytes 1024 --rounds 1000 --reps 11

#include <chrono>

#include "../common.cuh"

__global__ void level(const uint4* in, uint4* out, int n16, unsigned* counter) {
    extern __shared__ uint4 stage[];
    for (int i = threadIdx.x; i < n16; i += blockDim.x) stage[i] = in[i];
    __syncthreads();
    for (int i = threadIdx.x; i < n16; i += blockDim.x) out[i] = stage[i];
    if (threadIdx.x == 0) *counter += 1;   // the "frontier size" the host may read
}

static const char* VARIANT[] = {"stream", "graph", "host-check"};

int main(int argc, char** argv) {
    Args a;
    parse_args(argc, argv, &a, "kernel_boundary",
               "  --bytes N         bytes handed from one level to the next, multiple of 16\n"
               "                    (default 128)\n"
               "  --rounds N        levels (kernel launches) per repetition (default 1000)\n"
               "  --block-size N    threads per block                       (default 128)\n");
    const int S = a.bytes;
    if (S < 16 || S % 16 || S > 48 * 1024) {
        fprintf(stderr, "kernel_boundary: --bytes must be a multiple of 16, 16..49152\n");
        return 1;
    }
    if (a.rounds < 1) { fprintf(stderr, "kernel_boundary: --rounds must be >= 1\n"); return 1; }

    char extra[160];
    snprintf(extra, sizeof extra, "# bytes: %d\n# levels: %d\n", S, a.rounds);
    print_header(argc, argv, a, S / 4, extra,
                 "benchmark,variant,bytes,block_size,levels,seed,rep,ns,ns_per_level\n");

    uint4 *buf[2];
    unsigned *counter, *host_counter;
    CUDA_CHECK(cudaMalloc(&buf[0], S));
    CUDA_CHECK(cudaMalloc(&buf[1], S));
    CUDA_CHECK(cudaMemset(buf[0], 0, S));
    CUDA_CHECK(cudaMalloc(&counter, sizeof(unsigned)));
    CUDA_CHECK(cudaMallocHost(&host_counter, sizeof(unsigned)));
    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));

    const int n16 = S / 16;
    auto launch = [&](int l) {
        level<<<1, a.block_size, S, st>>>(buf[l & 1], buf[(l + 1) & 1], n16, counter);
    };

    // The graph: every level of one repetition, captured once.
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for (int l = 0; l < a.rounds; l++) launch(l);
    CUDA_CHECK(cudaStreamEndCapture(st, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));

    auto run = [&](int variant) {
        CUDA_CHECK(cudaStreamSynchronize(st));
        auto t0 = std::chrono::steady_clock::now();
        if (variant == 0) {
            for (int l = 0; l < a.rounds; l++) launch(l);
        } else if (variant == 1) {
            CUDA_CHECK(cudaGraphLaunch(exec, st));
        } else {
            for (int l = 0; l < a.rounds; l++) {
                launch(l);
                CUDA_CHECK(cudaMemcpyAsync(host_counter, counter, sizeof(unsigned),
                                           cudaMemcpyDeviceToHost, st));
                CUDA_CHECK(cudaStreamSynchronize(st));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(st));
        CUDA_CHECK(cudaGetLastError());
        return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - t0).count();
    };

    std::vector<double> per[3];
    for (int w = 0; w < a.warmup; w++)       // untimed: module load, graph upload, clocks
        for (int v = 0; v < 3; v++) run(v);
    for (int rep = 0; rep < a.reps; rep++)
        for (int v = 0; v < 3; v++) {
            const double ns = run(v), npl = ns / a.rounds;
            printf("kernel_boundary,%s,%d,%d,%d,%d,%d,%.0f,%.3f\n", VARIANT[v], S, a.block_size,
                   a.rounds, a.seed, rep, ns, npl);
            per[v].push_back(npl);
        }

    fprintf(stderr, "kernel_boundary: %d B per level, %d levels, block %d\n", S, a.rounds,
            a.block_size);
    for (int v = 0; v < 3; v++) {
        char label[40];
        snprintf(label, sizeof label, "ns/level %s", VARIANT[v]);
        print_stats_line(label, per[v]);
    }

    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(st);
    cudaFree(buf[0]); cudaFree(buf[1]); cudaFree(counter); cudaFreeHost(host_counter);
    return 0;
}
