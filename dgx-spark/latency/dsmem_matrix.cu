// dsmem_matrix.cu — the FULL rank-to-rank DSMEM latency matrix.
//
// dsmem_remote.cu only ever reads FROM rank 0: it measures latency(0 -> d)
// for each target d. That leaves an obvious question open. Rank 0 reading
// rank 1 costs 210.9 cycles, while rank 0 reading rank 10 costs 179.9 — the
// supposedly "nearest" peer is the slowest. Is that a property of the pair,
// of the reader, or of the target? Only the whole matrix can say.
//
// So: for every ordered pair (reader k, target d) this runs one launch in
// which ONLY thread 0 of rank k chases rank d's shared memory. One reader at
// a time, so nothing queues and every cell is an unloaded latency directly
// comparable with dsmem_remote.
//
// It also records %smid for every rank on every launch. Ranks are an
// abstraction; SMs are the hardware. Converting the matrix from rank
// coordinates to SM coordinates is what turns "rank 1 is slow" into a
// statement about the physical layout (TPC = smid/2, GPC = (smid/2) % 4,
// established in gpc.cu).
//
// Build: make dsmem_matrix
// Run:   ./dsmem_matrix [--cluster-size 12] [--reps 21] [--steps 16384]

#include <cooperative_groups.h>

#include "../common.cuh"

namespace cg = cooperative_groups;

__global__ void matrix_chase(const unsigned* __restrict__ perm, int reader,
                             int target, int warmup, int steps, Result* out,
                             unsigned* smids, int active) {
    __shared__ unsigned sbuf[SBUF];
    cg::cluster_group cluster = cg::this_cluster();

    for (int i = threadIdx.x; i < SBUF; i += blockDim.x) sbuf[i] = perm[i];
    __syncthreads();

    unsigned rank = cluster.block_rank();
    unsigned cl = (unsigned)cg::this_grid().cluster_rank();
    if (threadIdx.x == 0) smids[cl * cluster.num_blocks() + rank] = smid();
    cluster.sync();                                // all buffers ready

    // Exactly ONE thread in the whole cluster chases: no contention, so this
    // is the unloaded latency of the single hop (reader -> target).
    if ((int)cl == active && (int)rank == reader && threadIdx.x == 0) {
        const unsigned* buf = cluster.map_shared_rank((const unsigned*)sbuf, target);
        warm_then_chase(buf, 0, warmup, steps, out);
    }

    cluster.sync();   // keep the target alive while its memory is read
}

int main(int argc, char** argv) {
    Args a;
    a.reps = 21;   // 144 cells, so a smaller default than the other programs
    parse_args(argc, argv, &a, "dsmem_matrix",
               "  --cluster-size N  blocks per cluster, 2..12            (default 12)\n"
               "  --clusters N      clusters to launch; 4 covers all GPCs  (default 1)\n"
               "  --active N        which cluster is measured, 0..N-1      (default 0)\n");
    if (a.cluster_size == 2) a.cluster_size = 12;   // Args' default means "unset" here
    int cs = a.cluster_size;
    if (cs < 2 || cs > 12) {
        fprintf(stderr, "dsmem_matrix: --cluster-size must be 2..12 on GB10\n");
        return 1;
    }
    print_header(argc, argv, a, SBUF);

    std::mt19937 rng(a.seed);
    std::vector<unsigned> hperm(SBUF);
    make_pattern(hperm, rng, a.stride_bytes);
    unsigned* dperm;
    CUDA_CHECK(cudaMalloc(&dperm, SBUF * sizeof(unsigned)));
    CUDA_CHECK(cudaMemcpy(dperm, hperm.data(), SBUF * sizeof(unsigned),
                          cudaMemcpyHostToDevice));

    Result* out;   unsigned* smids;
    CUDA_CHECK(cudaMallocManaged(&out, sizeof(Result)));
    CUDA_CHECK(cudaMallocManaged(&smids, (size_t)cs * a.clusters * sizeof(unsigned)));

    if (cs > 8) allow_big_clusters((const void*)matrix_chase);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(cs * a.clusters, 1, 1);
    cfg.blockDim = dim3(a.block_size, 1, 1);
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = cs;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;

    std::vector<double> med(cs * cs, 0.0);
    std::vector<int> smid_of(cs, -1);
    int smid_changes = 0;

    for (int reader = 0; reader < cs; reader++) {
        for (int target = 0; target < cs; target++) {
            std::vector<double> v;
            int sr = -1, st = -1;
            for (int rep = 0; rep < a.reps; rep++) {
                CUDA_CHECK(cudaLaunchKernelEx(&cfg, matrix_chase,
                                              (const unsigned*)dperm, reader,
                                              target, a.warmup, a.steps, out, smids,
                                              a.active));
                CUDA_CHECK(cudaDeviceSynchronize());
                const unsigned* mine = smids + (size_t)a.active * cs;
                sr = (int)mine[reader];   st = (int)mine[target];
                // Is the rank -> SM assignment stable across launches?
                for (int k = 0; k < cs; k++) {
                    if (smid_of[k] < 0) smid_of[k] = (int)mine[k];
                    else if (smid_of[k] != (int)mine[k]) smid_changes++;
                }
                int dist = (target - reader + cs) % cs;
                print_row("dsmem_matrix", cs, dist, 1, a.block_size, a.steps, 0,
                          a.stride_bytes, a.seed, rep, *out, 1, 0,
                          reader, target, sr, st);
                v.push_back((double)out->cycles / a.steps);
            }
            std::sort(v.begin(), v.end());
            med[reader * cs + target] = v[v.size() / 2];
        }
    }

    // ---- human-readable summary on stderr ----
    fprintf(stderr, "measured cluster %d of %d  ->  GPC %d\n", a.active, a.clusters,
            smid_of[0] >= 0 ? (smid_of[0] / 2) % 4 : -1);
    fprintf(stderr, "rank -> SM map (stable across launches: %s)\n",
            smid_changes ? "NO" : "yes");
    fprintf(stderr, "  rank:"); for (int k = 0; k < cs; k++) fprintf(stderr, "%5d", k);
    fprintf(stderr, "\n  smid:"); for (int k = 0; k < cs; k++) fprintf(stderr, "%5d", smid_of[k]);
    fprintf(stderr, "\n  TPC :"); for (int k = 0; k < cs; k++) fprintf(stderr, "%5d", smid_of[k] / 2);
    fprintf(stderr, "\n  GPC :"); for (int k = 0; k < cs; k++) fprintf(stderr, "%5d", (smid_of[k] / 2) % 4);

    fprintf(stderr, "\n\nmedian cycles/load, rows = reader rank, cols = target rank\n     ");
    for (int t = 0; t < cs; t++) fprintf(stderr, "%7d", t);
    fprintf(stderr, "\n");
    for (int r = 0; r < cs; r++) {
        fprintf(stderr, "%4d |", r);
        for (int t = 0; t < cs; t++) fprintf(stderr, "%7.1f", med[r * cs + t]);
        fprintf(stderr, "\n");
    }

    // Symmetry: does k->d cost the same as d->k?
    double worst = 0; int wr = 0, wt = 0;
    for (int r = 0; r < cs; r++)
        for (int t = r + 1; t < cs; t++) {
            double diff = fabs(med[r * cs + t] - med[t * cs + r]);
            if (diff > worst) { worst = diff; wr = r; wt = t; }
        }
    fprintf(stderr, "\nlargest asymmetry |m[k][d] - m[d][k]| = %.1f cy (ranks %d and %d)\n",
            worst, wr, wt);

    // Same TPC vs different TPC, using the measured SM map.
    double same = 0, diff = 0; int nsame = 0, ndiff = 0;
    for (int r = 0; r < cs; r++)
        for (int t = 0; t < cs; t++) {
            if (r == t) continue;
            if (smid_of[r] / 2 == smid_of[t] / 2) { same += med[r * cs + t]; nsame++; }
            else { diff += med[r * cs + t]; ndiff++; }
        }
    if (nsame) fprintf(stderr, "same TPC  (n=%3d): mean %.1f cy\n", nsame, same / nsame);
    if (ndiff) fprintf(stderr, "other TPC (n=%3d): mean %.1f cy\n", ndiff, diff / ndiff);

    // ---- fit  latency(k -> d) = base + w[TPC(k)] + w[TPC(d)] ----
    // If this fits with no error, each endpoint pays a cost that depends only
    // on itself: there is no pairwise-distance term, which is what a shared
    // interconnect point inside the GPC looks like from the outside.
    {
        std::vector<int> tpc;                       // distinct TPCs, ascending
        for (int k = 0; k < cs; k++) {
            int t = smid_of[k] / 2;
            if (std::find(tpc.begin(), tpc.end(), t) == tpc.end()) tpc.push_back(t);
        }
        std::sort(tpc.begin(), tpc.end());
        int nt = (int)tpc.size();
        // a representative rank on each TPC, and a second one if it exists
        std::vector<int> r0(nt, -1), r1(nt, -1);
        for (int k = 0; k < cs; k++) {
            int p = (int)(std::find(tpc.begin(), tpc.end(), smid_of[k] / 2) - tpc.begin());
            if (r0[p] < 0) r0[p] = k; else if (r1[p] < 0) r1[p] = k;
        }
        bool pairs = true;
        for (int p = 0; p < nt; p++) if (r1[p] < 0) pairs = false;
        if (pairs && nt > 1) {
            std::vector<double> diag(nt), w(nt);
            for (int p = 0; p < nt; p++) diag[p] = med[r0[p] * cs + r1[p]];
            double base = *std::min_element(diag.begin(), diag.end());
            for (int p = 0; p < nt; p++) w[p] = (diag[p] - base) / 2.0;
            double maxerr = 0;
            for (int p = 0; p < nt; p++)
                for (int q = 0; q < nt; q++) {
                    int rp = r0[p], rq = (p == q) ? r1[q] : r0[q];
                    maxerr = fmax(maxerr, fabs(med[rp * cs + rq] - (base + w[p] + w[q])));
                }
            fprintf(stderr, "\nadditive fit:  latency(k -> d) = %.1f + w[TPC k] + w[TPC d]\n", base);
            fprintf(stderr, "  TPC :"); for (int p = 0; p < nt; p++) fprintf(stderr, "%8d", tpc[p]);
            fprintf(stderr, "\n  w   :"); for (int p = 0; p < nt; p++) fprintf(stderr, "%8.0f", w[p]);
            fprintf(stderr, "\n  max error over %d TPC pairs: %.2f cy\n", nt * nt, maxerr);
            fprintf(stderr, "  (compare w with part 3 of ../extra/l2_topology.cu: the same\n"
                            "   distances, recovered from L2 reads instead of SM-to-SM hops)\n");
        }
    }

    cudaFree(dperm); cudaFree(out); cudaFree(smids);
    return 0;
}
