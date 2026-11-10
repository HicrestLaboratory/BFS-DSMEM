// gpc.cu — the number of GPCs (GPU Processing Clusters) and the SM-to-GPC
// mapping of the installed GPU, measured with discover_topology()
// (topology.cuh: SMs seen in the same thread block cluster are in the same GPC).
//
// It prints every GPC with its SMs, checks that both SMs of a TPC (smid / 2)
// are in the same GPC, and checks GB10's layout, GPC = (smid / 2) % GPCs: on
// GB10 consecutive TPCs go round-robin over the 4 GPCs, so consecutive SM ids
// are TPC siblings, not GPC neighbours. Other GPUs (H100: GPCs of unequal size)
// need not follow it; the programs that care use the measured mapping.
//
// Build: make gpc
// Run:   build/gpc

#include "topology.cuh"

int main() {
    const Topology t = discover_topology();

    printf("=== SM -> GPC mapping ===\n");
    int covered = 0;
    for (int g = 0; g < t.num_gpcs; g++) {
        printf("  GPC %d (%zu SMs):", g, t.sms[g].size());
        for (int sm : t.sms[g]) printf(" %d", sm);
        printf("\n");
        covered += (int)t.sms[g].size();
    }

    printf("\n=== Result ===\n");
    printf("GPCs: %d, SMs: %d of %d\n", t.num_gpcs, covered, t.num_sms);
    bool uniform = true;
    for (int g = 1; g < t.num_gpcs; g++)
        if (t.sms[g].size() != t.sms[0].size()) uniform = false;
    if (uniform) printf("SMs per GPC: %zu (uniform)\n", t.sms[0].size());
    else printf("SMs per GPC: not uniform (normal on dies with disabled TPCs)\n");
    printf("Both SMs of every TPC (smid / 2) in the same GPC: %s\n", t.tpc_pairs_ok ? "yes" : "NO");

    bool round_robin = true;
    for (int sm = 0; sm < t.num_sms; sm++)
        if ((sm / 2) % t.num_gpcs != t.gpc_of_sm[sm]) round_robin = false;
    printf("GB10 layout, GPC = (smid / 2) %% %d: %s\n", t.num_gpcs,
           round_robin ? "yes" : "no (use the mapping above)");
    return 0;
}
