#!/usr/bin/env python3
"""Synthetic traces for a2av.cu: the simple, balanced patterns of transfer.cu
and cluster.cu, in the same file format as the axccl traces next to this script.

    python make_synthetic_traces.py        (run from this directory)

writes four files, for a cluster of 12 blocks (ranks):

  P2P_SM0-SM1_pairs.log      rank 0 -> rank 1 only: the slowest DSMEM pair
                             (physical SM 0 -> SM 1; rank 0 sits on the worst TPC)
  P2P_SM10-SM11_pairs.log    rank 10 -> rank 11 only: the fastest pair
                             (physical SM 40 -> SM 41), the default of transfer.cu
  SYNTH_RING_pairs.log       rank (k+1) % 12 -> rank k, for every k: the
                             `--pattern ring` of cluster.cu, where rank k
                             reads rank k+1
  SYNTH_BROADCAST_pairs.log  rank 0 -> every other rank: the
                             `--pattern broadcast` of cluster.cu, where every
                             rank reads rank 0

The direction follows cluster.cu: a block READS its target, so the data flows
from the target to the reader. With `a2av --method dsmem-pull` the reader
copies it, as in cluster.cu.

Every non-zero message is S bytes, for S = 16, 32, ..., 16384 (16 KiB, the
shared-memory size the benchmarks use), one
`iteration` per size. All 144 pairs are written, the zero ones included.

Unlike the axccl traces, `avg_msgsize` here is S, the size of EACH non-zero
message, not the average over all 144 pairs: that is what `a2av --avg-msgsize`
selects, and what the plots put on the x axis.

Columns (the axccl `_pairs.log` format):
    nranks,iteration,avg_msgsize,avgoverallbuffersize,src_rank,dst_rank,sendsize,recvsize
"""

NRANKS = 12
SIZES = [16 << i for i in range(11)]   # 16 B ... 16 KiB


def p2p_0_to_1(src, dst):
    return src == 0 and dst == 1


def p2p_10_to_11(src, dst):
    return src == 10 and dst == 11


def ring(src, dst):
    return src == (dst + 1) % NRANKS


def broadcast(src, dst):
    return src == 0 and dst != 0


# File name -> pattern. a2av labels a trace with the text between the first
# `_` and `_pairs`: SM0-SM1, SM10-SM11, RING, BROADCAST.
PATTERNS = {"P2P_SM0-SM1_pairs.log": p2p_0_to_1,
            "P2P_SM10-SM11_pairs.log": p2p_10_to_11,
            "SYNTH_RING_pairs.log": ring,
            "SYNTH_BROADCAST_pairs.log": broadcast}


def write_trace(filename, sends):
    """One file: for every size, the full NRANKS x NRANKS matrix."""
    with open(filename, "w") as f:
        f.write("nranks,iteration,avg_msgsize,avgoverallbuffersize,"
                "src_rank,dst_rank,sendsize,recvsize\n")
        for iteration, size in enumerate(SIZES):
            for src in range(NRANKS):
                for dst in range(NRANKS):
                    b = size if sends(src, dst) else 0
                    f.write(f"{NRANKS},{iteration},{size},{size},{src},{dst},{b},{b}\n")


if __name__ == "__main__":
    for filename, sends in PATTERNS.items():
        write_trace(filename, sends)
        print(f"wrote {filename}")
