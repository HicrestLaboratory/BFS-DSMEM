#!/usr/bin/env python3
"""Synthetic traces for a2av.cu: the simple, balanced patterns of transfer.cu
and cluster.cu, in the same file format as the axccl traces next to this script.

    python make_synthetic_traces.py        (run from this directory)

writes three files, for a cluster of 12 blocks (ranks):

  SYNTH_P2P_pairs.log        rank 0 -> rank 1 only: one pair, the slowest
                             DSMEM pair (rank 0 sits on the worst SM)
  SYNTH_RING_pairs.log       rank (k+1) % 12 -> rank k, for every k: the
                             `--pattern ring` of cluster.cu, where rank k
                             reads rank k+1
  SYNTH_BROADCAST_pairs.log  rank 0 -> every other rank: the
                             `--pattern broadcast` of cluster.cu, where every
                             rank reads rank 0

The direction follows cluster.cu: a block READS its target, so the data flows
from the target to the reader. With `a2av --method dsmem-pull` the reader
copies it, as in cluster.cu.

Every non-zero message is S bytes, for S = 16, 32, ..., 32768, one
`iteration` per size. All 144 pairs are written, the zero ones included.

Unlike the axccl traces, `avg_msgsize` here is S, the size of EACH non-zero
message, not the average over all 144 pairs: that is what `a2av --avg-msgsize`
selects, and what the plots put on the x axis.

Columns (the axccl `_pairs.log` format):
    nranks,iteration,avg_msgsize,avgoverallbuffersize,src_rank,dst_rank,sendsize,recvsize
"""

NRANKS = 12
SIZES = [16 << i for i in range(12)]   # 16 B ... 32 KiB


def p2p(src, dst):
    return src == 0 and dst == 1


def ring(src, dst):
    return src == (dst + 1) % NRANKS


def broadcast(src, dst):
    return src == 0 and dst != 0


PATTERNS = {"P2P": p2p, "RING": ring, "BROADCAST": broadcast}


def write_trace(name, sends):
    """One file: for every size, the full NRANKS x NRANKS matrix."""
    with open(f"SYNTH_{name}_pairs.log", "w") as f:
        f.write("nranks,iteration,avg_msgsize,avgoverallbuffersize,"
                "src_rank,dst_rank,sendsize,recvsize\n")
        for iteration, size in enumerate(SIZES):
            for src in range(NRANKS):
                for dst in range(NRANKS):
                    b = size if sends(src, dst) else 0
                    f.write(f"{NRANKS},{iteration},{size},{size},{src},{dst},{b},{b}\n")


if __name__ == "__main__":
    for name, sends in PATTERNS.items():
        write_trace(name, sends)
        print(f"wrote SYNTH_{name}_pairs.log")
