#!/bin/bash
# Per-warp latency tomography
# For each major op, run a single-warp single-op latency test and tabulate
cd /root/github/QuickRunCUDA

cat <<'EOF'
B300 Single-warp single-op latency reference (cycles, locked 1500 MHz):

OPERATION                  CYCLES   COMMIT
-------------------------- -------- ----------
HMMA chained (single)      20       V6 J1 (9c34e95)
HMMA ILP=8 saturated       8        V6 J1
FFMA (chained)             4-6      V4 (catalog)
LDS (broadcast)            10.6     V6 G6 (4498386)
LDS (sequential)           ~13      V5 various
LDG L1 hit                 18.6     V6 G5 (34d5521)
LDG L1 miss / L2 hit       ~95      V6 G2 (feb97fc)
LDG L2 miss / HBM          54 ns    V6 G3 (2ed3250)
LDC (cmem)                 13       V6 G6
LDCU (cmem broadcast)      3.4      V6 K5 (88eb08b)
LDTM (TMEM)                ~17      V6 B2 (b6648e2)
SHFL.bfly chained          24       V6 J4 (2101fcc)
REDUX.SUM                  1        V4 various
MUFU.RCP chained           ~15      V6 A3 (17cf0d4)
MUFU.RCP ILP=8             ~3       V5 J series
IADD3 chained              3.1      V6 A2 (aa8b7eb)
F2FP cvt (any narrow FP)   2.7-5.4  V6 H1/H2 (c702139, bb569c1)
INT8 cvt                   2.9      V6 H4 (f9ed4ea)
cp.async.cg + commit+wait  759      V6 I1 (943b7d7)
cp.async.bulk (1 KB)       137      V6 I4 (6fff122)
ATOMS local                55       V6 J5 (ff19f64)
ATOM.E.STRONG.GPU global   75       V6 J5
mbarrier arrive            24       V5 A6 (a549d92)
mbarrier try_wait spin     ~140 ns  V6 A1 (acd3f46)
__syncthreads              30       V5 A6
cluster.barrier (CSIZE 2-8) ~390    V5 C2 (1f193aa)
DSMEM peer access          214      V5 C1 (9c8fec8)
NVLink peer atomic         0.54 G/s V5 I4

LAUNCH OVERHEAD:
Direct kernel launch       2.3 us   V6 K1
Cluster launch (≥4)        2.6 us   V6 K2 (FASTER!)
Cooperative launch         4.1 us   V6 K1
Persistent kernel RTT      2.77 us  V6 E1
CUDA Graph (warm launch)   1.0 us   V6 D4
EOF
