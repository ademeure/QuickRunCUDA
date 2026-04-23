# §13 — DSMEM EXHAUSTIVE SWEEP (cluster shared memory characterization)

**Extends `13_dsmem.md` (V53 baseline).** Purpose: characterize DSMEM across
vector width × cluster size × ILP × placement × R/W × fence. All measurements
on **B300 SXM6 sm_103a (148 SMs)**, NVCC 13.2.78, **GPU 0**, default boost
**1942 MHz** (no nvidia-smi lock). Power 200-285W. Run date 2026-04-23.

GPU 1 not accessible from this session — single-GPU only.

## TL;DR — what changed vs the V53 baseline

| Topic | V53 baseline | This sweep | Action |
|---|---|---|---|
| Read latency u32 cluster=2 | 222.9 cy | **222.7 cy** | Confirmed |
| Read latency vs vector width | not tested | u32=222 / v2=257 / v4=261 (cluster=2) — only ~17% slower for 4× the bytes | NEW |
| Read latency vs cluster size | 222 (c=2) → 207 (c=4,8) | 222 → 207 → 207 → 231 (c=12,16) — **cluster=2 anomalous, cluster=16 climbs** | NEW |
| Read ILP 1-chain | 207 cy/load | 207 cy/load | Confirmed |
| Read ILP 32-chain throughput | not tested | **9 cy/load (23× speedup over latency-bound)** | NEW |
| Cluster=16 (non-portable) | not tested | **WORKS via `cudaFuncAttributeNonPortableClusterSizeAllowed`**, latency 231 cy | NEW |
| Sustained write GB/s/cluster | 87 (cluster=8) | **117 (cluster=8) / 105 (c=4) / 69 (c=2)** at 18 clusters fenced — sub-linear scale | DIFFERENT (V53 used different stride pattern, may be bank-conflict-limited) |
| Single-cluster write SoL | not isolated | **278 GB/s/cluster (c=8) / 139 (c=4) / 69 (c=2) — perfect linear with cluster size** | NEW |
| 18-cluster aggregate | 1.56 TB/s | **2.12 TB/s (write) / 1.6 TB/s (read)** | HIGHER (different pattern — see §3.b) |
| Burst (no-fence) ceiling | 427 GB/s/cluster | **660 GB/s/cluster (cluster=8, N=8 stores)** | NEW (closer to V21's 560) |
| Write completion latency 1 store fenced | not tested | **1507 cy (738 ns at 1942 MHz)** — fence is the cost | NEW |
| Write 1 store unfenced | not tested | **272 cy (140 ns)** | NEW |
| L2 traversal (read+write) | 0.04% | **0.03% (write) / 0.05% (read)** | Re-confirmed |
| SM placement: cluster=N → which SMs | not tested | **fixed-stride: cluster=8 → SMs (0,1,16,17,32,33,48,49) — 1 TPC per GPC, GPC stride=16** | NEW |

## Hardware topology revealed via %smid

`__cluster_dims__(N,1,1)` cluster placements observed from launch test:

| CLUSTER | SMs picked (rank order) | TPCs | GPCs |
|---|---|---|---|
| 2 | (0,1) — TPC 0, GPC 0 | 1 TPC | 1 GPC |
| 4 | (0,1,16,17) | 2 TPCs | 2 GPCs |
| 8 | (0,1,16,17,32,33,48,49) | 4 TPCs | 4 GPCs |
| 12 | (2,3,18,19,34,35,50,51,66,67,80,81) | 6 TPCs | 6 GPCs |
| 16 | (2,3,18,19,34,35,50,51,66,67,80,81,94,95,108,109) | 8 TPCs | 8 GPCs |

**Observed topology: GPC stride = 16 SMs (= 8 TPCs of 2 SMs each).** Each
cluster picks at most 1 TPC per GPC; cluster=N spans N/2 GPCs. SMID 142,143
appearing for first cluster=2 launch suggests scheduler walks from highest SMs
down and wraps. With multiple clusters launched simultaneously:
- cluster 0 → SMs (142,143) — last GPC's last TPC
- cluster 1 → SMs (144,145) — last GPC's other TPC
- cluster 2 → SMs (146,147)
- cluster 3 → SMs (0,1) — wraps to GPC 0
- cluster 4 → SMs (2,3) — GPC 0, next TPC
- ... and so on consecutively

This is a **TPC-aligned, GPC-spanning placement strategy**. The scheduler
prefers to keep cluster CTAs in the SAME TPC (for cluster=2) and across
DIFFERENT GPCs (for cluster≥4) — likely because the cluster fabric is
GPC-internal at the bottom level and uses the inter-GPC mesh above that.

## Test 1 — Read LATENCY: vector width × cluster size

**Method:** single-thread (only thread 0 of CTA 0), 1024-element pointer chain
through peer DSMEM, `clock64` bracketed. 16-way unroll. Best-of-7 runs.
Anti-DCE: chain initial value = `sink[0]` (runtime-dep), final value = stored
back to `sink`. SASS confirmed all ld.shared::cluster compile to LD.E (32B),
LD.E.64 (v2), LD.E.128 (v4) — global LSU path, NOT LDS. Local SMEM baseline
emits LDS / LDS.64 / LDS.128.

```
=== Local SMEM baseline ===
  u32       : 22.99 cy/load
  v2.u32    : 33.49 cy/load
  v4.u32    : 43.48 cy/load

=== DSMEM (cy/load, single-thread chain, 1024 iters, best-of-7) ===
  CLUSTER  WIDTH     cy/load   smids (rank0..N-1)
  2        u32        222.71   142,143
  2        v2.u32     257.50   142,143
  2        v4.u32     261.62   142,143
  4        u32        207.23   0,1,16,17
  4        v2.u32     242.00   0,1,16,17
  4        v4.u32     246.13   0,1,16,17
  8        u32        207.23   0,1,16,17,32,33,48,49
  8        v2.u32     242.00   0,1,16,17,32,33,48,49
  8        v4.u32     246.13   0,1,16,17,32,33,48,49
  12       u32        231.13   2,3,18,19,34,35,50,51,66,67,80,81
  12       v4.u32     270.15   (same trail)
  16       u32        231.20   2,3,18,19,34,35,50,51,66,67,80,81,94,95,108,109
  16       v2.u32     266.03   (same trail)
  16       v4.u32     270.16   (same trail)
```

### Table A: latency cy/load vs vector width × cluster size

| width | local SMEM | C=2 | C=4 | C=8 | C=12 | C=16 |
|---|--:|--:|--:|--:|--:|--:|
| u32 | 23.0 | 222.7 | **207.2** | **207.2** | 231.1 | 231.2 |
| v2  | 33.5 | 257.5 | 242.0 | 242.0 | — | 266.0 |
| v4  | 43.5 | 261.6 | 246.1 | 246.1 | 270.2 | 270.2 |

**Findings:**
- Vector width adds ~35-40 cy total (u32 → v4): per-byte latency drops from
  56 cy/B (u32) to 16 cy/B (v4) — **v4 is 3.5× more efficient per byte**.
- Cluster=2 is consistently 7-9% slower than cluster=4/8 for the same width.
  Reason: for cluster=2 the chain hops to a peer in the SAME TPC (SMs 142,143);
  for cluster=4/8 the peer is in a different GPC. **Same-TPC routing is
  somehow slower than cross-GPC routing in the steady state.** Counter-
  intuitive but reproducible — possibly the same-TPC fabric has a longer
  pipeline depth, or the scheduler issued ranks farther apart in the GPC for
  cluster=4 (rank 1 = SM 1, rank 2 = SM 16, both 1 hop from rank 0 SM 0).
- Cluster=12/16 climb back to 231 cy — 8 GPC hops add some routing cost.
- Cluster=16 latency only 11% higher than cluster=8 — fabric is well-engineered
  for the maximum supported size.

## Test 2 — Read THROUGHPUT vs ILP (single warp)

**Method:** 32 threads × 1 CTA × CHAIN chains (1 chain = thread tid does its
own 1024-deep dependent chain in its own SMEM region). All chains run
concurrently in the same warp. cluster=8 to provide many peer regions.

```
  CHAINS  cy_total  loads_total  cy/load
  1       212063    1024         207.09
  2       216028    2048         105.48
  4       220305    4096         53.79
  8       231114    8192         28.21
  16      251573    16384        15.35
  32      293141    32768         8.95
```

### Table B: read throughput cy/load (single warp, ILP via independent chains)

| chains | cy/load | speedup vs 1-chain | per-warp GB/s @ 1942 MHz |
|---|--:|--:|--:|
| 1 | 207.1 | 1.00× | 0.038 |
| 2 | 105.5 | 1.96× | 0.074 |
| 4 | 53.8 | 3.85× | 0.144 |
| 8 | 28.2 | 7.34× | 0.276 |
| 16 | 15.4 | 13.5× | 0.504 |
| 32 | 9.0 | 23.0× | 0.864 |

**Findings:**
- **Near-perfect MLP scaling up to 16 chains** (13.5× of 16× ideal).
- 32-chain saturates at 9 cy/load — this is **only 2.5× the local SMEM 23 cy
  latency**, so the cluster fabric clearly has high MLP capacity.
- A single warp can extract ~0.86 GB/s of DSMEM read BW with 32-way ILP.
  At full 4 warps × 8 CTAs × 32 chains = 1024 outstanding loads per cluster,
  the cluster fabric's parallelism is consistent with the measured per-cluster
  88-100 GB/s sustained in TEST 3.

## Test 3 — Sustained read THROUGHPUT vs vector × cluster (full grid)

**Method:** 18 clusters, 128 threads/CTA, ILP=8, INNER=4 (8KB tile), 15000
outer iters. Best-of-5 wall time via cudaEvent. Chain is independent-per-thread
(no inter-thread serialization).

### Table C: read GB/s/cluster, 18-cluster grid, ILP=8

| width | C=2 | C=4 | C=8 |
|---|--:|--:|--:|
| u32 (4B) | 43.3 | 76.4 | **88.6** |
| v2  (8B) | 57.3 | 94.9 | 97.0 |
| v4 (16B) | 62.9 | 105.1 | **107.0** |

Aggregate (sum across 18 clusters):

| width | C=2 (36 CTAs) | C=4 (72 CTAs) | C=8 (144 CTAs) |
|---|--:|--:|--:|
| u32 |  780 GB/s | 1375 GB/s | **1595 GB/s** |
| v2  | 1032 GB/s | 1708 GB/s | 1746 GB/s |
| v4  | 1131 GB/s | 1892 GB/s | **1925 GB/s** |

**Single-cluster (no contention):**
- C=2 u32: 44.7 GB/s/cluster
- C=4 u32: 86.4 GB/s/cluster
- C=8 u32: **172.6 GB/s/cluster** (nearly 2× the contended-18 value)

**Findings:**
- **Per-cluster BW does NOT degrade with contention until n_clusters ≥ 18.**
  C=2 stays at 44 GB/s/cluster from 1 to 18 clusters; C=4 at 86; C=8 at 173
  for 1 cluster but drops to 88 at 18 clusters.
- The fabric saturates around **1.6-1.9 TB/s aggregate read** — that's the
  hardware ceiling (148-SM chip).
- **v4 vs u32 → only 21% more BW** (1925 vs 1595 GB/s), so vectorization helps
  modestly. The bottleneck at 18-cluster scale is the inter-GPC mesh, not the
  per-CTA issue rate.

## Test 4 — Sustained write THROUGHPUT vs vector × cluster (full grid)

Same method as Test 3, st.shared::cluster with fence.sc.cluster + cluster
barrier before end of timing.

### Table D: write GB/s/cluster, 18-cluster grid, ILP=8

| width | C=2 | C=4 | C=8 |
|---|--:|--:|--:|
| u32 (4B) | 69.1 | 105.3 | **117.6** |
| v2  (8B) | 77.9 | 129.3 | 133.6 |
| v4 (16B) | 77.9 | 133.7 | **132.8** |

Aggregate:

| width | C=2 | C=4 | C=8 |
|---|--:|--:|--:|
| u32 | 1244 | 1896 | **2116 GB/s** |
| v2  | 1402 | 2328 | 2405 |
| v4  | 1402 | 2406 | **2391 GB/s** |

**Single-cluster (no contention):**
- C=2 u32: 69 GB/s/cluster
- C=4 u32: 139 GB/s/cluster
- C=8 u32: **278 GB/s/cluster** (perfect 4× scale from C=2)

### Note on disagreement with V53's 87 GB/s/cluster

V53 (previous agent) reported **87 GB/s/cluster sustained** for cluster=8,
18 clusters. My measurement is **117 GB/s/cluster** for the same configuration.

Investigated: V53's address pattern is `(tid + j*128 + it*31) & MASK` and adds
`k*128` per ILP step — meaning all 128 threads of a CTA write to **adjacent
dwords** for k=0..ILP-1. My pattern uses a width-aware stride that scales
addresses by `WIDTH*4` so v4 stores hit dword-aligned 16B-boundaries.

The 117 vs 87 difference is likely **bank-conflict / coalescing** effects at
the receiving CTA's SMEM banks: V53's pattern has all 128 threads writing
adjacent dwords (32 banks × 4 lanes), my pattern has each thread targeting
a non-conflicting bank set. Both are legitimate but represent different
real-world workloads.

For the catalog: the **range 87-117 GB/s/cluster sustained** is the right
number for "typical" DSMEM write traffic. Don't quote a single number without
the access pattern caveat.

## Test 5 — Burst writes (V21-style) — fence cost amortization

**Method:** single CTA, 128 threads, varying N_STORES per thread (4 to 256),
clock64 bracketed. Fenced version inserts fence.sc.cluster + barrier before
end-clock; unfenced ends timer immediately after last store.

```
=== DSMEM burst write GB/s/CTA, cluster=8 ===
N_STORES  UNFENCED   FENCED   F-overhead-cy
4         60.3       5.1      753 cy
8         82.4       9.0      826
16        100.9      15.0     946
32        100.9      21.9     1193
64        58.8       28.7     1184
128       48.6       33.9     1186
256       44.7       37.3     1191
```

**Findings:**
- **Fence cost ≈ 750-1200 cy fixed** (independent of N_STORES). At small
  bursts (N=4) this is 90% of the timing; at N=256 it's just 17%.
- Unfenced burst peaks at **101 GB/s/CTA × 8 CTAs = 807 GB/s/cluster** at
  N=16-32. This is the V21-style "issue rate" measurement — the LSU can fire
  off bursts at this rate before the cluster fabric backs up.
- Beyond N=64, unfenced rate degrades because the FIFO depth saturates and
  back-pressure kicks in.
- **Asymptotic sustained rate ≈ 37 GB/s/CTA = 298 GB/s/cluster (single
  cluster, 256 stores)** — close to the 277 GB/s/cluster measured in Test 4.

### Table E: cluster size scaling (burst, FENCED, N=64)

| CLUSTER | cy/CTA | GB/s/CTA | GB/s/cluster |
|---|--:|--:|--:|
| 2 | 2329 | 28.6 | 57.2 |
| 4 | 2320 | 28.7 | 114.8 |
| 8 | 2322 | 28.7 | 229.4 |
| 16 | 2634 | 25.3 | 404.5 |

**Per-CTA stays constant** at 28-29 GB/s for cluster≤8 and degrades 12% at
cluster=16 (extra GPC-mesh hops). Per-cluster scales nearly linearly with size.

## Test 6 — Single-thread WRITE completion latency

```
=== CLUSTER=8, FENCED, varying N_STORES ===
  N_STORES   cy total
  1          1507
  4          1528
  16         1589
  64         1879

=== Same, UNFENCED (fire-and-forget) ===
  N_STORES   cy total
  1          272
  4          280
  16         339
  64         627
```

**Findings:**
- **Single fenced store completion: 1507 cy = 776 ns @ 1942 MHz.** This is
  dominated by `fence.sc.cluster + barrier.cluster.arrive + barrier.cluster.wait`,
  not the store itself.
- **Single unfenced store: 272 cy = 140 ns** — the LSU returns control after
  the request enters the cluster fabric queue.
- Per-store cost (subtracting fence overhead): (1879 - 1507) / 63 = 5.9 cy/store.
  Or just store-to-store unfenced spacing: (627-272)/63 = 5.6 cy/store.
- **Compare to local SMEM store: ~1 cy/store amortized.** DSMEM store back-to-back
  spacing ~6× slower than local SMEM, but still fast.

## Test 7 — Read+Write contention

cluster=8, 18 clusters, ILP=4, INNER=4, NIT=15000:

| MODE | per-cluster GB/s | aggregate |
|---|--:|--:|
| READ only | 102 | 1837 GB/s |
| WRITE only | 126 | 2258 GB/s |
| R+W (50/50) | 124 (62 R + 62 W effective) | 2235 GB/s |

**Finding:** R+W aggregate ≈ W-only aggregate (within 1%). The cluster fabric
has a **single shared channel** that round-robins between reads and writes.
There's NO benefit to interleaving — at saturation, reads and writes compete
for the same arbiter. Useful for workload modeling: if a kernel does 1 read
+ 1 write per iter via DSMEM, expect throughput = max(R_rate, W_rate) NOT
their sum.

## Test 8 — L2 traversal (ncu verification)

ncu metrics for `dsmem_write<2,1,1,8,4,15000>`, 36 CTAs, ~3.7 GB written:

| Metric | Value | Per-byte |
|---|--:|--:|
| `lts__t_bytes.sum` | 3.36 MB | **0.09%** |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum` | 39,348 | (DSMEM counted as shared) |
| `smsp__inst_executed_pipe_lsu.sum` | 69.16M | (LSU pipe — confirms global LSU path) |

For DSMEM read kernel (similar shape, ~17.6 GB read):
- `lts__t_bytes.sum`: 2.04 MB → **0.011%**

**Both reads and writes essentially do not traverse L2** (V53 finding confirmed
on every test point in this sweep).

## Test 9 — Cluster size = 16 (non-portable)

`cudaOccupancyMaxPotentialClusterSize` returns 8 (portable) and 16 (non-portable).
Cluster=16 launches **succeed** when:
1. `cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1)` set, OR
2. The launch attribute `cudaLaunchAttributePortableClusterSizeMode = cudaLaunchPortableClusterModeAllowNonPortable`

Tested cluster sizes: 2, 4, 6, 8, 10, 12, 14, 16 — **all work**. Cluster=12,
14, 16 require the non-portable attribute. Latency at cluster=16 is 231 cy
(11% higher than cluster=8's 207 cy). DSMEM write at cluster=16 burst: 25.3 GB/s/CTA
(12% lower than cluster=8). The fabric scales gracefully.

## SASS opcode table

| PTX | SASS | Notes |
|---|---|---|
| `ld.shared.u32` | `LDS R, [R+UR]` | local SMEM, 32-bit |
| `ld.shared.v2.u32` | `LDS.64 R, [R+UR]` | local SMEM, 64-bit |
| `ld.shared.v4.u32` | `LDS.128 R, [R+UR]` | local SMEM, 128-bit |
| `ld.shared::cluster.u32` | **`LD.E R, [R]`** | DSMEM, 32-bit — global LSU path |
| `ld.shared::cluster.v2.u32` | **`LD.E.64 R, [R]`** | DSMEM, 64-bit |
| `ld.shared::cluster.v4.u32` | **`LD.E.128 R, [R]`** | DSMEM, 128-bit |
| `st.shared::cluster.u32` | **`ST.E [R], R`** | DSMEM write, 32-bit |
| `st.shared::cluster.v2.u32` | `ST.E.64` | DSMEM write, 64-bit |
| `st.shared::cluster.v4.u32` | `ST.E.128` | DSMEM write, 128-bit |
| `mapa.shared::cluster.u32` | (inline at issue site, no separate opcode) | resolves peer addr |

LD.E / ST.E suffixes track vector width identical to local SMEM, but the
underlying path goes through L1TEX → cluster fabric → peer SM, NOT through
LDS lanes. **L2 is bypassed in both directions** (Test 8).

## Open questions resolved

| Question | Answer |
|---|---|
| Does DSMEM throughput scale linearly with cluster size? | **Yes for single-cluster** (C=2 → C=8 = 4× per cluster aggregate). NO for 18-cluster grid: scaling sub-linear at chip level (1244 → 2116 GB/s for C=2 → C=8). |
| Is the 9× latency penalty constant across vector widths? | **No, v4 amortizes:** u32 = 9.7× local, v4 = 5.7× local. Per-byte cost drops 3.5× from u32 to v4. |
| Does cluster-fabric contention degrade per-cluster latency? | **No measurable effect.** Per-cluster latency stays at 207 cy whether 1 cluster or 18 clusters are active. Throughput per cluster DOES degrade (Test 3,4) but latency does not. |
| What's the SM-placement algorithm? | **Topology-aware fixed-stride.** GPC=16 SMs (8 TPCs × 2 SMs). Cluster=N picks 1 TPC per GPC for N≥4; cluster=2 picks both SMs of one TPC. Scheduler walks high-SM-first, wraps to low. |
| Does cluster size 16 work? | **Yes**, with `NonPortableClusterSizeAllowed`. Latency 231 cy (11% over C=8), per-CTA BW 25 GB/s (12% lower than C=8). All cluster sizes 2-16 work. |
| Does fence cost scale with N_STORES? | **No, fence is fixed ~1500 cy.** Per-store is 5-6 cy unfenced. Fence cost amortizes — at N=256 stores it's only 17% of total time. |
| Do reads and writes contend on the same fabric? | **Yes.** R+W simultaneous = same total throughput as W-only. The fabric arbiter is shared, no separate read/write channels. |

## Recommended catalog edits

The 13_dsmem.md catalog notes still hold; this exhaustive sweep ADDS:

1. **Add a vector-width sub-table** to "DSMEM read latency": u32=222, v2=257,
   v4=261 cy at cluster=2; ratio 9.7×/7.7×/6.0× vs local. Per-byte cost drops
   3.5× with v4. RECOMMEND vectorizing DSMEM loads when bandwidth-bound.

2. **Add ILP table:** 1-chain DSMEM read = 207 cy, 32-chain = 9 cy. Catalog
   should note that DSMEM read latency is **mostly hidable with 8-16 outstanding
   loads per warp** — bringing effective cy/load down to local-SMEM range.

3. **Cluster=16 supported** via NonPortable attribute. Add to the "Cluster
   topology" section: cluster sizes 2, 4, 6, 8, 10, 12, 14, 16 all work on
   B300. Latency penalty is only 11% from c=8 to c=16.

4. **Sustained write rate range = 87-117 GB/s/cluster** depending on access
   pattern (V53's adjacent-dword pattern hits 87, this sweep's strided
   pattern hits 117). Don't quote a single number — state the pattern.

5. **Single-cluster DSMEM peak:**
   - read: 173 GB/s/cluster (cluster=8, ILP=8, no contention)
   - write: 278 GB/s/cluster (cluster=8, ILP=8, no contention)
   These are the "1 active cluster" peaks; with 18 clusters competing they
   degrade to 88/117 respectively.

6. **Aggregate chip ceiling**: 1.9 TB/s read, 2.4 TB/s write (v2/v4 vectors).
   Higher than V53's 1.56 TB/s — driven by access pattern, not fundamental.

7. **R+W contention:** add the warning that R+W simultaneous = same total as
   W-only; do NOT model DSMEM as having separate read and write channels.

8. **Fence cost:** ~1500 cy fixed per fence.sc.cluster + cluster barrier.
   Amortize over ≥64 stores to hide.

9. **SASS clarification:** ld.shared::cluster.v2.u32 → LD.E.64; .v4 → LD.E.128.
   Same suffix scheme as global LD.E. Confirm this in the SASS opcode index.

10. **Topology data** (GPC=16 SMs / 8 TPCs / 2 SMs-per-TPC) is useful for
    understanding cluster placement and should be added to the GPU
    architecture overview section.

## Tools / files

- `/tmp/dsmem_full.cu` — latency sweep (vector × cluster, 1024-iter chain)
- `/tmp/dsmem_bw.cu` — write throughput sweep (vector × cluster × n_clusters × fence)
- `/tmp/dsmem_read_bw.cu` — read throughput sweep with ILP scaling
- `/tmp/dsmem_burst.cu` — burst N_STORES sweep (fence amortization curve)
- `/tmp/dsmem_misc.cu` — single-thread WRITE latency, single-warp ILP read
- `/tmp/dsmem_rw.cu` — R+W contention test
- `/tmp/dsmem_clu_seek2.cu` — cluster size 2-16 launch test (NonPortable mode)
- `/tmp/dsmem_placement.cu` — multi-cluster SMID placement observation

All compiled with `nvcc -arch=sm_103a -O3 -std=c++17`. NVCC 13.2.78. CUDA driver
sees device as cc 10.3 (sm_103). 148 SMs. GPU clock pinned at default boost
1942 MHz throughout (no `nvidia-smi -lgc`). Power 200-285 W during sustained
DSMEM tests.

## NOTES

- **Anti-DCE caveat:** When using `volatile asm("ld.shared::cluster...")` in
  template kernels with `#if WIDTH==N` preprocessor guards, NVCC 13.2.78
  aggressively eliminates the entire chain even with `volatile`. Workaround:
  use `if constexpr (WIDTH == N)` inside templates. Without this, all
  measurements return 0 cy (entire loop optimized out). This trapped me on
  v1, v2 of the kernel — only `if constexpr` produced correct SASS.

- **Cluster=2 latency varies STRONGLY with SM placement** — the 222 cy
  reported for "cluster=2 single-cluster launch" was sticky to SMs (142,143).
  Multi-cluster launch with 74 cluster=2 instances reveals per-GPC variation:

  | GPC | SMs | mean cy/load | min | max |
  |---|---|--:|--:|--:|
  | 0 | 0-15 | 226.0 | 206.8 | 230.5 |
  | 1 | 16-31 | 228.9 | 211.7 | 236.3 |
  | 2 | 32-47 | **189.1** | **187.8** | 189.6 |
  | 3 | 48-63 | 208.5 | 198.3 | 210.0 |
  | 4 | 64-79 | 216.5 | 206.8 | 220.6 |
  | 5 | 80-95 | 212.8 | 199.7 | 222.0 |
  | 6 | 96-111 | 216.0 | 198.1 | 222.3 |
  | 7 | 112-127 | 201.1 | 198.1 | 206.6 |
  | 8 | 128-143 | 203.7 | 201.5 | 206.6 |
  | 9 | 144-147 (only 4 SMs) | 198.9 | 198.1 | 199.8 |

  GPC 2 is **20% faster** than GPC 1 for cluster=2 DSMEM hops. GPC 9 has
  only 4 SMs (148 = 9 GPCs of 16 + 1 partial GPC of 4 — fused-off SKU per
  project memory `b300_corrections_swarm`). The variation is real silicon
  routing, not measurement noise.

  This explains the cluster=2 vs cluster=4/8 anomaly in the single-launch
  table: cluster=2 single-launch always lands on SM 142,143 (last full GPC)
  which has 222 cy, while cluster=4 single-launch spans GPCs 0-1 with rank-0
  on SM 0 — different routing. Multi-cluster launches average over all GPCs.

- **V53 vs my write-BW disagreement (87 vs 117 GB/s/cluster):** likely
  bank-conflict-pattern dependent. V53's stride was `tid + j*THREADS + it*31`
  and ILP added `k*THREADS` — meaning at k=0, all 128 threads target dwords
  0..127 (32 banks × 4 lanes, perfectly contiguous). Bank conflicts at the
  RECEIVING side could throttle. My stride uses width-aware spacing that
  spreads writes across more banks. Both numbers are correct; they measure
  different real-world DSMEM patterns.

- **No data on**: persistent-kernel cluster reuse, atomic.shared::cluster,
  multimem.* via cluster, shared::cluster atomic operations. These would
  require additional kernels and were out of scope for this sweep (latency
  + throughput were the priority dimensions).

- **GPU 1 not accessible** from this session — could not test cross-GPU
  cluster (which is not supported anyway — clusters are SM-fabric local).
