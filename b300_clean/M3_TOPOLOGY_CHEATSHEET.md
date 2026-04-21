# M3 — B300 Topology + Scheduling Cheatsheet (2026-04-21)

**Physical and logical topology of B300 SXM6 + how the scheduler maps work to it.**
Synthesizes I6, I7, I8, A2, F4, I3, I4 into a single cheatsheet.

---

## Hardware topology

```
  GPU (1 chip)
   ├── 148 SMs total
   ├── 12 HBM3E stacks
   ├── 18 NVLinks @ 53.125 GB/s (956 GB/s/dir)
   └── PCIe Gen 6 ×16

  Per SM
   ├── 4 SMSPs (sub-partitions)
   │    ├── Each SMSP issues ≤1 inst/cy from one warp (per A1)
   │    └── Issue port shared across pipes (FFMA + MUFU contend, per A2)
   ├── 256 KB unified L1 + SMEM
   │    └── Per-SM L1 effective ≈ 128 KB / 1024 lines (D2)
   ├── 65536 32-bit registers (255 max per thread, P3)
   └── TMEM 256 KB (zero-power idle, H7)

  TPC (Texture Processing Cluster) = 2 SMs
   └── SMs paired by adjacency: (0,1), (2,3), ..., (146,147)  [I8]

  GPC (Graphics Processing Cluster) = 8 TPCs = 16 SMs
   └── GPC-row stride = 16 SMs (I8 finding)
   └── 148 / 16 ≈ 9.25 GPC-rows worth (148 isn't multiple of 16)
```

---

## Block-to-SM scheduling (I6 #5145766)

**Consecutive blockIdx pairs → TPC siblings → next GPC row.**

For 148 blocks (1 per SM), the launch order is:
```
block 0  → SM 142    block 6  → SM 0      block 12 → SM 48     block 16 → SM 2
block 1  → SM 143    block 7  → SM 1      block 13 → SM 49     block 17 → SM 3
block 2  → SM 144    block 8  → SM 16     block 14 → SM 64     block 18 → SM 18
block 3  → SM 145    block 9  → SM 17     block 15 → SM 65     block 19 → SM 19
block 4  → SM 146    block 10 → SM 32
block 5  → SM 147    block 11 → SM 33
```

Pattern: **(N, N+1) pair → TPC; (N+2, N+3) pair → next TPC by +16 GPC-row stride**.
Last 6 SMs (142-147) are launched FIRST.

**Practical:** if you want two cooperative blocks (e.g. for DSMEM cluster work),
choose blockIdx 2K and 2K+1 — they will land on the same TPC.

---

## Warp-to-SMSP scheduling (I7 #3775f32)

**`warp_id % 4 = SMSP_id`** (round-robin in groups of 4).

For NWARPS=8 (2 warps per SMSP):
- warp 0, 4 → SMSP 0
- warp 1, 5 → SMSP 1
- warp 2, 6 → SMSP 2
- warp 3, 7 → SMSP 3

If warp K does heavy MUFU work, it slows its same-SMSP sibling
(FFMA on warp K+4) by ~10% — issue port shared.

**Practical:** for latency-critical work, place it on a unique SMSP:
mix work types across `warp_id % 4` quadrants to avoid sibling slowdown.

---

## Cluster topology (I8 #8f6f6e9)

For `__cluster_dims__(N, 1, 1)`:

| CSIZE | SM members | Pattern |
|------:|------------|---------|
| 2 | (0,1), (2,3), ..., (146,147) | TPC siblings (always +1 stride) |
| 4 | (0,1,16,17), (2,3,18,19), ... | 2 TPC pairs × +16 stride |
| 8 | (0,1,16,17,32,33,48,49), ... | 4 TPC pairs across 4 GPC-rows |

Cluster of 4 = TPC pair × 2 different GPC rows. Cluster of 8 = 4 TPC pairs across 4 rows.

**Cluster wraps when 148 SMs exhausted** (last cluster of CSIZE=8 has weird +31 jump).

**Cluster barrier cost** (F4 #8a6285b): ~395 cy floor, regardless of cluster size.
28× more expensive than `__syncthreads` (14 cy). PTX `barrier.cluster.sync` requires
ALL CTAs — no subset variant. Use `arrive_drop` to release individual CTAs early.

---

## Scheduler fairness (A2 #80c2d56)

- **Per-warp variance ≤ 0.04% for uniform work** (no GTO starvation observed).
- **Mixed work bias: ~2-3%** (LDS warps slightly slower than FFMA warps).
- **Heavy warp slows same-SMSP siblings**: NWARPS=8 with warp 0 = MUFU,
  warps 0 AND 4 took 292K cy (both → SMSP 0 with MUFU work);
  warps 1-3, 5-7 took 267K cy (other SMSPs).

**Conclusion:** B300 scheduler is loose-round-robin; SMSP-level work distribution dominates.

---

## Concurrency limits

| Limit | Value | Source |
|-------|------:|--------|
| Threads / SM | 2048 | spec |
| Registers / SM | 65536 | spec |
| Registers / thread | 255 hard cap | P3 (#0c2d0e7) |
| SMEM / SM | 228 KB | spec |
| Blocks / SM | 32 | spec |
| Active blocks (BSZ=256, NREGS=64) | ~2-4 observed (I1 partial) | I1 (#9e31770) |
| **HW dispatch slots** | **~128** | I4 (#5302086) |
| **Stream queue depth** | **~1024 launches** | I3 (#e83d506) |
| Cluster size max | 8 (verified) / 16 (per spec) | I8 |

**Concurrent kernel scaling (I4):**
- 1-128 streams: 74-92% efficiency (good speedup)
- 256 streams: 44% efficiency (HW slot ceiling reached)

**Stream queue (I3):** up to 900 launches non-blocking; 1024 starts blocking.

---

## Launch overhead ladder (L3, L4, L5, L6, L7)

| Operation | Cost |
|-----------|------|
| cudaLaunchKernel (no args) | **2.05 µs CPU** (L3) |
| cudaLaunchKernelEx (no/with attr) | 2.05 µs (L4 — same as legacy) |
| cudaLaunchKernel + 3 int args | +58 ns/arg = 2.22 µs |
| cudaDeviceSynchronize alone | 5.6 µs (L3) |
| cudaEventQuery on PENDING | **120 ns/poll** (L6) |
| cudaEventQuery on COMPLETED | 1254 ns (10× higher) |
| cudaStreamGetCaptureInfo | **24 ns** (idle/capturing identical) (L5) |
| cudaIpcGetMemHandle | **30 ns** (J4) |
| cudaGraph launch | 512 ns (4× faster than direct, prior) |

**L7 finding:** kernel launch is 99% CPU-bound; single-thread limit ~488K launches/sec.

---

## Multi-GPU topology (J1, J2, J3, J5)

- 2× B300 SXM6 with NV18 (18 NVLinks per side)
- Theoretical NVLink BW: 956 GB/s/dir, 1.9 TB/s bidir
- **Measured peak: ~740 GB/s** (77% of theoretical, via cudaMemcpyPeer 256 MB)
- **NVLink one-way latency: 1.55 µs** (J1)
- **Concurrent streams: marginal scaling** (1.23× from 1 → 16, J3) — fewer larger transfers preferred
- **GPU clocks fully independent** (J5) — no cross-GPU thermal/power coupling

---

## Quick selectors

**For cooperative pairs:** use `blockIdx.x` 2N and 2N+1 (same TPC).
**For latency-critical work:** ensure unique `warp_id % 4` across hot warps.
**For >128 concurrent kernels:** expect serialization on dispatch slots.
**For >1024 launches:** host blocks; use multiple streams or cudaGraph.
**For NVLink:** use few large cudaMemcpyPeer calls; avoid kernel-direct peer writes.
**For CPU↔GPU signal:** managed mem + CPU spin = 4.4 µs (L1).
**For GPU→CPU notify:** cuStreamWriteValue32 = 460 ns (L2 — 8× faster than kernel-write).
