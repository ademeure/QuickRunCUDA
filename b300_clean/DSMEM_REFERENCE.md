# B300 DSMEM Complete Reference

**GPU**: NVIDIA B300 SXM6 AC (sm_103a) | **Clock**: 1920 MHz locked | **Date**: 2026-04-21

All data SASS-verified + clock64-timed (DCE-immune dependent-chain methodology). 30+ kernel
variants across `tests/standalone/v11-v30_*`. Resolves prior contradictions in AUDIT_NOTES.md,
B300_PIPE_CATALOG.md, V5-V10 findings.

## 1. Architecture

### Cluster placement (cx=8 deterministic)
```
CTA 0 → SM 0   |  CTA 4 → SM 32
CTA 1 → SM 1   |  CTA 5 → SM 33
CTA 2 → SM 16  |  CTA 6 → SM 48
CTA 3 → SM 17  |  CTA 7 → SM 49
```
Four TPC pairs (x, x+1) spread across 4 GPCs. Stable across launches.

### SASS codegen
`ld.shared::cluster.u32` with scalar-register address compiles to **`LD.E`** (via global window),
not LDS. Goes through L2 (ncu shows ~4 sectors/load).

### Cluster limits on B300
- Max cluster = 8 (16 advertised but 8 is practical)
- `__cluster_dims__` attribute + `cudaLaunchAttributeClusterDimension` at launch

## 2. Latency (1-thread dep-chain)

| Mem | Latency | Time@1920MHz |
|-----|---------|--------------|
| Local SMEM (LDS) | **24 cy** | 12.5 ns |
| DSMEM self (mapa→me) | 54 cy | 28.1 ns |
| DSMEM cluster=2 | 214.75 cy | 111.8 ns |
| DSMEM cluster=3..8 (avg) | **~180 cy** | 94 ns |
| DSMEM best pair (SM32↔SM33) | 164.80 cy | 85.8 ns |
| DSMEM worst pair (SM16↔SM17) | 204.97 cy | 107.0 ns |
| DSMEM write (fenced) | 34 cy | 17.7 ns |
| DSMEM atomic .add | 188-239 cy | 98-124 ns |
| DSMEM atomic .cas | 206 cy | 107 ns |

**Key findings**:
- Cluster=2 is 21% slower than cluster≥3 (single-GPC vs multi-GPC routing)
- Reads pair-dependent (25% spread); writes pair-uniform (3%)
- Atomics inherit read-path asymmetry
- Self-read via mapa still pays LD.E cost (2× local)

## 3. Throughput / Bandwidth

### Per-warp read BW vs ILP (1 warp per CTA, ring)
| ILP | cy/load | per-CTA BW |
|-----|---------|-----------|
| 1   | 6.42 | 1.20 GB/s |
| 4   | 2.17 | 3.53 GB/s |
| 8   | 1.45 | 5.29 GB/s |
| 16  | 1.08 | 7.11 GB/s |

### Multi-warp read BW (ring, all CTAs active)
| warps × ILP | per-CTA BW | Aggregate |
|-------------|-----------|-----------|
| 1 × 4 | 2.76 GB/s | 22.08 GB/s |
| 2 × 4 | 4.25 | 33.99 |
| 4 × 4 | **5.02** | **40.16** ← read ceiling |
| 8 × 4 | 4.59 | 36.69 |
| 4 × 8 | 5.08 | 40.62 |

### Write BW (ring, all CTAs active)
| warps × ILP | per-CTA | Aggregate |
|-------------|---------|-----------|
| 1 × 4 | 21.19 | 169.5 GB/s |
| 2 × 4 | 42.23 | 337.8 |
| 4 × 4 | **70.08** | **560.7** ← write ceiling |
| 8 × 4 | 52.04 | 416.3 |

### TMA multicast (cp.async.bulk.shared::cluster.multicast)
| Tile | Time | Effective (×8 delivery) |
|------|------|-------------------------|
| 1 KB | 0.27 µs | 30.58 GB/s |
| 4 KB | 0.33 µs | 99.69 GB/s |
| 16 KB | 0.41 µs | 323.50 GB/s |
| 32 KB | 0.56 µs | **470.68 GB/s** |

**For cluster-level data movement: use TMA multicast with ≥16 KB tiles.**

## 4. Contention

### Ring (diff targets): NO contention
N=1..8 active: 1.00× (188 cy flat). Dedicated point-to-point routing.

### Hot-spot reads (all → 1 peer): serving port cap
- N=2: 20.4 GB/s
- N=8: 14.0 GB/s aggregate, 1.80× slowdown per reader

**Per-CTA serving port caps ~15 GB/s.**

### Hot-spot writes: NO cap
N=2..8: all at 20 GB/s each (no slowdown). Writes posted/async.

### Split-ILP across peers (single reader, N peers)
16 ILP to 1 peer: 7.11 GB/s. 4 peers × 2 ILP: 5.92 GB/s. **Reader issue rate, not peer serving, caps per-CTA BW.**

### Peer concurrent activity
| Peer doing | DSMEM reader slowdown |
|-----------|----------------------|
| Local SMEM reads | +30% (263 vs 200 cy) |
| FFMA compute | 0% (210 vs 211 cy) |

DSMEM competes for peer's SMEM subsystem, not compute/SMSP.

## 5. Fences + Barriers

### Fence scope costs (1 thread, per fence)
| Fence | cy |
|-------|-----|
| fence.acq_rel.cluster | 320 |
| fence.sc.cluster | 320 |
| fence.sc.gpu | 320 |
| fence.sc.sys | 2870 (9× slower) |

cluster/gpu identical cost. Use sc.gpu for safety.

### Atomic scope (local SMEM)
| Scope | cy/atom |
|-------|---------|
| .cta (default) | 29.97 |
| .gpu | 29.97 |
| .cluster | 31.40 (+1.4 cy) |

## 6. Producer-Consumer Handoff

| Mechanism | cy/msg | Notes |
|-----------|--------|-------|
| barrier.cluster per msg | 613 | 320 ns/msg |
| Batched fence (1 per N writes) | ~80 amortized | **best, 42 ns/msg** |
| mbarrier.shared::cluster | (needs debug) | advanced |
| 8-CTA ring all-reduce | 3.07 µs / 842 cy/step | with fence + barrier |

**Rule**: batch DSMEM writes, emit ONE `fence.sc.cluster` + ONE `barrier.cluster.arrive/wait`
to amortize the 320 cy fence cost over many messages.

## 7. Store width (single-thread CTA 0→1)

| Width | cy/st | bytes/cy |
|-------|-------|----------|
| u32 | 33.26 | 0.12 |
| u64 | 45.21 | 0.18 |
| v4.u32 | 41.34 | 0.39 |
| v2.u64 (128-bit) | **29.66** | **0.54** |

Prefer v2.u64 for widest per-thread store.

## 8. Crash behavior

30/30 success at CL=50 cluster=4 across all patterns (dep-chain, fixed, strided, SAT-chain).
**04_dsmem's "100% crash" observation not reproducible** in 2026-04-21 environment —
likely was driver state / thermal / background procs at the time.

## 9. Summary rules

1. **Use TMA multicast for cluster data movement** (470 GB/s effective at 32 KB tiles)
2. **Use DSMEM writes over reads** (13× higher aggregate BW: 560 vs 40 GB/s)
3. **Batch writes + 1 fence per batch** (42 ns/msg amortized vs 320 ns per fence)
4. **Don't target a hot peer for reads** (14 GB/s serving cap)
5. **Cluster≥3 beats cluster=2** (21% faster routing)
6. **Single reader per CTA → use ≥4 warps** to saturate per-CTA BW (5 GB/s)
7. **Peer's local SMEM busy costs you 30%; peer's compute costs you 0%**
8. **DSMEM self-read is NOT a free shortcut** (54 cy vs 24 cy LDS)
9. **fence.sc.gpu == fence.sc.cluster** in cost → prefer .gpu for safety
10. **DSMEM latency ~180 cy, local SMEM 24 cy, ratio ~7.5×** — not 0.8%, not 4.7×

## 9.5 TMA overlap + hot-spot atomics (V31)

### DSMEM reads concurrent with TMA multicast
With 16 KB TMA in-flight: 216 cy/load | Without: 216 cy/load (0.04% diff)
**TMA and DSMEM loads use independent data paths**. Perfect overlap possible.

### Hot-spot atomics (all CTAs → same target)
| N senders | cy/atom | cluster agg |
|-----------|---------|-------------|
| 2 | 214 | 9 Matom/s |
| 4 | 214 | 27 Matom/s |
| 8 | 214 | **63 Matom/s** |

**Linear scaling (unlike hot-spot reads)**. Each sender 9 Matom/s, dest's atomic
unit pipelined at 33 atoms/clock. Contrast: hot-spot reads cap at 3.5 Gload/s
(serving port), hot-spot atomics scale N× to dest's pipelined atomic rate.

## 10. Test files
- `tests/standalone/v11-v31_*.cu` — 21 standalone tests
- `b300_clean/DSMEM_FINDINGS_V2.md` — detailed running log
- `b300_clean/DSMEM_MASTER_PLAN.md` — original roadmap
- `investigations/04_dsmem_overhead.md` — earlier foundation (2026-04-17)
