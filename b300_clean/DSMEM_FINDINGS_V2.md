# DSMEM Comprehensive Characterization — V11-V20 Findings

**Clock**: 1920 MHz locked. **GPU**: B300 SXM6 AC (sm_103a). **Date**: 2026-04-21.

## Latency (1-thread dep-chain, DCE-immune)

| Test | Local SMEM | DSMEM cx=2 | DSMEM cx=3..8 |
|------|-----------|-----------|---------------|
| V12 avg cy/load | 24.00 (σ=0) | 214.75 (σ=0.1) | ~177.1-177.5 flat |
| Ratio vs local | 1.0× | 8.95× | 7.39× |

**New finding vs 04_dsmem**: cluster=2 is 21% slower than cluster≥3 (not 11%). Cluster≥3 is flat.

## CTA→SM placement (cx=8, deterministic)

```
CTA 0 → SM 0    CTA 4 → SM 32
CTA 1 → SM 1    CTA 5 → SM 33
CTA 2 → SM 16   CTA 6 → SM 48
CTA 3 → SM 17   CTA 7 → SM 49
```

Four (SM×2) TPC pairs across 4 GPCs. Placement is 100% stable across launches.

## Per-pair READ latency (V15/V16, 8×8 matrix)

| Pair type | Latency range | Notes |
|-----------|--------------|-------|
| Min observed | 164.80 cy (SM32↔SM33) | TPC pair in GPC 2 |
| Max observed | 204.97 cy (SM16↔SM17) | TPC pair in GPC 1 |
| Spread | 40 cy (25%) | same ΔSM=1 bucket varies 40 cy |
| Symmetry | ≤3 cy asymmetry s→d vs d→s | effectively symmetric |

Distance-bucket averages flat (183-186 cy for |ΔSM|=1..49). Routing topology hides physical distance.

## Per-pair WRITE latency (V20, 1 thread fence.sc.cluster)

| | Range | Spread |
|---|---|---|
| Writes | 34.1 - 35.1 cy | 3% (uniform) |

**Key finding**: writes are pair-uniform, reads are pair-dependent. Write and read use different routing paths.

## Per-pair ATOMIC latency (V20, atom.add.shared::cluster)

Same pattern as reads: 188-239 cy, fastest at SM32↔SM33 (188.9), slowest at SM16↔SM17 (239.0). Atomics inherit read-path asymmetry since they return values.

## Throughput / Bandwidth

### Single-CTA BW (32t, varying ILP)
| ILP | cy/load/thread | per-CTA BW |
|-----|---------------|-----------|
| 1   | 6.23  | 1.23 GB/s |
| 4   | 2.14  | 3.59 GB/s |
| 8   | 1.42  | 5.42 GB/s |

### Multi-warp ring (V20, CX=8, all CTAs active)
| N warps | per-CTA cy/load | per-CTA BW |
|---------|----------------|-----------|
| 1       | 2.15 | 3.57 GB/s |
| 2       | 1.25 | 6.17 GB/s |
| 4       | 0.90 | 8.52 GB/s |
| 8       | 0.76 | 10.10 GB/s |

**Per-CTA read-port ceiling: ~10 GB/s**. Aggregate CX=8: ~80 GB/s.

### 128-bit loads (V20)
| ILP | cy/128b-load | per-CTA BW |
|-----|-------------|-----------|
| 1   | 9.04 | 3.40 GB/s |
| 4   | 3.13 | 9.81 GB/s |

128b helps — but same per-CTA ceiling ~10 GB/s as many-warp 32b.

### Ring writes (V19)
| ILP=4, N active | cy/store | aggregate BW |
|-----------------|----------|--------------|
| N=1 | 0.37 | 20.7 GB/s |
| N=8 | 0.37 | 165.3 GB/s |

Writes ~5-6× faster per-instruction than reads. **Aggregate write BW ~165 GB/s** at CX=8.

### Ring atomics (V19)
| ILP=4, N active | cy/atom | aggregate |
|-----------------|---------|-----------|
| N=1 | 2.03 | 0.95 Gatom/s |
| N=8 | 2.03 | 7.58 Gatom/s |

Atomics scale perfectly linearly with N (no contention). Atomic ILP=1..8 sweep: 6.52 → 1.21 cy/atom (5.4× ILP speedup).

## Contention (key finding)

### NO contention when CTAs read DIFFERENT peers (ring)
V17: N=1 to N=8 all at 188 cy. **Ratio 1.00× across N.** Dedicated point-to-point routing.

### Contention ONLY at hot-spot (same-DST)
V19: N=2 → N=8 all reading CTA 0 — aggregate BW plateaus at ~14 GB/s.
**Per-CTA serving port caps at ~14 GB/s** when many readers target it.

## Fence cost (implied)

V20 write-latency test uses `fence.sc.cluster` between stores and timing end. The 34-cy write latency *includes* the fence. So fence ≈ small (few cy).

## Crash pattern (V12 confirms 04_dsmem)

| Cluster | CHAIN_LEN crash threshold |
|---------|--------------------------|
| cx=2 | ~50 iters works, higher risky |
| cx=3..8 | ~5 iters safe, 10+ crashes, 15+ always |

Required: trailing barrier.cluster to keep all CTAs alive while one times.

## Summary table

| Metric | Value |
|--------|-------|
| Local SMEM latency | 24.00 cy |
| DSMEM min latency | 164.80 cy (SM32↔SM33) |
| DSMEM max latency | 204.97 cy (SM16↔SM17) |
| DSMEM avg latency | ~177-185 cy |
| Latency ratio DSMEM/local | 7.4-8.9× |
| DSMEM write latency | 34.5 cy (uniform) |
| DSMEM atomic latency | 188-239 cy |
| Per-CTA read BW ceiling | ~10 GB/s |
| Aggregate cluster read BW | ~80 GB/s |
| Aggregate cluster write BW | ~165 GB/s |
| Contention (ring, diff targets) | 1.00× (no slowdown N=1..8) |
| Contention (hot-spot same target) | 1.80× at N=8 |
| Crash threshold cx=4..8 | CHAIN_LEN=8 |

## V21-V24 additions

### Aggregate BW ceiling (V21, ring, 1920 MHz)
| Config | Reads (total) | Writes (total) |
|--------|--------------|----------------|
| 1 warp × ILP=1 | 9.29 GB/s | 43.3 GB/s |
| 1 warp × ILP=4 | 22.08 GB/s | 169.5 GB/s |
| 1 warp × ILP=8 | 28.21 GB/s | 174.0 GB/s |
| 2 warp × ILP=4 | 33.99 GB/s | 337.8 GB/s |
| 4 warp × ILP=4 | 40.16 GB/s | **560.7 GB/s (PEAK)** |
| 8 warp × ILP=4 | 36.69 GB/s | 416.3 GB/s |
| 8 warp × ILP=8 | 38.43 GB/s | 332.2 GB/s |

Read saturates ~40 GB/s. Writes peak at **~560 GB/s** (4w × ILP=4).

### Fence costs (V22, 1 thread, CL=20)
| Fence | cy/fence |
|-------|----------|
| fence.acq_rel.cluster | 320 |
| fence.sc.cluster | 320 |
| fence.sc.gpu | 320 |
| fence.sc.sys | 2870 |

cluster/gpu same cost. sys adds ~9× for PCIe coherence.

### Broadcast (V22, CTA 0 source, CX-1 readers)
| ILP | Aggregate BW |
|-----|--------------|
| 1   | 7.92 GB/s |
| 4   | 13.95 GB/s |
| 8   | 15.02 GB/s |

Per-source serving port caps ~15 GB/s.

### Self-read via mapa (V23)
`mapa.shared::cluster(me)` then load = 54 cy (vs 24 cy local). Pays LD.E global-window cost even for own CTA.

### Local SMEM atomic scope (V24, CL=100)
| Scope | cy/atom |
|-------|---------|
| .cta (default) | 29.97 |
| .cluster | 31.40 |
| .gpu | 29.97 |

`.cluster` adds 1.4 cy (5%), `.gpu` is free.

### Concurrent DSMEM+local (V24, warp 0 local + warp 1 DSMEM)
| | Isolated | Concurrent |
|---|---------|-----------|
| Local warp | 24.04 | 24.62 cy/load (+2.4%) |
| DSMEM warp | ~200 | 263.38 cy/load (+30%) |

**DSMEM reader IS slowed when peer's local SMEM is busy.** Local is barely affected. Asymmetric.

### Hot-spot writes (V24, all → CTA 0, ILP=4)
| N senders | cy | aggregate BW | Slowdown |
|-----------|-----|-------------|----------|
| 2 | 241 | 20.4 GB/s | 1.00× |
| 8 | 243 | 141.6 GB/s | 1.01× |

**No contention on write hot-spot** (unlike read hot-spot). Posted writes via dedicated lane.

### Critical architectural insights
1. **Read path ≠ write path**. Reads: per-source serving port (~15 GB/s cap). Writes: per-destination posted (no cap visible).
2. **Reads share peer's local-SMEM BW** (+30% when peer busy). Writes don't.
3. **Fence.sc.cluster costs 320 cy** — producer/consumer patterns pay this per sync point.
4. **CTA→SM is deterministic** {0,1,16,17,32,33,48,49} for cx=8.
5. **Dedicated point-to-point routing** — N=1..8 ring readers: 1.00× (no contention).

### Producer-consumer handoff (V27)
| Mechanism | cy/msg | us/msg | Notes |
|-----------|--------|--------|-------|
| barrier.cluster per msg | 613 | 0.320 | legacy |
| Batched fence (1 fence per CL writes) | 80 amortized | 0.042 | **best** |

Use batched fence + single barrier.cluster to amortize 320 cy fence over many messages.

### TMA multicast (V28, cp.async.bulk.shared::cluster.multicast)
| Pattern | Time | Effective BW | Per-CTA |
|---------|------|-------------|---------|
| Multicast TMA (8-way) | 0.35 µs | 375.80 GB/s | 47 GB/s |
| Per-CTA TMA (8 independent) | 0.39 µs | 334.56 GB/s | 42 GB/s |

TMA multicast 16 KB × 8 CTAs in 0.35 µs. **Peak DSMEM data-movement primitive for cluster GEMM/softmax**.

### Routing: L2 or dedicated? (V13 + ncu)
DSMEM LD.E loads: 5330 L2 sectors per 1280 loads → ~4 sectors per load = matches global load coalesced access. **DSMEM DOES route through L2** (`LD.E` via global address window).

dram__bytes < 4 KB → most hits stay in L2 (recently-written SMEM values coherent through L2).

## Tests (standalone .cu files)
- `v11/v12_dsmem_canonical.cu` — initial + multi-sample latency
- `v15_dsmem_pairs.cu` — all-pair read latency (8×8 matrix)
- `v16_dsmem_smid_topo.cu` — SM-ID topology capture
- `v17_dsmem_m2m.cu` — many-to-many contention
- `v18_dsmem_loadtest.cu` — BW scaling threads/ILP
- `v19_dsmem_hotspot.cu` — hotspot + writes + atomics
- `v20_dsmem_deeper.cu` — per-pair R/W/atom + multi-warp + 128-bit
- `v21_dsmem_ceiling.cu` — BW ceiling (reads 40 GB/s, writes 560 GB/s)
- `v22_dsmem_fences_broadcast.cu` — fence scopes + broadcast + atom scope
- `v23/v24_dsmem_corners.cu` — self-read + concurrent + hot-spot writes
- `v25_dsmem_patterns.cu` — all-to-all + store widths + all-reduce
- `v26_dsmem_crash_debug.cu` — crash thresholds (30/30 succ, not reproducible)
- `v27_dsmem_mbarrier.cu` — producer-consumer handoff comparison
- `v28_dsmem_tma_multicast.cu` — TMA cp.async.bulk.multicast

## Headline numbers

| Quantity | Value |
|----------|-------|
| DSMEM read latency (best pair) | 164.8 cy (85.8 ns) |
| DSMEM read latency (avg) | ~180 cy (94 ns) |
| DSMEM read latency (worst pair) | 204.97 cy (107 ns) |
| Local/DSMEM latency ratio | ~7.5× |
| DSMEM write latency | 34 cy (17.7 ns) — pair-uniform |
| DSMEM atomic latency | 188-239 cy — pair-dep like reads |
| Fence.sc.cluster | 320 cy (167 ns) |
| Cluster barrier cost | ~613 cy per-msg |
| Peak DSMEM read BW (aggregate) | ~40 GB/s |
| Peak DSMEM write BW (aggregate) | ~560 GB/s |
| **Peak TMA multicast** | **375 GB/s effective** |
| Hot-spot read serving cap | ~15 GB/s |
| Hot-spot write | unlimited scale |
| 8-CTA all-reduce via DSMEM | 3 µs |

All SASS + clock64 verified DCE-immune. 1920 MHz locked.
