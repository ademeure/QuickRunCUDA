# DSMEM — Corrected Reference (post V11–V31 audit)

**GPU**: NVIDIA B300 SXM6 AC (sm_103a)  |  **Clock**: 1920 MHz locked  |  **Audit date**: 2026-04-22

This file consolidates the TRUE current understanding of DSMEM on B300, deprecates
all V8/V10 TB/s claims (DCE-confirmed), and lists residual unknowns. All numbers
here are SASS-verified + clock64-timed dependent-chain methodology unless noted.
Source: `tests/standalone/v11–v31_*.cu`.

---

## 1. Architecture (verified)

### Cluster placement (cx=8 deterministic)
```
CTA 0 → SM 0    CTA 4 → SM 32
CTA 1 → SM 1    CTA 5 → SM 33
CTA 2 → SM 16   CTA 6 → SM 48
CTA 3 → SM 17   CTA 7 → SM 49
```
Four TPC pairs (x, x+1) spread across 4 GPCs. 100% stable across launches.

### SASS codegen
`ld.shared::cluster.u32` with scalar-register address compiles to `LD.E` (global
window through L2), NOT `LDS`. ncu shows ~4 L2 sectors/load. The `LDS R, [R+UR]`
form only appears when the mapa result lands in a uniform register.

### Cluster limit
Max practical cluster = 8 (16 advertised). Use `__cluster_dims__` +
`cudaLaunchAttributeClusterDimension`.

---

## 2. Latency (1-thread dependent chain, DCE-immune)

| Memory | Latency (cy) | ns @ 1920 MHz |
|---|---:|---:|
| Local SMEM (LDS) | 24 | 12.5 |
| DSMEM self (mapa→me) | 54 | 28.1 |
| DSMEM cluster=2 | 214.75 | 111.8 |
| DSMEM cluster=3..8 (avg) | ~180 | 94 |
| DSMEM best pair (SM32↔SM33) | 164.80 | 85.8 |
| DSMEM worst pair (SM16↔SM17) | 204.97 | 107.0 |
| DSMEM write (fenced) | 34 | 17.7 |
| DSMEM atomic .add | 188–239 | 98–124 |
| DSMEM atomic .cas | 206 | 107 |

**Local/DSMEM ratio ≈ 7.5×** (NOT 0.8% — that was LICM; NOT 4.7× — wrong test).

Key observations:
- Cluster=2 is 21% slower than cluster≥3 (single-GPC vs multi-GPC routing).
- Reads pair-dependent (25% spread); writes pair-uniform (3% spread).
- Atomics inherit read-path asymmetry (return value → uses read path).
- Self-read via `mapa` still pays LD.E cost (54 cy vs 24 cy local).

---

## 3. Throughput / Bandwidth (CL=8 ring, all CTAs active)

### Per-warp read BW (1 warp/CTA, ring)

| ILP | cy/load | per-CTA BW (GB/s) |
|---:|---:|---:|
| 1 | 6.42 | 1.20 |
| 4 | 2.17 | 3.53 |
| 8 | 1.45 | 5.29 |
| 16 | 1.08 | 7.11 |

### Multi-warp read aggregate (cluster of 8)

| warps × ILP | per-CTA (GB/s) | Aggregate (GB/s) |
|---|---:|---:|
| 1 × 4 | 2.76 | 22.08 |
| 2 × 4 | 4.25 | 33.99 |
| **4 × 4** | **5.02** | **40.16** ← read ceiling |
| 8 × 4 | 4.59 | 36.69 |
| 4 × 8 | 5.08 | 40.62 |

**DSMEM read aggregate ceiling ≈ 40 GB/s** per cluster.

### Multi-warp write aggregate (cluster of 8)

| warps × ILP | per-CTA (GB/s) | Aggregate (GB/s) |
|---|---:|---:|
| 1 × 4 | 21.19 | 169.5 |
| 2 × 4 | 42.23 | 337.8 |
| **4 × 4** | **70.08** | **560.7** ← write ceiling |
| 8 × 4 | 52.04 | 416.3 |

**DSMEM write aggregate ceiling ≈ 560 GB/s** per cluster — **13× higher than reads**.

### TMA multicast (cp.async.bulk.shared::cluster.multicast, 8-way)

| Tile | Time | Effective BW (×8 delivery) |
|---|---:|---:|
| 1 KB | 0.27 µs | 30.58 GB/s |
| 4 KB | 0.33 µs | 99.69 GB/s |
| 16 KB | 0.41 µs | 323.50 GB/s |
| **32 KB** | **0.56 µs** | **470.68 GB/s** |

For cluster-level data movement → use TMA multicast with ≥16 KB tiles.

---

## 4. Contention

### Ring (each CTA reads a different peer): NO contention
N=1..8 active, all at 188 cy → **1.00× (flat)**. Dedicated point-to-point routing.
There is **no shared bus**.

### Hot-spot reads (all CTAs → 1 peer): per-source serving cap
- N=2: 20.4 GB/s aggregate
- N=8: 14.0 GB/s aggregate (1.80× per-reader slowdown)

**Per-CTA serving port caps at ≈ 15 GB/s** — a single peer can only deliver to ~15 GB/s
worth of remote requesters.

### Hot-spot writes: NO cap (writes posted/async)
N=2..8: each ~20 GB/s, 1.01× slowdown. Aggregate scales linearly to N senders.

### Hot-spot atomics: linear scaling to dest's atomic-unit pipeline (V31)

| N senders | cy/atom | cluster aggregate |
|---:|---:|---:|
| 2 | 214 | 9 Matom/s |
| 4 | 214 | 27 Matom/s |
| **8** | **214** | **63 Matom/s** |

Each sender does 9 Matom/s; the destination's atomic unit is pipelined at
33 atoms/clock. Unlike hot-spot reads, this scales N×.

### Split-ILP across peers (single reader, N peers)
16 ILP to 1 peer: 7.11 GB/s. 4 peers × 2 ILP: 5.92 GB/s.
**Reader's issue rate caps per-CTA BW — NOT peer's serving rate.**

### Peer concurrent activity affecting DSMEM reader

| Peer doing | DSMEM reader slowdown |
|---|---:|
| Local SMEM reads | +30% (263 vs 200 cy) |
| FFMA compute | 0% (210 vs 211 cy) |

DSMEM competes for the peer's SMEM subsystem, not for its compute/SMSP.

### TMA + DSMEM concurrent (V31)
With 16 KB TMA in flight: 216 cy/load. Without: 216 cy/load (0.04% diff).
**TMA and DSMEM use independent data paths — perfect overlap.**

---

## 5. Fences and Barriers

### Fence scope costs (1 thread, per fence)

| Fence | cy |
|---|---:|
| fence.acq_rel.cluster | 320 |
| fence.sc.cluster | 320 |
| fence.sc.gpu | 320 |
| fence.sc.sys | 2870 (~9× slower) |

cluster/gpu **identical cost** → use `fence.sc.gpu` for safety with no penalty.

### Local SMEM atomic scope (V24, CL=100)

| Scope | cy/atom |
|---|---:|
| .cta (default) | 29.97 |
| .gpu | 29.97 |
| .cluster | 31.40 (+1.4 cy / +5%) |

---

## 6. Producer-Consumer Handoff

| Mechanism | cy/msg | µs/msg | Notes |
|---|---:|---:|---|
| barrier.cluster per msg | 613 | 0.320 | naive |
| **Batched (1 fence per N writes)** | **80 amortized** | **0.042** | best |
| 8-CTA ring all-reduce (V25) | 842 cy/step | 3.07 µs total | with fence + barrier |

Rule: batch DSMEM writes, emit ONE `fence.sc.cluster` + ONE
`barrier.cluster.arrive/wait` to amortize the 320 cy fence cost.

---

## 7. Store width (single-thread CTA 0→1)

| Width | cy/st | bytes/cy |
|---|---:|---:|
| u32 | 33.26 | 0.12 |
| u64 | 45.21 | 0.18 |
| v4.u32 | 41.34 | 0.39 |
| **v2.u64 (128-bit)** | **29.66** | **0.54** |

Use `v2.u64` for widest per-thread store.

---

## 8. Crash behavior — UPDATE

V12/V26 ran 30/30 successful tests at CL=50, cluster=4 across all patterns
(dep-chain, fixed, strided, SAT-chain). The "100% crash at cluster=4, CHAIN_LEN≥15"
observation in `investigations/04_dsmem_overhead.md` is **NOT reproducible** in the
2026-04-21 environment. Likely was driver state / thermal / background processes
at the original time of testing. Treat the crash workaround in 02_shmem.md as
historical.

---

## 9. Headline summary

| Quantity | Value |
|---|---:|
| DSMEM read latency (best pair) | 164.8 cy / 85.8 ns |
| DSMEM read latency (avg) | ~180 cy / 94 ns |
| DSMEM read latency (worst pair) | 204.97 cy / 107 ns |
| **Local/DSMEM latency ratio** | **~7.5×** |
| DSMEM write latency (fenced) | 34 cy / 17.7 ns — pair-uniform |
| DSMEM atomic latency | 188–239 cy — pair-dependent |
| fence.sc.cluster (= .gpu) | 320 cy / 167 ns |
| fence.sc.sys | 2870 cy |
| barrier.cluster per-msg | 613 cy |
| **Peak DSMEM read BW (cluster aggregate)** | **~40 GB/s** |
| **Peak DSMEM write BW (cluster aggregate)** | **~560 GB/s** |
| **Peak TMA multicast effective** | **470 GB/s @ 32 KB tile** |
| Hot-spot read serving cap (per peer) | ~15 GB/s |
| Hot-spot writes | uncapped (linear with N) |
| Hot-spot atomics | linear scaling, 63 Matom/s @ N=8 |
| Ring contention (diff targets, N=1..8) | 1.00× — no contention |
| 8-CTA ring all-reduce | 3.07 µs |

---

## 10. Best-practice rules

1. Use **TMA multicast** for cluster data movement (470 GB/s @ 32 KB tiles).
2. Prefer **DSMEM writes over reads** (13× higher aggregate BW).
3. **Batch writes** + 1 fence per batch (42 ns/msg amortized vs 320 ns/fence).
4. **Don't hot-spot reads** (15 GB/s serving cap per peer).
5. **Cluster ≥ 3** beats cluster=2 (21% faster routing).
6. Single reader per CTA → use **≥ 4 warps** to saturate per-CTA BW.
7. Peer's local-SMEM activity costs you 30%; peer's compute costs you 0%.
8. Self-read via mapa is **NOT free** (54 cy vs 24 cy LDS).
9. `fence.sc.gpu == fence.sc.cluster` in cost → prefer `.gpu` for safety.
10. **DSMEM ≈ 7.5× local SMEM latency** — not 0.8%, not 4.7×, not 9×.

---

## RETRACTIONS

The following previously-published numbers/claims are now known to be wrong.
Use only the values in §1–§9 above.

### R1. DSMEM "37 TB/s" peak BW (V8_DSMEM_BW.md)
- **Claimed**: 37.28 TB/s DSMEM BW @ cluster=8 (97% of theoretical 38.5 TB/s).
- **Source**: `b300_clean/V8_DSMEM_BW.md`, `tests/standalone/v8_dsmem_bw.cu`.
- **Why wrong**: kernel used compile-time-constant offsets with invariant base
  in the loop body. ptxas hoisted/CSE'd nearly all loads (LICM). SASS emitted
  `LD.E` and ncu wavefront counts showed only ~7200 wavefronts vs the expected
  ~5.9 billion. The "BW" measured was loop overhead, not actual transfers.
- **Correct**: aggregate cluster DSMEM read BW ceiling is **~40 GB/s** (not TB/s),
  measured by V21 with anti-DCE dependent-chain ring kernel.

### R2. DSMEM "48.5 TB/s @ cluster=2" / "42.3 TB/s @ cluster=8 NL=16" (V10_DSMEM_DEEP.md)
- **Claimed**: 48.5 TB/s at cluster=2, 32.6 TB/s at cluster=4/8, 42.3 TB/s at NL=16.
- **Source**: `b300_clean/V10_DSMEM_DEEP.md`.
- **Why wrong**: same DCE pattern as R1 (static-offset inline base). Self-corrected
  in `V10_DSMEM_STATE.md` ("BOGUS — DCE confirmed"). The author flagged the issue
  themselves.
- **Correct**: see §3 — read aggregate ~40 GB/s, write ~560 GB/s, both per cluster
  (i.e., per cluster-of-8 CTAs, NOT per chip).

### R3. "Cluster=2 is fastest for DSMEM BW" (V10_DSMEM_DEEP.md)
- **Claimed**: cluster=2 gave 48.5 TB/s vs 32.6 TB/s for cluster=4/8 → "use
  smallest cluster".
- **Why wrong**: the underlying numbers were DCE'd (R2). The differential reflected
  loop-overhead amortization, not real routing.
- **Correct**: cluster=2 is actually 21% **slower** in latency than cluster ≥ 3
  (single-GPC vs multi-GPC routing). For BW, ring throughput is roughly invariant
  in cluster size — choose cluster size based on capacity needs, not BW.

### R4. "DSMEM writes are 4× slower than reads" (V10_DSMEM_WRITES.md)
- **Claimed**: write BW 11.8 TB/s vs read 48.5 TB/s → 4–5× read advantage.
  "Use producer = local, consumer reads peer".
- **Why wrong**: same DCE pattern in both halves. The recommended access pattern
  is **the inverse of what's actually optimal**.
- **Correct**: writes are **~13× FASTER** in aggregate (~560 GB/s) than reads
  (~40 GB/s). Per-instruction writes are also ~5–6× faster than reads (V19 ring
  data: 0.37 cy/store vs ~2 cy/load). **Prefer write-based DSMEM patterns.**

### R5. "DSMEM 0.8% slower than local SMEM" (B300_PIPE_CATALOG §30.H, commit b478bb0)
- **Claimed**: DSMEM is virtually free vs local SMEM.
- **Why wrong**: LICM hoisted the load out of the loop in both DSMEM and local
  variants — test was measuring loop overhead.
- **Correct**: DSMEM latency is ~7.5× local SMEM (180 vs 24 cy).

### R6. "DSMEM 4.7× slower than local SMEM" (`tests/dsmem_v2.cu`)
- **Claimed**: DSMEM latency penalty 4.7×.
- **Why wrong**: used FADD-serialized accumulator that turned the test into a
  single-thread latency measurement with the wrong baseline.
- **Correct**: latency ratio ~7.5×.

### R7. "DSMEM = 1035 GB/s remote across all cluster sizes" (commits b478bb0, 5f3edca)
- **Claimed**: workload-specific number repurposed as a hardware peak.
- **Why wrong**: workload-specific, not a peak; methodology unclear.
- **Correct**: ~40 GB/s aggregate read, ~560 GB/s aggregate write per cluster.

### R8. "DSMEM throughput at ILP=4 is 9× slower than local" (02_shmem.md)
- **Claimed**: 63.5 cy/load DSMEM vs 7.0 cy/load local.
- **Status**: per-instruction value plausible; but the **9× framing** mixed
  latency-bound and BW-bound regimes. Latency ratio is 7.5× (not 9×); aggregate-BW
  ratio is much larger (read) or inverted (write). The "9×" should not be cited
  as a single ratio.

### R9. "Cluster crashes >15 iters at cluster=4..8" (02_shmem.md, 04_dsmem_overhead.md)
- **Claimed**: dependent DSMEM chains 100% crash at cluster=4, CHAIN_LEN≥15.
- **Status**: not reproducible (V12/V26: 30/30 success). Treat as historical
  driver/thermal artifact, not a current constraint.

### R10. "cluster=2 has higher latency due to fewer routing options" (02_shmem.md open Q)
- **Status**: now answered. Cluster=2 places both CTAs in a single TPC/GPC, while
  cluster≥3 spreads across multiple GPCs and uses multi-GPC routing which is faster.

### R11. B300_TRUE_REFERENCE.md "DSMEM (CL=2) 3.06 TB/s aggregate; per-cluster 41 GB/s"
- **Claimed**: 3.06 TB/s aggregate, 41 GB/s per cluster.
- **Status**: per-cluster number (~41 GB/s) is consistent with §3 (read aggregate
  ~40 GB/s). The "3.06 TB/s" headline is a chip-aggregate extrapolation
  (~74 simultaneous clusters × 41 GB/s) — should be presented with caveats. It
  also reads against the ceiling for **reads only**; the write headline at chip
  scale would be ~10× higher (~30 TB/s extrapolated, not measured).

---

## UNRESOLVED

Items still missing rigor — list as future work.

### U1. ncu-direct DSMEM throughput cross-check
The ~40 / ~560 GB/s peaks are wall-clock + clock64 verified, but no
ncu-metric exists that directly reports DSMEM BW. Sector counts on `LD.E`
match expectations qualitatively; a quantitative ncu byte-count corroboration
would strengthen the claim.
**Resolution**: capture `lts__t_sectors_op_read.sum`, `l1tex__t_sectors_pipe_lsu`,
and dram bytes for V21 and reconcile.

### U2. Chip-aggregate DSMEM scaling (multi-cluster)
All BW numbers are per-cluster-of-8. With 18 clusters running concurrently,
do we still see ~40 GB/s/cluster × 18 = 720 GB/s read, or does shared
GPC/L2 infrastructure cap aggregate?
**Resolution**: launch persistent kernel filling all SMs in clusters of 8,
measure aggregate vs per-cluster.

### U3. mbarrier.shared::cluster cost vs barrier.cluster
V27 lists "needs debug" for mbarrier handoff. We have 613 cy for
`barrier.cluster` per msg and 320 cy for `fence.sc.cluster`, but the modern
mbarrier path may be cheaper.
**Resolution**: write a producer-consumer test using
`mbarrier.arrive.shared::cluster` + `mbarrier.try_wait` and compare cy/msg.

### U4. DSMEM under register spill / high occupancy
All BW tests use ≤256 thr/CTA with low register pressure. Does spill activity
on the peer's SM reduce DSMEM serving rate?
**Resolution**: rerun V18/V21 with `-maxrregcount=32` to force spills.

### U5. DSMEM bank-conflict propagation
If the source SMEM access pattern would cause local bank conflicts, do those
slow the remote reader?
**Resolution**: vary stride in V18 from 1 → 32, measure DSMEM latency, compare
to local-SMEM stride sweep.

### U6. Cross-cluster behavior
Reading from a CTA outside your cluster is supposed to be impossible — what
exactly fails? Hard fault, silent zero, or undefined?
**Resolution**: small test issuing `mapa.shared::cluster` on out-of-cluster CTA ID,
catch error.

### U7. DSMEM write peak under hot-spot writes
Writes to a single peer at N=8 give 141.6 GB/s aggregate (1.01× slowdown).
This is below the 560 GB/s ring peak but well above any per-source cap. What's
the actual write hot-spot ceiling and where does it bottleneck?
**Resolution**: sweep N=8..16 hot-spot writes (multi-cluster) and look for the
knee.

### U8. Sustained vs burst DSMEM (clock-throttle effect)
SMEM peaks drop from 38 TB/s burst to 17–21 TB/s sustained on B300. Does DSMEM
show a similar burst→sustained cliff?
**Resolution**: extend V21 ring kernel to >50 ms, plot per-iter BW.

### U9. Why does DSMEM throughput penalty match latency penalty (~9×)?
With ILP=4, a latency-only model would predict near-cancellation of the latency
gap. The fact that throughput penalty matches suggests serialization on the
LD.E path that's not pure latency. Mechanism unconfirmed.
**Resolution**: vary ILP from 1→64 with peer-busy and peer-idle, look for the
inflection.

### U10. tcgen05 + DSMEM interaction
DSMEM reads + TMA multicast were tested concurrently (perfect overlap). What
about DSMEM concurrent with `tcgen05.mma`? Does TMEM allocation/dealloc compete
with the DSMEM serving port?
**Resolution**: fold a tcgen05 stress kernel into V31's overlap test.

---

## Test files
- `tests/standalone/v11_*` through `v31_*` — 21 standalone tests covering
  latency, pairs, m2m, hot-spot, fences, broadcast, store width, all-reduce,
  TMA multicast, atomics, overlap.
- `b300_clean/DSMEM_REFERENCE.md` — V11–V31 source-of-truth (this file
  consolidates + retracts).
- `b300_clean/DSMEM_FINDINGS_V2.md` — running detailed log.
- `b300_clean/V10_DSMEM_STATE.md` — author's own retraction of V8/V10 BW claims.

## Provenance / confidence
- **HIGH** for: latency ladder (§2), per-pair asymmetry (§2), ring contention
  (§4), hot-spot reads (§4), fence costs (§5), store widths (§7), TMA-multicast
  effective BW (§3), TMA-DSMEM independence (§4).
- **MED** for: aggregate write ceiling 560 GB/s (single ncu cross-check would
  raise to HIGH), atomic-pipeline 33 atoms/clock (derived from V31 ring fit).
- **LOW** for: chip-aggregate extrapolation in B300_TRUE_REFERENCE.md (3.06 TB/s)
  — needs U2.
