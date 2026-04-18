# B300 HBM3E DRAM Bandwidth

**GPU:** NVIDIA B300 SXM6 AC, sm_103a, 148 SMs, 12 stacks HBM3E, 7680-bit usable bus (post-ECC), 287.4 GB capacity. All measurements at 2032 MHz boost (default, no clock lock).

**Theoretical peak (post-ECC):** 7672 GB/s = 7680 bit × 3996 MHz × 2 (DDR) / 8 / 1e9. Some catalog lines cite the pre-ECC 8183.8 GB/s (8192-bit × 3996 × 2 / 8); this is the wrong reference because ECC reserves 1/16 of the physical bus. Use 7672 GB/s as the spec ceiling.

---

## Headline numbers (HIGH confidence — `dram__bytes_*.sum.per_second` ncu-verified)

| Operation | Rate (TB/s) | % of 7672 GB/s spec | Recipe |
|---|---:|---:|---|
| Read peak | **7.29** | **95.0%** | v8 + per-warp coalesced + non-persistent (256 thr × 16384 blk for 4 GB, 32 iters/thread) |
| Write peak | **7.30** | **95.2%** | same recipe with `st.global.v8.b32` |
| cudaMemset (true DRAM rate) | **~7.30** | **~95%** | wall-clock 7.47–7.52 TB/s overstates by ~3% |
| Concurrent R+W (aggregate) | **6.74** | **88%** | sum of dram_read + dram_write during 1 kernel doing both — LOSS, not gain |

Read = write within 0.3% — **no inherent read/write asymmetry on B300 HBM3E** when measured correctly with the optimal recipe. Earlier "writes are 9–17% slower than reads" claims were artifacts of sub-optimal write patterns.

---

## Optimal recipe (HIGH confidence)

The "v8 + per-warp coalesced + non-persistent" pattern (commits a04d9c8, 5193694):

```cpp
// Each thread does 32 v8 stores = 32 × 32 B = 1024 B
// Per-warp pattern: 32 lanes × 32 B = 1024 B contiguous per iter
// 32 iters per thread = 32 KB written per warp
__global__ void w_v8_coalesced(int *data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *warp_base = data + warp_id * (32 * 1024 / 4);  // 32 KB per warp region
    int v = 0xab;
    #pragma unroll
    for (int it = 0; it < 32; it++) {
        int *p = warp_base + (it * 32 + lane) * 8;  // each lane writes 32 B
        asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
            :: "l"(p), "r"(v) : "memory");
    }
}
// Launch: <<<bytes / (256 * 1024), 256>>>   // 16384 blocks for 4 GB
```

SASS (ncu-verified): `STG.E.ENL2.256` (write) / `LDG.E.ENL2.256` (read).

**Why this recipe matters:**
- **v8 (256-bit) stores/loads** match the HBM channel burst granularity; 1024 B per warp per iter saturates per-channel queues without needing huge ILP per thread.
- **Per-warp coalesced** = 32 lanes write 32 contiguous 32-B chunks → one 1 KB cache-line burst per warp per iter. Breaking this into per-thread-consecutive layout halves BW.
- **Non-persistent**, exactly enough blocks to cover the buffer (16384 blocks × 256 thr for 4 GB) — no grid-stride loop, no tail effects, plenty of warps in flight.
- **Working set 4 GB** ≫ L2 (126 MB) so writes go straight to HBM (no L2 absorption inflating wall-clock).
- **`memory` clobber** in inline asm prevents compiler from eliminating the redundant stores; the read kernel uses XOR accumulator + impossible-branch store on `out[]` for anti-DCE.

---

## Sub-optimal patterns and why they fail

| Pattern | Measured | Why slower |
|---|---:|---|
| v4 + grid-stride, 296 CTA × 256 thr (persistent) | **6.20–6.27 TB/s** | Tail effects + lower per-warp burst width; original 08_hbm_write.md recipe |
| v4 + 8-ILP, 512 thr/blk, modest grid | 6.17 TB/s | 16-B stores under-fill HBM channel bursts |
| Per-thread consecutive (breaks per-warp coalesce) | **2.24 TB/s (-69%)** | Each warp scattered across 32 cache lines instead of one contiguous 1 KB |
| Naive 1-ILP read | 2.46 TB/s (32%) | Issue rate too low to hide HBM latency; not enough in-flight requests |
| TMA `cp.async.bulk` 8 KB chunks (WRITE) | 6.33 TB/s (82%) | Same ceiling as STG; the 17% gap to cudaMemset is NOT per-instruction width |
| TMA `cp.async.bulk` 8 KB chunks (READ, blocks=37888) | **7344 GB/s (95.7%)** | TMA read = LDG read within 0.3% — same HBM SoL via either method (commit below) |
| LDG.E.128 read peak (blocks=37888) | **7365 GB/s (96.0%)** | The HBM read SoL ceiling on B300 |
| Default-cache STG.E.128 default `st.global.v4` | 6.18–6.30 TB/s | One-direction ceiling for v4 + grid-stride is ~80% spec; v8 + per-warp lifts it to 95% |

**Pattern is everything.** Same instruction width (v8) with grid-stride hits 6.2 TB/s; with per-warp coalesced bursts hits 7.30 TB/s. The 17% gap is purely launch geometry / access pattern.

---

## Concurrent R+W (PARTIAL DUPLEX — MED-to-HIGH confidence)

HBM3E on B300 is a **shared bus, NOT full-duplex** (commit 30159f6, ncu-verified):

| Configuration | Read rate | Write rate | Aggregate |
|---|---:|---:|---:|
| Read alone (best recipe) | 7.29 TB/s | — | 7.29 |
| Write alone (best recipe) | — | 7.30 TB/s | 7.30 |
| **Concurrent R+W (1 kernel doing both)** | **3.39** | **3.35** | **6.74** |
| Two streams, R kernel + W kernel concurrent (commit 48d2a60, wall-clock) | 6.76 | 3.67 | "10.43" |

Two methodologies, two answers:
- **Single-kernel R+W via `dram__bytes_*.sum.per_second`**: aggregate **6.74 TB/s** = 0.92× one-direction. Concurrent R+W is **WORSE** than serializing: the bus shares cycles between reads and writes.
- **Two-stream R+W with per-stream wall-clock summing**: aggregate "10.43 TB/s" — but this **double-counts the overlap window**. Each stream's "duration" includes the period it shared the bus with the other; summing bytes/(overlapped-time) inflates the apparent aggregate. **The 10.4 number was retracted in commit 30159f6.**

Asymmetry observed in the two-stream case: reads ~preserved (-7%), writes halved (-50%). Plausible mechanism: HBM controller prioritizes reads on the shared bus (writes can be buffered to L2 / write-combiner; reads block dependent ops). Open question — needs a "short-R + long-W" reverse pattern to confirm prioritization.

**Practical:** for read-modify-write streaming kernels (axpy, copy), expect ~6.7 TB/s aggregate, NOT a duplex bonus. Architect kernels as read-PHASE then write-PHASE if separable.

---

## Wall-clock vs DRAM rate (the cudaMemset paradox)

Old docs claim `cudaMemset hits 7.47–7.52 TB/s = 97–98% of HBM3E peak`. ncu disagrees:

| Method | Wall-clock effective | ncu `dram__bytes_write.sum.per_second` |
|---|---:|---:|
| cudaMemset (4 GB) | 7.47 TB/s | ~7.30 TB/s |
| cudaMemset (16 GB, asymptote) | 7.54 TB/s | ~7.30 TB/s |
| User v8 + per-warp coalesced (4 GB) | 7.37 TB/s | 7.30 TB/s |

**Both cudaMemset and the optimal user kernel hit the same ~7.30 TB/s actual DRAM rate** (within 3%). The "98% memset" headline was wall-clock arithmetic: kernel time × bytes / time, where the kernel returns when stores hit L2 write-combine, not when they drain to DRAM. cudaMemset just exits a hair earlier than the user kernel because of its dispatch pattern.

**Sub-mystery: 17% wall-clock gap between cudaMemset and "STG.E.128 grid-stride" user kernels.** Investigated and not fully resolved (commits 8bdb3e6, b3dd31c, e1d77c6):
- Write-allocate hypothesis REFUTED (`dram__bytes_read.sum` = 103 KB during a 4 GB write, no RFO traffic).
- DMA hypothesis REFUTED (cudaMemset uses ALL 148 SMs; linear contention with FFMA load proves it's a kernel, not copy engine).
- TMA bulk write also caps at 82% (SAME ceiling as STG, so it's not per-instruction-width).
- Resolved by adopting per-warp coalesced v8 recipe — user kernels match cudaMemset's true DRAM rate within 3%. The "gap" was purely sub-optimal launch geometry in the prior comparison.

**Lesson: always use `dram__bytes_*.sum.per_second` from ncu** as the authoritative HBM rate. Wall-clock can over- or under-count depending on whether you're hitting L2 absorption, fire-and-forget timing, or read-modify-write overlap.

---

## Working-set / cache effects (HIGH confidence)

8-ILP, 512 thr/blk read sweep across WS sizes (commit cf226a6):

| WS | Effective BW | Tier |
|---|---:|---|
| 16 MB | 46.6 TB/s | L1 + L2 combined; per SM 314 GB/s |
| 64 MB | 23.0 TB/s | L2 plateau |
| 100 MB | 20.9 TB/s | L2 edge |
| 126 MB | 8.2 TB/s | **CLIFF — exactly L2 capacity** |
| 256 MB | 7.32 TB/s | DRAM-bound |
| 1024 MB | 7.12 TB/s | DRAM-bound (HBM peak) |
| 4 GB | 7.29–7.30 TB/s | DRAM-bound, true HBM3E ceiling |

Confirmed L2 capacity = `cudaDeviceProp.l2CacheSize = 126 MB` (132,644,864 B). Above the cliff, BW asymptotes to ~7.3 TB/s; below it, BW depends on how much fits in L1 (28–256 KB depending on carveout) and how the L2 hash distributes accesses. The L2-side detail (8 GPCs, address-hashed across 2 partitions) is treated in `06_l2_peak.md`.

---

## Theoretical accounting

- **Bus:** 12 × HBM3E stacks × 1024-bit physical = 12,288-bit raw; usable post-ECC = **7680 bit** (ECC reserves 1/16 = 1024 bit).
- **Memory clock:** 3996 MHz effective (= 1998 MHz I/O × 2 DDR), datasheet "8 Gbps/pin".
- **Spec peak:** 7680 × 3996 MHz × 2 / 8 / 1e9 = **7672 GB/s** post-ECC.
- **Best measured:** 7.30 TB/s = **95.2% of 7672 GB/s spec**.
- **Remaining 4.8% gap:** HBM controller / refresh / row-precharge / command-bus overhead. Per old §1 of B300_PIPE_CATALOG.md, NVIDIA-published HBM3E efficiency targets land around 90–93% for COPY (R+W); pure-direction streams reach ~95% on this implementation.

The 8183.8 GB/s figure (8192-bit pre-ECC) appears in some catalog sections as the "peak" and gives a misleading "75% of 8.17 TB/s" denominator when the real post-ECC bus is only 7672 GB/s. Anchor on 7672 GB/s.

---

## Findings being RETIRED (with reason)

| Old claim | Source | Why retired |
|---|---|---|
| "HBM write peak 8.5 TB/s with v4 + persistent" | catalog | Fire-and-forget L2 absorption — kernel exits when stores enter L2 write-combiner, not when DRAM drain finishes. Wall-clock measures L2 burst rate, not DRAM. |
| "HBM write peak 7.09 TB/s, 7.0 TB/s with v8 + 8 CTA/SM" | catalog §0 | Never ncu-verified; likely copied from read measurement. The v8 + 8 CTA/SM recipe actually measures ~6.1 TB/s in ncu (commit dbcdb0d). |
| "HBM write 3.4 TB/s = half of read" | catalog rule #12 | Used `st.global.v4.u32` with 1 GB working set; data fit partially in L2 so the 3.4 was the L2 drain-back rate, not HBM. |
| "HBM write peak 6.2 TB/s as the ceiling" (08_hbm_write.md, commit dbcdb0d) | investigations/08_hbm_write.md | Sub-optimal access pattern (296 CTA × 256 thr grid-stride). Used as "ground truth" because ncu-verified, but this was the ceiling of THAT pattern, not the architecture. **Correct ceiling is 7.30 TB/s with v8 + per-warp coalesced + non-persistent recipe** (commit a04d9c8). |
| "HBM write peak 6.17 TB/s = 80% of theoretical" | commit ffec751 | Same sub-optimal pattern issue; corrected by a04d9c8 to 7.30 TB/s. |
| "cudaMemset hits 97–98% of HBM peak" | commits a15154b, ccf6c54 | Wall-clock illusion; ncu shows true DRAM rate is ~7.30 TB/s = 95%, not 98%. |
| "cudaMemset uses copy engine / DMA fast path" | commits a15154b, b145ad7 | Refuted in 8bdb3e6 — cudaMemset uses ALL 148 SMs (linear contention sweep with FFMA load proves it). It's a driver-internal kernel, not DMA. |
| "Concurrent R+W aggregate 10.4 TB/s = 1.43× unidirectional (partial duplex bonus)" | commit 48d2a60 | Per-stream wall-clock summing double-counts the overlap window. ncu single-kernel R+W shows actual aggregate 6.74 TB/s = 0.92× = LOSS, not gain. Retracted in 30159f6. |
| "Reads 9% faster than writes" / "Writes 17% slower than reads" | 08_hbm_write.md, ffec751 | Both directions hit 7.30 TB/s with the optimal recipe. The "asymmetry" was a pattern-quality difference between the read and write benchmarks, not an HBM controller property. |
| "HBM peak 5.16 TB/s via ld.global" | bench_hbm_peak.cu, SKEPTICAL_LIST.md item #1 | 256 MB working set is mostly L2-resident after warmup; 5.16 was a mixed L2+DRAM number. |
| "HBM3E peak 8.17 TB/s" / "% of 8 TB/s spec" | catalog scattered | Pre-ECC bus number; correct post-ECC spec is 7672 GB/s. |
| "cudaMemset 7.47 TB/s = 21% faster than user kernel" | commits ffec751, b145ad7 | Faster only in wall-clock; ncu DRAM rate is identical within 3% for the optimal user kernel. The faster cudaMemset wall-clock = earlier kernel-exit, not higher HBM throughput. |

---

## D2D bandwidth (MED confidence)

| Operation | Single-direction | Effective R+W |
|---|---:|---:|
| `cudaMemcpyAsync` D2D (2 GB) | 3279 GB/s | 6557 GB/s |
| Kernel int4 8-ILP D2D copy | 2781 GB/s | 5562 GB/s |
| `cudaMemcpyAsync` D2D (256 MB) | 3005 GB/s | 6010 GB/s |

cudaMemcpyAsync wins by ~18% over kernel copy. The "single direction" half-rate (~3.3 TB/s) reflects that each byte traverses HBM twice (read source, write dest); the R+W aggregate of ~6.5 TB/s is 8% short of the unidirectional 7.3 TB/s peak — consistent with the partial-duplex finding above (concurrent R+W is the same physical situation as a copy kernel).

---

## Saturation requirements (HIGH confidence)

To hit the 7.30 TB/s ceiling you need **all of**:
1. **Working set ≥ 4 GB** (well past L2 capacity so wall-clock isn't inflated by L2 absorption).
2. **256-bit per-instruction width** (`v8.b32` for store/load) — v4 caps ~6.2 TB/s in the same pattern.
3. **Per-warp coalesced 1-KB bursts** (32 lanes × 32 B contiguous per iter) — per-thread-consecutive layout halves BW.
4. **High parallelism, low per-thread work** — 16384 blocks × 256 thr × 32 iters = 4 M warps' worth of 1-KB bursts. Persistent / grid-stride patterns leave tail effects.
5. **Many in-flight requests per warp** — already achieved by the unrolled 32-iter loop with no dependence between iterations.

CTA-count, occupancy, and cache hint do NOT matter once the above are satisfied (the controller is the bottleneck, not LSU issue rate or L2 policy).

---

## R:W ratio sweep — no ratio escapes 7.4 TB/s (HIGH confidence)

A6 rigor: per-thread mix of N_R 32-B reads and N_W 32-B writes (32 ops total per thread), single kernel, ncu `dram__bytes_read.sum.per_second` + `dram__bytes_write.sum.per_second`, two methods agree within 1%.

| R:W (ops/thr) | DRAM read (TB/s) | DRAM write (TB/s) | Aggregate (TB/s) | % of 7672 spec |
|---|---:|---:|---:|---:|
| 32:0  | 7.31 | ~0   | **7.31** | 95.3% |
| 28:4  | 6.22 | 0.86 | 7.08 | 92.3% |
| 24:8  | 5.40 | 1.72 | 7.12 | 92.8% |
| 20:12 | 4.32 | 2.50 | 6.82 | 88.9% |
| 16:16 | 3.39 | 3.29 | **6.68** ← min | **87.0%** |
| 12:20 | 2.52 | 4.09 | 6.61 | 86.2% |
| 8:24  | 1.65 | 4.85 | 6.50 | 84.7% |
| 4:28  | 0.83 | 5.72 | 6.55 | 85.4% |
| 0:32  | ~0   | 7.28 | **7.28** | 94.9% |

**U-shape, minimum at 50:50 (6.68 TB/s).** No ratio escapes 7.31 TB/s — the pure-direction peaks. Mixed-traffic workloads pay a 4–13% efficiency penalty vs single-direction. Asymmetry: 8:24 (heavy-W) lower than 24:8 (heavy-R), suggesting writes incur slightly higher per-byte controller cost than reads when mixed.

This **confirms HBM3E is a shared bus, not full-duplex** (the open question from commit 48d2a60). Mechanism: HBM3E read↔write turnaround penalty (tWTR/tRTW datasheet timing) — every direction switch costs cycles, and 50:50 maximizes switches.

Commit: rigor_a6_rw_ratio_sweep.cu — Method 1 wall-clock + Method 2 ncu, agree within 1%. Method 3 SASS verification: each kernel emits N_R `LDG.E.128` + N_W `STG.E.128` instructions as expected.

---

## Open questions / NEEDS NEW MEASUREMENT

1. **Why exactly the 5% spec gap?** Refresh + command bus + row-precharge are the candidates; no direct measurement to attribute the budget. (A2 partially addressed: bursts <1 KB hit 98.6%, longer bursts under-saturate parallelism — commit 66a2853.)
3. **What recipe (if any) gets cudaMemset to >7.30 TB/s in ncu?** The wall-clock 7.5 TB/s suggests there might be a marginally faster channel utilization pattern that nobody has reproduced from user PTX. SASS extraction of cudaMemset's driver kernel would settle it.
4. **HBM3E refresh-rate sensitivity.** No measurement of whether long-sustained streaming hits a refresh-induced floor lower than burst peak.
5. **Multi-GPU contention** — 2× B300 in the same chassis, both saturating HBM. Does the NVLink fabric introduce additional HBM-side contention? Untested.
6. **Stride / channel-locality sweep** — does going outside contiguous 1 KB bursts (e.g. 4 KB per warp, 256 B per warp) preserve 7.30 TB/s, or is the ~95% number tied tightly to per-warp 1 KB?

---

## Files of record

- `/root/github/QuickRunCUDA/investigations/rigor_v8_dram_peak.cu` — write recipe (commit a04d9c8)
- `/root/github/QuickRunCUDA/investigations/rigor_v8_read_peak.cu` — read recipe (commit 5193694)
- `/root/github/QuickRunCUDA/investigations/rigor_v8_rw_concurrent.cu` — two-stream R+W test (commit 48d2a60, wall-clock — superseded)
- `/root/github/QuickRunCUDA/investigations/rigor_v8_rw_ncu.cu` — single-kernel R+W with ncu (commit 30159f6, authoritative)
- `/root/github/QuickRunCUDA/investigations/rigor_dram_truth.cu` — ncu vs wall-clock comparison (commit 0675ffb)
- `/root/github/QuickRunCUDA/investigations/rigor_write_alloc.cu` — write-allocate refutation (commit e1d77c6)
- `/root/github/QuickRunCUDA/investigations/beat_memset.cu` — cudaMemset SM-contention proof (commit 8bdb3e6)
- `/root/github/QuickRunCUDA/investigations/rigor_tma_write.cu` — TMA bulk write 82% (commit b3dd31c)
- `/root/github/QuickRunCUDA/investigations/hbm_write_proper.cu`, `hbm_write_peak.cu`, `hbm_read_peak.cu`, `hbm_ilp.cu`, `l2_peak.cu`, `d2d_peak.cu`, `memset_audit.cu`, `memset_bw.cu`, `memset_sweep.cu`, `memset_variants.cu` — earlier (sub-optimal) measurements
- `/root/github/QuickRunCUDA/investigations/08_hbm_write.md` — investigation report from the 6.2 TB/s era; **superseded by a04d9c8/5193694**
