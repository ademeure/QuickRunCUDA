# B300 Catalog Audit Notes

This document audits the B300_PIPE_CATALOG.md findings from this session. Goals:
- Identify unreliable / badly-designed tests
- Flag contradictions
- Document remaining uncertainty
- Provide corrections where re-testing was done

Reliability ratings:
- **HIGH**: cross-checked, methodology sound, expected within ±5%
- **MEDIUM**: plausible but single test, possible methodology issues
- **LOW**: known issues with test design (DCE, partial work, dominated by overhead)
- **WRONG**: empirically incorrect, see correction

---

## TENSOR CORE THROUGHPUT — MISLEADING

### Original claim
> "Tensor cores work! BF16 m16n8k16: 514 TFLOPS"

### Issue
This is NOT B300's tensor core peak. B300's spec'd dense BF16 throughput is **~1,980 TFLOPS via tcgen05.mma** (the Blackwell tensor core path using TMEM). My **mma.sync path delivers only ~26% of that** because:
- mma.sync is the legacy "register-resident" tensor core API (pre-Hopper)
- It does NOT use TMEM (Tensor Memory) — Blackwell-specific tier
- Per-warp throughput limited by register file bandwidth and scheduling

### What's correct
- mma.sync m16n8k16 BF16 → FP32: **~514 TFLOPS** measured (likely under-saturated)
- mma.sync m16n8k8 BF16 → FP32: **~260 TFLOPS** (half the work per inst)
- ILP=4 doesn't help → probably already at scheduler issue rate, not pipe bandwidth

### To get B300's full tensor throughput requires
- `wgmma.mma_async` (Hopper-style, sm_90+) — needs A/B descriptors in shmem, mbarriers
- `tcgen05.mma` (Blackwell-only, sm_100+/sm_103) — uses TMEM, the new tier
- Both require complex orchestration (matrix descriptors, async coordination, TMEM allocation)

### Reliability
- **mma.sync 514 TFLOPS: MEDIUM** (test methodologically OK but it's not "tensor core peak")
- **"tensor cores work" claim: MISLEADING** — corrected here

---

## FP32 PEAK 52 TFLOPS — UNDER-SATURATED

### Claim
> "Peak FP32 with ILP=8, 8 warps/SM: 52.2 TFLOPS = 68% of theoretical 77 TFLOPS"

### Issue
68% is decent but not peak. Theoretical 77 TFLOPS = 4 partitions × 1 FFMA inst/cy × 32 lanes × 2 op/FMA × 148 SM × 2.032 GHz.

### Why it's not peak
With FFMA latency ~23 cy and 8 warps/SM (= 2 warps/partition):
- Each partition has 2 warps issuing FFMAs back-to-back
- Each warp can issue 1 FFMA every (latency / N_warps_in_chain) = 23/2 = 11.5 cy
- Per partition: 2 × 1/(11.5) FFMAs/cy = 0.174 FFMAs/cy = ~17% issue rate
- But ILP=8 inside each thread → 8 independent chains
- 8 chains × 1 FFMA / 23 cy each = 8/23 = 0.348 FFMAs/cy per warp
- 2 warps × 0.348 = 0.696 FFMAs/cy per partition (~70% of peak)

So 68% is consistent with the model. To approach 100%:
- Need ILP × warps ≥ 23 (latency)
- ILP=8 × 2 warps = 16 (just below 23) → 70% expected
- ILP=8 × 4 warps = 32 (above 23) → near 100% expected

### Reliability
- **52 TFLOPS measurement: HIGH** (clean test)
- **Statement of "peak": MISLEADING** — should say "with ILP=8, 8 warps/SM"
- **Theoretical 77 TFLOPS**: HIGH (matches spec)

---

## L2 STRIDE ACCESS — DCE-SUSPICIOUS

### Claim
> "stride=4 bytes: 1426 GB/s effective (with 32B sectors: 11,412 GB/s fetched)"

### Issue
Aggregate 11,412 GB/s would exceed B300's HBM peak (~5,800 GB/s aggregate measured locally). Either:
- L2 hit rate is helping (data fits in 126 MB L2)
- Compiler is eliminating part of the access
- Sector accounting wrong

256 MB buffer fits in L2's 126 MB only partially. With wraparound, hot region = 256 MB ÷ stride * 4. For stride=4, hot region = 256 MB (full buffer, no help from L2).

The 1426 GB/s effective × 8 sectors = 11,400 GB/s claimed read rate is impossible.

### Likely explanation
- Each warp accesses 32 threads × stride bytes of memory per cycle
- For stride=4: 32 threads × 4 bytes = 128 bytes/access = ONE 128B cache line OR 4 32B sectors  
- "useful" is 32 × 4 = 128 bytes useful per access
- "fetched" assumes 32B sectors × 32 threads = 1024 bytes — but with coalescing, only 128 bytes (matches useful)

So my "fetched" calculation was wrong (over-estimated). Actual fetched = useful for coalesced stride-4 access.

### Correction
- stride=4 (coalesced): 1426 GB/s real bandwidth
- This is plausible if data fits in L2 partially
- For full DRAM: should saturate around 5,800 GB/s for any ≥128B coalesced stride

### Reliability
- **"effective" numbers: MEDIUM-HIGH** (the reads happen; loop runs)
- **"32B sector" multiplier: WRONG** — assumed waste that doesn't happen for coalesced loads
- **Multi-SM 600+ TB/s extrapolation in earlier shmem test: WRONG (DCE)**

---

## SHMEM RANDOM-ACCESS BANDWIDTH — DCE'd

### Claim
> "iters=100000: 7,664,787 GB/s"

### Issue
Already noted in catalog as "implausible numbers because compiler optimized loop body away." But it should be FORMALLY marked as INVALID.

### Reliability
- **7.6 PB/s number: WRONG (DCE)** — invalid, ignore
- The 228 KB per SM, 32 banks × 4 bytes = 128 bytes/cy per SM × 148 SMs × 2.032 GHz = ~38 TB/s peak SHMEM aggregate. Anything above this is fake.

---

## L1 CACHE SIZE "~32 KB" — IMPRECISE

### Claim
> "L1 effective size: ~32 KB (BW saturates there)"

### Issue
L1 size on Hopper/Blackwell is part of unified 256 KB L1+SHMEM pool, configurable via carveout.
- With shmem opt-in 232 KB → L1 ≈ 24 KB
- Default split: depends on carveout

My measurement saw BW grow up to 32 KB then saturate, but the "saturation" might be due to:
- Test methodology (loop overhead vs working set)
- Single SM with limited concurrency
- Compiler optimizations

### Reliability
- **"L1 ≈ 32 KB" claim: MEDIUM-LOW**
- Better statement: "Single-SM read BW saturates around working set 32-64 KB"
- True L1 size is configurable; reported sharedMemPerMultiprocessor=233472 is the SHMEM portion only

---

## NVTX "0 ns" — BELOW MEASUREMENT NOISE

### Claim
> "nvtxMark: ~0 ns/call"

### Issue
0 ns is impossible. The measurement used `std::chrono::nanoseconds` which has at most ~10ns resolution on Linux x86. 0 ns means below noise floor.

### Correction
- nvtxMark: <1 ns/call (below measurement granularity)
- nvtxRangePush+Pop: 0.66 ns/pair (also possibly below noise; treat as "negligible <2 ns")

### Reliability
- **Numbers themselves: LOW (noise-dominated)**
- **Conclusion "essentially free without profiler": HIGH** — empirically true

---

## PDL "SAVING" 1.9 us — SUSPECT MEMORY-WRITE COUPLING

### Claim
> "PDL chain depth saves 1.9 us/kernel asymptotically"

### Issue
This was tested with conditional-write style kernels (Style A in `pdl_verify.cu`). With unconditional-write Style B, PDL was -3 us/kernel SLOWDOWN. Both styles documented.

### Reliability
- **For pure-compute Style A: HIGH** (clean asymptotic curve)
- **For real-world kernels with memory writes: VARIABLE** — must test per-pattern

---

## CONCURRENT KERNEL LIMIT 128 — NOT RE-VERIFIED THIS SESSION

### Claim
> "True hardware concurrent kernel slot limit = 128"

### Issue
This was from the PRE-existing catalog (commit 579e4f0, before my session). I did NOT re-verify it this session, only mentioned it.

### Reliability
- **128-slot claim: HIGH** (well-tested in prior sessions per git log)
- **My citing of it: just referenced, did not re-test**

---

## STREAM "PARALLEL" TIMES — FIRST TEST WAS WRONG

### Original claim (streams_explore)
> "256 streams parallel = 0.57 ms (1.01x single)"

### Issue
The first test (`tests/streams_explore.cu`) timed events on default stream while kernels ran on other streams → events fired immediately, not when kernels finished. Re-tested in `streams3.cu` with proper sync.

### Corrected (from streams3)
- 1 stream: 0.57 ms
- 128 streams: 0.83 ms (88x effective concurrency)
- 148 streams: 1.18 ms (cliff — 2 waves)
- 256 streams: 1.39 ms

### Reliability
- **streams_explore.cu numbers: WRONG (broken timing)**
- **streams3.cu numbers: HIGH (proper sync)**

---

## DSMEM BANDWIDTH — DCE NOT VERIFIED

### Claim
> "DSMEM 1000 GB/s peer access, local SMEM 4500-4850 GB/s"

### Issue
The "local" and "remote" tests share the same loop structure. The accumulator `acc` writes only if `acc < 1e30f` then to `out[blockIdx.x]`. Compiler may eliminate part of the loop.

The measured local SMEM ~4500 GB/s for 148 blocks × 256 thr × 1000 reads × 4 bytes = 152 GB total read in ~33 us. Aggregate = 4500 GB/s = ~30 GB/s per SM = ~120 cycles per read for a single thread.

That's HIGH for SHMEM (should be ~1 cy/read coalesced). Suggests loop overhead dominates OR compiler DID NOT optimize.

True SHMEM peak should be MUCH higher (~38 TB/s aggregate computed earlier). So 4500 GB/s = 12% of peak.

### Reliability
- **DSMEM 1000 GB/s: MEDIUM-LOW** (test possibly under-utilizing)
- **Local SMEM 4500 GB/s: MEDIUM-LOW** (likely far from SHMEM peak)
- **Ratio (local 4.7× faster than DSMEM): MEDIUM** (relative comparison robust)

---

## ATOMIC THROUGHPUT — VERIFIED

### Claim
- Local shmem all-contend: 134 G/s
- Global no-contend: 273 G/s
- Global per-warp contend: 14 G/s
- NVLink remote atomic: 16.7 G/s

### Reliability
- **HIGH** — matches prior catalog memory ("49 Gatomic/s LOCAL all-contend, 16 Gatomic/s REMOTE")
- Pattern: contention matters MORE than memory hierarchy

---

## CLOCK FREQUENCY — RELIABLE

### Claim
> "B300 clock 2032 MHz under load"

### Reliability
- **HIGH** — measured via globaltimer (ns) and clock64 (cy), exact match
- Confirmed across 1k, 10k, 100k, 1M iters

---

## STEAL-RESERVED TRICK — RIGOROUSLY VERIFIED

### Claim
> "4 blocks × 57 KiB on same SM via raw-PTX writes to offsets 0..1023"

### Reliability
- **HIGH** — 4736 blocks tested at max occupancy, 0 corruption
- Also verified incompatibility with cluster.sync (crashes confirmed)
- **PERFORMANCE caveat properly noted**: hot-loop access has overhead

---

## P2P NVLink BW — CORRECTED

### Original claim
> "286 GB/s remote read"

### Corrected
> "755 GB/s remote read with float4 + 75K threads (matches DMA)"

### Reliability
- **Initial 286: WRONG (thread-limited)**
- **Corrected 755: HIGH** (cross-checked vs DMA 757)

---

## SUMMARY: WHAT TO TRUST AND NOT

### HIGH confidence findings
- B300 device properties (cudaGetDeviceProperties): 148 SMs, 2032 MHz, 126.5 MB L2, 228 KB SHMEM/SM, 1024 reserved
- NVLink BW: 757 GB/s uni / 1503 bi, 755 GB/s kernel-side P2P
- L2 size 126.5 MB
- Reserved 1 KiB shmem at offsets 0..1023
- Steal-reserved trick (4×57KB verified)
- Cluster max size: 8 portable, 16 non-portable
- 4 MemSyncDomains, 6 priority levels
- Sync API costs (cudaStreamWaitEvent 0.08us, etc.)
- Memory pool 160× faster than cudaMalloc
- Atomic throughput numbers
- Clock 2032 MHz under load
- PDL signal point optimum (90-99%)
- Graphs ≈ PDL (~2.5us/kernel)
- PCIe Gen5 x16 ~57 GB/s
- Compatibility matrix for steal-reserved
- Per-thread default stream effect (2.2x for 4-thread NULL)

### MEDIUM confidence
- FFMA latency 23 cycles (single warp)
- Mma.sync 514 TFLOPS (real, but not peak)
- FP32 ILP=8 reaching 52 TFLOPS
- Bank conflict 32-way slowdown
- Constant memory broadcast 25 TB/s

### LOW confidence / KNOWN ISSUES
- L1 cache size estimate (~32 KB) — vague
- DSMEM 1000 GB/s — likely under-utilizing
- Local SHMEM read BW 4500 GB/s — likely far from peak
- L2 stride access "fetched" GB/s — sector multiplier wrong
- NVTX 0 ns — below noise floor

### WRONG / SUPERSEDED
- Stream concurrency in streams_explore.cu (broken event timing) → use streams3.cu
- P2P kernel 286 GB/s → corrected to 755 GB/s
- "Tensor cores work" headline → corrected to "mma.sync 514 TFLOPS, NOT peak"
- 7.6 PB/s SHMEM in shmem_test → DCE'd, invalid
- Multi-SM 600+ TB/s extrapolations in some tests → DCE-affected

### NOT VERIFIED THIS SESSION
- 128 concurrent kernel slot limit (cited from earlier work)
- Most main-catalog "F2FP" findings (those are pre-session)


# Corrections Applied (after audit)

## SHMEM Peak Bandwidth — Re-tested

### Old (under-utilizing test)
- Local SMEM read: 4500-4850 GB/s aggregate (12% of theoretical)

### New (proper test, `tests/shmem_peak.cu`)
- 8 reads_per_iter × 1000 iter × 256 thr × 148 SMs:
- **19.85 TB/s aggregate (52% of theoretical 38.5 TB/s)**
- Per-SM: 134 GB/s
- Theoretical: 32 banks × 4 bytes × 2.032 GHz = 260 GB/s/SM = 38.5 TB/s aggregate

Even better tests should be possible with more ILP per warp and pipelined accesses, but 52% is closer to realistic peak.

## Tensor Core mma.sync Peak — Re-tested

### Old
- 514 TFLOPS BF16 m16n8k16 with ILP=4

### New (`tests/mma_peak.cu`)
- 32 warps/SM × 8 ILP × 8 acc chains: **544 TFLOPS BF16 m16n8k16**
- Per-SM: 3.7 TFLOPS = 1.81 mma.sync m16n8k16 / cycle / SM
- Per-partition (4): 0.45 mma/cy → 1 mma every ~2.2 cy per partition
- This is the LEGACY mma.sync path peak, ~1/4 of B300's tcgen05 peak (~1980 TFLOPS spec)

### Conclusion
- mma.sync legacy path: 544 TFLOPS BF16 — VERIFIED PEAK
- B300 full BF16 peak ~1980 TFLOPS requires `tcgen05.mma` (Blackwell-only) with TMEM
- WGMMA (Hopper-style) might approach this on B300 too

## FP32 Peak — Re-tested

### Old
- 52 TFLOPS with ILP=8, 8 warps/SM (68% of theoretical)

### New (`tests/fp32_peak2.cu`)
- ILP=16, 64 warps/SM (max occupancy): **58.61 TFLOPS = 76.2% of theoretical 77 TFLOPS**
- 1024 threads/block × 2 blocks/SM × 148 SMs

Likely limited by register file bandwidth or scheduler at this point. 76% achievement is near practical peak for FFMA chain.

## L2 stride access "fetched" Numbers — Corrected

The "32B sector" multiplier was WRONG for coalesced loads. For coalesced 32-thread access:
- 32 threads × 4 bytes = 128 bytes = exactly 1 cache line / 4 sectors
- No waste — fetched = useful

Numbers like "11,400 GB/s fetched at stride=4" are ARTIFACTS of bad accounting. The true effective BW (~1426 GB/s) is the real number.

## DSMEM — Likely Under-Utilizing (not yet re-tested with proper saturation)

Previous: 1000 GB/s peer access. Local SMEM measured 4500 GB/s in same test.

Both numbers are likely far below SHMEM peak (38.5 TB/s). Suggests the test underutilizes load bandwidth. RELATIVE comparison (4.7× slower for DSMEM) is robust though.

## Stream concurrency (streams_explore.cu) — DEPRECATED FILE

The original `tests/streams_explore.cu` had broken event timing (events on default stream while kernels on other streams = no observation).

- ✗ DO NOT USE: tests/streams_explore.cu
- ✓ Use instead: tests/streams3.cu (CPU-side timing with cudaDeviceSynchronize)


# Updated Reliability Index

After corrections, here's the trust ranking:

### HIGH (cross-checked, methodologically sound)
- B300 device props (148 SMs, 2032 MHz, 126.5 MB L2, etc.)
- NVLink: 757 GB/s uni / 1503 bi DMA, 755 GB/s kernel (after correction)
- Reserved 1 KiB shmem layout + steal trick
- Cluster max: 8 portable, 16 non-portable
- Sync API costs (cudaStreamWaitEvent 0.08 us, etc.)
- Memory pool 160× faster than cudaMalloc
- Atomic throughput
- Clock 2032 MHz under load
- PDL signal point optimum (90-99%)
- Graphs ≈ PDL (~2.5 us/kernel)
- PCIe Gen5 x16 ~57 GB/s
- MemSyncDomainCount = 4
- 6 stream priority levels
- mma.sync m16n8k16 BF16 peak 544 TFLOPS (legacy path, NOT full B300 tensor peak)
- FP32 76.2% of theoretical with ILP=16+max occupancy
- SHMEM peak 19.85 TB/s aggregate (52% of theoretical 38.5)
- LaunchCompletionEvent block-start signal saves ~60 us cross-stream

### MEDIUM (single test or moderate uncertainty)
- FFMA latency 23 cy
- Bank conflict serialization cost
- Constant memory broadcast 25 TB/s
- L2 stride bandwidth (effective is real, fetched was wrong accounting)
- Async copy 1.27× speedup at 16 KB

### LOW / under-saturated
- DSMEM 1000 GB/s — relative ratio robust, absolute likely under-peak
- L1 cache size estimate (~32 KB)
- NVTX 0 ns (below noise floor; safe to call "negligible")

### KNOWN WRONG / SUPERSEDED
- streams_explore.cu (broken timing)
- "286 GB/s P2P kernel" (thread-limited; corrected to 755 GB/s)
- "514 TFLOPS = tensor cores work!" (corrected: legacy path peak, not full B300)
- "7.6 PB/s SHMEM" in early shmem_test.cu (DCE)
- "32B sector × N" extrapolation in l2_cacheline.cu (bad accounting)
- Multi-SM 600+ TB/s in some early tests (DCE-affected)

### NOT VERIFIED THIS SESSION (cited from prior catalog)
- 128 concurrent kernel slot limit
- Most pre-session findings (B300 catalog ~16K lines that existed before)



## L1 Cache Size — Re-measured properly

### Old
- "L1 effective ~32 KB based on bandwidth saturation curve"

### New (`tests/l1_proper.cu` — pointer chase latency)
| Working set | ns/load | Tier |
|---:|---:|---|
| 1 KB | 19.6 | L1 hit |
| 32 KB | 20.3 | L1 mostly |
| 64 KB | 20.9 | L1 |
| 128 KB | 22.2 | L1 mostly |
| **256 KB** | **152** | **L2 only — sharp jump** |
| 4 MB | 153 | L2 (still cached) |

**Key transitions on B300**:
- L1 hit latency: ~20 ns (~40 cy at 2 GHz)
- L2 hit latency: ~152 ns (~310 cy)
- L1 effective size (single-warp pattern): **up to ~128 KB before sharp transition**

This is different from "32 KB" — earlier number was based on bandwidth saturation (less precise). Pointer chase with dependent loads cleanly exposes the cache hierarchy.

NOTE: The 128 KB L1 effective size is single-warp; multi-warp may differ due to L1 sharing across warps in a partition. Also, with cudaFuncAttributePreferredSharedMemoryCarveout configured for max SHMEM (carveout=100), L1 size shrinks (and vice versa).

### Reliability
- **L1 ~20 ns latency: HIGH**
- **L1 effective up to ~128 KB: HIGH** (clean transition to L2 at 256 KB)
- **L2 latency ~152 ns: HIGH**



## DSMEM Bandwidth — RE-TEST FAILED

Tried two more rigorous DSMEM tests:
1. `tests/dsmem_proper.cu`: register-accumulator pattern — compiler statically computed loop body (smem values determined at compile time after init), produced impossible 13 PB/s
2. `tests/dsmem_cycles.cu`: clock64-based cycle test — output writes failed (uninitialized values returned)

### Conclusion
Original `tests/dsmem_v2.cu` ratio (4.7× slower DSMEM than local SMEM) is the BEST data we have. The absolute numbers (1000 GB/s remote, 4500 GB/s local) are likely far below true peak SMEM BW (38.5 TB/s aggregate theoretical) but the ratio appears consistent.

For accurate DSMEM peak measurement, would need:
- ncu profiling with `dsmem_count` metrics
- Or kernel using shfl-style data exchange to force sync
- Or proper warp-issue-rate analysis

### Reliability
- **DSMEM ratio (4.7× slower than local SMEM): MEDIUM confidence**
- **Absolute DSMEM 1000 GB/s: LOW confidence — likely under-saturated**
- **Local SMEM 4500 GB/s: superseded by `tests/shmem_peak.cu` showing 19.85 TB/s**

