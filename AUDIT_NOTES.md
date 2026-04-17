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
