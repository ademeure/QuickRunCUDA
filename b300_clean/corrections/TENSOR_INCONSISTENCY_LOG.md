# Tensor-Core Inconsistency Log — B300 SXM6

**Auditor**: tensor swarm, 2026-04-22
Cross-references: B300_TRUE_REFERENCE, V8/V9 measurements, TCGEN05_*, NVFP4_*, MMA_*, 06_tensor_cores.md, CLAUDE.md.

This file lists every cross-file inconsistency, ambiguity, or stale claim
I found in the tensor-core documentation. For corrected per-precision peak
numbers, see `06_tensor_cores_CORRECTED.md` in this directory.

---

## A. Numerical inconsistencies between files

### A1. BF16 mma.sync legacy — 569 vs 577 vs 578 TFLOPS
- `B300_TRUE_REFERENCE.md` row 47: **569 TFLOPS** (commit a37d989, locked 1920 MHz)
- `06_tensor_cores.md` row 22: **577 TFLOPS @ 2032 MHz boost** (with explicit note that catalog "1920→611 extrap" was wrong)
- `06_tensor_cores.md` Anatomy section: "**577.45 TFLOPS @ 1920 MHz** = 98.9% of ncu-derived ceiling (582). Extrapolates to ~611 TFLOPS @ 2032 MHz" — **CONTRADICTS** row 22 of the same file (which says 577 was at 2032).
- `V8_HMMA_F16_PEAK.md`: **578.6 TFLOPS** (boost 2032)
- `V8_HMMA_VARIANTS_PEAK.md`: same 578.6 TFLOPS for F16 / BF16 / both accumulator types

**Resolution**: 569 (1920 MHz) and 578 (2032 MHz boost) differ by 1.6%, consistent with clock ratio. The 06_tensor_cores Anatomy section has an internal contradiction about whether 577 was at 1920 or 2032; pick one and fix.

### A2. FP8 cuBLAS — 4425 vs 4474 vs 4486 vs 4491 vs 4651 TFLOPS
- TRUE_REFERENCE row 56: **4425 TFLOPS** (sustained via cudaGraph, 30 sec @ 943 W)
- 06_tensor_cores Conflict-Resolution row: 4486 sustained, 4491 30s sustained, 4474 older, 4651 microbench
- TRUE_REFERENCE row 68 (data-dep table): **4393 zero**, 3984 random, 3951 normal-ish
- README cite: **4486 TFLOPS**

**Resolution**: All within ~5% noise; the spread is real measurement variance. Recommended canonical: **4400-4500 TFLOPS zero-data, 4000 TFLOPS random** (cite both).

### A3. BF16 cuBLAS — 2237 / 2242 / 2246 / 2252 / 2259 / 2325 TFLOPS
- TRUE_REFERENCE row 48: **2242** (90% of 2500)
- TRUE_REFERENCE row 66 (data-dep): 2246 zero / 1883 random / 1850 normal
- 06_tensor_cores row 19: **2259** (cuBLAS 13.4)
- 06_tensor_cores row 20: 2325 (microbench tcgen05 direct)
- TCGEN05_PATH_NOTES section 4: 2237 (CUTLASS via algoId 66)
- TCGEN05_POWER_MASTER row "TRUE HARDWARE PEAKS": 2252 TF (100.5% spec) with constant data
- CLAUDE.md: ~1980 TFLOPS

**Resolution**: 2242–2259 span the cuBLAS spread (~1%). **2325 is the direct-tcgen05 microbench peak (no cuBLAS overhead)**. **CLAUDE.md's 1980** is an older NVIDIA spec quote and inconsistent with measured ~2250; CLAUDE.md should be updated.

### A4. NVFP4 spec — 10000 vs 15000 TFLOPS
- CLAUDE.md says 10000 TFLOPS for B300; mentions 15000 only via K=96 path which is "inaccessible via cuBLAS 13.2".
- D4_PRECISION_POWER_PERF_TABLE.md row "FP4 cuBLAS": 10800 TFLOPS, **15000 spec, 72%** — uses 15 PF as the spec number, contradicts CLAUDE.md's 10 PF.
- TRUE_REFERENCE row 51: 10297 TFLOPS = **103% of 10 PF spec** — uses 10 PF.
- TRUE_REFERENCE row 53: NVFP4 K=96 sm_103 path (15 PFLOPS B300 1.5×) inaccessible.
- NVFP4_CUBLAS_FULL_SWEEP.md: cites 11054 TFLOPS at boost = **73.7% of 15 PF**, columns labeled "%15PF".
- NVFP4_CUTEDSL_THROTTLE_DEEP_DIVE: explicit 15000 spec used throughout.

**Resolution per memory + TRUE_REFERENCE**: The 15 PF figure is the K=96-ULTRA path (1.5× of base). It is real (tcgen05 microbench reaches 10.91 PF at 1500 MHz lock = 73% of 15 PF), but **not reachable via cuBLAS 13.2** and only marginally reachable via cuBLAS 13.4 (~10.8 PF). Files using "% of 15 PF" as the headline are technically valid but misleading — recommend always pairing with "(B300 K=96 ULTRA, requires cuBLAS 13.4 large-N rect; cuBLAS 13.2 uses base 10 PF path)".

### A5. NVFP4 single-shot vs sustained — 11054 vs 10297 vs 9109 vs 8424 vs 6554 TFLOPS
- NVFP4_CUBLAS_FULL_SWEEP boost: **11054 TFLOPS** (M=N=8192 K=38400, 73.7% of 15 PF)
- TRUE_REFERENCE row 51: **10297** wide-N M=8192 N=65536 K=16384 (103% of 10 PF)
- TRUE_REFERENCE row 55: **9109** const-byte single-shot N=8192 (91%)
- TRUE_REFERENCE row 52: **8424** square N=K=24576 random (84%)
- TRUE_REFERENCE row 54: **6554** sustained random N=16384, 15 sec, throttled to 1057 MHz (65%)

**Resolution**: All are real — NVFP4 throughput depends massively on shape, data pattern, AND duration (power-throttle). Recommend a "NVFP4 throughput surface" table with all four axes (shape, data, single/sustained, clock state) explicit.

### A6. FP8 random-data 3983 vs FP8 microbench 4651 TFLOPS
- TRUE_REFERENCE row 57: 3983 random
- 06_tensor_cores row 17: 4651 microbench direct tcgen05 with kind::f8f6f4

**Resolution**: Microbench is single-shot pre-loaded SMEM; cuBLAS adds DRAM/L2 traffic + cluster coordination. 4651 = peak compute ceiling; 3983 = sustained real workload. Both correct; cite microbench as "ceiling" and cuBLAS as "achievable".

### A7. NVFP4 K=64 microbench — 4870 vs 7270 TFLOPS
- TCGEN05_N_MATRIX & TCGEN05_N_SWEEP: NVFP4 K=64 cta=2 = **4.87 PF** at 1005 MHz
- TCGEN05_PERFW_CLEAN: NVFP4 K=64 = **7.27 PF at 1500 MHz**
- TCGEN05_PERF_WATTS: NVFP4 K=64 = **7.27 PF at 1500 MHz**

**Resolution**: 4.87 PF / 1005 MHz × 1500 MHz = 7.27 PF. Consistent. Always cite clock state.

---

## B. Counter-metric misuse (per memory rule "ncu pipe_tensor doesn't measure tcgen05")

### B1. 06_tensor_cores.md line 11
"All numbers measured on sm_103a, single chip (148 SMs), MMA-only (no DRAM). Same `pipe_tensor` 128 cy/MMA at M=128, N=256."

**Issue**: The 128 cy/MMA was actually measured via `clock64` (TCGEN05_N_SWEEP, TCGEN05_N_MATRIX), not via `pipe_tensor`. Calling it "pipe_tensor 128 cy/MMA" risks readers thinking ncu pipe_tensor is the source — which would be wrong for tcgen05 (the table is mostly tcgen05 measurements).

**Fix**: rephrase to "Same 128 cy/MMA (clock64-measured at M=128, N=256)".

### B2. SUBTILE_DEDUP_MODEL.md line 109
"Cross-check with NCU `tpc__l1tex_*` or `sm__pipe_tensor_*` counters"

**Issue**: The whole SUBTILE_DEDUP discussion is about tcgen05.mma. ncu pipe_tensor will not see UTCOMMA/UTCHMMA reliably (per SESSION_2_DELTA's empirical findings). Suggesting pipe_tensor as a cross-check is misleading.

**Fix**: replace with `smsp__inst_executed_pipe_tensor_subpipe_hmma_op_utchmma_utcqmma_utcomma_scope_2cta.sum` which DOES include tcgen05 ops, OR with clock64 measurement.

### B3. SESSION_2_DELTA.md (multiple sections) — already documents the gotcha
- Lines 524-790 explicitly demonstrate pipe_tensor showing 99% active for only 2.42 ms of 9.37 ms wall-clock on a tcgen05 kernel.
- Concludes "(pipe_tensor active%) may have scope limitations" — this is the strongest evidence in the codebase for the user-memory rule.

**Status**: NOT a bug, this is the documentation of the gotcha. Recommend cross-linking from SUBTILE_DEDUP_MODEL.md and 06_tensor_cores.md.

### B4. CUBLAS_BIT_ENTROPY_CORRECTION.md lines 1086-1087
Uses both:
- `sm__pipe_tensor_subpipe_hmma_cycles_active.sum`
- `smsp__inst_executed_pipe_tensor_subpipe_hmma_op_utchmma_utcqmma_utcomma_scope_2cta.sum`

The latter explicitly contains `utchmma_utcqmma_utcomma` so DOES include tcgen05. The former (`subpipe_hmma_cycles_active`) is ambiguous — it might be hmma-only. Verify whether the author treated 524288 as full MMA count or just HMMA.

### B5. V8_HMMA_F16_PEAK.md, V8_HMMA_VARIANTS_PEAK.md, M14_V8_SOL_LADDER.md
All use `sm__pipe_tensor_*` to measure mma.sync legacy HMMA. **OK** — pipe_tensor is the correct counter for mma.sync. Not affected by the tcgen05 gotcha.

---

## C. The "1543 TFLOPS BF16 single-chain" retraction

**Authoritative retraction**: `B300_TRUE_REFERENCE.md` row 58:
"**BF16 mma.sync 8-chain** (multi-accumulator) | **~570** | matches catalog 'burst 569' | (83ef1c6) — single-chain '1543' was over-counted; RETRACTED"

The number "1543" still appears in the codebase for **NVLink P2P bidirectional GB/s** (12_nvlink_p2p.md, README.md) — those are LEGITIMATE and unrelated to the BF16 retraction. The numerical coincidence is unfortunate but not a bug.

**Status**: No remaining BF16 "1543" claims in the catalog (06_tensor_cores already uses 569-577). The retraction is documented but no live tensor file claims 1543 TFLOPS.

---

## D. CLAUDE.md "BF16 ~1980 TFLOPS" vs measured 2242-2259

CLAUDE.md says: "BF16 tensor via tcgen05.mma / cuBLAS: ~1980 TFLOPS (Blackwell path, used internally by cuBLAS)".

This is **INCONSISTENT WITH THE LIVE MEASUREMENTS** which span 1850 (realistic data) to 2325 (microbench peak).

**Recommended update to CLAUDE.md**:
- Change "BF16 tensor: ~1980 TFLOPS" → "BF16 tensor: ~2240 TFLOPS zero-data sustained, ~1850 realistic (cuBLAS 8K³)".

---

## E. Internal contradictions within 06_tensor_cores.md

### E1. BF16 mma.sync clock annotation
- Row 22: "**577 TFLOPS @ 2032 MHz boost**"
- Anatomy section: "Measured **577.45 TFLOPS @ 1920 MHz** = 98.9% of ncu-derived ceiling (582). Extrapolates to **~611 TFLOPS @ 2032 MHz**"

These contradict each other on whether 577 is the 1920 or 2032 number. **Pick one**. Per V8 (which is at 2032 boost): 578.6 TFLOPS → row 22 is the more accurate framing. Anatomy section needs editing.

### E2. FP8 sparse 7.44 PFLOPS
Listed in row 18 as "**HIGH** confidence" (the table column) but Retirement #4 downgrades it to MEDIUM. Table column should be MEDIUM.

### E3. FP4 block-scaled 9856 TFLOPS
Listed as **HIGH** confidence in 06_tensor_cores row 15, but `M3_REVERIFY_LOG.md` line 47 lists it as "Still MED: 3 (FP4 9856 TFLOPS, ...)". Row 15 should be MED until 3-method re-verified.

---

## F. Suggested authoritative-cite chain

When a downstream user asks "what's the BF16 TFLOPS?", reply with:

1. **Default answer**: **2240 TFLOPS** (cuBLAS 8K³ zero-data sustained), drops to **1850 TFLOPS** for realistic Gaussian-distributed inputs.
2. If they ask peak ceiling: **2325 TFLOPS** (microbench tcgen05 direct).
3. If they ask legacy mma.sync: **569 (1920 MHz) – 578 (2032 boost) TFLOPS**, with V8_HMMA_F16_PEAK as the highest-rigor measurement.
4. NEVER quote 1980 or 1543 TFLOPS for BF16.
