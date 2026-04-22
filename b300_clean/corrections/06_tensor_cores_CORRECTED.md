# 06_tensor_cores — CORRECTED Per-Precision Verified Peaks

**Auditor**: tensor swarm, 2026-04-22
**Authority order**: B300_TRUE_REFERENCE.md > V8/V9 measurements > TCGEN05_* power-ladder kernels > 06_tensor_cores.md catalog claims

This file consolidates the verified peak tensor-core throughput numbers on
B300 SXM6 (sm_103a, 148 SMs), separated by data-pattern (zero / random /
realistic) and clock state, and explicitly marks retracted catalog claims.

---

## 1. Per-precision peak ladder (data-aware)

All numbers are **chip-wide TFLOPS** unless flagged. "Zero" = const/zero
input data (best case, often used for cuBLAS spec quotes). "Random" =
uniformly-random fill. "Realistic" = normal-ish distribution proxy.
Drops vs zero come directly from `B300_TRUE_REFERENCE.md` § 2 table
(commit 6e40ef9, N=8192 cuBLAS).

### tcgen05 (Blackwell 5th-gen) path — cuBLAS internal

| Precision | Zero TFLOPS | Random TFLOPS | Realistic TFLOPS | % of NVIDIA spec | Clock | Confidence | Source |
|-----------|------------:|--------------:|-----------------:|-----------------:|:-----:|:----------:|--------|
| **FP16** (cuBLAS, 8K³) | 2246 | 1905 | 1744 | 91% of 2465 spec | 1920 | **HIGH** | TRUE_REFERENCE row 66 |
| **BF16** (cuBLAS, 8K³) | 2246 / 2242 | 1883 | 1850 | 90% of 2500 spec | 1920 | **HIGH** | TRUE_REFERENCE rows 48, 67 |
| BF16 microbench (tcgen05 direct) | 2325 | n/a | n/a | 93% | 1920 | HIGH | 06_tensor_cores row 20 |
| **TF32** (cuBLAS, 8K³) | 1113 | n/a | n/a | 90% of 1232 | 1920 | HIGH | 06_tensor_cores row 21 |
| **FP8 e4m3** (cuBLAS, 8K³, sustained) | 4425–4491 | 3983 | 3951 | 88-91% of 5000 spec | 1920 | **HIGH** | TRUE_REFERENCE rows 56–57, 68 |
| FP8 microbench (tcgen05 direct) | 4651 | n/a | n/a | 93% | 1920 | HIGH | 06_tensor_cores row 17 |
| **NVFP4** wide-N M=8192 N=65536 K=16384 | 10297 | n/a | n/a | **103% of 10 PF spec** | boost | **HIGH** | TRUE_REFERENCE row 51 |
| NVFP4 e2m1 cuBLAS square N=K=24576 | 8424 | 8424 | n/a | 84% of 10 PF | boost | HIGH | TRUE_REFERENCE row 52 |
| NVFP4 e2m1 single-shot N=8192 const | 9109 | n/a | n/a | 91% | boost | HIGH | TRUE_REFERENCE row 55 |
| NVFP4 e2m1 sustained random N=16384 | 6554 | 6554 | n/a | 65% (heavy throttle to 1057 MHz) | boost | HIGH | TRUE_REFERENCE row 54 |
| NVFP4 microbench K=64 1500 MHz | 7270 | 7270 | n/a | 73% | 1500 lock | HIGH | TCGEN05_PERFW_CLEAN row K=64 |
| NVFP4 K=96 ULTRA microbench, 1500 MHz | **10910** | 10910 | n/a | (= 73% of 15 PF spec) | 1500 lock | HIGH | TCGEN05_PERFW_CLEAN row K=96 |
| NVFP4 K=96 cuBLAS-accessible peak | **inaccessible via cuBLAS 13.2** | — | — | path exists (UTCOMMA SASS) but cuBLAS won't pick it | — | HIGH | TRUE_REFERENCE row 53 |
| NVFP4 K=96 cuBLAS 13.4 wide-rect | ~10800 | — | — | 72% of 15 PF spec | boost | MED | memory entry, prior session |
| **FP4 block-scaled microbench** (kind::mxf4nvf4.block_scale.block16) | 9856 | n/a | n/a | 99% of 10 PF spec | 1920 | MED (re-verify pending per M3_REVERIFY_LOG) | 06_tensor_cores row 15 |

### mma.sync (legacy warp-sync) path

| Precision | TFLOPS | Notes | Clock | Confidence | Source |
|-----------|-------:|-------|:-----:|:----------:|--------|
| **FP16/BF16 m16n8k16** (V8 8-chain, F16 acc) | **578.6** | 99.9% pipe_tensor saturation, 94.72M HMMAs in 670 µs | boost 2032 | **HIGH** | V8_HMMA_F16_PEAK |
| FP16/BF16 m16n8k16 (V8 F32 acc) | 578.6 | identical, F32 acc free | boost 2032 | HIGH | V8_HMMA_VARIANTS_PEAK |
| BF16 m16n8k16 (TRUE_REFERENCE figure) | 569 | 7.4× FFMA, "matches catalog burst 569" | 1920 | HIGH | TRUE_REFERENCE row 47 |
| BF16 mma.sync 8-chain "burst" (commit 83ef1c6) | ~570 | matches V8 within 2% | 1920 | HIGH | TRUE_REFERENCE row 58 |
| TF32 m16n8k8 | 288 | half of FP16 (K=8 vs K=16) | ~2032 | MEDIUM | 06_tensor_cores row 23 |
| INT8 m16n8k32.s32.s8 | 143 TOPS | HW-throttled (5 NOPs/issue) | ~2032 | HIGH | 06_tensor_cores row 24 |
| FP8 mma.sync (kind::f8f6f4) | 276 (effective) | NOT native — F2FP.UNPACK + HMMA emulation | 1920 | MED (DCE-suspect) | MMA_FP8_KIND_F8F6F4_NOT_NATIVE |
| FP4 mma.sync | REJECTED on sm_103a | only sm_120a Geforce | — | HIGH | 06_tensor_cores row 27 |

### FP64 tensor

| Operation | TFLOPS | Notes | Source |
|-----------|-------:|-------|--------|
| DMMA / DGEMM | **1.05** | matches DFMA — no FP64 tensor speedup on B300 | 06_tensor_cores rows 28-29 |

### HMMA latency (single-MMA serial chain)

| Op | Latency (cy) | Throughput cap | Sat ILP | Source |
|----|-------------:|---------------:|--------:|--------|
| HMMA m16n8k16 F32 acc | 20.09 (converged) | 1/(4cy)/SMSP | 5 chains | V9_HMMA_LATENCY |

---

## 2. RETRACTIONS

These claims appear in the codebase and are **wrong / superseded**.

### R1. "1543 TFLOPS BF16 single-chain mma.sync" — RETRACTED
- **Source of retraction**: `B300_TRUE_REFERENCE.md` row 58, "single-chain '1543' was over-counted; RETRACTED".
- True peak: ~570 TFLOPS (8-chain multi-accumulator) and 578.6 TFLOPS (V8 99.9% pipe saturation).
- Note: 1543 also appears legitimately as the NVLink-5 P2P **bidirectional GB/s** number (12_nvlink_p2p.md) and as one cell in N_DEPENDENCE_DEEPDIVE M=N=K=28672 — those are unrelated and not retracted.

### R2. "6 357 TFLOPS FP8 via mma.sync" — RETRACTED
- DCE-folded loop, only 2 HMMAs in SASS for claimed 65K iters. Catalog self-retracts (06_tensor_cores retirement #1).

### R3. "2 336 / 2 400 TFLOPS FP8 via mma.sync" — RETRACTED
- FADD artifact (compiler folded 99.99% of MMA chain). Catalog self-retracts (06_tensor_cores retirement #2).

### R4. "ncu pipe_tensor measures tcgen05" — DOES NOT APPLY to tcgen05
- Per the user-memory entry "ncu pipe_tensor doesn't measure tcgen05".
- Files using pipe_tensor metrics:
  - `V8_HMMA_F16_PEAK.md` row 13 — `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active = 99.90%`. **OK** because V8 measures legacy mma.sync HMMA (F16 m16n8k16), not tcgen05. The pipe_tensor metric IS the right counter for legacy mma.sync.
  - `V8_HMMA_VARIANTS_PEAK.md` — same context, OK.
  - `M14_V8_SOL_LADDER.md` line 109 — generic "ncu cross-check (pipe_fma / pipe_tensor / pipe_xu / dram)". Generic, OK if applied to mma.sync.
  - `06_tensor_cores.md` line 11 — "Same `pipe_tensor` 128 cy/MMA at M=128, N=256" — **POTENTIALLY MISLEADING** if read as a tcgen05 measurement; tcgen05 cy/MMA was actually measured via `clock64` in the lane-fast benchmarks (TCGEN05_N_SWEEP / TCGEN05_N_MATRIX), NOT via `pipe_tensor`. Recommend a footnote saying the 128 cy/MMA is from clock64 measurements and that `sm__pipe_tensor_*` does not see UTCHMMA / UTCOMMA.
  - `SESSION_2_DELTA.md` lines 524-790 — explicitly DEMONSTRATES the misleading nature: pipe_tensor showed 99% active on a tcgen05 kernel for only 2.42 ms of 9.37 ms wall-clock; concludes "pipe_tensor active% may have scope limitations". This is the strongest internal evidence for the user-memory rule. **OK** as a documented gotcha.
  - `CUBLAS_BIT_ENTROPY_CORRECTION.md` line 1086-1087 — uses `sm__pipe_tensor_subpipe_hmma_cycles_active.sum` AND `smsp__inst_executed_pipe_tensor_subpipe_hmma_op_utchmma_utcqmma_utcomma_scope_2cta.sum`. The latter explicitly contains `utchmma_utcqmma_utcomma` so DOES include tcgen05 ops. **Verify usage** — if author counted them as MMAs without unpacking the metric naming, watch for inflation.
  - `SUBTILE_DEDUP_MODEL.md` line 109 — "Cross-check with NCU `tpc__l1tex_*` or `sm__pipe_tensor_*` counters" — **MISLEADING** for the tcgen05 dedup discussion; flag.

### R5. "FP8 cuBLAS Not Available on B300" — RETRACTED
- Buggy descriptors in original test. Real FP8 via LtMatmul works at 88-91% MFU (06_tensor_cores retirement #3).

### R6. "FP8 sparse 7.44 PFLOPS = 74% of 10 PF spec" — DOWNGRADED to MEDIUM
- Sparse metadata may be garbage; the 7.44 PFLOPS is a steady-state ceiling for that test only (06_tensor_cores retirement #4).

### R7. "830 TB/s / 295 TB/s TMEM read" — RETRACTED
- DCE-inflated; real ~60 TB/s (06_tensor_cores retirement #5).

### R8. "838 / 420 TFLOPS HMMA FP16/TF32" — RETRACTED
- ILP-override bug; superseded (06_tensor_cores retirement #6).

### R9. "INT8 tensor would scale with ILP" / "151 TOPS INT8 latency-bound" — RETRACTED
- INT8 IMMA SASS shows 5 NOPs/issue → HW-throttled, not latency-bound. 143 TOPS is the steady-state ceiling regardless of ILP (06_tensor_cores retirement #9).

### R10. "FP4 block-scaled rejected on sm_103a" — RETRACTED
- Was an early `kind::mxf4` rejection; correct PTX `kind::mxf4nvf4.block_scale.block16` works (06_tensor_cores retirement #10).

### R11. "tcgen05 unsupported on sm_103a" — RETRACTED
- Refers to static ptxas in CUDA 13.2 only; NVRTC supports it (06_tensor_cores retirement #8, TCGEN05_PATH_NOTES).

---

## 3. UNRESOLVED / OPEN

### U1. NVFP4 K=96 cuBLAS-accessible vs spec
- Spec for B300 NVFP4 = 10000 TFLOPS (B200 base). The 1.5× K=96 ULTRA path is real (UTCOMMA.BLOCK16 confirmed in SASS) and the microbench reaches 10.91 PFLOPS at 1500 MHz lock (TCGEN05_PERFW_CLEAN), but cuBLAS 13.2 will NOT dispatch it (TRUE_REFERENCE row 53).
- cuBLAS 13.4 on K=96 large-N rect reportedly reaches ~10.8 PF = 72% of 15 PF spec (per memory entry), but NOT verified in this clean directory.
- CUTLASS / CuTeDSL stuck at 8.7 PF = 58% per memory entry (and SESSION_2_DELTA confirms 7757 TF = 51.7% as best CuTeDSL config).
- Bottom line: **the 1.5× K=96 ULTRA path is not reachable in any public library on this version**; effective cuBLAS NVFP4 ceiling is ~10.3 PF (wide-N) per TRUE_REFERENCE.

### U2. NVFP4 single-shot vs sustained disparity
- Single-shot const N=8192: 9109 TFLOPS (91% of 10 PF spec).
- Sustained random N=16384 cudaGraph 15s: 6554 TFLOPS (65%) — clock throttles to 1057 MHz, 1186 W instant peak.
- Gap of ~28% is power-throttle, not algorithmic. Recommended quotation: "NVFP4 best single-shot const = 9-10 PF; sustained random = 6.5 PF".

### U3. cuBLAS internal A↔B swap for NVFP4
- NVFP4_PURE_TCGEN05_RESULTS.md flags that the cuBLAS NVF4 "A dominates power" observation may be due to cuBLAS internally swapping A and B before issuing UTCOMMA. Pure-tcgen05 microbench shows B-dominance for ALL 6 precisions (BF16/FP16/FP8 e4m3/e5m2/NVFP4 K=64/K=96), with B/A power ratio 13-29×. Swap unverified.

### U4. FP4 block-scaled (9856 TFLOPS) rigor
- M3_REVERIFY_LOG line 47 lists "FP4 9856 TFLOPS" as still MED confidence (not yet 3-method verified). Should be flagged in any quote.

### U5. Data-dependent throughput drop (cuBLAS, N=8192)
- TRUE_REFERENCE table (commit 6e40ef9) shows:
  - FP16 random vs zero: −15%; normal-ish vs zero: −22%
  - BF16 random vs zero: −16%; normal-ish vs zero: −18%
  - FP8 random vs zero: −9%; normal-ish vs zero: −10%
- Under 600 W power cap: FP8 random hits 3087 TFLOPS = −43% from zero baseline.
- **Quoting any cuBLAS peak without specifying data pattern is misleading**. The 4500/2200/2200 numbers are zero-data; subtract 10-22% for realistic.

### U6. Catalog header in 06_tensor_cores says "Same `pipe_tensor` 128 cy/MMA at M=128, N=256"
- Cycle count 128 is correct (cross-verified by clock64 in TCGEN05_N_SWEEP), but attributing the measurement source to "pipe_tensor" without qualifier risks confusion since pipe_tensor doesn't see tcgen05. Recommend rewriting to "Same 128 cy/MMA (clock64-measured) at M=128, N=256".

---

## 4. Quick-cite cheat sheet

| Need | Use this number | Source |
|------|-----------------|--------|
| BF16 cuBLAS realistic | **1850 TFLOPS** | TRUE_REFERENCE row 67 |
| BF16 cuBLAS zero (best case) | **2246 TFLOPS** | TRUE_REFERENCE row 66-67 |
| BF16 mma.sync legacy | **569-578 TFLOPS** | V8 + TRUE_REFERENCE row 47 |
| FP16 mma.sync legacy | **578 TFLOPS** | V8_HMMA_F16_PEAK |
| FP8 cuBLAS realistic | **3983 TFLOPS** | TRUE_REFERENCE row 57 |
| FP8 cuBLAS zero (best case) | **4425 TFLOPS** sustained / **4486** microbench | TRUE_REFERENCE row 56 |
| FP8 cuBLAS under 600 W cap | **3087 TFLOPS** | TRUE_REFERENCE warning |
| TF32 cuBLAS | **1113 TFLOPS** | 06_tensor_cores |
| NVFP4 cuBLAS best (wide-N, single-shot) | **10297 TFLOPS** (103% B200 spec) | TRUE_REFERENCE row 51 |
| NVFP4 cuBLAS sustained random | **6554 TFLOPS** (heavy throttle) | TRUE_REFERENCE row 54 |
| NVFP4 K=96 ULTRA microbench (no cuBLAS) | **10910 TFLOPS** @ 1500 MHz lock | TCGEN05_PERFW_CLEAN |
| FP4 block-scaled microbench | 9856 TFLOPS (re-verify pending) | 06_tensor_cores |
| INT8 mma.sync | 143 TOPS (HW-throttled) | 06_tensor_cores |
| FP64 DMMA / DGEMM | 1.05 TFLOPS (no tensor speedup) | 06_tensor_cores |
| HMMA latency m16n8k16 F32 | 20 cy | V9_HMMA_LATENCY |
