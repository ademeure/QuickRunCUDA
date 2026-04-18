# Tensor Cores — B300 SXM6 (sm_103a)

**Hardware**: 148 SMs, 1 tensor pipe per SM, async MMA via TMEM (Blackwell gen-5).
**Clock annotations**: peaks @ 2032 MHz boost unless flagged 1920 MHz (multi-GPU power-throttle).
**Sources**: `investigations/03_mma_sync_peak.md`, `investigations/02_cublas_fp8.md`, `B300_PIPE_CATALOG.md` lines 113-130, 1083-1109, 5260-5295, 6700-6790, 9271-9322, 9792-10140, 17935; `AUDIT_NOTES.md`; `CRITIQUE.md`.

---

## Precision Ladder (the only table you need)

All numbers measured on sm_103a, single chip (148 SMs), MMA-only (no DRAM). Same `pipe_tensor` 128 cy/MMA at M=128, N=256.

| Precision | Best path | Chip TFLOPS / TOPS | % of NVIDIA spec | Confidence | Clock |
|-----------|-----------|-------------------:|-----------------:|:----------:|:-----:|
| **FP4 block-scaled** (E2M1 + UE4M3, K=64) | `tcgen05.mma kind::mxf4nvf4.block_scale.block16` (UTCOMMA.BLOCK16) | **9 856** (9.9 PFLOPS) | 99% (~10 PF spec) | **HIGH** | 1920 |
| **FP8 dense** (E4M3/E5M2, K=32) | cuBLAS LtMatmul → internal `tcgen05.mma kind::f8f6f4` | **4 486** sustained @ M=N=K=8192 | **91%** of 4929 theor | **HIGH** | 1920 |
| FP8 dense (microbench) | `tcgen05.mma kind::f8f6f4` direct | **4 651** (4.65 PFLOPS) | **93%** | HIGH | 1920 |
| FP8 sparse 2:4 | `tcgen05.mma kind::f8f6f4.sp` | **7 440** (7.44 PFLOPS) | 74% of 10 PF | MEDIUM (metadata may not be properly 2:4 — see retirement #4) | 1920 |
| **FP16 / BF16 dense** (K=16) | cuBLAS → `tcgen05.mma kind::f16` | **2 259** sustained @ 8192³ (cuBLAS 13.4) | **92%** of 2465 theor | **HIGH** | 2032 |
| FP16 / BF16 (microbench) | `tcgen05.mma kind::f16` direct | **2 325** | **93%** of 2465 | HIGH | 1920 |
| **TF32** (K=8) | `tcgen05.mma kind::tf32` / cuBLAS TF32 | **1 113-1 163** | **90-93%** of 1232 theor | HIGH | 1920 |
| FP16 / BF16 — **mma.sync legacy** (K=16, m16n8k16) | `mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32` ILP=2, 64w/SM | **577 TFLOPS @ 2032 MHz boost** (CORRECTION: catalog "1920→611 extrap" was wrong — NVML showed 2032 throughout). 93.7% of 616 theoretical | tcgen05 path 4× faster (2325) | HIGH | **2032 boost** |
| TF32 — mma.sync legacy (m16n8k8) | `mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32` | **288** | half of mma.sync FP16 (K=8 vs K=16) | MEDIUM | ~2032 |
| **INT8 dense** | `mma.sync m16n8k32.s32.s8.s8.s32.satfinite` (IMMA.16832.S8.S8.SAT) | **143 TOPS** | ~3% of theoretical FP8 — **deliberately throttled** (5 NOPs/issue) | HIGH | ~2032 |
| INT8 via tcgen05 | `tcgen05.mma kind::i8` | **NOT SUPPORTED on sm_103a** (gated to sm_100a/100f/110a/110f) | — | HIGH | — |
| FP8 — mma.sync legacy | `mma.sync m16n8k32.kind::f8f6f4` | **276** (emulated: F2FP.UNPACK + HMMA, NOT native QMMA) | ~6% of native FP8 | MEDIUM (DCE-suspect — see retirement #1) | 1920 |
| FP4 — mma.sync legacy | — | **REJECTED on sm_103a** (only sm_120a Geforce) | — | HIGH | — |
| FP64 (DMMA) | `mma.sync.m16n8k4.f64.f64.f64.f64` (SASS `DMMA.8x8x4`) | **1.05 TFLOPS** (REFUTED catalog "~2 TF") — same as DFMA, same as cuBLAS DGEMM | DMMA goes through the same FP64 pipe as DFMA; no parallel tensor path on B300 | HIGH | 2032 |
| **FP64 cuBLAS DGEMM** | `cublasDgemm` (CUDA 13.2 / cuBLAS 13.4) | **1.05 TFLOPS** (87% of 1.20 DFMA spec) — ALL configs identical | **No FP64 tensor speedup** (commit `126e052`): GemmEx + COMPUTE_64F + TENSOR_OP also 1.05 TF; B300 FP64 tensor effectively dead | HIGH | 2032 |

**Pattern**: Within `tcgen05.mma`, throughput is exactly proportional to K dimension. TF32:FP16:FP8:FP4 = 1 : 2 : 4 : 8 (K=8, 16, 32, 64). Identical 128 cy/MMA across all formats at M=128, N=256.

---

## Anatomy

### tcgen05.mma — the real B300 tensor path

- **Async, warp-group orchestrated**: needs `tcgen05.alloc` (TMEM column slot), TMA/UBLKCP load of A/B into smem, idesc encoding (UMMA::InstrDescriptor bit layout), `tcgen05.commit.mbarrier::arrive`, `tcgen05.wait`, `tcgen05.dealloc`, `tcgen05.relinquish_alloc_permit`.
- **Single tensor pipe per SM** — multiple warps issuing tcgen05.mma serialize at the dispatcher (TFLOPS/SM is constant across 1, 2, 4 warps).
- **148-SM scaling: linear** (128.13 cy/MMA at any SM count → 31.4 TFLOPS/SM × 148 = 4 651 TFLOPS chip).
- **Shape**: `cy/MMA = max(44, N/2)` for M=128. Use M=128 N≥128 for full pipe utilization. M=64 caps at 50%. M=256 requires `cta_group::2` (cluster of 2 CTAs).
- **GEMM K-pipeline crossover**: at K≥6 steps per pipeline stage, MMA is the bottleneck (TMA hides). Below K=6, TMA-limited (MMA free in shadow). Crossover is format-independent.

### mma.sync (legacy warp-synchronous) — 4× slower than tcgen05

`mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32` is the pre-Blackwell tensor-core PTX. On B300 it goes through the same `pipe_tensor` but with the warp-synchronous dispatch model (1 HMMA per SMSP per ~8 cycles). Per-SM rate caps at 0.50 HMMA/cy (verified via `sm__inst_executed_pipe_tensor.avg.per_cycle_active`).

**Optimal config** (from investigations/03):
- Kernel: `mma_bf16_ilp2<<<SM_COUNT*2, 1024>>>` with `__launch_bounds__(1024, 2)`
- 22 regs/thread → 2 blocks × 1024 threads/SM = 64 warps/SM
- ILP=2 (two independent accumulator chains per warp) hides 26-cy HMMA latency
- ILP=1: 362 TFLOPS (latency-bound). ILP=8 catalog config (32w/SM): 530 TFLOPS. ILP=16: spills, 10× slower
- Measured **577.45 TFLOPS @ 1920 MHz** = 98.9% of ncu-derived ceiling (582). Extrapolates to **~611 TFLOPS @ 2032 MHz**

**FP16 vs BF16 vs FP16-acc**: identical (same HMMA pipe, FP16 accumulator gives no speedup unlike older arch).

### cuBLAS internals

- **FP8 kernel name** (ncu-verified): `nvjet_sm103_qqtst_128x256_128x6_2x2f_2cta_h_bz_NNT` — cuBLAS dispatches `tcgen05.mma kind::f8f6f4` internally, despite NOT exposing tcgen05 as user API. **No user-facing API needed for FP8 tensor peak**; just call `cublasLtMatmul` with E4M3 inputs.
- **TF32 cuBLAS** at 8K³: 1113 TFLOPS (90% of 1232 theoretical).
- **BF16 cuBLAS** at 8K³: 2259 TFLOPS (92% MFU). Identical for FP16. Scales cleanly to 24K³.
- **FP8 sustained**: 30 sec at 8192³ → 4491 TFLOPS held FLAT (1.00× ratio across 600 batches). Power: 180 W → 899 W (89% TGP). Clock STAYS at 2032 MHz. **No throttling under full tensor load**.
- **`-lgc 2032` paradoxically pins to 1920 MHz** — TFLOPS measurements vary 6% by clock state.
- **Sgemm FP32** (CUBLAS_COMPUTE_32F, no TF32): **68.6 TFLOPS** — non-tensor scalar FFMA path. **Route to FP32 agent.**
- **INT8 cuBLAS does NOT use tensor cores** (CUBLAS_COMPUTE_32I falls back to scalar IMAD): 66 TOPS, 22-34× slower than BF16 at same shape. Dequantize to FP8/BF16 instead.

### ldmatrix / stmatrix (LDSM / STSM)

- `pipe_uniform + pipe_lsu` shared dispatch.
- `ldmatrix.x4.b16`: **2.30 cy/warp-inst** (0.43 ldmatrix/cy/SM). 666 GB/s/SM smem-read peak. **6× faster than `ld.shared.v4`**.
- `ldmatrix.b8x16.b6x16_p32` (FP8/FP6 LDSM Blackwell): **identical 2.30 cy** — pipe is element-width-agnostic.
- `stmatrix.x4.b16`: **32 cy/warp-inst** = **14× slower than ldmatrix**. Use `st.shared.v4` for register→smem write-back.
- **LDSM fully hides behind HMMA**: ILP=16 HMMA + 1 ldmatrix/16 HMMA = identical time (0.140 vs 0.141 ms). Free overlap in real GEMM tile-streaming.

### Tensor Memory (TMEM)

- 256 KB/CTA = 512 columns × 128 lanes × 4 B. Bump-pointer allocator, max 1 alloc per CTA (sizes pow2: 32/64/128/256/512 cols).
- `tcgen05.alloc.cta_group::1, 128`: **253 cy** (uncontended). Under chip-wide contention: ~1030 cy.
- **Read peak: ~60 TB/s chip** (32x32b.x16 = 53 B/cy/warp). Down from earlier "295/830 TB/s" claims (catalog self-retracted those as DCE-inflated).
- **Write peak: 97-131 TB/s chip** (`tcgen05.st.32x32b.x4`). Asymmetric write-heavy — matches MMA accumulator usage.
- **Write hides behind read** (1R+3W = same chip BW as 4W; "read essentially free when writes dominate").
- **`tcgen05.alloc/dealloc/relinquish` are `.sync.aligned`** — must be called by ALL threads. Putting alloc behind `if(tid==0)` deadlocks the warp.

### WMMA fragments

- **Deprecated path** for B300. Catalog has no microbench; assume it lowers to `mma.sync.m16n8k16` (legacy) and shares the 577 TFLOPS ceiling, NOT the 2.3 PFLOPS tcgen05 peak.

---

## Driver / Compiler Notes (FLAG)

1. **`tcgen05.mma` and `tcgen05.alloc` compile via NVRTC but NOT via static ptxas in CUDA 13.2** on sm_103a. ptxas error: "Instruction 'tcgen05.alloc' not supported on .target 'sm_103'". This is a **compiler-version restriction, NOT a hardware limitation** — QuickRunCUDA's NVRTC path runs them correctly. Earlier "tcgen05 unsupported on B300" claims conflated ptxas rejection with HW absence.
2. **`wgmma.*` (Hopper warp-group MMA) is REJECTED on sm_103a** — port to `tcgen05.mma`.
3. **`kind::i8` rejected on sm_103a** — gated to sm_100a/100f/110a/110f. Use `kind::f8f6f4` with `kind::i8`-style idesc encoding for INT8 tensor (or just use FP8).
4. **`kind::mxf4` (default block32) and `kind::mxf8f6f4` rejected by ptxas 13.2**; only `kind::mxf4nvf4.block_scale.block16` works for FP4 block-scaled.

---

## Conflict Resolution

| Reported number | Source | Reconciliation |
|-----------------|--------|----------------|
| 514 / 540 / 544 / 569 / 576 / 577 / 580 TFLOPS BF16 mma.sync | various sections | All correct for **same kernel at different (clock × ILP × occupancy)** points. Final answer: **577 TFLOPS @ 1920 MHz** with optimal ILP=2/64w-SM config; **545 TFLOPS @ 1920 MHz** with catalog ILP=8/32w-SM config; the "577" cheat-sheet entry was at 2032 MHz with catalog config (extrapolates the same way: 545 × 2032/1920 = 577). 514 was an under-saturated ILP=4 measurement. |
| 6 357 / 2 336 / 2 247 TFLOPS "FP8 mma.sync" | CATALOG line 1093-1101, line 1154 | **All wrong** — DCE-folded or FADD-artifact. Catalog self-retracts. **Real mma.sync FP8 = 276 TFLOPS** (emulated F2FP+HMMA path, ncu-verified). |
| 4 474 vs 4 486 vs 4 491 vs 4 651 TFLOPS FP8 | various | All correct: 4486 = cuBLAS sustained @ 1920 MHz (investigations/02); 4491 = same, 30s sustained; 4474 = older measurement same kernel; 4651 = direct tcgen05 microbench peak (no cuBLAS overhead). |
| "FP8 cuBLAS Not Available" (line 17883) vs "4474 TFLOPS measured" | CATALOG | The "Not Available" section was a **buggy test** (wrong scaleType / computeType). **FP8 via cublasLtMatmul WORKS** for all E4M3-input combos; only E5M2×E5M2 fails NOT_SUPPORTED (investigations/02). |
| 2 325 vs ~1 980 vs 2 465 TFLOPS BF16 | spec vs measured | 2 465 = theoretical at 1920 MHz; 2 325 = measured @ 93% of theor; ~1 980 = NVIDIA-quoted (older spec). Use **2 325** as measured peak. |
| 1 109 vs 1 113 vs 1 163 TFLOPS TF32 | various sections | All correct, within noise: 1 109 from agent finding, 1 113 from cuBLAS sweep @ 8K, 1 163 from tcgen05 microbench. Use **~1 113** as cuBLAS-measured. |
| 142 vs 143 vs 151 TOPS INT8 | catalog | All correct (different kernels, ~5% noise). The 151 TOPS in agent finding may include latency-bound single-chain inflation. **143 TOPS is steady-state IMMA peak**; "would scale with ILP" is wrong — IMMA is HW-throttled (5 NOPs per issue, 65 cy/inst), ILP doesn't help. |

---

## Retirement Section (numbers to NOT cite)

1. **"6 357 TFLOPS FP8 via mma.sync"** — DCE-folded loop, only 2 HMMAs in SASS for claimed 65 K iters. CATALOG line 1101 self-retracts.
2. **"2 336 / 2 400 TFLOPS FP8 via mma.sync"** — FADD artifact (compiler folded 99.99% of MMA chain). CATALOG line 27, 1101 self-retracts.
3. **"FP8 cuBLAS Not Available on B300"** (CATALOG line 17883-17892) — buggy test (wrong descriptors). FP8 via cuBLAS LtMatmul WORKS at 91% MFU.
4. **"FP8 sparse 7.44 PFLOPS = 74% of 10 PF spec"** — flagged by AUDIT_NOTES as sparse-metadata-may-be-garbage. The 7.44 PFLOPS is the steady-state ceiling for the test, but proper 2:4 metadata may yield closer to spec. Cite as "MEDIUM confidence" only.
5. **"830 TB/s TMEM read"**, **"295 TB/s TMEM read"** — DCE-inflated, CATALOG line 1265-1293 retracts. Use **~60 TB/s**.
6. **"838 / 420 TFLOPS HMMA FP16/TF32"** (CATALOG line 2071-2073) — ILP-override bug, superseded by current numbers. Don't cite.
7. **"154 TFLOPS FP32"** — 2× formula error (assumed 256 FP32 cores/SM; B300 has 128 like Hopper). Real FP32 = 65-77 TFLOPS scalar, route to FP32 agent.
8. **"tcgen05 unsupported on sm_103"** — refers to static ptxas in CUDA 13.2 only. NVRTC supports it; HW supports it. The agent finding is correct: flag as a driver/compiler bug, NOT a HW limitation.
9. **"INT8 tensor would scale with ILP" / "151 TOPS INT8 latency-bound"** — INT8 IMMA SASS shows 5 NOPs per issue (HW-throttled, not latency-bound). 143 TOPS is the steady-state ceiling regardless of ILP. INT8 deliberately deprecated on B300.
10. **"FP4 block-scaled rejected on sm_103a"** — was an early `kind::mxf4` rejection; the correct PTX `kind::mxf4nvf4.block_scale.block16` DOES compile and delivers 9.9 PFLOPS. Catalog updates.

---

## Quick Recipe — Hit the Tensor Peak

| Goal | Path | Config |
|------|------|--------|
| FP4 9.9 PFLOPS | `tcgen05.mma kind::mxf4nvf4.block_scale.block16` | M=128, N=256, A in TMEM, scale UE4M3, K-step ≥ 6 |
| FP8 4.5-4.65 PFLOPS | `cublasLtMatmul` E4M3 → BF16 OR direct `tcgen05.mma kind::f8f6f4` | M=N=K ≥ 8192, COMPUTE_32F |
| FP16/BF16 2.3 PFLOPS | `cublasGemmEx` BF16 → BF16, F32 compute | M=N=K ≥ 8192, multiple-of-128 |
| TF32 1.1 PFLOPS | `cublasGemmEx` FP32 → FP32, COMPUTE_32F_FAST_TF32 | M=N=K ≥ 8192 |
| BF16 mma.sync 577 TFLOPS (legacy) | `mma_bf16_ilp2<<<SM*2, 1024>>>` `__launch_bounds__(1024, 2)` | bs=1024, 2 blocks/SM, ILP=2, 64 warps/SM |
| Anti-DCE | accumulator init from `(float)warp_id * 1e-30f`; final write under `if(__float_as_int(sum) == seed)` | — |

**Always**: avoid GEMM dim X+1 past tile boundary (M=4097 = 11× slower than M=4096 due to cuBLAS algo selection). Pad to multiple of 8 minimum, 128 optimal.
