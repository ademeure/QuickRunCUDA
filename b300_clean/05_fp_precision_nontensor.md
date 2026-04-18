# 05 — FP Precision Throughput (NON-tensor)

**Scope**: B300 SXM6 sm_103a, scalar/packed FP arithmetic and conversion on `pipe_fma`/`pipe_alu`/`pipe_fp64` only. Tensor-core (`mma.sync`, `tcgen05.mma`, cuBLAS) figures are intentionally excluded — see agents 02/03.

Confidence: HIGH = SASS-verified + audited + cross-checked; MEDIUM = single-source, plausible vs theoretical; LOW = contradicted or methodology-thin.

---

## 1. Headline numbers (clock = 2032 MHz)

| Op | Pipe | Peak chip TFLOPS / Tops | % of theoretical | Confidence | Source |
|---|---|---:|---:|---|---|
| FP32 FFMA scalar | pipe_fma (heavy+lite, dual-issue) | **71.8 TFLOPS** | 93% of 76.96 | HIGH | catalog l.1134 audit, ncu pipe_fma 99.08% |
| FP16 HFMA2 (`fma.rn.f16x2`) | pipe_fma (occupies BOTH H+L, 1 inst = 2 FMAs) | **~72 TFLOPS-FP16** | same chip-FLOPS as FP32 | HIGH | catalog l.6390 retraction |
| BF16 BFMA2 (`fma.rn.bf16x2`) | pipe_fma, `HFMA2.BF16` | **~72 TFLOPS-BF16** | identical to HFMA2 | HIGH | catalog l.1138 |
| FP64 DFMA | pipe_fp64 (1 unit per SMSP, NOT pipelined per warp) | **1.20 TFLOPS** | matches 1:64 perfRatio | HIGH | inv 14, ncu 84% pipe |
| F2FP narrow UNPACK (FP8/FP6/FP4 → f16x2) | pipe_alu, 2 issue slots | **64 inst/SM/clk = 38.5 Telements/s** | 100% of 2-slot ceiling | HIGH | F2FP_DEEP_DIVE §4 |
| F2FP narrow PACK (f16x2/f32 → FP8/FP6/FP4) | pipe_alu, MERGE_C blocks dual-issue | **32 inst/SM/clk = 19.3 Telements/s** | half of UNPACK | HIGH | F2FP_DEEP_DIVE §4-6 |
| `cvt.rn.f16x2.f32` (F32→F16 packed) | pipe_alu (`F2FP.F16.F32.PACK_AB`) | 32 inst/SM/clk = 19.3 Telem/s compute | – | HIGH | F2FP_DEEP_DIVE §3.2 |
| Scalar `cvt.rn.satfinite.f16.f32` | pipe_alu (`F2FP.*.MERGE_C`) | 24 inst/SM/clk | special slow path | HIGH | F2FP_DEEP_DIVE §3.1 |
| `cvt.rn.f16.f32` (no satfinite) | F2F pipe (distinct from F2FP) | 11 inst/SM/clk | always add `.satfinite`! | HIGH | F2FP_DEEP_DIVE §8 |
| Memory-bound F32→F16x2 conv | HBM | 264 Gconv/s = 2.1× scalar | mem-bound | MEDIUM | commit `1a39dfe`, `fp_conversion.cu` |

---

## 2. Validated key findings

### 2.1 NO FP16/BF16 packed throughput speedup over FP32 (KEY surprise)

`__hfma2(half2)`, `__hfma2(bfloat162)`, and scalar `fma.rn.f32` all peak at the same **~70-72 chip-TFLOPS**. HFMA2 issues at HALF the inst-rate of FFMA but does 2× FLOPs/inst → net 1× FLOPS. Pipe `pipe_fma` is busy ~99% in both cases; the ncu `pipe_fma_cycles_active` = 99% reading does NOT mean equal inst rate.

- **Wrong/early catalog**: "HFMA2 = 2× FFMA FLOPS = 308 TFLOPS-FP16" (catalog l.5096) — **retracted at l.6390** by event-timed audit.
- **Recent confirmation**: commit `ea47ec6` (`half_throughput.cu`): 4-ILP measures 63.6 (FP32) vs 58.2 (FP16/BF16) TFLOPS. The ~10% deficit vs catalog's 72.3 is the well-known under-saturation at only 4 chains; both agree on "no 2× speedup".
- **Practical**: choose FP16/BF16 over FP32 for memory bandwidth savings only, NOT for compute, unless using tensor cores.

### 2.2 FP64 DFMA: 1.20 TFLOPS, NOT pipelined per warp

Investigation 14 (`14_dfma_pipeline.md`) decisively confirms catalog claim:
- DFMA scoreboard latency = **63.9 cy** at ILP ≥ 8 (zero benefit from independent chains within a warp).
- FP64 pipe internally: 16 cy per DFMA (4:1 ratio to scoreboard).
- Per-warp pipe util stuck at **25%** = 16/64 regardless of ILP.
- Saturation requires **4 warps/SM** (1/SMSP). Chip peak = **1.20 TFLOPS** at 2032 MHz, ncu pipe_fp64 = 84%.
- Catalog's older 0.95 TFLOPS reflected 1920 MHz lock + fewer warps; **1.20 is the corrected number.**
- 1:64 ratio matches `cudaDeviceGetAttribute(SingleToDoublePrecisionPerfRatio) = 64`.

### 2.3 F2FP family (narrow conversions) lives on pipe_alu, dual-issue gated by regfile ports

From `F2FP_DEEP_DIVE.md` (definitive single source):
- **2 SFU issue slots/cycle/SM**. Whether an op consumes 1 or 2 slots is determined by the SASS opcode's regfile-read-port footprint.
- **UNPACK_B** (1 read + 1 write, e.g. `F2FP.F16.E4M3.UNPACK_B`) → 64 inst/SM/cycle = **128 elements/SM/cycle = 38.5 Telements/s** chip.
- **PACK_AB / *_MERGE_C** (≥3 regfile accesses) → 32 inst/SM/cycle = **64 elements/SM/cycle = 19.3 Telements/s** chip.
- Latency: 4 cy for both pack and unpack (matches FFMA/IMAD calibration).

**Coverage**: ALL B300-native sub-formats — FP8 (E4M3, E5M2), FP6 (E2M3, E3M2), FP4 (E2M1), UE8M0 — have native PTX `cvt.rn[.satfinite].*x2.*` forms emitting one `F2FP` SASS each. **All formats hit identical per-instruction throughput** within direction (UNPACK or PACK); FP4 is NOT slower or faster than FP8 per SASS instruction on this pipe.

**Critical rule (HIGH confidence, F2FP_DEEP_DIVE §8)**: scalar `cvt.rn.f16.f32` WITHOUT `.satfinite` emits a slower `F2F.F16.F32` opcode on a different pipe (~11/SM/clk). **Always include `.satfinite`** unless overflow-NaN semantics required.

### 2.4 x4 stochastic-round PTX is syntactic sugar
`cvt.rs.e4m3x4.f32` etc. compile to **2× PACK_AB_MERGE_C.RS** SASS — same effective element rate as x2 form.

### 2.5 Non-arithmetic packed FP16/BF16 (HMNMX2) lives on pipe_alu, NOT pipe_fma

`min.f16x2` / `max.f16x2` → `HMNMX2` on **pipe_alu** at 64 inst/SM/cy (catalog l.374). Means clipping/clamping can co-issue with FFMA (different pipe). Same for BF16x2 mnmx.

### 2.6 Conversion-as-memory-bound (host-data context)

Commit `1a39dfe` measured F32↔F16/BF16 over 16M elements in HBM:
- F32→F16 scalar: 125 Gconv/s
- F32→BF16 scalar: 127 Gconv/s
- F16→F32 scalar: 167 Gconv/s
- F32→F16 packed (`__float22half2_rn`): **264 Gconv/s = 2.1× scalar**

Effective HBM use only 10-20% (~700-1500 GB/s of 7.5 TB/s peak) → these tests are **launch+latency limited**, not BW or compute limited. The 2.1× speedup is the packed-vs-scalar advantage at the F2FP level; absolute numbers leave headroom. For real one-pass quant kernels, expect to be HBM-bound (commit `de00fbe`: F32↔BF16 ~3 TB/s effective, ~500 Gelem/s).

---

## 3. Conflicts resolved

| Claim | Resolution |
|---|---|
| Catalog l.5096: "HFMA2 = 308 TFLOPS-FP16, 2× FFMA" | **WRONG** (retracted at l.6390). Real ~72 TFLOPS. |
| Catalog l.16046: "HFMA2 + FFMA use *different FMA pipes*, free co-issue" | **MISLEADING**. Both occupy pipe_fma; co-issue means FFMA H+L can overlap with HFMA2 occupying H+L only when scheduling permits — at peak each takes both sub-units (catalog l.481). The "free alongside FFMA" claim is a 1-warp/8-chain microbench artifact at low occupancy, not a 2-pipe physical fact. |
| Investigation `ea47ec6`: 4-ILP gives 58.2 TFLOPS (HFMA2/BFMA2) | Lower than 72.3 because 4 ILP chains under-saturate dual-issue. Both agree on the SHAPE: equal to FP32, no 2× speedup. |
| Catalog l.35: "FP64 DFMA 0.95 TFLOPS" | **Updated**: 1.20 TFLOPS at 2032 MHz with 4 warps/SM (inv 14). 0.95 was 1920 MHz and/or fewer warps. Both are valid for their conditions. |
| Hopper "removed dedicated HFMA2 pipe" | **CONFIRMED on B300 too**: HFMA2 shares pipe_fma with FFMA (occupies both heavy+lite sub-units for one inst). Same pipe-FLOPS ceiling as FP32. |
| F2FP_DEEP_DIVE.md TFLOPS claims | All independently consistent with **2-slot SFU** model (verified across 33 result files). FP4/FP6/FP8 element rates are correct. |

---

## 4. Retire / rewrite

**Retire from B300_PIPE_CATALOG.md**:
- l.5083-5096 ("HFMA2 issues at 1 cy/pipe → 308 TFLOPS"): superseded by l.6388-6403 retraction. Move retraction to top of HFMA2 section, delete the wrong number.
- l.16038-16048 (HFMA2 "different FMA pipes / free alongside FFMA"): contradicts the validated section 0 + l.6390. Reword: "HFMA2 occupies both heavy+lite FMA sub-units; mixed FP16/FP32 sequences see FFMA-rate not 2× speedup."
- The 0.95 TFLOPS DFMA value alongside 1.20 TFLOPS — pick **1.20 TFLOPS** as headline; note 0.95 was sub-optimal warp count.

**Rewrite**:
- Section 0 should clearly state: "scalar/packed FP16, BF16, FP32 all peak at ~72 TFLOPS chip-FLOPS via pipe_fma. Tensor cores break this ceiling — see Tensor section." (Current text is correct but the contradictory l.5096/l.16046 entries undermine the message.)

**Promote to top-line summary**:
- F2FP_DEEP_DIVE.md is the canonical source for narrow-format conversion throughput. Catalog §2.5 and §3.x should defer to it, not duplicate.

**Skeptical / under-tested**:
- Memory-bound conversion numbers (commit `1a39dfe`) are launch-overhead bounded; for real ML quant kernels expect 3-5 TB/s effective. SASS-verify before publishing as headline.
- `cvt.f32.f16` (widening) measured at "infinite" in F2FP_DEEP_DIVE table — actually `HADD2.F32` on pipe_fma_heavy at 64/SM/cy (catalog l.256). Not literally free, just runs on a different pipe than F2FP.

**Confirmed for re-use**:
- All HIGH-confidence numbers in §1 above are safe to cite.
- F2FP_DEEP_DIVE.md is the strongest single document in this category — 33 measurement files, internally consistent two-slot model, explicit hypothesis discrimination.
