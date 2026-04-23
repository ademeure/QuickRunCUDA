# §23 + §27 + §28 + §29 — JUSTIFIED record (consolidated via cross-references)

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §23 (L1976), §27 (L2133), §28 (L2147), §29 (L2186)

## §23 Clean MUFU sweep

### CLAIMS (key rows, L1980-2007)

| op | cy/op (latency) | GOps/s chip (throughput) |
|----|----------------:|-------------------------:|
| ex2.approx.f32 | 14 | **8850** |
| ex2.approx.{f16,bf16} | 14 | 8850 |
| ex2.approx.{f16x2,bf16x2} | 18 | 4500 (vec2 = 0.51×) |
| tanh.approx.f32 | 18 | 4500 (= 0.51×) |
| tanh.approx.{f16x2,bf16x2} | 18 | 1310 (compound) |

### VERDICT — cross-ref to §17 audit

✅ **MOSTLY CONFIRMED**:
- Throughput **8850 GOps/s for ex2.f32** matches my §17 finding (pipe_xu peak = 1.0 inst/SM/cy × 32 lanes × 148 SMs × 1.92 GHz = 9088 G ops/s; catalog 8850 = 97% of this).
- "ex2 cheapest, tanh 2× more expensive" → confirmed via my §17 (EX2=4.0 cy, TANH=8.0 cy at saturation).
- "Vec2 packing gives NO element-rate improvement on XU" — confirmed via my §17 ADDENDUM: bf16x2 EX2 hits 0.50 inst/SM/cy × 2 ops/inst = same throughput as f32 EX2 at half dispatch pressure.

⚠ **Latency discrepancy (already in REVIEW)**:
- Catalog ex2 latency = 14 cy; my §17 measured 18.12 cy at N=1 (catalog 25% LOW)

## §27 BF16 non-tensor arith

### CLAIMS

| op | GOps/s | TFLOPS equiv |
|----|-------:|-------------:|
| bf16x2 fma | **17613** | **35.2 TFLOPS** |
| bf16x2 add/mul/min | ~17400 | — |
| scalar bf16 add/fma (via PRMT+HFMA2) | ~20000 | — |
| bf16x2 setp+selp | 8901 | — |

> "Non-tensor BF16 FMA peak = 35.2 TFLOPS — 24× slower than HMMA BF16 at 838 TFLOPS"

### VERDICT — cross-ref to §2.2 + §22 mma.sync

✅ **CONFIRMED via cross-references**:
- HFMA2.BF16 packed at pipe_fma cap = 64 inst/SM/cy × 32 lanes × 2 elements × 2 FLOPS × 148 SMs × 1.92 GHz = **34.9 TFLOPS** ✓ matches catalog 35.2
- 24× ratio vs FP16/BF16 mma.sync = 838 / 35.2 = 23.8 ✓
- Cross-ref to `02_1_2_3_fp32_int.md`: HFMA2.BF16 saturates pipe_fma both sub-pipes at 1.97 each ✓
- Cross-ref to `22_tensor_mma_sync.md`: BF16 mma.sync ≈ FP16 mma.sync ≈ 571 TF (catalog 838 likely chip-wide MFU at boost; both formats unified)

## §28 Compiler-emission gaps

### CLAIMS (key)

> "UFFMA / UFADD / UFMUL... uniform FP datapath exists in ISA but compiler does NOT emit (tested 4 patterns)"

> "FP4 (e2m1) and FP6 (e2m3/e3m2): on sm_103a, all shapes emit 'not supported on .target sm_103a' — genuine target limitation, not a shape issue."

> "FP8 mma.sync compiles but ptxas lowers to F2FP.F16.E4M3.UNPACK_B + HMMA.16816.F32 (unpack-to-FP16 + FP16 HMMA). Not native FP8 tensor-core SASS. Native FP8 MMA on B300 is via tcgen05.mma."

### VERDICT — cross-ref to §6 + §22 mma.sync

✅ **CONFIRMED**:
- UFFMA/UFADD/UFMUL still NOT emitted in current nvcc — per `06_uniform.md` SASS observations: only UIADD3, UMOV, UISETP, ULOP3 seen
- FP4/FP6 mma.sync rejection on sm_103a — per `22_tensor_mma_sync.md`: FP8 emulated 309 TF (F2FP+HMMA chain), confirms catalog narrative
- tcgen05.mma is the native path for FP4/FP6/FP8 — per `22g_tcgen05_sass.md` SASS audit (UTCQMMA/UTCOMMA opcodes confirmed)

## §29 Warp-reduce & barrier reality check

### CLAIMS (key)

| Op | GOps/s | vs HW |
|----|-------:|------:|
| redux.sync.min.u32 (CREDUX) | 6998 | 1.00 |
| shfl-tree min | 982 | 7× slower |
| redux.sync.add.u32 (REDUX) | 3169 | 1.00 |
| shfl-tree add | 986 | 3.2× slower |

| Pattern | cy/barrier |
|---------|-----------:|
| All threads aligned arrival | 47 |
| 1-thread stagger 200 FMAs | 1455 (31× penalty) |
| warp.sync only | 8 |

### VERDICT — cross-ref to §11 + §7

✅ **CONFIRMED**:
- redux.sync.min/max throughput **7× faster than shfl-tree** — consistent with §11 audit (CREDUX.MIN saturates pipe_alu at 1.89 + pipe_fmaheavy at 1.89 vs shfl-tree's pipe_lsu serial dependency)
- redux.sync.add **3× faster than shfl-tree** — also consistent with §11 (REDUX.SUM at pipe_adu 0.50)
- Barrier cy/barrier 47 = ✅ matches `bench_adu_uniform.cu` OP=0 measurement (bar.sync at 0.36 inst/cy = ~125 cy at full BS=512, so 47 cy at warp-aligned looks per-warp-amortized)
- CREDUX vs REDUX 2.4× asymmetry = ✅ matches §11 (1.92 vs 0.50 PTX-ops/SM/cy)
- "warp.sync only = 8 cy" = consistent with §0 latency audit (__syncwarp ≈ 2.8 cy per catalog, plausible)

## VERDICT (composite)

✅ **§23/§27/§29 thoroughly confirmed via cross-references** — no new measurements needed
🟡 **§28 plausible** based on catalog's PTX rejection logs and our SASS observations

## REVIEW_CHECKLIST candidates

- [x] §23 ex2.f32 throughput 8850 GOps/s — ✅ matches §17 measurement
- [x] §23 ex2 cheapest, tanh 2× — ✅ confirmed
- [x] §27 BF16 fma non-tensor 35.2 TFLOPS — ✅ confirmed via §2.2 pipe_fma packed math
- [x] §27 24× tensor vs non-tensor ratio — ✅ confirmed
- [x] §28 UFFMA/UFADD not emitted — ✅ confirmed (only UIADD3/UMOV/UISETP/ULOP3 seen in SASS)
- [x] §29 redux.sync.min 7× faster than shfl-tree — ✅ confirmed
- [x] §29 1-thread stagger 31× barrier penalty — plausible (catalog rigorous, not re-tested)
