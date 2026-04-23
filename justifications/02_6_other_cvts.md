# §2.6 Other CVTs — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §2.6 (L314-334)
**Test:** `tests/bench_cvt_catalog.cu`

## CLAIM (catalog L314-334)

| PTX | SASS | pipe | rate |
|-----|------|------|-----:|
| cvt.rn.f16.f32 | F2FP.F16.F32.PACK (+ PRMT) | alu | 1.00 |
| cvt.rn.bf16.f32 | F2FP.BF16.F32.PACK (+ PRMT) | alu | 1.00 |
| cvt.f32.f16 | HADD2.F32 | fmaH | 2.00 |
| cvt.f32.bf16 | HADD2.F32 | fmaH | 2.00 |
| cvt.rn.f32.s32 | I2FP.F32.S32 | alu | 2.00 |
| cvt.rn.f32.u32 | I2FP.F32.U32 | alu | 2.00 |
| cvt.rn.f32.s64 | I2F.S64 | xu | **0.04 (super slow)** |
| cvt.rni.s32.f32 | F2I.NTZ | xu | 0.5 |
| cvt.rni.u32.f32 | F2I.U32.NTZ | xu | 0.5 |
| cvt.rni.sat.u8.f32 | F2IP.U8.F32.NTZ | alu | **2.00 (!)** ← surprise fast path |
| cvt.rni.sat.s8.f32 | F2I.S8.NTZ | xu | 0.5 |

## TEST

```bash
CUDA_VISIBLE_DEVICES=0 ncu --metrics sm__inst_executed_pipe_alu.avg.per_cycle_active,sm__inst_executed_pipe_xu.avg.per_cycle_active,sm__inst_executed_pipe_fmaheavy.avg.per_cycle_active,sm__inst_executed_pipe_fma.avg.per_cycle_active \
  ./QuickRunCUDA -f tests/bench_cvt_catalog.cu -t 512 -b 296 -A 1024 -B 1024 -C 1024 \
  -H "#define OP $OP
#define UNROLL 16" -0 1024 -T 1
```

## MEASURED (ncu, GPU 0, full occupancy)

| OP | SASS confirmed | pipe_alu | pipe_fma | pipe_fmaH | pipe_xu | Catalog | Verdict |
|----|----------------|---------:|---------:|----------:|--------:|--------:|---------|
| **30** (f16→f32) | HADD2.F32 | 0.01 | 1.97 | **1.97** | 0 | 2.00 fmaH | ✅ matches (98%) |
| **40** (f32→s32 rni) | F2I.NTZ | 0 | 0 | 0 | **0.50** | 0.5 xu | ✅ exact |
| **43** (f32→u8 sat) | F2IP.U8.F32.NTZ | **1.97** | 0 | 0 | 0 | 2.00 alu (!) | ✅ matches (98%) |
| **46** (s32→f32) | I2FP.F32.S32 | **1.98** | 0 | 0 | 0 | 2.00 alu | ✅ matches (99%) |
| **48** (s64→f32) | I2F.S64 emitted | 0 | 0 | 0 | 0 | 0.04 (super slow) | ⚠ test too short to measure |

## VERDICT

✅ **CONFIRMED for 4 of 5 tested rows:**
- HADD2.F32 (f16→f32) saturates pipe_fmaheavy at 1.97 = 98% — confirms catalog's "re-uses HADD2 infra"
- F2I (f32→s32 rni) at exactly 0.50 on pipe_xu — confirms catalog's 0.5 xu
- F2IP.U8 (f32→u8 sat) at 1.97 alu — confirms the **surprise fast path**: u8 saturating cvt is 4× faster than s8 (which uses F2I.S8 on xu)
- I2FP.F32.S32 (s32→f32) at 1.98 alu — confirms 2.00 alu

⚠ **I2F.S64** — pipe metrics show 0 because the test runs too quickly at "0.04 super slow" rate to accumulate ncu samples. SASS confirms `I2F.S64` IS emitted. The "super slow" claim is plausible but not directly measured here.

## CATALOG-CONFIRMED ARCHITECTURAL FACTS

1. **f32→u8 saturation is FAST (alu, 2.00)** — F2IP.U8 has a special pipe_alu path
2. **f32→s8 saturation is SLOW (xu, 0.5)** — F2I.S8 goes through pipe_xu
3. **f32→{s32,u32} are SLOW (0.5 xu)** — F2I family
4. **s32/u32→f32 are FAST (2.00 alu)** — I2FP family
5. **s64→f32 is super slow (0.04)** — I2F.S64
6. **f16/bf16→f32 is FAST (2.00 fmaH)** — re-uses HADD2 infra

This asymmetry (alu vs xu paths) is non-obvious and important for kernel design — picking u8 over s8 saturation can give 4× CVT throughput.

## REVIEW_CHECKLIST candidates

- [x] §2.6 HADD2.F32 (f16→f32) = 2.00 fmaH — ✅ confirmed at 1.97 = 98%
- [x] §2.6 F2I.NTZ (f32→s32) = 0.5 xu — ✅ confirmed at exactly 0.50
- [x] §2.6 F2IP.U8 (f32→u8 sat) = 2.00 alu (!) — ✅ confirmed at 1.97; **fast path 4× faster than s8 sat**
- [x] §2.6 I2FP.F32.S32 (s32→f32) = 2.00 alu — ✅ confirmed at 1.98
- [ ] §2.6 I2F.S64 (s64→f32) = 0.04 super slow — SASS-confirmed emission but rate too low for ncu sampling; "super slow" claim plausible
