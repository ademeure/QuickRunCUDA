# §2.5 Narrow-format CVT (F2FP family) — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §2.5 (L277-314)
**Test:** `tests/bench_cvt_from_narrow.cu` (existing, well-designed)

## CLAIM (catalog L277-314)

> "All six [UNPACK formats] peak identically at **2.00 warp-inst/SM/cy = 128 elements/SM/cy** when no co-issuing ALU op."
>
> | PTX | SASS | rate | elements/SM/cy |
> |-----|------|-----:|---------------:|
> | cvt.rn.f16x2.e4m3x2 | F2FP.F16.E4M3.UNPACK_B | 2.00 | 128 |
> | cvt.rn.f16x2.e5m2x2 | F2FP.F16.E5M2.UNPACK_B | 2.00 | 128 |
> | cvt.rn.f16x2.e2m1x2 (FP4) | F2FP.F16.E2M1.UNPACK_B | 2.00 | 128 |
> | cvt.rn.f16x2.e2m3x2 (FP6) | F2FP.F16.E2M3.UNPACK_B | 2.00 | 128 |
> | cvt.rn.f16x2.e3m2x2 (FP6) | F2FP.F16.E3M2.UNPACK_B | 2.00 | 128 |
> | cvt.rn.bf16x2.ue8m0x2 | F2FP.BF16.E8.UNPACK_B | 2.00 | 128 |

## TEST

```bash
CUDA_VISIBLE_DEVICES=0 ncu --metrics sm__inst_executed_pipe_alu.avg.per_cycle_active,sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active \
  ./QuickRunCUDA -f tests/bench_cvt_from_narrow.cu -t 1024 -b 296 -A 1024 -B 1024 -C 1024 \
  -H "#define CVT_ASM cvt.rn.f16x2.e4m3x2
#define UNROLL 16" -0 8192 -T 1
```

(For e2m1x2 FP4: add `#define CVT_B8` to the header for `.b8` input wrapping.)

## MEASURED (ncu, GPU 0, full occupancy 4 CTAs/SM)

| PTX | SASS emitted | pipe_alu inst/cy/SM | % of peak |
|-----|--------------|--------------------:|----------:|
| cvt.rn.f16x2.e4m3x2 | F2FP.F16.E4M3.UNPACK_B | **2.00** | 99.98% |
| cvt.rn.f16x2.e5m2x2 | F2FP.F16.E5M2.UNPACK_B | **2.00** | 99.98% |
| cvt.rn.f16x2.e2m1x2 (FP4) | **F2FP.F16.E2M1.UNPACK_B** | **2.00** | 99.98% |
| cvt.rn.f16x2.e2m3x2 (FP6) | F2FP.F16.E2M3.UNPACK_B | **2.00** | 99.99% |
| cvt.rn.f16x2.e3m2x2 (FP6) | F2FP.F16.E3M2.UNPACK_B | **2.00** | 99.99% |
| cvt.rn.bf16x2.ue8m0x2 | F2FP.BF16.E8.UNPACK_B | **2.00** | 99.99% |

## VERDICT

✅ **CONFIRMED** — all 6 narrow-format UNPACK ops saturate pipe_alu at exactly 2.00 inst/SM/cy = 99.98%+ of architectural peak.

**At 1.92 GHz:** 2.00 inst/SM/cy × 148 SMs × 1.92 GHz × 32 lanes × 2 elements/inst = **36.4 G elements/s/chip** for each format (= catalog §5 claim).

**FP4 is NOT faster or slower per SASS instruction than FP8 on B300's ALU pipe.** All formats share the same 64 warp-inst/SM/cy ceiling on pipe_alu.

## NOT TESTED (preserved from catalog)

- PACK rates (e.g., F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C ≈ 1.0 alu solo with LOP3 pollution)
- Round-trip PACK + UNPACK (catalog claims 2.00/SM/cy total split 1:1)
- FP4 packed PACK with extra mov (~0.45 alu)

These are claimed by catalog at lower rates due to LOP3 feedback chain pollution; not independently re-verified here.

## REVIEW_CHECKLIST candidates

- [x] §2.5 UNPACK 6 formats all = 2.00 = 99.98% — ✅ CONFIRMED
- [x] §2.5 catalog claim "FP4 not faster than FP8" — ✅ CONFIRMED (all at same rate)
- [ ] §2.5 PACK rates (1.0 / 0.67 / 0.45) — NOT verified independently
- [ ] §2.5 round-trip PACK+UNPACK = 2.00 split 1:1 — NOT verified independently
