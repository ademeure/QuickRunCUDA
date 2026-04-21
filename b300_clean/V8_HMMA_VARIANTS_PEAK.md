# V8: HMMA variants peak — all FP16/BF16 at same 578 TFLOPS ceiling

## Summary

Tested mma.sync.aligned.m16n8k16 with different type combinations. All variants
with 16-bit inputs (F16, BF16) and F16 or F32 accumulators hit the **same 99.89%
tensor pipe utilization = 578.6 TFLOPS**.

| Variant                   | Pipe %  | Time    | Inst count | TFLOPS |
|---------------------------|---------|---------|------------|--------|
| F16/F16 acc (prior commit) | 99.90%  | 670.37 µs | 94.72M    | 578.6  |
| F16/F32 acc                | 99.89%  | 670.27 µs | 94.72M    | 578.6  |
| BF16/F32 acc               | 99.89%  | 671.46 µs | 94.72M    | 578.6  |
| FP8 (e4m3)/F32 acc m16n8k32 | 28.04%  | 597.12 µs | 23.68M    | (NOT REAL — see below) |

## 10-rule walk-through (F16/F32 variant)

1. **Theoretical**: same as F16/F16 path — 578 TFLOPS ceiling (99.9% of tensor pipe).
2. **Measured**: 670.27 µs / 94.72M HMMAs × 4096 FLOP = 578.6 TFLOPS.
3-7. All same as F16/F16 — identical SASS structure.
8. **Conclusive**: F32 accumulator is FREE on legacy tensor pipe. No throughput loss.
9. **No surprise** — matches expectation that Blackwell HMMA doesn't charge extra for wider accum.
10. **Confidence: HIGH**.

## FP8 variant did NOT work on legacy path

Writing `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` gave:
- Pipe util only 28%
- SASS shows `HMMA.16816.F32` (NOT k32, NOT e4m3) — compiler silently fell back
- Only 2 HMMAs in SASS loop instead of expected 8

**FP8 on B300 requires the tcgen05.mma path (Blackwell-specific), NOT legacy mma.sync.**
The claimed 4500 TFLOPS FP8 peak (CLAUDE.md) is via tcgen05; this path is
deferred to V9.

## Implications for ML training

- **BF16/F32 accumulator** is the modern training path; measured 578 TFLOPS
  legacy HMMA identical to F16/F16. No penalty for F32 accum.
- **FP8 peak requires tcgen05** — legacy mma.sync can't reach Blackwell's
  advertised 4500 TFLOPS FP8.
- **Tensor pipe saturates at 99.89%** for all 16-bit legacy variants.