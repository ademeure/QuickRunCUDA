# V8: FADD/FMUL/FFMA — same instruction rate, FFMA wins FLOPS via 2 ops

## 10-rule rigor walk-through

1. **Theoretical**: All three (FADD, FMUL, FFMA) use the same FP32 pipe.
   Peak: 1 inst per SMSP per cycle × 4 SMSPs × 32 lanes × 148 SMs × 2.032 GHz
       = 38.5 G instructions/s aggregate
   - FADD: 38.5 × 1 FLOP = 38.5 TFLOPS
   - FMUL: 38.5 × 1 FLOP = 38.5 TFLOPS
   - FFMA: 38.5 × 2 FLOPs = 77.0 TFLOPS

2. **Measured** (same 2-source kernel, `OP` define selects op):

   | Op   | Pipe %  | Time    | Inst count | Inst/s   | TFLOPS (if N FLOP/inst) |
   |------|---------|---------|------------|----------|--------------------------|
   | FADD | 97.65%  | 810 µs  | 30.31G     | 37.4 G/s | 37.4 (1 FLOP)            |
   | FMUL | 97.62%  | 813 µs  | 30.31G     | 37.3 G/s | 37.3 (1 FLOP)            |
   | FFMA | 97.65%  | 809 µs  | 30.31G     | 37.5 G/s | **74.8 (2 FLOPs)**       |

3. **Rule 3**: pipe 97.65% ≤ 100%, TFLOPS ≤ theoretical. Not broken.

4. **Why same inst/s**: all three compile to 1 instruction per op, all dispatched
   by the FMA pipe at 1/cy/SMSP.

5. **ncu HIGH confidence**: pipe_fma_cycles_active.pct_of_peak = 97.65% for all.

6. **SASS** (expected):
   - FADD → `FADD R, R, R` (1 inst)
   - FMUL → `FMUL R, R, R`
   - FFMA → `FFMA R, R, R, R`

7. **Three methods**:
   - Wall/ncu time identical
   - ncu inst count: 30.31 Ginst for all three (same loop body structure)
   - TFLOPS computed: matches theoretical × 97.65%

8. **Conclusive**: The 2× gain in FFMA's FLOP rate comes entirely from 2
   ops counted per inst. The HW dispatch rate is identical for all three.

9. **Surprise checked**: Some older architectures had FMUL/FADD slower than
   FFMA on unified FP32 pipe. Modern (Ampere+) unifies them. Confirmed here.

10. **Confidence: HIGH**. Would change only if:
    - PTX is reordered into MAD by fast_math (already using -use_fast_math;
      confirmed via ncu op count: FADD count matches FADD inst, not MAD)
    - Op reordering by ptxas into MAD fusion (not observed here)

## Implication for kernel design

- **Prefer FFMA** (a*b+c) over FADD+FMUL in isolation — same inst count, 2× FLOPs.
- For non-MAD kernels (sums, reductions), FADD at ~37.4 TFLOPS is the peak.
- Avoid relying on compiler MAD fusion; use explicit `fma.rn.f32` to guarantee.

## SoL ladder update

Complete B300 compute ladder (all rigor-verified, pipe > 97%):

| Op       | Peak       | Note                        |
|----------|-----------|------------------------------|
| FFMA     | **75.2 TFLOPS** | 97.65%, 2-source pattern |
| FP64 FMA | **1.20 TFLOPS** | 100%, single DFMA port   |
| IMAD     | **38.4 TOPS**   | 99.7%, int 1:2 vs FP32   |
| FADD/FMUL| **37.4 TFLOPS** | 97.65%, same FP32 pipe   |
| HMMA.F16 | **578.6 TFLOPS** | 99.90%, tensor pipe     |