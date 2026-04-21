# V9: Mixed FMA + ALU pipes — partial overlap, NOT 2× peak

## Verification of "114 TOPS combined" claim from V9_INT_OPS_PIPES.md

**Hypothesis tested**: A kernel with both FFMA and IADD interleaved should
hit FMA pipe + ALU pipe simultaneously, doubling total throughput.

## Measurements (sweep N_FMA × N_INT chains)

| N_FMA | N_INT | FMA % | ALU % | Sum % | Effective combined work |
|-------|-------|-------|-------|-------|------------------------|
| 4     | 4     | 64.5  | 64.5  | 129%  | ~73 TOPS              |
| 4     | 8     | 40.5  | 80.9  | 121%  | ~63 TOPS              |
| 4     | 16    | 22.4  | 89.5  | 112%  | ~51 TOPS              |
| 8     | 4     | 78.4  | 39.2  | 118%  | ~74 TOPS              |
| **8**     | **8**     | **65.6**  | **65.6**  | **131%**  | **~74 TOPS**         |
| 8     | 16    | 39.7  | 79.4  | 119%  | ~60 TOPS              |
| 16    | 4     | 87.9  | 22.0  | 110%  | ~74 TOPS              |
| 16    | 8     | 79.0  | 39.5  | 118%  | ~74 TOPS              |

## Key findings

### Both pipes ARE active simultaneously
Sum of pipe utilizations exceeds 100% (peak 131% at 8/8) — proves the
hardware DOES overlap pipes. Not just sequential.

### But combined throughput plateaus near solo FMA peak (~75 TOPS)
The 114 TOPS hypothesis was WRONG. Mixing FFMA + IADD plateaus at
~74 TOPS combined work, similar to solo FFMA's 75 TFLOPS peak.

### Why not 2× peak?
- Each warp issues 1 instruction per cycle (can't issue FMA + IADD same cycle)
- 4 SMSPs per SM can dispatch in parallel, but each warp has serialized issue
- Full 2× would need TWO warps/SMSP issuing different ops simultaneously
- HW does this partially (131% summed pipe activity) but not fully

## Honest conclusion

Mixing pipes is **useful** but NOT a 2× perf hack:
- ✓ Integer ops PIGGYBACK at moderate cost
- ✓ Mixed kernel processes both work types at ~combined rate of single peak
- ✗ Cannot stack peaks for 2× throughput
- ✓ For kernels that need BOTH (e.g., GEMM with address arithmetic),
  the IADDs are essentially "free" work alongside FFMAs

## Practical recipe

For kernels mixing FMA + INT (most ML workloads):
- **Best ratio: ~8 FMA / 4 INT** (FMA 88%, ALU 22%) — keeps FMA near peak
  while using ALU for cheap address ops
- **Avoid: ALU-heavy mixes** (4 FMA / 16 INT) — FMA drops to 22%, total throughput suffers

## Correction to V9_INT_OPS_PIPES.md

That doc claimed "Mixed FFMA + IADD: up to 114 TOPS combined." This is
**MISLEADING**. Empirical max is ~74 TOPS combined for mixed work. The
114 TOPS would require 2 different warps issuing in parallel via HW
scheduling — which doesn't happen at full rate.

## 10-rule application

This is a clean Rule 9 follow-up: original "114 TOPS" claim was
SUM of theoretical peaks, not measured. Re-test with actual mixed
kernel showed empirical max = solo FMA peak + small fraction.

Confidence: **HIGH** for "mixing partially overlaps but ≠ 2× peak".