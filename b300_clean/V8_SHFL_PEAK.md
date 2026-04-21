# V8: SHFL.BFLY (warp shuffle) = 3.0 G warp-SHFL/s

## Findings

- 128 `SHFL.BFLY` in loop body, SASS-verified
- Time: 3.16 ms for 9.47 G warp-SHFLs (148 SMs × 8 warps × 62500 × 128)
- Rate: 3.0 G warp-SHFL/s
- Per SM per cycle: 0.01 warp-SHFL/cy/SM (= 1 per 100 cy per SM)

## Interpretation

SHFL throughput is far below "1 per cy per SMSP" that some docs suggest.
The chain dependency (v = shfl(v)) may add per-SHFL latency, serializing.

ncu pipe metrics all show 0% — SHFL doesn't map to FMA/ALU/XU; likely uses
MIO path which isn't covered by those specific counters.

## Confidence: MEDIUM

- Rate matches SASS-verified count
- But mapping to architectural peak unclear without `pipe_shuffle` metric

## Practical takeaway

For warp reductions using SHFL, expect ~3 G ops/s aggregate. At 256-thread
blocks (8 warps/SM), per-SM rate = 20 M SHFL/s. Don't expect FP32-like
throughput from shuffle-heavy code.

Alternative: `redux.sync.min/max` (Blackwell) is reportedly 4× SHFL on B300
per V4 prior findings.