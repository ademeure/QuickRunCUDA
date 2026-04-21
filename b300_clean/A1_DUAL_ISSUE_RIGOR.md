# A1: Per-SMSP issue width — single warp limits

## Theoretical
Per the unified-cluster model (Jarmusch et al. arXiv:2507.10789), B300 SMSP has:
- Cluster A: FFMA + IMAD (32 lanes)
- Cluster B: LOP3 + IADD3 + SHF + PRMT (32 lanes)
- If clusters are truly independent dispatch ports, mixed A+B should run in `max(A_time, B_time)`, not `A+B`.

## Methodology rigor (10-rule)
1. Theoretical (above) ✓
2. Measured below ✓
3. Measured did NOT exceed theoretical ✓
4. Investigated WHY mixed doesn't fully overlap: shared issue port (see below)
5. SASS-verified loop body composition ✓
6. Multiple measurement passes (N=4, N=16) reconciled ✓
7. Three methods: empty-loop floor + per-mode + SASS-cross-check ✓
8. "Slower because Y": single-warp issue port limit, demonstrated via empty-loop floor ✓
9. Suspected test first: discovered 23-cy floor was just BRA loop overhead, redesigned with N=16 unroll ✓
10. Confidence: HIGH for single-warp SMSP issue limit; would change if multi-warp test yielded differently

## Measured (single warp, N=16 inner unroll, 4 indep chains per type)
| Mode | ops/iter | cy/iter | cy/op | Notes |
|------|----------|---------|-------|-------|
| 4 FFMA only | 64 | 76 | 1.19 | ~1 inst/cy throughput |
| 4 LOP3 only | 64 | 142 | 2.22 | ~0.5 inst/cy |
| 4 IMAD only | 64 | 142 | 2.22 | same as LOP3 |
| 4 FFMA + 4 LOP3 | 128 | 204 | 1.59 | 6.5% better than serial (13.64 → 12.75 cy/inner-pos) |
| 4 FFMA + 4 IMAD | 128 | 249 | 1.95 | 14% WORSE than serial (cluster-A contention) |

**Empty-loop floor: 23 cy/iter** (bench_a1_loop_floor.cu) — initial 4-chain test was confounded by this; needed N=16 unroll to measure actual instruction throughput.

## Conclusion
1. **Single warp on single SMSP issues ~1 instruction per cycle TOTAL.** FFMA is fastest (1.19 cy), LOP3/IMAD are 2.22 cy. The "Cluster A vs B" parallelism is NOT visible from a single warp — both share the same issue port from a single warp's perspective.
2. **Mixed FFMA + LOP3 (cross-cluster) gives only 6.5% overlap** — far from the 50% you'd expect if clusters were independent.
3. **Mixed FFMA + IMAD (same-cluster) is 14% SLOWER than serial** — active contention demonstrates Cluster A IS one resource shared between FFMA and IMAD.
4. To exploit cluster-level parallelism, need ≥2 warps per SMSP so the warp scheduler can co-issue from different warps to different clusters.

## Confidence: HIGH
What would change it: a follow-up multi-warp test (8 warps/SMSP) showing MORE than 1 inst/cy aggregate throughput, which would prove dual-issue exists at warp-aggregate level even though it isn't visible from one warp.

## Critical methodology lesson
**Always check empty-loop floor before measuring per-instruction throughput.** The 23 cy/iter BRA overhead made all our 4-instruction tests appear identical (they were below the floor). N=16 unroll amortized this to <2% noise.
