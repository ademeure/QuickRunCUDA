# __syncwarp / sync primitive cost — V4 / F6

**Date: 2026-04-20.** Test `bench_syncwarp_cost.cu`. 1500 MHz, single
warp single block, 8 ops/iter compute work, clock64 timing.

## Result (cy per iteration, including 8 compute ops)

| Sync method | cy/iter | Cost vs baseline |
|-------------|---------|------------------|
| no sync (baseline) | 23.00 | 0 |
| `__syncwarp(0xFFFFFFFF)` | 24.00 | **+1 cy** |
| `__syncwarp(0x0000FFFF)` (16 lanes) | 25.00 | +2 cy |
| `bar.warp.sync 0xFFFFFFFF` (PTX) | 24.00 | +1 cy |
| `__syncthreads()` (single warp block) | 25.00 | +2 cy |
| `membar.cta` (memory fence) | 29.00 | **+6 cy** |

## Key takeaway: __syncwarp() is essentially FREE

**A fully-convergent `__syncwarp(0xFFFFFFFF)` costs only 1 cycle.**
This destroys the common belief that warp sync is expensive. Use it
liberally to:

- Avoid implicit-divergence bugs (post-Volta no implicit reconverge)
- Document the convergence point in code clearly
- Get clean SASS without surprising scheduling behavior
- Force warp convergence after divergent code paths

## Restricted-mask syncwarp slightly slower

Using a partial mask (e.g., 16 lanes) costs 2 cy vs 1 for full mask.
Possibly due to mask-matching overhead in the BSYNC instruction.

## __syncthreads() on single-warp block

Only 2 cy because there's only one warp — no actual cross-warp
synchronization to do. Multi-warp blocks would be much more expensive
(see commit `a9736ca`: 12 ns @ 32 thr → 49 ns @ 1024 thr in B300_TRUE_REFERENCE).

## membar.cta is the heaviest

6 cy for a memory fence (no thread sync, just orders memory ops at
CTA scope). This is heavier than __syncwarp+__syncthreads combined!
Use sparingly.

## Practical implications

For multi-warp kernels:
- **Sprinkle __syncwarp liberally** — barely measurable overhead
- **Avoid membar unless really needed** — 6× more expensive than syncwarp
- **__syncthreads cost grows with block size** (per `a9736ca`)
- **Replace mask-restricted syncwarp with full mask** when possible

## Confidence

- **HIGH** for syncwarp ≈ 1 cy (3 trials, 1 cy delta clear)
- **HIGH** for membar.cta = 6 cy
- **MED** for "syncthreads on multi-warp" — only tested single-warp
  here, deferred to existing `a9736ca` finding for the curve

## Files

- `tests/bench_syncwarp_cost.cu` — modes 0-5
