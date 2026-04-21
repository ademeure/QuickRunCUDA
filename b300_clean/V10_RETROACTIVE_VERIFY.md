# V10: Retroactive SASS+ncu verification of prior BW claims

After DSMEM DCE discovery, systematically re-verify prior "HIGH confidence"
BW measurements via SASS + ncu to catch similar bugs.

## Verification results

| Claim                           | Verification                                         | Status |
|---------------------------------|------------------------------------------------------|--------|
| V9 cp.async HBM = 97% (6.98 TB/s) | SASS: 8× LDGSTS.E.128 in loop; ncu: 60.6M sectors × 32 B = 1.94 GB DRAM = matches expected | ✅ VALID |
| V8 SMEM LDS = 26.9 TB/s (plain) | SASS: 128 LDS in loop; ncu: 60.9M wavefronts × 128 B = 7.79 GB; bank conflicts=0 | ✅ VALID |
| V10 streaming BW 19.4 TB/s (1 MB L1) | ncu: 1.21G L1 sectors × 32 B = 38.8 GB; DRAM tiny (L1 hits) | ✅ VALID |
| **V8 DSMEM 37 TB/s** | SASS: LD.E (global-style); ncu: 7200 wavefronts vs 5.9B expected | ❌ **DCE'd** |
| **V10 DSMEM 48-67 TB/s** | same DCE pattern | ❌ **DCE'd** |
| **V10 DSMEM writes "4× slower"**| same pattern | ❌ **likely BOGUS** |

## What saved the other measurements

1. **cp.async**: The `cp.async.ca.shared.global` PTX emits `LDGSTS` SASS cleanly —
   can't be constant-folded because destination is SMEM and source address
   varies with iter counter. ncu confirmed real DRAM traffic.

2. **V8 SMEM LDS**: plain `ld.shared.u32` with varying index (loop counter i
   in address) → compiler CAN'T hoist. 8-way ILP × 16 unroll × 10000 iters
   all executed per ncu.

3. **V10 streaming**: independent indexed loads (`A[gtid + i*N]`) with `i`
   varying → no CSE possible. L1 serves from cache when footprint small.

## Why DSMEM DCE'd

DSMEM used inline-offset PTX (`[%base+32]`) with `base` being invariant across
iterations. Compiler saw the loads as loop-invariant → hoisted entirely.

If instead I'd used varying base (e.g., `base + i*stride`), loads would be
unique per iter — but that triggers the driver crash.

## Methodological lesson

**Rule 6 (SASS-verify) must be applied to LOAD COUNT, not just opcode.**
My prior verifications checked "is there an LDS instruction in the loop?" —
that was not enough. The question is "does the loop actually EXECUTE all the
LDS instructions per iter, or does the compiler hoist them?"

Correct rigor protocol:
1. Compile and dump SASS
2. Count load instructions in loop body
3. Count loop iterations in the test
4. Expected = (inner loads) × (iters) × (warps in grid)
5. Compare against ncu wavefront / sector count
6. If ncu << expected → DCE detected, result BOGUS

## Corrections to public catalog

V8/V10 DSMEM numbers should be struck from any published doc set.
Valid findings from V8/V9/V10 remain:
- cp.async.ca HBM 97%
- SMEM LDS 74% of peak
- Streaming BW curve
- Pipe latencies (all clock64-based, not DCE-prone)
- __syncthreads/fence costs (also clock64)
- Atomic latency / throughput
- DVS curve
- Register spill cost
- Bank conflict cost

## Confidence: HIGH

This verification is itself ncu-confirmed.