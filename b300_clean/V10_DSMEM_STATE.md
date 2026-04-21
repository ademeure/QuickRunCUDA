# V10: DSMEM measurement — honest state after deep investigation

## Summary

**DSMEM BW is NOT reliably measurable in this environment** with the
patterns I tried. Prior V8 37 TB/s claim is also suspect for the same
reasons.

## What I tried and what happened

| Attempt                                   | Result                 |
|-------------------------------------------|-------------------------|
| `ld.shared::cluster` inline offsets (static) | "Works" but DCE'd — SASS shows LD.E, ncu shows 7200 wavefronts total vs expected 5.9 billion |
| Varying base each iter                    | Segfault / unspecified launch failure |
| True dep chain (cur = ld; next_addr uses cur) | Segfault |
| Exact V8 bench_dsmem_definitive repro     | Segfault on sync        |
| V8 bench_v8_dsmem_bw.cu (still compiles)  | Returns old 37 TB/s but same DCE suspicion |

## Root cause of DCE

`ld.shared::cluster.u32 %r, [%base+imm]` with:
- base = invariant in the iteration loop
- imm = compile-time constant
- result → accumulator + XOR reduction at end

ptxas + nvcc are aggressively recognizing that:
1. The source SMEM is initialized once, doesn't change
2. Constant-offset loads from invariant base = loop-invariant result
3. Can hoist and CSE, eliminating almost all loads

SASS emits `LD.E` (global load) instead of `LDS` or `LDS.CTL`. ncu confirms
the loads aren't actually happening in the kernel.

## Root cause of segfault

Varying base + cluster DSMEM consistently crashes with "unspecified launch
failure". Replicated across:
- My V10 variants (smaller SMEM, different block sizes)
- Minimal 1-warp single-load test
- Direct reproduction of V8 `bench_dsmem_definitive`

Only `cluster=8 + static SMEM 2 KB + __launch_bounds__(128, 1)` works, but
the DCE issue makes measurements meaningless.

## What CAN be claimed with rigor

- **Cluster DSMEM path EXISTS and compiles** (PTX emits `ld.shared::cluster`)
- **Cluster launches themselves are free** (V10 `d13ef34`): 7.4 vs 7.7 µs
- **Cluster barriers work**: barrier.cluster = 370 cy (V8)
- **Cluster DSMEM can transmit data** (V8 K3, V8 K1): cross-process IPC atomics
  and host-visible mapping work

## What CANNOT be claimed

- ❌ **DSMEM BW peak** (V8 37 TB/s, V10 48-67 TB/s): likely DCE'd
- ❌ **DSMEM latency** (chained crashes; inline-offset would be 1 cy via CSE)
- ❌ **DSMEM vs local SMEM BW ratio**: both affected by same compiler issues

## Lesson (thanks to user intervention)

Rule 9: SASS + ncu verification is ESSENTIAL. My earlier V8 DSMEM measurement
probably had the same DCE issue — didn't catch it then.

The "47 TB/s" / "67 TB/s" / "37 TB/s" numbers previously claimed should all be
re-verified with ncu byte counts. I now have reason to believe they're all
bogus.

## Corrected claims

| Claim                  | Previous | Status after rigor |
|------------------------|----------|---------------------|
| V8 "37 TB/s DSMEM"     | HIGH conf| **SUSPECT (DCE possible)** |
| V10 "48 TB/s DSMEM"    | MED conf | **BOGUS (DCE confirmed)** |
| V10 "67 TB/s DSMEM ILP"| MED conf | **BOGUS**           |
| V10 "DSMEM writes 4× slower" | HIGH conf | **BOGUS** (same pattern)|
| V10 "cluster=2 fastest" | HIGH conf | **BOGUS** (data was fake)|

## Path forward

To truly measure DSMEM BW:
1. Use cuTLASS DSMEM helpers (they handle all the PTX quirks)
2. Try driver update (current env may have cluster DSMEM bugs)
3. Use TMA multicast (Blackwell's intended path for cluster data sharing)
4. Wall-clock test a REAL cuTLASS kernel with known DSMEM usage, subtract known-no-DSMEM equivalent

**Deferred to future work with proper cuTLASS tooling.**

## Confidence: HIGH on this investigation state
## Confidence: LOW on any prior DSMEM BW number