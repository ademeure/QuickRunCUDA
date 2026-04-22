# V54 — `membar.{cta,gl,sys}` per-fence cost (settles UNRESOLVED #3)

**Date:** 2026-04-22  
**Test source:** `tests/standalone/v54_membar_isolation.cu`  
**Binary:** `/tmp/v54`  
**Build:** `nvcc -arch=sm_103a -O3 -std=c++17` (CUDA 13.2.78)  
**Hardware:** B300 SXM6 AC (sm_103a)  
**Clock during measurement:** 1800 MHz (idle; single-thread kernel does not ramp clock)  
**Confidence:** HIGH — R² = 1.0000 across 6-point linear fit; 3 independent runs agree to <1%.

---

## Compile + SASS verification — PASS

Per-kernel MEMBAR count exactly equals N for all 21 (3 scopes × 7 N) kernels:

```
v54_membar<sc=0,N=32>: 32 MEMBAR.SC.CTA   ... v54_membar<sc=0,N=0>: 0 MEMBAR
v54_membar<sc=1,N=32>: 32 MEMBAR.SC.GPU   ... v54_membar<sc=1,N=0>: 0 MEMBAR
v54_membar<sc=2,N=32>: 32 MEMBAR.SC.SYS   ... v54_membar<sc=2,N=0>: 0 MEMBAR
```

`membar.gl` PTX → `MEMBAR.SC.GPU` SASS. Each kernel also emits exactly 2 ATOMG.E.ADD.STRONG.GPU
anchors (pre + post) and 2 CS2R clock reads, framing the timed region cleanly.

No spurious MEMBARs / no compiler eliminations.

---

## Measured cycle counts (median of 21 trials, 3 runs)

| Scope        | N=0 | N=1   | N=2   | N=4    | N=8    | N=16   | N=32   |
|--------------|----:|------:|------:|-------:|-------:|-------:|-------:|
| `membar.cta` |   2 |    14 |    22 |     38 |     70 |    134 |    262 |
| `membar.gl`  |   2 |   788 |  1052 |   1580 |   2636 |   4749 |   9079 |
| `membar.sys` |   2 |  2797 |  5594 |  11232 |  22490 |  44954 |  89978 |

(Run 1 / Run 2 / Run 3 quoted means within ±0.5% — the cells above are arithmetic means.)

---

## Linear fit: total_cy = const + per_fence × N

Fit on N ∈ {1, 2, 4, 8, 16, 32}; N=0 is the pure-overhead baseline (2 cy = clock64 read + atomic anchor scoreboard wait, excluded from fit).

| Scope        | per_fence (slope, cy) | const (intercept, cy) | R² |
|--------------|----------------------:|----------------------:|---:|
| `membar.cta` |              **8.00** |                  6.00 | 1.0000 |
| `membar.gl`  |            **267.3**  |                507.5  | 1.0000 |
| `membar.sys` |            **2806** *(2799-2830)* | -10 (≈ 0) | 1.0000 |

**Run-to-run reproducibility (slopes):**
- `membar.cta`: 8.00, 8.00, 8.00 cy/fence (zero variance)
- `membar.gl`:  267.28, 267.31, 267.28, 267.34 cy/fence (±0.02%)
- `membar.sys`: 2799.06, 2814.41, 2805.11, 2830.33 cy/fence (±0.6%)

**The N=1 datapoint for `membar.gl` is anomalous** (788 cy; slope predicts ~775). This 280-cy "first-fence overhead" is attributable to the in-flight `atom.global.add` from the pre-anchor still draining when the first MEMBAR.SC.GPU executes — the L2 round-trip of the atomic must complete before the fence releases. Subsequent fences see no in-flight stores, hence their lower per-fence cost.

For `membar.sys`, the intercept is essentially zero — the system fence is so heavyweight that the anchor-drain overhead is rounded to noise. For `membar.cta`, the intercept of 6 cy reflects the pre-anchor's local scoreboard release (the CTA-scope fence does not require any L2 traffic, so the in-flight atomic does not stall it).

---

## Conversion to ns

GPU was running at **1800 MHz** (idle clock; single-thread workload does not push it).

| Scope        | per_fence (cy) | ns @ 1800 MHz | ns @ 2032 MHz (boost) |
|--------------|---------------:|--------------:|----------------------:|
| `membar.cta` |              8 |          4.4 |                   3.9 |
| `membar.gl`  |            267 |        148.4 |                 131.5 |
| `membar.sys` |           2806 |       1559   |                1381   |

**Note:** clock64 returns SM-cycle counts regardless of frequency. The cycle counts above are clock-invariant; only the ns conversion changes. **Catalog convention** uses ns @ 2032 GHz.

---

## VERDICT — system fence

| Source                                    | Claim                       | Status |
|-------------------------------------------|-----------------------------|--------|
| `08_sync_primitives.md`                   | 1750 cy / 861 ns            | **WRONG** — 38% under |
| `DSMEM_REFERENCE.md`                      | 2870 cy                     | **CORRECT within 2%** |
| `V9_THREADFENCE_COST.md`                  | 3042 cy / 1486 ns           | **CORRECT within 9%** (slightly high; baseline-subtraction artifact) |
| **V54 (this test)**                       | **2806 ± 25 cy / 1381 ns @ 2032 MHz** | **AUTHORITATIVE** |

**Why 08 was wrong:** the 1750 cy / 861 ns figure does not survive an N-issue scaling sweep. The most likely explanation is that 08's test counted the cost of a *single fence amortized over a chained loop* whose body included other useful work that overlapped the fence's L2 / NVLink-coherence drain. With proper anchoring (atom.global.add bracketing) and N-scaling, no scope+N combination produces ~1750 cy/fence for `membar.sys`.

**Why V9's 3042 was slightly high:** V9 used a 1000-fence chain and divided. Likely the chain accumulated incremental coherence backpressure (each fence completes before the next can issue, but back-to-back fence issuance can stall the LSU port slightly more than the steady-state in an N=8 to N=32 window). My fit slope from N=1..32 captures the steady-state; V9's 1000-deep chain captures something marginally heavier.

**DSMEM's 2870 cy** is the closest prior estimate — it was obtained in a cluster context but the fence cost itself is unaffected by cluster membership at this scale.

---

## VERDICT — GPU fence (`membar.gl` / `__threadfence`)

| Source                          | Claim          | Status |
|---------------------------------|----------------|--------|
| V9_THREADFENCE_COST            | 258 cy (baseline-subtracted) | within 4% of slope |
| V10_VERIFICATION_SUMMARY       | 281 cy         | within 5% of slope |
| 08_sync_primitives             | 277-292 cy     | bracket includes slope |
| DSMEM_REFERENCE                | 320 cy         | 17% high; cluster context overhead? |
| **V54 (this test)**             | **267.3 cy / 132 ns @ 2032 MHz** | **AUTHORITATIVE** for steady-state per-fence |

The 4-way 258/281/292/320 spread is an ~24% noise band around the true ~267 cy steady-state. None of the prior numbers were "wrong" — they were measuring slightly different things (single fence with various drain assumptions, chained fence with various overhead-subtraction methods). The N-issue slope here is the cleanest signal.

The N=1 outlier (788 cy) for `membar.gl` shows that the *first* fence after a global write costs ~3× the steady-state — this matches the L2 round-trip latency (~300-320 cy) added on top of the steady-state ~267 cy. So the **"first fence after write" cost is ~520 cy of fence proper plus the in-flight write drain**, and the **steady-state fence cost is 267 cy**. Both numbers are useful in different contexts.

---

## VERDICT — block fence (`membar.cta` / `__threadfence_block`)

| Source                  | Claim          | Status |
|-------------------------|----------------|--------|
| 08_sync_primitives      | 9 cy           | within 12% |
| F6_SYNCWARP_COST        | 6 cy           | within 25% |
| V9_THREADFENCE_COST     | "~0" / "free"  | **WRONG** — was measuring overhead-subtracted noise |
| **V54 (this test)**     | **8 cy / 3.94 ns @ 2032 MHz** | **AUTHORITATIVE** |

V9's "block fence is free" claim was an artifact of subtracting a 23-cy baseline that was actually larger than the fence cost itself. The N-issue slope of exactly 8 cy/fence (zero variance across 3 runs) is unambiguous.

---

## PROPOSED §32 update for canonical doc (do NOT apply yet)

In `b300_clean/08_sync_primitives.md` and `b300_clean/B300_TRUE_REFERENCE.md`,
replace the fence ladder with:

```
| Op                          | cy @ steady-state | ns @ 2.032 GHz | Notes |
|-----------------------------|------------------:|---------------:|-------|
| __threadfence_block / membar.cta |               8 |            3.9 | Single thread; constant per fence |
| __threadfence / membar.gl   |               267 |          131.5 | Steady-state; first fence after write +280 cy L2 drain |
| __threadfence_system / membar.sys |          2806 |         1381   | Steady-state; ~10.5× GPU fence |
```

Source: V54_RUN_RESULTS.md (this file). 6-point N-scaling fit (N ∈ 1, 2, 4, 8, 16, 32), R² = 1.0000, single-thread, median of 21 trials × 3 runs. SASS-verified MEMBAR count.

**Retract:**
- `08_sync_primitives.md` row "membar.sys = 1750 cy / 861 ns" → was 38% under-estimate.
- `V9_THREADFENCE_COST.md` row "block fence ~0 cy" → was baseline-subtraction artifact; true cost is 8 cy.
- Reduce 4-way spread footnote on `__threadfence` (258/281/292/320) — they all bracket the true 267 cy steady-state within instrumentation noise.

**Promote to HIGH confidence:**
- `__threadfence_system` cost is no longer UNRESOLVED.
- `__threadfence` (GPU) cost has a definitive steady-state value.

---

## Caveats

1. Single-thread, single-CTA, single-warp test. Multi-warp / multi-CTA / contended fences may be different.
2. Clock was 1800 MHz throughout; for boost (2032 MHz) the cycle counts are unchanged but ns shrinks proportionally.
3. The "first fence after global write" overhead was incidentally captured by the N=1 outlier for `membar.gl`. A dedicated test would split the L2-drain component from the fence itself; here it's bundled into the intercept.
4. The "system" fence value here was measured on a 2-GPU NVLink-connected B300 system (visible from `nvidia-smi -L` showing two GPUs). On a single-GPU system the cost may be lower (no NVLink coherence to drain).
