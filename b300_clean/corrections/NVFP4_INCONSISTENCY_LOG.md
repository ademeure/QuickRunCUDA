# NVFP4 Inconsistency Log

**Date: 2026-04-22.** Direct numerical conflicts found across the
NVFP4 documentation set. Each entry: claim A vs claim B, files involved,
proposed resolution, severity (CRITICAL / MAJOR / MINOR).

---

## CRITICAL — operand A vs B power dominance (direction reversed)

| File | Claim |
|------|-------|
| NVFP4_POWER_DECOMPOSITION.md (Hierarchy table) | "A→uniform saves −250 W vs B→uniform −64 W" → **A is 4× more impactful than B** |
| NVFP4_PURE_TCGEN05_RESULTS.md (Headline) | "B is dominant operand by 15-30× across ALL precisions" |
| NVFP4_K96_AB_FULL.md (Headline) | "B impact ≈ 2.6× A impact (averaged across configurations)" |
| CLAUDE memory note | "A:B impact ~1:3" (B dominant) |
| Memory note (separate) | "A is FREE" |

**Resolution (per PURE_TCGEN05's own correction section):**
- Pure tcgen05 multiplier: B >> A (15-30×) — universal across 6 precision variants
- cuBLAS NVFP4 path: A appears dominant because cuBLAS multicasts B (TMA), halving B's L2/memory pipeline cost
- K=96 single-kernel (NVFP4_K96_AB_FULL): B > A by 2.6× — closest to inference reality (no multicast advantage)

**The CLAUDE memory note "A:B impact ~1:3" matches the K=96 paper, NOT cuBLAS.** Different tests measure different things.

---

## CRITICAL — K-uniform-per-N savings (28% retracted)

| File | Claim |
|------|-------|
| NVFP4_SIGN_K64_K96.md (commit `4c1e60a`, PRIOR) | "K=64 K-uniform-per-N saves 124 W (-28%)" |
| NVFP4_SIGN_K64_K96.md (current) | "Saves only ~1-3% — earlier was contaminated baseline" |

**Resolution:** RETRACTED. The earlier claim came from background processes inflating the baseline by ~100 W. Real baseline ~343 W (not ~448 W). Documented in CLAUDE memory note `feedback_clock_stuck_no_lock`.

---

## MAJOR — cuBLAS NVFP4 ceiling

| File | Claim |
|------|-------|
| CLAUDE memory note `project_b300_nvfp4_k96_ceiling` | "cuBLAS 13.4 caps 10.8 PF (72% of 15 PF spec)" |
| NVFP4_CUBLAS_FULL_SWEEP.md | Boost peak: 11054 TF (73.7%) at M=N=8192 K=38400 |
| NVFP4_CUDAGRAPH.md | BPG=16 cudaGraph: **11423 TF (76.2%)** — exceeds the 73.1% model ceiling |
| NVFP4_CUTEDSL_THROTTLE_DEEP_DIVE.md ("TRUE peak" section) | 11068 TF (73.8%) at K=38400 |

**Resolution:** Memory note is stale. True cuBLAS peaks:
- Plain Lt: 11.07 PF (73.7-73.8%)
- With cudaGraph BPG=16: 11.42 PF (76.2%)
- 10.8 PF was a smaller-K shape (~16K). K=38400 is the optimum.

Memory note should be updated.

---

## MAJOR — CUTLASS C++ vs CuTeDSL ceiling claim

| File | Claim |
|------|-------|
| CLAUDE memory note `project_b300_nvfp4_k96_ceiling` | "CUTLASS C++/CuTeDSL stuck at 8.7 PF (58%)" |
| NVFP4_CUTEDSL_THROTTLE_DEEP_DIVE.md (boost zero-data) | CuTeDSL hits 9118 TF on 8K³ (60.8%); 8112 TF on 16K² K=15K |
| Same file (CUTLASS section) | CUTLASS 89 caps at 8285 TF (2SM, 8K² K=15K) |

**Resolution:** Memory note is approximately right but understated (8.7 PF vs measured 8.1-9.1 PF depending on shape). Update memory note to "CUTLASS C++/CuTeDSL stuck at ~8.3-9.1 PF (55-61%) depending on shape".

---

## MAJOR — Period sweep p=64 power numbers (1-CTA vs 2-CTA difference acknowledged)

| File | K=96 N=256 p=64 power |
|------|----------------------|
| NVFP4_PERIOD_SWEEP.md (1-CTA) | 534 W |
| NVFP4_PERIOD_2CTA.md (2-CTA) | 580 W |
| NVFP4_K96_AB_FULL.md (worst sign-period at 1005 MHz) | 605 W (with all-mag=1, p_n=64) |
| NVFP4_K96_AT_1500MHZ.md (1500 MHz) | 999 W |

**Resolution:** Different test geometries. NOT contradictions:
- 534 W is 1-CTA at K=96 N=256, mixed magnitudes period sweep
- 580 W is 2-CTA at same shape, same period sweep
- 605 W is the worst sign-only test (mag=+1 only, A=random) — strips out magnitude variation, isolating sign cost
- 999 W is the same 605 W test scaled to 1500 MHz (1.65× from voltage+freq super-linearity)

---

## MAJOR — Sign-bit cost contradiction

| File | Sign-bit cost (random vs all-positive) |
|------|---------------------------------------|
| NVFP4_K96_AB_FULL.md | "Sign bit alone costs ~80 W" (8-pos 472 W → 16-rand 552 W) |
| NVFP4_PURE_TCGEN05_RESULTS.md (NVFP4 sign-bit clean test) | "All-positive saves 64 W from random 470 W → 406 W" |
| NVFP4_K96_B_DISTRIBUTION.md | "Killing the sign bit saves 80 W" |
| NVFP4_SIGN_K64_K96.md | "K=64 best savings: -36 W (-10.5%); K=96 best: -56 W (-13.2%)" |

**Resolution:** NOT a contradiction once normalized:
- 80 W: K=96, M=N=256, 2-CTA, magnitude diversity randomized (compares 8-pos vs 16-rand)
- 64 W: K=64, M=N=128, 1-CTA, sign-only mode (other bits random)
- 36-56 W: K=64/K=96, M=N=128, 1-CTA, total constant-pattern savings
Different tile sizes, different cluster configs, different bit-isolation.

The K=96 2-CTA worst-vs-best swing is up to ~131 W (NVFP4_K96_SIGNMATCH.md p_n=32 vs p_n=64), ~80 W from sign alone (random vs structured).

---

## MAJOR — N-stride cliff: BF16 has it, NVFP4 doesn't

| File | Claim |
|------|-------|
| NVFP4_PURE_TCGEN05_RESULTS.md (BF16 N-stride sweep) | "Sharp cliff between N-stride-16 (592 W) and N-stride-32 (480 W) — 112 W drop in single step" |
| Same file (NVFP4 N-stride sweep) | "NVFP4 has NO 32-element MAC cliff!" — NVFP4 stride 32 = +3 W (essentially flat) |
| Same file (later note) | "Earlier NVFP4 stride results NOT trustworthy due to FP4 packing encoding bugs" |

**Resolution:** Acknowledged in source file. NVFP4 stride results are bug-prone; the cliff verdict (NVFP4 NO cliff vs BF16 YES cliff) is the most carefully measured (sign-bit-only) and is correct. But individual stride numbers should be treated as approximate.

---

## MINOR — int reduction speedup framing

| File | Claim |
|------|-------|
| NVFP4_INT_REDUCTION.md | "INT + redux.sync.add: 2.75× speedup vs SHFL chain" (cy/iter 60 vs 165) |
| NVFP4_FULL_PIPELINE.md | "Mode 7 (int redux per-tile) at 158 cy/tile = 1.59×; mode 2 (FP32 + final SHFL) at 85 cy/tile = 2.96× — beats mode 7" |

**Resolution:** Both correct but different scopes:
- 2.75× is the inner-loop reduction microbench (only the warp-reduce step)
- 2.96× vs 1.59× is the FULL pipeline (decode + accumulate + reduce); deferring reduction to end of kernel is the bigger win
- These are NOT in conflict; the int-reduction speedup matters only when reduce is structurally per-tile

Should be flagged in the consolidated doc to prevent users from picking redux.sync for end-to-end pipelines (where deferred SHFL wins).

---

## MINOR — Outlier "12% throughput loss" retracted

| File | Claim |
|------|-------|
| NVFP4_K96_AB_FULL.md (early version, embedded comment) | "1 outlier per K16 → 12% throughput loss" |
| Same file (3-trial verified addendum) | "Throughput loss is gradual: 1/K16 outlier costs ~2%, saturates at ~7%" |

**Resolution:** Old claim was a `--reuse-cubin` bug (cached wrong cubin = bench_excludes mode 0 = full random). Current addendum has correct numbers.

---

## MINOR — Catalog "10.8 PF" vs measured "11.07 PF / 11.42 PF"

Several memory notes still reference "10.8 PF cuBLAS catalog" which is now superseded. See MAJOR cuBLAS ceiling entry above.

---

## OPEN — Cluster scheduling ceiling

NVFP4_PARALLEL_STREAMS.md predicts NS=2 sweet spot from "cluster scheduler caps at ~15 max active clusters". CUTEDSL_THROTTLE_DEEP_DIVE confirms `Max Active Clusters = 15` from ncu. These agree. But:
- NVFP4_PARALLEL_STREAMS.md does NOT cross-test with cudaGraph
- NVFP4_CUDAGRAPH.md does NOT cross-test with parallel streams
- Predicted compounding (~+35% on small shapes) is unverified

---

## OPEN — K-id speedup applicability to NVFP4

CLAUDE memory note says "K-id speedup is shape-conditional ... cuBLAS 1.40× K-id BF16 speedup ONLY at N ∈ {K, 2K, K/2}". Whether NVFP4 K=96 ULTRA has the same restriction is NOT measured in any of the 22 NVFP4 files reviewed. Open for future investigation.

---

## OPEN — NVFP4 cache capacity per axis

PN_PK_SWEEP.md raises an explicit open question: random K-shift LOW for p_n ≤ 8 (8 unique patterns) but kphase_n shows 3 patterns = HIGH. Different cache behavior per axis (K-row vs N-column)? The "2-entry cache" model holds for one axis, "8-entry" for another. Mechanism unverified.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 2 (A/B dominance reversal, K-uniform retraction) |
| MAJOR | 5 |
| MINOR | 3 |
| OPEN | 3 |

Most "contradictions" are actually different test geometries; only 2 are true retractions (K-uniform 28%, outlier 12%). The single biggest source of confusion is the **operand A vs B dominance**, which has THREE different correct answers depending on which test you run.
