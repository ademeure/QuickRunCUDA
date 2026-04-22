# tcgen05 / MMA Dedup — Inconsistency Log

**Auditor**: dedup swarm, 2026-04-22
**Files in scope**: 2CTA_DEDUP, A_VS_B_ASYMMETRY, A_B_ZERO_ASYMMETRY, BF16_SUBTILE_DEDUP, CROSS_MMA_DEDUP, CROSS_PRECISION_KVARY_FINAL, CROSS_PRECISION_SUBTILE, DIAGONAL_DEEP_DIVE, MMA_SHAPE_DEDUP, N_DEPENDENCE_DEEPDIVE, SPARSITY_3TIER, SUBTILE_DEDUP_MODEL, SUBTILE_HALVES, SUBTILE_PARTIAL_BREAK, SUBTILE_SPARSE_VALIDATION, TCGEN05_PATH_NOTES, DISABLE_LANE_POWER, MMA_FP8_KIND_F8F6F4_NOT_NATIVE.

For consolidated model + retraction list, see `TCGEN05_DEDUP_CONSOLIDATED.md`.

---

## A. Numerical inconsistencies between files

### A1. Cache depth: "1 slot" vs "2 slots" vs "4 slots"

| Source | Claim | Test method |
|--------|-------|-------------|
| `BF16_SUBTILE_DEDUP.md` lines 70-77 (early section) | "HW caches up to ~4 distinct sub-tile patterns for free" | K_break ≤ 3 free in N=128 |
| `SUBTILE_DEDUP_MODEL.md` lines 36-39 | "1 slot for FP8 / BF16; 2 slots for NVFP4" | Pattern-count rotation test, HW-distinct count |
| `N_DEPENDENCE_DEEPDIVE.md` lines 365-380 | "Dedup cache holds 2 unique sub-patterns max" | cuBLAS K-id period sweep at sustained throttle |
| `BF16_SUBTILE_DEDUP.md` lines 117-148 (later) | Recants the 4-slot claim → "STICKY ACTIVATION" model | Position permutation modes 3000-3007 |
| `MMA_SHAPE_DEDUP.md` lines 50-68 | "N=128 has 4-slot free zone, N=64 doesn't — invalidates simple 4-slot pattern cache hypothesis" | N=64 vs N=128 K_break sweep |

**Resolution**: 4-slot claim RETRACTED (R1 in consolidated). 1-slot vs 2-slot is genuinely unresolved (U1) — likely two different HW paths (per-cycle byte-compare vs alternation predictor) framed differently by the two test methodologies.

### A2. K-row dedup: "pairwise" vs "every 4 K-rows" vs "every 8 K-rows"

| Source | Claim |
|--------|-------|
| Memory entry "tcgen05.mma power model" | "K-row pairwise dedup" |
| `N_DEPENDENCE_DEEPDIVE.md` chunk-size curve | period 1 ✓, period 2 ✓, period 3 ✗ (matches "pairwise") |
| `CROSS_PRECISION_KVARY_FINAL.md` | K-vary cost scales sub-linearly with K (BF16 100%, FP8 76%, NVFP4 53%) — not a binary dedup, a smooth cost |

**Resolution**: "Pairwise" = up to 2-pattern alternation works, but the underlying HW also has a "K-tile-wide constancy" path (chunks ≥ 8 also work). No file claims "every 4" or "every 8 K-rows" specifically; if such a claim existed elsewhere it would be wrong. The single-MMA K-vary tests show smooth sub-linear scaling, NOT binary "pairs free above which costs full."

### A3. A K-vary cost: 0 W vs 60 W

| Source | Claim |
|--------|-------|
| `A_VS_B_ASYMMETRY.md` lines 57-58 | "A K-vary cost ≤ +5W vs constant. … No K-cliff for A either." |
| `A_B_ZERO_ASYMMETRY.md` lines 50-52 | "A varying matters ~5x less but only when B also varies: 60W marginal" |
| `CROSS_PRECISION_KVARY_FINAL.md` row table | A K-vary 16 cost: BF16 +2 W, FP8 ≈0 W, NVFP4 +7 W |

**Resolution**: Not contradictory once composed (refined model in consolidated § 1, § 2). A K-vary ≈ 0 when B is constant (`A_VS_B_ASYMMETRY` and `CROSS_PRECISION_KVARY_FINAL` both use B const). When B is also random, A K-vary adds ~60 W marginal (`A_B_ZERO_ASYMMETRY` mode 200 vs 1700). The "≤ 5 W" and "60 W" both correct under different conditions.

### A4. NVFP4 subtile cliff: at N=64 or at N=65?

| Source | Claim |
|--------|-------|
| `BF16_SUBTILE_DEDUP.md` cross-precision summary | NVFP4 cliff "at N_unique=64-128" (loosely) |
| `CROSS_PRECISION_SUBTILE.md` row "NVFP4" | Cliff at 64 → 65 (281 W → 463 W) |
| `SUBTILE_DEDUP_MODEL.md` summary | "NVFP4: HW sub-tile = 64 N (= 32 bytes)" |

**Resolution**: All consistent at 65 = first cliff. The "64-128" loose framing in `BF16_SUBTILE_DEDUP` was before the fine-grain N_unique=65 measurement. Use 65 as the canonical cliff.

### A5. Random-data baseline by precision

| Source | BF16 random (W) | FP8 random (W) | NVFP4 random (W) |
|--------|----------------:|---------------:|-----------------:|
| `CROSS_PRECISION_KVARY_FINAL.md` | 599 | 630 | 473 |
| `CROSS_PRECISION_SUBTILE.md` | 609 | 642 | 463 |
| `BF16_SUBTILE_DEDUP.md` | 605-623 (range over modes) | — | — |
| `A_VS_B_ASYMMETRY.md` | 609 | — | — |
| `MMA_SHAPE_DEDUP.md` | 609 (N=128) | — | — |

**Resolution**: All within 10-20 W noise. Use **610 / 640 / 470 W** as canonical (BF16 / FP8 / NVFP4 random per CTA at -lgc 1005 MHz).

### A6. Sub-tile dedup contribution: dominant or negligible?

| Source | Claim | Test |
|--------|-------|------|
| Memory ("tcgen05.mma power model") + most per-MMA files | Sub-tile dedup is the dominant B-side power lever (~250-310 W save) | Single-MMA pattern tests |
| `N_DEPENDENCE_DEEPDIVE.md` lines 924-942 | "Sub-tile dedup contributes ~1-3% in cuBLAS; K-row dedup gives 42%" | cuBLAS workloads at N=K=8192 |

**Resolution**: NOT a contradiction once context is matched. Sub-tile dedup IS the dominant per-MMA mechanism for power; in CUBLAS WORKLOADS the K-row dedup amortizes far better across K-tile iteration so it dominates THROUGHPUT speedup. Both are simultaneously true.

This distinction isn't always explicit in the source files — readers could reasonably extract opposite conclusions about which mechanism "matters." The consolidated doc clarifies (U5).

### A7. cuBLAS / random-data peak BF16

| Source | TFLOPS at N=K=8192 random |
|--------|--------------------------:|
| `N_DEPENDENCE_DEEPDIVE.md` random baseline | 1480 |
| `B300_TRUE_REFERENCE.md` (per TENSOR_INCONSISTENCY_LOG) | 1883 |
| `06_tensor_cores.md` cuBLAS BF16 8K³ random | 1905 |

**Resolution**: Not a dedup-doc problem per se, but worth flagging — `N_DEPENDENCE_DEEPDIVE` random baseline (1480) is significantly LOWER than the cuBLAS sustained number from TRUE_REFERENCE (~1880). Likely because `N_DEPENDENCE_DEEPDIVE` was measured during the same session that hit the "stuck at 1005 MHz" issue. The N-dependence RATIOS (K-id 1.42× over random) are still valid; the absolute random number is suppressed. Already flagged in `N_DEPENDENCE_DEEPDIVE.md` itself ("clock state contamination").

---

## B. Logical / mechanism inconsistencies

### B1. "N-vary is FREE" vs "32-byte sub-tile cliff"

| Source | Claim |
|--------|-------|
| Older measurements (pre-2026-04-19) | "N-vary direction is FREE — only K-vary costs" |
| `BF16_SUBTILE_DEDUP.md` lines 7-12 | "N-vary is FREE was a LOW-ENTROPY ARTIFACT. With HIGH-ENTROPY values, sharp cliff at N_unique=17." |
| `CROSS_PRECISION_KVARY_FINAL.md` row | "N-direction variation is FREE — universal" |

**Resolution**: The CROSS_PRECISION_KVARY_FINAL "N-vary free" is correct ONLY because their N test stayed below the 32-byte cliff (≤16 distinct values per K row at BF16). For N_unique > cliff, N-vary is decidedly NOT free. The framing "N-vary is FREE" is misleading without the qualifier "below the 32-byte sub-tile boundary."

The retraction is documented in `BF16_SUBTILE_DEDUP.md` itself (R3 in consolidated).

### B2. "Diagonal patterns stay LOW universally" vs popcount-conditional

| Source | Claim |
|--------|-------|
| Earlier diagonal measurements (referenced in `DIAGONAL_DEEP_DIVE.md`) | "Diagonal patterns at p_n=8 stay LOW despite 16 unique sub-tile patterns - cache must hold more than 2 patterns" |
| `DIAGONAL_DEEP_DIVE.md` (devil's-advocate) | "p_n=8 diagonal LOW, p_n=16 diagonal HIGH. The diagonal mechanism depends on whether sub-tile popcount stays invariant across K rows." |

**Resolution**: Earlier "diagonal LOW" framing is too general. The mechanism is **popcount invariance**, not "diagonality" per se. Retraction R5 in consolidated.

### B3. "2:4 sparsity hurts dense GEMM" vs "+11% on dense"

| Source | Claim |
|--------|-------|
| `N_DEPENDENCE_DEEPDIVE.md` line 633 (early) | "popular 2:4 structured sparsity (50% zeros) actually hurts dense GEMM throughput on B300" |
| `N_DEPENDENCE_DEEPDIVE.md` line 659+ (correction) | "Earlier claim was WRONG. STRUCTURED 2:4 patterns give 1.11× speedup on dense GEMM." |
| Memory entry "Units sanity" | "Got '28× ratio' wrong by mixing combined+uncombined" — different example but same self-correction pattern |

**Resolution**: File self-corrects in same document. Retraction R4 in consolidated. **Crucially, the difference is FIXED vs RANDOM zero positions**: random 50% sparse hurts (5% slowdown), fixed-position 2:4 helps (+11%). Some downstream readers might pick up only the early "hurts" framing.

### B4. Pattern-count anomaly: 2-pattern WORSE than 3-pattern

| Source | Observation |
|--------|-------------|
| `CROSS_PRECISION_SUBTILE.md` modes 3101-3108 | 2-pattern (ABAB) = 623 W; 3-pattern (ABCABC) = 538 W; 4-pattern = 610 W |
| Predicted by simple slot model | 2-pattern should be FREE (fits in 2 slots), 4-pattern should be expensive |

**Resolution**: Genuinely UNRESOLVED (U3). The sticky-activation model fits 4 of 8 cases. The HW likely has multiple detection paths interacting. Two-half processing partially explains BF16 cases but not all.

### B5. cuBLAS "2 LRU slots" vs single-MMA "1-slot or sticky"

| Source | Claim about cache structure |
|--------|----------------------------|
| `SUBTILE_DEDUP_MODEL.md` | "1 slot for FP8 / BF16; 2 slots for NVFP4" (single-MMA) |
| `N_DEPENDENCE_DEEPDIVE.md` | "Dedup cache holds 2 unique sub-patterns max" + "NOT simple LRU" (sustained cuBLAS) |
| `N_DEPENDENCE_DEEPDIVE.md` chunk-size curve | "Two HW paths: alternation predictor (chunk=1) AND per-K-tile constancy (chunk divides 64)" |

**Resolution**: UNRESOLVED (U1, U4). The "2 slots" claim from cuBLAS K-id is at sustained throttle which engages additional HW (alternation predictor + clock throttle response); the "1 slot" claim from per-MMA pattern rotation is in a more isolated test. The HW likely has 1 byte-compare slot AND a separate alternation predictor — but no test conclusively isolates them.

---

## C. Counter-metric / test-methodology issues

### C1. NCU pipe_tensor for tcgen05.mma — known unreliable
- Memory entry "B300 benchmarking pitfalls": "ncu pipe_tensor doesn't measure tcgen05"
- `SUBTILE_DEDUP_MODEL.md` line 109 suggests "Cross-check with NCU `tpc__l1tex_*` or `sm__pipe_tensor_*` counters" — already flagged in `TENSOR_INCONSISTENCY_LOG.md` B2.
- Most dedup files use `clock64`-based cy/MMA + NVML power, which are appropriate for tcgen05.
- Use `smsp__inst_executed_pipe_tensor_subpipe_hmma_op_utchmma_utcqmma_utcomma_scope_2cta.sum` for tcgen05 — most dedup files don't reference any ncu counter so are not affected.

### C2. Stuck clock contamination in `N_DEPENDENCE_DEEPDIVE.md`
- File documents that initial measurements were at stuck-1005-MHz (gave ~1190 TF universally with no ratio visible).
- After `nvidia-smi -rgc -i 0`, measurements show clean 1.42× ratio.
- Operational lesson: ALL dedup measurements at -lgc 1005 should be cross-checked that clock is actually at 1005 (not lower) via in-run sampling.
- Per memory entry "Clock stuck without lock", this is a recurring B300 issue.

### C3. Per-MMA vs sustained measurement equivalence
- Per-MMA tests use 50M iters at -lgc 1005 MHz; report power per CTA.
- cuBLAS tests use sustained workloads at boost or locked clocks; report wall-clock TFLOPS.
- These two methodologies probe DIFFERENT aspects of the same HW — per-MMA tests show the static dedup behavior; cuBLAS tests include the dynamic clock-throttle response.
- Several apparent contradictions (A6, B5) reduce to "different methodologies probing different facets."

---

## D. Coverage gaps (not necessarily inconsistencies, but worth noting)

### D1. A-major MMA layouts
- `A_VS_B_ASYMMETRY.md` confidence: "LOW on whether this transfers to A-major MMA layouts (untested)."
- All measurements are with B as the distributed operand. tcgen05.mma kind variants with different operand orientations not tested.

### D2. cluster_group::2 (2-CTA) with structured B
- `2CTA_DEDUP.md` open follow-up: "Test K-rotating + sub-tile patterns in 2-CTA mode to verify save-recipes work"
- Single-CTA recipes should apply per-CTA in cluster mode (no shared dedup), but not directly verified for structured B.

### D3. mma.sync kind::f8f6f4 dedup
- Not tested whether the F2FP.UNPACK + HMMA path on mma.sync exhibits any sub-tile dedup. Likely not (different pipeline) but unverified.
- All dedup characterizations in this corpus are tcgen05.mma; mma.sync would need its own characterization.

### D4. NVFP4 with non-uniform scale-factor (SF) entropy
- `SUBTILE_SPARSE_VALIDATION.md` mentions "NVFP4: SF entropy adds ~27W independently"
- The scale-factor pathway has its own data dependence not fully characterized.
- See also `NVFP4_K96_AB_FULL.md` (not in scope here) for SF behavior.

---

## E. Summary table of file-by-file status

| File | Stand-alone correct | Needs cross-link to | Self-corrects |
|------|:-------------------:|---------------------|:-------------:|
| 2CTA_DEDUP | YES | A_VS_B_ASYMMETRY | — |
| A_VS_B_ASYMMETRY | YES | A_B_ZERO_ASYMMETRY (refines) | — |
| A_B_ZERO_ASYMMETRY | YES (refinement) | A_VS_B_ASYMMETRY (parent) | — |
| BF16_SUBTILE_DEDUP | YES | MMA_SHAPE_DEDUP, SUBTILE_DEDUP_MODEL | YES (R3) |
| CROSS_MMA_DEDUP | YES | — | — |
| CROSS_PRECISION_KVARY_FINAL | YES | qualify "N-free" claim with sub-tile context | — |
| CROSS_PRECISION_SUBTILE | YES | SUBTILE_DEDUP_MODEL | partial (notes anomaly) |
| DIAGONAL_DEEP_DIVE | YES | — | YES (popcount revision) |
| MMA_SHAPE_DEDUP | YES | SUBTILE_HALVES (explanation for N=128 free zone) | — |
| N_DEPENDENCE_DEEPDIVE | YES | — | YES (R4 in same file) |
| SPARSITY_3TIER | YES (memory-side only) | — | — |
| SUBTILE_DEDUP_MODEL | YES | MMA_SHAPE_DEDUP (RETRACTS its 4-slot framing) | YES (revises BF16_SUBTILE_DEDUP) |
| SUBTILE_HALVES | YES | SUBTILE_SPARSE_VALIDATION (cross-check) | — |
| SUBTILE_PARTIAL_BREAK | YES | — | — |
| SUBTILE_SPARSE_VALIDATION | YES | SUBTILE_HALVES | — |
| TCGEN05_PATH_NOTES | YES (PTX/SASS notes, not dedup) | — | — |
| DISABLE_LANE_POWER | YES | — | — |
| MMA_FP8_KIND_F8F6F4_NOT_NATIVE | YES | any FP8 mma.sync claim elsewhere | — |
