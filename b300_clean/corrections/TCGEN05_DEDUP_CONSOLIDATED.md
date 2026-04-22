# tcgen05 / MMA Dedup, Subtile, Sparsity, Asymmetry — CONSOLIDATED

**Auditor**: dedup swarm, 2026-04-22
**Authority order**: B300_TRUE_REFERENCE → cross-checked measurements (multiple files agreeing) → newer files supersede older when explicit "REFINED" / "CORRECTION" notes present.
**Scope**: tcgen05.mma B-side power model, sub-tile granularity, K-row dedup, A-vs-B asymmetry, sparsity tiers, MMA shape effects, mma.sync FP8 path. Originals NOT modified.

This doc is the unified model. Per-file inconsistencies are catalogued in `DEDUP_INCONSISTENCY_LOG.md`.

---

## 1. The unified power / dedup model

```
P(MMA) = P_baseline                                    // ~280-305 W per CTA precision-dep
       + Σ over HW sub-tiles (B-side, 32-byte each):   // sub-tile dedup
            0                                  if byte-identical to active cache slot
            ~32 W activation + ~18 W per broken byte   otherwise (BF16 m128n128)
       + Σ over K iterations (B K-vary cost):          // K-row pairwise dedup
            5-25 W per added unique K row pattern      // sub-linear in K count
       + ε(A varying) only when B varies               // A is broadcast → ~0-60 W marginal
       + sparsity / disable_lane terms                 // see § 6
```

**A is FREE in the broadcast sense** (no per-bit penalty when B is constant). When B is also random, A varying adds ~60 W marginally. A K-vary at random B adds ≤ 7 W in all 3 precisions tested.

---

## 2. A vs B operand asymmetry — DEFINITIVE

Synthesis of `A_VS_B_ASYMMETRY.md` + `A_B_ZERO_ASYMMETRY.md` (consistent, the second refines the first):

| Configuration | Power (W) | Δ vs Tier B (299 W) |
|---------------|----------:|--------------------:|
| const A + const B | 299 | 0 |
| FULL random A + const B | 297-302 | ~0 (FREE) |
| const A + random B | 549 | +250 |
| random A + random B | 609-611 | +310 |
| zero A + random B | 490 | +191 (A=0 saves only 60 W when B varies) |
| random A + zero B | 298 | +0 (B=0 fully gates multiplier, -313 W save) |

**Mechanism**: A operand is broadcast through fanout (one value drives ~32 N MACs); B is distributed (each of ~32 N MACs holds its own per-cycle value). The 32-byte sub-tile dedup cache is a **B-only** mechanism.

A vs B reconciled: `A_VS_B_ASYMMETRY` originally said "A varying alone is FREE." `A_B_ZERO_ASYMMETRY` refines this: A varying is FREE *when B is constant*; when B varies, A varying adds ~60 W marginal. Not contradictory — they compose into the model in § 1.

---

## 3. Sub-tile dedup — UNIVERSAL 32-BYTE BOUNDARY

Synthesis across `BF16_SUBTILE_DEDUP.md`, `CROSS_PRECISION_SUBTILE.md`, `SUBTILE_DEDUP_MODEL.md`, `MMA_SHAPE_DEDUP.md`, `SUBTILE_PARTIAL_BREAK.md`:

| Precision | N values per HW sub-tile | Bytes per sub-tile | Cliff at N_unique |
|-----------|--------------------------|---------------------|-------------------|
| BF16      | 16                       | 16 × 2 = **32**     | 16 → 17           |
| FP8 e4m3  | 32                       | 32 × 1 = **32**     | 32 → 33           |
| NVFP4     | 64                       | 64 × 0.5 = **32**   | 64 → 65           |

**32 bytes is the universal HW B-side sub-tile granularity.** Confirmed by N_unique cliff search at all 3 precisions; cliff lands at exactly 17 / 33 / 65 unique values.

**Independent of SMEM descriptor LBO** (LBO=16 vs LBO=32 give identical power — see `MMA_SHAPE_DEDUP.md`). The dedup operates on logical N-values, not on physical SMEM access.

**Independent of MMA_N shape** (N=64 and N=128 share the 32-byte cliff position; see `MMA_SHAPE_DEDUP.md`).

### Partial-break linear scaling (BF16 m128n128, `SUBTILE_PARTIAL_BREAK.md`)
- 1 byte broken in a 32-byte sub-tile: +32 W (activation cost)
- Each additional broken byte: +18 W
- Full sub-tile broken (16 bytes): +306 W (matches full random)

### Cache-depth model (REFINED multiple times)

Earlier "4-slot pattern cache" claims are RETRACTED — the apparent free zone in BF16 K_break ≤ 3 is **STICKY ACTIVATION**, not slot caching. Confirmed by:
- N=64 has NO free zone (`MMA_SHAPE_DEDUP.md`)
- 2-pattern alternation costs MORE than 3-pattern rotation (`CROSS_PRECISION_SUBTILE.md` modes 3101-3108)
- Two-half processing (§ 4) gives the apparent "K_break ≤ 3 free" in N=128 BF16

The right model is **STICKY ACTIVATION + TWO-HALF PROCESSING (BF16 only)**:
1. B port starts in low-power gated state.
2. First non-matching sub-tile activates the port; it stays active.
3. (BF16 m128n128 only) Half A (sub-tiles 0-3) and Half B (sub-tiles 4-7) have INDEPENDENT activation state.

For cuBLAS K-id measurements specifically (`N_DEPENDENCE_DEEPDIVE`), the dedup cache appears to behave as **2 LRU slots per sub-tile position** with shape-modulated effective depth. This is a *different framing* of the same HW than the per-MMA single-shot tests; see § 8 Unresolved.

---

## 4. BF16 two-half processing (BF16 m128n128 only)

From `SUBTILE_HALVES.md` + `SUBTILE_SPARSE_VALIDATION.md`:

- **Half A** (N=0..63, sub-tiles 0-3): single unique sub-tile here costs +126 to +169 W
- **Half B** (N=64..127, sub-tiles 4-7): single unique sub-tile here costs +4 to +6 W (FREE)
- Boundary cliff is sharp: position 3 (+166 W) → position 4 (+49 W) → position 5 (+4 W)
- **No cross-half dedup**: matching content in Half B against Half A still costs the Half A activation (mirror test, modes 6700-6704)

**Cross-precision check**: FP8 and NVFP4 do NOT exhibit the two-half asymmetry. Position-uniform within ±10 W. So **two-half is BF16 m128n128k16 specific**.

Optimization recipe: pack the most-repetitive B columns at LOW N (Half A); arbitrary high-entropy data is essentially free at HIGH N (Half B). Saves up to 269 W vs mirrored layout.

---

## 5. K-row / K-vary dedup

From `CROSS_PRECISION_KVARY_FINAL.md` + `N_DEPENDENCE_DEEPDIVE.md`:

| Precision | K | B K-vary 16 cost (W) | A K-vary 16 cost (W) | Ratio B/A |
|-----------|--:|---------------------:|---------------------:|----------:|
| BF16      | 16 | +47 | +2 | 24× |
| FP8 e4m3  | 32 | +71 | ≈0 | >70× |
| NVFP4     | 64 | +99 | +7 | 12× |

K-cost scales **sub-linearly** with K (76% of linear at FP8, 53% at NVFP4) — narrower precisions process more K positions per cycle.

### "Pairwise" vs longer-period dedup

Memory says "K-row pairwise dedup". The actual finding (`N_DEPENDENCE_DEEPDIVE.md`):
- **Period 1 (K-row identical)** → full ~1.42× speedup
- **Period 2 (ABAB chunk=1)** → full ~1.42× speedup (alternation predictor, content-agnostic)
- **Period ≥ 3** → essentially no speedup

So "pairwise" = supports up to 2-pattern alternation, not "every-2-rows". Memory wording is slightly misleading but directionally correct.

### Chunk-size non-monotonic curve at N=K=8192

| chunk | TFLOPS | Speedup | Note |
|------:|-------:|--------:|------|
| 1 | 2102 | 1.42× | alternation predictor |
| 2 | 1525 | 1.03× | worst case |
| 4 | 1728 | 1.16× | |
| 8 | 2019 | 1.36× | |
| 16/32/64 | 2051-2079 | 1.39-1.40× | divides K-tile size 64 |
| 128 | 1919 | 1.30× | exceeds K-tile |

Two HW paths: alternation predictor (chunk=1 only) AND per-K-tile constancy (chunk divides 64). Not pairwise LRU.

### Shape-conditional speedup (cuBLAS-only mechanism)

K-id speedup ONLY triggers at N ∈ {K/2, K, 2K} AND N divisible by 256 AND transB=0 AND data has period-1 or period-2 K structure. All five conditions required. Real ML inference satisfies essentially none → ~2-6% practical benefit.

---

## 6. Sparsity 3-tier model

Three INDEPENDENT sparsity-related mechanisms identified across the corpus:

### Tier 1: Memory-side popcount sparsity (`SPARSITY_3TIER.md`)
**Wire/SerDes toggle energy**, not multiplier-side.
- Smooth monotonic decay of read power with sparsity at L1/L2/DRAM tiers.
- 0% (random base) ≈ baseline; 100% (all-zero) = max savings.
- Tier amplification: L1 swing 37 W, L2 swing 181 W, DRAM swing 245 W.
- Knee at sp ≈ 10-15% ; needs ≥ 50% sparsity for material savings.
- Granularity barely matters (byte/dword/32B/128B within ±15-20 W).
- Value asymmetry at sp=100%: zero < alt55 < one (HBM PHY active-low termination).

### Tier 2: tcgen05 multiplier zero-shortcut (`N_DEPENDENCE_DEEPDIVE` U-curve)
- 0% (dense random): 1480 TF baseline
- **30-50% RANDOM sparse: DIP, 1410-1440 TF (5% SLOWER than dense!)** — pattern-detector thrashing increases power.
- 75% sparse: 1565 TF
- 90% sparse: 1730 TF (zero shortcuts dominate)
- 99% sparse: 1930 TF
- 100% (all zero): 2253 TF (universal entropy detector, multiplier fully gated)

### Tier 3: Structured 2:4 sparsity (predictable-position pattern detector)
- FIXED 2:4 zeros at positions {0,1}, {0,2}, {1,3}, etc.: **1.11× speedup on dense GEMM** (no sparse API needed)
- 1:2 alternating: also 1.11×
- RANDOM 2:4 (random which 2 of 4 are zero): NO speedup
- Rotating 2:4: slight regression
- **Requires CONSISTENT zero positions per 4-element group** — that's why NVIDIA's 2:4 spec mandates fixed positions.
- Stacks with FP8: FP8 + 2:4 structured = 3033 TF (vs 2683 random) = +14% (NVFP4-class scaling).

**Key correction noted in source files**: An EARLIER claim that "2:4 hurts dense GEMM" was based on RANDOM 50% sparsity; STRUCTURED 2:4 actually gives +11%. The source file (`N_DEPENDENCE_DEEPDIVE.md`) self-corrects this in line 659+.

---

## 7. MMA shape dedup (`MMA_SHAPE_DEDUP.md`)

| Config | N=64 (W) | N=128 (W) |
|--------|---------:|----------:|
| Random | 466 | 609 |
| Const  | 266 | 299 |
| Cliff at N_unique=17 | yes (32-byte) | yes (32-byte) |
| Free zone for K_break ≤ 3 | NO | YES (sticky activation) |

Cliff at 32 bytes is universal across MMA_N. Per-MAC random penalty constant at ~75 µW/MAC. Sticky-activation free-zone is N=128-specific (related to two-half processing — see § 4).

LBO=16 vs LBO=32 SMEM descriptor: identical behavior. Dedup is HW-intrinsic, not SMEM-pattern-driven.

---

## 8. 2-CTA cluster dedup (`2CTA_DEDUP.md`)

cluster_group::2 (2-CTA mma) shows **identical per-cluster power dependence** to single-CTA mode. **NO cluster-shared dedup pooling**. Each CTA's B operand has its own 32-byte sub-tile dedup cache. Optimization recipes apply per CTA, not cluster-wide.

---

## 9. Cross-MMA dedup state (`CROSS_MMA_DEDUP.md`)

Dedup state is **per-MMA, NOT cross-MMA**. Alternating different B descriptors gives the AVERAGE of per-MMA powers, not a penalty or carry-over benefit. This means each individual MMA pays its own data-dep cost, validating that real cuBLAS GEMMs (which iterate K-tiles) get the optimization recipes per K-tile.

---

## 10. disable_lane (`DISABLE_LANE_POWER.md`)

Selective output column gating. Linear ~2.4 W per disabled column on BF16 m128n128. Cycle count unchanged. Composes with sub-tile dedup (lower marginal saves when dedup already active). Best-case combination: 254 W (vs 610 W random) = -58% reduction.

---

## 11. Diagonal patterns (`DIAGONAL_DEEP_DIVE.md`)

Devil's-advocate revision of "diagonal patterns stay LOW" claim. Mechanism actually appears to be **sub-tile popcount invariance**: when each sub-tile (16 N values) has identical bit-count across all K rows, power stays low; when popcount varies, it goes high. Diagonal works at p_n=8 (all popcount=8) but NOT at p_n=16 (popcounts 0..15 vary). The popcount hypothesis explains the data; pure "diagonal" framing was over-general.

---

## 12. mma.sync FP8 kind::f8f6f4 — NOT NATIVE (`MMA_FP8_KIND_F8F6F4_NOT_NATIVE.md`)

**Critical distinction**: `mma.sync.aligned.m16n8k32.kind::f8f6f4` on sm_103a is **NOT a native FP8 instruction**. ptxas lowers it to:
- F2FP.F16.E4M3.UNPACK_B (12+ unpack instructions per K=32 worth)
- 2× HMMA.16816.F32 (standard FP16 m16n8k16)

Measured: FP8 mma.sync = 104 TFLOPS effective vs 2×BF16 mma.sync = 143 TFLOPS = **1.37× SLOWER**.

**Native FP8 throughput on B300 is ONLY available via tcgen05.mma**.

Anywhere a file casually says "FP8 mma.sync = 276 TFLOPS effective" or treats kind::f8f6f4 as native FP8 dedup behavior, the framing is misleading. The dedup numbers in this doc are all **tcgen05.mma kind::f8f6f4** (which IS the native FP8 path on B300, despite the same syntactic kind name as the legacy mma.sync emulation).

---

## RETRACTIONS

### R1. "4-slot HW pattern cache" — RETRACTED
- **Original claim** (BF16_SUBTILE_DEDUP, SUBTILE_DEDUP_MODEL early sections): HW caches up to ~4 distinct sub-tile patterns for free.
- **Retraction source**: `MMA_SHAPE_DEDUP.md` (N=64 has no free zone; would not happen if 4-slot cache existed). `CROSS_PRECISION_SUBTILE.md` modes 3101-3108 (2-pattern rotation costs MORE than 3-pattern; impossible under simple slot model).
- **True mechanism**: STICKY ACTIVATION + TWO-HALF PROCESSING (BF16-only). The "K_break ≤ 3 free" zone in N=128 is because the unique sub-tiles fall in Half B, which is independently gated.

### R2. "K-row dedup is the dominant cost" (early model) — RETRACTED in favor of refined model
- **Original claim** (BF16_SUBTILE_DEDUP early section): K-vary cost was the primary mechanism.
- **Retraction source**: `BF16_SUBTILE_DEDUP.md` itself (lines 81-91): "True picture: power cost is driven by per-cycle SUB-TILE PATTERN DIVERSITY, where… K-vary's cost likely came from BREAKING THE PER-CYCLE DEDUP across K iterations within a sub-tile."
- **True picture**: B sub-tile mismatch (32-byte) is the dominant per-MMA cost; K-vary contributes secondarily via the same sub-tile-mismatch mechanism applied across K iterations.

### R3. "N-vary is FREE" — RETRACTED (was a low-entropy artifact)
- **Original claim** (pre-2026-04-19 measurements with low-entropy val tables): N-direction variation costs nothing.
- **Retraction source**: `BF16_SUBTILE_DEDUP.md` lines 7-12 (low-entropy val tables only have ≤16 distinct entries, hiding the cliff at N_unique=17).
- **True picture**: N-vary is FREE up to 32 bytes (sub-tile granularity), then hits a cliff. Whether N-vary "costs" depends entirely on whether you stay below or above the 32-byte cliff.

### R4. "2:4 sparsity hurts dense GEMM" — RETRACTED
- **Original claim** (`N_DEPENDENCE_DEEPDIVE` early section, around line 633): "popular 2:4 structured sparsity (50% zeros) actually hurts dense GEMM throughput."
- **Retraction source**: `N_DEPENDENCE_DEEPDIVE.md` lines 659-700: "Earlier claim '2:4 sparsity hurts dense GEMM' was WRONG. The error was from using random sparsity instead of structured."
- **True picture**: STRUCTURED 2:4 (fixed zero positions) gives +11% on dense GEMM. RANDOM 50% sparse (random which 2 of 4 are zero) gives the dip / regression. Memory entry "structured sparsity = real, random = U-curve" is correct.

### R5. "Diagonal patterns stay LOW universally" — RETRACTED in favor of popcount-invariance
- See § 11. Diagonal stays LOW only when sub-tile popcount is invariant across K rows (e.g., p_n=8 shift=k yields all popcounts=8). At p_n=16 the popcount varies and diagonal is HIGH. The popcount-invariance hypothesis better fits the data.

### R6. "FP8 mma.sync = 276 TFLOPS native" — RETRACTED
- See § 12. kind::f8f6f4 in mma.sync compiles to F2FP.UNPACK + HMMA, not native FP8. Native FP8 only via tcgen05.mma.

---

## UNRESOLVED

### U1. Cache depth: "1 slot" (single-MMA) vs "2 slots" (cuBLAS K-id) reconciliation
- `SUBTILE_DEDUP_MODEL.md` concludes "1-slot cache for BF16/FP8, 2-slot equivalent for NVFP4" (single-MMA pattern-rotation tests).
- `N_DEPENDENCE_DEEPDIVE.md` concludes "Dedup cache holds 2 unique sub-patterns max" (cuBLAS sustained K-id period-2 tests).
- Possible resolutions: (a) different framings of the same HW (single-MMA pattern detection vs sustained K-row alternation predictor are different paths); (b) 1-slot LRU per cycle, 2-slot effective via the alternation predictor over K. Not directly tested in the same kernel.

### U2. Two-half processing: why BF16 m128n128 only?
- `SUBTILE_HALVES.md`: BF16 has clear 2-half asymmetry; FP8 and NVFP4 do not.
- Hypotheses (none verified): (a) BF16 m128n128 has a specific 2× 64-N MAC array geometry; (b) FP8/NVFP4 K is larger so pipeline depth uniformizes; (c) something in the descriptor format / TMEM layout differs.
- Untested: BF16 with smaller M, BF16 with cluster_group::2.

### U3. Pattern-count anomaly (3-pattern < 2-pattern < 4-pattern)
- `CROSS_PRECISION_SUBTILE.md` modes 3101-3108: 3-pattern rotation gives 538 W, 2-pattern gives 623 W (worse), 4-pattern back to 610 W. No clean model fits this.
- Sticky-activation framing fits 4 of 8 cases but underestimates savings for some.
- Likely needs hardware reverse-engineering or NCU counter that doesn't exist publicly.

### U4. Cache replacement policy: NOT simple LRU
- `N_DEPENDENCE_DEEPDIVE.md`: chunk=1 ABAB triggers full speedup; chunk=2 AABB does not. A simple 2-slot LRU should keep both A and B in cache regardless of arrangement.
- Hypotheses: (a) HW has a specific "alternating period-1" pattern recognizer separate from a per-K-tile constancy detector; (b) per-K-tile constancy detector is what powers chunk≥8 cases.
- Two HW detection paths likely; their interaction at intermediate chunks creates the dip-then-recovery curve.

### U5. Sub-tile dedup vs K-row dedup magnitude in cuBLAS
- `N_DEPENDENCE_DEEPDIVE.md` "Sub-tile dedup vs K-row dedup": at N=K=8192, sub-tile dedup contributes only ~1-3% to cuBLAS speedup, while K-row gives 42%.
- This suggests cuBLAS's 64-K-stage iteration amortizes K-row dedup far better than per-row sub-tile dedup. Custom tcgen05 kernels may show stronger sub-tile effects (as the per-MMA tests do).
- Whether the per-MMA sub-tile dedup mechanism actively contributes to cuBLAS performance, or only the K-row mechanism dominates, is not cleanly separated.

### U6. A-vs-B asymmetry generalizes to A-major MMA layouts?
- `A_VS_B_ASYMMETRY.md` confidence: "LOW on whether this transfers to A-major MMA layouts (untested)."
- All measurements use the standard B-distributed layout. tcgen05.mma in 2-CTA cluster mode or with different operand orientations might show different A/B behavior.

### U7. Cross-precision two-half analog?
- BF16 has two halves at N=64 boundary. FP8 (K=32) and NVFP4 (K=64) might have analogous structure at different N positions; uniform position test only tested 0..7 sub-tiles, not finer/coarser.

### U8. tcgen05 vs mma.sync kind::f8f6f4 dedup behavior
- All FP8 dedup measurements in this corpus are tcgen05.mma kind::f8f6f4 (the native B300 path).
- Whether the same 32-byte sub-tile dedup applies to the mma.sync kind::f8f6f4 emulated path (F2FP.UNPACK + HMMA) is untested. It probably doesn't, because the conversion runs on the regular MUFU/INT pipeline, not the multiplier array. But not measured.

---

## Confidence summary

| Claim | Confidence |
|-------|:----------:|
| 32-byte universal sub-tile boundary | HIGH (3 precisions, cliff at 17/33/65) |
| A-vs-B asymmetry magnitude | HIGH (5 measurements, A K-vary < 7 W everywhere) |
| BF16 two-half processing | HIGH (8 monotonic measurements, mirror test confirms) |
| Two-half is BF16-only | HIGH (FP8/NVFP4 uniform within ±10 W) |
| Sticky activation > slot caching | HIGH (N=64 has no free zone, would not happen with slot model) |
| K-row dedup pairwise (≤2 pattern works) | HIGH (period 1, 2 work; period 3+ no) |
| Sparsity 3-tier independence | HIGH (mechanisms cleanly separable) |
| 2:4 structured = +11% on dense | HIGH (5 layouts tested, only fixed-position triggers) |
| 2-CTA = no cluster pooling | HIGH (4 measurements match single-CTA pattern) |
| Per-MMA dedup state (no cross-MMA) | HIGH (4 tests, mode 2/3 prediction matches) |
| disable_lane linear 2.4 W/col | HIGH (5 measurements monotonic) |
| Diagonal popcount-invariance | MEDIUM (best fit but not direct popcount test) |
| Cache slot count (1 vs 2) | UNRESOLVED (see U1) |
| Pattern-count anomaly mechanism | UNRESOLVED (see U3) |
| Replacement policy NOT LRU | HIGH (chunk=1 vs chunk=2 evidence) |
| Cross-precision two-half analog | UNRESOLVED (see U7) |

---

## Appendix: Recipes for power-aware tcgen05 GEMM (BF16 m128n128 example)

1. **Quantize / sort B columns** so byte-identical 32-byte chunks cluster contiguously along N. Saves ~250-310 W vs unsorted random.
2. **Place repeating sub-tiles at LOW N (Half A)**, arbitrary at HIGH N (Half B). Saves up to 269 W more.
3. **Group K rows so consecutive rows match or alternate ABAB-style**. Adds 5-25 W penalty per unique K row pattern (vs full random 47-99 W per precision).
4. **disable_lane unused output columns**: ~2.4 W per column on BF16.
5. **Combined realistic best case**: 254 W (vs 610 W random) = -58%, applies per CTA.

For cuBLAS workloads, only steps 1-3 are accessible (no disable_lane control). Real ML weights satisfy essentially none of the trigger conditions for K-id speedup → ~2-6% practical inference benefit. Maximum production stack (FP8 + structured 2:4 sparse, batch ≥1024): ~3033 TFLOPS on Llama-70B FFN = 67% of HW peak.
