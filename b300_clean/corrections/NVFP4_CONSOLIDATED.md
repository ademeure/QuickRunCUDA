# NVFP4 Consolidated Reference (Corrections Audit)

**Date: 2026-04-22.** Cross-reference of every NVFP4 measurement across
~22 source files in `b300_clean/`. Originals NOT modified. This document
flags conflicts, supersedes stale numbers, and lists open questions.

Trust order applied (most → least authoritative):
1. Latest NVFP4_K96_AB_FULL.md addendum (2026-04-20, TDP-cap regime)
2. NVFP4_CUTEDSL_THROTTLE_DEEP_DIVE.md (model + king-shape verified)
3. NVFP4_CUBLAS_FULL_SWEEP.md (sustained, multi-clock)
4. NVFP4_CUDAGRAPH.md (record-breaking peak)
5. Per-area deep-dives (NVFP4_K96_*, NVFP4_PERIOD_*, NVFP4_PURE_TCGEN05_*)

---

## 1. Per-config absolute peak TFLOPS table

### cuBLAS NVFP4 (production library)

| Clock      | Best shape (M, N, K)         | TFLOPS  | %15PF spec | Source                              |
|------------|------------------------------|---------|------------|-------------------------------------|
| Boost ~2032| 8192, 8192, 38400 + cudaGraph BPG=16 | **11423** | **76.2%** | NVFP4_CUDAGRAPH.md (record)         |
| Boost ~2032| 8192, 8192, 38400 (plain Lt) | 11068   | 73.8%      | NVFP4_CUBLAS_FULL_SWEEP.md / CUTEDSL_THROTTLE |
| Boost (zero-data)| 8192² K=15K            | 9987    | 66.6%      | CUTEDSL_THROTTLE big table          |
| Boost (random) | 8192² K=15K (sustained)  | ~7000   | ~47%       | CUTEDSL_THROTTLE (TDP cap @1455 MHz)|
| 1500 MHz   | 8192², K=38400               | 9273    | 83.7% @clk | NVFP4_CUBLAS_FULL_SWEEP.md          |
| 1005 MHz   | 8K², K=38K (cuBLAS)          | 6776 (CuTeDSL) / cuBLAS K-sweep peak | per-clock | various |
| 510 MHz    | 16384², K=61440              | 3558    | 94.5% @clk | NVFP4_CUBLAS_FULL_SWEEP.md          |

### CuTeDSL persistent kernel

| Clock | Best shape           | TFLOPS | MFU/total | Source                       |
|-------|----------------------|--------|-----------|------------------------------|
| Boost | M=N=16384, K=15360, cluster (2,4) | **8112** | 54.1% | CUTEDSL_THROTTLE (zero data) |
| 1005  | 8192² K=61440, cluster (2,1) | 6776 | **91.3%** | CUTEDSL_THROTTLE (per-total record) |
| 1005  | 32768² K=15360, (2,4) | 5319 | **88.4% per-active** | CUTEDSL_THROTTLE |

### CUTLASS C++ (sample 89, sm103_fp4_ultra_gemm)

| Config | TFLOPS | MFU | Source |
|--------|--------|-----|--------|
| 1005 MHz, 8K² K=15K, 2SM, cluster (2,4) | 5544 | 77.7% @1005 | CUTEDSL_THROTTLE (CUTLASS section) |
| Boost, 8K² K=15K | 8285 (2SM) | ~55% | CUTEDSL_THROTTLE big table |

### Pure tcgen05.mma microbench (single kernel, SMEM-only)

| Precision | Tile  | cy/MMA | PFLOPs (cluster) | MFU | Source |
|-----------|-------|--------|------------------|-----|--------|
| K=64 std  | M=N=256, 2-CTA cluster | 128 | 4.87 | 98.4% | NVFP4_K96_AB_FULL.md (1005 MHz) |
| K=96 ULTRA| M=N=256, 2-CTA cluster | 128 | 7.31 | 98.5% | NVFP4_K96_AB_FULL.md (1005 MHz) |
| K=96 ULTRA| same, 1500 MHz lock     | 128 | 10.89 | 98.5% | NVFP4_K96_AT_1500MHZ.md |
| K=96 ULTRA| unlocked, all-zero (zero-skip) | 128 | **14.78** @ 2032 MHz | 98.5% | NVFP4_K96_AB_FULL.md addendum |

### Memory ladder
| K-id boundary peak (cuBLAS) | 10.8 PF "catalog" / 11423 PF actual | 72-76% of 15 PF spec |

---

## 2. Power signature summary (NVFP4 K=96 ULTRA, M=N=256, 2-CTA, 1005 MHz lock)

### Operand A vs B ratio — RESOLVED contradiction

There were TWO seemingly opposite claims:

- **NVFP4_POWER_DECOMPOSITION.md** (cuBLAS-level): "A dominates, A:B impact ≈ 3:1"
- **NVFP4_PURE_TCGEN05_RESULTS.md** (microbench-level): "B dominates 15-20× over A"
- **CLAUDE memory note**: "A:B impact ~1:3" (B dominant)
- **Memory note**: "32-byte sub-tile B-side dedup, A is FREE"
- **NVFP4_K96_AB_FULL.md**: "B impact ≈ 2.6× A impact" (B dominant)

**Reconciliation (per NVFP4_PURE_TCGEN05_RESULTS.md "Correction" section):**
- **At pure-tcgen05 multiplier level: B dominates A by 15-30× across ALL precisions** (FP16/BF16/FP8/NVFP4).
- **In cuBLAS NVFP4 path: A dominates because cuBLAS multicasts B** (TMA multicast halves B's L2/memory pipeline cost).
- The "A is FREE" memory note refers to the K=96 single-kernel microbench (NVFP4_K96_AB_FULL.md), where A swing is 9-95 W vs B swing 13-249 W (B/A ≈ 2.6×, NOT 15-30×).

**Three different ratios** depending on test geometry:
| Source | A vs B | Why |
|--------|--------|-----|
| cuBLAS NVF4 | A > B (3:1) | TMA multicast on B masks B's true cost |
| Pure tcgen05 (A=zero baseline) | B >> A (15-30×) | Stripped of memory pipeline; pure multiplier port asymmetry |
| K=96 single-kernel matrix (A,B both varying) | B > A (~2.6×) | Both fed via SMEM-resident; close to the K=96 inference reality |

The K=96 paper's 2.6× is the most representative number for production NVFP4 K=96 inference power modeling.

### Sign-bit cost (CONSISTENT across files)

- Sign bit alone (8 positive vs 16 random): **~80 W per CTA** at K=96, 1005 MHz, M=N=256 (NVFP4_K96_AB_FULL.md, NVFP4_K96_B_DISTRIBUTION.md agree)
- Worst-case p_n=64 sign pattern: **+50 W on top** of full random → 605 W
- `+0` vs `-0` for zeros (sp=100%): saves **103 W** (NVFP4_K96_SIGNMATCH.md)
- Constant-sign weight encoding savings: 10-13% at M=N=128, 14-18% at M=N=256 2-CTA (NVFP4_SIGN_K64_K96.md)

### Magnitude diversity
- ~40 W per "additional magnitude" (saturates after ~5 distinct |x|)
- Constant any-value baseline: 295-303 W (within 10 W noise across all 16 codes)

### SF tensor (UE4M3)
- SF=random adds ~27 W vs SF=1.0 (B const) (NVFP4_SF_POWER.md)
- SF=0 saves 10-18 W (gates outputs)
- Confirmed independent and additive (NVFP4_SIGN_K64_K96.md "SF effects independent")
- Note: NVFP4_K96_AB_FULL.md SF table says ~25 W max range — agrees with NVFP4_SF_POWER.md within 2 W

### Outliers (1 in 16 weights)
- +29 W per CTA per outlier per K-block-of-16 (NVFP4_K96_AB_FULL.md, NVFP4_K96_B_DISTRIBUTION.md identical)
- Peak at 50/50 mix (526 W) > 100% outlier (519 W) — counter-intuitive but reproduced

### TDP-cap regime (unlocked clock, NVFP4_K96_AB_FULL.md addendum)
- All-zero B triggers zero-skip → 14.78 PFLOPs @ 2032 MHz, 633 W, **23.4 TF/W** (best ever)
- Random B → 13.01 PFLOPs @ 1788 MHz, 1095 W, 11.88 TF/W
- 12% throughput swing from data quality alone under TDP cap

### Best efficiency points
- **17.0 TF/W** at 1005 MHz, K=96 ULTRA, B = 5-pos {+0..+2} (NVFP4_K96_AB_FULL.md)
- **23.4 TF/W** at boost, K=96 ULTRA, B = all-zero (zero-skip path)
- 12-13.6 TF/W typical real workload (cuBLAS large-N, sustained)

---

## 3. Period sweep reconciliation

### Same 1-CTA finding from multiple files
- **K=96, N=256, p=64 (chunk-4 sub-tile)**: WORST case, 534 W (NVFP4_PERIOD_SWEEP.md)
- **K=96, N=64**: REMARKABLY FLAT — 0.5% spread (1.5 W across 11 periods)

### 2-CTA confirms
- K=96, M=N=256, p=64: 580 W worst (NVFP4_PERIOD_2CTA.md)
- Note slight numeric difference: 1-CTA p=64 gives 534 W, 2-CTA p=64 gives 580 W. This is consistent with 2-CTA having more multipliers active and the chunk-4 thrash propagating across the cluster.

### Universal recommendations (period sweep + cross-precision)
- Use period=2 (++--) for any precision (NVFP4_PERIOD_2CTA.md, PRECISION_PERIOD_SWEEP.md agree)
- Avoid period=N/2 ("two halves") at large N
- Avoid multiples of 3 (periods 3, 6, 12, 24, 48) — universal +12-23% penalty
- For BF16: avoid p=16. For NVFP4/FP8: avoid p=64 / p=32 respectively.

### N-stride contradicts BF16 sweep — BUT is acknowledged
- NVFP4 has NO 32-element MAC cliff (unlike BF16 which drops 126 W at stride 32)
- See NVFP4_PURE_TCGEN05_RESULTS.md "NVFP4 has NO 32-element MAC cliff like BF16"
- Mechanism: block-scale ULTRA path reorganizes B feeding through SF lookup; no parallel-broadcast structure to exploit

---

## 4. K-direction findings (consistent)

- **K-direction sign flips are essentially FREE** for sub-tile dedup (PN_PK_SWEEP.md, K_DIRECTION_LINEAR.md)
- **Per-K-row contribution is LINEAR**: ~19.5 W per non-zero K row in BF16 (K_DIRECTION_LINEAR.md)
- No "halves" effect in K direction (only N has the chunk structure)
- K-flip cost grows with K size: BF16 K=16 → +8 W, NVFP4 K=64 → +11 W, FP8 K=32 → +25 W, NVFP4 K=96 → +33 W (PN_PK_SWEEP.md)

---

## 5. CuTeDSL throttle deep-dive — key model results (HIGH confidence)

- **Throttle invalidates short benchmarks**: 10-iter random shows 7949 TF, but 5000-iter sustained shows 6902 TF (1455 MHz throttled)
- **Zero-data: holds 2032 MHz**, 8112 TF (CuTeDSL), 12.25 TF/W
- **Cluster (2,4) is best for boost+zero**, (2,1) wins at 1005 MHz lock
- **Tile = 256×256** is universal winner (powers of 2 only supported)
- **Clock-scaling model**: t_utcmma = 132/clock + 19 (ns) — fits 4 clock points within ~6%
- **Theoretical ceiling at boost: 73.1%** (132/(132 + 19×2.032)) — matches measured 73.7% at K=38400

---

## RETRACTIONS (superseded numbers)

| Old claim | Source file | Superseded by | Correct value |
|-----------|-------------|---------------|---------------|
| K=64 K-uniform-per-N saves 124 W (-28%) | NVFP4_SIGN_K64_K96.md commit `4c1e60a` (PRIOR) | NVFP4_SIGN_K64_K96.md current | Saves only ~1-3% (background processes contaminated baseline; rule #9) |
| K=96 K-row dedup is "weak/absent" | NVFP4_SIGN_K64_K96.md PRIOR | NVFP4_SIGN_K64_K96.md | K=64 and K=96 respond similarly (~10-13% sign-bit savings, no architectural difference) |
| "Sign-bit-only stride U-shape" interpretation | NVFP4_PURE_TCGEN05_RESULTS.md (early section) | Same file "CORRECTION" | Was a labeling error: mode 4 ≠ "stride 128"; real curve is monotonic-with-entropy + sharp cliff at 16/32 |
| "100% alternation is bad due to flip rate" | NVFP4_PURE_TCGEN05_RESULTS.md | Same file CORRECTION 2 | FORCED +-+- saves max power; HW responds to # unique patterns, not flip rate |
| NVFP4 cuBLAS "A dominates because of multiplier asymmetry" | NVFP4_POWER_DECOMPOSITION.md initial | NVFP4_PURE_TCGEN05_RESULTS.md | Multiplier itself has B>>A (15-30×); cuBLAS A-dominance is artifact of TMA multicast on B |
| "1 outlier per K16 costs 12% throughput at TDP cap" | NVFP4_K96_AB_FULL.md early version | Same file addendum (3-trial verified) | ~2% loss, not 12% (was a `--reuse-cubin` bug — measured wrong cubin) |
| NVFP4 stride results table | NVFP4_PURE_TCGEN05_RESULTS.md (NVFP4 N-stride) | Same file note | "NOT trustworthy — within-word strides had encoding bugs" (replaced by sign-bit-only test) |
| "10.8 PF cuBLAS catalog" as ceiling | Older catalog (CLAUDE memory note) | NVFP4_CUDAGRAPH.md + CUBLAS_FULL_SWEEP.md | True peak with cudaGraph BPG=16 = **11.42 PF** (76.2%); plain Lt = 11.07 PF (73.8%) |
| K=96 N=256 p=64 = 580 W (1-CTA) | (potential confusion) | NVFP4_PERIOD_SWEEP.md (1-CTA) gives 534 W; NVFP4_PERIOD_2CTA.md (2-CTA) gives 580 W | Both correct — different cluster sizes |

---

## UNRESOLVED open questions

1. **Why does CUTLASS C++ 89 stuck at 5.5-5.7 PF (1005 MHz) when CuTeDSL hits 6.78 PF and cuBLAS hits ~10.8 PF?** Same hardware, same shape. CUTEDSL_THROTTLE notes 61% wall-time is host overhead (`gemm.initialize()` in loop) — but even discounting that, CUTLASS C++ 89 lags CuTeDSL by 8-15pp MFU. CuTeDSL persistent kernel design and cluster-broadcast tighter? Open.

2. **What is cuBLAS doing that hits 76.2% MFU with cudaGraph?** Model predicts 73.1% ceiling from 19 ns fixed overhead. Graph eliminates launch-prep CPU time but shouldn't eliminate per-utcmma stall. Suggests the "19 ns" model has hidden launch-related component. Open.

3. **NVFP4 has no 32-element MAC cliff (unlike BF16)** — confirmed multiple times. The block-scale ULTRA path reorganizes B feeding so parallel-broadcast structure is invisible. Mechanism not measured at HW level (no ncu pipe metric for tcgen05 sub-tile dedup).

4. **Diagonal sign patterns (`sign[k][n] = ((n + k*shift) / p_n) & 1`) stay LOW even with many unique sub-tile patterns** (PN_PK_SWEEP.md). Contradicts the strict "≤2 patterns triggers LOW" rule. May indicate cache holds 8+ patterns OR there's a separate shift predictor. Open.

5. **Random-K-shift LOW for p_n ≤ 8 (8 unique patterns) but kphase_n shows 3 patterns = HIGH** (PN_PK_SWEEP.md). Different cache behavior per axis (K-row direction vs N-column direction)? Mechanism unclear.

6. **K=96 N=64 saturation (0.5% spread, FLAT across all periods)** — only K=96 N=64 has this property, no clear architectural explanation (NVFP4_PERIOD_SWEEP.md, NVFP4_PERIOD_2CTA.md both confirm).

7. **K-id speedup is shape-conditional** (CLAUDE memory note): cuBLAS 1.40× K-id BF16 speedup ONLY at N ∈ {K, 2K, K/2}. Real ML inference (N/K=2.5-3.5) outside window. Practical benefit ~2-6%. Whether NVFP4 K=96 ULTRA has the same restriction is not measured.

8. **Lossless int reduction**: NVFP4_INT_REDUCTION.md claims redux.sync.add gives 2.75× speedup, but NVFP4_FULL_PIPELINE.md shows mode 2 (HW decoders + per-thread fp32 + final SHFL) at 2.96× beats mode 7 (per-tile redux at 1.59×). Reduction speedup matters only when reduce is structurally per-tile. Apparent contradiction is reconciled in NVFP4_FULL_PIPELINE.md but should be flagged: **2.75× is for an isolated inner-loop reduce; not for end-to-end NVFP4 pipeline**.

9. **CUTLASS C++ vs CuTeDSL vs cuBLAS ceiling**: catalog says "cuBLAS 10.8 PF / CUTLASS 8.7 PF / CuTeDSL stuck below" — but CUBLAS_FULL_SWEEP shows cuBLAS hitting 11068 TF and CUDAGRAPH shows 11423. The "10.8 PF" claim is from an older session (before cudaGraph + K-sweep optimization). Memory note should be updated to "11.42 PF cuBLAS" and "8.3 PF CUTLASS C++ 89 / 8.1 PF CuTeDSL" (latter from CUTEDSL_THROTTLE 8112 TF).

10. **Tested but not deeply verified**: NVFP4_PARALLEL_STREAMS.md claims +20% from NS=2 parallel streams on 4K² K=6K (44.6% MFU vs 37.6% single-stream). Not cross-checked against cudaGraph in same file. Whether parallel streams + cudaGraph compounds (predicted +35% on small shapes) is not measured.

---

## File status snapshot

| File | Era | Trust | Notes |
|------|-----|-------|-------|
| NVFP4_K96_AB_FULL.md | 2026-04-20 | HIGH | Most comprehensive; addendum has TDP-cap clock data |
| NVFP4_K96_AT_1500MHZ.md | 2026-04-20 | HIGH | Clock-scaling validation |
| NVFP4_K96_B_DISTRIBUTION.md | 2026-04-20 | HIGH | Subset of K96_AB_FULL data |
| NVFP4_K96_SIGNMATCH.md | 2026-04-20 | HIGH | mo=64 lane-pair confirmation |
| NVFP4_PERIOD_SWEEP.md | 2026-04-20 | HIGH | 1-CTA period table |
| NVFP4_PERIOD_2CTA.md | 2026-04-20 | HIGH | 2-CTA confirmation |
| NVFP4_POWER_DECOMPOSITION.md | 2026-04-19 | MED-HIGH | cuBLAS-level; A>B finding superseded by pure-tcgen05 |
| NVFP4_PURE_TCGEN05_PLAN.md | 2026-04-19 | LOW | Planning doc only |
| NVFP4_PURE_TCGEN05_RESULTS.md | 2026-04-19 | HIGH (with self-corrections) | 6 precision deep-dive |
| NVFP4_SF_POWER.md | 2026-04-20 | MED | Single random hash; 4 measurements |
| NVFP4_SIGN_K64_K96.md | 2026-04-20 | HIGH (after rule #9 cleanup) | Definitive K-row vs sign-bit |
| NVFP4_PARALLEL_STREAMS.md | 2026-04-19 | MED | NS=2 sweet spot; PDL untested with cuBLAS |
| NVFP4_INT_REDUCTION.md | 2026-04-20 | HIGH | Theory + bench |
| NVFP4_FULL_PIPELINE.md | 2026-04-20 | HIGH | 7 modes A-B compared |
| NVFP4_CUBLAS_FULL_SWEEP.md | 2026-04-19 | HIGH | Per-clock optima |
| NVFP4_CUDAGRAPH.md | 2026-04-19 | HIGH | Record 11423 TF |
| NVFP4_CUTEDSL_THROTTLE_DEEP_DIVE.md | 2026-04-19 | HIGH | Most thorough single doc |
| KPHASE_SWEEP.md | 2026-04-20 | HIGH | per-column random phase |
| K_DIRECTION_LINEAR.md | 2026-04-20 | HIGH | BF16 |
| M_DIMENSION_HALVES.md | 2026-04-20 | HIGH | BF16 |
| PRECISION_PERIOD_SWEEP.md | 2026-04-20 | HIGH | Cross-precision |
| PN_PK_SWEEP.md | 2026-04-20 | HIGH | Combined N×K |
