# tcgen05 Power Family — INCONSISTENCY LOG

**Auditor**: tcgen05-power swarm wave-3, 2026-04-22
**Scope**: Inconsistencies across `TCGEN05_PERFW_CLEAN_2TRIAL.md`, `TCGEN05_PERF_WATTS.md`, `TCGEN05_POWER_GUIDE.md`, `TCGEN05_POWER_MASTER.md`, `TCGEN05_N_MATRIX.md`, `TCGEN05_N_SWEEP.md`, `TCGEN05_PATH_NOTES.md`, `MMA_SYNC_POWER.md`, plus cross-checks with `corrections/06_tensor_cores_CORRECTED.md`, `corrections/16_power_clock_CORRECTED.md`, `corrections/TCGEN05_DEDUP_CONSOLIDATED.md`, and `B300_TRUE_REFERENCE.md`.
**Status**: catalogued, NOT amended in originals.

---

## I1. Single-trial vs 2-trial perf/W table — MATERIAL DIFFERENCES

Same kernel, same lock (1500 MHz), same shape (M=N=256, cta=2), measured weeks apart:

| Format            | PERF_WATTS W (1-trial) | CLEAN_2TRIAL W (2-trial) | Δ        |
|-------------------|------------------------:|--------------------------:|---------:|
| TF32 K=8          | 816                     | 787                       | -29 W    |
| FP16 K=16         | 940                     | 935                       | -5 W     |
| BF16 K=16         | 829                     | 876                       | +47 W    |
| FP8 e4m3          | 854                     | 1073                      | **+219 W**|
| MXFP8             | 851                     | 1041                      | **+190 W**|
| NVFP4 K=64        | 797                     | 689                       | **-108 W**|
| NVFP4 K=96 N=256  | 795                     | 870                       | **+75 W** |

**Self-explained in PERFW_CLEAN_2TRIAL §"Methodology pitfall confirmed"**: 5 leftover QuickRunCUDA processes silently inflate cy/MMA up to 8.5×. Single-trial PERF_WATTS likely has the same contamination (NVFP4 K=64 ≈ K=96 power is impossibly close given 1.5× work — diagnostic signature of contamination).

**Resolution**: PERFW_CLEAN_2TRIAL supersedes PERF_WATTS for ALL absolute W values and TF/W. PERF_WATTS retained for peak PF and N=192 saturation discussion. Captured in `TCGEN05_POWER_CONSOLIDATED.md` R1.

---

## I2. NVFP4 K=96 headline TF/W — three different "peak" claims

| Source                                | TF/W   | Conditions                |
|---------------------------------------|--------|---------------------------|
| TCGEN05_PERF_WATTS row 18             | 13.72  | "1500 MHz, random data"   |
| TCGEN05_PERFW_CLEAN_2TRIAL random row | 12.54  | 1500 MHz, random          |
| TCGEN05_PERFW_CLEAN_2TRIAL B-positive | 15.16  | 1500 MHz, B sign-bit zero |
| TCGEN05_PERFW_CLEAN_2TRIAL A+B-pos    | 15.74  | 1500 MHz, A+B sign-zero   |
| TCGEN05_PERFW_CLEAN_2TRIAL realistic  | 13.4   | LLM-like quant            |

The 13.72 PERF_WATTS headline corresponds to no real condition under clean methodology; it falls between random-clean (12.54) and B-positive (15.16) by accident of contamination.

**Resolution**: cite 12.54 (random) / 15.16 (B-positive) / 15.74 (A+B-pos) explicitly. Drop "13.72" usage.

---

## I3. NVFP4 K=96 power (770s vs 870s) — the headline contradiction

- `TCGEN05_PERF_WATTS.md`: 795 W → 13.72 TF/W
- `TCGEN05_PERFW_CLEAN_2TRIAL.md` (random N=256): 870 W → 12.54 TF/W
- `TCGEN05_POWER_MASTER.md` (NVFP4 random, 1005 MHz): 463 W
- Multiplied to 1500 MHz via 1.80× ratio (per MASTER boost-validation table) → 833 W. Closest to 870 W from CLEAN_2TRIAL, NOT 795 W from PERF_WATTS.

The 463 W → 870 W (1500 MHz) extrapolation lands within 5% of CLEAN, supporting CLEAN as authoritative.

---

## I4. Power baseline at 1005 MHz — minor inconsistency BF16 const

- TCGEN05_POWER_MASTER §"Cross-Precision Random Baselines" line 95: BF16 const 299 W
- TCGEN05_POWER_GUIDE row 13: "Optimized GEMM (BF16) | 287-400" (range incl. floor)
- TCGEN05_POWER_GUIDE row 11: "Active floor (A=0, B=0, 148 SMs) | 287"
- POWER_FLOOR.md (cited): 287 W absolute minimum
- A=B=0 (all zero) = 287, const non-zero = 299 — these are TWO DIFFERENT operating points, not a contradiction. GUIDE conflates them in the "287-400" range.

**Resolution**: distinguish A=B=0 floor (287) vs Tier B const non-zero (299) when citing.

---

## I5. PERF_WATTS NVFP4 K=64 vs K=96 are NOT distinguishable

PERF_WATTS rows 17-18 show NVFP4 K=64 = 797 W and K=96 = 795 W. Throughput ratio is 1.5× (4.87 vs 7.31 PF) → power must be different by SOMETHING (multiplier activity or memory traffic). 2 W gap is impossibly small.

CLEAN_2TRIAL: K=64 = 689 W, K=96 = 870 W (gap 181 W). This gap is consistent with the K-axis cost analysis in CLEAN_2TRIAL §"K-axis vs N-axis variation".

**Resolution**: PERF_WATTS rows 17-18 are diagnostically corrupted; replaced.

---

## I6. cy/MMA at N=192 (NVFP4 K=96) — three sources, all consistent

- TCGEN05_PERF_WATTS row 18 implies cy/MMA = 128 (claimed peak from N=256)
- TCGEN05_N_SWEEP row 192: 96 cy/MMA (NVFP4 K=96)
- TCGEN05_N_MATRIX row 192 (cta=2): 96 cy/MMA
- PERFW_CLEAN_2TRIAL row "NVFP4 K=96 N=192" notes 1.33× iters at N=192

These are CONSISTENT: cy/MMA scales linearly past saturation, so N=128 and N=192 give same MAC/cy. The discrepancy in PERF_WATTS row 18 column header "cy/MMA 128" likely refers to N=256 (where cy=128 with the same 49,152 MAC/cy as N=192). No inconsistency.

---

## I7. NVFP4 N=192 vs N=256 power — predicted equal, measured equal

- PERFW_CLEAN_2TRIAL: N=192 = 880 W, N=256 = 870 W (within 10 W)
- PERF_WATTS: only N=256 reported (795 W contaminated)

Conclusion in CLEAN: N=192 is optimal (same throughput, lower per-MMA latency, less SMEM). This is consistent. Captured in CONSOLIDATED §1.

---

## I8. Clock-state TF/W: implicit "1500 MHz lock" everywhere — no boost ladder

All TF/W in the family is at 1500 MHz lock. No tcgen05 doc reports TF/W at boost (2032 MHz) or at 1005 MHz lock. This is a gap, NOT an inconsistency, but means cross-document quotes must explicitly say "1500 MHz lock". Memory rule "ML inference USE BOOST" cannot be directly validated from this family alone.

Captured in CONSOLIDATED §3, U1, U3.

---

## I9. POWER_GUIDE "values scale ~1.7-2x at boost" vs MASTER actual measurement

- POWER_GUIDE line 16: "At boost (2032 MHz), values scale ~1.7-2x"
- MASTER §"Boost-clock validation":
  - Random BF16: 609 → 1097 = **1.80×** ✓
  - All-zero BF16: 294 → 621 = **2.11×** (slightly above range)
  - Const +1.0: 299 → 643 = **2.15×** (above range)

The "1.7-2x" range covers random data well but UNDERSHOOTS for low-power operating points (which scale more than 2×). Minor; affects active-floor power estimates at boost more than peak random.

---

## I10. POWER_GUIDE quick-reference `Random data GEMM (BF16) = 611 W` — clock NOT specified

POWER_GUIDE row 12: "Random data GEMM (BF16) | 611 | Worst case (typical training gradient)" — does not annotate clock. Implicit 1005 MHz (matches MASTER 609 W within noise). At 1500 MHz lock, BF16 random = 876 W; at boost, 1097 W. Reader could mis-quote.

**Resolution**: when citing POWER_GUIDE numbers, always check footer "All measurements at 1005 MHz fixed clock" (line 17).

---

## I11. cuBLAS BF16 sustained 962 W (corrections/16) vs tcgen05 BF16 random microbench 876 W (PERFW_CLEAN, 1500 MHz)

- corrections/16 §2 row 1920 MHz: cuBLAS BF16 random = 1099 W (TDP-cap)
- corrections/16 §2 row 2032 MHz: cuBLAS BF16 random = 962 W (under cuBLAS BF16, sustained)
- PERFW_CLEAN BF16 random @ 1500: 876 W

Different clocks, different occupancy structures (cuBLAS uses persistent + cluster_group::2 from `CUBLAS_BIT_ENTROPY_CORRECTION`). Consistent: 876 W × (boost/1500)^2 ≈ 1612 W (capped to 1100 W TDP). The 962-1099 W cuBLAS range is what TDP-throttle gives. No inconsistency in the underlying physics.

---

## I12. Per-CTA scaling — 1-CTA vs 2-CTA "throughput-neutral" but power per CTA?

- TCGEN05_N_MATRIX §"cta=1 vs cta=2 is throughput-neutral" for BF16/FP8/MXFP8/NVFP4-K64
- TCGEN05_DEDUP_CONSOLIDATED §8: 2-CTA shows IDENTICAL per-cluster power dependence to 1-CTA mode; per-CTA cache, NO cluster pooling

These are consistent: power scales linearly per CTA, but for the same total work, the cta_group::2 launch consumes the same total power as 2 cta_group::1 launches. No power benefit to clustering.

PERFW_CLEAN runs cta=2 throughout but reports per-cluster power (via NVML on whole GPU, then divided by cluster count = 74). This means PERFW W values are PER-CLUSTER (= 2 CTAs); TF/W is GPU-wide. Cross-doc citation must clarify.

---

## I13. mma.sync FP8 path "kind::f8f6f4 = 276 TFLOPS native" — RETRACTED in DEDUP_CONSOLIDATED

- MMA_SYNC_POWER.md still presents FP8 mma.sync as 251 W active and "+54 W random gap" without flagging that kind::f8f6f4 on mma.sync is NOT native FP8 (per `MMA_FP8_KIND_F8F6F4_NOT_NATIVE.md`).
- This is a documentation hazard for power readers: the FP8 mma.sync power numbers are for the EMULATED path (F2FP.UNPACK + 2× HMMA), not native FP8. Power per MAC will differ from tcgen05 native FP8.

DEDUP_CONSOLIDATED §12 captures this for the dedup story; carrying over here for the perf/W story too. Recommend a footnote in MMA_SYNC_POWER addendum on FP8.

---

## I14. "1500 MHz is TDP-safe max" claim — NOT actually safe

- PERF_WATTS line 5: "1500 MHz (TDP-safe max — `-lgc 1500,1500`)"
- PERFW_CLEAN_2TRIAL FP8 random = 1073 W (within 27 W of 1100 W TDP cap)
- corrections/16 §2 1500 MHz row: DRAM-read random = 1071 W → "TDP-cap APPROACHED"

At 1500 MHz, FP8 tcgen05 random nearly hits TDP. Calling this "safe" is generous — the clean measurement worked but a slightly hotter test would have throttled. CLEAN_2TRIAL doesn't flag this margin.

**Status**: the "TDP-safe max" framing is technically WRONG for FP8 random; should be "TDP-headroom max for typical workloads, FP8 random barely fits". Affects how readers interpret "no throttle" assumption.

---

## I15. ncu pipe_tensor — clean (no false usage)

Verified via grep: NO `pipe_tensor` references in any tcgen05 power-family doc. (Some references in mma.sync and cuBLAS-related docs which is correct usage.) Captured in CONSOLIDATED §5.

---

## Summary of recommendations

1. **Cite TCGEN05_PERFW_CLEAN_2TRIAL** for all tcgen05 perf/W; treat PERF_WATTS as superseded for absolute W values.
2. **Always state clock** when quoting tcgen05 W or TF/W (1005 MHz lock vs 1500 MHz lock vs boost).
3. **Distinguish active-floor (287) from Tier-B-const (299)** when discussing tcgen05 baseline power.
4. **Add headroom caveat** to "1500 MHz TDP-safe max" claim — FP8 random is within 27 W of the cap.
5. **Footnote MMA_SYNC_POWER FP8 results** with the kind::f8f6f4-not-native note from DEDUP_CONSOLIDATED.
6. **Run boost-clock TF/W ladder** to validate the "1500 MHz ≈ best perf/W" implicit assumption (currently only 1500 MHz is comprehensively measured).
7. **Drop "13.72 TF/W" headline** in favor of explicit-condition values from CLEAN_2TRIAL (12.54 random / 15.16 B-pos / 15.74 A+B-pos).
