# PRECISION_NVRTC_INCONSISTENCY_LOG

Cross-file inconsistencies for FP non-tensor precision conversions and the
NVRTC / module / cuLibrary toolchain. Detected 2026-04-22.

Originals NOT modified. See `05_fp_precision_nontensor_CORRECTED.md` and
`17_nvrtc_module_CORRECTED.md` for the per-file deltas.

---

## Inconsistency #1 — Per-instruction vs per-element cvt rates

| File | Claim |
|---|---|
| `05_fp_precision_nontensor.md` sec 2.3 | "All B300-native sub-formats... All formats hit identical per-instruction throughput within direction (UNPACK or PACK); FP4 is NOT slower or faster than FP8 per SASS instruction on this pipe." |
| `V41_V48_FINDINGS.md` (V43) | FP8 cvt 17.6 Gelem/s, BF16/F16 cvt 9.05 Gelem/s — **2.0× gap** |
| User MEMORY topic notes | "FP8 cvt is 2× faster than BF16 cvt (output bit-width hypothesis)" |

Resolution: BOTH measurements are correct at their respective levels. SASS
instructions ARE same-rate; PTX-instruction elem/s differs because
(hypothesized) BF16/F16 PTX requires a `MERGE_C` SASS variant that halves the
F2FP rate. Original 05 sec 2.3 needs a qualifier.

ACTION: edit 05 sec 2.3 (corrected file §C).

---

## Inconsistency #2 — Catalog 05 silent on FADD/FMUL throughput

| File | Claim |
|---|---|
| `05_fp_precision_nontensor.md` sec 1 table | Lists FFMA, HFMA2, BFMA2, DFMA, F2FP — no row for FADD or FMUL |
| `V8_FADD_FMUL_PEAK.md` | FADD = 37.4 TFLOPS, FMUL = 37.3 TFLOPS, FFMA = 74.8 TFLOPS, all at 97.65% pipe util |
| `V9_OP_LATENCY.md` line 13 | FADD/FMUL/FFMA all 4.22 cy latency |
| `M16_V9_FULL_SYNTHESIS.md` line 21 | "FFMA / FADD / FMUL : 4.22 cy" |
| `04_fp32_peak.md` line 173 | Acknowledges "no FADD/FMUL split" exists |

Resolution: FADD = FMUL = FFMA at SASS level, all 1 inst/cy/SMSP, 4.22 cy
latency, same FMA pipe. FFMA "wins" only in FLOPS-counted-per-instruction.
05 sec 1 should explicitly add 2 rows so naive readers don't infer wrong
TFLOPS for sum-reductions or scaling kernels.

ACTION: add 2 rows to 05 sec 1 (corrected file §D).

---

## Inconsistency #3 — "FP16 packed = 2× FP32" — already correctly retired everywhere

| File | Claim |
|---|---|
| `B300_TRUE_REFERENCE.md` §7 #2 | "FP16/BF16 packed FMA = FP32 throughput" (NO 2×) |
| `05_fp_precision_nontensor.md` sec 2.1 | "NO FP16/BF16 packed throughput speedup over FP32" |
| `README.md` line 15 + 42 | "FP16/BF16 = FP32 (no 2× packed)" |
| `16_power_clock.md` line 145 | "FP16/BF16 packed FMA gives 2× FP32 throughput" — explicitly RETIRED |
| `D4_PRECISION_POWER_PERF_TABLE.md` | (silent on this; no per-precision FP16/BF16 non-tensor row) |

Resolution: This inconsistency is ALREADY resolved across the catalog. No new
corrections needed. Flag this row only because the swarm prompt asked us to
"flag any file claiming FP16 packed = 2× FP32." None of the b300_clean files
make this claim. The claim only appears as a retracted/retired entry.

NO ACTION required.

---

## Inconsistency #4 — `cvt.rn.satfinite.e2m1x4.f32` rejection scope

| File | Claim |
|---|---|
| `V41_V48_FINDINGS.md` lines 61-62 | "CUDA 13.2 BUG: rejected on sm_103a despite valid in CUDA 12.x. Need PTX syntax migration." |
| `CURIOSITY_LIST_V6.md` H3 | "fails with arg mismatch. Likely needs cvt.scalefactor variant (per-block scale, 8-element groups)." |
| `M10_V6_SYNTHESIS.md` line 111 | "Standard cvt.rn.satfinite.e2m1xN.f32 syntax fails" |
| `17_nvrtc_module.md` sec 8 row 8 | Implies NVRTC accepts MORE PTX than ptxas — but does NOT explicitly note the e2m1x4 case where BOTH reject |

Resolution: NVRTC accepts MORE for `tcgen05.*`; NVRTC accepts SAME (= rejects)
for narrow-x4 cvt forms. The 17 sec 8 framing is too broad.

ACTION: edit 17 sec 8 (corrected file §C).

---

## Inconsistency #5 — `--use_fast_math` impact prominence

| File | Claim |
|---|---|
| `17_nvrtc_module.md` sec 2 line 54 | Parenthetical: "(and **already on by default in QuickRunCUDA**, `cuda_helper.h:227`)" |
| `14_math_intrinsics.md` lines 87-88 | Loud, top-of-section: "QuickRunCUDA harness: `-use_fast_math` ON by default. All `/`, `sqrtf`, `1.0f/x` get the approximate path." |
| User MEMORY `feedback_nvrtc_fast_math_ftz` | "All FFMA become .FTZ; can't measure subnormal handling without removing flag" |
| `M7_V5_SYNTHESIS.md` D1 | Standalone nvcc with `-ftz=false` measures subnormal FFMA at 4.11 cy / no penalty (different harness) |

Resolution: 17 sec 2's parenthetical massively understates impact. Even
`14_math_intrinsics.md` makes this a top-line callout. 17 (the dedicated NVRTC
catalog) should ALSO promote.

ACTION: promote in 17 (corrected file §B).

---

## Inconsistency #6 — Init/main kernel arg-slot sharing not documented

| File | Claim |
|---|---|
| User MEMORY `feedback_compute_pipe_methodology` | "QuickRunCUDA passes same -0/-1/-2 to both init and main kernel. Don't use same arg slot for `iters` (main) and an init param. Pack init params in u1 via bit-shift instead." |
| `17_nvrtc_module.md` (entire file) | Silent on this harness quirk |
| `CLAUDE.md` "Kernel contract" section | Notes init kernel "with the same signature" but doesn't warn about arg-slot collision |

Resolution: Real, repeatable bug class. Belongs in 17 as a "Harness quirks"
subsection.

ACTION: add subsection to 17 (corrected file §D).

---

## Inconsistency #7 — DFMA TFLOPS (1.20 vs 1.0)

| File | Claim |
|---|---|
| `05_fp_precision_nontensor.md` sec 1, sec 2.2 | DFMA = **1.20 TFLOPS** at 2032 MHz, 4 warps/SM |
| `B300_TRUE_REFERENCE.md` §2 | DFMA = **1.20 TFLOPS** (commit 2d64696) |
| `D4_PRECISION_POWER_PERF_TABLE.md` row 4 | DFMA = **1.0 TFLOPS** (84% of 1.2 spec) |
| `V8_FADD_FMUL_PEAK.md` line 61 | DFMA = **1.20 TFLOPS** (100%, single DFMA port) |

Resolution: 1.20 is achievable at full warp count + boost (per 05 sec 2.2 +
B300_TRUE_REFERENCE + V8). D4's 1.0 reflects a "scalar 4-chain" measurement at
84% — a sub-saturated regime. Pick 1.20 as headline; note 1.0 as
under-saturated.

ACTION: D4 should add a footnote that the 1.0/84% is a 4-chain regime, not the
true peak (1.20 at full warp count).

---

## RETRACTIONS (consolidated)

1. **05 sec 2.3 "all formats identical per-instruction"** — needs qualifier;
   per-PTX-instruction Gelem/s differs 2× between FP8 and BF16/F16.
2. **05 sec 1 silent on FADD/FMUL** — add 2 rows; both = 37.4 TFLOPS = half
   FFMA in FLOPS but same SASS rate.
3. **17 sec 8 "tcgen05 PTX rejected by NVRTC — Wrong direction"** — direction
   correct but over-stated. Specify it applies to `tcgen05.*` only, NOT to
   narrow-cvt PTX (which BOTH NVRTC and ptxas reject).
4. **17 sec 2 parenthetical on `-use_fast_math`** — promote to top-of-section
   callout; this changes FFMA to .FTZ and breaks subnormal-handling tests run
   via the harness.

## UNRESOLVED (consolidated)

1. **Mechanism of "FP8 cvt 2× BF16 cvt"** — needs SASS dump of all 4 PTX
   forms to confirm MERGE_C vs no-MERGE_C hypothesis.
2. **`cvt.scalefactor.*` end-to-end via NVRTC** — never tested; replaces
   rejected x4 narrow forms in CUDA 13.2.
3. **CUDA 13.2 PTX rejection scope** — only sm_103a tested; sm_100a / sm_90a
   would isolate.
4. **`cvt.f32.f16` widening "infinite" rate** — claimed `HADD2.F32` on
   pipe_fma_heavy at 64/SM/cy in 05 sec 4; not directly measured anywhere.
5. **`-use_fast_math` interaction with cvt rounding mode** — does it force
   FTZ on `cvt.rn.satfinite.e4m3x2.f32` subnormals? Probably no (rounding mode
   is explicit) but unverified.
6. **Init/main arg-slot quirk** — not in 17; widely-tripped, deserves a
   dedicated subsection.
7. **NVRTC vs nvcc cubin-equivalence diff** — never SASS-diffed in catalog.
8. **`cuLibrary*` re-measure with current driver** — catalog inherits older
   6.5× claim from a single legacy line.
9. **`-G` cubin runtime impact** — 4× compile, 10× cubin known; runtime
   slowdown not quantified.
10. **D4 row "DFMA 1.0 / 84%"** vs all other files at 1.20 / 100% — D4 should
    footnote that 1.0 is a sub-saturated regime not a corrected number.
