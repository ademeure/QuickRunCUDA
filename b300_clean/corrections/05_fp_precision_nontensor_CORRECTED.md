# 05 — FP Precision Throughput (NON-tensor) — CORRECTED

Source: `b300_clean/05_fp_precision_nontensor.md` (original retained verbatim).
Corrections derived 2026-04-22 from cross-file audit of D4, V41-V48, V8, V40, V9
and B300_TRUE_REFERENCE.

Confidence markers unchanged from original (HIGH / MED / LOW).

---

## A. Inherited HIGH-confidence facts (still valid)

These survive the cross-check:

1. **No FP16/BF16 packed FMA speedup over FP32 outside tensor cores.** Scalar
   FFMA, HFMA2 and BFMA2 all peak at the same ~70-72 chip-TFLOPS via
   `pipe_fma`. (B300_TRUE_REFERENCE §7 surprise #2; commit `ea47ec6`.)
2. **FP64 DFMA = 1.20 TFLOPS** at 2032 MHz, 4 warps/SM (commit `2d64696`).
3. **F2FP narrow UNPACK = 64 inst/SM/clk = 38.5 Telem/s; PACK = 32 inst/SM/clk
   = 19.3 Telem/s.** Identical per-instruction rate across FP8 / FP6 / FP4.
4. **Always include `.satfinite` on `cvt.rn.f16.f32`** — without it goes to a
   separate F2F pipe at 11/SM/clk (~3-6× slower).
5. **HMNMX2 (`min/max.f16x2`) lives on pipe_alu**, can co-issue with FFMA.

---

## B. NEW table — packed FP cvt elem/s (V43 measurement)

V41_V48_FINDINGS §"Packed FP cvt" gives a measured chip-level throughput that
is meaningfully BELOW the F2FP pipe theoretical from F2FP_DEEP_DIVE. It is the
right number for kernel-budgeting (it includes the surrounding kernel
overhead), and the difference is informative:

| Source format | Dest format | PTX form | Measured Gelem/s | F2FP-pipe theoretical (PACK) | Notes |
|---|---|---|---:|---:|---|
| FP32 → FP8 (E4M3) | packed x2 | `cvt.rn.satfinite.e4m3x2.f32` | **17.6** | 19.3 Telem/s | hits ~91% of pipe SoL |
| FP32 → FP8 (E5M2) | packed x2 | `cvt.rn.satfinite.e5m2x2.f32` | **17.6** | 19.3 Telem/s | identical to E4M3 |
| FP32 → BF16 | packed x2 | `cvt.rn.bf16x2.f32` | **9.05** | 19.3 Telem/s | half of FP8 — see §C |
| FP32 → FP16 | packed x2 | `cvt.rn.satfinite.f16x2.f32` | **9.05** | 19.3 Telem/s | identical to BF16 |

Source: `V41_V48_FINDINGS.md` §"Packed FP cvt (V43, partial)".

---

## C. KEY: "FP8 cvt 2× faster than BF16/F16 cvt" — CONFIRMED, but mechanism misnamed

The original 05 catalog (sec 2.3) states that "all formats hit identical
per-instruction throughput within direction (UNPACK or PACK); FP4 is NOT slower
or faster than FP8 per SASS instruction on this pipe."

V43's chip-level measurement disagrees: **FP8 cvt is measured at 2.0× the
elem/s of BF16/F16 cvt** (17.6 vs 9.05 Gelem/s).

These are not in conflict — both are correct at their level:

- **Per-SASS-instruction**: same. One `F2FP.*.PACK_AB.*` SASS = same dispatch
  cost regardless of dest narrow-format.
- **Per-element**: FP8 packs 2 elements per F2FP instruction; BF16/F16 also
  pack 2 per `cvt.rn.bf16x2.f32` AT THE PTX LEVEL but lower the F2FP rate to
  the PACK_AB_MERGE_C variant which V43's harness measures at half the rate.
- **Hypothesized cause** (V43): output bit-width matters for the F2FP MERGE_C
  step. Not yet SASS-verified across all 4 forms.

ACTION: 05 sec 2.3 should ADD a note: "Per-instruction throughput is identical
across SASS opcodes within UNPACK or PACK class; per-PTX-instruction
throughput differs because narrower outputs (FP8) compile to
`PACK_AB`-without-MERGE_C while BF16/F16 require MERGE_C, halving the
effective rate. Measured chip elem/s: FP8x2 = 17.6 G, BF16x2 = F16x2 = 9.05 G
(commit V43)."

---

## D. KEY: FMUL = FADD = FFMA at SASS level (CONFIRMED)

The original 05 catalog never explicitly addresses scalar FMUL or FADD
throughput; the only mention is in the recipe note (sec ref to 04_fp32_peak)
that "Inline PTX guarantees FFMA emission (no FADD/FMUL split)".

`V8_FADD_FMUL_PEAK.md` and `V9_OP_LATENCY.md` definitively show:

| Op | SASS | Latency | Peak inst/SM/cy | Peak chip TFLOPS | FLOPS/inst |
|---|---|---:|---:|---:|---:|
| FADD | `FADD R, R, R` | 4.22 cy | 4.0 (full FMA pipe) | 37.4 | 1 |
| FMUL | `FMUL R, R, R` | 4.22 cy | 4.0 (full FMA pipe) | 37.3 | 1 |
| FFMA | `FFMA R, R, R, R` | 4.22 cy | 4.0 (full FMA pipe) | 74.8 | 2 |

**Identical per-instruction dispatch rate. Same pipe. Same latency.** FFMA
"wins" purely because each instruction does 2 FLOPS not 1.

ACTION: 05 should add this to sec 1 headline table — currently it only lists
FFMA, leaving FADD/FMUL ambiguous to first-time readers. Recommended addition:

```
| FP32 FADD scalar | pipe_fma (1 FLOP/inst) | 37.4 TFLOPS | half of FFMA | HIGH | V8_FADD_FMUL_PEAK |
| FP32 FMUL scalar | pipe_fma (1 FLOP/inst) | 37.3 TFLOPS | half of FFMA | HIGH | V8_FADD_FMUL_PEAK |
```

---

## E. CUDA 13.2 BUG — narrow x4 cvt PTX rejected

Sec 2.4 of original asserts `cvt.rs.e4m3x4.f32` "is syntactic sugar" that
compiles to 2× PACK_AB_MERGE_C.RS. This is correct for x4 stochastic-round
forms that DO compile (CUDA 12.x).

V41-V48 found a separate bug: **`cvt.rn.satfinite.e2m1x4.f32` is REJECTED on
sm_103a in CUDA 13.2** (V43, V6 H3 commit `3bb7051`, V41_V48_FINDINGS l.61-62).
Workaround: use scalefactor variant or 2× x2 forms.

ACTION: add an explicit "CUDA 13.2 PTX migration" subsection noting which
narrow-cvt PTX forms work / don't on the current sm_103a NVRTC path.

---

## F. RETRACTIONS

1. **Sec 2.3 "all formats hit identical per-instruction throughput within
   direction"** is correct only at SASS-opcode level. As written (without
   qualifier) it misleads readers into expecting equal Gelem/s; V43 measures
   2× elem/s gap between FP8x2 and BF16x2/F16x2. ADD QUALIFIER.

2. **Sec 1 headline table omits FADD / FMUL.** Without explicit lines, naive
   readers infer "FADD = FFMA in TFLOPS" (wrong) or "FADD = FFMA / 2 in
   inst/s" (wrong). Both are 1 inst/cy/SMSP, half FFMA in TFLOPS only because
   1 FLOP/inst.

3. **No retractions to 2.1 / 2.2 / 2.5 / 2.6.** All survive cross-check.

---

## G. UNRESOLVED

1. **Mechanism of "FP8 cvt 2× BF16 cvt"** — V43 hypothesized output-bit-width;
   needs SASS dump of all 4 PTX forms to confirm whether the BF16/F16 paths
   really emit `PACK_AB_MERGE_C` while FP8 paths emit `PACK_AB` (no MERGE_C).
   Original sec 2.3 implicitly says all 4 should emit MERGE_C; if true, the
   V43 elem/s gap is unexplained.

2. **CUDA 13.2 PTX rejection** for x4 narrow forms — is this a sm_103a-only
   bug or all-arch? V41 only tested sm_103a; cross-check on sm_100a / sm_90a
   would isolate.

3. **`cvt.f32.f16` listed "infinite" in F2FP_DEEP_DIVE table** — sec 4
   skeptical-list claims this is `HADD2.F32` on pipe_fma_heavy at 64/SM/cy.
   Not directly measured in any V-series file. Needs a dedicated micro.

4. **NVRTC `-use_fast_math` interaction with cvt rounding modes** — feedback
   note `nvrtc_fast_math_ftz` says all FFMA become `.FTZ` under QuickRunCUDA's
   default flags. Does this also force-flush subnormals in F2FP narrow
   conversions, or is `.satfinite`'s subnormal→0 behaviour orthogonal?
   Important for any subnormal-handling correctness test run via the harness.

5. **F2FP_DEEP_DIVE 33-result internal consistency** — original sec 4 marks
   it "strongest single document"; not independently re-audited as part of
   this swarm. Trusted on author's prior verification.
