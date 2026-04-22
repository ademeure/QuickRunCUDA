# Dual-issue 55%/74% headline — adversarial doubt report

Date: 2026-04-22. Subject: V49 (`501134a`) "FFMA+LOP3 same-warp dual = 55%" and
V50 (`fbe1c18`) "warp-specialized = 74% of separate-pipe theoretical".

## TL;DR

The V49/V50 headline numbers are **methodologically suspect**. The
"separate-pipe theoretical" they divide by is itself wrong, because the
FFMA-solo baseline is **latency-bound at ~67% of FP32 peak**, not pipe-
saturated. The "55%" and "74%" overlap fractions are therefore measuring
something other than dispatch-slot contention. Confidence in the
"4 inst/cy/SM dispatch cap" claim: **LOW–MED**.

## Key methodological problems

1. **FFMA solo is NOT at peak.** V49 reports `FFMA solo = 26066 Glane/s`.
   Peak at 2032 MHz = 148 SM × 128 lane × 2.032 GHz = 38484 Glane/s.
   26066 / 38484 = **67.7%** — exactly the latency-bound regime
   `A4_FFMA_PORT_PRESSURE.md` and `B1_DUAL_ISSUE_FFMA_IADD3.md` warn about.
   `V8_FFMA_PEAK_VERIFIED.md` shows FFMA reaches 97.6% of peak only at
   ≥4 warps/SMSP with proper NCHAIN. V49 uses 128 thr × 2 CTAs/SM = 8
   warps/SM = 2 warps/SMSP — under-occupied. So the "100% baseline"
   the dual-issue ratio is computed against is itself a 67% number.

2. **`__launch_bounds__(128, 2)` = 2 warps/SMSP, not enough.** A1 explicitly
   notes "to exploit cluster-level parallelism, need ≥2 warps per SMSP so
   the warp scheduler can co-issue from different warps to different
   clusters". V49 has exactly 2 warps/SMSP — the borderline case. V50 uses
   256 thr × 2 CTAs = 4 warps/SMSP, which IS enough — and V50 measures 74%,
   higher than V49's 55%. The 55% → 74% jump tracks the warp-count
   change, not the "warp specialization" hypothesis.

3. **`+f`/`+r` self-dep chains.** Both FFMA `%0,%0,imm,imm` and LOP3
   `%0,%0,imm,imm` have a **self-RAW dependency** on the destination
   register. With 4-cy FFMA latency and 8 independent chains
   (`f[0]..f[7]`), per-warp issue rate caps at 8 inst / 4 cy = 0.5 inst/cy.
   Two warps/SMSP can fill 1 inst/cy/SMSP only if scheduler perfectly
   interleaves — explaining the 67% solo number above.

4. **Mixed-mode register count doubles.** OP=2 uses BOTH `f[0..7]` and
   `u[0..7]` = 16 live registers vs 8 in solo mode. This may cross a
   register-allocator boundary that changes occupancy. SASS not verified
   in V49/V50 for OP=2 register count or spill behaviour.

5. **A1 (single-warp) measures 6.5% same-warp overlap, B1 measures 17%
   FFMA+IADD3 overlap, V49 measures 54-55%.** These three numbers
   "agreeing" qualitatively but disagreeing quantitatively (6.5% vs 17% vs
   55%) is itself a red flag — they were measured under different occupancy
   regimes, and the V49 number is likely the most-affected by point (1).

## Alternative explanations the V49/V50 author did NOT rule out

- **RF read-port pressure** (A4) — the FFMA in OP=2 has 1 unique register
  source, so RF ports are FINE for FFMA itself. But the LOP3 also reads its
  own `u[k]`, so combined warp-cycle reads could exceed the 2 ports/cy
  budget if the scheduler tries to co-issue.
- **Insufficient warps in flight** — see point 1/2 above. The clean
  control would be: re-run V49 with `__launch_bounds__(128, 4)` or 256
  thr per CTA so each SMSP has 4 warps. Prediction: same-warp dual would
  rise above 55%.
- **Scheduler latency-hiding limit** — at 2 warps/SMSP, when one warp
  stalls on the 4-cy FFMA RAW, only 1 other warp can hide it. With
  4+ warps/SMSP, the scheduler has 3 alternatives, exposing more dual-
  issue opportunity.
- **`%clock64` measurement vs CUDA event** — V49/V50 use CUDA events
  (which is correct). But the per-clock metric is computed from event
  time × 2.032 GHz; if clock state was actually 1920 (clock paradox per
  CLAUDE.md), all percentages shift by 6%.

## What SHOULD settle it (one-shot test)

Run `bench_dual_issue_warps_sweep.cu`: same OP=2 pattern as V49, but sweep
warps/SMSP ∈ {1, 2, 4, 8} via `__launch_bounds__`. Add ncu metrics
`smsp__inst_issued.avg.per_cycle_active`,
`sm__inst_issued.avg.pct_of_peak_sustained_elapsed`,
`smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active`,
and the alu/lsu equivalents. **Predictions:**

- If dispatch-cap hypothesis is correct: at 8 warps/SMSP, dual still
  caps at ≤80% of separate-pipe theoretical, AND
  `smsp__inst_issued.per_cycle_active` saturates at ~1.0 (= 4/SM).
- If latency/occupancy hypothesis is correct: dual rises to ≥90% of
  separate-pipe theoretical at 4-8 warps/SMSP, AND
  `pipe_fma_cycles_active` + `pipe_alu_cycles_active` sum to >130%
  of single-pipe peak.

V49/V50 did NOT collect ncu metrics. This is the smoking-gun evidence
needed; until then both hypotheses fit the data equally well.

## Confidence verdict

| Claim | Original | Revised |
|---|---|---|
| Same-warp FFMA+LOP3 = 55% of unverified-baseline | "HIGH" | **MED** (number is real, but baseline is wrong) |
| Warp-spec FFMA+LOP3 = 74% | "HIGH" | **MED** (occupancy still under) |
| "B300 dispatch slot is shared 4 inst/cy/SM regardless of pipe" | implied | **LOW** (extrapolation; needs ncu confirmation at high warps/SMSP) |
| "Dual-issue NEVER reaches 100% even with separate pipes" | implied | **LOW** (V8/A6 already show MUFU+FFMA ≈ 100%; HMMA+LDS at 73-96%; counter-examples exist in M8) |

The architectural claim that "dispatch is 4-wide per SM regardless of pipe"
contradicts M8's MUFU+FFMA = 100%+ and HMMA+HMMA = 69%, both of which
require multi-pipe co-issue per SM. V49's data is consistent with a
weaker claim: "same-warp can't dual-issue different pipes when the warp
is already at peak" (because each warp can issue at most 1 inst/cy/warp
= classical SIMT). That's the textbook answer, not a new finding.

## Recommendation

- **Do not promote V49/V50 to canonical** in `B300_TRUE_REFERENCE.md`.
- Cite as "preliminary; baseline under-saturated; needs warps-sweep + ncu
  verification".
- The headline dual-issue answer remains M8: it depends on the pipes.
  FFMA+LOP3 same-warp 55% is **plausible as a same-warp ceiling** but not
  a per-SM dispatch ceiling.
