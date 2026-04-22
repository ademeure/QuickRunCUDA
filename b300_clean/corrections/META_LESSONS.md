# META LESSONS — distilled wisdom from waves 1–6

**Date:** 2026-04-22
**Inputs:** the dual-issue 5-level zigzag (V49/V50 → W3b → W4 → W5a → V52),
the HBM denominator 4-iteration zigzag (W3 → W4 → W5b → W6b), and the
DSMEM/NVFP4/MUFU sub-zigzags.
**Purpose:** capture the *meta-pattern* of the rigor sweep, distinct from
the per-claim verdicts. If only one file from `b300_clean/corrections/` is
read by a future Claude or a future engineer, this is a strong candidate.

---

## The 5-level dual-issue zigzag (canonical case study)

| Wave | Verdict | Mechanism cited | What was right | What was wrong |
|---|---|---|---|---|
| **W1+W2** (V49/V50) | HIGH 55 % / 74 % "B300 ALU pipes share scheduler dispatch" | reproducibility of clock64 ratio | the measurement is reproducible | reproducibility ≠ validity; no SASS audit, no ncu, no matched solo baseline |
| **W3b doubt** | LOW | under-occupancy at 2 warps/SMSP × 8 ILP | numbers are unsafe | wrong specific mechanism; AND wrong implicit inference that the cap exists at 55 % |
| **W4 meta-doubt** | MED (re-promote) | V8 hits 97.6 % at the same 2 warps/SMSP, falsifying under-occupancy | right falsification of W3b's mechanism | conflated "two kernels at same occupancy" with "two kernels with same methodology" |
| **W5a SASS-verify** | LOW (re-downgrade) | inner body 8 ops/type + BRA + UIADD3/UISETP contaminates the ALU pipe being measured | right mechanism for the artifact | implicitly carried W3b's dispatch-cap inference forward; never tested whether the cap exists at all |
| **W6 V52 + ncu** | **HIGH (architectural truth) + RETRACT-NUMBER** | `pipe_alu + pipe_fma = 147 %` simultaneously per ncu | settled | (potentially): could in principle be wrong if ncu metric semantics misinterpreted |

**Reading:** the verdict moved 5 times. The actual measurement (V49 wall-clock) never changed. Only the **interpretive framework** changed. The claim was finally settled not by deeper armchair reasoning but by one careful empirical test that combined V8-style methodology AND a previously-unread ncu metric.

---

## The headline meta-lesson

> **Five waves of nested doubt converge slowly without an empirical anchor. One careful empirical test with the right diagnostic settles the question in a single shot.**

Doubt-the-doubt is valuable — each wave caught a real bug in the previous wave's reasoning, and stopping at any wave before W6 would have left the canonical reference incorrect (W1+W2 wrong about cap existing; W3b wrong about mechanism; W4 wrong about MED; W5a wrong about implied cap). But doubt-without-empirical-test is structurally limited:

- Each wave can only catch errors that are *visible* from the prior wave's evidence.
- A wave cannot rule out errors that require *new* evidence (e.g., ncu metrics nobody had collected).
- Architectural inferences hidden inside an artifact-detection argument tend to be inherited silently across waves.

The remedy is not "more doubt waves" — it is **"one wave with the right empirical anchor"**. V52 ran in roughly the time of one armchair doubt wave and yielded a decisive answer.

---

## Five mandatory rigor rules (derived from the zigzag)

These are the rules that, applied to V49 in W1+W2, would have prevented the entire 5-wave detour.

### Rule 1 — Inner body must amortize loop overhead (≥ 64 ops/type)

V49's 8 FFMA + 8 LOP3 inner body let BRA + UIADD3 + UISETP consume ~10–15 % of ALU dispatch slots. V8's 128-deep inner amortizes branch overhead by ~16×. Below 64 ops/type, a "dual-issue measurement" measures branch contamination as much as it measures dual-issue.

### Rule 2 — Solo and dual baselines must use IDENTICAL methodology

Same unroll depth, same `__launch_bounds__`, same warps/SMSP, same anti-DCE strategy. V49's solo FFMA ran at 67 % but V8's solo FFMA ran at 97.6 % at the same occupancy because the **inner body shapes** differed. Always run a **matched solo control** in the same harness as the dual test.

### Rule 3 — Always SASS-verify inner body composition before reporting %

`cuobjdump -sass` (or look at `sass/<basename>_<hash>.sass`) and count actual emitted instructions in the hot loop. If the inner body contains UIADD3 / UISETP / BRA / LDC / IMAD.MOV.U32 not part of the pipe being measured, those count against the dispatch budget. Source-level `#pragma unroll N` doesn't guarantee SASS-level unroll.

### Rule 4 — For dual-issue claims, the ONLY decisive metric is `pipe_X + pipe_Y` simultaneously from ncu

Wall-clock GLane/s ratios are NOT decisive — they confound dispatch with per-instruction issue cadence. (Solo LOP3's 2-cycle cadence at 16.8 K Glane/s does NOT mean the ALU pipe is at 50 %; ncu shows it's at 99.5 %.) The diagnostic signature of true dual-issue is `pipe_X.pct + pipe_Y.pct > 100 %`. The diagnostic signature of serial issue is `pipe_X.pct + pipe_Y.pct ≈ 100 %`. V49/V50 collected NEITHER metric.

### Rule 5 — Separate "the artifact is real" from "the architectural inference from the artifact is real"

These are TWO independent claims and can each be true or false independently:

- (a) Is the measured number trustworthy as published?
- (b) If not, what is the true architectural value?

W3b/W5a got (a) right (no, V49/V50 are contaminated) but silently inherited a wrong (b) (the cap exists at the artifact value). Rule: when downgrading a measurement, explicitly state whether you are also overturning the architectural inference, or merely flagging the number for re-test.

---

## Secondary lessons from the parallel HBM zigzag

The HBM denominator went through 4 iterations: 8 TB/s marketing → 7672 calculated → 7680 spec / 7672 ghost → 7.68 spec / 7.67 this-device (7680-bit fused AC SKU). The pattern is the same as dual-issue:

- Each wave found a real refinement.
- The bus-width revelation (7680, NOT 8192) required one CUDA query — no amount of armchair denominator debate would have surfaced it.
- The lesson reinforces Rule 4: **query the hardware directly**. `cudaGetDeviceProperties` is a 3-line program.

---

## The "doubt-the-doubt" pattern: useful but slow

Doubt-the-doubt converges to the right answer, but slowly and at the cost of intermediate publishable claims being wrong. The 5-wave dual-issue arc shows:

- Wave 1: published wrong (HIGH).
- Wave 2: published wrong (HIGH).
- Wave 3b: published right answer for wrong reason (LOW + under-occupancy).
- Wave 4: published wrong (MED).
- Wave 5a: published right answer for partly-wrong reason (LOW + loop-overhead contamination but inherited cap inference).
- Wave 6: published right (HIGH for architecture, RETRACT-NUMBER for V49/V50 specific values).

If a downstream consumer had read the catalog at any wave 1–5, they would have gotten a wrong reading. Only at wave 6 — driven by an empirical test — did the catalog stabilize.

**Practical implication:** prefer empirical tests over additional doubt waves whenever the next test is feasible (V52 took roughly one doubt-wave's worth of effort).

---

## The gold standard for B300 microbench claims

For any non-trivial architectural claim — e.g., "pipe X's SoL is Y" or "feature Z exists" — the catalog should require ALL of:

1. **Wall-clock measurement** with V8-style methodology (Rules 1–3).
2. **SASS verification** of the inner body composition.
3. **ncu pipe metrics** (Rule 4) — for dual-issue claims, BOTH pipes simultaneously.
4. **Multiple independent recipes** — at least 2 kernel variants that should give the same answer if the architectural claim is right.
5. **Reproduced 3× within 1 %** with `pkill -9 <bin> && sleep 5–8` between runs.

Anything less is provisional. The catalog should mark provisional rows as MED or LOW, not HIGH, even when the measurement looks clean.

---

## What could overturn V52 (preserved doubt)

V52 is the strongest evidence in the catalog, but is not infallible:

- ncu metric definitions are software-defined. If `smsp__pipe_alu_cycles_active` counts cycles where the pipe is *holding* an instruction (not just *issuing* one), then `alu + fma > 100 %` is consistent with serial issue at the dispatch port too. We have NOT verified the ncu metric definition against PTX-level event counters or a public NVIDIA spec.
- "Free overlap" was measured for FMA + ALU specifically. Other pipe combinations (LSU + tensor, MUFU + FMA in non-LOP3 setting, etc.) are NOT settled by V52.
- The 2-cycle LOP3 cadence is inferred from `inst_issued/cy = 0.51`. An alternative explanation is "1-cycle issue but 50 % stall on RF read port". V52 cannot distinguish them; both predict the same `inst_issued` and `pipe_alu`.
- If V52's GLane/s reading drifts > 1 % across runs in future re-tests, the methodology may have a yet-undetected issue.

What would overturn V52: an ncu metric-definition bug for sm_103a, an alternative interpretation of `pipe_X_cycles_active`, or a clean test where `alu + fma` reproducibly stays at ≤ 100 % under V8-style methodology with an alternate recipe. None are expected.

---

## TL;DR — three takeaways

1. **Test, don't theorize.** 5 waves of armchair doubt vs 1 careful empirical test. The test won.
2. **ncu pipe metrics + SASS + multiple-recipe baseline = the gold standard.** Wall-clock alone is not enough for architectural claims about dispatch / pipes / overlap.
3. **Distinguish "the number is artifact" from "the architecture is at the artifact value".** Conflating them propagates errors silently across doubt waves.
