# HEADLINE CORRECTIONS — v5 (wave-6)

**Date:** 2026-04-22
**Supersedes:** `HEADLINE_CORRECTIONS_v4.md` (wave-5)
**Drivers:** V52_RUN_RESULTS.md (W6a empirical settlement), HBM_STACKS_INDEPENDENT_VERIFY.md (W6b).

> **Caveat up front:** v5 is the **5th iteration** of these top-line claims. The
> dual-issue verdict has now been settled by an empirical ncu measurement (not
> just SASS reasoning), but ncu metrics themselves can be misinterpreted. Trust
> v5 as the CURRENT best with strong empirical anchor — but not infallible.

---

## Top-line corrections (12 entries; v4 → v5 deltas marked)

| # | Topic | v4 said | v5 says | Confidence | Wave |
|---:|---|---|---|---|---|
| 1 | HBM3E stack count | 8 stacks of 12-Hi (bus-width-confirmed 8192-bit) | **8 stacks of 12-Hi physically; this device's bus width is 7680-bit** (yield SKU, 1/16 controllers fused off; "AC" suffix in part name) | HIGH | W6b |
| 2 | "12 stacks" error location | NOT in CLAUDE.md; in B300_TRUE_REFERENCE.md L15 + 01_hbm_bandwidth.md L3 + L136 | unchanged | HIGH | W5c |
| 3 | HBM denominator | Cite BOTH 7680 (spec) AND 7672 (this-device at 3996 MHz) | **Cite BOTH 7.68 TB/s (spec-comparable: 8.000 Gbps × 8192 bits ÷ 1.0625 ECC) AND 7.67 TB/s (this-device: 7.992 Gbps × 7680 bits ÷ 1.0625)**. The two numbers happen to be near-identical because the 7680-bit width × +0.10 % per-pin tracking nearly cancels — coincidental, not arithmetic. | HIGH | W6b |
| 4 | HBM SoL recipe (V46 TMA 8-deep) | 7.20 TB/s = 93.75 % of 7680 / 93.85 % of 7672 | **7.20 TB/s = 93.9 % of 7.67 TB/s this-device peak** (re-anchored vs full-bus 7.68 spec); previously cited as "88.8 % of 8 TB/s marketing" — the higher per-cent is the right one for SoL on this part | HIGH | W6b |
| 5 | HBM best measured | 7.30 TB/s = 95.0 % of 7680 / 95.2 % of 7672 | **7.30 TB/s = ~95 % of 7.67 TB/s this-device peak**; framing simplified | HIGH | W6b |
| 6 | HBM ECC reservation | 1/16 SECDED → 7680 from 8192 raw | unchanged | HIGH | W4 |
| 7 | **V49/V50 dual-issue** | **LOW (RE-DOWNGRADED, 3rd time)** — methodology contamination, V52 retest required | **HIGH (architectural truth) + RETRACTED (numbers)** — V52 ncu shows `pipe_alu = 98.0 %` AND `pipe_fma = 49.4 %` simultaneously (sum = 147 %), proving free pipe overlap. The 55 % / 74 % from V49/V50 are confirmed measurement artifacts (loop-overhead contamination); the architectural inference of a "shared dispatch cap" was ALSO wrong. **The pipes overlap freely; LOP3 just has a 2-cycle issue cadence.** | HIGH | W6a |
| 8 | V51 multi-stream HBM | Bug-fix in progress; treat all V51 as suspect | unchanged | LOW | W4 |
| 9 | NVFP4 K=96 A:B asymmetry (3 readings) | MED — 3 framings retained pending V56 | unchanged | MED | W4 |
| 10 | DSMEM read 40 GB/s, write 560 GB/s, ring "no shared bus" | LOW/LOW-MED — chain-bound / issue-rate-only / under-issued | unchanged | LOW | W4 |
| 11 | `__threadfence_system` 1.74× spread | LOW | unchanged | LOW | W3b |
| 12 | HBM write SoL 7.57 TB/s NINJA | MED — provenance unclear | unchanged; flag for V53 | MED | W3b |
| **NEW 13** | **M8 PIPE_OVERLAP_MATRIX** (MUFU+FFMA ≈ 100 %, HMMA+LDS ≈ 73-96 %) | wave-3b: "counter-evidence to dispatch-cap reading" but unverified | **CONFIRMED by V52 + first-principles**: pipes overlap freely on B300 SMSP, so M8's overlap claims are architecturally consistent. M8 was right; the dispatch-cap critique of M8 was wrong. | HIGH | W6a |

---

## Methodology rules (1–13; rule 13 NEW)

1. **Sanity-check measured against theoretical FIRST.**
2. **State clock state.** Default boost = 2032 MHz; `-lgc 2032` paradoxically pins 1920 MHz.
3. **B300 has 128 FP32 cores per SM, NOT 256.**
4. **DCE defenses:** unconditional STG of result, runtime-input-derived loop values, kernel runtime ≥ 1 ms.
5. **Self-op chains inflate latency 2×.**
6. **Launch-overhead-dominated tests:** ensure ≥ 10 ms runtime for peak measurements.
7. **SASS-verify.**
8. **Cross-check with ncu** where available.
9. **One PTX path failing ≠ HW capability absent.**
10. **Pair Gops/s with bytes/s.**
11. **HBM denominator: cite spec (7.68 TB/s) AND this-device (7.67 TB/s, accounting for the 7680-bit fused bus on AC SKU).** Default to the this-device value for SoL claims; pair with spec for cross-vendor.
12. **Loop-overhead contamination:** unroll inner body ≥ 64 ops per type, anti-DCE via STG, dual-pipe ncu read.
13. **(NEW) For dual-issue / pipe-overlap claims, the ONLY decisive metric is `smsp__pipe_X_cycles_active.pct + smsp__pipe_Y_cycles_active.pct` simultaneously read from a single ncu profile.** If sum > 100 %, pipes overlap. If sum ≤ 100 %, claim either dispatch sharing OR that one pipe was idle (read `smsp__inst_issued.avg.per_cycle_active` to disambiguate). Wall-clock GLane/s ratios are NOT decisive — they confound dispatch with per-instruction issue cadence (e.g. LOP3's 2-cycle).

---

## Lessons from the dual-issue 5-level zigzag (META-RULE updated)

The dual-issue verdict has flipped five times: HIGH → LOW → MED → LOW → HIGH (now empirically settled).

| Layer | Found | Missed |
|---|---|---|
| W1+W2 measured 55 % / 74 % | Reproducible | Loop-overhead contamination |
| W3b doubt | Numbers are unsafe | Wrong mechanism (under-occupancy) AND wrong inference (cap exists) |
| W4 meta-doubt | W3b's mechanism falsified | Did not look at SASS or ncu pipes |
| W5a SASS-verify | Real mechanism (loop-overhead) | Implicitly preserved the dispatch-cap inference |
| **W6 V52 + ncu** | **Pipes overlap freely (alu+fma = 147 %); the dispatch cap was a phantom** | TBD — ncu metrics could in principle be misinterpreted |

**Meta-rule:** "the artifact is real" and "the architectural inference from the artifact is real" are TWO independent claims. W3b/W5a conflated them. Always state separately:
- (a) Is the measured number trustworthy as published?
- (b) If not, what is the true architectural value?

W3b/W5a got (a) right (no, V49/V50 are contaminated) and (b) wrong (the cap doesn't exist at the contaminated value; it doesn't exist at all).

---

## What v5 readers should believe right now

- **HBM3E:** ~7.20 TB/s sustained = ~94 % of this-device peak (~7.67 TB/s); ~93 % of spec-comparable peak (~7.68 TB/s). 8 stacks present; one /16 controller fused on this AC SKU. **HIGH confidence.**
- **Dual-issue FMA + ALU:** **YES, pipes overlap freely.** `alu + fma ≈ 147 %` per V52 ncu. The 55 %/74 % from V49/V50 are RETRACTED as artifacts. **HIGH confidence (empirical ncu anchor).**
- **M8 PIPE_OVERLAP_MATRIX (MUFU+FFMA, HMMA+LDS):** consistent with V52, **CONFIRMED**.
- **Everything else:** unchanged from v4.

---

## Acknowledged risk that v5 is also wrong

This is iteration 5. The empirical ncu anchor is the strongest evidence to date — `pipe_alu` and `pipe_fma` are documented metrics whose sum > 100 % is unambiguously dual-issue. But:

- ncu metrics are themselves software-defined; if NVIDIA's metric calculation has a bug for sm_103a, our reading would be wrong. We have NOT verified the ncu metric definition against PTX-level event counters.
- "Free overlap" applies to FMA + ALU specifically. Other pipe combinations (e.g., LSU + tensor) are NOT settled by V52.
- The 2-cycle LOP3 cadence is inferred from `inst_issued/cy = 0.51`; an alternative explanation is "1-cycle issue but 50 % stall on RF read port". The two predict the same `inst_issued` and the same `pipe_alu` so V52 cannot distinguish them.

What could overturn v5: an ncu metric-definition bug for sm_103a, an alternative interpretation of `pipe_X_cycles_active`, or a clean test where `alu + fma` reproducibly stays at ≤ 100 % under V8-style methodology. None of these are expected, but all are non-zero.

---

## Single recommended entry point

**`HEADLINE_CORRECTIONS_v5.md`** (this file). Then in order:
1. This file (top-line + methodology rules)
2. `WAVE6_CHANGES.md` (delta vs wave-5)
3. `V52_RUN_RESULTS.md` (the empirical anchor)
4. `HBM_STACKS_INDEPENDENT_VERIFY.md` (rule 11 derivation)
5. `META_LESSONS.md` (5-level zigzag distilled wisdom)
6. `CONFIDENCE_LADDER_PATCH_v3.md` (delta vs original 303-row ladder)
7. `CONFIDENCE_LADDER.md` (apply v3 patch inline)
