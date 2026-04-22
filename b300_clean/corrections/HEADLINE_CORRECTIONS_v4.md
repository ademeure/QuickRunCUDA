# HEADLINE CORRECTIONS — v4 (wave-5)

**Date:** 2026-04-22
**Supersedes:** `HEADLINE_CORRECTIONS_v3.md` (wave-4)
**Drivers:** SASS_VERIFY_DUAL_ISSUE.md (5a), HBM_DENOMINATOR_FINAL.md (5b),
PROPOSED_FIXES.md (5c), CROSS_LINK_AUDIT.md (5e).

> **Caveat up front:** v4 is the **4th iteration** of these top-line claims.
> Each prior iteration found a real issue the previous missed. Further
> verification — especially V52 (dual-issue with 128-deep unroll + dual ncu
> pipe metrics) — could overturn or refine v4 in turn. Trust this file as the
> CURRENT best, not the FINAL word.

---

## Top-line corrections (12 entries; v3 → v4 deltas marked)

| # | Topic | v3 said | v4 says | Confidence | Wave |
|---:|---|---|---|---|---|
| 1 | HBM3E stack count | 8 stacks of 12-Hi (raw 288 GB) | **8 stacks of 12-Hi** — confirmed by **bus width** (8192-bit / 1024-b per stack), not capacity | HIGH | W4+W5b |
| 2 | "12 stacks" error location | CLAUDE.md + B300_TRUE_REFERENCE.md + 01_hbm_bandwidth.md | **NOT in CLAUDE.md** (W4 hallucination — `grep` proved no "stack" string). Real locations: B300_TRUE_REFERENCE.md L15, 01_hbm_bandwidth.md L3 + L136 | HIGH | W5c |
| 3 | HBM denominator | 7680 canonical; **7672 is "arithmetic ghost"** | **Cite BOTH**: 7680 (spec, 8.000 Gbps/pin, comparable across vendors) AND 7672 (this-device actual at 3996 MHz I/O measured via `nvidia-smi -q`). The 0.10 % gap is real silicon under-spec, not arithmetic noise | HIGH | W5b |
| 4 | HBM SoL recipe (V46 TMA 8-deep) | 7.20 TB/s = 93.75 % of 7680 / 93.85 % of 7672 | unchanged; both denominators valid | HIGH | W4+W5b |
| 5 | HBM best measured (per-warp coalesced) | 7.30 TB/s = 95.0 % of 7680 / 95.2 % of 7672 | unchanged; both denominators valid | HIGH | W4+W5b |
| 6 | HBM ECC reservation | 1/16 SECDED → 7680 from 8192 raw | unchanged | HIGH | W4 |
| 7 | V49/V50 dual-issue | **MED** (REVERSED from v2 LOW; under-occupancy hypothesis falsified by V8_FFMA at same 2 warps/SMSP) | **LOW (RE-DOWNGRADED, 3rd time)** — SASS analysis (W5a) shows V49 inner body is 8 FFMA + 8 LOP3 + BRA + UIADD3/UISETP. Loop-overhead contamination, not under-occupancy, not RAW. V8 unrolls 128-deep so amortizes 16×. Architectural question (does B300 SMSP dual-issue FMA + ALU?) **remains OPEN**. V52 retest required. | LOW | W5a |
| 8 | V51 multi-stream HBM | Bug-fix in progress; treat all V51 as suspect | unchanged | LOW | W4 |
| 9 | NVFP4 K=96 A:B asymmetry (3 readings) | MED — 3 framings retained pending V56 | unchanged | MED | W4 |
| 10 | DSMEM read 40 GB/s, write 560 GB/s, ring "no shared bus" | LOW/LOW-MED — chain-bound / issue-rate-only / under-issued | unchanged | LOW | W4 |
| 11 | `__threadfence_system` 1.74× spread | LOW — DOWNGRADED in W3b, no new evidence | unchanged | LOW | W3b |
| 12 | HBM write SoL 7.57 TB/s NINJA | MED — provenance unclear, beats memset by 5 % | unchanged; flag for V53 | MED | W3b |

---

## Methodology rules (1–12; rule 11 SOFTENED, rule 12 NEW)

1. **Sanity-check measured against theoretical FIRST.** If measured > theoretical, test is broken (DCE, formula bug, clock mismatch).
2. **State clock state.** "Default boost" = 2032 MHz; `-lgc 2032` paradoxically pins 1920 MHz. Mixing is ~6 % noise.
3. **B300 has 128 FP32 cores per SM, NOT 256.** Dual-issue claims doubling this are wrong.
4. **DCE defenses:** unconditional STG of result, runtime-input-derived loop values, kernel runtime ≥ 1 ms.
5. **Self-op chains inflate latency 2×.** Use distinct sources for true throughput.
6. **Launch-overhead-dominated tests:** ensure ≥ 10 ms runtime for peak measurements.
7. **SASS-verify.** Source-level inline asm does NOT determine SASS — compiler hoists IMMs, applies `.reuse`, etc. Always check compiled SASS.
8. **Cross-check with ncu** where available (e.g., `pipe_fma.avg.pct_of_peak_sustained_active`).
9. **One PTX path failing ≠ HW capability absent.** Check all paths (mma.sync vs tcgen05.mma).
10. **Pair Gops/s with bytes/s.** Cache-line combining can inflate Gops without proportional BW.
11. **HBM denominator: cite BOTH 7680 (spec, 8.000 Gbps/pin) AND 7672 (this-device, 3996 MHz I/O empirical).** Default to 7680 for cross-vendor comparison; pair with 7672 when SoL precision matters. **Never call 7672 a "ghost"** (W3 v3 rule overruled by W5b). The 0.10 % gap is real silicon under-spec.
12. **(NEW) Loop-overhead contamination is its own pitfall.** Tiny inner bodies (< 32 ops) contaminate dual-issue / pipe-utilization measurements because branch + loop-counter (UIADD3 / UISETP) consume scheduler slots on the same ALU pipe. Fix: unroll inner body ≥ 64 ops per type, use `__launch_bounds__(N, 1)`, anti-DCE via STG (not clock-diff conditional), AND read both `pipe_fma` + `pipe_alu` from ncu simultaneously.

---

## Lessons from the dual-issue 4-level zigzag (NEW META-RULE)

The dual-issue verdict has flipped four times: HIGH → LOW → MED → LOW. Each layer of doubt found a real issue the prior missed:

| Layer | Found | Missed |
|---|---|---|
| W1+W2 measured 55 % / 74 % | The numbers are reproducible | Measurement methodology bug |
| W3b doubt | Numbers are unsafe | Wrong specific mechanism (under-occupancy) |
| W4 meta-doubt | W3b's specific mechanism is falsified | Did not look at SASS; assumed reproducibility = MED |
| W5a SASS-verify | The actual mechanism (loop-overhead contamination + tiny body) | TBD — may itself be wrong |

**Meta-rule:** When a verdict has flipped ≥ 2×, do NOT publish as HIGH or even MED until a fresh test (V52) eliminates ALL hypothesized failure modes simultaneously. Publish as LOW with the OPEN question explicit.

---

## Files modified vs unmodified by wave-5

**Modified by wave-5:** none (this wave produces synthesis files only).

**Unapplied diffs (from PROPOSED_FIXES.md / W5c, awaiting user accept):**
- `b300_clean/B300_TRUE_REFERENCE.md` L15 (12 → 8 stacks), L25 (7672 spec → post-ECC 7680)
- `b300_clean/01_hbm_bandwidth.md` L3 (12 → 8), L136 (full theoretical-accounting rewrite)
- (Optional) `CLAUDE.md` L85 — enrichment only (add 8-stack annotation), NOT a fix

**Stale and needing patch (per W5e + W5d):**
- `CONFIDENCE_LADDER_PATCH.md` §2 — wave-5d promoted dual-issue MED; W5a re-downgrades to LOW
- `HEADLINE_CORRECTIONS_v3.md` rule 11 — superseded by rule 11 in this file
- `WAVE4_CHANGES.md` §2 + §6 — "7672 is ghost" framing superseded by W5b
- `MASTER_INDEX_v2.md` L8 — points at v2 HEADLINE; should point at v4

---

## Single recommended entry point

**`HEADLINE_CORRECTIONS_v4.md`** (this file). Then in order:
1. This file (top-line + methodology rules)
2. `WAVE5_CHANGES.md` (delta vs wave-4)
3. `HBM_DENOMINATOR_FINAL.md` (rule 11 derivation)
4. `SASS_VERIFY_DUAL_ISSUE.md` (rule 7 + rule 12 + V52 sketch)
5. `META_DOUBT_REPORT.md` (W3b → W4 reasoning)
6. `WAVE4_CHANGES.md` (caveat: §2 + §6 superseded by HBM_DENOMINATOR_FINAL)
7. `UNRESOLVED_PROMOTED.md` (V52–V56 retest queue)
8. `CONFIDENCE_LADDER.md` (303-row table; apply 4 dual-issue LOW + HBM dual-cite patches inline)

---

## What v4 readers should believe right now

- **HBM3E:** 7.30 TB/s best measured; 95 % of 7680 spec post-ECC ≈ 95 % of 7672 this-device actual. **HIGH confidence.**
- **HBM3E stack physics:** 8 stacks × 12-Hi × 3 GB/die × 1024-b/stack at 3996 MHz I/O. **HIGH confidence**, sourced from NVIDIA's bus-width disclosure.
- **Dual-issue FMA + ALU:** verdict **OPEN**. The 55 %/74 % numbers are measurement artifacts. Whether B300 SMSP can dual-issue is **unanswered** until V52. **LOW confidence.**
- **Everything else:** unchanged from v3.

---

## Acknowledged risk that v4 is also wrong

This is iteration 4 of the dual-issue claim. SASS analysis (W5a) is structurally sounder than source-pattern analysis (W4) which is structurally sounder than verdict-by-reproducibility (W1+W2). But until V52 actually runs with the prescribed methodology AND ncu confirms `pipe_fma + pipe_alu ≈ 200 %`, even W5a's "loop-overhead" diagnosis is hypothesis, not measurement. v5 may exist.
