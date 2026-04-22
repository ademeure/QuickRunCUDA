# Canonical Doc Doubt Report

**Target:** `/root/github/QuickRunCUDA/b300_clean/B300_CANONICAL_REFERENCE.md` (17,631 lines, v1, 2026-04-22).
**Adversarial pass date:** 2026-04-22.
**Method:** Read TOC + front matter (1-230). Sampled §3, §4, §5, §6, §9, §11 partial, §13, §16, §17, §21, §22, §27, §41, §44, §45, §50, §51, §56, §65 in full or near-full. Read **all** of Appendix A (lines 14089-14688), Appendix C (lines 15530-16270), and Appendix E (lines 16736-17574). Verified the stitching notes (lines 17577-17629) against the body. Cross-checked external claims against `CLAUDE.md`, `b300_clean/corrections/` directory listing, and Bash `grep` of suspect strings.

---

## Findings table

| # | Sev | Location | Issue |
|---|---|---|---|
| F1 | **HIGH** | line 297 (front matter) | Cross-ref **broken**: front matter says "Agent F's appendix §65 has the full zigzag case study". §65 is "Time-stamping + version" (line 14016). The zigzag is **Appendix A** (line 14089). Should read "Appendix A". |
| F2 | **HIGH** | line 1377 (§9 see-also) | Cross-ref **broken**: §9 see-also says "§65 (popcount synthesis, Agent F)". §65 is "Time-stamping + version", not popcount synthesis. Real popcount synthesis lives in §43 (line 8503). Should read "§43" (and drop "Agent F"). |
| F3 | **MED** | 29 occurrences across body | Pre-stitch sibling-agent labels leaked through. Strings like "Agent A/B/C/D/E/F", "Agent C", "Agent E §50", "Agent F's appendix" appear in: lines 290, 297, 310, 413, 452, 474, 559, 607, 1299, 1377, 1491, 1754, 1850, 1909, 2216, 2399, 2636, 3490, 3780, 3984, 4114, 4143, 4155, 4354-4355 (and probably more). The stitcher claimed authority over the doc but did NOT scrub these. Doesn't break any number, but signals "not fully proofread". |
| F4 | **MED** | line 14117 (Appendix A.1 chronology table) | Last cell of W6 row says "(potentially): could in principle be wrong if ncu metric semantics misinterpreted". Appendix A.11 then describes this caveat in detail and footgun E.P.5 lists it — internally consistent, but the table cell text reads like an unfinished editorial fragment. Cosmetic. |
| F5 | **LOW** | Appendix C (lines 15530-16270) vs stitching notes (17599-17608) | Stitching notes claim "6 preserved contradictions", and lists §22, §44, §7, §13, §32, §50. Appendix C only sketches **4 retests (V53/V54/V55/V56)** covering: §13 (V53), §32 (V54), §7 (implicitly via V55 HBM floor anchor — but §55 sketch only addresses HBM **read** denominator, NOT write provenance), §50 (V56). **§44 (M11 vs 16_power_clock 2× discrepancy) has NO retest sketch in Appendix C.** §22 doesn't need one (V52 already settled it). So strictly only 4 of the 6 contradictions have retest plans, and §44 is silently missing. |
| F6 | **LOW** | §65.1 (line 14030) | Front matter line 14030 says hardware is "8 HBM3E stacks of 12-Hi each". This is correct (matches §4) but the stitcher's stitching-notes claim about line 297 also conflates §65 and Appendix A. |
| F7 | **MED** | §22 vs §27 — pipe overlap matrix | §22.7 mentions M8 PIPE_OVERLAP_MATRIX should be re-measured with V8/V52-style methodology. §27.7 says "M8 confirmed by V52" via the matrix. These are not strictly contradictory — §27 says the **directional finding** is confirmed; §22 says the **numerical entries** still need re-measure. But a casual reader gets two different impressions. Worth a 1-line note in §27.7 acknowledging the §22.7 caveat. |
| F8 | **LOW** | §50 (NVFP4 A:B) vs Appendix E.N.2 | §50 is tagged 🟡 **MED** in TOC and section header. Footgun E.N.2 says "Wave-2 over-resolved A:B asymmetry to 'TMA multicast halves B's memory cost'. Underlying source itself lists 4 plausible mechanisms. Preserve all 3 readings". Body §50.4 ("Why ALL three are simultaneously 'right'") and §50.5 ("Practical guidance") both implicitly anchor on "B-reuse mechanism" as the real explanation, then §50.2.2 calls the 32-MAC cliff "the single most direct evidence" — that's a much stronger claim than 🟡 MED warrants. The MED tag is technically correct but the body trends HIGH-confident without the prefix-warning matching. |
| F9 | **LOW** | §44 (line 8842) | Headlined as "🟡 MED — UNRESOLVED". Body presents both numbers as "individually correct at their stated configuration" then immediately gives a TF/W ladder (line 8930) that quotes BOTH M11's 0.111 and 16_power_clock's 0.21 side-by-side without the disclaimer. A reader scrolling to the ladder will copy one or the other without the operating-point context. |
| F10 | **LOW** | §27.6 line 4831 | Says "the dual-issue verdict for B300 has flipped 5 times in the corrections cycle". This matches §22 and Appendix A. ✓ Internally consistent. |
| F11 | **MED** | §22 footgun #2 wording bug | Lines 3768-3771: "It is correct PER PIPE but each SM has multiple pipes that overlap, so total can reach ~256 inst/SM/cy (FFMA + LOP3 = 1 + 0.5 per SMSP × 4 SMSPs × 32 lanes = 192 inst/SM/cy in the V52 measured case)." The "~256 inst/SM/cy" claim then concrete-computes 192 inst/SM/cy. The "256" is the FFMA-only ceiling × 2 (free overlap), but in the V52 case LOP3 has 2-cy cadence → effective 192. Either drop "256" entirely (the 192 is what V52 actually measured) or specify "~256 max if both pipes ran 1 inst/cy". As written it conflates the two. Footgun E.P.2 in Appendix E quotes the same wording — same fix needed in both. |
| F12 | **LOW** | §43 vs §9 | §9 (line 1226) headlines popcount bell at d=16 with +240-554 W swing. §43 (line 8503) is "Power data-dependence — popcount bell curve, peak at d=16 random". Numbers consistent. §65 see-also reference (F2 above) should point to §43, not §65 itself. |
| F13 | **LOW** | §65 hardware spec (line 14031) vs §15 (PCIe) | §65 doesn't explicitly mention CUDA driver other than 580.126.09 — fine; cross-checks with §15 PCIe info. No contradiction. |
| F14 | **MED** | §50 (line 10322) header confidence vs front-matter TOC | TOC line 195 lists §50 as "🟡 MED". Section §50.3 table tags the K=96 single-kernel reading as 🟢 HIGH (line 10533) — that is per-row confidence, fine — but the **section** confidence is MED on a topic where one of the three readings is HIGH. A user querying "what's the A:B asymmetry?" needs to know the answer is conditionally HIGH for K=96 production case but DISPUTED across regimes. The way §50 is structured this is recoverable, but the TOC alone misleads. |
| F15 | **LOW** | §44 see-also (not present) | §44 on its own does NOT explicitly call out §43 (data dependence) or §41 (TDP). Cross-ref weakness; minor. |
| F16 | **MED** | Appendix A.6.6 (line 14470) | Same wording as §22 footgun #2 (F11 above): "total inst/SM/cy can reach ~256". Again, the V52 measured case is 192 inst/SM/cy, not 256. The 256 is the theoretical ceiling that *was not reached in V52*. Wording fix needed for parity. |
| F17 | **LOW** | §22 see-also "M8 PIPE_OVERLAP_MATRIX (now architecturally consistent, re-measure pending)" (line 3781) | Good; this is the right framing. Just §27.7 needs to match (see F7). |
| F18 | **LOW** | §65.1 line 14029 | "Driver: CUDA 13.2 V13.2.78 / Driver 580.126.09" — not cross-checked against §60. §60 ("Device props / nvml") would be the natural place to anchor driver version; the canonical-version block in §65.1 should not be the only place. |

---

## What survived doubt

These items I checked carefully and found **internally consistent and well-grounded**:

- **§22 dual-issue verdict + Appendix A zigzag.** Both reach the same V52-settled conclusion (alu+fma=147%, free overlap, V49/V50 retracted). Appendix A is a complete 5-level chronology with all four lessons. **Confidence:** HIGH placement is correct. Only the "256 vs 192" wording bug (F11/F16) is a real defect.
- **§5 / §6 HBM denominator handling.** §5 explicitly retires the 7.31 TB/s denominator and re-anchors V46 (98.5% → 93.8%). §6 quotes the corrected ladder. §56 (TMA) re-anchors against 7.67 this-device. §6 footgun, §56.3 callout, and Appendix E.B.4 all consistent. The cross-doc denominator drift is genuinely fixed in the canonical doc, with the historical artifact preserved.
- **§4 HBM 8-stacks.** Two-method confirmation (NVIDIA blog quote + cudaGetDeviceProperties output). Footgun E.B.1 correctly attributes the "12 stacks" claim to **other catalog files** (B300_TRUE_REFERENCE.md line 15, 01_hbm_bandwidth.md lines 3, 136), not to CLAUDE.md. ✓ I `grep`'d CLAUDE.md and confirmed it does NOT contain "12 stacks". The footgun does not falsely attribute to CLAUDE.md.
- **§45 -lgc 2032 paradox.** Quadruple-confirmed in 4 sources, full lock-state truth table, stuck-at-1005 separate failure mode covered. Footgun E.K.1, E.K.2, E.K.3, E.K.4 all consistent. **Solid.**
- **§9 HBM_DATA_DEPENDENCE.md superseded.** Explicit "Why HBM_DATA_DEPENDENCE.md is superseded" subsection (line 1362). Footgun E.B.6 and E.Z.1 both consistent. **Solid.**
- **§16 FFMA peak.** 76.96 theoretical, 74.62 measured at boost; -lgc 2032→1920 paradox flagged in body **and** footgun #2; 128 cores/SM (NOT 256) flagged in footgun #1. **Solid.**
- **CURIOSITY_LIST_V2 hash hallucination.** Flagged in footgun E.H.1 with "22/25 (88%) hashes hallucinated" matching CLAUDE.md memory `feedback_task_list_hashes`. **Captured.**
- **Appendix E footgun index.** Comprehensive, alphabetised, cross-references each callout back to a body section. The symptom-keyed lookup table at the end is well-built. Per-symptom entries match the body footguns I sampled.
- **Appendix A completeness.** Full 5-level zigzag including W6 V52 settlement (sections A.1 through A.12). Lessons (A.7), meta-lesson (A.8), what-V49/V50-should-have-done (A.9), ncu metrics with semantics (A.10), preserved doubt (A.11), cross-references (A.12). **Strongest single appendix in the doc.**
- **Appendix C retest sketches.** V53 (DSMEM fenced, settles §13), V54 (membar isolation, settles §32), V55 (HBM floor anchor, partly settles §6 denominator), V56 (NVFP4 A:B mechanism, settles §50). All four sketches have hypothesis matrices, kernel sketches, ncu metrics, predicted outcomes, decision rules, expected effort. **Production-quality.** Caveat: §44 power discrepancy and §7 HBM write provenance are NOT directly addressed (F5).

---

## What needs fixing (proposed edits)

Ordered by severity:

**Edit 1 (HIGH, F1).** Line 297, replace `Agent F's appendix §65` → `Appendix A`.

**Edit 2 (HIGH, F2).** Line 1377, change `§65 (popcount synthesis, Agent F)` → `§43 (popcount synthesis)`.

**Edit 3 (MED, F11+F16).** Lines 3768-3771 and line 14470, fix the "~256 inst/SM/cy" wording. Suggest: "Single-pipe IS capped at 128/SM/cy. With two pipes overlapping, the **theoretical** total is ~256 inst/SM/cy, but in the V52 measured FFMA+LOP3 case the LOP3 2-cy cadence makes the **actual** total 192 inst/SM/cy (1 FFMA + 0.5 LOP3 per SMSP × 4 SMSPs × 32 lanes)."

**Edit 4 (MED, F3).** Sweep the doc for "Agent A/B/C/D/E/F" and "Section [A-F]'s" tokens (29 occurrences). Either delete the parenthetical or substitute the actual section number. Doesn't change any number; cosmetic / signals proofread quality.

**Edit 5 (MED, F7).** §27.7 should add a 1-line note: "Numerical entries above are the M8 measurements; they should be **re-measured** with V8/V52-style 128-deep unroll to remove loop-overhead contamination (see §22.7). Directional finding (free overlap) is confirmed by V52."

**Edit 6 (LOW, F5).** Add a note to Appendix C.5 ("Recommended order") or as a new C.4.5 explicitly stating §44 (M11 vs 16_power_clock 2× discrepancy) is **NOT** in the V53-V56 backlog — it requires its own V57 sketch (joint power × ncu pipe utilization sweep at fixed frequency). Or actually add a V57 sketch.

**Edit 7 (LOW, F8/F14).** §50 in the TOC could be tagged `🟡 MED (regime-dependent: HIGH for K=96 production)` to better signal that the production-case answer (2.6× B>A) is well-grounded while the cross-regime synthesis is MED.

**Edit 8 (LOW, F9).** §44 TF/W ladder (line 8930) should label rows with operating-point context: "FFMA peak (high-ILP)" and "FFMA-bound mid-occ (M11)" — already present, but bold or callout the 2× gap so a casual reader doesn't quote 0.21 as "the" number.

---

## Confidence in the doc as a whole

**Verdict: USABLE AS-IS for HIGH-confidence quoting, but proofread is incomplete.**

Strengths (load-bearing):
- The dual-issue zigzag (§22 + Appendix A) is **definitive** — the 5-wave history is preserved, V52 is anchored, the methodology lessons are extracted.
- HBM denominator chaos (3 candidates: 7672/7.31/8.0) is **resolved** with dual-citation rule and clear retraction of 7.31-as-denominator.
- The clock-state paradox (-lgc 2032→1920 + stuck-at-1005) is **quadruple-confirmed** with a full truth table and detection protocol.
- Appendix E (footguns index) is **alphabetised and complete**, with a symptom-lookup table that lets a downstream agent match suspicious-looking numbers to known retractions.
- Appendix C (retest sketches) is **production-quality** with kernels, predictions, and decision rules.

Defects (none load-bearing for headline numbers):
- 2 broken §65 references (F1, F2) — cosmetic but embarrassing for a "100% verified cross-references" claim by the stitcher.
- 29 leaked sibling-agent labels (F3) — signals incomplete scrub.
- "~256 inst/SM/cy" off-by-arithmetic (F11/F16) — appears twice, both cases conflate theoretical ceiling vs measured V52 case.
- §44 has a 2× UNRESOLVED discrepancy that is acknowledged but not in the V53-V56 retest backlog (F5).

For an LLM agent quoting numbers from this doc: **the headline answers and confidence tags are trustworthy**. For a human reader doing methodology forensics: **the cross-references should be verified individually**, since at least 2 are broken and 29 carry leaked agent labels. Recommend a v1.1 patch with the 8 edits above before declaring "production-ready".

---

## Top 5 most concerning issues (ranked)

1. **Broken `§65` cross-references at line 297 and 1377** — the stitcher explicitly claimed "0 broken §N refs" but F1 and F2 are real. §65 is "Time-stamping", not the zigzag and not popcount.
2. **`~256 inst/SM/cy` wording bug (F11/F16)** — appears identically in §22 footgun #2 and Appendix A.6.6. Conflates theoretical ceiling (256) with V52 measured (192). A reader will quote one or the other without context.
3. **29 leaked "Agent A/B/C/D/E/F" labels (F3)** — signals incomplete proofread; doesn't change numbers but undermines the "single canonical doc" framing.
4. **§44 power discrepancy has no retest in Appendix C (F5)** — stitching notes claim 6 preserved contradictions, Appendix C only addresses 4 with sketches. §44 falls through; needs V57 or explicit acknowledgement.
5. **§50 NVFP4 A:B asymmetry tagged 🟡 MED but body trends HIGH-confident on B-reuse mechanism (F8/F14)** — the 32-MAC cliff is called "single most direct evidence", which is HIGH-grade language inside a MED-tagged section. Should either upgrade the per-row confidence for the B-reuse mechanism or downgrade the body language.
