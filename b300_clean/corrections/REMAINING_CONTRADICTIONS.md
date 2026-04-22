# Remaining contradictions after wave-7

**Date:** 2026-04-22
**Scope:** What is still UNRESOLVED after wave-7 sub-agents (V53, V54, canonical-doc-doubt, citation-verify) finish.
**Predecessors:** wave-6 preserved 6 contradictions; V53 settled #2 (DSMEM), V54 settled #3 (membar.sys). 4 of the 6 wave-6 contradictions remain. Wave-7 doc-doubt also surfaced 4 new defects (#6-#9).

---

## Summary table

| # | Topic | Source wave | Severity | Settler needed | Estimated effort |
|---:|---|---|---|---|---|
| 1 | HBM 7.57 TB/s NINJA write provenance (NINJA STG vs TMA bulk) | W3b → W6 → W7 | MED | V55 — HBM floor empirical (sketched in canonical Appendix C, not yet run) | 0.5 day |
| 4 | per-pipe power M11 vs 16_power_clock 2× (TF/W: 0.111 vs 0.21) | W3b/W4 → preserved | MED | **V57 — joint power × ncu pipe utilization sweep (NEW sketch needed; NOT in Appendix C)** | 1 day |
| 5 | NVFP4 K=96 A:B 3-way reading (B-reuse mechanism vs alt) | W4 → preserved | MED | V56 — NVFP4 A:B mechanism kernel (sketched in Appendix C, not yet run) | 0.5-1 day |
| 6 | §44 missing retest sketch in canonical Appendix C | W7 doc-doubt | LOW (process) | Add V57 sketch to Appendix C OR explicit "no retest planned" note | 1 hour |
| 7 | "~256 inst/SM/cy" wording bug (§22 footgun #2 + Appendix A.6.6) | W7 doc-doubt | MED (cosmetic but misleading) | Edit per `CANONICAL_PATCH_v1_1.md` Edits 7+8 | 5 min |
| 8 | 29 leaked "Agent A/B/C" labels across the canonical doc | W7 doc-doubt | MED (cosmetic) | Edit per `CANONICAL_PATCH_v1_1.md` Edit 9 (sweep + scrub) | 30 min |
| 9 | §50 NVFP4 body trends HIGH but tagged 🟡 MED | W7 doc-doubt | MED | Edit per `CANONICAL_PATCH_v1_1.md` Edit 11 (soften body OR per-row promote B-reuse) | 15 min |

---

## Detail per contradiction

### #1 — HBM 7.57 TB/s NINJA write provenance

**Status:** OPEN. Carried since wave-3b.

**What's contradictory:**
The "7.57 TB/s NINJA write" figure appears in the catalog with unclear provenance. Two candidate paths:
- (a) **NINJA STG** (Mr-NINJA's persistent-thread STG benchmark, vector stores)
- (b) **TMA bulk store** (cp.async.bulk to global)

These two paths have meaningfully different SoL implications because TMA bulk has its own anti-DCE story and HBM-channel-saturation pattern that differs from naive STG.

**Current best understanding:**
HBM3E has a true read peak of ~7.20-7.30 TB/s sustained on this device (95-96% of 7.67 this-device peak). Write peak is **expected** to be similar but has not been measured under the same denominator framework. The 7.57 figure exceeds the 7.30 read peak by ~3.7%, which is plausible-but-suspicious — likely due to either (a) the figure being measured against an inflated denominator, (b) the figure including some L2-write-coalescing inflation, or (c) writes genuinely peaking slightly higher than reads on HBM3E.

**What would settle it:**
**V55 — HBM floor empirical** (sketched in canonical Appendix C, not yet run). Should:
- Run NINJA STG and TMA bulk store side-by-side under identical denominator (this-device 7.67 TB/s)
- Use cudaEvents to ensure full kernel-drain timing (not in-kernel clock64)
- ncu cross-check with `dram__bytes_write.sum / time` to compute true HBM-write SoL
- Compare to read-side (V46 7.20 TB/s)

**Severity:** MED. The 7.57 number is in the catalog as MED-confidence; if V55 confirms it, promote to HIGH. If V55 contradicts it (likely lands at ~7.20 TB/s like reads), retract the 7.57 in favor of "writes peak ~7.20 TB/s, indistinguishable from reads at SoL".

---

### #4 — per-pipe power M11 vs 16_power_clock 2× discrepancy

**Status:** OPEN. Carried since wave-3b/wave-4. **Has NO retest sketch in canonical Appendix C** — silently missing per `CANONICAL_DOC_DOUBT_REPORT.md` finding F5.

**What's contradictory:**
- M11 (per-pipe power synthesis) reports **0.111 TF/W** for FFMA-bound mid-occupancy.
- `16_power_clock_CORRECTED.md` reports **0.21 TF/W** for FFMA peak (high-ILP).

These are 2× apart. Both are presented as "correct at their stated configuration" in canonical §44, but the ladder at line 8930 quotes BOTH side-by-side without disclaimer — a casual reader copies one or the other.

**Current best understanding:**
Both are likely correct at their respective operating points:
- 0.21 TF/W = high-ILP FFMA peak where pipe utilization is ~98% and power is dominated by SMSP active count
- 0.111 TF/W = FFMA-bound but mid-occupancy where dispatch is partial and idle-power-per-active-FLOP is amortized differently

But this is a hypothesis. Neither is anchored against a clean joint sweep where ncu pipe utilization AND nvml power are co-measured at fixed frequency.

**What would settle it:**
**V57 — joint power × ncu pipe utilization sweep at fixed freq** (NEW sketch needed; **NOT in Appendix C**). Should:
- Lock clock to 1500 MHz via `nvidia-smi -lgc 1500` (avoids the -lgc 2032→1920 paradox per §45)
- Sweep ILP ∈ {1, 2, 4, 8, 16, 32}, occupancy ∈ {25%, 50%, 75%, 100%} of max blocks
- For each cell: ncu `smsp__pipe_fma_cycles_active.pct` + `smsp__pipe_alu_cycles_active.pct`, nvml `nvmlDeviceGetPowerUsage`, FFMA throughput from clock64
- Plot TF/W vs (ILP × occupancy)
- Identify which cell M11 measured (low-ILP or low-occupancy), which cell 16_power_clock measured (high-ILP, high-occupancy)
- Reconcile

**Severity:** MED. Until V57 runs, both 0.111 and 0.21 TF/W remain defensible at their stated configs but neither is "the" answer. Risk of casual readers quoting one without context.

**Action needed:** Add V57 sketch to Appendix C OR add explicit "no retest planned, both numbers correct in their regime" note to §44 (per `CANONICAL_PATCH_v1_1.md` Edit 10).

---

### #5 — NVFP4 K=96 A:B 3-way reading

**Status:** OPEN. Carried since wave-4.

**What's contradictory:**
For NVFP4 K=96 cuBLAS workloads, three plausible mechanisms explain the A:B asymmetry (~2.6× B>A power impact):
- (a) **TMA multicast halves B's memory cost** — B-side dedup via cluster broadcast
- (b) **B-reuse via tcgen05 K-row pairwise dedup** — internal hardware sharing
- (c) **A-side zero-skip when A=const+0** — explains the -50W signature when A is degenerate

Footgun E.N.2 explicitly preserves all 3 readings ("Preserve all 3 readings"). But canonical §50 body language ("the single most direct evidence", "Why ALL three are simultaneously right") implicitly anchors on (b) the B-reuse mechanism.

**Current best understanding:**
Per `project_b300_nvfp4_k96_signature.md` (user memory): A:B impact ~1:3, sign-bit alone 80W, N-64 is multiplier lane stride, A=const+0 triggers zero-skip (-50W), 1 outlier per K16 = +30W. The K=96 production case (large rect N) shows 2.6× B>A which is consistent with (b), but the signature elements are also consistent with (a) and (c) at different operating points.

**What would settle it:**
**V56 — NVFP4 A:B mechanism kernel** (sketched in canonical Appendix C, not yet run). Should:
- Construct test kernels that isolate each mechanism: (a) varying TMA multicast factor, (b) varying B-row reuse pattern, (c) varying A-side zero density
- Co-measure cuBLAS K=96 power via nvml across all three sweeps
- Decision rule: which mechanism, when isolated, reproduces the most of the observed 2.6× swing?

**Severity:** MED. The K=96 production-case answer (2.6× B>A) is well-grounded; the cross-regime synthesis is genuinely MED. After V56, expect to either promote (b) to HIGH and demote (a)/(c), or keep all 3 at MED with each anchored to its operating point.

---

### #6 — §44 missing retest sketch in canonical Appendix C

**Status:** OPEN. Process defect surfaced by W7 doc-doubt (F5).

**What's contradictory:**
Stitching notes claim "6 preserved contradictions" with retest sketches in Appendix C. Appendix C only has sketches for 4 (V53/V54/V55/V56). **§44 (per-pipe power M11 vs 16_power_clock) is silently missing.** §22 doesn't need a sketch (V52 already settled it).

**Current best understanding:**
This is a process gap, not a measurement defect. The sketcher omitted §44 — likely because the contradiction is hard to characterize (operating-point-dependent rather than methodology-dependent) and the V57 sketch is genuinely harder to write than V53-V56 (requires joint power + ncu, fixed-freq protocol).

**What would settle it:**
Either:
- (a) Add a V57 sketch to Appendix C (per #4 above)
- (b) Add explicit "no V57 sketch; UNRESOLVED 🟡 status maintained" note to §44 (per `CANONICAL_PATCH_v1_1.md` Edit 10)

**Severity:** LOW (process/documentation defect, not measurement defect).

---

### #7 — "~256 inst/SM/cy" wording bug

**Status:** OPEN. Surfaced by W7 doc-doubt (F11/F16).

**What's contradictory:**
Lines 3768-3771 (§22 footgun #2) and line 14470 (Appendix A.6.6) both say "total inst/SM/cy can reach ~256" then concrete-compute 192 inst/SM/cy. The 256 is the theoretical ceiling (both pipes at 1 inst/cy) but the V52 measured case is 192 (LOP3 at 2-cy cadence → 0.5 inst/cy). The wording conflates ceiling and measurement.

**Current best understanding:**
Both numbers are correct at what they describe; the **wording** is the bug. A reader quotes "256" out of context and reports it as a measured peak, or quotes "192" and reports it as the ceiling.

**What would settle it:**
Edit per `CANONICAL_PATCH_v1_1.md` Edits 7+8: separate "**theoretical** 256 (both pipes at 1 inst/cy)" from "**actual** 192 (V52 measured, LOP3 at 2-cy)".

**Severity:** MED (cosmetic but actively misleading).

---

### #8 — 29 leaked Agent A/B/C labels

**Status:** OPEN. Surfaced by W7 doc-doubt (F3).

**What's contradictory:**
Lines 290, 297, 310, 413, 452, 474, 559, 607, 1299, 1377, 1491, 1754, 1850, 1909, 2216, 2399, 2636, 3490, 3780, 3984, 4114, 4143, 4155, 4354-4355 (29 occurrences) carry "Agent A/B/C/D/E/F" or "Section X's Agent" tokens that survived stitching. The stitcher claimed authority over the doc but did NOT scrub these.

**Current best understanding:**
Cosmetic. Doesn't change any number. Signals incomplete proofread. Two of the 29 (lines 297, 1377) are also broken cross-refs (#1, #2 in doc-doubt) — those are both wording AND link defects.

**What would settle it:**
Edit per `CANONICAL_PATCH_v1_1.md` Edit 9: sweep + scrub. Either delete the parenthetical or substitute the actual section number.

**Severity:** MED (cosmetic but undermines "single canonical doc" framing).

---

### #9 — §50 NVFP4 body↔TOC tag mismatch

**Status:** OPEN. Surfaced by W7 doc-doubt (F8/F14).

**What's contradictory:**
§50 is tagged 🟡 MED in TOC line 195 and section header. But §50.2.2 calls the 32-MAC cliff "the single most direct evidence", §50.4 says "Why ALL three are simultaneously 'right'", §50.5 anchors practical guidance on B-reuse mechanism. These are HIGH-confidence framings under a MED tag.

**Current best understanding:**
The MED tag is technically correct because the cross-regime synthesis is genuinely uncertain (see #5 above). But the body language sets a HIGH-confident expectation that doesn't match the prefix warning. A user querying "what's the A:B asymmetry?" needs to know the answer is conditionally HIGH for the K=96 production case but DISPUTED across regimes.

**What would settle it:**
Two options:
- (a) **Soften body language** per `CANONICAL_PATCH_v1_1.md` Edit 11 — "single most direct" → "one of the strongest pieces of evidence", "Why all three are right" → "How three readings can each be valid at different operating points"
- (b) **Per-row promote** — keep §50.3 K=96 single-kernel reading at 🟢 HIGH, downgrade only the cross-regime synthesis rows to 🟡 MED, update TOC to "🟡 MED (regime-dependent: HIGH for K=96 production)"

Option (b) is more informative; option (a) is more conservative. Both are improvements.

**Severity:** MED (mismatch between body confidence and tag confidence).

---

## Closing notes

After wave-7:
- **Empirically settled:** 2 (DSMEM completion BW, membar costs)
- **Empirically open, sketches exist:** 2 (V55 HBM floor, V56 NVFP4 A:B)
- **Empirically open, NO sketch yet:** 1 (V57 per-pipe power)
- **Documentation defects:** 4 (§44 missing sketch, ~256 wording, 29 Agent labels, §50 body↔tag)

The doc remains USABLE AS-IS for HIGH-confidence quoting. v1.1 patch (proposed in `CANONICAL_PATCH_v1_1.md`) closes the documentation defects and 2 of the empirical contradictions; V55-V57 close the rest.

Recommended next-wave priority order:
1. **Apply CANONICAL_PATCH_v1_1.md** (closes #6, #7, #8, #9 — all documentation, mechanical)
2. **Run V55** (closes #1 — HBM write provenance, 0.5 day)
3. **Run V56** (closes #5 — NVFP4 A:B mechanism, 0.5-1 day)
4. **Run V57** (closes #4 — per-pipe power TF/W; needs new sketch first, ~1 day total)
