# CROSS-LINK AUDIT — `b300_clean/corrections/`

**Date:** 2026-04-22 (wave-5e).
**Auditor:** integrity-checker agent.
**Scope:** the 11 synthesis / index / doubt / headline files that should
form the canonical narrative chain. The 60+ topical CORRECTED and
INCONSISTENCY_LOG files are out of scope — they are leaf nodes that
the synthesis files cite, not synthesis themselves.

---

## 1. Per-file cross-reference table

| File | Internal links it makes | Target exists? | Latest? |
|---|---|---|---|
| `MASTER_INDEX.md` (v1, wave-1+2) | `HEADLINE_CORRECTIONS.md`, `B300_TRUE_REFERENCE_v2_DRAFT.md`, all 17 `*_CORRECTED.md`, all `*_INCONSISTENCY_LOG.md` | Yes | **STALE — points to v1 HEADLINE; should redirect readers to MASTER_INDEX_v2** |
| `MASTER_INDEX_v2.md` (wave-3c) | `HEADLINE_CORRECTIONS_v2.md` (×3), `DOUBT_LOG.md` (×2), `B300_TRUE_REFERENCE_v2_DRAFT.md`, `MASTER_INDEX.md` (as superseded), `HEADLINE_CORRECTIONS.md` (as superseded), wave-3a topical files, wave-3b doubt reports | Yes | **STALE on HEADLINE — should now point to v3, not v2. Missing: WAVE4_CHANGES, META_DOUBT_REPORT, UNRESOLVED_PROMOTED, HBM_DENOMINATOR_RESOLUTION, HBM_DENOMINATOR_FINAL, CONFIDENCE_LADDER** |
| `HEADLINE_CORRECTIONS.md` (v1) | `MASTER_INDEX.md`, `B300_TRUE_REFERENCE_v2_DRAFT.md`, individual `*_INCONSISTENCY_LOG.md` files | Yes | **OK for its era**, but no forward pointer to v2/v3 — readers landing here will not know they're reading wave-1+2 |
| `HEADLINE_CORRECTIONS_v2.md` (wave-3c) | `HEADLINE_CORRECTIONS.md` (as superseded), `DOUBT_LOG.md`, `MASTER_INDEX_v2.md` | Yes | **STALE — itself superseded by v3, but does not advertise that** |
| `HEADLINE_CORRECTIONS_v3.md` (wave-4) | `HEADLINE_CORRECTIONS_v2.md` (as superseded), `DOUBT_LOG.md`, `WAVE4_CHANGES.md`, `UNRESOLVED_PROMOTED.md`, `HBM_DENOMINATOR_RESOLUTION.md` (×2), `META_DOUBT_REPORT.md` (×2), `V51_INVESTIGATION.md`, V46_DOUBT, NVFP4_DOUBT, DSMEM_DOUBT, COMPUTE/HBM/INT/MATH/NVFP4/NVLink topical logs | Yes | **OK — current as of wave-4. Missing: HBM_DENOMINATOR_FINAL (wave-5b) which partially reverses HBM_DENOMINATOR_RESOLUTION's "7672 is a ghost" framing (rule 11 is now wrong)** |
| `B300_TRUE_REFERENCE_v2_DRAFT.md` (wave-2) | `MASTER_INDEX.md`, individual `*_CORRECTED.md` files, `*_INCONSISTENCY_LOG.md` files | Yes | **STALE — built before META_DOUBT, before HBM denom resolution, before HEADLINE v3. Header still says "12 HBM3E stacks" (wrong — HEADLINE_v3 #2 says 8); §1 still anchors "7672 GB/s post-ECC spec" (wave-4 retired, wave-5b partially un-retired); V49/V50 dual-issue framing (§6) is wave-2 era. Should be marked SUPERSEDED across the head; per-row downgrades in CONFIDENCE_LADDER are the live source.** |
| `CONFIDENCE_LADDER.md` (wave-3 era, Apr 22 05:29) | `B300_TRUE_REFERENCE_v2_DRAFT.md` (as TRUE_REF prefix only — note at file end), individual CORRECTED files, NO mention of HEADLINE v2/v3, NO mention of WAVE4_CHANGES / META_DOUBT_REPORT / UNRESOLVED_PROMOTED | Mostly | **STALE — predates wave-4 reversal of V49/V50 from LOW→MED. CONFIDENCE_LADDER §12 still grades V49/V50 dual-issue rows as "DOWNGRADED LOW", but META_DOUBT_REPORT §1 + WAVE4_CHANGES §1 + HEADLINE_CORRECTIONS_v3 #7 all upgrade to MED. Header says "12 HBM3E stacks" (wave-4 retired). Also still uses "7672 GB/s" as HIGH "doubt-confirmed" denominator — wave-5b says cite both.** |
| `DOUBT_LOG.md` (wave-3c) | All 6 wave-3b doubt reports (SYNTHESIS_DOUBT, V46_DOUBT, DUAL_ISSUE_DOUBT, DSMEM_DOUBT, NVFP4_DOUBT, CROSS_AGENT_DOUBT_LOG); wave-1+2 HEADLINE/MASTER/TRUE_REFERENCE; wave-3a topical files | Yes | **STALE on item #6 — META_DOUBT_REPORT §7 explicitly flags "DOUBT_LOG headline #6 LOW is too strong; should be MED". DOUBT_LOG never updated. Also missing forward links to WAVE4_CHANGES, UNRESOLVED_PROMOTED, HEADLINE_CORRECTIONS_v3.** |
| `UNRESOLVED_PROMOTED.md` (wave-4f) | `RETEST_PROPOSALS.md`, `DOUBT_LOG.md`, `META_DOUBT_REPORT.md`, `CONFIDENCE_LADDER.md`, `WAVE4_CHANGES.md`, `V51_INVESTIGATION.md` | Yes | **OK — current as of wave-4. Doesn't yet reference HBM_DENOMINATOR_FINAL (wave-5b).** |
| `WAVE4_CHANGES.md` (wave-4) | `META_DOUBT_REPORT.md`, `HBM_DENOMINATOR_RESOLUTION.md`, `RETEST_PROPOSALS.md`, `CONFIDENCE_LADDER.md`, `V51_INVESTIGATION.md`, `DOUBT_LOG.md` (as superseded on listed items only), `HEADLINE_CORRECTIONS_v2.md` (as superseded on listed items only) | Yes | **STALE — the §6 new methodology rule "Use 7680 GB/s post-ECC as canonical anchor" is partially superseded by HBM_DENOMINATOR_FINAL §5 which says "cite BOTH 7680 spec AND 7672 actual"; the "7672 is arithmetic ghost" framing in §2 is explicitly retracted by HBM_DENOMINATOR_FINAL §1. Also does not reference HBM_DENOMINATOR_FINAL (wave-5b) or this audit (wave-5e) — but those files post-date it, so this is unavoidable.** |
| `META_DOUBT_REPORT.md` (wave-3d) | All 6 wave-3b doubt reports as inputs; original CORRECTED files; raw V8/V49/V50/V21 sources; M8 matrix; `DOUBT_LOG.md` (audited) | Yes | **OK as a snapshot. Its conclusions FLOW INTO WAVE4_CHANGES and HEADLINE_CORRECTIONS_v3 — those are the operational outputs.** |

---

## 2. Stale-link list with proposed correct target

| Source file | Stale link | Proposed target |
|---|---|---|
| `MASTER_INDEX.md` (v1) header | `HEADLINE_CORRECTIONS.md` | Add forward note: "this file is wave-1+2; for current top-line see `HEADLINE_CORRECTIONS_v3.md` and `MASTER_INDEX_v2.md`". (Don't rewrite history; just signpost.) |
| `MASTER_INDEX_v2.md` line 8 | `HEADLINE_CORRECTIONS_v2.md` | `HEADLINE_CORRECTIONS_v3.md` |
| `MASTER_INDEX_v2.md` §3 (UNRESOLVED list) | (no link to retest proposals) | Add cross-ref to `UNRESOLVED_PROMOTED.md` and `RETEST_PROPOSALS.md` |
| `MASTER_INDEX_v2.md` §5 reading order | starts with `HEADLINE_CORRECTIONS_v2.md` | Should start with `HEADLINE_CORRECTIONS_v3.md`; insert `WAVE4_CHANGES.md` and `META_DOUBT_REPORT.md` between v2 and DOUBT_LOG |
| `HEADLINE_CORRECTIONS.md` (v1) header | (no forward pointer at all) | Add "SUPERSEDED by `HEADLINE_CORRECTIONS_v3.md` (current). v2 is intermediate." |
| `HEADLINE_CORRECTIONS_v2.md` header | (no forward pointer) | Add "SUPERSEDED by `HEADLINE_CORRECTIONS_v3.md`" |
| `HEADLINE_CORRECTIONS_v3.md` header / rule 11 | "Use 7680 GB/s post-ECC as canonical denominator" | Update to match `HBM_DENOMINATOR_FINAL.md` §5: "cite BOTH 7680 spec AND 7672 actual; the 0.10 % gap matters at SoL precision". Drop the "7672 is arithmetic ghost" framing (HBM_DENOMINATOR_FINAL §1 explicitly retracts it). |
| `HEADLINE_CORRECTIONS_v3.md` row #3 | "**7672 is an arithmetic ghost**" | Update: "**7672 GB/s = empirical post-ECC at 3996 MHz I/O on this silicon; 7680 GB/s = spec post-ECC at 8.000 Gbps/pin. Both correct, cite both for SoL claims.**" |
| `B300_TRUE_REFERENCE_v2_DRAFT.md` header (line 7) | "12 HBM3E stacks" | "8 HBM3E stacks of 12-Hi" — match HEADLINE_CORRECTIONS_v3 #2. Also add SUPERSEDED banner at top: "Wave-2 draft. For current top-line see HEADLINE_CORRECTIONS_v3, CONFIDENCE_LADDER (per-row), WAVE4_CHANGES, HBM_DENOMINATOR_FINAL." |
| `B300_TRUE_REFERENCE_v2_DRAFT.md` §1 row "HBM3E theoretical post-ECC spec (7672)" + every "% of 7672" entry | "7672 GB/s post-ECC" | Match `HBM_DENOMINATOR_FINAL.md`: cite 7680 spec and 7672 actual side by side |
| `B300_TRUE_REFERENCE_v2_DRAFT.md` §6 dual-issue ladder rows | "55% (V49)", "74% (V50)" with no confidence tag | Add tag matching CONFIDENCE_LADDER + WAVE4_CHANGES: "MED — reproducible measurement; dispatch-cap interpretation needs ncu (V52)" |
| `CONFIDENCE_LADDER.md` system header line 20 | "12 HBM3E stacks" | "8 HBM3E stacks of 12-Hi" |
| `CONFIDENCE_LADDER.md` §1 HBM denominator rows | "HBM3E theoretical post-ECC spec (denominator) 7672 — HIGH doubt-confirmed" | Per `HBM_DENOMINATOR_FINAL.md`: split into two rows (7680 spec, 7672 actual), HIGH for both; update doubt-status to "WAVE5B-RECONCILED" |
| `CONFIDENCE_LADDER.md` §12 V49/V50 dual-issue rows (lines 340-343) | "LOW — DOWNGRADED" | "MED — REVERSED in wave-4" per `WAVE4_CHANGES.md` §1 + `HEADLINE_CORRECTIONS_v3.md` #7 + `META_DOUBT_REPORT.md` §1 |
| `CONFIDENCE_LADDER.md` §15 launch overhead rows referring to "DOWNGRADED" without mentioning wave-4 | various | Add "see WAVE4_CHANGES" cross-ref where relevant |
| `DOUBT_LOG.md` §3 row 6 (NVFP4 cuBLAS 11.42 PF caveat is fine), but rows 1-2 (HBM denom + dual-issue) | "DOWNGRADED to LOW" for V49/V50 | Append note: "REVERSED in wave-4; see `WAVE4_CHANGES.md` §1 + `META_DOUBT_REPORT.md` §1 — now MED" |
| `DOUBT_LOG.md` §6 (net assessment) | "Promoted V49/V50 dual-issue 55%/74% to HIGH without acknowledging…" | Update: "wave-3c synthesis incorrectly DOWNGRADED to LOW; META_DOUBT (wave-3d) reversed to MED" |
| `WAVE4_CHANGES.md` §2 + §6 | "7672 is an arithmetic ghost", rule 11 = "use 7680 as canonical" | Reconcile with `HBM_DENOMINATOR_FINAL.md` §1: 7672 IS the literal hardware rate; cite alongside 7680, not in place of |
| `META_DOUBT_REPORT.md` (no stale links found) | — | OK |
| `UNRESOLVED_PROMOTED.md` P0 row 2 | "denominator standardization is RULE-stated (7680 GB/s post-ECC)" | Add: "see HBM_DENOMINATOR_FINAL.md for the 7680/7672 dual-citation rule" |

**Stale link summary:** **17 stale or incomplete cross-references** across 8 files. The most concerning are the 3 v1→v3 jumps (MASTER_INDEX_v2 still aiming readers at v2 HEADLINE, v1 HEADLINE with no forward pointer, B300_TRUE_REFERENCE_v2_DRAFT silently 2 waves out of date) and the 2 LOW-vs-MED contradictions on V49/V50 dual-issue (DOUBT_LOG §3 + CONFIDENCE_LADDER §12 still say LOW; HEADLINE v3 + WAVE4_CHANGES + META_DOUBT_REPORT say MED).

---

## 3. Broken story / contradictions

The narrative chain has three distinct contradictions that a new reader would hit:

### Contradiction A: V49/V50 dual-issue confidence

| File | Verdict | Era |
|---|---|---|
| HEADLINE_CORRECTIONS.md (v1) | (not graded — proudly headlined) | wave-1+2 |
| HEADLINE_CORRECTIONS_v2.md row #6 | **LOW** ("DOWNGRADED") | wave-3c |
| DOUBT_LOG.md headline #6 + §3 row 2 + §5 | **LOW** ("DOWNGRADED") | wave-3c |
| CONFIDENCE_LADDER.md §12 lines 340-343 | **LOW** ("DOWNGRADED to LOW") | wave-3 era |
| META_DOUBT_REPORT.md §1 + §7 | **MED** ("OVERSTATED — should be MED, not LOW") | wave-3d |
| WAVE4_CHANGES.md §1 | **MED** ("REVERSED — wave-3c was too harsh") | wave-4 |
| HEADLINE_CORRECTIONS_v3.md row #7 | **MED** ("REVERSED from v2") | wave-4 |

**Three files still hold the LOW verdict that has been formally REVERSED in three subsequent files.** A reader who lands on CONFIDENCE_LADDER first will quote LOW; one who lands on HEADLINE_v3 first will quote MED. This is the most dangerous contradiction in the corpus.

### Contradiction B: HBM denominator framing (7672 vs 7680 vs both)

| File | Framing | Era |
|---|---|---|
| MASTER_INDEX.md, B300_TRUE_REFERENCE_v2_DRAFT.md, 01_hbm_bandwidth_CORRECTED.md, CONFIDENCE_LADDER.md | **7672 GB/s post-ECC (HIGH "doubt-confirmed")** | wave-1+2 / wave-3 |
| HEADLINE_CORRECTIONS_v3.md #3 + WAVE4_CHANGES.md §2 | **7672 retired as "arithmetic ghost"; use 7680** | wave-4 |
| HBM_DENOMINATOR_FINAL.md §1 | **7672 is NOT a ghost — it is the literal hardware rate at the empirical 3996 MHz I/O clock; cite BOTH 7680 spec and 7672 actual** | wave-5b |

**3-way contradiction.** The wave-4 retirement is itself partially retired by wave-5b. CONFIDENCE_LADDER's "HIGH doubt-confirmed" label on 7672 is now accidentally correct (wave-5b's verdict) by skipping wave-4 entirely.

### Contradiction C: HBM stack count (12 vs 8)

| File | Claim | Era |
|---|---|---|
| `B300_TRUE_REFERENCE_v2_DRAFT.md` line 7 | "12 HBM3E stacks" | wave-2 |
| `CONFIDENCE_LADDER.md` line 20 | "12 HBM3E stacks" | wave-3 |
| CLAUDE.md | (also said 12 — see HEADLINE v3 row 2) | pre-wave-4 |
| HEADLINE_CORRECTIONS_v3.md #2 + WAVE4_CHANGES.md §2 + HBM_DENOMINATOR_FINAL.md §4 | **8 HBM3E stacks of 12-Hi** | wave-4 + wave-5b |

Two synthesis files still in active use carry the wrong stack count in their header.

---

## 4. Recommended SINGLE entry point

**`HEADLINE_CORRECTIONS_v3.md`** is the only file that:
- Knows about every wave (1+2, 3a, 3b, 3c, 3d, 4)
- Carries forward pointers to all current operational docs (DOUBT_LOG, WAVE4_CHANGES, UNRESOLVED_PROMOTED, META_DOUBT_REPORT, HBM_DENOMINATOR_RESOLUTION)
- Has correct claims on the wave-4 reversals (V49/V50 LOW→MED) and wave-4 new findings (8 stacks, V51 bug)
- Is short enough (~80 lines) to be the actual top-line document

**Caveat the reader must know:** v3's rule #11 ("use 7680 as canonical") was partially un-done by `HBM_DENOMINATOR_FINAL.md` (wave-5b) — the new rule is "cite both 7680 spec and 7672 actual, mention the 0.10 % gap when SoL precision matters". This is the single open patch needed on v3.

Recommend appending to v3 a one-line note: **"After reading: see `HBM_DENOMINATOR_FINAL.md` for the wave-5b correction to rule 11."**

---

## 5. Linearization order for a new reader

To get current ground truth, a reader should consume in this order:

1. **`HEADLINE_CORRECTIONS_v3.md`** — top-line TL;DR (12 wave-4 corrections, all wave-3 retractions, methodology rules 1-11)
2. **`HBM_DENOMINATOR_FINAL.md`** — patches v3's rule 11 (7672 is real, cite both)
3. **`WAVE4_CHANGES.md`** — what shifted vs wave-3c (with caveat: §2 + §6 retire 7672 — superseded by step 2 above)
4. **`META_DOUBT_REPORT.md`** — why three wave-3 doubt reports overstated their case (drives the v3 reversals)
5. **`UNRESOLVED_PROMOTED.md`** — open questions with concrete retest sketches (V52–V56)
6. **`MASTER_INDEX_v2.md`** — directory map (CORRECTED + INCONSISTENCY_LOG files); ignore its line 8 v2 reference, jump back to v3
7. **`CONFIDENCE_LADDER.md`** — 303-row look-up table (with 2 known stale verdicts: V49/V50 dual-issue and 7672 denominator — apply patches from steps 1+2 inline)
8. **`B300_TRUE_REFERENCE_v2_DRAFT.md`** — full proposed reference (treat as wave-2 snapshot; cross-check every "% of HBM peak" against step 2 and every dual-issue row against step 1)
9. **`DOUBT_LOG.md`** — wave-3c per-claim verdicts (apply step 4's reversal on item #6 inline)
10. **`HEADLINE_CORRECTIONS_v2.md`** — historical context only
11. **`HEADLINE_CORRECTIONS.md` (v1)** + **`MASTER_INDEX.md` (v1)** — historical context only

**Files to mark SUPERSEDED in their headers** (for next pass, not done by this audit since no edits were requested to those files):
- HEADLINE_CORRECTIONS.md → SUPERSEDED by v3
- HEADLINE_CORRECTIONS_v2.md → SUPERSEDED by v3
- B300_TRUE_REFERENCE_v2_DRAFT.md → wave-2 snapshot; per-row verdicts in CONFIDENCE_LADDER + per-topic patches in HEADLINE_v3 / HBM_DENOMINATOR_FINAL are live
- CONFIDENCE_LADDER.md → 2 known stale verdicts; rest current
- DOUBT_LOG.md → 1 known stale verdict (item #6); rest current
- MASTER_INDEX.md → SUPERSEDED by v2 (which is itself partially stale; flag in MASTER_INDEX_v2 to point at HEADLINE_v3)

---

## 6. The 3 most concerning broken links (TL;DR for the report-back)

1. **CONFIDENCE_LADDER.md §12 + DOUBT_LOG.md §3 still grade V49/V50 dual-issue as LOW**, but META_DOUBT_REPORT.md, WAVE4_CHANGES.md, and HEADLINE_CORRECTIONS_v3.md all reverse this to MED. A reader pulling a confidence grade from CONFIDENCE_LADDER will publish a stale verdict.

2. **MASTER_INDEX_v2.md line 8 directs readers to HEADLINE_CORRECTIONS_v2.md** as "the doubt-aware TL;DR", but v3 supersedes v2. The single most-likely-clicked link in the index is wrong by one major version.

3. **B300_TRUE_REFERENCE_v2_DRAFT.md** carries the wrong HBM stack count (12 vs real 8) in its header, anchors its entire memory ladder to "% of 7672" (which wave-4 retired and wave-5b partially un-retired), and pre-dates the V49/V50 reversal — yet remains the only "B300_TRUE_REFERENCE_v2" candidate. It should carry a SUPERSEDED banner pointing readers to (HEADLINE_v3 + CONFIDENCE_LADDER + HBM_DENOMINATOR_FINAL) for current ground truth.

A close 4th: **HEADLINE_CORRECTIONS_v3.md rule 11** ("use 7680 as canonical") needs a one-line patch to direct readers to `HBM_DENOMINATOR_FINAL.md` for the wave-5b dual-citation rule.
