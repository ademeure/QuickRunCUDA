# Canonical doc v1.1 PATCH — proposed edits (NOT yet applied)

**Target:** `/root/github/QuickRunCUDA/b300_clean/B300_CANONICAL_REFERENCE.md` (v1, 17631 lines).
**After patch:** v1.1, ~17650 lines, 2 settled contradictions removed from "preserved" list, 2 confidence tags upgraded to 🟢 HIGH.
**Drivers:** Wave-7 — `CANONICAL_DOC_DOUBT_REPORT.md`, `V53_RUN_RESULTS.md`, `V54_RUN_RESULTS.md`, `CITATION_VERIFY_REPORT.md`.

> **Status:** PROPOSED ONLY. Per task constraint, the canonical doc itself is NOT modified. Apply patch on review.

---

## Edit 1 — Line 297 (front matter): broken §65 cross-ref → Appendix A

**Severity:** HIGH (F1 from doc-doubt).

**Current text (line 297):**
> Agent F's appendix §65 has the full zigzag case study

**Proposed text:**
> Appendix A has the full zigzag case study

**Reason:** §65 is "Time-stamping + version" (line 14016), not the zigzag. The zigzag is **Appendix A** (line 14089). The stitcher claimed "0 broken §N refs" but this is real. Also drops leaked Agent label (see Edit 4).

---

## Edit 2 — Line 1377 (§9 see-also): broken §65 → §43

**Severity:** HIGH (F2 from doc-doubt).

**Current text (line 1377):**
> §65 (popcount synthesis, Agent F)

**Proposed text:**
> §43 (popcount synthesis)

**Reason:** Popcount synthesis lives at §43 (line 8503); §65 is Time-stamping. Also drops leaked Agent label.

---

## Edit 3 — §13 DSMEM rewrite (regime-stratified)

**Severity:** HIGH (V53 settles preserved contradiction #2).

**Target:** §13 ladder (locate the "560 GB/s/cluster write ceiling" row) and §13.1 SASS codegen note.

### 3a. Replace the write-BW ladder with a regime-stratified table

**Current text (DSMEM_REFERENCE.md §3-equivalent; flowed into canonical §13):**
> | 4 × 4 | 70.08 GB/s/CTA | 560.7 GB/s/cluster ← write ceiling |

**Proposed text:**
> ### Write BW (ring) — regime-stratified per V53 fenced retest (2026-04-22)
>
> | Regime | Per-CTA | Per-cluster | Aggregate (18 clu) | Confidence |
> |---|---:|---:|---:|---|
> | V21 burst (5 outer iters), unfenced (issue rate) | 53 GB/s | 428 GB/s | n/a | 🟢 HIGH |
> | V21 burst (5 outer iters), fenced (completion) | 14 GB/s | 110 GB/s | n/a | 🟢 HIGH |
> | Sustained 1-cluster (≥500 outer iters) | 27 GB/s | 217 GB/s | n/a | 🟢 HIGH |
> | **Sustained all-18-clusters (≥10 ms)** | **10 GB/s** | **82 GB/s** | **1.47 TB/s** | 🟢 **HIGH** |
>
> **Use 82 GB/s/cluster (10 GB/s/CTA, 1.47 TB/s aggregate) for any sustained-throughput model.** The original V21 number of 560 GB/s/cluster was measured at 5 outer iters WITHOUT a completion fence; adding `fence.sc.cluster + barrier.cluster.{arrive,wait}` drops the same burst 3.88× to ~110 GB/s/cluster — confirming V21 was issue-rate. At sustained lengths (kernel runtime ≥10 ms), fence cost is invisible to wall-clock because the kernel itself drains before `cudaDeviceSynchronize`.
>
> Source: V53_RUN_RESULTS.md (3 runs, ncu-verified, SASS-verified MEMBAR.SC.GPU + UCGABAR placement).

### 3b. Add bonus finding to §13.1 SASS codegen note

**Current text (§13.1, near "Goes through L2"):**
> Goes through L2 (ncu shows ~4 sectors/load)

**Proposed text:**
> For READS, `ld.shared::cluster.u32` with scalar address compiles to LD.E and goes through the L2 path (ncu confirms ~4 sectors/load). For WRITES, `st.shared::cluster.u32` also compiles to ST.E but does NOT measurably hit L2 (ncu `lts__t_sectors_op_write` shows <1 sector per 30,000 stores) — DSMEM writes use a dedicated inter-CTA fabric.
>
> Source: V53 ncu of `v53_write<0,4,1,50000>` shows `lts__t_sectors_op_write.sum = 3,312` for 115.36M LSU stores.

### 3c. Update §9 summary rule #2

**Current text (line ~1226 area, summary rules):**
> Use DSMEM writes over reads (13× higher aggregate BW: 560 vs 40 GB/s)

**Proposed text:**
> Use DSMEM writes over reads (~2× higher per-cluster sustained BW: 82 vs ~40 GB/s)

### 3d. §13 confidence tag

**Current:** ⚫ DISPUTED (preserved contradiction #2)
**Proposed:** 🟢 HIGH (sustained 82 GB/s, V53-settled); preserve a 🟡 MED-with-context note for the legacy 560 figure framed as issue-rate-only.

**Reason:** V53 partially refuted DSMEM_DOUBT (sustained = 1.00× gap, fence is free at steady state) AND partially confirmed it (burst = 3.88× gap, V21's 560 was issue-rate). Both readings need to live in the doc.

---

## Edit 4 — §32 `membar.sys`: collapse 1.74× spread to 2806 cy

**Severity:** HIGH (V54 settles preserved contradiction #3).

**Target:** §32 (`__threadfence_system` ladder row) and any cross-references in §31, §30.

### 4a. Replace the §32 fence-cost ladder

**Current text (§32 area):**
> | __threadfence_system / membar.sys | 1750-3042 cy | 861-1486 ns | spread 1.74×, ⚫ DISPUTED |

**Proposed text:**
> | __threadfence_system / membar.sys | **2806 cy** | **1381 ns @ 2032 MHz** | 🟢 HIGH (V54-settled, R²=1.0000) |

### 4b. Confidence tag

**Current:** ⚫ DISPUTED
**Proposed:** 🟢 HIGH

### 4c. Add retraction note (in §32 footnote or footgun E.S.x)

**Add:**
> **Retract:** `08_sync_primitives.md` row "membar.sys = 1750 cy / 861 ns" was 38% under-estimate (likely amortized over chained loop with overlapping work). V9_THREADFENCE_COST.md row "3042 cy" was slightly high (chain accumulated incremental coherence backpressure). V54 6-point N-scaling fit (R²=1.0000) is authoritative: per_fence = 2806 cy, ±0.6% across 3 runs.

---

## Edit 5 — §31 `membar.gl`: collapse 24% spread to 267 cy

**Severity:** MED (V54 settles).

**Target:** §31 (`__threadfence` ladder row).

### 5a. Replace §31 ladder row

**Current text:**
> | __threadfence / membar.gl | 258-320 cy | 127-158 ns | 🟡 MED (24% spread across 4 sources) |

**Proposed text:**
> | __threadfence / membar.gl | **267 cy steady-state** | **131.5 ns @ 2032 MHz** | 🟢 HIGH (V54-settled) |
>
> **Note (first-fence-after-write outlier):** The first `membar.gl` after a global write costs ~520 cy (267 steady-state + ~280 cy L2 round-trip drain). V54 N=1 datapoint shows 788 cy vs slope-predicted 775. Subsequent fences see no in-flight stores → steady-state 267 cy.

### 5b. Confidence tag

**Current:** 🟡 MED
**Proposed:** 🟢 HIGH

---

## Edit 6 — §30 `membar.cta`: 8 cy / 3.9 ns

**Severity:** MED (V54 settles).

**Target:** §30 (`__threadfence_block` ladder row).

### 6a. Replace §30 ladder row

**Current text:**
> | __threadfence_block / membar.cta | 6-16 cy | 3.0-7.9 ns | 🟡 MED (spread + V9 "free" claim) |

**Proposed text:**
> | __threadfence_block / membar.cta | **8 cy** | **3.9 ns @ 2032 MHz** | 🟢 HIGH (V54-settled, zero variance) |
>
> **Retract:** V9_THREADFENCE_COST.md "block fence is free" was a baseline-subtraction artifact (subtracted 23-cy baseline larger than the fence cost itself). V54 N-scaling slope of exactly 8 cy/fence has zero variance across 3 runs.

### 6b. Confidence tag

**Current:** 🟡 MED
**Proposed:** 🟢 HIGH

---

## Edit 7 — Lines 3768-3771 (§22 footgun #2): "~256 inst/SM/cy" wording

**Severity:** MED (F11 from doc-doubt).

**Current text (lines 3768-3771):**
> It is correct PER PIPE but each SM has multiple pipes that overlap, so total can reach ~256 inst/SM/cy (FFMA + LOP3 = 1 + 0.5 per SMSP × 4 SMSPs × 32 lanes = 192 inst/SM/cy in the V52 measured case).

**Proposed text:**
> Single-pipe IS capped at 128 inst/SM/cy. With two pipes overlapping, the **theoretical** total is ~256 inst/SM/cy (both pipes at 1 inst/cy). In the V52 measured FFMA+LOP3 case, the LOP3 2-cy issue cadence makes the **actual** total 192 inst/SM/cy (1 FFMA + 0.5 LOP3 per SMSP × 4 SMSPs × 32 lanes). The 256 figure is the architectural ceiling, not what V52 measured.

**Reason:** As written, the wording conflates the theoretical ceiling (256) with the V52 measured result (192). A reader will quote one or the other without context.

---

## Edit 8 — Line 14470 (Appendix A.6.6): same "~256" wording fix

**Severity:** MED (F16 from doc-doubt).

**Current text (line 14470):**
> total inst/SM/cy can reach ~256

**Proposed text:**
> total inst/SM/cy can theoretically reach ~256 (both pipes at 1 inst/cy); the V52 measured FFMA+LOP3 case is 192 inst/SM/cy because LOP3 is at 2-cy cadence

**Reason:** Same defect as Edit 7; needs parity fix.

---

## Edit 9 — Agent A/B/C/D/E/F label scrub (29 occurrences)

**Severity:** MED (F3 from doc-doubt).

**Targets:** Lines 290, 297, 310, 413, 452, 474, 559, 607, 1299, 1377, 1491, 1754, 1850, 1909, 2216, 2399, 2636, 3490, 3780, 3984, 4114, 4143, 4155, 4354-4355, plus any not yet enumerated.

**Action:** For each occurrence, either:
- (a) Delete the parenthetical "(Agent X)" / "Agent X's" — most common case;
- (b) Substitute the actual section number where the cite is doing real work (e.g., "Agent F's appendix" → "Appendix A").

**Reason:** Cosmetic; signals proofread quality. Doesn't change any number, but the stitcher claimed authority over the doc and these leaked through.

---

## Edit 10 — Add note to §44 (UNRESOLVED, missing retest)

**Severity:** MED (F5 from doc-doubt).

**Target:** §44 header note.

**Add (after the 🟡 MED — UNRESOLVED tag):**
> **Note:** No V57 sketch exists in Appendix C. The M11 vs 16_power_clock 2× per-pipe power discrepancy is genuinely UNRESOLVED. A future V57 should jointly sweep ncu pipe utilization (`smsp__pipe_X_cycles_active`) AND nvml power at fixed frequency (`-lgc 1500`) across FFMA-bound and LSU-bound configurations to identify which operating-point assumption produces which TF/W. Until V57 runs, both 0.111 and 0.21 TF/W are defensible at their stated configs but neither is "the" answer. UNRESOLVED 🟡 status maintained.

**Optional companion edit:** Add §44 ladder row labels with operating-point context: "FFMA peak (high-ILP)" and "FFMA-bound mid-occ (M11)".

**Reason:** Stitching notes claim "6 preserved contradictions"; Appendix C has retests for only 4. §44 silently falls through. Either add a V57 sketch to Appendix C or explicitly acknowledge the gap.

---

## Edit 11 — §50 NVFP4 A:B body language

**Severity:** MED (F8 / F14 from doc-doubt).

**Targets:** §50.2.2 ("the single most direct evidence"), §50.4 ("Why ALL three are simultaneously 'right'"), §50.5 ("Practical guidance").

### 11a. Tone down §50.2.2

**Current:** "the single most direct evidence"
**Proposed:** "one of the strongest pieces of evidence (MED-tagged: alternative mechanisms remain plausible per E.N.2)"

### 11b. Tone down §50.4 header

**Current:** "Why ALL three are simultaneously 'right'"
**Proposed:** "How the three readings can each be valid at different operating points (MED-tagged synthesis)"

### 11c. §50 TOC tag

**Current (TOC line 195):** `🟡 MED`
**Proposed:** `🟡 MED (regime-dependent: HIGH for K=96 production case, MED for cross-regime synthesis)`

**Reason:** §50 is tagged 🟡 MED but body language is HIGH-confident on the B-reuse mechanism. Footgun E.N.2 explicitly preserves all 3 readings; body should match. The K=96 production-case answer (2.6× B>A) is well-grounded; the cross-regime synthesis is genuinely MED.

---

## Edit 12 — Line 13282 typo: HEADLINE_v5 → HEADLINE_CORRECTIONS_v5

**Severity:** LOW (citation-verify finding).

**Current text (line 13282):**
> src: HEADLINE_v5.md

**Proposed text:**
> src: corrections/HEADLINE_CORRECTIONS_v5.md

**Reason:** Single-occurrence typo. The intended file is `corrections/HEADLINE_CORRECTIONS_v5.md` (used 29× elsewhere correctly).

---

## Edit 13 (optional, LOW) — §27.7 PIPE_OVERLAP_MATRIX phrasing

**Severity:** LOW (F7 from doc-doubt).

**Target:** §27.7 ("M8 PIPE_OVERLAP_MATRIX confirmed by V52").

**Add 1-line note:**
> Numerical entries above are the M8 measurements; they should be **re-measured** with V8/V52-style 128-deep unroll to remove loop-overhead contamination (see §22.7). Directional finding (free overlap) is confirmed by V52.

**Reason:** §22.7 says numerical entries need re-measurement; §27.7 says "confirmed". Not strictly contradictory but a casual reader gets two different impressions.

---

## Edit 14 (optional, LOW) — Wrong-path .cu cites

**Severity:** LOW (citation-verify finding).

**Targets:** Lines 4636, 5349.

**Current text:**
> src: cluster_raw_barrier.cu / cluster_sass_audit.cu

**Proposed text:**
> src: ../investigations/cluster_raw_barrier.cu / ../investigations/cluster_sass_audit.cu

**Reason:** Both .cu files live at `/root/github/QuickRunCUDA/investigations/`, outside `b300_clean/`. The relative-to-`b300_clean/` convention can't reach them.

---

## Summary — what v1.1 looks like

| Aspect | v1 | v1.1 (after patch) |
|---|---|---|
| Total lines | 17 631 | ~17 650 (net +19 from regime-stratified §13 table + §32/§31/§30 retraction notes + §44 V57 note) |
| Preserved contradictions | 6 | **4** (#2 DSMEM and #3 membar.sys settled) |
| Confidence tags upgraded to 🟢 HIGH | — | **3** (§30 membar.cta, §31 membar.gl, §32 membar.sys) |
| Confidence tags reframed | — | 1 (§13 DSMEM: ⚫ DISPUTED → 🟢 HIGH for sustained 82 GB/s, with 🟡 MED note for legacy 560 figure) |
| Broken cross-refs | 2 (lines 297, 1377) | 0 |
| Leaked Agent labels | 29 | 0 |
| Wording bugs ("~256 inst/SM/cy") | 2 | 0 |
| Citation typos | 1 (line 13282) | 0 |
| Missing retest acknowledgment | §44 silent | §44 explicit "no V57 sketch" note |
| §50 body↔tag mismatch | HIGH-language under MED tag | softened to MED-language |

**Net effect:** the doc retains all headline numbers, gains 3 HIGH-confidence ladder rows (membar.{cta,gl,sys}), removes 2 preserved contradictions from the open list, and fixes the cross-ref / wording / citation defects flagged by wave-7 doc-doubt.
