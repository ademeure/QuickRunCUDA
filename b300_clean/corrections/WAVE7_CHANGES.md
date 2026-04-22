# WAVE-7 CHANGES — adversarial doubt, V53/V54 settlements, citation audit

**Date:** 2026-04-22
**Supersedes:** `WAVE6_CHANGES.md`
**Drivers:** 4 wave-7 sub-agents — canonical-doc adversarial pass, V53 DSMEM fenced retest, V54 membar isolation, citation integrity scan.

> **Caveat up front:** wave-7 did NOT modify the canonical doc. All findings here propose a v1.1 patch (see `CANONICAL_PATCH_v1_1.md`). 2 of the 6 preserved contradictions are now empirically settled; 4 remain (see `REMAINING_CONTRADICTIONS.md`).

---

## 1. Canonical-doc adversarial pass (Agent: doc-doubt)

**Source:** `CANONICAL_DOC_DOUBT_REPORT.md` (18 findings F1-F18, 5 ranked top concerns).

### What was found

- **F1, F2 (HIGH):** Two `§65` cross-references are broken. Front matter line 297 says "Agent F's appendix §65" but §65 is "Time-stamping + version" (line 14016); the zigzag is **Appendix A** (line 14089). §9 see-also (line 1377) cites "§65 (popcount synthesis)" but popcount synthesis is **§43** (line 8503).
- **F3 (MED):** **29 occurrences** of leaked sibling-agent labels ("Agent A/B/C/D/E/F", "Agent C", "Agent E §50", "Agent F's appendix") survived stitching. Cosmetic — does not change any number — but signals incomplete proofread.
- **F5 (LOW→MED):** Stitching notes claim "6 preserved contradictions"; Appendix C only sketches retests for 4 (V53/V54/V55/V56). **§44 (M11 vs 16_power_clock 2× discrepancy) has NO retest sketch** — silently missing. Also §7 HBM write provenance is only partially covered by V55.
- **F8 / F14 (MED):** §50 is tagged 🟡 MED in the TOC, but the body uses HIGH-confidence language ("the single most direct evidence", "Why ALL three are simultaneously right"). The MED tag is technically correct, but body language trends HIGH-confident on the B-reuse mechanism without matching the prefix warning.
- **F11 / F16 (MED):** "~256 inst/SM/cy" wording bug appears in both §22 footgun #2 and Appendix A.6.6. Conflates theoretical ceiling (~256, both pipes at 1 inst/cy) with V52 measured (192 inst/SM/cy, since LOP3 is at 2-cy cadence). Reader will quote one or the other without context.

### Severity

HIGH (load-bearing, broken cross-refs); MED (proofread quality); LOW (cosmetic).

### What it changes in the canonical doc

8 proposed edits (see `CANONICAL_PATCH_v1_1.md`). None invalidate any headline number.

---

## 2. V53 DSMEM fenced retest (Agent: dsmem-retest)

**Source:** `V53_RUN_RESULTS.md`. Code: `tests/standalone/v53_dsmem_fenced.cu`. SASS-verified, ncu cross-checked.

### What was found

V53 ran identical kernels with `FENCED` template differing only in adding `fence.sc.cluster + barrier.cluster.{arrive,wait}` between the last DSMEM store and the closing `clock64`. Three runs, results stable to <1%.

**Empirical results (median of 3 runs):**

| Regime | Per-CTA | Per-cluster | UF/F gap | Conclusion |
|---|---|---|---|---|
| V21 burst (5 outer iters) | UF 53.5 / F 13.8 GB/s | UF 428 / F 110 GB/s | **3.88×** | V21's 560 was ISSUE rate |
| Burst 50 outer iters | 28.7 / 24.4 GB/s | 230 / 195 GB/s | 1.18× | gap closes |
| Sustained 1-cluster (≥500) | 27.2 / 27.0 GB/s | 217 / 216 GB/s | 1.00× | fence is free |
| Sustained 18-cluster (≥10ms) | 10 GB/s | 82 GB/s | 1.00× | fabric saturation |

**Bonus finding (HIGH):** ncu shows `lts__t_sectors_op_write.sum = 3,312` for 115M DSMEM stores — i.e. **<1 sector per 30,000 stores hits L2**. DSMEM writes do NOT measurably traverse L2; they go through the dedicated inter-CTA SMEM fabric. Previous claim in `DSMEM_REFERENCE.md` line 22 ("Goes through L2 (ncu shows ~4 sectors/load)") is **wrong for stores** (correct for reads).

### Severity

HIGH (settles preserved contradiction #2; rewrites canonical §13 numbers).

### What it changes in the canonical doc

- **§13 DSMEM ladder must be regime-stratified.** "560 GB/s/cluster write ceiling" should be replaced with: 428 GB/s burst-issue (5-iter), 110 GB/s burst-completion, 217 GB/s sustained 1-cluster, **82 GB/s sustained 18-cluster (1.47 TB/s aggregate)**.
- **§13 SASS codegen note:** add "DSMEM **writes** do not traverse L2 (ncu confirms <1 sector per 30k stores)".
- **§13 confidence:** ⚫ DISPUTED → 🟢 HIGH for sustained 82 GB/s; 🟡 MED-with-context for the 560 figure (now reframed as issue-rate-only).
- **§9 summary rule #2** ("Use DSMEM writes over reads, 13× higher") softened to "~2× higher per-cluster sustained BW: 82 vs ~40 GB/s".

### Honest disclosure

V53 partially **refuted** DSMEM_DOUBT (sustained 1.00× gap means the fence is free at steady state) AND partially **confirmed** it (burst 3.88× gap means V21's specific number was issue-rate). Both readings need to be in the doc.

---

## 3. V54 membar isolation (Agent: membar-retest)

**Source:** `V54_RUN_RESULTS.md`. Code: `tests/standalone/v54_membar_isolation.cu`. SASS-verified, R²=1.0000 fits.

### What was found

V54 ran a 6-point N-scaling sweep (N ∈ {1, 2, 4, 8, 16, 32}) of `membar.{cta,gl,sys}` per kernel, with `atom.global.add` anchors framing the timed region. SASS confirms exactly N MEMBAR instructions per kernel. R² = 1.0000 across all 3 scopes.

**Linear-fit slopes (per-fence cost, steady-state):**

| Scope | per_fence (cy) | ns @ 2032 MHz | Run-to-run |
|---|---:|---:|---|
| `membar.cta` | **8.00** | 3.9 | zero variance |
| `membar.gl` | **267.3** | 131.5 | ±0.02% |
| `membar.sys` | **2806** | 1381 | ±0.6% |

**Notable:** the N=1 outlier for `membar.gl` (788 cy vs slope-predicted 775) is the "first-fence-after-write" L2 drain (~280 cy on top of the steady-state 267 cy). All prior 4-way 258/281/292/320 cy estimates are within ±24% of the true 267 cy steady-state.

### Severity

HIGH (settles preserved contradiction #3; collapses 1.74× spread to ±0.6%; promotes 3 ladder rows to 🟢 HIGH).

### What it changes in the canonical doc

- **§30 `membar.cta`:** 8 cy / 3.9 ns (was 6-16 cy spread). Promote to 🟢 HIGH.
- **§31 `membar.gl`:** 267 cy / 131.5 ns steady-state, with N=1 outlier of +280 cy first-fence-after-write L2 drain. 🟡 MED → 🟢 HIGH.
- **§32 `membar.sys`:** 2806 cy / 1381 ns @ 2032 MHz (V54-settled). ⚫ DISPUTED → 🟢 HIGH.
- **Retract:** `08_sync_primitives.md` row "membar.sys = 1750 cy / 861 ns" (38% under-estimate); `V9_THREADFENCE_COST.md` "block fence ~0 cy" (was baseline-subtraction artifact).

---

## 4. Citation integrity audit (Agent: citation-verify)

**Source:** `CITATION_VERIFY_REPORT.md`. Scanned 130 confidence-tagged claims, 69 unique src strings, 117 unique file paths.

### What was found

**Verdict: SOLID.** 0 broken-in-spirit citations (no claim is unsupported). Defects:

- **1 typo (LOW):** Line 13282 says `HEADLINE_v5.md` — should be `HEADLINE_CORRECTIONS_v5.md` (used 29× elsewhere). Single occurrence.
- **2 wrong-path .cu cites (LOW):** `cluster_raw_barrier.cu` and `cluster_sass_audit.cu` (lines 4636, 5349) live at `/root/github/QuickRunCUDA/investigations/`, not `b300_clean/`. The relative-to-`b300_clean/` convention can't reach them; should cite as `../investigations/<name>` or with absolute prefix.
- **15 bare-filename CORRECTED cites:** ambiguous-but-resolvable (e.g. `META_LESSONS.md` should be `corrections/META_LESSONS.md`). Cosmetic.
- **3 user-memory pointers** (`project_b300_*.md`): intentional cross-system refs, correctly disambiguated by inline text.

### Severity

LOW (1 typo, 2 wrong-path; otherwise solid).

### What it changes in the canonical doc

- Line 13282 typo fix: `HEADLINE_v5.md` → `HEADLINE_CORRECTIONS_v5.md`.
- Optional: prefix `../investigations/` on the 2 .cu cites at lines 4636, 5349.
- Optional cosmetic: prefix `corrections/` on the 15 ambiguous CORRECTED cites.

Spot checks performed: 10/10 quick-nav ↔ body answers match; 10/10 §N anchors resolve; 10/10 Appendix D provenance map entries exist on disk.

---

## Settled contradictions

| # | Topic | Wave-6 status | Wave-7 settler | New status |
|---:|---|---|---|---|
| 2 | DSMEM write fenced vs unfenced (560 vs 110 GB/s) | LOW; flag for V53 | **V53 (this wave)** | 🟢 HIGH — regime-stratified: 428 burst-issue, 110 burst-completion, **82 sustained 18-cluster (1.47 TB/s agg)** |
| 3 | `__threadfence_system` 1.74× spread (1750–3042 cy) | LOW; flag for V54 | **V54 (this wave)** | 🟢 HIGH — **2806 cy / 1381 ns @ 2032 MHz** (R²=1.0000, ±0.6% across runs) |

**Bonus settlement from V53 not previously listed as a contradiction:**
- DSMEM write L2 traversal (`DSMEM_REFERENCE.md` claim was wrong for stores) — settled HIGH.

---

## Remaining contradictions

(See `REMAINING_CONTRADICTIONS.md` for full disposition; 4 from preserved + 4 from canonical-doc-doubt.)

| # | Topic | What would settle it | Severity |
|---:|---|---|---|
| 1 | HBM 7.57 NINJA write provenance (NINJA STG vs TMA bulk) | V55 — HBM floor empirical, sketched in Appendix C, not yet run | MED |
| 4 | per-pipe power M11 vs 16_power_clock 2× discrepancy | V57 — NEW joint power × ncu pipe utilization sweep at fixed freq; **NOT in Appendix C backlog** | MED |
| 5 | NVFP4 A:B 3-way reading (preserved) | V56 — NVFP4 A:B mechanism kernel, sketched in Appendix C, not yet run | MED |
| 6 | §44 retest sketch missing in Appendix C | Add V57 sketch to Appendix C | LOW (process) |
| 7 | "~256 inst/SM/cy" wording bug (§22 + Appendix A.6.6) | Edit per CANONICAL_PATCH_v1_1.md | MED (cosmetic but misleading) |
| 8 | 29 leaked Agent A/B/C labels | Sweep + scrub | MED (cosmetic) |
| 9 | §50 NVFP4 body trends HIGH but tagged MED | Tone-down per patch, OR per-row promote B-reuse to HIGH | MED |
| 10 | §27.7 vs §22.7 PIPE_OVERLAP_MATRIX phrasing | Add 1-line note to §27.7 | LOW |

---

## What survived doubt

Approximately **90% of the canonical doc** survived the wave-7 adversarial pass without any change:

- **§22 + Appendix A dual-issue zigzag.** Both reach the same V52-settled conclusion (alu+fma=147%, free overlap, V49/V50 retracted). Appendix A is the most complete single appendix (5-level chronology, 4 lessons, A.7-A.12 meta-content). Only the "256 vs 192" wording bug is a real defect. ✓ **HIGH placement is correct.**
- **§5/§6 HBM denominator handling.** Explicitly retires the 7.31 TB/s denominator, re-anchors V46 at 93.8%, §56 anchors against 7.67 this-device. Cross-doc denominator drift genuinely fixed; historical artifact preserved. ✓
- **§4 HBM 8-stacks.** Two-method confirmation (NVIDIA blog quote + cudaGetDeviceProperties output). Footgun E.B.1 correctly attributes the "12 stacks" claim to other catalog files (verified via grep — CLAUDE.md does NOT contain "12 stacks"). ✓
- **§45 -lgc 2032 paradox.** Quadruple-confirmed in 4 sources, full lock-state truth table, stuck-at-1005 separate failure mode covered. **Solid.** ✓
- **§9 HBM_DATA_DEPENDENCE.md superseded.** Explicit subsection at line 1362; footguns E.B.6, E.Z.1 consistent. **Solid.** ✓
- **§16 FFMA peak.** 76.96 theoretical, 74.62 measured at boost; -lgc paradox flagged in body and footgun #2; 128 cores/SM in footgun #1. **Solid.** ✓
- **CURIOSITY_LIST_V2 hash hallucination.** Flagged in footgun E.H.1 with 22/25 hashes hallucinated; matches CLAUDE.md `feedback_task_list_hashes`. **Captured.** ✓
- **Appendix E footgun index.** Comprehensive, alphabetised, symptom-keyed lookup table well-built. ✓
- **Appendix C retest sketches.** V53/V54/V55/V56 all production-quality (kernels, predictions, decision rules). Caveat: §44 power and §7 HBM write provenance not directly addressed. ✓
- **Citation integrity.** 130/130 confidence-tagged claims have a resolvable source. 1 typo, 2 wrong-paths, 0 broken-in-spirit. ✓
- **Quick-nav ↔ body answer consistency.** 10/10 spot-check rows match. ✓

**Verdict for the doc as a whole: USABLE AS-IS for HIGH-confidence quoting; v1.1 patch recommended before declaring "production-ready".**
