# HEADLINE CORRECTIONS — v6 (wave-7)

**Date:** 2026-04-22
**Supersedes:** `HEADLINE_CORRECTIONS_v5.md` (wave-6)
**Drivers:** V53_RUN_RESULTS.md (DSMEM fenced retest), V54_RUN_RESULTS.md (membar isolation), CANONICAL_DOC_DOUBT_REPORT.md (adversarial pass), CITATION_VERIFY_REPORT.md.

> **Caveat up front:** v6 is the **6th iteration**. Two of the six wave-6 preserved contradictions are now empirically settled by V53 + V54 with R²=1.0000 fits and ncu / SASS verification. The canonical doc has NOT been patched yet; v1.1 patch is proposed in `CANONICAL_PATCH_v1_1.md` but unapplied.

---

## Top-line corrections (15 entries; v5 → v6 deltas marked)

| # | Topic | v5 said | v6 says | Confidence | Wave |
|---:|---|---|---|---|---|
| 1 | HBM3E stack count | 8 of 12-Hi (this device 7680-bit AC SKU) | unchanged | HIGH | W6b |
| 2 | "12 stacks" error location | not in CLAUDE.md | unchanged | HIGH | W5c |
| 3 | HBM denominator | 7.68 spec / 7.67 this-device | unchanged | HIGH | W6b |
| 4 | HBM SoL recipe (V46) | 7.20 = 93.9% of 7.67 | unchanged | HIGH | W6b |
| 5 | HBM best measured | 7.30 = ~95% of 7.67 | unchanged | HIGH | W6b |
| 6 | HBM ECC reservation | 1/16 SECDED → 7680 from 8192 raw | unchanged | HIGH | W4 |
| 7 | V49/V50 dual-issue | HIGH (architectural) + RETRACTED (numbers); pipes overlap freely (alu+fma=147%) | unchanged | HIGH | W6a |
| 8 | V51 multi-stream HBM | bug-fix; LOW | unchanged | LOW | W4 |
| 9 | NVFP4 K=96 A:B asymmetry | MED — 3 framings retained pending V56 | unchanged (V56 still pending) | MED | W4 |
| 10 | DSMEM read 40 / write 560 / "no shared bus" | LOW/LOW-MED | **SETTLED by V53 (this wave)** — see new entry 14 | RESOLVED | W7 |
| 11 | `__threadfence_system` 1.74× spread | LOW | **SETTLED by V54 (this wave)** — see new entry 15 | RESOLVED | W7 |
| 12 | HBM write SoL 7.57 NINJA | MED — provenance unclear | unchanged (V55 sketched, not run) | MED | W3b |
| 13 | M8 PIPE_OVERLAP_MATRIX | CONFIRMED by V52 + first-principles | unchanged | HIGH | W6a |
| **NEW 14** | **DSMEM write completion BW** | (was rolled into #10) | **82 GB/s/cluster sustained 18-cluster (1.47 TB/s aggregate); 217 GB/s 1-cluster sustained; 110 GB/s burst-completion (5-iter, fenced); 428 GB/s burst-issue (5-iter, unfenced) — V21's 560 was issue rate, ~7× over completion at the same geometry. Bonus: DSMEM writes do NOT traverse L2 (<1 sector / 30k stores).** | **HIGH** | **W7** |
| **NEW 15** | **membar per-fence cost (V54)** | (was 1.74× spread + "block fence free" claims) | **`membar.cta` = 8 cy / 3.9 ns; `membar.gl` = 267 cy / 131.5 ns steady-state (+280 cy first-fence-after-write L2 drain); `membar.sys` = 2806 cy / 1381 ns @ 2032 MHz. R²=1.0000 N-scaling fit, ±0.6% across 3 runs. Retract: 08_sync_primitives 1750-cy was 38% under; V9 "block fence free" was baseline-subtraction artifact.** | **HIGH** | **W7** |

---

## Status of the 6 wave-6 preserved contradictions

| # | Topic | W6 status | W7 settler | W7 status |
|---:|---|---|---|---|
| 1 | HBM 7.57 NINJA write provenance | MED | not yet run (V55 sketched) | MED — STILL OPEN |
| 2 | DSMEM write fenced vs unfenced | LOW; flag V53 | **V53 (this wave)** | 🟢 **SETTLED** |
| 3 | `__threadfence_system` 1.74× spread | LOW; flag V54 | **V54 (this wave)** | 🟢 **SETTLED** |
| 4 | per-pipe power M11 vs 16_power_clock 2× | (preserved) | V57 NOT in Appendix C backlog | MED — STILL OPEN |
| 5 | NVFP4 A:B 3-way reading | MED | V56 not yet run | MED — STILL OPEN |
| 6 | (V51 multi-stream — see entry 8) | LOW | unchanged | LOW |

**2 of 6 settled this wave.** 3 require V55/V56/V57 to settle.

---

## Methodology rules (1-14; rule 14 NEW)

1-13: unchanged from v5.

14. **(NEW) For per-fence cost claims, use N-scaling slope (N ∈ {1, 2, 4, 8, 16, 32}) with anchored timed regions (`atom.global.add` pre+post, `clock64` outside).** SASS-verify exact MEMBAR count per kernel. Single-fence + chain-divide methods inflate or deflate by ~10-40% depending on baseline subtraction; only the slope is methodology-invariant. The N=1 datapoint will outlier for `membar.gl` due to in-flight L2 drain — this is **the** "first-fence-after-write" cost; report it separately from steady-state.

---

## Lessons from the dual-issue 5-level zigzag

(Unchanged from v5 — settled at W6 V52.)

---

## What v6 readers should believe right now

- **HBM3E:** 7.20 TB/s sustained = ~94% of this-device peak (7.67); ~93% of spec-comparable (7.68). 8 stacks; one /16 controller fused on AC SKU. **HIGH**. (unchanged)
- **Dual-issue FMA + ALU:** YES, pipes overlap freely; alu+fma ≈ 147%. **HIGH (V52 ncu)**. (unchanged)
- **M8 PIPE_OVERLAP_MATRIX:** CONFIRMED. (unchanged)
- **DSMEM write completion (NEW V7):** 82 GB/s/cluster sustained 18-cluster, 1.47 TB/s aggregate. The 560 GB/s V21 figure was burst issue rate; under fence, the same burst is 110 GB/s. Use 82 for sustained models. **HIGH (V53)**.
- **DSMEM writes do not traverse L2 (NEW V7):** <1 L2 sector per 30k stores. Reads still go via L2 (~4 sectors/load). **HIGH (V53 ncu)**.
- **membar costs (NEW V7):** cta=8 cy / 3.9 ns; gl=267 cy / 131.5 ns steady-state; sys=2806 cy / 1381 ns @ 2032 MHz. **HIGH (V54 R²=1.0000)**.
- **Everything else:** unchanged from v5.

---

## Canonical doc v1.1 patch — proposed but UNAPPLIED

`CANONICAL_PATCH_v1_1.md` proposes 14 edits (8 substantive + 6 cosmetic) to apply on review. Net effect:

- 2 preserved contradictions removed (#2 DSMEM, #3 membar.sys)
- 3 ladder rows promoted to 🟢 HIGH (§30 membar.cta, §31 membar.gl, §32 membar.sys)
- §13 DSMEM regime-stratified ladder added; bonus L2-traversal note added
- §44 explicit "no V57 sketch" note added (preserves UNRESOLVED status honestly)
- 2 broken §65 cross-refs fixed (front matter line 297, §9 see-also line 1377)
- "~256 inst/SM/cy" wording bug fixed (§22 + Appendix A.6.6)
- 29 leaked Agent A/B/C labels scrubbed
- §50 NVFP4 body language softened to match MED tag (or per-row promote B-reuse to HIGH)
- Citation typo fixed (line 13282)

After patch: v1.1, ~17 650 lines, **4 preserved contradictions remaining** (was 6).

---

## Acknowledged risk that v6 is also wrong

V53 caveat: the 1.47 TB/s "aggregate" was measured directly with 18 clusters active, not extrapolated. The single-cluster 187 GB/s sustained could itself be inflated if surrounding 17 SMs are idle — a per-CTA serving cap on the inter-CTA fabric is **inferred** but the 0.44× scaling-from-1-cluster-to-18 is the only data anchoring it. Could be SM-issue throttling rather than fabric saturation. MEDIUM confidence on the mechanism, HIGH on the per-cluster numbers.

V54 caveats: single-thread, single-CTA, single-warp test. Multi-warp / multi-CTA / contended fences may differ. The "system" fence value was measured on a 2-GPU NVLink-connected B300; on a single-GPU system the cost may be lower.

What could overturn v6: a multi-CTA contended-fence test that shows the V54 N-scaling slope is not the steady-state under realistic load; or a DSMEM write test at non-V21 geometry that shows the 82 GB/s/cluster sustained ceiling moves significantly. Neither is expected, but neither has been run.

---

## Single recommended entry point

**`HEADLINE_CORRECTIONS_v6.md`** (this file). Then in order:
1. This file (top-line + methodology rules)
2. `WAVE7_CHANGES.md` (delta vs wave-6, with the 4 sub-agent breakdowns)
3. `V53_RUN_RESULTS.md` (DSMEM fenced empirical anchor)
4. `V54_RUN_RESULTS.md` (membar isolation empirical anchor)
5. `CANONICAL_DOC_DOUBT_REPORT.md` (adversarial pass on the canonical doc)
6. `CITATION_VERIFY_REPORT.md` (citation integrity audit)
7. `CANONICAL_PATCH_v1_1.md` (proposed v1 → v1.1 edits, unapplied)
8. `REMAINING_CONTRADICTIONS.md` (what's still open after wave-7)
9. `META_LESSONS.md` (5-level dual-issue distilled wisdom)
10. `CONFIDENCE_LADDER.md` + `CONFIDENCE_LADDER_PATCH_v3.md` (ladder; patch needs a v4 to absorb V53/V54 ⚫ → 🟢)
