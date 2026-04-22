# STRAYS — INCONSISTENCY LOG (Wave-3 Audit)

**Companion to `STRAYS_CORRECTED.md`.** Lists every conflict found across
the stray file set without re-explaining the resolution (see CORRECTED
file). Format: source A vs source B → severity → action.

---

## A. HBM data-dependence magnitude (5–7× discrepancy)

| Source | Claim | Conf |
|---|---|---|
| `HBM_DATA_DEPENDENCE.md` line 30–32 | DRAM data-dep contributes **<50 W** | LOW (self-rated) |
| `L2_DRAM_DATA_PWR.md` line 51–54 | **5.4 W** spread (constant patterns) | HIGH |
| `POPCOUNT_3TIER.md` line 51 | **234 W** range DRAM-1G (random-popcount) | HIGH |
| `POPCOUNT_3TIER.md` line 52 | **240 W** range DRAM-8G | HIGH |
| `POPCOUNT_VS_CLOCK.md` lines 65–75 | **554 W** swing @ 1500 MHz | HIGH |

**Severity: HIGH** (5–11× factor).
**Resolution:** mode-dependent (constant vs random-position popcount).
HBM_DATA_DEPENDENCE generalized constant-pattern result to all data variation.
**Action:** mark HBM_DATA_DEPENDENCE.md as SUPERSEDED at top of file.
Already implicit in `16_power_clock_CORRECTED.md` line 307.

---

## B. L2 capacity = "96 MB" cosmetic error

| Source | Claim | Should be |
|---|---|---|
| `L2_BITSTRIDE_SWEEP.md` line 11 | "fits in 96 MB L2; full L2-warm" | **126 MB** |
| `POPCOUNT_3TIER.md` line 11 | "fits 96 MB L2" | **126 MB** |
| `L2_DRAM_DATA_PWR.md` line 17 | "fits in 96 MB L2 by 12×" (1.2 GB ws) | **126 MB** (still fits the 12× math by accident: 1.2 GB / 96 MB = 12.5×; / 126 MB = 9.5×; doesn't matter for measurement) |
| `03_caches_CORRECTED.md` §2.1 | 132,644,864 B = **126.5 MB** | canonical |
| `B300_TRUE_REFERENCE.md` row 134 | **126 MB** | canonical |

**Severity: LOW (cosmetic).** No measurement is wrong; just the printed
capacity is off. **Action:** sed pass `s/96 MB L2/126 MB L2/g` on the 3
listed files. None of the conclusions change.

---

## C. L2 atomic unit count "~32" confidence rating

| Source | Rating | Basis |
|---|---|---|
| `L2_UNITS_REFINED.md` line 33 | **MEDIUM-HIGH** | "~32 active in parallel; could be 30 or 35" |
| `CLOCK_DOMAINS_AND_L2_UNITS.md` line 41 | "Catalog claim: ~32 L2 atomic units (E4 task)" | references curiosity-list, not measurement |
| `B300_TRUE_REFERENCE.md` line 162 (surprise #18) | listed as a "counterintuitive finding" | implied verified |
| `07_atomics_CORRECTED.md` §RETRACTIONS #2 | **LOW** confidence | "ATOMIC_REVERIFY shows VERSION A reaches 20.4 L2 packets/cy suggesting ceiling could be MUCH higher" |
| Memory feedback "Dispatch ceiling skepticism" | derived ceilings need ncu cross-check before HIGH/MED-HIGH | applies here |

**Severity: MEDIUM.** Catalog implies HIGH-ish; reverify says LOW.
**Action:** harmonize to **MEDIUM** with explicit "inferred from stride-sweep
plateau, no direct ncu unit count" caveat. Update
`B300_TRUE_REFERENCE.md` surprise #18 to flag inference.

---

## D. __syncwarp cost (1 cy vs 23 cy framing)

| Source | cy | Framing |
|---|---:|---|
| `F2_SYNCWARP_RIGOR.md` | **0–2** (NOPs only) | "true cost (full mask)" — SASS verified |
| `F6_SYNCWARP_COST.md` | **+1** vs no-sync baseline | per-iter delta in 8-op loop |
| `B300_TRUE_REFERENCE.md` row 77 | **1 ns** | published canonical |
| V9_THREADFENCE_COST baseline | **23 cy** | LOOP-OVERHEAD proxy, mis-framed as syncwarp cost |
| `08_sync_primitives_CORRECTED.md` §RETRACTIONS #3 | already retracted V9 framing | resolved |

**Severity: LOW (already resolved upstream).** F2 and F6 agree; B300_TRUE_REFERENCE
1 ns is canonical; V9's misframing already retracted.
**Action:** none for stray files. The CLAUDE.md V9 memory note "atomic 697 cy
chained, 16 cy pipelined" still echoes the V9 framing — should reference
F2/F6 instead.

---

## E. F2 partial-mask cost (7.25 cy vs F6 +2 cy)

| Source | partial-mask cy | Mask | Method |
|---|---:|---|---|
| F2_SYNCWARP_RIGOR | **7.25** | 0x0000FFFF (16 lanes) | unrolled, divergent participation |
| F6_SYNCWARP_COST | **+2** vs no-sync | 0x0000FFFF (16 lanes) | clock64 unrolled loop, 8 ops baseline |

**Severity: LOW.** Different methodologies; both reference same 16-lane mask
but F2 has divergent participation (only 16 lanes call sync) vs F6 has
all-lanes-converged then half-mask. Costs measure different things.
**Action:** sweep mask-popcount in single test to characterize
properly. Currently OPEN per `STRAYS_CORRECTED §5 UNRESOLVED`.

---

## F. D9 LDG.cg slowdown (2.24× vs 1.25×)

| Source | gap | Working set |
|---|---:|---|
| D9_E4_LDG_ATOM_SASS line 39 | **2.24×** slower | 16 KB (L1-resident) |
| `03_caches_CORRECTED.md` §2.10 | **1.25×** slower | L2-hot |
| Older summaries (retracted) | "4.7×" | typo |

**Severity: LOW (regime-dependent).** Both correct in their regime.
**Action:** D9 should add "L1-resident only" qualifier; recommend
combined-recipe note "LDG.E.128 cached" referring to V10_LDG_WIDTH.

---

## G. A3 scoreboard depth confidence (HIGH vs MED self-flag)

| Source | Rating | Method count |
|---|---|---|
| A3 conclusion | "**HIGH** that scoreboard is ≥32" | 1 (clock64 only) |
| A3 self-confidence | "MED for scoreboard exact depth" | (acknowledged) |
| Rigor protocol (CLAUDE.md §4) | requires 3-method (wall + ncu + SASS) | not met |

**Severity: MEDIUM.** Self-contradiction in single file.
**Action:** Downgrade scoreboard ≥32 from HIGH to MED, run a test with N>32
to find plateau, add ncu metric (`smsp__average_warps_issue_stalled_long_scoreboard`).

---

## H. L1 LDG hit latency (26 cy vs 38–47 cy)

| Source | cy | What's measured |
|---|---:|---|
| A3_SCOREBOARD_DEPTH | **26** | issue→issue throughput floor |
| `03_caches_CORRECTED.md` §1.3 | **38–47** | full round-trip L1 hit latency |
| D2_L1_CAPACITY_RIGOR | 39 @ 1500 MHz | RT |
| V10_L1_CAPACITY | 47 @ random | RT |

**Severity: LOW** (different metrics).
**Action:** A3 should re-label "L1 LDG hit latency" → "back-to-back issue rate".

---

## I. FP8 K-vary scaling (2× predicted vs 1.51× measured)

| Source | factor | Notes |
|---|---:|---|
| FP8_KVARY_POWER linear K-cycle prediction | 2× BF16 | K=32 vs K=16 |
| FP8_KVARY_POWER measured | **1.51×** | sub-linear |

**Severity: LOW** (acknowledged inline; hypothesis = 2-K-positions-per-cycle
pipelining mitigates).
**Action:** None — file already discusses; OPEN for V8/V9 cross-check on
multiplier microarchitecture.

---

## J. FP8 vs BF16 A vs B asymmetry

| Source | BF16 ratio | FP8 ratio |
|---|---:|---:|
| FP8_KVARY_POWER (corrected w/ encoding fix) | **24×** (+2 W A vs +47 W B) | **>70×** (~0 W A vs +71 W B) |
| `16_power_clock_CORRECTED.md` §6 | "+2 vs +48 W" / "24× ratio" | (only BF16 listed there) |
| `B300_TRUE_REFERENCE.md` row 167–168, 175 | A free / B dominates (BF16) | (no FP8 explicit) |

**Severity: LOW** (consistent qualitatively, FP8 strengthens BF16 finding).
**Action:** add FP8 row to `16_power_clock_CORRECTED.md` §6 K-vary
asymmetry table.

---

## K. L2 read d=16 active power scaling

| Source | 1005 MHz | 1500 MHz | 1800 MHz | Ratio @ 1800 |
|---|---:|---:|---:|---|
| `POPCOUNT_3TIER.md` | 405 (peak) | — | — | — |
| `POPCOUNT_VS_CLOCK.md` | 405 | — | **771** | 1.90× |
| `L2_POPCOUNT_SWEEP.md` | **398** | — | — | — |
| `16_power_clock_CORRECTED.md` §5 | 405 | — | — | — |

**Severity: NONE** — 405 vs 398 W is within ±2% noise across
independent kernels. Files are consistent. **Action:** none.

---

## L. fence/syncthreads ladder cross-references

The stray F2/F6 cost match `08_sync_primitives_CORRECTED` §"RECOMMENDED
CANONICAL TABLE":

| Op | Canonical | F2/F6 | OK? |
|---|---|---|---|
| __syncwarp full | 1 cy / 0.5 ns | F2: 0–2 / F6: +1 | YES |
| __syncwarp partial | 7 cy / 3.5 ns | F2: 7.25 / F6: +2 (different test) | partial (see E) |
| membar.cta | 6–16 (one-thread) | F6: +6 cy | YES (in range) |

**Severity: NONE.** Consistent.

---

## SUMMARY (all severities)

- HIGH: 1 (HBM_DATA_DEPENDENCE supersession, item A)
- MEDIUM: 2 (A/G/H — A3 confidence + L2 atomic unit confidence)
- LOW (cosmetic / regime-dependent / acknowledged): 8 (B, D, E, F, I, J, plus K/L which are NONE-severity)

**Top fixes recommended (in priority order):**
1. Mark `HBM_DATA_DEPENDENCE.md` SUPERSEDED at top.
2. Sed `96 MB L2` → `126 MB L2` in 3 stray files.
3. Downgrade `L2_UNITS_REFINED.md` "~32" rating to MEDIUM with caveat.
4. Downgrade `A3_SCOREBOARD_DEPTH.md` to MED-confidence headline; re-run
   with N>32 + ncu.
5. Add "L1-resident only" qualifier to `D9_E4_LDG_ATOM_SASS.md` 2.24×
   slowdown claim.
6. Re-label A3's "L1 LDG hit latency 26 cy" → "back-to-back issue rate".
