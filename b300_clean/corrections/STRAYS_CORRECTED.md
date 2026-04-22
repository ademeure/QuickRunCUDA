# STRAYS — CORRECTED Reference (Wave-3 Audit)

**Scope.** Audit of stray b300_clean files not covered by the main 17-category
sweep: `FP8_KVARY_POWER`, `HBM_DATA_DEPENDENCE`, `A3_SCOREBOARD_DEPTH`,
`D9_E4_LDG_ATOM_SASS`, `F2_SYNCWARP_RIGOR`, `F6_SYNCWARP_COST`, the
POPCOUNT family (`POPCOUNT_3TIER`, `POPCOUNT_VS_CLOCK`, `POPCOUNT_WRITES`,
`L2_POPCOUNT_SWEEP`), `L2_DRAM_DATA_PWR`, `L2_BITSTRIDE_SWEEP`,
`L2_UNITS_REFINED`, `CLOCK_DOMAINS_AND_L2_UNITS`. Cross-checked against
already-corrected files (`07_atomics_CORRECTED`, `08_sync_primitives_CORRECTED`,
`16_power_clock_CORRECTED`, `03_caches_CORRECTED`) and `B300_TRUE_REFERENCE`.

**Originals UNMODIFIED.** Source-of-truth date 2026-04-22.

---

## 1. FP8_KVARY_POWER.md

**Subject confirmed.** The file is about **tcgen05.mma FP8 e4m3 power vs B-operand
K-vary diversity**, NOT FP8 cvt. The memory note "FP8 cvt 2× faster than BF16"
is unrelated and does not apply here.

| Headline | File value | Verdict |
|---|---|---|
| FP8 e4m3 K=32 mma random power @ 1005 MHz | 630 W | HIGH (matches `B300_TRUE_REFERENCE` "FP8 random 642 W" within 12 W noise) |
| FP8 e4m3 B=const Tier B power | 304 W | HIGH (matches BF16 Tier B 299 W) |
| B K-vary 16 unique cost (FP8) | +71 W | HIGH (replicable, 1.51× BF16's +47 W) |
| A K-vary effectively 0 W (after encoding fix) | ~0 W | HIGH (4 datapoints in noise) |
| A vs B K-vary asymmetry FP8 | >70× | HIGH (consistent with BF16's 24× asymmetry pattern) |
| K-cycle scaling prediction (BF16 K=16 vs FP8 K=32 → 2×) | measured 1.51× | MED (sub-linear; hypothesis 2-K-positions-per-cycle is plausible but unverified) |

**No retraction needed.** The CORRECTED 16_power_clock §6 mentions "B K-vary
16-unique adds +48W (24× ratio)" for BF16 — file's BF16 baseline (+47 W) and
the FP8 +71 W extension are mutually consistent.

### RETRACTIONS
None. The "A K-vary encoding bug acknowledged" is correctly self-disclosed
inline; the corrected post-fix table supersedes the original +11 W figure.

### UNRESOLVED
- 2-K-positions-per-cycle pipelining hypothesis (1.5× vs 2× theoretical).
- Memory note "FP8 cvt 2× faster than BF16" is unrelated to this file —
  belongs to a separate cvt/MUFU benchmark; flag for memory cleanup.

---

## 2. HBM_DATA_DEPENDENCE.md — SUPERSEDED

**Power agent finding CONFIRMED.** This file claims HBM data-dependent
power is **<50 W** out of TDP, with self-acknowledged LOW confidence on
magnitude ("proper DRAM benchmark not completed", 20.4 GB/s test was
"poorly optimized"). The file is the **inferred / pre-sweep version**.

Contradicted by the rigorous popcount sweep family:

| Source | DRAM data-dep range (active W) |
|---|---|
| HBM_DATA_DEPENDENCE.md (this file) | <50 W |
| `L2_DRAM_DATA_PWR.md` (constant patterns) | 522.6 → 528.0 = **5.4 W** ← agrees |
| `POPCOUNT_3TIER.md` DRAM-1G (random-position popcount) | **234 W** range (369 → 604) |
| `POPCOUNT_3TIER.md` DRAM-8G (random-position popcount) | **240 W** range (397 → 637) |
| `POPCOUNT_VS_CLOCK.md` DRAM-8G @ 1500 MHz | 367 → 921 = **554 W** swing |
| `POPCOUNT_WRITES.md` DRAM-8G writes @ 1005 MHz | 264 → 405 = 141 W |

**Reconciliation:** the disagreement is real but ONLY when comparing
"constant-pattern data" (`L2_DRAM_DATA_PWR` controls inter-dword toggle to
near-zero by repeating the same pattern → 5.4 W spread) vs
"random-position-popcount data" (`POPCOUNT_3TIER` deliberately varies bit
positions per dword → 240 W spread at d=0..d=32).

`HBM_DATA_DEPENDENCE.md` was written BEFORE the popcount sweep distinguished
these regimes; it generalized the inter-pattern (constant-vs-constant)
result to ALL data variation. The popcount work proves that **inter-dword
toggle activity (random-position) is the dominant lever, not popcount per se**,
and the swing is 5–7× larger than HBM_DATA_DEPENDENCE estimated.

### RETRACTIONS
1. "HBM data-dependent power likely contributes <50W out of total 1100W TDP"
   — WRONG. Real swing under random-position popcount is **240 W active /
   554 W at 1500 MHz**. Use `POPCOUNT_3TIER`/`POPCOUNT_VS_CLOCK`/
   `16_power_clock_CORRECTED` §5 going forward.
2. "Memory bandwidth doesn't have a strong throttling-driven speedup
   mechanism" — partially WRONG. Bandwidth is content-INDEPENDENT
   (correctly captured by `L2_DRAM_DATA_PWR.md`), but POWER is strongly
   content-dependent and CAN throttle clocks/cap TDP at high clocks
   (1100 W TDP wall hit at 1700–1800 MHz with d=8..28 random data).
3. "Memory-bound workloads: no significant throttling avoidance" — WRONG.
   At 1500–1800 MHz, random-data DRAM workloads CAN reach TDP cap and
   throttle; low-popcount data avoids this and saves 240 W.

### UNRESOLVED
- High-quality DRAM peak kernel test at 7+ TB/s with random vs zero data
  is now COMPLETE in `POPCOUNT_3TIER.md` (DRAM-8G achieves saturation).
  The file's "next investigation needed" is satisfied by later work.
  Recommended action: mark `HBM_DATA_DEPENDENCE.md` as SUPERSEDED in a
  header note (already implicit in `16_power_clock_CORRECTED.md` line 307
  which lists it as "now superseded").

---

## 3. A3_SCOREBOARD_DEPTH.md

**Claim.** Scoreboard depth ≥32 per warp; L1 LDG hit latency ~26 cy
(issue→issue, NOT round-trip); cy/load decreases monotonically through N=32.

| Headline | File value | Verdict |
|---|---|---|
| Scoreboard depth ≥32 per warp | HIGH | PARTIAL (file itself flags MED on exact depth — could be 32, 48, or higher; never reached plateau) |
| L1 LDG issue→issue latency | 26 cy | MED (file calls it L1 hit latency but explicitly notes catalog L1 hit = 38–47 cy in `03_caches_CORRECTED` §1.3; the 26 cy is issue→issue throughput minimum, not full RT) |
| Per-warp peak LDG throughput | 4.72 cy/load at N=32 | HIGH (clean monotonic curve) |

### Three-method rigor check

**FAIL.** A3 used **only one method** (clock64 inside kernel, single
benchmark `bench_scoreboard_slots.cu`). It does NOT have:
- ncu cross-check (e.g. `smsp__inst_executed_pipe_lsu` or
  `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_cycle_active`)
- SASS verification beyond "registers in regs[32]"
- A second independent test with N>32 to find plateau

This means by the rigor protocol (CLAUDE.md §4) A3 should be downgraded to
MED on scoreboard ≥32, NOT HIGH. The file partially admits this in its
own confidence statement ("MED for scoreboard exact depth").

### Cross-ref against catalog 04/A1/A2
- **04_fp32_peak.md** does not discuss scoreboard depth (no overlap).
- **A1_DUAL_ISSUE_RIGOR.md** and **A2_SCHEDULER_RIGOR.md** address
  dispatch and pipeline issue, not load scoreboard. No conflict.
- **V10_LDG_WIDTH.md** uses dispatch limit "4 inst/cy/SM" from V9 — is a
  per-cycle ceiling, not scoreboard-depth ceiling. **Compatible** with A3.

### RETRACTIONS
1. "L1 LDG hit latency = ~26 cy (consistent with catalog 43 cy when
   including consume overhead)" — the parenthetical reconciliation is
   sloppy. Catalog L1 hit latency (`03_caches_CORRECTED` §1.3) is
   38–47 cy round-trip; A3's 26 cy is issue→issue throughput-floor, NOT
   the same metric. Should re-label as "back-to-back issue rate ~26 cy"
   not "L1 hit latency".
2. Confidence "HIGH that scoreboard is ≥32" is overstated; downgrade to
   MED until a test with N>32 finds the plateau.

### UNRESOLVED
- True scoreboard depth (could be 32, 48, 64+).
- Whether scoreboard slots are warp-private or shared across warps in an
  SMSP.
- Three-method verification needed (currently 1-method only).

---

## 4. D9_E4_LDG_ATOM_SASS.md

**Claim.** SASS encoding map for LDG variants and atom/red variants;
`__ldg ≠ ld.global.ca` at SASS level but **same speed** at runtime.
`ld.global.cg` is 2.24× SLOWER than `.ca`/`__ldg`/default at L1-resident
workloads.

| Headline | File value | Verdict |
|---|---|---|
| `__ldg` emits `LDG.E.CONSTANT` | HIGH | HIGH (SASS-verified) |
| `ld.global.ca` emits `LDG.E.STRONG.SM` | HIGH | HIGH |
| `ld.global.cg` emits `LDG.E.STRONG.GPU` | HIGH | HIGH |
| `red.relaxed.gpu` ≡ `red.global` SASS | HIGH | HIGH (matches `07_atomics_CORRECTED` §6 retirement of red.release.gpu) |
| `ld.global.cg` 2.24× slower than `.ca` at 16 KB hot region | HIGH | **CONFLICT — see below** |
| `__ldg` NOT measurably faster than `ld.global.ca` | HIGH | HIGH |

### Conflict with V10_LDG_WIDTH and 03_caches_CORRECTED

`03_caches_CORRECTED.md` §2.10 cache hint table:
- `.ca` vs `.cg` at L2-hot = **1.25×**, NOT 4.7× (older summary typo)
- `.ca` 13.1 TB/s vs `.cg` 10.5 TB/s = -20%

D9's "2.24× slower" was measured at **16 KB hot region (L1-resident)**, where
`.cg` bypasses L1 entirely and pays L2 latency (~300 cy vs L1's ~40 cy =
~7.5× latency penalty, partially hidden by ILP). At larger working sets
(L2-resident, ≥4 MB), the gap collapses to 1.25× because both miss L1.

**No actual contradiction once regime is specified.** Recommend D9 add the
"L1-resident only" qualifier prominently.

### Agreement with V10_LDG_WIDTH (LDG.128 best)

V10_LDG_WIDTH.md establishes **128-bit LDG = 5.76 TB/s vs 32-bit = 1.95 TB/s
(2.95× speedup)** at HBM-bound workloads. D9 only tested **width-32 variants**
across cache hints — the two findings are orthogonal:
- D9 says "for read-only data, prefer `__ldg`/`.ca`/default; avoid `.cg`"
- V10_LDG_WIDTH says "always use widest natural access (LDG.128)"

Both are simultaneously true. **The combined recipe** (per V8/V10 agent
finding) is: **LDG.E.128 cached (`.ca` or default) is the SoL**.

### RETRACTIONS
1. D9's "Big lesson: avoid `ld.global.cg`" should be qualified as
   "...for L1-resident workloads only". For DRAM-bound work the difference
   is essentially zero (per `03_caches_CORRECTED` §2.10).
2. D9 row "Use `ld.global` (no qualifier) or `ld.global.ca`" for read-write
   data — the "no qualifier" emits `LDG.E` (no STRONG.SM) vs `.ca`'s
   `LDG.E.STRONG.SM`. These differ in cache scope semantics; not strictly
   equivalent. Should add "...with caveat that no qualifier is weaker
   ordering than `.ca`".

### UNRESOLVED
- ncu `lts__t_sectors_op_read` cross-check for "constant cache more
  efficient" claim (file flags MED).
- Width × cache-hint matrix (LDG.128 + .ca vs LDG.128 + .cg vs LDG.32 +
  .ca etc.) not measured anywhere in the catalog.

---

## 5. F2_SYNCWARP_RIGOR vs F6_SYNCWARP_COST

Both agree on the qualitative finding: **`__syncwarp(0xFFFFFFFF)` is
essentially FREE.** They differ on the precise cycle count and on what
counts as "the cost".

| Source | Full-mask cy | Partial-mask cy | Method |
|---|---:|---:|---|
| F2_SYNCWARP_RIGOR | **0–2** (NOPs only, no SASS emitted) | **7.25** | Unrolled loop, SASS dump verified |
| F6_SYNCWARP_COST | **+1** vs no-sync baseline (24 vs 23 cy/iter, 8 ops/iter) | **+2** | Single-warp clock64, 8 ops baseline |
| `08_sync_primitives_CORRECTED` (canonical) | **1 cy / 0.5 ns** @ boost | **7 cy / 3.5 ns** | Both, reconciled |
| `B300_TRUE_REFERENCE` row | "**__syncwarp = 1 ns**" (5d632d5) | not listed | published |

### Sync agent finding CONFIRMED

The note "V9 baseline of 23 cy was loop overhead — F2 proves true cost is 1 cy"
is **EXACTLY RIGHT**. F6's 23 cy is the per-iter cost of "8 compute ops + loop
control with no sync"; the "+1 cy" delta when adding `__syncwarp(0xFFFFFFFF)` IS
the syncwarp cost (or rather, the cost-above-noise — F2 proves it's actually
0 SASS instructions, so the 1 cy is just measurement granularity).

**B300_TRUE_REFERENCE's "__syncwarp = 1 ns" matches both** (1 cy at
boost 2032 MHz = 0.49 ns; the published "1 ns" rounds up — within noise).

### Both F2 and F6 are RIGHT, NOT contradictory

- F2 measures **emitted SASS** → 0 instructions for full mask (compiler
  removes the BAR.WARP.SYNC entirely when mask is provably full).
- F6 measures **observable per-iter cycles** → +1 cy delta (likely just
  the granularity floor of clock64).

### RETRACTIONS
1. **No retraction needed for F2 or F6.** Both are correct.
2. The earlier V9 framing of "23 cy syncwarp" should be retracted (already
   handled in `08_sync_primitives_CORRECTED.md` RETRACTION #3 — V9 used
   syncwarp as loop-overhead proxy not as the measurand).

### UNRESOLVED
- F2's "if a test with recent intra-warp divergence + full-mask syncwarp
  shows >2 cy, would mean compiler tracks divergence state" — open
  experiment.
- Partial mask cost: F2 says 7.25 cy, F6 says +2 cy. These are different
  configurations (F2's 7.25 is for 0x0000FFFF half-mask in a different
  test setup; F6's +2 is for the same 0x0000FFFF in the unrolled loop).
  Need a single sweep across mask popcount to settle.

---

## 6. POPCOUNT family — all 4 files MUTUALLY CONSISTENT

Cross-check of the popcount bell-curve at d=16 across all 4 files:

| File | Tier | d=16 active W (1005 MHz) | Verdict |
|---|---|---:|---|
| `L2_POPCOUNT_SWEEP.md` | L2 | **398** | source for L2 bell |
| `POPCOUNT_3TIER.md` | L2 | **404.8** | matches L2_POPCOUNT (~400 ± 7) |
| `POPCOUNT_3TIER.md` | DRAM-8G | **636.5** | rigorous |
| `POPCOUNT_VS_CLOCK.md` | DRAM-8G | **637** @ 1005 / 921 @ 1500 / 942 @ 1800 (TDP-capped) | matches 3TIER |
| `POPCOUNT_WRITES.md` | L2 write | **235** | distinct measurement |
| `POPCOUNT_WRITES.md` | DRAM-8G write | **405** | distinct measurement |

**All four files agree on the bell-curve peaking at d=16** with the
mechanism (inter-dword toggle activity, NOT per-dword popcount alone)
and the toggle-energy model (verified by constant-pattern control in
`POPCOUNT_3TIER` §"Theory: bus-toggle (Hamming) energy"). The toggle
coefficient ladder (L1 ~30 → L2 ~325 → DRAM ~360 W ceiling) is consistent
across all four.

The asymmetry (d=32 > d=0 by 11–45 W, growing with cache distance) is
reported consistently across `L2_POPCOUNT_SWEEP` (+22 W L2),
`POPCOUNT_3TIER` (+11.8 L1 / +22.8 L2 / +41.6 DRAM-1G / +44.8 DRAM-8G),
and `POPCOUNT_WRITES` (+19 L2W / +24 DRAM-W). All blame HBM3E PHY DBI /
active-low termination.

### RETRACTIONS
None. The 4 files form a coherent body of work. `16_power_clock_CORRECTED`
already cites them as the canonical power-data-dependence source.

### UNRESOLVED
1. Per-DRAM-channel `dram__bytes_*.per_dram` ncu metric to confirm even
   distribution across 6 HBM3E stacks at high clock (flagged by all 4 files).
2. Voltage probe to attribute the 5% super-linearity in L2 reads at high
   clock (`POPCOUNT_VS_CLOCK`).
3. TMA bulk loads — different memory subsystem path, not yet swept.
4. Hold popcount fixed but vary inter-dword Hamming distance directly
   (predicted by toggle theory; flagged in `POPCOUNT_3TIER` §"What would
   change conclusions").
5. Real production weight tensors vs synthetic d=16 not directly verified
   (also flagged in `16_power_clock_CORRECTED` §UNRESOLVED #10).

---

## 7. L2 = 96 MB cosmetic error — CONFIRMED in 4 files

**Cache agent finding CONFIRMED.** Real L2 capacity is **126.5 MB**
(`cudaDeviceProp.l2CacheSize = 132,644,864 B`, per
`03_caches_CORRECTED` §2.1 and `B300_TRUE_REFERENCE` row 134).

| File | "96 MB" mention | Lines | Verdict |
|---|---|---|---|
| `L2_BITSTRIDE_SWEEP.md` | "fits in 96 MB L2; full L2-warm" | line 11 | **WRONG — should be 126 MB** |
| `L2_POPCOUNT_SWEEP.md` | (does not mention 96 MB explicitly; says "64 MB ws (L2-warm)" only) | — | OK |
| `L2_UNITS_REFINED.md` | (does not mention 96 MB) | — | OK |
| `CLOCK_DOMAINS_AND_L2_UNITS.md` | (does not mention 96 MB) | — | OK |
| `POPCOUNT_3TIER.md` | "64 MB ws (fits 96 MB L2)" | line 11 | **WRONG — should be 126 MB** |
| `L2_DRAM_DATA_PWR.md` | "8 MB working set, fits in 96 MB L2" | line 17 | **WRONG — should be 126 MB** |

**4 files (not just 1) carry the cosmetic 96 MB error.** None of the
underlying measurements depend on the wrong number — all working sets
(8 MB, 64 MB, 1 GB, 8 GB) are correct relative to either the wrong
(96 MB) or right (126 MB) capacity. The 64 MB WS does fit in 126 MB L2,
just as well as it fit in the imagined 96 MB.

### RETRACTIONS
1. **`L2_BITSTRIDE_SWEEP.md` line 11**: "96 MB L2" → use **126 MB**.
2. **`POPCOUNT_3TIER.md` line 11**: "fits 96 MB L2" → use **126 MB**.
3. **`L2_DRAM_DATA_PWR.md` line 17**: "fits in 96 MB L2" → use **126 MB**.

These are documentation-only; no measurement is affected. Future revisions
(or a global sed pass) should fix.

### UNRESOLVED
- Origin of the 96 MB confusion. Possibly an early B100 spec, or
  estimation from carved-out persisting L2 (79.1 MB max) rounded up.
- Whether any other strays carry the same error — scan
  `b300_clean/*.md` for "96 MB" before next reference build.

---

## 8. L2_UNITS_REFINED + CLOCK_DOMAINS_AND_L2_UNITS

| Headline | Value | Verdict |
|---|---:|---|
| Per-unit throughput, single line | 0.83 packets/video-cy | HIGH |
| Aggregate uncombined | ~27 packets/video-cy = 50 Gops/s | HIGH |
| Inferred L2 atomic unit count | ~32 (MEDIUM-HIGH self-rated) | **see RETRACTIONS** |
| L2 video clock = 1860 MHz constant | HIGH | HIGH (matches `03_caches_CORRECTED` §2.7) |
| Combined atomics SM-issue-bound; uncombined L2-bound | HIGH | HIGH |

### Conflict with 07_atomics_CORRECTED + memory feedback

`07_atomics_CORRECTED` §RETRACTIONS #2 explicitly flags "L2 atomic units
count = ~32" as **LOW confidence** (plateau-derived inference, not direct
measurement; reverify shows VERSION A reaches 20.4 L2 packets/cy
suggesting ceiling could be HIGHER).

Memory feedback "Dispatch ceiling skepticism" says: ">128 SASS-inst/SM/clk
total is suspicious; validate with ncu before reporting." The "~32 L2
atomic units" claim is the same kind of derived ceiling that should be
LOW conf until ncu exposes a per-partition unit count.

### RETRACTIONS
1. `L2_UNITS_REFINED.md` "MEDIUM-HIGH" rating on ~32 L2 atomic units
   should be **MEDIUM** (matching `07_atomics_CORRECTED` §RETRACTIONS #2).
   Per memory's "Dispatch ceiling skepticism" rule, derived
   ceilings need ncu cross-check before HIGH/MED-HIGH ratings.
2. `CLOCK_DOMAINS_AND_L2_UNITS.md` "Catalog claim: ~32 L2 atomic units
   across 2 partitions (E4 task)" — the "E4 task" reference is to the
   curiosity list, not a direct measurement. Should annotate as
   "**inferred**".

### UNRESOLVED
- True L2 atomic unit count, per partition. ncu `lts__t_*` per-partition
  metrics could resolve.
- VERSION A vs VERSION B (REVERIFY) discrepancy for combine=32 (see
  `07_atomics_CORRECTED` §UNRESOLVED).

---

## SUMMARY OF NET CHANGES

- **`HBM_DATA_DEPENDENCE.md` is SUPERSEDED** by `POPCOUNT_3TIER.md` +
  `POPCOUNT_VS_CLOCK.md` + `L2_DRAM_DATA_PWR.md`. Real DRAM data-dep
  swing is 240–554 W, NOT <50 W.
- **A3_SCOREBOARD_DEPTH** is 1-method (no ncu, no SASS-counting); flag MED
  not HIGH.
- **D9_E4_LDG_ATOM_SASS** is correct but its "2.24× cg-slowdown" needs
  the "L1-resident only" qualifier; combined recipe with V10_LDG_WIDTH
  is "LDG.E.128 cached (.ca)".
- **F2 vs F6**: both right, B300_TRUE_REFERENCE 1 ns is canonical.
- **POPCOUNT family** (4 files): mutually consistent, canonical.
- **L2 = 96 MB** appears in **4 files** as cosmetic error; should be 126 MB.
- **L2 atomic units = ~32**: MEDIUM not MEDIUM-HIGH.
- **FP8_KVARY_POWER**: confirmed mma power, not cvt; unrelated to memory
  note about "FP8 cvt 2× faster than BF16".
