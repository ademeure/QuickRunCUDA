# CONFIDENCE LADDER — Wave-5 Patch v2 (2026-04-22)

**Purpose:** delta against `CONFIDENCE_LADDER.md` (v1, 303 rows). Supersedes
`CONFIDENCE_LADDER_PATCH.md` (v1 = wave-5d) which became STALE the moment
`SASS_VERIFY_DUAL_ISSUE.md` (wave-5a) re-downgraded the dual-issue rows that
v1 had just promoted.

**Why v2 exists:** the wave-5d patch promoted the four V49/V50 dual-issue rows
LOW → MED on the basis that wave-4 META_DOUBT_REPORT had falsified the
under-occupancy mechanism behind their wave-3b downgrade. Wave-5a then ran
SASS analysis and found a *different* methodology bug (loop-overhead
contamination from a tiny inner body — 8 FFMA + 8 LOP3 + BRA + UIADD3/UISETP
on the same ALU pipe), re-downgrading them to LOW. v2 is the corrected patch.

**Wave 5 status tags:**
- `KEEP` — row stands as graded
- `RE-PROMOTE` — wave-3b downgrade was overstated; raise grade
- `RE-DOWNGRADE` — wave-3b promote/HIGH was overstated; lower grade
- `RE-DOWNGRADE-2` — wave-5d promote was overstated by wave-5a; lower again
- `NUANCED` — value stands but framing or denominator needs paired statement
- `NEW-CITE` — denominator/source line edited

Only rows whose grade or annotation changes are listed; everything else is
`KEEP` and not reproduced here.

---

## Section 1 — HBM denominator (wave-5b reversal) — UNCHANGED FROM v1

| Row | Old grade / annotation | New grade / annotation | Wave 5 status | Latest verdict (post-W5a) | Reason |
|---|---|---|---|---|---|
| HBM3E theoretical post-ECC spec (denominator) — value `7672 GB/s` | HIGH / doubt-confirmed | **HIGH / NUANCED** — both **7680** (spec, 8.000 Gbps/pin) **and 7672** (this-device, empirical 7.992 Gbps/pin at 3996 MHz I/O) are valid; cite both | NEW-CITE | HIGH / NUANCED (no W5a impact) | `nvidia-smi -q` reports 3996 MHz Memory clock → 7672 is the literal hardware rate, 0.10 % under-spec; W4 "ghost" framing wrong. |
| HBM read (V46 TMA 8-deep pipelined) — `7.20 TB/s` REFRAMED | HIGH / "98.5% used wrong denom" | **HIGH / NUANCED** — 93.75 % of 7680 spec **and** 93.85 % of 7672 actual | NEW-CITE | HIGH / NUANCED (no W5a impact) | W5b promotes 7680 to default citation, with 7672 paired when SoL precision matters. |
| (implicit denominator note) every "% of HBM peak" row | HIGH / 7672 implicit | **HIGH / NUANCED** — default to 7680, optionally pair with 7672 | NEW-CITE | HIGH / NUANCED (no W5a impact) | Methodology rule 11 (HEADLINE_v4) standardizes 7680 as default; v5b lifts the 7672 ban. |

## Section 2 — Dual-issue (wave-5a SASS RE-DOWNGRADE-2) — REPATCHED FROM v1

| Row | Old grade (v1 of `CONFIDENCE_LADDER`) | v1-patch grade (W5d) | **v2-patch grade (post-W5a)** | Wave 5 status | Iteration history (W1+2 → W3b → W4 → W5a) | Reason |
|---|---|---|---|---|---|---|
| Same-warp dual-issue FFMA + LOP3 — `55 %` | LOW / DOWNGRADED | MED / RE-PROMOTED | **LOW / RE-DOWNGRADE-2** | RE-DOWNGRADE-2 | HIGH (V49 measured) → LOW (W3b: under-occupancy) → MED (W4: V8 ran 97.6 % at same 2 warps/SMSP, falsifies under-occupancy) → **LOW (W5a: loop-overhead contamination)** | Loop-overhead contamination per SASS analysis (wave-5a); inner body = 8 FFMA + 8 LOP3 + BRA + UIADD3 + UISETP. `#pragma unroll 1` keeps body too small to amortize branch overhead on ALU pipe. V8's 128-deep unroll amortizes 16× better. Architectural question (does B300 SMSP dual-issue FMA+ALU?) **remains OPEN**, V52 settles. |
| Same-warp dual-issue FFMA + IADD3 — `54 %` | LOW / DOWNGRADED | MED / RE-PROMOTED | **LOW / RE-DOWNGRADE-2** | RE-DOWNGRADE-2 | Same trail | Loop-overhead contamination per SASS analysis (wave-5a); inner body 8 ops/type + `#pragma unroll 1` too small to amortize BRA + UIADD3 + UISETP overhead on ALU pipe. |
| Same-warp dual-issue FFMA + PRMT — `51 %` | LOW / DOWNGRADED | MED / RE-PROMOTED | **LOW / RE-DOWNGRADE-2** | RE-DOWNGRADE-2 | Same trail | Loop-overhead contamination per SASS analysis (wave-5a); inner body 8 ops/type + `#pragma unroll 1` too small to amortize BRA + UIADD3 + UISETP overhead on ALU pipe. |
| Warp-specialized FFMA + LOP3 (4+4 warps) — `74 %` | LOW / DOWNGRADED | MED / RE-PROMOTED | **LOW-MED / RE-DOWNGRADE-2** | RE-DOWNGRADE-2 | HIGH → LOW → MED → **LOW-MED (W5a: same loop-overhead, body is per-warp homogeneous so cleaner than V49 mixed body — slightly better but still not V8-comparable)** | Loop-overhead contamination per SASS analysis (wave-5a); per-warp body cleaner than V49's interleaved body but still 8 ops/type + `#pragma unroll 1`. V52 with 128-deep unroll required to settle. |

## Section 3 — DSMEM (KEEP from v1, unchanged by W5a)

| Row | Old grade | v1-patch / v2-patch grade | Wave 5 status | Latest verdict (post-W5a) | Reason |
|---|---|---|---|---|---|
| DSMEM read aggregate per cluster (4×4 ring) — `40 GB/s` | MED / DOWNGRADED | **MED / KEEP** (caveat preserved) | KEEP | MED / KEEP | V21 source-verified as chain-bound; non-chained ILP could be 60-80. V53 settles. |
| DSMEM write aggregate per cluster — `560 GB/s` | LOW-MED / DOWNGRADED | **LOW-MED / KEEP** (caveat preserved) | KEEP | LOW-MED / KEEP | `push_ring_wr` lines 96-105 has NO fence between stores and timing. V53 + `fence.sc.cluster` settles. |
| DSMEM ring contention (N=1..8 readers) — `1.00× flat`, "NO shared bus" | LOW / DOWNGRADED | **LOW / KEEP** | KEEP | LOW / KEEP | V17 was 30× under-issued. Architectural claim unproven from V17. |

## Section 4 — NVFP4 A:B (KEEP from v1, unchanged by W5a)

| Row | Old grade | v1-patch / v2-patch grade | Wave 5 status | Latest verdict (post-W5a) | Reason |
|---|---|---|---|---|---|
| NVFP4 K=96 A:B asymmetry (3 mechanism readings) | MED / preserved | **MED / KEEP** — all 3 framings retained pending V56 | KEEP | MED / KEEP | Doubt agent CITED, not invented, the wave-3 caveats. V56 (4-mode discriminator) settles. |

## Section 5 — Top-10 LOW list (RESTORE dual-issue at #1)

v1-patch dropped dual-issue from the LOW list (because of the LOW→MED
promotion). v2-patch RESTORES it at #1. The original ordering returns:

1. **Same-warp / warp-spec dual-issue (55 %, 54 %, 51 %, 74 %)** — RESTORED (W5a SASS: loop-overhead contamination, V52 retest queued)
2. `__threadfence_system` 1.74× spread
3. HBM write SoL 7.57 TB/s provenance
4. DSMEM read 40 GB/s chain-bound
5. DSMEM write 560 GB/s issue-rate-only
6. DSMEM "NO shared bus" V17 under-issued
7. `__threadfence` GPU 281 cy 24 % spread
8. IADD3 rate 0.50 vs 0.66
9. PRMT rate V40 0.36 vs A6 0.50
10. PCIe Gen 6 cap mechanism RETRACTED, root cause UNCONFIRMED

## Section 6 — HIGH-row caveat survey (UNCHANGED FROM v1)

Survey of the 81 `doubt-confirmed` HIGH rows for any whose underlying argument
resembled the dual-issue mechanism error: **none re-warrant a downgrade**.
HBM/L2/SHMEM/FFMA/FP64/Sync/NVLink/PCIe/Math/Power doubt-confirmations rest
on cross-test agreement, not on mechanism inference. No additional HIGH rows
need caveat-bumping.

---

## Updated distribution

Original (v1 of `CONFIDENCE_LADDER`, 303 rows):
| HIGH | MED | LOW (incl. LOW-MED) | DISPUTED |
|---:|---:|---:|---:|
| 259 | 31 | 11 | 2 |

After v1-patch (W5d, post-wave-5b, 303 rows): **WRONG (now superseded)**
| HIGH | MED | LOW (incl. LOW-MED) | DISPUTED |
|---:|---:|---:|---:|
| 259 | 35 (+4) | 7 (-4) | 2 |

After v2-patch (W5a SASS RE-DOWNGRADE-2, 303 rows): **CURRENT**
| HIGH | MED | LOW (incl. LOW-MED) | DISPUTED |
|---:|---:|---:|---:|
| 259 | 31 | **11** (back to original) | 2 |

Net movement vs original: ZERO. The W5d → W5a zigzag cancels out at the
distribution level. HBM denominator rows stay HIGH; only their annotation
changes (NEW-CITE).

---

## Iteration history (zigzag summary for the 4 dual-issue rows)

| Wave | Verdict | Reason | Disposition |
|---|---|---|---|
| W1+W2 | HIGH | V49/V50 measurement reproducible | wrong (number is unsafe) |
| W3b doubt | LOW (DOWNGRADED) | Suspected under-occupancy (2 warps/SMSP × 8 ILP too thin) | right answer, partly right reason |
| W4 meta-doubt | MED (RE-PROMOTED) | V8_FFMA hits 97.6 % at the IDENTICAL 2 warps/SMSP geometry → under-occupancy hypothesis falsified | wrong answer, valid falsification of W3b's *specific* mechanism |
| W5d patch | MED (mirrored W4) | Applied W4 verdict to ladder | superseded by W5a |
| **W5a SASS-verify** | **LOW (RE-DOWNGRADE-2)** | Loop-overhead contamination: V49 inner body = 8 FFMA + 8 LOP3 + BRA + UIADD3/UISETP. V8 unrolls 128-deep, amortizing branch overhead 16×. Source-pattern (immediates vs registers) is NOT the issue; SASS shows both compile to single-RF-read FFMA. | right answer, NEW reason — current best |

**Meta-rule (HEADLINE_v4 §lessons):** When a verdict has flipped ≥ 2×, do
NOT publish as HIGH or even MED until a fresh test eliminates ALL hypothesized
failure modes simultaneously. Publish as LOW with the OPEN question explicit.

---

## How to apply this v2 patch

1. In `CONFIDENCE_LADDER.md` §12, edit the 4 dual-issue rows to grade **LOW**
   (3 entries) and **LOW-MED** (V50 warp-spec) with note "RE-DOWNGRADED by
   SASS_VERIFY_DUAL_ISSUE.md (wave-5a); loop-overhead contamination from
   tiny inner body; V52 retest required to settle architectural question".
2. In `CONFIDENCE_LADDER.md` §1, update HBM denominator row to read
   `7680 (spec) / 7672 (this-device empirical)` and cross-reference
   `HBM_DENOMINATOR_FINAL.md` (unchanged from v1-patch).
3. Reframe V46 % column to "93.75 % post-ECC spec / 93.85 % this-device"
   (unchanged from v1-patch).
4. Update summary statistics block at the bottom: HIGH 259 / MED 31 / LOW 11
   / DISPUTED 2 (back to original counts; v1-patch's 35/7 was wrong).
5. RESTORE dual-issue at #1 of the top-10 LOW list (v1-patch's renumbering
   to top-9 is REVERTED).

---

## Expected next update (V52 / wave-6a is running NOW)

V52 (`tests/standalone/v52_*.cu`, currently executing per session log) is the
purpose-built fix for the W5a-identified bug. Recipe per
`SASS_VERIFY_DUAL_ISSUE.md` §6 / `WAVE5_CHANGES.md` §1.3:

1. Inner unroll depth ≥ 64 ops per type (vs V49's 8)
2. `__launch_bounds__(256, 1)` (match V8, vs V49's `(128, 2)`)
3. Anti-DCE via STG of accumulator XOR (vs V49's clock-diff conditional)
4. ncu reads `sm__inst_executed_pipe_fma.avg.pct_of_peak` AND
   `sm__inst_executed_pipe_alu.avg.pct_of_peak` simultaneously

**Possible V52 outcomes and consequent v3-patch shape:**

| V52 result | Architectural reading | Resulting v3-patch action |
|---|---|---|
| Same-warp dual-issue ≥ 90 % at full unroll + dual ncu pipes confirm | B300 SMSP DOES dual-issue FMA + ALU same-warp; W1+W2 numbers were just measurement-undersaturated | RE-PROMOTE 4 rows LOW → HIGH; update HEADLINE_v5 |
| Same-warp dual-issue stays in [55–80 %] window even at 128-deep unroll | Partial dual-issue (architectural cap, not measurement bug) | RE-PROMOTE LOW → MED; ncu pipe utilization decides exact framing |
| Same-warp dual-issue collapses to ≤ 55 % even at full unroll | Confirms NO same-warp FMA+ALU dual-issue; warp-spec (74 %) is the only path | KEEP LOW; promote V50 warp-spec LOW-MED → MED |
| V52 has its own methodology bug | Another zigzag layer; verdict stays LOW pending V53 | KEEP LOW; v3-patch documents new bug class |

**Whichever it is, expect a CONFIDENCE_LADDER_PATCH_v3.md within hours of V52
results landing.** The dual-issue verdict is the most-iterated claim in the
catalog and is unlikely to be settled in fewer than 5 waves.

---

## Files referenced

- `/root/github/QuickRunCUDA/b300_clean/corrections/CONFIDENCE_LADDER.md` (v1, 303 rows)
- `/root/github/QuickRunCUDA/b300_clean/corrections/CONFIDENCE_LADDER_PATCH.md` (v1-patch / wave-5d, STALE)
- `/root/github/QuickRunCUDA/b300_clean/corrections/SASS_VERIFY_DUAL_ISSUE.md` (wave-5a, source of v2 re-downgrade)
- `/root/github/QuickRunCUDA/b300_clean/corrections/WAVE5_CHANGES.md` (consolidates the zigzag)
- `/root/github/QuickRunCUDA/b300_clean/corrections/HEADLINE_CORRECTIONS_v4.md` (current authoritative top-line)
- `/root/github/QuickRunCUDA/b300_clean/corrections/META_DOUBT_REPORT.md` (wave-4, source of W5d's now-overruled MED promote)
- `/root/github/QuickRunCUDA/b300_clean/corrections/HBM_DENOMINATOR_FINAL.md` (wave-5b)
- `/root/github/QuickRunCUDA/tests/standalone/v52_*.cu` (running NOW, will trigger v3-patch)
