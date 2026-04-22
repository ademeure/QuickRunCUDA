# CONFIDENCE LADDER — Wave-6 Patch v3 (2026-04-22)

**Purpose:** delta against `CONFIDENCE_LADDER.md` (v1, 303 rows). Supersedes
both `CONFIDENCE_LADDER_PATCH.md` (W5d) and `CONFIDENCE_LADDER_PATCH_v2.md`
(W5a/W5f), which are now STALE.

**Why v3 exists:** V52 (`v52_dual_issue_clean.cu`) ran with V8-style 128-deep
unroll AND simultaneous ncu `pipe_alu` + `pipe_fma` reads. Result: solo FFMA
`pipe_fma = 97.6 %`, solo LOP3 `pipe_alu = 99.5 %`, **dual `pipe_alu = 98.0 %`
AND `pipe_fma = 49.4 %` simultaneously**, sum = **147 %**. The pipes overlap
freely. The "dispatch cap" hypothesized by W3b/W5a was wrong. The previous
LOW grade (v2) is **architecturally retracted**, but the V49/V50 numerical
claims (55 %, 74 %) are still RETRACTED as measurement artifacts.

**Wave-6 status tags:**
- `KEEP` — row stands as graded
- `RE-PROMOTE` — wave-3b/wave-5a downgrade was overstated (architecturally); raise to HIGH
- `RETRACT-NUMBER` — confidence in MEASUREMENT promoted but the number itself is methodology-tainted; flag the number RETRACTED
- `NEW-ROW` — row added to capture wave-6 finding

---

## Section 1 — HBM denominator (W5b/W6b NEW-CITE) — UPDATED

| Row | Old grade / annotation | New grade / annotation | Wave 6 status | Reason |
|---|---|---|---|---|
| HBM3E theoretical post-ECC spec (denominator) | HIGH / both 7680 + 7672 | **HIGH / NEW-CITE** — both **7.68 TB/s** (spec, full-bus) **and 7.67 TB/s** (this-device, 7680-bit fused-AC SKU); cite the latter for SoL on this hardware | NEW-CITE | `HBM_STACKS_INDEPENDENT_VERIFY.md` confirms 8 stacks present; this-device bus width = 7680 bits per `cudaGetDeviceProperties`. |
| HBM read (V46 TMA 8-deep) — `7.20 TB/s` | HIGH / 93.75 % of 7680 / 93.85 % of 7672 | **HIGH / NEW-CITE** — **93.9 % of 7.67 TB/s this-device peak** (vs 88.8 % of 8 TB/s marketing) | NEW-CITE | Re-anchored against the smaller this-device denominator. |
| (implicit) every "% of HBM peak" row | HIGH / 7672 implicit | HIGH / 7.67 TB/s default for this-device, 7.68 TB/s for cross-vendor | NEW-CITE | Methodology rule 11 (HEADLINE_v5) replaces rule 11 (HEADLINE_v4). |

## Section 2 — Dual-issue (W6a EMPIRICAL SETTLEMENT) — RE-PROMOTED to HIGH

| Row | v1 (orig) | v2 patch (W5a) | **v3 patch (W6a)** | Wave 6 status | Reason |
|---|---|---|---|---|---|
| Same-warp dual-issue FFMA + LOP3 — `55 %` measurement | LOW / DOWNGRADED | LOW / RE-DOWNGRADE-2 | **HIGH (measurement) + RETRACT-NUMBER** | RE-PROMOTE / RETRACT-NUMBER | V52 confirms the measurement IS reproducible (so confidence in the number is HIGH), but the number itself is loop-overhead artifact (so the architectural reading "55 % is the dispatch cap" is RETRACTED). The architectural truth (next row) supersedes. |
| Same-warp dual-issue FFMA + IADD3 — `54 %` measurement | LOW | LOW / RE-DOWNGRADE-2 | **HIGH (measurement) + RETRACT-NUMBER** | RE-PROMOTE / RETRACT-NUMBER | Same as above. |
| Same-warp dual-issue FFMA + PRMT — `51 %` measurement | LOW | LOW / RE-DOWNGRADE-2 | **HIGH (measurement) + RETRACT-NUMBER** | RE-PROMOTE / RETRACT-NUMBER | Same as above. |
| Warp-specialized FFMA + LOP3 (4+4 warps) — `74 %` measurement | LOW | LOW-MED / RE-DOWNGRADE-2 | **HIGH (measurement) + RETRACT-NUMBER** | RE-PROMOTE / RETRACT-NUMBER | Warp-spec helped because it hid loop overhead, not because it broke a cap; same root cause. |

## Section 3 — Dual-issue (NEW ARCHITECTURAL ROWS from W6a)

| Row | Grade | Reason |
|---|---|---|
| **NEW: B300 SMSP FMA + ALU pipes overlap freely (`pipe_alu + pipe_fma ≈ 147 %`)** | **HIGH** | V52 ncu Geometry A ILP=8: pipe_alu=98.0 %, pipe_fma=49.4 %, inst_issued/cy=1.00 (vs 0.51 solo LOP3). Three independent runs reproducible within 1 %. |
| **NEW: LOP3 has 2-cycle issue cadence per SMSP** | **HIGH** | V52 solo LOP3 ncu: pipe_alu=99.5 % at inst_issued/cy=0.51. Solo LOP3 GLane/s ≈ ½ × solo FFMA GLane/s, exactly matching the 0.51 vs 1.00 ratio. |
| **NEW: M8 PIPE_OVERLAP_MATRIX (MUFU+FFMA ≈ 100 %) is architecturally consistent** | **HIGH** | V52 confirms the underlying claim that B300 SMSPs do not enforce a shared dispatch cap; the W3b "M8 has counter-evidence" remark was correct. |

## Section 4 — DSMEM (KEEP from v2, unchanged by W6a)

| Row | Grade | Wave 6 status |
|---|---|---|
| DSMEM read aggregate per cluster (4×4 ring) — `40 GB/s` | MED / KEEP | KEEP |
| DSMEM write aggregate per cluster — `560 GB/s` | LOW-MED / KEEP | KEEP |
| DSMEM ring contention (N=1..8 readers) — "NO shared bus" | LOW / KEEP | KEEP |

## Section 5 — NVFP4 A:B (KEEP from v2)

| Row | Grade | Wave 6 status |
|---|---|---|
| NVFP4 K=96 A:B asymmetry (3 mechanism readings) | MED / KEEP | KEEP |

## Section 6 — Top-10 LOW list — REORDERED

The four V49/V50 dual-issue rows are REMOVED from the LOW list (architectural inference retracted; the measurements are flagged RETRACTED separately). New top-10 LOW:

1. `__threadfence_system` 1.74× spread (1750 / 2870 / 3042 cy)
2. HBM write SoL 7.57 TB/s NINJA provenance
3. DSMEM read 40 GB/s chain-bound
4. DSMEM write 560 GB/s issue-rate-only
5. DSMEM "NO shared bus" V17 under-issued
6. `__threadfence` GPU 281 cy 24 % spread
7. IADD3 rate 0.50 vs 0.66 (UNRESOLVED)
8. PRMT rate V40 0.36 vs A6 0.50 (UNRESOLVED)
9. PCIe Gen 6 cap mechanism RETRACTED, root cause UNCONFIRMED
10. V51 multi-stream HBM (in-progress bug-fix)

## Section 7 — HIGH-row caveat survey — UNCHANGED

The 81 doubt-confirmed HIGH rows still rest on cross-test agreement, not on shared-dispatch inference. No additional HIGH rows need re-grading.

---

## Updated distribution

| Patch level | HIGH | MED | LOW (incl. LOW-MED) | DISPUTED |
|---|---:|---:|---:|---:|
| Original v1 (303 rows) | 259 | 31 | 11 | 2 |
| v1-patch (W5d) — STALE | 259 | 35 (+4) | 7 (-4) | 2 |
| v2-patch (W5a) — STALE | 259 | 31 | 11 | 2 |
| **v3-patch (W6a) — CURRENT (303 + 3 NEW rows = 306)** | **263** (4 dual-issue measurements RE-PROMOTED + 3 NEW HIGH rows − 3 absorbed by NEW rows) | **27** (-4 dual-issue moved out) | **11** (unchanged; the 4 dual-issue rows are flagged RETRACT-NUMBER but counted in HIGH-measurement) | 2 |

Net movement vs v2: 4 dual-issue rows promoted LOW → HIGH (with RETRACT-NUMBER flag), 3 NEW HIGH architectural rows added.

---

## Iteration history (5-level zigzag, final)

| Wave | Verdict | Mechanism cited | Disposition |
|---|---|---|---|
| W1+W2 | HIGH | reproducibility | wrong: loop-overhead artifact |
| W3b doubt | LOW | under-occupancy + dispatch cap | partly right (number unsafe), partly wrong (cap doesn't exist) |
| W4 meta-doubt | MED | V8 falsifies under-occupancy | right falsification, wrong verdict |
| W5d patch | MED | mirrored W4 | superseded |
| W5a SASS-verify | LOW | loop-overhead contamination | right about contamination, wrong about implied cap |
| **W6a V52 + ncu** | **HIGH (architectural overlap = TRUE) + RETRACT-NUMBER** | empirical `alu + fma = 147 %` | settled |

**Meta-rule (HEADLINE_v5 §lessons):** "the artifact is real" and "the architectural inference from the artifact is real" are TWO independent claims. State them separately. W3b/W5a conflated them.

---

## How to apply this v3 patch

1. In `CONFIDENCE_LADDER.md` §12, edit the 4 dual-issue rows:
   - Grade: **HIGH** (measurement)
   - Note: "The 55 %/74 %/etc. numbers are CONFIRMED reproducible but RETRACTED as architectural claims — V52 ncu shows pipe_alu + pipe_fma = 147 %, proving free overlap. The numbers were loop-overhead methodology artifacts; the implied dispatch cap does not exist. See `WAVE6_CHANGES.md` and `V52_RUN_RESULTS.md`."
2. ADD 3 NEW rows to §12 capturing the architectural truth:
   - "B300 SMSP FMA + ALU pipes overlap freely (alu + fma ≈ 147 %)" — HIGH (V52 ncu, 3-run reproducible).
   - "LOP3 has 2-cycle issue cadence per SMSP" — HIGH.
   - "M8 PIPE_OVERLAP_MATRIX is architecturally consistent" — HIGH.
3. In `CONFIDENCE_LADDER.md` §1, update HBM denominator to read `7.68 TB/s (spec, 8.000 Gbps × 8192 bits ÷ 1.0625) / 7.67 TB/s (this-device, 7.992 Gbps × 7680 bits ÷ 1.0625, AC SKU with 1/16 controllers fused)` and cross-reference `HBM_STACKS_INDEPENDENT_VERIFY.md`.
4. Reframe V46 % column to "93.9 % of this-device 7.67 TB/s peak" and remove the "88.8 % of 8 TB/s" framing.
5. Update summary statistics block at the bottom to v3 distribution above.
6. REORDER top-10 LOW list per §6 above.

---

## Files referenced

- `b300_clean/corrections/CONFIDENCE_LADDER.md` (v1, 303 rows)
- `b300_clean/corrections/CONFIDENCE_LADDER_PATCH.md` (W5d, STALE)
- `b300_clean/corrections/CONFIDENCE_LADDER_PATCH_v2.md` (W5a, STALE)
- `b300_clean/corrections/V52_RUN_RESULTS.md` (W6a empirical anchor)
- `b300_clean/corrections/HBM_STACKS_INDEPENDENT_VERIFY.md` (W6b denominator update)
- `b300_clean/corrections/WAVE6_CHANGES.md` (consolidates W6 deltas)
- `b300_clean/corrections/HEADLINE_CORRECTIONS_v5.md` (current authoritative top-line)
- `b300_clean/corrections/META_LESSONS.md` (distilled rigor rules from 5-level zigzag)
