# CONFIDENCE LADDER — Wave-5 Patch (2026-04-22)

**Purpose:** delta against `CONFIDENCE_LADDER.md` (v1, 303 rows). Applies
`META_DOUBT_REPORT.md` (wave-3d), `WAVE4_CHANGES.md` (wave-4), and
`HBM_DENOMINATOR_FINAL.md` (wave-5b) to re-grade rows whose verdict shifted.

**New "Wave 5 status" tags used here:**
- `KEEP` — row stands as graded
- `RE-PROMOTE` — wave-3b downgrade was overstated; raise grade
- `RE-DOWNGRADE` — wave-3b promote/HIGH was overstated; lower grade
- `NUANCED` — value stands but framing or denominator needs paired statement
- `NEW-CITE` — denominator/source line edited for both 7680 and 7672

Only rows whose grade or annotation changes are listed; everything else
is `KEEP` and not reproduced here.

---

## Section 1 — HBM denominator (wave-5b reversal)

| Row | Old grade / annotation | New grade / annotation | Wave 5 status | Reason |
|---|---|---|---|---|
| HBM3E theoretical post-ECC spec (denominator) — value `7672 GB/s` | HIGH / doubt-confirmed | **HIGH / NUANCED** — both **7680** (spec, 8.000 Gbps/pin) **and 7672** (this-device, empirical 7.992 Gbps/pin at 3996 MHz I/O) are valid; cite both | NEW-CITE | Wave-4 retired 7672 as "ghost"; wave-5b (`HBM_DENOMINATOR_FINAL.md`) shows `nvidia-smi -q` reports **3996 MHz Memory clock**, so 7672 is the literal hardware rate. 0.10 % below the 7680 spec is real silicon under-spec. |
| HBM read (V46 TMA 8-deep pipelined) — `7.20 TB/s` REFRAMED | HIGH / "98.5% used wrong denom; 7.20/7672=93.8%" | **HIGH / NUANCED** — 93.75 % of 7680 spec **and** 93.85 % of 7672 actual; both are correct under their framings | NEW-CITE | Wave-4 reframing math used 7672 as the "correct" anchor and 7680 as the alternative; wave-5b promotes **7680 to default citation**, with 7672 paired when SoL precision matters. |
| (implicit denominator note) every "% of HBM peak" row | HIGH / 7672 implicit | **HIGH / NUANCED** — default to 7680, optionally pair with 7672 | NEW-CITE | Methodology rule 11 (HEADLINE_v3 §6) standardizes 7680 as default; v5b lifts the 7672 ban. |

## Section 2 — Dual-issue (wave-3d meta-doubt reversal)

| Row | Old grade | New grade | Wave 5 status | Reason |
|---|---|---|---|---|
| Same-warp dual-issue FFMA + LOP3 — `55 %` | LOW / DOWNGRADED | **MED / RE-PROMOTED** | RE-PROMOTE | `META_DOUBT_REPORT.md` §1: V8_FFMA_PEAK_VERIFIED hits 97.64 % at the **identical 2 warps/SMSP** geometry V49 uses (`__launch_bounds__(256,1)`). The under-occupancy mechanism the doubt agent invoked is FALSIFIED. The 55 % measurement is reproducible; only the dispatch-cap interpretation is underdetermined. V52 settles. |
| Same-warp dual-issue FFMA + IADD3 — `54 %` | LOW / DOWNGRADED | **MED / RE-PROMOTED** | RE-PROMOTE | Same chain of reasoning. |
| Same-warp dual-issue FFMA + PRMT — `51 %` | LOW / DOWNGRADED | **MED / RE-PROMOTED** | RE-PROMOTE | Same chain of reasoning. |
| Warp-specialized FFMA + LOP3 (4+4 warps) — `74 %` | LOW / DOWNGRADED | **MED / RE-PROMOTED** | RE-PROMOTE | Same; the 74 % is reproducible, the 4-wide dispatch cap interpretation needs ncu (V52). |

## Section 3 — DSMEM (meta-doubt confirmed sound; KEEP)

| Row | Old grade | New grade | Wave 5 status | Reason |
|---|---|---|---|---|
| DSMEM read aggregate per cluster (4×4 ring) — `40 GB/s` | MED / DOWNGRADED | **MED / KEEP** (caveat preserved) | KEEP | `META_DOUBT_REPORT.md` §3: V21 source-verified as chain-bound; non-chained ILP could be 60-80. Caveat stands; V53 settles. |
| DSMEM write aggregate per cluster — `560 GB/s` | LOW-MED / DOWNGRADED | **LOW-MED / KEEP** (caveat preserved) | KEEP | §3: `push_ring_wr` lines 96-105 has NO fence between stores and timing. "Issue rate, not completion" stands. V53 + `fence.sc.cluster` settles. |
| DSMEM ring contention (N=1..8 readers) — `1.00× flat`, "NO shared bus" | LOW / DOWNGRADED | **LOW / KEEP** | KEEP | §3: V17 was 30× under-issued (1 thr/CTA, single-issue). Architectural claim unproven from V17. Stays LOW. |

## Section 4 — NVFP4 A:B (meta-doubt confirmed; KEEP all 3 readings)

| Row | Old grade | New grade | Wave 5 status | Reason |
|---|---|---|---|---|
| NVFP4 K=96 A:B asymmetry (3 mechanism readings) — entries spread across §7 | MED / preserved | **MED / KEEP** — all 3 framings retained pending V56 | KEEP | `META_DOUBT_REPORT.md` §4: doubt agent CITED, not invented, the wave-3 caveats. Wave-2's single-mechanism story is over-resolved. V56 (4-mode discriminator) settles. |

## Section 5 — Top-10 LOW list reordering

The top-10 LOW/DISPUTED list at the bottom of `CONFIDENCE_LADDER.md` items
**#1 (dual-issue 55 %/74 %) drops off LOW list** — promoted to MED. The
"needs ncu + warps sweep" action remains on the V52 retest queue.

Renumbered top-9 LOW/UNRESOLVED:
1. `__threadfence_system` 1.74× spread (was #2)
2. HBM write SoL 7.57 TB/s provenance (was #3)
3. DSMEM read 40 GB/s chain-bound (was #4)
4. DSMEM write 560 GB/s issue-rate-only (was #5)
5. DSMEM "NO shared bus" V17 under-issued (was #6)
6. `__threadfence` GPU 281 cy 24 % spread (was #7)
7. IADD3 rate 0.50 vs 0.66 (was #8)
8. PRMT rate V40 0.36 vs A6 0.50 (was #9)
9. PCIe Gen 6 cap mechanism RETRACTED, root cause UNCONFIRMED (was #10)

## Section 6 — HIGH entries that came from "doubt-confirmed" sources — caveat survey

`META_DOUBT_REPORT.md` shows wave-3b can be wrong on mechanism even when
grading "doubt-confirmed". Survey of the 81 `doubt-confirmed` HIGH rows
for any whose underlying argument resembles the dual-issue mechanism error:

- **None in HBM/L2/SHMEM/FFMA/FP64/Sync/NVLink/PCIe/Math/Power sections
  re-warrant a downgrade**: those doubt-confirmations rest on cross-test
  agreement (multiple independent recipes hit the same number), not on
  mechanism inference.
- **One borderline:** "L2 atomic units count `~32` — DOWNGRADED" already
  carries the "ATOMIC_REVERIFY_DEEP says ceiling could be higher"
  qualifier. No change.
- **One borderline:** "FFMA realistic 3-distinct-source GEMM ~50 TFLOPS"
  is HIGH on multi-source consensus (A4 + D6 + V10). No change.

Net: no additional HIGH rows require caveat-bumping based on the
meta-doubt finding.

---

## Updated distribution

Old (303 rows):
| HIGH | MED | LOW (incl. LOW-MED) | DISPUTED |
|---:|---:|---:|---:|
| 259 | 31 | 11 | 2 |

New (post-wave-5b, 303 rows):
| HIGH | MED | LOW (incl. LOW-MED) | DISPUTED |
|---:|---:|---:|---:|
| 259 | **35** (+4) | **7** (-4) | 2 |

Movement: 4 dual-issue rows promoted LOW → MED. No HIGH↔MED shifts.
HBM denominator rows stay HIGH; only their annotation changes (NEW-CITE).

---

## How to apply this patch

1. In `CONFIDENCE_LADDER.md`, edit the 4 dual-issue rows in §12 to grade
   **MED** with note "DOWNGRADE REVERSED by META_DOUBT_REPORT §1; ncu
   confirmation pending (V52)".
2. Update the HBM denominator row in §1 to read
   `7680 (spec) / 7672 (this-device empirical)` and add the cross-reference
   to `HBM_DENOMINATOR_FINAL.md`.
3. Reframe V46's % column to "93.75 % post-ECC spec / 93.85 % this-device".
4. Update summary statistics block at the bottom (HIGH 259 / MED 35 / LOW 7).
5. Drop dual-issue from top-10 LOW; renumber to top-9 as listed in §5 above.
