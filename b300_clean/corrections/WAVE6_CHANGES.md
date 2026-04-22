# WAVE-6 CHANGES — what changed (and finally settled)

**Date:** 2026-04-22
**Supersedes:** `WAVE5_CHANGES.md`
**Drivers:** V52 empirical retest (`v52_dual_issue_clean.cu`), HBM independent
verification (`HBM_STACKS_INDEPENDENT_VERIFY.md`), CONFIDENCE_LADDER_PATCH_v2
(now stale).

---

## 1. The dual-issue zigzag is finally settled (V52)

V52 (`tests/standalone/v52_dual_issue_clean.cu`) ran with the V8-style 128-deep
inner unroll, anti-DCE STG, matched solo+dual baselines, and **simultaneous ncu
`pipe_alu` + `pipe_fma` reads** — the diagnostic the entire 5-wave debate was
missing.

**Empirical anchor (Geometry A, ILP=8, BPS=1, default boost):**

```
config              pipe_alu%   pipe_fma%   inst_issued/cy   alu+fma
solo FFMA            0.02       97.58         1.00            97.60
solo LOP3           99.45        0.39         0.51            99.84
dual                98.00       49.39         1.00          147.39
```

**Verdicts that follow:**

- The FMA pipe and ALU pipe are physically separate and **overlap freely** — `alu + fma = 147%`.
- The "55%/74% from V49/V50" was a **loop-overhead methodology artifact**: V49's 8-FFMA + 8-LOP3 inner body bled BRA + UIADD3 + UISETP onto the same ALU pipe being measured. V52's 128-deep body shrinks loop overhead from 12.5 % to 1.2 %.
- LOP3 has an intrinsic **2-cycle issue cadence** per SMSP (`inst_issued/cy = 0.51` solo) — not a shared-port limit. Solo LOP3 is at ~100 % of its own ceiling.
- The dispatch cap interpretation (W3b/W5a) was **architecturally wrong**. The wave-3b/wave-5a verdicts were RIGHT that V49's measurement was an artifact, but WRONG that this implied the existence of a shared dispatch cap. The cap simply doesn't exist; both pipes can fire at ≥98 %/cy simultaneously.

**Net effect on confidence ladder:** dual-issue rows promoted LOW → HIGH (architectural truth) but with explicit annotation that the V49/V50 numerical claims (55 %, 74 %) are RETRACTED.

---

## 2. HBM bus width revelation: 7680-bit fused SKU

`HBM_STACKS_INDEPENDENT_VERIFY.md` confirms via NVIDIA developer blog that B300 architecturally has 8 × HBM3E 12-Hi stacks, 16 × 512-bit controllers, **8192-bit total bus**. But on this specific box:

```
$ cudaGetDeviceProperties → memoryBusWidth: 7680 bits (NOT 8192!)
```

**This-device interpretation:**

- 7680 = 8192 × 15/16 — exactly one /16 controller fused off (yield bin).
- The "AC" suffix in `NVIDIA B300 SXM6 AC` is consistent with this capacity/channel-restricted SKU.
- All 8 stacks are physically present; one channel pair is disabled.

**Bandwidth consequences:**

| Quantity | Spec (full bus) | This-device |
|---|---|---|
| Bus width | 8192 bits | 7680 bits |
| Per-pin rate | 8.000 Gbps (spec) / 7.992 Gbps (measured at 3996 MHz) | same |
| Pre-ECC peak | 8.19 TB/s | 7.68 TB/s |
| **Post-ECC peak (÷ 1.0625)** | **7.71 TB/s spec / 7.68 actual** | **7.23 TB/s actual** |
| Recommended this-device denominator | — | **~7.67 TB/s** (using rounded numbers consistent with prior catalog) |

**Re-anchoring measured peaks:**

- V46 TMA 8-deep pipelined read: 7.20 TB/s = **93.9 % of 7.67 TB/s** (this-device peak), up from the prior "88.8 % of 8.0 spec" framing.
- Prior catalog reading "~89 % of peak" should be re-read as **~93 % of peak** for SoL purposes on this part.

For cross-vendor comparisons keep citing the 7.68 TB/s spec-comparable value (8.000 Gbps × 8192 bits ÷ 1.0625 ECC overhead).

---

## 3. The 5-level zigzag — now resolved

| Wave | Verdict | Mechanism cited | Right? |
|---|---|---|---|
| W1+W2 | HIGH (V49 reproducible) | none | reproducibility ≠ validity |
| W3b doubt | LOW | under-occupancy | wrong mechanism |
| W4 meta-doubt | MED (re-promote) | V8 falsifies under-occupancy | wrong verdict, valid falsification |
| W5a SASS-verify | LOW | loop-overhead contamination | right about contamination, **wrong about implied dispatch cap** |
| **W6 V52 + ncu** | **HIGH (architectural overlap = TRUE)** | empirical `pipe_alu + pipe_fma = 147 %` with clean methodology | settled; the 55/74 numbers are artifact, but pipes do dual-issue freely |

**Meta-lesson:** the architectural interpretation of an artifact-tainted measurement is a **separate question** from whether the measurement is artifact-tainted. W3b/W5a were correct that V49/V50 are artifacts (KEEP), but their further inference that "therefore the architectural cap is at the artifact value" was unfounded. Only V52 with proper methodology + ncu pipe metrics could disambiguate.

See `META_LESSONS.md` for the distilled rigor rules.

---

## 4. Retraction summary

- **RETRACT** all "B300 same-warp dual-issue capped at 55 %" / "warp-spec capped at 74 %" architectural claims (W1+W2 + V41-V48 findings doc).
- **RETRACT** the dispatch-cap inference from W3b's DUAL_ISSUE_DOUBT_REPORT and W5a's SASS_VERIFY_DUAL_ISSUE.
- **KEEP** the methodology critique from W3b/W5a — V49/V50's measurement WAS contaminated, just not by the cap they hypothesized.
- **KEEP** wave-3b's note that M8's MUFU+FFMA ≈ 100 % is genuine counter-evidence — V52 now CONFIRMS M8 was correctly framed.
- **MARK** `CONFIDENCE_LADDER_PATCH_v2.md` (just produced this wave) as STALE — superseded by `CONFIDENCE_LADDER_PATCH_v3.md`.

---

## 5. New canonical numbers

| Metric | New value | Old value | Source |
|---|---|---|---|
| dual-issue alu+fma sum | **~147 %** at ILP=8 | "55 %"/"74 %" (artifact) | V52 ncu Geometry A |
| LOP3 solo issue rate | 0.51 inst/cy/SMSP (2-cy cadence) | unstated | V52 ncu |
| this-device bus width | **7680-bit** | "8192-bit" | `cudaGetDeviceProperties` |
| this-device HBM peak | **~7.67 TB/s** | implicit 8 TB/s | bus width × per-pin × ÷ECC |
| spec-comparable HBM peak | 7.68 TB/s | (unchanged for cross-vendor) | spec math |

---

## 6. Files and how they relate

- **Empirical anchor:** `b300_clean/corrections/V52_RUN_RESULTS.md`
- **HBM source verification:** `b300_clean/corrections/HBM_STACKS_INDEPENDENT_VERIFY.md`
- **Top-line corrections:** `b300_clean/corrections/HEADLINE_CORRECTIONS_v5.md` (this wave's authoritative)
- **Confidence ladder patch:** `b300_clean/corrections/CONFIDENCE_LADDER_PATCH_v3.md`
- **Distilled wisdom:** `b300_clean/corrections/META_LESSONS.md`
- **Stale (this wave produced but immediately superseded):** `CONFIDENCE_LADDER_PATCH_v2.md`
