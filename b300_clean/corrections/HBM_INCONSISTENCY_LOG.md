# HBM3E Bandwidth Inconsistency Log

Cross-file audit, Topic = HBM / DRAM bandwidth.
Sources audited:
- `01_hbm_bandwidth.md` (catalog)
- `V8_HBM_WRITE_SOL.md`
- `HBM_DATA_DEPENDENCE.md`
- `L2_DRAM_DATA_PWR.md`
- `V41_V48_FINDINGS.md`
- `V32_V40_FINDINGS.md`
- `B300_TRUE_REFERENCE.md`
- `CLAUDE.md` HBM section

---

## A. Theoretical-peak inconsistencies

| File | Theoretical peak quoted | Notes |
|---|---|---|
| `01_hbm_bandwidth.md` | **7672 GB/s post-ECC** (=7680 bit × 3996 MHz × 2 / 8 / 1e9) | Most rigorous derivation. Explicitly retires the pre-ECC 8183.8 number. |
| `B300_TRUE_REFERENCE.md` §1 | **7672 spec** in % column | Consistent with 01. |
| `V32_V40_FINDINGS.md` | **7.31 TB/s** repeatedly used as theoretical (V32-V36) | This is NOT the HBM3E spec; it is the empirical pure-direction peak. Using it as denominator inflates % numbers. |
| `V41_V48_FINDINGS.md` | "7.20 TB/s = **98.5% of HBM peak**" | Denominator is implicitly 7.31 (matches V32_V40 convention). vs 7672 spec, 7.20/7672 = **93.8%**, not 98.5%. |
| `V8_HBM_WRITE_SOL.md` | "HBM3E spec = **8 TB/s**. Effective read peak observed = 7.2 TB/s" | Uses 8 TB/s nominal AND 7.2 effective; reports both. |
| `CLAUDE.md` HBM section | "**~7-7.5 TB/s read peak (matches 8 TB/s spec)**" | Hand-wavy nominal; not aligned to the 7672 post-ECC number used elsewhere. |

**Inconsistency #1 (BIGGEST):** Three different denominators for "% of peak" appear across the corpus:
- 7672 GB/s (post-ECC, correct per 01_hbm_bandwidth.md) — used in `01` and `B300_TRUE_REFERENCE`.
- 7.31 TB/s (empirical pure-direction peak, NOT spec) — used in V32-V48 series.
- 8.0 / 8.18 TB/s (pre-ECC nominal) — used in V8_HBM_WRITE_SOL and old catalog text.

The "98.5%" headline in V41_V48 is therefore not directly comparable to the "95% spec" headlines in 01 and TRUE_REFERENCE.

---

## B. Headline read peak inconsistencies

| File | Read peak quoted | Recipe |
|---|---:|---|
| `01_hbm_bandwidth.md` table | **7.29 TB/s = 95.0%** of 7672 | v8 + per-warp + non-persistent (a04d9c8) |
| `01_hbm_bandwidth.md` sub-optimal table | **7344 GB/s = 95.7%** TMA bulk read | TMA cp.async.bulk |
| `01_hbm_bandwidth.md` sub-optimal table | **7365 GB/s = 96.0%** LDG.E.128 | "the HBM read SoL ceiling" |
| `01_hbm_bandwidth.md` R:W sweep | **7.31 TB/s = 95.3%** R-only | A6 rigor, dram__bytes ncu |
| `B300_TRUE_REFERENCE.md` | **7.30 TB/s = 95%** | Same recipe, citing a04d9c8 |
| `B300_TRUE_REFERENCE.md` SoL recipes | "Best HBM read 7.31 TB/s = 95.3%" | NINJA IT=2 |
| `V41_V48_FINDINGS.md` | **7.20 TB/s = 98.5%** TMA pipelined | V46, denominator 7.31 |
| `V8_HBM_WRITE_SOL.md` | "Effective read peak observed = 7.2 TB/s" | Anecdotal |
| `V32_V40_FINDINGS.md` V33 | 6.72 TB/s = 92% TMA per-CTA | After L2-cache fix |

**Inconsistency #2:** Multiple read peaks coexist (6.72, 7.20, 7.29, 7.30, 7.31, 7.344, 7.365) without a single canonical number. Range: 6.72 - 7.365 TB/s. The TRUE_REFERENCE picks 7.30 (a04d9c8); 01_hbm_bandwidth.md silently has both 7.30 (top) and 7.365 (later table). They differ by <1% but the documents do not reconcile.

---

## C. Headline write peak inconsistencies

| File | Write peak | Source |
|---|---:|---|
| `01_hbm_bandwidth.md` table | **7.30 TB/s = 95.2%** of 7672 | v8 + per-warp coalesced (a04d9c8) |
| `B300_TRUE_REFERENCE.md` | **7.30** | Same |
| `B300_TRUE_REFERENCE.md` NINJA | **7.57 TB/s = 98.7%** | NINJA recipe e75c7e1 (1 v8 store per warp) — beats cudaMemset by 5% |
| `V8_HBM_WRITE_SOL.md` (this file) | "TMA bulk store 7.57 TB/s = **105% (exceeds read peak)** = 95% of 8 TB/s" | TMA path, commit 28211ce |
| `V8_HBM_WRITE_SOL.md` (this file) | Plain STG.E.128 = **6.11 TB/s = 85%** of 7.2 TB/s | Plain path |
| `V41_V48_FINDINGS.md` V47 | TMA write pipelined = 6.34 TB/s "no benefit" | |
| `V32_V40_FINDINGS.md` V34 | TMA write 7.17 TB/s = 98% (denominator 7.31) | |

**Inconsistency #3 (significant):** The 7.57 TB/s headline.
- `B300_TRUE_REFERENCE.md` attributes 7.57 to "NINJA recipe (e75c7e1)" — i.e. a STG-based ninja kernel.
- `V8_HBM_WRITE_SOL.md` attributes 7.57 to **TMA bulk store (commit 28211ce)** and frames plain STG as capping at 6.11 TB/s.
- These two attributions cannot both be right. Either NINJA STG hits 7.57 (per TRUE_REFERENCE), or TMA hits 7.57 and STG caps at 6.11 (per V8). The "STG plateau = 6.11" claim from V8 contradicts the 01_hbm_bandwidth claim that v8+per-warp STG hits 7.30 with the right launch geometry.

**Inconsistency #4:** V8_HBM_WRITE_SOL claims "TMA bulk store 7.57 TB/s = 105% of read peak". This violates the read=write within 0.3% finding in 01_hbm_bandwidth (rule "no inherent read/write asymmetry on B300 HBM3E"). The "exceeds read" framing is a denominator-mismatch artifact (uses 7.2 effective for read, 8.0 nominal for write).

---

## D. V46 (TMA read 7.20 TB/s) vs older claims

| Source | Number | Compatible with V46? |
|---|---:|---|
| V46 (V41_V48) | 7.20 TB/s pipelined TMA | — |
| 01_hbm_bandwidth.md | 7344 GB/s TMA bulk read | 7.20 < 7.34 ; **V46 is NOT a new ceiling**, it is below the older single-shot TMA number quoted in 01. The "98.5%" framing is purely a 7.31-denominator artifact. |
| 01_hbm_bandwidth.md | 7365 GB/s LDG.E.128 read SoL | 7.20 < 7.365; same issue. |
| B300_TRUE_REFERENCE.md | 7.30 / 7.31 (NINJA) | 7.20 < 7.30. V46 is NOT the new master peak. |
| V33 (single-deep) | 6.72 TB/s TMA per-CTA | V46 confirms single-deep was leaving perf on the table; that part of the V46 claim is consistent. |

**Inconsistency #5 (BIG):** V41_V48_FINDINGS announces V46 as "**NEW BEST**" of 7.20 TB/s = 98.5%. But:
- 01_hbm_bandwidth already had **7.344 TB/s TMA** and **7.365 TB/s LDG** (HIGHER).
- 7.20 / 7672 = 93.85% (not 98.5%).
- 7.20 < 7.30 (the TRUE_REFERENCE master number).

V46 is a real improvement over V33's 6.72 TB/s but is **not** an architectural new SoL. The "98.5%" headline is an artifact of using 7.31 (an empirical peak) as the denominator instead of 7.672 (the spec).

---

## E. V42 prefetch claim vs older claims

`V41_V48_FINDINGS.md` V42: "TMA + prefetch.L2 = **27% SLOWER**. Rule: never combine prefetch.L2 with cp.async.bulk."

Audit:
- `01_hbm_bandwidth.md` does not mention prefetch.L2 at all.
- `B300_TRUE_REFERENCE.md` does not mention prefetch.L2 in the HBM context.
- V41_V48 itself notes "(V6 I3 1.58× speedup was for old cp.async — not bulk.)" — i.e., the contradiction is reconciled by saying old cp.async benefits but cp.async.bulk does not.

**No direct contradiction** within HBM scope. The "rule" is narrowly correct as stated. Worth promoting into 01_hbm_bandwidth.md as a "do not" caveat.

---

## F. V46 single-deep vs pipelined reconciliation

V41_V48 V46 supersedes V33's "single-deep is enough" implicit claim. But:
- 01_hbm_bandwidth's TMA bulk number (7.344 TB/s) was already higher than both V33 (6.72) and V46 (7.20). 01 does not specify pipeline depth for its TMA test.
- It is unclear whether 01's 7.344 was pipelined or whether it used a different launch geometry that achieved similar effect.

**Open question:** Is 01_hbm_bandwidth's 7.344 TMA result reproducible? Was it pipelined? The corpus does not say. V46's 7.20 is below it.

---

## G. cudaMemset claims

| Source | Number | Status |
|---|---:|---|
| 01_hbm_bandwidth (top) | 7.30 TB/s ncu (= true DRAM rate); wall-clock 7.47-7.52 overstates | Authoritative; explicitly retires the wall-clock 98% claim. |
| 01_hbm_bandwidth retirement table | "97-98% of HBM peak" — RETIRED | Consistent. |
| B300_TRUE_REFERENCE NINJA recipe | "beats cudaMemset by 5%" (7.57 vs 7.20) | Implies cudaMemset is ~7.20 wall-clock. Consistent with 01's 7.47. |

**No inconsistency** here; well-converged.

---

## H. Concurrent R+W

| Source | Aggregate | Notes |
|---|---:|---|
| 01_hbm_bandwidth (single-kernel, ncu) | 6.74 TB/s = 0.92× | Authoritative |
| 01_hbm_bandwidth A6 R:W sweep | 6.68 TB/s @ 50:50 = 87% | Refines |
| B300_TRUE_REFERENCE | 6.68 TB/s (de3b4d5) | Consistent |
| V32_V40 V35/V36 | 6.11 / 6.21 TB/s TMA stream copy | Lower because TMA-specific overhead |

**No major inconsistency**, just different recipes. Two-stream "10.4 TB/s" claim was already explicitly retracted in 01.

---

## I. Data-dependence

| Source | Finding |
|---|---|
| HBM_DATA_DEPENDENCE.md | "Inferred" <50W variation, kernel was poorly optimized (20.4 GB/s) — LOW confidence |
| L2_DRAM_DATA_PWR.md | <1% power variance across 11 patterns, well-optimized kernel — HIGH confidence |

**Inconsistency #6 (minor):** HBM_DATA_DEPENDENCE.md is superseded by L2_DRAM_DATA_PWR.md but is not marked as such. It also reports a 20.4 GB/s "throughput" which is broken-low and should not appear in any catalog without a "BROKEN TEST" tag.

---

## Summary table of contradictions

| # | What | Severity | Resolution |
|---|---|---|---|
| 1 | Three denominators for "% of peak" (7672 / 7.31 / 8.0) | HIGH | Standardize on 7672 GB/s (post-ECC). |
| 2 | Read peak quoted as 6.72, 7.20, 7.29, 7.30, 7.31, 7.344, 7.365 in different tables | MED | Pick canonical: 7.30-7.36 cluster (within 1%). |
| 3 | 7.57 TB/s attributed to NINJA STG (TRUE_REF) vs TMA (V8_HBM_WRITE_SOL) | HIGH | Re-verify which path produced 7.57. |
| 4 | V8 claims TMA write "exceeds read peak by 5%" violating "no R/W asymmetry" rule | HIGH | Denominator artifact; retire the framing. |
| 5 | V46 announced as "NEW BEST 98.5%" but is below 01's 7.344 TMA + 7.365 LDG | HIGH | Demote V46 from "new SoL" to "improvement over V33 single-deep". |
| 6 | HBM_DATA_DEPENDENCE.md superseded by L2_DRAM_DATA_PWR.md, not marked | LOW | Mark superseded. |
| 7 | CLAUDE.md says "8 TB/s spec"; canonical is 7672 post-ECC | LOW | Update CLAUDE.md. |
| 8 | V8_HBM_WRITE_SOL says STG ceiling = 6.11 TB/s; 01 says STG with v8+per-warp = 7.30 | HIGH | V8 used inferior recipe; clarify it's pattern-limited not architectural. |
