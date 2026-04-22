# DOUBT LOG (Wave-3c Synthesis)

**Date**: 2026-04-22.
**Inputs**: 6 wave-3b adversarial doubt reports + 4 wave-3a topical
corrections + the 3 wave-1+2 synthesis files (`MASTER_INDEX.md`,
`HEADLINE_CORRECTIONS.md`, `B300_TRUE_REFERENCE_v2_DRAFT.md`).
**Purpose**: per-claim verdict on every wave-1+2 headline; build a new
top-10 list reflecting what survived doubt and what didn't.

Originals NOT modified. The earlier synthesis files remain the wave-1+2
record; this file is the doubt-aware overlay.

---

## 1. Per-claim verdicts on the 10 wave-1+2 HEADLINE_CORRECTIONS.md headlines

| # | Headline (wave-1+2 text) | Verdict | Doubt source | Notes |
|---|---|---|---|---|
| 1 | NVLink generation is **NVLink-5**, not v7 | **CONFIRMED** | SYNTHESIS_DOUBT (web-confirmed) | NVLink-5 = Blackwell, 1.8 TB/s bidi = 2× NVLink-4 |
| 2 | NVLink spec denominator is **900 GB/s/dir** | **CONFIRMED** | SYNTHESIS_DOUBT, CROSS_AGENT #7 | NVLink + PCIe agents agree |
| 3 | **HBM read SoL = ~7.30 TB/s** at 95% of 7672; V46's 98.5% was a denominator artifact | **REFRAMED** | V46_DOUBT, SYNTHESIS_DOUBT H2/H3, CROSS_AGENT #1+#10 | The 7.20 TB/s measurement is **honest**; V46 itself runs a 4-pt sweep avg-of-5. The "demotion" is purely the denominator change (7.20/7672=93.8% vs 7.20/7.31=98.5%). The architectural lesson "TMA reads need 8-deep pipelining" remains valid. SYNTHESIS_DOUBT also flags "BELOW V44/V45" as the wrong comparators (V44/V45 are SMEM, not HBM). |
| 4 | MUFU 47.8 G is 1-CHAIN latency-bound, NOT pipe-saturated; saturated = 4.74 G | **REFINED (wording)** | SYNTHESIS_DOUBT H1 | M14/M16 reported a real measurement of a real regime; the "10× mislabel" framing reads as if the original number was wrong. Better wording: "M14/M16 row is 1-chain latency-bound, not pipe-saturated; saturated peak is 10× lower at 4.74 G". |
| 5 | **IADD3 lives on the FMA pipe** (V40 / d1d09c5), not on a separate ALU pipe | **CONFIRMED** | CROSS_AGENT #6 | All 4 agents that mention IADD3 agree. Clean retraction of older "ALU pipe" framing. |
| 6 | **Same-warp dual-issue = 55%; warp-spec = 74%** (V49/V50) | **DOWNGRADED to LOW** | DUAL_ISSUE_DOUBT (entire report) | FFMA solo baseline is itself only 67% of FFMA peak (under-occupied at 2 warps/SMSP). The "55% / 74% of separate-pipe theoretical" ratio divides by a baseline that's already broken. V49/V50 collected NO ncu metrics. M8 has counter-evidence: MUFU+FFMA ≈ 100%, HMMA+LDS 73-96%. Architectural claim "dispatch is 4-wide per SM regardless of pipe" is not supported. **Rerun with warps/SMSP ∈ {1,2,4,8} sweep + ncu before promoting to canonical.** |
| 7 | **All V8/V10 DSMEM TB/s peaks were DCE artifacts**; real read 40 GB/s/cluster, write 560 GB/s/cluster | **CONFIRMED with caveats** | DSMEM_DOUBT | V8 DCE retraction is HIGH (SASS-verified). 40 GB/s read is **chain-bound**, not absolute (a non-chained ILP test might reach 60-80 GB/s). 560 GB/s write is **issue rate**, not completion (no fence between stores and clock64 end); real delivery rate may be lower. "NO shared bus" claim is **under-issued by 30×** in V17 — test cannot rule out a shared bus. The 7.5× local/DSMEM **latency** ratio is HIGH. Also: SYNTHESIS_DOUBT H4 notes v1's 3.06 TB/s "aggregate" was actually per-cluster 41 × 74 ≈ 3 TB/s **chip aggregate** — internally consistent with v2's per-cluster 40 GB/s; the "mixed measurements" framing was wrong, the disagreement was scope (per-cluster vs chip-aggregate). |
| 8 | **A vs B operand power has 3 different "correct" answers** depending on test geometry (cuBLAS A>B 3:1; pure tcgen05 B>>A 15-30×; K=96 single-kernel B>A 2.6×) | **REFINED — preserve all 3** | NVFP4_DOUBT | The 3 source readings are all real and traced. Wave-2's "TMA multicast halves B's memory cost" mechanism IS supported by an explicit ncu table (multicast 0% BF16 vs 78% NVF4). BUT the underlying source itself (`NVFP4_PURE_TCGEN05_RESULTS.md` Correction §) lists 4 plausible mechanisms and walks back the single-mechanism story. Wave-2 over-resolved. **Surface all 3 readings; don't pick one.** Notably, the K=96 single-kernel 2.6× matches BF16 cuBLAS's 2.0-2.9× — suggests pure-tcgen05's 15-30× is the **artifact** (over-isolation), not the truth. |
| 9 | **3-source FFMA caps at ~50 TFLOPS (65%)** due to RF port pressure | **CONFIRMED** | A_TO_D_RIGOR_AUDIT, V46_DOUBT (no challenge) | Cleanly cross-validated: A4 + D6 + V10_FMA_SOURCE_COUNT (75.2 vs 51.3 = ratio 0.683 ≈ 2/3, exactly 2-RF-port prediction). |
| 10 | NVFP4 cuBLAS+cudaGraph BPG=16 = **11.42 PF (76.2%)** supersedes 10.8 PF | **REFINED — single-shape** | SYNTHESIS_DOUBT M2, NVFP4_DOUBT #2, CROSS_AGENT #4 | Number is real (`NVFP4_CUDAGRAPH.md`, 200-iter sustained). But it is **one shape** (8K² K=38400) at one BPG value — not a sweep. Treat as **upper-bound at this shape**, not sustained ceiling. Tensor agent's stale 10.8 PF should defer to NVFP4 agent's 11.42, but both are correct in their own context. |

---

## 2. Top-10: HEADLINES THAT SURVIVED DOUBT (HIGH confidence)

| # | Claim | Source |
|---|---|---|
| 1 | NVLink-5 (NOT v7); spec 900 GB/s/dir | NVLink + PCIe agents, web-confirmed |
| 2 | IADD3 lives on FMA pipe (V40, d1d09c5) | All 4 agents agree |
| 3 | 3-source FFMA = ~50 TFLOPS (~65% of 2-source) | A4 + D6 + V10_FMA_SOURCE_COUNT |
| 4 | V8/V10 DSMEM TB/s = DCE artifacts (real read 40 GB/s/cluster *chain-bound*) | DSMEM_DOUBT confirms |
| 5 | DSMEM is **7.5× slower than local SMEM** (latency) | V12/V15/V16 cross-test consistent |
| 6 | `pipe_tensor.cycles_active` does NOT measure tcgen05 | TENSOR log §B |
| 7 | L2 = **126 MB** (not 50/96/192/256); "96 MB" cosmetic error in 4 files | STRAYS §7 |
| 8 | TMEM = ~60 TB/s read (not 295/830 — those were DCE) | 06_tensor + V4 D7 |
| 9 | Random data is up to 43% slower than zero data for FP8 cuBLAS under power cap | TRUE_REF v1 row 68 |
| 10 | Power d=16 random popcount = 240-554 W swing (HBM_DATA_DEPENDENCE.md's <50 W is SUPERSEDED) | STRAYS §2, POPCOUNT 4-file family |

## 3. Top-10: HEADLINES THAT NEED REVISION (MED–LOW confidence)

| # | Claim | Issue | Recommendation |
|---|---|---|---|
| 1 | "V46 = NEW HBM read SoL at 98.5%" | Denominator artifact (used 7.31 empirical instead of 7672 spec) | Reframe as "V46 is honest 7.20 TB/s measurement; 7.20/7672 = 93.8%; below 7.34 (TMA bulk), 7.365 (LDG), 7.30 (NINJA). Architectural lesson on TMA pipelining stays." |
| 2 | "Same-warp dual-issue = 55%; warp-spec = 74% (FFMA+LOP3)" | Baseline is 67%-of-peak (under-occupancy); no ncu metrics; M8 counter-evidence | **DOWNGRADE to LOW**. Re-run with warps/SMSP ∈ {1,2,4,8} sweep + ncu before promoting. |
| 3 | "DSMEM 40 GB/s read aggregate" | Chain-bound (V21 dependent-chain test); non-chained ILP could reach 60-80 GB/s | Annotate as "chain-bound ceiling, not absolute asymptote" |
| 4 | "DSMEM 560 GB/s write aggregate" | Issue rate, not completion (no fence between stores and clock64 end) | Annotate as "issue rate ceiling; real delivery rate unverified" |
| 5 | "NO shared bus" (DSMEM) | V17 ring test was **30× under-issued** (1 thr/CTA, single-issue chained) | Demote to "consistent with point-to-point per architecture; not proven by V17" |
| 6 | "TMA multicast halves B's memory cost" (NVFP4 A:B story) | Underlying source itself lists 4 plausible mechanisms; wave-2 picked one | Preserve all 3 A:B readings (cuBLAS, pure-tcgen05, K=96 single-kernel) |
| 7 | "NVFP4 cuBLAS 11.42 PF supersedes 10.8 PF" | Single-shape, single-BPG (no sweep) | Frame as "upper-bound at 8K² K=38400 BPG=16; sustained ceiling needs sweep" |
| 8 | "MUFU mislabeled by ~10×" | Number was real measurement of real regime, not "wrong number" | Soften to "M14/M16 row is 1-chain latency-bound, not pipe-saturated" |
| 9 | "NINJA STG hit 7.57 TB/s" (HBM write SoL) | Provenance contested with V8 TMA bulk store path | Mark UNRESOLVED until both kernels re-run back-to-back with ncu `dram__bytes` |
| 10 | "Persistent kernel 2.03 µs supersedes v1's 4 µs because v1 used release variant" | Hypothesis about why v1 was higher; no SASS evidence cited | Demote to "hypothesis: v1 likely used release; not verified" |

## 4. New headlines from wave-3 (post-doubt findings)

| # | Finding | Source |
|---|---|---|
| 1 | **HBM denominator is the #1 cross-agent contradiction**: 7672 / 7.31 / 7.2 / 8.0 TB/s appear across 3+ files; ALL "% of peak" cross-doc numbers are not directly comparable | CROSS_AGENT #1+#2 |
| 2 | **System fence cost is UNRESOLVED 1.74× spread**: 1750 cy (08) vs 2870 cy (DSMEM) vs 3042 cy (V9). TRUE_REFERENCE v1 picked 861 ns (=1750 cy) WITHOUT justification | CROSS_AGENT #2+#10, SYNTHESIS_DOUBT L3 |
| 3 | **CURIOSITY_LIST V2 has 88% hallucinated hashes** (22/25); V4-V8 are 100% git-verified | CURIOSITY_LISTS_AUDIT |
| 4 | **TCGEN05_PERF_WATTS single-trial table is contaminated** (5 leftover QuickRunCUDA processes); use TCGEN05_PERFW_CLEAN_2TRIAL. NVFP4 K=96 = 12.54 TF/W random / 15.74 best (NOT 13.72) | TCGEN05_POWER_CONSOLIDATED §2 + R1 |
| 5 | **HBM_DATA_DEPENDENCE.md is SUPERSEDED**: real DRAM data-dep swing is 240-554 W, NOT <50 W (that was constant-pattern-only) | STRAYS §2 |
| 6 | **A6's 4-tier pipe ladder (FMA/INT-bit/permute/compare) supersedes "unified ALU/FMA cluster"** model | A_TO_D_RIGOR_AUDIT #5 |
| 7 | **L2 BW must be labelled** with one of {kernel-effective ≈24, wire-lts ≈13, L1-amplified ≈30} TB/s; bare numbers float | CROSS_AGENT #4 |
| 8 | **Atomics SMEM aggregate is 2.27 T (not 4.2 T)**; CLAUDE memory's 4.2 T is unsourced | CROSS_AGENT #12 |
| 9 | **L2 atomic units count "~32" should be MEDIUM not HIGH** (derived ceiling, not direct measurement) | STRAYS §8 |
| 10 | **Wave-2 took credit for upstream retractions** (BF16 1543 / FP8 7500-8200 / BF16 90.5%): all already self-retracted by their original docs | SYNTHESIS_DOUBT H5 |

---

## 5. Confidence ladder summary (by topic)

| Topic | Confidence | Reason |
|---|---|---|
| HBM read peak (7.30 TB/s @ 95% of 7672) | HIGH | Multi-source NINJA/TMA/LDG agree; denominator stated |
| HBM write SoL (7.57 TB/s) | MED — contested provenance | NINJA STG vs V8 TMA bulk |
| L2 BW (with metric tag) | HIGH | Cache agent disambiguates 3 metrics |
| TMEM read (~60 TB/s) | HIGH | DCE-corrected |
| SHMEM peak (38.4 TB/s) | HIGH | 99.8% of spec |
| DSMEM latency (7.5× slower than local SMEM) | HIGH | V12/V15/V16 cross-test |
| DSMEM read aggregate 40 GB/s/cluster | MED — chain-bound | V21 dependent-chain only |
| DSMEM write aggregate 560 GB/s/cluster | LOW-MED — issue rate | No fence in V21 |
| DSMEM "NO shared bus" | LOW | V17 30× under-issued |
| FP32 FFMA peak (75.9 TFLOPS, 2-source) | HIGH | Multi-recipe |
| FP32 FFMA realistic (50 TFLOPS, 3-source) | HIGH | A4/D6/V10 agree |
| Same-warp dual-issue 55% / warp-spec 74% | **LOW** | Under-occupied baseline; no ncu |
| IADD3 on FMA pipe | HIGH | All 4 agents agree |
| MUFU saturated 4.74 G/chip (NOT 47.8) | HIGH | V41 pipe rate; framing only |
| NVFP4 11.42 PF cuBLAS+graph | MED — single shape | No BPG sweep |
| NVFP4 K=96 ULTRA 10.91 PF (98.5% per CTA) | HIGH | TF/W cross-validated |
| NVFP4 A:B 3-way (cuBLAS / pure / K96) | MED — preserve all 3 | Wave-2 over-resolved single mechanism |
| Tcgen05 perf/W 2-trial table | HIGH | PERFW_CLEAN_2TRIAL supersedes contaminated single-trial |
| Power d=16 popcount bell curve | HIGH | 4-file family agrees |
| HBM_DATA_DEPENDENCE.md "<50 W" | RETRACTED | Superseded by POPCOUNT family |
| NVLink-5 spec 900 GB/s/dir | HIGH | Web-confirmed |
| `__threadfence_system` cost | UNRESOLVED 1.74× spread | 1750/2870/3042 cy |
| `__threadfence` (GPU) 281 cy | MED — 4-way 24% spread | Sync agent preserves; DSMEM picks 320 silently |
| Persistent kernel 2.03 µs | MED — mechanism unverified | Number real; "v1 used release" is hypothesis |
| L2 atomic units ~32 | MED (downgraded from MED-HIGH) | Derived ceiling, not direct |
| SMEM atomic aggregate 2.27 T | HIGH | Atomics + SHMEM agree (CLAUDE memory's 4.2 T unsourced) |
| `pipe_tensor.cycles_active` doesn't measure tcgen05 | HIGH | TENSOR log §B |

---

## 6. Net assessment of wave-1+2 synthesis

The wave-1+2 synthesis is **broadly faithful** but:

- **Took credit for some upstream retractions** (1543/7500-8200/90.5%) that the originals already made.
- **Prematurely picked one denominator** in the HBM debate (7672 is right but 7.31 is also defensible for "% pure-direction").
- **Conflated measurement scopes** in the DSMEM 3.06 TB/s supersession (v1 was per-cluster × ~74 = chip aggregate, internally consistent with v2's per-cluster 40 GB/s).
- **Flattened 30% UNRESOLVED gaps** into headline single numbers (IADD3 0.5 vs 0.66; PRMT 0.36 vs 0.5).
- **Promoted V49/V50 dual-issue 55%/74% to HIGH** without acknowledging the under-occupied baseline or M8's counter-evidence.

No CRIT-level fabrication detected.
