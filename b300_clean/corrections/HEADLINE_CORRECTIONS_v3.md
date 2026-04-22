# HEADLINE CORRECTIONS v3 — 1-page TL;DR (wave-4 doubt-aware)

**Supersedes:** `HEADLINE_CORRECTIONS_v2.md` (wave-3c). v2 remains the
historical record; this v3 incorporates wave-4 reversals, the HBM
denominator resolution, and the V51 forensics.

For per-claim verdicts see `DOUBT_LOG.md` + `WAVE4_CHANGES.md`.
For the priority retest queue see `UNRESOLVED_PROMOTED.md`.

---

## If you remember nothing else…

| # | What changed | Old claim | New claim (wave-4) | Confidence | Source |
|---|---|---|---|---|---|
| 1 | **NVLink generation** | "NVLink v7" (CLAUDE.md memory) | **NVLink-5** (5th gen, Blackwell, 900 GB/s/dir spec, 1800 GB/s bidi) | HIGH | NVLink + PCIe agents, web-confirmed |
| 2 | **HBM stack count (NEW)** | CLAUDE.md & `01_hbm_bandwidth.md` line 136: "12 HBM3E stacks" | **B300 has 8 HBM3E stacks of 12-Hi (12 = die-stack height, NOT stack count)**. Real bus = 8 × 1024 = 8192-bit. | HIGH | `HBM_DENOMINATOR_RESOLUTION.md` (NVIDIA Developer Blog + 5 teardowns) |
| 3 | **HBM denominator (NEW: 7672 retired)** | TRUE_REFERENCE: "7672 GB/s spec" (also in 01_hbm, HBM_INCONSISTENCY_LOG) | **7672 is an arithmetic ghost** (intermediate 1998 MHz × 2 mismatch). Use **7680 GB/s post-ECC** (8192 × 15/16) as canonical denominator. Use **8192 GB/s raw** if comparing against bus capability. | HIGH | `HBM_DENOMINATOR_RESOLUTION.md` §3 |
| 4 | **HBM read peak (V46 reframed)** | V46: "98.5% NEW BEST at 7.20 TB/s" | V46's 7.20 TB/s is honest. Re-normalized: **93.75% post-ECC (7680)** / 87.9% raw (8192). Apples-to-apples among READ recipes: LDG 7.365 > TMA bulk 7.344 > V8 NINJA 7.30 > A6 R-only 7.31 > V46 7.20. Architectural lesson "TMA reads benefit from 8-deep pipelining" stays valid. | HIGH (number) / REFRAMED (%) | `META_DOUBT_REPORT.md` §2; V46_DOUBT |
| 5 | **MUFU saturated peak** | M14/M16: "XU peak 47.8 G MUFU/s @ 99.5%" | 47.8 G is **1-chain LATENCY-bound rsqrt**. **Saturated MUFU = 4.74 G/chip**, EX2 outlier 9.22 G. Better wording: "1-chain latency-bound, not pipe-saturated". | HIGH | MATH log #3 |
| 6 | **IADD3 pipe placement** | V9: "separate ALU pipe at 38 TOPS" | **IADD3 lives on the FMA pipe** (V40, d1d09c5, 25-26 Glane/s = 67% of FMA SoL). Caveat: A6 (0.50/SMSP/cy) vs V40 (0.66) gap is UNRESOLVED 30%. | HIGH (placement) / MED (rate) | INT log #A; CROSS_AGENT #6 |
| 7 | **Dual-issue cap (REVERSED from v2)** | v2: "**LOW** — V49 baseline only 67% of FFMA peak; under-occupied" | **MED** (was LOW). `META_DOUBT_REPORT.md` §1: V8_FFMA_PEAK_VERIFIED hits **97.64%** at the IDENTICAL 2 warps/SMSP V49 uses. Under-occupancy mechanism FALSIFIED. The 55%/74% measurements are reproducible; only the "dispatch is 4-wide per SM" interpretation needs ncu confirmation (V52 sketch settles it). | **MED** (was LOW) | `META_DOUBT_REPORT.md` §1 |
| 8 | **DSMEM aggregate BW** | V8/V10: "37 TB/s read / 11.8 TB/s write" | **All V8/V10 TB/s = DCE artifacts**. Real per-cluster: read **40 GB/s chain-bound** (non-chained ILP could be 60-80 — V53 settles); write **560 GB/s issue-rate-only** (no fence in V21 — V53 settles). Local/DSMEM latency ratio = 7.5× (HIGH). | MED-HIGH (DCE) / MED (BW caveats) | DSMEM_DOUBT (source-verified) |
| 8b | **DSMEM "NO shared bus"** | V17: "no contention, point-to-point" | V17 ring was **30× under-issued** (1 thr/CTA, single-issue chained). Likely point-to-point per architecture but V17 doesn't prove it. | LOW | DSMEM_DOUBT |
| 9 | **A vs B operand power** | CLAUDE memory: "A:B impact ~1:3, B dominant" | **3 different ratios depending on test geometry — preserve all 3**: cuBLAS A>B 3:1; pure tcgen05 B>>A 15-30× (over-isolation artifact?); K=96 single-kernel B>A 2.6× (matches BF16 cuBLAS). Wave-2's "TMA multicast resolves it all" is **over-resolved** — V56 sketch discriminates 4 candidate mechanisms. | MED — preserve all 3 | NVFP4_DOUBT |
| 10 | **3-source FFMA cap** | Headlines all use 2-source FFMA (75 TFLOPS) | **3-source GEMM caps at ~50 TFLOPS = 65% of peak** due to 2 RF read ports + reuse-cache. Cleanly cross-validated: A4 + D6 + V10_FMA_SOURCE_COUNT (75.2 vs 51.3 = 0.683 ≈ 2/3). | HIGH | COMPUTE log #G |
| 11 | **NVFP4 cuBLAS ceiling** | Memory: "10.8 PF (72%)" | **11.07 PF plain Lt → 11.42 PF with cudaGraph BPG=16 (76.2%)** at K=38400. Caveat: **single shape** — no BPG sweep. | MED — single shape | NVFP4 log |
| 12 | **V51 multistream HBM (NEW)** | v2: not mentioned | `tests/standalone/v51_multistream_hbm.cu` passes `(const float*)d_src` (host stack address of pointer-array) instead of `d_src[s]` — **per-stream-buffer fix never wired through; output is UB**. **REMOVE the file.** Question is architecturally trivial anyway (one HBM bus). | n/a — file deleted | `V51_INVESTIGATION.md` |

---

## UNRESOLVED — propagated as-is, now with retest sketches

| Topic | Spread | Sketch that settles it |
|---|---|---|
| Dual-issue 55%/74% interpretation | LOW→MED reversed; mechanism unconfirmed | **V52** (`v52_dual_issue_warp_sweep.cu`) — warps/SMSP ∈ {1,2,4,8} sweep + ncu |
| `__threadfence_system` cost | 1750 / 2870 / 3042 cy = 1.74× | **V54** (`v54_membar_isolation.cu`) — N-issue scaling at locked 1920 MHz |
| HBM write SoL provenance | 7.57 TB/s NINJA STG (e75c7e1) vs V8 TMA bulk (28211ce) | **V55** (`v55_hbm_floor_BEST.cu`) — back-to-back ncu `dram__bytes` |
| DSMEM 40 GB/s read | chain-bound | **V53** (`v53_dsmem_fenced_retest.cu`) read mode |
| DSMEM 560 GB/s write | issue-rate-only | **V53** write mode + `fence.sc.cluster` |
| NVFP4 A:B asymmetry mechanism | 4 candidates | **V56** (`v56_nvfp4_AB_mechanism.cu`) — 4-mode discriminator |
| `__threadfence` (GPU) cost | 24% spread (258/281/292/320 cy) | bench_fence_cost variant |
| IADD3 rate 0.50 vs 0.66 | 30% gap | A6-style 4+ warps/SMSP sweep |
| L2 atomic units count | TRUE_REF: 32; could be higher | stride sweep with ncu lts__t_bytes per partition |
| Cluster=2 21% slower than ≥3 | unverified topology hypothesis | ncu `gpc__cycles_active.per_pgpc_id` |

---

## RETRACTIONS still standing from v2

Already-upstream (do NOT take credit): BF16 1543 TFLOPS, FP8 mma.sync
7500-8200, BF16 90.5%, K-uniform 28% NVFP4 saving.

Genuinely-new wave-3: TCGEN05_PERF_WATTS contaminated, HBM_DATA_DEPENDENCE
"<50 W" superseded, CURIOSITY_LIST V2 88% hash hallucination, L2 = 126 MB
not 96, L2 atomic units MED not MED-HIGH, SMEM atomic 2.27 T not 4.2 T,
A6 unified-cluster model superseded by V40 4-tier ladder.

**New from wave-4:** "12 HBM3E stacks" (CLAUDE.md, 01_hbm) is wrong;
real = 8 stacks × 12-Hi. "7672 GB/s" denominator is an arithmetic ghost.

---

## Methodology rules (carry forward — adds rule 11)

1. State HBM denominator (now standardized: **7680 post-ECC** is canonical).
2. Label L2 BW: kernel-effective (~24) / wire-lts (~13) / L1-amplified (~30) TB/s.
3. Sweep warps/SMSP for dual-issue / pipe-overlap tests.
4. Cite ncu metrics for "% of pipe peak" claims.
5. Git-verify TODO-list hashes (V2 88% hallucination).
6. Specify chain vs non-chain for memory BW.
7. Specify issue-rate vs completion for write BW.
8. Don't pick a single number from a 1.5×+ spread.
9. Don't take credit for upstream retractions.
10. Don't promote one mechanism when sources list multiple.
11. **NEW: Before re-measuring any HBM number, standardize on 7680 GB/s post-ECC as the denominator** (matches what ncu `dram__bytes_*` actually counts). 7.31 TB/s is empirical, NEVER call it "spec" or "theoretical".
