# HEADLINE CORRECTIONS v2 — 1-page TL;DR (doubt-aware)

**Supersedes**: `HEADLINE_CORRECTIONS.md` (wave-1+2). The wave-1+2 file
remains the historical record; this v2 incorporates the wave-3a topical
corrections and the wave-3b adversarial doubt findings.

For per-claim verdicts see `DOUBT_LOG.md`. For the full file index see
`MASTER_INDEX_v2.md`.

---

## If you remember nothing else…

| # | What changed | Old claim | New claim (doubt-aware) | Confidence | Source |
|---|---|---|---|---|---|
| 1 | **NVLink generation** | "NVLink v7" (CLAUDE.md memory) | **NVLink-5** (5th gen, Blackwell, 900 GB/s/dir spec) | HIGH | NVLink + PCIe agents, web-confirmed |
| 2 | **HBM read peak (V46 reframed)** | V46: "98.5% NEW BEST at 7.20 TB/s" | V46's **7.20 TB/s is an honest measurement**. Re-normalized to 7672 GB/s post-ECC spec = **93.8%** (V46 used 7.31 empirical denominator). Existing read peaks are 7.30 (NINJA), 7.34 (TMA bulk), 7.365 (LDG). V46's architectural lesson "TMA reads need 8-deep pipelining" remains valid; the % framing was the issue, not the number. | HIGH (number) / REFRAMED (%) | V46_DOUBT, HBM log #5 |
| 3 | **HBM denominator** | Three different denominators across docs (7672 / 7.31 / 7.2 / 8.0) | **Recommend standardizing on 7672 GB/s post-ECC spec** for all "% of peak" claims; 7.31 (empirical pure-direction) is also defensible for "% achievable in isolation" — preserve both framings explicitly. ALL cross-doc % numbers are not directly comparable today. | RULE | CROSS_AGENT #1, SYNTHESIS_DOUBT H3 |
| 4 | **MUFU saturated peak** | M14/M16: "XU peak 47.8 G MUFU/s @ 99.5%" | 47.8 G is **1-chain LATENCY-bound rsqrt** (a real measurement of a real regime, not a "wrong number"). **Saturated MUFU = 4.74 G/chip**, EX2 outlier 9.22 G. Better wording: "row is 1-chain latency-bound, not pipe-saturated". | HIGH | MATH log #3, SYNTHESIS_DOUBT H1 |
| 5 | **IADD3 pipe placement** | V9: "separate ALU pipe at 38 TOPS" | **IADD3 lives on the FMA pipe** (V40, d1d09c5, 25-26 Glane/s = 67% of FMA SoL). LOP3/PRMT/IMUL are the real INT-bit/permute pipes (half rate). Caveat: A6 (0.5/SMSP/cy) vs V40 (0.66) gap is UNRESOLVED 30%. | HIGH (placement) / MED (rate) | INT log #A, COMPUTE log #E, CROSS_AGENT #6 |
| 6 | **Dual-issue cap** | Catalog: "FFMA + IADD3 free / 100% / 114 TOPS combined" | V49 same-warp = 55%, V50 warp-spec = 74%. **DOWNGRADED to LOW confidence pending warps-sweep.** Baseline (FFMA solo at 26066 Glane/s) is itself only 67% of FFMA peak — under-occupied at 2 warps/SMSP. NO ncu metrics collected. M8 has counter-evidence: MUFU+FFMA ≈ 100%, HMMA+LDS 73-96%. The architectural claim "dispatch is 4-wide per SM regardless of pipe" is **not supported** — a same-warp 1 inst/cy SIMT limit explains 55% trivially. | **LOW** | DUAL_ISSUE_DOUBT (entire) |
| 7 | **DSMEM aggregate BW** | V8/V10: "37 TB/s read / 11.8 TB/s write / writes 4-5× SLOWER than reads" | **All V8/V10 TB/s = DCE artifacts** (HIGH). Real per-cluster: read ≈ 40 GB/s **chain-bound** (non-chained ILP could be 60-80); write ≈ 560 GB/s **issue rate, not completion** (no fence in V21). Local/DSMEM ratio = 7.5× (latency, HIGH). v1's 3.06 TB/s "aggregate" was per-cluster × ~74 = chip aggregate — internally consistent with v2's per-cluster 40 GB/s; it was a scope difference, NOT mixed measurements. | MED-HIGH (DCE) / MED (BW caveats) | DSMEM_DOUBT, DSMEM log A/B/C, SYNTHESIS_DOUBT H4 |
| 7b | **DSMEM "NO shared bus"** | V17: "no contention, point-to-point" | V17 ring was **30× under-issued** (1 thread/CTA, single-issue chained). Test cannot rule out a shared bus. Likely point-to-point per architecture, but V17 doesn't prove it. | LOW | DSMEM_DOUBT |
| 8 | **A vs B operand power** | CLAUDE memory: "A:B impact ~1:3, B dominant" / "A is FREE" | **3 different ratios depending on test geometry — all real, preserve all 3**: cuBLAS A>B 3:1 (TMA multicasts B); pure tcgen05 B>>A 15-30× (over-isolation artifact?); K=96 single-kernel B>A 2.6× (matches BF16 cuBLAS). Wave-2's "TMA multicast resolves it all" is **over-resolved** — the underlying source itself lists 4 plausible mechanisms. | MED — preserve all 3 | NVFP4_DOUBT |
| 9 | **3-source FFMA cap** | Headlines all use 2-source FFMA (75 TFLOPS = 97% peak) | **3-source GEMM caps at ~50 TFLOPS = 65% of peak** due to 2 RF read ports + reuse-cache as effective 3rd port. Cleanly cross-validated: A4 + D6 + V10_FMA_SOURCE_COUNT (75.2 vs 51.3 = ratio 0.683 ≈ 2/3). | HIGH | COMPUTE log #G, V8/V10 misc log §C |
| 10 | **NVFP4 cuBLAS ceiling** | Memory: "10.8 PF (72%)" | **11.07 PF plain Lt → 11.42 PF with cudaGraph BPG=16 (76.2%)** at K=38400. Caveat: **single shape, single BPG** — no sweep. Treat as upper-bound at this shape, not sustained ceiling. The 10.8 PF was a smaller-K shape; both correct in their context. | MED — single shape | NVFP4 log "MAJOR cuBLAS ceiling", SYNTHESIS_DOUBT M2 |

---

## UNRESOLVED — propagated as-is from doubt swarm

| Topic | Spread | Recommendation |
|---|---|---|
| `__threadfence_system` cost | 1750 (08) vs 2870 (DSMEM) vs 3042 (V9) cy = 1.74× | **DO NOT pick one.** TRUE_REFERENCE v1 picked 861 ns (=1750 cy) without justification. Re-run `bench_fence_cost.cu` at LOCKED 1920 MHz. |
| `__threadfence` (GPU) cost | 258 / 281 / 292 / 320 cy = 24% spread | Sync agent's "281 ± 25" with explicit 4-way spread is correct; DSMEM agent's silent 320 cy should adopt the spread. |
| HBM write SoL provenance | 7.57 TB/s attributed to NINJA STG (e75c7e1) AND V8 TMA bulk store (28211ce) | Re-run both kernels back-to-back with ncu `dram__bytes`. |
| IADD3 rate | A6: 0.5/SMSP/cy; V40: 0.66/SMSP/cy = 30% gap | A6-style sweep at 4+ warps/SMSP for IADD3 + PRMT side-by-side. |
| L2 atomic units | TRUE_REF: 32; ATOMIC_REVERIFY_DEEP: could be higher | Stride sweep with ncu lts__t_bytes per partition. |
| Cluster=2 21% slower than ≥3 | Topology hypothesis (single-GPC vs multi-GPC) unverified | `gpc__cycles_active.per_pgpc_id` ncu pass. |

---

## RETRACTIONS (already done upstream — do NOT take credit)

The wave-1+2 synthesis presented these as new findings, but they were
**ALREADY RETRACTED in-place** in their original docs:

- **BF16 1543 TFLOPS single-chain** RETRACTED (real ~570) — META log B1 says already-retracted upstream
- **FP8 mma.sync 7500-8200 TFLOPS** RETRACTED (SASS showed HMMA.16816 not 16832; real ~3760) — META log B2
- **BF16 90.5% of 2500 spec** RETRACTED (mislabeled — 23% of tcgen05 spec or 93.7% of legacy 616) — META log B3
- **K-uniform-per-N 28% NVFP4 power saving** RETRACTED (background-process contamination; real 1-3%) — already in NVFP4_SIGN_K64_K96.md MAJOR CORRECTION header

## RETRACTIONS (genuinely new, from wave-3)

- **TCGEN05_PERF_WATTS single-trial table CONTAMINATED** (5 leftover QuickRunCUDA processes); use TCGEN05_PERFW_CLEAN_2TRIAL. NVFP4 K=96 = 12.54 TF/W random / 15.74 best (NOT 13.72)
- **HBM_DATA_DEPENDENCE.md "<50 W" SUPERSEDED** by POPCOUNT_3TIER family — real swing is 240-554 W (5-7× larger)
- **CURIOSITY_LIST V2: 22/25 hashes hallucinated** (88%); V4-V8 are 100% git-verified
- **L2 = 96 MB cosmetic error in 4 files** (real = 126 MB); no measurement affected
- **L2 atomic units ~32** should be MEDIUM not MEDIUM-HIGH (derived ceiling, not direct measurement)
- **SMEM atomic aggregate = 2.27 T** (CLAUDE memory's 4.2 T is unsourced)
- **A6 "unified ALU/FMA cluster" model** SUPERSEDED by V40 4-tier ladder (FMA 67% / INT-bit 48% / permute 36% / compare 22%)

---

## Methodology rules learned (carry forward)

1. **Always state HBM denominator**: 7672 GB/s post-ECC (spec) or 7.31 (empirical pure-direction). Never bare %.
2. **Always label L2 BW** with one of: kernel-effective (~24), wire-lts (~13), L1-amplified (~30) TB/s.
3. **Always sweep warps/SMSP** for dual-issue / pipe-overlap tests (V49/V50 lesson). 2 warps/SMSP is under-occupied.
4. **Always cite ncu metrics** for "% of pipe peak" claims. clock64-only is single-method (insufficient per CLAUDE.md §4).
5. **Always git-verify TODO-list hashes** (V2 lesson — 88% hallucination rate when cited from memory).
6. **Always specify chain vs non-chain** for memory BW claims (DSMEM 40 GB/s lesson).
7. **Always specify issue-rate vs completion** for write BW (DSMEM 560 GB/s lesson — fence required).
8. **Don't pick a single number from a 1.5×+ spread** (system fence lesson — flag as UNRESOLVED).
9. **Don't take credit for upstream retractions** — check if the source already retracted before re-headlining.
10. **Don't promote one mechanism when sources list multiple** (NVFP4 A:B lesson).
