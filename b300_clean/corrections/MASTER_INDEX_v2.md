# MASTER INDEX v2 — `b300_clean/corrections/`

**Date**: 2026-04-22.
**Supersedes**: `MASTER_INDEX.md` (wave-1+2). The wave-1+2 file is the
historical record; this v2 incorporates wave-3a topical corrections and
wave-3b adversarial doubt findings.

For the doubt-aware TL;DR see `HEADLINE_CORRECTIONS_v2.md`.
For per-claim verdicts see `DOUBT_LOG.md`.

---

## 1. Document tree (60+ files)

### Wave-1+2 outputs (46 files)

| Type | Files |
|---|---|
| `*_CORRECTED.md` (17 categories + 5 special) | `01_hbm_bandwidth`, `02_shmem`, `03_caches`, `04_fp32_peak`, `05_fp_precision_nontensor`, `06_tensor_cores`, `07_atomics`, `08_sync_primitives`, `09_memory_apis`, `10_launch_overhead`, `11_block_scheduling`, `12_nvlink_p2p`, `13_pcie_system`, `14_math_intrinsics`, `15_integer_bit_ops`, `16_power_clock`, `17_nvrtc_module`, `DSMEM_CORRECTED`, `META_DOCS_CORRECTED`, `M_SYNTHESIS_CORRECTIONS`, `V8_V10_MISC_CORRECTED`, `NVFP4_CONSOLIDATED`, `TCGEN05_DEDUP_CONSOLIDATED` |
| `*_INCONSISTENCY_LOG.md` (per-topic) | 20 logs (one per category +  cross-cuts) |
| Wave-1+2 synthesis | `MASTER_INDEX.md`, `HEADLINE_CORRECTIONS.md`, `B300_TRUE_REFERENCE_v2_DRAFT.md` |

### Wave-3a topical corrections (4 files)

| File | Subject | Outcome |
|---|---|---|
| `CURIOSITY_LISTS_AUDIT.md` | All `CURIOSITY_LIST_V*.md` hash verification | V2: 22/25 hashes hallucinated. V4-V8: 100% verified. |
| `TCGEN05_POWER_CONSOLIDATED.md` | tcgen05 perf/W cross-doc reconciliation | Single-trial PERF_WATTS RETRACTED; use 2-trial CLEAN |
| `STRAYS_CORRECTED.md` | 14 stray b300_clean files not in 17-category sweep | HBM_DATA_DEPENDENCE.md SUPERSEDED; 4 files have 96 MB L2 cosmetic error |
| `A_TO_D_RIGOR_AUDIT.md` | A1/A2/A3/A4/A6/B1/B2/C3/D2/D3/D5/D6 vs V40-V51 | 8 retractions (A1/A6/B1/D5 partials); A4/D6/C3/D2/D3 confirmed |

### Wave-3b adversarial doubt reports (6 files)

| File | Audit subject | Verdict |
|---|---|---|
| `SYNTHESIS_DOUBT_LOG.md` | MASTER_INDEX, HEADLINE, TRUE_REFERENCE_v2_DRAFT | 5 HIGH-severity over-resolutions; no CRIT |
| `V46_DOUBT_REPORT.md` | V46 demotion verdict | Demotion CORRECT; soften wording — number is honest, only % framing was wrong |
| `DUAL_ISSUE_DOUBT_REPORT.md` | V49/V50 55%/74% methodology | **DOWNGRADE to LOW** — under-occupied baseline; no ncu; M8 counter-evidence |
| `DSMEM_DOUBT_REPORT.md` | DSMEM 40 GB/s + "no shared bus" plausibility | 40 GB/s chain-bound; 560 GB/s issue-rate; "no shared bus" 30× under-issued |
| `NVFP4_DOUBT_REPORT.md` | NVFP4 A:B 3-way reconciliation | All 3 readings real; wave-2 over-resolved single mechanism |
| `CROSS_AGENT_DOUBT_LOG.md` | 14 cross-agent contradictions | 4 SEV1, 6 SEV2, 4 SEV3 ranked |

### Wave-3c synthesis (3 files, this batch)

| File | Purpose |
|---|---|
| `DOUBT_LOG.md` | Per-claim verdicts on 10 wave-1+2 headlines + new top-10s |
| `HEADLINE_CORRECTIONS_v2.md` | Doubt-aware 1-page TL;DR |
| `MASTER_INDEX_v2.md` | THIS FILE |

---

## 2. Confidence ladder per CORRECTED file (post-doubt)

| File | Original tag | Doubt-aware tag | Reason |
|---|---|---|---|
| `01_hbm_bandwidth_CORRECTED.md` | HIGH | **HIGH** | Anchors 7672 spec; multi-source ladder; only HBM write 7.57 attribution open |
| `02_shmem_CORRECTED.md` | HIGH | **HIGH** | Bank-conflict regime split well-flagged; UNRESOLVED honest |
| `03_caches_CORRECTED.md` | HIGH | **HIGH** | Sole agent that disambiguates 3 L2 BW metrics |
| `04_fp32_peak_CORRECTED.md` | HIGH | **MED** | Adopts V49/V50 55%/74% which doubt swarm downgraded to LOW |
| `05_fp_precision_nontensor_CORRECTED.md` | MED | MED | No challenge from doubt swarm |
| `06_tensor_cores_CORRECTED.md` | HIGH | **MED** | Cites stale 10.8 PF K=96 without deferring to NVFP4 agent's 11.42 |
| `07_atomics_CORRECTED.md` | MED | **MED** | L2 atomic units ~32 should be MED not HIGH per STRAYS audit |
| `08_sync_primitives_CORRECTED.md` | MED | **MED-HIGH** | Cleanly preserves spread on threadfence; F2/F6 correctly reconciled |
| `09_memory_apis_CORRECTED.md` | MED | **LOW-MED** | Uses 7.2 TB/s denominator (yet another denominator); promotes V46 to "NEW SoL" which HBM agent demotes |
| `10_launch_overhead_CORRECTED.md` | MED | MED | No challenge |
| `11_block_scheduling_CORRECTED.md` | MED | MED | Topology-only |
| `12_nvlink_p2p_CORRECTED.md` | HIGH | **HIGH** | NVLink-5 web-confirmed |
| `13_pcie_system_CORRECTED.md` | MED | MED-HIGH | Defers correctly to NVLink agent |
| `14_math_intrinsics_CORRECTED.md` | MED | **HIGH** | Cross-agent consensus on 4.74/9.22 G MUFU |
| `15_integer_bit_ops_CORRECTED.md` | MED | **MED** | IADD3 0.5 vs 0.66 unresolved |
| `16_power_clock_CORRECTED.md` | MED | MED | No direct challenge |
| `17_nvrtc_module_CORRECTED.md` | MED | MED | No challenge |
| `DSMEM_CORRECTED.md` | HIGH | **MED** (downgraded) | 40 GB/s chain-bound; 560 GB/s issue-rate; "no shared bus" under-issued |
| `META_DOCS_CORRECTED.md` | MED | MED | Correctly demotes V46 framing |
| `M_SYNTHESIS_CORRECTIONS.md` | MED | **LOW-MED** | Adopts V49/V50 dual-issue (LOW); flattens IADD3/PRMT 30% gaps |
| `V8_V10_MISC_CORRECTED.md` | MED | MED | 3-source FFMA cap well-supported |
| `NVFP4_CONSOLIDATED.md` | MED + open | **MED** (preserve all 3 A:B readings) | Wave-2 over-resolved single mechanism |
| `TCGEN05_DEDUP_CONSOLIDATED.md` | MED + open | MED + open | No challenge |
| `B300_TRUE_REFERENCE_v2_DRAFT.md` | HIGH | **MED** (per row) | See HEADLINE_CORRECTIONS_v2 for row-by-row downgrades |
| `MASTER_INDEX.md` (wave-1+2) | reference | reference (preserved) | Superseded by this v2 |
| `HEADLINE_CORRECTIONS.md` (wave-1+2) | reference | reference (preserved) | Superseded by HEADLINE_CORRECTIONS_v2 |
| `TCGEN05_POWER_CONSOLIDATED.md` (wave-3a) | HIGH | **HIGH** | 2-trial supersedes contaminated single-trial |
| `STRAYS_CORRECTED.md` (wave-3a) | MED-HIGH | MED-HIGH | HBM_DATA_DEPENDENCE supersession well-supported |
| `A_TO_D_RIGOR_AUDIT.md` (wave-3a) | HIGH | HIGH | Per-doc verdicts cleanly cite V40-V51 |
| `CURIOSITY_LISTS_AUDIT.md` (wave-3a) | HIGH | HIGH | Hash verification empirical |

---

## 3. Open mechanism questions (UNRESOLVED, pulled from doubt + corrections)

| # | Question | Required test | Source |
|---|---|---|---|
| 1 | `__threadfence_system` true cost (1750 / 2870 / 3042 cy = 1.74× spread) | `bench_fence_cost.cu` at LOCKED 1920 MHz, explicit cy+ns | SYNC log #3, CROSS_AGENT #2, SYNTHESIS_DOUBT L3 |
| 2 | `__threadfence` (GPU) cost (258/281/292/320 cy = 24% spread) | Single ncu pass with all 4 patterns | CROSS_AGENT #2 |
| 3 | HBM write 7.57 TB/s SoL provenance (NINJA STG vs V8 TMA bulk) | Re-run both back-to-back with ncu `dram__bytes` | HBM log #3 |
| 4 | V49/V50 dual-issue: at 4-8 warps/SMSP does dual stay capped or rise to 100%? | warps/SMSP ∈ {1,2,4,8} sweep + ncu `smsp__inst_issued.avg.per_cycle_active` | DUAL_ISSUE_DOUBT |
| 5 | DSMEM read non-chained ILP ceiling (40 GB/s chain vs ?) | Address-derived-from-i ILP test | DSMEM_DOUBT |
| 6 | DSMEM write delivery rate vs issue rate | Fenced ring write retest | DSMEM_DOUBT |
| 7 | DSMEM "no shared bus" — full-issue 8-cluster ring sweep | 8 CTAs × 4 warps × ILP=16 ring, sweep N=2..8 clusters | DSMEM_DOUBT |
| 8 | NVFP4 A:B 3-way mechanism — does TMA multicast OR A↔B swap OR pipeline depth dominate? | Custom NVFP4 kernel WITHOUT multicast at cuBLAS shape | NVFP4_DOUBT |
| 9 | NVFP4 11.42 PF reproducibility | cudaGraph BPG sweep across 5+ shapes | NVFP4_DOUBT |
| 10 | IADD3 rate 0.5 (A6) vs 0.66 (V40) — which is architectural? | A6-style sweep at 4+ warps/SMSP | INT log #A, A_TO_D_RIGOR_AUDIT U2 |
| 11 | PRMT pipe placement — V40 "permute" vs A6 "INT-bit"? | A6-style sweep on PRMT | INT log UNRESOLVED #2 |
| 12 | A3 scoreboard depth — true value (≥32, never plateaued in test) | Test with N>32, ncu `smsp__inst_executed_pipe_lsu` | STRAYS §3, A_TO_D_RIGOR_AUDIT U3 |
| 13 | L2 atomic unit count — true value | Stride sweep with ncu lts__t_bytes per partition | ATOMICS log A5, STRAYS §8 |
| 14 | Single-MMA cache depth — 1 slot, 2 slots, or 1+alternation? | Pattern rotation under per-MMA isolation AND sustained throttle | DEDUP log A1, B5 |
| 15 | Why is 2-pattern (ABAB) sub-tile WORSE than 3-pattern (ABCABC)? | Microsweep with per-cycle clock64 | DEDUP log B4 |
| 16 | TMA pipeline-depth optimum (V46 used 8; knee unknown) | Sweep depth 2..16 with ncu | TMA log #1 |
| 17 | TMA multicast at cluster ∈ {2,4,6,8} — only cluster=8 tested | Cluster sweep | TMA log #2 |
| 18 | `cuStreamWriteValue32` cost — 0.45 µs (memory) or 2.47 µs (catalog)? | Decompose host-call vs full pair | TMA log F |
| 19 | LDS 32-way bank-conflict cost across regimes | Single-warp vs multi-warp matrix | SHMEM log #2 |
| 20 | Cluster=2 21% slower than ≥3 — single-GPC vs multi-GPC hypothesis | `gpc__cycles_active.per_pgpc_id` ncu pass | DSMEM log E, TOPOLOGY log #2 |
| 21 | Cooperative-grid SM mapping (never measured) | `cudaLaunchCooperativeKernel` + per-CTA SM-id dump | TOPOLOGY log #12 |
| 22 | `mma.sync kind::f8f6f4` sub-tile dedup behaviour (only tcgen05 tested) | Repeat dedup recipe with mma.sync FP8 | DEDUP log D3 |
| 23 | NVRTC vs nvcc cubin equivalence (never SASS-diffed) | Compile both paths, sass-diff | PRECISION log #7 |
| 24 | A4/D6 broadcast operand reuse cache mechanism | ncu `smsp__inst_executed_pipe_fma_collector_*` if available | A_TO_D_RIGOR_AUDIT U6 |
| 25 | A1 SHFL + FFMA 14.7% overlap mechanism ("SHFL occupies issue port multi-cycle"?) | Redo dual-issue with V37/V38 setup | A_TO_D_RIGOR_AUDIT U4 |
| 26 | B2 LDG no-chain SLOWER than chain-dep mechanism ("queue backpressure"?) | Retest with current anti-DCE recipes | A_TO_D_RIGOR_AUDIT U7 |
| 27 | DRAM data-dependence at boost clock (only 1005 / 1500 measured) | Popcount sweep at 1920/2032 MHz | POWER log #G |
| 28 | NVFP4 K=96 ULTRA at non-square N (K-id shape-conditional) | NVFP4 N-shape sweep with constant baseline | NVFP4 log "OPEN K-id" |
| 29 | Cross-precision NVFP4 K=96 path: does K-id 5-gate model from BF16 apply? | Cross-precision K-id N-stride sweep | NVFP4 log |
| 30 | TF/W boost clock (full 7-precision ladder; only spot checks done at boost) | Full ladder at boost | TCGEN05_POWER_CONSOLIDATED U1 |

---

## 4. Methodology rules learned from this audit

| # | Rule | Lesson source |
|---|---|---|
| 1 | **Always specify HBM denominator** (7672 GB/s post-ECC vs 7.31 empirical vs 8.0 nominal) | V46_DOUBT, CROSS_AGENT #1 — 4 different denominators across 3 docs |
| 2 | **Always specify L2 BW metric** (kernel-effective ~24 / wire-lts ~13 / L1-amplified ~30 TB/s) | CROSS_AGENT #4 — only cache agent disambiguates |
| 3 | **Always sweep warps/SMSP** for dual-issue / pipe-overlap tests; 2 warps/SMSP is under-occupied | DUAL_ISSUE_DOUBT, A_TO_D_RIGOR_AUDIT (A1/A6/B1) |
| 4 | **Always git-verify TODO-list hashes** — V2 had 88% hallucination rate when hashes cited from memory | CURIOSITY_LISTS_AUDIT |
| 5 | **Always pair clock64 with ncu** for "% of pipe peak" claims (CLAUDE.md §4 — 3-method rigor) | DUAL_ISSUE_DOUBT, A_TO_D_RIGOR_AUDIT |
| 6 | **Always specify chain vs non-chain** for memory BW claims | DSMEM_DOUBT (40 GB/s is chain-bound) |
| 7 | **Always specify issue-rate vs completion** for write BW (require fence between stores and timer) | DSMEM_DOUBT (560 GB/s is issue rate) |
| 8 | **Don't pick a single number from a 1.5×+ spread** (system fence; PRMT placement) — flag UNRESOLVED | CROSS_AGENT #2, SYNTHESIS_DOUBT M3 |
| 9 | **Don't take credit for upstream retractions** — check source self-retractions before re-headlining | SYNTHESIS_DOUBT H5 |
| 10 | **Don't promote one mechanism when sources list multiple** (NVFP4 A:B; A4/D6 reuse cache) | NVFP4_DOUBT, A_TO_D_RIGOR_AUDIT U6 |
| 11 | **Always run `pkill -9 QuickRunCUDA` + 6s cooldown** between power measurements (single-trial contamination is silent) | TCGEN05_POWER_CONSOLIDATED §2 |
| 12 | **Always state clock state** (boost / 1920 lock / 1500 lock / 1005 lock) on every TFLOPS or W number | TCGEN05_POWER_CONSOLIDATED §7 |
| 13 | **Always state data-dependence** (zero / const / random) on every TFLOPS or W number | 06_tensor_cores_CORRECTED v2 rule |
| 14 | **Always state per-call vs sustained-via-cudaGraph** for cuBLAS numbers | 06_tensor_cores_CORRECTED v2 rule |
| 15 | **Always state denominator scope** (per-CTA / per-cluster / chip-aggregate) for memory claims — DSMEM v1 vs v2 disagreement was scope, not measurement | SYNTHESIS_DOUBT H4 |
| 16 | **Never cite `pipe_tensor.cycles_active` for tcgen05** (use `smsp__inst_executed_pipe_tensor_subpipe_hmma_op_utchmma_utcqmma_utcomma_scope_2cta.sum`) | TENSOR log §B |
| 17 | **Never cite "% of NVLink-4 spec 757 GB/s"** — B300 is NVLink-5 (900 GB/s/dir spec); legacy % numbers are inflated by 19% | NVLink + PCIe agents |
| 18 | **Never trust CLAUDE.md memory as authoritative** — it has at least 4 retracted entries (NVLink v7, HBM 8 TB/s, SMEM atomic 4.2 T, K-96 10.8 PF) | CROSS_AGENT pattern |

---

## 5. Reading order for new investigations

1. **`HEADLINE_CORRECTIONS_v2.md`** — 1-page TL;DR with confidence flags
2. **`DOUBT_LOG.md`** — per-claim verdicts; new top-10 confirmed/needs-revision/new
3. Per-topic CORRECTED file (this index §1)
4. Matching INCONSISTENCY_LOG for alternative numbers
5. **`B300_TRUE_REFERENCE_v2_DRAFT.md`** for canonical row + supersedes pointers (NOTE: not all rows are doubt-aware; cross-check against HEADLINE_CORRECTIONS_v2 for downgrades)
6. Original docs in `b300_clean/` (NOT modified) for full-context

For NEW measurements: `utils/rigor_run.sh` + the methodology rules in §4.
