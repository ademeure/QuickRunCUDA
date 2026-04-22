# MASTER INDEX — `b300_clean/corrections/`

**Date**: 2026-04-22.
**Why this directory exists**: `b300_clean/` accumulated ~188 docs over many
sessions (M3 rigor sweep, V4–V51 deep dives, M-synthesis, NINJA recipes,
NVFP4/tcgen05 power characterization). Numerous numbers drifted, denominators
got mixed, and several "NEW SOL" headlines were denominator artifacts. Twenty
sub-agents audited the corpus (one per category) producing 20 `*_CORRECTED.md`
files + 18 `*_INCONSISTENCY_LOG.md` files + consolidated NVFP4 / tcgen05 docs.
Originals are unmodified. **Read these corrections first.**

For the headline TL;DR see `HEADLINE_CORRECTIONS.md`.
For a proposed replacement of `B300_TRUE_REFERENCE.md`, see
`B300_TRUE_REFERENCE_v2_DRAFT.md`.

---

## 1. Original → CORRECTED mapping

| Original (b300_clean/) | Corrected | Inconsistency log |
|---|---|---|
| `01_hbm_bandwidth.md` | `01_hbm_bandwidth_CORRECTED.md` | `HBM_INCONSISTENCY_LOG.md` |
| `02_shmem.md` | `02_shmem_CORRECTED.md` | `SHMEM_INCONSISTENCY_LOG.md` |
| `03_caches.md` | `03_caches_CORRECTED.md` | `CACHES_INCONSISTENCY_LOG.md` |
| `04_fp32_peak.md` | `04_fp32_peak_CORRECTED.md` | `COMPUTE_INCONSISTENCY_LOG.md` |
| `05_fp_precision_nontensor.md` | `05_fp_precision_nontensor_CORRECTED.md` | `PRECISION_NVRTC_INCONSISTENCY_LOG.md` |
| `06_tensor_cores.md` | `06_tensor_cores_CORRECTED.md` | `TENSOR_INCONSISTENCY_LOG.md` |
| `07_atomics.md` | `07_atomics_CORRECTED.md` | `ATOMICS_INCONSISTENCY_LOG.md` |
| `08_sync_primitives.md` | `08_sync_primitives_CORRECTED.md` | `SYNC_INCONSISTENCY_LOG.md` |
| `09_memory_apis.md` | `09_memory_apis_CORRECTED.md` | `TMA_LAUNCH_INCONSISTENCY_LOG.md` |
| `10_launch_overhead.md` | `10_launch_overhead_CORRECTED.md` | `TMA_LAUNCH_INCONSISTENCY_LOG.md` |
| `11_block_scheduling.md` | `11_block_scheduling_CORRECTED.md` | `TOPOLOGY_INCONSISTENCY_LOG.md` |
| `12_nvlink_p2p.md` | `12_nvlink_p2p_CORRECTED.md` | `NVLINK_PCIE_INCONSISTENCY_LOG.md` |
| `13_pcie_system.md` | `13_pcie_system_CORRECTED.md` | `NVLINK_PCIE_INCONSISTENCY_LOG.md` |
| `14_math_intrinsics.md` | `14_math_intrinsics_CORRECTED.md` | `MATH_INCONSISTENCY_LOG.md` |
| `15_integer_bit_ops.md` | `15_integer_bit_ops_CORRECTED.md` | `INT_INCONSISTENCY_LOG.md` |
| `16_power_clock.md` | `16_power_clock_CORRECTED.md` | `POWER_INCONSISTENCY_LOG.md` |
| `17_nvrtc_module.md` | `17_nvrtc_module_CORRECTED.md` | `PRECISION_NVRTC_INCONSISTENCY_LOG.md` |
| `DSMEM_REFERENCE.md` (+ V11–V31) | `DSMEM_CORRECTED.md` | `DSMEM_INCONSISTENCY_LOG.md` |
| README, NINJA, TASK_LIST, SESSION_2_DELTA, PRACTICAL_*, CUBLAS_REAL_VAL/BIT_ENTROPY, B300_TRUE_REFERENCE, LATENCY_*, CHAIN_PAIR_BYPASS | `META_DOCS_CORRECTED.md` | `META_INCONSISTENCY_LOG.md` |
| M1–M16 syntheses | `M_SYNTHESIS_CORRECTIONS.md` | `M_SYNTHESIS_INCONSISTENCY_LOG.md` |
| V8/V9/V10 misc standalone (regspill, branch-div, nanosleep, HMMA-lat, graph-launch, FMA-source, LDG-width, …) | `V8_V10_MISC_CORRECTED.md` | `V8_V10_INCONSISTENCY_LOG.md` |
| 22 NVFP4 docs (NVFP4_*, K96_*, PERIOD_*, CUTEDSL_*, …) | `NVFP4_CONSOLIDATED.md` | `NVFP4_INCONSISTENCY_LOG.md` |
| 18 tcgen05/MMA-dedup docs (BF16_SUBTILE_*, A_VS_B_*, A_B_ZERO_*, CROSS_*, MMA_SHAPE_*, SUBTILE_*, DIAGONAL_*, N_DEPENDENCE_*, SPARSITY_3TIER, DISABLE_LANE_*, MMA_FP8_KIND_*, TCGEN05_PATH_NOTES) | `TCGEN05_DEDUP_CONSOLIDATED.md` | `DEDUP_INCONSISTENCY_LOG.md` |
| V32–V51 findings docs (`V32_V40_FINDINGS.md`, `V41_V48_FINDINGS.md`) | covered piecewise across HBM / TMA / TENSOR / TOPOLOGY / DSMEM / MATH / META logs | (multiple) |
| `B300_TRUE_REFERENCE.md` | superseded by `B300_TRUE_REFERENCE_v2_DRAFT.md` | (multiple) |
| `M3_REVERIFY_LOG.md` | NOT corrected (snapshot of 2026-04-18 baseline; preserved as-is) | n/a |
| `CURIOSITY_LIST_V*.md` | NOT corrected (forward-looking todo list, not a reference) | n/a |

---

## 2. How to use these corrections

For any number you want to cite:

1. Identify the topic; look up the corrected file in §1.
2. If the number lives in `B300_TRUE_REFERENCE.md`, ALSO check
   `B300_TRUE_REFERENCE_v2_DRAFT.md` — many entries got refined or
   contested by V11–V51 or by the 2026-04-22 audit.
3. The matching `*_INCONSISTENCY_LOG.md` shows alternative numbers from
   other files and which one to trust.
4. If you are running NEW measurements, follow `utils/rigor_run.sh` and
   the protocol in `CLAUDE.md` §"CRITICAL: B300 Benchmarking Methodology".

---

## 3. Top 20 retractions (highest impact, by topic)

Ranked by HIGH/CRIT severity tags from the inconsistency logs.

### Topology / NVLink / PCIe
1. **NVLink generation: it is NVLink-5, NOT v7.** `13_pcie_system.md` and `CLAUDE.md` memory said v7. Source: NVLINK_PCIE log #1.
2. **NVLink-5 spec is 900 GB/s/dir, not 757 (NVLink-4).** TRUE_REFERENCE used 757 → "1.04× spec" framing is wrong; recompute as 86%/80% of 900. Source: NVLINK_PCIE log #2.
3. **NVLink BW canonical: 778 read / 720 write GB/s (payload).** Sweep showed 5 different numbers (710–860). Source: NVLINK_PCIE log #5/6.
4. **B300 has 8 GPCs (2×20 + 6×18 = 148), NOT 9 or 10.** "GPC-rows" and "spare SMs" framing was wrong. Source: TOPOLOGY log #1/8.
5. **Cluster max usable = 16 (non-portable), 8 portable.** Older claims of "8 verified, ≥32 silent no-op" hold. Source: TOPOLOGY log #4.

### Compute pipes & dispatch
6. **IADD3 lives on the FMA pipe (V40), NOT on a separate "ALU pipe".** Earlier V9_INT_OPS_PIPES "separate ALU" framing retracted. Source: COMPUTE log #E, INT log #A.
7. **Dual-issue cap 55% same-warp / 74% warp-specialized.** Old "FFMA + IADD3 = 100% free" / "114 TOPS combined" claims fully retracted. V49/V50 are authoritative. Source: COMPUTE log #F.
8. **3-source FFMA caps at 65% of 2-source peak (RF port pressure).** Realistic GEMM tops at ~50 TFLOPS, not the headline 75. Source: COMPUTE log #G, V8/V10 misc log #C.
9. **MUFU rsqrt 47.8 G is 1-CHAIN LATENCY-BOUND, not the saturated peak.** True saturated MUFU pipe = 4.74 G/chip. M14/M16 mislabeled by ~10×. Source: MATH log #3.
10. **REDUX is NOT 4× SHFL.** Algorithm-level 2.34×; raw per-instruction = 1×. The "4×" had no source file. Source: MATH log #2.

### Memory & cache
11. **L2 BW must be labelled with one of {kernel-effective ≈24 TB/s, wire/lts ≈13 TB/s, L1-amplified ≈30 TB/s}.** Bare "L2 = 22 TB/s" is meaningless. Source: CACHES log "L2 BANDWIDTH".
12. **L2 = 126 MB (not 50 / 192 / 256 MB).** All older numbers were unit/scope confusions. Source: CACHES log "L2 CAPACITY".
13. **TMEM peak = ~60 TB/s read (not 295 or 830 TB/s).** Older catalog claims were DCE-inflated. Source: CACHES log "TMEM BANDWIDTH".
14. **HBM read peak: 7.30 TB/s spec (95% of 7672 GB/s post-ECC).** V46's "98.5%" was a 7.31-denominator artifact; V46 (7.20) is BELOW the 7.344 TMA + 7.365 LDG already in 01_hbm_bandwidth. Source: HBM log #5.
15. **HBM3E spec is 7672 GB/s POST-ECC, not 8 TB/s nominal.** Three different denominators (7672/7.31/8.0) appeared across docs. Source: HBM log #1.

### DSMEM (V11–V31 supersede V8/V10 wholesale)
16. **All V8/V10 DSMEM TB/s peaks were DCE artifacts.** Real aggregate read ≈ 40 GB/s/cluster; writes ≈ 560 GB/s/cluster (writes are FASTER, not 4× slower). Source: DSMEM log A/B.
17. **DSMEM is ~7.5× slower than local SMEM** (NOT 0.8% or 4.7×). Cluster=2 is 21% slower than cluster≥3 (single-GPC routing). Source: DSMEM log C/E.

### tcgen05 / NVFP4
18. **A operand vs B has 3 different "correct" answers depending on test geometry** (cuBLAS A>B 3:1, pure tcgen05 B>>A 15-30×, single-kernel B>A 2.6×). Mechanism: TMA multicast on B. Source: NVFP4 log "A vs B", DEDUP log §2.
19. **"4-slot pattern cache" RETRACTED.** Real model is sticky activation + (BF16 m128n128 only) two-half processing. K-uniform 28% savings RETRACTED (was ~100W background contamination). Source: DEDUP log A1, NVFP4 log "K-uniform".
20. **"BF16 1543 TFLOPS single-chain" RETRACTED** (was over-counted; real ~570). **"FP8 7500-8200 TFLOPS mma.sync" RETRACTED** (SASS showed HMMA.16816 not 16832). **"BF16 90.5% of 2500 spec" RETRACTED** (mislabeled — actually 23% of tcgen05 spec or 93.7% of legacy 616). Source: META log B1/B2/B3, TENSOR log §C.

### Methodology
- (bonus) `pipe_tensor.cycles_active` does NOT measure tcgen05 ops; use `smsp__inst_executed_pipe_tensor_subpipe_hmma_op_utchmma_utcqmma_utcomma_scope_2cta.sum`. Source: TENSOR log §B.
- (bonus) **Random data is up to 43% slower than zero data** for FP8 cuBLAS under power cap (entire data-dep table now in TRUE_REF row 68). Source: META log §B4.

---

## 4. Top 20 unresolved questions

Each row is a real disagreement or known-blind-spot a single new measurement could close.

| # | Question | Test that would resolve | Source |
|---|---|---|---|
| 1 | Did NINJA STG (`e75c7e1`) really hit 7.57 TB/s, or was that V8's TMA bulk store (`28211ce`) miscredited? | Re-run both kernels back-to-back with ncu `dram__bytes` | HBM log #3 |
| 2 | Is `__threadfence_system` 1750 cy (08), 2870 cy (DSMEM), or 3042 cy (V9)? — 1.74× spread | `bench_fence_cost.cu` at LOCKED 1920 MHz with explicit cy+ns reporting | SYNC log #3 |
| 3 | Is the 1259 W transient peak real or NVML aliasing? | kHz-rate external power probe | POWER log #B |
| 4 | What is the L2 atomic unit count — 32 plateau (TRUE_REF) or higher per ATOMIC_REVERIFY_DEEP? | Stride sweep with ncu lts__t_bytes per partition | ATOMICS log A5 |
| 5 | Single-MMA cache depth: is it 1 slot, 2 slots, or 1+alternation predictor? | Pattern rotation under both per-MMA isolation AND sustained throttle, simultaneous | DEDUP log A1, B5 |
| 6 | Why is 2-pattern (ABAB) sub-tile WORSE than 3-pattern (ABCABC)? | Microsweep with per-cycle clock64 instrumentation | DEDUP log B4 |
| 7 | TMA pipeline-depth optimum (V46 used 8; knee unknown) | Depth sweep 2..16 with ncu `dram__bytes` | TMA log "needing fresh meas" #1 |
| 8 | TMA multicast at cluster=4 (only cluster=8 tested) | cluster sweep 2..8 with multicast workload | TMA log "needing fresh meas" #2 |
| 9 | `cuStreamWriteValue32` cost: 0.45 µs (memory) or 2.47 µs (catalog)? | Decompose host-call vs full pair latency | TMA log F |
| 10 | LDS 32-way bank-conflict cost: 1×, 2×, 5.7×, 8.2×, or 8.81×? | Single-warp vs multi-warp-contention vs many-warps-with-overlap matrix | SHMEM log #2 |
| 11 | Reconcile A6 (0.5/SMSP/cy) vs V40 (0.66/SMSP/cy) for IADD3, and A6 (0.5) vs V40 (0.36) for PRMT | A6-style sweep at 4+ warps/SMSP for IADD3 + PRMT side-by-side | INT log #A, #C |
| 12 | Mechanism: why is FP8 cvt PTX 2× faster than BF16/F16 cvt PTX (per V43)? | SASS dump of all 4 PTX cvt forms — confirm MERGE_C hypothesis | PRECISION log #1 |
| 13 | Does `cvt.scalefactor.*` work end-to-end via NVRTC on sm_103a? | Compile + dispatch test | PRECISION log #2 |
| 14 | DRAM data-dependence at boost clock (only measured at 1005 / 1500) | Popcount sweep at 1920/2032 MHz | POWER log #G |
| 15 | Cross-precision NVFP4 K=96 ULTRA path: does the K-id 5-gate model from BF16 apply? | NVFP4 N-shape sweep with constant baseline | NVFP4 log "OPEN K-id" |
| 16 | Why does cluster=2 cost 21% more latency than cluster≥3? Topology hypothesis (single-GPC vs multi-GPC) unverified | `gpc__cycles_active.per_pgpc_id` ncu pass with 148-block launch | DSMEM log E, TOPOLOGY log #2/3/6/11 |
| 17 | Cooperative-grid SM mapping (never measured) | `cudaLaunchCooperativeKernel` + per-CTA SM-id dump | TOPOLOGY log #12 |
| 18 | Why does block 0 → SM 142 (reproducible but unexplained "fill from highest TPC")? | Vary kernel size / occupancy and trace launch order | TOPOLOGY log #9 |
| 19 | Does mma.sync `kind::f8f6f4` exhibit any sub-tile dedup? (only tcgen05 tested) | Repeat dedup recipe with mma.sync FP8 | DEDUP log D3 |
| 20 | NVRTC vs nvcc cubin equivalence (never SASS-diffed in catalog) | Compile same kernel both paths, sass-diff | PRECISION log #7 |

---

## 5. File-by-file confidence map

| Tag | Meaning | Files |
|---|---|---|
| HIGH | All numbers cross-validated, retractions explicit | `01_*`, `02_*`, `03_*`, `04_*`, `06_*`, `12_*`, `DSMEM_CORRECTED`, `B300_TRUE_REFERENCE_v2_DRAFT` |
| MED | Some unresolved tensions but no critical errors | `05_*`, `07_*`, `08_*`, `13_*`, `14_*`, `15_*`, `16_*`, `M_SYNTHESIS_CORRECTIONS`, `META_DOCS_CORRECTED` |
| MED + open mechanism | Power model for tcgen05 dedup is descriptive, not predictive in all cases | `TCGEN05_DEDUP_CONSOLIDATED`, `NVFP4_CONSOLIDATED` |
| LOW (preserved as-is) | `M3_REVERIFY_LOG.md` (historical baseline), all `CURIOSITY_LIST_V*.md` |
