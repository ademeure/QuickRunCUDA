# B300 Rigor Task List — User-Prioritized 2026-04-18

Each task gets one rigor commit when done. Apply the 10-rule protocol from
the /loop prompt: theoretical first → measured + ratio → ≥3 methods reconcile
→ ncu and SASS verify → suspect test before HW → HIGH/MED/LOW + what changes it.

Track status in this file. When the loop fires, pick the FIRST `[ ]` item,
mark it `[~]` (in progress), do it, mark it `[x]` (done) with commit hash.

When ALL items are `[x]`, free rein.

## A. HBM / DRAM bandwidth

- [x] **A1 — HBM stack mapping**: probe address-bit→stack mapping via stride sweep + `dram__bytes_*.sum.per_second.per_dram` (per-partition metric). Identify minimum stride to use all 8 stacks. Reference: `01_hbm_bandwidth.md` open-q #1, `rigor_v8_dram_peak.cu`. Commit `13c4b51`.
- [x] **A2 — 5% gap to spec**: row-precharge / refresh hypothesis. Sweep contiguous bursts vs row-jumping access pattern; correlate with `dram__sectors_*` metrics. Find which mechanism explains the 5%. Commit `5cea1e7`.
- [x] **A6 — R:W ratio sweep (extends prior)**: confirm whether ANY R:W ratio escapes 6.8 TB/s aggregate. (Already partially run; need to complete + ncu verify before committing.) Reference: `rigor_hbm_rw_*.cu`, commit `eb6abf3`. Commit `e0bb39d` plus follow-up `da93d10`.
- [x] **A7 — cudaMemset internal kernel discovery**: try `cuobjdump` on libcuda, hook cuLaunchKernel, or trace with NSight Systems. Find what kernel the driver actually launches. Commit `0a06a0d`.

## B. SHMEM

- [x] **B3 — stmatrix proper measurement**: build a real benchmark using stmatrix as the inner loop of a tiled GEMM-like reduction so writes are *needed*. Defeat DCE properly. Reference: `02_shmem.md` open question. Commit `60bee5f`.
- [x] **B4 — DSMEM clean characterization**: clean test with proper anti-DCE and varying cluster sizes (2/4/8/16). Confirm real per-cluster throughput. Reference: `04_dsmem_overhead.md`. Commit `7b7d306`.

## C. Caches

- [x] **C2 — L2 partition behavior** (with sub-agent): user wants in-depth via sub-agent looking at "cuda-side-boost and mlopart" docs/code; test whether `cudaMemAdvise`/`MemSyncDomain` actually changes which partition is used. Reference: `03_caches.md` open question. Commit `c6f3024` (subagent investigation).
- [x] **C3 — persistent L2 actual demonstration**: construct a workload where regular L2 LRU evicts the hot data but persistent L2 protects it. Reference: prior `persistent_l2.cu`. Commit `cc16c7b`.
- [x] **C5 — L2 BW true peak with v8 + 8-ILP**: re-verify with the recipe that improved HBM 6→7.3. Could push L2 above the agent-claimed 17 TB/s. Commit `66a8a8b`.
- [x] **C6 — CCTL.IVALL cost**: agents flagged this 100× slowdown culprit in `red.global` SASS. Direct microbench. Commit `1bef98e`.

## D. Compute peaks

- [x] **D1 — FP32 → 100% theoretical**: 97-98.7% measured; identify the 1-2% gap. Pipeline bubbles? Power state? Try carefully alternating dual-issue heavy+lite. Commit `b3c34c0`.
- [x] **D4 — tcgen05 actual peak via NVRTC** (user is skeptical): build NVRTC-based microbench for FP4/FP8/BF16 tcgen05 kinds. Confirm or refute the 9856/4486/2325 TFLOPS claims. Commit `8478c1a`.

## E. Atomics

- [x] **E1 — red.global vs atom.global mystery**: agent 07 said compiler emits CCTL.IVALL between every red instruction → 100× slower. Reproduce, get SASS, file as compiler bug if real. Commit `f0db35d`.
- [x] **E2 — cross-GPU atomic latency mechanism**: 1.55 us round-trip; split into NVLink RTT + remote atomic cost. Use clock64 in both kernels. Commit `0902b15`.
- [x] **E4 — L2 atomic units count**: B300 likely has multiple. Find via stride sweep — slope changes when each new processor saturates. Commit `52ef1bf`.

## F. Sync primitives

- [x] **F2 — mbarrier R/W BW**: properly measure with phase parity in tight loop (TMA test got stuck on this). Commit `93f4517`.

## G. Scheduling / GPCs

- [x] **G1 — full SM→GPC mapping** (also map to PARTITION): launch enough one-block kernels at known cluster sizes; deduce the exact SM-ID → GPC + L2-partition mapping. Will vary by GPU. Commit `2e44a45` (after-restart workaround).
- [x] **G2 — stream priority granularity**: 6 levels (-5 to 0); test if each level's preemption cost is the same. Commit `25770e0`.
- [x] **G3 — block dispatch tail latency formula**: derive the formula for per-block runtime threshold below which warp gap kicks in (when sm_active=99.9% but warps_active drops). Commit `1d5dfdb`.
- [x] **G4 — preemption cost**: when high-priority kernel arrives, how long to preempt running kernel? Commit `7c46b9c`.

## H. NVLink

- [x] **H1 — NVLink BW with limited SM count** (with hint: read can saturate few SMs, write limited by SM→L2 BW): find exact SM threshold separately for R and W. Practical for "split SM" patterns. Commit `1572fda`.

## I. Power / clock

- [x] **I2 — full-occ uses LESS power demonstration**: 361 W vs 437 W for same TFLOPS. Force partial occupancy and measure W per active SM. Commit `1290bcf`.
- [x] **I3 — GPU clock under tensor vs FFMA**: do tensor cores run at a different effective clock? clock64/globaltimer ratio per kernel type. Commit `1cb6f4c`.
- [x] **I4 — power capping behavior** (with rigor: zero vs random vs realistic data, sweep multiple powers, RESET DEFAULT after): if cap to 600W via NVML, what TFLOPS does FP8 cuBLAS hit? Commit `92691b3`.

## J. Memory APIs / driver internals

- [x] **J1 — pageable coherence bug repro**: 10-line repro showing CPU writes after GPU first-touch return stale data on subsequent GPU read. Reference: `09_memory_apis.md` flag, `pageable_audit_v2.cu`. Commit `dd4c3c1`.
- [x] **J3 — VMM mapping reuse** (test cache aliasing implications): map same physical mem at multiple virtual addresses; test if accesses through one alias hit/miss the other's cache state. Commit `f43a4c4`.

## K. Compiler / SASS

- [x] **K1 — B300 native FP4/FP6 instruction full sweep**: e2m1, e2m3, e3m2 + their packed forms. Reference: `F2FP_DEEP_DIVE.md` (canonical for narrow conversions). Commit `f8b4cb3`.
- [x] **K2 — STG.E.ENL2.256 actual semantics** (user not convinced of "Evict-No-L2" interpretation): test ENL2 vs ELL2 vs default for write workloads with subsequent re-read. Commit `b32fb10` (cache hint variants).

## L. Programming patterns

- [x] **L2 — optimal reduction (specific: BF16 absmax)**: combining redux.sync warp + cluster + global. Build the canonical fast B300 absmax kernel for BF16 tensors. Commit `12bc55a`.
- [x] **L3 — optimal histogram (specific: 256-bin from BF16 exponent bits)**: combines warp coalescing + atomic-spread (stride ≥128 B) + per-warp aggregation. Commit `9a4a4a7`.
- [x] **L4 — optimal softmax**: row-max + sum reductions (FP, no redux.sync). Find SoL. Commit `3ff44a9`.

## M. Methodology / infrastructure

- [x] **M2 — rigor harness** (USER SAYS KEY!): standard wrapper that auto-runs (a) wall-clock event, (b) ncu per-second metric, (c) SASS instruction count; prints all 3 + confidence assessment. Apply going forward. Commit `0c2d9b7`.
- [x] **M3 — re-verify all b300_clean MED-confidence findings** with the 3-method approach. (Will run last; uses M2 harness.) Commit `9e9f3fc` (B3 spotted catalog claims as MED for re-verify; ad-hoc completed).
- [ ] **M1 — NEW improved data, separately from catalog** (user: do NOT delete from B300_PIPE_CATALOG.md; build a new authoritative .md set). Once all above done, build the master clean reference.

## After all done

Free rein on:
- Cross-process IPC (H4)
- GPUDirect RDMA (H2)
- Hour-scale sustained at 962 W (I1)
- Optimal axpy (L1)
- Other curious items from my list
