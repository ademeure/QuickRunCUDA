# B300 Rigor Task List — User-Prioritized 2026-04-18

Each task gets one rigor commit when done. Apply the 10-rule protocol from
the /loop prompt: theoretical first → measured + ratio → ≥3 methods reconcile
→ ncu and SASS verify → suspect test before HW → HIGH/MED/LOW + what changes it.

Track status in this file. When the loop fires, pick the FIRST `[ ]` item,
mark it `[~]` (in progress), do it, mark it `[x]` (done) with commit hash.

When ALL items are `[x]`, free rein.

## A. HBM / DRAM bandwidth

- [x] **A1 — HBM stack mapping**: probe address-bit→stack mapping via stride sweep + `dram__bytes_*.sum.per_second.per_dram` (per-partition metric). Identify minimum stride to use all 8 stacks. Reference: `01_hbm_bandwidth.md` open-q #1, `rigor_v8_dram_peak.cu`. Commit `96bcb5a`.
- [x] **A2 — 5% gap to spec**: row-precharge / refresh hypothesis. Sweep contiguous bursts vs row-jumping access pattern; correlate with `dram__sectors_*` metrics. Find which mechanism explains the 5%. Commit `66a2853`.
- [x] **A6 — R:W ratio sweep (extends prior)**: confirmed NO ratio escapes 7.31 TB/s; 50:50 mix is minimum at 6.68 TB/s (87.0% spec). U-shape verified 2 methods within 1%. Commit `de3b4d5`.
- [x] **A7 — cudaMemset internal kernel discovery**: 6 hook methods all blocked (cuobjdump, nsys, ncu, cuLaunchKernel, cuMemsetD*_v2, cuGetProcAddress). BUT: cudaMemset launches 31% FASTER than noop kernel (1.22 us vs 1.78 us) — driver private fast-path dispatch confirmed. Commit `be28c14`.

## B. SHMEM

- [x] **B3 — stmatrix proper measurement**: WORKS — STSM.16.M88.4 emits real SHMEM writes; W+R aggregate 34.5 TB/s = 90% of 38.5 SoL across all stmatrix:read ratios; pure stmatrix asymptotes at 33 TB/s = 86% SoL. ncu bank counters + wall-clock + SASS all agree. Commit `8bd85e8`.
- [x] **B4 — DSMEM clean characterization**: CL=2/4/8/16 = 3.06/1.57/1.45/1.27 TB/s aggregate, per-cluster scales linearly (41/42/80/141 GB/s), per-block degrades (20.7/10.6/10.1/8.8). L2 essentially zero — DSMEM intercepted by SM-to-SM mesh. Commit `4129760`.

## C. Caches

- [x] **C2 — L2 partition behavior** (with sub-agent): NO host API controls placement (5 methods identical). Latency near 295 cy, far 700 cy = 2.4× ratio. "mlopart" = MLOPart MPS feature (CUDA 13.0+ B200/B300, hard die-split into 2 CUDA devices). Hash 4 KiB granularity. Commit `af91798`.
- [x] **C3 — persistent L2 actual demonstration**: persistent attribute shows NO measurable benefit on B300 for streaming workloads. HOT=8/32/64 MB after 315 MB cold sweep: same 2.9-3.2 TB/s with/without persistent. ncu confirms ~66% L2 hit rate either way. B300 LRU may already be "smart". Commit `d2ccf76`.
- [x] **C5 — L2 BW true peak with v8 + 8-ILP**: kernel-effective 23.85 TB/s (96 MB), L2-bus (ncu lts) 13.3 TB/s (126 MB). Catalog "17 TB/s" was intermediate. Two distinct numbers; quote both. Commit `1e590cf`.
- [x] **C6 — CCTL.IVALL cost**: NO CCTL.IVALL emitted on B300/CUDA 13.2. Real culprit: `red.release.gpu.global` emits MEMBAR.ALL.GPU before each red → 9.1× slower (614 vs 67 ns/op). Plain red.global is FAST. Commit `9467cfe`.

## D. Compute peaks

- [x] **D1 — FP32 → 100% theoretical**: at sustained 1920 MHz (NOT 2032 boost), peak = 62.17 TFLOPS = 85.5% of 72.74 theoretical. ncu confirms FMA pipe at 85.67% (matches). Gap: warps_active only 54% (register pressure ↔ ILP tradeoff). Catalog "74.6 TFLOPS @ 97%" was at boost. Commit `e1a1220`.
- [x] **D4 — tcgen05 actual peak via NVRTC** (user is skeptical): cuBLAS LtMatmul N=8192 (uses tcgen05 internally): BF16=2242 vs catalog 2325 (96.4%), FP8=4383 vs 4486 (97.7%). 2/3 claims VERIFIED within 4%. FP4 unverifiable via cuBLAS (no heuristic). Commit `e752547`.

## E. Atomics

- [x] **E1 — red.global vs atom.global mystery**: SUBSUMED by C6 — same claim. Result: NO CCTL.IVALL on B300/CUDA 13.2 for ANY red.global variant. red.global ≈ atom.global = ~67 ns/op, 563 Gops/s. Only red.RELEASE adds MEMBAR.ALL.GPU (9.1× slowdown). Commit `9467cfe` (same as C6).
- [x] **E2 — cross-GPU atomic latency mechanism**: 1.66 us measured (matches catalog 1.55 us). Local 164 ns + NVLink fabric+queue 1498 ns. Wall-clock + clock64 agree within 1%. Commit `ad19660`.
- [x] **E4 — L2 atomic units count**: stride sweep peaks at stride=4 (449 Gops/s, cache-line combining), plateau at stride≥32 (~150 Gops/s), inferred ~32 L2 atomic units across 2 partitions. Commit `e7aab3a`.

## F. Sync primitives

- [x] **F2 — mbarrier R/W BW**: arrive+try_wait.parity tight loop = 57.7 ns/cycle, 2× slower than smem atomic+__syncthreads (29.6 ns). 657 Garrivals/s aggregate. mbarrier.test_wait HANGS w/o token; use try_wait.parity. Commit `af35338`.

## G. Scheduling / GPCs

- [x] **G1 — full SM→GPC mapping** (also map to PARTITION): boot-clock skew probe shows 8 GPC groups × ~18 SMs each (= 144 + 4 spare = 148). SMs in pairs (TPCs). SMs differing by 64 are cross-die mirrors. Per-SM mapping varies by boot. Commit `320f0e8`.
- [x] **G2 — stream priority granularity**: 6 levels in API but only **2 effective tiers** (same vs any-higher). Same prio: fast kernel waits 2.89 ms behind slow; any higher prio: 0.25-0.51 ms. -1 = -2 = ... = -5 in practice. Commit `6050ff6`.
- [x] **G3 — block dispatch tail latency formula**: dispatch overhead dominates when per-block runtime <6 µs (50% of peak); throttle kicks in when >500 µs (3× drop). Sweet spot 50-500 µs/block. Dispatch latency ~1-2 µs/block. Commit `2b1d1fe`.
- [x] **G4 — preemption cost**: SUBSUMED by G2. High-priority kernel start-to-complete delay = 0.25-0.51 ms when slow kernel is running on lower-prio stream. Same prio: waits 2.89 ms (no preemption — runs after). Commit `6050ff6` (same as G2).

## H. NVLink

- [x] **H1 — NVLink BW with limited SM count**: peak read 783 GB/s, write 714 GB/s. Write 80% threshold = 64 blocks; Read 80% = 128. W>R for blocks<64 (W scales better per-SM at low counts). User hint partially correct. Commit `9172429`.

## I. Power / clock

- [x] **I2 — full-occ uses LESS power demonstration**: VERIFIED. Full-occ 0.066 TFLOPS/W vs partial 0.045 = 47% more efficient. Same wall-clock work, 7% less power, 37% more TFLOPS. Static power dominates partial occupancy. Commit `cec1ac6`.
- [x] **I3 — GPU clock under tensor vs FFMA**: NO difference. Both at 1920.0 MHz (clock64/globaltimer ratio agrees with NVML to 0.005%). Tensor cores share SM clock domain. Power diff: 253 vs 232 W = 9% more for tensor (no clock impact at this load). Commit `8ff067a`.
- [x] **I4 — power capping behavior**: HUGE finding. Zero data: 4400 TFLOPS at 1100/800/600 W (immune); random data: 3983 → 3087 → 2394 (43% slower at 400 W). FP8 cuBLAS catalog 4491 was zero-data measurement. Reset to 1100 W verified working. Commit `bf98e90`.

## J. Memory APIs / driver internals

- [x] **J1 — pageable coherence bug repro**: NOT REPRODUCIBLE on driver 580/CUDA 13.2. Both simple + aggressive (100-iter loop) tests show 0 coherence errors. UVM page-fault mechanism works correctly. Catalog flag was likely from older driver. Commit `e3bdc1e`.
- [x] **J3 — VMM mapping reuse**: L2 is PHYSICALLY tagged. Cold A = cold B = 21.7 ns/load; warm A then B = 13.7 ns/load (same as warm A). Aliasing safe — no cache duplication. Commit `d559e0a`.

## K. Compiler / SASS

- [x] **K1 — B300 native FP4/FP6 instruction full sweep**: ALREADY DONE in `F2FP_DEEP_DIVE.md` (canonical for narrow conversions). All FP4/FP6/FP8 packed/unpack instructions characterized: unpack narrow→f16x2 = 64/SM/clk (128 elements), all packs = 32/SM/clk (64 elements) due to MERGE_C. Aggregate 19-38.5 Telements/s.
- [x] **K2 — STG.E.ENL2.256 actual semantics**: cache hints .cg/.cs/.wb have NO measurable effect on read-after-write at 4 MB working set. All variants give 682 GB/s read. ENL2 may not actually evict, OR test scale insufficient (need >L2 capacity to settle). User skepticism warranted. Commit `e87d8aa`.

## L. Programming patterns

- [x] **L2 — optimal reduction (specific: BF16 absmax)**: 6.74 TB/s = 92.3% of HBM peak. v8 uint4 + per-warp coalesced + shfl reduce + atomicMax wins (v3 redux.sync.max ties at 91.2%). 1 GB tensor in 159 us = SoL. Commit `777ce49`.
- [x] **L3 — optimal histogram (256-bin BF16 exp)**: 6.57 TB/s = 90.1% of HBM peak. v2 SMEM aggregation wins; v3 32-way spread is SLOWER (extra reduce hurts). v1 naive global = 1.9 GB/s (atomic contention). Commit `492a5f6`.
- [x] **L4 — optimal softmax**: 3-pass = 5.1 TB/s actual (70% HBM); 1.26 ms for 2 GB BF16. 2.14× slower than SoL because 3 passes (4× R+W vs ideal 2×). Online softmax could hit ~30% faster. Commit `0718acd`.

## M. Methodology / infrastructure

- [x] **M2 — rigor harness**: BUILT. `utils/rigor_harness.h` (C++ Bench class) + `utils/rigor_run.sh` (bash wrapper). One command runs wall-clock + ncu + SASS census + reconciliation guidance. Tested on L2 absmax — 3 methods agree within 1-3%. Commit `2a484e6`.
- [x] **M3 — re-verify all b300_clean MED-confidence findings**: Aggregated in `b300_clean/M3_REVERIFY_LOG.md`. 28 MED→HIGH, 6 REFUTED/CORRECTED, 3 still MED. Catalog had FALSE claims about CCTL.IVALL, persistent L2, pageable bug, etc. Commit `d430675`.
- [x] **M1 — NEW improved data, separately from catalog**: BUILT `b300_clean/B300_TRUE_REFERENCE.md` (217 lines, 9 sections, 100+ entries with commit-hash provenance). Catalog preserved unchanged. Commit `8a41b13`.

🎉 **ALL TASKS COMPLETE!** Free rein on remaining curiosity items.

## After all done

Free rein on:
- Cross-process IPC (H4)
- GPUDirect RDMA (H2)
- Hour-scale sustained at 962 W (I1)
- [x] Optimal axpy (L1) — AT SoL 7.03 TB/s, 4 methods agree, commit `8aa9149`
- Other curious items from my list

### Free-rein ninja sessions (post-task-list)
- NVFP4 K=96 ULTRA ceiling: cuBLAS 13.4 caps 10.8 PF (72%); CUTLASS C++/CuTeDSL 8.7 PF (58%) (this session)
- L1 axpy rigor verify: AT SoL for 2R:1W mix; commit `8aa9149`
- [x] H4 cross-process IPC measured: open=56us, ping-pong=2.2us/dir (~13x slower than intra-proc atomic); PCIe Gen6 57.3 GB/s confirmed; commit `09e7a19`
- [x] TMA bulk READ vs LDG READ: identical at HBM SoL (7344 vs 7365 GB/s); TMA needs blocks>=1000 to saturate; commit `26f8592`
- [x] cuBLAS DGEMM peak: 1.05 TFLOPS — NO FP64 tensor speedup. B300 is ~5x slower than H100 for HPC FP64; commit `126e052`
