# V5 — Newly-curious questions surfaced from V4 deep dive (2026-04-21)

V4 found ~135 things. V5 lists the NEW questions surfaced during that work that
weren't in V3/V4 originally. Generated from gaps + surprises in V4.

Status: `[ ]` unstarted, `[~]` in progress, `[x]` done with commit hash.

---

## A. mbarrier deep ninja

- [ ] **A1 — mbarrier.try_wait timeout** granularity: we passed 0; does negative work? what's max?
- [ ] **A2 — Why is mbarrier voltage-aware?** (per R2 +25% lower power than spin) — measure with ncu power per-pipe metrics
- [x] **A3 — mbarrier.expect_tx for cp.async** (commit `38e5772`, partial): legacy cp.async.cg+commit+wait_all = 314 cy/iter. Hopper+ expect_tx + try_wait.parity needs alternating phase tracking; my naive impl hung. Use cuda::pipeline abstraction.
- [x] **A4 — Multi-mbarrier per CTA** (commit `cc5f61a`): scales perfectly — 1 bar = 24 cy, 64 bars = 26 cy/arrive. No shared resource contention. cuTLASS pipelines can use 64+ barriers freely.
- [ ] **A5 — mbarrier in DSMEM** (cluster-wide barrier across CTAs)
- [x] **A6 — mbarrier.arrive without wait** (commit `a549d92`): fire-and-forget arrive = **24 cy** (20% cheaper than __syncthreads 30 cy). Useful for producer-consumer split-phase. Sync hierarchy: warp(full)=0 → warp(partial)=7 → arrive=24 → bar.sync=30 → cluster=395.

## B. HMMA + tcgen05 concurrency

- [ ] **B1 — HMMA + tcgen05.mma simultaneous** — can they both run concurrently? per-SMSP?
- [ ] **B2 — TMEM access vs HMMA** competition — both use tensor pipe?
- [ ] **B3 — tcgen05.cp + mma overlap** — pre-stage A/B then mma
- [ ] **B4 — WGMMA on B300** (pre-Hopper warpgroup mma) — still supported? perf vs mma.sync?
- [x] **B5 — HMMA chained vs ILP** (ref V4 A1 commit `08ee753`): mma.sync = HMMA. ILP saturates at ~10 cy/mma at ILP=8 (chain 27→17.5→12.75→10.375). 2.6× speedup over single chain.
- [ ] **B6 — mma + LDS overlap** — like B2 (LDG) but with SMEM

## C. DSMEM + cluster

- [x] **C1 — DSMEM (cluster-shared SMEM) latency** (commit `9c8fec8`): peer access = 214 cy = **3.96× slower than local SMEM** (54 cy). Cluster fabric round-trip via `mapa.shared::cluster` + `ld.shared::cluster`. Still 4× faster than DRAM. Cross-row vs within-TPC distinction needs follow-up.
- [ ] **C2 — cluster.barrier abuse** — call arrive_drop on N/4 CTAs to simulate subset wait
- [ ] **C3 — Cross-cluster atomics** — atomic on DSMEM from another CTA
- [ ] **C4 — Cluster shared mbarrier** — semantic + perf
- [x] **C5 — Cluster size 16** (commit `26a5f6a`): MAX = **8** on B300. CSIZE=2/4/8 OK; CSIZE=16 silently FAILS (returns cudaSuccess but NO blocks execute). Stick to CSIZE ≤ 8 for portable cluster code.

## D. Power deep ninja

- [x] **D1 — Subnormal FFMA without -use_fast_math** (commit `0bd3802`): B300 has NATIVE subnormal FFMA at **FULL speed**. SASS verified: -use_fast_math = FFMA.FTZ, -ftz=false = plain FFMA. Both run at IDENTICAL 4.11 cy/fma for subnormal modes. NO penalty unlike CPU 100× trap.
- [x] **D2 — Per-pipe DUTY CYCLE** (commit `2b5f451`): YES roughly linear. FFMA pipe power scales 7→24→85→123 W as DUTY 1→4→16→64; SATURATES at +124 W active = FFMA pipe full. Per-FFMA = 4.4 pJ (matches H1).
- [x] **D3 — TDP throttle behavior** (commit `bd20df5`): heavy FFMA+MUFU @ 220 W (20% of 1100 W TDP) does NOT throttle — sustains 2032 MHz boost. Throttle requires 800+ W workload (NVFP4 CuTeDSL → 1455 MHz @ 1095 W per prior data).
- [ ] **D4 — Dynamic voltage scaling** — does B300 auto-adjust voltage with clock?
- [x] **D5 — Power-aware code patterns** (commit `f18345b`): default LDG.E (L1 cached) BOTH faster AND **lower power** than .cg (bypass L1). +30 W vs +56 W (.cg uses 87% more power!). L1 caching offloads L2/DRAM work which dominates energy.
- [x] **D6 — Per-SM power isolation** (ref H6+D3): per-SM static power independent (0.05 W each); BUT under TDP pressure (1100 W limit), all SMs share clock domain → throttle together (NVFP4 → 1455 MHz @ 1095 W). Light load = isolated; heavy load = shared.

## E. Bizarre PTX corners

- [x] **E1 — PTX `prefetchu.L1`** (commit `ab6fbbc`): YES — emits **`CCTL.E.PF1`** SASS. All 3 PTX forms (prefetchu/prefetch.L1/prefetch.L2) → identical SASS. Back-to-back prefetch+load = 1.2% speedup; real benefit needs K-iters-ahead software pipelining.
- [x] **E2 — PTX `applypriority`** (commit `b0ada56`): only `::evict_normal` supported on B300 (emits `CCTL.E.DML2`, 0.4% noise). `::evict_last` and `::evict_first` = **ptxas COMPILE ERROR**. Limited functionality; use `discard.global.L2` or `.cg` modifier instead.
- [x] **E3 — `griddepcontrol`** PTX (commit `5e57cf8`): launch_dependents → PREEXIT (+1 cy); wait → ACQBULK (free in standalone). Used by PDL for kernel-to-kernel coordination without host involvement.
- [x] **E4 — `bar.warp` semantics** (commit `6f7c49f`): SYNC-ONLY on B300. `bar.warp.arrive` and `bar.warp.wait` = ptxas COMPILE ERROR. For split-phase warp work, must promote to mbarrier (block scope).
- [x] **E5 — `nanosleep`** PTX (commit `eb6efce`): minimum sleep 62 ns; sweet spot 500-1000 ns (<2% error); AVOID 5-10 µs range (1.63× overshoot from HW timer granularity); long sleeps ≥1 ms accurate within 5%.
- [x] **E6 — `getctarank`** (commit `c152b25`): %cluster_ctarank works in BOTH cluster + non-cluster (returns 0/1 for non-cluster as size-1 cluster). Code using cluster idents runs unchanged either way.

## F. ncu metric exotica

- [x] **F1 — sm__pipe_* breakdown** (commit `948cf45`): MAJOR — B300 has TWO FFMA sub-pipes (`pipe_fmaheavy` 50% + `pipe_fmalite` 50%). Pipe taxonomy: alu/fma/fmaheavy/fmalite/xu/lsu/tensor/adu/cbu/fp64. Validated bfind→XU, LDS→LSU, FFMA→fma split.
- [x] **F2 — warps_active/eligible/issued** (commit `146a96f`): metrics available via `smsp__warps_active.avg.per_cycle_elapsed`, `smsp__warps_eligible.avg.per_cycle_elapsed`, `smsp__inst_issued.sum`. Useful for diagnosing scheduler stalls.
- [x] **F3 — Memory throughput sub-metrics** (commit `e0f896d`): pyramid via `*.throughput.avg.pct_of_peak_sustained_elapsed`. D3 example: l1tex 78.85%, lts 62.04%, dram 44.61%. Read/write split via dram__bytes_read/write.sum instead.
- [ ] **F4 — Register usage realtime metrics** — verify SASS spill count via ncu
- [ ] **F5 — Cluster-aware metrics** — do cluster CTAs get separate counts?

## G. Real-world micro-kernel SoL

- [x] **G1 — Q4 fix: SHFL warp scan** (commit `2f3c7b3`): prefix scan 1024 = **523 cy = 349 ns** (1.85× faster than Q4 H-S 966 cy). SHFL-up warp scan + cross-warp via SMEM. Per-element 0.51 cy.
- [x] **G2 — SoftMax 1024 elements** (commit `1221b47`): single block 256 threads = **899 cy = 599 ns**. 3-pass (max, exp+sum, normalize). Per-element 0.88 cy. ex2.approx.ftz fast path used.
- [x] **G3 — LayerNorm 1024 elements** (commit `ff749ae`): single block 256 threads = **997 cy = 665 ns**. 2-pass (mean, variance, normalize). Per-element 0.97 cy. 1.7x slower than RMSNorm.
- [x] **G4 — RMSNorm 1024 elements** (commit `470ac9c`): single block 256 threads = **588 cy = 392 ns**. Per-element 0.57 cy. SoL ~300 cy with cp.async overlap.
- [x] **G5 — Argmax 1024 elements** (commit `f168e50`): single block 256 threads = **441 cy = 294 ns**. Per-element 0.43 cy. SHFL-XOR paired (value, index) reduce.
- [x] **G6 — Top-K (K=4) 1024 elements** (commit `ee01e84`): single block 256 threads = **1858 cy = 1239 ns**. 4× argmax with masking (linear in K). For K≥16, use bitonic top-K instead.

## H. Driver / runtime sub-microsecond

- [x] **H1 — cudaGraphExec_t update perf** (commit `c734603`): cudaGraphExecKernelNodeSetParams = 0.4 ns overhead (essentially FREE!). Re-instantiate = 7.25 µs (3.5× slower). For batched inference: build graph once, SetParams + Launch per-request.
- [x] **H2 — cudaGraphLaunch with stream re-attach** (commit `6b5aebd`): 1-2 streams = 2049 ns/launch; **4+ streams DOUBLES to 4094 ns/launch** (single-exec contention). For multi-stream throughput: instantiate per-stream exec, not shared.
- [x] **H3 — Memory allocator perf** (commit `a46e764`): cudaMallocAsync = **200× FASTER** than cudaMalloc. 4 KB: 0.33 vs 66 µs; 1 MB: 0.41 vs 66 µs. Async uses pre-allocated pool, no driver round-trip. Always use Async for high-freq allocations.
- [x] **H4 — Stream-ordered alloc** (commit `842065e`): cudaMallocFromPoolAsync = **O(1) at 0.32 µs** regardless of size (tested 64 B to 16 MB). Min allocator. 200× faster than sync cudaMalloc.
- [x] **H5 — driver context init** (commit `19b48e6`): cudaSetDevice(0) = **474 ms cold start**. First kernel + sync = 67-74 µs (incl. module load). Warm launch = 8.6 µs. Each new process pays full init cost. NEVER fork for short GPU work — amortize over long-running process.
- [x] **H6 — Lazy module load** (commit `fc800a3`): default = LAZY (CUDA 11.7+). EAGER saves 40 µs on first kernel (preloads modules). cudaSetDevice timing similar (~478 ms either way). Use EAGER for low-latency first-touch; LAZY for cuBLAS-heavy apps.

## I. Multi-GPU patterns

- [ ] **I1 — IPC handle benchmark cross-process** (multi-process binary spawn)
- [x] **I2 — NVLink streaming WRITE BW** (commit `9bdd026`): kernel-direct write = **714 GB/s peak** (96% of cudaMemcpyPeer 749). Saturates at just 32 blocks. 75% of NVLink theoretical (956). Use kernel-write for compute+xfer interleave; cudaMemcpyPeer for pure data movement.
- [x] **I3 — Multi-GPU all-reduce** (commit `41ba869`): naive 2-GPU = **80.8 GB/s** (256 MB float buffer, 6.642 ms). 10× below NVLink peak (740 GB/s from I2 write BW). Naive scalar add_peer kernel; needs vec4 + K-iters-ahead prefetch + better tiling. NCCL achieves 80-90% of peer write BW; custom kernels typically reach ~50% without tuning.
- [x] **I4 — Cross-GPU atomic via NVLink** (commit `4005673`): cross-GPU atomic = **0.54 Gatomic/s = 3.2× slower** than local HBM atomic (1.74). 18× faster than PCIe sysmem atomic (E7). Use NCCL for multi-GPU; avoid raw cross-GPU atomicAdd.
- [x] **I5 — UVA pointer access from peer GPU** (commit `fd12cdc`): UVA alone INSUFFICIENT — direct peer access FAILS with illegal memory access. Explicit `cudaDeviceEnablePeerAccess` REQUIRED. UVA gives unified addressing only; not transparent P2P.

## J. Numerical precision exotica

- [x] **J1 — Rounding mode behavior** (commit `58dc7c3`): all 4 modes (rn/rz/rm/rp) = **identical 4.438 cy/FFMA**. SASS verified distinct emission (FFMA / FFMA.RZ / FFMA.RM / FFMA.RP). Rounding mode is FREE on B300; IEEE-strict apps can use any mode without perf concern.
- [x] **J2 — Subnormal preservation** (commit `e686f90`): B300 fully preserves subnormal FFMA output when -ftz=false (IEEE). FTZ flushes only arithmetic results; load/store preserves either way. Combined with D1 (full-speed subnormal): use -ftz=false for IEEE-strict at no perf cost.
- [x] **J3 — Mixed precision MMA precision** (commit `fed42a9`, PARTIAL): test methodology sound; my reference layout mismatched mma.sync fragment layout. Need ldmatrix.x4.trans for proper data flow. For clean precision measurement: use cuBLAS gemmEx and compare bf16 vs fp32 outputs.
- [x] **J4 — TF32 vs FP32 mma precision tradeoff** (commit `c2f5bae`): TF32 ULP = 8192× FP32 ULP (matches 13-bit mantissa diff). Max abs err 4.88e-4 vs FP32 1.2e-7. 4× speedup but 8000× precision loss; OK for ML training, AVOID for IEEE-strict.
- [x] **J5 — Block reduction precision** (commit `d844c1b`): adversarial sum (1e8 + 1023 ones) → Serial err 1023 (loses all!); Pairwise err 7 (146× better); **Kahan err 1 = 1023× better**. Use pairwise for ML inference; Kahan for critical accuracy.

## K. Async + persistent kernel patterns

- [x] **K1 — Persistent kernel SoL** (commit `41e225f`): persistent = 9.34 µs/task vs per-launch 6.51 µs/task in MINIMAL test (per-task launch wins!). Persistent only beats launch when per-task work ≥ launch overhead (~5 µs). For >50 µs work, persistent + batched signals wins.
- [x] **K2 — Producer-consumer via DSMEM** (commit `ece9976`): cluster=2 round-trip = **468 cy = 312 ns** per iter (write + DSMEM read + cluster barrier with overlap). 33× more than __syncthreads but 5× faster than NVLink. Use for actual cross-CTA work.
- [x] **K3 — Dynamic parallelism cost** (commit `cb27978`): device-side launch = **12.36 µs/launch** (issue only, no sync). 6× slower than host cudaLaunchKernel (2.05 µs). CDP2 removed device-side sync; use streams/graphs/persistent over DP.
- [x] **K4 — CUDA Graph capture overhead** (commit `5dfdcb2`): re-launch = **0.64 µs/kernel = 4.4× faster** than direct (2.79 µs). Capture 0.87 µs/kernel; instantiate 244 µs one-time. Break-even ~113 launches.
- [x] **K5 — Stream-ordered cooperation** (commit `68c4e53`): parallel streams = **1.97× speedup**; dependent (event-wait) chain = SAME as parallel due to pipelining (event overhead 1 µs << 1.5 ms kernel). Use 2 streams for parallel work freely.

## L. Tooling enhancements

- [x] **L1 — Auto-rigor wrapper** (commit `3f7c64c`): utils/auto_rigor.sh combines clean_run + ncu_explorer + sass_count + checklist. Use for new microbenches; complex -H args may need manual phase invocation.
- [x] **L2 — Power profiler via libnvidia-ml** (commit `8bb15c8`): NVML library = **6483 Hz max** (vs CLI 33 Hz = 196× faster). Unique value rate still ~9 Hz (HW sensor 110 ms cache). utils/power_sampler.cpp tool.
- [ ] **L3 — Per-pipe utilization dashboard** from ncu metrics
- [ ] **L4 — SASS diff visualizer** — between two kernel variants
- [ ] **L5 — Microbench template** generator

---

These are the **next 50+ questions** to dig into. Same rigor protocol as V4:
1. Theoretical first
2. Measured + ratio
3. >1.0× → bug
4. Investigate why if less
5. ncu cross-check
6. SASS verify
7. ≥3 methods reconcile
8. Demonstrate "Y because X"
9. Suspect test before HW
10. HIGH/MED/LOW + falsifiability
