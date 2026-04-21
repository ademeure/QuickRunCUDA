# V7 Curiosity List (B300 sm_103a) — generated 2026-04-21

After V6 60/65 [x] (M10 synthesis). V7 carries forward V6 deferred items
+ new questions raised during V6 work + brand-new investigation areas.

Apply 10-rule rigor protocol. Mark `[x]` with commit hash when done.

---

## A. tcgen05.mma full implementation (carried from V6 B-series)

- [ ] **A1 — UmmaDesc raw PTX encoder** mirroring cute::SmemDescriptor for SM_103
- [ ] **A2 — Working tcgen05.mma m64n8k16 BF16** with non-zero output verification
- [ ] **A3 — tcgen05.mma + HMMA concurrency** (closes V5 B1)
- [ ] **A4 — tcgen05.mma idesc bit format** documentation (5-arg + 4-tuple format)
- [ ] **A5 — tcgen05.mma latency/throughput** vs HMMA mma.sync at same precision
- [ ] **A6 — tcgen05.mma power signature** (does it match HMMA pJ/op?)
- [ ] **A7 — tcgen05.mma kind=tf32/f8/f16/sparse** all variants

## B. Multicast TMA (carried from V6 I5)

- [ ] **B1 — Working multicast cp.async.bulk.shared::cluster.multicast** with DSMEM mbarrier
- [ ] **B2 — Multicast BW vs unicast** — measure savings for N CTAs receiving same data
- [ ] **B3 — TMA tensor descriptor** (cuTensorMapEncodeTiled-built) for 2D tile copies
- [ ] **B4 — TMA prefetch combo** — does prefetch help TMA like it helps cp.async (V6 I3 1.58×)?

## C. NVFP4 conversion + scale variants (carried from V6 H3)

- [ ] **C1 — cvt.scalefactor variant** for NVFP4 e2m1 with per-block scale
- [ ] **C2 — NVFP4 dequant cost** per scale group (8 elements)
- [ ] **C3 — NVFP4 + tcgen05.mma** end-to-end (depends on A2)

## D. Cluster patterns deeper

- [ ] **D1 — Cluster-wide all-reduce primitive** via DSMEM + cluster.barrier
- [ ] **D2 — Cluster broadcast pattern** via single-CTA-write + cluster.barrier
- [ ] **D3 — Cluster.barrier vs cluster.barrier.aligned** — perf difference?
- [ ] **D4 — Async cluster.barrier** (arrive only, wait separately)
- [ ] **D5 — Cluster-wide atomic vs per-CTA atomic** (partial overlap with V5 C3)

## E. CUDA Graph advanced

- [ ] **E1 — cudaGraphInstantiate flags** (cudaGraphInstantiateFlagAutoFreeOnLaunch, etc.)
- [ ] **E2 — Graph node priority** via cudaGraphAddKernelNode kernelParams.kernelExtra
- [ ] **E3 — Graph debugging** via cudaGraphDebugDotPrint
- [ ] **E4 — Graph clone + replay** (multi-execution from same graph)
- [ ] **E5 — Conditional graph WHILE loop** (CU_GRAPH_COND_TYPE_WHILE) — does it work?
- [ ] **E6 — Conditional graph SWITCH** (CU_GRAPH_COND_TYPE_SWITCH)

## F. Streams + concurrency advanced

- [ ] **F1 — Stream callback chain** (cudaStreamAddCallback → cudaLaunchHostFunc cascade)
- [ ] **F2 — Stream attribute SyncPolicy** (cudaSyncPolicyAuto/Spin/Yield/BlockingSync)
- [ ] **F3 — cudaMemcpyAsync between streams** (peer-to-peer)
- [ ] **F4 — Stream merge/split patterns**
- [ ] **F5 — Stream with cudaLaunchHostFunc** ordering (V6 F5 baseline = 2.46 us)

## G. Memory + L2 deeper

- [ ] **G1 — L2 prefetch.L2 hit ratio** ncu confirmation (V6 I3 indirect evidence)
- [ ] **G2 — L2 hit_rate per access pattern** (sequential vs strided vs random)
- [ ] **G3 — L1 occupancy effect** on L1 hit rate
- [ ] **G4 — DRAM bank conflicts** measurement on B300 12-stack HBM3E
- [ ] **G5 — Cache line replacement policy** detection (LRU vs other)
- [ ] **G6 — TLB / address translation cost** for huge buffers (>L2)

## H. Numeric format conversion deeper

- [ ] **H1 — cvt.rn vs cvt.rz vs cvt.rm vs cvt.rp** rounding mode latency (already known FREE per V5 J1 but verify for narrow FP)
- [ ] **H2 — cvt.satfinite vs cvt unsaturated** cost diff
- [ ] **H3 — cvt with FP exception handling** (overflow/inf/NaN)
- [ ] **H4 — Packed vs scalar cvt** (e2m1x4 vs 4× e2m1) — already partial in V6 H3
- [ ] **H5 — TF32 cvt precise format** (10-bit mantissa via cvt.rna.tf32.f32)

## I. Async copy deeper

- [ ] **I1 — cp.async.commit_group depth** (1, 4, 16 groups in flight)
- [ ] **I2 — cp.async.wait_group N** vs wait_all
- [ ] **I3 — cp.async.bulk vs cp.async** for varying sizes (16 B → 1 KB → 16 KB)
- [ ] **I4 — Pipelined async copy** with double-buffering pattern
- [ ] **I5 — cp.async.bulk + mbarrier expect_tx** detailed perf

## J. Persistent kernel patterns deeper

- [ ] **J1 — Multi-task batching in persistent kernel** (process N tasks per signal)
- [ ] **J2 — Persistent kernel + work stealing** (multi-block coordination)
- [ ] **J3 — Persistent kernel + cudaGraphLaunch** combination
- [ ] **J4 — Persistent kernel power profile** — sustained vs intermittent
- [ ] **J5 — Cross-process persistent kernel** via IPC handle (extends V5 I1)

## K. Power / thermal advanced

- [ ] **K1 — Sustained throttle threshold** more precisely (V5 D3: ≥800 W)
- [ ] **K2 — Per-pipe energy consumption** (FFMA vs LDG vs HMMA J/op)
- [ ] **K3 — Memory traffic vs energy** scaling
- [ ] **K4 — Sleep modes** for unused SMs (does HW power-gate?)
- [ ] **K5 — Workload mix transitions** — does workload change cause re-clock?

## L. Tooling V3

- [ ] **L1 — Auto kernel Roofline classifier** (extend V6 L5)
- [ ] **L2 — Per-warp instruction trace** (proxy for nsight without nsight)
- [ ] **L3 — Kernel power waterfall** (per-section breakdown)
- [ ] **L4 — End-to-end LLM inference Pareto plotter**
- [ ] **L5 — Auto-bisect: find optimal blocksize/clock per kernel**

## M. Brand new explorations

- [ ] **M1 — PDL (Programmatic Dependent Launch)** kernel-to-kernel coordination
- [ ] **M2 — Dynamic parallelism on B300 + tcgen05** (mix DP with TMEM)
- [ ] **M3 — Driver API vs Runtime API** overhead (cuLaunchKernel vs cudaLaunchKernel)
- [ ] **M4 — Compute capability detection** via cudaDeviceProp deep dive
- [ ] **M5 — Memory pool allocators** detailed perf (V5 H4 baseline)
- [ ] **M6 — CUDA Stream Memory Operations** (cuStreamWriteValue, cuStreamWaitValue)
- [ ] **M7 — GPU clock alignment** (globaltimer vs CPU clock for accurate TTF)
- [ ] **M8 — HW Decoupled Look-Ahead** (DLA) in scheduler — does it exist on B300?

---

## Completion tracking

Total: ~50 items. Many are V6 deferred (A, B, C series) + new questions (D, E, F, etc).

When ALL `[x]`, generate V8 + write M11 synthesis.
