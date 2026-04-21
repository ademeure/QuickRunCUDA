# V8 Curiosity List (B300 sm_103a) — generated 2026-04-21

After V7 = 44/50 [x] (M12 synthesis). V8 carries forward V7 deferred items
(tcgen05, multicast, NVFP4) + new questions raised during V7 work + brand-new
explorations.

Apply 10-rule rigor protocol. Mark `[x]` with commit hash when done.

---

## A. tcgen05.mma full implementation (carried from V7 A-series)

- [ ] **A1 — UmmaDesc raw PTX encoder** mirroring cute::SmemDescriptor; verify against cuTLASS reference
- [ ] **A2 — Working tcgen05.mma m64n8k16 BF16** with non-zero output verification
- [ ] **A3 — tcgen05.mma + HMMA concurrency**
- [ ] **A4 — tcgen05.mma idesc bit packing reference** (5-arg + 4-tuple format)
- [ ] **A5 — tcgen05.mma latency vs HMMA** at same precision
- [ ] **A6 — tcgen05.mma power signature**
- [ ] **A7 — tcgen05.mma kind=tf32/f8/f16/sparse all variants**

## B. Multicast TMA (V7 B-series)

- [ ] **B1 — Working multicast cp.async.bulk.shared::cluster.multicast** with proper DSMEM mbarrier
- [ ] **B2 — Multicast BW vs unicast** (1 PCIe transfer to N CTAs)
- [ ] **B3 — TMA tensor descriptor** via cuTensorMapEncodeTiled
- [ ] **B4 — TMA prefetch combo** (does prefetch.L2 help TMA?)

## C. NVFP4 (V7 C-series)

- [ ] **C1 — cvt.scalefactor variant** for NVFP4 (per-block scale 8 elem)
- [ ] **C2 — NVFP4 dequant cost** per scale group
- [ ] **C3 — NVFP4 + tcgen05.mma end-to-end**

## D. Sparsity in tcgen05.mma

- [ ] **D1 — 2:4 sparse weight loading** + tcgen05.mma.sp variant
- [ ] **D2 — Sparsity speedup measurement** (vs dense)
- [ ] **D3 — Sparsity power signature**

## E. cuStreamWaitValue patterns deeper

- [ ] **E1 — cuStreamWaitValue for semaphore patterns** (multi-producer/consumer)
- [ ] **E2 — cuStreamBatchMemOp** (batched write/wait operations)
- [ ] **E3 — cuStreamWaitValue with NCCL coordination**
- [ ] **E4 — Multi-stream serialization via WriteValue/WaitValue chains**

## F. Multi-GPU patterns

- [ ] **F1 — cudaMemcpyAsync peer-to-peer** between streams (V7 F3 carry)
- [ ] **F2 — NCCL allreduce vs custom kernel** detailed comparison
- [ ] **F3 — NVLink chain multi-hop** (2 → 4 → 8 GPU patterns)
- [ ] **F4 — Cross-GPU IPC handle perf scaling** (V5 I1 detailed)
- [ ] **F5 — Multi-GPU cluster** support? (cluster CTAs across GPUs)

## G. Real LLM inference deep dive

- [ ] **G1 — Per-token energy in real LLM (vs synthetic FFMA/memory)**
- [ ] **G2 — Decode latency optimization** for 7B/13B models
- [ ] **G3 — KV cache energy/byte per access pattern** (sequential vs strided)
- [ ] **G4 — Attention kernel SoL ladder** (FlashAttention vs naive)
- [ ] **G5 — Quantization energy savings** (INT8 vs BF16 vs FP4)

## H. PCIe / Host transfer

- [ ] **H1 — PCIe Gen 6 x16 optimal transfer size** (1 KB → 1 GB sweep)
- [ ] **H2 — Host pinned vs unified memory** transfer overhead
- [ ] **H3 — Async H2D + kernel overlap** patterns
- [ ] **H4 — UMA migration cost** (when does GPU page-fault?)
- [ ] **H5 — H2D + D2H in same stream** vs different streams

## I. HBM3E channel parallelism

- [ ] **I1 — 12-stack channel routing** (does scheduler distribute?)
- [ ] **I2 — Stack contention measurement** (concentrate vs distribute access)
- [ ] **I3 — DRAM refresh impact** on tail latency
- [ ] **I4 — HBM error correction overhead** (ECC cost)

## J. Microsecond autotuning

- [ ] **J1 — Auto-bisect optimal block size** per kernel
- [ ] **J2 — Auto-clock for energy minimum** (workload-specific)
- [ ] **J3 — Auto-occupancy tuning** via launch_bounds variants
- [ ] **J4 — Cluster size auto-pick** (1 vs 2 vs 4 vs 8)

## K. Cross-process atomic semantics

- [ ] **K1 — IPC handle + cross-process atom.* perf**
- [ ] **K2 — Shared memory pool across processes**
- [ ] **K3 — POSIX shm + cudaHostRegister** for x-process zero-copy

## L. Tooling V3

- [ ] **L1 — End-to-end LLM inference Pareto plotter**
- [ ] **L2 — Auto-bisect optimal block size** (extends J1)
- [ ] **L3 — Power waterfall** (per-section breakdown)
- [ ] **L4 — Per-warp instruction trace** (proxy for Nsight)
- [ ] **L5 — Workload classifier** (compute/memory/mixed → recommend clock)

## M. Brand new explorations

- [ ] **M1 — Distributed L2 cache hint** for cross-cluster reuse
- [ ] **M2 — Async error checking** vs sync (cudaPeekAtLastError perf)
- [ ] **M3 — Stream callback safety under errors**
- [ ] **M4 — Graph node attribute extras** (kernel_extra slot uses)
- [ ] **M5 — Nsight markers from kernel** (nvtxRangePushA in device code)
- [ ] **M6 — DLA (Deep Learning Accelerator)** check — does B300 have one?
- [ ] **M7 — Hardware queue depth** for kernel launches
- [ ] **M8 — Driver thread affinity** impact on multi-stream perf

---

## Completion tracking

Total: ~50 items. Many carry forward from V7 deferred + new V8 explorations.

When ALL `[x]`, generate V9 + write M13 synthesis.
