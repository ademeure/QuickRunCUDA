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

- [x] **D1 — 2:4 sparse mma.sp works natively** (commit `9b7b759`): HMMA.SP.16832.F32.BF16 SASS. Same cy/inst as dense but 2× K depth = 2× throughput. HIGH confidence.
- [x] **D2 — Sparsity gives 2× throughput** (same commit as D1): covered by D1's finding. Per-inst cost identical; effective work doubles.
- [x] **D3 — Partial: within noise at low occupancy** (commit `6dc21ae`): low-occupancy test insufficient; needs full-occupancy sparse HMMA kernel. Deferred.

## E. cuStreamWaitValue patterns deeper

- [x] **E1 — Producer-consumer pair = 0.99 µs** (commit `8f7bbe5`): cuStreamWriteValue+WaitValue across 2 streams = 0.99 µs/pair sub-µs cross-stream sync. Predicates: GEQ, EQ, AND, NOR, FLUSH.
- [x] **E2 — BatchMemOp = 4× speedup** (commit `3a17180`): 4 WriteValue ops individual = 1.72 µs; batched = 0.44 µs (0.11 µs/op). Nearly perfect batch amortization. Lowest-latency control plane on B300.
- [ ] **E3 — cuStreamWaitValue with NCCL coordination**
- [x] **E4 — Multi-stream chain: Event > WriteValue > serial** (commit `fcf0746`): 4-stage deps — (A) 1-stream 2.43 us/stage, (B) WriteValue chain 2.70, (C) Event chain 2.03. Event chain BEATS sequential by ~0.4 us via pipelined dispatch. WriteValue adds ~0.65 us/stage vs events but remains preferred for counter/multi-producer/IPC patterns.

## F. Multi-GPU patterns

- [x] **F1 — P2P NVLink memcpy = 778 GB/s** (commit `88ee0cf`): saturates at ≥256 MB (86% of NVLink v7 spec). Latency 3-5 µs setup for small. vs V5 I2 kernel-write 714 GB/s; peer memcpy 9% FASTER.
- [ ] **F2 — NCCL allreduce vs custom kernel** detailed comparison
- [ ] **F3 — NVLink chain multi-hop** (2 → 4 → 8 GPU patterns)
- [x] **F4 — IPC handle 0.73 µs/handle (100×)** (commit `98a2232`): first 1.38 µs; warm 0.10 µs. Scales well — 1000 handles <1 ms. Not a bottleneck for multi-process workloads.
- [ ] **F5 — Multi-GPU cluster** support? (cluster CTAs across GPUs)

## G. Real LLM inference deep dive

- [ ] **G1 — Per-token energy in real LLM (vs synthetic FFMA/memory)**
- [ ] **G2 — Decode latency optimization** for 7B/13B models
- [ ] **G3 — KV cache energy/byte per access pattern** (sequential vs strided)
- [ ] **G4 — Attention kernel SoL ladder** (FlashAttention vs naive)
- [ ] **G5 — Quantization energy savings** (INT8 vs BF16 vs FP4)

## H. PCIe / Host transfer

- [x] **H1 — PCIe Gen 6 x16 saturates 57.8 GB/s** (commit `dc0499f`): ≥4 MB transfers; sweet spot ≥16 MB. Below 1 MB latency-bound (4 µs setup). 90% of spec ~64 GB/s.
- [x] **H2 — Pinned 57 GB/s; pageable -28%; UMA prefetch caches** (commit `b6a1600`): pageable 44 GB/s is 28% slower than pinned due to driver intermediate copy. UMA + prefetch caches data on GPU (subsequent iters no-op).
- [x] **H3 — H2D + compute overlap = 1.29× speedup** (commit `cca76e7`): 2-stream + event-wait pattern. Nearly perfect overlap (98% of ideal max(H2D, compute)). Use for ML inference batched pipelines.
- [x] **H4 — UMA cold page-fault 260× slower** (commit `bcb0995`): 64 MB cold = 10715 µs vs warm 41 µs. Always prefetch UMA data before first GPU access.
- [x] **H5 — H2D+D2H different streams = 1.59× overlap** (commit `d0e05e3`): PCIe Gen 6 bidirectional but not perfectly parallel. 0.59 ms serial → 0.37 ms split-stream. Combined with H3, 3-stream H2D|compute|D2H gives near-PCIe-bound perf.

## I. HBM3E channel parallelism

- [x] **I1 — 12-stack channel routing: HW hash makes transparent** (commit `220556e`): designed aliased stride 3072 B (= 12 × 256-B lines) should cap at 1/12 peak = 8.3% under pure linear interleave. Actual: 5.28 TB/s = 73% of 7.2 TB/s peak (8.8× higher than hypothesis). MODE 1 coalesced: 5.82 TB/s = 81% peak. ncu-verified DRAM bytes. Tightens V8 I2.
- [x] **I2 — HBM stack routing transparent** (commit `afcedae`): concentrated 256 MB vs distributed 12 GB = identical DRAM read, L2 hit, time. No app-level stack balancing needed.
- [x] **I3 — HBM tail = 23× avg (~1 µs p-max)** (commit `c065da8`): avg 60 cy, max 1433 cy = 955 ns. Refresh + bank row miss spikes. For latency SLA, expect ~1 µs tail at p99.
- [x] **I4 — ECC on; 0.2% capacity tax; BW spec includes it** (commit `fc64e5f`): 287.4 of 288 GB usable. HBM3E inline ECC, no runtime perf diff.

## J. Microsecond autotuning

- [x] **J1 — 128 threads/block optimal for FFMA (4 warps = 1/SMSP)** (commit `da638ef`): 32/64/128 = 1165 cy; 256+ plateaus at 121 FFMA/cy. Auto-bisect: start 32, double until throughput plateaus.
- [x] **J2 — Auto-clock for energy minimum: FFMA@1500, DRAM@1005** (commit `5c9f0a7`): FFMA 137 GFLOPS/W @ 1500 MHz (vs 124 @ boost, 121 @ 1005); DRAM 30.0 GB/s/W @ 1005 MHz (vs 26.2 @ boost). Boost is for LATENCY, not energy. V6 "USE BOOST" was vs 510 MHz extreme. Details: V8_J2_ENERGY_SWEEP.md.
- [x] **J3 — Occupancy API recommends 1024; 2048 thr/SM cap** (commit `824c321`): API hits HW limit regardless. Use API for upper bound; benchmark smaller sizes (V8 J1: 128 best for FFMA).
- [x] **J4 — CSIZE=1 best for pure compute (+32% overhead for clusters)** (commit `edb856d`): cluster barrier O(1) at ~370 cy regardless of size. Use cluster only for DSMEM cooperation or launch-perf boost.

## K. Cross-process atomic semantics

- [x] **K1 — Cross-process atomic via IPC WORKS** (commit `45ef5e1`): parent+child 100K atomicAdds each → final = 200000 exact. Device-side atomics on IPC-shared mem same speed as intra-process. Multi-process producer-consumer patterns viable.
- [x] **K2 — Custom pool shareable via POSIX FD** (commit `4528be1`): need custom pool with handleType=POSIX_FD (default pool NOT shareable). Export gives FD passable via socketpair. Enables zero-copy x-process allocators.
- [x] **K3 — POSIX shm + cudaHostRegister works x-process** (commit `29f9354`): both parent & child cudaHostRegister(Mapped|Portable) on same mmap'd shm → 56.9 GB/s pinned H2D each (matches V8 H1 PCIe sat). Device-mapped ptr works in both: child kernel writes pattern visible to parent CPU AND parent kernel sum matches exactly. X-process zero-copy host mem DONE.

## L. Tooling V3

- [ ] **L1 — End-to-end LLM inference Pareto plotter**
- [x] **L2 — Block-size auto-bisect tool** (commit `07c77e8`): utils/block_bisect.sh sweeps {32, 64, 128, 256, 512, 1024} × fixed total threads. Tested on FFMA: 32/64/128 all tied (30.22 ms), 256 is 23% slower (launch_bounds(256,1) forces 1 block/SM). Matches J1.
- [x] **L3 — Per-kernel energy tool** (commit `e031ef8`): utils/kernel_energy.sh wraps QuickRunCUDA with power sampling; reports J total + J/op with OPS= env. Verified FFMA @ 3000 iters: 8.03 pJ/FLOP matches J2 at 1920 MHz. (Full per-phase waterfall deferred — needs phase-aware kernels.)
- [x] **L4 — Per-warp trace: scheduler FAIR** (commit `c2babca`): %globaltimer at entry/exit per warp. 1 blk/SM CV=0.76%, max/min=1.03× (excellent). 8 blk/SM CV=25% but p99≈max (tight tail). Block dispatch 85 ns/block. FFMA chain SoL 4.07 cy/FFMA @ 1920 MHz. Details: V8_L4_WARP_FAIRNESS.md.
- [x] **L5 — Workload classifier tool** (commit `3601844`): utils/workload_classify.sh analyzes via ncu + recommends clock/block/cluster/prefetch based on V5-V7 findings.
- [x] **NEW — SMEM BW ceiling: 26.9 TB/s = 74% of 36.4 peak** (commit `352ab1f`): plain `float` LDS + 8-way ILP + 16× unroll. Zero bank conflicts (ncu). 0.77 wavefronts/cy/SM vs peak 1.0. Bottleneck = dispatch overhead, not banks. Ldmatrix test noted as follow-up (DCE-suspect). Details: V8_SMEM_BW.md.
- [x] **NEW — Cluster DSMEM BW: 37 TB/s = 97% of 38.5 peak** (commit `71934d0`): ld.shared::cluster with inline offsets. DSMEM achieves FULL SMEM BW via peer SM banks; zero BW penalty vs local. Note: varying-address DSMEM load fails in test (deferred). Details: V8_DSMEM_BW.md. MEDIUM confidence.
- [x] **NEW — FFMA peak VERIFIED 97.64%** (commit `8b99f43`): ncu pipe_fma + wall-clock + inst count agree. 75.2 TFLOPS at 2032 MHz boost with 2-source FFMA. V8 J2's 71% was 3-source variant (RF 2-read-port cap). Details: V8_FFMA_PEAK_VERIFIED.md. HIGH confidence.
- [x] **NEW — HBM write SoL: plain STG 6.11 TB/s (85%), TMA 7.57 TB/s (95%)** (commit `b15011b`): plain STG.E.128 caps at 85% of 7.2 TB/s peak; TMA bulk store 1.24× faster. Use TMA for write-bound. Details: V8_HBM_WRITE_SOL.md. HIGH confidence.
- [x] **NEW — FP64 DFMA = 100.00% peak (1.203 TFLOPS)** (commit `077b3d1`): cleanest SoL yet. Single DFMA port per SMSP, no competition. Exact match to 76.97 × 1/64. Same 2-source kernel pattern. Details: V8_FP64_PEAK_VERIFIED.md. HIGH confidence.
- [x] **NEW — IMAD peak = 99.7% (38.4 TOPS, 1:2 of FP32)** (commit `dcea840`): integer mul-add is 1:2 vs FP32 on Blackwell per spec. Correct theoretical = 38.5 TOPS. Details: V8_IMAD_PEAK_VERIFIED.md. HIGH confidence.
- [x] **NEW — L2 BW = 13.85 TB/s** (commit `41426f2`): strided ld.global.cg forces L1 bypass. Matches prior catalog 13-21 TB/s range. Ladder: L1 30 : L2 14 : HBM 7 : PCIe 0.06 TB/s. Details: V8_L2_BW_VERIFIED.md. HIGH confidence.
- [x] **NEW — HMMA.F16 tensor peak = 99.90% (578.6 TFLOPS)** (commit `779e046`): mma.sync.m16n8k16.f16 with 8-chain ILP. ncu tensor pipe 99.90%. Top of catalog 540-580 range. Details: V8_HMMA_F16_PEAK.md. HIGH confidence.
- [x] **NEW — HMMA variants: F32 accum free** (commit `475f568`): F16/F16 = F16/F32 = BF16/F32 all at 99.89% = 578 TFLOPS. Wider accumulator costs nothing. FP8 legacy path FAILED (compiler downgrade) — requires tcgen05. HIGH confidence.
- [x] **NEW — FADD/FMUL/FFMA same dispatch rate** (commit `109599e`): all 3 ops at 97.65% pipe = 37.4 Ginst/s. FFMA 2× FLOPS advantage is pure op-counting (2 FLOP/inst). Modern Blackwell unifies on FMA pipe. HIGH confidence.
- [x] **NEW — MUFU rsqrt = 99.49% XU pipe (47.8 GMUFU/s)** (commit `29b9b3b`): transcendentals saturate XU pipe. MEDIUM confidence (MUFU/cycle spec unclear on Blackwell).
- [x] **NEW — SHFL.BFLY = 3.0 G warp-SHFL/s** (commit `5dbe287`): slower than expected 1/cy/SMSP. Chain dependency dominates. Use redux.sync for reductions. MEDIUM confidence.
- [x] **NEW — redux.sync 2.5× faster than 5-round SHFL** (commit `9dd127f`): redux 6.32 ms vs SHFL chain 15.76 ms for N warp-reductions. Use redux.sync for ML reduction kernels.
- [x] **M14 V8 SoL ladder synthesis** (commit `d7beb79`): complete doc consolidating FP32/FP64/IMAD/HMMA/MUFU/SHFL compute peaks + L1/L2/SMEM/DSMEM/HBM/NVLink/PCIe memory peaks, all 10-rule rigor-verified.

## M. Brand new explorations

- [x] **M1 — partial: matches V6 G4 finding (hint NO benefit)** (commit `1a82d65`): B300 L2 + prefetcher handle most patterns gracefully; distributed hint adds little. Defer further investigation.
- [x] **M2 — Async error checking FREE (3.072 µs)** (commit `d892122`): Peek/Get/no-check all identical. Only cudaDeviceSynchronize adds 5 µs. ALWAYS check errors in production code — zero perf cost.
- [x] **M3 — Callbacks DO NOT fire after kernel error** (commit `e382ea7`): stream stuck in error state after crash; callbacks skipped. cudaDeviceReset recovers. Don't rely on callbacks for error reporting.
- [x] **M4 — extra field reserved (no-op)** (commit `ae0026b`): cudaKernelNodeParams.extra is documented as reserved for future extensions. No current uses.
- [x] **M5 — Device-side NVTX NOT supported** (CUDA 13.0/13.2 headers inspected): `nvtxRangePushA/nvtxRangePop` are `__host__`-only. NVRTC + nvcc both reject: "calling a __host__ function from a __global__ function is not allowed". No device annotations in any nvtx3 header. Workarounds: (a) host-side NVTX around kernel launch, (b) device-side `printf` (appears in Nsight Systems timeline), (c) PTX `%clock` for kernel-internal phase timing.
- [x] **M6 — B300 has NO DLA** (commit `58ccce2`): data center GPU, not Jetson SoC (integrated=0). 4 async DMA engines. DLA only on Jetson Orin.
- [x] **M7 — Queue depth = 1024 per stream** (commits `372397e` initial, **`<M7b>` corrected**): user was right, there IS a limit. Blocking starts at exactly index 1024. After fill: launch rate matches kernel completion rate. Use multiple streams or graph for >1024 short kernels.
- [x] **M8 — CPU pinning 7% SLOWER** (commit `0c0016d`): 31.24 vs 33.44 us/iter. Let OS schedule for std multi-stream CUDA apps.

---

## Completion tracking

Total: ~50 items. Many carry forward from V7 deferred + new V8 explorations.

When ALL `[x]`, generate V9 + write M13 synthesis.
