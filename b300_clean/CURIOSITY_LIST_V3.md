# B300 Curiosity List V3 — Exhaustive Research Inventory (2026-04-20)

Built from the ground up after auditing what's been done.

**Already covered extensively** (see `B300_TRUE_REFERENCE.md`, `TASK_LIST.md`, all
`b300_clean/*.md`, V2 list): HBM/L2/L1/SHMEM peaks, FFMA/FP64/all tcgen05
peaks, mma.sync, atomics, sync primitives, scheduling, NVLink P2P, popcount/
sparsity/bitstride memory power, tcgen05 A vs B asymmetry, K-axis binary,
DWORD-pattern threshold, B-positive sign saving, perf/W ladder, K=96 ULTRA
shape sweep, realistic LLM weight power.

This V3 list focuses on what's STILL unexplored. Pick any unstarted `[ ]`
item, run rigor protocol, mark `[x]` with commit hash.

---

## A. TMA (Tensor Memory Accelerator) deep dive

- [ ] **A1 — TMA bulk async vs cp.async vs LDG** at varying tile sizes
  (16×16, 32×32, 64×64, 128×128, 256×256). Measure latency-to-first-byte
  and full-tile completion. Scale with cluster size 1/2/4/8/16.
- [ ] **A2 — TMA multicast** to multiple CTAs in a cluster: how does cost
  scale with destination count? Latency overhead per added receiver.
- [ ] **A3 — TMA 3D / 4D / 5D tensors**: how do strides + dims affect throughput?
  Find the sweet-spot dim count for typical attention tile shapes.
- [ ] **A4 — TMA + tcgen05 fully overlapped**: build a kernel that pipelines
  TMA loads of next-tile while tcgen05 computes current. Measure how close
  to true SoL you get vs naive sequential.
- [ ] **A5 — TMA with non-power-of-2 dimensions**: 200x200, 192x192, 96x96
  tile shapes. Does TMA degrade vs 256x256?
- [ ] **A6 — TMA write (cp.reduce.async.bulk)**: throughput, async behavior,
  vs LDG-then-store baseline.
- [ ] **A7 — TMA store with cluster-shared destination**: distribute one TMA
  load result across cluster via DSMEM.

## B. CUDA Graphs deep dive

- [ ] **B1 — Conditional graph nodes** (CUDA 13.0+): overhead vs runtime if-else.
  Measure for both compute-conditional and memory-conditional cases.
- [ ] **B2 — Graph instantiate / update / launch latencies as function of
  graph size** (10, 100, 1K, 10K nodes). Find practical scaling limits.
- [ ] **B3 — Graph capture mode** vs explicit construction: any perf delta?
- [ ] **B4 — Cross-stream graph dependencies**: cost of stream-fork/join inside graph.
- [ ] **B5 — Graph + cuStreamWaitValue** for kernel-to-kernel sync without host.
- [ ] **B6 — Graph child node memory allocations**: cudaMallocAsync inside
  graph capture — does it work, what's the cost?

## C. Multi-GPU + collectives

- [ ] **C1 — NCCL allreduce throughput** on 2× B300 NVLink, ring vs tree,
  varying buffer sizes. Compare to raw NVLink P2P numbers.
- [ ] **C2 — NVLink Sharp** (in-network reduction): is it usable on this NV18
  topology? Latency / throughput vs CPU-mediated.
- [ ] **C3 — GPUDirect RDMA** path: CPU-to-GPU peer copy without staging.
- [ ] **C4 — GDRCopy small-message latency** for kernel signaling between GPUs.
- [ ] **C5 — NUMA awareness** of host pinned memory: locality affects which
  IB / NVLink path is used? Impact on multi-GPU async copy.
- [ ] **C6 — NCCL ncclSend/Recv pairs**: 2-GPU all-to-all simulation.
- [ ] **C7 — Multi-process GPU (MPS)**: kernel-launch latency, isolation.
- [ ] **C8 — MIG (Multi-Instance GPU)**: any partial enablement on B300?

## D. Persistent kernels + work-stealing

- [ ] **D1 — Persistent kernel with mapped CPU queue**: latency from CPU
  enqueue → first GPU thread sees task. (Earlier: 4 µs floor)
- [ ] **D2 — Inter-block work-stealing** via global atomic queue. Throughput
  cost per stolen task.
- [ ] **D3 — Persistent kernel for streaming inference**: warm vs cold path,
  no relaunch. Compare end-to-end with cudaGraph re-launch.
- [ ] **D4 — Persistent kernel + tcgen05**: can a single persistent kernel
  saturate FLOPS for indefinite duration?

## E. CUDA streams + scheduling

- [ ] **E1 — Stream priority effective tiers**: we know "2 tiers" — verify by
  ramping up many streams of various priorities under contention.
- [ ] **E2 — Stream queue depth limit**: how many launches can pile up before
  the host thread blocks?
- [ ] **E3 — Concurrent kernel slot 128 limit**: when exceeded, do slots queue
  fairly or starve? Measure tail latency under saturation.
- [ ] **E4 — Stream callbacks** (cudaLaunchHostFunc): host-thread overhead per call.
- [ ] **E5 — cuStreamWaitValue / WriteValue** for kernel-to-kernel sync without
  host: latency vs mbarrier vs cooperative-groups.
- [ ] **E6 — Stream capture overhead**: cost of recording vs running the same
  kernel directly.

## F. Compiler / SASS deeper

- [ ] **F1 — PTX optimization-hint pragmas** (e.g. `__builtin_assume`):
  measurable benefit on SASS quality?
- [ ] **F2 — `__noinline__` regression** (we know 14× slower) — what's the
  break-even for inline-vs-noinline by function size?
- [ ] **F3 — Register spill thresholds**: at what `__launch_bounds__(threads, blocks_per_SM)`
  does the compiler start spilling?
- [ ] **F4 — Branch divergence cost**: warp-divergent ifs vs warp-uniform.
  Measure perf cliff.
- [ ] **F5 — SASS immediate value width limits**: what triggers ULDC fetches
  vs immediate operand?
- [ ] **F6 — Cross-translation-unit inlining** with separate compilation:
  any LTO benefit?
- [ ] **F7 — Compiler treatment of `__forceinline__` vs `inline`**.
- [ ] **F8 — `#pragma unroll` cost**: how aggressively does compiler unroll?
  When does it stop helping?

## G. Numerical / arithmetic

- [ ] **G1 — DPX instructions** (dynamic programming): peak throughput, use cases.
- [ ] **G2 — Sparse tensor cores (2:4 sparsity)**: real throughput vs dense at
  same shape, A100/H100 had this — is it on B300?
- [ ] **G3 — bf16x2 vs bfloat16 packed FMA**: what does the compiler emit?
- [ ] **G4 — IMAD vs IMAD.SHL.U32**: throughput differences for various ops.
- [ ] **G5 — INT8 / INT4 tensor cores**: peak throughput, throughput by data.
- [ ] **G6 — FP8 E5M2 vs E4M3**: cy/MMA, power, accuracy tradeoffs.
- [ ] **G7 — MXFP4 (block_scale.block32)** [task #387]: get it working.
- [ ] **G8 — FP6 (E2M3, E3M2) tcgen05** kinds: do they exist? Throughput?
- [ ] **G9 — MUFU pipe interactions with FFMA**: known parallel issue, but
  what about MUFU + tcgen05 simultaneously?
- [ ] **G10 — IMUL.LO vs IMUL.HI** throughput.

## H. Memory subsystem

- [ ] **H1 — Constant memory** (read-only) BW: what's the peak? Latency?
- [ ] **H2 — Local memory** spill/restore overhead: build a kernel that
  intentionally spills, measure.
- [ ] **H3 — Texture / surface memory** on B300: is it deprecated, or does
  it have any niche advantage?
- [ ] **H4 — SHMEM bank conflict edge cases**: 16-bit / 8-bit accesses,
  strided patterns, broadcast.
- [ ] **H5 — Read-only data cache (`__ldg`)** vs L1 default: any difference?
- [ ] **H6 — Async memcpy with mbarrier completion**: latency and BW.
- [ ] **H7 — cudaMemcpyAsync H2D vs D2H asymmetry**.
- [ ] **H8 — Page-locked variants**: cudaHostAllocPortable / Mapped / WriteCombined.
- [ ] **H9 — Unified Memory prefetch hints**: cudaMemAdvise effectiveness.
- [ ] **H10 — DRAM refresh visibility**: can you observe the refresh cycle?

## I. Power / thermal deeper

- [ ] **I1 — Hour-scale sustained power**: does TDP cap drift over 1 hour?
  What about clock state at hour 5? Thermal soak behavior.
- [ ] **I2 — Power-state transitions**: idle → busy → idle latency. Boost
  ramp-up time. Wake-up cost.
- [ ] **I3 — Per-rail power breakdown**: if NVML exposes core / SRAM / HBM
  / NVLink rails separately.
- [ ] **I4 — GSP firmware overhead** (B300 has GSP): measure interrupt cost,
  command latency.
- [ ] **I5 — Voltage scaling under power cap**: NVML voltage reading vs clock.
- [ ] **I6 — TDP cap tightening behavior**: 1100 → 800 → 600 → 400 W, measure
  throughput collapse curve per workload.
- [ ] **I7 — Multi-GPU power sharing** in DGX: do siblings throttle each other?
- [ ] **I8 — NVENC / NVDEC power draw**: does video engine compete for TDP?

## J. Real-workload end-to-end

- [ ] **J1 — Transformer block end-to-end**: RMSNorm + QKV proj + attention +
  MLP + residual, BF16, 1 block. Measure utilization vs SoL.
- [ ] **J2 — Llama-style 70B inference token/s**: realistic sustained.
- [ ] **J3 — FlashAttention v2/v3 implementation**: minimal version, hit 50%+ spec.
- [ ] **J4 — RMSNorm fused with bias + activation**: optimal recipe.
- [ ] **J5 — Sampled sparse attention** (random k tokens): perf and TF/W.
- [ ] **J6 — KV-cache update kernel**: throughput and latency.
- [ ] **J7 — MoE routing kernels**: token-to-expert dispatch overhead.
- [ ] **J8 — Top-k / argmax** at various K and N: optimal SoL.

## K. API / driver overhead

- [ ] **K1 — Driver API vs Runtime API**: per-call overhead delta.
- [ ] **K2 — CUPTI instrumentation overhead**: marker, range, callback.
- [ ] **K3 — NVTX overhead in profiler vs no-profiler** (verified 19 ns no-profiler).
- [ ] **K4 — `cudaGetDeviceProperties` on hot path**: cached vs not.
- [ ] **K5 — `cuModuleLoad` cost** for various PTX/CUBIN sizes.
- [ ] **K6 — `cudaStreamSynchronize` polling vs blocking**: latency floor differences.
- [ ] **K7 — `cudaEventSynchronize` blocking vs spin**: same.

## L. Cooperative groups + clusters

- [ ] **L1 — Cluster size > 16** (we know 16 max non-portable, 8 portable).
  What happens at attempted 32? Error vs split?
- [ ] **L2 — Cooperative grid launch** (cooperative groups grid-sync) vs
  classical launch.
- [ ] **L3 — `cluster.barrier::arrive` async** with mbarrier wait pattern.
- [ ] **L4 — Coalesced groups dynamic group formation** overhead.

## M. Specialization + pipelining patterns

- [ ] **M1 — Producer/consumer warp specialization** with TMA + tcgen05.
  Real-kernel-quality recipe.
- [ ] **M2 — `cuda::pipeline` (cuda/pipeline.h)** primitives for SHMEM staging.
- [ ] **M3 — Multi-stage software pipelining** of MMA: how many stages
  before diminishing returns?
- [ ] **M4 — Async copy + compute overlap quantification** at various
  arithmetic intensities.

## N. Speculative / mysteries

- [ ] **N1 — Tensor core warmup**: does first MMA after idle take longer?
- [ ] **N2 — Cluster launch tail behavior**: under contention, do clusters
  stall fairly?
- [ ] **N3 — ECC overhead**: B300 has ECC; can we measure correction events
  via NVML, observe perf hit?
- [ ] **N4 — Dynamic Parallelism throughput** (we know launch is 8.56 µs):
  bulk DP launches per kernel.
- [ ] **N5 — Warp-divergent atomics**: cost of atomic when divergent within
  warp vs uniform.
- [ ] **N6 — `__threadfence_system` cost across MIG instances** (if MIG works).
- [ ] **N7 — Kernel preemption granularity**: at what instruction can a
  kernel be interrupted? Measure latency.
- [ ] **N8 — JIT compilation overhead**: PTX → SASS at first launch, second.
- [ ] **N9 — SASS instruction encoding limits**: max immediate width,
  predicate count, register set size per instruction.

## O. Storage / I/O

- [ ] **O1 — GPUDirect Storage** path: SSD → GPU skipping CPU.
- [ ] **O2 — `cuFile`** API throughput.
- [ ] **O3 — Page-locked staging buffer optimal size** for H2D streaming.

## P. Power: data-dependent on non-tcgen05 paths

- [ ] **P1 — mma.sync data-dep power**: same A vs B asymmetry as tcgen05?
- [ ] **P2 — FFMA data-dep power**: per-bit-pos popcount sensitivity.
- [ ] **P3 — IMAD / IMUL data-dep power**: int multiply same toggle model?
- [ ] **P4 — MUFU data-dep power**: special functions, are they data-dep?
- [ ] **P5 — LDG data-dep power**: address bits or value bits?

## Q. Speed-of-Light recipes still missing

- [ ] **Q1 — Optimal absmin reduction** (we have absmax).
- [ ] **Q2 — Optimal sort kernel** (small radix, single block).
- [ ] **Q3 — Optimal scan** (prefix sum) vs cub::DeviceScan.
- [ ] **Q4 — Optimal compact (predicate-filter)**.
- [ ] **Q5 — Optimal transpose** at various tile sizes.

---

## Methodology reminder for picking + executing

1. Read the rigor protocol (CLAUDE.md section "B300 Methodology").
2. Mark item `[~]` (in progress) before starting.
3. Verify with ≥3 methods (wall-clock + ncu + SASS).
4. Always `pkill -9 QuickRunCUDA` between measurements (lessons learned!).
5. State HIGH/MED/LOW confidence + what would change conclusions.
6. Commit: `<topic>: <one-line finding>` style.
7. Mark `[x]` with commit hash in this file.

When a `[ ]` item turns out to be already done elsewhere, mark `[duplicate of X]`
and remove. When it's not feasible (e.g., needs hardware we don't have),
mark `[blocked: <reason>]`.

---

**SUPERSEDED 2026-04-20** by `CURIOSITY_LIST_V4.md` (ninja microarchitecture
focus). V3 had too much LLM/framework-level work (transformer block,
FlashAttn, MoE). V4 is pure low-level/SASS/microarchitecture.
