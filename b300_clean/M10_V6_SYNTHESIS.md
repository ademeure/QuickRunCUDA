# M10: V6 Curiosity List Synthesis (B300 sm_103a)

V6 = 60/65 [x] + 5 deferred to V7 (tcgen05.mma full descriptor work).

---

## Headline findings (most actionable)

### 1. Pipe overlap is bottlenecked by SCHEDULER ISSUE PORT, not pipe-specific contention

| Pair (single warp) | Overlap |
|---|---|
| FFMA + LDS | 96% |
| MUFU + FFMA | 100%+ (super-linear) |
| HMMA + LDS | 73% |
| HMMA + MUFU | 71% |
| HMMA + HMMA (indep) | 69% |
| FFMA + IADD3 | 56% |
| HMMA + IADD3 | 32% |
| HMMA + FFMA | 31% |
| HMMA + LDTM | 28% |

3 op categories: async-queue (LSU 96%+), pipe-queued (XU 71-100%), issue-port-bound (FMA/ALU 30-56%). Hide LSU/XU work behind compute freely; don't mix two compute ops expecting parallelism.

(See M8_PIPE_OVERLAP_MATRIX.md for full architectural model.)

### 2. Workload-dependent min-energy clock

| Workload | Min-energy clock |
|---|---|
| Pure FFMA-bound | 510 MHz (16% savings) |
| Memory-bound | 800 MHz (36% savings) |
| **Mixed (typical ML)** | **1992 MHz boost** (3× lower energy than 510!) |

ML inference USE BOOST CLOCK. Common DVFS belief "low clock = low energy" is FALSE for typical workloads.

(See M9_ENERGY_PARETO.md for full Pareto.)

### 3. Cluster launch is FASTER than direct launch

cudaLaunchKernelEx with cluster ≥4 = 2.57 µs vs direct 3.09 µs (17% faster).
Combined with DSMEM affinity, cluster launch wins on TWO axes for HMMA-bound kernels.

### 4. cmem args 3× faster than global LDG for scalars

Pass scalars as kernel ARGS (cmem b0), not via memory pointers. 3.4 vs 10.5 cy/op for repeated reads.

### 5. prefetch.L2 + cp.async = 1.58× speedup

Schedule `prefetch.global.L2` 4 iters ahead of cp.async — 37% latency reduction. Major optimization for cuTLASS-style pipelines.

### 6. CUDA Graph patterns

- Build: capture 12% slower than explicit (1.0 vs 0.85 µs/node)
- Instantiate: 854 µs for 100-node graph (one-time cost)
- ExecUpdate: 4-16× faster than re-instantiate
- SetParams (V5 H1): 0.4 ns FREE
- Launch warm: 1.0 µs/kernel (vs direct 2.3 µs)
- Linear chain: 1.18× slower than parallel
- **Conditional graphs: 1220× slower** — avoid for hot loops

### 7. Persistent kernels

- Single-block RTT: 2.77 µs (3.5× faster than cold launch 9.8 µs)
- Multi-block coordination: 2.5 µs/block linear (don't scale beyond cluster)
- Use spin-wait, not cudaDeviceSync
- Saves 7 µs/task vs cold launch — wins for high-freq dispatch

### 8. Stream parallelism scales to 128 (108×!)

Linear scaling 1→128 streams (98%+ efficiency); saturates at 128 (matches HW dispatch limit). Stream priority does NOT preempt — scheduling hint only. Always use cudaStreamNonBlocking flag (regular custom stream is +20% slower due to implicit nullstream sync).

### 9. HMMA latency curve

ILP=1: 20 cy/op → ILP=4: 8 cy/op (saturated). cuTLASS warpgroup (4 chains) is exactly the right ILP target.

### 10. HBM single-load latency

True cold-DRAM = 81 cy = 54 ns at 1500 MHz (matches HBM3E spec). B300 L2 = **126 MB** (much larger than Hopper's 50 MB).

---

## Architectural facts confirmed/discovered

- L2 = **126 MB** (NOT 60 MB as some docs say)
- SMEM/SM = 228 KB
- Cluster MAX = 8 CTAs (CSIZE=16 silently fails)
- HBM3E single-load latency = 54 ns (much faster than Hopper-era 250+ cy estimates)
- F2FP cvt unit is mantissa-width-agnostic: FP4/FP6/FP8/FP16 all 5.4 cy
- INT8 cvt = 2.9 cy (similar fast path)
- WGMMA REMOVED on B300 (ptxas explicit error)
- cudaLaunchCooperativeKernelMultiDevice REMOVED in CUDA 13
- Texture/RO cache OBSOLETE (no perf benefit over LDG)
- 128 HW dispatch slot limit (matches V5 stream finding)

---

## Deferred to V7

1. **tcgen05.mma full descriptor encoder** (B1-B5):
   - cuTLASS source `cute/arch/mma_sm100_umma.hpp` shows complex 5-arg + 4-tuple format
   - Requires precise idesc bit packing for B300
   - Use cuTLASS abstractions for production; raw PTX needs careful study

2. **Multicast TMA** (I5):
   - cp.async.bulk.shared::cluster.multicast::cluster compiles
   - Runtime fails — needs DSMEM mbarrier + cluster smem addressing
   - Use cute::SM90_TMA_LOAD_MULTICAST abstraction for production

3. **NVFP4 cvt scalefactor variant** (H3):
   - Standard cvt.rn.satfinite.e2m1xN.f32 syntax fails
   - Likely needs cvt.scalefactor (per-block scale, 8-element groups)

---

## V6 commits per category

A (pipe overlap): 13f5a16 aa8b7eb 17cf0d4 086ed25 d7da49c f1b2f4d
C (energy): b3486dc 389bdbb 4970264 81c7a74
D (graphs): 01caa1b 9a98440 eb1693d 4df42c9 3d21086
E (persistent): 98a1e25 e231b12 e54264c ce9ebb9 83a33c0
F (streams): 5a5a6e2 c85d736 77ea4b8 4b7c545 3034d59
G (memory): 4498386 feb97fc 2ed3250 4392814 7850d98 34d5521
H (numeric): c702139 bb569c1 3bb7051 f9ed4ea 0a999ba
I (async): 943b7d7 46a391e ff18e05 6fff122
J (random): c692407 36dbe20 2101fcc ff19f64 9c34e95 7150c04
K (launch): 917919d ca64455 2b363d4 5bba2c2 88eb08b
L (tooling): 272ae48 cb4c9f4 9947490 e39f34e c3e086b 678264c

Total V6 commits: ~60+ (each item produced 1+ commits)

---

## Tools delivered (utils/)

| Tool | Purpose |
|------|---------|
| pipe_dashboard.sh | Per-pipe utilization from ncu |
| sass_diff.sh | Opcode diff between cubins |
| mkbench.sh | Microbench template generator |
| overlap_matrix.sh | Pipe overlap matrix lookup |
| roofline.sh | Roofline plotter (AI vs FLOPS) |
| warp_latency.sh | Single-warp latency reference |
| auto_rigor.sh | Combined clean_run + ncu + sass count |
| power_sampler | NVML 6483 Hz power profiler |
| clean_run.sh | pidof-based wrapper |
| ncu_explorer.sh | ncu metric explorer |
