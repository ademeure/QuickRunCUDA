# B300 Consolidated Verified Findings (Session Investigations)

This document consolidates CONFIRMED findings from multiple independent sub-agent investigations. Each number here is cross-checked between sources.

Clock context: B300 SXM6 AC defaults to 2032 MHz boost, but `nvidia-smi -lgc 2032` paradoxically pins to 1920 MHz. Report measurements at BOTH clocks.

---

## HIGH Confidence Findings (cross-verified ≥2 sources)

### Clock behavior
- **Default (no lock): 2032 MHz** under FFMA load (verified: clock64/globaltimer ratio)
- **`nvidia-smi -lgc 2032`: pins to 1920 MHz** (paradox!)
- Only ever 2032 or 1920; no intermediate clocks observed

### FP32 FFMA latency
- **4 cycles** per FMA (Agent 10, corrected from earlier "23 cy" which included loop overhead)
- Self-op `FFMA Ra,Ra,Ra,RZ` = diff-src `FFMA Ra,Rb,Rc,Ra` = 4.019 cy (NO register port penalty on Blackwell)
- ILP scaling: 1→4.0, 2→2.3, 4→1.2, 8→1.1 cy/FMA (approaches 1 cy/FMA per warp = single-pipe limit)

### FP32 FFMA peak throughput
- **Theoretical at 2032 MHz: 76.96 TFLOPS** = 148 SMs × 128 FP32 cores × 2 op/FMA × 2.032 GHz
- **Theoretical at 1920 MHz: 72.74 TFLOPS**
- Measured (at 2032 MHz, ILP=16, 64 warps/SM): **64.75 TFLOPS = 84% of theoretical**
- NO dual-issue (154 TFLOPS claim from Agent 01's formula was a 2× mistake — "256 cores" was wrong; B300 has 128 FP32 cores/SM like Hopper)

### Tensor core BF16 peak
- **mma.sync m16n8k16 peak: ~576 TFLOPS** at 2032 MHz with ILP=2, 64 warps/SM (Agent 03)
- **tcgen05.mma kind::f16: ~2325 TFLOPS** at 2032 MHz — 4× faster than mma.sync
- tcgen05 compiles via **NVRTC** but NOT via static ptxas (CUDA 13.2 bug — QuickRunCUDA works)
- cuBLAS FP8 (internally uses tcgen05): **4486 TFLOPS = 91% MFU** on 8192³ GEMM

### HBM Bandwidth
- **Write peak: 6.1-6.3 TB/s** ncu-verified via `dram__bytes_write.sum.per_second`
- **Read peak: ~6.88 TB/s** (9% higher than write)
- Catalog's earlier 3.4 / 7.09 / 8.5 claims all wrong:
  - 3.4 = L2 write buffer, not DRAM
  - 8.5 = fire-and-forget timing artifact
  - 7.09 = copied from read number
- Store width (scalar/v4/v8) doesn't matter at peak
- Cache hint (default/.cs/.wb/.volatile) doesn't matter at peak
- 1 CTA/SM already saturates HBM controllers

### Launch latency
- **Event floor ~4.3 µs** for small grids (1-296 blocks)
- **Linear regime at large grids**: ~0.52 µs/block at 1M blocks
- **GWS dispatch rate: 2M CTAs/s**
- Real noop kernel runtime: 3-4 ns (inside event overhead)
- "2.05 µs invariant" catalog claim was event-floor behavior only, not true across grid sizes

### B300 GPC Topology
- **8 GPCs total** (NOT 10 as earlier catalog claimed)
- ncu counter confirms exactly 8.0000
- Structure: 2 GPCs × 20 SMs + 6 GPCs × 18 SMs = 148 SMs
- SM IDs are round-robin/column-major across GPCs (NOT consecutive)
- Max cluster=8 correlates with GPC width
- Cluster=16 non-portable requires cross-GPC, fails on B300

### SHMEM Peak BW — Resolved
- **Peak: 37.6-38.0 TB/s @ 2032 MHz** (98% of 38.49 TB/s theoretical)
- Access driver: `LDS.128` (vector load) vs `LDS` (scalar)
- Both volatile and non-volatile v4 give IDENTICAL 37.63 TB/s
- **Catalog's "volatile forces re-reads" explanation is WRONG** — it's just scalar vs vector
- Scalar LDS × 8 = only 19-26 TB/s due to 4× more issue slots
- Thermal throttle: sustained load drops 2032→1920 MHz after ~2ms; peak is burst number

### Atomic Peak — Resolved
- **530+ Gops/s peak** at stride=4B + UNROLL=16 (L2-resident)
- Catalog's "137/273/372" are ALL correct for their conditions (different ILP)
- **L2/DRAM boundary at stride=64B** (footprint crosses 126 MB L2)
- DRAM-bound = only 10-14 Gops/s
- "2.7× coalescing speedup" claim was actually L2 residency effect

### L1 Cache Structure (Agent 12)
- **L1 + SHMEM = 256 KB unified pool** per SM (confirmed at every carveout)
- L1 hit latency: **42-45 cy** at 2032 MHz (confirmed via ca vs cg: ca=40, cg=552 at 8 KB WS)
- L1 sizes by carveout:
  - co=0 (max L1): 228 KB L1, 28 KB SHMEM
  - co=25: 192 KB L1, 64 KB SHMEM
  - co=50: 128 KB L1, 128 KB SHMEM
  - co=75: 52-56 KB L1, 200 KB SHMEM
  - co=100 (max SHMEM): 20-22 KB L1, 234 KB SHMEM
- **Default carveout = high-SHMEM / minimal-L1** (kernels need explicit setattr(co=0) for full L1)
- Earlier catalog claims "L1=32/128/192 KB" all correct at different carveouts

### FP64 DFMA
- **Peak: 1.20 TFLOPS at 2032 MHz** (Agent 14; FP32/64 ratio = 64, matches 76.96/64 = 1.203)
- **Zero-pipelined per warp**: 64 cy spacing between DFMA issues, despite 16-cy FP64 pipe depth (25% util per warp)
- **Need 4 warps/SM** to saturate (not more ILP — ILP doesn't help DFMA)
- Each SMSP has 1 FP64 unit

### Fence costs (Cycles per fence, single warp)
- `fence.sc.cta`: **8 cy** (catalog's 14-29 was loop-mixed; isolated fence ~8)
- `fence.sc.gpu`: **277 cy**
- `fence.sc.sys`: **2883 cy**
- `fence.acq_rel.cta`: 9 cy
- `fence.acq_rel.gpu`: 271 cy (~sc.gpu)

### Atomic operations (cy per atomic, per-thread addresses, 2032 MHz)
| Operation | smem.cta | smem.cluster | smem.gpu | smem.sys | gmem.cta | gmem.cluster | gmem.gpu | gmem.sys |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| relaxed | 44 | 44 | 44 | 44 | 413 | 413 | 413 | 413 |
| acquire | 50 | 50 | 50 | 50 | 419 | 421 | 421 | 421 |
| release | 52 | 304 | 304 | ~5000 | 421 | 1455 | 1455 | ~6000 |
| acq_rel | 58 | 312 | 312 | ~5000 | 427 | 1463 | 1463 | ~6000 |

- Scope is FREE at relaxed ordering
- Release/acq_rel at cluster/gpu scope emits MEMBAR.ALL.GPU (~300 cy smem, ~1500 cy gmem)
- .sys scope with release adds system fence (~5000-6000 cy)
- seq_cst NOT supported on sm_103a (ptxas rejects)

### cuBLAS alignment cliff
- 4096 vs 4097: **1842 vs 180 TFLOPS = 10.2× slowdown** (not 30×)
- cuBLAS algo 66 (fast Blackwell tensor) only offered for aligned sizes
- Algo 24 fallback for misaligned = 10× slower
- M and K alignment critical; N forgiving
- Mitigation: pad M/K to multiple of 32 (workspace size doesn't help)

---

## B300 Hardware Topology (confirmed)

| Attribute | Value |
|---|---:|
| Compute Capability | 10.3 (sm_103a) |
| SMs | 148 |
| Clock (boost) | 2032 MHz |
| Clock (base) | 1920 MHz |
| FP32 cores per SM | **128** (NOT 256) |
| 4 SMSPs per SM, 32 lanes each | 4 × 32 = 128 lanes/SM |
| L2 cache | 126.5 MB |
| Max persisting L2 | 79.1 MB |
| SHMEM per SM | 228 KB usable (233,472 - 1024 reserved) |
| SHMEM opt-in per block | 227 KB |
| Reserved shmem per block | 1024 bytes at offset 0..1023 |
| Registers per SM | 65,536 |
| HBM total | 288 GB (267 GB usable post-ECC) |
| HBM bus width | 7680 bits |
| HBM stacks | 12 × HBM3E |
| Mem sync domains | 4 |
| Async engines | 4 |
| Stream priority levels | 6 (-5 to 0) |
| Cluster max size | 8 portable, 16 non-portable |

---

## Methodology Rules (learned the hard way)

1. **Always state theoretical BEFORE measured.** If measured > theoretical, test is broken.
2. **128 FP32 cores/SM, NOT 256.** Don't double-count.
3. **Runtime-load constants** for loop iterators to defeat compile-time folding.
4. **`-lgc 2032` pins to 1920 MHz**, not 2032. Clock mismatch = 6% error in TFLOPS.
5. **Fully unroll** inner loops for latency measurement (`#pragma unroll 1024`).
6. **Separate loop overhead from instruction latency** — use inner+outer unroll pattern.
7. **Self-op chains are SAFE on B300** (unlike Volta/Turing which had 2× penalty).
8. **ncu counters > event timing** for bandwidth measurements.
9. **cuBLAS algo selection varies by alignment** — test both aligned and misaligned sizes.
10. **NVRTC accepts more PTX than static ptxas** (tcgen05 works in NVRTC, rejected by ptxas).

---

## Still Pending (8+ investigations running)

- Agent 04: DSMEM 0.8% vs 4.7× slowdown
- Agent 05: FP32 38 vs 72 TFLOPS Power section anomaly
- Agent 06: L2 peak BW (17-36 TB/s range)
- Agent 07: SMEM peak 19.85 vs 35 TB/s
- Agent 09: GPC count 8 vs 10
- Agent 11: PDL with realistic kernels
- Agent 12: L1 size at all carveouts
- Agent 13: Atomic peak 137 vs 372 G/s
- Agent 14: DFMA zero-pipelining claim
