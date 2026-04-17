# B300 SXM6 AC — Definitive Reference

Comprehensive characterization of NVIDIA B300 SXM6 AC (sm_103a, CC 10.3) from
217 measured tests. Numbers are MEASURED unless explicitly noted as theoretical.

System: AMD EPYC 9575F + B300 SXM6, CUDA 13.2 runtime / 13.0 driver, NVIDIA driver 580.126.09.

## Hardware Identity

| Property | Value |
|----------|-------|
| Compute Capability | 10.3 (sm_103a) |
| SMs | 148 |
| Threads/SM | 2048 (NOT 4096) |
| Threads/block | 1024 |
| Warps/SM | 64 |
| Regs/SM | 65,536 |
| HBM3E | 287.4 GB total, 7672 GB/s theoretical |
| L2 Cache | 126 MB |
| Max persisting L2 | 79 MB (62.5%) |
| SHMEM/SM | 228 KB total, 227 KB opt-in/block, 1024 B reserved |
| L1+SHMEM unified | 256 KB per SM |
| Boost clock | 2032 MHz |
| Power | min 200 W, max 1100 W (NOT 700 W) |
| PCIe | Gen 6 x16 (256 GB/s spec, only 57.7 GB/s effective) |
| GPCs | 8 (visible from cluster placement) |
| Async copy engines | 4 |
| Concurrent kernel slots | 128 (NOT 148 SMs) |
| Max cluster size | 16 (non-portable; 32+ silently fails) |

## Compute Throughput Ladder

| Operation | TFLOPS | % of Theoretical |
|-----------|--------|------------------|
| FP64 cuBLAS Dgemm | 1.1 | 92% |
| FP32 FFMA peak (256 thr × 24 ILP × full occ) | **74.6** | 97% |
| FP32 cuBLAS Sgemm | 68.6 | 89% |
| FP16/BF16 packed FMA (no tensor) | 58.2 | (= FP32, NO 2x speedup) |
| BF16 mma.sync m16n8k16 | 569 | (7.4x FFMA) |
| TF32 cuBLAS GemmEx | 1109 | (16x FP32) |
| FP16 cuBLAS Hgemm | 2221 | (32x FP32) |
| FP8 cuBLAS LtMatmul | 4491 | 91% (60x FP32) |

Math intrinsics (vs FFMA = 1.0x):
- exp2: 5.1x slower, sin/cos: 6.8x, exp: 9.3x, log: 11.1x
- log2: 29x, sqrt: 44x, div: 62x, rsqrt: 94x

## Memory Bandwidth Ladder

| Memory | BW (GB/s) | % of Theoretical |
|--------|-----------|------------------|
| L1 (16 MB workset, 8-ILP) | 46,562 | 314 GB/s/SM |
| L2 (64 MB workset) | 22,965 | (60% of L2 spec) |
| L2 edge (126 MB workset) | 8,182 | (cliff exactly here) |
| HBM read (4 GB workset, 8-ILP) | **7,120** | 93% |
| HBM write user kernel (8-ILP) | 6,174 | 80% |
| HBM write cudaMemset | **7,470** | 97% (fastest path) |
| SHMEM peak (4 reads + 1 write/iter) | 27,200 | 71% of 38.5 TB/s spec |
| Constant mem broadcast | 15,953 | (27x slower for divergent) |
| PCIe Gen 6 x16 H2D | 57.7 | only 23% of spec (CPU bottleneck) |

## CPU↔GPU Coordination Latency Ladder

| Mechanism | Latency |
|-----------|---------|
| __syncwarp / shfl_sync | 0.85-1 ns |
| __threadfence_block | 8 ns |
| __syncthreads (256 thr) | 14 ns |
| cluster.sync (any size 2-16) | 190 ns |
| __threadfence (device) | 385 ns |
| __threadfence_system | 861 ns |
| Cross-block flag wait (one-way) | 790 ns |
| cudaMemcpy sync (small) | 3.6 us |
| **Persistent kernel + mapped mem** | **4 us** ← best CPU↔GPU |
| cudaStreamWaitValue | 6 us |
| cudaStreamSynchronize per launch | 7 us |
| Event-based cross-stream | 27 us |
| Cross-process IPC | 100+ us |
| cuMemAdvise + Prefetch | ms-scale |

## API Overhead Ladder

| API | Cost |
|-----|------|
| cudaGetLastError / GetDevice | 20 ns |
| NVTX (no profiler) | 19 ns (free) |
| cudaPointerGetAttributes | 50-80 ns |
| nvmlDeviceGetClockInfo | 120 ns |
| cudaMallocAsync hot reuse | 328 ns |
| cuCtxGetCurrent / Push / Pop | 30 ns |
| cuStreamWaitValue alone | 6 us |
| cudaMemcpyAsync submission | 1.2 us |
| cudaStreamSynchronize (idle) | 1.2 us |
| Stream capture per kernel | 0.2 us |
| cudaGraph instantiate (100 nodes) | 35 us |
| cudaGraphExecUpdate | 1.4 us (35x faster than reinstantiate) |
| **PrimaryCtxRetain+Release** | **240 ms** (one-time init!) |
| cudaMallocAsync vs cudaMalloc | 184-770x speedup |

## Atomic Op Cost (uncontended, hot location)

| Op | Cost (ns) |
|----|-----------|
| atomicInc / atomicDec | 4 (FASTEST) |
| atomicAdd / Sub / Min / Max FP32 | 7-8 |
| atomicAnd / Or / Xor | 11 |
| atomicAdd FP64 | 4.5 (HW path) |
| atomicAdd packed half2/bfloat162 | 16 per element (HW path) |
| atomicExch / atomicCAS | 24-26 |
| atomicAdd_block on shared | 14 (FAST shared path) |
| **atomicAdd scalar __half/__nv_bfloat16** | **700 (200x slower! No HW)** |
| atomicAdd_system | same as atomicAdd (97 ns w/ contention) |

Atomic contention scaling (per-warp targets):
- Same cache line (stride 1): 7 Gops/s
- Cache-line apart (stride 32): 64 Gops/s
- 1024 B apart: 111 Gops/s (15x faster than collocated)

## Power Efficiency (TFLOPS/W)

| Workload | Power | TFLOPS | TFLOPS/W |
|----------|-------|--------|----------|
| FFMA peak (full occ) | 361 W | 74.6 | 0.21 |
| FFMA non-peak (low occ) | 437 W | 73 | 0.17 |
| BF16 mma.sync | 411 W | 569 | **1.39** (8x FFMA) |
| FP8 cuBLAS | 886 W | 4491 | **5.07** (30x FFMA) |

Sustained: B300 maintains boost (2032 MHz) for 30+ sec without throttling.
FP8 sustained 12 sec @ 886 W, 59 °C.

## Key Surprises (vs Common Assumptions)

1. **FP16/BF16 packed FMA = FP32** throughput outside tensor cores (no 2x speedup)
2. **Texture cache 2-3x SLOWER** than __ldg on B300 (use only for filtering)
3. **Pageable malloc memory MIGRATES** to GPU HBM at 1.5 TB/s; pinned stays on host (84 GB/s via PCIe)
4. **atomicAdd_block on global has NO benefit** vs default scope (only matters for shared)
5. **2-way thread divergence is FREE** (compiler uses select); 4+ paths: 6-60x slowdown
6. **match_any 38x slower** than regular shfl
7. **Cluster blocks NOT in same GPC** (8-block cluster spans 4 GPCs)
8. **cudaMemset hits 97% HBM peak** (faster than user kernel writes can)
9. **`nvidia-smi -lgc 2032` paradoxically pins to 1920 MHz** (-6% perf)
10. **cudaMallocAsync 184-770x faster** than cudaMalloc (pool reuse 328 ns)
11. **redux.sync only supports INTEGER** on sm_103a (no FP)
12. **cg::reduce 3.6x faster** than hand-rolled shfl_xor (uses HW redux.sync)
13. **128 concurrent kernel slot limit** (cliff at 144 streams)
14. **Persistent kernel + mapped memory = 4 us** CPU↔GPU round-trip (best available)
15. **__noinline__ device function call 14x slower** than inline
16. **__constant__ array divergent access 21-27x slower** than uniform
17. **Tensor + FFMA on separate streams 8x SLOWER** (NOT independent units)
18. **Branchless bit-tricks 35% SLOWER** than if-else (compiler is smarter)
19. **PCIe Gen 6 x16 confirmed** but effective only 57.7 GB/s (23% of spec)
20. **cudaEventDisableTiming saves 33%** on event sync (3.5 us)

## Recipes for Peak Performance

**Peak FFMA** (74.6 TFLOPS = 97% theoretical):
- 256 thr/block × 8 blocks/SM = 2048 thr/SM (full occupancy)
- 24-way ILP via 3 chains × 8 vars
- 28 regs/thread (low pressure)

**Peak HBM read** (7120 GB/s = 93% theoretical):
- 512 thr/block × 4 blocks/SM
- 8-way ILP unrolled int4 loads
- __launch_bounds__(BS, MIN_BLK) hint

**Peak FP8 GEMM** (4491 TFLOPS = 91%):
- cuBLAS LtMatmul, M=N=K=8192
- CUDA_R_8F_E4M3 inputs, CUDA_R_16BF output
- CUBLAS_COMPUTE_32F

**Peak SHMEM** (27.2 TB/s):
- 256 thr/block, 4 reads + 1 write per iter
- Avoid bank conflicts (stride 33 if 32-way conflict)

**Lowest-latency CPU↔GPU** (4 us round-trip):
- Persistent kernel polling on mapped memory
- ld.acquire.sys.u32 on GPU side
- volatile + __sync_synchronize on CPU side

## Anti-DCE Checklist for Microbenchmarks

The compiler is aggressive. To get real measurements:
1. Use runtime constants from kernel args, not literals
2. Final output must be unconditionally written to global
3. Use loop counter `i` somewhere in computation
4. Accumulate values that depend on previous iter (defeat unrolling)
5. Verify with SASS: `nvcc -keep -arch=sm_103a` and inspect `.sass`
6. Cross-check ratio to theoretical peak (>1.5x is impossible — DCE)
7. Cross-check ratio to FFMA: typical ALU ops within 5x of FFMA;
   anything 100x slower or 100x faster is suspicious

See CLAUDE.md for the full B300 benchmarking methodology.
