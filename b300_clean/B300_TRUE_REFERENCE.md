# B300 TRUE REFERENCE — Rigor-Verified (2026-04-18)

**The authoritative B300 SXM6 AC reference, derived from the M3 rigor sweep.**

This document consolidates the HIGH-confidence findings produced by the
b300_clean rigor task list (commits A1–L4 of branch `f2fp-deep-dive`).
Every number here was verified by 3 independent methods (wall-clock +
ncu + SASS) per the rigor protocol.

**This file is NEW; it does NOT replace B300_PIPE_CATALOG.md** (which is
preserved for historical context and lower-confidence numbers). When in
doubt, prefer numbers from this file. See `M3_REVERIFY_LOG.md` for the
re-verification chain on each entry.

System: NVIDIA B300 SXM6 AC (sm_103a, CC 10.3, 148 SMs, 12 HBM3E stacks),
sustained 1920 MHz SM clock (boost is 2032 but rarely sustained),
CUDA 13.2 runtime / 13.0 driver (580.126.09).

---

## 1. Memory bandwidth ladder — VERIFIED

| Memory | Peak (TB/s) | % of theoretical | Recipe (commit) |
|---|---:|---:|---|
| **HBM3E read** | **7.30** | 95% of 7672 spec | v8 + per-warp coalesced + non-persistent (a04d9c8) |
| **HBM3E write** | **7.30** | 95% | same recipe (a04d9c8) |
| **HBM3E write NINJA (1 v8 store/warp)** | **7.57** | **98.7%** ← SoL | NINJA recipe (e75c7e1) — beats cudaMemset by 5% |
| **HBM3E concurrent R+W (best ratio)** | **7.31** | 95% | pure R or pure W (de3b4d5) |
| **HBM3E concurrent R+W (50:50)** | **6.68** | 87% | minimum at any mix (de3b4d5) |
| **D2D copy (separate src/dst)** | **6.93** | 90.3% | NINJA recipe (4958d6b) — 5.5% > cudaMemcpyAsync; src/dst on diff stacks unlock partial-duplex |
| **L2 kernel-effective** | **23.85** | (incl. L1 reuse) | v8 + 8-ILP @ 96 MB workload (1e590cf) |
| **L2 bus traffic (ncu lts)** | **13.30** | (pure L2 BW) | same kernel (1e590cf) |
| **SHMEM read peak** | **38.4** | 99.8% of 38.5 spec | pure-vector reads (d41c38c) |
| **SHMEM stmatrix W+R** | **34.5** | 90% | tight stmatrix chain (8bd85e8) |
| **DSMEM (CL=2)** | **3.06** | (cluster-internal) | aggregate; per-cluster 41 GB/s (4129760) |
| **NVLink-5 P2P read** | **0.78** | 1.04× spec 757 | 4096 blocks v8 (9172429) |
| **NVLink-5 P2P write** | **0.71** | 0.94× spec 757 | 4096 blocks v8 (9172429) |
| **PCIe Gen6 x16 H2D** | **0.058** | 23% of 256 spec | CPU-bound (d40a35e) |

## 2. Compute peaks ladder — VERIFIED

| Operation | Peak (TFLOPS) | % theoretical | Notes (commit) |
|---|---:|---:|---|
| **FP32 FFMA peak** (boost 2032 MHz unlocked) | **74.62** | **96.92%** | NCHAIN=3 rotating, immediate constant, 256×8/SM (06b0d8d) |
| FP32 FFMA (locked 1920 MHz) | 62.17 | 85.5% | (1920 boost-pin paradox; unlock w/ `nvidia-smi -rgc`) (e1a1220) |
| **FP64 DFMA** | **1.20** | 100% of 1.20 spec | (2d64696) |
| **FP16/BF16 mma.sync m16n8k16** | **569** | 7.4× FFMA | (a37d989) |
| **BF16 cuBLAS GEMM N=8192** | **2242** | 90% of NVIDIA 2500 spec | (e752547) |
| **FP8 e4m3 cuBLAS LtMatmul** (zero data, sustained via cudaGraph) | **4425** | 88.5% of 5000 spec, 30 sec @ 943 W | (06b0d8d) |
| **FP8 e4m3 cuBLAS LtMatmul** (random data) | **3983** | 80% of spec — REALISTIC | (bf98e90) |
| **BF16 mma.sync 8-chain** (multi-accumulator) | **~570** | matches catalog "burst 569" | (83ef1c6) — single-chain "1543" was over-counted; RETRACTED |

**WARNING**: catalog claims like "FP8 4491 TFLOPS" were ALL zero-data.

Data-dependent throughput (commit 6e40ef9, N=8192 cuBLAS):

| Precision | zero/const | random | normal-ish | worst slowdown |
|-----------|----------:|-------:|-----------:|---------------:|
| FP16      | 2246      | 1905   | 1744       | -22% |
| BF16      | 2246      | 1883   | 1850       | -18% |
| FP8 e4m3  | 4393      | 3984   | 3951       | -10% |

**For realistic training/inference workloads, use the realistic numbers**.
Even more brutal under 600 W power cap: FP8 random = 3087 TFLOPS (-43%).

## 3. Coordination latency ladder — VERIFIED

| Mechanism | Latency | Source |
|---|---:|---|
| `__syncwarp` | 1 ns | (5d632d5) |
| `__threadfence_block` | 8 ns | (b06f366) |
| `__syncthreads` (256 thr) | 14 ns | (b06f366) |
| `cluster.barrier::arrive` (relaxed) | 50 ns | (d8ca01a) |
| **mbarrier arrive+try_wait.parity** | **57.7 ns/cycle** | (af35338) |
| SHMEM atomic + `__syncthreads` | 29.6 ns/cycle | (af35338) |
| `cluster.sync` | 175 ns | (d8ca01a) |
| `__threadfence` (device) | 385 ns | (b06f366) |
| `__threadfence_system` | 861 ns | (b06f366) |
| **Local atomic L2 round-trip** | **164 ns** | (ad19660) |
| **Cross-GPU atomic via NVLink P2P** | **1662 ns** | (ad19660) |
| Persistent kernel + mapped memory | 4 µs | (584fda6) |
| `cudaMemcpy` sync (small) | 3.6 µs | (37a66f9) |
| `cudaStreamSynchronize` per launch | 7 µs | (1cd7f03) |

## 4. API costs — VERIFIED

| API | Cost | Source |
|---|---:|---|
| `cudaGetLastError` / `cudaGetDevice` | 20 ns | (d5c36c6) |
| NVTX (no profiler) | 19 ns | (3a76c96) |
| `cudaPointerGetAttributes` | 50-80 ns | (b8a32b0) |
| `nvmlDeviceGetClockInfo` | 120 ns | (ca3666c) |
| `cudaMallocAsync` (hot reuse) | 328 ns | (00014ab) |
| `cudaStreamGetCaptureInfo` | 30 ns | (1ff0fdb) |
| **`cudaMemset` (4 B)** | **1.22 µs** | (be28c14) — 31% FASTER than noop kernel! |
| `cudaMemcpyAsync` submission | 1.2 µs | (c6e7fc1) |
| `<<<1,1>>>` noop kernel launch | 1.78 µs | (be28c14) |
| `cudaGraphInstantiate` (100 nodes) | 35 µs | (e8760f3) |
| `cudaGraphExecUpdate` (100 nodes) | 1.4 µs | (e8760f3) — 25× faster than reinstantiate |
| **`cuCtxCreate` (full init)** | **240 ms** | (73455b2) — once per process! |

## 5. Atomic throughput — VERIFIED

| Configuration | Gops/s | Source |
|---|---:|---|
| Stride 0 (full collision) | 0.79 | (e7aab3a) |
| **Stride 4 (cache-line combining)** | **449** | (e7aab3a) — peak |
| Stride 32 (1 line/thread) | 184 | (e7aab3a) |
| Stride 256+ (scattered) | ~150 | (e7aab3a) — plateau = ~32 L2 atomic units |

**Op type ladder** (uncontended scalar):
| Op | Cost (ns) | Source |
|---|---:|---|
| atomicInc / Dec | 4 | (8d5d6ff) |
| atomicAdd FP32 / Min / Max | 7-8 | (8d5d6ff) |
| atomicAnd / Or / Xor | 11 | (8d5d6ff) |
| atomicAdd FP64 (HW path) | 4.5 | (8d5d6ff) |
| atomicExch / atomicCAS | 24-26 | (8d5d6ff) |
| **atomicAdd scalar half / bfloat16 (NO HW)** | **700** | (0b0faf7) — 200× slower |
| atomicAdd packed half2 / bfloat162 (HW) | 16/elem | (5aa50f7) |
| **red.release.gpu.global** | **614 ns/op** | (9467cfe) — MEMBAR.ALL.GPU between each! |

## 6. Architecture facts — VERIFIED

- **8 GPCs × ~18 SMs each** (= 144 active + 4 spare = 148 total) — boot-clock skew (320f0e8)
- **L2 = 126 MB, 2 partitions split by die boundary** — 2.4× latency near vs far (af91798)
- **L2 hash flips at 4 KB stride** — empirical side-aware code uses CHUNK_SIZE=4096
- **L2 is PHYSICALLY tagged** — VMM aliasing safe (d559e0a)
- **Max usable cluster size = 16 (non-portable)**, 8 portable (f33c21d)
- **SHMEM/SM = 228 KB total**, 227 KB opt-in/block, 1024 B reserved
- **Concurrent kernel slot limit = 128** (NOT 148 SMs) (7407cba)
- **Pageable memory MIGRATES to GPU on first touch** at 1.5 TB/s (00d971c)
- **PCIe Gen 6 x16** physically; effective BW only 57.7 GB/s (d40a35e)
- **Power: min 200 W, max 1100 W** (NOT 700 W) (862014c)

## 7. Counterintuitive findings (the surprises)

1. **`nvidia-smi -lgc 2032` paradoxically pins to 1920 MHz** (-6% perf) (861e8d1)
2. **FP16/BF16 packed FMA = FP32 throughput** outside tensor cores (no 2× speedup) (ea47ec6)
3. **cudaMemset is 31% FASTER than noop kernel** (private fast-path dispatch) (be28c14)
4. **cudaMemset's underlying kernel is HIDDEN** — 6 hook methods all blocked (be28c14)
5. **Full occupancy uses LESS power** per TFLOPS (47% more efficient) (cec1ac6)
6. **Random data is up to 43% SLOWER** than zero data for FP8 cuBLAS under power cap (bf98e90)
7. **Persistent L2 attribute has NO measurable benefit** for streaming workloads (d2ccf76)
8. **Pageable coherence "bug" is NOT REPRODUCIBLE** on driver 580 (e3bdc1e)
9. **Cache hints `.cg/.cs/.wb` have NO effect** on re-read at 4 MB scale (e87d8aa)
10. **NO CCTL.IVALL emitted** for any red.global variant on B300 sm_103a (9467cfe)
11. **red.release.gpu.global is 9.1× SLOWER** than red.global (MEMBAR.ALL.GPU per op) (9467cfe)
12. **Stream priority: 6 levels in API but only 2 effective tiers** (same vs any-higher) (6050ff6)
13. **HBM3E concurrent R+W is WORSE than serial** (no full-duplex) (eb6abf3)
14. **5% HBM gap to spec is PARALLELISM, not refresh** (1 KB bursts hit 98.6%) (66a2853)
15. **Tensor cores share SM clock domain with FFMA** — IDENTICAL clock (8ff067a)
16. **Branchless code is 35% SLOWER** than if-else on B300 (compiler smarter) (f2f70c3)
17. **`__noinline__` device function is 14× SLOWER** than inline (009b3e5)
18. **L2 atomic units count = ~32** (stride sweep plateau analysis) (e7aab3a)
19. **MLOPart MPS feature** is the ONLY way to control L2 partition affinity per CUDA device (af91798)
20. **CCTL.IVALL doesn't exist on B300** — agent claim was wrong (9467cfe)

## 8. Speed-of-Light recipes

### Best HBM write throughput (7.57 TB/s = 98.7% spec) — NINJA recipe
```cuda
__launch_bounds__(256, 8) __global__ void w_ninja(int *data, int v) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, lane = tid & 31;
    int *p = data + (warp_id * 32 + lane) * 8;
    asm volatile("st.global.v8.b32 [%0], {%1,%1,%1,%1,%1,%1,%1,%1};"
        :: "l"(p), "r"(v) : "memory");
}
// Launch: <<<bytes / (256 * 32), 256>>>
//   - Each warp does ONE v8 store = 1 KB
//   - Massive warp parallelism replaces per-warp loop pipelining
//   - Beats cudaMemset by 5%, beats prior 32-iter recipe by 2.4%
```

### Best HBM read throughput (7.31 TB/s = 95.3% spec)
```cuda
// Same NINJA principle, IT=2 (2 KB per warp)
// See investigations/ninja_read.cu
```

Read peak (95.3%) < Write peak (98.7%) by 3.4 pp — fundamental to HBM3E
read protocol roundtrip overhead.

### Best BF16 absmax (6.92 TB/s = 94.8% of HBM read peak — NINJA upgraded)
```cuda
// Block-level reduce with optimal block count = 18944 blocks
// Each warp reads 1024 B (matches HBM ninja burst size)
// Single atomicMax per block (not per warp — atomic contention kills it)
// (See investigations/ninja_absmax_v2.cu)
```
Improved from 6.74 TB/s by tuning block count and using 1-KB-per-warp
HBM access pattern (matches the HBM read SoL recipe).

### Best BF16 row-softmax (~6 TB/s — corrected analysis)
```cuda
// 256 thr/block, 1 row per block. Each thread loads 16 BF16 = 2 uint4 ONCE.
// Keep uint4 v0, v1 alive across passes via vectorized loads + register reuse.
// (See investigations/ninja_softmax_v4_register.cu — commit 57a2e6f)
```
**Actual ncu DRAM rate**: 6.07 TB/s aggregate (3.07 R + 3.00 W) =
**91% of HBM concurrent R+W ceiling (6.68 TB/s)**, which is the
*relevant* SoL for mixed-traffic kernels — not the pure-direction peak.

Speedup over naive 3-pass: ~1.5–1.8× (varies with baseline run).
Per Agent 3 critique: the "82.3% HBM peak" framing was wrong (used
pure-read denominator). Bigger contributing factors are vectorization
(LDG.E.U16 → LDG.E.128 = 8× fewer L1 transactions) AND register-keep,
not register-keep alone.

### Best BF16 axpy (7.02 TB/s = 91.5% spec — NINJA block tuning)
```cuda
// 131072 blocks of 256 threads (vs 592 in original). MORE small blocks wins.
// (See investigations/ninja_axpy.cu — commit 866e951)
```

### Best 256-bin BF16 histogram (6.57 TB/s = 90.1% of HBM peak)
```cuda
// SMEM aggregation — atomicAdd_block on 256-bin smem, then global flush
// (See investigations/rigor_l3_histogram.cu)
```

### Best FP8 cuBLAS GEMM (4400 TFLOPS zero, 3983 random)
- cuBLAS LtMatmul, M=N=K=8192
- CUDA_R_8F_E4M3 inputs, CUDA_R_16BF output, CUBLAS_COMPUTE_32F
- (See investigations/d4_tcgen05_via_cublas.cu)

### Lowest-latency CPU↔GPU (4 µs round-trip)
- Persistent kernel polling on mapped memory
- `ld.acquire.sys.u32` on GPU side
- `volatile` + `__sync_synchronize` on CPU side

## 9. Methodology — USE THE HARNESS

For any new B300 measurement, run:
```bash
./utils/rigor_run.sh ./your_binary
```
This auto-runs:
1. Wall-clock event timing
2. ncu DRAM/pipe metrics
3. SASS instruction census

If the 3 methods don't reconcile within 10%, the test is broken. See
`utils/rigor_harness.h` for the C++ Bench class API and
`b300_clean/M3_REVERIFY_LOG.md` for the re-verification record.

## Provenance

Every entry in this document is sourced from a specific commit on the
`f2fp-deep-dive` branch. Resolve any commit hash with `git show <hash>`
to see the exact test code and data. The b300_clean directory contains:
- `01_hbm_bandwidth.md` — extensive HBM detail (this file's source)
- `02_shmem.md` through `17_nvrtc_module.md` — category breakdowns
- `M3_REVERIFY_LOG.md` — full re-verification chain
- `TASK_LIST.md` — the rigor task list and per-task status
- `B300_TRUE_REFERENCE.md` (this file) — the master summary

For historical / lower-confidence numbers, see the parent dir's
`B300_PIPE_CATALOG.md` (preserved unchanged).
