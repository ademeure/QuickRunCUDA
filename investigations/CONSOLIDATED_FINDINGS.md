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

### Atomic Peak — Fully Resolved
- **1005 Gops/s at UNROLL=32** (stride=4B, L2-resident) = 125 G HW packets/s
- Linear ILP scaling: UNROLL=4→126, 8→252, 16→502, 32→1005 Gops/s
- Model: 148 L2 slices × 2.032 GHz / 2.4 cy/packet = 125 G packets/s ✓
- Each L2 slice = 1 atomic unit with ~2.4 cy throughput
- Catalog's "137/273/372/530" all correct for their UNROLL depths
- **L2/DRAM boundary at stride=64B** (footprint crosses 126 MB L2)
- DRAM-bound = 12-38 Gops/s
- Full contention (all → A[0]): 27 Gops/s
- atomicCAS: -43% vs atomicAdd
- u64 atomic: -47% vs u32
- `red.global`: only 5 Gops/s (compiler emits CCTL.IVALL — avoid)
- Global atom latency (dep chain): 1169-1172 cy
- Scope ordering has negligible effect on DRAM-bound throughput
- "2.7× coalescing speedup" claim was actually L2 residency effect

### L2 Peak Bandwidth (Agent 06) — RESOLVED
- **L2 read peak: 17 TB/s chip-wide** (`.cg` L1-bypass, full occupancy)
- With `.ca` hint at WS ≤ 4 MB: up to 20 TB/s (L1 contributes)
- Per-SM rate: 113 GB/s/SM × 148 = 16.7 TB/s theoretical (= measured)
- **L2 hit BW ≈ L2-as-conduit-to-DRAM BW** — same ~17 TB/s either way
- TLP-critical: 128 thr/SM=4.7 TB/s → 1024 thr/SM=17.8 TB/s
- ncu cross-check: 18.1 TB/s (with 5% ncu overhead)
- Catalog's "22-26 TB/s" not reproducible; "30 TB/s" was L1-dominated (`.ca` @ 1 MB WS); "36 TB/s" was actually SHMEM

### DSMEM (Agent 04) — RESOLVED
- **Local SMEM latency: 28 cy/load** (dependent chain, single warp)
- **DSMEM latency: 201-224 cy/load** = **7-8× slower** than local SMEM
- **Throughput ratio (ILP=4)**: DSMEM 63.5 cy/load vs local 7 cy = **9× slower**
- Cluster size (2/4/8) minimally affects DSMEM latency (~201-224 cy)
- SASS: `ld.shared::cluster` compiles to **LD.E (global-scope load)** via PRMT+IMAD address construction
- DSMEM traverses L2/interconnect — hence higher latency than local SMEM crossbar
- Catalog's "0.8% slower" claim was LICM artifact (ptxas hoisted load out of loop, confirmed via SASS)
- Catalog's "4.7× slower" was latency measured via FADD-serialized accumulator (real latency is ~8×)
- **Non-deterministic crashes** observed on DSMEM dependent chains (~50% at cluster=2 w/ 50+ iters)

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

ALL 17 AGENTS NOW COMPLETE (as of 2026-04-17).

---

## Additional Direct Measurements (post-agents)

### Math Function Throughput (Gops/s per SM, 2032 MHz)
| Function | Gops/s/SM |
|---|---:|
| `exp2f` (MUFU.EX2) | **34.9** |
| `__expf` | 30.6 |
| `sqrtf`, `rsqrtf`, `__logf`, `log2f` | 22.5 |
| `__sinf`, `__cosf` | 20.6 |
| `__tanf` | 8.7 |
| `__frcp_rn` (precise) | **3.8** |

### Integer Arithmetic (ILP=8, 4 warps/SM)
| Op | Gops/s/SM |
|---|---:|
| `int add`, `shl` | 84 |
| `int mad.lo`, `mul`, `xor` | 75 |
| `__popc` | 24 |

### Warp Primitives (cycles)
| Op | Cycles |
|---|---:|
| `__ballot_sync` | 2.28 |
| `__syncwarp` | 2.86 |
| `__popc` | 9.7 |
| `__shfl_xor/down/up_sync` | 24 |
| `__shfl_sync` (indexed) | 26 |

### Sync / Fence Costs (cycles)
| Primitive | Cycles |
|---|---:|
| `__syncwarp` | 2.86 |
| `fence.sc.cta` | 8 |
| `membar.cta` | 9 |
| `__syncthreads` (128 thr) | 20 |
| `fence.acq_rel.cta` | 9.3 |
| `fence.acq_rel.gpu` | 271 |
| `membar.gl` | 272 |
| `fence.sc.gpu` | 277 |
| `cluster.sync` (c=2..8) | 353-381 |
| `membar.sys` | 1683 |
| `fence.sc.sys` | 2883 |

### Memory API Costs (per call, warm)
| API | Time | Notes |
|---|---:|---|
| `cudaMallocAsync + Free` | **0.31 µs** | Constant (any size) |
| `cudaMemset 1 MB` | 1.86 µs | Likely uses async engine |
| `cudaMemcpyAsync 4 KB + sync` | 7.1 µs | |
| `cudaMemcpy 4 KB H2D` | 7.7 µs | |
| `cudaMemcpy 4 KB D2H` | 9.0 µs | |
| `cudaMalloc+Free ≤16 MB` | 68 µs | Size-independent for small |
| `cudaMalloc+Free 256 MB` | 217 µs | |
| `cudaMallocHost+Free 4 KB` | **415 µs** | Pinned — 6× SLOWER than cudaMalloc |
| `cudaMallocHost+Free 16 MB` | **2907 µs** | |

### mbarrier Primitives (cycles, single warp)
| Op | Cycles |
|---|---:|
| `mbarrier.init` | 6.4 |
| `mbarrier.arrive` | 14 |
| `mbarrier.test_wait` (passed) | 63 |
| `mbarrier.try_wait` (passed) | 35 |

### SHMEM Atomics (cycles per atomic, single warp)
| Pattern | Cycles |
|---|---:|
| `atomicAdd` uncontended (per-thread) | 4.6 |
| `atomicAdd` warp-contended (HW reduction) | 8.0 |
| `atomicCAS` | 6.1 |

### Register Pressure (ILP=N independent FFMA chains, 148×128)
| Chains (regs/thread) | Peak TFLOPS (chip) |
|---:|---:|
| 4 (12 regs) | 6.7 (under-saturated) |
| 16 (21 regs) | 35.5 |
| 64 (70 regs) | 57.6 |
| **96 (102 regs)** | **61.4 (peak)** |
| 128 (134 regs) | 56.3 (occupancy drops) |
No register spills up to 134 regs/thread.

### PTX Static Support (sm_103a, CUDA 13.2 ptxas)
- ✓ mbarrier, cp.async.bulk (TMA), atom.acq_rel.sys, ld.global.nc
- ✗ wgmma, tcgen05.alloc, tcgen05.fence, atom.seq_cst, barrier.cluster.sync
- tcgen05 DOES work via NVRTC (Agent 03 + my verification)

### TMA cp.async.bulk (GMEM → SHMEM)
- Aggregate: 1633 GB/s (likely under-peaked due to mbarrier wait overhead)

### Cooperative launch overhead: +32 ns vs plain launch (essentially free)

### 128 concurrent kernel slot limit (verified)
- 1-128 streams: 3.65 ms (1 wave)
- 130+ streams: 7.34 ms (2 waves)
- Sharp cliff exactly at 128

### Branch Divergence Cost (1 warp, cy/iter)
| Divergence | Cy/iter | Overhead |
|---|---:|---:|
| None | 4.0 | 1.00× |
| 2-way if/else | 5.3 | 1.3× (compiler PREDICATES — nearly free) |
| 4-way switch | 99.8 | 25× (real divergence) |
| 32-way (lane-unique) | 1157 | 289× (fully serial) |

**Key**: 2-way branches are basically free (compiler uses SELP). 4+ way requires real serialization.

### Local Memory (Spill) Cost
- **~20 cy per LMEM op** (1024 bytes/thread, runtime-indexed)
- LMEM served from L1 cache
- 20× slower than register access

### Bit Manipulation Throughput (1 warp, cy/iter)
| Op | Cy/iter |
|---|---:|
| LOP3.b32 | 4.16 |
| PRMT.b32 | 4.03 |
| BFE.u32 | 8.08 |
- `__brev`, `__clz`, `__ffs` intrinsics: DCE'd in my test (compiler eliminated)

### cuBLAS GEMM Precision Comparison (TFLOPS @ 2032 MHz, 8192³)
| Precision | TFLOPS | Ratio vs BF16 |
|---|---:|---:|
| FP16→FP16 | NOT_SUPPORTED | — |
| FP16→FP32 | 2238 | 1.0× |
| BF16→FP32 | 2234 | 1.0× |
| FP8→FP16 | 4398 | 1.97× |
| FP8→BF16 | 4398 | 1.97× |
- FP16 and BF16 identical throughput
- FP8 ≈ 2× BF16 at large sizes
- Clock state (1920 vs 2032) made <0.2% difference — GEMM is power-limited

### Managed Memory Migration
| Pattern | Throughput |
|---|---:|
| Cold h→GPU (page fault driven) | 6.5 GB/s |
| Warm GPU access (already migrated) | 3352 GB/s |
| Migrate back h←GPU | 6.9 GB/s |
| After `cudaMemPrefetchAsync` | **2409 GB/s** (300× cold!) |
- Reference: pageable `cudaMemcpy` 49 GB/s — still 7× faster than cold managed
- **Prefetch hints ESSENTIAL for managed memory**

### cudaMemcpy Bandwidth Curves (GB/s peak at 256 MB)
| Mode | BW |
|---|---:|
| H2D pinned | 57.6 |
| D2H pinned | 57.3 |
| H2D pageable | 38.0 |
| D2D | 3005 |

H2D/D2H pinned peaks at 91% of Gen5 x16 theoretical (63 GB/s).

### CUDA Graph Launch Speedup (vs direct launches)
| Chain N | Speedup |
|---:|---:|
| 1 | 1.20× |
| 8 | 1.94× |
| 32 | 2.27× |
| 128 | 2.45× |
| 1024 | **2.52×** |
- Per-kernel overhead: direct 2.07 µs → graph 0.84 µs

### B300 Device Attributes (highlights from 115-attribute probe)
- ComputeCapability: 10.3 (sm_103a)
- MaxGridDim: 2.1B × 65535 × 65535
- MaxBlocksPerMultiprocessor: 32
- SingleToDoublePrecisionPerfRatio: 64
- Max Texture 3D: 16384³ (4 TB volume)
- TimelineSemaphoreInteropSupported: 1 (Vulkan interop)
- SparseCudaArraySupported: 1
- DeferredMappingCudaArraySupported: 1
- IpcEventSupport: 1
- MemoryPoolSupportedHandleTypes: 9 (bitmask)
- HostRegisterReadOnlySupported: 0
- CanFlushRemoteWrites: 0
- GPUDirectRDMAFlushWritesOptions: 1
- GPUDirectRDMAWritesOrdering: 100

### Compute Preemption
- `cudaDevAttrComputePreemptionSupported: 1` (supported per attribute)
- Empirically: priority does NOT actively preempt when SMs can hold both kernels
- High+low priority parallel = both run CONCURRENTLY (sharing SMs), not serial

### Power & Thermal Behavior
- **B300 SXM6 TGP**: ~1000 W
- **Idle**: 137 W, 120 MHz, 33°C
- **FFMA stress (~6 sec)**: 400 W, 2032 MHz, 46°C (only 40% TGP)
- **Sustained FP8 GEMM (3 sec, ncu-verified)**: 870 W, 2032 MHz, 55°C
- **Sustained throughput = burst throughput**: 4489 TFLOPS held flat over 60 batches × 0.05 sec each
- ⚠️ **EARLIER "53% throttle" was a MEASUREMENT ARTIFACT** — `sleep_for` in test code inflated wall time, making throughput appear lower than reality
- B300 maintains 2032 MHz boost clock through 3+ sec of FP8 GEMM at 870W
- Did NOT test multi-minute durations — long-term thermal behavior unmeasured

### Pinned Memory: cudaMallocHost vs cudaHostRegister
| Op | 4 KB | 16 MB |
|---|---:|---:|
| `cudaMallocHost+Free` | 414 µs | 2888 µs |
| `cudaHostRegister+Unreg` | **106 µs** | **750 µs** |
- HostRegister 3-4× faster for setting up pinned memory
- Once pinned, both have identical 56 GB/s memcpy throughput
- HostRegister works on existing malloc'd buffers (useful for library interop)
