# B300 Extended Findings — Session-2 Investigations

This document captures NEW findings from the post-CONSOLIDATED_FINDINGS investigation cycles.
All numbers measured on B300 SXM6 AC, sm_103a at default boost (2032 MHz).

## 1. CPU↔GPU Coordination Latency Ladder (CRITICAL)

The full latency hierarchy for getting work to/from the GPU:

| Mechanism | Latency | Notes |
|-----------|---------|-------|
| __syncthreads (256 thr block) | 14 ns | In-warp/block sync |
| __threadfence_block | 8 ns | Block visibility |
| cluster.sync (any size 2-16) | 190 ns | Constant cost |
| __threadfence (device) | 385 ns | Device-scope |
| __threadfence_system | 861 ns | CPU-visible |
| Cross-block flag wait (one-way) | 790 ns | Real signal latency |
| Persistent kernel + mapped mem | 4 us | Best CPU↔GPU round-trip |
| cudaStreamWaitValue (CPU side) | 6 us | Per call |
| Stream sync per kernel launch | 7.5 us | StreamSync floor |
| cudaMemcpy sync (1 byte) | 3.6 us | Fast path for small data |
| cudaMemcpyAsync (1 byte + sync) | 5.5 us | Per-call overhead |
| Event-based cross-stream sync | 27 us | Worst common pattern |

**Action**: For low-latency dispatch (e.g., LLM token streaming), use persistent kernel + mapped memory (4 us). Beats per-token kernel launches by 2x.

## 2. Concurrent Kernel Limit (CONFIRMED)

- **128 hardware dispatch slots**, NOT 148 SMs
- Up to 128 concurrent single-block kernels run in parallel (98% efficient)
- 144+ kernels: must run in 2 batches → 2x wall time
- Stream priority (`cudaStreamCreateWithPriority`) BYPASSES the 128 queue (verified)

## 3. Memory System

### cudaMemset is the FASTEST way to zero memory
- 7.47 TB/s — at HBM3E write peak
- Manual ulonglong/ulonglong2 stores: only 6.3 TB/s (84% of cudaMemset)
- cudaMemset uses a special HBM fast-path

### cudaMallocAsync is 184-770x faster than cudaMalloc
- Hot reuse (1 MB): 328 ns (essentially pointer bump)
- Default release threshold = 0; reserved baseline = 32 MB
- Cross-stream alloc/free works fine (~380 ns)

### Pageable memory MIGRATES to GPU on touch (B300 PageableMemoryAccess=1)
- malloc'd memory: GPU access at 1.5 TB/s (HBM speed) after first migrate
- Pinned (cudaMallocHost) memory: stays on host, 84 GB/s via PCIe
- Practical: for repeated GPU access, malloc beats pinned!

### Texture cache is now SLOWER than __ldg (no longer beneficial)
- L1 hit: tex 8 TB/s vs ldg/global 15 TB/s (2x SLOWER)
- L2/DRAM: tex 3 TB/s vs ldg 9 TB/s (3x SLOWER)
- Use texture only for filtering/clamping — never for raw fetches

### 4 Copy Engines (PCIe-bound)
- H2D pinned single stream: 57.6 GB/s (~90% PCIe gen5 x16 peak)
- More streams give NO H2D BW improvement
- Full-duplex H2D + D2H: 100.6 GB/s aggregate (87.5% of sum)
- D2D same device (HBM): 3019 GB/s

## 4. Compute Throughput Reality

### FP precision throughput (4-ILP, real measurement)
| Type | TFLOPS | Notes |
|------|--------|-------|
| FP32 FFMA | 63.6 | 83% of 76.96 theoretical peak |
| FP16 HFMA2 | 58.2 | NO 2x speedup over FP32! |
| BF16 BFMA2 | 58.2 | Same as FP16 |
| FP64 DFMA | 1.2 | 1:64 ratio per spec |

**KEY**: FP16/BF16 vector ops do NOT give 2x throughput on B300 in non-tensor mode.
Only tensor cores (mma.sync ~580 TFLOPS, tcgen05 ~1980 TFLOPS) provide the speedup.

### Math intrinsic throughput (ratios to FMA)
| Op | TOPS/s | Slowdown vs FMA |
|----|--------|------------------|
| FMA baseline | 30,266 | 1.0x |
| exp2 | 5,956 | 5.1x slower |
| sin/cos (intrin) | 4,451 | 6.8x slower |
| exp (intrin) | 3,253 | 9.3x slower |
| log (intrin) | 2,732 | 11.1x slower |
| log2 | 1,039 | 29x slower |
| sqrt | 687 | 44x slower |
| div (1.0/x) | 492 | 62x slower |
| rsqrt (frsqrt_rn) | 322 | 94x slower |

**Action**: Prefer exp2 over exp, log2 over log, multiply by reciprocal not divide.

### Integer ops (with runtime constants)
- imad (mul+add): 18 Gops/s
- udiv (32-bit): 1.8 Gops/s (10x slower than imad!)

### Constant memory: 27x slower for divergent access
- Uniform broadcast: 15.9 TB/s
- Per-thread divergent: 0.58 TB/s (27x slower)
- SHMEM uniform reads: 19.4 TB/s (faster than cmem broadcast!)

## 5. Synchronization Costs

### __threadfence ladder (per-fence cost):
- block: 8 ns (cheap)
- device: 385 ns (45x more)
- system: 861 ns (2.2x more)

### Atomic scope (no contention)
- atomicAdd_block on shared: 13.6 ns (fast path)
- atomicAdd / _block / _system on global: ALL 97 ns (no scope difference!)
- Warp-uniform atomic gets coalesced into single transaction

### Thread divergence
- 2 paths: ZERO cost (compiler uses select)
- 4 paths: 6x slowdown
- 8 paths: 11.6x
- 32 paths: 60x slowdown

### Warp shuffle
- shfl.xor / shfl.up: 5.8 TOps/s
- ballot: 3.2 TOps/s (1.8x slower)
- match_any: 0.15 TOps/s (38x slower!)

## 6. Launch Overhead

- Kernel launch + sync floor: **7.5 us** (constant regardless of kernel size)
- Pure async submission: **1.89 us per kernel** (530K kernels/sec submit rate)
- HostFn alone + sync: 1.89 us
- WaitValue is 3 us faster than event sync

### CUDA Graph (HUGE wins for repeated launches)
- Capture: ~0.2 us/node
- Instantiate: 6 us baseline + 0.4 us/node
- Launch beats stream submission by 1.2x@1 node, 3.4x@1000 nodes
- **ExecUpdate (100 nodes)**: 1.4 us
- **Destroy + reinstantiate**: 49.3 us (35x slower)
- Per-node SetParams: 0.30 us (essentially free)

### Argument passing is FREE up to 4 KB
- 12 B args: 7.39 us launch+sync
- 4092 B args: 7.73 us (same)

## 7. NVRTC and Compile

### NVRTC compile cost
- tiny: 5.4 ms
- 100-FMA medium: 5.8 ms
- 5000-FMA large: 23.0 ms
- Debug -G: 4x slower than O3
- sm_80 PTX path: 66 ms (12x slower than native sm_103a)

### Module load
- cubin (any size 5-37 KB): ~10 us
- PTX JIT 5000-FMA (104 KB): 1659 us (155x slower than cubin!)
- cuModuleGetFunction: 39 ns (essentially free)

### Cubin output sizes
- Default: 8360 B for 200-FMA kernel
- Debug -G: 90072 B (10x larger)
- Lineinfo: 22848 B (3x larger)

## 8. Block Scheduling

- ≤148 blocks: each gets unique SM (perfect distribution)
- 296 blocks: exactly 2/SM
- 1000 blocks: 6-7/SM (some imbalance)
- Block 0 lands on SM 142 (last GPC), then round-robins across GPCs with stride 16
- Confirms 8-9 GPCs of width ~16 SMs

## 9. Cluster Launch Specifics

- cluster.sync(2-16): all 190 ns (constant!)
- 8-block cluster placement: spread across 4 GPCs, NOT same GPC
- Cluster does NOT provide GPC locality on B300 — only DSMEM and sync benefit

## 10. Resource Limits Confirmed

- Max regs/SM: 65,536
- Max threads/SM: 2048 (NOT 4096)
- Max threads/block: 1024
- Max blocks/SM: 32 (small blocks)
- Max opt-in SHMEM/block: 232,448 B (227 KB)
- L1+SHMEM unified pool: 256 KB per SM
- L2 cache: 126 MB (verified by access-pattern cliff)
- Max persisting L2: 79 MB (62.5% of L2)

### SHMEM-per-block vs occupancy (256 thr blocks)
- 0-16 KB: 8 blocks/SM (100% occupancy)
- 32 KB: 6 blocks/SM (75%)
- 56 KB: 4 blocks/SM (50%)
- 100 KB: 2 blocks/SM (25%)
- 128+ KB: 1 block/SM (12.5%)

### Register pressure cliffs (256 thr blocks)
- 32 regs: 100% occ
- 40 regs: 75% (just 8 extra regs costs 25% occ)
- 56 regs: 50%
- 96 regs: 25%
- 168+ regs: 12%

## 11. CRITICAL: Clock Locking Paradox

**`nvidia-smi -lgc 2032` paradoxically pins to 1920 MHz** (NOT 2032).

| Mode | Actual Clock | FFMA TFLOPS |
|------|--------------|-------------|
| Default boost (no lock) | 2032 MHz | 63.6 |
| `nvidia-smi -lgc 2032` | **1920 MHz** | 60.1 |
| `nvidia-smi -lgc 1410` | 1410 MHz | 44.3 |
| `nvidia-smi -rgc` reset | 2032 MHz | 63.6 |

Linear scaling preserved at 0.0313 TFLOPS/MHz across all modes (83% MFU).

**ALL TFLOPS measurements should explicitly state which clock state.**
This is the source of "1920 vs 2032 MHz" inconsistencies in the catalog.

