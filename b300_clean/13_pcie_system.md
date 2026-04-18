# 13 — PCIe + System Topology + Device Attributes

**GPU:** NVIDIA B300 SXM6 AC, sm_103a (CC 10.3), 148 SMs, 2.032 GHz boost, 287.4 GB HBM3E.
**Host:** AMD EPYC 9575F, 60-core, single NUMA node (0-59 visible to each GPU).
**Software:** NVIDIA driver 580.126.09, CUDA driver 13.0 (`cudaDriverGetVersion=13000`), CUDA runtime 13.2 (`cudaRuntimeGetVersion=13020`), cuBLAS 13.4.
**Topology:** 2 × B300 connected via NVLink v7 (NV18, 18 lanes) — see `multigpu/` for cross-GPU numbers; this doc is single-card host↔device.

---

## Headline numbers

| Quantity | Value | % of theory | Confidence |
|---|---:|---:|---|
| **PCIe link state (NVML)** | **Gen 6 x16, 64 GT/s** | — | **HIGH** (NVML + lspci `LnkSta: Speed 64GT/s, Width x16`) |
| **PCIe H2D peak (pinned, ≥64 MB)** | **57.7 GB/s** | **23% of Gen 6 spec / 90% of Gen 5 spec** | **HIGH** (3 sources agree: 57.5–57.7) |
| **PCIe D2H peak (pinned, ≥64 MB)** | **57.4 GB/s** | **22% of Gen 6 spec / 90% of Gen 5 spec** | **HIGH** |
| **Full-duplex H2D + D2H aggregate** | **98.8–100.6 GB/s** | **77% of Gen 5 full-duplex 128 GB/s** | **HIGH** (1.72× sum of either alone) |
| **Async copy engines** | **4** (`cudaDevAttrAsyncEngineCount`) | — | HIGH |
| **D2D same device (cudaMemcpyAsync, 2 GB)** | **3279 GB/s** (3.28 TB/s) | 45% of HBM 7.30 TB/s | HIGH |
| **D2D same device (256 MB)** | 3005 GB/s | 41% | HIGH |
| **Small-copy floor — H2D 1 B sync** | **3.6 µs** | — | HIGH |
| Small-copy floor — H2D async + sync (4 KB) | 6.5 µs | — | HIGH |
| Small-copy floor — D2H async + sync (4 KB) | 9.0 µs | — | HIGH (D2H 37% slower; reads need ack, writes are posted) |
| Pageable H2D (touched-page memcpy, 64 MB) | 38.0 GB/s | 66% of pinned | HIGH |
| Power: max limit (NVML) | **1100 W** (NOT 700 W) | — | HIGH |
| Power: min limit (NVML) | 200 W | — | HIGH |
| Power: idle baseline | ~180–197 W | — | HIGH |
| ECC | always on (1/16 bus reserved) | — | HIGH |

---

## PCIe Gen 5 vs Gen 6 — the link is Gen 6, the throughput is Gen 5

**Confirmed Gen 6 negotiation:**
- NVML `nvmlDeviceGetCurrPcieLinkGeneration` = 6, width x16.
- NVML `nvmlDeviceGetMaxPcieLinkGeneration` = 6, width x16.
- `lspci -vv` `LnkSta: Speed 64GT/s, Width x16` (Gen 6 PAM4 = 64 GT/s vs Gen 5 NRZ = 32 GT/s).
- PCIe replays: 0 (stable link).

**But effective bandwidth caps at Gen 5 speeds:**
- Gen 6 x16 theoretical: ~256 GB/s/direction (PAM4, 64 GT/s × 16 lanes × ~1 B/transfer with overhead).
- Gen 5 x16 theoretical: ~64 GB/s/direction.
- Measured: **57.7 GB/s** = 90% of Gen 5, 23% of Gen 6.

**Hypotheses (none verified — needs Gen 6 host):**
1. Host PCIe slot / root complex / retimer negotiates Gen 6 PHY but data path runs Gen 5 (BMC/SBIOS configuration).
2. AMD EPYC 9575F IOD does not actually deliver Gen 6 DMA rates (Genoa-X / Turin platform PCIe controller).
3. PLX switch / re-driver in chassis is Gen 5 only.

To verify Gen 6 data rates would require a different chassis with confirmed Gen 6 root complex. For now, **treat B300's effective host BW as Gen 5 x16** = 57 GB/s/direction, 100 GB/s full-duplex.

---

## 4 copy engines, but PCIe is the ceiling

`cudaDevAttrAsyncEngineCount = 4`. Multiple H2D streams **do not** improve aggregate H2D BW (all four engines share the single PCIe link):

| n_streams (256 MB H2D each) | Aggregate GB/s |
|---:|---:|
| 1 | 57.6 |
| 2 | 57.6 |
| 4 | 57.7 |
| 8 | 57.7 |

The 4 engines DO matter for **independent direction parallelism**:

| Configuration | Time (256 MB) | BW (GB/s) | Notes |
|---|---:|---:|---|
| H2D alone | 2.33 ms | 57.5 | one engine, one direction |
| D2H alone | 2.35 ms | 57.1 | one engine, opposite direction |
| **H2D + D2H concurrent (different dirs)** | **2.72 ms** | **98.8 (1.72× sum)** | **full-duplex** |
| 2× H2D + 2× D2H (4 streams) | 5.47 ms | 98.2 | no improvement past 1+1 |

**Practical guidance:**
- Use 1 stream for any single-direction H2D or D2H — more is wasted.
- Use 2 streams (1 H2D + 1 D2H) for max bidirectional pipelining.
- For triple-stage pipelines (next-chunk-H2D || compute || prev-chunk-D2H), this delivers ~2× the sequential time (BW-bound case).

---

## Pinned vs pageable: cudaHostRegister beats cudaMallocHost for setup

Allocation overhead (per call, average over many trials):

| API | 4 KB | 16 MB |
|---|---:|---:|
| `cudaMallocHost` + `cudaFreeHost` | 414 µs | 2888 µs |
| `cudaHostRegister` + `cudaHostUnregister` (on `malloc` buffer) | **106 µs** | **750 µs** |

`cudaHostRegister` is **3.9× faster** for pinning at 4 KB and 3.85× at 16 MB. Once the memory is pinned, the actual H2D throughput is identical (~56 GB/s for 16 MB) — both routes establish the same PCIe-mapped page table entries from the GPU's view.

**`cudaMallocHost` allocation rate ~5 GB/s, `cudaHostRegister` ~28–40 GB/s.** Use `cudaHostRegister(malloc()...)` if you already have host data; only use `cudaMallocHost` when you'd allocate fresh anyway.

### `cudaHostAlloc` flag variants — WriteCombined has no GPU-side effect
| Flags | H2D GB/s | D2H GB/s | Notes |
|---|---:|---:|---|
| `Default` | ~57 | ~57 | baseline |
| `WriteCombined` | ~57 | ~57 | **no GPU-side benefit** (CPU-side attribute only) |
| `Mapped` | ~57 | ~57 | + `cudaHostGetDevicePointer` zero-copy works |
| `Portable` | ~57 | ~57 | shareable across contexts |
| `WriteCombined + Mapped` | ~57 | ~57 | combined |

**Don't bother setting `cudaHostAllocWriteCombined` for GPU paths** — both directions use the same root-complex-mapped page tables. WC only changes how the CPU writes to that buffer (uncached, write-combine for streaming).

### Zero-copy mapped H2D
Kernel reading from `cudaHostAlloc(..., cudaHostAllocMapped)` + `cudaHostGetDevicePointer`: **54 GB/s** (≈85% of pinned PCIe peak). Useful for low-latency scatter where you don't want explicit memcpy, but always slower than DMA copy + kernel.

---

## Pageable memory — the "1.5 TB/s pageable" trap

A naive test that does `aligned_alloc → memset → kernel reads pageable buffer` reports up to 1.5 TB/s "pageable" BW. **This is a measurement bug**:
- Once pages are touched, they migrate into HBM the first time the kernel touches them. After migration, the data is in HBM, not host RAM.
- Subsequent access is HBM speed, not PCIe speed.

**Real pageable cudaMemcpy H2D: 38.0 GB/s** = 66% of pinned. Pageable transfers must be staged through a driver-internal pinned buffer, costing the extra copy.

For repeated host-to-device GPU access patterns: malloc + first-touch (which migrates) does end up at HBM speeds — but only because the data left the host. If you need the data to stay on host (e.g. CPU producer / GPU consumer), use pinned + explicit memcpy or zero-copy mapped.

---

## Small-transfer latency floor

| Method | Size | Latency |
|---|---:|---:|
| `cudaMemcpy` (sync) | 1 B H2D | **3.6 µs** |
| `cudaMemcpyAsync` + `cudaStreamSynchronize` | 1 B H2D | 5.5 µs |
| `cudaMemcpyAsync` + sync | 4 KB H2D | 6.5 µs |
| `cudaMemcpyAsync` + sync | 4 KB D2H | 9.0 µs |
| `cudaMemcpyAsync` + sync | 1 B D2H | 8.1 µs |
| `cudaMemsetAsync` (1 B) | — | ~6 µs |
| Empty kernel launch + sync | — | 4.3 µs (event floor) |
| Persistent kernel + mapped memory poll | — | **~4 µs** (best CPU↔GPU round-trip) |
| `cuStreamWriteValue32` (4 B) | — | ~5 µs |

**Sync `cudaMemcpy` is faster than async+sync for tiny payloads** — the async path adds queue + event overhead.

**D2H is consistently 37% slower than H2D at small sizes** (9 µs vs 6.5 µs at 4 KB). Mechanism: H2D writes are posted by the CPU and run async; D2H reads require a round-trip ack from GPU memory.

**For ML inference / token streaming: `persistent kernel + mapped memory polling` ≈ 4 µs round-trip beats `kernel launch per token` (10+ µs) by 2×.** See `EXTENDED_FINDINGS.md` for the recipe.

---

## D2D bandwidth (within same B300, copy engine path)

`cudaMemcpyAsync` D2D (HBM→HBM same device):

| Size | Single-direction | R+W aggregate |
|---|---:|---:|
| 256 MB | 3005 GB/s | 6010 GB/s |
| 2 GB | 3279 GB/s | 6557 GB/s |
| Kernel int4 8-ILP D2D | 2781 GB/s | 5562 GB/s |

D2D via `cudaMemcpyAsync` wins by ~18% over hand-coded kernel copies. The single-direction half-rate (~3.3 TB/s) reflects each byte traversing HBM twice (read source, write dest); aggregate ~6.5 TB/s is 8% short of the unidirectional 7.3 TB/s peak — consistent with the partial-duplex finding (HBM3E shared bus, see `01_hbm_bandwidth.md`).

---

## Device attributes (full table from `cudaDeviceGetAttribute`)

`investigations/B300_all_attributes.txt` has all 115 successfully queried attrs. Highlights:

### Compute capacity
- `ComputeCapabilityMajor.Minor` = **10.3** (sm_103a)
- `MultiProcessorCount` = **148**
- `MaxThreadsPerMultiProcessor` = 2048 (64 warps/SM)
- `MaxBlocksPerMultiprocessor` = 32
- `MaxRegistersPerMultiprocessor` = 65536, `MaxRegistersPerBlock` = 65536
- `WarpSize` = 32
- `SingleToDoublePrecisionPerfRatio` = 64 (FP64 = 1/64 FP32)

### Memory
- `totalGlobalMem` = **287.4 GB** (288 GB nominal HBM3E)
- `MemoryClockRate` = 3996 MHz, `GlobalMemoryBusWidth` = 7680 bit (post-ECC; physical 8192)
- `L2CacheSize` = 132,644,864 B = **126 MB**
- `MaxPersistingL2CacheSize` = 82,903,040 B = **79 MB** (62% of L2)
- `MaxAccessPolicyWindowSize` = 134,217,728 B (128 MB)
- `MaxSharedMemoryPerBlock` = 49,152 B (default 48 KB)
- `MaxSharedMemoryPerBlockOptin` = 232,448 B (227 KB opt-in via `cudaFuncSetAttribute`)
- `MaxSharedMemoryPerMultiprocessor` = 233,472 B (228 KB)
- `ReservedSharedMemoryPerBlock` = 1024 B
- `TotalConstantMemory` = 65,536 B (64 KB)

### Features
- `EccEnabled` = 1 (always on)
- `ConcurrentKernels` = 1, `CooperativeLaunch` = 1, `ClusterLaunch` = 1
- `AsyncEngineCount` = **4** (copy engines)
- `UnifiedAddressing` = 1, `ManagedMemory` = 1, `ConcurrentManagedAccess` = 1
- `PageableMemoryAccess` = 1, but `PageableMemoryAccessUsesHostPageTables` = 0
- `DirectManagedMemAccessFromHost` = 0
- `HostNativeAtomicSupported` = **0** (no NVLink-C2C / pure PCIe variant)
- `HostRegisterSupported` = 1, `HostRegisterReadOnlySupported` = 0
- `CanFlushRemoteWrites` = 0
- `GPUDirectRDMASupported` = 1, `GPUDirectRDMAFlushWritesOptions` = 1, `GPUDirectRDMAWritesOrdering` = 100
- `MemoryPoolsSupported` = 1, `MemoryPoolSupportedHandleTypes` = 9
- `IpcEventSupport` = 1, `TimelineSemaphoreInteropSupported` = 1, `SparseCudaArraySupported` = 1
- `DeferredMappingCudaArraySupported` = 1
- `MemSyncDomainCount` = 4
- `IsMultiGpuBoard` = 0 (each GPU is a separate board; multi-GPU via NVLink fabric)
- `NumaConfig` = 0, `NumaId` = -1, `HostNumaId` = 0, `MpsEnabled` = 0
- `KernelExecTimeout` = 0 (no watchdog — server card)
- `Integrated` = 0, `TccDriver` = 0
- `ComputePreemptionSupported` = 1, `StreamPrioritiesSupported` = 1

### Limits (`cudaDeviceGetLimit`)
- `cudaLimitStackSize` per thread, `cudaLimitPrintfFifoSize`, `cudaLimitMallocHeapSize`, `cudaLimitDevRuntimeSyncDepth`, `cudaLimitDevRuntimePendingLaunchCount`, `cudaLimitMaxL2FetchGranularity`, `cudaLimitPersistingL2CacheSize` — all queryable, see `dev_limits.cu`.

### Power (NVML)
- `nvmlDeviceGetPowerManagementLimitConstraints`: min = **200 W**, max = **1100 W**
- `nvmlDeviceGetPowerManagementDefaultLimit`: typically the max
- **Not 700 W**, not 1.4 kW — the SXM6 AC variant tops out at 1100 W per GPU.
- Idle ~180–197 W; sustained tensor load measured ~490 W; BF16 GEMM peaks ~886 W. Approaching but not hitting the 1100 W cap at 100% utilization.

### Topology / NUMA
- Single NUMA node on AMD EPYC 9575F (Turin / Genoa platform).
- Both GPUs see CPUs 0–59 (60-core CPU).
- `nvidia-smi topo -m`: GPU0 ↔ GPU1 = NV18 (NVLink v7, 18 lanes). See `multigpu/` doc.

---

## Theoretical accounting

- PCIe Gen 5 x16: 32 GT/s × 16 lanes × 128b/130b ≈ 63 GB/s/direction. Measured 57.7 = **91% of Gen 5**.
- PCIe Gen 6 x16: 64 GT/s × 16 lanes × FLIT/PAM4 ≈ 256 GB/s/direction. Measured 57.7 = **22.5% of Gen 6** — the gap is the unverified hypothesis above.
- Full-duplex Gen 5: ~128 GB/s combined; measured 99 GB/s = **77%**.

---

## RETIRED claims

| Old claim | Source | Why retired |
|---|---|---|
| "PCIe Gen 6 delivers 256 GB/s on B300" | naive spec quote | Link negotiates Gen 6 (lspci/NVML), but effective BW = Gen 5 = 57.7 GB/s. Untested whether host bottleneck or PHY-only Gen 6. |
| "PCIe peaks at 39 GB/s (61% of Gen 5)" | catalog `bench_pcie_simple.cu` | Used non-pinned or warm-up too short. Pinned + ≥64 MB hits 57.7 = 90% of Gen 5 reliably across `pcie_audit2`, `pcie_max_bw`, `copy_engines`. |
| "Pageable memory delivers 1.5 TB/s host-access BW" | early `pageable_audit.cu` | First-touch migrates pageable pages into HBM; subsequent kernel reads are HBM-rate, not PCIe. `pageable_verify.cu` corrects: real pageable PCIe = 38 GB/s. |
| "B300 TDP = 700 W" | datasheet inference | NVML reports max power limit = 1100 W per GPU (SXM6 AC). 700 W applies to a different SKU. |
| "cudaHostAllocWriteCombined improves H2D speed" | folklore | Measured: identical to Default for GPU-side BW. WC only affects CPU-side write semantics. |
| "More streams improve H2D bandwidth" | naive concurrency assumption | 4 copy engines exist but all share the single PCIe link. 1 stream = 8 streams = 57.7 GB/s. |
| "B300 has NVLink-C2C / native CPU↔GPU atomics" | catalog assumption | `cudaDevAttrHostNativeAtomicSupported` = 0. This is the pure PCIe variant, not GH200 / GB200 NVL. |
| "NUMA: 2 nodes (one per GPU)" | dual-socket assumption | EPYC 9575F is 1 socket, 1 NUMA node. `cudaDevAttrNumaConfig` = 0. |
| "B300 SXM6 idle = 50 W" | rough estimate | Measured idle baseline 180–197 W (memory refresh + fabric + PCIe). |

---

## Open questions / NEEDS NEW MEASUREMENT

1. **Why does PCIe Gen 6 link cap at Gen 5 throughput?** Need: try same B300 in a chassis with verified Gen 6 root complex (e.g., Intel Sapphire Rapids EE+ or future EPYC). Until tested, cannot attribute the bottleneck to AMD EPYC 9575F vs the chassis switch fabric.
2. **Multi-GPU concurrent PCIe**: is the 57 GB/s shared across both GPUs on the same root complex, or per-GPU? `multigpu/` dual-PCIe test would settle this.
3. **GPUDirect RDMA bandwidth from NIC**: `GPUDirectRDMASupported = 1` but no end-to-end NIC→HBM throughput test in this catalog.
4. **Persistent-kernel-mapped-memory polling latency floor**: 4 µs reported but not characterized as function of polling interval, write/poll cycle structure.
5. **PCIe Gen 6 PAM4 forward error correction overhead**: not directly measurable from CUDA; requires lower-level tooling.

---

## Files of record

- `/root/github/QuickRunCUDA/investigations/copy_engines.cu` — 4 copy engines, multi-stream sweep, full-duplex measurement (commit db45658)
- `/root/github/QuickRunCUDA/investigations/pcie_max_bw.cu` — single-direction sweep
- `/root/github/QuickRunCUDA/investigations/pcie_audit2.cu` — independent verification
- `/root/github/QuickRunCUDA/investigations/version_info.cu` — driver/runtime/PCIe link gen via NVML (commit 862014c)
- `/root/github/QuickRunCUDA/investigations/device_attr_dump.cu` — `cudaDeviceProp` full dump (commit 5733647)
- `/root/github/QuickRunCUDA/investigations/dev_attr_all.cu` — `cudaDeviceGetAttribute` 115-attr probe
- `/root/github/QuickRunCUDA/investigations/dev_limits.cu` — `cudaDeviceGetLimit` enumeration
- `/root/github/QuickRunCUDA/investigations/B300_all_attributes.txt` — captured attribute output
- `/root/github/QuickRunCUDA/investigations/host_register.cu` — cudaMallocHost vs cudaHostRegister allocation cost + memcpy throughput
- `/root/github/QuickRunCUDA/investigations/host_register2.cu` — extended host-register variants
- `/root/github/QuickRunCUDA/investigations/host_alloc_flags.cu` — Default / WriteCombined / Mapped / Portable comparison
- `/root/github/QuickRunCUDA/investigations/pageable_verify.cu` — real pageable BW (refutes 1.5 TB/s)
- `/root/github/QuickRunCUDA/investigations/pageable_audit.cu`, `pageable_audit_v2.cu` — earlier pageable measurements
- `/root/github/QuickRunCUDA/investigations/small_copy.cu` — 1B–1MB latency methods
- `/root/github/QuickRunCUDA/investigations/memcpy_rate.cu` — async vs sync submission rate
- `/root/github/QuickRunCUDA/investigations/memcpy_kind.cu` — explicit kind vs `cudaMemcpyDefault` (UVA)
- `/root/github/QuickRunCUDA/investigations/memcpy_curve.cu`, `memcpy3d.cu` — size sweeps and 3D patterns
- `/root/github/QuickRunCUDA/B300_PIPE_CATALOG.md` §"PCIe Gen5 host↔device bandwidth", "PCIe Gen 6 + ECC + P-state", "PCIe full-duplex concurrency", "Tiny PCIe / D2D transfer latency"
- `/root/github/QuickRunCUDA/investigations/B300_REFERENCE.md` — top-level summary card
- `/root/github/QuickRunCUDA/investigations/CONSOLIDATED_FINDINGS.md` — pinned/pageable comparison table
- `/root/github/QuickRunCUDA/investigations/EXTENDED_FINDINGS.md` — copy engine analysis, persistent-kernel CPU↔GPU pattern
