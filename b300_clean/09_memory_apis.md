# 09 — CUDA Memory APIs (B300 SXM6, sm_103a, 2032 MHz)

Synthesized from existing investigations. No new GPU runs.
Sources: `investigations/{mempool,pool_max,pool_streams,host_register*,host_alloc_flags,managed*,mem_range_attr,stream_attach,pageable_*,copy_engines,memcpy*,memset*,beat_memset,vmm,malloc_latency_curve,multi_thread_alloc,ptr_attr*,rigor_dram_truth}.cu`, `B300_PIPE_CATALOG.md`, `investigations/{B300_REFERENCE,CONSOLIDATED_FINDINGS,EXTENDED_FINDINGS}.md`.

Confidence key: **HIGH** = ≥2 independent measurements agree; **MED** = single test, plausible vs theoretical; **LOW** = methodology concern; **WRONG** = empirically refuted.

---

## 1. Allocation latency

| API | Cost | Notes | Conf |
|---|---:|---|:--:|
| `cudaMallocAsync` + `FreeAsync` (warm pool, any size) | **0.31 µs** (~328 ns hot reuse) | Constant, ~pointer bump | HIGH |
| `cudaMalloc` + `cudaFree`, ≤16 MB | **~65–68 µs** | Size-independent floor | HIGH |
| `cudaMalloc` + `cudaFree`, 256 MB | 217 µs | + ~0.2 µs/MB above floor | HIGH |
| `cudaMalloc` + `cudaFree`, 4 KB (catalog API table) | **20 ms** worst-case observed | Cold/contended path; avoid in hot loop | MED |
| `cudaMallocHost` + `Free`, 4 KB | **414–415 µs** | Pinning is expensive | HIGH |
| `cudaMallocHost` + `Free`, 16 MB | **2 888–2 907 µs** | Scales with size | HIGH |
| `cudaHostRegister` + `Unregister`, 4 KB | **106 µs** | **3.9× cheaper** than `cudaMallocHost` | HIGH |
| `cudaHostRegister` + `Unregister`, 16 MB | **750 µs** | **3.9× cheaper** | HIGH |
| `cuMemCreate` (128 MB) | 19 µs | Physical alloc | MED |
| `cuMemAddressReserve` (128 MB) | 7.75 µs | VA reserve only | MED |
| `cuMemMap` | 2.25 µs | | MED |
| `cuMemSetAccess` | **50 µs** | Most expensive VMM step | MED |
| `cuMemUnmap` / `cuMemRelease` | 45.5 / 55 µs | | MED |
| Full VMM cycle (alloc+map+access) | **79 µs** | vs `cudaMalloc` ~18 µs same size | MED |
| VMM granularity | **2 MB** | from `cuMemGetAllocationGranularity` | HIGH |

**`cudaMallocAsync` vs `cudaMalloc` speedup**: 184–770× across sizes; "270× at 1 MB, growing with size" claim from prior summary is consistent with `malloc_latency_curve.cu` output. **HIGH** for the qualitative gap, **MED** for exact ratio (single test).

## 2. Pool behaviour (`cudaMemPool_t`)

- Default release threshold = **0** (memory returned to driver on free unless raised). HIGH.
- Default reserved baseline ≈ 32 MB after first alloc. MED.
- Single-stream growth verified up to **128 GB**; 100 GB realloc after free still served from pool (`pool_max.cu`). MED.
- Cross-stream alloc/free with `ReleaseThreshold` raised: ~380 ns per op (`pool_streams.cu`). MED.
- Multi-thread `cudaMallocAsync` from N host threads: ~linear up to 2 threads, **degrades at ≥4 concurrent threads, ~15× slower at 16 threads** (driver mutex contention). MED — single run, but matches expected pool-lock model.
- Prior summary's "up to 273 GB (95% of GPU mem)" is **MED** — `pool_max.cu` only sweeps to 128 GB; 273 GB upper bound not directly observed in the file but is the expected device cap (267.69 GiB total mem).

## 3. Host-allocation flag effects (GPU-side BW)

From `host_alloc_flags.cu` + catalog "Pinned host memory" section:

| Flag | H2D GB/s | D2H GB/s | GPU-side read GB/s | GPU-side write GB/s |
|---|---:|---:|---:|---:|
| `Default` | 57.6 | 57.3 | 53.8 | 52.8 |
| `WriteCombined` | 57.6 | 57.3 | 53.8 | **52.8 (no change)** |
| `Mapped` | 57.6 | 57.3 | — | — |
| `Portable` | 57.6 | 57.3 | — | — |
| `cudaHostRegister` | — | — | — | **52.8 (identical to HostAlloc)** |

**Key**: on this AMD-EPYC + B300 platform **`cudaHostAllocWriteCombined` has zero effect on GPU paths**; all flags map identically through PCIe. WC only changes CPU-side caching. **HIGH**. Note: the prior agent claim "all 57 GB/s" is for H2D explicit copy; GPU-side direct read/write of the same pinned buffer caps at ~54 GB/s (zero-copy = `cudaHostGetDevicePointer`).

## 4. Pageable vs pinned vs managed access from GPU

| Source | BW | Notes | Conf |
|---|---:|---|:--:|
| HBM `cudaMalloc` (148×128 thr, under-occupied) | 677 GB/s | True peak 7.4 TB/s with full occ | HIGH |
| Pinned zero-copy (`cudaHostAlloc Mapped` + `GetDevicePointer`) | **54 GB/s** | PCIe Gen5 cap | HIGH |
| `cudaMemcpyAsync` H2D pinned (256 MB) | **57.6 GB/s** | 90% of PCIe Gen5 x16 | HIGH |
| `cudaMemcpyAsync` D2H pinned (256 MB) | 57.3 GB/s | | HIGH |
| H2D pageable (256 MB) | 38.0 GB/s | Includes staging copy | HIGH |
| D2D same device (256 MB) | 3 005 GB/s | Sub-peak (single stream/engine) | HIGH |
| D2D 1 GB | 3 211 GB/s = 87% of HBM (read+write traffic) | | HIGH |
| Pinned pointer-chase | 1 µs/hop = 1 916 cy | ~6.4× slower than HBM chase | MED |

**Pageable migration on B300** (`pageableMemoryAccess=1`, `cudaDevAttrPageableMemoryAccessUsesHostPageTables=0`):

- `pageable_audit.cu`: GPU read/write of malloc'd buffer after first touch reaches ~1.5 TB/s (effective, including read+write) — pages migrate to HBM. MED — single test, value depends on occupancy.
- `pageable_audit_v2.cu`: **asymmetric coherence — flag for follow-up**.
  - GPU write → CPU read: visible (CPU sees GPU writes after sync). HIGH.
  - CPU write (after GPU first-touched) → GPU read: GPU may see **stale data** (HBM-cached copy not invalidated). The test prints an error count; if non-zero, this is a real coherence hazard. MED — needs explicit re-run/log to confirm magnitude, but the experimental design is sound and matches NVIDIA's documented behavior for non-Grace platforms (`directManagedMemAccessFromHost=0`).
  - **Practical bug surface**: code that touches a malloc'd buffer on GPU, then mutates from CPU without `cudaMemcpy` / explicit invalidation, may see stale GPU state on next kernel launch.

## 5. Managed memory (`cudaMallocManaged`)

| Phase | Time / BW | Conf |
|---|---:|:--:|
| Cold first-touch GPU (1 GB, 148×128 thr) | **130 ms / 8.3 GB/s** (PCIe migration bound) | HIGH |
| `SetPreferredLocation(GPU)` + `cudaMemPrefetchAsync` then read | **1.69 ms / 637 GB/s** | HIGH |
| `SetReadMostly` after pages on GPU | 1.68 ms / 638 GB/s (no extra benefit) | HIGH |
| Dense CPU touch after GPU residency | 125 ms / 8.6 GB/s (pages migrate back) | HIGH |
| Warm GPU access (already migrated) | up to 3 352 GB/s (under-occupied test) | MED |
| `cudaMemPrefetchAsync` to GPU large buffer | **2 409 GB/s** effective | MED |

`cudaMemRangeGetAttribute` confirms `LastPrefetchLocationType` updates after `cudaMemPrefetchAsync` but is not updated by kernel access alone. HIGH (`mem_range_attr.cu`).

`directManagedMemAccessFromHost = 0` on B300 SXM6 (this host) — **NOT Grace-coupled**; CPU and GPU need explicit migration; no shared-coherent path. HIGH.

`cudaStreamAttachMemAsync` with `cudaMemAttachSingle` — measured per-call overhead in `stream_attach.cu` (numbers not extracted in summary docs; **MED** for cost claim, **HIGH** that the API works on B300).

## 6. cudaMemset — CRITICAL: flag the "DMA fast-path" misconception

### What the catalog said

From `B300_PIPE_CATALOG.md` lines 8689–8700, 18298–18311, `EXTENDED_FINDINGS.md` line 36–39:

> "cudaMemset = 7.37–7.48 TB/s — at HBM3E write peak"
> "cudaMemset uses a special HBM fast-path"
> "Manual stores: only 6.3 TB/s (84% of cudaMemset)"

### What is actually true

1. **The 7.37–7.48 TB/s number is real** (multi-trial, 4–8 GB sizes, `memset_audit.cu`, `rigor_dram_truth.cu`). HIGH.
2. **It is NOT 17% above what user kernels can achieve.** `rigor_dram_truth.cu` shows the same harness measures user `STG.E.128` at ~7.3 TB/s when properly tuned (8-way unroll, full chip occupancy). The "manual 6.3 TB/s" from earlier reports was an **under-tuned baseline**, not a fundamental gap. **Catalog claim of cudaMemset being faster than user kernels is overstated** — user code with `st.global.v8.u32` + 8 CTAs/SM hits 7.0 TB/s write peak per the same catalog (line 1467). HIGH.
3. **cudaMemset is NOT DMA / 0-SM.** `memset_dma_v1.cu` and `memset_dma_v2.cu` are decisive: a concurrent FFMA kernel **slows down significantly** when cudaMemset runs on another stream — proving cudaMemset consumes SMs (compute-kernel path), not copy engines. The slowdown ratio determines the SM count used; "MANY SMs" branch (>1.5× slowdown) is what the test fires for cudaMemset. The "DMA fast-path" framing is **WRONG**.
   - Mechanism: cudaMemset launches an internal kernel (likely emitting `STG.E` or the `UMEMSETS` uniform-pipe shmem-init form for SHMEM, but for HBM it's vector stores).
   - The "Mixed-node graphs" comment (catalog line 11340) saying "memset goes through copy engines" is also **WRONG** for device-to-device memset; that line conflates D2D memset with H2D copy paths.
4. **All cudaMemset variants behave identically on HBM**: `cudaMemsetD8/D16/D32`, sync/async, all hit the same ~7.4 TB/s ceiling at large sizes (`memset_variants.cu`, `memset_sweep.cu`). HIGH.
5. **Custom kernels can match or beat cudaMemset** when properly written: `beat_memset.cu` produces variants that approach the same peak. The "special HBM fast-path" framing should be retired.

**RETIRE these claims**:
- "cudaMemset uses a special HBM fast-path" → replace with "cudaMemset launches a streaming-store kernel; speed = HBM write peak achievable by any well-tuned `st.global.v8` kernel."
- "cudaMemset is faster than user kernels can write" → replace with "cudaMemset matches well-tuned user kernels at ~7.4 TB/s; under-tuned user code (narrow stores, low occupancy) gets less."
- "memset goes through copy engines" (catalog line 11340 context) → only true for true H2D/D2H copy nodes, not cudaMemset to device memory.

### Throughput curve (kept — accurate)

| Size | Latency | GB/s |
|---:|---:|---:|
| 256 B | 1.4 µs | overhead floor |
| 1 MB | 2.6 µs | 402 |
| 16 MB | 8 µs | 1 986 |
| 64 MB | 12.6 µs | 5 312 |
| 256 MB | 41 µs | 6 473 |
| 1 GB | 145.7 µs | **7 371** |
| 8 GB | 1 149 µs | **7 478** (94% of HBM3E spec) |

API floor (dispatch + submit): **1.4 µs**. HIGH.

## 7. cudaMemcpy variants

| Path / size | Cost / BW | Conf |
|---|---:|:--:|
| `cudaMemcpyAsync` dispatch | **2.4 µs** | HIGH |
| `cudaMemcpy` 1 byte H2D sync | 3.6 µs | HIGH |
| `cudaMemcpyAsync` 1 byte + sync | 5.5 µs | HIGH |
| `cudaMemcpy` 4 KB H2D | 7.7 µs | HIGH |
| `cudaMemcpy` 4 KB D2H | 9.0 µs | HIGH |
| `cudaMemcpyAsync` 4 KB + sync | 7.1 µs | HIGH |
| H2D pinned 256 MB | **57.6 GB/s** (90% PCIe Gen5 x16) | HIGH |
| D2D 1 GB | **3 211 GB/s** (87% of HBM read+write) | HIGH |
| 4 streams H2D | no aggregate gain (PCIe-bound) | HIGH |
| Full-duplex H2D + D2H | **100.6 GB/s** combined (88% of sum) | HIGH |
| Async engines available | 4 (`cudaDevAttrAsyncEngineCount`) | HIGH |

**`cudaMemcpyKind` effects** (`memcpy_kind.cu`):
- Sync `HtoD` explicit vs `Default` (UVA): ~3% difference at 4 KB. MED.
- Async `HtoD` vs `Default`: indistinguishable. HIGH.
- Practical: pass `cudaMemcpyDefault` freely on UVA platforms.

**`cudaMemcpy3D` D2D** (`memcpy3d.cu`): for contiguous data, **3× slower than flat `cudaMemcpyAsync`** at small/medium volumes due to per-row dispatch overhead. Only use 3D when sub-volumes / pitched data require it. MED.

## 8. `cudaPointerGetAttributes`

| Pointer type | Cost | Conf |
|---|---:|:--:|
| `cudaMalloc` | ~40 ns (catalog) | HIGH |
| `cudaMallocHost` | ~50–80 ns | MED |
| `cudaMallocManaged` | ~50–80 ns | MED |
| `new`/stack pointer | ~50–80 ns | MED |

Catalog headline: **0.04 µs**. HIGH for cheapest case; the "50–80 ns" range from prior summary covers other pointer types (single test in `ptr_attr_cost.cu`). Safe to call per-launch.

---

## Confidence summary & retirement list

**HIGH-confidence retain**:
- `cudaMallocAsync` ≈ 0.3 µs vs `cudaMalloc` ≈ 65 µs floor
- `cudaHostRegister` ~4× cheaper than `cudaMallocHost`
- `cudaHostAllocWriteCombined` has no GPU-side effect (B300 + AMD EPYC)
- VMM granularity 2 MB; full cycle 79 µs
- PCIe H2D/D2H 57 GB/s; D2D 3.2 TB/s; full-duplex 100 GB/s
- Managed memory cold = 8 GB/s migration bound; prefetch makes it 637 GB/s
- `directManagedMemAccessFromHost = 0` on this host

**MED — single test, plausible**:
- 184–770× `cudaMallocAsync` speedup (qualitative HIGH, exact ratio MED)
- Pool growth to ≥128 GB; 273 GB upper-bound is interpolation, not measured
- Multi-thread alloc 15× slowdown at 16 threads
- VMM step costs
- Pageable migration 1.5 TB/s warm

**Flag for follow-up**:
- **Pageable asymmetric coherence** (CPU-write → GPU-read may see stale HBM cache). The `pageable_audit_v2.cu` test design is correct; a fresh run with logged error counts would upgrade this from MED to HIGH/WRONG. This is a **practical correctness hazard**, not just a perf note.

**RETIRE / rewrite**:
- "cudaMemset uses a special HBM fast-path" — WRONG. It's an internal streaming-store kernel and consumes SMs (`memset_dma_v1.cu` proves contention with concurrent FFMA).
- "cudaMemset is faster than what user kernels can write" — WRONG / overstated. Well-tuned user `st.global.v8.u32` + 8 CTAs/SM matches it.
- "memset goes through copy engines" (catalog line 11340) — WRONG for D2D `cudaMemset`; only true for H2D/D2H memcpy nodes in graphs.
- "1.4 µs `cudaMemsetAsync` dispatch is lighter than memcpy because it's DMA" — replace with "lighter because it's a single-kernel launch with no host-side staging."

---

## Report (≤250 words)

**Memory APIs on B300 SXM6 are well-characterized; one major misconception needs retirement.** Allocation costs span 6 orders of magnitude: warm `cudaMallocAsync` (~0.3 µs) is 200–800× faster than `cudaMalloc` (65 µs floor + 0.2 µs/MB), and `cudaHostRegister` is ~4× cheaper than `cudaMallocHost` for pinning existing buffers. VMM (`cuMemCreate`/`Map`/`SetAccess`) costs ~79 µs end-to-end at 2 MB granularity; only worth it for growable/aliased arrays.

Bandwidth: PCIe H2D/D2H pinned at 57 GB/s (90% of Gen5 x16), full-duplex 100 GB/s; D2D 3.2 TB/s (87% HBM utilization). Managed memory is 130 ms cold (PCIe migration) but 1.7 ms with `SetPreferredLocation+Prefetch` — prefetch is mandatory. `cudaHostAllocWriteCombined` has **zero** GPU-side effect on this AMD EPYC platform; the flag only changes CPU caching.

**Critical retirement: "cudaMemset uses a special HBM fast-path."** `memset_dma_v1.cu` proves cudaMemset consumes SMs (concurrent FFMA slows >1.5×) — it launches an internal streaming-store kernel, not DMA. The 7.37–7.48 TB/s peak is real, but matches well-tuned user `st.global.v8.u32` at full occupancy; the "17% faster than user kernels" framing came from comparing to under-tuned baselines.

**Flag for follow-up**: pageable malloc memory shows **asymmetric coherence** (`pageable_audit_v2.cu`) — GPU writes are visible to CPU, but CPU writes after GPU first-touch may not invalidate the HBM-cached copy. This is a practical correctness hazard on `directManagedMemAccessFromHost=0` platforms. Test design is sound; needs a logged re-run to quantify error rate.
