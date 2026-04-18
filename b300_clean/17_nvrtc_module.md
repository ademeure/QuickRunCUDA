# B300 NVRTC + Module Load + JIT Cache + Tooling Overheads

**Scope:** NVRTC compile cost & options, `cuModule*` load (cubin vs PTX JIT), `cuModuleGetFunction`, cubin output sizes per flag, low-level VMM API (`cuMemCreate` / `cuMemMap` / `cuMemSetAccess`), NVTX zero-cost overhead, CUDA error-check call cost.

**GPU:** NVIDIA B300 SXM6 AC, sm_103a, 148 SMs, default boost 2032 MHz. Driver 580.126.09, CUDA 13.2, NVRTC 13.2.78.

Confidence markers: **HIGH** = direct timer in dedicated investigation, multiple sources agree. **MED** = single investigation, plausible. **LOW** = older catalog claim, not re-measured.

---

## 1. NVRTC compile cost (HIGH, `investigations/nvrtc_cost.cu`, commit 0847c01)

Targeting `--gpu-architecture=sm_103a`, default options, `nvrtcCompileProgram` only (NOT including create/destroy).

| Kernel | Source size | Compile time | Notes |
|---|---:|---:|---|
| Tiny (1 store) | ~70 B | **5.4 ms** | floor — NVRTC framework init |
| Medium (100 FMA) | ~2.3 KB | **5.8 ms** | barely above floor |
| Large (5000 FMA) | ~115 KB | **23.0 ms** | scales with body size |
| 50-fn many-fn | ~6 KB | ~6 ms | function count alone is cheap |

**~5 ms framework floor** below which NVRTC cannot go regardless of source size. Cold first call (NVRTC lazy init) adds **+11 ms** on top per process. Subsequent compiles in the same process pay only the steady-state cost.

### Architecture matters for compile time

| `--gpu-architecture` | Compile (medium 100-FMA) | Notes |
|---|---:|---|
| `sm_103a` (native) | ~6 ms | direct cubin path |
| `sm_100a` | ~6 ms | direct cubin path |
| `sm_90a` | ~6 ms | direct cubin |
| `sm_80` | **~66 ms (12×)** | falls back to PTX → embedded JIT path |
| `sm_70` | similar | same fallback |
| `compute_103` (PTX-only) | ~6 ms | NVRTC produces PTX, no SASS |

**Always target the device's native arch.** Targeting older arch on B300 forces an internal PTX-JIT step that is ~12× slower.

---

## 2. NVRTC option flags (HIGH, `investigations/nvrtc_options.cu`, commit b8b13ce)

Options accepted/rejected and their effect on a 200-FMA medium kernel.

| Option | Compile (us) | Cubin (B) | Status |
|---|---:|---:|---|
| `sm_103a` (baseline) | ~6000 | **8360** | default |
| `-O0` | — | — | **REJECTED — NVRTC error** |
| `-O1` | — | — | **REJECTED** |
| `-O2` | — | — | **REJECTED** |
| `-O3` | — | — | **REJECTED** |
| `--maxrregcount=32` | ~6000 | similar | accepted |
| `--maxrregcount=64` | ~6000 | similar | accepted |
| `-G` (device debug) | **~24,000 (4×)** | **90,072 (10.8×)** | accepted |
| `-lineinfo` | ~6500 | **22,848 (2.7×)** | accepted |
| `--use_fast_math` | ~6000 | similar | accepted (and **already on by default in QuickRunCUDA**, `cuda_helper.h:227`) |
| `--restrict` | ~6000 | similar | accepted |
| `--no-source-include` | ~6000 | similar | accepted |
| `compute_103` (PTX-only) | ~6000 | (PTX out) | accepted |

**Critical: NVRTC does NOT accept the `-O0/-O1/-O2/-O3` flags directly.** Optimization is implicit (always at NVRTC's chosen default). To pass ptxas opts, use `--ptxas-options=...`. `-G` is the only way to get a debug-quality cubin via NVRTC, and it costs 4× compile time + 10× cubin size.

**Practical:** size your cubin cache for `-G` builds at 10× headroom over release. `-lineinfo` is the right "small" debug aid (3× cubin, near-baseline compile time).

---

## 3. Module load: cubin vs PTX-JIT (HIGH, `investigations/module_load.cu`)

`cuModuleLoadData` from in-memory buffer + `cuModuleUnload`, best-of-50.

| Kernel | Cubin (B) | PTX (B) | Load cubin | Load PTX (JIT) | PTX/cubin ratio |
|---|---:|---:|---:|---:|---:|
| Tiny | ~5.5 KB | ~13 KB | **~10 us** | ~250 us | 25× |
| 100-FMA medium | ~10 KB | ~30 KB | **~10 us** | ~700 us | 70× |
| 2000-FMA large | ~37 KB | ~100 KB | **~10 us** | ~1300 us | 130× |
| 5000-FMA xlarge | ~80 KB | ~104 KB | ~10 us | **1659 us** | **155×** |

**cuModuleLoadData on cubin = ~10 us flat, regardless of cubin size** (5–80 KB tested). The driver maps the cubin & symbol table — no compilation, no per-byte cost worth speaking of in the tested range.

**PTX JIT scales with PTX size**, dominated by an internal ptxas invocation inside the driver. At 5000 FMAs PTX it is **155× slower than the cubin path**. Older catalog table at line 11288–11303 cites `cuModuleLoadData ~46 us` (with `cuModuleUnload ~46 us`) for a 23 KB cubin — that is the **load+unload pair**; per-call is the ~10 us in the modern dedicated test.

### `cuModuleGetFunction` — essentially free

| Operation | Cost |
|---|---:|
| `cuModuleGetFunction` (post-load symbol lookup) | **39 ns** (best-of-1000) |
| Older catalog estimate | 107 ns |

Either way: **a hash-table lookup in the symbol table — never a hot-path concern.**

### Modern alternative — `cuLibrary*` (CUDA 12+, MED)

Catalog (line 10975–10995) reports `cuLibraryLoadData + cuLibraryUnload = 14.4 us` (vs `cuModule*` legacy ~94 us pair) and `cuLibraryGetKernel = 13 ns` (vs `cuModuleGetFunction` 107 ns). **Not re-measured in the dedicated investigation**, but the architectural argument (single library handle holding multiple kernels) suggests preferring `cuLibrary*` for production JIT if the workload loads multiple modules in sequence.

---

## 4. VMM API: cuMemCreate / cuMemMap / cuMemSetAccess (HIGH, `investigations/vmm.cu`, commit 764d1c4)

Low-level virtual-memory API used for growable allocations, multi-GPU aliasing, and fine-grained access control. Default `CU_MEM_ALLOCATION_TYPE_PINNED`, `CU_MEM_LOCATION_TYPE_DEVICE`.

### Granularity

| Query | Bytes | Equivalent |
|---|---:|---|
| `CU_MEM_ALLOC_GRANULARITY_MINIMUM` | **2,097,152** | **2 MB** |
| `CU_MEM_ALLOC_GRANULARITY_RECOMMENDED` | **2,097,152** | **2 MB** |

**Both are 2 MB on B300.** Allocations are page-aligned to this granularity — there is no smaller unit, even for a single byte's worth of allocation.

### Per-step cost vs allocation size (best-of-30/100)

| Size | `cuMemCreate` | `cuMemAddressReserve` | `cuMemMap` | `cuMemSetAccess` | Total |
|---|---:|---:|---:|---:|---:|
| 2 MB | **17 us** | <1 us | <1 us | <1 us | ~20 us |
| 16 MB | ~25 us | <1 us | <1 us | <1 us | ~28 us |
| 256 MB | ~140 us | <1 us | <1 us | <1 us | ~143 us |
| 1 GB | **529 us** | <1 us | <1 us | <1 us | ~533 us |

**`cuMemCreate` is the dominant cost and scales linearly with size** (~0.5 us per MB above the floor). **All three other steps are <1 us each — essentially free.** The driver does the actual physical-memory pinning during `cuMemCreate`; the rest is page-table bookkeeping.

### Growable allocation: 256 × 4-MB chunks into 1 GB VA reservation

```
Total time: ~7.9 ms = 256 chunks × 31 us each
```

**~31 us per 4 MB chunk** for the full `cuMemCreate + cuMemMap` per-chunk pattern, plus a single end-of-loop `cuMemSetAccess` over the full range. Use this for allocators that grow on demand without committing the whole virtual range up front.

### Older catalog vs new measurement

Catalog table at line 11082–11102 reports `cuMemCreate(128 MB) = 19 us` and `cuMemSetAccess = 50 us`. The **new dedicated measurement at 16 MB shows the alloc cost is the dominant scaling factor (~17–25 us range), and SetAccess is sub-µs at all tested sizes** when called as a single per-region operation. The older 50 us SetAccess number probably reflects per-mapping descriptor overhead or a different test configuration; trust the dedicated test.

**Practical:**
- For a single large permanent allocation, `cudaMalloc` (~18 us) wins on cost and simplicity.
- Use VMM only when you need: growable backing, multi-device aliasing, peer access without symmetric VA, or fine-grained permission revocation.
- For growable, expect ~31 us per 4 MB chunk added.

---

## 5. NVTX overhead — FREE without profiler (HIGH, `investigations/nvtx_overhead.cu`, commit 3a76c96)

Best-of-1000 with no profiler attached.

| Call | ns/call | vs baseline |
|---|---:|---:|
| `nvtxRangePushA` + `nvtxRangePop` | ~19 ns | baseline (= int++) |
| `nvtxRangeStart` + `nvtxRangeEnd` | ~19 ns | baseline |
| `nvtxMarkA` | ~19 ns | baseline |
| `nvtxRangePushEx` (color + ASCII) | ~19 ns | baseline |
| **(reference: `volatile int x; x++`)** | **~19 ns** | — |

**All NVTX calls hit exactly the noise floor of the timing harness itself.** The NVTX shared library installs no-op stubs by default; ncu/Nsight swaps in real implementations only when the profiler is attached. Older catalog reports as "0 ns" / "0.20 ns / pair" — these all mean **below measurement noise**.

**Practical:** leave NVTX annotations in production code. Mark inference stages, kernel groups, pipeline phases — there is **no runtime cost** when no profiler is attached, and you get a complete timeline whenever `nsys profile` is run.

---

## 6. Error checking — also FREE (HIGH, `investigations/err_check_cost.cu`, commit d5c36c6)

Best-of-1000, idle context.

| Call | ns |
|---|---:|
| `cudaGetLastError()` | **~20 ns** |
| `cudaPeekAtLastError()` | ~20 ns |
| `cudaGetErrorString(cudaSuccess)` | <50 ns |
| `cudaGetDevice()` | <50 ns |
| `cudaDeviceGetAttribute()` | ~50–80 ns |
| `cudaStreamQuery()` (idle) | ~1.2 us (older catalog) |
| `cudaPointerGetAttributes` | ~40–80 ns |

**`cudaGetLastError` and `cudaPeekAtLastError` are thread-local-variable reads — no GPU interaction.** Always call after every kernel launch; the cost is in the noise relative to the launch itself (~2 us). `cudaPointerGetAttributes` is also cheap enough (~50–80 ns) to use freely.

The earlier catalog note "+0.00 us overhead" for error checks added to a kernel launch is correct: at 20 ns vs 2050 ns launch, the overhead is below measurement resolution.

---

## 7. Cross-API summary (the JIT pipeline at a glance)

For a fresh process loading and running one kernel:

| Phase | Cost |
|---|---:|
| `cuInit(0)` + context creation | **~2000 ms** (one-time per process) |
| NVRTC cold first-compile floor | ~11 ms framework init + 5.4 ms compile = **~16 ms** |
| NVRTC steady-state (warm) tiny kernel | **~5.4 ms** |
| NVRTC steady-state medium kernel | ~5.8 ms |
| NVRTC steady-state large kernel (5000 FMA) | ~23 ms |
| `cuModuleLoadData(cubin)` | **~10 us** |
| `cuModuleGetFunction` | **~39 ns** |
| `cuLibraryLoadData` (CUDA 12+, faster path) | ~14 us |
| `cuLibraryGetKernel` | ~13 ns |
| First kernel launch | ~2 us |
| Per-kernel error check | ~20 ns |
| Per-kernel NVTX annotation | ~19 ns |

**Total fresh-process JIT pipeline:** ~2 s (context) + 16 ms (cold compile) + 10 us (load) + 39 ns (lookup) + 2 us (launch) ≈ **~2.02 s for the first kernel**.

**Total warm/cached JIT pipeline:** 5.4 ms (compile) + 10 us (load) + 39 ns (lookup) + 2 us (launch) ≈ **~5.4 ms per unique kernel** — compile dominates.

**For sweeps:** server mode (e.g. QuickRunCUDA's `--server` over FIFO) avoids the 2-second context cost entirely; per-iteration cost reduces to compile + load + launch. **Cache compiled cubins by source hash** to avoid even the 5–6 ms compile.

---

## 8. Findings being RETIRED (with reason)

| Old claim | Source | Why retired |
|---|---|---|
| "NVRTC accepts `-O3` to control optimization" | implied by older docs | **NVRTC rejects `-O0/-O1/-O2/-O3` outright.** The `-O3` line in the help text refers to **ptxas options passed via `--ptxas-options=`**, not direct NVRTC flags. |
| "NVRTC compile = 6 ms steady-state regardless of size" | catalog line 8703 | True only for small/medium kernels (≤100 FMA). 5000-FMA scales to **23 ms**. |
| "cuModuleLoadData ~46 us" | catalog line 11294 | That was load + unload combined for a 23 KB cubin. **Per-call is ~10 us** in the dedicated test (any size 5–80 KB). |
| "cuModuleGetFunction ~107 ns" | catalog line 11296 | Newer measurement: **~39 ns** (best-of-1000). Either way, free. |
| "NVRTC compile 64–700 ms range" | catalog line 16805–16811 | Includes initial CUDA context init in some measurement paths. **Pure NVRTC compile (warm) is 5–23 ms**; the 64+ ms numbers conflate cold-context init or non-native arch fallback. |
| "VMM granularity is 4 KB / page-size" | implied | **2 MB on B300** (HBM page size). Allocations smaller than 2 MB still consume 2 MB of physical & VA. |
| "cuMemSetAccess = 50 us" | catalog line 11092 | Dedicated test: **<1 us at all tested sizes (2 MB → 1 GB)**. Older 50 us figure was per-region with multiple descriptors or different config. |
| "NVTX = 0 ns" | catalog line 10852–10854 | **~19 ns**; the "0" was below the resolution of the older harness, not literally zero. Practical conclusion ("free without profiler") unchanged. |
| "cudaGetLastError = 11 ns" | catalog line 10708 | **~20 ns** in the dedicated test. Both are well below kernel-launch noise; treat as "free". |
| "tcgen05 PTX rejected by NVRTC" | error from external builds | **Wrong direction.** tcgen05 PTX **DOES** compile via NVRTC; it is rejected by some static-ptxas (CUDA 13.2 bug). NVRTC accepts more PTX than static ptxas in this catalog's environment (CONSOLIDATED_FINDINGS.md line 30, 187, 291). |

---

## 9. Open questions / NEEDS NEW MEASUREMENT

1. **`cuLibrary*` modern API end-to-end timing** with the current driver — only legacy `cuModule*` was directly measured in the dedicated investigation. The catalog's claim of 6.5× speedup deserves re-verification.
2. **NVRTC compile scaling above 5000 FMA** — does the curve stay roughly linear in source size, or does it knee somewhere? Important for code-generators that emit very large unrolled bodies.
3. **NVRTC vs nvcc cubin compatibility** — same SASS for same source? Worth a diff for one or two production kernels to confirm the JIT path matches the offline path.
4. **`-G` runtime impact** — beyond the 10× cubin size and 4× compile cost, what is the actual runtime slowdown for a debug-cubin kernel? Anecdotally large; not quantified here.
5. **VMM growable cost amortization** — at what allocator chunk granularity does the per-chunk 31 us become negligible relative to actual workload?

---

## 10. Files of record

- `/root/github/QuickRunCUDA/investigations/nvrtc_cost.cu` — compile cost vs source size, optimization, arch (commit 0847c01)
- `/root/github/QuickRunCUDA/investigations/nvrtc_options.cu` — option flag exploration (commit b8b13ce)
- `/root/github/QuickRunCUDA/investigations/module_load.cu` — cubin vs PTX JIT load
- `/root/github/QuickRunCUDA/investigations/vmm.cu` — VMM API per-step costs (commit 764d1c4)
- `/root/github/QuickRunCUDA/investigations/nvtx_overhead.cu` — NVTX free without profiler (commit 3a76c96)
- `/root/github/QuickRunCUDA/investigations/err_check_cost.cu` — `cudaGetLastError` etc. cost (commit d5c36c6)
- `/root/github/QuickRunCUDA/investigations/EXTENDED_FINDINGS.md` §7 — consolidated NVRTC numbers
- `/root/github/QuickRunCUDA/B300_PIPE_CATALOG.md` lines 8703, 10704, 10846, 10975, 11082, 11288, 13727, 16794, 16841, 19045 — older catalog entries (some superseded)
- `/root/github/QuickRunCUDA/utils/cuda_helper.h` line 227 — confirms `--use_fast_math` is on by default in QuickRunCUDA's NVRTC pipeline
