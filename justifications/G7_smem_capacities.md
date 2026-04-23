# G7 — Per-SM smem capacity + per-CTA opt-in cap — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` L84 (referenced by REVIEW_CHECKLIST G7)
**Method:** `cudaGetDeviceProperties` on this rig (B300 SXM6 AC, GPU 0)

## CLAIM (catalog L84, paraphrased)

> "228 KB hardware max smem per-SM, 200 KB per CTA without opt-in"

## TEST

```cpp
cudaDeviceProp p;
cudaGetDeviceProperties(&p, 0);
```

Compile: `nvcc -arch=sm_103a devprop.cu`. Run on GPU 0.

## MEASURED (cudaDeviceProp on this rig)

| Field | Value |
|-------|------:|
| `name` | NVIDIA B300 SXM6 AC |
| `multiProcessorCount` | 148 |
| `major.minor` | 10.3 (sm_103) |
| **`sharedMemPerBlock`** (default per-CTA cap, NO opt-in) | **49,152 B = 48 KB** |
| **`sharedMemPerBlockOptin`** (opt-in per-CTA max) | **232,448 B = 227 KB** |
| **`sharedMemPerMultiprocessor`** (per-SM hardware max) | **233,472 B = 228 KB** |
| `regsPerBlock` | 65,536 |
| `regsPerMultiprocessor` | 65,536 (= 64K) |
| `maxThreadsPerBlock` | 1,024 |
| `maxThreadsPerMultiProcessor` | 2,048 (= 64 warps) |
| `warpSize` | 32 |
| `l2CacheSize` | 132,644,864 B = **126.5 MB** |
| `memoryBusWidth` | **7,680 bits** |

## VERDICT

✅ **228 KB per-SM hardware max — CONFIRMED EXACTLY.**

✅ **127 MB L2 — CONFIRMED EXACTLY** (matches the audit's "126 MB" claim and corrections-swarm finding of "8 HBM stacks 12-Hi" implying ~126.5 MB usable).

✅ **7680-bit bus — CONFIRMED** (matches corrections-swarm "fused-off SKU" finding).

⚠ **"200 KB per CTA without opt-in" — FALSIFIED.** The actual default cap (`sharedMemPerBlock`) is **48 KB**, not 200 KB. The 227 KB number IS the opt-in `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` ceiling.

The catalog likely conflated:
- **Hardware per-SM max:** 228 KB (`sharedMemPerMultiprocessor`)
- **Opt-in per-CTA max:** 227 KB (`sharedMemPerBlockOptin`) — roughly the same as per-SM since each SM hosts 1 CTA at this size
- **Default per-CTA cap (no opt-in):** 48 KB (`sharedMemPerBlock`)

The "200 KB" number doesn't match any of these — it may be confused with H100's 100 KB-per-block-without-opt-in or some intermediate value.

## NEW finding worth flagging

The 1 KB delta between `sharedMemPerBlockOptin` (227 KB = 232,448 B) and `sharedMemPerMultiprocessor` (228 KB = 233,472 B) is the **driver-reserved smem floor** (1 KB per CTA reserved for register spill / stack / driver bookkeeping). At 1 CTA/SM this means user-addressable smem is 227 KB, not the full 228 KB.

For 2 CTAs/SM at full smem use, you can get 113 KB each (228 / 2 - 1 KB driver overhead). For 4 CTAs/SM, ~56 KB each. These are derived, not observed here.

## Implications

To use the full 227 KB per CTA, your kernel MUST opt in:
```cpp
cudaFuncSetAttribute(my_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 227 * 1024);
my_kernel<<<grid, block, 227*1024>>>(...);  // dynamic smem at launch
```
Without the `cudaFuncSetAttribute`, the kernel cannot allocate more than 48 KB.

## REVIEW_CHECKLIST candidate updates

- [x] G7 hardware per-SM 228 KB max — ✅ CONFIRMED (`sharedMemPerMultiprocessor` = 233,472 B)
- [x] G7 "200 KB per CTA without opt-in" — ❌ FALSIFIED — real default is **48 KB**; opt-in max is 227 KB
- [x] L2 = 126 MB — ✅ CONFIRMED (`l2CacheSize` = 132,644,864 B)
- [x] Bus width 7680-bit — ✅ CONFIRMED
- [x] 148 SMs — ✅ CONFIRMED
- [x] 64K regs/SM — ✅ CONFIRMED
- [x] 64 warps/SM max occupancy — ✅ CONFIRMED (`maxThreadsPerMultiProcessor` = 2048)
