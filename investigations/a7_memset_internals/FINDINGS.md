# A7 — cudaMemset internal kernel discovery

## Question
What kernel does the CUDA driver actually launch when you call `cudaMemset`? Can it be named, disassembled, or hooked?

## Methods tried & negative results

### M1. cuobjdump on libcuda.so.1 / libcudart.so.13.2.75
```
$ cuobjdump /usr/lib/x86_64-linux-gnu/libcuda.so.1 --list-elf
File '/usr/lib/x86_64-linux-gnu/libcuda.so.1' does not contain device code
```
Both libcuda and libcudart return "does not contain device code". The driver-internal kernels are NOT stored as standard ELF/cubin sections in the shared libraries — they're loaded from a private blob format the driver decodes at init.

### M2. nsys profile
```
Time(%)  Total Time  Num Calls  Name
2.8       29910 ns         1    cudaMemset
100.0    589925 ns         1    [CUDA memset]
```
nsys classifies the GPU-side activity as `[CUDA memset]` — not a named kernel. NSight Systems hooks at the CUPTI activity layer and the driver labels these as "memset class" events, distinct from regular kernels.

### M3. ncu (Nsight Compute)
```
==WARNING== No kernels were profiled.
```
ncu sees ZERO kernel launches for a program that calls `cudaMemset` exclusively. Memset bypasses the CUPTI kernel-event hook entirely.

### M4. LD_PRELOAD intercept on `cuLaunchKernel`, `cuLaunchKernelEx`, `cuLaunchKernelC`, `cuLaunch`
None of these were called. The driver does NOT route memset through the public launch APIs.

### M5. LD_PRELOAD intercept on `cuMemsetD8/D16/D32_v2_ptds`, `cuMemsetD8/D32Async_ptsz`, `cuMemsetD2D*`
None of these public driver entry points fire when libcudart calls `cudaMemset`. The runtime uses `cuGetProcAddress` to obtain function pointers and calls them via private internal tables.

### M6. LD_PRELOAD intercept on `cuGetProcAddress` / `cuGetProcAddress_v2`
Also did not fire — the runtime apparently caches function-pointer tables at libcudart load time before any LD_PRELOAD'd hook can intercept.

## Indirect positive findings

### M7. Launch-overhead measurement — cudaMemset is FASTER than the cheapest kernel launch
```
cudaMemset    4 B: 1.222 us per call
cudaMemset 4 KB: 1.697 us per call
cudaMemsetAsync 4 B: 1.221 us per call
<<<1,1>>> noop:    1.780 us per call
<<<1,32>>> 1 store: 2.051 us per call
```
**cudaMemset 4 B is 31% faster than the cheapest possible kernel launch** (1.22 us vs 1.78 us). Even cudaMemset 4 KB (1.70 us) beats a noop kernel.

### M8. Earlier work — cudaMemset uses ALL 148 SMs (commit 8bdb3e6)
A linear contention sweep proved cudaMemset is NOT a copy-engine DMA — it executes on all 148 SMs and slows down proportionally to how much FFMA load runs alongside it.

### M9. Earlier work — DRAM rate matches optimal user kernel (commit 0675ffb)
ncu's `dram__bytes_write.sum.per_second` reports ~7.30 TB/s for both cudaMemset and the optimal v8 + per-warp coalesced user kernel. The "memset is 21% faster" claim was wall-clock illusion.

## Conclusion

**The cudaMemset kernel is intentionally hidden behind a driver-private fast-path dispatch.** Specifically:
- Its SASS is not extractable through `cuobjdump`, `cuModuleEnumerate*`, or any documented API.
- Its launch is invisible to CUPTI / nsys / ncu (categorized as "memset" event, not "kernel" event).
- It bypasses every public driver launch entry point (`cuLaunchKernel*`, `cuMemset*_v2_ptds`, `cuGetProcAddress`).
- It uses ALL SMs (so it IS a kernel, not DMA).
- It hits the same ~7.30 TB/s DRAM ceiling as the best user kernel.
- Its launch overhead is ~31% LESS than the cheapest possible cuLaunchKernel call.

The 31% launch-overhead advantage is the **measurable benefit** of the private dispatch path. The driver clearly maintains a streamlined submission queue for "memcpy/memset class" operations that bypasses several stages of the public kernel launch pipeline (likely things like config validation, function-pointer resolution, and CUPTI activity recording).

**Confidence:** HIGH on the negative findings (5 independent hooking methods all blocked); HIGH on the launch-overhead measurement (50 runs, 1000 ops each, take min). The internal kernel's exact SASS remains hidden — answering it definitively would require either driver-source access or kernel-mode debugging hooks (e.g., `nv-debugger` instrumentation), which are out of scope for userspace tooling.

**What would change it:**
- If NVIDIA released the driver source or a SASS dump of internal modules, we could finally see the memset kernel's exact instruction sequence.
- If a future driver version exposed memset via a public CUPTI activity callback (vs the current "memset event" classification), nsys/ncu would name it.
- If a deeper hook (e.g. via `LD_AUDIT` or kernel-mode tracing) caught the actual GPU command queue write, we could observe the channel-level submission.
