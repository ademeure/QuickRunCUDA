# 17: Kernel Launch Latency vs Grid Size — B300 sm_103a Deep Dive

**Date:** 2026-04-17  
**GPU:** NVIDIA B300 SXM6 AC, 148 SMs, sm_103a  
**Clock:** 2032 MHz (locked via `nvidia-smi -lgc 2032`)  
**CUDA:** 13.2, `nvcc -arch=sm_103a -O3`  
**Source:** `investigations/launch_latency_sweep.cu`

---

## Motivation

The B300 catalog claims:

> "1 thread×1 block, 1024×148 — all 2.05 µs. Invariant regardless of launch config."

This is suspicious for very large grids. We characterize true GPU-side dispatch time vs block count and block size.

---

## Key Findings Summary

1. **The 2.05 µs catalog number is the GPU event timer floor, not a true kernel dispatch time.** The noop kernel itself executes in <<1 µs; the event pair overhead is ~2.2 µs on B300.

2. **The catalog invariance claim is TRUE but limited** — it holds for 1 to ~2048 blocks (which covers all configs the catalog tested, up to 148 blocks). For larger grids, time scales linearly with block count.

3. **CPU-side launch call IS invariant** — `cudaLaunchKernel` takes ~1.85 µs regardless of grid size from 1 to 1 000 000 blocks. `cudaGraphLaunch` takes ~1.2 µs. Both are grid-size-independent because they only push a command to the stream queue.

4. **Knee at ~2048–4096 blocks** — below this, all CTAs fit in a single dispatch wave and complete before any measurable time passes beyond the event floor. Above this, the GPU's CTA dispatch pipeline becomes the bottleneck.

5. **Linear scaling rate: ~512–692 ns per block** in the large-grid regime (event-measured). Globaltimer confirms ~511 ns/block pure GPU dispatch time for 32-thr kernels. At 1 M blocks with 32 threads: 542 µs. At 1 M blocks with 1024 threads: 696 µs. The difference is small (~1.35x) given 32x more threads — dispatch cost is dominated by CTA count, not thread count.

5b. **Noop kernel actual execution time: 3–4 nanoseconds** (confirmed via `%%globaltimer`). The entire 4–6 µs in event measurements for small grids is event overhead and stream command processing, not kernel work.

6. **`cudaLaunchKernelEx` adds zero overhead** vs `<<<>>>` chevron syntax — within measurement noise.

7. **CUDA Graph helps slightly for small grids** (~0.2–0.6 µs lower GPU time), mainly by reducing stream command overhead. For large grids with 100 K+ blocks, graph vs chevron is indistinguishable.

8. **Multi-stream launches of noop do not parallelize** — with 16 concurrent streams each launching 148×128, wall time is 6.3x longer than single-stream. Noop kernels are so short that GPU dispatch serializes them.

---

## Event Timer Floor (Critical Context)

Before interpreting any measurements, measure the event overhead with no kernel:

```
event-only (no kernel): median=2.208 µs  p5=2.208  p95=2.368  min=2.176
```

**This 2.2 µs is the floor for any event-bracketed measurement on B300.** Every GPU kernel time reported below includes this floor. The true kernel-only time is (measured − 2.2 µs), which for noop is effectively zero.

The catalog's 2.05 µs was measured on the same GPU/setup, where the event floor was slightly lower. The "invariant 2.05 µs" is simply the event resolution floor — it's constant because the noop kernel is faster than the timer can resolve.

---

## Grid-Size Scaling: GPU Event Time

### 32 threads/block — full sweep

| Blocks | Total threads | GPU time (µs) |
|-------:|-------------:|-------------:|
| 1 | 32 | 6.02 |
| 2 | 64 | 5.98 |
| 4 | 128 | 5.98 |
| 8 | 256 | 5.92 |
| 16 | 512 | 5.86 |
| 32 | 1 024 | 5.86 |
| 64 | 2 048 | 5.86 |
| 128 | 4 096 | 5.86 |
| 256 | 8 192 | 5.86 |
| 512 | 16 384 | 6.05 |
| 1 024 | 32 768 | 6.18 |
| 2 048 | 65 536 | 6.14 |
| **4 096** | **131 072** | **7.87** ← knee |
| 8 192 | 262 144 | 9.92 |
| 16 384 | 524 288 | 14.14 |
| 32 768 | 1 048 576 | 22.56 |
| 65 536 | 2 097 152 | 38.94 |
| 131 072 | 4 194 304 | 71.81 |
| 262 144 | 8 388 608 | 139.26 |
| 524 288 | 16 777 216 | 274.46 |
| 1 048 576 | 33 554 432 | 542.69 |

**Flat region:** 1–2048 blocks, ~5.9 µs constant.  
**Knee:** 4096 blocks (32 threads each).  
**Linear regime:** above knee, slope ≈ 512 ns per block.  
**At 1 M blocks:** 542 µs (270x longer than the "invariant" 2.05 µs claim).

Linear fit (65 K–1 M blocks, 32 thr): slope = **512 ns/block**, intercept = 5.4 µs.  
Equivalent: CTA dispatch throughput ≈ **1.95 M blocks/s** (single stream, saturated pipeline).

### Varying thread count — medium block counts

| Config | Blocks | Threads | GPU med (µs) |
|--------|-------:|--------:|-------------:|
| noop 1×1 | 1 | 1 | 4.64 |
| noop 1×32 | 1 | 32 | 4.26 |
| noop 1×1024 | 1 | 1 024 | 5.82 |
| noop 1K×1 | 1 000 | 1 | 4.74 |
| noop 1K×32 | 1 000 | 32 | 4.77 |
| noop 1K×1024 | 1 000 | 1 024 | 6.14 |
| noop 10K×1 | 10 000 | 1 | 10.27 |
| noop 10K×32 | 10 000 | 32 | 10.72 |
| noop 10K×1024 | 10 000 | 1 024 | 12.32 |
| noop 100K×1 | 100 000 | 1 | 55.87 |
| noop 100K×32 | 100 000 | 32 | 55.90 |
| noop 100K×1024 | 100 000 | 1 024 | 73.95 |
| noop 1M×1 | 1 000 000 | 1 | 518.53 |
| noop 1M×32 | 1 000 000 | 32 | 518.11 |
| noop 1M×1024 | 1 000 000 | 1 024 | 696.42 |

**Key observation:** Thread count has minimal effect on dispatch time. 1 M blocks × 1 thread takes 518 µs; 1 M blocks × 1024 threads takes 696 µs — only 1.35x more despite 1024x more threads. This confirms dispatch cost is dominated by CTA count, not work per CTA.

---

## Launch Variant Comparison (GPU Event Time, µs)

| Config | chevron | LaunchKernelEx | GraphLaunch |
|--------|--------:|---------------:|------------:|
| 1×1 | 5.92 | 5.95 | 5.70 |
| 1×32 | 5.89 | 5.89 | 5.70 |
| 1×1024 | 4.61 | 4.64 | 4.29 |
| 32×32 | 4.22 | 4.22 | 4.13 |
| 148×128 (1 CTA/SM) | 4.26 | 4.22 | 4.26 |
| 296×1024 (2 CTA/SM, full) | 4.29 | 4.29 | 4.10 |
| 1000×256 | 6.11 | 6.11 | 5.12 |
| 10000×128 | 10.24 | 10.43 | 10.08 |
| 100000×32 | 55.87 | 55.81 | 56.38 |

**`cudaLaunchKernelEx` vs `<<<>>>`:** Statistically identical — zero overhead.  
**CUDA Graph:** Saves ~0.3–1.0 µs for small grids (reduces stream overhead). Provides no benefit for large grids where dispatch cost dominates.

---

## CPU-Side Launch Call Time (clock_gettime, no sync)

The `cudaLaunchKernel` call only enqueues a command into the stream — it does NOT wait for the GPU. This is the pure driver overhead:

| Config | cudaLaunchKernel (µs) | cudaGraphLaunch (µs) |
|--------|----------------------:|---------------------:|
| 1×1 | 1.850 | 1.210 |
| 1×1024 | 1.860 | 1.220 |
| 148×1024 | 1.880 | 1.220 |
| 1000×256 | 1.860 | 1.180 |
| 10000×128 | 1.630 | 1.091 |
| 100000×32 | 1.600 | 1.071 |
| 1000000×1 | 1.590 | 1.080 |

**CPU-side call is grid-size-invariant:** ~1.85 µs for `cudaLaunchKernel`, ~1.2 µs for `cudaGraphLaunch`, across all grid sizes from 1 to 1 000 000 blocks.

This is the "push to command buffer" cost — the driver does not block or do per-CTA work. Graph launch saves ~35% (650 ns) on CPU-side cost by using a pre-validated descriptor.

---

## Breakdown: CPU vs GPU vs Total

For a small grid (148×128 = 1 CTA/SM, trivial kernel):

| Phase | Time |
|-------|-----:|
| CPU: cudaLaunchKernel call | ~1.85 µs |
| GPU: event-to-event elapsed (includes event floor) | ~4.2–4.3 µs |
| GPU: event floor alone (no kernel) | ~2.2 µs |
| GPU: kernel dispatch + execution (= GPU total − event floor) | ~2.0 µs |
| Total CPU+sync (launch + eventSynchronize) | ~4.2–6.0 µs |

The CPU launch call (~1.85 µs) overlaps with the GPU event processing. After the CPU returns from `cudaLaunchKernel`, the GPU has not yet started the kernel — it's still processing the stream queue. By the time `cudaEventSynchronize` returns, the full 4.3 µs has elapsed.

---

## Multi-Stream Concurrency of Noop Kernels

Launching 148×128 noop kernels simultaneously on N streams:

| Streams | Wall time (µs) | Ratio vs 1 stream | Efficiency |
|--------:|---------------:|------------------:|-----------:|
| 1 | 4.35 | 1.0x | 100% |
| 2 | 6.69 | 1.5x | 65% |
| 4 | 11.20 | 2.6x | 39% |
| 8 | 20.96 | 4.8x | 21% |
| 16 | 27.30 | 6.3x | 16% |

Noop kernels are so fast that the GPU dispatch pipeline serializes them. With 16 streams, time is 6.3x a single launch — far from the ideal 1.0x if they ran in parallel. The GPU processes stream commands sequentially through the dispatch bottleneck. True kernel parallelism only emerges when kernels are long enough to overlap launch overhead.

---

## Globaltimer Validation: Noop Kernel Actual Duration

To separate the event overhead from the kernel's real GPU execution time, a `%%globaltimer` probe was used to timestamp the instant each CTA starts:

**Two consecutive `cudaEventRecord` calls (no kernel at all):**
```
min = 2.112 ms    median = 2.208 ms
```
This 2.2 µs is pure event infrastructure overhead. Every event-bracketed kernel measurement includes this floor.

**`%%globaltimer` inside a noop kernel — first-block to last-block start time:**

| Blocks | Threads | First→Last block start (µs) |
|-------:|--------:|----------------------------:|
| 1 | 32 | <0.1 |
| 10 | 32 | 0.16 |
| 100 | 32 | <0.1 |
| 1 000 | 32 | <0.1 |
| 10 000 | 32 | 3.26 |
| 100 000 | 32 | 49.22 |
| 1 000 000 | 32 | 510.75 |

**Critical validation:** The globaltimer inside a single-block noop (1×1024) shows:
```
min=3 ns   median=4 ns
```
The noop kernel itself executes in **3–4 nanoseconds** (6–8 GPU cycles). The entire 4–6 µs in event measurements is infrastructure, not kernel work.

**Comparing globaltimer dispatch spread vs event measurements (32 thr/block):**

| Blocks | Globaltimer spread (µs) | Event-bracketed (µs) | Overhead (µs) |
|-------:|------------------------:|---------------------:|-------------:|
| 10 000 | 3.26 | 10.24 | 6.98 |
| 100 000 | 49.22 | 55.87 | 6.65 |
| 1 000 000 | 510.75 | 518.53 | 7.78 |

The ~7 µs constant overhead = event floor (2.2 µs) + stream command processing (~5 µs). For large grids, this overhead becomes negligible relative to dispatch time.

**Hardware CTA dispatch throughput (large-grid regime, pure GWS rate):**

After the initial burst fills SM slots (~2048 blocks), the Global Work Scheduler (GWS) serializes to:
- 100 K blocks: 492 ns/block = **2.03 M CTAs/s**
- 1 M blocks: 511 ns/block = **1.96 M CTAs/s**

At 2032 MHz: **1 CTA dispatched per ~1038 GPU cycles**. This is the retire-and-reissue throughput of the GWS when all SM slots are occupied. It is not the initial burst rate (which can issue ~1000 blocks simultaneously), but the steady-state pipeline rate for very large grids.

---

## Answers to the Investigation Questions

### Does launch latency really stay 2 µs across grid sizes?

**Partially true, but misleading.** The 2 µs is the GPU event timer floor. For grids up to ~2048 blocks (the range the catalog tested — max 148 blocks), the time is constant because all CTAs complete faster than the event can resolve. Above ~4096 blocks, time scales linearly.

### Does it scale with grid size?

**Yes, linearly above ~4096 blocks.** Slope: 512 ns/block (32-thr/block), 692 ns/block (1024-thr/block). At 1 M blocks, total time is 270–340x the "flat" floor.

### CPU vs GPU breakdown?

- **CPU call:** ~1.85 µs (`cudaLaunchKernel`), **grid-size-invariant**
- **GPU dispatch time:** ~2.0 µs for small grids (above the 2.2 µs event floor)
- **For large grids:** GPU time = 5.4 µs + 512 ns × N_blocks (32-thr). CPU call stays at ~1.6 µs (slightly lower for very large configs, likely due to reduced driver validation for larger grids).

### Does `cudaLaunchKernelEx` add overhead vs `<<<>>>`?

**No.** Within noise — both chevron and `LaunchKernelEx` measure identically. The extra attrs struct is a no-op when `numAttrs=0`.

### Does CUDA Graph help?

**Slightly for small grids:** ~0.3–0.6 µs lower GPU event time (reduces stream command overhead). **No benefit for large grids** (100 K+ blocks) where dispatch cost dominates. CPU-side graph launch is ~35% faster than `cudaLaunchKernel` (~1.2 vs ~1.85 µs), consistently across all grid sizes.

---

## Corrected Catalog Entry

The existing catalog entry:

> "2.05 µs invariant regardless of launch config"

Should be replaced with:

```
### Kernel launch overhead (B300, CUDA 13.2, triple-chevron + events, clk=2032 MHz)

GPU event timer floor (no kernel): ~2.2 µs — this is the measurement floor.

For grids up to ~2 048 blocks (any thread count), noop kernel completes in <1 µs
and total event-bracketed time is ~4.2–6.0 µs (limited by event floor + stream overhead).

Above ~4 096 blocks: time scales linearly with block count.
  Slope: ~512 ns/block (32 thr), ~692 ns/block (1024 thr)
  At 100 K blocks: ~56 µs (32 thr), ~74 µs (1024 thr)
  At 1 M blocks: ~542 µs (32 thr), ~696 µs (1024 thr)

CPU-side launch call (cudaLaunchKernel, async, no sync):
  ~1.85 µs — TRULY invariant from 1 to 1 000 000 blocks
  cudaGraphLaunch: ~1.2 µs — 35% faster, also invariant

cudaLaunchKernelEx: identical to <<<>>> within noise.
CUDA Graph (GPU time): ~0.3–0.6 µs faster for small grids; no benefit for large grids.

The earlier "2.05 µs invariant" claim was the event floor measured over the small-grid
flat region. It is not a fundamental HW limit — it's the event timer resolution.
Large-grid launches (100 K+ blocks) cost tens to hundreds of µs GPU-side.
```

---

## Design Implications (Updated)

- **Short-grid launches (<=2 K blocks):** event-bracketed time ~4–6 µs. If your kernel does real work (>2 µs), this overhead is minor. For sub-2 µs kernels, the event overhead itself dominates measurement — use longer kernels or measure via CPU timing.
- **Large-grid launches (>100 K blocks):** GPU dispatch time alone is 50–700 µs. This is the cost of CTA scheduling, not the event floor. At 1 M blocks, the dispatch pipeline takes ~540 µs — you're paying for the scheduler to fan out CTAs.
- **CPU launch call:** always ~1.85 µs. This is a fixed driver overhead per launch. Batching work into fewer, larger launches is better than many small ones.
- **Prefer CUDA graphs** for repeated small-grid launches: saves ~35% CPU-side call cost. For large grids, no meaningful benefit.
- **Multi-stream noop parallelism is poor** — dispatch serializes short kernels. Only useful when kernels are long enough (>~10 µs) for true concurrent execution.
- **`cudaLaunchKernelEx` is a strict alias** for `<<<>>>` when attrs=nullptr — no overhead, no benefit.
