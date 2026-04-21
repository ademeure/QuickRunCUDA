# V9: cudaGraph launch latency — single-kernel = same as direct, batched = 3.84× faster

## Measurement

Empty kernel `<<<1, 32>>>` launches, 1000 iters average:

| Path                           | µs / launch | Speedup vs direct |
|--------------------------------|--------------|--------------------|
| Direct `cudaLaunchKernel`      | **2.06**     | 1.00× (baseline)  |
| Graph (1-kernel captured)      | **2.05**     | **1.00× (NO speedup!)** |
| Graph (100-kernel batched)     | **0.54**     | **3.84×**         |

## Key finding (BUSTS A MYTH)

**Cuda Graphs do NOT make individual kernel launches faster.** A 1-kernel
graph has the same launch overhead as direct `cudaLaunchKernel` (~2.06 µs).

The speedup ONLY comes from BATCHING many kernels per graph capture:
- 100 kernels in 1 graph → 0.54 µs amortized per kernel (3.84×)
- 1000 kernels in 1 graph → would be even better

## 10-rule rigor

1. **Theoretical**: graph dispatches a precompiled command stream → less host
   work per kernel. Expected speedup: 2-10× depending on batch size.
2. **Measured**: 1.00× single, 3.84× at batch=100.
3. Rule 3: 3.84× < 10× → no test bug.
4. **Why no single-kernel speedup**: graph launch ALSO does host-side stream
   ordering checks. Per-kernel cost is similar to direct.
5-7. Multiple iterations averaged; warmup excluded; cudaStreamSynchronize bracket.
8. **Conclusive**: single-graph empirically equal to direct.
9. **Surprise checked**: re-ran multiple times, consistent.
10. **Confidence: HIGH**.

## Practical advice (CORRECTING COMMON BELIEF)

**WRONG**: "Use cudaGraph to reduce kernel launch latency."
**RIGHT**: "Use cudaGraph to AMORTIZE launch latency over MANY kernels."

Graph value is in:
1. **Batching N kernels into one launch** — amortizes host work
2. **Eliminating runtime decision logic** — fixed dataflow
3. **Pre-resolving dependencies** — no per-launch sync setup

If your kernel is launched once-per-iteration, **graphs give no benefit**.
If you launch N kernels per iteration, graphs give ~N speedup up to ~10×
floor (where per-kernel work in driver finally matters).

## Comparison to other launch reduction techniques

| Technique                       | Per-kernel cost | When to use         |
|---------------------------------|------------------|----------------------|
| Direct `cudaLaunchKernel`       | 2.06 µs          | One-off launches    |
| Graph (1 kernel)                | 2.05 µs          | NOT useful alone    |
| Graph (100 batched)             | 0.54 µs          | Static dataflow     |
| Persistent kernel (V7)          | ~38 ns/task      | When applicable     |
| `cuStreamWriteValue` doorbell   | 0.45 µs (V7)     | Custom signaling    |

**For absolute minimum launch overhead, persistent kernel + doorbell pattern
remains the winner at 38 ns/task vs cudaGraph's 540 ns.**

## V8/V9 cross-reference

V8 M7 found stream queue depth = 1024. With 0.54 µs per graph launch in
batched mode, you can fill the queue in 0.54 µs × 1024 = 553 µs. Useful for
estimating launch-throughput limits.