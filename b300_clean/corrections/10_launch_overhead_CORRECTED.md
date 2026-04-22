# 10 — Launch Overhead, CUDA Graphs, Capture, Events (CORRECTED)

Scope: revises kernel-launch / cudaGraph / persistent-kernel / doorbell
numbers using V9 (graph launch) and V8 / V7 / CLAUDE.md memory findings.
Sections of the original `10_launch_overhead.md` that don't conflict
(grid-size scaling, args, capture costs, sync variants, events, ctx, PDL,
multi-stream noop, env vars, cooperative) are preserved.

## CORRECTED launch / coordination ladder (B300 SXM6, 2032 MHz)

| Mechanism | Cost | Source |
|---|---:|---|
| `<<<1,1>>>` direct CPU enqueue | **1.78 - 1.85 µs** | TRUE_REFERENCE be28c14 / 10 catalog |
| `cudaLaunchKernel` async no-sync | 1.85 µs (grid-invariant 1 → 1 M blocks) | 10 catalog |
| **Single-kernel cudaGraphLaunch** | **2.05 µs (≈ direct, NO speedup)** | **V9_GRAPH_LAUNCH** |
| **100-kernel cudaGraphLaunch (amortized)** | **0.54 µs / kernel = 3.84×** | V9_GRAPH_LAUNCH |
| 1000-kernel cudaGraph (amortized) | 0.56 µs / kernel = 3.7× | 10 catalog |
| `cuStreamWriteValue32` (doorbell) | **0.45 µs** | CLAUDE.md memory / V7 |
| `cuStreamWaitValue32` (already met) | 1.65 µs | 10 catalog |
| Persistent kernel + mapped-mem polling | **4 µs CPU↔GPU one-shot** | TRUE_REFERENCE 584fda6 |
| **Persistent kernel batched task dispatch** | **38 ns / task** | CLAUDE.md V7 memory |
| `cudaMemset` (4 B) | 1.22 µs (faster than noop kernel) | TRUE_REFERENCE be28c14 |
| `cudaMemcpyAsync` submit | 1.2 µs | TRUE_REFERENCE c6e7fc1 |

## CUDA Graphs — corrected mental model

**MYTH BUSTED (V9): cudaGraph does NOT speed up single-kernel launches.**
A 1-node graph is 2.05 µs vs direct 2.06 µs. The only place graphs win is
amortizing host work over MANY kernels per launch.

| Graph size | µs/launch | µs/kernel amortized | Speedup vs direct |
|---:|---:|---:|---:|
| 1 | 2.05 | 2.05 | **1.00× (no benefit)** |
| 10 | 8.23 | 0.82 | 2.5× |
| 100 | 59.4 | 0.59 | 3.5× |
| 1000 | 562 | 0.56 | 3.7× |

### Graph Update path

| Operation | Cost | Speedup |
|---|---:|---:|
| `cudaGraphInstantiate` (10 nodes) | 11.3 µs | baseline |
| `cudaGraphInstantiate` (100 nodes) | 35 µs | baseline |
| `cudaGraphExecKernelNodeSetParams` (1 node) | 0.30 µs | per-node |
| **`cudaGraphExecUpdate` (10 nodes)** | **0.145 µs** | **77× vs reinstantiate** |
| **`cudaGraphExecUpdate` (100 nodes)** | **1.4 µs** | **25× vs reinstantiate** |
| Destroy + reinstantiate (100 nodes) | 49.3 µs | path to AVOID |

Practical "35× speedup" claim from CLAUDE.md memory is consistent with the
77× / 25× range; the exact factor depends on node count.

## RETRACTIONS

1. **"cudaGraph always faster than direct launch"** — WRONG.
   - V9 measured: 1-node graph = 2.05 µs ≈ direct 2.06 µs.
   - Graphs amortize host work; they don't reduce per-launch cost intrinsically.
   - Original 10 catalog row "cudaGraphLaunch 1.20 µs (35% cheaper)" applies only
     to the CPU-enqueue half of the call; the full sync round-trip is identical.
2. **"Cooperative launch overhead = +32 ns"** (CONSOLIDATED_FINDINGS L296)
   — already retired in original 10 catalog. Keep retired.
3. **"WaitValue 3 µs faster than event sync"** — already re-framed in original 10.
   Keep: true for host-call only; full producer→consumer pair equivalent.
4. **"BlockingSync 5–7× slower"** — already re-framed to 25% steady-state.
5. **"2.05 µs invariant launch latency as a HW property"** — already retired
   as event-floor artifact; keep.

## UNRESOLVED

- `CUDA_DEVICE_MAX_CONNECTIONS` claimed "no effect on B300" — original 10
  flagged this for re-verification. Still unresolved.
- `cudaGraphLaunch` from device code (`DeviceLaunch` flag) measured 13.7 µs;
  whether this stacks with cluster launch overhead untested.
- Persistent kernel "38 ns / task" comes from V7; not independently re-verified
  in V8/V9 cycle.
- `cuStreamWriteValue32` 0.45 µs (V7) vs the 2.47 µs in original 10 catalog —
  these may be measuring different things (host-call vs full doorbell pair).
  Needs reconciliation.
- Graph capture for cuBLAS: original catalog says "no speedup, slightly hurts."
  CLAUDE.md project_b300_pitfalls: "cuBLAS needs cudaGraph" for sustained
  measurements. These can BOTH be true (capture for measurement isolation,
  not perf) but should be made explicit.

## Confidence

| Claim | Confidence | Source |
|---|---|---|
| Single-kernel graph = no speedup vs direct | HIGH | V9_GRAPH_LAUNCH 1000-iter avg |
| 100-kernel graph = 3.84× amortized | HIGH | V9 |
| `cudaGraphExecUpdate` 25-77× vs reinstantiate | HIGH | 10 catalog cross-check |
| `cuStreamWriteValue32` 0.45 µs doorbell | MED | V7 memory; not re-verified V9 |
| Persistent batched 38 ns/task | MED | V7 memory only |
| Persistent + mapped-mem 4 µs CPU↔GPU | HIGH | TRUE_REFERENCE 584fda6 |
| 1.85 µs CPU enqueue floor | HIGH | 10 catalog sweep |
