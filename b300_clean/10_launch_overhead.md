# 10. Launch Overhead, CUDA Graphs, Capture, Events

**Hardware:** B300 SXM6 AC, sm_103a, 148 SMs, 2032 MHz boost. CUDA 13.2, driver 580.126.09.
**Sources:** investigations/17_launch_latency.md (primary deep-dive); B300_PIPE_CATALOG.md (multiple sections); investigations/B300_REFERENCE.md (consolidated table); investigations/EXTENDED_FINDINGS.md section 6; investigations/launch_latency_sweep.cu, graph_*, event_*, capture_*, host_fn, wait_value, ctx_costs, sync_*, dev_flags, kernel_args, grid_constant, stream_*, coop_launch, pdl_realistic.

---

## 1. Per-launch latency floor (measured)

| Phase | Time | Notes |
|---|---:|---|
| CPU enqueue, async, no sync (`cudaLaunchKernel` / `<<<>>>` / `cudaLaunchKernelExC`) | **1.85 µs** | grid-size invariant from 1 to 1 000 000 blocks |
| CPU enqueue with `cudaGraphLaunch` (pre-instantiated) | **1.20 µs** | 35 % cheaper than direct launch |
| GPU event-pair floor (no kernel between Record/Record) | **2.2 µs** | this is the timer floor; below this use `clock64`/`%globaltimer` |
| Empty kernel `<<<1,1>>>` + `cudaStreamSynchronize` | **6.75 – 7.5 µs** | round-trip floor |
| `cudaLaunchHostFunc` alone in a stream | **1.74 – 1.94 µs** | similar to a kernel launch |
| `cudaStreamAddCallback` (legacy) | 1.85 – 2.22 µs | slightly slower than HostFunc |
| Cooperative grid.sync() per phase | 1.82 µs (≤8 syncs) → 3.5 µs (32 syncs) | matches kernel re-launch cost |

Real noop kernel execution measured by `%globaltimer` inside a single CTA = **3-4 ns** (6-8 SM cycles). The entire 4-7 µs in event-bracketed measurements is event floor + stream command processing, NOT kernel work.

**The earlier "2.05 µs invariant" claim is the event-timer floor**, not a true HW dispatch number; it's only valid for grids ≤ ~2048 blocks where all CTAs land in one wave faster than the timer can resolve.

## 2. Launch APIs are equivalent

`<<<>>>`, `cudaLaunchKernel`, `cudaLaunchKernelEx` (and `cudaLaunchKernelExC`) all measure within 0.1 µs of each other; attribute-bearing variants with `numAttrs=0` are a strict alias. Setting `cluster=2`, `priority=-5`, etc. on `cudaLaunchKernelEx` adds <10 ns. Kernel launch + `cudaGetLastError`/`PeekAtLastError` adds ~11 ns — always call after launches, the cost is in the noise.

Adding `cudaLaunchAttributeProgrammaticStreamSerialization` (PSS) on `cudaLaunchKernelEx` shaves ~0.6 µs (28 % faster, 1.47 µs per launch) — the driver gets to reorder launch-prep work while the stream preserves execution order.

## 3. Grid-size scaling

CPU launch call is grid-size-invariant. GPU dispatch IS NOT for very large grids:

| Blocks (32 thr/block) | GPU event time |
|---:|---:|
| 1 – 2 048 | ~5.9 µs (flat, dominated by event floor) |
| 4 096 (knee) | 7.9 µs |
| 100 000 | 56 µs |
| 1 000 000 | 543 µs |

Above ~4 096 blocks, time scales linearly at **~512 ns/block (32-thr CTAs)**, **~692 ns/block (1024-thr CTAs)** — only 1.35× difference for 32× more threads, so **dispatch cost is dominated by CTA count, not threads/CTA**. Steady-state Global Work Scheduler throughput at saturation = **~1.96 M CTAs/s = 1 CTA per ~1038 GPU cycles**.

## 4. Kernel arguments are free up to 4 KB

| Args | Launch+sync (µs) |
|---:|---:|
| 0 | 7.39 |
| 1 int (4 B) | 7.39 |
| 128 B struct | 7.39 |
| 4 KB struct | 7.73 |

Driver places args in a managed buffer; the copy is hidden inside the launch overhead. Don't pre-stage arg structs through global memory if they fit — pass them directly. `__grid_constant__` is functional but provides no measurable speedup vs ordinary value parameters in this size range.

## 5. CUDA Graphs

Per-kernel cost amortizes hard with graph size:

| N nodes | µs/graph | µs/kernel (amortized) | vs direct (2.05 µs) |
|---:|---:|---:|---:|
| 1 | 2.15 | 2.15 | 1.0× |
| 10 | 8.23 | 0.82 | 2.5× |
| 100 | 59.4 | 0.59 | 3.5× |
| 1 000 | 562 | **0.56** | **3.7×** |

Per-node breakdown (10-node graph, default flags):

| Operation | Cost | Notes |
|---|---:|---|
| Stream capture (per node) | **0.2 µs** | essentially free during capture |
| Capture + EndCapture (10 nodes) | ~8 µs | one-shot |
| `cudaGraphInstantiate` (10 nodes) | **11.3 µs** (~0.4 µs/node + 6 µs base) | one-shot per topology |
| `cudaGraphInstantiate` (100 nodes) | ~35 µs | scales linearly |
| `cudaGraphLaunch` (per launch, any size) | constant 1.4 µs CPU + N×0.5 µs dispatch | hot path |
| `cudaGraphExecKernelNodeSetParams` (one node) | 0.30 µs | per-node update |
| **`cudaGraphExecUpdate` (whole graph from template)** | **0.145 µs (10 nodes), 1.4 µs (100 nodes)** | **77× faster than reinstantiate, 20× faster than per-node SetParams for ≥10 nodes** |
| Destroy + reinstantiate (100 nodes) | 49.3 µs | the path to avoid |

**Rule:** if topology is stable and only kernel args change, use `cudaGraphExecUpdate` with a template captured each iter. Reinstantiate only when the graph shape changes.

`cudaStreamBeginCaptureToGraph` (CUDA 12.3+): append captured work to an existing graph at ~111 µs for 10 kernels — same magnitude as classic capture; useful for incremental composition.

### Instantiate flags (10-node graph)

| Flag | Instantiate | Per launch | Use when |
|---|---:|---:|---|
| Default | 0.11 ms | 8.19 µs | baseline |
| `AutoFreeOnLaunch` | **0.01 ms (10× faster)** | 8.19 µs | rapid graph rebuilds with allocations |
| `DeviceLaunch` | 0.04 ms | 13.7 µs (1.7× slower) | graph launched from device code |
| `UseNodePriority` | 0.01 ms | 8.19 µs | always free; set if you use per-node priorities |

### Conditional graph nodes (CUDA 12.3+)

`cudaGraphNodeTypeConditional` works on B300 with `If`, `While`, `Switch`. Device-side while loop measured at **6.16 µs/iter** (kernel + launch + condition check) — same magnitude as host-driven launch, but no host involvement.

### Mixed-node graphs

Memset / memcpy nodes go through copy engines and don't batch like kernel launches. A 10-node mixed graph (5 noops + 5 memsets) = 20.5 µs/launch ≈ 2 µs/op — same as direct dispatch. **Graph savings only apply to kernel nodes.**

### cuBLAS in graphs

Wrapping cuBLAS GEMMs in a graph gave **zero speedup** (cuBLAS already pipelines via streams) — graph capture/instantiate slightly hurts. Use graphs for custom kernel pipelines, not cuBLAS.

## 6. Stream capture modes

`cudaStreamBeginCapture` accepts `Global`/`ThreadLocal`/`Relaxed`. No measurable per-call cost difference observed — `Relaxed` is documented to allow potentially-unsafe operations during capture (enqueueing work to non-captured streams) without erroring. `cudaStreamGetCaptureInfo` returns the active graph + capture status at 0.13 µs (essentially free, safe in tight loops).

## 7. Event flags & comparison

| Variant | Event sync (kernel + record + sync) |
|---|---:|
| `cudaEventCreate` (default, has timing) | 7.27 – 7.38 µs |
| `cudaEventCreateWithFlags(DisableTiming)` | **~5 µs (33 % faster)** |
| `cudaEventCreateWithFlags(BlockingSync)` | sleep CPU; only useful for low-CPU-load workflows |
| `cudaEventCreateWithFlags(Interprocess)` | required for IPC events |

`cudaEventRecord` host-call cost: ~0.97 µs. `cudaEventQuery` (completed): 1.24 µs. `cudaStreamWaitEvent` host enqueue: **0.08 – 0.13 µs** (just queues a dependency, actual wait is on GPU). `cudaEventElapsedTime` resolution floor = 4 µs — for shorter intervals use `clock64` or `%globaltimer` inside the kernel.

**Always use `DisableTiming`** for events used purely as dependencies (`cudaStreamWaitEvent`); reserve timing-enabled events for measurement.

## 8. Wait/Write Value (driver API stream memops)

The runtime API was removed in CUDA 12+. Use driver API:

| Call | Cost |
|---|---:|
| `cuStreamWriteValue32` | 2.47 µs |
| `cuStreamWaitValue32` (value already met) | 1.65 µs |
| `cuStreamBatchMemOp` (write + wait) | matches the sum |

**Cross-stream sync benchmark (full producer→consumer pair):**
- `cudaEventRecord` + `cudaStreamWaitEvent`: 0.128 ms
- `cuStreamWriteValue32` + `cuStreamWaitValue32`: 0.128 ms (within noise on B300)
- `cuStreamBatchMemOp`: 0.128 ms (also within noise)

The stronger "WaitValue is 3 µs faster than event sync" finding from earlier appears to be the host-call comparison (`cuStreamWaitValue32` 1.65 µs vs `cudaEventSynchronize` ~5 µs), not the full cross-stream pair which is equivalent. PDL `ProgrammaticEvent` (signal at block-START rather than kernel-end) saves ~5 µs vs `cudaStreamWaitEvent` for true cross-stream chains — see PDL section.

## 9. Synchronization variants

| Method | µs/iter (kernel + sync) |
|---|---:|
| `cudaStreamSynchronize` | **1.28** |
| Spin-poll (`cudaStreamQuery`) | 1.26 |
| `cudaDeviceSynchronize` | 1.38 |
| `cudaEventSynchronize` (timing event) | **7.27 (5.7× slower)** |
| `cudaEventSynchronize` (DisableTiming) | ~5 |

**`cudaStreamSynchronize` is optimal** — driver already uses efficient internal spin-wait. Don't write manual polling loops.

`cudaSetDeviceFlags(cudaDeviceScheduleSpin/Yield/Auto/BlockingSync)` per-device, or `cudaLaunchAttributeSynchronizationPolicy` per-kernel:

| Policy | µs/(launch+sync) | Behavior |
|---|---:|---|
| Spin | 8.50 | busy-wait CPU (lowest latency) |
| Yield | 8.91 | yield to OS |
| Auto | 9.23 | driver chooses |
| BlockingSync | 10.91 | sleep CPU thread (lowest CPU usage) |

Range = 25 % (2.4 µs) between fastest and slowest. **For latency-critical loops use Spin; for background workloads use BlockingSync.** `BlockingSync` is 5-7× slower for very short kernels because the OS wakeup dominates (catalog had a 5-7× claim under more aggressive conditions; under steady state it's only 25 %).

## 10. Environment-variable & device-flag pitfalls

- **`CUDA_LAUNCH_BLOCKING=1`** forces every kernel into a synchronous launch — measured ~4× slower per launch in repeated-launch workloads. Use only for debugging.
- **`CUDA_DEVICE_MAX_CONNECTIONS`** has **no measurable effect on B300** (default 8 software queues). The 128 hardware concurrent-kernel slots are the real limit; software queue count doesn't help past defaults.
- **PTDS (`--default-stream per-thread`)** when 4 host threads launch on the implicit NULL stream: total time drops from 2.51 ms (legacy serializes) to 1.14 ms (2.2× faster). Use it whenever multiple threads share a process.

## 11. Cooperative launch

`cudaLaunchAttributeCooperative=1` enables `cooperative_groups::this_grid().sync()`:

| Pattern | Per-phase cost |
|---|---:|
| Persistent kernel, atomic-counter grid sync (no Cooperative attr) | ~2.2 µs (~4200 cy) |
| Cooperative `grid.sync()` (≤8 syncs) | 1.82 µs |
| Cooperative `grid.sync()` (32 syncs) | 3.51 µs |
| 32-kernel chain (re-launch per sync) | 3.19 µs |

Cooperative grid.sync ≈ kernel re-launch in cost. Wins for ≤8 syncs (lower amortized overhead per sync), loses to kernel chain for ≥16. Cooperative attr also has an **inherent ~50 µs flag-setup cost** for first launch. Plain launch overhead penalty: **+1.5 µs** for cooperative vs regular.

## 12. PDL (Programmatic Dependent Launch)

`griddepcontrol.launch_dependents` and `griddepcontrol.wait` PTX instructions are essentially free; the cost is the launch overhead they're embedded in.

`cudaLaunchAttributeProgrammaticStreamSerialization` saves ~0.5–1 µs/pair in 1-stream chains (28 % faster launch). For real LLM-pipeline patterns (~150 kernels in graphs): graph + PDL saves ~2.2 µs/kernel = 2.7 % time reduction. **Graphs and PDL are SUBSTITUTES, not complements** — both fight the same launch-overhead bottleneck; combined adds only +0.15 µs over either alone.

`cudaLaunchAttributeProgrammaticEvent` saves ~5 µs vs `cudaStreamWaitEvent` for cross-stream chains by signaling at block-START rather than kernel-end. Caveat: best-effort — under SM resource pressure the consumer may still wait for full producer completion.

## 13. Context creation cost

| Operation | Cost |
|---|---:|
| `cuInit(0)` (cold) | 197 ms |
| `cuCtxCreate` | 128 ms |
| `cudaSetDevice` (cold) | **2116 ms** (driver-level fresh init) |
| `PrimaryCtxRetain` + `Release` (round-trip) | **240 ms** |
| `cuCtxPushCurrent` / `cuCtxPopCurrent` / `cuCtxGetCurrent` | **30 ns** (essentially free in steady state) |
| `cudaDeviceReset` | 1400 ms |

A fresh CUDA process pays ~326 ms (init + ctx create) before the first kernel; **server-mode/long-lived daemons amortize this**. Push/Pop/GetCurrent are free — context selection is not a hot-path concern.

## 14. CPU↔GPU coordination latency ladder (full)

For latency-critical CPU↔GPU dispatch (e.g., LLM token streaming):

| Mechanism | Latency |
|---|---:|
| `__syncwarp` / `shfl_sync` | 0.85–1 ns |
| `__syncthreads` (256 thr block) | 14 ns |
| cluster.sync (any size 2–16) | 190 ns |
| `__threadfence` (device) | 385 ns |
| `__threadfence_system` | 861 ns |
| Cross-block flag wait (one-way) | 790 ns |
| **Persistent kernel + mapped memory polling** | **4 µs** ← best CPU↔GPU |
| `cuStreamWaitValue32` (CPU side) | 6 µs |
| `cudaMemcpy` sync (1 byte) | 3.6 µs |
| `cudaMemcpyAsync` (1 byte) + sync | 5.5 µs |
| Stream sync per kernel launch | 7 µs |
| Event-based cross-stream sync | 27 µs |
| Cross-process IPC | 100+ µs |

**Persistent kernel with `ld.acquire.sys.u32` polling on mapped memory beats per-token kernel launches by ~2×** for low-latency dispatch.

## 15. Multi-stream noop concurrency

Launching the same `148×128` noop kernel on N concurrent streams:

| Streams | Wall time | Ratio | Efficiency |
|---:|---:|---:|---:|
| 1 | 4.35 µs | 1.0× | 100 % |
| 2 | 6.69 µs | 1.5× | 65 % |
| 4 | 11.20 µs | 2.6× | 39 % |
| 8 | 20.96 µs | 4.8× | 21 % |
| 16 | 27.30 µs | 6.3× | 16 % |

**Noop kernels serialize through the dispatch pipeline.** True kernel concurrency only emerges when kernels are >~10 µs (long enough to overlap launch-overhead). The 128-hardware-slot concurrency limit applies to running kernels, not to noops that complete instantly.

---

## Confidence

| Claim | Confidence | Verification |
|---|---|---|
| Event floor 2.2 µs, true noop = 3–4 ns | HIGH | direct `%globaltimer` measurement in 17_launch_latency.md |
| CPU launch invariant 1.85 µs across grid sizes | HIGH | sweep 1–1 M blocks; cross-confirmed 3 ways |
| GPU dispatch slope 512 ns/block above 4096 blocks | HIGH | linear fit, globaltimer-confirmed |
| `<<<>>>`/`cudaLaunchKernel`/`cudaLaunchKernelEx` equivalent | HIGH | ≤0.1 µs noise across variants |
| Args free to 4 KB | HIGH | direct sweep |
| `cudaGraphExecUpdate` 35–77× faster than reinstantiate | HIGH | two independent measurements (0.145 µs / 1.4 µs vs 11.3 µs / 49 µs) |
| Stream capture ~0.2 µs per kernel | HIGH | matches multiple capture-cost measurements |
| `DisableTiming` saves 33 % on event sync | HIGH | catalog item #20 in B300_REFERENCE.md |
| Cooperative grid.sync ≈ kernel re-launch | HIGH | matched cost table |
| `cudaSetDeviceFlags(BlockingSync)` 25 % slower steady-state | HIGH | direct table |
| PSS (PDL) 28 % faster launch | MEDIUM-HIGH | catalog measurement; 0.6 µs save in noisy regime |
| `CUDA_DEVICE_MAX_CONNECTIONS` no effect on B300 | MEDIUM | absence of evidence; would need re-test to be definitive |
| `cudaLaunchBlocking=1` 4× slower | MEDIUM | inferred from per-launch sync cost; not the headline 5–7× |
| Cross-stream WaitValue ≈ event sync (full pair) | HIGH | direct 3-way comparison at 0.128 ms each |
| **Sole-finding "WaitValue 3 µs faster than event sync"** | **DOWNGRADED to LOW** | the difference is in CPU-side host call only; full producer→consumer pair is equivalent |
| Context create 240 ms / push-pop 30 ns | HIGH | B300_REFERENCE.md #97; multiple measurements |

## Retirement (claims to drop or re-frame)

1. **"2.05 µs invariant launch latency"** — RETIRE as a HW property. It's the event-timer floor for grids ≤2048 blocks. Cite as: "CPU enqueue is 1.85 µs invariant; GPU event-bracketed time is ~5–7 µs floor (event overhead) for small grids and scales at 512 ns/block above 4 096 blocks."
2. **"Cooperative launch overhead = +32 ns" (CONSOLIDATED_FINDINGS.md L296)** — RETIRE; the catalog deep-dive shows ~1.5 µs penalty from the launch path plus a ~50 µs cooperative-flag setup on first launch.
3. **"WaitValue is 3 µs faster than event sync" (EXTENDED_FINDINGS.md L127)** — RE-FRAME. True for the host-side `cuStreamWaitValue32` call (1.65 µs) vs `cudaEventSynchronize` (5 µs). FALSE for full cross-stream producer→consumer pair (both ≈ 0.128 ms within noise).
4. **"BlockingSync 5–7× slower" (key-findings preamble)** — RE-FRAME. Steady-state penalty is 25 % (10.91 vs 8.50 µs). The 5–7× number applies only to extremely short, back-to-back kernels where OS wakeup dominates.
5. **`CUDA_LAUNCH_BLOCKING=1` "4× slower"** — keep with caveat that exact factor depends on kernel duration; the qualitative impact is large enough that this should never be set in production.
6. **`cudaDeviceMaxConnections`** — keep "no effect on B300" but flag as "needs re-verification with explicit oversubscription test."

---

## Report

**Done.** Synthesized into `/root/github/QuickRunCUDA/b300_clean/10_launch_overhead.md` (15 sections + confidence/retirement). Primary source `investigations/17_launch_latency.md` is high-quality; numbers cross-check across `B300_PIPE_CATALOG.md`, `B300_REFERENCE.md`, and `EXTENDED_FINDINGS.md`.

**Key findings retained with HIGH confidence:**
- CPU launch ≈ 1.85 µs invariant (1 to 1 M blocks); event floor 2.2 µs; true noop kernel = 3-4 ns.
- All launch APIs equivalent (<<<>>>, cudaLaunchKernel, cudaLaunchKernelExC) within 0.1 µs.
- Args free to 4 KB.
- `cudaGraphExecUpdate` is 35-77× faster than reinstantiate.
- `DisableTiming` events: 33 % faster.
- Stream capture: 0.2 µs/node; instantiate ~0.4 µs/node + 6 µs base; launch 0.56 µs/op amortized at N≥1000.
- PrimaryCtxRetain+Release = 240 ms; push/pop = 30 ns.
- Cooperative grid.sync matches kernel re-launch.

**Re-framed/retired:**
1. "2.05 µs invariant" — retire as HW property; it's the event floor.
2. "Cooperative launch +32 ns" — retire, true cost ~1.5 µs/launch + ~50 µs first-time flag setup.
3. "WaitValue 3 µs faster than event sync" — true only for host call; full cross-stream pair equivalent (0.128 ms each).
4. "BlockingSync 5-7× slower" — re-frame to 25 % steady-state.
5. `CUDA_DEVICE_MAX_CONNECTIONS` "no effect" — flag for re-verification.
