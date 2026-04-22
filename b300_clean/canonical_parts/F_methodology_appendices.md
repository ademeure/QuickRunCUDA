# Section F — Methodology & Appendices

> Sections §56–65 plus Appendices A–E of the **B300 SXM6 AC Canonical
> Reference**. The earlier sections (§1–55) are produced by sibling agents;
> the stitcher concatenates. This part is the methodology spine: how the
> catalog was built, how to build on it, and how to avoid repeating the
> mistakes the rigor sweep already caught.

---

## §56. TMA / cp.async family

**Answer:** TMA bulk read 7.34 TB/s · TMA bulk write 7.17 TB/s · TMA copy R+W
6.21 TB/s combined · TMA 8-deep pipelined read 7.20 TB/s mean-of-5 · TMA
multicast effective 14.91 TB/s (cluster=8) · cp.async stack 3-4× speedup vs
plain LDG.  `[🟢 HIGH · src: 09_memory_apis_CORRECTED.md§ladder + V32/V46/V47/V48]`

The TMA / `cp.async.bulk` family is the most rigorously characterised memory
path on B300. Below is the canonical ladder, the architectural lessons that
held up after waves 1–6 of audit, and the footguns that were only discovered
when someone tried to combine paths.

### §56.1 Canonical ladder

| Path | Peak BW | % of 7.67 TB/s this-device peak | Source |
|---|---:|---:|---|
| Plain LDG.32 coalesced | 1.95 TB/s | 25.4 % | V10 |
| Plain LDG.64 coalesced | 3.65 TB/s | 47.6 % | V10 |
| Plain LDG.128 coalesced | 5.76 TB/s | 75.1 % | V10 |
| Plain STG coalesced | 6.11 TB/s | 79.7 % | V8 I1 |
| `cp.async.ca.shared.global` (LDGSTS) | 6.91 TB/s | 90.1 % | V9_CP_ASYNC_BW |
| TMA single-deep read (`cp.async.bulk` 64 KB) | 6.72 TB/s | 87.6 % | V33 |
| TMA write (`cp.async.bulk.tensor`) | 7.17 TB/s | 93.5 % | V34 |
| TMA copy R+W pipelined | 6.21 TB/s combined | 81.0 % of A6 mix peak | V35/V36 |
| TMA 8-deep pipelined read (16 KB tile) | 7.20 TB/s | 93.9 % | V46 |
| TMA write 8-deep pipelined (V47) | 6.34 TB/s | 82.7 % | V47 — NO BENEFIT |
| TMA multicast aggregate (cluster=8, single-deep) | 14.91 TB/s effective | n/a (multicast) | V32 |
| TMA multicast 2-deep (CAPPED, single engine) | 13.96 TB/s | n/a (multicast) | V48 |
| LDG.E.128 SoL (37888 blocks) | 7.365 TB/s | 96.0 % | 01_hbm §1 |
| NINJA HBM read (v8 + per-warp coalesced) | 7.30 TB/s | 95.2 % | 01_hbm §1 |

The denominator `7.67 TB/s` is the this-device post-ECC peak (8 stacks ×
1024-bit raw × 7680/8192 controller-fused × 8 Gbps/pin ÷ 1.0625 ECC ÷ 8
B/byte). For cross-vendor or spec-sheet comparisons substitute `7.68 TB/s`
(spec-bus 8192-bit). See §60 for the device-property derivation and §6 (HBM
section, sibling agent) for the denominator-history footgun.

### §56.2 Architectural lessons that survived the audit

1. **Reads need pipelining; writes are already async fire-and-forget.**
   Single-deep TMA read tops at 6.72 TB/s (V33). 8-deep mbarrier-pipelined
   TMA read hits 7.20 TB/s (V46 = 93.9 % of this-device peak). Pipelining
   *writes* gives **zero** benefit (V47: 6.34 vs V34's 7.17 TB/s — actively
   worse, because pipelined writes contend with stage refill on the same TMA
   engine and the consumer was already saturating the egress bus). The
   architectural reading: TMA write is fire-and-forget and the issue queue
   alone is enough to feed the egress; reads block until the data actually
   arrives, so you need overlap.

2. **Multicast cannot be pipelined deeper than 1 stage.** Single multicast
   engine per cluster → ceiling = 14.9 TB/s effective at cluster=8 (V32 =
   V48). Adding pipeline depth (V48 2-deep) hurts slightly (13.96 TB/s).

3. **`cp.async.ca` (LDGSTS) is the best non-TMA read path** at 6.91 TB/s
   (90.1 %) — better than plain LDG.128 (75.1 %) because async loads bypass
   register pressure and L1 bank conflicts.

4. **TMA pipelining is depth-limited, not width-limited.** The V46 8-deep
   recipe sweeps tile size from 4 KB to 64 KB and finds 16 KB optimal at
   8-deep. Smaller tiles (4 KB) need 16-deep to hit the same SoL — same
   total in-flight bytes (128 KB) but more issue overhead. Larger tiles (64
   KB) cap at 4-deep (SMEM exhausted at 256 KB working set / CTA) and lose
   ~3 % (7.05 TB/s vs 7.20).

### §56.3 The V46 "98.5 % NEW SoL" framing — denominator artifact

V46's original write-up stated "98.5 % NEW HBM read SoL"; this was a
denominator artifact, not a real architectural breakthrough. The 7.20
TB/s measurement is honest, but the percentage is computed against an
empirical 7.31 TB/s (V32 pure-direction peak), not against the spec-derived
7672 GB/s post-ECC.

Re-anchored:
- 7.20 / 7.31 = **98.5 %** (V46's framing — uses an empirical denominator)
- 7.20 / 7.67 = **93.9 %** (this-device-SKU framing — correct for SoL)
- 7.20 / 7.68 = **93.8 %** (spec-comparable framing — correct for cross-vendor)

The architectural lesson "TMA reads benefit from 8-deep pipelining" is
**still valid** (V46 7.20 > V33 6.72 = +7 % over single-deep). The
"BELOW V44/V45" comparators in the original framing were also wrong:
V44/V45 are SMEM-side measurements, not HBM. See §6 (HBM, sibling) for
the full denominator chronology.

### §56.4 Recipe (V46 pattern, 7.20 TB/s = 93.9 % of this-device peak)

```cuda
// V46 pattern: 8-deep TMA pipeline, 16 KB tiles, mbarrier per stage.
// 148 CTAs (1×SM), 4 issuer warps × 2 in-flight slots per warp.
// Working set ≥ 4 GB defeats L2 (126 MB).
__shared__ alignas(128) uint8_t tile[8][16384];
__shared__ uint64_t mbar[8];

// per-stage:
//   mbarrier.arrive.expect_tx [&mbar[stage]], 16384;
//   cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
//                          [&tile[stage]], [src + off], 16384, [&mbar[stage]];
//   try_wait.parity [&mbar[(stage - 8 + 8) & 7]], parity;   // round-robin
```

### §56.5 cp.async stack hierarchy (3-4× speedup over plain LDG)

`cp.async` (LDGSTS) achieves 6.91 TB/s vs plain LDG.32 at 1.95 TB/s — a
3.5× speedup at the same 4-byte access width. The mechanism is async issue:
LDG demands the load result into a register before the next dependent
instruction can issue, while `cp.async.ca` returns to SMEM without blocking
the issuing thread, freeing it to issue more loads.

**Stack ladder (B300, single CTA, 4 KB working set inside L1):**

| Stack | BW (TB/s) | Speedup vs LDG.32 |
|---|---:|---:|
| LDG.32 chained | 1.95 | 1.0× |
| LDG.128 chained | 5.76 | 3.0× |
| LDG.128 + 8 ILP | 6.85 | 3.5× |
| cp.async.ca (LDGSTS) 4 B | 6.91 | 3.5× |
| cp.async.ca + 8 batches | 6.95 | 3.6× |
| TMA single-deep | 6.72 | 3.4× |
| TMA 8-deep pipelined | 7.20 | 3.7× |

The "3-4×" range covers the practical regime; small writes lose less
because STG was already 6.11 TB/s.

### §56.6 Footguns

**Footgun:** ⚠ TMA + `prefetch.L2` = **−27 % BW** (V42).

V6 I3 originally reported "prefetch.L2 = 1.58× speedup" for `cp.async`
(LDGSTS). This DOES NOT carry over to `cp.async.bulk` / TMA. V42 measured
the combination at **−27 %**: TMA already owns its own DMA path and the
explicit `prefetch.L2` issue stalls forward progress on the TMA engine.

Rule: **never combine bulk TMA with explicit prefetch.L2.** Do combine
LDGSTS with prefetch.L2 (the original V6 I3 finding holds for LDGSTS).

**Footgun:** ⚠ TMA write pipelining gives ZERO benefit, slightly hurts
(V47 6.34 vs V34 7.17). Do not pipeline TMA stores; they are already async
fire-and-forget at the issue port.

**Footgun:** ⚠ Multicast cannot be deepened. Cluster=8 single-deep is the
ceiling (V32 = 14.9 TB/s effective). Anybody claiming "16-deep multicast"
is measuring single-engine queue depth growth, not real overlap. V48
explicitly tested 2-deep and found it slightly worse.

**Footgun:** ⚠ V46's "98.5 % NEW HBM read SoL" headline used an empirical
denominator (7.31 TB/s pure-direction). Properly anchored to spec or
this-device, V46 is 93.9 %, **below** the LDG.E.128 SoL (96.0 %) and
NINJA HBM read (95.2 %). The pipelining win is real; the framing was
wrong. See §6 footgun (sibling agent).

**Footgun:** ⚠ TMA `wait_group(N)` vs `wait_all` for non-bulk cp.async —
flagged as deferred in V9_CP_ASYNC_BW.md, never resolved. If you build
on `wait_group` at depth N, validate that ncu confirms the expected
in-flight count, because the catalog has no anchor.

**Footgun:** ⚠ TMA multicast reads with cluster < 8 not measured (V32/V48
only ran cluster=8). Cluster=4 multicast may behave differently because
the GPC topology is different — see §58 for the topology background.

**Footgun:** ⚠ Working set size matters: at < 126 MB the L2 absorbs the
read and TMA effectively measures L2 BW (~24 TB/s kernel-effective), not
HBM. Use ≥ 4 GB working set with stride-per-iter to defeat L2.

### §56.7 Sources

- `b300_clean/corrections/09_memory_apis_CORRECTED.md` (canonical ladder)
- `b300_clean/V32_TMA_MULTICAST_FINDINGS.md` (multicast ceiling)
- `b300_clean/V41_V48_FINDINGS.md` (V46 pipelined read; V47 write null;
  V48 multicast 2-deep null)
- `b300_clean/corrections/V46_DOUBT_REPORT.md` (denominator audit)
- `b300_clean/corrections/HEADLINE_CORRECTIONS_v5.md` row 4 (re-anchor)
- `b300_clean/corrections/HBM_DENOMINATOR_FINAL.md` (7.67 vs 7.68 dual-cite rule)

---

## §57. Launch overhead — kernel / cudaGraph / cuStreamWriteValue

**Answer:** Direct kernel launch 1.85 µs cold (CPU enqueue floor); cudaGraph
single-node = 2.05 µs (NO speedup vs direct — V9 myth-bust); cudaGraph batch
of 100 = 0.59 µs/kernel amortized (3.5× faster); `cudaGraphExecUpdate`
25–77× faster than re-instantiation; `cuStreamWriteValue32` 0.45 µs (5–6×
cheaper than launching a noop kernel); persistent kernel batched = 38
ns/task; `cudaMemset` 1.22 µs (31 % cheaper than a noop kernel).
`[🟢 HIGH · src: 10_launch_overhead_CORRECTED.md§ladder + V9]`

### §57.1 Launch & coordination ladder

| Mechanism | Cost | Source |
|---|---:|---|
| `<<<1,1>>>` direct CPU enqueue (cold path) | **1.78 – 1.85 µs** | TRUE_REFERENCE be28c14 |
| `cudaLaunchKernel` async no-sync | 1.85 µs (grid-invariant 1 → 1 M blocks) | 10 catalog |
| Single-kernel `cudaGraphLaunch` | **2.05 µs (≈ direct, NO speedup)** | V9_GRAPH_LAUNCH |
| 10-kernel `cudaGraphLaunch` (amortized) | 0.82 µs/kernel = 2.5× | V9 |
| 100-kernel `cudaGraphLaunch` (amortized) | **0.59 µs/kernel = 3.5×** | V9 |
| 1000-kernel `cudaGraphLaunch` (amortized) | 0.56 µs/kernel = 3.7× | 10 catalog |
| `cudaGraphInstantiate` (10 nodes) | 11.3 µs | 10 catalog |
| `cudaGraphInstantiate` (100 nodes) | 35 µs | 10 catalog |
| `cudaGraphExecKernelNodeSetParams` (1 node) | 0.30 µs/node | 10 catalog |
| **`cudaGraphExecUpdate` (10 nodes)** | **0.145 µs = 77× vs reinstantiate** | 10 catalog |
| **`cudaGraphExecUpdate` (100 nodes)** | **1.4 µs = 25× vs reinstantiate** | 10 catalog |
| Destroy + reinstantiate (100 nodes) | 49.3 µs (path to AVOID) | 10 catalog |
| `cuStreamWriteValue32` (host-call only) | **0.45 µs** | CLAUDE.md V7 memory |
| `cuStreamWaitValue32` (already met) | 1.65 µs | 10 catalog |
| Persistent kernel + mapped-mem polling | **4 µs CPU↔GPU one-shot round-trip** | TRUE_REFERENCE 584fda6 |
| Persistent kernel batched task dispatch | **38 ns/task** | CLAUDE.md V7 memory |
| `cudaMemset` (4 B) | **1.22 µs (faster than noop kernel)** | TRUE_REFERENCE be28c14 |
| `cudaMemcpyAsync` submit | 1.2 µs | TRUE_REFERENCE c6e7fc1 |
| Cooperative launch overhead | +32 ns over regular launch | 11 catalog |

### §57.2 The cudaGraph myth and its bust

A widely-cited belief is "cudaGraph always speeds up launches". V9
empirically refutes this for the single-node case. A 1-node graph costs
2.05 µs to launch — within noise of direct `cudaLaunchKernel` at 2.06 µs.
The performance benefit only materialises when the graph batches host work
across many kernels in a single submit-and-sync round trip:

| Graph size | µs/launch | µs/kernel amortized | Speedup vs direct |
|---:|---:|---:|---:|
| 1 | 2.05 | 2.05 | **1.00× (no benefit)** |
| 10 | 8.23 | 0.82 | 2.5× |
| 100 | 59.4 | 0.59 | 3.5× |
| 1000 | 562 | 0.56 | 3.7× |

The asymptote (~0.55 µs/kernel) is the GPU-side scheduler issue cost; the
~1.5 µs gap to a single-launch direct call is the host-side `enqueue +
fence` overhead that gets amortized.

### §57.3 cudaGraphExecUpdate — the hidden gem

If you have a graph whose topology is fixed but the kernel parameters
change (e.g., the buffer pointer rotates between iterations), use
`cudaGraphExecUpdate` instead of destroying and reinstantiating.

| Operation | Cost (100 nodes) | Speedup |
|---|---:|---:|
| Destroy graph + `cudaGraphInstantiate` | 49.3 µs | baseline |
| `cudaGraphExecUpdate` (in-place, 100 nodes) | **1.4 µs** | **35×** |

CLAUDE.md memory `project_b300_session2` documented this as a "35×" win;
V9 ncu measurements confirm the 25–77× range (depending on node count).

### §57.4 cuStreamWriteValue32 — when launching a kernel is overkill

`cuStreamWriteValue32` lets the CPU enqueue a single 32-bit write to
device memory, bypassing the kernel-launch path entirely. At 0.45 µs
host-call cost, this is **5–6× cheaper than launching a 1-thread kernel**
that does the same store. Use cases:

- "Mark slot ready" doorbells in producer-consumer pipelines.
- Updating a flag for a persistent-kernel poll loop.
- Triggering an `cuStreamWaitValue32` on a downstream stream without a
  kernel boundary.

The catalog lists 0.45 µs (V7 memory) and 2.47 µs (10 catalog). These are
NOT contradictory: 0.45 µs is the host-call cost (issue, no wait); 2.47 µs
is the full producer→consumer pair (write on stream A, observe on stream
B with `cuStreamWaitValue32`). State which framing you mean. See
UNRESOLVED in 10_launch_overhead_CORRECTED.md §F for the open
reconciliation.

### §57.5 Persistent kernel: 38 ns/task batched

A persistent kernel — one launch that loops on a CPU-fed task queue
indefinitely — amortizes launch overhead across millions of tasks. The
B300-measured task latency in batched mode is **38 ns/task**
(CLAUDE.md V7 memory; `project_b300_v7_complete`). This is the lower
bound for any "fine-grained CPU↔GPU coordination" workload.

The single-shot CPU↔GPU round-trip via persistent kernel + mapped memory
is **4 µs** (TRUE_REFERENCE 584fda6) — cheaper than `cudaEventSynchronize`
on a kernel completion (~6 µs) by ~30 %.

When NOT to use persistent: cluster-launched kernels (cluster shape is
fixed at launch time, so re-clusterizing requires re-launch); kernels
that need varying register pressure (persistent picks one occupancy at
launch); kernels with strict per-task `__shared__` lifetime needs.

### §57.6 cudaMemset — the surprise (1.22 µs vs 1.85 µs noop kernel)

`cudaMemset` for a 4-byte target costs **1.22 µs**, **31 % faster than a
noop kernel launch at 1.85 µs**. The mechanism:

- `cudaMemset` issues a CE (Copy Engine) descriptor, not a kernel-launch
  descriptor. CE descriptors take a different path through the host
  driver — fewer queue-management hops.
- `cudaMemset` is invisible to ncu (CLAUDE.md V8 finding `2ead93a`):
  ncu's launch-counter does not increment, but `dram__bytes_write.sum`
  does increment. This is a useful feature when you want to "warm" L2
  / DRAM between timed iterations without contaminating ncu pipe
  metrics.

### §57.7 Footguns

**Footgun:** ⚠ "cudaGraph always faster than direct launch" — **WRONG**.
V9 measured: 1-node graph = 2.05 µs ≈ direct 2.06 µs. Only batch ≥ 10
kernels per launch yields speedup. Original 10 catalog row "cudaGraphLaunch
1.20 µs (35 % cheaper)" applied only to the CPU-enqueue half of the call;
the full sync round-trip is identical.

**Footgun:** ⚠ "2.05 µs invariant launch latency as a HW property" —
RETRACTED as event-floor artifact. The 2.05 µs floor is `cudaEventRecord`
overhead, not kernel-launch latency. To measure kernel-launch latency,
use `cuStreamWriteValue32` to a device flag plus a busy-wait kernel, then
the round-trip is 0.45 µs + kernel poll cycles.

**Footgun:** ⚠ "WaitValue 3 µs faster than event sync" — true for the
host-call only, equivalent for the full pair. Always state which side
you measured.

**Footgun:** ⚠ "BlockingSync 5–7× slower than spinning sync" — was true
in older drivers; on B300 / CUDA 13.2 the gap is 25 % steady-state. The
old "5–7×" number is from CPU-pinned single-thread; with 4+ host threads
the BlockingSync wakeup latency dominates and the number flips.

**Footgun:** ⚠ "Cooperative launch overhead = +32 ns" — already retired
in original 10 catalog. Cooperative launch on B300 is essentially free
relative to a regular launch (the +32 ns is grid-sync setup, not
launch-side).

**Footgun:** ⚠ Persistent-kernel "38 ns/task" comes from V7 memory, not
re-verified in V8/V9 cycle. Treat as MED confidence until re-anchored
with current driver.

**Footgun:** ⚠ `cudaGraphLaunch` from device code (`DeviceLaunch` flag)
measured 13.7 µs; whether this stacks with cluster launch overhead
untested. If you use device-side graph launch in a clustered kernel,
you may pay 13.7 + cluster-setup µs per launch. Measure before
publishing.

**Footgun:** ⚠ Graph capture for cuBLAS: the catalog says "no speedup,
slightly hurts." CLAUDE.md `project_b300_pitfalls` says "cuBLAS needs
cudaGraph for sustained measurements." Both true: capture is for
*measurement isolation* (eliminates per-call host overhead from the
timed region), not for runtime perf gain.

**Footgun:** ⚠ `cuStreamWriteValue32` 0.45 µs is the HOST CALL ONLY.
The full producer-consumer round trip with `cuStreamWaitValue32` on a
second stream is 2.47 µs. State which you mean.

### §57.8 Sources

- `b300_clean/corrections/10_launch_overhead_CORRECTED.md` (canonical ladder)
- `b300_clean/V9_GRAPH_LAUNCH.md` (single-node bust + amortization curve)
- CLAUDE.md memory `project_b300_v7_complete` (38 ns persistent task,
  cuStreamWriteValue 0.45 µs)
- CLAUDE.md memory `project_b300_session2` (35× ExecUpdate, persistent
  4 µs)

---

## §58. Block scheduling / cluster topology

**Answer:** 148 SMs across 8 GPCs; the dominant model is **2 GPCs × 20 SMs +
6 GPCs × 18 SMs = 148 active**. Cluster max=8 portable / 16 non-portable /
32+ silently no-ops. Cluster=8 spans 4 GPCs deterministically (DSMEM_REFERENCE
SM set {0,1,16,17,32,33,48,49}). Stride-16 column = different GPC (consistent
with bus-width math but never directly verified by `gpc__cycles_active.per_pgpc_id`).
Cluster placement is deterministic when the GPU is otherwise idle, runtime-chosen
otherwise.  `[🟡 MED · src: 11_block_scheduling_CORRECTED.md§reconciled]`

### §58.1 Verified consensus

| Topic | Value | Confidence | Source agreement |
|---|---|---|---|
| Total SM count | 148 (IDs 0..147 dense) | HIGH | All sources |
| GPC count | 8 | HIGH | 11, M3, TRUE_REFERENCE, ncu `gpc__cycles_elapsed` |
| TPC = 2 SMs (consecutive IDs, stride +1) | YES | HIGH | I6, I8, M3, DSMEM all agree |
| GPC-row stride between TPCs in a cluster | +16 SMs | HIGH | I6, I8, M3, DSMEM all agree |
| Concurrent kernel dispatch slots | 128 | HIGH | 11, M3, TRUE_REFERENCE |
| Cluster placement spans multiple GPCs | YES | HIGH | DSMEM, 11 (vs prior "same-GPC" claim retracted) |
| Cluster-launch attribute overhead | ~0 vs regular launch | HIGH | 11, M3 |
| Cooperative-launch overhead | +32 ns | HIGH | 11 |

### §58.2 The "2×20 + 6×18" GPC model

The single canonical answer to "how are 148 SMs distributed across 8 GPCs":

- **2 GPCs have 20 SMs each.** These are the "long" GPCs.
- **6 GPCs have 18 SMs each.** These are the "standard" GPCs.
- Total: 2×20 + 6×18 = **148 active SMs.**

This is from `11_block_scheduling.md` line 16 (HIGH confidence). It is the
only model that arithmetically lands on 148 with the column-stride-16
layout AND matches the I8 cluster-8 cluster-15 wraparound (cluster 8 cy
in I8 shows +13 stride instead of +15 — consistent with 2 long GPCs of 20
SMs offsetting the modulo).

**Two competing models, both retracted:**

1. `B300_TRUE_REFERENCE.md` line 132 says "8 GPCs × ~18 SMs each (= 144
   active + 4 spare = 148 total)". The "4 spare" framing has no
   architectural basis — those SMs are active and used, just unevenly
   distributed. Retract.

2. `I8_CLUSTER_TOPOLOGY.md` and `M3_TOPOLOGY_CHEATSHEET.md` use the
   phrase "9.25 GPC-rows of 16 SMs". This conflates "GPC" (NVIDIA
   hardware unit, of which there are 8) with "stride-16 column window"
   (a scheduler addressing unit). Retract the "9.25 GPC-rows" phrasing.

### §58.3 Cluster placement

**Cluster-8 spans 4 GPCs.** DSMEM_REFERENCE measured the deterministic SM
placement set as `{0, 1, 16, 17, 32, 33, 48, 49}`. The pairs (0,1),
(16,17), (32,33), (48,49) each are a TPC. The stride between TPCs is 16 —
consistent with "stride-16 column = different GPC". So cluster=8 occupies
4 of the 8 GPC columns.

**Cluster placement is deterministic when the GPU is otherwise idle.**
DSMEM measured the set above repeatedly across launches; all stable. With
concurrent work the runtime selects free SMs — the relative *topology*
(which TPC pairs span which 16-SM columns) is preserved, but absolute SM
IDs may shift. I8 emphasizes the non-determinism of in-flight workloads,
DSMEM emphasizes the determinism on a quiet GPU. Both right in their
contexts.

### §58.4 Cluster size limits

| Size | Status | Note |
|---:|---|---|
| ≤ 8 | Portable, full HW support | Use this for portable code |
| 16 | Non-portable, launches succeed | B300-specific; some SMs participate twice in placement |
| 32+ | Silently no-op | Launch returns success, all blocks land in CTA 0's SM only |

The "32+ silently no-op" is the most surprising; it is not flagged as an
error by the runtime, just produces no benefit. If you depend on cluster
behavior, **check `cudaOccupancyMaxActiveClusters`** to confirm your
target is supported.

### §58.5 Open questions on topology

| # | Question | Why open |
|---|---|---|
| 1 | SM-id → GPC mapping not directly verified | No test reads `gpc__cycles_active.per_pgpc_id` per-CTA |
| 2 | Are 2 GPCs really 20 SMs each, or is it a different distribution? | 11.md asserts HIGH but cites no specific ncu metric |
| 3 | I8 cluster-8 cluster-15 anomaly: SMs (66, 67, 80, 81, 94, 95, 108, 109) show gap +13 instead of +15 | Attributed to "partial row" but not reconciled against 11.md "2 long GPCs" model |
| 4 | `cudaOccupancyMaxActiveClusters` for cluster_size = {4, 8, 16} | Never reported; 11.md asserts "142 SMs participate at cluster=8" without API confirmation |
| 5 | Why does block 0 launch on SM 142 (the *last* TPC pair) instead of SM 0? | Hypothesized as "partial GPC gets priority" in I6, not verified |
| 6 | Cluster size ≥ 16 placement topology | If cluster_size=16 spans 8 GPCs (= all GPCs), DSMEM cost may differ from cluster=8; not measured |

### §58.6 Footguns

**Footgun:** ⚠ TRUE_REFERENCE's "144 active + 4 spare SMs" framing has no
architectural basis. The 8 GPC × (2×20 + 6×18) = 148 active is the model.
The "spare" SMs phrasing was an early hypothesis, retracted in 11.md.

**Footgun:** ⚠ "GPC-row" in I8/M3 ≠ "GPC". A "GPC-row" in those docs
means a 16-SM stride window. There are 8 GPCs, not 9.25.

**Footgun:** ⚠ Cluster placement SM-id set {0, 1, 16, 17, …} is
deterministic only on an otherwise-idle GPU. Production workloads will
see runtime-chosen SM IDs; the *topology* (TPC pairs and stride-16
columns) is preserved but the absolute SM IDs are not.

**Footgun:** ⚠ Cluster size 32+ launches return success but produce no
multi-CTA placement (silently no-op). Always check
`cudaOccupancyMaxActiveClusters` if you depend on cluster behavior.

**Footgun:** ⚠ "Cluster blocks placed within same GPC" — old catalog
claim, RETRACTED in 11.md commit 79372e6. Cluster of 8 spans **4 GPCs**.

**Footgun:** ⚠ "10 GPCs (9×16 + 1×4)" — old catalog claim, RETRACTED.
B300 has 8 GPCs.

**Footgun:** ⚠ Cluster placement claims of "stride-16 = different GPC"
are consistent with bus-width math but never directly verified by
`gpc__cycles_active.per_pgpc_id`. If you build new analysis on this,
collect the per-GPC ncu metric to anchor.

### §58.7 Sources

- `b300_clean/corrections/11_block_scheduling_CORRECTED.md` (full
  reconciliation with 6 contradictions)
- `b300_clean/I6_BLOCK_SCHEDULE_TOPOLOGY.md` (TPC stride)
- `b300_clean/I8_CLUSTER_TOPOLOGY.md` (cluster-8 anomaly)
- `b300_clean/M3_TOPOLOGY_CHEATSHEET.md` (16-SM window framing)
- `b300_clean/DSMEM_REFERENCE.md` (deterministic placement set)

---

## §59. NVRTC + module APIs

**Answer:** NVRTC compile cost ladder 5.4 / 5.8 / 23 ms (tiny / medium /
5000-FMA). Module load `cuModuleLoadData(cubin)` ≈ 10 µs flat 5–80 KB.
Cold-process +11 ms framework init; +240 ms `cuCtxCreate`. PTX-JIT load
scales with PTX size up to 155× cubin-load at 5000 FMA. `cuModuleGetFunction`
≈ 39 ns; `cuLibraryGetKernel` ≈ 13 ns. **CUDA 13.2 quirks:** NVRTC accepts
`tcgen05.*` PTX that static ptxas rejects; both NVRTC and ptxas reject
`cvt.rn.satfinite.e2m1x4.f32` on sm_103a (PTX 8.7 migration needed).
QuickRunCUDA injects `--use_fast_math` → all FFMA emit as `FFMA.FTZ`.
`[🟡 MED · src: 17_nvrtc_module_CORRECTED.md]`

### §59.1 Compile / load cost ladder

| Operation | Cost | Source |
|---|---:|---|
| NVRTC compile (tiny kernel ≤ 50 SASS) | 5.4 ms | 17 catalog §1 |
| NVRTC compile (medium ≈ 500 SASS) | 5.8 ms | 17 catalog §1 |
| NVRTC compile (5000-FMA kernel) | 23 ms | 17 catalog §1 |
| Cold-process framework init (one-shot) | +11 ms | 17 catalog §1 |
| `cuCtxCreate` (one-shot per process) | +240 ms | TRUE_REFERENCE §4 |
| `cuModuleLoadData(cubin)` 5–80 KB | ~10 µs flat | 17 §3 |
| `cuModuleLoadData` from PTX (5000 FMA) | 1550 µs (155× cubin path) | 17 §3 |
| `cuModuleGetFunction` | ~39 ns | 17 §6 |
| `cuLibraryGetKernel` (CUDA 12+ path) | ~13 ns | 17 §6 |
| `cuMemCreate` (VMM) | ~0.5 µs/MB beyond 2 MB floor | 17 §4 |
| NVTX (no profiler attached) | ~19 ns | 17 §5 |
| `cudaGetLastError` | ~20 ns | 17 §5 |

### §59.2 The `--use_fast_math` quirk in QuickRunCUDA

`QuickRunCUDA.cpp` calls into `utils/cuda_helper.h:227` which sets
`--use_fast_math` unconditionally on every NVRTC compile. This has
*concrete numerical consequences* that affect every benchmark in this
catalog:

1. **All FFMA emit as `FFMA.FTZ`.** Subnormals are flushed to zero on the
   FFMA pipe. You **cannot measure non-FTZ subnormal handling via
   QuickRunCUDA without first patching out `--use_fast_math`** in
   `cuda_helper.h:227`. Standalone `nvcc` builds in
   `b300_clean/M7_V5_SYNTHESIS.md` D1 confirmed B300 supports full-speed
   subnormal FFMA at 4.11 cy when `-ftz=false` is used.

2. **All `__fdividef`, `1.0f/x`, `sqrtf` get the approximate path.** This
   bypasses the ~243 cy `div.rn.f32` of standalone `nvcc`. Reduces
   per-op latency by 2–60×.

3. **All MUFU instructions are routed via the approximate XU path.** This
   is why `tests/` kernels measure rsqrt at MUFU rates.

If your test depends on IEEE-correct rounding, denormal handling, or
full-precision div, **either patch `cuda_helper.h:227` or use standalone
`nvcc` builds.** This is documented in `feedback_nvrtc_fast_math_ftz`
memory but easy to miss in catalog reading.

### §59.3 NVRTC vs ptxas acceptance — narrow PTX forms

CUDA 13.2 has a known bug class around narrow numerical formats. NVRTC
and static ptxas do NOT have identical acceptance rules:

| PTX form | NVRTC sm_103a | Static ptxas (CUDA 13.2) | Source |
|---|:---:|:---:|---|
| `tcgen05.mma` | accepts | rejects | 06_tensor_cores §6 line 93 |
| `tcgen05.alloc` | accepts | rejects | 06_tensor_cores §6 line 93 |
| `cvt.rn.satfinite.e2m1x4.f32` | **REJECTS** | rejects | V41_V48 lines 61-62 |
| `cvt.scalefactor` variants | not yet tested | not yet tested | V8/V9 follow-up |
| `cvt.rz/.rm/.rp.e4m3x2.f32` | rejects | rejects | CURIOSITY V7 H1 |

Pattern: **NVRTC > ptxas only for `tcgen05.*` PTX.** For narrow `cvt`
forms BOTH compilers reject. The earlier "NVRTC accepts more than ptxas"
generalization is wrong. If you have a `cvt` PTX bug, NVRTC will not
save you.

### §59.4 Init/main kernel arg conflict — QuickRunCUDA harness quirk

QuickRunCUDA passes the same `-0/-1/-2` ints to BOTH the optional `init`
kernel AND the timed `kernel`. If you reuse arg slot 0 as `iters` for the
main kernel, the init kernel's "use" of `iters` may be invalid or
destructive.

**Workaround**: pack init parameters into a single int and bit-shift
extract inside the init kernel; reserve `-0` (typically `iters`) for the
main kernel.

```cuda
// init kernel: extract from arg0 packed as
// [u8 init_param0][u8 init_param1][u16 init_param2]
__global__ void init(float* A, float* B, float* C,
                     int packed, int unused1, int unused2) {
    int p0 = packed & 0xff;
    int p1 = (packed >> 8) & 0xff;
    int p2 = (packed >> 16) & 0xffff;
    // ...
}

// main kernel: arg0 is iters
__global__ void kernel(float* A, float* B, float* C,
                       int iters, int arg1, int arg2) {
    for (int i = 0; i < iters; ++i) { /* ... */ }
}
```

This is documented in CLAUDE memory `feedback_compute_pipe_methodology`
but easy to miss.

### §59.5 Module / library API — one-time vs per-call costs

`cuModuleLoadData` cost dominates startup; once loaded, getting a kernel
handle is essentially free:

- `cuModuleGetFunction`: ~39 ns
- `cuLibraryGetKernel` (CUDA 12+ Library API): **~13 ns** (3× faster)

If you launch the same kernel 1000+ times, **always cache the function
handle**. The 39 ns lookup quickly dominates a 1.85 µs launch cost when
done per-launch.

The `cuLibrary*` 6.5× speedup claim from older catalog is based on a
single line; not re-measured with current driver. Use as MED-confidence.

### §59.6 PTX-JIT vs cubin loading

If you ship PTX (forward-portable but slow to load) instead of cubin
(SM-specific, fast to load), you pay:

- 1× cost: PTX-JIT compile happens once per `cuModuleLoad`.
- For a 5000-FMA kernel, PTX-JIT load is **~1550 µs vs ~10 µs for
  cubin** (155× slower).
- The compiled cubin is cached in `/var/tmp/.nv/ComputeCache` (or
  `$CUDA_CACHE_PATH`); subsequent runs hit the cache in ~10–20 µs.

For dev workflows shipping cubin via NVRTC + `cuModuleLoadData` is the
fastest iteration loop (5.4 ms compile + 10 µs load). For production,
ship cubin and avoid the PTX-JIT path.

### §59.7 Footguns

**Footgun:** ⚠ `--use_fast_math` is set unconditionally in QuickRunCUDA
(`utils/cuda_helper.h:227`). Every FFMA is `.FTZ`; every reciprocal is
the approx path. Patch out before running subnormal-handling or
IEEE-correctness tests.

**Footgun:** ⚠ NVRTC rejects bare `-O0..-O3`. Use
`--ptxas-options="-O3"` for ptxas opts.

**Footgun:** ⚠ NVRTC > ptxas for `tcgen05.*` PTX, but BOTH reject
`cvt.rn.satfinite.e2m1x4.f32` (PTX 8.7 needed). Do not assume NVRTC
solves narrow-cvt bugs.

**Footgun:** ⚠ QuickRunCUDA passes same `-0/-1/-2` to init and main
kernel. Pack init params via bit-shift; reserve `-0` for main kernel.

**Footgun:** ⚠ `-G` (debug) compile makes cubin much larger but
runtime impact NOT quantified in catalog. Treat as suspect for any
"% of peak" measurement in `-G` mode.

**Footgun:** ⚠ The `/var/tmp/.nv/ComputeCache` PTX-JIT cache can hide
your actual NVRTC compile time. Set `CUDA_CACHE_DISABLE=1` for true
cold compile measurements.

**Footgun:** ⚠ `cuLibrary*` 6.5× speedup over `cuModule*` is from a
single older catalog line. Re-measure with current driver before
publishing as a cold-start optimization.

### §59.8 Sources

- `b300_clean/corrections/17_nvrtc_module_CORRECTED.md`
- `b300_clean/14_math_intrinsics.md` line 87 (FTZ confirmation)
- `b300_clean/06_tensor_cores.md` §6 line 93 (NVRTC tcgen05 acceptance)
- `b300_clean/V41_V48_FINDINGS.md` lines 61-62 (cvt e2m1x4 reject)
- CLAUDE memory `feedback_nvrtc_fast_math_ftz`
- CLAUDE memory `feedback_compute_pipe_methodology` (init/main arg)

---

## §60. Device props / nvml — what to query and how

**Answer:** `cudaGetDeviceProperties` exposes the dispositive hardware
identifiers. Critical fields on this device: `name = "NVIDIA B300 SXM6 AC"`,
`totalGlobalMem = 275040 MiB` (≈ 287 GB), `memoryBusWidth = 7680` (NOT
8192 — this is a yield-fused SKU), `memoryClockRate = 1998000` kHz (= 1998
MHz pre-DDR doubling = 3996 MT/s effective), `l2CacheSize = 132 MiB` (~126
MB practical), `multiProcessorCount = 148`. NVML clock query during run
samples the actual clock to detect stuck-at-1005-MHz states.
`[🟢 HIGH · src: cudaGetDeviceProperties + 16_power_clock + HBM_STACKS_INDEPENDENT_VERIFY]`

### §60.1 The query and its outputs (this device)

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
// prop.name              = "NVIDIA B300 SXM6 AC"
// prop.major             = 10
// prop.minor             = 3              → CC = 10.3, sm_103a
// prop.totalGlobalMem    = 275040 MiB ≈ 288.4 GB
// prop.memoryBusWidth    = 7680            → KEY: not 8192 (15/16 of spec)
// prop.memoryClockRate   = 1998000 kHz     = 1998 MHz pre-DDR
//                                          = 3996 MT/s after DDR
// prop.l2CacheSize       = 138_412_032 B  ≈ 132 MiB ≈ 126 MB practical
// prop.multiProcessorCount = 148
// prop.warpSize          = 32
// prop.maxThreadsPerBlock = 1024
// prop.maxThreadsPerMultiProcessor = 2048   (= 8 CTA × 256 thr or 4×512 etc.)
// prop.regsPerBlock      = 65536           (256 KiB / SM partition)
// prop.regsPerMultiprocessor = 65536       (per SMSP)
// prop.sharedMemPerBlock = 49152           (default 48 KiB)
// prop.sharedMemPerBlockOptin = 233472     (228 KiB opt-in via cudaFuncSetAttribute)
// prop.sharedMemPerMultiprocessor = 233472 (228 KiB total)
// prop.clockRate         = 2032000 kHz     (boost; deprecated in CUDA 13)
// prop.singleToDoublePrecisionPerfRatio = 64
// prop.maxBlocksPerMultiProcessor = 32
// prop.totalConstMem     = 65536           (64 KiB cmem)
// prop.maxGridSize[0]    = 2147483647      (2^31 - 1)
// prop.computeMode       = 0               (Default)
```

The CUDA-13-deprecated `prop.clockRate` field still works but the
recommended replacement is:

```cpp
int khz;
cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0);
// khz = 2032000  (boost)
```

### §60.2 The bus-width revelation: 7680, not 8192

The single most important field for HBM math is `memoryBusWidth`. On this
device it returns **7680**, not the 8192 you'd expect from "8 stacks ×
1024 bits". The interpretation:

- B300 architecturally has 8 × HBM3E 12-Hi stacks, 16 × 512-bit
  controllers, **8192-bit total bus**.
- `7680 = 8192 × 15/16` — **exactly one /16 controller fused off** (yield
  bin). The "AC" suffix in the part name is consistent with this
  capacity/channel-restricted SKU.
- All 8 stacks are physically present; one channel pair is disabled.

This fact controls every "% of HBM peak" denominator in the catalog. See
§6 (HBM, sibling agent) and §62 rule 12 below for the proper "spec / actual /
this-device" denominator framework.

### §60.3 Memory clock interpretation

`prop.memoryClockRate = 1998000` kHz = 1998 MHz. This is the I/O clock,
**before** DDR doubling. The effective transfer rate is 1998 × 2 = **3996
MT/s = 3.996 Gbps/pin** (per pin per direction).

The HBM3E datasheet quotes 8.000 Gbps/pin spec. The 0.10 % gap (3.996
vs 4.000 GT/s pre-DDR; equivalently 7.992 vs 8.000 Gbps post-DDR) is
**real silicon under-spec**, not arithmetic noise. Use:

- 7.68 TB/s = spec post-ECC (8.000 Gbps × 8192 bits ÷ 1.0625 ECC ÷ 8)
- 7.67 TB/s = this-device post-ECC (7.992 Gbps × 7680 bits ÷ 1.0625)
- 7.31 TB/s = empirical pure-direction peak (V32, never as "spec")

These three numbers are within 5 % of each other; choose the framing
deliberately.

### §60.4 L2 capacity (132 MiB nominal; 126 MB practical)

`prop.l2CacheSize = 138_412_032 bytes = 132 MiB`. The **practical**
working-set capacity is closer to **126 MB** (= 132,120,576 bytes) once
you subtract the persisting carveout overhead and inclusive-victim
metadata. Both numbers are reported in different catalog files:

- "L2 = 132 MiB" — `cudaGetDeviceProperties` raw
- "L2 = 126 MB" — practical working-set ceiling per `D3_L2_SECTOR_RIGOR.md`
- "L2 = 96 MB" — **WRONG**, cosmetic error in 4 catalog files (RETRACTED)

The "96 MB" was a transcription error from H100 specs; do NOT use.

L2 max persisting cache:

```cpp
int max_persist;
cudaDeviceGetAttribute(&max_persist,
                      cudaDevAttrMaxPersistingL2CacheSize, 0);
// max_persist = 79.1 MB
```

### §60.5 nvml — clock locking and live sampling

`utils/nvmlClass.h` wraps NVML for clock-lock and live sampling. Critical
operations:

```cpp
// Lock clock (CAUTION: paradoxically pins to 1920 MHz for argument 2032)
nvmlDeviceSetGpuLockedClocks(dev, 1920, 1920);

// Live sample of current SM clock (works during kernel execution)
unsigned int sm_clock;
nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &sm_clock);
// e.g. 2032 MHz under sustained FFMA, default boost
// e.g. 1005 MHz under thermal throttle (see footgun)

// Power draw at this instant
unsigned int power_mW;
nvmlDeviceGetPowerUsage(dev, &power_mW);
// e.g. 552_000 (= 552 W) under sustained FFMA at 2032 MHz

// Reset clock (CAUTION: -lgc / NVML lock does NOT reset on process exit)
nvmlDeviceResetGpuLockedClocks(dev);
```

### §60.6 The "stuck at 1005 MHz" detection pattern

CLAUDE memory `feedback_clock_stuck_no_lock` documents a real pitfall:
B300 can be stuck at 1005 MHz under load with NO explicit clock lock.
`nvidia-smi -q` won't show it as "locked"; you must sample during the
run.

Detection:

```bash
# during benchmark run (in another terminal)
while true; do
    nvidia-smi --query-gpu=clocks.sm,power.draw \
               --format=csv,noheader,nounits
    sleep 0.1
done
```

If `clocks.sm` shows 1005 (or any non-2032 value) when you expected
boost, the throttle is real. Recovery:

```bash
nvidia-smi -rgc          # reset graphics clock
sleep 2
# re-run benchmark
```

### §60.7 Hardware-spec constants (use cudaDeviceProp, NOT hardcoded)

`utils/cuda_helper.h` defines `GPU_SM_COUNT=132` etc. These are H100/H200
defaults and are **not auto-detected**. For B300, query via API:

```cpp
int sm_count, l2_bytes;
cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0);
// sm_count = 148 on B300
// l2_bytes = 138_412_032 (132 MiB)
```

If you hardcode `132` in benchmark math, you'll under-count B300's
performance by 12 %. Always query.

### §60.8 Footguns

**Footgun:** ⚠ `memoryBusWidth = 7680` NOT 8192 means this AC SKU has
1/16 controllers fused off; affects ALL "% of HBM peak" math. Use 7.67
TB/s as this-device denominator; use 7.68 TB/s for spec-comparable
cross-vendor numbers. See §6 (sibling).

**Footgun:** ⚠ `prop.clockRate` deprecated in CUDA 13. Use
`cudaDeviceGetAttribute(cudaDevAttrClockRate)` instead.

**Footgun:** ⚠ "L2 = 96 MB" is WRONG (cosmetic transcription from H100
specs in 4 catalog files). Real L2 = 132 MiB nominal / 126 MB practical.

**Footgun:** ⚠ Hardcoded `GPU_SM_COUNT=132` in `cuda_helper.h` is the
H100 default. Always query `cudaDevAttrMultiProcessorCount`. B300 = 148.

**Footgun:** ⚠ `nvidia-smi -lgc 2032` paradoxically pins to 1920 MHz
(base clock), NOT 2032. This is documented in CLAUDE.md §2 but
surprises every new user.

**Footgun:** ⚠ Clock lock from NVML (`nvmlDeviceSetGpuLockedClocks`)
does NOT reset on process exit. You can leave the GPU locked across
sessions. Always pair `Set` with a deferred `Reset` or call
`nvidia-smi -rgc` between runs.

**Footgun:** ⚠ B300 can be stuck at 1005 MHz with NO explicit lock.
Sample `nvmlDeviceGetClockInfo(NVML_CLOCK_SM)` during the run; if
non-2032 when you expected boost, run `nvidia-smi -rgc`.

**Footgun:** ⚠ `prop.clockRate = 2032 MHz` is the boost ceiling, not
the actual sustained clock. Default behavior under load is to boost to
2032; locked behavior at `-lgc 2032` is 1920. State which you measured.

### §60.9 Sources

- `b300_clean/corrections/HBM_STACKS_INDEPENDENT_VERIFY.md` (bus-width
  derivation)
- `b300_clean/16_power_clock_CORRECTED.md` (clock state ladder)
- `b300_clean/D3_L2_SECTOR_RIGOR.md` (126 MB practical L2)
- `b300_clean/corrections/STRAYS_CORRECTED.md` §7 (96 MB error catalog)
- CLAUDE memory `feedback_clock_lock_works`,
  `feedback_clock_stuck_no_lock`

---

## §61. Rigor protocol — minimum viable measurement

**Answer:** 3-method verification (wall-clock + ncu + SASS) is the gold
standard. For dual-issue: ≥64 ops/type body + matched solo/dual methodology +
simultaneous `pipe_fma + pipe_alu` ncu reads. For BW: anti-DCE final-write
commit + non-LICM-able pattern + stride-per-iter for HBM (defeat L2 cache
hits). Always pkill leftover processes + sleep 5–8 s. Always sample clock
during run. Always pair Gops/s with bytes/s.
`[🟢 HIGH · src: CLAUDE.md §3-4 + META_LESSONS.md + V52_RUN_RESULTS.md]`

### §61.1 The 3-method principle

A measurement is HIGH-confidence only if **three orthogonal evidence
sources agree**:

1. **Wall-clock with cudaEvent**: time the kernel from outside, anti-DCE
   defeated.
2. **ncu pipe metrics**: `smsp__pipe_fma_cycles_active.pct_of_peak_sustained_active`
   etc. Confirm the pipe you think you're measuring is the one ncu sees
   active.
3. **SASS inspection**: `cuobjdump --dump-sass <binary>` or look at
   `sass/<basename>_<hash>.sass`. Count actual emitted instructions in
   the hot loop. Source-level `#pragma unroll N` does NOT guarantee
   SASS-level unroll.

If any two disagree, you have a methodology bug. The V52 episode (see
Appendix A) is the canonical case study: V49 had wall-clock evidence
that looked solid (55 % dual-issue, reproducible), but had no ncu and
no SASS audit. When V52 added both, the wall-clock interpretation
flipped from "55 % dispatch cap" to "147 % free overlap of two pipes".

### §61.2 Anti-DCE checklist

Every benchmark MUST defeat dead code elimination. Compiler optimization
is aggressive on B300 nvcc 13.x. The mandatory defenses:

1. **Unconditional STG of the final accumulator.** Not just "if (tid ==
   0) STG …" — that gets eliminated when the compiler proves the
   condition is impossible-but-reachable. Use:

   ```cuda
   if (acc != 0xdeadbeef) C[blockIdx.x * blockDim.x + threadIdx.x] = acc;
   ```

   The compiler cannot prove `acc != 0xdeadbeef` is false at compile
   time without solving the loop semantically.

2. **Make loop values depend on runtime inputs.** If `arg0 = 100`
   (compile-time constant from `-0 100`), the compiler can unroll and
   precompute. Pass `arg0` through the kernel signature so the
   value is opaque.

3. **Make the loop trip count depend on runtime input.** `for (int i =
   0; i < ITERS; ++i)` where `ITERS` is a template parameter compiles
   to a fixed-trip loop the compiler can fully unroll. Use `for (int i
   = 0; i < arg0; ++i)` with a runtime arg.

4. **Make addresses depend on `threadIdx.x`** so the compiler cannot
   factor the load out of the loop (LICM defense).

5. **Make values depend on `threadIdx.x`** so the compiler cannot CSE
   loads across threads.

6. **Check kernel runtime ≥ 1 ms.** A kernel that runs in 0.001 ms is
   either fully eliminated or measuring launch overhead.

### §61.3 Methodology gates for dual-issue / pipe-overlap claims

For any "pipe X reaches Y % of theoretical Z" or "co-issue of pipes X+Y
yields Q % of summed peak":

1. **Inner body must amortize loop overhead — minimum 64 ops/type per
   iter.** V49's 8 FFMA + 8 LOP3 inner body let branch + loop-counter
   (UIADD3 + UISETP) consume ~10–15 % of ALU dispatch slots. V8's
   128-deep inner amortizes branch overhead by ~16×. Below 64 ops/type,
   a "dual-issue measurement" measures branch contamination as much as
   it measures dual-issue.

2. **Solo and dual baselines must use IDENTICAL methodology.** Same
   unroll depth, same `__launch_bounds__`, same warps/SMSP, same
   anti-DCE strategy, same registers-distinct-or-not. V49's solo FFMA
   ran at 67 % but V8's solo FFMA ran at 97.6 % at the same occupancy
   because the **inner body shapes** differed.

3. **Always SASS-verify inner body composition before reporting %.** Run
   `cuobjdump -sass` and count the actual emitted instructions in the
   hot loop. If the inner body contains UIADD3 / UISETP / BRA / LDC /
   IMAD.MOV.U32 not part of the pipe being measured, those count
   against the dispatch budget.

4. **Always cite ncu `pipe_fma + pipe_alu` (and `pipe_lsu` where
   applicable) simultaneously for dual-issue claims.** A single pipe
   metric cannot prove dual-issue. The diagnostic signature of true
   dual-issue is `pipe_fma.pct + pipe_alu.pct > 100 %`. The diagnostic
   signature of serial issue (no dual-issue) is `pipe_fma.pct +
   pipe_alu.pct ≈ 100 %`. V49/V50 collected NEITHER metric.

5. **Cross-check against published literature.** Hopper (sm_90) has the
   same SMSP dispatch architecture as B300 (sm_103); H100 measurements
   show summed `pipe_fma + pipe_alu > 100 %` in well-formed dual-issue
   tests. A B300 result that says "dispatch is 4-wide per SM regardless
   of pipe" should be cross-checked against H100 baseline.

### §61.4 Methodology gates for bandwidth claims

For any "memory peak Y TB/s" or "% of HBM peak Z %":

1. **Anti-DCE: write a runtime-dependent value to global** so the load
   chain cannot be eliminated.

2. **Defeat L2: working set ≥ 4 GB** with stride-per-iter so each load
   misses L2. L2 is 126 MB practical; anything smaller will be absorbed
   by L2 and you'll measure 24 TB/s kernel-effective L2, not 7 TB/s
   HBM.

3. **Stride per iter, not within iter.** A within-iter stride looks
   like a cache-friendly stream; a per-iter stride defeats the
   prefetch/eviction predictor.

4. **State the denominator precisely.** Cite both 7.68 TB/s (spec) and
   7.67 TB/s (this-device, 7680-bit) when comparing. Never write "% of
   8 TB/s" — that's marketing rounding, not a real spec number.

5. **Cross-check ncu `dram__bytes_read.sum.per_second`.** This is the
   authoritative read rate; matches the wall-clock-derived BW within
   2 % when methodology is clean.

6. **Distinguish chain-bound vs ILP-fed.** A dependent-chain test
   measures latency × N, not throughput. A non-chained 8-ILP test
   measures the actual throughput.

7. **Distinguish issue-rate vs completion-rate** for writes. A write
   benchmark with no fence between stores and the timer end is
   measuring issue rate, not delivery. Add `fence.sc.cluster` or
   equivalent before stopping the clock.

### §61.5 Methodology gates for atomic / contention claims

For any "atomic Y Tops/s" or "contention Z % penalty":

1. **State chain depth.** A single-pointer atomic chained across
   threads has totally different throughput than a per-thread atomic
   with no contention.

2. **Pair Gops/s with bytes/s.** Cache-line combining can inflate Gops
   8× without proportional BW. Got "28× ratio" wrong by mixing
   combined+uncombined atomics (CLAUDE memory `feedback_units_sanity`).

3. **State stride.** Stride-4 (1 atomic per 32-bit word) and stride-32
   (1 atomic per cache line) measure different things. The L2 atomic
   units pack atomics within a cache line, inflating apparent throughput
   8×.

4. **State unroll factor.** Catalog has rows at UNROLL=1, 16, 32 with
   3-way spread (449 / 504 / 1005 Gops/s on stride-4 L2 atomic).

5. **Check ncu `lts__t_sectors_op_atom.sum`** for atomic count
   verification.

### §61.6 Process hygiene

Always do these between benchmark runs:

```bash
pkill -9 QuickRunCUDA   # or your test binary name
sleep 6                  # let GPU contexts clean up
nvidia-smi -rgc          # release any clock locks
sleep 2
nvidia-smi --query-gpu=power.draw,clocks.sm \
           --format=csv,noheader  # confirm idle state
```

The pitfall: leftover processes silently inflate cy/MMA up to **8.5×**
(see CLAUDE memory `feedback_b300_pitfalls`). The 8.5× was measured: 5
zombie QuickRunCUDA processes from a previous session were sharing SM
0–4, the test launched on SM 5+ and saw 8.5× cycles per MMA without
any visible error.

### §61.7 Clock state discipline

Every TFLOPS / W / latency number in the catalog must state which clock
state:

| Clock state | What it means | When you get it |
|---|---|---|
| **Default boost** | 2032 MHz | No `nvidia-smi -lgc`; sustained load lets boost engage |
| **Locked 2032** | 1920 MHz (paradox!) | `nvidia-smi -lgc 2032` actually pins to 1920 |
| **Locked 1920** | 1920 MHz | `nvidia-smi -lgc 1920` |
| **Locked 1500** | 1500 MHz | Used in DRAM data-dep stress tests |
| **Locked 1005** | 1005 MHz | Used in NVFP4 power isolation |
| **Stuck 1005** | 1005 MHz under load with NO lock | Anomalous throttle; nvidia-smi -rgc to fix |

The 6 % gap between locked-2032 (= 1920) and default-boost (= 2032)
contaminates ANY catalog cross-section that mixes them. The default
catalog convention is **default boost (2032 MHz)** unless explicitly
stated otherwise.

### §61.8 Reproducibility checklist

Before publishing ANY HIGH-confidence number:

- [ ] Run 3× back-to-back, agreement within 1 %.
- [ ] `pkill -9` between runs.
- [ ] Sample clock during run; confirm expected state.
- [ ] Sample power during run (NVML); confirm not throttled.
- [ ] SASS-verify inner body composition.
- [ ] ncu pipe metrics confirm the pipe you think is active.
- [ ] Anti-DCE defenses present and SASS-confirmed.
- [ ] Working set defeats expected cache (L1 < L2 < HBM regime
      crossover).
- [ ] Denominator stated explicitly (7.68 spec / 7.67 actual / 7.31
      empirical for HBM; 76.97 TFLOPS / 72.65 TFLOPS for FFMA at
      2032 / 1920).

### §61.9 Sources

- `CLAUDE.md` §3-4 (rigor protocol)
- `b300_clean/corrections/META_LESSONS.md` (5 mandatory rules from
  zigzag)
- `b300_clean/corrections/V52_RUN_RESULTS.md` (the empirical anchor that
  validated the rules)
- CLAUDE memory `feedback_b300_pitfalls`, `feedback_units_sanity`,
  `feedback_microbench_rigor`

---

## §62. The 13-rule rigor protocol

**Answer:** 13 numbered rules (10 from CLAUDE.md + 3 wave-derived) that
together constitute the minimum viable rigor for B300 microbenchmarks.
`[🟢 HIGH · src: CLAUDE.md §3 + META_LESSONS.md + HEADLINE_v5.md]`

This is the explicit numbered list, expanded with worked examples for each
rule. Worked examples come from real catalog incidents (not contrived
toys). Each rule is followed by an example of catching it in the wild.

### Rule 1 — State theoretical maximum first

Before claiming peak throughput, ALWAYS compute the theoretical first.
If measured > theoretical, the test is broken.

**B300 theoretical peaks (at 2032 MHz boost):**
- **FP32 FFMA: 76.96 TFLOPS** = 148 SMs × 128 FP32 cores/SM × 2 op/FMA
  × 2.032 GHz
- **FP64 DFMA: 1.20 TFLOPS** (ratio 1:64 per
  `singleToDoublePrecisionPerfRatio`)
- **FP16/BF16 mma.sync m16n8k16: ~540-580 TFLOPS** (legacy tensor path)
- **BF16 tensor via tcgen05.mma: ~1980 TFLOPS** (Blackwell path)
- **FP8 tensor via cuBLAS (tcgen05): ~4500 TFLOPS** (verified 91 % MFU)
- **HBM3E: ~7.68 TB/s spec / ~7.67 TB/s this-device-SKU**
- **L2 BW: 23.85 TB/s kernel-effective / 13.30 TB/s wire**
- **Shared memory: 38.49 TB/s theoretical**

**Worked example (caught in wave-1):** A V8 row claimed "DSMEM 37 TB/s
= 97 % of 38.5 peak". Theoretical SHMEM-equivalent = 38.5 TB/s × 4
(cluster-of-4 amplification) = 154 TB/s upper bound; real is ~40 GB/s
per cluster (chain-bound, 1000× off). Without stating theoretical first,
the 37 TB/s looked plausible. With theoretical-first, the 37 / 38.5
ratio is suspect because DSMEM is a cluster-shared bus, not a
per-cluster amplifier.

### Rule 2 — State measured number with denominator

Always: "Measured Z TFLOPS = Z/X of theoretical, where X = …"

**Worked example (V46):** "98.5 % NEW HBM read SoL" used denominator
7.31 TB/s (V32 empirical). Re-anchored against 7.68 spec post-ECC =
93.8 %; against 7.67 this-device = 93.9 %. The "98.5 %" framing was
the artifact, not the 7.20 TB/s measurement. Always cite both number
AND denominator.

### Rule 3 — If measured > theoretical: STOP

Test is broken. Look for DCE, formula bugs, clock mismatch.

**Worked example (V8 DSMEM 37 TB/s):** SHMEM theoretical = 38.5 TB/s.
DSMEM (distributed SHMEM, cluster-shared) cannot exceed SHMEM peak ×
something interpretable. The 97 % ratio was suspicious because DSMEM
adds latency on top of SHMEM. SASS investigation revealed: V8's
compile-time invariant offsets were LICM'd / CSE'd; ncu wavefront count
was 7200 vs expected 5.9 B. Real aggregate was 40 GB/s/cluster — off
by ~1000×.

### Rule 4 — If measured > 1.5× theoretical: almost certainly DCE

The DCE failure mode is so common that any measurement substantially
above theoretical should be assumed eliminated until proven otherwise.

**Worked example:** A bench_fma test at 200 TFLOPS on a 76 TFLOPS HW
peak. The compiler had unrolled the loop, observed the result wasn't
written, and eliminated the entire body. Wall-clock measured launch
overhead × repetitions, divided by zero work, gave a meaningless
"throughput" number.

### Rule 5 — If measured < 0.5× theoretical: under-saturated

Methodology issue: ILP too low, occupancy too low, dependent chain in
hot loop, register port pressure.

**Worked example (V49 solo FFMA):** Measured 67 % of peak on 8-ILP × 2
warps/SMSP. V8 hits 97.6 % at the same occupancy. The gap was
methodology (smaller inner body amortized branch overhead worse),
not architectural.

### Rule 6 — If measured in [0.5×, 1.0×]: plausible but verify SASS

This is the regime where most real benchmarks live. SASS-verify the
inner body composition. ncu cross-check the pipe.

**Worked example (V52):** Solo FFMA at 84-87 % of peak, with V8 hitting
97.7 %. SASS showed V52's inner body had 1.2 % loop overhead vs V8's
amortization of 128-deep unroll. The gap was loop-tail / launch
overhead at small N_OUTER (1k-4k iterations). Above 1M iterations, V52
would also reach 97 %+.

### Rule 7 — SASS-verify with `nvcc -keep`

```bash
nvcc -arch=sm_103a -O3 -std=c++17 -keep your_test.cu -o your_test
ls *.sass *.ptx *.cubin
cuobjdump --dump-sass your_test > your_test.sass
# look for the inner loop:
grep -A 50 "your_kernel" your_test.sass
```

Count emitted FFMA, LOP3, IMAD, BRA, UIADD3, UISETP. If you wrote
"#pragma unroll 8" but see only 4 FFMA in the SASS, the compiler
re-rolled or your unroll didn't take.

**Worked example:** V49 showed `8 FFMA + 8 LOP3 + UIADD3 + UISETP +
BRA` in the inner body — the BRA / UIADD3 / UISETP consumed ALU pipe
slots, contaminating the FFMA+LOP3 dual-issue measurement. V52 with
128-deep unroll showed `128 FFMA + 136 LOP3 + 1 UIADD3 + 1 UISETP +
1 BRA` — the loop overhead is 1.2 % of body, properly amortized.

### Rule 8 — Cross-check ncu

For FFMA: `smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active`
For ALU (LOP3, IADD3): `smsp__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active`
For LSU: `smsp__pipe_lsu_cycles_active.avg.pct_of_peak_sustained_active`
For tensor (mma.sync, NOT tcgen05): `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`
For HBM: `dram__bytes_read.sum.per_second`,
        `dram__bytes_write.sum.per_second`
For L2: `lts__t_bytes.sum.per_second` (wire), or
        `lts__t_sector_hit_rate.pct`
For atomics: `lts__t_sectors_op_atom.sum`,
             `l1tex__data_pipe_lsu_wavefronts_mem_lg_op_atom.sum`
Total instructions: `smsp__inst_issued.avg.per_cycle_active`

**Worked example (V52):** ncu showed solo FFMA `pipe_fma = 97.58 %`,
solo LOP3 `pipe_alu = 99.45 %`, dual `pipe_alu = 98.0 % AND pipe_fma =
49.39 %` simultaneously. Sum = 147 % — decisive proof that the pipes
overlap freely. No amount of wall-clock reasoning could have settled
this; ncu pipe metrics did it in one ncu run.

### Rule 9 — If too-good-to-be-true: it is

Specific too-good signs:

- BW > theoretical
- TFLOPS > theoretical
- Latency < hardware unit minimum (e.g., L2 latency < 200 cy)
- "Same-warp dual-issue" > 100 % gain
- "Multicast pipelined deeper than 1 stage helps" (V48 disproved this)
- "cudaGraph single-node speedup" (V9 disproved this)

Each of these has been claimed in some catalog draft and later
retracted.

### Rule 10 — Multi-method agreement required for HIGH confidence

A claim is HIGH only if AT LEAST 3 of:
- wall-clock cudaEvent
- ncu pipe metric
- ncu memory-traffic metric
- SASS-verified instruction count
- multiple-recipe baseline (≥ 2 kernel variants)
- reproduced 3× within 1 %

agree. Anything less is MED.

**Worked example (V49):** Had only wall-clock + reproducibility. No ncu,
no SASS, no second-recipe baseline. Was published HIGH. After
W3b/W4/W5/W6, downgraded to LOW for the architectural claim, then
RETRACTED entirely with V52's `pipe_alu + pipe_fma = 147 %`. The
3-method requirement, applied at W1+W2, would have prevented the
entire 5-wave detour.

### Rule 11 (NEW from V52) — ≥ 64 ops/type body for dual-issue

For any "co-issue of pipes X+Y yields Q % of summed peak" claim:
inner body must have at least 64 ops of each type per loop iteration.
Below 64, branch + loop-counter overhead contaminates the ALU pipe
being measured.

**Quantification:** V49's 8 ops/type inner body had ~12.5 % loop
overhead (BRA + UIADD3 + UISETP per 16 ops). V8/V52's 128 ops/type
inner body has ~1.2 % loop overhead. The 11.3 percentage-point delta
mostly explains the V49 67 % vs V8 97.6 % solo FFMA gap.

### Rule 12 (NEW from W4-W6) — Standardize denominators

For HBM:
- **7.68 TB/s** = spec post-ECC (8.000 Gbps × 8192 bits ÷ 1.0625);
  use for cross-vendor.
- **7.67 TB/s** = this-device post-ECC (7.992 Gbps × 7680 bits ÷
  1.0625); use for SoL on this part.
- **7.31 TB/s** = empirical pure-direction peak (V32); use ONLY when
  framing as "% of best-known recipe", never as "spec" or "theoretical".

For FFMA peak: **76.96 TFLOPS** = 148 × 128 × 2 × 2.032; use boost
clock unless the row explicitly says otherwise.

For tensor cores (cuBLAS / tcgen05): cite the specific PTX form. Bare
"% of tensor" is meaningless.

### Rule 13 (NEW from CURIOSITY V2 audit) — Always git-verify "[x] done" hashes

CURIOSITY_LIST_V2 had **22/25 hallucinated hashes** (88 %). The author
filled in plausible-looking hashes from memory without verifying. V4-V8
git-verify rate: 100 %.

**Verification command:**

```bash
for h in 7647eba fbe1c18 501134a 8fd660a; do
    git rev-parse --short=7 "$h" 2>&1 | head -1
done
# Verifies these are real commits in the tree
```

Then verify the topic matches:

```bash
git log --oneline -1 "$h"  # confirm message matches the [x] claim
```

**Worked example:** CURIOSITY_LIST_V2 cited `c0c2d48` for "S2 tcgen05
alloc breakthrough". `git log --oneline -1 c0c2d48` returns no match.
Topic search for "S2 BREAKTHROUGH: tcgen05 alloc/dealloc WORKS" found
real commit `ec25f05`. Always topic-search BEFORE citing.

### §62.1 Why the 13-rule list

The original CLAUDE.md §3 list was 10 rules. Wave 6 added 3 new rules
based on incidents the 10-rule protocol failed to catch:

- Rule 11 (ops/type ≥ 64) was added because V49 passed all 10 rules
  but had a methodology bug that took 5 waves to find.
- Rule 12 (denominator standardization) was added because the V46
  "98.5 %" claim passed rule 1-10 but used a non-standard denominator.
- Rule 13 (git-verify hashes) was added because CURIOSITY_LIST_V2
  hallucinated 88 % of its hashes despite passing all the per-claim
  rigor checks.

Each new rule closes a class of bug the previous rules didn't catch.
Future waves are likely to add more rules; the current 13 are a
snapshot of the rigor lessons through wave 6.

### §62.2 Sources

- CLAUDE.md §3 (rules 1-10)
- `b300_clean/corrections/META_LESSONS.md` (rule 11 derivation)
- `b300_clean/corrections/HEADLINE_CORRECTIONS_v5.md` (rule 12)
- `b300_clean/corrections/CURIOSITY_LISTS_AUDIT.md` (rule 13)

---

## §63. Common measurement pitfalls (catalog)

**Answer:** Catalogue of every pitfall caught during the rigor sweep.
DCE / LICM / self-op chains / launch overhead / clock state / leftover
processes / pipe_tensor for tcgen05 / -lgc 2032 paradox / cuBLAS needs
cudaGraph / sub-agent critique catches what you miss / HBM_DATA_DEPENDENCE
5-7× wrong. Each entry: symptom, mechanism, defense.
`[🟢 HIGH · src: CLAUDE.md §3 + corrections/META_LESSONS.md + memory feedback_*]`

### §63.1 Dead Code Elimination (DCE)

**Symptom:**
- Measured time doesn't scale with iteration count.
- Measured BW > theoretical peak.
- Kernel runtime 0.001 ms on a "massive" test.
- Runtime is independent of kernel body size.

**Mechanism:** Compiler proves loop output is unused, eliminates
entire loop body. Result is a kernel that just does launch + return.

**Defense:**
- Unconditional STG of accumulator at end (under impossible-but-not-
  provable condition).
- Make accumulator depend on loop trip count (so eliminating the
  loop changes the output).
- Make addresses depend on `threadIdx.x`.
- Make trip count depend on runtime input.
- Sanity check: kernel runtime ≥ 1 ms.

**Worked example:** A bench_fma kernel reported 200 TFLOPS. Theoretical
peak is 76.96 TFLOPS. The 2.6× excess was DCE: the inner FFMA loop
wrote to a local register that was never STG'd. SASS showed the entire
loop body was eliminated, only the prologue and epilogue remained.

### §63.2 LICM (Loop-Invariant Code Motion)

**Symptom:**
- Measured BW or throughput is too high but not absurdly so (e.g.,
  1.5×–3× over expected).
- Pipe metric is inconsistent with pattern (e.g., LSU metric low when
  you expect high LSU).
- Timing scales sub-linearly with loop trip count.

**Mechanism:** Compiler observes that some computation in the loop is
loop-invariant and hoists it out. The hot loop becomes smaller than you
think.

**Defense:**
- Make the operation inputs depend on the loop counter (`acc = fma(acc,
  acc, k)` where `k` is loop counter).
- Use `volatile` on the address (last resort — kills lots of
  optimization).
- SASS-verify the inner loop instruction count.

**Worked example (V8 DSMEM):** Compile-time invariant offsets were
LICM'd / CSE'd. SASS showed `LD.E` once per CTA instead of once per
iteration. ncu wavefront count was 7200 vs expected 5.9 billion.
Real aggregate ≈ 40 GB/s/cluster, not 37 TB/s.

### §63.3 Self-op chains

**Symptom:**
- FFMA chain measures 2× the latency you expected.
- Single-chain ILP can't reach > 50 % of pipe throughput.

**Mechanism:** `fma a, a, a, a` (where `a` is the same register as the
destination) creates a register-port dependency. The hardware needs to
wait for `a` to be available as both source and destination. Latency
inflates by 1 cycle (the WAR through the register file).

**Defense:** Use distinct sources: `fma.rn.f32 d, a, b, c` with `a, b,
c` from different registers (or one register-immediate-immediate).

**Worked example:** V49 used `fma %0, %0, imm, imm` (1 RF source). V8
used `fma %0, %0, %1, %0` (2 RF sources, but `%1` constant-foldable).
Both work; the false claim was that V8's pattern was somehow worse.
Both kernels avoid the 3-distinct-source RF port pressure that capped
V6_C1 at 65–71 %.

### §63.4 Launch-overhead-dominated tests

**Symptom:**
- Tiny kernel reports 50 % of its expected throughput.
- Kernel runtime < 100 µs.
- "Throughput" doesn't change much when you double the inner loop count.

**Mechanism:** Kernel-launch latency is ~1.85 µs cold. If your kernel
runs in 10 µs, 18.5 % of measured time is launch overhead. ncu pipe
metrics measure "while-kernel-active" so they're correct, but
wall-clock-derived numbers are inflated.

**Defense:**
- Ensure runtime ≥ 10 ms for peak throughput tests.
- For latency tests, use `clock64` inside the kernel to exclude launch.
- Use cudaEvents on the stream, not on the kernel itself.

### §63.5 Clock state

**Symptom:**
- Cross-section of catalog has 6 % noise that's hard to explain.
- Same kernel reports different TFLOPS on different days.

**Mechanism:** Default boost = 2032 MHz. `nvidia-smi -lgc 2032` =
1920 MHz. Stuck at 1005 MHz under load with no lock = some throttle
condition. The 2032 / 1920 / 1005 trio differs by 50 %.

**Defense:**
- Always state which clock state.
- Sample `nvmlDeviceGetClockInfo` during run.
- `nvidia-smi -rgc` between runs to release any stuck locks.

**Worked example:** CLAUDE memory `feedback_clock_lock_works` documents
that `-lgc 1920` IS honored on B300 SXM6 (verified 510-1500 MHz);
apparent "1942 floor" was leftover background processes thrashing the
GPU.

### §63.6 Leftover processes

**Symptom:**
- ncu reports 5–8.5× higher cycles per op than expected.
- Power draw shows non-zero baseline before benchmark starts.
- `nvidia-smi` shows other processes on the GPU.

**Mechanism:** Unkilled benchmark processes hold contexts on some SMs.
Your benchmark gets the remaining SMs but ncu metrics are
chip-aggregate, so they include the zombie load.

**Defense:**
- `pkill -9 <bench-name>` between runs.
- `sleep 5–8` after pkill to let driver reclaim contexts.
- Check `nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv` shows
  idle baseline (~50 W, ~210 MHz).

**Worked example:** TCGEN05_PERF_WATTS single-trial table was
contaminated (5 leftover QuickRunCUDA processes); use
TCGEN05_PERFW_CLEAN_2TRIAL instead. NVFP4 K=96 numbers shifted by 2.4
TF/W after the cleanup.

### §63.7 `pipe_tensor` does NOT measure tcgen05

**Symptom:**
- ncu reports `pipe_tensor.cycles_active = 0 %` for a kernel that
  clearly uses tensor cores.
- Or reports 100 % for a kernel that doesn't.

**Mechanism:** `sm__pipe_tensor_cycles_active` measures the LEGACY
`mma.sync` tensor pipe. The NEW `tcgen05.mma` instructions go through a
different pipe and are NOT covered by this metric on sm_103a.

**Defense:** Use the explicit tcgen05 metric:
`smsp__inst_executed_pipe_tensor_subpipe_hmma_op_utchmma_utcqmma_utcomma_scope_2cta.sum`

(yes, the metric name is that long).

**Worked example:** A wave-1 row claimed "tcgen05 60 % MFU on K=96"
with `pipe_tensor.cycles_active = 60 %`. The same kernel run with the
explicit tcgen05 metric showed 90 %+. The 60 % was just the leftover
mma.sync activity from the cuBLAS warmup, not the tcgen05 hot loop.

### §63.8 The `-lgc 2032` paradox

**Symptom:**
- You ran `nvidia-smi -lgc 2032` to lock at boost.
- Throughput is 6 % lower than default-boost.

**Mechanism:** `-lgc 2032` locks BOTH boost ceiling AND base clock to
2032. But the SM clock then runs at the BASE clock = 1920 MHz (because
the boost-state machine is disabled).

**Defense:**
- For boost: do NOT lock; let default boost engage under sustained
  load.
- For 1920 reproducibility: use `-lgc 1920`.
- Sample `nvmlDeviceGetClockInfo` during run; if it reports 1920 when
  you expected 2032, you hit the paradox.

### §63.9 cuBLAS needs cudaGraph for sustained measurements

**Symptom:**
- cuBLAS GEMM benchmark reports lower TFLOPS than the spec sheet.
- ncu shows long inter-kernel idle gaps.

**Mechanism:** cuBLAS dispatches multi-kernel internal sequences. The
host-side dispatch overhead between kernels (~1.85 µs each) adds up to
significant idle time at small problem sizes.

**Defense:** Capture the cuBLAS call into a cudaGraph, then launch the
graph repeatedly. The graph batches host work, reducing per-iteration
overhead from 1.85 µs/kernel to 0.55 µs/kernel.

**Worked example:** NVFP4 K=96 cuBLAS bare = 10.8 PF; cuBLAS + cudaGraph
BPG=16 = 11.42 PF (76.2 % of 15 PF spec). The 6 % gap was per-call host
overhead.

### §63.10 Sub-agent critique catches what you miss

**Symptom:** A measurement looks clean and you're about to publish.

**Defense:** Spawn a sub-agent specifically to audit the methodology.
Phrase the prompt adversarially: "find every reason this measurement
might be wrong". The agent has fresh eyes and no ego investment.

**Worked example:** The 5-wave dual-issue zigzag (Appendix A). Each
wave caught real bugs the prior wave missed. Without the doubt-the-doubt
process, V49's 55 % would have shipped as canonical.

### §63.11 HBM_DATA_DEPENDENCE 5-7× wrong

**Symptom:** A catalog file claims "HBM data-dependence is < 50 W".

**Mechanism:** That file used a constant-pattern test (all-zero) which
doesn't exercise the toggle-energy curve.

**Defense:** Use random data with controlled popcount; sweep d=0..32
and observe the bell curve peak at d=16 (240–554 W swing). HBM is
heavily data-dependent; the "< 50 W" finding was an artifact of
non-toggling data.

`b300_clean/HBM_DATA_DEPENDENCE.md` is RETRACTED. Use POPCOUNT_3TIER
or similar. CLAUDE memory `project_b300_power_data_dep` is the
authoritative summary.

### §63.12 Other named pitfalls

| Pitfall | Symptom | Defense |
|---|---|---|
| **Self-op chains** | Latency 2× expected | Distinct registers per source |
| **3-source FFMA RF port** | FFMA caps at 65 % of peak | Use 2-source pattern (Rd × imm + Rd) |
| **`#pragma unroll 1`** | Body unrolls anyway | Check SASS; compiler ignores when it sees benefit |
| **TMA + prefetch.L2** | −27 % BW (V42) | Never combine bulk TMA with explicit prefetch |
| **Multicast pipeline depth > 1** | Slightly slower | Multicast engine is single |
| **cluster ≥ 32** | Silently no-op | Check `cudaOccupancyMaxActiveClusters` |
| **NVRTC narrow-cvt PTX** | Compile fails on sm_103a | Migrate to PTX 8.7 forms |
| **`cuLibrary*` 6.5× claim** | Single source, not re-verified | Re-measure on current driver |

### §63.13 Sources

- CLAUDE.md §3-4 (rigor protocol)
- `b300_clean/corrections/META_LESSONS.md` (5-level zigzag)
- `b300_clean/corrections/HEADLINE_CORRECTIONS_v5.md` (rule 11-13)
- CLAUDE memory `feedback_b300_pitfalls`
- CLAUDE memory `project_b300_power_data_dep` (HBM_DATA_DEPENDENCE
  retraction)

---

## §64. Cross-tool cheat-sheet

**Answer:** QuickRunCUDA harness for rapid kernel iteration · NVRTC for
JIT compile · ncu for pipe / memory metrics · cuobjdump / nvdisasm for
SASS · nvprof (deprecated) → use ncu · NVML for clock / power · nvidia-smi
for state · `/usr/local/cuda/bin/` for the binaries. Common one-liners
follow.  `[🟢 HIGH · src: CLAUDE.md + utils/cuda_helper.h + utils/nvmlClass.h]`

### §64.1 QuickRunCUDA — rapid kernel iteration

```bash
# Build the host once
make
# Run a kernel
./QuickRunCUDA tests/bench_fp32_fma.cu \
    -t 256 -b 148 \
    -A $((64 * 1024 * 1024)) \
    -B $((64 * 1024 * 1024)) \
    -C $((64 * 1024 * 1024)) \
    -T 100 \
    -P 1024 -U TFLOPS -L 76.96 \
    -r --randomMask 0xffffffff
```

Common flags:

| Flag | Meaning |
|---|---|
| `-t N` | Threads per block |
| `-b N` | Blocks per grid |
| `-p` | Persistent (gridDim = SM count) |
| `-A/-B/-C N` | Buffer sizes in dwords |
| `-r --randomB` | Fill A/B with random data |
| `-T N` | N timed iterations |
| `-P N -U TFLOPS -L X` | Throughput multiplier + unit + speed-of-light |
| `-N N` | Per-thread multiplier |
| `--l2flush {0,1,2}` | None / at start / every run |
| `--timesPerRun` | Print every iteration time |
| `-H "<str>"` | Prepend text to kernel source (inject defines) |
| `--reuse-cubin` | Skip NVRTC, load `output.cubin` directly |
| `--clock-speed N` | Lock GPU clock via NVML (0=no force, 1=unlocked) |

### §64.2 ncu — pipe / memory metrics

```bash
# Common dual-issue / pipe-utilization pass
ncu --metrics \
  smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__inst_issued.avg.per_cycle_active,\
smsp__warps_active.avg.pct_of_peak_sustained_active \
  ./QuickRunCUDA tests/bench_fp32_fma.cu -t 256 -b 148 -T 1

# HBM peak sustained
ncu --metrics \
  dram__bytes_read.sum.per_second,\
dram__bytes_write.sum.per_second,\
lts__t_sector_hit_rate.pct \
  ./QuickRunCUDA tests/bench_hbm_read.cu -t 256 -b 148 -T 1

# Atomic intensity
ncu --metrics \
  lts__t_sectors_op_atom.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_lg_op_atom.sum \
  ./QuickRunCUDA tests/bench_atom.cu -t 256 -b 148 -T 1

# Section-mode for full picture
ncu --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis \
    --section SchedulerStats \
    ./bench
```

Output format options:

```bash
ncu --csv --log-file out.csv ./bench
ncu --print-summary per-gpu ./bench
```

### §64.3 cuobjdump / nvdisasm — SASS inspection

```bash
# Disassemble a kernel from a binary or cubin
cuobjdump --dump-sass ./QuickRunCUDA > QuickRunCUDA.sass

# Or from the auto-emitted SASS file (after each compile)
ls sass/
# bench_fp32_fma.sass  bench_fp32_fma_<hash>.cubin

# Just a specific kernel symbol
cuobjdump --dump-sass --function "_Z6kernelPfS_S_iii" ./bench

# Direct nvdisasm of cubin
nvdisasm output.cubin

# Get PTX too
cuobjdump --dump-ptx ./bench
```

For NVRTC kernels (compiled at runtime), QuickRunCUDA writes the cubin
to `output.cubin` and SASS to `sass/<basename>_<hash>.sass`.

To get a single emit-then-disassemble for a standalone:

```bash
nvcc -arch=sm_103a -O3 -std=c++17 -keep my_kernel.cu -o my_kernel
ls my_kernel.{ptx,cubin,sass}
cuobjdump --dump-sass my_kernel | less
```

### §64.4 nvprof (deprecated)

`nvprof` is deprecated as of CUDA 11. **Use ncu instead.** Old
`nvprof --metrics flop_count_sp` becomes:

```bash
# new: ncu equivalent
ncu --metrics smsp__sass_thread_inst_executed_op_ffma_pred_on.sum \
    ./bench
```

### §64.5 NVML — clock / power live sampling

`utils/nvmlClass.h` wraps the most common operations:

```cpp
nvmlClass nvml(0);  // device 0
nvml.lockClock(1920);                  // lock SM clock to 1920 MHz
unsigned int sm_mhz = nvml.getClockSM();
unsigned int power_w = nvml.getPowerW();
nvml.unlockClock();
```

Or directly:

```cpp
#include <nvml.h>
nvmlInit();
nvmlDevice_t dev;
nvmlDeviceGetHandleByIndex(0, &dev);

unsigned int sm_clock;
nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &sm_clock);

unsigned int power_mW;
nvmlDeviceGetPowerUsage(dev, &power_mW);

// Lock
nvmlDeviceSetGpuLockedClocks(dev, 1920, 1920);
// ... benchmark ...
nvmlDeviceResetGpuLockedClocks(dev);

nvmlShutdown();
```

### §64.6 nvidia-smi — state & quick queries

```bash
# Verify B300 device + clock + power right now
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,power.draw,utilization.gpu \
           --format=csv,noheader

# Lock clock
nvidia-smi -lgc 1920          # lock to 1920 MHz
nvidia-smi -lgc 2032          # PARADOX: actually pins to 1920
nvidia-smi -rgc               # reset / unlock

# Memory clock
nvidia-smi -lmc 1593          # lock memory clock
nvidia-smi -rmc               # reset memory clock

# Power limit
nvidia-smi -pl 1100           # set power cap to 1100 W
nvidia-smi -pl default        # restore

# Reset entire device (requires sudo, no users)
nvidia-smi --gpu-reset

# Persistence mode (kernel module stays loaded)
nvidia-smi -pm 1

# Full enumeration (verbose)
nvidia-smi -q | head -100

# Compute mode (Default = 0, Process Exclusive = 3)
nvidia-smi -c 0
```

### §64.7 Common one-liner workflows

**Full rigor sweep on a new kernel:**

```bash
pkill -9 your_bench && sleep 6
nvidia-smi -rgc && sleep 2
nvcc -arch=sm_103a -O3 -std=c++17 -keep bench.cu -o bench
cuobjdump --dump-sass bench > bench.sass
ncu --metrics \
  smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,\
dram__bytes_read.sum.per_second,\
sm__warps_active.avg.pct_of_peak_sustained_active \
  ./bench 2>&1 | tee bench_ncu.txt
./bench  # wall-clock run
```

**Fastest "is the GPU idle?" check:**

```bash
nvidia-smi --query-gpu=power.draw,clocks.sm,utilization.gpu \
           --format=csv,noheader,nounits
# expected idle: ~50 W, ~210 MHz, 0 %
# if you see > 100 W or > 1000 MHz, something is running
```

**Fast kernel iteration loop (with QuickRunCUDA server mode):**

```bash
# in terminal 1
./QuickRunCUDA --server &
# in terminal 2
echo "tests/bench_fp32_fma.cu -t 256 -b 148 -T 100" \
  > /tmp/quickruncuda_cmd
cat /tmp/quickruncuda_resp
# (subsequent invocations skip CUDA init = ~250 ms)
```

**Live clock sample during run:**

```bash
# terminal 1
./bench &
# terminal 2
while ps -p $! > /dev/null 2>&1; do
    nvidia-smi --query-gpu=clocks.sm,power.draw \
               --format=csv,noheader,nounits
    sleep 0.1
done
```

### §64.8 Sources

- CLAUDE.md (build/run/server-mode docs)
- `utils/cuda_helper.h` (NVRTC wrapper)
- `utils/nvmlClass.h` (NVML wrapper)
- `utils/CLI11.hpp` (CLI parser)

---

## §65. Time-stamping + version

**Answer:** This document is dated **2026-04-22**. It is the wave-6
post-V52 synthesis. It supersedes `B300_TRUE_REFERENCE.md` and
`corrections/HEADLINE_CORRECTIONS_v5.md` for top-line claims; defer to
those for full per-row context. Future sessions: re-verify HIGH-conf
entries before quoting (memory can become stale; HW behavior can shift
across drivers).  `[🟢 HIGH · src: this document, dated 2026-04-22]`

### §65.1 Document version

- **Version:** AC v1 (post-V52, post-W6)
- **Snapshot date:** 2026-04-22
- **Driver:** CUDA 13.2 V13.2.78 / Driver 580.126.09
- **Hardware:** NVIDIA B300 SXM6 AC (sm_103a, CC 10.3, 148 SMs, 8 HBM3E
  stacks of 12-Hi each, 7680-bit fused bus, 288 GB)
- **Default clock state:** sustained boost 2032 MHz (no `nvidia-smi
  -lgc`) unless explicitly stated otherwise

### §65.2 Document supersession chain

This document supersedes:

- `b300_clean/B300_TRUE_REFERENCE.md` (wave-2 snapshot; rows still
  valid but framing is pre-V52)
- `b300_clean/corrections/HEADLINE_CORRECTIONS.md` (v1, wave-1+2)
- `b300_clean/corrections/HEADLINE_CORRECTIONS_v2.md` (wave-3c)
- `b300_clean/corrections/HEADLINE_CORRECTIONS_v3.md` (wave-4)
- `b300_clean/corrections/HEADLINE_CORRECTIONS_v4.md` (wave-5)
- `b300_clean/corrections/HEADLINE_CORRECTIONS_v5.md` (wave-6,
  immediately superseded by this canonical reference)
- `b300_clean/corrections/MASTER_INDEX.md` (v1, wave-1+2)
- `b300_clean/corrections/MASTER_INDEX_v2.md` (wave-3c)
- All 17 `0X_*_CORRECTED.md` files (wave-1+2 / wave-3a)
- All `*_INCONSISTENCY_LOG.md` files

For the rigor sweep audit trail: see Appendix D (provenance map).

### §65.3 Re-verification policy

For any HIGH-confidence entry, before quoting it in a new context:

1. **Hash check** any commits cited (Rule 13).
2. **Run cudaGetDeviceProperties** to confirm the hardware identifiers
   match what the row assumed.
3. **For HBM denominators**, confirm `memoryBusWidth` is still 7680
   (driver upgrades or rebinning could change this).
4. **For tensor / mma claims**, re-run the test against the current
   cuBLAS / driver — the cuBLAS internal kernel selection changes
   between versions.
5. **For NVRTC PTX acceptance claims**, retest with the current CUDA
   release; `cvt.rn.satfinite.e2m1x4.f32` may eventually be accepted
   in CUDA 13.3+.

### §65.4 What might invalidate this document

| Scenario | Affected entries | Re-test required |
|---|---|---|
| New CUDA release (13.3+) | NVRTC PTX acceptance, ncu metric definitions, some pipe placements | All "ncu confirms X" rows |
| New B300 driver | Power/clock behavior, NVML semantics | All `power.draw` / clock rows |
| Replaced HBM stack (RMA) | bus width might change | All HBM denominator rows |
| Different SKU (B300 SXM6 non-AC) | bus width = 8192 instead of 7680 | All HBM "% of peak" rows |
| New ncu version | metric definitions | All `pipe_X_cycles_active` rows |
| Power-cap change | sustained throughput rows | All sustained TFLOPS rows |

### §65.5 Sources

- All files cited in §1-65 and Appendices.
- Source-of-record: `git log --since=2026-01-01 --oneline` on this
  repo's `f2fp-deep-dive` branch.

---

# APPENDIX A. The 5-Level Dual-Issue Zigzag (Case Study)

> The single most instructive episode of the rigor sweep. It illustrates
> the **doubt-the-doubt** dynamic, the architectural-vs-artifact split,
> the methodology gates that became Rules 11–13, and what the empirical
> ncu anchor finally settled. This appendix is intentionally long; treat
> it as a worked case study to apply to any future "this measurement
> doesn't smell right" intuition.

---

## A.1 The narrative arc

The V49/V50 dual-issue claim went through **5 levels of doubt** before
settling. The actual measurement (V49 wall-clock) never changed. Only
the **interpretive framework** changed. The final settlement came from
**one careful empirical test (V52)** with V8-style methodology + ncu pipe
metrics — a single test that took roughly the time of one armchair doubt
wave.

Here is the chronology:

| Wave | Verdict | Mechanism cited | What was right | What was wrong |
|---|---|---|---|---|
| **W1+W2** (V49/V50, commits 501134a / fbe1c18) | HIGH 55 % / 74 % "B300 ALU pipes share scheduler dispatch" | reproducibility of clock64 ratio | the measurement is reproducible | reproducibility ≠ validity; no SASS audit, no ncu, no matched solo baseline |
| **W3b doubt** (DUAL_ISSUE_DOUBT_REPORT.md) | LOW | under-occupancy at 2 warps/SMSP × 8 ILP | numbers are unsafe | wrong specific mechanism; AND wrong implicit inference that the cap exists at 55 % |
| **W4 meta-doubt** (META_DOUBT_REPORT.md) | MED (re-promote) | V8 hits 97.6 % at the same 2 warps/SMSP, falsifying under-occupancy | right falsification of W3b's mechanism | conflated "two kernels at same occupancy" with "two kernels with same methodology" |
| **W5a SASS-verify** (SASS_VERIFY_DUAL_ISSUE.md) | LOW (re-downgrade) | inner body 8 ops/type + BRA + UIADD3/UISETP contaminates the ALU pipe being measured | right mechanism for the artifact | implicitly carried W3b's dispatch-cap inference forward; never tested whether the cap exists at all |
| **W6 V52 + ncu** (V52_RUN_RESULTS.md) | **HIGH (architectural truth) + RETRACT-NUMBER** | `pipe_alu + pipe_fma = 147 %` simultaneously per ncu | settled | (potentially): could in principle be wrong if ncu metric semantics misinterpreted |

---

## A.2 W1+W2 — The original V49/V50 measurement

### A.2.1 What was measured

V49 (`tests/standalone/v49_dual_pipe.cu`, commit 501134a) and V50
(`tests/standalone/v50_warp_specialized.cu`, commit fbe1c18) measured
the throughput of a kernel running FFMA + LOP3 in the same warp (V49) or
in warp-specialized fashion (V50, with some warps doing FFMA and others
doing LOP3).

Both kernels used:

- `__launch_bounds__(128, 2)` = 2 CTAs/SM × 4 warps/CTA = 8 warps/SM = 2
  warps/SMSP.
- ILP=8: 8 independent FFMA chains and 8 independent LOP3 chains.
- N_ITERS=5000 outer iterations.
- `#pragma unroll 1` on the outer loop.
- Inner body: 8 FFMA + 8 LOP3 (V49) or 8 FFMA-only / 8 LOP3-only per
  warp (V50).

The measurement was wall-clock via cudaEvent + a clock64 inside the
kernel for cross-check. Results:

| Test | Throughput (Glane/s) | % of summed pipes |
|---|---:|---:|
| V49 solo FFMA (OP=0) | 25.2 | 67.0 % of 37.6 |
| V49 solo LOP3 (OP=1) | 16.7 | (n/a — single pipe ratio unclear) |
| V49 dual FFMA+LOP3 (OP=2) | 22.6 | 55.0 % of 41.0 (sum of solo) |
| V50 warp-spec (OP=2 split) | 30.5 | 74.0 % of 41.0 |

The headline conclusion: **"B300 ALU pipes share scheduler dispatch;
same-warp dual-issue caps at 55 %, warp-specialized at 74 %."**

### A.2.2 What was right

The measurements themselves were reproducible within 1 % across
back-to-back runs. The relative ordering (warp-spec > same-warp) was
real and architecturally meaningful (warp-spec has cleaner per-warp
homogeneous bodies).

### A.2.3 What was wrong

- **Reproducibility ≠ validity.** A deterministic kernel with a
  methodology bug produces a deterministic wrong answer.
- **No SASS audit.** Nobody checked what SASS the kernel actually
  emitted. As we'll see in W5a, the inner body had branch + loop-counter
  ops that consumed ALU dispatch slots.
- **No ncu cross-check.** Wall-clock GLane/s ratios are NOT decisive for
  dispatch claims. They confound dispatch with per-instruction issue
  cadence. (Solo LOP3's 2-cycle cadence at 16.8 K Glane/s does NOT mean
  the ALU pipe is at 50 %; ncu shows it's at 99.5 %.)
- **No matched solo baseline.** V49's solo FFMA at 67 % was compared
  against an abstract "100 % theoretical peak" — but V8's solo FFMA at
  the same occupancy hit 97.6 %. The V49 solo was itself anomalously
  low.
- **No second-recipe baseline.** Only one kernel implementation was
  used. A second implementation with the same architectural target but
  different methodology would have surfaced the methodology bug.

### A.2.4 The downstream blast radius

The "55 %/74 % dual-issue" claim was published as HIGH in:

- `B300_TRUE_REFERENCE.md` §6 dual-issue ladder.
- `corrections/04_fp32_peak_CORRECTED.md` §dual-issue.
- `corrections/M_SYNTHESIS_CORRECTIONS.md` (M8 PIPE_OVERLAP_MATRIX
  superseded).
- All M-synthesis docs that quoted the 55 %/74 % ratio.
- Implicit in any "B300 dispatch is capped at 128 inst/SM/cy" downstream
  claim.

A reader landing on any of these docs in waves 1–5 would have published
a wrong number.

---

## A.3 W3b — DUAL_ISSUE_DOUBT_REPORT (LOW for under-occupancy)

### A.3.1 What W3b argued

W3b (the DUAL_ISSUE_DOUBT_REPORT.md adversarial agent) noticed the
following anomaly:

- V49's solo FFMA = 25.2 Glane/s = 67.0 % of theoretical 37.6 Glane/s.
- A "well-saturated" kernel should reach 90 %+ at 2 warps/SMSP with 8
  ILP (FFMA latency = 4 cy, ILP = 8 should hide it).
- Therefore V49's baseline is itself broken; the 55 % dual / 67 % solo
  ratio = 82 % "of solo FFMA peak" might be the real architectural
  number, but the 55 % / 100 % framing is wrong.

W3b hypothesized **under-occupancy at 2 warps/SMSP × 8 ILP** as the
mechanism: not enough warps to hide back-to-back FFMA latency at the
chosen ILP. Verdict: LOW for V49/V50 dual-issue.

### A.3.2 What was right

- Correctly noticed that V49's solo FFMA was anomalously low.
- Correctly inferred that the published % was suspect because the
  baseline was broken.
- Cited M8 PIPE_OVERLAP_MATRIX (MUFU+FFMA ≈ 100 %, HMMA+LDS ≈ 73-96 %)
  as **counter-evidence** to the "4-wide dispatch cap" interpretation.

### A.3.3 What was wrong

- The cited mechanism ("under-occupancy") was **wrong**. V8 hits 97.6 %
  FFMA at the SAME 2 warps/SMSP geometry. So under-occupancy alone
  cannot explain V49's 67 % solo.
- The implicit further inference that "the architectural cap exists at
  the V49 measured value" was **never argued explicitly** but became
  embedded in the canonical reading: "V49 has a dispatch cap that's
  worse than W1+W2 thought, so it's LOW".

### A.3.4 What the next wave caught

W4 (meta-doubt) ran the comparison: are V8 and V49 actually at the same
occupancy? Yes (both `__launch_bounds__(*, *)` = 2 warps/SMSP). So
under-occupancy is falsified by V8's 97.6 %. W3b's mechanism does not
explain the data.

---

## A.4 W4 — META_DOUBT_REPORT (MED for "honest measurement")

### A.4.1 What W4 argued

W4 (the META_DOUBT_REPORT.md, auditing the doubt reports themselves)
noted:

- W3b's specific mechanism (under-occupancy) is **falsified** by V8's
  97.6 % at the same occupancy.
- Therefore the W3b LOW verdict was **right answer for wrong reason**.
- The numbers themselves (V49 55 %, V50 74 %) are **honest measurements**
  of an under-determined architectural question. Re-promote to MED.

### A.4.2 What was right

- The falsification of W3b's specific mechanism IS valid. V8 at 97.6 %
  is a real counter-example.
- Correctly identified that the re-grade decision and the mechanism
  attribution are separate questions.

### A.4.3 What was wrong

- **Conflated "two kernels at same occupancy" with "two kernels with
  same methodology".** V8 has 128-deep inner unroll and
  `__launch_bounds__(256, 1)`. V49 has 8-deep with branch every 8 ops
  and `__launch_bounds__(128, 2)`. Same warps/SMSP, but very different
  inner body structure.
- **Re-promoted V49/V50 to MED** based on this erroneous comparison.
  The MED verdict was wrong because the methodology gap (not occupancy)
  was the real issue.

### A.4.4 What the next wave caught

W5a (SASS-verify) inspected the actual emitted SASS for both kernels
and found: V49's inner body has 8 FFMA + 8 LOP3 + UIADD3 + UISETP + BRA
per outer iteration; V8's has 128 FFMA per outer iteration. The
branch-overhead amortization differs by ~16×.

---

## A.5 W5a — SASS_VERIFY_DUAL_ISSUE (LOW for loop-overhead contamination)

### A.5.1 What W5a argued

W5a (the SASS-verify wave) compiled both V49 and V8 with `nvcc -keep`
and inspected the SASS for the inner loop:

**V49 OP=2 inner body (8-deep):**
```
/*0200*/-/*0270*/   8× FFMA Rk, Rk, R0.reuse, 0.5
/*0280*/-/*02e0*/   8× LOP3.LUT Rk, Rk, 0xa5a5a5a5, R18, 0x96, !PT
/*02f0*/            UIADD3 R3, R3, 0x1, RZ      ; loop counter
/*0300*/            UISETP.NE R4, R3, c[0x0][0x180]
/*0310*/            BRA.U UP0, 0x1b0            ; branch back
```

Body composition: 8 FFMA + 8 LOP3 + 1 UIADD3 + 1 UISETP + 1 BRA = 19
inst, 16 of which are the body, 3 of which are loop overhead. Loop
overhead = **3/19 = 15.8 %**, of which UIADD3 and UISETP go to the ALU
pipe (the same pipe being measured for LOP3).

**V8 inner body (128-deep):**
```
/*0...0*/-/*0...f*/   128× FFMA Rd, Rsrc, Rd, Rd
/*0...g*/             UIADD3 R3, R3, 0x1, RZ
/*0...h*/             UISETP.NE R4, R3, c[0x0][0x180]
/*0...i*/             BRA.U UP0, ...
```

Body composition: 128 FFMA + 1 UIADD3 + 1 UISETP + 1 BRA = 131 inst.
Loop overhead = **3/131 = 2.3 %**.

W5a concluded: V49's measurement is contaminated by branch + loop-counter
overhead consuming ~13 percentage points more ALU pipe slots than V8's
methodology. The V49 67 % vs V8 97.6 % gap (= 30.6 percentage points) is
mostly explained by this difference.

Verdict: LOW for V49/V50 (re-downgrade from MED).

### A.5.2 What was right

- **Identified the actual mechanism** (loop-overhead contamination) for
  the V49/V8 solo gap.
- **SASS-grounded** rather than ratio-grounded — strongest of the four
  preceding verdicts.
- Correctly framed as "V49 is contaminated; the architectural question
  remains OPEN until V52".

### A.5.3 What was wrong

- **Implicitly carried W3b's dispatch-cap inference forward.** W5a
  concluded "the ratio is wrong" but didn't address "is the dispatch
  cap itself a real architectural feature?"
- Wrote: "Re-run V49 OP=2 / V50 OP=2 with: 1. Inner unroll depth ≥ 64
  ops per type … Until then, the '55%/74% same-warp vs warp-specialized'
  gap is **measurement artifact, not architectural finding**."

  This formulation **leaves open the possibility** that even with
  proper methodology, dispatch is capped at some value. W5a never
  predicted "if you do this right, dispatch is uncapped" — it just
  said "do it right and remeasure".

### A.5.4 What the next wave caught

V52 + ncu showed the answer: **dispatch is NOT capped between FMA and
ALU pipes; they overlap freely.** `pipe_alu + pipe_fma = 147 %`
simultaneously. The "dispatch cap" was a phantom; W5a was right about
the contamination but didn't go far enough.

---

## A.6 W6 — V52_RUN_RESULTS (HIGH for free overlap)

### A.6.1 What V52 did

V52 (`tests/standalone/v52_dual_issue_clean.cu`) implemented the W5a
recommendations explicitly:

1. **Inner unroll depth = 128 ops/type per outer iteration** (matching
   V8).
2. **`__launch_bounds__(256, 1)`** (matching V8, not V49's `(128, 2)`).
3. **Anti-DCE: STG of accumulator XOR**, not clock-diff conditional.
4. **ncu pipe_fma + pipe_alu simultaneously** — the diagnostic the
   entire 5-wave debate was missing.

V52 ran 18 kernel templates: 3 modes (solo FFMA / solo LOP3 / dual) × 3
ILPs (4 / 8 / 16) × 2 BPS (1 / 2 CTA per SM).

### A.6.2 SASS verification

Inspected mode=2 (dual), ILP=8, BPS=1 — the V8-recipe target:

```
FFMA: 128
LOP3: 136     (128 in loop body + ~8 in init/anti-DCE)
UIADD3: 1
UISETP: 1
BRA: 1
STG: 2
```

V49's contaminated body had **8 FFMA + 8 LOP3 + 1 UIADD3 + 1 UISETP + 1
BRA** (loop overhead ≈ 12.5 % of body). V52's body has loop overhead ≈
**1.2 %** — within V8's amortization regime.

Inner FFMA encoding (mode=0):
```
FFMA R11, R11, 1.5, R11
FFMA R12, R12, 1.5, R12
...
```

Same 2-source self-feed pattern V8 uses (Rd × IMM + Rd).

### A.6.3 Wall-clock results

Geometry A: 148 blocks × 256 thr (BPS=1, 2 warps/SMSP — V8 recipe)

| ILP | solo FFMA Glane/s | solo LOP3 Glane/s | dual total Glane/s | dual / max(solo) | dual / sum(solo) |
|---:|---:|---:|---:|---:|---:|
| 4  | 32 060 | 16 428 | 32 525 | **101.5 %** | 67.0 % |
| 8  | 32 706 | 16 824 | 33 114 | **101.2 %** | 66.9 % |
| 16 | 33 172 | 16 763 | 28 173 |  84.9 % | 56.4 % |

Solo FFMA hits 84-87 % of 76.97 TFLOPS (V8 reaches 97.7 % with N_OUTER
≥ 1M; V52 uses 1k-4k outer iters, so loop tail / launch overhead leaves
~10pp on the table). The relative dual-vs-solo ratios are unaffected.

### A.6.4 The decisive ncu metrics (Geometry A, all ILPs)

```
config (mode,ILP,BPS,N_OUTER)   pipe_alu%   pipe_fma%   inst_issued/cy   alu+fma
<0,4,1,4096>  solo FFMA           0.01       95.39        1.00            95.40
<1,4,1,4096>  solo LOP3          97.27        0.76        0.52            98.03
<2,4,1,4096>  dual                96.17       48.84        0.99           145.01

<0,8,1,2048>  solo FFMA           0.02       97.58        1.00            97.60
<1,8,1,2048>  solo LOP3          99.45        0.39        0.51            99.84
<2,8,1,2048>  dual                98.00       49.39        1.00          147.39

<0,16,1,1024> solo FFMA           0.04       98.66        1.00            98.70
<1,16,1,1024> solo LOP3          99.74        0.20        0.51            99.94
<2,16,1,1024> dual                87.96       44.15        0.89          132.11
```

### A.6.5 Interpretation — the architectural truth

Both pipes ARE running concurrently. Each FMA-pipe and ALU-pipe slot
fires ≈98 %/cycle when the kernel has work for it. The reason `dual ≈
max(solo)` in wall-clock GLane/s is **not** a shared dispatch port —
it's because **LOP3 issues at half the rate of FFMA per cycle**:

- `smsp__inst_issued.avg.per_cycle_active` = **1.00** for FFMA-only,
  **0.51** for LOP3-only, **1.00** for dual.
- `smsp__pipe_alu_cycles_active` = **97-99 %** for solo LOP3 — the ALU
  pipe is saturated, but each LOP3 takes ~2 issue cycles.
- In dual mode, FFMA fills the 50 % of slots LOP3 leaves idle:
  pipe_alu+pipe_fma = **145-147 %** at ILP=8.

So:

- The **FMA pipe and ALU pipe are physically separate** — they overlap
  freely.
- **LOP3 has a 2-cycle issue cadence per SMSP** (likely the fundamental
  ALU pipe rate, or LOP3-specific). Solo LOP3 throughput is ~16.8 K
  Glane/s = ~43 % of the 38.5 K Glane/s "1 inst/cy/SMSP" upper bound —
  it is actually 100 % of its own real ceiling (which is half FFMA's).
- Dual mode reaches **inst_issued = 1.00/cy and pipe_fma+pipe_alu =
  147 %** — this is **clear dual-issue at the dispatch port**, not a
  shared cap.
- The "harmonic mean" framing in V49 was wrong: the pipes don't share,
  but LOP3's intrinsic 2-cycle issue means dual is bottlenecked by
  FFMA's slot count, with LOP3 piggy-backing in the otherwise-idle ALU
  port.

ILP=16 dual drops to alu+fma = 132 % — this is REGISTER PRESSURE (16
floats + 16 ints = 32 live regs/thread × 256 thr ≈ saturates the 64K
RF). Not architectural; an ILP=4 or 8 result is the architectural
answer.

### A.6.6 Verdict

| Question | Answer |
|---|---|
| Does FFMA + LOP3 dual-issue work on B300? | **YES.** Both pipes fire at 98 %+ simultaneously. |
| Is V49's "55 % same-warp ceiling" architectural? | **NO.** Methodology artifact (8-deep loop, ALU loop overhead). |
| Is V50's "74 % warp-specialized ceiling" architectural? | **NO.** Same root cause; warp-split helped because it hid loop overhead. |
| What's the real dispatch behaviour? | 1 inst/SMSP/cy on each pipe, FREELY OVERLAPPING. LOP3 happens to need 2 issue slots per inst → solo LOP3 = ½× solo FFMA but dual = 1× FFMA + ½× LOP3 = 1.5× FFMA-issue-rate worth of work. |
| Is the "B300 dispatch capped at 128 inst/SM/cy" claim wrong? | **PARTIALLY.** Single-pipe IS capped at 128/SM/cy (= 1/SMSP/cy × 4 SMSPs × 32 lanes), but the FMA + ALU pipes can BOTH be at 128 simultaneously, so total inst/SM/cy can reach ~256. The "128 ceiling" is per-pipe, not per-SM. |

**Final dual-issue confidence: HIGH.**
- Three independent runs reproducible within 1 %.
- ncu pipe_fma + pipe_alu sum = 145-147 % directly proves overlap.
- ncu inst_issued = 1.00/cy in dual mode (vs 0.51/cy solo LOP3) proves
  dispatch can issue more when pipe diversity allows.
- SASS verified — V49's loop-overhead contamination is gone (1.2 % vs
  12.5 %).

V49's 55 % and V50's 74 % **must be retracted** as architectural claims
about B300 dispatch. They were measuring loop-overhead-contaminated
artifacts.

---

## A.7 The four lessons

The 5-level zigzag delivers four lessons that together constitute the
meta-rigor framework for any future B300 architectural claim.

### Lesson 1 — Reproducibility-only verdicts (W1+W2) miss methodology

W1+W2 graded the V49/V50 numbers HIGH because the kernel ran cleanly
and the ratios were stable across reruns. **Stability of a measurement
is not evidence that the measurement measures what its label claims**.
The V49 inner body was contaminated by branch/loop-counter dispatch,
but the contamination was deterministic — so it produced a perfectly
stable wrong answer. A single-source HIGH grade is fragile; promote
only after methodology audit.

The correct gate to apply at W1+W2: 3-method verification (Rule 10).
Wall-clock reproducibility alone = MED, not HIGH.

### Lesson 2 — Mechanism-inference verdicts (W3b "under-occupancy") can be falsified by counter-example

W3b correctly noticed something was off (denominator broken), then
**inferred** the mechanism (occupancy). The mechanism was check-able:
is there a kernel that hits high FFMA throughput at the **same**
geometry? There was — V8.

W4 ran the comparison and the inferred mechanism failed.

The correct gate to apply at W3b: state the mechanism in **falsifiable
terms** so the next wave can test it directly. Don't say "occupancy is
the issue"; say "occupancy is the issue **and a kernel at the same
occupancy with proper methodology should hit < 80 %**". Then W4 can
falsify it cleanly.

### Lesson 3 — Counter-example verdicts (W4 meta-doubt) can confuse different test contexts

W4 had the right falsification (W3b's mechanism is wrong) but wrong
verdict (re-promote V49/V50 to MED). The error: treating V8 and V49 as
"two kernels at identical occupancy" when in fact they differ on
**multiple** axes — V8 has 128-deep inner unroll and
`__launch_bounds__(256, 1)`, V49 has 8-deep with branch every 8 ops
and `__launch_bounds__(128, 2)`. Same occupancy ≠ same methodology.

The correct gate to apply at W4: a single matched variable does not
license a re-promote unless **all OTHER variables** also match. Best to
ask "is there ANY axis on which V8 and V49 differ?" and only re-promote
if the answer is "no, they're identical".

### Lesson 4 — SASS verdicts (W5a) are structurally soundest but still hypothesis until ncu

W5a inspected the actual emitted SASS and found a concrete,
mechanism-grounded reason: V49's inner body is contaminated by branch
dispatch in a way V8's is not. This is the strongest of the four
verdicts because it is grounded in the **actual code** rather than in a
ratio or an inferred mechanism.

**But it is still hypothesis** until V52 reruns with V8-style methodology
AND ncu `sm__inst_executed_pipe_fma` + `sm__inst_executed_pipe_alu`
simultaneously. SASS inspection narrows the hypothesis space; only ncu
confirms which hypothesis is right.

The correct gate to apply at W5a: SASS narrows the space. Predict the
ncu metric that would confirm or refute. Don't publish a verdict
without running the ncu test.

---

## A.8 The meta-lesson

> **Five waves of nested doubt converge slowly without an empirical
> anchor. One careful empirical test with the right diagnostic settles
> the question in a single shot.**

Doubt-the-doubt is valuable — each wave caught a real bug in the
previous wave's reasoning, and stopping at any wave before W6 would have
left the canonical reference incorrect (W1+W2 wrong about cap existing;
W3b wrong about mechanism; W4 wrong about MED; W5a wrong about implied
cap). But doubt-without-empirical-test is structurally limited:

- Each wave can only catch errors that are *visible* from the prior
  wave's evidence.
- A wave cannot rule out errors that require *new* evidence (e.g., ncu
  metrics nobody had collected).
- Architectural inferences hidden inside an artifact-detection argument
  tend to be inherited silently across waves.

The remedy is not "more doubt waves" — it is **"one wave with the right
empirical anchor"**. V52 ran in roughly the time of one armchair doubt
wave and yielded a decisive answer.

---

## A.9 What V49/V50 should have done differently

Based on the V52 outcome, the corrected V49/V50 methodology would be:

1. **Unroll the inner body to ≥ 64 ops/type per outer iteration.** V49's
   8-deep was contaminated; V52's 128-deep is clean.

2. **Match `__launch_bounds__` between solo and dual baselines.** V49
   used (128, 2) and compared against an abstract peak; V52 uses (256,
   1) and compares against its own solo runs at the same occupancy.

3. **Add anti-DCE STG of accumulator XOR.** V49 used a clock-diff
   conditional that's ambiguous; V52 unconditionally writes the XOR
   accumulator if it's not 0xdeadbeef.

4. **Run ncu pipe_fma + pipe_alu simultaneously.** V49 ran neither;
   V52 ran both. The diagnostic `pipe_alu + pipe_fma > 100 %` is the
   ONLY way to prove dual-issue.

5. **Run multiple recipe variants.** V49 had one kernel; V52 has 18
   templates (3 modes × 3 ILPs × 2 BPS). Cross-checking templates
   confirms the ratio is structural, not specific to one config.

6. **Reproduce 3× with `pkill -9 + sleep 6` between runs.** V49
   reproduced within 1 %, but on a possibly-contaminated GPU
   (leftover processes can inflate cy/MMA up to 8.5×). V52 ran
   `pkill -9 v52 && sleep 6` between every run.

7. **State the architectural prediction before running.** V49 cited
   "55 % same-warp ceiling" without first stating "if dispatch
   doesn't share, we expect alu+fma > 100 %; if it shares, we expect
   alu+fma ≤ 100 %". Predicting the discriminating outcome forces
   you to choose the right diagnostic.

---

## A.10 ncu metric definitions used in V52

For reference, these are the ncu metrics V52 collected, with what each
one means:

| Metric | What it counts |
|---|---|
| `smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | % of active SMSP cycles in which the FMA pipe issued an instruction. 100 % means every SMSP cycle issued an FMA-pipe inst. |
| `smsp__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active` | Same for ALU pipe (LOP3, IADD3, etc.) |
| `smsp__inst_issued.avg.per_cycle_active` | Average instructions issued per active SMSP cycle. Max 1.0 per pipe; if you can dual-issue across pipes, can exceed 1.0. |
| `smsp__warps_active.avg.pct_of_peak_sustained_active` | Occupancy: % of slots filled by active warps. |
| `smsp__cycles_active.avg` | Total active SMSP cycles in the kernel run. |

The decisive interpretation:

- `pipe_fma + pipe_alu > 100 %` ↔ pipes overlap freely (dual-issue).
- `pipe_fma + pipe_alu ≈ 100 %` ↔ pipes share a dispatch port (no
  dual-issue).
- `inst_issued/cy > 1.0` ↔ dispatch port issues more than one inst/cy
  (only possible with pipe diversity).
- `inst_issued/cy < 1.0` ↔ either the pipe is stalled or the
  instruction takes multiple issue cycles (LOP3 case).

V52 results: `pipe_alu + pipe_fma = 147 %` AND `inst_issued/cy = 1.0`
in dual mode. Both signatures of free overlap.

---

## A.11 What could overturn V52 (preserved doubt)

V52 is the strongest evidence in the catalog, but is not infallible:

1. **ncu metric definitions are software-defined.** If
   `smsp__pipe_alu_cycles_active` counts cycles where the pipe is
   *holding* an instruction (not just *issuing* one), then `alu + fma >
   100 %` is consistent with serial issue at the dispatch port too. We
   have NOT verified the ncu metric definition against PTX-level event
   counters or a public NVIDIA spec.

2. **"Free overlap" was measured for FMA + ALU specifically.** Other
   pipe combinations (LSU + tensor, MUFU + FMA in non-LOP3 setting,
   etc.) are NOT settled by V52.

3. **The 2-cycle LOP3 cadence is inferred** from `inst_issued/cy =
   0.51`. An alternative explanation is "1-cycle issue but 50 % stall
   on RF read port". V52 cannot distinguish them; both predict the same
   `inst_issued` and `pipe_alu`.

4. **If V52's GLane/s reading drifts > 1 %** across runs in future
   re-tests, the methodology may have a yet-undetected issue.

What would overturn V52: an ncu metric-definition bug for sm_103a, an
alternative interpretation of `pipe_X_cycles_active`, or a clean test
where `alu + fma` reproducibly stays at ≤ 100 % under V8-style
methodology with an alternate recipe. None are expected.

---

## A.12 Cross-references

- **Empirical anchor:** `b300_clean/corrections/V52_RUN_RESULTS.md`
- **SASS analysis:** `b300_clean/corrections/SASS_VERIFY_DUAL_ISSUE.md`
- **Meta-doubt audit:** `b300_clean/corrections/META_DOUBT_REPORT.md`
- **Wave-3c synthesis:** `b300_clean/corrections/DOUBT_LOG_v2.md`
- **Wave-6 final:** `b300_clean/corrections/HEADLINE_CORRECTIONS_v5.md`
  row 7
- **Distilled wisdom:** `b300_clean/corrections/META_LESSONS.md`
- **Original V49:** `tests/standalone/v49_dual_pipe.cu`, commit 501134a
- **Original V50:** `tests/standalone/v50_warp_specialized.cu`, commit
  fbe1c18
- **Original V8:** `tests/bench_ffma_warps_per_sm.cu`
- **V52:** `tests/standalone/v52_dual_issue_clean.cu`


---

# APPENDIX B. Methodology rules learned from waves 1-6

> Extended elaboration of §62, with worked examples for each rule. Each
> rule below is followed by: the catalog incident that motivated it, the
> mechanism, the worked example, and the corrected practice.


---

## B.1 Rule 1 — State theoretical maximum first

### B.1.1 Why the rule exists

Without an explicit theoretical first, you cannot tell whether a
measurement is reasonable. The "is X plausible?" check is the cheapest
methodology gate — it costs nothing and catches the worst class of bug
(DCE / formula error / unit mismatch).

### B.1.2 The catalog incident — V8 DSMEM 37 TB/s

V8 commit 71934d0 reported "Cluster DSMEM BW = 37 TB/s = 97 % of 38.5
peak". This passed early review because 97 % looked plausible against
a 38.5 TB/s peak.

The peak the row cited was **SHMEM peak**, not DSMEM peak. DSMEM
(distributed SHMEM, cluster-shared) operates over an inter-CTA bus and
is fundamentally slower than per-CTA SHMEM. There is no stated DSMEM
peak in any architecture document; the 38.5 TB/s denominator was wrong
by definition.

When V8 was re-checked with theoretical-first analysis:

- SHMEM peak per SM = 32 banks × 4 B × 2.032 GHz = 260 GB/s/SM.
- 148 SMs × 260 GB/s = 38.49 TB/s chip-wide SHMEM peak.
- DSMEM is cluster-shared, so the per-cluster peak is at most SHMEM-per-SM
  × cluster-size × inter-CTA-bus-efficiency. With cluster=8 and ~10 %
  efficiency expected, peak is ~2 TB/s/cluster × 18 clusters = 36 TB/s
  upper bound.
- 37 TB/s is barely under this upper bound, which is a red flag if the
  measurement isn't testing the right thing.

SASS investigation showed V8's compile-time invariant offsets were
LICM'd / CSE'd. The kernel was actually doing 7200 SHMEM transactions
total (visible in ncu wavefront count), not the 5.9 billion the
formula assumed. Real BW per cluster: ~40 GB/s — off by ~1000×.

### B.1.3 Worked example — applying Rule 1 to a hypothetical NVLink claim

You see a claim "NVLink-5 measured 1.2 TB/s on B300". Apply Rule 1:

- B300 NVLink-5 spec: 18 lanes × 50 GB/s = 900 GB/s/dir.
- 1.2 TB/s would be 133 % of spec. **STOP** — almost certainly wrong.

What's likely happening: the measurement is summing both directions
(read + write simultaneously), so 1.2 TB/s is bidirectional ~600
GB/s/dir each = 67 % of spec, which is plausible. Or the measurement
is in different units (MiB/s vs GB/s). Either way, Rule 1 catches the
discrepancy before publication.

### B.1.4 Corrected practice

For every NEW measurement:

1. Look up theoretical peak for the operation.
2. State the theoretical first in your analysis.
3. Compute measured / theoretical ratio.
4. If ratio > 1.0: STOP, find the bug.
5. If ratio < 0.5: investigate methodology (under-saturation).
6. If 0.5 ≤ ratio ≤ 1.0: proceed to SASS / ncu verification (Rules 6-8).

---

## B.2 Rule 2 — State measured number with denominator

### B.2.1 Why the rule exists

A measured number without a denominator is ambiguous. "98.5 %" is
meaningless; "98.5 % of 7.31 TB/s empirical pure-direction peak" is
meaningful. The denominator carries the architectural framing.

### B.2.2 The catalog incident — V46 "98.5 % NEW SoL"

V46 measured 7.20 TB/s for an 8-deep TMA pipelined read. The headline
read "98.5 % NEW HBM read SoL". The denominator implicit in this
framing was 7.31 TB/s (V32 empirical pure-direction).

Re-anchored against three different denominators:

- 7.20 / 7.31 (V32 empirical) = **98.5 %** — V46's framing.
- 7.20 / 7.67 (this-device post-ECC) = **93.9 %** — correct for SoL.
- 7.20 / 7.68 (spec post-ECC) = **93.8 %** — correct for cross-vendor.
- 7.20 / 8.0 (marketing) = **90.0 %** — the original "marketing-rounded"
  framing.

The architectural lesson "TMA reads need 8-deep pipelining" remains
valid (V46 7.20 > V33 6.72 = +7 % over single-deep). But the "98.5 %
NEW SoL" framing was the artifact, not the measurement.

### B.2.3 Worked example — denominator drift across the catalog

Three different files cited three different HBM denominators:

- `01_hbm_bandwidth_CORRECTED.md`: 7672 GB/s post-ECC (derived 7680b ×
  3996 MHz × 2 / 8).
- `09_memory_apis_CORRECTED.md`: 7.2 TB/s ("% of HBM 7.2 TB/s").
- CLAUDE.md memory snippet: "~8 TB/s spec".

The result: cross-doc % numbers are NOT comparable. A "95 % of HBM
peak" claim in one doc is not the same as "95 %" in another. This
made the headline corrections process a mess (4 waves of denominator
debate before finalizing the dual-cite rule in W6).

### B.2.4 Corrected practice

For HBM:
- **7.68 TB/s** = spec post-ECC (use for cross-vendor)
- **7.67 TB/s** = this-device post-ECC (use for SoL on this part)
- **7.31 TB/s** = empirical pure-direction (use ONLY when explicitly
  framed as "% of best-known recipe")

For tensor: cite the specific PTX form. "% of 15 PF" is meaningless
without the kind:: clause.

For FFMA: state which clock state (boost 76.96, locked 72.65).

For latency: state cy at which clock state (cy at boost = ns × 2.032).

---

## B.3 Rule 3 — If measured > theoretical: STOP

### B.3.1 Why the rule exists

A measurement above theoretical is mathematically impossible. The HW
cannot exceed its physical peak. If your measurement says it can, your
test is broken.

### B.3.2 The catalog incident — V8/V10 DSMEM TB/s peaks

V8 reported "DSMEM 37 TB/s = 97 % of 38.5 peak". V10 reported "1.84
mma/SM/cy aggregate at 16 warps". Both passed early review.

DSMEM physical peak (cluster-shared bus): ~2 TB/s/cluster × 18 clusters
= 36 TB/s upper bound. V8's 37 TB/s exceeds the cluster aggregate
upper bound. STOP.

mma at 1.84/SM/cy: B300 has 1 tensor pipe / SM, so 1 mma/cy is the
per-SM upper bound. 1.84 > 1.0 = STOP.

Both were investigated and retracted:

- V8 DSMEM: SASS showed compile-time-invariant offsets LICM'd, real
  rate ~40 GB/s/cluster × 18 = 720 GB/s aggregate. Off by 50×.
- V10 1.84 mma/SM/cy: aggregating across overlapping kernels in
  different streams; per-stream rate was ~0.9 mma/SM/cy (saturated).
  Off by 2×.

### B.3.3 Worked example — applying Rule 3 to a tcgen05 result

A test reports "tcgen05 NVFP4 5000 TFLOPS". Apply Rule 3:

- B300 NVFP4 spec: 15 PF (15 000 TFLOPS).
- 5000 TFLOPS = 33 % of spec. **OK, plausible.**

Now consider "tcgen05 NVFP4 16 PF". Apply Rule 3:

- 16 PF > 15 PF spec. **STOP.**

What might be happening: the test is including sparsity (which doubles
spec to 30 PF), or measuring instruction-issue rate (not actual MMA
ops). Investigate before publishing.

### B.3.4 Corrected practice

If measured > theoretical:

1. STOP immediately.
2. Re-derive theoretical with explicit unit checks.
3. SASS-verify the inner body (DCE check).
4. Re-derive measured with explicit unit checks.
5. ncu cross-check the relevant pipe / memory metric.
6. If still > theoretical, the test is broken.
7. If < theoretical after cleanup, publish the cleaned number.

---

## B.4 Rule 4 — If measured > 1.5× theoretical: almost certainly DCE

### B.4.1 Why the rule exists

The DCE failure mode is so common that any measurement substantially
above theoretical should be assumed eliminated until proven otherwise.
1.5× is the threshold above which clock-skew, units-mismatch, or
formula-shift are not enough to explain the gap — only DCE (or full
unit / scale error) can.

### B.4.2 The catalog incident — bench_fma at 200 TFLOPS

A bench_fma test at 200 TFLOPS on a 76 TFLOPS HW peak. The compiler
had unrolled the loop, observed the result wasn't written, and
eliminated the entire body. Wall-clock measured launch overhead ×
repetitions, divided by zero work, gave a meaningless "throughput"
number.

SASS investigation: the inner FFMA loop was completely gone. Only the
prologue (load constants) and epilogue (return) remained.

### B.4.3 Worked example — applying Rule 4 to a SHMEM benchmark

A SHMEM read kernel reports 80 TB/s. SHMEM peak = 38.5 TB/s. Ratio =
2.1×. **STOP.**

Possible causes:
1. DCE: kernel was eliminated.
2. L1 hit (SHMEM kernel actually measuring L1).
3. ILP across 16 chains divided wrong (per-chain rate × 16 instead of
   summed rate).

In this case, it was cause #2: the test had a 4 KB working set that
fit in L1 with high reuse; ncu showed `l1tex__t_bytes_pipe_lsu` was
saturating at 80 TB/s. The kernel was correctly measuring L1, just
mislabeled as SHMEM.

### B.4.4 Corrected practice

If measured > 1.5× theoretical:

1. **Default assumption: DCE.** Open the SASS and search for the inner
   loop. If it's not there or radically smaller than expected, DCE.
2. **Second guess: scale error.** Are the units right? Is "Glane/s"
   accidentally aggregating per-chain instead of total?
3. **Third guess: wrong target.** Is the kernel actually measuring
   what its label says? Check ncu metrics — `dram_bytes_read` for
   HBM, `l1tex__t_bytes` for L1, etc.

---

## B.5 Rule 5 — If measured < 0.5× theoretical: under-saturated

### B.5.1 Why the rule exists

A measurement substantially below theoretical usually indicates a
methodology issue: not enough ILP to hide latency, not enough
occupancy to hide pipeline bubbles, dependent chains in the hot loop,
or register port pressure.

### B.5.2 The catalog incident — V49 solo FFMA at 67 %

V49's solo FFMA was 25.2 Glane/s = 67 % of theoretical 37.6 Glane/s
peak. Below 80 %, this is in the "investigate methodology" zone.

Hypotheses:
1. Under-occupancy (W3b's hypothesis — 2 warps/SMSP × 8 ILP not enough
   to hide 4-cy FFMA latency).
2. Loop overhead (W5a's hypothesis — branch + UIADD3 + UISETP take
   ALU pipe slots).
3. RF port pressure (3-source FFMA caps at 65 %).

V8 hits 97.6 % at the SAME 2 warps/SMSP geometry, so hypothesis 1 is
falsified. SASS shows V49 has 2-source FFMA (not 3-source), so
hypothesis 3 is falsified. V52 confirms hypothesis 2: V49's small
inner body has 12.5 % loop overhead, and matching V8's 128-deep
methodology lifts solo FFMA to 84-87 %.

### B.5.3 Worked example — applying Rule 5 to a tensor benchmark

A tensor kernel reports 800 TFLOPS BF16. BF16 spec = 1980 TFLOPS via
tcgen05. Ratio = 40 %. **Investigate.**

Hypotheses:
1. Under-occupancy (not enough warps to feed the tensor pipe).
2. Operand staging stalls (SMEM bank conflicts on A or B load).
3. Wrong PTX form (using mma.sync instead of tcgen05).
4. Per-call host overhead (bare cuBLAS without cudaGraph).

ncu investigation: `pipe_tensor_cycles_active = 50 %`, occupancy =
85 %, SMEM transactions normal. So #1 and #2 are not the issue.

PTX investigation: the test uses `mma.sync m16n8k16` not `tcgen05.mma`.
Spec for legacy mma.sync = 540-580 TFLOPS, not 1980. The test is
hitting 800 / 580 = 137 % of mma.sync peak — actually impossible per
Rule 3. Re-investigate: ncu `dram__bytes_read` shows 1.2 TB/s — the
"800 TFLOPS" was actually compute-bound at 580 + 220 from a coincidental
re-mapping, not a true measurement.

The real issue is wrong-form PTX. Switch to tcgen05.mma to measure
true peak.

### B.5.4 Corrected practice

If measured < 0.5× theoretical:

1. Check occupancy — is `warps_active.pct_of_peak_sustained > 50 %`?
2. Check ILP in the SASS — are there ≥ 4 independent chains?
3. Check pipe utilization — is `pipe_X.cycles_active > 90 %`?
4. Check chain depth — is the operation chained (latency-bound) or
   parallel (throughput-bound)?
5. Check inner body for register port pressure (3-source FFMA at 65 %).
6. Check for the right PTX form (mma.sync vs tcgen05.mma).

---

## B.6 Rule 6 — If measured in [0.5×, 1.0×]: plausible but verify

### B.6.1 Why the rule exists

This is the regime where most real benchmarks live. The measurement is
plausible, but you can't claim HIGH confidence without verification.

### B.6.2 The catalog incident — V52 solo FFMA at 84-87 %

V52's solo FFMA hit 84-87 % of 76.97 TFLOPS. V8 at 97.7 %. The gap is
~10 percentage points.

Investigation:
- SASS confirms 128-deep inner body, no loop-overhead contamination.
- ncu confirms `pipe_fma = 97.58 %` — the pipe IS saturated.
- Wall-clock vs ncu agree, so it's not a ncu metric issue.

The remaining gap is **launch overhead and loop tail**: V52 uses 1k-4k
outer iterations, so kernel runtime is ~5-20 ms. Launch overhead
~1.85 µs is a ~0.04 % effect, but the loop tail (the last few
iterations may not have full ILP coverage) and the start-up ramp can
account for ~10 percentage points.

V8 uses N_OUTER ≥ 1M with longer runtime, amortizing the ramp.

So V52's 84-87 % is "plausible at this N_OUTER", and ncu confirms the
pipe is saturated. The architectural answer (pipes overlap freely)
holds; the wall-clock % is just regime-dependent.

### B.6.3 Worked example — applying Rule 6 to an HBM benchmark

A new HBM read kernel reports 6.5 TB/s. HBM peak = 7.67 TB/s
this-device. Ratio = 85 %. **Verify.**

Steps:
1. SASS check: is the load pattern as expected? `LD.E.128` on a
   coalesced address range? **YES.**
2. ncu check: `dram__bytes_read.sum.per_second = 6.5 TB/s`? **YES.**
3. ncu L2 check: `lts__t_sector_hit_rate.pct < 5 %`? If > 5 %, the test
   is contaminated by L2 reuse.
4. ncu pipe check: `lsu_cycles_active.pct > 80 %`? The LSU is the
   load issue port.
5. ILP check: SASS shows 8 independent load chains? Or just 1?

If all checks pass, publish 85 % with HIGH confidence. If any fail,
investigate.

### B.6.4 Corrected practice

For measurements in [0.5×, 1.0×]:

1. SASS-verify the inner body (Rule 7).
2. ncu cross-check the relevant pipe / memory metric (Rule 8).
3. State the regime (warps/SMSP, ILP, working set, etc.).
4. If 3-method verification passes, publish HIGH.
5. If any method fails, downgrade to MED with the failure noted.

---

## B.7 Rule 7 — SASS-verify

### B.7.1 Why the rule exists

Source-level `#pragma unroll N` does NOT guarantee SASS-level unroll.
The compiler may re-roll if it estimates better cache behavior. Inline
asm `fma %0, %0, %1, %0` may compile to a different SASS encoding than
expected (e.g., `FFMA Rd, Rd, R0.reuse, 0.5` if the compiler hoists
the immediate into R0). Without SASS verification, you don't know what
the kernel actually does.

### B.7.2 The catalog incident — V49 vs V8 inner body

W4 (meta-doubt) compared V49 and V8 at the source level and concluded
they were "identical" (both `fma %0, %0, IMM, IMM` vs `fma %0, %0,
%1, %0`, but both 2-source patterns with 1 RF read).

W5a (SASS-verify) inspected the actual emitted SASS:

V49:
```
FFMA Rd, Rd, R0.reuse, 0.5     // R0 holds 1.5f, immediate 0.5f
```
- R0 is hot in the operand reuse cache (`.reuse` cache hit).
- 1 unique RF read (Rd).

V8:
```
FFMA Rd, Rsrc1, Rd, Rd          // Rsrc1 distinct from Rd
```
- Rsrc1 is loop-constant, also `.reuse`-able.
- 2 unique RF reads, but both `.reuse`-able.

**Both kernels avoid the 3-distinct-source RF port pressure.** Source
inspection wasn't enough; SASS revealed the RF port behavior.

The REAL methodology gap (which only SASS revealed) was the LOOP
OVERHEAD: V49's 8-deep inner body had branch + counter consuming ALU
slots; V8's 128-deep amortized this 16×.

### B.7.3 Worked example — SASS-verifying a tensor benchmark

A bench_tensor kernel reports 1.5 PF BF16. Apply Rule 7:

```bash
nvcc -arch=sm_103a -O3 -keep bench_tensor.cu -o bench_tensor
cuobjdump --dump-sass bench_tensor | grep -A 100 "_kernel"
```

Look for:
- `HMMA.16816.F32.BF16 Rd, ...` — legacy mma.sync path
- `UTCMMA.M128N128K16.BF16 ...` — Blackwell tcgen05 path
- Both? Mixed measurement, not pure tensor.

If you see only HMMA, the measurement is mma.sync (max 580 TFLOPS,
so 1.5 PF is impossible — Rule 3).
If you see UTCMMA, the measurement is tcgen05 (max 1980 TFLOPS, so
1.5 PF = 76 % which is plausible).

### B.7.4 Corrected practice

For every measurement that quotes a % of peak:

1. `nvcc -arch=sm_103a -O3 -keep <test>.cu`
2. `cuobjdump --dump-sass <bin> > <bin>.sass`
3. Find the inner loop in the SASS.
4. Count emitted instructions:
   - The pipe being measured (FFMA, LOP3, HMMA, UTCMMA, ...).
   - Loop overhead (UIADD3, UISETP, BRA).
   - Memory ops (LDG, STG, cp.async, TMA).
   - Anti-DCE writes (STG).
5. Verify the count matches what you wrote in source.
6. If different, investigate (re-roll, CSE, hoisting, etc.).

---

## B.8 Rule 8 — Cross-check ncu

### B.8.1 Why the rule exists

Wall-clock measures end-to-end time but doesn't reveal what fraction of
that time was spent on the pipe you think you're measuring. ncu
metrics decompose the kernel time into pipe-level activity:

- `pipe_fma_cycles_active` = cycles in which the FMA pipe issued an
  instruction.
- `pipe_alu_cycles_active` = same for ALU pipe.
- `pipe_lsu_cycles_active` = same for LSU pipe.
- `inst_issued.per_cycle_active` = average instructions issued per
  active cycle.

For dual-issue claims, the diagnostic `pipe_fma + pipe_alu > 100 %` is
the only decisive metric. Wall-clock GLane/s ratios are not enough.

### B.8.2 The catalog incident — V49/V50 dual-issue

V49/V50 collected ZERO ncu metrics. Wall-clock GLane/s ratios were
the only evidence. The "55 % / 74 %" headlines were published HIGH on
single-method evidence.

W6 V52 collected `pipe_alu + pipe_fma`. Result: 147 % at ILP=8. Decisive
proof of free overlap. The 55 % / 74 % were artifacts.

The 5-wave detour was caused by lack of ncu metrics. With ncu from
the start, the verdict would have been settled at W1+W2.

### B.8.3 Worked example — ncu metrics for an FP32 FFMA test

A bench_fma test reports 60 TFLOPS = 78 % of 76.97 peak. ncu pass:

```bash
ncu --metrics \
  smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__inst_issued.avg.per_cycle_active,\
smsp__warps_active.avg.pct_of_peak_sustained_active \
  ./bench_fma
```

Expected output:
- pipe_fma ≈ 78 % (matches wall-clock %).
- pipe_alu ≈ 1 % (no ALU activity, pure FFMA).
- inst_issued/cy ≈ 0.78.
- warps_active ≈ 100 %.

If pipe_fma is much lower than wall-clock %, you have a methodology
issue (kernel not actually FFMA-bound). If pipe_alu is high, you have
ALU contamination (loop overhead, IADD3, etc.). If inst_issued/cy is
much less than pipe_fma, the pipe is stalled (RF port pressure).

### B.8.4 Corrected practice

For every measurement:

1. Identify the relevant pipe(s) for your operation.
2. Run `ncu --metrics` with the corresponding `pipe_X_cycles_active`
   metric(s).
3. Cross-check that ncu agrees with wall-clock to within 5 %.
4. If they disagree, you have a methodology issue (DCE, wrong target,
   stall, etc.).
5. For dual-issue / pipe-overlap claims, ALWAYS collect both pipes
   simultaneously.

---

## B.9 Rule 9 — If too-good-to-be-true: it is

### B.9.1 Why the rule exists

The catalog has a long history of "too good" claims that were later
retracted. Specific symptoms to flag:

- BW > theoretical
- TFLOPS > theoretical
- Latency < hardware unit minimum
- "Same-warp dual-issue" > 100 % gain (e.g., V49's 55 % was actually
  a misframed loss, not a gain)
- "Multicast pipelined deeper than 1 stage helps" (V48 disproved)
- "cudaGraph single-node speedup" (V9 disproved)
- "DSMEM TB/s scaling" (V8 retracted, real ~40 GB/s/cluster)

Each of these has been claimed and later retracted.

### B.9.2 The catalog incident — V8 DSMEM 37 TB/s

97 % of SHMEM peak for distributed-SHMEM is too good to be true. DSMEM
adds inter-CTA bus latency and bandwidth dilution; reaching 97 % of
local SHMEM peak would be remarkable.

When investigated, V8 was actually measuring 7200 SHMEM transactions
total (LICM'd / CSE'd loop), not the 5.9 billion implied by the formula.
Real BW: 40 GB/s per cluster.

### B.9.3 Worked example — applying Rule 9 to a launch overhead claim

A claim says "cudaGraph 5× faster than direct launch". Apply Rule 9:

- Direct launch = 1.85 µs. 5× faster = 0.37 µs.
- B300 single-kernel launch ≈ 2 µs floor (event sync overhead).
- 0.37 µs < 1 µs is below the host-side enqueue floor.

V9 measured: cudaGraph single-node = 2.05 µs (no speedup); 100-kernel
graph = 0.59 µs/kernel (3.5× speedup). The "5×" claim was actually
"3.5× at 100-kernel batch", reduced to "5×" through inflation.

### B.9.4 Corrected practice

If a claim makes you think "wow, that's surprising":

1. State explicitly why it's surprising (which physical assumption
   it violates).
2. Apply Rules 1-3 (theoretical, denominator, > theoretical check).
3. Look for the methodology issue specifically suggested by the
   surprise.
4. Consult the retraction log (Appendix E) — has this exact claim been
   made before?

---

## B.10 Rule 10 — Multi-method agreement required for HIGH confidence

### B.10.1 Why the rule exists

Single-method evidence is fragile. The 5-wave dual-issue zigzag (App A)
is the canonical example: 5 waves of armchair doubt converged slowly,
while 1 careful empirical test settled it.

### B.10.2 The catalog incident — V49/V50 published HIGH on single method

V49 had wall-clock + reproducibility within 1 %. Two methods agree?
Yes, both wall-clock and reproducibility. But that's two views of the
same evidence (the kernel-internal clock64 ratio). Not three orthogonal
methods.

The actual three methods needed:
1. Wall-clock cudaEvent.
2. ncu pipe metric.
3. SASS verification.

V49 had only #1. V52 added #2 and #3. The result flipped.

### B.10.3 Worked example — what 3-method agreement looks like

A new bench_lop3 test claims "LOP3 hits 16.8 K Glane/s = 100 % of ALU
pipe". Apply Rule 10:

Method 1 (wall-clock cudaEvent): kernel takes T ms, computes 16.8 K
Glane/s. **PASS.**

Method 2 (ncu pipe): `pipe_alu_cycles_active = 99.5 %`. **PASS.**

Method 3 (SASS): inner body 128 LOP3 + 1 BRA + 1 UIADD3 + 1 UISETP.
Loop overhead = 2.3 %. **PASS.**

Multi-method agreement: HIGH confidence. Publish.

If any method fails:
- Method 1 fails (wall-clock disagrees with ncu): probably DCE or
  contamination. Investigate.
- Method 2 fails (ncu pipe is much lower): the kernel isn't ALU-bound;
  some other pipe is. Investigate.
- Method 3 fails (SASS shows LICM, missing inner body): the source
  unroll didn't take. Fix and re-test.

### B.10.4 Corrected practice

For every HIGH-confidence claim:

- [ ] Wall-clock cudaEvent measurement.
- [ ] ncu pipe metric for the operation.
- [ ] ncu memory metric (if memory-bound).
- [ ] SASS verification of the inner body.
- [ ] At least 2 kernel variants (different methodology) agree.
- [ ] Reproduced 3× within 1 %.

If 4+ pass, HIGH. If 2-3 pass, MED. If only 1, LOW.

---

## B.11 Rule 11 — ≥ 64 ops/type body for dual-issue (NEW from V52)

### B.11.1 Why the rule exists

V49's 8-ops/type inner body had 12.5 % loop overhead. V52's 128-ops/type
body has 1.2 % loop overhead. The 11.3 percentage-point difference
explains most of the V49 / V8 solo gap.

For dual-issue measurements specifically, the loop overhead consumes
ALU pipe slots — exactly the pipe being measured. So loop overhead in
a dual-issue test contaminates the measurement.

### B.11.2 Quantification

| Inner body size | Loop overhead % | Risk for dual-issue |
|---:|---:|---|
| 4 ops/type | 25 % | EXTREME — measurement is mostly loop |
| 8 ops/type | 12.5 % | HIGH — V49 case, 5-wave detour |
| 16 ops/type | 6.25 % | MED — borderline |
| 32 ops/type | 3.1 % | LOW — acceptable |
| 64 ops/type | 1.6 % | MINIMAL — recommended floor |
| 128 ops/type | 0.78 % | NEGLIGIBLE — V8/V52 standard |
| 256 ops/type | 0.39 % | NEGLIGIBLE — diminishing returns |

The recommended floor is **64 ops/type per inner iteration**. This caps
loop overhead at < 2 %, which is below the typical 5 % "noise floor"
for cross-method agreement.

### B.11.3 Worked example — V52 sizing

V52 chose 128 ops/type to match V8 exactly:

```cuda
template<int MODE, int ILP, int BPS>
__global__ __launch_bounds__(256, BPS)
void v52_kernel(...) {
    float f[ILP];
    unsigned u[ILP];
    // init...

    #pragma unroll 1
    for (int outer = 0; outer < N_OUTER; ++outer) {
        // Inner unroll: 128/ILP times of an ILP-wide block
        #pragma unroll
        for (int inner = 0; inner < (128 / ILP); ++inner) {
            #pragma unroll
            for (int k = 0; k < ILP; ++k) {
                if (MODE == 0 || MODE == 2)
                    asm("fma.rn.f32 %0, %0, %1, %0;" : "+f"(f[k]) : "f"(1.5f));
                if (MODE == 1 || MODE == 2)
                    asm("lop3.b32 %0, %0, 0xa5, %1, 0x96;" : "+r"(u[k]) : "r"(0x12345678));
            }
        }
    }
    // Anti-DCE STG of accumulator XOR
    if (acc != 0xdeadbeef) ...
}
```

Total inner ops per outer iter:
- MODE=0 (solo FFMA): 128 FFMA
- MODE=1 (solo LOP3): 128 LOP3
- MODE=2 (dual): 128 FFMA + 128 LOP3

All meet the 64-ops/type floor.

### B.11.4 Corrected practice

For dual-issue / pipe-overlap measurements:

1. Choose inner unroll depth ≥ 64 per type.
2. SASS-verify the inner body has the expected count.
3. Confirm loop overhead < 2 % of body.
4. If any of the above fail, increase unroll.

---

## B.12 Rule 12 — Standardize denominators (NEW from W4-W6)

### B.12.1 Why the rule exists

Three different denominators across three docs in the catalog made
cross-doc % numbers non-comparable. The W4/W5/W6 rigor sweep settled
on a dual-citation rule: cite both spec and this-device, mention the
gap when SoL precision matters.

### B.12.2 Denominator framework

For HBM:
- **7.68 TB/s** — spec post-ECC (8.000 Gbps × 8192 bits ÷ 1.0625
  ECC ÷ 8 B/byte). Use for cross-vendor or "what NVIDIA promised".
- **7.67 TB/s** — this-device post-ECC (7.992 Gbps × 7680 bits ÷
  1.0625). Use for SoL on THIS box (the AC SKU has 1/16 fused).
- **7.31 TB/s** — empirical pure-direction peak (V32). Use ONLY when
  framing as "% of best-known recipe", never as "spec" or
  "theoretical".
- **8.0 TB/s** — marketing rounded. NEVER use as a denominator.

For FFMA (boost 2032 MHz):
- **76.96 TFLOPS** = 148 SMs × 128 cores × 2 op/FMA × 2.032 GHz.
- Locked at 1920: 72.65 TFLOPS.
- State which clock state.

For BF16 tensor:
- mma.sync legacy: 540-580 TFLOPS.
- tcgen05.mma Blackwell: 1980 TFLOPS.
- Cite the specific PTX form.

For NVFP4:
- Spec: 15 PF dense / 30 PF sparsity-on.
- Cite the form (cuBLAS / cuBLASLt / direct tcgen05.mma).
- Note: cuBLAS K=96 K-id wide-rect ceiling = 11.42 PF (76 % of spec).

For SHMEM:
- 38.49 TB/s theoretical (32 banks × 4 B × 2.032 GHz × 148 SMs).
- 38.4 TB/s measured peak.

For atomics:
- L2 atomic packets: state stride and unroll explicitly.
- SMEM atomic: state contention level (uncontended vs 32-way).

### B.12.3 Worked example — applying Rule 12 to a new HBM benchmark

You measure 7.0 TB/s on a new HBM kernel. Apply Rule 12:

- 7.0 / 7.68 = 91.1 % of spec post-ECC.
- 7.0 / 7.67 = 91.3 % of this-device post-ECC.
- 7.0 / 7.31 = 95.8 % of empirical pure-direction.

Publish: "7.0 TB/s = 91.1 % of spec post-ECC (7.68 TB/s) / 91.3 % of
this-device peak (7.67 TB/s)".

Don't publish: "95.8 % of HBM peak" without naming the empirical
denominator.

### B.12.4 Corrected practice

For HBM: dual-cite spec and this-device. For all other metrics: state
the denominator explicitly with units.

---

## B.13 Rule 13 — Always git-verify "[x] done" hashes (NEW from CURIOSITY V2 audit)

### B.13.1 Why the rule exists

CURIOSITY_LIST_V2 had **22/25 hallucinated hashes** (88 %). The author
filled in plausible-looking hashes from memory without verifying. V4-V8
git-verify rate: 100 %.

This pattern is dangerous because hash citations look authoritative,
but if hallucinated, there's no audit trail. Future readers can't
verify the claim.

### B.13.2 Verification pattern

```bash
# Verify the hash exists
git rev-parse --short=7 "<hash>" 2>&1
# If output is the hash, it exists. If error, it doesn't.

# Verify the topic matches
git log --oneline -1 "<hash>"
# Output should match the [x] claim's topic.
```

### B.13.3 Worked example — V2 hallucinated hashes

V2 cited `c0c2d48` for "S2 tcgen05 alloc breakthrough". Verify:

```bash
$ git rev-parse --short=7 c0c2d48
c0c2d48                                # exists in tree
$ git log --oneline -1 c0c2d48
c0c2d48 some unrelated commit          # topic doesn't match
```

The hash exists in the tree but is NOT the commit V2 claimed. Topic
search for "S2 BREAKTHROUGH: tcgen05 alloc/dealloc WORKS" found real
commit `ec25f05`.

V2 had 22/25 such hallucinations. The pattern: the agent recalled
"there was a commit about X" but invented a plausible-looking 7-char
hash from memory.

### B.13.4 Corrected practice

For every cited hash:

1. `git rev-parse --short=7 <hash>` — confirm exists.
2. `git log --oneline -1 <hash>` — confirm topic matches.
3. If either fails, search by topic: `git log --grep="<keyword>"
   --oneline | head`.
4. Cite only verified hashes.

For task lists with multiple hashes, batch-verify:

```bash
for h in $(grep -oE '\b[0-9a-f]{7,8}\b' TASK_LIST.md | sort -u); do
    if ! git rev-parse "$h" > /dev/null 2>&1; then
        echo "HALLUCINATED: $h"
    fi
done
```

---

## B.14 Sources

- CLAUDE.md §3 (rules 1-10)
- `b300_clean/corrections/META_LESSONS.md` (rule 11 derivation)
- `b300_clean/corrections/HEADLINE_CORRECTIONS_v5.md` (rule 12)
- `b300_clean/corrections/CURIOSITY_LISTS_AUDIT.md` (rule 13)
- `b300_clean/corrections/V52_RUN_RESULTS.md` (worked example for
  rules 7, 8, 11)
- `b300_clean/corrections/SASS_VERIFY_DUAL_ISSUE.md` (worked example
  for rule 7)


---

# APPENDIX C. Open questions + proposed test sketches V53–V56

> Per `RETEST_PROPOSALS.md` but expanded with rationale, decision rules,
> and expected outcomes. These are the four highest-priority unresolved
> items on the corrections backlog after V52 settled the dual-issue
> question. They follow the V52 template: standalone .cu files, V8-style
> methodology, ncu cross-checks, dual decision rules.

---

## C.1 V53 — DSMEM fenced retest

**Settles:** Is V21's DSMEM `push_ring_wr` measurement a true read/write
SoL or did missing fences let stores go in-flight at clock64?

### C.1.1 Background

V21 measured DSMEM aggregate write at 560 GB/s/cluster. The catalog
flagged this as **issue rate, not completion** because V21's
`push_ring_wr` had NO `fence.sc.cluster` between the
`st.shared::cluster.u32` stores and the closing `clock64`. Stores
might still be in flight when the timer stopped.

V21 also measured DSMEM read at 40 GB/s/cluster, but the kernel used
a dependent-chain pattern (loaded value feeds next address). This
makes the measurement **chain-bound, not absolute** — a non-chained
ILP test could reach 60-80 GB/s.

Both numbers were demoted in W3b (DSMEM_DOUBT_REPORT.md). V53 settles
both with proper methodology.

### C.1.2 Hypothesis matrix

For writes:
- H_no_fence: V21's 560 GB/s is real completion BW (fences are no-op
  for st.shared::cluster.u32 in this regime).
- H_inflated: V21's 560 GB/s is issue rate; real completion is much
  lower (e.g., 200 GB/s).

For reads:
- H_chain_bound: V21's 40 GB/s is the chain-bound asymptote;
  non-chained ILP can do 60-80 GB/s.
- H_absolute: V21's 40 GB/s IS the architectural ceiling.

### C.1.3 Two-axis design

Axis A — write side: explicit `fence.sc.cluster` between every store
batch and the closing `clock64`. Compare fenced vs unfenced.

Axis B — read side: ILP loop where the next read's address is loop-
carried but NOT result-dependent. Compare against V21's chain-dep.

### C.1.4 Kernel sketch

```cpp
// V53: DSMEM read/write SoL with proper fences.
// V21 timed pushes without fence -> stores still in flight at clock64 stop.
// Add fence.sc.cluster after every store batch. Separate non-chained ILP read.

#include <cuda_runtime.h>
#include <cstdio>
#include <cooperative_groups.h>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

// Cluster of CLUSTER_SIZE CTAs share DSMEM. Each CTA writes ILP doublewords/iter into peer's smem.
template<int CLUSTER_SIZE, int ILP, int N_ITERS, int FENCED>
__global__ __cluster_dims__(CLUSTER_SIZE,1,1) __launch_bounds__(128, 1)
void v53_dsmem_write(unsigned* out) {
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();
    __shared__ __align__(16) unsigned smem[1024];

    int tid = threadIdx.x;
    int my_rank = cluster.block_rank();
    int peer = (my_rank + 1) % CLUSTER_SIZE;
    unsigned* peer_smem = cluster.map_shared_rank(smem, peer);

    cluster.sync();

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            unsigned val = it * 17 + tid * 31 + k;
            unsigned addr = (unsigned)__cvta_generic_to_shared(
                                &peer_smem[(tid + k * 32) & 1023]);
            asm volatile("st.shared::cluster.u32 [%0], %1;"
                :: "r"(addr), "r"(val) : "memory");
        }
        if (FENCED) {
            asm volatile("fence.sc.cluster;" ::: "memory");
        }
    }
    if (FENCED) asm volatile("fence.sc.cluster;" ::: "memory");

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    cluster.sync();
    if (tid == 0) out[blockIdx.x] = smem[0] + (unsigned)(t1 - t0);
}

// Read SoL with non-chained ILP — addresses are loop-carried but values are NOT.
template<int CLUSTER_SIZE, int ILP, int N_ITERS>
__global__ __cluster_dims__(CLUSTER_SIZE,1,1) __launch_bounds__(128, 1)
void v53_dsmem_read(unsigned* out) {
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();
    __shared__ __align__(16) unsigned smem[1024];

    int tid = threadIdx.x;
    int my_rank = cluster.block_rank();
    int peer = (my_rank + 1) % CLUSTER_SIZE;
    unsigned* peer_smem = cluster.map_shared_rank(smem, peer);
    smem[tid] = tid;
    cluster.sync();

    // Address chain: next address depends on loop variable, NOT loaded value.
    // Values (vals[]) accumulate via XOR but are not in the address path.
    unsigned vals[8] = {0};
    unsigned addrs[8];
    #pragma unroll
    for (int k = 0; k < ILP; k++)
        addrs[k] = (unsigned)__cvta_generic_to_shared(
                       &peer_smem[(tid + k * 32) & 1023]);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            unsigned v;
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                : "=r"(v) : "r"(addrs[k]));
            vals[k] ^= v;     // not in addr path
        }
        // Address rotation is loop-carried (cheap ALU) but NOT result-dependent
        #pragma unroll
        for (int k = 0; k < ILP; k++) addrs[k] += 4;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    cluster.sync();
    if (tid == 0) {
        unsigned acc = 0;
        for (int k = 0; k < ILP; k++) acc ^= vals[k];
        out[blockIdx.x] = acc + (unsigned)(t1 - t0);
    }
}

int main() {
    CK(cudaSetDevice(0));
    unsigned* d_out; CK(cudaMalloc(&d_out, 4096));
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    const int N_ITERS = 8192, ILP = 8, CLUSTER = 8, BLOCKS = 1184;

    printf("=== V53 DSMEM read/write SoL with fence.sc.cluster ===\n");

    auto bench = [&](const char* lbl, auto kern, double bytes_per_op) {
        kern<<<BLOCKS, 128>>>(d_out);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) {
            printf("%s FAIL\n", lbl); cudaGetLastError(); return;
        }
        float total = 0;
        for (int r = 0; r < 5; r++) {
            cudaEventRecord(e0);
            kern<<<BLOCKS, 128>>>(d_out);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); total += ms;
        }
        float ms = total/5;
        double ops = (double)BLOCKS * 128 * ILP * N_ITERS;
        double tbs = ops * bytes_per_op / (ms/1e3) / 1e12;
        printf("%-30s ms=%.3f ops=%.2eG TB/s=%.3f\n",
               lbl, ms, ops/1e9, tbs);
    };

    bench("WR no-fence (V21 mode)",
          v53_dsmem_write<CLUSTER,ILP,N_ITERS,0>, 4.0);
    bench("WR fence.sc.cluster",
          v53_dsmem_write<CLUSTER,ILP,N_ITERS,1>, 4.0);
    bench("RD non-chained ILP",
          v53_dsmem_read<CLUSTER,ILP,N_ITERS>,    4.0);

    return 0;
}
```

### C.1.5 ncu metrics

```
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum
sm__inst_executed_pipe_lsu.sum
smsp__inst_executed_op_st_shared.sum
smsp__inst_executed_op_ld_shared.sum
sm__cycles_elapsed.avg
```

### C.1.6 Predicted outcomes

| Test | If V21 was correct | If V21 was missing-fence artifact |
|---|---|---|
| WR no-fence | Same as V21 | Same as V21 (high) |
| WR fenced | Same as V21 | **Significantly slower** (true latency surfaces) |
| RD non-chained | Same as RD-chained V21 | **Higher** than V21 (no dep chain) |

### C.1.7 Decision rules

- If `WR_fenced / WR_unfenced` ratio > 1.3 → V21's write SoL was
  inflated; reduce to fenced number; demote V21 write to LOW.
- If ratio < 1.05 → V21 measurement holds; promote to MED.
- If `RD non-chained > RD V21` by > 1.2× → DSMEM read SoL needs
  upgrading; quote the non-chained number.
- If `RD non-chained ≈ RD V21` → V21's chain-bound IS the architectural
  ceiling.

### C.1.8 Expected effort

~1 hour to write, compile, run, ncu, write up. The kernel is
straightforward; the cluster setup is the main complexity.

---

## C.2 V54 — membar isolation

**Settles:** `__threadfence_system` 1750 / 2870 / 3042 cy spread (1.74×).
Establish authoritative number with N-issue scaling.

### C.2.1 Background

The catalog has three different numbers for `__threadfence_system`
cost:
- 1750 cy (08_sync_primitives_CORRECTED.md, picked for TRUE_REFERENCE)
- 2870 cy (DSMEM_REFERENCE.md, no spread acknowledged)
- 3042 cy (V9_GRAPH_LAUNCH.md context)

The 1.74× spread means the canonical value is uncertain. TRUE_REFERENCE
picked 1750 / 861 ns WITHOUT justification. V54 settles by measuring
single-issue baseline + N-issue slope.

### C.2.2 Hypothesis matrix

- H_amortized: 1750 cy is N=8-amortized (true cost ~3000 cy single-shot,
  but pipelining brings amortized to ~250 cy/issue × 7 = 1750).
- H_setup_dominated: There's fixed setup ~1500 cy + per-issue ~250 cy.
  N=1 sees full setup (3042 cy); N=8 sees amortized (~1500 + 8×250 =
  3500 cy total = 437 cy/issue).
- H_one_is_wrong: One of 1750 / 2870 / 3042 is just wrong.

### C.2.3 Kernel sketch

```cpp
// V54: membar.{cta,gpu,sys} latency, single-thread, single-issue baseline + N-issue scaling.
// Uses fence.acq_rel as inert barrier marker so clock64 deltas frame exactly the membar.

#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int SCOPE, int N_ISSUE>  // SCOPE: 0=cta 1=gpu 2=sys
__global__ __launch_bounds__(32, 1)
void v54_membar(unsigned long long* out, volatile unsigned* probe) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Touch global so the prior store has something coherent to flush.
    probe[0] = 0xdeadbeef;

    // Inert acq_rel marker (no fabric round trip on its own, but blocks reordering).
    asm volatile("fence.acq_rel.gpu;" ::: "memory");

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll
    for (int i = 0; i < N_ISSUE; i++) {
        if (SCOPE == 0)      asm volatile("membar.cta;" ::: "memory");
        else if (SCOPE == 1) asm volatile("membar.gl;"  ::: "memory");
        else                 asm volatile("membar.sys;" ::: "memory");
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    out[SCOPE * 16 + N_ISSUE] = t1 - t0;
}

int main() {
    CK(cudaSetDevice(0));
    unsigned long long* d_out; CK(cudaMalloc(&d_out, 4096));
    unsigned* d_probe; CK(cudaMalloc(&d_probe, 4));
    cudaMemset(d_out, 0, 4096);

    printf("=== V54 membar isolation (1-warp, 1-thread) ===\n");
    printf("scope     N=1     N=2     N=4     N=8    cy/issue (avg)\n");

    const char* names[] = {"membar.cta", "membar.gl ", "membar.sys"};

    auto launch = [&](int sc, int ni) {
        // template dispatch...
        if (sc == 0) {
            if (ni == 1) v54_membar<0,1><<<1,32>>>(d_out, d_probe);
            // ...
        }
        // (full template dispatch elided)
    };

    for (int sc = 0; sc < 3; sc++) {
        unsigned long long cy[5] = {0};
        for (int idx = 0; idx < 4; idx++) {
            int ni = 1 << idx;
            // Median of 21 runs
            unsigned long long samples[21];
            for (int s = 0; s < 21; s++) {
                launch(sc, ni); cudaDeviceSynchronize();
                cudaMemcpy(&samples[s], d_out + sc*16 + ni,
                           8, cudaMemcpyDeviceToHost);
            }
            // Bubble sort
            for (int a=0;a<21;a++)
                for (int b=a+1;b<21;b++)
                    if (samples[b]<samples[a]) {
                        auto t=samples[a]; samples[a]=samples[b]; samples[b]=t;
                    }
            cy[idx] = samples[10];  // median
        }
        // Per-issue slope (robust to fixed clock64 overhead)
        double per_issue = (double)(cy[3] - cy[0]) / (8 - 1);
        printf("%s  %4llu    %4llu    %4llu    %4llu    %.1f\n",
               names[sc], cy[0], cy[1], cy[2], cy[3], per_issue);
    }
    return 0;
}
```

### C.2.4 ncu metrics

```
sm__cycles_elapsed.avg                          # cross-check clock64 base
smsp__inst_executed_op_membar.sum               # confirm count
sm__warps_active.avg.per_cycle_active           # should be ~1/SM (single warp)
```

Plus offline: `cuobjdump --dump-sass v54_membar` and grep for `MEMBAR`.

### C.2.5 Predicted outcomes

| N=1 cy | per-issue cy (slope) | Interpretation |
|---:|---:|---|
| ~1750 | ~250 | Matches `fence.sc.sys` 2870/8 ≈ 320 (DSMEM-style amortized issue). 1750 was 6-deep amortized batch; 3042 was over-counting setup. |
| ~3042 | ~250 | V9 number wins; 08's 1750 was undercount. Setup ~1500 cy + ~250/issue. |
| ~1500 | ~250 | Fixed setup ~1500 cy + ~250/issue; both prior numbers were partial truths. |
| ~3000 | ~3000 | No amortization possible; single-shot only. Catalog should retire amortized framings. |

### C.2.6 Decision rules

Pick the median single-issue at locked 1920 MHz as the canonical value.
Append the full N=1..8 table for context.

If `membar.cta` differs by > 2× from F6's 6 cy → revisit F6 too.

### C.2.7 Expected effort

~1.5 hours including data analysis. The median-of-21 strategy is to
defeat ~5 % run-to-run noise on a single-thread test.

---

## C.3 V55 — HBM floor empirical anchor

**Settles:** Anchor "% of HBM peak" denominator with the BEST-known
recipe.

### C.3.1 Background

The catalog has 4 different HBM denominators floating around (7672 /
7.31 / 7.2 / 8.0 TB/s). Wave 6 settled on dual-citation (7.68 spec /
7.67 this-device). But the **empirical** ceiling is not anchored. V55
sweeps the V32/V46/V48-style recipes to find the highest sustained
read BW, which becomes the empirical anchor.

### C.3.2 Recipe (from V32, V46, V48 lessons)

- TMA bulk loads (`cp.async.bulk.shared::cluster.global`).
- 16 KB tile, 8-deep in-flight per CTA.
- 148 CTAs (1×SM), per-warp issue (4 issuer warps × 2 inflight = 8
  inflight/CTA).
- Working set = 4 GB so L2 (126 MB) hit rate ≈ 0.
- N_ITERS chosen to give ≥ 10 ms wall (anti-launch-overhead).
- Sweep tile size {4, 8, 16, 32, 64} KB to find sweet spot.

### C.3.3 Kernel sketch

```cpp
// V55: HBM3E empirical floor — best-known recipe.
// Goal: maximum sustained HBM read BW. Used to anchor "% of peak" denominator.

#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int TILE_BYTES, int N_INFLIGHT, int N_ITERS, int N_ISSUE_WARPS>
__global__ __launch_bounds__(128, 1)
void v55_hbm_best(const float* src, unsigned long long* out,
                  unsigned total_ctas, size_t cap_words) {
    extern __shared__ __align__(16) char buf_raw[];
    __shared__ __align__(8) unsigned long long mbar[16];

    int tid = threadIdx.x;
    int wid = tid / 32;
    int bid = blockIdx.x;

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < N_INFLIGHT; i++)
            asm volatile("mbarrier.init.shared.b64 [%0], 1;"
                :: "r"((unsigned)__cvta_generic_to_shared(&mbar[i]))
                : "memory");
    }
    __syncthreads();

    unsigned bufs[16], mbars[16];
    #pragma unroll
    for (int i = 0; i < N_INFLIGHT; i++) {
        bufs[i]  = (unsigned)__cvta_generic_to_shared(&buf_raw[i * TILE_BYTES]);
        mbars[i] = (unsigned)__cvta_generic_to_shared(&mbar[i]);
    }

    // Strided over 4 GB with bid-bias to defeat L2.
    size_t stride = total_ctas * (TILE_BYTES / 4);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // PER-WARP issue: 4 warps, each owns N_INFLIGHT/N_ISSUE_WARPS slots.
    int slots_per_warp = N_INFLIGHT / N_ISSUE_WARPS;
    int slot_base = wid * slots_per_warp;

    #pragma unroll 1
    for (int it = 0; it < N_ITERS; it++) {
        if (tid % 32 == 0 && wid < N_ISSUE_WARPS) {
            #pragma unroll
            for (int s = 0; s < slots_per_warp; s++) {
                int i = slot_base + s;
                asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                    :: "r"(mbars[i]), "r"(TILE_BYTES) : "memory");
                size_t off = (bid * (size_t)(TILE_BYTES/4)
                            + (size_t)(it * N_INFLIGHT + i) * stride)
                            % (cap_words - TILE_BYTES/4);
                asm volatile("cp.async.bulk.shared::cluster.global"
                    ".mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                    :: "r"(bufs[i]), "l"(src + off),
                       "r"(TILE_BYTES), "r"(mbars[i])
                    : "memory");
            }
        }
        __syncthreads();
        if (tid == 0) {
            #pragma unroll
            for (int i = 0; i < N_INFLIGHT; i++) {
                int done = 0; int spin = 0;
                while (!done && spin < 1000000) {
                    asm volatile(
                        "{.reg .pred p;"
                        " mbarrier.try_wait.shared.b64 p, [%1], 0;"
                        " selp.u32 %0,1,0,p;}"
                        : "=r"(done) : "r"(mbars[i]) : "memory");
                    spin++;
                }
            }
        }
        __syncthreads();
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (tid == 0 && bid == 0) {
        out[0] = t1 - t0;
        ((float*)&out[2])[0] = ((float*)buf_raw)[0];  // anti-DCE
    }
}

int main() {
    CK(cudaSetDevice(0));
    size_t words = 1ull << 30;  // 4 GB
    float* d_src; CK(cudaMalloc(&d_src, words * 4));
    cudaMemset(d_src, 0xa5, words * 4);
    unsigned long long* d_out; CK(cudaMalloc(&d_out, 256));
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    printf("=== V55 HBM3E empirical floor (best recipe) ===\n");
    printf("Spec: 7.68 TB/s. This-device: 7.67. Prior best: 7.31 (V32).\n");
    printf("tile_KB inflight issue_warps shmem_KB N_iters wall_ms TB/s pct_of_7.67\n");

    #define TRY(TILE, NI, NW, ITS) do {                          \
        cudaFuncSetAttribute(v55_hbm_best<TILE,NI,ITS,NW>,       \
            cudaFuncAttributeMaxDynamicSharedMemorySize,         \
            200*1024);                                           \
        int shmem = NI * TILE; if (shmem > 200*1024) break;      \
        v55_hbm_best<TILE,NI,ITS,NW><<<148,128,shmem>>>          \
            ((const float*)d_src,d_out,148,words);               \
        cudaDeviceSynchronize();                                 \
        if (cudaGetLastError())                                  \
            { printf("%d/%d/%d FAIL\n",TILE,NI,NW);              \
              cudaGetLastError(); break; }                       \
        float total = 0;                                         \
        for (int r=0;r<5;r++) {                                  \
            cudaEventRecord(e0);                                 \
            v55_hbm_best<TILE,NI,ITS,NW><<<148,128,shmem>>>      \
                ((const float*)d_src,d_out,148,words);           \
            cudaEventRecord(e1); cudaEventSynchronize(e1);       \
            float ms; cudaEventElapsedTime(&ms,e0,e1); total+=ms;\
        }                                                        \
        float ms = total/5;                                      \
        double bytes = (double)148*ITS*NI*TILE;                  \
        double tbs = bytes/(ms/1e3)/1e12;                        \
        printf("%5d   %4d     %3d         %5d   %4d   %.3f"      \
               "  %.3f  %.1f%%\n",                               \
            TILE/1024,NI,NW,shmem/1024,ITS,ms,tbs,tbs/7.67*100); \
    } while(0)

    // Sweep tile size at fixed depth
    TRY( 4096, 8, 4, 1024);
    TRY( 8192, 8, 4,  512);
    TRY(16384, 8, 4,  256);  // V46 baseline
    TRY(32768, 8, 4,  128);
    TRY(65536, 4, 4,   64);
    // Sweep depth at best tile
    TRY(16384, 4, 2,  256);
    TRY(16384, 8, 2,  256);
    TRY(16384, 8, 4,  256);
    // Persistent-equivalent (1×SM) but more iters to ensure ≥20 ms
    TRY(16384, 8, 4, 1024);

    return 0;
}
```

### C.3.4 ncu metrics

```
dram__bytes_read.sum.per_second                  # authoritative HBM read BW
lts__t_sectors_op_read.sum.pct_of_peak_sustained # L2 traffic (should be ~0)
lts__t_sector_hit_rate.pct                       # confirm L2 hit rate < 5%
sm__warps_active.avg.pct_of_peak_sustained
```

### C.3.5 Decision rule

Take MAX TB/s across the sweep where `lts__t_sector_hit_rate.pct <
10 %` → empirical HBM floor. Compare to V32's 7.31 TB/s and to 7.67
TB/s spec. Use this number as the empirical anchor for ALL "% of HBM
peak" claims going forward.

If the empirical floor is e.g. 7.5 TB/s, retroactively rescale claims
that used 7.31 TB/s denominator.

### C.3.6 Expected effort

~2 hours including ncu sweep. The mbarrier wait loop is the trickiest
part; verify it doesn't go infinite via the `spin < 1000000` guard.

---

## C.4 V56 — NVFP4 A:B mechanism discriminator

**Settles:** Why is power so asymmetric A>>B vs B>>A in NVFP4 K=96
tcgen05.mma? Four candidate mechanisms.

### C.4.1 Background

The catalog has 3 different "correct" answers for NVFP4 A vs B operand
power asymmetry:
- cuBLAS A>B 3:1
- pure tcgen05 B>>A 15-30×
- K=96 single-kernel B>A 2.6× (matches BF16 cuBLAS 2-2.9×)

W3b (NVFP4_DOUBT_REPORT.md) flagged that the underlying source itself
lists 4 plausible mechanisms and walks back the single-mechanism story.
W6 left this as MED with all 3 readings preserved.

V56 attempts to discriminate the mechanisms by isolating each axis.

### C.4.2 Candidate mechanisms

1. **TMA multicast** — A is multicast to N CTAs in cluster, B is
   unicast. Different bus.
2. **A↔B operand swap** — internally MMA treats A and B differently
   in the SMEM staging.
3. **SMEM dwell time** — A sits longer in SMEM (reused across K
   loop), B refreshed each step.
4. **Pipeline depth** — A side has deeper double-buffer than B.

### C.4.3 Mode matrix

- MODE=0 baseline (A toggling random, B static zero)
- MODE=1 swap roles (A static zero, B toggling random) → tests #2 swap
- MODE=2 use cluster MULTICAST for B too → tests #1 multicast asymmetry
- MODE=3 single-buffered A (no double-buffer reuse) → tests #3/#4
  dwell/depth
- MODE=4 cluster size 1 (no multicast at all) → tests #1 again

### C.4.4 Kernel sketch

```cpp
// V56: NVFP4 A vs B mechanism discriminator (tcgen05.mma).
// Run 4 controlled tilt tests; each ELIMINATES one candidate.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

template<int MODE, int CLUSTER, int K_DEPTH>
__global__ __cluster_dims__(CLUSTER,1,1) __launch_bounds__(128,1)
void v56_nvfp4_mma(const uint4* A_data, const uint4* B_data, uint32_t* C_out) {
    extern __shared__ __align__(1024) char smem_raw[];
    // Layouts: A is 128×K, B is K×128 in NVFP4.
    // Two SMEM tiles for double-buffer; A tile 0 / 1 / B tile 0 / 1.
    uint8_t* A_buf[2] = { (uint8_t*)smem_raw, (uint8_t*)smem_raw + 4096 };
    uint8_t* B_buf[2] = { (uint8_t*)smem_raw + 8192,
                          (uint8_t*)smem_raw + 12288 };

    // A-tilt: A_data alternates dense/sparse popcount; B_data fixed all-zero
    //         -> baseline shows A>>B power
    // B-tilt: swap which side toggles
    // ...
    // (full TMA setup elided — same as v46 pattern, 8 KB tiles, mbarriers)

    // Issue tcgen05.mma in a K_DEPTH loop:
    //   for k in K_DEPTH:
    //     if MODE==3 or k_iter == 0: load_A_tile()
    //     load_B_tile()
    //     if MODE==2: TMA_multicast B to all peers
    //     tcgen05.mma.cta_group::1.kind::mxf4 [d], [a], [b],
    //                 [scaleA], [scaleB], 1;
    //     fence.async tcgen05;

    // ... timing + power-probe via NVML in host loop
}

int main() {
    CK(cudaSetDevice(0));
    // Allocate A and B with two contents:
    //   "static": all-zero
    //   "toggle": random with ~16 popcount per byte (peak power)
    // ...

    printf("=== V56 NVFP4 A:B mechanism discriminator ===\n");
    printf("Each row = 1 mode × 1 contents config; record W (NVML), "
           "TFLOPS, ncu pipe util.\n");
    printf("mode  cluster  A_state  B_state   W_avg   TFLOPS  pipe_tensor_pct\n");
    // Drive with NVML sampling at 100 Hz during a 5-sec sustained run per config.
    // Collect: rows for (M0..M4) × (Astatic/Atoggle) × (Bstatic/Btoggle).
    return 0;
}
```

### C.4.5 Power table predictions

(W per CTA at 1005 MHz, baseline B-static A-toggle = 600 W reference)

| Mode | What it changes | Predicts which mechanism if power asymmetry inverts/equalizes |
|---|---|---|
| 0 (base) | none | reference |
| 1 (swap A/B contents) | If swap also flips A>>B → it's CONTENTS not pipeline | rules out asymmetric pipeline (#2/#3/#4) → confirms data-side |
| 2 (B multicast too) | If A=B power gap closes → multicast is the cause (#1) | confirms #1 |
| 3 (single-buffer A) | If A>>B gap GROWS → A dwell time matters (#3) | confirms #3 |
| 4 (cluster=1) | If A=B equalize → multicast was the cause (#1) | confirms #1 |

### C.4.6 Decision tree

1. Mode 1 inverts → mechanism is purely contents-driven; mechanisms
   #1-4 all wrong; revisit data-dep
   (`project_b300_power_data_dep`).
2. Mode 2 equalizes AND mode 4 equalizes → **#1 multicast** is the
   mechanism.
3. Mode 3 amplifies AND mode 1 does NOT invert → **#3 dwell time** is
   mechanism.
4. None of 1-4 changes the asymmetry meaningfully (< 5 % W shift) →
   **#2 swap** (operand asymmetry built into MMA path, not data /
   transport).

### C.4.7 ncu metrics (the harder ones)

```
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
    # caveat: doesn't track tcgen05 well (use the long metric for tcgen05)
lts__t_sectors_aperture_device_op_read.sum
    # multicast vs unicast traffic
sm__inst_executed_pipe_tex.sum
    # TMA issue count
dram__bytes_read.sum.per_second
    # if multicast, B side BW changes
```

NVML power: 100 Hz sample, 5 s sustained, mean of last 4 s.

### C.4.8 Expected effort

~3-4 hours including kernel writing, debug, NVML sampling, decision-
tree walk. NVFP4 + tcgen05 + cluster + multicast is the full Blackwell
stack — getting it to compile is half the work.

---

## C.5 Recommended order

1. **V52 (DONE in W6).** Settled dual-issue.
2. **V55 first.** Anchors the empirical HBM denominator. Affects every
   "% of HBM peak" claim downstream.
3. **V53 second.** Settles DSMEM read/write SoL caveats. Two W3b LOW
   verdicts depend on this.
4. **V54 third.** Settles `__threadfence_system` 1.74× spread.
   Standalone fix, no downstream blast radius.
5. **V56 last.** NVFP4 A:B mechanism discrimination. Most exploratory;
   may not converge to a single answer.

---

## C.6 Files

After running all four, the catalog should add:

- `b300_clean/corrections/V53_RUN_RESULTS.md` (DSMEM fenced)
- `b300_clean/corrections/V54_RUN_RESULTS.md` (membar)
- `b300_clean/corrections/V55_RUN_RESULTS.md` (HBM floor)
- `b300_clean/corrections/V56_RUN_RESULTS.md` (NVFP4 A:B)
- An updated `HEADLINE_CORRECTIONS_v6.md` rolling all four into the
  top-line summary.


---

# APPENDIX D. Provenance map / cross-references

> Every catalog file → its corrections file. Every wave → which
> corrections it spawned. Time-ordered supersession chain. "If you read
> X, also read Y" pairings. The single most useful page when re-reading
> the catalog with knowledge of the rigor sweep.

---

## D.1 Original catalog (188 docs in `b300_clean/`) → corrections file

The wave-1+2 audit produced 17 categorical CORRECTED files plus 5
special-topic CORRECTED files. The mapping is:

### D.1.1 Per-category mapping

| Original `b300_clean/` doc | Wave-1+2 CORRECTED file | Wave-3+ supersedes |
|---|---|---|
| `01_hbm_bandwidth.md` | `corrections/01_hbm_bandwidth_CORRECTED.md` | + `HBM_DENOMINATOR_RESOLUTION.md` (W4) + `HBM_DENOMINATOR_FINAL.md` (W5b) + `HBM_STACKS_INDEPENDENT_VERIFY.md` (W6b) |
| `02_shmem.md` | `corrections/02_shmem_CORRECTED.md` | (no further) |
| `03_caches.md` | `corrections/03_caches_CORRECTED.md` | (no further) |
| `04_fp32_peak.md` | `corrections/04_fp32_peak_CORRECTED.md` | + `V52_RUN_RESULTS.md` (W6a, retracts dual-issue rows) |
| `05_fp_precision_nontensor.md` | `corrections/05_fp_precision_nontensor_CORRECTED.md` | (no further) |
| `06_tensor_cores.md` | `corrections/06_tensor_cores_CORRECTED.md` | + `NVFP4_CONSOLIDATED.md` (W3a, NVFP4 cross-doc) |
| `07_atomics.md` | `corrections/07_atomics_CORRECTED.md` | + `STRAYS_CORRECTED.md` §8 (L2 atomic units MED downgrade) |
| `08_sync_primitives.md` | `corrections/08_sync_primitives_CORRECTED.md` | (V54 sketch in C.2 will close this) |
| `09_memory_apis.md` | `corrections/09_memory_apis_CORRECTED.md` | + `V46_DOUBT_REPORT.md` (V46 demotion) |
| `10_launch_overhead.md` | `corrections/10_launch_overhead_CORRECTED.md` | (no further) |
| `11_block_scheduling.md` | `corrections/11_block_scheduling_CORRECTED.md` | (no further) |
| `12_nvlink_p2p.md` | `corrections/12_nvlink_p2p_CORRECTED.md` | (no further; NVLink-5 web-confirmed) |
| `13_pcie_system.md` | `corrections/13_pcie_system_CORRECTED.md` | (defers to NVLink agent) |
| `14_math_intrinsics.md` | `corrections/14_math_intrinsics_CORRECTED.md` | + `MATH_INCONSISTENCY_LOG.md` |
| `15_integer_bit_ops.md` | `corrections/15_integer_bit_ops_CORRECTED.md` | + `INT_INCONSISTENCY_LOG.md` |
| `16_power_clock.md` | `corrections/16_power_clock_CORRECTED.md` | (no further) |
| `17_nvrtc_module.md` | `corrections/17_nvrtc_module_CORRECTED.md` | (no further) |
| `DSMEM_REFERENCE.md`, `DSMEM_DOUBT_REPORT.md`, `2CTA_DEDUP.md` | `corrections/DSMEM_CORRECTED.md` | + `DSMEM_DOUBT_REPORT.md` (W3b) |
| `M3_TOPOLOGY_CHEATSHEET.md`, `M5_MEMORY_CHEATSHEET.md`, `M14_OVERVIEW.md`, `M16_REFINED.md` | `corrections/META_DOCS_CORRECTED.md`, `corrections/M_SYNTHESIS_CORRECTIONS.md` | + `M_SYNTHESIS_INCONSISTENCY_LOG.md` |
| `V8_*.md`, `V10_*.md` (legacy) | `corrections/V8_V10_MISC_CORRECTED.md` | (V8 DSMEM 71934d0 retracted) |
| `NVFP4_*.md` series | `corrections/NVFP4_CONSOLIDATED.md` | + `NVFP4_DOUBT_REPORT.md` (W3b) + `NVFP4_INCONSISTENCY_LOG.md` |
| `TCGEN05_PERF_WATTS.md`, `TCGEN05_PERFW_CLEAN_2TRIAL.md`, `TCGEN05_DEDUP*.md` | `corrections/TCGEN05_DEDUP_CONSOLIDATED.md`, `corrections/TCGEN05_POWER_CONSOLIDATED.md` | + `DEDUP_INCONSISTENCY_LOG.md` |
| `HBM_DATA_DEPENDENCE.md` | `corrections/STRAYS_CORRECTED.md` §2 | RETRACTED (real swing 240-554 W per POPCOUNT_3TIER) |
| `CURIOSITY_LIST_V{2,3,4,5,6,7,8}.md` | `corrections/CURIOSITY_LISTS_AUDIT.md` | (V2 hashes 88% hallucinated, V4-V8 clean) |

### D.1.2 Files with NO direct corrections (still valid as-is)

These docs were checked during the rigor sweep and found to need NO
corrections:

- `b300_clean/A1_DUAL_ISSUE_RIGOR.md` (still valid pending V52 reframing)
- `b300_clean/A2_SCHEDULER_RIGOR.md`
- `b300_clean/A3_SCOREBOARD_DEPTH.md`
- `b300_clean/A4_FFMA_PORT_PRESSURE.md` (3-source FFMA cap confirmed)
- `b300_clean/A6_PER_PIPE_REFERENCE.md` (4-tier pipe ladder confirmed)
- `b300_clean/B1_DUAL_ISSUE_FFMA_IADD3.md` (now superseded by V52)
- `b300_clean/B2_FFMA_LDG_DUAL.md`
- `b300_clean/C3_LOP3_LUT_DEEP.md`
- `b300_clean/D2_L1_CAPACITY_RIGOR.md`
- `b300_clean/D3_L2_SECTOR_RIGOR.md` (126 MB practical L2 confirmed)
- `b300_clean/D5_*.md` (subset of A_TO_D_RIGOR_AUDIT)
- `b300_clean/D6_*.md` (3-source FFMA confirmed)
- `b300_clean/D7_TMEM_BW.md` (60 TB/s consensus)
- `b300_clean/V32_TMA_MULTICAST_FINDINGS.md` (14.91 TB/s ceiling
  confirmed)
- `b300_clean/V33_TMA_SINGLE_DEEP.md` (6.72 TB/s confirmed)
- `b300_clean/V34_TMA_WRITE.md` (7.17 TB/s confirmed)
- `b300_clean/V40_PIPE_PLACEMENT.md` (IADD3 on FMA pipe confirmed)
- `b300_clean/V41_V48_FINDINGS.md` (V46 reframed; V47/V48 confirmed)

---

## D.2 Wave timeline and what each wave produced

### D.2.1 Wave 1 (initial sweep, late March 2026)

- 188 individual docs in `b300_clean/`.
- No corrections; just measurements.
- Initial CLAUDE.md memory entries (some later flagged for retraction).

### D.2.2 Wave 2 (CORRECTED files batch, early April 2026)

- 17 categorical CORRECTED files (`01_*` through `17_*`).
- 5 special-topic CORRECTED files (DSMEM, META, NVFP4, etc.).
- 20 INCONSISTENCY_LOG files.
- Synthesis: `MASTER_INDEX.md`, `HEADLINE_CORRECTIONS.md`,
  `B300_TRUE_REFERENCE_v2_DRAFT.md`.

### D.2.3 Wave 3a (topical audit, April 2026)

- `CURIOSITY_LISTS_AUDIT.md` (V2 hashes hallucinated)
- `TCGEN05_POWER_CONSOLIDATED.md` (single-trial retracted)
- `STRAYS_CORRECTED.md` (HBM_DATA_DEPENDENCE retracted)
- `A_TO_D_RIGOR_AUDIT.md` (A1/A6/B1/D5 partial retractions)

### D.2.4 Wave 3b (adversarial doubt swarm)

Six doubt reports:
- `SYNTHESIS_DOUBT_LOG.md`
- `V46_DOUBT_REPORT.md`
- `DUAL_ISSUE_DOUBT_REPORT.md`
- `DSMEM_DOUBT_REPORT.md`
- `NVFP4_DOUBT_REPORT.md`
- `CROSS_AGENT_DOUBT_LOG.md`

### D.2.5 Wave 3c (synthesis of doubt)

- `DOUBT_LOG.md` (per-claim verdicts)
- `HEADLINE_CORRECTIONS_v2.md` (doubt-aware TL;DR)
- `MASTER_INDEX_v2.md`

### D.2.6 Wave 3d (META_DOUBT — auditing the doubt reports)

- `META_DOUBT_REPORT.md`
- Result: V49/V50 re-promoted from LOW to MED (which was wrong, see
  W5/W6).

### D.2.7 Wave 4 (April 22, mid-day)

- `WAVE4_CHANGES.md`
- `HBM_DENOMINATOR_RESOLUTION.md` (declared 7672 a "ghost", later
  retracted by W5b)
- `HEADLINE_CORRECTIONS_v3.md`
- `RETEST_PROPOSALS.md` (V52-V56 sketches)
- `V51_INVESTIGATION.md` (V51 untracked bug found)
- `UNRESOLVED_PROMOTED.md`
- `CONFIDENCE_LADDER.md` (303-row table)

### D.2.8 Wave 5 (April 22, afternoon)

- `WAVE5_CHANGES.md`
- `SASS_VERIFY_DUAL_ISSUE.md` (W5a, V49 SASS investigation)
- `HBM_DENOMINATOR_FINAL.md` (W5b, 7672 is real not ghost)
- `PROPOSED_FIXES.md` (W5c, fix diffs for stack count + denominator)
- `CONFIDENCE_LADDER_PATCH.md` (W5d, immediately stale)
- `CROSS_LINK_AUDIT.md` (W5e, 17 stale cross-references)
- `HEADLINE_CORRECTIONS_v4.md`
- `DOUBT_LOG_v2.md`

### D.2.9 Wave 6 (April 22, late afternoon)

- `WAVE6_CHANGES.md`
- `V52_RUN_RESULTS.md` (THE empirical anchor)
- `HBM_STACKS_INDEPENDENT_VERIFY.md` (8 stacks confirmed,
  bus width fused)
- `HEADLINE_CORRECTIONS_v5.md` (current top-line)
- `CONFIDENCE_LADDER_PATCH_v2.md` (immediately stale)
- `CONFIDENCE_LADDER_PATCH_v3.md` (current)
- `META_LESSONS.md` (5-level zigzag distilled)
- `RETEST_PROPOSALS.md` updated

### D.2.10 Wave 7 (this canonical reference)

- `b300_clean/canonical_parts/F_methodology_appendices.md` (this file)
- Sibling parts A/B/C/D/E producing §1-55.
- Stitched into `b300_clean/B300_AC_CANONICAL.md`.

---

## D.3 Headline correction chronology

### D.3.1 v1 (wave-1+2)

`b300_clean/corrections/HEADLINE_CORRECTIONS.md`. The original 10
headlines: NVLink v7 (wrong), HBM 8 TB/s (rounded), V46 NEW SoL 98.5%
(framing artifact), MUFU 47.8 G (latency-bound mislabel), V49/V50 dual
55%/74% (HIGH, would later be retracted), DSMEM TB/s peaks (DCE
artifacts), NVFP4 A:B story, 3-source FFMA cap 50 TFLOPS, NVFP4 K=96
ceiling, etc.

### D.3.2 v2 (wave-3c)

`HEADLINE_CORRECTIONS_v2.md`. Added the 10 wave-3 doubts: V46 demoted,
V49/V50 LOW (but for wrong reason — under-occupancy), DSMEM caveats,
NVFP4 A:B preserve all 3.

### D.3.3 v3 (wave-4)

`HEADLINE_CORRECTIONS_v3.md`. Reversed V49/V50 LOW → MED based on
META_DOUBT (later reversed again). Established 7680 GB/s as canonical
HBM denominator (later softened in v4).

### D.3.4 v4 (wave-5)

`HEADLINE_CORRECTIONS_v4.md`. Re-downgraded V49/V50 to LOW based on
SASS analysis. Adopted dual 7680/7672 denominator citation per W5b.

### D.3.5 v5 (wave-6, current)

`HEADLINE_CORRECTIONS_v5.md`. V49/V50 dual-issue empirically settled by
V52 ncu metrics (`pipe_alu + pipe_fma = 147 %`). HBM bus revealed as
7680-bit fused on this AC SKU. Both top-line claims now HIGH with
strong empirical anchor. M8 PIPE_OVERLAP_MATRIX confirmed.

### D.3.6 The supersession chain

```
v1 (W1+W2)
  ↓ v2 (W3c) — dual-issue LOW for under-occupancy reason
    ↓ v3 (W4) — V49/V50 reversed to MED (wrong)
      ↓ v4 (W5) — re-downgraded to LOW with SASS reason
        ↓ v5 (W6) — empirically RETRACTED (artifact) AND
                    architectural answer (free overlap) HIGH
```

Each version supersedes the prior. v5 is the current top-line. This
canonical reference (Wave 7) supersedes v5 by integrating into a
unified document.

---

## D.4 "If you read X, also read Y" pairings

When reading the catalog, certain files have essential companions:

| If you read | Also read |
|---|---|
| `B300_TRUE_REFERENCE.md` | + `corrections/HEADLINE_CORRECTIONS_v5.md` (current row downgrades) + `corrections/V52_RUN_RESULTS.md` (dual-issue retraction) |
| `01_hbm_bandwidth.md` | + `corrections/01_hbm_bandwidth_CORRECTED.md` + `corrections/HBM_STACKS_INDEPENDENT_VERIFY.md` (8-stack derivation) + `corrections/HBM_DENOMINATOR_FINAL.md` (dual-cite rule) |
| `04_fp32_peak.md` | + `corrections/04_fp32_peak_CORRECTED.md` + `corrections/V52_RUN_RESULTS.md` (dual-issue retraction) + `corrections/SASS_VERIFY_DUAL_ISSUE.md` |
| `06_tensor_cores.md` | + `corrections/06_tensor_cores_CORRECTED.md` + `corrections/NVFP4_CONSOLIDATED.md` + `corrections/NVFP4_DOUBT_REPORT.md` |
| `09_memory_apis.md` | + `corrections/09_memory_apis_CORRECTED.md` + `corrections/V46_DOUBT_REPORT.md` (TMA pipelined demotion) |
| `10_launch_overhead.md` | + `corrections/10_launch_overhead_CORRECTED.md` + `V9_GRAPH_LAUNCH.md` (cudaGraph myth-bust) |
| `11_block_scheduling.md` | + `corrections/11_block_scheduling_CORRECTED.md` (6 contradictions reconciled) |
| `DSMEM_REFERENCE.md` | + `corrections/DSMEM_CORRECTED.md` + `corrections/DSMEM_DOUBT_REPORT.md` |
| `V41_V48_FINDINGS.md` | + `corrections/V46_DOUBT_REPORT.md` (V46 demotion + denominator) + V52 (retracts the 5-wave dual-issue inferences) |
| Any `M*.md` synthesis | + `corrections/M_SYNTHESIS_CORRECTIONS.md` (M8/M14/M16 retractions) |
| `CURIOSITY_LIST_V2.md` | + `corrections/CURIOSITY_LISTS_AUDIT.md` (88% hash hallucination) |
| `HBM_DATA_DEPENDENCE.md` | + `corrections/STRAYS_CORRECTED.md` §2 (RETRACTED, see POPCOUNT_3TIER for real numbers) |
| `TCGEN05_PERF_WATTS.md` | + `corrections/TCGEN05_POWER_CONSOLIDATED.md` (use TCGEN05_PERFW_CLEAN_2TRIAL) |
| Any "% of NVLink-4" claim | + `corrections/12_nvlink_p2p_CORRECTED.md` (NVLink-5, 900 GB/s/dir) |
| Any "12 HBM stacks" claim | + `corrections/HBM_STACKS_INDEPENDENT_VERIFY.md` (it's 8 stacks of 12-Hi) |

---

## D.5 Time-ordered superseded chain

The catalog has many "I once said X" claims that have been overturned.
This is the chronological list of major retractions:

### D.5.1 NVLink — "v7" → "v5"

| Date | Claim | Retracted by |
|---|---|---|
| Pre-W1 | "NVLink v7, 757 GB/s/dir spec" | CLAUDE.md memory (still cited in legacy contexts) |
| W2 | Cross-checked against vendor specs | NVLink + PCIe agents in `12_nvlink_p2p_CORRECTED.md` |
| W3c | "NVLink-5, 900 GB/s/dir spec" | `HEADLINE_CORRECTIONS_v2.md` row 1 |
| W6 | unchanged | `HEADLINE_CORRECTIONS_v5.md` row 1 (HIGH) |

Status: HIGH confidence. CLAUDE.md memory file still has the old "v7"
reference; treat as unauthoritative.

### D.5.2 HBM stack count — "12" → "8 of 12-Hi"

| Date | Claim | Retracted by |
|---|---|---|
| Pre-W1 | "12 HBM stacks × 1024-bit" | Multiple early docs |
| W4 | "8 stacks × 12-Hi each" | `HBM_DENOMINATOR_RESOLUTION.md` §1 |
| W6b | confirmed by `cudaGetDeviceProperties` `memoryBusWidth = 7680` | `HBM_STACKS_INDEPENDENT_VERIFY.md` |

Status: HIGH confidence. Files still containing "12 stacks":
- `B300_TRUE_REFERENCE.md` line 15 (proposed fix in `PROPOSED_FIXES.md`
  Fix 2)
- `01_hbm_bandwidth.md` lines 3, 136 (proposed fix in `PROPOSED_FIXES.md`
  Fix 3a, 3b)

### D.5.3 HBM denominator — "8" → "7672" → "7680" → "7680/7672 dual"

| Date | Claim | Retracted by |
|---|---|---|
| Pre-W1 | "~8 TB/s HBM peak" | CLAUDE.md memory |
| W2 | "7672 GB/s post-ECC spec" | `01_hbm_bandwidth_CORRECTED.md` |
| W4 | "7680 GB/s spec; 7672 is arithmetic ghost" | `HBM_DENOMINATOR_RESOLUTION.md` |
| W5b | "7672 is real (not ghost); cite both 7680 spec and 7672 actual" | `HBM_DENOMINATOR_FINAL.md` |
| W6 | "7.68 spec (full bus) / 7.67 this-device (7680-bit fused)" | `HEADLINE_CORRECTIONS_v5.md` row 3 |

Status: HIGH confidence. Use dual citation per Rule 12.

### D.5.4 V49/V50 dual-issue — HIGH → LOW → MED → LOW → HIGH+RETRACT

The 5-level zigzag (full chronology in Appendix A).

| Wave | Verdict | Mechanism |
|---|---|---|
| W1+W2 | HIGH 55%/74% | reproducibility |
| W3b | LOW | under-occupancy (wrong) |
| W4 | MED | V8 falsification of W3b (right falsification, wrong verdict) |
| W5a | LOW | loop-overhead contamination (right mechanism) |
| W6 | HIGH (architectural) + RETRACTED (numbers) | V52 ncu shows pipes overlap freely |

Status: HIGH confidence on architectural truth (free overlap). Numbers
55%/74% RETRACTED.

### D.5.5 V46 TMA pipelined — "98.5% NEW SoL" → 93.9% (re-anchored)

| Date | Claim | Retracted by |
|---|---|---|
| W2 | V46 = 7.20 TB/s = "98.5% NEW HBM read SoL" | `09_memory_apis_CORRECTED.md`, M14 update |
| W3b | "98.5% used wrong denominator (7.31 empirical)" | `V46_DOUBT_REPORT.md` |
| W4 | "Re-anchor as 93.8% of 7672 spec" | `HEADLINE_CORRECTIONS_v3.md` |
| W6 | "93.9% of 7.67 this-device peak / 93.8% of 7.68 spec" | `HEADLINE_CORRECTIONS_v5.md` row 4 |

Status: HIGH confidence. Architectural lesson "TMA reads benefit from
8-deep pipelining" remains valid. Number framing was the artifact.

### D.5.6 DSMEM TB/s peaks — RETRACTED as DCE artifacts

| Date | Claim | Retracted by |
|---|---|---|
| W2 | V8 DSMEM 37 TB/s = 97% of SHMEM peak (commit 71934d0) | `DSMEM_INCONSISTENCY_LOG.md` §A |
| W2 | Real aggregate ~40 GB/s/cluster | `DSMEM_CORRECTED.md` |
| W3b | Read 40 GB/s is chain-bound (not absolute) | `DSMEM_DOUBT_REPORT.md` |
| W3b | Write 560 GB/s is issue rate (not completion) | `DSMEM_DOUBT_REPORT.md` |
| W3b | "No shared bus" claim under-issued by 30× in V17 | `DSMEM_DOUBT_REPORT.md` |
| W6 | unchanged; V53 sketch in C.1 will close caveats | `HEADLINE_CORRECTIONS_v5.md` row 10 |

Status: 37 TB/s RETRACTED as DCE artifact. Real numbers caveated:
read MED (chain-bound), write LOW-MED (issue rate), no-shared-bus LOW
(under-issued).

### D.5.7 MUFU 47.8 G/chip — relabeled from "XU peak" to "1-chain latency"

| Date | Claim | Retracted by |
|---|---|---|
| W2 | "MUFU rsqrt = 99.49% XU pipe (47.8 GMUFU/s)" (V8 commit 29b9b3b) | `MATH_INCONSISTENCY_LOG.md` #3 |
| W2 | True saturated MUFU = 4.74 G/chip | `14_math_intrinsics_CORRECTED.md` §1 |
| W6 | "M14/M16 row is 1-chain latency-bound, not pipe-saturated; saturated peak is 10× lower at 4.74 G" | `HEADLINE_CORRECTIONS_v5.md` row 4 (refined wording) |

Status: HIGH confidence. The 47.8 G is REAL but is the latency-bound
1-chain throughput, not the pipe-saturated peak.

### D.5.8 SMEM atomic aggregate — 4.2 T → 2.27 T

| Date | Claim | Source |
|---|---|---|
| Pre-W1 | "SMEM atomic 4.2 Tops/s no-contention" | CLAUDE memory `project_b300_v8_complete` |
| W2 | "Real SMEM atomic INT32 aggregate = 2.27 Tatomic/s" | `02_shmem_CORRECTED.md`, `07_atomics_CORRECTED.md` |
| W3b | "4.2 T figure NOT reproduced; provenance unknown" | `CROSS_AGENT_DOUBT_LOG.md` #12 |

Status: 2.27 T HIGH (atomics + SHMEM agree). 4.2 T unsourced; CLAUDE
memory needs updating.

### D.5.9 NVFP4 K=96 ceiling — 10.8 PF → 11.42 PF

| Date | Claim | Source |
|---|---|---|
| Pre-W1 | "cuBLAS 13.4 caps 10.8 PF (72% of 15 PF spec) at large-N rect" | CLAUDE memory `project_b300_nvfp4_k96_ceiling` |
| W2 | "cuBLAS + cudaGraph BPG=16 = 11.42 PF (76.2%)" | `NVFP4_CONSOLIDATED.md` §1 |
| W3b | "Single-shape, single-BPG (no sweep); treat as upper-bound at this shape" | `NVFP4_DOUBT_REPORT.md` #2 |

Status: 11.42 PF MED (single shape, no sweep). Treat as upper-bound at
8K² K=38400 BPG=16. CLAUDE memory's 10.8 PF should be updated.

### D.5.10 cudaGraph single-node speedup — RETRACTED

| Date | Claim | Source |
|---|---|---|
| Pre-W1 | "cudaGraph always faster than direct launch" | older catalog rows |
| W2 | "1-node graph = 2.05 µs ≈ direct 2.06 µs (NO speedup)" | `V9_GRAPH_LAUNCH.md` |
| W6 | "Only batch ≥ 10 kernels per launch yields speedup" | `10_launch_overhead_CORRECTED.md` |

Status: HIGH confidence. 100-kernel batch = 3.5×; 1000-kernel batch =
3.7×. Single-node = NO speedup.

---

## D.6 Confidence ladder per CORRECTED file

After the rigor sweep, here's the per-file confidence assessment:

| File | Original tag | Doubt-aware tag | Reason |
|---|---|---|---|
| `01_hbm_bandwidth_CORRECTED.md` | HIGH | **HIGH** | Anchors 7672 spec; multi-source ladder; only HBM write 7.57 attribution open |
| `02_shmem_CORRECTED.md` | HIGH | **HIGH** | Bank-conflict regime split well-flagged; UNRESOLVED honest |
| `03_caches_CORRECTED.md` | HIGH | **HIGH** | Sole agent that disambiguates 3 L2 BW metrics |
| `04_fp32_peak_CORRECTED.md` | HIGH | **HIGH** (after V52 update) | Adopts V49/V50 — V52 settles dispute in favor of free overlap |
| `05_fp_precision_nontensor_CORRECTED.md` | MED | MED | No challenge from doubt swarm |
| `06_tensor_cores_CORRECTED.md` | HIGH | **MED** | Cites stale 10.8 PF K=96 without deferring to NVFP4 agent's 11.42 |
| `07_atomics_CORRECTED.md` | MED | **MED** | L2 atomic units ~32 should be MED not HIGH per STRAYS audit |
| `08_sync_primitives_CORRECTED.md` | MED | **MED-HIGH** | Cleanly preserves spread on threadfence; F2/F6 correctly reconciled |
| `09_memory_apis_CORRECTED.md` | MED | **LOW-MED** | Uses 7.2 TB/s denominator; promotes V46 to "NEW SoL" which HBM agent demotes |
| `10_launch_overhead_CORRECTED.md` | MED | MED | No challenge |
| `11_block_scheduling_CORRECTED.md` | MED | MED | Topology-only; some open contradictions |
| `12_nvlink_p2p_CORRECTED.md` | HIGH | **HIGH** | NVLink-5 web-confirmed |
| `13_pcie_system_CORRECTED.md` | MED | MED-HIGH | Defers correctly to NVLink agent |
| `14_math_intrinsics_CORRECTED.md` | MED | **HIGH** | Cross-agent consensus on 4.74/9.22 G MUFU |
| `15_integer_bit_ops_CORRECTED.md` | MED | **MED** | IADD3 0.5 vs 0.66 unresolved |
| `16_power_clock_CORRECTED.md` | MED | MED | No direct challenge |
| `17_nvrtc_module_CORRECTED.md` | MED | MED | No challenge |
| `DSMEM_CORRECTED.md` | HIGH | **MED** (downgraded) | 40 GB/s chain-bound; 560 GB/s issue-rate; "no shared bus" under-issued |
| `META_DOCS_CORRECTED.md` | MED | MED | Correctly demotes V46 framing |
| `M_SYNTHESIS_CORRECTIONS.md` | MED | **LOW-MED** | Adopts V49/V50 dual-issue (LOW); flattens IADD3/PRMT 30% gaps |
| `V8_V10_MISC_CORRECTED.md` | MED | MED | 3-source FFMA cap well-supported |
| `NVFP4_CONSOLIDATED.md` | MED + open | **MED** (preserve all 3 A:B readings) | Wave-2 over-resolved single mechanism |
| `TCGEN05_DEDUP_CONSOLIDATED.md` | MED + open | MED + open | No challenge |

---

## D.7 Open backlog (UNRESOLVED items requiring further work)

After waves 1-6, 30 items remain UNRESOLVED. The retest sketches in
Appendix C cover items 1-5 above. The full list:

| # | Item | Required test |
|---|---|---|
| 1 | `__threadfence_system` true cost (1750/2870/3042 cy spread) | V54 in C.2 |
| 2 | `__threadfence` (GPU) cost (258/281/292/320 cy = 24% spread) | Single ncu pass with all 4 patterns |
| 3 | HBM write 7.57 TB/s SoL provenance | Re-run NINJA STG vs V8 TMA bulk back-to-back |
| 4 | DSMEM read non-chained ILP ceiling | V53 read kernel in C.1 |
| 5 | DSMEM write delivery vs issue rate | V53 write kernel in C.1 |
| 6 | DSMEM "no shared bus" — full-issue 8-cluster ring sweep | 8 CTAs × 4 warps × ILP=16 ring |
| 7 | NVFP4 A:B 3-way mechanism | V56 in C.4 |
| 8 | NVFP4 11.42 PF reproducibility (BPG sweep) | cudaGraph BPG sweep across 5+ shapes |
| 9 | IADD3 rate 0.5 (A6) vs 0.66 (V40) | A6-style sweep at 4+ warps/SMSP |
| 10 | PRMT pipe placement (V40 "permute" vs A6 "INT-bit") | A6-style sweep on PRMT |
| 11 | A3 scoreboard depth (≥32, never plateaued) | Test with N>32 + ncu |
| 12 | L2 atomic unit count | Stride sweep with ncu lts__t_bytes per partition |
| 13 | Single-MMA cache depth (1 slot, 2 slots, or 1+alternation?) | Pattern rotation under per-MMA isolation |
| 14 | Why is 2-pattern (ABAB) sub-tile WORSE than 3-pattern (ABCABC)? | Microsweep with per-cycle clock64 |
| 15 | TMA pipeline-depth optimum (V46 used 8; knee unknown) | Sweep depth 2..16 with ncu (V55 in C.3) |
| 16 | TMA multicast at cluster ∈ {2,4,6,8} | Cluster sweep |
| 17 | `cuStreamWriteValue32` 0.45 µs (memory) vs 2.47 µs (catalog) | Decompose host-call vs full pair |
| 18 | LDS 32-way bank-conflict cost across regimes | Single-warp vs multi-warp matrix |
| 19 | Cluster=2 21% slower than ≥3 — single-GPC vs multi-GPC | `gpc__cycles_active.per_pgpc_id` ncu pass |
| 20 | Cooperative-grid SM mapping | `cudaLaunchCooperativeKernel` + per-CTA SM-id dump |
| 21 | `mma.sync kind::f8f6f4` sub-tile dedup | Repeat dedup recipe with mma.sync FP8 |
| 22 | NVRTC vs nvcc cubin equivalence (never SASS-diffed) | Compile both paths, sass-diff |
| 23 | A4/D6 broadcast operand reuse cache mechanism | ncu `pipe_fma_collector_*` if available |
| 24 | A1 SHFL + FFMA 14.7% overlap mechanism | Redo with V37/V38 setup |
| 25 | B2 LDG no-chain SLOWER than chain-dep | Retest with current anti-DCE |
| 26 | DRAM data-dependence at boost clock (only 1005/1500 measured) | Popcount sweep at 1920/2032 MHz |
| 27 | NVFP4 K=96 ULTRA at non-square N (K-id shape-conditional) | NVFP4 N-shape sweep |
| 28 | Cross-precision NVFP4 K=96 path: K-id 5-gate model from BF16? | Cross-precision K-id N-stride sweep |
| 29 | TF/W boost clock full ladder | All 7 precisions at boost |
| 30 | Per-ncu pipe-cycles-active semantic verification (V52 caveat) | Compare ncu to PTX-level event counter |

---

## D.8 The reading-order linearization

For a reader new to this catalog, the recommended consumption order is:

1. **This document (`F_methodology_appendices.md`)** — methodology
   spine.
2. Sibling parts §1-55 — categorical content.
3. `b300_clean/B300_AC_CANONICAL.md` (the stitched whole) once
   assembled.
4. For specific topics, jump to the relevant CORRECTED file via §D.1
   above.
5. For doubt context, read the matching `*_DOUBT_REPORT.md` if one
   exists.
6. For the 5-wave dual-issue arc specifically, read Appendix A then
   `META_LESSONS.md`.

For a reader updating the catalog (next wave):

1. Read this document end-to-end.
2. Pick an UNRESOLVED item from §D.7.
3. Apply the V52-style methodology (see Appendix A.9).
4. Run the test, collect ncu, SASS-verify.
5. Write up in `corrections/V<N>_RUN_RESULTS.md`.
6. Update `HEADLINE_CORRECTIONS_v6.md` with the resolution.
7. Update this document (§D.7) to mark the item RESOLVED.


---

# APPENDIX E. Footguns index

> Alphabetized by topic. Every ⚠ callout from §1-65 plus this appendix.
> The single most useful page for an LLM trying to avoid quoting stale
> info. Each entry: short symptom, mechanism, defense, source section.

---

## E.A — Atomics

### A.1 SMEM atomic aggregate

⚠ "SMEM atomic 4.2 Tops/s no-contention" is unsourced. Real number is
2.27 T (atomics + SHMEM agents agree). CLAUDE memory `project_b300_v8_complete`
has the wrong 4.2 T; treat as unauthoritative.
**Source:** §63 row 12; CROSS_AGENT_DOUBT_LOG #12.

### A.2 L2 atomic units

⚠ L2 atomic units count "~32" should be MED (derived ceiling, not
direct measurement).
**Source:** §D.6; STRAYS_CORRECTED §8.

### A.3 Atomic stride / unroll inflation

⚠ L2 atomic stride-4 peak: 449 / 504 / 1005 Gops/s across UNROLL=1, 16,
32. Always pair stride × UNROLL × L2-residency in atomic claims.
Cache-line combining can inflate Gops 8× without proportional BW.
**Source:** CLAUDE memory `feedback_units_sanity`; CROSS_AGENT #13.

### A.4 SMEM atomic FP32 32-way contention

⚠ SMEM atomic FP32 32-way contention costs 5729 cy (67× penalty) vs
INT32 4.6 cy uncontended. Always specify dtype and contention level
in atomic claims.
**Source:** 02_shmem_CORRECTED §atomics.

---

## E.B — Bandwidth (HBM)

### B.1 HBM "12 stacks" claim

⚠ B300 has **8 HBM3E stacks of 12-Hi each**, NOT "12 stacks". The
"12" refers to die-stack height, not stack count. Files still
containing the wrong claim:
- `B300_TRUE_REFERENCE.md` line 15
- `01_hbm_bandwidth.md` lines 3, 136
**Source:** §60.2; HBM_STACKS_INDEPENDENT_VERIFY.md.

### B.2 HBM denominator drift

⚠ Three different HBM denominators across catalog: 7672 / 7.31 / 8.0
TB/s. Cross-doc % numbers are NOT comparable. Use **dual citation**:
- 7.68 TB/s (spec post-ECC) for cross-vendor.
- 7.67 TB/s (this-device post-ECC) for SoL on this part.
**Source:** §62 rule 12; HBM_DENOMINATOR_FINAL.md.

### B.3 HBM "8 TB/s spec" marketing

⚠ "8 TB/s" is marketing-rounded; never use as a real denominator. The
actual spec post-ECC is 7.68 TB/s; this-device is 7.67 TB/s.
**Source:** §60.3.

### B.4 V46 "98.5 % NEW SoL"

⚠ V46's "98.5 % NEW HBM read SoL" used denominator 7.31 TB/s
(empirical pure-direction). Re-anchored against spec/this-device =
93.8 / 93.9 %. The architectural lesson holds; the framing was the
artifact.
**Source:** §56.3; V46_DOUBT_REPORT.md.

### B.5 HBM write 7.57 TB/s provenance

⚠ HBM write SoL 7.57 TB/s (NINJA STG vs V8 TMA bulk) has CONTESTED
PROVENANCE. UNRESOLVED until back-to-back retest.
**Source:** §D.5.5; HBM_INCONSISTENCY_LOG #3.

### B.6 HBM_DATA_DEPENDENCE.md "<50 W"

⚠ `HBM_DATA_DEPENDENCE.md` claim "<50 W" is RETRACTED. Real DRAM
data-dep swing is 240–554 W (popcount d=0→16→32 bell curve). Use
POPCOUNT_3TIER or `project_b300_power_data_dep` as authoritative.
**Source:** §63.11; STRAYS_CORRECTED §2.

### B.7 HBM working set

⚠ Working set < 126 MB will be absorbed by L2 (capacity 132 MiB
nominal / 126 MB practical). Use ≥ 4 GB working set with stride-per-iter
to defeat L2 for HBM measurements.
**Source:** §61.4 rule 2; §60.4.

### B.8 memoryBusWidth = 7680 not 8192

⚠ `cudaGetDeviceProperties.memoryBusWidth = 7680` on this AC SKU
(yield-fused, 1/16 controller off). Affects ALL "% of HBM peak" math.
Other (non-AC) SKUs may have full 8192-bit bus.
**Source:** §60.2; HBM_STACKS_INDEPENDENT_VERIFY.md.

---

## E.C — Cluster topology

### C.1 Cluster ≥ 32 silently no-ops

⚠ Cluster size ≥ 32 launches return success but produce no multi-CTA
placement (silently no-op). Always check `cudaOccupancyMaxActiveClusters`
if you depend on cluster behavior.
**Source:** §58.4.

### C.2 Cluster placement determinism

⚠ Cluster placement SM-id set {0, 1, 16, 17, ...} is deterministic
ONLY on an otherwise-idle GPU. Production workloads see runtime-chosen
SM IDs; topology preserved but absolute IDs vary.
**Source:** §58.3; DSMEM_REFERENCE.

### C.3 GPC count and "spare SMs"

⚠ TRUE_REFERENCE's "144 active + 4 spare SMs" framing has no
architectural basis. The model is **8 GPCs: 2 with 20 SMs, 6 with 18
SMs = 148 active**.
**Source:** §58.2.

### C.4 "GPC-row" ≠ "GPC"

⚠ I8/M3 use "GPC-row" to mean "16-SM stride window". B300 has 8 GPCs
(NVIDIA hardware unit), not 9.25 "rows". Standardize on "stride-16
column" for the scheduler addressing window.
**Source:** §58.2 retraction 2.

### C.5 SM-id → GPC mapping unverified

⚠ Cluster placement claims "stride-16 = different GPC" are consistent
with bus-width math but never directly verified by
`gpc__cycles_active.per_pgpc_id`. Collect this metric to anchor.
**Source:** §58.5 #1; §58.6 last footgun.

### C.6 "Cluster blocks placed within same GPC"

⚠ RETRACTED in 11.md commit 79372e6. Cluster of 8 spans **4 GPCs**.
**Source:** §58.6.

### C.7 "10 GPCs" claim

⚠ RETRACTED. B300 has 8 GPCs. The "10 GPCs (9×16 + 1×4)" claim was an
old miscount.
**Source:** §58.6.

---

## E.D — DSMEM

### D.1 DSMEM "37 TB/s = 97 % SHMEM peak"

⚠ V8 commit 71934d0 RETRACTED as DCE artifact. SASS showed compile-time
invariant offsets LICM'd / CSE'd. Real aggregate ≈ 40 GB/s/cluster (off
by ~1000×).
**Source:** §D.5.6; CURIOSITY_LISTS_AUDIT V8.

### D.2 DSMEM 40 GB/s read

⚠ DSMEM read aggregate "40 GB/s/cluster" is **chain-bound**, NOT
absolute. V21 used dependent-chain pattern. Non-chained ILP could
reach 60-80 GB/s. V53 in C.1 settles.
**Source:** §D.5.6; DSMEM_DOUBT_REPORT.

### D.3 DSMEM 560 GB/s write

⚠ DSMEM write aggregate "560 GB/s/cluster" is **issue rate**, NOT
completion. V21 had no fence between stores and clock64 end. Real
delivery rate may be lower. V53 in C.1 settles.
**Source:** §D.5.6; DSMEM_DOUBT_REPORT.

### D.4 DSMEM "no shared bus"

⚠ "No shared bus" claim is unprovable from V17. V17 ring test was
**30× under-issued** (1 thread/CTA, single-issue chained). Demote to
"consistent with point-to-point per architecture; not proven".
**Source:** §D.5.6; DSMEM_DOUBT_REPORT.

### D.5 DSMEM 7.5× latency

⚠ DSMEM is 7.5× SLOWER than local SMEM (latency). Don't conflate
DSMEM with SMEM in cluster benchmarks.
**Source:** §D.5.6; V12/V15/V16 cross-test.

---

## E.E — Errors / DCE / LICM

### E.1 Dead Code Elimination (DCE)

⚠ Compiler will eliminate any loop whose output isn't used. Signs:
measured time doesn't scale with iters; BW > theoretical; runtime
0.001 ms.
**Defenses:** unconditional STG of result, runtime-input-derived loop
values, kernel runtime ≥ 1 ms.
**Source:** §61.2; §63.1.

### E.2 LICM (Loop-Invariant Code Motion)

⚠ Compiler hoists loop-invariant work out of the inner loop. Symptom:
sub-linear timing scaling, pipe metric inconsistent with pattern.
**Defense:** make operation inputs depend on loop counter.
**Source:** §63.2.

### E.3 Self-op chains inflate latency 2×

⚠ `fma a, a, a, a` (single register Rd referenced as both source and
destination) creates RF port dependency. Latency inflates by 1 cy.
**Defense:** distinct sources `fma d, a, b, c`.
**Source:** §63.3; CLAUDE.md §3.

### E.4 3-source FFMA RF port pressure

⚠ 3-source FFMA caps at ~50 TFLOPS = 65 % of 2-source peak (75 TFLOPS).
Use 2-source pattern for peak measurements.
**Source:** A4 + D6 + V10_FMA_SOURCE_COUNT.

### E.5 `#pragma unroll 1` doesn't always honor

⚠ `#pragma unroll 1` may be ignored if the compiler estimates a
better cache. Always SASS-verify the unroll factor took.
**Source:** CLAUDE memory `feedback_b300_pitfalls`.

---

## E.F — Fences / sync

### F.1 `__threadfence_system` cost

⚠ 1.74× spread across catalog (1750 / 2870 / 3042 cy). TRUE_REFERENCE
picked 1750 / 861 ns without justification. UNRESOLVED until V54 in
C.2.
**Source:** §D.5; CROSS_AGENT_DOUBT_LOG #2.

### F.2 `__threadfence` (GPU) cost

⚠ 24 % spread (258 / 281 / 292 / 320 cy). Sync agent says "281 ± 25";
DSMEM picks 320 silently. UNRESOLVED until single-pass ncu.
**Source:** CROSS_AGENT_DOUBT_LOG #2.

### F.3 `membar.cta` (block fence)

⚠ ~6 cy (F6). If V54 differs by > 2× from F6, revisit F6 too.
**Source:** §C.2.6.

---

## E.G — Graphs / launch / coordination

### G.1 cudaGraph single-node speedup

⚠ "cudaGraph always faster than direct launch" is WRONG. 1-node
graph = 2.05 µs ≈ direct 2.06 µs (NO speedup). Only batch ≥ 10
kernels per launch yields speedup.
**Source:** §57.2; §63.9; V9_GRAPH_LAUNCH.md.

### G.2 cudaGraph capture for cuBLAS

⚠ "cuBLAS needs cudaGraph for sustained measurements" is for
*measurement isolation*, NOT runtime perf gain. State which framing.
**Source:** §57.7; CLAUDE memory `feedback_b300_pitfalls`.

### G.3 Cooperative launch +32 ns

⚠ "Cooperative launch overhead = +32 ns" was retired in original 10
catalog. The +32 ns is grid-sync setup, not launch-side.
**Source:** §57.7.

### G.4 cuStreamWriteValue32 framing

⚠ 0.45 µs is HOST CALL ONLY. Full producer-consumer pair with
`cuStreamWaitValue32` on a second stream = 2.47 µs. State which.
**Source:** §57.7; UNRESOLVED in 10_launch_overhead_CORRECTED §F.

### G.5 "2.05 µs invariant launch latency as HW property"

⚠ RETIRED as event-floor artifact. The 2.05 µs floor is
`cudaEventRecord` overhead, not kernel-launch latency.
**Source:** §57.7.

### G.6 BlockingSync 5–7× slower

⚠ True for single-thread CPU pinned in older drivers. On B300/CUDA
13.2 the gap is 25 % steady-state. With 4+ host threads, BlockingSync
wakeup latency dominates and the comparison flips.
**Source:** §57.7.

### G.7 cudaGraphLaunch from device code

⚠ Device-side `cudaGraphLaunch` measured 13.7 µs. Whether this stacks
with cluster launch overhead is untested. Measure before publishing
device-side graph claims.
**Source:** §57.7.

### G.8 Persistent kernel "38 ns/task"

⚠ From V7 memory, not re-verified in V8/V9 cycle. Treat as MED until
re-anchored.
**Source:** §57.7.

---

## E.H — Hashes / task lists

### H.1 CURIOSITY_LIST_V2 hash hallucination

⚠ V2 has 22/25 hashes hallucinated (88 %). Author filled in
plausible-looking hashes from memory without verifying. V4-V8 are
100 % git-verified.
**Defense:** always `git rev-parse --short=7 <hash>` AND `git log
--oneline -1 <hash>` before citing.
**Source:** §62 rule 13; CURIOSITY_LISTS_AUDIT.md.

### H.2 V4 "Commit history" placeholders

⚠ V4 has 32/135 items citing "Commit history" instead of a hash.
These are UNVERIFIED. Could be cross-referenced via topic search if
needed; prevalence (24 %) is itself a reliability concern.
**Source:** CURIOSITY_LISTS_AUDIT V4.

### H.3 CLAUDE.md memory unauthoritative

⚠ CLAUDE.md memory has at least 4 retracted entries (NVLink v7, HBM
8 TB/s, SMEM atomic 4.2 T, K-96 10.8 PF). Never trust CLAUDE memory as
authoritative.
**Source:** §D.6; MASTER_INDEX_v2 §4 rule 18.

---

## E.I — Init / kernel args

### I.1 init/main kernel arg conflict

⚠ QuickRunCUDA passes same `-0/-1/-2` to both init and main kernel.
If init reuses arg slot 0 as `iters` for main, init may corrupt or
under-use it. Workaround: pack init params via bit-shift; reserve
`-0` for main kernel.
**Source:** §59.4; CLAUDE memory `feedback_compute_pipe_methodology`.

---

## E.K — Clock state

### K.1 `-lgc 2032` paradox

⚠ `nvidia-smi -lgc 2032` paradoxically pins to 1920 MHz (base clock),
NOT 2032. For boost: do NOT lock; let default boost engage. For 1920
reproducibility: use `-lgc 1920`.
**Source:** §63.8; CLAUDE.md §2.

### K.2 NVML lock doesn't reset

⚠ Clock lock from NVML (`nvmlDeviceSetGpuLockedClocks`) does NOT
reset on process exit. You can leave the GPU locked across sessions.
Always pair `Set` with deferred `Reset` or `nvidia-smi -rgc` between
runs.
**Source:** §60.8.

### K.3 Stuck at 1005 MHz

⚠ B300 can be stuck at 1005 MHz with NO explicit lock. `nvidia-smi
-q` won't show as "locked". Sample `nvmlDeviceGetClockInfo` during
run; if non-2032 when expecting boost, run `nvidia-smi -rgc`.
**Source:** §60.6; CLAUDE memory `feedback_clock_stuck_no_lock`.

### K.4 Default clock state ambiguity

⚠ Default behavior under load is to boost to 2032 MHz; but background
processes can prevent boost. Always sample clock during the
benchmark; state which clock state in your published number.
**Source:** §61.7.

---

## E.L — Library / API

### L.1 cuLibrary 6.5× speedup

⚠ "`cuLibrary*` 6.5× faster than `cuModule*`" is from a single older
catalog line, not re-verified. Re-measure on current driver before
publishing as cold-start optimization.
**Source:** §59.5.

### L.2 NVRTC PTX acceptance

⚠ NVRTC > ptxas only for `tcgen05.*` PTX. Both reject
`cvt.rn.satfinite.e2m1x4.f32` (PTX 8.7 needed). Don't assume NVRTC
solves all narrow-cvt bugs.
**Source:** §59.3.

### L.3 NVRTC `--use_fast_math` always on

⚠ QuickRunCUDA sets `--use_fast_math` in `cuda_helper.h:227`. Every
FFMA emits as `FFMA.FTZ`. Patch out before subnormal-handling tests.
**Source:** §59.2; CLAUDE memory `feedback_nvrtc_fast_math_ftz`.

### L.4 NVRTC `-O0..-O3` rejection

⚠ NVRTC rejects bare `-O0..-O3`. Use `--ptxas-options="-O3"` for
ptxas opts.
**Source:** §59.7.

---

## E.M — Multicast / TMA

### M.1 TMA + prefetch.L2 = −27 %

⚠ V42: TMA + `prefetch.L2` = −27 % BW. Never combine bulk TMA with
explicit prefetch. TMA already owns its own DMA path; explicit
prefetch instructions block forward progress.
**Source:** §56.6; V42 in V41_V48_FINDINGS.md.

### M.2 V6 I3 "prefetch.L2 = 1.58×" applies only to LDGSTS

⚠ V6 I3's "prefetch.L2 = 1.58× speedup" applies to OLD `cp.async`
(LDGSTS), NOT to `cp.async.bulk` / TMA. Do not propagate to TMA.
**Source:** §56.6; 09_memory_apis_CORRECTED.

### M.3 TMA write pipelining null

⚠ Pipelining TMA writes gives ZERO benefit, slightly hurts (V47 6.34
vs V34 7.17). Single multicast engine per cluster.
**Source:** §56.2; §56.6; V47.

### M.4 Multicast can't be pipelined

⚠ Single multicast engine per cluster. Cluster=8 single-deep =
ceiling 14.9 TB/s effective (V32 = V48). Adding 2-deep hurts slightly.
**Source:** §56.2; V48.

### M.5 TMA single-deep 6.72 TB/s "SoL"

⚠ V33 single-deep = 6.72 TB/s; SUPERSEDED by V46 8-deep = 7.20 TB/s.
The architectural lesson "TMA reads need pipelining" holds.
**Source:** §56.1; 09_memory_apis_CORRECTED retraction.

### M.6 TMA at cluster < 8

⚠ TMA multicast reads with cluster < 8 not measured (V32/V48 only ran
cluster=8). Cluster=4 multicast may behave differently (different GPC
topology).
**Source:** §56.6.

### M.7 TMA `wait_group(N)` vs `wait_all`

⚠ Flagged as deferred in V9_CP_ASYNC_BW.md, never resolved. If you
build on `wait_group` at depth N, validate via ncu that the in-flight
count is correct.
**Source:** §56.6.

---

## E.N — NVFP4

### N.1 NVFP4 K=96 cuBLAS 13.4 = 10.8 PF

⚠ "10.8 PF (72 % of 15 PF spec)" is from CLAUDE memory
`project_b300_nvfp4_k96_ceiling`. SUPERSEDED by NVFP4 agent's 11.42 PF
(76.2 % via cuBLAS + cudaGraph BPG=16). Single-shape, single-BPG; treat
as upper-bound at 8K² K=38400.
**Source:** §D.5.9; NVFP4_DOUBT_REPORT #2.

### N.2 NVFP4 A:B asymmetry "single mechanism"

⚠ Wave-2 over-resolved A:B asymmetry to "TMA multicast halves B's
memory cost". Underlying source itself lists 4 plausible mechanisms.
Preserve all 3 readings (cuBLAS A>B 3:1, pure-tcgen05 B>>A 15-30×,
K=96 single-kernel B>A 2.6×).
**Source:** §D.5; NVFP4_DOUBT_REPORT.

### N.3 NVFP4 K-id speedup "always 1.40×"

⚠ K-id speedup ONLY at N ∈ {K, 2K, K/2}, NOT a kernel switch. Real
ML inference (N/K=2.5-3.5) is outside the window so practical benefit
is 2-6 %.
**Source:** CLAUDE memory `project_kid_speedup_shape_dependent`.

### N.4 NVFP4 cvt e2m1x4 PTX bug

⚠ CUDA 13.2 NVRTC AND ptxas BOTH reject `cvt.rn.satfinite.e2m1x4.f32`
on sm_103a. Migrate to PTX 8.7 forms.
**Source:** §59.3; V41_V48 lines 61-62.

---

## E.O — One-pipe-form-failure ≠ HW absent

### O.1 One PTX path fail

⚠ If one PTX form's compile fails, that's NOT evidence the HW
capability is absent. Check all paths (mma.sync vs tcgen05.mma, etc.).
**Source:** CLAUDE memory `feedback_careful_claims`.

---

## E.P — Pipe overlap / dual-issue

### P.1 V49/V50 "55 %/74 % dispatch cap"

⚠ RETRACTED. V52 ncu shows `pipe_alu + pipe_fma = 147 %`
simultaneously. The "55 %/74 %" were loop-overhead methodology
artifacts. The architectural truth: pipes overlap freely.
**Source:** §D.5.4; V52_RUN_RESULTS.md; Appendix A.

### P.2 "B300 dispatch capped at 128 inst/SM/cy"

⚠ Single-pipe IS capped at 128/SM/cy (= 1/SMSP/cy × 4 SMSPs × 32
lanes), but the FMA + ALU pipes can BOTH be at 128 simultaneously, so
total inst/SM/cy can reach ~256. The "128 ceiling" is per-pipe, not
per-SM.
**Source:** §A.6.6; V52.

### P.3 LOP3 issue cadence

⚠ LOP3 has 2-cycle issue cadence per SMSP (`inst_issued/cy = 0.51`
for solo LOP3). Solo LOP3 throughput = ½× solo FFMA, but dual mode
LOP3 piggy-backs in idle ALU slots = pipes overlap freely.
**Source:** §A.6.5; V52.

### P.4 "Same-warp dual-issue can never reach 100 %"

⚠ V52 hits `alu+fma = 147 %` same-warp dual-issue. The "100 % cap"
hypothesis was wrong.
**Source:** §A.7 lesson 1.

### P.5 ncu pipe metric semantic risk

⚠ `smsp__pipe_X_cycles_active` semantic not verified against
PTX-level event counters. If ncu has a metric-definition bug for
sm_103a, V52's interpretation could be wrong (preserved doubt). No
expected; non-zero risk.
**Source:** §A.11; META_LESSONS.md "What could overturn V52".

### P.6 IADD3 pipe placement

⚠ All four agents agree: IADD3 lives on the **FMA pipe** (V40 commit
d1d09c5). Older "ALU pipe" framing is RETRACTED.
**Source:** CROSS_AGENT_DOUBT_LOG #6.

### P.7 PRMT pipe placement

⚠ V40 says PRMT 13.9 Glane/s = 36 % = "permute pipe"; A6 says PRMT
14.08 = 0.5/SMSP/cy = same tier as LOP3 (INT-bit). UNRESOLVED;
needs A6-style sweep on PRMT.
**Source:** CROSS_AGENT_DOUBT_LOG #14.

---

## E.Q — Quick checks / sanity

### Q.1 "Too good to be true" patterns

⚠ Specific too-good signs: BW > theoretical; TFLOPS > theoretical;
latency < HW unit minimum; "same-warp dual-issue" > 100 % gain;
"multicast pipelined deeper than 1 stage helps"; "cudaGraph
single-node speedup". Each has been claimed and retracted.
**Source:** §63.10.

### Q.2 Pair Gops/s with bytes/s

⚠ Cache-line combining can inflate Gops 8× without proportional BW.
Always pair throughput with traffic.
**Source:** §61.5; CLAUDE memory `feedback_units_sanity`.

### Q.3 Sub-agent critique

⚠ Sub-agent outputs are NOT authoritative without verification. Common
failure modes: agent presents formula as measurement; agent uses wrong
constants (e.g., "256 cores/SM" when B300 has 128); agent trusts
compiler-emitted code without SASS verification; agent runs test too
short.
**Source:** CLAUDE.md §6.

---

## E.R — Resources / contention

### R.1 Leftover process contamination

⚠ Unkilled benchmark processes inflate ncu cy/MMA up to **8.5×**.
Always `pkill -9 <bench-name>` and `sleep 5-8` between runs.
**Source:** §63.6; CLAUDE memory `feedback_b300_pitfalls`.

### R.2 TCGEN05_PERF_WATTS single-trial

⚠ Single-trial table contaminated (5 leftover QuickRunCUDA processes).
Use `TCGEN05_PERFW_CLEAN_2TRIAL.md`. NVFP4 K=96 numbers shifted by 2.4
TF/W after cleanup.
**Source:** §63.6; TCGEN05_POWER_CONSOLIDATED §2.

---

## E.S — SASS / instruction encoding

### S.1 `#pragma unroll N` doesn't guarantee SASS unroll

⚠ Compiler may re-roll if it estimates better cache. Always
SASS-verify the unroll factor took.
**Source:** §61.1; §B.7.

### S.2 Inline asm SASS divergence

⚠ Source-level `fma %0, %0, %1, %0` may compile to different SASS
encoding than expected (e.g., `FFMA Rd, Rd, R0.reuse, 0.5` if compiler
hoists immediate into R0). SASS-verify.
**Source:** §61.1; SASS_VERIFY_DUAL_ISSUE.md.

### S.3 Loop overhead < 64 ops/type contaminates dual-issue

⚠ Inner body < 64 ops/type leaks UIADD3 + UISETP + BRA into the ALU
pipe being measured. V49's 8-deep had 12.5 % loop overhead; V8/V52's
128-deep has 1.2 %.
**Source:** §62 rule 11; §B.11.

---

## E.T — Tensor / cuBLAS

### T.1 `pipe_tensor.cycles_active` does NOT measure tcgen05

⚠ ncu `sm__pipe_tensor_cycles_active` measures LEGACY mma.sync, NOT
tcgen05 on sm_103a. Use the explicit metric:
`smsp__inst_executed_pipe_tensor_subpipe_hmma_op_utchmma_utcqmma_utcomma_scope_2cta.sum`
**Source:** §63.7; TENSOR log §B.

### T.2 mma.sync vs tcgen05 spec

⚠ mma.sync legacy: 540-580 TFLOPS BF16 max; tcgen05 Blackwell: 1980
TFLOPS BF16 max. Cite the specific PTX form.
**Source:** §62 rule 12 / B.12.2.

### T.3 NVFP4 spec 15 PF dense / 30 PF sparse

⚠ NVIDIA quotes peak "with sparsity" for tensor ops. Dense is 2× less.
Check context.
**Source:** CLAUDE.md §7.

---

## E.U — Units / formatting

### U.1 Number formatting

⚠ Don't insert thin/narrow spaces as thousands separators in numeric
values. Use plain digits.
**Source:** CLAUDE memory `feedback_number_formatting`.

### U.2 Per-cluster vs chip-aggregate scope

⚠ DSMEM v1's "3.06 TB/s aggregate" was actually per-cluster × ~74 = 3
TB/s **chip aggregate**. Internally consistent with v2's per-cluster
40 GB/s. The "mixed measurements" framing was wrong; the disagreement
was scope.
**Source:** SYNTHESIS_DOUBT H4.

### U.3 Atomic Gops/s vs bytes/s

⚠ L2 atomic units pack within a cache line. State stride and unroll
explicitly. Got "28× ratio" wrong by mixing combined+uncombined
atomics.
**Source:** §61.5.

---

## E.V — V-numbered specific

### V.1 V8 NEW DSMEM `71934d0`

⚠ "Cluster DSMEM BW = 37 TB/s = 97 % of 38.5 peak" RETRACTED. SASS
showed LICM/CSE; real ~40 GB/s/cluster.
**Source:** §D.5.6; CURIOSITY V8 NEW DSMEM.

### V.2 V8 NEW MUFU `29b9b3b`

⚠ "MUFU rsqrt = 99.49 % XU pipe (47.8 GMUFU/s)" RETRACTED LABEL. Real
saturated MUFU = 4.74 G/chip; 47.8 G is 1-chain latency-bound.
**Source:** §D.5.7; MATH_INCONSISTENCY_LOG #3.

### V.3 V8 F1 NVLink `88ee0cf`

⚠ "P2P NVLink memcpy = 778 GB/s = 86 % of NVLink v7" RETRACTED
DENOMINATOR. NVLink is **v5** not v7; spec 900 GB/s/dir not 757.
Correct framing: 86 % of 900 GB/s/dir.
**Source:** §D.5.1; CURIOSITY V8 F1.

### V.4 V49 / V50 dual-issue

⚠ See §E.P.1.

### V.5 V51 multistream HBM

⚠ V51 (`tests/standalone/v51_multistream_hbm.cu`) has CRITICAL
wrong-pointer UB — passes `(const float*)d_src` (host stack address of
pointer-array) instead of `d_src[s]`. RECOMMEND DELETE; the question
is architecturally trivial.
**Source:** V51_INVESTIGATION.md.

---

## E.W — Working set / cache

### W.1 L2 = 96 MB

⚠ "L2 = 96 MB" is WRONG (cosmetic transcription from H100 specs in 4
catalog files). Real L2 = 132 MiB nominal / 126 MB practical.
**Source:** §60.4; STRAYS_CORRECTED §7.

### W.2 L1 BW 30.5 vs 46 TB/s

⚠ L1 BW (default ld, 8-ILP × 16 unroll) = 30.5 TB/s (V8) vs M5
cheatsheet 46 TB/s. UNRESOLVED.
**Source:** CONFIDENCE_LADDER §2.

### W.3 L2 BW metric tagging

⚠ Always specify L2 BW metric: kernel-effective ~24 / wire-lts ~13 /
L1-amplified ~30 TB/s. Bare numbers float.
**Source:** §62 rule 12; CROSS_AGENT_DOUBT_LOG #4.

---

## E.X — Cross-agent / cross-doc

### X.1 Wave-2 took credit for upstream retractions

⚠ BF16 1543 / FP8 7500-8200 / BF16 90.5 % were already self-retracted
by their original docs. Wave-2 synthesis took credit for retractions
made before it.
**Source:** SYNTHESIS_DOUBT H5; HEADLINE_v2.

### X.2 M-synthesis flattens contradictions

⚠ M-synthesis docs flatten cross-agent contradictions in the
dual-issue / V46 cases by picking one side without flagging the other.
**Source:** CROSS_AGENT_DOUBT_LOG patterns.

### X.3 Single-shape NVFP4 ceiling

⚠ NVFP4 11.42 PF cuBLAS+graph is **single shape, single BPG** (no
sweep). Frame as "upper-bound at 8K² K=38400 BPG=16; sustained ceiling
needs sweep".
**Source:** NVFP4_DOUBT_REPORT #2.

### X.4 Persistent kernel "v1 used release"

⚠ "v1's 4 µs vs v2's 2.03 µs because v1 used release variant" is a
HYPOTHESIS, no SASS evidence cited. Demote to "hypothesis: v1 likely
used release; not verified".
**Source:** DOUBT_LOG §3 row 10.

---

## E.Y — Yield-fused SKU specifics

### Y.1 7680-bit fused bus

⚠ This box is NVIDIA B300 SXM6 **AC** SKU. Bus width 7680 = 8192 ×
15/16 (one /16 controller fused off). All "% of HBM peak" math must
account for this. Other (non-AC) SKUs may have full 8192-bit bus.
**Source:** §60.2; HBM_STACKS_INDEPENDENT_VERIFY.md.

### Y.2 288 GB capacity

⚠ totalGlobalMem = 275040 MiB ≈ 287.4 GB practical / 288 GB marketed.
Both numbers appear; cite both for clarity.
**Source:** §60.1.

---

## E.Z — Zero-cases / edge

### Z.1 Empty / zero-data benchmarks

⚠ Constant-zero data does NOT exercise toggle-energy curve.
HBM_DATA_DEPENDENCE.md's "<50 W" is from constant-pattern only;
RETRACTED. Use random data with controlled popcount.
**Source:** §63.11; POPCOUNT_3TIER.

### Z.2 Tiny kernel runtime

⚠ Kernel runtime < 100 µs has > 10 % launch overhead. For peak
throughput, ensure runtime ≥ 10 ms. For latency, use clock64 inside
the kernel to exclude launch.
**Source:** §63.4.

### Z.3 Anti-DCE conditional that compiler can prove false

⚠ "if (tid == 0) STG ..." can be eliminated when the compiler proves
the condition is impossible-but-reachable. Use unconditional STG of
accumulator under impossible-but-not-provable condition (e.g., `if
(acc != 0xdeadbeef) STG`).
**Source:** §61.2.

---

## E.* — Quick lookup by symptom

For symptoms-first lookup (when you see a number that looks wrong and
need to find the relevant footgun):

| Symptom | Likely footgun |
|---|---|
| BW > theoretical | E.E.1 DCE |
| BW around 1.5-3× theoretical | E.E.2 LICM |
| TFLOPS > theoretical | E.E.1 DCE; E.B.3 8 TB/s denominator |
| Latency < HW unit minimum | E.G.5 event-floor artifact; E.E.1 DCE |
| Wall-clock disagrees with ncu | E.E.1 DCE; E.S.1 unroll |
| Same kernel different days | E.K.1-K.4 clock state; E.R.1 leftover proc |
| Atomic Gops/s seems high | E.A.3 cache-line combining |
| dual-issue claim | E.P.1-P.7 entire dual-issue family |
| HBM % discrepancy across docs | E.B.2 denominator drift |
| "12 stacks HBM" cited | E.B.1 should be 8 of 12-Hi |
| "NVLink v7" cited | E.D.5 / V.3 — should be NVLink-5 |
| "10 GPCs" cited | E.C.7 — should be 8 |
| L2 = 96 MB cited | E.W.1 — should be 126 MB |
| SMEM atomic 4.2 T cited | E.A.1 — should be 2.27 T |
| K=96 10.8 PF cited | E.N.1 — should be 11.42 PF (or stale) |
| MUFU 47.8 G "XU peak" | E.V.2 — relabel as 1-chain latency |
| ncu pipe_tensor for tcgen05 | E.T.1 — wrong metric |
| cudaGraph "always faster" | E.G.1 — wrong; only batch ≥ 10 |
| TMA + prefetch.L2 | E.M.1 — never combine |
| Multicast pipelined | E.M.4 — single engine, can't deepen |
| `-lgc 2032` not boost | E.K.1 — paradoxically pins to 1920 |
| "37 TB/s DSMEM" | E.D.1 — DCE artifact, real ~40 GB/s |
| "98.5 % NEW HBM SoL" | E.B.4 — denominator artifact |
| "55 % dual-issue cap" | E.P.1 — methodology artifact |

---

## E.End

This index is exhaustive as of 2026-04-22 (wave 6 + canonical
synthesis). For new footguns discovered after this date, append to the
appropriate section here AND update the symptom lookup table.

