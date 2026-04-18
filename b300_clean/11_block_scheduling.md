# B300 SXM6 — Block Scheduling, Occupancy, GPC Topology, Concurrent Kernels

**Device**: NVIDIA B300 SXM6 AC, sm_103a, 148 SMs (IDs 0..147 dense), 2032 MHz boost
**Date**: 2026-04-17
**Sources**: investigations/09_gpc_count.md, sm_dist.cu, sm_enum.cu, cluster_max{,_size}.cu, cluster_gpc.cu, concurrent_kernel_limit.cu, many_blocks{,_v2}.cu, block_size_sweep.cu, reg_occupancy.cu, sub_gpu.cu, contend_audit{,_v2}.cu, split_sm_workloads.cu, concurrent_tensor_ffma.cu, stream_priority.cu, coop_launch.cu, cluster_overhead.cu

---

## TL;DR

| Topic | Verified value | Confidence |
|---|---|---|
| SM count / IDs | 148, contiguous 0..147 | HIGH |
| GPCs | **8** (NOT 10) — `ncu gpc__cycles_elapsed.sum/avg = 8.0000` | HIGH |
| GPC layout | column-major: SM IDs round-robin across 8 GPCs with stride 16 | HIGH |
| GPC sizes | 2 GPCs × 20 SMs + 6 GPCs × 18 SMs = 148 | HIGH |
| Concurrent kernel slots | **128** (cliff at 144); 1-128 = 1 wave, ≥130 = 2 waves | HIGH |
| Stream priority | bypasses 128-slot queue (high-prio finishes promptly when queue full) | HIGH |
| Stream priority | does NOT preempt mid-kernel; coexists if SMs available | HIGH |
| Max cluster size | 16 (non-portable); 32+ silently no-op (cluster_8 portable) | HIGH |
| Cluster placement | spans MULTIPLE GPCs (not same-GPC) — 8-block cluster spans 4 GPCs | HIGH |
| Cluster launch overhead | ZERO vs regular launch | HIGH |
| Cooperative launch overhead | +32 ns vs plain (essentially free) | HIGH |
| Block size effect on FFMA | NONE: any block size 32..1024 reaches ~73 TFLOPS at full occ | HIGH |
| Block→SM at low count | ≤148: 1/SM perfect; 296: exact 2/SM; 1000+: 6-7/SM imbalance | HIGH |
| Block dispatch tail | per-block runtime <5 us → 5-10% throughput loss (warp gap, NOT scheduler) | HIGH |

---

## Hardware topology

**8 GPCs**, SMs interleaved column-major (stride 16):

| GPC | # SMs | SM IDs (every 16) |
|---|---|---|
| 0 | 20 | 0,1, 16,17, 32,33, 48,49, 64,65, 80,81, 96,97, 112,113, 128,129, 144,145 |
| 1 | 20 | 2,3, 18,19, ..., 130,131, 146,147 |
| 2..7 | 18 each | offsets 4..15 paired |

TPCs 71-73 (SMs 142-147) participate in regular launches and cluster-2, but the hardware excludes them from cluster-4/8 routing (cluster-8 max_active_clusters = 142, not 148).

## Concurrent kernel dispatch

`concurrent_kernel_limit.cu` with 5-ms single-block kernels per stream:
- 1..128 streams: ~3.65 ms wall time (1 dispatch wave, 98% efficient)
- 130, 144, 148, 160, 200 streams: ~7.34 ms (2 waves)
- Cliff is sharp at slot 129; the dispatcher holds 128 in-flight kernels max regardless of SM availability.

**Stream priority bypasses this queue** (`stream_priority.cu`): a high-prio launch behind 130 saturating low-prio kernels finishes in ~1 ms (one wave's worth), not 2 ms. But priority does NOT preempt; once both kernels are resident they share SMs.

## CORRECTED: tensor + FFMA "8x slower" mechanism

Prior catalog claim (commit d723943): "Tensor + FFMA on separate streams 8x SLOWER concurrent." **RETRACTED** by commit d468680 (`contend_audit_v2.cu`).

Per-stream events show:
- GEMM duration: 0.25 ms (UNCHANGED vs alone)
- FFMA duration: 3.58 ms (8x its alone-time)
- ncu `launch__grid_size`: GEMM=2048 blocks, FFMA=148 blocks
- Predicted FFMA share = 148/(148+2048) = 6.7%; measured = 13%

Mechanism is **proportional SM-share scheduling**, not hardware unit contention. Tensor and FFMA pipes can co-issue freely; the smaller kernel just gets dispatched to fewer SMs because the scheduler round-robins block-by-block. **Operational rule**: match block counts (or use stream priority) for balanced co-execution. Confirmed by `split_sm_workloads.cu`: 74-block tensor + 74-block FFMA on separate streams runs without contention.

## CORRECTED: 1M blocks "10% scheduler overhead"

Prior catalog claim: "1M blocks shows 10% scheduler overhead." **RETRACTED** by commit d712c2a (`many_blocks_v2.cu` + ncu).

ncu of 1,000,000-block FFMA case:
- `sm__cycles_active.avg.pct_of_peak_sustained_elapsed`: 99.92%
- `sm__inst_executed.avg.per_cycle_active`: 3.98 / 4.00 max (99.5%)
- `smsp__warps_active.avg.pct_of_peak_sustained_active`: 90.13%

SMs are fully active and at peak issue; the missing 10% is **warp gap** — block-to-block dispatch tail when per-block runtime is small (~4 us). NOT a fixed scheduler overhead.

**Updated rule (HIGH confidence)**: per-block runtime ≥100 us → 100% scheduling efficiency at any block count; <5 us → 5-10% throughput loss from dispatch tail.

## Other measurements

- **Sub-GPU concurrency** (`sub_gpu.cu`): 2 streams × half SMs each = 0.99x baseline; 148 streams × 1 SM each = 0.42x (overhead from extreme stream count).
- **Block size sweep** (`block_size_sweep.cu`, commit 98fdfb3): 32, 64, 128, 256, 512, 1024 threads/block all hit ~73 TFLOPS — block size is irrelevant for FFMA when occupancy is full.
- **Register pressure cliff** (`reg_occupancy.cu`): 96 ILP chains (102 regs/thread) = peak 61.4 TFLOPS; 128 chains (134 regs) drops to 56.3 because occupancy halves.
- **SHMEM carveout occupancy** (cross-cut with SHMEM agent): 8 blocks/SM at ≤16 KB SHMEM/block, halves at each carveout step, 1 block/SM at ≥128 KB.
- **Cluster sync** (any size 2..16): 190 ns constant; cluster launch attribute adds ZERO overhead vs regular.

---

## Retired / superseded claims

| Old claim (location) | Status | Replacement |
|---|---|---|
| "10 GPCs (9×16 + 1×4)" (catalog L7490) | RETIRED — confused stride-16 column-major IDs with contiguous ranges | 8 GPCs (ncu-verified) |
| "1M blocks → 10% scheduler overhead" | RETIRED (commit d712c2a) | warp gap from short per-block runtime |
| "Tensor + FFMA 8x slower concurrent (HW contention)" | RETIRED (commit d468680) | proportional SM-share; smaller-block kernel gets less SM time |
| "Cluster blocks placed within same GPC" | RETIRED (commit 79372e6) | spreads across multiple GPCs (8-block cluster spans 4 GPCs) |
| "Concurrent limit = 148 SMs" | RETIRED (commit 579e4f0) | 128 hardware dispatch slots |
| "Max cluster = 8" portable wording | KEPT for portability; non-portable max = 16 on B300; 32+ silently fails |

## Flag for catalog

The "tensor + FFMA contention" finding is the highest-impact correction here: it changes the user-facing recommendation from "don't co-execute tensor and scalar kernels" to "co-execute is fine, just match block counts (or use priority) so the smaller kernel isn't starved."
