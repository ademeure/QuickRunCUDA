# B300 Sync Primitives — Clean Reference

**Platform**: NVIDIA B300 SXM6 AC, 148 SMs, sm_103a, CUDA 13.2
**Clock convention**: 2.032 GHz default boost (= 0.4921 ns/cy). Where catalog data was at 1.92 GHz (locked) it has been rescaled.
**Date**: 2026-04-17

---

## Latency Ladder (sorted, single-warp / single-CTA, isolated cost)

| Primitive | cy | ns @ 2032 MHz | SASS-verified | Source |
|---|---:|---:|:-:|---|
| `__shfl_sync` broadcast (idx=0) | ~2 | **0.85** | UIMOV/R2UR (uniform fast-path) | syncwarp.cu, sec 16 |
| `__syncwarp` / `bar.warp.sync` | 2.8–6 | **1.0–3.0** | WARPSYNC | sync_cost.cu, sec 26/30.D |
| `membar.cta` / `fence.acq_rel.cta` | 9 | **4.4** | MEMBAR.ALL.CTA | sync_cost.cu, sec 16/30 |
| `__threadfence_block` | ~16 | **8** | MEMBAR.ALL.CTA | fence_cost.cu |
| `__syncthreads` (256 thr) | 28 | **14** | BAR.SYNC.DEFER | mbar_vs_syncthreads.cu |
| `__syncthreads` (512 thr) | 45 | 22 | BAR.SYNC.DEFER | sec 16 |
| `__syncthreads` (1024 thr) | 77 | **38** | BAR.SYNC.DEFER | sec 30.2, mbar_vs_syncthreads.cu |
| `mbarrier.arrive + test_wait` (32 thr, count=1) | 54 | 26 | SYNCS.ARRIVE + SYNCS.PHASECHK | sec 30.2 |
| **`barrier.cluster.arrive.relaxed + wait`** (cluster=2) | **102** | **50** | UCGABAR_ARV/WAIT + CCTL.IVALL (NO MEMBAR.ALL.GPU) | cluster_raw_barrier.cu, sec 30.E |
| `cluster.sync()` / `barrier.cluster.{arrive,wait}.aligned` | 373–380 | **184–187** | UCGABAR_ARV + MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR | cluster_sass_audit.cu, sec 30.E |
| `__threadfence` / `membar.gl` / `fence.sc.gpu` | 277–292 | **136–144** | MEMBAR.SC.GPU + ERRBAR + CGAERRBAR + CCTL.IVALL | fence_cost.cu, sec 30.B |
| `__threadfence` (with chip-wide write traffic, drain) | 783 | **385** | (drain dominates) | EXTENDED §1, sec 30.G |
| Cross-block flag wait, one-way (volatile spin + threadfence) | 1605 | **790** | — | pingpong.cu |
| `__threadfence_system` / `membar.sys` | 1750 | **861** | MEMBAR.SC.SYS + ERRBAR + CGAERRBAR + CCTL.IVALL | fence_cost.cu |
| `fence.sc.sys` (saturated chip + 16 writers) | ~19000 | ~9300 | — | sec 30.G |

---

## Headline Findings (validated)

1. **`barrier.cluster.arrive.relaxed + wait` is 3.7× faster than `cluster.sync()`** — 102 vs 373 cy. SASS confirms: relaxed emits `UCGABAR_ARV/WAIT + CCTL.IVALL`; the strict cluster.sync adds `MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR`. Use relaxed when you don't need release/acquire — it saves ~270 cy / ~135 ns per cluster sync. (Validated: cluster_raw_barrier.cu, B300_PIPE_CATALOG sec 30.E; the often-cited "3.9×" rounds to 4× — exact ratio is 3.66×.)
2. **`cluster.sync()` cost is constant for cluster size 2 → 8** (all ≈ 175-190 ns). Cluster size does not scale the barrier; cost is fixed by the GPU-fence component, not the barrier-arrival count. (cluster_sync_size.cu, EXTENDED §9.)
3. **`cg::reduce(warp, x, plus<unsigned>())` lowers to `__reduce_add_sync` → REDUX.SUM** SASS, identical cost (154 vs 155 cy with control-flow). It is **3.6× faster than a hand-rolled 5-step shfl_xor tree** (215 cy with overhead) and **3.2× faster than shfl-tree** in pure throughput (3169 vs 986 GOps/s). (cg_reduce.cu, cg_sass_verify.cu, sec 26/29.)
4. **`redux.sync` FP variants are NOT supported on sm_103a** — `redux.sync.add.f32` and `redux.sync.min.f32` both fail at runtime. Hardware support is **integer only** (u32/s32/b32 add/min/max/and/or/xor). FP reductions must use shfl trees. (redux_fp_probe.cu, EXTENDED §10, B300_REFERENCE finding 11.)
5. **`redux.sync.min.u32` (CREDUX.MIN) is 2.2× faster than `redux.sync.add.u32` (REDUX.SUM)** — 6923 vs 3107 GOps/s, 18 vs 44 cy latency. Different SASS pipes: CREDUX runs on alu+fmaheavy (1.92 PTX-op/SM/cy); REDUX runs on adu (0.50). Min/max should be preferred when semantics allow. (redux_sync.cu, sec 26/29.)
6. **`mbarrier.arrive + test_wait` block-wide is 3-9× SLOWER than `__syncthreads()`** for plain block-wide barriers (262 vs 89 cy at 1024 threads). mbarrier wins only when async/transaction (`expect_tx`) semantics are needed. (mbar_vs_syncthreads.cu, sec 30.2.)
7. **`cg::tiled_partition<N>` sub-warp shuffles cost 3-22× more than raw `__shfl_xor_sync`** — tile<16> = 57 cy/shfl vs raw warp = 6 cy/shfl. The cg API adds masking/bounds overhead per call. Use raw shuffles in perf-critical paths. (cg_partition.cu, B300_PIPE_CATALOG line 11059-11079.)
8. **`cg::coalesced_threads()` is 5× slower than full-warp reduce** (183 vs 37 cy) due to dynamic activemask group construction. Avoid in hot loops. (sec line 10743.)
9. **Broadcast `__shfl_sync(mask, v, 0)` is essentially free (~1 cy)** — compiler recognises constant index 0 and lowers to UIMOV through the uniform pipe, NOT through SHFL. Same trick does not apply to `idx != 0`. (syncwarp.cu, sec 16 line 1376.)

---

## SASS Verification Receipts

| Primitive | SASS emitted (verified by cuobjdump) |
|---|---|
| `cluster.sync()` (cg::this_cluster().sync()) | `UCGABAR_ARV` + `UCGABAR_WAIT` + `MEMBAR.ALL.GPU` + `ERRBAR` + `CGAERRBAR` |
| `barrier.cluster.arrive.relaxed.aligned` + `barrier.cluster.wait.aligned` (PTX) | `UCGABAR_ARV` + `UCGABAR_WAIT` + `CCTL.IVALL` (no MEMBAR.ALL.GPU) |
| `cg::reduce(warp, x, plus<unsigned>())` | `REDUX.SUM` (single SASS instruction on adu pipe) |
| `cg::reduce(warp, x, less<unsigned>())` (min) | `CREDUX.MIN` + `IMAD.U32` (alu + fmaheavy intrinsic-coupled) |
| `__syncthreads()` | `BAR.SYNC.DEFER` (single SASS, 45 cy at 512 threads) |
| `__syncwarp()` | `WARPSYNC` (~3 cy when mask is `0xffffffff` compile-time) |
| `mbarrier.arrive.shared.b64` | `SYNCS.ARRIVE.TRANS64.A1T0` (Blackwell-new SYNCS family on adu) |
| `mbarrier.test_wait.shared.b64` | `SYNCS.PHASECHK.TRANS64` + `SEL` |
| `__threadfence()` / `membar.gl` | `MEMBAR.SC.GPU` + `ERRBAR` + `CGAERRBAR` + `CCTL.IVALL` (4 inst) |
| `__threadfence_system()` / `membar.sys` | `MEMBAR.SC.SYS` + `ERRBAR` + `CGAERRBAR` + `CCTL.IVALL` (4 inst) |

(Sources: cluster_sass_audit.cu commit 7e94d2b, cluster_raw_barrier.cu commit d8ca01a, cg_sass_verify.cu commit 3040cd5, B300_PIPE_CATALOG sec 16/30.B/30.E.)

---

## Confidence Markers

| Finding | Reliability | Reason |
|---|---|---|
| Latency-ladder ordering (warp < block < cluster-relaxed < cluster < device < system) | **HIGH** | Reproduced across 4+ tests (sync_cost, sync_methods, mbar_vs_syncthreads, fence_cost) with consistent ordering |
| Absolute warp/block/fence cycle counts (±10%) | **HIGH** | clock64-bracketed, multiple iter counts, consistent across runs |
| Cluster.sync 175-190 ns invariant in size 2-8 | **MEDIUM-HIGH** | cluster_sync_size.cu measured 2/4/8; size 16 requires opt-in |
| `barrier.cluster.arrive.relaxed + wait` 3.7× speedup | **HIGH** | Direct A/B comparison in cluster_raw_barrier.cu, ratio reproduced in B300_PIPE_CATALOG sec 30.E (102 vs 373 cy) |
| `cg::reduce` lowers to REDUX.SUM HW | **HIGH** | SASS dump (cg_sass_verify.cu) shows the instruction directly |
| `redux.sync` FP not supported on sm_103a | **HIGH** | Empirical: redux_fp_probe.cu observes runtime "illegal instruction" |
| `__threadfence_system` 861 ns | **MEDIUM** | Single-warp isolated; with concurrent writers cost grows to ~9000 ns (see EXTENDED §1) |
| Cross-block flag wait 790 ns one-way | **MEDIUM** | pingpong.cu single-flight; varies with chip occupancy and L2 invalidation pattern |
| `cg::coalesced_threads` 5× slower | **MEDIUM** | Single test; dynamic-mask path may improve in future libcu++ |

---

## Items Retired / Corrected

| Earlier claim | Correction | Source |
|---|---|---|
| `cluster.sync` cost grows with cluster size | RETIRED — constant 175-190 ns across size 2,4,8 | EXTENDED §9, cluster_sync_size.cu |
| `cg::tiled_partition<32>` shuffles == raw warp shuffles | RETIRED — tile<16> is 10× slower; subdividing has overhead | line 11059-11079, cg_partition.cu |
| `redux.sync.add.f32` works on B300 | RETIRED — FP redux is unsupported on sm_103a; only int u32/s32/b32 | redux_fp_probe.cu |
| Catalog cell "0.20 ms" / "0.057 ms" for cluster barrier (sec 16) | KEEP but note: those were chip-wide ms numbers from a 148-CTA test, not per-call ns; per-call is 184 vs 50 ns | cluster_raw_barrier.cu reconciled with sec 30.E |
| AUDIT_NOTES "membar.gl 38 ms" (line 418) | RETIRED — was a confused write-up; actual cost 277-292 cy = 144 ns isolated, 819 cy w/ writes | sec 30.B/30.G |
| CRITIQUE entry "cluster.sync = ~353-381 cy" (CONSOLIDATED line 246) | KEPT, rounds to "175-190 ns" headline | matches sec 30.E/8032 |

---

## Practical Recipes

| Goal | Use | Cost |
|---|---|---|
| Within-warp barrier | `__syncwarp()` | 1.4 ns |
| Within-warp broadcast | `__shfl_sync(0xffffffff, v, 0)` | 0.85 ns (uniform-pipe fast path) |
| Within-warp reduction (sum) | `__reduce_add_sync` or `cg::reduce(warp, x, plus)` | 27 ns (REDUX.SUM HW) |
| Within-warp reduction (min/max) | `__reduce_min_sync` or `cg::reduce(warp, x, less)` | 9 ns (CREDUX HW — fastest) |
| Within-warp FP reduction | shfl-xor 5-step tree (FP REDUX missing) | 105 ns (no HW alternative) |
| Block barrier | `__syncthreads()` | 14-38 ns (256-1024 thr) |
| Block barrier + reduction | `__syncthreads_and/or/count(pred)` | 75 ns (~2× syncthreads, but saves separate reduce pass) |
| Cluster barrier (no ordering needed) | `barrier.cluster.arrive.relaxed.aligned` + wait | **50 ns** (USE THIS by default) |
| Cluster barrier (release/acquire needed) | `cluster.sync()` | 184 ns |
| Block-scope memory fence | `__threadfence_block()` / `membar.cta` | 8 ns |
| Device-scope fence (no traffic) | `__threadfence()` / `fence.sc.gpu` | 144 ns |
| Device-scope fence (chip-wide write traffic) | same | 385-790 ns (drain dominates) |
| System-scope fence | `__threadfence_system()` | 861 ns (avoid in hot loops) |
| Cross-block signal (one-way) | volatile flag + threadfence | 790 ns floor |
| Async transaction barrier (TMA, etc.) | `mbarrier` (`SYNCS.*`) | 54 cy RTT (1 thr); 117-262 cy block-wide |

**Anti-pattern reminders**:
- `cg::coalesced_threads()` for warp work — 5× slower than `__ballot_sync` + manual mask.
- `cg::tiled_partition<N<32>` shuffles in hot loops — 3-22× slower than raw `__shfl_xor_sync` with explicit mask.
- `mbarrier` for plain block barriers — `__syncthreads` is 3-9× faster.
- Default `cluster.sync()` when relaxed semantics suffice — relaxed barrier is 3.7× faster.
