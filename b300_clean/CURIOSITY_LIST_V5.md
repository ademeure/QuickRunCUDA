# V5 — Newly-curious questions surfaced from V4 deep dive (2026-04-21)

V4 found ~135 things. V5 lists the NEW questions surfaced during that work that
weren't in V3/V4 originally. Generated from gaps + surprises in V4.

Status: `[ ]` unstarted, `[~]` in progress, `[x]` done with commit hash.

---

## A. mbarrier deep ninja

- [ ] **A1 — mbarrier.try_wait timeout** granularity: we passed 0; does negative work? what's max?
- [ ] **A2 — Why is mbarrier voltage-aware?** (per R2 +25% lower power than spin) — measure with ncu power per-pipe metrics
- [ ] **A3 — mbarrier.expect_tx for cp.async** transaction barriers — mature usage pattern
- [ ] **A4 — Multi-mbarrier per CTA** — concurrent barrier objects, do they share resources?
- [ ] **A5 — mbarrier in DSMEM** (cluster-wide barrier across CTAs)
- [ ] **A6 — mbarrier.arrive without wait** — fire-and-forget cost (vs full sync)

## B. HMMA + tcgen05 concurrency

- [ ] **B1 — HMMA + tcgen05.mma simultaneous** — can they both run concurrently? per-SMSP?
- [ ] **B2 — TMEM access vs HMMA** competition — both use tensor pipe?
- [ ] **B3 — tcgen05.cp + mma overlap** — pre-stage A/B then mma
- [ ] **B4 — WGMMA on B300** (pre-Hopper warpgroup mma) — still supported? perf vs mma.sync?
- [ ] **B5 — HMMA chained vs ILP** — at what ILP does HMMA saturate? (we measured mma.sync ILP, not HMMA)
- [ ] **B6 — mma + LDS overlap** — like B2 (LDG) but with SMEM

## C. DSMEM + cluster

- [ ] **C1 — DSMEM (cluster-shared SMEM) latency** vs local SMEM — same TPC vs across-row
- [ ] **C2 — cluster.barrier abuse** — call arrive_drop on N/4 CTAs to simulate subset wait
- [ ] **C3 — Cross-cluster atomics** — atomic on DSMEM from another CTA
- [ ] **C4 — Cluster shared mbarrier** — semantic + perf
- [ ] **C5 — Cluster size 16** (max per spec) — does it actually launch?

## D. Power deep ninja

- [ ] **D1 — Subnormal FFMA without -use_fast_math** — modify harness, measure penalty
- [ ] **D2 — Per-pipe DUTY CYCLE measurement** — when FFMA pipe is 50% utilized, is power 50%?
- [ ] **D3 — TDP throttle behavior** — sustained kernel that pulls peak; does clock drop?
- [ ] **D4 — Dynamic voltage scaling** — does B300 auto-adjust voltage with clock?
- [ ] **D5 — Power-aware code patterns** — confirmed mbarrier (R2); what else? cp.async vs LDG?
- [ ] **D6 — Per-SM power isolation** — can one heavy SM pull power from others?

## E. Bizarre PTX corners

- [ ] **E1 — PTX `prefetchu.L1`** — does B300 honor user prefetch hints?
- [ ] **E2 — PTX `applypriority`** — L2 cache priority hint
- [ ] **E3 — `griddepcontrol`** PTX — programmatic dependent launch
- [ ] **E4 — `bar.warp` semantics** beyond sync — bar.warp.arrive variants?
- [ ] **E5 — `nanosleep`** PTX — sleep duration, accuracy, power
- [ ] **E6 — `getctarank`** semantics in nested contexts

## F. ncu metric exotica

- [ ] **F1 — sm__pipe_*** breakdown — extract per-pipe utilization for any kernel
- [ ] **F2 — sm__warps_active** vs eligible vs issued — measure scheduler state
- [ ] **F3 — Memory throughput sub-metrics** — read vs write, HBM vs L2 vs L1
- [ ] **F4 — Register usage realtime metrics** — verify SASS spill count via ncu
- [ ] **F5 — Cluster-aware metrics** — do cluster CTAs get separate counts?

## G. Real-world micro-kernel SoL

- [ ] **G1 — Q4 fix: SHFL warp scan** working — get to ~250 cy SoL
- [ ] **G2 — SoftMax 1024 elements** at SoL (1 block; FFMA + MUFU + redux)
- [ ] **G3 — LayerNorm 1024 elements** at SoL
- [ ] **G4 — RMSNorm 1024 elements** (closer to fundamental — only var, no mean)
- [ ] **G5 — Argmax 1024 elements** at SoL
- [ ] **G6 — Top-K (K=4) 1024 elements** at SoL (small bitonic + threshold)

## H. Driver / runtime sub-microsecond

- [ ] **H1 — cudaGraphExec_t update perf** — modify args without re-compile
- [ ] **H2 — cudaGraphLaunch with stream re-attach** — minimum launch overhead
- [ ] **H3 — Memory allocator perf** — cudaMallocAsync vs cudaMalloc throughput
- [ ] **H4 — Stream-ordered alloc** — minimum cudaMallocFromPoolAsync overhead
- [ ] **H5 — driver context init** — fresh process to first kernel launch
- [ ] **H6 — Lazy module load** — delay until first use

## I. Multi-GPU patterns

- [ ] **I1 — IPC handle benchmark cross-process** (multi-process binary spawn)
- [ ] **I2 — NVLink streaming WRITE bandwidth** at multi-block parallel
- [ ] **I3 — Multi-GPU all-reduce** at SoL (NCCL-free, custom kernel)
- [ ] **I4 — Cross-GPU atomic via NVLink** — same-stack vs different-GPU
- [ ] **I5 — UVA pointer access from peer GPU** — implicit P2P perf

## J. Numerical precision exotica

- [ ] **J1 — Rounding mode behavior** — fma.rn vs fma.rz vs fma.rm vs fma.rp throughput
- [ ] **J2 — Subnormal preservation** without -ftz (modify harness)
- [ ] **J3 — Mixed precision MMA precision** — measure ULP error of m16n8k16 BF16
- [ ] **J4 — TF32 vs FP32 mma** precision tradeoff
- [ ] **J5 — Block reduction precision** — pairwise vs serial vs Kahan

## K. Async + persistent kernel patterns

- [ ] **K1 — Persistent kernel SoL** — minimum overhead per task in flight
- [ ] **K2 — Producer-consumer via DSMEM** — cluster cooperative pattern
- [ ] **K3 — Dynamic parallelism cost** — cudaLaunchKernel from device
- [ ] **K4 — CUDA Graph capture overhead** — when kernels are captured
- [ ] **K5 — Stream-ordered cooperation** — multiple streams writing to same buffer

## L. Tooling enhancements

- [ ] **L1 — Auto-rigor wrapper** — combines clean_run + ncu_explorer + sass_count
- [ ] **L2 — Power profiler** with libnvidia-ml.so direct (vs CLI 33 Hz limit)
- [ ] **L3 — Per-pipe utilization dashboard** from ncu metrics
- [ ] **L4 — SASS diff visualizer** — between two kernel variants
- [ ] **L5 — Microbench template** generator

---

These are the **next 50+ questions** to dig into. Same rigor protocol as V4:
1. Theoretical first
2. Measured + ratio
3. >1.0× → bug
4. Investigate why if less
5. ncu cross-check
6. SASS verify
7. ≥3 methods reconcile
8. Demonstrate "Y because X"
9. Suspect test before HW
10. HIGH/MED/LOW + falsifiability
