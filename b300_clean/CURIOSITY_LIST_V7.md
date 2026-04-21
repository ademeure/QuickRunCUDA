# V7 Curiosity List (B300 sm_103a) — generated 2026-04-21

After V6 60/65 [x] (M10 synthesis). V7 carries forward V6 deferred items
+ new questions raised during V6 work + brand-new investigation areas.

Apply 10-rule rigor protocol. Mark `[x]` with commit hash when done.

---

## A. tcgen05.mma full implementation (carried from V6 B-series)

- [ ] **A1 — UmmaDesc raw PTX encoder** mirroring cute::SmemDescriptor for SM_103
- [ ] **A2 — Working tcgen05.mma m64n8k16 BF16** with non-zero output verification
- [ ] **A3 — tcgen05.mma + HMMA concurrency** (closes V5 B1)
- [ ] **A4 — tcgen05.mma idesc bit format** documentation (5-arg + 4-tuple format)
- [ ] **A5 — tcgen05.mma latency/throughput** vs HMMA mma.sync at same precision
- [ ] **A6 — tcgen05.mma power signature** (does it match HMMA pJ/op?)
- [ ] **A7 — tcgen05.mma kind=tf32/f8/f16/sparse** all variants

## B. Multicast TMA (carried from V6 I5)

- [ ] **B1 — Working multicast cp.async.bulk.shared::cluster.multicast** with DSMEM mbarrier
- [ ] **B2 — Multicast BW vs unicast** — measure savings for N CTAs receiving same data
- [ ] **B3 — TMA tensor descriptor** (cuTensorMapEncodeTiled-built) for 2D tile copies
- [ ] **B4 — TMA prefetch combo** — does prefetch help TMA like it helps cp.async (V6 I3 1.58×)?

## C. NVFP4 conversion + scale variants (carried from V6 H3)

- [ ] **C1 — cvt.scalefactor variant** for NVFP4 e2m1 with per-block scale
- [ ] **C2 — NVFP4 dequant cost** per scale group (8 elements)
- [ ] **C3 — NVFP4 + tcgen05.mma** end-to-end (depends on A2)

## D. Cluster patterns deeper

- [x] **D1 — Cluster all-reduce 487-537 cy** (commit `36a7e7c`): CSIZE=4 487 cy; CSIZE=8 537 cy. DSMEM peer loads pipeline with barrier. For ML primitives needing cluster-wide aggregation (softmax denominator, layer norm).
- [x] **D2 — DSMEM broadcast 2.4× faster than global** (commit `0f7d90f`): DSMEM 575 cy vs global 1374 cy per broadcast iter. Use DSMEM for cluster-wide value sharing.
- [x] **D3 — .aligned modifier IDENTICAL perf** (commit `7b462d4`): both 373 cy/iter. .aligned is correctness HINT (uniform call), not perf knob.
- [x] **D4 — Async = sync (374 cy)** (commit `4207971`): no overlap benefit when compute (4 FFMAs) << barrier (390 cy). Useful only when compute ≈ barrier time.
- [x] **D5 — Cluster atomic 21% slower** (commit `9a6c43c`): atom.shared::cluster 74 cy vs ::cta 61 cy. Cluster atomic uses ATOM.E (global path per V5 C3) — extra overhead. Prefer per-CTA local + cluster.barrier merge.

## E. CUDA Graph advanced

- [x] **E1 — Graph flags accept; launch identical** (commit `053771b`): default/AutoFreeOnLaunch/Upload/DeviceLaunch/UseNodePriority all 2.06-2.47 µs launch. Subsequent instantiate 30-45× faster (driver caching).
- [x] **E2 — Node priority API works (hint-level)** (commit `163559e`): cudaGraphKernelNodeSetAttribute priority accepted. Per V6 F2, B300 priority doesn't preempt; hint-only.
- [x] **E3 — DotPrint works (DOT format)** (commit `a26ad75`): cudaGraphDebugDotPrint emits Graphviz DOT. Use `dot -Tpng` for visualization. Useful for ML inference graphs with 100+ nodes.
- [x] **E4 — cudaGraphClone = 3.75 µs** (commit `f6f4e2c`): clone executes identically. Use for per-thread or modified graph variants from template.
- [x] **E5 — WHILE = 28.86 µs (18× faster than IF 532 µs)** (commit `8904f14`): cudaGraphCondTypeWhile works. Skipped body launches in 28.86 µs vs IF's 532 µs (V6 D5). For conditional graph patterns, prefer WHILE.
- [x] **E6 — SWITCH = 241 µs (case 0 runs)** (commit `41023b3`): 4-case SWITCH works; correct routing verified. WHILE 28 µs (skipped) < SWITCH 241 µs (1 case) < IF 532 µs (both paths). SWITCH is right primitive for runtime variant dispatch.

## F. Streams + concurrency advanced

- [x] **F1 — Callback chain 0.85 µs/callback batched** (commit `4bb5918`): vs V6 F5 single+sync 2.46 µs. Batching 2000 callbacks + 1 sync amortizes overhead 3×. Use batch mode for multi-stage CPU pipelines.
- [x] **F2 — Spin 4% faster, BlockingSync 70% SLOWER** (commit `3903595`): Auto/Yield 8.25 µs; Spin 7.9 µs; BlockingSync 14.01 µs (+70%, 6 µs wakeup overhead). Use Spin for latency-critical; AVOID BlockingSync.
- [ ] **F3 — cudaMemcpyAsync between streams** (peer-to-peer)
- [x] **F4 — Stream sync vs event-merge identical** (commit `e6aab60`): 1020 µs both methods. Choice is stylistic. For continuing in stream 0 after merge, use event-wait; for "wait for all to finish" use sync.
- [ ] **F5 — Stream with cudaLaunchHostFunc** ordering (V6 F5 baseline = 2.46 us)

## G. Memory + L2 deeper

- [ ] **G1 — L2 prefetch.L2 hit ratio** ncu confirmation (V6 I3 indirect evidence)
- [x] **G2 — Sequential 94% L2 hit; Random 35%** (commit `b23c56e`): HW prefetcher very effective for sequential (even with buffer > L2). Random patterns get 35% L2 hit rate. For random-access workloads (graphs, hash): design for HBM BW.
- [x] **G3 — Higher occupancy = HIGHER L1 hit (shared data)** (commit `69bf8b2`): 32 thr 97.37% → 512 thr 99.84%. Counterintuitive — for shared working set, more threads = more cache reuse. The "occupancy thrashes L1" myth only applies to per-thread disjoint data.
- [x] **G4 — 64KB stride 29% slower than 4KB** (commit `4b26bf4`): real bank conflicts. 256B 735 cy; 4KB 760 cy; 64KB 979 cy. Avoid 64KB+ stride patterns; SoA float4 stride is safe.
- [x] **G5 — L2 holds hot data well** (commit `8d4c8f1`): hot-only 67.84% vs alternating hot/cold 67.14% — LRU effective at keeping 16 MB hot in 126 MB L2 even with 256 MB cold sweep.
- [x] **G6 — TLB negligible up to 1 GB** (commit `dace00f`): 1 MB = 56 cy (L1/L2 hits); 64 MB+1 GB = 80 cy (HBM single-load, NO TLB overhead). B300 TLB handles GB-scale workloads transparently.

## H. Numeric format conversion deeper

- [x] **H1 — Narrow FP cvt ONLY supports .rn** (commit `74b9bd5`): .rz/.rm/.rp REJECTED for e4m3x2. Restricts FP8 quantization to round-to-nearest. For custom rounding (stochastic etc.), use FP32 cvt + manual rounding + pack.
- [x] **H2 — satfinite FREE on FP16 cvt** (commit `3dc7d76`): both 5.4 cy/cvt. HW provides saturation as free side-effect. Always use satfinite for ML quantization.
- [x] **H3 — FP8 cvt satfinite semantics** (commit `975b1e7`): Inf/overflow → MAX (clamp); NaN → NaN encoding; subnormal → 0 (flush). IEEE-style saturate finite for ML quantization.
- [ ] **H4 — Packed vs scalar cvt** (e2m1x4 vs 4× e2m1) — already partial in V6 H3
- [ ] **H5 — TF32 cvt precise format** (10-bit mantissa via cvt.rna.tf32.f32)

## I. Async copy deeper

- [x] **I1 — cp.async DEEP pipelining = 15× speedup** (commit `2b0bafc`): depth=1: 757 cy/op; depth=4: 192; depth=16: **51 cy/op** (15×). HW supports 16+ outstanding groups. Combine with prefetch.L2 (V6 I3) for compound speedup. cuTLASS uses depth 4-16 for sustained HBM throughput.
- [x] **I2 — wait_group N 6% faster than wait_all** (commit `77a82d4`): wait_all = 816 cy; wait_group 8 = 765 cy. Use wait_group K in double-buffer pipelines for partial drain.
- [x] **I3 — cp.async amortizes 12× at large sizes** (commit `59e4881`): 512B = 1.48 cy/byte; 16KB = 0.12 cy/byte. wait_all overhead spreads across more data. cp.async.bulk variant hung (mbarrier alternation needs proper phase setup).
- [x] **I4 — Double-buffer cp.async = 1.94× speedup** (commit `1f63976`): 788 → 406 cy/iter. Ping-pong SMEM with wait_group 1. Compound with V6 I3 (prefetch.L2 1.58×) + V7 I1 (depth=16: 15×) for 3-4× over simple pattern.
- [x] **I5 — bulk + mbarrier phase tracking hangs** (commit `19d8f5f`): PTX compiles but try_wait.parity hangs (parity semantics tricky). Use cuTLASS PipelineTransactionAsync abstraction; defer raw PTX to V8.

## J. Persistent kernel patterns deeper

- [x] **J1 — Batched persistent 74× per-task** (commit `85dbac5`): batch=1: 2.80 µs/task; batch=64: **0.038 µs/task**. Signal RTT amortized across N tasks. For high-freq token dispatch.
- [x] **J2 — Work stealing 16-block sweet spot** (commit `5319afb`): 16 blocks = 96% efficiency; 64 = 80%; 148 = 52% (atomic contention). Use 16-64 persistent workers with atomic counter for many-small-tasks.
- [x] **J3 — Persistent in cudaGraph works (no overhead diff)** (commit `071acd8`): 2.74 µs RTT vs direct 2.77. cudaGraph wraps fine; benefit only if mixing with conditional nodes or other graph patterns.
- [x] **J4 — Persistent kernel ~0 W above baseline** (commit `c01310a`): 1 thread spinning adds only -0.43 W (within noise). Static 167 W paid regardless. Persistent dispatchers FREE in power; cost is opportunity (1 SM unavailable).
- [x] **J5 — X-process persistent via IPC = 2334 µs/task TOO SLOW** (commit `aec1422`): cudaMemcpy from child process is the bottleneck. For high-freq x-process signaling, use shared host pinned memory via mmap or POSIX shm.

## K. Power / thermal advanced

- [x] **K1 — Sustained 2032 MHz at 552 W (no throttle)** (commit `552d1c1`): FFMA-saturated workload holds boost indefinitely at 50% TDP. Confirms V5 D3. For typical ML workloads, don't budget throttling.
- [x] **K2 — Per-pipe energy synthesis** (M11 commit `47b8b1d`): table of pJ/op for FFMA/LDG/HMMA/LDS/LDC/MUFU/IADD3/cvt/sync/barrier with commit refs. DVS scales V² (3.1→10 pJ across clocks).
- [x] **K3 — partial; per-device sampling fix needed** (commit `6942e85`): memory-stream hits ~460 W active. Established prior: V6 C2 pure memory-bound = 12 pJ/byte at 800 MHz; V6 C3 mixed at boost lower.
- [x] **K4 — NO power-gating; static 165 W constant** (commit `d71bdc8`): B300 does NOT power-gate idle SMs. Per-block dynamic ~0.7 W; static baseline same for 1 SM or 148 SMs. Low occupancy WASTES static power — confirms V6 C4 (75× worse efficiency).
- [x] **K5 — Boost-up <100ms; idle-down ~1s lazy** (commit `bfbef09`): clock jumps to 2032 MHz instantly when load arrives; drops gradually back to 120 MHz over 1-2 sec. Sticky boost wins for burst ML inference (no re-clock penalty between short tasks).

## L. Tooling V3

- [ ] **L1 — Auto kernel Roofline classifier** (extend V6 L5)
- [ ] **L2 — Per-warp instruction trace** (proxy for nsight without nsight)
- [ ] **L3 — Kernel power waterfall** (per-section breakdown)
- [ ] **L4 — End-to-end LLM inference Pareto plotter**
- [ ] **L5 — Auto-bisect: find optimal blocksize/clock per kernel**

## M. Brand new explorations

- [x] **M1 — PDL works; needs early launch_dependents to win** (commit `4961876`): regular vs PDL chain identical 2.02 ms/pair when A's launch_dependents is at end. PDL benefit needs early trigger (cuTLASS-style epilogue overlap with next mainloop).
- [ ] **M2 — Dynamic parallelism on B300 + tcgen05** (mix DP with TMEM)
- [x] **M3 — Triple-chevron = cudaLaunchKernel** (commit `052fecd`): both 3.07 µs/launch identical (chevron compiles to cudaLaunchKernel). Driver API hot-launch similar; one-time setup is the diff.
- [x] **M4 — Comprehensive devprop dump** (commit `6561784`): B300 SXM6 AC (sm_103a). 148 SMs × 4 SMSPs; L2=126.5MB; SMEM/SM=228KB; bus=7680b (12 stacks); 287GB. Note memoryClockRate removed in CUDA 13.
- [x] **M5 — Mempool wins 64K-16M; Sync better at tiny + huge** (commit `f037a26`): 4KB Sync 1.1µs > Async 4.0µs; 1-16MB Async 2-7× faster; 256MB Sync 95µs < Async 329µs. Sweet spot 64K-16M. For huge: pre-alloc once.
- [x] **M6 — cuStreamWriteValue = 0.45 µs (5-6× faster than kernel)** (commit `0eb236e`): Write 0.45 µs, Wait 0.44 µs, pair 0.89 µs. vs kernel launch (2.3) or persistent RTT (2.77). Underused primitive — major win for ultra-low-latency control plane (token decoding, multi-GPU handshakes).
- [x] **M7 — GPU offset stable within 2.3 µs** (commit `eb56fde`): GPU globaltimer lags CPU by 1.81 sec but offset is stable within 2.3 µs. Use min(offset) as ref. Unblocks accurate TTF measurement.
- [x] **M8 — Scheduler look-ahead weak** (commit `20a6fc0`): LDG+dep vs indep = 2% diff. B300 relies on COMPILER (SASS reorder) + ILP + warp-level parallelism, not HW reorder. Developer must expose ILP.

---

## Completion tracking

Total: ~50 items. Many are V6 deferred (A, B, C series) + new questions (D, E, F, etc).

When ALL `[x]`, generate V8 + write M11 synthesis.
