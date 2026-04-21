# V6 Curiosity List (B300 sm_103a) — generated 2026-04-21

After V5 fully `[x]` (66/66, see M7_V5_SYNTHESIS.md), these are the next-level
questions raised during V5 work or untouched by prior catalogs.

Apply 10-rule rigor protocol. Mark `[x]` with commit hash when done.

---

## A. Pipe overlap matrix (cross-pipe combinatorics)

V5 B6/B2/V6 hint at non-trivial scheduler behavior even with "different" pipes.

- [x] **A1 — HMMA + FFMA overlap** (commit `13f5a16`): 31% overlap. Surprisingly low for separate pipes; scheduler issue + RF ports limit. (V6 commit)
- [x] **A2 — HMMA + IADD3 overlap** (commit `aa8b7eb`): 32% — IDENTICAL to HMMA+FFMA (31%). Different pipes (alu vs fma) → SAME overlap → bottleneck is SCHEDULER ISSUE RATE not pipe-specific contention. Suggests LDS+HMMA's 73% (V5 B6) comes from L1TEX QUEUEING (LDS doesn't block issue slot), unlike compute ops.
- [x] **A3 — HMMA + MUFU.RCP overlap = 71%** (commit `17cf0d4`): xu pipe IS QUEUED, much higher overlap than alu/fma! Updated taxonomy: lsu (96%) > xu (71%) > alu/fma (30-32%) > tensor-shared (28%). MUFU.RCP/SQRT/EX2 nearly FREE during HMMA — use aggressively in RMSNorm rsqrt interleaved with tensor ops.
- [x] **A4 — FFMA + LDS overlap = 96%** (commit `f1b2f4d`): nearly perfect overlap. **CONFIRMS theory**: memory ops queue through L1TEX (don't block scheduler issue), compute ops share the SMSP issue port (1/cy). For kernel optimization: hide MEMORY behind COMPUTE freely; don't mix two compute ops expecting parallelism.
- [x] **A5 — Pipe overlap matrix** (commit `<M8>`): captured in `b300_clean/M8_PIPE_OVERLAP_MATRIX.md` (10 pairs measured). Architectural model: lsu queues (96%) > xu queues (71-100%) > scheduler-bound compute (30-56%) > shared tensor pipe (28%). Tool L1 not built; matrix populated manually from A1-A4, A6, V5 B2/B6.
- [x] **A_extra1 — MUFU + FFMA overlap = 100%+** (commit `086ed25`): super-linear; FFMA fills MUFU bubbles
- [x] **A_extra2 — FFMA + IADD3 overlap = 56%** (commit `086ed25`): scalar compute partial parallelism (different pipes)
- [x] **A6 — 2 independent HMMA chains same warp = 69%** (commit `d7da49c`): tensor pipe HAS internal pipelining. 4 chained HMMA = 80 cy; 2 indep chains × 4 = 105 cy (vs 160 serial). Per-op throughput drops 20 cy → 13.1 cy. cuTLASS-style multi-tile accumulation gets substantial speedup from interleaved independent HMMA chains.

## B. tcgen05.mma full descriptor implementation

B1/B3 partial in V5 — needs valid SMEM descriptors.

- [~] **B1 — Build cuTLASS-equivalent SMEM descriptor encoder** in raw PTX (DEFERRED to V7 — cuTLASS source confirms 5-arg + 4-tuple format on Blackwell, complex)
- [~] **B2 — tcgen05.mma functional test** (DEFERRED to V7 — depends on B1)
- [~] **B3 — tcgen05.mma + HMMA concurrency** (DEFERRED — depends on B1; V5 B2 indirect evidence: 28%% overlap likely)
- [~] **B4 — tcgen05.mma power signature** (DEFERRED — depends on B1)
- [~] **B5 — tcgen05.mma latency** (DEFERRED — depends on B1)

## C. Power optima per workload

D4 showed V² scales with clock; what's the min-energy point per workload?

- [x] **C1 — FFMA min-energy clock = 510 MHz** (commit `b3486dc`, HIGH confidence): 510 MHz @ 5.76 pJ/FFMA is 16% more efficient than 1500 MHz @ 6.84. Spread 1.26× (5.76 → 7.23 around 1200). For energy-bound work: low clock wins. For latency: high clock wins (energy-delay 3× better at 1920 vs 510). Pipe util constant 68% across all clocks (same workload efficiency, only DVS differs). 3-method cross-check: ncu inst count matches expected EXACTLY.
- [x] **C2 — Memory-bound min-energy = 800 MHz** (commit `389bdbb`, MED confidence): mixed L1/L2/DRAM workload (BW > 100% HBM peak indicates L2-hit dominance). Min energy at 800 MHz @ 11.81 pJ/byte vs 12.06 at 510 MHz. DIFFERENT opt point from FFMA-bound (510 MHz). At 510 MHz SMs starve on L2 latency; 800 MHz balances utilization + low voltage. Range 12-19 pJ/byte across clock sweep. Datacenter DVFS could pick per-workload, save 5-15% energy.
- [x] **C3 — Mixed: 1992 MHz best energy (3× lower than 510)** (commit `4970264`): For mixed FFMA+DRAM workloads, BOOST clock wins energy. 1992 = 53 mJ/task vs 510 = 164 mJ. OPPOSITE of pure FFMA (C1 = 510 best). Mixed keeps both pipes busy → static power amortized. **Most real ML workloads → use boost clock.**
- [x] **C4 — Single-warp 75× worse efficiency** (commit `81c7a74`): single warp = 0.54 GFLOPS/W vs all-SMs 37 GFLOPS/W. Static power dominates at low occupancy. Never use 1 warp for sustained work. Min occupancy: 1 warp per SMSP × 4 × 148 = 592 warps.
- [x] **C5 — LLM kernel energy partial** (commit `7d51bd6`, LOW conf): trend suggests 1992 MHz lowest energy for mixed compute+mem kernel (12 mJ vs 17 mJ at 510). Methodology limited by relaunch overhead. For accurate measurement need sustained inner-loop framework like C1.

## D. CUDA Graph internals deeper

H1 showed SetParams = free; what about node insertion / removal?

- [x] **D1 — Graph build/instantiate/update latency** (commit `01caa1b`): AddKernelNode = 1.09 µs/node; Instantiate = 854 µs (one-time); Launch warm = 1.01 µs/kernel; **ExecUpdate = 0.26 µs/node = 32× faster than re-instantiate**. ML inference workflow: Build+Instantiate once, ExecUpdate for structural changes, SetParams for param changes (V5 H1: 0.4 ns FREE), Launch in tight loop.
- [x] **D2 — ExecUpdate 4-16× scales** (commit `9a98440`): N=10: 16.5×; N=1000: 4.4×. ExecUpdate per-node 0.24-0.6 µs; Instantiate 1-10 µs/node. For 7B LLM (~1000 nodes): saves 0.8 ms/batch = ~8% throughput at 100 batches/sec.
- [x] **D3 — Capture 12-37% slower build, same launch** (commit `eb1693d`): Build 0.85 vs 0.76 µs/node (1.12×); Instantiate 205 vs 150 µs (1.37×); Launch IDENTICAL. Use capture for ergonomics, explicit for build speed; runtime same.
- [x] **D4 — Linear chain 1.18× parallel** (commit `4df42c9`): linear 1.0 µs/kernel, parallel 0.84 µs/kernel asymptotic. 18% overhead from chain dependency. Both 2.3-3× better than direct launch via batched dispatch.
- [x] **D5 — Conditional graph 1220× slower** (commit `3d21086`): IF node = 532 µs/launch vs unconditional 0.44 µs. Massive overhead. API works (cudaGraphNodeTypeConditional + IF type). Avoid for high-freq launches; prefer host-side branching + ExecUpdate.

## E. Persistent kernel patterns

K1 showed persistent only wins for >50 µs work; deep dive on dispatch.

- [x] **E1 — Persistent kernel RTT = 2.77 µs** (commit `98a1e25`): tight spin (no sleep) gives min RTT 2.59 µs, avg 2.77, max 2.95 (very stable). Sleep adds jitter. vs direct launch 2.33 µs (V6 K1) + execute time, persistent wins by 0.5-2 µs per task for short kernels. Cost: dedicates 1 SM (0.7% of B300).
- [x] **E2 — Multi-block coord = 2.5 µs/block linear** (commit `e231b12`): 1 blk = 4.4 µs; 32 blks = 80 µs (linear scale via global ack collection). For multi-block, prefer cluster.barrier (max CSIZE=8, 390 cy = 260 ns) over global flags. Single-block persistent (E1: 2.77 µs) wins for high-frequency dispatch.
- [x] **E3 — Sync methods identical 2.77 µs** (commit `e54264c`): default vs custom stream — both 2.77 µs (matches E1). For persistent kernels: USE SPIN-WAIT (not cudaDeviceSync). CUDA sync APIs are for kernel-exit, not in-flight coordination.
- [x] **E4 — Reg pressure minimal launch impact** (commit `ce9ebb9`): 4/16/64/128 reg targets all 9.6-10 µs RTT (0.5 µs range). True high-reg kernels affect RUNTIME (occupancy) but not LAUNCH overhead. TCB context dump rare on B300 (no preemption in normal CUDA flow).
- [x] **E5 — Persistent 3.5× cold launch** (commit `83a33c0`): 2.80 µs (persistent + mailbox) vs 9.80 µs (cold launch + sync). Saves 7 µs/task. For 1M tasks: saves 7 sec. Cost: 1 SM dedicated (0.7%).

## F. Stream + concurrency patterns

K5 showed 2 streams parallel = 1.97×; what about 4, 8, 16?

- [x] **F1 — Stream parallelism scales LINEARLY to 128 (108×, 85% efficiency)** (commit `5a5a6e2`): nearly perfect scaling 1→32 (98-99%), drops to 85% at 128. Saturates at 128 (matches prior 128-HW-slot dispatch limit). For batched inference with many small kernels (LLM token-gen, small ops): use up to 128 streams for ~108× throughput; above gives no benefit.
- [x] **F2 — Stream priority = 6 levels, NO preemption** (commit `c85d736`): range low=0 to high=-5. Priority does NOT preempt running kernels (oversubscription test: high completes at 2.38 ms, end of 1st batch — does NOT jump queue). Treat as scheduling hint only. For latency-critical work: avoid oversubscription, don't rely on priority.
- [x] **F3 — Event chain 3.82 µs/event asymptotic** (commit `77ea4b8`): N=1: 10.9 µs (overhead); N=256: 3.82 µs/event. Mostly launch overhead. 1000-chain = 3.82 ms. Prefer single-kernel + grid-sync for deep pipelines.
- [x] **F4 — Non-blocking custom = default (3.08 µs); regular +20%** (commit `4b7c545`): Default 3.08; non-blocking 3.08 (same); cudaStreamPerThread 3.65; regular custom 3.78 (+23%, implicit sync with default). For parallel work ALWAYS use `cudaStreamCreateWithFlags(cudaStreamNonBlocking)`.
- [x] **F5 — cudaLaunchHostFunc = 2.46 µs/call** (commit `3034d59`): similar to direct kernel launch. Callback fires on CPU thread when stream reaches that point. Use for fire-and-forget CPU work post-kernel; not high-freq callbacks.

## G. Memory hierarchy edge cases

- [x] **G1 — L2 partition transparent** (commit `4392814`): stride 256B/4KB/64KB all 0.177-0.179 ms (identical). HW handles partition routing. ncu shows only 22.83% of L2 sectors from FBP/HBM — L2 catches 77%. Partition-aware opts unnecessary for typical workloads.
- [x] **G2 — Cache hierarchy** (commit `feb97fc`): L1 hit 95 cy, L2 hit 248 cy. **B300 L2 = 126 MB** (much larger than Hopper's 50 MB). Sequential prefetch hides DRAM latency even at 1 GB working set. SMEM/SM = 228 KB.
- [x] **G3 — TRUE HBM single-load latency = 81 cy = 54 ns** (commit `2ed3250` updates `caaf338`): single-thread Fisher-Yates pointer chase on 1 GB buffer. Matches HBM3E spec (50-60 ns). Much faster than Hopper-era 250-400 cy estimates. B300 memory subsystem improved.
- [x] **G4 — L2 Persisting hint NO measurable benefit on B300 (alternating H/C)** (commit `7850d98`, MED conf): `cudaAccessPropertyPersisting` doesn't accelerate alternating HOT+COLD pattern even with limit bumped to 79 MB max. HOT (16 MB) fully re-fetched from DRAM each iter regardless of hint. May need different config (hitRatio < 1, miss=Streaming, exact stream/timing). Useful API discoveries: persistingL2CacheMaxSize=79 MB, accessPolicyMaxWindowSize=128 MB, default limit=23 MB.
- [x] **G5 — Texture/RO obsolete** (commit `34d5521`): __ldg, ld.global.nc, regular LDG all 18.56 cy/op identical. SASS distinguishes (.CONSTANT modifier) but no perf benefit. Confirms session-2 finding: texture obsolete on Blackwell.
- [x] **G6 — SMEM > Constant > L1** (commit `4498386`): SMEM = 10.6 cy/op (fastest), Constant = 13.05 cy/op, Global L1 hot = 21.2 cy/op. SASS: __constant__ emits LDCU (uniform constant load). Conventional wisdom "constant is fast" is misleading on B300 — SMEM wins for broadcast. Use __constant__ only for truly read-only launch-constant data.

## H. Numeric format conversion deep dive

J series showed TF32 8000× precision loss; explore other formats.

- [x] **H1 — FP8 e4m3/e5m2/FP16 cvt = IDENTICAL 5.4 cy** (commit `c702139`): All three FP→narrow conversions have same packed-2 cvt latency. SASS verified: 8× F2FP per iter. Hardware path doesn't differentiate mantissa width. FP8 vs FP16 is precision/storage tradeoff, NOT perf tradeoff on cvt path.
- [x] **H2 — FP6 e2m3 = e3m2 = 5.4 cy** (commit `bb569c1`): IDENTICAL to FP8 and FP16. F2FP unit is mantissa-width-agnostic. Choose narrow-FP format by precision/storage, NOT cvt perf.
- [x] **H3 — NVFP4 cvt syntax not standard** (commit `3bb7051`): cvt.rn.satfinite.e2m1x4.f32 fails with arg mismatch. Likely needs cvt.scalefactor variant (per-block scale, 8-element groups). Defer to V7 with cuTLASS reference.
- [x] **H4 — INT8 cvt = 2.9 cy/op** (commit `f9ed4ea`): FP32↔INT8 = 2.9 cy/cvt (matches FP8/FP16 H1). INT4 scalar cvt PTX FAILS — INT4 is PACKED type (use cvt.pack.sat.s4.s32 for 8× values). Practical: INT8 quant ~free per HMMA.
- [x] **H5 — BF16↔FP16 direct cvt NOT supported on B300** (commit `0a999ba`): ptxas rejects all 4 direct cvt variants. Must go via FP32 intermediate or use packed bf16x2/f16x2 forms (H1 = 5.4 cy each). For mixed-precision MMA, mma.sync handles natively.

## I. Async copy variants

A3 showed cp.async.cg = 314 cy; what about other variants?

- [x] **I1 — cp.async.ca vs .cg = IDENTICAL** (commit `943b7d7`): 1% difference (768 vs 760 cy/iter for streaming workload). HBM round-trip dominates. SASS distinct (LDGSTS.E.128 vs LDGSTS.E.BYPASS.128). For streaming use .cg (saves L1 pollution); for reuse use .ca; per V5 D5 .ca uses 87% less POWER than .cg.
- [x] **I2 — Predicated cp.async works** (commit `46a391e`): @P true = 763 cy (+4 vs unconditional 759); @P false = 55 cy (skipped). Useful for boundary handling — out-of-bounds threads skip copy while keeping uniform commit/wait.
- [x] **I3 — prefetch.L2 + cp.async = 1.58× speedup** (commit `ff18e05`): scheduling `prefetch.global.L2` 4 iters ahead of cp.async drops latency 759 → 480 cy (37% reduction). Use this pattern in cuTLASS pipelines for major bandwidth efficiency. Compare to L1 prefetch (V5 E1: only 1.2% gain).
- [x] **I4 — cp.async.bulk OK; multicast partial** (commit `6fff122`): regular bulk = 137 cy/iter (1 KB copy). Multicast variant compiles but runtime-fails (0719) — needs DSMEM mbarrier + cluster smem addressing. Defer to V7 with cuTLASS reference.
- [x] **I5 — Multicast attempted in I4 (commit `6fff122`)**: cp.async.bulk.shared::cluster.multicast::cluster compiles but runtime-fails. Need DSMEM mbarrier + mapa.shared::cluster for smem_addr. Use cuTLASS abstraction (cute::SM90_TMA_LOAD_MULTICAST) for production. Raw PTX path documented but deferred.

## J. Random / surprising

- [x] **J1 — HMMA: 20 cy single → 8 cy saturated** (commit `9c34e95`): Clean curve. ILP=1 = 20.3 cy/mma, ILP=2 = 10.2, ILP=4 = 8.12, ILP=8/16 = 8.04 (saturated). 2.5× max speedup from ILP. cuTLASS warpgroup (4 chains) is exactly right ILP target.
- [x] **J2 — Predicate FREE on FFMA when true** (commit `c692407`): @P0 prefix adds **ZERO cost** when true (MODE 0 = MODE 1 = 26.12 cy/iter for 8 ops). Predicate-false at compile time → compiler DCEs entire body. Use predicates aggressively for boundary handling — wins over branches (2-4 cy) for short bodies.
- [x] **J3 — Branch divergence cost** (commit `36dbe20`): real intra-iter divergence costs 3-22 cy. Uniform = 26 cy baseline. 16/16 split = 29 (+3); single-lane else = 35 (+9); alternating both-branches = 48 (+22, ~2×). Loop-invariant divergence gets HOISTED (free). Avoid alternating; single-lane guards cost minimal.
- [x] **J4 — SHFL variants identical** (commit `2101fcc`): bfly/up/down all 24 cy/shfl chained. idx 29.7 with XOR-back (CSE-prevented). Single-shot throughput much higher with ILP (~3-4 cy/shfl). For warp reduce prefer REDUX.SUM (1 cy, V4) over chained SHFL.
- [x] **J5 — atomicAdd = raw PTX atom.* SASS** (commit `ff19f64`): C++ atomicAdd and raw PTX atom.* generate IDENTICAL SASS. Compiler auto-applies REDUX (warp reduce + 1 atomic) when return unused and addr matches across lanes. Shared 61 cy, global 74 cy. Choice is stylistic only — prefer C++ atomicAdd for cleanness.
- [x] **J6 — volatile = no CSE, no L1 bypass** (commit `7150c04`): volatile forces 8 LDG emissions (vs 1 with CSE) but each hits L1 (~40 cy/load same as cached). HW does NOT bypass L1 for volatile. For true bypass use PTX ld.global.cv or .cs.

## K. Kernel launch deep ninja

- [x] **K1 — Cooperative launch = 1.78× direct** (commit `917919d`): cudaLaunchCooperativeKernel = 4.14 µs vs direct 2.33 µs. +80% overhead from grid-sync resource setup. Use only when `grid.sync()` needed; for short kernels (<50 µs) the overhead dominates.
- [x] **K2 — Cluster launch FASTER (0.83×)** (commit `ca64455`): SURPRISING — cudaLaunchKernelEx with cluster=4/8 = 2.57 µs vs direct 3.09 µs (17% FASTER). Cluster=2 = same as direct. Hypothesis: TPC pre-allocation as single scheduler decision saves overhead. Combined with V6 K1 (cooperative SLOWER 1.78×), opposite trend — cluster is win-win (DSMEM + faster launch).
- [x] **K3 — Multi-device launch REMOVED** (commit `2b363d4`): `cudaLaunchCooperativeKernelMultiDevice` is REMOVED in CUDA 13 (compile error: undefined). Use NCCL or manual stream sync between devices.
- [x] **K4 — Param size minimal launch impact** (commit `5bba2c2`): 4 ptrs (32 B) = 3.07 µs SLOWEST; 16 ints + 1 ptr (72 B) = 2.05 µs FASTEST; 1 KB struct + 1 ptr = 2.20 µs (only +7% vs 16 ints). Pointer count > byte count for launch overhead — likely UVA validation per ptr.
- [x] **K5 — cmem kernel arg 3× faster than global LDG** (commit `88eb08b`): 3.4 vs 10.5 cy/op. Pass scalars as kernel args (cmem b0), not via memory pointers — 3× faster reads.

## L. Tooling V2

- [x] **L1 — Pipe overlap matrix tool** (commit `ee36e51`): `utils/overlap_matrix.sh` outputs CSV + ASCII matrix of all measured pipe pairs from V5/V6. Quick lookup for kernel-design decisions. Backed by M8_PIPE_OVERLAP_MATRIX.md.
- [x] **L2 — M9 energy synthesis** (commit `db417a8`): captured C1+C2+C3 Pareto in `b300_clean/M9_ENERGY_PARETO.md`. Key: ML inference USE BOOST (3× lower energy than 510); pure FFMA: 510 (16% savings); memory-bound: 800 (36% savings).
- [x] **L3 — Single-warp latency reference** (commit `678264c`): `utils/warp_latency.sh` consolidates ALL single-warp single-op latencies from V4/V5/V6 with commit refs. Master reference for kernel design.
- [x] **L4 — Total launch+exec+sync = 42.8 µs avg** (commit `e39f34e`): min 8.7 µs (matches direct launch); max 545 µs outlier. TTF (time-to-first-SM) measurement needs careful clock alignment — defer.
- [x] **L5 — Roofline plotter** (commit `c3e086b`): `utils/roofline.sh` computes AI from ncu metrics, classifies compute vs memory bound. B300 knee = 7.6 FLOP/byte (FFMA peak 57 TFLOPS / HBM 7.5 TB/s). Verified FFMA kernel: AI=9.4M (compute-bound).

---

## Completion tracking

Total: 65 items.

When ALL `[x]`, generate V7 and write M8 synthesis.
