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

- [ ] **B1 — Build cuTLASS-equivalent SMEM descriptor encoder** in raw PTX
- [ ] **B2 — tcgen05.mma m64n8k16 BF16 functional test** with descriptors
- [ ] **B3 — tcgen05.mma + HMMA simultaneous concurrency test** (closes V5 B1)
- [ ] **B4 — tcgen05.mma power signature** (does it match HMMA pJ/op?)
- [ ] **B5 — tcgen05.mma latency vs throughput** (single op vs back-to-back)

## C. Power optima per workload

D4 showed V² scales with clock; what's the min-energy point per workload?

- [ ] **C1 — FFMA-saturated min-energy clock sweep** (sweep 510-2032 MHz, find min pJ × time product)
- [ ] **C2 — DRAM-bound min-energy clock** (does workload BW change opt point?)
- [ ] **C3 — Mixed FFMA+DRAM min-energy** (per-workload Pareto frontier)
- [ ] **C4 — Single-warp min-energy** (does occupancy affect opt point?)
- [ ] **C5 — Energy efficiency tokens/J for LLM kernels** (RMSNorm/SoftMax/HMMA at each clock)

## D. CUDA Graph internals deeper

H1 showed SetParams = free; what about node insertion / removal?

- [ ] **D1 — cudaGraphAddKernelNode latency** (build-time cost)
- [ ] **D2 — cudaGraphInstantiate vs ExecUpdate** (use ExecUpdate for in-place changes)
- [ ] **D3 — Graph capture vs explicit construction** speed
- [ ] **D4 — Multi-node graph: dependency chain depth latency**
- [ ] **D5 — Conditional graphs (CUDA 12.3+)** — perf overhead

## E. Persistent kernel patterns

K1 showed persistent only wins for >50 µs work; deep dive on dispatch.

- [ ] **E1 — Persistent kernel signal latency** (per-task dispatch via mailbox)
- [ ] **E2 — Persistent kernel + mbarrier signaling** (cross-block coordination)
- [ ] **E3 — Persistent kernel + cudaDevice synchronization** primitives
- [ ] **E4 — Per-SM persistent state cost** (when does TCB context dump dominate?)
- [ ] **E5 — Persistent kernel restart cost** vs cold launch

## F. Stream + concurrency patterns

K5 showed 2 streams parallel = 1.97×; what about 4, 8, 16?

- [ ] **F1 — Stream parallelism scaling** (1, 2, 4, 8, 16 streams)
- [ ] **F2 — Stream priority effect** (cudaStreamCreateWithPriority high vs low)
- [ ] **F3 — Stream + event chain depth latency** (long dependency chains)
- [ ] **F4 — Default stream vs custom stream** overhead difference
- [ ] **F5 — cudaLaunchHostFunc cost** (callback overhead)

## G. Memory hierarchy edge cases

- [ ] **G1 — L2 partition awareness** — does access pattern affect L2 partition use?
- [ ] **G2 — L1 vs L2 cache eviction policies** (LRU vs other?)
- [ ] **G3 — DRAM bank conflict measurement** (interleaved access patterns)
- [ ] **G4 — Persistent L2 cache hint** (cudaCtxResetPersistingL2Cache + setAccessPolicyWindow)
- [ ] **G5 — Read-only cache (texture path)** vs L1 — does TEX still work on B300?
- [ ] **G6 — Constant cache vs L1** (cudaSymbolToPtr vs `__constant__`)

## H. Numeric format conversion deep dive

J series showed TF32 8000× precision loss; explore other formats.

- [ ] **H1 — FP8 e4m3 vs e5m2 cvt latency**
- [ ] **H2 — FP6 e2m3 vs e3m2** (newer Blackwell formats)
- [ ] **H3 — NVFP4 conversion** (already covered in catalog?)
- [ ] **H4 — INT4/INT8 cvt** symmetric vs asymmetric quantization
- [ ] **H5 — Half BF16↔FP16 cvt** cost (rare path)

## I. Async copy variants

A3 showed cp.async.cg = 314 cy; what about other variants?

- [ ] **I1 — cp.async.ca vs .cg** (L1 cached vs bypass)
- [ ] **I2 — cp.async with predicate** (ptx form .pred)
- [ ] **I3 — cp.async + L2 prefetch combo**
- [ ] **I4 — cp.async.bulk.tensor with TMA descriptor** (full TMA path)
- [ ] **I5 — Multicast TMA (cluster-wide bulk copy)**

## J. Random / surprising

- [ ] **J1 — Why does HMMA latency go down with ILP?** (Hopper documented as 16/8/4 cy stride; B300 saturates ~10 cy)
- [ ] **J2 — Predicate evaluation cost** (FFMA vs FFMA @P0 on)
- [ ] **J3 — Branch divergence cost** (ifelse with 1 vs 31 threads diverging)
- [ ] **J4 — Shuffle-XOR vs shuffle-down vs shuffle-up** latency
- [ ] **J5 — atom.shared.add vs atomicAdd** (compile down to same SASS?)
- [ ] **J6 — VOLATILE READS in tight loop** (does HW recognize patterns?)

## K. Kernel launch deep ninja

- [ ] **K1 — Cooperative launch overhead** (cudaLaunchCooperativeKernel)
- [ ] **K2 — Cluster launch overhead** (cudaLaunchKernelEx with cluster dim)
- [ ] **K3 — Multi-device launch** (cudaLaunchCooperativeKernelMultiDevice — deprecated?)
- [ ] **K4 — Kernel parameter passing cost** (large vs small param size)
- [ ] **K5 — Kernel arg via constant mem vs grid arg** speed difference

## L. Tooling V2

- [ ] **L1 — Auto pipe-overlap matrix tool** (run all pipe pairs through 11×11 grid)
- [ ] **L2 — Power-vs-throughput Pareto plotter** (sweep clock + workload)
- [ ] **L3 — Per-warp latency tomography** (per-clock per-pipe per-warp)
- [ ] **L4 — Kernel dispatch latency profiler** (time from launch to first SM start)
- [ ] **L5 — Roofline plotter** (FLOP rate vs arithmetic intensity)

---

## Completion tracking

Total: 65 items.

When ALL `[x]`, generate V7 and write M8 synthesis.
