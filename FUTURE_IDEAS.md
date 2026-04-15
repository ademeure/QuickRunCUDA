# B300 Characterization — Future Test Ideas & Gap Analysis

Living document of untested / under-tested angles. Use this as a starting point when the loop fires and we want a new measurement.

## HIGH-VALUE GAPS (not yet tested or weakly covered)

### Compute / tensor
- **tcgen05.mma peak TFLOPS** — the big one. Requires real descriptors (A/B matrices in smem), TMEM accumulator, proper commit/wait. Expected ~5 PFLOPS FP8, ~2.5 PFLOPS FP16, ~10 PFLOPS FP4 with scaling. Existing `bench_tcgen05_mma.cu` only tests the runtime path, not peak.
- **FP8 block-scaled MX format** (`kind::mxf8f6f4`, `kind::mxf4`) — catalog notes "rejected by nvcc 13.2 ptxas" as of April 2026. Retry periodically.
- **FP6/FP4 dense MMA** — catalog has F2FP conversion TFLOPs but not the MMA-path peak for native narrow-format.
- **IMMA (integer MMA)** peak — only HMMA/BMMA tested.
- **BMMA/xor-popc** peak — popcount-style MMA for binary nets.
- **Integer scalar IMAD / IMUL throughput** — no peak measurement; assume same as FFMA but verify.
- **Sub-warp MMA patterns** — wgmma was rejected, tcgen05 is replacement. Small shapes vs large shapes.

### Memory hierarchy
- **TMEM read/write throughput** at various layouts (partial catalog coverage) — validate with ncu.
- **TMEM concurrent access from multiple warpgroups** — contention?
- **Shared memory bank conflicts** by stride pattern — LDS/STS.
- **ldmatrix / stmatrix throughput** with various shapes (`x1`, `x2`, `x4`).
- **L1 instruction cache pressure** — what happens with huge kernels?
- **Constant memory latency + BW**.
- **Texture / tex.1d / tex.2d path** — still exists on B300?
- **`ld.global.nc.*` (non-coherent)** — does it help at all vs `.cg`?
- **`ld.global.L1::evict_normal/first/last`** eviction hints — do they matter?

### Multi-GPU (continuing thread)
- **cp.async.bulk (TMA) cross-GPU** — does it work with P2P remote memory? If yes, huge bulk-path for comms.
- **cp.async (non-bulk) cross-GPU**.
- **Cluster launch cross-GPU** — likely rejected, confirm semantic.
- **MBARRIER on remote-mapped memory** — can a producer on GPU 0 arrive on a barrier that GPU 1 waits on?
- **Host pinned memory access (zero-copy)** from GPU — PCIe latency path.
- **Multiple writers, single reader** ping-pong — realistic sync primitive.
- **Clock skew** between GPU 0 and GPU 1 (via globaltimer).
- **NVLink queue depth saturation** — how many outstanding requests until link backpressure?
- **Single-SM NVLink saturation** — how many warps/SM needed to saturate 1 GPU's outbound link?

### Fences & sync
- **Fence cost matrix**: sc vs acq_rel × cta/gpu/sys × with/without writes × LOCAL/REMOTE.
- **fence.proxy.*** generic↔async conversion cost.
- **fence.mbarrier_init** overhead.
- **System-level release-acquire pattern** (atomic + fence + read).
- **Barrier cost per warp count** (e.g., `bar.sync 0, N` for subsets).

### Warp / SM
- **Warp scheduler stalls under load** via ncu (`sm_warp_issue_stalled_*`).
- **Instruction issue rate per SMSP** — when does ILP help?
- **Register bank conflicts** (same-bank operand reads).
- **Predicated execution cost** — `@P0 INST` vs unpredicated.
- **Branch divergence** cost vs warp convergence cost — see below (tested).

### Primitives / intrinsics
- **`__shfl_sync` throughput** at various masks.
- **`__ballot_sync` / `__any_sync` / `__all_sync`**.
- **CREDUX / warp-wide reduce** variants.
- **`__nanosleep` HW quantum** — partial (64 ns steps).
- **`setmaxnreg.aligned`** (dynamic register balancing, sm_100+).

### Misc
- **CUDA graph launch overhead** vs individual launches.
- **Stream priority scheduling**.
- **Driver overhead** for small kernels (persistent vs launch).
- **Power / clock throttling** under sustained load — does clock drop from 1920 MHz?
- **SM scheduling placement** — CTA N always lands on SM N?

## CURRENT SESSION TESTED (done in this or earlier cycles)

- Multi-GPU atomic/fence/BW comprehensive (write 718 GB/s, read 820 GB/s, atomic 1.5-2.9 TB/s local)
- ncu validation of L1/L2/HBM peaks
- LOCAL/REMOTE mixing granularity
- Deep NVLink pipelining (28.7× speedup with 32 chains/thread)
- Atomic op types & data widths (all similar cost)
- Clock64 overhead (36 cy) + __nanosleep quantization (64 ns steps)
- Branch divergence (true vs predicated)

## DESIGN RULES DISCOVERED

1. **Coalesce atomic addresses** within a warp → 32× fewer HW packets → up to 32× throughput.
2. **Dedicate CTAs** (not warps) to LOCAL or REMOTE atomic patterns.
3. **~32 SMs saturate NVLink** — reserve the other 116 for compute.
4. **fence more than once per ~50 µs** of cross-GPU work is wasteful (REMOTE fence caps at ~50 µs).
5. **REMOTE atomic contention merging** saves NVLink packets, giving HIGHER semantic throughput than unique.
6. **Deep pipelining (N_CHAINS=32)** hides cross-GPU latency ~29×.

## RECENT ADDITIONS (this cycle)

Tested since FUTURE_IDEAS.md was first created:
- [x] clock64 read overhead (36 cy) + globaltimer (32 cy)
- [x] __nanosleep quantization (~64 ns steps, 40 ns floor)
- [x] Branch divergence (true): 2-way = 2.2×, 4-way = 4.7×
- [x] Clock/power/thermal: no throttling at 339 W (B300 has huge headroom)
- [x] Shared memory bank conflicts: 32-way = 2.55× slowdown
- [x] Warp primitives: shfl 41 cy, ballot 33, match_any 387 (9× slower)
- [x] Integer IADD parallel with FFMA on pipe_alu (free)
- [x] MUFU throughput: rsqrt 727, sqrt 623, sin/cos 284, log2 143 inst/ns
- [x] Constant memory broadcast: 2 cy/load (43× faster than cached global)

## DONE THIS FIRING (BIG WINS)

- [x] **tcgen05.mma peak TFLOPS** — verified 4.65 PFLOPS FP8, 2.33 FP16, 1.16 TF32 (93% of spec)
- [x] **tcgen05.mma.sp sparse**: 7.44 PFLOPS FP8 sparse (1.6× over dense, 74% of 10 PFLOPS spec)
- [x] **tcgen05.mma cta_group::2 cluster**: same 4.65 PFLOPS total (NOT a 2× peak)
- [x] **kind::i8 NOT supported on sm_103a (B300)** — FP8 only for B300 inference
- [x] **mxf8f6f4 / mxf4 block-scaled**: rejected (need scale TMEM allocation, complex)
- [x] **FP6/FP4 under f8f6f4**: same throughput as FP8 (shared K=32 path)
- [x] **Multi-SM linear scaling 1→148 SMs**: confirmed perfect
- [x] **Multi-warp per SM serialized**: ONE tensor pipe per SM
- [x] **ldmatrix.{x1,x2,x4}**: x4 is 4× the BW of x1 in same cycles (29 cy)
- [x] **stmatrix.{x1,x2,x4}**: x4 = 14.2 B/cy/warp
- [x] **tcgen05.ld.16x64b sweet spot**: x16 at 57 B/cy/warp (~65 TB/s chip TMEM read)
- [x] **Smem peak BW**: ~250 GB/s/SM, ~37 TB/s chip-wide
- [x] **Integer intrinsics**: prmt/shf.l/dp4a fast (3.25 cy), popc/brev slow (8 cy), clz slowest (15 cy)
- [x] **Fence/membar costs**: cta-scoped 27-36 cy, fence.proxy.async (full) 179, gpu-wide 292
- [x] **setmaxnreg**: dec=73 cy, inc=50 cy (constant), valid range 32-232
- [x] **L1/L2 cache hints**: most are no-ops; **L2::256B prefetch = 23% faster streaming**

## STILL UNTESTED (next cycles)

- TMA cross-GPU (P2P-mapped remote source)
- Host pinned memory (zero-copy) latency from GPU
- DSMEM proper test (cluster shared memory)
- Cooperative kernel launch overhead
- cudaGraph launch latency
- mxf8f6f4 with proper scale TMEM allocation (would unlock block-scaled FP8 path)
- Sparse FP8 with PROPER 2:4 metadata (might close gap from 7.44 → 9.3 PFLOPS spec)
- mxf4 with proper scale (would unlock FP4 native ~9 PFLOPS)
- tcgen05.cp throughput (smem ↔ TMEM bulk)
- Full GEMM end-to-end (TMA load + MMA + tcgen05.ld + store) timing
- BMMA/xor-popc binary tensor cores
