# B300 Curiosity List V2 — What's Actually Worth Investigating Next

After ~30 ninja experiments + sub-agent critique, here's what I'm
genuinely curious about, prioritized by potential impact and intellectual interest.

## Tier S — Genuinely mysterious / would change my mental model

### S1. [x] RESOLVED — mma.sync hits 90.5% of spec (NOT 23%)
**REFUTED**: prior "570 TFLOPS = 23% of spec" claim was sub-optimal launch geom.

True peak via raw mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32:
  warps/SM=1:  0.23 PFLOPS (9.2%)
  warps/SM=2:  0.46 PFLOPS (18.3%)
  warps/SM=4:  0.91 PFLOPS (36.4%)  -- 1 per SMSP
  warps/SM=8:  1.81 PFLOPS (72.5%)  -- 2 per SMSP
  warps/SM=16: **2.26 PFLOPS (90.5%)** -- 4 per SMSP, SoL
  warps/SM=32: 2.22 PFLOPS (88.8%)  -- 8 per SMSP, slight regress (RF pressure)

ncu confirms: at warps/SM=8, sm__pipe_tensor_cycles_active.sum / cycles = 78%.
Linear scaling 1→4 warps/SM proves each SMSP can issue mma independently.

So path (a) is correct: per-SMSP tensor units exist, but each saturates at
~0.46 mma/SMSP/cycle. Need 4-8 warps per SMSP to fill pipeline.

The 12% / 23% claim was a measurement artifact (low warp count, ILP).
**mma.sync DOES reach NVIDIA's spec.** No need for tcgen05 to hit 90%+ BF16.

Investigated this session, commit `fceb94d`.

### S2. Direct tcgen05 PTX — actually make it compile and run
We tried earlier and failed (alloc/dealloc hung; mma errored). cuBLAS
demonstrably uses it (achieves 4400 TFLOPS FP8). The PTX exists. The
issue was likely missing TMA setup + cluster size + memory descriptor.
Working tcgen05 microbench would let us:
- Verify NVIDIA's published 5000/2500/10000 TFLOPS spec (FP8/BF16/FP4)
- Bypass cuBLAS overhead
- Measure tcgen05 sustained at full power (962W?)

**Test**: write minimal tcgen05.kind::f8f6f4 kernel using TMA + cluster,
measure single-call TFLOPS, compare to cuBLAS.

### S3. [PARTIAL] mma.sync alone CANNOT hit 940W TGP — needs tcgen05
Hypothesis (c) FAILS for legacy mma.sync. Sustained power tests:
  Idle:                       200 W
  HMMA+LDG mixed (1 warp):    305 W   (LDG stalls warps → less power, not more)
  Kitchen sink (4 roles):     280 W   (under-utilization per pipe)
  Pure HMMA 16 warps/SM:      405 W
  HMMA + FFMA same warp:      424 W   (+20W from FFMA pipe)
  cuBLAS BF16/FP8 via cudaGraph: 940 W (tcgen05, prior measurement)

So legacy mma.sync caps at ~425W (≈ 45% of cuBLAS power). The 940W from
cuBLAS REQUIRES tcgen05 (newer per-CTA tensor units that draw much more
power per FLOP). This is consistent with tcgen05 being a major architectural
upgrade, not just a programming convenience.

Path forward (S2): get raw tcgen05.mma compiling so we can measure its power
draw vs mma.sync to quantify the per-FLOP power cost difference.

Investigated this session, commit `affce3d`.

### S3-original. (legacy text preserved)

### S4. [x] RESOLVED — 4 tensor cores per SM ARE per-SMSP independent
Same warp-sweep as S1 directly answers this:
  1 warp/SM (1 SMSP):  0.19 mma/SM/cy
  2 warps/SM (2 SMSPs, 1 each): 0.37 mma/SM/cy (2× linear)
  4 warps/SM (4 SMSPs, 1 each): 0.74 mma/SM/cy (4× linear)
  8 warps/SM (4 SMSPs, 2 each): 1.47 mma/SM/cy (2× over 4 warps)
  16 warps/SM (4 SMSPs, 4 each): 1.84 mma/SM/cy (saturated)

PATH (a) CONFIRMED: 4 independent tensor units, one per SMSP. Each can issue
mma.sync simultaneously and they scale linearly 1→4 with SMSP count.

Per-SMSP tensor unit cap = 0.46 mma/cy (saturated with 4 warps per SMSP for
pipeline depth). Aggregate per SM = 1.84 mma/cy = 4 × per-SMSP.

Investigated this session, commit `fceb94d`.

## Tier A — High value, worth several days each

### A1. FlashAttention-style attention kernel SoL
Real-world: head_dim=128, seqlen=8192, batch×heads=32. Currently we
have softmax (91% of R+W ceiling). FlashAttention adds the QKV GEMMs
and online softmax. What's the achievable SoL?
**Test**: implement minimal FlashAttention v2-style kernel for 1 head;
compare to xFormers/FlashAttention library; find the gap.

### A2. [x] RESOLVED — Fused kernel hits 89% HBM (RMS+bias); GeLU bottleneck

Test: BF16, D=4096, N=1M tokens (8 GB input + 8 GB output = 16 GB traffic).

Memory SoL: 16 GB / 7.31 TB/s = 2.19 ms.

Pure copy at warp-per-row pattern: 2.58 ms = **6.66 TB/s = 91% HBM** ✓

Fused kernel results (1 warp per row, 256 thr/blk, 8 rows/blk, vec uint4):
   8 blocks/SM (default):  RMS+bias = 7.43 ms = 2.31 TB/s (32%) ← reg spill!
   8 blocks/SM:            RMS+GeLU+bias = 9.50 ms = 1.81 TB/s (25%)
   2 blocks/SM:            RMS+bias    = 2.63 ms = **6.54 TB/s (89%)** ← SoL
   2 blocks/SM:            RMS+GeLU+bias = 6.35 ms = 2.71 TB/s (37%)

**KEY FINDING: register pressure was the bottleneck**, not compute or memory.

Each thread stages 16 uint4 (256 B = 64 32-bit registers) for the row.
At 8 blocks/SM × 256 thr = 2048 threads/SM × 64 reg = 128 K registers needed,
but RF cap = 64 K → spill to L1 → 2.8× slowdown (6.54 → 2.31 TB/s).

Lowering occupancy to 2 blocks/SM keeps row in registers, hits 89% HBM SoL.

GeLU still costs 2.4× over RMS+bias (6.35 vs 2.63 ms) due to __expf MUFU.
For inference recipe: fuse RMS+matmul instead of RMS+GeLU separately, since
matmul's compute-bound nature hides GeLU's MUFU cost.

Practical recipe for SoL fused elementwise:
- __launch_bounds__(threads, 2) to keep occupancy low and registers in RF
- Vectorize loads as uint4 (8 BF16 per LDG.128)
- Stage row in registers (avoid SMEM round-trip)
- Use __shfl_xor_sync tree for warp sum (REDUX.f32 not on sm_103)

Investigated this session, commit `e9be3e3`.

### A3. All-Reduce on 2× B300 via NVLink
Catalog has cross-GPU atomic at 1.66 us, NVLink P2P at 783 GB/s.
What's the SoL for an all-reduce of, say, 1 GB BF16 across 2 GPUs?
Useful for tensor-parallel inference.
**Test**: ring/tree all-reduce vs raw cudaMemcpyP2P; measure roundtrip.

### A4. Pipeline-parallel GEMM across 2 GPUs
Split a large GEMM (e.g. 32K × 32K BF16) so half the matrix is on each
GPU. Each does its half-row × full-col, communicates partial result
via NVLink, accumulates. What's the speedup vs single-GPU?

### A5. Verify the BF16 tcgen05 path via cuBLAS algo selection
cuBLAS picks an algorithm via heuristic. Force it to specific algorithms
(sweep via `cublasLtMatmulAlgoCheck`) and identify which IDs use tcgen05
vs legacy mma.sync. Then we know what we're benchmarking.

## Tier B — Interesting curiosities

### B1. [x] RESOLVED — ZERO inter-GPC clock drift
clock64 range across 148 SMs = constant 1.35e9 cy over 50 sequential samples.
SM 0 and SM 147 both ticked exactly 19,603,969 cy over the same window — net
drift = 0. All SMs run from synchronized clock; the 1.35e9 cy offset is a
fixed boot-time constant from staggered GPC group startup.

Implication: cluster-wide algorithms relying on globaltimer or clock64
relative ordering are SAFE — no drift correction needed.

Investigated this session, commit `54aadb2`.

### B2. [x] RESOLVED — YES, L2 viable as tier-2 SHMEM (3.7× BW, 1.7× latency)

BW (best of 5, ~3-10 ms launches with inner-loop reads):
   WS=  8 MB:    20.5 TB/s
   WS= 32 MB:    26.0 TB/s   ← L2 sweet spot
   WS= 64 MB:   **26.7 TB/s** ← peak L2 read BW (3.66× HBM 7.3 TB/s)
   WS= 79 MB:    19.8 TB/s   (persistent cap; partial eviction in 126 MB L2)
   WS=128 MB:    14.2 TB/s   (mixed, near L2 capacity)
   WS=256 MB+:    7.0 TB/s   (pure HBM)

Latency (random pointer-chase, full chain visit):
   L2-resident (≤32 MB): 152 ns / 308 cy  ← constant
   Just over L2 (79 MB):  329 ns           ← THRASHING WORSE THAN HBM
   Pure HBM (≥256 MB):    264 ns / 537 cy

**Persistent attribute provides NO benefit** when L2 retains data naturally
between back-to-back launches (no other workload evicting). Identical BW for
baseline vs persistent at every WS. At 128 MB the persistent attr is actually
WORSE (11.87 vs 14.20 TB/s baseline) — likely tags too much for L2 capacity.

THRASHING WARNING: 79-100 MB working sets have higher latency (329 ns) than
pure HBM (264 ns). The L2 keeps fetching+evicting. AVOID this WS range.

Investigated this session, commit `034e2ff`.

### B3. cuBLAS algo selection — how does it choose?
For N=8192 FP8, cuBLAS picks one algo. For N=512 it picks another.
Where's the crossover? Map out the algo space.

### B4. cudaMemset SASS extraction — final attempt
We BEAT cudaMemset by 1%. Now can we make it leak its kernel?
Try: cuModuleLoadDataEx with a known driver blob, cuFuncGetName on
all internal functions, dlsym hacks.

### B5. [x] RESOLVED — PDL saves launch overhead (~2us/pair), NO SM overlap
HMMA-then-HBM chain (each kernel uses all 148 SMs):
   no PDL:  0.848 ms/pair
   PDL:     0.845 ms/pair    → 0.4% savings (negligible)

When both kernels saturate SMs, PDL provides no overlap. The hardware can't
backfill SMs that are still busy with kernel A.

Short kernel chain (64 thr/blk × 148 blocks, FFMA-only, varying N):
   N=100   no PDL: 4.23 us/pair  PDL: 3.72 us/pair  save 0.51 us = 12%
   N=1000  no PDL: 8.25 us/pair  PDL: 5.88 us/pair  save 2.37 us = 28%
   N=10000 no PDL: 49.2 us/pair  PDL: 45.2 us/pair  save 4.0  us = 8%

PDL only saves the LAUNCH OVERHEAD of B (~2-4 us) by overlapping B's launch
prep with A's tail. Doesn't enable SM-level overlap for full-grid kernels.

WHEN PDL HELPS:
- Chains of short kernels (~5-50 us each) where launch overhead is significant
- Real-world: layer-by-layer transformer inference with small per-layer ops

WHEN PDL DOESN'T HELP:
- Both kernels saturate SMs (no spare SMs for B to backfill)
- Long kernels (>100 us) where launch overhead is negligible

Investigated this session, commit `9e7c592`.

### B6. NVLink-broadcast atomics
Cross-GPU atomic.sys.add to multiple peers via mapped memory. Could
enable hardware-accelerated all-reduce primitives.

## Tier C — Smaller curiosities

### C1. [x] RESOLVED — HBM dominates power (520W ΔMEM vs 200W ΔCOMPUTE)
Decomposition (200W idle baseline subtracted):
   Pure FFMA:       +235 W (compute pipe ALU)
   Pure HMMA:       +205 W (legacy tensor pipe)
   Pure HBM read:   +520 W ← **2.5× compute!**
   cuBLAS at TGP:   +740 W (HBM saturate + tcgen05 overlap)

Memory dominates. The path to TGP is HBM saturation + async-overlapped compute.
Investigated this session, commit `65f3795`. Full table in 16_power_clock.md.

### C2. cudaMallocAsync vs cudaMalloc fragmentation
After many alloc/free, does the pool fragment? Test long-running
sequences.

### C3. [x] RESOLVED — cuBLAS warmup costs 64 ms total before steady-state

For BF16 GEMM M=N=K=4096:
   cublasLtCreate (handle):       1666 us
   cublasLtMatmulAlgoGetHeuristic: **60776 us (60 ms!)** ← biggest cost
   1st cublasLtMatmul call:       1755 us / 79 TFLOPS  (~22× slow)
   2nd cublasLtMatmul call:        94 us / 1602 TFLOPS (18.7× faster than 1st)
   3rd-20th calls:               85-91 us / 1640-1780 TFLOPS (steady)

Steady-state: 1740 TFLOPS BF16 = 70% of NVIDIA's 2500 PF spec at this size.
(Larger M=N=K=8192 reaches higher ~90% via mma.sync; cuBLAS picks per-size.)

WARMUP RECIPE for benchmarks: at least 2-3 iterations of cublasLtMatmul before
timing. The first call is 18× slower (CUDA module load + JIT).

The 60 ms heuristic selection is the BIG cost — done once per (desc,layout)
combination. If sweeping many shapes, expect 60 ms × N_shapes startup time.

Investigated this session, commit `23a363f`.

### C4. [x] RESOLVED — cudaGraph inst ~2 us/node; launch ~0.65 us/node
Linear chain of noop kernels:
   nodes  build_us  inst_us  launch_us  per_node_inst  per_node_launch
       1     1.1     3.8       8.0       3.81           8.0
       4     1.7     9.0      10.8       2.25           2.7
      16     2.4    15.2      19.6       0.95           1.2
      64     7.8    99.0      53.8       1.55           0.84
     256    31.9   433.5     195.1       1.69           0.76
    1024   129.3  1705.7     794.3       1.67           0.78
    4096   600.4  8502.2    2754.1       2.08           0.67
   16384  3303.0 37768.7   10329.3       2.31           0.63

Steady-state:
- Build:    ~0.2-0.3 us per node
- Instantiate: ~1.7-2.3 us per node (LINEAR — biggest cost)
- Launch:   ~0.63-0.78 us per node (vs ~5 us direct dispatch = 7-8× faster)

Amortization point: graph instantiation costs ~2 us/node; saves ~4 us/launch
vs direct dispatch. Pays for itself if graph re-launched ≥1 time.

For 1024-node graph: 1.7 ms inst + 0.79 ms launch = total 2.5 ms first time,
0.79 ms each subsequent launch.

For 16K nodes: 38 ms inst is significant — consider splitting into smaller
graphs or using cudaGraphInstantiateWithFlags for lower-cost paths.

Investigated this session, commit `d80e1f8`.

### C5. Async copy hint actual semantics
cp.async.bulk has multiple variants. Which gives lowest latency
for SHMEM→GMEM streaming?

## Tier D — Methodology / infrastructure

### D1. "Ninja runner" tool
A standard wrapper that:
- Checks `nvidia-smi --query-compute-apps` first; refuses to run if dirty
- Runs `nvidia-smi -rgc` to unlock clock
- Compiles with the right `-arch=` for tcgen05 if needed
- Runs the kernel + ncu + SASS census
- Reports a 3-method-reconciled result with confidence

### D2. Persistent kernel template
Idiomatic, correct persistent kernel pattern: stop_flag in mapped mem,
big inner loop, n_outer counter that's actually correct.

### D3. Sub-agent critique CI
A scheduled job that periodically spawns sub-agents to audit recent
ninja claims. Found 4 over-claims this session; could catch them
automatically.

### D4. Per-precision power/perf table
For each of (FP4, FP8 e4m3, FP8 e5m2, FP16, BF16, FP32, TF32, FP64),
tabulate: peak TFLOPS, sustained TFLOPS, power, TFLOPS/W. Standardized
benchmark suite.

## What I would do FIRST if given a fresh day

1. **S1 (mma.sync vs spec)** — would settle the tensor-core architecture question
2. **S2 (tcgen05 PTX direct)** — biggest potential SoL win; bypasses cuBLAS overhead
3. **A1 (FlashAttention SoL)** — most practically valuable

These three together would give us complete tensor-core visibility on B300.

## Anti-list — things NOT worth doing more

- HBM peak (already at 98.7%)
- SHMEM peak (already at 99.8%)
- Single-GPU FFMA peak (now correct at 97%)
- Single-buffer HBM ceilings (multi-buffer didn't break them)
- TMA bulk for peak throughput (proven slower than v8 inline)

Those are AT SoL. No further effort needed.
