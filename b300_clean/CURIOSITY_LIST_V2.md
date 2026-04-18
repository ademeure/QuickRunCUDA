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

### S3. The 962 W TGP question — what actually hits it?
Catalog says max 1100 W TDP, 962 W typical sustained. cuBLAS via cudaGraph
hit 943 W (Agent 2 verified). What gets us to 1100 W?
Hypotheses: (a) FP4 dense tcgen05 (highest TOPS/inst), (b) Mixed FP8 GEMM +
HBM saturation, (c) Tensor core + SFU + LSU all firing simultaneously.

**Test**: build a kernel that combines (FP8 GEMM via tcgen05) + (FP32 FFMA)
+ (HBM read) on different warps in the same block. Does power approach 1100W?

### S4. Are 4 tensor cores per SM REALLY independent?
On Hopper, the 4 TCs were per-SMSP. On B300, are they:
- (a) 4 independent units, can fire 4 mma.sync simultaneously
- (b) 1 wider unit that processes 4 warp-mma worth of data per cycle
- (c) Something else (tcgen05-specific)

**Test**: launch 4 warps per SM each issuing different mma.sync chains —
measure aggregate TFLOPS/SM. Compare to single warp × 4 chains.

## Tier A — High value, worth several days each

### A1. FlashAttention-style attention kernel SoL
Real-world: head_dim=128, seqlen=8192, batch×heads=32. Currently we
have softmax (91% of R+W ceiling). FlashAttention adds the QKV GEMMs
and online softmax. What's the achievable SoL?
**Test**: implement minimal FlashAttention v2-style kernel for 1 head;
compare to xFormers/FlashAttention library; find the gap.

### A2. RMS norm + GeLU + bias fused — fully fused elementwise kernel
Most inference pipelines have these. SoL = 1R + 1W per element = HBM peak.
Current PyTorch / cuBLAS may not fuse these well. What's achievable as
a custom kernel?
**Test**: BF16 RMS norm 4096 dim + GeLU + bias add, target N=1M tokens.

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

### B1. Inter-GPC clock drift over sustained operation
A1/G1 showed 8 GPC boot groups. Once running for minutes, do GPCs
drift apart? Could affect distributed-cluster algorithm correctness.
**Test**: persistent kernel reading globaltimer per GPC for 5 minutes,
look for skew growth.

### B2. L2 used as "tier-2 SHMEM" via persistent attribute
We have 79 MB persisting region. Can we use it as a 79 MB shared
data store accessible from any SM, faster than HBM? What's the
read latency?

### B3. cuBLAS algo selection — how does it choose?
For N=8192 FP8, cuBLAS picks one algo. For N=512 it picks another.
Where's the crossover? Map out the algo space.

### B4. cudaMemset SASS extraction — final attempt
We BEAT cudaMemset by 1%. Now can we make it leak its kernel?
Try: cuModuleLoadDataEx with a known driver blob, cuFuncGetName on
all internal functions, dlsym hacks.

### B5. PDL (Programmatic Dependent Launch) — meaningful workload
Catalog says PDL gave no measurable speedup. But what about for
HBM-bound workloads where kernel B can prefetch while kernel A finishes?
**Test**: kernel A = compute, kernel B = HBM read for next layer's input.
With PDL, does B's prefetch start while A's tail is still running?

### B6. NVLink-broadcast atomics
Cross-GPU atomic.sys.add to multiple peers via mapped memory. Could
enable hardware-accelerated all-reduce primitives.

## Tier C — Smaller curiosities

### C1. Memory power vs compute power decomposition
NVML reports total board power. Memory + compute. What fraction is
each? Run pure-HBM kernel vs pure-FFMA kernel and compare.

### C2. cudaMallocAsync vs cudaMalloc fragmentation
After many alloc/free, does the pool fragment? Test long-running
sequences.

### C3. cuBLAS warmup characterization
First call to cublasLtMatmul does heuristic selection (~1 ms?).
Second call faster. Quantify the warmup curve.

### C4. cudaGraph instantiation cost vs node count
Graph with 100 nodes vs 1000 vs 10000 — instantiation time scaling.

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
