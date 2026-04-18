# B300 SXM6 AC — Clean Findings Index

System: 2× NVIDIA B300 SXM6 AC (sm_103a, CC 10.3, 148 SMs/GPU, 2032 MHz boost), AMD EPYC 9575F host, NVLink-5 (NV18) between GPUs, PCIe Gen 6 x16 link (Gen 5 effective rate), CUDA 13.2 runtime, driver 580.126.09.

This directory consolidates ~250 microbenchmark .cu files + 30+ legacy .md docs into 17 category files. Each was synthesized by a dedicated Opus 4.7 agent; this index ties them together.

## Categories

| # | File | Topic | Headline |
|---|------|-------|----------|
| 01 | [hbm_bandwidth](01_hbm_bandwidth.md) | HBM3E DRAM read/write/concurrent | **7.30 TB/s read & write** (95% spec) |
| 02 | [shmem](02_shmem.md) | Shared memory + bank conflicts | **38.4 TB/s pure read** (99.8% spec) |
| 03 | [caches](03_caches.md) | L1, L2, persistent L2, texture/surface | L2=126 MB, persisting≤79 MB; texture obsolete |
| 04 | [fp32_peak](04_fp32_peak.md) | FP32 FFMA peak (CUDA cores) | **74.6 TFLOPS** = 97% of 76.96 TFLOPS theoretical |
| 05 | [fp_precision_nontensor](05_fp_precision_nontensor.md) | FP16/BF16/FP64 non-tensor | FP16/BF16 = FP32 (no 2× packed); FP64 = 1.20 TFLOPS |
| 06 | [tensor_cores](06_tensor_cores.md) | mma.sync, tcgen05, cuBLAS GEMM | FP4 9856 / FP8 4486 / FP16 2259 / TF32 1113 / BF16 mma 577 TFLOPS |
| 07 | [atomics](07_atomics.md) | Atomic op costs, scope, contention | Inc/Dec 4ns, scalar __half 200× slower, scope free |
| 08 | [sync_primitives](08_sync_primitives.md) | syncthreads, cluster, mbarrier, fence | Full latency ladder; **cluster.relaxed 50ns** = 3.7× cluster.sync |
| 09 | [memory_apis](09_memory_apis.md) | cudaMalloc/Memset/cpy, host alloc, VMM | cudaMallocAsync 184-770× faster; **cudaMemset NOT DMA** |
| 10 | [launch_overhead](10_launch_overhead.md) | Kernel launch, graphs, capture, events | 7us launch+sync floor; ExecUpdate 35-77× faster than reinstantiate |
| 11 | [block_scheduling](11_block_scheduling.md) | SMs/GPCs, occupancy, concurrent kernels | 148 SMs, 8 GPCs, 128 dispatch slots |
| 12 | [nvlink_p2p](12_nvlink_p2p.md) | Multi-GPU NVLink + IPC + atomics | **776/1543 GB/s** uni/bidir; 1.55 us cross-GPU atomic latency |
| 13 | [pcie_system](13_pcie_system.md) | PCIe + topology + device attrs | **57.7 GB/s** PCIe (Gen 5 effective); 1100 W TDP |
| 14 | [math_intrinsics](14_math_intrinsics.md) | sin/cos/exp/log/sqrt/div on MUFU | exp2 fastest; rsqrt anomaly = IEEE refinement chain |
| 15 | [integer_bit_ops](15_integer_bit_ops.md) | imad, popc, lop3, prmt, shfl, setp | IADD3 25% faster than LOP3/PRMT; ~17.8 TOPS pipe peak |
| 16 | [power_clock](16_power_clock.md) | TDP, sustained power, clock state | 1100W max; 5.07 TFLOPS/W FP8; **lgc 2032 paradoxically pins 1920** |
| 17 | [nvrtc_module](17_nvrtc_module.md) | NVRTC compile + module load + VMM | Compile floor 5ms; PTX-JIT 155× slower than cubin |

Each file uses the schema: **Headline** → **Optimal recipe** → **Sub-optimal patterns** → **Theoretical** → **Findings retired** → **Open / needs verification**, with HIGH/MED/LOW confidence markers per claim.

## Top-line peak ladder (all HIGH confidence, ncu-verified where possible)

```
Compute (TFLOPS / TOPS)
  FP4 block-scaled tcgen05:   9856 (99% of spec)
  FP8 cuBLAS sustained:       4491 (91%)
  FP16/BF16 cuBLAS:           2259 (91%)
  TF32 cuBLAS:                1113 (90%)
  BF16 mma.sync legacy:        577 (best ILP)
  INT8 mma.sync:               143 (HW-throttled)
  FP32 FFMA (peak recipe):    74.6 (97%)
  FP16/BF16 non-tensor:       58-72 (= FP32, NO packed speedup)
  FP64 DFMA:                  1.20 (1:64)

Memory (GB/s)
  L1-resident (16 MB ws):    46562 = 314 GB/s/SM
  SHMEM pure read peak:      38400 = 99.8% of 38.5 TB/s spec
  L2-resident (64 MB ws):    22965
  HBM read peak (v8 recipe):  7290 = 95.0% spec
  HBM write peak (v8 recipe): 7300 = 95.2% spec
  cudaMemset (D2D):           7510 ≈ HBM peak
  HBM R+W concurrent (FLAG):  needs ncu re-verify (wall-clock 10.4 unreliable)
  D2D same device:            3019
  NVLink P2P unidir:           776 (95% of 900 spec)
  NVLink P2P bidir:           1543 (perfect 2× scaling)
  PCIe H2D pinned:              57.7 (Gen 5 effective on Gen 6 link)

Coordination latency (ns)
  shfl_sync broadcast:           0.85
  __syncwarp:                    1
  __threadfence_block:           8
  __syncthreads (256 thr):      14
  cluster barrier RELAXED:      50    ← NEW finding from rigor audit
  cluster.sync (any size):     185
  __threadfence (device):      385
  cross-block flag wait:       790
  __threadfence_system:        861
  cross-GPU atomic round-trip: 1550 (5× local)
  persistent kernel + mapped:  4000  (best CPU↔GPU)
  cudaStreamWaitValue:         6000
  Stream sync per launch:      7500
  Event-based cross-stream:   27000

Launch / API (us)
  cudaGetLastError:           0.020
  NVTX (no profiler):         0.019
  cudaPointerGetAttributes:   0.05-0.08
  cuModuleGetFunction:        0.04
  Async kernel submit:        1.9
  cuModule load (cubin):       10
  Graph capture (per kernel): 0.2
  Graph instantiate (100 nd): 35
  Graph ExecUpdate:           1.4
  cudaMemcpyAsync submit:     1.2
  cuMemCreate (1 GB):         529
  PrimaryCtxRetain+Release:   240,000  (one-time init!)

Power efficiency (TFLOPS/W)
  FFMA (full occ):            0.21 (74.6 / 361 W)
  BF16 mma.sync:              1.39 (8× FFMA)
  FP8 cuBLAS:                 5.07 (24× FFMA)
```

## Cross-category contradictions / re-verify queue (highest priority first)

These showed up in MULTIPLE agents' "needs verification" sections OR represent
direct contradictions between agents. Listed roughly by importance.

### 🔴 CRITICAL — outright contradictions

1. **Concurrent HBM R+W aggregate** (HBM agent + NVLink agent). Wall-clock test
   gave 10.4 TB/s aggregate (sum of two streams) — but ncu single-kernel R+W
   variant gave 6.74 TB/s aggregate (LOSS, not gain). Need to run ncu
   `dram__bytes_read.sum.per_second + dram__bytes_write.sum.per_second`
   simultaneously during a TRUE concurrent (different streams) R+W run to settle
   whether HBM is partial-duplex or strictly serial. **Until then, neither
   number is trustworthy.**

2. **Tensor + FFMA "8× slowdown"** (corrected by block scheduling agent;
   tensor agent and earlier docs may still claim raw "8× contention"). True
   mechanism is **proportional SM-share scheduling**: smaller block-count kernel
   gets less SM time. Agent 11 has the correct framing; cross-link from agent
   06 (tensor) needed.

3. **PTX-JIT slowdown** — module load agent says "155× slower than cubin",
   launch overhead agent docs the same; but inv 17_launch_latency claims
   different ratios. Reconcile by stating per-kernel-size table.

4. **Cluster cost** — three agents (08, 11, 16) all reference cluster.sync
   ~190 ns. The new finding is `barrier.cluster.arrive.relaxed.aligned + wait`
   = 50 ns (3.7× faster). This MUST be cross-referenced from agent 11
   (block scheduling) which currently says cluster launch is "ZERO overhead"
   without distinguishing launch vs sync.

### 🟡 MEDIUM — needs single new measurement to lock down

5. **HBM 5% gap to spec**. Best measured 7.30 / spec 7.672 = 95.2%. ECC already
   accounted for (7680-bit usable). The remaining 5% gap (refresh? row
   precharge? command bus?) could be distinguished by sweeping access pattern
   row-stride.

6. **Pageable memory asymmetric coherence** (memory APIs agent flagged as
   PRACTICAL BUG): GPU writes propagate to CPU but CPU writes don't invalidate
   GPU's HBM-cached copy. Need a logged repro showing the exact CPU-write →
   GPU-read sequence that returns stale data. Relevant for any code mixing
   `malloc()` buffers with GPU access without `cudaHostRegister`.

7. **Why PCIe Gen 6 link only delivers Gen 5 throughput** (PCIe agent open
   question). Could be (a) AMD EPYC IOMMU rate, (b) DMA engine clock, (c) HW
   actually negotiates Gen 5 internally despite Gen 6 link-status. Won't be
   resolved without a different host or NVIDIA-specific tooling.

8. **cudaMemset wall-clock 7.5 vs ncu 7.30** — 3% gap. Within noise but the
   "fire-and-forget L2 absorption" theory predicted bigger gap. Either the
   theory is overstated for this 4 GB workload, or there's still a small
   real edge to driver kernel.

### 🟢 LOW — nice to have but minor

9. SM ID 142 mystery — block 0 lands on SM 142 (last GPC) before round-robin.
   Why? Hardware scheduler quirk; not perf-relevant.

10. Realistic stmatrix.x4 cost in non-uniform output patterns (SHMEM agent).

11. udiv vs IMAD true latency (rather than throughput) — int+bit ops agent.

12. Hour-scale sustained at 962 W — power agent has tested ≤60 sec, longer-term
    thermal stability untested.

## Findings retired across many docs (master list)

| Old claim | Where it appeared | Why wrong |
|-----------|-------------------|-----------|
| FFMA peak 154 TFLOPS / "256 cores per SM" | Catalog, several investigations | B300 has 128 FP32 cores/SM (like Hopper); 2× formula error |
| HBM write peak 8.5 TB/s | Catalog | Fire-and-forget L2 absorption, not DRAM |
| HBM write peak 6.2 TB/s as ceiling | inv 08_hbm_write.md | Sub-optimal pattern (persistent + grid-stride); v8 + non-persistent hits 7.30 |
| cudaMemset uses DMA / "0-CTA" | Multiple docs | Linear contention sweep proves it uses ALL SMs; just doesn't appear in nsys/ncu kernel list |
| "cudaMemset 17% faster than user kernel" | My recent commits | Wall-clock illusion; both hit ~7.3 TB/s actual DRAM |
| Tensor + FFMA "8× HW contention" | My recent commits | Actually proportional SM-share scheduling; demonstrated by per-stream events + ncu block counts |
| 1M blocks "10% scheduler overhead" | My recent commits | Actually warp gap (sm_active 99.9% but warps_active 90%) when per-block runtime ≤ 5 us |
| TDP 700 W | Multiple docs | nvml: max enforced 1100 W |
| FP8 sustained "53% throttle" | Old measurement | Methodology bug; sustained 4491 TFLOPS for 30+ s, no throttle |
| nvidia-smi -lgc 2032 → 2032 MHz | Many places | Paradoxically pins to 1920 MHz; default boost (no lock) reaches 2032 |
| ld.volatile 1.8× faster than ld.b32 | Catalog | SASS-identical; was a measurement artifact |
| DSMEM "0.8% slower" | Catalog | LICM artifact (compiler hoisted load) |
| 154 GB/s P2P, 286 GB/s P2P kernel | Old catalog | Thread-limited tests; true is 776 GB/s |
| NV18 = "NVLink generation 18" | Naming confusion | NV18 = 18 NVLink-5 lanes |
| WriteCombined helps GPU H2D | Old assumption | No measurable benefit on B300 + EPYC |
| Native CPU↔GPU atomics | Assumption | hostNativeAtomicSupported = 0 on this system |
| Pageable malloc → 1.5 TB/s | Recent commits | Was GPU-cache effect, not real PCIe BW (which is 57 GB/s) |
| 256 GB/s PCIe Gen 6 throughput | nvml-derived | Effective rate is ~57.7 GB/s (= PCIe Gen 5) |
| FFMA latency 23 cycles | Old | Included loop overhead; true is 4 cycles |
| Self-op FFMA 2× slower than diff-src | Volta-era folklore | No register-port penalty on Blackwell; 4.02 cy each |
| match_any same as shfl | Some docs | match_any is 38× slower (375 cy vs ~10 cy) |
| WaitValue "3 us faster than event sync" steady-state | My commit | True only for host-call cost; full cross-stream pair equivalent |
| Cooperative launch +32 ns | Old | Actual ~1.5 us/launch + 50 us first-time setup |
| BlockingSync 5-7× slower steady-state | My commit | Steady-state 25% slower (10.91 vs 8.50 us); 5-7× is short-kernel only |

## How to use this directory

If you have a question about B300, start with the relevant category file. The
old `B300_PIPE_CATALOG.md` (19,742 lines) is partially superseded — sections
that contradict the cleaned files should be assumed wrong unless a clean file
explicitly defers to it. The clean files cite the catalog where it's still
authoritative (e.g., the SASS opcode → pipe classification in catalog §8 is
still the canonical reference; `F2FP_DEEP_DIVE.md` remains canonical for
narrow-format conversions).

For new measurements, start by checking the "Open / needs verification" section
of the relevant file before adding to the existing chaos.

For the rigor methodology used, see `CLAUDE.md` (the "B300 BENCHMARKING
METHODOLOGY" section).
