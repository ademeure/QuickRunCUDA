# M7: V5 Curiosity List Synthesis (B300 sm_103a)

Master synthesis of 66 V5 microbench commits. Each finding cites commit hash.

---

## §1 — mbarrier deep ninja (A-series)

| ID | Finding | Commit |
|----|---------|--------|
| A1 | `mbarrier.try_wait` HINT honored. HW caps suspend at ~1 µs regardless of hint. Sweet spot: HINT 1000-10000 ns | `acd3f46` |
| A2 | mbarrier.try_wait saves **20 W** (-10.4%) vs busy spin. Same as `__nanosleep`. Spin uses 2.9× more dynamic power | `9b2b151` |
| A3 | Legacy cp.async.cg + commit + wait_all = 314 cy/iter. expect_tx pattern needs cuda::pipeline | `38e5772` |
| A4 | Multi-mbarrier per CTA scales perfectly: 1 bar = 24 cy, 64 bars = 26 cy. cuTLASS-style 64+ barriers FREE | `cc5f61a` |
| A5 | Cluster mbarrier `arrive.shared::cluster` works (sink dest required); `try_wait.parity.shared::cluster` **ILLEGAL** | `da371a4` |
| A6 | mbarrier.arrive without wait = 24 cy (20% cheaper than `__syncthreads` 30 cy) | `a549d92` |

**Sync hierarchy** (cycles, fastest to slowest): warp(full)=0 → warp(partial)=7 → arrive=24 → bar.sync=30 → cluster.barrier=395

---

## §2 — HMMA + tcgen05 concurrency (B-series)

| ID | Finding | Commit |
|----|---------|--------|
| B1 | tcgen05.mma PTX compiles; needs valid 64-bit SMEM descriptors (zero/garbage = illegal-instruction err 0715). Indirect: shares tensor pipe with HMMA per B2 | `7c9ac83` |
| B2 | HMMA + tcgen05.ld (LDTM) overlap = **28%** (vs 73% for HMMA + LDS). Tensor sequencer / RF ports compete | `b6648e2` |
| B3 | tcgen05.cp PTX requires SMEM descriptor; raw addr = "Arguments mismatch" | `71971e9` |
| B4 | **WGMMA NOT supported on B300** — ptxas explicitly rejects on sm_103a. Port to mma.sync or tcgen05.mma | `ee437a1` |
| B5 | mma.sync = HMMA. ILP saturates at ~10 cy/mma at ILP=8 (chain 27→17.5→12.75→10.375). 2.6× speedup | `08ee753` |
| B6 | HMMA + LDS overlap = **73%**. Tensor pipe and LSU pipe ARE independent | `e07621d` |

**Take-home:** Tensor pipe is a shared sequencer. HMMA + tensor-related ops (LDTM/STTM/MMA) compete; HMMA + non-tensor (LDS/LDG/FFMA) overlap well.

---

## §3 — DSMEM + cluster (C-series)

| ID | Finding | Commit |
|----|---------|--------|
| C1 | DSMEM peer access = 214 cy = 3.96× slower than local SMEM (54 cy). Still 4× faster than DRAM | `9c8fec8` |
| C2 | cluster.barrier latency is **FLAT O(1)** for CSIZE = 2-8 (~390 cy ± 7%). HW tree-style sync | `1f193aa` |
| C3 | **NO dedicated cross-cluster atomic SASS exists.** atom.shared::cluster → `ATOM.E.ADD.STRONG.GPU` (same as global) | `1356dcb` |
| C4 | Cluster mbarrier supports SUBSET (K-of-N) semantics. Arrive cost grows non-linearly: 1 CTA = 53 cy, 7-8 = 182 cy (HW saturation) | `bdf02fd` |
| C5 | Cluster MAX = **8** on B300 (CSIZE=16 silently fails) | `26a5f6a` |

**Take-home:** Build clusters of 8 freely (no latency penalty). Use cluster.barrier for symmetric sync; use mbarrier in DSMEM only for asymmetric K-of-N (where K is small).

---

## §4 — Power deep ninja (D-series)

| ID | Finding | Commit |
|----|---------|--------|
| D1 | Subnormal FFMA at FULL speed (4.11 cy/fma). NO penalty unlike CPU 100× trap | `0bd3802` |
| D2 | Per-pipe duty cycle: FFMA pipe power scales 7→24→85→123 W as DUTY 1→4→16→64. Saturates at +124 W. **4.4 pJ/FFMA** | `2b5f451` |
| D3 | Heavy FFMA+MUFU @ 220 W (20% TDP) does NOT throttle | `bd20df5` |
| D4 | **DVS confirmed: V scales QUADRATIC with clock.** P/f grew 3.2× from 510→1920 MHz. pJ/FFMA: 3.1 → 9.96 (3.2× more energy at boost) | `42bfa01` |
| D5 | Default LDG.E (L1 cached) is BOTH faster and **lower power** than .cg (bypass L1). +30 W vs +56 W (.cg uses 87% more) | `f18345b` |
| D6 | Per-SM power isolation: idle independent (0.05 W each); under TDP pressure all share clock domain | (ref H6+D3) |

**Take-home:** Energy-bound workloads benefit from clock-down (DVFS); throughput-bound want max clock. Use mbarrier/__nanosleep for waits >1 µs to save ~20 W per SM.

---

## §5 — Bizarre PTX corners (E-series)

| ID | Finding | Commit |
|----|---------|--------|
| E1 | `prefetchu.L1` emits CCTL.E.PF1. Back-to-back prefetch+load = 1.2% speedup; needs K-iters-ahead pipeline | `ab6fbbc` |
| E2 | `applypriority`: only `::evict_normal` works (CCTL.E.DML2). `::evict_last/first` = ptxas error | `b0ada56` |
| E3 | `griddepcontrol`: launch_dependents → PREEXIT (+1 cy); wait → ACQBULK (free standalone) | `5e57cf8` |
| E4 | `bar.warp` SYNC-ONLY on B300. `bar.warp.arrive/wait` = ptxas error | `6f7c49f` |
| E5 | `nanosleep`: min 62 ns; sweet spot 500-1000 ns (<2% error); AVOID 5-10 µs (1.63× overshoot); ≥1 ms accurate within 5% | `eb6efce` |
| E6 | `%cluster_ctarank` works in BOTH cluster + non-cluster (size-1 cluster) | `c152b25` |

---

## §6 — ncu metric exotica (F-series)

| ID | Finding | Commit |
|----|---------|--------|
| F1 | **Two FFMA sub-pipes**: pipe_fmaheavy 50% + pipe_fmalite 50%. Pipe taxonomy: alu/fma/fmaheavy/fmalite/xu/lsu/tensor/adu/cbu/fp64 | `948cf45` |
| F2 | Scheduler stall metrics: `smsp__warps_active.avg.per_cycle_elapsed`, `smsp__warps_eligible.avg.per_cycle_elapsed`, `smsp__inst_issued.sum` | `146a96f` |
| F3 | Memory throughput pyramid: l1tex 78.85% → lts 62.04% → dram 44.61% via `*.throughput.avg.pct_of_peak_sustained_elapsed` | `e0f896d` |
| F4 | Spill detection: `launch__registers_per_thread` + `l1tex__t_requests_pipe_lsu_mem_local_op_{ld,st}.sum` | `282e941` |
| F5 | Cluster: `gpc__cgas_*`. DSMEM: `l1tex__data_pipe_lsu_wavefronts_mem_lgds`. **TMEM: `sm__mem_tensor_{reads,writes}_op_{ldt,utcmma_*,utcshift,stt,utccp}`** | `98ef5c0` |

---

## §7 — Real-world micro-kernel SoL (G-series)

| Kernel | Single-block 256-thread cy/elem (1024 elements) | Commit |
|--------|---------|--------|
| **Argmax** | 441 cy = **294 ns** (0.43 cy/elem) | `f168e50` |
| **RMSNorm** | 588 cy = 392 ns (0.57 cy/elem) | `470ac9c` |
| **Prefix scan SHFL** | 523 cy = 349 ns (0.51 cy/elem; 1.85× faster than H-S) | `2f3c7b3` |
| **SoftMax** | 899 cy = 599 ns (0.88 cy/elem; ex2.approx.ftz fast path) | `1221b47` |
| **LayerNorm** | 997 cy = 665 ns (0.97 cy/elem; 1.7× slower than RMSNorm) | `ff749ae` |
| **Top-K (K=4)** | 1858 cy = 1239 ns (4× argmax with masking) | `ee01e84` |

---

## §8 — Driver/runtime sub-microsecond (H-series)

| ID | Finding | Commit |
|----|---------|--------|
| H1 | cudaGraphExecKernelNodeSetParams = **0.4 ns** (FREE!). Re-instantiate = 7.25 µs (3.5× slower) | `c734603` |
| H2 | cudaGraphLaunch: 1-2 streams = 2049 ns/launch; 4+ streams DOUBLES to 4094 ns/launch | `6b5aebd` |
| H3 | cudaMallocAsync = **200× FASTER** than cudaMalloc (0.33 µs vs 66 µs). Always use Async | `a46e764` |
| H4 | cudaMallocFromPoolAsync = **O(1) at 0.32 µs** regardless of size (64 B to 16 MB) | `842065e` |
| H5 | cudaSetDevice(0) cold start = 474 ms. NEVER fork for short GPU work | `19b48e6` |
| H6 | LAZY module load default. EAGER saves 40 µs first-kernel | `fc800a3` |

---

## §9 — Multi-GPU patterns (I-series)

| ID | Finding | Commit |
|----|---------|--------|
| I1 | IPC handles cross-process: cudaIpcGetMemHandle = 5-7 µs; cudaIpcOpenMemHandle = **53.6 µs** one-time. Bidirectional R/W verified | `c9179d6` |
| I2 | NVLink WRITE BW: kernel-direct = **714 GB/s** peak (96% of cudaMemcpyPeer 749). Saturates at just 32 blocks | `9bdd026` |
| I3 | 2-GPU all-reduce naive = 80.8 GB/s (10× below NVLink peak; needs vec4 + tiling) | `41ba869` |
| I4 | Cross-GPU atomic = 0.54 Gatomic/s = 3.2× slower than local HBM atomic. Use NCCL | `4005673` |
| I5 | UVA alone INSUFFICIENT — need cudaDeviceEnablePeerAccess for direct peer access | `fd12cdc` |

---

## §10 — Numerical precision exotica (J-series)

| ID | Finding | Commit |
|----|---------|--------|
| J1 | All 4 rounding modes (rn/rz/rm/rp) = identical 4.438 cy/FFMA. FREE on B300 | `58dc7c3` |
| J2 | B300 fully preserves subnormal FFMA output when `-ftz=false` (IEEE) | `e686f90` |
| J4 | TF32 ULP = **8192× FP32 ULP** (matches 13-bit mantissa diff). 4× speedup but 8000× precision loss | `c2f5bae` |
| J5 | Block reduction precision: serial err 1023 → pairwise err 7 (146× better) → Kahan err 1 (**1023× better**) | `d844c1b` |

---

## §11 — Async + persistent kernel patterns (K-series)

| ID | Finding | Commit |
|----|---------|--------|
| K1 | Persistent kernel beats per-launch only when work ≥ launch overhead (~5 µs). For >50 µs work: persistent + batched signals wins | `41e225f` |
| K2 | Producer-consumer DSMEM round-trip = **468 cy = 312 ns** (cluster=2). 33× more than `__syncthreads` but 5× faster than NVLink | `ece9976` |
| K3 | Dynamic parallelism = 12.36 µs/launch (6× slower than host launch). CDP2 removed device-side sync | `cb27978` |
| K4 | CUDA Graph re-launch = **0.64 µs/kernel = 4.4× faster** than direct (2.79 µs). Break-even ~113 launches | `5dfdcb2` |
| K5 | 2 streams parallel = **1.97× speedup**. Event-wait dependent chain ≈ same (pipelining hides 1 µs event vs 1.5 ms kernel) | `68c4e53` |

---

## §12 — Tooling (L-series)

| Tool | Path | Commit |
|------|------|--------|
| Auto-rigor wrapper | `utils/auto_rigor.sh` | `3f7c64c` |
| Power sampler (NVML) | `utils/power_sampler.cpp` (6483 Hz) | `8bb15c8` |
| Per-pipe dashboard | `utils/pipe_dashboard.sh` | `272ae48` |
| SASS diff | `utils/sass_diff.sh` | `cb4c9f4` |
| Microbench template | `utils/mkbench.sh` | `9947490` |

---

## Architectural summary

**Pipes (4 SMSPs per SM):**
- fma (parent of fmaheavy + fmalite — 2 sub-pipes)
- alu (INT)
- xu (SFU/MUFU)
- lsu (LD/ST global+shared)
- tensor (HMMA + tcgen05 sequencer — SHARED)
- adu, cbu, fp64, tex (specialized)

**Pipe overlap matrix (single warp, B300):**
- HMMA + LDS = 73% overlap (tensor + LSU independent)
- HMMA + LDTM = 28% overlap (BOTH tensor pipe — competition)
- HMMA + FFMA = should be ~90%+ (tensor + fma independent — to verify)

**Key constants:**
- Cluster MAX = 8 CTAs
- Cluster.barrier = 390 cy O(1) up to 8
- mbarrier arrive = 24 cy (cheaper than `__syncthreads` 30)
- 4.4 pJ/FFMA at 1500 MHz; scales V² with clock
- 4 IPC (instructions per cycle) per SMSP achievable for FFMA at ILP=8

**Power model:**
- Idle ≈ 165 W
- FFMA-saturated +124 W (sustains 2032 MHz boost)
- TDP throttle requires 800+ W (NVFP4 cuTLASS gets there)
- mbarrier wait saves 20 W vs spin

---

## V6 candidates (not yet investigated)

These emerged DURING V5 work and warrant V6:

1. **Full tcgen05.mma with valid descriptors** — needs CUTLASS reference
2. **HMMA + FFMA simultaneous** — tensor + fma overlap (predicted 90%+)
3. **LDG + LDTM overlap** — both load ops, different pipes
4. **DVS + workload-specific power optima** — sweep clock per workload to find min-energy point
5. **cluster mbarrier with multiple host CTAs** — alternative to single-host pattern
6. **mbarrier in TMEM?** — does TMEM addressing work for mbarrier objects?
7. **tcgen05 power signature** — does it match HMMA's pJ/op or different?
8. **Cross-process IPC + UVA event sharing** (cudaIpcGetEventHandle)
9. **tcgen05.cp DMA bandwidth peak** (vs cp.async.bulk peak)
10. **Per-pipe latency tomography** — full latency matrix between all 11 pipes
