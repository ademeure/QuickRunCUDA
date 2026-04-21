# M1 — V4 Deep Dive Synthesis Index (2026-04-21)

**135 V4 microbenches completed across 19 categories.** This document is the master index;
detailed findings are in commit messages (use `git show <hash>` per row) and per-task rigor docs.

**System:** NVIDIA B300 SXM6 AC (sm_103a, 148 SMs, 12 HBM3E stacks), CUDA 13.2.

This is a NEW reference. Does NOT replace `B300_PIPE_CATALOG.md` or `B300_TRUE_REFERENCE.md`.

---

## A. SM microarchitecture: issue, scheduler, dependency

| ID | Finding | Hash | Conf |
|---|---|---|---|
| A1 | Single warp = 1 IPC max; FFMA+LOP3 mix only 6.5% overlap | 08ee753 | HIGH |
| A1+ | Multi-warp aggregate scaling: 4 warps/SMSP needed for FFMA peak | 0e63b46 | HIGH |
| A2 | Scheduler is FAIR (≤0.04% variance); SMSP issue port is SHARED across pipes (MUFU+FFMA siblings contend) | 80c2d56 | HIGH |
| A3 | Scoreboard ≥32 slots/warp (no stall) | (M3) | HIGH |
| A4 | FFMA RF port pressure | (M3) | HIGH |
| A6 | Per-pipe instruction reference | (M3) | HIGH |
| A7 | Branch density cost ladder | (M3) | HIGH |

## B. Pipe interleaving + ILP

| B1 | FFMA+IADD3 only 14.2% overlap; FFMA+MUFU ~100% | (M3) | HIGH |
| B2 | FFMA+LDG only 1-12% overlap (LDG too fast) | (M3) | HIGH |
| B3 | MUFU+FFMA = ~100% overlap up to 4 FFMA per MUFU | e6c4227 | HIGH |
| B4 | Tensor+FFMA+IMAD triple-mix 36% overlap | (M3) | HIGH |
| B5 | FFMA+ULDC 70% overlap | (M3) | HIGH |
| B6 | Same-pipe ILP saturates at NC=4 | (M3) | HIGH |
| B7 | Predictable branches FREE behind FFMA latency | (M3) | HIGH |

## C. SASS encoding + immediates

| C1 | 32-bit immediates inline in IMAD/LOP3/FFMA | (M3) | HIGH |
| C2 | IMAD.MOV vs MOV: ptxas folds identity ops; can't isolate | 79534a4 | LOW |
| C3 | LOP3.LUT throughput 14.16 TIPS @ 1500 MHz | (M3) | HIGH |
| C4 | IADD3 vs add.cc carry chain | (M3) | HIGH |
| C5 | BREV @ 3.54 TIPS_inst (XU) | (M3) | HIGH |
| C6 | POPC = FLO = BREV (XU) | (M3) | HIGH |
| C7 | PRMT 14.08 TIPS_inst (ALU peak) | (M3) | HIGH |
| C8 | BMSK/BFE 7.07 TIPS_inst | (M3) | HIGH |
| C9 | SHF 14.12 TIPS_inst | (M3) | HIGH |
| C10 | R2P/P2R = LOP3-class ALU; no special pipe | 6c85d6f | HIGH |

## D. Memory subsystem ninja

| D1 | L2 = pseudo-LRU/hash-based, NOT strict LRU; 2.06× hot ratio asymptote | e1bc97a | HIGH |
| D2 | Per-SM L1 = 128 KB / 1024 lines (sharp boundary); no power-of-2 stride aliasing | 538339e | HIGH |
| D3 | **L2 sector = 32 B; sub-sector writes = 7× DRAM read amp** (MAJOR) | 8c6e5b5 | HIGH |
| D4 | HBM channel bonding (deferred — no per-channel ncu metric) | - | - |
| D5 | SMEM bank rotation under broadcast | (M3) | HIGH |
| D6 | **RF = 2 read ports/cy; .reuse cache = 3rd port; pure 3-source FFMA caps 65%** (MAJOR) | 998e947 | HIGH |
| D7 | TMEM ld x16 = 57 B/cy/warp = 65 TB/s chip-wide | (eff12d4) | HIGH |
| D8 | AGEN pipelined 6-7 cy; address-dep chain = 33 cy LDS latency | f18899b | HIGH |
| D9 | __ldg → LDG.E.CONSTANT (constant cache) | (M3) | HIGH |
| D10 | Stride 128 B = 804 GB/s local minimum (partition concentration) | 4eefc38 | MED |

## E. Atomics & RMW

| E1-E4 | Atomic ladder (SMEM/L2/HBM) | (M3) | HIGH |
| E5 | Atomic per-block 1.8× variance across L2 partitions | d958607 | HIGH |
| E6 | atomicMin/Max FP throughput | (M3) | HIGH |
| E7 | sysmem (host-mapped) atomic = **50× SLOWER** than device | 77bfaf3 | HIGH |
| E8 | Atomic + fence release/acquire SASS | (M3) | HIGH |

## F. Sync primitives

| F1 | Named barriers | (M3) | HIGH |
| F2 | **`__syncwarp(0xFFFFFFFF)` is FREE** (no SASS); partial mask = 7.25 cy BSYNC | 6fdfd70 | HIGH |
| F3 | mbarrier.arrive_drop = atomic decrement+arrive | d4e5529 | HIGH |
| F4 | Cluster barrier = 395 cy floor; no subset variant | 8a6285b | HIGH |
| F5 | cp.async pipe depth = 16 chunks (10.4× gain) | 7e6a992 | HIGH |
| F6 | __syncwarp on convergent warp = 1 CYCLE | (M3) | HIGH |

## G. Compiler / nvcc

| G1 | -Xptxas=-O0 = 4.4× slower (no .reuse, no unroll) | d657f69 | HIGH |
| G2 | **-use_fast_math = 3.28× faster + 4× lower energy** (MAJOR) | 5a5a393 | HIGH |
| G4 | `__restrict__` emits LDG.E.CONSTANT + groups loads (4.4% speedup) | 13c2f2d | HIGH |
| G6 | Default LDG = STRONG.SM (L1); only `.cg` bypasses L1 | 9eb988c | HIGH |
| G7 | **`__noinline__` = 15.2× SLOWER** than `__forceinline__` | 21fa032 | HIGH |
| G8 | LTO -dlto did NOT enable cross-unit inlining in test | ccb9016 | LOW |
| G9 | launch_bounds reg budget: 32-reg = 15.8× spill slowdown | dde07c8 | HIGH |
| G10 | min_blocks > 1 forces register reduction | (M3) | HIGH |

## H. Per-pipe power

| H1 | FFMA = 2.2 pJ/FLOP at 1500 MHz | dedd2b1 | MED |
| H2 | IMAD +37 W (148 SMs) | 2713af5 | MED |
| H3 | MUFU rsqrt.ftz +24 W; sin +39 W | 2713af5 | MED |
| H4 | **LDG +177 W = 5× IMAD** (memory dominates energy) | 2713af5 | HIGH |
| H5 | Branches cost 3.4-4.2× more energy per FFMA | 356f0be | HIGH |
| H6 | Per-SM static = 0.05 W; dynamic = 0.4 W/SM (8-9× ratio) | e642e65 | HIGH |
| H7 | TMEM idle allocation = ZERO power overhead | c245f13 | HIGH |
| H8 | SMEM write +41% hotter than read; vec ops 2.5× scalar | 7b6ec38 | MED |
| H9 | Extra RF read = 0.3 pJ/FFMA; .reuse saves 1.49× energy | b489c02 | MED |
| H10 | Predicate RF essentially free (setp+selp dominate) | 87a7d10 | MED |

## I. Concurrency / HW queues

| I1 | Active CTA limit (partial; need cudaOccupancy API) | 9e31770 | LOW |
| I2 | Dispatch beyond 128 follows TPC-paired ordering | ef8d3d2 | MED |
| I3 | Stream queue depth ~1024 launches | e83d506 | HIGH |
| I4 | ~128 HW dispatch slots; >128 = serialize | 5302086 | HIGH |
| I5 | Preemption granularity (deferred, no user-space hooks) | - | - |
| I6 | **Block scheduler pairs on TPC siblings (+16 GPC stride)** (MAJOR) | 5145766 | HIGH |
| I7 | **`warp_id % 4 = SMSP_id`** confirmed | 3775f32 | HIGH |
| I8 | **Cluster topology: TPCs at +1, GPC rows at +16** | 8f6f6e9 | HIGH |

## J. NVLink / multi-GPU

| J1 | NVLink ping-pong = **1.55 µs one-way / 3.09 µs RT** | 7ac9c8d | HIGH |
| J2 | NVLink R/W symmetric ~740 GB/s = 77% of 956 theoretical | e16901f | HIGH |
| J3 | Concurrent NVLink xfer 1→16 streams = 1.23× scaling | 038fc2c | HIGH |
| J4 | cudaIpcGetMemHandle = 30 ns; cross-process only | 0806958 | HIGH |
| J5 | 2× B300 clocks are FULLY INDEPENDENT | ef95587 | HIGH |

## K. PTX → SASS

| K1 | cvt round-trips (M3) | - | HIGH |
| K2 | mad vs mad.wide (M3) | - | HIGH |
| K3 | selp → SEL ≈ 4 cy (covered by C10) | 6c85d6f | HIGH |
| K4 | NO predicated LDG variant | (M3) | HIGH |
| K5 | STS = 26 cy; STG = 35 cy (all scope modifiers identical) | d394b5d | HIGH |
| K6 | PTX vector ops EMULATED (vmin4 = 44 cy via 2 PRMT) | d05c50b | HIGH |

## L. Driver / runtime

| L1 | **CPU spin on managed flag = 4.39 µs RT** (3.6× faster than cuStreamWaitValue32) | a3632b7 | HIGH |
| L2 | **cuStreamWriteValue32 = 460 ns** (8× FASTER than kernel-write) | 63c19de | HIGH |
| L3 | per-launch driver overhead = 2.05 µs; +58 ns/arg | 91001fe | HIGH |
| L4 | cudaLaunchKernelEx = same 2050 ns as legacy | 87676c9 | HIGH |
| L5 | cudaStreamGetCaptureInfo = 24 ns/call | ad43067 | HIGH |
| L6 | cudaEventQuery PENDING = 120 ns/poll (10× faster than COMPLETED 1254 ns) | 79fea2e | HIGH |
| L7 | **Launch is CPU-bound: 488K kernels/sec single-thread ceiling** | 49e3146 | HIGH |

## M. Numerical exotic

| M1 | LOP3 fuses 3-input boolean to 1 inst (256 truth tables) | b63da88 | HIGH |
| M2 | **@p IADD = IMAD.IADD** (Cluster A) — 2× slower than bare IADD3 (B) | c2c26db | HIGH |
| M7 | bfind/popc/clz/brev = 24 cy chained; bfe 2× faster (12 cy) | 8919bf2 | HIGH |
| M8 | Native add.u64 = 2.81 cy; add.cc+addc 2× slower | e83724c | HIGH |

## N. Cache hierarchy + replacement

| N2 | ncu l1tex_sector_hit_rate calibrated; sharp drop at 1024 lines | 5e452ea | HIGH |
| N3 | cctl L1 ops NOT supported on sm_103a; only `discard.global.L2` works | 512b470 | HIGH |
| N4 | cudaCacheConfigPreferShared has NO effect on B300 (legacy hint) | f41bd9b | HIGH |
| N5 | threadfence latency (M3) | - | HIGH |

## O. Surprises / falsifiable claims

| O1 | STG.NA suppress write (M3) | - | HIGH |
| O2 | tensor warmup (M3) | - | HIGH |
| O3 | branch density (M3) | - | HIGH |
| O4 | wide multiply (M3) | - | HIGH |
| O7 | clock variants (M3) | - | HIGH |

## P. SM-level resource limits

| P1, P2 | Resource limits (M3) | - | HIGH |
| P3 | Max regs/thread = 255 hard cap (silent ptxas warning above) | 0c2d0e7 | HIGH |
| P4 | Spill depth UNBOUNDED — tested 40 MB/thread | 10cbe9c | HIGH |
| P5 | Max LMEM tested up to 1 GB/thread (Hopper+ removed 512 KB cap) | 7774ecd | HIGH |

## Q. Hand-tuned kernel SoL

| Q1 | TMA + cluster multicast = 8× BW savings (prior `f890323`) | f890323 | HIGH |
| Q2 | 32 KB SMEM reduction = 496 cy = 330 ns (redux+SHFL) | 6b70a02 | HIGH |
| Q3 | Warp reduce SHFL chain (M3) | - | HIGH |
| Q4 | Hillis-Steele 1024 prefix = 966 cy (SoL ~200-300 cy estimated) | 5840b3d | LOW |
| Q5 | Bitonic sort 1024 ints = 29398 cy = 19.6 µs | a7e068c | MED |
| Q6 | SHMEM transpose 8.2× speedup (M3) | - | HIGH |

## R. Power oddities

| R1 | Pipe gating threshold (partial; methodology issue) | 741ce96 | LOW |
| R2 | **mbarrier.try_wait 25% LOWER power than spin** (idle pattern) | c30248c | HIGH |
| R3 | SM in BRA loop +0.04 W; EXIT-immediate = 0 W | 4f2c793 | HIGH |
| R4 | Partial-warp predication has NULL power impact | de87c5a | HIGH |

## S. Methodology + tooling

| S1 | clean_run.sh wrapper utility | c6ea321 | HIGH |
| S2 | ncu_explorer.sh utility | 5eb8eb8 | HIGH |
| S3 | sass_count.sh opcode counter | 47b1c6b | HIGH |
| S4 | nvidia-smi power = 33 Hz max; NVML 50 ms cache | 8d57328 | HIGH |

---

## Top "must-know" findings (TL;DR for kernel writers)

1. **`__syncwarp(0xFFFFFFFF)` is FREE** (F2) — no SASS emitted, use freely
2. **L2 sector = 32 B; sub-sector writes = 7× DRAM amp** (D3) — pad scatter writes
3. **RF = 2 read ports/cy; .reuse → 96% peak vs 65% no-reuse** (D6) — broadcast operands
4. **Warp_id % 4 = SMSP_id** (I7) + **TPC = 2 SMs at +1 stride; GPC row at +16** (I8) + **block scheduler pairs on TPC siblings** (I6) — full topology
5. **fast_math = 3.28× faster + 4× lower energy** (G2) — always default for ML/HPC
6. **`__noinline__` = 15.2× slower** (G7) — always `__forceinline__` device helpers
7. **CPU spin on managed flag = 4.39 µs RT** (L1) — best CPU↔GPU signal pattern
8. **mbarrier.try_wait 25% lower power than spin** (R2) — for persistent kernels
9. **Stream queue depth ~1024** (I3) + **~128 HW dispatch slots** (I4) — concurrency limits
10. **NVLink one-way = 1.55 µs / 740 GB/s** (J1, J2) — multi-GPU baseline

---

## Top "surprising" findings

1. **NVRTC harness uses fast_math by default** — all .ftz everywhere unless removed (memory-only)
2. **L1 is per-SM 128 KB / 1024 lines** with hashed indexing (no power-of-2 aliasing)
3. **L2 is pseudo-LRU not strict LRU** — even sub-L2 sweep evicts hot data
4. **mbarrier saves 25% power vs spin** — HW barriers are voltage-aware
5. **PTX vector ops are EMULATED** (vmin4 = 44 cy via 2 PRMT) — no native SASS
6. **TMEM idle = ZERO power cost** — allocate freely
7. **Cluster barrier 28× more expensive than __syncthreads** (395 vs 14 cy)
8. **Sysmem atomic = 50× slower** than device atomic (PCIe round-trip)
9. **cuStreamWriteValue32 = 8× FASTER** than kernel-write for GPU→CPU notify
10. **cudaCacheConfigPreferShared is a NO-OP** on Hopper+ (legacy hint ignored)

---

For per-task rigor docs and detailed analysis, see commit messages via `git show <hash>`
or the per-category .md files (A1_*, A2_*, D2_*, D3_*, D6_*, F2_*, I6_*, I8_*, etc.).
EOF
