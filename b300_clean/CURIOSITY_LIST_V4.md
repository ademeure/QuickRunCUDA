# B300 Curiosity List V4 — Ninja Microarchitecture Edition (2026-04-20)

Built from scratch. **Focus: low-level GPU/CUDA microarchitecture, ninja
optimization, surprising HW behavior. Avoid LLM/framework-level work.**

Mix of curiosity-driven mysteries and high-value optimization recipes.
Use sub-agents (Plan / Explore / general-purpose) for parallel research.

---

## A. SM microarchitecture: issue, scheduler, dependency

- [ ] **A1 — Per-SMSP issue width per cycle**: build SASS with mixed FMA + IMAD
  + LDG + BRA in same warp; measure how many issue per cycle. Find dual-issue
  conditions vs serial.
- [ ] **A2 — Warp scheduler policy under contention**: 8 warps/SMSP, all ready
  — does scheduler round-robin, LRU, oldest-first? Measure starvation.
- [x] **A3 — Scoreboard slot count**: ≥32 slots per warp (no stall
  observed at N=32; cy/load decreases monotonically). L1 latency=26 cy.
  See `b300_clean/A3_SCOREBOARD_DEPTH.md`.
- [x] **A4 — Register read port count**: **FFMA RF has 2 read ports**;
  3 distinct sources = 0.61/SMSP/cy vs 1-2 sources = 0.97/SMSP/cy (37% slower).
  LOP3/ALU pipe at 0.5/SMSP/cy peak doesn't show this (pipe-bound first).
  See `b300_clean/A4_FFMA_PORT_PRESSURE.md`.
- [ ] **A5 — Predicate register file**: how many predicates can be live?
  Push past 7 to see spill behavior.
- [x] **A6 — Per-pipe latency table**: measured for 13 op types at 1500 MHz.
  ALU pipe (0.5/SMSP/cy): LOP3/IADD3/SHF/PRMT/BFI. XU (0.125/cy): BREV/
  POPC/CLZ. LSU (0.25/cy): SHFL. FFMA 0.66/cy at 2 warps (→ 0.98 at 4+).
  See `b300_clean/A6_PER_PIPE_REFERENCE.md`. Commit: this batch.
- [x] **A7 — Active mask transition cost**: even if(true) costs +8 cy
  (compiler emits BSSY/BSYNC). Half-warp divergence: +2.5 cy. Data-dep
  divergence: +13 cy. Branch cost is NOT free even when predictable.
  Commit history.
- [x] **A8 — SETP throughput** (with caveat): FSETP+SELP at ALU peak
  (0.49/SMSP/cy). ISETP+SELP slightly lower (~0.33/inst). nvcc PTX→SASS
  fusion makes per-inst rate hard to isolate cleanly.

## B. Pipe interleaving + ILP at pipe-level

- [x] **B1 — FFMA + IMAD parallel issue** (partial — IADD3 used): mixed
  FFMA+IADD3 only 14.2% overlap (unified cluster). FFMA+SHFL 14.7%.
  FFMA+MUFU ~100% (per commit 8012b98). Issue-port duration > pipe diversity.
  See `b300_clean/A6_PER_PIPE_REFERENCE.md` + `b300_clean/B1_DUAL_ISSUE_FFMA_IADD3.md`.
- [x] **B2 — FFMA + LDG parallel issue**: only **1% overlap with chain**
  dep, 12% without. NOT the classic free overlap people assume. LDG too
  fast (16 cy/inst) to leave dispatch slots for FFMA. Only slow ops
  (MUFU) get the 100% overlap. See `b300_clean/B2_FFMA_LDG_DUAL.md`.
- [ ] **B3 — MUFU + FFMA**: catalog says yes; verify and measure overlap.
- [x] **B4 — Tensor + FFMA + IMAD all simultaneous**: 13-36% overlap
  (NOT 100%). Issue port serializes even tensor pipe. Triple-mix
  better than pair (36% > 13-22%) — long MMA window has room for
  2 scalar streams. Commit history.
- [ ] **B5 — FFMA + ULDC** (uniform datapath): does ULDC steal an issue slot?
- [ ] **B6 — Same-pipe ILP**: 2 independent FFMAs in one issue slot? (Probably no).
- [ ] **B7 — Branch + compute parallel**: cost of BRA when fully predictable
  vs not.

## C. SASS instruction encoding + immediate forms

- [x] **C1 — Maximum immediate width**: B300 SASS embeds **up to 32-bit
  immediates directly** in IMAD, LOP3, IMNMX, FFMA. No ULDC fallback
  observed. FFMA emits FP32 literal inline (e.g. 4.295e+09).
  Compiler folds dead IMNMX into LOP3 XOR. Commit history.
- [ ] **C2 — `IMAD.MOV` (mov via IMAD)**: when nvcc uses it, throughput
  benefit vs `MOV`.
- [x] **C3 — `LOP3.LUT`**: 256 truth tables. Throughput, latency, hot-path tricks.
  → 22b06b3: imm-INDEPENDENT (12 imms 14.08±0.04% TIOPS); lat 4.5 cy;
    0.5/SMSP/cy = 14.16 TIOPS @ 1500 MHz; 3 unique reads = no port penalty.
    See `b300_clean/C3_LOP3_LUT_DEEP.md`.
- [x] **C4 — `IADD3` with predicate output**: carry chain (add.cc+addc)
  is 42% slower than plain IADD3; IADD3+setp+@p is 2.3× slower; IMAD.IADD
  21% faster (fewer SASS inst). Commit history.
- [x] **C5 — `BREV`**: 3.54 TIPS_inst at 1500 MHz = 0.125/SMSP/cy (XU pipe, 8 cy/inst). See A6.
- [x] **C6 — `POPC` / `FLO`**: 3.54 / 3.53 TIPS_inst = same as BREV (XU pipe). See A6.
- [x] **C7 — `PRMT`**: 14.08 TIPS_inst = 0.5/SMSP/cy (ALU pipe peak). See A6.
- [x] **C8 — `BMSK`/BFE**: 7.07 TIPS_inst = 0.25/SMSP/cy (XU sub-pipe or ALU-half). See A6.
- [x] **C9 — `SHF`**: 14.12 TIPS_inst = 0.5/SMSP/cy (ALU pipe peak, L and R identical). See A6.
- [ ] **C10 — `R2P` / `P2R`**: predicate-to-register conversion cost.

## D. Memory subsystem ninja

- [ ] **D1 — L2 replacement policy**: LRU? Pseudo-LRU? Hash of {addr, time}?
  Test by walk patterns + measuring evictions via ncu.
- [ ] **D2 — L1 cache associativity**: test by allocating exact-size aliased
  tiles, watch for conflict misses.
- [ ] **D3 — L2 sector vs line size mismatch**: write 1 sector of a 4-sector
  line; observe read amplification.
- [ ] **D4 — HBM channel bonding granularity**: which address bits select
  which HBM stack? `dram__bytes.per_dram` per-stack metric sweep.
- [x] **D5 — SHMEM bank rotation under broadcast**: ALWAYS — broadcast
  is faster than distinct (13 vs 14.6 cy). N-way partial broadcasts
  free. Stride-32 (32-way conflict) = 5.7×, not 32×. Skewed stride-33
  avoids conflict. See `b300_clean/D5_SMEM_BANK_BEHAVIOR.md`.
- [ ] **D6 — Register file bandwidth per cycle per SMSP**: max FMA chain
  with 4 unique sources per inst — RF reads ≤ 12/cycle?
- [ ] **D7 — TMEM bandwidth**: load/store TMEM throughput, distinct from
  GMEM/SMEM.
- [ ] **D8 — Address generation pipeline depth**: measurable stall when
  address-bound vs compute-bound.
- [x] **D9 — `__ldg` vs `ld.global.ca` SASS**: NOT THE SAME. `__ldg` →
  `LDG.E.CONSTANT` (constant cache); `ld.global.ca` → `LDG.E.STRONG.SM`.
  Use `__ldg` for read-only data.
  See `b300_clean/D9_E4_LDG_ATOM_SASS.md`.
- [ ] **D10 — L2 partitioning across HBM channels**: which L2 partition serves
  which HBM stack? Per-partition ncu metrics.
- [x] **D11 — Cache line size inference**: effective DRAM transaction
  granularity ≈ **256 B** (plateau in stride sweep starts at 256 B).
  Likely HBM3E burst length = 2× L2 sector. Plan layouts for 256 B
  alignment. Commit history.

## E. Atomics & RMW edge cases

- [ ] **E1 — atomicAdd misaligned (e.g. ½ word offset)**: error or split?
- [ ] **E2 — atomicAdd b16 packed `__half`** vs scalar throughput.
- [x] **E3 — atomicCAS contention scaling**: aggregate **FLAT at 1.1
  successful CAS/μs** for any N threads (1-128). Each CAS = ~900 ns
  ≈ 1380 cy unavoidable. Linear slowdown per thread under contention.
  Commit history.
- [x] **E4 — `red` vs `atom` SASS**: DIFFERENT opcodes. `atom` →
  `ATOMG.E.ADD.STRONG.GPU` (returns old); `red` → `REDG.E.ADD.STRONG.GPU`
  (no return). `red.relaxed` SASS-identical to `red.global`.
  See `b300_clean/D9_E4_LDG_ATOM_SASS.md`.
- [ ] **E5 — Atomic across L2 partitions**: latency penalty when address
  hashes to "far" partition.
- [x] **E6 — atomicMin/Max FP throughput**: int min/max 10.5 Gops/s
  (33% FASTER than atomicAdd float 7.9 Gops/s). atomicCAS-based min
  2.4× slower than native atomicMin. half atomic ≈ float (no benefit).
  Commit history.
- [ ] **E7 — sysmem atomics**: cudaAtomic on host-mapped memory.
- [ ] **E8 — atomic w/ fence release/acquire** SASS effect.

## F. Sync primitive deep ninja

- [x] **F1 — `bar.sync 0..15`**: 16 named barriers ARE independent
  resources (~31 cy each for 128 threads). Different IDs don't share.
  Cost adds linearly when multiple per iter. Useful for SW pipelines.
  Commit history.
- [ ] **F2 — `bar.warp.sync` arbitrary mask** vs `__syncwarp(0xFFFFFFFF)`.
- [ ] **F3 — `mbarrier.arrive_drop`** semantic: when does it actually drop?
- [ ] **F4 — Cluster barrier with subset of CTAs**: can you have 2/4 CTA
  participate vs 4/4? Latency.
- [ ] **F5 — Async transaction barriers** (mbarrier + cp.async): pipe depth.
- [x] **F6 — `__syncwarp` cycle cost vs no-op**: **1 CYCLE** on convergent
  warp. Use liberally. membar.cta = 6 cy (heaviest fence).
  See `b300_clean/F6_SYNCWARP_COST.md`.

## G. Compiler / nvcc behavior

- [ ] **G1 — `-O0` vs `-O3` SASS divergence**: exact areas where
  optimization matters most.
- [ ] **G2 — `-use_fast_math` impact on POWER** (not just speed).
- [ ] **G3 — `__builtin_assume` impact on SASS** for varied assumptions.
- [ ] **G4 — `__restrict__` impact on real schedules**.
- [ ] **G5 — Loop unrolling thresholds** (compiler default vs explicit).
- [ ] **G6 — `-dlcm=ca/cg/cs` default cache mode** behavior.
- [ ] **G7 — `__forceinline__` vs LTO link-time inline**.
- [ ] **G8 — Whole-program optimization** with separate compilation.
- [ ] **G9 — PTX `.maxnreg` directive** effect.
- [ ] **G10 — `__launch_bounds__` exact impact** on register allocation.

## H. Per-pipe power (sub-tcgen05)

- [ ] **H1 — FFMA pipe power per inst**: pure FFMA loop, varying ILP/density.
- [ ] **H2 — IMAD pipe power per inst**: same.
- [ ] **H3 — MUFU pipe power**: ex2/log2/sin/cos/rcp/rsqrt per inst.
- [ ] **H4 — LDG pipe power per inst** (separate from cache subsystem).
- [ ] **H5 — Branch pipe power**: BRA cost.
- [ ] **H6 — Idle SM static power** (subtract from min-active).
- [ ] **H7 — TMEM idle power** when allocated but unused.
- [ ] **H8 — SHMEM read vs write power per byte**.
- [ ] **H9 — Register file power** per RF access.
- [ ] **H10 — Predicate register file power**.

## I. Concurrency / hardware queues

- [ ] **I1 — Active CTA limit per SM** at varying register/SHMEM use.
- [ ] **I2 — Concurrent kernel slot 128 fairness**: when N>128, FIFO or hash?
- [ ] **I3 — Stream queue depth** (host-side enqueue limit).
- [ ] **I4 — Hyperqueue / HW queue count** observation.
- [ ] **I5 — Preemption granularity**: at what SASS instruction boundary
  can a kernel be interrupted? Measure latency.
- [ ] **I6 — TPC-level vs SM-level scheduling**: blocks of same kernel
  prefer same TPC?
- [ ] **I7 — Warp slot allocation policy**: round-robin across SMSPs?
- [ ] **I8 — Cluster handle assignment**: which SMs get clustered for
  `__cluster_dims__(2,1,1)` — adjacent SMs? Per-GPC?

## J. NVLink / multi-GPU low-level

- [ ] **J1 — NVLink raw packet latency** (via custom round-trip kernel).
- [ ] **J2 — NVLink read vs write asymmetry** per-link bandwidth.
- [ ] **J3 — NVLink with multiple in-flight ops** (transactions).
- [ ] **J4 — Cross-GPU SHMEM access** via cudaIpcMemHandle.
- [ ] **J5 — Multi-GPU clock independence verification** (we have one data
  point; build a stronger test).

## K. PTX → SASS translation

- [ ] **K1 — `cvt` chains**: when does PTX cvt sequence become single SASS inst?
- [ ] **K2 — `mad` vs `mad.wide`** SASS encoding differences.
- [ ] **K3 — `selp`** translation (predicate select).
- [ ] **K4 — `setp` followed by `@p ld`** combine into LD with predicate?
- [ ] **K5 — `st.shared` vs `st.global` SASS encoding family**.
- [ ] **K6 — `vshl/vshr` (vector shift)** PTX → SASS.

## L. Driver / runtime ninja

- [ ] **L1 — `cuStreamWaitValue32` latency** vs CPU-side spin.
- [ ] **L2 — `cuStreamWriteValue32` latency** vs kernel-write.
- [ ] **L3 — Driver-side queue dispatch overhead** per kernel launch.
- [ ] **L4 — `cuLaunchKernelEx` vs `cuLaunchKernel` perf delta**.
- [ ] **L5 — `cudaStreamGetCaptureInfo` cost while idle vs capturing**.
- [ ] **L6 — `cudaEventQuery` polling overhead**.
- [ ] **L7 — Driver thread CPU cost** under heavy kernel-launch load.

## M. Numerical exotic

- [ ] **M1 — `LOP3.LUT` for fused boolean ops**: how many ops can collapse
  into one LOP3?
- [ ] **M2 — `IADD3` + predicate** for branchless code patterns.
- [x] **M3 — `IMNMX` (min/max) throughput**: 0.99/SMSP/cy (28 TIPS_inst)
  for both S32 and U32 → FMA pipe peak (since they take 2 source operands).
- [x] **M4 — `FMNMX` (FP min/max) throughput**: 0.99/SMSP/cy (28 TIPS_inst)
  for f32, f32.NaN, f16x2 — all peak. NaN-aware variant has zero overhead.
- [x] **M5 — SIMD intrinsics**: most native (ALU peak ~0.4/SMSP/cy):
  __viaddmax_s32, __vimax3_s32, __vsadu4, __vmaxs2. **EMULATED & slow**:
  __vmaxu4 (5×), __vavgu2 (3.4×). Always SASS-check.
  Commit history.
- [ ] **M6 — Saturated ops (`add.sat`, `sub.sat`)** throughput.
- [ ] **M7 — `bfind` throughput** (bit find).
- [ ] **M8 — Carry propagation** in IADD chains: hardware carry vs explicit
  ADC instruction.

## N. Cache hierarchy + replacement

- [ ] **N1 — L2 prefetcher behavior**: does B300 have stride detection?
- [ ] **N2 — L1 cache hit metric calibration**: build known-hit kernel,
  verify ncu.
- [ ] **N3 — `cctl::ivall` / `cctl::wb`**: does B300 emit these for any
  pattern? (Earlier finding: NO. Verify under different conditions.)
- [ ] **N4 — `cudaCacheConfigPreferShared`** effect.
- [ ] **N5 — `__threadfence_block` vs nothing** in single-warp test.

## O. Surprises / falsifiable claims

- [ ] **O1 — Compiler emits `STG.NA` when?** (Non-temporal store)
- [x] **O2 — Tensor core warmup**: **NO warmup penalty** for mma.sync.
  First MMA = steady-state MMA (~20 cy inc. clock64 overhead, 16 cy
  pipeline). No need for dummy-MMA warmup. See `b300_clean/O2_TENSOR_WARMUP.md`.
- [ ] **O3 — Branch density vs back-pressure**: kernel with 50% branches
  vs 0%.
- [x] **O4 — IMAD wide multiply**: 32x32→64 is 2× IMAD cost, 32x32→32hi
  is 1.9×, 64x64→64 (low) same as IMAD, **__umul64hi is 8.8× SLOWER**
  (emulated). Avoid splitmix64-style PRNGs in hot loops. Commit history.
- [x] **O5 — `__brevll` vs reverse lookup table**: BREV intrinsic always
  wins. `__brev`=3.53 TIPS, shift-based=3.53 TIPS (compiler folds to
  BREV), 8-bit LUT=0.93 TIPS (4× slower due to constant mem). Commit history.
- [ ] **O6 — Atomic on volatile pointer**: SASS difference vs non-volatile.
- [x] **O7 — `clock()` vs `clock64()` cost**: clock64 is **2× CHEAPER**
  than clock() (2.1 vs 4.0 cy/read after baseline subtract). Both
  compile to `CS2R SR_CLOCKLO`; clock() pays mask/shift overhead.
  See `b300_clean/O7_CLOCK_VS_CLOCK64.md`.

## P. SM-level resource limits

- [ ] **P1 — Max threads / block** boundary: 1024 → exact failure mode.
- [ ] **P2 — Max SHMEM / block** at full opt-in: 227 KB or some other limit.
- [ ] **P3 — Max registers / thread**: 255 limit; what happens at 256?
- [ ] **P4 — Max register spill** depth before crash.
- [ ] **P5 — Max LMEM** size per thread.

## Q. Hand-tuned kernel speed-of-light

- [ ] **Q1 — Max-throughput memcpy with TMA + cluster multicast**: beat
  cudaMemset's 7.57 TB/s NINJA recipe.
- [ ] **Q2 — Fast SHMEM-only reduction** (single block): 32 KB → 1 value
  in fewest cycles.
- [x] **Q3 — Fast cross-warp reduction** without SHMEM: `redux.sync.add`
  is **2.34× FASTER** than 5-step SHFL chain (11.6 vs 27.2 cy/reduce).
  Integer only (no .f32). REDUX.SUM writes to uniform register.
  See `b300_clean/Q3_WARP_REDUCE_RECIPES.md`.
- [ ] **Q4 — Vectorized scan** (prefix sum) at SHMEM SoL.
- [ ] **Q5 — Sort 1024 keys in single block at SoL**.
- [x] **Q6 — Transpose 32×32 SHMEM tile** without bank conflicts:
  smem[32][33] padding = **8.2× faster** than smem[32][32] naive.
  Skewed indexing smem[i][(i+k)&31] equally fast. Commit history.

## R. Power oddities + microarchitectural

- [ ] **R1 — Per-pipe gating threshold**: at what utilization does FMA pipe
  power-gate?
- [ ] **R2 — Idle SM "active" cost**: with grid waiting on launch, how much
  power do "ready" SMs draw?
- [ ] **R3 — Power difference between SM in `BRA` loop vs `EXIT`**.
- [ ] **R4 — Effect of large warps idle (mid-divergence)** on power.

## S. Methodology + tooling

- [ ] **S1 — Build `pkill -9 + sleep` wrapper** as a `clean_run.sh` utility,
  used by all sweeps to prevent contention.
- [ ] **S2 — ncu metric explorer**: dump every available metric for a
  sample kernel; categorize.
- [ ] **S3 — SASS auto-counter** that correctly counts ops in unrolled loops.
- [ ] **S4 — Power sampling at >10 Hz** (NVML query rate limit?).

---

## Methodology reminder

1. Read CLAUDE.md "B300 Methodology" section.
2. Mark item `[~]` before starting.
3. Verify ≥3 methods (wall-clock + ncu + SASS).
4. **ALWAYS** `pkill -9 QuickRunCUDA && sleep 6` between measurements.
5. State HIGH/MED/LOW + what would change conclusion.
6. Commit + mark `[x]` with hash.
7. **Use sub-agents** (Plan, Explore, general-purpose) for parallel research
   on independent items.

When picking parallel items, choose ones that don't share GPU state
(e.g., one compiler-research agent + one architectural-test agent).
