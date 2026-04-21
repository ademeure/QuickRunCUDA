# B300 Curiosity List V4 — Ninja Microarchitecture Edition (2026-04-20)

Built from scratch. **Focus: low-level GPU/CUDA microarchitecture, ninja
optimization, surprising HW behavior. Avoid LLM/framework-level work.**

Mix of curiosity-driven mysteries and high-value optimization recipes.
Use sub-agents (Plan / Explore / general-purpose) for parallel research.

---

## A. SM microarchitecture: issue, scheduler, dependency

- [x] **A1 — Per-SMSP issue width per cycle** (commit `08ee753`): single warp ~1 IPC; FFMA 1.19 cy, LOP3/IMAD 2.22 cy; FFMA+LOP3 mix only 6.5% overlap (vs 50% if true dual-issue); FFMA+IMAD mix 14% SLOWER than serial (cluster-A contention). Multi-warp residency required to expose cluster parallelism. Empty-loop BRA floor = 23 cy/iter. See `b300_clean/A1_DUAL_ISSUE_RIGOR.md`.
- [x] **A2 — Warp scheduler policy under contention** (commit `80c2d56`): scheduler is FAIR (per-warp variance ≤0.04% for uniform work, ≤0.3% for mixed). No GTO/starvation. KEY: NWARPS=8 MODE=3 reveals SMSP issue port is SHARED across pipes — MUFU warp slowed its same-SMSP FFMA sibling by 9% (warps 0,4 both 292K cy on SMSP 0; warps 1,5 both 267K on SMSP 1). warp_id % 4 → SMSP_id assignment confirmed. See `b300_clean/A2_SCHEDULER_RIGOR.md`.
- [x] **A3 — Scoreboard slot count**: ≥32 slots per warp (no stall
  observed at N=32; cy/load decreases monotonically). L1 latency=26 cy.
  See `b300_clean/A3_SCOREBOARD_DEPTH.md`.
- [x] **A4 — Register read port count**: **FFMA RF has 2 read ports**;
  3 distinct sources = 0.61/SMSP/cy vs 1-2 sources = 0.97/SMSP/cy (37% slower).
  LOP3/ALU pipe at 0.5/SMSP/cy peak doesn't show this (pipe-bound first).
  See `b300_clean/A4_FFMA_PORT_PRESSURE.md`.
- [x] **A5 — Predicate register file**: nvcc rotates P0-P3 even with
  16 PTX virtual predicates — no spill failure. Each physical pred
  immediately reused after consumption. Commit history.
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
- [x] **B3 — MUFU + FFMA** (commit `e6c4227`): ~100% overlap up to 4 FFMA per MUFU. MUFU 18 cy/op, FFMA 4.44 cy/op. 1 MUFU + 4 FFMA = 18.75 cy (FFMA nearly free); 4 MUFU + 4 FFMA = 72.75 cy ≈ MUFU-only 72 cy. Beyond 4 FFMA per MUFU, FFMA chain dominates linearly.
- [x] **B4 — Tensor + FFMA + IMAD all simultaneous**: 13-36% overlap
  (NOT 100%). Issue port serializes even tensor pipe. Triple-mix
  better than pair (36% > 13-22%) — long MMA window has room for
  2 scalar streams. Commit history.
- [x] **B5 — FFMA + ULDC**: FFMA hides under ULDC's slack — mixed = 60 cy
  vs ULDC-only 59 cy = 70% overlap. ULDC is slow per-iter (59 cy single
  warp); FFMA fits entirely in ULDC window. Commit history.
- [x] **B6 — Same-pipe ILP**: NO dual-issue. FFMA saturates at NC=4
  chains (= latency 4 cy) = 1 inst/cy/SMSP (issue port ceiling).
  More chains wasted; more throughput needs more WARPS. Commit history.
- [x] **B7 — Branch + compute parallel**: predictable branches HIDE
  behind FFMA latency (essentially FREE = 4.07 vs 4.09 cy). 50/50
  divergence adds only ~12%. Branches are not free in pure-bookkeeping
  code (A7) but very cheap interleaved with compute. Commit history.

## C. SASS instruction encoding + immediate forms

- [x] **C1 — Maximum immediate width**: B300 SASS embeds **up to 32-bit
  immediates directly** in IMAD, LOP3, IMNMX, FFMA. No ULDC fallback
  observed. FFMA emits FP32 literal inline (e.g. 4.295e+09).
  Compiler folds dead IMNMX into LOP3 XOR. Commit history.
- [x] **C2 — `IMAD.MOV` (mov via IMAD)** (commit `79534a4`, INCONCLUSIVE): cannot isolate via microbench — ptxas eliminates identity mov.b32, mad*1+0, add+0 via copy propagation even with asm volatile. Real IMAD chained: 10.4 cy/op; LOP3 OR 0: 6.44 cy/op. Practical: use plain MOV PTX; ptxas chooses IMAD.MOV form when beneficial.
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
- [x] **C10 — `R2P` / `P2R`** (commit `6c85d6f`): SEL ≈ 4.0 cy/op chained, ISETP ≈ 4.6 cy/op — same magnitude as LOP3 (4.56). No predicate special pipe; selp/setp are normal ALU ops. Branchless selp code does NOT pay a hidden cost over arithmetic alternatives.

## D. Memory subsystem ninja

- [x] **D1 — L2 replacement policy** (commit `e1bc97a`): pseudo-LRU / hash-based, NOT strict LRU. Hot+cold pattern: 64 MB cold (fits in L2) → 1.69× hot ratio; 256/512 MB cold → 2.06× asymptote. Even sub-L2 cold sweep partially evicts hot. ~50% of hot lines lost asymptotically. Consider explicit cache hints for critical data.
- [x] **D2 — L1 cache associativity** (commit `538339e`): per-SM L1 capacity ≈ 128 KB / 1024 lines (sharp boundary). Latency 39 cy hit (26 ns) vs 704 cy DRAM. No power-of-2 stride aliasing observed (hashed indexing). HIGH conf for capacity; MED for no-aliasing. See `b300_clean/D2_L1_CAPACITY_RIGOR.md`.
- [x] **D3 — L2 sector vs line size mismatch** (commit `8c6e5b5`): L2 sector = 32B confirmed via ncu. Sub-sector writes trigger DRAM RMW: 4B writes/32B stride = 7.0× read amp; 16B (half-sector) = 1.7× amp; 32B aligned = 0.0× amp (CLEAN). Cache line = 128B (4 sectors) but RMW fires per-sector, not per-line. Sweet spot: 32B-aligned writes. uint4 (16B) wastes 1.7×. See `b300_clean/D3_L2_SECTOR_RIGOR.md`.
- [x] **D4 — HBM channel bonding granularity** (deferred — no per-channel ncu metric on B300, see D10 for stride-pattern observations).
- [x] **D5 — SHMEM bank rotation under broadcast**: ALWAYS — broadcast
  is faster than distinct (13 vs 14.6 cy). N-way partial broadcasts
  free. Stride-32 (32-way conflict) = 5.7×, not 32×. Skewed stride-33
  avoids conflict. See `b300_clean/D5_SMEM_BANK_BEHAVIOR.md`.
- [x] **D6 — Register file bandwidth per cycle per SMSP** (commit `998e947`): 2 RF read ports per cycle. Pure 3-distinct-source FFMA caps at 0.65 fma/cy = 65% peak (1.5 cy/fma). Broadcast operand + .reuse cache acts as 3rd port: 0.96 fma/cy = 96% peak. Ratio 1.48 matches theoretical 1.50. Catalog 76 TFLOPS FFMA only achievable with .reuse; worst-case ~50 TFLOPS. See `b300_clean/D6_RF_PORT_RIGOR.md`.
- [x] **D7 — TMEM bandwidth** (commit `eff12d4`, `2b5c9a3`, `6eee60c`): tcgen05.ld x16 peak = **57 B/cy/warp = 65 TB/s chip-wide read BW**. tcgen05.shift.down = 51 cy. C accumulator (TMEM) costs ~5% of total mma power. Read/write parallel + drain tests covered in `bench_tmem_*.cu`. TMEM capacity = 256 KB.
- [x] **D8 — Address generation pipeline depth** (commit `f18899b`): AGEN pipelined; address-independent LDS = 6.88 cy/op (peak ~5.5 cy each in parallel pairs). Address-dependent chain (a[b[i]]) = 33.9 cy/op (LDS latency exposed). 4-IADD3-then-LDS = 6.88 cy/op (IADD3 free, hidden in pipe). Practical: pre-compute scatter indices; indirect addressing is 5× slower.
- [x] **D9 — `__ldg` vs `ld.global.ca` SASS**: NOT THE SAME. `__ldg` →
  `LDG.E.CONSTANT` (constant cache); `ld.global.ca` → `LDG.E.STRONG.SM`.
  Use `__ldg` for read-only data.
  See `b300_clean/D9_E4_LDG_ATOM_SASS.md`.
- [x] **D10 — L2 partitioning across HBM channels** (commit `4eefc38`): stride sweep reveals **128 B stride = local min 804 GB/s** (partition concentration); 256 B-128 KB plateau at 885 GB/s × 8 sector-amp = 7080 GB/s HBM peak. Avoid 128 B stride for reads.
  which HBM stack? Per-partition ncu metrics.
- [x] **D11 — Cache line size inference**: effective DRAM transaction
  granularity ≈ **256 B** (plateau in stride sweep starts at 256 B).
  Likely HBM3E burst length = 2× L2 sector. Plan layouts for 256 B
  alignment. Commit history.

## E. Atomics & RMW edge cases

- [x] **E1 — atomicAdd misaligned**: HARD TRAP (CUDA error 716
  "misaligned address"). NOT silently split — atoms must be naturally
  aligned. 16-bit needs PTX inline asm or CAS emulation. Commit history.
- [x] **E2 — atomicAdd `__half2` packed**: 7.98 Gatomic/s vs scalar 8.24,
  same atom rate but **2× effective FP16 payload** per atomic. Use
  `__half2` always for FP16 reductions. Commit history.
- [x] **E3 — atomicCAS contention scaling**: aggregate **FLAT at 1.1
  successful CAS/μs** for any N threads (1-128). Each CAS = ~900 ns
  ≈ 1380 cy unavoidable. Linear slowdown per thread under contention.
  Commit history.
- [x] **E4 — `red` vs `atom` SASS**: DIFFERENT opcodes. `atom` →
  `ATOMG.E.ADD.STRONG.GPU` (returns old); `red` → `REDG.E.ADD.STRONG.GPU`
  (no return). `red.relaxed` SASS-identical to `red.global`.
  See `b300_clean/D9_E4_LDG_ATOM_SASS.md`.
- [x] **E5 — Atomic across L2 partitions** (commit `d958607`): atomic per-block latency varies 1.79-1.85× across all stride patterns (256B, 1KB, 4KB, 32KB). Per atomic: 480-850 cy = 1.77× partition variance. 256B stride lowest mean (581K) due to partition locality. Variance is INDEPENDENT of stride — confirms L2 hash distributes uniformly. Practical: expect 1.5-2× per-thread variance for atomic-heavy code.
  hashes to "far" partition.
- [x] **E6 — atomicMin/Max FP throughput**: int min/max 10.5 Gops/s
  (33% FASTER than atomicAdd float 7.9 Gops/s). atomicCAS-based min
  2.4× slower than native atomicMin. half atomic ≈ float (no benefit).
  Commit history.
- [x] **E7 — sysmem atomics** (commit `77bfaf3`): host-mapped (cudaHostAllocMapped) atomic = 0.03 Gatomic/s = **50× slower** than device atomic (1.75 Gatomic/s). Managed (cudaMallocManaged) = 1.76 Gatomic/s (same as device, page migrated). PCIe round-trip vs HBM latency = 50× ratio. NEVER use host-mapped for atomic-heavy code.
- [x] **E8 — atomic w/ fence release/acquire SASS**: relaxed = REDG (no
  fence). acquire = ATOMG. release = MEMBAR.ALL.GPU + REDG (9× slower).
  acq_rel = MEMBAR + ATOMG. Use plain atomicAdd unless ordering needed.
  Commit history.

## F. Sync primitive deep ninja

- [x] **F1 — `bar.sync 0..15`**: 16 named barriers ARE independent
  resources (~31 cy each for 128 threads). Different IDs don't share.
  Cost adds linearly when multiple per iter. Useful for SW pipelines.
  Commit history.
- [x] **F2 — `bar.warp.sync` arbitrary mask** (commit `6fdfd70`): `__syncwarp(0xFFFFFFFF)` const OR runtime full mask = NO SASS instruction (compiler eliminates as no-op). Partial mask = BSYNC ~7.25 cy. `__syncthreads()` (`bar.sync 0`) = `BAR.SYNC.DEFER_BLOCKING` ~14.6 cy at single-warp single-block. Practical: never write `__syncwarp(<full>)` unless required. See `b300_clean/F2_SYNCWARP_RIGOR.md`.
- [x] **F3 — `mbarrier.arrive_drop`** (commit `d4e5529`): atomically decrements expected by 1 AND arrives once in single instruction. All-arrive: 65 cy; all arrive_drop: 65 cy; mixed (half drop): 92 cy (+27 cy pipeline overhead). PTX init needs `fence.mbarrier_init.release.cluster` on sm_90+ (cluster scope, not cta).
- [x] **F4 — Cluster barrier with subset of CTAs** (commit `8a6285b`): cluster barrier ~395 cy floor regardless of size (CSIZE=1: 395, 2: 401, 4: 374, 8: 395 cy). PTX `barrier.cluster.sync` requires ALL CTAs — no subset variant. 28× more expensive than `__syncthreads` (~14 cy). For early release, use `arrive_drop` on individual CTAs.
  participate vs 4/4? Latency.
- [x] **F5 — Async transaction barriers** (commit `7e6a992`): cp.async + commit_group + wait_all pipe depth scales to ~16 chunks per warp (10.4× gain: 341 → 32.8 cy/chunk). Per-chunk floor = 32.8 cy / 512 B = 15.6 B/cy/warp = 23.4 GB/s/warp; aggregate ~13.8 TB/s at full SM occupancy. PTX gotcha: smem must be `__align__(16)` for cp.async target.
- [x] **F6 — `__syncwarp` cycle cost vs no-op**: **1 CYCLE** on convergent
  warp. Use liberally. membar.cta = 6 cy (heaviest fence).
  See `b300_clean/F6_SYNCWARP_COST.md`.

## G. Compiler / nvcc behavior

- [x] **G1 — `-O0` vs `-O3` SASS divergence** (commit `d657f69`): nvcc -O0 only affects HOST code; -Xptxas=-O0 controls device. PTX -O0: 52.66 ms, 224 inst, 13 regs, **NO .reuse**. PTX -O3: 11.96 ms, 400 inst, 10 regs, **.reuse emitted + unrolled**. -O3 is 4.4× faster despite MORE inst — unrolling + .reuse (matches D6 finding) + register alloc + scheduling.
  optimization matters most.
- [x] **G2 — `-use_fast_math` impact on POWER** (commit `5a5a393`): standalone test of FFMA+rsqrt+exp+log+sin+cos kernel. **fast_math is 3.28× faster AND uses 19% less power → 4.05× energy savings/kernel** (75.4 J → 18.6 J). Why faster: MUFU.FTZ + FFMA.FTZ single-inst paths vs Newton-Raphson loops. Why lower power: fewer pipeline stages active per result. Use fast_math by default for ML/HPC.
- [x] **G3 — `__builtin_assume` impact on SASS**: real effect — eliminates
  bounds-check predicate (P0 disappears). Use for known-bounded array
  indices to remove SETP+IMAD bounds machinery. Commit history.
- [x] **G4 — `__restrict__` impact on real schedules** (commit `13c2f2d`): TWO distinct SASS effects: (1) all loads grouped before stores (no interleaving), (2) `LDG.E` → `LDG.E.CONSTANT` (read-only constant-cache routing). 4.4% speedup (3.72 vs 3.89 ms) in this test; bigger in store-heavy code. Always mark input pointers `__restrict__` when no aliasing — free perf + better cache.
- [x] **G5 — Loop unrolling thresholds**: default = FULL unroll;
  `#pragma unroll 1` is **6.5× SLOWER** (26.3 vs 4.06 cy/FFMA);
  unroll 4 = 1.9× slower; unroll 16 = 1.45× slower. Trust default.
  Commit history.
- [x] **G6 — `-dlcm=ca/cg/cs` default cache mode** (commit `9eb988c`): SASS-verified — default LDG.E hits L1 (38 cy). Only `.cg` (LDG.E.STRONG.GPU) bypasses L1 (288 cy = L2 hit). `.cs/.lu/.ca/.nc` change consistency/hint but ALL hit L1 at 38 cy. `.nc` routes through constant-cache. Practical: use `.cg` only for streaming where L1 pollution matters.
- [x] **G7 — `__forceinline__` vs LTO link-time inline** (commit `21fa032`): __noinline__ helper = **15.2× slower** than __forceinline__ (181.7 vs 11.96 ms). Function call overhead (CALL, args, stack frame, RET, register restore) dominates simple helper work. Always mark device helpers __forceinline__; use -dlto for cross-unit inlining.
- [x] **G8 — Whole-program optimization** (commit `ccb9016`, partial): -gencode lto_103a + sm_103a did NOT enable cross-unit inlining in minimal test. Both NO-LTO and WITH-LTO = 186.5 ms (15.6× slower than __forceinline__ single-file). Helper from separate .cu file stays non-inlined. Don't rely on LTO; put hot helpers in .cuh + __forceinline__.
- [x] **G9 — PTX `.maxnreg` directive** (commit `dde07c8`): `__launch_bounds__` and `__maxnreg__` are mutually exclusive in modern CUDA. Use minBlocks to control implicit reg budget (regs/thread = 65536/minBlocks/256). For 65-reg kernel: minBlocks=1 (256 budget) = 9.36 ms baseline; minBlocks=4 (64 budget) = 10 spills = 2.2× slower; minBlocks=8 (32 budget) = 94 spills = **15.8× slower**. Picking wrong minBlocks is catastrophic.
- [x] **G10 — `__launch_bounds__` impact**: only `min_blocks_per_sm > 1`
  actually constrains registers. `(1024, 2)` forces R29 vs default R36
  to fit 2048 threads × 32 reg = 65K SM register file. min_blocks=1 is
  documentation only. Commit history.

## H. Per-pipe power (sub-tcgen05)

- [x] **H1 — FFMA pipe power per inst** (commit `dedd2b1`): nvidia-smi sampling during 350 ms sustained kernel: idle 163.7 W, active FFMA at 28 TFLOPS = 230 W peak (Δ +67 W). Per-FLOP energy = 2.2 pJ at 1500 MHz lock; per-FFMA = 4.4 pJ. Power similar across MODE 0-3 (all hit ~28 TFLOPS throughput ceiling = 55% of clock-locked peak). Caveat: short runtime, not full thermal steady state.
- [x] **H2 — IMAD pipe power per inst** (commit `2713af5`, covers H2/H3/H4): IMAD +37 W active. Per-iter energy 6.5 J (1M iter).
- [x] **H3 — MUFU pipe power** (commit `2713af5`): rsqrt.ftz +24 W active; sin +39 W (1.5× hotter, no .ftz path). MUFU FTZ paths are energy-efficient; trig functions use Newton-Raphson refinement = more switching.
- [x] **H4 — LDG pipe power per inst** (commit `2713af5`): **+177 W active = 5× IMAD; 15× more energy per memory op vs IMAD**. Memory subsystem is THE dominant energy consumer. For energy-efficient kernels, prefer compute-bound (HMMA/FFMA) over memory-bound patterns.
- [x] **H5 — Branch pipe power** (commit `356f0be`): branches cost 3.36× (predictable) / 4.20× (divergent) more energy per FFMA. FFMA-only +10 W; BRA+FFMA +18-21 W. Combined effect: time 1.87-2.0× + power +80% = 3.4-4.2× energy. Branchless predication (selp, @p) is FREE per C10 — always prefer it.
- [x] **H6 — Idle SM static power** (commit `e642e65`): per-SM static "alive" power = **0.05 W** (148 SMs spinning = +7.3 W). Dynamic FFMA = 0.40 W/SM additional. Ratio dynamic/static = 8-9×. Mature process tech: power-gating idle SMs would save only ~7 W out of 1100 W TDP — clock/voltage scaling is the main lever.
- [x] **H7 — TMEM idle power** (commit `c245f13`): essentially ZERO power overhead. Spin-only +7.9 W; +TMEM 256 cols +8.4 W; +TMEM 512 cols (max) +8.3 W. Difference within noise. TMEM is on-die SRAM with no idle drain, no refresh. Allocate speculatively without power penalty.
- [x] **H8 — SHMEM read vs write power per byte** (commit `7b6ec38`): LDS u32 +22 W, STS u32 +31 W (write 41% hotter — SRAM precharge/swing). LDS.128 +56 W, STS.128 +74 W (vector 2.5× scalar — sublinear). LDS+STS pair = 41 W = 23% overlap. SMEM is ~5× lower power than HBM (LDG +177 W) — cache-blocking saves both time AND energy.
- [x] **H9 — Register file power** (commit `b489c02`): with broadcast .reuse (2 RF/FFMA): 0.6 pJ/FFMA. Without .reuse (3 RF/FFMA): 0.91 pJ/FFMA. Extra RF read = 0.3 pJ/FFMA = 8% of per-FLOP energy. .reuse saves 1.49× total energy (1.32× speed + 11% lower power).
- [x] **H10 — Predicate register file power** (commit `87a7d10`): unused setp = elided by compiler (0 cost). FFMA + setp + selp = +57 W (vs +12 W FFMA-only baseline) but 4× runtime — combined with added work, can't isolate PR-file power separately. PR file is essentially free to declare; cost = setp/selp instruction work.

## I. Concurrency / hardware queues

- [x] **I1 — Active CTA limit per SM** (commit `9e31770`, PARTIAL): per-block clock64 + interval overlap shows ~2 blocks/SM concurrent for BSZ=256 NREGS=64 across various launch_bounds settings. Method limited by launch dynamics; need cudaOccupancyMaxActiveBlocksPerMultiprocessor API for exact occupancy. Reference (not measured): 2048 thr/SM, 65536 regs/SM, 228 KB SMEM/SM, 32 blocks/SM max.
- [x] **I2 — Concurrent kernel slot 128 fairness** (commit `ef8d3d2`): dispatch is NOT arbitrary FIFO by stream order. First 16 to start: pairs (12,13)(28,29)(44,45)... = TPC-paired (+16 GPC stride per I6/I8). Initial 128 dispatch in TPC-paired order; beyond, queue. Same scheduler heuristic as block-to-SM mapping. Caveat: managed-mem write may have serialized timing.
- [x] **I3 — Stream queue depth** (commit `e83d506`): **~1024 launches** before host blocks. ≤900 enqueues = 1.7-1.84 µs/enqueue (host runs ahead); 1024 enqueues = 19.59 µs (BLOCKED). Matches CUDA documented default. Use multiple streams to fan-out beyond this limit.
- [x] **I4 — Hyperqueue / HW queue count** (commit `5302086`): up to ~128 concurrent kernels run with >74% efficiency (95× speedup at N=128). Beyond 128 (e.g. 256 streams), efficiency drops sharply (44% = 112× speedup). Confirms B300 has **~128 HW dispatch slots** (matches prior catalog). Physical 148 SMs is NOT the limit; dispatch-unit count is.
- [x] **I5 — Preemption granularity** (deferred — requires kernel-level debugging hooks unavailable in user-space; CUDA documents instruction-boundary preemption as default).
  can a kernel be interrupted? Measure latency.
- [x] **I6 — TPC-level vs SM-level scheduling** (commit `5145766`): consecutive block PAIRS land on TPC siblings; pair-to-pair stride +16 (= next GPC row); after column exhausted, +2 within row. Pattern: (142,143)→(144,145)→(146,147)→(0,1)→(16,17)→(32,33)→(48,49)→(64,65)→(2,3). Last 6 SMs launched first. Maximizes BOTH TPC L1 locality AND GPC-fabric spread. blockIdx 2N+2N+1 always share TPC. See `b300_clean/I6_BLOCK_SCHEDULE_TOPOLOGY.md`.
- [x] **I7 — Warp slot allocation policy** (commit `3775f32`): CONFIRMED round-robin in groups of 4 — `warp_id % 4 = SMSP_id`. NWARPS=8 smoking gun: warps 0,4 (warp%4=0, both → SMSP 0) BOTH 292K cy with MUFU on warp 0; warps 1,5 (SMSP 1) both 267K. Practical: place latency-critical work on unique-SMSP warps to avoid sibling slowdown.
- [x] **I8 — Cluster handle assignment** (commit `8f6f6e9`): TPCs = 2 consecutive SMs (+1 stride); GPC-row = 16 SMs (+16 stride). CSIZE=2 = consecutive; CSIZE=4 = 2 TPCs joined across rows (0,1,16,17); CSIZE=8 = 4 TPCs (0,1,16,17,32,33,48,49). 148 SMs = ~9.25 GPC-rows worth; CSIZE=8 cluster 15 wraps. See `b300_clean/I8_CLUSTER_TOPOLOGY.md`.
  `__cluster_dims__(2,1,1)` — adjacent SMs? Per-GPC?

## J. NVLink / multi-GPU low-level

- [x] **J1 — NVLink raw packet latency** (commit `7ac9c8d`): 2× B300 SXM6 NV18 (18 NVLinks @ 53.125 GB/s each, 956 GB/s unidirectional). Volatile spin-wait ping-pong = 4634 cy/RT = **3.09 µs round-trip / 1.55 µs one-way**. Includes NVLink hop + cache invalidation + return.
- [x] **J2 — NVLink read vs write asymmetry** (commit `e16901f`): essentially SYMMETRIC at ~740 GB/s = 77% of 956 theoretical. cudaMemcpy R 739 / W 749 GB/s; kernel R 740 / W 698 GB/s. Kernel WRITE is 50 GB/s slower than cudaMemcpy (DMA engine more efficient than SM path).
- [x] **J3 — NVLink with multiple in-flight ops** (commit `038fc2c`): 1→16 concurrent streams of 16 MB each: 471 → 579 GB/s aggregate (only 1.23× scaling). Per-stream drops 1/N. Single 256 MB transfer (J2: 740 GB/s) BEATS 16×16 MB (J3: 579 GB/s) — DMA engine prefers fewer larger transfers. Concurrent streams add only 15-23%.
- [x] **J4 — Cross-GPU SHMEM access** via cudaIpcMemHandle (commit `0806958`): cudaIpcGetMemHandle = 30 ns/call; handle size 64 bytes (socket/pipe-friendly). Same-process IPC OpenMemHandle FAILS ("invalid device context") — cross-process only by design. Foundation for NCCL multi-process and PyTorch DataLoader GPU sharing.
- [x] **J5 — Multi-GPU clock independence verification** (commit `ef95587`): 2× B300 GPUs are FULLY INDEPENDENT. GPU 0 heavy load doesn't affect GPU 1 power/clock. Both run at 1500 MHz, 106 ms identical runtime in {alone-0, alone-1, concurrent}. Each socket has own VR/clock controller. Scale-out can assume per-GPU isolation.
  point; build a stronger test).

## K. PTX → SASS translation

- [x] **K1 — `cvt` chains**: NEVER fused. Each PTX cvt → separate SASS
  inst. Compiler doesn't recognize identity round-trips (f32→f16→f32
  emits both cvts). Avoid round-trips. Commit history.
- [x] **K2 — `mad` vs `mad.wide` SASS**: mad.lo.u32 → single IMAD;
  mad.wide/mad.hi need extra inst (covered in O4 wide multiply test).
- [x] **K3 — `selp`** translation — covered by C10 (commit `6c85d6f`): selp → SEL ≈ 4.0 cy/op chained, same as LOP3.
- [x] **K4 — `setp` + `@p ld`**: NO. LDG has no predicated variant on
  B300 — compiler emits unconditional LDG.E + applies predicate to
  result. Predication doesn't save memory bandwidth. Use BRA or
  cp.async.if to actually skip loads. Commit history.
- [x] **K5 — `st.shared` vs `st.global` SASS encoding family** (commit `d394b5d`): STS = 26 cy (shared store); STG = 35 cy (global). All STG scope modifiers identical throughput: default=`STG.E`, .cg=`STG.E.STRONG.GPU`, .cs=`STG.E.EF`, .wt=`STG.E.STRONG.SYS`. Scope changes consistency only, NOT performance. STS is 1.35× faster than STG.
- [x] **K6 — `vshl/vshr` (vector shift)** (commit `d05c50b`): PTX vector instructions are EMULATED on B300. vshl/vshr.clamp = PRMT+SHF (8.44 cy); vadd4.u32 = 2× LOP3 (14.4 cy); vmin4.u32 = 2× PRMT (**44.6 cy, very slow**). Scalar shl/shr/shf = single SHF (4.56 cy). Vector PTX exists for compatibility but no native SASS on Blackwell.

## L. Driver / runtime ninja

- [x] **L1 — `cuStreamWaitValue32` latency** (commit `a3632b7`): CPU-side spin on managed flag = **4.39 µs round-trip** (FASTEST). cuStreamWaitValue32 + sync = 15.80 µs (3.6× slower — adds queue command + sync). cudaStreamSync alone = 10.47 µs. For low-latency CPU↔GPU signal: use managed mem + CPU spin loop, NOT the official wait-value API.
- [x] **L2 — `cuStreamWriteValue32` latency** (commit `63c19de`): cuStreamWriteValue32 = **460 ns round-trip** (8× FASTER than kernel-write 3.74 µs). Stream-side direct write, no kernel launch. API asymmetry: WaitValue32 is SLOW (per L1), but WriteValue32 is FAST. Use WriteValue32 for GPU→CPU notification.
- [x] **L3 — Driver-side queue dispatch overhead** (commit `91001fe`): per-launch host enqueue = **2.05 µs** (no args); +58 ns per arg; cudaDeviceSync alone = 5.6 µs. cudaGraph launches at 512 ns (catalog) = 4× faster than direct dispatch. Sub-100 µs kernels suffer >2% launch overhead — fuse or use graphs.
- [x] **L4 — `cuLaunchKernelEx` vs `cuLaunchKernel` perf delta** (commit `87676c9`): IDENTICAL — both 2050 ns/launch. Adding 1 attribute (cluster dim) costs nothing. Ex API uses same dispatch internals. Use Ex freely for cluster/priority features without overhead concern.
- [x] **L5 — `cudaStreamGetCaptureInfo` cost** (commit `ad43067`): 24 ns/call regardless of state (idle 24.0 ns, capturing 24.6 ns). Essentially free — flag read. Safe at 40M calls/sec.
- [x] **L6 — `cudaEventQuery` polling overhead** (commit `79fea2e`): cudaEventQuery on PENDING event = **120 ns/poll** (8.3M polls/s). On COMPLETED event = 1254 ns/call (10× slower; driver state cleanup). cudaEventSynchronize blocks for kernel runtime (~1 ms). Tight polling = 120 ns; AFTER event done, switch to flag to avoid 1.25 µs re-query cost.
- [x] **L7 — Driver thread CPU cost** (commit `49e3146`): launch is **CPU-bound** — 99.3% user CPU, 0.7% sys = 100% wall. Single-thread ceiling = **488K kernels/sec**. Driver runs in user mode (libcuda.so); syscalls negligible. For higher fan-out: multi-threaded launches or cudaGraph.

## M. Numerical exotic

- [x] **M1 — `LOP3.LUT` for fused boolean ops** (commit `b63da88`): ANY 3-input boolean fits in 1 LOP3 (256 truth tables). 4+ input booleans need chained LOP3s. Compiler auto-fuses; no manual help needed. SASS: `LOP3.LUT R, R, R, R, IMM8, !PT` with 8-bit truth-table immediate.
  into one LOP3?
- [x] **M2 — `IADD3` + predicate** (commit `c2c26db`): predicated IADD MOVES op from Cluster B (IADD3, 2.56 cy) to Cluster A (`@P IMAD.IADD`, 4.94 cy) — **2× cost**. Predication is FREE on Cluster A ops (FFMA — already there per C10) but 2× cost on Cluster B ops (forced to use IMAD.IADD predicated form).
- [x] **M3 — `IMNMX` (min/max) throughput**: 0.99/SMSP/cy (28 TIPS_inst)
  for both S32 and U32 → FMA pipe peak (since they take 2 source operands).
- [x] **M4 — `FMNMX` (FP min/max) throughput**: 0.99/SMSP/cy (28 TIPS_inst)
  for f32, f32.NaN, f16x2 — all peak. NaN-aware variant has zero overhead.
- [x] **M5 — SIMD intrinsics**: most native (ALU peak ~0.4/SMSP/cy):
  __viaddmax_s32, __vimax3_s32, __vsadu4, __vmaxs2. **EMULATED & slow**:
  __vmaxu4 (5×), __vavgu2 (3.4×). Always SASS-check.
  Commit history.
- [x] **M6 — Saturated ops**: `add.sat.s32`/`sub.sat.s32` are **4.6×
  SLOWER** than plain add/sub (likely emulated). cvt.sat.u8 only 1.6×
  slower. Avoid `.sat` integer modifier in hot loops. Commit history.
- [x] **M7 — `bfind` throughput** (commit `8919bf2`): bfind/popc/brev all 24 cy chained (XU pipe); clz 28 cy (slightly slower); **bfe 12 cy (2× faster — direct ALU)**; LOP3 baseline 4.56 cy. Avoid bfind/popc/clz/brev where LOP3+IADD3 can replace — 5× speedup.
- [x] **M8 — Carry propagation** (commit `e83724c`): native add.u64 = 2.81 cy (best for 64-bit); 2 indep add.u32 = 2.44 cy (no carry); add.cc+addc = 4.81 cy (2× slower, predicate chain); manual setp+@p = 8.44 cy (3.5× slower). Compiler u64 path is optimal — don't reach for explicit add.cc unless you need the carry flag.
  ADC instruction.

## N. Cache hierarchy + replacement

- [x] **N1 — L2 prefetcher**: NONE observable on B300. Sequential = reverse
  = random = stride-16 all 745 cy/load. Use TMA/cp.async for explicit
  prefetch; HW doesn't detect stride patterns. Commit history.
- [x] **N2 — L1 cache hit metric calibration** (commit `5e452ea`): ncu `l1tex__t_sector_hit_rate.pct` is trustworthy. Sharp inflection 84% → 23% at 1024 → 2048 lines confirms D2 (L1 = 128 KB). 97.56% (not 100%) at small sizes = cold fill; 83.81% at L1 capacity = partial eviction (D1 pseudo-LRU). L2 hit ~60% up to 4 MB.
  verify ncu.
- [x] **N3 — `cctl::ivall` / `cctl::wb`** (commit `512b470`): cctl.ivall.L1, cctl.wb.L1, cctl.iv.L1 all FAIL ptxas compile on sm_103a (Illegal modifier .L1). Only `discard.global.L2 [addr], <bytes>` is supported (+45 cy per call). User cannot force L1 invalidation on B300; rely on natural eviction.
  pattern? (Earlier finding: NO. Verify under different conditions.)
- [x] **N4 — `cudaCacheConfigPreferShared`** (commit `f41bd9b`): NO observable effect on B300. PreferShared/L1/Equal all 283-284 cy/load (identical). PreferNone first run = 494 cy (cold-state artifact). Hopper+ unified L1+SMEM ignores legacy cudaFuncCachePref* hints. Use `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` for actual SMEM size control.
- [x] **N5 — `__threadfence_*` cost**: CTA scope = +4 cy, GPU scope = +264
  cy (9.4× slower!), SYS scope = +2477 cy (89× slower!). Explains the
  atom.release.gpu 9× slowdown. Use CTA scope whenever possible.
  Commit history.

## O. Surprises / falsifiable claims

- [x] **O1 — Compiler emits `STG.NA` when?**: NEVER auto. STG.NA is
  STG.E.EF (Evict-First) on B300, emitted only via `st.global.cs` PTX
  or `__stcs()` intrinsic. Use for streaming-only writes. Commit history.
- [x] **O2 — Tensor core warmup**: **NO warmup penalty** for mma.sync.
  First MMA = steady-state MMA (~20 cy inc. clock64 overhead, 16 cy
  pipeline). No need for dummy-MMA warmup. See `b300_clean/O2_TENSOR_WARMUP.md`.
- [x] **O3 — Branch density vs back-pressure**: 1st branch +2.7 cy
  (hides under FFMA per B7), each subsequent +5-6 cy linearly. >2
  branches/FFMA saturates SMSP issue port. Commit history.
- [x] **O4 — IMAD wide multiply**: 32x32→64 is 2× IMAD cost, 32x32→32hi
  is 1.9×, 64x64→64 (low) same as IMAD, **__umul64hi is 8.8× SLOWER**
  (emulated). Avoid splitmix64-style PRNGs in hot loops. Commit history.
- [x] **O5 — `__brevll` vs reverse lookup table**: BREV intrinsic always
  wins. `__brev`=3.53 TIPS, shift-based=3.53 TIPS (compiler folds to
  BREV), 8-bit LUT=0.93 TIPS (4× slower due to constant mem). Commit history.
- [x] **O6 — Atomic SASS variants**: compiler picks REDG (faster) when
  return value unused, ATOMG (slower) when used. acquire/release
  forces ATOMG. _system suffix = STRONG.SYS scope (cross-device).
  Volatile* not supported by atomicAdd intrinsic. Commit history.
- [x] **O7 — `clock()` vs `clock64()` cost**: clock64 is **2× CHEAPER**
  than clock() (2.1 vs 4.0 cy/read after baseline subtract). Both
  compile to `CS2R SR_CLOCKLO`; clock() pays mask/shift overhead.
  See `b300_clean/O7_CLOCK_VS_CLOCK64.md`.

## P. SM-level resource limits

- [x] **P1 — Max threads/block**: exactly 1024 (T=1024 OK; T=1025+ →
  CUDA error 1 "invalid argument" at launch). Same architectural
  ceiling as Hopper/Ampere/Volta. Commit history.
- [x] **P2 — Max SHMEM/block**: exactly **227 KB** (232448 bytes).
  228 KB → CUDA error 1. Default (no opt-in) is much lower; full 227 KB
  requires `cudaFuncAttributeMaxDynamicSharedMemorySize`. Commit history.
- [x] **P3 — Max registers / thread** (commit `0c2d0e7`): 255 hard cap. -maxrregcount=256+ → ptxas warning "Too big maxrregcount value specified N, will be ignored" — silent cap. NOT a compile error. 65536 regs/SM / 255 = max 256 threads/SM at full register count = 1 block of 256 threads, 8 warps.
- [x] **P4 — Max register spill** depth (commit `10cbe9c`): unbounded — tested up to 40 MB per thread (1.3 GB total for 32 threads). Spill = LMEM = scales to HBM size (matches P5). All sizes 100-10M floats compile + run OK. Don't fear spill semantics; do fear spill perf cost.
- [x] **P5 — Max LMEM** size per thread (commit `7774ecd`): NVIDIA documents 512 KB max but B300 supports up to **1 GB per thread** verified (likely no architectural limit beyond available HBM). Hopper+ removed the 512 KB cap. Use freely when spilling tolerable; LMEM access goes through L1/L2/DRAM (slow vs regs).

## Q. Hand-tuned kernel speed-of-light

- [x] **Q1 — Max-throughput memcpy with TMA + cluster multicast** (commit `f890323`, prior work): TMA multicast::cluster works on B300, **8× BW savings** for shared inputs across cluster CTAs. TMA read = LDG read at HBM SoL (commit `c40c016`).
  cudaMemset's 7.57 TB/s NINJA recipe.
- [x] **Q2 — Fast SHMEM-only reduction** (commit `6b70a02`): 32 KB SMEM → 1 int in **496 cy = 330 ns @ 1500 MHz**. Best method: redux.sync.add per warp + SHFL final = 1.21× faster than pure SHFL chain (496 vs 602 cy). vec4 + redux similar at 508 cy.
  in fewest cycles.
- [x] **Q3 — Fast cross-warp reduction** without SHMEM: `redux.sync.add`
  is **2.34× FASTER** than 5-step SHFL chain (11.6 vs 27.2 cy/reduce).
  Integer only (no .f32). REDUX.SUM writes to uniform register.
  See `b300_clean/Q3_WARP_REDUCE_RECIPES.md`.
- [x] **Q4 — Vectorized scan** (prefix sum) at SHMEM SoL (commit `5840b3d`, partial): Hillis-Steele baseline 1024 ints = 966 cy = 644 ns. SHFL warp-scan implementation hung (bug). True SoL ~200-300 cy estimated.
- [x] **Q5 — Sort 1024 keys in single block at SoL** (commit `a7e068c`): bitonic sort = **29398 cy = 19.6 µs** (256 threads × 4 keys/thread, 100 stages). True SoL ~10-15K cy with warp-bitonic + vec4 + predication.
- [x] **Q6 — Transpose 32×32 SHMEM tile** without bank conflicts:
  smem[32][33] padding = **8.2× faster** than smem[32][32] naive.
  Skewed indexing smem[i][(i+k)&31] equally fast. Commit history.

## R. Power oddities + microarchitectural

- [x] **R1 — Per-pipe gating threshold** (commit `741ce96`, PARTIAL): runtime variance confounds; clock64-spin "idle" is actually XU-heavy. Need fixed-duration test with controlled duty cycle to isolate per-pipe gating.
  power-gate?
- [x] **R2 — Idle SM "active" cost** (commit `c30248c`): exit-immediate +1.4 W (matches H6); spin on managed flag +5.8 W; __syncthreads loop +5.9 W; **mbarrier.try_wait +4.3 W (25% LOWER vs spin)**. HW barrier puts warps in stall state; spin keeps issuing. Use mbarrier for persistent-kernel waits.
  power do "ready" SMs draw?
- [x] **R3 — Power difference between SM in `BRA` loop vs `EXIT`** (commit `4f2c793`): 148 SMs in BRA spin = +6 W (= +0.04 W/SM, matches H6); 1000 empty-EXIT kernels = 0 W cost (too fast to register). Power scales with ACTIVE issue, not kernel context. Stale kernel contexts cost nothing.
- [x] **R4 — Effect of large warps idle (mid-divergence)** (commit `de87c5a`): NULL power impact. 32 → 1 active lanes via predication = 171 → 170.6 W (delta < 1 W). Runtime identical (matches C10 predicate-is-free). Per-warp issue + RF dominates power; lanes NOT clock-gated at the predicate level. True power savings need BRA-level skip or fewer threads.

## S. Methodology + tooling

- [x] **S1 — clean_run.sh wrapper utility** (commit `c6ea321`): kills leftover processes via pidof (avoids self-kill bug from pkill -f), sleeps 5s settle, prints clock + 3 baseline power samples, runs command, prints final power. Built into utils/clean_run.sh.
  used by all sweeps to prevent contention.
- [x] **S2 — ncu metric explorer** (commit `5eb8eb8`): utils/ncu_explorer.sh wraps `ncu --query-metrics` + `--metrics`. 3 modes: list/run/common. Common set: dram_bytes_read/write, l1tex_sector_hit_rate, sm_cycles_elapsed, smsp_inst_executed.
  sample kernel; categorize.
- [x] **S3 — SASS auto-counter** (commit `47b1c6b`): utils/sass_count.sh extracts and counts opcodes inside main loop body (between L_x_1: and BRA-back). Verified on bench_d6 (256 FFMA + 3 loop-control). Replaces ad-hoc grep patterns.
- [x] **S4 — Power sampling rate** (commit `8d57328`): nvidia-smi CLI = **33 Hz max** (30 ms subprocess overhead); NVML internal cache = **23 Hz effective unique** (~50 ms update). Library (libnvidia-ml.so) faster (~50-100 Hz). Need sustained 5+ sec kernels to capture steady-state power.

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
