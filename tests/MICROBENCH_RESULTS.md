# Blackwell CVT, MUFU & Compute Microbenchmark Results

**GPU:** NVIDIA B300 SXM6 AC (SM 10.3a, 148 SMs, CUDA 13.0)
**Clocks:** SM locked 2032 MHz, Mem 3996 MHz (HBM3e, 8192-bit bus)
**All kernels:** UNROLL=8, inputs depend on `threadIdx.x`, loop-carried anti-LICM
**Date:** 2026-04-09

### What is an "op"?

One **op** = one PTX instruction executed by one thread. Different instructions convert different numbers of **elements** (scalar values):


| Suffix                                                   | Elements per op | Example                               |
| -------------------------------------------------------- | --------------- | ------------------------------------- |
| Scalar (`.f16.f32`, `.bf16.f32`, `.tf32.f32`)            | **1**           | 1 f32 -> 1 f16                        |
| Packed x2 (`.f16x2.f32`, `.e4m3x2.f32`, `.e4m3x2.f16x2`) | **2**           | 2 f32 -> 2 e4m3, or 1 f16x2 -> 2 e4m3 |
| Packed x4 (`.e4m3x4.f32`)                                | **4**           | 4 f32 -> 4 e4m3                       |


---

## 1. Baseline Benchmarks


| Benchmark                            | Measured        | Theoretical           | % SOL     |
| ------------------------------------ | --------------- | --------------------- | --------- |
| DRAM Bandwidth                       | **6,378 GB/s**  | 8,184 GB/s            | **77.9%** |
| FP32 FMA (8 chains, UNROLL=8, 256t)  | **63.8 TFLOPS** | 76.9 TFLOPS (148 SMs) | **82.9%** |
| FP32 FMA (8 chains, UNROLL=8, 1024t) | **66.9 TFLOPS** | 76.9 TFLOPS           | **86.9%** |


---

## 2. Narrow Format Conversions (FP4 / FP6 / FP8)

### To-narrow from f16x2 (`cvt.rn.satfinite.{fmt}.f16x2`)

1 op = 1 f16x2 (2 f16 elements) -> 1 packed narrow pair = **2 elements/op**


| Format                 | GOps/s | GElems/s   | +relu       |
| ---------------------- | ------ | ---------- | ----------- |
| **e2m1x2 (FP4/NVFP4)** | 5,919  | **11,838** | same        |
| **e4m3x2 (FP8 E4M3)**  | 6,895  | **13,790** | same        |
| **e5m2x2 (FP8 E5M2)**  | 6,896  | **13,792** | same        |
| e2m3x2 (FP6)           | --     | --         | UNSUPPORTED |
| e3m2x2 (FP6)           | --     | --         | UNSUPPORTED |


### To-narrow from f32 pair (`cvt.rn.satfinite.{fmt}.f32`)

1 op = 2 f32 inputs -> 1 packed narrow pair = **2 elements/op**


| Format            | GOps/s | GElems/s   | +relu |
| ----------------- | ------ | ---------- | ----- |
| e2m1x2 (FP4)      | 6,157  | **12,314** | same  |
| e4m3x2 (FP8 E4M3) | 6,865  | **13,730** | same  |
| e5m2x2 (FP8 E5M2) | 6,865  | **13,730** | same  |
| e2m3x2 (FP6 E2M3) | 6,867  | **13,734** | same  |
| e3m2x2 (FP6 E3M2) | 6,865  | **13,730** | same  |


### From-narrow to f16x2 (`cvt.rn.f16x2.{fmt}`)

1 op = 1 packed narrow pair -> 1 f16x2 (2 f16 elements) = **2 elements/op**


| Format            | GOps/s | GElems/s   | +relu |
| ----------------- | ------ | ---------- | ----- |
| e2m1x2 (FP4)      | 6,942  | **13,884** | same  |
| e4m3x2 (FP8 E4M3) | 6,941  | **13,882** | same  |
| e5m2x2 (FP8 E5M2) | 6,941  | **13,882** | same  |
| e2m3x2 (FP6 E2M3) | 6,940  | **13,880** | same  |
| e3m2x2 (FP6 E3M2) | 6,941  | **13,882** | same  |


All formats: **identical throughput** (~13.9 TElems/s). `.relu` is free.

### x4 Stochastic Rounding (`cvt.rs.satfinite.{fmt}.f32`)

1 op = 4 f32 inputs + rbits -> 1 packed narrow quad = **4 elements/op**
(Maps to 2 SASS `F2FP...PACK_AB_MERGE_C.RS` instructions per PTX instruction)


| Format            | GOps/s | GElems/s   | +relu |
| ----------------- | ------ | ---------- | ----- |
| e2m1x4 (FP4)      | 3,224  | **12,896** | same  |
| e4m3x4 (FP8 E4M3) | 3,424  | **13,696** | same  |
| e5m2x4 (FP8 E5M2) | 3,424  | **13,696** | same  |
| e2m3x4 (FP6 E2M3) | 3,424  | **13,696** | same  |
| e3m2x4 (FP6 E3M2) | 3,424  | **13,696** | same  |


---

## 3. Standard FP Conversions

### Scalar f32 -> f16 / bf16 (1 elem/op)


| Instruction                   | GOps/s | GElems/s  | Notes                                          |
| ----------------------------- | ------ | --------- | ---------------------------------------------- |
| cvt.{rn/rz}.f16.f32           | 4,043  | **4,043** | SASS: `F2F.F16.F32` (slow pipe)                |
| cvt.{rn/rz}.relu.f16.f32      | 6,441  | **6,441** | SASS: `F2FP.RELU.F16.F32.MERGE_C` (fast pipe!) |
| cvt.{rn/rz}.satfinite.f16.f32 | 6,441  | **6,441** | also uses `F2FP` fast pipe                     |
| cvt.{rn/rz}.bf16.f32          | 4,045  | **4,045** | SASS: `F2F.BF16.F32` (slow pipe)               |
| cvt.{rn/rz}.relu.bf16.f32     | 6,440  | **6,440** | SASS: `F2FP` fast pipe                         |


**Why +59% with .relu/.satfinite?** Bare `cvt.rn.f16.f32` compiles to SASS opcode `F2F.F16.F32` which runs on the legacy F2F pipe. Adding `.relu` or `.satfinite` causes the compiler to emit `F2FP.RELU.F16.F32.MERGE_C` instead, which runs on the newer, faster F2FP pipe. The modifier is essentially free -- the speedup comes from routing to a different functional unit.

### Packed f32,f32 -> f16x2 / bf16x2 (2 elems/op)


| Instruction                               | GOps/s | GElems/s    |
| ----------------------------------------- | ------ | ----------- |
| cvt.{rn/rz}{.relu}{.satfinite}.f16x2.f32  | ~7,390 | **~14,780** |
| cvt.{rn/rz}{.relu}{.satfinite}.bf16x2.f32 | ~7,389 | **~14,778** |


All modifier combinations: identical throughput. Always uses `F2FP` pipe.

### f32 -> tf32 (1 elem/op)


| Instruction                | GOps/s | GElems/s  | Notes                                            |
| -------------------------- | ------ | --------- | ------------------------------------------------ |
| cvt.rn.satfinite.tf32.f32  | 7,968  | **7,968** | fastest CVT                                      |
| cvt.rz.satfinite.tf32.f32  | 7,971  | **7,971** |                                                  |
| cvt.rna.satfinite.tf32.f32 | 3,837  | **3,837** | **2x slower** (RNA rounding uses different path) |


### f16/bf16 -> f32 upcast (1 elem/op)


| Instruction  | GOps/s  | GElems/s    | SASS     | Notes                                   |
| ------------ | ------- | ----------- | -------- | --------------------------------------- |
| cvt.f32.f16  | ~15,500 | **~15,500** | **PRMT** | bit shuffle on integer ALU, no F2F pipe |
| cvt.f32.bf16 | ~15,500 | **~15,500** | **PRMT** | same                                    |


**Why so fast?** The compiler does NOT use the F2F/F2FP pipe. `cvt.f32.f16` compiles to SASS `PRMT` (byte permute) -- it's just rearranging bits, not doing floating-point work. Runs on the general-purpose integer ALU at full throughput.

### UE8M0x2 Scale Factors (2 elems/op)


| Instruction                            | GOps/s | GElems/s    |
| -------------------------------------- | ------ | ----------- |
| cvt.{rz/rp}{.satfinite}.ue8m0x2.bf16x2 | ~5,940 | **~11,880** |
| cvt.{rz/rp}{.satfinite}.ue8m0x2.f32    | ~7,382 | **~14,764** |
| cvt.rn.bf16x2.ue8m0x2                  | 6,308  | **12,616**  |


---

## 4. MUFU (Special Function Unit) Throughput

### Architectural throughput per SM per cycle


| Unit               | Arch ops/SM/clk        | Measured ops/SM/clk | Chip GOps/s         | % SOL     | Notes                                |
| ------------------ | ---------------------- | ------------------- | ------------------- | --------- | ------------------------------------ |
| **FP32 FMA**       | **128** (x2=256 FLOPS) | 222 FLOPS           | 66,800 GFLOPS       | **86.8%** | 128 CUDA cores                       |
| **MUFU.EX2 f32**   | **32**                 | 29.8                | 8,968               | **93.1%** | **2x historical** (was 16)           |
| MUFU.EX2 f16       | **32**                 | 29.4                | 8,833               | 91.8%     | same as f32                          |
| MUFU.EX2 f16x2     | **16** (x2=32 elems)   | 14.9 (x2=29.8)      | 4,504 (9,008 elems) | 93.1%     | half-rate inst, 2x elems             |
| MUFU.EX2 bf16x2    | **16** (x2=32 elems)   | 14.9 (x2=29.8)      | 4,504 (9,008 elems) | 93.1%     | same as f16x2                        |
| **MUFU.LG2 f32**   | **16**                 | 15.0                | 4,503               | **93.6%** | historical rate                      |
| **MUFU.RSQRT f32** | **16**                 | 15.0                | 4,504               | **93.6%** | historical rate                      |
| **MUFU.SIN f32**   | **16**                 | 15.0                | 4,501               | **93.6%** | historical rate                      |
| **MUFU.COS f32**   | **16**                 | 15.0                | 4,502               | **93.6%** | historical rate                      |
| **MUFU.SQRT f32**  | **16**                 | 15.0                | 4,503               | **93.6%** | historical rate                      |
| **MUFU.RCP f32**   | **16**                 | 14.1                | 4,245               | **88.1%** | compiler prescale/postscale overhead |


**Key: Blackwell doubled MUFU.EX2 to 32 ops/SM/clk** (from 16 on prior architectures). All other MUFU functions remain at the historical 16 ops/SM/clk. The f16x2/bf16x2 packed variants are NOT faster per element -- they process 2 elements per instruction at half the instruction rate, netting the same element throughput.

### F2FP (narrow format conversion) per-SM throughput


| Kernel                | F2FP | Warp insts | Uniform (free) | Chip GOps/s | ops/SM/clk | Notes                       |
| --------------------- | ---- | ---------- | -------------- | ----------- | ---------- | --------------------------- |
| 8 F2FP (ULOP3 inputs) | 8    | 12         | 16             | 11,990      | 39.9       | 67% warp F2FP density       |
| 16 F2FP (LOP3 inputs) | 16   | 45         | 2              | 6,631       | 24.9       | 36% density, overhead-bound |


**Architectural F2FP throughput: ~32 ops/SM/clk** (consistent with sharing the MUFU.EX2 pipe). The variation (24.9-39.9) is entirely explained by warp instruction overhead:

- The 8-F2FP kernel gets 16 input XORs on the **uniform pipe** (ULOP3, free), leaving only 4 warp overhead instructions. Result: 67% F2FP density, close to F2FP-bound.
- The 16-F2FP kernel's input XORs compile to warp-pipe LOP3 (not uniform), adding 25 warp instructions. Result: 36% density, overhead-bound.

Clock scaling is linear (24.6/24.8/24.9 at 1200/1500/1800 MHz), confirming no throttling.

F2FP does **not** need high ILP: 16 independent CVTs already saturates the pipe. Adding more (32, 64, 128) gives ~same GOps/s. Unlike MUFU.EX2 (latency ~6-8 cycles needing 6-8 chains), F2FP saturates with far fewer independent ops, suggesting shorter pipeline or different scheduling.

### How this compares to prior architectures


| Arch                    | FP32 FMA/SM/clk | MUFU EX2/SM/clk | MUFU other/SM/clk | FMA:MUFU ratio              |
| ----------------------- | --------------- | --------------- | ----------------- | --------------------------- |
| SM 8.x (Ampere)         | 128             | 16              | 16                | 8:1                         |
| SM 9.0 (Hopper)         | 128             | 16              | 16                | 8:1                         |
| **SM 10.x (Blackwell)** | **128**         | **32**          | **16**            | **4:1 (EX2), 8:1 (others)** |


Blackwell halved the FMA:EX2 ratio from 8:1 to 4:1, making EX2 relatively more powerful.

### MUFU.EX2 ILP x TLP scaling (GOps/s, f32)

Shows how throughput scales with instruction-level parallelism (chains = independent dependency chains per thread) and thread-level parallelism (threads per block).


| Chains | t=128 | t=256 | t=512 | t=1024 |
| ------ | ----- | ----- | ----- | ------ |
| 1      | 1,991 | 3,985 | 7,212 | 8,537  |
| 2      | 3,895 | 7,211 | 8,638 | 8,732  |
| 4      | 6,711 | 8,692 | 8,846 | 8,879  |
| 6      | 7,601 | 8,734 | 8,875 | 8,917  |
| 8      | 7,961 | 8,845 | 8,934 | 8,944  |
| 16     | 8,422 | 8,917 | 8,941 | 8,967  |


**Latency analysis:** With 1 chain + 128 threads (4 warps/SM) = 1,991 GOps/s = 22% of peak. Doubling chains doubles throughput linearly until ~6-8 chains, then saturates. This implies **MUFU.EX2 latency ≈ 6-8 cycles**. With enough TLP (512+ threads), even 1 chain reaches ~80% peak -- the warp scheduler hides the latency across warps.

### MUFU.EX2 f16x2 ILP x TLP scaling (GOps/s)


| Chains | t=128 | t=256 | t=512 | t=1024 |
| ------ | ----- | ----- | ----- | ------ |
| 1      | 1,803 | 3,604 | 4,423 | 4,454  |
| 2      | 3,786 | 4,426 | 4,458 | 4,487  |
| 4      | 4,057 | 4,426 | 4,487 | 4,495  |
| 8      | 4,387 | 4,480 | 4,495 | 4,500  |
| 16     | 4,441 | 4,492 | 4,500 | 4,504  |


**f16x2 saturates much faster** (2 chains + 256 threads ≈ 98% peak). This implies **MUFU.EX2.F16x2 latency ≈ 2-4 cycles** -- shorter pipeline for narrower types.

### MUFU.LG2 f32 ILP x TLP scaling (GOps/s)


| Chains | t=128 | t=256 | t=512 | t=1024 |
| ------ | ----- | ----- | ----- | ------ |
| 1      | 870   | 1,741 | 3,473 | 4,418  |
| 2      | 1,612 | 2,737 | 4,454 | 4,467  |
| 4      | 3,028 | 4,263 | 4,486 | 4,492  |
| 8      | 4,069 | 4,457 | 4,493 | 4,500  |
| 16     | 4,112 | 4,433 | 4,500 | 4,503  |


Exactly half of EX2 at every configuration. **MUFU.LG2 is half-rate hardware**, not a software emulation -- the SASS shows a single `MUFU.LG2` instruction, same structure as EX2. SIN, COS, SQRT show identical scaling.

### MUFU SASS analysis

With `-use_fast_math` (QuickRunCUDA default), most MUFU variants compile to **clean SASS**:

```
EX2 loop (4 chains, UNROLL=1): 19 instructions, 4 MUFU
  4x FSETP (denormal check)    -- predicated, usually not-taken
  4x @!P FMUL (prescale)       -- only for denormals
  4x MUFU.EX2                  -- the actual work
  4x @!P FMUL (postscale)      -- only for denormals
  + loop control

LG2 / RSQRT / SQRT loop (4 chains, UNROLL=1): 19 instructions, 4 MUFU
  Same structure as EX2: FSETP + prescale + MUFU.{LG2/RSQ/SQRT} + postscale
  Half-rate is HARDWARE, not compiler bloat.

SIN / COS loop (4 chains, UNROLL=1): ~19 instructions, 4 MUFU
  Range reduction FMUL + MUFU.{SIN/COS} + fixup. Clean.

RCP loop (4 chains, UNROLL=1): 34 instructions, 4 MUFU
  Extra prescale for BOTH denormals AND overflow (2 FSETP + 2 FSEL + FMUL per chain).
  This is why RCP measures 88% vs 93% -- more compiler-generated overhead, not slower hardware.
```

At UNROLL=8 + 16 chains, the overhead grows proportionally:

- EX2: 128 MUFU / 179 total = 72% MUFU density
- RSQRT: 128 MUFU / 515 total = 25% density (overhead dominates at scale)
- RCP: 128 MUFU / 900 total = 14% density (massive prescale/postscale)

Despite the overhead bloat, **the measured GOps/s for RSQRT/LG2/SQRT are identical** (4,503 ±1), confirming the hardware throughput is the same -- the non-MUFU instructions execute on other pipes in parallel.

---

## 5. Instruction Mix: F2FP (e2m1x2) + Companion Instructions

Tests whether F2FP co-issues with other functional units. All runs: 4 e2m1x2 CVTs + varying companion count, persistent blocks, 256 threads.

### Time per iteration (ms) -- lower is better


| Companion      | 0 comp | 4 comp    | 8 comp    | 16 comp   | Co-issue?                                    |
| -------------- | ------ | --------- | --------- | --------- | -------------------------------------------- |
| (CVT only)     | 0.209  | --        | --        | --        | baseline                                     |
| **LOP3 (XOR)** | 0.209  | **0.213** | **0.213** | **0.213** | **ARTIFACT** (see note)                      |
| **FFMA**       | 0.209  | **0.211** | 0.249     | 0.418     | ~4 free, then contends                       |
| **FMUL**       | 0.209  | **0.215** | 0.273     | 0.418     | ~4 free, then contends                       |
| **IMAD**       | 0.209  | **0.224** | 0.256     | 0.416     | ~2 free, then contends                       |
| **IADD3**      | 0.209  | 0.244     | 0.276     | 0.344     | contends immediately                         |
| **MUFU (EX2)** | 0.209  | **0.371** | **0.584** | **1.084** | **SAME PIPE as F2FP** (confirmed, see below) |


### Interpretation


| Unit           | Pipe             | Relationship to F2FP                                                                                               |
| -------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| **LOP3/XOR**   | Integer logic    | **Fully independent** -- 0-16 LOP3s add zero cost to F2FP                                                          |
| **FFMA/FMUL**  | FP32 ALU         | **Partially independent** -- ~4 can co-issue free, then competes for issue bandwidth                               |
| **IMAD**       | Integer MAD      | **Partially independent** -- ~2 free, then contends                                                                |
| **IADD3**      | Integer ADD      | **Contends** -- every IADD3 adds time, likely shares issue slot                                                    |
| **MUFU (EX2)** | Special function | **SAME PIPE** -- F2FP and MUFU share the conversion/SFU pipe. Adding 4 MUFU nearly doubles time (0.209 -> 0.371ms) |


### SASS instruction counts (from same kernel, verified)


| Config       | F2FP | FFMA | LOP3 | IADD3 | IMAD | MUFU | Total |
| ------------ | ---- | ---- | ---- | ----- | ---- | ---- | ----- |
| 4 CVT only   | 16   | 0    | 30   | 0     | 1    | 0    | 55    |
| 4C + 4 FFMA  | 16   | 16   | 30   | 0     | 2    | 0    | 72    |
| 4C + 4 LOP3  | 16   | 0    | 31   | 0     | 1    | 0    | 56    |
| 4C + 4 IADD3 | 16   | 0    | 30   | 8     | 1    | 0    | 63    |
| 4C + 4 MUFU  | 16   | 0    | 30   | 0     | 2    | 16   | 75    |
| 4C + 16 LOP3 | 16   | 0    | 31   | 0     | 1    | 0    | 56    |


**LOP3 note:** The "free" LOP3 result was a **compiler artifact** -- `x ^= y` repeated N times folds to a no-op (even XOR). The SASS shows the same 31 LOP3 instructions regardless of N_COMP (0, 4, 8, or 16). Retesting with non-foldable integer ops (IMAD accumulator) shows they DO add time: 4C+4 IMAD = 0.252ms (+22%), 4C+16 IMAD = 0.693ms (+235%). Integer ALU work is **not free** alongside F2FP.

### Definitive pipe-sharing test (1500 MHz, no throttle)

Method: measure time(F2FP alone), time(X alone), time(F2FP+X combined). Compare to sum (shared) vs max (independent).


| Companion      | Comp ms | Mixed ms | Sum ms | Max ms | Blend | Verdict         |
| -------------- | ------- | -------- | ------ | ------ | ----- | --------------- |
| **MUFU.EX2**   | 0.191   | 0.479    | 0.455  | 0.264  | 1.13  | **SAME PIPE**   |
| **MUFU.LG2**   | 0.388   | 1.001    | 0.652  | 0.388  | 2.32  | **SAME PIPE**   |
| **MUFU.SIN**   | 0.356   | 0.673    | 0.619  | 0.356  | 1.20  | **SAME PIPE**   |
| **MUFU.RSQRT** | 0.388   | 1.001    | 0.652  | 0.388  | 2.32  | **SAME PIPE**   |
| **FFMA**       | 0.106   | 0.267    | 0.370  | 0.264  | 0.03  | **INDEPENDENT** |


F2FP baseline: 0.264ms. Blend: 0.0 = independent (max model), 1.0 = shared (additive model).

**F2FP = MUFU/SFU pipe.** All MUFU functions (EX2, LG2, SIN, RSQRT) are additive with F2FP, confirming they share hardware. FFMA is completely independent (blend=0.03, mixed ≈ max). LG2/RSQRT blend >2.0 because the compiler's denormal-handling code (FSETP+FMUL) creates additional warp ALU contention on top of the shared MUFU pipe.

---

## 6. Key Findings

### CVT / F2FP

1. `**.relu` and `.satfinite` are free** on packed/narrow CVTs, but **+59% faster on scalar f16/bf16** because bare `cvt.rn.f16.f32` uses the slow `F2F` SASS pipe while `.relu`/`.satfinite` variants use the fast `F2FP` pipe
2. **Packed x2 is ~83% faster than scalar** for f16/bf16 in ops/s (7.4K vs 4.0K), and **~3.7x in elems/s** (14.8K vs 4.0K)
3. **All from-narrow upcasts have identical throughput** regardless of format (~6.9K GOps/s = ~13.9K GElems/s)
4. **e2m1x2 (FP4) consistently ~7-10% slower** than FP8/FP6 due to .b8 register handling overhead
5. **x4 SR = same speed as x2 RNE at ISO-elements.** Verified: at 16 elems/iter (8 SASS F2FP each), x2=0.721ms vs x4=0.723ms. The earlier ~9% x4 advantage was from comparing at different element counts, not intrinsic speed. Stochastic rounding has NO throughput penalty vs RNE
6. **f16/bf16 -> f32 upcast compiles to PRMT** (byte permute), not F2F. Runs on integer ALU, genuinely very fast (~15.5K GOps/s)
7. **f32 -> bf16 scalar: RTZ is NOT faster than RNE** (both ~4K GOps/s on slow F2F pipe). Add `.satfinite` for +59% (routes to fast F2FP pipe)
8. **bf16x2 source/dest for narrow formats: UNSUPPORTED** on SM 10.3a
9. **FP6 (e2m3/e3m2) from f16x2: UNSUPPORTED** -- must use f32 pair source

### MUFU

1. **Blackwell doubled MUFU.EX2 to 32 ops/SM/clk** (was 16 on all prior architectures). All other MUFU functions (LG2, RSQRT, SIN, COS, SQRT, RCP) remain at historical 16 ops/SM/clk
2. **f16x2/bf16x2 EX2 is NOT 2x faster** per element -- processes 2 elements per instruction at half the instruction rate, netting the same ~9.0 TElems/s as f32
3. **MUFU.EX2 latency ≈ 6-8 cycles** (f32), **≈ 2-4 cycles** (f16x2/bf16x2). Narrower types have shorter pipeline
4. **RSQRT is clean** -- SASS shows just `MUFU.RSQ` with denormal prescale/postscale (same as LG2/SQRT). Half-rate is hardware, not compiler bloat. RCP is slower due to extra compiler overflow handling
5. **F2FP = MUFU/SFU pipe** (confirmed). F2FP is additive with ALL MUFU variants (EX2, LG2, SIN, RSQRT -- blend ≥1.0). **FFMA is fully independent** (blend=0.03, separate FP32 ALU pipe). This means narrow format conversions compete with transcendentals but NOT with FP32 math
6. **LOP3 "free" was a benchmark artifact** -- compiler folded repeated XOR to no-op. Non-foldable integer ops (IMAD) DO add cost alongside F2FP

### General

1. **RNA rounding (tf32) is 2x slower** than RN/RZ (3.8K vs 8.0K GOps/s)
2. **FP32 FMA reaches 86.9% SOL** with UNROLL=8, 1024 threads (66.9 TFLOPS)
3. **DRAM bandwidth: 78% SOL** (6,378 GB/s of 8,184 GB/s theoretical with ECC)

---

## 7. Instruction Availability Matrix (SM 10.3a)


| Direction   | Source/Dest | e2m1x2 | e4m3x2 | e5m2x2 | e2m3x2 | e3m2x2 |
| ----------- | ----------- | ------ | ------ | ------ | ------ | ------ |
| To-narrow   | f32 pair    | Y      | Y      | Y      | Y      | Y      |
| To-narrow   | f16x2       | Y      | Y      | Y      | **N**  | **N**  |
| To-narrow   | bf16x2      | **N**  | **N**  | **N**  | **N**  | **N**  |
| From-narrow | -> f16x2    | Y      | Y      | Y      | Y      | Y      |
| From-narrow | -> bf16x2   | **N**  | **N**  | **N**  | **N**  | **N**  |
| x4 stoch.   | f32 quad    | Y      | Y      | Y      | Y      | Y      |
| .relu       | all above   | Y      | Y      | Y      | Y      | Y      |


---

## 8. Files


| File                                 | Description                                           |
| ------------------------------------ | ----------------------------------------------------- |
| **Benchmark kernels**                |                                                       |
| `tests/bench_dram_bw.cu`             | DRAM bandwidth                                        |
| `tests/bench_fp32_fma.cu`            | FP32 FMA (configurable UNROLL)                        |
| `tests/bench_cvt_to_narrow_f16x2.cu` | f16x2 -> narrow                                       |
| `tests/bench_cvt_to_narrow_f32.cu`   | f32 pair -> narrow                                    |
| `tests/bench_cvt_from_narrow.cu`     | narrow -> f16x2                                       |
| `tests/bench_cvt_x4_f32.cu`          | x4 stochastic rounding                                |
| `tests/bench_cvt_generic.cu`         | Generic CVT template for sweep                        |
| `tests/bench_mufu.cu`                | MUFU throughput (configurable chains, types)          |
| `tests/bench_mix_e2m1.cu`            | Instruction mix (F2FP + companions)                   |
| `tests/test_cvt_correctness.cu`      | Correctness validation                                |
| **Scripts**                          |                                                       |
| `tests/run_microbench.sh`            | Main runner (`GPU_ID=N bash tests/run_microbench.sh`) |
| `tests/sweep_cvt.sh`                 | Full CVT sweep                                        |
| `tests/sweep_mix_e2m1.sh`            | Instruction mix sweep                                 |
| **Output**                           |                                                       |
| `sass/`                              | 107+ auto-saved SASS + CUBIN files                    |
| `ncu/`                               | 9 NCU profiles (sampling across tiers, `--set full`)  |
| `results_unrolled.txt`               | Main benchmark raw output                             |
| `sweep_results_unrolled.txt`         | Full CVT sweep raw output                             |
| `mix_results.txt`                    | Instruction mix sweep raw output                      |


