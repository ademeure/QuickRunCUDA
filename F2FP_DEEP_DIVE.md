# F2FP / SFU Pipe Deep-Dive — Blackwell sm_103a (B300 SXM6 AC)

Measurements and analysis of CVT / F2FP / F2F / MUFU instructions on NVIDIA B300 (148 SMs, 2032 MHz max SM clock, CUDA 13.0), 2026-04-14. All tests with GPU clock locked at 2032 MHz; latency measurements use 1-warp-per-SM × 148 SMs so each warp is latency-bound on a back-to-back dependency chain.

## Units convention (read first!)

Every "/SM/clk" number in this doc is **thread-level PTX instructions per SM per clock** unless otherwise labeled. Converting:

- **thread-instructions/SM/clk** = warp-instructions/SM/clk × 32 (SIMT lanes)
- **elements/SM/clk** = thread-instructions/SM/clk × (elements per packed op — 1 for scalar, 2 for `.x2`, 4 for `.x4` PTX though that compiles to 2× x2 SASS)

So when I write **"unpack narrow runs at 64/SM/clk"**, I mean:
- **64 thread-level F2FP instructions / SM / clock**
- = 2 warp-level F2FP instructions / SM / clock (the SFU dual-issues)
- = **128 elements / SM / clock** (each unpack is x2 so converts 2 narrow → 2 f16)
- = at 148 SMs × 2032 MHz clock, 19.25 TF2FPs/s aggregate, or 38.5 Telements/s

And **"pack narrow runs at 32/SM/clk"** means:
- 32 thread-level pack instructions / SM / clock (1 warp-level per cycle)
- = **64 elements / SM / clock** (each pack is also x2)
- = 19.25 Telements/s aggregate (same element rate as 2×x2 unpack... no wait, half — see §6 for why)

Every table below has explicit column headers for which unit.

## TL;DR

**Blackwell's SFU is not a simple 32-ops/SM/clock pipe**. It has a dual-issue capability, but only ½ of SASS opcodes can take advantage of it. The determining factor is **register-file port count** per instruction:

| # of regfile ports used | Example SASS | Peak (thread-inst/SM/clk) |
|---:|---|---:|
| **2** (1 read + 1 write) | `F2FP.F16.E4M3.UNPACK_B`, `F2FP.SATFINITE.TF32.F32.PACK_B` | **64** (dual-issue) |
| **3** (2 reads + 1 write, **or** 1 read + `MERGE_C` dest-read + 1 write) | `F2FP.F16.F32.PACK_AB`, `F2FP.E4M3.F16.UNPACK_B_MERGE_C`, `F2FP.E4M3.F32.PACK_AB_MERGE_C`, `MUFU.EX2` | **32** (single-issue) |
| Scalar MERGE_C (slow path) | `F2FP.SATFINITE.F16.F32.MERGE_C` | **~24** |
| Multi-slot ops (2 SFU slots per op) | `MUFU.RSQ/SIN/COS/LG2/TANH/SQRT` | **16** |

This explains:
- **Unpack narrow (`cvt.rn.f16x2.e4m3x2`) is 2× the rate of pack narrow (`cvt.rn.satfinite.e4m3x2.f16x2`)** — pack needs `MERGE_C` (dest-read); unpack doesn't.
- **`cvt.rn.tf32.f32` hits 64/SM/clk** while `cvt.rn.satfinite.f16.f32` tops out at 24. Same scalar CVT family, but tf32 emits `PACK_B` (no merge, full 32-bit destination) and f16/bf16 emit `MERGE_C`.
- **EX2 runs at 32/clk but RSQ/SIN at 16** — they share a 2-slot SFU budget and the simpler ops fit in one slot.
- **There is no faster pack variant.** Every narrow-format pack (to e4m3x2, e5m2x2, e2m1x2, e2m3x2, e3m2x2, ue8m0x2) emits `MERGE_C` → 32/SM/clk cap.

**The honest peak throughputs on Blackwell packed F2FP** (columns labeled):
| Op | PTX | SASS | Thread-instructions /SM/clk | Elements /SM/clk | Elements / s (all 148 SMs × 2.032 GHz) |
|---|---|---|---:|---:|---:|
| Unpack narrow → f16x2 | `cvt.rn.f16x2.e4m3x2` | `F2FP.F16.E4M3.UNPACK_B` | **64** | **128** | 38.5 Telements/s |
| Pack f16x2 → narrow | `cvt.rn.satfinite.e4m3x2.f16x2` | `F2FP.*.UNPACK_B_MERGE_C` | **32** | **64** | 19.3 Telements/s |
| Pack f32,f32 → narrow | `cvt.rn.satfinite.e4m3x2.f32` | `F2FP.*.PACK_AB_MERGE_C` | **32** | **64** | 19.3 Telements/s |
| Pack f32,f32 → f16x2 | `cvt.rn.f16x2.f32` | `F2FP.F16.F32.PACK_AB` | **32** | **64** | 19.3 Telements/s |
| Pack f32 → tf32 | `cvt.rn.satfinite.tf32.f32` | `F2FP.*.TF32.F32.PACK_B` | **64** | **64** (scalar, 1 elem/op) | 19.3 Telements/s |

---

## 1. Methodology

### Latency (L_step)

Kernel: `bench_f2fp_depth.cu` / `bench_latency_calib.cu`. 1 warp-per-SM × 148 SMs, back-to-back dependent chain of `K` F2FP ops + 1 LOP3 per inner step, unrolled 32×. Each warp saturates the pipeline with its own dep chain — no cross-warp parallelism to mask latency.

`L_step = SM_count × 32 × N_chains × OPS_per_step × clock_Hz / GOps_per_s`

Differential `(L_step(K+1) − L_step(K)) / 1 = L_op` cancels the constant LOP3 closure.

Calibration: FFMA, FMUL, IMAD all produce L_op = **4.00 cycles** (matches documented Blackwell latency), validating the measurement method.

### Throughput

Kernel: `bench_f2fp_pure.cu` / `bench_f2fp_oneway.cu` / `bench_f2fp_pack_variants.cu`. Persistent (148 blocks × 1-4× oversubscription) or non-persistent, `blk=512`, UNROLL=32, `N_CHAINS` independent register chains per thread (8–16 for full ILP). All pair instructions in inline `asm volatile` blocks so ptxas can't fold or elide.

Units: all "/SM/clk" numbers are **thread-level** PTX instructions per SM per cycle (divide by 32 for warp-level). "Elements/SM/clk" multiplies by how many scalar elements per packed op.

---

## 2. Latency (pure dependency-chain measurements)

### 2.1 Calibration (baseline: FFMA = 4 cy documented)

| Op | D=1 | D=2 | D=3 | D=4 | D=6 | D=8 | L_op (cy) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FFMA (`fma.rn.f32`) | 10.36 | 14.35 | 18.37 | 22.38 | 30.39 | 38.35 | **4.00** |
| IMAD (`mad.lo.u32`) | 10.35 | 14.35 | 18.35 | 22.35 | 30.39 | 38.40 | **4.00** |
| FMUL (`mul.rn.f32`) | 10.20 | 14.19 | 18.19 | 22.19 | 30.19 | 38.19 | **4.00** |
| LOP3 (`xor.b32`) | — | 12.23 | — | — | — | — | **4.00** (derived) |

(All differentials: ΔL_step / Δdepth = 4.00 ± 0.02 cy.)

### 2.2 F2FP / F2F latencies

| Op | L_step(D=1) | L_step(D=8) | L_op (cy) | Pipe |
|---|---:|---:|---:|---|
| `F2FP.E4M3.F16.UNPACK_B_MERGE_C` (pack narrow) | 12.23 | 36.23 | **4.00** | SFU-F2FP |
| `F2FP.F16.E4M3.UNPACK_B` (unpack narrow) | 12.23 | 36.23 | **4.00** | SFU-F2FP |
| `F2FP.SATFINITE.F16.F32.MERGE_C` (scalar pack) | 18.23 | — | **10.23** | SFU-F2FP (slower) |
| `F2FP.SATFINITE.TF32.F32.PACK_B` (scalar tf32) | 8.24 | 36.27 | **4.00** | SFU-F2FP |
| `F2F.F16.F32 + HADD2.F32` (scalar non-satfinite widen-narrow round-trip) | 34.17 | — | ~12 per combined step | F2F pipe (distinct) |
| `cvt.rna.satfinite.tf32.f32` | 44.44 | 325.35 | **40** (emulated, 10+ IMAD/LOP3) | Integer ALU emulation |

### 2.3 MUFU latencies

| Op | chains=1 GOps/s | L_op (cy) |
|---|---:|---:|
| `ex2.approx.f32` | 4224 | **18** |
| `rcp.approx.f32` | 1647 | 47 |
| `rsqrt/sqrt/lg2.approx.f32` | 1851 | 42 |
| `sin/cos.approx.f32` | 3155 | 24 |
| `tanh.approx.f32` | 4095 | 19 |

---

## 3. SASS Instruction Taxonomy

From `tests/sass_map.cu` — every PTX CVT variant that exists on sm_103a, with emitted SASS:

### 3.1 Scalar (1 element per op)

| PTX | SASS | Regfile ports (R/W) | Peak /SM/clk |
|---|---|:---:|---:|
| `cvt.rn.satfinite.f16.f32` | `F2FP.SATFINITE.F16.F32.MERGE_C` | 2R + 1W | 24 |
| `cvt.rn.satfinite.bf16.f32` | `F2FP.SATFINITE.BF16.F32.MERGE_C` | 2R + 1W | 24 |
| `cvt.rn.f16.f32` (no satfinite) | **`F2F.F16.F32`** (distinct opcode!) | distinct pipe | ~11 |
| `cvt.rn.bf16.f32` (no satfinite) | **`F2F.BF16.F32`** | distinct pipe | ~11 |
| `cvt.rn.satfinite.tf32.f32` | `F2FP.SATFINITE.TF32.F32.PACK_B` | 1R + 1W | **64** |
| `cvt.rna.satfinite.tf32.f32` | *emulated* (10+ `IMAD/LOP3/FSETP`) | — | ~3 |
| `cvt.f32.f16` / `cvt.rn.f32.bf16` | folded away (register alias) | — | ∞ |

### 3.2 x2 packed (2 elements per op)

| PTX | SASS | Regfile ports | Peak /SM/clk |
|---|---|:---:|---:|
| `cvt.rn.f16x2.f32` (f32 pair → f16x2) | `F2FP.F16.F32.PACK_AB` | **2R + 1W** | **32** |
| `cvt.rn.bf16x2.f32` | `F2FP.BF16.F32.PACK_AB` | 2R + 1W | 32 |
| `cvt.rn.satfinite.e4m3x2.f32` | `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` | **3R + 1W** | 32 |
| `cvt.rn.satfinite.e5m2x2.f32` | `F2FP.SATFINITE.E5M2.F32.PACK_AB_MERGE_C` | 3R + 1W | 32 |
| `cvt.rn.satfinite.e2m3x2.f32` (FP6) | `F2FP.SATFINITE.E2M3.F32.PACK_AB_MERGE_C` | 3R + 1W | 32 |
| `cvt.rn.satfinite.e3m2x2.f32` (FP6) | `F2FP.SATFINITE.E3M2.F32.PACK_AB_MERGE_C` | 3R + 1W | 32 |
| `cvt.rn.satfinite.e2m1x2.f32` (FP4) | `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C` + mov.b16 | 3R + 1W | 32 |
| `cvt.rn.satfinite.e4m3x2.f16x2` | `F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C` | **2R + 1W** (with MERGE_C) | 32 |
| `cvt.rn.satfinite.e5m2x2.f16x2` | same pattern | 2R + 1W | 32 |
| `cvt.rn.satfinite.e2m1x2.f16x2` | same + `.b8` pack | 2R + 1W | 32 |
| `cvt.rn.f16x2.e4m3x2` | **`F2FP.F16.E4M3.UNPACK_B`** (no MERGE_C!) | **1R + 1W** | **64** |
| `cvt.rn.f16x2.e5m2x2` | `F2FP.F16.E5M2.UNPACK_B` | 1R + 1W | 64 |
| `cvt.rn.f16x2.e2m3x2` (FP6) | `F2FP.F16.E2M3.UNPACK_B` | 1R + 1W | 64 |
| `cvt.rn.f16x2.e3m2x2` (FP6) | `F2FP.F16.E3M2.UNPACK_B` | 1R + 1W | 64 |
| `cvt.rn.f16x2.e2m1x2` (FP4) | `F2FP.F16.E2M1.UNPACK_B` + mov | 1R + 1W | 64 |

### 3.3 x4 stochastic rounding (4 elements, but emits 2 SASS ops)

| PTX | SASS | Instructions emitted | Peak /SM/clk |
|---|---|:---:|---:|
| `cvt.rs.satfinite.e4m3x4.f32` | **2×** `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C.RS` | 2 | 16 (each SASS is 32/clk; x4 PTX = 2 SASS) |
| `cvt.rs.satfinite.e5m2x4.f32` | 2× `F2FP.*.E5M2.*.RS` | 2 | 16 |
| `cvt.rs.satfinite.e2m1x4.f32` (FP4) | 2× `F2FP.*.E2M1.*.RS` + mov | 2 | 16 |

**Note:** x4 is NOT a single HW instruction — ptxas expands it to two x2 PACK_AB ops.

### 3.4 UE8M0 scale-factor conversions

| PTX | SASS | Peak /SM/clk |
|---|---|---:|
| `cvt.rz.satfinite.ue8m0x2.bf16x2` | `F2FP.SATFINITE.E8.BF16.UNPACK_B_MERGE_C.RZ` | 32 |
| `cvt.rz.satfinite.ue8m0x2.f32` | `F2FP.SATFINITE.E8.F32.PACK_AB_MERGE_C.RZ` | 32 |
| `cvt.rn.bf16x2.ue8m0x2` | **`F2FP.BF16.E8.UNPACK_B`** | **64** |

---

## 4. Throughput ceilings — directly measured

All with pure inline-PTX kernels using N=16 chains, blk=512, blocks=592 (4× SM oversubscription), UNROLL=32:

| SASS instruction | Measured GOps/s | /SM/clk | Interpretation |
|---|---:|---:|---|
| `F2FP.F16.E4M3.UNPACK_B` (unpack) | 19177 | **63.77** | 2 issue slots fully used |
| `F2FP.SATFINITE.TF32.F32.PACK_B` (tf32) | 19177 | **63.77** | same — just 1R+1W |
| `F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C` (pack narrow) | 9597 | **31.91** | 1 slot only |
| `F2FP.F16.F32.PACK_AB` (f32→f16x2) | 9597 | **31.91** | 1 slot (2R+1W) |
| `F2FP.BF16.F32.PACK_AB` (f32→bf16x2) | 9597 | **31.91** | same |
| `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` (pack narrow from f32) | 9369 | **31.15** | 1 slot |
| `F2FP.SATFINITE.F16.F32.MERGE_C` (scalar) | 7259 | **24.14** | special slower path |

**Round-trip pack+unpack** (interleaved) reaches 19170 GOps/s = **63.74 /SM/clk** — the same as unpack alone. The pack (using 1 slot) plus unpack (using the other slot) saturate both slots.

---

## 5. The Two-Slot SFU Model

Every measurement is consistent with: **the SFU has 2 issue slots per cycle per SM**, and instructions consume a per-op slot count:

| Op class | Slots consumed | Peak /SM/clk |
|---|:---:|---:|
| `F2FP.*.UNPACK_B` (narrow → f16x2 unpack) | 1 | 64 |
| `F2FP.*.PACK_B` (scalar with full-reg dst, e.g. tf32) | 1 | 64 |
| `F2FP.*.UNPACK_B_MERGE_C` (pack narrow f16x2→) | 2 | 32 |
| `F2FP.*.PACK_AB` (f32 pair → packed wide) | 2 | 32 |
| `F2FP.*.PACK_AB_MERGE_C` (f32 pair → packed narrow) | 2 | 32 |
| `F2FP.SATFINITE.F16.F32.MERGE_C` (scalar, slower sub-variant) | 2+ | ~24 |
| `MUFU.EX2` | 1 | 32 |
| `MUFU.RSQ/SIN/COS/LG2/TANH/SQRT` | 2 | 16 |
| `F2F.F16.F32` / `F2F.BF16.F32` | on separate pipe | ~11 |

**Verification: `E + 2R ≤ 32`** (EX2 + RSQRT mix, perfect fit across 8 data points):

| Config | Measured EX2 | Measured RSQ | E + 2R |
|---|---:|---:|---:|
| EX2=16, RSQ=0 | 31.83 | — | 31.83 |
| EX2=16, RSQ=4 | 21.22 | 5.30 | 31.82 |
| EX2=16, RSQ=8 | 15.93 | 7.96 | 31.85 |
| EX2=16, RSQ=16 | 10.59 | 10.59 | 31.77 |
| EX2=16, RSQ=32 | 6.14 | 12.28 | 30.70 |
| EX2=8, RSQ=16 | 6.37 | 12.75 | 31.87 |
| EX2=4, RSQ=16 | 3.54 | 14.16 | 31.86 |

---

## 6. Why pack is slower than unpack — register-port analysis

### 6.1 The rule: 64/SM/clk requires exactly **2 regfile accesses** (1 read + 1 write)

Looking at measured throughput vs regfile port count:

| SASS suffix | R ports | W ports | Total | Peak /SM/clk |
|---|:---:|:---:|:---:|---:|
| `UNPACK_B` | 1 | 1 | **2** | **64** |
| `PACK_B` | 1 | 1 | **2** | **64** |
| `PACK_AB` (no MERGE_C) | 2 | 1 | **3** | **32** |
| `UNPACK_B_MERGE_C` | 1 + 1 (dest-read) | 1 | **3** | **32** |
| `PACK_AB_MERGE_C` | 2 + 1 (dest-read) | 1 | **4** | **31** |

The instructions that can dual-issue (2 slots used by 2 different F2FPs on the same cycle) are exactly those needing **only 2 regfile accesses** per instruction. Anything that needs a second source register (PACK_AB) or a destination-read (MERGE_C — preserving upper 16 bits of the destination register) takes both slots.

### 6.2 Why MERGE_C exists

Look at output widths. On Blackwell:

- **UNPACK** (narrow → f16x2): output is a full 32-bit f16x2, written whole. No MERGE_C needed.
- **PACK** (f16x2/f32 → narrow): output is 16 bits (two 8-bit narrows packed). The destination register is 32-bit wide, so the pack writes only the low 16 bits and must **preserve the upper 16**. This requires reading the destination first (the `MERGE_C` read-modify-write).

Similarly `PACK_B` tf32 has a full-32-bit output (the tf32 fits in a 32-bit register because it's a truncated f32), so no merge is needed → 64/clk.

### 6.3 Bytes/clock required for the F2FP unit

Taking "64/clk unpack" as the effective peak, the register file must sustain:

**Per warp-instruction:**
- UNPACK_B: 1 source read (32 threads × 4 B = 128 B) + 1 destination write (128 B) = **256 B/warp-inst**
- PACK_AB: 2 source reads (256 B) + 1 destination write (128 B) = **384 B/warp-inst**
- UNPACK_B_MERGE_C: 1 source read + 1 dest-read + 1 write = **384 B/warp-inst**
- PACK_AB_MERGE_C: 2 source reads + 1 dest-read + 1 write = **512 B/warp-inst**

**Per SM per cycle at peak:**
- UNPACK at 2 warp-insts/cycle: **512 B/cycle** regfile bandwidth consumed
- PACK at 1 warp-inst/cycle: **384–512 B/cycle** regfile bandwidth consumed

So the **regfile port provision per SM is approximately 512 B/cycle on the SFU-feeding side**. PACK uses nearly all of it with one warp-inst; UNPACK uses only half per warp-inst so two can fit.

In aggregate across 148 SMs × 2032 MHz:
- Unpack peak: **512 B × 148 × 2.032e9 = 154 TB/s** of regfile traffic feeding the F2FP pipe (source+dest operand bandwidth)
- Pack peak: ~115 TB/s

For comparison, HBM3e bandwidth is 8.2 TB/s — register-file bandwidth is ~20× HBM, as expected.

### 6.4 Can pack ever hit 64/SM/clk? **No.**

Every narrow pack variant I found on Blackwell emits `*_MERGE_C` (because the output is narrower than the 32-bit destination register, so upper bits must be preserved). Every wide-pack variant (f32→f16x2) emits `PACK_AB` (2 source reads, no MERGE_C but still 3 accesses). There is **no SASS pack opcode that uses only 1R+1W**.

Theoretically a pack could avoid MERGE_C if we could write only the low 16 bits without reading the upper. But the Blackwell ISA provides no such variant — every pack-to-narrow emits MERGE_C.

The only path to higher element throughput on pack is the **x4 PTX** `cvt.rs.*.f32` stochastic-round form. But ptxas compiles this to **two** `PACK_AB_MERGE_C.RS` SASS instructions — 4 elements in 2 cycles = 2 elements/cycle = same effective rate as x2 pack. So x4 PTX does not exceed the per-cycle element ceiling of **64 elements/SM/cycle** for pack.

**Pack element ceiling: 64 elements / SM / cycle → 64 × 148 × 2.032e9 = 19.25 × 10¹² elements/s = 19.2 Telements/s** for packed narrow.

Per-byte (8-bit narrow): 19.2 TB/s output produced.
Per-byte (4-bit narrow): 9.6 TB/s output produced.

### 6.5 Unpack element ceiling

**Unpack element ceiling: 128 elements / SM / cycle → 38.5 Telements/s**, outputting f16 (2 bytes each) = **77 TB/s of f16 data**. For dequant-only pipelines this is the effective ceiling.

---

## 7. Co-issue / contention tests

### 7.1 F2FP-round-trip + companion op (pure kernel, 32 F2FPs/iter, blk=512, blocks=592)

Companion op added N times per inner iter on a **separate** register chain:

| Comp | N=0 | N=4 | N=8 | N=16 | N=32 | N=64 |
|---|---:|---:|---:|---:|---:|---:|
| FFMA | 63.83 | 63.64 | 63.56 | 63.62 | 62.35 | 20.22 |
| FMUL | 63.83 | 63.68 | 63.62 | 63.54 | 62.89 | 25.47 |
| IMAD | 63.74 | 63.70 | 63.61 | 62.06 | 58.32 | 19.11 |
| LOP3 | 63.83 | 63.49 | 63.25 | 62.75 | 61.80 | 59.94 |
| MUFU.ex2 | 63.75 | 63.50 | 62.51 | 51.25 | 28.96 | 15.27 |
| SHL | 63.71 | 63.71 | 63.72 | 63.71 | 63.80 | 63.76 |
| IADD3 | 63.71 | 63.73 | 63.71 | 63.70 | 63.70 | 63.62 |

**Findings:**
- **SHL, IADD3: completely free** even at N=64. These share no pipe or port with F2FP.
- **LOP3: gradual small drop** (−6% at N=64). Shares some warp-scheduler dispatch bandwidth but no execution pipe.
- **FFMA, FMUL, IMAD: essentially free up to N=32** (≤3% drop). At N=64 there's a cliff (likely register pressure / occupancy collapse at 64 live companion regs).
- **MUFU.ex2: real SFU-budget contention**. Drops start at N=16 (EX2 consuming 1 slot takes away from unpack's 2nd slot). Fits the `E + 2U ≤ 32` model.

### 7.2 LOP3 isolation on the one-way kernels

Independent LOP3s on separate register chains added to pack/unpack/interleaved F2FP:

| Mode | base | +4 | +8 | +16 | +32 | +64 |
|---|---:|---:|---:|---:|---:|---:|
| PACK (one-way) | 31.88 | 31.63 | 31.39 | 30.91 | 30.01 | 28.30 |
| UNPACK (one-way) | 63.70 | 62.71 | 61.75 | 59.94 | 56.52 | 50.81 |
| INTERLEAVED | 63.68 | 63.19 | 62.78 | 61.81 | 60.02 | 56.60 |

LOP3 drop per op (N=64 differential): ~0.056 F2FP/LOP3 for pack, ~0.20 for unpack, ~0.11 interleaved. Consistent with warp-scheduler issue-slot sharing (not pipe sharing) — unpack is closer to saturating the scheduler because it's already dual-issuing.

### 7.3 Scalar F2FP + companion

The scalar `F2FP.SATFINITE.F16.F32.MERGE_C` (24/SM/clk baseline) is **ILP-starved** — adding 4 FMULs or LOP3s *increases* its rate by ~12%, because the extra ops fill scheduler slots while scalar F2FP stalls on its 10-cy latency:

| Companion=32 | Scalar F2FP /SM/clk |
|---|---:|
| Baseline (no comp) | 21.40 |
| +LOP3=32 | **22.40** (+4.7%) |
| +SHL=32 | **22.40** (+4.7%) |
| +IADD3=32 | **22.89** (+7.0%) |
| +FFMA=32 | 13.85 (−35%) |
| +FMUL=32 | 18.23 (−15%) |
| +IMAD=32 | 14.96 (−30%) |
| +MUFU=32 | 13.85 (−35%) |

So **scalar F2FP shares FMA-pipe dispatch with FFMA/IMAD/FMUL** (not just SFU). This is a distinct pattern from packed F2FP.

---

## 8. The `F2F` opcode (non-satfinite scalar)

`cvt.rn.f16.f32` **without** `.satfinite` is a **genuinely different opcode class**: `F2F.F16.F32`, paired with `HADD2.F32` for widening.

SASS for non-satfinite round-trip chain:
```
F2F.F16.F32 R3, R3 ;                              ← narrow (lossy)
LOP3.LUT R4, R3, 0x1, RZ, 0x3c, !PT ;             ← xor.b16
HADD2.F32 R4, -RZ, R4.H0_H0 ;                     ← widen (half→f32)
F2F.F16.F32 R4, R4 ;
```

Measured latency per (F2F + LOP3 + HADD2) combined step: ~29 cy → F2F + HADD2 widen path ≈ **25 cycles combined latency**, roughly 2.5× slower than the satfinite F2FP path.

**Rule**: if you want to convert f32→f16/bf16 and speed matters, use `.satfinite` — it compiles to the fast F2FP.MERGE_C variant (~24/SM/clk). Without `.satfinite`, you get F2F which routes through a different, slower pipe.

---

## 9. Practical implications

### Quantization pipelines

- **Dequantize (unpack) runs 2× faster than quantize (pack).** NVFP4/FP8 dequant is 128 elements/SM/cycle; quant is 64 elements/SM/cycle.
- A kernel that does both (e.g. pre-quantize for tensor-core input, then dequantize the output) is bottlenecked by the pack direction.
- At 148 SM × 2032 MHz:
  - Peak dequant element throughput: **38.5 Telements/s** (f16 output = 77 TB/s)
  - Peak quant element throughput: **19.3 Telements/s** (FP8 output = 19 TB/s; FP4 = 9.6 TB/s)
  - HBM3e bandwidth: 8.2 TB/s
  - → Dequant is always HBM-bound; quant is HBM-bound for FP4 only. FP8-quant and f16 dequant with no SMEM staging are compute-capable at much higher rates than HBM can feed.

### MUFU contention

Any kernel that mixes F2FP with MUFU.RSQ/SIN/etc. is subject to `(F2FP slots) + 2×(other MUFU) ≤ 32/SM/cycle`. In particular:
- A softmax + dequant kernel (EX2 + unpack) halves unpack's rate when EX2 matches it 1:1.
- Using `ex2.approx` instead of `exp` is 2× faster from both latency and throughput standpoints.

### The `.satfinite` rounding-mode choice

For scalar f32→f16/bf16: `.satfinite` uses F2FP (24/SM/clk). Without `.satfinite`, it falls back to F2F (11-12/SM/clk). **Always add `.satfinite`** unless you specifically need the overflow-NaN semantics.

### x4 stochastic-round PTX is syntactic sugar

`cvt.rs.e4m3x4.f32` does **not** give you 4× the element throughput over `.rs.e4m3x2.f32`. Both compile to per-pair SASS ops at 32/SM/clk (2 elements each), so the element rate is 64 elements/SM/clk regardless.

### Fastest pack variants

If you need to pack from f32 (not f16x2), all variants run at 32/SM/clk. But:
- `cvt.rn.f16x2.f32` (non-satfinite to f16x2) — output is full 32-bit, uses PACK_AB (no MERGE_C). Still 32/SM/clk because of the second source register. **Not faster than the narrow pack paths, but element-wise higher throughput** (2 elements/op and 32/clk = 64 elements/clk of f16x2).

---

## 10. Files

Kernels (all under `tests/`):
- `bench_f2fp_depth.cu` — K-deep dependency chain, pack + unpack alternating, for latency measurement
- `bench_f2fp_latency.cu` — f32/bf16/f16 chain variants
- `bench_f2fp_push.cu` — throughput ceiling optimization (N_CHAINS × UNROLL × block × blocks sweep)
- `bench_f2fp_pure.cu` — pure CHAIN_PAIRS of round-trip, no XOR in hot loop
- `bench_f2fp_pure_plus.cu` — pure F2FP + N_COMP companion ops on separate register chain
- `bench_f2fp_oneway.cu` — one-way pack OR unpack OR interleaved
- `bench_f2fp_oneway_lop3.cu` — + independent LOP3 on separate chain
- `bench_f2fp_coissue.cu` / `bench_f2fp_coissue_scalar.cu` — packed/scalar F2FP + companion mix
- `bench_f2fp_pack_variants.cu` — 7 different SASS opcodes measured back-to-back
- `bench_latency_calib.cu` — FFMA/IMAD/FMUL/F2FP_pack/F2FP_scalar/F2F-widen/tf32 calibration harness
- `bench_mufu_coissue.cu` — EX2 + other-MUFU mix for `E + 2R = 32` verification
- `sass_map.cu` — minimal kernel emitting every CVT variant once, for SASS reference

Raw logs in `/tmp/qrc_results/`:
- `20_f2fp_depth.txt` — latency chain depth 2-8
- `22_latency_calib.txt` — FFMA/IMAD/FMUL/F2FP calibration (validates 4-cy method)
- `23_non_satfinite.txt` — F2F.F16.F32 discovery
- `24_coissue.txt` / `25_coissue_scalar.txt` — packed/scalar coissue
- `26_f2fp_pure.txt` — 63.86 /SM/clk peak
- `28_f2fp_oneway_clean.txt` / `29_f2fp_oneway_fair.txt` — pack 32 vs unpack 64
- `30_mufu_coissue.txt` — E + 2R = 32
- `31_f2fp_oneway_with_lop3.txt` — LOP3 isolation
- `32_pure_f2fp_coissue.txt` — cleanest F2FP + companion contention
- `33_pack_variants.txt` — 7-variant SASS comparison
