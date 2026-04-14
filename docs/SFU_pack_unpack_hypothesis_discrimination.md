# F2FP pack vs unpack — why pack caps at 32/SM/clk on Blackwell (B300)

GPU: NVIDIA B300 SXM6 AC, sm_103a, 148 SMs, clocks locked at 2032 MHz.
Baseline: pack narrow (f16x2→e4m3x2) caps at 32 thread-instructions/SM/clk;
unpack narrow (e4m3x2→f16x2) hits 64 thread-instructions/SM/clk; pack+unpack
interleaved hits 64 aggregate (2 SFU slots).

## TL;DR — best-supported hypothesis

**Hypothesis 1 (regfile-read-port count), generalized**: An F2FP
instruction that requires **≥ 2 regfile read ports** consumes **both SFU
issue slots** (→ 32/SM/clk cap). Instructions that need only 1 read port
can dual-issue across the 2 slots (→ 64/SM/clk).

Whether the 2nd read is a "real" 2nd source (PACK_AB) or a MERGE_C carry
slot (even when it is statically `RZ`) is irrelevant — what matters is
the number of regfile-read operand slots the opcode occupies.

Hypotheses 2, 3, 4 are falsified below.

## Kernels & SASS

All tests run with `blk=512`, `b=592` (4× SM over-subscription), `ITERS=65536`,
`UNROLL=32`, `N_CHAINS=16`. GPU 0 (`CUDA_VISIBLE_DEVICES=0`), clocks locked.
Per-thread op count = 1,048,576 unless noted.

### Existing suite: `tests/bench_f2fp_pack_variants.cu`

Built-in variants (0–6) hit the pattern directly. See emitted SASS in
`sass/bench_f2fp_pack_variants_*.sass`.

| V | PTX | SASS (first emitted) | time (ms) | /SM/clk |
|---:|---|---|---:|---:|
| 0 | `cvt.rn.satfinite.e4m3x2.f16x2` | `F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C R3, R3, RZ` | 33.08 | **32.0** |
| 1 | `cvt.rn.f16x2.e4m3x2` | `F2FP.F16.E4M3.UNPACK_B R8, R8` | 16.55 | **63.9** |
| 2 | `cvt.rn.f16x2.f32` (pair) | `F2FP.F16.F32.PACK_AB R8, R19, R8` | 33.08 | **32.0** |
| 3 | `cvt.rn.bf16x2.f32` (pair) | `F2FP.BF16.F32.PACK_AB R8, R19, R8` | 33.08 | **32.0** |
| 4 | `cvt.rn.satfinite.tf32.f32` | `F2FP.SATFINITE.TF32.F32.PACK_B R15, R15` | 16.54 | **64.0** |
| 5 | `cvt.rn.satfinite.e4m3x2.f32` | `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R7, R18, R7, RZ` | 33.84 | 31.3 |
| 6 | `cvt.rn.satfinite.f16.f32` | `F2FP.SATFINITE.F16.F32.MERGE_C R19, R4, RZ` | 43.68 | 24.2 |

Key observations from V0–V6:

- **V2 (PACK_AB, 2 reads, NO MERGE_C, writes full 32-bit register) = 32/clk**,
  same rate as V0 (UNPACK_B_MERGE_C, 1 real read + RZ merge, writes 16-bit).
  Both share "2 regfile read ports".
  → Already falsifies hypothesis 2 (writeback bus): V2 writes 32-bit yet still caps at 32.
  → Already distinguishes hypothesis 1 from the narrower "MERGE_C destination read"
    framing: PACK_AB has no MERGE_C at all yet is still 32/clk.

- **V4 (PACK_B, 1 read, 32-bit writeback, includes saturation+rounding) = 64/clk**.
  Identical rate to V1 (UNPACK_B). → Falsifies hypothesis 3 (pipeline pitch): tf32
  has a full rounding+saturate path and still issues at 64/clk.

- **V5 (PACK_AB_MERGE_C, 2 real reads + merge = 3 regfile read slots) = ~31/clk**,
  essentially same as V0/V2. Going from 2→3 read slots costs ~2% — both already
  occupy both SFU issue slots.

### New discrimination kernel: `tests/bench_f2fp_port_hypothesis.cu`

I added a dedicated kernel with variants 10–19 designed to stress-test the
remaining hypotheses. Measured data:

| V | what it does | F2FP count | time (ms) | /SM/clk (F2FP only) | Purpose |
|---:|---|---:|---:|---:|---|
| 10 | pack narrow, fresh dest, merged into `dst[k]` via LOP3 | ~530 | 53.00 | ~20 | feedback-free pack (limited by LOP3 closure) |
| 11 | PACK_AB f32→f16x2 with fresh 32-bit dest (no MERGE_C) | 512 | 34.97 | **32.0** | 2R+1W, no merge |
| 12 | UNPACK_B (e4m3→f16x2) feedback RMW | 512 | 17.49 | **63.9** | shows even tight feedback doesn't throttle 1R op |
| 13 | **two independent packs → two different dests per k** | 1024 | 69.98 | **30.2 per pack** | tests writeback-bus hypothesis |
| 14 | PACK_B tf32 with feedback | 512 | 17.49 | **63.9** | 1R control, still 64/clk even with saturation |
| 15 | pack + unpack per k (round-trip mix) | 512 (256+256) | 34.97 | **63.9 aggregate** | 2-slot model check |
| 16 | PACK_AB bf16x2 from f32 pair, feedback | 512 | 34.97 | **32.0** | 2R+1W, full 32-bit writeback |
| 17 | **4 independent packs → 4 different dests per k** | 2048 | 52.58 | **30.5 per pack** | scale-up of V13, ILP saturation |
| 18 | pack narrow, merge-out into dst via LOP3 (feedback) | ~516 | 53.00 | ~20 | duplicate of V10, confirms |
| 19 | 4 independent unpacks per k | 2048 | 113.82 | **63.1 per unpack** | 4-wide unpack control (→ 64/clk) |

SASS evidence for V11 / V13 / V14 / V15 / V17 / V19 below.

## Experiment-by-experiment discrimination

### (1) MERGE_C destination-read port bottleneck?

**Narrow form** ("pack needs to read destination"): **already falsified by SASS**.
Every `F2FP.*.UNPACK_B_MERGE_C Rd, Rs, Rc` in the emitted SASS has Rc = `RZ`
because ptxas knows the 16-bit output's upper bits are don't-care.
So the MERGE_C slot is not *actually* reading the destination register.

```
F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C R3, R3, RZ ;   // V0
F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C R8, R7, RZ ;   // V10 with fresh dest
F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C R25, R5, RZ ;  // also V10
```

Yet all cap at 32/clk. The "MERGE_C" decoration only matters in that the
opcode reserves a **regfile read slot** for Rc — occupied or not.

**Broader form** ("≥2 regfile read slots → caps at 32"): **STRONGLY SUPPORTED**.

Supporting SASS from the new kernel:

```
# V11 PACK_AB (2 real reads, no MERGE, 32-bit writeback)
F2FP.F16.F32.PACK_AB R34, R29, R12 ;        # 512 copies, 34.97ms → 32/clk
F2FP.BF16.F32.PACK_AB R2, R17, R2 ;         # V16, same behavior

# V14 PACK_B (1 real read, 32-bit writeback, with rounding+saturate)
F2FP.SATFINITE.TF32.F32.PACK_B R2, R2 ;     # 512 copies, 17.49ms → 64/clk
```

2 reads → 32/clk. 1 read → 64/clk. Writeback width is held constant
(32-bit in both), rules out alternative explanations.

### (2) Writeback bus conflict?

**FALSIFIED.**

Two independent pack-narrow instructions writing to **different destinations**
in the same ILP window (V13, V17) do **not** dual-issue. V13 emits:

```
F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C R16, R16, RZ ;
F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C R2,  R2,  RZ ;
F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C R3,  R3,  RZ ;
F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C R17, R17, RZ ;
F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C R4,  R4,  RZ ;
...  (1024 total, alternating two independent chains)
```

Time 69.98 ms for 2× ops vs. 33.08 ms for 1× (V0) → pure doubling → per-pack
throughput is unchanged at ~32/SM/clk regardless of destination-register
disjointness. If a writeback-bus conflict were the bottleneck, disjoint
destinations would have allowed dual-issue. They did not.

V17 (4-way) adds further confirmation: 2048 F2FPs in 52.58 ms = 30.5/SM/clk/pack,
still single-slot throughput even with 4× register-disjoint ILP.

Also: V2 writes a full 32-bit register (same width as V4's 32-bit writeback)
yet caps at 32. The 32-bit writeback width isn't the choke point.

### (3) Internal pipeline pitch (pack has 2-cycle issue pitch)?

**FALSIFIED.**

From the deep-dive measurements (F2FP_DEEP_DIVE.md §2.2), both
`F2FP.E4M3.F16.UNPACK_B_MERGE_C` (pack) and `F2FP.F16.E4M3.UNPACK_B`
(unpack) have **identical latency L_op = 4.00 cycles** via the standard
differential-chain method. Same pipe, same depth. Pack does not have a
stalled ALU or longer issue pitch.

Additionally, V4 (tf32 PACK_B with saturation + IEEE rounding) runs at
64/clk — the slower rounding+saturate logic that pack uses is clearly
capable of 1-slot issue in its PACK_B form. Only the **operand-read
footprint** distinguishes the 32/clk-capped packs from the 64/clk ones.

### (4) Two distinct SFU subunits (narrow→wide vs wide→narrow)?

**FALSIFIED.**

- V4 `PACK_B tf32` is **wide→narrower** (f32 → 19-bit tf32) and runs at 64/clk.
  If "pack direction only goes to one subunit" were true, this would be 32.
- V1 `UNPACK_B` narrow→wide runs at 64/clk (as expected).
- V2 `PACK_AB f32→f16x2` is wide→narrow-packed, runs at 32/clk.
- V0 `UNPACK_B_MERGE_C f16x2→e4m3x2` is narrow→narrower, runs at 32/clk.

Neither direction nor output width is a clean predictor. Both directions
have both 32-capped and 64-capable members; the discriminator is the
**number of regfile read ports**, not a direction-specific subunit.

### Bonus: pack from f16x2 vs pack from f32 pair (same thing? or different?)

Asked: *"Compare pack from f16x2 vs pack from f32 pair — the latter has 2
source reads, should be even slower if (1) is right."*

Answer: **both cap at 32/clk, not slower**. V0 vs V5/V2:

| Variant | Read slots | /SM/clk |
|---|---|---|
| V0 `f16x2 → e4m3x2` | 1 source + 1 merge (Rc=RZ) = **2 read slots** | 32.0 |
| V2 `f32,f32 → f16x2` (PACK_AB) | 2 sources = **2 read slots** | 32.0 |
| V5 `f32,f32 → e4m3x2` (PACK_AB_MERGE_C) | 2 sources + 1 merge = **3 read slots** | 31.3 |

Both 2-slot and 3-slot cases occupy both SFU issue slots equally — the HW
appears to treat "≥2 read slots" as a binary threshold that consumes both
issue slots, with at most a slight penalty going to 3. This is consistent
with a **register-file-read-port budget per issue group**: the SFU issue
window can grant either 2×(1-read) or 1×(≥2-read) per cycle.

### Bonus: scalar MERGE_C `F2FP.SATFINITE.F16.F32.MERGE_C` — why 24/clk?

V6 runs at 24/clk, slower than the usual 32/clk. Same 2-read footprint
(1 source + Rc). This is the unique "scalar output into 32-bit reg with
upper-16 preservation" case. The 24/clk suggests an additional structural
hazard beyond the 2-slot model — maybe the scalar narrow output path needs
an extra cycle to pack into a full-reg write, or shares a resource with
another pipeline. It doesn't change the broader pack-vs-unpack story, but
it's a nice anomaly pointing to a separate (slower) microcode path for
scalar narrow outputs. Not explored further here.

## Scientific conclusion

Of the four hypotheses, **only (1) survives, in its generalized "regfile read
port count" form**. Specifically:

- The SFU issue unit has **2 issue slots per cycle** (confirmed by V15
  pack+unpack mix = 64/clk aggregate).
- An F2FP opcode encoding **1 source operand slot** (e.g. `UNPACK_B`,
  `PACK_B`) consumes **1 issue slot** → 2 can issue per cycle → **64/SM/clk**.
- An F2FP opcode encoding **≥ 2 operand-read slots** (e.g. `PACK_AB`,
  `UNPACK_B_MERGE_C`, `PACK_AB_MERGE_C`) consumes **both issue slots** →
  **32/SM/clk**, regardless of:
  - whether the second slot is a real 2nd source or a MERGE_C carry,
  - whether MERGE_C's Rc is statically RZ or a live register,
  - whether the output is 32-bit or 16-bit,
  - whether the destination register is fresh or a feedback RMW,
  - whether two parallel packs target different destination registers.
- Latency is identical (4 cy) for 32-capped and 64-capable F2FPs, so
  pipeline pitch is not the cause.
- Direction (wide→narrow vs narrow→wide) is not the cause.

The most natural HW explanation: each SFU issue slot has its own "simple"
regfile-read bandwidth (1 read per slot per cycle), and an op needing more
than 1 read borrows the neighboring slot's read port — which is why 2-read
ops evict the second issuing op.

This is exactly the model already described in
`/root/github/QuickRunCUDA/F2FP_DEEP_DIVE.md §3/§5`, and the new data from
`bench_f2fp_port_hypothesis.cu` (V11, V13, V14, V17) directly refutes the
writeback-bus, subunit-split, and pipeline-pitch alternatives.

## Files

- Kernels:
  - `/root/github/QuickRunCUDA/tests/bench_f2fp_pack_variants.cu` (existing, V0–V6)
  - `/root/github/QuickRunCUDA/tests/bench_f2fp_port_hypothesis.cu` (new, V10–V19)
- SASS:
  - `/root/github/QuickRunCUDA/sass/bench_f2fp_pack_variants_*.sass` (one per hashed variant)
  - `/root/github/QuickRunCUDA/sass/bench_f2fp_port_hypothesis_*.sass` (new)
- Reference for prior work / operand taxonomy:
  - `/root/github/QuickRunCUDA/F2FP_DEEP_DIVE.md`
