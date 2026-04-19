# Pure tcgen05.mma Power Microbench Results - FP16/BF16/FP8

Date: 2026-04-19. Free-rein continuation.

## Setup

`tests/bench_tcgen05_power.cu` v4: tcgen05.mma kind::f16 OR kind::f8f6f4
with SMEM-resident A and B (loaded once at kernel start, then 100M iters
of MMA referencing the same SMEM addresses). NO DRAM/L2 traffic in the
inner loop — pure multiplier circuit power.

Configuration:
- m=128 n=128 (max single-CTA shape that gives 98.5% MFU)
- K=16 for f16 path, K=32 for f8f6f4 path
- TMEM alloc = 512 cols (m*n fp32 / 32 elem-per-col)
- 148 CTAs persistent, single warp issuer per CTA
- @ -lgc 1005 MHz (no throttle) → ~6.4 sec sustained per test

Pattern semantics:
- 0 = zero (0x00)
- 4 = random (per-index pseudo-random byte fill)
- 5 = +1.0 (precision-specific bit pattern)
- 2 = 0x55 (alternating bit "weird uniform")

## Results: per-tensor isolation @ 98.5% MFU

| Precision | RAND_RAND | zero_zero | A_rand_B_zero | A_zero_B_rand | A=+1 B=rand | A=rand B=+1 |
|-----------|----------:|----------:|--------------:|--------------:|------------:|------------:|
| **FP16** | 642 W | 279 W | 297 (Δ +18) | **552 (Δ +273)** | 581 | 299 |
| **BF16** | 594 W | 284 W | 292 (Δ +8) | **487 (Δ +203)** | 547 | 299 |
| **FP8 e4m3** | 612 W | 289 W | 298 (Δ +9) | **553 (Δ +264)** | 602 | 303 |
| **FP8 e5m2** | 587 W | 289 W | 309 (Δ +20) | **585 (Δ +296)** | 612 | 310 |

## 🎯 KEY FINDING — B is the DOMINANT operand at the multiplier level

For ALL 4 precisions tested, randomizing B alone (with A held at zero) costs
**10-30× more power** than randomizing A alone (with B held at zero).

| Precision | A-only rand cost | B-only rand cost | Ratio |
|-----------|-----------------:|-----------------:|------:|
| FP16 | +18 W | +273 W | **15.2×** |
| BF16 | +8 W | +203 W | **25.4×** |
| FP8 e4m3 | +9 W | +264 W | **29.3×** |
| FP8 e5m2 | +20 W | +296 W | **14.8×** |

## REVERSAL of cuBLAS NVF4 conclusion

Earlier I observed cuBLAS NVF4 had A operand dominant in power. This was
WRONG attribution — the multiplier itself has B dominant. The cuBLAS NVF4
"A dominates" came from TMA multicast on B which saves B's memory pipeline
cost, masking the multiplier's intrinsic B>A.

Updated mental model:

```
Total per-tensor power = (memory pipeline cost) + (multiplier datapath cost)

Memory pipeline cost depends on:
  - Whether the tensor is multicast (multicast halves L2 reads)
  - Tensor size (more bytes = more activity)

Multiplier datapath cost (THIS microbench measures):
  - B >> A intrinsically (~15-30× ratio for f16/f8 paths)
  - Same across FP16, BF16, FP8 e4m3, FP8 e5m2

cuBLAS NVF4 with multicast B: memory cost A > B; multiplier B > A; net A > B observed
cuBLAS BF16 without multicast: memory cost A ≈ B; multiplier B > A; net B > A observed
```

## Uniform patterns are all near-zero baseline

| Pattern | All 4 precisions | Δ vs zero |
|---------|-----------------:|----------:|
| zero (0x00) | 279-289 W | 0 |
| +1.0 | 283-293 W | +3-4 W |
| 0x55 (weird uniform) | 284-294 W | +5 W |

Confirmed: **only randomness costs**, the actual constant value (+1.0, +6.0, etc) is irrelevant.

## TODO: NVFP4 / MXFP4 / MXFP8 (block-scaled formats)

Block-scaled formats use different PTX:
- NVFP4: `kind::mxf4nvf4.block_scale.block16` with TMEM-resident SF operands
- MXFP4: `kind::mxf4.block_scale.block32`
- MXFP8: `kind::f8f6f4` (no separate block-scale instruction; SF applied in software)

NVFP4 PTX is structurally different (SFA/SFB are TMEM addrs, not SMEM
descriptors). Need to:
1. Allocate extra TMEM for SFA, SFB
2. Initialize TMEM scale region with constant scale (1.0 = UE4M3 byte 0x38)
   via tcgen05.cp from SMEM
3. Modify PTX to include [tsfa_addr], [tsfb_addr] operands

Pure NVFP4/MXFP4 multiplier asymmetry could be opposite to FP16/FP8 results
(different instruction, different multiplier circuit). High priority next.

## Confidence

- HIGH on B-dominance for FP16/BF16/FP8 (4 precisions × consistent pattern)
- HIGH on uniform-vs-random distinction (large 200+ W gap)
- HIGH on B/A ratio being intrinsic to multiplier (no DRAM in test)
- MED on extrapolation to NVFP4 (different PTX instruction not tested yet)

---

## NVFP4 Results (kind::mxf4nvf4.block_scale.block16, SF=1.0)

`tests/bench_tcgen05_nvfp4_power.cu` extends the microbench with NVFP4 PTX
(`tcgen05.mma.kind::mxf4nvf4.block_scale.block16`). Initialized the entire
TMEM region with UE4M3 byte 0x38 (=1.0) to ensure all SF reads land on 1.0.

m=128 n=128 results (K_SIZE=0=K64, K_SIZE=1=K96):

| Pattern | K=64 | K=96 |
|---------|-----:|-----:|
| RAND_RAND | 469 W | 421 W |
| zero_zero | 270 W | 246 W |
| +1.0_+1.0 | 272 W | 246 W |
| **A_rand only** | 278 W (Δ +8) | 252 (Δ +6) |
| **B_rand only** | **395 W (Δ +125)** | **364 (Δ +118)** |
| A=+1.0 B=rand | 454 W | 408 W |
| A=rand B=+1.0 | 281 W | 254 W |

**B/A ratio: 15.6× (K=64), 19.7× (K=96)** — same B-dominance as FP16/BF16/FP8.

## Cross-precision summary table

| Precision | A-only rand cost | B-only rand cost | B/A ratio |
|-----------|-----------------:|-----------------:|----------:|
| FP16 | +18 W | +273 W | 15.2× |
| BF16 | +8 W | +203 W | 25.4× |
| FP8 e4m3 | +9 W | +264 W | 29.3× |
| FP8 e5m2 | +20 W | +296 W | 14.8× |
| **NVFP4 K=64** | **+8 W** | **+125 W** | **15.6×** |
| **NVFP4 K=96** | **+6 W** | **+118 W** | **19.7×** |

**Universal finding**: B operand dominates multiplier power by 15-30× across
ALL tested precisions and instruction kinds (kind::f16, kind::f8f6f4,
kind::mxf4nvf4.block_scale).

This DEFINITIVELY proves the cuBLAS NVF4 "A dominates" was an artifact of
TMA multicast on B saving B's memory pipeline cost. The intrinsic multiplier
hardware has B >> A power for all formats.

## M × N shape sweep (NVFP4 K=64)

Only m=128 valid for single-CTA NVFP4. Other M values (64, 256) raise
illegal instruction — likely need cta_group::2 path.

| M=128 N | cy/MMA | TF total | MFU @ NVFP4 7.4 PF |
|--------:|-------:|---------:|-------------------:|
| 64 | 48 | 3249 | 43.8% |
| 128 | 64 | 4874 | 65.7% |
| 256 | 128 | 4874 | 65.7% (cy doubles but FLOPs double too) |

m=128 n=128 is the throughput sweet spot for single-warp issue.

## TODO: experiments that would push NVFP4 beyond 65% MFU

1. **Multi-warp parallel issue** — 4 warps per CTA each issuing to different
   TMEM regions. Should boost MFU 4× since current bottleneck is single-warp
   issue rate.
2. **cta_group::2 (2-CTA MMA)** — m=256 case requires this; might also
   unlock NVFP4 ULTRA (1.5× throughput at K=96)
3. **Pipelined mbarrier batching** — issue a batch of N MMAs without
   waiting between, then wait once. Reduces commit overhead.

These are all PTX-level changes requiring more careful descriptor / cluster
setup. Power asymmetry (B>>A) likely holds at higher MFU but absolute
numbers will scale.

---

## Correction: pure-tcgen05 vs cuBLAS gap NOT yet conclusively explained

Earlier I claimed the cuBLAS NVF4 "A dominates" was "definitively explained
by TMA multicast on B". That overreaches. The pure-tcgen05 result (B>>A in
multiplier across 6 formats) is solid. But the gap to cuBLAS observation
has multiple plausible causes:

1. **cuBLAS may swap A↔B internally**. The user-facing API "A" matrix may
   be fed as the tcgen05 multiplier's B operand (or vice versa). Common
   optimization for matching SMEM layout. Need to check cuBLAS kernel
   layout transformation logic to confirm.
2. **TMA multicast pattern** (the original hypothesis) - B multicast saves
   B's memory cost.
3. **Per-operand SMEM dwell time** - B might be loaded once and re-read
   many times while A is streamed; or vice versa.
4. **Operand pipeline depth** - the cuBLAS kernel might have different
   buffering depths for A vs B feeds, with different per-bit-toggle costs.

To resolve: would need to either (a) trace cuBLAS kernel A/B SMEM addresses
to determine if they're swapped vs API order, or (b) compare with a
custom kernel that explicitly does NOT swap and measure if A dominates.

The pure-tcgen05 microbench result remains robust and useful for predicting
power impact when YOU control the multiplier inputs directly. Whether your
"A" matches cuBLAS's tcgen05 "A" is a separate question.

---

## M×N power scaling (BF16 cta_group::1)

Sweep at -lgc 1005 MHz, BF16, single-warp issuer per CTA, 50M iter sustained.

### Baseline RR (rand) and ZZ (zero) — peak power scaling:

| M | N | RR | ZZ | RR-ZZ Δ |
|--:|--:|---:|---:|--------:|
| 64 | 8 | 186 | 166 | +20 |
| 64 | 32 | 249 | 186 | +63 |
| 64 | 64 | 336 | 213 | +123 |
| 64 | 128 | 404 | 234 | +170 |
| 64 | 256 | 401 | 233 | +168 (sat) |
| 128 | 8 | 213 | 175 | +38 |
| 128 | 32 | 326 | 206 | +120 |
| 128 | 64 | 466 | 246 | +220 |
| 128 | 128 | 602 | 284 | **+318 (peak)** |
| 128 | 256 | 584 | 283 | +301 (sat) |

### A_only vs B_only Δ (vs ZZ baseline):

| M | N | A_only Δ | B_only Δ | B/A ratio |
|--:|--:|---------:|---------:|----------:|
| 64 | 64 | +8 | +79 | 9.9× |
| 64 | 128 | +6 | +112 | 18.7× |
| 64 | 256 | +4 | +112 | 28× |
| 128 | 32 | +13 | +75 | 5.8× |
| 128 | 64 | +12 | +143 | 11.9× |
| **128 | 128** | **+9** | **+205** | **22.8×** |
| 128 | 256 | +4 | +196 | 49× |

## Headline: A power is size-independent; B power scales with REUSE

**A operand contribution stays flat (4-13 W) regardless of M or N.**
**B operand contribution scales with N (= B is reused M times per inst).**

Going m=64→128 (B size unchanged, B-reuse doubles): B delta jumps 112→205 W.
This is a per-cycle data-toggle effect, not a load-count effect.

Mechanism (hypothesis): For each MMA inst, B is read once into the multiplier
operand-B port and used to multiply M different rows of A. As M grows, more
MAC units fire B through, increasing per-cycle switching activity on the
B-side multiplier interconnect.

A is read once and used N times (less for small N). So A's reuse is high
but each "use" is a single multiply (lower per-element activity).

This per-MMA-cycle reuse asymmetry is the underlying cause of the B>>A
power dominance observed in pure-tcgen05 microbenches across all 6 formats.


---

## Cross-precision verification: FP8 e4m3 confirms B-reuse mechanism

FP8 e4m3 M×N sweep at -lgc 1005 MHz, single-warp issuer, 50M iters:

| M, N | ZZ | A_only Δ | B_only Δ | B/A ratio |
|------|---:|---------:|---------:|----------:|
| 64, 64 | 214 | +9 | +106 | 11.8× |
| 64, 128 | 237 | +6 | +151 | 25.2× |
| 128, 64 | 250 | +12 | +178 | 14.8× |
| **128, 128** | 289 | **+11** | **+266** | **24.2×** |
| 128, 256 | 288 | +7 | +255 | 36.4× |

Identical pattern to BF16:
- A operand contribution stays nearly FLAT (6-12W) regardless of size
- B operand contribution scales LINEARLY with M (B reuse rate)
- Both saturate at N>=128 (multiplier issue rate maxed)

Cross-precision absolute B-only at m=128 n=128:
| Precision | B-only Δ | Note |
|-----------|---------:|------|
| BF16 (K=16) | +205 W | |
| FP8 e4m3 (K=32) | +266 W | +30% (2× elements per cycle) |
| NVFP4 (K=64) | +125 W | (single-warp limited at 65% MFU) |

**The B-reuse-drives-power mechanism is universal** across BF16, FP8, NVFP4
when measured at single-CTA tcgen05.mma direct PTX. The exact magnitude
depends on per-cycle element count (precision × K).

This strengthens the multiplier-port-asymmetry conclusion:
- B operand sits on the "multiplicand" port that fires across M MAC units
  per cycle (more parallel switching activity per B byte loaded)
- A operand sits on the "multiplier" port read once per multiplier element
  (less per-byte activity)

This is HARDWARE-ARCHITECTURE level, not format-specific.
