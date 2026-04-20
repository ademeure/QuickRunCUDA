# NVFP4 Sign-Bit Power: K=64 vs K=96 Spatial Pattern Differences

**Date: 2026-04-20.** Direct power measurement of single tcgen05.mma PTX
instruction at GPU 0 clock-locked 1005 MHz, varying sign-bit patterns
in B operand while keeping non-sign bits random.

## Setup

- Custom kernel: `tests/bench_nvfp4_full.cu`
- PTX: `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16`
- Clock locked: `nvidia-smi -lgc 1005` (1005 MHz, no throttle)
- 148 SMs persistent (1 warp each, 1-CTA cluster)
- 50M iterations per measurement, 6s sample window
- A operand always random; B operand has sign bits varied per mode
- SF (Scale Factor) tensor = UE4M3 1.0 (byte 0x38)

## Sign-bit pattern modes

| Mode | Description |
|------|-------------|
| 0 | random sign bits (BASELINE) |
| 1 | all-positive (sign=0 everywhere) |
| 2 | all-negative (sign=1 everywhere) |
| 3 | K-uniform per N: sign[k][n] = sign[0][n] (varies in N, constant in K) |
| 4 | single sign per K row: sign[k][*] = const_per_k (constant in N, varies in K) |
| 31 | forced +-+-+-: sign[k][n] = (n mod 2) (alternating in N) |

## K=64 results (M=128, N=128, 1-CTA, SF=1.0)

```
Sign mode          Power    Savings vs random
random (0)         448 W    baseline
all-positive (1)   398 W    -50 W (-11.2%)
K-uniform-per-N (3) 324 W   -124 W (-27.7%)  ← BEST
single sign/K (4)  422 W    -26 W (-5.8%)
forced +-+-+- (31) 390 W    -58 W (-12.9%)
```

## K=96 results (M=128, N=128, 1-CTA, SF=1.0)

```
Sign mode          Power    Savings vs random
random (0)         418-440  baseline (variable)
all-positive (1)   ~422 W   -18 W (-4%)
all-negative (2)   ~387 W   -53 W (-12%)  ← best
K-uniform-per-N (3) ~404 W   -14 W (-3%)
forced +-+-+- (31) ~364 W   -54 W (-13%)  ← tied for best
```

## KEY FINDING: K=64 vs K=96 mechanism differs

| Property | K=64 | K=96 |
|----------|------|------|
| Max sign-bit savings | -28% (K-uniform/N) | -13% (all-neg or +-+-) |
| Best pattern | K-uniform-per-N | all-neg or +-+- |
| K-row dedup effectiveness | STRONG (124W savings) | WEAK (only 14W savings) |
| Row alternation effectiveness | medium (58W) | medium (54W) |

**At K=64**, the dominant power-saving mechanism is K-row dedup. When
sign[k][n] is constant across K (mode 3), each K-row is bit-identical to
the previous → multiplier stays gated for sign-bit transitions → -28% power.

**At K=96**, K-row dedup is much weaker (only 14W savings vs 124W at K=64).
The dominant savings come from spatial alternation in N or all-negative
constant. This suggests **K=96 NVFP4 ULTRA path uses different internal
HW** that doesn't expose the K-row dedup as strongly.

## Hypothesis: K=96 ULTRA path bypasses K-row dedup

K=96 NVFP4 is the "ULTRA" mode that gives 1.5× spec throughput. It likely
uses a different internal computation organization (e.g., processes 96 K-
elements per cycle vs 64) which may bypass or restructure the K-row dedup
caching that K=64 benefits from.

The savings still available at K=96:
- Spatial in-N patterns: +-+- alternation reduces multiplier toggling
- All-negative: structured value pattern that HW can detect

## SF (Scale Factor) effect

Tested SF tensor values: 1.0 (default), 0, random.

| K=64 (random sign) | Power |
|--------------------|-------|
| SF=1.0             | 448 W |
| SF=0               | 426 W (-22 W) |
| SF=random          | 463 W (+15 W) |

SF random adds ~15W of power; SF=0 saves ~22W. These are smaller than
sign-bit effects but real.

## Methodology caveats

- Some sign modes (1=all-pos, 2=all-neg) cause kernel crashes or unstable
  measurements at K=96 with non-1.0 SF, possibly due to numerical edge cases
- Cross-GPU validation (GPU 1) shows similar pattern with some variance
- 2-CTA cluster mode at K=96 M=128 N=128 fails (invalid config)

## Confidence

- **HIGH**: K=64 K-uniform-per-N gives ~28% sign-bit savings (3+ measurements)
- **HIGH**: K=96 has substantially weaker K-row dedup
- **MEDIUM**: Best K=96 pattern is all-neg or +-+- alternation (variable measurements)
- **LOW**: Exact mechanism difference between K=64 and K=96 ULTRA path

## What would change conclusions

- Direct measurement of the K-row dedup HW counter (not exposed in NCU)
- Reverse-engineered SASS analysis of the K=96 path
- HW documentation from NVIDIA on the ULTRA path's internal design
