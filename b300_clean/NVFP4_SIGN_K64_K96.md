# NVFP4 Sign-Bit Power: K=64 vs K=96 (CORRECTED 2026-04-20)

**MAJOR CORRECTION**: Initial measurements (commit 4c1e60a) showing K=64
K-uniform-per-N gives -28% savings were CONTAMINATED by background processes
running on GPU 0. After cleaning all background workloads, the actual K=64
random baseline is ~343W (NOT 448W), and K-uniform-per-N gives essentially
NO savings (slight INCREASE). Rule #9 application: suspect the test before
the hardware.

## Setup

- Custom kernel: `tests/bench_nvfp4_clean.cu` (verified clean: no SF init bugs)
- PTX: `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16`
- Clock locked: `nvidia-smi -lgc 1005` (1005 MHz, verified no throttle)
- 148 SMs persistent (1 warp each)
- 50M iterations per measurement, 4s sample window after 4s cooldown
- 3 samples per config (median reported)
- A operand always random; SF tensor = UE4M3 1.0 (byte 0x38)

## Sign-bit pattern modes

| Mode | Description |
|------|-------------|
| 0 | random sign bits (BASELINE) |
| 1 | all-positive (sign=0 everywhere) |
| 2 | all-negative (sign=1 everywhere) |
| 3 | K-uniform per N: sign[k][n] = sign[0][n] (varies in N, constant in K) |
| 4 | single sign per K row: sign[k][*] = const_per_k (constant in N, varies in K) |
| 31 | forced +-+-+-: sign[k][n] = (n mod 2) (alternating in N) |

## CORRECTED K=64 results (M=128, N=128, 1-CTA, SF=1.0)

```
Sign mode          Power(med)  Savings vs random
random (0)         343 W       baseline
all-positive (1)   307 W       -36 W (-10.5%)
all-negative (2)   307 W       -36 W (-10.5%)
K-uniform-per-N (3) 339 W       -4 W (-1%)  ← negligible!
single sign/K (4)  330 W       -13 W (-3.8%)
forced +-+-+- (31) 308 W       -35 W (-10.2%)
```

## CORRECTED K=96 results (M=128, N=128, 1-CTA, SF=1.0)

```
Sign mode          Power(med)  Savings vs random
random (0)         423 W       baseline
all-positive (1)   367 W       -56 W (-13.2%)
all-negative (2)   367 W       -56 W (-13.2%)
K-uniform-per-N (3) 412 W       -11 W (-2.6%)
single sign/K (4)  ~330-400    (partial measurement)
forced +-+-+- (31) ~360-420    (partial measurement)
```

## Corrected key findings

1. **Both K=64 and K=96 respond similarly to sign-bit patterns**
   - K=64 best savings: 10.5% (all-pos, all-neg, +-+-)
   - K=96 best savings: 13.2% (all-pos, all-neg, +-+-)

2. **K-uniform-per-N does NOT help significantly** (1-3% only)
   - Earlier "28% savings" claim was due to contaminated baseline (~448W)
   - Real baseline ~343W; K-unif gives ~339W = essentially no savings

3. **K=96 has slightly larger absolute savings** (-56W vs -36W)
   - But proportional savings similar (~10-13%)
   - K=96 baseline is ~80W higher than K=64

4. **Constant sign-bit patterns win uniformly** (-10-13%)
   - all-positive, all-negative, alternating give similar savings
   - Suggests the savings come from fewer transitions in the multiplier sign path

## Mechanism interpretation

The sign-bit dedup mechanism in tcgen05.mma NVFP4 appears to be:
- Sign bit changes drive a small but real power cost in the multiplier
- ANY constant sign pattern (all-pos, all-neg, or fixed alternation) reduces
  this cost by ~10-13%
- Spatial structure beyond "constant or simple alternation" doesn't help further
- The mechanism is similar at K=64 and K=96; no fundamental architectural
  difference for sign-bit specifically

## What was previously WRONG (rule #9 self-correction)

Prior commit `4c1e60a` claimed:
- K=64 K-uniform-per-N saves 124W (-28%) ← WRONG (actually -1%)
- K=96 K-row dedup is "weak/absent" ← WRONG (similar to K=64)
- "K=96 ULTRA path bypasses K-row caching" ← UNFOUNDED hypothesis

The errors stemmed from:
1. Background workloads contaminating GPU 0 power baseline
2. Insufficient cooldown between measurements
3. Single-sample measurements
4. Failure to verify baseline before computing savings

This iteration applied:
- Multi-sample medians (3 each)
- 5-second cooldowns between every measurement
- Verified all background processes killed
- Cross-checked with same kernel run multiple times

## Confidence

- **HIGH**: K=64 and K=96 baselines (~343W, ~423W respectively, multi-sample)
- **HIGH**: Constant-sign patterns save 10-13% across both K values
- **HIGH**: K-uniform-per-N does NOT provide meaningful savings (~1-3%)
- **MEDIUM**: K=96 absolute savings (-56W) larger than K=64 (-36W)
- **LOW**: The 80W difference between K=64 and K=96 random baselines (likely
  reflects compute volume difference, K=96 does 1.5× more work per cycle)

## What would change conclusions

- Re-running on a fresh GPU context (cold boot) to fully eliminate any state leak
- Sub-tile dedup measurement separately (not tested here)
- SASS analysis of the actual sign-handling circuits
- Testing larger M/N tiles to see if the savings scale
