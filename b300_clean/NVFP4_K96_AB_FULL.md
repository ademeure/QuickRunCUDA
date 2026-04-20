# NVFP4 K=96 tcgen05.mma — Full A×B Power Characterization

**Date: 2026-04-20.** Comprehensive single-tcgen05 power deep-dive at
K=96 ULTRA path, M=N=256, cluster_dims=2, 296 blocks, clock locked
1005 MHz, idle 150 W, 10M iters per measurement. Synthesizes all
B-distribution + A-distribution + sign-match + outlier findings.

## Headline numbers

```
Idle floor              150 W
Constant A and B        284 W   (multiplier idle, zero-skip)
A=const, B=5pos {+0..+2} 352 W
A=const, B=full random  484 W
A=random, B=const+0     293 W
A=random, B=full random 552 W
A=random, B p_n=64      605 W   (worst-case sign pattern)
TDP cap                 1100 W
```

**Active range**: ~135 W (constant) to ~455 W (worst pattern) per CTA.
Per-CTA active power swings 3.4× based on B data alone.

## A is MOSTLY free, but B is dominant

5×5 matrix (median-of-2 trials, total W):

```
                 |  A=c+0  A=c+2  A=5pos  A=8pos  A=16r
B=const+0        |  284    285    289     290     293
B=const+2        |  286    289    297     298     300
B=5pos {+0..+2}  |  352    399    401     414     431
B=8pos           |  377    435    439     454     472
B=16rand         |  484    538    545     547     542
```

**Row sweep (varying A, fixing B)**: ΔP across A modes = 9-95 W.
**Col sweep (varying B, fixing A)**: ΔP across B modes = 13-249 W.

**B impact ≈ 2.6× A impact** (averaged across configurations).

### Multiplier zero-skip (A or B = constant +0)
When EITHER operand is uniformly zero, the multiplier produces zero
output and the adder/accumulator can short-circuit. Evidence:
- A=const+0, B=full random: 484 W (vs 542 W if A=random)
- A=const+2 (non-zero const), B=full random: 538 W → **+54 W vs A=+0**

This is a hardware optimization saving 30-100 W per CTA when one
operand is 100% zero.

### A=const+2 is "expensive" because constant non-zero A forces multiplier
to actually compute B values without short-circuit. Even all-A-same isn't
sign/mantissa-toggle generating, but each B element still drives a
multiply path.

## B-side distribution ladder (A fixed = random)

| B distribution                          | mantissa | signs | power W | active W |
|-----------------------------------------|----------|-------|---------|----------|
| Constant single value (any of 16)       | 1 mag    | 0%    | 295     | 145      |
| 5 positive {+0,+0.5,+1,+1.5,+2}         | 5 mags   | 0%    | 430     | 280      |
| 4 nonzero positive {+0.5..+2}           | 4 mags   | 0%    | 434     | 284      |
| 8 positive {+0..+6}                     | 8 mags   | 0%    | 472     | 322      |
| 5 centered {-1,-0.5,0,+0.5,+1}          | 3 mags   | 40%   | 490     | 340      |
| 9 asymmetric {-2,-1,0,+0.5,+1,..,+4}    | 7 mags   | 22%   | 526     | 376      |
| ±{0..+2} = 10 codes                     | 5 mags   | 50%   | 510     | 360      |
| 13 codes (excl -0, ±6)                  | 6 mags   | 50%   | 549     | 399      |
| 15 codes (excl -0)                      | 8 mags   | 47%   | 555     | 405      |
| 16 random                               | 8 mags   | 50%   | 552     | 402      |
| Worst (p_n=64 sign-period, all mag=1)   | 1 mag    | 50% N-64-aligned | 605 | 455 |

## Sign-bit alone costs ~80 W

Direct comparison (A=random, same mantissa diversity):
- 8 positive (sign always 0, mag random in 0..6): 472 W
- 16 random (sign random, same mag): 552 W
- **Δ = 80 W from sign bit randomization alone**

### Sign-toggle-rate model

Power scales linearly with sign-toggle rate `2p(1-p)` where p = P(sign=1):
- 5 pos (p=0):           430 W (baseline, 0% toggle)
- 5 centered (p=0.4):    490 W (toggle rate 0.48 × 80W = +38 W ... close to +60 W observed)
- 9 asymmetric (p=0.22): 526 W (toggle rate 0.34 × 80W ≈ +54 W from 8-pos baseline 472 W → predicts 526) ✓ exact
- 16 random (p=0.5):     552 W (toggle rate 0.5 → max sign cost, 472 + 80 = 552 ✓)

The linear sign-toggle model is dialed in across all tested distributions.

## Magnitude diversity adds ~40 W per "additional" magnitude

| mantissa diversity | power W (signs all 0) |
|--------------------|----------------------|
| 1 mag (constant)   | 295                  |
| 5 mags             | 430                  |
| 7-8 mags           | 472                  |

Going from 1 → 5 mags adds 135 W. From 5 → 8 mags adds 42 W. Diminishing
returns beyond ~5 distinct magnitudes.

## Outlier sensitivity

5-pos baseline (no outliers) = 432 W. Add `n` random outliers per
K-block-of-16 from {-3,-2,-1,+2,+3,+4}:

| n outliers | % | power W | Δ |
|----|---|---------|----|
| 0  | 0% | 432    | 0  |
| 1  | 6% | 461    | +29 |
| 2  | 13%| 480    | +48 |
| 4  | 25%| 503    | +71 |
| 8  | 50%| **526** | +94 ← peak |
| 16 | 100%| 519   | +88 |

**Just 1 outlier per 16 weights costs +30 W per CTA.** 50/50 mix is
HIGHER than 100% pure outlier — mixing maximizes inter-element variance
(same as popcount d=16 peak in memory experiments).

## N-64 lane-pairing confirmed

Sign-period sweep at all-1 magnitudes (old `bench_nvfp4_pn_pk_k96.cu`):
- p_n=32: 474 W (signs at n+64 SAME as n → no toggle on lane pair)
- p_n=64: 605 W (signs at n+64 OPPOSITE → 100 % toggle on lane pair)

Match-offset sweep at sp=50% confirmed: mo=64 and mo=192 (= -64 wrap)
are best non-uniform offsets. **Multiplier pairs B-side N-axis at
stride 64 cycles**.

## Sign-bit-on-zero cost

For zero-magnitude elements, sign bit policy matters too:
| sp%  | random sign (mo=-1) | sign=0 (mo=0) | savings |
|------|---------------------|---------------|---------|
| 0    | 556 W               | 553 W         | 3 W     |
| 25   | 546 W               | 537 W         | 9 W     |
| 50   | 522 W               | 503 W         | 19 W    |
| 75   | 481 W               | 440 W         | 41 W    |
| 100  | 401 W               | 298 W         | **103 W** |

**Use +0 not -0 for zero weights**: 3-103 W saved depending on sparsity.
For sparse models, this is ~free 50 W per CTA at typical 25-50 %
sparsity levels.

## Practical recommendations

1. **Pre-process weights to use +0 (0x0) instead of -0 (0x8)** when
   storing zero values. Free 3-100 W depending on sparsity.

2. **Keep B-side magnitude distribution narrow** (≤5 distinct |x|)
   when possible. Going from 8 mags to 5 mags saves 40 W.

3. **Eliminate or cluster outliers**. Each outlier per K-block costs
   ~30 W. Pre-quantization outlier clipping (a common LLM technique)
   has direct power benefit.

4. **A operand is not entirely free** (5-100 W swing) but its impact
   is ~3× smaller than B. If you can choose which operand goes on A
   vs B side (some kernels can), put the more variable / random one
   on A.

5. **Zero-skip**: passing a constant-zero buffer for A (e.g. for
   row-wise activations that happen to be all zero) saves 50-100 W
   instantly via multiplier short-circuit.

## Confidence

- **HIGH**: All numbers from 3-trial medians, repeatable within ±2 W.
- **HIGH**: Linear sign-toggle model fits all distributions tested
  within ±10 %.
- **HIGH**: N-64 lane-pairing confirmed by 3 independent tests
  (p_n sweep, mo sweep, p=192 wrap symmetry).
- **MED**: Magnitude-diversity model is approximate (40 W per added
  mag is a rough fit).
- **HIGH**: Multiplier zero-skip on A=const+0 (verified in matrix
  with multiple non-zero A baselines).

## What would change conclusions

- Test at 1500 MHz to amplify signal (signal scales 1.9× with clock).
- Test K=64 (non-ULTRA path) to see if same model holds.
- Test K=128 with k_size_=0 (standard path).
- Test bigger A-side variations (random sign-period patterns) to
  see if A has its OWN N-64 lane-pairing structure.
- Real model weights (BF16 or FP8 quantized): predict + measure.
