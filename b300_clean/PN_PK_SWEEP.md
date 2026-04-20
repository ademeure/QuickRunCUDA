# BF16 N-period × K-flip-period Sweep at M=N=256

**Date: 2026-04-20.** Pattern `sign[k][n] = ((n / p_n) & 1) XOR ((k / pk) & 1)`.
Tests whether K-direction flipping affects N-direction dedup state.

## p_n=16 with various pk (all HIGH state)

```
p_n  pk    power(W)
16   inf   568   ← baseline (N-period 16 = HIGH due to non-divisor of sub-tile 8)
16   1     563
16   2     568
16   3     571
16   4     555
16   6     553
16   8     557
```

K-flip CAN'T rescue p_n=16. Range only 18W within HIGH state.

## pk=3 with various p_n (key test - K-flip doesn't divide K=16)

```
p_n   pk   power(W)
  1    3   469  ← LOW (alternating N stays LOW even with K-flip)
  2    3   469  ← LOW
  4    3   469  ← LOW
  8    3   469  ← LOW
 16    3   555  ← HIGH (N-period 16 misaligns sub-tile)
 32    3   515  ← partial HIGH
256    3   470  ← LOW (all-same N)
```

## Key insights

### 1. K-flip is essentially FREE

Adding K-direction sign flip every pk rows produces only 2 distinct N-sub-tile
patterns (original and flipped). Both fit in 2-entry cache → no power impact.

### 2. N-direction structure DOMINATES

Power is determined almost entirely by p_n (N-period):
- p_n divides 8 (sub-tile boundary): LOW regardless of pk
- p_n=16 or 32 (misaligns): HIGH regardless of pk

### 3. The 2-entry cache holds {original, flipped} simultaneously

When pattern alternates between A and ~A (bitwise complement) along K, the
cache holds both. Per the prior K-phase finding, only ≤2 patterns fit.
{A, ~A} = 2 patterns, fits perfectly.

### 4. Practical implication

For weight encoding: K-direction sign-flips are "free" power-wise. The
constraint is N-direction sub-tile alignment.

This means INT4 quantization with sign-symmetric ranges (e.g., +/- equal
distributions per row) doesn't add power penalty - only the N-direction
pattern within rows matters.

## Confidence

- **HIGH**: K-flip {original, flipped} fits in 2-entry cache (LOW power preserved)
- **HIGH**: N-direction sub-tile alignment is the dominant factor
- **HIGH**: pk doesn't matter when p_n is in non-aligned state (HIGH)

## NVFP4 K=64 p_n × pk results (M=N=256, 2-CTA)

```
p_n=1 (alternating +-+-) with various K-flip pk:
  pk=inf : 394 W
  pk=1   : 405 W
  pk=2   : 407 W
  pk=3   : 405 W
  pk=4   : 402 W
  pk=8   : 398 W
  pk=16  : 397 W
  pk=32  : 398 W

p_n=16 (sub-tile aligned) with various pk:
  pk=inf : 396 W
  pk=1   : 407 W
  pk=4   : 405 W
  pk=16  : 398 W
  pk=32  : 396 W

pk=3 with various p_n:
  p_n=1  : 408 W LOW
  p_n=2  : 406 W LOW
  p_n=4  : 406 W LOW
  p_n=8  : 409 W LOW
  p_n=16 : 406 W LOW (16 IS sub-tile boundary for NVFP4)
  p_n=64 : 480 W HIGH (chunk-4 sub-tile thrash)
```

## NVFP4 vs BF16 contrast

NVFP4 has MUCH narrower dynamic range than BF16 here because sub-tile=16 elements aligns p_n=16 (whereas BF16 sub-tile=8 means p_n=16 misaligns).

BF16 pk=3 sweep showed p_n=16 HIGH (555W), p_n=8 LOW (469W).
NVFP4 pk=3 sweep shows p_n=16 LOW (406W), p_n=64 HIGH (480W).

The "HIGH transition" happens at p_n = sub-tile size for each precision.

## Universal mechanism summary (all data combined)

Power state determined by sub-tile pattern set size:
1. **≤2 distinct sub-tile patterns** → LOW power (cache fits)
2. **3+ distinct sub-tile patterns** → HIGH power (cache thrashes)

What creates "1 pattern": same p_n that divides sub-tile boundary.
What creates "2 patterns": K-direction sign flip ({original, flipped}).
What creates "3+ patterns": random per-column phase, p_n that doesn't align to sub-tile.

## NVFP4 K=96 ULTRA p_n × pk results

```
K=96 p_n=1 with K-flip pk:
  pk=inf : 467 W (baseline)
  pk=1   : 500 W (+33W ← K-flip costs more at K=96!)
  pk=2   : 486 W
  pk=3   : 486 W
  pk=4   : 478 W
  pk=8   : 474 W
  pk=16  : 473 W
  pk=32  : 471 W
  pk=48  : 471 W

K=96 pk=3 with various p_n:
  p_n=1  : 486 W LOW
  p_n=2  : 487 W LOW
  p_n=4  : 487 W LOW
  p_n=8  : 488 W LOW
  p_n=16 : 488 W LOW
  p_n=64 : 575 W HIGH ← chunk-4 sub-tile
```

### K=64 vs K=96 K-flip impact

```
Format        pk=inf  pk=1   delta
NVFP4 K=64    394     405    +11 W
NVFP4 K=96    467     500    +33 W   ← K=96 ULTRA: K-flip ~3x more costly
BF16 K=16     461     469    +8 W
```

**K=96 ULTRA path has higher K-flip cost** - possibly because the larger
K-pipeline (96 vs 64) accumulates more state transitions per K-flip event.

### Universal pattern across all precisions and K values

For SUSTAINED LOW power across all tested precisions:
- p_n must divide sub-tile boundary (8 for BF16, 16 for NVFP4)
- pk doesn't matter much (K-flip is "near-free" at all K values)

For HIGH power:
- p_n = 2 × sub-tile (i.e., chunk-4 at sub-tile level): WORST case
- p_n that doesn't align: HIGH

The 2-pattern cache holds {original, flipped} reliably across precisions.

## FP8 e4m3 p_n × pk results (M=N=256, K=32, 2-CTA)

```
FP8 p_n=1 with K-flip pk:
  pk=inf : 481 W
  pk=1   : 506 W (+25W ← intermediate between BF16/NVFP4 K=64)
  pk=2   : 495 W
  pk=3   : 493 W
  pk=4   : 490 W
  pk=8   : 488 W
  pk=16  : 486 W

FP8 pk=3 with various p_n:
  p_n=1   : 495 W LOW
  p_n=2   : 496 W LOW
  p_n=4   : 497 W LOW
  p_n=8   : 494 W LOW
  p_n=16  : 485 W LOW (sub-tile boundary)
  p_n=32  : 644 W ← WORST (chunk-4 sub-tile)
  p_n=64  : 572 W HIGH
  p_n=128 : 505 W partial
```

## Cross-precision K-flip cost (pk=1 vs pk=inf)

```
Format       cy/MMA  K-flip cost   Notes
BF16 K=16    128     +8 W          smallest K, smallest flip cost
NVFP4 K=64   128     +11 W
FP8 K=32     128     +25 W         intermediate
NVFP4 K=96   128     +33 W         largest K, largest flip cost
```

Same cy/MMA across all precisions = same per-cycle compute = consistent test.

K-flip cost roughly proportional to K size. K=96 ULTRA pays most for K-flips
because every K-row has more "context" that must be re-evaluated.

## Cross-precision WORST p_n (pk=3)

```
Format       sub-tile  WORST p_n  WORST power  baseline
BF16 K=16    8         16         555 W        469 W (LOW)
NVFP4 K=64   16        64         480 W        408 W (LOW)
FP8 K=32     8         32         644 W        497 W (LOW)
NVFP4 K=96   16        64         575 W        488 W (LOW)
```

WORST p_n = 4 × sub-tile size for FP4/FP8 formats (chunk-4 sub-tile).
For BF16 sub-tile=8, WORST is p_n=16 (chunk-2 sub-tile, different!).

The WORST chunk size at sub-tile level differs between formats:
- BF16: chunk-2 (++--...) at sub-tile level = 16 elements
- NVFP4/FP8: chunk-4 at sub-tile level = 4 × 16 = 64 (for NVFP4) or 4 × 8 = 32 (for FP8)

## BF16 diagonal stripe pattern: sign[k][n] = ((n + k*shift) / p_n) & 1

### p_n=2 with various shifts:
```
shift=0 : 462 W (baseline, no diagonal)
shift=1 : 468 W (small diagonal)
shift=2 : 473 W (mild diagonal)
shift=3 : 469 W
shift=4 : 465 W
shift=8 : 465 W
```

### p_n=8 (sub-tile boundary) with shifts:
```
shift=0 : 466 W
shift=1 : 469 W
shift=2 : 469 W
shift=4 : 471 W
shift=8 : 474 W
```

### Curious: diagonal patterns stay LOW despite many unique sub-tile patterns

For p_n=8 shift=1, each K row has a different shifted version of the
sub-tile pattern. With K=16 and shift=1, theoretically 8-16 distinct
sub-tile contents per K-iteration.

Yet measured power stays LOW (~470W). This contradicts the strict
"≤2 patterns triggers LOW" rule.

Possible explanations:
1. The cache may hold more than 2 patterns transiently (LRU works through them)
2. There's a separate "ramp/shift predictor" that handles linear shifts
3. Pattern matching may use position-relative comparison that diagonals satisfy
4. The mechanism is more complex than the 2-entry cache model

Diagonal patterns are common in some quantization schemes (e.g., shifted
weights for block-wise operations). This finding suggests they may be more
power-friendly than expected.

## Confidence
- **HIGH**: Diagonal patterns stay LOW for shift ≤ 8 at p_n in {2, 8}
- **MEDIUM**: Mechanism behind diagonal-LOW behavior (suggests cache > 2 entries OR special predictor)
- **LOW**: Specific HW circuit responsible

## What would change conclusions

- Test diagonal at higher shifts (32, 64) to see if eventually breaks
- Test diagonal × K-flip combinations
- Test with non-linear K dependencies (e.g., quadratic n+k*k)
- Cross-check with NVFP4 to see if same diagonal-LOW behavior

## Random shift per K row: cache might hold MORE than 2 patterns

Tested `sign[k][n] = ((n + rand_shift[k]) / p_n) & 1` where rand_shift[k] is
random in [0, p_n) per K-row. Each K row gets a different N-pattern.

```
p_n=2  random-shift-per-K : 461 W LOW  (2 possible patterns - fits)
p_n=4  random-shift-per-K : 465 W LOW  (4 possible patterns - still fits!)
p_n=8  random-shift-per-K : 467 W LOW  (8 possible patterns - STILL fits!)
p_n=16 random-shift-per-K : 569 W HIGH (16 patterns - finally breaks)
p_n=32 random-shift-per-K : 530 W partial HIGH
```

### CONTRADICTS the strict "2-entry cache" model

Previous kphase_n test showed pk=4 num_phases=3 → HIGH (3 patterns thrash).
Now we see random-K-shift p_n=8 with 8 unique patterns stays LOW.

### Possible reconciliation

The two tests differ in WHICH AXIS has variation:

1. **kphase_n**: PER-COLUMN K-pattern variation
   - Column 0 has K-pattern A
   - Column 1 has K-pattern B
   - 3+ patterns → HIGH
   - Cache checks "is THIS column's K-pattern same as cached?"

2. **random_kshift**: PER-K-ROW sub-tile pattern variation
   - K-row 0 has N-pattern A
   - K-row 1 has N-pattern B
   - 8 unique patterns → STILL LOW
   - Cache checks "is THIS K-row's sub-tile content cached?"

The K-row dimension may have HIGHER cache capacity than the N-column dimension.

### Hypothesis: per-K-row sub-tile cache holds 8+ patterns

In K=16, the 16 K-rows are evaluated sequentially in one MMA. The cache may
hold up to 8 sub-tile patterns and apply LRU/random replacement. If the
total unique patterns ≤ cache size, no thrashing.

### Power transition point: between 8 and 16 unique sub-tile patterns

```
p_n  unique patterns  power state
 2   2                LOW
 4   4                LOW
 8   8                LOW
16   16               HIGH (transition!)
32   32               partial HIGH
```

Sharp transition at 16 unique patterns suggests cache capacity ≈ 8-16.

### Refined model

```
SUB-TILE CACHE (K-row direction):
- Per sub-tile position (16 BF16 N values)
- Holds ~8 unique sub-tile contents
- Triggered by sub-tile content match
- K-rows in same MMA share cache state
```

This is a major refinement of the prior "2-entry cache" model. The cache
may actually be much larger but operates on different axes with different
characteristics.

## Confidence

- **HIGH**: Random K-shift LOW for p_n ≤ 8 (8 unique patterns OK)
- **HIGH**: Transition at p_n between 8 and 16 (cache capacity reached)
- **MEDIUM**: Cache capacity ~8-16 for sub-tile direction
- **OPEN**: Why kphase_n shows 3 patterns = HIGH but random_kshift shows 8 = LOW
  (different cache mechanism per axis?)

## Next experiments to disambiguate

- Test with EXACTLY N distinct shifts (e.g., 4 shifts from a fixed set, not random)
- Test K-shift frequency per row (1 row vs many rows per shift)
- Test combined: per-column phase (multiple K-patterns) AND per-row shift
