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
