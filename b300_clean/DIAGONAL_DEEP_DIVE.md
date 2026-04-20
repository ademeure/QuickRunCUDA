# Diagonal Sign Pattern: Bit-Count-Invariant Cache Hypothesis

**Date: 2026-04-20.** Devil's-advocate re-examination of "diagonal patterns
stay LOW" finding. Discovered the actual mechanism appears to be
**sub-tile popcount (bit count) invariance**, not pattern count.

## The CORRECTION

Prior claim: "Diagonal patterns at p_n=8 stay LOW despite 16 unique sub-tile
patterns - cache must hold more than 2 patterns."

NEW evidence with explicit verification:

### p_n=8 diagonal sweep (sub-tile boundary aligned)
```
shift=0  : 466 W LOW  (no diagonal)
shift=1  : 469 W LOW  
shift=2  : 473 W LOW
shift=4  : 471 W LOW
shift=8  : 474 W LOW
shift=16 : 469 W LOW
shift=32 : 467 W LOW
```

### p_n=16 diagonal sweep (above sub-tile boundary)  
```
shift=1  : 569 W HIGH ← starkly different!
shift=2  : 569 W HIGH
shift=4  : 573 W HIGH
shift=8  : 570 W HIGH
shift=16 : 550 W HIGH (slightly less)
```

**p_n=8 diagonal stays LOW. p_n=16 diagonal goes HIGH.** Same diagonal
mechanism, different power state.

## Devil's-advocate analysis

### Hypothesis 1: "Cache holds many patterns" (REJECTED)

If cache could hold 16 patterns, both p_n=8 and p_n=16 diagonals would be
LOW. They aren't. So cache size doesn't explain it.

### Hypothesis 2: "Sub-tile popcount invariance" (FAVORED)

Compute the popcount (count of - signs) per sub-tile (16 N values) for each
K row in the 16 K-rows of one MMA iteration:

**p_n=8 shift=1:**

| K | Sub-tile bits (16 N values) | popcount |
|---|----------------------------|----------|
| 0 | ++++++++ -------- | 8 |
| 1 | +++++++- ------- + | 8 |
| 2 | ++++++-- ------++ | 8 |
| 3 | +++++--- -----+++ | 8 |
| 4 | ++++---- ----++++ | 8 |
| 5 | +++----- ---+++++ | 8 |
| 6 | ++------ --++++++ | 8 |
| 7 | +------- -+++++++ | 8 |
| 8 | -------- ++++++++ | 8 (inverted) |
| 9 | -------+ +++++++- | 8 |
|10 | ------++ ++++++-- | 8 |
|... | rotations | 8 each |

**ALL 16 K-row sub-tiles have popcount = 8.**

**p_n=16 shift=1:**

| K | Sub-tile bits (16 N values) | popcount |
|---|----------------------------|----------|
| 0 | ++++++++++++++++ | 0 |
| 1 | +++++++++++++++ - | 1 |
| 2 | ++++++++++++++ -- | 2 |
| 3 | +++++++++++++ --- | 3 |
|... | growing - signs | k |
|15 | + --------------- | 15 |

**Each K-row sub-tile has DIFFERENT popcount (0 to 15).**

### The popcount-invariant hypothesis explains:

1. **diag p_n=8 shift=1 LOW** (all popcounts = 8)
2. **diag p_n=16 shift=1 HIGH** (popcounts 0..15 all different)
3. **random_kshift p_n=8 LOW** (all rotations have popcount 8)
4. **random_kshift p_n=16 HIGH** (popcounts 0..15)

### Hypothesis 3: "Pattern hash with rotation invariance"

HW could compute a hash that's invariant to rotation. Both popcount and
rotation-invariant hash would explain the data, but popcount is simpler.

### Hypothesis 4: "Cache works on per-position basis"

Maybe the cache compares (k, n) positions independently. Then for p_n=8
shift=1, position 0 across K-rows sees signs (0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1)
= 1 sign change. Position 8 sees (1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0) =
1 sign change. ALL positions have 1 sign change across K=16 rows.

For p_n=16 shift=1, position 0 sees (0,0,0,0,0,0,0,0,...,0) = 0 changes.
Position 15 sees (0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1) = 1 change.
Position 8 sees (0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1) = 1 change.

Hmm same number of changes per position. But power differs.

Actually for p_n=16 shift=1, each position's K-pattern differs: position 0 is all-zero, position 15 is mostly-one. The cache might have to track 16 different per-position K-patterns.

Hmm getting complicated.

## Methodology critique (devil's advocate)

Things I might have wrong:
1. **Pattern construction**: I'm computing `((n + k*shift) / p_n) & 1`. Devil's advocate: is this actually what the multiplier sees? With BF16 packing 2 per word, my n0/n1 split might miscount positions.
2. **Power measurement noise**: ±5W is typical noise. Some "LOW" values are 466-474W (8W spread). Could "diagonal LOW" be partially noise?
3. **Confounding K-row dedup vs sub-tile dedup**: Both mechanisms might be active. The "LOW" state could be due to either or both. Hard to isolate without controlled tests.
4. **N-direction vs K-direction**: At each N position, the sign varies along K. Both axes have structure. Hard to attribute.
5. **Cache replacement policy unknown**: LRU? FIFO? Pseudo-LRU? Different policies give different behavior.

## CRITICAL conclusion

The "diagonal LOW" finding was REAL but DEPENDS on p_n. Not a universal
"diagonal works" rule. The mechanism appears to depend on whether sub-tile
popcount stays invariant across K-rows.

For real-world weight encoding:
- Patterns where sub-tile bit count is constant across K → LOW power
- Patterns where bit count varies → HIGH power
- Sign-symmetric quantization (equal +/- distribution per sub-tile per K) → likely LOW

Confidence:
- HIGH: p_n=8 diagonal LOW (multiple shifts tested)
- HIGH: p_n=16 diagonal HIGH (multiple shifts tested)
- HIGH: popcount-invariance hypothesis explains the data
- MEDIUM: that popcount is the EXACT mechanism (could be related but not identical)
- LOW: prior "cache holds 8+ patterns" claim — likely wrong; bit-count is the right framework

## Next experiment to definitively prove popcount hypothesis

Construct two patterns:
- Pattern A: 8 + signs and 8 - signs (popcount=8)
- Pattern B: 8 + signs and 8 - signs (different positions, popcount=8)

If A and B alternating gives LOW: popcount theory wins.
If A and B alternating gives HIGH (because positions differ): position matters too.

Then test:
- Pattern C: 7 + signs and 9 - signs (popcount=9)

A,B alternating with C inserted: should be HIGHER if popcount differs trigger.

## What would change conclusions

- Direct test of popcount-invariance (above)
- ncu metrics for sub-tile dedup mechanism (not currently exposed)
- HW circuit reverse engineering
- Larger sub-tile test (e.g., p_n that produces 100% opposite-sign sub-tile = popcount 16)
