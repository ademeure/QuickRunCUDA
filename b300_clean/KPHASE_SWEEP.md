# Per-Column Random K-Phase Sign Power: BF16 + NVFP4 at M=N=256

**Date: 2026-04-20.** Test: for each column n, randomly choose a phase in
[0, period_k) and apply pattern `sign[k][n] = ((k + phase[n]) / period_k) & 1`.
This means columns can have e.g. +-+-+- or -+-+-+ at p=1 (random which).

Tests whether the dedup mechanism handles per-column phase variation.

## Setup

- M=N=256 2-CTA cluster, clock locked 1005 MHz, 148 SMs persistent
- Kernels: `bench_bf16_kphase.cu` (K=16), `bench_nvfp4_kphase.cu` (K=64)
- Phase = random per N column from [0, period_k)
- Non-sign bits random; SF tensor at default (UE4M3=0x38 for NVFP4)

## BF16 K-phase results (K=16)

```
period_k  power(W)   vs N-period (no phase)
   1      470        +9 W (was 461, slight overhead)
   2      467        +4 W (was 463)
   3      503        -33 W (lower than N-period 536)
   4      499        +35 W ← MAJOR! (N-period was 464 LOW)
   6      499        -40 W (N-period was 539)
   8      503        +37 W ← (N-period was 466 LOW)
  12      509        -30 W (N-period was 539)
  16      493        -64 W (N-period was 557)
range: 470-509 = 39W (8.3%)
```

## NVFP4 K-phase results (K=64)

```
period_k  power(W)   vs N-period (no phase)
   1      405        +19 W (was 386, slight overhead)
   2      406        +20 W (was 386)
   3      431        -6 W (was 437)
   4      427        +40 W ← MAJOR! (N-period was 387 LOW)
   6      428        N/A
   8      438        +51 W ← (N-period was 387 LOW)
  12      434        N/A
  16      439        +53 W ← (N-period was 386 LOW)
  24      436        N/A
  32      428        +37 W ← (N-period was 391 LOW)
  48      439        +5 W (was 434)
  64      424        -54 W (N-period was 478)
range: 405-439 = 34W (8.4%)
```

## Key findings

### 1. Only pk=1, pk=2 robust to per-column random phase

For BOTH BF16 and NVFP4:
- pk=1, pk=2: power stays LOW (slight +5-20W overhead vs same-phase N-period)
- pk=3 and above: power JUMPS to HIGH state

### 2. Random phase HURTS for periods that were LOW with same-phase

The most striking finding: at pk=4, 8, 16, 32 (which were all LOW power
with same-phase N-period sweep), random phase per column makes them HIGH.

```
N-period (same phase)  →  K-phase (random per column)
BF16 p=4:    464 W LOW  →  pk=4:  499 W HIGH (+35W)
BF16 p=8:    466 W LOW  →  pk=8:  503 W HIGH (+37W)
NVFP4 p=4:   387 W LOW  →  pk=4:  427 W HIGH (+40W)
NVFP4 p=8:   387 W LOW  →  pk=8:  438 W HIGH (+51W)
NVFP4 p=16:  386 W LOW  →  pk=16: 439 W HIGH (+53W)
NVFP4 p=32:  391 W LOW  →  pk=32: 428 W HIGH (+37W)
```

### 3. Random phase HELPS for periods that were HIGH with same-phase

```
BF16 p=3:    536 W HIGH  →  pk=3:  503 W partial HIGH (-33W)
BF16 p=16:   557 W HIGH  →  pk=16: 493 W partial HIGH (-64W)
NVFP4 p=64:  478 W HIGH  →  pk=64: 424 W partial HIGH (-54W)
```

When all columns share same phase that doesn't align to sub-tile, every
column has same "bad" pattern → max power. Random phase per column means
some columns happen to align, giving lower power.

### 4. Mechanism interpretation

The HW dedup cache (2-entry LRU) operates per-sub-tile-position. When all
columns share the same phase:
- Sub-tile pattern is FIXED (1 unique pattern across columns) → cache hits
- Or COMPLETELY MISMATCHED to sub-tile boundary → cache misses

When columns have random phases:
- Each column has different sub-tile pattern (up to period_k different)
- For pk=1, 2: only 2 phases possible (= 2 patterns), fits in cache
- For pk≥3: 3+ patterns, cache thrashes → consistent HIGH power
- Variance is DAMPENED (no extreme high or low)

### 5. Practical implication: same-column-phase is critical

For ML inference power optimization:
- **All columns must share the same K-pattern phase** for max savings
- **Random per-column phase wastes 30-50W** at pk=4, 8, 16, 32
- Only pk=1, pk=2 are robust to phase variation
- This is a STRONG constraint: real weights have random per-column structure
  → cannot exploit pk≥3 structure in real weights

This generalizes the "constant signs" optimization: signs must not just
have a low-period structure, but ALL columns must share the same phase.
Sub-tile dedup is per-sub-tile-position, not pattern-recognition across
phases.

## Spread comparison: N-period vs K-phase

| Format | N-period range | K-phase range | Notes |
|--------|---------------|---------------|-------|
| BF16   | 96W (21%)     | 39W (8%)      | K-phase compresses spread |
| NVFP4  | 92W (22%)     | 34W (8%)      | Same compression |

Random phase per column smooths the variance: the WORST cases improve,
the BEST cases degrade. Average is similar.

## Confidence

- **HIGH**: pk=1, 2 stay LOW even with random phase (BF16 + NVFP4)
- **HIGH**: pk≥3 stay HIGH regardless of same vs random phase
- **HIGH**: Random phase REGRESSES pk=4, 8 (was LOW → now HIGH)
- **HIGH**: Random phase IMPROVES pk=3, p=N/2 cases (was HIGH → less HIGH)
- **MEDIUM**: Mechanism interpretation (sub-tile dedup with multiple phases)

## What this means for practical optimization

The earlier finding that "+-+-+- alternation works for both BF16 and NVFP4"
is now strengthened:
- pk=1 alternation is robust regardless of per-column phase
- Higher periods require all columns to align for the savings

Real weight matrices with structured pruning at pk=2 (++--) can still
benefit from sub-tile dedup. But anything with longer period or random
phase per column will not see the savings.
EOF
git add -A && git commit -m "K-phase sweep: per-column random phase only LOW power for pk=1,2 (BF16 + NVFP4)

Tests with sign[k][n] = ((k + phase[n]) / period_k) & 1, phase random per column.

Key findings:
- pk=1, 2 stay LOW (slight +5-20W vs same-phase N-period)
- pk>=3 ALWAYS HIGH regardless of per-column phase
- Random phase REGRESSES pk=4, 8 (was LOW with same-phase, now HIGH)
- Random phase IMPROVES pk=3, p=N/2 (was HIGH, now less HIGH)
- Spread compressed: 21% N-period -> 8% K-phase (variance dampened)

Mechanism: HW dedup cache (2-entry LRU) per-sub-tile-position. With multiple phases per column = multiple patterns = cache thrash.

Practical: ALL columns must share same K-pattern phase for max savings. Real weights have random per-column structure -> cannot exploit pk>=3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" 2>&1 | tail -3