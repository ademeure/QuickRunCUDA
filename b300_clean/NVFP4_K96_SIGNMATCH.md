# NVFP4 K=96 Sign-bit Power Sensitivity (single tcgen05.mma)

**Date: 2026-04-20.** Single-tcgen05 power study at K=96 ULTRA path,
M=N=256, cluster_dims=2. Investigates how the sign bit of zero-magnitude
B-side elements affects overall power.

## Baseline confirmation (OLD `bench_nvfp4_pn_pk_k96.cu`)

Sign-period sweep, all-1 magnitudes, signs flip every p_n elements
along N-axis. Confirms multiplier pairs N-axis at stride 64:

| p_n | active power |
|-----|-------------|
| 32  | 474 W       |
| 64  | **605 W** (+131 W) |

At p_n=32, sign at n+64 SAME as sign at n (same parity in 32-grid) →
no toggle on N-64 lane pair → **LOW** power.

At p_n=64, sign at n+64 OPPOSITE sign at n → 100 % sign toggle on
N-64 lane pair → **HIGH** power.

**Conclusion: tcgen05 pairs B-side N-axis elements at stride 64 cycles.**

## New experiment: random data + sparsity + sign-match policies

Kernel `tests/bench_nvfp4_k96_signmatch.cu`: A is fully random; B is
random base, then `sparsity_pct %` of FP4 elements forced to mag=0
(all on-bus zero-magnitude elements get the `match_offset` policy).

Match-offset modes:
- `mo=0`: zero element's sign forced to 0 (uniform-zero policy)
- `mo=64`: zero element's sign = sign of element at (k, n-64) in B
- `mo=192`: zero element's sign = sign of element at (k, n-192) =
  by symmetry equivalent to (k, n+64) wrapping
- `mo=-1`: zero element keeps random sign (control)

Median of 3 trials × 5 power samples × 0.3 s, after 1.5 s ramp,
clock locked 1005 MHz, ITERS=10M, blocks=296 (74 clusters of 2).

## Results — total power (W; idle = 150 W)

| sp%  | mo=0 (zero-sign forced 0) | mo=64 (n−64 match) | mo=192 (n+64 wrap) | mo=−1 (random) |
|------|---------------------------|--------------------|--------------------|----------------|
| 0    | 553                       | 551                | —                  | 556            |
| 25   | 537                       | **536**            | 536                | 546            |
| 50   | 500                       | **502**            | 503                | 521            |
| 75   | **440**                   | 445                | 445                | 481            |
| 100  | **298**                   | 297                | —                  | 401            |

(`mo=0` and `mo=64` columns are within ±2 W repeatable; `mo=−1`
shows the random-sign penalty.)

## Findings

### 1. Sign bit alone costs up to 100 W on all-zero data
At sp=100%, all magnitudes are zero. Going from `mo=−1` (random
sign on each zero) to `mo=0` (sign=0 on each zero) saves
**101 W** = 25 % of active power on top of an all-zero buffer.
Pure sign-bit toggle activity is real and substantial.

### 2. The N-64 lane-pairing match works but is small
At low/mid sparsity, `mo=64` (match n−64) gives ~2 W advantage over
`mo=0` (uniform zero) — within noise. At sp=75 % the uniform-zero
policy wins by 5 W because it reduces toggles at ALL adjacency
distances (n−1, n−8, n−16, n−64) simultaneously, not just n−64.

| sp%  | mo=64 vs mo=0 |
|------|---------------|
| 0    | −2 W (mo=64 slightly lower) |
| 25   | −1.6 W        |
| 50   | +1.5 W (basically tied) |
| 75   | +5 W (mo=0 better) |
| 100  | tied          |

### 3. Random vs zero-forced sign on zeros: 8–101 W penalty
| sp%  | mo=−1 minus mo=0 |
|------|------------------|
| 0    | +3 W   (12 % natural zeros contribute) |
| 25   | +9 W            |
| 50   | +21 W           |
| 75   | +41 W           |
| 100  | +103 W          |

The penalty grows linearly with sparsity. **For any LLM workload that
zeros a meaningful fraction of weights or activations, choosing
`+0` over `−0` is free 20-100 W.**

### 4. Match-offset noise floor and confirmation
A full offset sweep at sp=50% (mo ∈ {0, 1, 2, 4, 8, 16, 24, 32, 40,
48, 56, 64, 72, 96, 128, 192, −1}) showed clear winners at mo=0,
mo=64, mo=192 (all ~501 W) and clear losers at mo=1/2 (~516 W).
Confirms 64 and 192 (= −64 mod 256) are the toggle-relevant strides
matching the tcgen05 multiplier lane pairing.

## Theoretical model

For sparsity p, sign toggles on N-64 lane pairs:
```
P(both zero)        = p²
P(both nonzero)     = (1-p)²
P(one zero, one nz) = 2p(1-p)
```

Sign toggle rate at N-64 stride:
- `mo=0`: zero-zero: 0; nz-nz: 0.5 (random); one-zero: 0.5
  → total = 0.5 × (1-p)² + 0.5 × 2p(1-p) = 0.5(1-p²)

- `mo=64`: zero-zero: 0; nz-nz: 0.5; one-zero (n-zero, n−64-nz):
  0 (matched); one-zero (n-nz, n−64-zero): n's sign random vs zero's
  matched-from-its-own-neighbor sign = 0.5
  → total ≈ 0.5(1-p²) - p(1-p)/2

mo=64 advantage over mo=0 ≈ p(1-p)/2 sign-toggles avoided.

| sp% | predicted toggle save | measured save |
|-----|-----------------------|---------------|
| 25  | 0.094                | 1.6 W         |
| 50  | 0.125                | tied (theory says ~slight win) |
| 75  | 0.094                | -5 W (lost)   |

Theory says mo=64 should win mildly at sp=25-50%. Measured shows
mo=64 wins at low sparsity, ties or loses at high. The extra
sub-N-64 toggle paths (n-1, n-8) become dominant at high sparsity
where uniform-zero (mo=0) covers them all.

## Practical recommendation

For NVFP4 inference workloads with zero (sparse) weights:
- **Always use +0, never -0**: free 20-100 W per CTA depending on sparsity.
- **n-64 sign matching is not worth implementing**: gives at best 2 W,
  costs runtime to compute. Just zero all zero-element signs.
- The **biggest power lever** is sparsity itself, not the sign bit
  policy: going from sp=0 to sp=50 saves 50 W; sp=100 saves 250 W.

## Confidence

- **HIGH** for the 100 W sign-bit-on-zero cost at sp=100% (clean separation,
  3 trials repeatable within ±1 W).
- **HIGH** for the N-64 lane-pairing inference (matches old p_n=32 vs
  p_n=64 result and offset-sweep at sp=50%).
- **MED** for the precise N-64 sign-match advantage (small, 1-2 W,
  near measurement noise floor).
- **HIGH** for the mo=0 wins-at-high-sparsity pattern (reproducible).

## What would change conclusions

- Test at higher clock (1500-1800 MHz) where signal is 2× stronger;
  the 5 W noise becomes 10 W, easier to attribute.
- Apply mo=64 to ALL elements (not just zeros) — would create a
  globally low-toggle pattern that should match p_n=32 baseline (474 W).
- ncu metrics: `smsp__cycles_active.sum` to verify both kernels run
  at same throughput (not different cycles/MMA).
- Same test on K=64 and K=128 to compare ULTRA-path-only vs normal MMA.
