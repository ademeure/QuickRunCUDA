# 2-CTA cluster_group::2 MMA Power Dedup

Date: 2026-04-20. Built `tests/bench_tcgen05_2cta_v2.cu` (fixed from
hanging original — the original had blockIdx.x == 0 leader check which
only ran ONE block out of 148; cluster-relative leader needed).

## Setup
- BF16 m256n256k16 with cluster_dims(2,1,1)
- Each CTA pair issues tcgen05.mma.cta_group::2 from leader (every even blockIdx.x)
- 148 blocks → 74 clusters → 74 active leader CTAs
- a_pat / b_pat: 0=zero, 5=const +1.0, 4=random
- @ -lgc 1005 MHz, 30M iters

## Results

| A pattern | B pattern | Power (W) | Vs single-CTA equivalent |
|-----------|-----------|----------:|-------------------------:|
| const +1.0 | const +1.0 | 287 | matches single-CTA Tier B (~299W) |
| random | random | 597 | matches single-CTA random (~609W) |
| const | random | 546 | matches single-CTA "B random + A const" (549W) |
| random | const | 294 | matches single-CTA "A random + B const" (297W) |

## Conclusion: NO cluster-level cache pooling

2-CTA cluster MMA exhibits **IDENTICAL data-dependent power behavior** to
single-CTA mode (per cluster). The 250W A-vs-B asymmetry holds:
- Random A + const B: ~free overhead (per cluster equivalent)
- Const A + random B: full +250W penalty

This means:
- Each CTA's B operand has its own 32-byte sub-tile dedup cache
- No "cluster-shared" dedup that could amortize structured B across pairs
- Optimization recipes (sort B columns, group K rows) must be applied PER CTA

## Implications for cluster-mode kernels

cuBLAS / CUTLASS often use cluster_group::2 for 2× throughput per pair.
Power optimization requires:
1. Per-CTA B operand structuring (not just cluster-wide)
2. Both CTAs in a pair benefit if B is structured for each
3. Cross-CTA shared B (multicast TMA) doesn't get free dedup — same B-side
   cost applies whether multicast from 1 CTA or loaded by 2 CTAs

## Confidence

- HIGH on the 4 measurements being clean (matches single-CTA pattern within noise)
- HIGH on no cluster-level dedup pooling (asymmetry preserved)
- MEDIUM on whether structured B per CTA achieves expected savings in 2-CTA
  mode (not separately tested with structured patterns)

## Open follow-up
- Test K-rotating + sub-tile patterns in 2-CTA mode to verify save-recipes work
