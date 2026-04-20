# tcgen05.mma Power: MASTER SUMMARY (B300)

Date: 2026-04-20. Comprehensive investigation across BF16/FP8/NVFP4 with
1000+ data points. Provenance commits in this branch (`f2fp-deep-dive`).

## TL;DR

The B300 tcgen05.mma multiplier consumes power based on three independent
mechanisms:

1. **Static baseline** (~280-305 W per precision)
2. **K-row toggle cost** (per-K-row content transitions, ~5W/row BF16)
3. **N sub-tile cliff** (within-row B operand pattern diversity)

A operand is essentially FREE regardless of randomness. B operand drives
all data-dependent power. The 32-byte sub-tile is the universal HW
granularity for B-side dedup.

Worst case (random data): 609 W BF16 / 642 W FP8 / 463 W NVFP4.
Best case (structured): ~baseline (300 W) achievable even with full bit entropy.

## The Mechanism

### B operand path (POWER HOT PATH)

```
Per K row (16 in BF16, 32 in FP8, 64 in NVFP4):
  B is loaded as N=128 values via SMEM descriptor
  Sub-tile = 32-byte chunk of B operand (16 BF16 / 32 FP8 / 64 NVFP4 N values)
  
  WITHIN K row:
    HW maintains 1-slot dedup cache for sub-tile pattern
    Sticky activation: cache fills on first sub-tile, stays loaded
    If subsequent sub-tile bit-identical to cached: GATED (free)
    If different: ACTIVATED (full power)
    Once activated, stays active for rest of N sweep within row
  
  ACROSS K rows:
    Cache compares CURRENT row's content vs PREVIOUS row's content
    If row content identical: dedup, ~free
    If different: activation cost (~5W/row BF16, scales 0.025 W/byte)
```

### A operand path (BROADCAST, FREE)

```
A is broadcast across all N MACs at given (m, K)
Single value driven through fanout buffer per cycle
NO sub-tile dedup mechanism observed
Random A values (per (m,k)) cost essentially 0W extra vs constant A
```

## The Universal 32-Byte Boundary

| Precision | Sub-tile size | N values per sub-tile | Cliff at N_unique |
|-----------|--------------:|----------------------:|------------------:|
| BF16      |       32 bytes |                    16 |             16→17 |
| FP8 e4m3  |       32 bytes |                    32 |             32→33 |
| NVFP4     |       32 bytes |                    64 |             64→65 |

When within-row N pattern fits in 32 bytes (≤16/32/64 unique values per
precision): all 8 sub-tiles match → 1 cache fill, 7 dedups → FREE.

When over the boundary: cache thrashes, all sub-tiles activate → +250-310W.

## Sticky Activation Position Effect

For BF16 with 4 unique sub-tiles + 4 shared (mode 3000-3007):

| Pattern position | Power (W) |
|------------------|----------:|
| Shared first, unique last (clustered) | 342 |
| Unique first, shared last (mirror)    | 611 |
| Alternating (010101...)               | 614 |

POSITION MATTERS hugely. Putting same-pattern sub-tiles at LOW N indices
saves ~270W vs reverse order.

## A-vs-B Asymmetry (250W gap)

| A | B | Power (W) |
|---|---|----------:|
| const | const | 299 |
| FULL random (m,k) | const | 297-302 |
| const | random | 549 |
| random | random | 609 |

A randomness contributes ~0W. B randomness contributes ~250W. Combined
adds ~60W from saturation.

## Cross-Precision Random Baselines

| Precision | Random (W) | Const (W) | Gap |
|-----------|-----------:|----------:|----:|
| BF16      | 609 | 299 | +310 |
| FP8       | 642 | 305 | +337 |
| NVFP4     | 463 | 280 | +183 |

NVFP4 has SMALLEST random gap because:
- 4-bit values have lower mantissa popcount
- Less multiplier transistor activity per multiply

## Software Optimization Recipes

### For inference (decoder GEMMs)
- **Activations** (high-entropy, request-varying) → A operand
- **Weights** (static, can be quantized) → B operand
- Quantize weights to fit ≤16 unique values per N-16-group (BF16 case)
- Sort weight columns by similarity for K-row dedup

### For training (gradient GEMMs)
- Both operands typically vary → near full power expected
- Consider data-aware tile reordering: cluster low-variance regions

### For attention
- Q × K^T: usually both vary fully (full power)
- Attention × V: V has structure, opportunity for dedup

### Universal advice
- **B operand is the optimization target**. A is free regardless.
- **Pre-sort B columns** for sticky-activation dedup (~250W save per CTA).
- **Group consecutive K rows by content** (~85W save per CTA, BF16).

## Summary Table: Power Breakdown by Configuration

| Configuration | BF16 (W) | FP8 (W) | NVFP4 (W) |
|---------------|---------:|--------:|----------:|
| All zero (Tier A) | 294 | 300 | 280 |
| Const non-zero (Tier B) | 299 | 305 | 280 |
| Inf/NaN (Tier C) | 308 | -   | -   |
| K-vary alone (low ent) | 345 | 376 | 379 |
| K-vary alone (high ent) | 387 | ~440 | ~470 |
| N-vary fits cache (≤32B) | 302 | 309 | 285 |
| N-vary misses cache | 595-621 | 595-657 | 457-463 |
| Position-optimized (4 same first) | 342 | -   | -   |
| Random (worst case) | 609 | 642 | 463 |
| A random + B const | 302 | -   | -   |
| A const + B random | 549 | -   | -   |

## Reference Tests

All tests at -lgc 1005 MHz, 50M iters, 148 SMs persistent, GPU 0.
50M × ~64 cy/MMA ≈ 3.2 sec sustained → reliable plateau power readings.

Test kernels:
- `tests/bench_tcgen05_bf16_perbit_power.cu` — BF16 modes 0-5232
- `tests/bench_tcgen05_fp8_kvary_power.cu` — FP8 modes 0-5232
- `tests/bench_tcgen05_nvfp4_kvary_power.cu` — NVFP4 modes 0-5264

Provenance documents:
- `b300_clean/BF16_PERBIT_POWER.md` — per-bit decomposition (sign 18% etc.)
- `b300_clean/BF16_SUBTILE_DEDUP.md` — first sub-tile dedup discovery
- `b300_clean/CROSS_PRECISION_SUBTILE.md` — 32-byte boundary universal
- `b300_clean/SUBTILE_DEDUP_MODEL.md` — pattern-count anomaly resolution
- `b300_clean/A_VS_B_ASYMMETRY.md` — A is free regardless of randomness
- `b300_clean/POWER_FINAL_MODEL.md` — K-row pairwise + per-byte 0.025 W/byte

## Confidence

- HIGH on universal 32-byte sub-tile (3 precisions, 3 hash functions, multiple modes)
- HIGH on A vs B asymmetry (5+ measurements consistent)
- HIGH on K-row pairwise dedup (cross-precision consecutive grouping data)
- HIGH on sticky activation (position invariance test)
- HIGH on per-byte cost ~0.025 W/byte (cross-precision agreement)
- HIGH on per-CTA dedup (NOT cluster-shared: 2-CTA kernel test confirms)
- HIGH on SMEM descriptor invariance (LBO=16 vs LBO=32 same power)
- HIGH on BF16-specific two-half processing (FP8/NVFP4 don't show it)
- MEDIUM on multiplicative saturation between K and N components
- MEDIUM on accumulator entropy cost (~5% of total)
- LOW on whether this transfers to NVIDIA's high-level libraries verbatim
  (but cuBLAS RANDOM vs CONSTANT gap matches predictions per I4 finding)

## Additional findings since v1

| Finding | File | Headline |
|---------|------|----------|
| 2-CTA dedup | 2CTA_DEDUP.md | Per-CTA cache, NO cluster pooling |
| MMA shape | MMA_SHAPE_DEDUP.md | 32-byte cliff universal; "4-slot free zone" is N=128-specific |
| Two halves | SUBTILE_HALVES.md | BF16 m128n128 has 2 halves; pos 4-7 unique nearly free; FP8/NVFP4 don't show this |
| SMEM desc | MMA_SHAPE_DEDUP.md (appendix) | LBO doesn't affect dedup (intrinsic to HW) |
| Accumulator | POWER_FINAL_MODEL.md (appendix) | C accumulator costs ~5% of total power |
| disable_lane | DISABLE_LANE_POWER.md | Linear power scaling ~2.4W/bit, position-independent |
| disable_lane composition | DISABLE_LANE_POWER.md (appendix) | Combined with sub-tile dedup → 60% reduction |
| Combined extremes | DISABLE_LANE_POWER.md (appendix) | 254W (vs 610W random) = 58% reduction achievable |
| mma.sync legacy | MMA_SYNC_POWER.md | A vs B asymmetry exists in legacy mma.sync (17W gap) |
| NVFP4 SF | NVFP4_SF_POWER.md | Scale factor adds independent ~27W when random |
| Sparse validation | SUBTILE_SPARSE_VALIDATION.md | Per-byte sparsity ineffective; sub-tile-level confirms two-half |
| K-direction linear | K_DIRECTION_LINEAR.md | NO K-halves; uniform ~19.5W per random K row |
| A vs B zero asym | A_B_ZERO_ASYMMETRY.md | A=0 saves 121W, B=0 saves 313W (full gating) |
| Latency data-indep | LATENCY_DATA_INDEPENDENT.md | per-MMA timing IDENTICAL across patterns - dedup is power-only |
| **PRACTICAL: 18% gain** | PRACTICAL_THROUGHPUT_GAIN.md | Structured B at boost = 4.06s vs random 4.78s (avoids 1590MHz throttle) |
| **Power cap sweep: 2.09× at 400W** | PRACTICAL_THROUGHPUT_GAIN.md (appendix) | Optimization speedup grows as cap tightens: 1.18x@1100W → 2.09x@400W |
| Per-SM scaling | PER_SM_POWER_SCALING.md | 3.1W per SM random, 1.0W per SM const, linear up to 148 |
| Power-frequency curve | POWER_FREQUENCY_CURVE.md | Peak 2.36× ratio at 1500 MHz; best TF/W at 1300-1500 MHz |
| M dimension halves | M_DIMENSION_HALVES.md | BF16 halves preserved at M=64 (N-direction structural) |
| Cross-MMA dedup | CROSS_MMA_DEDUP.md | Per-MMA only; alt MMAs avg the powers; partial opt gives 12% gain |
| Power floor | POWER_FLOOR.md | 287W absolute minimum for 148 SMs of active tcgen05 |
| **cuBLAS real GEMM** | CUBLAS_REAL_VALIDATION.md | **1.49× speedup in cuBLAS BF16 8192³ (2139 vs 1432 TFLOPS)** |
| Sub-tile partial break | SUBTILE_PARTIAL_BREAK.md | Linear ~19W per broken byte; first byte +32W activation |

---

## Boost-clock validation (2032 MHz)

Same kernel, same modes, just at -rgc (boost) instead of -lgc 1005:

| Mode | Description | 1005 MHz (W) | Boost 2032 MHz (W) | Δ |
|-----:|-------------|-------------:|-------------------:|---:|
| 200 | Random | 609 | 1097 | +488 |
| 300 | All zero | 294 | 621 | +327 |
| 1400 | B const +1.0 | 299 | 643 | +344 |
| 2900 | All-shared sub-tiles | 301 | 645 | +344 |
| 2904 | 4-unique sub-tiles | 342 | 779 | +437 |
| 2908 | All-unique sub-tiles | 601 | 1098 | +497 |

**Dedup savings scale UP at boost:**
- 1005 MHz: K_break=0 vs K_break=8 = 300 W gap
- Boost:    K_break=0 vs K_break=8 = 453 W gap

At boost, sub-tile dedup optimization saves ~450 W per CTA. With 148 SMs
all running tcgen05.mma, total potential savings = 148 × 450/X (where X
is CTA-per-SM concentration).

Practical implication: a structured-data BF16 GEMM workload could run at
~250W lower power than equivalent random-data workload on the SAME hardware
at the same throughput. Crucial for power-constrained inference.
