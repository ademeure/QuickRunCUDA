# tcgen05.mma Power Guide for B300 (Practitioner's Reference)

Date: 2026-04-20. Comprehensive practical reference distilled from 30+
detailed power experiments on B300 SXM6.

## Quick Reference: Power Range

| Configuration | Power (W) | Use Case |
|---------------|----------:|----------|
| Idle GPU | 150 | Not running anything |
| Active floor (A=0, B=0, 148 SMs) | 287 | Theoretical minimum for tcgen05 active |
| Random data GEMM (BF16) | 611 | Worst case (typical training gradient) |
| Optimized GEMM (BF16) | 287-400 | Sorted weights + sub-tile-friendly |
| Power cap (TDP) | 1100 | Maximum allowed sustained |

All measurements at 1005 MHz fixed clock. At boost (2032 MHz), values scale ~1.7-2x.

## The 3 Power Components (Universal Model)

```
P = P_baseline + P_K_toggle + P_N_subtile_cliff
```

### P_baseline (~287W for 148 SMs)
- Static + minimum multiplier overhead
- Per-SM: ~1W
- Cannot be reduced

### P_K_toggle (0-110W, scales with K transitions)
- Per-K-row content transitions
- ~5W per BF16 K-row transition (15 max → 84W max)
- ~3.5W per FP8 K-row transition (31 max → 109W)
- ~1.7W per NVFP4 K-row transition (63 max → 110W)
- **Total K-vary cost: ~110W universal across precisions**

### P_N_subtile_cliff (0-310W, sharp cliff)
- Triggered when within-K-row N pattern exceeds 32-byte cache
- Cliff at: 17 unique BF16 / 33 unique FP8 / 65 unique NVFP4 values per K row
- Below cliff: free
- Above cliff: full activation (~250-310W)

## Top 5 Optimization Recipes

### 1. Sort B columns to cluster identical sub-tiles at LOW N
- Saves up to 270W per CTA (from sticky activation in BF16)
- Works because HW activates on first non-matching sub-tile, stays active

### 2. Quantize B to fit ≤ cache size per K-row N stripe
- BF16: ≤16 unique values per 16 N positions (sub-tile cache hit)
- FP8: ≤32 unique values per 32 N
- NVFP4: ≤64 unique values per 64 N
- Saves the entire 250-310W N-cliff cost

### 3. Group consecutive K rows by content similarity
- Pre-sort K dimension so adjacent K rows have matching values where possible
- Saves ~5W per BF16 K-row transition (84W max)

### 4. Use disable_lane for sparse attention
- Each disabled N column saves ~2.4W (BF16 random)
- Linear scaling, position-independent

### 5. (BF16 m128n128k16 SPECIFIC): Put unique sub-tiles in HIGH N (Half B)
- BF16 m128n128 has TWO-HALF processing in N direction
- Half B (sub-tiles 4-7, N=64..127) is essentially FREE for unique patterns
- Doesn't apply to FP8/NVFP4 or other M/N shapes

## Practical Performance Gains

| Power cap | BF16 speedup | FP8 speedup | NVFP4 speedup |
|----------:|-------------:|------------:|--------------:|
| 1100W (default) | 1.18× | 1.18× | 1.04× |
| 800W | 1.40× | est 1.42× | est 1.20× |
| 600W | 1.74× | 1.78× | 1.41× |
| 400W | 2.09× | 2.14× | 1.62× |

## Key Insights

1. **Performance is data-INDEPENDENT** - per-MMA cycle count identical
   regardless of data pattern. Optimization is FREE - never slows kernel.

2. **A operand is FREE** - random A only +5W vs constant A (when B const).
   Put high-entropy operand on A side; structured operand on B.

3. **B operand drives all power** - 250-310W difference between random B
   and constant B per CTA.

4. **Sub-tile dedup is universal at 32 BYTES** - across BF16/FP8/NVFP4.

5. **Dedup is multiplier-fundamental** - present in both legacy mma.sync
   AND modern tcgen05; mechanism is intrinsic to BF16/FP8 multiplier HW.

6. **Per-CTA only** - 2-CTA cluster_group::2 doesn't share dedup cache.

7. **No NCU counter** - dedup is transistor-level clock gating, invisible
   to instruction/cycle counters.

## Software Implementation Hints

### For inference deployment

```python
def prepare_b_for_tcgen05(weights):
    """Optimize weight tensor for B300 tcgen05 power efficiency."""
    # 1. Quantize to ~16 unique values per 16-column group (BF16)
    weights = quantize_per_group(weights, group_size=16, num_levels=16)
    # 2. Sort columns within each 16-N group by similarity
    weights = sort_columns_by_pattern(weights, tile=16)
    # 3. Sort K rows by content similarity
    weights = sort_rows_by_content(weights)
    return weights
```

### For training (where both ops vary)

- Limited optimization potential for the matmul itself
- Consider mixed-precision: use NVFP4 for grad reduction (less power-sensitive)
- Group inputs by similarity (semantic batching)

### For attention (sparse)

- Use disable_lane mask for known-zero attention positions
- Saves ~2.4W per disabled column linearly

## File index

- `TCGEN05_POWER_MASTER.md` - master findings table
- `BF16_SUBTILE_DEDUP.md` - first sub-tile discovery
- `CROSS_PRECISION_SUBTILE.md` - 32-byte boundary universal
- `SUBTILE_DEDUP_MODEL.md` - mechanism analysis
- `A_VS_B_ASYMMETRY.md` - A is broadcast, B is distributed
- `POWER_FINAL_MODEL.md` - K-row pairwise dedup
- `MMA_SHAPE_DEDUP.md` - shape effects
- `SUBTILE_HALVES.md` - BF16 two-half processing
- `K_DIRECTION_LINEAR.md` - K is uniform linear
- `A_B_ZERO_ASYMMETRY.md` - asymmetric zero gating
- `LATENCY_DATA_INDEPENDENT.md` - dedup is power-only
- `2CTA_DEDUP.md` - per-CTA dedup
- `MMA_SYNC_POWER.md` - legacy path also has dedup
- `NVFP4_SF_POWER.md` - SF entropy effect
- `DISABLE_LANE_POWER.md` - linear column disable scaling
- `SUBTILE_SPARSE_VALIDATION.md` - sparse zero validation
- `PER_SM_POWER_SCALING.md` - linear per-SM model
- `PRACTICAL_THROUGHPUT_GAIN.md` - end-to-end throughput
- `POWER_FLOOR.md` - absolute minimum power
