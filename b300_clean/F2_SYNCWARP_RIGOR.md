# F2: __syncwarp() mask cost — full mask is FREE, partial costs BSYNC

## Theoretical
- `__syncwarp(mask)` on a fully-converged warp where `mask == 0xFFFFFFFF` should be a no-op (the warp is already synchronized at instruction granularity).
- Partial mask requires hardware to track active lanes via BSYNC.
- CTA-wide barrier (bar.sync) requires cross-warp synchronization through SM-shared barrier hardware.

## Methodology rigor
- N=16 inner unroll to amortize loop overhead
- All 32 lanes always call sync (no condition divergence)
- Tested with both compile-time const and runtime-derived masks
- SASS-verified actual instruction emission

## Measured (cy per sync, single warp, all lanes participate)
| Mode | Code | cy/sync | SASS |
|------|------|---------|------|
| 0 | `__syncwarp(0xFFFFFFFFu)` const | 1.75 | **NOPs only** (no sync emitted) |
| 1 | `__syncwarp(mask)` runtime full | 1.88 | **NOPs only** (eliminated) |
| 2 | `bar.warp.sync 0xFFFFFFFF` PTX const | 1.75 | NOPs only |
| 3 | `bar.warp.sync %0` PTX runtime full | 1.88 | NOPs only |
| 4 | `bar.sync 0` (__syncthreads single block) | 14.63 | `BAR.SYNC.DEFER_BLOCKING` |

From v1 test (with divergent participation, half-warp masks):
- `__syncwarp(0x0000FFFFu)` partial: 7.25 cy/sync = **BSYNC** instruction emitted

## Conclusion
1. **`__syncwarp(0xFFFFFFFF)` is FREE on B300** — the compiler emits zero SASS instructions because the warp is implicitly converged at the full mask. Even runtime-computed full mask is eliminated (the compiler can't prove it's full but the BAR.WARP.SYNC with full mask becomes a no-op the hardware skips).
2. **Partial-mask `__syncwarp(<full)`: ~7.25 cy** = real BSYNC hardware instruction
3. **`__syncthreads()` (bar.sync 0): ~14.6 cy** at single-warp single-block (with multi-warp, contention scaling adds more)

## Practical implications
- Use `__syncwarp()` (default arg = 0xFFFFFFFF) freely — it's a true compile-time no-op
- AVOID `__syncwarp(arbitrary_mask)` unless you specifically need partial-warp sync — costs 7.25 cy
- `__syncthreads()` is 8-10× more expensive than warp sync; use scope appropriately
- The `bar.sync` cost grows with block size (catalog: 12 → 49 cy from 32 → 1024 threads)

## Confidence: HIGH
- SASS-verified emission for all modes
- Two independent measurement points (v1 with conditional, v2 with unconditional) agree
- Matches the architectural intuition: warp is always fully-converged at instruction-level when no recent divergence

What would change it: if a test with recent intra-warp divergence + full-mask syncwarp shows >2 cy, would mean the compiler tracks divergence state.
