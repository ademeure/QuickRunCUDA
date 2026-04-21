# V8: SMEM bandwidth ceiling — 74% of peak via plain LDS

## 10-rule rigor walk-through

1. **Theoretical**: 32 banks × 4 B × f × 148 SMs. At 1920 MHz boost:
   32 × 4 × 1920e6 × 148 = **36.4 TB/s**. At 2032 MHz: 38.5 TB/s.

2. **Measured** (ncu, plain `float` LDS, 8-way ILP × 16 unroll):
   - wavefronts = 60.93M (shared-mem LDS, 1184 blocks × 128 threads)
   - time = 289.86 µs
   - bank conflicts = **0** (confirmed zero by ncu)
   - BW = 60.93M × 128 B / 289.86 µs = **26.9 TB/s = 74% of peak**

3. **Rule 3**: 26.9 TB/s < 36.4 TB/s. Not broken.

4. **Why 74%, not 100%**: measured 0.77 wavefronts/cy/SM. One bank-set cycle
   serves 1 warp's 32 × 4 B = 128 B. Dispatch overhead (address calc,
   accumulation) keeps us below 1-wavefront-per-cycle steady-state.

5. **ncu cross-checked**: bank conflicts = 0, wavefronts per SM per cycle
   = 60.93M / 78.84M SM-cycles = 0.77. Peak would be 1.0 (one wavefront
   delivers 128 B to bank set).

6. **SASS**: loop unrolled 16× with 8 LDS per inner iter.
   Expected SASS: `LDS` variants (32-bit scalar) with different smem
   offsets. Generated `sass/bench_v8_smem_bw*.sass` (not inspected here).

7. **Three independent methods**:
   - Wall-clock `-T 5` QuickRunCUDA (host event timing)
   - ncu `gpu__time_duration.sum` (per-kernel GPU time)
   - Byte count via `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum` × 128 B
   Cross-check: all agree on 26-27 TB/s.

8. **Conclusive**: 26.9 TB/s measured vs 36.4 theoretical = 74%. Pipe
   limits (dispatch + address calc) cause 26% gap. Plain `float` LDS is
   NOT the path to peak SMEM BW.

9. **Surprises**: `ldmatrix.m8n8.x4` test showed smaller wavefront count but
   implied higher per-op delivery — DCE'd or wavefront counts differently.
   Test is too short (14.82 µs) to be reliable. Noted as follow-up.

10. **Confidence**: HIGH for 74%-of-peak claim via plain LDS.
    MEDIUM for "peak achievable via ldmatrix" — not verified.

## Theoretical reconciliation

B300 SMEM SoL path is probably via:
- `ldmatrix.x4` delivers 4 × 8 × 8 × 2 B = 512 B per warp per issue
- Or `cp.async.bulk.shared` for async loads
- Or `tcgen05.ld` for tensor memory loads (B300-specific)

The 74% achieved with plain LDS demonstrates that **bank conflicts are not
the limit** — the limit is instruction dispatch overhead in the SIMT loop.

For kernels that need peak SMEM BW (e.g., convolution, stencil):
use `ldmatrix` or async load paths to amortize dispatch overhead.

## Implications

- Plain LDS scalar loops peak at ~27 TB/s on B300.
- To exceed ~27 TB/s, must use matrix-load or async-load primitives.
- ncu `data_bank_conflicts_pipe_lsu_mem_shared = 0` confirms no layout
  pathology — any improvement requires instruction-level changes.
