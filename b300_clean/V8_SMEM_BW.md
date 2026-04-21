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

## ldmatrix follow-up (ADDED)

Tested `ldmatrix.sync.aligned.m8n8.x1.shared.b16` (128 B per warp issue):
- 0 bank conflicts ✓ (ncu)
- wavefronts = 47.67M in 2.03 ms
- BW = 47.67M × 128 B / 2.03 ms = **3.0 TB/s** (8% of peak)

ldmatrix.x4 attempt gave 50% bank conflicts (189M / 379M) — layout wrong.

**ldmatrix does NOT improve SMEM BW over plain LDS** in our tests. Why:
- Each ldmatrix issue = 1 wavefront same as LDS
- ldmatrix has higher instruction latency (synchronizes across warp lanes)
- Dispatch rate lower than plain LDS loop (which unrolls aggressively)

ldmatrix's value is its **format**: directly feeds `mma.sync` without
re-layout. For raw BW, plain LDS with good ILP is faster.

## True SMEM SoL path

To exceed 27 TB/s requires:
- `cp.async.bulk.shared` from DRAM (async) — but this is H2S, not intra-SMEM
- `tcgen05.ld` (Blackwell tensor memory) — different memory pool
- Swizzled layouts + careful bank-avoidance

Plain LDS at 74% is the practical ceiling for scalar kernels on B300.

## Implications

- Plain LDS scalar loops peak at ~27 TB/s on B300.
- To exceed ~27 TB/s, must use matrix-load or async-load primitives.
- ncu `data_bank_conflicts_pipe_lsu_mem_shared = 0` confirms no layout
  pathology — any improvement requires instruction-level changes.
