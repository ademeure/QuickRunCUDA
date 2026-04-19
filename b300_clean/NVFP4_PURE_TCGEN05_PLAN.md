# NVFP4 Pure tcgen05 Power Microbench — Plan & Initial Scaffold

Date: 2026-04-19. Free-rein continuation.

## Motivation

cuBLAS power studies always include some L2/HBM/TMA contribution. To truly
isolate the **multiplier circuit** power as a function of input bit pattern,
we need a microbench where:

1. SMEM is loaded ONCE with controllable A/B/SF data
2. Inner loop issues many tcgen05.mma instructions referencing the SAME SMEM
3. No DRAM/L2 traffic in the steady-state measurement window
4. Pattern (zero / +1.0 / random / specific bits) controllable per run

This isolates pure multiplier power from memory pipeline activity, and
removes the TMA-multicast-strategy confound seen between cuBLAS NVF4 and BF16.

## Scaffold built

`tests/bench_tcgen05_power.cu` — based on existing `bench_tcgen05_real.cu`
which has working tcgen05.mma kind::f16 m=64 n=8 PTX. Adds:
- Pattern selection via `seed` arg (0=zero, 1=ones, 2=0x55, 3=0xaa, 4=random, 5=+1.0)
- Outer/inner loop structure (outer drives sustained workload for dmon, inner
  keeps utcmma issue queue full)
- Per-CTA SMEM init to chosen pattern; both A and B the same except in random mode

## Status

- Compiles cleanly (NVRTC via QuickRunCUDA harness)
- Runs without errors at all patterns (no exceptions/aborts)
- BUT: kernel time only ~10 us regardless of outer_iters — suggests MMA isn't
  actually firing N times. Likely descriptor encoding issue or async semantics
  not waiting for completion before kernel exit.

## TODO to make functional

1. Debug why each kernel exits in ~10us (MMA not firing or completion not
   reached). Possibilities:
   - SMEM descriptor `LBO`/`SBO` wrong for our exact m=64 n=8 k=16 layout
   - tcgen05.commit/mbarrier setup releases too eagerly (phase tracking)
   - PTX `tcgen05.mma` may need explicit operand sizing matching idesc
2. Add `printf` for cycles-per-iter (read from C[0] = t1-t0 inside kernel) to
   verify MMA throughput before measuring power
3. Once functional with kind::f16, extend to kind::mxf4nvf4.block_scale.block16:
   - Allocate extra TMEM cols for SFA, SFB
   - Use `tcgen05.cp` to copy SF from SMEM to TMEM
   - Different idesc encoding for FP4 dimensions
   - Update PTX operand list

## Power-test matrix (once functional)

For each MMA kind (f16, mxf4nvf4):
- Per-tensor pattern combinations (zzzz, rzzz, zrzz, rrrr, etc.)
- Bit-pattern sweep (force one bit on A only, on B only, on both)
- SF pattern variations for mxf4nvf4
- Comparison vs cuBLAS measurements to attribute power between
  multiplier (this microbench) vs memory pipeline (cuBLAS Δ)

## Expected hardware insight

If the microbench shows EQUAL A vs B power asymmetry for kind::f16 vs
kind::mxf4nvf4, then the cuBLAS NVF4-vs-BF16 reversal is purely TMA-multicast
driven (memory-pipeline power, not multiplier).

If the microbench shows DIFFERENT A vs B asymmetry by kind, then there
ARE separate multiplier circuits (HMMA-style vs UTCMMA-style) with
different operand-port topologies, and our cuBLAS observations also have
a multiplier-level component on top of the TMA effect.
