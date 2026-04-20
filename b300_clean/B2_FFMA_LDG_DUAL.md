# FFMA + LDG dual-pipe issue — V4 / B2

**Date: 2026-04-20.** Test `bench_ffma_ldg_dual.cu`. 1500 MHz, persistent
256-thread blocks, NC=8, ILP=8. LDG hot in L1 (small footprint).

## Result

| Mode | Description | Time (ms) | Per-op cy/thread |
|------|-------------|-----------|------------------|
| 0 | FFMA only (2-source) | 1.098 | 2.06 |
| 1 | LDG only, chain dep on addr | 8.455 | 15.85 |
| 2 | FFMA + LDG mixed (LDG chain) | 9.457 | 8.87 (per pair: 17.7) |
| 3 | LDG only, NO chain dep | 15.657 | 29.36 |
| 4 | FFMA + LDG mixed (LDG no chain) | 14.726 | 13.81 |

## Mixed overlap analysis

- **MIX 2 (FFMA + LDG-chain)**: sequential predicted 9.553 ms → measured
  9.457 ms = **1.0% overlap**. Essentially serial.
- **MIX 4 (FFMA + LDG-no-chain)**: sequential predicted 16.755 ms →
  measured 14.726 ms = **12.1% overlap**.

## Comparison to other pipe pairs

| Mix | Overlap | Pipe relationship |
|-----|---------|-------------------|
| FFMA + IADD3 | 14.2% | unified ALU/FMA cluster |
| FFMA + SHFL  | 14.7% | LSU/SHFL (separate per docs) |
| FFMA + LDG (chain) | **1.0%** | LSU/LDG (separate per docs) |
| FFMA + LDG (no-chain) | 12.1% | LSU/LDG |
| FFMA + MUFU | ~100% | XU/MUFU (commit `8012b98`) |

## Surprise: LDG no-chain is SLOWER than LDG-chain

15.66 vs 8.45 ms for LDG-only. The chain-dep version uses per-thread
distinct addresses (perturbed by chain) → 32 lanes hit 32 different
L1 banks. The no-chain version has all 32 lanes reading SAME address
(`idx = (i + k*64) & 0xFF` has no threadIdx dependency) — should be
broadcast = fast, but measures slower.

**Hypothesis**: with no chain dep, many LDGs fly in-flight; L1 queue
fills up; backpressure stalls the SMSP. With chain dep, LDGs gate
naturally, never queueing past 8 outstanding (one per chain).

## Key takeaway

**Memory-bound kernels (LDG-bound) cannot hide FFMA latency with
LDG**, contrary to common assumption. The SMSP issue port + LDG
register write port dominate; FFMA can fit into ~12% of LDG
slack at best.

This is the OPPOSITE of MUFU + FFMA where MUFU is so slow per-inst
that FFMA fits 100% in the slack. LDG at 16 cy/inst is too fast to
leave room for FFMA dispatch.

## Implications for kernel design

- **Don't expect FFMA to hide under L1 LDG**. Plan for additive cost.
- **MUFU operations DO hide under FFMA** — softmax / GELU kernels
  benefit from this without effort.
- **For memory-bound kernels**, focus optimization on LDG pipe (cache
  hits, prefetch, vectorization), not on adding more FFMA.

## Confidence

- **HIGH** for measurements (3 trials, stable, SASS-verified)
- **HIGH** for FFMA-only and LDG-chain timings
- **HIGH** for "1% overlap with chain"
- **MED** for "12% overlap without chain" — depends heavily on cache
  state (L1 hit) and queue depth, may differ for L2 / DRAM LDG
- **MED** for hypothesis "queue backpressure" causing no-chain
  slowdown — should verify with ncu `lts__t_pipeline_active` etc.

## Files

- `tests/bench_ffma_ldg_dual.cu` — modes 0-4
