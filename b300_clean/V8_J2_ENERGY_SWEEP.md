# V8 J2: Auto-clock energy minimum — workload-dependent sweet spot

Method: lock clock via `nvidia-smi -lgc`, run kernel for 3+ seconds,
sample `power.draw` during steady state, compute energy = P × T.

## FFMA (compute-bound, V6 C1 kernel, 148 SMs × 256 threads)

| Clock     | Time (ms) | Power (W) | Energy (J) | Perf (TFLOPS) | Eff (GFLOPS/W) |
|-----------|-----------|-----------|------------|---------------|----------------|
| 1005 MHz  | 4446      | 221       | 982        | 26.8          | 121            |
| 1500 MHz  | 2965      | 293       | 870        | 40.2          | **137 ← best** |
| 1920 MHz  | 2312      | 415       | 960        | 51.5          | 124            |

**FFMA optimum: 1500 MHz** — 10% better efficiency than boost, 13% better than 1005.

## DRAM-ish (V6 C2 kernel, 256 MB buffer — partial L2)

| Clock     | Time (ms) | Power (W) | Energy (J) | BW (GB/s) | Eff (GB/s/W) |
|-----------|-----------|-----------|------------|-----------|--------------|
| 1005 MHz  | 4222      | 236       | 995        | 7057      | **30.0 ← best** |
| 1500 MHz  | 3512      | 285       | 1002       | 8484      | 29.7         |
| 1920 MHz  | 3130      | 363       | 1136       | 9521      | 26.2         |

**DRAM optimum: 1005 MHz** — 14% better than boost. At higher clocks, core spinning power
grows faster than incremental memory throughput gain (DRAM bound already near peak).

## Idle power baseline (GPU with open context, no kernel)

| Clock     | Idle power |
|-----------|------------|
| 1005 MHz  | 153 W      |
| 1500 MHz  | 167 W      |
| 1920 MHz  | 197 W      |

So "compute-only" power (total − idle) for FFMA is ~68 / 126 / 219 W. Idle
overhead is substantial (~150-200 W) and is the reason ultra-low clocks are
terrible for energy — the idle-time fraction of total explodes.

## Guidance

- **Latency-critical / throughput-critical**: always BOOST (2032 MHz). Energy is secondary.
- **Compute-bound batch (FFMA, large MMA)**: **1500 MHz** is the efficiency sweet spot on B300.
- **Memory-bound streaming**: **1005 MHz** — DRAM peak is already saturated at that clock.
- **Avoid < 1005 MHz**: idle/leak power dominates at very low frequencies (V6 finding).

## V6 cross-check

V6 M10 said "USE BOOST for ML inference — 3× lower energy than 510 MHz". That compared
to 510 MHz where everything crawls. In the 1005–2032 MHz range (the practical range),
BOOST actually loses to 1500 MHz on compute and to 1005 MHz on memory.

## How to use

For a given kernel, if you know it's compute-bound (`pipe_fma.pct_of_peak > 70%`), lock to
1500 MHz to save ~10% energy at minor perf cost. If memory-bound
(`dram.throughput.pct_of_peak > 70%`), lock to 1005 MHz to save ~14% energy.

Automation: `utils/workload_classify.sh` (V8 L5) can be extended to emit the recommended
clock based on `pipe_fma` vs `dram.throughput` dominance.
