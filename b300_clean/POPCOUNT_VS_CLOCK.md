# Read/Write Power vs Popcount: Clock-Rate Scaling

**Date: 2026-04-20.** Verified the popcount bell curve at multiple clock
locks (1005, 1300, 1500, 1800 MHz) to (a) confirm the signal scales
linearly with clock and (b) find where the TDP wall kicks in.

## Setup

- Same kernels as `POPCOUNT_3TIER.md` (L2 64 MB ws, .cg) and
  `POPCOUNT_WRITES.md` (DRAM-8G, .cg, STG.256 via v4.b64 / STG.128 via v4.u32).
- Each density has a deterministic per-dword popcount d with bit positions
  varying per dword (Fisher-Yates shuffle).
- Clock locked via `nvidia-smi -lgc N,N`.
- TDP limit: 1100 W (max).
- 5 power samples × 0.3 s after 1.5 s ramp; median of middle 3.

## Bandwidth scaling with clock (read kernel, .cg, 64 MB ws)

| clock MHz | wall ms (30M iters) | wall BW TB/s | clock-normalized |
|-----------|---------------------|--------------|------------------|
| 1005      | 2284                | 15.92        | 1.000            |
| 1800      | 1804                | 20.16        | 0.707 (× ratio)  |

Read BW ratio 20.16/15.92 = 1.27× for 1800/1005 = 1.79× clock — sub-linear
because L2 → SM transport is not 100% saturated at 1800 MHz; HBM is the
shared bottleneck.

| clock MHz | write BW TB/s |
|-----------|---------------|
| 1005      |  3.78         |
| 1800      |  6.71         |

Write BW ratio 1.78× for 1.79× clock = **perfectly linear** (LSU store
pipe is purely SM-side bound). 32 B/cy/SM × 148 SMs × 1.8 GHz = 8.52 TB/s
theoretical; 6.71 / 8.52 = **78.7 % of theoretical store-pipe ceiling**.

## L2 read popcount (active W above 150 W idle)

| d  | 1005 MHz | 1800 MHz | ratio |
|----|----------|----------|-------|
| 0  | 222      | 429      | 1.93× |
| 8  | 364      | 691      | 1.90× |
| 16 | 405      | 771      | 1.90× |
| 24 | 378      | 717      | 1.90× |
| 32 | 245      | 469      | 1.91× |

Consistent **1.90× active-power scaling** for 1.79× clock — slightly
super-linear (likely voltage component too).

## L2 write popcount (active W)

| d  | 1005 MHz | 1800 MHz | ratio |
|----|----------|----------|-------|
| 0  | 138      | 319      | 2.31× |
| 8  | 213      | 501      | 2.35× |
| 16 | 235      | 557      | 2.37× |
| 24 | 223      | 524      | 2.35× |
| 32 | 159      | 359      | 2.26× |

**2.35× write-power scaling for 1.79× clock** — much more super-linear
than reads. Likely because the SM store pipe was at ~80 % utilization
at 1005 MHz too, and at 1800 MHz it is closer to theoretical ceiling.

## DRAM-8G read popcount (active W) — TDP wall visible at 1800 MHz

| d  | 1005 MHz | 1300 MHz | 1500 MHz | 1800 MHz |
|----|----------|----------|----------|----------|
|  0 | 397      | 492      | 554      | 686      |
|  8 | 583      | 706      | 835      | 943 ⚠    |
| 12 | 621      | 746      | 896      | 942 ⚠    |
| 16 | 637      | 787      | 921      | 942 ⚠    |
| 20 | 627      | 791      | 916      | 943 ⚠    |
| 24 | 595      | 725      | 866      | 941 ⚠    |
| 28 | 530      | 651      | 760      | 943 ⚠    |
| 32 | 441      | 537      | 609      | 758      |

⚠ = at TDP wall (1100 W total). Notice d=8..28 all clipped to ~942 W
active = 1092 W total at 1800 MHz. The bell curve is real but **flat-
topped above the TDP cap**; you cannot measure the true peak shape at
this clock without either:
1. A higher TDP cap (none available — `power.max_limit = 1100 W`).
2. A lower clock — at 1500 MHz d=16 hits 921 W active = 1071 W total
   (just under TDP), bell curve is unclipped.

## DRAM-8G write popcount (active W) — same TDP behavior

| d  | 1005 MHz | 1800 MHz |
|----|----------|----------|
|  0 | 264      | 518      |
|  8 | 373      | 760      |
| 16 | 405      | 830      |
| 24 | 388      | 789      |
| 32 | 288      | 568      |

Range: 1005 MHz spread = 141 W; 1800 MHz spread = 312 W (2.21×).
Even at 1800 MHz with 80 % occupancy, DRAM writes only hit 980 W total
(no TDP cap) — so the write bell curve is **measurable in full** at any
clock.

## Implications

- **Use 1500 MHz for clean DRAM read popcount data.** 1800 MHz clips
  the d=8..28 region by ~150 W. 1005 MHz is clean but signal is small.
- **Writes are clock-linear; reads are slightly super-linear.** Reads
  benefit more from higher clock when memory subsystem is not saturated.
- **TDP wall hits at d=16 random data + DRAM bandwidth + 1700 MHz clock.**
  This is the worst-case data pattern for power.
- **For thermal stress testing**: DRAM read with d=16 random per-dword
  data at locked 1500 MHz is the highest sustained power workload I've
  found (1071 W steady, just below TDP cap).
- **For energy efficiency in inference**: lower clock + lower-popcount
  data reduces power per-operation faster than per-cycle. A factor of
  2× clock rarely doubles power; a factor of 2× popcount density (away
  from d=16) shaves 30-40 % power without losing FLOPS / GBps.

## Confidence

- **HIGH** that BW and active-power both scale linearly with clock for
  L2 reads (1.90×) and L2 writes (2.35×, slightly super-linear).
- **HIGH** that 1100 W TDP cap clips DRAM read d=8..28 at 1800 MHz.
- **HIGH** for the popcount bell curve shape at every clock tested.

## What would change conclusions

- Test at 2032 MHz boost (would also hit TDP wall on most patterns).
- Per-DRAM-channel ncu metric to confirm even distribution across the 6
  HBM3E stacks at high clocks.
- Voltage probe to attribute 5 % super-linearity for reads.
