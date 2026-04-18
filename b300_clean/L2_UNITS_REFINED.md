# L2 Atomic Unit Count — REFINED with single-address data

Earlier I downgraded the catalog "~32 L2 atomic units" claim. With the
new single-address measurement, we can re-derive more carefully.

## Two key data points

1. **Single cache line peak** (commit b39fada):
   - All 1184 blocks atomicAdd to address [0]
   - Saturation: 1.54 Gwarp/sec = 0.83 L2 packets per video clock cycle
   - This is the per-LINE throughput → **1 L2 atomic unit handles ~0.83 pkt/cy**
   - Each L2 atomic unit serves one cache line at a time

2. **All-distinct-lines peak** (earlier):
   - Each thread targets unique cache line
   - Aggregate: 50 Gops/sec uncombined = 27 L2 packets per video cycle

## Derivation
   ~27 packets/cy / 0.83 packets/cy/unit = ~32.5 active L2 atomic units

So the catalog claim of "~32 L2 atomic units" appears VALID after all,
but only when you look at TWO scenarios together:
- Single-line measurement gives per-unit throughput
- Multi-line measurement gives aggregate parallelism
- Ratio = unit count

This is a CONSISTENT interpretation. Earlier my "downgrade to LOW conf"
was hasty — given the single-address data I can now reconcile.

## Confidence levels
- Per-unit throughput 0.83 pkt/video-cy: **HIGH** (4 block counts converge,
  ncu DRAM near zero, clean single-line saturation)
- Aggregate ~27 pkt/video-cy with distinct lines: **HIGH** (multiple WS,
  multiple combine ratios converge)
- Inferred ~32 L2 atomic units: **MEDIUM-HIGH** (consistent with both
  measurements; could be 30 or 35; not directly observable in ncu)

## What we now know about L2 atomics on B300

Per L2 atomic unit:
  Throughput: 0.83 packets per video-clock cycle (1.86 GHz)
            = 1.55 G packets per sec per unit
  Bottleneck for single-address: one unit per line
  
Aggregate (with line-spread):
  ~32 units * 0.83 = 26.6 = ~27 packets/cy across whole chip
  = 50 G packets/sec = 5.5 TB/s DRAM (at line-RMW for unique lines)

