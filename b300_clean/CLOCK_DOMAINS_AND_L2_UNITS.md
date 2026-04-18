# B300 sm_103a Clock Domains and L2 Atomic-Unit Count Analysis

## Three independent clock domains on B300

  SM/Graphics clock: variable (boost 2032 MHz, sustained-locked floor 1920 MHz)
  Video/L2 clock:    1860 MHz CONSTANT (cannot be locked separately)
  HBM3E memory:      3996 MHz CONSTANT

The "video" clock domain on B300 = effectively the L2/XBAR clock
(non-graphics subsystem clock). NVML query: clocks.current.video.

When SM clock locked to "1800 MHz" via nvidia-smi -lgc 1800:
  Actual SM clock pinned at 1920 MHz (lock-floor; can't go below)
  Video clock UNCHANGED at 1860 MHz

## What each clock bottlenecks

Atomic test results, comparing SM=2032 boost vs SM=1920 locked, video=1860 always:

Pattern               | Boost (2032) | Locked (1920) | Ratio  | Bottleneck
----------------------|--------------|---------------|--------|------------
int32 c=32 OCC=8      | 769 Gops     | 729.7 Gops    | 94.9%  | SM clock (1920/2032=94.5%)
                      | 24.0G L2 pkt | 22.8G L2 pkt  |        | (warp issue rate limit)
int32 c=1 OCC=8       | 49.7 Gops    | 50.0 Gops     | 100%   | L2/DRAM (video clock)
uint64 c=1 OCC=8      | 49.8 Gops    | 49.7 Gops     | 100%   | L2/DRAM
b128 c=1 OCC=8        | 42.2 Gops    | 41.9 Gops     | 100%   | L2/DRAM
b128 c=8 OCC=8        | 174.3 Gops   | 174.4 Gops    | 100%   | DRAM (5.51 TB/s)

CONCLUSION: combined atomics are SM-issue-bound;
            uncombined atomics are L2/DRAM-bound (no SM dependency).

## L2 atomic units count via video-clock derivation

For uncombined atomic (each thread = 1 L2 packet):
  Throughput: 50 Gops/sec L2 packets (no merging)
  Video clock: 1.860 GHz
  L2 packets per video-clock cycle: 50/1.86 = 26.9 in parallel

  Catalog claim: "~32 L2 atomic units across 2 partitions" (E4 task)
  Measurement:   27 utilized / 32 catalog = 84% of catalog atomic units

For combine=2 (different stride pattern):
  98.9 Gops/sec thread-atomics / 2 lanes-per-line = 49.5 G L2 packets/sec
  At 1.860 GHz: 26.6 L2 packets/cy = SAME as combine=1 ceiling (~27)
  → confirms ~27 is the architectural L2-atomic-unit ceiling per cycle

For combine=32 (warp-merged, 1 packet per warp):
  769 Gops/sec / 32 = 24.0 G L2 packets/sec at SM 2032 MHz
  At video 1860 MHz: 24.0/1.86 = 12.9 L2 packets/cy
  But combine=32 is SM-bound, not L2-bound, so this is NOT the L2 ceiling.

## L2 BW per video cycle (assuming 32B sector)

Lts traffic (lts__t_bytes ncu): 4.79 TB/s for combine=1 atomic
At 1.860 GHz video: 4.79e12 / 1.86e9 = 2575 B/cy of L2-bus traffic

If each L2 atomic = 32B sector, expected: 27 × 32 = 864 B/cy
Measured: 2575 B/cy = 3.0x expected
→ Each L2 atomic causes ~95 B traffic = ~3 sectors (read + write + atomic unit)

DRAM traffic at video clock:
  5.52e12 B/s / 1.86e9 cy/s = 2968 B/cy of DRAM
  Per L2 atomic: 2968/27 = 110 B/op ≈ 1 cache line (128B with some efficient batching)

## Summary
  L2 atomic unit count on B300: ~27 active in parallel per video cycle
                                = 84% of catalog "~32 units"
  L2 video clock: 1860 MHz (constant; doesn't follow SM clock)
  Each L2 atomic op at the wire = ~95B L2 traffic / ~110B DRAM traffic
