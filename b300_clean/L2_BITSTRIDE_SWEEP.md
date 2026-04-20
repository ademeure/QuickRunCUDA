# L2 Read Power vs Bit-Stride Duplication: NULL RESULT

**Date: 2026-04-20.** Tested whether L2 reads of data with bit-level
duplication patterns draw less power than full-random data. Hypothesis:
HBM3E / L2 signaling could be sensitive to chunk-level redundancy at
8B / 32B / 128B / 1KB granularity (cache-line, sector, page boundaries).

## Setup

- `tests/bench_l2_bitstride.cu` (runtime `pattern_mode` arg).
- 16 v4 .cg loads per inner iter, UNROLL=32, BLOCK=512, 148 blocks.
- Working set 64 MB (fits in 96 MB L2; full L2-warm).
- Sustained ~16 TB/s L2 bandwidth (full peak).
- Clock locked 1005 MHz.
- Sample window: 1.5s window after 1.8s ramp, 5 samples × 0.3s, drop
  first + last, median of middle 3.
- Init pattern: each pair of `pattern_mode`-bit chunks is duplicated; pairs
  carry independent random data. Verified visually with
  `tests/verify_bitstride_runtime.cu`.

## Results (all 64-MB L2-warm, ~16 TB/s sustained)

```
p_bits  granularity        med_W   delta vs zeros   delta vs random
0       all zeros          365.0   —                -184  (BASELINE LOW)
1       bit-pair           537.1   +172            -12.1
2       2-bit chunks       537.9   +172.9          -11.3
4       nibble pairs       538.0   +173.0          -11.2
8       byte pairs         540.3   +175.3           -8.9
16      halfword pairs     539.5   +174.5           -9.7
32      word (4B) pairs    538.3   +173.3          -10.9
64      8B pairs           547.9   +182.9           -1.3
128     16B pairs          547.7   +182.7           -1.5
256     32B pairs          532.5   +167.5          -16.7  ← anomalous LOW
512     64B pairs          549.8   +184.8           +0.6
1024    128B pairs         546.7   +181.7           -2.5
2048    256B pairs         547.3   +182.3           -1.9
4096    512B pairs         548.9   +183.9           -0.3
8192    1024B pairs        549.2   +184.2            0.0
99      full random        549.2   +184.2            0.0  (BASELINE HIGH)
```

## Findings

### 1. Bit-stride duplication is NOT a power lever
Range across all duplication patterns (p=1..8192, excluding extremes):
532 – 550 W = **18 W spread (3.3% of mean 540 W)**. The "wire entropy"
from L2 to SMs is dominated by **per-cycle bit toggling on randomly-shaped
data**, regardless of whether two 1KB chunks happen to be identical.

This refutes the naive hypothesis that L2/HBM line-level dedup or scrambling
benefits from line-pair duplication. **Whatever the L2 reads to SM data path
does, it does not aggressively exploit chunk-level repetition.**

### 2. Zeros are the only ~50% power lever (-184 W = -33%)
Going from full random (549 W) to all zeros (365 W) saves 184 W — and
that effect dominates everything else by an order of magnitude.

### 3. Mild fine-grain trend (~12 W ≈ 2%)
Patterns with p=1..32 cluster slightly LOW (537-540 W), patterns p=64..8192
cluster slightly HIGH (547-549 W). There is a real but tiny "fine-grain
helps" effect of ~12 W.

Possible mechanism: at very fine grain (bit/nibble pair), even though the
RAW bit toggle rate looks similar to random, some SR-latch or precharge
stage in the L2 → SM mesh sees cleaner row patterns. But the magnitude is
small enough to be on the edge of measurement noise.

### 4. p=256 anomaly: 532.5 W (-17 W vs neighbors)
At pattern_mode=256 (= 32-byte pair size), power dips by 17 W vs neighboring
p=128 and p=512. **32 bytes is the L2 sector size on B300.** This may
indicate a real architectural feature where L2 sector-aligned duplication
hits a fast path. Worth re-running 3× to confirm.

## Devil's advocacy

- **Could the test be wrong?** Init verified bit-stride patterns correctly
  (`verify_bitstride_runtime.cu`); main kernel achieves ~16 TB/s same as
  reference peak BW kernel; same SASS for all patterns. Unlikely to be
  measurement bug.
- **Could it be L2 hit/miss differential?** No — working set 64 MB fits in
  96 MB L2, and same access pattern across all patterns. Cache behavior is
  identical; only bytes-on-wire content differs.
- **Could clock be unstable?** Clock locked 1005 MHz; verified before/after
  via nvidia-smi. No throttling at this load (~550 W < 1100 W TDP).
- **Could it be init-residual?** No — we wait 1.8s after launch; init
  finishes in ~50 ms; main kernel runs ~3.7 s; sample window is in the
  middle of main-kernel-only steady state.

## What WOULD vary with content (preview of next test)

The previous L2 peak BW test (`bench_l2_data_peak.cu`) showed:
```
0x00000000  zeros        363 W    (popcount 0)
0x12121212  byte_const   365 W    (popcount 8)  ← LOW
0xFFFFFFFF  all-ones     542 W    (popcount 32) ← HIGH
0xFF00FF00  byte_alt     543 W    (mixed, popcount 16)
random      ~50%         545 W    (popcount ~16) ← HIGH
```

This suggests power may depend on **average bit density (popcount)**, not
on within-data redundancy patterns. Specifically:
- Low popcount (0-8 bits set per dword): low power
- Mid popcount (~16): high power
- High popcount (32): also HIGH (542 W), so it's NOT a symmetric DBI effect

**Next investigation**: sweep popcount density 0%, 5%, 10%, ..., 100% with
otherwise random bit positions, to map the actual energy-per-bit curve.

## Confidence

- **HIGH** that bit-stride duplication patterns p=1..8192 do NOT reduce
  L2 read power below the "random" baseline. Range is small (~3%).
- **MED** on the p=256 anomaly — could be 1× lucky sample. Re-run needed.
- **HIGH** that zeros alone draw 33% less power than random.

## What would change conclusions

- Re-run with 3 trials per pattern to confirm p=256 dip (or not).
- Test `popcount`-controlled patterns to find the actual energy-vs-density curve.
- Test patterns where both halves of pairs are SAME but with controlled
  popcount (decouple content from duplication).
