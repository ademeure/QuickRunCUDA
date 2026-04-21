# V9: Memory hierarchy latency ladder (pointer-chase)

## Measurements

Pointer-chase via `idx = A[idx]` with LCG-permuted chain (1024 hops):

| Buffer     | Target tier | Latency (cy/hop) | Latency (ns @ 2.032 GHz) |
|------------|-------------|-------------------|--------------------------|
| 1 KB       | L1 hit      | 47 cy             | 23 ns                    |
| 4 KB       | L1 hit      | 73 cy             | 36 ns                    |
| 16 KB      | L1/L2 mix   | 164 cy            | 81 ns                    |
| 64 KB      | L2 hit      | 255 cy            | 125 ns                   |
| 256 KB     | L2 hit      | 295 cy            | 145 ns                   |
| 1 MB-128 MB| L2 hit      | 305-309 cy        | 150-152 ns               |
| 1 GB       | DRAM (+ L2) | 317 cy            | 156 ns                   |

## Interpretation

- **L1 hit latency: ~47 cy = 23 ns** (1 KB chain, least prefetch-prone)
- **L1 miss → L2: ~73-165 cy** (transitional at 4-16 KB)
- **L2 steady-state: ~295-310 cy = 145-152 ns**
- **DRAM (> L2 size 126 MB): ~317 cy = 156 ns**

The DRAM vs L2 difference is only ~12 cy (~4%), which is SURPRISING. Possible explanations:
1. HW prefetcher is catching the LCG pattern despite random-ish hops
2. L2 has "adjacent line prefetch" that helps pointer chase
3. True DRAM random latency on B300 is surprisingly low

## 10-rule caveats

- Rule 9 (suspect test before HW): the small DRAM-vs-L2 gap is unexpected.
  A truly random (Fisher-Yates) permutation might give different numbers.
- SASS shows `LDG.E` in tight loop — no prefetch asm.
- **Confidence: MEDIUM**. Raw latency numbers are reliable; the L2/DRAM
  discrimination is suspect due to possible prefetcher interference.

## Cross-reference to V8 findings

V8 I3 reported "HBM avg 60 cy, max 1433 cy" — that was a different workload
(latency sampling under load). The 60 cy matched L2 hit in that context;
max 1433 cy matches refresh/row-miss spikes. My 317 cy likely average case.

## Op latency + memory latency combined ladder

| Tier              | Latency  | Bandwidth      |
|-------------------|----------|----------------|
| FFMA / FADD       | 4.22 cy  | 75 TFLOPS      |
| IMAD              | 4.25 cy  | 38.4 TOPS      |
| DFMA              | 63.68 cy | 1.20 TFLOPS    |
| L1 hit            | ~47 cy   | 30.5 TB/s      |
| L2 hit            | ~300 cy  | 13.85 TB/s     |
| DRAM (pointer chase) | ~317 cy | 7.0 TB/s seq  |

The 300 cy L2 vs 4 cy FFMA ratio (75×) means hiding 1 L2 miss requires
filling 75 cy of compute — doable with 8-chain ILP × 10 nested iters.

## Implication

To hide DRAM latency (~317 cy), a kernel needs ~75× more compute ops
than memory hops. For a BW-bound kernel doing 1 LDG per FFMA, latency
hiding requires ~75 warps/SM (= 8 blocks × 8 warps each) to saturate.