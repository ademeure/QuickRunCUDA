# B300 Power Data-Dependence — Synthesis

**Date: 2026-04-20.** Top-level summary of the 2026-04-20 power deep-dive.
Ties together 5 sub-investigations covering compute and memory power
sensitivity to data content. **All numbers at GPU clock locked 1005 MHz
unless stated otherwise. TDP cap = 1100 W. Idle = 150 W.**

## TL;DR

**Active power on B300 is dominated by toggle-energy on whatever bus is
in flight.** The lever is per-cycle bit-flip count, not bit set count
or chunk-level redundancy:

| subsystem            | pattern at d=16 random | active W peak | range vs constant |
|----------------------|------------------------|---------------|--------------------|
| L1 reads             | per-block 64 KB, .ca   | 107           | 35 % swing         |
| L2 reads             | 64 MB ws, .cg          | **405**       | 45 % swing         |
| DRAM-1G reads        | 1 GB ws, .cg           | 604           | 39 % swing         |
| DRAM-8G reads        | 8 GiB ws, .cg          | **637**       | 38 % swing         |
| L2 writes            | 64 MB ws, .cg          | 235           | 41 % swing         |
| DRAM-8G writes       | 8 GiB ws, .cg          | 405           | 35 % swing         |
| FFMA compute         | 8-ILP self-chain       | 103           | 60 % swing         |
| IADD3 compute        | 8-ILP self-chain       | 103           | 95 % swing         |

Going to all-zeros saves 184 W (33 %) on L2 reads, **240 W (38 %) on
DRAM reads**, and even more in absolute terms at higher clocks
(see `POPCOUNT_VS_CLOCK.md`).

## What we tested

1. **Bit-stride duplication** (`L2_BITSTRIDE_SWEEP.md`):
   chunk-level redundancy from 1 bit to 8192 bits. **NULL RESULT** —
   3 % spread across all p values. The L2/DRAM signaling does NOT
   exploit chunk-level repetition.

2. **Per-dword popcount** (`L2_POPCOUNT_SWEEP.md`, `POPCOUNT_3TIER.md`):
   each dword has EXACTLY d bits set, positions vary per dword.
   **CLEAN BELL CURVE** centered at d=16 (50 % bit density), at every
   cache tier. Range 184-240 W active power at d=16 vs d=0. Mechanism:
   per-cycle bit toggle activity on the bus.

3. **Constant-pattern control** (in `POPCOUNT_3TIER.md`): every dword
   in the buffer = same constant. Only **18 W spread** across constant
   popcounts 0..32 — proves inter-dword toggle activity dominates,
   per-dword popcount alone is a small (0.56 W/bit) static-power offset.

4. **Sparsity** (`SPARSITY_3TIER.md`): random base data, X% of
   "elements" of granularity G replaced by constant value V.
   - Smooth monotonic decay from random max to constant min.
   - Below sp=10% no measurable saving; knee at sp=15-25%.
   - Granularity barely matters (byte / 4 B / 32 B / 128 B chunks
     give same curve to ±5 %).
   - Replacement value: zero < alt55 < one (always, all tiers).

5. **Writes** (`POPCOUNT_WRITES.md`): same kernel template but stores.
   - Same bell curve shape; lower absolute (0.58× L2 reads, 0.64× DRAM).
   - **B300 SM-side store pipe = 32 B/cy/SM × 148 SMs**:
     6.71 TB/s at 1800 MHz = 78.7 % of 8.52 TB/s theoretical.
     Hard structural ceiling unrelated to access pattern.
   - 1.5× write-allocate amplification persists even with full-sector
     STG.256 stores in clean coalesced patterns — it is per-transaction
     L2 metadata cost.

6. **Clock scaling** (`POPCOUNT_VS_CLOCK.md`): popcount bell at 1005 /
   1300 / 1500 / 1800 MHz.
   - Read BW: sub-linear (HBM bottleneck) / Write BW: perfectly linear.
   - Active power ~1.9× scaling for 1.79× clock (slight super-linear).
   - **DRAM reads at ≥1500 MHz hit 1100 W TDP wall** at d=8..28 random.
   - Use 1005-1500 MHz for clean popcount measurements.

7. **Compute (FFMA + IADD3)**: same bell-curve shape, much smaller
   absolute swings (60 W). Compute pipe is small relative to memory.

## The toggle model

```
P_active(memory, d, constant=False) ≈
    P_baseline_per_tier
  + alpha × popcount_d × (32 - d) / 31    (toggle energy, peaks at d=16)
  + beta × d                              (static, ~0.56 W/bit)
```

Where `alpha` and `beta` are tier-dependent:

| tier      | baseline | alpha (toggle peak coefficient) | beta (per-bit) |
|-----------|----------|----------------------------------|----------------|
| L1        | ~70 W    | ~80 W                           | ~0.4 W/bit     |
| L2        | ~220 W   | ~325 W                          | ~0.6 W/bit     |
| DRAM      | ~390 W   | ~360 W                          | ~1.4 W/bit     |
| FFMA      | ~40 W    | ~120 W                          | ~0.5 W/bit     |

For constant-data (no inter-dword toggle), only the `beta` term
contributes, giving ~18 W spread across popcount 0..32 at L2.

## Practical implications

### Inference workloads (FP4/FP8 quantized weights)
- Weights have low popcount per element (mantissas often near zero,
  signs distributed). DRAM read power for weight load could be
  100-200 W (15-25 %) lower than synthetic d=16 worst-case data.
- The "sparsity" people care about for accuracy/perf has independent
  benefit on power IF zeros aggregate at chunk level (≥byte). Below
  10 % sparsity no measurable power saving.

### Thermal validation
- **Worst-case sustained power: DRAM read at d=16 random per-dword,
  1500 MHz clock, 8 GiB working set.** Hits 1071 W steady, just below
  TDP cap. This is the recipe for max-stress thermal testing.
- Pure compute (FFMA / IADD3) is much lower power than memory.
  Stress = memory pattern + memory access.

### Power-bound throughput
- For a fixed power budget (e.g. 750 W), low-popcount data lets you
  do MORE memory operations than random data does at TDP wall.
- Example: 750 W active limit → at d=16 random can do ~1.5 GB/s less
  than at d=4 popcount (typical FP4 pattern).

### Architecture notes
- L1 vs L2 vs DRAM all show the same bell-curve shape. Toggle energy
  scales with the number of physical SerDes stages crossed (L1 inside
  SM < L2 mesh < HBM3E PHY).
- Read vs write asymmetry is 1.7-1.8× at L2 port level, but 4× at
  wall-clock level due to write-allocate amplification and SM store
  pipe ceiling.
- d=0 (zeros) saves 30-50 W more than d=32 (all ones) at every tier.
  HBM3E PHY appears to have asymmetric drive (active-low termination?).

## Confidence

- **HIGH** for the bell-curve shape and tier-amplification picture
  (5 independent investigations cross-validate).
- **HIGH** for the clock-scaling story (4 clocks tested).
- **MED** for the per-tier alpha/beta coefficients (fitted from limited
  density points, not a regression).
- **MED** for the d=0 vs d=32 asymmetry mechanism (HBM DBI is plausible
  but not proven by side-channel).

## What would change conclusions

- **Mixed read+write workload sweep at varying R:W ratios** — would
  reveal if R+W power sub-adds (shared engines) or adds (parallel pipes).
  Quick test at L2 RFRAC=100/75/50/25/0 showed: 372 W (pure R) →
  376 → 397 → 387 → 328 W (pure W). Mid-mix gives slightly higher
  power than either pure R or pure W alone. Likely worth a deeper
  sweep at higher clock to amplify signal.
- **TMA bulk loads** vs LDG.cg — different memory subsystem path,
  may have different toggle profile.
- **Per-DRAM-channel `dram__bytes_*.per_dram` ncu** to confirm even
  distribution across the 6 HBM3E stacks.
- **Actual production weight tensors** (sample BF16 / FP8 weights from
  a real model) — predict ~120-180 W reduction in DRAM read power
  vs synthetic d=16, but not yet measured directly.

## Files in this investigation

- `L2_BITSTRIDE_SWEEP.md` — null result on chunk dedup (the original
  hypothesis that started this investigation)
- `L2_POPCOUNT_SWEEP.md` — initial L2-only popcount with constant data control
- `POPCOUNT_3TIER.md` — extended to L1 / L2 / DRAM-1G / DRAM-8G
- `SPARSITY_3TIER.md` — 4 gran × 3 val × 11 sp = 396 measurements
- `POPCOUNT_WRITES.md` — write power vs popcount + write BW analysis
- `POPCOUNT_VS_CLOCK.md` — clock-scaling story with TDP wall
- `POWER_DATA_DEPENDENCE_SUMMARY.md` — this document

## Kernels in this investigation
```
tests/bench_l2_bitstride.cu
tests/bench_l2_popcount.cu
tests/bench_l2_sparsity.cu
tests/bench_pwr_l1_popcount.cu
tests/bench_pwr_l1_sparsity.cu
tests/bench_pwr_l2_popcount.cu
tests/bench_pwr_l2_popcount_write.cu
tests/bench_pwr_l2_write_warp_coalesced.cu
tests/bench_pwr_dram_popcount.cu
tests/bench_pwr_dram_popcount64.cu
tests/bench_pwr_dram_popcount_write64.cu
tests/bench_pwr_dram_sparsity64.cu
tests/bench_pwr_ffma_popcount.cu
tests/bench_pwr_iadd3_popcount.cu
tests/verify_bitstride.cu
tests/verify_bitstride_runtime.cu
tests/sweep_*.sh
```
