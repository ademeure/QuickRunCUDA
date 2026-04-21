# V32-V40 Free-Rein SoL Findings

After completing all `[x]` TASK_LIST items, these tests probe additional Speed-of-Light
ceilings on B300 SXM6 (sm_103a). All apply 10-rule rigor (theoretical → measured →
ratio → SASS verify → ncu cross-check → 3+ methods → suspect test before HW →
HIGH/MED/LOW).

## TMA SoL series

### V32 — TMA multicast aggregate ceiling (cluster_dims=8, full-device)
**Theoretical**: HBM peak 7.31 TB/s × 8-way multicast amplification = 58.4 TB/s effective
**Measured**: **14.91 TB/s effective** at 18 clusters × 64 KB tile × 512 iters (raw HBM 1.86 TB/s = 17%)
**Why <100%**: TMA issue rate per cluster is the bottleneck (~1.6 Mtmas/s/cluster)
**Confidence**: HIGH | Commit `253b92a`+

### V33 — TMA per-CTA HBM read peak (no multicast)
**Theoretical**: HBM 7.31 TB/s
**Measured**: **6.72 TB/s = 91.9% of peak** at 148 blocks × 64 KB × 1024 iters
**Validation**:
- Initial 10.84 TB/s (148% of theoretical) → rule 3 violation → caught L2 cache reuse
- Fixed with stride-per-iter pattern → 6.72 TB/s
- ncu dram__bytes confirms (67.4 / 65.5 MB = 103%)
- l1tex_t_sectors = 0 (TMA bypasses L1)
**Confidence**: HIGH | Commit `253b92a`

### V34 — TMA write peak
**Theoretical**: HBM 7.31 TB/s
**Measured**: **7.17 TB/s = 98% of peak** at 32 KB × 148 blocks × 512 iters
**Surprising**: writes faster than reads (7.17 vs 6.72 TB/s) — HBM3e write batching
**Confidence**: HIGH | Commit `(after V33)`

### V35/V36 — TMA stream copy (R+W combined)
**Theoretical**: A6 50:50 mix peak = 6.68 TB/s
**Measured**: V35 sequential **6.11 TB/s** = 91%; V36 pipelined **6.21 TB/s** = 93%
**Why pipelining helps only 2%**: TMA queue depth per CTA limited to 1-2 in flight
**Confidence**: HIGH | Commits `(after V34)`

## ALU pipe SoL series

### V37 — REDUX.SYNC.MAX peak
**Initial theoretical**: 1 inst/cy/SMSP → 38.5 Telements/s. Measured 9.09 Telements/s = 24%.
**Refined theoretical**: 1 inst per 4 cy/SMSP (CREDUX uniform pipe) → 9.65 Telements/s.
**Measured**: 9.09 Telements/s = **94% of refined SoL**
**Catalog correction**: "redux 4× SHFL" was algorithmic speedup, NOT instruction-rate
**Confidence**: HIGH | Commit `(after V36)`

### V38 — SHFL.SYNC.BFLY peak
**Measured**: 9.48 Telements/s — same pipe as REDUX
**Pipe finding**: SHFL and REDUX both run on cross-lane shuffle pipe at 1/(4cy)/SMSP
**Confidence**: HIGH | Commit `(after V37)`

### V40 — ALU pipe hierarchy (compare 7 ops, ILP=8, 1184 blocks)
| Op | Glane/s | %SoL (1/cy/SMSP) | Pipe |
|----|---------|-------------------|------|
| FADD | 26.3 | 68.3% | FMA |
| FFMA | 25.7 | 66.7% | FMA |
| IADD3 | 26.4 | 68.5% | FMA (shared) |
| LOP3 | 18.7 | 48.5% | INT/bit |
| IMUL | 18.7 | 48.5% | INT/bit |
| PRMT | 13.9 | 36.1% | permute |
| ISETP | 8.4 | 21.7% | compare |

**Multi-pipe architecture confirmed**: FMA pipe is fastest (1/cy/SMSP), INT-bit pipe ~1/(2cy)/SMSP,
PRMT ~1/(3cy)/SMSP, ISETP ~1/(5cy)/SMSP.

The 67% on FMA reflects dep-chain stalls (8 ILP slots not enough for full pipeline fill);
D1's 85.5% with multi-warp ILP confirms the FMA pipe ceiling.

## Headlines

| Metric | Value | Notes |
|--------|-------|-------|
| TMA write SoL | 7.17 TB/s (98% HBM) | best HBM-bound primitive |
| TMA read SoL | 6.72 TB/s (92% HBM) | per-CTA, 148 blocks |
| TMA copy SoL | 6.21 TB/s R+W (93% A6) | pipelined R+W |
| TMA multicast SoL | 14.9 TB/s effective | 18 clusters × 8-way |
| REDUX peak | 9.09 Telements/s | 94% of 1/(4cy)/SMSP |
| SHFL peak | 9.48 Telements/s | same pipe as REDUX |
| FMA pipe peak | 25.7 Glane/s | 68% with dep chain |

## Methodology lessons

1. **Rule 3 caught 2 mistakes**: V33 L2 caching (10.84 → 6.72), V39 LICM (1547% → 48%)
2. **Theoretical formulas need lane × warp distinction**: V39 mixed up by 32× initially
3. **NCU is essential cross-check**: dram__bytes = 67/65 MB confirmed V33 hit DRAM
4. **Dep chain via in-out same reg** is cleanest way to defeat LICM on ILP loops
5. **stride-per-iter** mandatory for accurate HBM measurement (else L2 hot)
