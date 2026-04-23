# §12 pipe_alu ceiling — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §12 (L937-957)
**Test:** `tests/bench_lop3_pure.cu`

## CLAIM (catalog L937-957)

- **pipe_alu cap = 2.00 warp-instructions/SM/cycle = 64 thread-ops/SM/cy.**
- This budget is shared across ALL alu-resident opcodes (LOP3, PRMT, IADD3, ISETP, FMNMX, FSEL, F2FP, CREDUX, etc.) — no dual-issue among alu-resident ops.
- **Contrast** with pipe_fma which has heavy+lite sub-units (4.00 cap for scalar FFMA).

## TEST

```bash
ncu --metrics sm__inst_executed_pipe_alu.avg.per_cycle_active,sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active \
  ./QuickRunCUDA -f tests/bench_lop3_pure.cu -t 512 -b 1184 -A 1024 -B 1024 -C 1024 \
  -H "#define N_CHAINS 16
#define UNROLL 16
#define BLOCK_SIZE 512
#define MIN_BLOCKS 4" -0 1024 -T 1
```

## MEASURED (sweep N_CHAINS and MIN_BLOCKS to find true ceiling)

### N_CHAINS sweep (at MIN_BLOCKS=2):

| N_CHAINS | pipe_alu inst/SM/cy | % of 2.00 cap |
|---------:|--------------------:|--------------:|
| 4 | 1.37 | 68.5% |
| 8 | 1.61 | 80.5% |
| 16 | 1.81 | 90.5% |
| 32 | 1.90 | 95.0% |

### MIN_BLOCKS sweep (at N_CHAINS=16):

| MIN_BLOCKS | pipe_alu inst/SM/cy | % of 2.00 cap |
|-----------:|--------------------:|--------------:|
| 1 | 1.82 | 91.0% |
| 2 | 1.90 | 95.0% |
| **4** | **1.94** | **97.0%** ← peak |
| 8 | NaN (resource exhausted) | — |

## VERDICT

✅ **CONFIRMED** — pipe_alu cap = 2.00 inst/SM/cy. Achievable to **97%** with NC=16 + MIN_BLOCKS=4. Higher occupancy than 4 CTAs/SM exhausts resources.

**Same finding via DENSE §1**: Pure LOP3 saturates at 96.97% pipe_alu (matches our 1.94).

## Methodology lesson

To verify a "pipe X cap = N" catalog claim:
1. Run a pure-pipe-X benchmark
2. Sweep ILP (N_CHAINS) and occupancy (MIN_BLOCKS)
3. Use ncu `pct_of_peak_sustained_active` to read silicon-relative utilization
4. The peak achievable should be 95%+; if much lower, increase ILP or occupancy

This methodology was missed for the original §17 MUFU and §4 rate-cheatsheet entries (which gave under-saturated single-warp values).

## REVIEW_CHECKLIST candidates

- [x] §12 pipe_alu cap = 2.00 warp-inst/SM/cy — ✅ confirmed at 1.94 (97%)
- [ ] §12 "no dual-issue among alu-resident ops" — needs CREDUX+FMNMX co-issue test (catalog claims total 2.19 sm_inst, alu=2.00; not re-tested here)
- [ ] §12 contrast with pipe_fma 4.00 cap — ✅ confirmed via FFMA peak audit (`00a_ffma_peak.md` shows 99.5% of pipe_fma 4-cap)
