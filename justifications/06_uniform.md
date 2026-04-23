# §6 Uniform datapath — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §6 (L521-536)
**Test:** `tests/bench_adu_uniform.cu` (existing, OP=9 LDSM)

## CLAIM (catalog L521-536)

> "pipe_uniform hits **~1.0 warp-inst/SM/cy** in practice for ACTIVEMASK and LDSM. It does NOT contend with pipe_alu / pipe_fma — uniform ops issue in parallel with vector ops from the same SMSP.
> New on Blackwell: full uniform FP32 datapath (UFFMA, UFADD, UFMUL, etc.) — warp-invariant FP32 arithmetic can run on the uniform side, freeing vector FMA pipes for divergent work.
> pipe_uniform also handles: LDSM and ACTIVEMASK emit as uniform ops."

## TEST

```bash
CUDA_VISIBLE_DEVICES=0 ncu --metrics sm__inst_executed_pipe_uniform.avg.per_cycle_active,sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_active \
  ./QuickRunCUDA -f tests/bench_adu_uniform.cu -t 512 -b 296 -A 1024 -B 1024 -C 4096 \
  -s 8192 -H "#define OP 9" -0 1024 -T 1
```

## MEASURED (ncu, GPU 0, full occupancy)

| OP | pipe_uniform inst/SM/cy | % of peak |
|----|------------------------:|----------:|
| OP=9 (LDSM `LDSM.sync.aligned`) | **0.70** | **35.08%** |

Implied pipe_uniform peak from ncu pct: 0.70 / 0.3508 = **2.00 inst/SM/cy**.

## VERDICT

✅ **CONFIRMED — pipe_uniform peak = 2.0 inst/SM/cy (NOT 1.0 as catalog implied):**

Strong wall-clock evidence via `tests/bench_uniform.cu` OP=0 (UIADD3 chain):

```
CUDA_VISIBLE_DEVICES=0 ncu --metrics sm__inst_executed_pipe_uniform.avg.per_cycle_active \
  ./QuickRunCUDA -f tests/bench_uniform.cu -t 128 -b 1184 -A 1024 -B 1024 -C 4096 \
  -H "#define OP 0
#define UNROLL 16" -0 8192 -T 1
```

| Test | pipe_uniform inst/SM/cy | % of peak (ncu) |
|------|------------------------:|----------------:|
| bench_uniform OP=0 (UIADD3 chain) | **1.94** | **97%** |
| bench_uniform OP=2 (ULOP3 chain) | **1.86** | 93% |
| bench_uniform OP=1 (UFMUL via blockIdx) | 0.37 | 18% |
| bench_uniform OP=3 (mixed lane+uniform) | 0.10 | 5% |
| bench_adu_uniform OP=9 (LDSM) | 0.70 | 35% |

**Both UIADD3 and ULOP3 clearly exceed 1.0 inst/SM/cy** (1.94 and 1.86). Wall-clock 0.021 ms confirms — at 1.92 GHz × 148 SM × 2 inst/cy = 568 G uniform-inst/s chip max; we measured rate consistent with this.

**Architectural peak = 2.0 inst/SM/cy confirmed.**

LDSM at 0.70 (35%) is under-saturated for this test — LDSM has its own bandwidth constraints. UIADD3 / ULOP3 show the true pipe ceiling.

**Catalog "~1.0 warp-inst/SM/cy" was a regime-specific measurement (likely ACTIVEMASK or LDSM, not the pure uniform-int chain). True peak is 2.0.**

## REVIEW_CHECKLIST candidates

- [ ] §6 catalog "~1.0 warp-inst/SM/cy" — actual peak is 2.0; LDSM hits 0.70 (35%) at full occupancy — could be higher with more ILP. Original "~1.0" claim is below true peak. — `[regime-narrow]`
- [ ] §6 "uniform ops issue in parallel with vector ops from same SMSP" — not independently re-tested here; needs co-issue test
- [ ] §6 UFFMA / UFADD / UFMUL "full uniform FP32 datapath" — not tested; compiler emission rare in observed SASS
