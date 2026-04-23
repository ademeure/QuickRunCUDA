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

⚠ **PARTIALLY VERIFIED:**

- pipe_uniform exists and is exercised by LDSM ✅
- The architectural peak is **2.0 inst/SM/cy** (not 1.0 as catalog claimed)
- LDSM saturates at 35% of pipe_uniform peak in this test — likely could go higher with more LDSM-rich ILP
- ACTIVEMASK was tested but at single-warp; not enough data to confirm "~1.0/SM/cy" claim

**Catalog "~1.0 warp-inst/SM/cy" is HALF the architectural peak (2.0).** Could be a measurement-config artifact (the original test may have run at different occupancy).

## REVIEW_CHECKLIST candidates

- [ ] §6 catalog "~1.0 warp-inst/SM/cy" — actual peak is 2.0; LDSM hits 0.70 (35%) at full occupancy — could be higher with more ILP. Original "~1.0" claim is below true peak. — `[regime-narrow]`
- [ ] §6 "uniform ops issue in parallel with vector ops from same SMSP" — not independently re-tested here; needs co-issue test
- [ ] §6 UFFMA / UFADD / UFMUL "full uniform FP32 datapath" — not tested; compiler emission rare in observed SASS
