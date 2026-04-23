# §13 Predication / divergence — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §13 (L957-971)
**Test:** `tests/bench_predication.cu`

## CLAIM (catalog L957-971)

> "Per-thread predication (`@p instr`): zero effect on pipe rate. Measured: `fma.rn.f32` unpredicated = 0.570 ms; same op wrapped in `@p` with only 16/32 lanes active = 0.575 ms; with only 1/32 lanes active = 0.574 ms. The hardware issues the warp-instruction regardless of how many lanes are live — pipe time is the same."

## TEST

```bash
CUDA_VISIBLE_DEVICES=0 ncu --metrics sm__inst_executed_pipe_fma.avg.per_cycle_active,sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active \
  ./QuickRunCUDA -f tests/bench_predication.cu -t 128 -b 1184 -A 1024 -B 1024 -C 4096 \
  -H "#define ACTIVE_MASK 0xFFFFFFFF" -T 1
```

## MEASURED (ncu, GPU 0 only, full occupancy)

| Active lanes | inst/cy/SM | % of pipe_fma peak |
|-------------:|-----------:|-------------------:|
| 32 of 32 (0xFFFFFFFF) | 2.91 | 72.64% |
| 16 of 32 (0x0000FFFF) | 2.94 | 73.43% |
| 1 of 32 (0x00000001) | 2.94 | 73.46% |

## VERDICT

✅ **CONFIRMED** — pipe_fma rate is **independent of active-lane count** (within 1%). Predication does NOT save throughput.

The 73% (vs theoretical 100%) reflects the `if (active)` branch overhead in the test (the branch itself takes pipe slots). The KEY POINT is that the rate is the SAME across all three masks, confirming predication has zero effect on pipe time.

## Implications

- **You cannot save pipe throughput by divergence or partial predication.**
- **Warp specialization** (e.g., `elect.sync` → 1 lane does work) does NOT free up pipe slots for the rest. The warp-inst still consumes its cycle.
- What predication DOES save: register-read traffic, write-back to masked-off lanes, semantic correctness — not throughput.

## REVIEW_CHECKLIST candidates

- [x] §13 predication-zero-effect — ✅ CONFIRMED via ncu pipe_fma rate identical across 32/16/1 lane masks (within 1%)
