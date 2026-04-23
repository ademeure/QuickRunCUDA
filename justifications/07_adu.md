# §7 ADU (`pipe_adu`) — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §7 (L536-554)
**Cross-reference:** `justifications/11_redux.md` (REDUX.SUM hits ADU at 0.50/cy)
**Test:** `tests/bench_adu_uniform.cu` (existing, OP=0 bar.sync)

## CLAIM (catalog L536-554)

> "ADU hosts the slow warp-wide synchronization and status-register operations.
> **Peak issue rate**: ~0.4–0.5 warp-inst/SM/cy for simple cases (BAR.ARV, REDUX.OR).
> Wall-clock time is dominated by cross-thread waiting rather than the pipe's own throughput.
> **Contention with ALU/FMA: none observed.** ADU ops do not consume alu or fma slots."

SASS opcodes on pipe_adu (per catalog):
- Barriers: BAR, BAR.SYNC, BAR.ARV, BAR.RED.{POPC,AND,OR}, B2R, BMOV, DEPBAR, LDGDEPBAR, SYNCS
- CGA barriers: UCGABAR_ARV/WAIT, CGAERRBAR, ACQBULK, ACQSHMINIT
- Warp sync: WARPSYNC, BSYNC, BSSY, BREAK, NANOSLEEP, YIELD
- Match/reduce: MATCH.ANY/ALL, REDUX.{SUM,OR,AND,XOR}
- Fences: MEMBAR.SC.GPU/SYS (partial)

## TEST

```bash
CUDA_VISIBLE_DEVICES=0 ncu --metrics sm__inst_executed_pipe_adu.avg.per_cycle_active,sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active \
  ./QuickRunCUDA -f tests/bench_adu_uniform.cu -t 512 -b 296 -A 1024 -B 1024 -C 4096 \
  -s 8192 -H "#define OP 0" -0 1024 -T 1
```

## MEASURED (ncu, GPU 0, full occupancy)

| Op | pipe_adu inst/SM/cy | % of peak | Catalog claim |
|----|--------------------:|----------:|---------------|
| OP=0 (`bar.sync 0`) | **0.36** | **72.5%** | "~0.4-0.5 simple cases" |
| `redux.sync.add.u32` (from §11 audit) | **0.50** | **100%** | "0.5" |

Implied pipe_adu peak from bar.sync: 0.36 / 0.725 = **0.50 inst/SM/cy** ✅ matches REDUX.SUM saturating at 0.50.

## VERDICT

✅ **CONFIRMED** — pipe_adu peak is **0.50 inst/SM/cy** (consistent across two independent tests):
- REDUX.SUM (§11): 0.50/cy = 100% of peak
- bar.sync 0 (§7): 0.36/cy = 72.5% of peak (under-saturated due to cross-warp wait time)

The architectural peak matches the catalog's "~0.4-0.5" range exactly. bar.sync running at 72% is consistent with catalog's note that "wall-clock time is dominated by cross-thread waiting".

## REVIEW_CHECKLIST candidates

- [x] §7 pipe_adu cap = 0.50 inst/SM/cy — ✅ confirmed (REDUX.SUM at exactly 0.50, bar.sync at 0.36 = 72% of cap)
- [ ] §7 "no contention with ALU/FMA" — not independently re-tested; would need co-issue test
- [ ] §7 nanosleep "0.25/cy = 8/SM/cy" (per §14 L988) — not separately verified
- [ ] §7 MATCH.ANY "serial, slow" — not benchmarked (catalog calls it slow without exact rate)
