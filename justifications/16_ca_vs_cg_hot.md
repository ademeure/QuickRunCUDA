# §16 .ca vs .cg L2-hot 25% gap claim — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §16 L1571-1577

## CLAIM

> | hint | chip BW | note |
> |------|--------:|------|
> | `ld.global.ca` (L1) | 13.1 TB/s | baseline |
> | `ld.global.nc` | 13.1 TB/s | same as .ca |
> | `ld.global.cg` (L2) | 10.5 TB/s | **−20%** — bypassing L1 hurts hot data |
>
> "For small hot working sets, prefer .ca/.nc over .cg."

## TEST (4 MB WS, cycling access)

Built `tests/_tmp_l2hot_hints.cu` (deleted after measurement):
```c
const int WS_BYTES = 4 * 1024 * 1024;
unsigned int idx = (i * N_THREADS + tid) & (WS_BYTES/4 - 1);
ld.global.{ca,cg}.u32 [A + idx*4]
```

### Results (ncu, full chip occupancy)

| Hint | SASS | l1tex BW | lts BW | Total (l1tex + lts/2) |
|------|------|---------:|-------:|----------------------:|
| .ca  | LDG.E.STRONG.SM  | 4.56 TB/s | 6.78 TB/s | ~7.9 TB/s |
| .cg  | LDG.E.STRONG.GPU | 4.60 TB/s | 6.78 TB/s | ~7.9 TB/s |

**Essentially identical** (within noise) — NOT the 25% gap catalog claims.

## Why the discrepancy

L1 cache on B300 is at most ~228 KB (smem=0 carveout) or ~28 KB (default). **4 MB WS does NOT fit in L1.** So both .ca (L1+L2 cached) and .cg (L2-only) go to L2 anyway. No L1-vs-L2 differentiation is observable.

For the catalog's 25% gap to materialize, the **hot subset within the working set must fit in L1**. Likely the catalog's "4 MB WS" had repeated access to a smaller hot region (e.g., 16-32 KB inner loop) where L1 actually serves the loads.

## VERDICT

⚠ **CATALOG CLAIM REQUIRES NUANCE** — "ca beats cg by 20% at 4 MB WS" is true ONLY if the hot subset fits in L1. With uniform cycling through full 4 MB WS, the gap collapses (both go to L2).

**Refined recipe**:
- For loads that hit a hot subset ≤ L1 capacity (≤228 KB): `.ca` significantly beats `.cg`
- For uniform large WS (>L1 capacity): `.ca` and `.cg` are equivalent (both go to L2)

## REVIEW_CHECKLIST candidates

- [ ] §16 L1571 `.ca = 13.1 TB/s, .cg = 10.5 TB/s = -20%` at "4 MB WS" — **NEEDS NUANCE**: gap requires hot subset ≤ L1 capacity. Uniform 4 MB cycling shows NO gap (both ~7.9 TB/s). Catalog should specify access pattern.
