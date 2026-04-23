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

---

## ADDENDUM 2026-04-23 — Reproduced with proper L1-fitting WS

Built focused test with WS_KB sweep (16/64/128/196 KB):
```c
const int WS_DWORDS = WS_KB * 256;
const int MASK = WS_DWORDS - 1;
unsigned int idx = (i * 256 + tid) & MASK;
ld.global.{ca,cg}.u32 [A + idx*4]
```

Launch: `-t 256 -b 1184 -A 1048576 -H "#define WS_KB 16 ... 196"`

### Results (l1tex BW = effective load bandwidth)

| WS_KB | .ca l1tex BW | .cg l1tex BW | .ca/.cg ratio |
|------:|-------------:|-------------:|--------------:|
| 16 | **13.13 TB/s** | 6.97 TB/s | **1.88×** |
| 64 | 13.11 TB/s | 6.97 TB/s | 1.88× |
| 128 | 13.10 TB/s | 6.94 TB/s | 1.89× |
| 196 | 13.13 TB/s | 6.98 TB/s | 1.88× |

### Findings

✅ **Catalog .ca = 13.1 TB/s CONFIRMED EXACTLY** (my measurement = 13.10-13.13 TB/s across all WS sizes ≤196 KB)

⚠ **Catalog .cg gap is UNDERSTATED**: catalog says .cg = 10.5 TB/s (-20%); my measurement shows .cg = **6.97 TB/s = -47% from .ca** (i.e., .ca is **1.88× faster**, not 1.25×).

The HW path differentiation is much more dramatic than catalog reports:
- .ca uses LDG.E.STRONG.SM (L1+L2) → L1TEX serves at 13.1 TB/s
- .cg uses LDG.E.STRONG.GPU (L2-only) → L1TEX serves at 6.97 TB/s (HALF the rate)

This is consistent with the L1 path being WIDER than the L2-bypass path on the L1TEX unit.

## Updated VERDICT

✅ **Catalog ".ca beats .cg for hot data" CONFIRMED but UNDERSTATED**:
- True ratio: **1.88× (88% faster)**, not 25%
- Holds across all WS ≤196 KB (L1-fitting)
- Falls apart at WS > L1 (4 MB earlier test showed equal rates)

## REVIEW_CHECKLIST candidates (REVISED)

- [x] §16 .ca = 13.1 TB/s — ✅ CONFIRMED EXACTLY at all WS ≤196 KB
- [ ] §16 .cg = 10.5 TB/s — measured **6.97 TB/s** (-33% lower than catalog); catalog underrepresents the .cg-path speed gap
- [ ] §16 ".cg = -20% from .ca" — actually **-47% (1.88× ratio)** — catalog claim understates the gap by ~3.5×
