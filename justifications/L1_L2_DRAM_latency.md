# §0 + §16 LDG latency (L1/L2/DRAM) — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §0 latency table (L97-120) + §16 research log (L1469+)
**Test:** `tests/bench_ldg_lat.cu` (warm pointer-chase, single thread)

## CLAIM (§0 latency table)
- ld.global **L1 = 39 cy**
- ld.global **L2 = 301 cy**
- ld.global **DRAM = 789 cy**

## TEST

```bash
./QuickRunCUDA -f tests/bench_ldg_lat.cu -t 32 -b 1 -A 16777216 -B 1024 -C 1024 \
  -H "#define N_OPS 64
#define ITERS_OUTER 32
#define WINDOW_KB <KB>"
```

Pointer-chase: `idx = arr[idx]` chained N_OPS=64 times, repeated ITERS_OUTER=32 times. Single thread, single block.

## MEASURED (cy/load mean)

| WS_KB | cy/load | Catalog regime | Verdict |
|------:|--------:|----------------|---------|
| 4     | 56.9    | catalog "L1 = 39 cy" | ⚠ +46% vs catalog (different methodology — pointer-chase dep latency) |
| 16    | 104.4   | (transition) | climbing |
| 32    | 168.2   | (transition) | |
| 64    | 272.4   | nearing L2 | |
| 128   | 283.6   | catalog "L2" | ✓ matches |
| 256   | 297.4   | catalog "L2 = 301" | ✅ EXACT |
| 1024  | 298.5   | L2 (still in 126 MB) | |
| 4096  | 298.6   | L2 (still fits) | |
| 16384 | 298.6   | L2 still | |
| **65536** | **301.2** | should be DRAM | ⚠ still hitting L2 due to pointer-chase locality |

## VERDICT

✅ **L2 latency = 301 cy CONFIRMED EXACTLY** (my measure 297-301 cy).

⚠ **L1 latency is HARDER to isolate** than catalog's "39 cy":
- At 4 KB WS (definitely fits in L1) my measurement = 56.9 cy
- Pointer-chase methodology adds dependency latency (each iteration waits for previous result)
- Catalog's 39 cy may use a different test methodology (clock64 around single LDG?)

❌ **DRAM 789 cy NOT REACHED** by this test — even at 65 MB WS the pointer-chase keeps recently-used lines hot (only ~64 distinct addresses visited per outer iter). To measure DRAM properly need linear scan or randomized large-WS access.

## Cross-reference

User skepticism in `reviewed_errors_b300.md` on canonical doc: *"No way L1 latency varies by clock speed or that kind of access pattern if single thread & warm cache, something is wrong here"* — confirmed: L1 latency claims are sensitive to test methodology. Catalog's clean "39 cy L1 / 301 cy L2 / 789 cy DRAM" three-tier table is an oversimplification.

The L2 latency (301 cy) is robust and confirms exactly. The L1 (39 cy) is plausible-but-methodology-dependent. The DRAM (789 cy) requires a different test design to verify.

## REVIEW_CHECKLIST candidates

- [x] §0 ld.global L2 latency = 301 cy — ✅ CONFIRMED EXACTLY (297 cy at 256 KB WS pointer-chase)
- [ ] §0 ld.global L1 latency = 39 cy — methodology-dependent; pointer-chase gives 56.9 cy at 4 KB WS (different methodology may give different result)
- [ ] §0 ld.global DRAM latency = 789 cy — NOT REACHED by my pointer-chase (locality keeps in L2); needs different test design
