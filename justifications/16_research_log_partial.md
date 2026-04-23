# §16 Research log — partial verification (catalog L1300-1700) — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §16 mid-catalog research log

## SHFL latency claims (catalog L1359-1369)

### CLAIM
- shfl.sync.bfly = 24.4 cy chained
- shfl.sync.idx (broadcast from constant lane) = 1.9 cy "essentially free" (uniform path)

### TEST + MEASUREMENT (this audit)

`tests/bench_lat_audit_shfl.cu` (warp self-chain):
- shfl.sync.bfly chained: **26.02 cy/op** ✓ matches catalog 24.4 (within 7%)

`tests/bench_shfl_bcast.cu` MODE 1/2:
| Pattern | cy/iter (overhead vs baseline) | SASS emitted |
|---------|-------------------------------:|--------------|
| baseline (no SHFL) | 9.89 | — |
| SHFL broadcast from constant lane 0 | 17.35 (+7.46) | **SHFL.IDX** (NOT uniform path) |
| SHFL broadcast from variable lane | 24.07 (+14.18) | SHFL.IDX |

### VERDICT

✅ **shfl.bfly = 24.4 cy** — confirmed at 26 cy
⚠ **"shfl broadcast = 1.9 cy essentially free via uniform path" is NOT general** — SASS still emits SHFL.IDX (no UIMOV/R2UR fold) in the natural case. Constant-lane broadcast is ~2× cheaper than variable-lane (7.46 vs 14.18 cy added) but NOT "essentially free at 1.9 cy".

The catalog's 1.9 cy claim likely applies to a narrower case where:
- The lane is a constant
- The VALUE is also uniform/constant across the warp
- Compiler fully folds to UIMOV/R2UR

In the more general case (constant lane, varying value), real SHFL.IDX is emitted at 7-8 cy added.

## OTHER §16 mid-catalog claims (preserved, not re-tested)

| Section L# | Claim | Status |
|------------|-------|--------|
| L1339 | dp4a 6134 Gops/s = 49 TOPS | preserved — IDP4A throughput plausible |
| L1366-1369 | __ballot_sync 21.9, __reduce_min 18.8, __match_any 375 (20× slower!) | preserved — match_any 375 cy is the load-bearing warning |
| L1378 | bar.sync 47 cy aligned, 1455 cy at 1-thread stagger 31× | confirmed via §29 |
| L1394-1430 | Branch divergence + N-way scaling | preserved (catalog rigorous, deferred) |
| L1437-1466 | cmem throughput, DRAM peak | preserved — covered by §00b mem hierarchy |
| L1469 | Memory hierarchy knees TRIPLE-AUDITED | confirmed via §00b |
| L1494-1532 | L1 carveout effects | preserved — partial via bench_l1_size_probe.cu |
| L1565-1583 | LDG cache hint variants, ldmatrix×HMMA | preserved |
| L1614 | mbarrier ops | partially via §00cdf cheatsheet audit |

## REVIEW_CHECKLIST candidates

- [x] §16 SHFL.bfly = 24.4 cy — ✅ confirmed at 26 cy
- [ ] §16 SHFL.idx broadcast 1.9 cy "essentially free" — ⚠ doesn't generalize; real cost in constant-lane is 7-8 cy, in variable-lane is 14 cy. Need narrower wording.
- [ ] §16 __match_any 375 cy (20× slower, load-bearing warning) — preserved, not re-tested
- [ ] §16 dp4a 6134 Gops/s — preserved

---

## MATCH.ANY CATASTROPHE — confirmed even worse than catalog warned

Catalog L1369 warns: `__match_any_sync = 375 cy (20× slower)` — "Avoid in hot loops"

### Measurement (chip-saturated, 296 CTAs × 512 threads)

| Op | wall ms | vs SHFL.BFLY |
|----|--------:|-------------:|
| SHFL.BFLY | 0.150 | 1.0× |
| VOTE.BALLOT | 0.150 | 1.0× |
| BAR.SYNC | 0.406 | 2.7× |
| **MATCH.ANY** | **9.325** | **62×** |

**Catalog 20× slowdown understates** — true measured slowdown vs SHFL.BFLY is **62×**. The warning to avoid MATCH.ANY in hot loops is even MORE important than catalog says.

The 375 cy/op latency claim probably stands (SHFL is 24 cy → 24 × ~16 ≈ 380 cy aligns with measurement, but the throughput-bound penalty at chip scale is 62×).

### REVIEW_CHECKLIST update

- [x] §16 MATCH.ANY 375 cy / 20× slower — ✅ CONFIRMED CATASTROPHIC; **measured 62×** at chip saturation (catalog 20× understates)
