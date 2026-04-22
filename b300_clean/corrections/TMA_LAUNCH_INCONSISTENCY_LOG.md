# TMA / cp.async / Launch / cudaGraph — Inconsistency Log

Cross-document discrepancies and superseded claims for the swarm review.

## A. TMA read SoL — superseded chain

| File / Section | Claim | Status |
|---|---|---|
| 09_memory_apis.md (no TMA section) | does not mention TMA at all | INCOMPLETE |
| V9_CP_ASYNC_BW.md ladder row | "TMA bulk load ~7.5 TB/s (est) ~95-100%" | ESTIMATE — supersede |
| V32_V40_FINDINGS V33 | TMA single-deep 6.72 TB/s = 92% — "best per-CTA HBM read peak" | SUPERSEDED by V46 |
| **V41_V48_FINDINGS V46** | **TMA 8-deep pipelined = 7.20 TB/s = 98.5%** | **NEW READ SoL — canonical** |

**Action**: every doc that quotes "6.72 TB/s TMA SoL" should be updated to
"7.20 TB/s (V46, 8-deep pipelined)" with V33 retained as "single-deep
baseline 6.72 TB/s."

## B. TMA write SoL — consistent

| File | Claim | Status |
|---|---|---|
| V32_V40 V34 | TMA write 7.17 TB/s = 98% | HIGH |
| V41_V48 V47 | TMA write pipelined 8-deep 6.34 TB/s = 87% | NO BENEFIT |

**Action**: rule "writes already async, don't pipeline" is documented;
no inconsistency.

## C. prefetch.L2 + cp.async — direct contradiction across V6 / V42

| File | Claim | Applies to |
|---|---|---|
| CLAUDE.md memory project_b300_v6_complete | "prefetch.L2 1.58×" | OLD `cp.async` (LDGSTS) |
| V41_V48 V42 | "prefetch.L2 + TMA = 27% SLOWER" | `cp.async.bulk` (TMA) |

**Resolution**: NOT a contradiction — different instructions. But ANY doc
that says "prefetch.L2 helps cp.async.bulk" or "prefetch.L2 helps TMA"
without distinguishing is **WRONG**.

**Search any other catalog file** for "prefetch.L2" + "TMA" or "+ bulk"
co-mentions and flag.

## D. TMA multicast — single engine ceiling

| File | Claim |
|---|---|
| V32_V40 V32 | 14.91 TB/s effective at cluster=8, single-deep |
| V41_V48 V48 | Multicast pipelined 2-deep = 13.96 TB/s — capped, single engine/cluster |

No inconsistency; V48 is the explicit "can't be pipelined" verification.

## E. cudaGraph single-kernel — myth in original 10 catalog

| File | Claim | Status |
|---|---|---|
| 10_launch_overhead.md row "cudaGraphLaunch 1.20 µs, 35% cheaper" | implies graphs always cheaper | MISLEADING |
| V9_GRAPH_LAUNCH | 1-kernel graph = 2.05 µs ≈ direct 2.06 µs (NO speedup) | HIGH (1000-iter avg) |
| CLAUDE.md memory project_b300_v8 | "cudaGraph single = no speedup" | confirms V9 |

**Resolution**: the 1.20 µs row was the CPU-enqueue half only; the full
launch+sync is identical. **Original 10 catalog row should be footnoted**:
"Enqueue-only; sync round-trip equals direct launch (V9)."

## F. cuStreamWriteValue32 — two numbers in two docs

| File | Cost |
|---|---:|
| 10_launch_overhead.md catalog | 2.47 µs |
| CLAUDE.md V7 memory | 0.45 µs ("hidden gem 5-6× faster than kernel") |

**Likely**: 2.47 µs is measured `host-call + complete` round-trip;
0.45 µs is the host-call CPU cost only. Needs explicit reconciliation in
a future investigation.

## G. cudaMemset relative to noop kernel

| File | Claim |
|---|---|
| 09_memory_apis.md | cudaMemset 1.4 µs floor = standard kernel launch |
| TRUE_REFERENCE be28c14 | cudaMemset (4 B) = 1.22 µs vs noop kernel 1.78 µs — **31% faster** |

**Inconsistency**: 09 says "API floor 1.4 µs", TRUE_REFERENCE says 1.22 µs.
Likely measurement noise ± methodology difference; both within "low µs"
ballpark but the headline claim differs. TRUE_REFERENCE is more recent.

## H. cuBLAS in graphs — partial contradiction

| File | Claim |
|---|---|
| 10 catalog §5 | "cuBLAS in graphs gave ZERO speedup; slightly hurts" |
| CLAUDE.md project_b300_pitfalls | "cuBLAS needs cudaGraph" for sustained measurements |

**Resolution**: graph wrap is for measurement isolation (sustained-throughput
locking), not perf speedup. Both true; phrasing in 10 catalog should clarify
"no perf benefit, but useful for sustained-rate measurement."

## I. Persistent-kernel numbers spread

| File | Claim | Source |
|---|---|---|
| TRUE_REFERENCE | "persistent kernel + mapped memory = 4 µs CPU↔GPU" | 584fda6 |
| CLAUDE.md V7 memory | "persistent batched 38 ns/task" | V7 |
| CLAUDE.md V6 memory | "persistent 3.5× cold" | V6 |

These measure different metrics: 4 µs = single CPU↔GPU round-trip;
38 ns = per-task in batched dispatch from a backlog; 3.5× = vs cold launch.
**No inconsistency, just three regimes** — should be tabulated together
somewhere canonical.

## Items needing fresh measurement

1. TMA pipeline depth optimum (V46 used 8 — knee unknown)
2. TMA multicast at cluster=4 vs 8 (only cluster=8 tested)
3. `cp.async.cg` variant (V9 only `.ca`)
4. `cuStreamWriteValue32` host-call vs full-pair reconciliation
5. cudaMemset vs noop kernel — re-measure 1.22 µs vs 1.4 µs gap
6. Persistent-kernel "38 ns/task" V7 number with V8/V9 rigor
7. Confirm V6 I3 prefetch.L2 result was on `cp.async` (not bulk) — prevent
   future misapplication
