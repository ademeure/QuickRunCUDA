# DSMEM Inconsistency Log — pairwise comparison across files

GPU: B300 SXM6 sm_103a. Audit date 2026-04-22.

Truth column = current (post V11–V31) consensus from
`b300_clean/corrections/DSMEM_CORRECTED.md`.

## A. Bandwidth (peak read)

| file_a | claim_a | file_b | claim_b | who's right | why |
|---|---|---|---|---|---|
| V8_DSMEM_BW.md | DSMEM read = **37.28 TB/s** @ cluster=8 (97% of 38.5 TB/s peak) | DSMEM_REFERENCE.md / V21 | aggregate read = **40 GB/s** per cluster (4w × ILP=4 ring) | **DSMEM_REFERENCE.md** | V8 used compile-time invariant offsets → ptxas LICM/CSE'd nearly all loads (DCE confirmed in V10_DSMEM_STATE.md). SASS showed `LD.E` but ncu wavefront count was 7200 vs expected 5.9 B. The 37 TB/s is loop overhead, not BW. Off by ~1000×. |
| V10_DSMEM_DEEP.md | cluster=2 = **48.5 TB/s**, cluster=8 NL=16 = **42.3 TB/s** | DSMEM_REFERENCE.md | ~40 GB/s aggregate read | **DSMEM_REFERENCE.md** | Same DCE pattern as V8. Author retracted in V10_DSMEM_STATE.md. |
| 02_shmem.md (DSMEM section, headline 1035 GB/s mention) | "DSMEM = 1035 GB/s remote" (cited from b478bb0) | DSMEM_REFERENCE.md | ~40 GB/s read aggregate | **DSMEM_REFERENCE.md** | 02_shmem itself flags the 1035 number as "uncertain, workload-specific". Was never a hardware peak. |

## B. Bandwidth (peak write)

| file_a | claim_a | file_b | claim_b | who's right | why |
|---|---|---|---|---|---|
| V10_DSMEM_WRITES.md | DSMEM writes = **11.8 TB/s @ cluster=2**, "writes 4–5× SLOWER than reads" | DSMEM_REFERENCE.md / V21 | writes = **560 GB/s aggregate per cluster, ~13× FASTER than reads** | **DSMEM_REFERENCE.md** | V10 writes also DCE'd (admitted in V10_DSMEM_STATE.md). The direction was inverted — writes are actually the strong side of DSMEM. |
| V10_DSMEM_WRITES.md (recommendation) | "Use producer = local, consumer reads peer" | corrections | "Prefer write-based DSMEM patterns" | **corrections** | Practical advice was inverted. Writes are async/posted; reads pay full LD.E latency. |

## C. Latency ratio (DSMEM vs local SMEM)

| file_a | claim_a | file_b | claim_b | who's right | why |
|---|---|---|---|---|---|
| B300_PIPE_CATALOG §30.H (commit b478bb0) | DSMEM **0.8% slower** than local SMEM | 02_shmem.md / DSMEM_REFERENCE.md | DSMEM **~7.5× slower** | **DSMEM_REFERENCE.md** | LICM hoisted the load out of the loop; test measured loop overhead. Sass dump in `sass/bench_dsmem_*.sass` confirms. |
| `tests/dsmem_v2.cu` | DSMEM **4.7×** slower | DSMEM_REFERENCE.md | DSMEM **~7.5×** slower | **DSMEM_REFERENCE.md** | dsmem_v2 used FADD-serialized accumulator → wrong baseline; not really comparing latencies. |
| 04_dsmem_overhead.md / DSMEM_MASTER_PLAN | "ratio 7.2–8.0×" | DSMEM_REFERENCE.md | "~7.5×" | **both consistent** | Same value; resolves earlier 0.8% vs 4.7× contradictions. |
| DSMEM_MASTER_PLAN | references V5 era ratio of 3.96× | DSMEM_REFERENCE.md | ~7.5× | **DSMEM_REFERENCE.md** | V5 used wrong local baseline (54 cy "self via mapa" mis-attributed as local; real local is 24 cy). |

## D. Per-pair latency asymmetry

| file_a | claim_a | file_b | claim_b | who's right | why |
|---|---|---|---|---|---|
| 04_dsmem_overhead.md | DSMEM cluster=4/8 = **201 cy uniform** | DSMEM_FINDINGS_V2 (V15/V16) | per-pair ranges **164.8 cy (SM32↔33) to 204.97 cy (SM16↔17)**, 25% spread | **DSMEM_FINDINGS_V2** | V15 measured an 8×8 matrix; the original 201 cy was a coarse average. Pairs really do differ by 25%. |
| DSMEM_REFERENCE / V20 | writes pair-uniform (3% spread) | reads pair-dependent (25%) | — | **consistent within DSMEM_REFERENCE** | Reads vs writes use different routing; this is the new architectural insight. |

## E. Cluster-size effect on latency

| file_a | claim_a | file_b | claim_b | who's right | why |
|---|---|---|---|---|---|
| 04_dsmem_overhead.md | cluster=2 = 224 cy, cluster=4/8 = 201 cy → **11% slower at cx=2** | DSMEM_FINDINGS_V2 (V12) | cluster=2 = 214.75 cy, cluster≥3 = ~177 cy → **21% slower at cx=2** | **DSMEM_FINDINGS_V2** (newer fine sweep) | V12 measured cx=3 separately and found the cliff is between cx=2 and cx=3 (single-GPC vs multi-GPC routing). 21% is correct. |
| 02_shmem.md OPEN Q | "Why does cluster=2 pay more?" listed as unresolved | corrections | answered: TPC/GPC routing topology | **corrections** | Resolved by V11–V20 evidence. |

## F. Cluster-size effect on BW

| file_a | claim_a | file_b | claim_b | who's right | why |
|---|---|---|---|---|---|
| V10_DSMEM_DEEP.md | "Use cluster=2 for BW, cluster=8 for capacity" (cluster=2 is 1.5× faster) | DSMEM_REFERENCE.md | BW essentially invariant in cluster size; choose by capacity | **DSMEM_REFERENCE.md** | V10 BW data DCE'd. Real ring throughput at all cluster sizes is similar, capped by per-CTA issue rate. |

## G. Contention model

| file_a | claim_a | file_b | claim_b | who's right | why |
|---|---|---|---|---|---|
| (none earlier) | implicit "shared bus" assumption in older docs | DSMEM_REFERENCE.md | **NO shared bus** — N=1..8 ring readers = 1.00× (flat) | **DSMEM_REFERENCE.md** | V17 m2m confirms dedicated point-to-point routing; contention only when targeting same peer. |
| V19 hotspot text | "per-CTA serving port caps at ~14 GB/s" | DSMEM_REFERENCE.md | "per-CTA serving port caps ~15 GB/s" | **same number, rounding** | 14.0 GB/s is N=8 actual aggregate; 15 GB/s is the architectural ceiling. Both correct. |
| V10/V8-era assumption | reader BW = peer serving rate | V20/V21 | **reader BW = reader's own issue rate** | **V20/V21** | Split-ILP test: 16 ILP→1 peer = 7.11 GB/s vs 4 peers × 2 ILP = 5.92 GB/s. Per-CTA cap is reader-side, not peer-side. |

## H. Fence costs

| file_a | claim_a | file_b | claim_b | who's right | why |
|---|---|---|---|---|---|
| (older catalog) | fence.sc.gpu assumed much more expensive than .cluster | V22 / DSMEM_REFERENCE.md | **fence.sc.cluster == fence.sc.gpu == 320 cy** | **DSMEM_REFERENCE.md** | New fine measurement; only fence.sc.sys (2870 cy) is expensive. |

## I. Crash threshold (DSMEM dependent chains)

| file_a | claim_a | file_b | claim_b | who's right | why |
|---|---|---|---|---|---|
| 04_dsmem_overhead.md / 02_shmem.md | "100% crash at cluster=4 CHAIN_LEN≥15" | V12 / V26 | **30/30 success at CL=50 cluster=4** | **V12/V26** (newer environment) | Likely was driver state / thermal / background procs at original test time. Treat the workaround as historical. |

## J. TMA multicast as DSMEM alternative

| file_a | claim_a | file_b | claim_b | who's right | why |
|---|---|---|---|---|---|
| (no earlier coverage) | n/a | V28 / DSMEM_REFERENCE.md | **TMA multicast = 470 GB/s effective** at 32 KB tile | **DSMEM_REFERENCE.md** | New finding; TMA multicast is the right primitive for cluster data movement, not raw DSMEM reads. |

## K. Hot-spot atomics

| file_a | claim_a | file_b | claim_b | who's right | why |
|---|---|---|---|---|---|
| (extrapolated from hot-spot reads) | implicit "atomics inherit ~15 GB/s cap" | V31 / DSMEM_REFERENCE.md §9.5 | **atomics scale linearly to 63 Matom/s @ N=8** (dest atomic unit pipelined at 33 atoms/clock) | **DSMEM_REFERENCE.md** | Different mechanism: atomic unit is pipelined, the 15 GB/s "serving port" cap doesn't apply. |

## L. TMA + DSMEM concurrency

| file_a | claim_a | file_b | claim_b | who's right | why |
|---|---|---|---|---|---|
| (no earlier coverage) | implicit "TMA and DSMEM share L2 path so contend" | V31 | **TMA + DSMEM = 0.04% interference** (independent paths) | **V31** | New finding; TMA uses cp.async path, DSMEM uses LD.E path, but they don't measurably contend. |

---

## Summary
- **All TB/s DSMEM BW numbers in V8 / V10** are DCE artifacts → retracted.
- **V10's "writes 4× slower"** had the direction inverted → writes are actually 13× faster aggregate.
- **"DSMEM 0.8% slower" and "4.7× slower"** were both methodology errors → real ratio is 7.5×.
- **"Cluster=2 fastest for BW"** false; cluster=2 is actually 21% slower in latency.
- **V11–V31 introduced**: per-pair asymmetry (25% read spread), pair-uniform writes, no-shared-bus contention model, fence.sc.gpu == .cluster, TMA multicast as preferred primitive, hot-spot atomic linearity, perfect TMA+DSMEM overlap.
- **Crash workarounds** in 02_shmem.md / 04_dsmem_overhead.md are no longer reproducible — treat as historical.
