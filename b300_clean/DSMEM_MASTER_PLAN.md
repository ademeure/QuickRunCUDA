# DSMEM Master Plan — Comprehensive Re-investigation

## State of knowledge after sub-agent survey

### VALID claims (DCE-immune, clock64-based)
| Claim | Value | Source | Kernel pattern |
|-------|-------|--------|----------------|
| DSMEM latency cluster=4/8 | 201 cy (105 ns) | `investigations/04_dsmem_overhead.md` (2026-04-17) | Dep pointer chain, 50×30 iters |
| DSMEM latency cluster=2 | 224 cy (117 ns) | same | Dep pointer chain |
| DSMEM/local ratio | 7.2–8.0× | same | Matched methodology |
| DSMEM tput ILP=4 | 63.5 cy/load | same | 4 parallel chains |
| cluster.barrier | ~390 cy O(1) | M7_V5 (1f193aa) | Bare barrier |
| V5 producer-consumer round-trip | 468 cy (312 ns) | M7_V5 (ece9976) | Two-phase + barrier |

### REJECTED claims (DCE or LICM artifacts)
- All TB/s bandwidth numbers: V8 37, V10 48/67 — all static-offset DCE'd
- V10 writes "4× slower than reads" — both DCE'd
- Early "0.8% overhead" — LICM
- Early "99% of local" — broken test

### CONTRADICTIONS TO RESOLVE

1. **Latency ratio conflict**: 4.7× vs 7-8× vs V5's 3.96× — same quantity, 3 different answers in history. Root cause of each:
   - 4.7× from early `dsmem_v2.cu` conflated throughput with latency
   - V5 3.96× had wrong local baseline (54 cy vs 28 cy actual)
   - 7-8× from proper pointer chain — this is the answer, but V5 era gave 3.96×. Why?

2. **Cluster size effect on latency**: cluster=2 gives 224 cy vs cluster=4/8 at 201 cy — 11% slower. Is this real HW or test artifact? Need to isolate.

3. **Driver crash**: varying-address DSMEM crashes with unspec launch failure. Whole class of measurements blocked. Root cause unknown.

4. **What IS the real DSMEM BW?**: all BW numbers rejected. No DCE-immune BW measurement has been made. Gap in the catalog.

### EXHAUSTIVE LIST — further DSMEM things to characterize

#### Fundamentals (latency-ladder)
- [ ] DSMEM read latency across cluster sizes {2,3,4,5,6,7,8} — fine sweep
- [ ] DSMEM read latency vs peer distance (adjacent vs far peer in cluster)
- [ ] DSMEM write latency (chained via CAS or via load-after-write fence)
- [ ] DSMEM atomic latency per op type (add/min/CAS, similar to V10 atomic_ops)
- [ ] DSMEM latency under CTA barrier pressure (barrier in flight)
- [ ] DSMEM latency vs local SMEM bank conflict impact
- [ ] Write-through vs write-back semantics via producer/consumer test

#### Throughput (the big unresolved gap)
- [ ] **True DSMEM BW measurement** using dep-chain that survives DCE AND doesn't crash
- [ ] BW vs cluster size {2,4,8} with verified wavefront counts
- [ ] BW vs ILP (independent chains) sweep
- [ ] BW with all-to-all (CTA i reads from all N-1 peers) — what's aggregate?
- [ ] BW with ring pattern vs all-to-all
- [ ] BW vs access stride within peer SMEM (bank conflicts across cluster?)
- [ ] BW: read-only vs write-only vs mixed
- [ ] BW saturation point: how many in-flight loads before peer SMEM bank-limited?

#### HW/routing insights
- [ ] Does peer distance in cluster affect BW? (adjacent SM may be faster than far)
- [ ] SM-to-SM interconnect topology: which SMs get grouped into clusters?
- [ ] DSMEM bank conflicts: if peer has conflicts, does it backpressure to requester?
- [ ] DSMEM + local SMEM simultaneously: does peer serving local reads slow remote reads?
- [ ] Does DSMEM routing go through L2 or dedicated interconnect? (ncu metric investigation)

#### Primitive interactions
- [ ] DSMEM + mbarrier async completion (if supported)
- [ ] DSMEM + cp.async.bulk.shared::cluster (multicast TMA) — real cluster-async
- [ ] DSMEM inside HMMA tile-prep (cuTLASS-style)
- [ ] DSMEM broadcast pattern: cluster.wide atomic or single CTA writes + all read
- [ ] DSMEM all-reduce pattern: fastest algorithm (tree, butterfly, recursive-doubling)
- [ ] DSMEM + __ballot/redux.sync composition

#### Semantics/correctness
- [ ] DSMEM + fence_block cost (do writes need fence to be visible to peer?)
- [ ] Producer writes to DSMEM + consumer reads: what fence is required?
- [ ] DSMEM atomic scope: .cta vs .cluster vs .gpu differences
- [ ] Cross-cluster DSMEM (cluster 0 CTA reading cluster 1 CTA) — impossible but verify error

#### Real workload
- [ ] Cluster-scale softmax using DSMEM to pool max/sum
- [ ] Cluster-scale GEMM tile sharing via DSMEM
- [ ] Cluster all-reduce timing vs cooperative grid.sync
- [ ] DSMEM vs global memory for stencil patterns (4-8 neighbors)

#### Edge cases
- [ ] DSMEM address alignment requirements (4/8/16 B?)
- [ ] DSMEM with __shfl-style cross-lane after (does shfl see updated value?)
- [ ] DSMEM under register pressure (spill + DSMEM concurrent)
- [ ] DSMEM with cluster membership 0 (self-read via mapa)

## Plan for this session

Step 1: Resolve contradictions #1 and #2 (latency ratio + cluster size effect) via ONE clean kernel with rigor.

Step 2: Attempt #3 (driver crash debug).

Step 3: If #3 tractable → fix, then do #4 (real BW measurement).

Step 4: Move to "exhaustive list" items in priority order.
