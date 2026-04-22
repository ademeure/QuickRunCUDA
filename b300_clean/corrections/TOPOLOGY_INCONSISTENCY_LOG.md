# Topology / Block scheduling — Inconsistency Log

**Reviewed**: 11_block_scheduling.md, I6, I8, M3, DSMEM_REFERENCE.md, B300_TRUE_REFERENCE.md
**Date**: 2026-04-22

| # | Topic | Source A | Source B | Resolution |
|---|---|---|---|---|
| 1 | SMs per GPC distribution | 11.md L16: "2×20 + 6×18 = 148" | TRUE_REFERENCE L132: "8×~18 = 144 + 4 spare" | Adopt 11.md (2×20+6×18); "spare" phrasing in TRUE_REFERENCE is wrong, no such concept |
| 2 | "GPC rows" vocabulary | I8 L28: "~9 GPC-rows of 8 TPCs (64 SMs)"; M3 L31: "148/16 ≈ 9.25 GPC-rows" | All other sources: "8 GPCs" | Standardize: 8 GPCs; "16-SM column" replaces "GPC-row" in I8/M3 |
| 3 | Cluster-8 spans how many GPCs | DSMEM L18: 4 GPCs; 11 L21: 4 GPCs | I8 L17: ≤4 *TPC pairs* (silent on GPCs) | Consensus 4 GPCs; I8 wording is weaker but compatible |
| 4 | Cluster max usable size | 11 L20: 16 non-portable, 32+ no-op; TRUE_REFERENCE L136: same | M3 L120: "8 verified / 16 spec"; DSMEM L26: "8 practical, 16 advertised" | 8 portable, 16 launches but non-portable, ≥32 silent no-op (consensus) |
| 5 | Cluster placement determinism | DSMEM L18: deterministic at cx=8 | I8 L33: "no control over which SMs" | Both correct: deterministic on idle GPU; runtime-selected with concurrency. Topology stable, abs IDs may shift |
| 6 | Max active SMs in cluster-8 | 11 L40: "max_active_clusters = 142, not 148" | M3 L120: silent | Phrasing wrong: 142 is *SMs* not *clusters*. Should say "17 clusters × 8 = 136 SMs participate, ≤142 SMs eligible" — verify with API |
| 7 | "Cluster blocks in same GPC" | Older catalog (pre-79372e6) | 11.md retracted; DSMEM, I8 confirm spread | RETIRED in 11.md; spreads across multiple GPCs |
| 8 | "10 GPCs (9×16 + 1×4)" | Original catalog L7490 | 11.md retracted via ncu gpc__cycles | RETIRED; B300 has 8 GPCs |
| 9 | Block 0 → SM 142 launch order | I6 L18, M3 L42, README L150 | (no contradicting source) | Reproducible but **unexplained**; "partial GPC priority" hypothesis unverified |
| 10 | Cluster=8 cluster-8 stride anomaly (+13 instead of +15) | I8 L20 | (no contradicting source) | Likely artifact of 2 long-GPCs model; not formally reconciled |
| 11 | DSMEM "best pair SM32↔SM33" vs "worst SM16↔SM17" | DSMEM L36-37 | I8/I6: TPCs uniform | Latency varies 25% across pairs even within "same column" — implies SM-id → GPC mapping has more structure than column index |
| 12 | Cluster size verified for cooperative launches | I6 L58: "what would change it: cooperative grids" | (no follow-up done) | OPEN — cooperative launch topology never measured |

## Summary of action items for re-verification

1. Run `cuOccupancyMaxActiveClusters(blockSize, clusterSize)` for {2,4,8,16} and report.
2. Use ncu `gpc__cycles_active.per_pgpc_id` with a 148-block launch to definitively map
   SM-id → GPC-id (settles items 1, 3, 6, 11).
3. Repeat I8 cluster topology dump while measuring GPC ID per CTA (not just SM ID).
4. Test cluster_size=16 placement (item 4).
5. Test cooperative-grid SM mapping (item 12).
6. Investigate why block 0 → SM 142 (item 9): could be a "fill from highest TPC" priority
   to leave low SM IDs free for OS/driver overhead.

## Cross-reference: terminology to standardize

| Term | Definition | Use |
|---|---|---|
| **SM** | Streaming Multiprocessor, ID 0..147, 148 total | universal |
| **TPC** | Texture Processing Cluster, 2 consecutive SMs (id N, N+1) | I6, I8, M3, DSMEM agree |
| **GPC** | Graphics Processing Cluster, **8 total**, sizes 18 or 20 | use "8 GPCs" exclusively |
| **stride-16 column** | scheduler addressing window of 16 SM IDs | replaces "GPC-row" in I8, M3 |
| **cluster** | thread-block cluster, ≤8 portable, ≤16 hard limit | universal |
| **dispatch slot** | concurrent-kernel HW slot, 128 max | 11, M3, TRUE_REFERENCE |
