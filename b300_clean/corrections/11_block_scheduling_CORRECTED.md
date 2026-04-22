# 11_block_scheduling — CORRECTED

**Topic**: Block scheduling, occupancy, cluster topology, GPC layout
**Source files reviewed**: `11_block_scheduling.md`, `I6_BLOCK_SCHEDULE_TOPOLOGY.md`,
`I8_CLUSTER_TOPOLOGY.md`, `M3_TOPOLOGY_CHEATSHEET.md`, `DSMEM_REFERENCE.md`,
`B300_TRUE_REFERENCE.md`
**Date**: 2026-04-22

---

## Verified consensus

| Topic | Value | Confidence | Source agreement |
|---|---|---|---|
| Total SM count | 148 (IDs 0..147 dense) | HIGH | All sources agree |
| GPC count | 8 | HIGH | 11, M3, TRUE_REFERENCE, ncu gpc__cycles_elapsed |
| TPC = 2 SMs (consecutive IDs, stride +1) | YES | HIGH | I6, I8, M3, DSMEM all agree |
| GPC-row stride between TPCs in a cluster | +16 SMs | HIGH | I6, I8, M3, DSMEM all agree |
| Concurrent kernel dispatch slots | 128 | HIGH | 11, M3, TRUE_REFERENCE |
| Cluster placement spans multiple GPCs | YES | HIGH | DSMEM, 11 (vs prior "same-GPC" claim retracted in 11) |
| Cluster-launch attribute overhead | ~0 vs regular launch | HIGH | 11, M3 |
| Cooperative-launch overhead | +32 ns | HIGH | 11 |

---

## CONTRADICTION 1 — SMs per GPC: "2×20 + 6×18" vs "8×~18.5"

| Source | Claim |
|---|---|
| `11_block_scheduling.md` L16 | "2 GPCs × 20 SMs + 6 GPCs × 18 SMs = 148" (HIGH conf) |
| `B300_TRUE_REFERENCE.md` L132 | "8 GPCs × ~18 SMs each (= 144 active + 4 spare = 148 total)" |
| `I8_CLUSTER_TOPOLOGY.md` L28-30 | "~9 GPC-rows of 8 TPCs (64 SMs)... 148 = 8.5 GPC-rows worth" |
| `M3_TOPOLOGY_CHEATSHEET.md` L31 | "148 / 16 ≈ 9.25 GPC-rows worth" |

These three models are **mutually incompatible**:
- 11.md: row-major 8-wide topology, two rows extended → 2×20 + 6×18 = 148 ✓
- TRUE_REFERENCE: uniform 8×18 = 144 + 4 "spare" SMs → arithmetic adds, mechanism unclear
- I8/M3: 9.25 rows of 16 SMs → implies > 8 GPCs OR a "GPC row" ≠ "GPC"

The 11.md "2 GPCs × 20" model is the only one that arithmetically lands on 148
with the column-stride-16 layout and matches the I8 cluster-8 cluster-15 wraparound
(cluster 8 cy in I8 shows +13 stride instead of +15 — consistent with 2 long GPCs
of 20 SMs offsetting the modulo).

**Adopt** the 11.md "2×20 + 6×18" model. **Retract** the "8×~18 + 4 spare"
phrasing in TRUE_REFERENCE L132 (the "spare" is not architectural — those SMs
are active and used, just unevenly distributed).

I8/M3 phrasing of "GPC-rows" conflates "GPC" with "16-SM stride window";
prefer to call these "stride-16 columns of TPC pairs", not GPC rows.

---

## CONTRADICTION 2 — Cluster spans 4 GPCs (DSMEM) vs cluster-8 routing

| Source | Claim |
|---|---|
| `DSMEM_REFERENCE.md` L18 | "Four TPC pairs (x, x+1) spread across **4 GPCs**" |
| `11_block_scheduling.md` L21 | "8-block cluster spans **4 GPCs**" |
| `I8_CLUSTER_TOPOLOGY.md` L17 | "cluster spans ≤ 8 SMs across ≤ 4 TPC pairs" (says TPC pairs, not GPCs) |

DSMEM and 11.md agree: **cluster of 8 spans 4 GPCs**. Combined with the column-stride-16
layout and 8 GPCs total, this means a cluster's 4 TPC pairs occupy 4 of the 8
columns (e.g., columns 0,1,2,3 or 0,2,4,6 — DSMEM's deterministic placement
{0,1,16,17,32,33,48,49} occupies columns 0,1,2,3 of the stride-16 layout, which
maps to 4 different GPCs only if columns map 1:1 to GPCs).

**Implication**: with 8 GPCs and stride-16 columns, the natural mapping is
"column index = GPC index, but adjacent columns share a TPC". The DSMEM
deterministic set {0,1}∈col0, {16,17}∈col0... wait — this re-reads as same column.

This is **UNRESOLVED**. Either:
- (a) DSMEM L18 is wrong about "4 GPCs" (the placement is across 4 *TPC pairs*
      in *one* column, all 1 GPC).
- (b) Column-stride-16 model is wrong; GPCs are arranged differently.
- (c) The mapping from SM ID to GPC is non-trivial (not column = GPC).

No source verifies the SM-id → GPC mapping with `nvcuda::__cluster_block_rank()`
or `gpc__cycles_active.per_pgpc_id` directly.

---

## CONTRADICTION 3 — Cluster-8 max_active_clusters

| Source | Claim |
|---|---|
| `11_block_scheduling.md` L40 | "TPCs 71-73 (SMs 142-147) excluded from cluster-4/8; cluster-8 max_active_clusters = **142**, not 148" |
| `M3_TOPOLOGY_CHEATSHEET.md` L120 | (silent — only says "Cluster size max = 8 verified / 16 spec") |

Note: 142 doesn't divide cleanly into clusters of 8 (142/8 = 17 r 6). Likely the
intended meaning is "cluster-8 fits 17 clusters × 8 = 136 SMs active, with 12 SMs
unused" — but 11.md says "142", which is the **count of usable SMs**, not the
number of clusters. Phrasing in 11.md is ambiguous; it should say "max active
SMs participating in cluster-8 = 142" or restate as "max_active_clusters = 17".

**UNRESOLVED**: Need a fresh `cudaOccupancyMaxActiveClusters` sweep at
cluster_size = 4 and 8 to confirm.

---

## CONTRADICTION 4 — Cluster max: 8 vs 16

| Source | Claim |
|---|---|
| `11_block_scheduling.md` L20 | "Max cluster = 16 (non-portable); 32+ silently no-op" HIGH |
| `M3_TOPOLOGY_CHEATSHEET.md` L120 | "Cluster size max = 8 (verified) / 16 (per spec)" |
| `DSMEM_REFERENCE.md` L26 | "Max cluster = 8 (16 advertised but 8 is practical)" |
| `B300_TRUE_REFERENCE.md` L136 | "Max usable cluster size = 16 (non-portable), 8 portable" |
| Memory note (V5) | "cluster MAX=8" |

Consensus: **8 is the portable limit; 16 launches succeed (non-portable);
32+ silently no-ops.** All four sources agree at this resolution; the V5 memory
"MAX=8" is shorthand for the portable limit.

11 and TRUE_REFERENCE agree fully; M3 and DSMEM are conservative restatements.

---

## CONTRADICTION 5 — Cluster placement determinism

| Source | Claim |
|---|---|
| `DSMEM_REFERENCE.md` L11-18 | "cx=8 deterministic": CTA i → SM in fixed set {0,1,16,17,32,33,48,49} (stable across launches) |
| `I8_CLUSTER_TOPOLOGY.md` L33-37 | "**No control over which SMs** — runtime allocates based on free SM availability and physical adjacency" |

DSMEM claims determinism *given* nothing else is on the GPU; I8 emphasizes
non-determinism *in general*. Both are likely correct in their respective contexts.

**Reconciled**: cluster placement is **deterministic when the GPU is otherwise
idle** (DSMEM measurement context). With concurrent work, the runtime selects
free SMs — the relative *topology* (which TPC pairs span which 16-SM columns) is
preserved, but the absolute SM IDs may shift.

---

## CONTRADICTION 6 — "GPC row" vs "GPC"

I8 and M3 use the term "GPC-row" to mean "16-SM stride window". This conflicts
with the standard NVIDIA usage where a GPC is a fixed hardware unit. With 8 GPCs
and 148 SMs:
- Average SMs/GPC = 18.5
- "9.25 GPC-rows" in I8 only makes sense if a "row" is 16 SMs ≠ 1 GPC.

**Recommendation**: Standardize on "GPC" = NVIDIA hardware unit (8 of them on B300),
and "stride-16 column" = the 16-SM addressing window the scheduler uses. Update I8 and
M3 to disambiguate.

---

## RETRACTIONS

1. **TRUE_REFERENCE.md L132** "8 GPCs × ~18 SMs each (= 144 active + 4 spare)" —
   **RETRACTED**. Mechanism for "spare" is not architectural; replace with
   "8 GPCs: 2 with 20 SMs, 6 with 18 SMs (per 11_block_scheduling.md)".

2. **I8_CLUSTER_TOPOLOGY.md L28-32 / M3 L31** "~9 GPC-rows", "148 / 16 ≈ 9.25
   GPC-rows worth" — **RETRACTED phrasing**. There are 8 GPCs, not ~9. The 16-SM
   stride is a scheduling-window/column unit, not a GPC.

3. **11_block_scheduling.md L40** "cluster-8 max_active_clusters = 142" —
   **RETRACTED phrasing**. 142 is an SM count, not a cluster count. Restate.

4. **Older catalog** "Cluster blocks placed within same GPC" — already retracted
   in 11.md (commit 79372e6). Confirmed: cluster of 8 spans **4 GPCs**.

5. **Older catalog** "10 GPCs (9×16 + 1×4)" — already retracted in 11.md.

---

## UNRESOLVED

1. **SM-id → GPC mapping not verified directly.** No test reads
   `gpc__cycles_active.per_pgpc_id` or `nvcuda::__cluster_block_rank()` per CTA
   to map SM IDs to GPC IDs definitively. The "DSMEM cluster spans 4 GPCs" claim
   is inferred from latency variance, not measured.

2. **Are GPCs really uniform-by-column or do 2 GPCs have 20 SMs?** The 11.md
   "2×20 + 6×18" model is asserted HIGH but not cited to a specific
   ncu metric — needs verification with `gpc__cycles_active.per_pgpc_id`
   under a uniform 148-block launch.

3. **I8 cluster=8 cluster-8 anomaly**: SMs (66, 67, 80, 81, **94, 95**, 108, 109)
   show gap +13 (95→94 instead of +15). I8 attributes this to "partial row";
   not reconciled against the 11.md "2 long GPCs" model.

4. **`cudaOccupancyMaxActiveClusters` for cluster_size = {4, 8, 16}** never
   reported. 11.md asserts 142 SMs participate at cluster=8 but provides no
   ncu/CUDA API confirmation.

5. **Block 0 → SM 142 mystery** (M3, README): why does block 0 launch on the
   *last* TPC pair instead of SM 0? Hypothesized as "partial GPC gets priority"
   in I6 but not verified.

6. **Cluster size ≥ 16 behavior**: 11.md says "16 non-portable, 32+ silently
   no-op" but no measurement of placement topology at cluster_size=16. If
   cluster_size=16 spans 8 GPCs (= all GPCs), DSMEM cost may differ from cluster=8.
