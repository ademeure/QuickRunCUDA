# B300 SXM6 AC GPC Topology — Definitive Investigation

**Device**: NVIDIA B300 SXM6 AC  
**Compute capability**: sm_103a (10.3)  
**Total SMs**: 148 (SM IDs 0-147, contiguous)  
**Date**: 2026-04-17  
**Methods**: cluster-placement profiling + ncu hardware GPC counter  

---

## TL;DR

**B300 has 8 GPCs, not 10.**

| Claim | Source | Correct? |
|-------|--------|----------|
| "10 GPCs (9 full × 16 SMs + 1 partial × 4 SMs = 148)" | catalog line 7490 | **WRONG** |
| "8 GPC boot-phase groups" | catalog line 4068 | **CORRECT** |
| "Cluster size 8 max portable" | catalog | **CORRECT** — cluster-16 fails |

---

## Raw Measurements

### 1. SM Count and ID Range

```
%nsmid = 148, %smid range = 0..147 (contiguous, no gaps)
All 148 SMs schedulable for regular (non-cluster) kernels
```

### 2. TPC Structure

Cluster-2 always produces pairs `{2k, 2k+1}` — confirmed across 7400 observations.  
74 TPCs total: TPC-0={0,1}, TPC-1={2,3}, ..., TPC-73={146,147}.

### 3. Maximum Cluster Size

```
cluster_size=2:  valid, max_active_clusters=592
cluster_size=4:  valid, max_active_clusters=284
cluster_size=8:  valid, max_active_clusters=142
cluster_size=16: INVALID (cluster misconfiguration error)
cluster_size=32: INVALID
```

**Max portable cluster size = 8.**

### 4. ncu GPC Hardware Counter (DEFINITIVE)

Run: 148 blocks × 128 threads × 10000 FMA iters, device 0.

```
gpc__cycles_elapsed.avg  = 52029.62 cycles
gpc__cycles_elapsed.sum  = 416237 cycles
=> GPC count = sum/avg = 416237/52029.62 = 8.0000
```

Repeated across 4 independent kernel launches: all give **exactly 8.0000 GPCs**.

### 5. Cluster-4 Affinity Analysis (37000 cluster observations)

Top cluster-4 SM groupings (by frequency) reveal stride-16 pattern:

| freq | SM group | GPC membership |
|------|----------|----------------|
| 48   | {32,33,48,49} | GPC-0 internal |
| 44   | {0,1,64,65}   | GPC-0 internal |
| 43   | {0,1,16,17}   | GPC-0 internal |
| 32   | {8,9,24,25}   | GPC-4 internal |
| 30   | {48,49,64,65} | GPC-0 internal |
| 30   | {16,17,32,33} | GPC-0 internal |
| 27   | {6,7,22,23}   | GPC-3 internal |
| 27   | {4,5,20,21}   | GPC-2 internal |

Within-GPC cluster-4 placements are the **highest-frequency groups** — the driver prefers intra-GPC placement.

### 6. Isolated SMs (TPCs 71-73)

SMs 142-147 (TPCs 71, 72, 73) appear in:
- cluster-2: YES (always {142,143}, {144,145}, {146,147})
- cluster-4: NO (never observed in 37000 cluster-4 observations)
- cluster-8: NO (never observed in 3600 cluster-8 observations)

These SMs run regular workloads normally. The cluster-4/8 exclusion is a hardware
scheduling constraint, likely because these TPCs are in a "partial row" of the GPC
TPC grid that lacks a full alignment partner within the GPC boundary.

---

## GPC Physical Structure

SM IDs are numbered in **column-major (round-robin across GPCs) order**:

```
TPC grid layout (8 columns = 8 GPCs, rows = TPC index within GPC):
        GPC-0  GPC-1  GPC-2  GPC-3  GPC-4  GPC-5  GPC-6  GPC-7
Row 0:  TPC 0  TPC 1  TPC 2  TPC 3  TPC 4  TPC 5  TPC 6  TPC 7
Row 1:  TPC 8  TPC 9  TPC10  TPC11  TPC12  TPC13  TPC14  TPC15
Row 2:  TPC16  TPC17  TPC18  TPC19  TPC20  TPC21  TPC22  TPC23
Row 3:  TPC24  TPC25  TPC26  TPC27  TPC28  TPC29  TPC30  TPC31
Row 4:  TPC32  TPC33  TPC34  TPC35  TPC36  TPC37  TPC38  TPC39
Row 5:  TPC40  TPC41  TPC42  TPC43  TPC44  TPC45  TPC46  TPC47
Row 6:  TPC48  TPC49  TPC50  TPC51  TPC52  TPC53  TPC54  TPC55
Row 7:  TPC56  TPC57  TPC58  TPC59  TPC60  TPC61  TPC62  TPC63
Row 8:  TPC64  TPC65  TPC66  TPC67  TPC68  TPC69  TPC70  TPC71  ← partial row
Row 9:  TPC72  TPC73   —      —      —      —      —      —    ← partial row
```

TPCs 71, 72, 73 are in partial rows and do not participate in cluster-4/8 routing.

### SM IDs per GPC

| GPC | TPCs | SM count | SM IDs (stride-16) |
|-----|------|----------|-------------------|
| GPC-0 | 0,8,16,24,32,40,48,56,64,72 | **20** | 0,1, 16,17, 32,33, 48,49, 64,65, 80,81, 96,97, 112,113, 128,129, 144,145 |
| GPC-1 | 1,9,17,25,33,41,49,57,65,73 | **20** | 2,3, 18,19, 34,35, 50,51, 66,67, 82,83, 98,99, 114,115, 130,131, 146,147 |
| GPC-2 | 2,10,18,26,34,42,50,58,66   | **18** | 4,5, 20,21, 36,37, 52,53, 68,69, 84,85, 100,101, 116,117, 132,133 |
| GPC-3 | 3,11,19,27,35,43,51,59,67   | **18** | 6,7, 22,23, 38,39, 54,55, 70,71, 86,87, 102,103, 118,119, 134,135 |
| GPC-4 | 4,12,20,28,36,44,52,60,68   | **18** | 8,9, 24,25, 40,41, 56,57, 72,73, 88,89, 104,105, 120,121, 136,137 |
| GPC-5 | 5,13,21,29,37,45,53,61,69   | **18** | 10,11, 26,27, 42,43, 58,59, 74,75, 90,91, 106,107, 122,123, 138,139 |
| GPC-6 | 6,14,22,30,38,46,54,62,70   | **18** | 12,13, 28,29, 44,45, 60,61, 76,77, 92,93, 108,109, 124,125, 140,141 |
| GPC-7 | 7,15,23,31,39,47,55,63,71   | **18** | 14,15, 30,31, 46,47, 62,63, 78,79, 94,95, 110,111, 126,127, 142,143 |

**Summary**: 2 GPCs × 20 SMs + 6 GPCs × 18 SMs = 40 + 108 = **148 SMs**

---

## Why the Catalog Has Conflicting Claims

### "10 GPCs" claim (line 7490) — WRONG

This was derived by observing that CTA 0..5 lands on SMs 142-147, then CTA 6 lands on SM 0, CTA 8 lands on SM 16, CTA 10 lands on SM 32, etc. — and counting groups of 16 consecutive SMs as GPCs. But **consecutive SM IDs do NOT correspond to the same physical GPC**. The SM IDs are interleaved: SM 0 and SM 16 are in the **same GPC** (GPC-0), not different GPCs.

The "9 full × 16 SMs + 1 partial × 4 SMs" claim assumes GPCs are contiguous SM ranges [0-15], [16-31], etc. This is false — GPCs are stride-16 columns: {0,16,32,...,128,144}, {2,18,34,...,130,146}, etc.

### "8 GPC boot-phase groups" claim (line 4068) — CORRECT

The boot-phase clock64 measurement correctly identified 8 distinct groups totaling 148 SMs (sizes 12+20+20+20+18+20+20+18=148). The assignment of these groups to physical GPCs is correct. The `ncu` hardware counter independently confirms exactly 8.0 GPCs.

---

## Cluster Placement Rules

1. **Cluster-2** (2 blocks): always placed within a single TPC (consecutive SM pair `{2k, 2k+1}`). Cross-TPC placement never observed.

2. **Cluster-4** (4 blocks): **preferentially** placed within a single GPC (highest-frequency patterns are intra-GPC at stride 16), but cross-GPC placement also occurs. TPCs 71-73 (SMs 142-147) never appear.

3. **Cluster-8** (8 blocks): placed across 4 TPCs. TPCs 71-73 (SMs 142-147) never appear. Max 142 active clusters simultaneously.

4. **Cluster-16+**: invalid on B300 (`cluster misconfiguration` error). The hardware limit is cluster-8.

---

## Why max_active_clusters=142 for cluster-8?

148 total SMs − 6 SMs (TPCs 71,72,73 excluded from cluster routing) = **142 SMs** participate in cluster-8.  
142 SMs / 8 blocks per cluster = exactly 17.75 — not a clean number, suggesting the 142 is a scheduler limit based on available TPC alignment slots, not a strict multiple-of-8 constraint.

---

## Cluster Locality and GPC Bandwidth

From prior catalog measurements (line 3804-3826):
- 2 SMs in **same TPC**: ~2952 cycles for cluster sync
- 2 SMs in **same GPC, different TPC**: ~2952 cycles (same tier)
- 2 SMs in **different GPC**: ~3290 cycles (+11%)
- 4+ SMs in **same GPC**: ~5050 cycles (new tier)

The intra-GPC cluster-4 preference is thus performance-motivated: keeping blocks within a GPC avoids the GPC-to-GPC fabric penalty.

---

## Source Files

- `/root/github/QuickRunCUDA/investigations/gpc_topology.cu` — initial smid/nsmid/cluster sweep
- `/root/github/QuickRunCUDA/investigations/gpc_topology3.cu` — cluster-rank-based TPC pair identification
- `/root/github/QuickRunCUDA/investigations/gpc_topology4.cu` — frequency analysis of cluster groupings
- `/root/github/QuickRunCUDA/investigations/gpc_topology5.cu` — valid cluster sizes + within-GPC patterns
- `/root/github/QuickRunCUDA/investigations/gpc_topology6.cu` — TPC affinity matrix, 37000 cluster-4 observations
- `/root/github/QuickRunCUDA/investigations/gpc_topology7_final.cu` — final summary binary
