# I8: B300 cluster SM topology — TPCs at +16 stride

## Methodology
Each block reads `%smid` and `%cluster_ctarank`, writes SM-ID to a per-cluster slot. Tested cluster sizes 2, 4, 8 with 16 clusters each.

## Measured (raw SM IDs per cluster)
**CSIZE=2** — consecutive SMs (= same TPC):
- cluster 0: SMs (0, 1)
- cluster 1: SMs (2, 3)
- ... (all consecutive pairs)

**CSIZE=4** — 2 TPCs joined, 16 SMs apart:
- cluster 0: SMs (0, 1, 16, 17), diffs (1, 15, 1)
- cluster 8: SMs (32, 33, 48, 49)
- Pattern: SM_pair_0 + SM_pair_16

**CSIZE=8** — 4 TPCs, 16 SMs apart each:
- cluster 0: SMs (0, 1, 16, 17, 32, 33, 48, 49)
- diffs: (1, 15, 1, 15, 1, 15, 1)
- cluster 8: SMs (66, 67, 80, 81, 94, 95, 108, 109) — diff pattern (1, 13, 1, 13, 1, 13, 1)
- cluster 15: SMs (0, 1, 16, 17, 32, 33, 64, 65) — wraps, with gap of +31

## Topology inferred
- **TPC (Texture Processing Cluster) = 2 consecutive SMs** (the +1 spacing)
- **GPC-row stride = 16 SMs** (the +15 / +16 spacing)
- Each cluster spans ≤ 8 SMs across ≤ 4 TPC pairs
- B300 has 148 SMs arranged in:
  - 74 TPCs (2 SMs each)
  - ~9 GPC-rows of 8 TPCs (64 SMs)
  - With some rows partial (148 = 8.5 GPC-rows worth)
- Cluster 15 with CSIZE=8 shows wraparound — physical layout exhausted

## Conclusion
1. **Cluster of 4** spans 2 TPCs from different GPC-rows (good for cross-bandwidth tests)
2. **Cluster of 8** spans 4 TPCs from 4 different GPC-rows
3. **TPCs are stable units** (always same 2 SMs paired)
4. **No control over which SMs** — runtime allocates based on free SM availability and physical adjacency
5. The +16 GPC-row stride suggests 8 TPC pairs per row; the +13 anomaly at cluster 8/CSIZE=8 likely reflects partial row populated (148 not a multiple of 16)

## Practical implications
- For cluster-shared SMEM (DSMEM) bandwidth tests: cluster size 4 gives "TPC + far-row TPC" topology — different from "all-same-TPC"
- DSMEM latency may differ within-TPC vs across-TPC
- Don't expect cluster CTAs to be "physically adjacent" beyond the 2-SM TPC pair

## Confidence: HIGH
Reproducible patterns; consistent with NVIDIA TPC architecture documentation.
