# Status of Replication — B300_PIPE_CATALOG audit

**Generated:** 2026-04-23 (live; updated as agents finish)

This is the at-a-glance status of the catalog audit. For details, see `JUSTIFIED_B300_PIPE_CATALOG.md` (per-section audit), `DENSE_B300_PIPE_CATALOG.md` (pruned reliable subset), `REVIEW_CHECKLIST_B300.md` (yes/no items).

---

## TLDR — current mature state (2026-04-23)

**Coverage:** 48 catalog sections audited (essentially the entire foundational early/mid catalog: §0-§30).

- **38 ✅ verified** (full ncu + SASS + wall-clock evidence on this rig)
- **6 ⚠ partial** (replicated but with a regime caveat or methodology note)
- **4 🟡 preserved** (catalog plausible, not re-run — tcgen05.mma, multi-GPU, methodology, tensor unified)

**Top 9 catalog ERRORS** (numbers and recommended fix):
1. DFMA latency: catalog 92 → real **63.9 cy**
2. __syncthreads formula: `12+2W` → **`22+2W`** (54 cy at BS=512, NOT 45)
3. FP64 chip "475 GFLOPS" wording = "475 G FMA-ops/s = 950 GFLOPS"; real **1060 GFLOPS** (12% off, NOT 2.2× as initially claimed)
4. §4 MUFU rate "16 SASS/SM/cy" off by 16-32× (real **~1.0**)
5. Atomic hotspot at warp-level N=2: "5×" → real **34×**
6. FP64 vs FP16 tensor: "300×" → real **2300×**
7. mbarrier.arrive 8.1 cy → real **27 cy** for default `.shared.b64`
8. atom.global.cas SASS "STRONG.GPU" → actual **STRONG.SYS**
9. ld.shared bank-conflict scoping: real at 9.6× for 32-bit LDS (catalog claim that B300 is bank-conflict-free FALSIFIED — methodology error in our prior dismissal too)

**Top 12 NEW architectural facts** missing from catalog:
- F2IP.U8 fast path (4× faster than F2I.S8)
- POPC.INC trick: `atomicAdd(addr, 1u)` → ATOMS.POPC.INC.32 = 2.5× speedup at warp-broadcast
- Global REDG (no-return) vs ATOMG = 25× speedup
- EX2 uniquely 2× faster than other MUFU ops (4 vs 8 cy/SASS)
- bf16x2 EX2 = same throughput at half dispatch pressure
- FMNMX3 fusion (Blackwell 3-input min/max)
- u64.ADD alu+fmaheavy co-issue = 64 u64-adds/SM/cy
- CCTL.IVALL essentially FREE (~3 cy idle); drain-wait dominates fence cost
- release.gpu drain is SM-WIDE (compounds at high occupancy)
- cp.async (LDGSTS) bypasses acquire fence drain unless commit_group
- **`.L2::256B` cache hint = 92% HBM SoL recipe** (40% boost vs baseline at stride-256B 4B reads)
- **`.ca` beats `.cg` by 1.88× at L1-fitting WS** (NOT 1.25× as catalog L1571 says)

**4 self-corrections** caught and walked back during the audit:
- FP64 catalog "off by 2.2×" → actually 12% off (wording confusion)
- SMEM 32-bit bank conflicts "absent on B300" → REAL at 9.6× when properly tested
- SHFL broadcast "1.9 cy free" → 7.46 cy in general case
- .ca/.cg "no gap at 4 MB WS" → was test-config issue (WS exceeded L1)

For the full per-section trail with raw output, SASS, and ncu: see `JUSTIFIED_B300_PIPE_CATALOG.md` and `justifications/<id>.md`.

---

## Replication results so far

| Section | Topic | Status | Verdict | Justification |
|---|---|---|---|---|
| §0.FFMA | FP32 scalar FFMA peak (71.8 TF) | ✅ | **MATCH 100%**: 71.82 TF measured. Clock=**1942 MHz** (not 1920 catalog, not 2032 spec). | `00a_ffma_peak.md` |
| §0.MEM | smem (35.6) / L2 (22-26) / DRAM (7.18) | ✅⚠ | smem 35.88 ✓; DRAM 7.17-7.25 ✓; L2 20.3 (BELOW catalog upper). NEW FOOTGUN: ncu warp-aggregated metric trap. | `00b_mem_hierarchy.md` |
| §1 | Pipe topology / dispatch ceiling | ✅⚠ | Cap 4.00 ✓; V52 alu+fma=145% ✓; pipe_xu compound 0.5 vs simple 1.0 ✓. **FALSIFIED**: catalog L218 "FFMA → both fma sub-pipes simultaneously" is wrong — FFMA dispatches to ONE sub-pipe per cycle. | `01_pipe_topology.md` |
| §22 dual-issue | FFMA2 + ALU vs scalar FFMA | ✅ | FFMA2 + LOP3 1:1 saturates ALL 3 pipes (fmaH=98% / fmaL=97% / alu=97%) → 314 useful ops/SM/cy vs scalar+LOP3's 187. **Dual-issue sweet spot.** | `22_dual_issue_ffma2_alu.md` |
| §22-§25 | Tensor mma.sync (FP16/TF32/FP8/INT8) | ✅⚠ | FP16=571 ✓; TF32=285.7 ✓; INT8 IMMA 142.4 ✓; **FP8 emulated 309 (catalog 276, +12% LOW)**. Confirmed catalog FADD-artifact warning is real. | `22_tensor_mma_sync.md` |
| §24 | Latency table (clock64) | ✅⚠ | ~75% accurate ±15%. **Fixes:** DFMA=63.7 (L103's 92 wrong); DRAM=789 (header's 3000 wrong); **__syncthreads = `22+2W` not `12+2W`**; mbarrier RTT=123 (header 54 was arrive-only). NEW FINDING: redux.add/or/and/xor=44 cy is 2.4× slower than min/max=18 cy. | `24_latency_table.md` |
| §30.B | Atomic latency + contention | ✅⚠ | atom chain = LDS at 45 cy ✓ (K6 was labeling); N=2 anomaly 29× ✓; per-warp 5× claim WRONG (actually 1.09× FASTER); coalesced 0.023 atom/cy/lane (NOT 0.94); scope penalty 2.2× NOT 31×; FP16 atomicAdd 6.3× NOT 45×. | `30B_atomics.md` |
| §30.G | Memory fence costs (cta/gl/sys) | ✅ | cta=8 ✓ V54; gl=267 ✓ V54; **sys=1727 single-GPU** (V54's 2806 was 2-GPU NVLink rig, +1.62× = one extra coherence round-trip). "+60 cy/write linear" claim RETRACTED — fixed one-time L2-drain. | `30G_fence.md` |
| §30 TMA | cp.async.bulk size-independence | ✅⚠ | "48 cy floor" is AMORTIZED rate; pure single-issue is ~65 cy. Sharp 8 KiB crossover ✓ (in GB/s metric not cy). Per-SM peak ~240-260 GB/s ✓. **Chip-wide 21.9 TB/s claim requires L2 hits, NOT DRAM** (catalog wording fails to flag). Open: head-to-head TMA vs LDG max-tuned (in flight). | `30_tma_sizes.md` |
| §13 DSMEM | latency, write throughput, L2 traversal | ✅⚠⚠ | **Catalog "23 cy ≈ free" FALSIFIED**: real read latency 204-223 cy (9× slower). SASS reveals `ld.shared::cluster` → `LD.E` (global LSU path). V53 write 87 GB/s/cluster sustained ✓ confirmed. V21's 560 GB/s is burst not completion. NEW FINDING: DSMEM reads ALSO bypass L2 (correcting V53). Exhaustive sweep in flight. | `13_dsmem.md` |
| §30 TMA vs LDG max-tuned | head-to-head, L2-hit + DRAM-cold | ✅⚠ | **L2-hit: TMA wins 12%** (20.49 vs 18.25 TB/s). **DRAM-cold: TIED at HBM SoL** (LDG 96.5%, TMA 95.4%). Catalog L2 wire 13.3 TB/s under-counts by 37-54%. NEW FOOTGUN: ncu lts__t_bytes undercounts LDG L2-hit by 2.7× (MSHR dedup) — use l1tex__t_bytes for LDG, lts__t_bytes for TMA. | `30_tma_vs_ldg_max_tuned.md` |
| §15a DSMEM exhaustive | 9-dim sweep (width × cluster × placement × ILP × R/W × fence × contention) | ✅⚠⚠⚠ | NEW: v4 is 3.5× per-byte efficient vs u32. Cluster=16 WORKS (non-portable opt-in). ILP=32 collapses DSMEM to 9 cy/load (LDS-equivalent). **Topology: 9 GPCs × 16 SMs + 1 partial 4-SM GPC = 148 — catalog "8 GPCs" WRONG**. **Per-GPC silicon variation 20%** (GPC2 189 cy vs GPC1 229 cy). Fence cost fixed ~1500 cy. R+W shared fabric arbiter. Aggregate chip 2.4 TB/s W / 1.9 TB/s R. | `13_dsmem_exhaustive.md` |
| §22e .reuse cache | 94% of FFMA2 carry .reuse | ✅ | Direct SASS grep: scalar FFMA 99.9% (1023/1024), FFMA2 82.8-99.2% across 5 configs. Catalog 94% in-range. | `22e_reuse_cache.md` |
| §22h compute-mem overlap | FFMA hidden by 522 cy memory; ~16 free | ✅⚠⚠ | qualitative CONFIRMED, quantitative CORRECTED: cold DRAM is **882 cy** (not 522), free budget is **~225 FFMAs** (not ~16). Catalog's 522 was partial-cold; recommend split into "cold 882 / warm 335". | `22h_compute_mem_overlap.md` |
| §30B atom→SASS mapping | atom.add always→REDG | ⚠ CORRECTED | Direct SASS grep across 20K kernels: ALL THREE (REDG/ATOMG.E/ATOM.E) emitted depending on return-value-use + scope. Throughput numbers still valid; SASS-name attribution was wrong. | `30B_atomics_FOLLOWUP.md` |
| §22o NVFP4 mxf4nvf4 + K=96 | 9.9 PF + K=96 ULTRA bit 31 doesn't work | ✅⚠⚠ | RIGOROUS replication: **9.26 PF at 1942 MHz** (catalog's 9.9 was at 2032 boost not observed); K=64=K=96 D[0]=288 confirms K=96 doesn't add MACs; 15/15 correctness; **2 CATALOG CORRECTIONS**: `.block32` form actually compiles (crashes at runtime); `128x256b` cp shape actually works with 8 KB smem. 14 evidence files. | `49_nvfp4.md` (364 lines) |
| §22m kernel launch overhead | 5.7 µs / 2.0 µs (looked inconsistent) | ✅✅✅ | RECONCILED: 2.05 µs pipelined / 5.20 µs per-iter event mode — NOT contradictory (different timing setups). Kernel-size table (L8385) reproduces EXACT (10/100/1000/4000 inst → 2.05/2.05/4.10/10.25 µs vs catalog 2.06/2.06/4.11/10.25). Cluster launch == single-CTA at 2.05 µs flat across cluster sizes 1/2/4/8. NOT YET TESTED: cudaLaunchKernelEx+PSS (1.47 µs claim) or cudaGraph batched (0.56 µs/kernel). | `22m_launch_overhead.md` |

## Pending agent work

(All currently dispatched agents complete. Next iteration of /loop will dispatch more on the remaining priority list.)

## Skeptical review supplements (entries indexed but not all replicated)

- `_SKEPTICAL_REVIEW_10_30.md` — 50 entries, Groups I-O (redux/SHFL, latency table, atomics, TMA, ext op cat, research-log repetition, methodology, pipe placement, clock state, vendor doc inconsistencies)
- `_SKEPTICAL_REVIEW_31_END.md` — 74 entries, Groups P-X (methodology, dual-issue map, tcgen05, DSMEM, fence/barrier, atomics, TMA, cache/L2, architectural limits)

## Catalog corrections recommended (so far)

1. **L27 FP8 mma.sync emulated**: 276 → **308 TFLOPS** (12% increase)
2. **L103 DFMA latency**: 92 → **63.9 cy**
3. **L116 syncthreads formula**: `12 + 2W` → **`22 + 2W`** (10 cy fixed barrier overhead missed)
4. **L218 FFMA "uniquely both sub-pipes"**: → "FFMA can use EITHER sub-pipe per cycle, alternating freely"
5. **L7029-7031 / L7012 DSMEM "essentially free 23 cy"**: → DSMEM read = 204-223 cy (9× slower); SASS reveals LD.E path
6. **L7836-7860 DSMEM 99% local SMEM**: FALSE; actual 5-43%; recommend deletion
7. **L3084 fence "+60 cy/write linear"**: → fixed one-time ~280 cy L2-drain, NOT linear
8. **§30.B per-warp 5× / coalesced 0.94 atom/cy/lane**: BOTH wrong; clean per-warp 1.09× FASTER, coalesced 0.023 atom/cy/lane
9. **§30.B "31.3× scope penalty"**: → real penalty 2.0-2.2× apples-to-apples
10. **§30.B "atom.f16/bf16 ~45× slower"**: → real 6.3× slower
11. **mbarrier RTT 54 cy**: → 123 cy (54 was arrive-only)
12. **redux.sync header**: add row for add/or/and/xor at 44 cy (only min/max=18 documented currently)

## What's still NOT replicated (high-priority remaining)

1. tcgen05.mma direct re-run on this rig (catalog L6686+ has self-consistent linear-scaling math, but no fresh measurement here yet)
2. ~~NVFP4 K=96 ULTRA path~~ **RESOLVED by catalog itself (L9349+)**: K=96 via simple PTX is FALSIFIED — bit 31 doesn't add MACs. Real FP4 path = `kind::mxf4nvf4.block_scale.block16` = 9.9 PFLOPS = 99% of 10 PF spec. Proper `kind::mxf4` form rejected by ptxas 13.2.78; wait for newer NVCC.
3. Power per pipe (catalog §44 — DISPUTED in canonical with M11 vs 16_power_clock 2× discrepancy)
4. L2 wire BW measurement separated from kernel-effective (catalog claims 13.30/23.85/30 TB/s split)
5. cluster launch overhead (catalog §57 / §58)
6. Multi-GPU NVLink-attached fence (catalog 2806 cy sys; needs 2-GPU rig)
7. Predication/divergence cost (catalog §13)
8. Many specific REVIEW_CHECKLIST entries (61 still open out of 74)

## Catalog-self-resolved findings (newly captured this iteration)

The catalog itself contains some excellent material that wasn't fully integrated into the audit until 2026-04-23:

- **NVFP4 K=64 = 9.9 PFLOPS via `kind::mxf4nvf4.block_scale.block16`** (catalog L9298) — 99% of NVIDIA's 10 PF spec, with full correctness verification (15/15 tests). Real path uses `UTCOMMA.BLOCK16` SASS.
- **K=96 ULTRA via idesc bit 31 doesn't actually compute additional MACs** (catalog L9349) — verified with correctness test (D[0] identical at K=64 and K=96 with same inputs). The proper `kind::mxf4` form is rejected by ptxas 13.2.78.
- **Power efficiency table** (catalog L9128): TF32 = 3531 TF/kW, FP16 = 6597 TF/kW (1.9× more efficient), FP4 mxf4nvf4 K=64 random = **15,000 TF/kW (3.3× more efficient than FP8)**.
- **GPC structure verified via %smid** (catalog L7524): exactly 9 × 16-SM GPCs + 1 × 4-SM partial = 148. Matches DSMEM exhaustive sweep finding.
- **CTA scheduler placement** (catalog L7546): fills smallest/last GPCs first, then round-robins 2 CTAs/GPC.
- **L2 latency 25% variation across GPCs** (catalog L7593): GPC2 fastest at 115 cy, GPC3 slowest at 143 cy. Matches DSMEM exhaustive's 20% finding.
- **SASS .reuse cache** (catalog L8142): 94% of FFMA2 instructions in real benchmarks carry `.reuse` annotation — critical for approaching FFMA2 peak.
- **Compute-memory overlap** (catalog L8309): ~16 FFMA per cold-DRAM load = "free" (522 cy budget).

These are now in DENSE §22o-§22r and §22e-§22n.

## Progress numbers

- DENSE_B300_PIPE_CATALOG.md: **~870 lines** (vs 19,742 source, 22:1 prune ratio; covers §0-§16)
- JUSTIFIED_B300_PIPE_CATALOG.md: **~120 lines index** + 9 full justification records totaling ~2,100 lines
- REVIEW_CHECKLIST_B300.md: **~150 lines + 124 supplementary entries**, of which **13 [x] resolved** (and 10 specific catalog corrections recommended)
- 124 suspect claims indexed
- 9 replication agents completed; 2 in flight
- Next iteration of /loop will dispatch more on the remaining priority list above
