# UNRESOLVED PROMOTED — open questions with concrete retest proposals

**Date:** 2026-04-22 (wave-4f synthesis).
**Source for sketches:** `RETEST_PROPOSALS.md` (V52–V56 ready-to-compile `.cu`).
**Source for status:** `DOUBT_LOG.md` §3 (top-10 wave-3c) +
`META_DOUBT_REPORT.md` re-promotions + `CONFIDENCE_LADDER.md` top-10 LOW.

Each row maps an UNRESOLVED item to the concrete sketch that would settle it,
with a 1-line decision rule. Priority ordered by **downstream blast radius**
(how many other claims would shift if the answer flipped).

---

## P0 — anchors multiple downstream docs

| # | UNRESOLVED item | Conf | Wave-3c top-10 row | Sketch | Decision rule |
|---|---|---|---|---|---|
| 1 | **Dual-issue 55%/74% interpretation** — V49 same-warp / V50 warp-spec FFMA+LOP3. Numbers reproducible (re-promoted to MED in wave-4); "dispatch cap 4 inst/SM/cy" mechanism unconfirmed. | MED | row 1 (was LOW) | **`v52_dual_issue_warp_sweep.cu`** — sweep warps/SMSP ∈ {1,2,4,8} + ncu `smsp__inst_issued.per_cycle_active`, `pipe_fma_cycles_active`, `pipe_alu_cycles_active` | If `inst_issued.per_cycle_active` plateaus ≤ 1.05 across all warps/SMSP → dispatch cap REAL (architectural). If sum of `pipe_fma + pipe_alu` > 130% at warps≥4 → pipes ARE separate, V49 was occupancy-starved. **Blast radius: 3 downstream docs (DUAL_ISSUE_DOUBT, V49 commit, V50 commit), the FFMA peak interpretation, NVFP4 throughput model, power-vs-throughput tradeoffs.** |
| 2 | **HBM "% of peak" anchor** — denominator standardization is RULE-stated (7680 GB/s post-ECC) but no recipe yet locks the empirical ceiling on B300 SXM6. Affects every HBM % claim across `01_hbm_bandwidth.md`, V32–V48 docs, TRUE_REFERENCE row 25. | RULE→MED | new (HBM denom resolution) | **`v55_hbm_floor_BEST.cu`** — TMA bulk loads, 4–64 KB tile sweep × 4–8-deep × per-warp issue, 4 GB working set, 148 CTAs. ncu `dram__bytes_read.sum.per_second` + L2 hit < 5% gate. | Take MAX TB/s where `lts__t_sector_hit_rate.pct < 10%`. Compare to V32 7.31 TB/s and 7680 GB/s post-ECC. **Result becomes the canonical % anchor** for all "% of HBM peak" claims. **Blast radius: every HBM number in the catalog.** |

## P1 — settles a 1.5×+ spread or a known caveat

| # | UNRESOLVED item | Conf | Wave-3c top-10 row | Sketch | Decision rule |
|---|---|---|---|---|---|
| 3 | **DSMEM 40 GB/s read** — chain-bound (V21 dependent-chain). Non-chained ILP could be 60-80. | MED | row 4 | **`v53_dsmem_fenced_retest.cu`** read mode — non-chained ILP, addresses loop-carried but values not in addr path. | If RD non-chained > V21 RD by ≥ 1.2× → upgrade DSMEM read SoL; restate the 40 GB/s as a chain-bound floor. |
| 4 | **DSMEM 560 GB/s write** — issue rate, no fence between stores and clock64 stop (V21). | LOW-MED | row 5 | **`v53_dsmem_fenced_retest.cu`** write mode — `fence.sc.cluster` between every batch + before final clock64. | If WR_fenced/WR_unfenced > 1.3× → V21 number was inflated; downgrade to fenced number. If < 1.05× → V21 number holds. |
| 5 | **`__threadfence_system` cost** — 1750 / 2870 / 3042 cy = 1.74× spread across 08 / DSMEM / V9. | DISPUTED | row 2 | **`v54_membar_isolation.cu`** — N-issue scaling N ∈ {1,2,4,8}, median of 21 runs at locked 1920 MHz, SASS-verify MEMBAR.SYS count. | Per-issue slope determines fixed setup vs amortized cost. Report median single-issue at locked 1920 MHz with N=1..8 table; supersedes all three prior values. |
| 6 | **HBM write SoL 7.57 TB/s provenance** — attributed to NINJA STG (e75c7e1) AND V8 TMA bulk store (28211ce). | DISPUTED | row 3 | Re-run both kernels back-to-back with ncu `dram__bytes_write.sum.per_second`. Falls out of `v55_hbm_floor_BEST.cu` if extended with write mode. | Whichever kernel actually delivers ≥ 7.5 TB/s with HBM hit rate < 5% wins the attribution; the other becomes a retraction. |

## P2 — exploratory mechanism work

| # | UNRESOLVED item | Conf | Wave-3c top-10 row | Sketch | Decision rule |
|---|---|---|---|---|---|
| 7 | **NVFP4 K=96 A:B asymmetry mechanism** — 4 candidates: TMA multicast / operand-swap / SMEM dwell / pipeline depth. cuBLAS A>B 3:1 vs pure tcgen05 B>>A 15-30× = unresolved. | MED | wave-3c #6 (preserve-3) | **`v56_nvfp4_AB_mechanism.cu`** — 5 modes (baseline / role swap / B-multicast / single-buffered A / cluster=1) × NVML 100 Hz × 5 s sustained. | Decision tree: mode 1 inverts → contents-driven (mechanisms #1-4 wrong). Mode 2+4 equalize → multicast (#1). Mode 3 amplifies → dwell (#3). None changes → operand asymmetry built into MMA path (#2). |
| 8 | **DSMEM "NO shared bus"** — V17 ring was 30× under-issued. | LOW | row 6 | Combine with V53 — add multi-CTA contention loop with non-chained ILP at full issue rate. | If aggregate ≤ N × per-pair → no shared bus confirmed. If sub-linear → shared fabric exists. |

## P3 — narrow-impact UNRESOLVED items (no sketch yet)

| # | UNRESOLVED item | Conf | Wave-3c top-10 row | Path forward |
|---|---|---|---|---|
| 9 | **`__threadfence` (GPU) 281 cy** — 4-way 24% spread (258/281/292/320 cy). | MED | row 7 | Sync agent's "281 ± 25" framing is correct; just adopt-the-spread, no new test. |
| 10 | **IADD3 rate 0.50 vs 0.66 inst/SMSP/cy** — V40 vs A6 = 30% gap. | MED | row 8 | A6-style sweep at 4+ warps/SMSP for IADD3 + PRMT side-by-side; small variant of V52's harness. |
| 11 | **PRMT rate** — V40 0.36 vs A6 0.50. | MED | row 9 | Same harness as #10. |
| 12 | **L2 atomic units count "~32"** — derived ceiling, not direct measurement. | MED | wave-3c new finding #9 | Stride sweep with ncu `lts__t_bytes` per partition. |
| 13 | **Cluster=2 21% slower than ≥3** — single-GPC vs multi-GPC topology hypothesis. | MED | wave-3c HEADLINE_v2 UNRESOLVED | ncu `gpc__cycles_active.per_pgpc_id` pass on the existing V12 cluster sweep. |
| 14 | **PCIe Gen 6 cap mechanism** — "CPU-bound" RETRACTED, real root cause unconfirmed. | LOW | row 10 | Needs CPU-side profiling (perf record on the host process during a 60 GB/s sustained transfer) — outside the QuickRunCUDA harness. |
| 15 | **`cuStreamWriteValue32` 0.45 vs 2.47 µs** — single-write vs full-pair latency disagreement. | MED | wave-3c #14 (latency table) | Run both modes back-to-back with `cudaEvent` brackets. |
| 16 | **Persistent kernel 2.03 µs mechanism** — "v1 used release" hypothesis unverified. | MED | wave-3c #10 | SASS-diff between v1 (584fda6) and v2 (dcc0f20). |

---

## Items removed from the open list (wave-4)

- **Single-GPU multi-stream HBM aggregate** — `V51_INVESTIGATION.md`
  shows V51 was UB (host-pointer cast bug, never fixed). Question is
  architecturally trivial (one HBM bus per GPU). **Remove
  `tests/standalone/v51_multistream_hbm.cu`** and do not re-add to the
  curiosity list.

---

## How to use this document

1. Pick the highest-priority [P0 → P3] item that interests you.
2. Apply CLAUDE.md §3 rigor: `pkill -9 QuickRunCUDA && sleep 5-8`,
   3-method verification (wall-clock + ncu + SASS), state clock state.
3. Use the linked sketch from `RETEST_PROPOSALS.md` as the starting kernel.
4. On completion: update `DOUBT_LOG.md` §3 (move row out of top-10), update
   `CONFIDENCE_LADDER.md` row(s) (HIGH or DOWNGRADED with citation), append
   to `WAVE4_CHANGES.md` §1 if the verdict reverses again.
