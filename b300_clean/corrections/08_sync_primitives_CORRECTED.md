# 08 Sync Primitives — CORRECTED LADDER

**Platform**: B300 SXM6, sm_103a. **Clock**: cy quoted at the clock the source
test ran on; ns rescaled to **2.032 GHz boost** unless otherwise noted.
**Source-of-truth date**: 2026-04-22.

---

## Reconciled Latency Ladder (single warp / single CTA, isolated)

| Primitive | cy (best-evidence) | ns @ 2.032 GHz | Source(s) |
|---|---:|---:|---|
| `__syncwarp(0xFFFFFFFF)` const | **0–2** (NOPs only, no SASS emitted) | **0–1.0** | F2_SYNCWARP_RIGOR (commit verified), 08_sync_primitives row 1 |
| `__syncwarp(partial mask)` (BSYNC) | 7.25 | 3.6 | F2_SYNCWARP_RIGOR |
| `__shfl_sync` broadcast idx=0 | ~2 (UIMOV uniform pipe) | 0.85 | 08_sync_primitives |
| `membar.cta` / `fence.acq_rel.cta` / `__threadfence_block` | 6–16 (see UNRESOLVED) | 3–8 | F6 (6 cy), 08 (9 cy), V9_THREADFENCE_COST (~0 single-thread!) |
| `__syncthreads` (32 thr / 1 warp) | 23–24 | 11–12 | V9_SYNCTHREADS_COST (formula 22+2W) |
| `__syncthreads` (256 thr / 8 warp) | 38 | 19 | V9 formula, 08 catalog |
| `__syncthreads` (1024 thr / 32 warp) | 86 | 42 | V9 formula, 08 catalog (77 cy) — small disagreement |
| `mbarrier.arrive` (no wait) | 24 | 12 | M7_V5 A6 |
| `mbarrier.arrive + test_wait` (1 thr) | 54 | 26 | 08 catalog |
| `mbarrier.arrive + wait` (block-scope, single thr loop) | 123 | 60 | V10_VERIFICATION_SUMMARY, V10_GRID_SYNC |
| `barrier.cluster.arrive.relaxed + wait` (cluster=2) | **102** | **50** | 08 catalog (cluster_raw_barrier.cu) |
| `cluster.sync()` strict | 370–380 | 175–187 | 08 catalog, V9 (370), DSMEM (613 cy per-msg incl barrier overhead) |
| `__threadfence` / `fence.sc.gpu` (no traffic) | **277–292 (cycles only)** OR **~258 cy = 138 ns** OR **320 cy** | **138–144** | 08 catalog (281), V9_THREADFENCE_COST (258), DSMEM_REFERENCE (320) |
| `__threadfence` (chip-wide write traffic) | 783 | 385 | 08 EXTENDED §1 |
| `__threadfence_system` / `fence.sc.sys` (idle) | 1750–3042 | 861–1486 | **DISAGREES**: 08 says 1750/861ns; V9 says 3042/1486ns |
| `fence.sc.sys` (saturated chip) | ~19000 | ~9300 | 08 sec 30.G |
| `grid.sync()` (148 blocks × 128 thr) | 2376 | 1170 | V10_GRID_SYNC |
| Cross-block flag wait (volatile + threadfence) | 1605 | 790 | 08 (pingpong) |

`fence.sc.gpu == fence.sc.cluster` in cost (DSMEM_REFERENCE rule 9; both 320 cy).

---

## Cross-GPU NVLink Fence Drain (12_nvlink_p2p §6, HIGH)

| Scope | LOCAL | REMOTE | NVLink drain |
|---|---:|---:|---:|
| fence.sc.cta | 495 | 5786 | +5291 |
| fence.sc.gpu | 1852 | 19645 | +17793 |
| fence.sc.sys | 8952 | 26738 | +17786 |

(Cycles; no boost-clock conversion documented in source — units may be cy at 1500 MHz lock.)

---

## RETRACTIONS

1. **CLAUDE.md memory "8-channel membar.sys fabric limit / 36-cell matrix"** — NO file in `b300_clean/` documents either an 8-channel membar.sys split or a 36-cell `fence.sc/acq_rel × scope × ordering` matrix. The closest evidence is the 6-row fence ladder in `08_sync_primitives.md` and the 4-row table in `DSMEM_REFERENCE.md`. **The 36-cell matrix appears unsourced and should be considered a memory-level hallucination until a producing test is found.**
2. **CLAUDE.md memory "fence.sc vs fence.acq_rel identical cost"** — partially supported (DSMEM_FINDINGS_V2 rows show fence.acq_rel.cluster == fence.sc.cluster == 320 cy) but ONLY at cluster scope, single thread. Not a complete 36-cell sweep.
3. **V9_THREADFENCE_COST baseline label** — V9 calls baseline "syncwarp 23 cy" but that 23 cy is **loop overhead**, not the syncwarp itself (F6 + F2 prove syncwarp full-mask = 1 cy / 0 SASS). The "281 cy" fence cost is correct; the framing is misleading.
4. **08_sync_primitives "membar.cta = 9 cy"** vs **F6 "membar.cta = 6 cy"** vs **V9 "fence_block ≈ 0 cy"** — the V9 single-thread number is right for cost-above-noise; the 6–9 cy figures include scoreboard wait and are equally defensible. RETRACT the "single canonical value" framing in 08.
5. **08_sync_primitives row "`__syncthreads (1024 thr) = 77 cy`"** disagrees with V9 formula (22+2·32=86 cy). Prefer V9's formula (3 independent sweeps); 77 cy is likely a clock-state mismatch.

---

## UNRESOLVED

1. **GPU fence cost: 258 vs 281 vs 292 vs 320 cy** — all four numbers exist in the catalog:
   - V9_THREADFENCE_COST: ~258 cy (1500 MHz baseline-subtracted, single thread)
   - V10_VERIFICATION_SUMMARY: 281 cy
   - 08_sync_primitives: 277–292 cy ("isolated")
   - DSMEM_REFERENCE: 320 cy (cluster-launch context)
   The spread is ~24%. Hypothesis: differences come from (a) loop-overhead subtraction methodology, (b) clock state (1500 vs 1920 vs 2032 MHz), (c) presence of cluster context (CCTL.IVALL adder). **Action: re-run a single fence_cost.cu at boost without baseline subtraction; report all four sub-instructions (MEMBAR.SC.GPU + ERRBAR + CGAERRBAR + CCTL.IVALL) separately.**
2. **System fence: 1750 vs 3042 cy** — 08_sync_primitives says 1750 cy/861 ns; V9_THREADFENCE_COST says 3042 cy/1486 ns. **1.74× discrepancy.** TRUE_REFERENCE adopts the 861 ns number (b06f366) but cites no per-test reproduction. Possible causes: NVLink coherence variability, peer-CTA traffic in V9 environment, or 1500 vs 2032 MHz mismatch (1750·(2032/1500)=2371, still not 3042; clock alone doesn't bridge it). **Re-run isolated.**
3. **mbarrier "123 cy" arrive+wait latency** (V10) vs "54 cy arrive+test_wait" (08 catalog) vs "24 cy arrive-only" (M7 A6) — three numbers because they measure three different operations. The latency-ladder convention should be **`arrive+wait` = 123 cy** (full RTT) when comparing with `__syncthreads`. **08_sync_primitives should add a row for the 123 cy data point.**
4. **`__syncwarp` cost 1 cy (F6) vs 23 cy (V9 baseline)** — V9 uses syncwarp as loop-overhead proxy, not as the measurand. F6 + F2 are the authoritative answers (1 cy full-mask, 7.25 cy partial). **Update V9's framing; do not retract 281 cy.**
5. **TRUE_REFERENCE row "cluster.barrier::arrive (relaxed) = 50 ns"** matches 08 catalog's 102 cy / 50 ns. **CONSISTENT.** No correction needed.
6. **"membar.sys 8-channel" claim** — no producing test exists in the tree. Until verified by an N-writer scaling sweep on `fence.sc.sys`, this is conjecture.
7. **NVLink 17.8 kcy drain adder** — units (cy at which clock?) not stated in `12_nvlink_p2p.md`. Likely 1500 MHz lock (~11.9 µs); needs a header annotation.

---

## RECOMMENDED CANONICAL TABLE (use this)

| Op | cy @ boost | ns @ 2.032 GHz | Confidence |
|---|---:|---:|---|
| __syncwarp full-mask | 1 (NOP) | 0.5 | HIGH |
| __syncwarp partial | 7 | 3.5 | HIGH |
| __threadfence_block | 6–16 (one-thread) | 3–8 | MED (range) |
| __syncthreads(W) | 22 + 2W | (22+2W)/2.032 | HIGH |
| mbarrier.arrive | 24 | 12 | HIGH |
| mbarrier.arrive+wait | 123 | 60 | HIGH |
| barrier.cluster.arrive.relaxed+wait | 102 | 50 | HIGH |
| cluster.sync (strict) | 373 | 184 | HIGH |
| __threadfence (GPU) | 281 ± 25 | 138 ± 12 | MED (4-way spread) |
| __threadfence_system | **UNRESOLVED 1750–3042** | 861–1486 | LOW |
| grid.sync (148 blocks) | 2376 | 1170 | HIGH |
