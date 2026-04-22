# SYNC / FENCE / BARRIER — INCONSISTENCY LOG

Format: claim — file:line — verdict.

## Numeric disagreements

| # | Claim A | Claim B | Verdict |
|---|---|---|---|
| 1 | `__threadfence` GPU = **281 cy / 138 ns** (V9_THREADFENCE_COST.md:9) | `__threadfence` GPU = **277–292 cy / 144 ns** (08_sync_primitives.md:23) | Same range. CONSISTENT. |
| 2 | `__threadfence` GPU = **258 cy** baseline-subtracted (V9_THREADFENCE_COST.md:11) | `fence.sc.gpu` = **320 cy** (DSMEM_REFERENCE.md:117, DSMEM_FINDINGS_V2.md:151) | 24% gap. Likely loop-overhead vs cluster-context; needs re-run. UNRESOLVED. |
| 3 | `__threadfence_system` = **1750 cy / 861 ns** (08_sync_primitives.md:26) | `__threadfence_system` = **3042 cy / 1486 ns** (V9_THREADFENCE_COST.md:12) | **1.74× discrepancy.** TRUE_REFERENCE follows 861 ns. Real disagreement. UNRESOLVED. |
| 4 | `fence.sc.sys` = **2870 cy** (DSMEM_REFERENCE.md:118) | Same as above 1750 / 3042 | DSMEM number sits between the two; suggests V9's 3042 may include extra coherence traffic. |
| 5 | `__syncthreads(1024)` = **77 cy** (08_sync_primitives.md:19) | `__syncthreads(1024)` = **86 cy** by formula 22+2W (V9_SYNCTHREADS_COST.md:14) | 12% gap. V9 has 6-point linear fit; trust V9. 08 should be updated. |
| 6 | `__syncwarp` ≈ **1 cy** (F6_SYNCWARP_COST.md, F2_SYNCWARP_RIGOR.md) | `__syncwarp` ≈ **23 cy** (V9_THREADFENCE_COST.md:9) | V9's 23 cy is **loop overhead**, mislabeled. F2/F6 are authoritative. RETRACT V9 framing. |
| 7 | `membar.cta` = **9 cy** (08_sync_primitives.md:15) | `membar.cta` = **6 cy** (F6_SYNCWARP_COST.md:15) | 33% gap. Both single-thread; methodology differs. Prefer F6 (newer, explicit baseline). |
| 8 | `__threadfence_block` = **8 ns / 16 cy** (08_sync_primitives.md:16) | `__threadfence_block` = **~0** (V9_THREADFENCE_COST.md:11) | Same instruction. V9 likely captures only the post-issue cost, 08 includes scoreboard wait. UNRESOLVED. |
| 9 | `cluster.sync` = **373–380 cy** (08_sync_primitives.md:22) | `cluster.sync` = **370 cy** (V9_SYNCTHREADS_COST.md:50) | CONSISTENT (within rounding). |
| 10 | `barrier.cluster.relaxed` = **102 cy / 50 ns** (08_sync_primitives.md:21) | TRUE_REFERENCE same 50 ns (B300_TRUE_REFERENCE.md:80) | CONSISTENT. |
| 11 | `mbarrier arrive+wait` = **123 cy** (V10_VERIFICATION_SUMMARY.md:65, V10_GRID_SYNC.md:42) | `mbarrier.arrive + test_wait` = **54 cy** (08_sync_primitives.md:20) | Different ops (full wait vs test_wait). Both correct; need to add 123-cy row to 08 ladder. |
| 12 | `mbarrier.arrive` only = **24 cy** (M7_V5_SYNTHESIS.md:14, A6) | not in 08 catalog | Add as separate row. |

## CLAUDE.md memory claims requiring sourced verification

| Memory claim | Status |
|---|---|
| "Memory fence 3-tier system + 8-channel membar.sys fabric limit" | **NO SOURCE FOUND** — no 8-channel sweep test exists in `b300_clean/`. **Treat as hallucination until reproduced.** |
| "fence.sc vs fence.acq_rel identical cost (36-cell matrix)" | **PARTIAL** — DSMEM_FINDINGS_V2 confirms ==320 cy at cluster scope only (2 of 36 cells). The remaining 34 cells of any "fence × scope × ordering" matrix are not produced by any test in the tree. **No 36-cell matrix exists.** |
| "Fence: block free / GPU 281 / sys 3042 cy" | **PARTIAL.** Block-free agrees with V9. GPU 281 cy agrees with V9_THREADFENCE_COST. Sys 3042 cy agrees with V9 but DISAGREES with 08 (1750) and TRUE_REFERENCE (861 ns ⇒ 1750 cy). |
| "mbarrier 123 cy" latency | **VERIFIED** (V10_VERIFICATION_SUMMARY.md:65, V10_GRID_SYNC.md:42). Specifically arrive+wait, not arrive-only. |

## Arithmetic spot-check

DSMEM ratio: sys/gpu = 2870 / 320 = **8.97×** ≈ "9× slower" claim ✓
V9 ratio: sys/gpu = 3042 / 281 = **10.8×** — V9's "12× GPU fence" rounding is off (says 12×, actual 10.8×).

08 ratio: sys/gpu = 1750 / 281 = **6.2×** — much smaller than DSMEM's 9× and V9's 11×. **The system fence number in 08 is the outlier.**

Conversion check (V9 cycles): 281 cy / 1.500 GHz = 187 ns; 281 / 2.032 = 138 ns. V9 reports 138 ns ⇒ **V9 used 2.032 GHz for ns conversion but its raw cy were measured at 1.500 GHz** per CLAUDE.md "feedback_microbench_rigor" guidance? Unclear. Need test-script clock verification.

3042 cy / 2.032 GHz = 1497 ns ≈ V9's "1486 ns" ✓ (V9 used boost for conversion).

But: if the test ran at 1500 MHz, true ns = 3042/1.5 = **2028 ns**, not 1486. **Possible 1.36× over-claim of throughput / under-claim of latency in V9.**

## Recommended actions

1. Re-run `tests/bench_fence_cost.cu` at LOCKED 1920 MHz with explicit cycle-and-ns-double-reporting to settle GPU-fence (258/281/292/320) and sys-fence (1750/2870/3042) splits.
2. Add `mbarrier.arrive` (24 cy) and `mbarrier.arrive+wait` (123 cy) rows to 08_sync_primitives.md ladder.
3. Annotate 12_nvlink_p2p.md fence-drain table with the clock state used.
4. Strike "8-channel membar.sys" and "36-cell matrix" from CLAUDE.md memory until a producing test exists.
5. Add explicit "cy measured at: X MHz; ns rescaled to: Y MHz" header to every latency table going forward.
