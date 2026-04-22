# WAVE-4 CHANGES — what shifted vs wave-3c

**Date:** 2026-04-22. **Supersedes:** wave-3c `DOUBT_LOG.md` and
`HEADLINE_CORRECTIONS_v2.md` on the items listed below; everything else in
those files still stands.

**Inputs that drove this update:**
- `META_DOUBT_REPORT.md` (audit of the doubt reports themselves)
- `HBM_DENOMINATOR_RESOLUTION.md` (settles spec / post-ECC / empirical anchors)
- `RETEST_PROPOSALS.md` (5 concrete .cu sketches V52–V56)
- `CONFIDENCE_LADDER.md` (303 graded rows)
- `V51_INVESTIGATION.md` (V51 cast-bug forensics)

---

## 1. Reversed downgrades (wave-3c was too harsh)

| Item | Wave-3c verdict | Wave-4 verdict | Why |
|---|---|---|---|
| V49 same-warp dual-issue **55%** | LOW | **MED** | `META_DOUBT_REPORT.md` §1: V8_FFMA_PEAK_VERIFIED hits **97.64%** at the **identical 2 warps/SMSP** geometry V49 uses (`__launch_bounds__(256,1)` = 2 warps/SMSP). The "under-occupancy" mechanism is FALSIFIED. The 55% measurement is real; only the dispatch-cap **interpretation** is underdetermined. |
| V50 warp-spec dual-issue **74%** | LOW | **MED** | Same chain of reasoning; the 74% is reproducible, the architectural framing needs ncu before promotion to HIGH. |

**What still needs measuring:** ncu `smsp__inst_issued.sum.per_cycle_active`
+ `pipe_fma_cycles_active` + `pipe_alu_cycles_active` on V49 OP=2. Sketch
`v52_dual_issue_warp_sweep.cu` settles it. Real anomaly worth chasing:
V49's solo-FFMA baseline is itself 67% of peak (vs V8's 97.6%) despite
identical occupancy — the source-operand pattern (`fma %0,%0,imm,imm` vs
`fma %0,%0,%1,%0`) differs.

## 2. New facts (HBM denominator finally settled)

`HBM_DENOMINATOR_RESOLUTION.md` cites NVIDIA Developer Blog + multiple
third-party teardowns:

- **B300 has 8 HBM3E stacks (12-Hi each).** "12" = die stack height, NOT
  stack count. **CLAUDE.md and `01_hbm_bandwidth.md` line 136 are WRONG**
  ("12 stacks × 1024 bit"). Real bus = 8 × 1024 = **8192-bit**.
- **HBM raw (pre-ECC) = 8192 GB/s** (= 8.19 TB/s, marketing-rounded "8 TB/s").
- **HBM post-ECC = 7680 GB/s** (1/16 reserved, matches B200 192→180 ratio).
- **The "7672 GB/s" in TRUE_REFERENCE / 01_hbm / HBM_INCONSISTENCY_LOG is
  an arithmetic ghost** (intermediate 1998 MHz × 2 = 3996 GT/s estimate
  loses 0.1% to rounding). Replace with clean **7680**.
- **7.31 TB/s is empirical, not theoretical** — never use as `% of peak`
  denominator without naming it as "% of SoL recipe a04d9c8".

## 3. Bug found (wave-4-original)

`V51_INVESTIGATION.md`: `tests/standalone/v51_multistream_hbm.cu` (untracked)
**never had its per-stream-buffer fix wired through**. Both warmup and
timed launches pass `(const float*)d_src` — the **host stack address of
the pointer-array** — instead of `d_src[s]`. Result: undefined behavior;
any "result" the file emits is fabricated.

**Action: REMOVE.** The single-GPU multi-stream-aggregate question is
architecturally trivial (one HBM bus → streams cannot exceed it) and is
not on any UNRESOLVED list. Do not ship.

## 4. Newly-unlocked: 5 concrete re-test sketches

`RETEST_PROPOSALS.md` provides ready-to-compile `.cu` files for the top
unresolved items. See `UNRESOLVED_PROMOTED.md` for the priority ordering.

| Sketch | Settles |
|---|---|
| `v52_dual_issue_warp_sweep.cu` | Dual-issue 55%/74% interpretation (#1) |
| `v53_dsmem_fenced_retest.cu` | DSMEM 40 GB/s read & 560 GB/s write caveats (#4, #5) |
| `v54_membar_isolation.cu` | `__threadfence_system` 1.74× spread (#2) |
| `v55_hbm_floor_BEST.cu` | HBM read empirical ceiling for `% of peak` anchor |
| `v56_nvfp4_AB_mechanism.cu` | NVFP4 K=96 A:B asymmetry — 4-mechanism discriminator |

**Recommended order:** V52 first (highest blast-radius — anchors three
downstream docs), then V55 (anchors every HBM % claim), then V53/V54/V56
in parallel.

## 5. What did NOT change from wave-3c (still valid)

- DSMEM caveats: read chain-bound, write issue-rate-only, "no shared bus" unproven (V17 30× under-issued). All three caveats source-verified by `META_DOUBT_REPORT.md` §3.
- NVFP4 A:B preserves all three readings; wave-2's single-mechanism story remains over-resolved.
- NVLink-5 (NOT v7), 900 GB/s/dir spec — confirmed.
- All 10 wave-3c "RETRACTIONS already done upstream" still apply.
- All 7 "RETRACTIONS genuinely new from wave-3" still apply.
- `__threadfence_system` 1.74× spread still UNRESOLVED.
- HBM write SoL 7.57 TB/s provenance still UNRESOLVED.

## 6. New methodology rule (add to wave-3c list)

**11. Before re-measuring any HBM number, standardize the denominator.**
Use **7680 GB/s post-ECC** as the canonical % anchor (matches what ncu's
`dram__bytes_*` actually counts). State 8192 GB/s if comparing against
raw bus capability. Never call 7.31 TB/s "spec" or "theoretical".
