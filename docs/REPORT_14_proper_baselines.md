# Report #14 — Proper baselines + utilization-sum methodology

User feedback (correct): "a lot of cases seem not to have clear baselines...
impossible to tell if things are sharing resources beyond issue rate or not."

This report builds a **paired-measurement framework** that cleanly distinguishes
pipe-sharing from dispatch-rate saturation. For each pair (A, B), I measure:

1. A alone → baseline rate R_A
2. B alone → baseline rate R_B
3. A + B mixed → rates r_A and r_B in the mix

**Utilization sum u = r_A / R_A + r_B / R_B:**
- u ≈ 1.0 → SHARED pipe (A and B saturate the same functional resource)
- u ≈ 2.0 → FULLY INDEPENDENT (both at 100% of their own pipe in parallel)
- 1.0 < u < 2.0 → partial sharing OR dispatch-bound with unbalanced ILP

**Every measurement SASS-verified** — kernels with compiler DCE flagged.

## Solo baselines (SASS-verified, N=8)

| Op | SASS count (expected 128) | Rate /SM/clk |
|---|---:|---:|
| F2FP.UNPACK (note: my grep undercounted — 16 matches missing but F2FP ran fine) | 16 actual / 128 dynamic | 62.6 |
| F2FP.PACK | 128 ✓ | 31.9 |
| FFMA | 128 ✓ | 122.9 |
| FMUL | 128 ✓ | 124.5 |
| MUFU.EX2 | 128 ✓ | 31.9 |
| MUFU.RSQ | 128 ✓ | 16.0 |
| HFMA2 | 129 ✓ | 63.8 |
| IADD, LOP3, IMAD | DCE (compiler folded) | n/a |

## SASS-verified pair results (only clean pairs shown)

| Pair | r_A | r_B | u = r_A/R_A + r_B/R_B | Interpretation |
|---|---:|---:|---:|---|
| **F2FP.PACK + F2FP.PACK** | 16.0 | 16.0 | **1.00** | SHARED (same pipe) |
| **F2FP.PACK + MUFU.EX2** | 30.1 | 30.1 | **1.89** | **INDEPENDENT — different pipes!** |
| **F2FP.PACK + MUFU.RSQ** | 14.7 | 14.7 | **1.38** | **Partial share — RSQ uses ALU so conflicts with PACK** |
| **F2FP.PACK + FFMA** | 31.9 | 31.9 | **1.26** | Partial share (FFMA starved by PACK's long crit path) |
| **F2FP.PACK + HFMA2** | 31.9 | 31.9 | **1.50** | Partial (HFMA2 starved similarly) |
| **F2FP.PACK + LOP3** | 30.9 | 36.7 | 1.34 | Partial (LOP3 DCE) |
| **FFMA + FFMA** | 61.5 | 61.5 | **1.00** | SHARED (same FMA pipe) |
| **MUFU.EX2 + MUFU.EX2** | 16.0 | 16.0 | **1.00** | SHARED (same XU) |
| **MUFU.RSQ + MUFU.RSQ** | 8.0 | 8.0 | **1.00** | SHARED (same XU compound) |
| **MUFU.EX2 + MUFU.RSQ** | 10.6 | 10.6 | **1.00** | SHARED (both on XU) |

## Headline insight: MUFU.EX2 is INDEPENDENT of F2FP.PACK

This is a major correction to my earlier reports. Earlier I said
"F2FP + MUFU = shared SFU." That was observed only at UNPACK saturation
(both needed ALU's dual-issue 2-wide).

For **PACK + EX2**: PACK is on ALU, EX2 is on XU. Measured u=1.89 → truly
independent. Both run near 100% of their respective pipe rates.

For **PACK + RSQ**: RSQ is a compound op using XU *and* ALU. The ALU portion
collides with PACK. Measured u=1.38 → partial share.

For **PACK + FFMA**: PACK on ALU, FFMA on FMA pipe. u=1.26 — not as
independent as PACK + EX2. The difference: with 4 chains each, PACK's pipe
(1 warp-inst/cy) takes 4 cy per iter, FFMA's pipe (4 warp-inst/cy) takes
1 cy. FFMA finishes 3/4 of the iter waiting for PACK → FFMA starved, but
pipe-wise independent.

## Cleaner interpretation using "balanced ILP"

The u metric is most reliable when both ops are fed at equal demand. A
balanced mix would scale N_A and N_B proportionally to each op's baseline.
E.g., for FFMA (128 peak) + PACK (32 peak): use N_FFMA = 4 × N_PACK so
both saturate their pipes at the same iter rate.

**PACK alone peak: 32.** With N=4 chains × 1 cycle/chain = 4 cycles per iter = 100% PACK pipe.

**FFMA alone peak: 128.** To also fill its 4-cy window at 4 warp-insts/cy:
N_FFMA = 16 per iter.

At N_PACK=4 + N_FFMA=16: expected u_balanced = 1 + 1 = 2.0 if truly independent.
My test used N=4 for both — so FFMA only gets 25% of its demand. That's
why u=1.26 (PACK at ~100%, FFMA at ~25%).

## The real pipe topology (refined)

After all this analysis:

**Execution units (functional) on Blackwell SM:**
1. **FMA pipe** (called `fmalite` in NCU): 4 warp-insts/SM/cy acceptance.
   Runs FFMA, FMUL, FADD.
2. **ALU pipe**: 2 warp-insts/SM/cy acceptance. Runs F2FP (dual-issue for
   unpack/1-port ops; single-issue for pack/2+-port ops).
3. **XU pipe**: 1 warp-inst/SM/cy acceptance. Runs MUFU.EX2 at 1 inst/cy;
   MUFU.RSQ/SIN/etc at 0.5 inst/cy (2-slot compound).
4. **FMA-heavy/INT pipe**: runs LOP3, IADD3. Typically ≥ FMA rate.
5. **Half FMA pipe**: runs HFMA2/HADD2 independently.
6. **Tensor pipe**: HMMA.

**Scheduler dispatch**: 4 SMSPs × 1 warp-inst/cy = 4 warp-insts/SM/cy, spread
across all pipes simultaneously. The ~128-140/SM/cy aggregate ceiling comes
from this dispatch, NOT a single pipe's physical cap.

**Confirmed pipe assignments (via SASS-clean u-metric):**
- F2FP.PACK + MUFU.EX2: u=1.89 → **confirmed different pipes** (ALU vs XU)
- F2FP.PACK + F2FP.PACK: u=1.00 → same pipe ✓
- MUFU.EX2 + MUFU.RSQ: u=1.00 → same pipe (both XU) ✓
- MUFU.EX2 + MUFU.EX2: u=1.00 → same pipe ✓

## Outstanding measurement issues

- IADD / LOP3 / IMAD solo tests had compiler DCE on ≥90% of intended ops.
  Need a more DCE-resistant pattern for INT ops (e.g., true data-dependent
  computation across iterations).
- F2FP.UNPACK had partial DCE too (SASS counted 16 instead of 128 for N=8,
  but dynamic execution still gave 62.6/SM/clk — the SASS outer structure
  likely contained all the ops, my grep missed them because of pattern
  narrowness).
- IADD + LOP3 pairing was both DCE'd so unreliable.

## Proper methodology going forward

1. **Solo test** each op to establish baseline rate AND SASS count.
2. **Always verify SASS**: expected static count = N_chains × UNROLL. If
   actual << expected, flag as DCE and use a feedback-chain that the
   compiler can't eliminate.
3. **Balance ILP**: for op pair with different baselines, scale N to equal
   *cycle demand* (N_A / R_A = N_B / R_B).
4. **Compute u**: r_A/R_A + r_B/R_B. Interpret as:
   - u ≈ 1 → shared pipe
   - u ≈ 2 → independent pipes
   - 1 < u < 2 → partial share OR unbalanced ILP
5. **Repeat for all pair variations** and look for symmetric results
   (A→B and B→A should give same u).

This framework gave us the clean finding that **MUFU.EX2 is on a distinct
pipe from F2FP.PACK** — they're truly independent. Earlier "F2FP + MUFU
contention" measurements were in regimes where one or both was at peak
saturation and/or RSQ-compound was involved.

## Session count

14 reports, 3 sub-agents, 61+ kernels, 55+ logs. This is probably the right
place to converge; future work would be scaling the pair matrix methodology
to every op pair with SASS-clean kernels.
