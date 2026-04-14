# Report #15 — Balanced-ILP pair matrix (DCE-clean SASS verified)

User called out correctly: prior pair tests (sub-agent 1's in particular)
used N_COMP=32 MUFU against smaller F2FP, which just made MUFU the limiter.
That doesn't prove pipe sharing — it proves MUFU takes cycles.

**Proper methodology: scale N_A and N_B so each op's cycle demand is equal.**

Kernel: `tests/bench_balanced_pair.cu` with per-chain unique XOR feedback
to prevent compiler CSE. All results SASS-verified (128/128 match).

## Baselines (with 1 LOP3 in feedback chain, 8 chains each)

| Op | Rate /SM/clk | Baseline for u-metric |
|---|---:|---|
| UNPACK (F2FP.F16.E4M3.UNPACK_B) | 32 | (reduced from 64 by feedback LOP3 latency) |
| PACK (F2FP.*.UNPACK_B_MERGE_C) | 21 | (reduced from 32) |
| EX2 (MUFU.EX2) | 32 | (matches theoretical) |
| RSQ (MUFU.RSQ) | 16 | |
| FFMA | 123 | |

All ops share the same feedback kernel pattern, so ratios are fair.

## Results (SASS-verified; u = r_A/R_A + r_B/R_B)

### Same-pipe controls

| Pair | u | Status |
|---|---:|---|
| UNPACK + PACK | **1.00** | ✓ SAME PIPE (ALU — both F2FP) |
| EX2 + RSQ | **1.00** | ✓ SAME PIPE (XU/SFU) |

### Independent-pipe tests

| Pair | u | Status |
|---|---:|---|
| **UNPACK + EX2** | **1.85** | **INDEPENDENT** (ALU vs XU) |
| **PACK + EX2** | **1.61** | INDEPENDENT with slight dispatch coupling |
| UNPACK + RSQ | 1.24 | Partial (RSQ uses ALU too — it's a compound op) |
| PACK + RSQ | 1.45 | Partial (same reason) |
| UNPACK + FFMA | 1.31 | Partial (dispatch coupling) |
| PACK + FFMA | 1.11 | Partial/near-shared |
| EX2 + FFMA | 1.54 | Partial |

## What this confirms vs overturns

**Confirms (with proper methodology):**
1. **F2FP (ALU pipe) and MUFU.EX2 (XU pipe) are truly independent.**
   u = 1.85 for UNPACK+EX2, 1.61 for PACK+EX2.
2. **F2FP and MUFU.RSQ partially share.** RSQ is a compound op that uses
   ALU (per NCU metric) — it contends with F2FP on ALU.
3. **Same-pipe controls validate the methodology** (u=1.00 for UNPACK+PACK,
   EX2+RSQ).

**Overturns (or qualifies) earlier claims:**
- Sub-agent 1's "SFU is 2-slot, F2FP and MUFU share" was too coarse.
  Correct picture: F2FP on ALU, MUFU.EX2 on XU (independent); MUFU.RSQ
  compound uses ALU + XU so shares with F2FP via ALU.

## Dispatch vs pipe contention (PACK + FFMA)

PACK at 21/SM/clk + FFMA at 123/SM/clk, balanced with N_P=2, N_F=12:
measured PACK 12, FFMA 70. u = 12/21 + 70/123 = 0.57 + 0.57 = 1.14.

This is surprising — if truly independent, u should be 2. Possible reasons:
- FFMA pipe also accepts some ops that compete with PACK's FMA-pipe input
  (IMAD is on fmaheavy per NCU — maybe PACK uses a fmaheavy or shared resource)
- Scheduler dispatch saturation: at ~140 thread-ops/SM/cy we may hit the
  aggregate scheduler ceiling as seen in prior reports.

## Open question: why PACK/UNPACK + FFMA partial share?

Sub-agent 4 is currently running a full pipe-topology investigation
including LOP3, IADD3, IMAD, SHL contention, to answer this.

## Session count

15 reports. Key finding that F2FP.UNPACK+MUFU.EX2 are INDEPENDENT (u=1.85)
supersedes the earlier coarse "share SFU" claim. The 2-slot SFU was a
functional approximation; the true topology has ALU and XU as separate pipes
with MUFU.RSQ being a compound op that touches both.
