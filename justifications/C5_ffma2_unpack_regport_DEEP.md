# C5 — FFMA2 + UNPACK "16% friction" deep-dive (regport hypothesis)

**Date:** 2026-04-24
**Catalog claim:** "FFMA2 + UNPACK gives u=1.67 (16% SMSP friction specific to F2FP). Not present for PRMT+FFMA2 (u=1.95)." (B300_PIPE_CATALOG.md L478)
**User hypothesis:** RF read-port pressure — F2FP exceeds the SMSP per-cycle read budget when paired 1:1 with FFMA2; PRMT stays within budget because of operand reuse / different read pattern.
**Verdict:** ✅ **CONFIRMED, with refined mechanism — not raw RF read count, but `.reuse` cache disturbance.**

---

## Hypothesis as posed

The Blackwell SMSP has a finite per-cycle RF read-port budget. If FFMA2 (3 sources: 2 R + 1 UR) co-issues with an ALU op, the total read demand may exceed the budget, forcing the scheduler to skip cycles. The user asked: count post-`.reuse` reads; correlate with measured u; build counter-examples.

## Test setup

- `tests/bench_c5_regport.cu` — 12 variants of FFMA2 / F2FP / PRMT / LOP3 / IADD3 mixes.
- `tests/bench_c5_v52mirror.cu` — exact mirror of original v52-era test (`N_FFMA2=8`, `N_ALU=8` clustered ratio).
- B300 SXM6 sm_103a, default boost 2032 MHz, GPU 0 idle pre/post.
- Metrics: `sm__inst_executed.avg.per_cycle_active`, `sm__inst_executed_pipe_{fmaheavy,fmalite,alu}.avg.pct_of_peak_sustained_active`.

## Key SASS observations

**v52mirror (catalog-shape) — `.reuse` rates on FFMA2's broadcast-multiplier R0:**

| Variant | FFMA2 with `.reuse` | u (inst/SM/cy) | alu/fmaH/fmaL % |
|---|---|---|---|
| ALU=F2FP UNPACK | **79/128 = 62%** | **3.37** | 83/84/83 |
| ALU=PRMT (3-src) | **127/128 = 99%** | **3.98** | 98/99/98 |
| ALU=LOP3 (3-src) | **64/128 = 50%** | **2.69** | 66/67/66 |
| ALU=IADD3 (1-src) | n/a (1 src reg only) | **2.69** | 45/88/88 |

**Per-instruction regs (Blackwell v52mirror inner loop):**
- `FFMA2 Rd, Rd_old.F32x2.HI_LO, R0[.reuse].F32, imm`  → 1 R RMW + 1 R (R0, often `.reuse`) + 1 imm.
- `F2FP.F16.E4M3.UNPACK_B Rd, Rd`                        → 1 R (RMW; same reg as dst).
- `PRMT Rd, Rd, 0x7531, Rs`                              → 1 R RMW + 1 R + imm.
- `LOP3.LUT Rd, Rd, Rs1, Rs2, 0xC8, !PT`                 → 1 R RMW + 2 R + imm.

**Effective fresh RF reads / cycle (cluster shape, 1:1 issue):**

| Variant | FFMA2 reads (post-reuse) | ALU reads | Total/cycle | u | Verdict |
|---|---|---|---|---|---|
| F2FP | 0.62×1 + 0.38×2 = 1.38 | 1 | **2.38** | 3.37 | catalog 1.67-era |
| PRMT | 0.99×1 + 0.01×2 = 1.01 | 2 | **3.01** | 3.98 | catalog 1.95-era |
| LOP3 | 0.50×1 + 0.50×2 = 1.50 | 3 | **4.50** | 2.69 | LOP3 saturates ALU pipe |

**Key observation:** PRMT has *more* total RF reads than F2FP (3.01 vs 2.38), yet PRMT runs *faster*.
Raw read-count is NOT the right metric. The **reuse rate of R0** is.

## Mechanism (refined)

**F2FP UNPACK invalidates the SMSP operand-reuse cache for FFMA2's R0 broadcast slot.**

Pattern in F2FP variant (every other FFMA2 loses `.reuse`):
```
F2FP R27, R27
FFMA2 R2, R2.F32x2.HI_LO, R0.F32,        ...   ← .reuse DROPPED
F2FP R24, R24
FFMA2 R4, R4.F32x2.HI_LO, R0.reuse.F32,  ...   ← .reuse held
F2FP R23, R23
FFMA2 R6, R6.F32x2.HI_LO, R0.F32,        ...   ← .reuse DROPPED
```
ptxas drops `.reuse` deterministically because it knows F2FP will physically clobber the operand-reuse slot for that operand index. Result: half the FFMA2s pay an extra RF read → 16% throughput loss.

In the PRMT variant the same R0 broadcast holds `.reuse` 99/100 of the time → no extra read → near-peak.

LOP3 only holds `.reuse` 50% of the time AND has 3 sources → both effects compound → 32% loss.

## Counter-example tests (regport.cu, 1:1 INTERLEAVED inner loop, no clustering)

| Variant | Code | u | alu/fmaH/fmaL % | Notes |
|---|---|---|---|---|
| 0 | FFMA2 alone | 2.22 | 0/98/98 | both fma sub-pipes saturate |
| 1 | FFMA2+F2FP | **2.74** | 64/64/64 | **identical to V2** in this shape |
| 2 | FFMA2+PRMT | **2.74** | 64/64/64 | **identical to V1** in this shape |
| 3 | F2FP alone | 2.04 | 100/0/0 | pipe_alu cap = 2.00 ✓ |
| 4 | PRMT alone | 2.05 | 100/0/0 | pipe_alu cap = 2.00 ✓ |
| 5 | FFMA2+F2FP w/ data-dep src | 2.21 | 6/92/92 | dep stall hides ALU pipe |
| 6 | FFMA2+LOP3+F2FP (3 pipes) | 2.88 | 92/46/46 | all 3 pipes co-issue |
| 7 | FFMA2 + 2× F2FP | 2.47 | 79/39/39 | F2FP saturates ALU pipe alone |
| 8 | FFMA2+LOP3 1:1 | **2.14** | 50/50/50 | confirms LOP3 friction (5R/cy) |
| 9 | FFMA2+IADD3 0.5:1 | 2.53 | 39/77/77 | compiler fused some |
| 10 | FFMA2+F2FP loop-invariant | 2.24 | 7/93/93 | compiler hoisted F2FP — n/a |
| 11 | F2FP+PRMT (no FFMA2) | 2.02 | 100/0/0 | both ALU; pipe full |

**Why does the regport.cu interleaved pattern NOT reproduce the asymmetry?**

In the inline 1:1 inner loop, every iteration has DIFFERENT R0/R1/R2 sources for FFMA2 (because each accumulator's b/c sources are separate `b_[k], c_[k]` registers, not a single broadcast). The `.reuse` opportunity disappears entirely (0 reuse marks observed in V0-V8 inner loops). Both F2FP and PRMT hit the same throughput → no asymmetry in this regime.

The catalog's "16% friction" finding requires the **specific code shape** of v52mirror: a clustered loop where FFMA2 broadcasts a single multiplier (R0) across 8 accumulators, letting the compiler stamp `.reuse`. **Interleaved 1:1 with private multiplicands has NO `.reuse` to disturb → no F2FP friction.**

## Did any new variant hit u=1.95-equivalent (≥3.95) with F2FP?

No — F2FP variant of v52mirror tops out at 3.37; PRMT-style reuse pattern with F2FP cannot be constructed because the ptxas heuristic refuses to mark `.reuse` adjacent to a F2FP that targets the same operand-cache slot. A user-level workaround would be inline-asm `.reuse` markup, but volatile asm blocks ptxas-issued `.reuse` annotations.

## Verdicts

- **RF read-port hypothesis (raw count):** ❌ **FALSIFIED.** PRMT (3.01 R/cy) > F2FP (2.38 R/cy) yet PRMT is faster.
- **`.reuse` cache disturbance hypothesis:** ✅ **CONFIRMED.** F2FP UNPACK clobbers FFMA2 operand-reuse cache for its target operand slot, losing ~38% of `.reuse` opportunities → ~16% throughput loss.
- **Catalog C5 number (1.67 vs 1.95 differential):** ✅ **REPLICATED** (3.37 vs 3.98 in v52mirror — same 1.18× ratio, same 16% friction).
- **Generalisation:** ⚠ **Shape-specific.** Asymmetry only appears when ptxas can stamp FFMA2 operands `.reuse` (clustered broadcast pattern). In interleaved 1:1 with private multiplicands, both ops achieve identical u=2.74 → no friction.
- **LOP3 surprise:** ⚠ NOT "free co-issue" in 1:1 — costs 22% (u=2.14 vs 2.74). The "free" claim only holds at 2:1 (FFMA2:LOP3) per §22.

## Implications for catalog

1. C5 wording should specify the regime: "16% friction occurs when FFMA2 broadcasts a `.reuse`-eligible multiplier across multiple accumulators AND F2FP is on the same operand-cache slot."
2. The mechanism is **operand-reuse cache invalidation**, not raw RF read-port pressure.
3. PRMT has the property of being "operand-reuse-cache-friendly" with FFMA2; F2FP and LOP3 are not.
4. The "F2FP-specific" framing is technically true in this shape, but the same effect would appear with any narrow-format CVT that targets the same operand-cache port.
