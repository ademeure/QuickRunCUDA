# 15 — Integer / Bit-Op Throughput & Pipe Placement (CORRECTED)

Date: 2026-04-22.

Cross-references audited:
- `15_integer_bit_ops.md` (catalog)
- `V9_INT_OPS_PIPES.md`, `V9_MIXED_PIPES.md` (early V9 pipe sweep)
- `C3_LOP3_LUT_DEEP.md`
- `D9_E4_LDG_ATOM_SASS.md` (atomic SASS, no INT pipe content)
- `V41_V48_FINDINGS.md` (V40 ALU pipe ladder)
- `A6_PER_PIPE_REFERENCE.md` (per-pipe instruction reference)
- `B1_DUAL_ISSUE_FFMA_IADD3.md`
- `V8_IMAD_PEAK_VERIFIED.md`
- `04_fp32_peak_CORRECTED.md`

V40 (commits leading up to 7647eba / V41-V48 doc) supersedes the
earlier V9 pipe naming for B300 SXM6.

---

## 1. CORRECTED pipe-placement table

V40 measured ALU-pipe ops at 1500 MHz lock with persistent grid +
asm-volatile anti-DCE. Numbers are **Glane/s** (chip-wide
thread-instructions/sec, = warp-inst/cy/SMSP × 32 lanes × 4 SMSPs ×
148 SMs × clock). "%SoL" is vs an FFMA-pipe peak of ~38.5 Glane/s (1
inst/SMSP/cy at 1920/2032). At 1500 MHz lock the FFMA-pipe SoL itself
is ~28.4 Glane/s/inst.

| Op | Pipe per V40 (B300) | Glane/s @ 1500 lock | %SoL of FFMA pipe | inst/SMSP/cy | Notes |
|---|---|---:|---:|---:|---|
| **FFMA / FADD / FMUL** | **FMA** | 25–26 | **67%** (not 100%) | ~0.66 | Solo FMA pipe peak (V40, A6) |
| **IADD3** | **FMA** (V40) | 25–26 | **67%** | ~0.66 | NOT a separate ALU pipe; shares FMA-pipe issue slot |
| **IMAD / IMUL (32-bit, .lo)** | **FMA** (V8/V40) | 18.7 | **48%** (~half of FMA peak) | 0.5 | "INT-bit half-rate" tier |
| **LOP3.LUT** | **INT-bit** (V40); imm-independent (C3) | 18.7 | **48%** | 0.5 | C3 verified across 12 truth-tables; ≥3 unique reads no penalty |
| **PRMT** (byte permute) | **permute** (V40) | 13.9 | **36%** | ~0.46 | V39 LICM-fixed; original 1547% bogus retired |
| **SHF.L/R / SHL / SHR** | INT-bit (A6) | 14.12 (A6) | ~48% | 0.5 | Same tier as LOP3 |
| **BFI.b32** | INT-bit (A6) | 13.15 | ~46% | 0.46 | folds to LOP3.LUT in many cases |
| **ISETP / FSETP** | **compare** (V40) | 8.4 | **22%** | 0.25 | Lower tier than LOP3/PRMT; do NOT lump into "ALU @ 19 TIOPS" |
| **BFE.u32** | XU (A6) | 7.07 | ~25% | 0.25 | 2-SASS path (SHF.R + SGXT) |
| **SHFL.{IDX,BFLY,UP,DOWN}** | LSU/SHFL pipe | 7.06 | ~25% | 0.25 | A6 single-source verified |
| **POPC / BREV / CLZ / FLO** | XU | 3.5 | ~12% | 0.125 | 4× slower than LOP3 tier |
| **MUFU.EX2** | MUFU (XU) | 9.62 Gops/s | — | 0.003 | Stands out at 95.8% of 1/(4 cy)/SMSP per V41 |
| **MUFU.{LG2,RCP,RSQRT,SQRT,SIN,COS}** | MUFU | 4.74 Gops/s | — | 0.0015 | Half rate of EX2 (V41) |
| **REDUX / SHFL** (V37/V38) | shuffle pipe | 9.0–9.5 Telements/s | — | — | Same pipe; "redux 4× SHFL" was algorithmic |

### What "INT-bit at half rate" means in practice

V40's "INT-bit pipe at half rate" is a description of measured
throughput (0.5 inst/SMSP/cy) relative to the FMA pipe (1 inst/SMSP/cy
SoL). Whether this is "FMA-pipe-with-half-rate-cycle-allocation",
a "separate physical pipe whose dispatch slot opens every other
cycle", or a shared dispatch port that the scheduler hands LOP3 every
2 cycles, V40 + A6 cannot disambiguate. See UNRESOLVED below.

---

## 2. RETRACTIONS (wrong pipe placements / inflated numbers)

| Claim | Source | Correct value | Why wrong |
|---|---|---|---|
| **"IADD hits ALU pipe at 99.94% (separate from FMA pipe)"** | `V9_INT_OPS_PIPES.md` headlines + `V9_MIXED_PIPES.md` | IADD3 is on the **FMA pipe** at 0.66 inst/SMSP/cy ≈ 67% (V40, A6). The "ALU/FMA unified cluster" model in A6 is the corrected story. | V9 used ncu `pipe_alu` % which on B300 is partially aliased to a unified ALU+FMA dispatch slot; the "separate pipe" reading was wrong. |
| **"Mixed FFMA + IADD = 114 TOPS combined" (then "74 TOPS")** | `V9_INT_OPS_PIPES.md` then partially retracted by `V9_MIXED_PIPES.md` | NEITHER. FFMA + IADD3 measured overlap 14.2% (A6) / 17% (B1). Mixed throughput plateaus near solo FFMA peak. | Both were wrong: original 114 was a sum-of-peaks formula, the 131%-pipe-sum number was a misread of unified-cluster dual-counting. |
| **"All fast integer ops cap at 2 warp-inst/SM/cy on pipe_alu (~19 TIOPS)" — applies to LOP3, PRMT, SHF, IMAD, IMUL, IADD3, ISETP, FSETP, IMNMX, BFI** | `15_integer_bit_ops.md` §Key facts #1 | Tiered, NOT uniform. V40: IADD3 at 67% (FMA pipe), LOP3/IMUL at 48%, PRMT at 36%, ISETP at 22%. The "all converge at 19 TIOPS" reading is true ONLY for LOP3/SHF/PRMT-class ops, NOT for IADD3 (faster) or ISETP (slower). | V40 ladder explicitly disproves this. |
| **"IADD3 = 2.46 w-inst/SM/cy (25% faster than LOP3)"** | `15_integer_bit_ops.md` row 14, key fact #2 | IADD3 is faster than LOP3 because it sits on the FMA pipe (1 inst/SMSP/cy SoL ≈ 2/SM/cy effective vs LOP3's 0.5/SMSP/cy = 2/SM/cy nominal). The 2.46 w-inst/SM/cy figure conflates "logical adds" with "instruction count". V40: IADD3 at 25–26 Glane/s; LOP3 at 18.7 Glane/s — that is 1.34× speedup, NOT 1.25×. | Catalog mixed inst-count and op-count counters. |
| **"PRMT = 1547% of SoL"** (not in catalog but in raw V39 first pass) | V39 `bench_prmt_*` first-pass log | After LICM fix: 13.9 Glane/s = **48% of FFMA pipe peak**. | Constant operand was hoisted out of the loop, leaving an empty body. |
| **"ISETP = pipe_alu @ 2.00/SM/cy (~19 TIOPS chip)"** | `15_integer_bit_ops.md` row 26, key fact #11 | ISETP runs at **0.25 inst/SMSP/cy = 22% of FMA pipe peak ≈ 8.4 Glane/s** per V40. The catalog conflates ISETP with LOP3/PRMT. | Catalog assigned all "ALU" ops to one rate; V40's compare-pipe ladder shows ISETP is in its own slower tier. |
| **"setp.lt.f32 ≈ setp.lt.s32 ≈ setp.lt.u32 all at pipe_alu @ 2.00/SM/cy"** | `15_integer_bit_ops.md` key fact #11 | The SASS-equivalence is correct (FSETP/ISETP) but the rate is 0.25 inst/SMSP/cy per V40, not 2.0/SM/cy. | Same root cause as ISETP retraction. |
| **"shfl.idx with literal 0 src = 85 K Gops/s"** | shfl_bw.cu sub-agent, retired in catalog | Already retired in `15_integer_bit_ops.md`. Note correctly: this is uniform-pipe broadcast (R2UR / UIMOV), not a SHFL. | (Already retired — kept here for cross-ref.) |
| **"IMAD at same rate as FP32"** (V8 initial assumption before correction) | V8 first-pass | IMAD is 1:2 of FP32 = 38.4 Tops at 2032 boost = 99.7% of corrected theoretical (V8 already self-corrected). | Listed for completeness — the corrected V8 number stands. |

---

## 3. UNRESOLVED

1. **Is "INT-bit at half rate" a separate physical pipe, or a
   half-rate slot on the FMA pipe?** V40 calls LOP3/IMUL "INT-bit
   half rate" but does not show whether (a) it is the FMA pipe
   issuing LOP3 every 2 cycles, (b) a shared dispatch port between
   LOP3 and IMUL with 0.5/SMSP/cy throughput, or (c) a dedicated
   physical INT-bit pipe whose native cycle is 2 clocks. Direct
   FFMA + LOP3 dual-issue measurements would discriminate; A6 only
   has FFMA + IADD3 (14.2% overlap) and FFMA + SHFL.
2. **PRMT @ 36% — separate "permute" pipe or further-throttled
   INT-bit?** V40 lists PRMT below LOP3 (13.9 vs 18.7). A6 lists
   PRMT at 14.08 = 0.5/SMSP/cy = same tier as LOP3. The two
   measurements disagree (V40 implies ~0.36/SMSP/cy effective; A6
   says 0.5). Likely V40 ran PRMT under different ILP/op-mix
   conditions; needs A6-style port-pressure sweep on PRMT.
3. **ISETP @ 22% — explanation?** V40 places ISETP on a "compare"
   sub-pipe at 0.25/SMSP/cy. CUDA C PG does not list a per-SM
   ISETP rate. C3-style sweep over ISETP imm/predicate combinations
   would confirm whether the 22% figure is fundamental or methodology.
4. **IADD3 at 67% (V40) vs 50% (A6/B1).** V40 claims IADD3 sits
   with FFMA at 25–26 Glane/s = 67% of FMA pipe SoL. A6 / B1 measure
   IADD3 at 14.13 TIPS_inst = 0.50/SMSP/cy at 1500 MHz lock.
   25 Glane/s @ 2032 implies ~0.66/SMSP/cy. The discrepancy may be
   ILP/warp-count: A6 used 2 warps/SMSP, V40 used full persistent
   block. **Resolution requires re-running A6 IADD3 sweep with 4+
   warps/SMSP to confirm IADD3 closes to FMA-pipe peak.**
5. **REPORT_06 8×8 BASE×COMPANION matrix referenced in CLAUDE.md
   memory** — not found in `b300_clean/`. The memory says it shows
   "IMAD is on FMA pipe (not INT)" — already consistent with V8 and
   V40. If REPORT_06 exists outside `b300_clean/` it should be
   incorporated.
6. **IADD3 "fuses 2 adds into 1 SASS" double-counting.** B1 says
   `add.s32; add.s32` fuses to a single IADD3, doubling the
   effective add rate. The catalog's 2.46 w-inst/SM/cy folds this
   into a per-instruction rate. A clean per-add-vs-per-inst
   reporting convention is needed.

---

## 4. Re-stated headline numbers (use these)

At **1920 MHz locked** (CLAUDE.md "default `-lgc 2032` paradox"):

| Op | Glane/s (chip) | Pipe |
|---|---:|---|
| FFMA (FMA pipe peak) | 36–38 (97% × 38.5) | FMA |
| IADD3 | 25–26 | FMA (V40) — but see UNRESOLVED #4 |
| LOP3 / IMUL | 18.7 | INT-bit (V40) |
| PRMT | 13.9 | permute (V40) — A6 disagrees, see UNRESOLVED #2 |
| ISETP | 8.4 | compare (V40) |
| POPC / BREV / CLZ | 4.7 | XU |
| SHFL.IDX | 4.7 | LSU/SHFL |
| EX2 | 9.62 Gops/s | MUFU |
| Other MUFU | 4.74 Gops/s | MUFU |

Multiply by 1.058 for 2032 MHz boost.

---

## 5. Files

Originals (do not edit):
- `b300_clean/15_integer_bit_ops.md`
- `b300_clean/V9_INT_OPS_PIPES.md`
- `b300_clean/V9_MIXED_PIPES.md`
- `b300_clean/C3_LOP3_LUT_DEEP.md`
- `b300_clean/V41_V48_FINDINGS.md` (V40 ladder)
- `b300_clean/A6_PER_PIPE_REFERENCE.md`
- `b300_clean/B1_DUAL_ISSUE_FFMA_IADD3.md`
- `b300_clean/V8_IMAD_PEAK_VERIFIED.md`
- `b300_clean/D9_E4_LDG_ATOM_SASS.md` (no INT-pipe content; LDG/ATOM only)

Companion log: `INT_INCONSISTENCY_LOG.md`.
