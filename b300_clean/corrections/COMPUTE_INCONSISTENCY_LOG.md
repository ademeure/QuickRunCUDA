# Compute (FP32/FFMA/FP64/IMAD) Inconsistency Log — 2026-04-22

Cross-file inconsistencies discovered while producing
`04_fp32_peak_CORRECTED.md`. Originals NOT modified.

## A. Headline FFMA peak — three numbers disagree

| File | Number | Clock | Kernel pattern |
|---|---:|---|---|
| `B300_TRUE_REFERENCE.md` (line 44) | 74.62 TFLOPS = 96.92% | 2032 boost | NCHAIN=3 rotating, imm const |
| `V8_FFMA_PEAK_VERIFIED.md` | 75.2 TFLOPS = 97.64% | 2032 boost | 2-source `fma %0,%0,%1,%0`, 8 ILP |
| `04_fp32_peak.md` (TL;DR) | 75.9 TFLOPS = 98.7% | 2032 boost | `fp32_peak_definitive.cu`, ILP=8 BS=1024 MB=6 |
| `05_fp_precision_nontensor.md` §1 | 71.8 TFLOPS = 93% | unstated (likely 1920) | catalog l.1134 |

**Resolution**: 71.8 is a 1920-MHz number being divided by 2032-MHz theoretical; that's
an arithmetic bookkeeping bug. The other three are within 1.5% measurement noise; pick
75.9 (highest, most rigorous methodology) as canonical headline.

## B. "154 TFLOPS / 256 cores per SM" — looked, found NONE in the corpus

CLAUDE.md warns about this. Greppped `04_fp32_peak.md`, `05_*`, V8_*, A1, A4, A6, D6,
B1, B2, V32-V48 findings, B300_TRUE_REFERENCE: NO file currently asserts 154 TFLOPS or
256 FP32 cores/SM. The retraction is already explicit in `04_fp32_peak.md` "RETIRED"
table line 216. Good — but external sub-agent outputs frequently regenerate the error,
so the warning in CLAUDE.md must stay.

## C. IMAD throughput — "1:1 vs FP32" historically claimed, now 1:2

`V8_IMAD_PEAK_VERIFIED.md` corrects an earlier (in-document) wrong assumption that IMAD
runs at FP32 rate. The correct value is **38.5 Tops = 1:2 of FP32**, per CUDA C
Programming Guide Table 13-1, and measured at 99.7% of that. Any older catalog entry
showing IMAD ~76 TFLOPS-equiv is wrong.

## D. FP64 DFMA — 0.95 vs 1.20 TFLOPS

`05_fp_precision_nontensor.md` §2.2 already resolves this: 0.95 was 1920-MHz +
under-warp-saturated; 1.20 TFLOPS = 100% at 2032 MHz with ≥4 warps/SM
(`V8_FP64_PEAK_VERIFIED.md`). Headline should be **1.20 TFLOPS** everywhere.

## E. IADD3 pipe placement — disagreement

| File | IADD3 lives on |
|---|---|
| V40 / `V32_V40_FINDINGS.md` | **FMA pipe** (25-26 Glane/s, top tier) |
| A6 `A6_PER_PIPE_REFERENCE.md` | "ALU (unified)" (lumped with LOP3, IADD3 at 0.5/SMSP/cy) |
| B1 `B1_DUAL_ISSUE_FFMA_IADD3.md` | "ALU pipe" (14.13 TIPS_inst at 1500) |
| A1 `A1_DUAL_ISSUE_RIGOR.md` | "Cluster B: LOP3 + IADD3 + SHF + PRMT" |

**Resolution**: V40 is most direct (per-op rate sweep, 67% of FMA-pipe ceiling for
IADD3 vs 48% for LOP3 = different pipes). Earlier A1/A6/B1 used "ALU/Cluster B"
nomenclature that conflated IADD3 (FMA-pipe-class) with LOP3 (INT-bit pipe). The
V40 measurement is decisive: IADD3 = FMA-pipe rate, NOT LOP3 rate.

## F. Dual-issue efficiency — multiple disagreeing numbers

| File | Pattern | Overlap |
|---|---|---|
| Catalog `a0bde33` (older) | "IADD3 free w/ FMA" | implied 100% |
| Catalog `f578755` | "dual-issue confirmed" | 100% implied |
| B1 `B1_DUAL_ISSUE_FFMA_IADD3.md` | FFMA+IADD3, NC=8 | **17%** |
| A1 `A1_DUAL_ISSUE_RIGOR.md` | FFMA+LOP3 single-warp | 6.5% |
| A1 | FFMA+IMAD same-cluster | **-14% (slower than serial)** |
| V49 `501134a` | FFMA+LOP3 same-warp | **55%** |
| V49 | FFMA+IADD3 same-warp | **54%** |
| V50 `fbe1c18` | FFMA+LOP3 warp-specialized | **74%** |
| A6 | FFMA + MUFU | ~100% |
| B2 | FFMA + LDG (chained) | 1% |
| B2 | FFMA + LDG (no chain) | 12% |
| Catalog | "FFMA2+ALU = 116 inst/clk/SM 2× FFMA2" (`d2e3212`) | implied perfect |

**Resolution**: Older "free" / "100%" claims (`a0bde33`, `f578755`, `d2e3212`) are
WRONG for the general case. The empirical truth (V49/V50/B1/B2/A1/A6) is:
- Same-pipe-cluster ops: 0% to negative overlap (contend for dispatch).
- Different-pipe ops same warp: ~55% (V49 dispatch slot is shared per-SM).
- Different-pipe ops warp-specialized: ~74% (V50, best practical).
- Slow vs fast pipe (MUFU @ 4 cy + FFMA @ 1 cy): ~100% (gaps swallow MUFU).
- Memory pipe + FFMA (LDG chained): ~1% (LSU back-pressure dominates).

## G. Self-op / RF port pressure — A4/D6 vs older catalog

`A4_FFMA_PORT_PRESSURE.md` and `D6_RF_PORT_RIGOR.md` agree: B300 SMSP has 2 RF read
ports + reuse cache. 1 or 2 unique sources = 0.97/SMSP/cy = peak; 3 unique = 0.61/SMSP/cy
= 65%. Older Volta-era folklore "self-op `Ra,Ra,Ra,RZ` is 2× slower" is wrong on B300:
self-op = diff-2-source = 4.02 cy. (Already retired in `04_fp32_peak.md`.)

But this means **all "near-peak FFMA" headline kernels (74-76 TFLOPS) carefully avoid
3-distinct-source patterns**. A real outer-product GEMM with 3-distinct sources tops
out at ~50 TFLOPS = 65% of theoretical. This caveat is not currently visible in any
top-line summary; it should be added.

## H. FADD vs FFMA same instruction rate

`V8_FADD_FMUL_PEAK.md` confirms FADD/FMUL/FFMA all dispatch at 1 inst/SMSP/cy. The
2× FFMA FLOPS is purely from "2 ops per inst", not from a different pipe. This is
correct everywhere; no inconsistencies.

## I. NO sign of "256 cores", "154 TFLOPS", or FP64 deviation from 1.20

Greppped all listed files. Within this category the catalog is mostly self-consistent;
the issues are (1) clock-state bookkeeping (A above), (2) older sub-agent
"perfect dual-issue" claims that need explicit retraction (F), and (3) IADD3 pipe
placement nomenclature (E).
