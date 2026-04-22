# Integer / Bit-Op Inconsistency Log

Cross-file audit, Topic = INT and bit-op pipe placement & throughput.
Sources audited:
- `15_integer_bit_ops.md` (catalog)
- `V9_INT_OPS_PIPES.md`, `V9_MIXED_PIPES.md`
- `C3_LOP3_LUT_DEEP.md`
- `V41_V48_FINDINGS.md` (V40 ladder)
- `A6_PER_PIPE_REFERENCE.md`
- `B1_DUAL_ISSUE_FFMA_IADD3.md`
- `V8_IMAD_PEAK_VERIFIED.md`
- `D9_E4_LDG_ATOM_SASS.md` (no INT-pipe content found)
- `B300_TRUE_REFERENCE.md` (no INT lines)
- `CLAUDE.md` memory matrix mention of REPORT_06

---

## A. Pipe-placement inconsistencies for IADD3

| File | Pipe claim | Rate claim |
|---|---|---|
| `V9_INT_OPS_PIPES.md` | **"ALU pipe (separate from FMA pipe)"** at 99.94% pipe_alu | "~38 TOPS = full ALU pipe peak" |
| `V9_MIXED_PIPES.md` | implicit "ALU pipe; co-issuable with FMA" but SUM stalls at 131% (74 TOPS) | "Cannot stack peaks for 2× throughput" |
| `15_integer_bit_ops.md` | "alu (+ split fmaH)" — straddles two pipes | 2.46 w-inst/SM/cy = 158 Glane/s/SM (catalog headline) |
| `A6_PER_PIPE_REFERENCE.md` | **"ALU (unified)" — unified ALU/FMA cluster** | 14.13 TIPS_inst @ 1500 = 0.50 inst/SMSP/cy |
| `B1_DUAL_ISSUE_FFMA_IADD3.md` | "ALU pipe; nvcc fuses add;add to single IADD3" | 14.13 TIPS_inst, 17% overlap with FFMA |
| `V41_V48_FINDINGS.md` (V40) | **"FMA pipe"** — IADD3 on the same pipe as FFMA/FADD | 25–26 Glane/s = 67% of FMA pipe SoL |

**Top-level inconsistency:** V9 says IADD3 is on a separate ALU pipe;
A6 says it's on a "unified ALU/FMA cluster"; V40 says it's flat-out
on the FMA pipe. The unified-cluster framing in A6 is the most
honest reading and is consistent with V40 (whether you call the
shared dispatch slot "FMA pipe" or "unified" is naming, not physics).
The "separate ALU pipe" framing in V9 is wrong.

**Throughput inconsistency #1:** A6/B1 measure IADD3 at 0.50 inst/SMSP/cy
(half of FMA pipe peak). V40 implies IADD3 hits 0.66 inst/SMSP/cy
(same tier as FFMA at 67%). Different ILP/warp configs may explain.
Needs A6-style sweep with 4+ warps/SMSP to resolve.

**Throughput inconsistency #2 (logical adds vs instructions):**
B1 explicitly notes nvcc fuses `add.s32; add.s32` → 1 IADD3, so per-add
rate is 2× per-inst rate. The catalog's "2.46 w-inst/SM/cy" headline
is the per-add count, not per-inst. The "25% faster than LOP3" claim
in §2 of `15_integer_bit_ops.md` is built on this conflation; the
true per-inst gap is V40's 25/18.7 = 1.34×.

---

## B. LOP3 throughput

| File | Rate |
|---|---|
| `C3_LOP3_LUT_DEEP.md` | 14.16 TIPS_inst @ 1500 = 0.50/SMSP/cy = 19.18 TIPS @ 2032 (imm-independent, port-pressure-free) |
| `A6_PER_PIPE_REFERENCE.md` | 14.16 TIPS_inst @ 1500 = 0.50/SMSP/cy ("ALU (unified)") |
| `15_integer_bit_ops.md` | 2.00 w-inst/SM/cy = 128 Gops/s/SM = 19.0 TIOPS chip ("alu @ 4 cy") |
| `V41_V48_FINDINGS.md` V40 | 18.7 Glane/s = 48% of FMA pipe SoL ("INT-bit half rate") |

**Consistent:** all four agree on ~19 TIOPS @ 2032 MHz. Naming differs:
"ALU pipe", "ALU (unified)", "INT-bit half rate". V40's framing is
the cleanest because it makes the relative-to-FMA-pipe ratio explicit.

**Inconsistency:** the catalog claims LOP3 reaches 1 inst/SMSP/cy
"on pipe_alu @ 2.00/SM/cy" (= 0.5/SMSP/cy effective per-SMSP because
2/SM/cy ÷ 4 SMSPs = 0.5/SMSP/cy). A6 and C3 both confirm 0.5/SMSP/cy.
**No claim of full 1 inst/cy/SMSP for LOP3 was found in the audited
files** — V40 says LOP3 is 48% of FMA pipe peak, which is 0.5/SMSP/cy
relative to FMA's 1.0/SMSP/cy. So the audit goal "look for LOP3 full
rate claims" finds NONE in `b300_clean/` (would need to check older
B300_PIPE_CATALOG / EXTENDED_FINDINGS in repo root if they survive).

---

## C. PRMT throughput

| File | Rate |
|---|---|
| `A6_PER_PIPE_REFERENCE.md` | 14.08 TIPS_inst @ 1500 = 0.50/SMSP/cy ("ALU") = ~19 TIPS @ 2032 |
| `15_integer_bit_ops.md` | 2.00 w-inst/SM/cy = ~19 TIOPS chip ("alu") |
| `V41_V48_FINDINGS.md` V40 | **13.9 Glane/s = 36% of FMA pipe SoL ("permute" pipe)** |
| `V41_V48_FINDINGS.md` V39 (raw) | 1547% (DCE/LICM artifact, retired before publication) |

**Inconsistency:** A6 and catalog claim PRMT = LOP3 = 0.5/SMSP/cy
(~19 TIPS). V40 explicitly carves PRMT out into a "permute" pipe at
13.9 Glane/s = ~36% of FMA peak vs LOP3's 18.7 Glane/s = 48%.
A6 and V40 measurements disagree by ~30%. The likely explanation
is methodology (V40 may be a tighter bound; A6 may have slightly
favorable conditions). Needs a clean re-test with both LOP3 and PRMT
under matched ILP/warp count to resolve.

**Retraction noted:** V39's first-pass 1547% was a constant-fold/
LICM artifact; the V40 number is post-fix.

---

## D. ISETP throughput

| File | Rate |
|---|---|
| `15_integer_bit_ops.md` row 26 + key fact #11 | 2.00 w-inst/SM/cy ("pipe_alu") = ~19 TIOPS chip |
| `V41_V48_FINDINGS.md` V40 | **8.4 Glane/s = 22% of FMA pipe peak ("compare" pipe)** |
| `CURIOSITY_LIST_V4.md` C10 | "ISETP ≈ 4.6 cy/op chained — same magnitude as LOP3" (latency, not throughput) |

**Inconsistency:** catalog puts ISETP in the same tier as LOP3 (~19
TIOPS). V40 measures ISETP at less than half the LOP3 rate (8.4 vs
18.7 Glane/s). C10's "same magnitude as LOP3" is a *latency* claim
(both ~4 cy chained) — does not contradict V40's *throughput* claim.

The catalog's "all setp variants are equally fast" is correct as a
relative claim (FSETP ≈ ISETP at SASS level), but the absolute
throughput number it cites (~19 TIOPS) is wrong per V40.

---

## E. IMAD / IMUL

| File | Pipe | Rate |
|---|---|---|
| `V8_IMAD_PEAK_VERIFIED.md` | **FMA pipe** (1:2 of FFMA) | 19.18 GIMAD/s = 38.4 Tops = 99.7% of true peak |
| `V9_INT_OPS_PIPES.md` | "FMA pipe at 49.81%" (1:2 vs FP32) | matches V8 |
| `15_integer_bit_ops.md` | "fmaH @ 2.00/SM/cy" | matches V8 |
| `V41_V48_FINDINGS.md` V40 | **"INT-bit (half rate)"** at 18.7 Glane/s | matches V8 in numbers but disagrees in pipe label |

**Mostly consistent:** all four agree on ~19 G IMAD/s @ 2032 MHz =
half of FFMA. V40's "INT-bit" label and V8/V9's "FMA pipe at 1:2"
label point to the same physical fact (whatever you call the
half-rate slot).

**Memory matrix claim ("REPORT_06: IMAD on FMA pipe not INT")** —
all `b300_clean/` sources support "IMAD lives in the FMA-pipe family"
either as direct claim (V8/V9) or as the half-rate slot of the
unified cluster (V40 / A6). No file places IMAD on a "pure INT" pipe,
so the memory's claim is consistent with the corpus.

---

## F. POPC / BREV / CLZ

Consistent across `15_integer_bit_ops.md` (XU @ 0.5/SM/cy = 4.7 TIOPS)
and `A6_PER_PIPE_REFERENCE.md` (3.54 TIPS @ 1500 = 0.125/SMSP/cy = 4.7
TIPS @ 2032). No inconsistencies found.

---

## G. SHFL

`15_integer_bit_ops.md` says SHFL is on `pipe_lsu` at 32 SASS/SM/cy
= 4.7 Gops chip. `A6_PER_PIPE_REFERENCE.md` says 0.25/SMSP/cy = 7.06
TIPS. **Inconsistency factor of ~1.5×** — likely 32-bit vs 1-warp-per-
SMSP saturation. `V41_V48_FINDINGS.md` V38 says "SHFL peak 9.48
Telements/s" which is even higher than A6 (likely 32 lanes × 0.25/SMSP/cy
× 4 SMSPs × 148 SMs × 2.032 GHz = ~9.6 Telements/s). The element-rate
vs instruction-rate naming clash explains all three numbers being
internally consistent but optically different.

**Retraction reminder:** "shfl.idx with literal 0 src = 85 K Gops/s"
is a uniform-pipe broadcast, not SHFL — already retired in catalog.

---

## H. Mixed-pipe overlap

| File | FFMA + IADD3 overlap |
|---|---|
| `V9_INT_OPS_PIPES.md` | "up to 114 TOPS combined" (formula prediction) |
| `V9_MIXED_PIPES.md` | 131% pipe-sum, ~74 TOPS effective (corrected) |
| `B1_DUAL_ISSUE_FFMA_IADD3.md` | **17% overlap** (1.17× speedup vs sequential) |
| `A6_PER_PIPE_REFERENCE.md` | **14.2% overlap** |

**Inconsistency:** V9's two docs disagree with each other; V9_MIXED's
"74 TOPS effective" differs from B1/A6's 14–17% overlap framing
because V9_MIXED counts pipe-utilization-sum (a different metric).
The B1/A6 14–17% number is the right "wall-clock speedup" headline.

**Bottom line for users:** mixing FFMA + IADD3 gives **~15% wall-clock
benefit**, NOT 2× and NOT 50%. The "114 TOPS combined" claim from
V9_INT_OPS_PIPES is fully retracted by V9_MIXED + B1 + A6.

---

## I. REPORT_06 reference

CLAUDE.md memory mentions: "REPORT_06: 8×8 BASE×COMPANION matrix
showing IMAD is on FMA pipe (not INT)". File not found in
`b300_clean/`. If it exists in repo root or `investigations/`,
it should be folded into `15_integer_bit_ops_CORRECTED.md`.

The claim itself ("IMAD on FMA pipe") is already consistent with
all five `b300_clean/` files that mention IMAD pipe placement (see §E).

---

## Summary of action items

1. Retract V9_INT_OPS_PIPES.md "separate ALU pipe" framing for IADD3
   in favor of A6/V40 unified-cluster / FMA-pipe framing.
2. Retract V9_INT_OPS_PIPES.md "114 TOPS combined" claim entirely.
3. Update `15_integer_bit_ops.md` ISETP/PRMT rows to V40 values
   (8.4 / 13.9 Glane/s) instead of the over-optimistic ~19 TIOPS.
4. Reconcile A6 vs V40 numbers for PRMT (0.5 vs 0.36 inst/SMSP/cy)
   with a re-test.
5. Reconcile A6 vs V40 numbers for IADD3 (0.5 vs 0.66 inst/SMSP/cy)
   with a 4-warp/SMSP A6-style sweep.
6. Locate REPORT_06 in repo if it exists; otherwise stop citing it.
