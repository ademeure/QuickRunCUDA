# SASS verification: V49/V50 dual-issue claim vs V8 FFMA peak

**Date**: 2026-04-22
**Trigger**: Wave-4 META_DOUBT_REPORT claimed V49's measurement bug was "self-RAW
with immediate constants" (`fma %0, %0, imm, imm`) vs V8's "distinct sources"
(`fma %0, %0, %1, %0`). Needs SASS proof.
**Toolchain**: nvcc 13.2 V13.2.78, sm_103a, -O3.

---

## 1. Source-level inline asm

| Kernel | Inline asm | Operands |
|---|---|---|
| **V49 solo / dual FFMA** | `fma.rn.f32 %0, %0, 0f3FC00000, 0f3F000000` (`+f`) | self-feed + 2 IMMEDIATES (1.5f, 0.5f) |
| **V50 OP=0 all FFMA** | identical to V49 | self-feed + 2 IMMEDIATES |
| **V8 (`bench_ffma_warps_per_sm.cu`)** | `fma.rn.f32 %0, %0, %1, %0` (`+f`/`f`) | self-feed + 1 REGISTER `b[k]` (constant per chain) |

The meta-doubt's source-pattern claim is **TRUE at PTX level**.

## 2. SASS for V49 OP=0 / OP=2 / V50 OP=0 (compiled fresh)

All three reduce to the same FFMA encoding:

```
FFMA Rk, Rk, R0.reuse, 0.5 ;   // R0 = 0x3FC00000 = 1.5f, third operand IMM 0.5f
                               // opcode 0x...7423, "0.5" is a literal IMM in the encoding
```

So the SASS is **`FFMA Rd, Rd, Rsrc, IMM`** (2-source: self + register holding 1.5f, with 0.5f as immediate addend). The compiler hoisted the 1.5f immediate into R0 once before the loop (`IMAD.MOV.U32 R0, RZ, RZ, 0x3fc00000`) — it did NOT emit a true 2-immediate FFMA encoding.

V8's `bench_ffma_warps_per_sm` SASS:
```
FFMA Rd, Rsrc1, Rd, Rd ;       // d = src1 * d + d  -- 2 distinct registers
```

So at SASS the actual register-port footprints are:
- **V49/V50 FFMA**: reads `Rd` (self) + `R0` (shared across all 8 ILP chains, .reuse cache hits) + 0.5f IMM. Effectively **1 unique RF read** (Rd), since R0 is hot in the operand reuse cache.
- **V8 FFMA**: reads `Rd` (self, twice as 2nd and 3rd source) + `Rsrc1` (per-chain, distinct register). **2 unique RF reads**, but Rsrc1 is also `.reuse`-able since it's loop-constant.

**Both kernels avoid the 3-distinct-source RF port pressure that capped V6_C1 at 65-71%.**

## 3. The REAL methodology issue (not what meta-doubt claimed)

V49's OP=2 (FFMA+LOP3 dual) compiles to a **tiny inner body**: 8 FFMA + 8 LOP3 + 1 BRA, repeated 5000× via the OUTER loop with `#pragma unroll 1`. Inspecting `_Z8mix_pipeILi2ELi8ELi5000EEvPj`:

```
/*0200*/-/*0270*/   8× FFMA Rk, Rk, R0.reuse, 0.5
/*0280*/-/*02e0*/   8× LOP3.LUT Rk, Rk, 0xa5a5a5a5, R18, 0x96, !PT
/*02f0*/            BRA.U UP0, 0x1b0   ;; tight inner loop branch
```

V8's `bench_ffma_warps_per_sm` unrolls 16 × 8 = **128 FFMAs per outer iteration**,
no INT-pipe ops in between. The branch overhead amortizes ~16× better.

V49 measures the *steady-state issue rate of an FFMA+LOP3+branch loop*, not pure
FFMA+LOP3 dual-issue. The branch + loop-counter (UIADD3+UISETP) consume scheduler
slots on the same ALU pipe as LOP3, contaminating the dual-issue measurement.

V50 has the same shape but per-template specializes (OP=0 has 8 FFMA per outer
iter, OP=2 splits warps so each warp does 8 of its own op).

## 4. Was the meta-doubt's specific critique correct?

**No.** The meta-doubt said V49 fails because of "self-RAW with immediate constants" while V8 has "distinct sources." Reality:

- **Self-RAW exists in both**: V8 is `Rd = Rsrc * Rd + Rd` (Rd appears twice as source, then written). V49 is `Rd = Rd * R0 + 0.5` (Rd appears once as source). V49 actually has **less** self-feedback in the multiply chain.
- **Immediates do NOT cost RF read ports** — they come from the instruction word. V49's pattern uses **fewer** RF read ports than V8's, not more. If anything V49 *should* be at least as fast as V8 for solo FFMA.
- The REAL issue is loop-overhead contamination + small inner body, not source distinction.

V49's OP=0 SOLO FFMA result (~25 Glane/s = 65% of peak per the V49 doc) is
under-saturated mostly because of the tiny inner loop and possibly only 2
warps/SMSP × 8 ILP not being enough chain depth to hide FFMA's 4-cycle latency.
V8 achieves 97.7% with 256 thr × launch_bounds(256,1) + 128-deep unroll.

## 5. Verdict on V49/V50 dual-issue confidence

| Wave | Claim | Evidence | New confidence |
|---|---|---|---|
| V49 | "Same-warp dual-issue 55% efficient" | Loop-overhead contaminated; tiny body | **LOW** — methodology unsafe |
| V50 | "Warp-specialized dual = 74% efficient" | Same loop-overhead, but cleaner per-warp homogeneous body | **LOW-MED** — better than V49 but not matching V8 unroll depth |
| Meta-doubt's specific reason | "Self-RAW + immediates causes the gap" | **Falsified by SASS** — V8 also self-RAWs, immediates use *fewer* ports | **WRONG** |
| The downgrade itself | "V49/V50 should be MED at best, not HIGH" | **Correct conclusion, wrong reasoning** | Keep MED → re-DOWNGRADE to **LOW** until reproduced with V8-style 128-deep inner loop |

## 6. Recommendation

Re-run V49 OP=2 / V50 OP=2 with:
1. Inner unroll depth ≥ 64 ops per type (not 8)
2. `__launch_bounds__(256, 1)` like V8 (not 128, 2)
3. Anti-DCE via STG of accumulator XOR, not via clock-diff-conditional
4. ncu `sm__inst_executed_pipe_fma.avg.pct_of_peak` AND `sm__inst_executed_pipe_alu` simultaneously

Until then, the "55%/74% same-warp vs warp-specialized" gap is **measurement
artifact, not architectural finding**. The architectural question (does B300
SMSP dual-issue FMA + ALU?) remains **OPEN**.

## Files
- `/root/github/QuickRunCUDA/tests/standalone/v49_dual_pipe.cu`
- `/root/github/QuickRunCUDA/tests/standalone/v50_warp_specialized.cu`
- `/root/github/QuickRunCUDA/tests/bench_ffma_warps_per_sm.cu`
- `/tmp/sass_verify/v49.sass`, `/tmp/sass_verify/v50.sass`
- `/root/github/QuickRunCUDA/sass/bench_ffma_warps_per_sm_1983379184.sass`
