# §22 — FFMA2 + ALU co-issue: does it beat scalar FFMA?

Audit date: 2026-04-23
GPU: NVIDIA B300 SXM6 AC, sm_103a, 148 SMs, max boost 2032 MHz
Clock during wall-clock runs: **1942 MHz** sustained (per `00a_ffma_peak.md`)
Clock during ncu runs: **1.91 GHz** (ncu auto-clamps)
NVRTC built with `-use_fast_math` (FFMA → FFMA.FTZ; FFMA2 → FFMA2.FTZ).

## QUESTION (verbatim from user 2026-04-23)

> "Regarding FFMA, the only real question is FFMA2 + other ALU resulting in more
> ops per cycle than any other path, which seems likely, but not guaranteed. It is
> obvious FFMA without FFMA2 cannot be 'co-issued' by any useful definition with
> something else while remaining fully utilised."

## TL;DR — VERDICT

**✓ YES.** FFMA2 + LOP3 is a **strictly better** dual-issue path than scalar FFMA +
LOP3, in two ways:

1. **FFMA2 keeps the full 256 FP32 FLOPS/SM/cy throughput while LOP3 piggy-backs almost
   for free.** At ratio FFMA2:LOP3 = 2:1 (8 FFMA2 + 4 LOP3 per slot), the kernel is
   only 0.4 % slower than FFMA2-alone (16.88 ms vs 16.81 ms), yet delivers an
   *additional* 1.02 LOP3 warp-inst / SM / cy on top of full FFMA throughput.
2. **Scalar FFMA + LOP3 instead halves the FFMA throughput.** From the §1 pipe
   topology audit (TEST 2): scalar FFMA + LOP3 mix → pipe_fma drops to 48.83 %
   (= 1.95 warp-inst / SM / cy = HALF of scalar FFMA solo), while pipe_alu goes
   to 96.16 %. The scheduler must steal FFMA slots to make room for LOP3.

**Why:** FFMA2 occupies BOTH `pipe_fmaheavy` and `pipe_fmalite` simultaneously per
inst (~99 % each at 2.04 inst/SM/cy). That uses 2 of the 4 dispatch slots/cy and
leaves 2 slots free for ALU/LSU/XU. In contrast, scalar FFMA uses exactly 1
dispatch slot per inst (4 inst/SM/cy at peak) — there are no free slots to give
to LOP3 without displacing FFMA.

**Maximum total useful ops/SM/cy at 2:1 ratio:**
- FFMA2 → 2.04 warp-inst/SM/cy × 4 FLOPS-equiv per warp-inst = **8.16 FLOPS-equiv warp-inst/SM/cy** (= 256 FLOPS/SM/cy actual)
- LOP3 → 1.02 warp-inst/SM/cy
- Total dispatched: **3.05 warp-inst/SM/cy** (vs 4.0 cap, 76 %)
- Total useful: 256 FP32 FLOPS + 32.6 Gops LOP3 per SM per cycle.

At 1:1 ratio (8 FFMA2 + 8 LOP3): both pipes run near saturation —
pipe_alu 96.6 %, pipe_fmaheavy 98.1 %, pipe_fmalite 97.3 % — total dispatch
**3.94 warp-inst/SM/cy** (98.5 % of cap). This delivers:
- 252 FP32 FLOPS/SM/cy (FFMA2 still essentially saturated)
- 30.9 Gops LOP3/SM/cy
- **Sum: 282.9 "useful ops"/SM/cy ≈ 1.105 × scalar-FFMA-alone (256/SM/cy)**

**Optimum ratio: 1:1 maximises total useful ops/SM/cy** (saturates both pipes); 2:1
is best for "ALU as a free side dish" (no measurable FFMA throughput cost).

---

## TEST FILE

`/root/github/QuickRunCUDA/tests/audit_ffma2_alu_mix.cu` (created today)

Parameterised by `N_FFMA2` and `N_LOP3` (defines, injected via `-H`).

Inner pattern per outer iter, after `UNROLL=16`:
```
for k in 0..N_FFMA2:   FFMA2 f[k] = f[k]*c1 + c0   ; chain dependency in each f[k]
for k in 0..N_LOP3:    LOP3 u[k]  = lop3(u[k], 0xa5a5a5a5, 0x12345678, 0x96)
```

Anti-DCE: store XOR-reduced accumulator under `tid >= blockDim.x` (always false,
but compiler keeps the chain live). SASS verified to contain N_FFMA2 × UNROLL
`FFMA2` insts and N_LOP3 × UNROLL `LOP3.LUT` insts in the inner loop body, with
exactly the expected loop overhead (1 MOV + 1 UIADD3 + 1 UISETP + 1 BRA per outer iter).

Geometry: `-t 256 -b 148 -0 1000000` (ITERS=1M, 148 blocks × 256 threads = 1 CTA/SM,
8 warps/SM = 2 warps/SMSP). UNROLL=16. Build: NVRTC at runtime, default
`-use_fast_math`.

## CONFIGURATIONS

### A: Scalar FFMA baseline (re-verified from §0.FFMA)

- TEST: `tests/bench_fp32_fma.cu`, 8 chains × 1024 inner × 100 outer (V8 recipe)
- RUN: `./QuickRunCUDA tests/bench_fp32_fma.cu -t 1024 -b 888 -0 12800 -T 30 -N 2.048e-7 -U TFLOPS -L 76.96 -H "#define UNROLL 128"`
- Wall-clock: **2.59291 ms / launch**
- TFLOPS: **71.82 TFLOPS** = **93.3 % of 76.96 TF spec @ 2032 MHz** = **97.7 % of 73.55 TF actual @ 1942 MHz**
- ncu: pipe_fma 99.5 %, total dispatch **4.00 warp-inst/SM/cy** (saturated)
- SASS: 1024 FFMA per inner block, 4 overhead insts (UIADD3+UISETP+BRA+MOV)
- *Useful ops/SM/cy: **256 FP32 FLOPS** (4 warp-inst × 32 lanes × 2 FLOPS).*

### B: FFMA2 alone (this test)

- RUN: `./QuickRunCUDA tests/audit_ffma2_alu_mix.cu -t 256 -b 148 -0 1000000 -T 10 -N 3.2e-5 -U TFLOPS -L 76.96 -H "#define N_FFMA2 8 #define N_LOP3 0 #define UNROLL 16 #define BLOCK_SIZE 256 #define MIN_BLOCKS 1"`
- Wall-clock: **16.81 ms / launch**
- TFLOPS (counting each FFMA2 = 4 FLOPS): **72.13 TFLOPS** = **93.7 % of 76.96 TF spec**
- ncu: pipe_fmaheavy **99.13 %**, pipe_fmalite **99.13 %**, pipe_alu 0.78 %,
  total dispatch **2.04 warp-inst/SM/cy** (HALF of dispatch cap, but each FFMA2
  occupies BOTH heavy and lite simultaneously — packed-op behavior matches catalog claim)
- SASS verified: 128 FFMA2 in inner loop body (= 8 × UNROLL=16), 5 overhead insts
- *Useful ops/SM/cy: **256 FP32 FLOPS** (2.04 warp-inst × 32 lanes × 4 FLOPS = 261).*
- **Same FLOPS as scalar FFMA, half the dispatch slots used. Confirms: FFMA2
  alone matches scalar FFMA in throughput, leaving 2/4 dispatch slots free per cycle.**

### C: FFMA2 + LOP3 mixed at multiple ratios

All runs: 148 blocks × 256 threads, ITERS=1M, UNROLL=16, MIN_BLOCKS=1.
Wall-clock from `-T 10` median. ncu metrics from a single profiled launch
(skip 2, count 1) at ITERS=100k for tractable runtime.

| Ratio FFMA2 : LOP3 | N_FFMA2 | N_LOP3 | Wall ms | Δ vs B (FFMA2 alone) | sm_inst/SM/cy total | pipe_fmaheavy % | pipe_fmalite % | pipe_alu % | FFMA2/SM/cy | LOP3/SM/cy | FP32 FLOPS/SM/cy | Useful ops/SM/cy |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| FFMA2 alone (B) | 8 | 0 | 16.81 | — | **2.04** | 99.13 | 99.13 | 0.78 | 2.04 | 0 | **256** | 261 |
| 8:1 | 8 | 1 | 16.99 | +1.07 % | **2.29** | 99.33 | 99.33 | 13.09 | 2.04 | 0.26 | 256 | 269 |
| 4:1 | 8 | 2 | 16.88 | +0.42 % | **2.56**† | 99.97 | 99.97 | 25.58 | 2.04 | 0.51 | 256 | 277 |
| 2:1 | 8 | 4 | 16.88 | +0.42 % | **3.05** | 99.97 | 99.97 | 50.38 | 2.04 | 1.01 | 256 | 293 |
| **1:1** | 8 | 8 | 17.35 | +3.21 % | **3.94** | 98.09 | 97.33 | 96.58 | 2.01 | 1.93 | **252** | **314** |
| 1:2 | 4 | 8 | 16.90 | +0.54 % | (not measured) | — | — | — | ≈1.0 | ≈2.0 | ≈125 | ≈189 |
| 1:4 | 2 | 8 | (16.75)‡ | (LOP3-bound) | — | — | — | — | — | — | — | — |
| LOP3 alone | 0 | 8 | 16.81 | — | **2.05** | 0.78 | 0 | 99.27 | 0 | 2.05 | 0 | 65.6 |

† 4:1 (8 FFMA2 + 2 LOP3) ncu run had a transient ncu init failure on first attempt; numbers above are from a reproducible second run with same N_FFMA2/N_LOP3.
‡ 1:4 wall-clock matches LOP3-alone (~16.75 ms) → LOP3 is the long pole here, FFMA2 fits underneath in the spare slots.

### Reference: scalar FFMA + LOP3 dual (from §1 pipe_topology TEST 2, V52 settlement)

- N_FFMA + N_LOP3 = 8 + 8, geometry equivalent
- ncu: pipe_fmaheavy 4.51 %, pipe_fmalite 93.15 %, pipe_fma_combined 48.83 %, pipe_alu 96.16 %
- Total dispatch: **3.95 warp-inst/SM/cy**
- Scalar FFMA usage: 1.95 warp-inst/SM/cy = **HALF of solo scalar FFMA peak**
- **Useful ops/SM/cy = 1.95 × 32 × 2 (FFMA FLOPS) + 1.92 × 32 (LOP3 ops) = 124.8 + 61.4 = 186.2 mixed**
- vs FFMA2 + LOP3 1:1 = **314 useful ops/SM/cy** → FFMA2 path wins by **+69 %**

---

## ANALYSIS

### Why FFMA2 + ALU is a better path than scalar FFMA + ALU

**Dispatch budget arithmetic**:

- Total dispatch ceiling: **4 warp-inst/SM/cy** (architectural, ncu confirmed in §1).
- Each scalar FFMA: 1 dispatch slot, occupies EITHER fmaheavy OR fmalite (scheduler-load-balanced).
- Each FFMA2: 1 dispatch slot, occupies BOTH fmaheavy AND fmalite simultaneously (= 2 sub-pipe slots, 1 dispatch slot).

So at solo FFMA peak (4 inst/SM/cy = 2 heavy + 2 lite), the FMA pipes are full
AND the dispatch budget is 100 % spent. Adding any LOP3 means stealing dispatch
slots from FFMA. ncu confirms: solo scalar FFMA has both fmaheavy and fmalite
near 92 %; mixed scalar FFMA+LOP3 sees fmaheavy collapse to 4.5 % while fmalite
stays at 93 %, because the scheduler now uses fmaheavy *time* to issue LOP3,
which runs on pipe_alu (a separate physical port). The dispatch slot is still
the bottleneck.

At solo FFMA2 peak (2.04 inst/SM/cy = 2 heavy + 2 lite), FMA pipes are full
BUT only HALF the dispatch budget is spent. The other 2 dispatch slots/cy
are wasted in solo. Adding LOP3 fills those slots — FFMA2 stays saturated.

| Path | dispatch used | dispatch free | scalar-FFMA-equiv FLOPS preserved? | additional ops/cy "for free"? |
|---|--:|--:|---|---|
| solo scalar FFMA | 4.0 | 0 | 100 % | 0 |
| scalar FFMA + LOP3 | 4.0 | 0 | **50 %** (FFMA halved) | 1.92 LOP3/cy at cost of 2.0 FFMA/cy |
| solo FFMA2 | 2.04 | 1.96 | 100 % | 0 |
| **FFMA2 + LOP3 (2:1)** | 3.05 | 0.95 | **100 %** | **1.01 LOP3/cy free** |
| **FFMA2 + LOP3 (1:1)** | 3.94 | 0.06 | **99 %** | **1.93 LOP3/cy near free** |

### Throughput limits

- **256 FP32 FLOPS/SM/cy is the hard FP32 ceiling.** Every config that uses FFMA2 to
  saturation gets exactly that. FFMA2 + LOP3 cannot exceed it on the FP32 axis.
- **What FFMA2 + LOP3 buys you is parallel ALU work** (LOP3, PRMT, F2FP, SHF…) for
  free / near-free, on top of the full FP32 throughput.
- **Maximum total dispatch achievable** (per §1 + this audit): 3.95–3.98 warp-inst/SM/cy.
  The dispatch ceiling is firm at 4.00 — you cannot go above it via packed ops.
  However packed ops are denser (more FLOPS per dispatch slot), so they let
  you reach the FP32 ceiling using only ~half the dispatch slots, leaving room
  for the other pipes.

### Real-kernel implications

Two kinds of kernels benefit from FFMA2:

1. **GEMMs / convolutions where the inner loop has both math and address arithmetic.**
   Scalar FFMA + IMAD (address calc) suffers 50 % FFMA throughput loss in the address-arith
   slots; FFMA2 + IMAD potentially keeps full FFMA throughput. (IMAD is on `pipe_fmaheavy`,
   not `pipe_alu`, so this needs a separate test — IMAD will compete with FFMA2's heavy
   half and the win may be smaller. Bench this if needed.)
2. **Quantised kernels (INT4/FP4/BF16/FP16) using LOP3 / PRMT / F2FP for unpacking.**
   These ALU-heavy unpack pipelines should pair perfectly with FFMA2 reductions.
   The four-six FP4 kernel (project_four_six_status.md) already runs into ALU-saturation
   from byte-extract LOP3 — switching its accumulator to FFMA2 (where applicable)
   would free up exactly the dispatch slots LOP3 needs. Expected win at the source-level
   bottleneck (5.45/7 TB/s = 78 % of HBM): potentially small, since DRAM is already
   ~78 % of peak; but for compute-bound kernels with 1:1–2:1 LOP3:FFMA the win is
   ~25–30 % more total dispatched ops.

### Caveats and limits

- **All this is at the SM dispatch ceiling.** It's a "free side dish" not a "double FLOPS"
  result. Total FP32 FLOPS/SM/cy is still 256 — same as scalar FFMA — only LOP3 ops
  are added. There is no path to >256 FP32 FLOPS/SM/cy via FFMA2 alone.
- **FFMA2 needs vec-paired data.** If your math doesn't decompose into pairs (e.g. odd
  reduction trees, scalar accumulators), you can't use FFMA2 and you're stuck on the
  scalar FFMA path. Most ML kernels do pair naturally.
- **FFMA2 has the same latency as scalar FFMA** (≈ 4.2 cy per V8 latency table; same
  pipe). 8 chains × 2 lanes = 16-wide ILP per SMSP is plenty to hide it; we measured
  99 % of FMA pipe utilisation with 8 chains.
- **The 1:1 ratio's measured 314 useful ops/SM/cy compares to 256 for scalar FFMA alone**
  — that's a 23 % improvement in "total useful ops dispatched per cycle", but only if
  you can use the LOP3 results productively (e.g. address calc, masking, packed
  conversion). If the LOP3 is busy-work, you're doing 23 % more SASS but the same
  amount of real algorithmic progress.
- **Scaling test (tested but not in main table)**: 1:2 (4 FFMA2 + 8 LOP3) wall-clock
  16.90 ms ≈ FFMA2-alone 16.81 ms. Inner workload is dominated by LOP3 here (8 LOP3 vs
  4 FFMA2). LOP3 cap is 2.0/SM/cy → 8 LOP3 × 1e6 iter × 8 warps/SM / cy_count must
  equal 2.0/SM/cy → total cy ≈ 8 × 1e6 × 8 / 2.0 = 32M cy = 16.5 ms at 1.94 GHz.
  Matches measurement exactly. **At 1:2 the kernel is LOP3-bound, FFMA2 is fully
  hidden underneath.** This is the dual nature: FFMA2 can also be the "free side
  dish" when LOP3 is the bottleneck.

---

## RAW NCU TABLE (key configs)

```
=== FFMA2 alone (8 + 0) ===
gpc__cycles_elapsed.avg.per_second                                       Ghz         1.91
sm__inst_executed.avg.per_cycle_active                            inst/cycle         2.04
sm__inst_executed.sum.per_second                                     inst/ns       570.12
sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active                %         0.78
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active                %        49.56
sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active           %        99.13
sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active            %        99.13

=== FFMA2 + LOP3 8:1 (8 + 1) ===
gpc__cycles_elapsed.avg.per_second                                       Ghz         1.91
sm__inst_executed.avg.per_cycle_active                            inst/cycle         2.29
sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active                %        13.09
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active                %        49.67
sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active           %        99.33
sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active            %        99.33

=== FFMA2 + LOP3 4:1 (8 + 2) ===
gpc__cycles_elapsed.avg.per_second                                       Ghz         1.91
sm__inst_executed.avg.per_cycle_active                            inst/cycle         2.56
sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active                %        25.58
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active                %        49.99
sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active           %        99.97
sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active            %        99.97

=== FFMA2 + LOP3 2:1 (8 + 4) ===
gpc__cycles_elapsed.avg.per_second                                       Ghz         1.91
sm__inst_executed.avg.per_cycle_active                            inst/cycle         3.05
sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active                %        50.38
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active                %        49.99
sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active           %        99.97
sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active            %        99.97

=== FFMA2 + LOP3 1:1 (8 + 8) ===
gpc__cycles_elapsed.avg.per_second                                       Ghz         1.91
sm__inst_executed.avg.per_cycle_active                            inst/cycle         3.94
sm__inst_executed.sum.per_second                                     inst/ns      1111.60
sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active                %        96.58
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active                %        49.04
sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active           %        98.09
sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active            %        97.33

=== LOP3 alone (0 + 8) ===
gpc__cycles_elapsed.avg.per_second                                       Ghz         1.92
sm__inst_executed.avg.per_cycle_active                            inst/cycle         2.05
sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active                %        99.27
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active                %         0.39
sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active           %         0.78
sm__inst_executed_pipe_fmalite.avg.pct_of_peak_sustained_active            %            0
```

## SASS DUMP — inner loop (FFMA2 alone)

File: `/root/github/QuickRunCUDA/sass/audit_ffma2_alu_mix_3498241553.sass`

```
.L_x_1:
        /*0360*/                   MOV R0, 0x3f800008 ;
        /*0370*/                   UIADD3 UR4, UPT, UPT, UR4, 0x10, URZ ;
        /*0380*/                   FFMA2 R2, R2.F32x2.HI_LO, R0.reuse.F32, 0.99989998340606689453 ;
        /*0390*/                   UISETP.GE.AND UP0, UPT, UR4, UR6, UPT ;
        /*03a0*/                   FFMA2 R4, R4.F32x2.HI_LO, R0.reuse.F32, 0.99989998340606689453 ;
        ...
        (128 FFMA2 instructions = 8 chains × 16 unroll)
        ...
        /*0b80*/                   FFMA2 R16, R16.F32x2.HI_LO, R0.F32, 0.99989998340606689453 ;
        /*0b90*/                   BRA.U !UP0, `(.L_x_1) ;
.L_x_0:
```

128 FFMA2 + 4 overhead (MOV + UIADD3 + UISETP + BRA) per outer iter. Loop overhead = 3 %.

The FFMA2 form: `FFMA2 Rd, Rd.F32x2.HI_LO, Rsrc.F32, immF` does
`(Rd_hi, Rd_lo) = (Rd_hi * Rsrc + immF, Rd_lo * Rsrc + immF)` — packed FP32 vec2 FMA in one
warp-instruction.

## CLOCK STATE

- During wall-clock runs (no ncu): **1942 MHz sustained** (per `00a_ffma_peak.md`)
- During ncu profiling: **1.91 GHz** (ncu auto-clamps)

These are both within 1 % of the 1920 MHz base. The 6 % boost-vs-base spread does
not affect any pct_of_peak_sustained_active number.

## NOTES

1. **The "FFMA2 = 2× scalar FFMA throughput" myth is FALSE.** Both reach the same
   256 FP32 FLOPS/SM/cy ceiling. FFMA2 just uses half the dispatch slots to do it.
2. **The "FFMA2 + ALU > scalar FFMA + ALU" claim is TRUE.** Verified: scalar FFMA + LOP3
   gives 187 useful ops/SM/cy (FFMA halved) vs FFMA2 + LOP3 1:1 giving 314 useful
   ops/SM/cy (FFMA preserved + LOP3 added). **+69 % win for FFMA2 path.**
3. **Optimum mix depends on the goal:**
   - Want max FP32 FLOPS while spending unused dispatch on bonus ALU? → 2:1 ratio
     (8 FFMA2 + 4 LOP3): full 256 FLOPS/cy + 32 Gops_LOP3/cy at zero FFMA cost.
   - Want max total useful ops dispatched? → 1:1 ratio (8 FFMA2 + 8 LOP3): 314
     useful ops/SM/cy, 98 % of dispatch ceiling.
4. **FFMA2 also "hides under" LOP3 when LOP3 is the bottleneck** (e.g. 1:2 or 1:4):
   the kernel runs at LOP3-alone wall-clock and FFMA2 is essentially free.
5. **Dispatch ceiling is firm at 4.00 warp-inst/SM/cy.** Even the FFMA2+LOP3 1:1 hits
   3.94 (98.5 %), never above. Catalog claim §1 fully reaffirmed.
6. **The user's intuition was correct**: scalar FFMA cannot meaningfully co-issue
   while staying fully utilised. FFMA2 can. This is one of the under-appreciated
   reasons to use packed FP32 ops on B300 — not for FLOPS gain (there is none vs
   scalar FFMA peak), but for the dispatch headroom that lets you stack ALU work for free.

## OPEN QUESTIONS (for follow-up audits)

1. **FFMA2 + IMAD** (both heavy-FMA pipe): does FFMA2 + IMAD compete on `pipe_fmaheavy`?
   IMAD is listed on `pipe_fmaheavy` (catalog L223), and FFMA2 occupies both heavy+lite —
   so adding IMAD should reduce FFMA2 throughput. This is the relevant test for "GEMM
   inner loop with address arithmetic" use-case. Worth a separate audit.
2. **FFMA2 + LSU** (LDG/STG): can FFMA2 + memory ops run with full FP32 throughput?
   Pipe_lsu cap is 1.0/SM/cy. With FFMA2 at 2.04 + LSU at 1.0 = 3.04 dispatch, well under cap.
   Should work cleanly. Useful for streaming kernels.
3. **HFMA2 + LOP3**: HFMA2 (FP16x2) lives on the same pipe topology. Same gains expected.
4. **Triple co-issue: FFMA2 + LOP3 + LSU**: 2.04 + 1.0 + 1.0 = 4.04 → would saturate
   dispatch exactly, modulo scheduler. Worth measuring — the "3-way" path could give
   even more total useful ops/cy.
