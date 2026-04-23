# §2.12.B9 Constant memory broadcast (LDC.32) — JUSTIFIED record

**Date:** 2026-04-23
**Auditor:** sub-agent (B9 task from REVIEW_CHECKLIST_B300.md)
**GPU:** B300 SXM6 (sm_103a), CUDA_VISIBLE_DEVICES=0
**Clock:** ~1.92 GHz during kernel (ncu sm__cycles_elapsed.avg.per_second). NOT explicitly locked; this is the rig's typical sustained clock under load.

---

## CLAIM (verbatim from `B300_PIPE_CATALOG.md` line 47)

> Constant mem broadcast (`LDC.32`, 4B/inst) | **17.8 TB/s eff** (~0.55 TB/s actual cache traffic) | — | 120 GB/s/SM eff

Cross-reference (B300_PIPE_CATALOG.md line 48):
> Constant mem broadcast (`LDC.64`, 8B/inst, via `uint2`) | **33.7 TB/s eff** (~1.05 TB/s actual cache traffic) | — | 228 GB/s/SM eff

REVIEW_CHECKLIST B9 flagged for:
> "broadcast amplification × 32 lanes; verify denominator — `[unit-confusion]`"

---

## Theoretical SoL

The constant-cache load opcode is `LDC` (per-thread to GPR) or `LDCU` (uniform to UR). When all 32 lanes of a warp use the same address (broadcast), one `LDC` warp-instruction issues a SINGLE cache fetch and hardware-broadcasts the 4-B result to all 32 lanes.

`LDC` is dispatched by the **ADU** pipe (Address Distribution Unit), not LSU and not uniform pipe — empirically confirmed below. ADU peak issue = **0.5 warp-inst/SM/cy** (cross-verified in `07_adu.md`).

Theoretical effective BW (broadcast amplified per-lane):
- 0.5 LDC/SM/cy × 148 SM × 4 B/lane × 32 lanes/warp × clock
- At 1.92 GHz: 0.5 × 148 × 4 × 32 × 1.92e9 = **18.18 TB/s**
- At 2.032 GHz boost: **19.24 TB/s**

Theoretical actual cache traffic (one 4-B fetch per warp-LDC):
- 0.5 × 148 × 4 × 1.92e9 = **0.568 TB/s** at 1.92 GHz (or 0.601 at 2.032 GHz)

Catalog claim 17.8 TB/s eff / 0.55 TB/s actual sits within ~98% of theoretical at 1.92 GHz.

---

## Test source

`/root/github/QuickRunCUDA/tests/bench_ldc_broadcast.cu`

Three modes selected via `-H "#define MODE N"`:
- **MODE=0**: warp-uniform index `(i + u0)` — broadcast pattern (compiler may emit either LDC or LDCU)
- **MODE=1**: per-lane varying index `(i + threadIdx.x + u0)` — non-broadcast (LDC with replay)
- **MODE=2**: inline `ld.const.u32` PTX with explicit address (not used in the final result)

Anti-DCE: the XOR-accumulator `sum` is written to `C[tid]` under impossible `if (sum == 0xDEADBEEFu)` — survives DCE because compiler can't prove sum ≠ 0xDEADBEEF.

`__constant__` array `CMEM[256]` declared at file scope; NVRTC accepts it.

---

## Build & run commands

```bash
make                       # produces ./QuickRunCUDA (sm_103a)

# Peak measurement (BS=512, 1 CTA/SM, NC=8 LDC per inner iter, 4096 outer iters):
pkill -9 QuickRunCUDA; sleep 5
CUDA_VISIBLE_DEVICES=0 ./QuickRunCUDA tests/bench_ldc_broadcast.cu \
    -t 512 -b 148 -A 64 -B 64 -C 1024 -0 1 -1 0 -2 0 -T 50 \
    -H "#define MODE 0
#define N_ITERS 4096
#define NC 8
#define BS 512
"
```

---

## Wall-clock measurement (peak BS=512, b=148, NC=8, iters=4096)

| Run | Wall ms | gpu_time_active (us) |
|---|---|---|
| Average over 50 launches | **0.588 ms** | 552 us (ncu) |

Bytes delivered effective (per-lane) = 148 SM × 512 thr × 4096 outer × 8 LDC × 4 B = **9.93 GB**

**Effective BW = 9.93 GB / 0.552 ms = 17.99 TB/s** ≈ catalog 17.8 TB/s (0.4% deviation, well within clock noise)

**Effective BW = 9.93 GB / 0.588 ms (wall) = 16.9 TB/s** (95% of catalog including event overhead)

Bytes delivered actual (per warp-LDC, broadcast = 1 fetch):
- Warp-LDC count = 148 × 16 warps/CTA × 4096 × 8 = 77.59 M warp-instructions
- Bytes = 77.59M × 4 = 310 MB

**Actual cache traffic = 310 MB / 0.552 ms = 0.562 TB/s** ≈ catalog 0.55 TB/s (2% deviation)

---

## ncu metrics (BS=512 peak)

```
gpu__time_active.sum                                          ms  0.55232
sm__cycles_elapsed.avg                                     cycle  1059637
sm__cycles_elapsed.avg.per_second                            GHz  1.92
sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_active   %  99.51   <-- LDC bottleneck
sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active   %  43.52
sm__inst_executed_pipe_fmaheavy.avg.pct_of_peak_sustained_active %  21.76
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active    %   0.00  <-- LDC NOT on LSU
sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_active %   6.22 <-- LDC NOT on uniform
smsp__inst_executed.sum                                      inst  310M
```

**Pipe assignment correction:** LDC.32 dispatches via the **ADU** pipe (saturated at 99.51%), NOT LSU (0%) and NOT the uniform pipe (6.22%, just from address arith). This is a notable architectural fact missing from the catalog's pipe table at §2.12.

Issue rate per SMSP = 0.124 LDC/SMSP/cy → **0.495 LDC/SM/cy** = exactly the ADU pipe peak (0.5/SM/cy from §7).

---

## SASS confirmation

File: `/root/github/QuickRunCUDA/sass/bench_ldc_broadcast_1230985133.sass` (MODE=0 NC=8 BS=256 build) and matching MODE=0 BS=512 build.

Inner loop body (8 LDC per outer iter, broadcast — same source register used for the address across all 32 lanes because R15 is initialized from a uniform value):

```
.L_x_0:
        IMAD.SHL.U32 R2, R15.reuse, 0x4, RZ ;
        ... [LOP3.LUT addr arith, 8x] ...
        LDC R3, c[0x3][R3] ;
        LDC R4, c[0x3][R4] ;
        LDC R5, c[0x3][R13] ;
        LDC R6, c[0x3][R14] ;
        LDC R7, c[0x3][R11] ;
        LDC R2, c[0x3][R12] ;
        LDC R9, c[0x3][R10] ;
        LDC R8, c[0x3][R8] ;
        LOP3.LUT R0, R8, R9, R2, 0x96, !PT ;  -- accumulate
        BRA.U UP0, `(.L_x_0) ;
```

8 × `LDC R, c[0x3][Rn]` per inner loop body. Outer loop count = `N_ITERS=4096`. Total LDC per warp = **32,768** as expected.

---

## Block sweep (BS=512, NC=8, N_ITERS=4096)

| blocks | wall ms | comment |
|---|---|---|
| 37 | 0.588 | Single CTA per ~4 SMs — saturated within CTA |
| 74 | 0.588 | Saturated |
| 148 | 0.588 | 1 CTA/SM — peak |
| 296 | 1.172 | 2× CTAs → 2× time (linear) |
| 592 | 2.340 | 4× CTAs → 4× time |

ADU pipe is **per-SM**, so doubling resident CTAs doesn't double throughput per CTA; total work doubles → time doubles. Saturation is reached at b=37 (because each CTA already nearly fills its host SM's ADU pipe).

---

## Cross-check: MODE=1 (non-broadcast) — confirms 32× amplification

| Mode | wall ms | pipe_adu | per-warp LDC behavior |
|---|---|---|---|
| MODE=0 broadcast | 0.588 | 99.5% | 1 cache fetch + 32× HW broadcast |
| MODE=1 per-lane | **18.66** | 99.98% | 32 serialized cache fetches (replay) |

**Slowdown ratio = 18.66 / 0.588 = 31.7×** ≈ 32 lanes — exactly the broadcast amplification factor. ADU pipe stays 99-100% saturated in both, but each non-broadcast LDC consumes 32 ADU cycles vs 1 cycle for broadcast.

This empirically validates the catalog's "broadcast amplification × 32 lanes" mechanism.

---

## Verdict

**✅ REPLICATED — both numbers match within 1-2% at 1.92 GHz**

| Quantity | Catalog | Measured (this audit) | Match |
|---|---|---|---|
| Effective BW (per-lane) | 17.8 TB/s | **17.99 TB/s** (gpu_time) | ✅ +1.1% |
| Actual cache traffic | 0.55 TB/s | **0.562 TB/s** | ✅ +2.2% |
| Per-SM eff | 120 GB/s | **121.6 GB/s** | ✅ |
| Broadcast amplification | "32 lanes" | **31.7× vs non-broadcast** | ✅ |

**Catalog is correct. The denominator confusion in the REVIEW_CHECKLIST entry is resolved: catalog correctly distinguishes effective (per-lane × 32 = 17.8 TB/s) from actual cache traffic (per-warp-LDC = 0.55 TB/s); they differ by exactly the 32-lane broadcast factor.**

---

## NEW INSIGHT (architectural fact missing from catalog §2.12)

**LDC dispatches on the ADU pipe, NOT LSU.** Catalog §2.12 (L432-444) PTX→pipe table omits LDC entirely — the closest entry is `ld.global.u32 → LDG.E → lsu`, which doesn't apply.

Recommended catalog addition:
```
| ld.const.u32 (broadcast) | LDC.32  | adu | 0.5 warp-inst/SM/cy peak; HW-broadcasts to 32 lanes |
| ld.const.u32 (per-lane)  | LDC.32  | adu | 32× slower (replays per address); same opcode |
| ld.const.u32 (uniform)   | LDCU.32 | uniform | issues to UR (uniform RF), pipe_uniform peak 2.0 |
```

The compiler picks LDC vs LDCU based on whether the destination is needed in per-thread regs (LDC) or only as uniform/scaffolding (LDCU). MODE=0's `(i + u0) & 255` indexing yields LDC because the result is XOR'd into a per-thread `sum`.

---

## Methodology gotchas

1. **`__constant__` declaration works fine via NVRTC.** No special handling needed.
2. **BS matters a LOT for measuring ADU peak.** BS=256 reaches only 90% pipe_adu (40% effective BW shortfall vs catalog) because alu/scaffolding starves issue. BS=512 hits 99.5% — recommended for any LDC measurement.
3. **`-T` events vs ncu `gpu__time_active`** differ by 6% (event overhead). Use ncu for definitive throughput.
4. **`--reuse-cubin` silently re-runs the OLD kernel** — easy to mis-sweep configs. Always force re-compile or check the SASS hash in output.
5. **`pipe_lsu = 0%` is the diagnostic** — LDC is not on LSU. If a tester sees pipe_lsu high while measuring "constant memory throughput", they're actually measuring something else (likely a compiler fallback to LDG via generic-space cast).
6. **MODE=2 inline asm with `ld.const.u32 [%1]`** would NOT compile cleanly because PTX `ld.const` requires a constant-space symbol/offset, not a generic 64-bit pointer; the compiler tends to fall back to `ld.global` in that case. The clean way is just `CMEM[idx]` from a `__constant__` array.

---

## Files

- Test source: `tests/bench_ldc_broadcast.cu`
- SASS dump (MODE=0 NC=8 BS=256): `sass/bench_ldc_broadcast_1230985133.sass`
- SASS dump (MODE=1 — non-broadcast): `sass/bench_ldc_broadcast_2364261131.sass`
