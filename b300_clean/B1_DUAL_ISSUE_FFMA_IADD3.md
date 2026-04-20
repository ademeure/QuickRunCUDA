# Dual-pipe issue: FFMA + IADD3 — V4 / B1 partial

**Date: 2026-04-20.** Tests `bench_iadd3_throughput.cu` and
`bench_dual_pipe_ffma_iadd3.cu`. 1500 MHz clock-locked, persistent grid.

## Per-instruction throughput (1500 MHz)

| Inst | Per-inst peak (TIPS_inst) | per SMSP/cy | Notes |
|------|---------------------------|-------------|-------|
| LOP3.LUT | 14.16 | 0.50 | ALU pipe; imm-independent |
| IADD3 | 14.13 | 0.50 | ALU pipe; nvcc fuses `add;add` to single IADD3 |
| FFMA | 18.87 | 0.66 | FMA pipe; **only 66% of theoretical 28.4** |

## IADD3 → 28.32 TIOPS in raw "additions"

PTX `add.s32 v,v,b; add.s32 v,v,c` fuses to single SASS `IADD3 v,v,b,c`.
- Per-inst rate: 14.13 TIPS_inst (same as LOP3)
- Per-add rate: 28.32 G adds/s ⇒ this is the "2.46 IADD3/SM/cy"
  catalog claim — it counts each IADD3 as ~1.23 ALU-ops because the
  fused inst does ~2 adds. The instruction throughput is 14.16 TIPS,
  same as LOP3.

## Dual-pipe FFMA + IADD3: NO clean overlap

Test config: 8 FFMA chains + 8 IADD3 chains, all independent registers.

| Mode | Time/iter | TIPS_inst | Speedup vs sequential |
|------|-----------|-----------|----------------------|
| FFMA-only | 1.607 ms | 18.87 | — |
| IADD3-only | 2.146 ms | 14.13 | — |
| Sequential predicted | 3.753 ms | — | 1.0× |
| Mixed measured | 3.219 ms | 18.83 | **1.17× (only 17%)** |
| Perfect overlap predicted | 2.146 ms (max) | 33.0 | 1.75× (not achieved) |

**Conclusion**: FFMA and IADD3 do NOT freely dual-issue. The partial 17%
overlap is small, suggesting:
1. SMSP dispatch slot is shared across pipes — only 1 inst/cy issued
   regardless of pipe.
2. OOO can hide a few cycles when one pipe stalls.
3. The catalog "FFMA + IADD3 parallel" claim oversimplifies.

## Open questions

- What about FFMA + MUFU? (Different pipe family.)
- What about FFMA + LDG? (Memory pipe definitely separate.)
- Is the SMSP issue rate fundamentally 1/cy for ALL ALU/FMA pipes?

## Why FFMA is at 66% (not 100%)

Theoretical FFMA peak at 1500 MHz from CLAUDE.md formula:
  148 SMs × 128 cores × 2 op/FMA × 1.5 GHz = 56.83 TFLOPS = 28.4 TIPS_inst.

Per-SMSP: 1 FFMA inst/cy/SMSP (32 cores × 1 FMA per cy = 1 warp-inst).

We measure 0.66 inst/SMSP/cy. With 2 warps/SMSP (256 thr, persistent),
each warp issues every 4 cy at NC=8 (latency-bound), but SMSP can only
issue 1 inst/cy total → max 1 inst/SMSP/cy.

The 33% gap likely reflects warp-scheduler stall on alternating-warp
issue: when both warps want to issue same pipe, scheduler may take 2-3
cy to alternate cleanly. Could verify with 4 warps/SMSP.

## SASS verification

```
FFMA-only inner: 384 FFMA instructions (NC=24 × UNROLL=16)
IADD3 inner: 257 IADD3 (NC=16 × UNROLL=16 + 1 setup)
Mixed: alternating FFMA / IADD3 (need closer look)
```

Both IADD3 and FFMA fuse correctly from 2-op PTX to 1 SASS inst.

## Confidence

- **HIGH** for IADD3 = 14.13 TIPS_inst = 0.5/SMSP/cy (3 trials, stable)
- **HIGH** for mixed showing only 17% overlap (not pure dual-issue)
- **MED** for "33% FFMA bubble" cause — needs warp-count sensitivity test

## Files

- `tests/bench_iadd3_throughput.cu`
- `tests/bench_dual_pipe_ffma_iadd3.cu`
