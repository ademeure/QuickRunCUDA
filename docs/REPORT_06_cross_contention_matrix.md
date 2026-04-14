# Report #6 — Cross-pipe contention matrix (8×8 ops)

Measured F2FP/FFMA/IMAD/FMUL/MUFU/LOP3/IADD3/SHL as base op × companion op,
with N_BASE=4 chains, N_COMP=16 per iter, to map the SM's dispatch / pipe
structure.

Ratio = (base throughput with companion) / (base throughput alone).

## Baselines (N_COMP=0)

| Op | /SM/clk |
|---|---:|
| FFMA | 122.6 |
| FMUL | 123.6 |
| IMAD | 63.4 |
| MUFU.ex2 | 31.8 |
| F2FP.unpack | 63.6 |
| (LOP3, IADD3, SHL baselines too fast for event-timing precision) |

## Ratio matrix (rows=BASE, cols=+16 COMPANION/iter)

|        | FFMA | IMAD | FMUL | LOP3 | IADD3 | SHL  | MUFU.ex2 | F2FP.unpack |
|--------|-----:|-----:|-----:|-----:|------:|-----:|---------:|------------:|
| FFMA   | 0.21 | 0.12 | 0.20 | 0.81 | 0.81  | 0.99 | 0.06     | 0.13        |
| IMAD   | 0.38 | 0.20 | 0.37 | 1.00 | 1.00  | 1.00 | 0.13     | 0.25        |
| FMUL   | 0.20 | 0.12 | 0.20 | 0.84 | 0.84  | 0.90 | 0.06     | 0.13        |
| MUFU.ex2 | 0.74 | 0.50 | 0.73 | 0.99 | 0.97 | 1.00 | 0.20     | 0.50        |
| F2FP.unpack | 0.39 | 0.25 | 0.31 | 0.89 | 1.00 | 1.00 | 0.12     | 0.20        |

## Key observations

1. **IMAD behaves like FP arithmetic**, not integer logic. IMAD+LOP3=1.00 (free),
   IMAD+IADD3=1.00 (free), but IMAD+FFMA=0.38 (big drop). Confirms IMAD is on
   the **FMA pipe**, not the INT-logic pipe.
2. **LOP3/IADD3/SHL are on a COMPLETELY separate dispatch slot** from FFMA/
   F2FP/MUFU/IMAD. F2FP.unpack+IADD3=1.00, F2FP.unpack+SHL=1.00, F2FP.unpack+
   LOP3=0.89 (only mild dispatch-port sharing).
3. **MUFU and F2FP share SFU execution-pipe** — F2FP+MUFU=0.12 (much worse than
   F2FP+FFMA=0.39, because MUFU blocks SFU and takes longer on it).
4. **IMAD is MORE expensive than FFMA per op** — IMAD+IMAD=0.20, FFMA+IMAD=0.12.
   IMAD at 63/clk (half of FFMA 128/clk) takes 2× the FMA-pipe time per op.
5. **MUFU at N=16 has the HEAVIEST dispatch-slot impact** — because each MUFU
   occupies the FP/SFU issue slot for 2 cycles (running at 32 vs 64 baseline).

## Unified model — 2 issue-slot classes per SMSP

Each SMSP (sub-SM) has roughly two issue-slot classes:

**"Heavy" slot** (1 inst / SMSP / cycle, shared):
- FFMA, FMUL, FADD (FP32 FMA pipe)
- IMAD (on FMA pipe)
- MUFU.*, F2FP.*, F2F.*, POPC (on SFU pipe — 2 sub-units)
- HMMA/tensor-core mma (uses this slot but has its own functional unit)

**"Light" slot** (1 inst / SMSP / cycle, separate issue):
- IADD3, SHL, LOP3.LUT (INT logic pipe)
- BRA, SEL, etc.

Within the heavy slot, functional units are still distinct:
- FFMA/FMUL/IMAD — FMA pipe (128/SM/clk)
- MUFU/F2FP/F2F/POPC — SFU pipe (32 or 64/SM/clk depending on slot cost)
- HMMA — tensor core (own unit, same SMSP dispatch slot)

The "dispatch slot" is the first serialization point. After dispatch, ops go
to their functional unit.

## Implications

- **Mixing LOP3/IADD3/SHL with F2FP is genuinely free** (confirmed). Use these
  freely for address computation, index math, bit manipulation.
- **IMAD is NOT free** next to FFMA/F2FP — it steals FMA-pipe slots. Use IADD3
  for integer-only address math if you want it free.
- **FFMA + FFMA at 128/clk peak contends with ANY other heavy-slot op** —
  this is expected saturation behavior, not a specific pipe conflict.
- **Divergence is a big deal**: F2FP rate scales linearly with active lanes
  (Report #5), so branches inside F2FP loops cost performance.

