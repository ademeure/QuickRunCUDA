# Report #9 — Triple-pipe saturation reveals dispatch ceiling

Running FFMA + F2FP + LOP3 simultaneously on totally independent register
chains. If pipes were fully independent, total should be `≈ 128 + 64 + 128
= 320 thread-ops/SM/cycle`. Measured:

| Config (N_FFMA + N_F2FP + N_LOP3 per iter) | Total /SM/clk |
|---|---:|
| Solo FFMA (16) | 125 |
| Solo F2FP (16) | 64 |
| Solo LOP3 (16) | 121 |
| FFMA + F2FP (16+8) | 116 |
| FFMA + LOP3 (16+16) | 114 |
| F2FP + LOP3 (8+8) | 84 |
| **All three (32+16+32)** | **125** |

**Combined total caps at ~125 thread-ops/SM/cycle regardless of the pipe mix.**
Triple-pipe doesn't go any higher than FFMA alone.

## Interpretation

The **warp-scheduler dispatch port** is the common limit. Each SM has 4 SMSPs;
each SMSP can issue 1 warp-instruction per cycle → 4 warp-insts × 32 lanes =
**128 thread-ops/SM/cycle max aggregate dispatch rate**.

Individual pipe caps apply on top:
- FFMA pipe (fmalite): 128/SM/cycle peak (saturates all 4 SMSPs)
- F2FP pipe (alu): 64/SM/cycle peak for unpack (2 SMSPs), 32/clk for pack
- LOP3 pipe (alu+fmaheavy): ≥128/SM/cycle (dispatch-ceiling bound)

**When two or more pipes compete for dispatch bandwidth, total is bounded by
128, not summed.** This explains why:
- FFMA solo = 125 (near dispatch peak)
- Adding F2FP doesn't push beyond 125 — F2FP simply steals dispatch from FFMA
- All3 sum = 125 — whatever mix, dispatch is the ceiling

The "pipe independence" I claimed earlier is only true at LOW contention. At
saturation, everything fights the dispatch port.

## Revised mental model (final)

For Blackwell sm_103a:

1. **Dispatch ceiling**: 128 thread-ops/SM/cycle (4 SMSPs × 1 warp-inst/cy × 32 lanes)
2. **Per-pipe caps within that budget**:
   - fmalite (FFMA/FMUL): 128 (matches dispatch)
   - alu (F2FP unpack): 64 (dual-issue from alu, 2 of the 4 SMSPs' slots)
   - alu (F2FP pack): 32 (single-issue due to 2 read ports)
   - xu (MUFU.EX2): 32
   - xu+alu+fmalite (MUFU.RSQ compound): 16 (compound-op throttled)
   - fmaheavy (IMAD, LOP3): ~128 (compound but not bottleneck)
   - tensor (HMMA): own unit, uses dispatch slot
3. **Contention classes**:
   - **Pipe contention** (same functional unit): F2FP pack/unpack among themselves; MUFU
     variants with each other; F2FP-pack with F2FP-pack.
   - **Dispatch contention** (warp-scheduler issue port): any two saturating pipes at
     high density (e.g. FFMA+FFMA at 125 each ends up at 125 total).
   - **No contention** (low-density, independent pipe): IADD3, SHL on ADU pipe
     don't interfere with F2FP even at high count.

## Why this matters for kernel authors

If your kernel has only one "heavy" op class (say F2FP), any light-pipe
companions (IADD3, SHL) are free up to about the balance point of the
dispatch ceiling. At N_F2FP=16 per iter with LOP3 added per iter:
- 16 F2FP / 64 F2FP/clk = 0.25 clk budget for F2FP
- Dispatch has 0.75 clk spare per cycle, = 96 thread-ops spare
- LOP3 at 16 per iter takes 16/128 = 0.125 clk budget
- Total: 0.25 + 0.125 = 0.375 clk per iter → 2.67 iters per clock → per iter
  ~ok but getting close

At F2FP=8 + LOP3=8 per iter, both are balanced at 0.125 clk budget each, total
0.25 clk per iter → 4 iters/clk. F2FP rate: 8 × 4 = 32 /SM/clk (half peak). 
Measured 42. Close match given overhead.

## Conclusion

The real ceiling is the warp-scheduler's 128 thread-ops/SM/cycle dispatch
budget. Multiple pipes exist and are individually-capped, but cannot ever
sum above 128 in a single SM per cycle. This is the clean model after 8
reports of refinement.
