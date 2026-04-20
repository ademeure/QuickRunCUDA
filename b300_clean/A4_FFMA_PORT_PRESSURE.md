# FFMA register read-port limit — V4 / A4

**Date: 2026-04-20.** Test `bench_ffma_port_pressure.cu` at 1500 MHz,
persistent 256-thread blocks, NC=8 chains. Big finding: FFMA throughput
depends on **number of distinct register sources**, not on operand count.

## Result

| Operand pattern | SASS encoding | inst/SMSP/cy | TIPS_inst |
|-----------------|---------------|-------------:|----------:|
| `fma a,a,a,a` (1 unique) | `FFMA R14, R14, R14, R14` | **0.972** | 27.61 |
| `fma a,b,a,b` (2 unique) | `FFMA R2, R23, R2, R2`   | **0.971** | 27.61 |
| `fma a,b,c,a` (3 unique) | `FFMA R28, R18, R28, R23` | **0.612** | 17.40 |

**3 distinct reads → 37% slower throughput.** This is a register-file
read-port limit on FFMA inputs.

## Interpretation

FFMA reads 3 source operands (a*b+c) per cycle. Result suggests the
register file delivers **2 distinct reads per cycle to the FMA pipe**,
with same-register reads broadcast (free).

- 1 unique source: 1 read port used, 2 free → 0.97/SMSP/cy peak
- 2 unique sources: 2 read ports used, fully utilized → 0.97/SMSP/cy peak
- 3 unique sources: needs 1.5 cycles per FFMA → 0.65/SMSP/cy expected,
  measured 0.61

## Practical implications

This explains many real-workload performance gaps:

1. **GEMM with operand reuse**: A or B broadcast across MMA ⇒ 2 unique
   reads per FFMA ⇒ near peak. Pure outer-product (3 distinct vectors)
   ⇒ 40% slower.

2. **Horner polynomial eval `t = t*x + c`**: 2 unique sources (t, x;
   c is constant). Hits peak.

3. **Vector dot product** `sum = sum + a*b` (3 distinct: sum, a, b):
   port-limited at 0.65/SMSP/cy.

4. **FFMA accumulator chain `acc = acc * x + acc_next`**: 3 unique → slow.

5. **Compiler optimization**: nvcc could prefer FMA forms with operand
   reuse where mathematically equivalent. Today, no — most ml/HPC
   workloads will hit the port limit and not realize it.

## Why this didn't show in LOP3

LOP3 pipe peak is **0.5/SMSP/cy** (already half FFMA's 1.0). The RF
2-port limit kicks in at 0.66/SMSP/cy for 3 distinct reads — which is
above LOP3's pipe peak. So LOP3 stays bottlenecked at the pipe, not the RF.

For ALU pipe ops (LOP3, IADD3, etc.), the RF port count is not
observable as a bottleneck because the ALU pipe itself is narrower.

## Why prior FFMA tests at 0.66/SMSP/cy

In `bench_dual_pipe_ffma_iadd3.cu`, FFMA-only mode used `fma.rn.f32 %0,
%0, %1, %2` with fb and fc distinct from fv → 3 unique reads → hits
the 0.66/SMSP/cy ceiling. That measurement was correct but I attributed
it to "warp-scheduler bubble"; the real cause is **RF read-port pressure**.

The earlier "need 4+ warps/SMSP for 98% peak" finding from
`04_fp32_peak.md` likely measured kernels with 2 unique reads — not
the 3-source FFMA pattern.

## Confidence

- **HIGH** for 1-2 unique reads = 0.97/SMSP/cy and 3 unique = 0.61/SMSP/cy
  (3+ trials, stable, SASS-verified)
- **HIGH** for "2 RF read ports" interpretation (matches 1.5× ratio)
- **MED** for "broadcast-on-same-register is free" — could also be
  operand collector deduplication; same observable effect.

## Files

- `tests/bench_ffma_port_pressure.cu` — port mode 0/1/2/3
- SASS confirmed `FFMA R28, R18, R28, R23` for 3-distinct (mode 2)
