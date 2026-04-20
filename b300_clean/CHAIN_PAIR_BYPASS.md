# Per-pair chain latency + register-forwarding bypass map — V4

**Date: 2026-04-20.** Single warp single block, 1000-deep chain through one
register, anti-DCE perturbation via runtime u2. SASS-verified for each pair.

## Key finding: TWO forwarding clusters with +2 cy cross penalty

Pure-pipe chain latency is **4 cy** (matches catalog). When chaining ops
from the SAME cluster, latency stacks linearly (8 cy/pair). When crossing
clusters, an extra +2 cy "register file round-trip" penalty applies:

| Cluster | Members |
|---------|---------|
| **Cluster A** (FMA-style) | FFMA, IMAD (and IMAD.IADD when emitted) |
| **Cluster B** (bit-manip) | LOP3, IADD3-as-IADD3, SHF, PRMT |

## Pair latency matrix (cy/pair)

```
        FFMA   IMAD   LOP3  IADD3*  SHF   PRMT 
FFMA    8.04   8.04  10.04  10.04  10.03  10.04 
IMAD    8.04   8.04  10.04  10.04  10.04  10.04 
LOP3   10.04  10.04   8.04   8.04   8.04   8.04 
IADD3* 10.04  10.04   8.04   8.04  10.04   8.04   ← IADD3→SHF anomaly
SHF    10.03  10.04   8.04  10.03   8.05   8.04   ← SHF→IADD3 anomaly
PRMT   10.04  10.04   8.04   8.04   8.04   8.04 
```

`*` IADD3 entries reflect what the PTX `add.u32; add.u32` DSL produces.
The compiler may rewrite to IMAD.IADD in some contexts (see anomaly note).

## SASS verification of the IADD3↔SHF "anomaly"

When testing `OP_A=IADD3, OP_B=SHF`, the SASS census showed:

```
100 IMAD.IADD
 99 LEA.HI
 ...
```

NOT 100 IADD3. The PTX `add.u32 %0,%0,%1; add.u32 %0,%0,%2` was rewritten
by nvcc to `IMAD.IADD` (= integer multiply-and-add with implied multiplier
of 1). IMAD.IADD belongs to **Cluster A** — so IADD3→SHF is actually
IMAD.IADD→SHF, which IS cross-cluster A→B = 10 cy.

**Anomaly resolved**: the pair matrix is fully consistent with the
two-cluster model once you correct for what nvcc actually emits.

## Practical consequence

For minimum chain latency, **stay within one cluster**:
- FMA-heavy code: chain FFMA→IMAD→FFMA→IMAD (8 cy/pair)
- Bit-manip code: chain LOP3→PRMT→LOP3→SHF (8 cy/pair)
- Mixing FFMA with LOP3 etc. costs 2 extra cy per transition

For 1000-step chain:
- Pure cluster: 1000 × 4 = 4000 cy
- Half-and-half cross-cluster: 1000 × 5 = 5000 cy → 25% slower

## Why? Speculative explanation

Likely two physical operand-collector / writeback networks:
- One serving FMA pipe + IMAD multiplier
- One serving the ALU/bit-manip array
- Cross-network requires writeback to RF + re-fetch (+1-2 cy)
- Within-network can use forwarding bypass

This matches Hopper-style architecture where multiple pipes share an
SMSP issue port but have separate operand collectors.

## Anti-DCE caveat (important methodology lesson)

First measurement showed IADD3→IADD3 = 0.02 cy/pair. SASS revealed only 4
IADD3 in the body — the compiler **semantically folded** the chain
(asm volatile preserves the inline asm but doesn't prevent constant
propagation across instances).

Fix: perturb constants with runtime `u2` value (passed as 0). Then asm
operands change per call, defeating the fold.

## Confidence

- **HIGH** for two-cluster model (matrix self-consistent, SASS-verified)
- **HIGH** for "+2 cy cross-cluster penalty" (16 pairs all show 8 or 10)
- **HIGH** for the IADD3↔SHF anomaly being SASS-emission-driven
- **MED** for "operand-collector explanation" — speculative,
  not architecturally documented

## Files

- `tests/bench_chain_pair_matrix.cu` — 6×6 pair sweep
- `tests/bench_chain_latency_mix.cu` — earlier 5-mode test
- `tests/bench_ffma_chain_latency.cu` — pure FFMA chain (4.02 cy)
