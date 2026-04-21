# V9: Op latency ladder — FFMA/FADD/FMUL/IMAD all 4.22 cy, DFMA 63.68 cy

## 10-rule rigor walk-through

1. **Theoretical**: Hopper/Blackwell FP32 latency literature = 4-6 cy.
   FP64 throughput 1:64 → DFMA latency ~64 cy (1 issue slot every 64 cy).

2. **Measured** (serial dependency chain, single thread, CHAIN_LEN=4096):

   | Op   | Cycles / 4096 | Latency (cy) | Ratio vs FFMA |
   |------|--------------|---------------|---------------|
   | FFMA | 17,283       | **4.219**     | 1.00× (baseline) |
   | FADD | 17,283       | 4.219         | 1.00×         |
   | FMUL | 17,283       | 4.219         | 1.00×         |
   | IMAD | 17,419       | 4.252         | 1.008×        |
   | **DFMA** | 260,823  | **63.677**    | 15.1×         |

3. **Rule 3**: 4.22 cy > 4 cy lower bound. Not broken.

4. **Why FP32 ≈ IMAD ≈ 4.22 cy**: all share the FMA/ALU pipe, same latency.
   IMAD has 0.03 cy extra overhead (possibly integer-specific unit).

5. **Why DFMA 15×**: FP64 throughput is 1:64 of FP32, so per-SMSP FP64
   unit issues 1 DFMA per 64 cycles. Serial chain = chain through same
   slow unit = 63.68 cy/op.

6. **SASS**: each op emits the expected instruction:
   - FFMA → `FFMA R, R, R, R`
   - IMAD → `IMAD R, R, R, R`
   - DFMA → `DFMA R, R, R, R`

7. **Three methods**:
   - `clock64` direct measurement (per-warp at thread 0)
   - Sweep chain lengths 64-16384 → converges to 4.218 for long chains
   - Low-count startup overhead <1% at chain=256+

8. **Conclusive**: latency independent of chain length (≥256 iters):
   - chain=64: 4.265 (1% overhead)
   - chain=16384: 4.218 (converged)
   Consistent stable latency.

9. **No surprise**: matches published Hopper/Blackwell numbers.

10. **Confidence: HIGH**. Would change if:
    - Low-level thing like `setp.reg`/`mov` latency differs
    - MUFU/SHFL latency different (tested separately — MUFU was 99.49% pipe,
      latency ~4-8 cy per issue likely)

## B300 latency ladder (cy per op on 4.22 cy baseline)

| Op       | Latency | Throughput limit    | Saturation                       |
|----------|---------|----------------------|----------------------------------|
| FFMA     | 4.22 cy | 1 per cy per SMSP   | 97.64% pipe (2-source pattern)   |
| FADD     | 4.22 cy | same                | 97.65% pipe                      |
| FMUL     | 4.22 cy | same                | 97.62% pipe                      |
| IMAD     | 4.25 cy | 1 per 2 cy per SMSP | 99.7% (1:2 of FP32)              |
| DFMA     | 63.68 cy| 1 per 64 cy per SMSP| 100.00% (solo DFMA port)         |
| MUFU rsqrt | ~est 8 cy | 1 per 16 cy per SM | 99.49% XU pipe                 |

## Relation to peak throughput

Peak TFLOPS = issue_rate × SMSPs × lanes × 2.032 GHz × FLOPs/inst.
Latency determines minimum chain depth for hiding:
- FFMA chain: 1 warp × 4.22 cy latency. 8 chains ILP = 32 cy window → 1 FFMA/cy (saturated).
- DFMA chain: 64 cy latency. 1 chain fills the pipe.

The 2-source `fma a, a, b, a` avoids 3-source RF read port limit (V6 D6)
because only 2 unique registers are read — one source (`a`) is self-feeding.

## Implication for kernel design

- **FP32 pipeline depth = 4.22 cy**: need ≥4 independent FMA chains per warp
  to hide latency and saturate pipe. Our 8-chain kernel = 2× over minimum.
- **FP64 needs ≥16 independent chains** (64 cy / 4 unique per cy) but the
  single port serializes anyway. Lower ILP tolerated.
- **Mixed workload**: FFMA + IMAD on same pipe — can interleave to hide
  each other's latency.