# D6: Register File read ports per SMSP — definitively 2 reads/cy

## Theoretical
- FFMA needs 3 reads (a, b, c) and 1 write (d)
- If RF has N read ports/cycle/SMSP, FFMA throughput = min(1, N/3) inst/cy
- Common architectures (Volta+): 2 RF read ports + operand reuse cache for broadcast operands

Predictions:
- 2 reads (broadcast operand): 1 FFMA/cy = 1.0 cy/fma
- 3 unique reads (no reuse): 1 FFMA/1.5 cy = 0.667 fma/cy

## Methodology rigor
- ILP-rich (16 indep accumulator chains) to saturate beyond chain latency
- N=16 inner unroll to amortize loop overhead (~23 cy/iter floor)
- Two configurations: (a) broadcast `za` (compiler emits `.reuse`); (b) per-chain `za[k]` (defeats reuse)
- SASS-verified `.reuse` count in both: (a) 255/256 with reuse; (b) 0/256

## Measured
| Config | NC=1 | NC=2 | NC=4 | NC=8 | NC=16 | Peak fma/cy |
|--------|------|------|------|------|-------|--------------|
| Broadcast za + .reuse | 4.44 | 2.31 | 1.19 | 1.09 | 1.05 cy/fma | 0.96 |
| Per-chain za, no reuse | 4.56 | 2.25 | 1.63 | 1.56 | 1.53 cy/fma | 0.65 |

Theoretical 2 reads + 1 reuse → 1 cy. Measured 1.05 cy → 95% of theoretical.
Theoretical 3 reads / 2 ports → 1.5 cy. Measured 1.53 cy → 98% of theoretical.

## Conclusion
**B300 SMSP has 2 register-file read ports per cycle.** The operand reuse cache (visible as `.reuse` in SASS) provides an effective 3rd port for any operand that's broadcast across consecutive instructions.

For practical FFMA-heavy kernels:
- Maximum throughput requires the compiler to emit `.reuse` on at least one operand
- Pure outer-product GEMM (3 fully-distinct sources every FFMA) caps at **65% of peak FFMA throughput**
- Broadcast or shared operand patterns (typical of inner-product GEMM where one operand is reused across FFMAs) hit **96% of peak**

This is THE constraint behind the FFMA peak gap: pure FFMA microbenchmarks routinely measure ~74-77 TFLOPS (96-100% of theoretical 76 at boost) only because broadcast patterns dominate. A worst-case all-distinct FFMA load would hit ~50 TFLOPS.

## Confidence: HIGH
Validated by:
- Clean ratio: 0.96/0.65 = 1.48 vs theoretical 1.50
- SASS-confirmed reuse count
- Saturating curve at NC≥4 in both modes
- Cross-checks with A1 finding (1.19 cy/fma at NC=4 with broadcast)

What would change it: if a future test shows 3-unique-source FFMA at >0.7 fma/cy, would mean 3+ read ports.
