# NVFP4 cuBLAS B300 Comprehensive A/B Shape Sweep
Date: 2026-04-19. Session continuation.
Setup: cuBLASLt LtMatmul, ZERO data inputs (cudaMemset 0x00), sustained ≥50 iters.

## Per-clock optimum + at-clock MFU

| Clock | Peak shape | TFLOPS | %15PF | **%spec@clock** |
|------:|-----------|-------:|------:|----------------:|
| 510 MHz | M=N=16384 K=61440 | 3558 | 23.7% | **94.5%** |
| 1500 MHz | M=N=8192 K=38400 | 9273 | 61.8% | 83.7% |
| Boost (2032) | M=N=8192 K=38400 | **11054** | **73.7%** | 73.7% |

cuBLAS at 510 MHz hits **94.5% MFU at-clock** — within 5% of theoretical
peak. Coordination overhead is a tiny fraction of the slow per-cycle time.

## TOP 10 by absolute TFLOPS at each clock

### -lgc 510 MHz (deep-K dominates):
| M | N | K | TFLOPS | %15PF | %@510 |
|--:|--:|--:|-------:|------:|------:|
| 16384 | 16384 | 61440 | 3558 | 23.7% | 94.5% |
| 16384 | 16384 | 46080 | 3523 | 23.5% | 93.6% |
| 16384 | 16384 | 38400 | 3496 | 23.3% | 92.8% |
| 14336 | 14336 | 38400 | 3480 | 23.2% | 92.4% |
| 24576 | 24576 | 30720 | 3476 | 23.2% | 92.3% |
| 20480 | 20480 | 30720 | 3469 | 23.1% | 92.1% |
| 16384 | 20480 | 30720 | 3464 | 23.1% | 92.0% |
| 16384 | 16384 | 30720 | 3456 | 23.0% | 91.7% |
| 20480 | 16384 | 30720 | 3455 | 23.0% | 91.7% |
| 10240 | 10240 | 38400 | 3447 | 23.0% | 91.5% |

### -lgc 1500 MHz:
| M | N | K | TFLOPS | %15PF | %@1500 |
|--:|--:|--:|-------:|------:|-------:|
| 8192 | 8192 | 38400 | 9273 | 61.8% | 83.7% |
| 8192 | 8192 | 46080 | 9091 | 60.6% | 82.1% |
| 8192 | 8192 | 30720 | 8996 | 60.0% | 81.2% |
| 16384 | 16384 | 61440 | 8949 | 59.7% | 80.8% |
| 32768 | 32768 | 15360 | 8858 | 59.1% | 80.0% |
| 16384 | 16384 | 46080 | 8798 | 58.7% | 79.5% |
| 16384 | 16384 | 12288 | 8790 | 58.6% | 79.4% |
| 32768 | 8192 | 15360 | 8783 | 58.6% | 79.3% |
| 24576 | 24576 | 30720 | 8758 | 58.4% | 79.1% |
| 16384 | 16384 | 18432 | 8753 | 58.4% | 79.0% |

### Boost (no -lgc):
| M | N | K | TFLOPS | %15PF |
|--:|--:|--:|-------:|------:|
| **8192** | **8192** | **38400** | **11054** | **73.7%** ← global peak |
| 8192 | 8192 | 30720 | 10786 | 71.9% |
| 16384 | 16384 | 12288 | 10500 | 70.0% |
| 32768 | 32768 | 15360 | 10463 | 69.8% |
| 32768 | 8192 | 15360 | 10396 | 69.3% |
| 8192 | 32768 | 15360 | 10357 | 69.0% |
| 16384 | 16384 | 18432 | 10322 | 68.8% |
| 8192 | 8192 | 46080 | 10299 | 68.7% |
| 8192 | 8192 | 12288 | 10276 | 68.5% |
| 16384 | 16384 | 61440 | 10239 | 68.3% |

## Key observations

1. **8192² K=38400 is global optimum at boost AND 1500 MHz** (same shape wins both)
2. **At 510 MHz, deep-K square 16384²×61440 wins** — slow compute lets memory keep up,
   bigger problem amortizes better
3. **Bottom of MFU at all clocks: small problems (2K²K=3K) — too few tiles to fill SMs**
4. Tall-skinny vs wide: at boost both 32768×8192 K=15360 (69.3%) and 8192×32768 K=15360 (69.0%)
   nearly tie — cuBLAS handles asymmetry equally well unlike CuTeDSL's (2,4) cluster
5. Rectangular M=N peak at boost: 16384×16384 K=12288 = 70% (vs 73.7% best square)
6. Llama-style (8K × 14K K=8K): only 53-58% MFU — K=8064 is too narrow

## Shape selection guidelines for each clock regime

### At boost (~2032 MHz):
- Compute-bound regime; minimize overhead/compute ratio
- Prefer M=N=8192 with K∈[12K, 46K]
- Avoid K<6K (too little compute amortization)
- Avoid K>46K (HBM-bound)
- Square problems beat asymmetric

### At 1500 MHz:
- Same optimal as boost (M=N=8192 K=38400)
- Slightly more tolerant of bigger problems (16384² in top 5)

### At 510 MHz:
- Memory-headroom-bound regime
- Prefer LARGE problems (M=N=16384 or 24576)
- Deep K (38K-61K) wins
- Achieves 94.5% MFU at-clock — near-perfect compute utilization

## Reproducibility

```bash
# Build cuBLAS NVF4 minimal binary
nvcc -gencode arch=compute_103a,code=sm_103a -O3 \
  /tmp/cublas_nvf4_minimal.cu -lcublas -lcublasLt -o /tmp/cublas_nvf4_zero
# data init via cudaMemset(0x00) in source

# Sweep
for clk in 510 1500 boost; do
  if [ "$clk" = "boost" ]; then nvidia-smi -i 0 -rgc; else nvidia-smi -i 0 -lgc $clk; fi
  for shape in "8192 8192 38400" ...; do
    /tmp/cublas_nvf4_zero $shape 100  # iters=100
  done
done
```

## Confidence
- HIGH on absolute peak 11054 TF (4 verifications across this session)
- HIGH on 510 MHz 94.5% MFU at-clock (multiple shapes hit ≥91% at-clock)
- HIGH on optimum shape shift with clock
- MED on whether even better shapes exist beyond tested grid (didn't sweep > M=N=32768)
