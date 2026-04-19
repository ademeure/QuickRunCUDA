# NVFP4 CuTeDSL Performance — Throttle, Cluster, Tile, Clock Investigation
Date: 2026-04-19. Session: free-rein continuation.
Sample script: /root/cutlass/examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py (CUTLASS 4.4.2 + CuTe DSL 4.4.2)

## Headline finding: SHORT vs SUSTAINED benchmarks lie about boost clock

Single ncu metric snapshot vs `nvidia-smi dmon -s pucm` continuous sampling reveal
that under sustained NVFP4 load, B300 SXM6 hits **1095 W TDP** and SM clock
**throttles 2032 → 1455 MHz** (28% drop).

| Bench length | Per-iter | TFLOPS | Run-time clock | Power |
|--------------|---------:|-------:|---------------:|------:|
| 10 iter (~10 ms) | 1037 us | 7949 | 2032 MHz (no throttle yet) | spike to 942 W |
| 2000 iter (~2 sec) | 1187 us | 6946 | **1455 MHz throttled** | **1093 W sustained** |
| 5000 iter (~6 sec) | 1195 us | 6902 | 1455 MHz | 1095 W |

Implication: **all prior "boost MFU" measurements via short benches were
reading the unthrottled first-iter clock, not the steady-state clock.**
The "MFU climbs at low clock" effect is partly real (staging keeps up at
low clock) but also partly an artifact of comparing against the wrong
spec clock (2032) when the actual run clock was 1455.

## Per-active-SM MFU re-derived at sustained clock

Cluster (2,4) launches grid (2,4,15)×224 = 120 blocks → only **120 of 148 SMs active**
(15 clusters × 8 CTAs). Effective per-active-SM peak at run-clock C =
15 PF × (C/2032) × (120/148).

| Run | TFLOPS | Run-clock | Active SMs | Peak active | **MFU/active-SM** | TF/W |
|-----|-------:|----------:|-----------:|------------:|------------------:|-----:|
| Boost sustained | 6902 | 1455 MHz | 120 | 8710 | **79.2%** | 6.30 |
| 510 MHz lock | 2698 | 510 MHz | 120 | 3053 | **88.4%** | (idle pwr) |

(Per-total-SM MFU at boost = 46% of 15-PF spec; at 510 MHz lock = 18% of 15-PF spec.)

**Conclusion**: this CuTeDSL example reaches **~79–88% of compute SoL per active SM**,
but the cluster_size=8 design forfeits 28 of 148 SMs (19%), pushing
total-SM MFU below 50%. cuBLAS NVF4 catalog 10.8 PF at 940 W (11.5 TF/W) is
1.8× more power-efficient — it doesn't hit the 1100 W TDP cap, so doesn't
throttle, and likely uses all 148 SMs.

## Cluster shape sweep (mma_tiler 256×256, 16384² K=15360, M=N=K=16K)

**At default boost (run-clock 1455 MHz sustained throttle):**

| Cluster shape | Cluster size | Wall TFLOPS | %15PF |
|---------------|-------------:|------------:|------:|
| (2, 1) | 2 | 6111 | 40.7% |
| (2, 2) | 4 | 7460 | 49.7% |
| **(2, 4)** | 8 | **7949** | **53.0%** ← short-bench (sustained ~6900) |
| (4, 1) | 4 | 5440 | 36.3% |
| (4, 2) | 8 | 6979 | 46.5% |
| (4, 4) | 16 | 7495 | 50.0% |

(1, *) shapes raise TypeError "Unsupported testcase" — kernel requires M≥2 for 2cta MMA.

**At -lgc 510 MHz (no throttle):**

| Cluster | TFLOPS | %15PF | MFU at-clock |
|---------|-------:|------:|-------------:|
| **(2, 1)** | **3280** | **21.9%** | **87.1% per-total-SM** ← optimum |
| (2, 2) | 2944 | 19.6% | 78.1% |
| (2, 4) | 2696 | 18.0% | 71.6% |
| (4, 1) | 2937 | 19.6% | 78.0% |
| (4, 2) | 2554 | 17.0% | 67.8% |
| (4, 4) | 2629 | 17.5% | 69.8% |

**Crossover insight**: At low clock (no throttle), small cluster_size=2 wins
because all 148 SMs become active and the slow compute isn't out-running
DRAM. At boost, large cluster_size=8 wins despite leaving 28 SMs idle —
data reuse via cluster shared SMEM beats raw SM count when compute is
the bottleneck (or when memory subsystem can't keep up).

## Tile shape sweep (16384² K=15360, BOOST — short-bench numbers)

Only **powers of 2** in M and N supported (192 raises TypeError).

| Tile | (2,1) | (2,2) | (2,4) |
|------|------:|------:|------:|
| 128×128 | 4618 | 4389 | 4540 |
| 128×256 | 5049 | 5394 | 5195 |
| 256×128 | 3266 | 2769 | 2577 ← bad (asym B-load) |
| **256×256** | 6181 | 7458 | **7949** ← best |

**At -lgc 510 MHz** flips: 256×256/(2,1) = 3280 TF = 87.1% MFU = best.

## ncu profile (16384² K=15360, cluster (2,4), default ncu clock 1.87 GHz)

```
sm__throughput.avg.pct_of_peak_sustained_elapsed:    67.21 %
sm__cycles_active.avg.pct_of_peak_sustained_elapsed: 80.15 %
l1tex__throughput.avg.pct_of_peak_sustained_elapsed: 52.69 %
lts__throughput.avg.pct_of_peak_sustained_elapsed:   47.50 %
dram__throughput.avg.pct_of_peak_sustained_elapsed:  37.02 %
sm__sass_inst_executed_op_utcmma.sum.per_second:      640 M/s
gpu__time_duration.sum:                              1025 us (matches wall-clock 1037)
```

ncu --clock-control none + -lgc 510 (BIG case):
```
SM Throughput:    74.77 % (climbs from 67% → 75% as clock drops)
L1/TEX Cache:     64.69 % (climbs 53% → 65%)
L2 (lts):         24.85 % (drops 47% → 25% — has headroom at low clock)
DRAM:             12.43 % (drops 37% → 12%)
sm_active:        80.10 % (unchanged)
utcmma rate:      214 M/s (= 22% better-than-linear vs clock ratio)
```

**SM Throughput climbs as clock drops** → some non-clock-scaled overhead
(TMA fill latency, mbarrier coordination) dominates at boost. utcmma rate
itself is 22% more efficient at low clock for the same reason.

## Reproducibility commands

Build sample patched to print perf:
```bash
cp /root/cutlass/examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py /tmp/sm103_perf.py
sed -i 's|^    run($|    _ex = run(|' /tmp/sm103_perf.py
sed -i 's|^    print("PASS")|    flops = 2*args.mnkl[0]*args.mnkl[1]*args.mnkl[2]*args.mnkl[3]; print(f"{_ex:.2f}us  {flops/(_ex*1e-6)/1e12:.1f} TF  ({flops/(_ex*1e-6)/1e12/15000*100:.1f}% of 15PF)")|' /tmp/sm103_perf.py
```

Sustained boost (true throttle behavior):
```bash
nvidia-smi dmon -i 0 -s pucm -d 1 -c 12 &
CUDA_VISIBLE_DEVICES=0 python3 /tmp/sm103_perf.py \
  --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16 \
  --c_dtype Float16 --mma_tiler_mn 256,256 --cluster_shape_mn 2,4 \
  --mnkl 16384,16384,15360,1 --warmup_iterations 20 --iterations 5000 --skip_ref_check
```

ncu with custom clock lock (CRITICAL: --clock-control none):
```bash
nvidia-smi -i 0 -lgc 510
CUDA_VISIBLE_DEVICES=0 ncu --clock-control none --print-kernel-base mangled --csv \
  --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__sass_inst_executed_op_utcmma.sum.per_second,sm__cycles_elapsed.avg.per_second \
  python3 examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py \
  ...args...
```

## Confidence

- **HIGH** on throttle behavior (dmon directly shows 1095 W → 1455 MHz)
- **HIGH** on cluster_shape sweep (3 methods agree, 2 GPUs identical)
- **HIGH** on tile_shape sweep within powers-of-2 (no other shapes valid)
- **HIGH** that data is RANDOM (verified script source: TensorInitType.RANDOM default)
- **MED** on per-active-SM MFU 79.2% — depends on assumption that all 120 launched blocks stay resident the entire kernel (true for persistent design but unverified per-cluster)
- **MED** that cuBLAS at 11.5 TF/W vs CuTeDSL 6.3 TF/W means cuBLAS uses all 148 SMs — needs sustained dmon test to verify

## What would change conclusions

- If cuBLAS NVF4 also throttles: the 11.5 TF/W catalog number is wrong (was short-bench)
- If we modify CuTeDSL to use all 148 SMs (smaller cluster, smaller SHMEM/block):
  the boost number could push up but probably with worse data reuse
- If TDP could be raised >1100 W: would unlock the un-throttled boost behavior
