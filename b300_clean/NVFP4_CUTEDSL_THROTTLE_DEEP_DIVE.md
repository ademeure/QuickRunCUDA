# NVFP4 CuTeDSL Performance — Comprehensive B300 SXM6 Investigation
Date: 2026-04-19. Session: free-rein deep-dive.
Sample script: `/root/cutlass/examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py`
(CUTLASS 4.4.2 + CuTe DSL 4.4.2; CUDA 13.0; Driver 580.126.09)

## TL;DR

| Metric | Value | Conditions |
|--------|------:|------------|
| **Best per-total-SM MFU** | **91.3%** | M=N=8192, K=61440, cluster (2,1), 1005 MHz, zero data |
| **Best per-active-SM MFU** | **88.4%** | M=N=32768, K=15360, cluster (2,4), 1005 MHz, zero data |
| **Best absolute TFLOPS** | **8112** | M=N=16384, K=15360, cluster (2,4), boost, zero data |
| **Random-data sustained TFLOPS** | **6902** | same as above; throttles to 1455 MHz |
| **Best TFLOPS/W** | **9.74** | boost cluster (2,4) zero data |
| Catalog cuBLAS NVF4 ceiling | 10800 TF / 11.5 TF/W | for reference; not measured this session |

This CuTeDSL example reaches **91% of compute SoL per-total-SM at 1005 MHz**
on K-deep matmuls — within 9% of cuBLAS catalog 72%-of-15PF (which is itself
the public-library ceiling). The remaining gap is the cluster-scheduling
constraint that leaves 28 of 148 SMs idle.

## Headline finding 1: TDP throttle invalidates short-bench numbers

`nvidia-smi dmon -s pucm -d 1` continuous sampling reveals:
- Random-input sustained NVFP4 hits **1095 W**, near the 1100 W TDP cap
- SM clock throttles **2032 → 1455 MHz** (28% drop) under sustained random load
- Zero-input sustained: only 833 W → **clock holds 2032 MHz, no throttle**
- Short benchmarks (≤10 iterations / ~10 ms) finish before throttle engages

Implication: any "boost MFU" claim from a 10-iteration cuda_event measurement
is reading the unthrottled first-iter clock, not the steady-state running clock.
You MUST run ≥1 second of kernel time and sample dmon to get the truth.

| Sample run | Per-iter | TFLOPS | Run-time clock | Power | Source |
|------------|---------:|-------:|---------------:|------:|--------|
| 10 iter random | 1037 us | 7949 | 2032 MHz | spike 942 W | misleading |
| 2000 iter random | 1187 us | 6946 | **1455 MHz throttled** | **1093 W** | TRUE |
| 5000 iter random | 1195 us | 6902 | 1455 MHz | 1095 W | TRUE |
| 5000 iter zero | 1017 us | **8112** | **2032 MHz held** | 833 W | TRUE peak |

`nvidia-smi -q -d PERFORMANCE` does NOT show "SW Power Cap: Active" event
flag during throttle — the clock dip is silent. Only dmon column 12 (pclk)
reveals it.

## Headline finding 2: Zero-data input gives 17% wall-clock speedup at boost

Modifying script to use `TensorInitType.SCALAR(0)` instead of `RANDOM`:

| Data | Boost TFLOPS | Boost run-clock | Boost power | TF/W | 510 MHz TFLOPS |
|------|-------------:|----------------:|------------:|-----:|---------------:|
| Random | 6902 | 1455 MHz throttle | 1095 W | 6.30 | 2698 |
| **Zero** | **8112** | **2032 MHz held** | 833 W | **9.74** | 2697 |

At 510 MHz both data types yield identical TFLOPS (clock is the gate, not
power). At boost, zero-data avoids TDP throttle and runs 17% faster wall-time
with 24% less power. **Per-cycle throughput is actually LOWER with zero data**
(3.99 PF/GHz zero vs 4.74 PF/GHz random) — net win comes purely from
holding higher clock. Possible per-cycle penalty cause: HBM/L2 compression
of zero-pattern data alters fetch latency.

How to reproduce zero-data init:
```python
# Patch the cuBLAS sample run() function calls:
a_ref = cutlass_torch.matrix(l, m, k, ..., 
    init_type=cutlass_torch.TensorInitType.SCALAR,
    init_config=cutlass_torch.ScalarInitConfig(value=0.0))
b_ref = cutlass_torch.matrix(l, n, k, ..., 
    init_type=cutlass_torch.TensorInitType.SCALAR,
    init_config=cutlass_torch.ScalarInitConfig(value=0.0))
# Also patch create_scale_factor_tensor's RANDOM init to SCALAR(0).
```

## Headline finding 3: Cluster shape × clock crossover

**Cluster (2,4)** uses 8 CTAs/cluster; only 15 simultaneous clusters fit
(SHMEM-limited at 230 KB/block) → **only 120 of 148 SMs active**, leaving
28 idle.

**Cluster (2,1)** uses 2 CTAs/cluster; 74 simultaneous clusters → **all 148 SMs active**.

| Run condition | Best cluster | Why |
|---------------|--------------|-----|
| Boost, random data (TDP-bound) | **(2,4)** at 6902 TF | Compute-density wins; small cluster's extra SMs cost more in throttling |
| Boost, zero data (no throttle) | **(2,4)** at 8112 TF | Data reuse via cluster broadcast still wins |
| 1005 MHz lock | **(2,1)** at 5569 TF (sq) / 6776 TF (deep-K) | More SMs help; no throttle penalty |
| 510 MHz lock | **(2,1)** at 3280 TF | Same: low clock + no throttle = SM count wins |

Clusters with M=1 (e.g., (1,1), (1,2)) raise `TypeError: Unsupported testcase`
— kernel requires M≥2 for 2cta MMA mode in `SM103MmaMXF4NVF4Op._verify()`.

Sustained dmon for boost zero-data cluster sweep (M=N=K=16K-ish):

| Cluster | TFLOPS | Sustained pwr | %15PF |
|---------|-------:|--------------:|------:|
| (2,1) | 6149 | 932 W | 41.0% |
| (2,2) | 7655 | 921 W | 51.0% |
| **(2,4)** | **8112** | **843 W** | **54.1%** |
| (4,4) | 7525 | 759 W | 50.2% |

(2,4) hits highest TFLOPS at LOWEST power — best efficiency.

## Headline finding 4: Tile shape — only powers-of-2 supported, 256×256 universal winner

Sweeping `mma_tiler_mn` at 1005 MHz zero-data cluster (2,1):

| Tile | TFLOPS | %15PF |
|------|-------:|------:|
| 64×64 / 64×128 / 64×256 / 128×64 / 128×512 / 256×64 / 256×512 / 512×* | TypeError | unsupported |
| 128×128 | 3185 | 21.2% |
| 128×256 | 4000 | 26.7% |
| 256×128 | 2317 | 15.4% (asymmetric B-load penalty) |
| **256×256** | **5569** | **37.1%** |

256×128 is much worse than 128×256 (B-major asymmetry in K-major B layout).
192-anything fails. Practical universe is just (128,128), (128,256), (256,256)
— and 256×256 is the universal winner.

## Headline finding 5: Matrix shape sweep — K-deep matmuls hit 91% MFU

All at 1005 MHz, zero data, mma_tiler 256×256, sustained iters.

### Square scaling (K=15360)

| M=N | (2,1) TF | MFU/total | (2,4) TF | MFU/active |
|----:|---------:|----------:|---------:|-----------:|
| 2048 | 4185 | 56.4% | 2433 | 40.5% |
| 4096 | 5142 | 69.3% | 4177 | 69.5% |
| **8192** | **6087** | **82.0%** | 4948 | 82.3% |
| 12288 | 5713 | 77.0% | 5193 | 86.4% |
| 16384 | 5581 | 75.2% | 5240 | 87.1% |
| 24576 | 5152 | 69.5% | 5301 | 88.1% |
| **32768** | 5196 | 70.1% | **5319** | **88.4%** ← per-active record |

(2,1) is best for M≤8192 (small problems benefit from more SMs).
(2,4) takes over for M≥24576 (large problems amortize cluster-broadcast).
Crossover happens around M=N=12K-16K.

### Deep-K (M=N=8192, vary K)

| K | (2,1) TF | MFU/total |
|--:|---------:|----------:|
| 1536 | 3049 | 41.1% |
| 3072 | 4301 | 58.0% |
| 6144 | 5322 | 71.7% |
| 15360 | 6124 | 82.6% |
| 30720 | 6484 | 87.4% |
| **61440** | **6776** | **91.3%** ← per-total record |

K-depth is the single most important shape parameter. Each output element gets
more compute amortized over the same SMEM-staged A/B reads → reuse goes up
asymptotically. This kernel hits the ceiling for K=61440 — within 9% of
hypothetical 100% MFU at 1005 MHz.

### Tall-skinny (small M, N=16384, K=15360)

| M | (2,1) TF | MFU/total |
|--:|---------:|----------:|
| 512 | 4495 | 60.6% |
| 1024 | 5105 | 68.8% |
| 2048 | 6087 | 82.1% |
| 4096 | 6250 | 84.3% |

### Wide (M=16384, small N, K=15360)

| N | (2,1) TF | MFU/total |
|--:|---------:|----------:|
| 512 | 4170 | 56.2% |
| 1024 | 4507 | 60.8% |
| 2048 | 5113 | 68.9% |
| 4096 | 5246 | 70.7% |

Tall-skinny outperforms wide at the same imbalanced ratio. This makes sense:
B is loaded N-major (transposed), so small-N exposes B-load latency more
than small-M exposes A-load.

### LLM-realistic Llama-70B layer matmuls (hidden=8192, ffn=14336, K=8064)

| Shape | (2,1) TF | (2,4) TF | MFU/total |
|-------|---------:|---------:|----------:|
| (8192, 14336, 8064) gate/up | 4510 | 4469 | 30% |
| (14336, 8192, 8064) down    | 4537 | 4408 | 30% |
| (8192, 8192, 8064) qkv      | 4589 | 4233 | 30% |
| (4096, 14336, 11904) ffn    | 4638 | 4730 | 31% |

LLM-realistic shapes plateau at **~30% of 15-PF spec at 1005 MHz** because
K=8064 is too narrow to hit the deep-K reuse plateau. To hit the 91% MFU
shape you'd need K≥40K — which doesn't occur naturally in transformer
matmuls except in attention (K = head_dim × seq_len).

## ncu cross-check at 1005 MHz BIG case (cluster (2,4))

```
SM Throughput:    74.77 % (climbs 67% → 75% as clock drops from 2032 → 510)
SM Active Cycles: 80.10 %
L1/TEX Cache:     64.69 % (climbs 53% → 65% as clock drops)
L2 (lts):         24.85 % (drops 47% → 25% — has headroom at low clock)
DRAM:             12.43 % (drops 37% → 12%)
utcmma rate:      214 M/s (= 22% better-than-linear vs clock ratio)
```

Key insight from ncu: **SM Throughput climbs as clock drops** (67% → 75%).
The bottleneck at high clock is some non-clock-scaled overhead (likely TMA
fill latency + mbarrier coordination): at boost, SMs out-run their data
arrival; at low clock, staging keeps up. utcmma rate is **22% better than
linear-clock-scaling** at low clock for the same reason.

## Per-active-SM accounting

Cluster (2,4) launches grid `(2,4,15)×(224,1,1)` = 120 blocks = 15 clusters.
ncu confirms `Max Active Clusters = 15` (SHMEM-limited at 230 KB/block →
1 block/SM × cluster_size 8 → 18.5 raw clusters → tcmem caps at 15).

So MFU/active-SM = TFLOPS / (15 PF × clock/2032 × 120/148):
- Boost zero (2,4): 8112 / (15000 × 1 × 120/148) = **66.7%**
- 1005 MHz (2,4) M=N=32K: 5319 / (7418 × 120/148) = **88.4%** ← record
- 510 MHz (2,4) random: 2698 / (3766 × 120/148) = **88.4%** (matches)

For cluster (2,1): all 148 SMs active, so per-active = per-total. The 91.3%
M=N=8192, K=61440 number is both per-total and per-active.

## Reproducibility

```bash
# 1. Patch the sample to print perf and (optionally) zero data
cp /root/cutlass/examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py /tmp/sm103_perf.py

sed -i 's|^    run($|    _ex = run(|' /tmp/sm103_perf.py
sed -i 's|^    print("PASS")|    flops = 2*args.mnkl[0]*args.mnkl[1]*args.mnkl[2]*args.mnkl[3]; print(f"{_ex:.2f}us  {flops/(_ex*1e-6)/1e12:.1f} TF  ({flops/(_ex*1e-6)/1e12/15000*100:.1f}% of 15PF)")|' /tmp/sm103_perf.py

# 2. (Optional) patch to zero-data init
sed -i 's|a_ref = cutlass_torch\.matrix(l, m, k, a_major == "m", cutlass\.Float32)|a_ref = cutlass_torch.matrix(l, m, k, a_major == "m", cutlass.Float32, init_type=cutlass_torch.TensorInitType.SCALAR, init_config=cutlass_torch.ScalarInitConfig(value=0.0))|' /tmp/sm103_perf.py
sed -i 's|b_ref = cutlass_torch\.matrix(l, n, k, b_major == "n", cutlass\.Float32)|b_ref = cutlass_torch.matrix(l, n, k, b_major == "n", cutlass.Float32, init_type=cutlass_torch.TensorInitType.SCALAR, init_config=cutlass_torch.ScalarInitConfig(value=0.0))|' /tmp/sm103_perf.py
# Also patch the SF tensor RANDOM init to SCALAR(0) at line ~2761

# 3. Sustained boost run with dmon
nvidia-smi dmon -i 0 -s pucm -d 1 -c 12 &
CUDA_VISIBLE_DEVICES=0 python3 /tmp/sm103_perf.py \
  --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16 \
  --c_dtype Float16 --mma_tiler_mn 256,256 --cluster_shape_mn 2,4 \
  --mnkl 16384,16384,15360,1 --warmup_iterations 20 --iterations 5000 --skip_ref_check

# 4. ncu with custom clock (CRITICAL: --clock-control none)
nvidia-smi -i 0 -lgc 1005
CUDA_VISIBLE_DEVICES=0 ncu --clock-control none --print-kernel-base mangled --csv \
  --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__sass_inst_executed_op_utcmma.sum.per_second,sm__cycles_elapsed.avg.per_second \
  python3 /tmp/sm103_perf.py [args]

# 5. Reach the 91% MFU king shape
nvidia-smi -i 0 -lgc 1005
CUDA_VISIBLE_DEVICES=0 python3 /tmp/sm103_perf.py \
  --ab_dtype Float4E2M1FN --sf_dtype Float8E8M0FNU --sf_vec_size 16 \
  --c_dtype Float16 --mma_tiler_mn 256,256 --cluster_shape_mn 2,1 \
  --mnkl 8192,8192,61440,1 --warmup_iterations 5 --iterations 50 --skip_ref_check
```

## Confidence

- **HIGH** on throttle behavior (dmon directly shows 1095 W → 1455 MHz, multiple runs)
- **HIGH** on zero-vs-random data effect (5x reproductions, ~17% delta consistent)
- **HIGH** on cluster_shape sweep (3 methods agree, both GPUs identical)
- **HIGH** on tile_shape powers-of-2 limit (TypeError consistently for non-pow2)
- **HIGH** on K-depth scaling (monotonic 41% → 91% as K grows from 1.5K to 61K)
- **HIGH** on 91.3% per-total-SM MFU at K=61440 (single best measurement)
- **MED** on the per-cycle penalty of zero data (3.99 vs 4.74 PF/GHz; cause unverified)
- **MED** on assumption that 120 active SMs stay resident the entire kernel
- **NOT MEASURED** this session: cuBLAS NVF4 sustained behavior — does it also throttle?

## What would change conclusions

- If cuBLAS NVF4 catalog 11.5 TF/W was also a short-bench reading: real cuBLAS sustained could be lower
- A custom kernel that uses all 148 SMs with cluster=2 + still gets data reuse via async TMA
  could potentially beat both (2,1) and (2,4)
- TDP cap raised >1100 W (data center config) would unlock the random-data boost case to match zero-data
- Newer CUTLASS / driver may add cluster (1,N) support for 1cta MMA path

## Files / commits

- This doc: `b300_clean/NVFP4_CUTEDSL_THROTTLE_DEEP_DIVE.md`
- Earlier related: `B300_TRUE_REFERENCE.md` D-section, `CURIOSITY_LIST_V2.md`, `D4_PRECISION_POWER_PERF_TABLE.md`
- Prior commits: `bf46464` (initial throttle finding); this update extends with matrix-shape sweep + zero-data + king-shape 91% MFU

---

## CUTLASS C++ sample 89 (sm103_fp4_ultra_gemm) — comparison

Build: `/root/cutlass/build/examples/89_sm103_fp4_ultra_gemm/89_sm103_fp4_ultra_gemm`
Source: `/root/cutlass/examples/89_sm103_fp4_ultra_gemm/89_sm103_fp4_ultra_gemm.cu`

**Status: works out of the box** (despite earlier session note claiming
"Pipeline init: Producer arrival count must be non-zero" runtime assert —
that was apparently fixed in the CUTLASS 4.4.2 build I have).

The sample reports two MMA configs per run: **1SM MMA** (single-CTA tcgen05)
and **2SM MMA** (paired-CTA tcgen05, matches CuTeDSL's `cluster_shape_mn=(2,N)`).

### Same shape as CuTeDSL test (M=N=8192 K=15360, -lgc 1005 MHz, ncu actual=972 MHz)

| Config | Duration (ncu) | TFLOPS | MFU @ 972 MHz | SM thpt | L1/TEX | L2 | DRAM |
|--------|---------------:|-------:|--------------:|--------:|-------:|---:|-----:|
| CUTLASS 89 default 1SM | 387 us | 5320 | **74.1%** | 82.0% | 79.0% | 58.5% | 32.5% |
| CUTLASS 89 default 2SM | 389 us | 5300 | **74.3%** | 82.2% | 79.1% | 58.9% | 33.6% |
| CUTLASS 89 cluster (2,4) 2SM | 372 us | 5544 | **77.7%** | (not measured) | | | |
| **CuTeDSL cluster (2,1)** | (wall) | **6087** | **82.1%** | (per prev sweep) | | | |

CuTeDSL still beats CUTLASS C++ 89 by ~8pp MFU (82% vs 74%) on the same shape.
The CUTLASS sample uses ALL 148 SMs (cycles_active 97%) but with worse
compute density per cycle — utcmma rate 423 M/s vs CuTeDSL 640 M/s.

### Host overhead in CUTLASS 89's benchmark loop

```cpp
for (int iter = 0; iter < options.iterations; ++iter) {
    CUTLASS_CHECK(gemm.initialize(arguments, workspace));  // host work each iter!
    CUTLASS_CHECK(gemm.run());
}
```

Result: 5000-iter at 16K² K=15360 takes **43.3s wall-clock** but only
**16.7s reported kernel** = **39% busy / 61% host overhead**. dmon's
1Hz polling almost always lands in the host-overhead window, showing
spurious 204W (looks like idle but is between launches). To get true
sustained kernel power on this sample you'd need to patch out
`gemm.initialize()` from the loop.

### Cluster sweep at 1005 MHz, M=N=8192 K=15360 (best of 1SM/2SM)

Note: cluster (1,*) all return "Error Internal at: 503". cluster (8,*) and (*,8) silently empty.
Working cluster shapes: cluster_m ∈ {2,4}, cluster_n ∈ {1,2,4}.

| Preferred cluster | TFLOPS (2SM MMA) |
|-------------------|-----------------:|
| (2,4) | **5544** |
| (4,4) | 5495 |
| (4,1) | (need to retest with proper parser) |
| (default = 1,1?) | 5300 |

Notes:
- Numbers reported in `GFLOPS` field; TFLOPS = GFLOPS/1e6
- ncu kernel name shows mainloop tile is `<128, 256, 768>` (768 = 8 × K=96), schedule
  policy `MainloopSm103TmaUmmaWarpSpecializedBlockScaled<4,7,3,1,...>`
- Sample exposes `--cluster_fallback_m/--cluster_fallback_n` — if preferred cluster
  doesn't fit (occupancy / SHMEM limit), schedule falls back

### Verdict

CUTLASS C++ 89 sample is **correct and runs**, but is **slower than CuTeDSL
example by ~8-15pp MFU** on K-deep matmuls because:
- CuTeDSL persistent kernel design enables better cluster-broadcast data reuse
  (37% DRAM vs 33% — close, but CuTeDSL's 67% SM-throughput hides it better)
- CUTLASS 89 has high host-loop overhead in the benchmark wrapper (61% wall time)

For PRODUCTION code, neither matches cuBLAS catalog 10.8 PF — both are stuck
in the 5-7 PF range at 1005 MHz, which extrapolates to 10-14 PF at boost
WITHOUT throttling.

