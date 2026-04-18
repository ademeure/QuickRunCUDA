# B300 Per-Precision Power/Perf Table (D4)

Comprehensive synthesis of measured peak compute, achieved BW, sustained power,
and TFLOPS/W for every major B300 precision and operation type. All numbers
measured at 2032 MHz boost (default) on B300 SXM6 sm_103a, no clock lock.

## Compute peaks (mma.sync direct + cuBLAS)

| Precision | Path | Measured TFLOPS | NVIDIA Spec | % Spec | Power | TFLOPS/W |
|-----------|------|---------------:|------------:|-------:|------:|---------:|
| FP32 FFMA  | scalar 8-chain     | **70.6**   |    76 |   93%  | 435 W | **0.16** |
| FP64 DFMA  | scalar 4-chain     | **1.0**    |   1.2 |   84%  | (~250 W est) | 0.004 |
| BF16 mma.sync | m16n8k16 4-chain | **2309**   |  2500 |   92%  | 405 W | **5.7** |
| BF16 cuBLAS  | algoId=66 (8K³)   | **2237**   |  2500 |   89%  | (similar) | 5.5 |
| BF16 mma.sync (S1 peak) | m16n8k16 16 warps/SM | **2262** | 2500 | 90.5% | 405 W | **5.6** |
| FP8 cuBLAS  | tcgen05 (prior)    | **~4500**  |  5000 |   90%  | 940 W | **4.8** |
| FP4 cuBLAS  | tcgen05 K=96 (prior) | **10800** | 15000 |  72%  | 940 W | **11.5** |

Sources: S1 (mma.sync), B3 (cuBLAS sweep), prior commits in NINJA_ACHIEVEMENTS.md
+ b300_clean/04_fp32_peak.md, 06_tensor_cores.md.

## Memory bandwidth

| Operation | Measured BW | Spec | % Spec | Power |
|-----------|------------:|-----:|-------:|------:|
| HBM read  (grid-stride int4) | **7.31 TB/s** | 7.67 | 95.3% | 720 W |
| HBM write (cudaMemset DMA)   | **7.27 TB/s** | 7.67 | 94.8% | (~600 W est) |
| L2 read   (64 MB WS, warm)   | **26.7 TB/s** | 38   | 70%   | (~400 W est) |
| SMEM LDS  (LDS.128, 4 SMSPs) | **38.0 TB/s** | 38.5 | 99%   | (low) |
| SMEM STS  (STS.32, 4 SMSPs)  | **37.5 TB/s** | 38.5 | 97%   | (low) |
| NVLink P2P (single-dir, 1 GB) | **773 GB/s** | 783  | 99%   | — |
| NVLink P2P (bidirectional)    | **1509 GB/s** | 1566 | 96%   | — |

## Power decomposition (sustained, NVML 200ms sample)

Idle baseline: 200 W

| Workload | Power | Δ from idle |
|----------|------:|------------:|
| Pure FFMA (8 chains)         | 435 W | +235 W (compute pipe) |
| Pure HMMA legacy (16 warps/SM) | 405 W | +205 W (tensor) |
| HMMA + FFMA same warp        | 424 W | +224 W (compute + ALU) |
| HMMA + LDG (LDG stalls!)     | 305 W | +105 W |
| Kitchen sink (4 roles)       | 280 W | +80 W |
| **Pure HBM read**            | **720 W** | **+520 W (memory dominates!)** |
| HBM + HMMA streams           | 510 W | +310 W (SM contention) |
| **cuBLAS via cudaGraph**     | **940 W** | **+740 W = TGP** |
| TDP cap                      | 1100 W | (not reachable via mma.sync) |

KEY INSIGHT: Memory access is the BIGGEST power source (520 W), not compute.
The path to 940W TGP requires async TMA + tcgen05 to overlap memory + compute
without stalling each other.

## Latency table

| Operation | Measured | Notes |
|-----------|---------:|-------|
| L1 dependent-load chain    | 8 cy / 4 ns | per 02_shmem.md |
| SMEM dependent-load chain  | 28 cy / 14 ns | local; DSMEM = 224 cy |
| L2 read (≤32 MB WS)        | 308 cy / 152 ns | constant for L2-resident |
| HBM read (≥256 MB)         | 537 cy / 264 ns | (B2 chase) |
| Cross-GPU atomic (1 ns clock64-side, ~60 ns wallclock) | — | issue ≠ round-trip |
| __syncthreads (1024 thr) | 49 ns | (vs 12 ns for 32 thr) |
| Single kernel launch (CPU) | ~5 us | vs 0.5 us via cudaGraph |
| 2-GPU all-reduce 1 GB     | 1.97 ms | 67% NVLink SoL |

## Practical recipe per workload type

### LLM Inference Decode (M=1)
- Compute: 5-6 TFLOPS achieved (HBM-bound, weight load dominates)
- Power: ~250-300 W (mostly memory)
- Speedup from TP: minimal (HBM-bound stays HBM-bound on each GPU)

### LLM Inference Prefill (M=8K-32K)
- Compute: 2.2 PFLOPS via cuBLAS BF16 / 4.5 PF FP8
- Power: 940 W via cuBLAS cudaGraph
- TP-2 speedup: 1.69× at M=16K (85% efficient)

### Streaming workloads (RMS-norm, GeLU, softmax)
- Achievable: 89% HBM BW (6.54 TB/s) for fused norm+bias (commit `e9be3e3`)
- Pitfall: register spill at default occupancy → 2.8× slowdown
- Recipe: __launch_bounds__(thr, 2), uint4 vec, stage row in regs

### Compute-bound microkernels
- FFMA: 93% of 76 TF spec (70.6 TF measured)
- DFMA: 84% of 1.2 TF spec (B300 has minimal FP64)
- BF16 mma.sync: 90-92% of 2.5 PF spec
- FP8 (via cuBLAS): 90% of 5 PF spec

### Multi-GPU workloads
- NVLink P2P: 773 GB/s single-dir (99% of 783 spec)
- All-reduce 1 GB: 67% efficient with custom ring; NCCL likely 85-90%
- Atomic broadcast: 16.6 Gatom/s (= 132 GB/s — use ring all-reduce instead)

---

Generated 2026-04-18 in /loop session, from 14 resolved CURIOSITY_LIST_V2 items.
Investigations in `investigations/ninja_*.cu`. See CURIOSITY_LIST_V2.md for
full per-item commit hashes.
