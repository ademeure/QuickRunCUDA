# Ninja Mode Achievements — Speed-of-Light Hunt

**Goal**: push every B300 kernel as close to its hardware ceiling as possible.

Every entry below is rigor-verified (3 methods agree). All numbers in TB/s
unless noted. % is of HBM3E 7672 GB/s spec.

## Storage / bandwidth ladder — pushed to SoL

| Kernel | Baseline | NINJA | Δ pp | Notes |
|---|---:|---:|---:|---|
| HBM write | 95.2% (7.30) | **98.71%** (7.57) | **+3.5** | 1 v8 store/warp; beats cudaMemset by 5% (commit `e75c7e1`) |
| HBM read | 95.0% (7.30) | 95.29% (7.31) | +0.3 | 2 KB/warp; protocol-limited (`9f1c990`) |
| D2D copy | 85.7% (cudaMemcpyAsync) | **90.34%** (6.93) | **+4.7** | 1 KB/warp; src/dst on diff stacks (`4958d6b`) |
| BF16 axpy | 84.0% (6.45) | **91.54%** (7.02) | **+7.5** | 131K small blocks (`866e951`) |
| BF16 absmax | 87.8% (6.74) | 90.21% (6.92) | +2.4 | 18944 blocks block-reduce (`1b8ca7e`) |
| BF16 hist 256-bin | 85.6% (6.57) | 87.4% (6.71) | +1.8 | 740 blocks tuned (`866e951`) |
| BF16 softmax | 70.0% (5.10) | **82.3%** (6.01) | **+12.3** | register-keep-alive across 3 passes; **1.78× speedup** (`57a2e6f`) |

## Methodology insights discovered

### The Per-Warp Burst Principle
Per-warp work size dominates HBM efficiency:
| Per-warp burst | Write peak | Read peak |
|---|---:|---:|
| 1 KB | 98.7% | 93.6% |
| 2 KB | 98.7% | **95.3%** ← read peak |
| 4 KB | 98.6% | 92.4% |
| 32 KB | 96.3% | 83.4% |
| 128 KB | 90.9% | 79.7% |

**Smaller bursts → more warps in flight → better DRAM channel pipelining.**
The "32 iters per warp" recipe leaves 2-3 percentage points on the table
due to per-warp tail effects.

### Register-Keep-Alive for Multi-Pass Kernels
Multi-pass kernels (max-finding then sum-of-exp then write) traditionally
re-read the input through L1 cache. Keeping the input in REGISTERS
across passes eliminates L1 traffic and unlocks 1.5-1.8× speedups.

Pattern (softmax v4):
```cuda
// Pass 1: load uint4 v0, v1 from x ONCE
uint4 v0 = *(uint4*)&x[off]; uint4 v1 = *(uint4*)&x[off + 8];
// Compute max, sum, normalize ALL using v0, v1 in registers
```

### Asymmetric HBM3E R+W Duplex
- True full-duplex: 14.88 TB/s (R + W spec)
- Single-kernel mixed R+W: 6.68 TB/s = 87% spec (capped by turnaround)
- D2D copy (separate src/dst): 6.93 TB/s = 90.3% spec (partial duplex)
- 2-stream R+W: R preserved (7.06), W halved (3.70) — controller prioritizes reads
- **Practical R+W ceiling: ~7 TB/s** for any concurrent mix on B300

### Spatial Separation Unlocks Partial Duplex
When src and dst are on DIFFERENT HBM stacks (different physical
allocations on 12 stacks / 32 FBPA partitions), the controller can
pipeline R from one set of channels and W to another. This explains
copy at 6.93 vs in-place R+W at 6.68.

### Block-Count Sweet Spots
Single optimal block count exists per kernel; "more blocks" is NOT
universally better:

| Kernel | Optimal blocks |
|---|---:|
| HBM write | 524K (1 store/warp) |
| HBM read | 256K (2 KB/warp) |
| D2D copy | 524K |
| Axpy | 131K |
| Absmax | 18944 |
| Histogram | 740 |
| Softmax | 256K (1 row/block) |

Pattern: streaming kernels want MANY small blocks; reduction kernels
want FEWER large blocks (atomic contention).

## Null results worth knowing

| Attempt | Result | Why |
|---|---|---|
| TMA bulk store | 81.6% peak (vs 98.7% v8 inline) | TMA commit/wait overhead |
| FFMA + IADD3 dual-issue | Regression | Compete for FMA pipe |
| 32-way SMEM bin privatization | 78% (vs 87% original) | Reduce overhead |
| Online (1-pass) softmax | SLOWER than 3-pass | More expf in pass 1 |
| Histogram global atomics (no SMEM) | 0.03% peak | 9000 atomic ops in flight serialize |
| Cluster softmax | Crashes (DSMEM HW issue) | per 04_dsmem_overhead.md flag |

## Unbeatable ceilings (true SoL)

| Operation | SoL | Ratio achieved | Source of cap |
|---|---:|---:|---|
| HBM write | 7672 GB/s | 98.7% | HBM3E refresh + command bus |
| HBM read | 7672 GB/s | 95.3% | HBM3E read roundtrip protocol |
| HBM concurrent R+W | 6.68 TB/s | 87% (in-place) | direction turnaround |
| FP32 FFMA | 76.96 TFLOPS @ 2032 boost | 85.5% sustained at 1920 MHz | warp pipeline bubbles |
| SHMEM read | 38.5 TB/s | 99.8% | already at SoL |
| Tensor BF16 mma.sync | 569 TFLOPS | 100% (SoL hit) | already at SoL |

## How to reproduce

All ninja kernels are in `investigations/ninja_*.cu`. To re-verify
any number:
```bash
nvcc -arch=sm_103a -O3 investigations/ninja_<kernel>.cu -o /tmp/n
./utils/rigor_run.sh /tmp/n
```

The rigor harness will print wall-clock + ncu + SASS in one shot.
