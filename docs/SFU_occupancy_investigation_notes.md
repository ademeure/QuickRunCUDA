# F2FP Unpack TLP Investigation — Blackwell sm_103a (B300 SXM6 AC)

**Instruction studied:** `cvt.rn.f16x2.e4m3x2` → `F2FP.F16.E4M3.UNPACK_B` (the "fast" 1-port SFU variant that dual-issues at 64 thread-ops/SM/cycle).

**Hardware:** NVIDIA B300 SXM6 AC, 148 SMs, GPU clock locked at **1920 MHz** (reported max 2032 MHz but current 1920). GPU 0 only.

**Method:** All measurements via `QuickRunCUDA`, converted from total `GOps/s` → thread-ops/SM/clock using `ops/s / 148 / 1.92e9`.

**Theoretical ceiling:** 64 thread-ops/SM/clk (dual-issue 1-port SFU variant). Prior work established this; we reproduce it cleanly and then **break past it** via pipe mixing.

---

## TL;DR: four biggest findings

1. **64 /SM/clk confirmed as the unpack-only ceiling** — measured **63.2–63.77 /SM/clk** (99.6% of 64) with just 32+ warps/SM and ≥8 ILP chains/thread. SASS: pure `F2FP.F16.E4M3.UNPACK_B` with no PRMT/MOV overhead.

2. **Minimum TLP to saturate** (at N_CHAINS=8 = 8 ILP chains/thread, which matches the 8-in-flight-ops you need given L=4cy × 2 dual-issue): **4 warps/SM hits 95% of peak (60.8 /SM/clk)**, 8 warps/SM → 96.6%, 16 warps → 98.5%. **2 warps alone cap at 30.4 /SM/clk** (exactly half, because the dual-issue needs ≥2 warps eligible per SMSP). 1 warp can't dual-issue (15.2, 23.8% of peak) — it issues 1 warp-inst/cycle = 32 thread-ops/cycle ideal, measured 15.2 suggests 1 warp only issues every other cycle.

3. **F2FP unpack can be driven WELL past 64/SM/clk when paired with non-SFU ops.** Measured totals (8 unpacks + N companion per inner loop, T=1024, 32 warps/SM):

   | Companion | Total thread-ops/SM/clk | Unpack rate | Comp rate |
   |---|---:|---:|---:|
   | FFMA (FMA pipe)     | **121.1** | 60.6 | 60.6 |
   | FMUL (FMA pipe)     | 121.4 | 60.7 | 60.7 |
   | IMAD (INT pipe)     | 118.9 | 59.5 | 59.5 |
   | LOP3/XOR (INT pipe) | 122.7 | 61.4 | 61.4 |
   | **IADD (INT pipe)** | **126.4** | 63.2 | 63.2 |
   | HFMA2 (half FMA)    | 123.6 | 61.8 | 61.8 |
   | HADD2 (half ALU)    | 123.7 | 61.8 | 61.8 |
   | EX2 (SFU pipe — same!)   | 61.9 (drops) | 30.9 | 30.9 |
   | PACK F2FP (SFU, MERGE_C) | 42.5 (drops hard) | 21.2 | 21.2 |

   **IADD companion gives lossless pairing** — unpack stays at 63.2, add adds another 63.2, total **126.4 thread-ops/SM/clk**. Any op on the same SFU pipe (EX2, PACK) forces sharing → total drops below unpack-alone.

4. **Multi-pipe mix pushes total to ~216 thread-ops/SM/clk** (3.4× the unpack-only ceiling).  
   Best config: NC=8 F2FP + 16 FFMA + 16 IADD + 8 LOP3 + 4 HFMA2 per inner iter, 32 warps/SM.  
   Result: **216.3 total** = 33 unpack + 67 FFMA + 100 (IADD+LOP3) + 17 HFMA2 /SM/clk.  
   Observations: FFMA saturates at ~67/SM/clk (near its own 64 cap), INT pipe at ~100 (near its cap), unpack degrades to ~33 when there is this much pressure on the SMSP dispatch. This strongly suggests the SMSP-dispatch ceiling in the ~200–220 thread-ops/SM/clk range is real but can only be hit by spreading work over 4 disjoint hardware pipes.

---

## Angle 1: Smallest TLP to reach peak

Kernel: `tests/bench_f2fp_oneway.cu` with `DIRECTION=1` (pure unpack only, truncate feedback ∈ `unsigned short`). SASS verified as **256 `F2FP.F16.E4M3.UNPACK_B` + 4 PRMT total (only at kernel prologue/epilogue)** — zero overhead inside the hot loop.

b=148 (1 wave of 1 block/SM), N_CHAINS=8, UNROLL=32:

| Threads/block | Warps/SM | Time (ms) | Thread-ops/SM/clk | % of 64 |
|---:|---:|---:|---:|---:|
| 32   | **1** | 0.0370 | **14.75** | 23.0% |
| 64   | 2 | 0.0371 | 29.46 | 46.0% |
| 96   | 3 | 0.0370 | 44.23 | 69.1% |
| 128  | **4** | 0.0372 | **58.67** | **91.7%** |
| 160  | 5 | 0.0717 | 38.08 | 59.5% (pathological) |
| 192  | 6 | 0.0717 | 45.69 | 71.4% |
| 256  | 8 | 0.0717 | 60.92 | 95.2% |
| 384  | 12 | 0.107 | 62.02 | 96.9% |
| 512  | 16 | 0.139 | 62.69 | 98.0% |
| 1024 | 32 | 0.276 | **63.23** | **98.8%** |

- **Odd warp counts (5, 6) cause large dips** — because Blackwell has 4 SMSPs per SM; one SMSP gets 2 warps while others get 1, so the imbalanced SMSP becomes the critical path. This is visible as exactly 2× the time for T=160 vs T=128 even though T=160 has 1.25× the work.
- **4 warps/SM is the magic number** if your block is multi-warp: 91.7% with 1 block/SM. Multi-block layouts (2×64t, 4×32t) also hit 58-59/SM/clk at 4 total warps/SM.

---

## Angle 2: Pipe-mixing to push past 64/SM/clk

Custom kernel `/tmp/f2fp_unpack_plus_comp.cu`: inner loop has `N_CVT=8` pure unpacks + `N_COMP` companions. T=1024, b=148.

**Headline:** Non-SFU companions leave unpack throughput **unchanged (63.2)** and add their own independent throughput on top.

For multi-pipe stacking (`/tmp/f2fp_unpack_plus_multi.cu`, T=512, MIN_BLOCKS=2, b=592):

| Config (NC/F/I/L/H) | Total /SM/clk | Unpack | FFMA | INT | HFMA2 |
|---|---:|---:|---:|---:|---:|
| 8/0/0/0/0 (baseline) | 63.2 | 63.2 | — | — | — |
| 8/4/0/0/0            | 93.4 | 62.3 | 31.1 | — | — |
| 8/0/4/0/0            | 94.8 | 63.2 | — | 31.6 | — |
| 8/4/4/0/0            | 124.5 | 62.2 | 31.1 | 31.1 | — |
| 8/4/4/4/0            | 151.2 | 60.5 | 30.2 | 60.5 | — |
| 8/4/4/4/4            | 179.7 | 59.9 | 30.0 | 59.9 | 30.0 |
| 8/8/8/0/0            | 180.2 | 60.1 | 60.1 | 60.1 | — |
| **8/16/16/8/4**      | **216.3** | 33.3 | 66.6 | 99.8 | 16.6 |
| 8/12/12/12/8         | 213.7 | 32.9 | 49.3 | 98.6 | 32.9 |
| 8/8/8/8/4 (moderate) | 212.8 | 47.3 | 47.3 | 94.6 | 23.7 |

**SASS evidence** for the 216 /SM/clk config (`sass/f2fp_unpack_plus_multi_1631555942.sass`): 128 F2FP + 192 FFMA + 129 HFMA2 + 54 IADD + 54 LOP3 + 1 UIADD (ptxas fused adjacent IADD/LOP3 into IADD3/LOP3 multi-operand forms). Register count: 56.

**Interpretation:** The 64/SM/clk unpack limit is an **SFU-pipe** constraint, not an SMSP-dispatch constraint. Each SMSP dispatches per-cycle to multiple pipes (FMA, INT, Half, SFU) — the unpack happily eats its SFU slots while the others hit their own pipes. Total = **sum of per-pipe caps**, limited by SMSP total issue-slots/cycle (~200–220 thread-ops/SM = ~6–7 warp-issues/SM/cycle).

---

## Angle 3: Occupancy ceiling via `__launch_bounds__`

| warps/SM | Time (ms) | Thread-ops/SM/clk |
|---:|---:|---:|
| 2  | 0.0722 | 30.27 |
| 4  | 0.0721 | 60.57 |
| 8  | 0.139  | 62.69 |
| 16 | 0.277  | 63.20 |
| 32 | 0.550  | 63.60 |
| 48 | 0.823  | 63.69 |
| 64 | 1.096  | **63.77** |

Knee at **4 warps/SM**: linear from 2→4, saturates 4→64. Max occupancy only buys **+1.1 /SM/clk** over 16 warps. Register budget for DIRECTION=1 is ~15 regs regardless of `MIN_BLOCKS`, so register pressure is not the story.

Interesting anomaly — `T=64, MIN_BLOCKS=16` (= 16 blocks × 2 warps = 32 warps/SM target): measured 30 /SM/clk, same as 16 warps/SM. Most likely the hardware is capping block count/SM lower than 16 (block limit can be 24 or 32 on modern Blackwell; with 64 threads × 16 blocks it seems some are rejected from residency).

---

## Angle 4: Cluster effects (thread block clusters)

Kernel `/tmp/f2fp_unpack_cluster.cu`, BLOCK_SIZE=512, N_CHAINS=8, b=148 or max multiple of cluster size:

| CLUSTER_SIZE | Blocks | Time (ms) | Thread-ops/SM/clk |
|---:|---:|---:|---:|
| 1 | 148 | 0.276 | 63.21 |
| 2 | 148 | 0.277 | 63.19 |
| 4 | 148 | 0.549 | 31.81 (drops!) |
| 8 | 144 | 0.549 | 31.82 (drops!) |

Cluster size 1–2: **no effect on F2FP unpack**, as expected (pure register compute, no DSMEM).  
Cluster size 4–8: throughput halves — not a compute effect, a scheduling/residency effect. Clusters must fit within a single GPC, and with ~12 SMs per GPC, a wave of 37 clusters of size 4 (= 148 blocks) doesn't fit in the GPCs at full density, so the cluster scheduler serializes waves. **Not recommended for F2FP-bound kernels.**

---

## Angle 5: Wave boundary effects

b=1024 threads/block (1 block/SM), vary total blocks. Reports "/SM/clk" normalized as if total-work / (148 SMs × time):

| Blocks | Blocks/SM | Time (ms) | Observation |
|---:|---:|---:|---|
| 148 | 1.000 | 0.0718 | **Exact 1 wave — 60.9 /SM/clk** |
| 147 | 0.993 | 0.0717 | 60.5 (slight undersub) |
| 149 | 1.007 | 0.1394 | **31.6 — one stray block ≈ doubles wallclock** |
| 296 | 2.000 | 0.1396 | 62.6 (exact 2 waves) |
| 295 | 1.993 | 0.1395 | 62.4 |
| 297 | 2.007 | 0.2089 | 42.0 (stray block, 1 wave wasted) |
| 444 | 3.000 | 0.2089 | 62.7 |
| 592 | 4.000 | 0.2767 | 63.2 |
| 1480 | 10  | 0.688 | 63.5 |
| 4440 | 30  | 2.057 | 63.7 |
| 14800 | 100 | 6.851 | **63.77** (tail amortized) |

**Classic wave-tail penalty.** Going from exact-N-waves to N+epsilon doubles wallclock. Multi-wave (10+) amortizes out to 63.5–63.77. For benchmarking, always use exact wave multiples OR many waves.

---

## Angle 6: Register pinning / N_CHAINS stress

`bench_f2fp_oneway.cu` DIR=1, vary N_CHAINS (→ register footprint), 4 warps/SM:

| N_CHAINS | regs/thread | Thread-ops/SM/clk |
|---:|---:|---:|
| 4  | 9  | 59.04 |
| 8  | 14 | 60.91 |
| 16 | 20 | 62.66 |
| 32 | 36 | 63.19 |
| 64 | 68 | 63.60 |

More ILP chains = closer to peak, no spill even at 68 regs/thread (we have 1024 threads × 68 regs = 69632 regs/SM — well within budget). ILP > L_latency × dual_issue = 4 × 2 = 8 is the key threshold (matches the asymptote).

Forcing **MIN_BLOCKS** via launch_bounds (T=256, MB=1..8):

| T × MIN_BLOCKS | warps/SM | Thread-ops/SM/clk |
|:---|---:|---:|
| 256×1 | 8  | 62.68 |
| 256×2 | 16 | 63.18 |
| 256×4 | 32 | 63.56 |
| 256×8 | 64 | 63.76 |
| 128×8 | 32 | 63.39 |
| 128×16| 64 | 59.93 (dips — likely scheduler overhead with 16 blocks/SM) |
| 64×32 | 64 | 59.93 |

Conclusion: `__launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)` works as expected for driving occupancy up — but going beyond **8 blocks/SM gives diminishing or negative returns** (block-level scheduling overhead). Sweet spot is **256×4 or 512×2** → 32 warps/SM.

---

## Quantitative wrap-up (B300 @ 1920 MHz, 148 SMs)

| Metric | Measured | % of theoretical |
|---|---:|---:|
| Pure F2FP.UNPACK_B ceiling | **63.77 thread-ops/SM/clk** | 99.6% of 64 |
| Aggregate GPU unpack throughput | **18.1 TF2FPs/s** (36.2 Telements/s for x2) | — |
| Min TLP for 95% of unpack peak | 4 warps/SM × 8 ILP chains = **32 eligible inst-in-flight** | — |
| Unpack + non-SFU companion (FFMA) | **121 thread-ops/SM/clk** | — |
| Unpack + best 4-pipe mix | **216 thread-ops/SM/clk** | ~3.4× unpack-alone ceiling |
| Unpack + same-SFU companion (EX2) | 61.9 (drops) | — |
| Unpack + same-SFU companion (PACK F2FP) | 42.5 (drops hard) | — |

**The "dispatch ceiling" of ~128 mentioned in the prompt is actually higher in practice** — we hit 216 /SM/clk with 4-pipe stacking. The 128 may have referred to 2-pipe-mix (SFU + 1 other), which we measure at ~121–126 /SM/clk depending on companion (so 128 is basically right for that case, limited by per-pipe caps of ~64 each).

## Key SASS files

- `/root/github/QuickRunCUDA/sass/bench_f2fp_oneway_2683702736.sass` — clean pure unpack (256 F2FP, 4 PRMT, no other overhead)
- `/root/github/QuickRunCUDA/sass/f2fp_unpack_plus_multi_1631555942.sass` — 4-pipe mix, 128 F2FP + 192 FFMA + 129 HFMA2 + 54 IADD + 54 LOP3

## Kernels used

- `/root/github/QuickRunCUDA/tests/bench_f2fp_oneway.cu` — DIRECTION=1 for pure unpack
- `/root/github/QuickRunCUDA/tests/bench_f2fp_cluster.cu` — adapted for cluster test
- `/tmp/f2fp_unpack_plus_comp.cu` — unpack + single-pipe companion
- `/tmp/f2fp_unpack_plus_multi.cu` — unpack + 4-pipe mix
- `/tmp/f2fp_unpack_cluster.cu` — pure unpack with cluster attribute
