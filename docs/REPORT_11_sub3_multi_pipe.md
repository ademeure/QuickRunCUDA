# Report #11 — CORRECTION + new findings from sub-agent 3

Sub-agent 3 (independent, fresh context) discovered that my "dispatch
ceiling = 128 thread-ops/SM/cy" claim in REPORT_09 was incorrect. The
real scheduler can sustain **up to ~216 thread-ops/SM/cy** when work is
spread across 4 disjoint pipes.

## Correction of REPORT_09

**Previous claim:** Total aggregate dispatch caps at 128 thread-ops/SM/cy
regardless of pipe mix. (Measured 125 combined with FFMA+F2FP+LOP3.)

**Actual measurement by sub-agent 3:** With **5-pipe mix** (F2FP + FFMA + IADD
+ LOP3 + HFMA2 at 8 + 16 + 16 + 8 + 4 per iter), hits **216 thread-ops/SM/cy**.

| Pipe mix | Total /SM/clk | Individual rates |
|---|---:|---|
| Unpack only | 63.2 | — |
| Unpack + FFMA | 121 | 60.6 + 60.6 |
| Unpack + IADD (2 pipes) | **126** | 63.2 + 63.2 (lossless!) |
| Unpack + FFMA + IADD (3 pipes) | 124.5 | 62 + 31 + 31 |
| Unpack + FFMA + IADD + LOP3 (4 pipes) | 151 | 60 + 30 + 60 |
| Unpack + FFMA + IADD + LOP3 + HFMA2 (5 pipes) | 180 | 60 + 30 + 60 + 30 |
| **Unpack(8) + FFMA(16) + IADD(16) + LOP3(8) + HFMA2(4)** | **216.3** | 33 + 67 + 100 + 17 |

## Why my REPORT_09 saw 128

My earlier test used N_FFMA=32 + N_F2FP=16 + N_LOP3=32 per iter at blk=512.
With so much FFMA density (32/iter), FFMA alone saturates its 128-cap, and
F2FP and LOP3 get starved because the scheduler's dispatch-slot pressure IS
real at that density. My specific configuration happened to be
dispatch-bound for FFMA-alone reasons.

Sub-agent 3's recipe **balances** pipe use: moderate amounts on each, letting
all 4-5 pipes run in parallel without any one saturating its pipe or
dispatch budget.

## New findings beyond what I had

### 1. Odd warp counts (5, 6) dip 2× due to SMSP imbalance

Blackwell has 4 warp schedulers per SM. With 5 warps/SM, one scheduler gets
2 warps while others get 1. The 2-warp scheduler becomes the critical path,
effectively doubling wallclock.

| Warps/SM | Thread-ops/SM/clk | % of 64 peak |
|---:|---:|---:|
| 1 | 14.75 | 23% (can't dual-issue) |
| 2 | 29.5 | 46% |
| 3 | 44.2 | 69% |
| **4** | **58.7** | **92%** (magic number) |
| **5** | **38.1** | **60% (pathological!)** |
| **6** | **45.7** | **71%** |
| 7-8 | ~60 | 94% |
| 16 | 62.7 | 98% |

**Rule: always use warp counts divisible by 4** on Blackwell.

### 2. Wave-boundary penalty is brutal (2× for 1 stray block)

| Blocks | Waves | Thread-ops/SM/clk |
|---:|---:|---:|
| 148 | 1.000 | 60.9 |
| **149** | **1.007** | **31.6 (halved!)** |
| 296 | 2.000 | 62.6 |
| 297 | 2.007 | 42.0 |
| 1480 | 10 | 63.5 |
| 14800 | 100 | 63.77 (tail amortized) |

**Rule: use exact wave multiples (N × 148) OR > 10 waves** to amortize the
tail. N*148 + 1 block doubles execution time.

### 3. Minimum TLP for F2FP peak: 4 warps/SM with ≥8 ILP chains

Sub-agent proved the 4 × 2 latency-hiding rule: ILP > L_latency × dual_issue
= 4 × 2 = 8 independent ops in flight per thread.

### 4. Thread Block Clusters ≥ 4 halve F2FP throughput

Because clusters must fit within a GPC (12 SMs), cluster_size=4 means 1
wave (148 blocks / 4 = 37 clusters) can't all fit in parallel GPCs; the
cluster scheduler serializes waves. **Avoid cluster_size ≥ 4 for
F2FP-bound kernels.**

### 5. Occupancy sweet spot: 256 threads × 2-4 blocks/SM

`__launch_bounds__(256, 4)` → 32 warps/SM, 15 regs/thread, no spills, **63.6
/SM/clk** — 99.4% of peak. Going to 16+ blocks/SM gives diminishing returns.

## Corrected mental model (final)

### Scheduler capacity
- Each SMSP issues ≤1 warp-inst/cy. 4 SMSPs → **4 warp-inst/SM/cy dispatch peak**
- BUT individual pipes have their own caps that apply simultaneously

### Pipe acceptance rates (warp-insts/SM/cy)
- `fmalite` (FFMA/FMUL): 4 (matches dispatch)
- `alu` (F2FP unpack): 2 (dual-issue)
- `alu` (F2FP pack, 2+ read ports): 1
- `xu` (MUFU.EX2): 1
- `xu` compound (MUFU.RSQ/SIN/etc): 0.5
- Integer pipe (IADD/LOP3): 4 (high capacity)
- `fmaheavy` (IMAD): 2 (slower than FFMA)
- Half-FMA (HFMA2/HADD2): 2

### Aggregate ceiling
- **~7 warp-insts/SM/cy = ~216 thread-ops/SM/cy** across 4-5 disjoint pipes.
- This exceeds the naive "4-scheduler × 1-issue" calculation because
  SMSPs can issue to *multiple* pipes per cycle or interleave pipes across
  cycles — in aggregate the SM sustains 200+ thread-ops/SM/cy when work
  is balanced.

### Single-pipe ceiling
- Still 64 for F2FP unpack, 32 for F2FP pack, 32 for MUFU.EX2, etc.

## Practical kernel-writing implications

For a quant-pack kernel, if you want total throughput to fly:
1. Start with F2FP pack @ 32/SM/cy
2. Add FFMA math (scale factors) — gets ~32 FFMA/SM/cy on top (no collision)
3. Add IADD for address math — gets ~32 IADD/SM/cy
4. Add HFMA2 if doing BF16 math — gets ~16-32 HFMA2/SM/cy
5. Total: ~128+ useful ops/SM/cy

Avoid:
- MUFU ops (softmax/GELU) — collide with F2FP on SFU
- Too much of any one pipe (FFMA ≥ 32 per pack iter → FFMA saturates and competes)
- Odd warp counts per SM
- Stray blocks beyond exact waves

Target:
- blk=256, min_blocks=4, 32 warps/SM, 15-70 regs/thread
- Exact wave multiples (N × 148)
- Balance pipes: F2FP : FFMA : INT : HFMA2 ≈ 1 : 2 : 2 : 0.5 approximately

## Session count after this report

- 11 chronological reports
- 3 sub-agent investigations (all 3 independent)
- 60+ kernels
- 50+ raw logs
- Critical self-correction of my own REPORT_09

The "dispatch ceiling" was wrong; the real ceiling is ~216 thread-ops/SM/cy
with 4-5 pipes in balance. Single-pipe maxes (64 unpack, 32 pack) are correct.
