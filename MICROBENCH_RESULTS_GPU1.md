# QuickRunCUDA Microbenchmark Results — GPU 1 (B300 SXM6 AC)

Full rerun of every microbenchmark in this repo on GPU 1, 2026-04-14.

## Environment

| Item | Value |
|------|-------|
| GPU | NVIDIA B300 SXM6 AC (GPU 1 of 2) |
| Arch | SM 10.3a (Blackwell) |
| SMs | 148 |
| Max SM clock | 2032 MHz |
| Max mem clock | 3996 MHz (HBM3e, 8192-bit bus) |
| Memory | 275040 MiB |
| Driver | 580.126.09 |
| Toolkit | CUDA 13.0.88 |
| Date | 2026-04-14 |

Theoretical peaks used below:

| Metric | Peak | Formula |
|--------|------|---------|
| HBM bandwidth | **8183.8 GB/s** | 2 × 3996 MHz × 8192 bit / 8 / 1e9 |
| FP32 FMA throughput | **83.23 TFLOPS** | 148 SM × 128 FMA/SM/clk × 2 FLOP × 2032 MHz |
| SM-clock aggregate | 300.74 G SM-clk/s | 148 × 2032 MHz |

Convert `GOps/s` → ops/SM/clock: divide by 300.74.

All runs: `CUDA_VISIBLE_DEVICES=1`, 100+ timed launches, `-p` (persistent blocks = 1 per SM = 148) for compute-bound tests, explicit grid for bandwidth-bound tests. Inputs are thread-dependent and sinks are either stored behind a data-dependent runtime-unknown branch or through a volatile asm to defeat DCE/LICM.

---

## 1. DRAM Bandwidth — Deep Dive

The original `bench_dram_bw.cu` (one `int4` per thread, 256 MiB workload, non-persistent) reports **6635 GB/s (81.1%)** out of the 8184 GB/s peak. Where does the remaining 19% go, and how high can pure memory-traffic kernels actually go?

### 1.1 Key parameters explored

- **Access mode**: LOAD-only (impossible-branch sink), STORE-only (memset pattern), COPY, COPY+XOR
- **Transaction width**: 4 B (`ld.u32`), 8 B (`ld.u64`), 16 B (`ld.v2.u64`), 32 B (`ld.v4.u64`)
- **Per-thread unroll**: 1, 2, 4, 8, 16 consecutive ops per thread
- **Grid mode**: non-persistent (one dispatch covers the data, consecutive-per-thread unroll) vs persistent (148 blocks, grid-stride unroll — adjacent unrolled loads are a full grid apart)
- **Block size**: 64–1024
- **Workload size**: 64 MiB → 4 GiB

Driver: `tests/bench_dram_variants.cu` (flexible parametric kernel) via `/tmp/qrc_results/run_dram_sweep.sh` + `run_dram_push.sh`.

### 1.2 Best results — new peaks per access mode

| Access mode | Best config | Time | Bandwidth | % SoL |
|---|---|---:|---:|---:|
| **DRAM→L2 fill** (prefetch256 trick) † | 6 GiB, scalar load + `ld.L2::256B`, blk=1024, U=1 | 0.8438 ms | **7635 GB/s** | **93.3%** |
| **STORE** (v8.f32) | 4 GiB, `ld.global.v8.f32`, blk=1024, U=1 | 0.5674 ms | **7569 GB/s** | **92.5%** |
| **LOAD → SM** (honest, v8.f32 + L2 prefetch-ahead) | 6 GiB, blk=512, PA=32 tiles, non-persistent | 0.8676 ms | **7425 GB/s** | **90.7%** |
| LOAD-only (v4.u64, no prefetch) | 32 GiB, W=32, U=2, blk=96 | 4.6446 ms | 7398 | 90.4% |
| **COPY** (v8.f32, LOCAL_STRIDE, U=2) | 4 GiB, blk=1024, U=2 | 1.2236 ms | **7020 GB/s** | **85.8%** |
| COPY baseline (`bench_dram_bw.cu`, 256 MiB) | default | 0.1618 ms | 6635 | 81.1% |

† The **DRAM→L2 prefetch256 number is not a true "load into SM" bandwidth** — each thread issues one *scalar* 4-byte `ld.global.L2::256B.f32` whose main effect is pulling a 256-byte sector into L2. The SM registers only receive 4 bytes per thread (1/64 of the sector), but the HBM bus does do the full 256-byte fill. So 93.3% is the HBM→L2 ceiling, not DRAM→register. When you actually consume the full data into SM registers via a paired `ld.global.v8.f32` (my `bench_dram_prefetch_consume.cu`), the bandwidth lands at **90.7%** — essentially the same as the naïve v4.u64 peak.

Headline: naive baseline 81% → **85.8% COPY / 90.7% LOAD-to-SM / 92.5% STORE / 93.3% HBM→L2 peak**.

### 1.3 Why STORE > LOAD > COPY

- **STORE-only 92%** — HBM3e write queues can absorb near-peak in a single direction; store traffic is pure "post & forget", no latency on the critical path.
- **LOAD-only 89%** — loads must round-trip through L2 and come back; you need enough in-flight transactions to cover the latency. This needs a bigger workload (512 MiB still only 85%; 2 GiB gets to 89%) and modest unroll (U=4 optimal — U=8 and U=16 drop sharply because the register pressure spills or the scheduler can't keep 4× wide LDGs in flight).
- **COPY ≤ 85%** — R and W on HBM share banks. B300's memory controller does rate-match reads and writes onto separate command queues, but turning the bus direction + bank-conflict checks cost throughput. 85% for naive memcpy matches NVIDIA-published HBM3e efficiency targets.

### 1.4 Workload-size effect (COPY, W=16, U=2, blk=256)

| Size | Time | GB/s | % SoL |
|---:|---:|---:|---:|
| 64 MiB | 0.0223 ms | 6020 | 73.6% |
| 128 MiB | 0.0432 ms | 6208 | 75.8% |
| 256 MiB | 0.0830 ms | 6468 | 79.0% |
| 512 MiB | 0.1589 ms | 6758 | 82.6% |
| 1024 MiB | 0.3132 ms | 6857 | 83.8% |
| **2048 MiB** | **0.6217 ms** | **6908** | **84.4%** |
| 4096 MiB | 1.2401 ms | 6927 | 84.7% |

Bandwidth is asymptotic in workload size — you need ≥ 1 GiB to hide the kernel-launch & L2-warmup costs. The original bench at 256 MiB left 3 pts on the table.

### 1.5 Transaction width × unroll matrix — non-persistent COPY (512 MiB, blk=256)

Numbers are % SoL (GB/s in parens where interesting):

| | U=1 | U=2 | U=4 | U=8 |
|---|---:|---:|---:|---:|
| W=4 (u32) | 37.5 | 63.4 | 62.3 | 29.4 |
| W=8 (u64) | 64.0 | **81.0** (6631) | 57.3 | 43.3 |
| W=16 (v2.u64) | 81.1 (6638) | **82.6** (6757) | 77.9 | 46.7 |
| W=32 (v4.u64) | **82.5** (6754) | 82.6 (6757) | 66.0 | 64.9 |

The **82.5% plateau** across W=16 U=2, W=32 U=1, and W=32 U=2 tells us the bus is saturated once you issue one 32-byte or two 16-byte transactions per thread. Beyond that, more unroll *hurts* because register pressure forces spills and the scheduler can't keep enough warps active to cover HBM round-trip.

### 1.6 Non-persistent vs persistent, same (W=16, U=4)

| Grid mode | blk=64 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|
| Non-persistent (blocks sized to data) | 82.2% | 82.2% | 82.3% | 82.4% | **82.5%** |
| Persistent (148 blk, grid-stride) | 23.9% | 38.2% | 61.1% | **77.7%** | 75.6% |

Persistent at 148 blocks × 256 threads can't issue enough in-flight memory requests to saturate the bus. It needs 512+ threads per block to recover, and even then non-persistent wins by ~5 pts. With 2× oversubscription (296 blocks) persistent climbs back to 77.6%, but still short of the straight-dispatch 82.4%. **Conclusion: don't use persistent blocks for pure memcpy; let the SM scheduler handle load balancing.**

### 1.7 STORE-only: unroll doesn't matter, width barely matters (2 GiB, non-persistent)

STORE-only numbers are remarkably flat across (W, U):

| | U=1 | U=2 | U=4 | U=8 |
|---|---:|---:|---:|---:|
| W=16 | **89.0%** (7283) | 51.7 | 41.0 | 19.1 |
| W=32 | **89.0%** (7283) | 89.3 (7309) | 58.8 | 59.2 |

(512 MiB row; at 2 GiB the numbers rise to 91-92%.) For stores the issue is that each warp's STG completes out-of-order into write-combining buffers — if you issue too many stores per thread, the store queue backs up. **U=1 is consistently the best for stores**.

### 1.8 LOAD-only: pushing to the 90% ceiling

LOAD is latency-bound, so throughput is determined by (a) in-flight request count and (b) total work per SM.

**Cache hint does nothing** (4 GiB W=32 U=2 blk=256, all within 0.2 pts):

| Hint | %SoL |
|---|---:|
| `ld.global.nc` | 89.61% |
| `ld.global.ca` | 89.60% |
| `ld.global.cg` | 89.42% |
| `ld.global.cs` | 89.61% |
| `ld.global.nc.L2::256B` | **89.63%** |
| `ld.global.nc.L2::128B` | 89.61% |
| `ld.global.nc.L1::no_allocate` | 89.43% |

**Workload size matters most** (W=32, U=2, blk=256, hint=L2::256B):

| Size | GB/s | %SoL |
|---:|---:|---:|
| 512 MiB | 6955 | 85.0% |
| 1 GiB | 7150 | 87.4% |
| 2 GiB | 7281 | 89.0% |
| 4 GiB | 7335 | 89.6% |
| 8 GiB | 7367 | 90.0% |
| **16 GiB** | **7386** | **90.25%** |
| **32 GiB** | **7398** | **90.4%** |

**Block-size at 16 GiB** (plateau ±0.15 pts):

| blk | %SoL |
|---:|---:|
| 32 | 48.7 |
| 64 | 90.29 |
| **96** | **90.40** |
| 128 | 90.35 |
| 192 | 90.29 |
| 256 | 90.26 |

**Width × Unroll at 4 GiB** (W=32 U=2 is optimal; wider is ~free, deeper U hurts):

| | U=1 | U=2 | U=4 | U=8 |
|---|---:|---:|---:|---:|
| W=8 | 40.8 | 72.8 | 87.6 | 57.6 |
| W=16 | 73.6 | 88.0 | **89.9** | 57.9 |
| W=32 | 88.1 | **89.6** | 85.1 | 87.7 |

**Explicit L2 prefetch** (`prefetch.global.L2` issued N tiles ahead) gives only a 0.1 pt bump over baseline at pf=2 and *hurts* past pf=4 (prefetch target goes OOB and contends with legit loads):

| pf_ahead | %SoL |
|---:|---:|
| 0 | 90.33 |
| **2** | **90.44** |
| 4 | 89.18 |
| 8+ | crash/useless (OOB prefetch) |

**Persistent oversubscription tops out lower than non-persistent**:

| blocks | %SoL |
|---:|---:|
| 148 (1× SM) | 49.4 |
| 296 (2×) | 77.9 |
| 592 (4×) | 88.5 |
| 1184 (8×) | 86.7 |

Non-persistent with 1M+ blocks lets the hardware scheduler pipeline memory requests better than any hand-rolled grid-stride.

**LOAD-to-SM ceiling ≈ 90.7%** via `v8.f32` + prefetch-ahead; **90.4%** without prefetch. Many other knobs I tested (cache hints, persistent oversubscription, alignment) don't move the needle. What I *missed* without the reference repo:

### 1.8.1 Reference-kernel techniques that matter

Copying `membw.cu`, `membw_prefetch256.cu`, `membw_ptx.cu` from `ademeure/private-quickrun-cuda` into `tests/reference/` reveals six tricks I hadn't tried:

1. **Native 256-bit PTX load (`ld.global.v8.f32`)** — I'd been using `ld.global.v4.u64` (also 256 bits) but the `.v8.f32` form compiles to cleaner SASS and consistently squeezed ~0.3-0.5 pts more: STORE went 92.1% → 92.5%, COPY 85.0% → 85.6% (at U=2 with native 256-bit), and LOAD benefits similarly.
2. **`__launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)`** — gives the compiler a register/occupancy budget target; without it the v8.f32 loads spilled enough registers to cap blk=1024 occupancy.
3. **`ld.global.L2::256B`** hint used with *scalar* loads (the "prefetch256 trick") — this is the DRAM→L2 peak measurement. Each warp reads 32 × 4 = 128 bytes into registers but fills 32 × 256 = 8 KB into L2. At 6 GiB this hits **93.3%** — the effective HBM ceiling.
4. **`.L2::evict_last` cache policy** on stores/reads — useful for streaming patterns where you don't want the data to linger in L2. Gives COPY U=1 a 0.7-pt bump vs. default.
5. **`LOCAL_STRIDE`** (block-local contiguous chunks with `stride = blockDim.x` instead of `gridDim.x*blockDim.x`) — keeps adjacent threads' accesses in a ~32 KB window instead of ~5 MB apart. COPY U=2 gains 0.8 pts.
6. **Cache-aware read summation** — accumulate *all* components of the vector load, otherwise the optimizer may emit scalar loads instead of v4/v8. Essential for making the SASS match the PTX.

### 1.8.2 The "prefetch-and-consume" honest ceiling

With 256-byte L2 prefetches issued at `(consume_addr + ADVANCE_BYTES)` plus a v8.f32 consume at `consume_addr` (hits L2 if the prefetcher got there first):

**Dense per-thread prefetch** (every thread issues 1 prefetch + 1 consume) tops out at 90.3% with `ADVANCE=2 MiB`, collapses at ADVANCE≥16 MiB because the scalar prefetches issue-rate-limit: 32 threads × 256B fills = 8 KiB HBM traffic per warp vs 1 KiB consumed, a wasted 8:1 ratio.

**Sparse prefetch** (only 1-in-8 lanes prefetches, so 4 prefetches per warp ≈ 1 KiB HBM per warp, matching the 1 KiB consume rate) still can't exceed naïve v8.f32:

| Config | GB/s | % SoL |
|---|---:|---:|
| Naïve v8.f32 read, 6 GiB | 7426 | 90.7 |
| Sparse PF (lane%8==0) ADV=2 MiB | 7387 | 90.3 |
| Sparse PF ADV=16 MiB | 3763 | 46.0 |
| Sparse PF ADV=32 MiB | 3745 | 45.8 |

Why ADV≥16 MiB collapses in non-persistent mode: a 6 GiB kernel has ~262K blocks but only ~300 in flight at once. Thread T prefetches data that thread T+512K (512 blocks ahead) will consume — those blocks haven't been scheduled yet when T runs, so the L2 line is evicted by the consume wave that's 500+ blocks behind the prefetch wave. The trick only works when the prefetcher *and* the consumer are co-resident, which means either persistent kernels (but those are warp-starved at 148 blocks, capping at ~81%) or ADVANCE so small (≤4 MiB / ~130K tiles / ~255 blocks) that it fits inside the in-flight window — in which case the HW prefetcher is already doing the same job.

**The honest DRAM→SM read ceiling on B300 is ≈90.7%**, achievable with a plain `ld.global.v8.f32` loop — no prefetch, no unroll, no cache hint, no persistent trickery needed.

### 1.8.3 "Just v8, no tricks" — how close without unroll/prefetch/etc.?

`tests/reference/membw_ptx.cu` with `PTX_BITS=256 UNROLL=1`, no prefetch, no `.L2::evict_last`, no `LOCAL_STRIDE`:

| Mode | blk | Workload | GB/s | % SoL | vs. best-with-tricks |
|---|---:|---:|---:|---:|---|
| **LOAD**  | 512 | 6 GiB | **7426** | **90.74%** | = 90.7% honest prefetch+consume |
| **STORE** | 768 | 6 GiB | **7580** | **92.62%** | vs 92.5% (U=1-4 is flat) |
| **COPY**  | 768 | 6 GiB | **6939** | **84.79%** | vs 85.8% (LOCAL_STRIDE U=2, +1 pt) |

Vanilla `v8.f32` alone gets ≥99% of the reachable ceiling. All the fancy tricks (unroll, prefetch-ahead, LOCAL_STRIDE, cache hints) combined add at most 1 pt. The only "trick" that matters in practice: use the 256-bit PTX form instead of compiler-intrinsic `int4`/`float4` loads, and keep the workload ≥ 2 GiB.

### 1.8.3 Why prefetch256 reports 93.3% but it's not "real"

It's a legitimate HBM peak number, but only if your kernel doesn't actually need the data — e.g. benchmarking the memory subsystem, confirming HBM clocks, or validating SoL for a compute-bound kernel that happens to be upstream of memory. For any kernel that *consumes* all the loaded bytes in SM registers, the relevant ceiling is 90.7%.

### 1.9 `cp.async` staged copy experiment

I also prototyped a persistent double-buffered kernel (`bench_dram_cpasync.cu`) using `cp.async.ca.shared.global.L2::128B` to pipeline loads into shared memory and stores from shared. It did **not** beat the naive path — the extra smem round-trip adds latency and contends with the store queue. For pure DRAM copy on B300, the naive `ld + st` pair with an adequate workload and a non-persistent grid is the right baseline.

### 1.10 Bottom line

| | %SoL | Caveat |
|---|---:|---|
| Naive baseline (original `bench_dram_bw.cu`, 256 MiB) | **81.1%** | — |
| Best naive COPY (2 GiB, v4.u64 W=32 U=1 blk=1024) | **85.0%** | — |
| Best COPY (v8.f32 + LOCAL_STRIDE U=2) | **85.8%** | honest |
| Honest LOAD-to-SM ceiling (v8.f32 + PA=32 prefetch) | **90.7%** | full data into registers |
| STORE ceiling (v8.f32 U=1) | **92.5%** | honest |
| DRAM→L2 ceiling (prefetch256 trick) | **93.3%** | **SM registers get only 1/64 of the sector** |

The remaining ~7% on honest reads and loads is the HBM3e command-bus / refresh / precharge floor. The remaining ~14% on COPY is read/write turnaround on the HBM bus — halving that would require TMA (`cp.async.bulk` with tensor descriptors) or per-die-local copy to avoid cross-die R/W contention.

---

## 2. FP32 FMA Throughput

| Measured | Peak | % SoL |
|---:|---:|---:|
| **67.87 TFLOPS** | 83.23 TFLOPS | **81.5%** |

At 256 threads × 8 chains × UNROLL=8, Blackwell hits 112.6 FMA/SM/clk (peak 128, -12%). Historical data shows 1024 thr/blk climbs to ~87% — insufficient warp count is the limiter, not the ALU.

---

## 3. Narrow-Format CVT Throughput (`cvt.rn.satfinite.*`)

1 op = 1 PTX CVT instruction. `.x2` variants convert 2 elements/op; `.x4` convert 4 elements/op.

### 3.1 To-narrow from `f16x2` (16 CVTs / iter)

| Destination | GOps/s | ops/SM/clk | Note |
|---|---:|---:|---|
| `e2m1x2` (FP4/NVFP4) | **6299** | 20.94 | — |
| `relu.e2m1x2` | 6298 | 20.94 | no penalty |
| `e4m3x2` (FP8) | **7332** | 24.38 | full-rate F2FP |
| `e5m2x2` (FP8) | 7332 | 24.38 | full-rate F2FP |
| `e2m3x2`, `e3m2x2` (FP6) | **UNSUPPORTED** on sm_103a | | |

FP4 is ~14% slower than FP8 from `f16x2` because it needs the extra `.b8→b16` pack/mov.

### 3.2 To-narrow from `f32` pair (8 CVTs / iter)

| Destination | GOps/s | ops/SM/clk |
|---|---:|---:|
| `e2m1x2` (FP4) | 6545 | 21.76 |
| `e4m3x2` | **7296** | 24.26 |
| `e5m2x2` | 7295 | 24.26 |
| `e2m3x2` (FP6) | **7296** | 24.26 |
| `e3m2x2` (FP6) | 7295 | 24.26 |

All non-FP4 hit ~24.3 ops/SM/clk — one PTX CVT every ~1.3 clocks per warp scheduler. FP6 is supported only from `f32`, not from `f16x2`/`bf16x2`.

### 3.3 From-narrow to `f16x2` (16 CVTs / iter)

| Source | GOps/s | ops/SM/clk |
|---|---:|---:|
| `e2m1x2` → `f16x2` | **7385** | 24.56 |
| `e4m3x2` | 7385 | 24.56 |
| `e5m2x2` | 7385 | 24.56 |
| `e2m3x2` | 7373 | 24.52 |
| `e3m2x2` | 7376 | 24.52 |

From-narrow is uniform and fast across all five encodings.

### 3.4 x4 stochastic-rounding `cvt.rs` from `f32` (4 CVTs / iter, 4 elements each)

| Destination | GOps/s | Elements/s |
|---|---:|---:|
| `e2m1x4` | 3429 | **13716 GEl/s** |
| `relu.e2m1x4` | 3434 | 13737 |
| `e4m3x4` | **3641** | **14562** |
| `e5m2x4` | 3641 | 14566 |
| `e2m3x4` | 3643 | 14571 |
| `e3m2x4` | 3641 | 14565 |

x4 runs at ~half the PTX-op rate of x2 but handles 4 elements per dispatch. **Per-element throughput is essentially equal** to the x2 path — making `.rs.x4.f32` a strict win for stochastic-rounding dequant pipelines.

---

## 4. MUFU / Transcendental Throughput

`bench_mufu.cu`, 256 thr × 148 blocks, UNROLL=8, swept over chain count.

### 4.1 f32 opcodes (GOps/s)

| Opcode | chains=1 | 2 | 4 | 8 | 16 | peak op/SM/clk |
|---|---:|---:|---:|---:|---:|---:|
| `rcp.approx.f32` | 1647 | 3028 | 4018 | 4190 | **4400** | 14.6 |
| `rsqrt.approx.f32` | 1851 | 2913 | 4532 | 4752 | **4719** | 15.8 |
| `sqrt.approx.f32` | 1851 | 2913 | 4533 | 4741 | **4719** | 15.8 |
| `sin.approx.f32` | 3155 | 4600 | 4726 | 4757 | **4765** | 15.8 |
| `cos.approx.f32` | 3155 | 4600 | 4726 | 4757 | **4764** | 15.8 |
| `lg2.approx.f32` | 1851 | 2914 | 4533 | 4752 | **4719** | 15.8 |
| `ex2.approx.f32` | **4224** | **7623** | **9252** | **9413** | **9471** | **31.5** |
| `tanh.approx.f32` | 4095 | 4689 | 4748 | 4767 | **4776** | 15.9 |

- Most MUFU ops saturate at **16 ops/SM/clk** (standard SFU rate), needing ≥ 4 chains/thread to approach peak.
- **`ex2.approx.f32` runs at 32 ops/SM/clk** — exactly 2× the other transcendentals, a Blackwell-specific acceleration.
- `rcp.approx.f32` peaks ~14.6/SM/clk — a back-to-back hazard in the reciprocal pipe.
- `sin`/`cos` reach peak at chain=2 (shorter pipeline).

### 4.2 Packed and scalar half variants (chains=8)

| Opcode | GOps/s | Elements/s | Notes |
|---|---:|---:|---|
| `ex2.approx.f16x2` | 4759 | 9518 GEl/s | same as f32 scalar rate → half per-element rate |
| `tanh.approx.f16x2` | **2389** | 4777 | **1/2 rate** (8 ops/SM/clk) |
| `tanh.approx.bf16x2` | 2387 | 4774 | same half-rate |
| `ex2.approx.bf16x2` | **UNSUPPORTED** | | — |
| `rcp.approx.f16x2`/`bf16x2` | UNSUPPORTED | | — |
| `ex2.approx.f16` (scalar) | 9487 | — | matches f32 scalar rate |
| `ex2.approx.bf16` | UNSUPPORTED | | — |
| `tanh.approx.f16` | 4750 | — | scalar rate = f32 |
| `tanh.approx.bf16` | 4751 | — | scalar rate = f32 |

**Packed `.f16x2`/`.bf16x2` forms do NOT get a 2× element throughput bump.** Only `ex2` has 2× throughput over baseline among all transcendentals.

---

## 5. BF16 → NVFP4 Quantization Kernels

`quantize_bf16_to_nvfp4.cu` (flat): each thread loads 16 BF16 (256 b), computes absmax over 16, scales, does 8 `cvt.rn.satfinite.e2m1x2.f32`, packs to 8 B FP4 + 1 scale byte. 128M BF16 inputs per run (≈ 343.9 MiB moved = 256 MiB read + 64 MiB FP4 write + 8 MiB scale write).

| Variant | Time | Bandwidth | % HBM SoL |
|---|---:|---:|---:|
| Flat, e4m3 scale (default) | 0.0523 ms | **6582 GB/s** | **80.4%** |
| Flat, bf16 scale (`#define SCALE_BF16`) | 0.0534 ms | **6603 GB/s** | **80.7%** |
| Row-wise, ROW_DIM=4096, contiguous scales | 0.0611 ms | 5629 GB/s | 68.8% |
| Row-wise, ROW_DIM=4096, swizzled (cuBLAS) | 0.1188 ms | 2895 GB/s | 35.4% |
| Row-wise, ROW_DIM=2048 | 0.0874 ms | 3935 GB/s | 48.1% |
| Row-wise, ROW_DIM=8192 | 0.0598 ms | 5752 GB/s | 70.3% |
| Row-wise, ROW_DIM=16384 | 0.0664 ms | 5177 GB/s | 63.3% |

- Flat kernel is HBM-bound (80%) — F2FP and scale work hide behind memory.
- Row-wise costs ~12% vs. flat because per-row absmax needs a warp shuffle + shared-mem barrier.
- ROW_DIM=4096 is optimal; below → not enough threads/block; above → the inter-warp reduction warp becomes the tail.
- **Swizzled scale layout halves throughput** — the cuBLAS `e8` layout produces fully uncoalesced stores (stride = `num_rows` bytes per adjacent thread). Emit contiguous scales if you can.

Note: the flat kernel at 6582 GB/s is already above the "naive memcpy" COPY baseline of 6635 GB/s but below the optimized ceiling of 6959 GB/s — room for tuning.

---

## 6. L2 Side-Aware Reduction (`side_aware.cu`)

Hackathon-winning kernel. Computes FP32 absmax over a 4 GB input (10⁹ floats). B300 is a dual-die package; 50% of naïve accesses cross the on-package interconnect at higher latency/power. The kernel discovers which 2 MiB pages live on which die and reschedules work so every SM reads only its **near** die.

Three build-time modes:

- **optimal** — each SM reads near-side pages only (the real optimization)
- **random** — `side ^= page_side` (50/50, no benefit, same overhead)
- **wrong** — `side ^= 1` (every access forced to the far die, pessimal)

Each mode × both zero-filled input and random input:

| Variant | Input | Time | Bandwidth | % HBM SoL |
|---|---|---:|---:|---:|
| **Optimal** | zero | 0.5748 ms | **6958 GB/s** | **85.0%** |
| Optimal | random | 0.5747 ms | 6960 | 85.0% |
| Random (50/50) | zero | 0.5876 ms | 6807 | 83.2% |
| Random | random | 0.5885 ms | 6797 | 83.0% |
| **Wrong** (far only) | zero | 0.8228 ms | **4861** | **59.4%** |
| Wrong | random | 0.8229 ms | 4861 | 59.4% |

- **Optimal vs wrong: +43.1% bandwidth / −30.1% time.** This is the full cost of cross-die traffic when it's 100% of accesses.
- **Optimal vs 50/50 random: only +2.2%.** At 50/50 traffic, half the accesses are already near-side and HBM bandwidth itself is the binding constraint. The optimization shines only when you can shift *all* the "far" half back.
- Input randomness doesn't affect throughput — the kernel is memory-bound.
- **The optimal kernel at 85.0% matches the optimized COPY ceiling from §1** — it's effectively saturating the per-die HBM.

---

## 7. Full CVT Conversion Sweep (`sweep_cvt.sh`, 163 variants)

### 7.1 Instruction availability on sm_103a

| To-narrow from | Available destinations |
|---|---|
| `f32` (pair) | `e2m1x2` (+relu), `e4m3x2` (+relu), `e5m2x2` (+relu), `e2m3x2` (+relu), `e3m2x2` (+relu) |
| `f16x2` | `e2m1x2` (+relu), `e4m3x2` (+relu), `e5m2x2` (+relu). **FP6 not supported** |
| `bf16x2` | **none** (except via `ue8m0x2` scale path) |
| `f32 × 4` (`.rs` stochastic) | all five narrow formats (+relu) |

| From-narrow to | Availability |
|---|---|
| `f16x2` | all 5 narrow formats |
| `bf16x2` | **none** |

| Scale-factor | Availability |
|---|---|
| `ue8m0x2` ↔ `bf16x2`, `ue8m0x2 ← f32` | supported, rz/rp rounding |

### 7.2 Standard-precision conversions (f16/bf16/tf32)

| Instruction | GOps/s | ops/SM/clk | Notes |
|---|---:|---:|---|
| `cvt.rn.f16.f32` (scalar) | 4326 | 14.4 | |
| `cvt.rn.satfinite.f16.f32` | **6819** | **22.7** | +57% vs. plain — `.satfinite` compiles to faster SASS |
| `cvt.rz.f16.f32` | 4328 | 14.4 | |
| `cvt.rz.satfinite.f16.f32` | 6814 | 22.7 | |
| `cvt.rn.bf16.f32` | 4326 | 14.4 | |
| `cvt.rn.satfinite.bf16.f32` | 6795 | 22.6 | |
| `cvt.rn.f16x2.f32` | **7960** | 26.5 | packed, no `.satfinite` needed |
| `cvt.rn.bf16x2.f32` | **7964** | 26.5 | |
| `cvt.rn.satfinite.tf32.f32` | 8420 | **28.0** | tf32 is fastest of all CVTs |
| `cvt.rna.satfinite.tf32.f32` | 4093 | 13.6 | `.rna` is half-rate vs `.rn`/`.rz` |
| `cvt.f32.f16` / `cvt.f32.bf16` | ~3e5 GOps/s | — | **elided by optimizer** — numbers meaningless |

**Footgun: always use `cvt.rn.satfinite` or `cvt.rz.satfinite` for scalar f32→f16/bf16.** Non-`satfinite` scalar forms are 57% slower.

### 7.3 UE8M0 scale-factor conversions

| Instruction | GOps/s | ops/SM/clk |
|---|---:|---:|
| `cvt.rz.ue8m0x2.bf16x2` | 6342 | 21.1 |
| `cvt.rp.ue8m0x2.bf16x2` | 6337 | 21.1 |
| `cvt.rz.ue8m0x2.f32` | 7763 | 25.8 |
| `cvt.rp.ue8m0x2.f32` | 7768 | 25.8 |
| `cvt.rn.bf16x2.ue8m0x2` (inverse) | 6729 | 22.4 |

Rounding mode doesn't affect throughput; `.f32` path beats `.bf16x2` by ~22%.

---

## 8. Instruction Mix Sweep: F2FP + Companion Pipes (`sweep_mix_e2m1.sh`)

Does F2FP co-issue with math pipes or share a dispatch slot? `bench_mix_e2m1.cu` interleaves N CVTs (`cvt.rn.satfinite.e2m1x2.f16x2`) with N companion instructions. 256 thr × 148 blocks, UNROLL=4, ITERS=8192.

Since total work is `ITERS × (N_CVT + N_COMP)` PTX instructions: **perfect co-issue = max(cvt-alone, comp-alone) ms**, **no co-issue = sum of the two**.

### 8.1 Baselines (ms)

| | cvt=1 | cvt=2 | cvt=4 | cvt=8 | cvt=12 | cvt=16 |
|---|---:|---:|---:|---:|---:|---:|
| CVT-only (e2m1x2.f16x2) | 0.0574 | 0.1060 | 0.1947 | 0.2683 | 0.3135 | 0.3387 |

| | comp=4 | comp=8 | comp=16 |
|---|---:|---:|---:|
| FFMA-only | 0.0758 | 0.1454 | 0.2744 |
| FMUL-only | 0.0758 | 0.1401 | 0.2695 |
| LOP3 (xor.b32) | 0.0302 | 0.0301 | 0.0302 |
| IADD3 (add.u32) | 0.0430 | 0.0758 | 0.1404 |
| IMAD (mad.lo.u32) | 0.0758 | 0.1402 | 0.2695 |
| MUFU (ex2.approx.f32) | 0.2520 | 0.4830 | 0.9359 |

### 8.2 Co-issue vs. serialized (cvt=4 + comp=4)

| Companion | comp=4 alone | cvt=4+comp=4 | perfect | serial | interpretation |
|---|---:|---:|---:|---:|---|
| FFMA | 0.0758 | **0.1989** | 0.1947 | 0.2705 | ≈ perfect co-issue (+2.2%) |
| FMUL | 0.0758 | 0.2021 | 0.1947 | 0.2705 | ≈ perfect co-issue (+3.8%) |
| IMAD | 0.0758 | 0.2111 | 0.1947 | 0.2705 | near-perfect (+8.4%) |
| IADD3 | 0.0430 | 0.2294 | 0.1947 | 0.2377 | partial overlap |
| LOP3 | 0.0302 | **0.1997** | 0.1947 | 0.2249 | ≈ perfect co-issue — LOP3 is free |
| MUFU (ex2.f32) | 0.2520 | **0.3485** | 0.2520 | 0.4467 | partial co-issue — shares SFU pipe |

**FFMA, FMUL, IMAD, and LOP3 all co-issue essentially for free with F2FP.** A fused quant+matmul-prep kernel can pack FFMAs alongside F2FPs and pay ≈0 time.

### 8.3 MUFU contention

Fix cvt=4, sweep N_MUFU:

| N_MUFU | ms | MUFU-alone | CVT-only | perfect-parallel |
|---:|---:|---:|---:|---:|
| 0 | 0.1946 | — | 0.1947 | 0.1947 |
| 1 | 0.2022 | 0.0634 | 0.1947 | 0.1947 |
| 2 | 0.2110 | 0.1266 | 0.1947 | 0.1947 |
| 4 | **0.3485** | 0.2520 | 0.1947 | 0.2520 |
| 8 | 0.5491 | 0.4830 | 0.1947 | 0.4830 |
| 16 | 1.0177 | 0.9359 | 0.1947 | 0.9359 |

Once MUFU count exceeds ~2, MUFU fully dominates. **F2FP and MUFU share the SFU execution pipe on Blackwell.** In a quant kernel that also does softmax/GELU, `ex2`/`tanh` cost is *added* to the F2FP cost, not hidden.

---

## 9. What Was NOT Rerun

- `icache_sweep.py` — GPC-discovery tool hardcoded for 114 blocks (H100-era). B300 has 148 SMs and a different GPC layout; running as-is would only probe the first 114 SMs. Not a throughput benchmark.
- `0_WIP.cu`, `1_LLMC.cu`, `1_RELU.cu`, `vector_add.cu`, `icache.cu` — scratch kernels without timing harnesses.

---

## 10. Artifacts

All captured output is under `/tmp/qrc_results/`:

| File | Contents |
|---|---|
| `01_main_suite.txt` | `run_microbench.sh` full output |
| `02_mufu.txt` | MUFU chain sweep |
| `03_quantize.txt` | BF16 → NVFP4 quant kernels |
| `04_side_aware.txt` | L2 side-aware, 3 variants × 2 inputs |
| `05_cvt_sweep.txt` | `sweep_cvt.sh` — 163 CVT variants |
| `06_mix_sweep.txt` | `sweep_mix_e2m1.sh` — co-issue analysis |
| `07_dram_sweep.txt` | Parametric DRAM variants (512 MiB) |
| `08_dram_push.txt` | Large-workload DRAM push to 92% / 89% / 85% |
| `09_dram_cpasync.txt` | cp.async staged-copy experiment (did not improve) |
| `10_dram_load_push.txt` | Cache-hint + block/unroll sweep for LOAD |
| `11_dram_load_mega.txt` | 16–32 GiB LOAD + L2 prefetch-ahead (90.4% ceiling) |
| `12_reference.txt` | Reference-repo kernels: v8.f32, prefetch256, evict_last, LOCAL_STRIDE |
| `13_prefetch256_scale.txt` | Workload-size sweep of prefetch256 trick (93.3% peak) |
| `14_prefetch_consume.txt` | Honest DRAM→SM with prefetch+v8 consume (90.7% ceiling) |
| `15_prefetch_ahead.txt` | ADVANCE_BYTES sweep (why "40 MiB ahead" doesn't work) |
| `16_sparse_prefetch.txt` | 1-in-8 lane sparse prefetcher (still capped at 90.7%) |
| `17_pure_v8.txt` | "Just v8.f32, no tricks" — 90.7 / 92.6 / 85.0 for LOAD/STORE/COPY |

New parametric kernels added:

- `tests/bench_dram_variants.cu` — 4-mode × 4-width × variable-unroll × persistent/non-persistent × 7 cache-hint variants
- `tests/bench_dram_cpasync.cu` — double-buffered cp.async copy prototype (didn't help)
- `tests/bench_dram_prefetch.cu` — explicit `prefetch.global.L2` ahead-of-read LOAD
- `tests/bench_dram_prefetch_consume.cu` — honest DRAM→SM LOAD using prefetch256 + v8.f32 consume
- `tests/reference/*.cu` — kernels copied from `ademeure/private-quickrun-cuda` (`membw.cu`, `membw_prefetch256.cu`, `membw_prefetch_inst.cu`, `membw_ptx.cu`)

SASS for each compiled kernel is dumped under `sass/`.
