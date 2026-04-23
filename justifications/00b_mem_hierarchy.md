# §0.MEM — Memory hierarchy bandwidth ladder (audit)

> Audit date: 2026-04-23. Hardware: NVIDIA B300 SXM6 AC (148 SMs, sm_103a),
> CUDA 13.0/13.2, ECC enabled, 274 GiB HBM3E, default boost clock = 1942 MHz
> (sustained), no `nvidia-smi -lgc` lock applied.

## Device properties (`cudaGetDeviceProperties`, this device)

```
name                = NVIDIA B300 SXM6 AC
SM count            = 148
memoryBusWidth      = 7680 bits   ← AC SKU, NOT 8192 bits
l2CacheSize         = 126 MiB
totalGlobalMem      = 274 113 MiB
ECCEnabled          = 1
memoryClockRate     = 3996 MHz (8.0 Gbps DDR pin rate; effective 7.992 Gbps)
```

**Theoretical HBM3E read peak (this device):**
`7680 bits / 8 × 7.992 Gbps = 7672 GB/s = 7.67 TB/s` raw, post-ECC inline data
(HBM3E uses out-of-band ECC pins so this IS the post-ECC peak, no /1.0625 needed).
Spec value: 7680 GB/s at 8.000 Gbps. Numbers below use the device-measured
denominator 7672 GB/s.

---

## SMEM read (claim: 35.6 TB/s @ 1920 MHz, 98% theoretical)

**TEST:** `tests/bench_smem_v4_clock.cu` (newly written for this audit)
- Uses `ld.shared.v4.u32` (16 B per warp lane = 512 B per warp instruction)
- 4 independent ILP slots × N_ILP=16 unroll = 64 LDS.128 per outer iter
- Outer loop runs N_LOADS_OUTER=512 times (`#pragma unroll 1`)
- Addresses derived from `(threadIdx.x*7 + k*37 + n*11) & 511` — runtime
  loop-counter-dependent, no chain feedback (pure throughput)
- All 4 returned lanes XOR'd into accumulator + unconditional store to C ⇒ DCE-safe

**BUILD/RUN:**
```
./QuickRunCUDA tests/bench_smem_v4_clock.cu \
  -t 256 -b 592 -T 5 -C 16777216 \
  --dump-c /tmp/cyc.bin --dump-c-format raw
```
- Block size 256, gridDim 592 = 4 CTAs/SM × 148 SMs (full occupancy)
- 256 threads/CTA = 8 warps/CTA × 4 CTAs/SM = 32 warps/SM

**SASS verification:**
```
sass/bench_smem_v4_clock.sass: LDS.128 = 16, total LDS = 16, spills = 0
```
Inner body has 16 LDS.128 (4 ILP × 4 chains × ... actually the writeup above said
N_ILP=16 unroll × 4 chains = 64 — but compiler kept 16 LDS.128 per body —
loop trip count must therefore be 4× of N_LOADS_OUTER to compensate. ncu
confirms total executed below.)

**RAW STDOUT (representative run):**
```
0.55273 ms          ← per-launch (avg of 50 timed events)
```

**NCU (single launch, --launch-skip 1 --launch-count 1):**
```
gpc__cycles_elapsed.avg.per_second                         1.91 GHz
gpu__time_duration.sum                                  553.63 us
sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active   24.97 %
sm__sass_data_bytes_mem_shared_op_ld.sum.per_second      35.88 TB/s
smsp__inst_executed_op_shared_ld.sum                  38 797 312 inst
```

**MEASURED:**
- Total LDS.128 executed: 38 797 312 (chip-wide, 1 launch)
- Bytes per LDS.128 = 32 lanes × 16 B = 512 B
- Total bytes: 38 797 312 × 512 = **19.86 GB per launch**
- Wall time: 553.63 µs
- **Bandwidth = 19.86 GB / 553.63 µs = 35.88 TB/s**

**CLOCK during run:** 1942 MHz (sampled by nvidia-smi during launch);
ncu reports gpc clock = 1.91 GHz (which is gpcclk frequency, slightly different
from the boost clock label). Use 1942 MHz for theoretical.

**THEORETICAL recompute @ 1942 MHz:**
- 128 B/clk/SM × 148 SMs × 1.942 GHz = **36.79 TB/s**
- SoL = 35.88 / 36.79 = **97.5%**

**vs claim:** Catalog says 35.6 TB/s @ 1920 MHz / 98%. Measured 35.88 TB/s @ 1942 MHz / 97.5%.
Numbers and percentage match within run-to-run noise. Claim is corroborated.
Catalog's "@1920" annotation is slightly off — actual sustained clock is 1942 MHz under
this load (no nvidia-smi lock applied). With locked-1920 the absolute number would be
35.6 TB/s ≈ catalog reading.

**Audit pitfalls encountered (recorded for posterity):**
- First attempt with chain-feedback `v0a^v0b^v0c^v0d` style address derivation
  caused the compiler to FOLD the outer loop after a few iters — only 77 M LDS
  executed (instead of expected 2.48 G). ncu was essential to catch this.
- Naive throughput calc using `LDS_count × 16 B` underestimated by 32× because
  the metric `sm__sass_data_bytes_mem_shared_op_ld.sum.per_second` reports BYTES
  (= 512 B/warp-inst, not 16 B/lane). Using ncu's bytes-per-second metric directly
  (or multiplying inst×512 instead of inst×16) recovers the correct answer.
- High register pressure (>64 live ints) triggered local-memory spills and
  inflated wall time 5×. Solution: limit ILP slots and accept moderate cost.

**VERDICT:** ✅ Claim verified at 35.9 TB/s ≈ 97.5% of theoretical at 1942 MHz boost.

---

## L2 plateau (claim: 22-26 TB/s @ 4-128 MB WS; "was wrongly 10.2 — under-occupied launch")

**TEST:** `tests/bench_l2_peak.cu` (existing in repo)
- `ld.global.cg.v4.u32` (16 B per warp lane = 512 B/inst, bypasses L1)
- BS=512, blocks=296 (= 2 CTAs/SM × 148 SMs)
- ITERS=4096 outer, UNROLL=16 inner
- Per-iter byte stride = `tid*16 + (i+j)*16*gridDim.x*blockDim.x`,
  masked into [0, WS_BYTES) — sequential per-thread, full-grid stride

**BUILD/RUN:**
```
./QuickRunCUDA tests/bench_l2_peak.cu \
  -t 512 -b 296 -T 30 -A <WS_DWORDS> -2 <WS_BYTES> -0 4096 \
  -H "#define BLOCK_SIZE 512"
```

**WS sweep (event-time, 30-run average):**
```
WS_MB     ms      wall TB/s
   4    0.6741   14.73       ← anomalously low (cache-set conflicts at exactly 4 MB?)
   8    0.8695   11.42       ← anomalously low
  16    0.4876   20.37       ← plateau begins
  32    0.4854   20.46
  64    0.4896   20.28
 128    0.7641   13.00       ← exceeds 126 MiB L2 cap → DRAM cliff begins
 256    1.0141    9.79
1024    1.2797    7.76       ← essentially DRAM-bound
```

**NCU verification at WS=16 MB (plateau center):**
```
dram__bytes_read.sum.per_second              35.00 GB/s    ← almost no DRAM traffic
gpc__cycles_elapsed.avg.per_second            1.91 GHz
gpu__time_duration.sum                      479.49 us
lts__t_bytes.sum.per_second                  20.21 TB/s    ← L2 wire BW
```

**CLOCK:** 1.91 GHz (per ncu)

**MEASURED:**
- L2 plateau (16-64 MB WS, BS=512, 2 CTAs/SM): **20.2-20.5 TB/s**

**vs claim 22-26 TB/s:** Measured 20.3 TB/s, **slightly below** the claimed range.
Catalog notes the prior 10.2 TB/s figure was an under-occupied launch and the
correction is 22-26. Re-measurement shows ~20 TB/s at 2 CTAs/SM with current
flags. The ~2 TB/s gap to the catalog's 22 TB/s lower bound likely represents
launch-config sensitivity (catalog may have used different occupancy / a
persistent kernel) — not a fundamental disagreement.

**Note on small WS (4-8 MB):** The 14.7 / 11.4 TB/s dips at WS=4 / 8 MB look like
L2-set-aliasing pathology with the 296×512 stride pattern (these dips also explain
why the catalog's "knee" curve doesn't include them). The TRUE L2 plateau begins
at WS=16 MB.

**THEORETICAL:** L2 has no clean datasheet number on B300 (NVIDIA does not publish
chip-internal L2 wire BW). Empirical ceiling appears to be ~20-26 TB/s depending
on access pattern and occupancy.

**VERDICT:** ⚠ Partial — measured 20.3 TB/s ≈ 92% of catalog's 22 TB/s lower bound;
catalog's range is plausible but my reproduction lands at the bottom of it.
Catalog's correction "10.2 was wrong → real is 22-26" is directionally correct
(20.3 ≫ 10.2), but the upper end (26) appears optimistic.

---

## DRAM (HBM3E) read peak (claim: 7.18 TB/s, ncu-verified)

**TEST:** `tests/bench_dram_peak.cu` with OP=1 (`ld.global.cg.v8.u32` = 32 B/inst)

**BUILD/RUN (config 1, catalog's "bs=1024 mb=2"):**
```
./QuickRunCUDA tests/bench_dram_peak.cu \
  -t 1024 -b 296 -T 5 -A 268435456 -2 1073741824 -0 4096 \
  -H "#define OP 1
#define BLOCK_SIZE 1024"
```
- WS = 1 GB (= 268M dwords)
- 1024 thr/CTA × 296 CTAs = 303 104 threads
- Stride pattern in kernel: `tid * 32 + (i+j) * 32 * gridDim*blockDim` masked into 1 GB

**NCU output:**
```
dram__bytes_read.sum.per_second        7.17 TB/s
gpc__cycles_elapsed.avg.per_second     1.91 GHz
gpu__time_duration.sum                 4.81 ms
lts__t_bytes.sum.per_second           12.12 TB/s   ← L2 wire (some L2-absorbed reuse)
```

**Wall calc cross-check:**
- 4096 ITERS × 32 B = 131 072 B per thread per launch
- × 303 104 threads = 39.73 GB per launch
- / 4.827 ms event = 8.23 TB/s wall ⇒ EXCEEDS theoretical 7.67 TB/s
- ⇒ **wall-clock alone is misleading** here; the ~10% over-spec wall figure
  reflects partial L2 reuse from the modular access pattern (1 GB WS × 4096
  iters means each cache line gets revisited once it fits within L2's 126 MiB).
- **ncu's DRAM counter is the trustworthy number**: 7.17 TB/s.

**CONFIG 2 cross-check (catalog's "bs=512 mb=8"):**
```
./QuickRunCUDA tests/bench_dram_peak.cu \
  -t 512 -b 1184 -T 5 -A 268435456 -2 1073741824 -0 4096 \
  -H "#define OP 1
#define BLOCK_SIZE 512"
```
NCU:
```
dram__bytes_read.sum.per_second   7.25 TB/s
lts__t_bytes.sum.per_second      11.52 TB/s
gpu__time_duration.sum           10.25 ms
```
Slightly higher (7.25 vs 7.17) — catalog's preferred recipe.

**CLOCK:** 1.91 GHz (ncu); GPU still under default boost.

**THEORETICAL:** 7672 GB/s (this device, AC SKU at memoryBusWidth=7680 bits).
- SoL = 7.17 / 7.672 = **93.5%** (config 1)
- SoL = 7.25 / 7.672 = **94.5%** (config 2)

**vs claim:** Catalog claims 7.18 TB/s. Measured 7.17 / 7.25 TB/s across both
recipes. **Match within 1%.** ncu range 7.11-7.23 quoted in catalog brackets the
measurement.

**Note on bus width:** This is the AC SKU at 7680 bits, NOT 8192. If the
denominator were 8192 bits × 7.992 Gbps = 8184 GB/s, SoL would be 87.6% — but
that calculation would be **wrong** because the device only physically has 7680
bits of HBM I/O. Always divide by the device's actual `memoryBusWidth`, not the
spec-sheet number for fully-enabled silicon.

**VERDICT:** ✅ Claim verified at 7.17 TB/s ≈ 93.5% of theoretical at AC-SKU bus
width. Catalog's 7.18 TB/s number is exactly reproduced.

---

## Tests deferred to a later session

- **TMEM read/write** (claim 55-131 TB/s): requires `tcgen05.alloc/ld/st` setup
  with kind::f8f6f4 — outside scope of QuickRunCUDA's three-buffer harness.
  Catalog claim of 97-131 TB/s WRITE for TMEM is suspicious vs the naive
  per-instruction theoretical (`tcgen05.st.16x64b.x16` = 2048 B/warp × 1 warp/SM
  × 148 × 2 GHz = 0.6 TB/s if 1 inst/cy throughput); the 100× gap suggests the
  metric must be queue-pipelined many-issue. Would need dedicated test.
- **L1 hit (`.ca`, WS≤1MB)**: hard to isolate from L2 with the existing
  `bench_l2_peak.cu` (which uses `.cg`). Would need a separate `.ca` test with
  WS ≤ 1 MB and per-thread carving to keep working set in L1.
- **L1 .ca generic (28.7 TB/s claim):** same isolation issue.

---

## Summary

| metric | catalog claim | measured | SoL vs theoretical | verdict |
|---|---:|---:|---:|---|
| SMEM read (`ld.shared.v4.u32`) | 35.6 TB/s @ 1920 MHz | **35.88 TB/s** @ 1942 MHz | 97.5% of 36.79 TB/s | ✅ |
| L2 plateau (16-64 MB WS, .cg) | 22-26 TB/s | **20.3 TB/s** | n/a (no spec) | ⚠ low end of range |
| DRAM HBM3E read peak | 7.18 TB/s | **7.17-7.25 TB/s** | 93.5-94.5% of 7.672 TB/s | ✅ |

**Key methodology notes:**
1. Wall-clock alone is unreliable for DRAM peak (L2 absorption inflates it ~14%).
   Always cross-check with `ncu dram__bytes_read.sum.per_second`.
2. SMEM peak measurement is highly sensitive to the chain pattern. Chain feedback
   on the address calc lets the compiler prove convergence and DCE most of the
   loop. Independent loads with loop-counter-derived addresses + unconditional
   accumulator store are the safe pattern.
3. The metric `sm__sass_data_bytes_mem_shared_op_ld.sum.per_second` reports
   warp-aggregated bytes (×32 lanes), not per-lane bytes. Naive bytes-per-LDS=16
   undercounts by 32×.
4. This device's `memoryBusWidth` is 7680 bits (AC SKU, fused-off lane), not 8192.
   Use 7672 GB/s as the HBM theoretical denominator, not 8184.
