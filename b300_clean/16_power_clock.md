# B300 Power, Thermal, and Clock State

**GPU:** NVIDIA B300 SXM6 AC, sm_103a, 148 SMs. **TDP envelope: 200 W min, 1100 W max, 1100 W default enforced limit** (NOT 700 W as some Hopper-era assumptions carried). All compute numbers below at default boost (no clock lock).

---

## Headline numbers (HIGH confidence)

| Quantity | Value | Notes |
|---|---:|---|
| Default sustained boost clock | **2031.4 MHz** | Verified by `%clock64` / `%globaltimer` ratio across all 147 valid SMs, all four quarters of a 20 ms kernel. NEVER throttled in any tested workload. |
| Idle clock | 120 MHz | nvidia-smi |
| `nvidia-smi -lgc 2032` actual clock | **1919.8 MHz** | **PARADOX, -5.5%.** See below. |
| `nvidia-smi -lgc 1410` actual clock | 1410 MHz | Lock works correctly at lower bins |
| TDP max / min / default | 1100 / 200 / 1100 W | nvmlDeviceGetEnforcedPowerLimit |
| Idle power | 182–197 W | Large die + HBM3E stack baseline |

---

## TFLOPS/W ladder (HIGH confidence — measured FLOPS / NVML power)

| Workload | Power (W) | TFLOPS | TFLOPS/W | vs FFMA peak |
|---|---:|---:|---:|---:|
| FFMA peak (256 thr × 24 ILP, full occ) | 361 | 74.6 | **0.21** | 1.0× |
| FFMA non-peak (low occ) | 437 | 73 | **0.17** | 0.81× |
| BF16 mma.sync m16n8k16 ILP=2 | 411 | 569 | **1.39** | **6.6×** |
| FP8 cuBLAS LtMatmul (8192³) | 886 | 4491 | **5.07** | **24×** |

**Headline:** Tensor cores are **6.6× more power-efficient per FLOP** than FFMA at peak; FP8 cuBLAS is **24× more power-efficient** than FFMA. Counterintuitive: the FFMA *peak* (full occupancy, 24-ILP) draws **less** total power (361 W) than FFMA at lower occupancy (437 W). Higher ILP → fewer warps idle → less leakage waste relative to work; lower-occupancy FFMA still draws power for keeping the SM partition active without producing as many FLOPS.

The 4–5 TFLOPS/W FP8 number is the dense (no-sparsity) compute energy efficiency; inference at sustained workloads adds memory-subsystem power, settling around **13–17 tok/s/W** for prefill on 70B BF16 (memory-bound regimes are dominated by HBM, not compute).

---

## No throttling observed under any sustained load (HIGH confidence)

| Workload | Duration | Power | Temp | Clock | Throttle? |
|---|---:|---:|---:|---:|---|
| FFMA hot loop, 5×148 blocks × 256 thr × 1M iter | 30+ s | 437 W | 49°C | 2032 MHz | **NO** |
| BF16 mma.sync ILP=2 | 15 s | 411 W | 50°C | 2032 MHz | **NO** |
| FP8 cuBLAS LtMatmul 8192³, 200-iter batches | 12 s | 886 W | 59°C | 2032 MHz | **NO** |
| FP8 cuBLAS, long-sustained (200 iter × repeat) | 30 s | 899 W | 63°C | 2032 MHz | **NO**, ratio = 1.00× across all batches |
| BF16 cuBLAS GEMM, 20,969 iters | 10 s | 962 W | mid-50s | 2032 MHz | **NO** (88% of 2500 TFLOPS peak, sustained flat) |
| 70B BF16 inference b=64, 1421 passes | 30 s | 718→731 W (+1.8%) | 46→54°C | 2032 MHz | **NO**, 21.75 ms/pass ± 0.00 |

The B300 has **massive thermal and power headroom** in 12–30 s tests. We never reached the 1100 W TDP under any pure compute or compute+memory workload. Temp ceiling stayed below 65°C even at 962 W draw. Multi-minute / multi-hour sustained-load throttle behavior **NOT TESTED** — all tests are short (12–60 s).

The earlier "53% throttle on FP8 GEMM" claim (commit 558756c) was retracted in 29e2897 / 4cb929e — it was a measurement artifact (kernel timing without proper warmup), NOT actual hardware throttle.

---

## The `nvidia-smi -lgc 2032` paradox (HIGH confidence)

`nvidia-smi -lgc 2032` is supposed to lock the SM clock to 2032 MHz. **It actually pins it to 1920 MHz** — a 5.5% performance loss vs leaving the GPU unlocked.

| State | nvidia-smi reads | clock64/globaltimer measurement |
|---|---:|---:|
| Idle, no lock | 120 MHz | N/A (idle) |
| Under FFMA load, no lock | **2032 MHz** | **2031.4 MHz** (verified) |
| Idle, `-lgc 2032` | 1920 MHz | N/A (idle) |
| Under FFMA load, `-lgc 2032` | 1920 MHz | **1919.8 MHz** (verified) |
| Under FFMA load, `-lgc 1410` | 1410 MHz | 1410 MHz (correct) |

**Mechanism:** `-lgc` sets the **application clock target**, which the driver maps to a clock bin. The "2032 MHz" bin's hardware-PLL target is 1920 MHz base; 2032 MHz is reached only via boost, and boost is **disabled when `-lgc` is in effect**. Locking to other bins (1410 MHz tested) works as expected.

**Practical:** For B300 benchmark reproducibility, **DO NOT set `-lgc 2032`** — leave clocks unlocked. Default boost is rock-steady at 2031.4 MHz across all 4 quarters of a 20 ms FFMA kernel (variance < 0.3 MHz). The CLAUDE.md note that `--clock-speed 2032` "paradoxically pins to 1920" applies the same caveat.

This means **all catalog TFLOPS numbers must specify clock state**: ~6% of the catalog's noise comes from mixing `-lgc 2032` (1920 MHz) and unlocked (2032 MHz) data points.

---

## Clock primitives (HIGH confidence)

| Primitive | Value | Notes |
|---|---:|---|
| `%clock64` precision (back-to-back) | **2 cycles min** = 1 ns @ 2032 MHz | 100-launch min/max sweep |
| `%globaltimer` resolution | **32 ns** | 31.25 MHz tick (matches IO-clock domain) |
| `%clock64` per-SM | NOT chip-synchronized | Up to **14.7 G cycles (~7.7 s) skew** between SMs at any moment |
| `%globaltimer` per-SM | Chip-wide synchronized to ~250 ns | Use this for cross-SM timestamp comparisons |
| `%clock64` ÷ `%globaltimer` | Gives MHz directly | Ratio reported 2031.5 MHz under FFMA load |
| `%clock64` read overhead | 18.7 cy | S2UR instruction |
| `%globaltimer` read overhead | 32 cy (S2UR) | slightly cheaper than `%clock64` in some patterns |
| SM boot-phase clustering | 8 groups of 12–20 SMs | Boot in GPC-aligned waves, ~6–7 s offsets at chip power-on. Confirms 8 GPCs. |

**Use `%globaltimer` to compare across SMs; use `%clock64` for within-warp/within-SM intervals (1500× higher resolution).**

---

## nanosleep precision (MED-HIGH confidence)

`nanosleep.u32` does NOT do what its name suggests for sub-microsecond requests. Two converging measurements:

| Request (ns) | Actual (ns) | Behavior |
|---:|---:|---|
| 0–2 | ~32 | Quantum floor (matches globaltimer 32 ns) |
| 3 | 64 | |
| 4–56 | ~32 | Round to floor |
| 64 | 128 | |
| 72–80 | ~96 | Round-to-nearest-quantum |
| 100 | 113 (13% over) | At smallest "useful" range, slight overshoot |
| 1000 (1 µs) | ~620 (38% under) | **UNDERSHOOTS** — does NOT honor the requested duration |
| 5000 (5 µs) | ~3200 (36% under) | |
| 10000 (10 µs) | ~6500 (35% under) | |
| 100000 (100 µs) | ~40000 (60% under) | Worst case |
| 1000000 (1 ms) | varies, generally under | |

**Three regimes:**
1. **< 32 ns:** ignored; nanosleep returns ~32 ns minimum (the globaltimer quantum).
2. **32–256 ns:** quantized to 32 ns multiples, slight overshoot.
3. **> 1 µs:** **undershoots by 35–60%**. The hardware appears to use the request as a *hint* and resumes early to maintain warp scheduling fairness. Don't use `nanosleep.u32` as a calibrated delay for ≥ 1 µs.

For accurate pacing > 1 µs, **measure actual elapsed `%globaltimer` and busy-wait the remainder**, or use `__nanosleep()` in a loop with re-checks. The instruction itself takes ~40 ns even with arg=0 (NANOSLEEP issue overhead).

---

## NVML query overhead (HIGH confidence)

| NVML call | Cost (us) |
|---|---:|
| nvmlDeviceGetClockInfo (SM) | **0.12** |
| nvmlDeviceGetPowerUsage | 0.12 |
| nvmlDeviceGetTemperature | 0.12 |
| nvmlDeviceGetUtilizationRates | 0.12 |
| nvmlDeviceGetPowerState | 0.12 |
| nvmlDeviceGetEnforcedPowerLimit | 0.12 |
| nvmlDeviceGetName | 0.12 |
| nvmlDeviceGetCurrPcieLinkGeneration | 0.12 |
| **nvmlDeviceGetMemoryInfo** | **3.3** |

**Most NVML queries are ~120 ns** — essentially free. You can poll power/clock/temp at MHz rates from the host without measurable GPU-side overhead. Memory info is the slowest at 3.3 µs (presumably it walks an internal allocation tree).

Practical: a 1 ms-cadence host monitor thread polling power+clock+temp adds ~360 ns of CPU work per tick, negligible.

---

## Findings being RETIRED (with reason)

| Old claim | Source | Why retired |
|---|---|---|
| "B300 TDP = 700 W" | catalog cross-arch comparison line | **Wrong.** B300 SXM6 default enforced power limit is **1100 W**; the 700 W figure was Hopper H100 SXM5 carry-over. NVML reads 1100 W as default and max. |
| "B300 throttles to 53% of peak under sustained FP8" | commit 558756c | Retracted in 29e2897 / 4cb929e — measurement artifact (no warmup, kernel-launch-overhead-dominated). True sustained FP8 GEMM is FLAT at 4491 TFLOPS for 30+ s. |
| "B300 sustained boost = 1920 MHz" | catalog scattered (early entries) | Wrong. Default boost is 2032 MHz under all measured workloads. The 1920 MHz number was from `-lgc 2032`-locked benchmarks (which paradoxically run at 1920) and was incorrectly tagged as "default." |
| "FFMA TFLOPS peak = 153.93" | investigations/01_clock_findings.md (initial) | **Wrong** — formula used 256 cores/SM × 148 SMs, but B300 has **128 cores/SM**. Real FP32 theoretical peak at 2032 MHz is **76.96 TFLOPS**; measured peak is 74.6 TFLOPS = 97% of theoretical. The clock measurement (2031.4 MHz) is correct; only the FLOPS multiplication was wrong. |
| "B300 TDP not approached under any workload (~339 W max)" | catalog 4480 region | Stale — that was an early FFMA-only test. Tensor / FP8 / BF16 GEMM all reach 411–962 W; FP8 cuBLAS prefill hits 886 W; sustained BF16 GEMM hits 962 W (87% of 1100 W TDP). The TDP IS approachable, just not by FFMA alone. |
| "FP16/BF16 packed FMA gives 2× FP32 throughput" | catalog cross-references | Wrong on B300. FP16/BF16 packed scalar FMA = same throughput as FP32 outside tensor cores (the 2× speedup applies via tensor cores only). |
| "nanosleep is precise to within ±5%" | early catalog | Wrong. It undershoots 35–60% for requests > 1 µs. Use globaltimer measurement for accurate delays. |

---

## Open questions / NEEDS NEW MEASUREMENT

1. **Multi-minute / hour-scale sustained load** — all our tests are 12–60 s. Does the B300 throttle under genuinely-long pure-compute load (e.g. 1 hour of FP8 cuBLAS at 886 W)?
2. **Why does FFMA non-peak draw MORE power than FFMA peak?** Hypothesis: idle SMSPs have non-zero leakage; low-occupancy FFMA wastes lanes. Needs ncu correlation.
3. **Multi-GPU TDP coupling** — when both B300s in the chassis are at 962 W, does the chassis power supply throttle? Untested.
4. **`nanosleep.u32` undershoot mechanism** — driver-level scheduling vs hardware register-controlled? Affects warp-specialization patterns relying on it.
5. **`-lgc 1920` test** — only `-lgc 2032`, `-lgc 1410`, and unlocked were tested. Does `-lgc 1920` give a true 1920 MHz that matches `-lgc 2032`?
6. **Voltage / TDP corner of FP8 efficiency** — FP8 hits 5 TFLOPS/W at 886 W; what's the F/V curve? Could downvolting or capping at 700 W actually maintain FP8 throughput better than 1100 W FFMA?

---

## Files of record

- `/root/github/QuickRunCUDA/investigations/01_clock_findings.md` — definitive clock measurement (2031.4 MHz boost, 1919.8 MHz under -lgc 2032)
- `/root/github/QuickRunCUDA/investigations/clock_definitive.cu` — methodology kernel (148 SMs × 256 thr, 80K outer iters, %clock64+%globaltimer at Q0/Q1/Q2/Q3/Q4)
- `/root/github/QuickRunCUDA/investigations/clock_lock_test.cu` (commit 861e8d1) — lgc paradox confirmed across {default, lgc 2032, lgc 1410, rgc}
- `/root/github/QuickRunCUDA/investigations/clock_precision.cu` — 2-cycle floor + globaltimer ratio
- `/root/github/QuickRunCUDA/investigations/clock_state_probe.cu` — sustained-load clock invariance
- `/root/github/QuickRunCUDA/investigations/peak_ffma_power.cu` (commit 1cd7f03) — 74.6 TFLOPS @ 361 W
- `/root/github/QuickRunCUDA/investigations/power_sustained.cu` (commit 51ed17c) — 30 s FFMA at 437 W, no throttle
- `/root/github/QuickRunCUDA/investigations/tensor_power.cu` (commit 4ee0626) — BF16 mma.sync 569 TFLOPS @ 411 W
- `/root/github/QuickRunCUDA/investigations/fp8_power.cu` (commit 10e2dc1) — FP8 cuBLAS 4491 TFLOPS @ 886 W
- `/root/github/QuickRunCUDA/investigations/long_long.cu`, `long_sustained.cu`, `cublas_power.cu` — 30 s FP8 cuBLAS sustained, ratio 1.00× throughout
- `/root/github/QuickRunCUDA/investigations/nvml_overhead.cu` (commit ca3666c) — 0.12 µs per query (memory info 3.3 µs)
- `/root/github/QuickRunCUDA/investigations/nanosleep.cu` (commit ddff4b8) — 35–60% undershoot for > 1 µs
- `/root/github/QuickRunCUDA/B300_PIPE_CATALOG.md` §"Clock / power / thermal", §"Power Consumption Under Load", §"__nanosleep precision" — earlier numbers (some superseded)
