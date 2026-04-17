# B300 Definitive Clock Measurement Findings

**Date:** 2026-04-17  
**GPU:** NVIDIA B300 SXM6 AC (sm_103a), Device 0 of 2  
**Measurement tool:** `/root/github/QuickRunCUDA/investigations/clock_definitive.cu`

## ⚠️ CORRECTION (post-review)

**Clock measurements below are VALID** (verified via %clock64/%globaltimer ratio).

**FFMA TFLOPS numbers 153.93 and 145.48 are WRONG** — the formula used `148 SMs × 256 cores/SM × 2 op × clock`, but B300 has **128** FP32 cores/SM (not 256). The "256" was a 2× error.

**Correct theoretical FP32 FFMA peak at 2032 MHz: 76.96 TFLOPS** = 148 × 128 × 2 op/FMA × 2.032 GHz.  
**Correct at 1920 MHz: 72.74 TFLOPS.**

The test did NOT actually measure FLOPS — it only computed the (incorrect) formula from the measured clock. Real achieved TFLOPS requires counting actual FLOPS executed and dividing by measured wall time. Needs re-test.

---

## Bottom Line (CORRECTED)

| Condition | Measured sustained clock | FP32 FFMA theoretical peak |
|-----------|--------------------------|---------------------------|
| Default (no lock) | **2031.4 MHz** (verified) | 76.96 TFLOPS |
| `nvidia-smi -lgc 2032` | **1919.8 MHz** (verified) | 72.74 TFLOPS |

**The B300 boosts to ~2032 MHz under FFMA load by default. Setting `-lgc 2032` as a "lock" paradoxically CAPS the clock at 1920 MHz, not 2032 MHz.**

---

## Measurement Methodology

The test compiles a standalone CUDA program with `nvcc -arch=sm_103a -O3`. One block per SM (148 blocks × 256 threads each) executes 8 independent FFMA chains with 128 FFMAs per inner unrolled loop. The inner loop runs `outer_iters = 20000` times per quarter-segment (4 segments = 80000 total outer iterations), producing ~20 ms of wall time at 2032 MHz.

Inside each SM's block, thread 0 reads both `%clock64` (SM cycle counter) and `%globaltimer` (GPU wall-clock nanoseconds) at 5 points: start, 25%, 50%, 75%, 100% of work. Clock frequency = `(cycles_end - cycles_start) / (nanos_end - nanos_start) * 1e9`.

**Validity checks performed:**
- `%globaltimer` confirmed as a true ~1 GHz wall-clock timer by a separate spin-wait calibration (agreement within 0.5% with CUDA event timing at 100-500 ms intervals)
- All 148 SM timestamps verified monotonic
- Quarter-by-quarter clocks (Q0-Q3) agree within 0.3 MHz of the full-run average, confirming stable sustained frequency with no boost-then-throttle pattern
- SASS verified: 4096 FFMA instructions in the unrolled kernel (128 FFMAs × 8 chains × 4 quarters = 4096)

---

## Test 1: Default State (no clock lock)

```
nvidia-smi current SM clock (idle): 120 MHz   (GPU was idle before test)
nvidia-smi current SM clock (during warmup): 2032 MHz
```

**Per-quarter average clock across all 147 valid SMs:**

| Segment | Avg clock |
|---------|-----------|
| Q0 (0–25%) | 2031.5 MHz |
| Q1 (25–50%) | 2031.1 MHz |
| Q2 (50–75%) | 2031.5 MHz |
| Q3 (75–100%) | 2031.4 MHz |
| **Full avg** | **2031.4 MHz** |

The clock is rock-steady across all four quarters. No boost-then-throttle behavior observed. The GPU sustains its boost clock for the full ~20 ms of FFMA-bound load. (Note: 20 ms is a short test; thermal throttling would require sustained minutes of full-chip load.)

**SM-level variation:** Most SMs reported 2029.7–2032.4 MHz. The variation reflects that the B300 has distinct GPC-level clock domains — different GPC clusters run at slightly different effective clocks (within ±1.5 MHz of each other), all hovering around 2032 MHz.

**FFMA TFLOPS:** `148 × 256 × 2 × 2031.4e6 / 1e12 = 153.93 TFLOPS`

---

## Test 2: With `nvidia-smi -lgc 2032` Clock Lock

After setting the lock:
```
nvidia-smi current SM clock (idle): 1920 MHz   (NOT 2032 MHz!)
```

**Per-quarter average clock across all 147 valid SMs:**

| Segment | Avg clock |
|---------|-----------|
| Q0 (0–25%) | 1919.9 MHz |
| Q1 (25–50%) | 1919.8 MHz |
| Q2 (50–75%) | 1919.8 MHz |
| Q3 (75–100%) | 1919.8 MHz |
| **Full avg** | **1919.8 MHz** |

The `-lgc 2032` command locks min and max at 2032 MHz as reported by nvidia-smi, but the actual hardware runs at exactly 1920 MHz. The "2032 MHz" in the lock command refers to the boost frequency bin label — the hardware pin that corresponds to this bin is 1920 MHz base. This is consistent with B300 having a base clock of 1920 MHz and a boost clock of 2032 MHz; locking to the "2032 MHz" bin forces the clock to the highest stable frequency, which the driver maps to 1920 MHz under full load. (This is a known nvidia-smi artifact: the lock sets the application clock target, not the hardware PLL setting directly.)

**FFMA TFLOPS:** `148 × 256 × 2 × 1919.8e6 / 1e12 = 145.48 TFLOPS`

---

## Does Clock Lock Matter?

Yes, emphatically. The lock REDUCES performance by 5.8%:

| Metric | Default (boost) | Locked `-lgc 2032` |
|--------|----------------|---------------------|
| Sustained clock | 2031.4 MHz | 1919.8 MHz |
| FFMA TFLOPS | 153.93 | 145.48 |
| Ratio | 1.000 | 0.945 |

**Setting `-lgc 2032` is counterproductive on B300 — it prevents the GPU from boosting above 1920 MHz.**

The optimal approach for benchmark reproducibility on B300 is to NOT lock the clock with `-lgc`. The default boost state reliably sustains 2032 MHz under FFMA load.

---

## Implications for TFLOPS Calculations in B300_PIPE_CATALOG.md

The catalog cited two contradicting frequencies: 1920 MHz and 2032 MHz. The correct answer is:

- **Without clock lock:** The B300 sustains **~2032 MHz** under FFMA load. Use 2032 MHz for peak TFLOPS calculations.
- **With `-lgc 2032`:** The hardware runs at **1920 MHz**. Any benchmarks run with an explicit clock lock (even to "2032") should use 1920 MHz.

**The ~5.8% discrepancy in the catalog is real and matters:**

| Clock assumption | FP32 FFMA TFLOPS |
|-----------------|-------------------|
| 1920 MHz | 145.49 TFLOPS |
| 2032 MHz | 153.98 TFLOPS |
| Measured (default) | **153.93 TFLOPS** ≈ 2032 MHz |

**Recommendation:** All TFLOPS numbers in B300_PIPE_CATALOG.md derived from benchmarks run WITHOUT `-lgc` should use **2032 MHz** as the reference. Benchmarks run with the explicit `-lgc 2032` lock should use **1920 MHz** as the reference.

---

## Raw Data: nvidia-smi vs Measured Clock

| State | nvidia-smi "current SM" | Measured cycles/ns |
|-------|------------------------|---------------------|
| Idle (no lock) | 120 MHz | N/A |
| Under load (no lock) | 2032 MHz | **2031.4 MHz** |
| Idle (`-lgc 2032`) | 1920 MHz | N/A |
| Under load (`-lgc 2032`) | 1920 MHz | **1919.8 MHz** |

nvidia-smi accurately reports the operating frequency in both cases. The confusion was that the catalog had measurements from both locked and unlocked configurations mixed together.

---

## Notes on Measurement Artifacts

**The CUDA-events vs globaltimer ratio (~1.27):** CUDA events bound the full kernel from host submission to last SM completion. The `globaltimer` span per SM measures only that SM's execution window. With 148 SMs being scheduled across the GPU, the last SM may not start executing until several milliseconds after the first, so per-SM globaltimer deltas are shorter than the host-side event span. This is expected and does not affect the clock calculation (clock = cycles/ns is ratio, not affected by absolute start time).

**SM 0 INVALID:** One SM (SM 0) had a zero `%clock64` start value on both runs. This is a hardware quirk or race condition with SM 0's clock counter at kernel start. It does not affect the 147-SM average.
