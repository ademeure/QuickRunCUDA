# Power / Clock / Energy Inconsistency Log

Cross-source audit. Topic: power, clock, and energy across 18 power-related
files in `b300_clean/`. Originals NOT modified.

---

## A. Idle / floor power

| File | Idle floor | Notes |
|------|-----------:|-------|
| `16_power_clock.md` | 182-197 W | "Large die + HBM3E baseline" |
| `M2_ENERGY_LADDER.md` | 164.7 W (true idle), 165-167 W "alive 1 SM" | @ 1500 MHz |
| `POWER_DATA_DEPENDENCE_SUMMARY.md` | 150 W | @ 1005 MHz lock |
| `POWER_FREQUENCY_CURVE.md` | 144 (510), 167 (1500), 197 (1920) | Idle scales WITH clock (leakage) |
| `V10_DVS_CURVE.md` | 144 (510) → 198 (1920) | Same scaling pattern |
| `M11_PER_PIPE_ENERGY.md` | "165-170 W regardless of utilization" | Misses clock dependence |

**Inconsistency #1:** Idle floor is reported as 144 W (510 MHz) up to
197 W (1920 MHz) — a 50 W spread, but several files quote one number
without specifying clock. M11 specifically says "regardless of utilization"
which is true only at fixed clock.

**Resolution:** Always report idle WITH the clock state. Use 150 W at
1005 MHz, 167 W at 1500 MHz, 197 W at boost as the canonical points.

---

## B. TDP cap

| File | TDP value | Notes |
|------|----------:|-------|
| `16_power_clock.md` | 1100 W (min 200 / max 1100 / default 1100) | nvmlDeviceGetEnforcedPowerLimit |
| `POWER_DATA_DEPENDENCE_SUMMARY.md` | 1100 W | Consistent |
| `POPCOUNT_VS_CLOCK.md` | 1100 W; observed clip at 1092-1099 W | Bell flat-tops above |
| `B300_TRUE_REFERENCE.md` | "TDP 1100 W (sustained avg ceiling 1093 W; transient peaks to 1259 W)" | 1259 W transient claim |
| CLAUDE.md memory | 1100 W TDP | Consistent |
| (legacy / old catalog) | "700 W" | Hopper carry-over — RETIRED |

**Inconsistency #2:** B300_TRUE_REFERENCE quotes a transient peak of
**1259 W** (15 % above the supposed enforced cap of 1100 W). Either:
- The enforced limit (1100 W) is a sustained-average soft cap with
  millisecond-scale transients allowed, OR
- The 1259 value is a sample-aliasing artifact in NVML.

UNRESOLVED — needs corroboration with high-rate (kHz+) power probe.

---

## C. "Always boosts to 2032" vs stuck-at-1005

| File | Claim |
|------|-------|
| `16_power_clock.md` | "Default sustained boost = 2031.4 MHz; NEVER throttled in any tested workload" |
| `B300_TRUE_REFERENCE.md` | "Sustained 1920 MHz SM clock (boost is 2032 but rarely sustained)" |
| `POWER_FREQUENCY_CURVE.md` | Boost row labeled "2032 MHz" |
| memory `feedback_clock_stuck_no_lock.md` | **B300 can stick at 1005 MHz under load with NO explicit lock; `nvidia-smi -q` won't show it** |
| memory `feedback_clock_lock_works.md` | **`-lgc` IS honored 510-1500 MHz; "1942 floor" was background procs** |

**Inconsistency #3:** `16_power_clock` says the chip never throttles;
`B300_TRUE_REFERENCE` line 16 says boost "rarely sustained" — these are
direct contradictions. Memory note on stuck-at-1005 reconciles BOTH:
default boost IS 2032 in clean tests, but background processes (or
silent throttle conditions) can pin it to 1005 with no warning. Always
sample clock during long runs.

---

## D. -lgc 2032 paradox

| File | Reading | Confirmed? |
|------|---------|------------|
| `16_power_clock.md` | -lgc 2032 → 1919.8 MHz | YES (clock64) |
| `V10_DVS_CURVE.md` | 2032 row power = 1920 row power (419 W ≈ 419 W) | YES |
| `B300_TRUE_REFERENCE.md` | "lgc 2032 paradoxically pins to 1920" | YES |
| `POWER_FREQUENCY_CURVE.md` | uses -lgc CLK; 1800/2032 boost | Implicit ack |

**Resolution:** UNANIMOUS. NEVER use `-lgc 2032`; use `-rgc` for true boost.
Lock at 510-1500 MHz works correctly.

---

## E. tcgen05 power floor

| File | Min active multiplier power | Conditions |
|------|----------------------------:|------------|
| `POWER_FLOOR.md` | **287 W** at 148 SMs (A=B=0, mode 1800), 1 W/SM | 1005 MHz |
| `POWER_FINAL_MODEL.md` | 280-305 W "Tier B" baseline (NVFP4 / BF16 / FP8) | 1005 MHz |
| `BF16_PERBIT_POWER.md` | Tier A (B all-0) = 294 W, Tier B (B const ≠ 0) = 299 W, Tier C (Inf/NaN) = 308 W | 1005 MHz |
| `PER_SM_POWER_SCALING.md` | 1.0 W/SM const, 3.1 W/SM random, 2.1 W/SM data-dep | 1005 MHz |

**Consistency:** Numbers are mutually compatible (287 ≈ 294 ≈ 1×148+150; 299 ≈ 1×148+150).
Differences within ±5 W noise.

---

## F. Random-data tcgen05 max power

| File | Random BF16 power | Clock |
|------|------------------:|------:|
| `POWER_FINAL_MODEL.md` | 609 W (BF16), 642 W (FP8), 463 W (NVFP4) | 1005 MHz |
| `BF16_PERBIT_POWER.md` | 605-609 W | 1005 MHz |
| `PER_SM_POWER_SCALING.md` | 612 W | 1005 MHz |
| `POWER_FREQUENCY_CURVE.md` | 613 W (1005), 1009 W (1500), 1099 W (boost, capped) | sweep |

**Consistency:** ±5 W. POWER_FREQUENCY_CURVE adds the clock-sweep dimension.

---

## G. Data-dependence amplitude

| File | DRAM read swing (random vs zero) | Clock |
|------|---------------------------------:|------:|
| `POPCOUNT_3TIER.md` | 240 W (DRAM-8G d=16 vs d=0: 637 vs 397) | 1005 MHz |
| `POWER_DATA_DEPENDENCE_SUMMARY.md` | 240 W | 1005 MHz |
| `POPCOUNT_VS_CLOCK.md` | 367 W (1500 MHz, 921 vs 554) | 1500 MHz |
| `HBM_DATA_DEPENDENCE.md` | "<50 W" — UNDERESTIMATE | inferred, low-quality kernel |

**Inconsistency #4:** `HBM_DATA_DEPENDENCE.md` claims HBM data-dep is
"<50 W out of total 1100 W" with LOW confidence (poor kernel, only 20.4 GB/s).
The proper popcount sweep (4 later files) shows it is **240-367 W** — the
old number is wrong by 5-7×.

**Resolution:** SUPERSEDE `HBM_DATA_DEPENDENCE.md` with `POPCOUNT_3TIER.md`
+ `POPCOUNT_VS_CLOCK.md`. Original explicitly notes "LOW confidence on
exact magnitude; need proper DRAM benchmark" and predicts <5 % difference;
this prediction is wrong.

---

## H. Static / dynamic split

| File | Static | Dynamic |
|------|-------:|--------:|
| `M2_ENERGY_LADDER.md` | 0.05 W/SM static | 0.4 W/SM dynamic FFMA — 8-9× ratio |
| `M11_PER_PIPE_ENERGY.md` | 165-170 W static GPU; 0.7 W/SM dynamic FFMA | "Static is 30-60 % of total" |
| `PER_SM_POWER_SCALING.md` | 1.0 W/SM const tcgen05 | 3.1 W/SM random tcgen05 |

**Inconsistency #5 (mild):** M2 says 0.4 W/SM dynamic FFMA; M11 says
0.7 W/SM dynamic FFMA. Both at 1500 MHz. Different kernels (likely
single-chain vs 16-chain). Within ~2× — not a true contradiction, just
two operating points.

---

## I. Min-energy clock

| File | Workload | Min-energy clock |
|------|----------|------------------|
| `M9_ENERGY_PARETO.md` | Mixed ML | **1992 MHz boost** (3× lower than 510) |
| `M9_ENERGY_PARETO.md` | Pure FFMA | 510 MHz |
| `M9_ENERGY_PARETO.md` | Memory-bound | 800 MHz |
| `M11_PER_PIPE_ENERGY.md` | FFMA | "510 MHz min" (3.1 pJ/FFMA) — **CONSISTENT with M9** |
| `M11_PER_PIPE_ENERGY.md` | Memory | 800 MHz min — CONSISTENT |
| `V10_DVS_CURVE.md` | FFMA | 1500-1700 MHz (134 GFLOPS/W) — **CONFLICT with above** |
| memory note | ML inference | USE BOOST CLOCK (3× lower energy than 510) — matches M9 |

**Inconsistency #6:** V10 picks 1500-1700 as best FFMA TFLOPS/W; M9 picks
510 MHz. Both are claims about "FFMA" but DIFFERENT metrics:
- M9 = pJ per FFMA op (energy per work unit; 510 wins)
- V10 = GFLOPS / W instantaneous (efficiency; 1500-1700 wins)

These are not really contradictory — pJ/op accounts for static-power
amortization, GFLOPS/W does not. Both correct in their own framing.

**Resolution:** When asked for "energy-optimal clock" specify what is being
optimized: per-task energy (pick M9's number) vs instantaneous TFLOPS/W
(pick V10's). For real ML workloads, M9's mixed-workload boost-clock
recommendation wins (matches memory note).

---

## J. TFLOPS/W ladder

| File | FFMA peak | BF16 mma | FP8 cuBLAS |
|------|----------:|---------:|-----------:|
| `16_power_clock.md` | 0.21 (74.6 TF / 361 W) | 1.39 (569/411) | **5.07** (4491/886) |
| CLAUDE.md memory | 5 TFLOPS/W FP8 | — | matches |
| `M11_PER_PIPE_ENERGY.md` | 9.0 J/TFLOP = 0.111 TF/W | — | — |

**Inconsistency #7:** M11 says FFMA = 9 J/TFLOP = 0.111 TF/W; 16_power_clock
says 0.21 TF/W (74.6 TF / 361 W) — nearly 2× discrepancy. Likely M11's
number is at low-occupancy / different test configuration (its 359 W /
39.7 TFLOPS is the "FFMA-bound" entry, which is half of peak). Compatible
when accounting for ILP-dependent TFLOPS — but the labels are inconsistent.

---

## K. Bit-stride / chunk-level dedup

| File | Claim |
|------|-------|
| `L2_POPCOUNT_SWEEP.md` | "bit-stride NULL result" — refuted earlier hypothesis |
| `POWER_DATA_DEPENDENCE_SUMMARY.md` | "NULL RESULT — 3 % spread across all p values" |
| Old `L2_BITSTRIDE_SWEEP.md` (referenced) | The original hypothesis; results were null |

**Consistency:** Multiple files agree the chunk-level dedup hypothesis is
DEAD. Only per-cycle bit-flip count (popcount d) matters.

---

## L. Stress recipe

CLAUDE.md memory: "DRAM read d=16 + 1500 MHz = 1071 W stress recipe."

| File | Confirms? |
|------|-----------|
| `POPCOUNT_VS_CLOCK.md` | YES — DRAM-8G d=16 @ 1500 MHz = 921 W active + 150 W idle = 1071 W |
| `POWER_DATA_DEPENDENCE_SUMMARY.md` | YES — explicit recipe quoted |
| `POPCOUNT_3TIER.md` | At 1005 MHz only; can't reach 1071 |

**Consistency:** Recipe verified. 1500 MHz is the highest clock that gives
an UNCLIPPED bell curve (1800 MHz hits TDP and clips d=8..28).

---

## Cross-source consistency status

| Topic | Status |
|-------|--------|
| TDP = 1100 W | CONSISTENT (modulo 1259 W transient question) |
| Idle scales with clock | INCONSISTENT presentation across files (some quote single number) |
| Default boost = 2032 MHz | MOSTLY consistent; B300_TRUE_REFERENCE caveat about "rarely sustained" needs flag for stuck-at-1005 |
| `-lgc 2032` paradox | UNANIMOUS |
| tcgen05 floor 287-299 W | CONSISTENT |
| Random tcgen05 ~610 W | CONSISTENT |
| DRAM data-dep magnitude | INCONSISTENT (`HBM_DATA_DEPENDENCE.md` is wrong; 4 newer files agree at ~240-367 W) |
| FFMA pJ/op (DVS) | CONSISTENT across M11 / V10 / V5 |
| Min-energy clock | METRIC-DEPENDENT (per-task energy vs TFLOPS/W) — not a real contradiction |
| Stress recipe d=16 / 1500 MHz / 1071 W | CONSISTENT |
| Static % | CONSISTENT (~30-60 %) but specific W/SM differs by config |
| Bit-stride NULL | CONSISTENT |

## Recommended retirement / supersession

1. **`HBM_DATA_DEPENDENCE.md`** → SUPERSEDED by `POPCOUNT_3TIER.md`
   + `POPCOUNT_VS_CLOCK.md`. Add forwarding header.
2. **Any remaining "B300 TDP = 700 W" reference** → REPLACE with 1100 W.
3. **Any unqualified "boost = 2032 MHz" claim** → ADD caveat about
   stuck-at-1005 silent failure mode.
4. **M11's "static 165-170 W regardless of utilization"** → ADD clock
   qualification (varies 144-198 W with clock).

## Items the user specifically flagged

| Memory note | Status in catalog |
|-------------|-------------------|
| TRUE perf at 2032 MHz: 40 tok/s 70B, 345 tok/s 8B; clock-lock was 2.35× bottleneck | NOT in any 16_power_clock-related file; lives only in CLAUDE.md memory. Worth adding to corrected reference. |
| PCIe Gen 6 x16, 1100 W TDP, 5 TFLOPS/W FP8 | All 3 in B300_TRUE_REFERENCE; FP8 number = 5.07 in 16_power_clock; PCIe Gen 6 in 13_pcie_system. |
| ML inference USE BOOST CLOCK (3× lower energy than 510) | M9 says 3.08× — MATCHES |
| Clock-lock works 510-1500 (1942 floor was background procs) | Implicit in V10 / POWER_FREQUENCY_CURVE; not explicit in 16_power_clock |
| Stuck-at-1005 with no lock | NOT in 16_power_clock — should be added (B300_TRUE_REFERENCE has hint at line 378-379 mentioning recovery via -rgc) |
| DRAM d=16 + 1500 MHz = 1071 W | EXPLICIT in 2 files |
| Memory power follows popcount bell, peak at d=16 random | UNANIMOUS in 4 files |
| DVS V² scaling | EXPLICIT in V10, M11; implicit in POWER_FREQUENCY_CURVE |
