# 16_power_clock — CORRECTED

Cross-source synthesis. Originals NOT modified. Inconsistencies catalogued in
`POWER_INCONSISTENCY_LOG.md`.

GPU: B300 SXM6 AC, sm_103a, 148 SMs. **TDP enforced limit = 1100 W**
(`nvmlDeviceGetEnforcedPowerLimit`). Idle baseline 150-200 W (varies by
clock state — see floor table). Min power limit = 200 W.

---

## 1. Clock state — corrected

| Clock state | Reported clock | Actual clock | Source |
|-------------|---------------:|-------------:|--------|
| True idle | 120 MHz | 120 MHz | 16_power_clock |
| Default boost (no lock, sustained FFMA) | 2032 MHz | **2031.4 MHz** | clock64/globaltimer ratio |
| `nvidia-smi -lgc 2032` | 2032 | **1919.8 MHz** (-5.5%) | replicated 4× |
| `nvidia-smi -lgc 1410` | 1410 | 1410 MHz (correct) | one-shot |
| `nvidia-smi -lgc 510` | 510 | 510 MHz | V10 DVS curve |
| **Stuck-without-lock state** (no -lgc, but pinned to 1005 MHz under load) | varies | **1005 MHz** | memory: `feedback_clock_stuck_no_lock.md` — recover with `-rgc` |

**Counter to old "always boosts to 2032" claim:** B300 CAN pin to 1005 MHz
with NO explicit lock when other procs thrash the GPU; sample clock during
EVERY long run.

`nvidia-smi -lgc 2032` paradox replicated independently in
`16_power_clock.md`, `V10_DVS_CURVE.md`, `B300_TRUE_REFERENCE.md`. Use
`-rgc` to release any lock.

---

## 2. Power floor + ceiling at each clock

(All values in W, sustained ≥3 s, TDP limit = 1100 W)

| Clock (MHz) | Idle floor | Active min (tcgen05 A=B=0) | FFMA active | DRAM-read random d=16 | TDP-cap reached? |
|------------:|----------:|---------------------------:|------------:|----------------------:|:----------------:|
| 510  | 144 | ~155 (extrapolated) | 178 (FFMA Δ34) | ~553 | NO |
| 800  | 147 | — | 202 (Δ54) | 631 | NO |
| 1005 | 150-152 | **287** (148 SMs, 1 W/SM) | 225 (Δ73) / 613 random BF16 GEMM | **787** | NO |
| 1300 | 158 | — | 254 (Δ96) | 942 | NO |
| 1500 | 167 | — | 300 (Δ133) / 1009 random BF16 | **1071** | **APPROACHED** |
| 1700 | 175 | — | 339 | — | YES (random BW) |
| 1800 | — | — | — | **1092** (clipped) | **YES (clipped)** |
| 1920 (=`-lgc 2032`) | 198 | — | 419 (Δ222) / 1099 random BF16 | TDP-cap | YES |
| 2032 (true boost) | 198 | — | 361 (peak ILP=24) / 437 (low-occ) | — | YES (under cuBLAS BF16 962W) |

Notes:
- **DRAM read d=16 random + 1500 MHz = 1071 W** — sustained worst-case
  thermal stress recipe (memory: confirmed; from POPCOUNT_VS_CLOCK).
- **DRAM read d=8..28 + 1800 MHz = 1100 W TDP wall** — bell flat-topped.
- FFMA peak (boost, ILP=24, 256 thr) = 361 W, surprisingly LESS than
  low-occupancy FFMA (437 W).
- Random-data BF16 cuBLAS = 962 W sustained with no throttle (87% TDP).

---

## 3. Energy / op table per pipe

(@ 1500 MHz lock unless noted, V² × f scaling for other clocks)

| Op | Energy/op | Source | Confidence |
|----|-----------|--------|------------|
| FFMA (with .reuse, broadcast operand) | **2.2 pJ/FLOP = 4.4 pJ/FFMA** | M2 H1 | HIGH |
| FFMA (no .reuse, 3 unique RF reads) | 6.5 pJ/FFMA (1.49× more) | M2 H9 | HIGH |
| RF read (incremental) | 0.3 pJ/read | M2 H9 | HIGH |
| IMAD chain | 6.5 nJ / 1M ops | M2 H2 | HIGH |
| LOP3 chain | 3.85 nJ / 1M ops | M2 H2 | HIGH |
| MUFU rsqrt.ftz | similar to LOP3 | M2 H3 | HIGH |
| MUFU sin (no .ftz) | 1.5× MUFU rsqrt | M2 H3 | HIGH |
| LDG (cold HBM) | **96.6 nJ / 1M ops** (15× IMAD) | M2 H4 | HIGH |
| LDG.ca (L1 hit) | ~5 pJ | M11 | MED |
| LDG.cg (bypass L1) | ~10 pJ | M11 | MED |
| LDS u32 | ~5 pJ | M2 H8 derived | MED |
| LDS.128 vec load | 56 W (2.5× scalar) | M2 H8 | HIGH |
| STS.128 vec store | 74 W (2.4× scalar) | M2 H8 | HIGH |
| HMMA (BF16 mma.sync, output) | ~50 pJ/output (12× FFMA) | M11 | MED |
| Branch (predictable) | 14.8 pJ (3.36× FFMA) | M2 H5 | HIGH |
| Branch (divergent half-warp) | 18.5 pJ (4.20× FFMA) | M2 H5 | HIGH |
| __syncthreads | ~30 pJ | M11 | MED |
| cluster.barrier | ~390 pJ | M11 | MED |
| L2 read (d=16 random) | 25.5 nJ/byte | POPCOUNT_WRITES | HIGH |
| L2 write (d=16) | 62.2 nJ/byte | POPCOUNT_WRITES | HIGH |
| DRAM read (d=16) | 86.1 nJ/byte | POPCOUNT_WRITES | HIGH |
| DRAM write | 115.7 nJ/byte | POPCOUNT_WRITES | HIGH |

**FFMA pJ/op vs clock (V² × f scaling per V5 D4 / M11):**

| Clock (MHz) | pJ/FFMA |
|------------:|--------:|
| 510  | 3.1 |
| 1005 | 4.9 |
| 1500 | 6.8 |
| 1920 | 9.96 |

**Memory pJ/byte vs clock — different sweet spot than FFMA:**

| Clock (MHz) | pJ/byte |
|------------:|--------:|
| 510  | 12.06 |
| **800** | **11.81 ← min** |
| 1005 | 12.49 |
| 1500 | 14.92 |
| 1920 | 18.60 |

---

## 4. Min-energy clock — workload-DEPENDENT

| Workload | Min-energy clock | Rationale |
|----------|-----------------:|-----------|
| Pure FFMA-bound | **510 MHz** (5.76 pJ/FFMA) | Idle dominates marginal cost |
| Pure memory-bound | **800 MHz** (11.81 pJ/byte) | L2 latency hides clock advantage |
| **Realistic mixed ML inference** | **1992 MHz boost** | Wall-clock dominates static-power amortization. **3× lower energy** than 510 MHz |

**ML inference USE BOOST CLOCK** — this overrides naive "lower clock = lower
energy" intuition. DVFS down-clocking ML inference INCREASES total energy.

---

## 5. Data-dependence summary

Active power on B300 is **dominated by toggle-energy on whatever bus is
in flight**. Per-cycle bit-flip count, NOT bit-set count or chunk-level
redundancy, is the lever.

**At 1005 MHz, 64 MB / 8 GB working sets, .cg loads** (active W above 150 W idle):

| Subsystem | d=0 (zeros) | d=16 (random peak) | d=32 (ones) | Range (peak swing) |
|-----------|------------:|-------------------:|------------:|-------------------:|
| L1 reads | 70 | **107** | 82 | 35 % |
| L2 reads | 223 | **405** | 245 | 45 % |
| DRAM-1G reads | 369 | **604** | 411 | 39 % |
| DRAM-8G reads | 397 | **637** | 441 | 38 % |
| L2 writes | 140 | **235** | 159 | 41 % |
| DRAM-8G writes | 264 | **405** | 288 | 35 % |
| FFMA compute (8-ILP self-chain) | 40 | **103** | — | 60 % |
| IADD3 compute (8-ILP self-chain) | — | **103** | — | 95 % |

**Bell curve peaks at d=16 (50% bit density) at every tier.** Symmetric
around d=16. Going to all-zeros saves **184 W (33%) on L2 reads** and
**240 W (38%) on DRAM reads** at 1005 MHz; saves more in absolute terms
at higher clocks (until TDP cap clips).

**Key model (toggle-energy, per tier):**
```
P_active(d) ≈ P_baseline + α × [2·d·(32-d) / 31] + β × d
α (toggle peak): L1 ~80, L2 ~325, DRAM ~360, FFMA ~120
β (per-bit static): L1 ~0.4, L2 ~0.6, DRAM ~1.4, FFMA ~0.5
```

**Constant-pattern (no inter-dword toggle) gives only 18 W spread across
popcount 0..32 at L2** — proves inter-dword toggling, not popcount per se,
is the dominant component.

**Asymmetry:** d=32 (all-ones) is 11-45 W higher than d=0 (all-zeros);
gap grows with cache distance. Likely HBM3E PHY DBI (active-low termination).

**Bit-stride / chunk-level dedup: NULL RESULT** (3% spread). DRAM/L2
signaling does NOT exploit chunk-level repetition.

**Sparsity > 10% gives measurable savings; below 10% no savings**;
granularity (byte vs 128-byte chunks) barely matters.

---

## 6. tcgen05.mma data-dependence (BF16 m128n128k16, 148 SMs, 1005 MHz)

Random vs structured B-operand:

| Pattern | Power (W) | Notes |
|---------|----------:|-------|
| Idle GPU | 150 | baseline |
| A=0 AND B=0 (mode 1800) | **287** | absolute floor for active multiplier |
| A=0, B=rand | 491 | A broadcast contributes little |
| A=rand, B=0 (mode 300) | 296 | B=0 fully gates multiplier |
| A=const +1.0, B=const +1.0 (Tier B) | **299** | static baseline |
| Inf/NaN constant (Tier C) | 308 | +9 W detector overhead |
| Random A & B (full random) | **609** | +310 W data-dependent cost |

**A-operand is ~FREE; B-operand carries all data-dependent power.**
A K-vary 16-unique adds only +2W; B K-vary 16-unique adds +48W (24× ratio).
This matches "A = broadcast, B = distributed across 32 MAC units".

Per-SM scaling: **3.1 W/SM random; 1.0 W/SM constant; 2.1 W/SM data-dep
delta**, linear up to 148 SMs.

**Per-bit decomposition (BF16):**
- Sign-bit forced 0 (ReLU): -56 W (18% of random penalty)
- Each exp bit: ~-30 W
- Each mantissa bit: ~-19 W
- Force-0/force-1 asymmetry on exp bits: +17 W (subnormal handling
  cost when random in subnormal range; constant subnormal is free)

**Structured-B optimizations** (composable):
1. Sub-tile dedup: keep ≤16 unique values per 16-N group → free
2. K-row sort (consecutive identical) → ~5W/transition saved (BF16)
3. disable_lane: 2.4 W/disabled column

Best practical low-power BF16 GEMM: 254 W vs 610 W random = **58 % reduction**.

---

## 7. Power-frequency (DVS) curve

CMOS V² × f scaling above ~1500 MHz. Idle scales too (leakage at higher V):

- 510 MHz idle: 144 W; 1920 MHz idle: 198 W (+37%).
- FFMA Δ-power: sublinear < 1005 MHz; linear 1005-1500; superlinear
  > 1700 MHz.
- Best FFMA GFLOPS/W: **134 at 1500-1700 MHz** (tied), drops to 123 at boost.
- DRAM read random saturates at TDP cap above 1500 MHz; **1500 MHz is the
  highest clock that gives an unclipped DRAM popcount bell curve**.

---

## 8. nanosleep.u32 (still standing — independent of clock)

| Request (ns) | Actual (ns) | Notes |
|-------------:|------------:|-------|
| 0-2 | 32 | Quantum floor (=globaltimer 32 ns) |
| 100 | 113 | +13% |
| 1000 | 620 | **-38%** |
| 5000 | 3200 | -36% |
| 10000 | 6500 | -35% |
| 100000 | 40000 | **-60%** |

Use globaltimer + busy-wait for accurate delays > 1 µs.

---

## RETRACTIONS

1. **"B300 TDP = 700 W"** — WRONG. Default enforced limit = 1100 W
   (Hopper carry-over).
2. **"B300 sustained boost = 1920 MHz"** — WRONG (early entries). Default
   boost = 2031.4 MHz. The 1920 came from `-lgc 2032`-locked benchmarks.
3. **"B300 always boosts to 2032 MHz"** — WRONG. Can stick at 1005 MHz
   under load with NO explicit lock; verify clock during long runs.
4. **"Apparent 1942 MHz floor"** — WRONG. Was leftover background procs
   thrashing the GPU; clock-lock works correctly 510-1500 MHz.
5. **"FFMA TFLOPS peak = 153.93"** — WRONG (used 256 cores/SM; B300 has
   128). True FP32 peak = 76.96 TFLOPS theoretical, 74.6 TFLOPS measured
   (97 %).
6. **"B300 throttles to 53 % of peak under sustained FP8"** — WRONG;
   measurement artifact (no warmup). True sustained FP8 GEMM is FLAT at
   4491 TFLOPS for 30+ s.
7. **"B300 TDP not approached under any workload (~339 W max)"** — WRONG;
   stale (FFMA-only). Tensor + cuBLAS hit 411-962 W; sustained BF16 GEMM
   = 962 W (87 % TDP). Random-data DRAM-read bell curves do hit 1071-1100 W.
8. **"FP16/BF16 packed FMA gives 2× FP32 throughput"** — WRONG outside
   tensor cores.
9. **"nanosleep precise to ±5%"** — WRONG; -35 to -60 % undershoot above 1 µs.
10. **"Subnormal handling intrinsically penalizes"** — REFINED: it's only
    *random within subnormal range* that penalizes (+20 W). Constant
    subnormal value sits at Tier B (no penalty).
11. **"Magnitude redirection drives exp-bit power"** — REFINED to subnormal
    handling cost in BF16_PERBIT_POWER.
12. **"Bit-stride / chunk-level dedup saves power"** — NULL RESULT;
    refuted (3 % spread across all p values).

---

## UNRESOLVED

1. **Multi-minute / hour-scale sustained load behavior.** All tests
   12-60 s. Whether B300 throttles under genuinely-long pure-compute
   load (e.g. 1 hour of FP8 cuBLAS at 886 W) — not tested.
2. **Why does FFMA non-peak draw MORE power (437 W) than FFMA peak
   (361 W)?** Hypothesis: idle SMSP leakage + low ILP wastes lanes.
   Needs ncu correlation.
3. **Multi-GPU TDP coupling** — does chassis power cap kick in when both
   B300s draw 962 W? Untested.
4. **`nanosleep.u32` undershoot mechanism** — driver vs hardware.
5. **`-lgc 1920` test** — only 2032/1410/unlocked tested. Does -lgc 1920
   give a true 1920?
6. **Per-DRAM-channel `dram__bytes_*.per_dram` ncu** — confirm even
   distribution across 6 HBM3E stacks at high clock (currently aggregate).
7. **2-CTA cluster_group::2 tcgen05 dedup-cache sharing** — LOW confidence
   in POWER_FINAL_MODEL.
8. **bit-14 (exp MSB) anomaly mechanism** — HIGH that it's real,
   LOW on the precise mechanism (more than just subnormal population
   statistics can explain the 17 W gap).
9. **TDP-ceiling vs transient peak.** B300_TRUE_REFERENCE cites
   "sustained avg ceiling 1093 W; transient peaks to 1259 W"
   (commit 862014c, agent-verified) — not corroborated by 16_power_clock
   which says 1100 W is the cap. Either NVML enforced limit (1100) is a
   software cap that the chip can BRIEFLY exceed, or the 1259 value is
   measurement transient. Needs reconciliation.
10. **Real production weight tensors** — predicted -120 to -180 W vs
    synthetic d=16 not directly verified.

---

## Files of record (all originals, NOT MODIFIED)

- `16_power_clock.md` — primary catalog
- `POWER_FINAL_MODEL.md` — tcgen05 4-component model
- `POWER_FLOOR.md` — 287 W absolute multiplier floor
- `POWER_FREQUENCY_CURVE.md` — random vs optimized clock sweep
- `POWER_DATA_DEPENDENCE_SUMMARY.md` — top-level toggle-energy synthesis
- `POPCOUNT_3TIER.md` — bell-curve at L1/L2/DRAM
- `POPCOUNT_VS_CLOCK.md` — clock-scaling story w/ TDP wall
- `POPCOUNT_WRITES.md` — write power vs popcount
- `L2_POPCOUNT_SWEEP.md` — original L2 popcount finding
- `HBM_DATA_DEPENDENCE.md` — short, pre-sweep version (now superseded)
- `V10_DVS_CURVE.md` — full clock × FFMA power curve, GFLOPS/W
- `M2_ENERGY_LADDER.md` — pipe-by-pipe energy table
- `M11_PER_PIPE_ENERGY.md` — DVS energy table per op
- `M9_ENERGY_PARETO.md` — workload-dependent min-energy clock
- `DISABLE_LANE_POWER.md` — 2.4 W/column scaling
- `PER_SM_POWER_SCALING.md` — linear in block count up to 148
- `BF16_PERBIT_POWER.md` — per-bit decomposition + 3-tier const model
- `B300_TRUE_REFERENCE.md` — meta reference (cross-cuts)
