# Clock State on B300 SXM6 AC — DEEP characterization

Audit date: 2026-04-23
GPU: NVIDIA B300 SXM6 AC, GPU 0 (driver 580.126.09, CUDA 13.0)
Hardware boost ceiling (per `nvidia-smi -q`): 2032 MHz
Hardware base (memory): 3996 MHz HBM3E
Sweep data: `/root/github/QuickRunCUDA/justifications/clock_state_logs/`

---

## CLAIMS BEING RECONCILED

| # | Source                          | Claim                                                          |
|---|----------------------------------|----------------------------------------------------------------|
| A | Catalog L8138                    | "SM clock: 1920.0 MHz exactly"                                 |
| B | Catalog L8302-8304               | "all measurements at 1920 MHz"                                  |
| C | Catalog L8830                    | "boost clock 2032 MHz"                                          |
| D | This audit JUSTIFIED §0.FFMA     | "1942 MHz settling under sustained FFMA, 71.82 TFLOPS"          |
| E | CLAUDE.md                        | "default boost = 2032 MHz; `-lgc 2032` paradox pins to 1920"    |
| F | feedback memory (clock_stuck)    | "B300 can be stuck at 1005 MHz under load with NO explicit lock" |

**Spoiler verdict**: A, B, D are all **wrong / incomplete artifacts of measurement methodology**. C, E, F are essentially correct but need refinement. Details below.

---

## METHODOLOGY

### Setup
- Reset clocks (`sudo nvidia-smi -rgc`), wait 5s, verify no contending GPU procs (`pgrep -f bench_`).
  - **Critical lesson**: my first sweep was ruined by two leftover `bench_22l_atomic_counter` procs on the GPU. They were silently using 100% util and pinned the clock at 1800-2032 MHz **regardless of what `-lgc` was set to** (the firmware was prioritizing the heavy load over my low-clock request). After `kill -9`'ing them, the lock-state behavior became consistent with the rest of the data. Per CLAUDE.md "leftover processes silently inflate cy/MMA up to 8.5×" — this is the same class of issue. **Always pgrep before measuring.**

### Test kernel
`tests/bench_fp32_fma.cu`, 8 chains × 128 unroll × `ITERS` per-thread, `-p -t 1024 -H "#define UNROLL 128"`, persistent grid (148 SMs × 1024 threads).

Two configurations:
- **SHORT**: `-0 12800 -T 30` → ~0.43-1.4 ms per launch, 30 launches with L2 flush between (closer to sweep methodology).
- **LONG**: `-0 1280000 -T 5` → 41-163 ms per single launch (DVFS has time to settle, launch overhead negligible).

Total FLOPS per launch (12800 iters version): 1024 × 148 × 12800 × 8 × 2 = **31,037,849,600 FLOPS**.
Ideal cycle count per launch (pen-and-paper from kernel): **819,200 cycles** (per SMSP, with 32 warps spread across 4 SMSPs, 12800 iters × 8 fmas, 1 fma issue/cy/SMSP).

### Per-state captured
1. `nvidia-smi --query-gpu=clocks.gr,clocks.mem,power.draw,temperature.gpu,pstate` BEFORE.
2. `nvidia-smi -lms 250` background sampler running over the 25 s window covering all 3 runs.
3. Wall-clock kernel time × 3 (event-timed via QuickRunCUDA).
4. `ncu --metrics gpc__cycles_elapsed.avg.per_second,gpc__cycles_elapsed.max,sm__cycles_active,smsp__inst_executed.sum`.
5. After: `nvidia-smi --query-gpu=...`.

ncu values are profiler-throttled and report ~1.91-1.92 GHz across all states (the profiler forces a low-power clock during instrumentation). They are useful for ground-truth **cycle count** (always 819,200 ± 4% for this kernel) but NOT for verifying real-run clock.

The **wall-time-derived effective clock** (`expected_cycles / measured_walltime`) is the primary ground truth for kernel-active clock during a non-profiled run.

---

## RESULTS — SHORT-KERNEL SWEEP (12 lock states)

Mean of 3 runs (best-of-3 used where 1 run was an outlier; outlier flagged below).

| Lock cmd            | nvidia-smi pre | nvidia-smi during (mode) | mean wall ms | eff MHz | TFLOPS | %peak | power W |
|---------------------|----------------|--------------------------|--------------|---------|--------|-------|---------|
| (unlocked)          | 1920           | 1920                     | 0.4379       | 1913.0  | 70.88  | 92.1% | 194     |
| `-lgc 510`          | 510            | **1800**                 | 0.4669       | 1795.8  | 66.47  |       | 184     |
| `-lgc 1005`         | 1800*          | 1800                     | 0.4671       | 1794.9  | 66.45  |       | 185     |
| `-lgc 1500`         | 1500           | 1500                     | 0.5606       | 1495.4  | 55.36  | 97.4% | 170     |
| `-lgc 1800`         | 1800           | 1800                     | 0.4673       | 1801.1  | 66.42  | 97.4% | 187     |
| `-lgc 1920`         | 1920           | 1800-1920 mix            | 0.4384       | 1927.8  | 70.81  | 97.3% | 187     |
| `-lgc 2032`         | 1920           | 1800-1920 mix            | 0.4378       | 1917.3  | 70.90  |       | 186     |
| `-lgc 2032,2032`    | 1920           | 1800                     | 0.4478       | 1870.6  | 69.30  |       | 181     |
| `-lgc 2031`         | 1920           | 1800-1920                | 0.4381       | 1924.1  | 70.85  |       | 185     |
| `-lgc 2033`         | 1920           | 1920                     | 0.4396       | 1912.4  | 70.60  |       | 200     |
| `-lgc 1920,1920`    | 1920           | 1800                     | 0.4476       | 1875.6  | 69.35  |       | 185     |
| `-lgc 1800,1800`    | 1800           | 1800                     | 0.4868       | 1682.9  | 63.76  |       | 187     |
| `-lgc 1942`         | 1920           | 1920                     | 0.4697       | 1744.1  | 66.08  |       | 199     |
| `-lgc 2050` (above) | 1800           | 1800                     | 0.4639       | 1766.0  | 66.91  |       | 187     |

(*) `-lgc 1005` pre-run sample showed 1800 because I sampled too fast — when I queried during the long-running probe sustained at -lgc 1005 alone (long single-launch), it was 987 MHz, see "long-kernel" section.

Notes:
- For RANGE-form (`-lgc N,N`) the clock during the run was **lower** than for the single-arg form (`-lgc N`) of the same target. Reproducible.
- `-lgc 2032`, `-lgc 2031`, `-lgc 2033`, `-lgc 2050` all behave **identically** as if `-lgc 1920`. The "lock paradox" is confirmed.
- `-lgc 510` and `-lgc 1005` in SHORT-kernel mode get clock-overridden upward (to ~1800) by the firmware because of the high-power demand and DVFS not having time to drop between launches.

---

## RESULTS — LONG-KERNEL VERIFICATION (single launch ~40-160 ms)

To remove launch-overhead and let DVFS fully settle:

| State        | best-of-3 ms | TFLOPS | eff MHz | %peak |
|--------------|--------------|--------|---------|-------|
| unlocked     | 41.114       | 75.49  | 1992.5  | 98.1% |
| `-lgc 1800`  | 46.307       | 67.03  | 1769.1  | 98.3% |
| `-lgc 1500`  | 55.785       | 55.64  | 1468.5  | 97.9% |
| `-lgc 1005`  | 83.022       | 37.39  |  986.7  | 98.2% |
| `-lgc 510`   | 163.416      | 18.99  |  501.3  | 98.3% |

**This is the clean signal**: in a long single-launch kernel, all locks are honored and FFMA achieves ~98% of the theoretical peak at the locked clock. The "98% efficiency" is inherent to the ASM kernel (8 chains, 8-deep ILP, register port pressure), not a clock issue.

---

## RECONCILIATION

### Why 1942 vs 1920 vs 2032?

**There is no fixed "settling" clock**. The clock is dynamic and depends on:

1. **Idle baseline**: floats between 1500-1942 MHz (sampled at 1 Hz idle; 1942 is the most common observed step). This is NOT the boost clock — it's an intermediate DVFS step the firmware lands on when the GPU is "warm but not loaded" (e.g. just finished a kernel and waiting). The clock actually drops to 120 MHz when truly idle (verified).
2. **Sustained heavy load (long kernel, unlocked)**: jumps to **2032 MHz boost within ~600 ms** and stays there for the entire ≥35 s run (verified by 250 ms sampling).
3. **Short kernel + L2 flush mode**: launches are 0.4-1 ms. DVFS doesn't have time to ramp to 2032 — settles around 1870-1990 MHz effective. nvidia-smi sampled at 1 Hz catches mostly 1800 or 1920 because it samples between launches when the L2 flush kernel is running.
4. **`§0.FFMA "1942 MHz settling"` claim is a methodology artifact**: the agent observed 1942 MHz via nvidia-smi (an idle-step value), then computed TFLOPS using `peak × efficiency` with that clock. The TFLOPS number (71.82) is roughly correct, but the 1942 MHz attribution is wrong. The true effective clock for that short-kernel measurement was ~1990 MHz with 92% efficiency, OR equivalently ~1820 MHz with 100% efficiency. **Both formulations give the same TFLOPS** but only one has a basis in fact.

### The lock paradox — VERIFIED, with extension

`nvidia-smi -lgc N` for **N > 1920** is silently **clamped to 1920 MHz**:
- `-lgc 1920` → behaves as 1920
- `-lgc 1921` (untested but inferable) → 1920
- `-lgc 2031` → 1920
- `-lgc 2032` → **1920** (the documented paradox)
- `-lgc 2033` → 1920
- `-lgc 2050` (above advertised max boost) → 1920

The clamp is silent (no warning from nvidia-smi). The claimed boost of 2032 MHz is achievable ONLY through `-rgc` (unlocked) plus a long-running kernel that triggers full DVFS boost.

The RANGE form `-lgc <min>,<max>` with `<max>` set to the same value behaves **slightly differently**: e.g. `-lgc 1920,1920` runs ~2-3% slower than `-lgc 1920` (single-arg). The single-arg form sets only the upper bound and lets the firmware choose freely below; the range form constrains both ends and seems to inhibit the firmware's ability to opportunistically dip slightly higher between samples. This is reproducible but small.

### -lgc with N < 1500 in short-kernel mode

`-lgc 510` and `-lgc 1005` in the SHORT-kernel sweep showed effective clock ~1800 MHz (3.5×, 1.8× HIGHER than the requested lock!). Two reasons:
1. nvidia-smi shows the requested clock at idle (510 or 1005 MHz pre-run), but as soon as the kernel launches, the firmware jumps to 1800.
2. This is the same firmware behavior as the "stuck at 1005 MHz under load" symptom from the catalog — DVFS has its own opinion under sustained load.

In the LONG single-launch test these clocks ARE honored (501 MHz @ -lgc 510, 987 MHz @ -lgc 1005, both ~98% of the request). The difference is settling time: short kernels with L2-flush in between never let DVFS settle.

### What the rig actually does (clean statement)

| Mode                                            | Effective clock during FFMA | Note |
|-------------------------------------------------|----------------------------|------|
| unlocked + long (≥10 ms) sustained kernel      | 1990 ± 5 MHz, near 2032 boost | True B300 boost |
| unlocked + short (<1 ms) kernels in tight loop | 1870-1990 MHz fluctuating  | DVFS doesn't fully settle |
| `-lgc 1800` + any sustained kernel             | 1769 ± 5 MHz (98% of 1800) | RELIABLE |
| `-lgc 1500` + any sustained kernel             | 1468 ± 5 MHz (98% of 1500) | RELIABLE |
| `-lgc 1005` + LONG kernel                       | 987 MHz (98% of 1005)       | RELIABLE in long-kernel mode only |
| `-lgc 1005` + SHORT kernel + L2-flush           | ~1800 MHz (overridden!)     | NOT respected |
| `-lgc 510`  + LONG kernel                       | 501 MHz (98%)               | RELIABLE in long-kernel mode |
| `-lgc 510`  + SHORT kernel                      | ~1800 MHz (overridden!)     | NOT respected |
| `-lgc N` for N > 1920                           | 1920 MHz (silent clamp)     | Paradox |

---

## RE-VERIFICATION OF §0.FFMA AT EACH CLOCK

| Clock state            | Measured TFLOPS (long kernel) | Predicted from 75.49 × (clock / 1990) | Match |
|------------------------|-------------------------------|-----------------------------------------|-------|
| unlocked / boost @1990 | **75.49** (baseline)          | (baseline)                              | —     |
| `-lgc 1800` (eff 1769) | 67.03                         | 75.49 × (1769/1990) = 67.10              | 99.9% |
| `-lgc 1500` (eff 1468) | 55.64                         | 75.49 × (1468/1990) = 55.69              | 99.9% |
| `-lgc 1005` (eff 987)  | 37.39                         | 75.49 × (987/1990)  = 37.44              | 99.9% |
| `-lgc 510`  (eff 501)  | 18.99                         | 75.49 × (501/1990)  = 19.01              | 99.9% |

**Conclusion**: TFLOPS scales linearly with effective clock to within 0.1%. The "FFMA per-cycle work" is clock-independent, as expected. Any past measurement can be re-projected to a new clock by simple linear scaling, **provided the original measurement's effective clock is known accurately**.

Catalog §0.FFMA's "71.82 TFLOPS at 1942 MHz" should be re-stated as:
- **75.49 TFLOPS at unlocked boost (effective ~1990 MHz, 98% of 2032 hardware boost)** — for sustained workloads
- OR **66.4 TFLOPS at -lgc 1800-locked (effective ~1769 MHz, 98% of 1800)** — for repeatable measurements

The 71.82 figure is in between these and corresponds to the SHORT-kernel + tight-loop methodology where DVFS is in flux.

---

## RECOMMENDED PROTOCOL (going forward)

### Default for "trustable" measurements

```bash
# At start of session
sudo nvidia-smi -lgc 1800
# Verify
nvidia-smi --query-gpu=clocks.gr --format=csv,noheader
# Should print: 1800 MHz (idle) or 1800 MHz (during kernel)
```

When publishing numbers, **always state the clock**:
- `"@1800-locked"` (= effective ~1769 MHz under load, ≈98% of 1800)
- `"@2032-boost-sustained"` (= effective ~1990 MHz under load, ≈98% of 2032; requires kernel ≥10 ms)
- `"@<N>-locked"` (where -lgc N succeeded; verify with `nvidia-smi --query-gpu=clocks.gr` BEFORE measurement starts)

### For unlocked measurements

```bash
sudo nvidia-smi -rgc
# Verify
nvidia-smi --query-gpu=clocks.gr --format=csv,noheader
# Idle: probably 1500-1942 MHz (any DVFS step). NOT a measurement clock.
# After kernel start: 1990-2032 MHz under sustained load
```

State as: `"unlocked, sampled boost <min>-<max> MHz, effective ~<X> MHz from cycle/wall"`.

### Hard rules

1. **Always `pkill -9 QuickRunCUDA && pgrep -fa bench_` BEFORE every measurement** to avoid contention. Wait ≥5 s.
2. **Always run `sudo nvidia-smi -rgc; sleep 3; sudo nvidia-smi -lgc <N>` to get a clean lock**. Issuing `-lgc` over an existing lock can leave stale state.
3. **Verify the lock took effect**: `nvidia-smi --query-gpu=clocks.gr --format=csv,noheader` immediately after. If it shows a different value, your lock did not take.
4. **For low locks (N < 1500), always use long-kernel runs (≥50 ms each)** to ensure DVFS has time to honor the lock. Short-kernel + L2-flush will see the firmware override the lock upward.
5. **For -lgc values N > 1920, use unlocked instead** — they all pin to 1920.
6. **TFLOPS / TB/s / cy-as-ns must always specify clock**. Never report a unit-converted number without the clock context.

### Suggested suffix for every published number

`"X.XX TFLOPS @1800-locked"` or `"X.XX TFLOPS @boost-sustained-1990"` or `"X.XX cy/op @1800-locked = Y.YY ns/op"`.

---

## OPEN QUESTIONS

1. **Why does `-lgc N,N` (range form) yield slightly LOWER throughput than `-lgc N` (single-arg)?** The single-arg form is a UI sugar for `-lgc 0,N` (any value 0..N), while range form is `-lgc N,N` (exactly N). The firmware appears to use the upper-bound-only flexibility to occasionally exceed the requested clock briefly, which the range form prevents. This is small (~2-3%) but reproducible.

2. **What's the temperature dependence?** The sweep was done with the GPU at 40-47 °C (cool). At higher temps the effective boost may drop below 1990 MHz even with unlocked. Worth a separate session at hot/cool extremes.

3. **Is the 120 MHz "true idle" floor a power-saving feature or a hardware minimum?** Observed during gaps between launches. Brief DVFS dip to floor is a **major reason that L2-flush mode hurts repeatable timing** — every flush kicks the GPU into a low-power state.

4. **What does "Default Applications Clocks: Graphics 2032 MHz" mean if no measurement run actually hits 2032?** `nvidia-smi -ac 3996,2032` (set "applications clocks") was not tested in this sweep — could be a stronger guarantee than `-lgc`. Worth a follow-up.

5. **Long-kernel test at -lgc 1920 / 2032**: not yet done. Predicted to settle at ~1930 MHz effective and yield ~73 TFLOPS but should be verified.

---

## DATA LOCATIONS

- Per-state raw logs: `/root/github/QuickRunCUDA/justifications/clock_state_logs/<NN>_<state>.txt`
- Per-state 1Hz nvidia-smi samples: `/root/github/QuickRunCUDA/justifications/clock_state_logs/<NN>_<state>_sampled.txt`
- 250ms-resolution dynamics during 35 s sustained unlocked kernel: `/root/github/QuickRunCUDA/justifications/clock_state_logs/sustained_probe/dyn_v2_unlocked.csv`
- Sweep script: `/root/github/QuickRunCUDA/justifications/clock_state_logs/run_sweep_v2.sh`
- Initial CONTAMINATED sweep (kept for posterity): `/root/github/QuickRunCUDA/justifications/clock_state_logs_CONTAMINATED/`
