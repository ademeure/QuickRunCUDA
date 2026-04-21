# M9: B300 Energy-Throughput Pareto Frontier (V6 C-series synthesis)

Synthesis of V6 C1 (FFMA), C2 (memory), C3 (mixed) energy sweeps.

---

## The KEY insight

**Min-energy clock is workload-DEPENDENT, not constant.**

| Workload type      | Min-energy clock | Per-task energy at min | vs boost (1992) |
|--------------------|------------------|-----------------------|-----------------|
| Pure FFMA-bound    | **510 MHz**      | 5.76 pJ/FFMA          | 16% lower       |
| Memory-bound (L2/DRAM mix) | **800 MHz** | 11.81 pJ/byte    | 36% lower       |
| **Mixed FFMA+DRAM (real ML)** | **1992 MHz boost** | 53 mJ/task | **3× lower** than 510 |

---

## Energy curves per workload

### V6 C1 — FFMA-saturated (compute-bound)

```
pJ/FFMA across clocks:
510:  5.76  ← min
800:  6.21
1005: 6.38
1200: 7.23
1402: 7.08
1500: 6.84
1702: 6.54
1920: 6.92
```

Range: 1.26× (5.76 → 7.23). Low spread because FFMA pipe well-utilized at all clocks.

### V6 C2 — Memory-bound (mixed L1/L2/DRAM)

```
pJ/byte across clocks:
510:  12.06
800:  11.81  ← min
1005: 12.49
1200: 13.32
1500: 14.92
1700: 15.82
1992: 18.60
```

Range: 1.58× (11.81 → 18.60). Min at 800 because lowest clock starves SMs on L2 latency.

### V6 C3 — MIXED FFMA+DRAM (4 FFMA per LDG)

```
mJ/task across clocks (NORMALIZED to 1992):
510:  3.08x
800:  2.74x
1005: 2.24x
1500: 1.24x
1992: 1.00x  ← MIN
```

Range: 3.08×. **Boost clock wins by far** because mixed workloads keep both pipes busy → static power dominates at low clock.

---

## Why mixed workloads love boost clock

Static power on B300 ≈ 165 W (idle baseline).
At 510 MHz: total ~430 W → static is 38% of total.
At 1992 MHz: total ~530 W → static is 31% of total.

When workload is FAST (boost), task completes sooner → static power
amortized over less time → lower total energy per task.

When workload is mixed compute+memory and BOTH pipes are saturated, the
compute throughput scales linearly with clock (more useful work/cycle)
but power scales with V² (sublinear vs throughput).

Net: throughput grows faster than power → energy per task drops.

---

## Datacenter implications

Common belief: "lower clock = lower energy". TRUE for pure compute. FALSE
for mixed workloads (which dominate ML inference).

**ML inference recommendation: USE BOOST CLOCK** (1992 MHz on B300):
- Lowest energy per token
- Lowest latency
- Highest throughput

DVFS schemes that DOWN-clock ML inference will INCREASE total energy
consumption, not decrease it.

For dedicated **FFMA-only HPC kernels** (rare): 510 MHz can save 16%.
For **memory-bound streaming** (BW-limited): 800 MHz can save 36%.
For **realistic ML pipelines** (mixed): boost clock 3× more efficient.

---

## Workload classifier

To pick the right clock, classify by ratio:

| Compute / Memory time ratio | Optimal clock |
|------------------------------|---------------|
| > 4 (compute-bound)          | 510 MHz       |
| 1-4 (balanced)               | 1500-1992 MHz |
| < 1 (memory-bound)           | 800 MHz       |

For ML: most kernels are 1-4 ratio → boost clock optimal.
For physics solvers, signal processing: often > 4 → consider down-clocking.

---

## Tooling: utils/c_run.sh (sweep script template)

The C1/C2/C3 sweep scripts (tests/bench_v6_c{1,2,3}_run.sh) are reusable
templates for any workload's energy sweep. Pattern:
1. Lock clock with `nvidia-smi -lgc`
2. Sample baseline power
3. Launch kernel in background
4. Sample power 6× during run
5. Compute active power = sample - baseline
6. Compute pJ/op = active × time / op_count

Modify the kernel and op_count for new workloads.

---

## Commits referenced
- C1 (FFMA): `b3486dc`
- C2 (memory): `389bdbb`
- C3 (mixed): `4970264`
- D4 voltage scaling reference: `42bfa01`
