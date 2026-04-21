# M11: B300 Per-Pipe Energy Reference

Synthesized from V5 D-series + V6 C-series + V7 K-series measurements.
All values @ 1500 MHz locked unless noted.

---

## Per-op energy table

| Op | Energy/op | Ratio to FFMA | Notes |
|----|-----------|----------------|-------|
| FFMA | **4.4 pJ** | 1.0× | V5 D2 baseline at 1500 MHz |
| LDG (.ca, L1 hit) | ~5 pJ | 1.1× | Implied from V5 D5 (.ca = 30 W active vs FFMA 124 W) |
| LDG (.cg, bypass L1) | ~10 pJ | 2.3× | V5 D5: .cg uses 87% MORE power than .ca |
| HMMA (BF16 mma.sync) | ~50 pJ/output | 12× | Tensor pipe is heavier per inst |
| tcgen05.mma | (deferred V8) | ? | Need full descriptor (V7 A1) |
| LDS (broadcast) | ~5 pJ | 1.1× | SMEM, fast |
| LDC/LDCU (cmem) | ~3 pJ | 0.7× | V6 K5: 3.4 cy/op cheap |
| MUFU (RCP/SQRT) | ~10 pJ | 2.3× | xu pipe; 15 cy/op |
| IADD3 | ~3 pJ | 0.7× | alu pipe lighter than fma |
| F2FP cvt (any narrow FP) | ~5 pJ | 1.1× | V6 H1/H2: 5.4 cy/op identical |
| __syncthreads | ~30 pJ | 6.8× | bar.sync overhead |
| cluster.barrier | ~390 pJ | 89× | Cross-CTA cost |

(Estimates derived by power × cycles / op count; HIGH variance ±30%)

---

## DVS scaling per op

Per V5 D4: V² scales with clock.

| Clock | pJ/FFMA |
|-------|---------|
| 510 MHz | 3.1 |
| 1005 MHz | 4.9 |
| 1500 MHz | 6.8 |
| 1920 MHz | 9.96 |

(Range 3.2× across clock sweep; ML loves boost per V6 C3)

---

## Per-byte memory energy

Per V6 C2 (memory-bound sweep):

| Clock | pJ/byte |
|-------|---------|
| 510 | 12.06 |
| 800 | **11.81** ← min |
| 1005 | 12.49 |
| 1500 | 14.92 |
| 1920 | 18.60 |

Memory ops are CHEAPEST at 800 MHz (different from FFMA 510 MHz min).

---

## Static vs dynamic split

Per V7 K4 (no SM power-gating):
- Static GPU: **165-170 W** regardless of utilization
- Per-block dynamic (FFMA): ~0.7 W
- All-148-blocks active: +103 W dynamic = 270 W total
- All-148-blocks at 552 W under FFMA-saturated boost = 387 W dynamic

Static is 30-60% of total power. Low-occupancy workloads waste 75-150× efficiency.

---

## Power per workload type

| Workload | Pavg | TFLOPS / GB/s | Joules/TF |
|----------|------|---------------|-----------|
| FFMA-bound | 359 W | 39.7 TFLOPS | 9.0 J/TFLOP |
| Mixed (4 FFMA + 1 LDG) | 463 W | 9.5 TFLOPS | 49 J/TFLOP |
| HBM streaming | 460 W | 7.5 TB/s | 61 nJ/byte |
| Idle | 167 W | 0 | ∞ (waste) |

Mixed workloads dominate energy because both compute and memory subsystems active.

---

## Tooling: how to extend

For new workload energy:
1. Use `tests/bench_v6_c1_run.sh` template
2. Modify kernel + total op count formula
3. Run sweep: `bash tests/bench_v6_cN_run.sh`
4. Reports pJ/op vs clock

For per-pipe instantaneous power:
- `utils/power_sampler` (NVML 6483 Hz, V5 L2 commit `8bb15c8`)
- Run kernel + sample at high freq → see ramp/decay

---

## Open questions for V8

1. tcgen05.mma actual pJ/op (needs V8 tcgen05 work)
2. Per-pipe SASS-level energy breakdown via ncu sm__pipe_*_cycles_active.ratio.metric
3. Cross-pipe energy interaction (does HMMA + LDG cost more than sum?)
4. Voltage rail measurement (NVML provides power; not voltage directly)

---

## Source commits

- V5 D2 (FFMA pipe duty): 2b5f451
- V5 D4 (DVS): 42bfa01
- V5 D5 (.ca vs .cg power): f18345b
- V6 C1 (FFMA min-energy): b3486dc
- V6 C2 (memory min-energy): 389bdbb
- V6 C3 (mixed best at boost): 4970264
- V6 C4 (single-warp 75× worse): 81c7a74
- V7 K1 (sustained no-throttle): 552d1c1
- V7 K4 (no SM power-gating): d71bdc8
- V7 J4 (persistent ~0 W): c01310a
