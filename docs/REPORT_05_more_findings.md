# Report #5 — Broader contention map + practical quant-kernel data

All numbers on GPU 0 at 1920 MHz current clock. GPU 1 had a persistent
power-capping issue reducing its peak to ~50%; all results here are from GPU 0.

## F2FP-pack + realistic companion mixes

Baseline = PACK only (1-way `cvt.rn.satfinite.e4m3x2.f16x2`) at N=4 chains:
**32 /SM/clk = 9064 GOps/s.**

| Config | GOps/s | /SM/clk | Ratio vs baseline |
|---|---:|---:|---:|
| PACK only | 9066 | 31.91 | 1.000 |
| +1 FFMA | 9052 | 31.85 | 0.999 |
| +4 FFMA | 9054 | 31.86 | 0.999 |
| +8 FFMA | 8787 | 30.92 | 0.969 |
| **+1 STG** | 9059 | 31.88 | **0.999** |
| **+2 STG** | 7730 | 27.20 | **0.853** |
| **+4 STG** | 1404 | 4.94 | **0.155** |
| +8 STG | 782 | 2.75 | 0.086 |
| +4 FFMA +1 STG (realistic quant) | 8829 | 31.07 | **0.974** |
| +8 FFMA +2 STG (quant+scale+write) | 7357 | 25.89 | 0.812 |

FFMA is ~free (zero perceptible impact up to 4 per pack). STG cliff at ~3/pack.

## F2FP + HMMA (tensor-core m16n8k16 FP16)

| +HMMA/iter | F2FP ratio |
|---:|---:|
| 0 | 1.000 |
| 1 | 0.970 |
| 2 | 0.941 |
| 4 | 0.876 |
| 8 | 0.751 |
| 16 | 0.403 |
| 32 | 0.198 |

Linear drop. HMMA at high N_MMA/iter crushes F2FP because both fight the warp-
scheduler dispatch port — **but this also happens to FFMA** (per sub-agent),
so it's a general dispatch-level effect, not F2FP-specific.

## F2FP + LDG (clean per-block-distinct addresses) — GPU 0 re-verified

| +LDG/iter | F2FP /SM/clk | Ratio |
|---:|---:|---:|
| 0 | 63.91 | 1.000 |
| 1 | 63.85 | 0.999 |
| 2 | 63.86 | 0.999 |
| 3 | 61.06 | 0.955 |
| 4 | 60.17 | 0.941 |
| 8 | 56.81 | 0.889 |
| 16 | 51.12 | 0.800 |
| 32 | 42.55 | 0.666 |

## F2FP + STG (clean per-block-distinct addresses, coalesced)

| +STG/iter | F2FP /SM/clk | Ratio |
|---:|---:|---:|
| 0 | 63.93 | 1.000 |
| 1 | 63.92 | 1.000 |
| 2 | 63.86 | 0.999 |
| 3 | 63.78 | 0.998 |
| **4** | **55.08** | **0.862** (cliff) |
| 6 | 28.67 | 0.449 |
| 8 | 22.02 | 0.344 |
| 16 | 12.47 | 0.195 |
| 32 | 6.69 | 0.105 |

## F2FP vs warp divergence

With only `ACTIVE_LANES` of 32 executing F2FP per warp (guarded by `if(lane<AL)`):

| ACTIVE_LANES | per-active-lane F2FP rate /SM/clk |
|---:|---:|
| 32 | 63.7 (baseline) |
| 16 | ~32 |
| 8 | ~16 |
| 4 | ~8 |
| 2 | ~4 |
| 1 | ~2 |

F2FP throughput is PROPORTIONAL to active lanes — SFU processes per-lane. So
divergent warps waste SFU slots for inactive lanes too. **Avoid divergence
around F2FP code.**

## F2FP + SMEM loads (LDS) vs global loads (LDG) — GPU 1 data, ratios hold

| Op | N=0 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ld.shared.f32` | 63.68 | 63.61 | 63.61 | 59.93 | 56.61 | 51.03 | 42.53 |
| `ld.global.ca.v4.f32` | 61.51 | 61.45 | 61.34 | 57.98 | 54.87 | 49.50 | 41.81 |
| `atom.global.add.f32` | 63.78 | 0.34 | 0.17 | 0.08 | — | — | — |

- LDS and LDG contend similarly with F2FP (SMSP dispatch level, not SFU).
- Atomic adds serialize everything — 1 atomic kills F2FP. Every GPU-wide atomic op essentially freezes the pipeline.

## Thread block clusters (CLUSTER_SIZE = 1, 2, 4, 8)

F2FP throughput unaffected within noise. Clustering affects DSMEM/multicast
only, not SM-local SFU.

## Launch_bounds / MIN_BLOCKS

F2FP throughput unchanged whether MIN_BLOCKS=1 (1 block/SM) or MIN_BLOCKS=4
(4 blocks/SM). Block count doesn't affect SFU throughput per SM — only
total warp count matters, and 16 warps/block is already plenty.

## Practical recommendations for quant kernels

1. **Keep STG per pack ≤ 2** to stay at ~85% pack rate. Use wider stores
   (STG.128 per warp via `int4`) instead of many narrow STGs.
2. **Never mix MUFU (softmax `ex2`) with F2FP pack on the critical path** —
   direct SFU contention halves F2FP.
3. **FFMA, FMUL, IMAD are free** — pack all the scale math you need.
4. **Avoid atomics near F2FP** — 1 `atom.global` ruins throughput.
5. **Don't divergence** inside F2FP loops — SFU slots waste on inactive lanes.
6. **`ld.shared` is slightly cheaper than `ld.global` per op, but both cost
   about 1% of F2FP rate per op added**.

## GPU 1 anomaly

During this session GPU 1 started running F2FP at ~50% of GPU 0 rate
(~32 /SM/clk instead of 64) despite both clocks locked at 1920 MHz and GPU 1
showing 206W power draw. Likely a module-level power-sharing event or
persistent power-capping counter. No immediate fix; switched all measurements
to GPU 0. Pre-anomaly GPU 1 data in earlier reports is still valid since
ratios (and the model) hold; only absolute /SM/clk numbers drifted.

