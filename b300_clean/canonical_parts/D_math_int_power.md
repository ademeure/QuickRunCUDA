# Section D — Math intrinsics, integer/bit ops, packed-FP cvt, power & clock

Sections §36 through §45. B300 SXM6 AC, sm_103a, 148 SMs.

---

## §36. MUFU per-op throughput — EX2 stands alone at 2.0× every other transcendental

**Answer:** `MUFU.EX2` runs at **9.22 Gops/s = 95.8% of the 1/(4cy)/SMSP SoL**, while every other MUFU op (LG2, RCP, RSQRT, SQRT, SIN, COS, TANH) sits at **4.74 Gops/s = 49%** — an exact 2.0× gap that is real, isolated, and load-bearing for softmax / `expf` / `tanhf` workloads. `[🟢 HIGH · src: 14_math_intrinsics_CORRECTED.md§1, V41_V48_FINDINGS.md§"ALU pipe (V41)"]`

The 2.0× ratio supersedes the pre-V41 catalog's "1.7× faster ~8.1 TGOps/s" which is **RETRACTED** (`MATH_INCONSISTENCY_LOG.md` Inc#1). EX2 is the only transcendental that is fast on B300 — every reduction, every direct-table polynomial-style intrinsic except EX2 is half-rate.

### Per-op MUFU throughput (V41, free-rein 10-rule rigor, sm_103a, 148 SMs)

| PTX op | SASS | Chain latency (cy) | Throughput (Gops/s, chip) | % of 1/(4cy)/SMSP SoL | Per-SM (Gops/s) | Notes |
|---|---|---:|---:|---:|---:|---|
| `ex2.approx.f32` | `MUFU.EX2` | **14.14** | **9.22** | **95.8%** | 62.3 | ANOMALY — 2× the rest |
| `lg2.approx.ftz.f32` | `MUFU.LG2` | 18 | 4.74 | 49% | 32.0 | half-rate tier |
| `rcp.approx.f32` | `MUFU.RCP` | **42.10** | 4.74 | 49% | 32.0 | longest pure-pipe latency |
| `rsqrt.approx.ftz.f32` | `MUFU.RSQ` | 18 (ftz) / **40.10** (rn) | 4.74 | 49% | 32.0 | non-FTZ doubles latency |
| `sqrt.approx.f32` | `MUFU.SQRT` | 18 / 40 | 4.74 | 49% | 32.0 | |
| `sin.approx.f32` | `MUFU.SIN` | **24.02** | 4.74 | 49% | 32.0 | |
| `cos.approx.f32` | `MUFU.COS` | 24 | 4.74 | 49% | 32.0 | |
| `tanh.approx.f32` | `MUFU.TANH` | 18 | ~4.74 | 49% | ~32 | catalog sec 7's 22.5 Gops/s/SM is LOW conf vs V41 |

The chip-level 4.74 vs 9.22 Gops/s split is **per-PTX-instruction throughput**, not per-element — these are scalar f32 intrinsics. Multiply by 1.058 for 2032 MHz boost vs the 1920 lock these were measured at. Per-SM = chip / 148.

### Why not measure as "GMUFU/s"?

The original V8 number (`V8_MUFU_PEAK.md`) reported **47.8 G thread-MUFU/s at 99.5% XU pipe utilization** for a self-dep `rsqrt` chain. That is a **1-chain latency-bound** measurement: one `MUFU.RSQ` issued, wait 40 cy for result, re-issue. ncu reads "XU 99.5% busy" because the XU sees one cycle of useful work followed by 39 cy of "I'm waiting for myself". The 1-chain harness produces 47.8 G; the V41 ILP-saturated harness produces **4740 G** (4.74 T) for non-EX2. **100× gap; both are correct at their level.** Do not quote 47.8 G as the "XU peak". (The synthesis doc `M16_V9_FULL_SYNTHESIS.md` cited the 47.8 G as "the" XU peak — that framing is **RETRACTED**, see §37 footgun.)

### EX2 anomaly — what we know vs hypothesis

What we know (V41, replicated):
- 2.0× gap is reproducible across 3 trials.
- Holds at both 1500 MHz lock and 2032 boost.
- SASS verifies one `MUFU.EX2` per loop iteration, no fusion / unrolling artifact.
- Chain-self latency is shorter for EX2 (14 cy) than for LG2/SIN/SQRT (18-24 cy).

Plausible mechanisms (none individually confirmed):
1. **Dedicated EX2 hardware lane.** Underlies `expf`, `__expf`, `expm1f`, `tanhf`, softmax — by far the most common transcendental in ML / quantization. NVIDIA may have widened this single sub-pipe to 1/(2cy)/SMSP while keeping LG2/RCP/etc at 1/(4cy).
2. **Smaller polynomial.** RCP/RSQRT/SQRT do Newton-Raphson refinement internally; EX2 is closer to a direct table+poly — fewer dependent micro-ops keeps the issue interval shorter.
3. **Separate writeback port.** EX2 may write back through a port that LG2/SIN/COS share, so EX2 never contends.

Test that would discriminate: mixed `EX2 + LG2` 50/50 chain. If both peak at 4.74 G summed, they share a port (rule out hyp 3). If they sum to 9.48 G, EX2 has an independent lane (confirm hyp 1). Not yet run.

Hopper (sm_90) reportedly does **not** show this 2× EX2 gap in published microbenches — needs verification on H100 to claim "Blackwell-specific."

### Implications for softmax / `expf` kernels

Softmax inner loop:
```
y_i = expf(x_i - max) / sum_expf(x_i - max)
```
The dominant intrinsic is `expf`, which lowers to `MUFU.EX2` after the constant-multiply by `log2(e)`. At 9.22 Gops/s chip-wide:
- 32 lanes × 4 SMSPs × 148 SMs × (1 EX2 / 2 cy) × 2.032 GHz = 19.27 G EX2/s theoretical at 1/(2cy)/SMSP ⟹ V41's 9.22 G = ~48% of that. The 95.8% figure is vs 1/(4cy)/SMSP (the SoL inherited from non-EX2 MUFU).

In practice, the softmax kernel will be MUFU-bound only if `expf` count per element ≥ ~5-8 FFMAs (depends on FFMA/EX2 mix). Most softmax implementations are FFMA-bound or HBM-bound, so EX2's anomaly headroom rarely converts to wall-clock speedup unless you pack ≥4 EX2 per warp without intervening dependent FFMA.

#### Toy softmax cycle accounting

For a 64-wide softmax row (typical attention head dim 64-128):
- 64 elements × 1 EX2 per element = 64 MUFU.EX2
- 64 elements × 2 FFMA (subtract max, multiply by inv_sum) = 128 FFMA
- Plus warp-reduce sum_exp = ~12 cy via REDUX or 27 cy via SHFL chain

EX2 throughput at 9.22 Gops/s chip ÷ 148 SMs ÷ 4 SMSPs = 15.6 Gops/SMSP/s = 7.7 Gops per SMSP (each SMSP has 32 lanes for thread-MUFU). So 64 EX2 per row, with one row per warp on each SMSP, takes 64 / 32 = 2 warp-EX2 = 8 cy at 1/(4 cy)/SMSP.

128 FFMA at 1 inst/cy/SMSP = 128/32 = 4 warp-FFMA = 4 cy.

Reduction: REDUX = 12 cy.

Total: 8 + 4 + 12 = 24 cy per row. EX2 is 33% of the cycle budget. With 4 SMSPs running 4 rows in parallel = 4 rows / 24 cy / SMSP = 0.667 rows/cy/SMSP × 4 SMSPs × 148 SMs × 2.032 GHz = 800 Grows/s … but wait, that ignores LDG / STG. In practice, attention softmax kernels are HBM-bound at moderate seq lengths and only become MUFU-bound if seq length > 4096 (when the row is too long to amortize the load).

So EX2's 2× anomaly DOES matter for very-long-context attention (Mamba, long-context GPT) where softmax dominates. For short-context softmax it is dwarfed by HBM.

#### `tanhf` cycle accounting

`tanhf(x) = 2 * sigmoid(2x) - 1`, where `sigmoid(x) = 1/(1+exp(-x))`. The dominant intrinsic is `MUFU.EX2` (after exp/log2 constant-mul) plus MUFU.RCP for the divide. Per-element:
- 1 FMUL (2x scaling)
- 1 MUFU.EX2 (exp inner)
- 1 FADD + 1 MUFU.RCP (1/(1+exp))
- 1 FMUL + 1 FADD (2× -1 wrap)

≈ 1 EX2 + 1 RCP + 4 FFMA-equivalent. RCP at 4.74 Gops/s = 32 Gops/SM; EX2 at 9.22 = 62 Gops/SM. The RCP is the bottleneck (slower than EX2), so tanhf does NOT benefit from the EX2 anomaly — it gates on RCP.

If a future B300 revision speeds RCP to 9.22 like EX2, tanhf would 2× speed up. As of sm_103a, tanhf bottoms out at 4.74 Gops/s/chip = 32 Gops/s/SM = 1 tanhf per ~6.3 cy/SM.

There is also a `MUFU.TANH` op (direct PTX `tanh.approx.f32`) which is in the LG2/RCP/SQRT/SIN tier at 4.74 Gops/s. If you want fast tanh on B300, use `MUFU.TANH` directly (one inst, ~18 cy chain latency, 4.74 Gops/s) rather than the `2 * sigmoid(2x) - 1` decomposition (which serializes EX2 → RCP and pays both latencies).

### Per-SM table — old catalog disagrees with V41

`14_math_intrinsics.md` sec 7 quotes the per-SM table: `exp2f 34.9 / sqrt-rsqrt 22.5 / sin-cos 20.6 Gops/s/SM`. V41 says all non-EX2 are **equal** at 32 Gops/s/SM, EX2 at 62 Gops/s/SM. The catalog sec 7 was lower-rigor (no SoL-% framing, no ncu cross-check). **Mark catalog sec 7 LOW; V41 is the AUTHORITATIVE source** until a re-test reproduces sec 7's variance.

The 32 Gops/s/SM number for non-EX2 = 4.74 chip / 148 SMs × 1000 = 32.0. The 62 Gops/s/SM for EX2 = 9.22 chip / 148 SMs × 1000 = 62.3. Per-SMSP: divide by 4 = 8.0 vs 15.6 Gops/s/SMSP. Per-cycle at 2032 MHz: 8.0 / 2.032 = 3.94 Gops/SMSP/cy = 0.123 ops/SMSP/cy ≈ 1/(8 cy) for non-EX2; EX2 hits 0.246 ops/SMSP/cy ≈ 1/(4 cy). So EX2's "1/(4 cy)/SMSP" is the standard SoL ceiling and EX2 is the only op that saturates it.

### Why "per-SMSP" matters for SMSP-class predictions

Each B300 SM has 4 SMSPs (sub-partitions), each with its own warp scheduler, FMA pipe segment, INT/bit-pipe segment, MUFU port, etc. The "1 inst per 4 cy per SMSP" framing means each SMSP can dispatch one MUFU.LG2 every 4 cycles; with 4 SMSPs in lock-step, that gives 1 MUFU per cycle per SM. Over the chip: 1 × 148 × 2.032 GHz × 32 lanes = 9.62 Telements/s if SMSP could really hold 1/(4 cy). V41 measured 4.74 — so non-EX2 MUFU effectively runs at 1/(8 cy) per SMSP, half the SoL.

Why the "SoL ceiling" is set at 1/(4 cy) and not 1/(8 cy): the catalog's prior assumption was that every MUFU op shared one ceiling; the V41 result shows EX2 alone reaches that ceiling. Whether the ceiling is "real" for non-EX2 ops in any execution mode (even theoretically) is open: maybe NVIDIA designed all MUFU ops at 1/(4 cy) but we're missing some pipeline/warp/ILP condition; maybe the architectural ceiling for non-EX2 is actually 1/(8 cy) and EX2 has a 2× lane that the others don't share. The mixed-EX2-with-LG2 chain test would discriminate.

### `__frsqrt_rn` is faster than `rsqrtf` — but not "faster than other MUFU"

`14_math_intrinsics.md` sec 3 cites `__frsqrt_rn` "2.69× faster than `rsqrtf`". This is `.approx` (1 MUFU.RSQ inst) vs IEEE-style (1 MUFU.RSQ + 7 NR refinement FMAs). It is NOT evidence rsqrt is faster than other MUFU — only that the approximate form skips refinement. Both bottom out at the same 4.74 Gops/s when issue-saturated. Do not generalize.

### sqrt / div anomalies — latency-bound by default

`14_math_intrinsics.md` sec 2 reports `sqrtf` 687 Gops/s and `1/x` 492 Gops/s — looks like an outlier vs the 4.74 Gops/s tier. Already explained inline in same file: nvcc default (no fast-math) emits `sqrt.rn.f32` (138 cy) and `div.rn.f32` (243 cy), both latency-bound and not pipe-bound. With `-use_fast_math` they collapse to `MUFU.SQRT` / `MUFU.RCP × FFMA` and rejoin the 4.74 Gops/s tier.

### Throughput ceiling reconciliation table

| Source | Claim | Verdict |
|---|---|---|
| `14_math_intrinsics.md` sec 1 | EX2 1.7× ~8.1 TGOps/s | LOW — should be 2.0× ~9.22 Gops/s, V41 supersedes |
| `V8_MUFU_PEAK.md` | rsqrt 47.8 G MUFU/s @ 99.5% XU util | OK as 1-chain figure; do NOT compare to 9.62 T |
| `M16_V9_FULL_SYNTHESIS.md` table I | XU peak 47.8 GMUFU/s | RETRACTED framing; saturated MUFU ~4.74 Gops/s/chip |
| `V41_V48_FINDINGS.md` | EX2 9.22 / others 4.74 Gops/s | AUTHORITATIVE |
| `14_math_intrinsics.md` sec 7 per-SM table | exp2f 34.9, log/sqrt/rsqrt 22.5, sin/cos 20.6 Gops/s/SM | LOW conf; V41 says all equal at 32 G/s/SM (62 for EX2) |

### How EX2's 2× anomaly affects libc / `__expf` / `__logf` consumers

CUDA libc has multiple `expf`-like functions:
- `expf(x)` — IEEE-style with full subnormal handling. Latency-bound; lowers to ~5-10 instructions including `MUFU.EX2` + range reduction + polish FMAs.
- `__expf(x)` (intrinsic, fast_math) — simplified path. Lowers to ~3-5 instructions including `MUFU.EX2` directly.
- `__expf_rn(x)` — round-to-nearest variant.

All three end up bottoming out on `MUFU.EX2` for the core transcendental. With `-use_fast_math`, the path is: const_mul × log2(e) → MUFU.EX2 → done. Without fast_math, you get NR-style refinement after the EX2 → 7+ FFMA chained behind the EX2. The EX2 anomaly only applies to the MUFU.EX2 inst itself; the wrapping FFMAs are at FFMA rate.

For `softmax(x)` patterns:
- Best case: pre-applied range reduction + raw MUFU.EX2. Hits 9.22 Gops/s.
- Typical case: `__expf(x)` + range reduction. Hits ~4.5-5 Gops/s effective.
- Worst case: `expf(x)` no fast_math. Hits ~1-2 Gops/s effective due to NR refinement chain.

Conclusion: USE `__expf` AND `-use_fast_math` for max EX2 anomaly leverage.

**Footgun:** ⚠ Don't generalize a single MUFU rate to "all transcendentals". EX2 is 2.0× the rest — the anomaly is real and load-bearing for any softmax / tanh / `expf`-heavy workload. Don't quote V8's 47.8 GMUFU/s as "the XU peak" — it is a single-chain rsqrt latency-bound number, ~100× below the saturated EX2 ceiling. If a reviewer asks "what is the MUFU peak", give two numbers: EX2 = 9.22 Gops/s chip; everything else = 4.74 Gops/s chip.

**See also:** §37 (MUFU latency split), §40 (packed FP cvt rate ladder), corrections/MATH_INCONSISTENCY_LOG.md, V41_V48_FINDINGS.md "ALU pipe (V41 — MUFU sweep)".

---

## §37. MUFU latency — EX2 has split issue/result-availability latencies

**Answer:** `MUFU.EX2` chained EX2→EX2 measures **14 cy/inst** (writeback shortcut), but `FFMA→EX2→FFMA` pays **~30 cy total** for a one-trip cross-pipe round-trip — about +22 cy excess vs the linear 4+14+4 = 22 cy expectation. RCP does NOT show this split (FFMA→RCP→FFMA = 50 cy = perfect linear sum vs 4+42+4). `[🟢 HIGH for the latency numbers; 🟡 MED for the issue-vs-availability mechanism · src: CHAIN_FP_MUFU_LATENCY.md, V41_V48_FINDINGS.md]`

For softmax-style kernels where EX2 result feeds an FFMA in the next instruction, **budget ~30 cy/EX2, not 14 cy.** The 14 cy figure only holds for back-to-back EX2 instructions that re-feed each other through the MUFU writeback path.

### Pure-pipe chain latencies (V4 / CHAIN_FP_MUFU_LATENCY)

| Op | cy/inst chain (issue→issue) | ns @ 2.032 GHz |
|---|---:|---:|
| FFMA / FADD / FMUL | 4.04-4.22 | 2.0-2.1 |
| HFMA2 / HMUL2 / HADD2 (FP16x2 packed) | 4.04 | 2.0 |
| HFMA2.F32 (mixed-precision FMA, FP16 inputs → FP32 acc) | 4.04 | 2.0 |
| MUFU.EX2 (EX2→EX2 chain) | **14.14** | 7.0 |
| MUFU.LG2 / SQRT / RSQ.ftz / TANH | 18 | 8.9 |
| MUFU.SIN / COS | 24.02 | 11.8 |
| MUFU.RSQ / SQRT (non-ftz IEEE rounded) | 40.10 | 19.7 |
| MUFU.RCP | 42.10 | 20.7 |
| `redux.sync.add.u32` (full warp) | ~11.6 | 5.7 |
| `SHFL.BFLY` (single-instruction chain) | ~5 | 2.5 |

### Cross-pipe composition (FFMA + MUFU)

| Composition | Measured | Sequential expected | Excess |
|---|---:|---:|---:|
| FFMA → RCP → FFMA | 50 cy | 4 + 42 + 4 = 50 | **0 (perfect linear)** |
| FFMA → EX2 → FFMA | 44 cy | 4 + 14 + 4 = 22 | **+22 cy (2× anomaly)** |
| RCP → EX2 → RCP | 120 cy | 42 + 14 + 42 = 98 | +22 cy |

The +22 cy excess appears **only** when EX2's result must be consumed by a different pipe. RCP does not show this — FFMA→RCP→FFMA exactly sums.

### Best-fit hypothesis (CHAIN_FP_MUFU_LATENCY)

MUFU has separate **issue-interval** (when the next MUFU inst can be issued) and **result-availability** (when the result is visible to a different pipe) latencies:
- EX2: issue-interval 14 cy (writeback path that fast-forwards to the next EX2), result-availability ~30+ cy for cross-pipe consumers.
- RCP: issue-interval ≈ result-availability ≈ 42 cy. No fast-path for chained RCPs.

This is the most economical explanation — but it is not yet proven by direct test. A clean discriminator would be a longer FFMA-EX2-FFMA-EX2-FFMA chain (NC=8+) where the result-availability delay can be extracted separately from the issue interval.

### Mixed-precision bridging cost (CHAIN_FP_MUFU_LATENCY)

A test of HFMA2 → cvt.f32.f16 → FFMA → cvt.f16x2.f32 → HFMA2 (5 inst per iter):
```
Measured: 24 cy/iter, 5 inst/iter → 4.8 cy/inst average
Expected pure-chain: 5 × 4 = 20 cy
Excess: 4 cy total (≈ 2 cy per cvt)
```

**`cvt` instructions cost ~2 cy each above the pure FMA chain rate.** Mixed-precision pipelines (FP16 multiply, FP32 accumulate, etc.) pay very little for the precision crossings.

This is the latency cost (chain-dependent). The throughput cost is separate and addressed in §40 — packed FP cvt has its own F2FP pipe, which limits throughput to 19.3 Telem/s PACK / 38.5 Telem/s UNPACK chip-wide.

For mixed-precision GEMM patterns (FP16 multiply, FP32 accumulate, FP16 store):
- Per output element: 1 HFMA2.F32 + 1 cvt.f16x2.f32 (store) + 1 cvt.f32.f16 (load)
- Cycles per element: 4 (HFMA2.F32) + 2 (cvt down) + 2 (cvt up) = 8 cy at chain
- Versus pure F32: 4 cy. So mixed-prec pays 2× latency penalty in the pure-chain regime.

But the throughput regime (independent inputs) is different — the F2FP pipe is separate from FMA pipe, so cvts overlap with FFMAs at the SM level. At full ILP, mixed-prec ≈ pure-FFMA throughput as long as you don't saturate the F2FP pipe.

### Why EX2 has split issue/availability latencies — possible mechanism

The 14 cy chain-self latency vs 30 cy cross-pipe latency is consistent with EX2 having a **bypass** writeback: the result is forwarded to the next EX2 issue at the MUFU pipe's internal staging, before being written to the register file. The full RF write takes another ~16 cy.

When a different pipe (FMA) consumes the EX2 result, it must read from the RF, which means waiting the full RF write completion = 14 + 16 = 30 cy.

RCP doesn't have this fast-path because RCP's internal Newton-Raphson refinement uses the RF as scratch space for intermediate values; the final result hits RF at the same time as it would be available for chain forwarding.

This is a PLAUSIBLE mechanism but unverified. Discriminating tests:
- Long FFMA-EX2-FFMA-FFMA-EX2-FFMA chain: should show two cross-pipe penalties stacked.
- EX2-FFMA-EX2 chain (FFMA in between): should show one cross-pipe penalty.
- EX2-NOP-EX2: if NOP is enough to drain the bypass path, this should show ~30 cy not 14.

None of these have been run. Best-fit hypothesis only.

### V8 MUFU "47.8 GMUFU/s" — what it actually is

`V8_MUFU_PEAK.md` ran `rsqrt.approx.f32` with NC=8 chain, 256 thr × 148 blocks, 100K iters. ncu reported `sm__pipe_xu_cycles_active.avg.pct_of_peak_sustained_active = 99.49%`. The kernel issued 303 G thread-MUFUs in 6.34 ms = 47.8 G/s. Per SM × clock: 0.159 thread-MUFU/SMcycle = 1 thread-MUFU per ~6.3 cy/SM (aggregate).

That 6.3 cy/SM aggregate is the **ratio of saturated XU work to wall time**, not the MUFU pipe's intrinsic issue rate. Because each rsqrt is fed back into the next rsqrt (chain dep), each MUFU.RSQ inst pays ~40 cy of result-availability latency before the next can issue. ncu sees the XU as "99.49% busy" because the only work the kernel offers it is one inst per 40 cy — and the XU does that one inst, then sits at 100% busy waiting for itself to finish.

V41 broke the chain dependency (independent MUFU streams), so the XU could issue 1 MUFU per 4 cy per SMSP — yielding 4740 G chip-Gops/s, **100× the V8 number**. Both are correct measurements; they answer different questions.

### M14 / M16 quoted "47.8 GMUFU/s = XU peak" — RETRACTED

The synthesis docs `M16_V9_FULL_SYNTHESIS.md` table I and `M14_V8_SOL_LADDER.md` quoted V8's 47.8 GMUFU/s at "99.5%" as the XU peak. That framing is **RETRACTED** (see `MATH_INCONSISTENCY_LOG.md` Inc#3). It is a 1-chain rsqrt latency-bound number, not a saturated peak. Replacement framing: "1-chain rsqrt latency-bound 47.8 G; saturated MUFU 4.74 G chip-Gops/s (others) / 9.22 G (EX2)".

The 100× discrepancy (47.8 G vs 4740 G) is a textbook ILP-saturation gap, not a controversy. A future reader who sees "47.8 GMUFU/s peak" should mentally re-tag it as "rsqrt chain-self-fed minimum throughput" and look for the V41 number for the actual ceiling.

**Footgun:** ⚠ M14/M16 quoted "47.8 GMUFU/s = XU peak" — that's 1-chain rsqrt LATENCY-bound, NOT throughput. Off by ~100×. RETRACTED. When budgeting MUFU-heavy kernels, pick the right number for the regime: chain-dependent code → use the latency table (4-42 cy depending on op); independent / ILP-saturated code → use 4.74 Gops/s (or 9.22 for EX2) as the ceiling.

**See also:** §36 (MUFU per-op throughput), §39 (overall ALU pipe ladder), corrections/MATH_INCONSISTENCY_LOG.md, CHAIN_FP_MUFU_LATENCY.md, V8_MUFU_PEAK.md.

---

## §38. SHFL = REDUX raw rate — both 9.5 Telements/s = 1/(4cy)/SMSP

**Answer:** **`SHFL.BFLY` and `redux.sync.add.u32` measure equal raw per-instruction throughput at ~9.5 Telements/s = 1 inst per 4 cy per SMSP** (V37 = 9.09; V38 = 9.48). At the algorithm level, REDUX.SUM (single inst) replaces a 5-step SHFL chain, giving a **2.34× speedup** at the warp-reduction-of-32-lanes level (Q3, single trial, NC=8, 1500 MHz lock). The legendary "REDUX 4× SHFL" appears nowhere in the measurement record — **RETRACTED**. `[🟢 HIGH · src: V41_V48_FINDINGS.md V37/V38, Q3_WARP_REDUCE_RECIPES.md, 14_math_intrinsics_CORRECTED.md§2]`

Both primitives share the **MIO / shuffle pipe** at 1 inst per 4 cy per SMSP. Per chip: 4 SMSPs × 148 SMs × 1/4 inst/cy × 2.032 GHz = 300 Ginst/s warp-level = 9.6 Telements/s thread-level. V37 hits 94% of this refined SoL; V38 hits 98%.

### Raw-rate table

| Primitive | Lat (cy, chained) | Raw throughput (Telements/s, chip) | Pipe |
|---|---:|---:|---|
| `SHFL.BFLY` (warp shuffle) | ~5 (chain) | **9.48** (V38) | MIO / shuffle |
| `redux.sync.{add,min,max,and,or,xor}.u32/s32` | ~11 | **9.09** (V37) | MIO / shuffle |

**Same pipe, same per-instruction rate.** The supposed "4× SHFL" advantage is folklore.

### Algorithm-level: REDUX is 2.34× SHFL chain (Q3, replicated)

For a sum-reduce of 32 lanes, the legacy CUDA pattern requires 5 iterations of `__shfl_xor_sync` + add. `redux.sync.add.u32` does the entire 32→1 reduction in one instruction, writing the result to the **uniform register file (URF)** via `REDUX.SUM UR<n>, R<m>` SASS encoding (since the result is identical across all 32 lanes, the URF is the natural target).

| Method | cy/reduce | Speedup vs SHFL chain |
|---|---:|---:|
| 5-step `SHFL.bfly` chain (manual PTX) | 27.19 | 1.0× (baseline) |
| 5-step `SHFL.up` chain (manual PTX)   | 27.19 | 1.0× |
| `__shfl_xor_sync` intrinsic chain (CUDA) | 27.19 | 1.0× |
| **`redux.sync.add.u32`** (single inst) | **11.61** | **2.34× faster** |

A 5-step SHFL chain has 5 sequential SHFL+ADD pairs, each ~5 cy (SHFL latency), so chain length = ~25-27 cy. The single `REDUX.SUM` instruction performs the entire 32→1 reduction in hardware in ~11 cy — roughly 2.3× faster than the chain because it bypasses 4 of the 5 SHFL latencies.

### Where the "4× SHFL" myth came from

The user's MEMORY entry from the V4 loop session says "redux.sync.min/max **4× SHFL** on B300". `V8_SHFL_PEAK.md` line 30 cites the 4× as "V4 prior findings" but does not source it. Searching the corpus, **no measurement file produces a 4× ratio** — closest is V8_SHFL_PEAK's own 3 G warp-SHFL/s in a chain-dep regime, vs Q3's 11.6 cy/REDUX. If you compute `V8 self-dep SHFL chain (~100 cy/SHFL effective)` against Q3's 11.6 cy/REDUX you get 8.6×, not 4×. The 4× number is **legend, not measurement** (`MATH_INCONSISTENCY_LOG.md` Inc#2). Cite 2.34× algorithm-level with Q3 as the authoritative source.

### V8_SHFL_PEAK's 3 G warp-SHFL/s — what it is

`V8_SHFL_PEAK.md` ran `SHFL.BFLY` in a chain-dep loop (`v = shfl(v, ...)`), 128 SHFL per loop, 148 SMs × 8 warps × 62500 iters × 128. Wall: 3.16 ms for 9.47 G warp-SHFLs ⟹ 3.0 G warp-SHFL/s (= 96 G thread-SHFL/s if you multiply by 32 lanes). Far below the 9.48 Telements/s of V38.

The discrepancy is **chain-dep latency vs ILP-saturated throughput**:
- V8: each SHFL waits for the previous SHFL's result before it can issue. Chain rounds at ~5 cy SHFL + scheduling = ~100 cy effective for the whole 128-SHFL inner loop ⟹ 3 G warp/s.
- V38: SHFLs are issued from independent registers, no chain dep ⟹ 1 inst / 4 cy / SMSP = 9.48 Telements/s.

Both numbers are correct in their own framing. ALWAYS label whether SHFL throughput is "chain-dep self-feed" or "ILP-saturated independent".

### Practical recipe — replace SHFL chains with REDUX where possible

```cpp
// SLOW (legacy CUDA pattern):
for (int offset = 16; offset > 0; offset /= 2)
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);

// FAST (Hopper+, sm_90+, single inst, 2.34× speedup):
asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;"
             : "=r"(sum) : "r"(sum));
```

`redux.sync` supports: `.add`, `.min`, `.max`, `.and`, `.or`, `.xor` for `.u32`/`.s32`. **It does NOT support FP** — for FP reduction you still need SHFL chain (or first-bitcast trick: cast f32 to u32, reduce as u32 if safe — only if monotone/positive).

### `redux.sync.add.f32` does NOT exist on B300 sm_103a

Despite the catalog occasionally implying FP REDUX is available, the PTX 8.x ISA does NOT define `redux.sync.add.f32`, `.f16`, or `.bf16`. INT-only.

Workaround for FP warp-reduce:
1. **SHFL chain** (5 steps for warp-32). Only choice for arbitrary FP.
2. **Bitcast-as-u32 trick** (only for monotone non-negative floats — IEEE-754 unsigned int representation is monotone for f32 with sign-bit clear). Cast f32 → u32, REDUX.MAX, cast back. Works for absmax. Does NOT work for general sum (loses precision after one step).
3. **`__reduce_max_sync` / `__reduce_min_sync` intrinsic** (CUDA cooperative_groups) — compiles to REDUX for INT, falls back to SHFL chain for FP.

For FP sum-reduce, expect the 5-step SHFL chain at 27 cy. There is no shortcut on B300 hardware.

### Where SHFL chains are still optimal

If you need cross-lane reductions of an arbitrary FP type or with arbitrary semantics (e.g., "sum but ignoring NaN", "reduce only lanes where mask is set"), SHFL chain is still required. Examples:
- Per-warp variance computation (Welford's algorithm needs running mean+variance, can't use REDUX)
- FP softmax warp-reduce sum (needs FP precision)
- Per-warp max-of-absolute-value with sign preservation (REDUX would lose sign)

For these, use the SHFL chain and budget ~27 cy/reduce.

### Cross-warp reduction (block-level)

Both SHFL and REDUX are warp-only. For block-level reduction (e.g., reducing across 256 threads = 8 warps):
1. Per-warp warp-reduce (REDUX or SHFL) — 1 result per warp
2. Per-warp lane 0 writes to SMEM
3. Block sync (`__syncthreads`)
4. First warp loads SMEM, REDUX again

Total cost: 27 cy SHFL + 4 cy STS + 30 cy syncthreads + 27 cy second SHFL + 4 cy LDS = ~92 cy for 256-thread block-level reduction. With REDUX: 12 + 4 + 30 + 12 + 4 = 62 cy. Saves ~30 cy per block-reduce — ~33% per stage.

For multi-block reductions (grid-wide), use atomic to a global accumulator or cooperative_groups grid sync (slow, ~µs).

### Clock-cycle analysis of SHFL.BFLY chain self-feed (V8 regime)

V8 ran `v = SHFL.bfly(v, ...)` in a chain dep. Each SHFL has ~5 cy issue-to-issue latency. With 128 inner SHFLs serialized: 128 × 5 = 640 cy per inner loop. With 8 warps × 256 thr × 148 blocks × 62500 outer iters: total throughput = 9.47 G warp-SHFLs in 3.16 ms = 3.0 G/s warp = 96 G thread-SHFL/s.

Compare to V38 (independent SHFLs, 1/(4 cy)/SMSP): 4 SMSPs × 148 SMs × (1/4) inst/cy × 2.032 GHz × 32 lanes = 9.62 Telements/s.

Ratio: 9620/96 = **100×**. The chain-self-feed is 100× slower than independent SHFLs. This is the same ILP-saturation gap as MUFU.RSQ in V8 (also ~100× off from V41 saturated ceiling). General lesson: any "self-feed chain" microbench measures latency, not throughput; do not extrapolate.

### Where REDUX shows up in production

- **NCCL allreduce primitives** at warp boundary use REDUX where supported (sm_80+).
- **Per-warp absmax / max for INT8 quantization** — REDUX.MAX directly.
- **Per-warp histogram bin sum** — REDUX.ADD.
- **Block-level reduction's first stage** (warp → SMEM → block) — REDUX.
- **CUTLASS / CUTeDSL warp-cooperative reductions** (e.g., epilogue reductions in BF16/FP8 GEMM kernels).

For a typical "warp-reduce + SMEM-reduce + final atomic" kernel, the warp-reduce is ~25% of total time (3 stages of ~equal cost), so 2.34× faster warp-reduce = ~14% kernel speedup (Q3 confidence MED on the 14% — depends on kernel profile).

### `__reduce_*_sync` cooperative_groups intrinsic — compiles to REDUX where possible

CUDA cooperative_groups library exposes:
```cpp
auto warp = cg::tiled_partition<32>(this_block());
int sum = cg::reduce(warp, lane_value, cg::plus<int>());      // → REDUX.ADD
float fsum = cg::reduce(warp, lane_value, cg::plus<float>()); // → SHFL chain (no FP REDUX)
```
The intrinsic auto-selects: REDUX for INT, SHFL for FP. Use this when writing portable code (sm_70 fallback to SHFL chain anyway).

### Bitcast-as-u32 trick for non-negative monotone floats

For specific reductions where the data is known non-negative (e.g., per-warp absmax on `fabsf(x)` outputs):

```cpp
float x = fabsf(input);
uint32_t u = __float_as_uint(x);                                  // monotone for non-negative floats
uint32_t umax;
asm("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(umax) : "r"(u));
float xmax = __uint_as_float(umax);
```

This works because IEEE-754 single-precision encoding is monotone-increasing in the unsigned int representation for non-negative values: 0.0 → 0, smallest subnormal → 1, 1.0 → 0x3F800000, +inf → 0x7F800000.

**Limitations:**
- ONLY for non-negative floats. Negative values have inverted ordering.
- Does NOT work for sum (loses precision after first step).
- Works for min if you flip the bits (or use `redux.sync.min.u32` on `~u` and re-flip).

For abs-max in attention / quantization codebases, this gives REDUX speedup over SHFL chain. Verify with bit-level tests.

### Beyond REDUX — collective groups for cluster-level reduction

For multi-CTA cluster reductions (sm_90+ DSMEM):
- `cluster.barrier` synchronizes all CTAs in cluster (390 pJ per call per M11)
- DSMEM allows cross-CTA reads
- No "cluster reduce" PTX op exists — must hand-roll using DSMEM + barrier

For 2-CTA cluster: each CTA does warp-level REDUX, writes result to its own SMEM, cluster.barrier, both CTAs read from each other's SMEM via DSMEM, REDUX or pairwise reduce. ~2× cost of single-CTA reduce + barrier.

For 8-CTA cluster (max on B300, MAX=8 per V5): tree-reduce across 8 partial sums takes 3 stages = ~50 cy per stage = ~150 cy total. Less efficient than warp-reduce within a single CTA, but enables larger working sets.

### Where this matters

- **AllReduce primitives** at warp boundary
- **Per-warp absmax / max** for quantization
- **Per-warp histogram bin sum**
- **Block-level reduction's first stage** (warp → SMEM → block)

For a typical "warp-reduce + SMEM-reduce + final atomic" kernel, the warp-reduce is ~25% of total time (3 stages of ~equal cost), so 2.34× faster warp-reduce = ~14% kernel speedup (Q3 confidence MED on the 14% — depends on kernel profile).

### Throughput vs latency framing — what to cite

| Claim | Right framing | Cite |
|---|---|---|
| "REDUX is 4× SHFL on B300" | RETRACT — folklore, no measurement | — |
| "REDUX is 2.34× faster than 5-step SHFL chain" | Algorithm-level, single warp-reduce-32 | Q3_WARP_REDUCE_RECIPES.md |
| "SHFL and REDUX have the same per-inst throughput" | Raw rate, ILP-saturated | V37/V38 |
| "SHFL throughput is 3 G warp/s" | Chain-self-fed latency-bound regime ONLY | V8_SHFL_PEAK.md (label clearly) |
| "SHFL throughput is 9.48 Telements/s" | Independent / ILP-saturated regime | V38 |
| "redux.sync 4× SHFL on B300" (V8_SHFL_PEAK line 30) | RETRACTED — no source measurement | — |

**Footgun:** ⚠ "REDUX is 4× faster than SHFL" without specifying algorithm-level vs raw inst is misleading. Raw rate they are equal. At the algorithm level (warp-reduce-32) REDUX is 2.34×, not 4×. Always specify framing. Also: REDUX is INTEGER-ONLY — no FP variant exists on B300, so for FP warp-reduce you still need the SHFL chain (no escape).

**See also:** §36 (MUFU rates that share the XU pipe), §39 (overall ALU/INT pipe ladder), corrections/MATH_INCONSISTENCY_LOG.md, Q3_WARP_REDUCE_RECIPES.md, V41_V48_FINDINGS.md V37+V38.

---

## §39. INT/bit-op pipe throughput ladder (rates only — see §27 for definitive pipe placement)

**Answer:** B300 has a tiered INT/bit-op throughput ladder, NOT a uniform "ALU @ 19 TIOPS". Top tier (FMA pipe at 67%): FFMA/FADD/IADD3 at 25-26 Glane/s. Half-rate tier (INT-bit at 48%): LOP3/IMUL/IMAD at 18.7 Glane/s. PRMT (permute pipe at 36%): 13.9 Glane/s. ISETP (compare sub-pipe at 22%): 8.4 Glane/s. POPC/BREV/CLZ (XU at 12%): 4.7 Glane/s. The legacy V9 "FFMA + IADD3 = 114 TOPS combined" is **FULLY RETRACTED**: measured FFMA + IADD3 overlap is 14-17%, NOT 2× and NOT 131% sum. `[🟢 HIGH for the rates per V40; 🟡 MED for the pipe labeling — see §27 for the definitive cross-source pipe map · src: 15_integer_bit_ops_CORRECTED.md, V41_V48_FINDINGS.md V40, A6_PER_PIPE_REFERENCE.md, INT_INCONSISTENCY_LOG.md]`

This section gives the rate ladder. For the load-bearing "which physical pipe" map (FMA / INT-bit / permute / compare / XU / MIO-shuffle / LSU / LDC / LDS / FP64 / TENSOR / TMA / etc., with overlap percentages) see Section C §27.

### V40 ALU pipe ladder (1500 MHz lock, persistent grid, asm-volatile anti-DCE)

| Op | Pipe per V40 (B300) | Glane/s @ 1500 lock | %SoL of FMA pipe | inst/SMSP/cy | Notes |
|---|---|---:|---:|---:|---|
| **FFMA / FADD / FMUL** | **FMA** | 25-26 | **67%** | ~0.66 | dep-chain stalls cap at 67%; D1 multi-warp ILP confirms 85.5% true ceiling |
| **IADD3** | **FMA** (V40) | 25-26 | **67%** | ~0.66 | shares FMA-pipe issue slot, NOT a separate pipe per V9 |
| **IMAD / IMUL (32-bit, .lo)** | **FMA** (V8/V40) | 18.7 | **48%** | 0.5 | "INT-bit half-rate" tier |
| **LOP3.LUT** | **INT-bit** (V40); imm-independent (C3) | 18.7 | **48%** | 0.5 | C3 verified across 12 truth-tables; ≥3 unique reads no penalty |
| **PRMT** (byte permute) | **permute** (V40) | 13.9 | **36%** | ~0.46 | V39 LICM-fixed; original 1547% bogus retired |
| **SHF.L/R / SHL / SHR** | INT-bit (A6) | 14.12 (A6) | ~48% | 0.5 | Same tier as LOP3 |
| **BFI.b32** | INT-bit (A6) | 13.15 | ~46% | 0.46 | folds to LOP3.LUT in many cases |
| **ISETP / FSETP** | **compare** (V40) | 8.4 | **22%** | 0.25 | Lower tier than LOP3/PRMT; do NOT lump into "ALU @ 19 TIOPS" |
| **BFE.u32** | XU (A6) | 7.07 | ~25% | 0.25 | 2-SASS path (SHF.R + SGXT) |
| **SHFL.{IDX,BFLY,UP,DOWN}** | LSU/SHFL pipe | 7.06 | ~25% | 0.25 | A6 single-source verified |
| **POPC / BREV / CLZ / FLO** | XU | 3.5 | ~12% | 0.125 | 4× slower than LOP3 tier |
| **MUFU.EX2** | MUFU (XU) | 9.62 Gops/s | — | 0.003 | 95.8% of 1/(4cy)/SMSP per V41 |
| **MUFU.{LG2,RCP,RSQRT,SQRT,SIN,COS}** | MUFU | 4.74 Gops/s | — | 0.0015 | half rate of EX2 (V41) — see §36 |
| **REDUX / SHFL** (V37/V38) | shuffle pipe | 9.0-9.5 Telements/s | — | — | Same pipe; "redux 4× SHFL" was algorithmic — see §38 |

%SoL is vs FMA pipe peak of 38.5 Glane/s (1 inst/SMSP/cy at 1920/2032 MHz). At 1500 MHz lock, FFMA-pipe SoL itself is ~28.4 Glane/s, so the tier %s are calculated relative to the boost-clock SoL.

### LOP3 LUT immediate-independence (C3)

`C3_LOP3_LUT_DEEP.md` swept 12 different LUT immediates and 4 input-uniqueness configurations. Throughput is **independent of LUT value** as long as ≥3 unique source reads (4-cy chain latency). Predicate-using LOP3 variants (LOP3.LUT.PT_X) match the rate. No degradation pattern for the 12 truth-tables tested (XOR3, MAJ3, MUX, etc.).

| Imm  | Op meaning              | Time/iter (ms) | TIOPS  |
|---|---|---:|---:|
| 0x00 | const 0                 | 2.15181        | 14.09  |
| 0xAA | A (pass-through)        | 2.15221        | 14.08  |
| 0xCC | B (pass-through)        | 2.15206        | 14.08  |
| 0xF0 | C (pass-through)        | 2.15214        | 14.08  |
| 0x96 | A^B^C (3-input XOR)     | 2.15206        | 14.08  |
| 0x69 | ~(A^B^C)                | 2.15225        | 14.08  |
| 0xE8 | MAJ(A,B,C)              | 2.15192        | 14.09  |
| 0xCA | A?B:C (mux)             | 2.15235        | 14.08  |
| 0x80 | A&B&C                   | 2.15197        | 14.08  |
| 0xFE | A|B|C                   | 2.15200        | 14.08  |
| 0xFF | const 1                 | 2.15194        | 14.09  |
| 0x55 | ~A                      | 2.15228        | 14.08  |

Variance: ±0.005 TIOPS = ±0.04%. **Truth-table value does NOT affect throughput.**

#### Port-pressure sweep (1 warp/SM = 1 warp/SMSP)

`-t 32 -p` ⇒ each block = 1 warp, sent to a single SMSP, other 3 SMSPs idle.

| PORT_MODE | NC=1 | NC=2 | NC=4 | NC=8 |
|---|---:|---:|---:|---:|
| 0 (a,a,a)  | 4.48 cy | 2.30 cy | 2.15 cy | **2.08 cy** |
| 2 (a,b,c)  | 4.47 cy | 2.30 cy | 2.17 cy | **2.08 cy** |

Mode 0 = LOP3(R, R, R) → 1 unique register read per inst.
Mode 2 = LOP3(R, S, T) → 3 unique register reads per inst.

**Identical latency and throughput.** RF can deliver ≥3 unique reads per LOP3 issue cycle without throttling. 4.5 cy latency in single-chain regime, 2.08 cy/op throughput-saturated regime, 0.5 inst/SMSP/cy throughput SoL.

#### Practical implications for LOP3-heavy kernels (radix-X conversion tables, packed bitfield extraction, fused boolean ops, narrow-format arithmetic):

1. **No need to pick "favorable" truth tables** — all 256 imms identical throughput.
2. **No need to limit unique source register reads** — 3 distinct sources cost no more than 1 reused register.
3. **Need only 4 warps/SM** to fully utilize LOP3 pipe.
4. **Need 3+ independent chains per warp** to overlap latency at peak.
5. **At 2032 MHz boost: ~19.2 TIOPS chip peak** (= 18.7 V40 corrected; the discrepancy is V40 ran higher-occupancy with persistent grid).

### What "INT-bit at half rate" means in practice

V40's "INT-bit pipe at half rate" is a description of measured throughput (0.5 inst/SMSP/cy) relative to the FMA pipe (1 inst/SMSP/cy SoL). Whether this is:
- (a) the FMA pipe issuing LOP3 every 2 cycles,
- (b) a shared dispatch port between LOP3 and IMUL with 0.5/SMSP/cy throughput, or
- (c) a separate physical INT-bit pipe whose native cycle is 2 clocks,

V40 + A6 cannot disambiguate. The 14-17% FFMA + IADD3 overlap in B1 (which uses the FMA pipe label) suggests (a) or (b) — if INT-bit were truly independent of FMA, mixed FFMA + LOP3 should show much higher overlap. See §27 for the cross-source pipe map.

### IADD3 — V40 vs A6/B1 disagreement

| Source | IADD3 rate | Pipe label |
|---|---|---|
| V9_INT_OPS_PIPES (legacy) | ~38 TOPS = "full ALU pipe peak" at 99.94% pipe_alu | "ALU pipe (separate from FMA)" — RETRACTED |
| V9_MIXED_PIPES | implicit; SUM stalls at 131% (74 TOPS) | "ALU pipe; co-issuable with FMA" — partially retracted |
| 15_integer_bit_ops.md catalog | 2.46 w-inst/SM/cy = 158 Glane/s/SM | "alu (+ split fmaH)" — straddles |
| A6_PER_PIPE_REFERENCE | 14.13 TIPS_inst @ 1500 = **0.50 inst/SMSP/cy** | "ALU (unified)" |
| B1_DUAL_ISSUE_FFMA_IADD3 | 14.13 TIPS_inst, 17% overlap with FFMA | "ALU pipe; nvcc fuses add;add to single IADD3" |
| **V41_V48_FINDINGS V40** | **25-26 Glane/s = 67% of FMA pipe** | **FMA pipe** (V40 label) |

V40 says IADD3 hits 0.66 inst/SMSP/cy (FMA-pipe peak class); A6/B1 measure 0.50 (half-rate class). The discrepancy may be ILP/warp-count: A6 used 2 warps/SMSP, V40 used a full persistent block. Resolution requires re-running the A6 IADD3 sweep with 4+ warps/SMSP to confirm IADD3 closes to FMA-pipe peak. **Cite V40 for ceiling claims, A6 for per-warp-pressure claims.**

Also: B1 explicitly notes nvcc fuses `add.s32; add.s32` → 1 IADD3, so per-add rate is 2× per-inst rate. The catalog's "2.46 w-inst/SM/cy = 158 Glane/s/SM" headline is the per-add count, not per-inst. The "25% faster than LOP3" claim in 15_integer_bit_ops.md is built on this conflation; the true per-inst gap is V40's 25/18.7 = **1.34× IADD3 over LOP3, not 1.25×**.

### IMAD/IMUL — agreed half-rate of FFMA

| Source | Pipe | Rate |
|---|---|---|
| V8_IMAD_PEAK_VERIFIED | **FMA pipe** (1:2 of FFMA) | 19.18 GIMAD/s = 38.4 TIOPS = 99.7% of true peak |
| V9_INT_OPS_PIPES | "FMA pipe at 49.81%" (1:2 vs FP32) | matches V8 |
| 15_integer_bit_ops.md catalog | "fmaH @ 2.00/SM/cy" | matches V8 |
| V41_V48_FINDINGS V40 | "INT-bit (half rate)" at 18.7 Glane/s | matches V8 in numbers, disagrees in pipe label |

All four agree on ~19 GIMAD/s @ 2032 = half of FFMA. V40's "INT-bit" label and V8/V9's "FMA pipe at 1:2" label point to the same physical fact (whatever you call the half-rate slot). The MEMORY claim "REPORT_06: IMAD on FMA pipe (not INT)" is consistent with all five sources — IMAD lives in the FMA-pipe family.

### PRMT — V40 vs A6 disagreement

| Source | PRMT rate |
|---|---|
| A6_PER_PIPE_REFERENCE | 14.08 TIPS_inst @ 1500 = 0.50/SMSP/cy = ~19 TIPS @ 2032 |
| 15_integer_bit_ops.md | 2.00 w-inst/SM/cy = ~19 TIOPS chip ("alu") |
| **V41_V48_FINDINGS V40** | **13.9 Glane/s = 36% of FMA pipe ("permute" pipe)** |
| V41_V48_FINDINGS V39 raw | 1547% (DCE/LICM artifact, RETRACTED before publication) |

A6 and V40 disagree by ~30%. V40 places PRMT in its own "permute" pipe at 0.36/SMSP/cy; A6 puts it in the same tier as LOP3 at 0.5/SMSP/cy. Likely V40 ran PRMT under different ILP/op-mix conditions; needs an A6-style port-pressure sweep on PRMT to resolve. The V39 1547% was a constant operand hoisted out of the loop (LICM), leaving an empty body — RETIRED before any synthesis cited it.

### ISETP — corrected from "19 TIOPS" to 8.4 Glane/s

| Source | ISETP rate |
|---|---|
| 15_integer_bit_ops.md row 26 | 2.00 w-inst/SM/cy = ~19 TIOPS chip ("pipe_alu") |
| **V41_V48_FINDINGS V40** | **8.4 Glane/s = 22% of FMA pipe ("compare" sub-pipe)** |
| CURIOSITY_LIST_V4 C10 | "ISETP ≈ 4.6 cy/op chained — same magnitude as LOP3" (latency claim) |

V40 measures ISETP at less than half the LOP3 rate (8.4 vs 18.7 Glane/s). C10's "same magnitude as LOP3" is a *latency* claim (both ~4 cy chained) — does not contradict V40's *throughput* claim. The catalog's "all setp variants are equally fast" is correct as a relative claim (FSETP ≈ ISETP at SASS level), but the absolute throughput (~19 TIOPS) is wrong per V40. **Headline value: ISETP = 8.4 Glane/s = 22% of FMA peak**, not the 19 TIOPS the legacy catalog claimed.

### "All ALU at 19 TIOPS" — RETRACTED

`15_integer_bit_ops.md` §Key facts #1 says: "All fast integer ops cap at 2 warp-inst/SM/cy on pipe_alu (~19 TIOPS) — applies to LOP3, PRMT, SHF, IMAD, IMUL, IADD3, ISETP, FSETP, IMNMX, BFI." The V40 ladder explicitly disproves this:
- IADD3: faster than 19 TIOPS (FMA pipe, ~25-26 Glane/s).
- LOP3 / IMUL / SHF: 19 TIOPS tier (correct).
- PRMT: ~14 TIOPS (slower).
- ISETP / FSETP: ~8.4 TIOPS (much slower).

The "19 TIOPS uniform ALU" reading is true ONLY for LOP3/SHF/PRMT-class ops, NOT for IADD3 (faster) or ISETP (slower). RETRACT.

### Mixed-pipe overlap — FFMA + IADD3 = 14-17%, NOT 2× and NOT 131%

| Source | FFMA + IADD3 overlap |
|---|---|
| V9_INT_OPS_PIPES | "up to 114 TOPS combined" (formula prediction) — RETRACTED |
| V9_MIXED_PIPES | 131% pipe-sum, ~74 TOPS effective (corrected from 114) — RETRACTED framing |
| **B1_DUAL_ISSUE_FFMA_IADD3** | **17% overlap** (1.17× speedup vs sequential) |
| **A6_PER_PIPE_REFERENCE** | **14.2% overlap** |

V9's two docs disagreed with each other; V9_MIXED's "74 TOPS effective" differs from B1/A6's 14-17% overlap framing because V9_MIXED counted pipe-utilization-sum (a different metric). **The B1/A6 14-17% number is the right wall-clock speedup headline.** Mixing FFMA + IADD3 gives ~15% benefit, NOT 2× and NOT 50%. The "114 TOPS combined" claim is **FULLY RETRACTED**.

If you want true dual-issue gain on B300, look at FFMA + LDG (memory pipe), FFMA + LDS (shared pipe), FFMA + MUFU (XU pipe). FFMA + IADD3 is essentially same-pipe contention.

### POPC / BREV / CLZ — XU @ 0.125/SMSP/cy

`15_integer_bit_ops.md` says POPC/BREV/CLZ on XU @ 0.5/SM/cy = 4.7 TIOPS chip. A6 confirms: 3.54 TIPS @ 1500 = 0.125/SMSP/cy = 4.7 TIPS @ 2032. **No inconsistencies.** 4× slower than the LOP3 tier — if you find yourself doing many POPCs in a hot loop, consider whether a LOP3-based bit-counting trick fits.

### "shfl.idx with literal 0 src = 85 K Gops/s" — already retired

Original `shfl_bw.cu` sub-agent reported a phantom 85 K Gops/s number for shfl.idx with a literal 0 source. Already retired in catalog: this is **uniform-pipe broadcast** (`R2UR` / `UIMOV`), not a SHFL. The compiler converts `__shfl_sync(mask, x, 0)` with 0 as a compile-time constant to a uniform broadcast, which lives on a different pipe at much higher rate. Do not benchmark "SHFL" using this pattern.

### Summary headline numbers (use these)

At **1920 MHz locked** (= `-lgc 2032` paradox; multiply by 1.058 for 2032 boost):

| Op | Glane/s (chip) | Pipe per V40 |
|---|---:|---|
| FFMA (FMA pipe peak) | 36-38 (97% × 38.5) | FMA |
| IADD3 | 25-26 | FMA (UNRESOLVED #4 with A6) |
| LOP3 / IMUL / IMAD | 18.7 | INT-bit (V40) |
| PRMT | 13.9 | permute (V40) — A6 disagrees |
| ISETP / FSETP | 8.4 | compare (V40) |
| POPC / BREV / CLZ | 4.7 | XU |
| SHFL.IDX | 4.7 | LSU/SHFL |
| EX2 | 9.62 Gops/s | MUFU |
| Other MUFU | 4.74 Gops/s | MUFU |

### Where to mix pipes for true wall-clock speedup (NOT FFMA + IADD3)

If you're trying to hide latency by mixing pipes, FFMA + IADD3 gives only 14-17% gain because they share the FMA-pipe family. Instead, look for:
- **FFMA + LDG**: separate memory pipe; near-100% overlap (but LDG is 15× more energy per op)
- **FFMA + LDS**: separate SMEM pipe; near-100% overlap
- **FFMA + MUFU.EX2**: separate XU pipe; near-100% overlap (use the EX2 anomaly)
- **FFMA + SHFL/REDUX**: separate MIO pipe; near-100% overlap
- **FFMA + LDC (cmem)**: separate cmem cache pipe; near-100% overlap
- **FFMA + PRMT**: separate permute pipe; ~90% overlap
- **FFMA + ISETP**: separate compare pipe; ~95% overlap (if you can find use for the predicate)
- **FFMA + LOP3**: maybe-separate INT-bit pipe; UNRESOLVED whether 50% or 100% overlap

The general rule: ANYTHING that is not in the FMA-pipe family stacks well with FFMA. ANYTHING that is in the FMA-pipe family (IADD3, FADD, FMUL, IMAD/IMUL via half-rate slot) does NOT.

### Per-pipe instruction summary (compact reference)

| Pipe family | Ops | inst/SMSP/cy | Glane/s @ 2032 chip |
|---|---|---:|---:|
| FMA | FFMA, FADD, FMUL, HFMA2, BFMA2, IADD3 (per V40) | 1.0 SoL | 38.5 |
| INT-bit (half-rate FMA family) | LOP3, IMUL, IMAD (.lo), SHF, BFI | 0.5 | 19.3 |
| permute | PRMT | ~0.36 (V40) / 0.5 (A6) | 13.9 |
| compare | ISETP, FSETP | 0.25 | 8.4 |
| MIO/shuffle | SHFL.{IDX,BFLY,UP,DOWN}, REDUX | 0.25 | 9.5 (per element, or 4.7 per inst) |
| XU (transcendental) | MUFU.EX2 (only) | 0.25 (1/4cy) | 9.62 (Gops/s) |
| XU (transcendental) | MUFU.LG2/RCP/RSQRT/SQRT/SIN/COS/TANH | 0.125 (1/8cy effective) | 4.74 (Gops/s) |
| XU (other) | POPC, BREV, CLZ, FLO | 0.125 | 4.7 |
| LSU | LDG, STG | depends | up to ~7 TB/s |
| LDS pipe | LDS.32, LDS.128, STS.* | 1 inst/cy/SM | up to 38.5 TB/s SMEM peak |
| LDC pipe | LDC, LDCU, LDC.U.32 | 1 inst/cy/SM | cmem fast |
| F2FP pipe | cvt.* narrow forms (PACK / UNPACK) | 1 PACK / 2 UNPACK per cy/SM | 19.3 / 38.5 Telem/s |
| TENSOR | mma.sync, wgmma, tcgen05.mma | varies (deferred to §51) | up to 4500 TFLOPS FP8 |
| TMA | cp.async.bulk | 1-2 in flight per CTA | up to 7.2 TB/s pipelined |

### REPORT_06 reference

CLAUDE.md memory mentions: "REPORT_06: 8×8 BASE×COMPANION matrix showing IMAD is on FMA pipe (not INT)". File **not found in `b300_clean/`** during audit. The claim itself ("IMAD on FMA pipe") is already consistent with all five `b300_clean/` files that mention IMAD pipe placement. If REPORT_06 exists in repo root or `investigations/`, it should be folded in but is not load-bearing.

**Footgun:** ⚠ V9 "114 TOPS combined" is FULLY RETRACTED — measured FFMA + IADD3 overlap is 14-17%, NOT 2× and NOT 131% sum. Don't quote "all ALU at 19 TIOPS" — V40 shows tiered ladder from 4.7 (POPC) to 26 (FFMA) Glane/s. Don't bench "SHFL" with `__shfl_sync(mask, x, 0)` — compiles to uniform broadcast at fake 85K Gops/s.

**See also:** §27 (definitive pipe placement table — Section C), §36 (MUFU EX2 anomaly), §38 (SHFL/REDUX shuffle pipe), §40 (packed FP cvt is on its own F2FP pipe), corrections/INT_INCONSISTENCY_LOG.md.

---

## §40. Packed FP cvt — output bit-width hypothesis (FP8 cvt 2.0× faster than BF16/F16 cvt)

**Answer:** **`cvt.rn.satfinite.{e4m3,e5m2}x2.f32` measures 17.6 Gelem/s; `cvt.rn.{bf16,f16}x2.f32` measures 9.05 Gelem/s** — FP8 packed cvt is exactly 2.0× faster than BF16/F16 packed cvt at the per-element level. Per-SASS-instruction throughput is identical (same F2FP pipe, same dispatch slot); the gap is per-PTX-instruction because narrower outputs (FP8) compile to `PACK_AB` while BF16/F16 require `PACK_AB_MERGE_C` which halves the rate. CUDA 13.2 has a separate **PTX-rejection BUG**: `cvt.rn.satfinite.e2m1x4.f32` is rejected on sm_103a despite being valid in CUDA 12.x — workaround via 2× x2 forms or scalefactor variant. `[🟢 HIGH for the elem/s numbers (V43); 🟡 MED for the PACK_AB-vs-MERGE_C mechanism (hypothesis, SASS not yet dumped) · src: V41_V48_FINDINGS.md§"Packed FP cvt", 05_fp_precision_nontensor_CORRECTED.md§B-D]`

### Measured rates (V43, partial — F2FP pipe)

| Source | Dest | PTX form | Measured Gelem/s | F2FP-pipe theoretical (PACK) | Notes |
|---|---|---|---:|---:|---|
| FP32 → FP8 (E4M3) | packed x2 | `cvt.rn.satfinite.e4m3x2.f32` | **17.6** | 19.3 Telem/s | hits ~91% of pipe SoL |
| FP32 → FP8 (E5M2) | packed x2 | `cvt.rn.satfinite.e5m2x2.f32` | **17.6** | 19.3 Telem/s | identical to E4M3 |
| FP32 → BF16 | packed x2 | `cvt.rn.bf16x2.f32` | **9.05** | 19.3 Telem/s | half of FP8 |
| FP32 → FP16 | packed x2 | `cvt.rn.satfinite.f16x2.f32` | **9.05** | 19.3 Telem/s | identical to BF16 |

F2FP pipe theoretical: 32 inst/SM/clk PACK = 19.3 Telem/s chip. FP8 hits 91% of this; BF16/F16 hit 47%.

### Per-SASS-instruction vs per-PTX-instruction

The original `05_fp_precision_nontensor.md` sec 2.3 states: "all formats hit identical per-instruction throughput within direction (UNPACK or PACK); FP4 is NOT slower or faster than FP8 per SASS instruction on this pipe." This is **correct at the SASS-opcode level**: one `F2FP.*.PACK_AB.*` SASS instruction has the same dispatch cost regardless of dest narrow-format.

V43's chip-level measurement at the **PTX level** disagrees: FP8 cvt at 17.6 vs BF16/F16 at 9.05 = 2.0×. These are not in conflict — both are correct at their own level:

- **Per-SASS-instruction:** same. (Catalog correct.)
- **Per-PTX-instruction:** FP8 packs 2 elements per F2FP instruction (PACK_AB only). BF16/F16 also pack 2 per `cvt.rn.bf16x2.f32` AT THE PTX LEVEL but lower to the **PACK_AB_MERGE_C** variant which V43's harness measures at half the rate.
- **Hypothesis (V43):** output bit-width matters for the F2FP MERGE_C step. Narrower outputs (FP8 = 8 bits) skip MERGE_C; wider narrow outputs (BF16/F16 = 16 bits) require MERGE_C. Not yet SASS-verified across all 4 forms.

ACTION pending: dump SASS of all 4 PTX forms to confirm whether BF16/F16 paths really emit `PACK_AB_MERGE_C` while FP8 paths emit `PACK_AB` (no MERGE_C). If catalog sec 2.3's implicit claim that all 4 emit MERGE_C holds, the V43 elem/s gap is unexplained. Until SASS dump: cite "FP8 cvt 2× BF16 cvt at PTX level" as the headline; cite "per-SASS-inst rates equal" as the deeper truth; flag the mechanism as MED-confidence hypothesis.

### F2FP pipe theoretical (from F2FP_DEEP_DIVE)

The F2FP pipe is separate from the FMA pipe and has these SASS-level theoretical peaks:

| Direction | inst/SM/clk | Gelem/s chip @ 2032 | Notes |
|---|---:|---:|---|
| UNPACK (narrow → FP32) | 64 | **38.5** | per element, all narrow formats |
| PACK (FP32 → narrow) | 32 | **19.3** | per element, all narrow formats |

Identical per-instruction rate across FP8 / FP6 / FP4 narrow formats at the SASS opcode level. The PTX-level Gelem/s gap (FP8 vs BF16) emerges from the MERGE_C dispatch.

### `cvt.rn.satfinite.f16x2.f32` MUST be `.satfinite`

Without `.satfinite`, the cvt.rn.f16.f32 path goes to a separate **F2F pipe at 11/SM/clk** = ~3-6× slower. Catalog `05_fp_precision_nontensor.md` HIGH-confidence rule. ALWAYS include `.satfinite` on the FP32 → FP16 cvt unless you specifically need overflow-to-Inf behavior (rare).

### CUDA 13.2 PTX rejection bug — narrow x4 cvt forms

V41_V48_FINDINGS l.61-62 + V6 H3 commit `3bb7051`:
```
cvt.rn.satfinite.e2m1x4.f32   ← REJECTED on sm_103a in CUDA 13.2
```
This PTX form was VALID in CUDA 12.x but no longer compiles in NVRTC under CUDA 13.2. Diagnosis: probably a PTX syntax migration where x4 narrow forms now require either:
- the `cvt.rn.satfinite.relu.e2m1x4.f32` variant, or
- the scalefactor variant (`cvt.rn.satfinite.e2m1x4.scale.f32`), or
- explicit 2× x2 forms manually packed into a register.

Workaround for now: use 2× `cvt.rn.satfinite.e2m1x2.f32` and pack manually with PRMT. This carries a minor latency penalty but is safe.

Open question: is this a sm_103a-only bug or all-arch in CUDA 13.2? V41 only tested sm_103a; cross-check on sm_100a / sm_90a needed to isolate. If it is sm_103a-only, file with NVIDIA as a CUDA 13.2 regression.

### Other survival items from `05_fp_precision_nontensor_CORRECTED.md`

(Inherited HIGH-confidence facts that are still valid):

1. **No FP16/BF16 packed FMA speedup over FP32 outside tensor cores.** Scalar FFMA, HFMA2 and BFMA2 all peak at the same ~70-72 chip-TFLOPS via pipe_fma. (B300_TRUE_REFERENCE §7 surprise #2; commit `ea47ec6`.)
2. **FP64 DFMA = 1.20 TFLOPS** at 2032 MHz, 4 warps/SM (commit `2d64696`).
3. **F2FP narrow UNPACK = 64 inst/SM/clk = 38.5 Telem/s; PACK = 32 inst/SM/clk = 19.3 Telem/s.** Identical per-instruction rate across FP8 / FP6 / FP4.
4. **HMNMX2 (`min/max.f16x2`) lives on pipe_alu**, can co-issue with FFMA.
5. **FMUL = FADD = FFMA at SASS level.** All on FMA pipe, all 4.04-4.22 cy chain latency, all 1 inst/SMSP/cy. FFMA wins TFLOPS only because each does 2 FLOPS not 1. (V8_FADD_FMUL_PEAK / V9_OP_LATENCY)

| Op | SASS | Latency | Peak inst/SM/cy | Peak chip TFLOPS | FLOPS/inst |
|---|---|---:|---:|---:|---:|
| FADD | `FADD R, R, R` | 4.22 cy | 4.0 (full FMA pipe) | 37.4 | 1 |
| FMUL | `FMUL R, R, R` | 4.22 cy | 4.0 (full FMA pipe) | 37.3 | 1 |
| FFMA | `FFMA R, R, R, R` | 4.22 cy | 4.0 (full FMA pipe) | 74.8 | 2 |

### Practical implications

For FP8 inference (NVFP4 / E4M3 / E5M2 quantized weights), the cvt step is rarely the bottleneck — it is one cvt per HBM line, dominated by the load + tensor-core cost. But:
- **Quantization-on-the-fly kernels** (FP32 activations → FP8 packed for tcgen05.mma input) can become cvt-bound if the matrix dim is small. At 17.6 Gelem/s chip × 1 byte/elem = 17.6 GB/s of FP8 produced — much less than HBM BW.
- **For BF16 quantization-aware-training**, the BF16 output cvt at 9.05 Gelem/s × 2 bytes = 18.1 GB/s. Same regime.
- If you need full HBM BW of FP8 packed output, you need ~7 TB/s ÷ 1 = 7 GB ops/s per byte — well above 17.6 G. Multi-warp / multi-block fan-out is required to keep cvt off the critical path.

### Pre-V41 catalog "all narrow-format cvts equal" — partial truth

The pre-V41 catalog `05_fp_precision_nontensor.md` sec 2.3 said "all formats hit identical per-instruction throughput within direction (UNPACK or PACK); FP4 is NOT slower or faster than FP8 per SASS instruction on this pipe." This is correct AT THE SASS LEVEL — one F2FP.PACK_AB instruction issues at the same rate regardless of dest format.

What changes between formats is what HAPPENS at the PTX-to-SASS lowering:
- FP32 → FP8 e4m3x2 (PACK 2 elements, 8 bits each, total 16 bits): emits `F2FP.E4M3.PACK_AB`. Single inst, no MERGE_C step.
- FP32 → BF16x2 (PACK 2 elements, 16 bits each, total 32 bits): emits `F2FP.BF16.PACK_AB.MERGE_C`. The MERGE_C step adds ~one cycle of pipeline pressure.
- FP32 → FP4 e2m1x4 (PACK 4 elements, 4 bits each, total 16 bits): emits `F2FP.E2M1.PACK_AB.RS` for the stochastic-round form. Two SASS for the rejected `cvt.rn.satfinite.e2m1x4.f32`.

V43's measurement at the PTX/element level naturally captures the MERGE_C overhead in the BF16/F16 path. **Catalog and V43 agree at their respective levels.** Both should be cited together for a complete picture.

### F2FP_DEEP_DIVE 33-result table (referenced)

`F2FP_DEEP_DIVE.md` (referenced in 05_fp_precision_nontensor sec 4) is a 33-row test of every PTX cvt form across narrow → narrow, narrow → FP32, FP32 → narrow, with rounding and saturation variants. Original sec 4 marks it "strongest single document". Not independently re-audited here — trusted on author's prior verification. Headline rates:
- UNPACK = 64 SASS-inst/SM/clk = 38.5 Telem/s chip
- PACK = 32 SASS-inst/SM/clk = 19.3 Telem/s chip
- All narrow formats (FP8 / FP6 / FP4 / e2m1 / e3m2 / e4m3 / e5m2) run at the same SASS-inst rate.
- PTX-level rates differ when MERGE_C is required (BF16/F16 PACK direction).

### CUDA 13.2 PTX migration — workarounds for narrow x4 forms

If you are migrating from CUDA 12.x to CUDA 13.2 and hit the `cvt.rn.satfinite.e2m1x4.f32` rejection on sm_103a, options are:

1. **Use 2× x2 forms manually packed:**
```cpp
__device__ uint16_t cvt_rn_satfinite_e2m1x4(float a0, float a1, float a2, float a3) {
    uint8_t pack01;
    uint8_t pack23;
    asm("cvt.rn.satfinite.e2m1x2.f32 %0, %1, %2;" : "=h"(pack01) : "f"(a0), "f"(a1));
    asm("cvt.rn.satfinite.e2m1x2.f32 %0, %1, %2;" : "=h"(pack23) : "f"(a2), "f"(a3));
    return (uint16_t)pack01 | ((uint16_t)pack23 << 8);
}
```
2. **Use the scalefactor variant** (if you have a UE8M0/E4M3 scale register):
```cpp
asm("cvt.rn.satfinite.e2m1x4.f32.scale %0, %1, %2, %3, %4, {%5};" :
    "=r"(packed4) : "f"(a0), "f"(a1), "f"(a2), "f"(a3), "h"(scale_e8m0));
```
3. **Wait for CUDA 13.3+** — the ISA is supposed to support the bare form, this seems to be a regression.

The 2-instruction workaround pays one extra F2FP cycle per 4 elements but lets you keep the algorithm structure. Not a major perf hit unless cvt is the bottleneck.

### Why output bit-width matters (V43 hypothesis)

For PACK direction:
- FP8x2 = 8 + 8 = 16 bits to pack into a 32-bit register half. F2FP fits this in one cycle without MERGE_C.
- BF16x2 = 16 + 16 = 32 bits, fills the entire register. F2FP must MERGE the two halves (the C step combines/aligns), adding pipeline pressure.
- FP4x4 = 4 × 4 = 16 bits. Like FP8x2, fits in one cycle without MERGE_C — should be at the 17.6 Gelem/s rate per V43's hypothesis (NOT V43-measured because the form is rejected).
- FP6x4 = 4 × 6 = 24 bits. Likely needs MERGE_C — should be at 9.05 Gelem/s rate per hypothesis (UNTESTED).

If V43's hypothesis holds, the rate ladder is:
- 16-bit total output (FP8x2, FP4x4, e3m2x2): 17.6 Gelem/s (PACK_AB only)
- 24-bit total output (FP6x4, e2m1x4 if it compiled): 9.05 Gelem/s (PACK_AB_MERGE_C)
- 32-bit total output (BF16x2, F16x2): 9.05 Gelem/s (PACK_AB_MERGE_C)

To confirm: dump SASS of all 7 PTX forms above and check for presence/absence of MERGE_C suffix. Open work item.

### Per-element throughput vs HBM bandwidth ratio

For FP8 cvt of HBM-loaded data:
- HBM read peak: 7.2 TB/s = 7.2e12 bytes/s
- FP8 elements per byte: 1
- FP8 cvt rate: 17.6 Gelem/s = 17.6 GB/s OUT
- Ratio: 7200 / 17.6 = **409× HBM > cvt**

Even at 17.6 Gelem/s, the F2FP pipe is tiny relative to HBM. So FP8 cvt is rarely an HBM-bottleneck issue. It IS a bottleneck for kernels that:
- Do many cvt per loaded element (e.g., FP32 → FP8 → FP6 → FP4 chained quantization)
- Have small matrix dims where cvt warm-up dominates
- Need extreme dispatch density (e.g., per-thread cvt in a loop)

For most production GEMM-with-cvt kernels, cvt is not the bottleneck.

### Stochastic rounding (RS) variants

For training kernels using stochastic rounding (NVFP4 scaled GEMM):
```
cvt.rs.satfinite.e2m1x4.f32  (rs = round-stochastic)
```
Per F2FP_DEEP_DIVE the RS variants run at the same SASS-inst rate as RN (round-nearest), but consume a uniform random source from the URF. The compiler emits a `cvt.rs` SASS that reads from `URand32` register. Each CTA must initialize the URand32 source via a setup PTX instruction.

Throughput: same as RN variants per V43 (17.6 Gelem/s for FP8x2 RS; 9.05 for BF16/F16x2 RS). The URF random source does NOT throttle the F2FP pipe.

For training: prefer RS over RN to avoid systematic bias in low-precision quantization. RS has no perf cost vs RN.

### Where the BF16/F16 rate gap matters in practice

Most production ML inference codebases use:
- FP8 cvt for quantization-on-the-fly (FP32 activations → FP8 for tcgen05.mma)
- BF16 cvt for output storage (FP32 accumulator → BF16 for next layer's input)

If your kernel does:
- 1 BF16 cvt per output element → bottlenecked at 9.05 Gelem/s × 2 bytes = 18.1 GB/s
- HBM peak: 7200 GB/s
- Ratio: HBM/cvt = 397×

So even at the 2× slower BF16 rate, cvt is rarely the bottleneck for HBM-resident output writes. It IS a bottleneck if you have fully-cvt-bound code (e.g., fp32-to-bf16 conversion of an entire tensor without other work).

### cvt energy cost

Per M2 H1 / M11: cvt narrow forms cost ~5 pJ/op (similar to LDS u32). At 17.6 Gelem/s × 5 pJ/elem = 88 mW per chip on cvt. Negligible vs 1100 W TDP. Cvt is energy-cheap.

### Mixed precision strategy summary

| Workload pattern | Recommended cvt strategy |
|---|---|
| FP32 activations → FP8 for mma input | Use `cvt.rn.satfinite.e4m3x2.f32` (17.6 Gelem/s) |
| FP32 accumulator → BF16 storage | Use `cvt.rn.bf16x2.f32` (9.05 Gelem/s) — accept 2× cvt cost since not bottleneck |
| FP32 → FP4 quantization | WORKAROUND: 2× `cvt.rn.satfinite.e2m1x2.f32` until CUDA 13.3+ |
| Stochastic rounding (training) | Use `cvt.rs.*` variants — same throughput as RN |
| FP32 → FP16 with overflow-to-Inf | OMIT `.satfinite` — slower (3-6× via F2F pipe) but correct |
| FP32 → FP16 with saturation | INCLUDE `.satfinite` — fast F2FP pipe |

**Footgun:** ⚠ The "FP8 cvt 2× BF16 cvt" framing is per-PTX-element, not per-SASS-instruction; per-inst rates are equal (same F2FP pipe). When budgeting kernel time, use the **per-PTX-element** rate (17.6 vs 9.05 Gelem/s); when reasoning about pipe contention, use the **per-SASS-inst** rate (both ~19.3 Telem/s in PACK direction). Don't conflate. Also: CUDA 13.2 rejects `cvt.rn.satfinite.e2m1x4.f32` on sm_103a — use 2× x2 forms as workaround.

**See also:** §36 (MUFU per-op rates), §39 (INT/bit pipe ladder), corrections/05_fp_precision_nontensor_CORRECTED.md, V41_V48_FINDINGS.md §"Packed FP cvt", F2FP_DEEP_DIVE.md.

---

## §41. Power floor / ceiling — TDP 1100 W enforced; idle 144-198 W (clock-dependent)

**Answer:** **TDP = 1100 W enforced** (`nvmlDeviceGetEnforcedPowerLimit`). Idle floor varies **144-198 W** with clock state (NOT "165-170 regardless"). True idle = 120 MHz / 144 W; default boost = 2032 MHz / 198 W idle. Stress recipe for sustained max power: **DRAM read d=16 random + 1500 MHz lock = 1071 W** (just under TDP). At 1800 MHz TDP cap clips d=8..28 popcount data to ~1092-1099 W. NEVER throttled in any tested workload at default boost — but B300 sticks at 1005 MHz silently if leftover procs / thermal. Transient peaks to 1259 W reported but contested. `[🟢 HIGH for TDP=1100 and stress recipe; 🟡 MED for 1259 W transient claim · src: 16_power_clock_CORRECTED.md, POWER_INCONSISTENCY_LOG.md§A-B, POPCOUNT_VS_CLOCK.md]`

### Idle floor — varies with clock (NOT a single number)

The legacy "165-170 W regardless of utilization" claim from `M11_PER_PIPE_ENERGY.md` is misleading — it fixed at 1500 MHz lock and did not sweep. Cross-source audit (`POWER_INCONSISTENCY_LOG.md` §A) confirms idle scales with clock due to leakage at higher voltage:

| Clock state | Reported clock | Idle floor (W) | Source |
|---|---|---:|---|
| True idle (no kernel, GPU asleep) | 120 MHz | **144** (POWER_FREQUENCY_CURVE) / 164.7 (M2 H6 @ 1500 lock) | conditional on lock state |
| `nvidia-smi -lgc 510` | 510 | **144** | V10 / POWER_FREQUENCY_CURVE |
| `nvidia-smi -lgc 800` | 800 | **147** | V10 |
| `nvidia-smi -lgc 1005` | 1005 | **150-152** | POWER_DATA_DEPENDENCE_SUMMARY |
| `nvidia-smi -lgc 1300` | 1300 | **158** | V10 |
| `nvidia-smi -lgc 1500` | 1500 | **167** | V10 / M2 / POWER_FREQUENCY_CURVE |
| `nvidia-smi -lgc 1700` | 1700 | **175** | V10 |
| `nvidia-smi -lgc 2032` (= 1920 actual) | 1920 | **198** | V10 |
| Default boost (`-rgc`, no lock) | 2032 actual | **198** | V10 |

**Always quote idle WITH the clock state.** A bare "165 W idle" implicitly assumes 1500 MHz lock; "150 W idle" assumes 1005 MHz; etc.

### Power floor + ceiling at each clock (full table)

(All in W, sustained ≥3 s, TDP limit = 1100 W)

| Clock (MHz) | Idle floor | Active min (tcgen05 A=B=0) | FFMA active | DRAM-read random d=16 | TDP-cap reached? |
|---:|---:|---:|---:|---:|:---:|
| 510  | 144 | ~155 (extrapolated) | 178 (FFMA Δ34) | ~553 | NO |
| 800  | 147 | — | 202 (Δ54) | 631 | NO |
| 1005 | 150-152 | **287** (148 SMs, 1 W/SM) | 225 (Δ73) / 613 random BF16 GEMM | **787** | NO |
| 1300 | 158 | — | 254 (Δ96) | 942 | NO |
| 1500 | 167 | — | 300 (Δ133) / 1009 random BF16 | **1071** | **APPROACHED** |
| 1700 | 175 | — | 339 | — | YES (random BW) |
| 1800 | — | — | — | **1092** (clipped) | **YES (clipped)** |
| 1920 (=`-lgc 2032`) | 198 | — | 419 (Δ222) / 1099 random BF16 | TDP-cap | YES |
| 2032 (true boost, `-rgc`) | 198 | — | 361 (peak ILP=24) / 437 (low-occ) | — | YES (under cuBLAS BF16 962W) |

Notes:
- **DRAM read d=16 random + 1500 MHz = 1071 W** — sustained worst-case thermal stress recipe. CONFIRMED in 2 files (POPCOUNT_VS_CLOCK, POWER_DATA_DEPENDENCE_SUMMARY).
- **DRAM read d=8..28 + 1800 MHz = 1100 W TDP wall** — bell flat-topped (cap clips ~150 W off the natural curve at d=16).
- **FFMA peak (boost, ILP=24, 256 thr) = 361 W**, surprisingly LESS than low-occupancy FFMA (437 W). This is UNRESOLVED — hypothesis is leakage at idle SMSPs in low-occ wastes lanes. Needs ncu correlation.
- **Random-data BF16 cuBLAS = 962 W sustained** with no throttle (87% TDP).

### TDP cap reconciliation

| File | TDP value | Notes |
|---|---:|---|
| `16_power_clock.md` | 1100 W (min 200 / max 1100 / default 1100) | nvmlDeviceGetEnforcedPowerLimit |
| `POWER_DATA_DEPENDENCE_SUMMARY.md` | 1100 W | Consistent |
| `POPCOUNT_VS_CLOCK.md` | 1100 W; observed clip at 1092-1099 W | Bell flat-tops above |
| `B300_TRUE_REFERENCE.md` | "TDP 1100 W (sustained avg ceiling 1093 W; transient peaks to 1259 W)" | **1259 W transient claim — contested** |
| CLAUDE.md memory | 1100 W TDP | Consistent |
| (legacy / old catalog) | "700 W" | Hopper carry-over — RETIRED |

**The 1259 W transient claim is contested.** Either the 1100 W enforced limit is a sustained-average soft cap with millisecond-scale transients allowed, or the 1259 value is a sample-aliasing artifact in NVML's 33 Hz max sample rate. UNRESOLVED — needs corroboration with high-rate (kHz+) power probe. In meantime, headline TDP = 1100 W.

### "B300 TDP = 700 W" — RETRACTED

Multiple early notes carry "700 W TDP" as a Hopper carry-over. **WRONG** for B300 SXM6 AC. The default enforced limit is 1100 W per nvml. Hopper H100 SXM5 was 700 W; B300 doubles that envelope.

### "B300 throttles to 53% under sustained FP8" — RETRACTED

Early measurement artifact (no warmup). True sustained FP8 GEMM is **flat at 4491 TFLOPS for 30+ s** (B300_TRUE_REFERENCE). Not a throttling result.

### "B300 TDP not approached under any workload (~339 W max)" — RETRACTED

Stale (FFMA-only). Tensor + cuBLAS hit 411-962 W; sustained BF16 GEMM = 962 W (87% TDP). Random-data DRAM-read bell curves do hit 1071-1100 W. **B300 absolutely DOES approach TDP under realistic workloads.**

### Min power limit + range

`nvmlDeviceGetPowerManagementLimitConstraints`:
- **Min: 200 W**
- **Max: 1100 W**
- **Default: 1100 W**

You can set `-pl <W>` between 200 and 1100 to throttle. Note this is different from clock locking; setting `-pl 500` will downclock to whatever frequency keeps power ≤ 500 W (typically ~1005 MHz under FFMA load).

### NEVER throttled in any tested workload at default boost — qualifier needed

`16_power_clock.md` says "NEVER throttled in any tested workload" at default boost. `B300_TRUE_REFERENCE.md` line 16 says boost "rarely sustained". These are direct contradictions. Reconciliation per CLAUDE.md memory:

- Default boost IS 2032 MHz in clean tests (no contending procs, fresh GPU state).
- Background processes (or silent throttle conditions) can pin clock to 1005 MHz with NO `-lgc`-applied lock.
- `nvidia-smi -q -d CLOCK` shows "Application Clocks Setting: 2032 MHz" but the **actual clock under load** is 1005 MHz. The "Idle: Active" performance reason flag does not reliably reflect this.

**Always sample clock during long runs** (see §45). The `16_power_clock.md` "NEVER throttled" claim is correct in the steady-state-without-stuck regime; the `B300_TRUE_REFERENCE.md` "rarely sustained" is correct in the production-without-cleanup regime.

### Stress recipe — DRAM read d=16 random + 1500 MHz = 1071 W

The single highest sustained-power workload measured on B300 SXM6 AC:

```bash
nvidia-smi -lgc 1500
# Run bench_pwr_dram_popcount64.cu with d=16 random per-dword data, 8 GB ws,
# 60M+ iterations to ensure ≥6 s sustained.
```

Yields **921 W active + 150 W idle = 1071 W total** (just under TDP). At 1800 MHz the bell curve flat-tops at 1092 W (TDP-clipped), so 1500 MHz is the highest clock that gives an unclipped DRAM popcount measurement.

This recipe is useful for:
- Thermal stress testing (find hot-spot)
- Validating cooling
- Worst-case datacenter power planning (a rack of B300s under this load scales linearly per GPU)

For ML inference where weights are FP4/FP8 quantized, many tensor elements have low popcount (high bits of mantissa-only values are often 0). A model that loads "mostly zero" data through DRAM will burn ~240 W less than synthetic d=16 — this is the "real production weights" prediction in `16_power_clock_CORRECTED.md` UNRESOLVED #10, not yet directly verified.

### tcgen05.mma power floor — 287 W absolute minimum

`POWER_FLOOR.md` measures the absolute floor for an active multiplier (148 SMs, all active, A=B=0 trivially): **287 W at 1005 MHz** (= 150 W idle + 1 W/SM × 148). This is the irreducible cost of having all SMs alive with the multiplier engaged but no useful data.

| Pattern | Power (W) at 1005 MHz | Notes |
|---|---:|---|
| Idle GPU | 150 | baseline |
| A=0 AND B=0 (mode 1800) | **287** | absolute floor for active multiplier |
| A=0, B=rand | 491 | A broadcast contributes little |
| A=rand, B=0 (mode 300) | 296 | B=0 fully gates multiplier |
| A=const +1.0, B=const +1.0 (Tier B) | **299** | static baseline |
| Inf/NaN constant (Tier C) | 308 | +9 W detector overhead |
| Random A & B (full random) | **609** | +310 W data-dependent cost |

(Detail in §51 Section E — tensor/tcgen05 power per CTA.)

### Idle ladder — different states (M2 H6+H7+R2)

| State | Power | Δ above true idle | Source |
|---|---:|---:|---|
| GPU true idle (no kernel) | 164.7 W | 0 | M2 |
| GPU + 1 SM kernel "alive" | 165.4 W | +0.7 W | H6 |
| GPU + all 148 SMs spinning (no work) | 172.0 W | +7.3 W = 0.05 W/SM | H6 |
| GPU + 148 SMs allocated TMEM (idle) | 172.0 W | +7.3 W (no extra) | H7 |
| GPU + 148 SMs in mbarrier.try_wait | 171.6 W | +6.9 W | R2 |
| GPU + 148 SMs spinning on managed flag | 173.1 W | +8.4 W | R2 |
| GPU + 148 SMs in __syncthreads loop | 173.2 W | +8.5 W | R2 |

Key: **TMEM allocation has ZERO power overhead.** mbarrier.try_wait saves ~25% power vs spin (4.3 vs 5.8 W delta on 148 SMs).

### Persistent kernel power — close to idle

For persistent kernels that spin waiting for work:
- mbarrier.try_wait spin: +6.9 W on 148 SMs above idle = 0.047 W/SM/spin
- Spin on managed flag: +8.4 W = 0.057 W/SM/spin
- __syncthreads loop: +8.5 W = 0.057 W/SM/spin

For 148 SMs idle with a persistent kernel pattern, expect ~150 + 7 = 157 W total. This is essentially "free" — barely above true idle.

For multi-GPU coordination where you need a GPU thread waiting for cross-GPU signal:
- Use mbarrier.try_wait (4.3 W per 148 SMs delta over idle, lowest power)
- Or just exit the kernel and re-launch from host (kernel launch is 2µs; spin-waits add up)

### Power-cap behavior — `nvidia-smi -pl <W>`

You can cap power at any value between 200 W and 1100 W via `nvidia-smi -pl <W>`. Behavior:
- Cap is enforced as an instantaneous limit (not a sustained-average soft cap)
- When workload demands exceed cap, GPU auto-throttles clock to keep power ≤ cap
- Throttling typically targets the SM clock (drops from 2032 → 1500 → 1300 → 1005 MHz as needed)
- Reset with `-pl 1100` (or `-pm 0` to remove power management)

Useful when:
- You need predictable power (datacenter scheduling)
- You want to compare two workloads at identical power budget
- You want to validate a thermal envelope

Caveat: `-pl <X>` may interact with `-lgc <Y>` in complex ways; if you specify both, the more restrictive wins. Use one or the other for clarity.

### Why FFMA non-peak draws MORE power than peak (UNRESOLVED)

Per `16_power_clock_CORRECTED.md` UNRESOLVED #2: FFMA peak (boost, ILP=24, 256 thr) = 361 W, FFMA low-occ = 437 W. Counterintuitive — fewer threads should mean less work, less power.

Plausible mechanisms:
1. **SMSP leakage at idle lanes.** When ILP is low, the FMA pipe is fed only ~25% of cycles; the unused cycles still pay leakage on idle SMSP scoreboard / queue / fetch logic.
2. **Higher voltage at low utilization.** GPU may DVS-up voltage when it sees "low active power, room to run faster" → wastes voltage on idle lanes.
3. **Compiler artifact.** Low-ILP code may have more synchronization / scoreboard waits, which keeps the dispatcher hot without producing useful work.

Needs ncu correlation to fix. For now: assume "low-occupancy FFMA = 437 W, high-occupancy = 361 W" is real and let it inform your kernel design (high occupancy is BOTH faster AND lower power).

### Multi-GPU TDP coupling — UNTESTED

For 2× B300 NV18 system: each GPU can draw 962 W (sustained BF16 cuBLAS). Total = 1924 W. At 1100 W TDP each, total = 2200 W. Chassis power supplies typically rated 3200-3500 W for 2-GPU systems. So no chassis-level cap should kick in.

But — UNTESTED. If you're running a sustained 2-GPU FP8 cuBLAS workload, sample chassis-level power (BMC, IPMI) to confirm. The B300_TRUE_REFERENCE chassis description does not include power-coupling tests.

### True idle — what does "idle" mean?

NVML's "idle" power is the floor power when no kernel is running but the application has CUDA context active. It includes:
- HBM3E refresh power
- L2 refresh / coherence
- Idle SM clock (some scoreboard activity even with no kernels)
- PCIe link state
- NVLink link state (always-on power)

True device sleep (powered off) would be much lower (10s of W), but is rarely useful in datacenter context.

### Idle clock = 120 MHz vs idle floor power

Idle clock state = 120 MHz when no kernel has run for several seconds AND no application has CUDA context. After a kernel runs, idle clock floats up to whatever the last clock state was. So:
- Cold start (no app): 120 MHz / ~144 W
- After 1500 MHz kernel run, app still alive: 1500 MHz / ~167 W
- After kernel exits + app exits + sleep: 120 MHz / ~144 W

This is why "idle floor" varies in measurements — it depends on the recent clock history. Always sample idle BEFORE the test, not after.

### Open questions (from 16_power_clock_CORRECTED §UNRESOLVED)

1. **Multi-minute / hour-scale sustained load behavior.** All tests 12-60 s. Whether B300 throttles under genuinely-long pure-compute load (e.g. 1 hour of FP8 cuBLAS at 886 W) — not tested.
2. **Why does FFMA non-peak draw MORE power (437 W) than FFMA peak (361 W)?** Hypothesis: idle SMSP leakage + low ILP wastes lanes.
3. **Multi-GPU TDP coupling** — does chassis power cap kick in when both B300s draw 962 W? Untested.
4. **TDP-ceiling vs transient peak.** B300_TRUE_REFERENCE cites "transient peaks to 1259 W" not corroborated. Either NVML enforced limit is briefly exceedable, or 1259 is measurement transient.
5. **Real production weight tensors** — predicted -120 to -180 W vs synthetic d=16 not directly verified.

### Power × clock × workload matrix

A combined view of where the GPU sits in power space across clock × workload space:

| Workload | 510 MHz | 1005 MHz | 1500 MHz | 1700 MHz | 1920 MHz | 2032 boost |
|---|---:|---:|---:|---:|---:|---:|
| Idle (no kernel) | 144 | 152 | 167 | 175 | 198 | 198 |
| Spin loop (148 SMs) | ~155 | ~170 | ~190 | ~200 | ~225 | ~225 |
| FFMA peak (low-occ) | 178 | 225 | 300 | 339 | 419 | 437 |
| FFMA peak (high-occ ILP=24) | — | — | — | — | — | 361 |
| BF16 mma random | 350 | 613 | 1009 | — | 1099 (capped) | 1099 (capped) |
| BF16 mma constant | 259 | 296 | 426 | — | 629 | 629 |
| FP8 cuBLAS sustained | — | — | — | — | — | 886 |
| HBM streaming d=16 | ~553 | 787 | 1071 | TDP cap | TDP cap | TDP cap |
| HBM streaming d=0 | — | 547 | 721 | — | — | — |
| Random d=16 + DRAM read | ~553 | 787 | 1071 | TDP cap | TDP cap | TDP cap |
| L2-warm random d=16 | — | 555 | — | — | — | — |
| tcgen05 random BF16 | 350 | 613 | 1009 | — | 1099 | 1099 |
| tcgen05 const A=B=0 | — | 287 | — | — | — | — |
| L2 read d=16 | — | 405 (active) | — | — | 771 (active) | — |
| L2 write d=16 | — | 235 (active) | — | — | 557 (active) | — |
| DRAM-8G read d=16 | — | 637 (active) | 921 | 943 (capped) | 942 (capped) | — |
| DRAM-8G write d=16 | — | 405 (active) | — | — | 830 (active) | — |

(Active = above ~150 W idle baseline; total = active + idle; TDP cap = clipped at ~1100 W)

This table is the load-bearing chart for thermal planning, datacenter power budgeting, and kernel-energy estimation. Note the asymmetry: FFMA is much lower power (~360 W) than DRAM-bound work (~700-1100 W). Compute is energy-cheap; memory is energy-expensive. Cache-blocking saves both time AND energy.

**Footgun:** ⚠ Don't quote a single "idle = 165 W" number — idle varies 144-198 W with clock state. Don't claim "B300 TDP = 700 W" (Hopper carry-over). Don't trust "NEVER throttled" without checking clock during run; B300 silently sticks at 1005 MHz under leftover-proc thrashing. Use `nvmlDeviceGetEnforcedPowerLimit` for the true TDP.

**See also:** §42 (DVS V² scaling), §43 (data-dependence popcount bell), §45 (clock-lock paradox + stuck-at-1005), §51 (tensor power per CTA — Section E), corrections/POWER_INCONSISTENCY_LOG.md, V10_DVS_CURVE.md, POWER_FREQUENCY_CURVE.md.

---

## §42. Power vs clock — DVS V² scaling above 1500 MHz; min-energy clock is metric-DEPENDENT

**Answer:** B300 follows **CMOS V² × f scaling above ~1500 MHz**. Below ~1005 MHz, idle/static dominates (per-op energy goes UP as clock drops). The min-energy clock is **workload-DEPENDENT, not constant**: pure FFMA → 510 MHz min; pure memory → 800 MHz min; mixed ML inference → **boost clock (1992-2032 MHz) is 3.08× lower energy than 510 MHz** per M9. **For ML inference USE BOOST CLOCK** — DVFS down-clocking ML inference INCREASES total energy. The legacy "lower clock is more efficient" intuition is WRONG for B300 mixed workloads. `[🟢 HIGH for the V²×f scaling; 🟢 HIGH for the workload-dependent min-energy claim · src: 16_power_clock_CORRECTED.md§7, V10_DVS_CURVE.md, M9_ENERGY_PARETO.md, M2_ENERGY_LADDER.md, POWER_INCONSISTENCY_LOG.md§I]`

### V10 DVS curve (FFMA-saturated workload, V6 C1 kernel, 148×256, ITERS=3000)

| Clock (MHz) | Time (ms) | Idle (W) | FFMA (W) | Δ (W) | TFLOPS | GFLOPS/W |
|---:|---:|---:|---:|---:|---:|---:|
| 510         | 8701 | 144.0 | 177.8 | 33.8 | 13.7 | 77 |
| 800         | 5530 | 147.2 | 201.5 | 54.3 | 21.6 | 107 |
| 1005        | 4448 | 152.0 | 225.2 | 73.2 | 26.8 | 119 |
| 1200        | 3701 | 157.6 | 254.0 | 96.4 | 32.2 | 127 |
| **1500**    | 2964 | 167.3 | 299.9 | 132.6 | 40.2 | **134** |
| **1700**    | 2625 | 174.9 | 339.2 | 164.3 | 45.4 | **134** |
| 1920        | 2312 | 197.7 | 419.5 | 221.8 | 51.5 | 123 |
| 2032 (=1920) | 2314 | 198.4 | 419.0 | 220.6 | 51.5 | 123 |

### Three regimes

1. **< 1005 MHz (idle-dominated):** efficiency 77-107 GFLOPS/W. Idle power doesn't scale down as much as compute, so per-op energy is high. Per-FFMA: 510 MHz wins on absolute pJ/FFMA only because V² × f favors low V — but *per-task* energy does NOT win at 510 because static power eats more wall time.
2. **1005-1700 MHz (sweet spot):** efficiency 119-134 GFLOPS/W. Linear or sublinear power scaling matches throughput growth.
3. **> 1700 MHz (DVS superlinear):** efficiency drops to 123 GFLOPS/W at 1920. **V² × f scaling kicks in** — extra clock costs disproportionate power.

Δ-power (above idle):
- 510 → 1005 MHz: +127% clock, +117% delta. Sublinear (good).
- 1005 → 1500: +49% clock, +81% delta. Slightly superlinear.
- 1500 → 1920: +28% clock, +67% delta. Strongly superlinear (DVS kicks in hard).

### Idle scales with clock (separate phenomenon)

Even with no work running, idle power varies with clock state:
- 510 MHz: 144 W
- 1920 MHz: 198 W (+37%)

This is leakage power growth from higher voltage. Constant V² × f even in "idle". The chip never truly sleeps once the application is running.

### Min-energy clock — per-task energy vs instantaneous TFLOPS/W

There are TWO competing efficiency metrics, often conflated:

| Metric | Best clock | Source |
|---|---|---|
| **Per-op energy (pJ/FFMA)** | **510 MHz** (3.1 pJ/FFMA at 1500 lock; 5.76 at 510 per M9) | M2 / M9 / M11 — accounts for static-power amortization at workload level |
| **Instantaneous GFLOPS/W** | **1500-1700 MHz** (134 GFLOPS/W) | V10 — ratio of throughput to instantaneous power |
| **Per-task energy (mixed workload)** | **1992-2032 MHz boost** (3.08× lower than 510) | M9 V6 C3 — wall-clock advantage dominates |

These are not contradictory — they answer different questions:
- pJ/op accounts for static-power amortization and assumes workload is unbounded (lots of ops to do).
- GFLOPS/W is instantaneous (does not account for total wall time).
- Per-task energy = power × wall time, which boosts the boost-clock case because tasks finish FASTER.

**For ML inference, per-task energy is the right metric.** Datacenter cost is dominated by tokens-per-second / J, which is per-task energy.

### M9 V6 C-series energy curves (M9_ENERGY_PARETO synthesis)

#### V6 C1 — FFMA-saturated (compute-bound), pJ/FFMA across clocks

```
510:  5.76  ← min
800:  6.21
1005: 6.38
1200: 7.23
1402: 7.08
1500: 6.84
1702: 6.54
1920: 6.92
```
Range: 1.26× (5.76 → 7.23). Low spread because FFMA pipe is well-utilized at all clocks.

#### V6 C2 — Memory-bound (mixed L1/L2/DRAM), pJ/byte

```
510:  12.06
800:  11.81  ← min
1005: 12.49
1200: 13.32
1500: 14.92
1700: 15.82
1992: 18.60
```
Range: 1.58× (11.81 → 18.60). Min at 800 because lowest clock starves SMs on L2 latency.

#### V6 C3 — MIXED FFMA + DRAM (4 FFMA per LDG), mJ/task NORMALIZED to 1992

```
510:  3.08x
800:  2.74x
1005: 2.24x
1500: 1.24x
1992: 1.00x  ← MIN
```
Range: **3.08×** — boost clock wins by far for mixed workloads.

### Why mixed workloads love boost clock

Static power on B300 ≈ 165 W (idle baseline at 1500 MHz lock).
- At 510 MHz: total ~430 W → static is **38% of total**.
- At 1992 MHz: total ~530 W → static is **31% of total**.

When workload completes FAST (boost), static power amortizes over LESS time → lower total energy per task.

When workload is mixed compute + memory and BOTH pipes are saturated, the compute throughput scales linearly with clock (more useful work/cycle) but power scales with V² (sublinear vs throughput). Net: throughput grows faster than power → energy per task drops.

### Datacenter implications

Common belief: "lower clock = lower energy". TRUE for pure compute (rare). **FALSE for mixed workloads (which dominate ML inference).**

**ML inference recommendation: USE BOOST CLOCK** (1992-2032 MHz on B300):
- Lowest energy per token (3× lower than 510 MHz)
- Lowest latency
- Highest throughput

**DVFS schemes that DOWN-clock ML inference will INCREASE total energy consumption, not decrease it.** This is counterintuitive and worth shouting at scheduler implementations.

For dedicated **FFMA-only HPC kernels** (rare in production): 510 MHz can save 16% per-op energy.
For **memory-bound streaming** (BW-limited): 800 MHz can save 36% pJ/byte.
For **realistic ML pipelines** (mixed): boost clock 3× more efficient.

### Workload classifier

To pick the right clock, classify by ratio:

| Compute / Memory time ratio | Optimal clock |
|---|---|
| > 4 (compute-bound) | 510 MHz |
| 1-4 (balanced) | 1500-1992 MHz (M9) or 1500-1700 MHz (V10 GFLOPS/W) |
| < 1 (memory-bound) | 800 MHz |

For ML: most kernels are 1-4 ratio → boost clock optimal.
For physics solvers, signal processing: often > 4 → consider down-clocking.

### Performance-frequency Pareto (POWER_FREQUENCY_CURVE — random vs optimized BF16 GEMM)

Random BF16 GEMM (saturates pipes; high data-dep cost) vs Optimized BF16 GEMM (Half A=0, 3 random in Half B; minimal data-dep cost):

| Clock (MHz) | Random P (W) | Opt P (W) | Random/Opt | Random Δ from prev clk |
|---:|---:|---:|---:|---:|
| 510  | 353 | 259 | 1.36× | -- |
| 800  | 484 | 254 | 1.90× | +131 |
| 1005 | 613 | 296 | 2.07× | +129 |
| 1300 | 830 | 369 | 2.25× | +217 |
| 1500 | 1009 | 426 | 2.36× | +179 |
| 1800 | 1095 | 537 | 2.03× | +86 (capped) |
| 2032 boost | 1099 | 629 | 1.74× | +4 (capped) |

Findings:
1. **Random saturates at TDP cap (~1100W) above 1500 MHz** — clock requested but power-cap-throttled. Random Δ stays at +86 / +4W vs ~+200W for unrestricted.
2. **Optimized stays well under cap at all clocks** — 629W at boost = 471W below TDP. Lots of headroom.
3. **Maximum power savings RATIO at 1500 MHz**: 2.36× (1009W / 426W). Highest RATIO at this clock.
4. **Static power floor visible at low clocks**: Optimized=254W at 800 MHz, essentially same as 259W at 510 MHz. Static dominates below ~1000 MHz.
5. **Frequency scaling for optimized**: 254→629W from 800→2032 MHz. Power scales 2.48× for 2.54× frequency = nearly linear (CMOS expectation).
6. **Random scales faster than linear**: 484→1099W from 800→2032 MHz = 2.27× for 2.54× frequency (less than linear because cap kicks in).

Performance-per-Watt (TF/W) for BF16 GEMM:

| Clock | Random TFLOPS | Opt TFLOPS | Random TF/W | Opt TF/W |
|---:|---:|---:|---:|---:|
| 510 | 25.5 | 25.5 | 0.072 | 0.098 |
| 800 | 40 | 40 | 0.083 | 0.158 |
| 1005 | 50 | 50 | 0.082 | 0.169 |
| 1300 | 65 | 65 | 0.078 | 0.176 |
| 1500 | 75 | 75 | 0.074 | 0.176 |
| 1800 | 90 | 90 | 0.082 | 0.168 |
| 2032 | 100 | 100 | 0.091 | 0.159 |

**Optimized peaks at 1300-1500 MHz with ~0.176 TF/W** (more than 2× random's ~0.078 TF/W at same clock). Best practical operating point for energy-efficient inference IF you can structure your weights for low data-dependence (e.g., post-quantization + sparsity).

### How to apply DVS on your kernel

1. Identify regime: compute-bound vs memory-bound vs mixed.
2. For peak ML inference: do nothing — let GPU boost to 2032.
3. For dedicated FFMA HPC: lock at 510 MHz (saves 16% per-op).
4. For DRAM streaming: lock at 800 MHz (saves 36% per-byte).
5. For batch background work where speed isn't critical: lock at 1500-1700 (best GFLOPS/W = 134).
6. NEVER use `-lgc 2032` (paradox — pins to 1920); use `-rgc` for true boost.

### Cross-validate energy claims with two metrics

When publishing a "best clock for X workload" claim, ALWAYS state two numbers:
- pJ/op or pJ/byte (per-work-unit energy)
- TF/W or GB/s/W (instantaneous efficiency)

If both agree on the clock, you have a robust answer. If they disagree (e.g., M9 says 510 for FFMA pJ/op but V10 says 1500-1700 for FFMA GFLOPS/W), then your "best clock" depends on which metric the user cares about. Be explicit about the framing.

### FFMA pJ/op vs clock (DVS V² × f)

| Clock (MHz) | pJ/FFMA (V5 D4 / M11 derived) |
|---:|---:|
| 510  | 3.1 |
| 1005 | 4.9 |
| 1500 | 6.8 |
| 1920 | 9.96 |

Note: M11's table differs from M9's — M11 reports raw V² × f scaling at fixed kernel; M9 reports per-task energy (different reference points). Both are correct in their own framing.

### Memory pJ/byte — different sweet spot than FFMA

| Clock (MHz) | pJ/byte |
|---:|---:|
| 510  | 12.06 |
| **800** | **11.81 ← min** |
| 1005 | 12.49 |
| 1500 | 14.92 |
| 1920 | 18.60 |

The 800 MHz min for memory is real and reproducible. L2 latency hides clock advantage; faster clock just spins SMs while waiting for HBM.

### V10 vs M9 reconciliation (POWER_INCONSISTENCY_LOG §I)

V10 picks 1500-1700 as best FFMA TFLOPS/W; M9 picks 510 MHz. Both are claims about "FFMA" but DIFFERENT metrics:
- M9 = pJ per FFMA op (energy per work unit; 510 wins because V² × f is lower)
- V10 = GFLOPS / W instantaneous (efficiency; 1500-1700 wins because throughput grows faster than power until DVS kicks in at 1700+)

These are not really contradictory — pJ/op accounts for static-power amortization at the workload level (more ops at 510 takes longer → static power eats more wall time, but pJ/op is calculated per work unit not per second), GFLOPS/W is instantaneous (does not account for static-power amortization in absolute terms because it normalizes by power not wall time).

**Resolution:** When asked for "energy-optimal clock" specify what is being optimized:
- Per-task energy → pick M9's number (depends on workload type; boost for ML).
- Instantaneous TFLOPS/W → pick V10's 1500-1700 MHz.
- Per-op energy with assumed unbounded workload → pick M9's per-pipe min (510 / 800 / boost depending on pipe).

For real ML workloads in production datacenters, M9's mixed-workload boost-clock recommendation wins (matches user MEMORY note "ML inference USE BOOST CLOCK 3× lower energy than 510 MHz lock").

### `-lgc 2032` paradox (preview of §45)

`nvidia-smi -lgc 2032` paradoxically pins to **1919.8 MHz** actual (-5.5%), NOT 2032. Confirmed in V10 (rows 1920 and 2032 have IDENTICAL time 2312/2314 ms and IDENTICAL power 419.5/419.0 W). Use `-rgc` to release any lock and reach true 2031.4 MHz boost. See §45 for the full paradox.

### CLAUDE.md memory match

User memory notes:
- "TRUE perf at 2032 MHz: 40 tok/s 70B, 345 tok/s 8B. Clock-lock was 2.35× bottleneck." — matches M9's 2.24× / 3.08× per-task energy advantage of boost.
- "ML inference USE BOOST CLOCK (3× lower energy than 510 MHz)" — matches M9 V6 C3.
- "DVS V² scaling" — matches V10 1700+ MHz superlinear regime.

**Footgun:** ⚠ "Lower clock is more efficient" is WRONG for B300 mixed-workload ML inference — inverted from intuition. Boost is 3× more energy-efficient per task. Scheduler implementations that DVFS-down ML inference will increase energy. Specify which efficiency metric you want before quoting a "min-energy clock": pJ/op (510 for FFMA), instantaneous TFLOPS/W (1500-1700), or per-task energy (boost for mixed). Don't conflate.

**See also:** §41 (TDP cap + idle floor), §43 (data-dependence popcount bell), §45 (clock-lock paradox), corrections/POWER_INCONSISTENCY_LOG.md §I, V10_DVS_CURVE.md, M9_ENERGY_PARETO.md.

---

## §43. Power data-dependence — popcount bell curve, peak at d=16 random

**Answer:** **Active power follows a bell curve in popcount density `d`** (random bit positions per dword), peaking at d=16 (uniform random). DRAM read tier: 240 W active swing @ 1005 MHz (369→637 W between d=0 and d=16); 554 W swing @ 1500 MHz. **Toggle-energy model** (inter-dword bit-flip count) — chunk-level dedup is NULL effect (3% spread). Constant-pattern data (no inter-dword toggle) yields only 18 W spread across popcount 0..32 at L2, proving inter-dword toggling — not popcount per se — is the dominant component. The legacy `HBM_DATA_DEPENDENCE.md` claim of "<50W swing" is **WRONG by 5-7×** — that file is **SUPERSEDED** by POPCOUNT_3TIER + POPCOUNT_VS_CLOCK + POPCOUNT_WRITES + L2_POPCOUNT_SWEEP. `[🟢 HIGH for popcount bell mechanism, 4 mutually consistent files; 🟡 MED for d=32 vs d=0 asymmetry mechanism (HBM3E DBI hypothesis) · src: POPCOUNT_3TIER.md, POPCOUNT_VS_CLOCK.md, POPCOUNT_WRITES.md, L2_POPCOUNT_SWEEP.md, POWER_DATA_DEPENDENCE_SUMMARY.md, STRAYS_CORRECTED.md§2]`

### POPCOUNT_3TIER active power above 150 W idle (1005 MHz)

```
density     L1     L2      DRAM-1G  DRAM-8G
 0          69.6   222.6   369.4    396.6
 1          75.7   250.6   408.0    438.5
 2          81.0   270.9   437.6    466.1
 4          87.7   309.3   484.0    515.7
 8          97.2   363.6   543.5    582.5
12         103.7   394.9   580.0    621.4
16         106.5   404.8   603.8    636.5    ← peak
20         106.5   402.7   603.6    627.0
24         102.8   378.1   560.3    595.1
28          96.0   325.4   500.1    530.0
30          90.9   290.0   450.9    497.9
31          87.2   270.6   444.4    473.0
32          81.4   245.4   411.0    441.4
```

All four tiers show a smooth, near-symmetric bell curve centered at d=16 (uniform random). Peak at d=16, valleys at d=0 and d=32.

### Power burden grows with cache distance

| tier | d=16 active W (peak) | d=0 (zeros) | range W | factor (L1=1) |
|---|---:|---:|---:|---:|
| L1        | 106.5 | 69.6  | 36.9  | 1.0× |
| L2        | 404.8 | 222.6 | 182.2 | 4.9× |
| DRAM-1G   | 603.8 | 369.4 | 234.4 | 6.4× |
| DRAM-8G   | 636.5 | 396.6 | 239.9 | 6.5× |

**L1 reads burn ~107 W active at peak vs DRAM-8G burning ~637 W** — 6× more power per data-dependent component as you go from on-chip cache to HBM.

### Tiered power model (3-component — verified by constant-pattern control)

```
P_active(L2) = 220 W (baseline, all-zeros)
              + 0.56 W × popcount             (static, per-dword)
              + P_toggle(d) × inter-dword toggle activity

with P_toggle(d) ≈ 325 W × [2·d·(32-d) / (32·31)]
```

The toggle-energy ceiling scales by tier:
- L1: ~30 W toggle ceiling
- L2: ~325 W toggle ceiling
- DRAM-1G: ~190 W toggle ceiling on top of L2
- DRAM-8G: ~200 W toggle ceiling on top of L2

This same decomposition fits all four tiers. The toggle-activity ceiling at each cache level matches the number of physical bus stages crossed (L1 inside SM < L2 mesh < HBM PHY), each contributing its own toggle component.

### Constant-pattern control proves inter-dword toggling dominates

L2-warm constant-pattern data (every dword = same constant value):

| const value | popcount | active W (above 150 idle) |
|---|---:|---:|
| `0x00000000` | 0 | 220 |
| `0x12121212` | 8 | 224 |
| `0x000000FF` | 8 | 229 |
| `0x55555555` | 16 | 230 |
| `0x55B71DAA` | 18 | 235 |
| `0xFFFFFFFF` | 32 | 238 |

Range: only **18 W** across 0..32 popcount when data is constant per dword, vs 182 W range for random-position popcount. This proves:
- **Per-dword popcount alone (static component) ≈ 0.56 W per bit-set.**
- **Inter-dword toggling is the dominant component (~163 W extra at d=16) and is ZERO when adjacent dwords are identical.**

### d=32 vs d=0 asymmetry — grows with cache distance

| tier | d=32 - d=0 (active W gap) |
|---|---:|
| L1      | +11.8 |
| L2      | +22.8 |
| DRAM-1G | +41.6 |
| DRAM-8G | +44.8 |

At every tier, all-ones is 10-45 W more than all-zeros even though both have zero per-dword variability. Asymmetry grows with cache depth.

This is **consistent with HBM3E PHY active-low termination / DBI behavior** where holding the wire at "0" is the lower-energy state and "1" requires active drive. The L2 mesh fabric also has some of this property. (MED confidence on the specific DBI mechanism — could be other static-power asymmetries.)

### Saturation by 1 GB ws

Going from 1 GB to 8 GB working set only moves d=16 from 604 W → 637 W (+5%) — meaning **a 1 GB working set is already DRAM-dominant**. There is no need to push beyond a few × L2 capacity to characterize HBM power.

### Clock-rate scaling — POPCOUNT_VS_CLOCK

L2 read popcount (active W above 150 W idle):

| d | 1005 MHz | 1800 MHz | ratio |
|---:|---:|---:|---:|
| 0  | 222 | 429 | 1.93× |
| 8  | 364 | 691 | 1.90× |
| 16 | 405 | 771 | 1.90× |
| 24 | 378 | 717 | 1.90× |
| 32 | 245 | 469 | 1.91× |

Consistent **1.90× active-power scaling for 1.79× clock** — slightly super-linear (likely voltage component too).

L2 write popcount (active W):

| d | 1005 MHz | 1800 MHz | ratio |
|---:|---:|---:|---:|
| 0  | 138 | 319 | 2.31× |
| 8  | 213 | 501 | 2.35× |
| 16 | 235 | 557 | 2.37× |
| 24 | 223 | 524 | 2.35× |
| 32 | 159 | 359 | 2.26× |

**2.35× write-power scaling for 1.79× clock** — much more super-linear than reads. Likely because the SM store pipe was at ~80% utilization at 1005 MHz and at 1800 MHz it is closer to theoretical ceiling.

### DRAM-8G read popcount (active W) — TDP wall visible at 1800 MHz

| d | 1005 MHz | 1300 MHz | 1500 MHz | 1800 MHz |
|---:|---:|---:|---:|---:|
|  0 | 397 | 492 | 554 | 686 |
|  8 | 583 | 706 | 835 | 943 ⚠ |
| 12 | 621 | 746 | 896 | 942 ⚠ |
| 16 | 637 | 787 | 921 | 942 ⚠ |
| 20 | 627 | 791 | 916 | 943 ⚠ |
| 24 | 595 | 725 | 866 | 941 ⚠ |
| 28 | 530 | 651 | 760 | 943 ⚠ |
| 32 | 441 | 537 | 609 | 758 |

⚠ = at TDP wall (1100 W total). Notice d=8..28 all clipped to ~942 W active = 1092 W total at 1800 MHz. The bell curve is real but **flat-topped above the TDP cap**; you cannot measure the true peak shape at this clock without either:
1. A higher TDP cap (none available — `power.max_limit = 1100 W`).
2. A lower clock — at 1500 MHz d=16 hits 921 W active = 1071 W total (just under TDP), bell curve is unclipped.

### Memory pJ/byte vs popcount and clock

| Subsystem | d=0 (zeros) | d=16 (random peak) | d=32 (ones) | Range (peak swing) |
|---|---:|---:|---:|---:|
| L1 reads | 70 | **107** | 82 | 35% |
| L2 reads | 223 | **405** | 245 | 45% |
| DRAM-1G reads | 369 | **604** | 411 | 39% |
| DRAM-8G reads | 397 | **637** | 441 | 38% |
| L2 writes | 140 | **235** | 159 | 41% |
| DRAM-8G writes | 264 | **405** | 288 | 35% |
| FFMA compute (8-ILP self-chain) | 40 | **103** | — | 60% |
| IADD3 compute (8-ILP self-chain) | — | **103** | — | 95% |

### Per-byte energy at 1005 MHz

| Op | nJ/byte |
|---|---:|
| L2 read (d=16 random) | 25.5 |
| L2 write (d=16) | 62.2 |
| DRAM read (d=16) | 86.1 |
| DRAM write (d=16) | 115.7 |

DRAM writes are ~1.34× more energy-intensive than reads. L2 writes are ~2.4× more than L2 reads.

### Bit-stride / chunk-level dedup — NULL RESULT

`L2_BITSTRIDE_SWEEP.md` and `L2_POPCOUNT_SWEEP.md` both tested the hypothesis that chunk-level repetition (e.g., dword-pairs of identical values) saves power via internal dedup. **Result: 3% spread across all tested patterns.** DRAM/L2 signaling does NOT exploit chunk-level repetition; only per-cycle bit-flip count matters.

CONFIRMED across 3 files. The dedup hypothesis is DEAD.

### Sparsity > 10% threshold

Sparsity > 10% (in random data) gives measurable savings; below 10% no measurable savings. Granularity (byte vs 128-byte chunks) barely matters — toggle activity per cycle is what counts.

### HBM_DATA_DEPENDENCE.md is SUPERSEDED — wrong by 5-7×

| Source | DRAM data-dep range (active W) |
|---|---|
| **`HBM_DATA_DEPENDENCE.md` (this file)** | **<50 W ← WRONG** |
| `L2_DRAM_DATA_PWR.md` (constant patterns) | 522.6 → 528.0 = 5.4 W ← agrees with the constant-pattern regime ONLY |
| `POPCOUNT_3TIER.md` DRAM-1G (random-position popcount) | **234 W** range (369 → 604) |
| `POPCOUNT_3TIER.md` DRAM-8G (random-position popcount) | **240 W** range (397 → 637) |
| `POPCOUNT_VS_CLOCK.md` DRAM-8G @ 1500 MHz | 367 → 921 = **554 W** swing |
| `POPCOUNT_WRITES.md` DRAM-8G writes @ 1005 MHz | 264 → 405 = 141 W |

`HBM_DATA_DEPENDENCE.md` was written BEFORE the popcount sweep distinguished "constant-pattern" from "random-position-popcount" regimes. It generalized the inter-pattern (constant-vs-constant) result to ALL data variation. The popcount work proves **inter-dword toggle activity is the dominant lever, not popcount per se**, and the swing is 5-7× larger than HBM_DATA_DEPENDENCE estimated.

**RETRACTED claims from HBM_DATA_DEPENDENCE.md:**
1. "HBM data-dependent power likely contributes <50W out of total 1100W TDP" — WRONG. Real swing under random-position popcount is **240 W active / 554 W at 1500 MHz**.
2. "Memory bandwidth doesn't have a strong throttling-driven speedup mechanism" — partially WRONG. BW is content-INDEPENDENT (correctly captured), but POWER is strongly content-dependent and CAN throttle clocks at high clocks (1100 W TDP wall hit at 1700-1800 MHz with d=8..28 random data).
3. "Memory-bound workloads: no significant throttling avoidance" — WRONG. At 1500-1800 MHz, random-data DRAM workloads CAN reach TDP cap and throttle; low-popcount data avoids this and saves 240 W.

**SUPERSEDE `HBM_DATA_DEPENDENCE.md` with `POPCOUNT_3TIER.md` + `POPCOUNT_VS_CLOCK.md` + `16_power_clock_CORRECTED.md` §5.**

### Practical implications

For LLM inference where weights are FP4/FP8 quantized, many tensor elements have low popcount (the high bits of mantissa-only values are often 0). A model that loads "mostly zero" data through DRAM will burn **~240 W less than a model with truly-random weights** — on a 1.1 kW B300 that is **~22% of TDP**.

For thermal stress testing: DRAM read with d=16 random per-dword data at locked 1500 MHz is the highest sustained power workload (1071 W steady, just below TDP cap).

For energy efficiency in inference: lower clock + lower-popcount data reduces power per-operation faster than per-cycle. A factor of 2× clock rarely doubles power; a factor of 2× popcount density (away from d=16) shaves 30-40% power without losing FLOPS / GBps.

### Open questions

1. Per-DRAM-channel `dram__bytes_*.per_dram` ncu metric to confirm even distribution across the 6 HBM3E stacks at high clock (flagged by all 4 popcount files).
2. Voltage probe to attribute the 5% super-linearity in L2 reads at high clock.
3. **Real production weight tensors** vs synthetic d=16 not directly verified — predicted -120 to -180 W vs synthetic.
4. TMA bulk loads — different memory subsystem path, not yet swept (different DBI behavior may apply).
5. Hold popcount fixed but vary inter-dword Hamming distance directly (predicted by toggle theory: "every dword = 0x12121212 (popcount 8, identical)" should be near d=0 power, NOT near d=8 power).

### How to verify popcount on your data

If you suspect data-dependence is dominating your kernel's power, check:

```python
# Per-dword popcount mean for a numpy array of f16/bf16/f32:
import numpy as np
def popcount_mean(arr):
    bits = np.unpackbits(arr.view(np.uint8))
    return bits.mean() * 32  # mean bits set per 32-bit dword

# For weights:
print(f"Mean popcount: {popcount_mean(model_weights):.1f}")
# < 12 or > 20: low data-dep cost (good for power)
# 14-18: high data-dep cost (worst case)
```

Random uniform weights: ~16. Fine-tuned models with sparsity: often 8-12. Quantized models with skewed distributions: often <12. Activations during inference: often higher (depends on layer).

### Bandwidth scaling vs power scaling

`POPCOUNT_VS_CLOCK` measured BW alongside power:

| clock MHz | wall ms (30M iters) | wall BW TB/s | clock-normalized |
|---:|---:|---:|---:|
| 1005      | 2284                | 15.92        | 1.000            |
| 1800      | 1804                | 20.16        | 0.707 (× ratio)  |

Read BW ratio 20.16/15.92 = 1.27× for 1800/1005 = 1.79× clock — sub-linear because L2 → SM transport is not 100% saturated at 1800 MHz; HBM is the shared bottleneck.

| clock MHz | write BW TB/s |
|---:|---:|
| 1005      |  3.78         |
| 1800      |  6.71         |

Write BW ratio 1.78× for 1.79× clock = **perfectly linear** (LSU store pipe is purely SM-side bound). 32 B/cy/SM × 148 SMs × 1.8 GHz = 8.52 TB/s theoretical; 6.71 / 8.52 = **78.7% of theoretical store-pipe ceiling** at 1800 MHz.

So writes scale linearly with clock; reads scale sub-linearly because HBM is the bottleneck. Reads benefit MORE from higher clock when memory subsystem is not saturated; once saturated, extra clock just spins SMs.

### Implications for thermal stress testing

**For thermal stress testing**: DRAM read with d=16 random per-dword data at locked 1500 MHz is the highest sustained power workload (1071 W steady, just below TDP cap). Useful for:
- Validating cooling design (can the heatsink handle 1071 W steady?)
- Stress-testing power delivery (PSU + voltage regulator)
- Reproducing field-failure conditions

**For datacenter energy planning**: Use 962 W (sustained BF16 cuBLAS, real workload) as the realistic ceiling, not 1071 W (synthetic stress). Real ML inference typically runs at 600-900 W per B300 depending on workload.

**For energy efficiency in inference**: lower clock + lower-popcount data reduces power per-operation faster than per-cycle. A factor of 2× clock rarely doubles power; a factor of 2× popcount density (away from d=16) shaves 30-40% power without losing FLOPS / GBps. This is the main lever for "more efficient inference per joule": choose data layouts that have low inter-dword toggle activity.

### POPCOUNT_WRITES detail

Write power follows the same bell-curve mechanism as reads but with different magnitudes. From `POPCOUNT_WRITES.md`:

L2 write at 1005 MHz:
| d | Active W | Δ from d=0 |
|---:|---:|---:|
| 0 | 138 | 0 |
| 8 | 213 | +75 |
| 16 | 235 | +97 |
| 24 | 223 | +85 |
| 32 | 159 | +21 |

DRAM-8G write at 1005 MHz:
| d | Active W | Δ from d=0 |
|---:|---:|---:|
| 0 | 264 | 0 |
| 8 | 373 | +109 |
| 16 | 405 | +141 |
| 24 | 388 | +124 |
| 32 | 288 | +24 |

Writes hit slightly lower peak than reads (405 vs 637 for DRAM-8G at d=16) but follow the same shape. Per-byte energy:
- L2 write d=16: 62.2 nJ/byte (vs read 25.5 → writes are 2.4× more energy per byte)
- DRAM write d=16: 115.7 nJ/byte (vs read 86.1 → writes are 1.34× more energy per byte)

The write/read ratio is more lopsided at L2 than at DRAM because L2 has internal cache-coherence overhead on writes that doesn't apply at DRAM (DRAM is just bus signaling).

### POPCOUNT vs OTHER POWER METRICS — what they tell us

Each popcount-style measurement tests a different question:

| File | Question | Conclusion |
|---|---|---|
| POPCOUNT_3TIER | Does L1/L2/DRAM data-dep follow the same shape? | YES, bell at d=16 with growing magnitude per tier |
| POPCOUNT_VS_CLOCK | Does the bell scale with clock? | YES, ~1.9× active power per ~1.79× clock; TDP clips at 1700+ |
| POPCOUNT_WRITES | Are writes the same as reads? | YES, same shape; lower peak; higher per-byte energy |
| L2_BITSTRIDE_SWEEP | Does chunk-level dedup save power? | NO (3% spread) |
| L2_DRAM_DATA_PWR | Does inter-pattern variance drive power? | LITTLE (5W spread for constant-pattern data) |
| BF16_PERBIT_POWER | Which bits matter most for tcgen05? | sign-bit forces -56W (ReLU), exp bits ~30W each, mantissa bits ~19W each |
| FP8_KVARY_POWER | Does FP8 mma have similar B-K-vary patterns? | YES; +71W for 16-unique B (FP8) vs +47W (BF16) |
| DISABLE_LANE_POWER | Does disable_lane save power? | YES, 2.4 W/disabled column for tcgen05 |
| POWER_FREQUENCY_CURVE | What is the random-vs-optimized power gap across clocks? | 1.36× at 510 MHz → 2.36× at 1500 MHz (peak ratio) |

All consistent with the **toggle-energy model**: per-cycle bit-flip count drives power, regardless of which subsystem (L1/L2/DRAM/tcgen05) does the toggling. The toggle coefficient ladder L1 < L2 < DRAM < HBM3E PHY just reflects the number of physical buses each bit must cross.

### How the toggle-energy model emerges from CMOS

CMOS gates dissipate dynamic power as `P = α × C × V² × f` per gate, where α is activity factor (probability of bit flip per cycle).

For a 32-bit word transferred between dwords with `d` bits set per dword:
- Hamming distance between two random dwords with d bits each: 2 × d × (32-d) / 32 (expected) on the dword level — i.e., `(d × (32-d) + (32-d) × d) / 32 = 2d(32-d)/32`.
- Sum over 32 wires: each wire toggles with probability d/32 × (32-d)/32 + (32-d)/32 × d/32 = 2 × d × (32-d) / (32 × 32). Sum across 32 wires: 32 × 2 × d × (32-d) / (32 × 32) = 2 × d × (32-d) / 32.
- This peaks at d=16 with 16 / 32 × 16 = 8 expected toggles per cycle on a 32-bit bus, or 8/32 = 25% activity factor.

`POPCOUNT_3TIER`'s formula `P_toggle(d) ≈ α × [2·d·(32-d) / 31]` matches this CMOS expression closely. The factor 31 in denominator is an empirical fit that accounts for normalization.

### Sparsity > 10% threshold

Sparsity > 10% (in random data) gives measurable savings; below 10% no measurable savings. Granularity (byte vs 128-byte chunks) barely matters — toggle activity per cycle is what counts.

This is consistent with the toggle-energy model: random data with d ≈ 16 has the maximum toggle, so any movement away from d=16 (toward d=0 or d=32) reduces power. 10% sparsity in floating-point activations typically corresponds to popcount density of 12-14, slightly below the d=16 peak — which explains why "10% sparsity" gives measurable savings.

### Constant-pattern data baseline

L2-warm constant-pattern data (every dword = same constant), 1005 MHz:

| const value | popcount | active W (above 150 idle) | Δ from constant=0 |
|---|---:|---:|---:|
| `0x00000000` | 0 | 220 | 0 |
| `0x12121212` | 8 | 224 | +4 |
| `0x000000FF` | 8 | 229 | +9 |
| `0x55555555` | 16 | 230 | +10 |
| `0x55B71DAA` | 18 | 235 | +15 |
| `0xFFFFFFFF` | 32 | 238 | +18 |

Range: only **18 W** across 0..32 popcount when data is constant per dword — vs 182 W range for random-position popcount. The 18 W is the static per-dword popcount component (~0.56 W per bit-set on the 220 W active floor).

This is the rigor proof that **inter-dword toggling is the dominant component (~163 W extra at d=16) and is ZERO when adjacent dwords are identical.**

**Footgun:** ⚠ HBM_DATA_DEPENDENCE.md claims "<50W swing" — WRONG by 5-7×. That file is SUPERSEDED. Always cite POPCOUNT_3TIER / POPCOUNT_VS_CLOCK / 16_power_clock_CORRECTED §5 for HBM data-dep. Don't confuse "constant-pattern data" (5 W swing per L2_DRAM_DATA_PWR) with "random-position popcount" (240-554 W swing) — they are different regimes. The chunk-level dedup hypothesis is DEAD (3% spread); per-cycle inter-dword bit-flip count is the only knob.

**See also:** §41 (power floor / TDP cap), §42 (DVS curve), §51 (tensor power per CTA — Section E), corrections/POWER_INCONSISTENCY_LOG.md §G+§K, POPCOUNT_3TIER.md, POPCOUNT_VS_CLOCK.md.

---

## §44. Power per pipe / per op — M11 vs 16_power_clock 2× discrepancy (UNRESOLVED)

**Answer:** Two synthesis docs disagree by ~2× on the headline FFMA TF/W: **M11 says FFMA = 9 J/TFLOP = 0.111 TF/W**; **16_power_clock says 0.21 TF/W** (74.6 TF / 361 W). The discrepancy is most likely an **operating-point mismatch**: M11's 359 W / 39.7 TFLOPS is the "FFMA-bound" entry which is half of peak (low ILP / different occupancy), while 16_power_clock's 361 W / 74.6 TFLOPS is the saturated peak. Both numbers are individually correct at their stated configuration — but quoting either as "the" FFMA TF/W without the operating-point qualifier is misleading. Reconcile via direct measurement at fixed ncu pipe utilization. `[🟡 MED — UNRESOLVED · src: POWER_INCONSISTENCY_LOG.md§J, M11_PER_PIPE_ENERGY.md, 16_power_clock_CORRECTED.md§3+§J]`

### The 2× discrepancy table

| File | FFMA peak TF | Power | TF/W | J/TFLOP | Operating point |
|---|---:|---:|---:|---:|---|
| **`M11_PER_PIPE_ENERGY.md`** | 39.7 TFLOPS | 359 W | **0.111** | **9.0** | "FFMA-bound" (low-ILP / mid-occ) |
| **`16_power_clock_CORRECTED.md` §3** | 74.6 TFLOPS | 361 W | **0.21** | 4.84 | Peak (ILP=24, 256 thr, boost) |
| CLAUDE.md memory "5 TFLOPS/W FP8" | — | — | — | — | unrelated (FP8 cuBLAS, not FFMA) |

Power is roughly the same (~360 W) but throughput is 2× different. M11's lower throughput suggests it ran at lower ILP / smaller blocks where the FMA pipe was underutilized. 16_power_clock's 74.6 TFLOPS = 97% of theoretical peak (76.96 TFLOPS at 2032 MHz, 128 cores/SM × 148 SMs).

The compatible reading is: M11 measured the same power as peak even at half the throughput, because static / leakage components dominate at low ILP (the SMSPs are alive but lanes are idle, paying leakage cost without producing FLOPS).

### Per-op energy table (consolidated from M2 + M11 + 16_power_clock_CORRECTED §3)

@ 1500 MHz lock unless noted, V² × f scaling for other clocks:

| Op | Energy/op | Source | Confidence |
|---|---|---|---|
| FFMA (with .reuse, broadcast operand) | **2.2 pJ/FLOP = 4.4 pJ/FFMA** | M2 H1 | HIGH |
| FFMA (no .reuse, 3 unique RF reads) | 6.5 pJ/FFMA (1.49× more) | M2 H9 | HIGH |
| RF read (incremental) | 0.3 pJ/read | M2 H9 | HIGH |
| IMAD chain | 6.5 nJ / 1M ops | M2 H2 | HIGH |
| LOP3 chain | 3.85 nJ / 1M ops | M2 H2 | HIGH |
| MUFU rsqrt.ftz | similar to LOP3 | M2 H3 | HIGH |
| MUFU sin (no .ftz) | 1.5× MUFU rsqrt | M2 H3 | HIGH |
| LDG (cold HBM) | **96.6 nJ / 1M ops (15× IMAD)** | M2 H4 | HIGH |
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
| DRAM write (d=16) | 115.7 nJ/byte | POPCOUNT_WRITES | HIGH |

### Per-pipe active power (M2, 148 SMs, ITERS for ~100ms+ steady-state)

| Pipe / Op | Δ power | Per-FFMA-eq energy | Source |
|---|---:|---:|---|
| Idle (loop only, DCE'd) | 0 W | 0 | H1 |
| FFMA (single chain) | +10 W | 0.6 pJ/FLOP | H1 (#dedd2b1) |
| FFMA (16 chains, 2-RF reads) | +24 W | 0.6 pJ/FFMA | H9 (#b489c02) |
| FFMA (16 chains, 3-RF reads, no .reuse) | +27 W | 0.91 pJ/FFMA | H9 |
| IMAD chain | +37 W | ~6.5 J / 1M ops | H2 (#2713af5) |
| LOP3 chain | +22 W | ~3.85 J / 1M ops | H2 |
| MUFU rsqrt.ftz | +24 W | similar to LOP3 | H3 |
| MUFU sin (no .ftz) | +39 W | 1.5× MUFU rsqrt | H3 |
| **LDG (memory)** | **+177 W** | **~96.6 J / 1M ops (15× IMAD)** | H4 |
| LDS (32-bit shared load) | +22 W | 4.4× lower than LDG | H8 (#7b6ec38) |
| STS (32-bit shared store) | +31 W | +41% vs LDS | H8 |
| LDS.128 (vec load) | +56 W | 2.5× scalar | H8 |
| STS.128 (vec store) | +74 W | 2.4× scalar | H8 |

**KEY: Memory pipe (LDG +177 W) is by FAR the dominant power consumer.** SMEM is 5× lower power than HBM. Cache-blocking saves both time AND energy.

### Static vs dynamic split

| Source | Static | Dynamic | Operating point |
|---|---:|---:|---|
| `M2_ENERGY_LADDER.md` | 0.05 W/SM static | 0.4 W/SM dynamic FFMA — 8-9× ratio | per H6, fixed kernel |
| `M11_PER_PIPE_ENERGY.md` | 165-170 W static GPU; 0.7 W/SM dynamic FFMA | "Static is 30-60% of total" | mixed across V6/V7 |
| `PER_SM_POWER_SCALING.md` | 1.0 W/SM const tcgen05 | 3.1 W/SM random tcgen05 | tcgen05 BF16 |

M2 says 0.4 W/SM dynamic FFMA; M11 says 0.7 W/SM dynamic FFMA. Both at 1500 MHz. Different kernels (likely single-chain vs 16-chain). Within ~2× — not a true contradiction, just two operating points.

The mature CMOS process means **leakage is ~10% of switching**. Power-gating idle SMs would save only ~7 W (out of ~1100 W TDP). Clock/voltage scaling is the main lever for power management.

### Compiler flag energy impact (M2 §4)

| Flag combo | Baseline runtime | With flag | Δ time | Δ energy |
|---|---:|---:|---:|---:|
| -use_fast_math (vs no fast_math) | 348 ms | 106 ms | 3.28× faster | **4.05× less energy** |
| __forceinline__ (vs __noinline__) | 182 ms | 12 ms | 15.2× faster | ~15× less energy |
| .reuse on broadcast (vs no .reuse) | (in same kernel) | -32% time | -19% power | **-49% energy** |
| -Xptxas=-O3 (vs -O0) | 53 ms | 12 ms | 4.4× faster | ~4× less energy |

### TFLOPS/W ladder

| Workload | Power | TFLOPS | TF/W |
|---|---:|---:|---:|
| FFMA peak (16_power_clock) | 361 W | 74.6 | **0.21** |
| FFMA-bound mid-occ (M11) | 359 W | 39.7 | **0.111** ← 2× lower |
| BF16 mma.sync (16_power_clock) | 411 W | 569 | 1.39 |
| FP8 cuBLAS (16_power_clock) | 886 W | 4491 | **5.07** |
| HBM streaming | 460 W | 7.5 TB/s | 16.3 GB/s/W |
| Idle | 167 W | 0 | ∞ (waste) |

**For FP8 inference: 5.07 TF/W is the rigor-verified number** (matches CLAUDE.md memory "5 TFLOPS/W FP8" and is in `B300_TRUE_REFERENCE.md`).

### Lane-level granularity

Predicating off lanes does NOT reduce per-warp power (R4 #de87c5a):
- 32 → 1 active lanes via @p: 171 → 170.6 W (delta < 1 W)
- Per-warp issue + RF dominates per-lane compute power

True lane-level power gating requires BRA divergence (warp-level skip). For "disable_lane" approaches with tcgen05.mma, see Section E §51 — the 2.4 W/disabled-column saving comes from disabled tensor MAC units, not from FFMA lane-level gating.

### Practical energy recipes

#### Maximum FFMA throughput at minimum energy
1. Use `-use_fast_math` (4× energy savings)
2. `__forceinline__` device helpers (15× speedup)
3. Use broadcast operand pattern → emit `.reuse` (49% energy savings via D6/H9)
4. Target compute-bound (FFMA dominant) over memory-bound (LDG dominant) — 5× lower energy

#### Persistent kernels waiting on host
1. Use `mbarrier.try_wait` not spin loop (25% lower power per R2)
2. Or just exit and re-launch (kernel launch is 2 µs; spin-waits cost more)

#### Multi-GPU coordination
- NVLink one-way 1.55 µs (J1)
- Use `cuStreamWriteValue32` not kernel-write (8× faster, J2)
- For CPU↔GPU signaling: managed mem + CPU spin (4.4 µs RT, L1)

### Methodology caveats

1. **nvidia-smi power = 33 Hz max CLI** sample rate. Need 5+ sec sustained kernels for steady-state.
2. **NVRTC harness uses fast_math by default** — many H/R measurements include FTZ behavior.
3. **Per-pipe isolation is hard** (R1 partial). Use ncu pipe metrics where possible.
4. **High variance on per-op energy** (M11: ±30% on derived numbers).

### How to resolve M11 vs 16_power_clock

To definitively reconcile the FFMA TF/W discrepancy:
1. Run the same FFMA kernel at multiple ILP / occupancy points.
2. Measure ncu `pipe_fma.avg.pct_of_peak_sustained_active` at each point.
3. Plot TF/W vs pipe utilization.
4. M11's 0.111 should fall on the curve at low utilization; 16_power_clock's 0.21 at high utilization.
5. If they don't fall on the same curve, there is a methodological bug to find.

This work is open. Until done, **don't quote a single TF/W; specify operating point.**

### Fully-decomposed energy budget for a typical ML kernel

A representative attention kernel with seq=4096, head=64, BF16 mm + softmax + BF16 mm:
- LDG (HBM read of QKV): ~3.2 GB × 96 nJ/byte = 307 mJ
- BF16 mma.sync: ~2.5 TFLOPS × 50 pJ/output = ~12 mJ
- Softmax (EX2 + RCP + REDUX + SHFL): ~16 G ops × ~10 pJ/op = 0.16 mJ
- LDS (SMEM staging): ~2 GB × 5 pJ/B = 10 mJ
- STG (HBM write of output): ~0.5 GB × 116 nJ/B = 58 mJ
- Total: ~388 mJ per attention block

Per-component fraction:
- HBM (read+write): 365 mJ = **94%**
- Tensor cores (BF16 mma): 12 mJ = 3%
- Softmax + reduction: 0.2 mJ = 0.05%
- SMEM: 10 mJ = 2.6%

**HBM dominates by 30-40×.** For ANY ML kernel above HBM-bandwidth-bound regime (most production attention / GEMM at large dim), reducing HBM traffic is the #1 energy lever. Tensor-core energy is rounding error in this regime.

### "FP8 inference: 5 TFLOPS/W" — where it comes from

CLAUDE.md memory and B300_TRUE_REFERENCE both cite "5 TFLOPS/W FP8". Source: `16_power_clock_CORRECTED.md` row "FP8 cuBLAS = 4491 TFLOPS / 886 W = 5.07 TF/W" sustained.

This is at boost clock with cuBLAS-internal cudaGraph batching (avoids per-call launch overhead) at M=N=K=8192. Under this regime:
- Tensor-core utilization: 91% MFU
- Power: 886 W (87% TDP)
- Throughput: 4491 TFLOPS dense FP8

Compare to FP32 FFMA peak: 74.6 TF / 361 W = 0.21 TF/W. **FP8 tensor-core is 24× more energy-efficient than FP32 FFMA.** This is the reason ML inference moved to FP8.

Compare to BF16 mma.sync: 569 TF / 411 W = 1.39 TF/W. FP8 is 3.6× better TF/W than BF16. The combination of half the bits + tensor-core acceleration gives the energy advantage.

Compare to NVFP4 cuBLAS: at K=96 sweet spot, ~10.8 PF / ~600 W ≈ 18 TF/W. **NVFP4 is 3.5× more energy-efficient than FP8.** This is the reason for the move to FP4 / NVFP4 in 2026.

### Per-tier energy ladder at fixed clock (1500 MHz lock, M2 + M11 synthesized)

| Tier | Δ power (W) | Energy/op (pJ) | vs FFMA |
|---|---:|---:|---:|
| Idle (loop only, DCE'd) | 0 | 0 | 0 |
| FFMA (single chain) | +10 | 0.6/FLOP | 1× |
| FFMA (16 chains, .reuse) | +24 | 4.4/FFMA | 1× baseline |
| FFMA (16 chains, no .reuse) | +27 | 6.5/FFMA | 1.49× |
| FADD / FMUL (single FLOP) | ~+12 | 4.0 | ~0.9× |
| IADD3 chain | +37 | 6.5/op | 1.5× |
| LOP3 chain | +22 | 3.85/op | 0.88× |
| MUFU rsqrt.ftz | +24 | ~4.5/op | 1× |
| MUFU sin (no .ftz) | +39 | ~6.8/op | 1.5× |
| LDG cold HBM | +177 | ~96/op | 22× |
| LDG L1 hit (.ca) | +30 | ~5/op | 1.1× |
| LDG L2 bypass (.cg) | +60 | ~10/op | 2.3× |
| LDS u32 | +22 | ~4/op | 0.9× |
| LDS.128 | +56 | ~10/op | 2.3× |
| STS.128 | +74 | ~13/op | 3× |
| STG.128 | +200 | ~120/op | 27× |
| HMMA BF16 mma.sync | +250 | ~50/output | 11× |
| tcgen05.mma BF16 (per CTA) | varies | ~0.5/output (deferred §51) | 1/9× |
| Branch predictable | +14 | 14.8/op | 3.4× |
| Branch divergent half-warp | +18 | 18.5/op | 4.2× |
| __syncthreads | — | ~30/op | 6.8× |
| cluster.barrier | — | ~390/op | 89× |
| cp.async.bulk (TMA) | varies | ~80/byte loaded | 1.2× LDG |

(Confidence varies HIGH for FFMA / LDG / LDS / branch; MED for HMMA / sync / cluster.barrier; LOW for TMA which has not been directly energy-measured.)

### Compiler flag energy impact (M2 §4 — concrete numbers)

| Flag combo | Δ time | Δ energy |
|---|---|---|
| `-use_fast_math` (vs no fast_math) | 3.28× faster | **4.05× less energy** |
| `__forceinline__` (vs `__noinline__`) | 15.2× faster | ~15× less energy |
| `.reuse` on broadcast (vs no `.reuse`) | -32% time | -19% power = **-49% energy** |
| `-Xptxas=-O3` (vs -O0) | 4.4× faster | ~4× less energy |
| LDG.128 (vs LDG.32) | 2.95× faster (V10_LDG_WIDTH) | ~3× less energy |
| `redux.sync.add.u32` (vs SHFL chain) | 2.34× faster (Q3) | ~2× less energy |
| TMA pipelined (vs TMA single-deep) | 1.07× faster (V46) | ~1× same energy |
| cudaGraph (vs sequential launches) | <100µs amortization saved | depends on workload |

Top energy levers (in priority order):
1. Use FP8 / NVFP4 over BF16 / FP32 (3.6×-24× TF/W win at the math-pipe level)
2. Cache-block to reduce HBM traffic (HBM is 30-40× of total energy in HBM-bound kernels)
3. `-use_fast_math` (4× energy)
4. `__forceinline__` (15× when applicable)
5. `.reuse` annotations (49% energy)
6. Use REDUX over SHFL chain for INT warp-reduce (2× energy)
7. Use LDG.128 / STG.128 over LDG.32 / STG.32 (3× energy on memory pipe)
8. Boost clock for ML inference, NOT down-clock (3× energy per task)

### Where energy gets wasted

1. **Static power on idle SMs** — at low occupancy, leakage of unused SMs is ~50% of total. Run persistent kernels with full grid where possible.
2. **Memory-bound kernels at boost clock** — clock spins SMs while waiting for HBM. At 800 MHz mem-bound is 36% lower energy.
3. **Divergent branches** — 4.2× FFMA energy per branch. Use predication where divergence is small.
4. **Wide register reads (no .reuse)** — 49% energy overhead vs broadcast pattern.
5. **HMMA accumulator initialization** — re-zeroing the FP32 accumulator every K-iter wastes mma cycles. Keep accumulator across the K-loop.
6. **Cold L1 access patterns** — 2.3× higher energy than hot L1. Use cache-blocking.
7. **Bank-conflicted SMEM access** — V44 shows 32-way conflict 2× cost in latency-bound regime; in throughput-bound regime ~1× (warp scheduler hides). Either way, 32-way conflict means STS power is 2× higher per useful work.

### Open questions (M11 §"Open questions for V8" — most still open)

1. tcgen05.mma actual pJ/op (in-progress; see Section E §51).
2. Per-pipe SASS-level energy breakdown via ncu sm__pipe_*_cycles_active.ratio.metric.
3. Cross-pipe energy interaction (does HMMA + LDG cost more than sum?).
4. Voltage rail measurement (NVML provides power; not voltage directly).

**Footgun:** ⚠ Don't quote a single TF/W for FFMA — M11 (0.111) and 16_power_clock (0.21) are both correct but at different operating points. Always specify ILP / occupancy. For consumer-facing rigor: use 0.21 TF/W (peak, 16_power_clock) with operating-point qualifier; use 0.111 TF/W (M11) only for low-ILP code. For headline marketing: FP8 cuBLAS = 5.07 TF/W is the right number to lead with.

**See also:** §41 (TDP cap), §42 (DVS curve), §43 (data-dep popcount), §51 (tensor power per CTA — Section E), corrections/POWER_INCONSISTENCY_LOG.md §H+§J, M11_PER_PIPE_ENERGY.md, M2_ENERGY_LADDER.md.

---

## §45. Clock-lock paradox + stuck-at-1005 — never use `-lgc 2032`; always sample clock during run

**Answer:** **`nvidia-smi -lgc 2032` PARADOXICALLY pins to 1919.8 MHz** (5.5% lower than requested), NOT 2032. Replicated 4× in 16_power_clock and confirmed by V10 DVS curve (rows 1920 and 2032 have IDENTICAL time and power). To reach true 2031.4 MHz boost, use `-rgc` (release graphics clock) — no lock at all. Separately, **B300 can stick at 1005 MHz silently under load with NO explicit lock**; `nvidia-smi -q -d CLOCK` will show "Application Clocks Setting: 2032 MHz" and "Idle: Active" with no throttle reasons, but the actual clock under load is 1005 MHz. Recover with `sudo nvidia-smi -rgc -i 0`. Always sample `clocks.current.sm` during the FIRST run of a benchmark session. `[🟢 HIGH for both paradoxes; UNANIMOUS in 4 sources · src: 16_power_clock_CORRECTED.md§1, V10_DVS_CURVE.md, B300_TRUE_REFERENCE.md, CLAUDE.md memory feedback_clock_stuck_no_lock.md]`

### The `-lgc 2032` paradox table

| Clock state | Reported clock | Actual clock | Source |
|---|---:|---:|---|
| True idle | 120 MHz | 120 MHz | 16_power_clock |
| Default boost (no lock, sustained FFMA) | 2032 MHz | **2031.4 MHz** | clock64/globaltimer ratio |
| `nvidia-smi -lgc 2032` | 2032 | **1919.8 MHz** (-5.5%) | replicated 4× |
| `nvidia-smi -lgc 1410` | 1410 | 1410 MHz (correct) | one-shot |
| `nvidia-smi -lgc 510` | 510 | 510 MHz | V10 DVS curve |
| **Stuck-without-lock state** | varies | **1005 MHz** (silent) | feedback_clock_stuck_no_lock |

**Counter to old "always boosts to 2032" claim:** B300 CAN pin to 1005 MHz with NO explicit lock when other procs thrash the GPU; sample clock during EVERY long run.

### Confirmation in V10 DVS curve

V10's DVS curve table includes both rows:
| Clock (MHz) | Time (ms) | Idle (W) | FFMA (W) |
|---:|---:|---:|---:|
| 1920        | 2312 | 197.7 | 419.5 |
| 2032 (=1920) | 2314 | 198.4 | 419.0 |

**Identical time and power to within measurement noise.** The "2032" row is just `-lgc 2032` which is silently treated as `-lgc 1920`. Confirmed independently.

### Why `-lgc 2032` is silently lowered to 1920

This appears to be a **driver behavior** where `-lgc 2032` is interpreted as "set the application clock to 2032 MHz" but the actual SM clock domain caps at the BASE clock (1920 MHz) for sustained loads, not the boost clock. The GPU only reaches the true 2031.4 MHz boost when given `-rgc` (no lock at all) AND the workload + thermals allow it.

UNANIMOUS in 4 files:
| File | Reading | Confirmed? |
|---|---|---|
| `16_power_clock.md` | -lgc 2032 → 1919.8 MHz | YES (clock64) |
| `V10_DVS_CURVE.md` | 2032 row power = 1920 row power (419 W ≈ 419 W) | YES |
| `B300_TRUE_REFERENCE.md` | "lgc 2032 paradoxically pins to 1920" | YES |
| `POWER_FREQUENCY_CURVE.md` | uses -lgc CLK; 1800/2032 boost | Implicit ack |

### Stuck-at-1005 silent failure mode (CLAUDE.md memory)

The B300 GPU can sometimes be stuck at a low clock (e.g., 1005 MHz) under sustained load EVEN when:
- `nvidia-smi -q -d CLOCK` shows Application Clocks Setting: 2032 MHz
- `nvidia-smi -q -d PERFORMANCE` shows "Idle: Active" (no throttle reasons listed)
- No explicit `-lgc` lock has been applied in the current session
- `nvidia-smi --query-compute-apps` shows no processes

**Symptom:** All cuBLAS BF16 measurements collapse to ~1190 TF (consistent with 1080 MHz average) instead of the proper ~1500 TF random / ~2100 TF constant. ML inference latency spikes 2.35×.

**Fix:**
```bash
sudo nvidia-smi -rgc -i 0   # reset graphics clock
```
After reset, boost-clock behavior returns and proper measurements resume.

**Why:** Possibly a leftover transient from a prior session's `-lgc 1005` that wasn't reset, or a driver state leak. The "Idle" reason flag does NOT reliably reflect the locked state — must verify by sampling clock during an actual long-running kernel.

### Detection protocol — sample clock during run

```bash
# In one terminal, start your benchmark.
./QuickRunCUDA tests/bench_v6_c1.cu -t 256 -b 1184 -p -T 100

# In another terminal, sample SM clock at 1 Hz during the run.
nvidia-smi --query-gpu=clocks.current.sm,power.draw --format=csv,noheader -i 0 -lms 1000
```

**If clocks.current.sm < 1900 MHz under heavy load**, you have the stuck-at-1005 problem (or an unexpected `-lgc` lock). Run `sudo nvidia-smi -rgc -i 0` and re-test.

**Don't trust `nvidia-smi -q -d CLOCK` Application Clocks reporting alone — sample under load.**

### Cross-source consistency on default boost

| File | Claim |
|---|---|
| `16_power_clock.md` | "Default sustained boost = 2031.4 MHz; NEVER throttled in any tested workload" |
| `B300_TRUE_REFERENCE.md` | "Sustained 1920 MHz SM clock (boost is 2032 but rarely sustained)" |
| `POWER_FREQUENCY_CURVE.md` | Boost row labeled "2032 MHz" |
| memory `feedback_clock_stuck_no_lock.md` | **B300 can stick at 1005 MHz under load with NO explicit lock; `nvidia-smi -q` won't show it** |
| memory `feedback_clock_lock_works.md` | **`-lgc` IS honored 510-1500 MHz; "1942 floor" was background procs** |

`16_power_clock` says the chip never throttles; `B300_TRUE_REFERENCE` line 16 says boost "rarely sustained" — these are direct contradictions. Memory note on stuck-at-1005 reconciles BOTH: default boost IS 2032 in clean tests, but background processes (or silent throttle conditions) can pin it to 1005 with no warning.

### "Apparent 1942 MHz floor" — RETRACTED

Early measurements observed a "1942 MHz floor" that turned out to be leftover background procs thrashing the GPU. CLAUDE.md memory `feedback_clock_lock_works.md` confirms: **clock-lock works correctly 510-1500 MHz**. The 1942 floor was a measurement artifact.

### Lock state truth table

| Command | Effect | Use when |
|---|---|---|
| `sudo nvidia-smi -lgc 510,510` | Pins to 510 MHz | Pure-FFMA energy measurements; pJ/FFMA tests |
| `sudo nvidia-smi -lgc 1005,1005` | Pins to 1005 MHz | Reproducible mid-clock baseline; popcount sweeps |
| `sudo nvidia-smi -lgc 1500,1500` | Pins to 1500 MHz | High-clock unclipped DRAM popcount; stress tests |
| `sudo nvidia-smi -lgc 1700,1700` | Pins to 1700 MHz | DVS sweet-spot (134 GFLOPS/W) |
| `sudo nvidia-smi -lgc 1920,1920` | Pins to 1920 MHz (= -lgc 2032 paradox target) | Same as default boost (1919.8 actual) |
| **`sudo nvidia-smi -lgc 2032,2032`** | **PARADOX: pins to 1919.8 MHz, NOT 2032** | NEVER USE — silently lower than expected |
| **`sudo nvidia-smi -rgc`** | Releases lock, returns to dynamic boost | For TRUE 2031.4 MHz boost; for production benchmarks |
| `sudo nvidia-smi -pl 500` | Caps power to 500 W | Power-cap experiments; auto-throttles clock |

### Workflow recipe — always do these steps

1. **Before first measurement of session:**
   ```bash
   sudo nvidia-smi -rgc -i 0    # ensure no stale lock
   nvidia-smi --query-gpu=clocks.current.sm,power.draw --format=csv,noheader -i 0
   ```
2. **During first long-running kernel (sample clock):**
   ```bash
   nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader -i 0 -lms 500
   ```
3. **If stuck at 1005:**
   ```bash
   sudo nvidia-smi -rgc -i 0
   # then re-run
   ```
4. **For stable / reproducible measurements:** lock at 510, 1005, 1500, or 1700; NEVER 2032.
5. **For peak-throughput measurements:** `-rgc` and verify clock during run.

### Headline rule

| Goal | Setting |
|---|---|
| Reproducible energy measurements | `-lgc 1500,1500` |
| Reproducible mid-clock baseline | `-lgc 1005,1005` |
| Maximum throughput | `-rgc` + verify clock during run |
| Avoid silent slowdown | NEVER `-lgc 2032`; always sample SM clock under load |

### Open questions

1. **`-lgc 1920,1920` test** — only 2032/1410/unlocked tested explicitly. Does -lgc 1920 give a true 1920? (Plausible per V10's row, but the row is labeled "1920" because of the -lgc 2032 paradox; haven't tested literal -lgc 1920.)
2. **Why does `-lgc 2032` silently fall back to 1920?** Driver bug, or a hardware limit on application clock vs SM clock? File with NVIDIA driver team.
3. **What triggers stuck-at-1005?** Driver state leak from prior session's `-lgc`? PCIe link state change? Background proc that died with held NVML handle? Reproduction recipe unclear.

### CLAUDE.md cross-reference

The CLAUDE.md documentation explicitly addresses this:
> **Default (no nvidia-smi lock): boost to 2032 MHz under sustained FFMA load.**
> **`nvidia-smi -lgc 2032` paradoxically pins to 1920 MHz** (base clock), NOT 2032.
> ALL TFLOPS claims must state which clock state:
> - "Default boost" → 2032 MHz
> - "Locked" → 1920 MHz (6% lower)

The CLAUDE.md's note "ALL TFLOPS claims must state which clock state" is a critical methodological rule — many measurements floating around the catalog are mixed between 1920 and 2032 actual, contributing to ~6% noise in published numbers. ALWAYS specify clock state when reporting TFLOPS.

### Multi-clock measurement protocol

For any measurement that you want to be defensible:

1. **Lock the clock** at one of the verified-stable points: 510, 800, 1005, 1300, 1500, 1700, or true boost.
2. **Verify the lock took effect** by sampling clocks.current.sm during the kernel run, not before.
3. **Report the clock with the measurement.** "FFMA = 74.6 TFLOPS @ 2032 MHz boost" is good. "FFMA = 74.6 TFLOPS" is ambiguous.
4. **For boost measurements** (no lock), repeat 3 times with 30-second gaps; if numbers vary by >5%, suspect stuck-at-1005 and re-test after `nvidia-smi -rgc -i 0`.
5. **For energy measurements**, lock at a stable clock to avoid V² × f variation across the test.
6. **Don't average across mixed clock states.** Each clock state is a different operating point.

### Verifying clock state matches expectation

Quick test for any benchmark:
```bash
# In one terminal:
sudo nvidia-smi -lgc 1500
./QuickRunCUDA tests/bench_v6_c1.cu -t 256 -b 1184 -p -T 100 &
BENCHMARK_PID=$!

# In another terminal, sample for 10 seconds:
for i in {1..20}; do
    nvidia-smi --query-gpu=clocks.current.sm,power.draw --format=csv,noheader -i 0
    sleep 0.5
done

wait $BENCHMARK_PID
sudo nvidia-smi -rgc
```

Expected: 20 samples all reading "1500 MHz, ~300 W" if the lock is honored. If you see 1005 MHz instead, you have stuck-at-1005 in the locked regime — driver bug.

### Why lock at 1500 for energy?

`16_power_clock.md` chose 1500 MHz lock as the canonical reference for energy measurements because:
- Stable (no DVS variation across test)
- High enough that idle is small fraction of active
- Low enough that you don't hit TDP cap on most workloads
- Verified-honored lock state (no paradox)

For peak throughput (independent of energy), use `-rgc` and verify clock during run.

### Multi-tier verification example

Suppose you run an FFMA benchmark and report "76.96 TFLOPS at 2032 MHz". Verify with:

1. **Theoretical:** 148 SMs × 128 FP32 cores × 2 op/FFMA × 2.032 GHz = 76.96 TFLOPS theoretical. Match: 100%. PASS.
2. **Clock state:** Sample clocks.current.sm during run. Should be 2031.4 MHz (boost). If 1920, you have `-lgc 2032` paradox. If 1005, you have stuck-at-1005.
3. **Power:** Should be 361 W (high-occ ILP=24) or 437 W (low-occ). If <300 or >500, suspect issue.
4. **SASS:** Should see N FFMA instructions in the kernel where N matches the math.
5. **ncu:** `pipe_fma.avg.pct_of_peak_sustained_active` should be 90-97%.

If ALL of these check out, you have a defensible measurement. If any fail, debug.

### Common stuck-clock symptoms

| Symptom | Likely cause |
|---|---|
| cuBLAS BF16 perf collapses to ~1190 TF (vs ~1500-2100 TF expected) | Stuck-at-1005 |
| All measurements ~50-60% of expected | Stuck-at-1005 or thermal throttling |
| Power scales with clock but throughput doesn't | Wrong clock applied (paradox) |
| nvidia-smi reports "Idle: Active" but kernel runs slow | Hidden throttle reason — sample during run |
| Multi-process measurements drift | Background proc thrashing GPU; `pkill -9 QuickRunCUDA` and `sleep 5-8` |

The user MEMORY note warns: "Always `pkill -9 QuickRunCUDA` + `sleep 5-8` between measurements (lessons learned: leftover processes silently inflate cy/MMA up to 8.5×)." This is critical — leftover processes on the same GPU cause measurement contamination that can be 5-8× off from clean.

### When `-rgc` doesn't help

If `-rgc` does not restore boost behavior and you still see 1005 MHz under load, the issue is deeper:
1. Check thermal: `nvidia-smi -q -d TEMPERATURE` — if GPU temp is >80°C, it may throttle.
2. Check hardware: `nvidia-smi -q -d POWER` — if power.management is `Disabled`, lock state is sticky.
3. Reset the device: `sudo nvidia-smi --gpu-reset -i 0` (kills all CUDA contexts!)
4. Reload driver: `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm` (kills all CUDA work!)
5. Reboot — last resort.

Steps 3-5 are destructive. Do step 1 + 2 first.

### `nvidia-smi -q` field reference for clock troubleshooting

```
$ nvidia-smi -q -i 0 -d CLOCK
        Clocks
            Graphics                  : 2032 MHz       # <-- current SM clock (not application setting)
            SM                        : 2032 MHz       # <-- same (sm and graphics are same domain)
            Memory                    : 9001 MHz       # <-- HBM3E memory clock
            Video                     : 1860 MHz       # <-- L2 video clock (constant on B300)
        Applications Clocks
            Graphics                  : 2032 MHz       # <-- requested setting
            Memory                    : 9001 MHz       # <-- requested setting
        Default Applications Clocks
            Graphics                  : 1920 MHz       # <-- factory default base clock
            Memory                    : 9001 MHz
        Max Customer Boost Clocks
            Graphics                  : 2032 MHz       # <-- max boost achievable
        Performance State             : P0             # <-- current perf state (P0 = max)
        Clocks Throttle Reasons
            Idle                      : Active         # <-- THIS IS MISLEADING under stuck-at-1005
            ...
```

The `Performance State : P0` line is a better indicator than throttle reasons. Under stuck-at-1005, P-state may show P3 or P4 even though throttle reasons report Idle.

Sample under load is the gold standard. Don't trust pre-run / post-run readouts for the steady-state clock.

**Footgun:** ⚠ NEVER use `-lgc 2032` — silently pins to 1919.8 MHz (-5.5%). Use `-rgc` for true boost. ALWAYS sample `clocks.current.sm` during the first run of a benchmark session — B300 can stick at 1005 MHz silently with NO explicit lock and `nvidia-smi -q` won't show the slowdown. Symptom: cuBLAS BF16 measurements collapse to ~1190 TF instead of ~1500 TF; recover with `sudo nvidia-smi -rgc -i 0`.

**See also:** §41 (idle/TDP at each clock), §42 (DVS curve + V² × f), §44 (TF/W operating points), corrections/POWER_INCONSISTENCY_LOG.md §C+§D, V10_DVS_CURVE.md, CLAUDE.md memory feedback_clock_stuck_no_lock.md + feedback_clock_lock_works.md.

---

## Cross-section synthesis — the unifying patterns

### Pattern 1: ILP and chain-self-feed measurements differ by 100×

A repeated lesson across this section:
- V8 MUFU rsqrt = 47.8 G (chain self-feed) vs V41 saturated MUFU = 4740 G/s = **100× gap**
- V8 SHFL = 3 G warp (chain self-feed) vs V38 SHFL = 9.48 Telements/s = **100× gap** at thread level

**Always label the regime.** If you measure "X Gops/s" and someone says "X is too low for the SoL", check if you're chain-dep or independent-issue. The SoL applies to independent-issue; chain-dep measures latency/throughput product.

### Pattern 2: Tiered pipe ladder, NOT uniform "ALU"

Old framing: "ALU pipe at 19 TIOPS for all integer ops". V40 + A6 corrected: tiered ladder from 4.7 (POPC) to 26 (FFMA/IADD3) Glane/s. The tier you land in depends on the pipe family — see §27 Section C for the definitive map.

Implications:
- Don't generalize "INT is 19 TIOPS"; some are faster, some slower.
- Pick ops based on rate tier when optimizing (e.g., prefer IADD3 over LOP3+ADD when possible).
- Mix pipes for true overlap (FFMA + LDG, not FFMA + IADD3).

### Pattern 3: Output bit-width matters more than op family for cvt

V43's FP8 cvt 2× BF16 cvt anomaly is consistent with output bit-width determining whether MERGE_C is needed. This generalizes to other narrow formats (FP4 untested due to CUDA 13.2 bug, but predicted to be in the FP8 tier).

### Pattern 4: Memory power dominates total energy in HBM-bound work

Memory pipe (LDG) is +177 W vs FFMA's +24 W in M2's per-pipe table. For HBM-bound kernels, memory is 30-40× of total kernel energy. Optimization priorities should reflect this:
1. Cache-block to reduce HBM traffic (#1 lever).
2. Use FP8/NVFP4 to reduce traffic per FLOP (orthogonal lever).
3. Use LDG.128 / STG.128 for higher per-byte energy efficiency (3× over LDG.32).
4. Tensor cores save energy at the math-pipe level (5-24× FP8/NVFP4 TF/W vs FFMA), but this only matters when not HBM-bound.

### Pattern 5: Data-dependence has a 2-tier model

Per-cycle bit-flip count (toggle activity) drives power, NOT static popcount:
- Toggle component: 0-180 W at L2 / 0-240 W at DRAM — bell at d=16
- Static popcount component: 0-18 W at L2 / similar at DRAM — proportional to d

For ML inference: chunk-level dedup is dead (3% effect); only inter-dword toggle matters. Data layouts that minimize per-cycle toggling save 20-30% on memory power.

### Pattern 6: Min-energy clock is workload-DEPENDENT

There is no single "best clock for energy". Pick based on workload:
- Pure FFMA: 510 MHz (3.1 pJ/FFMA)
- Pure memory-bound: 800 MHz (11.81 pJ/byte)
- Mixed ML inference: BOOST CLOCK (3.08× lower energy than 510)
- Best instantaneous TF/W: 1500-1700 MHz

USE BOOST CLOCK FOR ML INFERENCE — this overrides naive intuition.

### Pattern 7: Always sample clock during run

`-lgc 2032` paradoxically pins to 1920. B300 sticks at 1005 silently. NEVER trust pre-run clock readouts; sample under load.

---

## Quick-reference summary tables

### MUFU and warp-op latency table

| Op | Latency cy | ns @ 2.032 GHz | Throughput (Gops/s chip) |
|---|---:|---:|---:|
| FFMA / FADD / FMUL / HFMA2 | 4.04-4.22 | 2.0-2.1 | 74.6 (FFMA) / 37.4 (FADD) |
| MUFU.EX2 (chained EX2→EX2) | 14.14 | 7.0 | 9.22 |
| MUFU.EX2 (cross-pipe FFMA→EX2→FFMA) | ~30 | ~15 | (same throughput, longer chain pen.) |
| MUFU.LG2 / SQRT / RSQ.ftz / TANH | 18 | 8.9 | 4.74 |
| MUFU.SIN / COS | 24.02 | 11.8 | 4.74 |
| MUFU.RSQ / SQRT (non-ftz IEEE) | 40.10 | 19.7 | 4.74 |
| MUFU.RCP | 42.10 | 20.7 | 4.74 |
| `redux.sync.add.u32` | ~11.6 | 5.7 | 9.09 |
| `SHFL.BFLY` (chain) | ~5.4 | 2.7 | 9.48 |

### INT/bit op throughput table (chip Glane/s @ 2032 boost; multiply 1500/2032 = 0.738 for 1500 lock)

| Op | Glane/s @ 2032 | Pipe per V40 | Same pipe as FFMA? |
|---|---:|---|:---:|
| FFMA | ~38.5 SoL | FMA | YES |
| FADD / FMUL | ~37.5 each | FMA | YES |
| IADD3 | 25-26 | FMA | YES |
| LOP3.LUT | 18.7 | INT-bit | NO |
| IMUL / IMAD (.lo) | 18.7 | FMA-pipe half-rate | shared slot |
| SHF.L/R / SHL / SHR | 18.7 | INT-bit | NO |
| BFI.b32 | ~17 | INT-bit | NO |
| PRMT | 13.9 | permute (V40) / INT-bit (A6) | NO |
| ISETP / FSETP | 8.4 | compare | NO |
| BFE.u32 | 7.07 | XU (2-SASS path) | NO |
| SHFL.IDX/BFLY/UP/DOWN | 4.7 | LSU/SHFL (MIO) | NO |
| POPC / BREV / CLZ / FLO | 4.7 | XU | NO |
| MUFU.EX2 | 9.62 (Gops/s) | MUFU (XU) | NO |
| Other MUFU | 4.74 (Gops/s) | MUFU (XU) | NO |
| REDUX | 9.09 (Telements/s) | shuffle (MIO) | NO |

### Packed FP cvt rate table (Gelem/s chip)

| PTX form | Per-PTX-elem Gelem/s | Per-SASS-inst rate | MERGE_C required? |
|---|---:|:---:|:---:|
| `cvt.rn.satfinite.e4m3x2.f32` | 17.6 | 19.3 Telem/s SoL | NO |
| `cvt.rn.satfinite.e5m2x2.f32` | 17.6 | 19.3 Telem/s SoL | NO |
| `cvt.rn.bf16x2.f32` | 9.05 | 19.3 Telem/s SoL | YES (halves rate) |
| `cvt.rn.satfinite.f16x2.f32` | 9.05 | 19.3 Telem/s SoL | YES (halves rate) |
| `cvt.rn.satfinite.e2m1x4.f32` | REJECTED CUDA 13.2 sm_103a | — | (predicted NO; predicted FP8 tier) |

### Power-clock-workload reference (W)

| Workload | 510 MHz | 1005 MHz | 1500 MHz | 1700 MHz | 1920 MHz | 2032 boost |
|---|---:|---:|---:|---:|---:|---:|
| Idle | 144 | 152 | 167 | 175 | 198 | 198 |
| FFMA peak (high-occ) | 178 | 225 | 300 | 339 | 419 | 361 |
| FFMA peak (low-occ ILP=1) | — | — | — | — | — | 437 |
| BF16 mma random | 350 | 613 | 1009 | — | 1099 (cap) | 1099 (cap) |
| BF16 mma constant | 259 | 296 | 426 | — | 629 | 629 |
| FP8 cuBLAS sustained | — | — | — | — | — | 886 |
| HBM streaming d=16 | ~553 | 787 | **1071** | TDP cap | TDP cap | TDP cap |
| Stress-recipe target | — | — | **1071** ← USE | TDP cap | TDP cap | TDP cap |

### Energy per op (pJ/op, 1500 MHz lock; multiply by V²×f for other clocks)

| Op | pJ/op | vs FFMA |
|---|---:|---:|
| FFMA (with .reuse) | 4.4 | 1.0× |
| FFMA (no .reuse) | 6.5 | 1.49× |
| FADD / FMUL | ~4.0 | ~0.9× |
| IADD3 | 6.5 | 1.5× |
| LOP3 | 3.85 | 0.88× |
| IMAD chain | 6.5 | 1.5× |
| MUFU rsqrt.ftz | ~4.5 | 1× |
| MUFU sin (non-ftz) | ~6.8 | 1.5× |
| LDG cold HBM | ~96 | 22× |
| LDG L1 hit | ~5 | 1.1× |
| LDG L2 bypass (.cg) | ~10 | 2.3× |
| LDS u32 | ~4 | 0.9× |
| LDS.128 | ~10 | 2.3× |
| STS.128 | ~13 | 3× |
| STG.128 | ~120 | 27× |
| HMMA BF16 mma.sync (per output) | ~50 | 11× |
| Branch predictable | 14.8 | 3.4× |
| Branch divergent half-warp | 18.5 | 4.2× |
| __syncthreads | ~30 | 6.8× |
| cluster.barrier | ~390 | 89× |

### Per-byte energy at 1005 MHz (memory subsystem)

| Operation | nJ/byte |
|---|---:|
| L2 read (d=16 random) | 25.5 |
| L2 write (d=16) | 62.2 |
| DRAM read (d=16) | 86.1 |
| DRAM write (d=16) | 115.7 |

### Compiler flag impact summary

| Flag | Speedup | Energy reduction |
|---|---|---|
| `-use_fast_math` | 3.28× | **4.05× less energy** |
| `__forceinline__` | 15.2× | ~15× less |
| `.reuse` annotation | 1.32× | **1.96× less** (49%) |
| `-Xptxas=-O3` | 4.4× | ~4× less |
| LDG.128 over LDG.32 | 2.95× | ~3× less |
| REDUX over SHFL chain (INT) | 2.34× | ~2× less |
| Boost clock for ML | varies | **3.08× less** vs 510 MHz |

---

## Documentation cross-references

For deeper detail on each topic in Section D, refer to:

| Section | Topic | Primary source | Cross-corroboration |
|---|---|---|---|
| §36 | MUFU EX2 anomaly | V41_V48_FINDINGS.md§"ALU pipe (V41)" | 14_math_intrinsics_CORRECTED.md, MATH_INCONSISTENCY_LOG.md |
| §37 | MUFU latency split | CHAIN_FP_MUFU_LATENCY.md | V41_V48_FINDINGS.md, V8_MUFU_PEAK.md |
| §38 | SHFL/REDUX equal raw rate | V41_V48_FINDINGS.md V37+V38 | Q3_WARP_REDUCE_RECIPES.md, V8_SHFL_PEAK.md |
| §39 | INT/bit pipe ladder | V41_V48_FINDINGS.md V40 | A6_PER_PIPE_REFERENCE.md, B1_DUAL_ISSUE_FFMA_IADD3.md, C3_LOP3_LUT_DEEP.md, INT_INCONSISTENCY_LOG.md, V8_IMAD_PEAK_VERIFIED.md |
| §40 | Packed FP cvt | V41_V48_FINDINGS.md§"Packed FP cvt" | 05_fp_precision_nontensor_CORRECTED.md, F2FP_DEEP_DIVE.md |
| §41 | Power floor + ceiling | 16_power_clock_CORRECTED.md | POWER_FREQUENCY_CURVE.md, POWER_FLOOR.md |
| §42 | DVS curve + min-energy clock | M9_ENERGY_PARETO.md, V10_DVS_CURVE.md | M2_ENERGY_LADDER.md, M11_PER_PIPE_ENERGY.md, POWER_INCONSISTENCY_LOG.md§I |
| §43 | Popcount bell, HBM data-dep | POPCOUNT_3TIER.md, POPCOUNT_VS_CLOCK.md | POPCOUNT_WRITES.md, L2_POPCOUNT_SWEEP.md, POWER_DATA_DEPENDENCE_SUMMARY.md, STRAYS_CORRECTED.md§2 |
| §44 | Per-pipe energy reconciliation | M11_PER_PIPE_ENERGY.md, M2_ENERGY_LADDER.md | 16_power_clock_CORRECTED.md§3, POWER_INCONSISTENCY_LOG.md§J |
| §45 | Clock-lock paradox + stuck | 16_power_clock_CORRECTED.md§1 | V10_DVS_CURVE.md, B300_TRUE_REFERENCE.md, CLAUDE.md memory |

---

## Headline numbers cheat-sheet (one-line answers)

For copy-paste into reports / slides / quick-reference:

| Question | Answer |
|---|---|
| What is B300 SXM6 AC sm_103a TDP? | **1100 W enforced** (NVML) |
| Default boost clock? | **2031.4 MHz** under sustained FFMA |
| Idle power (true sleep)? | **120 MHz / 144 W** |
| Idle power (alive at boost)? | **2032 MHz / 198 W** |
| Highest-power sustained workload? | **DRAM read d=16 random + 1500 MHz lock = 1071 W** |
| FFMA peak throughput? | **74.6 TFLOPS at 2031.4 MHz boost** (97% of 76.96 theoretical) |
| FFMA TF/W (peak operating point)? | **0.21 TF/W** (74.6 TF / 361 W) |
| FFMA TF/W (M11 alternative number)? | 0.111 TF/W (different operating point — UNRESOLVED) |
| FP8 cuBLAS TF/W? | **5.07 TF/W** (4491 TF / 886 W) |
| BF16 mma.sync TF/W? | 1.39 TF/W (569 TF / 411 W) |
| MUFU.EX2 throughput? | **9.22 Gops/s** (2× faster than other MUFU) |
| MUFU non-EX2 throughput? | **4.74 Gops/s** (LG2/RCP/RSQRT/SQRT/SIN/COS) |
| MUFU.RCP latency? | 42 cy (single chain) |
| MUFU.EX2 chained latency? | 14 cy chain-self / ~30 cy cross-pipe |
| FP32 → FP8 packed cvt rate? | **17.6 Gelem/s** chip |
| FP32 → BF16 packed cvt rate? | **9.05 Gelem/s** chip (2× slower than FP8) |
| LOP3 throughput? | 18.7 Glane/s = 0.5 inst/SMSP/cy |
| IADD3 throughput? | 25-26 Glane/s = 0.66 inst/SMSP/cy (FMA pipe per V40) |
| FFMA + IADD3 dual-issue overlap? | **14-17%**, NOT 2× and NOT 131% |
| ISETP throughput? | 8.4 Glane/s = 0.25 inst/SMSP/cy |
| POPC throughput? | 4.7 Glane/s |
| SHFL throughput? | **9.48 Telements/s** (= REDUX, same shuffle pipe) |
| REDUX vs SHFL chain (algorithm)? | **2.34× faster** for warp-reduce-32 |
| Min-energy clock for FFMA pJ/op? | 510 MHz |
| Min-energy clock for memory pJ/byte? | 800 MHz |
| Min-energy clock for ML inference per-task? | **Boost (1992-2032 MHz) — 3.08× lower energy than 510** |
| Best GFLOPS/W for FFMA? | 1500-1700 MHz @ 134 GFLOPS/W |
| What does `-lgc 2032` actually pin to? | **1919.8 MHz** (paradox — never use) |
| How to get true 2031.4 MHz boost? | Use `-rgc` (release lock entirely) |
| How to detect stuck-at-1005? | Sample `clocks.current.sm` during run; if <1900 under load with no `-lgc`, suspect stuck |
| How to recover from stuck-at-1005? | `sudo nvidia-smi -rgc -i 0` |
| HBM data-dep power swing at 1005 MHz? | **240 W** (d=0 vs d=16, DRAM-8G) |
| HBM data-dep power swing at 1500 MHz? | **554 W** (d=0 vs d=16, DRAM-8G) |
| HBM_DATA_DEPENDENCE.md "<50W" claim? | **WRONG by 5-7×; SUPERSEDED** |
| Chunk-level dedup hypothesis? | **NULL RESULT** (3% spread) |

## Verification protocol for any new measurement

The CLAUDE.md "B300 Benchmarking Methodology" gives the rigor protocol. Applied to Section D's domain:

1. **For MUFU/SHFL/REDUX rates:** Run independent-issue ILP-saturated test (V41 style); verify ncu pipe util > 90%; SASS-check inst count; cross-check at 2 different clocks.
2. **For INT/bit pipe placement:** Run V40-style sweep (persistent grid + asm-volatile); verify ncu `pipe_*` per pipe; check overlap with FFMA in mixed kernel.
3. **For packed FP cvt rates:** Time to single-element rate; SASS-check for MERGE_C presence/absence; cross-check with NVRTC + cuBLAS-internal cvt.
4. **For power measurements:** Lock clock, sample power 5+ times in 0.5 s windows after 1.5 s ramp; median of middle 3; verify clock with `nvidia-smi --query-gpu=clocks.current.sm`.
5. **For data-dependence:** Use deterministic per-dword popcount with bit positions varying per dword (Fisher-Yates); compare against constant-pattern control to isolate per-dword popcount vs inter-dword toggle.
6. **For energy:** Measure at fixed clock to avoid V²×f variance; compute pJ/op = active power × time / op count; compare against M2/M11 reference table.
7. **For clock state:** Sample `clocks.current.sm` during run, NOT before. If <1900 under heavy load and no `-lgc` set, suspect stuck-at-1005.

If your number agrees with the corrected references in this section to within 5-10%, you have a defensible measurement. If it disagrees, identify which check failed and why.

---

## Worked examples — applying Section D to real kernel design

### Example 1: Softmax kernel performance budget

A 4096-wide softmax row, BF16 input/output, FP32 internals:

Per-row operations:
- 4096 LDG (FP32 load): 4096 / 32 lanes = 128 warp-LDG = ~5000 cy at high latency
- 4096 max-reduce per warp + cross-warp: ~30 cy SHFL chain × log(128 warps) = ~210 cy
- 4096 exp(x - max): 4096 EX2 / 32 lanes = 128 warp-EX2 at 1/(4cy)/SMSP = 512 cy/SMSP / 4 SMSPs = 128 cy
- 4096 sum-reduce: same as max
- 4096 div by sum: 1 RCP (32 cy chained) per element / 32 lanes = 128 RCPs / 4 SMSPs = 32 cy/SMSP × 32 cy = 1024 cy
- 4096 STG (BF16 store, after cvt): 128 warp-STG

Total: ~5000 (load) + 210 + 128 + 210 + 1024 + 1500 (store) = ~8000 cy per row.

EX2's 2× anomaly contributes 128 cy (instead of 256 cy if EX2 were at non-EX2 rate). Saves 128 cy = 1.6% of total. Marginal in this regime — load + RCP dominate.

For a 1024-wide softmax row (typical Q×K attention), the regime shifts: load is 5x smaller, RCP and EX2 become more prominent. EX2 anomaly may save ~5-10% of total.

### Example 2: BF16 GEMM energy budget

cuBLAS BF16 mma at M=N=K=8192, sustained boost clock:
- Total ops: 2 × 8192^3 = 1.1 PFLOP
- Time: 1.1 PFLOP / 1500 TFLOPS = 0.73 ms (estimate based on cuBLAS BF16 ~1500 TF random)
- Power during compute: ~411 W
- Energy per call: 411 × 0.73e-3 = **300 mJ per GEMM call**
- Per-FLOP: 300e-3 J / 1.1e15 FLOP = 0.27 pJ/FLOP for BF16 mma

Compare to FFMA: 4.4 pJ/FFMA = 2.2 pJ/FLOP. BF16 mma is **8× more energy-efficient per FLOP** than FFMA.

For FP8 cuBLAS (4491 TF/W vs BF16's 569 TF/W = 7.9×): per-FLOP energy is ~0.034 pJ/FLOP. **65× more efficient than FFMA.** This is the lesson of going FP8.

### Example 3: HBM-bound kernel — choose clock for energy

A simple memcpy kernel, 1 GB → 1 GB:
- HBM peak: 7.2 TB/s read + 7.2 TB/s write
- Time: 1 GB / 7.2 TB/s × 2 (read+write) = 0.28 ms
- Power at 800 MHz lock: ~460 W (mem-bound saturates at 800)
- Energy: 460 × 0.28e-3 = 129 mJ
- Per-byte: 129 mJ / 2 GB = 64 nJ/byte (matches POPCOUNT_3TIER d=16 active rate)

If you ran the same memcpy at boost (2032 MHz):
- Time: same 0.28 ms (HBM-bound, doesn't speed up)
- Power: ~460 W + idle delta = ~510 W
- Energy: 510 × 0.28e-3 = 143 mJ (~10% more)

So: lock at 800 MHz for memcpy / mem-bound. Saves 10-15% energy with no perf penalty.

### Example 4: ML inference at boost clock

LLaMA-3 70B at sequence length 8192, batch 1, FP8 precision:
- Total work: ~140 GFLOPS per token (rough estimate)
- Throughput at boost: 40 tokens/s (CLAUDE.md memory)
- Power at boost: ~600 W (ML inference is mixed; not pure mma)
- Energy per token: 600 / 40 = 15 J/token

If you locked to 510 MHz (DVS-down):
- Throughput: ~13 tokens/s (memory + compute mixed)
- Power: ~250 W
- Energy per token: 250 / 13 = 19.2 J/token (28% MORE energy)

CLAUDE.md memory says "clock-lock was 2.35× bottleneck" — this aligns with 40/13 = ~3× throughput penalty for going to 510 lock; the 2.35× number is from a different (likely 1005) lock tested.

USE BOOST for ML inference. The energy advantage is real and counterintuitive.

### Example 5: Quantization kernel — choose cvt format

A FP32 → FP8 quantization kernel for inference activations, 1 M elements:
- Cvt rate: 17.6 Gelem/s (FP8 packed)
- Time: 1e6 / 17.6e9 = 0.057 ms (microseconds!)
- Per-element energy: ~5 pJ
- Total energy: 5e-12 × 1e6 = 5 µJ (basically free)

Compare to BF16 storage path (FP32 → BF16, 9.05 Gelem/s):
- Time: 1e6 / 9.05e9 = 0.11 ms (2× longer)
- Total energy: ~10 µJ (still negligible)

Cvt kernel choice rarely matters for quantization energy. Choose based on downstream needs (FP8 for tcgen05.mma input; BF16 for next layer).

### Example 6: Diagnosing a slow benchmark

You measure FFMA at 38 TFLOPS (expected: 75 TFLOPS). Diagnostic walk:

1. **Check clock during run:** `nvidia-smi --query-gpu=clocks.current.sm`. If 1005 MHz, you have stuck-at-1005. Run `sudo nvidia-smi -rgc -i 0` and re-test.
2. **Check for `-lgc 2032` paradox:** if locked to 2032 → actually 1920 MHz → expect 70.6 TFLOPS. If you measured 38, that's not the paradox.
3. **Check power:** if 200 W instead of 360 W, you're under-saturated. Increase ILP / occupancy.
4. **Check SASS:** verify N FFMA inst in inner loop. If half what expected, DCE eliminated half the work.
5. **Check ncu pipe_fma util:** should be 95-99%. If <80%, you have a different bottleneck.

If all check out and you still see 38 TFLOPS, you have low-occupancy artifact (M11's 437 W / 39.7 TFLOPS regime). Run with full grid + persistent + max ILP.

### Example 7: DRAM-bound kernel that also saturates power

A DRAM streaming kernel at 1500 MHz with random d=16 data:
- BW: ~7 TB/s (HBM peak)
- Power: 1071 W (TDP wall at d=16)
- Energy: 1071 W × 1 second of work = 1.07 kJ per 7 TB transferred = 153 nJ/byte

If you can shape your data to d=8 (e.g., quantization with skewed distribution):
- BW: ~7 TB/s (same — BW is content-independent)
- Power: ~835 W (per POPCOUNT_VS_CLOCK d=8 row)
- Energy: 835 × 1 = 835 J per 7 TB = 119 nJ/byte (22% energy reduction)

This is the practical lever: reshaping data popcount distribution buys 20-30% energy in HBM-bound kernels with NO perf cost. Useful for inference at scale.

## Common errors to watch for in new sub-agent measurements

Per the user MEMORY notes:

1. **Don't generalize MUFU rates** — EX2 is 2× faster than other MUFU; the 9.22 vs 4.74 split is real.
2. **Don't quote "REDUX 4× SHFL"** — folklore; real algorithm-level is 2.34×, raw rates are equal.
3. **Don't claim "all ALU at 19 TIOPS"** — tiered ladder, ranges 4.7 to 26 Glane/s.
4. **Don't quote a single "FFMA TF/W"** — operating-point dependent; specify ILP/occupancy.
5. **Don't trust HBM_DATA_DEPENDENCE.md "<50W swing"** — wrong by 5-7×; superseded.
6. **Don't use `-lgc 2032`** — silently pins to 1920; use `-rgc` for true boost.
7. **Don't trust pre-run clock readouts** — B300 sticks at 1005 silently; sample during run.
8. **Don't confuse "per-SASS-inst" and "per-PTX-element"** — for cvt and IADD3, these differ by 2×.
9. **Don't claim FP16/BF16 packed FMA gives 2× over FP32** — outside tensor cores, all run at same FMA-pipe rate.
10. **Don't quote "B300 TDP = 700 W"** — Hopper carry-over; real is 1100 W.

---

## End of Section D

Sections covered: §36 (MUFU EX2 anomaly), §37 (MUFU latency split + V8 47.8G retraction), §38 (SHFL = REDUX raw rate, "4× SHFL" myth), §39 (INT/bit pipe ladder, "114 TOPS combined" retraction), §40 (packed FP cvt — FP8 2× BF16 cvt), §41 (TDP 1100 W + clock-dependent idle), §42 (DVS V²×f + min-energy clock workload-dependent), §43 (popcount bell + HBM_DATA_DEPENDENCE supersession), §44 (M11 vs 16_power_clock 2× FFMA TF/W discrepancy), §45 (clock-lock paradox + stuck-at-1005).

Authoritative sources cited: 14_math_intrinsics_CORRECTED.md, 15_integer_bit_ops_CORRECTED.md, 16_power_clock_CORRECTED.md, 05_fp_precision_nontensor_CORRECTED.md, MATH_INCONSISTENCY_LOG.md, INT_INCONSISTENCY_LOG.md, POWER_INCONSISTENCY_LOG.md, STRAYS_CORRECTED.md, V41_V48_FINDINGS.md, V32_V40_FINDINGS.md, CHAIN_FP_MUFU_LATENCY.md, V8_MUFU_PEAK.md, V8_SHFL_PEAK.md, Q3_WARP_REDUCE_RECIPES.md, POPCOUNT_3TIER.md, POPCOUNT_VS_CLOCK.md, POPCOUNT_WRITES.md, L2_POPCOUNT_SWEEP.md, POWER_DATA_DEPENDENCE_SUMMARY.md, POWER_FREQUENCY_CURVE.md, V10_DVS_CURVE.md, M2_ENERGY_LADDER.md, M11_PER_PIPE_ENERGY.md, M9_ENERGY_PARETO.md, B300_TRUE_REFERENCE.md, C3_LOP3_LUT_DEEP.md, A6_PER_PIPE_REFERENCE.md, B1_DUAL_ISSUE_FFMA_IADD3.md, V8_IMAD_PEAK_VERIFIED.md, F2FP_DEEP_DIVE.md, POWER_FLOOR.md, BF16_PERBIT_POWER.md, FP8_KVARY_POWER.md, DISABLE_LANE_POWER.md, L2_BITSTRIDE_SWEEP.md, L2_DRAM_DATA_PWR.md, CLAUDE.md memory (feedback_clock_stuck_no_lock + feedback_clock_lock_works + feedback_microbench_rigor).

Lane fences observed: pipe placement DETAIL deferred to §27 (Section C); tensor power per CTA deferred to §51 (Section E).
