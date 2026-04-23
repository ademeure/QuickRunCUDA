# §17 MUFU per-op throughput verification

## Test (`tests/bench_mufu.cu`, 1 warp on 1 SM, `-lgc 1800`)

Single-warp test: 1 block × 32 threads. Hot loop with N_CHAINS independent
MUFU operations × ITERS iters × UNROLL=8. cy/op = wall_ms × 1800 cy/µs /
(ITERS × N_CHAINS). Measures **per-warp throughput** (only 1 of 4 SMSPs
active, but MUFU pipeline saturated).

## Results — SASS-confirmed via grep

### Per-warp throughput at saturation (N_CHAINS=16, full ILP)

| PTX op | SASS | cy/op/warp | Rate vs FP32 FMA |
|--------|------|-----------:|------------------|
| ex2.approx.f32 | MUFU.EX2 | **4.28** | **~1/4** |
| tanh.approx.f32 | MUFU.TANH | 8.07 | ~1/8 |
| cos.approx.f32 | MUFU.COS | 8.56 | ~1/8 |
| sin.approx.f32 | MUFU.SIN | 8.56 | ~1/8 |
| rsqrt.approx.f32 | MUFU.RSQ | 8.74 | ~1/8 |
| sqrt.approx.f32 | MUFU.SQRT | 8.74 | ~1/8 |
| lg2.approx.f32 | MUFU.LG2 | 8.74 | ~1/8 |
| rcp.approx.f32 | MUFU.RCP | 9.09 | ~1/8 |

### Latency vs throughput (cy/op vs N_CHAINS)

| OP | N=1 (latency) | N=2 | N=4 | N=8 | N=16 (saturated) |
|----|--------------:|----:|----:|----:|------------------:|
| RCP | 43.82 | 25.10 | 13.54 | 9.75 | 9.10 |
| EX2 | 18.12 | 9.28 | 5.44 | 4.57 | 4.29 |
| SIN | 24.45 | 13.11 | 8.83 | 8.74 | 8.56 |

## Key findings

1. **EX2 is 2× faster than all other MUFU ops** (4.28 cy vs ~8-9 cy
   throughput). Has its own dedicated path or runs at higher rate.
2. **Most MUFU ops cluster at ~8.5 cy/op throughput** (RCP/RSQ/SQRT/LG2/SIN/COS/TANH).
3. **RCP slightly slowest** at 9.09 cy/op.
4. **Latency varies**: RCP ~44 cy, SIN ~24 cy, EX2 ~18 cy. Pipeline depth ≈ 4-5 stages.
5. **TANH (Hopper+ MUFU op) is fully supported on B300** at ~8.07 cy/op throughput.

## Rate interpretation

For B300 sm_103a at 1800 MHz, per-warp:
- EX2: 1.8e9 / 4.28 = **421 M ops/s/warp**
- RCP: 1.8e9 / 9.09 = 198 M ops/s/warp

Per SM (assuming MUFU pipe is per-SMSP, 4 SMSPs/SM):
- EX2: ~1.7 G ops/s/SM
- RCP: ~0.79 G ops/s/SM

For full grid (148 SMs):
- EX2: ~250 G ops/s
- RCP: ~117 G ops/s

## Catalog comparison

The catalog's claimed 10.5/13.9/15.5 cy figures seem to correspond to
**low-ILP** measurements (N=2-4 chains):

| Catalog claim | Closest match in my data |
|---------------|--------------------------|
| 10.5 cy | EX2 at N=2 (9.28) or SIN at N=4 (8.83) |
| 13.9 cy | SIN/COS at N=2 (13.11) or RCP at N=4 (13.54) |
| 15.5 cy | RCP at N=4 (13.54) or harder to match |

At full saturation (N=16), all my numbers are LOWER than catalog (4-9 cy vs
10.5-15.5). The catalog may have been measured at lower ILP or at lower clock.

## Pipeline structure inference

EX2 at 1/4 rate strongly suggests it's on the **same fast SFU pipe as the
faster Hopper FMA-fast-math sub-paths** (e.g., used by CUDA fast_math
expf/exp2f intrinsics).

Other MUFU ops at 1/8 rate use a **slower iterative pipe** — consistent
with the NEWTON-like iterative refinement needed for RCP/SQRT/etc accurate
to "approx" tolerance (4 ULP for single-precision approx).

SIN/COS at 1/8 rate (despite needing range reduction + polynomial eval)
suggests B300 has a dedicated trig accelerator, not just sequential
polynomial approximation.

## Practical implications

1. **Replace `rcp(x)` with `1.0f/x`** if you need single-divide: NVCC may
   actually use FFMA Newton-Raphson (~5 cy) instead of MUFU.RCP (9 cy).
2. **EX2 is your friend** for activation functions: GELU/SiLU using EX2 is
   significantly cheaper than using LOG/SIN-based formulas.
3. **TANH compiles to single MUFU.TANH** at ~8 cy/op — preferred over
   manual `(ex2(2x) - 1) / (ex2(2x) + 1)` which would cost 2 EX2 + RCP +
   FFMA = 4.3 + 4.3 + 9.1 + 4 = 22 cy.
4. **High ILP (8+ independent chains) needed** to saturate MUFU pipe.
   Without ILP, latency dominates (RCP 44 cy, SIN 24 cy, EX2 18 cy).
