# B300 Math Intrinsic Throughput — Clean Reference (FP32 Transcendentals + Division)

**Scope:** FP32 transcendentals (`sin`/`cos`/`exp`/`log`/`exp2`/`log2`/`tanh`/`rsqrt`/`sqrt`) and division on the MUFU (Multi-Function Unit, "pipe_xu") vs the FFMA pipe baseline. Approximation vs IEEE-correct paths.

**GPU:** NVIDIA B300 SXM6 AC (sm_103a, 148 SMs).
**Clock:** 2032 MHz boost (test was nvcc-built `math_throughput.cu`, not under `-lgc`).
**Driver:** 580.126.09, CUDA 13.2.

Confidence: **HIGH** = SASS+ncu cross-checked. **MED** = catalog cross-section consistent. **LOW** = single test, methodology suspect.

---

## 1. Pipe identity (HIGH)

The MUFU = Multi-Function Unit = `pipe_xu` (transcendental unit). It is a **separate physical pipe** from FFMA (`pipe_fma`). Native MUFU SASS opcodes:

| PTX `*.approx.f32` | SASS | Latency (clean chain) | Throughput |
|---|---|---:|---:|
| `ex2.approx.f32` | `MUFU.EX2` | **14 cy** | 0.5–0.63 issue/SMSP/cy (best MUFU) |
| `tanh.approx.f32` | `MUFU.TANH` | 18 cy | 0.5 |
| `sin.approx.f32` | `MUFU.SIN` (+FMUL range-red) | 24 cy | 0.5 |
| `cos.approx.f32` | `MUFU.COS` | 24 cy | 0.5 |
| `lg2.approx.ftz.f32` | `MUFU.LG2` | 18 cy | 0.5 |
| `lg2.approx.f32` (non-ftz) | + scaling | 40 cy | 0.5 |
| `rsqrt.approx.ftz.f32` | `MUFU.RSQ` | 18 cy (ftz) / 40 (non-ftz) | 0.5 |
| `sqrt.approx.f32` | `MUFU.SQRT` | 18 / 40 | 0.5 |
| `rcp.approx.f32` | `MUFU.RCP` | 42 cy (incl. NR FMAs) | ~0.4 |

Steady-state **chip-wide MUFU peak ≈ 4.8 TGOps/s** (16 inst/SM/cy × 148 SM × 2.032 GHz). `ex2` is **special — 1.7× faster** at ~8.1 TGOps/s (27 inst/SM/cy chip-wide).

Source: `B300_PIPE_CATALOG.md` lines 100-106, 390-397, 504, 7700-7720, 12440-12462, 16770-16790.

---

## 2. Headline numbers — `investigations/math_throughput.cu` (commit 63e3def) (MED)

**Methodology**: 4 independent ILP chains per thread, 148 blocks × 128 threads, 100 K iters of `a=op(a); b=op(b); c=op(c); d=op(d)`. Standalone `nvcc` build (NOT through QuickRunCUDA harness, so **NO `-use_fast_math`**). Reports best-of-3 event-timed.

| Op | Gops/s (chip) | Slowdown vs FMA | SASS path (likely) |
|---|---:|---:|---|
| FMA baseline | **30,266** | 1.0× | FFMA (60.5 TFLOPS = 78.5% of 76.96 theoretical) |
| `exp2f` | **5,956** | 5.1× | MUFU.EX2 |
| `__sinf` | 4,451 | 6.8× | MUFU.SIN + FMUL range-red |
| `__cosf` | 4,451 | 6.8× | MUFU.COS + FMUL range-red |
| `__expf` | 3,253 | 9.3× | MUFU.EX2 + ln2 scale |
| `__logf` | 2,732 | 11.1× | MUFU.LG2 + scale |
| `tanhf` | 1,788 | 16.9× | MUFU.EX2 + FMA chain |
| `log2f` | 1,039 | 29× | MUFU.LG2 + ftz-handling FMAs |
| `sqrtf` | 687 | 44× | **`sqrt.rn.f32` ≈ 138 cy NR chain (without fast-math)** |
| `1.0f / x` | 492 | 62× | **`div.rn.f32` ≈ 243 cy NR chain (without fast-math)** |
| `__frsqrt_rn` | 322 | 94× | MUFU.RSQ + 7-FMA refinement, **self-dep latency-bound** |

---

## 3. The "rsqrt is slowest" anomaly — RESOLVED (HIGH)

The 322-Gops/s `__frsqrt_rn` number is **NOT** evidence the MUFU rsqrt path is slow. The catalog's separate measurement shows `__frsqrt_rn` is the **fastest** MUFU (`bench_mufu_audit`: 727 inst/ns at 8 independent chains; 2.69× faster than `rsqrtf`).

The math_throughput.cu number is **bottom-low because of self-dependency latency**:

- Test pattern is `a = __frsqrt_rn(a); b = __frsqrt_rn(b); c=...; d=...`.
- `__frsqrt_rn` (round-to-nearest) emits **MUFU.RSQ + Newton-Raphson refinement FMAs** to reach IEEE-rn, total ~28-40 cy latency.
- With only 4 chains per thread and self-dep (a feeds a), the warp can't hide a 30-cy latency. Throughput is `4 / latency`, not `pipe_throughput`.
- Same root cause for `1/x` (243 cy `div.rn.f32` when fast-math disabled) and `sqrtf` (138 cy `sqrt.rn.f32`).

**`sqrt`, `div`, `rsqrt` numbers in math_throughput.cu are LATENCY-BOUND, not pipe-bound. The MUFU pipe itself runs at 4.8-8 TGOps/s under saturation.**

For a pipe-bound measurement use `_use_fast_math` builds with ≥ 8 independent chains and `*.approx.ftz.*` PTX, where rsqrt and sqrt sit at ~13-14 cy and ~5 TGOps/s — same range as sin/cos.

Sources: catalog lines 4575-4582 (`__frsqrt_rn` fastest), 6456-6469 (fast-math vs exact 2.3-2.7× speedup), 7359-7370 (div.rn = 243 cy vs div.approx = 5.5 cy = 44× difference).

---

## 4. Approximation vs IEEE-correct (HIGH)

The most important practical distinction. All numbers @ 2032 MHz, single-warp latency.

| Operation | Approx (`.approx.ftz.f32` / `__*_rn`) | IEEE-correct (`.rn.f32`) | Ratio |
|---|---:|---:|---:|
| `1/x` (rcp) | `div.approx.f32` ≈ 5.5 cy | `rcp.rn.f32` ≈ 78-185 cy | 14-34× slower IEEE |
| `a/b` (div) | `div.approx.f32` ≈ 5.5 cy | `div.rn.f32` ≈ 50-243 cy | 9-44× slower IEEE |
| `sqrt` | `sqrt.approx.f32` ≈ 13-14 cy | `sqrt.rn.f32` ≈ 56-138 cy | 4-10× slower IEEE |
| `rsqrt` | `__frsqrt_rn` / `rsqrt.approx.ftz` ≈ 13 cy | precise `rsqrtf` ≈ 28-40 cy | 2.7× slower IEEE |
| `exp2`, `log2`, `sin`, `cos`, `tanh` | MUFU only | n/a (no IEEE-correct path mandated) | — |

**Compile defaults matter**:
- QuickRunCUDA harness: `-use_fast_math` ON by default (`utils/cuda_helper.h:227`). All `/`, `sqrtf`, `1.0f/x` get the approximate path.
- Standalone `nvcc` (default): NO `-use_fast_math`. `1.0f / x` becomes `div.rn.f32` (~243 cy), `sqrtf(x)` becomes `sqrt.rn.f32` (~138 cy). This is the trap that makes the math_throughput.cu numbers look bad.

---

## 5. MUFU + FFMA co-issue — CONFLICT in catalog (MED)

Two contradicting findings exist on whether MUFU runs in the shadow of FFMA:

**Finding A (catalog line 15846-15856, "MUFU is FREE")**: 1 MUFU + 4 FMA per warp = 25 cy vs 24 cy FMA-only. **MUFU adds +1 cy** when interleaved with FMA-bound code.

**Finding B (catalog line 6630-6644, "MUFU NOT free under FFMA2 pressure")**: At 1 rsqrt per 8 FFMA2, FP32 throughput drops by **60%** (71.8 → 28.8 TFLOPS). Stall reason: `math_pipe_throttle` = 3.92.

**Reconciliation**: Both are correct in their regimes:
- **At low MUFU density** (≤ ~1 MUFU per 8-16 FMA), MUFU hides in the latency/scheduler shadow. Co-issue works (+1 cy).
- **At high MUFU density** with fully-saturated FFMA2 (the half-rate scalar FFMA path), MUFU shares some issue resources with the FMA dispatcher and serializes. Use the dual-issue catalog (line 7960) which shows triple-issue with MUFU degrades to 0.31 inst/cy/lane — MUFU is then the bottleneck, not FFMA.
- The "MUFU dominates / hides everything" claim (catalog line 16742-16750) is the right mental model: when MUFU is on the critical path (which it is at any non-trivial density given its 24 cy latency), FFMA, ALU, LSU, SHFL all hide in MUFU's shadow. That's the +0.4 to +2 cy "co-issue cost" pattern.

**Practical**: For softmax/GELU/normalization kernels, place the few MUFU ops in a separate pipeline phase — don't interleave 1:1 with FMAs unless MUFU count ≪ FMA count.

---

## 6. Function variant equivalence (HIGH)

| Function | cy/iter (single chain) | vs `exp2f` | SASS path |
|---|---:|---:|---|
| `exp2f(x)` | **20.1** | **1.00×** | MUFU.EX2 only |
| `__expf(x)` (fast) | 24.2 | 1.20× | MUFU.EX2 + ln2 scale FMUL |
| `expf(x)` (standard) | 24.2 | 1.20× | **identical SASS to `__expf`** |
| `tanhf(x)` | 24.1 | 1.20× | MUFU.EX2-based |
| `__logf(x)` (fast) | 27.4 | 1.36× | MUFU.LG2 + scale |
| `logf(x)` (standard) | 27.4 | 1.36× | **identical SASS to `__logf`** |
| `rsqrtf(x)` | 28.2 | 1.40× | MUFU.RSQ + FMA refine |
| `__frcp_rn(x)` (IEEE rcp) | 86.9 | 4.32× | MUFU.RCP + Newton-Raphson |

**On B300 (with `-use_fast_math`), `__expf`/`__logf`/`__sinf`/`__cosf` produce IDENTICAL SASS to `expf`/`logf`/`sinf`/`cosf`.** The "fast intrinsic prefix" is a no-op. Use either; pick for clarity.

**`exp2f(x * 1.4427f)` is 17% cheaper than `expf(x)`** because it skips the ln2 pre-multiply. For softmax inner loops, prefer.

Source: catalog lines 16775-16791.

---

## 7. Practical Gops/s/SM peaks (HIGH, single per-SM rates)

Steady-state MUFU output normalised per SM (from `bench_mufu_*` consolidated):

| Function | Gops/s/SM | Chip Gops/s (×148) |
|---|---:|---:|
| `exp2f` (best — MUFU.EX2 dedicated path) | **34.9** | 5,165 |
| `__expf` / `expf` | 30.6 | 4,529 |
| `sqrtf`, `rsqrtf`, `__logf`, `log2f` | 22.5 | 3,330 |
| `__sinf`, `__cosf` | 20.6 | 3,049 |
| `__tanf` | 8.7 | 1,288 |
| `__frcp_rn` (precise) | 3.8 | 562 |

Source: `investigations/CONSOLIDATED_FINDINGS.md` lines 209-217.

---

## 8. Design rules (synthesis)

1. **Use `exp2f` over `expf`** in softmax: 17% cheaper (saves 1 FMA per call), same SASS structure, no precision loss for masked/clipped inputs.
2. **Use `log2f` over `logf`** for the same reason; both use MUFU.LG2 underneath.
3. **Use `__frsqrt_rn(x) * x`** for normalization, NOT `sqrtf(x)` then `1/x` (saves ~30 cy → ~5 cy with fast-math).
4. **Never `/` in inner loops** unless `-use_fast_math` is on. Default `nvcc` emits `div.rn.f32` at **243 cy**. Use `__fdividef` or `*reciprocal` patterns.
5. **Never use FP64 division** anywhere hot — `div.rn.f64` is 4939 cy on B300.
6. **MUFU-heavy code is MUFU-bound**, not FMA-bound. Optimize by reducing MUFU call count (polynomial approximation, table lookup, fused exp+normalize), not by FMA tweaks.
7. **`__sincosf(x, &s, &c)` saves NO time vs separate `sinf`/`cosf`** — emits 2 separate MUFU ops on B300 (no HW sincos fusion).
8. **For deep MUFU pipelines (24 cy latency, 3-deep), need ≥ 3 independent chains/warp** to saturate. With only 1 chain (the typical softmax pattern), MUFU runtime equals latency × call count.

---

## 9. Confidence summary

| Claim | Confidence | Verification |
|---|---|---|
| MUFU is `pipe_xu`, separate from FFMA | HIGH | ncu pipe utilization metrics |
| `MUFU.EX2` is the cheapest at 14 cy | HIGH | clean SASS chain `bench_mufu_lat_clean` |
| MUFU steady-state ≈ 4.8 TGOps/s chip | HIGH | catalog 12440 + co-issue tests + consolidated math throughput |
| `ex2.approx` is 1.7× faster than other MUFUs | HIGH | catalog 12447, `bench_mufu_audit` |
| `1/x` 492 Gops/s = `div.rn.f32` not pipe-bound | HIGH | matches 243 cy in catalog 7361, latency-bound math |
| `__frsqrt_rn` 322 Gops/s = self-dep latency, NOT slow MUFU | HIGH | clean throughput sweeps show 727 inst/ns |
| `__expf`/`expf`, `__logf`/`logf` produce identical SASS | HIGH | catalog 16779, 16782 |
| `exp2f` is 17% cheaper than `expf` | HIGH | catalog 16777-16778 |
| MUFU + FFMA co-issue: free at low density, blocks at high | MED | two distinct measurements both reproducible |
| Per-SM throughput rates (table § 7) | MED | `bench_mufu_all`, single source |

---

## 10. What changed from earlier docs

- **"94× slower rsqrt" key finding RETIRED**: The 322 Gops/s figure is a 4-chain self-dep latency artifact, not a pipe characteristic. Real `__frsqrt_rn` throughput is ~3,300 Gops/s per the per-SM table.
- **"62× slower div" SOFTENED**: True for `div.rn.f32` (the default without fast-math). With `-use_fast_math` or `__fdividef`, `1/x` runs at ~660 Gops/s = MUFU rate (~5× FMA, same regime as sqrt).
- **"44× slower sqrt" SOFTENED**: True for `sqrt.rn.f32`. With fast-math, `sqrtf` ≈ MUFU.RSQ rate (~13 cy).
- **"sin/cos/exp/log roughly slower than FMA by 5-10×"**: this is HIGH confidence and matches the MUFU pipe ceiling (4.8 TGOps/s ≈ FMA / 6.3).
- **"MUFU is free alongside FMA"**: only at low density; under FFMA2 pressure with even 1:8 MUFU mix, FP32 throughput drops 60%.

---

## 11. Open / unresolved

- **Exact MUFU.RSQ latency under refinement vs without**: the math_throughput rsqrt path emits `MUFU.RSQ + ~7 FMAs` for IEEE-rn polish. SASS-verify needed.
- **MUFU-FMA co-issue cost as function of MUFU density**: catalog has data at 1:8 (60% drop) and 0:8 (full FMA), but not the smooth curve. Needed for softmax kernel design.
- **`tanhf` vs `tanh.approx.f32` cost gap**: 16.9× slowdown vs FMA in the latency-bound test, but pipe rate suggests ~3-4× under saturation. Needs ILP sweep.
- **Per-SMSP MUFU issue rate**: catalog says "0.5/SM/cy" = "1 per 2 cy per SMSP" but doesn't isolate the 4-SMSP behavior under conflict.
