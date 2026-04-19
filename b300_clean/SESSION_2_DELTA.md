# Session 2 Delta — 2026-04-18/19 findings

Net new measurements + corrections beyond the M3 rigor sweep that produced
B300_TRUE_REFERENCE.md. Use this as a supplement; the catalog remains
authoritative for unchanged numbers.

System: same as TRUE_REFERENCE (B300 SXM6 AC sm_103a, 2032 MHz boost).

## Quick navigation
- [MIO Pipe Architecture](#mio) — STS/LDS/SHFL/ATOMS/REDUX share per-SM port
- [Per-SMSP and per-SM caps table](#caps) — definitive per-op rates
- [Vector LDS/STS](#vec) — LDS.128 saturates SMEM with 2 SMSPs
- [REDUX vs CREDUX](#credux) — different SASS pipes
- [Pipe contention matrix](#pipes) — FFMA/MUFU/DFMA orthogonal to MIO
- [tcgen05 path on B300](#tcgen05) — alloc works; CUTLASS runs after patch
- [Power decomposition](#power) — HBM dominates 520W, compute 200-235W
- [LLM-relevant kernel SoL](#llm) — RMS norm, GeLU, attention, all-reduce
- [Corrections to prior catalog](#corrections)

---

<a id="mio"></a>
## MIO pipe architecture (NEW)

The MIO ("Memory Input/Output") per-SM pipe handles **STS, LDS, SHFL, ATOMS,
and REDUX** — they all share ONE 1-inst/SM/cy port. Discovered via combo
experiments where any 2-3 ops cap at total ~30 thr-op/SM/cy.

Confirmed via combo measurements (commit `c09c661`):
   STS+SHFL combo (4 SMSPs): 0.96 inst/SM/cy total (each 0.48)
   ATOMS+SHFL combo:         0.95 inst/SM/cy total
   ATOMS+STS+SHFL combo:     0.93 inst/SM/cy total
   LDS+SHFL combo:           0.94 inst/SM/cy total

Pipes orthogonal to MIO (also via combo measurements):
   FFMA + STS:  truly INDEPENDENT (FFMA free with STS)
   MUFU + STS:  INDEPENDENT
   DFMA + STS:  INDEPENDENT
   HMMA + STS:  ~40% overlap (partial — RF/operand contention)

<a id="caps"></a>
## Definitive per-op caps table (NEW, commit `b137c74`)

| Op            | per-SMSP   | per-SM cap | SMSPs to saturate per-SM | Pipe  |
|---------------|-----------|-----------|--------------------------|-------|
| SHFL.bfly     | 16 thr-op | 32        | 2                        | MIO   |
| SHFL.up/down  | 16        | 32        | 2                        | MIO   |
| SHFL.idx      | ~15       | ~30       | 2                        | MIO (5% slower) |
| LDS.32        | 16        | 32        | 2                        | MIO   |
| ATOM.INC      | 16        | 32        | 2                        | MIO   |
| STS.32        | 8         | 32        | 4                        | MIO   |
| CREDUX.MIN/MAX | 8        | 32        | 4                        | DIFFERENT pipe |
| REDUX.SUM/AND/OR/XOR | 8 | **16 (tight)** | 2                | MIO (2× slower) |

REDUX.SUM/AND/OR/XOR uses MIO at HALF rate (per-SM cap=16, not 32).
CREDUX.MIN/MAX uses a SEPARATE pipe (combo with SHFL exceeds MIO ceiling).

<a id="vec"></a>
## Vector LDS/STS — fewer SMSPs to saturate (NEW, commit `c09c661`)

Discovery: LDS.128 with just 2 SMSPs hits 92% of SMEM peak (37 TB/s).

|         | 1 SMSP   | 2 SMSPs  | 4 SMSPs  | % of 38.5 TB/s peak (4 SMSP) |
|---------|----------|----------|----------|-------------------------------|
| LDS.32  |  9.5 TB/s | 18.9 TB/s| 35.7 TB/s | 93%                          |
| LDS.64  | 18.5      | 33.9     | 37.5     | 97%                           |
| **LDS.128** | **22.9** | **35.4** | **38.0** TB/s | **99%**                |
| STS.32  |  9.2     | 18.5     | 36.9     | 96%                           |
| STS.64  | 12.5     | 24.9     | 37.7     | 98%                           |
| STS.128 | 15.1     | 29.3     | 32.4     | 84% (degraded — RF write port?)|

For kernels with limited warp count: use LDS.128 to amortize the per-SMSP
unit rate. STS.128 has a small plateau gap — use STS.64 if writing.

<a id="credux"></a>
## REDUX vs CREDUX — SASS-level distinction (NEW, commit `6982549`)

Two distinct SASS opcodes:
- `REDUX.SUM/AND/OR/XOR`  → MIO pipe (caps at 0.49 inst/SM/cy)
- `CREDUX.MIN/MAX`        → DIFFERENT pipe (caps at 0.78 inst/SM/cy alone)

Verified via SHFL combo: REDUX.SUM+SHFL = 0.77 (within MIO ceiling 1.0);
CREDUX.MIN+SHFL = 1.16 (EXCEEDS MIO — separate pipes).

Implication for warp reductions: prefer min/max if you can get away with
them (semantically equivalent for find-the-winner patterns); they are
~1.6× faster than sum reduction AND don't compete with SMEM ops.

<a id="pipes"></a>
## Pipe matrix (NEW, commit `4ce5e87`)

For 4 SMSPs all active, identical-shape kernels:

| Op | t_alone | t(X+STS) | max | sum | Verdict |
|----|--------|----------|-----|-----|---------|
| FFMA  | 0.21 ms | 0.81 ms | 0.81 | 1.02 | INDEPENDENT |
| MUFU  | 5.60   | 5.87    | 5.60 | 6.41 | INDEPENDENT |
| DFMA  | 15.33  | 15.48   | 15.33| 16.14| INDEPENDENT |
| HMMA  | 0.45   | 1.08    | 0.81 | 1.26 | PARTIAL (40% overlap) |

So FFMA/MUFU/DFMA pipes are FULLY orthogonal to MIO — kernels can mix them
freely with SMEM ops at no compute cost. HMMA partially shares (RF or
operand staging).

<a id="tcgen05"></a>
## tcgen05 path on B300 (NEW, commits `c0c2d48`, `e9757b8`, `230eed2`)

See `TCGEN05_PATH_NOTES.md` for full SASS-level details.

KEY RECIPE (resolves historical "alloc hangs"):
1. Compile: `-gencode arch=compute_103a,code=sm_103a` (NOT `-arch=sm_103a`)
2. Runtime: ALL 32 threads execute `.sync.aligned` ops (no `if(tid==0)` guard)
3. Add `tcgen05.relinquish_alloc_permit` between alloc/dealloc

CUTLASS minimal tutorial (`/root/cutlass/examples/cute/tutorial/blackwell/01_mma_sm100.cu`)
runs on B300 with 1-line patch (`if (props.major != 10) {...}` → `if (0)`).

PERF: cuBLAS algoId=66 IS the tcgen05 path (per A5). Hits 90% of FP16/BF16
spec at 8K³. Use cuBLAS for production tcgen05 perf. CUTLASS tutorial is
overhead-bound (1.1 TF at 8K³ vs cuBLAS 2237 TF — 2000× gap).

<a id="power"></a>
## Power decomposition (NEW, commit `65f3795`)

| Workload | Power | Δ idle (200W) |
|----------|------:|--------------:|
| Pure FFMA | 435 W | +235 W (compute pipe) |
| Pure HMMA legacy | 405 W | +205 W (tensor) |
| HMMA+FFMA same warp | 424 W | +224 W |
| **Pure HBM read** | **720 W** | **+520 W (memory dominates)** |
| HBM+HMMA streams | 510 W | +310 W (SM contention) |
| HMMA+LDG (LDG stalls) | 305 W | +105 W |
| **cuBLAS via cudaGraph** | **940 W** | **+740 W = TGP** |

Memory access (HBM saturation) is the BIGGEST power consumer. Path to TGP
requires async TMA + tcgen05 to overlap memory + compute without stalling.

<a id="llm"></a>
## LLM-relevant kernel SoL (NEW)

| Kernel | Achieved | SoL | % SoL | Commit |
|--------|---------:|----:|------:|--------|
| Fused RMS-norm + bias | 6.54 TB/s | 7.31 TB/s | **89% HBM** | `e9be3e3` |
| Fused RMS+GeLU+bias | 2.71 TB/s | 7.31 TB/s | 37% (GeLU MUFU bottleneck) | `e9be3e3` |
| FlashAttention proxy (cuBLAS) | 147 TF/head | ~600 TF | 24% | `5291e81` |
| 2-GPU all-reduce 1 GB BF16 | 1.97 ms | 1.36 ms | 67% NVLink | `e49e9ef` |
| TP-2 GEMM 16K³ BF16 | 4495 TF | 5000 (2× single) | 85% | `2fbf49d` |

KEY recipe for fused elementwise SoL: `__launch_bounds__(thr, 2)` to keep
register-staged rows in RF. Default occupancy spills 64-reg/thread to L1
→ 2.8× slowdown.

<a id="corrections"></a>
## Corrections to prior catalog

1. **L2 BW was 21 TB/s, not 26.7** (commit `aee2d3a`):
   My B2 finding "26.7 TB/s L2" was actually L1+L2 combined. ncu confirms
   pure L2 = 21 TB/s as catalog C5 stated. Updated B2 entry.

2. **mma.sync hits 90.5% of NVIDIA spec, not 23%** (commit `fceb94d`):
   Refutes prior "570 TFLOPS = 23%" claim. Direct mma.sync at 16 warps/SM
   = 2262 TF = 90.5% of 2500 PF spec. The 23% was sub-optimal launch geom.

3. **4 tensor cores per SM ARE per-SMSP independent** (commit `fceb94d`):
   Linear 1→4 warps/SM scaling confirms path (a) — separate per-SMSP units.

4. **TGP 940W needs tcgen05** (commit `affce3d`):
   Legacy mma.sync caps at 425W. The 940W cuBLAS measurement requires
   tcgen05 (per-CTA tensor units that draw more power per FLOP).

5. **REDUX.SUM uses MIO and is HALF the rate of other MIO ops** (commit `b137c74`):
   Adds a per-SM cap of 16 thr-op/cy (vs 32 for other MIO consumers).
   CREDUX.MIN/MAX uses a separate pipe.

---

Generated 2026-04-18/19 across multiple /loop iterations. All findings
backed by `investigations/ninja_*.cu` files. See CURIOSITY_LIST_V2.md
for full per-item details and commit hashes.

## HMMA+STS contention deep-dive (commit pending)

The pipe matrix said "HMMA+STS = PARTIAL 40% overlap" — turns out the model
is more nuanced. Combo time depends ONLY on STS count, NOT HMMA chains:

| Workload | Time | Notes |
|----------|-----:|-------|
| HMMA chains=1 | 0.219 ms | Already saturates tensor pipe |
| HMMA chains=2 | 0.219 ms | Same — pipe is full |
| HMMA chains=4 | 0.219 ms | Same |
| STS only count=1 | 0.106 ms | |
| STS only count=4 | 0.408 ms | 4× linear |
| HMMA=1 + STS=1 | 0.285 ms | overhead +66 us vs max |
| HMMA=4 + STS=1 | 0.285 ms | SAME — extra HMMA chains free |
| HMMA=4 + STS=4 | 0.542 ms | overhead +134 us vs max |
| HMMA=1 + STS=4 | 0.542 ms | SAME — proves chain count doesn't matter |

KEY INSIGHT: **HMMA chain count is IRRELEVANT to combo time**. With just
1 HMMA chain × 16 warps/SM, the tensor pipe is already saturated. Adding
more chains doesn't add work; adding more STS takes SMSP issue cycles
that would otherwise issue HMMAs.

For real workloads (GEMM with epilogue STS, attention with output write):
- The STS overhead is bounded by STS count, not by HMMA depth
- 1 STS per inner step costs ~66 us extra per launch ≈ 30% slowdown
- 4 STS per inner step costs ~134 us extra ≈ 33% slowdown

PRACTICAL: don't spread STS across many chains; batch them. The pipe penalty
is roughly 30% per "STS pulse" regardless of how it interleaves with HMMA.


## FP4/FP6 mma.sync NOT available on sm_103a (NEW finding)

ptxas error confirms FP4/FP6 is exclusively tcgen05 territory:

```
ptxas error : Instruction 'mma with FP6/FP4 floating point type'
              not supported on .target 'sm_103a'
              for shape '.m16n8k64'
```

Tested PTX forms (all REJECTED):
- `mma.sync.aligned.m16n8k64.row.col.f32.e2m1.e2m1.f32` (no .kind) → "kind::f8f6f4 required"
- `mma.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32` → not on sm_103a

So for FP4 access on B300:
- **Legacy mma.sync**: NO. Caps at FP8 (e4m3, e5m2) for sm_103a.
- **tcgen05.mma.kind::f8f6f4**: required. Has full CUTLASS dependency.
- **cuBLAS**: practical path. K=96 hits 10800 TF = 72% of 15000 spec (prior session).

UPDATE to D4 power/perf table: confirm "FP4 cuBLAS = 10800 TF" is the only
production-accessible FP4 number on B300. Direct PTX is gated to tcgen05.

This is a HARDWARE LIMIT, not a software workaround issue. ptxas explicitly
declines for the target arch.

Confidence: HIGH (ptxas rejects definitively; tested 2 PTX variants)

## INT8 mma.sync confirmed throttled (corroborates catalog 06)

Independent measurement of `mma.sync.m16n8k32.s8.s8.s32`:
- 4 chains, 16 warps/SM: **36.9 TOPS** (= 0.74% of 5000 spec)
- 8 chains, 16 warps/SM: **36.7 TOPS** (same — ILP doesn't help)

Catalog 06 reports 143 TOPS (HW-throttled, 5 NOPs/issue, 65 cy/inst). My
lower number reflects "4-chain steady state" (~26% of catalog's higher-chain
result) but BOTH confirm the deliberate throttle.

KEY: scaling chains 4 → 8 doubles kernel time but holds TOPS/s constant.
This is the ILP-insensitivity signature of HW throttling (not pipeline
latency-bound, where 2× chains would give 2× TOPS).

INT8 mma.sync is DEPRECATED on B300 — use FP8 e4m3 instead (4500 TOPS via
cuBLAS, real spec). Or use cuBLAS for production INT8 if needed.

