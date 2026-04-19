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


## FP8 mma.sync measurement is UNRELIABLE — SASS ambiguity (NEW)

Tried multiple variants of `mma.sync.m16n8k32.f32.e4m3.e4m3.f32`:
- "Naive" (zero-init c): SASS shows only 16-22 HMMAs total (DCE)
- Non-constant a/b inputs: still 22 HMMAs (compiler still folds)
- Reports 7500-8200 TFLOPS = 150-165% of 5000 spec → IMPOSSIBLE

KEY DIAGNOSIS via SASS inspection:
   `HMMA.16816.F32 R28, R28, R4, RZ` — this is m=16,n=8,k=**16** (BF16 form)
   NOT the m=16,n=8,k=32 FP8 form I requested in PTX!

ptxas appears to silently map `mma.sync.m16n8k32.e4m3` to the same SASS
opcode as BF16 m16n8k16 (same byte-rate per row: 32B). Whether the hardware
internally interprets as FP8 or BF16 is unclear from SASS alone.

CONCLUSION: legacy mma.sync FP8 path is fundamentally hard to measure
reliably from PTX. The compiler folds aggressively AND the SASS encoding
overlaps with BF16. Cannot trust direct-PTX FP8 numbers.

PRACTICAL: For FP8 perf on B300, use cuBLAS exclusively (algoId=66 family,
hits 4500 TF = 90% of 5000 spec per prior session). Direct PTX FP8 is a
debugging quagmire.

This is consistent with the FP4 finding: high-density precisions (FP8/FP4)
are gated to tcgen05 / cuBLAS paths on B300; legacy mma.sync is for
BF16/FP16/TF32.

Confidence: HIGH for "SASS shows HMMA.16816 not 16832" diagnosis
            HIGH for "use cuBLAS, not direct PTX" recommendation


## FP8 mma.sync UPDATE: time-scales linearly, but counting is ambiguous

Time-scaling check refutes the DCE hypothesis:
   N=200: 0.132 ms = 7528 TFLOPS (claimed)
   N=400: 0.258 ms = 7700 TFLOPS (claimed)

Time scales 1.95× for 2× work — so the kernel IS doing real work.

But the SASS encoding (HMMA.16816.F32 — same family as BF16) suggests
the compiler may be treating FP8 mma.sync as BF16-equivalent at byte level.

If that's true, ops_per_inst should be 4096 (BF16 rate) not 8192 (FP8 rate).
Recomputed: 7528 / 2 = **3764 TFLOPS = 75% of 5000 spec** — plausible.

Alternative interpretations:
1. FP8 truly runs at 1.5× spec on B300 (unlikely but possible — specs conservative)
2. SASS encoding shared but HW dispatches based on input tag (exotic)
3. Compiler folded 50% of work + counted wrong (unlikely given clean N-scaling)

UNRESOLVED. Most likely: legacy mma.sync with .e4m3 uses the same SASS path
as BF16 with the same byte rate, producing FP8-quality output. ops_per_inst
should be the BF16 rate (4096), giving ~3760 TF = 75% of spec.

For PRODUCTION FP8 perf, cuBLAS algoId=66 (tcgen05) is authoritative at
4500 TF = 90% spec. Direct PTX FP8 is in a gray zone.

## 🚨 MAJOR RETRACTION (2026-04-19): S1 was DCE'd. mma.sync ≈ 23% spec, NOT 90%

**S1's "mma.sync hits 90.5% of BF16 spec" claim is RETRACTED.**

Side-by-side test (`investigations/ninja_bf16_DCE_correction.cu`):

   V1 S1-style (constants, impossible-cond write):  0.219 ms = **2270 TF (90.8% spec)**
   V2 STRICT anti-DCE (thread-derived + always-write): **0.858 ms = 578 TF (23.1% spec)**

   SASS HMMA counts: V1 = 320, V2 = 1280  →  V1 was running 1/4 the work!

The compiler aliased c0/c1/c2/c3 chains in V1 because:
- All chains had identical constant inputs (0x3f80000* pattern)
- Output guard was `if (c0[0] == 0xDEADBEEF && N < 0)` — impossible, so c1/c2/c3
  outputs were never observably needed → DCE'd those chains
- Effective work = 1 chain × 4 (claimed) → 4× over-reported throughput

**THE ORIGINAL "23% of NVIDIA spec" CLAIM WAS CORRECT.** Legacy mma.sync on
B300 caps at ~580 TF = 23% of 2500 BF16 spec, EXACTLY as the Tier S1 prompt
in CURIOSITY_LIST stated before my "refutation".

This RE-CONFIRMS:
- cuBLAS algoId=66 (tcgen05 path) at 2237 TF = 90% spec was the only path
  to nominal throughput
- Legacy mma.sync is ~4× slower than tcgen05 — explaining the power gap (S3)
- For BF16 inference perf: MUST use cuBLAS, not raw mma.sync

TF32 RE-VERIFIED: strict anti-DCE kernel still gives 289 TF = 100% of catalog.
TF32 is NOT affected by this DCE issue (constants probably differently
interpreted by compiler for TF32 vs BF16 packing).

LESSON: the rigor protocol's "ALWAYS SASS-verify the kernel" rule (#6) is
LOAD-BEARING. SASS instruction count is the only definitive signal of how
much work the compiler kept.

CASCADING UPDATES NEEDED:
- S1 entry in CURIOSITY_LIST_V2: revert "RESOLVED" to "REFUTED — original
  23% claim STANDS"
- S4 ("4 tensor cores per SM") may also need re-verification
- D4_PRECISION_POWER_PERF_TABLE.md: BF16 mma.sync 90.5% → 23% (cuBLAS-only
  for BF16 SoL)
- Investigation kernels using S1-style impossible-condition guards may have
  similar DCE issues — audit needed

Commit: `14bf71f`. Confidence: HIGH (SASS counts + strict-DCE
test give convergent evidence for the retraction).

## S3 power CASCADING UPDATE: HMMA actual = 255W (was 405W DCE'd)

Re-tested HMMA sustained power with strict anti-DCE kernel (8 warps/SM, real
work confirmed via S4 strict re-test = 580 TF actual).

Sustained NVML power: **255 W** (vs prior S3 claim of 405W with DCE'd kernel)

Updated power decomposition:
   Idle:                              200 W
   Pure FFMA (8 chains):              435 W (+235 W) [prior, FFMA chains likely OK]
   **Strict HMMA 8 warps/SM (580 TF)**: **255 W (+55 W ΔTENSOR)** ← corrected
   Pure HBM read (720 W):              720 W (+520 W ΔMEM) [unaffected]
   cuBLAS via cudaGraph (940 W):       940 W (+740 W = TGP) [unaffected]

Implications:
- HBM dominates compute even MORE than previously thought (520W vs 55W ΔTENSOR)
- Compute pipes (FP32 ALU, tensor) draw modest power; memory + tcgen05 the killers
- The "+205W tensor" entry in D4 power table needs revision to "+55W"

The cuBLAS 940W remains the path to TGP, and is even MORE clearly memory-+-
tcgen05-bound. Legacy mma.sync = small power footprint AND small compute.

Confidence: HIGH (strict anti-DCE confirmed; power sampled 200ms over 5s+ run)

## DCE audit — which prior measurements survive?

After S1 retraction, audited each chain-based measurement for DCE susceptibility:

|   Op  | Original output guard | Strict re-test | Verdict |
|-------|-----------------------|----------------|---------|
| BF16 mma.sync | `c0[0]==0xDEAD && N<0` | 578 TF (was 2270) | ❌ DCE'd 4× |
| TF32 mma.sync | similar single-output | 289 TF (was 289) | ✓ OK |
| FFMA scalar 8-chain | `(a0+...+a7)==0 && N<0` | 70 TF (was 70) | ✓ OK |
| HMMA power | (impossible-cond) | 255W (was 405W) | ❌ DCE'd 1.6× |
| INT8 mma.sync | always-write | 37 TOPS | ✓ OK |
| HBM read/write | always-write XOR | 720W | ✓ OK |
| cuBLAS perf | external library | 2237 TF | ✓ OK |

**Pattern**: if the output guard only references ONE chain's first element
(like `c0[0]`), compiler aliases other chains. When the guard sums ALL
chains' outputs (FFMA's `(a0+...+a7)==0`), all chains stay alive.

DCE-affected (need correction):
- BF16 mma.sync: 23% spec (not 90%)
- HMMA power: 255W (not 405W)
- Pipe matrix HMMA+STS combo: re-verify needed (used same pattern)
- A1 FlashAttention (uses cuBLAS): unaffected

DCE-safe:
- TF32 mma.sync: 289 TF (catalog match holds)
- FFMA: 70 TF = 91% spec
- All HBM measurements (always-write XOR pattern)
- All cuBLAS measurements (external)
- INT8 mma.sync (always-write)

LESSON: prefer `out[idx] = (sum of all accumulators)` over `if (cond) out[idx] = c0[0]`
for anti-DCE. The latter only protects c0; compiler may alias c1, c2, ... .


## Reconciliation with catalog 06 — both are right (different reference)

Discovered after the retraction: **catalog 06_tensor_cores.md already has
the correct mma.sync number** — 577 TF at 2032 MHz boost. Matches my new
strict-DCE measurement of 578 TF EXACTLY.

But catalog frames it as **93.7% of 616 TF LEGACY theoretical**, while
CURIOSITY S1 framed it as **23% of 2500 TF tcgen05 spec**. Both are right!

   577 TF measured ÷ 616 LEGACY mma.sync theoretical = 93.7% (catalog)
   577 TF measured ÷ 2500 tcgen05 NVIDIA spec       = 23.1% (CURIOSITY)

The 4× ratio (2500 vs 616) is the **hardware-architectural gap** between
tcgen05 and legacy mma.sync pipes on B300. Both pipes saturate well at
their respective ceilings.

So the corrected picture:
- legacy mma.sync pipe: caps at ~616 TF theoretical, achieves 93.7%
- tcgen05 pipe: caps at 2500 TF theoretical, achieves 89-90% via cuBLAS
- 4× hardware gap between them is THE architectural rationale for tcgen05

My S1 "RESOLVED" finding was wrong because it claimed mma.sync hits
2500-spec (which is tcgen05's spec). Catalog 06 was right all along —
mma.sync hits its OWN spec (616) just fine.

PRACTICAL takeaway for D4 / production:
- BF16 inference on B300: USE cuBLAS (algoId=66 = tcgen05) for full 2500 TF
- Direct mma.sync legacy is fine for custom kernels but caps at 616 TF
- Don't mistake the two pipes' specs

LESSON: when "refuting" a catalog number, check what reference the catalog
is using BEFORE concluding it's wrong. The 23% claim and the 93.7% claim
are the same physical measurement, just framed against different pipes.

## Pipe matrix HMMA findings ALSO retracted (2026-04-19)

Re-tested with strict anti-DCE. Both HMMA chain-count AND HMMA+STS combos
were affected by the same DCE issue.

### HMMA chain scaling — strict anti-DCE

|chains| time   | TFLOPS | % LEGACY |
|------|-------:|-------:|---------:|
| 1    | 0.226 ms | 550 TF | 89.4% |
| 2    | 0.444   | 559   | 90.7% |
| 3    | 0.645   | 577   | 93.7% |
| 4    | 0.858   | 579   | 94.0% |

CORRECT picture: HMMA chains DO add work and time (nearly linear scaling).
Per-instruction throughput stays at 89-94% of legacy regardless of chain
count. Prior "1 chain ≡ 4 chains in time" was DCE artifact.

### HMMA+STS combos — strict anti-DCE

|H | S | time   | max(H,S) | overhead |
|--|---|-------:|---------:|---------:|
|1 | 0 | 0.225 ms |     —    |     —     |
|4 | 0 | 0.858   |     —    |     —     |
|0 | 1 | 0.106   |     —    |     —     |
|0 | 4 | 0.408   |     —    |     —     |
|1 | 1 | 0.254   | 0.225    | +13%      |
|4 | 1 | 0.885   | 0.858    | +3% (STS nearly FREE!) |
|1 | 4 | 0.544   | 0.408    | +33%      |
|4 | 4 | 0.968   | 0.858    | +13%      |

CORRECT picture: HMMA and STS pipes are MOSTLY INDEPENDENT (3-33% overlap
overhead, depending on which dominates). Adding 1 STS to a 4-chain HMMA
loop costs only +3% time — STS truly piggybacks on HMMA's idle cycles.

This is OPPOSITE of prior "40% partial overlap" claim and the bizarre
"chain count doesn't matter" finding. Both were DCE artifacts.

REAL guidance for kernels mixing HMMA + SMEM stores:
- 1 STS per HMMA-pulse: nearly free (<13% overhead)
- 4 STS per HMMA-pulse: noticeable (+33% if HMMA short)
- HMMA chains add proportional work — there's no "free" extra chain


## MIO unification VERIFIED stands (strict anti-DCE)

Re-tested STS+SHFL combo with always-write:
   STS only:    0.207 ms
   SHFL only:   0.209 ms
   STS+SHFL:    0.423 ms ≈ STS+SHFL summed (0.416 ms)

Combo time = approximate SUM of alones → STS and SHFL run SEQUENTIALLY
through the shared MIO port. They cannot overlap.

This CONFIRMS the MIO unification finding from prior iteration. The STS
volatile write kept the SHFL chain alive (chain-output-aliased DCE issue
didn't apply here because vsmem[] writes use v).

Practical: kernels mixing SMEM ops compete for MIO. Don't expect parallelism
between STS/LDS/SHFL/ATOMS — they take turns on the per-SM port.

So the MIO architecture (STS/LDS/SHFL/ATOMS share per-SM port) is REAL
and the original measurements from commit `c09c661` STAND.

Confidence: HIGH (strict re-test matches prior measurement to <2%)

## ncu confirms mma.sync ceiling is HARDWARE (not measurement)

For the corrected strict-DCE BF16 mma.sync kernel (4 chains, 16 warps/SM):

   sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active: **99.22%**
   smsp__inst_executed_pipe_tensor.sum: 30.3M (matches kernel work)
   gpc__cycles_elapsed.max: 419,703 cycles (= 207us @ 2.032 GHz, matches wall-clock)

The legacy tensor pipe is 99.22% ACTIVE but completes 0.94 mma per active cycle.
Each HMMA takes ~1.06 cycles to retire → architecturally capped, not bug.

This makes the corrected mma.sync = 580 TF (94% legacy, 23% tcgen05-spec)
finding HIGH-confidence:
- Wall-clock measurement (0.207 ms)
- ncu pipe_tensor 99.22% active
- ncu inst count matches kernel expectation
- SASS HMMA count (1280) matches expected work

Three convergent sources confirm the architectural ceiling.

The 4× gap to tcgen05 is REAL hardware, not measurement methodology.
B300's tcgen05 pipe is ~4× faster per cycle than legacy mma.sync.
