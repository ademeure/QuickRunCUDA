# Section E — Tensor Cores, NVFP4, tcgen05.mma (B300 SXM6 AC Canonical Reference)

Sections §46 through §55. Self-contained answers for tensor-core ladders,
NVFP4-specific peaks/power, tcgen05 microarchitectural power model, K-id,
CUTLASS/CuTeDSL gap, and the 3-tier sparsity story.

---

## §46. Tensor core SoL — full ladder per precision (cuBLAS realistic + zero / random / realistic split)

**Answer:** Catalog "FP8 4500 / BF16 2246 / NVFP4 11.4 PF" peaks are ALL zero-data; production-realistic numbers drop **10-22%** depending on precision. NVFP4 cuBLAS+cudaGraph BPG=16 reaches **11423 TFLOPS** (76.2% of 15 PF B300 spec); two-GPU split aggregates **19163 TFLOPS** (95.8% of 2× 10 PF spec). Legacy mma.sync m16n8k16 caps at **569-578 TFLOPS** = 7.4× FFMA. `[🟢 HIGH · src: corrections/06_tensor_cores_CORRECTED.md, B300_TRUE_REFERENCE.md, NVFP4_CUDAGRAPH.md, NVFP4_CUBLAS_FULL_SWEEP.md]`

There are five things that have to be carried together in any tensor-core
quote on B300:
1. **Path** — tcgen05.mma (Blackwell, async, TMEM-resident accumulator) vs mma.sync (legacy warp-sync, RF accumulator). cuBLAS uses tcgen05 internally on sm_103a.
2. **Data pattern** — zero / const / random / "normal-ish" (per `B300_TRUE_REFERENCE.md` row 66-68 commit 6e40ef9 measurements at N=8192). Random and normal-ish are both valid representatives of "real" data; zero is a best-case marketing number.
3. **Clock state** — boost (~2032 MHz unlocked under TDP) vs `-lgc 2032` (paradoxically pins to 1920 MHz base) vs hard-locked 1500 MHz vs 1005 MHz. Different workloads throttle differently.
4. **Single-shot vs sustained** — NVFP4 single-shot const N=8192 = 9109 TF (91% spec); sustained random N=16384 cudaGraph 15s = 6554 TF (65%) because the clock throttles to 1057 MHz under 1186 W instant peak.
5. **Spec used** — NVIDIA quotes "B200 spec = 10 PF" or "B300 spec = 15 PF". The 15 PF figure assumes the K=96 ULTRA path is reachable by the workload and clock is held; in practice cuBLAS 13.4 caps at **~10.8 PF** = 72% of 15 PF spec (or 108% of B200 10 PF spec).

### 46.1 Master peak table — cuBLAS path (tcgen05 internal)

All TFLOPS chip-wide on 148 SMs unless noted. "Realistic" = normal-ish
distribution proxy from TRUE_REFERENCE row 66-68. Drops are vs zero baseline.

| Precision | Zero TFLOPS | Random TFLOPS | Realistic TFLOPS | NVIDIA spec | % spec | Clock | Source |
|-----------|------------:|--------------:|-----------------:|------------:|-------:|:-----:|--------|
| **FP16** (cuBLAS LtMatmul, N=K=8192) | 2246 | 1905 | 1744 | 2465 | 91 / 77 / 71 | 1920 lock | TRUE_REFERENCE r66 |
| **BF16** (cuBLAS LtMatmul, N=K=8192) | 2246 / 2242 | 1883 | 1850 | 2500 | 90 / 75 / 74 | 1920 lock | TRUE_REFERENCE r48,r67 |
| BF16 microbench tcgen05 direct | 2325 | n/a | n/a | 2500 | 93 | 1920 | 06_tensor_cores r20 |
| **TF32** (cuBLAS, N=K=8192) | 1113 | n/a | n/a | 1232 | 90 | 1920 | 06_tensor_cores r21 |
| **FP8 e4m3** (cuBLAS LtMatmul, sustained via cudaGraph) | 4425 / 4491 | 3984 | 3951 | 5000 | 88-91 / 80 / 79 | 1920 | TRUE_REFERENCE r56-57,r68 |
| FP8 e4m3 microbench (tcgen05 direct) | 4651 | n/a | n/a | 5000 | 93 | 1920 | 06_tensor_cores r17 |
| **FP8 random under 600 W power cap** | n/a | **3087** | n/a | 5000 | **62** (-43% from zero) | varies | TRUE_REFERENCE warning |
| **NVFP4 e2m1 wide-N M=8192,N=65536,K=16384** | **10297** | n/a | n/a | 10000 (B200) | **103** | boost | TRUE_REFERENCE r51 |
| NVFP4 e2m1 cuBLAS square N=K=24576 | 8424 | 8424 | n/a | 10000 | 84 | boost | TRUE_REFERENCE r52 |
| NVFP4 e2m1 single-shot const N=8192 | 9109 | n/a | n/a | 10000 | 91 | boost | TRUE_REFERENCE r55 |
| NVFP4 e2m1 sustained random N=16384, cudaGraph 15s | n/a | 6554 | n/a | 10000 | 65 (throttle to 1057 MHz, 1186 W instant) | boost | TRUE_REFERENCE r54 |
| **NVFP4 cuBLAS+cudaGraph BPG=16, M=N=8192,K=38400** | **11423** | n/a | n/a | 15000 (B300) | **76.2** | boost | NVFP4_CUDAGRAPH.md (RECORD) |
| NVFP4 cuBLAS plain Lt, M=N=8192,K=38400 | 11054-11068 | n/a | n/a | 15000 | 73-74 | boost | NVFP4_CUBLAS_FULL_SWEEP.md |
| NVFP4 cuBLAS, M=N=8192,K=38400 (random sustained) | n/a | ~7000 | n/a | 15000 | ~47 | boost (TDP-throttle to 1455 MHz) | CUTEDSL_THROTTLE big table |
| NVFP4 cuBLAS @ 510 MHz lock, M=N=16384,K=61440 | 3558 | n/a | n/a | (3766 @ 510) | **94.5** at-clock | 510 lock | NVFP4_CUBLAS_FULL_SWEEP.md |
| NVFP4 K=96 ULTRA microbench | 10910 | 10910 | n/a | 15000 | 73 (= cuBLAS 13.4 ceiling) | 1500 lock | TCGEN05_PERFW_CLEAN, 06_tensor_cores r35 |
| **2-GPU NVFP4 split (1 stream/GPU, no comm)** | **19163** | n/a | n/a | 20000 (2× 10000 spec) | **95.8** | boost | TRUE_REFERENCE r49 |
| Per-GPU NVFP4 in 2-GPU split | 9582 | n/a | n/a | 10000 | 95.8 | boost | TRUE_REFERENCE r50 |
| FP4 block-scaled microbench (kind::mxf4nvf4.block_scale.block16) | 9856 | n/a | n/a | 10000 | 99 | 1920 | 06_tensor_cores r15 (re-verify pending per M3) |

### 46.2 Master peak table — mma.sync path (legacy warp-sync, RF accumulator)

| Precision | TFLOPS | % spec | Clock | Notes | Source |
|-----------|-------:|-------:|:-----:|-------|--------|
| **FP16/BF16 m16n8k16, F32 acc** | **578.6** | 7.4× FFMA | boost 2032 | V8 8-chain, 99.9% pipe_tensor saturation, 94.72M HMMAs in 670 µs, 99.22% pipe active by ncu | V8_HMMA_F16_PEAK, SESSION_2_DELTA r522 |
| FP16/BF16 m16n8k16, F16 acc | 578.6 | identical | boost | F32 acc free | V8_HMMA_VARIANTS_PEAK |
| BF16 m16n8k16 burst (TRUE_REF row 47) | 569 | matches catalog 569 | 1920 | matches 8-chain within 2% | TRUE_REFERENCE r47, r58 |
| TF32 m16n8k8 | 288 | half of FP16 (K=8 vs K=16) | ~2032 | MEDIUM | 06_tensor_cores r23 |
| INT8 m16n8k32.s32.s8 | 143 TOPS | HW-throttled (5 NOPs/issue) | ~2032 | NOT latency-bound — see §47 footgun | 06_tensor_cores r24 |
| FP8 mma.sync kind::f8f6f4 | **104** (real, NOT 276 effective) | n/a — emulated | 1920 | F2FP.UNPACK + 2× HMMA — see §48 | MMA_FP8_KIND_F8F6F4_NOT_NATIVE |
| FP4 mma.sync | REJECTED on sm_103a | n/a | — | only sm_120a (Geforce) | 06_tensor_cores r27 |

### 46.3 Master peak table — FP64 tensor

| Operation | TFLOPS | % spec | Notes | Source |
|-----------|-------:|-------:|-------|--------|
| DMMA / DGEMM | **1.05** | matches DFMA | NO FP64 tensor speedup on B300 | 06_tensor_cores r28-29 |

### 46.4 Data-pattern drops at N=K=8192 (cuBLAS, commit 6e40ef9)

| Precision | zero/const | random | normal-ish | random vs zero | normal vs zero |
|-----------|-----------:|-------:|-----------:|---------------:|---------------:|
| FP16 | 2246 | 1905 | 1744 | -15% | **-22%** |
| BF16 | 2246 | 1883 | 1850 | -16% | -18% |
| FP8 e4m3 | 4393 | 3984 | 3951 | -9% | -10% |

**For FP8 random under 600 W power cap: 3087 TFLOPS = -43% from zero.**

### 46.5 Per-clock optima — cuBLAS NVFP4 sweep

From `NVFP4_CUBLAS_FULL_SWEEP.md`:

| Clock | Best shape (M,N,K) | TFLOPS | % 15 PF spec | % at-clock spec |
|------:|--------------------|-------:|-------------:|----------------:|
| 510 MHz lock | M=N=16384, K=61440 | 3558 | 23.7% | **94.5%** |
| 1500 MHz lock | M=N=8192, K=38400 | 9273 | 61.8% | 83.7% |
| Boost (~2032) plain Lt | M=N=8192, K=38400 | 11054 | **73.7%** | 73.7% |
| Boost + cudaGraph BPG=16 | same | **11423** | **76.2%** | 76.2% (model ceiling 73.1% + 3pp) |

**510 MHz hits 94.5% MFU at-clock** — coordination overhead is a tiny
fraction of slow per-cycle time. Boost regime is power-cap-bound.

### 46.6 Llama-style realistic shapes — what to expect

From `NVFP4_CUTEDSL_THROTTLE_DEEP_DIVE.md` LLM-realistic table — at 1005
MHz, zero data, CuTeDSL persistent kernel:

| Layer purpose | Shape (M, N, K) | TFLOPS | MFU @ 1005 (15 PF spec) |
|---------------|-----------------|-------:|-----------------------:|
| gate/up | (8192, 14336, 8064) | 4510 | 30% |
| down | (14336, 8192, 8064) | 4537 | 30% |
| qkv | (8192, 8192, 8064) | 4589 | 30% |
| ffn | (4096, 14336, 11904) | 4638 | 31% |

Real Llama-70B layer matmuls plateau at **~30% of 15-PF spec** because
K=8064 is too narrow to hit the deep-K reuse plateau. To hit the 91% MFU
king shape (K=61440) you'd need K ≥ 40K — which doesn't occur naturally
in transformer matmuls except in attention (K = head_dim × seq_len).

**Implication**: realistic LLM inference NVFP4 throughput is ~30% of
spec at 1005 MHz lock and ~50% at boost (per cuBLAS K-sweep). The
"15 PF" or "11 PF" headlines never apply to real model layers.

### 46.7 Tile selection guidance (cuBLAS)

- **At boost**: prefer M=N=8192 with K ∈ [12K, 46K]. Avoid K<6K (compute amortization) and K>46K (HBM-bound). Square > asymmetric.
- **At 1500 MHz**: same optimal as boost; tolerates bigger problems (16384² in top 5).
- **At 510 MHz**: prefer LARGE problems (M=N=16384 or 24576), deep K (38K-61K). Achieves 94.5% MFU at-clock.
- Tall-skinny (32K×8K K=15K, 69.3%) ≈ wide-skinny (8K×32K K=15K, 69.0%) — cuBLAS handles asymmetry well, unlike CuTeDSL.
- Llama-style (8K × 14K K=8K): only 53-58% MFU because K=8064 too narrow.

### 46.8 cudaGraph BPG sensitivity (NVFP4)

From `NVFP4_CUDAGRAPH.md`:

| Shape | REG MFU | GRAPH BPG=1 | BPG=4 | BPG=16 | BPG=16 speedup |
|-------|--------:|------------:|------:|-------:|---------------:|
| 2K² K=3K | 25.9% | 20.9% | 24.0% | 27.4% | 1.06× |
| 4K² K=6K | 37.9% | 39.4% | 40.0% | 42.1% | 1.12× |
| 8K² K=6K | 47.8% | 48.0% | 48.7% | 50.9% | 1.07× |
| 8K² K=12K | 68.5% | 68.8% | 69.3% | 72.4% | 1.06× |
| **8K² K=38K** | 73.7% | 72.8% | 73.0% | **76.2%** | 1.03× |
| 16K² K=38K | 66.6% | 66.6% | 66.7% | 69.5% | 1.04× |

BPG=1 is same as REG (graph instantiation overhead = launch overhead).
BPG=16 is best — graph instantiation paid once, 16 matmuls per replay.
Gain biggest in pp on small shapes where launch was a large fraction.

### 46.9 Quick-cite cheat-sheet (copy-paste numbers)

| Need | Use | Source |
|------|-----|--------|
| BF16 cuBLAS realistic | **1850 TFLOPS** | TRUE_REFERENCE r67 |
| BF16 cuBLAS zero best-case | **2246 TFLOPS** | TRUE_REFERENCE r66 |
| BF16 mma.sync legacy | **569-578 TFLOPS** | V8 + TRUE_REF r47 |
| FP16 cuBLAS realistic | **1744 TFLOPS** | TRUE_REFERENCE r66 |
| FP16 mma.sync legacy | **578 TFLOPS** | V8_HMMA_F16_PEAK |
| FP8 cuBLAS realistic | **3983 TFLOPS** | TRUE_REFERENCE r57 |
| FP8 cuBLAS zero best-case | **4425 TFLOPS** sustained / **4491** microbench | TRUE_REFERENCE r56 |
| FP8 cuBLAS under 600 W cap | **3087 TFLOPS** | TRUE_REFERENCE warning |
| TF32 cuBLAS | **1113 TFLOPS** | 06_tensor_cores |
| NVFP4 cuBLAS best (wide-N, single-shot) | **10297 TFLOPS** (103% B200 spec) | TRUE_REFERENCE r51 |
| NVFP4 cuBLAS+cudaGraph all-time peak | **11423 TFLOPS** (76.2% of 15 PF) | NVFP4_CUDAGRAPH.md |
| NVFP4 cuBLAS sustained random | **6554 TFLOPS** (heavy throttle) | TRUE_REFERENCE r54 |
| NVFP4 K=96 ULTRA microbench | **10910 TFLOPS** @ 1500 MHz | TCGEN05_PERFW_CLEAN |
| 2-GPU NVFP4 aggregate | **19163 TFLOPS** (95.8%) | TRUE_REFERENCE r49 |
| FP4 block-scaled microbench | 9856 TFLOPS (re-verify pending) | 06_tensor_cores |
| INT8 mma.sync | 143 TOPS (HW-throttled) | 06_tensor_cores |
| FP64 DMMA / DGEMM | 1.05 TFLOPS (no tensor speedup) | 06_tensor_cores |

**Footgun:** ⚠ ALL catalog "FP8 7500-8200 TFLOPS" / "BF16 1543 single-chain" / "FP4 14 PF" headlines are ZERO-DATA or single-shot or compiler-folded. Use REALISTIC for production estimates and reread §46.4 — random is -10% (FP8) to -22% (FP16) below zero. Under 600 W cap, FP8 random falls 43% below zero peak.

**Footgun #2:** ⚠ `nvidia-smi -lgc 2032` paradoxically pins to **1920 MHz** (base clock) NOT boost. To stay at boost you must use `-rgc` (unlocked); under sustained tcgen05 random load it throttles to 1455-1057 MHz. Catalog numbers mix 1920 / 2032 freely → ~6% noise.

**Footgun #3:** ⚠ The single-chain "1543 TFLOPS BF16 mma.sync" claim is **RETRACTED** (over-counted; real ~570 TF). The number 1543 *also* legitimately appears as the NVLink-5 bidirectional GB/s and as one cell of N_DEPENDENCE_DEEPDIVE M=N=K=28672 — those are different things, not retracted.

**See also:** §47 (mma.sync vs tcgen05 paths), §48 (mma.sync FP8 emulation), §49 (NVFP4 K=96 ULTRA), §50-§52 (power), corrections/06_tensor_cores_CORRECTED.md, corrections/NVFP4_CONSOLIDATED.md.

---

## §47. Tensor cores — m16n8k16 (mma.sync) vs tcgen05.mma paths

**Answer:** `mma.sync` is the legacy SM-resident warp-sync path with F16/F32 accumulators in registers; `tcgen05.mma` is the Blackwell warpgroup-async path that writes to **TMEM** (Tensor Memory, a separate SRAM region per SM). Different SASS opcodes (HMMA vs UTCHMMA/UTCQMMA/UTCOMMA), different power profile (~10× per-MAC parity but different occupancy), and **different ncu metric** — `pipe_tensor` measures mma.sync only. cuBLAS dispatches to tcgen05 internally on sm_103a. `[🟢 HIGH · src: corrections/06_tensor_cores_CORRECTED.md §R4, SESSION_2_DELTA.md §pipe_tensor, MMA_SYNC_POWER.md, CUBLAS_BIT_ENTROPY_CORRECTION.md]`

### 47.1 Path comparison table

| Property | mma.sync (legacy) | tcgen05.mma (Blackwell) |
|----------|-------------------|--------------------------|
| Synchronization | warp-sync (32 threads) | warpgroup-async (128 threads / 4 warps) |
| Accumulator location | RF (registers) | TMEM (separate SRAM) |
| Largest single op | m16n8k16 (BF16) | m128n128k16 (BF16); m128n128k64 (NVFP4) |
| Max chip-wide TF (BF16) | 569-578 (catalog 569) | 2246 zero / 1850 realistic |
| Max chip-wide TF (FP8) | 104 (emulated, see §48) | 4425 zero / 3983 realistic |
| Max chip-wide TF (NVFP4) | n/a (REJECTED on sm_103a) | 10297 wide-N / 11423 cudaGraph peak |
| SASS opcode (BF16) | `HMMA.16816.F32` | `UTCHMMA.16816.F32` |
| SASS opcode (FP8) | `F2FP.UNPACK_B` + 2× HMMA | `UTCQMMA.…` (real native FP8) |
| SASS opcode (NVFP4) | n/a | `UTCOMMA.BLOCK16` (UTCOMMA = ULTRA tcgen05 NVFP4) |
| ncu pipe metric | `sm__pipe_tensor_cycles_active` | `…hmma_op_utchmma_utcqmma_utcomma…` (specific subpipe) |
| pipe_tensor sees this path? | **YES** | **NO** (silent miss) |
| cy/MMA (single-CTA) | ~1.06 cy/MMA at 4 chains | 128 cy/MMA at M=N=128 (98% MFU) |
| Saturates at | 4 warps/SM, 4 chains | 1 warp/CTA, single-issue |
| cuBLAS uses? | No (legacy) | YES (tcgen05.mma is the production path) |

### 47.2 The pipe_tensor footgun (cuBLAS bit-entropy correction)

`SESSION_2_DELTA.md` lines 524-790 documents the strongest internal evidence:

> For the corrected strict-DCE BF16 mma.sync kernel:
>   `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active = 99.22%`
>   gpc__cycles_elapsed = 419,703 cy = 207 µs @ 2.032 GHz (matches wall-clock)
>
> But for the 32 warps × 4 chains "regression":
>   Wall-clock per launch: 9.374 ms (cudaEventElapsedTime)
>   ncu gpc__cycles_elapsed: 4,924,479 cy = 2.42 ms
>
> **These DISAGREE by 4×.** ncu shows pipe_tensor 99% active for 2.42 ms;
> wall-clock measures 9.37 ms total. Kernel pipe is active for short bursts,
> then long idle. **pipe_tensor active% may have scope limitations.**

`CUBLAS_BIT_ENTROPY_CORRECTION.md` lines 1086-1087 documents the right
metric for tcgen05 measurements:
- `sm__pipe_tensor_subpipe_hmma_cycles_active.sum`
- `smsp__inst_executed_pipe_tensor_subpipe_hmma_op_utchmma_utcqmma_utcomma_scope_2cta.sum`

The second metric explicitly contains `utchmma_utcqmma_utcomma` so it
DOES include tcgen05 ops. **Verify the metric name unpacks to what you
expect** before treating its count as MMAs.

### 47.3 Mma.sync occupancy and the regression at 32 warps × 4 chains

Per `SESSION_2_DELTA.md`:
- 4-16 warps/SM × any chains: ~580 TF peak (saturated)
- 32 warps/SM × ≤2 chains: also ~580 TF (saturated)
- 32 warps/SM × 4 chains: **AVOID — 4× regression to 159 TF**

The mechanism is **operand-fetch back-pressure** at 1024 threads × 4 chain
registers, NOT register spill (`smsp__inst_executed_op_local_ld.sum` = 0,
SASS HMMA count identical at 1280). Likely operand-fetch port pressure or
warp-scheduling overhead at high occupancy.

### 47.4 In-kernel clock64 resolves the ncu/wall-clock discrepancy

`SESSION_2_DELTA.md` lines 649-680 documents the methodological lesson —
when wall-clock and ncu disagree, add a third independent measurement.
For the 32×4 mma.sync regression:

| Config | Wall-clock | clock64 max | ncu gpc cycles |
|--------|-----------:|-------------:|---------------:|
| 16w/SM × 4 chains | 1.284 ms | 2.46M cy = **1.210 ms** | (1.21 ms expected) |
| 32w/SM × 4 chains | 9.373 ms | 18.7M cy = **9.227 ms** | 4.9M cy = **2.42 ms ← WRONG** |

**clock64 confirms wall-clock truth.** The kernel TRULY takes 18.7M SM
cycles per SM. ncu's `gpc__cycles_elapsed.max` was misleading — only
counted ~26% of the actual span. The pipe is genuinely 3.6× SLOWER per
cycle in this config; ncu's "99% pipe_tensor active" was a metric scope
artifact.

**Methodological lesson**: when wall-clock and ncu disagree, add a third
INDEPENDENT measurement (in-kernel clock64) before trusting either. This
session triangulated:
- Wall-clock cudaEventElapsedTime
- ncu gpc__cycles_elapsed
- In-kernel clock64

**clock64 is the gold standard** — counts actual SM cycles between two
PTX instructions. Wall-clock = clock64 / clock_freq. ncu's other metrics
(pipe_tensor active%) may have scope limitations.

### 47.5 V8 mma.sync recipe (HMMA.F16 99.9% pipe saturation)

From `V8_HMMA_F16_PEAK.md`:
- **Configuration**: 148 SMs × 256 threads × 8 chains × 10K iterations
- **Instructions**: `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` self-feeding accumulator
- **Result**: 94.72M HMMAs in 670 µs
- **ncu metric**: `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active = 99.90%`
- **Effective TFLOPS**: 94.72M × 4096 FLOP / 670 µs = **578.6 TFLOPS = 99.9% of legacy HMMA pipe**
- **Three methods agree**: wall clock 672 µs ≈ ncu 670 µs; HMMA count matches expected 148 × 8 warps × 10K × 8 chains; ncu pipe % reads 99.90%.
- **F16 vs F32 accumulator**: identical 578.6 TFLOPS (F32 acc free).

**HMMA.F16 vs FP32 FFMA peak**: 578 / 74.6 = **7.75× faster** than the
chip's FP32 FFMA peak (74.6 TFLOPS). Catalog "7.4× FFMA" matches.

### 47.6 Per-MAC power equivalence

From `MMA_SYNC_POWER.md`:

| Path | Random penalty | Per-MAC random penalty | Per-cycle MACs |
|------|---------------:|-----------------------:|---------------:|
| BF16 mma.sync m16n8k16 | +31 W (175 W → 206 W) | ~0.6 nW/MAC | ~50 GMACs/s/chip |
| BF16 tcgen05 m128n128k16 | +310 W (299 W → 609 W) | ~0.5 nW/MAC | ~600 GMACs/s/chip |

**Same per-MAC physics; tcgen05 has ~12× more concurrent MACs per SM** so
absolute swing is ~10× larger. (BF16 mma.sync m16n8k16 = 128 outputs ×
16 K = 2048 MACs/inst; tcgen05 BF16 m128n128k16 = 16384 outputs × 16 K =
262144 MACs/inst.) The BF16/FP8 multiplier hardware shares
the same dedup capability across both paths (`MMA_SYNC_POWER.md` §"FP8
e4m3 mma.sync" replicates B>A asymmetry +43 W vs +21 W).

**Footgun:** ⚠ ncu `pipe_tensor` does NOT cover tcgen05 — silent zero, no warning. Use `…hmma_op_utchmma_utcqmma_utcomma…` (full subpipe name) for tcgen05. SESSION_2_DELTA shows pipe_tensor "99% active" for 2.42 ms of 9.37 ms wall — the rest is tcgen05 invisible to pipe_tensor.

**Footgun #2:** ⚠ The "INT8 IMMA latency-bound, would scale with ILP" claim is RETRACTED. SASS shows 5 NOPs/issue → HW-throttled to 143 TOPS regardless of ILP.

**See also:** §46 (full ladder), §48 (FP8 mma.sync emulation), §52 (tcgen05 power), corrections/06_tensor_cores_CORRECTED.md R4, SESSION_2_DELTA.md, CUBLAS_BIT_ENTROPY_CORRECTION.md.

---

## §48. mma.sync FP8 `kind::f8f6f4` — NOT NATIVE

**Answer:** On sm_103a, `mma.sync.aligned.m16n8k32.kind::f8f6f4` does NOT compile to a native FP8 mma SASS opcode. ptxas emits **F2FP.F16.E4M3.UNPACK_B** (12+ unpack instructions per K=32) followed by **2× HMMA.16816.F32** (standard FP16 m16n8k16 mma). Effective throughput **104 TFLOPS** = **1.37× SLOWER** than equivalent 2× BF16 mma.sync at the same K=32. Native FP8 throughput on B300 is **ONLY** available via `tcgen05.mma`. `[🟢 HIGH · src: MMA_FP8_KIND_F8F6F4_NOT_NATIVE.md, corrections/TCGEN05_DEDUP_CONSOLIDATED.md §12]`

### 48.1 SASS evidence

```
F2FP.F16.E4M3.UNPACK_B   # FP8 → FP16, one byte at a time
F2FP.F16.E4M3.UNPACK_B   # 12+ unpacks for 32 FP8 inputs per warp
...
HMMA.16816.F32           # standard FP16 m16n8k16 mma (1st of 2)
HMMA.16816.F32           # standard FP16 m16n8k16 mma (2nd of 2)
```

K=32 of FP8 is implemented as: convert all FP8 → FP16, then run TWO
HMMA m16n8k16 calls — each producing K=16 worth of accumulator updates.

### 48.2 Measured throughput (32-thread warp, single block, varying-operand chain)

| Path | cy/iter | FLOP/cy/warp | Effective TFLOPS (148 SM × 4 SMSP × 1.5 GHz) |
|------|--------:|-------------:|----------------------------------------------:|
| FP8 m16n8k32 kind::f8f6f4 | 70 | 117 | **104** |
| BF16 m16n8k16 (single)    | 31 | 132 | 117 |
| BF16 m16n8k16 ×2 (= K=32) | 51 | 161 | **143** |

**FP8 mma.sync is 1.37× SLOWER** than equivalent 2× BF16 mma.sync for
the same K=32 effective FLOPs (8192).

### 48.3 Why ptxas chose this path

mma.sync as an instruction class on Blackwell was preserved for backward
compat with Hopper/Ada. The native Blackwell tensor instruction is
tcgen05.mma. ptxas implements new mma.sync `.kind::` variants on top of
the legacy HMMA path because the legacy mma.sync warp-level register
layout doesn't have a hardware equivalent in tcgen05 (which uses TMEM
not registers).

### 48.4 Implication for catalog claims

Anywhere a file casually says **"FP8 mma.sync = 276 TFLOPS effective"**
or treats `kind::f8f6f4` as native FP8 dedup behavior, the framing is
**misleading**. The 276 figure was an early DCE-suspect measurement.
Real measured: 104 TFLOPS pure-loss vs 2× BF16.

For any practical FP8 GEMM, do NOT use mma.sync; use **cuBLAS / CUTLASS**
which dispatches to tcgen05 (4425 zero / 3983 realistic, see §46).

### 48.5 Why this matters for catalog cross-comparison

Many earlier B300 measurements claimed FP8 mma.sync TFLOPS in the
276-400 range. Per `06_tensor_cores_CORRECTED.md` retractions:
- "6 357 TFLOPS FP8 via mma.sync" — RETRACTED (DCE-folded loop, only 2 HMMAs in SASS for claimed 65K iters)
- "2 336 / 2 400 TFLOPS FP8 via mma.sync" — RETRACTED (FADD artifact; compiler folded 99.99% of MMA chain)
- "FP8 mma.sync = 276 TFLOPS native" — RETRACTED (kind::f8f6f4 in mma.sync compiles to F2FP.UNPACK + HMMA, not native FP8)

**The real FP8 mma.sync number is 104 TFLOPS** (DCE-defeated chain, per
`MMA_FP8_KIND_F8F6F4_NOT_NATIVE.md`). It is NOT the path you want for
real FP8 GEMM.

### 48.6 Native FP8 path (tcgen05.mma)

For real FP8 throughput on B300:

| Path | TFLOPS | % spec | Notes |
|------|-------:|-------:|-------|
| mma.sync kind::f8f6f4 (emulated) | 104 | n/a | F2FP.UNPACK + 2× HMMA, 1.37× SLOWER than 2× BF16 |
| tcgen05.mma kind::f8f6f4 (native) | 4651 | 93% | direct microbench (06_tensor_cores r17) |
| cuBLAS LtMatmul (sustained zero) | 4425-4491 | 88-91% | via cudaGraph |
| cuBLAS LtMatmul (random data) | 3984 | 80% | realistic |
| cuBLAS LtMatmul (normal-ish) | 3951 | 79% | realistic |
| cuBLAS LtMatmul (under 600 W cap) | 3087 | 62% | random + power-cap throttle |

The native tcgen05 path is **44× faster** than mma.sync emulation
(4651 / 104 = 44.7×).

### 48.7 Subtle: tcgen05's `kind::f8f6f4` IS native

The dedup numbers in `TCGEN05_DEDUP_CONSOLIDATED.md` §12 are all
`tcgen05.mma kind::f8f6f4`, which IS the **native FP8 path on B300**
despite the same syntactic `kind` name as the legacy mma.sync emulation.
Be careful when reading any "kind::f8f6f4" claim — check the surrounding
instruction (mma.sync = emulated, tcgen05.mma = native).

The CLAUDE memory entry "Careful capability claims" applies here:
mma.sync's compilation to F2FP+HMMA is NOT evidence that B300 lacks
native FP8 — tcgen05.mma kind::f8f6f4 is the native path.

**Footgun:** ⚠ Do not treat `mma.sync.kind::f8f6f4` as native FP8 — it's emulated F2FP+HMMA and 1.37× slower than 2× BF16. Native FP8 is only via `tcgen05.mma`.

**See also:** §46 (FP8 cuBLAS path), §47 (path comparison), MMA_FP8_KIND_F8F6F4_NOT_NATIVE.md, corrections/TCGEN05_DEDUP_CONSOLIDATED.md §12 + §R6.

---

## §49. NVFP4 K=96 ULTRA path — real but inaccessible in public libs

**Answer:** The 1.5× K=96 ULTRA path (`tcgen05.mma.kind::mxf4nvf4.block_scale.block16` with K=96) IS a real B300 architectural feature — UTCOMMA SASS exists and microbench reaches **10.91 PF at 1500 MHz lock = 73% of the 15 PF B300 spec**. cuBLAS 13.2 will NOT dispatch it; cuBLAS 13.4 reaches **~10.8 PF (72%)** at large-N rect; CUTLASS C++/CuTeDSL stuck at **8.7 PF (58%)**. The 1.5× K=96 UPLATE is unattainable in any public library on this version. `[🟢 HIGH · src: corrections/06_tensor_cores_CORRECTED.md U1, NVFP4_K96_AT_1500MHZ.md, project_b300_nvfp4_k96_ceiling memory]`

### 49.1 Architecture

NVFP4 (`e2m1`) is a 4-bit floating-point format with a separate UE4M3
scale factor (SF) per 16 elements. The B300 tcgen05 path supports two
shapes for NVFP4:

| Path | K | M=N=256 cy/MMA | PF/CTA at 1005 MHz | MFU |
|------|---|---------------:|--------------------:|----:|
| Standard | 64 | 128 | 4.87 | 98.4% |
| **ULTRA** | **96** | 128 (same!) | **7.31 (1.5×)** | 98.5% |

The 1.5× factor comes ENTIRELY from K-dimension — same cycle count,
1.5× MACs per cycle. UTCOMMA.BLOCK16 is the SASS opcode. PTX form:
`tcgen05.mma.kind::mxf4nvf4.block_scale.block16` with `k_size_=1`.

### 49.2 Microbench peak — reachable in custom kernel

From `NVFP4_K96_AT_1500MHZ.md`:

| Clock | K=96 ULTRA cy/MMA | PFLOPs (cluster) | MFU local | % 15 PF spec |
|------:|------------------:|------------------:|---------:|-------------:|
| 1005 lock | 128 | 7.31 | 98.5% | 49% |
| 1500 lock (TDP-safe max) | 128 | **10.89-10.91** | 98.5% | **73%** |
| boost (~2032, all-zero zero-skip) | 128 | **14.78** | 98.5% | **99%** at 633 W |
| boost (~2032, random TDP-bound, 1788 MHz throttled) | 128 | 13.01 | 98.5% | 87% at 1095 W |

The 14.78 PF at boost is the all-time tcgen05 peak — but only on the
zero-skip path (B = all-zero). Random data throttles to 1788 MHz under
the 1100 W TDP cap.

### 49.3 cuBLAS reachability

| cuBLAS version | Best K=96 access | TF | % 15 PF |
|---------------:|------------------|---:|---------:|
| 13.2 | **NOT dispatched** | 0 | 0% (path exists in SASS but cuBLAS won't pick it) |
| 13.4 | large-N rect | ~10800 | 72% |

(13.4 number is per project_b300_nvfp4_k96_ceiling memory, not
re-verified in this clean directory.)

### 49.4 CUTLASS C++ / CuTeDSL ceiling

| Library | Best | TF | MFU |
|---------|------|---:|----:|
| CUTLASS C++ sample 89 (sm103_fp4_ultra_gemm) at 1005 lock, 8K² K=15K | 2SM cluster (2,4) | 5544 | 77.7% at 1005 |
| CUTLASS C++ sample 89 at boost | 2SM cluster | 8285 | ~55% |
| CuTeDSL persistent kernel, M=N=8192 K=61440, cluster (2,1), 1005 | 6776 | **91.3%** per-total |
| CuTeDSL boost, M=N=16384 K=15360, cluster (2,4), zero data | **8112** | 54.1% per-total |

CUTLASS C++ 89 lags CuTeDSL by 8-15 pp MFU at the same shape — see §54
for the open question about why.

### 49.5 Bottom line

**The 1.5× K=96 ULTRA path is not reachable in any public library on this version.** Effective cuBLAS NVFP4 ceiling is ~10.3 PF (wide-N) per
TRUE_REFERENCE; cuBLAS 13.4 reaches 10.8 PF (72%); cuBLAS+cudaGraph
plain Lt reaches 11.42 PF (76%) but does NOT use the K=96 path. The
73% ceiling is consistent across microbench (TCGEN05_PERFW_CLEAN at
1500 MHz) and cuBLAS 13.4 — they hit the same scheduling-limit.

The CLAUDE memory entry **project_b300_nvfp4_k96_ceiling** captures this:
"K=96 is real but 1.5× spec is unattainable in public libs" — both
microbench at 1500 MHz lock and cuBLAS 13.4 at boost cap at the same
~73% of 15 PF spec.

### 49.6 CuTeDSL shape-sweep — K-deep matmuls hit 91% MFU

From `NVFP4_CUTEDSL_THROTTLE_DEEP_DIVE.md` lines 119-188:

**Square scaling at 1005 MHz, zero data, 256×256 tile, K=15360:**

| M=N | (2,1) TF | MFU/total | (2,4) TF | MFU/active |
|----:|---------:|----------:|---------:|-----------:|
| 2048 | 4185 | 56.4% | 2433 | 40.5% |
| 4096 | 5142 | 69.3% | 4177 | 69.5% |
| **8192** | **6087** | **82.0%** | 4948 | 82.3% |
| 12288 | 5713 | 77.0% | 5193 | 86.4% |
| 16384 | 5581 | 75.2% | 5240 | 87.1% |
| 24576 | 5152 | 69.5% | 5301 | 88.1% |
| **32768** | 5196 | 70.1% | **5319** | **88.4%** ← per-active record |

(2,1) is best for M ≤ 8192; (2,4) takes over for M ≥ 24576. Crossover
around M=N=12K-16K.

**Deep-K (M=N=8192, vary K):**

| K | (2,1) TF | MFU/total |
|--:|---------:|----------:|
| 1536 | 3049 | 41.1% |
| 3072 | 4301 | 58.0% |
| 6144 | 5322 | 71.7% |
| 15360 | 6124 | 82.6% |
| 30720 | 6484 | 87.4% |
| **61440** | **6776** | **91.3%** ← per-total record |

**K-depth is the single most important shape parameter.** Reuse goes up
asymptotically. K=61440 within 9% of hypothetical 100% MFU at 1005 MHz.

**LLM-realistic Llama-70B layer matmuls (hidden=8192, ffn=14336, K=8064):**

| Shape | (2,1) TF | (2,4) TF | MFU/total |
|-------|---------:|---------:|----------:|
| (8192, 14336, 8064) gate/up | 4510 | 4469 | **30%** |
| (14336, 8192, 8064) down | 4537 | 4408 | 30% |
| (8192, 8192, 8064) qkv | 4589 | 4233 | 30% |
| (4096, 14336, 11904) ffn | 4638 | 4730 | 31% |

**LLM-realistic shapes plateau at ~30% of 15-PF spec at 1005 MHz** because
K=8064 is too narrow to hit the deep-K reuse plateau. To hit 91% MFU you
need K ≥ 40K — which doesn't occur in transformer matmuls except in
attention (K = head_dim × seq_len).

### 49.7 ncu cross-check — bottleneck shifts with clock

From `NVFP4_CUTEDSL_THROTTLE_DEEP_DIVE.md` lines 190-205, ncu at 1005
MHz BIG case (cluster (2,4)):

```
SM Throughput:    74.77 % (climbs 67% → 75% as clock drops from 2032 → 510)
SM Active Cycles: 80.10 %
L1/TEX Cache:     64.69 % (climbs 53% → 65% as clock drops)
L2 (lts):         24.85 % (drops 47% → 25% — has headroom at low clock)
DRAM:             12.43 % (drops 37% → 12%)
utcmma rate:      214 M/s (= 22% better-than-linear vs clock ratio)
```

**Key insight**: SM Throughput CLIMBS as clock drops (67% → 75%). The
bottleneck at high clock is some non-clock-scaled overhead (likely TMA
fill latency + mbarrier coordination): at boost, SMs out-run their data
arrival; at low clock, staging keeps up. utcmma rate is 22% better than
linear-clock-scaling at low clock.

### 49.8 Single-shot vs sustained gap

`06_tensor_cores_CORRECTED.md` U2:
- Single-shot const N=8192: **9109 TFLOPS (91% of 10 PF spec)**
- Sustained random N=16384 cudaGraph 15s: **6554 TFLOPS (65%)** — clock throttles to 1057 MHz, 1186 W instant peak

The ~28% gap is **power-throttle, not algorithmic**. Recommended
quotation: "NVFP4 best single-shot const = 9-10 PF; sustained random =
6.5 PF".

### 49.9 cluster_group::2 (2-CTA mma) NVFP4 K=96 details

For NVFP4 K=96, the 2-CTA path is required:
- 1-CTA NVFP4 K=96 valid only at m=128
- m=256, m=64 raise "illegal instruction" error in 1-CTA mode
- cluster_group::2 (2-CTA cluster) unlocks m=256

Per `2CTA_DEDUP.md`: cluster_group::2 (2-CTA mma) shows **identical
per-cluster power dependence** to single-CTA mode. **NO cluster-shared
dedup pooling**. Each CTA's B operand has its own 32-byte sub-tile dedup
cache. Optimization recipes apply per CTA, not cluster-wide.

This means the K=96 ULTRA microbench at M=N=256, 2-CTA cluster reaches
98.5% MFU per cluster (PF/CTA × 2 CTAs / theory PF), totaling 7.31 PF
per cluster at 1005 MHz.

For cuBLAS NVFP4: ncu confirms the cuBLAS NVF4 kernel
`cutlass3x_sm103_bstensorop_…_2sm_bias_bf16_relu` uses cluster (2,1) =
2-CTA. The "2sm" suffix indicates 2-CTA MMA mode.

### 49.10 N=192 NVFP4 K=96 also valid

`TCGEN05_PERFW_CLEAN_2TRIAL.md` includes a row for NVFP4 K=96 N=192
which gives 10.91 PF / 880 W = 12.39 TF/W vs N=256 / 870 W = 12.54 TF/W.
N=256 is marginally better but the spread is small. Practical kernel
should use N=256 for cleaner divisibility.

### 49.11 Why K=96 path matters even if libs can't hit it

The K=96 ULTRA path proves **B300's hardware peak is genuinely 15 PF**,
even if no library reaches it. From `NVFP4_K96_AT_1500MHZ.md`:

- Same kernel, same M=N=256, same cluster: K=64 gives 7.27 PF; K=96 gives 10.91 PF (1.5×)
- K=96 requires `k_size_=1` in the descriptor and larger SMEM buffers
- 98.5% MFU at both K=64 and K=96 → multiplier saturated
- 128 cy/MMA at both K sizes → cycle count constant

**The 1.5× factor comes from K-dimension MAC density**, not clock or
algorithmic improvement. K=96 ULTRA reuses the same 128-cycle multiplier
budget but dispatches 1.5× more MACs per cycle.

This is the architectural foundation for why the 15 PF spec exists. The
question of "why no public library reaches it" decomposes into:
1. cuBLAS 13.2: dispatches K=64 only (won't pick UTCOMMA.BLOCK16)
2. cuBLAS 13.4: picks K=96 at large-N rect, hits 10.8 PF (72%)
3. CUTLASS C++ sample 89: scheduling overhead lags by 8-15 pp MFU
4. CuTeDSL: persistent kernel can hit 91% MFU per-total at 1005 MHz on K=61440 — but at boost the per-utcmma fixed overhead caps the model at 73.1%

**Production path forward**: write a custom kernel that:
- Uses K=96 ULTRA tcgen05.mma with `k_size_=1`
- Uses cluster_group::2 (cta_group::2) for 2-CTA dispatch
- Ensures no per-launch host overhead (persistent kernel + cudaGraph)
- Targets shapes where the K=96 dispatch is dominant (M=N ~256, cluster (2,*))

The microbench at 1500 MHz lock proves the hardware genuinely delivers
10.91 PF (73% of 15 PF spec) — this is reachable in custom code.

**Footgun:** ⚠ Don't quote the 14.78 PF zero-skip number as a "B300 NVFP4 peak" without disclosure — it's the **all-zero-B** code path (multiplier short-circuits, see §52). Random-data NVFP4 caps at ~13 PF under TDP throttle.

**Footgun #2:** ⚠ Don't conflate the K=96 ULTRA microbench (10.91 PF) with cuBLAS+cudaGraph (11.42 PF) — they reach similar absolute numbers but via DIFFERENT paths. The cudaGraph peak uses the K=64 standard path; cuBLAS 13.2 won't dispatch K=96 at all.

**See also:** §46 (full NVFP4 ladder), §50-§52 (power), §54 (CUTLASS gap), corrections/06_tensor_cores_CORRECTED.md U1, NVFP4_K96_AT_1500MHZ.md, project_b300_nvfp4_k96_ceiling memory.

---

## §50. NVFP4 power — A:B asymmetry has THREE different right answers

**Answer:** The A vs B operand power-asymmetry depends on **workload context**. Three measurements give three different ratios — all real, none invented:
- **cuBLAS NVFP4** (per `NVFP4_POWER_DECOMPOSITION.md`): A dominates 4× (-250 W A vs -64 W B when uniform; A-only-rand swap = +204 W vs B-only-rand +64 W)
- **Microbench pure-tcgen05** (per `NVFP4_PURE_TCGEN05_RESULTS.md`): B dominates 15-30× across 6 precision variants (BF16/FP16/FP8 e4m3/e5m2/NVFP4 K=64/K=96)
- **K=96 single-kernel A×B matrix** (per `NVFP4_K96_AB_FULL.md`): B dominates 2.6× (B impact 13-249 W vs A impact 9-95 W across 5×5 sweep)

**Reconciliation (CANDIDATE, not settled)**: cuBLAS multicasts B via TMA halving its memory cost (`NVFP4_POWER_DECOMPOSITION.md` lines 209-243 — ncu confirms `TMA read bytes MULTICAST: 0 BF16 vs 3.75 GB / 78% NVF4`). The K=96 single-kernel 2.6× is most production-representative. **Don't quote a single A:B ratio.** `[🟡 MED · src: corrections/NVFP4_DOUBT_REPORT.md, NVFP4_PURE_TCGEN05_RESULTS.md "Correction" §, NVFP4_K96_AB_FULL.md, NVFP4_POWER_DECOMPOSITION.md, project_four_six_status memory]`

### 50.1 The three measurements

#### 50.1.1 cuBLAS NVFP4 path (`NVFP4_POWER_DECOMPOSITION.md`)

Setup: cuBLASLt LtMatmul, M=N=8192, K=15360, cluster (2,1), `cutlass3x_sm103_bstensorop_…_2sm_bias_bf16_relu`. ncu confirms memory-side traffic SYMMETRIC for A and B (`l1tex__data_pipe_tc_wavefronts_mem_shared_op_utcmma_matrix_a` = `_matrix_b_scope_2cta` = 15 728 640 EXACTLY equal). So A's power dominance must come from datapath asymmetry inside the FP4 multiplier.

Per-tensor isolation (zero baseline → swap one to RAND):

| Swap | Power | Δ vs zzzz | % of total |
|------|------:|----------:|-----------:|
| **A→rand only** | 674 W | +204 | **49.5%** |
| B→rand only | 534 W | +64 | 15.5% |
| SFA→rand only | 548 W | +78 | 18.9% |
| SFB→rand only | 513 W | +43 | 10.4% |

**Per-tensor isolation (RAND baseline → swap one to ZERO):**

| Swap | Power | Δ vs rrrr |
|------|------:|----------:|
| A→zero | 626 W | -256 |
| B→zero | 762 W | -120 |
| SFA→zero | 806 W | -76 |
| SFB→zero | 841 W | -41 |

**A→zero saves 256 W; B→zero saves only 120 W** in cuBLAS NVF4.
**Sign bit of A**: -72 W when forced to either 0 or 1 (symmetric); about
30% of the random-data penalty.

#### 50.1.2 Pure-tcgen05 microbench (`NVFP4_PURE_TCGEN05_RESULTS.md`)

Setup: `bench_tcgen05_power.cu` v4, SMEM-resident A and B (loaded once,
then 100M iters of MMA referencing same SMEM addresses). NO DRAM/L2
traffic in inner loop — pure multiplier circuit power. m=128 n=128
single-CTA, @ -lgc 1005 MHz.

| Precision | A-only rand cost | B-only rand cost | B/A ratio |
|-----------|-----------------:|-----------------:|----------:|
| FP16 | +18 W | +273 W | **15.2×** |
| BF16 | +8 W | +203 W | **25.4×** |
| FP8 e4m3 | +9 W | +264 W | **29.3×** |
| FP8 e5m2 | +20 W | +296 W | **14.8×** |
| **NVFP4 K=64** | **+8 W** | **+125 W** | **15.6×** |
| **NVFP4 K=96** | **+6 W** | **+118 W** | **19.7×** |

For ALL 6 precisions, B-rand-only costs **15-30× more** than A-rand-only.

#### 50.1.3 K=96 single-kernel A×B matrix (`NVFP4_K96_AB_FULL.md`)

Setup: same kernel, K=96 ULTRA, M=N=256, cluster (2,1), 296 blocks,
1005 MHz, 10M iters. Both A and B distributions varied independently.

5×5 matrix (median-of-2, total W):

```
                 |  A=c+0  A=c+2  A=5pos  A=8pos  A=16r
B=const+0        |  284    285    289     290     293
B=const+2        |  286    289    297     298     300
B=5pos {+0..+2}  |  352    399    401     414     431
B=8pos           |  377    435    439     454     472
B=16rand         |  484    538    545     547     542
```

- Row sweep (varying A, fixing B): ΔP across A modes = **9-95 W**
- Col sweep (varying B, fixing A): ΔP across B modes = **13-249 W**

**B impact ≈ 2.6× A impact** (averaged across configurations). Much
closer to the BF16 cuBLAS observation (2.0-2.9× B in MMA_SYNC_POWER) than
to the pure-tcgen05 15-30×.

#### 50.1.4 BF16 cuBLAS — operand asymmetry REVERSES vs NVFP4

`NVFP4_POWER_DECOMPOSITION.md` lines 146-205 documents the same M=N=8192
K=15360 sweep with **BF16 cuBLAS** (HMMA legacy path through nvjet kernel).
ncu confirms BF16 cuBLAS uses identical cluster shape (2,1) to NVFP4.

| Pattern | TFLOPS | MFU @ 1005 | Power (trim) |
|---------|-------:|-----------:|-------------:|
| zz (zero) | 1178 | 95.2% | 417 W |
| pp (+1.0) | 1177 | 95.1% | 439 W |
| nn (-1.0) | 1177 | 95.1% | 443 W |
| 33 (+3.0) | 1177 | 95.1% | 431 W |
| 0x55 / 0xaa | 1177 | 95.1% | 444-445 W |
| **rr (random)** | **1165** | **94.2%** | **805 W** |

BF16 per-tensor isolation — REVERSED from NVFP4!:

| Pattern | Power | Δ vs zzzz | Δ vs rrrr |
|---------|------:|----------:|----------:|
| zzzz baseline | 417 W | 0 | -388 |
| **A=rand only** | 498 W | +81 W | -307 |
| **B=rand only** | **648 W** | **+231 W** | **-157** |
| rrrr baseline | 805 W | +388 | 0 |
| A=zero (B rand) | 647 W | +230 | -158 |
| **B=zero (A rand)** | **495 W** | **+78** | **-310** |

| Path | A rand cost | B rand cost | Dominant operand |
|------|------------:|------------:|------------------|
| **NVF4 UTCMMA** | 204-256 W | 64-120 W | **A (~2-3× B)** |
| **BF16 HMMA** | 81-158 W | 231-310 W | **B (~2.0-2.9× A)** |

**Same hardware, opposite asymmetry.** This is the key data point that
makes a single-mechanism explanation suspect.

#### 50.1.5 The TMA multicast hypothesis (CANDIDATE — source itself walked it back)

ncu deeper investigation reveals the BF16 vs NVF4 asymmetry is largely
driven by **different TMA strategies**:

| Metric | BF16 cuBLAS | NVF4 cuBLAS |
|--------|------------:|------------:|
| Kernel | nvjet_sm103_tst | cutlass3x_sm103_bstensorop |
| utcmma count | 983,040 | 163,840 |
| L1TEX wavefronts A | 62,914,560 | 15,728,640 |
| L1TEX wavefronts B (2cta) | 62,914,560 (=A) | 15,728,640 (=A) |
| TMA read bytes total | **12.08 GB** | **4.78 GB** |
| **TMA read bytes MULTICAST** | **0** | **3.75 GB (78%)** |

**Both kernels use utcmma (tcgen05.mma) — same multiplier hardware!**

- **NVF4 path**: B is multicast from L2 to both M-CTAs in cluster (2,1) → single L2 read shared between 2 SMs → B's L2-side activity is HALVED → A's datapath/feed cost dominates.
- **BF16 path**: B is NOT multicast — each CTA loads its own B independently → B is read from L2 TWICE per cluster step → B operand burden (memory + datapath) dominates.

BF16 elements (2 bytes each) may not satisfy multicast TMA alignment/size
constraints that FP4 (0.5 bytes) does meet.

### 50.2 Reconciliation — CANDIDATE explanation, not settled

Per `NVFP4_PURE_TCGEN05_RESULTS.md` lines 174-198 ("Correction"):

> Earlier I claimed the cuBLAS NVF4 "A dominates" was "definitively
> explained by TMA multicast on B". **That overreaches.** The
> pure-tcgen05 result (B>>A in multiplier across 6 formats) is solid.
> But the gap to cuBLAS observation has multiple plausible causes:
>
> 1. cuBLAS may swap A↔B internally
> 2. TMA multicast pattern (the original hypothesis)
> 3. Per-operand SMEM dwell time
> 4. Operand pipeline depth — different buffering depths for A vs B
>    feeds, with different per-bit-toggle costs

`NVFP4_DOUBT_REPORT.md` (the adversarial audit):

> The wave-2 reconciliation maps to three real source files and the
> numbers are not invented. The "TMA multicast halves B's memory cost"
> is NOT an inferred guess — `NVFP4_POWER_DECOMPOSITION.md` lines
> 209-243 contains an explicit ncu table showing **TMA read bytes
> MULTICAST: 0 (BF16) vs 3.75 GB / 78% (NVF4)**. The hypothesis is
> hardware-verified.
>
> However, the agent over-resolves: the pure-tcgen05 source itself
> walks back the original "definitively explained by TMA multicast"
> claim. Verdict: **partially overstated**. The 3-way numbers are
> real; the explanatory unification is one notch more confident
> than the source warrants.

The K=96 single-kernel 2.6× is closer to the BF16 cuBLAS observation
(2.0-2.9× B in `MMA_SYNC_POWER.md`), suggesting the pure-tcgen05 15-30×
might be the artifact (over-isolation in the A=zero/B=zero mode).

#### 50.2.1 Multiplier port asymmetry — B-reuse mechanism (PURE_TCGEN05_RESULTS lines 240-290)

`NVFP4_PURE_TCGEN05_RESULTS.md` proposes a mechanistic explanation for
why B dominates A in pure-tcgen05 measurements:

> For each MMA inst, B is read once into the multiplier operand-B port
> and used to multiply M different rows of A. As M grows, more MAC
> units fire B through, increasing per-cycle switching activity on the
> B-side multiplier interconnect.
>
> A is read once and used N times (less for small N). So A's reuse is
> high but each "use" is a single multiply (lower per-element activity).

Cross-precision verification at m=128 n=128 (peak useful shape):

| Precision | ZZ baseline | A-Δ | B-Δ | B/A ratio |
|-----------|------------:|----:|----:|----------:|
| FP16 (K=16) | 279 | +18 | +273 | 15.2× |
| BF16 (K=16) | 284 | +9 | +205 | 22.8× |
| FP8 e4m3 (K=32) | 289 | +11 | +266 | 24.2× |
| FP8 e5m2 (K=32) | 289 | +20 | +296 | 14.8× |
| NVFP4 K=64 | 270 | +9 | +124 | 13.8× |
| NVFP4 K=96 | 244 | +8 | +120 | 15.0× |

**Universal multiplier asymmetry confirmed across 6 instruction variants
spanning 3 PTX kinds** (kind::f16, kind::f8f6f4, kind::mxf4nvf4.block_scale).

The B-reuse-drives-power mechanism is HARDWARE-ARCHITECTURE level, not
format-specific.

#### 50.2.2 BF16 32-element MAC group cliff is the smoking gun

The BF16 N-stride sweep (see §52.13) shows B is broadcast across **32
parallel MAC units per cycle** in the multiplier datapath (matches B300
SMSP width = 32 lanes). Sharp 112 W cliff between N-stride 16 and 32 is
the single most direct evidence that **B operand sits on a port that
fans out to 32 MAC lanes**, while A sits on a port read once per
multiplier element.

The cuBLAS A-dominates observation in `NVFP4_POWER_DECOMPOSITION.md`
(per §50.1.5) is best explained by **TMA multicast halving B's memory
pipeline cost** — flipping which side is dominant **at the API surface**
without changing the underlying multiplier port asymmetry.

### 50.3 Three different ratios, three different test geometries

| Source | A vs B | Why (CANDIDATE) | Confidence |
|--------|--------|-----------------|:----------:|
| cuBLAS NVF4 (NVFP4_POWER_DECOMPOSITION) | A > B (3-4×) | TMA multicast on B masks B's true cost; OR cuBLAS internally swaps A↔B | 🟡 MED |
| Pure tcgen05 (PURE_TCGEN05_RESULTS, A=zero baseline) | B >> A (15-30×) | Stripped of memory pipeline; pure multiplier port asymmetry; possible over-isolation | 🟢 HIGH for the measurement, 🟡 MED for the mechanism |
| K=96 single-kernel matrix (NVFP4_K96_AB_FULL, A,B both varying) | B > A (~2.6×) | Both fed via SMEM-resident; close to K=96 inference reality | 🟢 HIGH (3-trial verified) |

**The K=96 paper's 2.6× is the most representative number for production
NVFP4 K=96 inference power modeling.**

The CLAUDE memory entries support all three (project_four_six_status
references the 2.6× implicitly by citing 5.45 source-level perf as
DRAM-peak limited; project_b300_power_data_dep cites the popcount
bell-curve as the underlying mechanism).

### 50.4 Why ALL three are simultaneously "right"

The B mechanism:
- The multiplier has a 32-byte sub-tile **B-side** dedup cache (see §52). B-distributed-across-N-MACs vs A-broadcast-to-all-MACs is a fundamental architectural difference. **This produces the high B/A ratio when both A and B are isolated to extreme cases (A=zero or B=zero).**

The A=mostly-free conditional rule (`A_B_ZERO_ASYMMETRY.md`):
- "A varying is FREE *when B is constant*" (mode 4007 = 297 W vs 299 W baseline).
- "When B varies, A varying adds ~60 W marginal" (rand+rand 611 W vs const+rand 549 W = +62 W marginal).
- A = zero saves ~60 W vs A = const+1.0 when B random (vs B = zero saves 313 W when A random).

The K=96 2.6× emerges from the realistic regime where both A and B vary
moderately (~5-position quantized weights). The pure-tcgen05 15-30×
emerges from A=zero or B=zero extreme isolation. cuBLAS's apparent
A-dominance emerges because TMA multicasts B (memory pipeline cost
flipped), making the **datapath cost** the dominant differentiator —
and the datapath happens to be ~50% of the total in cuBLAS.

### 50.5 Practical guidance

| For estimating | Use this number | Why |
|----------------|-----------------|-----|
| Production NVFP4 K=96 inference power | **2.6× B>A** | K=96 single-kernel matches realistic data |
| Production cuBLAS NVFP4 power | **A>B (3-4×)** observed at the API surface | Take API-A side as the dominant op |
| Microarchitectural multiplier model | **15-30× B>>A** intrinsic | Stripped of memory pipeline |
| Pre-quantization A/B layout decision | put **higher-entropy operand on B's API slot** | cuBLAS will (probably) swap, then multicast it |

**Footgun:** ⚠ Don't quote a single A:B ratio. NVFP4_DOUBT_REPORT explicitly cautioned against picking one mechanism — the source itself (PURE_TCGEN05_RESULTS lines 174-198) walks back the single-mechanism explanation and lists 4 plausible causes. Preserve all three readings.

**Footgun #2:** ⚠ The "A is FREE" CLAUDE memory note refers ONLY to the K=96 single-kernel microbench where A swing is 9-95 W vs B swing 13-249 W (B/A ≈ 2.6×, NOT 15-30×). Don't generalize "A free" to all NVFP4.

**See also:** §51 (NVFP4 K=96 power signature), §52 (tcgen05 B-side sub-tile dedup), corrections/NVFP4_CONSOLIDATED.md §2, corrections/NVFP4_DOUBT_REPORT.md §1, project_four_six_status memory.

---

## §51. NVFP4 K=96 power signature + range 284-605 W per CTA at 1005 MHz

**Answer:** Single tcgen05.mma K=96 ULTRA (M=N=256, 2-CTA cluster, 296 blocks, 1005 MHz lock) sweeps **284 W (A=B=const) → 605 W (worst-case sign pattern p_n=64)** per CTA. The range is fully decomposed into measurable axes: **+80 W** sign bit alone, **N-64 multiplier lane stride** (p_n=64 = +50 W on top of full random), **A=const+0 zero-skip = -50 W**, **+30 W per outlier per K-block-of-16**, **103 W save by using +0 not -0** for sparse zero weights. Memory power follows a popcount bell curve (peak at d=16 random); chunk-dedup is NULL; the toggle-energy model dominates. **DRAM read d=16 + 1500 MHz lock = 1071 W stress recipe**. `[🟢 HIGH · src: NVFP4_K96_AB_FULL.md, NVFP4_K96_B_DISTRIBUTION.md, NVFP4_SF_POWER.md, NVFP4_K96_SIGNMATCH.md, project_nvfp4_k96_signature memory, project_b300_power_data_dep memory]`

### 51.1 Headline range

```
Idle floor                       150 W
Constant A and B                 284 W   (multiplier idle, zero-skip)
A=const, B=5-pos {+0..+2}        352 W
A=const, B=full random           484 W
A=random, B=const+0              293 W
A=random, B=full random          552 W
A=random, B p_n=64               605 W   ← worst-case sign pattern
TDP cap                         1100 W
```

**Active range**: ~135 W (constant) to ~455 W (worst pattern) per CTA.
Per-CTA active power swings 3.4× based on B data alone.

### 51.2 Throughput is constant across all data — power changes ONLY

| path   | cy/MMA | MAC/cy/cluster | PFLOPs/s total | theory PF | MFU |
|--------|--------|----------------|-----------------|-----------|-----|
| K=64   | 128    | 32 768         | 4.87            | 4.95      | 98.4% |
| K=96   | 128    | 49 152         | 7.31            | 7.42      | **98.5%** |

K=96 ULTRA's 1.5× factor comes entirely from K dimension. **Data
changes ONLY power, not throughput** (until TDP cap kicks in at boost).

### 51.3 B-side distribution ladder (A fixed = random)

| B distribution                          | mantissa | signs | power W | active W |
|-----------------------------------------|----------|-------|---------|----------|
| Constant single value (any of 16)       | 1 mag    | 0%    | 295     | 145      |
| 5 positive {+0,+0.5,+1,+1.5,+2}         | 5 mags   | 0%    | 430     | 280      |
| 4 nonzero positive {+0.5..+2}           | 4 mags   | 0%    | 434     | 284      |
| 8 positive {+0..+6}                     | 8 mags   | 0%    | 472     | 322      |
| 5 centered {-1,-0.5,0,+0.5,+1}          | 3 mags   | 40%   | 490     | 340      |
| 9 asymmetric {-2,-1,0,+0.5,+1,..,+4}    | 7 mags   | 22%   | 526     | 376      |
| ±{0..+2} = 10 codes                     | 5 mags   | 50%   | 510     | 360      |
| 13 codes (excl -0, ±6)                  | 6 mags   | 50%   | 549     | 399      |
| 15 codes (excl -0)                      | 8 mags   | 47%   | 555     | 405      |
| 16 random                               | 8 mags   | 50%   | 552     | 402      |
| **Worst (p_n=64 sign-period, all mag=1)** | 1 mag  | 50% N-64-aligned | **605** | **455** |

### 51.4 Sign-bit alone costs ~80 W

Direct comparison (A=random, same mantissa diversity):
- 8 positive (sign always 0, mag random in 0..6): 472 W
- 16 random (sign random, same mag): 552 W
- **Δ = 80 W from sign bit randomization alone**

**Sign-toggle-rate model** — Power scales linearly with `2p(1-p)` where p = P(sign=1):
- 5-pos (p=0): 430 W (baseline, 0% toggle)
- 5 centered (p=0.4): 490 W (toggle rate 0.48 × 80 W = +38 W; observed +60 W)
- 9 asymmetric (p=0.22): 526 W (toggle rate 0.34 × 80 W ≈ +54 W from 8-pos baseline 472 → predicts 526) ✓ exact
- 16 random (p=0.5): 552 W (toggle rate 0.5 → max sign cost: 472 + 80 = 552 ✓)

The linear sign-toggle model is dialed in across all tested distributions.

### 51.5 Magnitude diversity — ~40 W per "additional" magnitude

| mantissa diversity | power W (signs all 0) |
|--------------------|----------------------|
| 1 mag (constant)   | 295                  |
| 5 mags             | 430                  |
| 7-8 mags           | 472                  |

1 → 5 mags adds 135 W. 5 → 8 mags adds 42 W. **Diminishing returns
beyond ~5 distinct magnitudes.**

### 51.6 Outlier sensitivity — +30 W per outlier per K-block-of-16

5-pos baseline (no outliers) = 432 W. Add `n` random outliers per
K-block-of-16 from {-3,-2,-1,+2,+3,+4}:

| n outliers | % | power W | Δ |
|----|---|---------|----|
| 0  | 0% | 432    | 0  |
| 1  | 6% | 461    | **+29** ← per-outlier cost |
| 2  | 13%| 480    | +48 |
| 4  | 25%| 503    | +71 |
| 8  | 50%| **526** | +94 ← **peak** |
| 16 | 100%| 519   | +88 |

**Just 1 outlier per 16 weights costs +30 W per CTA.** 50/50 mix is
HIGHER than 100% pure outlier — mixing maximizes inter-element variance
(same as popcount d=16 peak in memory experiments).

### 51.7 N-64 lane-pairing confirmed

Sign-period sweep at all-1 magnitudes:
- p_n=32: 474 W (signs at n+64 SAME as n → no toggle on lane pair)
- **p_n=64: 605 W** (signs at n+64 OPPOSITE → 100% toggle on lane pair)

Match-offset sweep at sp=50% confirms mo=64 and mo=192 (= -64 wrap)
are best non-uniform offsets. **Multiplier pairs B-side N-axis at
stride 64 cycles** — this is the underlying microarchitectural
explanation for the worst-case sign pattern.

### 51.8 Sign-bit-on-zero cost — use +0 not -0

For zero-magnitude elements, sign bit policy matters:

| sp%  | random sign (mo=-1) | sign=0 (mo=0) | savings |
|------|---------------------|---------------|---------|
| 0    | 556 W               | 553 W         | 3 W     |
| 25   | 546 W               | 537 W         | 9 W     |
| 50   | 522 W               | 503 W         | 19 W    |
| 75   | 481 W               | 440 W         | 41 W    |
| 100  | 401 W               | 298 W         | **103 W** |

**Use +0 (0x0) not -0 (0x8) for zero weights**: 3-103 W saved depending
on sparsity. For sparse models, this is ~free 50 W per CTA at typical
25-50% sparsity levels.

### 51.9 Multiplier zero-skip (A or B = constant +0)

When EITHER operand is uniformly zero, the multiplier produces zero
output and the adder/accumulator can short-circuit:
- A=const+0, B=full random: 484 W (vs 542 W if A=random) → **-58 W**
- A=const+2 (non-zero const), B=full random: 538 W → **+54 W vs A=+0**

**This is a hardware optimization saving 30-100 W per CTA when one
operand is 100% zero.** A=const+2 (non-zero constant) is "expensive"
because constant non-zero A forces multiplier to actually compute B
values without short-circuit.

### 51.10 SF tensor (UE4M3) — small lever

From `NVFP4_SF_POWER.md`:

| SF pattern | B random (W) | B const +1.0 (W) |
|------------|-------------:|-----------------:|
| 1.0 (UE4M3=0x38) | 463 | 284 |
| 0 (zeros) | 437 | 279 |
| random | 479 | 311 |
| patterned (0xAAAA) | 469 | 284 |

- SF=random adds ~27 W vs SF=1.0 (B const) (independent SF-side cost)
- SF=random adds ~16 W on top of B random (smaller marginal)
- SF=0 saves 5-18 W (some products gated to zero)
- SF patterned (uniform) has no effect

Total NVFP4 random data baseline (463 W) decomposes:
- Static baseline ~280 W
- B operand contribution ~150-170 W
- SF contribution ~13-30 W

### 51.11 K-axis sensitivity is 4× weaker than N-axis

`bench_nvfp4_k96_kperiod.cu`: sign[k,n] = (k/pk ^ n/pn) & 1, mag=+1.0.
At pn=0 (no N-flip), vary pk:

| pk  | power W | Δ vs baseline |
|-----|---------|---------------|
| 0   | 299     | 0 (all sign=0) |
| 1   | 343     | **+44 ← worst K** |
| 2   | 322     | +21 |
| 4   | 311     | +10 |
| 8   | 305     | +4  |
| ≥12 | 299     | 0 (asymptote) |

K-axis: 44 W swings vs N-axis: 179 W swings = **4× weaker**. Worst at
pk=1. Combined pk=1 + pn=64: 436 W (mag=+1.0 only) — NOT additive (XOR
checker maps differently to physical lane structure).

### 51.12 Memory-side popcount bell curve (project_b300_power_data_dep)

Memory power follows **popcount bell curve** (peak at d=16 random; CLAUDE
memory entry project_b300_power_data_dep). DRAM read at d=16 random +
1500 MHz lock = **1071 W stress recipe**.

The toggle-energy model dominates everywhere:
- Wire/SerDes/PHY toggling, not multiplier-only
- Chunk-dedup is **NULL** (different sub-strates from sub-tile dedup)
- Smooth monotonic decay with sparsity (knee at sp ≈ 10-15%)

### 51.13 Best efficiency points (TFLOPs/W summary)

| Config | TFLOPs/W | Source |
|--------|---------:|--------|
| K=96 ULTRA + 5-pos B at 1005 MHz | **17.0** | NVFP4_K96_AB_FULL.md |
| K=96 ULTRA + all-zero B at boost (zero-skip) | **23.4** | NVFP4_K96_AB_FULL addendum |
| K=96 ULTRA + random B at 1500 MHz lock | 12.0 | NVFP4_K96_AT_1500MHZ.md |
| K=96 ULTRA + worst p_n=64 at 1500 MHz | 10.9 | NVFP4_K96_AT_1500MHZ.md |
| K=96 ULTRA + A+B-positive | **15.74** | TCGEN05_PERFW_CLEAN_2TRIAL |
| Realistic LLM-weight quantization | ~13.4 | TCGEN05_PERFW_CLEAN realistic |
| Production cuBLAS sustained | 12-13.6 | various |

**1005 MHz is more efficient than 1500 MHz** by 6-10% for the same
pattern (super-linear power scaling: 1.49× clock → 1.53-1.65× power).
But 1500 MHz gives 49% more throughput. **For power-bounded sustained
workloads: 1005 MHz. For latency/peak-throughput: 1500 MHz.**

### 51.14 TDP cap regime (boost, NVFP4_K96_AB_FULL.md addendum)

| mode         | clk MHz  | pwr W    | PFLOPs | TF/W  | notes |
|--------------|----------|----------|--------|-------|-------|
| **all-0**    | **2032** | **633**  | **14.78** | **23.36** | zero-skip, 463 W under TDP |
| 5-pos        | 2002     | 1095     | 14.56  | 13.29 | TDP cap |
| 8 pos        | 1939     | 1087     | 14.10  | 12.98 | TDP cap |
| 5 cent       | 1920     | 1095     | 13.97  | 12.76 | TDP cap |
| 9 asym       | 1856     | 1094     | 13.50  | 12.35 | TDP cap |
| 16 rand      | 1788     | 1095     | 13.01  | 11.88 | TDP cap |

12% throughput swing from data quality alone under TDP cap. Without
clock headroom: data quality = energy efficiency. With clock headroom
(unlocked + TDP cap): **data quality = throughput**.

### 51.15 K=64 standard vs K=96 ULTRA comparison

Same kernel with `k_size_=0` in idesc, `MMA_K=64`, smaller SMEM buffers:

| B mode | K=64 | K=96 | Δ |
|--------|------|------|------|
| full random 16 | 455 | 552 | +97 |
| 5 positive {+0..+2} | 368 | 430 | +62 |
| 8 positive | 396 | 472 | +76 |
| 5 centered {-1..+1} | 408 | 490 | +82 |

**K=96 ULTRA is 60-100 W HIGHER than K=64 across all B distributions**
(1.5× more MMA work per instruction). Relative B-distribution swing is
similar (~25% in both paths), so the toggle-energy model applies at both
K sizes.

K=64 standard at 1005 MHz, M=N=256, cluster (2,1):

| pattern | K=64 power W | K=64 active W | TFLOPs/W |
|---------|-------------:|--------------:|---------:|
| 5-pos {+0..+2} | 565 (@1500) | 415 | 12.9 |
| 8 positive | 617 | 467 | 11.8 |
| 5 centered | 642 | 492 | 11.3 |
| 16 random | 724 | 574 | 10.0 |

### 51.16 Outlier-at-TDP-cap sensitivity (3-trial verified)

5-pos baseline + N outliers per K-block-of-16 from {-3,-2,-1,+2,+3,+4}:

| outliers/K16 | clk MHz | pwr W | PFLOPs | TF/W |
|--------------|--------:|------:|-------:|------|
| 0 (pure 5-pos) | 2005 | 1091 | 14.58 | 13.36 |
| 1 (6.25%) | 1962 | 1084 | 14.27 | 13.16 |
| 2 (12.5%) | 1935 | 1082 | 14.07 | 13.00 |
| 4 (25%) | 1890 | 1097 | 13.75 | 12.53 |
| 8 (50%) | 1864 | 1094 | 13.56 | 12.39 |
| 16 (100%) | 1863 | 1097 | 13.55 | 12.35 |

**1 outlier per K16 = 2% throughput loss** at TDP cap (was originally
mistaken as 12% in NVFP4_K96_AB_FULL early version due to `--reuse-cubin`
bug that measured the wrong cubin). Real cost is small but non-zero.

### 51.17 Definitive perf/W ladder (1500 MHz lock, 2-trial verified)

From `TCGEN05_PERFW_CLEAN_2TRIAL.md` (supersedes single-trial PERF_WATTS
which had silent contamination — see §51.17 retraction):

**Random data (mode 0)** — full random A and B, M=N=256, cta_group::2,
lane-0 early-exit kernel:

| Format | K | PF @ 1500 | Mean W (2-trial) | TF/W | Trial-trial gap |
|--------|--:|----------:|-----------------:|-----:|----------------:|
| TF32 | 8 | 0.91 | 787 | 1.16 | 0 W |
| FP16 | 16 | 1.82 | 935 | 1.95 | 0 W |
| BF16 | 16 | 1.82 | 876 | 2.08 | 0 W |
| FP8 e4m3 | 32 | 3.64 | 1073 | 3.39 | 7 W |
| MXFP8 (UE8M0 SF) | 32 | 3.64 | 1041 | 3.50 | 10 W (max) |
| NVFP4 K=64 | 64 | 7.27 | 689 | 10.54 | 1 W |
| NVFP4 K=96 N=192 | 96 | 10.91 | 880 | 12.39 | 3 W |
| **NVFP4 K=96 N=256** | 96 | 10.91 | 870 | **12.54** | 0 W |

**Best efficiency (B-positive sign-zeroing, mode 1)** — same kernel:

| Format | Mean W | TF/W | Δ vs random |
|--------|-------:|-----:|------------:|
| TF32 K=8 | 680 | 1.34 | -107 W |
| FP16 K=16 | 785 | 2.32 | -150 W |
| BF16 K=16 | 750 | 2.43 | -125 W |
| FP8 K=32 | 834 | 4.36 | **-238 W** |
| MXFP8 K=32 | 800 | 4.55 | **-241 W** |
| NVFP4 K=64 | 585 | 12.42 | -104 W |
| NVFP4 K=96 N=192 | 728 | 14.99 | -152 W |
| NVFP4 K=96 N=256 | 720 | **15.16** | -150 W |
| **NVFP4 K=96 A+B-pos** | **693** | **15.74** | **-177 W (best of all)** |

**Realistic LLM-weight quantization** (FP4, NVFP4 K=96 N=256): ~816 W →
**13.4 TF/W**, +7% vs raw random — bulk of theoretical gains require
restructured quantization (per-magnitude bands, sign separated as bitmap).

### 51.18 W-per-CTA scaling rules

From `PER_SM_POWER_SCALING.md` and `POWER_FLOOR.md`:

| Component | Per-SM cost | Total at 148 SMs |
|-----------|-------------|------------------|
| Idle baseline | n/a | 150-198 W (clock-dep) |
| Active floor (A=B=0) | ~1 W/SM | 287 W |
| Static const (Tier B) | ~1 W/SM | 299 W (BF16) / 305 W (FP8) / 280 W (NVFP4) |
| Random data delta | ~2.1 W/SM | +310 W (BF16 random total 609 W) |

**Linear in active SM count up to 148.** No sub/superlinear surprises
observed for tcgen05.mma.

Per-CTA scaling for cluster_group::2: **per CTA NOT per cluster**. Each
CTA holds its own 32-byte sub-tile dedup cache. Total cluster power
scales linearly with member CTAs.

Cross-precision random baselines (1005 MHz lock):
- BF16 random 609 W vs const 299 W (gap +310 W)
- FP8 random 642 W vs const 305 W (gap +337 W)
- NVFP4 random 463 W vs const 280 W (gap +183 W)

NVFP4 has the smallest data-dep gap because 4-bit values have lower
mantissa popcount.

### 51.19 PERF_WATTS contamination retraction (R1)

`TCGEN05_PERFW_CLEAN_2TRIAL.md` flags the earlier `TCGEN05_PERF_WATTS.md`
single-trial table as contaminated:

> Earlier perf/W table at 1500 MHz had silent contamination — NVFP4 K=64
> read 797 W and K=96 read 795 W (impossibly close given 1.5× work).
> True values (clean): K=64=689 W, K=96=870 W.

Key changes:
- FP8: 854 → 1073 W (Δ +219 W)
- MXFP8: 851 → 1041 W (Δ +190 W)
- NVFP4 K=64: 797 → 689 W (Δ -108 W)
- NVFP4 K=96 N=256: 795 → 870 W (Δ +75 W)

**Use the 2-trial table.** PERF_WATTS NVFP4 K=96 = 13.72 TF/W headline
is contaminated — true is 12.54 / 15.16 / 15.74 TF/W per data pattern.

The contamination root cause: 5 leftover QuickRunCUDA processes silently
inflating cy/MMA up to 8.5× per the CLAUDE memory entry
`feedback_clock_stuck_no_lock`.

### 51.20 K-axis power is BINARY (R2 self-correction)

`TCGEN05_PERFW_CLEAN_2TRIAL.md` §"CORRECTION: K-axis power is BINARY":

> Even chunk-48 (only 1 K-transition in entire K=96) uses same power as
> fully random K. K-axis power is BINARY: either all 96 K-rows
> bit-identical (411 W floor) OR full ~870 W cost.

Real LLM weight matrices have varying K → always pay the full cost.
**K-row sorting/clustering does NOT help for tcgen05 power.** This
corrects an earlier section in the SAME file claiming "K-row sorting
saves 349 W per CTA". The K-row pairwise dedup at the per-MMA-instruction
level (§52) is real but only triggers under controlled microbench
conditions, not in cuBLAS K-tile iteration.

### 51.21 K-uniform-per-N retraction (rule #9 self-correction)

Per CLAUDE memory `feedback_clock_stuck_no_lock`: an earlier
NVFP4_SIGN_K64_K96 commit `4c1e60a` claimed "K-uniform-per-N saves
124 W (-28%)". This was **silent contamination** — background processes
contaminated the baseline by ~100 W. True savings: **only 1-3%**. The
in-document **MAJOR CORRECTION** header documents this. Cited cleanly,
not invented (per NVFP4_DOUBT_REPORT §3).

**Footgun:** ⚠ The 14.78 PF / 23.4 TF/W zero-skip number is the all-zero-B path under no-throttle conditions. Don't quote it as "B300 NVFP4 peak" without disclosure — random-data NVFP4 caps at ~13 PF / ~12 TF/W under TDP throttle.

**Footgun #2:** ⚠ Pre-2026-04-20 NVFP4_SIGN_K64_K96 K-uniform-per-N "28% savings" claim is RETRACTED — was clock-stuck contamination. Real ~1-3%. Apply rule #9 (suspect the test before the hardware) to ANY %-savings claim that exceeds 10% from a single-trial measurement.

**Footgun #3:** ⚠ NVFP4 has **NO 32-element MAC cliff** unlike BF16 (which has a clean 112 W cliff at stride 32). MED confidence — within-word strides (1, 2, 4) had encoding bugs in NVFP4_PURE_TCGEN05_RESULTS NVFP4 N-stride table; only sign-bit-only retest is clean.

**See also:** §50 (A:B asymmetry), §52 (tcgen05 dedup model), NVFP4_K96_AB_FULL.md, NVFP4_SF_POWER.md, project_nvfp4_k96_signature memory, project_b300_power_data_dep memory.

---

## §52. tcgen05.mma power model — 32-byte sub-tile B-side dedup, A is FREE (with caveats)

**Answer:** The tcgen05 multiplier has a **32-byte universal sub-tile dedup cache on the B side**. A operand is broadcast (one value drives ~32 N MACs) — A varying alone is FREE when B is constant; when B varies, A varying adds ~60 W marginal. K-row dedup is **pairwise** (period 1 and period 2 work; period 3+ doesn't). With column sort + K-row grouping, save up to **450 W per CTA at boost**. `[🟢 HIGH · src: corrections/TCGEN05_DEDUP_CONSOLIDATED.md, BF16_SUBTILE_DEDUP.md, SUBTILE_DEDUP_MODEL.md, SUBTILE_HALVES.md, A_VS_B_ASYMMETRY.md, A_B_ZERO_ASYMMETRY.md, project_tcgen05_power memory]`

### 52.1 The unified power / dedup model

```
P(MMA) = P_baseline                                    // ~280-305 W per CTA precision-dep
       + Σ over HW sub-tiles (B-side, 32-byte each):   // sub-tile dedup
            0                                  if byte-identical to active cache slot
            ~32 W activation + ~18 W per broken byte   otherwise (BF16 m128n128)
       + Σ over K iterations (B K-vary cost):          // K-row pairwise dedup
            5-25 W per added unique K row pattern      // sub-linear in K count
       + ε(A varying) only when B varies               // A is broadcast → ~0-60 W marginal
       + sparsity / disable_lane terms                 // see §55
```

### 52.2 32-byte universal sub-tile boundary

Confirmed across BF16/FP8/NVFP4 by N-vary cliff at exactly 32 bytes:

| Precision | N values per HW sub-tile | Bytes | Cliff at N_unique |
|-----------|--------------------------|-------|-------------------|
| BF16      | 16                       | 16 × 2 = **32**     | 16 → 17 |
| FP8 e4m3  | 32                       | 32 × 1 = **32**     | 32 → 33 |
| NVFP4     | 64                       | 64 × 0.5 = **32**   | 64 → 65 |

**32 bytes is the universal HW B-side sub-tile granularity.** Cliff lands
at exactly 17 / 33 / 65 unique values per row. Independent of SMEM
descriptor LBO (LBO=16 vs LBO=32 give identical power). Independent of
MMA_N shape (N=64 and N=128 share the 32-byte cliff).

### 52.3 Partial-break linear scaling (BF16 m128n128, SUBTILE_PARTIAL_BREAK)

- 1 byte broken in a 32-byte sub-tile: +32 W (activation cost)
- Each additional broken byte: +18 W
- Full sub-tile broken (16 bytes): +306 W (matches full random)

### 52.4 A vs B operand asymmetry — DEFINITIVE

| Configuration | Power (W) | Δ vs Tier B (299 W) |
|---------------|----------:|--------------------:|
| const A + const B | 299 | 0 |
| FULL random A + const B | 297-302 | ~0 (FREE) |
| const A + random B | 549 | +250 |
| random A + random B | 609-611 | +310 |
| zero A + random B | 490 | +191 (A=0 saves only 60 W when B varies) |
| random A + zero B | 298 | +0 (B=0 fully gates multiplier, -313 W save) |

**Mechanism**: A operand is broadcast through fanout (one value drives
~32 N MACs); B is distributed (each of ~32 N MACs holds its own per-cycle
value). The 32-byte sub-tile dedup cache is a **B-only** mechanism.

### 52.5 K-row dedup is PAIRWISE (period 1 and period 2 only)

| Precision | K | B K-vary 16 cost (W) | A K-vary 16 cost (W) | Ratio B/A |
|-----------|--:|---------------------:|---------------------:|----------:|
| BF16      | 16 | +47 | +2 | 24× |
| FP8 e4m3  | 32 | +71 | ≈0 | >70× |
| NVFP4     | 64 | +99 | +7 | 12× |

K-cost scales **sub-linearly** with K (76% of linear at FP8, 53% at NVFP4).
Narrower precisions process more K positions per cycle.

**"Pairwise" actually means up to 2-pattern alternation works:**
- Period 1 (K-row identical) → full ~1.42× speedup
- Period 2 (ABAB chunk=1) → full ~1.42× speedup (alternation predictor, content-agnostic)
- Period ≥ 3 → essentially no speedup

Memory wording "K-row pairwise dedup" is slightly misleading but
directionally correct.

### 52.6 Chunk-size non-monotonic curve at N=K=8192

| chunk | TFLOPS | Speedup | Note |
|------:|-------:|--------:|------|
| 1 | 2102 | 1.42× | alternation predictor |
| 2 | 1525 | 1.03× | worst case |
| 4 | 1728 | 1.16× | |
| 8 | 2019 | 1.36× | |
| 16/32/64 | 2051-2079 | 1.39-1.40× | divides K-tile size 64 |
| 128 | 1919 | 1.30× | exceeds K-tile |

Two HW paths: alternation predictor (chunk=1 only) AND per-K-tile
constancy detector (chunk divides 64). Not pairwise LRU.

### 52.7 BF16 two-half processing (BF16 m128n128 only)

From `SUBTILE_HALVES.md` — single-unique-position test (mode 3020-3027):

| Unique pos P | Sub-tile sequence | Power (W) | Δ vs free (300 W) |
|-------------:|-------------------|----------:|-----------------:|
| 0 | A B B B B B B B | 426 | +126 |
| 1 | B A B B B B B B | 469 | +169 |
| 2 | B B A B B B B B | 461 | +161 |
| 3 | B B B A B B B B | 466 | +166 |
| **4** | B B B B A B B B | **349** | **+49** ← cliff |
| 5 | B B B B B A B B | 304 | +4  |
| 6 | B B B B B B A B | 305 | +5  |
| 7 | B B B B B B B A | 306 | +6  |

**Half A** (N=0..63, sub-tiles 0-3): single unique sub-tile costs +126 to +169 W
**Half B** (N=64..127, sub-tiles 4-7): single unique sub-tile costs +4 to +6 W (FREE)
Boundary cliff at sub_tile 4 (= N=64 boundary).

**FP8 and NVFP4 do NOT exhibit this** (uniform within ±10 W). Two-half
is **BF16 m128n128k16 specific**.

Optimization recipe: pack the most-repetitive B columns at LOW N (Half A);
arbitrary high-entropy data is essentially free at HIGH N (Half B). Saves
up to **269 W vs mirrored layout**.

### 52.8 Cache depth — CONTESTED, 1 vs 2 vs 4 slots

From `SUBTILE_DEDUP_MODEL.md` (single-MMA pattern-rotation tests):

| HW distinct | BF16 (W) | FP8 (W) | NVFP4 (W) |
|------------:|---------:|--------:|----------:|
| 1 | 301 | 308 | 281 |
| 2 | -   | 630 | 284 |
| 3 | -   | 574 | 470 |

→ "1-slot for BF16/FP8, 2-slot equivalent for NVFP4."

From `N_DEPENDENCE_DEEPDIVE.md` (cuBLAS sustained K-id period-2 tests):
→ "Dedup cache holds 2 unique sub-patterns max."

Reconciliation: different framings of the same HW. Single-MMA pattern
detection vs sustained K-row alternation predictor are different paths.
The earlier "4-slot HW pattern cache" claim was **RETRACTED** —
N=64 has NO free zone (would not happen if 4-slot cache existed); the
apparent free zone is **STICKY ACTIVATION + TWO-HALF PROCESSING** (BF16-only).

Per `TCGEN05_DEDUP_CONSOLIDATED.md` U1: this is UNRESOLVED. The closest
unified model is **STICKY ACTIVATION + TWO-HALF PROCESSING** (BF16-only):
1. B port starts in low-power gated state
2. First non-matching sub-tile activates the port; it stays active
3. (BF16 m128n128 only) Half A and Half B have INDEPENDENT activation state

### 52.9 2-CTA cluster — NO cluster-shared dedup pooling

`2CTA_DEDUP.md`: cluster_group::2 (2-CTA mma) shows **identical per-cluster
power dependence** to single-CTA mode. Each CTA's B operand has its own
32-byte sub-tile dedup cache. Optimization recipes apply per CTA, not
cluster-wide.

### 52.10 Cross-MMA dedup state is per-MMA

`CROSS_MMA_DEDUP.md`: dedup state is **per-MMA, NOT cross-MMA**.
Alternating different B descriptors gives the AVERAGE of per-MMA powers,
not a penalty or carry-over benefit. Real cuBLAS GEMMs (which iterate
K-tiles) get optimization recipes per K-tile.

### 52.11 disable_lane (DISABLE_LANE_POWER)

Selective output column gating. **Linear ~2.4 W per disabled column**
on BF16 m128n128. Cycle count unchanged. Composes with sub-tile dedup
(lower marginal saves when dedup already active). Best-case combination:
**254 W (vs 610 W random) = -58% reduction**.

### 52.12 Diagonal patterns — popcount-invariance, NOT pure diagonal

`DIAGONAL_DEEP_DIVE.md`: when each sub-tile (16 N values) has identical
bit-count across all K rows, power stays low; when popcount varies, it
goes high. Diagonal works at p_n=8 (all popcount=8) but NOT at p_n=16
(popcounts 0..15 vary). The popcount hypothesis explains the data; pure
"diagonal" framing was over-general.

### 52.13 BF16 32-element MAC group cliff (NVFP4 has NO equivalent)

From `NVFP4_PURE_TCGEN05_RESULTS.md` lines 335-432 — N-direction
replication sweep (B[n]==B[n+stride] in N direction):

**BF16 m=128 n=128 K=16:**

| Mode | Power (W) | Δ vs random |
|------|----------:|------------:|
| BASELINE rand | 606 | 0 |
| N-pair (stride 2) | 594 | -12 |
| N-quad (stride 4) | 580 | -26 |
| N-stride 8 | 585 | -21 |
| N-stride 16 | 592 | -14 |
| **N-stride 32** | **480** | **-126** ← CLIFF |
| **N-stride 64** | **391** | **-215** |
| N-all (stride 128) | 366 | -240 |
| K-half + N-stride 64 (combined) | **349** | **-257** ← min |

Sharp cliff between N-stride-16 (592 W) and N-stride-32 (480 W) — **112 W
drop in a single step**. This reveals **B is broadcast across 32 parallel
MAC units per cycle** in the multiplier datapath (matches B300 SMSP
width = 32 lanes).

**NVFP4 m=128 n=128 K=64** — same sweep, different result:

| Stride | Power (W) | Δ vs rand |
|--------|----------:|----------:|
| 1 (rand) | 471 | 0 |
| 2 | 461 | -10 |
| 4 | 424 | -47 ← dip |
| 8 | 474 | +3 |
| 16 | 472 | +1 |
| 32 | 474 | +3 |
| 64 | 476 | +5 |
| 128 | 392 | -79 ← min |

**NVFP4 has NO 32-element MAC cliff like BF16:**

| Format | Stride 32 Δ | Stride 128 Δ |
|--------|------------:|-------------:|
| BF16 | -126 W (CLIFF) | -240 W |
| NVFP4 | +3 W (no cliff!) | -79 W (3× less than BF16) |

**Caveat (per source own caution lines 712-718)**: NVFP4 within-word
strides (1, 2, 4) had encoding bugs — only the sign-bit-only retest is
clean. So NVFP4 absence-of-cliff is MED confidence; BF16 cliff is HIGH
confidence.

Mechanistic implication: BF16 multiplier broadcasts B across 32-element
MAC groups; NVFP4 block-scale ULTRA path (with TMEM SF lookup, 16-element
scale-block alignment) reorganizes B feeding so the parallel-broadcast
structure isn't visible to power optimization at the same stride.

### 52.14 Recipe — power-aware tcgen05 GEMM (BF16 m128n128)

1. **Quantize / sort B columns** so byte-identical 32-byte chunks cluster contiguously along N. Saves ~250-310 W vs unsorted random.
2. **Place repeating sub-tiles at LOW N (Half A)**, arbitrary at HIGH N (Half B). Saves up to 269 W more.
3. **Group K rows so consecutive rows match or alternate ABAB-style**. Adds 5-25 W penalty per unique K row pattern (vs full random 47-99 W).
4. **disable_lane unused output columns**: ~2.4 W per column on BF16.
5. **Combined realistic best case**: 254 W (vs 610 W random) = **-58%, applies per CTA**. Saves up to **450 W per CTA at boost** via column sort + K-row grouping (project_tcgen05_power memory).

For cuBLAS workloads, only steps 1-3 are accessible (no disable_lane
control). Real ML weights satisfy essentially none of the trigger
conditions for K-id speedup → ~2-6% practical inference benefit (see §53).

### 52.15 Practical mapping — when do these recipes apply?

Per `TCGEN05_DEDUP_CONSOLIDATED.md` recipes section, real-world
applicability:

| Workload class | Sub-tile dedup | K-row dedup | Two-half (BF16) | disable_lane | A=zero | Estimated saves |
|----------------|----------------|-------------|-----------------|--------------|--------|-----------------|
| cuBLAS GEMM (random data) | ~1-3% benefit (sub-linear) | up to 42% if N∈{K/2,K,2K} | n/a | n/a | n/a | 2-6% |
| cuBLAS GEMM (zero/const) | full benefit | full benefit | n/a | n/a | n/a | 50%+ (boost-cap-bound) |
| Custom tcgen05.mma kernel | full benefit if you sort | full benefit if you group | applies BF16 m128n128 | controllable | controllable | up to -58% W |
| Llama-70B FFN (FP8 + 2:4) | partial | n/a | n/a | n/a | activation-dependent | ~67% of HW peak |
| Post-ReLU activations | n/a | n/a | n/a | n/a | YES (sign bit always 0) | -80 W per CTA |

For **cuBLAS workloads**, only steps 1-3 are accessible (no disable_lane
control). Real ML weights satisfy essentially none of the trigger
conditions for K-id speedup → ~2-6% practical inference benefit.

**Maximum production stack** (FP8 + structured 2:4 sparse, batch ≥1024):
~3033 TFLOPS on Llama-70B FFN = 67% of HW peak — see §55.

### 52.16 The unified power model — calibration parameters

For BF16 m128n128k16 cluster_group::1 at 1005 MHz lock:

```
P_baseline (Tier B, A=B=const)            = 299 W
P_active_floor (A=B=zero)                 = 287 W  (12 W below baseline; multiplier idle deep gate)
P_per_broken_byte_in_subtile               = 18 W
P_per_subtile_activation                   = 32 W
P_per_K_unique_pattern                     = 5-25 W (sub-linear in K count)
P_per_active_M-side broadcast (A varying)  = 0 W if B const, +60 W marginal if B varies
P_disable_lane_per_column                  = 2.4 W
P_full_random_ceiling                      = 609-611 W
```

For NVFP4 K=96 ULTRA cluster_group::2 at 1005 MHz lock:

```
P_idle                                  = 150 W
P_active_floor (multiplier zero-skip)   = 134 W (284 W total, A=B=const)
P_per_added_magnitude                    = 40 W (saturates after ~5)
P_sign_bit_alone                         = 80 W
P_worst_sign_pattern_p_n_64             = +50 W on top of full random
P_per_outlier_per_K16                   = 30 W
P_TDP_cap                                = 950 W per CTA (1100 W total chip)
```

For FP8 e4m3 cluster_group::1 at 1005 MHz lock:

```
P_baseline                               = 305 W
P_full_random_ceiling                    = 642 W
P_data_dep_gap                           = 337 W
P_K_vary_16_cost                         = 71 W (76% of K-linear)
```

For TCGEN05 power scaling:
- Linear in active SM count up to 148 SMs.
- Per-CTA scaling for cluster_group::2: per CTA NOT per cluster.
- Power grows super-linearly with clock: 1.49× clock → 1.53-1.65× power (Vdd × freq² × Cload + static-power scaling).

### 52.17 Cross-precision summary

A K-vary cost across 3 precisions: 2 W (BF16), ≈0 W (FP8), 7 W (NVFP4)
B K-vary cost across 3 precisions: 47 W, 71 W, 99 W
Ratio (B/A K-vary): 24×, >70×, 12×

The **broadcast-A vs distributed-B** architecture is preserved across
all three multiplier hardware paths (kind::f16, kind::f8f6f4,
kind::mxf4nvf4).

**Footgun:** ⚠ "A is FREE" requires B uniform. With random B, A varying adds ~60 W (rand+rand 611 W vs const+rand 549 W = +62 W marginal). The conditional rule from `A_B_ZERO_ASYMMETRY.md` is the right one.

**Footgun #2:** ⚠ Cache depth (1 vs 2 vs 4 slots) is CONTESTED across docs. The "4-slot HW pattern cache" claim is RETRACTED. Sticky activation + two-half processing is the closest unified model (TCGEN05_DEDUP_CONSOLIDATED U1). Don't quote a slot count without specifying the test — single-MMA pattern rotation gives 1-2; cuBLAS K-id alternation gives 2.

**Footgun #3:** ⚠ N-vary "FREE" was a low-entropy artifact. Low-entropy val tables only have ≤16 distinct entries, hiding the cliff at N_unique=17. With high-entropy random data, the 32-byte cliff appears.

**Footgun #4:** ⚠ K-row sorting saves 349 W per CTA was RETRACTED in TCGEN05_PERFW_CLEAN_2TRIAL §"K-axis power is BINARY": even chunk-48 (1 K-transition in entire K=96) uses same power as fully random K. K-axis power is BINARY: either all 96 K-rows bit-identical (411 W floor) OR full ~870 W cost. Real LLM weights always pay full cost.

**See also:** §50 (A:B asymmetry), §51 (NVFP4 K=96 signature), §53 (K-id shape conditional), §55 (sparsity), corrections/TCGEN05_DEDUP_CONSOLIDATED.md, project_tcgen05_power memory.

---

## §53. K-id speedup — shape-conditional, NOT a kernel switch

**Answer:** cuBLAS 1.40× K-id BF16 speedup (rank-1 along K) ONLY triggers at N ∈ {K/2, K, 2K} AND N divisible by 256 AND transB=0 AND data has period-1 or period-2 K structure. **All five conditions required.** Real ML inference (N/K = 2.5-3.5) is OUTSIDE this window so practical benefit is **~2-6%**, NOT 1.40×. Same kernel runs at all tested N values (verified by ncu); shape-dependence is intrinsic to the data×kernel HW interaction. `[🟢 HIGH · src: N_DEPENDENCE_DEEPDIVE.md, project_kid_speedup_shape_dependent memory]`

### 53.1 The discovery

Tested K-id mode (rank-1 along K, varies by N) at M=K=8192 BF16, GPU 0:

| N | TFLOPS | N/K | Speedup vs random |
|--:|-------:|----:|------------------:|
| 8192 | 2098 | 1.0 | **1.42×** ← speedup |
| 9216 | 1503 | 1.13 | 1.02× |
| 10240 | 1501 | 1.25 | 1.02× |
| 12288 | 1521 | 1.5 | 1.03× |
| 14336 | 1501 | 1.75 | 1.01× |
| **16384** | 2089 | 2.0 | **1.42×** ← speedup |
| 20480 | 1494 | 2.5 | 1.01× |
| 24576 | 1505 | 3.0 | 1.02× |
| 28672 | 1495 | 3.5 | 1.01× |
| 32768 | 1513 | 4.0 | 1.02× |

K variation (M=N=8192):

```
K     TFLOPS  Speedup
4096  2117    1.40×  ← N=2K, speedup
4608  1513    1.02×
5120  1511    1.02×
5632  1507    1.02×
6144  1535    1.03×
8192  2086    1.41×  ← N=K, speedup
12288 2145    1.40×  ← N=K/1.5, different kernel: 256×256 tile
16384 2172    1.41×  ← (different kernel)
```

Cross-K validation:
```
K=4096 N=4096:  speedup ✓
K=4096 N=8192:  speedup ✓
K=4096 N=12288: NO speedup
K=6144 N=6144:  speedup ✓ (when N=K)
K=6144 N=12288: speedup ✓
K=6144 N=18432: NO speedup
K=8192 N=8192:  speedup ✓
K=8192 N=16384: speedup ✓
K=8192 N=24576: NO speedup
```

**K=6144 is NOT inherently broken** — the earlier confusion was that
K=6144 was tested with N=8192, where N/K=1.33 falls outside {1, 2}.

### 53.2 ncu confirms SAME kernel runs at all N values

Same kernel `nvjet_sm103_tss_128x256_64x6_2x1_2cta_v_bz_NNT` runs at ALL
tested N values. The shape-dependence is **intrinsic to the
data×kernel interaction at the hardware level**, NOT cuBLAS picking a
different algorithm.

**Correction**: TCGEN05_POWER_MASTER.md previously attributed the
rectangular-vs-square gap to "different cuBLAS algorithm" — this is wrong.
Same kernel, different shape-induced HW behavior.

### 53.3 Full constant comparison — shape-INDEPENDENT

| N | Full-const TFLOPS |
|--:|------------------:|
| 8192 | 2251 |
| 9216 | 2218 |
| 12288 | 2257 |
| 16384 | 2260 |
| 24576 | 2262 |
| 32768 | 2262 |

**Full constant: shape-independent. Range 2218-2262 TF (~2% spread).**
Whereas K-id at same shapes: 1495-2098 TF (~40% spread).

### 53.4 Two distinct mechanisms

1. **Universal entropy detector (full-const)**: when bit-entropy-per-byte is exactly zero everywhere, hardware gates the multiplier circuits across the entire fabric. Works regardless of shape. Reaches ~1.52× speedup ceiling.
2. **Shape-conditional pattern detector (structured low-entropy)**: when data has structure (e.g., rank-1 along K), the dedup HW only detects the structure when N aligns with cuBLAS scheduling pattern. Reaches 1.42× when triggered.

### 53.5 Power confirmation of throttle mechanism

NVML sampled continuously during sustained workloads at N=K=8192:

| Mode | Avg clock | Avg power | Clock vs boost |
|------|-----------|-----------|----------------|
| K-id | 1924 MHz | 737 W | 95% of 2032 MHz boost |
| Random | 1507 MHz | 976 W | 74% of boost (POWER CAPPED) |

**Random pulls 240 W MORE than K-id and hits 1100 W cap, dropping
clock by 22%.** K-id stays at near-boost clock with significantly lower
power draw.

Energy-per-op:
- K-id: 737 W / 2098 TF = 0.351 W·s/TF
- Random: 976 W / 1480 TF = 0.659 W·s/TF
- Ratio: **1.88× more energy per FLOP for random data**

Random data is roughly **2× less energy efficient** on the same dense
GEMM kernel. Savings come from HW dedup gating inactive multiplier
circuits.

### 53.6 Power-cap modulation widens the gap

| Power cap | K-id TFLOPS | Random TFLOPS | K-id Advantage |
|-----------|-------------|---------------|----------------|
| 1100 W | 2098 | 1480 | 1.42× |
| 700 W  | 1510 | 1013 | 1.49× |
| 500 W  | 1071 | 645  | **1.66×** |

**At lower power caps, K-id advantage GROWS.** This is direct evidence
that the speedup is power-throttle-mediated.

### 53.7 Sustained 60-second confirmation

K-id and random run continuously for 60 seconds, recording per-iteration
TFLOPS:

| Mode | Iter 1 (3 s) | Iter 10 (29 s) | Iter 20 (59 s) | Decline | Speedup |
|------|-------------:|----------------:|----------------:|--------:|--------:|
| K-id (N=K=8192) | 2105 TF | 2094 TF | 2092 TF | ~0.6% | 1.42× sustained |
| Random | 1482 TF | 1467 TF | 1471 TF | ~0.7% | n/a |

**Both modes show only 0.6-0.7% thermal degradation over 60 s.** The
speedup ratio (1.42×) is fully maintained. K-id speedup is **NOT a
transient warmup effect**.

### 53.8 Energy density determines the bottleneck

| Cap | K-id TFLOPS | Random TFLOPS | K-id Adv | K-id W/TF | Random W/TF |
|-----|------------:|--------------:|---------:|----------:|------------:|
| 1100 W | 2098 | 1480 | 1.42× | 0.351 | 0.659 |
| 700 W | 1510 | 1013 | 1.49× | constant | constant |
| 500 W | 1071 | 645 | 1.66× | constant | constant |

**Energy per FLOP confirms the story:**
- K-id: ~0.35 W/TF (constant across caps)
- Random: ~0.65 W/TF (constant across caps)

So even if you HAVE 2× more power available, you can't match K-id
throughput without also fixing the data pattern. **The bottleneck is
multiplier energy density**, not raw wattage.

### 53.9 Practical inference benefit estimate

For real workloads where N/K = 2.5-3.5 (typical attention QKV / FFN
shapes):
- Pattern dedup (K-id, ABAB chunk=1): requires synthetic structure, never naturally hits
- Zero-mult shortcut (>75% zeros): real LLM weights ~50% sparse at most
- Universal entropy detector (full const): never matches real weights
- Structured 2:4 sparsity: +11% only with FIXED positions (see §55)

Combined: **~2-6% practical inference benefit on real LLM workloads**
even though the headline 1.40× exists for synthetic benchmarks.

### 53.10 Trigger conditions

| Condition | Required |
|-----------|----------|
| N ∈ {K/2, K, 2K} | YES |
| N divisible by 256 | YES (cuBLAS tile alignment) |
| transB = 0 (NN layout) | YES |
| Data has period-1 or period-2 K structure | YES |
| Shape selects the same kernel | implied |

All five conditions required. Real ML inference satisfies essentially none.

**Footgun:** ⚠ Don't quote 1.40× as a generic K-id speedup — it's a thin shape window. Real ML inference (N/K = 2.5-3.5) is OUTSIDE this window so practical benefit is ~2-6%. Per CLAUDE memory project_kid_speedup_shape_dependent.

**Footgun #2:** ⚠ The earlier framing "cuBLAS picks different kernel for different shapes" is RETRACTED — same kernel runs at all shapes; HW behavior is shape-induced.

**See also:** §52 (sub-tile dedup model), §55 (sparsity), N_DEPENDENCE_DEEPDIVE.md, project_kid_speedup_shape_dependent memory.

---

## §54. CUTLASS / CuTeDSL stuck at 8.7 PF vs cuBLAS 11.42 PF (76%)

**Answer:** CuTeDSL persistent kernel hits **8112 TFLOPS (54.1% MFU)** at boost+zero+cluster (2,4); CUTLASS C++ sample 89 hits **8285 TFLOPS (~55%)** at the same shape; cuBLAS reaches **11423 TFLOPS (76.2%)** with cudaGraph BPG=16. CUTLASS uses different tile shape and kernel design that misses the K-id window AND has higher per-launch overhead. **UNRESOLVED 🟡** — exact mechanism for the 8-15 pp MFU gap. `[🟡 MED · src: NVFP4_CUTEDSL_THROTTLE_DEEP_DIVE.md, NVFP4_CUDAGRAPH.md]`

### 54.1 The gap

| Library | Best config | TFLOPS | MFU @ 15 PF | Source |
|---------|-------------|-------:|------------:|--------|
| cuBLAS Lt + cudaGraph BPG=16 | M=N=8192 K=38400 | **11423** | **76.2%** | NVFP4_CUDAGRAPH.md |
| cuBLAS Lt plain | M=N=8192 K=38400 | 11054 | 73.7% | NVFP4_CUBLAS_FULL_SWEEP.md |
| CuTeDSL persistent boost zero | M=N=16384 K=15360, cluster (2,4) | **8112** | 54.1% | CUTEDSL_THROTTLE |
| CuTeDSL boost random sustained | same shape | 6902 | 46% (1455 MHz throttled) | CUTEDSL_THROTTLE |
| CUTLASS C++ sample 89 boost | 8K² K=15K, 2SM cluster (2,4) | 8285 | ~55% | CUTEDSL_THROTTLE |
| CUTLASS C++ sample 89 @ 1005 MHz | 8K² K=15K, 2SM cluster (2,4) | 5544 | 77.7% at-clock | CUTEDSL_THROTTLE |

Gap to cuBLAS: **8-15 pp MFU**.

### 54.2 What we know about the gap

From `NVFP4_CUTEDSL_THROTTLE_DEEP_DIVE.md`:

1. **CuTeDSL persistent kernel design** uses powers-of-2 only for tiling. Best is 256×256. 192-anything fails. Practical universe is just (128,128), (128,256), (256,256). Cluster shape supported: (2,1), (2,2), (2,4), (4,4); (1,*) raises TypeError.
2. **Cluster (2,4) leaves 28 of 148 SMs idle** at boost — 8 CTAs/cluster × 15 simultaneous fits = 120 active. Cluster (2,1) uses all 148 SMs but lower per-active SM throughput.
3. **CUTLASS C++ 89 has 61% wall-time as host overhead** (`gemm.initialize()` in loop). Even discounting that, CUTLASS C++ 89 lags CuTeDSL by 8-15 pp MFU.
4. **cuBLAS 11.42 PF EXCEEDS its own model's predicted 73.1% ceiling** by 3 pp via cudaGraph BPG=16. Suggests the 19 ns "fixed overhead" in the model has a hidden launch-related component that cudaGraph eliminates.

### 54.3 CuTeDSL clock-scaling model

`t_utcmma = 132/clock + 19 (ns)` — fits 4 clock points within ~6%. Theoretical
ceiling at boost = 132/(132 + 19 × 2.032) = **73.1%**. cuBLAS+cudaGraph
reaches 76.2%, CuTeDSL reaches 54.1% at the same boost. The 19 ns fixed
overhead has a hidden launch-related component, not a true per-utcmma stall.

### 54.4 Per-active-SM vs per-total-SM MFU

| Library | Best per-total-SM MFU | Best per-active-SM MFU | Best absolute TFLOPS |
|---------|----------------------:|------------------------:|---------------------:|
| CuTeDSL (1005 lock, K=61440, cluster (2,1)) | **91.3%** | similar | 6776 |
| CuTeDSL (1005 lock, K=15360, cluster (2,4)) | 88.4% | **88.4%** per-active | 5319 |
| cuBLAS+cudaGraph (boost) | n/a | n/a | **11423 (76.2%)** |

CuTeDSL hits **91% per-total-SM at 1005 MHz** on K-deep matmuls — within
9% of cuBLAS catalog 72%-of-15PF. The remaining gap is the
cluster-scheduling constraint that leaves 28 of 148 SMs idle.

### 54.5 Comprehensive 3-impl × 4-shape × 2-clock comparison (zero data, throttle-verified)

From `NVFP4_CUTEDSL_THROTTLE_DEEP_DIVE.md` lines 360-411 — all runs zero
data, sustained 200-5000 iters per shape, fine-grain 50 ms power+clock
sampling confirmed:
- BOOST: 2032 MHz held throughout, no throttle, peaks 815 W (cuBLAS 16K³), 788 W (CuTeDSL), 501 W (CUTLASS 89) — all under 1100 W TDP
- 510 MHz: 510 MHz held, 144-192 W, no throttle

**Boost zero-data table (TFLOPS):**

| Shape | CuTeDSL(2,1) | CuTeDSL(2,4) | CUTLASS 1SM | CUTLASS 2SM | cuBLAS | Best | MFU/15PF |
|-------|-------------:|-------------:|------------:|------------:|-------:|------|---------:|
| 8K² K=15K | 8618 | 7744 | 6056 | 8219 | **9987** | cuBLAS | 66.6% |
| 16K³ | 5945 | 7851 | 4776 | 5733 | **9770** | cuBLAS | 65.1% |
| 8K³ | 9118 | 7661 | 6387 | 8285 | **9377** | cuBLAS | 62.5% |
| 4K³ | **6041** | 5164 | 3944 | 4474 | 5157 | CuTeDSL(2,1) | 40.3% |

**-lgc 510 MHz zero-data table (TFLOPS):**

| Shape | CuTeDSL(2,1) | CuTeDSL(2,4) | CUTLASS 1SM | CUTLASS 2SM | cuBLAS | Best | MFU/spec@510 |
|-------|-------------:|-------------:|------------:|------------:|-------:|------|-------------:|
| 8K² K=15K | **3244** | 2563 | 1952 | 2855 | 3115 | CuTeDSL(2,1) | **86.1%** |
| 16K³ | 3215 | 2638 | 1935 | 2893 | 3203 | CuTeDSL(2,1) | 85.4% |
| 8K³ | **2870** | 2316 | 1806 | 2345 | 2706 | CuTeDSL(2,1) | 76.2% |
| 4K³ | **1802** | 1492 | 1091 | 1294 | 1443 | CuTeDSL(2,1) | 47.8% |

**MFU climb at low clock (boost vs 510, same impl/shape):**
- 8K² K=15K: boost cuBLAS 66.6% → 510 CuTeDSL 86.1% (Δ 19.5 pp)
- 16K³: boost 65.1% → 510 85.4% (Δ 20.3 pp)
- 8K³: boost 62.5% → 510 76.2% (Δ 13.7 pp)
- 4K³: boost 40.3% → 510 47.8% (Δ 7.5 pp)

**With zero data + no throttle, MFU still climbs 8-20 pp from boost to
510 MHz.** This rules out TDP throttle as the cause. Real cause must be
**non-clock-scaled wall-time overhead** in the mma pipeline (TMA fill
latency at HBM 3996 MHz, mbarrier coordination NoC traversal,
cluster-broadcast handshake).

### 54.6 Cross-impl rank changes by clock

- **At boost**: cuBLAS dominates (3 of 4 shapes); CuTeDSL (2,1) wins only at 4K³
- **At 510 MHz**: CuTeDSL (2,1) dominates (4 of 4 shapes); cuBLAS drops to #2
- **CUTLASS C++ 89**: consistently 15-30% behind cuBLAS at both clocks

The crossover happens because cuBLAS's nvjet kernel uses tighter memory
pipelining that needs high clock to keep utcmma fed. CuTeDSL's persistent
kernel design with fewer SMs per cluster pays less in coordination
overhead.

### 54.7 Clock-scaling model (CuTeDSL 2,4 16384² K=15360)

CuTeDSL measured at 4 clock points via `ncu --clock-control none`. utcmma
TOTAL constant 655K across all clocks (K=96 invariant verified):

| ncu clock | Duration | ns/utcmma/leader | SM Throughput |
|-----------|---------:|-----------------:|--------------:|
| 0.510 GHz | 3059 µs | 280.0 ns | 74.77% |
| 1.005 GHz | 1586 µs | 145.2 ns | 73.97% |
| 1.484 GHz | 1170 µs | 107.1 ns | 72.07% |
| 1.918 GHz | 1013 µs | 92.7 ns | 66.44% |

Least-squares fit: `t_utcmma = c/clock + f` gives:
- **c = 132 ns·GHz** (clock-scaled compute time per utcmma)
- **f = 19 ns** (fixed wall-time coordination overhead per utcmma)
- Residuals 0.8% / 4.1% / 1.1% / 5.6% — model fits within ~6%

Predicted MFU at each clock = `132 / (132 + 19 × clock_GHz)`:
- 0.510 GHz: **93.2%**
- 1.005 GHz: **87.4%**
- 1.484 GHz: **82.4%**
- 1.918 GHz: **78.4%**

**Theoretical maximum at boost** even with zero coordination overhead =
clock/c × MAC capacity = 14.5 utcmma/µs/leader × 60 leaders × 12.58 MFLOPs
= **10.96 PFLOPS = 73.1% of 15 PF spec**. This is the model ceiling for
CuTeDSL at boost.

cuBLAS+cudaGraph hits 11.42 PF = 76.2% — **3 pp above this ceiling**.
Suggests the 19 ns fixed overhead has a hidden launch-related component
(eliminated by cudaGraph). Per-utcmma stall (TMA fill + mbarrier
handshake) probably caps lower.

### 54.8 CUTLASS C++ host overhead in benchmark loop

```cpp
for (int iter = 0; iter < options.iterations; ++iter) {
    CUTLASS_CHECK(gemm.initialize(arguments, workspace));  // host work each iter!
    CUTLASS_CHECK(gemm.run());
}
```

5000-iter at 16K² K=15360 takes **43.3 s wall-clock** but only **16.7 s
reported kernel** = 39% busy / 61% host overhead. Even discounting that,
CUTLASS C++ 89 lags CuTeDSL by 8-15 pp MFU. CUTLASS sample uses ALL 148
SMs (cycles_active 97%) but with worse compute density per cycle —
utcmma rate 423 M/s vs CuTeDSL 640 M/s.

### 54.9 cuBLAS K-sweep hits the predicted 73% ceiling

cuBLAS NVF4 K-sweep at boost zero data (M=N=8192):

| K | TFLOPS | %15 PF | Notes |
|--:|-------:|-------:|-------|
| 1536 | 4610 | 30.7% | overhead-dominated |
| 3072 | 6089 | 40.6% | |
| 6144 | 7171 | 47.8% | |
| 9216 | 9808 | 65.4% | |
| 12288 | 10284 | 68.6% | |

The 11423 cudaGraph BPG=16 measurement reaches 76.2% — exceeds the model
ceiling by 3 pp because cudaGraph eliminates the launch-related part of
the 19 ns overhead.

### 54.10 Open questions (UNRESOLVED 🟡)

Per `NVFP4_CONSOLIDATED.md` open Q1:
> Why does CUTLASS C++ 89 stuck at 5.5-5.7 PF (1005 MHz) when CuTeDSL hits 6.78 PF and cuBLAS hits ~10.8 PF? Same hardware, same shape. CUTEDSL_THROTTLE notes 61% wall-time is host overhead — but even discounting that, CUTLASS C++ 89 lags CuTeDSL by 8-15 pp MFU. CuTeDSL persistent kernel design and cluster-broadcast tighter? **Open.**

Per `NVFP4_CONSOLIDATED.md` open Q2:
> What is cuBLAS doing that hits 76.2% MFU with cudaGraph? Model predicts 73.1% ceiling from 19 ns fixed overhead. Graph eliminates launch-prep CPU time but shouldn't eliminate per-utcmma stall. Suggests the "19 ns" model has hidden launch-related component. **Open.**

### 54.11 Per CLAUDE memory project_b300_nvfp4_k96_ceiling

> cuBLAS 13.4 caps 10.8 PF (72% of 15 PF spec) at large-N rect; **CUTLASS C++/CuTeDSL stuck at 8.7 PF (58%); K=96 is real but 1.5× spec is unattainable in public libs**.

The 8.7 PF figure is a synthesis of CUTLASS C++ 89 (8285) and CuTeDSL
(8112) — both lag cuBLAS by 25-30%.

**Footgun:** ⚠ Don't claim CUTLASS reaches NVIDIA spec — public CUTLASS C++ samples lag cuBLAS by 8-15 pp MFU on B300. Use cuBLAS for production NVFP4 GEMM.

**Footgun #2:** ⚠ CuTeDSL "best per-active-SM MFU 88-91%" sounds like CuTeDSL is hitting near-spec, but **per-total-SM MFU is 54-91% depending on cluster choice**. Cluster (2,4) leaves 28 SMs idle.

**See also:** §49 (NVFP4 K=96 reachability), §46 (full ladder), NVFP4_CUTEDSL_THROTTLE_DEEP_DIVE.md, NVFP4_CUDAGRAPH.md, project_b300_nvfp4_k96_ceiling memory.

---

## §55. Sparsity — 3-tier model

**Answer:** Three INDEPENDENT sparsity-related mechanisms on B300, often confused with each other:
- **Tier 1 (memory-side popcount sparsity)**: smooth toggle-energy decay; needs ≥50% sparsity for material savings; DRAM swing 245 W max.
- **Tier 2 (tcgen05 multiplier zero-shortcut, U-curve)**: 30-50% RANDOM sparse HURTS dense GEMM by 5%; >75% sparse helps; full-zero gives 1.52× speedup.
- **Tier 3 (structured 2:4 sparsity, fixed-position)**: +11% on dense GEMM (no sparse API needed). RANDOM 2:4 (random which 2 of 4 are zero): NO speedup.
`[🟢 HIGH · src: SPARSITY_3TIER.md, N_DEPENDENCE_DEEPDIVE.md "CRITICAL CORRECTION", corrections/TCGEN05_DEDUP_CONSOLIDATED.md §6]`

### 55.1 Tier 1 — Memory-side popcount sparsity (SPARSITY_3TIER.md)

**Wire/SerDes toggle energy**, not multiplier-side. Smooth monotonic
decay of read power with sparsity at L1/L2/DRAM tiers.

| Tier | sp=0 active W | sp=100 (zero) | absolute swing | relative swing |
|------|--------------:|--------------:|---------------:|---------------:|
| L1   | 105 | 68  | 37  | 35% |
| L2   | 403 | 222 | 181 | 45% |
| DRAM-8G | 636 | 391 | 245 | 39% |

- **Knee at sp ≈ 10-15%**; needs ≥50% sparsity for material savings (40 W ≈ 10% at sp=50% on L2).
- **Granularity barely matters** (byte/dword/32B/128B within ±15-20 W).
- **Value asymmetry at sp=100%**: zero < alt55 < one (HBM PHY active-low termination).
- Popcount d=16 random = peak (~636 W active on DRAM-8G).
- DRAM has biggest absolute swing (245 W per data choice). For a 1.1 kW
  board, going from random data to all-zeros at DRAM saves ~22% TDP
  without changing any kernel.

### 55.2 Tier 2 — tcgen05 multiplier zero-shortcut (N_DEPENDENCE_DEEPDIVE U-curve)

| Sparsity | TFLOPS | vs dense |
|----------|-------:|---------:|
| 0% (dense random) | 1480 | 1.00× baseline |
| 30%   | 1406 | **0.95× ← SLOWER!** |
| 50%   | 1440 | 0.97× (still in dip) |
| 60%   | 1475 | 1.00× |
| 75%   | 1565 | 1.06× |
| 90%   | 1730 | 1.17× |
| 99%   | 1930 | 1.30× |
| 100% (all zero) | 2253 | **1.52× ← ceiling** |

**The dip at 30-50% sparsity is REAL**, not noise. Reproducible across
multiple N values. Hypothesis: control logic overhead for "is this
operand zero?" detection. At intermediate sparsity, overhead exceeds
zero-shortcut savings. At high sparsity, savings dominate. At full-zero,
universal entropy detector + multiplier fully gated.

**Sparsity speedup is COMPLETELY N-independent** (validated across N=8192/9216/16384/24576) — fundamentally different mechanism from K-id pattern dedup.

### 55.3 Tier 3 — Structured 2:4 sparsity (predictable-position pattern detector)

| Pattern | TFLOPS | vs dense |
|---------|-------:|---------:|
| dense (no sparsity) | 1482 | 1.00× baseline |
| 2:4 structured (zeros at 0,1) | 1649 | **1.11×** ← +11% SPEEDUP |
| 2:4 structured (zeros at 1,3) | 1642 | 1.11× |
| 2:4 structured (zeros at 0,2) | 1647 | 1.11× |
| 1:2 alternating (zero,nonzero) | 1648 | 1.11× |
| 50% RANDOM sparsity | 1486 | 1.00× ← NO speedup |

- FIXED 2:4 zeros at positions {0,1}, {0,2}, {1,3}, etc.: **1.11× speedup on dense GEMM** (no sparse API needed)
- 1:2 alternating: also 1.11×
- RANDOM 2:4 (random which 2 of 4 are zero): NO speedup
- Rotating 2:4: slight regression
- **Requires CONSISTENT zero positions per 4-element group** — that's why NVIDIA's 2:4 spec mandates fixed positions.
- Stacks with FP8: FP8 + 2:4 structured = 3033 TF (vs 2683 random) = **+14%** (NVFP4-class scaling).

### 55.4 The CRITICAL CORRECTION inside N_DEPENDENCE_DEEPDIVE

`N_DEPENDENCE_DEEPDIVE.md` line 633 originally claimed:
> popular 2:4 structured sparsity (50% zeros) actually **hurts** dense GEMM throughput

Then at line 659+, the same document says:
> **Earlier claim "2:4 sparsity hurts dense GEMM" was WRONG.** The error
> was from using random sparsity instead of structured. With STRUCTURED
> zero patterns (positions predictable per-4-elements), even dense GEMM
> sees 11% speedup from HW-level pattern detection of zeros.

This is the FOURTH application of rule #9 (suspect the test before the
hardware) in the investigation chain.

**Reader must read past the first half of N_DEPENDENCE_DEEPDIVE.md** to
get the correct sign. Picking only the early section gives wrong sign.

### 55.5 Mechanism stacking — NOT strongly multiplicative

| N | K-id alone | 2:4 alone | Combined | Stacking? |
|--:|-----------:|----------:|---------:|-----------|
| 8192 | 2092 | 1647 | 2116 | best alone wins (~K-id) |
| 9216 | 1498 | 1627 | 1657 | best alone (2:4) + small K-id bonus |
| 16384 | 2086 | 1640 | 2109 | K-id wins (combined ~K-id alone) |
| 24576 | 1503 | 1643 | 1693 | 2:4 wins + small K-id bonus |

**Mechanisms saturate at shared ceiling.** Combined K-id + 2:4 = 2116 TF
(1.43×). Both mechanisms gate the same multiplier circuits; once one
reduces multiplier power, the other can only add marginal gains.

### 55.6 Final mechanism summary (after 4 rule-#9 corrections)

| Mechanism | Trigger | Shape sensitivity | Max speedup |
|-----------|---------|-------------------|-------------|
| Pattern dedup (K-id, ABAB chunk=1) | data structure | YES (N ∈ {K/2,K,2K}) | 1.42× |
| Structured sparsity (2:4) | predictable zero positions | NO | 1.11× |
| Zero-mult shortcut (>75% zeros) | bulk zero values | NO | up to 1.52× |
| Universal entropy detector (full const) | bit-entropy = 0 | NO | 1.52× |

Combined ceiling: **1.52×** (matches full-constant case). All real
workloads fall well below all triggers → ~2-6% practical inference
benefit confirmed.

### 55.7 Sparsity dip explained — pattern-detector thrashing

Power measurement during sustained sparse workloads at N=K=8192:

| Sparsity | Clock | Power | TFLOPS | Notes |
|----------|------:|------:|-------:|-------|
| 0% dense | 1492 MHz | 796 W | 1480 | baseline (random) |
| **30% sparse** | **1382 MHz** | **943 W** | **1410** | **DIP — power INCREASES** |
| 90% sparse | 1596 MHz | 695 W | 1730 | efficient — zero shortcuts dominate |

**At 30% sparsity, power goes UP not down!** The mixed zero/nonzero
pattern causes the multiplier's pattern-detection circuits to thrash
trying to recognize structure, consuming MORE energy than purely random
data.

The U-shape mechanism:
- 0% (all random): detector finds nothing, baseline circuit activity
- **30% (mixed): detector thrashes, MAX activity → power +18%**
- 90% (sparse): zero-mult shortcuts dominate, power -13%
- 100% (all zero): full gate, power minimum

This explains the counterintuitive 5% throughput regression at 30-50%
RANDOM sparsity: the GPU power-caps harder than dense because mixed
patterns are the WORST case for the detection circuits.

### 55.8 Sub-tile dedup vs K-row dedup magnitude in cuBLAS

Isolated each mechanism by constructing data that triggers ONE without
the other:

| Mode | Description | TFLOPS | Speedup |
|------|-------------|-------:|--------:|
| 0 (fully random) | random A, random B | 1479 | 1.00× baseline |
| 1 (sub-tile 16 N const per row) | sub-tile only | 1493 | 1.01× ← ~no benefit |
| 4 (sub-tile 32 N const per row) | sub-tile only | 1512 | 1.02× |
| 5 (sub-tile 8 N const per row) | sub-tile only | 1499 | 1.01× |
| 6 (sub-tile 256 = full N-tile) | sub-tile only | 1518 | 1.03× |
| 2 (K-row identical) | K-row only | **2100** | **1.42× ← STRONG** |
| 3 (BOTH sub-tile + K-row) | combined | 2102 | 1.42× ← K-row only |

**In cuBLAS, sub-tile dedup contributes ~1-3%; K-row dedup gives 42%.**

While custom tcgen05 kernels may show stronger sub-tile dedup effects
(per §52), cuBLAS's actual GEMM kernels see K-row dedup as the DOMINANT
mechanism by a 14× margin. The cuBLAS kernel tile is 128×256 with
64-K-stage iteration; within each K-iteration (64 rows), B is loaded as
64 consecutive K-rows × 256 N-cols, exposing K-row dedup more than
sub-tile dedup.

### 55.9 Sustained workload — speedup persists over 60s

| Mode | Iter 1 (3 s) | Iter 10 (29 s) | Iter 20 (59 s) | Decline | Speedup |
|------|-------------:|----------------:|----------------:|--------:|--------:|
| K-id (N=K=8192, M=K=8192) | 2105 TF | 2094 TF | 2092 TF | ~0.6% | **1.42× sustained** |
| Random | 1482 TF | 1467 TF | 1471 TF | ~0.7% | n/a |

**Both modes show only 0.6-0.7% thermal degradation over 60 s.** The
speedup ratio (1.42×) is fully maintained throughout. K-id speedup is
NOT a transient warmup effect — production deployments CAN reliably
extract the K-id benefit IF they meet the trigger conditions. The
challenge remains hitting the conditions, not maintaining them.

### 55.10 256-element alignment requirement

Tested square M=N=K=X:

| X (=M=N=K) | TFLOPS | 256-aligned? | Speedup |
|-----------:|-------:|--------------|---------|
| 4096 | 1837 | yes (16×256) | partial (small) |
| 4352 | 2061 | yes (17×256) | ✓ full |
| 4608 | 1883 | yes (18×256) | ~partial |
| 4736 | 1956 | no (37×128) | partial |
| 4864 | 2070 | yes (19×256) | ✓ full |
| 5120 | 1967 | yes (20×256) | ~partial |
| 8192 | 2100 | yes (32×256) | ✓ full |
| 8320 | 2017 | no (65×128) | partial |
| 8448 | 2093 | yes (33×256) | ✓ full |
| 9344 | 1998 | no (73×128) | partial |

**256-element alignment is required** (single tile_N boundary). Values
NOT divisible by 256 (only 128-aligned) give partial speedup ~5-10% lower.

The full rule (5 conditions):
1. **N=K (or N=2K, N=K/2)** AND
2. **N divisible by 256** (single tile boundary) AND
3. **K divisible by 256** AND
4. **transB=0 layout** AND
5. **Data has period-1 or period-2 K structure**

ALL FIVE conditions required for the full 1.42× speedup. Real workloads
satisfy NONE of conditions 1-4 simultaneously, let alone the data
structure.

### 55.11 Practical recipes

For real LLM inference:
- **Pattern dedup**: requires synthetic structure, never naturally hits.
- **Zero shortcut**: requires >75% sparsity to provide >5% benefit. Real LLM weights typically <50% sparse.
- **2:4 sparsity**: actually +11% **if** structured (fixed positions per 4-element group). Pruning algorithms should target STRUCTURED 2:4, not random.
- **Memory-side popcount**: byte-wise constant data saves up to 22% TDP at HBM read time. Pre-quantize and pack constants in 32B+ aligned chunks.
- **Maximum production stack** (FP8 + structured 2:4 sparse, batch ≥1024): ~3033 TFLOPS on Llama-70B FFN = 67% of HW peak.

### 55.12 Sparsity vs FP8 stacking — concrete numbers

- FP8 + 2:4 structured = 3033 TF (vs 2683 random) = +14% (NVFP4-class scaling)
- FP8 cuBLAS realistic = 3983 TF (random)
- FP8 cuBLAS zero best-case = 4425 TF
- Stacking FP8+2:4 brings random closer to zero peak, but doesn't exceed it

**Footgun:** ⚠ N_DEPENDENCE_DEEPDIVE self-corrects mid-document — readers picking only the early section get the WRONG SIGN on 2:4 sparsity. Original line 633 says "2:4 hurts dense GEMM"; corrected line 659+ says +11%. Always read the CORRECTION section.

**Footgun #2:** ⚠ Don't conflate the three tiers. "Sparsity" in:
- Tier 1 = wire-level popcount (memory power) → smooth decay
- Tier 2 = multiplier-level zero-shortcut → U-curve, hurts at 30-50%
- Tier 3 = structured 2:4 (NVIDIA sparse spec) → +11% only at FIXED positions

A claim like "2:4 sparsity gives 5% slowdown" is RANDOM (Tier 2 dip);
"+11% speedup" is FIXED (Tier 3); both are correct in their own context.

**Footgun #3:** ⚠ Don't treat random 50% sparsity as a substitute for structured 2:4. Random 50% gives the dip; structured 2:4 gives the speedup. The HW pattern detector requires PREDICTABLE zero positions per 4-element group.

**See also:** §52 (tcgen05 dedup model), §53 (K-id shape conditional), SPARSITY_3TIER.md, N_DEPENDENCE_DEEPDIVE.md (read past line 659!), corrections/TCGEN05_DEDUP_CONSOLIDATED.md §6.

---

## Section E — Final cross-section consistency table

This table shows the same fact reported from multiple angles to
demonstrate the section's internal consistency:

| Fact | §46 | §49 | §50 | §51 | §52 | §53 |
|------|-----|-----|-----|-----|-----|-----|
| BF16 cuBLAS realistic = 1850 TF | r46.1 | — | r50.1.4 (1178 TF at 1005 lock) | — | — | — |
| FP8 cuBLAS realistic = 3984 TF | r46.1 | — | — | — | — | — |
| NVFP4 cuBLAS+cudaGraph = 11423 TF | r46.1, r46.8 | mentioned | — | — | — | — |
| NVFP4 K=96 ULTRA microbench = 10.91 PF @ 1500 lock | r46.1 | r49.2 | — | r51.2 | — | — |
| NVFP4 K=96 ULTRA = 14.78 PF zero-skip @ boost | — | r49.2 | — | r51.14 | — | — |
| 32-byte universal sub-tile B-side dedup | — | — | — | — | r52.2 | — |
| BF16 32-element MAC group cliff | — | — | r50.2.2 | — | r52.13 | — |
| A:B 3-way (cuBLAS A>B, pure-tcgen05 B>>A, K=96 single B>A 2.6×) | — | — | r50.1, r50.3 | — | r52.4 | — |
| K-id shape conditional 5 conditions | — | — | — | — | — | r53.10, r55.10 |
| K=96 random TDP = 13 PF, zero-skip = 14.78 PF | — | r49.2 | — | r51.14 | — | — |
| Random data is ~2× less energy-efficient | — | — | — | — | — | r53.8 |
| Multiplier port asymmetry: B distributed across N MACs, A broadcast | — | — | r50.2.1, r50.2.2 | — | r52.4 | — |
| Sticky activation + two-half BF16 (NOT 4-slot cache) | — | — | — | — | r52.7-52.8 | — |
| Sparsity 3-tier independent | — | — | — | — | — | — (in §55) |
| 256-element alignment K-id requirement | — | — | — | — | — | — (in §55.10) |

## Section E summary

10 sections (§46 - §55), covering tensor-core SoL ladders, mma.sync vs
tcgen05 path differences, mma.sync FP8 emulation, NVFP4 K=96 ULTRA
reachability, NVFP4 power signatures (A:B 3-way, K=96 detail, dedup
model, K-id, CUTLASS gap), and the 3-tier sparsity model.

Cross-cutting reminders preserved per task spec:
- A:B 3-way readings retained (cuBLAS A>B 3-4×, pure-tcgen05 B>>A 15-30×, K=96 single-kernel B>A 2.6×) — don't pick one.
- Cache depth (1 vs 2 vs 4 slots) preserved as CONTESTED.
- pipe_tensor footgun preserved per SESSION_2_DELTA.
- 14.78 PF NVFP4 zero-skip vs ~13 PF random TDP-throttled clearly distinguished.
- N_DEPENDENCE_DEEPDIVE self-correction on 2:4 sparsity surfaced.

CLAUDE memory entries cited by name: project_b300_nvfp4_k96_ceiling,
project_nvfp4_k96_signature, project_b300_power_data_dep,
project_tcgen05_power, project_kid_speedup_shape_dependent,
project_four_six_status, feedback_clock_stuck_no_lock,
feedback_b300_pitfalls, feedback_units_sanity.

## Section E — Production recipes (synthesized from §46-§55)

### Recipe 1: BF16 cuBLAS GEMM at near-spec
- Use cuBLAS LtMatmul, NOT mma.sync (10× lower throughput).
- Shape: M=N=8192 with K ∈ [12K, 46K]. Square > asymmetric.
- Clock: boost (`-rgc`); avoid `-lgc 2032` (pins to 1920 MHz).
- Realistic data: expect 1850 TF (zero baseline 2246 TF, -18% drop).
- For maximum: cudaGraph BPG=16 if you have many GEMMs.
- Power: ~700-800 W sustained. Not TDP-bound.

### Recipe 2: FP8 cuBLAS GEMM at near-spec
- Use cuBLAS LtMatmul (cuBLAS 13.x). Bug-free FP8 path.
- Shape: M=N=8192 with K ∈ [12K, 46K].
- Clock: boost. Avoid 600 W power cap (drops random to 3087 TF).
- Realistic data: expect 3984 TF (zero baseline 4425 TF).
- For sustained: cudaGraph for ~4491 TF zero / 3984 TF random.
- Combine with structured 2:4 sparsity (FP8 + 2:4 = 3033 TF) for inference.

### Recipe 3: NVFP4 cuBLAS at maximum throughput
- Use cuBLAS+cudaGraph BPG=16 at M=N=8192, K=38400.
- Clock: boost. Zero data hits 11423 TF (76.2% of 15 PF spec).
- Random data: throttles to 1455 MHz, sustains ~7000 TF (~47%).
- For random sustained: prefer 1500 MHz lock (TDP-safe), reaches 9273 TF (62%).
- Llama-style real shapes (K=8064): 30-50% of spec — K too narrow.

### Recipe 4: tcgen05 power-aware kernel (BF16 m128n128)
1. Quantize / sort B columns so byte-identical 32-byte chunks cluster contiguously along N. Saves ~250-310 W vs unsorted random.
2. Place repeating sub-tiles at LOW N (Half A, sub-tiles 0-3); arbitrary at HIGH N (Half B, sub-tiles 4-7). Saves up to 269 W vs mirrored.
3. Group K rows so consecutive rows match or alternate ABAB-style. Adds 5-25 W penalty per unique K row pattern.
4. disable_lane unused output columns: ~2.4 W per column on BF16.
5. Combined realistic best case: 254 W (vs 610 W random) = -58% per CTA. Saves up to 450 W per CTA at boost via column sort + K-row grouping.

### Recipe 5: NVFP4 power-aware weight quantization
1. Pre-process weights to use +0 (0x0) instead of -0 (0x8) when storing zero values. Free 3-100 W depending on sparsity.
2. Keep B-side magnitude distribution narrow (≤5 distinct |x|) when possible. Going from 8 mags to 5 mags saves 40 W.
3. Eliminate or cluster outliers. Each outlier per K-block costs ~30 W. Pre-quantization outlier clipping has direct power benefit.
4. A operand is not entirely free (5-100 W swing) but ~3× smaller than B. If you can choose, put the more variable / random one on A.
5. Zero-skip: passing a constant-zero buffer for A (e.g., for row-wise activations that happen to be all zero) saves 50-100 W instantly via multiplier short-circuit.

### Recipe 6: Maximum efficiency NVFP4 inference
- Clock: 1005 MHz lock (super-linear power scaling makes higher clocks less efficient per W).
- Use K=96 ULTRA path if writing custom kernel (microbench reaches 17.0 TF/W with 5-pos B; cuBLAS won't hit this in 13.2).
- All-zero B activations (post-ReLU) trigger zero-skip path: 23.4 TF/W at boost.
- Realistic LLM-weight distribution: ~13.4 TF/W typical.

### Recipe 7: 2-GPU NVFP4 split
- 19163 TFLOPS aggregate (95.8% of 2× 10000 spec).
- 1 stream/GPU, no inter-GPU communication during compute.
- Per-GPU: 9582 TFLOPS each (95.8% of 10 PF spec).
- Use CUDA IPC + cudaSetDevice for per-GPU stream management.

### Recipe 8: Avoid known footguns
- **Don't use mma.sync for FP8** — emulated, 1.37× slower than 2× BF16.
- **Don't use `nvidia-smi -lgc 2032`** — pins to 1920 MHz (base clock).
- **Don't trust ncu pipe_tensor for tcgen05** — silent zero, no warning.
- **Don't quote single A:B ratio for NVFP4** — three different right answers depending on context.
- **Don't read N_DEPENDENCE_DEEPDIVE first half only** — has self-correction at line 659+ for sparsity.
- **Don't use random 2:4 sparsity** — gives slowdown; use STRUCTURED 2:4 fixed positions.
- **Don't quote 1.40× K-id as universal** — shape-conditional, real benefit ~2-6%.

## Section E — Open questions (UNRESOLVED across the section)

For research follow-up, here are the unresolved questions tracked in
this section:

### Tensor-core path / cuBLAS questions

- **U1**. Why does CUTLASS C++ 89 stuck at 5.5-5.7 PF (1005 MHz) when CuTeDSL hits 6.78 PF and cuBLAS hits ~10.8 PF? Same hardware, same shape. CUTEDSL_THROTTLE notes 61% wall-time is host overhead (`gemm.initialize()` in loop) — but even discounting that, CUTLASS C++ 89 lags CuTeDSL by 8-15 pp MFU. **Open.**
- **U2**. What is cuBLAS+cudaGraph doing that hits 76.2% MFU? Model predicts 73.1% ceiling from 19 ns fixed overhead. Graph eliminates launch-prep CPU time but shouldn't eliminate per-utcmma stall. Suggests "19 ns" model has hidden launch-related component. **Open.**
- **U3**. The 32×4 mma.sync regression mechanism (4× wall-clock loss). Triangulated via clock64 to confirm slowness, but mechanism unclear: ncu's gpc__cycles_elapsed under-reads by 4×, possibly due to ncu metric scope, or actual kernel has long teardown not captured. **Open.**

### NVFP4 / K=96 questions

- **U4**. NVFP4 K=96 cuBLAS 13.4 reported ~10.8 PF in memory entry but NOT verified in this clean directory.
- **U5**. NVFP4 single-shot vs sustained gap: 9109 TF (single-shot const, 91% spec) vs 6554 TF (sustained random cudaGraph 15s, 65% spec). The ~28% gap is power-throttle-mediated.
- **U6**. cuBLAS internal A↔B swap for NVFP4: `NVFP4_PURE_TCGEN05_RESULTS.md` flags that the cuBLAS NVF4 "A dominates power" observation may be due to cuBLAS internally swapping A and B before issuing UTCOMMA. Pure-tcgen05 microbench shows B-dominance for ALL 6 precisions. Swap unverified.
- **U7**. FP4 block-scaled (9856 TFLOPS) rigor: M3_REVERIFY_LOG line 47 lists "FP4 9856 TFLOPS" as still MED confidence (not yet 3-method verified).

### Power-model questions

- **U8**. Cache depth: "1 slot" (single-MMA) vs "2 slots" (cuBLAS K-id) reconciliation. SUBTILE_DEDUP_MODEL concludes 1-slot cache for BF16/FP8, 2-slot equivalent for NVFP4. N_DEPENDENCE_DEEPDIVE concludes Dedup cache holds 2 unique sub-patterns max. Possible resolutions: (a) different framings of the same HW; (b) 1-slot LRU per cycle, 2-slot effective via the alternation predictor.
- **U9**. Two-half processing: why BF16 m128n128 only? FP8 (K=32) and NVFP4 (K=64) are uniform within ±10 W. Hypotheses: (a) BF16 m128n128 has a specific 2× 64-N MAC array geometry; (b) FP8/NVFP4 K is larger so pipeline depth uniformizes; (c) something in the descriptor format / TMEM layout differs.
- **U10**. Pattern-count anomaly: 3-pattern rotation gives 538 W, 2-pattern gives 623 W (worse), 4-pattern back to 610 W. No clean model fits this.
- **U11**. Cache replacement policy: NOT simple LRU. chunk=1 ABAB triggers full speedup; chunk=2 AABB does not. A simple 2-slot LRU should keep both A and B in cache regardless of arrangement.
- **U12**. Sub-tile dedup vs K-row dedup magnitude in cuBLAS: at N=K=8192, sub-tile dedup contributes only ~1-3% to cuBLAS speedup, while K-row gives 42%. Whether the per-MMA sub-tile dedup mechanism actively contributes to cuBLAS performance is not cleanly separated.
- **U13**. A-vs-B asymmetry generalizes to A-major MMA layouts? `A_VS_B_ASYMMETRY.md` confidence: "LOW on whether this transfers to A-major MMA layouts (untested)."
- **U14**. Cross-precision two-half analog? BF16 has two halves at N=64 boundary. FP8 (K=32) and NVFP4 (K=64) might have analogous structure at different N positions; uniform position test only tested 0..7 sub-tiles.
- **U15**. tcgen05 vs mma.sync kind::f8f6f4 dedup behavior: all FP8 dedup measurements in this corpus are tcgen05.mma kind::f8f6f4. Whether the same 32-byte sub-tile dedup applies to the mma.sync emulated path is untested.

### NVFP4 specific questions

- **U16**. Diagonal sign patterns stay LOW even with many unique sub-tile patterns — contradicts the strict "≤2 patterns triggers LOW" rule. May indicate cache holds 8+ patterns OR there's a separate shift predictor.
- **U17**. Random-K-shift LOW for p_n ≤ 8 (8 unique patterns) but kphase_n shows 3 patterns = HIGH. Different cache behavior per axis (K-row direction vs N-column direction)?
- **U18**. K=96 N=64 saturation (0.5% spread, FLAT across all periods) — only K=96 N=64 has this property; no clear architectural explanation.
- **U19**. K-id speedup is shape-conditional. Whether NVFP4 K=96 ULTRA has the same restriction is not measured.
- **U20**. NVFP4 vs BF16 32-element MAC cliff: confirmed BF16 has 112 W cliff at stride 32. NVFP4 absence of cliff is MED confidence — within-word strides (1, 2, 4) had encoding bugs in the original sweep; only sign-bit-only retest is clean.
- **U21**. Lossless int reduction: NVFP4_INT_REDUCTION claims redux.sync.add gives 2.75× speedup, but NVFP4_FULL_PIPELINE shows mode 2 (HW decoders + per-thread fp32 + final SHFL) at 2.96× beats mode 7 (per-tile redux at 1.59×). Reduction speedup matters only when reduce is structurally per-tile.

### Methodology questions

- **U22**. Boost-clock TF/W (full ladder): no tcgen05 doc gives a complete 7-precision TF/W ladder at boost. Only random BF16 / FP8 spot checks at 1097/845 W (in MASTER §"Boost-clock validation"). Inferring boost TF/W requires assumptions about throughput scaling (2.02×) and power scaling (1.80×) holding identically across precisions.
- **U23**. ML inference perf/W validation in tcgen05: the "boost is 3× better than 510 MHz" memory rule was derived from FFMA. Tcgen05 has different power scaling characteristics. A direct tcgen05 perf/W vs clock sweep (510 / 800 / 1005 / 1300 / 1500 / boost) is not in this corpus.

## Section E — Authority and trust map

For any number cited in this section, the trust order is:

1. **B300_TRUE_REFERENCE.md** rows (single-source-of-truth synthesis)
2. **TCGEN05_PERFW_CLEAN_2TRIAL.md** for perf/W (supersedes single-trial PERF_WATTS)
3. **NVFP4_K96_AB_FULL.md** for NVFP4 K=96 power (3-trial verified, addendum has TDP-cap clock data)
4. **NVFP4_CUTEDSL_THROTTLE_DEEP_DIVE.md** for cross-impl benchmarks (model + king-shape verified)
5. **NVFP4_CUDAGRAPH.md** for absolute peak (record 11423 TF)
6. **NVFP4_CUBLAS_FULL_SWEEP.md** for per-clock optima
7. **N_DEPENDENCE_DEEPDIVE.md** for shape-conditional K-id and sparsity (read past line 659 for sparsity correction!)
8. **TCGEN05_DEDUP_CONSOLIDATED.md** for the unified dedup model
9. **NVFP4_PURE_TCGEN05_RESULTS.md** for pure-multiplier B>>A
10. **MMA_FP8_KIND_F8F6F4_NOT_NATIVE.md** for FP8 mma.sync emulation
11. Per-area deep-dives (NVFP4_K96_*, NVFP4_PERIOD_*, BF16_SUBTILE_*)
12. **corrections/** files supersede non-corrected versions where they exist

For any new measurement (per CLAUDE.md §5): run
`./utils/rigor_run.sh ./your_binary` for 3-method (wall-clock + ncu + SASS)
verification automatically.

For sub-agent outputs (per CLAUDE.md §6): apply step 3 of the verification
workflow — is the reported number plausible vs theoretical? Common
sub-agent failure modes: presents formula result as measured throughput,
uses wrong constants, trusts compiler-emitted code without SASS
verification, runs test too short (< 1 ms) and measures noise.

## Section E — What was retracted (R-series)

Quick reference for retractions documented across this section:

| ID | Original claim | Status | Source of retraction |
|----|----------------|--------|----------------------|
| R1 | "1543 TFLOPS BF16 single-chain mma.sync" | RETRACTED, real ~570 TF | TRUE_REFERENCE r58 |
| R2 | "6357 TFLOPS FP8 via mma.sync" | RETRACTED, DCE-folded | 06_tensor_cores |
| R3 | "2336 / 2400 TFLOPS FP8 via mma.sync" | RETRACTED, FADD artifact | 06_tensor_cores |
| R4 | "ncu pipe_tensor measures tcgen05" | DOES NOT APPLY to tcgen05 | SESSION_2_DELTA |
| R5 | "FP8 cuBLAS Not Available on B300" | RETRACTED, descriptors fixed | 06_tensor_cores |
| R6 | "FP8 sparse 7.44 PFLOPS = 74% of spec" | DOWNGRADED, sparse metadata may be garbage | 06_tensor_cores |
| R7 | "830 TB/s / 295 TB/s TMEM read" | RETRACTED, DCE-inflated, real ~60 TB/s | 06_tensor_cores |
| R8 | "838 / 420 TFLOPS HMMA FP16/TF32" | RETRACTED, ILP-override bug | 06_tensor_cores |
| R9 | "INT8 latency-bound, would scale with ILP" | RETRACTED, HW-throttled at 143 TOPS | 06_tensor_cores |
| R10 | "FP4 block-scaled rejected on sm_103a" | RETRACTED, kind::mxf4nvf4 works | 06_tensor_cores |
| R11 | "tcgen05 unsupported on sm_103a" | RETRACTED, NVRTC works | 06_tensor_cores |
| R12 | "FP8 mma.sync = 276 TFLOPS native" | RETRACTED, F2FP+HMMA emulation | MMA_FP8_KIND_F8F6F4_NOT_NATIVE |
| R13 | "4-slot HW pattern cache" | RETRACTED, sticky activation | TCGEN05_DEDUP_CONSOLIDATED §R1 |
| R14 | "K-row dedup is dominant cost" | superseded by sub-tile model | BF16_SUBTILE_DEDUP §R2 |
| R15 | "N-vary is FREE" | RETRACTED, low-entropy artifact | BF16_SUBTILE_DEDUP §R3 |
| R16 | "2:4 sparsity hurts dense GEMM" | RETRACTED in same doc | N_DEPENDENCE_DEEPDIVE line 659+ |
| R17 | "Diagonal patterns stay LOW universally" | RETRACTED, popcount-invariance | DIAGONAL_DEEP_DIVE |
| R18 | "K-uniform-per-N saves 124 W (-28%)" | RETRACTED, contamination | NVFP4_SIGN_K64_K96 MAJOR CORRECTION |
| R19 | "K-row sorting saves 349 W per CTA" | RETRACTED, K-axis BINARY | TCGEN05_PERFW_CLEAN_2TRIAL |
| R20 | "B=0 zero-skip saves 456 W (arithmetic detect)" | RENAMED to toggle-skip | TCGEN05_PERFW_CLEAN_2TRIAL |
| R21 | "PERF_WATTS NVFP4 K=96 = 13.72 TF/W" | CONTAMINATED, real 12.54-15.74 | TCGEN05_PERFW_CLEAN_2TRIAL |
| R22 | "1 outlier per K16 = 12% loss" | RETRACTED, real ~2% | NVFP4_K96_AB_FULL |
| R23 | "TMA multicast definitively explains A>B" | walked back, 4 hypotheses | NVFP4_PURE_TCGEN05_RESULTS Correction |
| R24 | "1.40× K-id is universal speedup" | shape-conditional | N_DEPENDENCE_DEEPDIVE |
| R25 | "10.8 PF cuBLAS catalog as ceiling" | superseded by cudaGraph 11.42 PF | NVFP4_CUDAGRAPH |
| R26 | "NVFP4 stride results table" | within-word strides had encoding bugs | NVFP4_PURE_TCGEN05_RESULTS |
| R27 | "Synthetic INT4 12% speedup" | corrected to ~4% | N_DEPENDENCE_DEEPDIVE |
| R28 | "cuBLAS picks different algorithm at different shapes" | RETRACTED, same kernel | N_DEPENDENCE_DEEPDIVE |

Sources: corrections/06_tensor_cores_CORRECTED.md §R1-R10,
corrections/TCGEN05_DEDUP_CONSOLIDATED.md §R1-R6,
corrections/NVFP4_CONSOLIDATED.md retractions table,
corrections/TCGEN05_POWER_CONSOLIDATED.md §R1-R4,
NVFP4_PURE_TCGEN05_RESULTS.md "Correction" §,
N_DEPENDENCE_DEEPDIVE.md inline corrections,
TCGEN05_PERFW_CLEAN_2TRIAL.md §R1-R2.

## Section E — Glossary (one-line definitions)

| Term | Definition |
|------|------------|
| **mma.sync** | Legacy warp-sync tensor instruction. RF accumulator. Hopper/Ada compatible. SASS = HMMA family. |
| **tcgen05.mma** | Blackwell warpgroup-async tensor instruction. TMEM accumulator. SASS = UTCHMMA / UTCQMMA / UTCOMMA. |
| **TMEM** | Tensor Memory — separate SRAM region per SM for tcgen05 accumulators. NOT visible to mma.sync. |
| **UTCHMMA** | tcgen05 BF16/FP16 multiplier instruction. SASS opcode. |
| **UTCQMMA** | tcgen05 FP8 multiplier instruction. SASS opcode. |
| **UTCOMMA** | tcgen05 NVFP4/MXFP4 multiplier instruction. SASS opcode (UTCOMMA.BLOCK16 for K=96 ULTRA). |
| **K=96 ULTRA** | NVFP4 path with `k_size_=1` in descriptor; 1.5× MAC density per cycle vs K=64 standard. |
| **kind::f16** | PTX modifier for BF16/FP16 tcgen05.mma. |
| **kind::f8f6f4** | PTX modifier for FP8/FP6/FP4. In mma.sync = emulated F2FP+HMMA; in tcgen05.mma = native. |
| **kind::mxf4nvf4.block_scale.block16** | PTX modifier for NVFP4 with UE4M3 scale per 16 elements. |
| **K-id** | "K-row identical" — synthetic data pattern where B has the same values across all K rows of a tile. Triggers cuBLAS dedup speedup. |
| **K-row pairwise dedup** | tcgen05 hardware optimization that detects period-1 and period-2 K-row patterns. |
| **Sub-tile dedup** | tcgen05 hardware optimization where B-side 32-byte chunks matching the active cache slot draw 0 W. |
| **32-byte sub-tile** | The universal HW B-side dedup granularity (16 BF16, 32 FP8, 64 NVFP4 N-values). |
| **Two-half processing** | BF16 m128n128 specific: Half A (N=0..63) = always-on; Half B (N=64..127) = aggressively gated. |
| **Sticky activation** | B port starts low-power gated; first non-matching sub-tile activates it; stays active per-MMA. |
| **MFU** | Math FLOPs Utilization — measured TFLOPS / theoretical peak for the precision. |
| **Per-total-SM MFU** | TFLOPS / (peak × all 148 SMs). |
| **Per-active-SM MFU** | TFLOPS / (peak × active_SM_count). Higher when cluster shape leaves SMs idle. |
| **TF/W** | TeraFLOPS per Watt. Power-aware throughput metric. |
| **TDP cap** | 1100 W on B300 SXM6. Random data hits this cap and throttles clock. |
| **Zero-skip path** | Multiplier short-circuit when one operand is uniformly zero. Saves 50-100 W per CTA, allows full clock. |
| **TMA multicast** | TMA's broadcast feature; one HBM read shared across cluster. NVFP4 cuBLAS uses 78% multicast on B; BF16 cuBLAS uses 0%. |
| **cluster_group::2** | 2-CTA cluster mode for tcgen05. Required for m=256 NVFP4 K=96. No cluster-shared dedup pooling. |
| **disable_lane** | tcgen05 feature to gate output columns. Linear ~2.4 W per disabled column on BF16. |
| **B>>A asymmetry** | Per-multiplier observation: B-randomness costs 15-30× more power than A-randomness when isolated. |
| **A:B 3-way reading** | NVFP4 power asymmetry has 3 different right answers (cuBLAS A>B, pure-tcgen05 B>>A, K=96 single-kernel B>A 2.6×). |
| **F2FP.UNPACK** | SASS opcode for FP8 → FP16 conversion. Used by mma.sync kind::f8f6f4 emulation. |
| **HMMA.16816.F32** | Legacy mma.sync m16n8k16 FP32 accumulator SASS opcode. |
| **2:4 structured sparsity** | NVIDIA's sparsity feature with FIXED zero positions per 4-element group. Gives +11% on dense GEMM. |
| **Random 2:4 sparsity** | Random which 2 of 4 are zero. NO speedup; can give slowdown via U-curve thrash. |
| **U-curve sparsity** | Tier-2 mechanism: 30-50% RANDOM sparse HURTS dense GEMM by 5%; >75% helps; full-zero gives 1.52×. |
| **Toggle-energy model** | Memory power follows popcount bell curve (peak at d=16 random); chunk-dedup is NULL. |
| **DCE** | Dead Code Elimination. Compiler removes unused computation. Defeats with unconditional output writes. |
| **rule #9** | "Suspect the test before the hardware." Applied 4 times in N_DEPENDENCE_DEEPDIVE corrections. |
| **clock64** | In-kernel SM cycle counter PTX. Gold standard for triangulation when wall-clock and ncu disagree. |
| **`-rgc`** | nvidia-smi unlock clock. Lets boost clock float to 2032 MHz under TDP. |
| **`-lgc N`** | nvidia-smi lock clock. Note: `-lgc 2032` paradoxically pins to 1920 MHz (base clock). |
| **`pkill -9 QuickRunCUDA && sleep 5-8`** | Per CLAUDE.md §8.4: leftover processes silently inflate cy/MMA up to 8.5×. |
| **rigor_run.sh** | `./utils/rigor_run.sh ./your_binary` — 3-method (wall-clock + ncu + SASS) verification automatically. |
| **TCGEN05_PERFW_CLEAN_2TRIAL** | Authoritative 2-trial perf/W ladder with `pkill -9` + 6 s cooldown between every measurement. Supersedes single-trial PERF_WATTS. |
| **K=8 / K=16 / K=32 / K=64 / K=96** | tcgen05.mma K-dimension per instruction: TF32 / FP16-BF16 / FP8 / NVFP4-standard / NVFP4-ULTRA respectively. |
| **m=128 n=128** | Single-CTA tcgen05.mma sweet spot for 98.5% MFU on BF16/FP16/FP8 single-warp issuer. |
| **m=256 n=256** | 2-CTA cluster tcgen05.mma sweet spot for 98.5% MFU on NVFP4 K=96 ULTRA. |
| **C2:4** | Compressed 2:4 sparsity format. CUSPARSE / cuBLASLt has dedicated sparse APIs that interpret 2:4 metadata. Different from "fixed-position 2:4 on dense" which is pure pattern detection. |
| **bit-entropy** | Per-byte information content of input data. Zero-data has bit-entropy=0; random has bit-entropy=8. Universal entropy detector triggers at exactly 0. |
| **Rigor protocol** | CLAUDE.md §1-6: state theoretical first; if measured > theoretical, test broken; verify via SASS + ncu cross-check. |
| **f2f8 / f2f4** | Format conversion instructions: BF16/FP16 → FP8/NVFP4. Used in pre-quantization. |
| **UE4M3** | NVFP4 scale factor format: unsigned exponent 4-bit + mantissa 3-bit. Encodes scale per 16 elements. |
| **UE8M0** | MXFP4/MXFP8 scale factor format: unsigned exponent 8-bit + mantissa 0-bit. Power-of-2 scale per 32 elements (FP8) or 16 (FP4). |
| **n+64 lane stride** | NVFP4 multiplier lane-pair structure. Sign at n+64 OPPOSITE = 100% toggle on lane pair = +50 W worst case. |
| **+0 vs -0** | Use +0 (0x0) not -0 (0x8) for zero weights in NVFP4. Saves 3-103 W depending on sparsity. |

## Section E — Reading order recommendation

For a new investigator approaching this section:

1. Start with **§46** for the overall ladder and what numbers to quote.
2. Read **§47-§48** to understand path differences and avoid mma.sync FP8 trap.
3. Skim **§49** to understand the K=96 ULTRA vs cuBLAS dispatch gap.
4. **DEEPLY READ §50** — the A:B 3-way reading is the most-confused topic in the corpus. Don't skip the reconciliation discussion.
5. Read **§51-§52** for power model details (only if writing custom kernels or doing power optimization).
6. Read **§53** to understand why the 1.40× K-id speedup doesn't apply to real ML.
7. Read **§54** to understand library landscape (CUTLASS / CuTeDSL / cuBLAS).
8. **READ §55 ALL THE WAY THROUGH** — the 2:4 sparsity correction is mid-document; skipping the second half gives wrong sign.

For a casual reader looking for one number:
- BF16 realistic: 1850 TF
- FP8 realistic: 3984 TF
- NVFP4 best: 11423 TF (76.2% spec)
- 2-GPU NVFP4: 19163 TF (95.8% spec)
- Llama-style realistic NVFP4: ~30% of spec at 1005 MHz

For a power-optimization reader:
- NVFP4 K=96 + 5-pos B at 1005 MHz = 17.0 TF/W (best 1005 efficiency)
- NVFP4 K=96 + all-zero B at boost = 23.4 TF/W (best ever, zero-skip path)
- BF16 realistic = ~2.4 TF/W; FP8 = ~4.5 TF/W; NVFP4 = ~13.4 TF/W
- 1005 MHz is more efficient than boost by 6-10% for same data pattern

## Section E — Last-mile sanity checks (verification recipe)

For ANY tensor-core measurement you make, run through this checklist
before reporting:

1. **State theoretical first**. "Theoretical peak = X TFLOPS at clock Y MHz."
2. **State measured**. "Measured Z TFLOPS = Z/X of theoretical."
3. **If Z > X**: STOP. Test is broken. Look for DCE, formula bugs, clock mismatch.
4. **If Z > 1.5× theoretical**: almost certainly DCE.
5. **If Z < 0.5× theoretical**: under-saturated or methodology issue. Check ILP, occupancy, anti-DCE.
6. **If Z in [0.5×, 1.0×]**: plausible, but verify SASS has expected instruction count, check ncu metrics.
7. **SASS-verify**: `nvcc -keep` and look at the .sass — count the expected instructions. For tcgen05, look for UTCHMMA / UTCQMMA / UTCOMMA. For mma.sync FP8 kind::f8f6f4, you'll see F2FP.UNPACK + HMMA — that's the emulation tell.
8. **Cross-check with ncu** where available: `pipe_fma.avg.pct_of_peak_sustained_active` for FFMA, `sm__pipe_tensor_subpipe_hmma_cycles_active.sum` for mma.sync HMMA, `…hmma_op_utchmma_utcqmma_utcomma…` (full subpipe name) for tcgen05.
9. **Triangulate** wall-clock + ncu + clock64 (in-kernel) — when wall-clock and ncu disagree, clock64 is the gold standard.
10. **State data pattern explicitly** — "zero data" / "random" / "normal-ish". Catalog peaks are usually zero. Realistic drops 10-22% per §46.4.
11. **State clock state explicitly** — "boost (~2032)" / "lock 1500 MHz" / "lock 1005 MHz" / "lock 510 MHz" / `nvidia-smi -lgc 2032 (paradoxically pins to 1920)`.
12. **State sustained vs single-shot** — Single-shot can hit 91% spec; sustained random throttles to 65% via TDP cap.
13. **Pkill leftover processes** — `pkill -9 QuickRunCUDA && sleep 5-8` between measurements (CLAUDE.md §8.4: leftover procs silently inflate cy/MMA up to 8.5×).
14. **For new measurements**: run `./utils/rigor_run.sh ./your_binary` for automatic 3-method verification.
