# B300 CONFIDENCE LADDER (v1, 2026-04-22)

**Purpose:** single reference table grading every numeric claim across
`b300_clean/corrections/`. Each row carries a confidence grade and the
doubt-status flag from wave-3b adversarial reports.

**Confidence rubric:**
- `HIGH` = 3-method (wall-clock + ncu + SASS) verified AND not contradicted by any doubt report
- `MED`  = 1-2 methods OR has minor unresolved aspect (regime, denominator, single-shape, etc.)
- `LOW`  = doubt swarm flagged methodology issue OR cross-agent contradiction unresolved
- `DISPUTED` = multiple values across files, no consensus / 1.5×+ spread

**Doubt status:**
- `doubt-confirmed` = wave-3b report agreed with the value
- `DOWNGRADED` = wave-3b found methodology issue
- `REFRAMED` = value correct, framing/% denominator wrong
- `UNRESOLVED` = open contradiction
- `untouched` = no doubt report on this metric

System: B300 SXM6 AC (sm_103a, 148 SMs, 12 HBM3E stacks). Default sustained boost 2032 MHz.

---

## 1. HBM / DRAM bandwidth

| Metric | Value | Unit | Conf | Source file | Doubt status |
|---|---:|---|---|---|---|
| HBM3E theoretical post-ECC spec (denominator) | 7672 | GB/s | HIGH | 01_hbm_bandwidth_CORRECTED §0 | doubt-confirmed |
| HBM read peak (canonical NINJA, v8 + per-warp coalesced) | 7.30 | TB/s | HIGH | 01_CORRECTED §1; TRUE_REF_v2 | doubt-confirmed |
| HBM read peak (LDG.E.128 SoL, 37888 blocks) | 7.365 | TB/s | HIGH | 01_CORRECTED §1 | doubt-confirmed |
| HBM read peak (TMA bulk 8 KB) | 7.344 | TB/s | HIGH | 01_CORRECTED §1 | doubt-confirmed |
| HBM read (V46 TMA 8-deep pipelined) | 7.20 | TB/s | HIGH | V41_V48; 01_CORRECTED §2 | REFRAMED (98.5% used wrong denom; 7.20/7672=93.8%) |
| HBM read V33 single-deep TMA | 6.72 | TB/s | HIGH | V33; 01_CORRECTED §2 | doubt-confirmed |
| HBM read A6 R-only (32:0 sweep) | 7.31 | TB/s | HIGH | 01_CORRECTED A6 | doubt-confirmed |
| HBM write SoL (NINJA STG vs TMA bulk attribution) | 7.57 | TB/s | DISPUTED | 01_CORRECTED §3; HBM_LOG #3 | UNRESOLVED (provenance contested e75c7e1 vs 28211ce) |
| HBM write peak (v8 STG + per-warp coalesced 32-iter) | 7.30 | TB/s | HIGH | 01_CORRECTED §1 | doubt-confirmed |
| HBM write TMA single-deep (V34) | 7.17 | TB/s | HIGH | V34; 01_CORRECTED §2 | doubt-confirmed |
| HBM write TMA 8-deep pipelined (V47, NO BENEFIT) | 6.34 | TB/s | HIGH | V47; 01_CORRECTED §2 | doubt-confirmed |
| HBM concurrent R+W min (50:50 mix) | 6.68 | TB/s | HIGH | 01_CORRECTED §6 A6 | doubt-confirmed |
| HBM D2D NINJA (separate src/dst) | 6.93 | TB/s | HIGH | 4958d6b | doubt-confirmed |
| HBM D2D cudaMemcpyAsync | 6.56 | TB/s | HIGH | 01_CORRECTED §1 | doubt-confirmed |
| TMA multicast effective (cluster=8, single-deep) | 14.91 | TB/s | HIGH | V32 | doubt-confirmed |
| TMA multicast 2-deep (CAPPED, single engine) | 13.96 | TB/s | HIGH | V48 | doubt-confirmed |
| Plain LDG.32 coalesced | 1.95 | TB/s | HIGH | V10_LDG_WIDTH | untouched |
| Plain LDG.64 coalesced | 3.65 | TB/s | HIGH | V10_LDG_WIDTH | untouched |
| Plain LDG.128 coalesced | 5.76 | TB/s | HIGH | V10_LDG_WIDTH | untouched |
| cp.async.ca (LDGSTS) | 6.91 | TB/s | HIGH | V9_CP_ASYNC_BW; 09_CORRECTED | untouched |
| WS=126 MB cliff (DRAM-edge) | 8.2 | TB/s | HIGH | 01_CORRECTED §5 | untouched |
| HBM data-dependence swing (popcount d=0→16→32) | 240–554 | W | HIGH | POPCOUNT_3TIER | doubt-confirmed (HBM_DATA_DEPENDENCE.md <50W RETRACTED) |

---

## 2. L1 / L2 cache

| Metric | Value | Unit | Conf | Source file | Doubt status |
|---|---:|---|---|---|---|
| L1+SHMEM unified pool per SM | 256 | KB | HIGH | 03_caches_CORRECTED §1.1 | doubt-confirmed |
| L1 line size | 128 | B | HIGH | 03_caches_CORRECTED §1.1 | untouched |
| L1 effective capacity (4 KB stride) | ~128 | KB | HIGH | D2_L1_CAPACITY_RIGOR | untouched |
| L1 effective capacity (random Fisher-Yates) | 2-4 | KB | HIGH | V10_L1_CAPACITY | untouched |
| L1 hit latency (warm pointer-chase) | 38–47 | cy | HIGH | D2/V10/03 cross-test | untouched |
| L1 BW (default ld, 8-ILP × 16 unroll) | 30.5 | TB/s | MED | V8_L2_BW_VERIFIED | UNRESOLVED (M5 cheatsheet says 46) |
| L1 BW (M5 cheatsheet optimistic) | 46 | TB/s | LOW | M5_MEMORY_CHEATSHEET | UNRESOLVED vs V8's 30.5 |
| `.ca` vs `.cg` ratio at 8 KB WS | 13.8× | — | HIGH | 03_caches_CORRECTED §1.3 | untouched |
| L2 capacity | 126.5 | MB | HIGH | cudaDeviceProp.l2CacheSize; STRAYS §7 | doubt-confirmed (96 MB cosmetic error in 4 files) |
| L2 max persisting | 79.1 | MB | HIGH | cudaDevAttrMaxPersistingL2CacheSize | untouched |
| L2 partitions | 2 | sides | HIGH | 03_caches_CORRECTED §2.1 | untouched |
| L2 line size | 128 | B | HIGH | D3_L2_SECTOR_RIGOR | untouched |
| L2 sector size | 32 | B | HIGH | D3_L2_SECTOR_RIGOR | untouched |
| L2 BW kernel-effective (with L1 reuse) | 23.85 | TB/s | HIGH | 03_caches §2.3 | doubt-confirmed (label MANDATORY) |
| L2 BW wire (lts__t_bytes) | 13.30 | TB/s | HIGH | 03_caches §2.3 | doubt-confirmed (label MANDATORY) |
| L2 BW @ `.cg` carveout=100, 8–128 MB WS | 17 | TB/s | HIGH | 03_caches §3a | untouched |
| L2 BW @ `.cg` carveout=0, 4–128 MB | 22-26 | TB/s | MED | 03_caches §3b | untouched |
| L1-amplified small-WS peak | ~30 | TB/s | HIGH | 03_caches §3c | doubt-confirmed (NOT L2; mislabeled before) |
| L2 strided `.cg` 64 MB | 13.85 | TB/s | HIGH | V8_L2_BW_VERIFIED | untouched |
| L2 hit latency (avg) | 300–310 | cy | HIGH | 03_caches §2.5 | untouched |
| L2 near-far ratio | 1.27–2.4× | — | HIGH | 03_caches §2.5 | untouched |
| L2 atomic units count | ~32 | units | MED | L2_UNITS_REFINED; 03_caches §2.6 | DOWNGRADED (was MED-HIGH; ATOMIC_REVERIFY_DEEP says ceiling could be higher) |
| L2 atomic per-unit throughput | 1.55 | Gpkt/s | HIGH | L2_UNITS_REFINED | untouched |
| L2 video clock (independent of SM) | 1860 | MHz | HIGH | CLOCK_DOMAINS_AND_L2_UNITS | untouched |

---

## 3. Shared memory (SHMEM)

| Metric | Value | Unit | Conf | Source file | Doubt status |
|---|---:|---|---|---|---|
| SHMEM theoretical peak @ 2032 MHz | 38.49 | TB/s | HIGH | derivation 32 banks × 4 B × 148 × 2.032 | doubt-confirmed |
| SHMEM read peak (LDS.128 + RAW + 1 blk/SM) | 38.4 | TB/s | HIGH | 02_shmem_CORRECTED §1; d41c38c | doubt-confirmed |
| SHMEM read 4 SMSPs | 38.0 | TB/s | HIGH | 02_shmem ninja_smsp_vec | untouched |
| SHMEM read 2 SMSPs | 35.4 | TB/s | HIGH | 02_shmem | untouched |
| ldmatrix.x4.b16 (tensor feed) | 33–35 | TB/s | HIGH | 02_shmem 4ccda4f | untouched |
| stmatrix W+R chain | 34.5 | TB/s | HIGH | 02_shmem 8bd85e8 | untouched |
| Read+write mix (4R+1W/iter) | 27.2 | TB/s | HIGH | 02_shmem 4503a17 | untouched |
| Sustained throttled (>8000 iter @1920) | 17–21 | TB/s | MED | 02_shmem §6 | untouched |
| SMEM total per SM (opt-in) | 228 | KB | HIGH | 02_shmem §4 | untouched |
| SMEM atomic INT32 add uncontended | 4.6 | cy | HIGH | 02_shmem §atomics | untouched |
| SMEM atomic INT32 32-way (zero penalty) | 4.6 | cy | HIGH | 02_shmem §atomics | untouched |
| SMEM atomic FP32 32-way (67× penalty) | 5729 | cy | HIGH | 02_shmem §atomics | untouched |
| SMEM atomic aggregate INT throughput | 2.27 | Tatomic/s | HIGH | V10_SMEM 968e5b7 | DOWNGRADED (CLAUDE memory's 4.2 T is unsourced; HEADLINE_v2 #8) |
| Bank-conflict 32-way (latency-bound) | ~2× | — | HIGH | V44 | untouched |
| Bank-conflict 32-way (throughput-bound, hidden) | ~1× | — | HIGH | V45 | doubt-confirmed (regime-dependent, 1× to 8.8×) |
| Bank-conflict 32-way (multi-warp contention) | 8.81× | — | HIGH | bce8bf8 | untouched |
| Bank-conflict 32-way (single-warp transpose) | 8.2× | — | HIGH | Q6_SMEM_TRANSPOSE | untouched |

---

## 4. DSMEM (cluster-shared)

| Metric | Value | Unit | Conf | Source file | Doubt status |
|---|---:|---|---|---|---|
| Local SMEM (LDS) latency | 24 | cy | HIGH | DSMEM_CORRECTED §2 | doubt-confirmed |
| DSMEM self (mapa→me) latency | 54 | cy | HIGH | DSMEM_CORRECTED §2 | doubt-confirmed |
| DSMEM cluster=2 latency | 214.75 | cy | HIGH | DSMEM_CORRECTED | doubt-confirmed (cluster=2 is 21% slower than ≥3) |
| DSMEM cluster=3..8 avg latency | ~180 | cy | HIGH | DSMEM_CORRECTED §2 | doubt-confirmed |
| DSMEM best pair (SM32↔SM33) | 164.80 | cy | HIGH | DSMEM_CORRECTED | doubt-confirmed |
| DSMEM worst pair (SM16↔SM17) | 204.97 | cy | HIGH | DSMEM_CORRECTED | doubt-confirmed |
| DSMEM write (fenced) latency | 34 | cy | HIGH | DSMEM_CORRECTED | doubt-confirmed |
| DSMEM atomic .add latency | 188–239 | cy | HIGH | DSMEM_CORRECTED | untouched |
| Local/DSMEM latency ratio | 7.5× | — | HIGH | DSMEM_CORRECTED §2 | doubt-confirmed (NOT 0.8% LICM, NOT 4.7×) |
| DSMEM read aggregate per cluster (4×4 ring) | 40 | GB/s | MED | DSMEM_CORRECTED §3 | DOWNGRADED (chain-bound; non-chained ILP could be 60-80) |
| DSMEM write aggregate per cluster (4×4) | 560 | GB/s | LOW-MED | DSMEM_CORRECTED §3 | DOWNGRADED (issue rate, no fence; real delivery unverified) |
| DSMEM ring contention (N=1..8 readers) | 1.00× flat | — | LOW | DSMEM_CORRECTED §4 | DOWNGRADED ("NO shared bus" — V17 was 30× under-issued) |
| DSMEM hot-spot per-source serving cap | ~15 | GB/s | MED | DSMEM_CORRECTED §4 | untouched |
| DSMEM hot-spot atomic scaling (N=8) | 63 | Matom/s | HIGH | DSMEM_CORRECTED V11–V31 | untouched |
| TMA multicast 32 KB tile (8-way) | 470 | GB/s | HIGH | DSMEM_CORRECTED §3 | untouched |
| All V8/V10 "DSMEM 37 TB/s read / 11.8 TB/s write" | RETRACTED | — | — | DSMEM_DOUBT | DCE artifacts (HIGH that they're DCE) |

---

## 5. FP32 / FFMA peaks

| Metric | Value | Unit | Conf | Source file | Doubt status |
|---|---:|---|---|---|---|
| FFMA theoretical @ 2032 MHz | 76.96 | TFLOPS | HIGH | 04_fp32_peak_CORRECTED | doubt-confirmed |
| FFMA theoretical @ 1920 MHz | 72.71 | TFLOPS | HIGH | 04_fp32_peak_CORRECTED | doubt-confirmed |
| FFMA peak 2-source @ 2032 boost | 75.92 | TFLOPS | HIGH | 04_fp32_peak fp32_peak_definitive.cu | doubt-confirmed (97-99% noise band) |
| FFMA peak 2-source pipe_fma % | 97.65 | % | HIGH | V8_FFMA_PEAK_VERIFIED | doubt-confirmed |
| FFMA NCHAIN=3 rotating + immediate | 74.62 | TFLOPS | HIGH | TRUE_REF 06b0d8d | doubt-confirmed |
| FFMA realistic 3-distinct-source GEMM | ~50 | TFLOPS | HIGH | V10_FMA_SOURCE_COUNT; A4; D6 | doubt-confirmed (RF port: ratio 0.683 ≈ 2/3) |
| FFMA @ 1920 MHz locked | 62.17 | TFLOPS | HIGH | TRUE_REF e1a1220 | untouched |
| FADD scalar | 37.4 | TFLOPS | HIGH | V8_FADD_FMUL_PEAK | untouched |
| FMUL scalar | 37.3 | TFLOPS | HIGH | V8_FADD_FMUL_PEAK | untouched |
| FADD/FMUL/FFMA SASS rate | 1 | inst/SMSP/cy | HIGH | V8_FADD_FMUL_PEAK | untouched |
| FFMA/FADD/FMUL latency | 4.04–4.22 | cy | HIGH | V9_OP_LATENCY | untouched (V8 catalog "23 cy" RETRACTED) |
| FP16/BF16 packed FMA outside tensor cores | == FP32 | — | HIGH | TRUE_REF surprise #2 | untouched (HFMA2 NOT 2× FFMA) |

---

## 6. FP64

| Metric | Value | Unit | Conf | Source file | Doubt status |
|---|---:|---|---|---|---|
| FP64 DFMA theoretical (1:64 of FP32) | 1.203 | TFLOPS | HIGH | cudaDeviceGetAttribute SinglePerf=64 | untouched |
| FP64 DFMA peak measured @ 2032 boost | 1.203 | TFLOPS | HIGH | V8_FP64_PEAK_VERIFIED | doubt-confirmed (100% pipe_fp64) |
| FP64 DMMA / DGEMM tensor | 1.05 | TFLOPS | HIGH | 06_tensor_cores | doubt-confirmed (no FP64 tensor speedup) |

---

## 7. Tensor cores (per precision)

| Metric | Value | Unit | Conf | Source file | Doubt status |
|---|---:|---|---|---|---|
| FP16 cuBLAS 8K³ zero | 2246 | TFLOPS | HIGH | TRUE_REF row 66 | untouched |
| FP16 cuBLAS 8K³ random | 1905 | TFLOPS | HIGH | 06_tensor_CORRECTED | untouched |
| FP16 cuBLAS 8K³ realistic | 1744 | TFLOPS | HIGH | 06_tensor_CORRECTED | untouched |
| BF16 cuBLAS 8K³ zero (sustained) | 2242–2246 | TFLOPS | HIGH | TRUE_REF rows 48,67 | untouched |
| BF16 cuBLAS 8K³ random | 1850–1883 | TFLOPS | HIGH | 06_tensor_CORRECTED | untouched |
| BF16 microbench tcgen05 direct | 2325 | TFLOPS | HIGH | 06_tensor_CORRECTED | untouched |
| BF16 mma.sync m16n8k16 V8 8-chain @2032 | 578.6 | TFLOPS | HIGH | V8_HMMA_F16_PEAK (99.9% pipe sat) | untouched |
| BF16 mma.sync m16n8k16 @1920 locked | 569 | TFLOPS | HIGH | TRUE_REF row 47 | untouched |
| BF16 1543 TFLOPS single-chain | RETRACTED | — | — | TRUE_REF row 58 | doubt-confirmed (over-counted; real ~570) |
| TF32 cuBLAS 8K³ | 1113 | TFLOPS | HIGH | 06_tensor_CORRECTED | untouched |
| TF32 m16n8k8 mma.sync | 288 | TFLOPS | MED | 06_tensor_CORRECTED | untouched |
| FP8 e4m3 cuBLAS 8K³ zero (sustained) | 4425–4491 | TFLOPS | HIGH | TRUE_REF rows 56,68 | untouched |
| FP8 e4m3 cuBLAS 8K³ random | 3983 | TFLOPS | HIGH | TRUE_REF row 57 | untouched |
| FP8 e4m3 cuBLAS realistic | 3951 | TFLOPS | HIGH | 06_tensor_CORRECTED | untouched |
| FP8 microbench tcgen05 direct | 4651 | TFLOPS | HIGH | 06_tensor_CORRECTED | untouched |
| FP8 cuBLAS under 600 W cap (random) | 3087 | TFLOPS | HIGH | TRUE_REF row 68 | doubt-confirmed (top-10 #9: random 43% slower than zero under cap) |
| FP8 mma.sync 7500–8200 TFLOPS | RETRACTED | — | — | META log B2 | upstream-self-retracted (HMMA.16816 not 16832; real ~3760) |
| INT8 m16n8k32.s32.s8 mma.sync | 143 | TOPS | HIGH | 06_tensor_CORRECTED | doubt-confirmed (HW-throttled 5 NOPs/issue) |
| NVFP4 cuBLAS plain Lt (boost K=38400) | 11068 | TFLOPS | MED | NVFP4_CONSOLIDATED | doubt-confirmed |
| NVFP4 cuBLAS+cudaGraph BPG=16 (boost K=38400) | 11423 | TFLOPS | MED | NVFP4_CUDAGRAPH | DOWNGRADED to single-shape (no BPG sweep) |
| NVFP4 cuBLAS sustained random (15s, throttle 1057 MHz) | 6554 | TFLOPS | HIGH | TRUE_REF row 54 | untouched |
| NVFP4 K=64 std microbench @1005 (per-CTA) | 4.87 | PFLOPS | HIGH | NVFP4_K96_AB_FULL | untouched |
| NVFP4 K=96 ULTRA microbench @1500 lock (per-CTA) | 10.91 | PFLOPS | HIGH | NVFP4_K96_AT_1500MHZ | doubt-confirmed (98.5% per CTA; cuBLAS won't dispatch this path) |
| NVFP4 K=96 ULTRA all-zero zero-skip @ boost | 14.78 | PFLOPS | HIGH | NVFP4_K96_AB_FULL addendum | untouched |
| NVFP4 K=96 cuBLAS 13.4 wide-rect | ~10800 | TFLOPS | LOW | memory entry | UNRESOLVED (not verified in clean dir) |
| CUTLASS C++ NVFP4 (boost 8K² K=15K) | 8285 | TFLOPS | MED | CUTEDSL_THROTTLE | untouched |
| CuTeDSL NVFP4 (boost 8K² K=15K) | 9118 | TFLOPS | MED | NVFP4_CONSOLIDATED | untouched |
| CuTeDSL persistent kernel @1005 cluster (2,1) | 6776 | TFLOPS | HIGH | CUTEDSL_THROTTLE (91.3% MFU) | untouched |
| FP4 block-scaled microbench (mxf4nvf4.block16) | 9856 | TFLOPS | MED | 06_tensor_CORRECTED | UNRESOLVED (M3_REVERIFY_LOG: re-verify pending) |
| 2-GPU NVFP4 split (1 stream/GPU, no comm) | 19163 | TFLOPS | HIGH | TRUE_REF cbaadbc | untouched |
| HMMA m16n8k16 F32-acc latency | 20.09 | cy | HIGH | V9_HMMA_LATENCY (20 cy / 9.8 ns) | doubt-confirmed |
| TMEM read peak | ~57–65 | TB/s | HIGH | 06_tensor §3.2; D7 | doubt-confirmed (830/295 TB/s claims RETRACTED as DCE) |
| TMEM write peak | 97–131 | TB/s | HIGH | 06_tensor §3.2 | untouched |
| TMEM per CTA capacity | 256 | KB | HIGH | 06_tensor §3.1 | untouched |
| `pipe_tensor.cycles_active` measures tcgen05 | NO | — | HIGH | 06_tensor R4 | doubt-confirmed (top-10 #6) |

---

## 8. Atomics (per scope)

| Metric | Value | Unit | Conf | Source file | Doubt status |
|---|---:|---|---|---|---|
| Global atom.{cta,gpu,sys}.add chained RT | ~697 | cy | HIGH | V9_ATOMIC_LATENCY (343 ns) | untouched |
| Global atom.relaxed.gpu (per-thread, near-L2) | ~310 | cy | HIGH | 07_atomics §1 | untouched |
| Global atom.relaxed.gpu (per-thread, far-L2) | ~680 | cy | HIGH | 07_atomics §1 | untouched |
| Local atomic L2 round-trip (no chain, near-L2) | 164 | ns | HIGH | TRUE_REF row 86 | untouched |
| Cross-GPU atomic via NVLink P2P | 1662 | ns | HIGH | TRUE_REF row 87 | untouched |
| atomicInc | 7.9 | cy | HIGH | 07_atomics_CORRECTED §2 | untouched |
| atomicDec | 7.0 | cy | HIGH | 07_atomics §2 | untouched |
| atomicAdd / Sub | 15.2 | cy | HIGH | 07_atomics §2 | untouched |
| atomicMin / Max | 15.7 | cy | HIGH | 07_atomics §2 | untouched |
| atomicAnd / Or / Xor | 23.5 | cy | HIGH | 07_atomics §2 | untouched |
| atomicExch | 49.5 | cy | HIGH | 07_atomics §2 | untouched |
| atomicCAS | 52.5 | cy | HIGH | 07_atomics §2 | untouched |
| Stride 0 (full collision) | 0.79 | Gops/s | HIGH | TRUE_REF | untouched |
| Stride 4 (cache-line combining UNROLL=16) | 504 | Gops/s | HIGH | 07_atomics §8 | untouched |
| Stride 4 true peak (UNROLL=32, L2-resident) | 1005 | Gops/s | HIGH | 07_atomics §8 | doubt-confirmed (449 was lower-ILP) |
| Stride 32 (1 line/thread) | 184 | Gops/s | HIGH | TRUE_REF | untouched |
| Stride 256+ (scattered) | ~150 | Gops/s | HIGH | TRUE_REF | untouched |
| Stride 128 B per thread, no combine | 49.7 | Gatomic/s | HIGH | ATOMIC_LADDER §CASE1 | untouched |
| Combine=32, lane=offset, WS=32MB | 1230 | Gatomic/s | HIGH | REVERIFY_VERSION_A | untouched |
| Combine=32, lane=offset, WS=1024MB | 768 | Gatomic/s | HIGH | REVERIFY_VERSION_B | untouched |
| Universal atomic DRAM ceiling | ~5.5 | TB/s | HIGH | ATOMIC_LADDER (75% of 7.31) | untouched |
| Cross-GPU LOCAL atomic all-contend | 49.4 | Gatomic/s | HIGH | 12_nvlink_p2p §5 | untouched |
| Cross-GPU REMOTE atomic all-contend | 16.6 | Gatomic/s | HIGH | 12_nvlink_p2p §5 | untouched |
| Cross-GPU REMOTE atomic unique addresses | 9.2 | Gatomic/s | HIGH | 12_nvlink_p2p §5 | untouched |
| atomicAdd FP32 (HW path) | 7-8 | ns | HIGH | TRUE_REF | untouched |
| atomicAdd FP64 (HW path) | 4.5 | ns | HIGH | TRUE_REF | untouched |
| atomicAdd scalar half/bfloat16 (NO HW) | 700 | ns | HIGH | TRUE_REF | untouched |
| atomicAdd packed half2/bfloat162 (HW) | 16 | ns/elem | HIGH | TRUE_REF | untouched |
| red.release.gpu.global | 614 | ns/op | HIGH | TRUE_REF + ATOMICS_LOG A10 (cause: MEMBAR.ALL.GPU not CCTL.IVALL) | untouched |

---

## 9. Sync primitives

| Metric | Value | Unit | Conf | Source file | Doubt status |
|---|---:|---|---|---|---|
| `__syncwarp` full-mask | 1 (NOP) | cy | HIGH | F2_SYNCWARP_RIGOR; F6 | doubt-confirmed (V9's 23 cy was loop overhead) |
| `__syncwarp` partial mask (BSYNC) | 7.25 | cy | HIGH | F2_SYNCWARP_RIGOR | untouched |
| `__shfl_sync` broadcast idx=0 | ~2 | cy | HIGH | 08_sync §1 | untouched |
| `__threadfence_block` / membar.cta | 6–16 | cy | MED | F6/V9/08 (range) | DOWNGRADED (single-canonical-value framing retracted) |
| `__syncthreads(32 thr / 1 warp)` | 23–24 | cy | HIGH | V9_SYNCTHREADS_COST formula 22+2W | untouched |
| `__syncthreads(256 thr / 8 warp)` | 38 | cy | HIGH | 08 catalog | untouched |
| `__syncthreads(1024 thr / 32 warp)` | 86 (V9 formula) / 77 (08) | cy | MED | 08 vs V9 formula 12% gap | UNRESOLVED |
| `mbarrier.arrive` (no wait) | 24 | cy / 12 ns | HIGH | M7_V5 A6 | untouched |
| `mbarrier.arrive + test_wait` (1 thr) | 54 | cy / 26 ns | HIGH | 08 catalog | untouched |
| `mbarrier.arrive + wait` (block, single-thr loop) | 123 | cy / 60 ns | HIGH | V10_VERIFICATION_SUMMARY | untouched |
| `barrier.cluster.arrive.relaxed + wait` cluster=2 | 102 | cy / 50 ns | HIGH | 08 catalog cluster_raw_barrier | untouched |
| `cluster.sync()` strict | 370–380 | cy / 175–187 ns | HIGH | 08 catalog; V9 | untouched |
| `__threadfence` / fence.sc.gpu (idle, 4-way spread) | 281 ± 25 | cy | MED | 08; V9; DSMEM range 258–320 | UNRESOLVED (24% spread) |
| `__threadfence` chip-wide write traffic | 783 | cy / 385 ns | HIGH | 08 EXTENDED §1 | untouched |
| `__threadfence_system` | 1750–3042 | cy | DISPUTED | 08 (1750) vs DSMEM (2870) vs V9 (3042) | UNRESOLVED (1.74× spread; HEADLINE_v2 #34) |
| `fence.sc.sys` saturated chip | ~19000 | cy | HIGH | 08 sec 30.G | untouched |
| `grid.sync()` (148 blocks × 128 thr) | 2376 | cy / 1170 ns | HIGH | V10_GRID_SYNC | doubt-confirmed (79× syncthreads) |
| Cross-block flag wait (volatile + threadfence) | 1605 | cy / 790 ns | HIGH | 08 pingpong | untouched |
| fence.sc.cluster == fence.sc.gpu | 320 | cy | HIGH | DSMEM_REFERENCE rule 9 | untouched |

---

## 10. NVLink / PCIe

| Metric | Value | Unit | Conf | Source file | Doubt status |
|---|---:|---|---|---|---|
| NVLink generation | 5 (NV18) | gen | HIGH | 12_nvlink R1 (web-confirmed) | doubt-confirmed (top-10 #1; "v7" RETRACTED) |
| NVLink-5 spec per direction | 900 | GB/s | HIGH | 12_nvlink_CORRECTED | doubt-confirmed (757 was NVLink-4) |
| NVLink-5 spec bidi | 1800 | GB/s | HIGH | 12_nvlink | doubt-confirmed |
| NVLink-5 P2P read payload | 778 | GB/s | HIGH | 12_nvlink §3a; TRUE_REF | doubt-confirmed (86% of 900) |
| NVLink-5 P2P read NVLink RX (ncu, incl. protocol) | 860 | GB/s | HIGH | 12_nvlink §11 | doubt-confirmed (96% of 900) |
| NVLink-5 P2P write payload | 720 | GB/s | HIGH | 12_nvlink §3b | doubt-confirmed (80% of 900) |
| NVLink-5 P2P write NVLink TX (ncu) | 836 | GB/s | HIGH | 12_nvlink §11 | doubt-confirmed (93% of 900) |
| NVLink-5 P2P bidi aggregate | 1543 | GB/s | HIGH | 12_nvlink §2 (86% of 1800) | untouched |
| SM count to saturate NVLink | 32 | SMs | HIGH | 12_nvlink §3d | untouched |
| Per-SM unsaturated NVLink | ~38 | GB/s | HIGH | 12_nvlink | untouched |
| NVLink "820 GB/s = 91%" READ summary | UNRESOLVED | — | LOW | 12_nvlink §3c | UNRESOLVED (no source for 820) |
| Cross-GPU atomic latency | ~1.55 | µs | HIGH | 12_nvlink §4a | untouched |
| Cross-GPU fence drain adder | +17.8K | cy | HIGH | 12_nvlink §6 | untouched |
| `cudaDeviceEnablePeerAccess` cold | 131 | ms | MED | 12_nvlink §7 | untouched |
| `cudaIpcOpenMemHandle` cross-process | 56 | µs | HIGH | 12_nvlink | untouched |
| NCCL all-reduce floor | 10 | µs | HIGH | 12_nvlink §9 | untouched |
| Custom ring all-reduce floor | 21 | µs | HIGH | 12_nvlink §9 | untouched |
| PCIe link generation / width | Gen 6 x16 | — | HIGH | 13_pcie_CORRECTED | untouched |
| PCIe H2D pinned (≥64 MB) | 57.7 | GB/s | HIGH | 13_pcie | doubt-confirmed (90% Gen 5; 23% Gen 6 spec) |
| PCIe D2H pinned | 57.4 | GB/s | HIGH | 13_pcie | untouched |
| PCIe full-duplex aggregate | 98.8 | GB/s | HIGH | 13_pcie (1.72× single-dir) | doubt-confirmed (R5 corrected wording) |
| PCIe pageable H2D | 38.0 | GB/s | HIGH | 13_pcie | untouched |
| Async engines | 4 | engines | HIGH | cudaDevAttrAsyncEngineCount | untouched |
| D2D same device (2 GB) | 3279 | GB/s | HIGH | 13_pcie | untouched |
| H2D 1 B sync latency | 3.6 | µs | HIGH | 13_pcie + TRUE_REF | untouched |
| Persistent kernel + mapped poll (CPU↔GPU) | 2.03 | µs | MED | TRUE_REF NINJA dcc0f20 (ld.relaxed.sys) | DOWNGRADED (mechanism unverified — "v1 used release" is hypothesis) |
| Pageable migrates to GPU on first touch | 1.5 | TB/s | HIGH | TRUE_REF 00d971c | untouched |
| `HostNativeAtomicSupported` | 0 | — | HIGH | 13_pcie | untouched |
| PCIe Gen 6 cap mechanism | "CPU-bound" | — | LOW | TRUE_REF v1 | DOWNGRADED — RETRACTED by 13_pcie R2 (UNCONFIRMED root cause) |

---

## 11. Math intrinsics (MUFU / SHFL / REDUX)

| Metric | Value | Unit | Conf | Source file | Doubt status |
|---|---:|---|---|---|---|
| MUFU.EX2 chain latency (EX2→EX2) | 14.14 | cy | HIGH | V41; 14_math_CORRECTED | untouched |
| MUFU.EX2 cross-pipe (FFMA→EX2→FFMA) | ~30 | cy | MED | CHAIN_FP_MUFU_LATENCY | untouched |
| MUFU.EX2 throughput | 9.22 | Gops/s chip | HIGH | V41 | doubt-confirmed (95.8% of XU SoL; 2× faster than other MUFU) |
| MUFU.LG2/SQRT/RSQ.ftz/TANH chain latency | 18 | cy | HIGH | V41 | untouched |
| MUFU.LG2/RCP/RSQRT/SQRT/SIN/COS throughput | 4.74 | Gops/s chip | HIGH | V41 | doubt-confirmed (saturated peak; 47.8 was 1-chain) |
| MUFU.SIN/COS chain latency | 24.02 | cy | HIGH | V41 | untouched |
| MUFU.RSQ/SQRT (non-ftz IEEE) latency | 40.10 | cy | HIGH | V41 | untouched |
| MUFU.RCP latency | 42.10 | cy | HIGH | V41 | untouched |
| M14/M16 "XU peak 47.8 G/s" | REFRAMED | — | HIGH | M14/M16 | REFRAMED (real: 1-chain rsqrt latency-bound; saturated = 4.74 G; HEADLINE_v2 #4) |
| SHFL.BFLY (warp shuffle) chain latency | ~5 | cy | HIGH | V38; 14_math | untouched |
| SHFL.BFLY raw throughput | 9.48 | Telements/s chip | HIGH | V38 | untouched |
| `redux.sync.add.u32` chain latency | ~11.6 | cy | HIGH | Q3_WARP_REDUCE | untouched |
| `redux.sync.{add,min,max,...}.u32` raw throughput | 9.09 | Telements/s chip | HIGH | V37 | untouched |
| REDUX vs SHFL "4×" speedup | 2.34× | — | HIGH | Q3_WARP_REDUCE | doubt-confirmed (algorithmic; per-inst rates equal; "4×" is folklore) |

---

## 12. Integer / bit ops

| Metric | Value | Unit | Conf | Source file | Doubt status |
|---|---:|---|---|---|---|
| IMAD/IMUL theoretical (1:2 of FP32 @ 2032) | 38.48 | Tops | HIGH | CUDA C PG Table 13-1 | untouched |
| IMAD 32-bit measured | 38.4 | Tops | HIGH | V8_IMAD_PEAK_VERIFIED (99.7%) | doubt-confirmed |
| FFMA/FADD/FMUL pipe @ 1500 lock | 25–26 | Glane/s | HIGH | V40 | untouched |
| IADD3 pipe placement | FMA pipe | — | HIGH | V40 d1d09c5 | doubt-confirmed (top-10 #2; ALU placement RETRACTED) |
| IADD3 rate (V40) | 0.66 | inst/SMSP/cy | MED | V40 25–26 Glane/s | UNRESOLVED (V40 0.66 vs A6 0.50 = 30% gap) |
| IADD3 rate (A6/B1) | 0.50 | inst/SMSP/cy | MED | A6 14.13 TIPS | UNRESOLVED |
| LOP3.LUT (INT-bit half rate) | 18.7 | Glane/s | HIGH | V40; C3 verified | untouched |
| LOP3 rate | 0.50 | inst/SMSP/cy | HIGH | V40 (~48% of FMA peak) | untouched |
| PRMT (V40) | 13.9 | Glane/s | MED | V40 | UNRESOLVED (V40 0.36 vs A6 0.50) |
| PRMT (A6) | 14.08 | Glane/s | MED | A6 | UNRESOLVED |
| ISETP / FSETP (V40 compare pipe) | 8.4 | Glane/s | HIGH | V40 (0.25/SMSP/cy = 22%) | doubt-confirmed (catalog "19 TOPS" RETRACTED) |
| BFE.u32 | 7.07 | Glane/s | MED | A6 (XU pipe) | untouched |
| SHFL.{IDX,BFLY,UP,DOWN} | 7.06 | Glane/s | HIGH | A6 | untouched |
| POPC / BREV / CLZ | 4.7 | Glane/s | HIGH | XU 0.125 inst/SMSP/cy | untouched |
| Mixed FFMA + IADD3 "114 TOPS combined" | RETRACTED | — | — | 15_int_CORRECTED R | doubt-confirmed (real overlap 14–17%) |
| Same-warp dual-issue FFMA + LOP3 | 55 | % | LOW | V49 501134a | DOWNGRADED (DUAL_ISSUE_DOUBT entire; baseline only 67%-of-peak; no ncu) |
| Same-warp dual-issue FFMA + IADD3 | 54 | % | LOW | V49 | DOWNGRADED |
| Same-warp dual-issue FFMA + PRMT | 51 | % | LOW | V49 | DOWNGRADED |
| Warp-specialized FFMA + LOP3 (4+4 warps) | 74 | % | LOW | V50 fbe1c18 | DOWNGRADED (top-10 #6 needs ncu + warps sweep) |
| FFMA + LDG chained dual-issue | 1 | % | MED | B2 | untouched |
| FFMA + LDG independent | 12 | % | MED | B2 | untouched |
| FFMA + MUFU dual-issue | ~100 | % | MED | A6/8012b98 | untouched (M8 counter-evidence) |

---

## 13. Power / clock

| Metric | Value | Unit | Conf | Source file | Doubt status |
|---|---:|---|---|---|---|
| TDP enforced limit | 1100 | W | HIGH | nvmlDeviceGetEnforcedPowerLimit | doubt-confirmed (700 W RETRACTED) |
| Min power limit | 200 | W | HIGH | 16_power | untouched |
| True idle clock | 120 | MHz | HIGH | 16_power | untouched |
| Default boost (sustained FFMA) | 2031.4 | MHz | HIGH | clock64/globaltimer | untouched |
| `nvidia-smi -lgc 2032` actual | 1919.8 | MHz | HIGH | 16_power (paradox; 4× replicated) | doubt-confirmed |
| Stuck-without-lock pin | 1005 | MHz | HIGH | feedback_clock_stuck_no_lock | doubt-confirmed (#37) |
| Idle floor @ 510 MHz | 144 | W | HIGH | 16_power §2 | untouched |
| Idle floor @ 1920 MHz | 198 | W | HIGH | 16_power §2 | untouched |
| FFMA active @ 2032 boost (peak ILP=24) | 361 | W | HIGH | 16_power §2 | untouched |
| BF16 cuBLAS sustained random | 962 | W | HIGH | 16_power §2 (87% TDP) | untouched |
| DRAM-read random d=16 + 1500 MHz | 1071 | W | HIGH | POPCOUNT_VS_CLOCK | untouched (worst-case stress recipe) |
| Transient peak (NVML aliasing or real?) | 1259 | W | LOW | TRUE_REF v1 (862014c) | UNRESOLVED (16_power says 1100 cap) |
| FFMA energy (with .reuse, broadcast) | 2.2 | pJ/FLOP | HIGH | M2 H1 | untouched |
| FFMA energy (no .reuse, 3 unique reads) | 6.5 | pJ/FFMA | HIGH | M2 H9 | untouched |
| DRAM read energy (d=16) | 86.1 | nJ/byte | HIGH | POPCOUNT_WRITES | untouched |
| DRAM write energy (d=16) | 115.7 | nJ/byte | HIGH | POPCOUNT_WRITES | untouched |
| L2 read energy (d=16) | 25.5 | nJ/byte | HIGH | POPCOUNT_WRITES | untouched |
| L2 write energy (d=16) | 62.2 | nJ/byte | HIGH | POPCOUNT_WRITES | untouched |
| HMMA energy (BF16 mma.sync) | ~50 | pJ/output | MED | M11 | untouched |
| Min-energy clock pure FFMA | 510 | MHz | HIGH | M9_ENERGY_PARETO | untouched |
| Min-energy clock pure memory | 800 | MHz | HIGH | M9 | untouched |
| Min-energy clock realistic ML inference | 1992 boost | MHz | HIGH | M9 | doubt-confirmed (3× lower than 510) |
| Best FFMA GFLOPS/W | 134 | GFLOPS/W | HIGH | V10_DVS_CURVE (1500–1700 MHz) | untouched |
| nanosleep request 100 ns → actual | 113 | ns | HIGH | 16_power §8 | untouched |
| nanosleep request 1000 ns → actual | 620 | ns | HIGH | 16_power §8 | untouched (-38% undershoot) |
| nanosleep divergent semantics | MIN not MAX | — | HIGH | V9_NANOSLEEP_THREADS | doubt-confirmed |
| tcgen05 BF16 m128n128k16 A=B=0 floor | 287 | W | HIGH | 16_power §6 | untouched |
| tcgen05 random A & B (full random) | 609 | W | HIGH | 16_power §6 | untouched |
| Per-SM power scaling | 3.1 W/SM random / 1.0 W/SM constant | W/SM | HIGH | PER_SM_POWER_SCALING | untouched |
| Best practical low-power BF16 GEMM | 254 vs 610 = 58% reduction | W | HIGH | 16_power §6 | untouched |
| TCGEN05_PERF_WATTS single-trial table | CONTAMINATED | — | — | TCGEN05_POWER_CONSOLIDATED §2 | doubt-confirmed (5 leftover procs; use 2TRIAL clean) |
| NVFP4 K=96 perf/W (clean 2-trial random) | 12.54 | TF/W | HIGH | TCGEN05_PERFW_CLEAN_2TRIAL | doubt-confirmed (NOT 13.72) |

---

## 14. Latency ladder (compute / memory single-chain)

| Metric | Value | Unit | Conf | Source file | Doubt status |
|---|---:|---|---|---|---|
| Register read | 1 | cy | HIGH | catalog | untouched |
| L1 hit | 38–47 | cy | HIGH | D2/V10/03 | untouched |
| L1 → L2 transition warm | 130–200 | cy | HIGH | 03_caches | untouched |
| L2 hit avg | 300–310 | cy / 152–157 ns | HIGH | 03_caches §2.5 | untouched |
| L2 hit far-partition | ~660 | cy | HIGH | 03_caches | untouched |
| DRAM | 317 | cy | HIGH | catalog | untouched |
| FFMA latency | 4.04–4.22 | cy | HIGH | V9_OP_LATENCY | doubt-confirmed (legacy "23 cy" was BRA floor) |
| HMMA m16n8k16 F32 | 20.09 | cy / 9.8 ns | HIGH | V9_HMMA_LATENCY | doubt-confirmed |
| DFMA latency | 64 | cy | HIGH | V8_FP64_PEAK | untouched |
| Local atomic L2 RT (no chain) | 164 | ns | HIGH | TRUE_REF | untouched |
| Local atomic chained dependency RT | 343 ns / 697 cy | — | HIGH | V9_ATOMIC_LATENCY | doubt-confirmed |
| Cross-GPU atomic NVLink P2P | 1662 | ns | HIGH | TRUE_REF | untouched |
| Self-op `FFMA Ra,Ra,Ra,RZ` | 4.02 | cy | HIGH | 04_fp32 retracted "8.46 cy" | doubt-confirmed |

---

## 15. Launch overhead / API costs

| Metric | Value | Unit | Conf | Source file | Doubt status |
|---|---:|---|---|---|---|
| `<<<1,1>>>` direct CPU enqueue | 1.78–1.85 | µs | HIGH | TRUE_REF be28c14 | untouched |
| `cudaLaunchKernel` async no-sync | 1.85 | µs | HIGH | 10 catalog (grid-invariant 1→1M blocks) | untouched |
| Single-kernel cudaGraphLaunch | 2.05 | µs | HIGH | V9_GRAPH_LAUNCH | doubt-confirmed (NO speedup vs direct; myth busted) |
| 100-kernel cudaGraphLaunch (amortized) | 0.54 | µs/kernel | HIGH | V9_GRAPH_LAUNCH (3.84× vs direct) | untouched |
| 1000-kernel cudaGraph (amortized) | 0.56 | µs/kernel | HIGH | 10 catalog (3.7×) | untouched |
| `cuStreamWriteValue32` doorbell | 0.45 | µs | MED | CLAUDE memory V7 | UNRESOLVED (vs 10 catalog 2.47 µs full-pair) |
| `cuStreamWaitValue32` (already met) | 1.65 | µs | HIGH | 10 catalog | untouched |
| Persistent kernel + mapped-mem (CPU↔GPU) | 4 (or 2.03 with ld.relaxed.sys) | µs | MED | TRUE_REF 584fda6 / dcc0f20 | DOWNGRADED (mechanism for 2.03 µs unverified) |
| Persistent kernel batched task | 38 | ns/task | MED | CLAUDE V7 memory | untouched (not re-verified V8/V9) |
| `cudaMemset` (4 B) | 1.22 (TRUE_REF) / 1.4 (09 floor) | µs | MED | TRUE_REF be28c14; 09_memory_apis | UNRESOLVED (~14% gap) |
| `cudaMemcpyAsync` submit | 1.2 | µs | HIGH | TRUE_REF c6e7fc1 | untouched |
| `cudaMemcpy` sync small | 3.6 | µs | HIGH | TRUE_REF | untouched |
| `cudaStreamSynchronize` per launch | 7 | µs | HIGH | TRUE_REF | untouched |
| `cudaGraphInstantiate` 10 nodes | 11.3 | µs | HIGH | 10 catalog | untouched |
| `cudaGraphInstantiate` 100 nodes | 35 | µs | HIGH | 10 catalog | untouched |
| `cudaGraphExecKernelNodeSetParams` (1 node) | 0.30 | µs | HIGH | 10 catalog | untouched |
| `cudaGraphExecUpdate` 10 nodes | 0.145 | µs | HIGH | 10 catalog (77× vs reinstantiate) | untouched |
| `cudaGraphExecUpdate` 100 nodes | 1.4 | µs | HIGH | 10 catalog (25×) | untouched |
| Concurrent kernel HW slots | 128 | slots | HIGH | V10_CONCURRENT_KERNELS 7407cba | doubt-confirmed |
| Spill cliff | 9.25× @ 32 vars | — | HIGH | V9_REGSPILL_COST | doubt-confirmed |
| 2-way TRUE branch divergence cost | 2.57× | — | HIGH | V9_BRANCH_DIVERGENCE | doubt-confirmed (V9 "1.09×" was PREDICATED) |

---

## Summary statistics

Total rows graded: **303**.

| Confidence | Count | Approx % |
|---|---:|---:|
| HIGH | 259 | 85.5% |
| MED | 31 | 10.2% |
| LOW (incl. LOW-MED) | 11 | 3.6% |
| DISPUTED | 2 | 0.7% |

Doubt-status breakdown (308 status entries — some rows omit, some have multiple):
- `doubt-confirmed`: 81 rows (wave-3b agreed with the value)
- `DOWNGRADED`: 14 rows (methodology issues found by wave-3b)
- `REFRAMED`: 2 rows (value real, framing/% denominator wrong)
- `UNRESOLVED`: 16 rows (open contradictions, ≥2 incompatible values)
- `RETRACTED`: 2 explicit retracted rows + several inline retracted-rather-than-listed claims
- `untouched`: 192 rows (no doubt report directly on this metric)

---

## Top-10 LOW / DISPUTED items needing re-measurement

1. Same-warp dual-issue 55% / warp-spec 74% (V49/V50) — needs warps/SMSP sweep + ncu
2. `__threadfence_system` (1750 vs 2870 vs 3042 cy = 1.74× spread)
3. HBM write SoL 7.57 TB/s — provenance contested (NINJA STG vs TMA bulk)
4. DSMEM read aggregate 40 GB/s — chain-bound, non-chained ILP could be 60-80
5. DSMEM write aggregate 560 GB/s — issue rate, not completion (no fence)
6. DSMEM "NO shared bus" — V17 was 30× under-issued
7. `__threadfence` GPU 281 cy — 4-way 24% spread (258/281/292/320)
8. IADD3 rate 0.50 vs 0.66 inst/SMSP/cy (V40 vs A6) — 30% gap
9. PRMT rate (V40 0.36 vs A6 0.50) — methodology delta
10. PCIe Gen 6 cap mechanism — "CPU-bound" RETRACTED, root cause UNCONFIRMED

---

## How to use this ladder

- **Quick answer with confidence:** look up the metric, read Conf column.
- **Citing in code/docs:** if Conf=HIGH and doubt-status=`doubt-confirmed`, quote freely.
- **If MED:** quote with the qualifier from the Doubt status column.
- **If LOW or DISPUTED:** do NOT cite as authoritative; mark as unverified and re-run with `./utils/rigor_run.sh`.
- **For NEW measurements:** add a row here once 3-method verified.

Source files referenced are all in `b300_clean/corrections/` unless prefixed (e.g., `TRUE_REF` = `B300_TRUE_REFERENCE_v2_DRAFT.md`, `V9_*` / `V10_*` = original investigation files in `b300_clean/`).
