# B300 TRUE REFERENCE v2 — DRAFT (2026-04-22)

**Proposed replacement** for `b300_clean/B300_TRUE_REFERENCE.md` (2026-04-18).
Incorporates the 20-sub-agent audit (`b300_clean/corrections/`) and findings
from V11–V51 that postdate v1. Originals NOT modified; this is a draft only.

System: NVIDIA B300 SXM6 AC (sm_103a, CC 10.3, 148 SMs, 12 HBM3E stacks).
Default sustained boost: 2032 MHz (silent stuck-at-1005 failure mode is
real; sample clock during long runs and recover with `nvidia-smi -rgc -i N`).
CUDA 13.2 runtime / 13.0 driver (580.126.09).

Each row cites the corrected source file. **Supersedes** pointers indicate
which v1 entries are replaced.

---

## 1. Memory bandwidth ladder

| Memory | Peak (TB/s) | % theoretical | Source | Supersedes |
|---|---:|---:|---|---|
| **HBM3E read (canonical NINJA)** | **7.30** | 95.0% of 7672 GB/s post-ECC | `01_hbm_bandwidth_CORRECTED.md`, NINJA IT=2 | v1 row "7.30 95%" — kept |
| **HBM3E read (TMA bulk)** | 7.344 | 95.7% | `01_hbm_bandwidth_CORRECTED.md` sub-optimal table | new in v2 (was hidden in 01) |
| **HBM3E read (LDG.E.128 SoL)** | 7.365 | 96.0% | `01_hbm_bandwidth_CORRECTED.md` | new in v2 |
| **HBM3E read (V46 TMA 8-deep pipelined)** | 7.20 | 93.85% of 7672 | `HBM_INCONSISTENCY_LOG.md` #5 | **supersedes** "98.5% NEW BEST" framing — V46 is BELOW V44/V45 ceilings; the 98.5% used 7.31 (empirical) as denominator instead of 7.672 (post-ECC spec) |
| **HBM3E write NINJA (1 v8 store/warp)** | 7.57 | **98.7%** ← SoL | TRUE_REF v1 NINJA `e75c7e1`; **provenance contested** with V8_HBM_WRITE_SOL TMA path — see `HBM_INCONSISTENCY_LOG.md` #3 | unresolved |
| **HBM3E write (v8 + per-warp coalesced)** | 7.30 | 95.2% | `01_hbm_bandwidth_CORRECTED.md` | kept |
| **HBM3E concurrent R+W (50:50)** | 6.68 | 87% — minimum at any mix | `01_hbm_bandwidth_CORRECTED.md` A6 R:W sweep | kept |
| **D2D copy (separate src/dst)** | 6.93 | 90.3% | NINJA recipe `4958d6b` | kept |
| **L2 kernel-effective** | 23.85 | (incl. L1 reuse) | `03_caches_CORRECTED.md`, v8 + 8-ILP @ 96 MB | kept; **v2 makes "kernel-effective" label MANDATORY** |
| **L2 wire (lts__t_bytes)** | 13.30 | (pure L2 BW) | `03_caches_CORRECTED.md`, same kernel | kept; mandatory label |
| **L1-amplified small-WS peak** | ~30 | NOT L2; LSU dispatch | `03_caches_CORRECTED.md` | new label in v2 — was mis-listed as L2 in `03_caches.md` §3c |
| **L1 BW (strided default-ld)** | 30.5 | conservative measured | `V8_L2_BW_VERIFIED.md` | supersedes M5's "46 TB/s" (kept as ILP-max upper bound) |
| **TMEM read** | ~60 | (chip aggregate) | `06_tensor_cores_CORRECTED.md` | **supersedes** older catalog 295 / 830 TB/s (DCE-inflated) |
| **TMEM write** | 97-131 | | `06_tensor_cores_CORRECTED.md` | new |
| **SHMEM read peak** | 38.4 | 99.8% of 38.5 spec | `02_shmem_CORRECTED.md`, LDS.128 + RAW + 1blk/SM @ 2032 | kept |
| **SHMEM stmatrix W+R** | 34.5 | 90% | `02_shmem_CORRECTED.md` | kept |
| **DSMEM read aggregate** (per cluster) | **0.040** | — | `DSMEM_CORRECTED.md` (V11–V31) | **supersedes** v1 "DSMEM 3.06 TB/s aggregate" — that was a sum across clusters of mixed measurements; per-cluster ceiling is 40 GB/s read |
| **DSMEM write aggregate** (per cluster) | **0.560** | (writes are FAST, not slow) | `DSMEM_CORRECTED.md` | **supersedes** V10 "writes 4× slower" — direction was inverted |
| **TMA multicast effective (cluster=8)** | 14.91 | single-deep, single engine/cluster | V32; V48 confirms cannot pipeline | new in v2 |
| **NVLink-5 P2P read (payload)** | **0.778** | **86% of 900 GB/s/dir spec** | `12_nvlink_p2p_CORRECTED.md` | **supersedes** v1 "1.04× spec 757" — wrong denominator (NVLink-4 spec) |
| **NVLink-5 P2P read (link RX, ncu)** | 0.860 | 96% incl. protocol | ncu `nvlink__data_received` | new |
| **NVLink-5 P2P write (payload)** | **0.720** | 80% of 900 spec | `12_nvlink_p2p_CORRECTED.md` | **supersedes** v1 "0.94× spec 757" |
| **NVLink-5 P2P write (link TX, ncu)** | 0.836 | 93% of spec | ncu `nvlink__data_transmitted` | new |
| **NVLink-5 P2P bidi aggregate** | 1.543 | 86% of 1800 spec | symmetric | new |
| **PCIe Gen 6 x16 H2D pinned** | 0.0577 | 90% of Gen 5 / 23% of Gen 6 spec | `13_pcie_system_CORRECTED.md` | "CPU-bound" hypothesis from v1 RETRACTED — root cause unconfirmed |

**v2 rules**: always state denominator (spec=7672 post-ECC, NOT 7.31 empirical NOR 8.0 nominal); always label L2 BW with one of {kernel-effective, wire/lts, L1-amplified}.

---

## 2. Compute peaks ladder

| Operation | Peak (TFLOPS) | % theoretical | Notes / source | Supersedes |
|---|---:|---:|---|---|
| **FP32 FFMA peak (2-source recipe, 2032 MHz unlocked)** | 75.9 | 98.7% | `04_fp32_peak_CORRECTED.md` `fp32_peak_definitive.cu` | kept; v1 row 74.62 was a slightly older measurement, all within 1.5% noise |
| **FP32 FFMA realistic (3-distinct-source GEMM)** | ~50 | ~65% | `COMPUTE_INCONSISTENCY_LOG.md` #G — 2-RF-port theory | NEW in v2 — v1 hid this caveat |
| FP32 FFMA (locked 1920 MHz) | 62.17 | 85.5% | (1920 boost-pin paradox) | kept |
| **FP64 DFMA** | 1.20 | 100% of spec | `V8_FP64_PEAK_VERIFIED.md` | **supersedes** D4's 1.0 (84%) — sub-saturated regime |
| **IMAD / IMUL** | 38.5 Tops/s = 1:2 of FP32 | 99.7% | `V8_IMAD_PEAK_VERIFIED.md` | kept (older "1:1 FP32" assumption RETRACTED) |
| **FADD = FMUL = FFMA SASS rate** | 1 inst/SMSP/cy, 4.22 cy lat | — | `V8_FADD_FMUL_PEAK.md` | NEW in v2 — v1 didn't list FADD/FMUL |
| **FP16/BF16 mma.sync m16n8k16 (1920 MHz locked)** | 569 | matches catalog burst | v1 `a37d989` | kept |
| **FP16/BF16 mma.sync m16n8k16 (2032 boost)** | 578.6 | 99.9% | `V8_HMMA_F16_PEAK.md` | NEW row — v1 had only 569 |
| **BF16 cuBLAS GEMM N=8192 (zero data, sustained)** | 2242-2259 | 90% of 2500 spec | `06_tensor_cores_CORRECTED.md` | kept |
| **BF16 cuBLAS GEMM N=8192 (random data)** | 1850-1905 | 75-77% | `06_tensor_cores_CORRECTED.md` | NEW headline — v1 hid this in the data-dep addendum |
| **BF16 microbench tcgen05 direct** | 2325 | (peak compute ceiling) | `06_tensor_cores_CORRECTED.md` | NEW |
| **FP8 e4m3 cuBLAS LtMatmul (zero, sustained via cudaGraph)** | 4400-4500 | 88-90% of 5000 spec | `06_tensor_cores_CORRECTED.md` | kept; v1 row "4425" within noise |
| **FP8 e4m3 cuBLAS LtMatmul (random data)** | 3983 | 80% — REALISTIC | `06_tensor_cores_CORRECTED.md` | kept |
| **FP8 e4m3 cuBLAS under 600W power cap (random)** | 3087 | -43% from peak | TRUE_REF v1 row | kept |
| **FP8 microbench tcgen05 direct** | 4651 | (ceiling) | `06_tensor_cores_CORRECTED.md` | NEW |
| **NVFP4 cuBLAS plain Lt (boost, K=38400)** | 11068 | 73.8% of 15 PF | `NVFP4_CONSOLIDATED.md` | **supersedes** v1 row 10297 |
| **NVFP4 cuBLAS + cudaGraph BPG=16 (boost, K=38400)** | **11423** | **76.2% of 15 PF** ← record | `NVFP4_CUDAGRAPH.md` | NEW |
| NVFP4 cuBLAS sustained random (15s, throttle to 1057) | 6554 | 65.5% | v1 `23be661` | kept |
| **NVFP4 K=96 ULTRA tcgen05 microbench (1500 MHz lock)** | **10890** | 98.5% per CTA | `NVFP4_K96_AT_1500MHZ.md` | NEW — confirms K=96 path is real |
| **NVFP4 K=96 ULTRA tcgen05 microbench (boost, all-zero, zero-skip)** | 14780 | 98.5% per CTA | NVFP4_K96_AB_FULL addendum | NEW (best-case) |
| CUTLASS C++ NVFP4 (sample 89, boost, 8K² K=15K) | 8285 | ~55% | NVFP4 consolidated | kept |
| CuTeDSL NVFP4 (boost, 8K² K=15K) | 9118 | 60.8% | NVFP4 consolidated | kept |
| **2-GPU NVFP4 split (1 stream/GPU, no comm)** | 19163 | 95.8% of 2× spec | v1 `cbaadbc` | kept |

**v2 rules**: every TFLOPS row must annotate **(zero/const | random | normal)** AND **(per-call | sustained-via-cudaGraph)** AND **(boost | -lgc N MHz)**. cuBLAS spec = 5000 FP8 / 2500 BF16 / 10000 NVFP4 base / 15000 NVFP4-K96-ULTRA — cite which.

**Retracted in v2**:
- v1 row 58 "BF16 mma.sync 8-chain ~570" with note "single-chain 1543 was over-counted" → kept.
- "BF16 mma.sync 90.5% of 2500 spec" RETRACTED (was 23% of 2500 tcgen05 spec, or 93.7% of 616 legacy spec).
- "FP8 mma.sync 7500-8200 TFLOPS" RETRACTED (SASS showed HMMA.16816 not 16832; real ~3760).
- "256 cores per SM / 154 TFLOPS" RETRACTED (B300 has 128 FP32 cores per SM).

---

## 3. Coordination latency ladder

| Mechanism | Latency | Source | Supersedes |
|---|---:|---|---|
| `__syncwarp` | 1 ns / 1 cy | `08_sync_primitives_CORRECTED.md`, F2/F6 authoritative | **supersedes** V9's "23 cy" (loop overhead, mislabeled) |
| `__threadfence_block` | 8 ns / 16 cy | `08_sync_primitives_CORRECTED.md` | kept (V9 "≈0" captures only post-issue cost) |
| `__syncthreads (256 thr)` | 14 ns | v1 row | kept |
| `__syncthreads (1024 thr)` | 38-42 ns / 77-86 cy | `08_sync_primitives` (77) vs V9 formula 22+2W (86) | 12% gap unresolved |
| `cluster.barrier::arrive (relaxed)` | 50 ns | v1 row | kept |
| `mbarrier.arrive` only | 24 cy / 12 ns | `08_sync_primitives_CORRECTED.md` | NEW row |
| `mbarrier.arrive + wait` | 123 cy / 60 ns | `V10_VERIFICATION_SUMMARY.md` | NEW row (was missing from v1's "57.7 ns/cycle" framing) |
| `cluster.sync` | 175-187 ns / 370-395 cy | `08_sync_primitives_CORRECTED.md` | kept |
| `__threadfence` (device GPU) | 138-144 ns / 281 cy | `08_sync_primitives_CORRECTED.md`, V9 | kept (v1 said 385 ns — likely included extra overhead) |
| `__threadfence_system` | **1750–3042 cy / 861–1486 ns — UNRESOLVED 1.74× spread** | 08 (1750) vs DSMEM (2870) vs V9 (3042) | **supersedes** v1 single "861 ns" |
| **Local atomic L2 round-trip (no chain, near-L2)** | 164 ns | v1 `ad19660` — **annotation added** | kept; v2 makes "(no chain, near-L2)" mandatory |
| **Local atomic L2 chained dependency RT** | 343 ns / 697 cy | `V9_ATOMIC_LATENCY.md` | NEW row — both numbers correct, different definitions |
| Cross-GPU atomic via NVLink P2P | 1662 ns | v1 `ad19660` | kept |
| Persistent kernel + mapped memory | **2.03 µs** ← CPU↔GPU lowest | TRUE_REF v1 NINJA `dcc0f20` (`ld.relaxed.sys`, NOT acquire/release) | **supersedes** v1 "4 µs" — that used release variant emitting MEMBAR.ALL.SYS |
| `cudaMemcpy` sync (small) | 3.6 µs | v1 row | kept |
| `cudaStreamSynchronize` per launch | 7 µs | v1 row | kept |

---

## 4. API costs

(Largely matches v1; only changes noted.)

| API | Cost | Source |
|---|---:|---|
| `cudaGraphLaunch` (1-kernel) | 2.05 µs ≈ direct 2.06 µs (NO speedup for single launch) | `V9_GRAPH_LAUNCH.md` |
| `cudaGraphLaunch` (100-kernel batch) | dominant win at N≥100 (see M16) | M16 |
| `cuStreamWriteValue32` (host-call only) | 0.45 µs | CLAUDE.md V7 memory — **conflicts** with 10_launch_overhead `2.47 µs` (full pair); needs reconciliation |
| `cudaMemset` (4 B) | 1.22 µs (TRUE_REF v1) — but 09_memory_apis floor = 1.4 µs (~14% gap) | unresolved |

(Other v1 API rows preserved unchanged.)

---

## 5. Atomic throughput

| Configuration | Gops/s | Source | Supersedes |
|---|---:|---|---|
| Stride 0 (full collision) | 0.79 | v1 | kept |
| Stride 4 (cache-line combining, UNROLL=16) | 504 | `07_atomics_CORRECTED.md` §8 | refines v1's "449" |
| **Stride 4 true peak (UNROLL=32, L2-resident)** | **1005** | `07_atomics_CORRECTED.md` §8 | **NEW headline** — v1 "449 Gops/s peak" was a lower-UNROLL artifact |
| Stride 32 (1 line/thread) | 184 | v1 | kept |
| Stride 256+ (scattered) | ~150 | v1 | kept |

**v2 rule**: every Gatomic/s number MUST publish (combine, WS, L2-resident, DRAM B/s). v1's bare "449 peak" violates this.

| Op | Cost (ns) | Notes |
|---|---:|---|
| atomicInc / Dec | 4 | v1 |
| atomicAdd FP32 / Min / Max | 7-8 | v1 |
| atomicAnd / Or / Xor | 11 | v1 |
| atomicAdd FP64 (HW path) | 4.5 | v1 |
| atomicExch / atomicCAS | 24-26 | v1 |
| atomicAdd scalar half / bfloat16 (NO HW) | 700 — 200× slower | v1 |
| atomicAdd packed half2 / bfloat162 (HW) | 16/elem | v1 |
| **red.release.gpu.global** | 614 ns/op (MEMBAR.ALL.GPU between each, **NOT CCTL.IVALL**) | v1 + cause-attribution fixed: `ATOMICS_INCONSISTENCY_LOG.md` A10 |

---

## 6. Compute pipe ladder (NEW in v2)

Authoritative pipe placement and per-op rates @ 2032 MHz (V40 + A6 + V49 + V50 + B1 + B2):

| Op | Pipe | Per-SMSP rate | Chip Tops/s (2032) | Notes |
|---|---|---:|---:|---|
| FFMA / FADD / FMUL | FMA | 1.0 inst/cy | 75 (FFMA FLOPS), 37 inst/s | 2-source 97%, 3-source 65% |
| IMAD / IMUL | FMA (1:2) | 0.5 inst/cy | 38.5 | matches V8 |
| **IADD3** | **FMA pipe (V40)** — NOT separate ALU | 0.66 inst/cy (V40) — A6 says 0.5; UNRESOLVED 30% gap | 26 Glane/s | **supersedes** "separate ALU 38 TOPS" |
| LOP3 | INT-bit (half rate) | 0.50 inst/cy | 19 | |
| PRMT | permute | 0.36 inst/cy (V40) — A6 says 0.5; UNRESOLVED | 14 | |
| ISETP | compare | 0.22 inst/cy | 8.4 | catalog 19 TOPS was wrong |
| POPC / BREV / CLZ | XU | 0.125 inst/cy | 4.7 | consistent |
| MUFU rsqrt / rcp / log / sin / cos | XU (1/4 cy issue) | 0.25 inst/cy/SMSP | **4.74 G/chip saturated** | 47.8 G is 1-chain LATENCY-bound, NOT pipe peak |
| MUFU EX2 (outlier) | XU | 0.5 inst/cy | 9.22 G | 2× faster than other MUFU |
| SHFL | LSU | 0.25 inst/cy/SMSP | ~7-9 G (instruction vs element clash) | |
| LDS.128 | LSU | 1 inst/cy/SMSP @ proper recipe | 38.4 TB/s | |

**Dual-issue ladder** (V49/V50/B1/B2/A1/A6 — supersedes "free" / "100%" / "114 TOPS" claims):

| Pattern | Overlap |
|---|---:|
| FFMA + IADD3 same warp | **54-55%** (V49) |
| FFMA + LOP3 same warp | 55% (V49) |
| FFMA + LOP3 warp-specialized | **74%** (V50) — best practical |
| FFMA + MUFU (fast pipe + slow pipe) | ~100% (MUFU's 1/(4cy) gaps swallow it) |
| FFMA + LDG chained | ~1% (LSU back-pressure) |
| FFMA + LDG independent | ~12% |
| FFMA + IMAD same SMSP cluster | -14% (negative — slower than serial) |

---

## 7. Architecture facts

| Fact | Value | Source | Supersedes |
|---|---|---|---|
| **GPCs** | **8 GPCs** (2×20 + 6×18 = 148 SMs) | `11_block_scheduling_CORRECTED.md`, TOPOLOGY log #1/8 | "8×~18 + 4 spare" / "9 GPC-rows" / "10 GPCs" — all RETRACTED |
| L2 capacity | 126 MB (132 644 864 B) | CACHES log | RETRACTED: 50 / 192 / 256 / 280 MB (unit/scope confusions) |
| L2 partitions | 2, split by die boundary; 2.4× near-vs-far latency | v1 | kept |
| L2 sector | 128 B line, 32 B sector (4/line) | D3 | universal agreement |
| L2 atomic units | ~32 plateau (UNRESOLVED — could be higher per ATOMIC_REVERIFY_DEEP) | A8/A5 | demoted to MED confidence |
| L1 capacity | unified pool 256 KB; effective L1 20-228 KB depends on (carveout, access pattern) | CACHES log | RETIRED bare "L1 = 32 KB" |
| TMEM | 256 KB / CTA = 38 MB chip-wide | CURIOSITY V4 D7, 06_CORRECTED | kept |
| Max usable cluster | **16 (non-portable), 8 portable, ≥32 silent no-op** | TOPOLOGY log #4 | universal |
| SHMEM/SM | 228 KB total; 227 KB opt-in/block; 1024 B reserved | v1 | kept |
| Concurrent kernel slots | **128 (HW slots, NOT CTAs)** | V10_CONCURRENT_KERNELS, `7407cba` | clarification added |
| Pageable memory | MIGRATES to GPU on first touch at 1.5 TB/s | v1 `00d971c` | kept |
| **NVLink** | **NVLink-5 (Blackwell), 18 links total full-duplex (NV18)** | NVLINK_PCIE log #1, #4 | RETRACTED "NVLink v7" / "18 per direction" |
| **NVLink spec** | **900 GB/s/dir** (= 18 × 50) data; 956 raw | NVLINK_PCIE log #10 | RETRACTED 757 (NVLink-4) |
| PCIe | Gen 6 x16 physical; effective ~57.7 GB/s ("CPU-bound" hypothesis UNCONFIRMED) | 13_pcie | kept measurement; cause hypothesis withdrawn |
| **Power** | min 200 W / TDP 1100 W enforced; sustained avg ceiling 1093 W; **transient 1259 W (UNRESOLVED — NVML aliasing or real?)** | POWER log B | kept; flag uncertainty |

---

## 8. NEW SECTIONS (post-v1 V11–V51 findings)

### 8.1 DSMEM exhaustive characterization (V11–V31, supersedes V8/V10 wholesale)

Per `DSMEM_CORRECTED.md`:

- **All V8/V10 DSMEM TB/s peaks were DCE artifacts.** Real per-cluster aggregates are 40 GB/s read / 560 GB/s write.
- **Writes are FAST, not slow.** V10's "writes 4× slower" had direction inverted.
- **Local/DSMEM latency ratio = 7.5×** (NOT 0.8% from b478bb0 — that was LICM; NOT 4.7× from dsmem_v2 — wrong baseline).
- **Cluster=2 is 21% slower than cluster≥3** (single-GPC vs multi-GPC routing — DSMEM_FINDINGS_V2 V12 cliff between cx=2 and cx=3).
- **Reads pair-dependent (25% spread); writes pair-uniform (3% spread).** Best pair SM32↔SM33 = 164.8 cy; worst SM16↔SM17 = 204.97 cy.
- **No shared bus**: N=1..8 ring readers = 1.00× flat throughput.
- **fence.sc.cluster == fence.sc.gpu == 320 cy** (only fence.sc.sys at 2870 cy is expensive).
- **TMA multicast = 470 GB/s effective at 32 KB tile** — preferred primitive over raw DSMEM reads for cluster data movement.
- **TMA + DSMEM = 0.04% interference** (independent paths).
- Hot-spot atomics scale linearly to **63 Matom/s @ N=8** (atomic unit pipelined at 33 atoms/clock; the 15 GB/s "serving port" cap doesn't apply).
- **Crash workarounds in 02_shmem.md / 04_dsmem_overhead.md ("CHAIN_LEN≥15 crashes") are NO LONGER REPRODUCIBLE** (V12/V26: 30/30 success at CL=50). Treat as historical.

### 8.2 TMA pipelining and multicast (V32–V48)

| Operation | Best | Source | Note |
|---|---|---|---|
| TMA single-deep read | 6.72 TB/s | V33 | per-CTA |
| TMA 8-deep pipelined read | 7.20 TB/s | V46 | NOT a new SoL (LDG.E.128 already at 7.365); is a real improvement over single-deep |
| TMA bulk read (geometry-tuned) | 7.344 TB/s | 01 sub-optimal table | unverified pipeline depth |
| TMA write single-deep | 7.17 TB/s = 98% (denominator 7.31) | V34 | |
| TMA write 8-deep pipelined | 6.34 TB/s — NO BENEFIT | V47 | rule: writes already async, don't pipeline |
| TMA multicast cluster=8 single-deep | 14.91 TB/s effective | V32 | one engine per cluster |
| TMA multicast 2-deep | 13.96 TB/s — capped | V48 | cannot pipeline, single engine |
| **prefetch.L2 + cp.async.bulk (TMA)** | **27% SLOWER** | V42 | rule: NEVER combine prefetch.L2 with cp.async.bulk |
| prefetch.L2 + old `cp.async` (LDGSTS) | 1.58× FASTER | V6 I3 | different instruction; do NOT confuse |

### 8.3 ALU pipe ladder (V40-V50)

(See §6 above — full pipe placement and dual-issue tables.)

### 8.4 Bank conflict regimes (V44/V45)

| Regime | 32-way conflict cost |
|---|---:|
| Latency-bound single chain (V44) | ~2× |
| Throughput-bound, many warps with overlap (V45) | ~1× (hidden) |
| Single-warp dep chain (D5) | 5.74× |
| Single-warp transpose (Q6) | 8.2× |
| Multi-warp contention (`bce8bf8`) | 8.81× |

**v2 rule**: never quote a single "N-way conflict slowdown" without specifying the regime. The "scales N/4" rule of thumb is wrong outside the multi-warp-contention regime.

### 8.5 Dual-issue limits (V49/V50)

(See §6 — supersedes "FFMA + IADD3 free / 100% / 114 TOPS combined" claims.)

### 8.6 tcgen05 sub-tile dedup unified model

Per `TCGEN05_DEDUP_CONSOLIDATED.md`:

```
P(MMA) = P_baseline (~280-305 W per CTA precision-dep)
       + Σ over HW sub-tiles (B-side, 32-byte each): sticky activation
            0                                       if byte-identical to active slot
            ~32 W activation + ~18 W per broken byte otherwise (BF16 m128n128)
       + Σ over K iterations (B K-vary cost):
            5-25 W per added unique K row pattern   // sub-linear in K count
       + ε(A varying) only when B varies            // A is broadcast → ~0-60 W marginal
```

**32-byte universal HW sub-tile boundary**: BF16 cliff at N_unique=17, FP8 at 33, NVFP4 at 65 (all = 32 bytes worth).

**A vs B asymmetry: 3 different "correct" answers** depending on test geometry (cuBLAS A>B 3:1 because TMA multicasts B; pure tcgen05 B>>A 15-30×; K=96 single-kernel B>A 2.6×).

**Retracted from earlier**: "4-slot pattern cache" (R1), "K-uniform 28% saving" (R2 — was background-process contamination; real 1-3%), "diagonal patterns LOW universally" (R5 — depends on popcount invariance), "2:4 sparsity hurts dense GEMM" (R4 — actually helps +11%).

### 8.7 cuBLAS K-id 5-gate model (BF16 / FP8)

ALL FIVE conditions required for full 1.42× BF16 / 1.55× FP8 speedup:
1. N ∈ {K/2, K, 2K} (razor-sharp — off by 32 = total loss)
2. N divisible by 256
3. K divisible by 256
4. transB=0 layout (NN/TN, not NT/TT)
5. Data has chunk=1 alternation OR chunk divides 64 OR period 1/2

Real ML workloads satisfy NONE simultaneously → 2-11% practical benefit only. Sub-tile dedup contributes only 1-3% in cuBLAS workloads (K-row dedup dominates by 14× margin).

---

## 9. Counterintuitive findings (consolidated)

(Largely matches v1 §7; net additions noted.)

- v1 #1–#15 preserved.
- v1 #16 "Branchless 35% slower than if-else" — kept.
- v1 #17 "`__noinline__` 14× slower" — kept.
- v1 #18 "L2 atomic units = 32" — **demoted to MED confidence** (ATOMIC_REVERIFY_DEEP says ceiling could be much higher).
- v1 #19 (MLOPart MPS) — kept.
- v1 #20 "CCTL.IVALL doesn't exist on B300" — kept; ATOMICS log A10 confirms cause of red.release slowdown is MEMBAR.ALL.GPU (not CCTL.IVALL).
- v1 #21–#32 (tcgen05 dedup findings) — kept; consolidated mechanism in §8.6 above.
- **NEW #33**: NVLink is 5th gen, NOT v7. (NVLINK log #1)
- **NEW #34**: 2-pattern (ABAB) sub-tile is WORSE than 3-pattern (ABCABC). UNRESOLVED. (DEDUP log B4)
- **NEW #35**: `pipe_tensor.cycles_active` does NOT measure tcgen05 ops. (TENSOR log §B)
- **NEW #36**: cluster=2 DSMEM 21% slower than cluster≥3 (single-GPC routing). (DSMEM log E)
- **NEW #37**: B300 can stick at 1005 MHz under load with NO explicit lock; nvidia-smi -q won't show it. Sample clock during runs and use `-rgc` to recover. (CLAUDE memory `feedback_clock_stuck_no_lock`)
- **NEW #38**: Persistent kernel + `ld.relaxed.sys` (not acquire/release) gives 2.03 µs CPU↔GPU vs 4 µs prior. The MEMBAR.ALL.SYS in acquire variant adds 1.5 µs.
- **NEW #39**: 3-source FFMA caps at 65% of 2-source peak (RF port pressure). Realistic GEMM ≈ 50 TFLOPS, not the 75 headline.
- **NEW #40**: V9 "branch divergence 2-way = 1.09×" was PREDICATED. TRUE 2-way divergence = 2.57×.

---

## 10. Methodology — USE THE HARNESS (kept from v1)

For any new B300 measurement, run `./utils/rigor_run.sh ./your_binary` for 3-method (wall-clock + ncu + SASS) verification. If methods don't reconcile within 10%, the test is broken.

**v2 additions to the rigor protocol**:
1. ALWAYS sample SM clock during long runs (stuck-at-1005 silent failure mode).
2. ALWAYS use `dram__bytes` ncu metric to confirm DRAM is hit (not L2-resident).
3. ALWAYS state denominator for "% of peak" (HBM3E spec = 7672 GB/s post-ECC, NOT 7.31 nor 8.0).
4. ALWAYS label L2 BW with one of {kernel-effective, wire/lts, L1-amplified}.
5. ALWAYS state carveout AND access pattern when quoting L1 capacity / BW.
6. ALWAYS publish (combine, WS, L2-resident, DRAM B/s) for atomic Gops/s claims.
7. ALWAYS annotate (zero/const | random | normal-ish) AND (per-call | sustained-via-cudaGraph) AND (boost | -lgc N MHz) for tensor TFLOPS.
8. NEVER use `pipe_tensor.cycles_active` for tcgen05 — use `smsp__inst_executed_pipe_tensor_subpipe_hmma_op_utchmma_utcqmma_utcomma_scope_2cta.sum`.
9. NEVER use `nvidia-smi -lgc 2032` (paradoxically pins to 1920); use `-rgc` for true boost.

---

## 11. Provenance

Every entry cites either a v1 row, a corrections file, or a specific commit hash on `f2fp-deep-dive`. v1 entries that survive unchanged are tagged "kept"; entries that change are tagged with the corrections-file source and "supersedes".

For unresolved tensions, see `MASTER_INDEX.md` §4 (top 20 unresolved questions) and the per-topic `*_INCONSISTENCY_LOG.md` files.

---

## 12. Changelog (v1 → v2)

| Section | Net change |
|---|---|
| §1 Memory ladder | NVLink rows recomputed against 900 GB/s spec; DSMEM rows replaced wholesale; L2 BW labelling enforced; new TMEM/L1 rows |
| §2 Compute peaks | Added FADD/FMUL row; added 3-source FFMA realistic cap; added BF16/FP8 random-vs-zero columns; added NVFP4 cudaGraph BPG=16 record; demoted "1543 BF16" / "256 cores" / "FP8 mma.sync 7500 TFLOPS" claims |
| §3 Coordination | mbarrier split into arrive/arrive+wait; threadfence_system flagged unresolved; persistent kernel CPU↔GPU dropped 4 µs → 2.03 µs |
| §5 Atomics | True peak 1005 Gops/s replaces "449 peak"; mandatory metadata rule added |
| §6 (NEW) Compute pipe ladder | Authoritative IADD3-on-FMA-pipe placement; dual-issue 54-74% replaces "free" / "114 TOPS"; MUFU 4.74 G/chip replaces 47.8 G mislabel |
| §7 Architecture | NVLink-5 (not v7); 8 GPCs (not 9/10); L2 atomic units demoted to MED; transient 1259 W flagged |
| §8 (NEW) V11-V51 | DSMEM exhaustive characterization; TMA pipelining; bank conflict regimes; tcgen05 dedup unified model; K-id 5-gate |
| §9 Counterintuitive | 8 new entries (#33–#40); #18 demoted |
| §10 Methodology | 9 new mandatory rules added |
