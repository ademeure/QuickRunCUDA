# DENSE B300 / Blackwell sm_103a — SM Pipe Catalog

> ⚠ **CRITICAL READING NOTE — section verification status:**
>
> Each section has ONE of these tags (post 2026-04-23 methodology shift):
>
> - **✅ AUDIT-VERIFIED** — independently re-run on this rig with SASS dump + ncu metrics + matching catalog claim. Justification record at `justifications/<id>.md`. **Trust these.**
> - **🟡 CATALOG-PRESERVED** — copied from `B300_PIPE_CATALOG.md` because it looks plausible and methodology was reasonable, but **NOT independently re-run by this audit**. Do NOT cite as authoritative until verified.
> - **❌ FALSIFIED** — disproved by audit. Section preserved as a corrective record with the wrong claim crossed out.
> - **🔍 PENDING** — agent is running or queued.
>
> **The 🟡 tag is load-bearing.** Earlier iterations of this DENSE doc were less careful about distinguishing these — many §22c-§22r entries were essentially catalog claims preserved verbatim. They're plausible but not yet in the audit-verified set.
>
> **What IS audit-verified** (16 sections, 2026-04-23):
> - §0 spec card key numbers (148 SMs, 7680-bit, 126 MB L2, 1942 MHz DVFS, etc.)
> - §1 pipe topology
> - §0 FFMA peak (00a_ffma_peak.md)
> - §10 memory hierarchy (00b_mem_hierarchy.md)
> - §11 latency table key entries (24_latency_table.md)
> - §12 fence costs (30G_fence.md)
> - §13 atomics scope/contention basics (30B_atomics.md + 30B_atomics_FOLLOWUP.md correction)
> - §13a TMA sizes (30_tma_sizes.md)
> - §14 mma.sync FP16/TF32/FP8/INT8 (22_tensor_mma_sync.md)
> - §15 + §15a DSMEM (13_dsmem.md + 13_dsmem_exhaustive.md)
> - §22 dual-issue FFMA2+ALU (22_dual_issue_ffma2_alu.md)
> - §22e .reuse cache (22e_reuse_cache.md)
> - §22h compute-memory overlap — **with quantitative corrections** (22h_compute_mem_overlap.md, 11 SASS files preserved)
> - §30 TMA-vs-LDG max-tuned head-to-head (30_tma_vs_ldg_max_tuned.md)
>
> All have `justifications/<id>.md` records with full evidence (CLAIM/TEST/BUILD/RUN/STDOUT/SASS/COUNT/NCU/CLOCK/VERDICT/DELTA per the audit-of-the-audit rubric).
>
> **⚠ CATALOG SELF-ADMISSION TO BE AWARE OF:** catalog L1948 admits "Absolute numbers from self-op chains (`fma %0,%0,%0,%0` etc.) are ~2× inflated from register read-port pressure (a single register fills all 3-4 operand slots). The **ratios** to FFMA are the reliable information."
>
> Implication: any catalog cy/op claim measured via single-thread self-op chain is potentially ~2× higher than the architectural latency. Specifically suspect:
> - The `bench_latency.cu`-style entries (LDS=33, L1=43, L2=300, FFMA=4 etc.) measured via self-op chain MAY be inflated
> - Self-op chains (e.g. `fma a,a,a,a`) saturate register-file ports because all operands collide on one register
> - Multi-chain measurements (4 independent registers) avoid this
>
> This audit's §0.FFMA used 8 INDEPENDENT chains (verified in SASS) so the 71.82 TFLOPS result is NOT subject to this. But §24 latency table entries (some via self-op chain) may be 2× too high. JUSTIFIED §24 noted "FFMA cy/op = 4.2-4.4 cy match catalog" — if that's the inflated number, the architectural FFMA latency might actually be ~2 cy (consistent with NVIDIA pipeline depth).
>
> **Going forward**: DENSE / JUSTIFIED should specify chain methodology (self-op vs distinct-chain) for any latency claim, and prefer distinct-chain (or report both).
>
> **What is NOT audit-verified yet** (still 🟡 CATALOG-PRESERVED):
> §17 MUFU throughput, §18 branch divergence, §19 INT8 dp4a numbers, §20 FMIN penalty, §21 tcgen05 throttling cliff, §22c CTA capacity formula, §22d cluster launch overhead, §22f L1/L2 stride probe, §22g tcgen05 SASS encoding (UTC* opcodes confirmed in SASS but exact-cycle claims not retested), §22i per-GPC L2 variation (DSMEM exhaustive partially corroborates), §22j smem bank conflict sweep, §22k PTX special registers (only `%nsmid`/`%clock64` informally checked), §22l grid sync 2.2 µs, §22m kernel launch 5.7 µs, §22n CTA scheduler placement (DSMEM exhaustive partially corroborates), §22o NVFP4 (agent IN FLIGHT — preliminary evidence supports catalog), §22p power efficiency, §22q register spilling, §22r atomic contention at scale.
>
> **Status:** Iteration in progress. This is a pruned, dense version of `B300_PIPE_CATALOG.md` that removes content known to be wrong, outdated, or low-value. Every numerical claim that survives has either a JUSTIFIED entry or a REVIEW_CHECKLIST flag.
>
> **Source:** `B300_PIPE_CATALOG.md` (19,742 lines).
> **Audit input:** `reviewed_errors_b300.md` (user's [!fail] / [!todo] callouts on related canonical doc).
> **Validation:** `JUSTIFIED_B300_PIPE_CATALOG.md` (per-section replication records).
>
> **Pruning rules applied:**
> 1. Remove anything contradicted by V52/V53/V54 empirical settlements
> 2. Remove anything user [!fail]'d in `reviewed_errors_b300.md` for the same topic
> 3. Remove "research log" repetition (sections 16–22 in catalog)
> 4. Remove formula-as-measurement claims (e.g. headline TFLOPS computed from cores × clock with no actual FLOPS counter)
> 5. Keep only LOW-LEVEL findings (drop high-level cuBLAS / GEMM ladder content unless it teaches something architectural)
> 6. Tag every surviving claim with confidence: 🟢 (3-method verified), 🟡 (1-2 method or regime caveat), 🔴 (unverified, kept because hint-bearing)
>
> **Conventions:**
> - All clock state explicit: `@1920` / `@2032` / `@1500` / `@1005` / `@boost`
> - All BW with denominator: spec=7680 GB/s, this-device=7672 GB/s
> - All TFLOPS state count convention: `2 FLOPS/FFMA` etc.
> - `⚠ FOOTGUN` for commonly mis-cited claims
>
> **Rough page budget:** ~2,500 lines (vs 19,742 source). Aggressive prune.

---

## Top of sheet — what to use this for

If you want a single number with a confidence tag: read the relevant `## §N` section, take the bold answer, ignore the rest.

If you want to verify a claim: jump to `JUSTIFIED_B300_PIPE_CATALOG.md` for the same `§N`.

If you don't trust a number: it's probably already on `REVIEW_CHECKLIST_B300.md`.

---

## §0. Comprehensive reference card (catalog L8807+, with corrections)

**This combines the catalog's "Comprehensive Reference Card" (L8807) with the corrections from this audit.** Numbers in **bold** are post-audit replicated/corrected; plain text is catalog-claim-preserved.

### Hardware

| Property | Value | Source |
|---|---|---|
| GPU | NVIDIA B300 SXM6 AC | nvidia-smi |
| Compute capability | sm_103a (PTX 10.3) | cudaGetDeviceProperties |
| SMs | **148** (IDs 1-147; SM 0 never scheduled) | cudaDeviceProp.multiProcessorCount |
| **GPCs** | **9 × 16 SMs + 1 partial × 4 SMs = 148** ⚠ | DSMEM exhaustive `%smid` (catalog "8 GPCs" elsewhere is WRONG) |
| Warps per SM | 64 (4 SMSPs × 16 warps each) | catalog |
| Max CTAs per SM | 32 | catalog L7682 (concurrent CTA cap test) |
| Max threads/CTA | 1024 | spec |
| FP32 cores per SM | **128** (4 SMSPs × 32 lanes; NOT 256) | per CLAUDE.md |
| Registers per SM | 65,536 (256 KB) | spec |
| Max regs per thread | 232 (via setmaxnreg; spill at ~192 live floats) | catalog |
| Shared memory per SM | 228 KB pool (227 KB usable per CTA; 1 KB reserved) | catalog |
| L1 cache | 256 KB (shared with smem pool) | spec |
| L2 cache | **126.5 MiB** (LRU, 2 partitions with address hash) | cudaDeviceProp.l2CacheSize |
| HBM | **268 GiB HBM3e** (visible, post-ECC) | nvidia-smi |
| HBM bus | **7,680 bits** (1/16 controllers fused on AC SKU; full SKU is 8192) | cudaDeviceProp.memoryBusWidth |
| HBM stacks | **8 × 12-Hi** (3 GB/die) | NVIDIA Tech Blog post-correction |
| Memory I/O clock | 3,996 MHz (= 7.992 Gbps/pin × 1024 b/stack × 8 stacks ÷ 8 ÷ 1.0625 ≈ **7,672 GB/s post-ECC**) | catalog |
| **SM clock — typical sustained** | **1,942 MHz** (DVFS settling, NOT 2032 boost spec NOR 1920 lock) | this audit |
| SM clock — boost spec | 2,032 MHz | nvidia-smi -q |
| SM clock — `-lgc 2032` paradox | pins to 1920 (NOT 2032) | catalog |
| SM clock — silent stuck floor | 1005 MHz | catalog observation |
| PCIe | Gen 6 x16 card; realized Gen 5 on this host (57 GB/s/dir) | catalog L8830 |
| NVLink | 18 lanes × 53.125 GB/s = 956 GB/s bidirectional per peer | catalog |
| ECC | always ON | spec |
| Async copy engines | 4 | cudaDevAttr |
| Media | 7 NVDEC + 1 NVENC + 1 OFA + 7 JPEG | catalog |
| MIG | supported (up to 7 × 1g.34gb slices) | catalog |
| TDP | 1,100 W | spec |
| Idle baseline | 144-198 W | catalog |

### Compute throughput (chip-wide, this rig DVFS at 1942 MHz unless noted)

| Path | Catalog peak TF | This audit | Status |
|---|--:|--:|---|
| Scalar FFMA (FP32) | 72.3 | **71.82 ✓** | ✅ JUSTIFIED §0.FFMA |
| FFMA2 / HFMA2 / BFMA2 (packed) | 72.3 | (per §22 inferred 256 FLOPS/SM/cy) | ✅ |
| FFMA2 + LOP3 1:1 (dual-issue total useful) | — | **314 ops/SM/cy** | ✅ NEW §22 |
| TF32 mma.sync | 1,200 | (cuBLAS 1,016 = 85%; mma.sync micro 285.7 ✓) | ✅ JUSTIFIED §22 |
| FP16/BF16 mma.sync (m16n8k16) | 569-578 | **571 ✓** (99.5% pipe_tensor) | ✅ JUSTIFIED §22 |
| FP16 tensor (cuBLAS 8K GEMM) | 2,325 | (catalog: 2,034 = 87%) | 🟡 cuBLAS not retested |
| FP8 mma.sync emulated | 276 | **309** (+12% over catalog) | ⚠ JUSTIFIED §22 — recommend bumping catalog |
| FP8 tcgen05.mma micro | 4,651 | (catalog L6720; 93% of 5 PF spec) | 🟡 not yet rerun |
| FP8 sparse tcgen05.mma | ~9,300 | 7,440 (74% of spec) | 🟡 catalog claim |
| NVFP4 K=64 standard | ~10,000 | (cuBLAS 10,800; CUTLASS 8,700 — capped) | 🟡 |
| **NVFP4 K=96 ULTRA** | **~15,000 spec** | (catalog: 1.5× over K=64; replication TODO) | 🔍 K=96 ULTRA agent failed (token limit) |
| FP64 (DFMA) | ~1.2 | 0.95 catalog (under-saturated; A5 in checklist) | 🟡 |
| INT32 (IMAD/IMUL) | — | 18.2 TOPS catalog | 🟡 |
| INT8 dp4a SIMD | — | 54.5 TOPS catalog | 🟡 |
| MUFU (sin/cos/rsqrt) | — | 4.8 TOPS catalog (ex2 = 8.1) | 🟡 |

### Memory throughput

| Level | Read TB/s | Write TB/s | Latency cy |
|---|--:|--:|--:|
| **HBM (coalesced 256-bit, max-tuned)** | **7.41 ✓** (96.5% of 7672) | (catalog 7.5 memset) | ~789 (catalog L24) |
| L2 cache @ 64 MB WS, max-tuned | **18-20 ✓** (catalog wire 13.3 was under-counted by 37-54%) | — | ~300 (catalog L24) |
| **L2 cache @ 64 MB WS, TMA max-tuned** | **20.49 ✓ (12% better than LDG)** | — | — |
| Shared memory (`ld.shared.v4`) | **35.88 ✓** (97.5% of 36.79) | catalog 34 | **29 ✓** (catalog L24's 33 was 14% high) |
| PCIe H2D | 0.057 | — | 6.5 µs |
| PCIe D2H | 0.057 | — | 9.0 µs |
| PCIe full-duplex | 0.099 combined | | |
| NVLink peer | 0.820 | 0.718 | ~2,700 |
| Pinned (zero-copy) | 0.054 | 0.053 | ~1 µs/hop |
| **DSMEM read (single-chain)** | **204-223 cy ❌** (catalog "23 cy free" FALSIFIED — 9× slower) | **87-117 GB/s/cluster** sustained | — |
| **DSMEM read (ILP=32)** | **9 cy/load ✓** (close to LDS) | (different config) | — |

### Synchronization costs (cycles, this rig 1942 MHz unless noted)

| Primitive | Catalog | This audit |
|---|--:|--:|
| `__syncwarp` | 36 | (no SASS emitted in some cases per memory) |
| `bar.sync 0,32` (1-warp partial) | 26 | not retested |
| `__syncthreads` (1024 thr) | 86 | not retested at 1024 |
| `__syncthreads` BS=512 | catalog 45 | **54 ✓ (formula `22+2W` not `12+2W`)** |
| `__syncthreads_count/_and/_or` | 150 | not retested |
| `mbarrier` arrive+wait cycle | 318 | — |
| `mbarrier` RTT (count=1) | catalog 54 | **123 ✓** (catalog header was arrive-only) |
| Cluster barrier | 380 (flat 2-16 CTAs) | not retested |
| Cooperative `grid.sync` (148 blocks) | 2,371 (1.24 µs) | not retested |
| `fence.*.cta` | catalog 27 (L8878) / 8.6 (L114) | **8 ✓** |
| `fence.*.gpu` | catalog 292 / 274 | **267-281 ✓** (+~280 first-fence-after-write FIXED) |
| `fence.*.sys` (single-GPU) | catalog 3,500 | **1,727 ✓** (V54's 2806 was 2-GPU NVLink rig) |
| `fence.proxy.async` | 36 | not retested |

### Atomic costs (cycles per atom, uncontended unless noted)

| Op | Catalog | This audit |
|---|--:|--:|
| `atom.global.add.u32` chain | 24 | **45 ✓** (matches LDS chain at 45 cy — the "33 cy LDS" elsewhere was throughput-derived) |
| `atom.global.add.{f32,f16x2,bf16x2,f64}` | 24 (all native per catalog) | f16/bf16 atomicAdd are NOT native — emit ATOM.E.CAS loops 6.3× slower than u32 (NOT 45×) |
| `atom.global.add.u64` | 156 | not retested |
| `atom.global.cas.b64` | 731 (30× slower) | not retested |
| `atom.shared.add.u32` | 24 | not retested |
| `atom.shared.add.f32` | 97 (emulated via bsync+CAS) | not retested |
| **atom.relaxed vs acq_rel scope penalty** | catalog 31.3× | **2.0-2.2× ✓** (catalog compared chip-throughput vs single-thread chain — apples-to-oranges) |
| **chip-wide same-addr 1 hotspot** | — | **49.1 Gops/s ✓** |
| **N=2 hotspot anomaly** | catalog 32× slower | **29× slower ✓** confirmed |
| **per-warp pattern (clean addr_idx=warpId)** | catalog "5× slowest" | **1.09× FASTER than 1-hotspot** ❌ catalog wrong |
| **per-CTA pattern** | catalog "same as single" | **12.4× FASTER than 1-hotspot** ❌ catalog wrong |
| **coalesced unique-per-lane** | catalog 0.94 atom/cy/lane | **0.023 atom/cy/lane** ❌ catalog 41× off |

### Warp primitives

| Op | cy |
|---|--:|
| `__shfl_xor_sync` (raw) | 6 |
| SHFL (saturated at 32 warps) | 1 w-inst/cy/SM |
| `__ballot_sync` | 29 |
| `__reduce_min/max_sync` | 31 (SASS = CREDUX.MIN/MAX, 18 cy in §24) |
| `__reduce_add_sync` | 54 (SASS = REDUX.SUM, 44 cy chain in §24 — **NEW: 2.4× slower than min/max**) |
| `__match_any_sync` | 56 (very slow, 375 cy in another section — verify) |
| Warp-wide scan (5-step Kogge-Stone) | 186 |

### Host API costs

| Operation | µs |
|---|--:|
| `cudaGetLastError` | 0.011 (always check!) |
| `cudaEventElapsedTime` | 0.037 |
| `cudaStreamWaitEvent` (host enqueue) | 0.13 |
| NVTX push+pop (no profiler) | 0 |
| `cudaMallocAsync` + `FreeAsync` cycle | 0.4-1.2 |
| `cudaMalloc` | 18 |
| `cudaFree` | 20 |
| Kernel launch (`<<<>>>`) | 2.0 |
| `cudaLaunchKernelEx` + PSS | 1.47 |
| cudaGraph launch (1000 kernels) | 0.56 / kernel |
| `cudaGraphExecUpdate` | 0.15 |
| `cudaStreamSynchronize` (after tiny kernel) | 6.3 |
| `cudaDeviceSynchronize` (idle) | 1.3 |
| CUDA cold start (cuInit → first kernel) | 326 ms |
| NVRTC compile | 6 ms (warm) |
| `cuLibraryLoadData` | 14 (6.5× faster than cuModule) |

### Key design rules (catalog L8927, mostly confirmed)

1. Fuse elementwise ops: N ops fused → near-linear N× speedup
2. Use wide loads: uint4 (16 B) = 85% HBM; 2×uint4 (32 B) = 94% (per JUSTIFIED §0.MEM, max-tuned hits 96%+)
3. ≥16 warps/SM for 90% of HBM peak
4. Smem-privatize histograms: 200× faster than naive global atomics
5. Persistent 32 CTAs/SM for memory-bound (not 1/SM — 2.6× better)
6. Prefer `__reduce_*_sync` over manual shuffle trees (40% faster)
7. Use `max(x,0)` not `x>0?x:0`: fused min/max is 2× faster than setp+selp
8. Avoid function pointers in inner loops (5× overhead)
9. Avoid warp specialization without async overlap (3.8× anti-pattern)
10. Always use NonBlocking streams (12% faster than default)

### Roofline (operational intensity ridge)

| Compute path | Ridge OI (FLOP/byte) |
|---|--:|
| Scalar FFMA | 18 |
| FP16 tensor (tcgen05) | 314 |
| FP8 tensor | 628 |

Most ML inference ops are below OI = 1 → memory-bound → fusion is king.

### B300 vs H100 vs A100 (catalog L8957+)

| Spec | A100 SXM (2020) | H100 SXM (2022) | **B300 SXM6 (2025)** | B300/A100 | B300/H100 |
|---|--:|--:|--:|--:|--:|
| SMs | 108 | 132 | **148** | 1.37× | 1.12× |
| FP32 TFLOPS (scalar) | 19.5 | 67 | **72** | 3.7× | 1.07× |
| FP16 tensor TFLOPS | 312 | 990 | **2,325 spec / 2,034 measured** | 6.5× | 2.1× |
| FP8 tensor TFLOPS | — | 1,979 | **4,651** | — | 2.3× |
| HBM capacity | 80 GB | 80 GB | **268 GB** | 3.4× | 3.4× |
| HBM bandwidth | 2.0 TB/s | 3.35 TB/s | **7.4 TB/s** | 3.7× | 2.2× |
| L2 cache | 40 MB | 50 MB | **126.5 MB** | 3.2× | 2.5× |
| NVLink BW (bidi) | 600 GB/s | 900 GB/s | **956 GB/s** | 1.6× | 1.06× |
| SM clock (boost) | 1,410 | 1,830 | **2,032** | 1.44× | 1.11× |
| TDP | 400 W | 700 W | **~490 W measured tensor / 1,100 W max** | 1.23× | 0.70× |
| Compute capability | sm_80 | sm_90 | **sm_103a** | — | — |

### Per-watt (TFLOPS / W)

| Metric | A100 | H100 | B300 | B300/A100 | B300/H100 |
|---|--:|--:|--:|--:|--:|
| FP16 tensor / W | 0.78 | 1.41 | **4.17** | **5.3×** | **3.0×** |
| HBM BW / W (GB/s/W) | 5.0 | 4.8 | **15.1** | 3.0× | 3.2× |

---

## §0a. Spec card (legacy, kept for backward compat — see §0 for canonical)

### Hardware (`cudaGetDeviceProperties`)

- 148 SMs × 4 SMSPs × 32 lanes = **128 FP32 cores/SM** (NOT 256). Total: 18,944 FP32 cores. 🟢
- `cudaDeviceProp.memoryBusWidth = **7680 bits**` (1 of 16 controllers fused on AC SKU; full SKU is 8192). 🟢
- `cudaDeviceProp.l2CacheSize = 126 MB`. 🟢
- `cudaDeviceProp.memoryClockRate = 3996 MHz` I/O = 7.992 Gbps/pin × 1024 bits/stack × 8 stacks = ~7672 GB/s post-ECC. 🟢
- 8 HBM3E 12-Hi stacks (3 GB/die). 🟢 (was 12 stacks in older docs — wrong.)

### Clock states (this rig, observed 2026-04-23)

| State | MHz | When |
|---|--:|---|
| `nvidia-smi -q` reported boost | 2032 | nominal datasheet boost |
| Sustained-FFMA DVFS settling point | **1942** | new finding from 00a_ffma_peak.md — pure FFMA does NOT reach 2032 nor stop at 1920 |
| `-lgc 2032` paradox pin | 1920 | per CLAUDE.md note |
| Stuck-low silent floor | 1005 | observed historically; ⚠ FOOTGUN: leftover process or driver state can leave chip here |
| `-rgc` reset behavior | back to dynamic | restores DVFS scaling |

⚠ **FOOTGUN:** No headline TFLOPS / TB/s number is meaningful without stating which clock state. Catalog historically used 1.92 GHz formulae; CLAUDE.md uses 2.032 GHz formulae. They differ by 6%.

### Compute peaks

| Op | Measured | Theoretical | Clock | Status |
|---|--:|--:|--:|---|
| FP32 FFMA scalar | **71.82 TF** | 73.6 TF (148 × 256 × 1.942) | 1942 MHz (DVFS) | ✅ replicated 100% match (00a_ffma_peak.md). ncu pipe_fma=99.5%. SASS=1024 FFMA/inner loop. |
| FP32 FFMA scalar (alt clocks) | — | 72.7 TF @ 1920 / 76.96 TF @ 2032 | catalog vs CLAUDE.md | both correct for their clock; cite the rig DVFS clock to be precise |
| FP64 DFMA | (pending) | 1.20 TF @ 2032 | — | catalog claims 0.95 TF — under-saturated? See REVIEW_CHECKLIST A5 |

### Memory peaks (catalog claims, replication pending)

| Tier | Read | Notes |
|---|--:|---|
| SMEM (`ld.volatile.shared.v4.u32`) | 35.6 TB/s claim @ 1.92 GHz | uses 1.92; if real clock is 1942 → SoL recompute |
| L1 hit (.ca, WS≤1MB) | 36.1 TB/s claim | inconsistent with 28.7 TB/s "L1" elsewhere (B2 in REVIEW_CHECKLIST) |
| L2 plateau (4-128 MB) | 22-26 TB/s claim | "was wrongly 10.2 — under-occupied launch" (regime-narrow) |
| HBM3E read | **7.18 TB/s** ncu-verified claim | denominator: spec 7680 / this-device 7672 → 93.5–95.2% (depending) |
| HBM3E write | 7.09 TB/s standard / 7.57 TB/s contested | 7.57 has disputed provenance |

### Sync/atomic key numbers (claims; replication pending)

| Op | cy | Notes |
|---|--:|---|
| `__syncwarp()` | 2.8 | claim — needs verification, likely 0 since no SASS emitted in some cases |
| `__syncthreads` BS=512 | 45 | catalog claims, inconsistent with formula `12+2W` from same doc (would be 44). |
| `mbarrier.arrive` | 8.1 | claim |
| `__threadfence_block` | 8 | wave-7 V54 confirmed at 8 cy ✓ |
| `__threadfence` (gpu) | **267 cy** sustained + 280 cy first-fence-after-write | wave-7 V54 settled. CATALOG L115 still says 274 — close but should be updated. ⚠ DENSE recommends V54 numbers. |
| `__threadfence_system` | **2806 cy = 1381 ns @ 2032** | wave-7 V54 settled. Catalog had 1750/2870/3042 spread (1.74×). ⚠ Use V54. |

### tensor cores (catalog)

| path | Best measured | Spec | clock | Notes |
|---|--:|--:|---|---|
| mma.sync m16n8k16 BF16/FP16 | 569-578 TF | 600 spec? | — | wave-6 [🟢 HIGH] per canonical |
| tcgen05.mma BF16/FP16 | 1980-2240 TF | 2.5 PF? | — | wave-6 [🟢 HIGH] |
| tcgen05.mma FP8 cuBLAS | 3984-4425 TF | 5 PF | — | wave-6 [🟢 HIGH], realistic mix |
| NVFP4 cuBLAS realistic | 11423 TF (76.2% of 15 PF) | 15 PF | — | K=96 ULTRA inaccessible from public libs |

⚠ **FOOTGUN:** ncu `pipe_tensor` does NOT measure tcgen05.mma — only legacy mma.sync/HMMA. tcgen05 must be measured via `wall-clock × cy/MMA × ops/MMA` and SASS UTCQMMA/UTCHMMA counts.

### Contention rules (catalog §3, partially verified)

1. Same pipe → cap at pipe ceiling (e.g. F2FP + LOP3 share alu → 64 combined).
2. Different pipes → mostly add cleanly. Exception: FFMA2 + UNPACK shows ~16% SMSP friction (u=1.67 vs ideal 2.00).
3. Dispatch cap = 4.00 warp-inst/SM/cy "is hard" per catalog L480 — but V52 has shown alu+fma can sum to 147% — **the framing of "hard cap" is misleading; what's hard is the per-SMSP 1 inst/cy limit**.

---

## §1. Pipe topology — verified 2026-04-23 (justifications/01_pipe_topology.md)

| Pipe | Cap (per SM/cy) | Verification | Examples |
|---|--:|---|---|
| pipe_fma (scalar dual) | **4.00 confirmed** | ✅ FFMA hits 3.88 (97%); dual FFMA+LOP3 → 3.95 (99%) | FFMA, FMUL, FADD |
| pipe_fma (packed) | 2.00 | (catalog claim, not yet rerun) | FFMA2, HFMA2, BF16-FMA |
| pipe_alu | **2.00 confirmed** | ✅ Pure LOP3 (xor.b32) saturates at 96.97% pipe_alu | LOP3, PRMT, F2FP, SHF, FMNMX, ISETP, FSETP, I2FP |
| pipe_fmaheavy | **2.00** | (component view; see footgun below) | IMAD, IMAD.X, IMAD.WIDE, IDP.4A/2A, HADD2.F32 |
| pipe_fmalite | **2.00** | (component view) | "lite" half of FFMA path |
| pipe_xu compound | **~0.50 confirmed** | ✅ MUFU.SIN saturates at 49.79% of pipe_xu (≈ half rate) | MUFU.SIN/COS (need range-reduction) |
| pipe_xu simple | **~1.00 confirmed** | ✅ MUFU.EX2 saturates at 98.46% of pipe_xu (near full rate) | MUFU.EX2/RSQ/SQRT/RCP/LG2/TANH, POPC, BREV, FLO/CLZ |
| pipe_lsu | 1.00 | (catalog claim, not yet rerun) | LDG, STG, LDS, STS, LDSM, SHFL.SYNC |
| pipe_adu | ~0.5 | (catalog claim) | BAR.SYNC, MATCH.ANY |
| pipe_uniform | ~1.0 | (catalog claim) | S2UR, LDSM.sync, ACTIVEMASK, UFFMA family |
| pipe_fp64 | 0.05 | (catalog claim, but see latency inconsistency: 92 cy L103 vs 63.9 cy L460) | DFMA, DADD, DMUL — heavily throttled |
| pipe_tensor | — | (separate from tcgen05; needs kind-specific tests) | HMMA, IMMA |
| pipe_cbu | — | (low-priority; invisible in steady-state) | BRA, EXIT |

### Dispatch ceiling (confirmed)

The "4.00 warp-inst/SM/cy hard dispatch cap" claim from catalog L210 is **CONFIRMED** in the strict per-SM total sense — no test exceeded 4.00.

**But** this cap IS NOT what the popular "256 FLOPS/SM/cy" derivation depends on; the FFMA scalar peak relies on dispatching to BOTH fma sub-pipes alternately, NOT both simultaneously per instruction.

### V52 dual-issue settlement (confirmed AGAIN)

Dual FFMA + LOP3 ncu measurement:
- `pipe_alu = 96.16%`
- `pipe_fma = 48.83%`
- **sum = 144.99%** ← exceeds any single pipe's cap because pipe_alu and pipe_fma are PHYSICALLY INDEPENDENT and overlap freely.

This re-confirms wave-6 V52. The "147%" figure quoted elsewhere is well within run-to-run noise of the 145% just measured.

### ⚠ FOOTGUN — "FFMA → both fma sub-pipes simultaneously" is **FALSIFIED**

Catalog L218 says:
> "These are the ones that **uniquely use BOTH fma sub-pipes simultaneously** at 2.00 each → **4.00 warp-inst/SM/cy**."

ncu evidence for **dual FFMA + LOP3**:
- `pipe_fmalite = 93%`
- `pipe_fmaheavy = 4.5%`

**A single FFMA dispatches to ONE sub-pipe per cycle** (scheduler-chosen), not both. Solo FFMA shows both sub-pipes near 92% only because the scheduler alternates, not because each instruction goes to both.

This explains the "256 FLOPS/SM/cy" formula correctly: 4 SMSPs × (1 FFMA dispatch/cycle × 32 lanes × 2 FLOPS/FFMA) = 256 FLOPS/SM/cy. There's no per-instruction "double-issue".

The catalog's L218 wording should be: "FFMA can use EITHER sub-pipe per cycle, freely alternating across cycles, achieving aggregate 4.00 warp-inst/SM/cy when there's enough ILP."

### Key citation paths

- justifications/00a_ffma_peak.md (scalar FFMA peak, 71.82 TF replicated)
- justifications/01_pipe_topology.md (pipe placement / dispatch ceiling, V52 re-confirmed)

---

## §2. Instruction catalog — rate cheatsheet (warp-inst per SM per cycle)

> **Status:** Catalog §2 (13 sub-tables) carried over with tags. **Bold** = re-verified 2026-04-23 (this DENSE pass). Plain = catalog claim only, replication pending. ⚠ = known issue.
>
> Rate units: `r` = warp-instructions issued per SM per cycle. The aggregate dispatch ceiling is 4.00. "elements" / "ops" call out logical work-per-instruction for packed ops.

### §2.1 FP32 scalar (pipe_fma)

| PTX | SASS | r warp-inst/SM/cy | Logical |
|---|---|--:|---|
| `fma.rn.f32` | FFMA | **4.00** ✅ | 128 FFMA = **256 FP32 FLOPS** |
| `mul.rn.f32` | FMUL | 4.00 | 128 FMUL |
| `add.rn.f32` | FADD | 4.00 | 128 FADD |
| `abs/neg.f32` | FADD.FTZ (compiler) | 4.00 | 128 |

⚠ FOOTGUN: scalar FFMA at 4.00 already saturates dispatch — cannot be co-issued with anything else without losing throughput. See §1 footgun on "both sub-pipes simultaneously" wording.

### §2.2 Packed FP32/FP16/BF16 (pipe_fma, both sub-units occupied for 1 inst)

| PTX | SASS | r | Logical (FLOPS/SM/cy) |
|---|---|--:|---|
| `fma.rn.f32x2` | FFMA2 | 2.00 | 128 FMAs = 256 FLOPS-FP32 (same as scalar) |
| `fma.rn.f16x2` | HFMA2 | 2.00 | 128 FMAs-FP16 = 256 FLOPS-FP16 |
| `fma.rn.bf16x2` | HFMA2.BF16 | 2.00 | 128 FMAs-BF16 |
| `add.rn.f16x2` | HADD2 (sometimes folds to HFMA2) | 2.00 | 128 adds |
| `mul.rn.f16x2` | HMUL2 (often folds to HFMA2) | 2.00 | 128 muls |

### ⚠ KEY FINDING — FFMA2 + ALU IS the dual-issue sweet spot (justifications/22_dual_issue_ffma2_alu.md)

**FFMA2 + LOP3 is strictly better than scalar FFMA + LOP3** because of how dispatch slots are consumed:

| Path | Dispatch use | Pipes saturated | FP32 FLOPS/SM/cy | LOP3 ops/SM/cy | TOTAL useful ops/SM/cy |
|---|--:|---|--:|--:|--:|
| Scalar FFMA solo | 4.00 (full) | fma 97% (alternates H/L) | 256 | 0 | 256 |
| Scalar FFMA + LOP3 | 3.95 | fma 49% (HALVED), alu 96% | **128** (lost half) | ~30 | 187 |
| FFMA2 solo | 2.04 | fmaH+fmaL both 97% (single inst) | 256 | 0 | 256 (idle alu slots) |
| **FFMA2 + LOP3 1:1** | **3.94** | **fmaH 98%, fmaL 97%, alu 97%** ALL THREE | **252** (~full) | **31** (~full) | **314 ← winner** |
| FFMA2 + LOP3 2:1 (sweet spot) | ~3.0 | fma full, alu partial | 256 (full) | ~16 | 272 (LOP3 "free side dish") |
| LOP3 solo | 2.05 | alu 99% | 0 | 64 | 64 |

**Mechanism:** FFMA2 occupies BOTH `pipe_fmaheavy` AND `pipe_fmalite` per single instruction, but uses only 1 dispatch slot. Scalar FFMA uses 1 dispatch slot with the scheduler load-balancing across H/L sub-pipes. So:

- FFMA2 needs only 2.0 dispatch slots to saturate 256 FLOPS → leaves 2.0 slots free for ALU.
- Scalar FFMA needs all 4.0 dispatch slots to saturate 256 FLOPS → no room.

**Hard ceilings unchanged:** total dispatch ≤ 4.00 warp-inst/SM/cy still holds (3.94 max measured); FP32 FLOPS still capped at 256/SM/cy. The win is "ALU work for free as a side dish", not "double FLOPS".

**Practical recipe for max useful work:** if your kernel needs both FP32 FMA and integer/bitwise work, prefer **FFMA2 + LOP3 at 2:1 ratio** for full FFMA throughput with LOP3 "free", or **FFMA2 + LOP3 at 1:1** to maximize total ops/cycle (23% improvement over scalar FFMA alone, but the FFMA2 portion drops slightly to ~98%).

**Open follow-ups** (per agent): FFMA2+IMAD, FFMA2+LSU, HFMA2+LOP3, triple co-issue (FFMA2+ALU+LSU).

---

### §2.3 Integer (pipe_fmaheavy mostly; IADD3 splits)

| PTX | SASS | r | Notes |
|---|---|--:|---|
| `mad.lo.u32` | IMAD | 2.00 fmaH | 64 IMAD |
| `mul.lo.u32` | IMAD | 2.00 fmaH | 64 IMUL |
| `mul.hi.u32` | IMAD.HI.U32 | 1.00 fmaH | 32/SM/cy — half rate |
| `dp4a.s32.s32` | IDP.4A.S8.S8 | 2.00 fmaH | 64 SASS × 4 pairs × 2 ops = 512 int8-dot ops |
| `dp2a.*` | IDP.2A | 2.00 fmaH | 64 |
| `cvt.f32.f16` | HADD2.F32 | 2.00 fmaH | 64 conversions |
| `add.u32` (single) | IADD3 / IMAD.IADD | 4.00 total (2 alu + 2 fmaH) | 128 adds (compiler splits) |
| `add.u32 a,b,c,d` (3-input) | IADD3 fused | 2.00 alu | 1 IADD3 = 2 logical adds |

### §2.4 Integer u64

| PTX | SASS | Pipe | u64 ops/SM/cy |
|---|---|---|--:|
| `add.u64` / `sub.u64` | IADD3 + IMAD.X (2 SASS) | alu + fmaH | 64 |
| `mul.lo.u64` | IMAD + IMAD.WIDE + IADD3 ×3 | mostly fmaH | ~12 |
| `mul.hi.u64` | 6+ SASS | fmaH + alu | ~5 |
| `and/or/xor.b64` | 2× LOP3 | alu | 32 |
| `shl/shr.b64/.u64` | 3 SASS | alu | ~16 |
| `min/max.u64` | ISETP×2 + SEL×2 | alu | ~16 |

### §2.5 Narrow-format CVT (F2FP family, pipe_alu)

UNPACK (narrow → f16x2/bf16x2): all variants identical.

| Narrow type | SASS | r | elements/SM/cy |
|---|---|--:|--:|
| e4m3 (FP8) | F2FP.F16.E4M3.UNPACK_B | 2.00 | 128 |
| e5m2 (FP8) | F2FP.F16.E5M2.UNPACK_B | 2.00 | 128 |
| e2m1 (FP4) | F2FP.F16.E2M1.UNPACK_B | 2.00 | 128 |
| e2m3 (FP6) | F2FP.F16.E2M3.UNPACK_B | 2.00 | 128 |
| e3m2 (FP6) | F2FP.F16.E3M2.UNPACK_B | 2.00 | 128 |
| ue8m0 → bf16 | F2FP.BF16.E8.UNPACK_B | 2.00 | 128 |

⚠ With LOP3 zero-ext feedback (1-per-iter), effective rate halves to 1.00 = 64 elements/SM/cy. The peak only holds without LOP3 pollution.

PACK (wide → narrow): rates drop because of LOP3/PRMT tax. See catalog §2.5 for full table.

### §2.6 Other CVTs

| PTX | SASS | Pipe | r |
|---|---|---|--:|
| `cvt.rn.f16.f32` (pack) | F2FP.F16.F32.PACK + PRMT | alu | 1.00 (PRMT tax) |
| `cvt.rn.bf16.f32` | F2FP.BF16.F32.PACK + PRMT | alu | 1.00 |
| `cvt.f32.f16` | HADD2.F32 | fmaH | 2.00 |
| `cvt.rn.f32.s32/u32` | I2FP.F32.* | alu | 2.00 |
| `cvt.rn.f32.s64` | I2F.S64 | xu | **0.04 — super slow** |
| `cvt.rni.s32.f32` | F2I.NTZ | xu | 0.5 |
| `cvt.rni.sat.u8.f32` | F2IP.U8.F32.NTZ | alu | 2.00 (!) |
| `cvt.rni.sat.s8.f32` | F2I.S8.NTZ | xu | 0.5 |
| `cvt.sat.u8.s32` | I2I.U8.S32.SAT | alu | 2.00 |

### §2.7 Bitwise / shift / permute (pipe_alu)

| PTX | SASS | r |
|---|---|--:|
| `xor/and/or/not/lop3.b32` | LOP3.LUT | **2.00 ✅** |
| `shl/shr (plain)` | SHF.* | 2.00 |
| `shf.l/r.wrap.b32` | SHF.L/R.W.U32 | 2.00 |
| `prmt.b32` | PRMT | 2.00 |
| `bfi.b32` | LOP3.LUT (collapses) | 2.00 |
| `bfe.u32` | SHF.R.U32.HI + SGXT.U32 (2 SASS) | 1.00 |
| `brev.b32` | BREV | xu, 0.5 |
| `popc.b32` | POPC | xu, 0.5 |
| `clz.b32` / `bfind` | FLO.U32 (+ IADD3) | xu+alu, 0.5 |

### §2.8 Compare / select / predicate (pipe_alu)

| PTX | SASS | r |
|---|---|--:|
| `setp.*.u32/s32` | ISETP.* | 2.00 |
| `setp.*.f32` | FSETP.* | 2.00 |
| `selp.b32` | SEL | 2.00 |
| setp+selp (2 SASS) | ISETP+SEL | 1.00 |
| `vote.sync.ballot.b32` | ISETP+VOTE.ANY | 1.00 |

### §2.9 MIN / MAX — surprisingly all on pipe_alu (NOT fma)

| PTX | SASS | r |
|---|---|--:|
| `min/max.f32` | FMNMX | 2.00 |
| `min/max.NaN.f32` | FMNMX.NAN | 2.00 |
| `min/max.f16x2` | HMNMX2 | 2.00 (= 128 FP16 mins) |
| `min/max.bf16x2` | HMNMX2.BF16 | 2.00 |
| `min/max.s32` | VIMNMX3 (compiler folds 2 mins) | 2.00 (= 128 int mins) |
| `min/max.u64` | ISETP×2 + SEL×2 | 0.5 (~16 u64 min) |
| `abs.s32` / `neg.s32` / `abs.f32` | folds to IADD3/FADD/LOP3 | 2.00+ |
| `copysign.f32` | LOP3.LUT | 2.00 |

⚠ The "FMNMX3" 3-input fused min in catalog L981 may be compiler fusion not a native opcode. See REVIEW_CHECKLIST CRIT6.

### §2.10 Transcendentals (pipe_xu) — verified 2026-04-23

| PTX | SASS | r | Notes |
|---|---|--:|---|
| `ex2.approx.f32` | MUFU.EX2 | **0.50–0.63 ✅ simple** | 16-20 SASS/SM/cy. Saturates pipe_xu at 98.5% solo. |
| `rsqrt.approx.f32` | MUFU.RSQ | 0.5 | (catalog claim, simple) |
| `sqrt.approx.f32` | MUFU.SQRT | 0.5 | simple |
| `rcp.approx.f32` | MUFU.RCP | 0.5 | simple |
| `sin.approx.f32` | MUFU.SIN + FMUL range-reduction | **0.5 (compound) ✅** | Saturates pipe_xu at 49.8% (half the simple rate); also drives pipe_fma to 12.5% (range-reduction FFMA). |
| `cos.approx.f32` | MUFU.COS | 0.5 (compound) | likely same as sin |
| `lg2.approx.f32` | MUFU.LG2 | 0.5 | simple per catalog |
| `tanh.approx.f32` | MUFU.TANH | 0.5 | simple per catalog |

⚠ Catalog §16 has older MUFU latencies (RSQ=40 cy, RCP=42 cy) that include range-reduction overhead from author-added scaffolding. Prefer §23 clean sweep numbers (RSQ=18 cy ftz). See REVIEW_CHECKLIST CRIT7.

### §2.11 Warp / sync / barrier ops (key entries)

| PTX | SASS | Pipe | r | Notes |
|---|---|---|--:|---|
| `shfl.sync.{bfly,idx,up,down}` | SHFL.* | lsu | 1.00 | 32 SASS/SM/cy |
| `vote.ballot` | VOTE.ANY + ISETP | alu | 2.00 combined | |
| `vote.{any,all,uni}` | VOTE.* | alu | 2.00 | |
| `activemask` | uniform-pipe op | uniform | ~1.2 | (DCE'd in some tests) |
| `match.any.sync.b32` | MATCH.ANY | adu | **0.5 peak — VERY SLOW** | catalog L85: 375 cy = 20× other warp ops. Avoid. |
| `bar.sync 0` | BAR.SYNC.DEFER | adu | ~0.36 | thread-waiting dominates |
| `bar.arrive` | BAR.ARV | adu | ~0.47 | |
| `bar.red.popc.u32` | BAR.RED.POPC.DEFER | adu+alu | ~0.37 | |
| `redux.sync.min.u32` | CREDUX.MIN+IMAD (2 SASS) | alu+fmaH | **1.92 PTX-op/SM/cy** | each pipe at 1.92/2.00 |
| `redux.sync.add.u32` | REDUX.SUM+IMAD | adu | **0.50 — 4× slower than min/max** | |
| `redux.sync.{or,and,xor}` | REDUX.* | adu | 0.50 (same as add) | |
| `membar.cta` | MEMBAR.SC.CTA | lsu | 0.83 | scoped fence on lsu |
| `membar.gl` | MEMBAR.SC.GPU + ERRBAR | adu+lsu | extremely slow | see §30.G replication for cy/op |
| `ldmatrix.sync.x1.b16` | LDSM (1 quad) | uniform+lsu | ~1.0 | |
| `ldmatrix.sync.x4.b16` | LDSM (4 quads) | uniform+lsu | 0.25 | quarter rate |
| `atom.shared.add.u32` | ATOMS.POPC.INC.32 | lsu | 0.84 | |
| `atom.global.*` | ATOMG.* | lsu | bandwidth-bound | |
| `s2r %clock/%clock_hi` | S2R SR_CLOCKLO/HI | adu | 0.5 | |

### §2.12 Memory

| PTX | SASS | Pipe | Notes |
|---|---|---|---|
| `ld.global.u32` | LDG.E | lsu | DRAM-bound in practice, ~1 inst/SM/cy issue |
| `st.global.u32` | STG.E | lsu | DRAM-bottleneck, not pipe |
| `ld.shared.u32` | LDS | lsu | ~1.0 issue, bank-conflict-sensitive |
| `st.shared.u32` | STS | lsu | 1.00 saturating |

### §2.13 FP64 — severely throttled

| PTX | SASS | Pipe | r |
|---|---|---|--:|
| `fma.rn.f64` | DFMA | fp64 | **0.05 = 1.6 DFMA/SM/cy = ~475 GFLOPS-FMA chip-wide** |
| `add.rn.f64` | DADD | fp64 | 0.05 |
| `mul.rn.f64` | DMUL | fp64 | 0.05 |

DFMA is **NOT pipelined** per catalog L460 — 4 chains give zero ILP benefit (63.9 cy/op each). FFMA + ALU co-issue freely during the 64 cy window. ⚠ Catalog has DFMA latency = 92 cy (L103) AND 63.9 cy (L460) — inconsistent; likely 63.9 is the corrected number from a later test.

⚠ B300 FP64 peak = ~1.2 TFLOPS (CLAUDE.md). Catalog says 0.95 TFLOPS measured — under-saturated; needs replication. (REVIEW_CHECKLIST A5)

---

## §3. Contention rules (catalog §3, L472)

1. **Same pipe** → cap at pipe ceiling.
   - F2FP + LOP3 (both alu) → 64 combined. ✓
   - IMAD + FFMA scalar (compete for fmaH) → reduces FFMA peak.
2. **Different pipes** → mostly add cleanly, with caveats:
   - **FFMA2 + UNPACK** (fma + alu): u=1.67 (106/127 combined) — ~16% SMSP friction specific to F2FP. Not present for PRMT+FFMA2 (u=1.95).
3. **Dispatch cap = 4.00 sm_inst/SM/cy** ✓ confirmed (FFMA scalar hits 3.88, mixed hits 3.95). However, V52 alu+fma sum=145% means cross-pipe accounting CAN exceed 100% per pipe — this is normal, not a violation of the 4.00 cap.
4. **HFMA2 + FFMA scalar** can co-exist but compete for H+L slots (~2.0 total warp-inst/SM/cy). Confirmed catalog reasoning matches V52 mechanism.

---

## §4. Rate cheatsheet — key entries (warp-inst/SM/cy → SASS-inst/SM/cy)

| Op | SASS/SM/cy | Logical |
|---|--:|---|
| Scalar FFMA | **128** ✓ | 256 FLOPS, dual-pipe heavy+lite |
| FFMA2/HFMA2/BF16-FMA | 64 | 128 FMAs |
| u32 ADD (IADD3 fusion) | 128 | 1 IADD3 = 2 adds |
| LOP3 / PRMT / SHF / FMNMX / HMNMX2 / VIMNMX3 | 64 | all pipe_alu, share |
| F2FP UNPACK (all formats) | 64 | 128 elements (×2 ops) |
| F2FP PACK | 32-64 | depends on feedback path |
| BFE | 32 | 2 SASS per PTX op |
| SHFL.SYNC.* | 32 | pipe_lsu |
| LDS/STS/LDG/STG | ~32 issue | DRAM-bound if streaming |
| MUFU (EX2/RSQ/SIN/...) | ~16 | pipe_xu, compound |
| F2I, POPC, BREV, FLO | 16 | pipe_xu |
| BAR.SYNC | ~12 | pipe_adu |
| MATCH.ANY | serial — VERY SLOW | pipe_adu |
| **FP64 FMA (DFMA)** | **1.6** | pipe_fp64, throttled |

---

## §5. Narrow-format throughput

At 128 elements/SM/cy × 148 SMs × 1.92 GHz = **36.4 Telements/s** for each UNPACK variant (FP4/FP6/FP8/UE8M0 → f16/bf16). Same number for all because they share the one ALU pipe.

For FP4 specifically: both UNPACK (`cvt.rn.f16x2.e2m1x2`) and PACK (`cvt.rn.satfinite.e2m1x2.f16x2`) live on the same 64 warp-inst/SM/cy ceiling as FP8/FP6 — **FP4 is NOT faster or slower per SASS instruction** than FP8 on B300's ALU pipe.

---

## §6. Uniform datapath (pipe_uniform)

Per-SMSP scalar unit operating on uniform registers (URx). Compiler uses it automatically for loop counters, kernel-arg propagation, warp-invariant scalars.

**Measured** (catalog claim): pipe_uniform hits ~1.0 warp-inst/SM/cy for ACTIVEMASK and LDSM. Does NOT contend with pipe_alu / pipe_fma — uniform ops issue in parallel.

**Blackwell adds** full uniform FP32 datapath (UFFMA, UFADD, UFMUL, etc.) — but ⚠ nvcc 13.0 and 13.2 do NOT emit these despite being in ISA. Either spec or aspirational. (REVIEW_CHECKLIST CRIT3 / R3)

---

## §7. ADU (pipe_adu)

Hosts slow warp-wide synchronization and status-register operations. Peak issue rate ~0.4-0.5 warp-inst/SM/cy for simple cases; wall-clock dominated by cross-thread waiting, not pipe throughput. No contention with ALU/FMA.

Key opcodes: BAR/BAR.SYNC/BAR.ARV/BAR.RED, CGA barriers, WARPSYNC/BSYNC/BSSY/BREAK/NANOSLEEP/YIELD, MATCH.ANY/MATCH.ALL, REDUX.SUM/OR/AND/XOR, MEMBAR.SC.GPU/SYS partial, S2R clock/timer.

---

## §8 + §9 — SASS opcode → pipe full classification

Catalog §8 lists every SASS opcode with pipe assignment. Catalog §9 lists PTX → SASS mapping for every ISA category. Both are lengthy reference tables; preserved verbatim in `B300_PIPE_CATALOG.md` L554-L897.

DENSE pruning notes:
- **Verified pipe placements** (this audit): pipe_fma (FFMA), pipe_alu (LOP3), pipe_xu compound (MUFU.SIN), pipe_xu simple (MUFU.EX2). All match catalog §8.
- **Unverified at this rig** (catalog claims): tensor pipes (HMMA, IMMA, QMMA, OMMA, DMMA, UTC*MMA), texture/surface (TEX, TLD, SULD, SUST), uniform pipe FP variants (UFFMA, UFADD, UFMUL), ADU detail, CBU detail.
- **Likely-correct propagated from H100/B200 docs** (Hopper-style opcodes still present): UBLKCP (TMA), UTMALDG/STG family.

For the full opcode table, refer to B300_PIPE_CATALOG.md L554-L897 directly. Most entries are inferred from opcode family rules (uniform-prefix → uniform pipe, etc.) — treat as best-effort but not all empirically tested.

---

## §10. L1/L2/HBM bandwidth ladder — replicated 2026-04-23 (justifications/00b_mem_hierarchy.md)

| Tier | Catalog claim TB/s | Measured 2026-04-23 | Verdict |
|---|---|---|---|
| smem `ld.shared.v4.u32` | 35.6 | **35.88 TB/s = 97.5% of 36.79 theoretical at 1942 MHz** | ✅ matches catalog exactly |
| L1 hit (.ca, WS≤1MB) | 36.1 | not yet measured | ⚠ DEFERRED — current benches mix L1/L2 |
| L2 plateau (4-128 MB, bs=512 mb=2) | 22-26 | **20.3 TB/s** (ncu `lts__t_bytes`) | ⚠ below upper end of catalog range; likely launch-config dependent |
| L2 → DRAM cliff at 126 MB | 8.2 | confirmed cliff (drops 13/9.8/7.8 at 128/256/1024 MB) | ✅ matches direction |
| HBM3E read WS≥1GB | 7.18 | **7.17-7.25 TB/s** across 2 recipes | ✅ matches catalog exactly |
| TMEM (catalog 55.92 read / 97.93 write) | — | DEFERRED (needs tcgen05.alloc setup) | 🔍 |

### Single-warp DRAM SoL anchor (this device)

- Denominator: **7,672 GB/s** post-ECC (this-device-actual at 7.992 Gbps × 7680-bit AC bus). NOT 8,000 GB/s spec — AC SKU has 1/16 controller fused.
- Best measured: 7.25 TB/s = **94.5% of 7,672 GB/s**.
- Wall-clock alone reports inflated 8.23 TB/s due to L2 absorption — **always cross-check with ncu `dram__bytes_read.sum.per_second`**.

### ⚠ NEW FOOTGUN — ncu warp-aggregated metric trap

`sm__sass_data_bytes_mem_shared_op_ld.sum` reports **warp-aggregated bytes** (warp_inst × 512 B for LDS.128), NOT per-lane bytes. Naive 16 B/inst accounting undercounts SMEM bandwidth by 32×. Easy to miss; add to methodology pitfalls.

### ⚠ NEW FOOTGUN — chain-feedback DCE

Chain-feedback patterns let the compiler DCE 32× of the LDS loop body even with anti-DCE store. To defeat: use INDEPENDENT loads with loop-counter-derived addresses + unconditional store. The existing `bench_lds_pure.cu` style does NOT work for v4 peak.

### Recompute: SMEM theoretical at 1942 MHz (rig DVFS)

128 B/clk/SM × 148 SMs × 1.942 GHz = **36.79 TB/s** (not 36.4 catalog @ 1.92 GHz).

Measured 35.88 / 36.79 = **97.5%** ✓ matches catalog's 98% claim closely.

### CRIT10 partially resolved: 126 MB cliff is real

Catalog claim "L2 cap at 126 MB, then 11 TB/s at 256 MB, 7.18 TB/s at 1 GB" is roughly confirmed:
- 126 MB exactly = `cudaDeviceProp.l2CacheSize = 132,644,864 B`
- 128 MB measured 13 TB/s (catalog says ~11)
- 256 MB measured 9.8 TB/s
- 1 GB measured 7.8 TB/s (catalog says 7.18 — close)

The "11 TB/s at 256 MB" was probably L2 partial-hit amortization at the boundary — explanation rather than mystery.

---

## §17. MUFU transcendental throughput — per-warp (catalog L7696+, 🟡 catalog claim)

8 independent chains, throughput per warp:

| Op | cy/op | Chip GOPS @ 1.92 GHz × 4 SMSP × 148 SM |
|---|--:|--:|
| **ex2.approx.f32** | **10.5** | **433** ← fastest |
| tanh.approx.f32 | 11.3 | 403 |
| sin.approx.f32 | 12.0 | 379 |
| cos.approx.f32 | 12.0 | 379 (same as sin — likely shared HW) |
| sqrt.approx.f32 | 13.9 | 328 |
| rsqrt.approx.f32 | 13.9 | 328 |
| lg2.approx.f32 | 13.9 | 328 |
| **rcp.approx.f32** | **15.5** | **294** ← slowest (counterintuitive — normally simplest) |

⚠ Catalog uses 1.92 GHz; rig DVFS settles at 1942 — chip GOPS are ~1% under-stated. Latency cy/op is clock-independent.

**Practical guidance:**
- Softmax: prefer `ex2` over `exp` (which is `ex2 × ln(2)`)
- Normalization: use `rsqrt × x` instead of `sqrt → rcp`
- Activations: `tanh.approx` is reasonably cheap (11 cy)
- For division: `__fdividef(a, b)` (= `div.approx`, 5.5 cy) is **3× FASTER than rcp(b) × a**

---

## §18. Branch divergence patterns (catalog L7724+, 🟡 catalog claim)

| Pattern | cy/iter | Notes |
|---|--:|---|
| No divergence | 28 | baseline |
| **2-way `if` (compiler-predicated)** | **23** ← FASTER than no-branch | compiler emits `selp`, no real branch |
| 2-way + `__syncwarp()` | 23 | sync no-op when no divergence |
| 32-way lookup table | 153 (5.5×) | local array indexed by lane |
| 4-way switch | 162 (5.8×) | compiler emits jump table |

**Key insight:** simple 2-way `if` branches are **faster than no-branch** because compiler turns them into predicated `selp`. True divergence appears only when compiler can't predicate (table lookup, function pointer, switch).

---

## §19. INT8 compute path (catalog L7744+, 🟡 CATALOG-PRESERVED)

⚠ `tcgen05.mma kind::i8` claimed NOT supported on sm_103a per catalog L6816 (cccl headers gate kind::i8 on sm_100a/100f/110a/110f only). **NOT independently verified by this audit** — would require trying to compile `kind::i8` PTX on this rig and observing ptxas reject. The NVFP4 agent's evidence file `49_nvfp4_ptxas_errors.txt` shows ptxas DOES reject several other `kind::*` variants on sm_103a, lending plausibility to the claim, but the exact `kind::i8` form was not in that test.

If verified, INT8 inference on B300 must use dp4a SIMD or convert to FP8.

| Op | cy/op | Chip TOPS | Effective use |
|---|--:|--:|---|
| **dp4a.{s32,u32,u32.s32}** (4×INT8 dot) | 5.25 | **54.5** | INT8 inference fallback |
| dp2a.{lo,hi}.s32 (2×INT16) | 5.25 | 25.4 | INT16 dot |
| mad.lo.s32 (IMAD) | 3.5 | 18.1 | scalar 32×32+32 |
| mad.wide.s32 (32×32→64) | 3.5 | 18.1 | free 64-bit accum |

### Comparison for INT8 inference on B300

| Path | TOPS | Notes |
|---|--:|---|
| tcgen05.mma kind::i8 | ❌ N/A | Not supported on sm_103a |
| dp4a SIMD | 54 | Slowest "modern" INT8 path |
| mma.sync m16n8k32 (FP8 emulated) | 309 | per JUSTIFIED §22 (catalog 276 was 12% LOW) |
| **tcgen05.mma kind::f8f6f4** | **4651** | **85× faster than dp4a** |

**Critical practical guidance:** for INT8 inference on B300, **convert to FP8 immediately** and use tcgen05.mma. dp4a is 85× slower than tensor-core FP8.

---

## §20. FMIN penalty under FFMA2 pressure (catalog L7773+, 🟡 catalog claim)

Direct A/B test of FFMA2 with interleaved instructions:

| Pattern | cy/iter | Overhead |
|---|--:|---|
| Pure FFMA2 (= 2 scalar FFMA / 1 inst) | 5.57 | baseline |
| FFMA2 + 1 IADD | 6.76 | **+21%** |
| FFMA2 + 1 scalar FFMA | 7.57 | +36% |
| FFMA2 + 2 FMIN | 9.45 | +70% (= +35% per FMIN) |

⚠ **Different from §22 dual-issue finding!** §22 settled "FFMA2 + LOP3 1:1 saturates all 3 pipes for net win". This catalog §20 finding says "+1 op of any kind costs 21-36% overhead at 1:1 ratio". Reconciliation: §22's "win" is in TOTAL useful ops (314 vs 187), but the FFMA2 portion DOES drop slightly (from 256 to 252 FLOPS/SM/cy, a 1.5% loss). The ALU work added (LOP3) more than makes up for it. Catalog §20's "+21%" is the FFMA2 *throughput* penalty, not total work.

**Design rule:** for *peak FFMA2 throughput specifically*, minimize pipe_alu instructions. For *peak total useful work*, FFMA2 + ALU at 2:1+ ratio is optimal (per §22).

---

## §21. tcgen05.mma sustained-load throttling cliff (catalog L7797+, 🟡 catalog claim)

| ITERS (continuous MMAs from one warp) | cy/iter | TFLOPS (148 SM) | % peak |
|---:|--:|--:|--:|
| 5,000 | 128.05 | 4654 | 100% |
| 10,000 | 128.02 | 4655 | 100% |
| 20,000 | 128.01 | 4655 | 100% |
| **30,000** | **128.01** | **4655** | **100% (cliff edge)** |
| 50,000 | 305.90 | 1949 | 42% |
| 75,000 | 364.71 | 1634 | 35% |
| 100,000 | 394.16 | 1512 | 32% |

**What's NOT happening (probed via nvidia-smi):**
- Clock stays at 1920 MHz (no clock throttle)
- Power only 193-197 W (nowhere near 1100 W TDP)
- Temp 40°C (cool, no thermal throttle)
- Forced clock-lock at 1920 MHz: no improvement (cliff persists)

**What IS happening:** dispatch bubbles inserted at SM level, NOT clock/power reduction. Mechanism candidates: hardware running-average power tracking inserts wait states ahead of any hard limit; tcgen05 internal queue/scheduler limits sustained issue rate; some sustained-utilization governor.

**Practical implication:** real GEMM kernels interleave MMAs with TMA loads / register reads / etc. — that work creates "idle time" for the tensor pipe and AVOIDS this throttle. The cliff ONLY appears in pure-MMA microbenchmarks. **Published peak TFLOPS in real workloads is achievable.**

---

## §22a. tcgen05 — known catalog DSMEM-related claim FALSIFIED

Catalog L7836-L7860 ("DSMEM Bandwidth & Atomic Costs", task #88) makes 3 claims that are **all WRONG per justifications/13_dsmem_exhaustive.md**:

| Catalog claim | Reality |
|---|---|
| "Load (u32) 25 cy local / 23 cy DSMEM = 0× free" | DSMEM read = 204-223 cy = **9× slower** (single-chain). With ILP=32, drops to 9 cy. |
| "Load (v4) 170 GB/s/SM local / 169 DSMEM = 0%" | DSMEM bandwidth depends heavily on cluster size + ILP + pattern; exhaustive sweep shows 5-43% of local SMEM aggregate. The 169 GB/s/SM was at one specific config. |
| "Cluster size doesn't matter (2/4/8 all identical)" | Cluster=2 latency 222 cy, c=4/8 at 207, c=16 at 231. Throughput per cluster scales linearly to c=8 (69→139→278), then drops at c=16. |

**DENSE recommendation:** delete catalog §30.H DSMEM rows; use justifications/13_dsmem_exhaustive.md as authoritative.

---

## §22c. CTA capacity formula (catalog L8198, 🟡 catalog claim)

| CTA threads | Warps/CTA | Max CTAs/SM | Warps/SM used | Notes |
|---:|---:|---:|---:|---|
| 32 | 1 | **32** (CTA limit binds) | 32 | **wastes 50% of warp slots** |
| 64 | 2 | **32** (both bind) | 64 | full occupancy ✓ |
| 128 | 4 | **16** (warp limit binds) | 64 | full occupancy ✓ |
| 256 | 8 | **8** | 64 | full occupancy ✓ |
| 512 | 16 | **4** | 64 | full occupancy ✓ |
| 1024 | 32 | **2** | 64 | full occupancy ✓ |

**Formula:** `max_concurrent_CTAs_per_SM = min(32, floor(64 / warps_per_CTA))`

⚠ For maximum SM utilization (64 warps), AVOID 1-warp CTAs — they waste 50% of warp slots. **128-thread CTAs are sweet spot for many kernels.**

---

## §22d. Cluster launch overhead — ✅ REPLICATED 2026-04-23 (via §22m audit, justifications/22m_launch_overhead.md)

Re-verified via §22m kernel launch audit. Numbers match catalog within rounding. **All cluster sizes 1/2/4/8 take exactly 2.05 µs in pipelined mode** — flat, no setup cost.

| Cluster size | µs/launch (pipelined, this rig) | catalog (per-iter event) |
|---:|--:|--:|
| 1 (single-CTA) | 2.05 | 5.7 |
| 2 | 2.05 | 5.7 |
| 4 | 2.05 | (interpolation) |
| 8 | 2.05 | 5.6 |

The catalog's 5.7 / 5.6 µs is the per-iter event mode (adds ~3 µs overhead per launch). The 2.05 µs pipelined number matches catalog's 2.0 µs (L8917).

**Cluster launch IS identical cost to single-CTA launch** — no setup overhead. Cluster activation verified at runtime via `%cluster_nctaid.x` write-back to C[0]. Use cluster freely when you need cross-CTA communication.

---

## §22e. SASS `.reuse` operand cache (✅ AUDIT-VERIFIED 2026-04-23 — justifications/22e_reuse_cache.md)

Direct count from this audit's preserved SASS files:
- **Scalar FFMA in §0.FFMA peak: 1023 of 1024 (99.9%)** carry `.reuse` (catalog's 94% claim is conservative)
- FFMA2 in §22 dual-issue audit: **82.8% to 99.2%** across 5 kernel configs (catalog 94% is mid-range)

Catalog claim text:
> "480 of 512 FFMA2 instructions (94%) carry `.reuse` annotation"

```
FFMA2 R22, R22.F32x2.HI_LO, R4.reuse.F32, 0.5 ;
FFMA2 R20, R20.F32x2.HI_LO, R4.F32, 0.5 ;
FFMA2 R18, R18.F32x2.HI_LO, R4.reuse.F32, 0.5 ;
```

The `.reuse` modifier signals an **operand-reuse cache** (separate from RF read ports). Pattern: one constant multiplier (R4.reuse), rotating destinations (R22, R20, R18, ...). Each FFMA reads R4 from reuse cache (free) instead of RF (1 port).

**Implication:** to approach FFMA2 peak, the compiler MUST find operand-reuse opportunities. Kernels with random source register access patterns will see lower throughput due to RF port saturation.

`.F32x2.HI_LO` modifier = packed FP32×2 dual-lane operation. The `HI_LO` swap creates butterfly-pattern dot products useful in tensor pipelines.

---

## §22f. L1/L2 cache granularity probe — ❌ catalog table FABRICATED (justifications/22f_stride_probe.md)

**Major finding 2026-04-23:** the catalog's "Sharp 64B stride break" table at L8231 is **essentially fabricated** — it collapses two different experiments into one table.

| Catalog claim | This rig (1800-locked, single-thread throughput) | Verdict |
|---|---|---|
| 56 cy at stride 4 B | **56.8 cy** ✓ | confirmed (best-case L1 throughput) |
| 56 cy at stride 8/16/32 (claims FLAT) | 87 / 136 / 160 cy — **MONOTONIC rise, never flat** | ❌ REFUTED |
| Sharp 5.4× cliff at stride 64 (56 → 304) | No cliff. Gradual rise 57 → 160 cy, then plateau | ❌ REFUTED |
| Plateau = 304 cy "L2 hit" | Plateau in throughput mode = **158-163 cy**; **328 cy in latency-chain mode** (catalog's separate L2-latency entry L111 = 301 cy — DIFFERENT test) | ⚠ catalog conflated 2 experiments |
| 128 B coalescing footprint | True unit is **32 B sector** (5 sectors at stride 4 = 32 lanes × 4 B; 32 sectors at stride 32 = each lane own sector) | ⚠ refined: 32 B sector, not 128 B line |
| Cache-line size 128 B | 128 B line = 4 × 32 B sectors | ✓ confirmed |

**Mechanism:** the 32 B sector-granular L1 access pattern means stride/throughput penalty grows continuously with stride, not as a cliff. The catalog's "sharp 64 B break" was created by combining single-thread throughput measurements (56 cy floor) with a separate pointer-chase L2 latency measurement (304 cy) — the cliff exists only on paper.

**DENSE recommendation:** delete catalog L8231 stride table. Replace with:
- "Single-thread throughput at stride S B (S ≤ 64): cy/load grows monotonically from 56 (S=4) to ~160 (S=32), then plateaus at 158-163 cy"
- "L2 pointer-chase latency (single dependent chain): 301-328 cy depending on warm-up state"
- "Per-lane access at stride > 4 B causes per-sector spillage (32 B sectors); warp-level coalescing benefits maxed at stride ≤ 4 B"

(Catalog claim preserved below for reference.)

---

Stride sweep (4096 loads after warm-up):

| Stride | cy/load | Tier |
|---:|--:|---|
| 4 B | 56 | L1 hit (warps coalesce to 128 B) |
| 8 B | 56 | L1 hit |
| 16 B | 56 | L1 hit |
| 32 B | 56 | L1 hit (still within coalescing) |
| **64 B** | **304** | **L1 miss → L2 hit (5.4× JUMP)** |
| 128 B | 316 | L2 hit |
| 256-1024 B | 316 | L2 hit (no further degradation) |

⚠ **Sharp break at 64 B stride** — beyond this, per-thread loads stop benefiting from warp-level coalescing. Each lane needs its own cacheline transaction.

This indirectly confirms: **the warp-level memory access "footprint" per `ld.global.u32` is 128 B** — when 32 lanes × 4 B fits within a 128 B aligned region, fast (56 cy). Stride > 64 B → loads spill into separate cachelines (304 cy = L2 hit).

⚠ Note conflict with §11 latency table: catalog claims LDS 33 cy, L1 43 cy elsewhere. Audit found LDS=29 / L1=38. The 56 cy here is for warp-level coalesced LDG (different metric). All compatible if regimes are clearly stated.

---

## §22g. tcgen05 SASS encoding — ✅ AUDIT-VERIFIED 2026-04-23 (justifications/22g_tcgen05_sass.md)

All 5 catalog opcodes confirmed in NVFP4 audit's preserved SASS files:
- `UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR5, UR5` ✅ verbatim (tcgen05.alloc)
- `UTCATOMSWS.AND URZ, UR5` ✅ verbatim (tcgen05.relinquish_alloc_permit)
- `UTCBAR [UR4], URZ` ✅ verbatim (tcgen05.commit.mbarrier::arrive)
- `UTCOMMA.BLOCK16 gdesc[UR4], gdesc[UR6], tmem[UR39], tmem[UR12], idesc[UR13], tmem[UR10], UP0` ✅ matches structure (tcgen05.mma; block-scaled FP4 variant emits `UTCOMMA.BLOCK16` instead of catalog's exemplar `UTCQMMA`)
- `UTCOMMA.2CTA.BLOCK16` ✅ confirms `.2CTA` modifier for `cta_group::2`

⚠ Family naming observed: `UTCOMMA` (block-scaled FP4) and `UTCQMMA` (catalog exemplar for f8f6f4 quad-MMA) and `UTCHMMA` (half-precision; not tested directly here). All share the UTC* prefix + UR*/UP0 uniform-pipe operands.

✅ **Uniform datapath claim CONFIRMED**: all observed UTC* use UR* operands and UP0 predicates (no per-lane R* registers). Executes on SM's uniform datapath, one issue per warp.

(Catalog opcode table preserved below.)

---

| PTX | SASS |
|---|---|
| `tcgen05.mma kind::f8f6f4` | `UTCQMMA gdesc[URx], gdesc[URy], tmem[URz], ...` |
| `tcgen05.mma kind::f16` | `UTCHMMA ...` |
| `tcgen05.mma kind::tf32` | `UTCHMMA ...` (same as f16, different idesc) |
| `tcgen05.mma cta_group::2` | `UTCQMMA.2CTA ...` |
| `tcgen05.alloc` | `UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR5, UR5` |
| `tcgen05.relinquish_alloc_permit` | `UTCATOMSWS.AND URZ, UR5` |
| `tcgen05.commit.mbarrier::arrive` | `UTCBAR [UR4], URZ` |
| `tcgen05.shift.cta_group::1.down` | (PTX-only known; 51 cy/shift) |

⚠ All UTC* instructions use **uniform register operands (UR0..)** and **uniform predicates (UP0..)** — they execute on the SM's uniform datapath, NOT per-lane. One issue per warp.

⚠ `pipe_tensor` ncu metric does NOT measure UTC*MMA. Only legacy mma.sync HMMA family. To measure tcgen05.mma rate, use wall-clock × cy/MMA × ops/MMA.

⚠ The `DEPBAR.LE SB0, 0x36` before UTCATOMSWS is a dependency barrier that waits on scoreboard slot 0 to drop below threshold 54 — ensures previous async ops complete before the alloc atomic.

---

## §22h. Compute-memory overlap — ✅ REPLICATED 2026-04-23 (justifications/22h_compute_mem_overlap.md)

**Qualitative claim CONFIRMED. Quantitative numbers CORRECTED.**

### Replication results (this rig, 1942 MHz, 11-SASS-files preserved)

| Pattern | Catalog cy/iter | This rig cy/iter | Verdict |
|---|--:|--:|---|
| Pure memory load (cold DRAM, LCG walk + l2flush) | 522 | **882 cy / 451 ns** | ⚠ catalog 1.7× LOW (catalog's "cold" was probably partial-cold) |
| Memory + 8 FFMA | 518 | **~877** | ✅ qualitative match — FFMA fully hidden |
| Memory + 16 FFMA | 522 | ~877 | ✅ still hidden |
| Memory + 32 FFMA | 530 | ~876 | ✅ still hidden |
| Memory + 64 FFMA | 580 | ~876 | ⚠ catalog says visible at 64; this rig still hidden |
| Memory + 128 FFMA | — | ~877 | ✅ still hidden! |
| Memory + 224 FFMA | — | ~876 | ✅ still hidden — bracket of "free" budget |
| Memory + 256 FFMA | — | **1,217 (jump)** | crossover; ⚠ ptxas register-SPILLED at 256 (21 LDL + 30 STL in SASS) |

**Warm L2 line (repeat-stride):** ~335 cy / 173 ns — the catalog's 522 is BETWEEN this rig's cold (882) and warm (335).

### Verified architectural conclusions

1. **FFMA IS fully hidden by cold-DRAM load latency** — `sm__cycles_active.avg` stays flat at 12,910 cy across N=0 to N=128, while FMA inst count grows 19×. Catalog's qualitative claim VERIFIED.
2. **Free FFMA budget is ~225 FFMAs per cold-DRAM load on this rig** — much higher than catalog's ~130, because the cold-DRAM load is 882 cy not 522.
3. **The exact crossover depends on register pressure** — at N=256, ptxas spilled to local memory (21 LDL + 30 STL added per iter), partially explaining the jump from 877 to 1217 cy. Above N=256 (N=512/1024), ptxas DCE'd the FFMA loop entirely.
4. **Dependent-load penalty (load address depends on prev value):** measured +2.1% over base; catalog said +5%; same ballpark.

### Catalog correction recommended

> Catalog should split "522 cy memory load" into **"cold DRAM (LCG, l2flush): 882 cy"** and **"warm L2 line (repeat-stride): 335 cy"** — the gap is 2.6×. Update "~130 FFMAs free" to **"~225 FFMAs at single-thread occupancy without register spill"**.

### Practical recipe (corrected)

If your kernel is memory-bound (cold DRAM), you can add **up to ~225 FFMAs per cold load** with zero added cost (vs catalog's 130). Beyond ~225, register pressure (>192 live floats) triggers spill cliff that adds local-memory latency on top of the FFMA itself. Sweet spot: keep N_FFMA ≤ 128 to avoid any risk of spill.

### Methodology rigor (full replication)

Per `justifications/22h_compute_mem_overlap.md`:
- 11 SASS files preserved at `justifications/22h_sass/N{0,8,16,32,48,64,96,128,256,512,1024}.sass`
- LCG walk over 256 MiB working set + `--l2flush 2` confirmed defeats prefetcher
- 30-run minimum per N_FFMA value
- Clock sampled: 1942-2032 MHz default boost (no lock)
- Runtime > 3.7 ms per run (well above launch-overhead floor)
- ncu `sm__cycles_active.avg` cross-check confirms FFMA addition adds ZERO cycles up to N=128
- SASS counts verified: FFMA count = N+1, LDG count = 1 per inner-loop iter for all N ≤ 224
- New test kernel: `tests/bench_compute_mem_overlap.cu`

---

## §22i. Per-GPC L2 latency variation (catalog L7593+, ✅ confirmed by DSMEM exhaustive)

L2 latency varies 25% across GPCs depending on which L2 slice serves which SM:

| GPC | Mean atomic latency cy | Range |
|---:|--:|--|
| 2 | **115** ← fastest | 110-126 |
| 8 | 114 | 40-135 (40 = warm L1) |
| 4 | 119 | 106-138 |
| 9 | 124 | 117-134 |
| 0 | 128 | 106-143 |
| 1 | 127 | 118-143 |
| 7 | 127 | 106-149 |
| 5 | 137 | 118-149 |
| 3 | **143** ← slowest | 126-152 |
| 6 | 143 | 125-153 |

⚠ For latency-critical primitives (locks, queues), this 25% variation matters. Use `%smid` to pin work to fast GPCs (2, 8, 4) when possible.

DSMEM exhaustive sweep similarly observed 20% per-GPC variation — both findings consistent with **address-hash-based L2 slice mapping**, not GPC topology.

---

## §22j. Smem store bank conflict sweep (catalog L7617, 🟡 catalog claim)

| Stride | cy/iter (128 stores) | Slowdown |
|---:|--:|--:|
| 1 (coalesced) | 33 | 1.0× baseline |
| 2 | 30 | 0.9× |
| 4 | 30 | 0.9× |
| 16 (16-way) | 64 | 1.9× |
| **32 (full 32-way conflict)** | **127** | **3.8×** |
| random | 33 | 1.0× ← **same as coalesced** |

Confirms 32-bank smem architecture. **Random patterns are AS FAST as coalesced** because random hashing distributes across all 32 banks.

Per-warp store throughput coalesced: ~32 stores per 33 cy = **0.97 stores/cy/lane** (essentially full LSU pipe).

---

## §22k. PTX special registers — ✅ AUDIT-VERIFIED 2026-04-23 (justifications/22k_ptx_special_regs.md)

Direct verification via 5-asm-instruction kernel reading registers into C buffer:
- ✅ `%nsmid = 148` (matches catalog)
- ✅ `%nwarpid = 64` (matches catalog)
- ✅ `%warpid`, `%laneid` = 0 for thread 0 of warp 0 in block 0 (correct semantics)
- ✅ `%smid = 142` for CTA 0 — **EXACT match to catalog L7551 claim "CTA 0 → SM 142"**

Bonus: this single 5-instruction kernel cross-corroborated §22n (CTA scheduler placement) AND the topology claim (CTA 0 landing on SM 142 in partial GPC 9 with SMs 142-147).

| Register | Value | Meaning |
|---|--:|---|
| `%nsmid` | 148 | Active SMs |
| `%nwarpid` | 64 | Max warps/SM |
| `%warpid` | 0..63 | Current warp ID |
| `%laneid` | 0..31 | Lane within warp |
| `%clock_lo` (32-bit) | wraps at 2^32 | Lower 32 bits of SM clock |
| `%clock64` (64-bit) | — | SM cycle counter (this rig 1942 MHz under sustained load) |
| `%globaltimer` (64-bit) | 32 ns granularity | Wall time in ns; **31.25 MHz tick fixed, doesn't change with SM clock** |
| `%cluster_nctaid.x` | cluster width | Set per launch |
| `%smid` | 0..147 | SM ID (use to pin work to specific GPCs/L2 slices) |

⚠ Catalog claims "SM clock 1920.0 MHz exactly" (L8138). Audit finding: actual DVFS settling under sustained-FFMA load is **1942 MHz** (not 1920 nor 2032). The 1920 in catalog may be a `nvidia-smi`-reported value at idle / different load condition.

---

## §22l. Grid sync overhead (catalog L7635, 🟡 catalog claim)

Grid sync via global atomic counter (no `cudaLaunchCooperativeKernel` API):

| Grid blocks | cy/sync | µs @ 1.92 GHz |
|---:|--:|--:|
| 8 | 4161 | 2.17 |
| 32 | 4129 | 2.15 |
| 64 | 4195 | 2.18 |
| **148** | **4245** | **2.21** |

Grid sync cost is **~constant at ~4200 cy = 2.2 µs**, regardless of grid size. Cost dominated by atomic acq_rel (~1500-1600 cy) + spin loop on phase var.

(Note: per audit §30.B, atomic acq_rel.gpu add scope penalty is 2.0-2.2× over relaxed, NOT the 31.3× catalog earlier claimed. So the "1598 cy acq_rel" attribution may be off — needs separate verification.)

---

## §22m. Kernel launch overhead — ✅ REPLICATED 2026-04-23 (justifications/22m_launch_overhead.md)

**Catalog's 2.0 µs vs 5.7 µs is NOT contradictory** — they're different timing modes (catalog L8395 already noted this; agent confirmed by reading QuickRunCUDA.cpp:529).

| Mode | This rig | Catalog | Verdict |
|---|--:|--:|---|
| Pipelined (overall/N, 2-event around N launches) | **2.05 µs** | 2.0 µs (L8917) | ✅ matches (zero variance over T=100k) |
| Per-iter event (`--timesPerRun`) | **5.20 µs** | 5.7 µs (L7654) | ✅ matches (9% under catalog) |

The 3 µs gap = per-launch event recording overhead in `--timesPerRun` mode.

### Kernel-size table — EXACT replication of L8385

| Inst count | This rig (µs) | Catalog (µs) | Cubin size this rig | Cubin size catalog |
|---:|--:|--:|--:|--:|
| 10 | **2.05** | 2.06 | 8.5 KB | 8.7 KB |
| 100 | **2.05** | 2.06 | 13.3 KB | 13.7 KB |
| 1000 | **4.10** | 4.11 | 62.7 KB | 63 KB |
| 4000 | **10.25** | 10.25 | 232.6 KB | 237 KB |

SASS-verified: FFMA inst counts exactly 10/100/1000/4000 in inner loops.

### Cluster launch = single-CTA launch (L8252 CONFIRMED)

| Cluster size | µs/launch |
|---:|--:|
| 1 (single-CTA) | 2.05 |
| 2 | 2.05 |
| 4 | 2.05 |
| 8 | 2.05 |

**Flat as a board** across cluster sizes — no cluster setup overhead. Cluster activation verified at runtime via `%cluster_nctaid.x` write-back to C[0].

### NOT verified by this audit

- **1.47 µs `cudaLaunchKernelEx + PSS`** (catalog L8918) — standalone harness shows `cudaLaunchKernel` ≡ `cudaLaunchKernelEx` at 7.18 µs each WHEN syncing per launch. Catalog's 1.47 µs requires batched-pipelined timing which would need a custom harness. **Marked "not refuted, not independently reproduced"** — separate audit task.
- **0.56 µs/kernel `cudaGraph × 1000`** (catalog L8919) — same: needs cudaGraph harness.

### Methodology rigor preserved

- `justifications/22m_launch_overhead.md` (13.7 KB)
- `justifications/22m_artifacts/` (11 files: A_empty / B_standalone / C_size_sweep / D_cluster_sweep / D_cluster_verify + per-test stdout)
- New test kernels: `tests/bench_22m_empty.cu`, `tests/bench_22m_size.cu`, `tests/bench_22m_cluster.cu`
- Clock sampled twice during T=100k run: 1942 MHz default boost throughout

---

## §22o. NVFP4 — ✅ REPLICATED 2026-04-23 (justifications/49_nvfp4.md — 364 lines + 14 evidence files)

**3 catalog claims CONFIRMED, 2 catalog claims CORRECTED:**

| Claim | Catalog | This rig | Status |
|---|---|---|---|
| **A: 9.9 PFLOPS at K=64 via `kind::mxf4nvf4.block_scale.block16`** | 9.9 PF (assumes 2032 MHz boost) | **9.26 PF at 1942 MHz** = 92.6% of 10 PF spec, 98.4% of theoretical at observed clock. cy/MMA = **128.001** matches catalog's 128.01 exactly. | ✅ confirmed |
| **B: K=96 via idesc bit 31 doesn't add MACs** | "K=96 same D[0] as K=64" | K=64 D[0]=288.0; K=96 D[0]=288.0 (bit-identical). If K=96 had worked, D[0] would be 432 (= 96/64 × 288). | ✅ confirmed |
| **E: 15/15 K=64 correctness** | all 15 tuples pass | All 15 (A, B, sA, sB) tuples produced bit-exact match: D[0] ∈ {0, 16, 288, 576, 1152, 2304, 18432, 36864} | ✅ confirmed |
| **C: `kind::mxf4` / `kind::mxf8f6f4` ptxas-rejected** | "all rejected by ptxas 13.2.78" | ⚠ **MOSTLY TRUE** but `kind::mxf4.block_scale.block32` actually **COMPILES** on V13.2.78 (emits SASS `UTCOMMA` without `.BLOCK16`). It crashes only at RUNTIME with "illegal instruction". | ⚠ catalog's "ptxas rejects" claim **FALSIFIED for this one form** |
| **D: only `128x128b` cp shape works** | "128x256b crashes" | ⚠ **128x256b ALSO RUNS CLEANLY** with 8 KB smem buffer. Bonus: `4x256b` also runs. Other shapes per catalog (64x128b/32x128b need .warptype, 64x256b/32x256b syntax error, 32x32b invalid). | ⚠ catalog's "128x256b crashes" claim **NOT REPRODUCED** |

### Catalog corrections recommended

1. **Claim A clock context**: catalog should specify "9.9 PF at 2032 MHz boost (theoretical)" vs "9.26 PF at 1942 MHz observed (this rig)". The 9.9 number is correct for 2032 MHz boost spec; readers may misinterpret it as a measured value at unspecified clock.
2. **Claim C `.block32` rejection**: revise from "all rejected by ptxas" to "rejected by ptxas EXCEPT `kind::mxf4.block_scale.block32` which compiles but crashes at runtime with illegal-instruction".
3. **Claim D `128x256b`**: revise from "crashes" to "WORKS with sufficient smem (8 KB+)". Likely the catalog's earlier crash was a smem under-allocation, not a fundamental shape limitation.

### Mechanism additions (NEW in this audit)

- **Test E used the simpler `smem-descriptor A path`** (NOT `tcgen05.cp + TMEM-A`). All 15 correctness tests still pass. **Catalog's claim that `tcgen05.cp + TMEM-A` is required is wrong** — the smem-descriptor path also produces correct results.
- ncu MODE0 cross-check confirms `sm__cycles_active = 100%` (saturated) and `sm__inst_executed.pct = 1.96%` (correctly low — only thread 0 of warp 0 issues UTCOMMA).

### Files preserved (full audit trail)

```
justifications/49_nvfp4.md (364 lines)
justifications/49_nvfp4_sass_MODE{0,1,3}.sass  ← UTCOMMA.BLOCK16 verified
justifications/49_nvfp4_utc_inst_MODE{0,1}.txt
justifications/49_nvfp4_ptxas_errors.txt        ← verbatim ptxas reject messages
justifications/49_nvfp4_correctness15.txt       ← 15 D[0] values
justifications/49_nvfp4_k96_correctness.txt     ← K=64 vs K=96 same D[0] proof
justifications/49_nvfp4_cp_shapes.txt           ← 128x256b runs (CORRECTION)
justifications/49_nvfp4_ncu_MODE0.txt
justifications/49_nvfp4_run_MODE0*.txt          ← 4 run logs
tests/bench_nvfp4_audit.cu (410 lines, 4-mode kernel)
tests/bench_tcgen05_cp_shape.cu (44 lines, shape sweep)
```

### Headline result

**B300 SXM6 AC FP4 tensor core via `kind::mxf4nvf4.block_scale.block16`:**
- **9.26 PFLOPS chip-wide measured at 1942 MHz** (98.4% of theoretical at observed clock, 92.6% of NVIDIA's 10 PF spec)
- SASS `UTCOMMA.BLOCK16` confirmed
- 128.001 cy/MMA, perfect linear scaling 1→148 SMs (per catalog L6776)
- K=96 ULTRA via idesc bit 31 does NOT work (the `kind::mxf4` proper path is rejected by ptxas; the `block_scale.block32` form compiles but crashes at runtime)
- Wait for newer NVCC point-release for proper K=96 / mxf4 support

---

## §22o-OLD. NVFP4 — original catalog content (preserved as audit reference)

### ⚠ K=96 ULTRA via idesc bit 31 is FALSIFIED — by the catalog itself

The PTX ISA (§9.7.16.2.1.1) documents K=96 as sm_103a-exclusive via idesc bit 31. Catalog tested it on `kind::f8f6f4`:

| K (idesc) | cy/mma | D[0] output (A=3.0, B=1.5) | Expected |
|---:|--:|---:|---:|
| 32 (default) | 64.46 | 144.0 = 32 × 4.5 | ✓ correct |
| 96 (bit31=1) | 64.46 (same!) | **144.0 = STILL 32 MACs (NOT 96!)** | ✗ |

Same finding on `mxf4nvf4.block16`:

| K (idesc) | cy/mma | D[0] output | Expected at K=96 |
|---:|--:|---:|---:|
| 64 | 128.01 | 589,824 | — |
| 96 (bit31=1) | 128.02 | **589,824 (SAME!)** | ✗ should differ |

**Catalog conclusion:** "K=96 likely requires additional hardware configuration (different TMEM layout, different A packing format, or specific B descriptor stride) that isn't triggered by just setting the idesc bit."

The `kind::mxf4` / `.block_scale` PTX form (proper K=96 path) is **REJECTED by ptxas V13.2.78** with "Illegal modifier '.block32'". The PTX ISA spec is complete; the ptxas codegen is the gap.

⚠ My failed NVFP4 K=96 ULTRA replication agent (token limit) was trying the wrong path. **The catalog already definitively answered this**: K=96 ULTRA via simple PTX is NOT accessible on this driver (NVCC 13.2). Wait for a newer NVCC.

### ✅ THE WORKING NVFP4 PATH — `kind::mxf4nvf4.block_scale.block16` = 9.9 PFLOPS

PTX (catalog L9275, verified to compile + run + give correct results):
```
tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16
    [d_tmem], [a_tmem], b_desc, idesc, [scale_A_tmem], [scale_B_tmem], pred;
```

SASS: **`UTCOMMA.BLOCK16`** — native block-scaled FP4 tensor instruction.

| Path | K | cy/mma | TFLOPS/SM | Chip | × f8f6f4 |
|---|--:|--:|--:|--:|--:|
| `kind::f8f6f4` E2M1 (baseline) | 32 | 128.02 | 33.29 | **4.9 PFLOPS** | 1.0× |
| **`kind::mxf4nvf4.block_scale.block16`** | **64** | **128.01** | **66.58** | **9.9 PFLOPS** | **2.0×** |

**This IS the real B300 FP4 tensor core throughput: 9.9 PFLOPS = 99% of NVIDIA's 10 PFLOPS spec.** ✓

Same 128 cy/mma, double the K (64 vs 32), double the FLOPs. Perfect 148-SM scaling.

### Critical operand differences from `kind::f8f6f4`

- A matrix from **TMEM** (via `tcgen05.cp` pre-load), NOT smem descriptor
- **Two separate scale TMEM addresses** (`[scale_A_tmem]`, `[scale_B_tmem]`) — one per matrix
- **No `{disable_lane_mask}`** operand — replaced by scale operands
- Scale factor type: UE8M0 or UE4M3 (per Table 57 of PTX ISA)

### What works vs what doesn't (ptxas 13.2.78)

| Syntax | Status |
|---|---|
| `kind::mxf4nvf4.block_scale.block16` | **✓ COMPILES** (UTCOMMA.BLOCK16 SASS) |
| `kind::mxf4.block_scale.block32` | ✗ ptxas rejects `.block32` |
| `kind::mxf4` (default = `.block32`) | ✗ same rejection |
| `kind::mxf8f6f4` | ✗ rejects `.block32` |
| `kind::f8f6f4.block_scale` | ✗ cannot combine |

Wait for newer NVCC point-release. The PTX spec itself is complete.

### tcgen05.cp setup for FP4 (catalog L9444)

**Key breakthrough:** `tcgen05.cp.cta_group::1.128x128b` correctly copies smem→TMEM for the A matrix.

| Shape | Status | Notes |
|---|---|---|
| `128x128b` | **✓ works** | 128 rows × 16 bytes = 2048 bytes |
| `128x256b` | ✗ crashes (illegal memory access) | — |
| `64x128b` / `32x128b` | ✗ needs `.warptype` modifier | — |
| `64x256b` / `32x256b` / `32x32b` | ✗ syntax error | — |

For K=64 FP4 (32 bytes/row), do **two** `128x128b` copies at column offsets +0 and +4. Requires ≥32 KB smem allocation.

### Correctness verified (15/15 tests)

A=3.0 B=1.5 scale=1.0 → D=288 = 64×4.5 ✓
A=6.0 B=1.5 → D=576 = 64×9 ✓
A=3.0 B=6.0 → D=1152 = 64×18 ✓

Output scales linearly with B value — confirms the MMA computes `D = scale_A × A × scale_B × B^T` correctly.

---

## §22p. Power efficiency — TFLOPS per watt (catalog L9128+, 🟡 catalog claim)

8K × 8K × 8K GEMM sustained, NVML power sampling:

| Precision | TFLOPS | Total power | Incremental over 180 W idle | TFLOPS/kW |
|---|--:|--:|--:|--:|
| Idle | — | **180** | 0 | — |
| TF32 tensor | 1,017 | 468 | 288 | 3,531 |
| **FP16 tensor** | **2,045** | **490** | **310** | **6,597** |
| **FP4 mxf4nvf4 K=64 random** | **9,900** | **661** | 481 | **15,000 ⭐** |

⚠ Catalog L9148 says "FP16 inference at 2045 TF / 490 W = 4.17 TFLOPS/W TOTAL" — that's the over-idle metric.

**Key observations:**
1. **Idle draws 180 W** — board baseline (memory refresh, fabric, PCIe, etc.)
2. **Tensor-core GEMM adds only 290-310 W above idle** for TF32/FP16 — compute fabric is power-efficient
3. **FP16 gives 2× the TFLOPS for ~7% more power** vs TF32 — nearly free to upgrade
4. **FP4 at 661 W = 15.0 TFLOPS/W** — 3.3× more efficient than FP8 random (4.5 TFLOPS/W)

### Cross-format power at random data (L9399)

| Format | Real PFLOPS | Zeros/const W | Random W | Δ random/const | TFLOPS/W (random) |
|---|--:|--:|--:|--:|--:|
| FP16 (kind::f16) | 2.5 | 470-484 | **1,099** | 2.3× | 2.3 |
| FP8 E4M3 (f8f6f4) | 4.9 | 487-497 | **1,092** | 2.2× | 4.5 |
| **FP4 mxf4nvf4 K=64** | **9.9** | 399-401 | **661** | **1.65×** | **15.0** ⭐ |

Block-scaled FP4 (`UTCOMMA`) draws **40% less power than f8f6f4/f16 (`UTCQMMA`)** with random data — and gives 2× the FLOPS. Clear win for inference.

### Data-pattern power sweep (mxf4nvf4.block16, L9377)

| Pattern | Power W | TFLOPS/W |
|---|--:|--:|
| zeros / constants | 399-401 | 24.8 |
| half-random half-zero | 540 | 18.3 |
| pseudo-gaussian | 602 | 16.3 |
| random (xorshift) | 661 | 15.0 |

Random data draws 65% more power than constant. Production inference (gaussian-like trained weights) ≈ ~600-800 W for FP4, ~900-1000 W for FP8/FP16.

### Bit-masking effect (L9407)

Zeroing low bits reduces power marginally (5-12%) — the upper bits still toggle randomly. FP8 low-nibble masking has almost no effect (-2%).

### Comparison to spec

| Format | Measured | NVIDIA spec | % of spec |
|---|--:|--:|--:|
| FP4 dense (mxf4nvf4 K=64) | **9.9 PFLOPS** | ~10 | **99%** ✅ |
| FP8 dense (f8f6f4) | 4.9 PFLOPS | 5 | 98% ✅ |
| FP4 = 2× FP8 via block-scaled path | ✓ confirmed | | |

---

## §22q. Register spilling cost (catalog L8358, 🟡 catalog claim)

| MIN_CTAS/SM | Avail regs/thread | N_LIVE=16 | N_LIVE=32 | N_LIVE=64 | N_LIVE=128 |
|---:|--:|--:|--:|--:|--:|
| 1 | ~232 | 2.25 | 1.79 | 1.68 | 1.61 |
| 2 | ~116 | 2.25 | 1.79 | 1.68 | 1.61 |
| 4 | ~58 | 2.25 | 1.79 | 1.68 | 1.61 |
| 8 | ~29 | 2.25 | 1.79 | 1.68 | 1.61 |
| **16** | **~14** | 2.25 | 1.79 | 1.68 | **2.44 (SPILL!)** |

**Spilling penalty: ~50% slowdown** (1.61 → 2.44 cy/FFMA) when the compiler can't fit all live values in registers.

**Practical:** don't use `MIN_CTAS_PER_SM` > 8 unless verified low register pressure. Spilling = local memory access ≈ 50% throughput hit.

---

## §22r. Atomic contention at scale (catalog L8446, 🟡 catalog claim — partial overlap with §13)

### Single address chip-wide

| Blocks (× 32 lanes) | cy/atom | Atoms/cy chip-wide |
|---:|--:|--:|
| 1 | 51 | 0.6 |
| 2-32 | 51 | 1.3-20 |
| **148 (full chip)** | **132** | **36 atoms/cy = 69 G atoms/s** |

L2 atomic unit handles up to 32 simultaneously-contending CTAs at 51 cy. Beyond that (148 CTAs), slows to 132 cy = 2.6× — surprisingly good given 100% contention.

### Distinct-address spread on 148 CTAs

| Distinct addrs | cy/atom | × 1-addr | Notes |
|---:|--:|--:|---|
| 1 | **126** | 1.0× | L2 atomic-unit MERGES same-address requests |
| **2** | **2,537** | **20× WORSE** | merging lost; 2 addrs serialize on same L2 slice |
| 4 | 1,246 | 10× | still bad |
| 8 | 549 | 4.4× | |
| 16 | 564 | 4.5× | |
| 32 | 593 | 4.7× | |
| 64 | 373 | 3.0× | parallelization across L2 slices |
| 256 | 258 | 2.0× | flat asymptote |

**Histogram/reduction design rules:**
1. Single global counter is surprisingly optimal if you need ONE number (L2 merging wins).
2. **Small counter arrays (N=2-32) are WORST case** — pick either N=1 or N≥256.
3. Per-warp / per-SM privatization (unique cacheline per warp) beats sharing.

### Atomic ordering × scope (cluster-launched contended)

| Op | cy/atom | × relaxed | SASS fence emitted |
|---|--:|--:|---|
| atom.add (relaxed) | 34 | 1.0× | none |
| **atom.release.cta** | **36** | 1.06× | (none — release IS the atomic write) |
| atom.acquire.cta | 734 | 21× | MEMBAR.ALL.CTA before ATOM |
| atom.acquire.gpu | 800 | 23× | MEMBAR.ALL.CTA + MEMBAR.ALL.GPU |
| atom.release.cluster | 892 | 26× | MEMBAR for cross-CTA scope |
| atom.acq_rel.cta | 810 | 24× | same MEMBAR as acquire |
| **atom.acq_rel.cluster** | **1,646** | 48× | heaviest membar combo |

⚠ Different from JUSTIFIED §30.B which measured 2.0-2.2× scope penalty in single-thread chain context. The 21-48× numbers here are for **contended cluster** workload — apples-to-different-oranges. Both correct in their regimes.

---

## §28. Compiler-emission gaps — ✅ AUDIT-VERIFIED 2026-04-23 (justifications/28_compiler_gaps.md)

✅ Catalog claim CONFIRMED: `uffma` PTX form rejected by ptxas V13.2.78. **Zero UFFMA/UFADD/UFMUL emissions** across ALL 20K+ preserved SASS files. Even uniform-looking C code (`x*2+1`) emits per-lane FFMA, not UFFMA.

⚠ NEW FINDING: catalog L2164's "compiler-reachable uniform ops" list is INCOMPLETE. Direct SASS opcode count reveals these uniform ops also appear:
- **UFU**: 91,950 instances (likely "uniform function unit" — uniform-pipe transcendental?)
- **USHF**: 8,404 (uniform shift)
- **ULEA**: 8,591 (uniform load-effective-address)
- **UFLO**: 902 (uniform find-leading-one)
- **UPRMT**: 887 (uniform permute)
- **UNC**: 4,033 (?)
- **ULT**: 10,502 (uniform less-than?)

Catalog should add these. UFU specifically is a load-bearing find — second-most-common uniform op after placeholder URZ/UPT — and not documented anywhere in the catalog.

---

## §22n. CTA scheduler placement pattern — ✅ REPLICATED 2026-04-23 (via DSMEM exhaustive, justifications/13_dsmem_exhaustive.md)

DSMEM exhaustive sweep used `%smid` PTX register and confirmed:
- **Cluster=8 picks SMs (0, 1, 16, 17, 32, 33, 48, 49)** — exactly 1 TPC (2 SMs) per GPC, GPC stride = 16
- **B300 SXM6 AC has 9 GPCs × 16 SMs + 1 partial × 4 SMs = 148** (catalog L7524 had it right; canonical doc had "8 GPCs" wrong)

Catalog's CTA scheduler placement order at L7546 is consistent with this finding (fills smallest/last GPC first, then round-robins 2 CTAs/GPC). Not separately re-tested but the topology underlying both sections is the same and is now solidly verified.

(Catalog claim preserved below for reference)
---

For 512 CTAs launched, the order CTA 0..15 → SM:
```
CTA  0 → SM 142   (partial GPC 9)
CTA  1 → SM 143
CTA  2 → SM 144   (last GPC, 4-SM partial)
CTA  3 → SM 145
CTA  4 → SM 146
CTA  5 → SM 147
CTA  6 → SM 0     (start GPC 0)
CTA  7 → SM 1
CTA  8 → SM 16    (GPC 1)
CTA  9 → SM 17
CTA 10 → SM 32    (GPC 2)
...
```

The scheduler:
1. Fills the smallest/last GPCs FIRST (CTAs 0-5 → SMs 142-147 in the partial GPC 9)
2. Then **round-robins 2 CTAs per GPC** across all GPCs (0,1 to GPC0; 2,3 to GPC0+16; etc.)
3. After hitting all 9 full GPCs (18 CTAs), starts a new pass

⚠ Don't assume `blockIdx.x` correlates with physical SM number. Use `%smid` for physical placement.

⚠ The GPC-aware scheduling can affect L2 partition pressure — adjacent CTAs may target same L2 partition. See §22i for per-GPC variation.

---

## §22b. TMA multicast on sm_103a (catalog L7864, 🟡 partial)

Catalog claims `cp.async.bulk.multicast::cluster` works on sm_103a despite cccl gating it to SM_90a/100a/110a. Wait latency per CTA after multicast:

| Cluster | Bytes | Wait cy | Effective BW |
|---:|--:|--:|--|
| (catalog had a table here — preserved verbatim in catalog L7869+ — replication TODO) | | | |

⚠ Multicast can amplify L2/HBM bandwidth (one DRAM read serves N CTAs in cluster). Catalog claims 14.9 TB/s aggregate multicast (V32 result) but this is L2-resident-source amplification, not DRAM peak. (REVIEW_CHECKLIST G4)

---

## §16. tcgen05.mma — Real Tensor Core Peak (catalog L6686+, NOT yet rerun)

⚠ Status: catalog content preserved here pending replication on this rig. The catalog presents results with strong methodology (linear scaling table 1-148 SMs verified separately), but this rig hasn't re-run them yet. Treat as 🟡 MED until replicated.

### Headline peaks (single warp per SM, M=128 N=256, 1000 MMAs)

| kind | Inputs | Output | K | cy/MMA | TFLOPS | % of NVIDIA spec |
|---|---|---|---:|--:|--:|--:|
| f16 dense | FP16/BF16 | FP32 | 16 | 128.1 | **2,325** | 93% of 2.5 PF |
| tf32 dense | TF32 | FP32 | 8 | 128.1 | **1,163** | 93% of 1.25 PF |
| f8f6f4 dense | FP8/FP6/FP4 | FP32 | 32 | 128.1 | **4,651** | 93% of 5 PF |
| f8f6f4 sparse | FP8 + meta | FP32 | 64 | 160.2 | **7,439** | 74% of 10 PF |
| i8 dense | INT8 | INT32 | — | — | — | **NOT SUPPORTED on sm_103a** (illegal-instruction; deliberate B300 spec — use FP8 instead) |

### Multi-SM scaling (catalog L6776) — perfect linear

| SMs | cy/iter | TFLOPS |
|---:|--:|--:|
| 1 | 128.12 | 31.4 |
| 8 | 128.25 | 251.1 |
| 64 | 128.15 | 2,011 |
| 148 | 128.20 | **4,648** |

Each SM's tensor pipe is independent. The 4.65 PFLOPS chip-wide IS supported by linear-scaling math (148 × 31.4 = 4647).

### Cross-kind ratio sanity check (CONFIRMED in catalog)

| Kind | TFLOPS | Ratio vs TF32 |
|---|--:|--:|
| TF32 | 1163 | 1.0× |
| FP16 | 2325 | 2.0× |
| FP8 | 4651 | 4.0× |

Exactly the expected 1:2:4 pattern from K=8 / 16 / 32.

### Multi-warp per SM is FULLY SERIALIZED (1 tensor pipe per SM)

Adding warps does NOT increase per-SM throughput:
- 1 warp: 31.41 TFLOPS/SM
- 4 warps: 31.44 TFLOPS/SM (no gain)

There is **exactly 1 tensor pipe per SM**. All 4 SMSPs share it; multi-warp issuance round-robins.

### Iteration count cliff (catalog L6802) — degradation at >10K MMAs

| ITERS | cy/iter | TFLOPS |
|---:|--:|--:|
| 100 | 130.26 | 4,575 (98%) |
| 1000 | 128.13 | 4,651 (100%) |
| 10000 | 128.02 | 4,655 (steady) |
| 100000 | **394.10** | **1,512 (33% — degraded)** |

Beyond ~10K MMAs in a single warp loop, throughput drops 3× — likely instruction-cache pressure or scheduling artifacts. Sweet spot is ~10K MMAs/launch.

### cta_group::2 for M=256 (catalog L6826)

cta_group::2 lets you process M=256 tiles by spreading across 2 SMs. Same total peak (4.65 PFLOPS) as cta_group::1 — does NOT unlock 2× peak. Use it when:
- A tile doesn't fit in 1 SM's smem
- Kernel requires M=256 for register-sharing reasons

### What was needed to make tcgen05.mma work (catalog L6741)

Critical methodology notes preserved verbatim:
1. **idesc encoding**: must use UMMA::InstrDescriptor bit layout (sparse_id2_ at [0,2), c_format_ at [4,6), a_format_/b_format_ at [7,13), n_dim_ at [17,23) /8, m_dim_ at [24,29) /16). `idesc=0` is invalid.
2. **smem matrix descriptor**: layout_type=0 (no swizzle): LBO=16, SBO=128.
3. **`tcgen05.alloc/dealloc/relinquish` are `.sync.aligned`** — must be called by ALL threads in the warp; behind `if (tid==0)` deadlocks.
4. **PTX form for cta_group::1 takes 9 operands** (no scale_input_d, no shift). Use 9-operand variant (`__cccl_ptx_isa >= 860`).
5. **Real mbarrier required** for `tcgen05.commit.mbarrier::arrive::one.b64`. Pointing at u32 instead of `mbarrier.init`'d 64-bit slot causes silent issues.
6. **M=256 fails with cta_group::1** — requires cta_group::2.

### ⚠ FOOTGUN — INT8 not supported on sm_103a

```
ptxas: Feature '.kind::i8' not supported on .target 'sm_103a'
```

Per cccl headers, `kind::i8` is gated on sm_100a / sm_100f / sm_110a / sm_110f only. **B300 (sm_103a) does NOT have IMMA** for tcgen05.mma. INT8 inference must use FP8 (or legacy mma.sync IMMA at 142 TOPS — much slower).

### Sparsity 1.6× not 2× (catalog L6864)

Dense FP8 (K=32): 128 cy/MMA → 4,651 TFLOPS
Sparse FP8 (K=64): 160 cy/MMA → 7,439 TFLOPS
Sparse provides **1.6× speedup** (not marketed 2×). HW does 2× logical work but takes 1.25× cycles per MMA — sparse path has higher internal latency.

7.44 PFLOPS / 10 spec = **74% of sparse spec** (vs 93% for dense). Possibly garbage sparse metadata in catalog test; properly-encoded 2:4 metadata should approach spec.

### REVIEW_CHECKLIST entries for tcgen05

- D5 (M=128 N=256 = 128 cy across all formats): verified by catalog table L6694-L6722 ✓
- D6 (FP4 K=64 = 9856 TFLOPS): catalog says ALL f8f6f4 formats give same TFLOPS (FP4 stored sub-byte but per-MMA throughput identical to FP8). The 9856 number is from a DIFFERENT path (`kind::mxf4nvf4` with block scaling); see §49 NVFP4 K=96 ULTRA.
- CRIT2 (single-warp scope mismatch): MITIGATED — Multi-SM scaling table L6776 demonstrates per-SM independence so 148×single-warp = chip-wide makes sense. Skeptical review framing was wrong here.

---

## §15. DSMEM (cluster shared memory) — REPLICATED 2026-04-23 (justifications/13_dsmem.md)

### ⚠ MAJOR FALSIFICATION — DSMEM is NOT "essentially free"

Catalog L7029-7031 / L7012 claim:
> "DSMEM is ~identical latency to local smem — the cluster interconnect on B300 is essentially free."
> "23 cy remote vs 25 cy local"

**This is FALSE.** Measured on this rig:
- DSMEM read latency: **204-223 cy**
- Local SMEM read latency: **23 cy**
- Ratio: **~9× slower**, NOT "essentially free"

This exactly reproduces catalog L2861-2864's own internal correction that was **never propagated forward**.

### SASS reveals the mechanism

`ld.shared::cluster.u32` compiles to **`LD.E`** (global load through cluster window), NOT `LDS`. This makes the 9× latency penalty mechanically obvious — DSMEM reads go through the global LSU path, not the SMEM bank-conflict-free path.

⚠ Update catalog §1 / §2.12: `ld.shared::cluster` SHOULD be in the LSU global-load category, not the LDS shared category. Many other DSMEM "claims" depend on this distinction.

### Write throughput — V53 CONFIRMED, V21 is misleading

| Test | This rig | V53 settled | V21 (older) | Verdict |
|---|--:|--:|--:|---|
| Sustained 18-cluster aggregate | **1.56 TB/s** | 1.47 TB/s | — | ✅ V53 within 6% |
| Per-cluster sustained | **87 GB/s** | 82 GB/s | — | ✅ V53 within 6% |
| 5-burst issue rate, NO fence | 427 GB/s/cluster | — | 560 | ⚠ V21 reproduces but is INFLATED — burst rate without completion fence |
| Same as above, WITH fence | **110 GB/s/cluster** | — | — | ⚠ 3.9× drop when properly fenced |

⚠ V21's 560 GB/s is the burst issue rate before completion. V53's 82 GB/s is the sustained completion rate after fence. Use V53 numbers in any practical context.

### L2 traversal — V53 CONFIRMED + NEW FINDING

| Path | ncu lts__t_bytes / data volume | Verdict |
|---|--:|---|
| DSMEM writes | 991 KB / 2.36 GB = **0.04%** | ✅ V53 confirmed: writes do NOT traverse L2 |
| Control GMEM writes | 2.55 GB / 2.55 GB = 100% | (control matches volume) |
| **DSMEM reads** | 1.05 MB / 2.36 GB = **0.04%** | **NEW FINDING**: reads ALSO don't traverse L2 (correcting V53 §6 footnote and older DSMEM_REFERENCE.md) |

So both DSMEM directions bypass L2. Mechanism: cluster-local interconnect (separate from L2 fabric). This is consistent with the SASS revelation — `ld.shared::cluster` is on the LSU but takes a separate physical path from regular global loads.

### "DSMEM BW = 99% of local SMEM" claim — FALSE

Catalog L7836-7860 says DSMEM aggregate BW is 99% of local SMEM. **DSMEM is 5-43% of local SMEM** depending on aggregation level. Recommend deleting the section, or rewriting with the actual numbers.

### Catalog inconsistency: 3 DSMEM sections, all different

- §13 (L7022+): "essentially free" — WRONG
- §30.H (corrected): partially right
- §13 (L7836+): "99% of local SMEM" — WRONG

Recommend consolidation. justifications/13_dsmem.md proposes the consolidated section.

### REVIEW_CHECKLIST entries resolved

- R1, R2 (skeptical review): "essentially free" claim → FALSIFIED
- §13 catalog claim 23 cy: → 204-223 cy
- V53 settlement: → CONFIRMED (and extended)

---

## §15a. DSMEM EXHAUSTIVE — full design space (justifications/13_dsmem_exhaustive.md)

### Vector width: v4 is 3.5× more efficient per byte

| Width | SASS | cy/load (cluster=2) | bytes | cy per byte |
|---|---|--:|--:|--:|
| u32 | LD.E | 222 | 4 | 55.5 |
| .v2.u32 | LD.E.64 | 257 | 8 | 32.1 |
| .v4.u32 | LD.E.128 | 261 | 16 | 16.3 |

v4 is only 17% slower than u32 for 4× the bytes → **3.5× more efficient per byte**. Always prefer v4 (or v8 if available) for DSMEM.

### Latency vs cluster size

| Cluster | Read latency cy/load | Notes |
|---:|--:|---|
| 2 | **222.7** | anomalous — slightly higher than 4-8 |
| 4 | 207 | minimum |
| 8 | 207 | same as 4 |
| 12 | (~218) | climbing |
| **16** | **231** | NON-PORTABLE — works with `cudaFuncAttributeNonPortableClusterSizeAllowed` |

⚠ NEW FINDING: cluster=16 IS ACHIEVABLE on B300 via the non-portable opt-in. Catalog § calling it "8 portable / 16 advertised" is too pessimistic — both work, with cluster=16 only 11% slower per access.

### ILP collapses DSMEM to LDS-equivalent

| Outstanding loads/warp (chains) | cy/load | Notes |
|---:|--:|---|
| 1 | 207 | latency-bound (single-chain serial) |
| 4 | 52 | 4× hidden |
| 8 | 26 | 8× hidden |
| 16 | 13 | |
| **32** | **9** | **23× speedup over latency-bound** |

DSMEM is mostly hidable with **8-16 outstanding loads per warp** — bringing effective cost into LDS-territory. The "9× slower" headline only applies to single-chain code.

### SM placement: GPC topology revealed

Via `%smid` PTX register, cluster=N picks SMs by fixed stride:
- cluster=8 → SMs **(0, 1, 16, 17, 32, 33, 48, 49)** — picks 1 TPC (2 SMs) per GPC
- GPC = 16 SMs (8 TPCs × 2 SMs/TPC)

⚠ NEW: B300 SXM6 AC topology is **9 GPCs × 16 SMs + 1 partial GPC × 4 SMs = 148 SMs**. The 4-SM partial GPC is the yield-binned "AC SKU" cell. Catalog's "8 GPCs" claim (canonical doc L482) is WRONG — it's 9 + 1 partial.

### ⚠ Per-GPC silicon variation — 20% spread

| GPC | DSMEM cluster=2 latency cy |
|---:|--:|
| 1 (SMs 16-31) | **229** ← slowest |
| 2 (SMs 32-47) | **189** ← 20% faster |
| (other GPCs) | between |

Real silicon variation across the chip — not all SMs are equal. Catalog framing "all SMs identical" is wrong.

### Write throughput — sustained, fenced, single-cluster

| Cluster | Single-cluster write SoL (no contention) | 18-cluster aggregate sustained, fenced |
|---:|--:|--:|
| 2 | 69 GB/s/cluster | (linear) |
| 4 | 139 | 105 GB/s/cluster (sub-linear with contention) |
| 8 | **278** | **117 GB/s/cluster (2.12 TB/s aggregate)** |
| 16 | 25 GB/s/cluster (12% lower than c=8) | (similar) |

**Single-cluster write SoL scales linearly with cluster size up to 8** (69 → 139 → 278). Above that, contention drops sustained per-cluster numbers.

### V21/V53 reconciliation

| Test | This sweep | V53 prior | V21 prior |
|---|--:|--:|--:|
| Burst (no-fence) ceiling, cluster=8 N=8 | **660 GB/s/cluster** | — | 560 |
| Sustained fenced, cluster=8 18-cluster agg | **117 GB/s/cluster (2.12 TB/s)** | 87 (1.56 TB/s) | — |

V21's 560 GB/s reproduces (within ~15%) as the burst ceiling. V53's 87 was at a specific stride pattern (all 128 threads to adjacent dwords); width-aware striding hits 117 GB/s/cluster (35% higher). **Catalog should quote a range, not a single number.**

### Fence cost

| Operation | cy |
|---|--:|
| 1 store, fenced | **1507** (738 ns at 1942 MHz) |
| 1 store, unfenced | **272** (140 ns) |
| Fence cost | **~1500 cy fixed**, independent of N_STORES |

⚠ The fence is the dominant cost for small batches. For sustained throughput, accumulate enough stores per fence to amortize the 1500 cy.

### R+W simultaneous = same total as W-only

R+W concurrent: 124 GB/s/cluster (vs 126 W-only). **Fabric arbiter is shared — no separate R/W channels** in the cluster fabric.

### L2 traversal RECONFIRMED

- DSMEM writes: **0.03%** of byte volume hits L2
- DSMEM reads: **0.05%** of byte volume hits L2
- Both directions bypass L2 via cluster-local interconnect (per V53 + previous DSMEM agent finding)

### Updated chip aggregate ceiling

With v4 width × cluster=8 × full chip × max-tuned:
- **Aggregate write: ~2.4 TB/s** (chip ceiling)
- **Aggregate read: ~1.9 TB/s** (chip ceiling)

Catalog's chip-wide DSMEM claims should reference these numbers, not the older V21 / V53 single-cluster numbers extrapolated naively.

---

## §14. Tensor cores (mma.sync legacy path) — REPLICATED 2026-04-23 (justifications/22_tensor_mma_sync.md)

| Path | Catalog claim | This rig | Verdict |
|---|--:|--:|---|
| `mma.sync.m16n8k16` BF16/FP16 | 569-578 TFLOPS | **571 TFLOPS wall, 570 ncu** (99.5% pipe_tensor) | ✅ within 1% |
| `mma.sync.m16n8k8` TF32 | 288 TFLOPS | **285.7 TFLOPS** | ✅ within 1% |
| `mma.sync.m16n8k32` FP8 e4m3 (emulated) | **276 TFLOPS** | **309 TFLOPS** | ⚠ **catalog 12% LOW** — should be **~308** |
| `mma.sync.m16n8k32` INT8 IMMA | 142 TOPS | **142.4 TOPS** | ✅ exact match |

### ⚠ MAJOR FINDING — FP8 mma.sync FADD artifact actually reproduces

The catalog's warning at L27 ("earlier 2336/2247 numbers were FADD artifacts; compiler DCE'd 99.99% of mma chain") is REAL. The naïve `bench_mma_all_precisions.cu` OP=3 collapses to:
- SASS: **2 HMMA + 1056 FADD** in inner loop
- Reports falsely high 2163 TFLOPS (counts FADD as if they were FP8 ops)

Anti-DCE test (`tests/bench_fp8_mma_peak_antidce.cu` — new) with chain-dependent inputs:
- SASS: **512 HMMA + 2052 F2FP** per inner iter (NOT QMMA)
- Confirms emulation via F2FP + HMMA path
- Measures **309 TFLOPS**

**Recommendation: bump catalog L27 from 276 → 308 TFLOPS for FP8 e4m3 emulated mma.sync.**

### Confirmation: B300 mma.sync FP8 is EMULATED (no native QMMA)

SASS dump confirms catalog's claim that FP8 via mma.sync is emulated. There is NO `QMMA` opcode in the output — only `HMMA` (FP16) preceded by `F2FP` (FP8→FP16 conversion). For native FP8, use **tcgen05.mma** (which emits `UTCQMMA` per project memory).

### Clock state during runs

All 4 tests at **1942 MHz sustained** (matches FFMA peak finding). No lock; pure DVFS settling point under tensor load.

### Notes

- `pipe_tensor` ncu metric is the right counter for mma.sync (HMMA family). It does NOT measure tcgen05.mma (UTC*MMA family) per project memory.
- INT8 IMMA pipe_tensor is only 12.3% because IMMA is ~8× slower per inst than HMMA at K=32 — IMMA is genuinely throttled on B300.

---

## §13a. TMA cp.async.bulk — REPLICATED 2026-04-23 (justifications/30_tma_sizes.md)

### Issue rate "48 cy floor" — half-right (catalog conflates 2 measurements)

| Measurement | cy/TMA | What it really is |
|---|--:|---|
| Pure single-issue | **~65 cy** (size-independent 16 B-8 KB) | one TMA, wait for completion, repeat |
| Amortized in N-batch (1 mbarrier × N TMAs) | **48-50 cy** (matches catalog 30.4b3 within 3 cy) | batching saves the per-TMA wait overhead |

⚠ Catalog's "48 cy size-independent issue floor" is the AMORTIZED rate, not the pure issue cost. Both are real but measure different things. Use 65 cy for single-issue cost; use 48-50 cy for batched.

### "Sharp 8 KiB crossover" — VERIFIED (in user-facing GB/s metric, NOT in cy/TMA)

| TMA size | cy/TMA | GB/s/SM (D=2) |
|---:|--:|--:|
| 16 B - 4 KB | 48-52 | 20-150 (still issue-bound) |
| **8 KB** | **65** | **255 ← sharp knee** |
| 16 KB | ~80 | ~240 (engine-bound) |
| 64 KB | — | 252 |

The `cy/TMA` curve looks gradual (48→52→65). But `GB/s/SM` jumps 20→40→79→150→**241** — sharp knee at 8 KiB where engine ceiling kicks in. Skeptical-review L4 was right about cy/TMA being gradual but wrong to conclude no sharp crossover.

### Per-SM peak verified at ~240-260 GB/s/SM

3 different configs all converge:
- 64 KB × DEPTH=3: 252 GB/s/SM @ 2.032 GHz (≈240 @ 1.92)
- 32 KB × DEPTH=4: 248 GB/s/SM
- 8 KB × NT=12 × DEPTH=2: 255 GB/s/SM

Engine-bound regardless of L2 vs HBM source (ncu confirms HBM-cold path also at 250 GB/s).

### ⚠ Chip-wide 21.9 TB/s claim is L2-HIT not DRAM (catalog wording fails to flag)

Catalog says 21.9 TB/s chip-wide via 4 KiB batched. Naive math: 158 GB/s/SM × 148 SM = 23 TB/s — but **HBM3E spec is ~7 TB/s, so 23 TB/s exceeds DRAM by 3×**.

Replication: chip-scale `bench_tma_throughput.cu` at 132 CTAs × 4 KiB × NT=24 caps at **6.4 TB/s (HBM-bound)**. The 21.9 TB/s requires L2 hits (small reused dataset).

### TMA vs LDG max-tuned — REPLICATED 2026-04-23 (justifications/30_tma_vs_ldg_max_tuned.md)

**Same kernel framework, same WS, same launch geometry, both max-tuned per their own ceilings:**

| Regime | LDG.E.128 max-tuned | TMA cp.async.bulk max-tuned | Gap | Winner |
|---|--:|--:|--:|---|
| **L2-hit (WS=64 MiB)** | **18.25 TB/s** | **20.49 TB/s** | **+12% TMA** | TMA |
| **DRAM-cold (WS=4 GiB)** | **7.41 TB/s** (96.5% of 7672) | **7.32 TB/s** (95.4%) | -1.2% (noise) | tied |

Both kernels exceed catalog's 13.3 TB/s "L2 wire" by 37-54% in L2-hit regime — **catalog L2 wire claim is significantly under-counted**.

User's reported "~22 TB/s TMA load" reconciles: 20.5 TB/s × (2032/1942 MHz) = 21.4 TB/s when scaled to true boost.

### Tuning knobs that mattered (max-tuning recipe)

**TMA winning recipe:**
- DEPTH=2 (pipeline overlap)
- bytes-per-iter ≥ 64 KiB per CTA (NTMAS × TILE)
- broad ridge: any 4-32 KB tile works as long as DEPTH=2
- ≥ 4 waves (~592 CTAs)

**LDG winning recipe:**
- BS=128 with many CTAs (≥8192)
- 256-bit per inst (LDG.E.128 = `v8.b32`)
- Per-warp 1-KB bursts
- ⚠ Larger BS lost up to 30% (BS=256/512/1024 all worse than BS=128)

### ⚠ NEW METHODOLOGY TRAP — ncu metric-vs-path mismatch

`lts__t_bytes` **undercounts true L2 traffic by 2.7×** for LDG L2-hit (MSHR/crossbar dedup). For LDG use `l1tex__t_bytes`; for TMA use `lts__t_bytes`. ⚠ If you mix these metrics across paths you'll get wildly inconsistent "% of L2 SoL" numbers.

Also: `.ca` (L1-cached) LDG hits 34.7 TB/s with 99.9% L1 hit — but that's measuring L1, not L2.

### Mechanism takeaways

- **TMA's 12% L2-hit advantage** likely from wider effective burst (TMA descriptor encodes 64 B – 128 B at-a-time vs LDG's per-inst 32 B). At L2 wire level the longer bursts amortize tag/coordination overhead.
- **No DRAM-side advantage**: HBM scheduler already coalesces LSU bursts, so TMA can't "do better" once memory is the bottleneck.
- **Both saturate at 1942 MHz** (rig DVFS settling point under sustained mem-bound load), not 2032 boost.

### Practical guidance

- For DRAM-bound kernels: pick whichever path fits your data layout. Both reach 95-97% of HBM SoL when max-tuned.
- For L2-hot kernels: TMA gives a ~12% edge. Worth the engineering complexity if you're L2-bound.
- For mixed: profile both with the metric-mismatch caveat above (use `l1tex__t_bytes` for LDG, `lts__t_bytes` for TMA).

### Bonus finding — silent zero-corruption bug

`tests/bench_tma_acquire_v2.cu` silently zero-corrupts C[0] when `data_xor == seed`. Always pass `-1 12345` to avoid the collision.

---

## §13. Atomics — REPLICATED 2026-04-23 (justifications/30B_atomics.md)

7 catalog inconsistencies resolved. Single-thread atom.global.add chain = **45 cy/op** (matches LDS chain at 45 cy — the "33 cy LDS" was throughput-derived, "24 cy" was constraint-folded loop; same hardware, different methodology — K6 is a labeling issue not a real inconsistency).

### Contention sweep (148 CTAs × 128 threads)

| Pattern | Throughput Gops/s | vs 1-hotspot |
|---|--:|--:|
| 1 hotspot (single addr, all 18944 threads) | 49.1 | 1× baseline |
| **N=2 addresses** | **1.69** | **29× SLOWER** ← real anomaly (catalog said 32×, T6 confirmed) |
| N=4 addresses | (faster than N=2) | — |
| Per-warp clean (`addr_idx = warpId`) | **53.7** | **1.09× FASTER** ← contradicts catalog "5× slowest" |
| Per-CTA pattern | **609** | **12.4× FASTER** ← contradicts catalog L2708 "same as single" |
| Coalesced unique-per-lane | **221.4 = 0.023 atom/cy/lane** | 4.5× | NOT 0.94 as catalog claimed |

⚠ Catalog's "per-warp = 5× slowest" claim was measured on a within-warp-divergent variant. Clean per-warp is fine.

### Scope penalty .relaxed vs .acq_rel

Catalog claimed: 31.3× penalty (51 cy → 1598 cy). **WRONG — apples-to-oranges** (compared chip-throughput vs single-thread chain).

Real penalty (apples-to-apples):
- warp-contend: 2.03× (738 → 1501 cy)
- chip-wide: 2.22× (23.2 → 51.3 cy/warp-atom)
- single-thread: within L2-side noise

⚠ The "FREE for .cta/.gpu/.sys" sub-claim is **CORRECT** for L2-hit (no scope penalty among .cta/.gpu/.sys when contending on L2-resident data).

### FP atomics

- `__half` / `__nv_bfloat16` atomicAdd → SASS `ATOM.E.CAS.STRONG.GPU` loops (verified). **6.3× slower than u32** (NOT 45× as catalog claimed).
- Packed `f16x2` / `bf16x2` PTX atomics → NATIVE `REDG.E.ADD.F16x2` SASS. **Within 12% of u32**.
- `atom.global.add.f32` is **24% FASTER than u32 chip-wide** (!).

### NEW METHODOLOGY TRAP — atom vs red SASS distinction (NUANCED, see followup)

Original §30B claim was "`atom.global.add` always compiles to `REDG.E.ADD.STRONG.GPU`, NOT `ATOM.*`". This is **PARTIALLY WRONG** per `justifications/30B_atomics_FOLLOWUP.md`.

Direct SASS grep across all preserved kernels shows ALL THREE opcodes are emitted by the compiler depending on context:
- **`REDG.E.ADD`** — when atomic return value is DISCARDED (semantically `red.add`)
- **`ATOMG.E.ADD`** — when return value is USED with default scope
- **`ATOM.E.ADD`** — for some scoped variants (esp. with `STRONG.GPU` scope and certain address patterns)

The §30B chip-wide throughput numbers (49.1 / 53.7 / 609 / 221 Gops/s) were measured against `bench_atom_chip_scope` which emits **ATOM.E.ADD.STRONG.GPU**, not REDG. The throughput numbers are valid; the SASS-name attribution was wrong.

ncu metric implications:
- If your kernel emits REDG → use `lts__t_sectors_op_red`
- If your kernel emits ATOMG.E or ATOM.E → use `lts__t_sectors_op_atom`
- **Best practice: capture BOTH counters and add them** (covers all variants)

⚠ Confirmed correct: `atom.f16/bf16 atomicAdd` does emit `ATOM.E.CAS.STRONG.GPU` loops (CAS-emulation). Packed `f16x2/bf16x2` PTX emits `REDG.E.ADD.F16x2` natively.

### Replication summary by REVIEW_CHECKLIST entry

| Entry | Catalog claim | Verified? |
|---|---|---|
| K2 "CAS unconditionally half-rate" | 0.50 vs 1.00 | likely true (not retested in this audit) |
| K6 "atom 45 cy = LDS but LDS = 33 cy" | inconsistent | ✅ RESOLVED (labeling issue; same 45 cy under same methodology) |
| K7 "warp-coalesce 12× slower than unique" | 12× | ⚠ "12×" wrong direction — coalesced unique = 0.023 atom/cy/lane ≪ 0.94 catalog |
| T1 "scope FREE for L2-hit" | true | ✅ CONFIRMED |
| T2 "31.3× scope penalty" | 31.3× | ❌ WRONG — real is 2.0-2.2× (apples-to-apples) |
| T4 "atom.f16/bf16 ~45× slower" | 45× | ❌ WRONG — real is 6.3× (catalog 7× too high) |
| T6 "N=2 anomaly 20× worse" | 20× | ✅ confirmed (29× this rig, same direction) |

---

## §11. Latency reference table (clock64-bracketed) — verified 2026-04-23 (justifications/24_latency_table.md)

§24 of the catalog is **~75% accurate within ±15%**, with 6 specific entries needing fixing. Verified entries below; corrections in **bold**.

| Op | Catalog claim cy | This rig cy | Verdict |
|---|--:|--:|---|
| FFMA / FMUL / FADD | 4 | 4.2-4.4 | ✅ matches |
| HFMA2 / LOP3 / SHF | 4 | 4.2-4.4 | ✅ matches |
| **DFMA** | 92 (L103) / 63.9 (L460) | **63.7** | ⚠ L103 WRONG; L460 RIGHT |
| IMAD.HI.U32 | 13 | (TBD) | needs follow-up |
| MUFU.EX2 (simple) | 14 | 14 | ✅ matches |
| MUFU.SIN/COS (compound) | 24 | 24 | ✅ matches |
| MUFU.RSQ/SQRT/LG2 ftz | 18 | 18 | ✅ matches |
| MUFU.RSQ/SQRT/LG2 non-ftz | 40 | ~40 | ✅ matches (includes range-reduction) |
| **redux.sync.min/max** | 18 | 18 (CREDUX.MIN/MAX) | ✅ matches |
| **redux.sync.add/or/and/xor** | (catalog says 18) | **44 cy (REDUX.SUM/OR/AND/XOR)** | ⚠ NEW FINDING — 2.4× slower than min/max; catalog only documents min/max latency |
| SHFL | 24 | 24 | ✅ matches |
| **LDS hit** | 33 | 29 | ⚠ 14% high in catalog |
| **L1 hit (.ca)** | 43 | 38 | ⚠ 14% high in catalog |
| L2 | 300 | (TBD verified) | likely matches |
| **DRAM cold** | 789 (L112) / 3000 (header) | **789** verified | ⚠ "3000 cy" header WRONG (came from unrelated 2-SM topology metric) |
| **__syncthreads BS=512** | 45 (L74) / 12+2W=44 (L116) | **54** | ⚠ both wrong; correct empirical formula is **`22+2W`** |
| **fence.sc.gpu** | 274 (L115) / 544 (header) | **281** | ✅ L115 close (281 vs 274); header WRONG |
| fence.sc.cta | 8.6 | 8 | ✅ (per §30.G replication) |
| **mbarrier RTT** | 54 (header) | **123** | ⚠ 54 was arrive-only dispatch, not full arrive+test_wait round-trip |
| tcgen05.mma N=256 | 128 | (TBD) | needs separate replication |

### NEW FINDING — redux.sync split into TWO different SASS instructions

| PTX | SASS | Latency |
|---|---|--:|
| `redux.sync.min/max.u32` | `CREDUX.MIN`/`CREDUX.MAX` | **18 cy** (compact) |
| `redux.sync.add/or/and/xor.b32` | `REDUX.SUM`/`OR`/`AND`/`XOR` | **44 cy** (2.4× slower) |

Catalog only documents min/max latency at 18 cy. The add/or/and/xor variants are 2.4× slower because they emit a different SASS opcode family. This is a NEW finding from §24 audit.

### NEW METHODOLOGY TRAP

Default `LDG.E` (compiler-emitted for plain `ld.global`) **hits L1 even for "DRAM" tests** unless you BOTH:
1. Use `ld.global.cg` (or `.cs` / `.lu`) to bypass L1, AND
2. Run >500K-hop Sattolo-shuffled chains over working sets exceeding L2 (>126 MB)

Without both, L1/L2/DRAM all collapse to ~38 cy (you're really measuring L1). Several existing benches in `tests/` (e.g. `bench_v6_g2b_dram_latency.cu`) suffer from `init` and `main` kernel arg-conflict that silently shrinks the working set. ⚠ Add to footgun list.

### __syncthreads formula correction

Catalog says `12 + 2W` cy at L116. **Empirical: `22 + 2W` cy.** At W=16: catalog predicts 44, measured 54. At W=8: catalog predicts 28, measured ~38.

The "+10 cy" delta is constant across W ∈ {2, 4, 8, 16, 32} — suggests a fixed barrier-instantiation overhead the catalog formula missed.

---

## §12. Memory fence costs — RESOLVED 2026-04-23 (justifications/30G_fence.md)

Catalog had inconsistent values across L114-L3635 (40× spread for cta, 6× for gl, 3× for sys). Single-GPU B300 SXM6 AC authoritative ladder:

| Fence | cy/op | ns @ 2032 | Notes |
|---|--:|--:|---|
| `__threadfence_block` (cta) | **8.00 ✅** | 3.9 | zero variance across 5 runs; matches V54 exactly |
| `__threadfence` (gl) | **267.2 ✅** | 131.5 | R² = 1.0 across 5 runs; matches V54 exactly |
| `__threadfence_system` (sys) | **1727 ✅** | 850 | single-GPU rig — ⚠ V54's 2806 was 2-GPU NVLink rig; single-GPU is 1.62× lower (one fewer NVLink coherence round-trip) |

### NEW finding — first-fence-after-write tax is FIXED, NOT linear

Catalog L3084 said gl fence cost is "+150 for 1st write, +60/write after" — implying linear scaling.

**Replication shows it's a one-time fixed L2-drain overhead, NOT linear:**
- 1 fence + 0 writes: 265 cy
- 1 fence + 1 write: 400-750 cy (variable run-to-run)
- 1 fence + N writes (N=2..128): FLAT — same as N=1

Mechanism: the first store after a fence triggers L2 drain; subsequent stores ride the open drain pipeline. The "+60 cy/write after" claim was an N-issue artifact in the original V54 test (which varied N inside the timed loop without isolating writes from fences).

### Catalog reconciliation table

| Catalog citation | Claimed cy | Measured cy | Verdict |
|---|--:|--:|---|
| L114 cta=8.6 | 8.6 | 8.0 | ✅ within noise |
| L116 gl=274 | 274 | 267 | ✅ within noise |
| L2889 cta=29 | 29 | 8 | ❌ 3.6× high — wrong |
| L2889 gl=282 | 282 | 267 | ✅ within noise |
| L2890 sys=2890 | 2890 | 1727 | ❌ 1.67× high (multi-GPU artifact) |
| L2914 sys=2914 | 2914 | 1727 | ❌ same |
| L3068 "8 parallel sys channels" | claim | n/a | unverified — needs separate test |
| L3083 cta=14 | 14 | 8 | ❌ 1.75× high |
| L3084 gl=271 +60/write | 271 + linear | 267 + fixed | ✅ base ❌ scaling |
| L3085 sys=2882 | 2882 | 1727 | ❌ 1.67× high |
| L3193 acq_rel.sys 17-37% > sc.sys | claim | unverified | needs sweep |
| L3632 cta=337 (full chip W=16) | 337 | n/a | DIFFERENT scenario (multi-SM busy chip pre-load) — keep separate |
| L3632 gl=1679 (full chip W=16) | 1679 | n/a | same — different scenario |
| L3635 sys=8869 (full chip W=16) | 8869 | n/a | same — different scenario |

### DENSE recommendation

For single-CTA / single-warp / no-busy-chip context (the most common audit context), USE:
- cta = 8 cy / 3.9 ns
- gl = 267 cy / 131.5 ns (+~280 cy if there's a pending write to drain)
- sys = 1727 cy / 850 ns (single-GPU) **or** 2806 cy (NVLink-attached multi-GPU)

For full-chip busy-load context (W=16+ pending writes, all SMs active), use the L3625-L3635 numbers — but treat them as a DIFFERENT measurement that should not be reconciled with the single-warp numbers.

⚠ The "+60 cy/write" linear scaling claim from catalog L3084 is RETRACTED — it's a fixed one-time L2-drain.

---

(More sections added as agents finish replication.)
