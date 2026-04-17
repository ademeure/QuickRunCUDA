# CRITIQUE — B300_PIPE_CATALOG.md Audit

A ruthless, specific audit of the 19,704-line B300 SXM6 AC (sm_103a) catalog. The catalog is produced by an autonomous loop and is internally inconsistent in many places: numbers are revised multiple times, sections contradict each other, and methodology issues are documented-then-forgotten. This critique flags what to trust, what to fix, and what to re-investigate.

**Top-level sanity check**: the catalog's own warning banner at line 3-7 is correct — "some of it is definitely wrong or misleading." Treat this file as a research notebook with many half-retractions, not a polished reference.

---

## 1. Confidence Re-Rating

Rating scale: HIGH (cross-checked, solid methodology), MEDIUM (plausible, single test), LOW (methodology concerns), WRONG (contradicted by later data or impossible), UNVERIFIED (no independent check visible).

### Headline Compute Peaks

- **FP32 scalar FFMA 71.8 TFLOPS @ 98.8% SOL (sec 0 / line 30)** — HIGH: verified by audited kernel with SASS count, ncu pipe_fma=99.08%, cross-checked by 60.5/65/68.7 TFLOPS with progressively worse methodology elsewhere. But note clock ambiguity below (#15).
- **FFMA2/HFMA2/BFMA2 = 72.3 TFLOPS each (line 31-34)** — HIGH: same pipe, consistent numbers across audits. Sec. 17 self-identifies the earlier "HFMA2 = 2× FFMA FLOPS" claim as wrong.
- **FP16/BF16 tensor via mma.sync 577 TFLOPS (line 25)** — MEDIUM-HIGH: but **LABELED "TENSOR FP16"** in section 0 ambiguously; this is the **legacy** mma.sync ceiling only. The catalog retracts the "tensor cores work = 514 TFLOPS" headline (AUDIT_NOTES.md line 17-40) multiple times.
- **TF32 tensor 288 TFLOPS via mma.sync (line 26)** — MEDIUM: same methodology concerns; listed as corrected from "wrongly 141" earlier, so prior catalog had errors.
- **FP8 tensor via mma.sync 276 TFLOPS (line 27)** — MEDIUM: line explicitly flags this as emulated (F2FP + HMMA). The catalog later revises up ("2336 TFLOPS") then back down; this is *not* native FP8.
- **tcgen05.mma FP16 2325 TFLOPS @ 93% of spec (sec Tensor Core Peak)** — HIGH: multiple shape sweeps agree, MN=128×256 gives same cy/MMA across formats, cross-check with FP8/TF32 ratios matches expected K-scaling.
- **tcgen05.mma FP8 (kind::f8f6f4) 4651 TFLOPS (sec Tensor Core Peak)** — HIGH: 148-SM scaling is perfectly linear (128.13 cy/MMA at any SM count), correctness verified via data-sensitive output.
- **tcgen05.mma FP4 (kind::mxf4nvf4.block_scale.block16) 9.9 PFLOPS (line 9271)** — HIGH: correctness verified with 15/15 tests of arithmetic identities, SASS confirmed `UTCOMMA.BLOCK16`, linear data-scaling. This is the strongest single result in the catalog.
- **FP8 sparse tcgen05 7.44 PFLOPS = 74% of spec (sec Tensor Core Peak)** — MEDIUM: catalog itself notes the sparse metadata may be garbage; "properly-encoded 2:4 metadata should approach spec." UNVERIFIED claim as written.
- **FP64 DFMA 0.95 TFLOPS (line 35)** — HIGH: cross-checks with per-warp 92-cy latency, zero pipelining, cudaDeviceGetAttribute perf ratio = 64.
- **INT8 IMMA 142 TOPS (line 28)** — MEDIUM: real measurement but the catalog says IMMA emits 5 NOPs between each issue, which is a strong claim that needs SASS re-verification.
- **`__dp4a` 49-54 TOPS (sec Video/byte-SIMD)** — MEDIUM: plausible, cross-checked as 1058 inst/ns × 4 MACs × 2 ≈ 54 TOPS, but several numbers across the catalog disagree on whether this is 49, 54, or 271 TOPS.

### Memory Bandwidth Claims

- **HBM3E read peak 7.0-7.4 TB/s (section 0, sec 30.4)** — HIGH: cross-verified by cudaMemset 7.37 TB/s, cuBLAS GEMV 7.21 TB/s (some even shows 103%!), ncu `dram__bytes_read.sum` confirms. Multiple numbers in [7.0, 7.48] TB/s all plausible and converge.
- **HBM write 3.4-7.5 TB/s depending on width (line 46 note vs sec "DRAM read vs write")** — CONTRADICTION: section 0 says write = 7.09 TB/s; later section says 3.4 TB/s for v4, 8.5 TB/s for v8. These contradict without cleaner resolution. MEDIUM confidence in the right ballpark.
- **L2 peak 22-26 TB/s (section 0)** — MEDIUM: revised upward from earlier "10.2 TB/s", but catalog also states "20 TB/s", "17 TB/s", "14 TB/s" in different sections. See contradictions #3.
- **L1 peak 36 TB/s / 244 GB/s per SM (section 0)** — MEDIUM: matches theoretical `128 B/clk × 148 × 1.92 GHz = 36.4 TB/s`. But peaks appear only for 1 MB working sets (below true cache size 192 KB/SM × 148 = 28 MB), so this is "L2 with L1 hints" more than pure L1.
- **Smem v4 read 35.6-37.7 TB/s / 241 GB/s per SM (section 0, sec ~17.6)** — HIGH: explicitly triple-audited, theoretical is 38.5 TB/s, measured 98% — the methodology correction (ld.volatile) is compelling.
- **Smem 19.85 TB/s (AUDIT_NOTES.md)** — contradicts the above 35-37 TB/s figure. The 19.85 is from shmem_peak.cu (52% of theoretical), while 35 TB/s is from a later ld.volatile test. Catalog never reconciles.
- **TMEM read 55.92 TB/s / 60 TB/s chip (line 50, sec 16)** — MEDIUM: the catalog itself retracts the earlier "295 TB/s" and "830 TB/s" claims as DCE-inflated. 60 TB/s is self-consistent across a few shape sweeps.
- **TMEM write 97.9-131 TB/s (line 50)** — MEDIUM: consistent with read/write asymmetry (writes go into the pipe from scatter; read is the tighter path), but single-source measurements.
- **Constant mem broadcast 17.8-33.7 TB/s effective (line 47-48)** — MEDIUM: inflated by "effective BW = 32 × actual", catalog notes only ~0.55 TB/s actual cache traffic. The headline is semantic not physical.
- **NVLink 757 GB/s unidirectional / 1503 bidirectional (sec Multi-GPU)** — HIGH: matches 18 × 53.125 = 956 GB/s spec × 84% efficiency; cross-check of DMA and kernel-side P2P both reaching 755 GB/s after thread-count correction. Strongest multi-GPU result.
- **DRAM latency 789-860 cy / 400-423 ns (sec 17, catalog has multiple values)** — MEDIUM: range is large (789 to 14000 cy depending on TLB state); reasonable.
- **L2 latency 301-307 cy / 157 ns (sec 17 and scattered)** — MEDIUM: consistent in several places.
- **L1 latency 39 cy / 19-20 ns (sec 17)** — HIGH-ish: consistent across at least 5 different measurements.
- **Smem latency 24 cy / 12.5 ns (multiple places)** — HIGH: consistent.

### Microarchitecture / Pipe Topology

- **4 SMSPs per SM, 4.00 warp-inst/SM/cy dispatch cap (sec 1)** — HIGH: standard Blackwell knowledge, cross-verified.
- **pipe_fma dual-issue (heavy+lite) sustaining 128 SASS/SM/cy for scalar FFMA (sec 2.1)** — HIGH: explicit ncu metric confirmation, reproduced under multiple occupancy conditions.
- **ALU pipe 2.00/cy cap, many opcodes share single pipe (sec 12)** — HIGH: well-established, confirmed with mixing tests.
- **Uniform datapath UFFMA/UFADD NOT emitted by compiler (sec 28, line 2149)** — HIGH: CUDA 13.2 ptxas observed behavior, multiple tests.
- **FMA latency 4 cy / MUFU 24 cy / DFMA 63-92 cy / SHFL 24-26 cy (sec 15, sec ~17.4)** — MEDIUM: numbers drift slightly between sections (FMA: 4.03, 4.1, 4.5; DFMA: 63.9, 92, 125); the 4 cy FMA is the consensus.
- **DFMA zero-pipelined, linear with chain count (line 8120-8128)** — HIGH: very clean 1→8 chain sweep showing exact N× scaling.
- **148 SMs at 2032 MHz (or 1920 MHz)** — CONTRADICTION: catalog wavers between these. See #15.

### Sync / Fence Costs

- **fence.sc.cta = 14-29 cy, fence.sc.gpu = 270-307 cy, fence.sc.sys = 2880-5080 cy (sec 30.G)** — HIGH: extensively remeasured, with 36-point matrices, cross-checked between coalesced/uncoalesced.
- **"8-channel fabric limit" for membar.sys (sec 30.G)** — MEDIUM: clean 5K → 10K step at 9 warps/SM is suggestive; multiple measurements converge. But the "channel banking" mechanism is speculation.
- **Per-SM fence cost is local (mixed-load test)** — MEDIUM: the asymmetric heavy/light-SM test is interesting but only done once; the "150× variation" conclusion deserves re-investigation.
- **membar.gl NOT 8-channel-limited (sec 30.G)** — MEDIUM: consistent measurements support this, but again single-test-suite.
- **sc vs acq_rel semantically-backwards cost (line 3203)** — MEDIUM: interesting but mixed results (sometimes identical, sometimes 17-37% different); catalog itself notes coalescing mattered.

### Atomic Operations

- **ATOMS local shmem 24-34 cy / 1.5 TB/s u32 coalesced peak** — HIGH: multiple audited tests, ncu-cross-checked at the end.
- **ATOMG global 34 cy coalesced peak / 1.5 TB/s u32 (sec many)** — HIGH: ncu-verified.
- **CAS half-rate vs other atomics (sec 14, sec 30.B3)** — HIGH: both success and fail paths tested, same 2× slowdown.
- **Hotspot anomalies: N=2 addresses WORSE than N=1 (sec 18, 30.B2)** — MEDIUM: clear step at N=2 but mechanism is speculation (the catalog says "needs more investigation").
- **REMOTE NVLink atomic 2-3K cy (multi-GPU section)** — HIGH: bimodal distribution cleanly matches near/far L2 partition model.
- **Atomic with `.release`/`.acq_rel` costs 15-31× relaxed (multiple places)** — MEDIUM-HIGH: the numbers agree (~780 cy for release vs 34 cy relaxed) but context varies.
- **Atomic shared FP16/BF16 emulated 45× slower (sec 17.4 corrections)** — HIGH: reasonable given emulation via BSSY + CAS loop.

### Launch / Host API

- **Kernel launch ≈ 2.05 µs (lots of places)** — HIGH: consistent across configurations.
- **cudaMalloc 20ms, cudaMallocAsync 0.4 µs, 50,000× faster (sec ~17.12)** — HIGH-ish but numbers vary (160×, 64,500×). Reasonable magnitude.
- **cudaSetDevice cold 2116 ms (sec Cold Start)** — HIGH: single measurement but catalog emphasizes this strongly; seems realistic.
- **NVRTC compile 6 ms warm (sec ~17.6)** — MEDIUM.
- **CUDA graphs 2.8× speedup, 1.4 µs/launch amortized (sec graphs)** — HIGH: consistent.
- **PDL 1.9 µs save/kernel asymptotic (sec PDL)** — MEDIUM: Style A kernels only; Style B (unconditional writes) shows -3 µs slowdown, catalog acknowledges this explicitly.

### LLM / GEMM Claims

- **Llama-70B BF16 decode 40 tok/s (sec LLM)** — MEDIUM: end-to-end measurement plausible, cross-check predicts 42 tok/s (96% match).
- **Llama-70B FP8 decode 71 tok/s, 1.73× over BF16 (sec LLM)** — MEDIUM: plausible if cublasLt FP8 path was functional; but separate section explicitly says "FP8 not available via standard cuBLAS on B300" (line 17885). CONTRADICTION — see #7.
- **BF16 batch 1-64 same 22 µs latency "free up to 64" (sec BF16 Batch Scaling)** — MEDIUM: compelling claim but M=2-3 pathological case below (~1/5 throughput of M=1) undermines the simple story.
- **M=2-3 pathological (sec cuBLAS)** — HIGH: measurement is reproducible, shows real cuBLAS kernel selection artifact.
- **4097 alignment cliff 30× (sec GEMM alignment)** — HIGH-MEDIUM: clean single measurement, need to verify this is real and not a cuBLAS version artifact.
- **Llama-70B per-layer GEMM ms (sec DEFINITIVE 80-layer)** — HIGH: averaged over 80 layers, plausible.
- **MFU 78.8% at batch=2048 (sec MFU)** — MEDIUM: the "high end" MFU is plausible; the low-batch numbers are also consistent with decode theory.

### Architectural Probes

- **148 SMs / 2032 MHz / 126.5 MB L2 / 228 KB smem (sec Hardware)** — HIGH: cudaDeviceProp-authoritative.
- **Reserved 1 KiB shmem (sec Driver-Reserved)** — HIGH: exhaustive, compile-fail tests confirm the hard limit, verified starting offset via PTX probe.
- **"Steal reserved" trick working (sec Steal Reserved)** — HIGH: 4736 blocks tested, 0 corruption. Solid.
- **Cluster max 8 (portable) / 16 (non-portable) (sec Cluster Size Limit)** — HIGH: ptxas explicit rejection.
- **B300 = 10 GPCs (9×16 SMs + 1×4) (sec B300 Physical Architecture)** — HIGH: direct probe via %smid.
- **8 GPC boot groups (sec SM clock synchronization)** — LOW-MEDIUM: one off-catalog measurement, interesting but untested claim.
- **128 concurrent kernel dispatch slots (sec Two independent limits)** — HIGH per git log 579e4f0; AUDIT_NOTES.md says "not re-verified this session" but cited across catalog.
- **L2 partition = 2, 63 MB each (sec L2 Partition Architecture)** — MEDIUM: indirect measurement via "each CTA sees 64 MB" is suggestive.
- **Smem bank conflicts linear with way count (sec 17.8)** — HIGH: theory matches.

### DSMEM / TMA / Cluster

- **DSMEM load 0.8% slower than local smem (sec 30.H)** — MEDIUM: correctness verified via neighbor-read test. But this contradicts earlier sections where DSMEM reads 4.7× slower (DSMEM v2 from AUDIT_NOTES.md).
- **TMA cp.async.bulk 48 cy/inst issue rate (sec 30.4b3)** — HIGH: size-independent floor plausible.
- **TMA per-SM peak 241 GB/s (sec 30.4b)** — MEDIUM: several tests converge at ~241-248 GB/s for 1-CTA unloaded peak.
- **TMA chip peak 27.7 TB/s (sec 30.4c)** — MEDIUM: catalog explicitly retracts earlier "30.5 TB/s" (L2-reuse artifact) and "20.4 TB/s" (under-filled engine).
- **TMA multicast works on sm_103a at 211 GB/s for 8-CTA cluster (sec 30.4c?)** — MEDIUM: one successful experiment.

### Findings that Deserve WRONG/RETRACTED Label

- **"7.6 PB/s SHMEM"** — WRONG: DCE artifact (AUDIT_NOTES.md acknowledged).
- **"295 TB/s TMEM"** — WRONG: DCE-inflated (catalog self-retracts).
- **"830 TB/s TMEM"** — WRONG: DCE-inflated.
- **"286 GB/s NVLink P2P kernel"** — WRONG: thread-limited; corrected to 755 GB/s.
- **"11,412 GB/s L2 at stride=4"** — WRONG: bad sector-accounting multiplier.
- **"L1 size ~32 KB"** — WRONG: multiple later tests give 128-192 KB effective.
- **"streams_explore.cu 256 streams parallel"** — WRONG: broken event timing (AUDIT_NOTES.md).
- **"Tensor cores work: 514 TFLOPS"** — WRONG as a "peak tensor" headline.
- **"6357 TFLOPS FP8"** — WRONG: DCE-folded (catalog self-retracts at line 1101).
- **"FFMA latency = 4 cy" (clean) vs "23 cy" (self-dep) vs "8.46" (self-op) vs "4.03" (dep)** — confusing; catalog notes self-op chains are 2× inflated at line 1948.
- **"mma.sync m16n8k32.f32.e4m3.e4m3.f32 = 2336-2400 TFLOPS"** — WRONG: F2FP + HMMA emulated, half-retracted (line 1101).

### Operational / Metadata

- **NVTX ~0 ns (AUDIT_NOTES.md)** — LOW: below measurement noise floor; "negligible" conclusion is fine.
- **Clock 2032 vs 1920 MHz ambiguity** — See contradictions #15.

---

## 2. Contradictions Found

Concrete, numbered contradictions between catalog sections. "Line X" refers to the line in B300_PIPE_CATALOG.md unless noted.

### 1. L2 Peak Bandwidth: 10, 14, 19, 22-26, 30.3 TB/s

Multiple numbers in different sections:
- Section 0 / line 43: **22-26 TB/s** plateau
- Line 1483: **22.0-22.2 TB/s** at 128 MB
- Line 1478: **30.3 TB/s** at 1 MB (.cg) / **36.1 TB/s** (.ca)
- Line 1959-1961: **~19.7 TB/s** at 16-64 MB plateau
- Line 1492: "Earlier 10.2 TB/s was WRONG" — but 10.2 was reported 20 commits earlier
- Line 8079 (FP32 roofline section): **"Memory ceiling ~3.5 TB/s"** using `4736 × 512 config`

Root cause: L2 numbers depend on working set, occupancy, cache-hint, and whether L1 is "helping". There's no single canonical "L2 peak" — the catalog needs a careful summary.

### 2. Smem Peak: 17 vs 20 vs 35-37 vs 38.5 TB/s

- AUDIT_NOTES.md: **19.85 TB/s** (52% of 38.5 theoretical, via shmem_peak.cu)
- Section 0 / line 41: **35.6 TB/s** (98% of 36.4 theoretical, via ld.volatile)
- Line 1182: **35.6 TB/s** (same)
- Line 5220: "Smem v4 read 37.7 TB/s (98% of theoretical peak)"
- Line 1188: **17 TB/s** labeled "smem" then corrected to 35 TB/s

These can be reconciled if you read carefully: 19.85 TB/s was loop-overhead dominated, 35-37 TB/s is the true peak. But the catalog contains both numbers as though they were separate entries.

### 3. DSMEM vs Local SMEM: 0.8% vs 4.7× Slower

- Line 2856-2869 (sec 30.H): DSMEM `ld.shared::cluster` is **23.26 cy, vs local 23.07 cy = 0.8% slower**
- AUDIT_NOTES.md (dsmem_v2.cu): DSMEM is **4.7× slower than local smem**, 1000 vs 4500 GB/s

These numbers are radically different. Sec 30.H looks more rigorous (correctness-verified with neighbor read) so that's probably the right number. The 4.7× gap from dsmem_v2 is likely a methodology artifact, but the catalog doesn't reconcile them.

### 4. FFMA Latency: 4, 4.03, 4.5, 8.46, 23 cy

Different numbers across sections:
- Sec 2 / line 103: "FFMA 4 cy latency, 2.0 cy throughput (2 chains)"
- Line 1051: "FFMA (`fma.rn.f32 %0,%0,%0,%0`) 4.14 cy"
- Line 1734: "FFMA latency in dep chain: 4.53 cycles"
- Line 1947: "FFMA reference (self-op chain): 8.46 cy — 2× inflated from register read-port pressure"
- Line 19575: "**23 cycles** ... to hide need ~8 warps per partition"

The last one is a different metric ("~23 cycles for 1000 FFMA with 1 warp" = throughput-dep latency, not single-inst latency). Catalog doesn't distinguish clearly. **The 23 cy number is dangerous to cite** — it's not the FFMA latency, it's the warp-issue-limited throughput of a self-dep chain at low occupancy.

### 5. NVTX Cost: 0 ns vs 0.2-0.66 ns

- Sec ~17.X: "nvtxMark ~0 ns/call; rangePush+Pop 0.66 ns/pair"
- AUDIT_NOTES.md: "0 ns is impossible — measurement below resolution"

The AUDIT_NOTES correction is correct. The 0.66 ns may itself be noise. Treat as "negligible, <10 ns."

### 6. DRAM Write Peak: 3.4, 7.0, 7.09, 7.37, 8.5 TB/s

- Section 0 / line 46: "DRAM write = 7.09 TB/s"
- Section 0 key rule #12: "DRAM write is half of read BW (3.4 vs 7.3)"
- Line 1461-1466: "st.global.v4 = 3.42 TB/s, st.global.v8 needed for 7.0 TB/s"
- Line 8656: cudaMemset **7.37 TB/s**

Key design rule #12 is from an older measurement and is **contradicted** by the later corrected numbers (3.4 was v4-only, not peak). Rule should be removed or updated — it's currently misleading.

### 7. FP8 via cuBLAS: Works and Doesn't Work

- Section 0 cheat-sheet: FP8 tensor peak 4651 TFLOPS
- Line 17883-17892: "FP8 cuBLAS: Not Available via Standard API on B300" — 12 cublasLt configurations all CUBLAS_STATUS_NOT_SUPPORTED
- Line 10042-10064: "cuBLAS FP8 E4M3 GEMM ... 4474 TFLOPS measured!"
- Line 17506-17514: FP8 E4M3 input measured at "4474 TFLOPS"

The catalog claims FP8 cublasLt works in some sections and not in others. This is a major internal contradiction — either cublasLt FP8 is available (measurements in Section "cuBLAS FP8 E4M3 GEMM") or it isn't (measurements in "FP8 cuBLAS: Not Available"). Likely explanation: FP8 cublasLt works via cublasLtMatmul with specific descriptors, but fails with cublasGemmEx. Catalog doesn't clarify.

### 8. Concurrent Kernel Limits: "1" vs "128"

- cudaDeviceProp at line 8708 and line 18875: `concurrentKernels = 1`
- Line 18875 comment: "misleading; true HW concurrent slot count = 128"
- Section "Concurrent Kernel Execution" (line 18132-18195): "128 hardware kernel dispatch slots"

The `cudaDeviceProp.concurrentKernels = 1` is a property flag about feature support (boolean "do you support concurrent kernels at all?" = 1), not a count. The catalog comments correctly but the prop dump could confuse readers.

### 9. L1 Cache Size: 32 KB, 64 KB, 128 KB, 192 KB, 256 KB

- AUDIT_NOTES.md: "~32 KB (first guess)" → "up to 128 KB (corrected)" → "192 KB (re-measured)"
- Line 4996: "L1 data cache effective capacity = 192 KB per SM"
- Line 17293: "L1 data cache ~28 MB (192 KB × 148), Per-SM 192 KB"
- Line 1498-1507: "carveout=100: L1 ≈ 28 KB; carveout=0: L1 ≈ 256 KB"

Key unreconciled: the "192 KB" finding uses carveout=0 (max L1), and the 256KB unified pool is correct. But the "effective capacity" varies with carveout setting, which most of the catalog doesn't state. **Any "L1 = N KB" claim needs a carveout context.**

### 10. HBM3E Effective vs Raw: 4.2, 5.2, 6.1, 7.0-7.4, 8.17 TB/s

- Line 4329: **6.09 / 6.12 TB/s** (HBM peak via ncu, 75% of 8.17 theoretical)
- Line 4328: Theoretical HBM3e = "3996 × 2 × 8192 / 8 = **8.17 TB/s**"
- Section 0 / line 46: **7.18 TB/s** read, 7.09 TB/s write
- Line 8531-8534: "HBM3e **5.16 TB/s** cold reads" — section labeled WRONG
- Line 14944: "DRAM read = 4.2 TB/s"

The canonical HBM3e peak is 7.4 TB/s (90% of 8.17 theoretical). The 4.2 TB/s is an artifact of low occupancy; 5.16 was a WS error; 6.1 from ncu was a specific config. None of these are labeled clearly in the cheat sheet sections.

### 11. PDL Impact on Memory-Writing Kernels

- Line 18479-18483: "Style A conditional write: +2.09 µs save" / "Style B unconditional write: -3.23 µs slowdown"
- But Section 0 rule #7 / line 85: "PDL saves 1.9 µs/kernel asymptotic"
- AUDIT_NOTES.md PDL-saving: "For pure-compute Style A: HIGH; For real-world kernels with memory writes: VARIABLE"

This contradiction is acknowledged by the catalog but cheat-sheet summaries don't always caveat.

### 12. Warps to Saturate HBM: 8 vs 16 vs 32

- Line 4806-4815: "8 warps/SM = 60% HBM peak, 16 warps/SM = 80%+"
- Line 17312-17314: "Min warps/SM for 86% BW: 32"
- Line 17747-17749: "Min warps for BW: 8/SM (60% HBM)"

Different number (saturation) numbers depending on measurement — all plausible but the cheat-sheet should unify.

### 13. L2 Latency: 300-400 vs 144 vs 91 cy

- Line 4949: "L2 latency 295 cy"
- Line 4994-4999: "L1=39 cy, L2=301 cy, DRAM=789 cy"
- Line 11053-11062: L2 stride sweep shows "64B stride cy=304" but at 32B "cy=56 (L1 hit)" — stride drives latency
- Line 4978: "L2 hit 28-91 cy, varies with WS size"

Catalog resolves this as "L2 latency depends on WS size" (larger WS → more cross-XBAR traffic). Not a true contradiction but presentation is muddled.

### 14. FFMA Peak TFLOPS: 52, 58, 60.5, 65.8, 68.7, 71.8, 72.3, 72.7

Many numbers at different ILP/occupancy:
- AUDIT_NOTES.md: "52 TFLOPS (ILP=8, 8 warps/SM, 68%)"
- AUDIT_NOTES.md update: "**58.6 TFLOPS at ILP=16, 64 warps/SM (76%)**"
- Line 17522: "FP32 FFMA = 65.8 TFLOPS"
- Line 4343: "FFMA 68.7 TFLOPS via ncu"
- Line 4355-4357: "theoretical 72.7 TFLOPS"
- Line 30 / sec 0: "**71.8 TFLOPS = 98.8% SOL**"
- Line 31: "FFMA2 72.3 TFLOPS = 99.4%"
- Line 17914: "FP32 FMA @ 386 W = 38 TFLOPS" — this is the odd one out

The 38 TFLOPS number appears only in one power-related section and is likely an older under-saturated measurement. It contradicts the ~72 TFLOPS numbers elsewhere. **38 vs 72 is a 2× contradiction that needs reconciliation**.

### 15. Clock Frequency: 1800, 1920, 2032 MHz

Extensively contradictory:
- Line 12: "2032 MHz natural boost" (top of catalog)
- Section 31 methodology line: "`nvidia-smi` confirms **1920 MHz**"
- Line 8260-8266: "Actual: 1920 MHz; nvidia-smi -lgc 2032 succeeds but SM still runs at 1920 MHz"
- Line 17727: "**Boost Clock: 2032 MHz**"
- Line 17747: "All catalog measurements are at 1920 MHz" but also "Throughput numbers use 1.92 GHz (clock-locked)"
- Line 5416: "Locked to max (2032 requested) = 1920 MHz"
- Line 8795: Reference card says "**2032 MHz**"

This is a real mess. The prior catalog at the top says 2032 MHz. A later investigation concludes 1920 MHz is the real sustained clock. Many sections quote TFLOPS at 1.92 GHz (implying 1920). But the Quick Reference Card / cheat sheet still says 2032 MHz. **All tables need a clock annotation**, and this is a subtle 5.8% systematic error in the headline TFLOPS numbers if the clock is actually 1920.

Project memory (from user MEMORY.md) says "2032 MHz" for "TRUE perf at 2032 MHz" — so the catalog may now be running at 2032 under explicit clock lock. The claim "HW physically caps at 1920" may be a specific test condition.

### 16. HMMA Latency: 8.18 cy vs 14.5 cy vs 20 cy

- Line 1107: "HMMA FP16 m16n8k16 latency: **20.8 cy** from HMMA-issue to accumulator-ready"
- Line 4346: "HMMA 8.13 cy at 8 ILP throughput"
- Line 7014: "mma.sync m16n8k16.f32.f16.f16.f32: **14.5 cy**, 80 TFLOPS = 29× slower than tcgen05"

These correspond to different things (issue rate, latency with dep, throughput at ILP), and the 14.5 cy in "slow mma.sync" is entirely different regime. Catalog doesn't clarify which is which in headline tables.

### 17. Fusion Ratios — 2:1 vs 1:1 FFMA2:IADD3

- Line 6457: "Sweet spot: ~1 IADD per FFMA2"
- Line 6489-6493: "critical correction: sweet spot is **2 FMA2 per 1 IADD3**, NOT 1:1"
- Line 6540-6546: "8 FFMA2 + 4 (IADD3 | SHR | I2F)" — that's 2:1 again

Catalog auto-corrects itself but the 1:1 claim still appears in an earlier header. One of them is a leftover.

### 18. "Smem peak 38.5 TB/s" vs "17 TB/s"

- Section 0 / line 50 note: "Smem read peak is **~36 TB/s** chip at 128 B/clk/SM"
- Line 1187: "smem (17 TB/s) ≈ L2 (10 TB/s) × 1.7"

Two different "smem" values cited next to each other. The 17 TB/s is labeled as the old wrong number. But AUDIT_NOTES.md's 19.85 TB/s is closer to 17 than 36.

### 19. Membar.sys Cost Scaling with SMs

- Line 2957: "Full chip (148 CTAs × 32 thr) = **5046 cy**"
- Line 3066: "148 SMs: 5065-5089 cy" (various writes)
- Line 3086: "148SMs/0wr: 271 cy membar.gl, 5079 cy membar.sys" (stays ~5K)
- Line 3792: "Full-chip (not single-warp): 8,843 cy to 5075 cy depending"

Mild inconsistency across sections; catalog explicitly revises "5K vs 10K" at line 3877-3879 attributing to warps-per-SM. OK but confusing.

### 20. Decode Per-Layer Time

Llama-70B BF16 decode per-layer:
- Line 5539: "0.314 ms (b=1 TN layout)"
- Line 4921: 0.303 ms / layer
- Line 15177: "345 µs per layer" (24.2 ms / 80 layers / BF16)
- Line 17866: **263.8 µs** per layer breakdown

These differ by 20-30%. Some exclude attention/norm, some include. Specify clearly.

---

## 3. Methodology Problems

Specific methodological flaws in measurements. File paths assume `tests/` directory.

### 3.1 DCE-suspect Tests (likely inflated)

- **`tests/shmem_test.cu`** (DCE'd) — produced 7.6 PB/s. Invalid; retracted in AUDIT_NOTES.md but still referenced in some catalog sections.
- **`tests/l2_cacheline.cu`** — the "32B sector × N multiplier" accounting is wrong for coalesced loads, retracted in AUDIT_NOTES.md.
- **`tests/dsmem_proper.cu`** and **`tests/dsmem_cycles.cu`** (AUDIT_NOTES.md) — both DCE-corrupted; compiler folded the init loop. Give unreliable DSMEM peak numbers.
- **Early TMEM tests giving 295/830 TB/s** (catalog section 16) — explicit DCE.
- **Early FP8 tests giving 6357 TFLOPS** (line 1101) — explicit DCE.
- **AUDIT_NOTES.md has a whole section "DSMEM BANDWIDTH — DCE NOT VERIFIED"** where the methodology issues are documented but the numbers persist in some catalog sections.

### 3.2 Single-Warp Tests Reported as Chip-Wide

- **"Roofline crossover AI = 8" (sec Practical Roofline)**: measured single-warp, doesn't extrapolate cleanly to multi-warp. The section's own "theoretical crossover 5.4 FLOP/byte" disagrees with the empirical "8-16" and catalog notes "serial dep chain limiting compute."
- **Many "latency" numbers are self-op chains (AUDIT_NOTES.md / line 1947)**: MUFU/FFMA self-op chains are 2× inflated from register read-port pressure. Ratios are valid but absolute latencies aren't.
- **pipe_fma utilization via ncu** (many sections) — pipe-cycle-active metric is misleading when comparing scalar vs packed FFMA. The catalog flags this at line 6362-6364 but many derived numbers don't account for it.

### 3.3 Launch-Overhead-Dominated Tests

Kernels with <1 ms runtime are dominated by the ~2 µs kernel launch floor. Flagged examples:
- **"nvtxMark ~0 ns"** — below std::chrono resolution (AUDIT_NOTES.md).
- **Small kernel tests at 4.4 µs "minimum kernel"** (sec 18) — only 4.4 µs includes dispatch, so "per-op" overhead claims below this are noise.
- **RMSNorm at 10 µs/call, of which ~10 µs is launch** (line 15116-15119) — acknowledged but cited in several "per-layer breakdown" tables without footnote.

### 3.4 Test Not Saturated at Scale

- **AUDIT_NOTES.md "FP32 52 TFLOPS is under-saturated"** — corrected to 58.6 TFLOPS with more warps.
- **Line 4822 "L2 knee at 70 MB with low TLP"** — catalog correctly identifies this is a TLP-hiding artifact, not a real knee, but some tables still cite 70 MB as the L2 capacity inflection point.
- **Small-M cuBLAS tests** — M=1, 2, 3 behaviors are cuBLAS-library-artifacts, not HW limits. The "pathological M=2, M=3" finding is right to cite but catalog conflates with architectural analysis.

### 3.5 Shape-Dependent Issues

- **`tests/bench_tma_throughput.cu` / mistake #1** (line 2275-2277) — requested smem exceeds B300's 200 KB cap; silently failed, produced "680 GB/s" artifact. Catalog documents this but similar structural issues may exist in other `tests/bench_*.cu` files.
- **`tests/bench_tma_audit.cu` / mistake #2** (line 2277-2278) — used SMEM_STRIDE=0, all TMAs to same cache line; inflated to 196 GB/s.
- **`tests/bench_tma_real.cu` / mistake #3** — single-thread serialization, doesn't measure engine throughput.

### 3.6 Compiler Transformation Not Accounted For

- **Scalar HFMA → HFMA2 auto-packing** (line 1143): compiler silently packs adjacent independent `__half` chains into HFMA2. So "scalar HFMA throughput" = packed HFMA2 throughput, but measuring one as if it's the other gives wrong pipe attribution.
- **`1.000001f` round-to-1.0 in BF16/FP16** (line 1144): the compiler folds `v * 1 + v → 2v` if the multiplier rounds to 1 in low precision. Catalog notes the fix (use 1.5f) but may be an issue in any kernel using 1.000001 patterns.
- **FSETP + FSEL inline predication** (sec 14): 2-way branches don't produce divergent code; they become `selp`. Many "branch cost" measurements actually measure predication cost.
- **LEA emission for shift+add** (line 980): `mad.lo.u32 with power-of-2` emits `LEA + IMAD`, not IMAD. Changes pipe attribution.

### 3.7 `self-op` chain measurements

Many latency numbers use kernels like `fma.rn.f32 %0, %0, %0, 0` (all operands are the same register). The catalog notes at line 1948 that this inflates latency 2× due to register read-port pressure. But the inflated numbers propagate through many headlines without correction.

### 3.8 PDL Memory-Order Confound

- Sec PDL (line 18475-18483): **`griddepcontrol.wait` is a memory fence** — adds 3 µs overhead when kernels have pending writes.
- This means **all "PDL overhead" microbenchmarks using unconditional writes conflate control-sync with memory-sync costs**. The catalog documents this but the "PDL saves 1.9 µs" headlines don't caveat.

### 3.9 Occupancy-Dependent Extrapolation

- **"Per-SM BW 241 GB/s → chip 35 TB/s"** (sec 30.4b) — assumes perfect chip scaling which isn't always true.
- **"Atomic peak 137 Gatomic/s at 148 × 1024"** (line 3495) — extrapolation from per-SM measurement; catalog later finds "379 Gatomic/s at stride=4" contradicting this number.

### 3.10 No clock locking for many tests

Some measurements taken before/without explicit `nvidia-smi -lgc 2032`. The reference memory says "Clock-lock was 2.35× bottleneck" — meaning default clocks can be 2× lower. Many catalog numbers may be at reduced clock unless explicitly stated.

---

## 4. Missing Caveats

Numbers presented as facts that deserve caveats.

### 4.1 Tensor Core Peak — "which path"

Section 0 presents "FP16/BF16 tensor 577 TFLOPS" as the headline number. But the catalog also measures **2325 TFLOPS via tcgen05.mma** — 4× higher. Section 0 should lead with the tcgen05 number and relegate mma.sync to a footnote. As presented, readers will cite the legacy-path number as "B300 FP16 peak."

Similar issue with the entire FP4/FP6/FP8 section 0: "276 TFLOPS FP8 via mma.sync emulated" is a bizarre headline to present alongside 4.65 PFLOPS actual FP8 via tcgen05.

### 4.2 FFMA peak "99% SOL" needs clock annotation

The "72.3 TFLOPS = 99.4% of 72.7" calculation uses 1.92 GHz. If the chip is actually at 2.032 GHz as headline says, the theoretical is 77 TFLOPS and measured is ~94% instead of 99%. Every TFLOPS-vs-theoretical % needs the clock assumption stated.

### 4.3 L1 size depends on carveout

"L1 = 192 KB" is carveout-dependent. Need to state the carveout value for any L1 size claim. The catalog does this in places but cheat-sheet tables don't.

### 4.4 "N tokens/sec" claims assume specific config

Llama decode tok/s numbers assume cuBLAS warmup, specific CUDA version, clock lock, etc. In other sessions these will be different.

### 4.5 DCE warnings should be more prominent

Many tests document "DCE-suspect" inline, but summary tables don't flag. Example: "TMEM peak 60 TB/s" — the catalog has "stricter DCE defeat" applied, but the 60 TB/s might itself have subtler DCE that wasn't caught.

### 4.6 Clock Ambiguity

All throughput tables need to specify "at 1920 MHz" or "at 2032 MHz". The Quick Reference Card silently assumes 2032 MHz while the measurements use 1920 MHz.

### 4.7 "Free" qualifier

Many sections claim operations are "free" (e.g., "IADD free alongside FFMA2"). These are only free at specific ratios (2:1 for IADD3, not 1:1). Headlines don't always caveat.

### 4.8 Multi-GPU assumes NV18

NVLink numbers assume NV18 (18 links) — true for 2x B300 SXM6 AC but other configs differ. Catalog doesn't flag this dependency.

### 4.9 "Peak HBM 7.0 TB/s" assumes specific access pattern

The 7.4 TB/s peak needs ≥2 CTAs/SM, wide loads (v8), 1-4 GB WS. Small WS, narrow loads, or low occupancy all reduce this significantly (3-5 TB/s). Cheat sheets omit these conditions.

### 4.10 `mma.sync` vs `tcgen05.mma` conflation in FP8

Headline numbers in different sections attribute FP8 peak to different paths. Without context, a reader can't tell if "4.9 PFLOPS FP8" came from native tensor or emulated.

### 4.11 Fence cost matrix assumes coalesced stores

The "fence cost by warps/W" matrices in Sec 30.G revised multiple times when it was realized uncoalesced stores were 32× worse. Any cited fence cost should state "coalesced STG" or "STG scatter".

### 4.12 "MFU 78% at batch=2048" assumes BF16 cuBLAS

MFU is a single-GPU single-step number, doesn't include all-reduce for TP, doesn't include CPU overhead. Real end-to-end MFU is lower.

### 4.13 Atomic Peak depends on stride, not "coalesced"

Line 4245 corrects "137 Gatom/s" to "372 Gatom/s at stride=4B". Many sections cite 137 without qualifying.

### 4.14 Register Spill Threshold Is Kernel-Specific

"255 regs = 50% slowdown" is for a pure FFMA chain. Different kernel patterns will spill at different thresholds.

### 4.15 LDG.E.STRONG.SYS is Extra Slow

Many "write BW" numbers use `st.volatile.global` which emits `STG.E.STRONG.SYS`, significantly slower than `STG.E` (non-strong). Real code usually doesn't use .volatile; numbers underestimate practical throughput.

---

## 5. Top 20 Things Worth Re-Investigating

Prioritized by impact × likelihood-of-error. Each specifies: (1) the file and metric, (2) the suspected issue, (3) what better methodology to use.

### 1. CLOCK FREQUENCY AMBIGUITY (highest priority)

- **What**: Does B300 actually sustain 2032 MHz or 1920 MHz under FFMA load?
- **Suspected issue**: Catalog has both numbers with conflicting attribution. TFLOPS reported "at 2032 MHz" may be 6% inflated.
- **Methodology**: Take a single measurement with `nvidia-smi -lgc 2032` explicit, measure `(clock64_end - clock64_start) / (globaltimer_end - globaltimer_start)` over a sustained 1s workload. Repeat at `-lgc 1920`. Publish the actual sustained clock for the 71.8 TFLOPS test.

### 2. cuBLAS FP8 "Not Available" vs "Measured 4474 TFLOPS"

- **What**: Does cublasLt FP8 work on sm_103a for CUDA 13.2?
- **Suspected issue**: Sec "FP8 cuBLAS Not Available" says all 12 configurations failed. But earlier sections show measured FP8 GEMM at 4474 TFLOPS.
- **Methodology**: Re-run cublasLtMatmul with explicit E4M3 descriptors, check `cublasStatus_t` return, dump kernel names with ncu to see which path runs. Cross-reference CUTLASS 3.x Blackwell FP8 examples.

### 3. mma.sync FP16 Peak: 544 vs 577 TFLOPS

- **What**: Is the peak mma.sync m16n8k16 BF16 really 577 TFLOPS (section 0) or 544 TFLOPS (AUDIT_NOTES.md)?
- **Suspected issue**: Different audit iterations gave different peaks. Which is the current best measurement?
- **Methodology**: Single clean test with SASS inst-count verified, ncu pipe_tensor utilization >99%, vary ILP and occupancy, report peak with reproducible seed and config.

### 4. DSMEM Actual Overhead — 0.8% or 5×?

- **What**: Is DSMEM truly free relative to local smem, or is there a 4.7× slowdown?
- **Suspected issue**: Sec 30.H gives 0.8% overhead via one kernel; AUDIT_NOTES.md / dsmem_v2.cu gave 4.7× via different kernel.
- **Methodology**: Same test for both — measure same smem workload with `ld.shared` vs `ld.shared::cluster`, both at low and high in-flight request depth. Check with ncu `lts__*` metrics for actual cross-SM traffic.

### 5. FP32 FMA at 38 TFLOPS vs 72 TFLOPS (sec Power)

- **What**: Why does line 17914 claim "FP32 FMA 38 TFLOPS at 386 W" when section 0 says 71.8 TFLOPS at 390 W (line 17092)?
- **Suspected issue**: Likely an old/broken measurement. 2× TFLOPS discrepancy is not small.
- **Methodology**: Re-run the power-efficiency test with the audited FFMA peak kernel (`tests/bench_ffma_peak.cu` from the Scalar FFMA section). Verify 70+ TFLOPS at similar power.

### 6. L2 Peak "22-26 TB/s" vs "30 TB/s" vs "17 TB/s"

- **What**: What is B300's actual L2 read peak bandwidth?
- **Suspected issue**: Numbers span 17-36 TB/s across sections. No single clean "L2 peak" number.
- **Methodology**: Full occupancy sweep (296 CTAs × 1024 threads), working set 32 MB, use .cg loads to avoid L1 caching, unique-per-CTA addresses to avoid broadcast artifacts. Run with and without ncu to check `lts__t_sectors_op_read` for actual L2 traffic.

### 7. HMMA Peak via mma.sync: Is tcgen05 Really 4× Faster?

- **What**: "tcgen05.mma is 29× faster than mma.sync m16n8k16" (line 7014-7019) — this is a MAJOR perf gap.
- **Suspected issue**: If really 29×, most existing Hopper/older kernels must be rewritten for B300. This deserves thorough verification.
- **Methodology**: Run mma.sync m16n8k16 with best ILP/occupancy, then tcgen05.mma kind::f16 M=128 N=256 at same SM count. Compare TFLOPS directly. Also test mma.sync m16n8k32 (double K) to see if the architecture-specific larger K shape helps.

### 8. "Per-SM membar.sys is local" — real or artifact?

- **What**: The finding that heavy SMs don't affect light SMs' fence cost. Line 3267-3271 shows LIGHT stays at 5K cy with 140 heavy SMs.
- **Suspected issue**: Counterintuitive; fabric should have some shared resource.
- **Methodology**: Instrument with per-SM clock64 timestamps during the test, look for per-SM variance during heavy load. Check with ncu `fbpa__*` metrics (L2 fabric busyness). A single well-designed test.

### 9. Atomic Peak 372 Gatomic/s at stride=4

- **What**: Line 4242 shows atomicAdd at stride=4 gives 372 Matomic/s vs 137 at stride=256. 2.7× speedup from coalescing.
- **Suspected issue**: Catalog has lots of "atomic peak" numbers; this 372 Gatom/s peak isn't cross-referenced with ncu or alternative methodologies.
- **Methodology**: Run the stride=4 test again with ncu `l1tex__t_bytes_pipe_lsu_mem_global_op_atom` to verify actual L1/L2 atomic unit traffic matches the semantic op rate.

### 10. TMEM Read 60 TB/s "with stricter DCE defeat"

- **What**: Line 1265-1293 shows TMEM read peak ~60 TB/s, downgraded from earlier 295/830 TB/s due to DCE.
- **Suspected issue**: The "DCE defeat" is via xor-accumulator + conditional write. Possible that compiler still partial-folds. 60 TB/s is suspiciously close to FP32 FFMA SOL — coincidence?
- **Methodology**: Add per-iter unique address inputs + unconditional write to out[blk] per iter. SASS-verify the tcgen05.ld count matches UNROLL. Compare against ldmatrix.x4 smem path with same methodology for sanity.

### 11. PDL "Style B unconditional write = -3 µs slowdown"

- **What**: Line 18474-18483 shows PDL COSTS 3 µs for kernels with writes.
- **Suspected issue**: Implies PDL is useless for most real kernels that write output. Many PDL-recommended-use cases may be wrong.
- **Methodology**: Test a realistic pipeline (GEMM → softmax → GEMM with actual output writes in both). Compare to the Style A "conditional write" microbenchmark to isolate the slowdown source.

### 12. 4097 Alignment Cliff "30× Slowdown"

- **What**: cuBLAS BF16 4097³ at 166 TFLOPS vs 4096³ at 1837 TFLOPS — 11× cliff.
- **Suspected issue**: Single measurement. May be cuBLAS library version sensitive.
- **Methodology**: Test cuBLAS 13.2 and cuBLAS 13.4 at M=4097 (and M+1 for other sizes: 4097, 4129, 4097, 8193). Check `cublasLtMatmulAlgoGetHeuristic` count of returned algos for the off-by-one shape vs clean shape.

### 13. "SM Boot Groups = 8 GPCs" — measurement robustness

- **What**: Line 4068-4079 claims B300 has 8 GPC boot-phase groups with ~6s spacing.
- **Suspected issue**: Only reported once. "Each group's counters started together" is interesting but hasn't been cross-verified.
- **Methodology**: Re-run the clock64 boot-phase probe 3 times on fresh CUDA contexts. Compare spacings. Check against `%smid` groupings in other tests.

### 14. Steal-Reserved Trick at high occupancy

- **What**: Line 9420-9430 shows "corruption = 0 across all tests" for stealing the reserved 1 KiB.
- **Suspected issue**: Tested only PDL, simple compute, and __syncthreads. Real user kernels often include mbarriers, TMA, etc., which WILL crash (line 9417).
- **Methodology**: Run the "steal" test with a realistic workload (TMA-based GEMM-ish kernel) and confirm it crashes as documented. Also test with `cuda::barrier` which may use reserved space.

### 15. "DRAM latency = 860 cy" consistency

- **What**: Scattered DRAM latency measurements: 789 cy, 813 cy, 824 cy, 860 cy.
- **Suspected issue**: Varies 10% across tests. Not necessarily bad, but no single "canonical" number.
- **Methodology**: Pointer-chase, 2 GB WS (>L2), TLB-warm via single pre-iter pass, measure 1000 iters with clock64. Report median + p99. Also test TLB-cold case (>10 GB WS) to find TLB-miss cost.

### 16. Tensor Sparse FP8 = 7.44 PFLOPS (74% of spec)

- **What**: "Same 7.44 PFLOPS regardless of sparsity metadata pattern" (line 6998).
- **Suspected issue**: The catalog itself suggests the metadata may be garbage, leading to slow path. Needs proper 2:4 structured sparse data.
- **Methodology**: Generate true 2:4 structured sparse A matrix, encode metadata correctly per CUTLASS `compact_sparse_A`, verify against CUTLASS fp8 sparse GEMM, measure TFLOPS.

### 17. Reserved 1 KiB is hardware-reserved, not compiler-reserved

- **What**: Does the reserved space actually get USED at runtime?
- **Suspected issue**: Line 9169 says "contents at kernel start: ALL ZEROS" — implies not used. But catalog claims PDL/cluster/etc. use it.
- **Methodology**: Run a kernel that does TMA + mbarrier + cluster.sync, dump the reserved 1 KiB before/after each operation. Confirms whether the space is actually touched.

### 18. "All atomic scopes free" and "release atomic 15× slower"

- **What**: Two sections disagree:
  - Line 8020-8027: "atom.shared scope .cta/.cluster/.gpu/.sys = all 34 cy (FREE)"
  - Line 7098-7105: "atom.release.gpu = 872 cy, 17× slower"
- **Suspected issue**: Scope vs ordering are two different things; catalog conflates in summary tables.
- **Methodology**: Re-test atomic with systematically varied (scope × ordering) — 4 × 4 matrix. Verify that "scope" is free only at release-relaxed, and "ordering" is the slow part. Clearly present.

### 19. `membar.sys` "8-channel fabric limit" — real?

- **What**: Line 2942-2953 shows discontinuity at 9 warps/SM.
- **Suspected issue**: Interesting but only one measurement. The "8 channels" interpretation is inferred, not directly observed.
- **Methodology**: Vary warps/SM from 1 to 32 with very fine granularity, run each 10 times, plot the distribution. Check for the discrete step. Also see if this depends on SM count.

### 20. Launch Latency "2.05 µs" — is this really invariant of config?

- **What**: Line 4648-4651 shows 1024×148 launch = 32×1 launch = 2.05 µs.
- **Suspected issue**: Should scale with grid size somewhat. Possibly the CPU-side cost is all driver-serialized and 2 µs is the floor.
- **Methodology**: Measure launch latency for 1, 10, 100, 1K, 10K, 100K blocks (holding threads constant). Check with and without `cudaLaunchKernelEx`. Check with and without pinned memory.

---

## 6. Architectural Insights Possibly Wrong

Architectural claims in the catalog that may be incorrect based on internal inconsistencies.

### 6.1 "148 SMs × 4 SMSPs × 2032 MHz → 77 TFLOPS theoretical peak"

This is stated multiple places as the FP32 scalar theoretical peak. **Possibly wrong**: if the sustained clock is 1920 MHz (as AUDIT_NOTES.md and several sections indicate), theoretical is 72.7 TFLOPS. The "~99% SOL" claim then becomes 98% or 101% depending on clock. Need to re-audit the theoretical calculation.

### 6.2 "B300 has 2 L2 partitions / 63 MB each"

Line 7246-7254 argues this based on single-CTA seeing 64 MB max. **Possibly wrong**: could also be a measurement artifact — at 64 MB one partition is full, wraparound causes thrashing, not strictly "2 × 63 MB". An SM could see beyond 64 MB if coordinated via L2 hash. Need ncu `lts__t_sectors` per-partition breakdown to confirm true partition count.

### 6.3 "DFMA has zero pipelining"

Line 8120-8128 shows 1→8 chain linear scaling. **Plausibly correct** but the claim "only 1 FP64 op can be in flight per partition" is a strong architectural claim. Need to test with different chain structures (e.g., 8 DFMAs with different register allocations, ILP across warps) before asserting zero pipelining.

### 6.4 "UFFMA / UFADD not emitted by compiler" — architectural vs implementation gap

Line 2149 asserts the SASS opcodes exist but CUDA 13.2 doesn't emit them. **Could be wrong**: more recent CUDA toolkits may emit these. Catalog should be dated. Future CUDA 13.3/14.0 may change this.

### 6.5 "FFMA dual-issue heavy+lite at 4.00 warp-inst/SM/cy"

This is the fundamental basis of the "~72 TFLOPS peak" claim. **Cross-check**: ncu pipe_fma reports 99% utilization for FFMA-only. The catalog's 71.8 TFLOPS = 98.8% matches. But the theoretical 77 TFLOPS assumes this dual-issue always works, which is only true for scalar FFMA (not packed ops). Section 2 clarifies this; section 0 muddles.

### 6.6 "L1 cache 192 KB effective"

Line 4996 asserts this via pointer-chase sweep with sharp transition at 256 KB. **Possibly too simple**: the transition is stride-dependent (sec 3.5 shows 32B vs 64B stride changes L2/L1 partition). Also carveout-dependent. The "192 KB" is correct at carveout=0 but misleading without qualification.

### 6.7 "8 GPC boot-phase groups"

Line 4073 claims 148 SMs cluster into 8 boot-phase groups. **Possibly wrong**: the "8 GPCs" count contradicts the "10 GPCs" count found elsewhere (line 7490-7505 = 9 full + 1 partial GPC). So either 8 boot-phase groups ≠ 10 GPCs (possible, some GPCs may boot together), or one count is wrong.

### 6.8 "FFMA latency = 4 cy"

Widely cited. **Plausibly correct** but sec 30.L's `LATENCY=4.07 cy, THROUGHPUT at 8-chain=2.68 cy` gives ratio 1.52 = expected pipe depth × 2-heavy-lite. The 4 cy is the minimum; in-practice-observed can be higher in high-occupancy kernels. The "2 FMA pipes each 2 cy" arithmetic makes the "4 cy / 2-chain saturation" clean but depends on dual-issue.

### 6.9 "B300 has 10 GPCs" or "8 GPCs"

- Line 7490-7505: **10 GPCs** (9 full × 16 SMs + 1 partial × 4 SMs = 148)
- Line 4073: **8 GPC boot-phase groups**
- Line 17727: Hardware section: "148 SMs × 4 SMSPs, 2032 MHz boost, 288 GB HBM3E, 12 HBM stacks, 18 NVLink v7" (no GPC count stated)
- "Cluster size 8 max portable" argues for 8 GPCs

**8 vs 10 is an unresolved architectural contradiction.** Needs direct probe: distribute 1 CTA per SM, print `(blockIdx.x, smid, ctaid_cluster)` for cluster launches of various sizes, infer the hierarchy.

### 6.10 "Store forwarding is free on Blackwell"

Line 5191 claims `st + ld` same address = 0 cy. **Possibly wrong** or trivially true due to DCE. A proper test needs cross-warp st/ld where the compiler can't elide, then measure whether L1 store buffer forwards vs hits memory.

### 6.11 "No clock throttling" under sustained compute

Section 17.15 claims "10 seconds at 962 W, no throttling." **Possibly wrong** for longer durations. Many GPUs throttle at minute-timescales not measured here.

### 6.12 "L2 latency varies with WS due to cross-XBAR"

Claimed 28-91 cy at small WS, 144-199 cy at large WS. **Possibly too simple**: L2 latency doesn't actually increase with WS size; rather, at larger WS, more accesses go to "far" partition. The average rises, but min stays constant. Catalog conflates these.

### 6.13 "Graph launch 0.56 µs per kernel amortized"

This is 3.7× faster than direct launch (2.05 µs). **Possibly wrong** for graphs with non-kernel nodes — catalog notes at line 11303 that memsets don't batch like kernels. "0.56 µs" only applies to identical kernel chains.

### 6.14 "Dynamic parallelism 6× slower than host launch"

Line 19024 claims 9.3 µs/child vs 1.5 µs host. **Possibly wrong**: comparison is unfair — host launch has 2 µs latency from host, while device launch is already on the device. Need to compare apples-to-apples (e.g., from a persistent kernel already on device, is device-launch faster or slower?).

### 6.15 "Cross-warp poll latency ~1335 cy"

Line 4188-4192 reports min 1335 cy for writer → reader cv-load observation. **May be optimistic**: single measurement under controlled conditions, likely excludes the real use-case overhead.

### 6.16 "tcgen05 100K iter cliff" (memory/compute governor)

Line 6599-6650 describes a 100K-iter throttle. **May be SW scheduling artifact**, not HW. Real GEMMs don't hit this because they have TMA loads + register movement. The "hardware governor" interpretation is speculation.

### 6.17 "Cluster limit = 8 portable, 16 non-portable"

MEDIUM-HIGH confidence. But line 8035: "cluster 16 = cluster misconfiguration" = ptxas error (could be opt-in needed, not HW limit). Worth re-checking that 16-CTA clusters actually work when properly enabled.

---

## Conclusion

The B300_PIPE_CATALOG.md is valuable as a **research notebook** but unreliable as a **reference manual**. Most severe issues:

1. **Clock ambiguity** (1920 vs 2032 MHz) silently affects every TFLOPS claim by ±6%.
2. **Multiple retractions** not always propagated to summary sections — readers encounter both correct and incorrect numbers.
3. **mma.sync vs tcgen05.mma conflation** in section 0 misleads about actual tensor peak.
4. **cuBLAS FP8 support** is claimed both available and unavailable.
5. **L2 peak** varies 17-36 TB/s across sections without clear reconciliation.
6. **Many "latency" numbers** are self-op-chains that the catalog explicitly identifies as 2× inflated (line 1948) but doesn't correct.
7. **Methodology problems** documented inline don't always propagate to summary tables (e.g., PDL memory-write slowdown).

**Strongest results** (highest trust):
- tcgen05.mma peaks (FP16 2.3, FP8 4.65, FP4 9.9 PFLOPS) — well-cross-checked.
- Atomic op coalescing + stride analysis + `release` cost.
- Fence cost matrices (Sec 30.G) — thoroughly remeasured.
- Reserved 1 KiB steal-trick and compatibility matrix.
- NVLink P2P peak 755-820 GB/s after thread-count correction.
- End-to-end LLM predictions matching measured decode within 4%.

**Weakest results** (treat with skepticism until re-verified):
- L2 peak BW (massive disagreement across sections).
- Clock frequency (ambiguous).
- Smem peak (19.85 vs 35 TB/s).
- DSMEM overhead (0.8% vs 5×).
- Several tensor-core "peak" claims that mix mma.sync and tcgen05.
- FP32 FMA 38 vs 72 TFLOPS (sec "Power" vs cheat sheet).
- DFMA zero-pipelining claim.

**Recommended fix**: reorder section 0 cheat-sheet to lead with tcgen05 numbers, add explicit clock annotation to all tables, retract/remove the contradicted legacy measurements, and add a single "Known Issues / Contradictions" section at the top for readers.

The catalog should NOT be cited as a reference without this audit.
