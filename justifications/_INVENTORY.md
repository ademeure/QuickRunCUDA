# B300 Benchmark Catalog Inventory — Mapping Claims to Test Files

> **Note:** This table maps the first 50 sections/subsections of `B300_PIPE_CATALOG.md` to their likely source test files. Confidence levels: **HIGH** (test name matches claim topic exactly + SASS-verified), **MED** (multiple candidates or inferred), **LOW** (topic-based guess only).
>
> **Status (2026-04-23 final):** This inventory was the early-session scaffolding to identify which test files map to which catalog claim. **Most claims listed here have since been re-run in this audit** with full SASS+ncu evidence — see `JUSTIFIED_B300_PIPE_CATALOG.md` for the per-section verdicts. The "Likely Test File" column was generally correct; "MED/LOW" entries were resolved either by direct re-test (if a clear test existed) or by cross-reference to other justifications. For the authoritative claim-to-evidence mapping, use the JUSTIFIED index, not this inventory.

| § | Section Title | Headline Claim | Catalog Line | Likely Test File(s) | Confidence |
|---|---|---|---|---|---|
| 0 | B300 design cheat-sheet | FP16/BF16 tensor peak 577 TFLOPS (`mma.sync.m16n8k16`) | 25 | `bench_hmma_peak.cu`, `bench_a1_dual_issue.cu` | HIGH |
| 0.a | TF32 tensor | 288 TFLOPS (`mma.sync.m16n8k8`) | 26 | `bench_v8_hmma_bf16_peak.cu` or tensor benchmark | MED |
| 0.b | FP8 tensor via mma.sync | 276 TFLOPS emulated (F2FP + HMMA) | 27 | `bench_v8_hmma_fp8_peak.cu` | MED |
| 0.c | INT8 mma.sync | 142 TOPS via IMMA | 28 | `bench_imad_peak.cu` or `bench_v8_imad_peak.cu` | MED |
| 0.d | FP32 scalar FFMA peak | 71.8 TFLOPS (98.8% SOL) with 8-chain pattern | 30 | `bench_ffma_peak.cu` | HIGH |
| 0.e | FP32 FFMA2 packed | 72.3 TFLOPS (99.4% SOL) | 31 | `bench_fmul_vs_ffma.cu` | MED |
| 0.f | FP16 HFMA2 packed | 72.3 TFLOPS-FP16 | 32 | `bench_fp16_hmma.cu` | MED |
| 0.g | FP16 HFMA scalar | 72.2 TFLOPS-FP16 | 33 | `bench_fp16_hmma.cu` | MED |
| 0.h | BF16 BFMA2 | 72.3 TFLOPS-BF16 | 34 | `bench_v6_h5_bf16_fp16_cvt.cu` | MED |
| 0.i | FP64 DFMA scalar | 0.95 TFLOPS (1/76× of FFMA) | 35 | `bench_v8_fp64_peak.cu` | HIGH |
| 0.j | Registers (smem→reg) | 35.6 TB/s via `ld.volatile.shared.v4.u32` | 41 | `bench_smem_read_bw.cu` or `bench_shmem_reduce.cu` | MED |
| 0.k | L1 cache | 28.7 TB/s | 42 | L2/L1 sweep benchmarks | LOW |
| 0.l | L1 hit (WS ≤ 1 MB) | 36.1 TB/s | 43 | L2/memory hierarchy sweep | LOW |
| 0.m | L2 plateau (4–128 MB) | 22-26 TB/s | 44 | `bench_l2_peak.cu`, memory sweep | HIGH |
| 0.n | L2 knee transition | gradual 23 → 22 TB/s across 4-64 MB | 45 | Memory hierarchy sweep | MED |
| 0.o | DRAM HBM3E peak | 7.18 TB/s (ncu-verified) | 46 | `bench_dram_peak.cu` | HIGH |
| 0.p | DRAM write | 7.09 TB/s | 46 | `bench_v8_hbm_write_peak.cu` | HIGH |
| 0.q | Constant memory broadcast (LDC.32) | 17.8 TB/s eff | 47 | `bench_ldg_constant_vs_strong.cu` | MED |
| 0.r | Constant memory (LDC.64) | 33.7 TB/s eff | 48 | `bench_ldg_constant_vs_strong.cu` | MED |
| 0.s | Local memory spill | 1.3 TB/s (52× slower than smem) | 49 | Implicit in spill analysis | LOW |
| 0.t | TMEM read BW | 55.92 TB/s (1R/iter), drops to 31 with 4R/iter | 50 | `bench_tmem_*.cu`, tensor memory benchmarks | MED |
| 0.u | TMEM write BW | 97.93 TB/s (1W/iter), 131 TB/s with 4W/iter | 50 | `bench_tmem_*.cu` | MED |
| 0.v | TMA cp.async.bulk issue rate | 48 cy/inst floor | 58 | `bench_tma_*.cu`, `bench_tma_min.cu` | HIGH |
| 0.w | TMA crossover point | 8 KiB per TMA for issue→engine-bound | 59 | `bench_tma_throughput.cu` | MED |
| 0.x | TMA single-CTA peak | 241 GB/s/SM (64 KB, DEPTH=3) | 60 | `bench_tma_*.cu` | MED |
| 0.y | TMA chip-wide realistic | 29.2 TB/s / 197 GB/s/SM (8 KB batched) | 61 | `bench_tma_pc.cu` | MED |
| 0.z | TMA max size | 1 048 560 B (1 MB − 16) | 62 | Compilation property (not measured) | LOW |
| 0.aa | TMA 4 KiB batched peak | 151 GB/s/SM, 21.8 TB/s chip | 63 | `bench_tma_throughput.cu` | MED |
| 0.ab | mbarrier.arrive latency | 8.1 cy | 69 | `bench_membar_sm_sweep.cu` | HIGH |
| 0.ac | mbarrier test_wait/try_wait | 6–8 cy when ready | 71 | `bench_membar_*.cu` | MED |
| 0.ad | mbarrier RTT | 54 cy round-trip (single thread, count=1) | 73 | `bench_membar_sys_scenarios.cu` | MED |
| 0.ae | __syncthreads() at BS=512 | 45 cy | 74 | `bench_barrier_real.cu` | MED |
| 0.af | __syncthreads() at BS=1024 | 89 cy | 75 | `bench_barrier_real.cu` | MED |
| 0.ag | __syncwarp() | 2.8 cy | 76 | `bench_latency.cu` | HIGH |
| 1 | Pipe topology | 4 SMSPs, 4.00 warp-inst/SM/cy aggregate dispatch | 189 | `bench_all_pipes.cu`, `bench_a1_dual_issue.cu` | MED |
| 1.a | pipe_alu capacity | 2.00 warp-inst/SM/cy | 196 | `bench_alu_lat*.cu`, `bench_alu_lat_tp.cu` | MED |
| 1.b | pipe_fmaheavy capacity | 2.00 warp-inst/SM/cy | 197 | `bench_a1_dual_issue.cu` | MED |
| 1.c | pipe_fmalite capacity | 2.00 warp-inst/SM/cy | 198 | `bench_a1_dual_issue.cu` | MED |
| 1.d | pipe_fma (parent) | 4.00 dual, 2.00 packed | 199 | `bench_a1_dual_issue.cu`, `bench_ffma_peak.cu` | MED |
| 1.e | pipe_xu capacity | 0.50–1.00 (compound/simple) | 200 | `bench_v8_mufu_peak.cu`, `bench_mufu_lat.cu` | MED |
| 1.f | pipe_lsu | 1.00 nominal | 201 | `bench_dram_peak.cu`, `bench_tma_*.cu` | MED |
| 1.g | pipe_adu | ~0.5 (address/sync/match) | 202 | `bench_membar_*.cu` | MED |
| 1.h | pipe_uniform | ~1.0 (uniform register/LDSM) | 203 | `bench_adu_uniform.cu`, `bench_activemask.cu` | MED |
| 1.i | pipe_tensor | — (hmma/imma subpipes, not measured here) | 204 | `bench_hmma_peak.cu`, `bench_imad_peak.cu` | LOW |
| 1.j | pipe_fp64 | 0.05 warp-inst/SM/cy (throttled) | 205 | `bench_v8_fp64_peak.cu` | HIGH |
| 2 | Complete instruction catalog | Measured SASS throughput reference | 214 | Multiple `bench_*.cu` family | LOW |
| 2.1 | FP32 scalar FFMA | 128 FFMA/SM/cy = 256 FLOPS/SM/cy | 223 | `bench_ffma_peak.cu` | HIGH |
| 2.2 | FP32 FFMA2 packed | 64 warp-inst/SM/cy = 128 FMAs (256 FLOPS) | 238 | `bench_fmul_vs_ffma.cu` | MED |
| 2.3 | Integer IMAD | 64 IMAD/SM/cy | 251 | `bench_imad_peak.cu`, `bench_a1_dual_issue.cu` | HIGH |
| 2.4 | u64 integer add | 64 u64-adds/SM/cy (2 SASS/op) | 267 | Implicit in u64 tests | LOW |
| 2.5 | CVT F2FP UNPACK | 128 elements/SM/cy (2.00 warp-inst) | 286 | `bench_f2fp_oneway.cu`, `bench_cvt_catalog.cu` | MED |
| 2.6 | CVT f16→f32 HADD2.F32 | 64 warp-inst/SM/cy (2.00 fmaH) | 321 | `bench_v6_h5_bf16_fp16_cvt.cu` | MED |
| 2.7 | Bitwise LOP3 | 64 LOP3/SM/cy (pipe_alu) | 339 | `bench_alu_lat_tp.cu` | MED |
| 2.8 | Compare ISETP | 64 ISETP/SM/cy (pipe_alu) | 358 | `bench_alu_lat_tp.cu` | MED |
| 2.9 | MIN/MAX FMNMX | 64 FMNMX/SM/cy on **pipe_alu** (surprising) | 372 | `bench_alu_lat_tp.cu` | MED |
| 2.10 | Transcendental MUFU | 16–20 SASS/SM/cy (0.5 warp-inst) | 390 | `bench_v8_mufu_peak.cu`, `bench_mufu_lat.cu` | HIGH |
| 2.11 | SHFL.SYNC | 32 SASS/SM/cy (pipe_lsu) | 405 | `bench_latency.cu`, `bench_v8_shfl_peak.cu` | HIGH |
| 2.12 | LDG global load | ~32 issue capacity (DRAM-bound in practice) | 437 | `bench_dram_peak.cu` | MED |
| 2.13 | DFMA FP64 | 1.6 DFMA/SM/cy (0.05 warp-inst, throttled) | 449 | `bench_v8_fp64_peak.cu` | HIGH |

---

## Search Strategy for Remaining Sections

For sections 3–9 and beyond, use the following pattern-matching approach:

1. **Section 3 (Contention rules):** `bench_a1_dual_issue.cu` + any dual-pipe contention tests
2. **Section 4 (Rate cheatsheet):** Derived from all peak measurements in Section 2
3. **Section 5 (Narrow-format throughput):** `bench_f2fp_oneway.cu`, `bench_cvt_*.cu` family
4. **Section 6 (Uniform datapath):** `bench_adu_uniform.cu`, compiler-generated uniform ops
5. **Section 7 (ADU pipe_adu):** `bench_membar_*.cu`, `bench_barrier_*.cu`
6. **Section 8 (SASS opcode classification):** `bench_misc_ops.cu` (explicitly referenced line 971)
7. **Section 9 (PTX→SASS mapping):** ISA coverage from all micro-benchmarks

## Test File Categories

| Category | Files | Purpose |
|---|---|---|
| **Peak measurements** | `bench_*_peak.cu` (16 files) | Throughput ceiling for each opcode family |
| **Latency** | `bench_latency.cu`, `bench_mufu_lat.cu`, `bench_alu_lat*.cu` | Single-op and chain latencies |
| **Dual-issue / contention** | `bench_a1_*.cu` (4 files) | Pipe saturation and interaction |
| **Memory hierarchy** | `bench_dram_peak.cu`, `bench_l2_peak.cu`, memory sweep | DRAM/L2/L1/smem throughput |
| **Synchronization** | `bench_membar_*.cu`, `bench_barrier_*.cu` | Barrier and sync cost |
| **Conversions** | `bench_f2fp_*.cu`, `bench_cvt_*.cu` | Narrow-format conversions |
| **TMA / async** | `bench_tma_*.cu` (5+ files) | Tensor memory and cp.async | 
| **Atomics** | `bench_atom_*.cu`, `bench_atomic_*.cu` | Global and shared memory atomics |
| **FP16/BF16 non-tensor** | `bench_v6_h5_*.cu` | Packed FMA variants |

---

## Methodology Notes

**How to verify a mapping:**

1. Grep for the section's claim text (e.g., "71.8 TFLOPS") in test files:
   ```bash
   grep -r "71.8\|71\.8" /root/github/QuickRunCUDA/tests/
   ```

2. Check kernel signature for the expected pattern:
   - FFMA peak: 8-chain, 1024-FFMA inner loop, `#pragma unroll 1`
   - DRAM peak: `ld.global.v8.u32` with ≥2 CTAs/SM
   - Smem BW: `ld.volatile.shared.v4.u32` with 32-way bank-conflict-free stride

3. Cross-reference with `run_microbench.sh` and `MICROBENCH_RESULTS.md` for authoritative line numbers.

---

## Known Gaps & Caveats

- **Sections 10+:** Not included in this initial 50-item pass (covers sections 0–9 + subsections).
- **Line number shifts:** If the catalog was edited post-measurement, line numbers may differ by ±5 lines.
- **Multiple sources per claim:** Some numerical claims (e.g., "7.18 TB/s DRAM") were generated by 3–5 different kernel variants to establish rigor; only the primary one is listed.
- **DCE-inflated numbers:** Earlier measurements flagged as "DCE-folded" (e.g., TMEM 295 TB/s) are retracted in the catalog; their source tests are not recommended for verification.

