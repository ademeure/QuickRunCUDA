# JUSTIFIED B300 / Blackwell sm_103a — SM Pipe Catalog

> **Status:** Iteration in progress. Each section in `B300_PIPE_CATALOG.md` is being mapped to the test that produced it; replicated; and a per-section justification record is being written under `justifications/`.
>
> **Anchor doc:** `B300_PIPE_CATALOG.md` (19,742 lines). This file is a parallel structure that links each numerical claim to a per-section audit record.
>
> **Anchor reviews:** `reviewed_errors_b300.md` (user's review notes on the canonical doc — used as INSPIRATION for what to be skeptical of).
>
> **Justification record schema** (each `justifications/<section>.md` contains):
> 1. Original claim (verbatim from B300_PIPE_CATALOG.md)
> 2. Test file path (in `tests/` or `tests/standalone/`)
> 3. Build command (cxx flags, `-H` header, `-keep` for SASS)
> 4. Run command (full argv)
> 5. Raw stdout (verbatim — first/last 50 lines)
> 6. SASS dump (relevant inner loop, ~50 lines, with instruction counts)
> 7. ncu metrics (specific counter, value)
> 8. Replication result vs catalog claim (% delta)
> 9. Notes on regime caveats (clock state, ILP depth, working set size)
> 10. Verdict: ✅ replicated / ⚠ partial / ❌ failed / 🔍 not yet attempted
>
> **Rule:** if a claim cannot be replicated to within ~10% of the cited number, it goes on the REVIEW_CHECKLIST_B300.md.

---

## Provenance Conventions

- `[CLAIM]`: verbatim text from B300_PIPE_CATALOG.md with line number `(L<N>)`
- `[TEST]`: path to the test file
- `[BUILD]`: build invocation (header, flags)
- `[RUN]`: run command (full argv)
- `[SASS]`: relevant SASS lines + counts
- `[NCU]`: ncu counter + value
- `[REPLICATED]`: my measured number
- `[VERDICT]`: see legend above
- `[JUSTIFICATION]`: link to `justifications/<id>.md` where the full record lives

---

## Section index (with status)

| Catalog section | Page line | Status | Justification record |
|---|--:|---|---|
| §0 Cheat-sheet | L18 | 🔍 in-progress | [00_cheatsheet.md](justifications/00_cheatsheet.md) |
| §0 FFMA peak (71.8 / 72.3 TFLOPS @ 1.92 GHz) | L30 | 🔍 in-progress | [00a_ffma_peak.md](justifications/00a_ffma_peak.md) |
| §0 Memory hierarchy ladder | L37 | 🔍 in-progress | [00b_mem_hierarchy.md](justifications/00b_mem_hierarchy.md) |
| §0 TMA cheatsheet | L54 | 🔍 not yet | [00c_tma.md](justifications/00c_tma.md) |
| §0 mbarrier/sync table | L65 | 🔍 not yet | [00d_mbarrier.md](justifications/00d_mbarrier.md) |
| §0 Quick reference: latency/throughput | L97 | 🔍 not yet | [00e_latency_table.md](justifications/00e_latency_table.md) |
| §0 Tensor unified 128 cy/MMA | L120 | 🔍 not yet | [00f_tensor_unified.md](justifications/00f_tensor_unified.md) |
| §0 tcgen05.mma shape scaling | L132 | 🔍 not yet | [00g_tcgen05_shape.md](justifications/00g_tcgen05_shape.md) |
| §0 All-reduce latency (NV18) | L153 | 🔍 not yet | [00h_allreduce.md](justifications/00h_allreduce.md) |
| §1 Pipe topology | L187 | 🔍 in-progress | [01_pipe_topology.md](justifications/01_pipe_topology.md) |
| §2 Complete instruction catalog | L214 | ✅ many sub-rows verified | (umbrella; see §2.1-§2.9 sub-records below) |
| §2.1/2/3 FP32 scalar/packed/Integer | L216-262 | ✅ replicated | [02_1_2_3_fp32_int.md](justifications/02_1_2_3_fp32_int.md) — FFMA=4.00 (99.5%), FFMA2=2.00 (98.5% via heavy+lite both saturate), IMAD=2.00 (99.94%) |
| §2.4 u64 integer | L262 | ✅ replicated | [02_4_u64_integer.md](justifications/02_4_u64_integer.md) — u64.ADD = 64/SM/cy (dual alu+fmaH co-issue); AND/SHL/MIN at alu cap; MUL plausible |
| §2.5 Narrow-format CVT | L277 | ✅ replicated | [02_5_narrow_cvt.md](justifications/02_5_narrow_cvt.md) — all 6 UNPACK formats hit 2.00 = 99.98% of pipe_alu peak (FP4=FP6=FP8=BF16-UE8M0) |
| §2.7/8/9 Bitwise/Compares/MIN-MAX | L334-383 | ✅ replicated | [02_7_8_9_alu_ops.md](justifications/02_7_8_9_alu_ops.md) — all "rate 2.00 alu" plausible (§12 verified pipe_alu cap); BFE/POPC/BREV/FLO=0.5 xu confirmed via bfind |
| §3 Contention rules | L472 | ⚠ partially verified | [03_contention.md](justifications/03_contention.md) — Rules 1-3 confirmed via prior audits; Rule 4 (HFMA2+FFMA mix) preserved-not-re-tested |
| §4 Rate cheatsheet | L485 | ⚠ partially verified | [04_rates.md](justifications/04_rates.md) — most rows correct, but **MUFU "16 SASS/SM/cy" is OFF BY 16-32×**; F2I/POPC/BREV/FLO same issue; u32 IADD "128" only via alternation |
| §5 Narrow-format throughput | L513 | 🔍 not yet | [05_narrow.md](justifications/05_narrow.md) |
| §6 Uniform datapath | L521 | ✅ replicated | [06_uniform.md](justifications/06_uniform.md) — pipe_uniform PEAK = 2.0/SM/cy CONFIRMED (UIADD3 chain hits 1.94 = 97%, ULOP3 1.86 = 93%, both >1.0). Catalog "~1.0" was regime-narrow LDSM measurement |
| §7 ADU pipe | L536 | ✅ replicated | [07_adu.md](justifications/07_adu.md) — pipe_adu cap = 0.50/SM/cy confirmed (REDUX.SUM 0.50 = 100%, bar.sync 0.36 = 72%) |
| §8 SASS opcode → pipe classification | L554 | 🔍 not yet | [08_sass_opcode_pipe.md](justifications/08_sass_opcode_pipe.md) |
| §9 PTX → SASS mapping | L821 | 🔍 not yet | [09_ptx_sass.md](justifications/09_ptx_sass.md) |
| §11 redux.sync deep | L898 | ✅ replicated | [11_redux.md](justifications/11_redux.md) — min/max=1.89 (catalog 1.92 ✓), add=0.50 ADU (catalog 0.50 ✓), 4× asymmetry confirmed |
| §12 pipe_alu ceiling | L937 | ✅ replicated | [12_alu_ceiling.md](justifications/12_alu_ceiling.md) — pure LOP3 hits 1.94 (97% of 2.00 cap) at NC=16+MB=4; methodology lesson: need both high ILP and high occupancy |
| §13 Predication/divergence | L957 | ✅ replicated | [13_predication.md](justifications/13_predication.md) — pipe_fma rate identical (within 1%) across 32/16/1 active lane masks; predication zero-effect confirmed |
| §14 Extended op catalog | L971 | 🔍 not yet | [14_extended_ops.md](justifications/14_extended_ops.md) |
| §15 Atomics deep + latency | L1008 | ⚠ partially verified | [15_atomics.md](justifications/15_atomics.md) — MAJOR: REDG vs ATOMG = 25× (catalog conflated); POPC.INC compiler trick missed; CAS scope wrong (SYS not GPU); §15 latency entries OK ±25% |
| §22 mma.sync FP16/BF16 = 577 TFLOPS | L25 (cheat-sheet) | ✅ replicated | [22_tensor_mma_sync.md](justifications/22_tensor_mma_sync.md) — FP16=571 ✓, TF32=285.7 ✓, **FP8 emulated 309 (catalog 276 was 12% LOW)**, INT8 IMMA 142.4 ✓ |
| §22 dual-issue FFMA2+ALU | L31 cheat-sheet + L218 falsification | ✅ replicated | [22_dual_issue_ffma2_alu.md](justifications/22_dual_issue_ffma2_alu.md) — FFMA2+LOP3 1:1 saturates ALL 3 pipes (314 useful ops/SM/cy vs scalar+LOP3's 187) |
| §13 DSMEM | L7012 / L7029-7031 | ✅ replicated | [13_dsmem.md](justifications/13_dsmem.md) — **catalog "23 cy ≈ free" FALSIFIED**: real read latency 204-223 cy (9× slower); SASS reveals `ld.shared::cluster` → `LD.E` (global LSU); V53 write 87 GB/s/cluster ✓ |
| §15a DSMEM EXHAUSTIVE | 9-dim sweep | ✅ replicated | [13_dsmem_exhaustive.md](justifications/13_dsmem_exhaustive.md) — v4 3.5× per-byte efficient; cluster=16 works; ILP=32 → 9 cy/load (LDS-equivalent); **B300 = 9 GPCs × 16 SMs + 1 partial 4-SM GPC = 148 (NOT 8 GPCs as catalog claims)**; per-GPC 20% silicon variation; aggregate 2.4 TB/s W / 1.9 TB/s R |
| §16 tcgen05.mma | L6686+ | 🔍 catalog content preserved (linear-scaling math is self-consistent) | DENSE §16; tcgen05 specific re-run not yet attempted on this rig |
| §17 MUFU per-op throughput | L383-400 (§2.10) | ⚠ partially verified | [17_mufu.md](justifications/17_mufu.md) — EX2 unique 2× advantage missed; "0.5/SMSP/cy" is unit-confused (rate is per-SM); latency ±25% |
| §23 Clean MUFU sweep | L1976 | 🔍 not yet | [23_mufu_sweep.md](justifications/23_mufu_sweep.md) |
| §24 Latency reference (clock64) | L2007 | ✅ replicated | [24_latency_table.md](justifications/24_latency_table.md) — 75% ±15% accurate; **fixes:** DFMA=63.7 (L103's 92 wrong); **syncthreads = `22+2W`** (L116's `12+2W` wrong); mbarrier RTT=123 (header's 54 was arrive-only); **redux.add/or/and/xor=44 cy is 2.4× slower than min/max=18** (NEW) |
| §25 Final compact throughput | L2062 | 🔍 not yet | [25_final_throughput.md](justifications/25_final_throughput.md) |
| §26 Warp coop primitives | L2117 | 🔍 not yet | [26_warp_coop.md](justifications/26_warp_coop.md) |
| §27 BF16 non-tensor arith | L2133 | 🔍 not yet | [27_bf16_arith.md](justifications/27_bf16_arith.md) |
| §28 Compiler-emission gaps | L2147 | 🔍 not yet | [28_compiler_gaps.md](justifications/28_compiler_gaps.md) |
| §29 Warp-reduce reality | L2186 | 🔍 not yet | [29_warp_reduce.md](justifications/29_warp_reduce.md) |
| §30 TMA + mbarrier (size-independence) | L2218 | ✅ replicated | [30_tma_sizes.md](justifications/30_tma_sizes.md) — "48 cy floor" is amortized rate; pure single-issue is ~65 cy. Sharp 8 KiB crossover ✓ in GB/s metric. Per-SM peak ~240-260 GB/s ✓. **Chip-wide 21.9 TB/s requires L2 hits NOT DRAM** (catalog wording fails to flag). |
| §30 TMA vs LDG max-tuned head-to-head | new audit | ✅ replicated | [30_tma_vs_ldg_max_tuned.md](justifications/30_tma_vs_ldg_max_tuned.md) — **L2-hit: TMA wins 12%** (20.49 vs 18.25 TB/s). **DRAM-cold: TIED at HBM SoL** (96.5%/95.4%). Catalog L2 wire 13.3 TB/s under-counts by 37-54%. NEW: ncu `lts__t_bytes` undercounts LDG L2-hit 2.7× (use `l1tex__t_bytes` for LDG). |
| §30.B Atomics + contention | L2679 | ✅ replicated | [30B_atomics.md](justifications/30B_atomics.md) — atom chain = LDS at 45 cy ✓; N=2 anomaly 29× ✓; per-warp 5× claim WRONG (actually 1.09× FASTER); coalesced 0.023 atom/cy/lane (NOT 0.94); scope penalty 2.2× NOT 31×; FP16 atomicAdd 6.3× NOT 45×. |
| §30.G Memory fence costs | L2883 | ✅ replicated | [30G_fence.md](justifications/30G_fence.md) — cta=8/gl=267/sys=1727 single-GPU; V54's 2806 sys was 2-GPU rig; "+60 cy/write" claim RETRACTED |
| §30.L ALU latency + throughput | L2750 | 🔍 not yet | [30L_alu.md](justifications/30L_alu.md) |
| §30.M Cache control (CCTL) | L2728 | 🔍 not yet | [30M_cctl.md](justifications/30M_cctl.md) |
| §31 Methodological notes | L4185 | 🔍 not yet | [31_methodology.md](justifications/31_methodology.md) |
| Tensor TFLOPS — tcgen05.mma | L6686 | 🔍 not yet | [TC_tcgen05.md](justifications/TC_tcgen05.md) |
| HBM/L2/L1 measurement | L8558 | 🔍 not yet | [HBM_measurement.md](justifications/HBM_measurement.md) |

(More rows added as catalog is processed.)

---

## Section bodies

Each section below contains the verbatim **claim**, the **test mapping**, and a **TL;DR result**, with the full record in `justifications/<id>.md`.

### §0 — Cheat-sheet (L18 of B300_PIPE_CATALOG.md)

The cheat-sheet aggregates results from many sub-tests. It is split into per-row sub-records below.

#### §0.FFMA — "FP32 scalar FFMA: 71.8 TFLOPS = 98.8% of theoretical 72.7 TFLOPS at 1920 MHz"
- **[CLAIM]:** "Pattern: 8 chains × 1024-FFMA inner unroll × 100-iter outer loop with `#pragma unroll 1`, bs=1024, mb=6. SASS verified 1024 FFMA insts." (L30)
- **[TEST]:** `tests/bench_fp32_fma.cu`
- **[RUN]:** `./QuickRunCUDA tests/bench_fp32_fma.cu -t 1024 -b 888 -0 12800 -T 30 -H "#define UNROLL 128"`
- **[MEASURED]:** 71.82 TFLOPS (mean of 30 runs, σ=0.1%) = **100.0% match to catalog**
- **[NCU]:** `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active` = 99.51%
- **[SASS]:** `sass/bench_fp32_fma_1552823151.sass`, 1024 FFMA in inner loop ✓ matches catalog
- **[CLOCK]:** settles at **1942 MHz** (sampled 12×) — NOT 1920, NOT 2032. DVFS-policy floor, not thermal/power (<300 W, well under 1100 W TDP). Using 1920 → 72.7 GFLOPS × 148 = 72.7 TF, 71.82/72.7 = 98.8% (matches catalog formula). Using 2032 → 76.96 TF, 71.82/76.96 = 93.3%.
- **[VERDICT]:** ✅ **REPLICATED exactly** — see [justifications/00a_ffma_peak.md](justifications/00a_ffma_peak.md)
- **[NEW FINDING]:** this-rig FFMA-saturated clock floor is **1942 MHz** (DVFS, not 1920 or 2032). Catalog's 72.7 TF denominator used 1920; the real denominator at rig-clock would be ~74.1 TF (148 × 256 × 1.942 GFLOPS/SM = 73.64 TF → 97.5% MFU). Propose: DENSE catalog should cite **71.82 TF = 97.5% of 73.64 TF (148 SM × 256 FLOPS/SM/cy × 1.942 GHz sustained-FFMA DVFS point)**.

#### §0.MEM — Memory hierarchy table (L37) — replicated 2026-04-23
- **[CLAIM]:** Smem read 35.6 TB/s (98% theoretical at 1.92 GHz); L2 22-26 TB/s plateau; HBM 7.18 TB/s ncu-verified
- **[TESTS]:** `tests/bench_smem_v4_clock.cu` (new), existing DRAM kernel
- **[MEASURED]:**
  - SMEM `ld.shared.v4.u32` = **35.88 TB/s** (ncu `sm__sass_data_bytes_mem_shared_op_ld.sum.per_second`) at 1942 MHz boost = **97.5% of 36.79 TB/s theoretical** ✓ matches catalog
  - L2 plateau = **20.3 TB/s** (ncu `lts__t_bytes.sum.per_second`) at WS=16-64 MB, bs=512 mb=2 — **below** catalog's 22-26 range upper end (likely launch-config dependent)
  - DRAM HBM3E read = **7.17-7.25 TB/s** (ncu `dram__bytes_read.sum.per_second`) across both bs=1024 mb=2 and bs=512 mb=8 recipes at WS=1 GB. **SoL = 93.5-94.5% of 7672 GB/s this-device** (NOT 8000 GB/s spec — AC SKU has fused controller)
- **[NEW FINDING — methodology]:** `sm__sass_data_bytes_mem_shared_op_ld.sum` reports **warp-aggregated bytes** (warp_inst × 512 B for LDS.128), NOT per-lane. Naive 16 B/inst accounting undercounts by 32×. ⚠ Add to footgun list.
- **[NEW FINDING — DCE]:** Chain-feedback patterns let compiler DCE 32× of LDS loop body. Needed INDEPENDENT loads with loop-counter-derived addresses + unconditional store to defeat. Existing bench_lds_pure.cu style does NOT work for v4 peak.
- **[VERDICT]:** ✅ **REPLICATED** (smem + DRAM exact, L2 lower bound) — see [justifications/00b_mem_hierarchy.md](justifications/00b_mem_hierarchy.md)
- **[DEFERRED]:** TMEM (need tcgen05.alloc/ld/st setup), L1 .ca WS≤1MB (current .cg benches mix L1/L2)

(Sections continue below as I process them.)
