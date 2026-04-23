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
| §2 Complete instruction catalog | L214 | 🔍 not yet | [02_inst_catalog.md](justifications/02_inst_catalog.md) |
| §3 Contention rules | L472 | 🔍 not yet | [03_contention.md](justifications/03_contention.md) |
| §4 Rate cheatsheet | L485 | 🔍 not yet | [04_rates.md](justifications/04_rates.md) |
| §5 Narrow-format throughput | L513 | 🔍 not yet | [05_narrow.md](justifications/05_narrow.md) |
| §6 Uniform datapath | L521 | 🔍 not yet | [06_uniform.md](justifications/06_uniform.md) |
| §7 ADU pipe | L536 | 🔍 not yet | [07_adu.md](justifications/07_adu.md) |
| §8 SASS opcode → pipe classification | L554 | 🔍 not yet | [08_sass_opcode_pipe.md](justifications/08_sass_opcode_pipe.md) |
| §9 PTX → SASS mapping | L821 | 🔍 not yet | [09_ptx_sass.md](justifications/09_ptx_sass.md) |
| §11 redux.sync deep | L898 | 🔍 not yet | [11_redux.md](justifications/11_redux.md) |
| §12 pipe_alu ceiling | L937 | 🔍 not yet | [12_alu_ceiling.md](justifications/12_alu_ceiling.md) |
| §13 Predication/divergence | L957 | 🔍 not yet | [13_predication.md](justifications/13_predication.md) |
| §14 Extended op catalog | L971 | 🔍 not yet | [14_extended_ops.md](justifications/14_extended_ops.md) |
| §15 Atomics deep + latency | L1008 | 🔍 not yet | [15_atomics.md](justifications/15_atomics.md) |
| §22 mma.sync FP16/BF16 = 577 TFLOPS | L25 (cheat-sheet) | 🔍 not yet | [22_mma_sync_fp16.md](justifications/22_mma_sync_fp16.md) |
| §23 Clean MUFU sweep | L1976 | 🔍 not yet | [23_mufu_sweep.md](justifications/23_mufu_sweep.md) |
| §24 Latency reference (clock64) | L2007 | 🔍 not yet | [24_latency_clock64.md](justifications/24_latency_clock64.md) |
| §25 Final compact throughput | L2062 | 🔍 not yet | [25_final_throughput.md](justifications/25_final_throughput.md) |
| §26 Warp coop primitives | L2117 | 🔍 not yet | [26_warp_coop.md](justifications/26_warp_coop.md) |
| §27 BF16 non-tensor arith | L2133 | 🔍 not yet | [27_bf16_arith.md](justifications/27_bf16_arith.md) |
| §28 Compiler-emission gaps | L2147 | 🔍 not yet | [28_compiler_gaps.md](justifications/28_compiler_gaps.md) |
| §29 Warp-reduce reality | L2186 | 🔍 not yet | [29_warp_reduce.md](justifications/29_warp_reduce.md) |
| §30 TMA + mbarrier | L2218 | 🔍 not yet | [30_tma_mbarrier.md](justifications/30_tma_mbarrier.md) |
| §30.B Atomic latency (1-thread chain) | L2679 | 🔍 not yet | [30B_atom_latency.md](justifications/30B_atom_latency.md) |
| §30.G Memory fence costs | L2883 | 🔍 not yet | [30G_fence.md](justifications/30G_fence.md) |
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
- **[CANDIDATE TESTS]:** `tests/bench_fp32_fma.cu`, `tests/bench_a1_dual_issue.cu`, `tests/bench_a1_dual_v2.cu`
- **[STATUS]:** 🔍 — see [justifications/00a_ffma_peak.md](justifications/00a_ffma_peak.md)

#### §0.MEM — Memory hierarchy table (L37)
- **[CLAIM]:** Smem read 35.6 TB/s (98% theoretical at 1.92 GHz); HBM 7.18 TB/s ncu-verified
- **[CANDIDATE TESTS]:** `tests/bench_smem_bw.cu`, `tests/bench_hbm_bw.cu`, `tests/standalone/v45_smem_bw_conflict.cu`
- **[STATUS]:** 🔍 — see [justifications/00b_mem_hierarchy.md](justifications/00b_mem_hierarchy.md)

(Sections continue below as I process them.)
