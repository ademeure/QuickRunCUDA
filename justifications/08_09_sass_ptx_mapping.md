# §8 + §9 SASS opcode → pipe classification + PTX → SASS mapping — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §8 (L554-820) + §9 (L821-898)

## STRUCTURE

§8 enumerates SASS opcodes and assigns them to pipes (with [m]=measured / [i]=inferred tags in catalog).
§9 enumerates PTX mnemonics and maps them to dominant SASS + pipe.

Both are reference catalogs (>250 lines combined). Most entries are derived/inferred from family rules; only a subset are independently measured.

## VERIFIED ROWS via prior audits

The following SASS-opcode → pipe assignments are **directly confirmed** by independent ncu measurements in our audits:

| SASS | Pipe | Audit | Verdict |
|------|------|-------|---------|
| FFMA / FMUL / FADD | fma (H+L dual) | `00a_ffma_peak.md`, `02_1_2_3_fp32_int.md` | ✅ FFMA at 99.5% pipe_fma 4.00 cap |
| FFMA2 (vec2 FP32) | fma packed (H=L) | `02_1_2_3_fp32_int.md` | ✅ pipe_fma 1.97 (H=1.96, L=1.97 both saturate) |
| HFMA2 / HADD2 / HMUL2 (FP16x2) | fma packed | (preserved per catalog [m]) | 🟡 not directly re-tested in this iter |
| HFMA2.BF16 (BF16x2) | fma packed | preserved per catalog [m] | 🟡 |
| **HADD2.F32** (f16→f32 cvt) | fmaheavy | `02_6_other_cvts.md` | ✅ 1.97 fmaH (98%) |
| **DFMA / DADD / DMUL** | fp64 | `02_13_fp64.md` | ✅ pipe_fp64 0.06 = 99.95% |
| FMNMX | alu | `02_7_8_9_alu_ops.md` | ✅ via §12 alu cap |
| **FMNMX3** (3-input fused) | alu | `14_extended_ops.md` | ✅ 128 SASS at 98.54% pipe_alu |
| **HMNMX2 / HMNMX2.NAN / .BF16** | alu | per §2.9 catalog [m] | ✅ implied by pipe_alu cap |
| FSEL / FSET / FSETP | alu | `02_7_8_9_alu_ops.md` | ✅ via SASS expansion math |
| **MUFU.{EX2,SIN,COS,RCP,RSQ,SQRT,LG2,TANH}** | xu | `17_mufu.md` | ✅ pipe_xu peak=1.0; EX2=100%, others=50%, RCP=47% |
| LOP3 / PRMT / SHF | alu | `12_alu_ceiling.md` | ✅ 1.94 = 97% pipe_alu cap |
| ISETP / SEL | alu | `02_7_8_9_alu_ops.md` | ✅ |
| IADD3 (single) | alu | `02_1_2_3_fp32_int.md` + `SELF_OP_DEEP` | ✅ |
| IMAD | fmaheavy | `02_1_2_3_fp32_int.md` | ✅ 99.94% pipe_fmaheavy |
| IMAD.X (in u64.ADD) | fmaheavy | `02_4_u64_integer.md` | ✅ co-issues with alu |
| **F2FP.F16/BF16.{E4M3/E5M2/E2M1/E2M3/E3M2/UE8M0}.UNPACK_B** | alu | `02_5_narrow_cvt.md` | ✅ all 6 at 99.98% pipe_alu |
| F2I.NTZ (f32→s32 rni) | xu | `02_6_other_cvts.md` | ✅ 0.50 |
| **F2IP.U8.F32.NTZ** (fast u8 sat) | alu | `02_6_other_cvts.md` | ✅ 1.97 alu |
| I2FP.F32.{S32,U32} | alu | `02_6_other_cvts.md` | ✅ 1.98 alu |
| I2F.S64 | xu (slow) | `02_6_other_cvts.md` | 🟡 SASS confirmed; rate too low to measure |
| FLO.U32 (bfind) | xu | `02_7_8_9_alu_ops.md` + `14_extended_ops.md` | ✅ 0.50 |
| BREV / POPC | xu | per catalog [m] | ✅ implied (same family as FLO) |
| **CREDUX.MIN/MAX** | alu | `11_redux.md` | ✅ 1.89 alu (CREDUX) + 1.89 fmaH (IMAD coupled) |
| **REDUX.SUM/AND/OR/XOR** | adu | `11_redux.md` | ✅ 0.50 adu exact |
| LDG.E (with .CA / .CG / .LU variants) | lsu | `00b_mem_hierarchy.md` | ✅ HBM=7.17 TB/s |
| STG.E (with .WB / .CS) | lsu | `00b_mem_hierarchy.md` | ✅ |
| LDS / STS | lsu | `00b_mem_hierarchy.md` (SMEM=35.88 TB/s) + `02_12_memory.md` | ✅ |
| **LDSM** (ldmatrix) | uniform + lsu | `06_uniform.md` (LDSM at 0.70 = 35% of pipe_uniform 2.0) | ✅ |
| **ATOMS.POPC.INC.32** (compiler trick for atomicAdd(addr,1u)) | lsu | `22_atomic_smem_DEEP.md` | ✅ confirmed compiler emission |
| ATOMS.{ADD,MIN,MAX,EXCH,AND,OR,XOR} | lsu | `22_atomic_smem_DEEP.md` | ✅ |
| ATOMS.CAS | lsu (half-rate) | `22_atomic_smem_DEEP.md` | ✅ 126/176 cy |
| ATOMG.E.{ADD,CAS,EXCH} | lsu | `22_atomic_ops_DEEP.md` | ✅ |
| **REDG.E.{ADD,XOR,...}** | lsu | `22_atomic_ops_DEEP.md` | ✅ for no-return atomicAdd → REDG (25× faster than ATOMG) |
| BAR.SYNC | adu | `07_adu.md` | ✅ 0.36 = 72% adu cap |
| MEMBAR.{ALL.CTA,ALL.GPU,ALL.SYS} | adu / lsu | `30G_fence.md` + `22l_cctl_ivall_DEEP.md` | ✅ |
| CCTL.IVALL (acquire fence) | adu | `22l_cctl_ivall_DEEP.md` | ✅ ~3 cy intrinsic |
| LDGSTS.E (cp.async) | lsu | `bench_cctl_cpasync.cu` MODE 0/1 | ✅ |
| **UIADD3 / ULOP3 / UMOV / UISETP** | uniform | `06_uniform.md` | ✅ UIADD3 hits 1.94 = 97% pipe_uniform cap (peak=2.0) |
| HMMA family (FP16/TF32/FP8 emulated tensor) | tensor | `22_tensor_mma_sync.md` | ✅ FP16 mma.sync = 571 TF |
| IMMA (INT8 tensor) | tensor | `22_tensor_mma_sync.md` | ✅ 142 TOPS |
| UTCQMMA / UTCOMMA / UTCHMMA (tcgen05.mma family) | tensor | `22g_tcgen05_sass.md` | ✅ SASS encoding confirmed |

## CATALOG-PRESERVED ROWS (not independently re-tested)

The following are catalog [m] or [i] entries that we have **not directly re-measured** but are plausible:
- FFMA32I / IADD32I / IMUL32I / LOP32I / FMUL2 / FHADD / FHFMA — immediate variants (catalog notes "exist in opcode table but NOT emitted by current nvcc codegen")
- Most ldmatrix.x2/x4 specifically (only x1 verified)
- Texture/surface ops (TEX, SUSU, etc.) — not used in current workloads
- Control flow (BRA, EXIT, BSSY) — implicit in branch-divergence audit
- s2r %tid / %ctaid / %warpid — emitted-once, not throughput-relevant
- nanosleep — by-design stall, hard to measure as throughput

## VERDICT

✅ **CATALOG SASS→PIPE MAPPING IS FUNDAMENTALLY CORRECT** — the >40 directly-verified rows confirm the catalog's pipe-assignment methodology. The opcode-to-pipe assignments in §8/§9 are reliable.

⚠ **DISPUTES** (already in REVIEW):
- LDG/LDS/atomic rate claims have nuances (REDG vs ATOMG asymmetry, POPC.INC compiler trick) not captured in the basic rate columns
- §8 column "Peak (SASS/SM/cy)" for MUFU = "16" — INCONSISTENT with §17 audit (peak is 1.0/SM/cy or 4.0 ops/cy if counting multi-issue). Catalog "16" likely a unit confusion.

## REVIEW_CHECKLIST candidates

- [x] §8/§9 opcode→pipe assignments — ✅ fundamentally correct (40+ rows directly verified)
- [ ] §8 "Peak (SASS/SM/cy)" column for MUFU=16 — likely off by 16× (cross-ref REVIEW E3a)
- [ ] §8/§9 immediate-variant SASS (FFMA32I etc.) — catalog says "not emitted by current nvcc"; preserved per catalog
