# §4 Rate cheatsheet — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §4 (L485-513)
**Cross-references:** Various per-pipe justification records.

## CLAIM (catalog L485-513, §4)

| Op | SASS/SM/cy | Catalog notes |
|----|-----------:|---------------|
| Scalar FP32 FMA (FFMA) | 128 | dual-pipe heavy + lite |
| Scalar FP32 ADD/MUL | 128 | same |
| FFMA2 / HFMA2 / BF16-FMA | 64 | but 2 FLOPs per inst → 128 FLOPS/SM/cy |
| IMAD u32 | 64 | fmaH only |
| DP4A / DP2A | 64 | fmaH |
| u32 ADD | 128 | as IADD3 (1 SASS = 2 adds) or split alu+fmaH |
| u64 ADD | 64 | requires 1 alu + 1 fmaH per op |
| LOP3 / PRMT / SHL / SHR / SHF / FMNMX / HMNMX2 / VIMNMX3 | 64 | all pipe_alu |
| F2FP UNPACK (all formats) | 64 | = 128 elements/SM/cy (×2 ops) |
| F2FP PACK (all formats) | 32–64 depending on feedback path | pipe_alu |
| BFE | 32 | 2 SASS per PTX op |
| SELP / setp+selp / vote.ballot | 32–64 | pipe_alu |
| SHFL.SYNC.* | 32 | pipe_lsu |
| LDS / STS / LDG / STG | ~32 issue | pipe_lsu; DRAM-bound if streaming |
| **MUFU (EX2/RSQ/SIN/COS/LG2/TANH/SQRT/RCP)** | **~16** | pipe_xu, compound |
| F2I (f32→s32/s64/s8), POPC, BREV, FLO | 16 | pipe_xu |
| BAR.SYNC | ~12 | pipe_adu |
| MATCH.ANY | serial | pipe_adu, slow |
| FP64 FMA (DFMA) | 1.6 | pipe_fp64, throttled |

## VERDICT — row by row

### FFMA = 128 SASS/SM/cy ✅ CONFIRMED
- `00a_ffma_peak.md`: 71.82 TFLOPS = 99.5% pipe_fma. 128 SASS/SM/cy at 1.92 GHz × 148 SM × 2 FLOPS = 72.7 TFLOPS — matches measurement.

### FFMA2 / HFMA2 / BF16-FMA = 64 ✅ CONFIRMED
- Per `01_pipe_topology.md`: pipe_fma packed cap = 2.00 warp-inst/SM/cy = 64 SASS/SM/cy. Confirms ~128 FLOPS/SM/cy via 2×.

### IMAD u32 = 64 ✅ CONFIRMED
- `SELF_OP_DEEP_CORRECTION.md`: IMAD on pipe_fmaheavy at 0.5/SMSP/cy = 2.00/SM/cy = 64 SASS/SM/cy.

### u32 IADD = 128 ⚠ PARTIALLY CORRECT
- IADD3 alone hits ~64 (alu cap). Catalog claims 128 by combining IADD3 (alu) + IMAD.IADD (fmaH alternation), per `SELF_OP_DEEP_CORRECTION.md` mechanism. ✅ for the alternation case.
- For pure IADD3 (single pipe), max is 64. Catalog "128" requires the alternation pattern.

### LOP3 / PRMT / SHF / FMNMX = 64 ✅ CONFIRMED
- `22e_reuse_cache.md`: LOP3 saturates pipe_alu at 96.97% × 2.00 = 1.94 warp-inst/SM/cy = 62.1 SASS/SM/cy ≈ 64 ✅

### LDS / STS / LDG / STG = ~32 issue ✅ CONFIRMED
- `00b_mem_hierarchy.md`: SMEM 35.88 TB/s with v4.b32 loads → consistent with 32 SASS-LDS/SM/cy issue rate.

### **MUFU = ~16 SASS/SM/cy** ⚠ INCONSISTENT WITH §17 AUDIT

This row is the most problematic in §4.

- Catalog §4 claims MUFU = ~16 SASS/SM/cy (= 16 inst/SM/cy at saturation).
- `17_mufu.md` audit measured EX2 = 4.28 cy/op/warp at saturating ILP (N=16).
- Per-warp throughput: 0.234 SASS/cy/warp.
- Per SM (4 SMSPs all running EX2 in parallel): max 4 × 0.234 = 0.94 SASS/cy/SM **assuming pipe_xu is per-SMSP**.
- DENSE §1 reports "MUFU.EX2 saturates at 98.46% of pipe_xu" with pipe_xu cap = ~1.0/SM/cy → 0.98 SASS/cy/SM.

**Three conflicting numbers for MUFU.EX2 SM-rate:**
- Catalog §4: 16 SASS/SM/cy
- DENSE §1: 1.0 SASS/SM/cy
- §17 audit (4 SMSPs each at 0.234): 0.94 SASS/SM/cy

Either:
- (a) Catalog §4's 16 is **off by 16×** (unit confusion or transcription error)
- (b) "MUFU = 16" in catalog refers to a different unit (possibly 16 elements/SM/cy from packed 16-lane MUFU? — no, MUFU is 32-lane warp instruction)
- (c) MUFU pipe is per-SMSP (4 units/SM) and the "16" comes from 4 SMSPs × 4 cy/op = 16 cy/op? But that's not "SASS/SM/cy"

**Most likely**: catalog §4 row is **WRONG** (16× overstated). The DENSE numbers + §17 audit are consistent at ~1 SASS/SM/cy.

### F2I, POPC, BREV, FLO = 16 ⚠ SAME AMBIGUITY AS MUFU
- These are pipe_xu (simple) per catalog. Same rate cap as MUFU.EX2 = ~1 SASS/SM/cy per audit.
- Catalog "16" likely also wrong by 16×.

### BAR.SYNC = ~12 ⚠ NOT INDEPENDENTLY VERIFIED
- `24_latency_table.md` measured `__syncthreads` = 22 + 2W cy (54 cy at BS=512). Throughput is bounded by sync latency, not "12 SASS/SM/cy".
- Catalog "~12" is suggestive but not directly comparable to my measurement methodology.

### FP64 DFMA = 1.6 ✅ APPROXIMATELY CONFIRMED
- DENSE §1: pipe_fp64 = 0.05 cap. 0.05 × 32 lanes ≈ 1.6 SASS/SM/cy. Matches catalog.
- `24_latency_table.md`: DFMA latency 63.7 cy. With 1.6 SASS/SM/cy, ILP ~64 needed to saturate.

## VERDICT

⚠ **MIXED** — most rows correct, but **MUFU and pipe_xu rows (16 SASS/SM/cy) are likely wrong by 16×**. The actual saturation rate per audit + DENSE is ~1 SASS/SM/cy.

## REVIEW_CHECKLIST candidates

- [ ] **§4 catalog L504**: MUFU = "~16 SASS/SM/cy" — likely **off by 16×**. Audit measurements (`17_mufu.md`) and DENSE §1 (pipe_xu cap=1.0) both indicate ~1 SASS/SM/cy. — `[unit-confusion + inconsistent-with-other-catalog-section]`
- [ ] **§4 catalog L505**: F2I/POPC/BREV/FLO = 16 — same ambiguity as MUFU; also pipe_xu, also probably ~1 SASS/SM/cy not 16
- [ ] **§4 catalog L489**: u32 IADD "128" — only achievable via IADD3+IMAD.IADD alternation pattern; pure IADD3 caps at 64. Catalog row should distinguish.
- [ ] **§4 catalog L490**: u64 ADD "64 (1 alu + 1 fmaH per op)" — needs verification that the 2-SASS u64 add actually achieves 64 warp-inst/SM/cy when both pipes are otherwise idle
