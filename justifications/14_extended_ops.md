# §14 Extended op catalog — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §14 (L971-1008)
**Test:** `tests/bench_misc_ops.cu`

## CLAIMS (catalog L971-1008)

Many small claims; verifying the load-bearing ones.

## DIRECTLY VERIFIED ROWS

### FMNMX3 fusion ✅ CONFIRMED — Blackwell 3-input FP min/max

`min.f32 %0, %0, %1; min.f32 %0, %0, %2;` (2 chained mins)
- Catalog: compiler fuses to single **FMNMX3** SASS, rate 2.00 alu = **128 logical mins/SM/cy**

Test: `bench_misc_ops.cu OP=6` (single warp + chains)

```
CUDA_VISIBLE_DEVICES=0 ncu --metrics sm__inst_executed_pipe_alu.avg.per_cycle_active,sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active \
  ./QuickRunCUDA -f tests/bench_misc_ops.cu -t 512 -b 296 -A 1024 -B 1024 -C 1024 \
  -s 8192 -H "#define OP 6
#define UNROLL 16
#define N_CHAINS 8" -0 1024 -T 1
```

Result:
- **128 FMNMX3** SASS instructions emitted (verified via grep)
- pipe_alu = **1.97 (98.54%)** of cap
- **Effective rate: 128 logical mins/SM/cy** (64 FMNMX3 × 2 mins each) ✓

This is a REAL architectural feature on Blackwell — 3-input FP min/max in one instruction. Same trick as IADD3 for integer.

### bfind.u32 (FLO.U32) ✅ CONFIRMED at 0.50 xu

Already verified in `02_7_8_9_alu_ops.md` and `02_6_other_cvts.md`: 0.50 inst/SM/cy = 49.81% of pipe_xu.

### ATOMS family — confirmed via §15 audit

- ATOMS.MIN/EXCH = 1.00 ✓ (per `15_atomics.md`)
- ATOMS.CAS = 0.50 ✓ (per `15_atomics.md`: 126 cy = consistent with 0.50/SM/cy)

## CROSS-REFERENCED ROWS (existing audits)

| §14 row | Audit | Verdict |
|---------|-------|---------|
| FFMA w/ immediate (catalog: folds to FFMA, no FFMA32I) | `02_1_2_3_fp32_int.md` | ✅ FFMA at 4.00 cap confirmed; SASS shows FFMA not FFMA32I |
| FFMA.FTZ (catalog: free) | `00b_mem_hierarchy.md` notes "-use_fast_math forces FTZ" | ✅ FTZ universal in NVRTC harness |
| ATOMS.MIN/EXCH/CAS rates | `15_atomics.md` + `22_atomic_smem_DEEP.md` | ✅ |
| LDGSTS.E (cp.async) | `tests/bench_cctl_cpasync.cu` MODE 0/1/2 | ✅ ~50% pipe_lsu when issued |
| `nanosleep` SASS = NANOSLEEP | catalog claim | ⚠ rate 0.25 adu — by-design stall, can't measure utilization that way |

## NOT INDIVIDUALLY MEASURED (preserved from catalog)

- **vabsdiff.s32 → PRMT + SHF compiler path** — plausible per catalog; not re-tested
- **mov.u32 %ctaid.x = S2R cached once** — compiler-emission detail, not a perf claim
- **mov.u32 %nctaid.x → LDCU uniform 0.25** — uniform-pipe access, plausible
- **prefetch.global.L1/L2 = CCTL.E.PF1/PF2 "very slow"** — claim 255 ms for 128 prefetches; not re-tested but consistent with serialization-against-memory-system narrative

## VERDICT

✅ **CONFIRMED for major architectural features:**
- **FMNMX3 fusion** is real (compiler fuses 2× min.f32 into single FMNMX3 SASS) — load-bearing
- **bfind/FLO.U32 = 0.50 xu** confirmed
- **ATOMS family rates** confirmed via §15

🟡 **MOSTLY PRESERVED** for compiler-emission claims (immediate variants, S2R caching, etc.) — these are SASS observations rather than perf measurements; catalog statements plausible.

## REVIEW_CHECKLIST candidates

- [x] §14 FMNMX3 fusion = 128 logical mins/SM/cy — ✅ CONFIRMED via SASS + pipe_alu measurement
- [x] §14 ATOMS.{MIN,MAX,EXCH}=1.00, ATOMS.CAS=0.50 — ✅ confirmed via §15
- [x] §14 bfind/FLO.U32 = 0.50 xu — ✅ confirmed
- [ ] §14 prefetch.global.L1/L2 "very slow" (255 ms / 128) — not re-tested but plausible
- [ ] §14 nanosleep = 0.25 adu — by-design stall; rate hard to measure via utilization
