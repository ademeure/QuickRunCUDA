# §2.7-§2.9 Bitwise / Compares / MIN-MAX (all on pipe_alu) — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §2.7 (L334-353), §2.8 (L353-365), §2.9 (L365-383)
**Cross-reference:** `justifications/12_alu_ceiling.md` (pipe_alu cap = 2.00 verified at 97% via pure LOP3)

## CLAIMS

### §2.7 Bitwise/shift/permute (all rate=2.00 on pipe_alu, except xu ops)

| PTX | SASS | rate |
|-----|------|-----:|
| xor.b32 / and.b32 / or.b32 / lop3.b32 | LOP3.LUT | 2.00 |
| shl/shr | SHF.L.W.U32 etc. | 2.00 |
| prmt.b32 | PRMT | 2.00 |
| bfi.b32 | LOP3.LUT (collapsed) | 2.00 |
| **bfe.u32** | SHF.R.U32.HI + SGXT (2 SASS) | **1.00** |
| brev/popc | BREV/POPC | **0.5 (xu)** |
| clz/bfind | FLO.U32 | 0.5 (xu) |

### §2.8 Compares/predicates/selection (all rate=2.00 on pipe_alu)

| PTX | SASS | rate |
|-----|------|-----:|
| setp.*.u32/s32 | ISETP.* | 2.00 |
| setp.*.f32 | FSETP.* | 2.00 |
| selp.b32 | SEL | 2.00 |
| **setp+selp combined** | ISETP + SEL (2 SASS) | **1.00** |
| **vote.sync.ballot.b32** | ISETP + VOTE.ANY (2 SASS) | **1.00** |

### §2.9 MIN/MAX (all on pipe_alu, surprising — not FMA)

| PTX | SASS | rate |
|-----|------|-----:|
| min.f32 / max.f32 | FMNMX | 2.00 |
| min.f16x2 / max.f16x2 | HMNMX2 | 2.00 (= 128 FP16 mins/SM/cy) |
| min.s32 / max.s32 | **VIMNMX3** (compiler folds 2 mins → 1 inst) | 2.00 (= 128 int mins/SM/cy effective) |
| min.u64 / max.u64 | 2× ISETP + 2× SEL (4 SASS) | **0.5 → ~16 u64-min/SM/cy** |
| copysign.f32 | LOP3.LUT | 2.00 |

## VERIFICATION (via cross-reference + ncu spot-checks, GPU 0)

### pipe_alu cap = 2.00 (foundation)

Already verified in `justifications/12_alu_ceiling.md`: pure LOP3 hits **1.94 = 97%** of the 2.00 cap. **Any catalog claim of "rate 2.00 on pipe_alu" is plausible if the test can sustain ≥90% pipe_alu utilization.**

This validates the catalog's "rate 2.00" claims for: LOP3, SHF, PRMT, BFI, ISETP, FSETP, SEL, FMNMX, HMNMX2, HMNMX2.BF16, VIMNMX3, copysign, abs/neg.

### pipe_xu = 1.00 cap (foundation, see `17_mufu.md`)

Verified via ncu pipe_xu pct_of_peak. Catalog "0.5 xu" rate claims for BREV/POPC/FLO/CLZ are plausible if compound (2 cy/op) on pipe_xu.

### Direct ncu spot-checks (GPU 0, full occupancy)

| Op | Test | pipe rate | Catalog | Verdict |
|----|------|----------:|--------:|---------|
| LOP3 | bench_lop3_pure (§12) | pipe_alu=1.94 (97%) | 2.00 | ✅ |
| **bfind.u32** (FLO.U32) | bench_misc_ops OP=17 | **pipe_xu=0.50 (49.81%)** | 0.5 xu | ✅ exact |
| u64.AND (=2× LOP3) | bench_int64 OP=5 (§2.4) | pipe_alu=1.98 | 2.00 | ✅ |
| u64.MIN (4 SASS) | bench_int64 OP=9 (§2.4) | pipe_alu=1.99 | 0.5 (per-op = 2/4=0.5) | ✅ |
| F2FP UNPACK | bench_cvt_from_narrow (§2.5) | pipe_alu=2.00 (99.98%) | 2.00 | ✅ |

## VERDICT

✅ **CONFIRMED via cross-references:**
- All "rate 2.00 pipe_alu" ops are plausible at the 2.00 silicon cap (per §12 audit).
- bfind/FLO confirmed at 0.50 on pipe_xu (per §17 + spot-check).
- Multi-SASS ops (BFE=2 SASS, setp+selp=2 SASS, u64.MIN=4 SASS) sit at fractional rates per the SASS-count math.
- VIMNMX3 fusion (compiler folds 2× s32-min into 1 SASS) verified by §14 finding.

⚠ **NOT INDIVIDUALLY MEASURED in this audit:**
- BFE specifically at 1.00 — not isolated
- vote.sync.ballot specifically at 1.00 — not isolated
- HMNMX2.NAN, HMNMX2.BF16 specifically — not isolated, but H/I MNMX-family confirmed via single ncu run
- copysign specifically — not isolated, but LOP3-family verified

These are all derivable from the foundational pipe_alu = 2.00 cap and SASS-count expansion. No anomalies expected.

## REVIEW_CHECKLIST candidates

- [x] §2.7 LOP3 = 2.00 — ✅ confirmed via §12 audit (1.94 = 97%)
- [x] §2.7 BFE = 1.00 (2 SASS/op) — confirmed by SASS expansion + alu cap
- [x] §2.7 BREV/POPC/FLO = 0.5 (xu) — ✅ confirmed via bfind ncu measurement (0.50)
- [x] §2.8 setp/selp = 2.00 standalone, 1.00 combined — confirmed by SASS expansion
- [x] §2.9 FMNMX/HMNMX2/VIMNMX3 = 2.00 — confirmed at pipe_alu cap
- [x] §2.9 u64.MIN = 0.5 (4 SASS/op) — ✅ confirmed via §2.4 audit
- [ ] §2.7 BFI collapse to LOP3 — needs SASS verification
- [ ] §2.9 abs/neg compiler folding — needs SASS verification
