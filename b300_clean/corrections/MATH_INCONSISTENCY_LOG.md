# MATH_INCONSISTENCY_LOG — MUFU / SHFL / REDUX

Cross-file inconsistencies detected 2026-04-22.

## Inconsistency #1 — EX2 speedup factor

| File | Claim |
|---|---|
| `14_math_intrinsics.md` sec 1 line 29 | EX2 "1.7x faster ~8.1 TGOps/s" |
| `V41_V48_FINDINGS.md` lines 30, 80 | EX2 9.22 vs others 4.74 Gops/s = **2.0x** |

V41 is the more recent, isolated, 10-rule-rigor measurement. Adopt 2.0x.
14_math sec 1 needs a footnote update.

## Inconsistency #2 — REDUX vs SHFL ratio

| File | Claim |
|---|---|
| User MEMORY (V4 loop session) | "redux.sync.min/max **4x** SHFL" |
| `V8_SHFL_PEAK.md` line 30 | quotes the **4x** as "V4 prior findings" |
| `Q3_WARP_REDUCE_RECIPES.md` | REDUX.SUM 11.61 cy vs 5-step SHFL 27.19 cy = **2.34x** algorithm-level |
| `V41_V48_FINDINGS.md` lines 64-67 | REDUX 9.09 + SHFL 9.48 Telements/s = **equal raw rate** |

Resolution: the "4x" figure has no source file; the real measurements give
2.34x algorithm-level and 1.0x raw per-instruction. Retract "4x".

## Inconsistency #3 — XU peak framing

| File | Claim |
|---|---|
| `V8_MUFU_PEAK.md` | rsqrt 47.8 G MUFU/s at 99.5% XU pipe util |
| `M16_V9_FULL_SYNTHESIS.md` table I | "XU peak: 47.8 GMUFU/s @ 99.5%" |
| `V41_V48_FINDINGS.md` | non-EX2 MUFU 4.74 Gops/s/chip = 4740 G MUFU/s |
| `14_math_intrinsics.md` sec 1 | "MUFU peak ~4.8 TGOps/s" (matches V41) |

V8 reports a 1-chain self-dep rsqrt regime — it is latency-bound, not the
true XU peak. M16 quotes V8 as the canonical XU number, which is ~100x too
low if read as a saturated peak. Reframe M16 row XU as "1-chain rsqrt
latency-bound 47.8 G; saturated MUFU 4.74 G chip-Gops/s". Keep 4.74 G as the
canonical pipe peak (matches V41 and 14_math).

## Inconsistency #4 — Per-SM MUFU variance

| File | Claim |
|---|---|
| `14_math_intrinsics.md` sec 7 | exp2f 34.9, log/sqrt/rsqrt 22.5, sin/cos 20.6 Gops/s/SM |
| `V41_V48_FINDINGS.md` line 30 | All non-EX2 MUFU **equal** at 4.74 Gops/s/chip (= 32 G/s/SM) |

These do not agree: 14_math sec 7 has rsqrt 22.5 != sin 20.6 != log 22.5,
while V41 says they are all equal. V41 had the more rigorous methodology
(SoL %, single sweep). Mark sec 7 LOW confidence pending re-test.

## Inconsistency #5 — `__frsqrt_rn` "fastest" vs "all equal"

`14_math_intrinsics.md` sec 3 cites `bench_mufu_audit` showing `__frsqrt_rn`
is the fastest MUFU at "727 inst/ns at 8 chains, 2.69x faster than `rsqrtf`".
V41 contradicts: rsqrt sits in the 4.74 Gops/s tier with the rest.

The "2.69x faster" is `__frsqrt_rn` (approximate, 1 MUFU.RSQ inst) vs
`rsqrtf` standard (= MUFU.RSQ + 7 NR refinement FMAs, latency-bound when
chained). It is a `.approx` vs `.rn` comparison, NOT a "rsqrt is faster
than other MUFU" claim. Sec 3 needs clarifying language.

## Inconsistency #6 — sqrt/div anomalies (also semi-resolved)

`14_math_intrinsics.md` sec 2 reports sqrtf 687 Gops/s and 1/x 492 Gops/s
which look like outliers. Sec 3 already explains: nvcc default (no
fast-math) emits `sqrt.rn.f32` (138 cy) and `div.rn.f32` (243 cy), so these
are latency-bound, not pipe-bound. Already documented in same file but
worth flagging that other docs may quote these without the caveat.

## Suggested actions

1. Edit `14_math_intrinsics.md` sec 1 footnote: change "1.7x" -> "2x" with V41 cite.
2. Edit `V8_SHFL_PEAK.md` line 30: replace "4x SHFL" with "2.34x algorithm-level (Q3); raw rate equal (V37/V38)".
3. Edit `M16_V9_FULL_SYNTHESIS.md` table I XU row: add "(1-chain rsqrt; saturated 4.74 G)".
4. Edit `14_math_intrinsics.md` sec 7: add LOW-confidence stamp; reference V41.
5. Update user MEMORY entry "redux.sync.min/max 4x SHFL" to "2.34x".
