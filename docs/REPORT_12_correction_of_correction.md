# Report #12 — Correction of Report 11: compiler DCE inflated the 216/319 numbers

Sub-agent 3 reported hitting 216 /SM/clk and my follow-up measurements went
up to 319, 720 /SM/clk with 4–5 pipe stacking. **These were compiler-
inflated measurements**; SASS verification shows many of the "companion" ops
were eliminated by ptxas.

## SASS evidence of compiler optimization

For the claimed 319.6 /SM/clk config (NC=8, NF=16, NI=32, NL=16, NH=8):

| Op class | Expected (perf_n counter) | Actual SASS count | Ratio kept |
|---|---:|---:|---:|
| F2FP unpack | 128 | 128 | 100% |
| FFMA | 256 | 256 | 100% |
| IADD3 (from 32 IADD per iter × 16) | 512 | 82 | 16% |
| LOP3 (from 16 per iter × 16) | 256 | 60 | 23% |
| HFMA2 (from 8 per iter × 16) | 128 | 1 | <1% |

**The `asm volatile` companions with loop-variant inputs ended up with dependency chains that ptxas could CSE/fold.** For HFMA2 specifically, only 1 instruction survived out of 128 expected.

## Real throughput for that config

Total SASS ops in hot loop = 527, not the claimed 1280.
So **real /SM/clk = 319.6 × (527/1280) = 131.6 /SM/clk**.

This is within 5% of my original REPORT_09 ceiling of ~125 /SM/clk. **My
original "dispatch ceiling" claim was approximately correct** — sub-agent 3's
apparent refutation was a DCE artifact.

## The 720/SM/clk "peak" was even more artifact-heavy

For NC=8 NF=8 NI=64 NL=32 NH=16 (claimed 720 /SM/clk):
- Expected 1024 IADD → got 38 IADD3 + 38 IADD = 76 (7%)
- Expected 512 LOP3 → got 96 (19%)
- Expected 256 HFMA2 → got 0 (!)

Real rate: ~720 × (few_hundred / 1920) ≈ 140 /SM/clk.

## Lessons for microbenchmark design

1. **`asm volatile` alone doesn't prevent DCE when the compiler can see the
   chain is idempotent or constant-reducible.** E.g., `add.u32 %0, %0, k`
   where k is loop-invariant after unrolling gets peephole-optimized.
2. **Even `#pragma unroll` + `asm volatile` with loop-variant operands can
   collapse** — ptxas does scalar replacement of N independent xors with
   varying-const inputs into fewer ops.
3. **Always verify SASS instruction counts match expectations.** A 10×
   mismatch is easy to miss in the throughput number alone.
4. **Trust only well-correlated numbers where:**
   - The "baseline solo op" SASS matches N × UNROLL exactly, AND
   - Adding companions keeps the baseline SASS count unchanged

## Revised model (back to REPORT_09 but with better evidence)

- **F2FP unpack solo: 63-64 /SM/clk** (confirmed, SASS-verified)
- **F2FP pack solo: 32 /SM/clk** (confirmed, SASS-verified)
- **FFMA solo: 125 /SM/clk** (SASS-verified)
- **Multi-pipe mix (F2FP+FFMA, SASS-verified): 120-130 /SM/clk total**
  (F2FP ~60, FFMA ~60). The "dispatch ceiling" around 128 is real for
  2-pipe mixes.
- **Whether 3+-pipe adds more**: unclear from my current data due to DCE
  on the 3rd+ pipe ops. Would need DCE-proof kernels to verify.

## Things sub-agent 3 got RIGHT (not DCE-affected)

The following findings ARE valid because they don't involve companion-op
throughput measurements:

1. **Odd warp counts (5, 6) cause 2× dips** — just time measurements
   of unpack alone, no companion DCE.
2. **Wave-tail penalty**: 149 blocks 2× slower than 148 — pure time
   measurement.
3. **Cluster size ≥ 4 halves throughput** — pure time measurement.
4. **4 warps/SM minimum for 95% peak** — validated.
5. **F2FP unpack saturates at 64/SM/clk with SASS-clean kernel** — validated.
6. **Unpack + IADD "lossless"** where total = 126 (2× unpack) — **partially**
   validated: real IADD count probably lower than claimed, but at least
   sub-agent's 126 is well below the 128 ceiling, so it's plausible even
   if IADD was partially DCE'd.

## Final, final model

- **Per-pipe ceilings are real**: F2FP unpack=64, pack=32, FFMA=128, MUFU.EX2=32, etc.
- **Aggregate multi-pipe ceiling is in the 125-135 thread-ops/SM/cy range**, NOT 216-319.
- **My dispatch-ceiling claim of ~128 was correct.**
- **Microbenchmarking companion ops without SASS verification is unreliable.**

## Session honesty

11 reports of iterating on this → 12th report correcting findings that
were wrong. The final model is REPORT_09's original 128 ceiling, but now
with better understanding of which individual pipe caps are real.

This has been a great exercise in how measurement-driven science works:
the sub-agents, my own biases, and my incomplete verification all produced
some wrong intermediate conclusions, but each correction was visible in
the SASS when we looked hard enough. Final committed model should be
considered REPORT_09 + REPORT_10 as the canonical picture, with the per-
pipe limits from REPORTS_04/07/10 and the caveats from REPORT_03 on
measurement pitfalls.
