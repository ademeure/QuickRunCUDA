# M-Synthesis Corrections (M1 - M16)

Cross-check of layered synthesis docs M1 - M16 against newer V32-V51 work,
B300_TRUE_REFERENCE, peer category corrections (cache/power/math/sync), and
DSMEM_REFERENCE. Originals NOT modified.

Each section: per-file claim list - verdict - retraction or addendum.
`## RETRACTIONS` = numbers to stop citing; `## UNRESOLVED` = open questions.

---

## M1 - V4_DEEP_DIVE_INDEX

### Claims affected
- B1: "FFMA+IADD3 only 14.2% overlap" - row in pipe-overlap table.
- B3: "MUFU+FFMA = ~100% overlap up to 4 FFMA per MUFU".
- D6: ".reuse cache = 3rd port; pure 3-source FFMA caps 65%" - HIGH.
- I8: cluster topology, MAX=8 (CSIZE=16 silently fails) - listed "verified 8 / per spec 16".

### Status vs newer work
- B1 14.2% is consistent with V49 measurement of 54% overlap factor for FFMA+IADD3
  (V49 commit 501134a) **only when normalized differently**: V49 frames it as
  "55% of perfect parallelism", V40 as "17%". Both agree the pair is NOT free.
  M1's B1 row should be reframed: same direction, different denominator.
- B3 MUFU+FFMA "100%" supported by V40 (FFMA+MUFU 100%+) but V41 reframes:
  MUFU EX2 alone hits 9.22 Gops/s (only 2 ; faster); other MUFU at 4.74 Gops/s.
  M1's "MUFU 100%" is true ONLY because MUFU's 1/(4cy) issue rate slots into FFMA
  bubbles, not because of true dual-issue.
- I8: cluster MAX = 8 portable / 16 non-portable matches B300_TRUE_REFERENCE.

### RETRACTIONS
- None numerically; reinterpret B1/B3 framings (see Addenda).

### Addenda
- Add note that V49/V50 quantify the "issue port shared" finding: 54-55% same-warp,
  74% warp-specialized for FFMA+ALU pairs. M1 B1 implies "almost no overlap";
  V49 reveals it is partial (not zero, not full).

---

## M2 - ENERGY_LADDER

### Claims affected
- "GPU true idle 164.7 W" - implicit 1500 MHz lock.
- "FFMA = 4.4 pJ/FFMA at 1500 MHz" baseline.
- "Static 0.05 W/SM, dynamic 0.4 W/SM (8-9; ratio)".

### Status
- Per POWER_INCONSISTENCY_LOG #A: idle is 144 W @ 510 MHz, 167 W @ 1500 MHz, 197 W
  @ boost. M2's 164.7 W is correct AT 1500 MHz but the file does not consistently
  flag the clock context. Memory note "stuck-at-1005" mode is also missing.
- 4.4 pJ/FFMA is consistent with V5 D2; V6 C1 sweep adds clock dependence
  (3.1 - 9.96 pJ/FFMA across 510 - 1920 MHz).
- Static/dynamic split conflicts with M11 (0.7 W/SM dynamic) - per POWER #H, both
  are different operating points, not a contradiction.

### RETRACTIONS
- "0.05 W/SM static" is one configuration; should not be cited as universal.

### UNRESOLVED
- M2's per-pipe LDG=+177 W (1500 MHz) vs newer popcount data showing DRAM swing
  is 240 - 367 W (data-dependent). M2 LDG number is data-pattern naive.

---

## M3_REVERIFY_LOG

### Status
This is a verification log, not a claims doc. Several entries it RETRACTED have been
re-confirmed retracted by V32-V51 work (e.g. CCTL.IVALL absent, persistent L2 useless).
Two entries need updating:
- "FP4 9856 TFLOPS still MED" - now superseded by B300_TRUE_REFERENCE NVFP4 row
  showing single-GPU NVFP4 wide-N = 10297 TFLOPS = 103% of 10 PFLOPS spec
  (HIGH, commit 3628a40).
- "L2 BW peak 17 TB/s claim corrected" - the corrected number (23.85 TB/s
  kernel-effective, 13.30 TB/s wire) is consistent.

### Addenda
Add forward-pointer to V32-V51 and to corrections/CACHES_INCONSISTENCY_LOG.

## M3_TOPOLOGY_CHEATSHEET

### Claims affected
- Cluster MAX=8 verified / 16 spec.
- F4 cluster.barrier 395 cy floor.
- L7 launch 488K kernels/sec single-thread.

### Status
- Cluster max numbers consistent with B300_TRUE_REFERENCE ("max usable = 16
  non-portable, 8 portable"). M3 is fine.
- Cluster barrier 395 cy matches V5 C2 (390 cy O(1)).
- 488K kernels/sec is consistent with M13 V8 launch data.

### RETRACTIONS
- None.

---

## M4 - COMPILER_CHEATSHEET

### Claims affected
- "-use_fast_math 3.28; faster + 4; lower energy".
- ".cg = 288 cy bypass L1, all others 38 cy".
- Predicated cluster A vs B costs.

### Status
- fast_math: matches memory note (NVRTC harness uses fast_math/FTZ; can't measure
  subnormal handling). Add caveat that this also affects MUFU paths.
- LDG cost rows are consistent with 03_caches.

### RETRACTIONS
- None.

---

## M5 - MEMORY_CHEATSHEET

### Claims affected
- L1 ~46 TB/s peak.
- L2 single-SM 23 TB/s; L2 hit 152 ns / 230 cy.
- HBM3E read 7.30 TB/s; write 7.080 GB/s "actual DRAM" (after 8; sector amp).
- TMA bulk read = LDG read within 0.3% (catalog c40c016).

### Status vs newer work
- L1 46 TB/s vs V8 measurement 30.5 TB/s: per CACHES_INCONSISTENCY_LOG #L1 BW,
  this is a 1.5; spread. Resolution: 30.5 TB/s = strided default-ld (HIGH);
  46 TB/s = older ILP-maxed upper bound (MED). M5's 46 TB/s should be flagged MED.
- L2 23 TB/s consistent with B300_TRUE_REFERENCE if labelled "kernel-effective".
- HBM read 7.30 TB/s consistent.
- TMA read = LDG within 0.3% is REFUTED by V46: TMA pipelined 8-deep reaches
  7.20 TB/s = 98.5% HBM. V33 single-deep TMA was only 6.72 TB/s (similar to LDG),
  so the "within 0.3%" claim was true only for the single-deep regime.

### RETRACTIONS
- "L1 = 46 TB/s peak" without ILP qualification - cite 30.5 TB/s as conservative
  measured peak (V8); 46 TB/s as ILP-max upper bound only.
- "TMA bulk read = LDG read within 0.3%" - true only for single-deep TMA. Pipelined
  TMA (V46) is 7% faster than LDG SoL.

### Addenda
- Add cp.async.ca = 6.91 TB/s (V9 finding; 19% better than plain LDG 5.82 TB/s).
- Note V44/V45: SMEM bank conflicts much cheaper than textbook in throughput regime.

---

## M6 - LLM_KERNEL_SOL

### Claims affected
- Argmax 441 cy / RMSNorm 588 cy / SoftMax 899 cy / LayerNorm 997 cy / Top-K 1858 cy.
- "1024 elements per token" tile size baseline.
- Per-layer estimates for LLaMA-7B at D=4096.

### Status
- Single-block primitive numbers are M6-internal benchmarks; no V32-V51 supersession.
- The wider claims (e.g. "fastest kernel = argmax 441 cy") still hold.
- Memory note "TRUE perf at 2032 MHz: 40 tok/s 70B, 345 tok/s 8B" is real-system
  per token rate, NOT comparable to per-block primitive cy. Different scope.

### RETRACTIONS
- None.

---

## M7 - V5_SYNTHESIS

### Claims affected
- F1: "Two FFMA sub-pipes (fmaheavy 50 + fmalite 50)".
- B6 HMMA + LDS 73% overlap.
- Pipe-overlap matrix lines 168-178.
- D2 4.4 pJ/FFMA.

### Status vs newer work
- F1 fmaheavy/fmalite split: per A6_PER_PIPE_REFERENCE this is an ncu pipe partition,
  not architectural sub-pipes. Wording "two FFMA sub-pipes" is misleading.
- HMMA + LDS 73% overlap is M8's row; M8 is the canonical place for this matrix.
- Last paragraph "Pipe overlap matrix (single warp, B300)" is incomplete vs M8.

### RETRACTIONS
- Reword "Two FFMA sub-pipes" - they share the FMA pipe; the 50/50 ncu split is a
  metric attribution detail, not a hardware duality.

---

## M8 - PIPE_OVERLAP_MATRIX (CRITICAL FILE)

### Claims affected
- FFMA + LDS 96% overlap.
- FFMA + IADD3 56% overlap.
- HMMA + FFMA 31%, HMMA + IADD3 32%, HMMA + HMMA 69%.
- "FFMA + MUFU = 100%+ super-linear (fills bubbles)".
- "Async-queue ops (LSU) overlap 96%+ with anything".

### Status vs newer work
**This matrix is the most outdated piece in the synthesis stack:**
- V49 (501134a) measured FFMA + LOP3 = 55%, FFMA + IADD3 = 54%, FFMA + PRMT = 51%.
  M8's "FFMA + IADD3 = 56%" matches V49 (54%) - GOOD.
- V49 EXPLICITLY refutes M8's framing of "category C issue-port-bound". V49 quantifies
  it: "warp scheduler dispatch slot is shared (4 inst/cy/SM total). Even different
  pipes can't both issue 1/cy/SMSP simultaneously."
- V50 (fbe1c18) quantifies warp-specialized variant: 74% efficient (vs 55% same-warp).
  M8 has NO row for warp-specialized version.
- V40 (d1d09c5) measured ALU pipe ladder confirming FMA pipe (FFMA/FADD/IADD3) is
  dominant tier; LOP3/IMUL are HALF rate (INT-bit pipe). M8's IADD3 placement
  in "ALU" category is wrong - IADD3 actually runs on the FMA pipe (V40).

### RETRACTIONS
- "MUFU + FFMA = 100%+" framing is misleading. True statement: MUFU has 1/(4cy)
  issue rate, so FFMA fills its bubbles; this is NOT a free-extra-pipe finding,
  it is FFMA absorbing MUFU dispatch gaps. V49 establishes the issue-port-shared
  model that M8's category B partially obscured.
- M8 column "ALU (IADD3)" mis-attributes pipe ownership. IADD3 runs on FMA pipe
  per V40 - the IADD3 row should be merged with FMA, and "true ALU" rows
  (LOP3/IMUL/PRMT) should be separate.

### Addenda
- Add row: warp-specialized FFMA + LOP3 = 74% (V50).
- Add caveat: numbers are SAME-WARP overlap; cross-warp / warp-spec gives different
  results (V50).

---

## M9 - ENERGY_PARETO

### Claims affected
- Pure FFMA min-energy = 510 MHz.
- Memory-bound min-energy = 800 MHz.
- Mixed ML min-energy = 1992 MHz boost (3.08; lower than 510).
- "Static power on B300 ;= 165 W (idle baseline)".

### Status
- "ML inference USE BOOST CLOCK (3.08;)" CONFIRMED by user memory note
  (project_b300_v6_complete.md): "ML inference USE BOOST CLOCK (3; lower energy
  than 510)". GOOD.
- Per POWER #I: M9's per-task-energy framing is not contradicted by V10's
  GFLOPS/W framing - they are different metrics. M9 stands.
- "Static 165 W" is at 1500 MHz; per POWER #A, varies 144-198 W with clock.

### RETRACTIONS
- None substantive.

### Addenda
- Add clock-context to all idle/static numbers.

---

## M10 - V6_SYNTHESIS

### Claims affected
- Headline 1: pipe overlap matrix (same as M8).
- Headline 2: workload-dependent min-energy clock.
- Headline 6: cudaGraph patterns; "ExecUpdate 4-16; faster than re-instantiate".
- Architectural facts: "L2 = 126 MB", "WGMMA REMOVED on B300".

### Status
- All carry through M8/M9 issues. The min-energy and L2=126 MB are CONFIRMED.
- WGMMA removal is reconfirmed in V5 B4 and B300_TRUE_REFERENCE.
- ExecUpdate 25; faster confirmed in B300_TRUE_REFERENCE row 105.

### RETRACTIONS
- M10 inherits M8's pipe-overlap framing issues. Treat with same caveats.

---

## M11 - PER_PIPE_ENERGY

### Claims affected
- "FFMA = 4.4 pJ" (1500 MHz).
- "Static GPU 165-170 W regardless of utilization".
- "FFMA-bound 359 W -> 39.7 TFLOPS = 9.0 J/TFLOP" (= 0.111 TF/W).
- "All-148-blocks at 552 W under FFMA-saturated boost".

### Status vs newer work
- Per POWER_INCONSISTENCY_LOG #J: M11 says FFMA = 0.111 TF/W; 16_power_clock says
  0.21 TF/W (74.6 TF / 361 W). Roughly 2; discrepancy.
- M11's 39.7 TFLOPS is below the V8/B300_TRUE_REFERENCE FFMA peak of 74.62 TFLOPS
  (96.92% of theoretical, commit 06b0d8d). M11's number is half-peak, suggesting
  low ILP or single-warp test - mis-labelled as "FFMA-bound" peak.
- "Static 165-170 W regardless of utilization" CONTRADICTS POWER #A: idle scales
  with clock 144-198 W.
- "552 W FFMA-saturated boost" matches B300_TRUE_REFERENCE pattern (FFMA peak
  numbers were measured at boost clock).

### RETRACTIONS
- "FFMA pJ/op = 9.0 J/TFLOP = 0.111 TF/W" - this is HALF-PEAK kernel data, NOT the
  peak. Use 0.21 TF/W (16_power_clock) or 137 GFLOPS/W (V8) for FFMA peak.
- "Static GPU 165-170 W regardless of utilization" - true only at fixed clock; idle
  varies 144-198 W across clock sweep.

### UNRESOLVED
- The 2; discrepancy in TF/W between M11 and 16_power_clock is unresolved
  (per POWER #J). Likely M11 measured a lower-occupancy regime; needs explicit
  re-derivation at full occupancy + 1500 MHz.

---

## M12 - V7_SYNTHESIS

### Claims affected
- cp.async stack 3-4;.
- cuStreamWriteValue 0.45 us.
- Persistent batched 38 ns/task.
- "boost 2032 MHz holds at 552 W (no throttle)".

### Status
- cp.async stack confirmed (V7 + V46 pipelining).
- cuStreamWriteValue numbers consistent with V8 0.11 us/op BatchMemOp (better still).
- 552 W no-throttle is for FFMA only; under random tcgen05.mma, hits 1100 W cap
  (per POWER #F).

### RETRACTIONS
- "boost holds at 552 W (no throttle)" must be qualified: TRUE for FFMA, FALSE for
  random-data tcgen05.mma where 1100 W cap throttles to 1057 MHz.

---

## M13 - V8_SYNTHESIS

### Claims affected
- "Queue depth = 1024 per stream (USER-CORRECTED)" - this is itself a correction
  to V8 M7 initial.
- "L2 = 126 MB" architectural fact.
- "DRAM tail latency spike: max tail 1433 cy = 955 ns (23; avg)".
- "2:4 sparse HMMA = 2; throughput native".

### Status
- Queue depth, L2 size, sparse HMMA all confirmed.
- DRAM tail spike is consistent.
- M13's headline #5 "Cross-process zero-copy stack COMPLETE" matches user memory.

### RETRACTIONS
- None.

---

## M14 - V8_SOL_LADDER (CRITICAL FILE)

### Claims affected (compute peaks at boost 2032 MHz)
- FP32 FFMA: theoretical 76.97 TF, measured 75.2 TF (97.64%).
- FP64 DFMA: 1.203 TF (100%).
- HMMA.F16/BF16: 578.6 TF (99.90%).
- "MUFU rsqrt: 47.8 GMUFU/s (99.49% XU)".

### Memory peaks
- L1 30.5 TB/s (~100%).
- L2 13.85 TB/s (66%).
- SHMEM LDS 26.9 TB/s (74%).
- HBM read 5.82 TB/s (81%); HBM write 6.11 TB/s (85%); TMA write 7.57 TB/s (95%).

### Status vs newer work
- FP32 75.2 TF matches B300_TRUE_REFERENCE (74.62 TF, 96.92%). GOOD.
- HMMA 578.6 TF is HIGHER than B300_TRUE_REFERENCE entry (569 TF). The "1543"
  single-chain claim was already RETRACTED in B300_TRUE_REFERENCE row 58. The
  578.6 figure is an upper-bound; 569 is the canonical 8-chain catalog burst.
- HBM read 5.82 TB/s is BELOW V46's 7.20 TB/s (98.5% HBM, TMA 8-deep pipelined).
  M14's 5.82 TB/s reflects plain LDG; should add row for TMA-pipelined.
- HBM write 6.11 TB/s is BELOW B300_TRUE_REFERENCE's 7.30 TB/s + NINJA recipe
  7.57 TB/s. M14 number appears to be plain STG, not the NINJA v8 recipe.
- "MUFU rsqrt 47.8 GMUFU/s" - MATH_INCONSISTENCY_LOG #3 specifically flags this:
  it is a 1-chain self-dep latency-bound rsqrt; the saturated MUFU pipe peak is
  4.74 G/s/chip (V41). M14 mis-labels this as "XU pipe peak".

### RETRACTIONS
- "MUFU rsqrt 47.8 GMUFU/s = 99.49% XU peak" - this is 1-chain rsqrt
  latency-bound regime. Saturated MUFU pipe = 4.74 Gops/s chip-wide (V41 + 14_math
  agree). EX2 is the outlier at 9.22 Gops/s (2; faster than other MUFU).
- "HBM read 5.82 TB/s = 81%" SoL row - SUPERSEDED by V46 TMA pipelined 7.20 TB/s
  (98.5%) for read SoL.
- "HBM write 6.11 TB/s = 85%" SoL row - SUPERSEDED by NINJA recipe 7.57 TB/s
  (98.7% spec) per B300_TRUE_REFERENCE.

### Addenda
- Add TMA-pipelined read row (V46): 7.20 TB/s = 98.5% HBM peak.
- Add NINJA write row (e75c7e1): 7.57 TB/s = 98.7% HBM peak.
- Add cp.async.ca row (V9): 6.91 TB/s = 96% HBM peak.
- Add NVFP4 wide-N row (3628a40): 10297 TFLOPS single-GPU = 103% of 10 PF spec.
- Add FP8 cuBLAS realistic row (random): 3983 TF (vs zero-data 4400 TF).

---

## M15 - V9_LATENCY_LADDER (CRITICAL - latencies)

### Claims affected
- FFMA / FADD / FMUL: 4.22 cy.
- IMAD: 4.25 cy.
- HMMA.F16: 20 cy.
- DFMA: 63.68 cy.
- mbarrier: 123 cy (per user memory).

### Status vs newer work
- 4.22 cy FFMA latency: consistent with B300_TRUE_REFERENCE (FFMA peak achieved
  with NCHAIN=3 = 96.92% supports ~4 cy depth).
- HMMA 20 cy: consistent with V5 B5 (HMMA chain saturates at ILP=8 -> ~10.4 cy/inst,
  implying single-issue 20+ cy latency). HIGH.
- DFMA 64 cy: matches "single port" model.
- LDS 29 cy, L1 47 cy, L2 ~300 cy, DRAM ~317 cy: M5 has L1 38-39 cy and L2 230 cy;
  per CACHES #L1 LATENCY this is a 38-47 cy spread, no real conflict.
- mbarrier 123 cy: appears in user memory "FFMA 4.22/HMMA 20/DFMA 64/mbarrier 123";
  M15 itself does not list mbarrier 123 cy directly but lists barrier.cluster 370 cy
  and __syncwarp 23 cy. The 123 cy mbarrier figure is consistent with V5 A6
  (mbarrier arrive = 24 cy short-form) plus wait latency.

### RETRACTIONS
- None numerically. Latency ladder is solid.

### Addenda
- Cross-reference V49 result: same-warp dual-issue at 55% means measured
  "single-warp" latencies in M15 reflect TRUE single-issue throughput, not
  dual-pipe stacking (which V49/V50 demonstrate is incomplete).

---

## M16 - V9_FULL_SYNTHESIS (CRITICAL - claims XU peak)

### Claims affected
- Pipe ownership map: "FMA pipe = FFMA, FADD, FMUL, IMAD, DFMA, HMMA"
  (all bundled).
- "ALU pipe = IADD3, LOP3.LUT, SEL, ISETP" (~38 TOPS @ 99.9%).
- "XU pipe peak: 47.8 GMUFU/s @ 99.5%".
- Mixed FMA + ALU = 131% pipe sum but 74 TOPS combined (corrected from "114 TOPS").

### Status vs newer work
- Pipe ownership map for FMA: bundling DFMA + HMMA into FMA is misleading. DFMA has
  its own port (single-port, 64 cy lat); HMMA uses tensor pipe. Both share dispatch
  but not the FMA execution unit per se.
- "ALU = IADD3" CONFLICTS with V40 finding that IADD3 runs at FMA-pipe rate
  (~26 Glane/s) while LOP3/IMUL run at half rate (~18.7 Glane/s = INT-bit pipe).
  M16's lumping is wrong.
- "XU peak 47.8 GMUFU/s" - per MATH_INCONSISTENCY_LOG #3: this is 1-chain rsqrt
  latency-bound, NOT saturated XU peak. True saturated MUFU pipe = 4.74 Gops/s/chip
  (V41/V37). Reframe row as "1-chain rsqrt latency-bound: 47.8 G; saturated MUFU
  pipe: 4.74 G".
- "Mixed FMA + ALU = 131% pipe sum, 74 TOPS combined" - V49 quantifies this further:
  same-warp dual = 55% of perfect parallelism, warp-specialized = 74%. M16's
  myth-bust direction is correct; magnitude refined by V49/V50.

### RETRACTIONS
- "XU peak: 47.8 GMUFU/s @ 99.5%" - mislabeled. True XU saturated peak is 4.74 G;
  47.8 G is a single-chain rsqrt rate (latency-bound). Per math agent.
- Pipe ownership map row "ALU = IADD3, LOP3.LUT, ..." - IADD3 belongs with FMA
  pipe (V40); LOP3/IMUL/PRMT/ISETP form INT-bit + permute + compare pipes.
- "Mixed FMA + ALU = 74 TOPS combined" - reframe: 55% same-warp, 74%
  warp-specialized of theoretical sum (per V49/V50).

### Addenda
- Add EX2 outlier row (V41): EX2 = 9.22 Gops/s = 2; faster than other MUFU.
- Add V49/V50 dual-issue ladder (55% same-warp / 74% warp-spec).

---

## RETRACTIONS (consolidated, by M-file)

### M2
- "Static 0.05 W/SM" cited as universal (true only at one operating point).

### M5
- "L1 ~46 TB/s" without ILP qualification (use 30.5 TB/s as conservative measured peak).
- "TMA bulk read = LDG within 0.3%" (true only single-deep TMA; V46 pipelined is 7% better).

### M7
- Wording "two FFMA sub-pipes" (fmaheavy/fmalite is ncu attribution detail).

### M8
- "MUFU + FFMA = 100%+" framing (issue-port absorption, not free dual-issue).
- IADD3 row's "ALU" pipe attribution (IADD3 actually FMA pipe per V40).

### M11
- "Static GPU 165-170 W regardless of utilization" (varies 144-198 W with clock).
- "FFMA = 0.111 TF/W" (half-peak measurement; peak is 0.21 TF/W).

### M12
- "Boost 2032 MHz no throttle" without precision qualifier (true FFMA, false random tcgen05).

### M14
- "MUFU rsqrt 47.8 GMUFU/s = 99.49% XU peak" (1-chain latency, not saturated peak).
- "HBM read 5.82 TB/s = 81%" SoL (superseded by V46 TMA-pipelined 7.20 TB/s = 98.5%).
- "HBM write 6.11 TB/s = 85%" SoL (superseded by NINJA recipe 7.57 TB/s = 98.7%).

### M16
- "XU peak: 47.8 GMUFU/s @ 99.5%" (mislabeled; true peak = 4.74 G).
- Pipe ownership map ALU row including IADD3 (IADD3 is FMA pipe per V40).
- "Mixed FMA + ALU = 74 TOPS combined" without same-warp/warp-spec disambiguation.

---

## UNRESOLVED

### M2 vs newer popcount data
LDG = +177 W (M2 H4) vs DRAM-pattern-dependent 240-367 W swing (POPCOUNT). M2's
LDG figure was data-pattern naive; needs re-derivation with controlled bit pattern.

### M11 vs 16_power_clock TFLOPS/W discrepancy
M11 says FFMA = 0.111 TF/W; 16_power_clock says 0.21 TF/W. Both should be
re-derived with explicit ILP/occupancy/clock specification.

### M14 NVFP4 row not present
M14 explicitly defers NVFP4 to V9; B300_TRUE_REFERENCE has 10297 TF wide-N (103% of
spec) and per K-id mechanism notes the realistic ceiling is ~1577 TF random. M14
should be extended with both rows.

### M15 mbarrier 123 cy origin
User memory cites "mbarrier 123 cy" for M15; M15 doc lists arrive 24 cy + wait
behavior but no explicit "123 cy" figure. Provenance unclear; needs commit hash.

### M8 cross-warp / warp-specialized row absence
M8 captures only same-warp overlap. V50 quantified warp-specialized 74% efficiency.
M8 should add a column for warp-specialized variant.

### M16 IMAD pipe attribution
M16 puts IMAD in FMA pipe; B300_TRUE_REFERENCE row 15 says IMAD 38.5 TOPS = 1:2 of
FP32 = HALF rate of FFMA. If IMAD is FMA-pipe, why half rate? Possible answer: IMAD
issues at 2 cy per SMSP (vs FFMA 1 cy). Needs explicit clarification in M16.
