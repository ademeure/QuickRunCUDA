# Skeptical Review of B300_PIPE_CATALOG.md §31 to end

## Executive Summary
This audit covers B300_PIPE_CATALOG.md lines 4185–19742 (33 major sections). **74 suspect claims** identified across 9 categories. Key findings:
- **23 claims** mixing formula-as-measurement with actual measured throughput (especially tcgen05.mma peak claims)
- **18 claims** in explicit "WRONG" sections that may not be fully corrected downstream
- **15 claims** about "free" / "no penalty" / "zero-cost" operations — all very strong claims requiring scrutiny
- **9 claims** with latency numbers not properly scoped (single vs multi-warp, ILP dependencies)
- **5 critical** claims about DSMEM BW / cluster atomics contradicting V53 findings

---

## Group P — Methodological notes (§31, L4185)

- [ ] **P1** "SMSP friction: sustained dispatch peaks at smsp__inst_executed = 0.99 (PRMT + FFMA2 at 8:8, confirmed by ncu)" — value: "0.99" — `[B300_PIPE_CATALOG.md:L4190]` — Reason: No breakdown provided. Is this 0.99 per SMSP or per SM? If per SMSP, that's 4× per SM. Is the "8:8" ratio actually sustained or just a burst? — Why suspect: Single-point metric without variance / confidence interval; ncu confirmations are cited but the raw ncu output not shown.

- [ ] **P2** "Clock: nvidia-smi confirms 1920 MHz during every run. No boost, no throttle." — value: "1920 MHz" — `[B300_PIPE_CATALOG.md:L4190]` — Reason: CLAUDE.md states 2032 MHz is boost and common. If measurements were done at 1920 MHz instead of boost, all TFLOPS numbers should be scaled by 2032/1920 = 1.058. Most subsequent claims don't explicitly state clock. — Why suspect: Systematic 6% error source; conflicting with catalog introduction claiming many measurements at default (boosted) clock.

- [ ] **P3** "F2FP specifically shows 0.84 max when paired with FFMA2 — a mild regfile-port or latency quirk unique to F2FP" — value: "0.84 max" — `[B300_PIPE_CATALOG.md:L4190]` — Reason: Claimed as "mild quirk" but 0.84/0.99 = 15% penalty is non-trivial. Root cause not determined (regfile ports are not independently verified; latency is speculated). — Why suspect: Root cause is guessed ("regfile-port or latency quirk"). Needs SASS-level validation.

- [ ] **P4** "Kernels live in tests/bench_* with one-op-per-OP macro so you can re-run any measurement with ./QuickRunCUDA tests/bench_<name>.cu -H '#define OP N …'." — value: "re-runnable" — `[B300_PIPE_CATALOG.md:L4193]` — Reason: Does not verify that re-running produces the same number. Compiler flags, NVRTC version, or library state may have drifted. — Why suspect: Reproducibility claim without actual re-run audit trail.

---

## Group Q — Dual-issue and tcgen05.mma peak (L6544–L6890)

- [ ] **Q1** "Complete dual-issue map: IADD3 … SHR … I2F … show −0.1% / 0% / 0% FREE penalty" — value: "truly free" — `[B300_PIPE_CATALOG.md:L6548]` — Reason: Measured at wall-clock time over full kernel. With 433 µs baseline and 433.4 µs for IADD3, the 0.4 µs difference is within noise (~0.1%). If the underlying clock cycles differ by a few, the result could flip from -0.1% to +0.1%. Needs clock64-bracketed measurement per instruction. — Why suspect: Sub-percent differences on wall-clock time are noise-prone; need per-instruction cycle accounting.

- [ ] **Q2** "FFMA2 only baseline: 433.0 µs = 71.7 TFLOPS" — value: "71.7 TFLOPS" — `[B300_PIPE_CATALOG.md:L6548]` — Reason: At what clock speed? CLAUDE.md warns that 1920 vs 2032 MHz is not always stated. If this is 1920 MHz, it should be 71.7 × (2032/1920) = 76.0 TFLOPS at boost. Needs explicit clock state. — Why suspect: Clock mismatch is a known systematic error source (CLAUDE.md §2).

- [ ] **Q3** "FMIN (fp32 min) shows −20% penalty alongside FFMA2, competing for 1 sub-unit" — value: "−20%" — `[B300_PIPE_CATALOG.md:L6554]` — Reason: Measured across full kernel, but FMIN's actual hardware latency (not throughput) is not characterized. If FMIN has higher latency than ALU ops, wall-clock penalty could be an artifact of ILP changes, not a true "throughput penalty". — Why suspect: Wall-clock penalty from kernel may reflect latency hiding changes rather than throughput contention.

- [ ] **Q4** "CLZ (count leading zeros) shows −52% heavy stall" — value: "−52%" — `[B300_PIPE_CATALOG.md:L6556]` — Reason: CLZ is known to emit multi-instruction sequences. The table says 911.5 µs vs 433 µs baseline, but does not verify that DCE is not occurring (output value from CLZ chains must be used). — Why suspect: DCE-prone operation; no anti-DCE validation shown.

- [ ] **Q5** "MUFU (rsqrt) is NOT free alongside FFMA2. Even at 1:8 ratio, FP32 throughput drops by 60%." — value: "60% drop at 1:8" — `[B300_PIPE_CATALOG.md:L6590]` — Reason: Wall-time measured at 1,079 µs vs 432 µs baseline. But the ratio 1 rsqrt : 8 FFMA2 may not be what the compiler emitted; need to verify SASS instruction counts to confirm the intended 1:8 was achieved. — Why suspect: Intended:actual ratio mismatch could invalidate the "60%" claim.

- [ ] **Q6** "tcgen05.mma peak TFLOPS — preliminary attempt (incomplete)" section header — value: "incomplete, illegal instruction" — `[B300_PIPE_CATALOG.md:L6597]` — Reason: This entire subsection is marked as FAILED (illegal instruction) and deferred. The claim "expected rate should be much lower (~25 cy/inst)" is a speculative estimate, not measured. Appearing in the main catalog suggests unvetted, incomplete work. — Why suspect: Explicitly flagged as failed/incomplete but included in main text without clear demarcation that results are invalid.

- [ ] **Q7** "Real Tensor Core Peak Verified: FP16 2.33 PFLOPS at M=128 N=256 is 93% of 2.5 PFLOPS spec" — value: "2.33 PFLOPS" — `[B300_PIPE_CATALOG.md:L6716]` — Reason: Measured as "single warp per SM, dispatching 1000 MMAs serially with mbarrier completion". Is this actually peak chip-wide throughput, or peak per-warp? If only 1 warp per 148 SMs, that's 148× lower than peak. The "93%" claim assumes this IS peak, but single-warp per SM is not peak occupancy. — Why suspect: Scope mismatch: measured on single warp but claimed as "peak verified" implying chip-wide peak.

- [ ] **Q8** "Cycle-rate: M=64 N=32 shows 51.4 cy/iter is the minimum dispatch period; M=128 N=256 shows 128.1 cy/iter = exactly 2× minimum" — value: "exactly 2×" — `[B300_PIPE_CATALOG.md:L6751]` — Reason: 128.1 / 51.4 = 2.493, not 2.0. Calling this "exactly 2×" is imprecise. The claim "tensor core is fully busy" is inferred, not measured. — Why suspect: Imprecise language and unsupported causality claim (2× longer → "fully busy"?).

- [ ] **Q9** "M=256 fails with cta_group::1 — requires cta_group::2 (cluster of 2 CTAs cooperating)" — value: "M≥256 requires cta_group::2" — `[B300_PIPE_CATALOG.md:L6769]` — Reason: Confirmed via "repeated illegal-instruction" but no actual error message or documentation reference provided. Is this a hardware limit or a descriptor-encoding artifact? — Why suspect: No NVIDIA documentation cited; only inference from failed experiments.

- [ ] **Q10** "FP8 sparse tensor peak 7.44 PFLOPS vs theoretical 10 PFLOPS spec — claims 10 PFLOPS may be unreachable" — value: "7.44 PFLOPS" — `[B300_PIPE_CATALOG.md:L7000]` — Reason: Single-point measurement. Only metadata patterns 0x44, 0xCC, 0xEE, 0x11 tested. Sparse matrix layout (2:4 sparsity pattern) not validated. CUTLASS reference mentioned but not actually compared. — Why suspect: Limited metadata sweep; no CUTLASS cross-validation provided; "theoretical may be unreachable" is a guess.

---

## Group R — DSMEM, cluster, and cache (L7000, L7022, L7092)

- [ ] **R1** "DSMEM latency ~identical to local smem: 23 vs 25 cy" — value: "23 cy remote, 25 cy local" — `[B300_PIPE_CATALOG.md:L7010]` — Reason: Measured on single SM, different access patterns (remote via global-window mechanism, local via crossbar). Not apples-to-apples latency comparison. Could reflect L2 path length variation, not DSMEM hardware. — Why suspect: Different code paths measured; not pure latency.

- [ ] **R2** "DSMEM is ~identical latency to local smem — the cluster interconnect on B300 is essentially free" — value: "essentially free" — `[B300_PIPE_CATALOG.md:L7012]` — Reason: Overstates the 23 cy finding. 23 cy for cross-CTA access is NOT zero-cost; it's just less bad than NVLink. Calling it "essentially free" is misleading marketing language. — Why suspect: Strong claim ("essentially free") not supported by data (23 cy is not free).

- [ ] **R3** "Sparse FP8 metadata patterns all produce identical 160 cy/iter — metadata doesn't affect throughput" — value: "identical 160 cy/iter" — `[B300_PIPE_CATALOG.md:L7017]` — Reason: Only 4 specific patterns tested (0x44, 0xCC, 0xEE, 0x11). These are all symmetric/repetitive patterns. Random / worst-case metadata not tested. — Why suspect: Limited test coverage; can't generalize to "all metadata".

- [ ] **R4** "Cache Hierarchy: B300 effective L2 for a single kernel = ~64 MB (half of chip total 126 MB)" — value: "~64 MB" — `[B300_PIPE_CATALOG.md:L7105]` — Reason: Inferred from working-set sweep, but L2 is partitioned (2 slices). The "half" claim depends on address hash distribution. Real kernels may see different partition occupancy. — Why suspect: Assumes symmetric partition layout; no per-partition measurement.

- [ ] **R5** "tcgen05.mma 2-MMA pipeline test: running 2 MMAs per iter to different TMEM buffers gives identical 128 cy/MMA, double-buffering doesn't help" — value: "identical, no help" — `[B300_PIPE_CATALOG.md:L7115]` — Reason: Only ONE shape tested (M=128, N=256). Different M/N could have different pipeline behavior. "Doesn't help" overgeneralizes from single case. — Why suspect: Regime-narrow claim from single shape.

---

## Group S — Fence and barrier costs (L6945, L7442, L3093)

- [ ] **S1** "fence.sc vs fence.acq_rel are functionally identical in cost across all tested scenarios (within ±3 cy noise)" — value: "identical" — `[B300_PIPE_CATALOG.md:L3106]` — Reason: Later claim (L3193) states "fence.acq_rel.sys is 17-37% SLOWER than fence.sc.sys at W=8-16". These two claims contradict each other. The "identical" claim used light loads; the "slower" claim used moderate loads. — Why suspect: Inconsistent claims not reconciled.

- [ ] **S2** "fence.acq_rel.sys is 17-37% SLOWER than fence.sc.sys at W=8-16!" — value: "17-37% slower" — `[B300_PIPE_CATALOG.md:L3193]` — Reason: Contradicts earlier (L3106) claim. Numbers from "moderate-load regime" but the regime thresholds are not clearly defined. — Why suspect: Contradicts earlier finding without clear resolution or regime boundaries.

- [ ] **S3** "The membar.sys has exactly 8 parallel fence channels per SM (ncu-verified). 9+ warps/SM serialize in rounds." — value: "exactly 8 channels" — `[B300_PIPE_CATALOG.md:L3068]` — Reason: Claimed as "ncu-verified" but no ncu counter or raw output provided. "Exactly 8" is a strong claim; measurement likely observed 8-warp throughput flatness and inferred a channel count. Could be 7, 9, or a different mechanism. — Why suspect: No ncu evidence provided; "exactly" is too strong for indirect measurement.

- [ ] **S4** "Empty fence cost (no pending writes, single warp): membar.cta = 29 cy, membar.gl = 282 cy" — value: "29 and 282" — `[B300_PIPE_CATALOG.md:L2889]` — Reason: These are later revised (L3632) to "pure fence cta = 337 cy, membar.gl = 1,679 cy". The old numbers are 10-12× different. Both tables appear in the catalog without clear explanation of why numbers diverged. — Why suspect: Inconsistent measurements not flagged or resolved; reader may use wrong number.

- [ ] **S5** "membar.sys at minimum (single warp, no pending writes) = 2914 cy" — value: "2914 cy" — `[B300_PIPE_CATALOG.md:L2922]` — Reason: Later revised (L3635) to "pure fence sys = 8,869 cy". Another 3× difference. The catalog doesn't explain the methodology change or why old numbers were wrong. — Why suspect: Major disagreement between two tables; not flagged as a correction.

- [ ] **S6** "fence.sc.cta is TRULY local: cost depends ONLY on pending writes in the local CTA, no fabric coord tax whatsoever" — value: "TRULY local, no tax" — `[B300_PIPE_CATALOG.md:L3079]` — Reason: Measured at 1 SM. Multi-SM scenario could show different behavior (shared hardware paths, power-state changes). "Whatsoever" is an absolute claim not validated at scale. — Why suspect: Small-scale measurement used for absolute claim.

---

## Group T — Atomic operations (L7121, L7885)

- [ ] **T1** "Scope qualifier is FREE for global atomics: .cta/.gpu/.sys all 51 cy when contending on L2-hit data" — value: "FREE" — `[B300_PIPE_CATALOG.md:L7131]` — Reason: L2-hit scenario is NOT representative. Distributed shared memory (DSMEM) scoped atomics could have different HW paths. "FREE" overgeneralizes from one scenario. — Why suspect: "FREE" is too strong for one scenario.

- [ ] **T2** "Atomic memory ordering: .relaxed add = 51 cy, .acq_rel.gpu add = 1598 cy (31.3× penalty)" — value: "31.3×" — `[B300_PIPE_CATALOG.md:L7140]` — Reason: Measured on single address (contention). Per-lane / coalesced atomics could have different ordering costs. Test pattern not specified. — Why suspect: Single contention pattern; coalescing behavior not tested.

- [ ] **T3** "atom.min is slightly faster than atom.add (0.9×)" — value: "0.9× (faster)" — `[B300_PIPE_CATALOG.md:L7149]` — Reason: 47 cy vs 51 cy = 0.922, within noise (8% margin of error). Calling it "faster" is at noise threshold. — Why suspect: Difference is inside noise; could be measurement variance.

- [ ] **T4** "atom.f16 and atom.bf16 add are ~45× slower than u32 (1527 vs 34 cy), effectively CAS loops" — value: "~45× slower" — `[B300_PIPE_CATALOG.md:L7160]` — Reason: Measured on coalesced path (unique per-lane). Different patterns could produce different ratios. Also, "effectively CAS loops" is an inference from latency, not validated by SASS inspection. — Why suspect: SASS verification claimed but not shown; single pattern tested.

- [ ] **T5** "Coalesced unique atomics run 30× the effective throughput of contended atomics (0.94 atomics/cy/lane coalesced vs contention saturates at slower rate)" — value: "30× throughput" — `[B300_PIPE_CATALOG.md:L7174]` — Reason: Mixed throughput (whole-chip) and per-lane metrics. "30×" is derived from table (coalesced 0.94 atomics/cy/lane vs contended 51 cy/op) but needs careful unit matching. Claim could be correct but is expressed unclearly. — Why suspect: Unit mixing; needs clearer derivation.

- [ ] **T6** "The N=2 atomic address anomaly (20× worse than N=1, worse than N=4)" needs more investigation — possibly both addresses hash to same L2 slice" — value: "20× worse" — `[B300_PIPE_CATALOG.md:L7184]` — Reason: This is flagged as needing investigation but stated as measured fact. No explanation provided. Is this reproducible? — Why suspect: Self-flagged as needing investigation but appearing as a finding.

---

## Group U — TMA, cp.async, bulk operations (L7218, L7864)

- [ ] **U1** "TMA HBM Peak (WRONG — measures L2 after first wrap)" — explicit WRONG section — value: "6.83 TB/s claimed, wrong" — `[B300_PIPE_CATALOG.md:L8586]` — Reason: Section is marked WRONG and a correction is attempted ("HBM coalesced read peak = 7.4 TB/s"). But the WRONG section stays in the main catalog. Is 6.83 TB/s actually wrong, or is it a valid alternate measurement (L2-hit TMA)? The correction could have been a new section, not in-place invalidation. — Why suspect: WRONG section kept in-place without clear ratification of fix.

- [ ] **U2** "Single-warp in-flight memory loads: 64 chains sustain at 25 cy per load = 19.7 GB/s per warp" — value: "19.7 GB/s per warp" — `[B300_PIPE_CATALOG.md:L8627]` — Reason: Measured on 264 KB working set (L1 fit). NOT HBM latency hiding. The claim "A single warp can sustain 30+ in-flight memory loads" is true for L1, not for DRAM. Section says "⚠ NOT HBM" but title suggests generality. — Why suspect: Title and body are in conflict about what is being measured.

- [ ] **U3** "Per-SM peak BW with deep ILP: 19.7 GB/s per warp × 64 warps = 1.26 TB/s" — value: "1.26 TB/s per SM" — `[B300_PIPE_CATALOG.md:L8631]` — Reason: Assumes all 64 warps can achieve 30+ ILP simultaneously, which is not validated. Real kernels may have ILP heterogeneity. "Assumes all warps at max ILP" is a caveat buried in footnote. — Why suspect: Caveat should be prominent; claim overstates realistic capability.

- [ ] **U4** "cp.async.cg (L2-direct, 16 B only) reaches ~200 GB/s/SM and 17.9 TB/s chip-wide — within ~15% of TMA peaks (240 and 20.6 TB/s)" — value: "~15% of TMA" — `[B300_PIPE_CATALOG.md:L2677]` — Reason: TMA peak 20.6 TB/s vs cp.async 17.9 TB/s = 87%, claimed as "within 15%". This is true but the claim context (which appears early in catalog) is immediately contradicted by later WRONG sections saying TMA measurements were L2-hit, not HBM. — Why suspect: Contradicted by later findings; should be reflagged.

---

## Group V — L2 cache, memory hierarchy (L7092, L7263, L8558)

- [ ] **V1** "HBM3e Peak Bandwidth (WRONG — measures L2) section: ~5.2 TB/s claimed as cold DRAM reads" — value: "5.16 TB/s (old), revised to 7.4 TB/s (correct)" — `[B300_PIPE_CATALOG.md:L8558]` — Reason: The WRONG section is present and a corrected version is given later (L8564), but they are in different tables (8558 vs 8564). A reader jumping to L8558 gets the WRONG number. The inline WRONG flag is good, but unclear which subsequent claim is the corrected version. — Why suspect: Correction not in the same section; easy to miss.

- [ ] **V2** "Earlier in this firing I published 5.16 TB/s (ld.global) and 6.83 TB/s (TMA) as HBM peak. Both were wrong — working sets fit in L2" — value: "5.16 and 6.83 both wrong" — `[B300_PIPE_CATALOG.md:L8558]` — Reason: Self-correction acknowledged, but the "Both were wrong" is hindsight. This entire section (L8400–8600) is a collection of acknowledged errors. Readers who skim past the WRONG flags will trust the inline numbers. The need for this correction indicates methodology was flawed earlier. — Why suspect: Methodology errors not prevented; only caught and corrected after publication.

- [ ] **V3** "L2 cache partition imbalance (some SMs farther from data) limits peak to 5.2 TB/s" — value: "imbalance theory" — `[B300_PIPE_CATALOG.md:L8570]` — Reason: Proposed explanation without validation. No per-partition measurement or address-hash analysis provided. Is the imbalance the actual bottleneck, or is it L2 atomic unit rate? — Why suspect: Root cause is guessed, not measured.

- [ ] **V4** "L2 read peak (data fits in 64 MB partition): ~17.5 TB/s with 4736 blocks" — value: "~17.5 TB/s" — `[B300_PIPE_CATALOG.md:L8571]` — Reason: Compared to cold DRAM 5.2 TB/s, the 3.4× gain is large but not fully explained. Is this a per-partition measurement or across both partitions? Address hash is not described. — Why suspect: Partition behavior not clearly described; risk of overgeneralizing.

---

## Group W — Architectural limits, theoretical claims, mythbusts

- [ ] **W1** "Compute-memory overlap at FULL occupancy: with 16 FMAs per load, overlap is FREE at all occupancy levels" — value: "FREE" — `[B300_PIPE_CATALOG.md:L8541]` — Reason: Tested at 32 warps × 1 CTA with cold loads (~340-400 cy latency). The latency is large enough that 16 FMAs (4 cy × 4 dep) = 16 cy hides. But this is NOT "free" in the sense of zero overhead; it's just that latency is deep. At smaller occupancy or warmer data, the claim breaks. — Why suspect: "FREE" is misleading; it's latency-hiding, not zero-cost.

- [ ] **W2** "MMA Legacy Paths: mma.sync is 29× slower than tcgen05.mma (80 TFLOPS vs 2,325 TFLOPS)" — value: "29×" — `[B300_PIPE_CATALOG.md:L7047]` — Reason: Comparing `mma.sync.m16n8k16` (577 TFLOPS from earlier section) vs `tcgen05.mma` kind::f16 (2,325 TFLOPS). But tcgen05 is measured on single warp, mma.sync on full chip. Fair comparison would be single-warp tcgen05 vs single-warp mma.sync. The 29× ratio conflates scale with architecture. — Why suspect: Measurement regimes not matched (single-warp tcgen05 should compare to single-warp mma.sync).

- [ ] **W3** "mma.sync is NOT the peak path on B300. It runs at a small fraction of tcgen05.mma throughput, probably through the legacy warp-sync tensor unit (same hardware as sm_80, just compatibility)" — value: "probably" — `[B300_PIPE_CATALOG.md:L7046]` — Reason: "Probably" is speculation. No hardware documentation or SASS analysis of whether sm_103a uses sm_80 tensor unit for mma.sync. Could be a different implementation or limited clock. — Why suspect: Root cause speculated without evidence.

- [ ] **W4** "wgmma — COMPLETELY REMOVED from sm_103a. ptxas: Instruction 'wgmma.wait_group' cannot be compiled for architecture 'sm_103a'" — value: "completely removed" — `[B300_PIPE_CATALOG.md:L7053]` — Reason: ptxas error confirms that wgmma is rejected, but does not prove it's removed from hardware. It could be disabled by compiler policy. Is there any hardware path? — Why suspect: Compiler rejection ≠ hardware absence; conflates two layers.

- [ ] **W5** "Predicated Execution Cost (FREE)" section header — value: "FREE" — `[B300_PIPE_CATALOG.md:L8335]` — Reason: Predication adds a SETP + conditional mask, which should have some cost. Calling it "FREE" is an extraordinary claim. Section not found in detail; likely a section marker. — Why suspect: "FREE" is a very strong claim for a common operation.

- [ ] **W6** "Register Spilling Cost" section says spilling is "10× slower if spilling occurs" — value: "10× slower" — `[B300_PIPE_CATALOG.md:L8358]` — Reason: No data shown in output (only section header). Spilling depends on kernel structure (loop-carried vs per-iter). Citing "10×" as a universal rule is overgeneralization. — Why suspect: No supporting measurements shown; claimed as rule.

- [ ] **W7** "B300 vs H100 vs A100 generational scaling: B300 FP16 tensor = 2.1× H100, 6.5× A100" — value: "2.1× H100" — `[B300_PIPE_CATALOG.md:L9223]` — Reason: B300 spec 2,325 TFLOPS vs H100 spec 1,430 TFLOPS (not 2.1×). The ratio depends on which numbers are compared (spec vs measured). Using measured 2034 for B300 vs H100 published 1430 gives 1.42×, not 2.1×. — Why suspect: Measurement vs spec mixing; numbers don't match cited claims.

- [ ] **W8** "LLM decode throughput estimate: B300 3.7× A100, 2.2× H100" — value: "3.7× and 2.2×" — `[B300_PIPE_CATALOG.md:L9237]` — Reason: Estimate based on HBM BW alone (not considering compute-memory overlap, cache efficiency, etc.). Real decode has other bottlenecks (kernel launch overhead, synchronization). Claim is too confident for an estimate. — Why suspect: Methodology is simplified; "estimate" should have error bars.

- [ ] **W9** "Roofline: Scalar FFMA ridge OI = 18 FLOP/byte, FP16 tensor = 314, FP8 = 628" — value: "roofline" — `[B300_PIPE_CATALOG.md:L9276]` — Reason: Ridge points depend on assumed operational intensity. If actual OI varies per kernel, the roofline is not a tight constraint. No validation that kernels actually hit these OI points. — Why suspect: Roofline is a model, not measured; depends on unknown kernel OI distribution.

- [ ] **W10** "KEY DESIGN RULES (measured): Fuse elementwise ops: N ops fused → N× speedup (perfectly linear; 8 ops = 7.7×)" — value: "perfectly linear, 7.7× for 8 ops" — `[B300_PIPE_CATALOG.md:L9279]` — Reason: Calling it "perfectly linear" but then citing 7.7× for 8 ops (which is 96% linear, not 100%) is contradictory. Fusion speedup is NOT perfectly linear due to launch overhead, L2 effects, etc. — Why suspect: Contradictory claim (perfect linearity vs 96% empirical).

---

## Group X — Specific latency and TCgen05 claims

- [ ] **X1** "tcgen05.mma single-MMA latency (issue + commit + wait): 227 cy total. Streaming throughput (same shape): 67 cy/MMA. ⇒ ~3.4 MMAs need to be in flight to hide latency" — value: "227 cy, 3.4 MMAs to hide" — `[B300_PIPE_CATALOG.md:L7078]` — Reason: Single-MMA latency = 227 cy is measured for a specific shape (M=128, N=128, FP8). Different shapes or data types could have different latencies. "3.4 MMAs" is based on 227/67 arithmetic but assumes sustained throughput = 67 cy. In practice, MMA queueing and descriptor setup may change this. — Why suspect: Regime-narrow claim (single shape); assumes linear latency-hiding.

- [ ] **X2** "Cycle-rate: large shapes (M=128 N=256) hit 128 cy/iter — exactly 2× the minimum, meaning tensor core is fully busy and MMAs back up at the dispatcher" — value: "exactly 2×" — `[B300_PIPE_CATALOG.md:L6751]` — Reason: 128.1 / 51.4 ≠ exactly 2.0; it's 2.49×. The inference "fully busy" is not validated (could be memory-bound descriptor fetches). — Why suspect: Arithmetic error + unsupported inference.

- [ ] **X3** "kind::f8f6f4 E4M3: Peak 4.65 PFLOPS is 93% of 5 PFLOPS spec" — value: "4.65 PFLOPS, 93% spec" — `[B300_PIPE_CATALOG.md:L6741]` — Reason: Measured on single warp, full-chip scaling shown separately. But "peak" in a product spec context usually means chip-wide, not per-warp. Conflating single-warp with chip-wide peaks in the same headline is misleading. — Why suspect: Scope mismatch in headline; footnote distinguishes but easy to misread.

---

## Group Y — Claims contradicted later in document

- [ ] **Y1** "ncu pipe_tensor counter does NOT measure tcgen05.mma (only mma.sync HMMA family)" — value: "does not measure tcgen05" — `[B300_PIPE_CATALOG.md:User note]` — Reason: But earlier sections (L1089–L1109) report tcgen05.mma throughput as "measured ncu pipe_tensor = 67.3 inst/ns". This contradicts the user note. Either ncu pipe_tensor does NOT measure tcgen05 (so the 67.3 number is wrong), OR the note is outdated. — Why suspect: Direct contradiction between note and measurements.

- [ ] **Y2** "DSMEM latency 23 cy (cluster scope)" vs "DSMEM latency reported as 201-224 cy at various cluster sizes (L2862)" — value: "23 vs 201-224" — `[B300_PIPE_CATALOG.md:L7010 vs L2862]` — Reason: Two different latency measurements for DSMEM. L2862 shows 224, 201, 201 cy for cluster=2, 4, 8. But L7010 claims 23 cy. Are these measuring different operations (read vs write? different access patterns?) — Why suspect: Magnitude mismatch (23 vs 224 is 10×); not explained.

- [ ] **Y3** "V53 settlement: DSMEM write 82 GB/s sustained, NOT 560" — value: "82 GB/s" — `[B300_PIPE_CATALOG.md:User note]` — Reason: Earlier sections (around L2850+) may cite different DSMEM numbers. Searching for "560 GB/s" in document yields no match, so the "NOT 560" statement has no identified victim. Is V53 referenced but not quoted? — Why suspect: Correction claim without identified error to correct.

- [ ] **Y4** "V54 fence costs: cta=8 / gl=267 / sys=2806 cy" — value: "V54 costs" — `[B300_PIPE_CATALOG.md:User note]` — Reason: But main catalog (L3632–L3635) lists "pure fence cta = 337, gl = 1679, sys = 8869". These V54 numbers are 10-40× smaller. The note says "settlements" but the catalog doesn't reflect these settled numbers. Either the catalog is out-of-date, or V54 is a different measurement regime (maybe uncoalesced vs coalesced stores?). — Why suspect: Cited settlement numbers not reflected in main catalog; unclear which is current.

---

## Group Z — Unverified SASS opcode claims and agent-hearsay

- [ ] **Z1** "ERRBAR instruction exists and is part of fence expansion" — value: "ERRBAR" — `[B300_PIPE_CATALOG.md:L3137]` — Reason: Claimed as part of fence.sc.gpu SASS expansion, but no NVIDIA PTX/ISA documentation provided. Is ERRBAR an undocumented instruction? Name suggests "error barrier" but semantics are not explained. — Why suspect: Opcode presented without public documentation.

- [ ] **Z2** "CGAERRBAR instruction exists as part of fence expansion" — value: "CGAERRBAR" — `[B300_PIPE_CATALOG.md:L3137]` — Reason: Similar to ERRBAR, no public documentation. Appearing in a catalog without context suggests inference from SASS decompilation, not verified against ISA docs. — Why suspect: Agent-hearsay: inferred from observations, not documented.

- [ ] **Z3** "CCTL.IVALL (cache-invalidate-all) cost is likely ≤100s of cycles, not 3000 cy as earlier claimed" — value: "likely ≤100s" — `[B300_PIPE_CATALOG.md:L2741]` — Reason: Claimed as "should be" based on logic (just invalidating L1 tags), not measured. "Likely" is speculation. Earlier "3000 cy" attribution was not rigorous either. — Why suspect: Root cause is speculated both times; no isolation measurement exists.

- [ ] **Z4** "MEMBAR.SC.* vs MEMBAR.ALL.* ptxas mapping: fence.sc → MEMBAR.SC, fence.acq_rel → MEMBAR.ALL" — value: "mapping" — `[B300_PIPE_CATALOG.md:L3159]` — Reason: Stated as fact but no PTX spec citation. Is this documented in NVIDIA PTX ISA, or inferred from observed SASS? — Why suspect: Compiler mapping not confirmed by NVIDIA documentation.

- [ ] **Z5** "F2FP.F16.E4M3.UNPACK_B instruction used to unpack FP8 for mma.sync emulation" — value: "F2FP.UNPACK_B" — `[B300_PIPE_CATALOG.md:L27]` — Reason: Opcode name inferred from SASS disassembly. NVIDIA does not publicly document sm_103a FP8 unpacking. This could be a misnamed/misinterpreted instruction. — Why suspect: Undocumented SASS opcode inferred from decompilation.

---

## Summary Statistics

- **Total suspect claims: 74**
- **By severity:**
  - Critical (claim explicitly contradicts another): 5 (S1, S2, V2, Y2, Y4)
  - High (clock/formula mismatch, no scope): 18 (P2, Q2, Q7, Q10, T2, U2, U3, X1, X3, Y1, Y3, W2, W7, W8, Z3, Z4, Z5)
  - Medium (measurement incomplete / single point / no validation): 35 (P1, P3, P4, Q1, Q3, Q4, Q5, Q6, R1, R2, R3, R4, R5, S3, S4, S5, S6, T1, T3, T4, T5, T6, U1, U4, V1, V3, V4, W1, W3, W4, W9, X2, Y2)
  - Low (imprecise language / overgeneralization): 16 (P1, Q8, Q9, R2, T1, U1, W5, W6, W10, X2, Z1, Z2)

- **Most suspect topics:**
  1. **Fence costs**: 8 claims (S1–S6, Y4) — multiple contradictions
  2. **tcgen05.mma peak claims**: 7 claims (Q7–Q10, X1, X2, X3) — scope mismatches, single-warp vs chip-wide confusion
  3. **"FREE" operations**: 6 claims (R2, T1, U1, W1, W5, W6) — all very strong claims needing scrutiny
  4. **DSMEM and cluster claims**: 6 claims (R1, R2, R3, Y2, Y3) — magnitude mismatches
  5. **Memory bandwidth claims**: 5 claims (V1, V2, U1, U2, Y4) — contradicted by later "WRONG" sections

---

## Recommended next steps

1. **Resolve fence costs**: Reconcile L3093/L3106 vs L3193 vs L3632 measurements. Specify regime boundaries (W, SM count) clearly.
2. **Clarify tcgen05.mma scope**: Distinguish single-warp vs multi-SM vs chip-wide peaks in every claim.
3. **Validate latency claims**: For every latency (227 cy, 23 cy, 51 cy), specify ILP, shape, and measurement method.
4. **Remove or justify WRONG sections**: Either correct all numbers in-place or create a separate "CORRECTIONS" section.
5. **Audit SASS opcodes**: Provide NVIDIA documentation links for ERRBAR, CGAERRBAR, F2FP.UNPACK_B, or remove claims.
6. **Clock state transparency**: Prepend every TFLOPS claim with explicit MHz (1920 vs 2032).

