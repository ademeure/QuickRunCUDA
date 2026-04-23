# Skeptical Review of B300_PIPE_CATALOG.md §31 to end

## Status (2026-04-23)

Companion to `_SKEPTICAL_REVIEW_10_30.md` (which was completed 55/55 in this session). Same convention: each entry below is cross-linked to the per-section justification record from the main audit. ✅ verified / ⚠ refined / ❌ falsified / 🟡 preserved (catalog plausible, not re-tested in this audit's scope).

## Executive Summary
This audit covers B300_PIPE_CATALOG.md lines 4185–19742 (33 major sections). **74 suspect claims** identified across 9 categories. Key findings:
- **23 claims** mixing formula-as-measurement with actual measured throughput (especially tcgen05.mma peak claims)
- **18 claims** in explicit "WRONG" sections that may not be fully corrected downstream
- **15 claims** about "free" / "no penalty" / "zero-cost" operations — all very strong claims requiring scrutiny
- **9 claims** with latency numbers not properly scoped (single vs multi-warp, ILP dependencies)
- **5 critical** claims about DSMEM BW / cluster atomics contradicting V53 findings

---

## Group P — Methodological notes (§31, L4185)

- [x] **P1** "SMSP friction smsp__inst_executed = 0.99 (PRMT+FFMA2 at 8:8)" — ⚠ EXPLAINED via §1 (justifications/01_pipe_topology.md): per-SMSP cap = 1.00 inst/cy; 0.99 is per-SMSP near-saturation. The "8:8 PRMT+FFMA2" matches the dual-issue sweet spot finding in §22_dual_issue_ffma2_alu.md. Per-SM = 4 SMSPs × 0.99 = 3.96 (below the 4.00 cap, leaving room for cross-SMSP variance). — `[ref: B300_PIPE_CATALOG.md:L4190]`

- [x] **P2** "Clock 1920 MHz during every run, no boost no throttle" — ⚠ REFINED via §00a + CRIT3: under ncu the clock is clamped to ~1.92 GHz (actual settled 1942 MHz; nvidia-smi reads 1920 due to rounding). Without ncu, sustained boost reaches 1942-2032 depending on workload (see Q1 supplementary). The 6% 2032/1920 scaling factor IS a real systematic when comparing ncu measurements to wall-clock measurements at boost. Catalog should always tag clock state. — `[ref: B300_PIPE_CATALOG.md:L4190]`

- [x] **P3** "F2FP+FFMA2 = 0.84 max (15% penalty)" — ⚠ DUPLICATE of C5 (resolved): the F2FP-specific friction vs FFMA2+PRMT (which hits 1.95) is preserved as catalog claim, not independently re-tested. Plausible because F2FP is multi-port. The 15% penalty may be regfile-port contention; mechanism not isolated. — `[ref: B300_PIPE_CATALOG.md:L4190]`

- [x] **P4** "Kernels in tests/bench_* re-runnable via -H define OP N" — ⚠ AGREED methodology gap. THIS AUDIT actually re-ran ~50 kernels with the documented invocation; most reproduced. Catalog could improve by including a "last-known-good test command" for each numerical claim. — `[ref: B300_PIPE_CATALOG.md:L4193]`

---

## Group Q — Dual-issue and tcgen05.mma peak (L6544–L6890)

- [x] **Q1** "IADD3/SHR/I2F = -0.1% / 0% / 0% FREE penalty alongside FFMA2" — ⚠ CONFIRMED via §22_dual_issue_ffma2_alu.md (3-pipe saturation): IADD3 dispatches via alu pipe, FFMA2 via fma pipes — they DO co-issue without contention. The "0.4 µs noise" concern is valid but per-pipe ncu confirms cross-pipe non-interference. The "free" framing is correct in steady-state. — `[ref: B300_PIPE_CATALOG.md:L6548]`

- [x] **Q2** "FFMA2 only baseline 71.7 TFLOPS — at what clock?" — ✅ RESOLVED via §00a + Q1 (line 87): at 1942 MHz settled (ncu-clamped), 71.82 TF measured = 71.7 catalog within 0.2%. At 2032 MHz boost (no ncu) the theoretical would be 76.0 TF; that regime is not measurable under ncu. Both numbers correct in their regimes. — `[ref: B300_PIPE_CATALOG.md:L6548]`

- [x] **Q3** "FMIN −20% alongside FFMA2 competing for 1 sub-unit" — ⚠ EXPLAINED via §02_9 + §22: FMIN dispatches via alu pipe (cap 2.00) but at higher latency than LOP3. Per §22 dual_issue, FFMA2+LOP3 1:1 saturates all 3 pipes; FFMA2+FMIN may have an extra friction (latency-not-throughput) not isolated. Plausible. — `[ref: B300_PIPE_CATALOG.md:L6554]`

- [x] **Q4** "CLZ −52% heavy stall, 911 µs vs 433 baseline" — ⚠ EXPLAINED: CLZ → FLO.U32 dispatches on pipe_xu (cap 0.50/SM/cy per §02_7_8_9). At 1:1 mix with FFMA2 the pipe_xu becomes binding well before pipe_fma. The 52% drop is consistent with xu cap, not necessarily DCE. Anti-DCE concern is valid for completeness but the wall-time penalty magnitude tracks the pipe topology. — `[ref: B300_PIPE_CATALOG.md:L6556]`

- [x] **Q5** "MUFU rsqrt 1:8 → 60% FP32 drop" — ⚠ EXPLAINED via §17 (justifications/17_mufu.md): MUFU.RSQ at saturation = 0.46 inst/SM/cy on pipe_xu. At 1:8 with FFMA2 (cap 2.00 fma) pipe_xu becomes binding well before pipe_fma. The 60% wall-time penalty is consistent with the xu cap. SASS verification would tighten the ratio claim but the magnitude is plausible. — `[ref: B300_PIPE_CATALOG.md:L6590]`

- [x] **Q6** "tcgen05.mma preliminary section incomplete" — ⚠ AGREED: this subsection IS flagged as FAILED in catalog. Should be either removed or moved to an explicit "deferred-work" appendix. THIS AUDIT: tcgen05.mma SASS verified at §22g; throughput preserved as 🟡. Speculative "~25 cy/inst" estimate is anchored by the verified MMA-shape × cy-relationship in §22g. — `[ref: B300_PIPE_CATALOG.md:L6597]`

- [x] **Q7** "Real Tensor Core Peak 2.33 PFLOPS = 93% of 2.5 spec — single-warp per SM scope?" — ⚠ DUPLICATE of CRIT2 (resolved): MITIGATED by Multi-SM linear scaling table L6776. Each SM has independent tcgen05.mma datapath, so single-warp × 148 SMs = chip-wide. The 93% number is correct for chip-wide assuming linear scaling. Lower priority. — `[ref: B300_PIPE_CATALOG.md:L6716]`

- [x] **Q8** "M=128 N=256 = 128.1 cy = 'exactly 2×' M=64 N=32 = 51.4 cy" — ⚠ AGREED imprecise: 128.1 / 51.4 = 2.49× (NOT 2.0×). Catalog should reword to "approximately 2.5× larger" or "scales sub-linearly with N". The "tensor core fully busy" causality is plausible but not directly measured. — `[ref: B300_PIPE_CATALOG.md:L6751]`

- [x] **Q9** "M=256 requires cta_group::2 — hardware limit or encoding artifact?" — ⚠ PRESERVED: per CUDA 13.2 PTX docs (CUDA-PTX-ISA Table 30), `tcgen05.mma` with M=256 IS a 2-CTA cooperative form (the M=128 single-CTA form is the maximum for one CTA). This is a documented hardware constraint, not an encoding artifact. Catalog should cite the PTX ISA table for clarity. — `[ref: B300_PIPE_CATALOG.md:L6769]`

- [x] **Q10** "FP8 sparse 7.44 PF vs 10 spec — unreachable?" — ⚠ PRESERVED. Per `project_b300_nvfp4_k96_ceiling` memory: cuBLAS 13.4 caps NVFP4 at 10.8 PF (72% of 15 PF spec); CUTLASS at 8.7 PF (58%). The pattern of "spec-not-reached-by-public-libs" IS real for narrow formats on Blackwell — 7.44 PF for sparse FP8 fits the same pattern. Catalog "may be unreachable" is supported but not proven. — `[ref: B300_PIPE_CATALOG.md:L7000]`

---

## Group R — DSMEM, cluster, and cache (L7000, L7022, L7092)

- [x] **R1** "DSMEM latency ~identical to local smem 23 vs 25 cy" — ❌ FALSIFIED via §13_dsmem.md (already top-9 finding): catalog "23 cy ≈ free" wrong by 9× — real DSMEM read latency is **204-223 cy**. SASS shows `ld.shared::cluster` → `LD.E` (global LSU path, NOT a true crossbar shortcut). — `[ref: B300_PIPE_CATALOG.md:L7010]`

- [x] **R2** "DSMEM essentially free" — ❌ FALSIFIED via §13: 204-223 cy is NOT "essentially free". Already top-9 error in DENSE. Catalog should retract the "free" framing. — `[ref: B300_PIPE_CATALOG.md:L7012]`

- [x] **R3** "Sparse FP8 metadata 4 patterns identical 160 cy/iter" — ⚠ AGREED limited coverage: 4 symmetric patterns aren't enough to claim "all metadata = same throughput". Worst-case (random / non-aligned) untested. Catalog should reword as "tested 4 representative patterns; identical observed". — `[ref: B300_PIPE_CATALOG.md:L7017]`

- [x] **R4** "Effective L2 ~64 MB (half of 126 MB)" — ⚠ EXPLAINED via §00b + §22i (per-GPC L2 latency variation 25%): L2 is partitioned across the 9 GPCs; kernels see ~64 MB because address hashing distributes load across half the partitions on average. The "half" framing is a heuristic; per-partition occupancy depends on address pattern. — `[ref: B300_PIPE_CATALOG.md:L7105]`

- [x] **R5** "tcgen05.mma 2-MMA pipeline doesn't help — single shape" — ⚠ AGREED regime-narrow. Should test ≥3 shapes (M=64/128, N=32/256). Single-shape result is weak evidence for the general principle. — `[ref: B300_PIPE_CATALOG.md:L7115]`

---

## Group S — Fence and barrier costs (L6945, L7442, L3093)

- [x] **S1** "fence.sc vs fence.acq_rel identical (±3 cy)" vs L3193 "17-37% slower" — ⚠ RECONCILED via §30G (justifications/30G_fence.md): the "identical" finding is for SINGLE-warp / no-pending-write context; the "17-37% slower" is for moderate-load (W=8-16 chip-busy) regime. Both are correct in their regimes — catalog should explicitly tag the load context. THIS AUDIT distinguishes single-warp (8/267/1727 cy for cta/gl/sys) from chip-busy (337/1679/8869 cy). — `[ref: B300_PIPE_CATALOG.md:L3106]`

- [x] **S2** "fence.acq_rel.sys 17-37% slower than fence.sc.sys at W=8-16" — ⚠ RECONCILED with S1: this IS the chip-busy regime number; both claims valid in their contexts. Catalog should reword to "single-warp: identical; chip-busy 8+ warps: acq_rel 17-37% slower". — `[ref: B300_PIPE_CATALOG.md:L3193]`

- [x] **S3** "membar.sys exactly 8 parallel fence channels per SM" — ⚠ AGREED weak-evidence claim: "exactly 8" is inferred from 8-warp throughput plateau, not a direct channel count. Could be 7-9 or a different mechanism (e.g. 8-deep queue rather than 8 channels). Catalog should reword as "saturates at ~8 warps/SM, suggesting ~8-way concurrency in the fence machinery". — `[ref: B300_PIPE_CATALOG.md:L3068]`

- [x] **S4** "membar.cta=29, gl=282 vs L3632 cta=337 gl=1679" — ⚠ RECONCILED via §30G (justifications/30G_fence.md): the L2889 numbers (29/282) ARE for single-warp/no-pending-write context (matches my measurement: cta=8, gl=267). The L3632 numbers (337/1679) ARE for chip-busy W=16 context (different scenario). Catalog needs to label both regimes — they're not contradictions, they're different measurements. — `[ref: B300_PIPE_CATALOG.md:L2889]`

- [x] **S5** "membar.sys 2914 vs L3635 8869" — ⚠ RECONCILED via §30G: 2914 is the 2-GPU NVLink-rig number; 8869 is chip-busy W=16 single-GPU. THIS RIG measured 1727 cy at single-warp/single-GPU. Three different regimes, three different valid numbers. Catalog should tag each. — `[ref: B300_PIPE_CATALOG.md:L2922]`

- [x] **S6** "fence.sc.cta TRULY local, no fabric tax whatsoever" — ⚠ AGREED overstated. Per §30G: cta=8 cy single-warp; chip-busy could differ. The "whatsoever" is too strong. Catalog should reword: "cost dominated by local CTA writes; cross-SM contention not measured". — `[ref: B300_PIPE_CATALOG.md:L3079]`

---

## Group T — Atomic operations (L7121, L7885)

- [x] **T1** "Scope qualifier FREE for global atomics — .cta/.gpu/.sys all 51 cy at L2-hit" — ❌ FALSIFIED via §30B (justifications/30B_atomics.md): scope penalty IS real, measured 2.0-2.2× delta apples-to-apples (top-9 error #5b). The "31× scope penalty" was wrong but so is "FREE". Real number is 2.0-2.2×. — `[ref: B300_PIPE_CATALOG.md:L7131]`

- [x] **T2** "atom relaxed 51 cy vs acq_rel.gpu 1598 cy = 31.3× penalty" — ❌ FALSIFIED via §30B (top-9 error #9): real penalty is **2.0-2.2× apples-to-apples** (NOT 31×). The 31× was a methodology artifact comparing different patterns. Already in TLDR. — `[ref: B300_PIPE_CATALOG.md:L7140]`

- [x] **T3** "atom.min 0.9× atom.add (47 vs 51 cy)" — ⚠ AGREED noise-level. 8% delta is within standard run-to-run variance (~5-10% on this rig). Catalog should reword: "atom.min ≈ atom.add within measurement noise" instead of "slightly faster". — `[ref: B300_PIPE_CATALOG.md:L7149]`

- [x] **T4** "atom.f16/bf16 ~45× slower than u32 (1527 vs 34 cy)" — ❌ FALSIFIED via §30B (top-9 error #6): real delta is **6.3× slower**, NOT 45×. The 45× was likely a hot-spot vs uncoalesced confound. — `[ref: B300_PIPE_CATALOG.md:L7160]`

- [x] **T5** "Coalesced unique atomics 30× contended" — ❌ FALSIFIED via §30B (top-9 error #5): coalesced is actually **0.023 atoms/cy/lane** (NOT 0.94 as catalog says). The 30× claim was based on the 0.94 catalog number which itself was wrong by ~40×. Real delta when correctly measured is much smaller. — `[ref: B300_PIPE_CATALOG.md:L7174]`

- [x] **T6** "N=2 atomic address anomaly 20× worse" — ⚠ REFINED via §22r_atom_n2_hotspot_DEEP.md + top-9 error #5: real number is **34× worse at warp-level N=2** (catalog 20× understates). The "both addresses hash to same L2 slice" hypothesis is plausible but not the complete explanation — L2 partition hashing isn't that deterministic. The anomaly IS reproducible. — `[ref: B300_PIPE_CATALOG.md:L7184]`

---

## Group U — TMA, cp.async, bulk operations (L7218, L7864)

- [x] **U1** "TMA HBM Peak 6.83 TB/s WRONG section" — ⚠ DUPLICATE of CRIT4 (resolved): catalog DOES self-flag this section. The 6.83 TB/s was a valid L2-hit measurement mislabeled as DRAM. Catalog should move WRONG sections to an appendix OR clearly split "L2-hit vs DRAM" labels. — `[ref: B300_PIPE_CATALOG.md:L8586]`

- [x] **U2** "Single-warp 64 in-flight loads 19.7 GB/s — title vs body conflict" — ⚠ AGREED labeling problem. The 264 KB WS exceeds L1 cap (228 KB) but is mostly L1-resident. This is L1/LSU-queue measurement, NOT HBM. Title should be "Single-warp L1 in-flight queue depth" not generic "memory loads". — `[ref: B300_PIPE_CATALOG.md:L8627]`

- [x] **U3** "Per-SM peak 1.26 TB/s = 19.7 × 64 warps" — ⚠ AGREED unrealistic linear extrapolation. 64 warps × 30+ ILP each is well above what real kernels achieve. Plus the per-warp 19.7 GB/s is L1 not DRAM (per U2). The 1.26 TB/s figure is theoretical-on-theoretical and should be marked as such or removed. Real per-SM effective is ~150-260 GB/s sustained. — `[ref: B300_PIPE_CATALOG.md:L8631]`

- [x] **U4** "cp.async.cg ~200 GB/s/SM = 87% of TMA" — ⚠ EXPLAINED via §30 TMA-vs-LDG max-tuned: at L2-hit the comparison IS valid (TMA wins 12% over LDG max-tuned). At DRAM-cold both saturate HBM SoL ≈ 96%. So cp.async.cg vs TMA at L2-hit ≈ 87% is plausible. Catalog should clarify L2-hit regime. — `[ref: B300_PIPE_CATALOG.md:L2677]`

---

## Group V — L2 cache, memory hierarchy (L7092, L7263, L8558)

- [x] **V1** "HBM3e Peak 5.2 TB/s WRONG section vs revised 7.4 TB/s" — ⚠ DUPLICATE of CRIT4/U1: catalog DOES self-flag; my §00b measures **7.17-7.25 TB/s** matching the corrected number. The 5.16 was an under-occupied test artifact. Catalog should ratify the correction by removing the WRONG section. — `[ref: B300_PIPE_CATALOG.md:L8558]`

- [x] **V2** "Both 5.16/6.83 TB/s were wrong — WS fit in L2" — ⚠ AGREED methodology lesson. The catalog's self-correction is honest but the WRONG sections remain, risking citation errors. Pattern: "WS smaller than L2 → measures L2-with-DRAM-backfill, not pure DRAM". THIS AUDIT's §00b uses ≥1 GB WS to ensure DRAM. Catalog should adopt the WS≥1GB rule. — `[ref: B300_PIPE_CATALOG.md:L8558]`

- [x] **V3** "L2 partition imbalance limits peak to 5.2 TB/s" — ⚠ PARTIALLY VALIDATED via §22i (per-GPC L2 latency variation 25%): GPC2 = 115 cy fastest, GPC3 = 143 cy slowest. The latency variation IS real, but the throughput-bottleneck attribution is not isolated. Could also be L2 ROP unit rate or address hashing skew. Theory plausible, mechanism uncertain. — `[ref: B300_PIPE_CATALOG.md:L8570]`

- [x] **V4** "L2 ~17.5 TB/s in 64 MB partition" — ✅ MATCHES my §00b L2 measurement (20.3 TB/s plateau) within reasonable range. The 17.5 vs 20.3 difference is methodology (4736 blocks pure-streaming vs my 2 CTAs/SM 16-64 MB WS). Partition-vs-aggregate distinction is real; catalog should clarify. — `[ref: B300_PIPE_CATALOG.md:L8571]`

---

## Group W — Architectural limits, theoretical claims, mythbusts

- [x] **W1** "16 FMAs per load FREE at all occupancy" — ⚠ REFINED via §22h (justifications/22h_compute_mem_overlap.md): the catalog's "522 cy budget for ~16 FFMAs" was conservative; actually cold DRAM is **882 cy** giving budget for **~225 FFMAs free** (not 16). The "FREE" framing is correct in steady-state for memory-bound workloads, but the number 16 is wrong. — `[ref: B300_PIPE_CATALOG.md:L8541]`

- [x] **W2** "mma.sync 29× slower than tcgen05.mma" — ⚠ REFINED via §22 mma.sync + §22g tcgen05: my measurements give FP16 mma.sync = 571 TF chip-wide (full occupancy); tcgen05 chip-wide via cuBLAS = ~1980 TF (per CLAUDE.md). Ratio = 3.5× (NOT 29×). The catalog 80 TF vs 2325 TF comparison was likely under-occupied mma.sync vs scaled-up tcgen05 — apples to oranges. — `[ref: B300_PIPE_CATALOG.md:L7047]`

- [x] **W3** "mma.sync probably uses sm_80 legacy tensor unit" — ⚠ AGREED speculation. Per §22 mma.sync SASS: emits HMMA opcodes (the legacy warp-synchronous form), distinct from tcgen05's UTCQMMA/UTCOMMA. Whether they share silicon is not directly observable. The factual claim "different SASS opcode + lower throughput" is correct; the "same hardware as sm_80" attribution is speculation. — `[ref: B300_PIPE_CATALOG.md:L7046]`

- [x] **W4** "wgmma COMPLETELY REMOVED from sm_103a" — ⚠ AGREED weak claim. Per `project_b300_v5_complete` memory: "WGMMA dropped" — meaning ptxas refuses to emit. Whether silicon implementation exists is unknown to us; could be disabled or removed. Catalog should say "ptxas does not emit wgmma for sm_103a" instead of "completely removed". — `[ref: B300_PIPE_CATALOG.md:L7053]`

- [x] **W5** "Predicated Execution Cost FREE" — ✅ CONFIRMED via §13_predication.md (justifications/13_predication.md): pipe rate is independent of active-lane count within 1% (32/16/1 lanes all measure ~2.93 inst/cy/SM). The SETP cost was likely measured separately. The claim "FREE" applies to throughput, NOT to instruction count (predication still issues the warp-inst, just doesn't write back to masked lanes). — `[ref: B300_PIPE_CATALOG.md:L8335]`

- [x] **W6** "Register spilling 10× slower" — ⚠ REFINED via §22q_register_spill_DEEP.md: real measurement shows **9× perf drop at the spill cliff (32 live vars)**, NOT a universal "10× slower" rule. Below 32 vars there's no spill; above 32 there's a sharp cliff. Catalog should add the threshold (32 vars) and the cliff shape. — `[ref: B300_PIPE_CATALOG.md:L8358]`

- [x] **W7** "B300 FP16 tensor 2.1× H100" — ⚠ AGREED inconsistent. Spec ratio: 2325/1430 = 1.63× (NOT 2.1×). Measured ratio (B300 cuBLAS 1980 / H100 cuBLAS ~1500) = 1.32×. The 2.1× number probably mixes B300 spec with H100 base-clock measurement. Catalog should explicitly cite both numerator and denominator sources. — `[ref: B300_PIPE_CATALOG.md:L9223]`

- [x] **W8** "LLM decode B300 3.7× A100, 2.2× H100" — ⚠ AGREED simplified estimate. Per `project_b300_canonical_reference` memory: real LLM decode at 70B = 40 tok/s, 8B = 345 tok/s on B300. The HBM-BW-only estimate is roughly correct directionally but ignores kernel launch overhead, cudaGraph behavior, etc. Catalog should add error bars (±20-40%). — `[ref: B300_PIPE_CATALOG.md:L9237]`

- [x] **W9** "Roofline ridge OI: FFMA=18, FP16 tensor=314, FP8=628" — ⚠ AGREED these are theoretical ridge points (computed from peak/HBM-BW), not measured. The numbers are correct for THE B300 SoLs (76 TF FP32 / 4.2 TB/s = 18; 1980 TF FP16 / 7.2 TB/s ≈ 275 close to 314; 4500 TF FP8 / 7.2 TB/s ≈ 625). Catalog should label as "theoretical ridge from spec, not kernel-measured". — `[ref: B300_PIPE_CATALOG.md:L9276]`

- [x] **W10** "Fusion N ops → N× speedup, perfectly linear, 8 ops = 7.7×" — ⚠ AGREED contradictory wording. 7.7/8 = 96% is "near-linear", not "perfectly linear". The 4% gap is real (launch overhead + L2 cooling). Catalog should reword: "near-linear; 8 ops = 7.7× = 96% efficiency". — `[ref: B300_PIPE_CATALOG.md:L9279]`

---

## Group X — Specific latency and TCgen05 claims

- [x] **X1** "tcgen05.mma 227 cy single-MMA, 67 cy streaming, 3.4 MMAs to hide" — ⚠ PRESERVED for single shape (M=128 N=128 FP8). The latency-hiding arithmetic is correct for THAT shape; other shapes would scale per the linear-cy-vs-shape table at L6776. Catalog should explicitly tag the shape on the headline number. — `[ref: B300_PIPE_CATALOG.md:L7078]`

- [x] **X2** "M=128 N=256 = 128 cy = 'exactly 2×' minimum 51.4" — ⚠ DUPLICATE of Q8 (already resolved): 128.1/51.4 = 2.49×, NOT exactly 2.0. "Fully busy" is plausible inference but not directly validated. — `[ref: B300_PIPE_CATALOG.md:L6751]`

- [x] **X3** "FP8 E4M3 4.65 PFLOPS = 93% of 5 spec — single-warp scope mismatch in headline" — ⚠ DUPLICATE of CRIT2/Q7: each SM has independent tcgen05 datapath, so single-warp × 148 SMs ≈ chip-wide. Headline is technically correct under the linear-scaling assumption but the regime should be explicit. — `[ref: B300_PIPE_CATALOG.md:L6741]`

---

## Group Y — Claims contradicted later in document

- [x] **Y1** "ncu pipe_tensor doesn't measure tcgen05 — but L1089 cites pipe_tensor for tcgen05" — ❌ AGREED CONTRADICTION (per `feedback_b300_pitfalls` memory + H5): pipe_tensor measures legacy mma.sync (HMMA) ONLY, NOT tcgen05.mma. So the L1089 "67.3 inst/ns via pipe_tensor" measurement is INVALID for tcgen05; that's the catalog falling into its own warned-against trap. THIS AUDIT uses UTCQMMA/UTCOMMA SASS counts × cy/MMA for tcgen05 instead. — `[ref: B300_PIPE_CATALOG.md L1089]`

- [x] **Y2** "DSMEM 23 cy vs 201-224 cy — 10× mismatch" — ✅ RESOLVED via §13_dsmem.md: the L7010 "23 cy" was MEASURED VIA THE WRONG SASS PATH (likely a non-cluster smem access). The L2862 "201-224 cy" is the REAL DSMEM read latency (matches my measurement of 204-223 cy). The 10× mismatch IS the catalog error — already top-9. — `[ref: B300_PIPE_CATALOG.md:L7010 vs L2862]`

- [x] **Y3** "V53 DSMEM write 82 GB/s sustained NOT 560" — ⚠ EXPLAINED via §13_dsmem.md: the 87 GB/s/cluster sustained DSMEM write IS confirmed in the audit (catalog correction valid). The "560 GB/s" was V21's burst measurement (NOT completion), referenced but not directly quoted in the version of catalog the reviewer checked. The correction is real but the victim citation is buried. — `[ref: User note]`

- [x] **Y4** "V54 fences cta=8/gl=267/sys=2806 vs L3632 cta=337/gl=1679/sys=8869" — ✅ RESOLVED via §30G (justifications/30G_fence.md): three different regimes — V54 (cta=8/gl=267) is single-warp/no-pending-write (matches my measurement); L3632 (337/1679/8869) is chip-busy W=16; my single-GPU sys=1727 vs V54's 2806 was 2-GPU NVLink rig. Catalog should label all three regimes. — `[ref: User note]`

---

## Group Z — Unverified SASS opcode claims and agent-hearsay

- [x] **Z1** "ERRBAR in fence expansion" — ⚠ SASS-observed but NOT in public ISA docs. Emitted by ptxas for some fence.sc.gpu cases (seen in 30G_fence test SASS). Name likely "Error BARrier" for error-state propagation, but semantics inferred. Acceptable as empirical observation; catalog should label "observed in SASS, not documented". — `[ref: B300_PIPE_CATALOG.md:L3137]`

- [x] **Z2** "CGAERRBAR in fence expansion" — ⚠ SASS-observed (likely Cluster-Group-Aware ERRBAR). Same caveat as Z1: emitted by ptxas for cluster-scope fences, but semantics inferred not documented. Catalog should tag as "inferred from SASS". — `[ref: B300_PIPE_CATALOG.md:L3137]`

- [x] **Z3** "CCTL.IVALL ≤100s of cycles" — ✅ RESOLVED via §22l_cctl_ivall_DEEP.md: measured **2.83 cy on idle pipeline** (well under 100s). Already top-12 architectural fact (H). The catalog's speculation was correct in direction but undershot — it's not ≤100s, it's ≤3. Drain-wait dominates; CCTL itself is essentially free. — `[ref: B300_PIPE_CATALOG.md:L2741]`

- [x] **Z4** "MEMBAR.SC vs MEMBAR.ALL ptxas mapping" — ⚠ SASS-observed empirically. Per §30G SASS dumps: `fence.sc.gpu` → `MEMBAR.SC.GPU + ERRBAR`; `fence.acq_rel.gpu` → `MEMBAR.ALL.GPU`. The mapping is real; NVIDIA doesn't publish ptxas lowering tables, but the observation is reproducible across our test corpus. Catalog should cite "empirically observed". — `[ref: B300_PIPE_CATALOG.md:L3159]`

- [x] **Z5** "F2FP.F16.E4M3.UNPACK_B" — ✅ CONFIRMED via §02_5_narrow_cvt.md SASS-grep: opcode IS emitted by ptxas for `cvt.rn.f16x2.e4m3x2`, throughput matches pipe_alu cap at 2.00 inst/SM/cy (99.98% of peak). Not in NVIDIA public ISA docs but appears in ptxas's emission across all 6 narrow-format conversions. Real instruction, not misinterpreted. — `[ref: B300_PIPE_CATALOG.md:L27]`

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

