# DOUBT LOG v2 (Wave-5 Synthesis with Dual-Issue Case Study)

**Date**: 2026-04-22.
**Inputs**: Original DOUBT_LOG.md (wave-3c) + SASS_VERIFY_DUAL_ISSUE.md (5a)
+ META_DOUBT_REPORT.md (3d/W4) + DUAL_ISSUE_DOUBT_REPORT.md (3b)
+ WAVE5_CHANGES.md (5f) + V41_V48_FINDINGS.md (the original V49/V50 source).
**Purpose**: preserve the wave-3c synthesis verbatim **and** add a permanent
case-study record of the 4-level zigzag through which the V49/V50 dual-issue
claim was promoted, demoted, re-promoted, and re-demoted across waves 1-5.

Originals NOT modified. This file is the wave-5 doubt-aware overlay.

---

## CASE STUDY: The 4-Level Dual-Issue Zigzag

The single most instructive episode of this rigor sweep is what happened to
the V49/V50 "same-warp dual-issue = 55%, warp-specialized = 74%" claim. It
was graded HIGH → LOW → MED → LOW across **five** waves of analysis, and at
no point did the underlying measurement change. Only the **interpretive
frame** changed. The lesson: a single-pass verdict — even a careful one — is
not enough for high-stakes architectural claims.

### The zigzag table

| Wave | Verdict | Mechanism cited | What was right | What was wrong |
|---|---|---|---|---|
| **W1+W2** (V49 = `501134a`, V50 = `fbe1c18`, V41-V48 doc) | **HIGH** — 55% same-warp / 74% warp-spec, presented as architectural finding ("B300 ALU pipes share scheduler dispatch") | Reproducibility of the kernel-internal clock64 ratio across multiple OP variants | The **measurement itself** was reproducible; the relative ordering (warp-spec > same-warp) is real | Reproducibility ≠ validity. No SASS audit, no ncu cross-check, no separate occupancy sweep, no comparison vs an FFMA-saturating baseline |
| **W3b** DUAL_ISSUE_DOUBT_REPORT | **LOW** — "FFMA solo is itself only 67% of peak; the ratio's denominator is broken" | Under-occupancy at 2 warps/SMSP × 8 ILP, latency-hiding limit | Right verdict: the published ratios are not what they appear to be. M8 counter-evidence (MUFU+FFMA ≈ 100%, HMMA+LDS 73-96%) genuinely contradicts the "4-wide dispatch cap" interpretation | The cited mechanism ("under-occupancy") was wrong. V8 reaches 97.6% FFMA at the **same** 2 warps/SMSP geometry, so under-occupancy alone cannot explain V49's 67% solo |
| **W4** META_DOUBT_REPORT | **MED (RE-PROMOTE)** — "V8 falsifies the under-occupancy hypothesis at identical warps/SMSP; numbers are honest, only interpretation is open" | V8 vs V49 occupancy parity → under-occupancy is not the bug → numbers should be MED, not LOW | Right falsification: W3b's specific mechanism IS wrong. The V8 counter-example is real | Wrong verdict. W4 confused "the **denominator** comparison V8 vs V49 fails for under-occupancy" with "V49 measures what it claims". The two kernels differ in **methodology**, not just occupancy. Confusing different test contexts as if only one variable differed |
| **W5a** SASS_VERIFY_DUAL_ISSUE | **LOW (RE-DOWNGRADE, third independent reason)** — "V49 inner body = 8 FFMA + 8 LOP3 + BRA; loop overhead (BRA + UIADD3 + UISETP) eats the same ALU dispatch slots being measured. V8 unrolls 128-deep, amortizing the branch ~16×" | SASS-level inspection of the inner body of both kernels | Correct mechanism: methodology divergence, not occupancy or RF ports. The two kernels are not measuring the same thing | Still a hypothesis until V52 actually runs with V8-style 128-unroll + ncu pipe_fma + pipe_alu simultaneously. Architectural question (does B300 SMSP dual-issue FMA + ALU?) **remains OPEN** |

The W3b LOW verdict was correct **for the wrong reason**. The W4 re-promote
was wrong **for the right falsification**. The W5a re-downgrade is correct
**for a third, independent, mechanism-grounded reason** — but is itself
still hypothesis until V52 runs.

### Lesson 1 — Reproducibility-only verdicts (W1+W2) miss methodology

W1+W2 graded the V49/V50 numbers HIGH because the kernel ran cleanly and
the ratios were stable across reruns. **Stability of a measurement is not
evidence that the measurement measures what its label claims**. The V49
inner body was contaminated by branch/loop-counter dispatch, but the
contamination was deterministic — so it produced a perfectly stable wrong
answer. A single-source HIGH grade is fragile; promote only after
methodology audit.

### Lesson 2 — Mechanism-inference verdicts (W3b "under-occupancy") can be falsified by counter-example

W3b correctly noticed something was off (denominator broken), then
**inferred** the mechanism (occupancy). The mechanism was check-able: is
there a kernel that hits high FFMA throughput at the **same** geometry?
There was — V8. W4 ran the comparison and the inferred mechanism failed.
Always state the mechanism in falsifiable terms so the next wave can test
it directly.

### Lesson 3 — Counter-example verdicts (W4 meta-doubt) can confuse different test contexts

W4 had the right falsification (W3b's mechanism is wrong) but wrong
verdict (re-promote V49/V50 to MED). The error: treating V8 and V49 as
"two kernels at identical occupancy" when in fact they differ on
**multiple** axes — V8 has 128-deep inner unroll and `__launch_bounds__(256, 1)`,
V49 has 8-deep with branch every 8 ops and `__launch_bounds__(128, 2)`.
Same occupancy ≠ same methodology. A single matched variable does not
license a re-promote unless all OTHER variables also match.

### Lesson 4 — SASS-level verdicts (W5a) are structurally soundest but still hypothesis until ncu confirms

W5a inspected the actual emitted instructions and found a concrete,
mechanism-grounded reason: V49's inner body is contaminated by branch
dispatch in a way V8's is not. This is the strongest of the four
verdicts because it is grounded in the **actual code** rather than in a
ratio or an inferred mechanism. **But it is still hypothesis** until
V52 reruns with V8-style methodology AND ncu `sm__inst_executed_pipe_fma`
+ `sm__inst_executed_pipe_alu` simultaneously. SASS inspection narrows
the hypothesis space; only ncu confirms which hypothesis is right.

### Meta-lesson — nest doubt at least 3 levels for high-stakes architectural claims

Each wave caught a real bug in the previous wave's reasoning. If the
sweep had stopped at W1+W2 (HIGH), B300_TRUE_REFERENCE would have been
wrong. If it had stopped at W3b (LOW for under-occupancy), the cited
mechanism would have been wrong, and a sub-agent reading the reference
might have proposed an under-occupancy fix that didn't help. If it had
stopped at W4 (MED), the catalog would have re-published a contaminated
measurement as canonical. Only by going to W5a (SASS-level) did the
real mechanism emerge — and even W5a admits it is hypothesis until ncu.

**Default rigor protocol for any architectural claim that affects
B300_TRUE_REFERENCE: at least 3 nested levels of doubt
(promotion → mechanism doubt → counter-example doubt → SASS / ncu
verification), and explicitly mark the verdict OPEN until the deepest
level is actually executed, not just proposed.**

---

## RIGOR PROTOCOL UPDATE — derived from the dual-issue zigzag

The 4-level zigzag exposes specific, repeatable bugs in microbench
methodology. The following rules are now mandatory for any claim of the
form "pipe X reaches Y% of theoretical Z" or "co-issue of pipes X+Y
yields Q% of summed peak":

1. **Inner body must amortize loop overhead — minimum 64 ops/type per iter.**
   V49's 8 FFMA + 8 LOP3 inner body let branch + loop-counter
   (UIADD3 + UISETP) consume ~10-15% of ALU dispatch slots. V8's
   128-deep inner amortizes branch overhead by ~16×. Below 64 ops/type,
   a "dual-issue measurement" measures branch contamination as much as
   it measures dual-issue. Hard floor: 64 ops per pipe per inner iter.

2. **Solo and dual baselines must use IDENTICAL methodology.** Same
   unroll depth, same `__launch_bounds__`, same warps/SMSP, same
   anti-DCE strategy, same registers-distinct-or-not. V49's solo
   FFMA ran at 67% but V8's solo FFMA ran at 97.6% at the same
   occupancy because the **inner body shapes** differed. Comparing V49's
   dual against V8's solo — or against any abstract "theoretical
   peak" derived from an arch-spec sheet — is comparing across a hidden
   methodology gap. Always run a **matched solo control** in the same
   harness as the dual test.

3. **Always SASS-verify inner body composition before reporting %.** Run
   `cuobjdump -sass` (or look at `sass/<basename>_<hash>.sass`) and count
   the actual emitted instructions in the hot loop. If the inner body
   contains UIADD3 / UISETP / BRA / LDC / IMAD.MOV.U32 etc. that are NOT
   the pipe being measured, those count against the dispatch budget.
   Source-level `#pragma unroll N` doesn't guarantee SASS-level unroll;
   the compiler may re-roll. Verify the SASS, not the .cu.

4. **Always cite ncu pipe_fma + pipe_alu (and pipe_lsu where applicable)
   simultaneously for dual-issue claims.** A single pipe metric cannot
   prove dual-issue. The diagnostic signature of true dual-issue is
   `pipe_fma.pct + pipe_alu.pct > 100%`. The diagnostic signature of
   serial issue (no dual-issue) is `pipe_fma.pct + pipe_alu.pct ≈
   100%`. V49/V50 collected NEITHER metric. Until both are collected
   AND summed, no dual-issue claim is anything but a guess about which
   side of the 100% line you fall on.

5. **Cross-check against published literature.** Hopper (sm_90) has the
   same SMSP dispatch architecture as B300 (sm_103); academic and
   independent measurements of H100 show summed `pipe_fma + pipe_alu` >
   100% in well-formed dual-issue tests. If a B300 measurement claims
   "dispatch is 4-wide per SM regardless of pipe" (i.e. summed pipes
   never exceed 100%), that contradicts the H100 literature. Either
   B300 is architecturally different from H100 in a documented way
   (cite the doc), or the measurement is broken. The default
   expectation should be "B300 behaves like H100 unless proven
   otherwise".

---

## FUTURE-PROOFING — what could overturn wave-5a?

W5a's verdict (LOW, methodology contamination) is hypothesis. It will be
either confirmed or overturned by V52 (the proposed clean re-run). The
following outcomes are pre-registered so the verdict is unambiguous:

| V52 outcome | Implies | Action on canonical |
|---|---|---|
| **`pipe_fma.pct + pipe_alu.pct > 100%` with 128-unroll body, matched launch_bounds, ncu confirmed** | True dual-issue exists at the SMSP. V49/V50's 55%/74% were measurement artifacts of branch contamination. The architectural claim "B300 SMSP dual-issues FMA + ALU" is **PROMOTED to HIGH**. | Add row to B300_TRUE_REFERENCE: "B300 SMSP supports FMA + ALU dual-issue; sustained sum ≥ X% with proper unroll." Retire the 55%/74% numbers. |
| **`pipe_fma.pct + pipe_alu.pct ≈ 50-60%` with 128-unroll body** (i.e. dual still capped) | The dispatch ceiling is real; W1+W2's interpretation was right but their numbers were too pessimistic due to body contamination. The architectural claim is **CONFIRMED** and the canonical number should be the V52 figure, not V49's 55%. | Replace 55%/74% in TRUE_REFERENCE with V52's clean number; promote the architectural claim to HIGH. |
| **ncu pipe metrics disagree with kernel-internal clock64 throughput**, e.g. clock64 says 80% sum but ncu says 50% sum | Kernel-internal timing is unreliable for this measurement (likely because clock64 doesn't capture stall cycles correctly). All previous V49/V50/V41-V48 dual-issue numbers using clock64 are **suspect**. | Mark all clock64-derived dual-issue numbers UNRESOLVED. Adopt ncu pipe metrics as the canonical methodology. Re-run prior V41-V48 dual-issue tests under ncu. |
| **V52 hangs / ncu fails / reproducibility issues** | Even the "clean" methodology has unknown failure modes. | Defer the architectural verdict; do not publish a number. State publicly that the question is OPEN. |

**Default expectation per the H100 literature**: outcome row 1 is most
likely. If V52 shows `pipe_fma + pipe_alu > 100%`, this CASE STUDY itself
will need a wave-6 update — the W5a verdict (LOW, methodology) was right
about the contamination but **wrong** to imply that the architectural
ceiling is at the V49 number.

---

## Original wave-3c content preserved verbatim below

(Everything from the original DOUBT_LOG.md — section numbering preserved.)

---

## 1. Per-claim verdicts on the 10 wave-1+2 HEADLINE_CORRECTIONS.md headlines

| # | Headline (wave-1+2 text) | Verdict | Doubt source | Notes |
|---|---|---|---|---|
| 1 | NVLink generation is **NVLink-5**, not v7 | **CONFIRMED** | SYNTHESIS_DOUBT (web-confirmed) | NVLink-5 = Blackwell, 1.8 TB/s bidi = 2× NVLink-4 |
| 2 | NVLink spec denominator is **900 GB/s/dir** | **CONFIRMED** | SYNTHESIS_DOUBT, CROSS_AGENT #7 | NVLink + PCIe agents agree |
| 3 | **HBM read SoL = ~7.30 TB/s** at 95% of 7672; V46's 98.5% was a denominator artifact | **REFRAMED** | V46_DOUBT, SYNTHESIS_DOUBT H2/H3, CROSS_AGENT #1+#10 | The 7.20 TB/s measurement is **honest**; V46 itself runs a 4-pt sweep avg-of-5. The "demotion" is purely the denominator change (7.20/7672=93.8% vs 7.20/7.31=98.5%). The architectural lesson "TMA reads need 8-deep pipelining" remains valid. SYNTHESIS_DOUBT also flags "BELOW V44/V45" as the wrong comparators (V44/V45 are SMEM, not HBM). |
| 4 | MUFU 47.8 G is 1-CHAIN latency-bound, NOT pipe-saturated; saturated = 4.74 G | **REFINED (wording)** | SYNTHESIS_DOUBT H1 | M14/M16 reported a real measurement of a real regime; the "10× mislabel" framing reads as if the original number was wrong. Better wording: "M14/M16 row is 1-chain latency-bound, not pipe-saturated; saturated peak is 10× lower at 4.74 G". |
| 5 | **IADD3 lives on the FMA pipe** (V40 / d1d09c5), not on a separate ALU pipe | **CONFIRMED** | CROSS_AGENT #6 | All 4 agents that mention IADD3 agree. Clean retraction of older "ALU pipe" framing. |
| 6 | **Same-warp dual-issue = 55%; warp-spec = 74%** (V49/V50) | **DOWNGRADED to LOW** (W3c) → MED (W4) → **LOW** (W5a, FINAL — see CASE STUDY above) | DUAL_ISSUE_DOUBT (entire report) + META_DOUBT + SASS_VERIFY | See top-of-file CASE STUDY for full zigzag. Net: LOW until V52 reruns with 128-unroll inner body + ncu pipe_fma + pipe_alu simultaneously. The architectural question (does B300 SMSP dual-issue FMA + ALU?) **remains OPEN**. |
| 7 | **All V8/V10 DSMEM TB/s peaks were DCE artifacts**; real read 40 GB/s/cluster, write 560 GB/s/cluster | **CONFIRMED with caveats** | DSMEM_DOUBT | V8 DCE retraction is HIGH (SASS-verified). 40 GB/s read is **chain-bound**, not absolute (a non-chained ILP test might reach 60-80 GB/s). 560 GB/s write is **issue rate**, not completion (no fence between stores and clock64 end); real delivery rate may be lower. "NO shared bus" claim is **under-issued by 30×** in V17 — test cannot rule out a shared bus. The 7.5× local/DSMEM **latency** ratio is HIGH. Also: SYNTHESIS_DOUBT H4 notes v1's 3.06 TB/s "aggregate" was actually per-cluster 41 × 74 ≈ 3 TB/s **chip aggregate** — internally consistent with v2's per-cluster 40 GB/s; the "mixed measurements" framing was wrong, the disagreement was scope (per-cluster vs chip-aggregate). |
| 8 | **A vs B operand power has 3 different "correct" answers** depending on test geometry (cuBLAS A>B 3:1; pure tcgen05 B>>A 15-30×; K=96 single-kernel B>A 2.6×) | **REFINED — preserve all 3** | NVFP4_DOUBT | The 3 source readings are all real and traced. Wave-2's "TMA multicast halves B's memory cost" mechanism IS supported by an explicit ncu table (multicast 0% BF16 vs 78% NVF4). BUT the underlying source itself (`NVFP4_PURE_TCGEN05_RESULTS.md` Correction §) lists 4 plausible mechanisms and walks back the single-mechanism story. Wave-2 over-resolved. **Surface all 3 readings; don't pick one.** Notably, the K=96 single-kernel 2.6× matches BF16 cuBLAS's 2.0-2.9× — suggests pure-tcgen05's 15-30× is the **artifact** (over-isolation), not the truth. |
| 9 | **3-source FFMA caps at ~50 TFLOPS (65%)** due to RF port pressure | **CONFIRMED** | A_TO_D_RIGOR_AUDIT, V46_DOUBT (no challenge) | Cleanly cross-validated: A4 + D6 + V10_FMA_SOURCE_COUNT (75.2 vs 51.3 = ratio 0.683 ≈ 2/3, exactly 2-RF-port prediction). |
| 10 | NVFP4 cuBLAS+cudaGraph BPG=16 = **11.42 PF (76.2%)** supersedes 10.8 PF | **REFINED — single-shape** | SYNTHESIS_DOUBT M2, NVFP4_DOUBT #2, CROSS_AGENT #4 | Number is real (`NVFP4_CUDAGRAPH.md`, 200-iter sustained). But it is **one shape** (8K² K=38400) at one BPG value — not a sweep. Treat as **upper-bound at this shape**, not sustained ceiling. Tensor agent's stale 10.8 PF should defer to NVFP4 agent's 11.42, but both are correct in their own context. |

---

## 2. Top-10: HEADLINES THAT SURVIVED DOUBT (HIGH confidence)

| # | Claim | Source |
|---|---|---|
| 1 | NVLink-5 (NOT v7); spec 900 GB/s/dir | NVLink + PCIe agents, web-confirmed |
| 2 | IADD3 lives on FMA pipe (V40, d1d09c5) | All 4 agents agree |
| 3 | 3-source FFMA = ~50 TFLOPS (~65% of 2-source) | A4 + D6 + V10_FMA_SOURCE_COUNT |
| 4 | V8/V10 DSMEM TB/s = DCE artifacts (real read 40 GB/s/cluster *chain-bound*) | DSMEM_DOUBT confirms |
| 5 | DSMEM is **7.5× slower than local SMEM** (latency) | V12/V15/V16 cross-test consistent |
| 6 | `pipe_tensor.cycles_active` does NOT measure tcgen05 | TENSOR log §B |
| 7 | L2 = **126 MB** (not 50/96/192/256); "96 MB" cosmetic error in 4 files | STRAYS §7 |
| 8 | TMEM = ~60 TB/s read (not 295/830 — those were DCE) | 06_tensor + V4 D7 |
| 9 | Random data is up to 43% slower than zero data for FP8 cuBLAS under power cap | TRUE_REF v1 row 68 |
| 10 | Power d=16 random popcount = 240-554 W swing (HBM_DATA_DEPENDENCE.md's <50 W is SUPERSEDED) | STRAYS §2, POPCOUNT 4-file family |

## 3. Top-10: HEADLINES THAT NEED REVISION (MED–LOW confidence)

| # | Claim | Issue | Recommendation |
|---|---|---|---|
| 1 | "V46 = NEW HBM read SoL at 98.5%" | Denominator artifact (used 7.31 empirical instead of 7672 spec) | Reframe as "V46 is honest 7.20 TB/s measurement; 7.20/7672 = 93.8%; below 7.34 (TMA bulk), 7.365 (LDG), 7.30 (NINJA). Architectural lesson on TMA pipelining stays." |
| 2 | "Same-warp dual-issue = 55%; warp-spec = 74% (FFMA+LOP3)" | Baseline is 67%-of-peak; no ncu metrics; M8 counter-evidence; **W5a SASS shows methodology contamination** | **DOWNGRADE to LOW (W5a final)**. Re-run as V52 with 128-unroll inner body + matched launch_bounds + ncu pipe_fma + pipe_alu before promoting. |
| 3 | "DSMEM 40 GB/s read aggregate" | Chain-bound (V21 dependent-chain test); non-chained ILP could reach 60-80 GB/s | Annotate as "chain-bound ceiling, not absolute asymptote" |
| 4 | "DSMEM 560 GB/s write aggregate" | Issue rate, not completion (no fence between stores and clock64 end) | Annotate as "issue rate ceiling; real delivery rate unverified" |
| 5 | "NO shared bus" (DSMEM) | V17 ring test was **30× under-issued** (1 thr/CTA, single-issue chained) | Demote to "consistent with point-to-point per architecture; not proven by V17" |
| 6 | "TMA multicast halves B's memory cost" (NVFP4 A:B story) | Underlying source itself lists 4 plausible mechanisms; wave-2 picked one | Preserve all 3 A:B readings (cuBLAS, pure-tcgen05, K=96 single-kernel) |
| 7 | "NVFP4 cuBLAS 11.42 PF supersedes 10.8 PF" | Single-shape, single-BPG (no sweep) | Frame as "upper-bound at 8K² K=38400 BPG=16; sustained ceiling needs sweep" |
| 8 | "MUFU mislabeled by ~10×" | Number was real measurement of real regime, not "wrong number" | Soften to "M14/M16 row is 1-chain latency-bound, not pipe-saturated" |
| 9 | "NINJA STG hit 7.57 TB/s" (HBM write SoL) | Provenance contested with V8 TMA bulk store path | Mark UNRESOLVED until both kernels re-run back-to-back with ncu `dram__bytes` |
| 10 | "Persistent kernel 2.03 µs supersedes v1's 4 µs because v1 used release variant" | Hypothesis about why v1 was higher; no SASS evidence cited | Demote to "hypothesis: v1 likely used release; not verified" |

## 4. New headlines from wave-3 (post-doubt findings)

| # | Finding | Source |
|---|---|---|
| 1 | **HBM denominator is the #1 cross-agent contradiction**: 7672 / 7.31 / 7.2 / 8.0 TB/s appear across 3+ files; ALL "% of peak" cross-doc numbers are not directly comparable | CROSS_AGENT #1+#2 |
| 2 | **System fence cost is UNRESOLVED 1.74× spread**: 1750 cy (08) vs 2870 cy (DSMEM) vs 3042 cy (V9). TRUE_REFERENCE v1 picked 861 ns (=1750 cy) WITHOUT justification | CROSS_AGENT #2+#10, SYNTHESIS_DOUBT L3 |
| 3 | **CURIOSITY_LIST V2 has 88% hallucinated hashes** (22/25); V4-V8 are 100% git-verified | CURIOSITY_LISTS_AUDIT |
| 4 | **TCGEN05_PERF_WATTS single-trial table is contaminated** (5 leftover QuickRunCUDA processes); use TCGEN05_PERFW_CLEAN_2TRIAL. NVFP4 K=96 = 12.54 TF/W random / 15.74 best (NOT 13.72) | TCGEN05_POWER_CONSOLIDATED §2 + R1 |
| 5 | **HBM_DATA_DEPENDENCE.md is SUPERSEDED**: real DRAM data-dep swing is 240-554 W, NOT <50 W (that was constant-pattern-only) | STRAYS §2 |
| 6 | **A6's 4-tier pipe ladder (FMA/INT-bit/permute/compare) supersedes "unified ALU/FMA cluster"** model | A_TO_D_RIGOR_AUDIT #5 |
| 7 | **L2 BW must be labelled** with one of {kernel-effective ≈24, wire-lts ≈13, L1-amplified ≈30} TB/s; bare numbers float | CROSS_AGENT #4 |
| 8 | **Atomics SMEM aggregate is 2.27 T (not 4.2 T)**; CLAUDE memory's 4.2 T is unsourced | CROSS_AGENT #12 |
| 9 | **L2 atomic units count "~32" should be MEDIUM not HIGH** (derived ceiling, not direct measurement) | STRAYS §8 |
| 10 | **Wave-2 took credit for upstream retractions** (BF16 1543 / FP8 7500-8200 / BF16 90.5%): all already self-retracted by their original docs | SYNTHESIS_DOUBT H5 |

---

## 5. Confidence ladder summary (by topic)

| Topic | Confidence | Reason |
|---|---|---|
| HBM read peak (7.30 TB/s @ 95% of 7672) | HIGH | Multi-source NINJA/TMA/LDG agree; denominator stated |
| HBM write SoL (7.57 TB/s) | MED — contested provenance | NINJA STG vs V8 TMA bulk |
| L2 BW (with metric tag) | HIGH | Cache agent disambiguates 3 metrics |
| TMEM read (~60 TB/s) | HIGH | DCE-corrected |
| SHMEM peak (38.4 TB/s) | HIGH | 99.8% of spec |
| DSMEM latency (7.5× slower than local SMEM) | HIGH | V12/V15/V16 cross-test |
| DSMEM read aggregate 40 GB/s/cluster | MED — chain-bound | V21 dependent-chain only |
| DSMEM write aggregate 560 GB/s/cluster | LOW-MED — issue rate | No fence in V21 |
| DSMEM "NO shared bus" | LOW | V17 30× under-issued |
| FP32 FFMA peak (75.9 TFLOPS, 2-source) | HIGH | Multi-recipe |
| FP32 FFMA realistic (50 TFLOPS, 3-source) | HIGH | A4/D6/V10 agree |
| Same-warp dual-issue 55% / warp-spec 74% | **LOW (W5a final)** | SASS shows methodology contamination; needs V52 + ncu pipe_fma + pipe_alu |
| IADD3 on FMA pipe | HIGH | All 4 agents agree |
| MUFU saturated 4.74 G/chip (NOT 47.8) | HIGH | V41 pipe rate; framing only |
| NVFP4 11.42 PF cuBLAS+graph | MED — single shape | No BPG sweep |
| NVFP4 K=96 ULTRA 10.91 PF (98.5% per CTA) | HIGH | TF/W cross-validated |
| NVFP4 A:B 3-way (cuBLAS / pure / K96) | MED — preserve all 3 | Wave-2 over-resolved single mechanism |
| Tcgen05 perf/W 2-trial table | HIGH | PERFW_CLEAN_2TRIAL supersedes contaminated single-trial |
| Power d=16 popcount bell curve | HIGH | 4-file family agrees |
| HBM_DATA_DEPENDENCE.md "<50 W" | RETRACTED | Superseded by POPCOUNT family |
| NVLink-5 spec 900 GB/s/dir | HIGH | Web-confirmed |
| `__threadfence_system` cost | UNRESOLVED 1.74× spread | 1750/2870/3042 cy |
| `__threadfence` (GPU) 281 cy | MED — 4-way 24% spread | Sync agent preserves; DSMEM picks 320 silently |
| Persistent kernel 2.03 µs | MED — mechanism unverified | Number real; "v1 used release" is hypothesis |
| L2 atomic units ~32 | MED (downgraded from MED-HIGH) | Derived ceiling, not direct |
| SMEM atomic aggregate 2.27 T | HIGH | Atomics + SHMEM agree (CLAUDE memory's 4.2 T unsourced) |
| `pipe_tensor.cycles_active` doesn't measure tcgen05 | HIGH | TENSOR log §B |

---

## 6. Net assessment of wave-1+2 synthesis

The wave-1+2 synthesis is **broadly faithful** but:

- **Took credit for some upstream retractions** (1543/7500-8200/90.5%) that the originals already made.
- **Prematurely picked one denominator** in the HBM debate (7672 is right but 7.31 is also defensible for "% pure-direction").
- **Conflated measurement scopes** in the DSMEM 3.06 TB/s supersession (v1 was per-cluster × ~74 = chip aggregate, internally consistent with v2's per-cluster 40 GB/s).
- **Flattened 30% UNRESOLVED gaps** into headline single numbers (IADD3 0.5 vs 0.66; PRMT 0.36 vs 0.5).
- **Promoted V49/V50 dual-issue 55%/74% to HIGH** without acknowledging the under-occupied baseline or M8's counter-evidence — and as the W5a SASS-verify pass shows, the actual mechanism is methodology contamination (loop overhead in a tiny inner body), not occupancy. See top-of-file CASE STUDY for full chronology.

No CRIT-level fabrication detected.
