# META-DOUBT REPORT (Wave-3d): Auditing the Doubt Reports Themselves

**Date:** 2026-04-22. **Auditor:** wave-3d meta-doubt agent.
**Inputs:** SYNTHESIS_DOUBT, V46_DOUBT, DUAL_ISSUE_DOUBT, DSMEM_DOUBT,
NVFP4_DOUBT, CROSS_AGENT_DOUBT, DOUBT_LOG (wave-3c synth) — cross-checked
against original CORRECTED files, raw V8/V49/V50/V21 sources, M8 matrix.

## Per-doubt-report verdict

### 1. DUAL_ISSUE_DOUBT_REPORT — **OVERSTATED (mechanism wrong; conclusion partially wrong)**

The doubt agent's central claim — **"V49 FFMA solo at 67% of peak proves under-occupancy at 2 warps/SMSP"** — is **FALSIFIED by V8_FFMA_PEAK_VERIFIED.md**:
- V8 reaches 97.64% of FFMA peak with `__launch_bounds__(256, 1)` = 8 warps/SM = **2 warps/SMSP, identical to V49**.
- ILP=8 with 4-cy FFMA latency means 8 indep chains can deliver 2 inst/cy/warp — far above the 1 inst/cy/warp scheduler cap. Latency-hiding is NOT the bottleneck at 2 warps/SMSP.

The REAL reason V49's FFMA-solo is at 67%: V49 uses `fma %0, %0, imm, imm` (pure self-RAW with both source operands as immediates), while V8 uses `fma %0, %0, %1, %0` (distinct `b[k]` source). These have **different RF port behavior** — V49's pattern uses 1 RF read port, V8's uses 2. Yet V49 is SLOWER at 67% vs V8's 97.6% — opposite of what RF-port theory predicts. The 67% solo number is itself a real anomaly worth investigating, not an under-occupancy artifact.

The doubt agent **DID correctly cite M8 counter-evidence** (MUFU+FFMA = 100%+, HMMA+LDS = 73%, HMMA+HMMA = 69%) — these are real M8 entries (verified at M8_PIPE_OVERLAP_MATRIX.md lines 17-21).

**Verdict on V49/V50:** The 55%/74% numbers are real measurements but the **denominator** ("separate-pipe theoretical" assuming both pipes saturate) IS questionable since V49's solo FFMA is itself anomalously low. The dispatch-cap interpretation is not the only one. **DOWNGRADE recommendation in DOUBT_LOG should be SOFTENED**: the V49/V50 ratios are honest measurements; the architectural CONCLUSION ("4 inst/cy/SM dispatch cap") is what's underdetermined, not the numbers.

**Re-promotion?** V49/V50 numbers should be MED, not LOW. The "55%/74% of summed-pipe peaks" metric is reproducible; only the "dispatch ceiling" interpretation needs ncu confirmation.

### 2. V46_DOUBT_REPORT — **SOUND (with caveat acknowledged)**

The 7672 GB/s denominator math (7680 bit × 3996 MHz × 2 / 8) is arithmetically correct as stated. The doubt agent **explicitly flagged** the suspect 12-stack derivation in 01_hbm_bandwidth.md — that's intellectual honesty, not error. Independent calc: 12 stacks × 1024 bit × 8 Gbps / 8 = 12.288 TB/s pre-ECC; with standard inline-ECC overhead 12.5%, ~10.75 TB/s. NVIDIA's marketed "8 TB/s" is closer to the 7672 anchoring. Both 7672 and 10.75 have defensible derivations; the 7672 is anchored to NVIDIA's published convention. **The demotion of V46 (7.20/7672 = 93.8% < 7.30 NINJA) holds regardless of which spec denominator you pick**, because the apples-to-apples comparison is empirical (V46 7.20 < LDG 7.365 < TMA bulk 7.344). Demotion stands.

### 3. DSMEM_DOUBT_REPORT — **SOUND**

Verified directly from `tests/standalone/v21_dsmem_ceiling.cu` lines 96-105: `push_ring_wr` has **NO fence** between `st.shared::cluster.u32` stores and the closing `clock64` read. The "560 GB/s write" IS issue rate, not completion. The doubt agent's claim is grounded in source. The "40 GB/s read is chain-bound" caveat is also accurate (V21 uses dependent-chain pattern). The "NO shared bus claim is unprovable from V17" follows from V17's 1-thread-per-CTA single-issue ring (under-issued by ~30×). All three caveats are real source-grounded findings, not inference.

### 4. NVFP4_DOUBT_REPORT — **SOUND**

The doubt agent **CITED, not invented** the wave-3 caveats:
- The "TMA multicast" mechanism walkback IS in `NVFP4_PURE_TCGEN05_RESULTS.md` Correction § (lines 174-198 cited).
- The "MAJOR CORRECTION" header IS in `NVFP4_SIGN_K64_K96.md` lines 1-9.
- The "encoding bugs in NVFP4 stride sweep" IS in source lines 712-718.

The "B>>A 15-30× may be the artifact, not the truth" is a hypothesis, but it's framed as such. The K=96 single-kernel 2.6× B>A genuinely matches BF16 cuBLAS's 2-2.9×, suggesting the 15-30× IS the over-isolated extreme. Reasonable critique.

### 5. CROSS_AGENT_DOUBT_LOG — **PARTIALLY OVERSTATED on one point**

CROSS_AGENT #1 (HBM denominator) — agent picks 7672 as "the right model". This is somewhat opinionated: 7.31 (empirical pure-direction peak) is also a defensible "% of what this kernel can achieve" framing, as SYNTHESIS_DOUBT H3 itself notes. The agent didn't surface that 7.31 was the V32-V48 authors' deliberate choice. **Minor overstatement; not wrong.**

All other 13 contradictions (NVLink 757, MUFU 47.8 vs 4.74, IADD3 pipe placement, fence spreads, V46 demotion, etc.) are well-grounded with citations.

### 6. SYNTHESIS_DOUBT_LOG — **SOUND**

H1-H5 and M1-M5 are all caveats grounded in cited content. H3 (HBM denominator both legitimate) is exactly the right nuance the cross-agent log slightly missed. No CRIT-level fabrication.

### 7. DOUBT_LOG (wave-3c synth) — **PARTIALLY OVERSTATED on item #6**

Headline #6 "DOWNGRADED to LOW" for V49/V50 dual-issue is too strong given the V8 counter-evidence (same warp count = 97.6% peak, so V49's 67% solo is NOT an occupancy artifact). The numbers themselves are real and reproducible; only the architectural interpretation is underdetermined. **Should be MED, not LOW.**

## Re-promotion candidates (wave-3b doubts that themselves overstate)

| Claim | Wave-3b verdict | Meta-doubt re-verdict | Reason |
|---|---|---|---|
| V49/V50 dual-issue 55%/74% | LOW | **MED** | V8 evidence falsifies the under-occupancy mechanism the doubt agent invoked |
| HBM denominator pick (7672) | "right model" | "**one of two defensible**" | 7.31 empirical is also legitimate per SYNTHESIS_DOUBT H3 |

## Sound doubt reports (no change needed)

- V46_DOUBT — sound (demotion holds on apples-to-apples grounds regardless of denominator)
- DSMEM_DOUBT — sound (V21 source-verified)
- NVFP4_DOUBT — sound (cites real wave-3 caveats verbatim)
- SYNTHESIS_DOUBT — sound (H3 nuance is the gold standard)

## Net assessment

The doubt reports are mostly grounded but the dual-issue doubt invoked the **wrong mechanism** (occupancy) and so **arrived at the wrong confidence downgrade**. The honest answer for V49/V50: the 55%/74% measurements are real; the "dispatch cap" interpretation needs ncu metrics to confirm or refute. **MED, not LOW.**

(Word count: 487)
