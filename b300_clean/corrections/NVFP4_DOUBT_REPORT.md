# NVFP4 Wave-2 Doubt Report (Adversarial Audit)

Date: 2026-04-22. Reviewer: doubt agent.

## Per-claim verdicts

### 1. A:B 3-way reconciliation — PARTIALLY SOUND
The wave-2 reconciliation (cuBLAS A>B, pure-tcgen05 B>>A 15-30×, K=96 single-kernel B>A 2.6×) maps to three real source files and the numbers are not invented. **The "TMA multicast halves B's memory cost" is NOT an inferred guess** — `NVFP4_POWER_DECOMPOSITION.md` lines 209-243 contains an explicit ncu table showing `TMA read bytes MULTICAST: 0 (BF16) vs 3.75 GB / 78% (NVF4)`. The hypothesis is hardware-verified.

**However the agent over-resolves**: `NVFP4_PURE_TCGEN05_RESULTS.md` lines 174-198 ("Correction") explicitly walks back the original "definitively explained by TMA multicast" claim and lists 4 plausible mechanisms (multicast, A↔B swap inside cuBLAS, SMEM dwell time, pipeline depth). The wave-2 consolidation in `NVFP4_CONSOLIDATED.md`/`NVFP4_INCONSISTENCY_LOG.md` reverts to the single-mechanism story without surfacing the source's own caveat. Verdict: **partially overstated**. The 3-way numbers are real; the explanatory unification is one notch more confident than the source warrants.

### 2. "11.42 PF cuBLAS+cudaGraph supersedes 10.8 PF" — SOUND but THIN
`NVFP4_CUDAGRAPH.md` shows 11423 TF at BPG=16, 76.2% MFU. **Confidence note in source itself**: "single best measurement, 200 iter sustained". This is one shape (8K² K=38K) at one BPG value — not a sweep. The "exceeds 73.1% model ceiling by 3pp" claim is real and consistent with `NVFP4_CUBLAS_FULL_SWEEP.md`'s 11054 TF plain-Lt. Verdict: number is real but **single-measurement; treat as upper-bound, not sustained ceiling**. Memory note's "10.8 PF" is from earlier shape (~16K K). Both numbers are correct in their own context.

### 3. "K-uniform-per-N 28% retraction" — SOUND
`NVFP4_SIGN_K64_K96.md` lines 1-9 contain an explicit in-document **MAJOR CORRECTION** header. Wave-2 agent CITED, not invented, the retraction. Mechanism (background process contamination raising baseline ~100 W) is documented in CLAUDE memory `feedback_clock_stuck_no_lock`. Verdict: **clean retraction, properly cited**.

### 4. "K=96 ULTRA at 1500 MHz = 10.95 PF (73%)" — SOUND but conflating risk
`NVFP4_K96_AT_1500MHZ.md` line 20 confirms 10.89 PFLOPs (98.5% MFU local; ~73% of 15 PF spec). Wave-2 agent did NOT conflate this microbench number with cuBLAS's 11.42 PF — the consolidated table separates them by row. Memory note is consistent.

### 5. NVFP4 has NO 32-element MAC cliff (BF16 has 112 W cliff) — PARTIALLY SOUND
BF16 cliff: real, clean step from 592→480 W at stride 32 (`PURE_TCGEN05_RESULTS.md` line 358). NVFP4 absence: source itself flags **"Within-word strides (1, 2, 4) had encoding bugs"** and **"NVFP4 stride results are NOT trustworthy"** (lines 712-718). The cleaner sign-bit-only retest does test stride boundaries and finds no cliff (line 414). Verdict: **BF16 cliff = HIGH; NVFP4 absence = MED** — the agent's "verified not just untested" framing is too strong; the underlying NVFP4 stride sweep had encoding bugs and only the sign-bit retest is clean.

### 6. Mode 2 (deferred SHFL) 2.96× beats mode 7 (per-tile redux) 1.59× — SOUND
`NVFP4_FULL_PIPELINE.md` shows 7 modes with identical correctness (`facc=16560000.0`). Numbers reproduce. **Caveat correctly captured**: "redux speedup matters only when reduce is structurally per-tile". Apparent contradiction with `NVFP4_INT_REDUCTION.md`'s 2.75× is properly reconciled in the inconsistency log.

## Specific over-resolution flags

- The "TMA multicast" mechanism is treated as settled in the consolidated doc but the underlying file (PURE_TCGEN05 Correction §) lists 3 alternative live hypotheses. **Surface the caveat in CONSOLIDATED §2**.
- "Universal multiplier asymmetry" is universal across the tested precisions but K=96 single-kernel measurement gives 2.6× not 15-30×. The 2.6× in K=96 is closer to the BF16 cuBLAS observation (2.0-2.9× B). The wave-2 framing implies pure-tcgen05's 15-30× is THE truth and others are artifacts; reality is the 2.6× K96 single-kernel matches BF16 cuBLAS, suggesting the 15-30× is the artifact (over-isolation in the A=zero/B=zero mode).

## New tests to settle open questions

1. **Custom NVFP4 kernel WITHOUT TMA multicast** at cuBLAS shape — does A still dominate? (decisive multicast test)
2. **A↔B swap test** — feed identical data through swapped slots; if asymmetry follows the slot, it's hardware; if it follows the data, it's a cuBLAS layout decision.
3. **NVFP4 N-stride sweep with sign-bit-only encoding** at full stride ladder (bug-free) to nail the "no cliff" claim with HIGH confidence.
4. **cudaGraph BPG sweep across 5+ shapes** to confirm 11.42 PF is reproducible, not a one-shot peak.
5. **K=96 ULTRA at non-square N to test K-id shape-conditional restriction** (open Q7).
