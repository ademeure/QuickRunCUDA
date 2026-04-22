# Meta / Strategy Docs — Corrections

**Auditor**: meta-docs swarm agent, 2026-04-22
**Topic**: README.md, TASK_LIST.md, NINJA_ACHIEVEMENTS.md, SESSION_2_DELTA.md,
PRACTICAL_SCOPE_HONEST.md, PRACTICAL_THROUGHPUT_GAIN.md, REALISTIC_INT4_SCOPING.md,
CUBLAS_REAL_VALIDATION.md, CUBLAS_BIT_ENTROPY_CORRECTION.md, B300_TRUE_REFERENCE.md,
LATENCY_DATA_INDEPENDENT.md, CHAIN_PAIR_BYPASS.md.

This file lists CORRECTED stances on every meta-level claim. Originals in
`b300_clean/*.md` are NOT modified per swarm rules. For per-claim provenance,
see `META_INCONSISTENCY_LOG.md` in this same directory.

---

## 0. Trust order (corrected)

The README's "trust order" lists `B300_TRUE_REFERENCE.md` as #1, but this is
out of date as of 2026-04-22. Use:

1. **`corrections/*_CORRECTED.md`** ← apply these on top of TRUE_REFERENCE
2. `B300_TRUE_REFERENCE.md` (rigor-verified 2026-04-18, predates the V11–V51
   sweep; numerous entries refined or contested by later commits)
3. `b300_clean/01_*.md` through `17_*.md` (preferred for per-category drilldown)
4. `M3_REVERIFY_LOG.md` (the 2026-04-18 baseline)
5. AUDIT_NOTES.md > CRITIQUE.md > B300_PIPE_CATALOG.md (legacy)

---

## 1. README.md — index quality

**README is GOOD as an index but stale on 6 dimensions:**

- ✓ Categories 01–17 all link correctly; no dead links.
- ✓ "Top-line peak ladder" matches TRUE_REFERENCE for most rows.
- **HBM write 7.30** in the README ladder is conservative; NINJA achieved 7.57
  TB/s (e75c7e1) per TRUE_REFERENCE §1. README should at least mention NINJA.
- **HBM read 7.30** — README does not cite the V46 8-deep TMA = 7.20 result
  (V41_V48_FINDINGS) NOR the higher 7.344/7.365 already in 01_hbm_bandwidth.
  Per HBM agent, V46's "98.5%" is denominator artifact (uses 7.31 not 7.672);
  V46 is NOT a new SoL.
- **BF16 mma.sync legacy 577 TFLOPS** in README is consistent with the
  S1 RETRACTION (was 2270 with DCE, real is 578). Good.
- **FP8 cuBLAS 4486** in README needs the realistic-data caveat: TRUE_REF row 68
  shows 3984 random / 4393 zero → README's single 4486 is the zero-data only.
- **Cluster RELAXED 50 ns** is correctly flagged "NEW finding from rigor audit".

**Cross-category contradictions queue (README §94)**:
- Item 1 (Concurrent HBM R+W) is RESOLVED by HBM agent — 6.68 TB/s @ 50:50
  is the correct ceiling, "10.4 TB/s" was wall-clock illusion.
- Item 2 (Tensor + FFMA "8× slowdown") — README correctly notes this is
  "proportional SM-share scheduling" not contention.
- Item 3 (PTX-JIT slowdown 155×) — verify per-kernel; some kernels show
  smaller ratio.
- Item 4 (Cluster cost 50 vs 185 ns) — well-resolved.

---

## 2. TASK_LIST.md — commit-hash sanity check

Per CLAUDE.md memory: "Always git-verify commit hashes; prior loop sessions
hallucinated hashes". Without running git, sanity check:

**Format check (all hashes 7-char hex):**
- All ~70 commit hashes in TASK_LIST conform to 7-character lowercase hex.
- No obvious "placeholder" hashes (0000000, deadbef, abcdef0).
- No malformed hashes.

**Internal cross-reference check:**
- C6 and E1 both cite `9467cfe` (deduplicated, marked "SUBSUMED by C6"). ✓
- G2 and G4 both cite `6050ff6` (deduplicated, marked "SUBSUMED by G2"). ✓
- A7, J1, J3 all reference cudaMemset / pageable / VMM — no hash conflicts.

**Suspicious / needs git verify:**
- ⚠ M3 cites `d430675` for "M3_REVERIFY_LOG.md" but log file references 28
  MED→HIGH conversions; user should `git show d430675` to confirm.
- ⚠ The "free-rein" section appends ~30 hashes added in later loop iterations.
  Per memory, these are most likely to be hallucinated. The hashes look real
  (proper hex format) but content claims should be spot-checked:
  - `8aa9149` (axpy SoL), `09e7a19` (IPC), `26f8592` (TMA-LDG parity),
    `bf46464` (CuTeDSL throttle), `3f98880` & `4bdb423` (K-id deepdive),
    `7929d59` (structured B 18%) — these are referenced in MEMORY.md too,
    independently corroborating their existence.
- The N-dependence entry (last bullet) lists 5 hashes but only writes 3
  explicitly (`702d4bd` `3f98880` `4bdb423` plus 2 more) — the "plus 2 more"
  framing is a yellow flag.

**Recommendation**: `git log --oneline f2fp-deep-dive | head -200` should
recover the canonical hash set; cross-reference TASK_LIST hashes against
that list before publishing the table externally.

---

## 3. NINJA_ACHIEVEMENTS.md — best-of-best recipes

### 3.1 HBM write 7.57 TB/s NINJA — **CONTESTED PROVENANCE**

Per `corrections/HBM_INCONSISTENCY_LOG.md` Inconsistency #3:
- NINJA_ACHIEVEMENTS.md attributes **7.57 TB/s to "1 v8 store/warp" STG-based
  ninja kernel** (commit `e75c7e1`).
- `V8_HBM_WRITE_SOL.md` (the same era) attributes **7.57 TB/s to TMA bulk
  store** (commit `28211ce`) and says plain STG caps at 6.11 TB/s.
- Both cannot be true. STG = 7.57 (NINJA story) and STG cap = 6.11 (V8 story)
  are incompatible.

**Resolution**: TRUE_REFERENCE row 27 sides with NINJA. The V8_HBM_WRITE
"STG ceiling = 6.11" is from a sub-optimal launch geometry; the NINJA recipe
(1 store per warp, massive warp parallelism) is the real STG ceiling.
Re-attribute 7.57 to NINJA STG; mark V8's 6.11 as "STG with sub-optimal
launch", not architectural ceiling.

### 3.2 V46 8-deep TMA 7.20 TB/s — should NOT be in NINJA

Per HBM_INCONSISTENCY_LOG #5: V46's claimed "98.5% HBM" is denominator
artifact. 7.20 < 7.30 (NINJA ladder), 7.20 < 7.344 (older 01_hbm_bandwidth
TMA), 7.20 < 7.365 (older LDG.E.128). V46 is an improvement over V33's
6.72 single-deep, NOT a new architectural SoL. **Do NOT add V46 to NINJA**.

### 3.3 BF16 absmax 6.92 TB/s — **REVERIFY**

NINJA claims 6.92 TB/s = 90.21% of HBM read peak (7672). But per the
"asymmetric R+W duplex" finding in NINJA itself, the practical R+W ceiling
is ~7 TB/s. absmax does pure R (no W on weights) → 6.92 is plausible vs
7.30 R-peak. ✓

### 3.4 BF16 softmax "1.5–1.8× speedup" — **CORRECTED**

NINJA originally claimed 91% of HBM peak; per agent critique (TRUE_REF §8),
the correct denominator is the R+W concurrent ceiling 6.68 TB/s, not
pure-read 7.31. Achieved 6.07 TB/s = **91% of R+W ceiling**, not 91% of
read peak. NINJA appears to have been edited to acknowledge this; verify.

### 3.5 mma.sync BF16 multi-chain 100% — **STALE / WRONG**

NINJA's "Unbeatable ceilings" table claims:
> BF16 mma.sync (multi-chain) | ~570 TFLOPS | **100%** (catalog burst confirmed)

But per TRUE_REFERENCE row 47–58 + the S1 retraction in SESSION_2_DELTA:
- 569 TFLOPS at 1920 MHz lock = 92.4% of legacy 616 theoretical
- 578 TFLOPS at 2032 MHz boost = 93.8% of legacy 616 theoretical
- The "100%" framing requires picking 569 / 569 (circular).

**Correct**: ~94% of legacy mma.sync theoretical ceiling, NOT 100%. The
"100%" probably came from circular reasoning ("achieved = ceiling because
that's what we measured").

### 3.6 FP8 cuBLAS 4425 / 88.5% MFU — needs realistic-data caveat

NINJA cites 4425 TFLOPS @ 943 W "via cudaGraph". Per TRUE_REF row 68:
- Zero data: 4393–4425 TFLOPS
- Random data: 3984 TFLOPS
- Realistic Gaussian: 3951 TFLOPS

NINJA's headline is the zero-data number — should add "(zero-data; realistic
~3950)" caveat.

---

## 4. SESSION_2_DELTA.md — what's still valid

The main session-2 findings remain valid:
- ✓ MIO pipe unification (STS/LDS/SHFL/ATOMS/REDUX share per-SM port)
- ✓ REDUX vs CREDUX distinct pipes
- ✓ Pipe matrix (FFMA/MUFU/DFMA orthogonal to MIO)
- ✓ tcgen05 path on B300 (compile recipe)
- ✓ Power decomposition

**REVISED in V11–V51:**
- HMMA chain scaling — section "Pipe matrix HMMA findings ALSO retracted
  (2026-04-19)" already corrects this in-place. ✓
- BF16 mma.sync 90.5% claim — already retracted in-place to 23.1% (vs 2500
  spec) or 93.7% (vs 616 legacy). ✓
- FP8 mma.sync 7500-8200 TFLOPS — already retracted in-place to ~3760 TF
  via SASS-folding diagnosis. ✓
- 32×4 anomaly — partially resolved (clock64 confirms wall-clock truth);
  mechanism still open.
- HMMA power 405 W → 255 W correction — already in-place. ✓

**STALE in SESSION_2_DELTA (V11–V51 supersedes):**
- "tcgen05 path achieves cuBLAS algoId=66 = 90% of FP16/BF16 spec at 8K³"
  is consistent but doesn't account for the data-dependence finding (random
  data drops cuBLAS BF16 to ~1880 TF = 75% spec).
- "Pure HBM read 720 W = +520 W" — re-verified in I4 (commit `bf98e90`):
  random data hits 1100 W cap with FP8 cuBLAS, not 720. The 720 may be HBM-
  read only (no compute); needs precision.

---

## 5. PRACTICAL_SCOPE_HONEST / PRACTICAL_THROUGHPUT_GAIN — optimism check

### 5.1 PRACTICAL_SCOPE_HONEST.md

**Generally HONEST and well-scoped.** The "What works" / "What doesn't work"
table is accurate.

Minor concerns:
- Claims "Custom kernel + structured B + 400W cap | 2.09×" — extrapolation
  from the K-grouped microbench, not directly measured for arbitrary kernels.
- "cuBLAS BF16 + K-row identical + N∈{4096,8192,16384} + M≥256" → 1.40× —
  consistent with N_DEPENDENCE_DEEPDIVE 5-gate model.

### 5.2 PRACTICAL_THROUGHPUT_GAIN.md — **OPTIMISTIC, needs caveats**

The 1.18× / 1.74× / 2.09× speedup claims are real for the SPECIFIC test
kernel (BF16 m128n128k16, microbench mode 200 vs 5208) but the doc:

1. **Does NOT make clear** that "K-grouped structured" is microbench mode
   5208, not arbitrary K-row-similar data. Real ML weights rarely hit the
   exact pattern.
2. **Extrapolates "TFLOPS" from iters × FLOPS-per-iter formula** — admitted
   as MEDIUM confidence at the bottom but the bold-faced summary tables
   present these as measurements.
3. **The "100 GPU cluster equivalent to 174 GPUs" framing** is misleading —
   that requires WORKLOAD = exactly 100M iters of m128n128k16 with structured
   B = mode 5208. Real cluster workloads will not exhibit anywhere near this.

**REVISED honest framing**:
- ~18% gain at default cap — real for synthetic structured workloads.
- ~5–11% gain in real cuBLAS GEMMs (per CUBLAS_REAL_VALIDATION's Llama
  shape sweep, after restricting to actually-applicable shapes).
- 1.74× at 600W is real for microbench but for cuBLAS see CUBLAS_REAL_VAL
  table — 1.55–1.78× at 600W is the realistic range.

### 5.3 REALISTIC_INT4_SCOPING.md — **CORRECTLY CONSERVATIVE**

This doc IS honest, in fact it self-corrects multiple times:
- "12-18% synthetic" → "1.04× realistic Gaussian" → "4-6% across power caps"
- The per-power-cap sweep is consistent with "tighter caps don't amplify
  for realistic data".
- Final 4–6% claim is the right conservative number.

✓ Keep as-is. This doc is the model for the OTHER practical docs.

---

## 6. CUBLAS_REAL_VALIDATION / CUBLAS_BIT_ENTROPY_CORRECTION

Per tensor agent finding "FP8 random 3983 vs zero 4393":

### 6.1 CUBLAS_REAL_VALIDATION.md — **NEEDS REAL-DATA CAVEAT**

The headline "1.45× speedup; 2201 TFLOPS K-id; 1517 TFLOPS random" is
correct for BF16 8K³. But the "random" baseline IS Gaussian-like normal-
only random, NOT zero-data.

The "structured" peak 2201 TFLOPS is **97.7% of BF16 cuBLAS spec 2242**, AND
**97.7% of zero-data ceiling 2252** — so the K-row identical pattern is
nearly hitting the zero-data hardware ceiling, but via a different mechanism
(K-row dedup). This is consistent. ✓

What needs caveating:
- The "2143 TFLOPS K-row identical" assumes K-rows are LITERALLY identical.
  Realistic graceful-degradation (4-bit residual = INT4-like) is 1.20×, NOT
  1.41×.
- Per Llama shape scan: only N∈{4096,8192,16384} get the 1.41×. Llama's
  actual N values (28672, 10240, 14336) get **1.01× = NOTHING**.
- Llama down projection (large K) gets 1.13–1.40× depending on N.

Headline 1.45× is for the BEST CASE shape (8192³ K-id). For realistic
inference, the GPTQ test in this same doc shows **1.07× for Llama 70B
gate/up, 1.13× for Llama 70B down**. **THOSE are the deployment numbers.**

### 6.2 CUBLAS_BIT_ENTROPY_CORRECTION.md — **CORRECTLY ATTRIBUTES**

This doc correctly applies rule #9 to determine the speedup is bit-entropy,
not specifically K-row identity. It is well-rigored:
- ✓ Three independent timing methods agree (CUDA Graph / chrono / single-
  matmul) within 1.4%
- ✓ NCU verification: cycle counts identical (0.36% Δ) → mechanism is
  clock-frequency-via-throttle-avoidance, not cycle savings
- ✓ Cross-precision (FP16/BF16/FP8) consistent
- ✓ Cross-GPU (both B300 in cluster) agree exactly for ceiling
- ✓ Bit-position invariant (only count matters)

The TRUE hardware ceilings asserted (2252 BF16, 4420 FP8) are now in
TRUE_REF row 66+. ✓

What's missing:
- This doc claims "BF16 const = 100.5% of cuBLAS spec 2242" — but the
  "spec" itself is throttle-limited per the doc's own argument. The
  framing "exceeds spec" is a tautology when spec measurement was
  throttled.
- Realistic Gaussian baseline (per Realistic Int4 doc) is ~1405 TF, NOT
  the "1486 TF random" baseline used here. The 1.51× speedup vs 1486
  becomes ~1.60× vs 1405 — but the more honest framing is that the
  REALISTIC baseline is even lower than this doc's "random".

### 6.3 Combined recommendation

Both docs remain valuable. **Add to top of each**:
"Headline numbers in this doc are for synthetic test patterns. For
production deployment estimates, use the GPTQ table at the bottom of
CUBLAS_REAL_VALIDATION.md (1.07–1.13× for Llama-shape FFN) or the Llama
70B FFN estimate in CUBLAS_BIT_ENTROPY_CORRECTION.md ('+5% for W4A16,
+11% for W4A4')."

---

## 7. LATENCY_DATA_INDEPENDENT.md — methodology quality

**Doc is sound.** 3 modes give identical clock64 distributions (46–88 cy,
mean 63.83). The conclusion "dedup is transistor-level clock gating, not
pipeline stalling" is correct.

Minor stale-ness:
- "constant 64 cy/MMA" — TCGEN05_N_SWEEP / TCGEN05_N_MATRIX (V11+) found
  cy/MMA varies by N (32–256) and by MMA shape. The "64" here is for
  m128n128k16 specifically; should annotate.
- The 32-MMA warmup with mbarrier is standard; no critique.

✓ Keep as-is.

---

## 8. CHAIN_PAIR_BYPASS.md — methodology quality

**Doc is sound for the V4 era.** Two-cluster model (FMA-style A; bit-manip
B) with +2 cy cross-penalty. SASS-verified for all 6×6 pairs.

Minor:
- The IADD3↔SHF "anomaly" resolution (compiler emits IMAD.IADD instead of
  IADD3, putting it in cluster A) is well-explained. ✓
- The DCE caveat about runtime perturbation is good methodology.
- LATER work (V46–V51 + the FFMA RF port findings in V4 list MEMORY.md)
  refines this further: register-port .reuse hints can bypass the
  cross-cluster penalty. Should cross-link.

✓ Keep as-is, but add cross-link to V4 RF-port findings.

---

## RETRACTIONS

Listed in priority order:

1. **NINJA "BF16 mma.sync 100%" framing** is circular. Correct: ~94% of
   legacy 616 TF theoretical (= 23% of tcgen05 2500 spec). Use the legacy
   denominator only with explicit "(legacy mma.sync theoretical)" tag.

2. **PRACTICAL_THROUGHPUT_GAIN's "174 effective GPUs" framing** is
   misleading without caveat that it requires synthetic K-grouped workload.
   Real ML workloads see 5-13% gain (not 74%).

3. **CUBLAS_REAL_VALIDATION's "1.45× headline"** is for square 8K³ K-row-
   identical only. For realistic Llama FFN (rectangular, GPTQ-quantized):
   1.07–1.13× is the deployment number. The "100 GPU = 149 GPU" framing
   should be reframed.

4. **NINJA "HBM write 7.57 TB/s" provenance**: TRUE_REFERENCE attributes
   to NINJA STG (e75c7e1); V8_HBM_WRITE_SOL attributes to TMA (28211ce).
   TRUE_REFERENCE wins. V8's "STG cap = 6.11" is a sub-optimal launch
   artifact, not architectural.

5. **NINJA inclusion of V46 (TMA 7.20 TB/s = "98.5%")**: V46 should NOT
   be added to NINJA — its 7.20 is BELOW the existing NINJA HBM read 7.30
   and below 01_hbm_bandwidth's 7.344 / 7.365. The 98.5% is a denominator
   artifact (uses 7.31 not 7.672 spec).

6. **TASK_LIST "free rein" hashes plus-2-more framing** in N-dependence
   entry is suspicious — reverify all 5 hashes via git log before quoting
   externally.

7. **README ladder claims "FP8 cuBLAS 4486 TFLOPS"** without realistic-
   data caveat. Add: "(zero-data; ~3984 random per TRUE_REF row 68)".

8. **SESSION_2_DELTA "Pure HBM read 720 W"** likely HBM-read-only without
   compute; per I4 (commit `bf98e90`), FP8 cuBLAS hits 1100 W cap with
   random data. The 720 W is a different scenario.

---

## UNRESOLVED

1. **Whether 01_hbm_bandwidth's 7.344 TMA result is reproducible** — V46
   gets 7.20 with 8-deep pipelining. Was the 7.344 from a different test
   harness or launch geometry? Run rigor harness on the 01 recipe to settle.

2. **Whether realistic ML workloads achieve the 1.07–1.13× automatic GPTQ
   speedup in production frameworks** (vLLM, TensorRT-LLM, AWQ kernels).
   The cuBLAS-only test may not transfer to custom dequant+GEMM kernels.

3. **Whether the "free rein" TASK_LIST hashes are git-real**. Per CLAUDE.md
   memory rule, must verify with `git log` before claiming completeness.

4. **Whether mxfp8/mxfp4 has the same bit-entropy mechanism** as BF16/FP8
   (untested in CUBLAS_BIT_ENTROPY_CORRECTION).

5. **Whether the 32×4 mma.sync regression mechanism is RF-bandwidth or
   scoreboard slots** (SESSION_2_DELTA narrows but doesn't isolate).

6. **Long-term thermal stability beyond 60s** — TRUE_REF and PRACTICAL
   docs cap at ~30 sec runs. Hour-scale at 962+ W untested.

7. **Cross-architecture transfer (A100/H100)** — bit-entropy mechanism
   likely present per analogy but untested.

---

## Summary table of meta-doc status

| Doc | Status | Action |
|-----|--------|--------|
| README.md | ~95% accurate | Add NINJA cross-refs, realistic-data caveats |
| TASK_LIST.md | format ✓; content TBD | Git-verify "free rein" hashes |
| NINJA_ACHIEVEMENTS.md | mostly correct; 7.57 attribution OK | Drop "100%" mma.sync framing; do NOT add V46 |
| SESSION_2_DELTA.md | retractions in-place ✓ | Cross-link 720 W context |
| PRACTICAL_SCOPE_HONEST.md | conservative ✓ | Keep |
| PRACTICAL_THROUGHPUT_GAIN.md | optimistic | Add "synthetic-test" disclaimer at top |
| REALISTIC_INT4_SCOPING.md | model citizen ✓ | Keep |
| CUBLAS_REAL_VALIDATION.md | conditional 1.45× | Lead with realistic Llama 1.07–1.13× |
| CUBLAS_BIT_ENTROPY_CORRECTION.md | rigor ✓ | Add baseline-realism caveat |
| B300_TRUE_REFERENCE.md | 2026-04-18 baseline | Reference corrections/* on top |
| LATENCY_DATA_INDEPENDENT.md | sound ✓ | Annotate cy/MMA = 64 is m128n128k16-specific |
| CHAIN_PAIR_BYPASS.md | sound ✓ | Cross-link to V4 RF .reuse findings |
