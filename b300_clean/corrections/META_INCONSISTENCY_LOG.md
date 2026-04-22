# Meta / Strategy Docs — Inconsistency Log

**Auditor**: meta-docs swarm agent, 2026-04-22
**Sources audited**:
- `README.md`
- `TASK_LIST.md`
- `NINJA_ACHIEVEMENTS.md`
- `SESSION_2_DELTA.md`
- `PRACTICAL_SCOPE_HONEST.md`
- `PRACTICAL_THROUGHPUT_GAIN.md`
- `REALISTIC_INT4_SCOPING.md`
- `CUBLAS_REAL_VALIDATION.md`
- `CUBLAS_BIT_ENTROPY_CORRECTION.md`
- `B300_TRUE_REFERENCE.md` (master)
- `LATENCY_DATA_INDEPENDENT.md`
- `CHAIN_PAIR_BYPASS.md`

This file logs each cross-file inconsistency, ambiguity, or stale claim
relevant to the meta-strategy/scoping/validation docs. For corrected stances,
see `META_DOCS_CORRECTED.md`.

---

## A. Numerical inconsistencies between meta-docs

### A1. HBM write peak — 7.30 vs 7.57 TB/s
| File | Number |
|---|---:|
| README ladder | 7.30 TB/s "v8 + per-warp coalesced" |
| TRUE_REFERENCE row 27 | 7.30 (v8 base recipe) |
| TRUE_REFERENCE row 28 | **7.57 NINJA STG** (commit `e75c7e1`, beats cudaMemset 5%) |
| NINJA_ACHIEVEMENTS table | **7.57 NINJA STG** (1 v8 store/warp) |
| (V8_HBM_WRITE_SOL contests) | 7.57 attributed to TMA, not STG |

**Status**: README is conservative; TRUE_REFERENCE and NINJA agree on 7.57
NINJA STG. The TMA-vs-STG provenance contest with V8_HBM_WRITE_SOL needs
re-verification (HBM_INCONSISTENCY_LOG #3 covers this).

### A2. HBM read peak — 7.20 vs 7.30 vs 7.344 vs 7.365 TB/s
| File | Number | Recipe |
|---|---:|---|
| README ladder | 7.30 | v8 NINJA recipe |
| TRUE_REFERENCE row 27 | 7.30 (a04d9c8) |
| TRUE_REFERENCE NINJA recipe | 7.31 (NINJA IT=2) |
| NINJA_ACHIEVEMENTS table | 7.31 (95.3%) |
| 01_hbm_bandwidth.md sub-optimal table | 7.344 TMA bulk; 7.365 LDG.E.128 |
| V41_V48_FINDINGS V46 | 7.20 = "98.5% of HBM peak" (denominator 7.31) |

**Status**: V46's "98.5%" is denominator artifact (uses 7.31 not 7.672 spec).
V46's 7.20 is BELOW prior measurements. Should NOT be added to NINJA.

### A3. BF16 mma.sync — 569 vs 577 vs 578 TFLOPS
- README "BF16 mma.sync legacy 577 TFLOPS"
- TRUE_REFERENCE row 47: 569 (1920 MHz)
- NINJA "Unbeatable ceilings" table: ~570 TFLOPS = "100% catalog burst"
- (V8_HMMA_F16_PEAK: 578.6 at 2032 boost)

**Status**: 569 / 578 differ by 1.6% (clock ratio). The "100%" framing in
NINJA is circular — measured equals ceiling because that's what we measured.
Correct framing: ~94% of legacy 616 theoretical, 23% of tcgen05 2500 spec.

### A4. FP8 cuBLAS — 4393 zero / 3983 random / 4486 README
- README "FP8 4486 TFLOPS" (no caveat)
- TRUE_REFERENCE row 56: 4425 (sustained via cudaGraph @ 943W)
- TRUE_REFERENCE row 57: 3983 random (REALISTIC)
- TRUE_REFERENCE row 68 data-dep table: 4393 zero / 3984 random / 3951 normal-ish
- NINJA "Unbeatable ceilings" table: ~5000 spec, achieved 88.5% (4425) "via cudaGraph"

**Status**: README's 4486 lacks realistic-data caveat. Add: "(zero data;
~3984 random per TRUE_REF row 68)".

### A5. BF16 cuBLAS — 2242 vs 2252 (TRUE peak vs spec)
- TRUE_REFERENCE row 48: 2242 (90% of NVIDIA 2500 spec)
- CUBLAS_BIT_ENTROPY_CORRECTION final: 2252 const-data hardware ceiling
- 06_tensor_cores: spread 2237–2259 cuBLAS, 2325 microbench
- README ladder: 2259 (catalog row)

**Status**: All within 1% noise except 2325 (microbench tcgen05 direct).
"100.5% of spec" framing is tautological since spec measurement was
itself power-throttled.

### A6. cuBLAS K-row identical speedup — 1.41× vs 1.45× vs 1.49× vs 1.51×
- CUBLAS_REAL_VALIDATION raw-random table: 1.49×
- CUBLAS_REAL_VALIDATION normal-only table: 1.45×
- CUBLAS_REAL_VALIDATION K-row identity sweep: 1.41×
- CUBLAS_BIT_ENTROPY_CORRECTION FP16: 1.51× (random→const at 8K³)
- TRUE_REFERENCE addendum: 1.42×

**Status**: All within ~7% of each other. The headlines vary by ± 0.04
depending on whether the random baseline includes Inf/NaN. Recommend
canonical = "1.41×–1.45× depending on baseline noise".

### A7. PRACTICAL throughput gain at 1100W: 1.18× vs 1.49× vs 1.55× cuBLAS
- PRACTICAL_THROUGHPUT_GAIN microbench: 1.18× at 1100W
- CUBLAS_REAL_VALIDATION cuBLAS BF16 8K³: 1.45–1.49× at 1100W
- CUBLAS_BIT_ENTROPY_CORRECTION FP8 8K³: 1.55× at 1100W

**Status**: Microbench (1.18×) vs cuBLAS (1.45–1.55×) is real — different
mechanisms (microbench: 4× MMA cluster; cuBLAS: 2-CTA m256n256 cluster_group::2
amplifies dedup).

---

## B. Stale claims (V11–V51 supersedes)

### B1. SESSION_2_DELTA "BF16 mma.sync 90.5% of NVIDIA 2500 spec"
- ALREADY RETRACTED in-place ("MAJOR RETRACTION 2026-04-19" section)
- Correct: 23% of 2500 (tcgen05 spec) OR 93.7% of 616 (legacy spec)
- ✓ self-corrected

### B2. SESSION_2_DELTA "HMMA chains 1≡4 in time" + "HMMA+STS 40% overlap"
- ALREADY RETRACTED in-place ("Pipe matrix HMMA findings ALSO retracted")
- Correct: HMMA chains scale linearly; HMMA+STS mostly independent (3-33%)
- ✓ self-corrected

### B3. SESSION_2_DELTA "FP8 mma.sync 7500-8200 TFLOPS"
- ALREADY DIAGNOSED in-place (SASS shows HMMA.16816 not 16832)
- Correct: ~3760 TF = 75% of FP8 spec, OR use cuBLAS for 4500
- ✓ self-corrected

### B4. SESSION_2_DELTA "Pure HBM read 720 W = +520 W (memory dominates)"
- I4 (commit `bf98e90`) shows FP8 cuBLAS hits 1100 W cap with random data
- 720 W is HBM-read-only without compute; 1100 W is HBM + compute
- ⚠ Needs annotation; not formally retracted

### B5. NINJA "32-way SMEM bin privatization 78%" null result
- L3 histogram (commit `492a5f6`) achieved 6.57 TB/s = 90.1% with v2 SMEM
  aggregation
- The 78% number was an EARLIER attempt; NINJA correctly lists it as null
  but should cite L3 commit hash for the WORKING approach
- ⚠ Cosmetic

### B6. CHAIN_PAIR_BYPASS "two clusters" model
- SUPERSEDED by V4 RF .reuse findings (per MEMORY.md V4 deep-dive)
- The .reuse hint can bypass cluster boundaries
- ⚠ Cross-link missing

---

## C. Format / hash sanity

### C1. TASK_LIST commit hashes
- All ~70 hashes are 7-char lowercase hex ✓
- No placeholder hashes (0000000, deadbef) ✓
- No malformed hashes ✓
- Internal duplications correctly marked SUBSUMED (E1=C6, G4=G2) ✓

### C2. TASK_LIST "free rein" appendix
- Per CLAUDE.md memory rule: prior loops hallucinated hashes
- Appendix lists ~30 additional hashes with claims
- Format check: all valid 7-char hex
- ⚠ N-dependence entry says "5 commits" but writes only 3 explicitly
  (`702d4bd` `3f98880` `4bdb423` plus 2 more) — yellow flag

### C3. NINJA_ACHIEVEMENTS commits
- `e75c7e1`, `9f1c990`, `4958d6b`, `866e951`, `1b8ca7e`, `57a2e6f` etc
- All 7-char hex ✓
- No git verify performed

### C4. CUBLAS_REAL_VALIDATION specific commits
- No commit hashes referenced (refers to investigations/* files instead)
- Investigations directory existence not audited

---

## D. Framing / scope honesty issues

### D1. PRACTICAL_THROUGHPUT_GAIN "100 GPU = 174 GPU equivalent"
- Math: 100 × 1.74 = 174 ✓
- BUT requires synthetic structured workload (mode 5208)
- Real ML clusters won't see this magnitude
- ⚠ Misleading without disclaimer

### D2. CUBLAS_REAL_VALIDATION "100 GPU = 149 GPU"
- Math: 100 × 1.49 = 149 ✓
- BUT requires square 8K³ K-row identical workload
- Llama-shape inference (rectangular, large N) gets 1.01× = NOTHING
- The bottom of doc has GPTQ Llama numbers (1.07–1.13×) that are realistic
- ⚠ Headline (1.49×) ≠ deployment reality (1.07–1.13×)

### D3. CUBLAS_BIT_ENTROPY "TRUE hardware ceiling exceeds spec"
- 2252 BF16 vs spec 2242 = 100.5%
- This is real but tautological (spec was throttle-limited)
- The framing "exceeds spec" can mislead readers
- ⚠ Cosmetic

### D4. NINJA "Unbeatable ceilings" — many are tautologies
- "BF16 mma.sync 100% (catalog burst confirmed)" — circular
- "SHMEM read 99.8% (already at SoL)" — tautological
- These are correct as "we hit our ceiling" but mislabeled "unbeatable"

### D5. README cross-category contradictions table
- Item 1 (concurrent HBM R+W) — RESOLVED, should be moved out of "🔴 CRITICAL"
- Item 4 (cluster cost 50 vs 185 ns) — RESOLVED, should be moved out
- ⚠ Stale prioritization

---

## E. Tensor-agent cross-checks

### E1. CUBLAS_REAL_VALIDATION's "FP8 random 3983" — confirmed by tensor agent
- TRUE_REFERENCE row 68 also lists 3984 random for FP8 e4m3
- Within noise of CUBLAS_REAL's 3983 ✓
- The "FP8 zero 4393" comparison is also in TRUE_REF row 68 (4393 zero)

### E2. CUBLAS_REAL_VALIDATION's BF16 1517 random vs zero baseline
- BF16 1517 random matches TRUE_REF 1486 random within 2% ✓
- Both well below BF16 const ceiling 2252
- The K-row identical 2143 ≈ const ceiling 2252 (95% of ceiling)
- The mechanism convergence is consistent

### E3. NCU verification in CUBLAS_BIT_ENTROPY (cycle counts identical)
- This is the SMOKING GUN that mechanism is clock-frequency-via-throttle
- pipe_tensor 99% active misleading per memory rule (covered in TENSOR
  agent's B-section)
- ✓ Doc correctly uses smsp__inst_executed_pipe_tensor_subpipe_hmma_op_
  utchmma_utcqmma_utcomma_scope_2cta.sum (the tcgen05-inclusive metric)

---

## F. Methodology consistency (LATENCY_DATA_INDEPENDENT, CHAIN_PAIR_BYPASS)

### F1. LATENCY_DATA_INDEPENDENT "64 cy/MMA"
- Specifically m128n128k16
- TCGEN05_N_SWEEP / TCGEN05_N_MATRIX (V11+) shows cy/MMA varies by N (32-256)
- ⚠ Should annotate "for m128n128k16; see TCGEN05_N_SWEEP for other shapes"

### F2. CHAIN_PAIR_BYPASS "two clusters + 2cy cross-penalty"
- V4 era findings; the 6×6 matrix is SASS-verified
- Later V4 work in MEMORY.md found .reuse hints can bypass
- ⚠ Should cross-link to V4 deep-dive

### F3. Both methodology docs have explicit DCE caveats
- LATENCY uses 32-MMA warmup + clock64
- CHAIN uses runtime perturbation u2
- ✓ Both methodology-sound for their era

---

## G. Counter-metric misuse audit (per memory rule)

### G1. CUBLAS_BIT_ENTROPY uses pipe_tensor_subpipe_hmma_cycles_active
- Line 1086-1087 cited 137,094,592 sm_active and 536,870,912 tensor cycles
- ✓ Combined with the inst_executed_pipe_tensor_subpipe_hmma_op_utchmma_utcqmma_utcomma
  metric in the same section
- Mostly OK but readers should note pipe_tensor_subpipe_hmma_cycles_active
  is HMMA-only; the larger op count metric covers tcgen05

### G2. SESSION_2_DELTA contains the "ncu pipe_tensor doesn't measure tcgen05"
- Lines 524-790 demonstrate pipe_tensor showing 99% active for only 2.42 ms
  of 9.37 ms wall-clock
- ✓ This IS the documentation of the gotcha; recommend cross-linking

---

## Summary table

| # | What | Severity | Where to fix |
|---|------|----------|--------------|
| A1 | HBM write 7.30 vs 7.57 | LOW | README; add NINJA cross-ref |
| A2 | HBM read 7.20 / 7.30 / 7.344 / 7.365 | MED | NINJA: don't add V46 |
| A3 | BF16 mma.sync 569 vs 577 vs 578 vs "100%" | MED | NINJA: drop "100%" framing |
| A4 | FP8 4486 in README lacks realistic caveat | MED | README: add (zero/random) annotation |
| A5 | BF16 2242 spec vs 2252 const ceiling | LOW | Frame as "matches spec, ~92% of theoretical" |
| A6 | K-row identical speedup 1.41-1.49× | LOW | Cite range, not single number |
| A7 | Microbench 1.18× vs cuBLAS 1.45-1.55× at 1100W | LOW | Note both in PRACTICAL_THROUGHPUT_GAIN |
| B4 | "Pure HBM read 720 W" needs context vs I4's 1100 W | LOW | SESSION_2_DELTA annotation |
| B6 | CHAIN_PAIR_BYPASS missing V4 .reuse cross-link | LOW | Add link |
| C2 | TASK_LIST "free rein" hash verify needed | MED | git log audit |
| D1 | PRACTICAL_THROUGHPUT_GAIN "174 GPU" misleading | HIGH | Add disclaimer at top |
| D2 | CUBLAS_REAL "149 GPU" misleading | HIGH | Lead with realistic Llama 1.07-1.13× |
| D3 | "Exceeds spec" framing in BIT_ENTROPY | LOW | Cosmetic |
| D4 | NINJA "Unbeatable ceilings" tautologies | MED | Rename "Best achieved" |
| D5 | README contradictions queue stale | LOW | Move resolved items out of CRITICAL |
| F1 | LATENCY 64 cy/MMA shape-specific | LOW | Annotate |

---

## Cross-reference to other inconsistency logs

- HBM details: `corrections/HBM_INCONSISTENCY_LOG.md`
- Tensor details: `corrections/TENSOR_INCONSISTENCY_LOG.md`
- Compute details: `corrections/COMPUTE_INCONSISTENCY_LOG.md`
- Atomics: `corrections/ATOMICS_INCONSISTENCY_LOG.md`
- SHMEM: `corrections/SHMEM_INCONSISTENCY_LOG.md`
- Power: `corrections/POWER_INCONSISTENCY_LOG.md`
- TMA/Launch: `corrections/TMA_LAUNCH_INCONSISTENCY_LOG.md`
- Caches: `corrections/CACHES_INCONSISTENCY_LOG.md`
- Sync: `corrections/SYNC_INCONSISTENCY_LOG.md`
- DSMEM: `corrections/DSMEM_INCONSISTENCY_LOG.md`
- Math intrinsics: `corrections/MATH_INCONSISTENCY_LOG.md`
- Integer/bit: `corrections/INT_INCONSISTENCY_LOG.md`
