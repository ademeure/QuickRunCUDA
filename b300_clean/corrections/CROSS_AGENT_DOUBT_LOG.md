# CROSS-AGENT DOUBT LOG

**Auditor:** adversarial-doubt agent, 2026-04-22.
**Sources:** all 19 `*_CORRECTED.md` + `*_INCONSISTENCY_LOG.md` files in
`b300_clean/corrections/`.

**Method:** Per-topic comparison across the 12+8 sub-agent outputs. Logs
contradictions that the M-synthesis flattened or that two equally-confident
sub-agents took mutually-exclusive positions on. Severity: SEV1 = headline
number affected, SEV2 = mechanism / methodology disagreement, SEV3 = wording
or denominator quibble.

---

## 1. HBM theoretical denominator (CLAUDE memory says ~8 TB/s spec)

| Agent | File | Denominator used | Anchored in spec? |
|---|---|---|---|
| HBM | 01_hbm_bandwidth_CORRECTED §0 | **7672 GB/s post-ECC** (derived 7680b × 3996 MHz × 2 / 8) | YES — derived from chip params |
| HBM (V32-V48 corpus inheritance) | same, §2 | 7.31 TB/s (empirical pure-direction peak) | NO — empirical |
| Tensor | 06_tensor_cores_CORRECTED | does not normalize HBM; cites cuBLAS spec denominators only | n/a |
| NVFP4 | NVFP4_CONSOLIDATED §1 / 5 | does not denominate HBM | n/a |
| Memory APIs | 09_memory_apis_CORRECTED | **7.2 TB/s** ("% of HBM 7.2 TB/s") | NO — empirical-rounded |
| Sync | 08_sync_primitives_CORRECTED | n/a | n/a |
| CLAUDE.md | memory snippet | "~8 TB/s spec" | NO — pre-ECC nominal |

**Verdict: OPEN CONTRADICTION (SEV1).** HBM agent ANCHORS 7672 GB/s.
Memory-APIs agent uses 7.2 TB/s (a different empirical denominator).
CLAUDE.md says ~8. Three different denominators across three corrections
files; cross-doc % numbers are NOT comparable. **Recommend HBM agent's
7672 GB/s** because it's the only one derived from spec arithmetic.
M-synthesis did not unify these.

---

## 2. Threadfence (`__threadfence`) GPU cost — 258 / 281 / 292 / 320 cy spread

| Agent / source | Cited number |
|---|---|
| Sync agent (08_CORRECTED §RECONCILED ladder) | "281 ± 25" with explicit 4-way spread; flags UNRESOLVED |
| Sync agent (08_CORRECTED §RECOMMENDED CANONICAL) | **281 cy** |
| Sync agent inconsistency log (#2) | 258 (V9) vs 320 (DSMEM) — 24% gap UNRESOLVED |
| Atomics agent (07_CORRECTED §3) | does NOT cite a fence number; defers via "+1040 cy MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR" — implies ~1040 cy ATOMIC-overhead, distinct primitive |
| Memory-APIs agent (09_CORRECTED) | does NOT cite |
| DSMEM_CORRECTED (referenced in sync log) | **320 cy** for `fence.sc.gpu` (no spread acknowledged) |
| CLAUDE memory | "GPU 281 cy" |

**Verdict: PARTIAL CONTRADICTION (SEV2).** Sync agent correctly preserves
the spread. DSMEM agent quotes a single 320 cy without acknowledging the
4-way spread. Atomics agent's "+1040 cy MEMBAR.ALL.GPU" is from a
different primitive (release atomic, full fence stack), not directly
comparable, but no agent cross-checks the two. **More authoritative: sync
agent.** DSMEM should adopt the spread.

For `__threadfence_system`: 1750 (08) vs 3042 (V9) vs 2870 (DSMEM) = SEV1
1.74× spread, sync agent flags UNRESOLVED. No other agent cites a sys-fence
number, so no cross-contradiction — but TRUE_REFERENCE picked 861 ns
(=1750 cy) without justification.

---

## 3. MUFU rate — 4.74 Gops/s vs 47.8 G

| Agent | Number cited | Regime |
|---|---|---|
| Math (14_math_intrinsics_CORRECTED §1) | **4.74 Gops/s** non-EX2; **9.22 Gops/s** EX2 (chip-wide) | THROUGHPUT-bound (V41) |
| M-synthesis (M14, M16) original | "47.8 GMUFU/s @ 99.5% XU peak" | LATENCY-bound 1-chain rsqrt — FLAGGED as RETRACTION by M-synth corrections |
| M-synthesis CORRECTIONS doc | adopts math agent's 4.74 / 9.22 Gops/s for chip peak | reconciled |
| Tensor agent | does not cite MUFU; only HMMA / tcgen05 | n/a |
| FP32 agent (04_CORRECTED §dual-issue) | "FFMA + MUFU = ~100% overlap" — implicit MUFU rate consistent with 4.74 G | reconciled |

**Verdict: RESOLVED (SEV3).** All current agents converge on 4.74 G chip /
9.22 G EX2. M-synth properly flags M14/M16 retraction. **Math agent is
authoritative.** No cross-contradiction in the corrections set; only the
underlying catalog had stale numbers.

---

## 4. L2 bandwidth — three metrics (13.30 / 23.85 / ~30)

| Agent | Number(s) | Metric specified? |
|---|---|---|
| Cache (03_caches_CORRECTED §2.3) | 13.30 (lts wire) / 23.85 (kernel-effective) / ~17 (`.cg` carveout=100) / 30-36 (L1-amplified) | YES — explicit per-metric column |
| HBM (01_CORRECTED §5) | "L2 plateau 23.0 TB/s @ 64 MB" | NO — single number, doesn't say which metric |
| Memory-APIs (09_CORRECTED) | does NOT cite L2 BW | n/a |
| M-synthesis M5 row | "L2 single-SM 23 TB/s" | NO — labeled per-SM but matches kernel-effective |
| CLAUDE.md memory | "L2 = 22 TB/s" | NO — described as union of metrics 10-36 |

**Verdict: PARTIAL CONTRADICTION (SEV2).** Only the cache agent disambiguates
the three definitions. HBM agent's 23.0 number floats with no metric tag
(matches kernel-effective by coincidence). M5 cheatsheet is similarly
unannotated. **Cache agent is authoritative;** HBM agent's L2 row should
adopt the "(kernel-effective, includes L1 amp)" annotation.

---

## 5. Dual-issue cap — 55% / 74% (FP32) vs other agents

| Agent | Same-warp dual-issue % | Warp-spec dual-issue % | Pair tested |
|---|---:|---:|---|
| FP32 (04_CORRECTED §dual-issue) | 55% (FFMA+LOP3), 54% (FFMA+IADD3), 51% (FFMA+PRMT) | 74% (FFMA+LOP3) | V49/V50 |
| INT (15_int_bit_ops_CORRECTED §RETRACTIONS row 2) | "FFMA+IADD3 measured overlap 14.2% (A6) / 17% (B1)" | not stated | A6, B1 |
| M8 PIPE_OVERLAP_MATRIX original | "FFMA + IADD3 = 56% overlap" (CLOSE to V49 54%) | 0 (no warp-spec row) | (older) |
| M-synthesis CORRECTIONS M8 | adopts V49 55%, V50 74% | yes | yes |

**Verdict: SOFT CONTRADICTION (SEV1).** FP32 agent says 54-55% overlap,
INT agent says 14-17% overlap. They are using DIFFERENT denominators:
FP32's "55%" = "55% of perfect parallelism" (i.e. you get 55% of summed
peaks); INT's "17%" = "Mixed throughput exceeds solo FFMA peak by only
17%" (gain over best-single-pipe). M-synth M1 §addenda flags this exact
reframing issue ("same direction, different denominator"). **Both are
right.** FP32 agent is more authoritative for the canonical
overlap-fraction number; INT agent's 17% is the gain-over-baseline framing.
Catalog needs to pick ONE convention.

---

## 6. IADD3 pipe placement — FMA or ALU?

| Agent | Placement | Citation |
|---|---|---|
| FP32 (04_CORRECTED §pipe placement) | **FMA pipe** (V40, commit `d1d09c5`) | EXPLICIT |
| INT (15_int_bit_ops_CORRECTED §pipe table) | **FMA pipe** (V40, A6) | EXPLICIT |
| M-synthesis CORRECTIONS (M8 / M16 retractions) | **FMA pipe** (V40 supersedes M8/M16's "ALU" placement) | EXPLICIT |
| Tensor agent | does not address IADD3 | n/a |
| Power agent (16_CORRECTED §energy/op) | groups IADD3 with FFMA chains (consistent with FMA placement) | implicit consistent |

**Verdict: RESOLVED (SEV3).** All four agents that mention IADD3 agree:
FMA pipe per V40. The historical confusion ("ALU pipe") is correctly
flagged as RETRACTED across FP32, INT, and M-synth files. No live
contradiction. **Authoritative: V40 measurement, cited by all three.**

---

## 7. NVLink spec denominator — 757 vs 900 GB/s/dir

| Agent | Denominator | Resulting % |
|---|---:|---|
| NVLink (12_CORRECTED §R3) | **900 GB/s/dir** (NVLink 5: 18 × 50) | 778/900 = 86% |
| NVLink (12_CORRECTED §R6) | RETRACTS CLAUDE's 757/1503 NVLink-4 numbers | n/a |
| PCIe agent (13_CORRECTED) | DEFERS to NVLink agent's R1 / R6; agrees "NVLink 5", retracts "v7" naming | consistent |
| TRUE_REFERENCE (legacy) | 757 GB/s = NVLink-4 spec — **WRONG per NVLink agent** | inflated |
| CLAUDE memory | "NVLink (2× B300): 757 GB/s unidirectional / 1503 GB/s bidirectional (NVLink v7)" | both numbers wrong + name wrong |

**Verdict: RESOLVED with CLAUDE-memory action item (SEV1).** NVLink and
PCIe agents AGREE on 900 GB/s/dir spec, NVLink-5 naming. CLAUDE memory is
explicitly RETRACTED by both. **Authoritative: NVLink agent.** No cross-doc
still cites 757 — but TRUE_REFERENCE and CLAUDE memory must be updated.
Severity HIGH because every "% of peak" NVLink number in legacy docs is
inflated by 19%.

---

## 8. NVFP4 K=96 ceiling — 10.8 PF (memory) vs 11.42 PF (NVFP4 agent)

| Agent | Number | Source / framing |
|---|---:|---|
| NVFP4 (NVFP4_CONSOLIDATED §1) | **11423 TFLOPS** = 76.2% of 15 PF | cuBLAS + cudaGraph BPG=16 (NVFP4_CUDAGRAPH.md) |
| NVFP4 (§1) plain Lt | 11068 = 73.8% | cuBLAS w/o graph |
| Tensor (06_CORRECTED row "K=96 cuBLAS 13.4 wide-rect") | **~10800 TFLOPS** = 72% | "memory entry, prior session" — labelled **MED**, "not verified in this clean directory" |
| CLAUDE memory `nvfp4_k96_ceiling` | "cuBLAS 13.4 caps 10.8 PF" | older session |
| NVFP4 agent §RETRACTIONS row | RETRACTS "10.8 PF cuBLAS catalog" → 11.42 PF | EXPLICIT supersession |

**Verdict: SOFT CONTRADICTION (SEV1).** Tensor agent cites 10.8 PF without
acknowledging NVFP4 agent's 11.42 PF update. Tensor agent flags the row as
MED and "not verified in this clean directory" — so it's at least caveated,
but it should DEFER to the NVFP4 agent (which has the dedicated CUDAGRAPH
file) instead of recapping the older memory note. **NVFP4 agent
authoritative.** Tensor agent should cross-link, not duplicate.

---

## 9. Bank-conflict cost — regime-dependent (SHMEM) vs single-number elsewhere

| Agent | Cost framing | Regime stated? |
|---|---|---|
| SHMEM (02_CORRECTED §2) | **1× to 8.81×** depending on regime; latency-bound = 2-5×, throughput-bound = ~1× | YES — explicit table |
| SHMEM agent §UNRESOLVED 1 | flags V44/V45 throughput vs latency split as UNRESOLVED in catalog | yes |
| Cache agent (03_CORRECTED) | does not address SMEM bank conflicts | n/a |
| Tensor agent | does not | n/a |
| FP32 agent | does not | n/a |
| Math agent | does not | n/a |

**Verdict: RESOLVED (SEV3).** Only the SHMEM agent addresses bank conflicts;
no other agent simplifies it to one number. No cross-contradiction. **SHMEM
agent authoritative.** The naive "32× textbook" cost is correctly flagged as
never observed.

---

## 10. HBM read peak — 7.30 (NINJA) vs 7.20 (V46) vs 7.34/7.37 (TMA bulk / LDG)

| Agent | Headline read peak | Method |
|---|---:|---|
| HBM (01_CORRECTED §1 headline) | **7.30 TB/s = 95.2%** of 7672 spec | v8 + per-warp coalesced + non-persistent |
| HBM (§1 also lists) | 7.34 (TMA bulk), 7.37 (LDG.E.128) | both > 7.30; HBM agent does not pick a single SoL |
| HBM (§2 V46 row) | 7.20 = 93.8% | EXPLICITLY DEMOTED: "improves V33 single-deep but not architectural new SoL" |
| Memory-APIs (09_CORRECTED) | "**V46 = NEW READ SoL = 7.20 TB/s = 98.5%** ← NEW READ SoL" | 7.2 TB/s denominator → inflates % |
| M-synthesis CORRECTIONS M14 row | "HBM read 5.82 TB/s = 81% SoL — SUPERSEDED by V46 7.20 TB/s = 98.5%" | adopts 09's framing, NOT 01's |
| Meta-docs CORRECTIONS §3.2 | "V46 should NOT be added to NINJA — its 7.20 is BELOW NINJA HBM read 7.30 and below 01_hbm_bandwidth's 7.344 / 7.365" | adopts HBM agent's framing |
| NVFP4 NVFP4_FULL_PIPELINE | not directly cited; achieves "2.96×" speedup vs baseline (referenced indirectly) | n/a — different metric |

**Verdict: OPEN CONTRADICTION (SEV1).** Memory-APIs agent and M-synth M14
update both treat V46 7.20 as the new SoL. HBM agent and Meta-docs agent
both EXPLICITLY DEMOTE V46 to "not architectural new SoL" (because 7.20 <
7.30 < 7.34 < 7.37 when properly normalized to 7672 spec). The two
positions are mutually exclusive — V46 cannot be both "the new HBM read
SoL" AND "below the existing peak". **HBM agent + Meta-docs agent
authoritative** (denominator math is correct: 7.20/7672 = 93.8%, not 98.5%).
Memory-APIs and M14 update should be re-aligned to the HBM agent's
position.

---

## Additional contradictions surfaced during the sweep

### 11. HBM write SoL 7.57 TB/s — provenance contested

| Agent | Attribution |
|---|---|
| HBM (01_CORRECTED §3) | "CONTESTED PROVENANCE — UNRESOLVED §B" |
| Meta-docs CORRECTIONS §3.1 | "TRUE_REFERENCE sides with NINJA STG (e75c7e1); V8 attribution to TMA `28211ce` is wrong" |

SEV2 — flagged by both, no clean resolution; needs re-test.

### 12. SMEM atomic aggregate — 2.27 T (catalog) vs 4.2 T (CLAUDE memory)

| Agent | Number |
|---|---|
| Atomics (07_CORRECTED §4) | **2.15-2.27 T atomic/s** (V10_SMEM, INT32) |
| SHMEM (02_CORRECTED §3) | "2.2 T atomic/s aggregate" (consistent) |
| CLAUDE memory `project_b300_v8_complete` | "**SMEM atomic 4.2 Tops/s no-contention**" |
| Atomics agent §RETRACTIONS row 6 | "**4.2 T figure NOT reproduced; provenance unknown**" |

SEV2 — Both atomics and SHMEM agents converge on ~2.2 T. CLAUDE memory's
4.2 T is unsourced; flagged for cleanup. **Atomics agent authoritative.**

### 13. L2 atomic packets — stride-4 peak: 449 / 504 / 1005 Gops/s

| Source (per atomics agent §UNRESOLVED) | Value |
|---|---:|
| TRUE_REFERENCE row | 449 |
| 07_atomics §8 UNROLL=16 | 504 |
| 07_atomics §8 UNROLL=32 | 1005 |

SEV2 — atomics agent properly flags as "stride×UNROLL×L2-residency
combinatorial; catalog should ALWAYS pair the three". No other agent cites
a single number, so no cross-contradiction; only methodology pitfall.

### 14. PRMT pipe placement — V40 "permute" vs A6 "INT-bit"

INT agent (15_CORRECTED §UNRESOLVED 2): V40 says PRMT 13.9 Glane/s = 36% =
"permute pipe"; A6 says PRMT 14.08 = 0.5/SMSP/cy = same tier as LOP3
(INT-bit). Different ILP regimes; needs A6-style sweep on PRMT.
SEV2 UNRESOLVED.

---

## SUMMARY — top 10 most concerning cross-agent contradictions

| Rank | Topic | Severity | Status |
|------|-------|----------|--------|
| 1 | **HBM read SoL: V46 7.20 = "98.5%" vs HBM agent demotes to 93.8%** | SEV1 | OPEN — Memory-APIs + M14 update conflict with HBM + Meta-docs |
| 2 | **HBM theoretical denominator (7672 vs 7.31 vs 7.2 vs 8.0 TB/s)** | SEV1 | OPEN — three corrections files use three different denominators |
| 3 | **NVLink spec denominator (757 vs 900)** | SEV1 | RESOLVED in NVLink+PCIe agents; CLAUDE.md + TRUE_REFERENCE need update |
| 4 | **NVFP4 K=96 ceiling 10.8 vs 11.42 PF** | SEV1 | SOFT — tensor agent cites stale 10.8 without deferring to NVFP4 agent's 11.42 |
| 5 | **Dual-issue 55% vs 17% (denominator framing)** | SEV1 | SOFT — both agents right; catalog needs single convention |
| 6 | **HBM write 7.57 TB/s provenance (NINJA STG vs TMA)** | SEV2 | OPEN — flagged by HBM and Meta-docs; needs re-test |
| 7 | **L2 BW 13.30 / 23.85 / ~30 (metric tagging)** | SEV2 | PARTIAL — only cache agent annotates; HBM agent's 23.0 floats |
| 8 | **Threadfence GPU 258/281/292/320 cy spread** | SEV2 | OPEN — sync agent preserves spread; DSMEM picks 320 silently |
| 9 | **SMEM atomic 2.27 T (atomics+SHMEM agree) vs 4.2 T (CLAUDE memory)** | SEV2 | CLAUDE memory unsourced; agents converge |
| 10 | **System fence 1750 vs 3042 cy (1.74× spread)** | SEV1 | OPEN — sync agent flags; TRUE_REFERENCE picked 1750 without justification |

### Patterns observed

- **Denominator drift is the #1 source of cross-agent disagreement** (HBM,
  NVLink, dual-issue framing, NVFP4 K=96).
- **M-synthesis flattens contradictions** in the dual-issue / V46 cases by
  picking one side without flagging the other.
- **CLAUDE.md memory is the most contradicted source** (NVLink 757,
  HBM 8 TB/s, SMEM atomic 4.2 T, K-96 10.8 PF — all challenged or
  retracted by ≥1 agent in this sweep).
- **Sync / fence agent has the cleanest documentation of its own
  uncertainty** (4-way spread on GPU fence, 1.74× system fence).
- **HBM agent's denominator anchoring (7672)** is the right model for
  every other "% of peak" claim; should be propagated.
