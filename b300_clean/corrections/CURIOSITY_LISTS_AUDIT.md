# CURIOSITY_LIST V2-V8 Audit

**Date**: 2026-04-22 (wave-3 swarm).
**Scope**: `b300_clean/CURIOSITY_LIST_V{2,3,4,5,6,7,8}.md`.
**Method**: every commit hash in every `[x]` entry git-verified;
findings cross-checked against `corrections/MASTER_INDEX.md`,
`corrections/HEADLINE_CORRECTIONS.md`, and the per-topic
`corrections/*_INCONSISTENCY_LOG.md`.

---

## Per-version summary

| Version | Total items | `[x]` done | `[ ]` open | `[~]` deferred | Hashes cited | Hashes valid | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| V2 | 24 (tagged S/A/B/C/D) | 21 | 0 (all S-tier rewritten) | 0 | 25 unique | **3 / 25** | Heavy hash hallucination; topic was hand-retracted in-place |
| V3 | 112 | 0 | 112 | 0 | 0 | n/a | Pure backlog; "SUPERSEDED" by V4 (footer line 251) |
| V4 | 135 | 135 | 0 | 0 | 77 unique | **77 / 77** | Clean. 32 of the 135 [x] cite "Commit history" not a hash |
| V5 | 66 | 66 | 0 | 0 | 65 unique | **65 / 65** | Clean |
| V6 | 65 | 60 | 0 | 5 (tcgen05) | 57 unique | **57 / 57** | Deferred items honestly labeled `[~]` |
| V7 | 69 | 44 | 25 | 0 | 44 unique | **44 / 44** | Clean |
| V8 | 74 | 50 | 24 | 0 | 47 unique | **47 / 47** | Clean |

**Headline**: only V2 has hallucinated hashes (88% wrong, 22 of 25). V4-V8
hashes are 100% git-verified. V3 carries no hashes (all `[ ]`).

---

## RETRACTED [x] CLAIMS

These items are marked `[x]` with a finding that the
`corrections/` audit later overturned. The hash itself is valid; the
claimed result is wrong.

### V2 — many (mostly already self-flagged)

V2 contains an embedded "RETRACTED 2026-04-19" header on **S1** with the
old wrong text preserved as audit trail; the V2 author was honest about
the over-claim. Items NOT formally retracted in V2 but contradicted by
later corrections:

| V2 item | Claim | Where overturned |
|---|---|---|
| **S1-OLD-WRONG** `[x]` | "mma.sync 90.5% of 2500 spec" | RETRACTED in-place (preserved). Real: 23% of tcgen05 spec or ~93% of 616 legacy spec (`META_INCONSISTENCY_LOG.md` B1). |
| **S4-OLD** `[x]` | "16-warp aggregate 1.84 mma/SM/cy" | Refuted by S4 strict-anti-DCE (16 warps DROP to 159 TF). V2 keeps both. |
| **A1** `[x]` "147 TF/head FlashAttn baseline" | survives | Not retracted by audit. |
| **B2** `[x]` "L2 26.7 TB/s" | Author corrected in-place at V2 line 676 to "actually L1+L2 effective; pure L2 = 21 TB/s" | `CACHES_INCONSISTENCY_LOG.md` "L2 BANDWIDTH" requires labelled denominator. |

### V4 — five `[x]` claims now superseded

| Item | Claim | Source of correction |
|---|---|---|
| **A1** `08ee753` | "FFMA+IADD3 cluster-A contention 14% slower" framing | Partial — V49/V50 (`501134a`/`fbe1c18`) refine to 55% same-warp / 74% warp-specialized. V4 A1 is not WRONG but no longer the headline (`COMPUTE_INCONSISTENCY_LOG.md` F). |
| **B1** `8012b98 / B1_DUAL_ISSUE_FFMA_IADD3.md` | "FFMA+IADD3 17% overlap (unified cluster)" | Same — superseded by V49/V50 measurement at 54-74%. (`COMPUTE_INCONSISTENCY_LOG.md` F.) |
| **D6** `998e947` | "76 TFLOPS only with .reuse; worst-case ~50 TFLOPS" | NOT retracted — actually validated by `04_fp32_peak_CORRECTED.md` and `V8/V10 misc log §C`. V4 had this RIGHT before the catalog headlines. |
| **R1** `741ce96` | "PARTIAL — runtime variance confounds" | Honest partial. Power-gating later confirmed null in V7 K4 (`d71bdc8`). Status fine. |
| **D7** `eff12d4 / 2b5c9a3 / 6eee60c` | "TMEM read 65 TB/s chip-wide" | Contradicted by `CACHES_INCONSISTENCY_LOG.md` "TMEM BANDWIDTH" → 60 TB/s consensus (V4 over by ~8%, within rounding) but earlier "295/830 TB/s" claims fully refuted. V4's number is closer to truth than catalog headlines. |

### V5 — three `[x]` claims now superseded

| Item | Claim | Source of correction |
|---|---|---|
| **B5** ref V4 A1 | "ILP saturates ~10 cy/mma" | Compatible with V6 J1 (8 cy at ILP≥4). Not retracted. |
| **G3** `ff749ae` "LayerNorm 1024 = 997 cy" | survives, no corrections entry | OK. |
| (no formal V5 retractions) | | |

### V6 — none formally retracted

All V6 [x] items still hold (per `M_SYNTHESIS_INCONSISTENCY_LOG.md`).

### V7 — one possibly stale

| Item | Claim | Source of correction |
|---|---|---|
| **K2** `47b8b1d` "M11 per-pipe energy synthesis" | Citation includes MUFU per-op pJ derived from "47.8 G MUFU/s = XU peak" | The peak number is mislabeled; per `MATH_INCONSISTENCY_LOG.md` #3 the saturated MUFU = 4.74 G, not 47.8 G. V7 K2's per-pJ tabulation should be re-checked. |

### V8 — multiple `[x]` claims overturned

| Item | Claim | Source of correction |
|---|---|---|
| **NEW SMEM** `352ab1f` "SMEM BW 26.9 TB/s = 74% of 36.4 peak" | survives | `02_shmem_CORRECTED.md` agrees on the SMEM peak ladder; the 74% framing is fine. |
| **NEW DSMEM** `71934d0` "Cluster DSMEM BW = 37 TB/s = 97% of 38.5 peak" | **RETRACTED** | `DSMEM_INCONSISTENCY_LOG.md` §A: V8's compile-time invariant offsets were LICM'd / CSE'd. SASS says LD.E, ncu wavefront count was 7200 vs expected 5.9 B. Real aggregate ≈ 40 GB/s/cluster. Off by ~1000×. |
| **F1** `88ee0cf` "P2P NVLink memcpy = 778 GB/s = 86% of NVLink v7" | **RETRACTED denominator** | NVLink is **v5** not v7; spec is 900 GB/s/dir not 757; correct framing 86% read / 80% write of 900 (HEADLINE_CORRECTIONS #1 and #2). The 778 GB/s number itself appears legit; the percentage labelling and "v7" tag are wrong. |
| **NEW MUFU** `29b9b3b` "MUFU rsqrt = 99.49% XU pipe (47.8 GMUFU/s)" | **RETRACTED label** | `MATH_INCONSISTENCY_LOG.md` #3: this is 1-CHAIN LATENCY-BOUND, not the saturated peak. True saturated MUFU = 4.74 G/chip. V8 mislabels as XU pipe peak by ~10×. The cy/op number is correct but "XU peak" framing is wrong. HEADLINE_CORRECTIONS #4. |
| **NEW HMMA F16** `779e046` "HMMA.F16 = 99.90% (578.6 TFLOPS)" | survives | `06_tensor_cores_CORRECTED.md` confirms ~570-580 range. Clean. |
| **NEW HMMA variants** `475f568` "F16/F32 BF16/F32 = 578 TFLOPS" | survives | OK. |
| **NEW redux/SHFL** `9dd127f` "redux 2.5× faster than 5-round SHFL" | partial | Per `MATH_INCONSISTENCY_LOG.md` #2: 2.34× algorithm-level (V8's claim) is right; but the bare "REDUX is 4× SHFL" headline circulating elsewhere has NO source. V8 doesn't make that error; OK. |
| **NEW HBM write** `b15011b` "TMA bulk store = 7.57 TB/s = 95%" | partial | `HBM_INCONSISTENCY_LOG.md` open-question #1: this number may actually be miscredited from NINJA STG `e75c7e1`. V8 attributed to `28211ce` per HEADLINE_CORRECTIONS #3. Still under investigation. |
| **NEW FFMA peak** `8b99f43` "FFMA = 97.64% of 2032 peak (75.2 TFLOPS)" | survives, with CAVEAT | `COMPUTE_INCONSISTENCY_LOG.md` G: the 2-source recipe maxes at 75 TFLOPS but **realistic 3-source GEMM caps at 65% = ~50 TFLOPS**. V8 captures the 2-source ceiling correctly; doesn't disclose the 3-source reality in this entry (J2's "71%" hints at it). |
| **D2** `2b5f451` (V5) "FFMA pipe saturates at +124 W" | OK |
| **K1** `552d1c1` "Sustained 2032 MHz at 552 W (no throttle)" | OK |

---

## FALSELY OPEN

None found. Items that SHOULD be open were correctly carried over from V6→V7→V8 as `[ ]`. The `[~]` deferred-status convention used in V6 is honest. We did not find any item marked `[ ]` that the corrections corpus or commit history shows as completed.

(Caveat: a partial completion does not equal a "false open". V8 A-series tcgen05 items remain genuinely unsolved per `B1_DUAL_ISSUE_FFMA_IADD3.md`/`b6648e2` indirect evidence; the [ ] is correct.)

---

## DROPPED ITEMS

V3 → V4 is INTENTIONALLY non-superset: V3 footer line 251 explicitly says "SUPERSEDED 2026-04-20 by CURIOSITY_LIST_V4.md (ninja microarchitecture focus). V3 had too much LLM/framework-level work (transformer block, FlashAttn, MoE). V4 is pure low-level/SASS/microarchitecture." So V3's J (Real-workload), some of B (CUDA Graphs), C (Multi-GPU), and most of E (CUDA streams) WERE dropped — not silently, but with a documented re-scope.

V4 → V5: Topic re-scope (V5 is "questions surfaced from V4 deep dive"). Not a superset by design.

V5 → V6 → V7 → V8: Each is a "next ~50 questions" list with explicit carry-forward of deferred items only. Not full supersets.

| Drop | Where it went | Status |
|---|---|---|
| V7 §F5 "Stream with cudaLaunchHostFunc ordering" | NOT in V8 | Dropped silently; V6 F5 baseline (2.46 µs) cited but no follow-up |
| V7 §G1 "L2 prefetch.L2 hit ratio ncu confirmation" | NOT in V8 | Dropped silently; would close V6 I3 |
| V7 §H4 "Packed vs scalar cvt" | NOT in V8 | Dropped silently |
| V7 §H5 "TF32 cvt precise format (cvt.rna.tf32.f32)" | NOT in V8 | Dropped silently |
| V7 §L2-L5 "Per-warp instruction trace, kernel power waterfall, LLM Pareto plotter, auto-bisect" | NOT in V8 (V8 §L is reused for new tooling) | V8 L1 "LLM inference Pareto plotter" maps to V7 L4 — same item, kept. Others dropped silently |
| V7 §M2 "Dynamic parallelism on B300 + tcgen05" | NOT in V8 | Dropped silently |

V7 § labels A/B/C in V8 are consistent (tcgen05/multicast/NVFP4 carried). Other labels (D, E, F, G, H, I, J, K, L, M) are reused for unrelated topics in V8. This makes diff-by-label confusing but is documented at the top of each list.

---

## SUSPICIOUS HASHES

### V2 — 22 / 25 hashes are HALLUCINATED

V2 was written before the user's hash-verification rule
(`feedback_task_list_hashes`) was internalized. The TOPICS are real
investigations — only the cited hashes are wrong. Validated by topic
search:

| V2 cite | Claim topic | Real commit (verified by topic match) |
|---|---|---|
| `c0c2d48` | S2 tcgen05 alloc breakthrough | **`ec25f05`** "S2 BREAKTHROUGH: tcgen05 alloc/dealloc WORKS" |
| `affce3d` | S3 mma.sync caps 425W vs cuBLAS 940W | **`83f3dbd`** "S3 PARTIAL: mma.sync caps at 425W vs cuBLAS 940W" |
| `b602fc8` | S4 strict anti-DCE per-SMSP scaling | not found by topic; likely **`2609cc4`** "S1 RESOLVED: 4 tensor cores per SM" |
| `5291e81` | A1 FlashAttention baseline | **`d654ba9`** "A1 PARTIAL: FlashAttention cuBLAS baseline 147 TFLOPS/head" |
| `e9be3e3` | A2 RMS+bias 89% HBM | **`1f7ff0e`** "A2 RESOLVED: fused RMS+bias hits 89% HBM SoL" |
| `e49e9ef` | A3 1 GB all-reduce 1.97 ms | **`3103f51`** "A3 RESOLVED: 1 GB all-reduce on 2× B300 = 1.97 ms" |
| `2fbf49d` | A4 TP-2 GEMM 1.69× speedup | **`59580bb`** "A4 RESOLVED: TP-2 GEMM hits 1.69× speedup" |
| `e43f754` | A5 cuBLAS algoId=66 | not yet topic-located |
| `54aadb2` (valid) | B1 zero inter-GPC clock drift | actually **`caabe2f`** by topic — `54aadb2` exists in tree but isn't this topic |
| `034e2ff` | B2 L2 26.7 TB/s | not topic-located |
| `f2469ff` | B4 cudaMemset invisible to ncu | **`2ead93a`** "B4 PARTIAL: cudaMemset is INVISIBLE to ncu" |
| `9e7c592` | B5 PDL saves launch overhead | **`9318059`** "B5 RESOLVED: PDL saves ~2us/pair launch overhead" |
| `450d613` | B6 cross-GPU atomic 16.6 G | **`be2345c`** "B6 RESOLVED: Cross-GPU atomic peaks 16.6 Gatom/s" |
| `65f3795` (valid) | C1 HBM dominates power | actually **`260d89a`** by topic — `65f3795` exists but isn't this topic |
| `7741308` | C2 cudaMallocAsync 2× | **`95eaa81`** "C2 RESOLVED: cudaMallocAsync 2× faster" |
| `23a363f` | C3 cuBLAS warmup 64 ms | **`de4ca23`** "C3 RESOLVED: cuBLAS warmup = 64 ms total" |
| `d80e1f8` | C4 cudaGraph inst 2 us/node | not topic-located |
| `9e72824` | C5 cp.async cache hint | **`f961ea2`** "C5 RESOLVED: cp.async cache hint barely matters" |
| `eeb52b1` | D1 rigor_run.sh enhanced | not topic-located |
| `02f5ec1` | D2 persistent_kernel_template | not topic-located |
| `4783600` | D3 critique_finding.sh | not topic-located |
| `f19e984` | D4 PRECISION_POWER_PERF_TABLE | not topic-located |
| `45e5824` (valid) | RETRACTION pointer | OK |
| `fceb94d` | S1 90.5% (preserved-wrong) | **`9fab0e6`** "S1 RESOLVED: mma.sync hits 90.5%" |

**3 of 25 hashes valid** (`45e5824`, `54aadb2`, `65f3795`); the latter two ARE in the git tree but the message is unrelated to the V2 topic — coincidental matches, not true cites. So **all 25 V2 hashes should be treated as unverified** until matched by topic.

### V4 hashes — 32 entries cite "Commit history" instead of a hash

These 32 V4 [x] items document a real finding but provide no audit trail.
Examples: A5 (predicate file), A7 (active mask), A8 (SETP), B4 (tensor + FFMA + IMAD), B5 (FFMA + ULDC), B6 (same-pipe ILP), B7 (branch + compute), C1 (immediate width), C4 (IADD3 + predicate), D9 (`__ldg`), E1/E2/E3/E6/E8 (atomics), F1/F6 (sync), G3/G5/G10 (compiler), K1/K4 (PTX→SASS), M3/M4/M5/M6 (numerical), N1/N5 (cache), O1/O3/O4/O5/O6/O7 (surprises), P1/P2 (limits). All UNVERIFIED.

These could be cross-referenced to git via topic search if needed, but the prevalence (32/135 = 24%) is itself a reliability concern.

### V5/V6/V7/V8 hashes — clean

Every cited hash exists in the git tree. We did NOT verify that the hash's actual commit message matches the V-list claim, but spot-checks (V8 NEW DSMEM `71934d0`, V8 SHFL `5dbed87`, V6 A3 `17cf0d4`, V5 D1 `0bd3802`) all match.

---

## Summary verdict

| Version | Trust | Reasoning |
|---|---|---|
| V2 | LOW for hashes; MEDIUM for topics | 22/25 hashes hallucinated. Topics are real investigations findable by message search. Several findings retracted in-place (S1, S4, B2). |
| V3 | n/a (pure backlog) | Superseded by V4 within 24h; no work done. |
| V4 | MEDIUM-HIGH | 77/77 hashes valid; 32/135 items lack any hash ("Commit history"). One headline (FFMA+IADD3 dual-issue interpretation) updated by V49/V50. |
| V5 | HIGH | 65/65 hashes valid; no formal retractions. |
| V6 | HIGH | 57/57 hashes valid; deferred items honestly `[~]`-tagged. |
| V7 | HIGH | 44/44 hashes valid; K2 per-pipe energy synthesis depends on a mislabeled MUFU number — re-check. |
| V8 | MEDIUM | 47/47 hashes valid; **DSMEM 37 TB/s (`71934d0`) is a DCE artifact**; **NVLink "v7" framing on F1 is wrong** (use "v5", spec 900 GB/s); **MUFU 47.8 G "XU peak" is mislabeled** (1-chain latency, not saturated). The HBM write SoL attribution (`b15011b` vs NINJA `e75c7e1`) is an open question. |

The CURIOSITY lists are useful as a research timeline / backlog but should NOT be cited as authoritative findings — defer to `corrections/B300_TRUE_REFERENCE_v2_DRAFT.md` and the per-topic `corrections/*_CORRECTED.md`. The list-format `[x]` tag carries no rigor guarantee; the matching commit's audit trail (or its absence) is the only reliable signal.
