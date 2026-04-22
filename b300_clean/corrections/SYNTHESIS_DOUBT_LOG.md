# SYNTHESIS DOUBT LOG (2026-04-22)

Adversarial audit of `MASTER_INDEX.md`, `HEADLINE_CORRECTIONS.md`,
`B300_TRUE_REFERENCE_v2_DRAFT.md`. Severity: **CRIT** (false claim) /
**HIGH** (overstated / premature) / **MED** (missing nuance) / **LOW**
(wording).

---

## CRIT — none found.
The headline retractions all have at least one upstream log row supporting them.
The NVLink-5 claim is web-confirmed (NVLink-5 = Blackwell, 1.8 TB/s bidi =
2× NVLink-4; "v7" was a hallucination that propagated to memory).

## HIGH

### H1. "MUFU mislabeled by ~10×" overstates the M14/M16 wording.
Headline #4 + Top-20 #9 say "M14/M16 mislabeled by ~10×". The actual M14/M16
phrasing (verified) is `"MUFU rsqrt 47.8 GMUFU/s @ 99.49% XU"` with the
`rsqrt.approx.f32` recipe noted. M14/M16 reported a real measurement of a
real (latency-bound) regime; they did not claim "saturated XU pipe" anywhere
on those lines. The MATH log itself says "1-chain rsqrt latency-bound 47.8 G;
saturated MUFU 4.74 G". The synthesis flattens "ambiguous label" into "10×
mislabel" — that is fair if read as "the headline number is 10× the saturated
peak", **misleading** if read as "M14 reported a wrong number". Recommend:
soften to "M14/M16 row is 1-CHAIN latency-bound, not pipe-saturated; saturated
peak is 10× lower at 4.74 G".

### H2. V46 7.20 TB/s "BELOW V44/V45 ceilings" — V44/V45 don't have HBM ceilings.
TRUE_REFERENCE_v2 §1 row says V46 "is BELOW V44/V45 ceilings". V44 and V45
are the bank-conflict regime experiments, NOT HBM. The HBM comparators are
V33 (single-deep TMA, 6.72) vs the 01 sub-optimal table (TMA bulk 7.344, LDG
7.365). Should read "is BELOW the 7.344 TMA bulk + 7.365 LDG.E.128 ceilings
already in 01". HBM log #5 has it correct; the v2 draft text is sloppy. **Fix
the row text.**

### H3. HBM denominator dispute is collapsed prematurely.
HBM log #1 lists THREE denominators (7672 post-ECC, 7.31 empirical pure-dir,
8.0 nominal) and the agent recommends standardizing on 7672. But the V32–V48
findings authors deliberately picked 7.31 as "empirical pure-direction" — a
defensible choice for "% of what this kernel can achieve in isolation".
The synthesis converts disagreement into "denominator artifact" without
acknowledging that 7.31 is itself a measurement (in 01's R:W sweep, "**7.31
TB/s = 95.3%** R-only"). Both denominators are legitimate for different
questions. Headline #3's "V46 used 7.31 (empirical) as denominator instead of
7.672 (post-ECC spec)" reads as if 7.31 were a mistake; it isn't.
Recommend: keep both framings explicit.

### H4. "DSMEM 3.06 TB/s aggregate was sum across clusters of mixed measurements"
TRUE_REFERENCE_v2 §1 supersedes-note says v1's 3.06 was "a sum across clusters
of mixed measurements". V1 actually says: `"DSMEM (CL=2) 3.06 TB/s
(cluster-internal) — aggregate; per-cluster 41 GB/s (4129760)"`. V1 was
explicitly per-cluster 41 GB/s × ~74 = 3.06 chip aggregate, which is
INTERNALLY CONSISTENT with the new "40 GB/s per cluster read" claim. The
disagreement isn't "mixed measurements" — it's that v1 reported chip-aggregate
read while v2 reports per-cluster read AND swaps in writes (560 GB/s per
cluster). The synthesis is silent about whether v1's 3.06 is wrong or merely
different scope. Recommend: add a row "DSMEM read aggregate (chip)" =
~40 × 74 = ~3.0 TB/s, consistent with v1.

### H5. "BF16 1543 / FP8 7500-8200 RETRACTED" framed as new finding.
HEADLINE / Top-20 #20 / TRUE_REFERENCE_v2 retraction list all present these
as wave-1+2 contributions. META log B1/B2/B3 explicitly says all three were
"**ALREADY RETRACTED in-place**" in the original docs. The synthesis takes
credit for retractions that the originals already made. Recommend: cite as
"already retracted upstream; surfaced here for emphasis".

## MED

### M1. Cross-agent disagreement on DSMEM writes is flattened.
DSMEM log B says writes are "**~13× FASTER than reads**" (560 vs 40 GB/s).
HEADLINE #7 says "writes are FASTER, not 4× slower" + "Local/DSMEM ratio =
7.5×". The 7.5× number applies to LATENCY (DSMEM vs local SMEM), not to
write/read ratio. TRUE_REF_v2 §1 keeps the units clean but the headline is
ambiguous about which "ratio" the 7.5× refers to. Easy to misread.

### M2. NVFP4 11.42 PF "record" — no SASS / ncu citation.
TRUE_REFERENCE_v2 row "NVFP4 cuBLAS + cudaGraph BPG=16 = 11423 TF = 76.2%" is
sourced to `NVFP4_CUDAGRAPH.md`. Was rigor_run.sh applied? CLAUDE.md §6 says
sub-agent outputs are NOT authoritative without 3-method verification. The
synthesis promotes this to a "v2 record" without noting whether NVFP4_CUDAGRAPH
ran the rigor protocol. The K=96 ULTRA "10890 / 14780 TF/s" rows have the
same issue — "98.5% per CTA" is a per-CTA microbench number being presented
in the chip TFLOPS column without an explicit "(per-CTA microbench, not chip
GEMM)" tag.

### M3. IADD3 / PRMT 30% gap left "UNRESOLVED" but listed in pipe ladder.
COMPUTE log §E + INT log #A flagged a 30% disagreement between V40 (0.66) and
A6 (0.50) for IADD3. TRUE_REF_v2 §6 lists "IADD3 = 0.66 inst/cy (V40) — A6
says 0.5; UNRESOLVED 30% gap" but then derives "26 Glane/s" using only V40's
number. Either pick the conservative number (A6) for the headline or quote a
range. Same for PRMT (0.36 vs 0.5).

### M4. "Cluster max usable = 16" lacks supporting test-commit cite.
TRUE_REF_v2 §7 row supersedes "8 verified" with "16 (non-portable)". The
underlying TOPOLOGY log #4 source isn't shown in the v2 row. Without a
commit/SASS reference, "16 non-portable" is an unverified upgrade.

### M5. Persistent kernel "2.03 µs" — claim that v1 used release variant.
TRUE_REF_v2 §3 "supersedes v1 4 µs — that used release variant emitting
MEMBAR.ALL.SYS". V1 just lists "4 µs (584fda6)" — does the synthesis have
SASS evidence that 584fda6 used release semantics? If not, this is a guess
about why v1's number was higher. Demote to "hypothesis: v1 likely used
release; not verified".

## LOW

### L1. MASTER_INDEX §1 mapping — DSMEM_REFERENCE.md is in the originals
column but is itself an inter-corrections-era document; it doesn't have a
"_CORRECTED" sibling. Note explicitly that DSMEM_CORRECTED.md *consolidates*
DSMEM_REFERENCE.md + V11–V31, not "supersedes" it.

### L2. HEADLINE_CORRECTIONS Top 10 §6 wording: "100% / 114 TOPS combined" is
attributed to catalog `a0bde33`/`f578755` but COMPUTE log §F lists THREE
older claim sources (a0bde33, f578755, d2e3212). Add d2e3212 for completeness.

### L3. TRUE_REF_v2 §3 row `__threadfence_system` "1750–3042 cy / 861–1486 ns"
— the ns range is computed at unspecified clock. At 2032 MHz: 861–1497 ns.
At 1920: 911–1584 ns. Pick one and state it.

---

## What looks rigorous

- **NVLink-5 generation correction**: Web-confirmed. Memory note will need
  updating.
- **DSMEM DCE retraction (V8/V10)**: well-documented in DSMEM log A/B with
  explicit SASS/ncu evidence cited. Synthesis preserved the underlying
  story faithfully.
- **3-source FFMA RF-port cap (50 vs 75 TFLOPS)**: A4/D6 cited, COMPUTE log
  §G lays out the 2-port + reuse-cache mechanism. New caveat is well-grounded.
- **`pipe_tensor.cycles_active` does not measure tcgen05**: TENSOR log §B is
  rigorous; this is a real methodology fix.
- **K-uniform "28% saving" RETRACTED as background contamination**: NVFP4 log
  audits this carefully against repeated runs; the RETRACTION is well-supported.

## Net assessment

The synthesis is broadly faithful but **takes credit for some upstream
retractions** (1543/7500-8200), **prematurely picks one denominator in the
HBM debate**, **conflates measurement scopes in the DSMEM 3.06 TB/s
supersession**, and **flattens 30% UNRESOLVED gaps into headline single
numbers** (IADD3, PRMT). No CRIT-level fabrication detected.
