# V8 / V9 / V10 Misc Standalone — Inconsistency Log

Date: 2026-04-22.
Topic-specific logs (atomics, sync, tensor, launch, compute, dsmem) cover
their own conflicts. This log captures contradictions in the V8/V9/V10
miscellaneous standalone benches (concurrency, regspill, HMMA-latency,
divergence, energy, scheduling, LDG-width, graph-launch, nanosleep,
clock-vs-clock64, tensor-warmup, retroactive-verify).

---

## Conflict A — "Branch divergence cost" — predicated vs true

**Same file** (`V9_BRANCH_DIVERGENCE.md`) presents TWO incompatible tables.

| Source | 2-way | 4-way | 8-way / 32-way |
|--------|-------|-------|------------------|
| V9_BRANCH_DIVERGENCE "TRUE" (with CORRECTION header) | **2.57×** | 5.78× | 14.17× (8-way) |
| V9_BRANCH_DIVERGENCE "Original simple-case" (predicated) | 1.09× | 6.65× | 85× (32-way) |
| `project_b300_session2.md` memory | "ZERO cost" (= predicated) | 6–60× | — |
| `project_b300_v8_complete.md` memory | both reported (1.09 / 2.57) | both | both |

**Resolution**: the TRUE-divergence numbers are authoritative. session2 memory
predates the correction and is stale — recommend qualifying the line to
"predicated only; true divergence is 2.57×".

**Reader hazard**: V9_BRANCH_DIVERGENCE keeps both tables in the body, with
only a small "CORRECTION" header at top. A skimmer can easily pick the wrong
table.

---

## Conflict B — nanosleep divergent semantics not cross-referenced

**Across files** the rule "divergent values → MIN, not MAX" appears only in
`V9_NANOSLEEP_THREADS.md`.

- `V9_NANOSLEEP_LATENCY.md` describes only solo-thread rounding.
- `16_power_clock_CORRECTED.md` lists rounding behavior but NOT the
  divergent issue.
- `08_sync_primitives_CORRECTED.md` does not mention nanosleep at all.

**No factual contradiction**, but a reader using V9_NANOSLEEP_LATENCY in
isolation will design a divergent-nanosleep kernel and get unexpectedly short
sleeps.

**Recommend**: cross-reference V9_NANOSLEEP_THREADS from V9_NANOSLEEP_LATENCY,
and add the divergent-MIN warning to 16_power_clock_CORRECTED.md.

---

## Conflict C — "near-peak FFMA" recipe vs realistic GEMM

| Source | FFMA pipe % | TFLOPS |
|--------|-------------|--------|
| V8_FFMA_PEAK_VERIFIED.md (2-source `fma %0,%0,%1,%0`) | **97.65%** | 75.2 |
| V10_FMA_SOURCE_COUNT (3-source `fma %0,%0,%1,%2`) | **66.64%** | 51.3 |
| V8 J2 narrative ("FFMA 71%") | 71% | implied lower |
| 04_fp32_peak_CORRECTED.md | both rows present | both |

**Not a contradiction** but a definitional one: "peak FFMA" in the catalog
implicitly assumes a 2-source recipe (no realistic GEMM achieves it).

V8 J2's "71%" is consistent with V10's "67% + ~4% startup" for 3-source.

The FFMA agent finding ("all near-peak FFMA recipes use ≤2 unique register
sources") is **VERIFIED** by V10_FMA_SOURCE_COUNT — the gap is a clean 2/3
matching the 2-RF-port theory.

---

## Conflict D — V8 J2 "DRAM 7057 GB/s @ 1005 MHz" vs HBM peak

- V8_J2_ENERGY_SWEEP table 2 lists 7057 GB/s for the DRAM-ish workload at
  1005 MHz.
- B300_TRUE_REFERENCE caps HBM3E read peak at ~7.31 TB/s.
- V6 C2 kernel (the basis for J2's "DRAM" table) is described as a "256 MB
  buffer — partial L2".

**Possible explanation**: partial L2 hits inflate the apparent BW above the
DRAM ceiling. Doc should say "L2-amplified BW" rather than "DRAM BW" for
clarity. Mild conflict only.

---

## Conflict E — V8 L4 chain latency 4.07 cy @ 1920 MHz vs V9 4.22 cy

- V8_L4_WARP_FAIRNESS reports 2.12 ns/FFMA = **4.07 cy** at 1920 MHz
  (per-warp clock measurement).
- V9_OP_LATENCY reports **4.219 cy** for FFMA (chain=4096 converged).
- M15_V9_LATENCY_LADDER and B300_TRUE_REFERENCE list 4.22 cy.

**Discrepancy**: 4.07 vs 4.22 = **3.6%** gap. Both within "4-6 cy" literature
range but not identical. Likely sources:
- V8 L4 measured 1024 FFMAs (short chain); V9 used 4096 (converged)
- Clock calibration noise between the two runs
- V8 L4 used `__nanosleep`-derived `%globaltimer`; V9 used `clock64`

**Recommend**: V8 L4 should re-state as "4.07 cy ± noise", or use the
converged V9 figure as authoritative.

---

## Conflict F — "128 concurrent kernel slots" — V10 vs prior catalog

- V10_CONCURRENT_KERNELS measures **exactly 128** via clean batch-doubling.
- CLAUDE.md memory `project_b300_session2.md` says "Coordination ladder,
  dispatch limits (128)".
- Older catalog in `_run.sh` examples / older docs sometimes implied "no hard
  limit on streams".

**No contradiction with V10**: the prior "128 HW slots" matches. V10 just
provides the rigorous scaling curve.

**Reader hazard**: "concurrent kernels" can mean (a) HW slots tracking
distinct kernel IDs OR (b) total CTAs running. V10 measures (a) with 1 CTA
per kernel. Total CTAs in flight is bounded by SM × occupancy = much higher.
Doc should clarify "kernel slots, not CTAs".

---

## Conflict G — Tensor warmup: O2 says "no warmup" — vs general lore

- O2_TENSOR_WARMUP: no measurable warmup intra-kernel for mma.sync.
- Common practice in cuBLAS/cuDNN: insert dummy MMAs before timing.
- TENSOR_INCONSISTENCY_LOG / `06_tensor_cores_CORRECTED.md` discuss
  cudaGraph + sustained throughput differences at very long durations.

**No direct conflict**. O2's claim is narrow: "intra-kernel mma.sync = no
warmup". The "warmup" cuBLAS / cuDNN need is dispatch / cudaGraph
instantiation / clock ramp, not the tensor pipe itself.

---

## Conflict H — clock64 cost: O7 corrected itself

`O7_CLOCK_VS_CLOCK64.md` openly admits the FIRST measurement (clock64 =
6.75 cy/op) was DCE'd — only after `acc = acc * 31ull + c` did the SASS
include all 8 CS2R instructions.

**Lesson reinforced**: anti-DCE chains MUST use real arithmetic dependence
(multiply or non-trivial fold), not just `acc += x`. A simple add with
predictable values can still be CSE'd.

This methodological lesson is consistent with V10_RETROACTIVE_VERIFY's
broader warning ("count load instructions in loop body, compare ncu against
expected").

---

## Conflict I — V10 Retroactive Verify list vs current TRUE_REFERENCE

V10_VERIFICATION_SUMMARY lists DSMEM as "not reliably measurable".
B300_TRUE_REFERENCE.md and DSMEM_REFERENCE.md may state higher numbers in
older sections.

This is already tracked in `DSMEM_INCONSISTENCY_LOG.md` — listed here only
to flag the cross-reference to V10's invalidation mechanism (DCE due to
inline-offset PTX with invariant base).

---

## Summary of contradictions actionable

1. **V9_BRANCH_DIVERGENCE** — predicated vs true tables in same file confuse
   readers; flag prominently or remove the predicated table.
2. **session2 memory "2-way ZERO cost"** — stale; should be "predicated 2-way
   only".
3. **V9_NANOSLEEP_LATENCY** does not warn about divergent-MIN — add cross-ref.
4. **V8 J2 "DRAM 7057 GB/s"** — should be labeled L2-amplified.
5. **V8 L4 4.07 vs V9 4.22 cy** — V9 is authoritative; V8 L4 should
   acknowledge.
6. **V10_CONCURRENT_KERNELS** — clarify "kernel slots ≠ CTAs".

No HIGH-severity correctness conflicts; mostly documentation/cross-reference
hygiene.
