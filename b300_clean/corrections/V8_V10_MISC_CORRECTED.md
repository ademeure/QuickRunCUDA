# V8/V9/V10 Miscellaneous Standalone Benches — Consolidated Corrections

Date: 2026-04-22.
Scope: V8/V9/V10 + O2/O7 standalone benches NOT already covered by topic-
specific corrections (compute, atomics, sync, tensor, launch, memory).
Originals are untouched.

Sources audited:
- `V8_J2_ENERGY_SWEEP.md`, `V8_L4_WARP_FAIRNESS.md`
- `V9_BRANCH_DIVERGENCE.md`, `V9_GRAPH_LAUNCH.md`, `V9_HMMA_LATENCY.md`,
  `V9_NANOSLEEP_LATENCY.md`, `V9_NANOSLEEP_THREADS.md`,
  `V9_OP_LATENCY.md`, `V9_REGSPILL_COST.md`
- `V10_CONCURRENT_KERNELS.md`, `V10_FMA_SOURCE_COUNT.md`,
  `V10_GLOBAL_ATOMIC.md`, `V10_GRID_SYNC.md`, `V10_LDG_WIDTH.md`,
  `V10_RETROACTIVE_VERIFY.md`, `V10_VERIFICATION_SUMMARY.md`
- `O2_TENSOR_WARMUP.md`, `O7_CLOCK_VS_CLOCK64.md`
- Cross-ref: `B300_TRUE_REFERENCE.md`, `M15_V9_LATENCY_LADDER.md`,
  topic-specific `corrections/*.md`

---

## 1. HMMA latency (V9_HMMA_LATENCY)

| Source | Latency (cy) | Shape |
|--------|--------------|-------|
| V9_HMMA_LATENCY.md | **20.09 cy converged** (chain=4096) | m16n8k16 F32-acc F16 |
| M15_V9_LATENCY_LADDER.md | **20 cy / 9.8 ns** | matches |
| 06_tensor_cores_CORRECTED.md | "20 cy" | matches |
| B300_TRUE_REFERENCE.md | "20 cy" | matches |

**Verdict: CONSISTENT across all sources.** Saturation = 5 chains/SMSP per
V9_HMMA_LATENCY (20 cy / 4 cy throughput). Header memory `HMMA latency 20 cy`
matches.

Untested (per V9): m16n8k8 shape, .F16 accumulator, tcgen05.mma latency.

---

## 2. nanosleep — divergent semantics (V9_NANOSLEEP_LATENCY vs V9_NANOSLEEP_THREADS)

V9_NANOSLEEP_LATENCY: solo-thread rounding behavior:
- Floor ≈ 128 ns
- 500–1000 ns near-exact (~1.02×)
- 5,000 ns → 8,171 ns (1.63×, rounded to ~next pow2)
- 10,000 ns → 16,384 ns (= 2^14)
- 100,000 ns → 131,072 ns (= 2^17)
- 500,000 ns → 521,666 ns (1.04×)

V9_NANOSLEEP_THREADS: warp-divergent values cause **MIN, not MAX**:
- 16 lanes × 1000 ns + 16 lanes × 100 ns → 127 ns observed
- "lone lane sleeps 1000" → 1018 ns (uniform-ish)
- diverse 100..3200 → 351 ns

**The rule "divergent = MIN not MAX" is ONLY in V9_NANOSLEEP_THREADS.** The
V9_NANOSLEEP_LATENCY doc never references the divergent issue and a casual
reader could miss it.

The CLAUDE.md memory entry "nanosleep divergent = MIN not MAX" matches
V9_NANOSLEEP_THREADS exactly.

Cross-check vs `16_power_clock_CORRECTED.md`: that doc lists the rounding
behavior (-35 to -60% undershoot above 1 µs) but does NOT mention the
divergent MIN behavior. **Recommend** the divergent-MIN warning be promoted
into 16_power_clock_CORRECTED.md and/or 08_sync_primitives_CORRECTED.md.

---

## 3. Register spill cliff (V9_REGSPILL_COST)

| SPILL_VARS | GFFMA/s | Slowdown | Spill? |
|------------|---------|----------|--------|
| 8 | 35.6 | 1.00× | No |
| 16 | 37.2 | 0.96× | No |
| **32** | **4.01** | **9.25×** | **YES — cliff** |
| 64 | 1.95 | 19× | YES (2242 STL/LDL in SASS) |
| 128 | 1.79 | 20.8× | YES |

Threshold matches register budget: 65536 regs / 256 thr / 8 blk = 32 regs/thr
under `__launch_bounds__(256, 8)`.

**The "9× cliff at 32 vars" is consistent** with header memory entry "Spill
cliff 9× at 32 vars" (V8 finding listed in CLAUDE.md memory; actually V9 doc).

Caveat: the reported 35.6 GFFMA/s baseline is "50% of 75 TFLOPS peak" because
the test uses small ITERS — startup dominates. The MULTIPLIER (9×) is what
matters, not absolute throughput.

---

## 4. FMA source count (V10_FMA_SOURCE_COUNT)

| Variant | PTX | pipe_fma % | TFLOPS |
|---------|-----|------------|--------|
| 2-source | `fma %0, %0, %1, %0` | **97.65%** | 75.2 |
| 3-source | `fma %0, %0, %1, %2` | **66.64%** | 51.3 |

3-source / 2-source = 0.683 ≈ 2/3, consistent with B300's 2-RF-read-port
limit per SMSP (V6 D6).

**Cross-reference with FFMA agent finding** ("all near-peak FFMA recipes use
≤2 unique register sources"):
- `V8_FFMA_PEAK_VERIFIED.md` recipe = `fma %0, %0, %1, %0` (2 unique)
- `04_fp32_peak_CORRECTED.md` row: 2-source 75.2 TFLOPS / 97.65%; 3-source
  drops to ~65% pipe
- V10_FMA_SOURCE_COUNT directly demonstrates the gap

**CONFIRMED**: every "near-peak" FFMA in the catalog uses ≤2 unique register
sources. The 3-source path caps at 67% pipe due to RF read ports. Real-world
GEMM `c = a*b + c` is 3-source → caps at ~51 TFLOPS scalar FP32.

---

## 5. Branch divergence (V9_BRANCH_DIVERGENCE)

The doc contains TWO inconsistent tables in the same file:

(a) "TRUE divergence" (different instr types per branch):
- 2-way (FFMA + rsqrt) → **2.57×**
- 4-way → 5.78×
- 8-way → 14.17×

(b) "Original simple-case" (same instr type, different constants):
- 2-way → 1.09× ("essentially free, predicated")
- 4-way → 6.65×
- 32-way → 85×

The "CORRECTION" header at the top says (b) was "compiler predication, not
true divergence". So **the authoritative numbers are (a)**.

**Conflict with prior memory:** `project_b300_session2.md` says "2-way thread
divergence: ZERO cost" — that came from the SIMPLE/predicated case (b). After
V9_BRANCH_DIVERGENCE's correction, the truthful claim is **2-way TRUE
divergence costs 2.57×**, not free. The session2 line is now stale and should
be qualified ("predicated only — true div is 2.57×").

Memory entry `project_b300_v8_complete.md` already captures BOTH
interpretations explicitly (predicated 1.09× / true 2.57×) — that one is
correct.

---

## 6. LDG width (V10_LDG_WIDTH)

| Width | Time | BW | Inst count |
|-------|------|------|------------|
| 32-bit | 993 µs | 1.95 TB/s | 72.8M |
| 64-bit | 532 µs | 3.65 TB/s | 37.8M |
| **128-bit** | **337 µs** | **5.76 TB/s (80% peak)** | 20.3M |

128-bit is **2.95×** faster than 32-bit, identical DRAM bytes (1.94 GB) and
L1 sectors (60.6M) — pure dispatch-overhead reduction.

Ladder for HBM read (V10_LDG_WIDTH):
- 32-bit LDG: 1.95 TB/s
- 64-bit LDG: 3.65 TB/s
- 128-bit LDG: 5.76 TB/s
- cp.async.ca: 6.98 TB/s (97% peak)
- TMA bulk: ~7.5 TB/s (95–100% peak)

**Best for HBM peak: 128-bit LDG (`.128`)** — confirms header memory.
For absolute HBM SoL still cp.async.ca / TMA bulk.

---

## 7. Concurrent kernels (V10_CONCURRENT_KERNELS)

128 hardware slots. N=148 → 2 batches (11374 us ≈ 2 × 5666). Up to N=128 all
parallelize linearly (122× speedup at N=128). N>128 splits into ⌈N/128⌉ batches.

**Consistent with prior catalog claim "128 HW slots"**. V10 simply tightened
the measurement.

Caveat: this is one CTA per kernel, not concurrent kernels in the
`cudaStreamCreate` sense — it measures the kernel-dispatch slot table.

---

## 8. Grid sync cost (V10_GRID_SYNC)

`grid.sync()` = 2376 cy = 1170 ns @ 2.032 GHz = **79× __syncthreads**.

Consistent with topic-specific `08_sync_primitives_CORRECTED.md` (which
already lists grid.sync at 2376 cy). No conflict.

Practical guidance: avoid in inner loops; use mbarrier (123 cy) or cluster
barrier (370 cy) when the scope allows.

---

## 9. Global atomic U-curve (V10_GLOBAL_ATOMIC)

(Atomic throughput is covered by atomics agent. Non-throughput aspect:)
**The U-curve shape is the new finding**: CONTEND=1 (50 GRED/s) > CONTEND=2
(3.15 GRED/s, WORST) > then monotonically recovers to CONTEND=37888
(590 GRED/s). Mechanism: HW warp-combiner only works at CONTEND=1.

Practical: avoid 2–8 hot spots in histogram/binning. Either use a single
combined hot spot (HW combiner kicks in) or use SMEM atomics (contention-
invariant, 17 µs across all CONTEND values).

---

## 10. V8 J2 — energy sweet spot (V8_J2_ENERGY_SWEEP)

Compute-bound (FFMA) energy minimum: **1500 MHz** (137 GFLOPS/W vs 124 at
boost vs 121 at 1005). Memory-bound (DRAM): **1005 MHz** (30.0 GB/s/W vs
26.2 at boost). Idle baseline: 153 W @ 1005, 197 W @ 1920.

Cross-check vs CLAUDE.md memory `project_b300_v6_complete`: "USE BOOST CLOCK
(3× lower energy than 510)". V8_J2 explicitly addresses this — V6 compared
against 510 MHz, where things crawl. In the practical 1005–2032 range BOOST
loses to 1500 (compute) and 1005 (memory). **Both are correct in their
respective contexts**.

---

## 11. V8 L4 — scheduler fairness (V8_L4_WARP_FAIRNESS)

- 1 blk/SM (148 blocks): CV 0.76%, max/min 1.03× — TIGHT
- 8 blk/SM (1184 blocks): CV 24.9%, max/min 1.98×
- 16 blk/SM (2368 blocks): CV 17.9%, max/min 1.98× + 28.5 µs queue drain
- Block dispatch rate: ~85 ns/block
- Per-warp FFMA chain: 2.12 ns/FFMA = 4.07 cy @ 1920 MHz (matches FFMA
  latency catalog)

No external conflicts. The 4.07 cy at 1920 MHz scales to 4.30 cy at 2032 MHz —
slightly off from the V9 4.22 cy figure; both within ±2% noise.

---

## 12. V9 graph launch (V9_GRAPH_LAUNCH)

| Path | µs/launch |
|------|-----------|
| Direct cudaLaunchKernel | 2.06 |
| Graph (1-kernel) | 2.05 (no speedup!) |
| Graph (100-kernel batched) | 0.54 (3.84×) |

Already incorporated into `10_launch_overhead_CORRECTED.md` and
`TMA_LAUNCH_INCONSISTENCY_LOG.md` (myth bust: cudaGraph single ≠ speedup).
Memory entry `cudaGraph single = no speedup` matches.

---

## 13. V9 op latency (V9_OP_LATENCY)

| Op | Latency (cy) |
|----|--------------|
| FFMA / FADD / FMUL | 4.219 |
| IMAD | 4.252 |
| DFMA | 63.677 |

Consistent with M15 ladder, 04_fp32_peak_CORRECTED.md, B300_TRUE_REFERENCE.

---

## 14. O2 tensor warmup (O2_TENSOR_WARMUP)

**No measurable cold-start penalty for mma.sync on B300**. Steady-state =
20 cy/MMA from the very first instruction. Tested: m16n8k16 BF16, single
warp at 1500 MHz. NOT tested: tcgen05.mma; long-idle; cross-kernel.

Practical: don't insert dummy MMAs to "warm up" — driver/dispatch latency
is the actual first-MMA cost, not tensor cores.

This finding is NEW and not yet captured in any topic-specific correction.

---

## 15. O7 clock() vs clock64() (O7_CLOCK_VS_CLOCK64)

**Surprise: clock() is 2× MORE expensive than clock64().**

| Source | cy/op (overhead vs no-clock baseline) |
|--------|---------------------------------------|
| `clock()` | 4.00 cy |
| `clock64()` | 2.125 cy |
| `globaltimer` | 2.125 cy |

SASS evidence: same `CS2R` opcode for all three. Difference is post-processing
(32-bit shift/mask for clock()).

Methodology note (CRITICAL): initial test reported clock64() at 6.75 cy
because the inner-loop chain DCE'd 6 of 8 reads. Fixed via
`acc = acc * 31ull + c` to force a multiplicative chain dependency — only
then did all 8 CS2R appear in SASS.

**Practical**: always prefer `clock64()` for in-kernel timing. Both are
~2 cy each — essentially free. Avoid `clock()` (slower AND wraps every
~2 sec at 2 GHz).

This finding is NEW and not yet captured elsewhere. Should propagate into
the rigor/methodology cheatsheet.

---

## 16. V10 retroactive verification (V10_RETROACTIVE_VERIFY / V10_VERIFICATION_SUMMARY)

After DSMEM DCE bug, all major BW/throughput claims were SASS+ncu re-verified.

### Survivors (SASS+ncu match expected ops):
- V8 FFMA 75.2 TFLOPS (97.64%)
- V8 SMEM LDS 26.9 TB/s
- V9 cp.async 6.98 TB/s
- V9 L2 BW 10–14 TB/s
- V8 HBM write 6.11 TB/s
- V10 streaming 19.4 TB/s (L1)
- All clock64-based latency measurements (DCE-immune by construction)

### Retracted (DCE'd):
- V8 DSMEM 37 TB/s
- V10 DSMEM 48–67 TB/s
- V10 DSMEM "writes 4× slower than reads"

These are already retired in `DSMEM_CORRECTED.md` /
`DSMEM_INCONSISTENCY_LOG.md`. Cross-ref consistent.

### Methodological codification (now in M5/rigor protocol):
1. Count target op in SASS loop body
2. Compute expected = inner_count × iters × warps
3. Compare ncu wavefront/sector count vs expected
4. If ncu ≪ expected → DCE detected → REJECT
5. Latency via clock64 dep-chain is DCE-immune

---

## RETRACTIONS (from this audit)

1. **V9_BRANCH_DIVERGENCE original (predicated) table**:
   2-way "1.09× / essentially free / predicated" is NOT a divergence cost —
   it's compiler predication of switch-on-constant. The TRUE-divergence
   table (different instr types) shows 2-way = 2.57×. The original "ZERO
   cost" claim in `project_b300_session2.md` should be qualified to
   "predicated only".

2. **V8/V10 DSMEM throughput** (37 / 48 / 67 TB/s) — already retracted in
   DSMEM corrections; restated here for completeness.

3. **First clock64 measurement (O7) of 6.75 cy/op** — DCE'd; corrected to
   2.125 cy with proper anti-DCE chain.

4. **CLAUDE.md memory `cudaGraph single = no speedup`** — already
   correct; original 10_launch_overhead.md "graphs 35% cheaper" was
   misleading (only true when batched).

---

## UNRESOLVED / DEFERRED

### HMMA / tensor latency
- **m16n8k8** vs m16n8k16 latency — UNTESTED
- **.F16 accumulator** vs .F32 latency — V8 showed throughput equal but
  latency UNTESTED
- **tcgen05.mma latency** — DEFERRED in V9, never re-tested
- **Cross-kernel / long-idle warmup** — O2 only tested intra-kernel

### nanosleep
- Does the 2^N rounding pattern depend on clock state (boost vs locked)? — DEFERRED in V9_NANOSLEEP_LATENCY
- Can nanosleep be preempted by higher-priority work? — DEFERRED
- Is there a "nanosleep.u32 0" special-case opcode? — DEFERRED
- What is the EXACT divergent-warp scheduling rule? V9_NANOSLEEP_THREADS
  rates "MIN" only at MEDIUM confidence

### Spill cost
- The 9× cliff was measured for FFMA chain — does it generalize to
  memory-bound or tensor kernels? — UNTESTED

### Concurrent kernels
- Limit measured at one CTA per kernel; what about kernels that need
  many CTAs? Hardware slot count remains 128 but slot-vs-CTA accounting
  unclear — UNTESTED

### Branch divergence
- PTX switch vs nested-if compilation differs — UNTESTED
- Long-body branches may lose predication advantage — UNTESTED
- Did NOT test in presence of memory traffic (bypass/serialization
  could compound)

### O7 clock vs clock64
- "Shift/mask explanation" for the 2× gap is MEDIUM confidence — could
  be measurement noise around the 2-cy difference. SASS instruction-by-
  instruction cycle accounting needed for full certainty

### Global atomic U-curve
- 590 GRED/s peak claimed at CONTEND=37888 — not cross-checked against
  the atomic-throughput agent's findings; possible double-counting via
  REDG vs ATOM optimization

### V8 J2 energy
- DRAM "optimum 1005 MHz" reported BW = 7057 GB/s, which exceeds the
  HBM3E catalog peak of 7.31 TB/s only by counting both directions —
  unclear whether the test is pure read or R+W mix

---

## Confidence assessment

| Claim | Confidence | Method |
|-------|-----------|--------|
| HMMA latency 20 cy | HIGH | 4 chain lengths converge; cross-confirmed M15 |
| Spill cliff 9× at 32 vars | HIGH | SASS shows STL/LDL count |
| 2-source FFMA 97% / 3-source 67% | HIGH | ncu pipe_fma direct, V6 D6 confirms |
| 128 concurrent kernel slots | HIGH | Linear up to 128, exact 2× at 148 |
| 128-bit LDG = 2.95× faster | HIGH | SASS+ncu, identical DRAM bytes |
| grid.sync = 2376 cy | HIGH | clock64-based, 1001-call avg |
| Tensor no warmup | MED-HIGH | Single test config |
| clock() = 2× clock64() cost | HIGH on number, MED on mechanism | SASS confirmed; explanation uncertain |
| Divergent nanosleep = MIN | HIGH for "≠ MAX", MED for exact "MIN" | Multiple modes tested |
| 1500 MHz energy optimum | HIGH | 3 clock points × steady-state |
| Scheduler fair (CV 0.76%) | HIGH | per-warp globaltimer |
| Graph single = no speedup | HIGH | Multiple batch sizes |
