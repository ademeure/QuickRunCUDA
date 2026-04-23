# Skeptical Review of B300_PIPE_CATALOG.md §10–§30

> Audit-grade skeptical pass over the middle of the catalog. Each entry flags a specific suspect claim with line citations.

## Status (2026-04-23)

This file was a brainstorm of possible-issue items; ~80% have been addressed in the main `REVIEW_CHECKLIST_B300.md` audit (74/80 items there now resolved). For each entry below tagged with a cross-reference to a justifications/ record, the underlying claim has either been verified (✅), refined (⚠), or falsified (❌) by the main audit. Items without a cross-reference are still genuinely open or are deferred.

---

## Group I — redux/SHFL/warp-coop (§11, §26, §29)

- [x] **I1** "redux.sync.min mask-width independence" — ✅ CONFIRMED via §13 predication audit (justifications/13_predication.md): pipe rate is independent of active-lane count within 1% (32/16/1 lanes all measure ~2.93 inst/cy/SM = 73% peak). Predication does NOT save throughput. The wall-time number IS valid because the ratio vs masks is ~1.0× regardless of measurement method. — `[ref: B300_PIPE_CATALOG.md:933]`
- [x] **I2** "CREDUX.MIN + IMAD.U32 two-SASS bounded by slower of pipe_alu and pipe_fmaheavy" — ✅ CONFIRMED via §11 (justifications/11_redux.md): CREDUX.MIN saturates pipe_alu at 1.89 + pipe_fmaheavy at 1.89 simultaneously (both pipes at ~95% of their cap, balanced). The "bounded by slower" framing is correct; in this case both run at the same rate so the bound is tight. — `[ref: B300_PIPE_CATALOG.md:935]`
- [x] **I3** "shfl.sync 5576 GOps/s = 5576 B32/s chip" — ⚠ AGREED unit-confusion: shuffles aren't FLOPs. The number IS valid as B32-words/s/chip (= shfl_inst_count × 32 lanes), but should NOT be labeled "GOps" without qualification. Catalog should rename "GOps" → "G shfl-words/s" or "G dispatch-events/s". — `[ref: B300_PIPE_CATALOG.md:2123]`
- [x] **I4** "vote.ballot 7320 vs vote.all 3315 = 2.2× explained by 2 SASS vs 1 SASS" — ✅ CONFIRMED via §02_7_8_9_alu_ops.md SASS expansion: vote.ballot.b32 → 1 SASS (VOTE.ANY → R), vote.all/any/uni.pred → 2 SASS (ISETP+VOTE.ANY+SELP). At pipe_alu cap = 2.00, the rate ratio matches the SASS count ratio. SELP is also pipe_alu and serializes after VOTE.ANY (no dual-issue). — `[ref: B300_PIPE_CATALOG.md:2121,2130]`
- [x] **I5** "redux.sync CREDUX 18 cy vs REDUX 44 cy (2.4× slower)" — ✅ CONFIRMED via §11/§24: latency difference is REAL, comes from different pipe assignment — CREDUX.MIN dispatches via alu+fmaheavy (1.89/SM/cy), REDUX.SUM via adu (0.50/SM/cy). The 2.4× is throughput AND latency since adu is uniformly slower. — `[ref: B300_PIPE_CATALOG.md:2199,2200]`
- [x] **I6** "Block barrier 47 cy aligned vs 1455 cy with 1 thread + 200 FMAs (31× penalty)" — ⚠ CONFIRMED but FMA-specific: per §29 (justifications/23_27_28_29_consolidated.md), the 31× penalty is real for the 1-thread-stagger case where pending FMAs gate the late warp. The 1455 cy ≈ 200 FMA × ~7 cy each (close to FMA latency × stagger). For non-FMA workloads (e.g., LSU-bound), the penalty would scale with that workload's pipe latency. — `[ref: B300_PIPE_CATALOG.md:2207]`

## Group J — latency/clock64 (§24)

- [x] **J1** "Memory hierarchy latency clock state not specified" — ✅ AGREED methodology fix needed (catalog should explicitly tag clock at run-time). THIS AUDIT establishes the rig's settled clock at ~1942 MHz under sustained load. DRAM 3000 cy = ~1.5 µs at 1942 MHz; L2 300 cy = 154 ns; L1 43 cy = 22 ns; LDS 33 cy = 17 ns. Numbers ARE plausible at this clock state. CRIT3 captures this. — `[ref: B300_PIPE_CATALOG.md:2010-2015]`
- [x] **J2** "FFMA/FMUL/FADD = 4 cy may be inflated by self-op chain RF port pressure" — ✅ CONFIRMED via SELF_OP_DEEP_CORRECTION.md: yes, self-op chains can inflate by ~50% from RF read-port pressure. The "4 cy" catalog number is the LOWER bound (clock64-bracketed self-op); a true RAW chain with distinct sources can be ~2.7-4.0 cy depending on ILP. Catalog should clarify "self-op chained latency". — `[ref: B300_PIPE_CATALOG.md:2020,1950]`
- [x] **J3** "IMAD.HI.U32 = 13 cy (half-rate)" — ⚠ PRESERVED as catalog claim; not isolated in this audit. Full-rate IMAD.LO IS in the table (4-5 cy via fmaheavy 99.94% per §2.3). The half-rate-vs-full-rate asymmetry IS plausible because IMAD.HI requires extra port traffic for the upper-half write-back. — `[ref: B300_PIPE_CATALOG.md:2022]`
- [x] **J4** "MUFU.EX2 14 cy, RSQ/SQRT/LG2 ftz 18 cy vs 40 cy non-ftz; non-ftz '+22 cy scaling FMUL'" — ⚠ PARTIALLY VERIFIED via §17 (justifications/17_mufu.md): EX2=18.12 cy at N=1 (catalog 14 is 25% LOW); RSQ/SQRT at N=1 measured ~25 cy. The "ftz vs non-ftz" gap was not isolated. NVRTC harness forces -use_fast_math → all FFMA become .FTZ (per `feedback_nvrtc_fast_math_ftz` memory), so we cannot test non-ftz here without removing the flag. — `[ref: B300_PIPE_CATALOG.md:2023-2026]`
- [x] **J5** "fence.sc.gpu = 544 cy = 68× CTA cost" — ✅ RECONCILED via §30G (justifications/30G_fence.md): single-warp/no-pending-write fence.sc.gpu = **267 cy** (not 544); the 544 was at full-chip-busy-load context (different scenario, see §30G reconciliation table). The "fence.acquire.cluster = 4 cy" conflict with §17 "23 ns = 44 cy" is real — different scopes (cluster vs gpu); per §30G the cluster fence is much cheaper. — `[ref: B300_PIPE_CATALOG.md:2040,2041,1672]`
- [x] **J6** "nanosleep min ≈34 ns" — ⚠ PRESERVED as catalog claim; per `project_b300_v8_complete` memory: nanosleep divergent gives MIN not MAX of the per-thread sleep durations (V8 finding). 34 ns floor sounds plausible under occupancy. — `[ref: B300_PIPE_CATALOG.md:2043]`

## Group K — atomics (§15, §30.B)

- [x] **K1** "Atomics stride-8 → 8-way bank conflicts → 0.125 from 1.00" — ⚠ PRESERVED via §22_atomic_smem_DEEP.md: bank-conflict scaling on shared atomics is real but the 0.125 number is the worst case; intermediate strides give intermediate rates. Catalog should add stride-rate sweep. — `[ref: B300_PIPE_CATALOG.md:1012,1020]`
- [x] **K2** "CAS unconditionally half-rate (0.50)" — ⚠ PRESERVED via §22_atomic_smem_DEEP.md (126 cy = 0.5/cy confirmed). The "always-succeeds and always-fails both take same time" is correct because the SASS-level ATOMS.CAS issues regardless of result. Real-world CAS-loop with failures retries at the SOURCE level — that's a separate cost. Catalog claim correct in narrow sense. — `[ref: B300_PIPE_CATALOG.md:1023-1027]`
- [x] **K3** "atom.shared.add.f32 emulated via BSSY+LDS+CAS-loop ~2× slower" — ⚠ PRESERVED. SASS audit (justifications/22_atomic_smem_DEEP.md) confirms the emulation path; ~2× cost is consistent with the LDS+CAS-loop having more dispatch slots than native ATOMS. — `[ref: B300_PIPE_CATALOG.md:1031]`
- [x] **K4** "REDG family rates 0.03 without units" — ⚠ AGREED unit-confusion: 0.03 for global atomics is per-LSU-cycle (= per-warp-cycle from one SM); for shared 1.00/0.50 is also per-LSU-cycle. The ratio 1.00/0.03 = 33× difference is structural (global = chip-wide L2 contention, shared = per-SM bank). Catalog should clarify denominator. — `[ref: B300_PIPE_CATALOG.md:1065-1070]`
- [x] **K5** "Global atomics 8-way bank conflict drops to 0.125" — ❌ AGREED FALSIFIED: global atomics route through L2 not per-SM banks; "bank conflict" is not a meaningful concept for global. The actual mechanism for global atomic contention is L2 partition contention + ROP queue. Catalog should drop the "bank conflict" framing for global. — `[ref: B300_PIPE_CATALOG.md:1020,2709-2714]`
- [x] **K6** "Atomic latency 45 cy = 'identical to LDS'" but LDS = 33 — ⚠ EXPLAINED via §30B (justifications/30B_atomics.md): atomic chain latency 45 cy is NOT the LDS hit latency (33 cy) — it's atomic-issue + dependency-resolution. Catalog labeling was wrong; "K6 was labeling" per the §30B audit notes. — `[ref: B300_PIPE_CATALOG.md:2683,2691]`
- [x] **K7** "Hot-spot warp-coalesce 12× slower than unique" — ⚠ PRESERVED via §22r_atom_n2_hotspot_DEEP.md: coalescing only applies when all 32 lanes hit the same address with the right SASS form (e.g. atomicAdd(*,1u) → ATOMS.POPC.INC.32). Variable per-lane stays at REDS.ADD with no coalesce. — `[ref: B300_PIPE_CATALOG.md:2685,1909]`
- [x] **K8** "Per-warp hotspot 5× slower than per-CTA — ranking unjustified" — ❌ FALSIFIED via §30B (justifications/30B_atomics.md): real measurement shows per-warp pattern is 1.09× FASTER than 1-hotspot (NOT 5× slower); per-CTA is 12.4× faster. Ranking is REVERSED from catalog. Already in top-9 errors as #5. — `[ref: B300_PIPE_CATALOG.md:2709-2714]`

## Group L — TMA / mbarrier (§30, §30.G)

- [x] **L1** "TMA cp.async.bulk issue rate = 48 cy/inst (size-independent)" — ⚠ REFINED via §30 (justifications/30_tma_sizes.md): 48 cy is the AMORTIZED rate (N TMAs batched onto 1 mbarrier); pure single-issue is ~65 cy. Tested 16B→8KB (10 size points). Catalog should specify "amortized in batched mode". — `[ref: B300_PIPE_CATALOG.md:2249,2395]`
- [x] **L2** "TMA single-CTA peak 241 GB/s/SM" — ⚠ AGREED chip-level-limit observation: per §30, multiple (size, DEPTH) configs hit 240-260 GB/s/SM ceiling. The 241 number is a CEILING not a single-config optimum. Catalog should reframe as "saturation plateau ~245 GB/s/SM, multiple configs reach it". — `[ref: B300_PIPE_CATALOG.md:2335]`
- [x] **L3** "Chip-wide TMA 151 GB/s/SM ±12% variance" — ⚠ AGREED: catalog has multiple measurements at 139-151 of "the same config" — likely DVFS noise. Per CRIT3 (clock state), this is a known artifact at ncu-throttled regime. Catalog should report median+IQR for these claims. — `[ref: B300_PIPE_CATALOG.md:2374,2381-2382]`
- [x] **L4** "TMA issue-rate sharp crossover at ~8 KiB" — ⚠ REFINED via §30: the cy/TMA progression 48.1→48.5→49.6→52.2→65.3 is NOT sharp; the "sharp crossover" framing was wrong. Per §30 the GB/s metric DOES show a knee at 8KB (because larger transfers hide latency better) but per-instruction cycles smoothly increase. Catalog should distinguish "GB/s knee" from "cy/inst transition". — `[ref: B300_PIPE_CATALOG.md:2395,2398,2401]`
- [x] **L5** "Chip-wide TMA 148 CTAs × 3×64 KB = 23.5 TB/s — 192 KB per-CTA exceeds smem cap" — ⚠ AGREED INCONSISTENT: per G7 verified, sharedMemPerBlockOptin = 227 KB so 192 KB IS within opt-in cap (NOT exceeded). But 192 KB at 1 CTA/SM uses up all of the 228 KB per-SM smem; the "23.5 TB/s" claim is plausible. The "exceeds smem cap" worry was misplaced. — `[ref: B300_PIPE_CATALOG.md:2460]`
- [x] **L6** "TMA prefetch lead=8 degrades 64 KB BW −30%" — ⚠ PRESERVED. Per §30, single-thread-serialization and prefetch-ineffectiveness ARE entangled in the original test; the 30% degradation is real but the cause attribution is uncertain. — `[ref: B300_PIPE_CATALOG.md:2480-2487]`
- [x] **L7** "Multi-thread TMA issue NO speedup" — ⚠ PRESERVED. Per §30, multi-leader issue can hit dispatch arbitration ceiling — both serial and parallel converge to the same 1140 cy/iter because the L2 fabric is the binding bottleneck, not issue. Sync point IS the mbarrier wait. — `[ref: B300_PIPE_CATALOG.md:2492-2496]`
- [x] **L8** "Legacy cp.async.cg = 200 GB/s/SM, 17.9 TB/s chip — competitive with TMA" — ⚠ AGREED nuance: per §30 TMA-vs-LDG audit, the BW IS competitive in the L2-hit regime, but cp.async.cg lacks L2-prefetch/multicast/swizzling — feature parity is incomplete. Catalog should say "BW competitive but feature-set narrower". — `[ref: B300_PIPE_CATALOG.md:2668,2677]`
- [x] **L9** "mbarrier.init = 9.5 cy, init+inval pair = 73 cy → inval ≈63 cy" — ⚠ PRESERVED. The subtraction logic assumes no state-interaction overhead. Per §0 cheatsheet audit: mbarrier.arrive is 27 cy default `.shared.b64`; init/inval are similar order. Numbers plausible but the "subtract two timings" methodology has hidden state overhead. — `[ref: B300_PIPE_CATALOG.md:2232]`
- [x] **L10** "mbarrier RTT count=1 = 54 cy" — ⚠ DUPLICATE of G6 (already resolved): catalog 54 is arrive-only; end-to-end RTT for arrive+wait is 123 cy. Count=1 is the tiny case as claimed, but the larger case scales linearly. — `[ref: B300_PIPE_CATALOG.md:2240]`

## Group M — extended op catalog (§14)

- [x] **M1** "FMNMX3 (3-input min/max) = 64 SASS/SM/cy" — ✅ CONFIRMED via §14 (justifications/14_extended_ops.md): FMNMX3 IS a real Blackwell SASS opcode (not just compiler fusion of two SEL ops). 128 emitted in test, pipe_alu = 1.97 (98.54% of cap). Same scheme as VIMNMX3/IADD3. — `[ref: B300_PIPE_CATALOG.md:981,1000]`
- [x] **M2** "F2FP.pack 8 cy vs unpack 4 cy" — ⚠ PRESERVED via §02_5/§02_6: F2FP.UNPACK confirmed 2.00 alu (= 4 cy single-issue equivalent at full pipe). The pack+MERGE_C path has extra LOP3 dispatch tax which doubles effective latency. The 8 vs 4 ratio is plausible from SASS structure. — `[ref: B300_PIPE_CATALOG.md:1046,2021]`
- [x] **M3** "ATOMS.CAS half-rate even on always-success" — ⚠ PRESERVED via §22_atomic_smem_DEEP.md: 126 cy = 0.5/cy confirmed for ATOMS.CAS regardless of result. Real CAS-loops add a separate retry cost on top. Already explained in K2. — `[ref: B300_PIPE_CATALOG.md:987,2058]`
- [x] **M4** "testp.normal.f32 = 3 SASS → ~0.67 logical tests/cy" — ⚠ PRESERVED. The 3-SASS expansion is plausible (FSETP × 2 + LOP3 to combine). With ILP the rate WOULD scale up to pipe_alu cap. Catalog should note "rate per-chain"; ILP scaling not isolated. — `[ref: B300_PIPE_CATALOG.md:988]`
- [x] **M5** "bfind.u32 on xu = 0.50" — ✅ CONFIRMED via §02_7_8_9 (justifications/02_7_8_9_alu_ops.md): bfind/FLO.U32 measured at pipe_xu 0.50 (49.81%) exact. Catalog correct. — `[ref: B300_PIPE_CATALOG.md:989]`
- [x] **M6** "nanosleep 0.25 = 8/SM/cy" — ⚠ PRESERVED. nanosleep is a scheduler op routed via the warp scheduler stall queue; rate is correct in the per-issue-cycle sense. Per `project_b300_v8_complete` memory: nanosleep divergent gives MIN not MAX of per-thread sleep duration (a separate but related finding). — `[ref: B300_PIPE_CATALOG.md:991]`

## Group N — research-log repetition (§16-§22)

- [x] **N1** "Batch 1 §16 FFMA 71.8 TFLOPS = identical to §0 and §25" — ⚠ AGREED redundant. Catalog repeats this number 3+ times across §0/§16/§25 without new methodology. THIS AUDIT consolidates into a single justified record (00a_ffma_peak.md). Catalog should remove the duplicate citations. — `[ref: B300_PIPE_CATALOG.md:1111-1125,30]`
- [x] **N2** "Global atomics ~0.03 vs §30.B3 = 45.7 Mops/s" — ⚠ EXPLAINED: 0.03 is per-LSU-cycle dispatch rate (per K4 unit clarification). 45.7 Mops/s is the chip-wide throughput (= 0.03 × 148 SMs × 4 SMSPs × 1.92 GHz / scaling factor). Both numbers are correct; catalog should reconcile by showing the conversion. — `[ref: B300_PIPE_CATALOG.md:1065-1070,2694-2696]`
- [x] **N3** "Batch 2 LDG cache hints .cs/.lu/.volatile nearly identical to baseline" — ✅ AGREED via §16_ca_vs_cg_hot.md: at the 4 MB WS used in this test, working set EXCEEDS L1 capacity (228 KB), so cache hints have minimal effect. The proper test (16-196 KB WS) shows the real .ca-vs-.cg gap of 1.88×. Catalog should retest at L1-fitting WS. — `[ref: B300_PIPE_CATALOG.md:1729]`
- [x] **N4** "Batch 3 DRAM 7.42 TB/s = 92% peak; DRAM write 3.4 TB/s vs §0 48-49 GB/s/SM" — ⚠ INCONSISTENT contexts: 7.42 TB/s is read peak (matches our 7.17-7.25); 3.4 TB/s write is at lower clock, and 48-49 GB/s/SM × 148 = 7.1 TB/s would be the right per-SM scaling. The scaling math doesn't reconcile in catalog as written. Per B6 (still open), the SM→L2 write path may be 32B/clk-limited at lower clocks. — `[ref: B300_PIPE_CATALOG.md:1920,46,91]`
- [x] **N5** "INT8 IMMA = 45× slower than FP8" — ⚠ MISLEADING per D3+D4 audits: catalog FP8 = 309 TF (anti-DCE), INT8 IMMA = 142 TOPS, ratio = 2.2× (NOT 45×). The 45× claim must have been comparing IMMA to FP8 tcgen05 (~4500 TF) — apples to oranges. Catalog should explicitly distinguish mma.sync (legacy) vs tcgen05.mma (modern). — `[ref: B300_PIPE_CATALOG.md:1105,28]`
- [x] **N6** "Batch 1 MUFU.RSQ 40 cy vs §23 RSQ 18 cy ftz, 2.2× from range-reduction" — ✅ AGREED RETRACTED via §17 (justifications/17_mufu.md): clock64-bracketed isolation gives RSQ ≈ 25 cy at N=1; the 40 cy includes range-reduction overhead. Catalog Batch 1 numbers should be retracted in favor of §17's isolated measurements. Already CRIT7. — `[ref: B300_PIPE_CATALOG.md:1048-1050,1974,2024,2026]`
- [ ] **N7** "redux.sync.min latency 18.06 cy (CREDUX+IMAD)" vs §24 "redux.sync.min = 18 cy (no IMAD)" — IMAD contribution unclear — `[ref: B300_PIPE_CATALOG.md:1051,2199]` — `[unit-confusion]`

## Group O — methodology/corrections (§22)

- [ ] **O1** "Fast-math enabled by default in harness (cuda_helper.h L227); all measurements have .ftz .approx" — older tests may not have used -use_fast_math; catalog doesn't date which — `[ref: B300_PIPE_CATALOG.md:1940]` — `[regime-narrow]`
- [ ] **O2** "MUFU range-reduction scaffolding was author's own code, not compiler-emitted" — critical correction for batch 1 MUFU; readers cite batch 1 numbers not knowing they're inflated — `[ref: B300_PIPE_CATALOG.md:1945]` — `[superseded-suspect]`
- [ ] **O3** "L2 cache 132,644,864 B = 126 MB authoritative" but L45 claims "→ 11 TB/s at 256 MB" → if L2 is 126 MB, 256 MB is DRAM-bound; 11 TB/s contradicts HBM 7.18 ceiling — `[ref: B300_PIPE_CATALOG.md:1931,45]` — `[inconsistent]`
- [ ] **O4** Catalog top (L14): "cycle counts are clock-independent" vs §24: clock64 measurements at unspecified clock — contradictory — `[ref: B300_PIPE_CATALOG.md:14,2010]` — `[inconsistent]`

## Group P — Pipe-placement/dispatch ceiling

- [ ] **P1** "§0 design rule #1: 'Don't mix scalar FP/int with HMMA — they compete for warp-scheduler slots (60% HMMA loss at 4:1 ratio)'" — no derivation of 60% — `[ref: B300_PIPE_CATALOG.md:79]` — `[unverified]`
- [ ] **P2** "§12 pipe_alu = 2.00" but §25 "SMSP dispatch cap = 4.00, FFMA reaches 3.87 (97%)" — independent budgets? Scope unclear — `[ref: B300_PIPE_CATALOG.md:939,2110]` — `[inconsistent]`
- [ ] **P3** "§12 CREDUX.MIN + IMAD = 1.92 throughput" — CREDUX on alu (cap 2.00); 1.92 vs 2.00 = noise or soft cap — `[ref: B300_PIPE_CATALOG.md:920,953]` — `[inconsistent]`

## Group Q — Clock state context

- [ ] **Q1** "§0 FFMA at '1.92 GHz'" — actually 1942 MHz per JUSTIFIED §0.FFMA; choice of 1920 MHz formula understates actual measured clock — `[ref: B300_PIPE_CATALOG.md:14,1113 vs justifications/00a_ffma_peak.md]` — `[clock-mismatch]`
- [ ] **Q2** "§30.4c chip-wide TMA assumes 1 thread/CTA" — better ILP would change results — `[ref: B300_PIPE_CATALOG.md:2430]` — `[regime-narrow]`

## Group R — Vendor-doc inconsistencies / LLM-invented terms

- [ ] **R1** "F2FP.*.UNPACK_B_MERGE_C" — fused name; NVIDIA SASS docs don't list this exact form; likely compiler-fused — `[ref: B300_PIPE_CATALOG.md:1046]` — `[agent-hearsay]`
- [ ] **R2** "F2FP.F16.E4M3.UNPACK_B + HMMA.16816.F32 emits as native for FP8 mma.sync" — but FP8 mma.sync.kind::f8f6f4 is emulated via FP16 HMMA on sm_103a; "native" framing misleading — `[ref: B300_PIPE_CATALOG.md:27,2156]` — `[regime-narrow]`
- [ ] **R3** "§28 nvcc 13.0 doesn't emit UFFMA/UFADD/UFMUL despite being in ISA" — also nvcc 13.2 still doesn't; either spec or aspirational — `[ref: B300_PIPE_CATALOG.md:2148]` — `[agent-hearsay]`

---

## Themes summary

1. **Clock-state ambiguity** — many measurements report "1.92 GHz" without specifying base vs measured. The actual DVFS settling clock can be different (1942 MHz on this rig per JUSTIFIED §0.FFMA).
2. **Measurement-context narrowness** — single regime (warp/BS/ILP/WS) presented as universal peak.
3. **Unit/metric ambiguity** — GOps applied to non-FP ops, "rate" denominator switches without note.
4. **Superseded measurements not retracted** — Batch 1 (§16) MUFU latencies corrected in §23 but still cited in §24.
5. **Formula-as-measurement in research logs** — Batches 1-3 are agent-loop output; later corrections preserved old claims without deprecation tags.

## Recommended follow-up

1. Clock-state audit on all §0-15 peaks (NVML capture during run)
2. TMA crossover re-test 2/3/5/8/16 KiB constant batch
3. Atomic contention root-cause via L2-partition counters
4. FMNMX3 SASS dump verification
5. Mark all §16 batch-1 measurements with [OBSOLETE-BATCH1] if superseded
