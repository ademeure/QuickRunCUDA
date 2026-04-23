# Skeptical Review of B300_PIPE_CATALOG.md §10–§30

> Audit-grade skeptical pass over the middle of the catalog. Each entry flags a specific suspect claim with line citations.

---

## Group I — redux/SHFL/warp-coop (§11, §26, §29)

- [ ] **I1** "redux.sync.min takes 1.14–1.15 ms with full/half/quarter/4-lane/1-lane masks — 'mask-width independence'" — claim that warp hardware doesn't speed up for fewer active lanes; assumes equal issue rate regardless. Needs per-mask latency verification (clock64), not wall-time — `[ref: B300_PIPE_CATALOG.md:933]` — `[regime-narrow]` — Wall-time does not distinguish pipeline latency from dispatch bubbles.
- [ ] **I2** "CREDUX.MIN + IMAD.U32 two-SASS: effective PTX-op rate bounded by slower of pipe_alu and pipe_fmaheavy" — assumes strict serialization without evidence — `[ref: B300_PIPE_CATALOG.md:935]` — `[unverified]`
- [ ] **I3** "shfl.sync throughput 5576 GOps/s = 5576 B32/s chip" — conflates "shuffle words" with "FLOPs"; a shuffle is not a FLOP — `[ref: B300_PIPE_CATALOG.md:2123]` — `[unit-confusion]`
- [ ] **I4** "vote.ballot 7320 GOps/s vs vote.all/any/uni 3315 GOps/s — '2 SASS vs 1 SASS explains 2.2×'" — needs ILP verification that SELP doesn't dual-issue with vote — `[ref: B300_PIPE_CATALOG.md:2121,2130]` — `[unverified]`
- [ ] **I5** "redux.sync latency: CREDUX 18 cy, REDUX 44 cy (2.4× slower)" — pure latency or throughput artifact? — `[ref: B300_PIPE_CATALOG.md:2199,2200]` — `[regime-narrow]`
- [ ] **I6** "Block barrier cost 47 cy aligned vs 1455 cy with 1 thread + 200 FMAs (31× penalty)" — penalty assumes FMA latency is the blocker; test-specific to FMA workload — `[ref: B300_PIPE_CATALOG.md:2207]` — `[regime-narrow]`

## Group J — latency/clock64 (§24)

- [ ] **J1** "Memory hierarchy latency table uses clock64 bracketing but clock state not specified" — table claims LDS=33 cy, L1=43 cy, L2=300 cy, DRAM=3000 cy at unspecified clock — `[ref: B300_PIPE_CATALOG.md:2010-2015]` — `[clock-mismatch]` — DRAM RAS/CAS timing scales with clock when computed in cycles.
- [ ] **J2** "Compute latency per SASS: FFMA/FMUL/FADD = 4 cy" — measured as single-chain self-op; §16 noted self-op chains inflate 2× from RF port pressure. May be inflated — `[ref: B300_PIPE_CATALOG.md:2020,1950]` — `[inconsistent]`
- [ ] **J3** "IMAD.HI.U32 (half-rate) = 13 cy" — claimed as half-rate but no corresponding full-rate IMAD.LO in the table — `[ref: B300_PIPE_CATALOG.md:2022]` — `[regime-narrow]`
- [ ] **J4** "MUFU.EX2 = 14 cy, MUFU.RSQ/SQRT/LG2 ftz = 18 cy (vs 40 cy non-ftz)" — non-ftz overhead "+ scaling FMUL = 22 cy" needs SASS verification — `[ref: B300_PIPE_CATALOG.md:2023-2026]` — `[unverified]`
- [ ] **J5** "fence.sc.gpu = 544 cy = 68× CTA cost" vs fence.acquire.cluster = 4 cy — the 4 cy conflicts with §17 reporting 23 ns = 44 cy — `[ref: B300_PIPE_CATALOG.md:2040,2041,1672]` — `[inconsistent]`
- [ ] **J6** "nanosleep min ≈34 ns" — micro-benchmark may not reflect bare hardware minimum under occupancy — `[ref: B300_PIPE_CATALOG.md:2043]` — `[regime-narrow]`

## Group K — atomics (§15, §30.B)

- [ ] **K1** "Atomics on pipe_lsu: stride-8 32-byte → 8-way bank conflicts → 0.125 from 1.00" — linear scaling claim assumes no hardware coalescing fast-path — `[ref: B300_PIPE_CATALOG.md:1012,1020]` — `[unverified]`
- [ ] **K2** "CAS unconditionally half-rate (0.50): both always-succeeds and always-fails take 2.189 ms vs atom.add 1.096 ms" — real CAS retries on failure; artificial test doesn't capture that — `[ref: B300_PIPE_CATALOG.md:1023-1027]` — `[regime-narrow]`
- [ ] **K3** "atom.shared.add.f32 emulated via BSSY.RECONVERGENT + LDS + CAS-loop ~2× slower" — needs SASS verification per NVCC version — `[ref: B300_PIPE_CATALOG.md:1031]` — `[unverified]`
- [ ] **K4** "REDG family rates listed as 0.03 without units — pipe_lsu?" — comparison switches denominator from 1.00/0.50 (shared) to 0.03 (global) without clarifying — `[ref: B300_PIPE_CATALOG.md:1065-1070]` — `[unit-confusion]`
- [ ] **K5** "Global atomics 8-way bank conflict drops to 0.125" — global atomics don't have per-SM banks like shared; bank-conflict concept misapplied — `[ref: B300_PIPE_CATALOG.md:1020,2709-2714]` — `[agent-hearsay]`
- [ ] **K6** "Atomic latency 45 cy pure chain = 'identical to LDS'" — but §24 LDS hit = 33 cy. Inconsistent — `[ref: B300_PIPE_CATALOG.md:2683,2691]` — `[inconsistent]`
- [ ] **K7** "Hot-spot same-addr warp-coalesce 12× slower than unique" — coalescing claim suspiciously narrow ("only works for all 32 lanes same addr") — `[ref: B300_PIPE_CATALOG.md:2685,1909]` — `[regime-narrow]`
- [ ] **K8** "Atomic contention chip-wide: per-warp hotspot (592 addrs, 32-way intra) = 5× slower than per-CTA (148 hotspots)" — ranking unjustified — `[ref: B300_PIPE_CATALOG.md:2709-2714]` — `[regime-narrow]`

## Group L — TMA / mbarrier (§30, §30.G)

- [ ] **L1** "TMA cp.async.bulk issue rate = 48 cy/inst (size-independent floor)" — tested 16 B → 4 KB only; "size-independent" is a strong claim without ≥5 size points — `[ref: B300_PIPE_CATALOG.md:2249,2395]` — `[regime-narrow]`
- [ ] **L2** "TMA single-CTA peak 241 GB/s/SM at 64 KB × DEPTH=3" — but 32 KB × DEPTH=4 also gives 239 GB/s. Multiple configurations hit "241 ± 1" — chip-level limit not size optimum — `[ref: B300_PIPE_CATALOG.md:2335]` — `[regime-narrow]`
- [ ] **L3** "Chip-wide TMA batched 4 KB × 24/barrier × DEPTH=2 = 151 GB/s/SM" — earlier same config reports 139–148; ±12% variance for "peak" — `[ref: B300_PIPE_CATALOG.md:2374,2381-2382]` — `[inconsistent]`
- [ ] **L4** "TMA issue-rate sharp crossover at ~8 KiB" — table 30.4b3 actually shows gradual 48.1 → 48.5 → 49.6 → 52.2 → 65.3 cy/TMA. Not sharp — `[ref: B300_PIPE_CATALOG.md:2395,2398,2401]` — `[inconsistent]`
- [ ] **L5** "Chip-wide TMA honest (148 CTAs × 3×64 KB) = 23.5 TB/s" — 192 KB per-CTA exceeds smem cap; if hitting smem limit, not a fair "honest" peak — `[ref: B300_PIPE_CATALOG.md:2460]` — `[inconsistent]`
- [ ] **L6** "TMA prefetch with lead=8 degrades 64 KB BW −30%" — test conflates single-thread serialization with prefetch ineffectiveness — `[ref: B300_PIPE_CATALOG.md:2480-2487]` — `[regime-narrow]`
- [ ] **L7** "Multi-thread TMA issue (8 warp-leaders) NO speedup, both serial and parallel give 1140 cy/iter" — synchronization point unclear — `[ref: B300_PIPE_CATALOG.md:2492-2496]` — `[unverified]`
- [ ] **L8** "Legacy cp.async.cg 16 B = 200 GB/s/SM, 17.9 TB/s chip — competitive with TMA" — comparison glosses over feature differences (no L2-prefetch/multicast on cg) — `[ref: B300_PIPE_CATALOG.md:2668,2677]` — `[regime-narrow]`
- [ ] **L9** "mbarrier.init = 9.5 cy, init+inval pair = 73 cy → inval ≈63 cy" — assumes no state interaction — `[ref: B300_PIPE_CATALOG.md:2232]` — `[unverified]`
- [ ] **L10** "mbarrier RTT (1 arrive + try_wait, count=1) = 54 cy" — count=1 is tiny case; not representative — `[ref: B300_PIPE_CATALOG.md:2240]` — `[regime-narrow]`

## Group M — extended op catalog (§14)

- [ ] **M1** "FMNMX3 (3-input min/max, fused) = 64 SASS/SM/cy = 128 logical mins" — FMNMX3 may be compiler fusion not native SASS opcode; needs ISA reference — `[ref: B300_PIPE_CATALOG.md:981,1000]` — `[agent-hearsay]`
- [ ] **M2** "F2FP.pack latency = 8 cy vs unpack = 4 cy" — pack with MERGE_C implies extra load; latency may be confounded — `[ref: B300_PIPE_CATALOG.md:1046,2021]` — `[unverified]`
- [ ] **M3** "ATOMS.CAS half-rate vs ATOMS.ADD 1.0 even for always-success" — single-attempt; real CAS workloads loop — `[ref: B300_PIPE_CATALOG.md:987,2058]` — `[regime-narrow]`
- [ ] **M4** "testp.normal.f32 = 3 SASS → ~0.67 logical tests/cy" — assumes no ILP; with 4+ chains throughput should scale — `[ref: B300_PIPE_CATALOG.md:988]` — `[regime-narrow]`
- [ ] **M5** "bfind.u32 on xu = 0.50 = 16/SM/cy" — half-rate claim depends on internal xu pipe structure — `[ref: B300_PIPE_CATALOG.md:989]` — `[unverified]`
- [ ] **M6** "nanosleep 0.25 = 8/SM/cy" — nanosleep is scheduler op; rate may not reflect execution-pipe — `[ref: B300_PIPE_CATALOG.md:991]` — `[regime-narrow]`

## Group N — research-log repetition (§16-§22)

- [ ] **N1** "Batch 1 (§16) FFMA peak 71.8 TFLOPS" — identical to §0 and §25; no new methodology, copy-paste — `[ref: B300_PIPE_CATALOG.md:1111-1125,30]` — `[inconsistent]`
- [ ] **N2** "Global atomics ~0.03 rate" never clarified vs §30.B3 = 45.7 Mops/s — two metrics, same op, no reconciliation — `[ref: B300_PIPE_CATALOG.md:1065-1070,2694-2696]` — `[unit-confusion]`
- [ ] **N3** "Batch 2 (§17) LDG cache hints .cs/.lu/.volatile at 466/522/496 GB/s vs baseline 540" — cache hints should show variance with patterns; nearly identical = WS too small — `[ref: B300_PIPE_CATALOG.md:1729]` — `[regime-narrow]`
- [ ] **N4** "Batch 3 DRAM 7.42 TB/s = 92% of HBM3E peak", DRAM write 3.4 TB/s vs §0 48-49 GB/s/SM — inconsistent contexts — `[ref: B300_PIPE_CATALOG.md:1920,46,91]` — `[inconsistent]`
- [ ] **N5** "INT8 IMMA 143 TOPS = 45× slower than FP8" — comparison is to emulated FP8 not native tcgen05; 45× ratio misleading — `[ref: B300_PIPE_CATALOG.md:1105,28]` — `[regime-narrow]`
- [ ] **N6** "Batch 1 MUFU.RSQ 40 cy" vs §23 clean sweep "RSQ 18 cy ftz" — 2.2× discrepancy from range-reduction overhead in batch 1; should be retracted — `[ref: B300_PIPE_CATALOG.md:1048-1050,1974,2024,2026]` — `[superseded-suspect]`
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
