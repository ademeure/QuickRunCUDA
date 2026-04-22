# A1–D6 Rigor Protocol Docs — Audit vs V40–V51 Findings

Date: 2026-04-22. Wave-3 swarm audit.

The A/B/C/D series (commits ~22b06b3 to early-V4 timeframe, 2026-04-20)
were the first systematic per-pipe rigor protocol on B300 sm_103a.
V40 (`d1d09c5`, ALU ladder), V44/V45 (`372051b`/`eab0a6e`, SMEM regimes),
V49 (`501134a`, dual-issue limits), V50 (`fbe1c18`, warp-spec dual)
re-measured the same surfaces with cleaner anti-DCE, more ILP, and
multi-warp configs. Below: per-doc verdict against the latest evidence.

Originals NOT modified.

---

## Per-doc verdict

| Doc | Verdict | One-line |
|---|---|---|
| `A1_DUAL_ISSUE_RIGOR.md` | **Partially superseded** | Single-warp empty-loop floor lesson HOLDS; "1 inst/cy SMSP issues TOTAL across pipes" was a 1-warp-floor reading — V49 same-warp (8 warps) hits 55% dual-issue, V50 warp-spec hits 74%. |
| `A2_SCHEDULER_RIGOR.md` | **Consistent** | Fairness ≤0.04% and "MUFU warp slows sibling FFMA on same SMSP" both still HIGH. The "SMSP issue port shared across pipes" framing is consistent with V49/V50 finding that scheduler dispatch slot caps 4 inst/cy/SM. |
| `A3_SCOREBOARD_DEPTH.md` | **Consistent (open ceiling)** | LDG scoreboard ≥32 in flight, monotonic decrease through N=32 — never plateaued. True depth still unknown; could be 32, 48, or higher. No V40-V51 retest. |
| `A4_FFMA_PORT_PRESSURE.md` | **Consistent** | 1-2 unique sources = 0.97/SMSP/cy, 3 unique = 0.61. Cleanly reproduced in D6 and corroborated by V10 (`V10_FMA_SOURCE_COUNT`: 75.2 vs 51.3 TFLOPS, ratio 0.683 ≈ 2/3). |
| `A6_PER_PIPE_REFERENCE.md` | **Partially superseded** | Per-inst TIPS table (LOP3=14.16, IADD3=14.13 etc.) is correct AT 1500 MHz / 2 warps/SMSP, but the "FFMA at 0.66/SMSP/cy = ceiling" framing is regime-specific. V40 multi-warp confirms FMA tier (FFMA/FADD/IADD3) all 67% AT 1500 LOCK; V8 `pipe_fma 97.64%` at 2032 boost + 8 warps. A6's "all unified ALU/FMA cluster" coarse model is REFINED by V40 ladder into 4 explicit tiers (FMA 67% / INT-bit 48% / permute 36% / compare 22%). |
| `B1_DUAL_ISSUE_FFMA_IADD3.md` | **Superseded for headline overlap** | "17% overlap" (NC=8, single OP=count) is a snapshot. V49 same-warp FFMA+IADD3 measures **54%** overlap with deeper ILP and OP=3 setup. V50 warp-spec hits 74%. The "FMA pipe 33% bubble at 2 warps/SMSP" diagnosis is REFINED: V8 shows FMA pipe at 97.64% with 8 warps/SMSP — so "33% bubble" was occupancy, not pipe physics. |
| `B2_FFMA_LDG_DUAL.md` | **Consistent** | FFMA+LDG overlap 1% (chain) / 12% (no-chain) NOT contradicted by V40-V51. Hypothesis "queue backpressure when no-chain" still MED. No V49/V50 LDG retest — these targeted ALU pipes only. |
| `C3_LOP3_LUT_DEEP.md` | **Consistent** | LOP3 imm-independence + 0.5/SMSP/cy + 4.5 cy latency + ≥3 RF reads → all match V40 "INT-bit pipe at 48% = 18.7 Glane/s". The C3 chip-peak 14.16 TIPS @ 1500 ⇒ scaled to boost = 19.18 TIPS, which equals V40's 18.7 Glane/s within 3%. |
| `D2_L1_CAPACITY_RIGOR.md` | **Consistent** | "Sharp 128 KB / 1024-line boundary" for strided pointer-chase HOLDS. The cache-corrections doc (`03_caches_CORRECTED.md`) ratifies this and notes V10 random-access shows different effective capacity (~2-4 KB) — both right under their access pattern. No conflict with A2 (scheduler) — different surface. |
| `D3_L2_SECTOR_RIGOR.md` | **Consistent** | 32-byte sector, 7× DRAM read amp on 4B writes, 0× on 32B-aligned. Ratified verbatim in `03_caches_CORRECTED.md` §2.2. |
| `D5_SMEM_BANK_BEHAVIOR.md` | **Partially superseded** | Single-warp regime values (5.74× for stride-32 conflict, 1.44× for stride-4) are CORRECT for the latency-bound regime. V44 + V45 split shows two regimes: latency-bound (V44, ~2× chain-serial, D5's 5.74× compatible) and throughput-bound (V45, 32-way conflict ~1× = effectively free). D5 acknowledges "single-warp test is pessimistic" in §Caveats — but does not have the V44/V45 regime split context. The catalog `02_shmem.md bce8bf8` 8.81× multi-warp number complicates the picture; needs warp-count sweep. |
| `D6_RF_PORT_RIGOR.md` | **Consistent + corroborated** | "2 RF read ports/cycle/SMSP" with reuse-cache as effective 3rd port: HIGH-confidence reads, ratios match (0.96/0.65 = 1.48 vs theoretical 1.50). V10_FMA_SOURCE_COUNT independently measures 2-source 75.2 TFLOPS / 3-source 51.3 TFLOPS (ratio 0.683 ≈ 2/3, exactly the 2-RF-port prediction). Per FFMA agent: ALL near-peak FFMA recipes use ≤2 unique register sources — D6 fully agrees and explains why. |

---

## RETRACTIONS

The following A/B/C/D-series claims are RETRACTED in light of V40-V51:

1. **A1: "single SMSP issues ~1 instruction per cycle TOTAL" (across all pipes)** —
   RETRACTED as worded. True for 1 warp on 1 SMSP (issue-port-bound). For
   ≥2 warps mixing pipes, V49 measures FFMA + IADD3 same-warp at 27969 Glane/s
   (= 27969/26066 = 107% solo FFMA, NOT 100%). Same warp hits 54-55% of pipe
   sum; warp-spec (V50) hits 74%. The single-warp issue port is one limit;
   per-SM 4 inst/cy/SM scheduler dispatch is the multi-warp limit; pipe
   capacity is a third (separate) limit.

2. **A1: "Mixed FFMA + LOP3 (cross-cluster) gives only 6.5% overlap"** —
   RETRACTED. V49 same-warp FFMA + LOP3 at full ILP: **55% overlap**
   (24336 vs sum 44644). The 6.5% was N=16 unrolled but only OP=count=1;
   V49 used OP=3 with cleaner anti-DCE.

3. **A6: "FFMA at 0.66/SMSP/cy is the ceiling because issue port + warp-scheduler
   bubble"** — RETRACTED for "ceiling". The 0.66 number is correct AT 2 warps/SMSP
   1500 MHz, but is NOT the architectural ceiling. V8 `pipe_fma` ncu = 97.64%
   at 8 warps/SMSP 2032 boost; FFMA chip peak 75.2 TFLOPS = 97.65% of theoretical
   76.96. A6 itself flags this in its conclusion ("need 4+ warps/SMSP per
   `04_fp32_peak.md`"); the language should be tightened so "0.66/SMSP/cy" is
   never cited as a chip peak.

4. **A6: "FFMA + SHFL essentially no overlap (14.7%) because SHFL occupies issue
   port"** — PARTIALLY RETRACTED diagnosis. The 14.7% number stands but the
   "SHFL issue port" explanation is unconfirmed. V37/V38 show SHFL is on the
   shuffle pipe at 1/(4cy)/SMSP, plenty of slack; the small overlap likely
   reflects V8-era anti-DCE setup, not SHFL physics.

5. **A6: "Unified ALU/FMA cluster" model with one tier of inst/SMSP/cy** —
   SUPERSEDED by V40 4-tier ladder:
   - FMA tier (FFMA/FADD/IADD3): 67% of 1/cy/SMSP
   - INT-bit tier (LOP3/IMUL): 48%
   - Permute tier (PRMT): 36%
   - Compare tier (ISETP): 22%
   A6's per-inst table (BREV/POPC at 0.125, SHFL at 0.25, etc.) is
   correct for those pipes. The "unified" framing under-resolves this.

6. **B1: "FFMA+IADD3 mixed = 17% overlap, FFMA-bubble explanation"** —
   SUPERSEDED. V49 measures **54%** same-warp (with OP=3 + ILP=8 + multi-warp).
   The "33% FFMA bubble at 2 warps/SMSP" is real but is occupancy, not pipe
   physics. The B1 "open question 'is SMSP issue rate fundamentally 1/cy
   for all ALU/FMA pipes?'" is ANSWERED: NO. Per-SMSP issue is 1/cy *per warp*;
   multi-warp SM aggregate is bounded by the per-SM scheduler dispatch
   (~4 inst/cy/SM = 1 inst/cy/SMSP) and by pipe capacity, distinctly.

7. **B1: "FFMA peak 0.66/SMSP/cy at 1500 MHz is real architectural peak"** —
   RETRACTED with same reasoning as #3 above. It's a 2-warps/SMSP measurement
   floor.

8. **D5: "32-way SMEM bank conflict costs ~5.7×"** — INCOMPLETE. True for
   single-warp latency-bound regime. V45 shows throughput-bound regime
   (many warps queued) hides conflicts to ~1×. The 02_shmem `bce8bf8`
   multi-warp test shows 8.81× — so the picture is regime-dependent
   (1× → 5.7× → 8.8× across configs); the single number "5.7×" should
   not be cited as universal.

9. **A6: "MUFU + FFMA = ~100% overlap because MUFU runs long in
   background"** — STILL CONSISTENT but framing is rough. A2 + V49 evidence
   suggests MUFU occupies SMSP issue port for ~1 cy then runs in MUFU pipe
   for 200+ cy of latency, freeing issue port for FFMA. Mechanism is
   correct, magnitude (100%) MED-confidence.

---

## UNRESOLVED

1. **Why does V49 same-warp dual-issue cap at 55% when V50 warp-spec
   reaches 74%?** Both are measured (not formula). V49 says scheduler
   dispatch slot is shared (4 inst/cy/SM); V50 confirms warp-spec reaches
   higher because separate warps at separate SMSPs avoid the shared slot.
   But why not 100% (each pipe at its own peak)? Likely: warp-spec uses
   4 FFMA + 4 LOP3 warps = 1 of each per SMSP, so each SMSP still has
   to alternate 1 FFMA + 1 LOP3 inst per cycle (= same shared dispatch
   limit per SMSP). Direct ncu `smsp__inst_issued.avg.per_cycle_active`
   under V50 setup would confirm — not yet measured.

2. **A6/B1 IADD3 = 0.50/SMSP/cy vs V40 IADD3 = 0.66/SMSP/cy.** A6 + B1
   measured at 2 warps/SMSP @ 1500 MHz; V40 measured at full persistent
   block, 1184 blocks, also at 1500 MHz lock per `15_integer_bit_ops_CORRECTED`
   §3 (UNRESOLVED #4). The discrepancy is most likely warp count + ILP
   pattern. Need an A6-style sweep with 4+ warps/SMSP to resolve whether
   IADD3 closes to FMA-pipe peak as V40 implies, or stays at 0.50 as B1's
   sequential composition predicts.

3. **A3 scoreboard depth ≥32 — actual depth?** Test capped at N=32 (regs[32]
   array). cy/load was still decreasing at N=32. Real depth unknown; could
   be 32, 48, 64. No V40-V51 retest. Open.

4. **A1 SHFL + FFMA overlap 14.7% mechanism.** A6's "SHFL occupies issue
   port multi-cycle" hypothesis still unverified. V37/V38 measured SHFL
   pipe rate but did NOT redo the dual-issue test. Open.

5. **D5 → V45 reconciliation with `02_shmem.md` `bce8bf8` 8.81× multi-warp
   conflict number.** V45 says "throughput-bound regime hides conflicts
   to ~1×" but bce8bf8 is also throughput-bound and shows 8.81×. The
   `02_shmem_CORRECTED.md` UNRESOLVED #1 calls this out: "Need a single
   test that varies warp count to map the regime boundary." Still open.

6. **A4/D6 "broadcast operand reuse cache" mechanism.** Both observe
   3-distinct-source caps at 65%. D6 attributes to 2 RF read ports +
   reuse cache as effective 3rd port (matched by `.reuse` SASS hint).
   A4 mentions "operand collector deduplication" as alternative. Same
   observable; underlying mechanism not pinned to one explanation.
   ncu `smsp__inst_executed_pipe_fma_collector_*` if available could
   discriminate.

7. **B2 LDG no-chain SLOWER than chain-dep (15.66 vs 8.45 ms LDG-only).**
   "Queue backpressure" hypothesis still MED. No V40-V51 retest.

---

## Cross-references

- `04_fp32_peak_CORRECTED.md` — full FFMA reconciliation, includes V49/V50
  numbers (RETRACTION #9 there matches RETRACTION #6 here).
- `15_integer_bit_ops_CORRECTED.md` — full INT-pipe reconciliation, V40 ladder
  + V49 dual-issue (matches RETRACTIONS #5, #6 here).
- `02_shmem_CORRECTED.md` — V44/V45 split + bank-conflict regime gap (matches
  D5 partial-supersede here).
- `03_caches_CORRECTED.md` — D2/D3 ratified verbatim.
- `V32_V40_FINDINGS.md`, `V41_V48_FINDINGS.md` — primary V-series sources.
- `B300_TRUE_REFERENCE.md` — canonical post-V40 numbers.

---

## Confidence

HIGH for all retractions (each backed by a V40-V51 measurement with
matching commit hash). HIGH for unresolved items being genuinely open
(no V-series test attempted). MED only for the "mechanism" attributions
in A4/D6 (#6) and the SHFL framing (#9).
