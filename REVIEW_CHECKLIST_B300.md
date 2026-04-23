# B300 Catalog — Review Checklist (one-line yes/no per claim)

## TLDR — current audit state (2026-04-23)

**JUSTIFIED status:** **27 ✅ replicated/verified** sections + **5 🟡 partial** + **1 deferred** (multi-GPU). Audit covers essentially the entire foundational early/mid catalog (§0 through §29 + §30.G/L/M/B).

**Top catalog errors to flag for correction (mark these first):**

| # | Catalog | Real | Source |
|---|---------|------|--------|
| 1 | DFMA latency = **92 cy** (L103) | **63.9 cy** | E2 — §24, §2.13 |
| 2 | __syncthreads = **12+2W cy** | **22+2W** (54 at BS=512) | E5 — §24 |
| 3 | FP64 rate = **0.05 warp-inst/SM/cy** (L446) | **0.06 (99.95% peak)**; chip ≈ 1.06 TF (catalog 0.95 TF is 12% lower); "475 GFLOPS FMA" wording was 475 G FMA-ops/s (= 950 GFLOPS, NOT misstatement) | §2.13 refined |
| 4 | §4 MUFU rate = **~16 SASS/SM/cy** | **~1.0 SASS/SM/cy** (off by 16-32×) | §17 ncu pipe_xu peak |
| 5 | Rule 9 atomic hotspot = **5× slower** | **34×** at warp-level (none at CTA-level) | §22r |
| 6 | Rule 11 FP64 = **300×** slower than FP16 tensor | **~2300×** (1.06 TF / 2465 TF) | §2.13 |
| 7 | mbarrier.arrive = **8.1 cy** (cheat-sheet) | **27 cy** for default `.shared.b64` | §0 cheatsheet audit |
| 8 | atom.global.cas → **STRONG.GPU** | actual SASS is **STRONG.SYS** | §15 |
| 9 | "ld.shared bank-conflict-sensitive" | ✅ TRUE for ALL load widths (32-bit too); my prior "FALSE for 32-bit" claim was a methodology error (tested broadcast not conflict) | §22j ADDENDUM |
| 10 | smem **"200 KB per CTA without opt-in"** (L84) | ❌ Real default is **48 KB** (`sharedMemPerBlock`); opt-in MAX is 227 KB (`sharedMemPerBlockOptin`); per-SM hardware max is 228 KB (`sharedMemPerMultiprocessor`). Catalog conflated three different caps. | G7 — `justifications/G7_smem_capacities.md` |

**Top NEW architectural facts to ADD (audit-discovered, missing from catalog):**

| # | Fact | Source |
|---|------|--------|
| A | `cvt.rni.sat.u8.f32` (F2IP.U8 alu 2.00) is **4× faster** than `cvt.rni.sat.s8.f32` (F2I.S8 xu 0.5) | §2.6 |
| B | `atomicAdd(addr, 1u)` no-return → **ATOMS.POPC.INC.32** (2.5× speedup at warp-broadcast); only fires for constant=1 | §22 atomic_smem |
| C | Global REDG (no-return) is **25× faster than ATOMG** (with return) for ADD/MIN/MAX/etc. | §15 + 22 atomic_ops |
| D | EX2 is uniquely **2× faster than other MUFU** (4 cy vs 8 cy at saturation) | §17 |
| E | bf16x2 EX2 (`MUFU.EX2.BF16x2`) gives same throughput at **HALF dispatch pressure** | §17 ADDENDUM |
| F | FMNMX3 fusion (Blackwell 3-input fused FP min/max) — 2× chained min.f32 → 1 SASS, 128 logical mins/SM/cy | §14 |
| G | u64.ADD demonstrates clean **alu+fmaheavy co-issue** (IADD3 + IMAD.X both saturate together = 64 u64-adds/SM/cy) | §2.4 |
| H | CCTL.IVALL is **essentially FREE (~3 cy)** on idle pipeline; observed cost is drain-wait for in-flight loads (acquire fence semantics) | §22l ADDENDUMs 3-16 |
| I | release.gpu drain is **SM-WIDE** (drains all co-resident CTAs' loads + stores), can compound to 11,000+ cy at high occupancy | §22l ADDENDUMs |
| J | cp.async (LDGSTS) **bypasses acquire fence drain** unless commit_group is issued first | §22l ADDENDUM 12 |
| K | **`.L2::256B` cache hint gives 40% DRAM BW boost** (5.07→7.06 TB/s = 92% HBM SoL) for sparse-but-spatially-local access patterns (4B reads at 256B stride between threads). Compiler emits LDG.E.LTC256B. | `16_L2_256B_modifier.md` |
| L | **.ca beats .cg by 1.88× (NOT 1.25×)** at L1-fitting workloads — catalog L1571 understates the gap by 3.5×. .ca → 13.13 TB/s L1TEX (catalog ✓ exact); .cg → only 6.97 TB/s (catalog claimed 10.5). | `16_ca_vs_cg_hot.md` |
| M | **`ld.const.u32` (LDC.32) dispatches via the ADU pipe, NOT LSU** — ADU peak 0.5 inst/SM/cy. BS=512 required to saturate (BS=256 only 90%). 17.99 TB/s effective via 31.7× broadcast = catalog ✅ within 1.1%. Catalog §1 / §2.12 PTX→pipe table is missing the LDC row entirely. | `02_12b_const_mem_broadcast.md` |

**Format below:** group by catalog section, single-line yes/no per claim. RESOLVED items have `[x]`; OPEN items have `[ ]` for your review.

---

> **What this is:** My OWN skeptical review of `B300_PIPE_CATALOG.md` claims, sectioned for fast yes/no scanning. The user's `reviewed_errors_b300.md` notes were on a DIFFERENT (lower-quality) doc — I use them as INSPIRATION for the kinds of issues to look for here (clock-state assumptions, formula-as-measurement, DCE, denominator mixing, agent-hallucinations of names/mechanisms, etc.), but the claims listed below are the catalog's own.
>
> **Instructions for user:** Each row is a single claim I am NOT 100% certain of. Mark with `[x]` for "yes, this is correct", `[ ]` for "no, wrong", and add a `// short note` after the line if you want to leave a comment. The structure is grouped by catalog section so you can stop after any group without losing context.
>
> **Format:**
> - `[ ]` claim text — context (numeric value, regime if relevant) — `[ref: B300_PIPE_CATALOG.md:Lxxx]` — `[reason for uncertainty]`
> - One blank line between groups.
>
> **Color key for `[reason for uncertainty]`:**
> - `unverified`: I haven't been able to replicate it yet
> - `clock-mismatch`: number was at a different clock than headline implies
> - `formula`: claim looks like it's computed from spec, not measured
> - `DCE-suspect`: pattern looks vulnerable to compiler eliminating the loop
> - `agent-hearsay`: number came from an LLM swarm without independent re-verify, or names a mechanism (SASS opcode, memory path) that I haven't seen documentation for
> - `inconsistent`: same topic reported with different number elsewhere in the catalog
> - `regime-narrow`: holds only under one combination of (warps, BS, working set) and may not generalize
> - `unit-confusion`: GB vs GiB, wire-rate vs effective, packed-element-count vs SASS-inst-count
> - `superseded-suspect`: I think a later test contradicted this but haven't traced
>
> **Note on cross-references:** Where an entry says `[ref: reviewed_errors L###]`, that's because the user raised an ANALOGOUS skepticism point on the canonical doc — used here as the inspiration for flagging the catalog's similar claim. The catalog claim itself is the load-bearing thing being reviewed.

---

## RESOLVED SUMMARY (17 items, 2026-04-23)

| Entry | Catalog claim | Resolution |
|---|---|---|
| **§22e** (NEW) | ".reuse cache 94% of FFMA2" | ✅ AUDIT-VERIFIED via direct SASS grep: scalar FFMA 99.9%, FFMA2 82.8-99.2% across 5 configs. Catalog 94% is in-range for FFMA2; conservative for scalar FFMA. |
| **§22h** (NEW) | "FFMA fully hidden by 522 cy memory load; ~16 FFMA free" | ✅ qualitative CONFIRMED but quantitative DIFFERS — cold DRAM is **882 cy** (not 522), free budget is **~225 FFMAs** (not ~16). Catalog's 522 was partial-cold; updated to "cold 882 / warm 335". |
| **§30B SASS-mapping** (CORRECTION) | "atom.global.add → REDG NOT ATOMG" | ⚠ OVERSTATED: direct SASS grep across 20K kernels shows ALL THREE opcodes emitted (REDG / ATOMG.E / ATOM.E) depending on context. Throughput numbers still valid; only SASS-name attribution was wrong. See `30B_atomics_FOLLOWUP.md`. |
| **§11** (NEW) | redux.sync min/max=1.92, add/and/or/xor=0.50 ADU | ✅ CONFIRMED via ncu pipe metrics: min=1.89 (alu+fmaheavy), add=0.50 (adu) — within 2% of catalog. 4× asymmetry exact. |
| **§12** (NEW) | pipe_alu cap = 2.00 warp-inst/SM/cy | ✅ CONFIRMED at 1.94 (97%) via pure LOP3 at NC=16+MIN_BLOCKS=4. Methodology lesson: need both high ILP AND high occupancy for true SoL. |
| **§17 ADDENDUM** (CORRECTION) | "MUFU 0.5/SMSP/cy uniform" + "16 SASS/SM/cy in §4" | ⚠ Architectural truth via ncu at 32 warps/SM oversubscribed: pipe_xu peak=1.0/SM/cy. EX2=4.0 cy/op (100% pipe), compound MUFU=8.0 cy/op (50%), RCP=8.5 cy/op (47%, scaffolding-bound). bf16x2 EX2 hits 50% pipe with 2 ops/inst = SAME throughput at HALF dispatch pressure. Catalog §4 "16 SASS/SM/cy" is OFF by 16-32×. |
| **§13** (NEW) | predication zero-effect on pipe rate | ✅ CONFIRMED via ncu pipe_fma identical (within 1%) across 32/16/1 active-lane masks (2.91/2.94/2.94). |
| **§7** (NEW) | pipe_adu cap ~0.4-0.5 | ✅ CONFIRMED at 0.50 exactly (REDUX.SUM saturates 100%, bar.sync 72%). |
| **§6** (NEW) | pipe_uniform "~1.0 warp-inst/SM/cy" | ✅ STRONGLY CONFIRMED PEAK = 2.0/SM/cy via UIADD3 chain (1.94 measured = 97%) AND ULOP3 (1.86). Wall-clock evidence supports >1.0. LDSM hits only 0.70 = 35% (regime-narrow). Catalog "1.0" was measurement artifact. |
| **§2.4** (NEW) | u64.ADD = 64/SM/cy via IADD3+IMAD.X 2-pipe co-issue | ✅ CONFIRMED — pipe_alu=1.95 + pipe_fmaheavy=1.94 simultaneously saturate. 1 IADD3 (alu) + 1 IMAD.X (fmaH) per u64.ADD = 1 op/cy/warp × 32 lanes × 2 warp-inst-pipes = 64 u64-adds/SM/cy ✓. Demonstrates clean cross-pipe co-issue. |
| **§2.4** (NEW) | u64.AND/OR/XOR = 32/SM/cy via 2× LOP3 | ✅ CONFIRMED — pipe_alu=1.98 with 2 LOP3 per u64.AND = 32 u64-logic/SM/cy. |
| **§2.5** (NEW) | All 6 narrow-format UNPACK formats = 2.00 = 128 elements/SM/cy | ✅ CONFIRMED — F2FP.{E4M3,E5M2,E2M1,E2M3,E3M2,UE8M0}.UNPACK_B all hit 99.98% pipe_alu peak. FP4 not faster than FP8 confirmed. |
| **§2.1** (NEW) | FFMA = 4.00 (256 FLOPS/SM/cy via heavy+lite alternation) | ✅ CONFIRMED via 00a_ffma_peak: 71.82 TFLOPS = 99.5% pipe_fma. |
| **§2.2** (NEW) | FFMA2 / HFMA2 packed = 2.00 (= 128 packed FMAs/SM/cy) | ✅ CONFIRMED — pipe_fmaheavy=1.96 + pipe_fmalite=1.97 (both saturate together for 1 dispatch). |
| **§2.3** (NEW) | IMAD = 2.00 fmaheavy = 64 IMAD/SM/cy | ✅ CONFIRMED EXACTLY at 99.94% pipe_fmaheavy. |
| **§2.7-§2.9** (NEW) | All "rate 2.00 alu" claims (LOP3/PRMT/SHF/ISETP/FMNMX/HMNMX2/VIMNMX3/copysign) | ✅ CONFIRMED PLAUSIBLE — pipe_alu cap=2.00 verified at 97% via §12; all alu-resident "rate 2.00" claims fit within budget. bfind/FLO confirmed at 0.5 xu. |
| **§2.13** (NEW) | DFMA = 0.05 warp-inst/SM/cy = 475 GFLOPS chip | ⚠ REFINED: real rate is **0.06** (99.95% peak), not 0.05. **Real chip TFLOPS is 1.06** (= 88% of 1.20 theoretical at 2032 MHz boost). Catalog "475 GFLOPS FMA" wording = "475 G FMA-ops/s = 950 GFLOPS" per parenthetical (12% off measured, plausible from clock + rate). |
| **§25 FP64 propagation** | "475 GFLOPS" in summary table | ⚠ same as §2.13 — wording-confusion not 2.2× error |
| **§2.6** (NEW) | Other CVTs (HADD2.F32 fmaH 2.00; F2I=0.5 xu; F2IP.U8=2.00 alu surprise; I2FP=2.00 alu) | ✅ all 4 directly testable rows CONFIRMED at 98-99% (HADD2.F32=1.97, F2I=0.50, F2IP.U8=1.97, I2FP=1.98). I2F.S64 "super slow" plausible but too low for ncu sampling. |
| **§2.6** (NEW) | F2IP.U8 alu fast path is 4× faster than F2I.S8 xu | ✅ ARCHITECTURALLY CONFIRMED — pick `cvt.rni.sat.u8` over `cvt.rni.sat.s8` for 4× CVT throughput when format permits. |
| **§0 latency table** (NEW) | Headline-card table at top of catalog | ⚠ 11 entries CONFIRMED via §24 + per-section audits; 2 KNOWN WRONG (already in REVIEW_CHECKLIST as E2 + E5: DFMA 92→63.9, syncthreads 12+2W→22+2W); 3 plausible-but-not-re-tested. |
| **§2.12** (NEW) | "ld.shared bank-conflict-sensitive" | ⚠ NEEDS SCOPING: TRUE for v2/v4 wide LDS (1.58-1.68× penalty), FALSE for 32-bit LDS on B300 (random=stride for any pattern hits ~6.88 cy). |
| **§2.12** (NEW) | atom.* "not measured" placeholder | ✅ NOW MEASURED via §15 + 22_atomic_ops_DEEP.md + 22_atomic_smem_DEEP.md — comprehensive matrix of 10 ops × 2 return modes × 3 contention. |
| **§0 cheat-sheet Rule 9** (NEW) | "Per-warp atomic hotspot 5× SLOWER than single-address chip-wide" | ⚠ ALREADY in REVIEW (§22r): real factor at warp-level N=2 is **34×** (not 5×); CTA-level shows NO slowdown. Catalog rule misses warp-vs-CTA distinction. |
| **§0 cheat-sheet Rule 11** (NEW) | "FP64 is 300× slower than FP16 tensor" | ❌ Real ratio is **~2300×** (FP64=1.06 TF vs FP16 mma.sync ~2465 TF). Catalog 300× understates by 8×. |
| **§0 mbarrier.arrive = 8.1 cy** (NEW) | Catalog headline | ❌ Real **~27 cy** for default `mbarrier.arrive.shared.b64`. Catalog 8.1 likely measured `mbarrier.arrive.relaxed.cta` (lighter). Catalog should specify modifier. |
| **§0 __syncthreads 45/89 cy** (NEW) | At BS=512/1024 | ❌ Real 54/86 cy via formula 22+2W (E5 RESOLVED). Catalog values from wrong 12+2W formula. |
| **§14** (NEW) | FMNMX3 fusion (Blackwell 3-input FP min/max, 2× min.f32 → 1 SASS) | ✅ CONFIRMED via SASS (128 FMNMX3 emitted) + ncu pipe_alu=1.97 (98.54%). Real architectural feature. 128 logical FP min ops/SM/cy (same multiplier trick as IADD3 for ints). |
| **§8/§9 mapping** (NEW) | SASS opcode → pipe assignment tables | ✅ FUNDAMENTALLY CORRECT — 40+ rows directly verified across our pipe-rate audits. Catalog mapping reliable. Open: §8 "Peak SASS/SM/cy" column for MUFU=16 cross-conflicts §17 audit (which gives ~1.0). |
| **§30.M CCTL** (NEW) | "CCTL.IVALL cost UNKNOWN" (catalog open question) | ✅ RESOLVED via 22l_cctl_ivall_DEEP: CCTL.IVALL = 2-3 cy on truly idle pipeline (essentially FREE). Drain-wait for in-flight loads dominates observed fence cost (L1-hit +22 cy, L2-hit +83, DRAM +900). |
| **§30.L ALU** (NEW) | FFMA/FADD/LOP3=4 cy lat / 2.68 cy tp at 8 ILP; DFMA=64 cy NOT pipelined; HMMA=20/8 cy | ✅ ALL CONFIRMED via §24 + §2.13 + §22 mma.sync audits. |
| **§25 FP64** (NEW) | "FP64 DFMA scalar 475 GFLOPS" in summary table | ❌ propagates L446's wrong number; real **~1060 GFLOPS** (per §2.13 audit). |
| **§25 HMMA** (NEW) | "FP16 HMMA tensor 838k = 838 TF" in summary | ⚠ §22 mma.sync audit measured **571 TF** for FP16 (warp-scale). 838 might conflate mma.sync vs tcgen05 paths. |
| **§25 division ladder** (NEW) | div.rn 330× slower than FFMA, etc. | 🟡 plausible per pipe-mapping but not directly re-tested. |
| **§26 warp coop** (NEW) | vote.ballot 2× faster than vote.all/any/uni; redux.sync.min 7× shfl-tree | ✅ ALL CONFIRMED (vote per SASS expansion 1 vs 2 SASS; redux per §11) |

## RESOLVED SUMMARY (prior 15 items, 2026-04-23)

These items have been replicated, verified, or had their resolution settled. Skim this list to see what's verified vs what's still open.

| Entry | Catalog claim | Resolution |
|---|---|---|
| **A1+A2** | FFMA peak 71.8 TF | ✅ matches exactly (71.82 TF). Clock=1942 MHz (rig DVFS settling, NOT 1920 NOR 2032). |
| **C8** | "FFMA → both fma sub-pipes simultaneously" | ❌ FALSIFIED. FFMA dispatches to ONE sub-pipe per cycle (scheduler alternates). pipe_fmalite=93%, pipe_fmaheavy=4.5% in dual mode. |
| **C9** | FFMA2 + ALU dual-issue | ✅ FFMA2+LOP3 1:1 saturates 3 pipes → **314 useful ops/SM/cy** vs scalar+LOP3's 187. **Sweet spot for tuned kernels.** |
| **C1** | "Dispatch ceiling = 4.00 hard cap" | ✅ confirmed (3.88-3.95 measured); V52 alu+fma=145% (cross-pipe sum). |
| **D1** | mma.sync FP16 m16n8k16 = 577 TF | ✅ matches (571 TF, 99.5% pipe_tensor). |
| **D2** | mma.sync TF32 m16n8k8 = 288 TF | ✅ matches (285.7 TF). |
| **D3** | mma.sync FP8 emulated = 276 TF | ⚠ catalog **12% LOW**. Real value **309 TF**. Recommend bumping catalog L27. |
| **D4** | mma.sync INT8 IMMA = 142 TOPS | ✅ exact match (142.4). |
| **E2** | DFMA latency = 92 cy | ⚠ catalog L103 WRONG. Real **63.7 cy** (L460 was right). |
| **E4** | fence.sc.gpu = 274 cy | ✅ confirmed (267-281 cy single-warp + ~280 cy first-fence-after-write). |
| **E5** | __syncthreads BS=512 = 45 cy / formula 12+2W | ⚠ both wrong. Real formula `22+2W` (BS=512=54 cy). |
| **E6** | per-warp atomic 5× slower than 1-hotspot | ❌ catalog REVERSED. Per-warp clean is 1.09× FASTER; per-CTA 12.4× FASTER. |
| **G1** | TMA cp.async.bulk = 48 cy size-independent floor | ⚠ catalog conflates 2 measurements: 48 cy is amortized rate, pure single-issue is 65 cy. |
| **G2** | TMA chip-wide 29.2 TB/s | ⚠ requires L2 hits, NOT DRAM. Chip-scale measurement caps at 6.4 TB/s HBM-bound. |
| **G2b** | TMA vs LDG max-tuned head-to-head | ✅ NEW: L2-hit TMA wins 12% (20.49 vs 18.25); DRAM-cold TIED at HBM SoL (96.5%/95.4%). Catalog L2 wire 13.3 TB/s under-counts by 37-54%. |
| **CRIT1** | fence costs span 29-8869 cy | ✅ single-GPU ladder is cta=8 / gl=267 (+~280 first-after-write FIXED, NOT linear) / sys=1727. V54's 2806 was 2-GPU NVLink rig. |
| **CRIT5** | "FREE" claims | ⚠ DSMEM "free" FALSIFIED (real 9× slower); atomic scope FREE for L2-hit confirmed. |

**Resolution summary:** 8 ✅ catalog verified, 9 ⚠ catalog corrected/refined, 2 ❌ falsified.

## CRITICAL RESOLVED — DSMEM falsification

Catalog "DSMEM ~identical to local SMEM, essentially free" is **WRONG**. Real findings (justifications/13_dsmem.md + 13_dsmem_exhaustive.md):
- Read latency single-chain: **204-223 cy** (vs 23 cy local SMEM = 9× slower)
- Read latency ILP=32: **9 cy/load** (close to LDS — DSMEM IS hidable with enough ILP)
- Write throughput sustained, fenced: 87-117 GB/s/cluster (depends on stride)
- Write burst (no-fence): 660 GB/s/cluster (V21's 560 reproduces here)
- SASS: `ld.shared::cluster.u32` → `LD.E` (global LSU path), NOT LDS
- L2 traversal: 0.03-0.05% (both reads AND writes bypass L2)
- **Topology**: cluster=8 → SMs (0,1,16,17,32,33,48,49) — 1 TPC per GPC, GPC stride=16
- **Per-GPC silicon variation**: GPC2 189 cy vs GPC1 229 cy (20% spread)
- **Topology**: B300 SXM6 AC has **9 GPCs × 16 SMs + 1 partial 4-SM GPC = 148** (catalog "8 GPCs" elsewhere WRONG)

---

## STILL-OPEN HIGH-PRIORITY (top items to review)

These are the [ ] items the user is most likely to have a strong opinion on. Open `B300_PIPE_CATALOG.md` line by line for the catalog claim.

**Recently resolved (2026-04-23) — full details inline with their per-group entry below:**
- ✅ A3, A4 (HFMA2 SASS confirmed for both scalar+packed)
- ✅ B9 (LDC.32 broadcast = 17.99 TB/s eff, dispatches via ADU)
- ✅ B12, CRIT8 (ENL2 ≠ "bypass L1"; it's L2-sector-size hint)
- ✅ CRIT4 (WRONG sections at L8558/L8586 ARE self-flagged in catalog)
- ✅ CRIT6 (FMNMX3 confirmed real Blackwell opcode via SASS in §14)
- ✅ D6 (NVFP4 K=64 9.26 PFLOPS @ 1942 MHz; K=96 ULTRA bit 31 doesn't add MACs)
- ✅ D1-D4 (mma.sync FP16/TF32/FP8/INT8)
- ✅ E2, E3 (DFMA 63.7 cy not 92; MUFU.sin = 24.45 cy)
- ✅ G7 (smem 228 KB per-SM ✓; "200 KB per CTA" FALSIFIED)

**The above-listed top-of-file priority items were ALL also resolved per-group below 2026-04-23.** Removed the duplicate `[ ]` shadow entries for A5/D5/CRIT2/CRIT9/F1-F6 — see their per-group entries for the verdict.

**Genuinely measurement-blocked (deferred items, ~5):**
- 🟡 **B6** DRAM write 7.09 TB/s — user flag: SM→L2 write path may be 32B/clk-limited at lower clocks. Existing measurement is at 1942 MHz; verifying user's clock-dependence theory needs a clock-locked sweep deferred with the F-group power campaign.
- 🟡 **B7 + B8** TMEM read/write bandwidth (55-131 TB/s) — needs tcgen05.ld/.st focused throughput rig (alloc + mbarrier + per-quad accumulator). Catalog numbers very high vs first-principles bound; almost certainly include broadcast-amplification (similar to LDC.32 effect from B9).
- 🟡 **D5** tcgen05.mma per-format throughput (128 cy at M=128 N=256) — needs alloc/mbarrier/cp setup. D6 already ✅ verified (9.26 PF at 1942 MHz, line 155).
- 🟡 **D7 + E7** Multi-GPU items — single-GPU rig this session; preserved as plausible per `project_b300_multigpu` memory.

These five remaining items all need either a separate measurement campaign (B6/B7/B8/D5) or a multi-GPU rig (D7/E7) — outside this audit's scope.

---

## Group A — Headline FP32 / FFMA

- [x] **A1** "FP32 FFMA peak = 71.8 TFLOPS = 98.8% of theoretical 72.7 TFLOPS at 1.92 GHz" — ✅ CONFIRMED via §00a (justifications/00a_ffma_peak.md): measured **71.82 TFLOPS at 1942 MHz** (DVFS settled; ncu clamps to ~1.92 GHz). The 71.82 / 73.6 (theoretical at 1942 MHz) = 97.5% pipe_fma. The catalog's 72.7 TFLOPS denominator uses 1920 MHz — within 0.3% of the actual settled 1942 MHz. **Catalog correct in spirit; the 76.96 TF boost theoretical is unreachable under ncu (which throttles to base clock). Without ncu, sustained boost gives ~74-76 TF.** Two regimes; both legitimate. — `[ref: B300_PIPE_CATALOG.md:30]`
- [x] **A2** "256 FLOPS/clk/SM" formula — ✅ MECHANISM CORRECTED 2026-04-23 (justifications/01_pipe_topology.md): the 256 FLOPS/clk/SM peak holds (4 SMSPs × 32 lanes × 2 FLOPS = 256), but the **mechanism** in catalog L218 ("FFMA uniquely uses BOTH fma sub-pipes simultaneously") is FALSIFIED — FFMA dispatches to ONE sub-pipe per cycle, scheduler-chosen, not both. See C8 below. The 256 FLOPS peak comes from 4 SMSPs each issuing 1 FFMA/cycle (one sub-pipe used per SMSP per cycle, but 4 SMSPs × 1 = 4 FFMAs/SM/cy → 256 FLOPS). Number unchanged; story unchanged. — `[ref: B300_PIPE_CATALOG.md:30,210]`
- [x] **A3** "FP32 via FFMA2 packed = 72.3 TFLOPS = 99.4%" — ✅ CONFIRMED via §2.2 audit: pipe_fma=1.97 (98.5%) at packed FFMA2; SASS shows 128 FFMA2 / 0 scalar FFMA. — `[ref: B300_PIPE_CATALOG.md:31]`
- [x] **A4** "FP16 via HFMA2 = 72.3 TFLOPS-FP16, no extra throughput on scalar path; compiler packs anyway" — ✅ CONFIRMED via SASS dump (justifications/02_1_2_3_fp32_int.md ADDENDUM): scalar `fma.rn.f16` emits HFMA2 (no scalar HFMA1 exists on sm_103a); both halves compute the same value, so per-useful-FLOP throughput is half of true packed. To get 70.4 TF FP16 you MUST use `fma.rn.f16x2`. — `[ref: B300_PIPE_CATALOG.md:32,33]`
- [x] **A5** "FP64 DFMA = 0.95 TFLOPS = 1/76× of FFMA" — ⚠ REFINED via §2.13 (justifications/02_13_fp64.md): measured **1.06 TFLOPS at 1942 MHz**, pipe_fp64 = 99.95% peak. So pipe is fully saturated. The 0.95 catalog number was at slower clock; refining to **1.06 TF at this clock state** = 1:67 ratio (vs FFMA 71.82). Theoretical at boost would be 1.20 TF (1:64). The "1/76× of FFMA" line should be updated to "1/67×". The "0.95 not 1.20 implies under-saturation" interpretation is WRONG — pipe IS saturated, gap is purely clock-state. — `[ref: B300_PIPE_CATALOG.md:35]`
- [x] **A6** "FFMA dispatch ceiling = 4.00 sm_inst/SM/cy is hard" — ✅ CONFIRMED via §1 (justifications/01_pipe_topology.md): pure FFMA hits **3.88 = 97% of cap**; dual-issue mode (FFMA + LOP3) hits 3.95. The "alu+fma sum = 147%" is a CROSS-PIPE sum (different pipes saturate simultaneously); per-pipe is still ≤ cap, and total dispatch (warp-instructions/SM/cy) IS hard-capped at 4.00. Catalog framing is correct in strict-total sense. — `[ref: B300_PIPE_CATALOG.md:480]`

## Group B — Memory hierarchy

- [x] **B1** "smem 35.6 TB/s = 98% theoretical" — ✅ CONFIRMED via §00b (justifications/00b_mem_hierarchy.md): measured **35.88 TB/s** at 1942 MHz settled clock. Catalog 35.6 ≈ 35.88 within 1%. The 1.92 GHz denominator and the clock state at measurement match — no clock-mismatch issue. **At 2032 MHz boost the theoretical would be 38.4 TB/s; that regime is not measured here (ncu throttles to base).** — `[ref: B300_PIPE_CATALOG.md:41]`
- [x] **B2** "L1 hit 36.1 TB/s vs L1 generic 28.7 TB/s" — ⚠ EXPLAINED via §00b (justifications/00b_mem_hierarchy.md L246-249) + §16 ca/cg audit: the THREE different "L1" numbers in the catalog (36.1 / 28.7 / 13.1) reflect three DIFFERENT counters and regimes — (a) **36.1 TB/s** ≈ `l1tex__t_bytes` chip total (includes smem unified L1 path), (b) **28.7 TB/s** = wall-clock generic LDG with WS partially in L2, (c) **13.13 TB/s** ✅ measured = pure `.ca`-only LDG at L1-fitting WS (16-196 KB). Catalog should explicitly tag each row with the counter used. — `[ref: B300_PIPE_CATALOG.md:42,43]`
- [x] **B3** "L2 plateau 22-26 TB/s, was wrongly 10.2" — ⚠ REFINED via §00b: measured **20.3 TB/s stable** across 4-128 MB WS at 2 CTAs/SM (lts__t_bytes 20.21 TB/s direct ncu). Catalog's correction direction is right (20.3 ≫ 10.2), but the upper bound 26 is optimistic on this rig — actual ceiling is ~20-22 TB/s under standard launch config. Recommend catalog narrow to "20-22 TB/s plateau" or note the regime where 26 is reachable. The historical 10.2 was indeed an under-occupied-launch artifact. — `[ref: B300_PIPE_CATALOG.md:44]`
- [x] **B4** "L2 knee 23→22→20→11 TB/s at 1 GB" — ✅ MECHANISM CONFIRMED via §00b L190-200: at 1 GB WS the measurement shows **12.12 TB/s** wall-clock with `lts__t_bytes` measuring **L2 wire BW = ~12 TB/s** (not 7.18 HBM). The "11 TB/s at 1 GB" is exactly the L2-partial-reuse regime — each cache line is revisited as the modular access pattern wraps within L2's 126 MiB, inflating effective BW above raw HBM. Catalog correct that 11 ≠ HBM peak; the "knee" is the L2-reuse-amortization boundary, not a pure DRAM measurement. — `[ref: B300_PIPE_CATALOG.md:45]`
- [x] **B5** "DRAM (HBM3E) read = 7.18 TB/s ncu-verified" — ✅ CONFIRMED via §00b: measured 7.17-7.25 TB/s consistent across WS=1GB to 8GB. **Denominator clarification:** 7680 GB/s is the spec ceiling assuming 8 stacks × 12-Hi × full-rate (per `project_b300_corrections_swarm` finding); this-device measured 7672 GB/s SoL. Either denominator gives ~94% MFU which is the right ballpark. The 7.06 TB/s recipe with `.L2::256B` (top-12 fact K) gives 92% which closes the gap. — `[ref: B300_PIPE_CATALOG.md:46]`
- [ ] **B6** "DRAM write = 7.09 TB/s" — User flag: "SM→L2 *write* path is limited to 32B/clk, you cannot get peak HBM write at lower clocks; 1920 vs 2032 might affect this" — `[ref: B300_PIPE_CATALOG.md:46, reviewed_errors L1132]` — `[clock-mismatch]`
- [ ] **B7** "TMEM read 55.92 TB/s, drops to 31 with 4R/iter" — surprisingly high; my back-of-env says theoretical TMEM read is bounded by tcgen05.ld throughput per warp × SMs which gives much less. Methodology + SASS dump needed — `[ref: B300_PIPE_CATALOG.md:50]` — `[unverified]`
- [ ] **B8** "TMEM write 97.93 → 131 TB/s" — even more suspect; needs first-principles bound check — `[ref: B300_PIPE_CATALOG.md:50]` — `[unverified]`
- [x] **B9** "Constant memory broadcast LDC.32 = 17.8 TB/s eff (~0.55 TB/s actual cache traffic)" — broadcast amplification × 32 lanes; verify denominator — `[ref: B300_PIPE_CATALOG.md:47]` — `[unit-confusion]` // ✅ REPLICATED: measured 17.99 TB/s eff / 0.562 TB/s actual at BS=512 (pipe_adu=99.5%); 31.7× broadcast amplification confirmed via MODE=1 non-broadcast 32× slowdown. **NEW**: LDC dispatches on ADU pipe (NOT LSU/uniform). See `justifications/02_12b_const_mem_broadcast.md`.
- [x] **B10** "Local (register spill) 1.3 TB/s = 52× slower than smem" — ✅ COVERED via justifications/22q_register_spill_DEEP.md (NINJA-DEPTH): spill cliff at **32 live vars** (9× perf drop measured); register spill bandwidth ≈ smem÷52 confirms catalog ratio. Per-spill cost dominated by L1 store traffic (LDL/STL emitted, not LDS). — `[ref: B300_PIPE_CATALOG.md:49]`
- [x] **B11** "256-byte stride v8 .256B L2 cache modifier" — ✅ REPRODUCED 2026-04-23 (justifications/16_L2_256B_modifier.md): user L925 recipe achieves **7.06 TB/s = 92% HBM SoL** (40% boost over baseline 5.07 TB/s) with `ld.global.L2::256B.u32` + stride-256B + 4B reads; SASS confirms LDG.E.LTC256B emit. Optimal stride = 256B (matches sector size); ≥296 blocks for full saturation. Promoted to top-13 architectural facts (K). — `[ref: reviewed_errors L925]`
- [x] **B12** "LDG.E.ENL2.256 means bypassing L1" — ❌ FALSIFIED via SASS+catalog cross-check (justifications/16_ldg_cache_hints.md ADDENDUM): ENL2.256 is the L2-sector-size encode (same purpose as LTC256B on the load side, emitted by 256-bit v8 stores), NOT a cache-bypass directive. Catalog's own L4925-4932 table shows WIDTH=8 v8 stores (which use ENL2.256) populate L1 normally at 2.34 TB/s alongside DRAM 2.19 TB/s — L1 is NOT bypassed. Cache-bypass is controlled by `.cg`/STRONG.GPU scope tag, INDEPENDENT of the ENL2/LTC256B hint. — `[ref: reviewed_errors L1063]` — User was right to be skeptical.
- [x] **B13** "ptxas compiles differently based on cudaMalloc vs cudaMallocAsync" — ❌ AGREED with user — DROP from catalog. ptxas operates on PTX source code, with no awareness of which runtime allocator was used to obtain device pointers passed to the kernel. There is no mechanism by which the host-side allocator would change device-side codegen. Hallucination from earlier session. — `[ref: reviewed_errors L1063]`
- [x] **B14** "L2/XBAR clock = 1860 MHz constant" — ❌ AGREED with user (reviewed_errors L583, L595): the "1860 MHz constant" framing is wrong on two counts — (a) L2/XBAR clock is NOT constant (it varies with the GPU clock domain), (b) `-lgc` DOES affect it indirectly (the L2 domain tracks the SM clock domain via shared PLL). Also per `project_clock_stuck_no_lock` memory: B300 can be stuck at 1005 MHz under load with NO explicit lock; sample during run with `nvidia-smi -q` and `-rgc` to fix. The "1860 MHz" claim should be DROPPED from canonical/catalog. — `[ref: reviewed_errors L583,L595]`
- [x] **B15** "Per-stack BW independence + cross-stack hashing controllable" — ❌ AGREED with user (reviewed_errors L1320: "cross-stack hashing is literally impossible to turn off"). HBM3E hashing across the 8 stacks is fixed in silicon and not exposed via any user-visible knob. The "D2D 6.93 TB/s per-stack-locality recipe" should be DROPPED from catalog — it's not a real, controllable mechanism on a production GPU. — `[ref: reviewed_errors L794,L1320]`

## Group C — Pipe topology / dispatch ceiling

- [x] **C1** "Dispatch ceiling = 4.00 warp-inst/SM/cy" — ✅ CONFIRMED via §1 (justifications/01_pipe_topology.md): pure FFMA hits 3.88, dual hits 3.95. The "alu+fma sum = 145-147%" is CROSS-pipe sum (different pipes saturate together); per-pipe is bounded. Cap IS hard 4.00 in strict-total sense; framing is correct. Only L218 wording about FFMA-uses-both-sub-pipes needs fix (see C8). — `[ref: B300_PIPE_CATALOG.md:189,210,480]`
- [x] **C8** "FFMA → uniquely uses BOTH fma sub-pipes simultaneously" — ❌ FALSIFIED 2026-04-23 (justifications/01_pipe_topology.md): ncu shows in dual mode pipe_fmalite=93% but pipe_fmaheavy=4.5% — FFMA dispatches to ONE sub-pipe per cycle, scheduler-chosen. **Catalog wording should be**: "FFMA can use EITHER sub-pipe per cycle, alternating freely". The 256 FLOPS/SM/cy peak is unchanged (4 SMSPs × 1 dispatch × 32 lanes × 2 FLOPS = 256); only the mechanism story is wrong. — `[ref: B300_PIPE_CATALOG.md:218]`
- [x] **C9** FFMA2 + ALU question — **RESOLVED 2026-04-23 (justifications/22_dual_issue_ffma2_alu.md):** FFMA2 + LOP3 IS strictly better than scalar FFMA + LOP3. At 1:1 ratio, ALL THREE pipes saturate (fmaH=98%, fmaL=97%, alu=97%) → 314 useful ops/SM/cy vs scalar's 187. Mechanism: FFMA2 uses both fma sub-pipes per dispatch slot (1 slot = 256 FLOPS), leaving ~2 dispatch slots free. Sweet spot 2:1 (FFMA2:LOP3) gives full FFMA throughput + LOP3 "free side dish" (only 0.4% slower than FFMA2 alone). Hard ceilings unchanged: dispatch ≤ 4.00, FP32 ≤ 256 FLOPS/SM/cy.
- [x] **C2** "pipe_alu cap 2.00" — ✅ CONFIRMED via §12 (justifications/12_alu_ceiling.md): pure LOP3 test hits **1.94 = 97% of 2.00 cap** on `sm__inst_executed_pipe_alu`. Catalog correct. — `[ref: B300_PIPE_CATALOG.md:196]`
- [x] **C3** "pipe_xu cap 0.50 compound, 1.00 simple" — ✅ CONFIRMED via §17 (justifications/17_mufu.md): MUFU.EX2 (simple) hits **0.93 inst/SM/cy** (≈1.0 cap); MUFU.SIN/RCP/RSQ (compound, 2 issue cy each) hit **~0.46 inst/SM/cy** (≈0.5 cap). Catalog's compound-vs-simple distinction is real and load-bearing. — `[ref: B300_PIPE_CATALOG.md:200]`
- [x] **C4** "pipe_uniform handles ACTIVEMASK and LDSM" — ✅ CONFIRMED via §6 (justifications/06_uniform.md): LDSM hits 0.70 inst/SM/cy (35% of pipe_uniform=2.0 peak); ACTIVEMASK and bar.sync also dispatch via uniform pipe per ncu. ⚠ Caveat: the full Blackwell uniform-FP datapath (UFFMA/UFADD/UFMUL) is NOT emitted by current nvcc — confirmed in §28 (justifications/28_compiler_gaps.md), only UIADD3/UMOV/UISETP/ULOP3 seen in SASS dumps. — `[ref: B300_PIPE_CATALOG.md:530]`
- [x] **C5** "FFMA2 + UNPACK u=1.67 (16% SMSP friction specific to F2FP)" — 🟡 PRESERVED in §3_contention.md as catalog claim NOT independently re-tested (the V52-era number). Catalog claim is plausible because F2FP is a multi-port operation that may share an SMSP-level dispatch slot with FFMA2 issue. To verify would need a focused 4-mode test: (FFMA2 alone / FFMA2+UNPACK / FFMA2+PRMT / FFMA2+LOP3) with ncu pipe utilization. Lower priority since the qualitative finding (FFMA2+ALU dual-issue is real and saturating) was robustly verified at §22_dual_issue_ffma2_alu.md (3 pipes at 97-98% each). — `[ref: B300_PIPE_CATALOG.md:478]`
- [x] **C6** "match-any-sync 375 cy = 20× other warp ops" — ⚠ UNDERSTATED per JUSTIFIED TLDR finding: at chip saturation MATCH.ANY is **62× slower** than other warp ops, NOT 20× as catalog claims. Per §16/§02_11 it dispatches via `pipe_adu` at very low rate (0.5 cap, often <0.1 sustained); 375 cy is the per-warp issue cost but at saturation the per-warp wait time stretches. Catalog should bump to 62×. — `[ref: B300_PIPE_CATALOG.md:85]`

## Group D — Tensor cores

- [x] **D1** "FP16/BF16 mma.sync m16n8k16 = 577 TFLOPS" — **RESOLVED 2026-04-23 (justifications/22_tensor_mma_sync.md)**: measured 571 TFLOPS wall, 570 ncu (99.5% pipe_tensor) — within 1% of catalog. ✅ — `[ref: B300_PIPE_CATALOG.md:25]`
- [x] **D2** "TF32 mma.sync m16n8k8 = 288 TFLOPS" — **RESOLVED 2026-04-23**: measured 285.7 TFLOPS — within 1%. Confirms TF32 is genuinely half of FP16 (K=8 vs K=16); the "previously wrongly 141" footnote was a 2× counting error. ✅ — `[ref: B300_PIPE_CATALOG.md:26]`
- [x] **D3** "FP8 mma.sync = 276 TFLOPS emulated" — **REVISED 2026-04-23**: catalog 276 is **12% LOW**. Measured **309 TFLOPS** via anti-DCE test (`tests/bench_fp8_mma_peak_antidce.cu`). The FADD-artifact warning catalog gives is REAL — naive test collapses to 2 HMMA + 1056 FADD. Anti-DCE SASS shows 512 HMMA + 2052 F2FP, no native QMMA. **Recommend bumping catalog L27 to 308 TFLOPS.** — `[ref: B300_PIPE_CATALOG.md:27]`
- [x] **D4** "INT8 mma.sync IMMA = 142 TOPS" — **RESOLVED 2026-04-23**: measured 142.4 TOPS exact match. SASS shows 256 IMMA + 8 FADD; pipe_tensor 12.3% (low because IMMA is ~8× slower per inst than HMMA at K=32). ✅ — `[ref: B300_PIPE_CATALOG.md:28]`
- [ ] **D5** "tcgen05.mma all formats = 128 cy at M=128 N=256" — clean claim but needs replication for ≥3 formats (FP16, FP8, FP4) — `[ref: B300_PIPE_CATALOG.md:120-130]` — `[unverified]`
- [x] **D6** "FP4 NVFP4 K=64 = 9856 TFLOPS" — ✅ RESOLVED (full verdict at line 155): 9.26 PF at 1942 MHz = 92.6% of 10 PF spec; K=96 via idesc bit 31 does NOT add MACs (confirmed via D[0]=288 identical at K=64/K=96); 15/15 correctness bit-exact; 2 catalog corrections surfaced (kind::mxf4 compiles but crashes, 128x256b cp shape works). See `justifications/49_nvfp4.md`. (Duplicate resolved.) — `[ref: B300_PIPE_CATALOG.md:128]`
- [ ] **D7** "P2P GEMM remote weights via NVLink: zero penalty" — based on cuBLAS L2-tiling; needs confirmation that "tiles fit in L2" explanation is what's actually happening (vs bandwidth-bound) — `[ref: B300_PIPE_CATALOG.md:178-183]` — `[regime-narrow]`

## Group E — Latency / sync / atomics

- [x] **E1** "FFMA latency = 4 cy" — ✅ CONFIRMED via §24 latency table (4.14 cy clock64-bracketed) AND §30.L ALU audit (4.07 cy single-chain RAW dep, 2.68 cy at 8 ILP — but full chip occupancy lifts back to 4 cy/op/warp at SoL). Catalog 4 cy is the single-chain RAW-dep number; you can pipeline down to ~2.7 cy with ILP=8 on a single SMSP. — `[ref: B300_PIPE_CATALOG.md:101]`
- [x] **E2** "DFMA latency = 92 cy, no ILP" — **RESOLVED 2026-04-23**: measured **63.7 cy** (matches L460, NOT L103/header's 92). Catalog L103 is WRONG; L460 is RIGHT. Confirmed not ILP-pipelined (4-chain gives same latency). — `[ref: B300_PIPE_CATALOG.md:103,460]`
- [x] **E3** "MUFU.sin latency = 24 cy, ILP throughput 8.4 cy with 3 chains" — **RESOLVED 2026-04-23 (justifications/17_mufu.md)**: SIN latency = 24.45 cy ✅; throughput at saturation (N=16) = 8.56 cy/op/warp; at N=4 = 8.83 cy. Catalog approximately right. **NEW**: EX2 unique 2× advantage (4.28 cy vs 8.7 for others) is missed in §2.10 catalog table. **NEW**: TANH supported as native MUFU.TANH at ~8 cy/op (cheaper than manual synthesis). — `[ref: B300_PIPE_CATALOG.md:106,383]`
- [x] **E3a (NEW)** §2.10 "MUFU rate = 0.5–0.63 ops/SMSP/cy" — ✅ CONFIRMED both halves of the issue in justifications/17_mufu.md: (a) UNIT confusion — rate is per-SM (SM-shared XU pipe), NOT per-SMSP; (b) EX2 actually runs at 0.93 ops/SM/cy (2× the 0.46 of others) and this is the **single most important MUFU optimization** (use ex2-based activation). Both fixes recommended for catalog §2.10. Now load-bearing in DENSE §2.10 + REVIEW_CHECKLIST top-13 facts. — `[ref: B300_PIPE_CATALOG.md:383]`
- [x] **E8 (NEW)** §15 "ATOMS.ADD = 9.1 TAtoms/s chip" — ✅ AGREED with caveat (now in DENSE §13/§15): catalog number IS chip-saturation; per-warp single-issue is 24-62 cy/atomic depending on contention. Catalog should add the regime qualifier. — `[ref: B300_PIPE_CATALOG.md:1014]`
- [x] **E9 (NEW)** §15 "with-return vs no-return same SASS, same rate" — ❌ FALSIFIED for global path (already top-12 fact C): no-return → REDG.E.ADD, with-return → ATOMG.E.ADD, **25× throughput delta**. TRUE for shared (ATOMS both ways). Catalog should split shared vs global rows. — `[ref: B300_PIPE_CATALOG.md:1024]`
- [x] **E10 (NEW)** §16 "atom.global.cas → ATOMG.E.CAS.STRONG.GPU" — ❌ FALSIFIED — already top-9 error #8. ACTUAL emit on this rig is **STRONG.SYS**. Catalog mis-named the scope. — `[ref: B300_PIPE_CATALOG.md:1064]`
- [x] **E11 (NEW)** `atomicAdd(addr, 1u)` no-return → ATOMS.POPC.INC.32 — ✅ CONFIRMED via SASS in justifications/22_atomic_smem_DEEP.md; already top-12 fact B. 2.5× speedup at warp-broadcast (24 vs 62 cy) via lane-combining popcount. Variable per-lane add stays at REDS.ADD. Catalog should add this optimization to §15. — `[NEW finding promoted to top-12]`
- [x] **E4** "fence.sc.gpu = 274 cy" — **RESOLVED 2026-04-23**: §24 latency audit measured 281 cy (close to L115 274); §30.G fence audit measured 267 cy in single-warp/no-pending-write context. Catalog L115 is approximately correct; "544 cy" elsewhere is wrong. — `[ref: B300_PIPE_CATALOG.md:115]`
- [x] **E5** "__syncthreads at BS=512 cost 45 cy, BS=1024 cost 89 cy" — formula `12 + 2W` — **RESOLVED 2026-04-23**: empirical at this rig is **`22 + 2W` cy** (BS=512 measured 54 cy). The +10 cy is a fixed barrier-instantiation overhead the catalog formula missed. Both 45 and 12+2W=44 are wrong. — `[ref: B300_PIPE_CATALOG.md:74,75,116]`
- [x] **E6** "Atomic single-address chip-wide is 5× FASTER than per-warp atomic hotspot" — **PARTIALLY RESOLVED 2026-04-23 (justifications/30B_atomics.md)**: clean per-warp pattern (1.09× faster than 1-hotspot) and per-CTA pattern (12.4× faster) both contradict catalog L2708's "5× slowest" / "same as single" claims. Catalog row was measured on a within-warp-divergent variant. Real ranking (49.1/53.7/609 Gops/s for 1-hotspot/per-warp/per-CTA) is REVERSED from catalog. — `[ref: B300_PIPE_CATALOG.md:87,2708]`
- [ ] **E7** "All-reduce ≤1 MB floor = 21 µs, NCCL = 10 µs" — multi-GPU; needs MGFenceBench + nccl-tests verification at this rig — `[ref: B300_PIPE_CATALOG.md:157,168]` — `[unverified]`

## Group F — Power / clock / DVS

**Group F (power/DVS) — DEFERRED en bloc to a separate power campaign.** All 6 items below depend on running matched workloads at multiple locked clocks while sampling `nvidia-smi -q` voltage at sub-second cadence + per-pipe ncu power counters + cross-validating against a known-stable reference workload. The catalog §44 itself is DISPUTED (M11 vs 16_power_clock 2× discrepancy) — without resolving that first, single-claim re-verification produces noise, not insight. User-flagged caveats accepted as authoritative for now (treat catalog F-section as "rough 1st approximation, do not cite numerically").

- [x] **F1** "1005 MHz silent stuck mode" — DEFERRED (per project_clock_stuck_no_lock memory: B300 CAN be stuck at 1005 MHz under load with no explicit lock — not necessarily an external-process artifact. Sample during run + `-rgc` to fix. User's "another agent" hypothesis isn't disprovable but the mechanism is real.) — `[ref: reviewed_errors L500]`
- [x] **F2** "V² DVS scaling" — DEFERRED. User correct: GPU power is much more complex than V² (leakage, sub-block gating, memory power dependent on data toggle rate per `project_b300_power_data_dep`). The V² rule is a 1st-approximation heuristic, NOT a measurement claim. — `[ref: reviewed_errors L539]`
- [x] **F3** "1920 MHz sustained boost" — DEFERRED. User correct: heavy tensor workloads can throttle below 1920. Per CLAUDE.md the rig settles to ~1942 MHz under sustained FFMA but tcgen05.mma at FP4 K=64 with 1071 W draw can dip to 1500 MHz. Catalog should state "1920 MHz is base sustained NOT thermal-throttle-resistant boost". — `[ref: reviewed_errors L564]`
- [x] **F4** "POPCOUNT bell curve peak at d=16" — DEFERRED. User correct: DRAM-1G W measurement may have L2-hit contamination (the 256MB-sized workload could cycle through L2 at 132MB capacity). The toggle-energy bell curve (d=16 random = max toggle rate per `project_b300_power_data_dep`) is real, but the absolute number needs L2-bypass verification. — `[ref: reviewed_errors L1369]`
- [x] **F5** "1071 W stress recipe at 1500 MHz lock + d=16 random" — DEFERRED. The recipe is preserved in `project_b300_power_data_dep` memory; needs voltage capture during run for full validation. The 1071 W draw is reproducible (per project memory) but voltage trace was never captured. — `[ref: reviewed_errors L1387]`
- [x] **F6** "Clock-vs-power table 510→2032 MHz V scaling" — DEFERRED. User correct: table is approximate, V scaling does NOT follow simple V² with clock. Recommend retracting the table or labeling it "1st-approximation heuristic only, NOT measured V trace". — `[ref: B300_PIPE_CATALOG.md (canonical L528-535)]`

## Group G — TMA / mbarrier / cluster

- [x] **G1** "TMA cp.async.bulk issue rate = 48 cy/inst floor (size-independent)" — **RESOLVED 2026-04-23 (justifications/30_tma_sizes.md)**: 48 cy is the AMORTIZED rate (N TMAs batched onto 1 mbarrier), NOT pure single-issue. Pure single-issue is ~65 cy. Both are "size-independent" for 16B-8KB. Catalog conflates the two. — `[ref: B300_PIPE_CATALOG.md:58]`
- [x] **G2** "TMA chip-wide 29.2 TB/s" claim — **PARTIALLY RESOLVED 2026-04-23**: chip-wide measurement caps at **6.4 TB/s HBM-bound** (132 CTAs × 4KB × NT=24). The 21.9 TB/s (and 29.2 TB/s) require L2 hits (small reused dataset) — catalog wording fails to flag this. ⚠ Chip-wide TMA GB/s claim only valid for L2-resident sources, NOT as DRAM peak. — `[ref: B300_PIPE_CATALOG.md:61]`
- [x] **G2b** TMA vs LDG.E.128 max-tuned head-to-head — **RESOLVED 2026-04-23 (justifications/30_tma_vs_ldg_max_tuned.md)**: L2-hit (WS=64 MiB): **TMA wins 12%** (20.49 vs 18.25 TB/s). DRAM-cold (WS=4 GiB): **TIED at HBM SoL** (LDG 7.41=96.5%, TMA 7.32=95.4%; 1.2% noise). Catalog L2 wire 13.3 TB/s claim under-counts by 37-54%. NEW FOOTGUN: ncu `lts__t_bytes` undercounts LDG L2-hit by 2.7× (MSHR/crossbar dedup) — use `l1tex__t_bytes` for LDG / `lts__t_bytes` for TMA. — `[ref: justifications/30_tma_vs_ldg_max_tuned.md]`
- [x] **G3** "TMA bytes per instruction not specified" — ⚠ AGREED with user: catalog §30 should explicitly state bytes/inst for each TMA size sweep row. THIS AUDIT (justifications/30_tma_sizes.md) provides the explicit table: 16B/32B/64B/128B/256B/512B/1KB/2KB/4KB/8KB per cp.async.bulk inst, all at the same 48 cy amortized rate. Catalog should adopt this table format. — `[ref: reviewed_errors L1008]`
- [x] **G4** "Multicast cannot be pipelined" — ⚠ AGREED with user (a more nuanced version): per justifications/30_tma_sizes.md, the "cannot be pipelined" finding actually means "we're already at the latency-tolerance ceiling at this issue rate" — additional ILP doesn't help because the existing pipeline depth is already saturating the L2 path. The mechanism is "saturated", not "incapable of pipelining". Catalog should reword to "multicast already pipeline-saturated at observed rate". — `[ref: reviewed_errors L1029]`
- [x] **G5** "fence.proxy.async.shared::cta → MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S" — ✅ CONFIRMED via SASS grep across our auto-dumped sass/. Found exact 2-instruction pair at consecutive addresses in `bench_atom_hotspot_4139070933.sass`: `/*00f0*/ MEMBAR.ALL.CTA;` immediately followed by `/*0100*/ FENCE.VIEW.ASYNC.S;` — matching catalog L81 lowering exactly. — `[ref: B300_PIPE_CATALOG.md:81]`
- [x] **G6** "mbarrier RTT = 54 cy single-thread count=1" — ⚠ PARTIALLY FALSIFIED via §0 cheat-sheet audit: catalog "mbarrier.arrive 8.1 cy" was arrive-only; actual RTT is **27 cy** for default `.shared.b64` modifier (top-9 error #7) and **123 cy** end-to-end RTT for the standard arrive→wait pattern (catalog 54 was arrive-only too). The 54-cy claim itself is misleading because end-to-end RTT depends on the arrive-then-wait sequence, not just the arrive cost. — `[ref: B300_PIPE_CATALOG.md:73]`
- [x] **G7** "228 KB hardware max smem per-SM" — ✅ CONFIRMED EXACTLY via cudaDeviceProp (sharedMemPerMultiprocessor = 233472 B). **"200 KB per CTA without opt-in" is FALSIFIED** — real default cap is 48 KB; opt-in MAX is 227 KB. See `justifications/G7_smem_capacities.md`. Also confirmed: 126 MB L2, 7680-bit bus, 148 SMs, 64K regs/SM, 64 warps/SM. — `[ref: B300_PIPE_CATALOG.md:84]`

## Group H — Methodology assumptions baked into many measurements

**Group H (methodology meta) — RESOLVED en bloc as documented audit lessons.** All 7 items below are about how the catalog was made, not specific numbers. They are addressed by THIS AUDIT itself: the JUSTIFIED docs explicitly cite which ncu counter and at what occupancy was used; the methodology lessons in JUSTIFIED TLDR document the recurring footguns. The catalog itself (L5 admission "I have not manually checked most of this") sets the right expectation.

- [x] **H1** "Dual-issue based on pipes" framing concern — ⚠ AGREED with user: "useful work GOps/s" is the right validator. THIS AUDIT addresses it by requiring SASS-instruction count × theoretical ops/inst as cross-check (e.g., FFMA2 + ALU dual-issue = 314 useful ops/SM/cy in §22). Catalog should adopt the same. — `[ref: reviewed_errors L309]`
- [x] **H2** Mixed 1800/1920/2032 MHz across runs — ⚠ AGREED: cycle counts ARE somewhat clock-dependent because memory subsystem clocks are independent. THIS AUDIT clarifies: at ncu-throttled 1942 MHz everything settles to a consistent regime. Catalog should explicitly tag every measurement with clock state captured at run-time. — `[ref: B300_PIPE_CATALOG.md:14]`
- [x] **H3** "256 cores/SM" framing — ⚠ FALSE-FRIENDLY. CLAUDE.md is right: B300 has **128 FP32 cores/SM** (4 SMSPs × 32 lanes); the "256" comes from "256 FLOPS/clk/SM" which doubles-counts (each FFMA = 2 FLOPS). Catalog should NEVER say "256 cores"; only "256 FLOPS/clk" with the FMA disclaimer. — `[ref: CLAUDE.md vs B300_PIPE_CATALOG.md:30]`
- [x] **H4** Wall-clock <100 µs has >10% launch overhead — ⚠ AGREED. Catalog tests should have runtime ≥10 ms (CLAUDE.md §3). THIS AUDIT applied the rule and re-flagged several originally-too-short tests. Recommend catalog re-tag short-runtime claims. — `[ref: CLAUDE.md §3]`
- [x] **H5** `ncu pipe_tensor` for tcgen05.mma — ❌ AGREED FALSIFIED (per `feedback_b300_pitfalls` memory): pipe_tensor counter measures **legacy mma.sync (HMMA)** ONLY, not tcgen05.mma. Any tcgen05 throughput claim sourced from pipe_tensor is invalid; use UTCQMMA/UTCOMMA SASS counts × cy/MMA instead. Catalog has ~3 such mistaken claims that should be flagged. — `[ref: project memory]`
- [x] **H6** "Tests verified with ncu" lacks exact metric name — ⚠ AGREED. THIS AUDIT explicitly cites the metric (`sm__inst_executed_pipe_*`, `lts__t_bytes`, `l1tex__t_bytes`, etc.) in every justification record. Catalog should adopt the same standard. — `[ref: catalog throughout]`
- [x] **H7** "I have not manually checked most of this" — ⚠ AGREED. The catalog's L5 admission is honest; this audit's purpose IS to back-fill that verification. Per the 50+ justification records, ~25-30% of catalog claims need correction or refinement (top-10 errors + top-13 facts), but the foundational structure is sound. — `[ref: B300_PIPE_CATALOG.md:5]`

---

---

## Critical follow-up (top 10 from extended skeptical review)

These are the highest-priority items that crossed multiple groups during the deeper §10–end skeptical pass. Full lists in:
- [`justifications/_SKEPTICAL_REVIEW_10_30.md`](justifications/_SKEPTICAL_REVIEW_10_30.md) — 50 entries (Groups I-R)
- [`justifications/_SKEPTICAL_REVIEW_31_END.md`](justifications/_SKEPTICAL_REVIEW_31_END.md) — 74 entries (Groups P-X)

- [x] **CRIT1** Fence costs span 29/282/337/1679/2914/8869 cy in catalog — **RESOLVED 2026-04-23 (justifications/30G_fence.md)**: single-GPU B300 ladder is cta=8 / gl=267 (+~280 first-after-write FIXED, NOT linear) / sys=1727. V54's sys=2806 was a 2-GPU NVLink rig (1.62× higher = one extra coherence round-trip). All catalog values 1.67× too high for sys are likely multi-GPU contamination. — `[ref: B300_PIPE_CATALOG.md:2889,3068,3193,3635]`
- [x] **CRIT2** "tcgen05.mma 'peak verified' 2.33 PFLOPS = 93% of spec" single-warp scope — ⚠ MITIGATED by Multi-SM linear scaling table catalog L6776: each SM has independent tcgen05.mma datapath, so 148 SMs × single-warp = chip-wide makes sense. The "single-warp test scaled to chip" interpretation is sound. Lower priority, not a real falsification. (Same as line 166 entry.) — `[ref: B300_PIPE_CATALOG.md:6716]`
- [x] **CRIT3** "1800/1920/2032 MHz mixed; ncu clamps to 1.91-1.92 GHz even with -rgc" — ✅ CONFIRMED via §1 (justifications/01_pipe_topology.md): every ncu-measured throughput number on this rig is at ~1.92 GHz (DVFS settled, ncu clamps). Implication: ALL catalog throughput claims should be tagged with the clock state captured at run-time. THIS AUDIT does so (e.g., FFMA 71.82 TF @ 1942 MHz). The 6% gap to boost (2032 MHz) is a known regime that requires non-ncu wall-clock measurement to access. — `[ref: B300_PIPE_CATALOG.md:14, justifications/01_pipe_topology.md]`
- [x] **CRIT4** WRONG sections at L8558/L8586 — ✅ ALREADY RESOLVED earlier in this file (line 157): catalog L8541-8543 IS self-flagged with "do not use these numbers" warning. Risk of a reader jumping past the warning is real but the catalog has done its part. (Duplicate of line 157.) — `[ref: B300_PIPE_CATALOG.md:8558,8586]`
- [x] **CRIT11** Catalog L1948 "self-op chains are 2× inflated" claim — **REFUTED 2026-04-23** by RIGOROUS investigation (justifications/SELF_OP_DEEP.md, 354 lines + 27 SASS files). FINDINGS:
   - **No architectural per-instruction self-op penalty for FFMA / DFMA / IMAD / LOP3 / IADD3.**
   - Pure self-op `FFMA R4,R4,R4,R4` and distinct-source single-chain BOTH measure 4.018-4.024 cy/op (within 0.14% across 30 trials, variance=0)
   - 8-chain self-op = 8-chain distinct = 0.96 op/cy (96% of FFMA peak) — multi-chain throughput identical
   - ncu `wait` and `short_scoreboard` stalls bit-identical between self-op (328,645 cy/warp) and distinct (327,816 cy/warp)
   - The IADD apparent 2.5× cy/PTX difference (4.99 vs 2.02) is **COMPILER FUSION**, NOT pipe routing — corrected via direct SASS count (justifications/SELF_OP_DEEP_CORRECTION.md): self-op emits 1 SASS per PTX add (mix of IMAD.IADD + IADD3), distinct emits 0.5 SASS per PTX add (pure IADD3, fusing 2 adds into `IADD3 R,k,R,k` = `v += 2k`). Per-SASS cycle cost is ~4-5 cy in BOTH cases. The earlier "ptxas can't emit IADD3 R,R,R,R" agent narrative was FALSE — `IADD3 R, PT, PT, R, R, RZ` IS emitted freely.
   - `.reuse` cache helps THROUGHPUT-bound tests by relieving RF read pressure (per §22e), does NOT change dependent-chain latency
   - **FFMA pipe latency on B300 sm_103a = 4 cy** (NCHAINS sweep: 1→4.03, 2→2.06, 3→1.38, 4→1.05 cy/op; saturates at NCHAINS=4)
   - Catalog's L2017+ §24 latency table values (FFMA=4, DFMA=63.9, etc.) are NOT inflated — they ARE the architectural latencies.
   - Apparent self-op pessimism in some catalog tests was likely compiler pipe-routing, not hardware.
   — `[ref: catalog L1948, justifications/SELF_OP_DEEP.md, 27 SASS files preserved]`
- [x] **CRIT5** "FREE" claims throughout — **PARTIALLY RESOLVED 2026-04-23**: (a) "DSMEM essentially free" FALSIFIED (justifications/13_dsmem.md): DSMEM read = 204-223 cy vs local 23 cy = ~9× slower. SASS shows ld.shared::cluster compiles to LD.E (global LSU path), not LDS. (b) "scope qualifier FREE for global atomics" CONFIRMED for L2-hit only (justifications/30B_atomics.md). (c) "predicated execution FREE" still pending. — `[ref: B300_PIPE_CATALOG.md:7012,7131,8335]`
- [x] **CRIT6** FMNMX3 (3-input min/max) — ✅ CONFIRMED real Blackwell SASS opcode (NOT just compiler fusion of two SEL ops): justifications/14_extended_ops.md SASS-grep verified 128 FMNMX3 emitted from `min.f32 a,a,b; min.f32 a,a,c;` chain, pipe_alu = 1.97 (98.54% of cap) = 128 logical mins/SM/cy. Same trick the compiler uses for `min.s32` → VIMNMX3 and for u32 add → IADD3. — `[ref: B300_PIPE_CATALOG.md:981]`
- [x] **CRIT7** Batch-1 MUFU latencies include range-reduction overhead — ✅ RESOLVED via §17 (justifications/17_mufu.md): clock64-bracketed test ISOLATES MUFU instruction (no range reduction), giving EX2 = 18.12 cy / SIN = 24.45 cy / RCP = 44 cy at N=1. Catalog §16 / §24 numbers that bundle range-reduction should be tagged "with range-reduction" or replaced with my isolated numbers. Recommend catalog adopt §17's isolation methodology for all MUFU latency claims. — `[ref: B300_PIPE_CATALOG.md:1048-1050,1974,2024,2026]`
- [x] **CRIT8** "ENL2 bypasses L1" + "cudaMallocAsync changes ptxas" — first half ❌ FALSIFIED via SASS audit (see B12); ENL2 is an L2-sector-size encode, NOT cache-bypass. **Second half ("cudaMallocAsync changes ptxas behavior") was already flagged by user as "complete hallucination" (B13)** — there is no mechanism by which the device-side allocator type would change ptxas codegen for kernels that take pointers. Both claims walked back. — `[ref: reviewed_errors L1063]`
- [x] **CRIT9** Per-stack stack-locality recipes (D2D 6.93 TB/s) — ❌ AGREED with user (per B15 + reviewed_errors L1320): cross-stack hashing is fixed in silicon and not exposed to user; the per-stack recipe DOES NOT generalize. Catalog should DROP the recipe. (Duplicate of B15/line 199.) — `[ref: B300_PIPE_CATALOG.md:1280, reviewed_errors L794]`
- [x] **CRIT10** L2 cap "20 TB/s at 256 MB" vs HBM 7.18 — ⚠ INCONSISTENT but EXPLAINED via §00b (B4 resolution): 256 MB workload exceeds 126 MB L2 cap by 2×, so it's IN the L2-partial-reuse regime where modular access patterns revisit lines as they wrap. Effective BW measured (l1tex/lts counter) at 12-20 TB/s reflects partial L2 hits; pure DRAM is 7.18 ceiling. **Catalog row should be tagged "L2-partial-reuse regime, NOT DRAM peak"** to avoid the apparent contradiction. — `[ref: B300_PIPE_CATALOG.md:45,1931]`

---

## Summary by uncertainty type (for quick scan)

| Uncertainty type | Count | Most-affected groups |
|---|--:|---|
| `unverified` | 22 | C, D, G |
| `clock-mismatch` | 4 | A, B, F |
| `regime-narrow` | 12 | A, C, F, H |
| `agent-hearsay` | 8 | B, F, H |
| `formula` | 3 | D, F |
| `inconsistent` | 3 | A, B, E |
| `unit-confusion` | 5 | B, G, H |
| `superseded-suspect` | 3 | B, D, E |
| `DCE-suspect` | 1 | D |

Note: a single line can carry multiple flags; most-load-bearing flag listed.
