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
| 9 | "ld.shared bank-conflict-sensitive" | only TRUE for v2/v4 wide; FALSE for 32-bit on B300 | §22j |

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

## STILL-OPEN HIGH-PRIORITY (top 10 to review)

These are the [ ] items the user is most likely to have a strong opinion on. Open `B300_PIPE_CATALOG.md` line by line for the catalog claim.

- [ ] **A3** "FFMA2 packed = 72.3 TFLOPS = 99.4%" — needs SASS verification that FFMA2 (not FFMA scalar pair) is actually emitted — `[ref: catalog L31]`
- [ ] **A5** "FP64 DFMA = 0.95 TFLOPS" — CLAUDE.md says theoretical 1.20 TF; 0.95/1.20 = 79% suggests under-saturation; needs chip-occupancy retest — `[ref: catalog L35]`
- [ ] **D5** "tcgen05.mma all formats = 128 cy at M=128 N=256" — catalog math is self-consistent (linear scaling table), but no fresh measurement on this rig — `[ref: catalog L120-130, L6686+]`
- [x] **D6** "FP4 NVFP4 K=64 = 9856 TFLOPS" — ✅ **PROPERLY VERIFIED 2026-04-23 (justifications/49_nvfp4.md)** with rigorous audit (3 claims confirmed, 2 corrected): (a) **9.26 PFLOPS at 1942 MHz** = 92.6% of 10 PF spec; cy/MMA = 128.001 matches catalog's 128.01. (b) K=64 D[0]=288.0 = K=96 D[0]=288.0 — confirms K=96 via idesc bit 31 DOESN'T add MACs (would be 432 if it had). (c) 15/15 K=64 correctness tests bit-exact. (d) **CORRECTION**: catalog's "kind::mxf4.block_scale.block32 rejected by ptxas" is FALSIFIED — it actually compiles but crashes at runtime. (e) **CORRECTION**: catalog's "128x256b cp shape crashes" is NOT REPRODUCED — it works fine with 8 KB smem. 14 evidence files preserved. — `[ref: catalog L9213-L9430, justifications/49_nvfp4.md]`
- [ ] **CRIT2** "tcgen05 'peak verified' single-warp scope mismatch" — MITIGATED by Multi-SM linear scaling table L6776 (148 SMs each independent → 148× single-warp = chip-wide makes sense). Per audit, this is now lower priority. — `[ref: catalog L6716]`
- [ ] **CRIT4** WRONG sections kept in catalog without retroactive correction — needs catalog meta-edit pass — `[ref: B300_PIPE_CATALOG.md:8558,8586]`
- [ ] **CRIT6** FMNMX3 (3-input min/max) — likely compiler fusion not native opcode; needs cuobjdump — `[ref: catalog L981]`
- [ ] **CRIT8** "ENL2" SASS bypasses L1 / cudaMallocAsync changes ptxas behavior — both claims suspect, need ISA reference — `[ref: catalog SASS sections]`
- [ ] **CRIT9** Per-stack stack-locality recipes (D2D 6.93 TB/s) — cross-stack hashing hard to control; recipe may not generalize — `[ref: catalog L1280]`
- [ ] **F1-F6** All power claims (DVS scaling V², 1005 MHz stuck floor, 1071 W stress recipe, V² formula) — catalog §44 is DISPUTED (M11 vs 16_power_clock 2× discrepancy in canonical) — needs power-per-pipe replication

(The full ~50 still-open entries continue below by group.)

---

## Group A — Headline FP32 / FFMA

- [ ] **A1** "FP32 FFMA peak = 71.8 TFLOPS = 98.8% of theoretical 72.7 TFLOPS at 1.92 GHz" — the 72.7 denominator uses 1920 MHz; CLAUDE.md says theoretical at 2032 MHz boost is 76.96 TFLOPS — `[ref: B300_PIPE_CATALOG.md:30]` — `[clock-mismatch]`
- [ ] **A2** "256 FLOPS/clk/SM" formula assumes scalar FFMA dispatches to BOTH fma sub-pipes simultaneously — V52 confirmed this for FFMA but also showed "dispatch cap=4" framing is misleading (alu+fma=147%) — `[ref: B300_PIPE_CATALOG.md:30,210]` — `[regime-narrow]`
- [ ] **A3** "FP32 via FFMA2 packed = 72.3 TFLOPS = 99.4%" — same chip-FLOPS as scalar FFMA per claim; needs SASS verification that FFMA2 is actually emitted (not FFMA scalar pair) — `[ref: B300_PIPE_CATALOG.md:31]` — `[unverified]`
- [ ] **A4** "FP16 via HFMA2 = 72.3 TFLOPS-FP16, no extra throughput on scalar path" — claim is that compiler packs into HFMA2 anyway; verify with SASS for both packed and scalar tests — `[ref: B300_PIPE_CATALOG.md:32,33]` — `[unverified]`
- [ ] **A5** "FP64 DFMA = 0.95 TFLOPS = 1/76× of FFMA" — CLAUDE.md says 1.20 TFLOPS theoretical (1:64 ratio per SingleToDoublePrecisionPerfRatio); 0.95 < 1.20 implies under-saturation — `[ref: B300_PIPE_CATALOG.md:35]` — `[inconsistent]`
- [ ] **A6** "FFMA dispatch ceiling = 4.00 sm_inst/SM/cy is hard" — V52 demonstrated alu+fma sum = 147% which means total can exceed dispatch (different pipes overlap); need to clarify what "dispatch" means — `[ref: B300_PIPE_CATALOG.md:480]` — `[regime-narrow]`

## Group B — Memory hierarchy

- [ ] **B1** "smem 35.6 TB/s = 98% theoretical (128 B/clk/SM × 148 × 1.92)" — uses 1.92 GHz; if test ran at 2032 boost the SoL would be 38.4 TB/s. Need clock state captured — `[ref: B300_PIPE_CATALOG.md:41]` — `[clock-mismatch]`
- [ ] **B2** "L1 hit (.ca, WS≤1MB) = 36.1 TB/s" but "L1 = 28.7 TB/s" — same tier, different numbers; suggests L1 was sometimes measuring L2-bypass effects. Mechanism unclear — `[ref: B300_PIPE_CATALOG.md:42,43]` — `[inconsistent]`
- [ ] **B3** "L2 plateau 22-26 TB/s — was wrongly 10.2, under-occupied launch" — the 4× spread suggests methodology was historically broken; need to verify whether 22-26 is now stable across launches — `[ref: B300_PIPE_CATALOG.md:44]` — `[regime-narrow]`
- [ ] **B4** "L2 knee gradual 23 → 22 → 20 → 11 TB/s" at "1 GB → 11 TB/s" — but pure HBM is 7.18 TB/s, so 11 TB/s at 1 GB implies L2-amortization at boundary. User asks for ncu split — `[ref: B300_PIPE_CATALOG.md:45]` — `[unit-confusion]`
- [ ] **B5** "DRAM (HBM3E) read = 7.18 TB/s ncu-verified" with "WS≥1GB→8GB consistent" — likely correct; should denominator be spec 7680 or this-device 7672? — `[ref: B300_PIPE_CATALOG.md:46]` — `[unit-confusion]`
- [ ] **B6** "DRAM write = 7.09 TB/s" — User flag: "SM→L2 *write* path is limited to 32B/clk, you cannot get peak HBM write at lower clocks; 1920 vs 2032 might affect this" — `[ref: B300_PIPE_CATALOG.md:46, reviewed_errors L1132]` — `[clock-mismatch]`
- [ ] **B7** "TMEM read 55.92 TB/s, drops to 31 with 4R/iter" — surprisingly high; my back-of-env says theoretical TMEM read is bounded by tcgen05.ld throughput per warp × SMs which gives much less. Methodology + SASS dump needed — `[ref: B300_PIPE_CATALOG.md:50]` — `[unverified]`
- [ ] **B8** "TMEM write 97.93 → 131 TB/s" — even more suspect; needs first-principles bound check — `[ref: B300_PIPE_CATALOG.md:50]` — `[unverified]`
- [ ] **B9** "Constant memory broadcast LDC.32 = 17.8 TB/s eff (~0.55 TB/s actual cache traffic)" — broadcast amplification × 32 lanes; verify denominator — `[ref: B300_PIPE_CATALOG.md:47]` — `[unit-confusion]`
- [ ] **B10** "Local (register spill) 1.3 TB/s = 52× slower than smem" — likely correct but needs spill-depth-controlled test (catalog notes spill cliff at 32 vars) — `[ref: B300_PIPE_CATALOG.md:49]` — `[regime-narrow]`
- [ ] **B11** "256-byte stride v8 .256B L2 cache modifier" test — user remembers achieving HIGHEST DRAM BW % in any microbenchmark with this; catalog has lost track — needs to be re-located in tests/ — `[ref: reviewed_errors L925]` — `[superseded-suspect]`
- [ ] **B12** "LDG.E.ENL2.256 means bypassing L1" — user is skeptical: "Are you sure ENL2 means what you think it means?" — needs ISA reference check — `[ref: B300_PIPE_CATALOG.md, reviewed_errors L1063]` — `[agent-hearsay]`
- [ ] **B13** "ptxas compiles differently based on cudaMalloc vs cudaMallocAsync" — user: "complete hallucination" — `[ref: reviewed_errors L1063]` — `[agent-hearsay]`
- [ ] **B14** "L2/XBAR clock = 1860 MHz constant, NOT affected by -lgc" — user: "1860 MHz is NOT constant" and separately "L2/XBAR *is* affected by -lgc but only indirectly" — needs clock sweep test — `[ref: B300_PIPE_CATALOG.md (canonical L576), reviewed_errors L583,L595]` — `[agent-hearsay]`
- [ ] **B15** "Per-stack BW independence + cross-stack hashing controllable" — user: "cross-stack hashing is literally impossible to turn off" — stack-locality recipes for D2D 6.93 TB/s are dubious — `[ref: reviewed_errors L794,L1320]` — `[agent-hearsay]`

## Group C — Pipe topology / dispatch ceiling

- [ ] **C1** "Dispatch ceiling = 4.00 warp-inst/SM/cy" — V52 shows pipes overlap freely (alu+fma=147%), so the "hard cap" framing is misleading. Per-SMSP dispatch cap of 1 is the real story — `[ref: B300_PIPE_CATALOG.md:189,210,480]` — `[regime-narrow]` — **2026-04-23 update:** ncu re-confirmed (justifications/01_pipe_topology.md): pure FFMA hits 3.88, dual hits 3.95; alu+fma sum = 145%. Cap IS a hard 4.00 in strict-total sense; "free overlap" claim was always about cross-pipe sums, not within-pipe. Catalog framing OK; only L218 wording needs fix (see new C8 below).
- [ ] **C8** "FFMA → uniquely uses BOTH fma sub-pipes simultaneously" — **FALSIFIED 2026-04-23.** ncu shows in dual mode pipe_fmalite=93% but pipe_fmaheavy=4.5% — FFMA dispatches to ONE sub-pipe per cycle, scheduler-chosen, not both. — `[ref: B300_PIPE_CATALOG.md:218]` — `[agent-hearsay]` — Catalog wording needs to change to "FFMA can use EITHER sub-pipe per cycle, alternating freely". Doesn't change the 256 FLOPS/SM/cy peak (4 SMSPs × 1 dispatch × 32 lanes × 2 FLOPS = 256), but the mechanism story is wrong.
- [x] **C9** FFMA2 + ALU question — **RESOLVED 2026-04-23 (justifications/22_dual_issue_ffma2_alu.md):** FFMA2 + LOP3 IS strictly better than scalar FFMA + LOP3. At 1:1 ratio, ALL THREE pipes saturate (fmaH=98%, fmaL=97%, alu=97%) → 314 useful ops/SM/cy vs scalar's 187. Mechanism: FFMA2 uses both fma sub-pipes per dispatch slot (1 slot = 256 FLOPS), leaving ~2 dispatch slots free. Sweet spot 2:1 (FFMA2:LOP3) gives full FFMA throughput + LOP3 "free side dish" (only 0.4% slower than FFMA2 alone). Hard ceilings unchanged: dispatch ≤ 4.00, FP32 ≤ 256 FLOPS/SM/cy.
- [ ] **C2** "pipe_alu cap 2.00" — needs ncu verification of `sm__inst_executed_pipe_alu` for pure-LOP3 test; only V52 explicitly tested this — `[ref: B300_PIPE_CATALOG.md:196]` — `[unverified]`
- [ ] **C3** "pipe_xu cap 0.50 compound, 1.00 simple" — the compound vs simple distinction needs explicit test (MUFU.SIN vs MUFU.EX2) — `[ref: B300_PIPE_CATALOG.md:200]` — `[unverified]`
- [ ] **C4** "pipe_uniform handles ACTIVEMASK and LDSM" — partial verification; the full Blackwell uniform datapath claims (UFFMA, UFADD, etc.) are inferred from NVIDIA docs not measured here — `[ref: B300_PIPE_CATALOG.md:530]` — `[unverified]`
- [ ] **C5** "FFMA2 + UNPACK gives u=1.67 (16% SMSP friction specific to F2FP)" — catalog §3 contention rule; PRMT+FFMA2 hits u=1.95. Need replication — `[ref: B300_PIPE_CATALOG.md:478]` — `[regime-narrow]`
- [ ] **C6** "match-any-sync 375 cy = 20× other warp ops" — likely correct but value matters; should be in latency table — `[ref: B300_PIPE_CATALOG.md:85]` — `[unverified]`

## Group D — Tensor cores

- [x] **D1** "FP16/BF16 mma.sync m16n8k16 = 577 TFLOPS" — **RESOLVED 2026-04-23 (justifications/22_tensor_mma_sync.md)**: measured 571 TFLOPS wall, 570 ncu (99.5% pipe_tensor) — within 1% of catalog. ✅ — `[ref: B300_PIPE_CATALOG.md:25]`
- [x] **D2** "TF32 mma.sync m16n8k8 = 288 TFLOPS" — **RESOLVED 2026-04-23**: measured 285.7 TFLOPS — within 1%. Confirms TF32 is genuinely half of FP16 (K=8 vs K=16); the "previously wrongly 141" footnote was a 2× counting error. ✅ — `[ref: B300_PIPE_CATALOG.md:26]`
- [x] **D3** "FP8 mma.sync = 276 TFLOPS emulated" — **REVISED 2026-04-23**: catalog 276 is **12% LOW**. Measured **309 TFLOPS** via anti-DCE test (`tests/bench_fp8_mma_peak_antidce.cu`). The FADD-artifact warning catalog gives is REAL — naive test collapses to 2 HMMA + 1056 FADD. Anti-DCE SASS shows 512 HMMA + 2052 F2FP, no native QMMA. **Recommend bumping catalog L27 to 308 TFLOPS.** — `[ref: B300_PIPE_CATALOG.md:27]`
- [x] **D4** "INT8 mma.sync IMMA = 142 TOPS" — **RESOLVED 2026-04-23**: measured 142.4 TOPS exact match. SASS shows 256 IMMA + 8 FADD; pipe_tensor 12.3% (low because IMMA is ~8× slower per inst than HMMA at K=32). ✅ — `[ref: B300_PIPE_CATALOG.md:28]`
- [ ] **D5** "tcgen05.mma all formats = 128 cy at M=128 N=256" — clean claim but needs replication for ≥3 formats (FP16, FP8, FP4) — `[ref: B300_PIPE_CATALOG.md:120-130]` — `[unverified]`
- [ ] **D6** "FP4 NVFP4 K=64 = 9856 TFLOPS chip TFLOPS" — needs verification; the K=96 ULTRA path that gives 1.5× is mentioned elsewhere as inaccessible in public libs — `[ref: B300_PIPE_CATALOG.md:128]` — `[regime-narrow]`
- [ ] **D7** "P2P GEMM remote weights via NVLink: zero penalty" — based on cuBLAS L2-tiling; needs confirmation that "tiles fit in L2" explanation is what's actually happening (vs bandwidth-bound) — `[ref: B300_PIPE_CATALOG.md:178-183]` — `[regime-narrow]`

## Group E — Latency / sync / atomics

- [ ] **E1** "FFMA latency = 4 cy" — generally correct; needs to specify whether single-chain or RAW dependency — `[ref: B300_PIPE_CATALOG.md:101]` — `[regime-narrow]`
- [x] **E2** "DFMA latency = 92 cy, no ILP" — **RESOLVED 2026-04-23**: measured **63.7 cy** (matches L460, NOT L103/header's 92). Catalog L103 is WRONG; L460 is RIGHT. Confirmed not ILP-pipelined (4-chain gives same latency). — `[ref: B300_PIPE_CATALOG.md:103,460]`
- [x] **E3** "MUFU.sin latency = 24 cy, ILP throughput 8.4 cy with 3 chains" — **RESOLVED 2026-04-23 (justifications/17_mufu.md)**: SIN latency = 24.45 cy ✅; throughput at saturation (N=16) = 8.56 cy/op/warp; at N=4 = 8.83 cy. Catalog approximately right. **NEW**: EX2 unique 2× advantage (4.28 cy vs 8.7 for others) is missed in §2.10 catalog table. **NEW**: TANH supported as native MUFU.TANH at ~8 cy/op (cheaper than manual synthesis). — `[ref: B300_PIPE_CATALOG.md:106,383]`
- [ ] **E3a (NEW)** §2.10 "MUFU rate = 0.5–0.63 ops/SMSP/cy" for EX2 vs uniform 0.5 for others — likely UNIT CONFUSION (rate is per-SM, not per-SMSP — pipe is SM-shared). And EX2 actually runs at ~2× others (0.93 vs 0.46 ops/SM/cy), which catalog hints at but doesn't make load-bearing. — `[ref: B300_PIPE_CATALOG.md:383]` — `[unit-confusion + missing emphasis]`
- [ ] **E8 (NEW)** §15 "ATOMS.ADD pipe_lsu = 1.00 = 9.1 TAtoms/s chip" — needs explicit "AT CHIP SATURATION" qualifier. Per-warp single-issue = 24-62 cy/atomic depending on contention, NOT 32 atoms/SM/cy per warp. — `[ref: B300_PIPE_CATALOG.md:1014]` — `[regime-narrow]`
- [ ] **E9 (NEW)** §15 "with-return vs no-return same SASS, same rate" — TRUE for shared (ATOMS both ways). FALSE for global: no-return → REDG.E.ADD, with-return → ATOMG.E.ADD, **25× throughput delta** (32 cy vs 790 cy per atomic). — `[ref: B300_PIPE_CATALOG.md:1024]` — `[inconsistent]`
- [ ] **E10 (NEW)** §16 "atom.global.cas → ATOMG.E.CAS.STRONG.GPU" — ACTUAL emit on this rig is **STRONG.SYS** (system scope, NVLink-visible). Catalog mis-named the scope. — `[ref: B300_PIPE_CATALOG.md:1064]` — `[agent-hearsay]`
- [ ] **E11 (NEW)** Compiler optimization missing: `atomicAdd(addr, 1u)` no-return on shared compiles to `ATOMS.POPC.INC.32` (NOT REDS.ADD); broadcast contention 2.5× faster (24 vs 62 cy) via lane-combining popcount. Variable per-lane add stays at REDS.ADD. — Catalog should add this. — `[NEW finding, missing from catalog]`
- [x] **E4** "fence.sc.gpu = 274 cy" — **RESOLVED 2026-04-23**: §24 latency audit measured 281 cy (close to L115 274); §30.G fence audit measured 267 cy in single-warp/no-pending-write context. Catalog L115 is approximately correct; "544 cy" elsewhere is wrong. — `[ref: B300_PIPE_CATALOG.md:115]`
- [x] **E5** "__syncthreads at BS=512 cost 45 cy, BS=1024 cost 89 cy" — formula `12 + 2W` — **RESOLVED 2026-04-23**: empirical at this rig is **`22 + 2W` cy** (BS=512 measured 54 cy). The +10 cy is a fixed barrier-instantiation overhead the catalog formula missed. Both 45 and 12+2W=44 are wrong. — `[ref: B300_PIPE_CATALOG.md:74,75,116]`
- [x] **E6** "Atomic single-address chip-wide is 5× FASTER than per-warp atomic hotspot" — **PARTIALLY RESOLVED 2026-04-23 (justifications/30B_atomics.md)**: clean per-warp pattern (1.09× faster than 1-hotspot) and per-CTA pattern (12.4× faster) both contradict catalog L2708's "5× slowest" / "same as single" claims. Catalog row was measured on a within-warp-divergent variant. Real ranking (49.1/53.7/609 Gops/s for 1-hotspot/per-warp/per-CTA) is REVERSED from catalog. — `[ref: B300_PIPE_CATALOG.md:87,2708]`
- [ ] **E7** "All-reduce ≤1 MB floor = 21 µs, NCCL = 10 µs" — multi-GPU; needs MGFenceBench + nccl-tests verification at this rig — `[ref: B300_PIPE_CATALOG.md:157,168]` — `[unverified]`

## Group F — Power / clock / DVS

- [ ] **F1** "1005 MHz silent stuck mode" — user: "I strongly suspect this was the result of another agent or something else" — was it actually architectural or just process leftover? — `[ref: reviewed_errors L500]` — `[agent-hearsay]`
- [ ] **F2** "V² DVS scaling: V scales linearly with clock" — user: "GPU power does not scale in such a simple way, this is only a very rough 1st approximation, misleading" — should sample actual nvidia-smi voltage — `[ref: reviewed_errors L539]` — `[formula]`
- [ ] **F3** "1920 MHz is sustained boost" — user: "1920 MHz cannot be sustained for heavy tensor core workloads or even many other things" — throttling concerns — `[ref: reviewed_errors L564]` — `[regime-narrow]`
- [ ] **F4** "POPCOUNT bell curve peak at d=16" — user: "DRAM-1G W is probably just getting some L2 hits, this is confusing/misleading" — methodology has L2 amortization confound — `[ref: reviewed_errors L1369]` — `[regime-narrow]`
- [ ] **F5** "1071 W stress recipe at 1500 MHz lock + d=16 random" — user wants voltage capture — `[ref: reviewed_errors L1387]` — `[unverified]`
- [ ] **F6** "Clock-vs-power table 510→2032 MHz V scaling" — entire table approximate; user says misleading — `[ref: B300_PIPE_CATALOG.md (canonical L528-535)]` — `[formula]`

## Group G — TMA / mbarrier / cluster

- [x] **G1** "TMA cp.async.bulk issue rate = 48 cy/inst floor (size-independent)" — **RESOLVED 2026-04-23 (justifications/30_tma_sizes.md)**: 48 cy is the AMORTIZED rate (N TMAs batched onto 1 mbarrier), NOT pure single-issue. Pure single-issue is ~65 cy. Both are "size-independent" for 16B-8KB. Catalog conflates the two. — `[ref: B300_PIPE_CATALOG.md:58]`
- [x] **G2** "TMA chip-wide 29.2 TB/s" claim — **PARTIALLY RESOLVED 2026-04-23**: chip-wide measurement caps at **6.4 TB/s HBM-bound** (132 CTAs × 4KB × NT=24). The 21.9 TB/s (and 29.2 TB/s) require L2 hits (small reused dataset) — catalog wording fails to flag this. ⚠ Chip-wide TMA GB/s claim only valid for L2-resident sources, NOT as DRAM peak. — `[ref: B300_PIPE_CATALOG.md:61]`
- [x] **G2b** TMA vs LDG.E.128 max-tuned head-to-head — **RESOLVED 2026-04-23 (justifications/30_tma_vs_ldg_max_tuned.md)**: L2-hit (WS=64 MiB): **TMA wins 12%** (20.49 vs 18.25 TB/s). DRAM-cold (WS=4 GiB): **TIED at HBM SoL** (LDG 7.41=96.5%, TMA 7.32=95.4%; 1.2% noise). Catalog L2 wire 13.3 TB/s claim under-counts by 37-54%. NEW FOOTGUN: ncu `lts__t_bytes` undercounts LDG L2-hit by 2.7× (MSHR/crossbar dedup) — use `l1tex__t_bytes` for LDG / `lts__t_bytes` for TMA. — `[ref: justifications/30_tma_vs_ldg_max_tuned.md]`
- [ ] **G3** "TMA bytes per instruction not specified" — user [!fail]: "this section does not tell me what the number of bytes per TMA instruction is, so this is not very informative" — need explicit bytes/inst breakdown for each TMA test — `[ref: reviewed_errors L1008]` — `[unit-confusion]`
- [ ] **G4** "Multicast cannot be pipelined" claim — user: "this doesn't mean it cannot be pipelined - just that we are hitting maximum throughput with the amount of latency tolerance we already have" — `[ref: reviewed_errors L1029]` — `[regime-narrow]`
- [ ] **G5** "fence.proxy.async.shared::cta lowers to MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S" — needs SASS verification — `[ref: B300_PIPE_CATALOG.md:81]` — `[unverified]`
- [ ] **G6** "mbarrier RTT = 54 cy single-thread count=1" — needs replication — `[ref: B300_PIPE_CATALOG.md:73]` — `[unverified]`
- [ ] **G7** "228 KB hardware max smem per-SM, 200 KB per CTA without opt-in" — likely from cudaDeviceProp; verify on this machine — `[ref: B300_PIPE_CATALOG.md:84]` — `[unverified]`

## Group H — Methodology assumptions baked into many measurements

- [ ] **H1** Whole "dual-issue based on pipes" framing is dubious — user [!fail]: "depends how pipes are defined in ncu, to validate 'useful work' GOps/s as well, the fact this isn't clearly highlighted here as part of the methodology is worrying" — `[ref: reviewed_errors L309]` — `[agent-hearsay]`
- [ ] **H2** "Catalog mixes 1800/1920/2032 MHz across runs; cycle counts are clock-independent" claim at top of catalog — user concerns suggest cycle counts ARE often clock-dependent due to memory subsystem — `[ref: B300_PIPE_CATALOG.md:14]` — `[regime-narrow]`
- [ ] **H3** "256 cores/SM" vs "128 cores/SM" — CLAUDE.md is emphatic about 128. Catalog L30 uses "256 FLOPS/clk/SM" which assumes the FFMA-dispatches-to-2-pipes behavior. Anywhere else in catalog using "256 cores" should be flagged — `[ref: CLAUDE.md vs B300_PIPE_CATALOG.md:30]` — `[unit-confusion]`
- [ ] **H4** Wall-clock measurements affected by launch overhead for kernels <100 µs; catalog has many such tests — needs runtime ≥10 ms verification per test — `[ref: CLAUDE.md §3]` — `[regime-narrow]`
- [ ] **H5** Many tests use `ncu pipe_tensor` for tcgen05.mma — but that counter only measures legacy mma.sync (HMMA). Any tcgen05.mma claim measured via pipe_tensor is wrong — `[ref: project memory]` — `[agent-hearsay]`
- [ ] **H6** "Tests verified with ncu" claims often lack exact metric name — should always state which ncu counter — `[ref: catalog throughout]` — `[unverified]`
- [ ] **H7** "I have not manually checked most of this" warning at top is honest but means the catalog has many propagated agent-hallucinations — every claim needs SASS+ncu re-verify — `[ref: B300_PIPE_CATALOG.md:5]` — `[agent-hearsay]`

---

---

## Critical follow-up (top 10 from extended skeptical review)

These are the highest-priority items that crossed multiple groups during the deeper §10–end skeptical pass. Full lists in:
- [`justifications/_SKEPTICAL_REVIEW_10_30.md`](justifications/_SKEPTICAL_REVIEW_10_30.md) — 50 entries (Groups I-R)
- [`justifications/_SKEPTICAL_REVIEW_31_END.md`](justifications/_SKEPTICAL_REVIEW_31_END.md) — 74 entries (Groups P-X)

- [x] **CRIT1** Fence costs span 29/282/337/1679/2914/8869 cy in catalog — **RESOLVED 2026-04-23 (justifications/30G_fence.md)**: single-GPU B300 ladder is cta=8 / gl=267 (+~280 first-after-write FIXED, NOT linear) / sys=1727. V54's sys=2806 was a 2-GPU NVLink rig (1.62× higher = one extra coherence round-trip). All catalog values 1.67× too high for sys are likely multi-GPU contamination. — `[ref: B300_PIPE_CATALOG.md:2889,3068,3193,3635]`
- [ ] **CRIT2** "tcgen05.mma 'peak verified' 2.33 PFLOPS = 93% of 2.5 PF spec" — but measured at **single warp per SM** not chip-wide. If only 1 warp per 148 SMs, that's 1/148× peak — `[ref: B300_PIPE_CATALOG.md:6716]` — `[regime-narrow]`
- [ ] **CRIT3** Catalog top admits "1800/1920/2032 MHz mixed" — pipe topology agent confirmed ncu clamps to 1.91-1.92 GHz, **even with -rgc**. So ALL ncu-measured throughput numbers may be 6% under boost. — `[ref: B300_PIPE_CATALOG.md:14, justifications/01_pipe_topology.md]` — `[clock-mismatch]`
- [ ] **CRIT4** WRONG sections (HBM3e Peak L8558, TMA HBM Peak L8586) are kept in catalog without retroactive correction — readers may cite the wrong number — `[ref: B300_PIPE_CATALOG.md:8558,8586]` — `[superseded-suspect]`
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
- [ ] **CRIT6** FMNMX3 (3-input min/max) opcode claim — likely a compiler fusion not a native SASS opcode; needs cuobjdump verification — `[ref: B300_PIPE_CATALOG.md:981]` — `[agent-hearsay]`
- [ ] **CRIT7** Batch-1 MUFU latencies (§16) include range-reduction overhead never separated out; correction noted in §22 but earlier numbers still cited in §24 — `[ref: B300_PIPE_CATALOG.md:1048-1050,1974,2024,2026]` — `[superseded-suspect]`
- [ ] **CRIT8** "ENL2" SASS encoding interpretation — claimed to mean "bypass L1" and to be controlled by cudaMalloc vs cudaMallocAsync. Both claims are skeptical — needs ISA reference verification — `[ref: B300_PIPE_CATALOG.md (SASS analysis sections), reviewed_errors L1063]` — `[agent-hearsay]`
- [ ] **CRIT9** Per-stack stack-locality recipes (e.g., D2D 6.93 TB/s by separating src/dst on different stacks) — cross-stack hashing is hard to control; recipe may not generalize beyond one specific layout — `[ref: B300_PIPE_CATALOG.md:1280, reviewed_errors L794]` — `[agent-hearsay]`
- [ ] **CRIT10** L2 cap claims: catalog L45 says "20 TB/s at 256 MB" but cudaDeviceProp.l2CacheSize = 126 MB, so 256 MB is DRAM-bound; 20 TB/s contradicts HBM 7.18 ceiling — `[ref: B300_PIPE_CATALOG.md:45,1931]` — `[inconsistent]`

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
