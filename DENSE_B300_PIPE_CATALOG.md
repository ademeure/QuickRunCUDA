# DENSE B300 / Blackwell sm_103a — SM Pipe Catalog

> **Status:** Iteration in progress. This is a pruned, dense version of `B300_PIPE_CATALOG.md` that removes content known to be wrong, outdated, or low-value. Every numerical claim that survives has either a JUSTIFIED entry or a REVIEW_CHECKLIST flag.
>
> **Source:** `B300_PIPE_CATALOG.md` (19,742 lines).
> **Audit input:** `reviewed_errors_b300.md` (user's [!fail] / [!todo] callouts on related canonical doc).
> **Validation:** `JUSTIFIED_B300_PIPE_CATALOG.md` (per-section replication records).
>
> **Pruning rules applied:**
> 1. Remove anything contradicted by V52/V53/V54 empirical settlements
> 2. Remove anything user [!fail]'d in `reviewed_errors_b300.md` for the same topic
> 3. Remove "research log" repetition (sections 16–22 in catalog)
> 4. Remove formula-as-measurement claims (e.g. headline TFLOPS computed from cores × clock with no actual FLOPS counter)
> 5. Keep only LOW-LEVEL findings (drop high-level cuBLAS / GEMM ladder content unless it teaches something architectural)
> 6. Tag every surviving claim with confidence: 🟢 (3-method verified), 🟡 (1-2 method or regime caveat), 🔴 (unverified, kept because hint-bearing)
>
> **Conventions:**
> - All clock state explicit: `@1920` / `@2032` / `@1500` / `@1005` / `@boost`
> - All BW with denominator: spec=7680 GB/s, this-device=7672 GB/s
> - All TFLOPS state count convention: `2 FLOPS/FFMA` etc.
> - `⚠ FOOTGUN` for commonly mis-cited claims
>
> **Rough page budget:** ~2,500 lines (vs 19,742 source). Aggressive prune.

---

## Top of sheet — what to use this for

If you want a single number with a confidence tag: read the relevant `## §N` section, take the bold answer, ignore the rest.

If you want to verify a claim: jump to `JUSTIFIED_B300_PIPE_CATALOG.md` for the same `§N`.

If you don't trust a number: it's probably already on `REVIEW_CHECKLIST_B300.md`.

---

## §0. Spec card (verified 2026-04-23)

### Hardware (`cudaGetDeviceProperties`)

- 148 SMs × 4 SMSPs × 32 lanes = **128 FP32 cores/SM** (NOT 256). Total: 18,944 FP32 cores. 🟢
- `cudaDeviceProp.memoryBusWidth = **7680 bits**` (1 of 16 controllers fused on AC SKU; full SKU is 8192). 🟢
- `cudaDeviceProp.l2CacheSize = 126 MB`. 🟢
- `cudaDeviceProp.memoryClockRate = 3996 MHz` I/O = 7.992 Gbps/pin × 1024 bits/stack × 8 stacks = ~7672 GB/s post-ECC. 🟢
- 8 HBM3E 12-Hi stacks (3 GB/die). 🟢 (was 12 stacks in older docs — wrong.)

### Clock states (this rig, observed 2026-04-23)

| State | MHz | When |
|---|--:|---|
| `nvidia-smi -q` reported boost | 2032 | nominal datasheet boost |
| Sustained-FFMA DVFS settling point | **1942** | new finding from 00a_ffma_peak.md — pure FFMA does NOT reach 2032 nor stop at 1920 |
| `-lgc 2032` paradox pin | 1920 | per CLAUDE.md note |
| Stuck-low silent floor | 1005 | observed historically; ⚠ FOOTGUN: leftover process or driver state can leave chip here |
| `-rgc` reset behavior | back to dynamic | restores DVFS scaling |

⚠ **FOOTGUN:** No headline TFLOPS / TB/s number is meaningful without stating which clock state. Catalog historically used 1.92 GHz formulae; CLAUDE.md uses 2.032 GHz formulae. They differ by 6%.

### Compute peaks

| Op | Measured | Theoretical | Clock | Status |
|---|--:|--:|--:|---|
| FP32 FFMA scalar | **71.82 TF** | 73.6 TF (148 × 256 × 1.942) | 1942 MHz (DVFS) | ✅ replicated 100% match (00a_ffma_peak.md). ncu pipe_fma=99.5%. SASS=1024 FFMA/inner loop. |
| FP32 FFMA scalar (alt clocks) | — | 72.7 TF @ 1920 / 76.96 TF @ 2032 | catalog vs CLAUDE.md | both correct for their clock; cite the rig DVFS clock to be precise |
| FP64 DFMA | (pending) | 1.20 TF @ 2032 | — | catalog claims 0.95 TF — under-saturated? See REVIEW_CHECKLIST A5 |

### Memory peaks (catalog claims, replication pending)

| Tier | Read | Notes |
|---|--:|---|
| SMEM (`ld.volatile.shared.v4.u32`) | 35.6 TB/s claim @ 1.92 GHz | uses 1.92; if real clock is 1942 → SoL recompute |
| L1 hit (.ca, WS≤1MB) | 36.1 TB/s claim | inconsistent with 28.7 TB/s "L1" elsewhere (B2 in REVIEW_CHECKLIST) |
| L2 plateau (4-128 MB) | 22-26 TB/s claim | "was wrongly 10.2 — under-occupied launch" (regime-narrow) |
| HBM3E read | **7.18 TB/s** ncu-verified claim | denominator: spec 7680 / this-device 7672 → 93.5–95.2% (depending) |
| HBM3E write | 7.09 TB/s standard / 7.57 TB/s contested | 7.57 has disputed provenance |

### Sync/atomic key numbers (claims; replication pending)

| Op | cy | Notes |
|---|--:|---|
| `__syncwarp()` | 2.8 | claim — needs verification, likely 0 since no SASS emitted in some cases |
| `__syncthreads` BS=512 | 45 | catalog claims, inconsistent with formula `12+2W` from same doc (would be 44). |
| `mbarrier.arrive` | 8.1 | claim |
| `__threadfence_block` | 8 | wave-7 V54 confirmed at 8 cy ✓ |
| `__threadfence` (gpu) | **267 cy** sustained + 280 cy first-fence-after-write | wave-7 V54 settled. CATALOG L115 still says 274 — close but should be updated. ⚠ DENSE recommends V54 numbers. |
| `__threadfence_system` | **2806 cy = 1381 ns @ 2032** | wave-7 V54 settled. Catalog had 1750/2870/3042 spread (1.74×). ⚠ Use V54. |

### tensor cores (catalog)

| path | Best measured | Spec | clock | Notes |
|---|--:|--:|---|---|
| mma.sync m16n8k16 BF16/FP16 | 569-578 TF | 600 spec? | — | wave-6 [🟢 HIGH] per canonical |
| tcgen05.mma BF16/FP16 | 1980-2240 TF | 2.5 PF? | — | wave-6 [🟢 HIGH] |
| tcgen05.mma FP8 cuBLAS | 3984-4425 TF | 5 PF | — | wave-6 [🟢 HIGH], realistic mix |
| NVFP4 cuBLAS realistic | 11423 TF (76.2% of 15 PF) | 15 PF | — | K=96 ULTRA inaccessible from public libs |

⚠ **FOOTGUN:** ncu `pipe_tensor` does NOT measure tcgen05.mma — only legacy mma.sync/HMMA. tcgen05 must be measured via `wall-clock × cy/MMA × ops/MMA` and SASS UTCQMMA/UTCHMMA counts.

### Contention rules (catalog §3, partially verified)

1. Same pipe → cap at pipe ceiling (e.g. F2FP + LOP3 share alu → 64 combined).
2. Different pipes → mostly add cleanly. Exception: FFMA2 + UNPACK shows ~16% SMSP friction (u=1.67 vs ideal 2.00).
3. Dispatch cap = 4.00 warp-inst/SM/cy "is hard" per catalog L480 — but V52 has shown alu+fma can sum to 147% — **the framing of "hard cap" is misleading; what's hard is the per-SMSP 1 inst/cy limit**.

---

## §1. Pipe topology — verified 2026-04-23 (justifications/01_pipe_topology.md)

| Pipe | Cap (per SM/cy) | Verification | Examples |
|---|--:|---|---|
| pipe_fma (scalar dual) | **4.00 confirmed** | ✅ FFMA hits 3.88 (97%); dual FFMA+LOP3 → 3.95 (99%) | FFMA, FMUL, FADD |
| pipe_fma (packed) | 2.00 | (catalog claim, not yet rerun) | FFMA2, HFMA2, BF16-FMA |
| pipe_alu | **2.00 confirmed** | ✅ Pure LOP3 (xor.b32) saturates at 96.97% pipe_alu | LOP3, PRMT, F2FP, SHF, FMNMX, ISETP, FSETP, I2FP |
| pipe_fmaheavy | **2.00** | (component view; see footgun below) | IMAD, IMAD.X, IMAD.WIDE, IDP.4A/2A, HADD2.F32 |
| pipe_fmalite | **2.00** | (component view) | "lite" half of FFMA path |
| pipe_xu compound | **0.50 confirmed** | ✅ MUFU.SIN saturates at exactly 49.79% of pipe_xu | MUFU.SIN/COS (need range-reduction) |
| pipe_xu simple | **1.00 confirmed** | ✅ MUFU.EX2 saturates at 98.46% of pipe_xu | MUFU.EX2/RSQ/SQRT/RCP/LG2/TANH, POPC, BREV, FLO/CLZ |
| pipe_lsu | 1.00 | (catalog claim, not yet rerun) | LDG, STG, LDS, STS, LDSM, SHFL.SYNC |
| pipe_adu | ~0.5 | (catalog claim) | BAR.SYNC, MATCH.ANY |
| pipe_uniform | ~1.0 | (catalog claim) | S2UR, LDSM.sync, ACTIVEMASK, UFFMA family |
| pipe_fp64 | 0.05 | (catalog claim, but see latency inconsistency: 92 cy L103 vs 63.9 cy L460) | DFMA, DADD, DMUL — heavily throttled |
| pipe_tensor | — | (separate from tcgen05; needs kind-specific tests) | HMMA, IMMA |
| pipe_cbu | — | (low-priority; invisible in steady-state) | BRA, EXIT |

### Dispatch ceiling (confirmed)

The "4.00 warp-inst/SM/cy hard dispatch cap" claim from catalog L210 is **CONFIRMED** in the strict per-SM total sense — no test exceeded 4.00.

**But** this cap IS NOT what the popular "256 FLOPS/SM/cy" derivation depends on; the FFMA scalar peak relies on dispatching to BOTH fma sub-pipes alternately, NOT both simultaneously per instruction.

### V52 dual-issue settlement (confirmed AGAIN)

Dual FFMA + LOP3 ncu measurement:
- `pipe_alu = 96.16%`
- `pipe_fma = 48.83%`
- **sum = 144.99%** ← exceeds any single pipe's cap because pipe_alu and pipe_fma are PHYSICALLY INDEPENDENT and overlap freely.

This re-confirms wave-6 V52. The "147%" figure quoted elsewhere is well within run-to-run noise of the 145% just measured.

### ⚠ FOOTGUN — "FFMA → both fma sub-pipes simultaneously" is **FALSIFIED**

Catalog L218 says:
> "These are the ones that **uniquely use BOTH fma sub-pipes simultaneously** at 2.00 each → **4.00 warp-inst/SM/cy**."

ncu evidence for **dual FFMA + LOP3**:
- `pipe_fmalite = 93%`
- `pipe_fmaheavy = 4.5%`

**A single FFMA dispatches to ONE sub-pipe per cycle** (scheduler-chosen), not both. Solo FFMA shows both sub-pipes near 92% only because the scheduler alternates, not because each instruction goes to both.

This explains the "256 FLOPS/SM/cy" formula correctly: 4 SMSPs × (1 FFMA dispatch/cycle × 32 lanes × 2 FLOPS/FFMA) = 256 FLOPS/SM/cy. There's no per-instruction "double-issue".

The catalog's L218 wording should be: "FFMA can use EITHER sub-pipe per cycle, freely alternating across cycles, achieving aggregate 4.00 warp-inst/SM/cy when there's enough ILP."

### Key citation paths

- justifications/00a_ffma_peak.md (scalar FFMA peak, 71.82 TF replicated)
- justifications/01_pipe_topology.md (pipe placement / dispatch ceiling, V52 re-confirmed)

---

## §2. Instruction catalog — rate cheatsheet (warp-inst per SM per cycle)

> **Status:** Catalog §2 (13 sub-tables) carried over with tags. **Bold** = re-verified 2026-04-23 (this DENSE pass). Plain = catalog claim only, replication pending. ⚠ = known issue.
>
> Rate units: `r` = warp-instructions issued per SM per cycle. The aggregate dispatch ceiling is 4.00. "elements" / "ops" call out logical work-per-instruction for packed ops.

### §2.1 FP32 scalar (pipe_fma)

| PTX | SASS | r warp-inst/SM/cy | Logical |
|---|---|--:|---|
| `fma.rn.f32` | FFMA | **4.00** ✅ | 128 FFMA = **256 FP32 FLOPS** |
| `mul.rn.f32` | FMUL | 4.00 | 128 FMUL |
| `add.rn.f32` | FADD | 4.00 | 128 FADD |
| `abs/neg.f32` | FADD.FTZ (compiler) | 4.00 | 128 |

⚠ FOOTGUN: scalar FFMA at 4.00 already saturates dispatch — cannot be co-issued with anything else without losing throughput. See §1 footgun on "both sub-pipes simultaneously" wording.

### §2.2 Packed FP32/FP16/BF16 (pipe_fma, both sub-units occupied for 1 inst)

| PTX | SASS | r | Logical (FLOPS/SM/cy) |
|---|---|--:|---|
| `fma.rn.f32x2` | FFMA2 | 2.00 | 128 FMAs = 256 FLOPS-FP32 (same as scalar) |
| `fma.rn.f16x2` | HFMA2 | 2.00 | 128 FMAs-FP16 = 256 FLOPS-FP16 |
| `fma.rn.bf16x2` | HFMA2.BF16 | 2.00 | 128 FMAs-BF16 |
| `add.rn.f16x2` | HADD2 (sometimes folds to HFMA2) | 2.00 | 128 adds |
| `mul.rn.f16x2` | HMUL2 (often folds to HFMA2) | 2.00 | 128 muls |

⚠ Open: does FFMA2 + ALU give MORE total useful ops/cy than scalar FFMA? See REVIEW_CHECKLIST C9. Test in progress (justifications/22_dual_issue_ffma2_alu.md when ready).

### §2.3 Integer (pipe_fmaheavy mostly; IADD3 splits)

| PTX | SASS | r | Notes |
|---|---|--:|---|
| `mad.lo.u32` | IMAD | 2.00 fmaH | 64 IMAD |
| `mul.lo.u32` | IMAD | 2.00 fmaH | 64 IMUL |
| `mul.hi.u32` | IMAD.HI.U32 | 1.00 fmaH | 32/SM/cy — half rate |
| `dp4a.s32.s32` | IDP.4A.S8.S8 | 2.00 fmaH | 64 SASS × 4 pairs × 2 ops = 512 int8-dot ops |
| `dp2a.*` | IDP.2A | 2.00 fmaH | 64 |
| `cvt.f32.f16` | HADD2.F32 | 2.00 fmaH | 64 conversions |
| `add.u32` (single) | IADD3 / IMAD.IADD | 4.00 total (2 alu + 2 fmaH) | 128 adds (compiler splits) |
| `add.u32 a,b,c,d` (3-input) | IADD3 fused | 2.00 alu | 1 IADD3 = 2 logical adds |

### §2.4 Integer u64

| PTX | SASS | Pipe | u64 ops/SM/cy |
|---|---|---|--:|
| `add.u64` / `sub.u64` | IADD3 + IMAD.X (2 SASS) | alu + fmaH | 64 |
| `mul.lo.u64` | IMAD + IMAD.WIDE + IADD3 ×3 | mostly fmaH | ~12 |
| `mul.hi.u64` | 6+ SASS | fmaH + alu | ~5 |
| `and/or/xor.b64` | 2× LOP3 | alu | 32 |
| `shl/shr.b64/.u64` | 3 SASS | alu | ~16 |
| `min/max.u64` | ISETP×2 + SEL×2 | alu | ~16 |

### §2.5 Narrow-format CVT (F2FP family, pipe_alu)

UNPACK (narrow → f16x2/bf16x2): all variants identical.

| Narrow type | SASS | r | elements/SM/cy |
|---|---|--:|--:|
| e4m3 (FP8) | F2FP.F16.E4M3.UNPACK_B | 2.00 | 128 |
| e5m2 (FP8) | F2FP.F16.E5M2.UNPACK_B | 2.00 | 128 |
| e2m1 (FP4) | F2FP.F16.E2M1.UNPACK_B | 2.00 | 128 |
| e2m3 (FP6) | F2FP.F16.E2M3.UNPACK_B | 2.00 | 128 |
| e3m2 (FP6) | F2FP.F16.E3M2.UNPACK_B | 2.00 | 128 |
| ue8m0 → bf16 | F2FP.BF16.E8.UNPACK_B | 2.00 | 128 |

⚠ With LOP3 zero-ext feedback (1-per-iter), effective rate halves to 1.00 = 64 elements/SM/cy. The peak only holds without LOP3 pollution.

PACK (wide → narrow): rates drop because of LOP3/PRMT tax. See catalog §2.5 for full table.

### §2.6 Other CVTs

| PTX | SASS | Pipe | r |
|---|---|---|--:|
| `cvt.rn.f16.f32` (pack) | F2FP.F16.F32.PACK + PRMT | alu | 1.00 (PRMT tax) |
| `cvt.rn.bf16.f32` | F2FP.BF16.F32.PACK + PRMT | alu | 1.00 |
| `cvt.f32.f16` | HADD2.F32 | fmaH | 2.00 |
| `cvt.rn.f32.s32/u32` | I2FP.F32.* | alu | 2.00 |
| `cvt.rn.f32.s64` | I2F.S64 | xu | **0.04 — super slow** |
| `cvt.rni.s32.f32` | F2I.NTZ | xu | 0.5 |
| `cvt.rni.sat.u8.f32` | F2IP.U8.F32.NTZ | alu | 2.00 (!) |
| `cvt.rni.sat.s8.f32` | F2I.S8.NTZ | xu | 0.5 |
| `cvt.sat.u8.s32` | I2I.U8.S32.SAT | alu | 2.00 |

### §2.7 Bitwise / shift / permute (pipe_alu)

| PTX | SASS | r |
|---|---|--:|
| `xor/and/or/not/lop3.b32` | LOP3.LUT | **2.00 ✅** |
| `shl/shr (plain)` | SHF.* | 2.00 |
| `shf.l/r.wrap.b32` | SHF.L/R.W.U32 | 2.00 |
| `prmt.b32` | PRMT | 2.00 |
| `bfi.b32` | LOP3.LUT (collapses) | 2.00 |
| `bfe.u32` | SHF.R.U32.HI + SGXT.U32 (2 SASS) | 1.00 |
| `brev.b32` | BREV | xu, 0.5 |
| `popc.b32` | POPC | xu, 0.5 |
| `clz.b32` / `bfind` | FLO.U32 (+ IADD3) | xu+alu, 0.5 |

### §2.8 Compare / select / predicate (pipe_alu)

| PTX | SASS | r |
|---|---|--:|
| `setp.*.u32/s32` | ISETP.* | 2.00 |
| `setp.*.f32` | FSETP.* | 2.00 |
| `selp.b32` | SEL | 2.00 |
| setp+selp (2 SASS) | ISETP+SEL | 1.00 |
| `vote.sync.ballot.b32` | ISETP+VOTE.ANY | 1.00 |

### §2.9 MIN / MAX — surprisingly all on pipe_alu (NOT fma)

| PTX | SASS | r |
|---|---|--:|
| `min/max.f32` | FMNMX | 2.00 |
| `min/max.NaN.f32` | FMNMX.NAN | 2.00 |
| `min/max.f16x2` | HMNMX2 | 2.00 (= 128 FP16 mins) |
| `min/max.bf16x2` | HMNMX2.BF16 | 2.00 |
| `min/max.s32` | VIMNMX3 (compiler folds 2 mins) | 2.00 (= 128 int mins) |
| `min/max.u64` | ISETP×2 + SEL×2 | 0.5 (~16 u64 min) |
| `abs.s32` / `neg.s32` / `abs.f32` | folds to IADD3/FADD/LOP3 | 2.00+ |
| `copysign.f32` | LOP3.LUT | 2.00 |

⚠ The "FMNMX3" 3-input fused min in catalog L981 may be compiler fusion not a native opcode. See REVIEW_CHECKLIST CRIT6.

### §2.10 Transcendentals (pipe_xu) — verified 2026-04-23

| PTX | SASS | r | Notes |
|---|---|--:|---|
| `ex2.approx.f32` | MUFU.EX2 | **0.50–0.63 ✅ simple** | 16-20 SASS/SM/cy. Saturates pipe_xu at 98.5% solo. |
| `rsqrt.approx.f32` | MUFU.RSQ | 0.5 | (catalog claim, simple) |
| `sqrt.approx.f32` | MUFU.SQRT | 0.5 | simple |
| `rcp.approx.f32` | MUFU.RCP | 0.5 | simple |
| `sin.approx.f32` | MUFU.SIN + FMUL range-reduction | **0.5 (compound) ✅** | Saturates pipe_xu at 49.8% (half the simple rate); also drives pipe_fma to 12.5% (range-reduction FFMA). |
| `cos.approx.f32` | MUFU.COS | 0.5 (compound) | likely same as sin |
| `lg2.approx.f32` | MUFU.LG2 | 0.5 | simple per catalog |
| `tanh.approx.f32` | MUFU.TANH | 0.5 | simple per catalog |

⚠ Catalog §16 has older MUFU latencies (RSQ=40 cy, RCP=42 cy) that include range-reduction overhead from author-added scaffolding. Prefer §23 clean sweep numbers (RSQ=18 cy ftz). See REVIEW_CHECKLIST CRIT7.

### §2.11 Warp / sync / barrier ops (key entries)

| PTX | SASS | Pipe | r | Notes |
|---|---|---|--:|---|
| `shfl.sync.{bfly,idx,up,down}` | SHFL.* | lsu | 1.00 | 32 SASS/SM/cy |
| `vote.ballot` | VOTE.ANY + ISETP | alu | 2.00 combined | |
| `vote.{any,all,uni}` | VOTE.* | alu | 2.00 | |
| `activemask` | uniform-pipe op | uniform | ~1.2 | (DCE'd in some tests) |
| `match.any.sync.b32` | MATCH.ANY | adu | **0.5 peak — VERY SLOW** | catalog L85: 375 cy = 20× other warp ops. Avoid. |
| `bar.sync 0` | BAR.SYNC.DEFER | adu | ~0.36 | thread-waiting dominates |
| `bar.arrive` | BAR.ARV | adu | ~0.47 | |
| `bar.red.popc.u32` | BAR.RED.POPC.DEFER | adu+alu | ~0.37 | |
| `redux.sync.min.u32` | CREDUX.MIN+IMAD (2 SASS) | alu+fmaH | **1.92 PTX-op/SM/cy** | each pipe at 1.92/2.00 |
| `redux.sync.add.u32` | REDUX.SUM+IMAD | adu | **0.50 — 4× slower than min/max** | |
| `redux.sync.{or,and,xor}` | REDUX.* | adu | 0.50 (same as add) | |
| `membar.cta` | MEMBAR.SC.CTA | lsu | 0.83 | scoped fence on lsu |
| `membar.gl` | MEMBAR.SC.GPU + ERRBAR | adu+lsu | extremely slow | see §30.G replication for cy/op |
| `ldmatrix.sync.x1.b16` | LDSM (1 quad) | uniform+lsu | ~1.0 | |
| `ldmatrix.sync.x4.b16` | LDSM (4 quads) | uniform+lsu | 0.25 | quarter rate |
| `atom.shared.add.u32` | ATOMS.POPC.INC.32 | lsu | 0.84 | |
| `atom.global.*` | ATOMG.* | lsu | bandwidth-bound | |
| `s2r %clock/%clock_hi` | S2R SR_CLOCKLO/HI | adu | 0.5 | |

### §2.12 Memory

| PTX | SASS | Pipe | Notes |
|---|---|---|---|
| `ld.global.u32` | LDG.E | lsu | DRAM-bound in practice, ~1 inst/SM/cy issue |
| `st.global.u32` | STG.E | lsu | DRAM-bottleneck, not pipe |
| `ld.shared.u32` | LDS | lsu | ~1.0 issue, bank-conflict-sensitive |
| `st.shared.u32` | STS | lsu | 1.00 saturating |

### §2.13 FP64 — severely throttled

| PTX | SASS | Pipe | r |
|---|---|---|--:|
| `fma.rn.f64` | DFMA | fp64 | **0.05 = 1.6 DFMA/SM/cy = ~475 GFLOPS-FMA chip-wide** |
| `add.rn.f64` | DADD | fp64 | 0.05 |
| `mul.rn.f64` | DMUL | fp64 | 0.05 |

DFMA is **NOT pipelined** per catalog L460 — 4 chains give zero ILP benefit (63.9 cy/op each). FFMA + ALU co-issue freely during the 64 cy window. ⚠ Catalog has DFMA latency = 92 cy (L103) AND 63.9 cy (L460) — inconsistent; likely 63.9 is the corrected number from a later test.

⚠ B300 FP64 peak = ~1.2 TFLOPS (CLAUDE.md). Catalog says 0.95 TFLOPS measured — under-saturated; needs replication. (REVIEW_CHECKLIST A5)

---

## §3. Contention rules (catalog §3, L472)

1. **Same pipe** → cap at pipe ceiling.
   - F2FP + LOP3 (both alu) → 64 combined. ✓
   - IMAD + FFMA scalar (compete for fmaH) → reduces FFMA peak.
2. **Different pipes** → mostly add cleanly, with caveats:
   - **FFMA2 + UNPACK** (fma + alu): u=1.67 (106/127 combined) — ~16% SMSP friction specific to F2FP. Not present for PRMT+FFMA2 (u=1.95).
3. **Dispatch cap = 4.00 sm_inst/SM/cy** ✓ confirmed (FFMA scalar hits 3.88, mixed hits 3.95). However, V52 alu+fma sum=145% means cross-pipe accounting CAN exceed 100% per pipe — this is normal, not a violation of the 4.00 cap.
4. **HFMA2 + FFMA scalar** can co-exist but compete for H+L slots (~2.0 total warp-inst/SM/cy). Confirmed catalog reasoning matches V52 mechanism.

---

## §4. Rate cheatsheet — key entries (warp-inst/SM/cy → SASS-inst/SM/cy)

| Op | SASS/SM/cy | Logical |
|---|--:|---|
| Scalar FFMA | **128** ✓ | 256 FLOPS, dual-pipe heavy+lite |
| FFMA2/HFMA2/BF16-FMA | 64 | 128 FMAs |
| u32 ADD (IADD3 fusion) | 128 | 1 IADD3 = 2 adds |
| LOP3 / PRMT / SHF / FMNMX / HMNMX2 / VIMNMX3 | 64 | all pipe_alu, share |
| F2FP UNPACK (all formats) | 64 | 128 elements (×2 ops) |
| F2FP PACK | 32-64 | depends on feedback path |
| BFE | 32 | 2 SASS per PTX op |
| SHFL.SYNC.* | 32 | pipe_lsu |
| LDS/STS/LDG/STG | ~32 issue | DRAM-bound if streaming |
| MUFU (EX2/RSQ/SIN/...) | ~16 | pipe_xu, compound |
| F2I, POPC, BREV, FLO | 16 | pipe_xu |
| BAR.SYNC | ~12 | pipe_adu |
| MATCH.ANY | serial — VERY SLOW | pipe_adu |
| **FP64 FMA (DFMA)** | **1.6** | pipe_fp64, throttled |

---

## §5. Narrow-format throughput

At 128 elements/SM/cy × 148 SMs × 1.92 GHz = **36.4 Telements/s** for each UNPACK variant (FP4/FP6/FP8/UE8M0 → f16/bf16). Same number for all because they share the one ALU pipe.

For FP4 specifically: both UNPACK (`cvt.rn.f16x2.e2m1x2`) and PACK (`cvt.rn.satfinite.e2m1x2.f16x2`) live on the same 64 warp-inst/SM/cy ceiling as FP8/FP6 — **FP4 is NOT faster or slower per SASS instruction** than FP8 on B300's ALU pipe.

---

## §6. Uniform datapath (pipe_uniform)

Per-SMSP scalar unit operating on uniform registers (URx). Compiler uses it automatically for loop counters, kernel-arg propagation, warp-invariant scalars.

**Measured** (catalog claim): pipe_uniform hits ~1.0 warp-inst/SM/cy for ACTIVEMASK and LDSM. Does NOT contend with pipe_alu / pipe_fma — uniform ops issue in parallel.

**Blackwell adds** full uniform FP32 datapath (UFFMA, UFADD, UFMUL, etc.) — but ⚠ nvcc 13.0 and 13.2 do NOT emit these despite being in ISA. Either spec or aspirational. (REVIEW_CHECKLIST CRIT3 / R3)

---

## §7. ADU (pipe_adu)

Hosts slow warp-wide synchronization and status-register operations. Peak issue rate ~0.4-0.5 warp-inst/SM/cy for simple cases; wall-clock dominated by cross-thread waiting, not pipe throughput. No contention with ALU/FMA.

Key opcodes: BAR/BAR.SYNC/BAR.ARV/BAR.RED, CGA barriers, WARPSYNC/BSYNC/BSSY/BREAK/NANOSLEEP/YIELD, MATCH.ANY/MATCH.ALL, REDUX.SUM/OR/AND/XOR, MEMBAR.SC.GPU/SYS partial, S2R clock/timer.

---

## §8 + §9 — SASS opcode → pipe full classification

Catalog §8 lists every SASS opcode with pipe assignment. Catalog §9 lists PTX → SASS mapping for every ISA category. Both are lengthy reference tables; preserved verbatim in `B300_PIPE_CATALOG.md` L554-L897.

DENSE pruning notes:
- **Verified pipe placements** (this audit): pipe_fma (FFMA), pipe_alu (LOP3), pipe_xu compound (MUFU.SIN), pipe_xu simple (MUFU.EX2). All match catalog §8.
- **Unverified at this rig** (catalog claims): tensor pipes (HMMA, IMMA, QMMA, OMMA, DMMA, UTC*MMA), texture/surface (TEX, TLD, SULD, SUST), uniform pipe FP variants (UFFMA, UFADD, UFMUL), ADU detail, CBU detail.
- **Likely-correct propagated from H100/B200 docs** (Hopper-style opcodes still present): UBLKCP (TMA), UTMALDG/STG family.

For the full opcode table, refer to B300_PIPE_CATALOG.md L554-L897 directly. Most entries are inferred from opcode family rules (uniform-prefix → uniform pipe, etc.) — treat as best-effort but not all empirically tested.

---

## §10. L1/L2/HBM bandwidth ladder

(Pending — replication agent in progress; results will land in justifications/00b_mem_hierarchy.md.)

Catalog claims (preserved):
| Tier | Read TB/s |
|---|--:|
| smem (`ld.volatile.shared.v4.u32`) | 35.6 (98% theoretical at 1.92 GHz) |
| L1 hit (.ca, WS≤1MB) | 36.1 |
| L2 plateau (4-128 MB) | 22-26 |
| L2 → DRAM cliff at 126 MB | 8.2 |
| HBM3E (WS≥1GB) | 7.18 (ncu-verified) |

⚠ Cliff at 126 MB matches `cudaDeviceProp.l2CacheSize = 132,644,864 B` exactly. ✓

⚠ The L45 claim "→ 11 TB/s at 256 MB" doesn't make sense if L2 is 126 MB and HBM is 7.18 TB/s — 256 MB should be DRAM-bound at ~7 TB/s, not 11. (REVIEW_CHECKLIST CRIT10) Probably L2 amortization at the boundary.

---

## §11. Atomics — (pending replication §15+)

Catalog claims preserved verbatim in B300_PIPE_CATALOG.md L1008+, with skeptical-review entries in `_SKEPTICAL_REVIEW_10_30.md` Group K (8 entries flagged).

Key headlines (suspect):
- "Atomic single-address chip-wide is 5× FASTER than per-warp atomic hotspot" (catalog L87, mechanism is cache-line combining; verify)
- "CAS is unconditionally half-rate vs atom.add" (REVIEW_CHECKLIST K2)
- "Hot-spot same-addr warp-coalesce 12× slower than unique" (REVIEW_CHECKLIST K7)

---

## §12. Memory fence costs (pending §30.G replication)

Catalog has multiple inconsistent values for the same operation. Wave-7 V54 settled at:
- `__threadfence_block` (cta) = 8 cy / 3.9 ns
- `__threadfence` (gl) = 267 cy / 131.5 ns (+280 cy first-fence-after-write L2 drain)
- `__threadfence_system` (sys) = 2806 cy / 1381 ns

DENSE recommendation: USE V54 numbers; ignore catalog's L2889/L2922/L3068/L3193/L3632/L3635 spread until reconciled.

---

(More sections added as agents finish replication.)
