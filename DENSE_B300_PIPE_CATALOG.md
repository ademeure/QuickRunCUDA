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

## §2. Instruction catalog (selected, post-prune)

(Pending — populated as agents finish replication. Catalog §2 has 13 sub-tables which are mostly correct but have not all been re-verified post-V52.)

---

(More sections added as the JUSTIFIED catalog progresses.)
