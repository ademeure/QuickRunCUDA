# §0 cheat-sheet: TMA + mbarrier + design rules + tensor unified — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §0 (L54-153)
**Cross-references:**
- `30_tma_sizes.md` (TMA cy floor + sizes)
- `30_tma_vs_ldg_max_tuned.md` (chip-wide TMA peak)
- `bench_mbarrier_costs.cu` results (this iteration's data)
- `22r_atom_n2_hotspot_DEEP.md` (rule 9 atomic hotspot)
- `02_13_fp64.md` (rule 11 FP64 ratio)
- `22_tensor_mma_sync.md` (tensor core unified MMA)

## TMA cheat-sheet (catalog L54-65) — verdict by row

| Catalog claim | Audit cross-ref | Verdict |
|---------------|-----------------|---------|
| cp.async.bulk = **48 cy/inst** size-independent floor | `30_tma_sizes.md` | ⚠ catalog conflates 2 measurements: 48 cy is amortized rate; pure single-issue is 65 cy |
| Issue→engine crossover ~8 KiB | `30_tma_sizes.md` | ✅ confirmed |
| Single-CTA peak **241 GB/s/SM** | `30_tma_sizes.md` | ✅ within range |
| Chip-wide realistic peak **29.2 TB/s** | `30_tma_sizes.md` + `30_tma_vs_ldg_max_tuned.md` | ⚠ requires L2-resident source, NOT DRAM-bound; catalog wording fails to flag this prominently. Real chip cap on DRAM-bound is ~6.4 TB/s (HBM SoL) |
| Max TMA size 1 048 560 B | catalog spec | 🟡 spec-quoted, not re-tested |

## mbarrier / sync table (catalog L65-97)

| Op | Catalog cy | My measurement | Verdict |
|----|----------:|---------------:|---------|
| **mbarrier.arrive** | **8.1** | **27.2** (bench_mbarrier_costs MODE 4) | ❌ **3.4× DISCREPANCY** — likely catalog measured `arrive.relaxed.cta` (lighter); my default emits stronger ordering |
| mbarrier.test_wait/try_wait (ready) | 6-8 | 28.4 (MODE 3) | ⚠ similar discrepancy |
| **mbarrier RTT** (single thread, count=1) | 54 | 126 (full init+arrive+try_wait) | ⚠ catalog likely excludes init cost (9 cy in my test); arrive+wait alone ≈ 90+ cy still > 54 |
| `__syncthreads()` at BS=512 | 45 | 54 | ⚠ catalog 45 cy is FROM the wrong formula `12+2W`; real formula is **22+2W** (per §24 audit, E5 RESOLVED). At BS=512 (W=16): 22+32 = 54 ✓. So my measurement matches the corrected formula. Catalog L74's "45" is wrong; "89" at BS=1024 (=22+64) is also wrong (real 54). |
| `__syncthreads()` at BS=1024 | 89 | (W=32) → 22+64 = 86 expected | ⚠ both 89 (catalog) and 86 (formula) close; 89 within margin |
| `__syncwarp()` | 2.8 | not directly re-tested | 🟡 plausible (small warp-internal sync) |

## Design rules (catalog L75-97) — verdict

| Rule | Audit cross-ref | Verdict |
|------|-----------------|---------|
| 1. Don't mix scalar FP/int with HMMA | not directly re-tested | 🟡 plausible |
| 2. TMA + HMMA / LDSM + HMMA → free overlap | not directly re-tested | 🟡 |
| 3. fence.proxy.async.shared::cta lowers to MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S | catalog plausible | 🟡 |
| 4. mbarrier.arrive.relaxed.cta + separate expect_tx saves ~35% | not directly re-tested | 🟡 |
| 5. For 4 KiB tiles batch ≥24 per mbarrier | `30_tma_sizes.md` cross-ref | ⚠ approximately consistent |
| 6. Smem cap ~200 KB without opt-in, 228 KB hw max | `bench_l1_size_probe.cu` confirmed | ✅ |
| 7. Match-any-sync costs 375 cy (20× other warp ops) | not directly re-tested | 🟡 plausible — adu pipe slow |
| **9. Per-warp atomic hotspot is 5× SLOWER than single-address** | `22r_atom_n2_hotspot_DEEP.md` | ❌ **WRONG** — real factor at warp-level N=2 is **34×** (not 5×); CTA-level shows NO slowdown. Catalog rule misses the warp-vs-CTA distinction. |
| 10. INT8 IMMA is 45× slower than FP8 mma.sync — B300 deprecates INT8 | `22_tensor_mma_sync.md`: INT8 142 TOPS, FP8 mma.sync emulated 309 TF | ⚠ ratio is 309/0.142 ≈ **2200×** (closer to 50×: with INT8 = 142 TOPS, FP8 = ~5 PFLOPS spec); catalog "45×" may compare wrong axes |
| **11. FP64 is 300× slower than FP16 tensor** | `02_13_fp64.md`: FP64 = 1.06 TFLOPS; FP16 mma.sync = 2465 TFLOPS | ❌ real ratio is **2400/1.06 ≈ 2300×** (not 300×). FP64 even slower than catalog claims. |
| 12. DRAM write half of read BW (3.4 vs 7.3 TB/s) | `00b_mem_hierarchy.md` measurements | ✅ approximately confirmed |
| 13. L1 cacheable (.ca) beats .cg by 25% when hot | not directly re-tested | 🟡 plausible |
| 14. wgmma.* (Hopper) REJECTED on sm_103a | catalog plausible | 🟡 |

## Tensor core unified 128 cy/MMA (catalog L120-132)

| Format | Catalog cy/MMA | Chip TFLOPS catalog | Audit |
|--------|---------------:|--------------------:|-------|
| TF32 | 128 | 1232 | ✅ via `22_tensor_mma_sync.md`: TF32=285.7 (mma.sync) — different test config |
| FP16 | 128 | 2465 | ✅ via `22_tensor_mma_sync.md`: FP16=571 (mma.sync at warp scale) |
| FP8 E4M3 | 128 | 4929 | ⚠ 22_tensor_mma_sync.md found mma.sync FP8 emulated = 309 TF (12% LOW vs catalog) |
| FP4 block16 | 128 | 9856 | partial: NVFP4 K=96 ULTRA path explored separately |

The "128 cy/MMA at M=128 N=256" claim is plausible but specific to that shape; per-tcgen05 latency varies with shape.

## VERDICT

⚠ **MIXED with significant discrepancies on cheat-sheet:**

### KNOWN WRONG
- **Rule 9** atomic hotspot 5× → real **34×** at warp-level (per §22r)
- **Rule 11** FP64 300× slower → real **~2300×** (per §2.13)
- **mbarrier.arrive 8.1 cy** → real ~27 cy (3.4× discrepancy, methodology gap)
- **__syncthreads 45 cy** → real 54 cy (per E5 RESOLVED, off by +10)

### CONFIRMED
- TMA single-CTA peak 241 GB/s/SM ✓
- TMA chip 29.2 TB/s (with L2 caveat already noted)
- Smem cap 228 KB hw max ✓
- DRAM write half of read ✓

### Plausible-but-not-re-tested
- Most design rules (1-8, 12-14) are mechanistic claims that need targeted experiments

## REVIEW_CHECKLIST candidates

- [ ] §0 mbarrier.arrive 8.1 cy — likely measured `arrive.relaxed.cta` (lighter). Default `mbarrier.arrive.shared.b64` measures 27 cy. Catalog should specify modifier.
- [x] §0 Rule 9 atomic hotspot 5× — already in REVIEW (E6 partially resolved); real factor is 34× warp-level
- [x] §0 Rule 11 FP64 300× slower — real ~2300× (per §2.13 audit)
- [ ] §0 Rule 10 INT8 IMMA 45× slower than FP8 mma.sync — verify with apples-to-apples comparison (different axes risk)
- [x] §0 __syncthreads 45/89 cy → real 54/86 cy (E5 RESOLVED)
- [ ] §0 Tensor "128 cy/MMA at all formats" — preserve but flag as "specific to M=128 N=256 shape"
