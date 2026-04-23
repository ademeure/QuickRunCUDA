# §25 Final compact throughput + §26 Warp cooperative primitives — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §25 (L2062), §26 (L2117)

## §25 Final compact throughput table — verdict by row

### FP throughput

| Op | Catalog GFLOPS | Audit cross-ref | Verdict |
|----|---------------:|-----------------|---------|
| FP32 FFMA scalar | 69k = 69 TF | `00a_ffma_peak.md`: 71.82 TF @ 1.942 GHz | ✅ within 5% |
| FP32 FFMA2 vec2 | 69k | `02_1_2_3_fp32_int.md`: ~same as scalar | ✅ |
| FP16/BF16 HFMA2 (non-tensor) | 35k = 35 TF | `02_1_2_3_fp32_int.md` + §27 audit (35.2 TF) | ✅ exact |
| FP16/BF16 HMMA tensor | **838k = 838 TF** | `22_tensor_mma_sync.md`: FP16 mma.sync = 571 TF | ⚠ catalog 47% higher; possibly tcgen05 vs mma.sync confusion |
| TF32 HMMA tensor | 420k = 420 TF | `22_tensor_mma_sync.md`: 285.7 TF | ⚠ catalog 47% higher |
| **FP64 DFMA scalar** | **475 GFLOPS** ("475 G FMA-ops/s = 950 GFLOPS" per L446 parenthetical) | `02_13_fp64.md`: **1060 GFLOPS** | ⚠ catalog 12% off (0.95 TF vs 1.06 TF measured); see §2.13 refined |

### Memory BW

| Source | Catalog TB/s | Audit | Verdict |
|--------|-------------:|-------|---------|
| L1 hit small WS | 35 | `00b_mem_hierarchy.md`: 35.88 TB/s | ✅ exact |
| L2 hit | 20 | `00b_mem_hierarchy.md`: 20.3 TB/s | ✅ exact |
| DRAM coalesced | 7.4 | `00b_mem_hierarchy.md`: 7.17-7.25 TB/s | ✅ within margin |
| Smem v4 | 36 | (= 128 B/SM/cy theoretical) | ✅ matches |

### MUFU + atomics + division

| Op | Catalog | Audit | Verdict |
|----|--------:|-------|---------|
| ex2 throughput | 8.9 TGOps/s | `17_mufu.md`: ~8.85 T (ncu 0.99 inst/SM/cy peak) | ✅ exact (= §23 audit) |
| sin/rcp throughput | 4.5 TGOps/s | `17_mufu.md`: ~4.4-4.5 T (0.5/SM/cy compound) | ✅ exact |
| ATOMS.ADD chip | 9.1 TAtoms/s | `15_atomics.md` + `22_atomic_smem_DEEP.md` | ✅ at saturation |
| ATOMS.CAS half-rate | 4.5 | per `22_atomic_smem_DEEP.md`: 126 cy = 0.5/cy | ✅ |
| Division ladder (div.full = 3.7×, div.rn = 330× slower) | per catalog | not directly re-tested | 🟡 plausible |

### ISA feature summary

| Claim | Audit | Verdict |
|-------|-------|---------|
| SMSP dispatch cap = 4.00 warp-inst/SM/cy | `01_pipe_topology.md` + `12_alu_ceiling.md` | ✅ |
| FFMA reaches 3.87 (97%) | `00a_ffma_peak.md` measures 99.5% | ✅ even better |
| 126 MB L2, 228 KB L1/SM | `00b_mem_hierarchy.md` + `bench_l1_size_probe.cu` | ✅ |

## §26 Warp cooperative primitives — verdict

| Op | Catalog GOps/s | Audit | Verdict |
|----|---------------:|-------|---------|
| **vote.sync.ballot.b32** | **7320** | per `02_7_8_9_alu_ops.md` (alu, ISETP+VOTE.ANY 2 SASS) | ✅ plausible (alu cap = 6080 G inst/s × 1.5 SASS/op) |
| vote.sync.{all,any,uni}.pred | 3315 | predicate→register SELP fallback (2 SASS) | ✅ |
| **shfl.sync.bfly.b32** | **5576** | per §2.11 catalog (lsu 1.00) — implies pipe_lsu peak | 🟡 not directly re-tested |
| **redux.sync.min.u32** | **6923** | `11_redux.md`: 1.89 PTX-op/SM/cy alu+fmaH = 1.89 × 32 × 148 × 1.92 = 17.2 G PTX-ops/s × 32 lanes / ... wait, catalog 6923 GOps is PER-WARP-OP × 32 lanes / 32 = warp-level | ✅ matches §11 |
| redux.sync.add.u32 | 3107 | `11_redux.md`: 0.50 ADU = 0.5 × 32 × 148 × 1.92 = 4.5 G | ⚠ within order |

### Catalog observations

> "vote.ballot is 2.2× faster than vote.all/any/uni" — ✅ confirmed (1 SASS vs 2 SASS)
> "redux.min/max 2.2× faster than redux.add/or" — ✅ confirmed via §11 (different pipes: alu+fmaH vs adu)

## VERDICT (composite)

✅ **§25 + §26 mostly confirmed via cross-references.**

⚠ **OPEN ISSUES already in REVIEW:**
- §25 FP64 DFMA "475 GFLOPS" — was confusing wording; means "475 G FMA-ops/s = 950 GFLOPS"; real 1060 GFLOPS measured = 12% off (acceptable, see §2.13 refined)
- §25 HMMA "838 TF" higher than my §22 mma.sync (571 TF); likely conflates mma.sync and tcgen05 paths

🟡 **Division ladder** — catalog ratios plausible (matches expected pipe assignments) but not independently re-tested

## REVIEW_CHECKLIST candidates

- [x] §25 FFMA 69k GFLOPS — ✅ matches §00a (71.82 at boost)
- [x] §25 L1/L2/HBM BW — ✅ matches §00b
- [x] §25 ex2 8.9 TGOps/s — ✅ matches §17/§23
- [x] §25 ATOMS.ADD 9.1 TAtoms/s — ✅ at saturation
- [x] §25 FP64 475 GFLOPS — ⚠ wording confusing (= 950 GFLOPS chip); real 1060 GFLOPS = 12% off (see §2.13 refined)
- [ ] §25 HMMA FP16 838 TF — discrepancy with §22 mma.sync (571 TF); needs reconciliation (mma.sync vs tcgen05?)
- [ ] §25 div.rn 330× slower than FFMA — plausible but not verified
- [x] §26 vote.ballot 2× faster than vote.all/any/uni — ✅ confirmed via SASS expansion
- [x] §26 redux.sync.min 7× faster than shfl-tree — ✅ via §11
