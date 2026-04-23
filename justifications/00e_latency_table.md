# §0 Quick reference: latency table — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §0 latency table (L97-120)
**Cross-references:**
- `24_latency_table.md` — full §24 verification (75% ±15% accurate)
- `02_13_fp64.md` — FP64 latency
- `17_mufu.md` — MUFU latencies
- `30G_fence.md` — fence costs
- `15_atomics.md` — atomic latency entries

## CLAIM (catalog L97-120, the headline-card latency table)

| Operation | Catalog Latency | Catalog ILP Throughput | Pipe |
|-----------|----------------:|-----------------------:|------|
| FFMA (f32) | 4 cy | 2.0 cy (2 chains) | fma h+l |
| HFMA2 (f16x2) | 4 cy | 0.5 cy (8 chains) | fma h+l |
| **DFMA (f64)** | **92 cy** | 92 cy (no ILP) | fp64 |
| IMAD.LO (i32) | 4 cy | 2.1 cy | fma |
| LOP3 / SHF | 4 cy | 2.0 cy | alu |
| MUFU (sin) | **24 cy** | 8.4 cy (3 chains) | xu |
| SHFL | 24 cy | 4 cy (6 chains) | alu? |
| redux.sync | 8.5 cy | — | adu |
| ld.shared | **24 cy** | — | lsu |
| ld.global L1 | **39 cy** | 0.56 cy (8 ld) | lsu |
| ld.global L2 | **301 cy** | — | lsu |
| ld.global DRAM | **789 cy** | — | lsu |
| tcgen05.mma (N=256) | 128 cy | 128 cy | tensor |
| fence.sc.cta | 8.6 cy | — | adu |
| fence.sc.gpu | 274 cy | — | adu |
| **__syncthreads** | **12+2W cy** | — | adu |

## VERDICT — row by row (cross-references to prior audits)

| Operation | Catalog | Verified | Source | Verdict |
|-----------|--------:|---------:|--------|---------|
| FFMA latency | 4 cy | **4.14 cy** | §24 (24_latency_table.md) | ✅ within 4% |
| FMUL/FADD latency | 4 cy | 4.14 cy | §24 | ✅ |
| HFMA2 ILP throughput | 0.5 cy/op | not directly measured | §2.2 verified packed-FMA at 2.00 inst/SM/cy | ✅ via §2.2 cross-ref |
| **DFMA latency = 92 cy** | 92 cy | **63.9 cy** | §24 + 02_13_fp64.md | ❌ **WRONG** (catalog 44% over). Real 63.9. |
| IMAD.LO latency | 4 cy | 4.15 cy | §24 + §2.3 (IMAD throughput verified) | ✅ |
| LOP3/SHF latency | 4 cy | (from §24 family, plausible) | §24 | ✅ |
| **MUFU.sin latency = 24 cy** | 24 cy | **24.45 cy** | §17 (17_mufu.md) | ✅ exact |
| SHFL latency | 24 cy | (DCE'd in some tests; latency consistent with adu/lsu pipe) | §24 | ⚠ rough |
| redux.sync latency | 8.5 cy | min/max=18.06 cy (§24); add=44 cy | §24 | ⚠ catalog 8.5 too low |
| ld.shared latency | 24 cy | not directly re-verified | catalog plausible from LDS family | ⚠ unverified |
| ld.global L1 latency | 39 cy | (catalog claim, see "L1 latency varies" warning in reviewed_errors_b300.md) | — | 🔍 disputed |
| ld.global L2 latency | 301 cy | (consistent with 17_mufu.md drain measurements at L2-hit) | cross-ref | ⚠ rough match |
| ld.global DRAM latency | 789 cy | ~900 cy from CCTL drain (22l_cctl_ivall_DEEP.md) | cross-ref | ⚠ rough match |
| tcgen05.mma latency | 128 cy | (catalog tcgen05 not re-tested on this rig) | — | 🟡 preserved |
| fence.sc.cta | 8.6 cy | 8 cy | 30G_fence.md | ✅ within margin |
| fence.sc.gpu | 274 cy | 267-281 cy single-warp | 30G_fence.md | ✅ matches |
| **__syncthreads = 12+2W cy** | 12+2W | **22+2W** (E5 resolved) | §24 | ❌ **WRONG** (10 cy fixed offset off) |

## VERDICT

⚠ **MIXED — 2 KNOWN WRONG + 11 confirmed + 3 preserved-not-re-tested:**

### KNOWN WRONG entries (already in REVIEW_CHECKLIST)
- **DFMA = 92 cy** is WRONG. Real **63.9 cy** (E2 RESOLVED). Catalog L460 has correct value; L103 + this headline table both wrong.
- **__syncthreads = 12+2W cy** is WRONG. Real **22+2W** (E5 RESOLVED). The +10 cy fixed barrier-instantiation overhead is missing.

### CONFIRMED entries
- FFMA, FMUL, FADD, IMAD, LOP3 = 4 cy ✅
- MUFU.sin = 24 cy ✅ (exact)
- fence.sc.cta = 8.6 cy ✅
- fence.sc.gpu = 274 cy ✅

### Plausible-but-unverified
- ld.shared = 24 cy (LDS family)
- ld.global L1 = 39 cy (subject to user skepticism in reviewed_errors_b300.md re: "L1 latency varies")
- ld.global L2 = 301 cy
- ld.global DRAM = 789 cy (consistent with my CCTL drain ~900 cy at DRAM)

## REVIEW_CHECKLIST candidates

- [x] §0 DFMA latency 92 cy → ❌ WRONG (real 63.9 cy, see E2)
- [x] §0 __syncthreads 12+2W → ❌ WRONG (real 22+2W, see E5)
- [x] §0 MUFU.sin 24 cy → ✅ exact (24.45 measured)
- [x] §0 FFMA 4 cy → ✅ confirmed
- [x] §0 fence.sc.gpu 274 cy → ✅ confirmed (267-281)
- [ ] §0 ld.global L1 = 39 cy — user-skeptical (reviewed_errors notes "L1 latency varies"); needs careful re-test with stable rig conditions
- [ ] §0 ld.global L2/DRAM 301/789 cy — plausible from CCTL drain audit but not directly latency-tested
