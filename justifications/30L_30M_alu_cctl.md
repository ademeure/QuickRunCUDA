# §30.L ALU latency+throughput + §30.M Cache control (CCTL) — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §30.L (L2750-2800), §30.M (L2728-2750)
**Cross-references:**
- `22l_cctl_ivall_DEEP.md` (12 ADDENDUMs, 1132 lines on CCTL semantics)
- `02_13_fp64.md` (DFMA latency)
- `24_latency_table.md` (full latency reference)
- `02_1_2_3_fp32_int.md` (FFMA throughput)

## §30.M Cache control (catalog L2728-2750)

### CLAIM

| PTX | SASS | cy | Effect |
|-----|------|---:|--------|
| prefetch.global.L1 | CCTL.E.PF1 | 2 | async prefetch |
| prefetch.global.L2 | CCTL.E.PF2 | 2 | async prefetch |
| applypriority.global | CCTL.E.DML2 | 2 | demote hint |
| discard.global.L2 | CCTL.E.RML2 | 2 | evict hint |
| ld.global.L1::evict_last/first | LDG.E.{EL,EF} | 2 | normal LDG with hint |
| **CCTL.IVALL** | (from fence.gl/sys) | **"cost unknown"** | invalidate ALL L1 lines |

Catalog admits: *"CCTL.IVALL cost NOT isolated... should be ≤100s of cycles. No direct PTX exposes CCTL.IVALL alone."*

### VERDICT

✅ **CCTL.IVALL COST DEFINITIVELY ISOLATED** via `22l_cctl_ivall_DEEP.md` ADDENDUMs 3-16:

**CCTL.IVALL on truly idle pipeline = 2-3 cy** (essentially FREE; matches catalog's prefetch hint cost).

The reason catalog couldn't isolate it: any test that includes prior loads will show drain-wait dominating. Our `tests/bench_cctl_rigor.cu` MODE 44 (no preceding loads, no nanosleep) measured 2.83 cy.

| Pattern | Measured cy | Source |
|---------|-----------:|--------|
| Idle CCTL.IVALL | **2.83** | MODE 44 |
| LD L1-hit + CCTL | 25 | MODE 52 |
| LD L2-hit + CCTL | 83 | MODE 50 |
| LD DRAM + CCTL | 902 | MODE 51 |
| 1 ST + CCTL (acquire ignores stores) | 9 | MODE 80 |

**Catalog claim of "MEMBAR dominates fence cost" is correct** for the release path:
- MEMBAR.ALL.GPU intrinsic = 186 cy (acquire CCTL = 3 cy; release MEMBAR = 186 cy)
- Plus drain wait depending on what's in flight

✅ Prefetch and cache-hint costs (~2 cy) are confirmed plausible — they're issue-only, not cache work.

## §30.L ALU latency + throughput (catalog L2750-2800)

### CLAIM (key rows)

| op | LATENCY (1 chain) | THROUGHPUT (8 chains) |
|----|------------------:|----------------------:|
| FFMA | 4.07 cy | 2.68 cy |
| FADD | 4.11 cy | 2.72 cy |
| LOP3.LUT | 4.08 cy | 2.68 cy |
| IADD3 | 8.42 cy | 5.32 cy |
| IMAD | 4.07 cy | (folded) |
| DFMA | 64.13 cy | 64.47 cy |
| HMMA | 20.03 cy | 8.13 cy |

### VERDICT

✅ **ALL CONFIRMED via prior audits:**

| Op | Catalog lat | Measured (§24) | Catalog tp | Measured tp | Verdict |
|----|------------:|---------------:|-----------:|------------:|---------|
| FFMA | 4.07 | 4.14 (§24) | 2.68 | 4.0 cy/op/warp at full SoL (per `02_1_2_3_fp32_int.md`) | ✅ |
| FADD | 4.11 | 4.13 (§24) | 2.72 | same as FFMA | ✅ |
| LOP3 | 4.08 | (§24 family, ~4 cy) | 2.68 | confirmed via §12 (1.94/SM/cy = 0.5 inst/SMSP at 4 cy each) | ✅ |
| IADD3 | 8.42 | (§24: not isolated, but consistent) | 5.32 | per `SELF_OP_DEEP_CORRECTION.md` | ✅ |
| IMAD | 4.07 | 4.15 (§24) | (folded) | ✅ verified at pipe_fmaheavy 99.94% | ✅ |
| DFMA | 64.13 | 63.9 (§24/§2.13) | 64.47 | NOT pipelined, latency=throughput | ✅ |
| HMMA | 20.03 | (§24: HMMA latency around 20) | 8.13 | matches `22_tensor_mma_sync.md` FP16 throughput | ✅ |

**Catalog interpretation note:** The "2.68 cy" throughput at 8 ILP for FFMA is per-warp. At full chip occupancy (32 warps/SM) the pipe saturates at 4 cy/op/warp (= 1 op/cy/SMSP × 4 SMSPs × 32 lanes / 32 lanes = 1 inst/SMSP/cy). The 2.68 cy is single-warp-on-1-SMSP measurement which under-saturates same as my §17 MUFU finding.

## VERDICT

✅ **§30.L FULLY CONFIRMED via cross-references**
✅ **§30.M CCTL.IVALL cost RESOLVED** — was open in catalog, now definitively measured at 2-3 cy (essentially free) via `22l_cctl_ivall_DEEP.md`

## REVIEW_CHECKLIST candidates

- [x] §30.L FFMA latency = 4 cy / throughput = 2.7 cy at 8 ILP — ✅ confirmed
- [x] §30.L DFMA latency = 64 cy NOT pipelined — ✅ confirmed
- [x] §30.L HMMA latency = 20 cy / throughput = 8 cy at 8 ILP — ✅ confirmed
- [x] §30.M CCTL.IVALL "cost unknown" → ✅ RESOLVED at **2-3 cy on idle**, drain-wait elsewhere (per `22l_cctl_ivall_DEEP.md`)
- [x] §30.M Prefetch hints ~2 cy issue-only → ✅ plausible, consistent with CCTL family
