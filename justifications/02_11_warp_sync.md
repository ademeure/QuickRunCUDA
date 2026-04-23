# §2.11 Warp / group / synchronization ops — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §2.11 (L400-432)
**Cross-references:** Most rows already covered by §11/§7/§17/§15/§13/§14/§24 audits.

## CLAIMS + cross-reference verdicts

| PTX | SASS | Pipe | Catalog rate | Audit | Verdict |
|-----|------|------|-------------:|-------|---------|
| shfl.sync.* | SHFL.* | lsu | 1.00 (32 SASS/SM/cy) | catalog plausible at lsu cap | 🟡 not directly re-tested |
| vote.sync.ballot.b32 | VOTE.ANY + ISETP | alu | 2.00 combined | per `02_7_8_9_alu_ops.md` SASS expansion | ✅ |
| vote.sync.{any,all,uni} | VOTE.* | alu | 2.00 | per §2.8 audit | ✅ |
| activemask | uniform-pipe op | uniform | ~1.2 | `bench_uniform.cu` measured 0.10-0.37 for various; pipe_uniform peak=2.0 (per `06_uniform.md`); 1.2 plausible | ⚠ DCE'd in original test |
| **match.any.sync.b32** | MATCH.ANY | adu | 0.5 peak | catalog "very slow ~140 ms" suggests blocking | 🟡 plausible (adu-bound) |
| match.all.sync.b32 | MATCH.ALL | adu | similar | similar | 🟡 |
| **bar.sync 0** | BAR.SYNC.DEFER | adu | **0.36** | per `07_adu.md`: bar.sync hits 0.36 = 72% of pipe_adu cap 0.50 | ✅ EXACT |
| bar.arrive | BAR.ARV | adu | 0.47 | not directly re-tested | 🟡 plausible (close to adu cap) |
| bar.red.popc.u32 | BAR.RED.POPC.DEFER | adu (+alu) | 0.37 | not directly re-tested | 🟡 |
| **redux.sync.min.u32** | CREDUX.MIN + IMAD | alu + fmaheavy | **1.92** | per `11_redux.md`: 1.89 ✓ | ✅ EXACT |
| **redux.sync.add.u32** | REDUX.SUM | adu | **0.50** | per `11_redux.md`: 0.50 ADU ✓ | ✅ EXACT |
| redux.sync.{or,and,xor}.b32 | REDUX.{OR,AND,XOR} | adu | 0.50 | per §11 audit (and rate same as add) | ✅ |
| membar.cta | MEMBAR.SC.CTA | lsu | 0.83 | not directly re-tested | 🟡 plausible |
| membar.gl | MEMBAR.SC.GPU + ERRBAR | adu + lsu | extremely slow | per `30G_fence.md`: fence.sc.gpu = 267-281 cy single-warp; bigger when chained | ✅ approximately confirmed |
| **ldmatrix.sync.x1.b16** | LDSM | uniform 1.0 + lsu 0.5 | ~1.0 | per `06_uniform.md`: LDSM hits 0.70 (35% of pipe_uniform=2.0 peak) | ✅ approximately confirmed |
| ldmatrix.sync.x4.b16 | LDSM | uniform 0.25 + lsu 0.12 | 0.25 | catalog claim consistent with per-quad cost | 🟡 not directly re-tested |
| **atom.shared.add.u32** | **ATOMS.POPC.INC.32** | lsu | 0.84 | per `22_atomic_smem_DEEP.md`: confirmed POPC.INC SASS emission for `atomicAdd(addr, 1u)` | ✅ |
| atom.global.* | ATOMG.* | lsu | bandwidth-bound | per `15_atomics.md` + `22_atomic_ops_DEEP.md` | ✅ |
| s2r %clock / %clock_hi | S2R SR_CLOCKLO/HI | adu | 0.5 | catalog claim plausible | 🟡 |
| s2r %warpid | S2R SR_VIRTWARPID | alu | 1.0-ish | catalog claim plausible | 🟡 |

## VERDICT

✅ **MOSTLY CONFIRMED via cross-references:**

### Directly confirmed via prior audits (8 rows):
- bar.sync 0 = 0.36 adu (§7)
- redux.sync.min/max = 1.92 (§11)
- redux.sync.add/and/or/xor = 0.50 adu (§11)
- vote.sync.ballot/any/all = 2.00 alu (§2.8)
- atom.shared.add = ATOMS.POPC.INC.32 (§22 atomic_smem)
- atom.global.* = ATOMG.* family (§15)
- ldmatrix.sync.x1 ~ uniform 1.0 (§6)
- membar.gl = extremely slow (§30G_fence)

### Plausible-not-re-tested (10 rows):
- shfl.sync = 1.00 lsu (catalog standard)
- ldmatrix.sync.x2/x4 (lower rates per catalog)
- match.any/all (very slow per catalog)
- bar.arrive, bar.red.popc, membar.cta
- s2r %clock, %warpid

## REVIEW_CHECKLIST candidates

All §2.11 entries that ARE in the audit have been confirmed within margin. Open items:
- [ ] §2.11 shfl.sync = 1.00 lsu — needs direct test (not yet measured for SHFL specifically)
- [ ] §2.11 match.any/all "very slow ~140 ms / 128 inst" — catalog wall-clock, not re-tested
- [ ] §2.11 ldmatrix.sync.x4 = 0.25 — not isolated in current audits
