# §2.12 Memory ops — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §2.12 (L432-444)
**Cross-references:**
- `00b_mem_hierarchy.md` (memory bandwidth ladder)
- `15_atomics.md` + `22_atomic_ops_DEEP.md` (atomic SASS + rates)
- `22_atomic_smem_DEEP.md` (smem atomic + POPC.INC trick)

## CLAIM (catalog L432-444)

| PTX | SASS | pipe | note |
|-----|------|------|------|
| ld.global.u32 | LDG.E | lsu | DRAM-bound, ~1 inst/SM/cy issue |
| st.global.u32 | STG.E | lsu | bandwidth-bound, not pipe-bound |
| ld.shared.u32 | LDS | lsu | ~1.0 issue, bank-conflict-sensitive |
| st.shared.u32 | STS | lsu | 1.00 saturating |
| atom.* | ATOMS / ATOMG | lsu | not measured |

## VERDICT — cross-references

| Op | Verified via | Status |
|----|--------------|--------|
| ld.global.u32 (LDG.E) | `00b_mem_hierarchy.md`: HBM=7.17-7.25 TB/s = 95% of 7672 GB/s spec; LDG behavior confirmed | ✅ DRAM-bound at chip scale |
| st.global.u32 (STG.E) | `00b_mem_hierarchy.md` + DRAM streaming tests | ✅ bandwidth-bound, ~7 TB/s with streaming |
| ld.shared.u32 (LDS) | `00b_mem_hierarchy.md`: SMEM=35.88 TB/s = 97.5% of 36.79 TB/s theoretical at 1942 MHz | ✅ 1.0 issue confirmed |
| ld.shared bank-conflict | `22j_smem_bank_conflicts_DEEP.md`: 32-bit LDS shows NO conflict penalty; v2/v4 do | ✅ catalog "bank-conflict-sensitive" CORRECT but only for wider loads |
| st.shared.u32 (STS) | implicit from §1 pipe_lsu = 1.00 cap | ✅ |
| **atom.* "not measured"** | `15_atomics.md` + 2 detail docs comprehensively measure atomics | ✅ NOW MEASURED (was open) |

## NEW INSIGHT (cross-section)

The `22j_smem_bank_conflicts_DEEP.md` finding refines catalog's "bank-conflict-sensitive" wording:
- **32-bit LDS**: bank conflict penalty is essentially **zero** on B300 (random = stride-N for any N — all hit ~6.88 cy)
- **64-bit (v2.b32) LDS**: stride-2 = 10.94 cy = **1.58× slower** than seq
- **128-bit (v4.b32) LDS**: stride-2 = 19.69 cy = **1.68× slower**

So bank conflicts ARE real for wider loads but eliminated for single-dword on B300 (likely via hash-accelerated bank arbitration).

The catalog row "ld.shared.u32 ... bank-conflict-sensitive" is technically WRONG for u32 LDS on B300 — needs scoping to wider loads.

## VERDICT

✅ **CONFIRMED with one refinement:**
- LDG/STG/LDS/STS pipe assignments and rates all correct
- Atomic "not measured" entry: NOW comprehensively measured in §15 detail docs
- Bank-conflict claim needs scoping: TRUE for v2/v4 LDS, FALSE for u32 LDS on B300

## REVIEW_CHECKLIST candidates

- [x] §2.12 LDG/STG = LSU pipe — ✅ confirmed
- [x] §2.12 LDS/STS = LSU 1.00 — ✅ confirmed via §1 + bandwidth audit
- [ ] §2.12 "ld.shared bank-conflict-sensitive" — TRUE for v2/v4 wide LDS, FALSE for u32 LDS on B300; needs scoping
- [x] §2.12 atom.* "not measured" — RESOLVED via §15 + 22_atomic_ops_DEEP.md + 22_atomic_smem_DEEP.md
