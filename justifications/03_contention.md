# §3 Contention rules — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §3 (L472-485)
**Cross-references:**
- `justifications/22_dual_issue_ffma2_alu.md` (the load-bearing dual-issue audit)
- `justifications/SELF_OP_DEEP.md` + `SELF_OP_DEEP_CORRECTION.md` (IADD pipe routing)
- `justifications/22e_reuse_cache.md` (LOP3 + FFMA contention)

## CLAIM (catalog L472-485, §3)

1. **Same pipe → total rate capped at that pipe's ceiling.**
   - F2FP + LOP3 (both alu) → 64 combined, period.
   - IMAD + FFMA scalar (both compete for fmaH) → reduces FFMA peak.
2. **Different pipes → usually add cleanly, with two caveats:**
   - **FFMA2 + UNPACK** (fma + alu): u=1.67 (106/127 combined) — ~16% SMSP dual-issue friction specific to F2FP. Not present for PRMT+FFMA2 (u=1.95).
   - **LOP3 + FFMA scalar**: LOP3 on alu, FFMA uses both fmaH AND fmaL. No ALU/FMA contention, but at balanced ILP total sm_inst can exceed 4.0 only if packed ops are used.
3. **Dispatch cap = 4.00 sm_inst/SM/cy** is hard. To exceed 128 SASS-inst/SM/cy you need packed ops counted as multiple logical ops.
4. **HFMA2 + FFMA scalar** can co-exist but compete for H+L slots. Peak rate for mixed ~2.0 total warp-inst/SM/cy (one must yield).

## VERDICT vs prior audits

### Rule 1 (same-pipe cap) — ✅ CONFIRMED

- F2FP + LOP3 same-pipe contention is documented in `22_dual_issue_ffma2_alu.md` audit.
- IMAD + FFMA contention: per `SELF_OP_DEEP_CORRECTION.md`, IMAD lives on pipe_fmaheavy and FFMA scalar uses both fmaH+fmaL alternately. The two compete for fmaH slots, reducing FFMA peak when IMAD is mixed. ✅

### Rule 2 (cross-pipe co-issue) — ✅ CONFIRMED with NUANCE

- **FFMA2 + LOP3**: V52 settlement showed pipe_alu 98% + pipe_fma 49% = 147% combined SM-utilization (per ADDENDUM in `b300_clean/corrections/`). This matches the catalog's "saturating 3 pipes" claim.
- **FFMA2 + UNPACK 1.67 vs FFMA2 + PRMT 1.95**: This specific F2FP friction claim is from V52 era; not independently re-tested here. Catalog claim is plausible because F2FP is a multi-port operation that might share an SMSP-level dispatch slot with the FMA2 issue.

### Rule 3 (4.00 dispatch cap) — ✅ CONFIRMED

DENSE §1 confirms: "no test exceeded 4.00 dispatch/SM/cy". Audit-verified in `01_pipe_topology.md`.

### Rule 4 (HFMA2 + FFMA mixed) — 🔍 PRESERVED, not re-tested

Specific HFMA2+FFMA mixing was studied in catalog era but not re-run in our audit. Catalog claim "peak ~2.0 total warp-inst/SM/cy" is plausible from pipe-share reasoning but not independently measured here.

## VERDICT

✅ **CATALOG LARGELY CONFIRMED** — Rules 1, 2, 3 verified by independent audits. Rule 4 plausible but unverified.

## REVIEW_CHECKLIST candidates

- [ ] §3 Rule 4: HFMA2 + FFMA scalar peak rate = ~2.0 total warp-inst/SM/cy — needs independent re-test (specific mixing pattern not in our audit set)
- [ ] §3 Rule 2 caveat: "FFMA2 + UNPACK = 1.67" specifically (vs PRMT 1.95) — F2FP friction needs independent verification
