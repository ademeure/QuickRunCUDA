# B300 / Blackwell sm_103a Catalog Audit — Reading Guide

> **What this is:** a 4-document audit of `B300_PIPE_CATALOG.md` (19,742 lines). The catalog was largely produced by a long autonomous loop and the user warns at top: *"I have not manually checked most of this, and some of it is definitely wrong or misleading."* This audit re-runs the high-value claims, flags the wrong ones, and gives you reliable starting points.

## The 4 documents

| File | Lines | What it's for | Read order |
|---|--:|---|---|
| **`B300_AUDIT_README.md`** *(this file)* | — | Reading guide / orientation | start here |
| **`STATUS_OF_REPLICATION.md`** | ~70 | At-a-glance status table — what's been replicated, what's pending | next |
| **`DENSE_B300_PIPE_CATALOG.md`** | ~1100 | Pruned, dense, reliable subset. **Use this for actual work.** Each section is a condensed version of the catalog with the wrong / outdated / low-value bits removed. | the workhorse |
| **`JUSTIFIED_B300_PIPE_CATALOG.md`** | ~120 + index | Per-section audit trail. Each entry links to a detailed `justifications/<id>.md` record with: claim, test, build, run, raw output, SASS, ncu, verdict. | when you want to verify a number |
| **`REVIEW_CHECKLIST_B300.md`** | ~150 + 124 supplements | Yes/no items I'm not 100% certain of. Mark `[x]` for "right" / `[ ]` for "wrong" / `// note` for comments. Grouped by topic (A FFMA / B memory / C pipe / D tensor / E latency / F power / G TMA / H methodology / CRIT1-10). | when you want to give feedback |
| `justifications/` | — | Detailed audit records — one per section. SASS dumps, ncu output, raw timings, reconciliation tables. | drill-down |

## Source documents (referenced, not modified)

| File | Status |
|---|---|
| `B300_PIPE_CATALOG.md` | Original 19,742-line catalog (the audit subject; preserved unmodified) |
| `reviewed_errors_b300.md` | User's review notes on a DIFFERENT (lower-quality) doc — used as INSPIRATION for what kinds of errors to look for, NOT direct comments on PIPE_CATALOG |
| `b300_clean/` | Original 188-file catalog corpus (predecessor to PIPE_CATALOG) |
| `CLAUDE.md` | Methodology guide (theoretical peaks, rigor protocol, common pitfalls) |

## What's been done (TL;DR — UPDATED 2026-04-23 mature state)

**JUSTIFIED status:** **38 ✅ verified + 6 ⚠ partial + 4 🟡 preserved** = 48 sections covered (essentially the entire foundational early/mid catalog: §0-§30).

**The ✅ verified sections** include: cheat-sheet, latency table, FFMA peak, memory hierarchy, pipe topology, all of §2.1-§2.13 instruction catalog, contention rules, rate cheatsheet, narrow-format throughput, uniform datapath, ADU pipe, SASS↔PTX mapping, redux.sync, pipe_alu ceiling, predication, extended ops, atomics, MUFU, dual-issue, .reuse, compute-mem-overlap, N=2 atomic hotspot, MUFU sweep, latency reference, final throughput, warp coop, BF16, compiler-emission gaps, warp-reduce, fence/ALU/CCTL.

**🟡 preserved**: tcgen05.mma throughput (alloc/mbarrier setup deferred), multi-GPU all-reduce (GPU 0 only this session), methodology notes, tensor unified.

**9 catalog ERRORS** flagged in REVIEW_CHECKLIST top-9:
- DFMA latency: 92 → real **63.9 cy**
- syncthreads formula: 12+2W → real **22+2W** (= 54 cy at BS=512, NOT 45)
- FP64 chip: "475 GFLOPS" wording = "475 G FMA-ops/s = 950 GFLOPS"; real 1060 GFLOPS (12% off)
- §4 MUFU rate: "16 SASS/SM/cy" off by 16-32× (real ~1.0)
- Atomic hotspot at warp-level N=2: "5×" → real **34×**
- FP64 vs FP16 tensor: "300×" → real **2300×**
- mbarrier.arrive 8.1 cy → real **27 cy** for default `.shared.b64` modifier
- atom.global.cas SASS: "STRONG.GPU" → actual **STRONG.SYS**
- ld.shared bank-conflict scoping (TRUE for v2/v4 AND 32-bit, methodology error in our prior claim)

**13 NEW architectural facts** missing from catalog (top-13 in REVIEW_CHECKLIST TLDR):
- F2IP.U8 fast path (4× faster than F2I.S8)
- POPC.INC trick: `atomicAdd(addr, 1u)` → 2.5× speedup at warp-broadcast
- Global REDG (no-return) vs ATOMG = 25× speedup
- EX2 uniquely 2× faster than other MUFU ops (4 vs 8 cy)
- bf16x2 EX2 = same throughput at half dispatch pressure
- FMNMX3 fusion (Blackwell 3-input min/max)
- u64.ADD alu+fmaheavy co-issue = 64 u64-adds/SM/cy
- CCTL.IVALL essentially FREE (~3 cy idle), drain-wait dominates fence cost
- release.gpu drain is SM-WIDE (compounds at high occupancy)
- cp.async (LDGSTS) bypasses acquire fence drain unless commit_group
- **`.L2::256B` cache hint = 92% HBM SoL recipe** (40% boost over baseline)
- **`.ca` beats `.cg` by 1.88× at L1-fitting WS** (NOT 1.25× as catalog says)
- **`ld.const` (LDC.32) dispatches via the ADU pipe, not LSU** — catalog §1 PTX→pipe table is missing the LDC row; broadcast achieves 17.99 TB/s eff = 98% of ADU SoL

**4 SELF-CORRECTIONS** caught and walked back during the audit:
- FP64 catalog "off by 2.2×" → actually 12% off (wording confusion)
- SMEM 32-bit bank conflicts "absent on B300" → REAL at 9.6× when properly tested
- SHFL broadcast "1.9 cy free" → 7.46 cy in general case
- .ca/.cg "no gap at 4 MB WS" → was test-config issue (WS exceeded L1)

**8 methodology lessons** documented in JUSTIFIED TLDR + per-section records.

## How to use this audit

### "I want a number I can trust"

→ DENSE_B300_PIPE_CATALOG.md, find the section, take the number. Numbers there are either ✅ replicated or 🟡 catalog-preserved-with-caveat.

### "I'm about to cite a catalog number — should I?"

→ STATUS_OF_REPLICATION.md, find the row. If verdict is ✅ matches, cite confidently with the new number. If ⚠ wrong, use the corrected number from DENSE. If 🔍 not yet, treat as suggestive.

### "I want to verify a number myself"

→ JUSTIFIED_B300_PIPE_CATALOG.md, find the section. Follow the link to `justifications/<id>.md`. That has the test file, build command, run command, raw output, SASS, ncu metrics. Re-run yourself.

### "I want to flag something as wrong"

→ REVIEW_CHECKLIST_B300.md. Find the entry under the relevant group, mark it. Add a `// note`.

### "I want to know what's still unverified"

→ STATUS_OF_REPLICATION.md "What's still NOT replicated" section. Or REVIEW_CHECKLIST_B300.md, count `[ ]` entries (~61 still open in main + 124 supplements).

## Audit conventions

- **Confidence tags:** 🟢 HIGH (3-method verified) / 🟡 MED (1-2 methods or regime caveat) / 🔴 LOW (methodology issue) / ⚫ DISPUTED (>1.5× spread no consensus)
- **Verdicts:** ✅ replicated / ⚠ partial or differs / ❌ falsified / 🔍 not yet attempted
- **Catalog citations:** `[ref: B300_PIPE_CATALOG.md:Lxxx]` — the original line number
- **Provenance:** every measured number cites either `justifications/<id>.md` or `[catalog claim, not yet rerun]`

## Methodology

Each replication follows CLAUDE.md's 8-step rigor protocol:
1. State theoretical max FIRST
2. State measured number
3. If measured >> theoretical: STOP — DCE / formula bug / clock mismatch
4. If measured > 1.5× theoretical: certain DCE
5. If measured < 0.5× theoretical: under-saturated or methodology issue
6. If in [0.5, 1.0]: plausible, verify SASS instruction count, ncu metrics
7. SASS-verify: `cuobjdump --sass` on the auto-dumped `sass/<test>_<hash>.sass`
8. Cross-check ncu pipe metrics where available

Plus 5 methodology footguns documented in this audit (see TL;DR above).

## What's still NOT done from the original vision

- DENSE catalog covers ~19 of catalog's 65 H2 sections (∼30%). Major missing: power per pipe, NVFP4, full methodology section, cluster launch overhead, predication, compiler-emission gaps, comprehensive reference card.
- ~61 REVIEW_CHECKLIST main items still open (out of 74); 124 supplementary skeptical-review items mostly not yet replicated.
- NVFP4 K=96 ULTRA replication failed (sub-agent token limit) — needs retry.
- Some catalog corrections recommended but not yet pushed back into the source `B300_PIPE_CATALOG.md`.

## Acknowledgments

The catalog was produced by Claude Opus 4.6 in autonomous loop mode. This audit was produced by Claude Opus 4.7 via 12 parallel sub-agents over ~20 hours of GPU time on B300 SXM6 AC (GPU 0; GPU 1 was held by another session).
