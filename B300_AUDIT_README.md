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

## What's been done (TL;DR)

- **12 catalog sections fully replicated** with SASS + ncu evidence (FFMA peak, mem hierarchy, pipe topology, dual-issue FFMA2+ALU, tensor mma.sync FP16/TF32/FP8/INT8, latency table, atomics, fence costs, TMA size dependence, TMA-vs-LDG max-tuned head-to-head, DSMEM, DSMEM exhaustive 9-dim sweep)
- **12 catalog corrections recommended** (FP8 emulated 276→308 TF, syncthreads formula `12+2W`→`22+2W`, DFMA latency 92→63.7 cy, fence costs single-GPU vs multi-GPU split, FP16 atomicAdd 45×→6.3× slower than u32, etc.)
- **3 major catalog claims FALSIFIED** with mechanism explained:
  - "DSMEM is essentially free at 23 cy" → actually 204-223 cy = 9× slower; SASS reveals `ld.shared::cluster` compiles to `LD.E` (global LSU path), not `LDS`
  - "FFMA uniquely uses both fma sub-pipes simultaneously" (catalog L218) → FFMA dispatches to ONE sub-pipe per cycle, scheduler alternates
  - Catalog "L2 wire 13.3 TB/s" → real measurement 18-20 TB/s (catalog under-counts by 37-54%)
- **3 major NEW topology findings** beyond catalog:
  - **B300 SXM6 AC has 9 GPCs × 16 SMs + 1 partial 4-SM GPC = 148** (catalog claims "8 GPCs" — wrong)
  - **Per-GPC silicon variation 20%** — GPC2 latency 189 cy vs GPC1 229 cy (catalog assumes uniform)
  - **DVFS settling clock under sustained load is 1942 MHz** — neither catalog's 1920 nor spec's 2032
- **Methodology footguns documented** (5):
  - ncu `lts__t_bytes` undercounts LDG L2-hit by 2.7× (use `l1tex__t_bytes` for LDG instead)
  - `sm__sass_data_bytes_mem_shared_op_ld` reports warp-aggregated bytes not per-lane (32× undercount risk)
  - default `LDG.E` hits L1 even for "DRAM" tests unless `.cg` + Sattolo chain
  - chain-feedback patterns let compiler DCE 32× of inner loop body
  - `atom.global.add` compiles to `REDG.E.ADD.STRONG.GPU`; ncu `lts__t_sectors_op_atom` reports 0 — use `lts__t_sectors_op_red`

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
