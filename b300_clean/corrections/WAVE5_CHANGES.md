# WAVE 5 CHANGES — what shifted vs wave-4

**Date:** 2026-04-22
**Author:** wave-5f synthesis
**Inputs:** SASS_VERIFY_DUAL_ISSUE.md (5a), HBM_DENOMINATOR_FINAL.md (5b),
PROPOSED_FIXES.md (5c), CONFIDENCE_LADDER_PATCH.md (5d, **now stale**),
CROSS_LINK_AUDIT.md (5e).
**Status:** SUPERSEDES the dual-issue MED grade in CONFIDENCE_LADDER_PATCH.md.

---

## 1. Dual-issue: 4-level zigzag — final verdict LOW

### 1.1 The zigzag

| Wave | Verdict | Reason | Verdict was… |
|---|---|---|---|
| W1+W2 | HIGH (55 % same-warp / 74 % warp-spec) | V49/V50 measurement reproducible | **wrong** (not the right number) |
| W3b doubt | LOW | Suspected under-occupancy (2 warps/SMSP × 8 ILP too thin) | **right answer, partly right reason** |
| W4 meta-doubt | MED (RE-PROMOTE) | V8_FFMA hits 97.6 % at the IDENTICAL 2 warps/SMSP geometry — under-occupancy hypothesis falsified | **wrong answer, right falsification** |
| W5a SASS-verify | LOW (RE-DOWNGRADE) | Loop-overhead contamination: tiny inner body (8 FFMA + 8 LOP3 + BRA), branch + UIADD3 + UISETP eat ALU slots. V8 unrolls 128-deep so amortizes branch ~16×. | **right answer, NEW reason** |

W3b LOW was right. W4 MED was wrong but its falsification of W3b's mechanism stands. W5a LOW is right again, for a third (independent) reason.

### 1.2 What W5a SASS analysis actually found

- **Source-level (PTX):** V49/V50 use `fma.rn.f32 %0, %0, IMM, IMM`; V8 uses `fma.rn.f32 %0, %0, %1, %0`. Meta-doubt called this "self-RAW with immediates" → wrong: immediates use FEWER RF read ports than V8's pattern, not more.
- **SASS:** V49/V50 compile to `FFMA Rd, Rd, R0.reuse, 0.5` (compiler hoisted the 1.5f IMM into R0 once, third operand stays IMM). V8 compiles to `FFMA Rd, Rsrc, Rd, Rd`. Both avoid 3-distinct-source RF port pressure. Source-pattern is NOT the issue.
- **Real bug:** V49 inner body = 8 FFMA + 8 LOP3 + 1 BRA. Branch + loop counter (UIADD3 / UISETP) consume ALU dispatch slots that contaminate the FFMA+LOP3 dual-issue measurement. V8 unrolls 16×8 = 128 FFMA per outer iter, so branch overhead amortizes 16× better.

### 1.3 V52 fix recipe (settles dual-issue)

1. Inner unroll depth ≥ 64 ops per type
2. `__launch_bounds__(256, 1)` (match V8, not V49's `(128, 2)`)
3. Anti-DCE: STG of accumulator XOR (not clock-diff conditional)
4. ncu `sm__inst_executed_pipe_fma.avg.pct_of_peak` AND `sm__inst_executed_pipe_alu.avg.pct_of_peak` simultaneously

Until V52 actually runs, "same-warp 55 % vs warp-spec 74 %" is **measurement artifact, not architectural finding**. Whether B300 SMSP can dual-issue FMA + ALU **remains OPEN**.

---

## 2. HBM denominator — 7672 is REAL (not a ghost)

### 2.1 What W5b found

| Number | Meaning | Status |
|---:|---|---|
| 7680 GB/s | Spec post-ECC at 8.000 Gbps/pin | Default cite |
| 7672 GB/s | THIS-DEVICE post-ECC at empirical 7.992 Gbps/pin (3996 MHz I/O measured by `nvidia-smi -q`) | Real silicon, cite alongside 7680 |
| 8192 GB/s | Spec raw pre-ECC | When excluding ECC parity |
| 8183.8 GB/s | This-device raw at 3996 MHz | Symmetric to 7672 raw side |

**The 7680→7672 gap (0.10 %) is real silicon under-spec, NOT arithmetic noise.**

### 2.2 What this overturns

W4 retired 7672 as "arithmetic ghost from mixing 8 Gbps spec with 7.992 Gbps measured". W5b shows: `nvidia-smi -q` reports `Memory: 3996 MHz` on this device. 7672 IS the literal hardware rate. W4's "ghost" framing is wrong.

**Default rule from W5b:** Cite 7680 (spec, comparable across vendors). When SoL precision matters, also cite 7672 (this-device actual). Both correct under their framings.

---

## 3. CLAUDE.md does NOT contain the "12 stacks" claim (W4 hallucination)

W4 brief instructed wave-5c to fix "12 HBM stacks" in 3 files including CLAUDE.md. **W5c grep proved CLAUDE.md does not contain "stack" anywhere.** The HBM line (CLAUDE.md:85) is `**HBM3E: ~7-7.5 TB/s** read peak (matches 8 TB/s spec)` — no stack count cited.

**Files that DO contain the error:**
- `b300_clean/B300_TRUE_REFERENCE.md` line 15 ("12 HBM3E stacks")
- `b300_clean/01_hbm_bandwidth.md` line 3 ("12 stacks HBM3E")
- `b300_clean/01_hbm_bandwidth.md` line 136 ("12 × HBM3E stacks × 1024-bit physical = 12,288-bit raw")

W5c proposes diffs for both (PROPOSED_FIXES.md §2, §3). Optional CLAUDE.md enhancement (Fix 1) is enrichment only, NOT a correction.

8-stack is confirmed by **bus width, not capacity**: NVIDIA Developer Blog says 16 × 512-bit controllers = 8192-bit total bus / 1024 bits per HBM3E stack = 8 stacks. Capacity (288 GB) admits 8×12-Hi×3 GB OR 12×12-Hi×2 GB; bus width is the dispositive evidence.

---

## 4. CONFIDENCE_LADDER_PATCH.md (W5d) is STALE

W5d was generated before W5a. It promoted V49/V50 dual-issue rows LOW → MED based on W4's meta-doubt. W5a re-downgraded to LOW.

**Required re-patches to CONFIDENCE_LADDER_PATCH.md:**

| Section | Current W5d | Should be |
|---|---|---|
| §2 row 1 (V49 same-warp 55 %) | MED / RE-PROMOTED | **LOW / RE-DOWNGRADED (W5a)** |
| §2 row 2 (V49 same-warp 54 % FFMA+IADD3) | MED / RE-PROMOTED | **LOW / RE-DOWNGRADED (W5a)** |
| §2 row 3 (V49 same-warp 51 % FFMA+PRMT) | MED / RE-PROMOTED | **LOW / RE-DOWNGRADED (W5a)** |
| §2 row 4 (V50 warp-spec 74 %) | MED / RE-PROMOTED | **LOW-MED / RE-DOWNGRADED (W5a)** — slightly better than V49, body cleaner per warp |
| §5 top-9 LOW list | dual-issue dropped | **dual-issue restored at #1** |
| §6 distribution: HIGH 259 / MED 35 / LOW 7 | — | **Should be HIGH 259 / MED 31 / LOW 11 (no net move)** |

---

## 5. HEADLINE_CORRECTIONS_v3 rule 11 needs softening per W5b

W4 v3 §rule 11 says "Use 7680 GB/s post-ECC as canonical denominator; 7672 is arithmetic ghost". W5b reverses the "ghost" framing. New rule per W5b §5:

> "Default denominator: 7680 (spec). When SoL precision matters, cite both 7680 spec and 7672 actual (3996 MHz I/O). Never call 7672 a ghost — it is the literal hardware rate on this silicon."

This is patched in `HEADLINE_CORRECTIONS_v4.md` (sibling file).

---

## 6. Summary of W5 deltas vs W4

| Topic | W4 said | W5 says | Driver |
|---|---|---|---|
| V49/V50 dual-issue | MED (RE-PROMOTED) | **LOW** (RE-DOWNGRADED, 3rd time) | W5a SASS analysis |
| HBM 7672 | "Arithmetic ghost — retire" | **Hardware-actual at 3996 MHz I/O — cite alongside 7680** | W5b `nvidia-smi -q` |
| CLAUDE.md "12 stacks" fix | Targeted CLAUDE.md | **Not in CLAUDE.md** — fix is in B300_TRUE_REFERENCE.md + 01_hbm_bandwidth.md | W5c grep |
| 8-stack count evidence | Capacity arithmetic | **Bus width** (8192-bit / 1024-bit per stack) | W5b §4 |
| CONFIDENCE_LADDER_PATCH.md | Live | **Stale on §2 (4 rows)** | W5d predates W5a |
| HEADLINE_v3 rule 11 | "7672 is ghost; use 7680" | **Cite both 7680 and 7672** | W5b §5 |

---

## 7. Open per-W5e cross-link audit

W5e found 17 stale cross-references across 8 files. Most concerning for the wave-5 era:
1. CONFIDENCE_LADDER §12 + DOUBT_LOG §3 still grade dual-issue LOW (W4 said MED) — **W5a says LOW again, so they are accidentally right; document why**
2. MASTER_INDEX_v2 line 8 points at v2 HEADLINE; should point at v4 (this wave)
3. HEADLINE_v3 rule 11 still says "7672 ghost"; needs the W5b correction
4. W4_CHANGES.md §2 + §6 still call 7672 a ghost; superseded by W5b

---

## 8. Files NOT modified by W5

Per constraints in this wave's brief: NO source files were edited. PROPOSED_FIXES.md (5c) holds the diff for B300_TRUE_REFERENCE.md and 01_hbm_bandwidth.md but it is unapplied. v3 of HEADLINE_CORRECTIONS is unchanged. v4 (sibling file) is the operational top-line going forward.
