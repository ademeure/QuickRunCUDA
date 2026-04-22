# Proposed Fixes — Wave-4 Stack-Count + Denominator Errata

**Date:** 2026-04-22
**Author:** wave-4 fix-proposal agent
**Status:** PROPOSALS ONLY — no source files have been modified.
**Apply with:** `git apply` of the unified diff in section "How to apply" (after user accepts).

---

## Pre-flight clarification on the assignment

The wave-4 brief identified three errors. After reading each target file:

- **CLAUDE.md** does NOT contain "12 HBM3E stacks" or "12 stack" anywhere.
  `grep -ni "stack" CLAUDE.md` returns no hits, and the only HBM line
  (line 85: `**HBM3E: ~7-7.5 TB/s** read peak (matches 8 TB/s spec)`) does not
  cite a stack count. **No fix is needed in CLAUDE.md for the stack-count
  error.** A *minor* annotation improvement is possible (Fix 1 below).
- The "12 stacks" error lives in **`b300_clean/B300_TRUE_REFERENCE.md` line 15**
  and **`b300_clean/01_hbm_bandwidth.md` lines 3 and 136**. Two of those need
  the same correction (Fix 2 + Fix 3).
- The muddled arithmetic is in `01_hbm_bandwidth.md` line 136 (Fix 3).
- The `7672 spec` annotation is in `B300_TRUE_REFERENCE.md` line 25 (Fix 4).

The memory file `project_b300_session4.md` (line 25: `HBM read … 7120 (93% of
7672 spec)`) is in `~/.claude/projects/.../memory/`, NOT in the project root.
It is also flagged as 4 days old by the harness. **Out of scope** for this
fix-proposal pass; if the user wants memory-file edits they can be done
separately. (See "Out-of-scope notes" at the end.)

---

## Fix 1 (OPTIONAL): CLAUDE.md HBM line — add denominator clarification

**Location:** `CLAUDE.md` line 85
**Current:**
> `- **HBM3E: ~7-7.5 TB/s** read peak (matches 8 TB/s spec)`

**Proposed:**
> `- **HBM3E: ~7-7.5 TB/s** read peak (matches 8 TB/s spec = 8192 GB/s raw / 7680 GB/s post-ECC; B300 has 8 stacks × 12-Hi)`

**Reasoning:** The line is correct as-is (no error to "fix"), but adding the
explicit denominator anchors it to the wave-5b `HBM_DENOMINATOR_RESOLUTION.md`
canonical numbers (8192 raw / 7680 post-ECC, 8 stacks × 12-Hi each) so future
readers do not re-introduce the "12 stacks" / "7672 spec" confusion. **This is
an enhancement, not a correction — defer if the user wants minimal changes.**

---

## Fix 2: B300_TRUE_REFERENCE.md "12 HBM3E stacks"

**Location:** `b300_clean/B300_TRUE_REFERENCE.md` line 15
**Current:**
> `System: NVIDIA B300 SXM6 AC (sm_103a, CC 10.3, 148 SMs, 12 HBM3E stacks),`

**Proposed:**
> `System: NVIDIA B300 SXM6 AC (sm_103a, CC 10.3, 148 SMs, 8 HBM3E stacks (12-Hi each, 288 GB total)),`

**Reasoning:** Per `corrections/HBM_DENOMINATOR_RESOLUTION.md` §1 (verified
against NVIDIA Blackwell Ultra blog, The Register, Tom's Hardware, and Glenn
Lockwood's garden page): B300 has **8 stacks of HBM3E, each 12-Hi**.
The "12" was die-stack height misread as stack count. Surrounding context
(SM count, CC, sm_103a) is preserved.

---

## Fix 3: 01_hbm_bandwidth.md "12 stacks" + muddled arithmetic

### Fix 3a: line 3 header

**Location:** `b300_clean/01_hbm_bandwidth.md` line 3
**Current:**
> `**GPU:** NVIDIA B300 SXM6 AC, sm_103a, 148 SMs, 12 stacks HBM3E, 7680-bit usable bus (post-ECC), 287.4 GB capacity. All measurements at 2032 MHz boost (default, no clock lock).`

**Proposed:**
> `**GPU:** NVIDIA B300 SXM6 AC, sm_103a, 148 SMs, 8 stacks HBM3E (12-Hi each), 8192-bit raw / 7680-bit post-ECC bus, 287.4 GB capacity. All measurements at 2032 MHz boost (default, no clock lock).`

**Reasoning:** Same as Fix 2. Also adds raw bus width (8192-bit) for symmetry
with the corrected derivation in §3b.

### Fix 3b: line 136 muddled stack-count arithmetic

**Location:** `b300_clean/01_hbm_bandwidth.md` lines 136–139 (the "Theoretical
accounting" section)
**Current:**
> ```
> - **Bus:** 12 × HBM3E stacks × 1024-bit physical = 12,288-bit raw; usable post-ECC = **7680 bit** (ECC reserves 1/16 = 1024 bit).
> - **Memory clock:** 3996 MHz effective (= 1998 MHz I/O × 2 DDR), datasheet "8 Gbps/pin".
> - **Spec peak:** 7680 × 3996 MHz × 2 / 8 / 1e9 = **7672 GB/s** post-ECC.
> - **Best measured:** 7.30 TB/s = **95.2% of 7672 GB/s spec**.
> ```

**Proposed:**
> ```
> - **Bus:** 8 × HBM3E stacks (12-Hi each) × 1024-bit physical = **8192-bit raw**; usable post-ECC = **7680 bit** (ECC reserves 1/16 = 512 bit).
> - **Per-pin rate:** 8 Gbps/pin (4.0 GT/s × 2 DDR), per HBM3E datasheet (NVIDIA chose conservative 8.0 of JEDEC's 8.0–9.6 range).
> - **Spec peak (raw):** 8 stacks × 1024 bit × 8 Gbps / 8 = **8192 GB/s** pre-ECC.
> - **Spec peak (post-ECC):** 8192 × 15/16 = **7680 GB/s** = the right denominator for ncu `dram__bytes_read.sum` (which counts user payload only).
> - **Best measured:** 7.30 TB/s = **95.0% of 7680 GB/s post-ECC** (= 89.1% of 8192 GB/s raw).
> ```

**Reasoning:** Per `corrections/HBM_DENOMINATOR_RESOLUTION.md` §2 and §3:
(a) the 12-stack count is wrong (it's 8 stacks × 12-Hi);
(b) the 7672 GB/s number is an arithmetic ghost from plugging 3996 MHz × 2 =
7.992 Gbps into the formula instead of the clean 8 Gbps/pin spec — it differs
from 7680 by 8 GB/s (0.1%), which is rounding noise and not a real spec;
(c) the round 7680 GB/s matches the 1/16 SECDED ECC ratio exactly (same as
B200's published 192→180 GB capacity ratio = 6.25%);
(d) the 95.0% post-ECC headline (7.30/7680) matches the existing 95.2%
within rounding, so the *empirical* claim is unchanged — only the **denominator
label** is fixed.

Also drop line 142 reference to the obsolete derivation, OR change it to:

> `The 8192 GB/s figure is the pre-ECC raw bus capability; 7680 GB/s is the post-ECC user-payload ceiling (1/16 reserved for SECDED). Anchor on 7680 GB/s for ncu-derived percentages, on 8192 GB/s when comparing against the physical-bus ceiling.`

(Original line 142 already retires the spurious 8183.8 figure; the rewording
above replaces "Anchor on 7672 GB/s" with the dual-anchor recommendation from
HBM_DENOMINATOR_RESOLUTION.md §4.)

---

## Fix 4: B300_TRUE_REFERENCE.md "7672 spec" annotation

**Location:** `b300_clean/B300_TRUE_REFERENCE.md` line 25
**Current:**
> `| **HBM3E read** | **7.30** | 95% of 7672 spec | v8 + per-warp coalesced + non-persistent (a04d9c8) |`

**Proposed:**
> `| **HBM3E read** | **7.30** | 95.0% post-ECC (denom 7680) | v8 + per-warp coalesced + non-persistent (a04d9c8) |`

**Reasoning:** The 7.30 TB/s measurement is unchanged. Only the denominator
label is corrected per HBM_DENOMINATOR_RESOLUTION.md §3 row 4 ("retire 7672").
The percentage rounds to 95.0% (was 95.2% on 7672); the headline "95% of peak"
is unchanged at the precision shown. If the user prefers the earlier wording:

**Alternate (more conservative, preserves voice):**
> `| **HBM3E read** | **7.30** | 95% (post-ECC, denom 7680) | v8 + per-warp coalesced + non-persistent (a04d9c8) |`

Also lines 26, 28, 29, 245 reference 7672/7.31 in the same way; per the brief,
**only line 25 is in scope for this fix** (the brief says "the 7.30 TB/s = 95%
measurement is unchanged; only the denominator clarification"). The rest can
be batched in a follow-up sweep using the canonical name "post-ECC (7680)".

---

## How to apply

Run `git apply` of the unified diff below if accepted by user.

```diff
diff --git a/b300_clean/B300_TRUE_REFERENCE.md b/b300_clean/B300_TRUE_REFERENCE.md
--- a/b300_clean/B300_TRUE_REFERENCE.md
+++ b/b300_clean/B300_TRUE_REFERENCE.md
@@ -12,7 +12,7 @@
 doubt, prefer numbers from this file. See `M3_REVERIFY_LOG.md` for the
 re-verification chain on each entry.
 
-System: NVIDIA B300 SXM6 AC (sm_103a, CC 10.3, 148 SMs, 12 HBM3E stacks),
+System: NVIDIA B300 SXM6 AC (sm_103a, CC 10.3, 148 SMs, 8 HBM3E stacks (12-Hi each, 288 GB total)),
 sustained 1920 MHz SM clock (boost is 2032 but rarely sustained),
 CUDA 13.2 runtime / 13.0 driver (580.126.09).
 
@@ -22,7 +22,7 @@
 
 | Memory | Peak (TB/s) | % of theoretical | Recipe (commit) |
 |---|---:|---:|---|
-| **HBM3E read** | **7.30** | 95% of 7672 spec | v8 + per-warp coalesced + non-persistent (a04d9c8) |
+| **HBM3E read** | **7.30** | 95.0% post-ECC (denom 7680) | v8 + per-warp coalesced + non-persistent (a04d9c8) |
 | **HBM3E write** | **7.30** | 95% | same recipe (a04d9c8) |
 | **HBM3E write NINJA (1 v8 store/warp)** | **7.57** | **98.7%** ← SoL | NINJA recipe (e75c7e1) — beats cudaMemset by 5% |
 | **HBM3E concurrent R+W (best ratio)** | **7.31** | 95% | pure R or pure W (de3b4d5) |

diff --git a/b300_clean/01_hbm_bandwidth.md b/b300_clean/01_hbm_bandwidth.md
--- a/b300_clean/01_hbm_bandwidth.md
+++ b/b300_clean/01_hbm_bandwidth.md
@@ -1,6 +1,6 @@
 # B300 HBM3E DRAM Bandwidth
 
-**GPU:** NVIDIA B300 SXM6 AC, sm_103a, 148 SMs, 12 stacks HBM3E, 7680-bit usable bus (post-ECC), 287.4 GB capacity. All measurements at 2032 MHz boost (default, no clock lock).
+**GPU:** NVIDIA B300 SXM6 AC, sm_103a, 148 SMs, 8 stacks HBM3E (12-Hi each), 8192-bit raw / 7680-bit post-ECC bus, 287.4 GB capacity. All measurements at 2032 MHz boost (default, no clock lock).
 
 ## TL;DR
 
@@ -133,12 +133,13 @@
 
 ## Theoretical accounting
 
-- **Bus:** 12 × HBM3E stacks × 1024-bit physical = 12,288-bit raw; usable post-ECC = **7680 bit** (ECC reserves 1/16 = 1024 bit).
-- **Memory clock:** 3996 MHz effective (= 1998 MHz I/O × 2 DDR), datasheet "8 Gbps/pin".
-- **Spec peak:** 7680 × 3996 MHz × 2 / 8 / 1e9 = **7672 GB/s** post-ECC.
-- **Best measured:** 7.30 TB/s = **95.2% of 7672 GB/s spec**.
+- **Bus:** 8 × HBM3E stacks (12-Hi each) × 1024-bit physical = **8192-bit raw**; usable post-ECC = **7680 bit** (ECC reserves 1/16 = 512 bit).
+- **Per-pin rate:** 8 Gbps/pin (= 4.0 GT/s × 2 DDR), per HBM3E datasheet; NVIDIA chose conservative 8.0 of JEDEC's 8.0–9.6 range.
+- **Spec peak (raw):** 8 stacks × 1024 bit × 8 Gbps / 8 = **8192 GB/s** pre-ECC (matches "8 TB/s" marketing).
+- **Spec peak (post-ECC):** 8192 × 15/16 = **7680 GB/s** = the right denominator for ncu `dram__bytes_read.sum` (counts user payload only, not ECC parity).
+- **Best measured:** 7.30 TB/s = **95.0% of 7680 GB/s post-ECC** (= 89.1% of 8192 GB/s raw).
 - **Remaining 4.8% gap:** HBM controller / refresh / row-precharge / command-bus overhead. Per old §1 of B300_PIPE_CATALOG.md, NVIDIA-published HBM3E efficiency targets land around 90–93% for COPY (R+W); pure-direction streams reach ~95% on this implementation.
 
-The 8183.8 GB/s figure (8192-bit pre-ECC) appears in some catalog sections as the "peak" and gives a misleading "75% of 8.17 TB/s" denominator when the real post-ECC bus is only 7672 GB/s. Anchor on 7672 GB/s.
+The 8183.8 GB/s figure (from `8192 × 7992/8000`) appears in some catalog sections as the "peak"; it is equivalent to 8192 GB/s within rounding noise. Anchor on **7680 GB/s post-ECC** for ncu-derived percentages, on **8192 GB/s raw** when comparing against the physical-bus ceiling. Retire 7672 (arithmetic ghost; see `corrections/HBM_DENOMINATOR_RESOLUTION.md`).
```

---

## Out-of-scope notes (flagged for future passes)

1. **CLAUDE.md "12 stacks" error does not exist.** The brief asked me to find
   it; `grep -ni "stack" /root/github/QuickRunCUDA/CLAUDE.md` returns nothing.
   Fix 1 above is an *optional enhancement*, not a correction. If the user
   thought the error was in CLAUDE.md, they may be conflating it with the
   `B300_TRUE_REFERENCE.md` line 15 occurrence. Confirm before applying Fix 1.

2. **`project_b300_session4` memory file** at
   `/root/.claude/projects/-root-github-QuickRunCUDA/memory/project_b300_session4.md`
   line 25 says `HBM read (4 GB workset, 8-ILP): 7120 (93% of 7672 spec)`.
   That file is in the user's auto-memory (not the project tree) AND the
   harness flagged it as 4 days old / point-in-time. Brief asked to "flag"
   it — flagged: the "7672 spec" denominator is wrong (should be "7680
   post-ECC" per HBM_DENOMINATOR_RESOLUTION.md). Edit only if user requests.

3. **B300_TRUE_REFERENCE.md lines 26, 28, 29, 245** have the same 7672
   denominator pattern. Brief explicitly limited scope to line 25; a follow-up
   sweep should align all of them on the canonical name
   `% post-ECC (denom 7680)`.

4. **01_hbm_bandwidth.md line 160** retired-claims row also references
   "7672 GB/s" — it should be updated to read "7680 GB/s post-ECC" for
   consistency with the new derivation. Not in scope per brief.
