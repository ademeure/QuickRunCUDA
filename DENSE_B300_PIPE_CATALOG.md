# DENSE B300 / Blackwell sm_103a — SM Pipe Catalog

> **Status:** Iteration in progress. This is a pruned, dense version of `B300_PIPE_CATALOG.md` that removes content known to be wrong, outdated, or low-value. Every numerical claim that survives has either a JUSTIFIED entry or a REVIEW_CHECKLIST flag.
>
> **Source:** `B300_PIPE_CATALOG.md` (19,742 lines).
> **Audit input:** `reviewed_errors_b300.md` (user's [!fail] / [!todo] callouts on related canonical doc).
> **Validation:** `JUSTIFIED_B300_PIPE_CATALOG.md` (per-section replication records).
>
> **Pruning rules applied:**
> 1. Remove anything contradicted by V52/V53/V54 empirical settlements
> 2. Remove anything user [!fail]'d in `reviewed_errors_b300.md` for the same topic
> 3. Remove "research log" repetition (sections 16–22 in catalog)
> 4. Remove formula-as-measurement claims (e.g. headline TFLOPS computed from cores × clock with no actual FLOPS counter)
> 5. Keep only LOW-LEVEL findings (drop high-level cuBLAS / GEMM ladder content unless it teaches something architectural)
> 6. Tag every surviving claim with confidence: 🟢 (3-method verified), 🟡 (1-2 method or regime caveat), 🔴 (unverified, kept because hint-bearing)
>
> **Conventions:**
> - All clock state explicit: `@1920` / `@2032` / `@1500` / `@1005` / `@boost`
> - All BW with denominator: spec=7680 GB/s, this-device=7672 GB/s
> - All TFLOPS state count convention: `2 FLOPS/FFMA` etc.
> - `⚠ FOOTGUN` for commonly mis-cited claims
>
> **Rough page budget:** ~2,500 lines (vs 19,742 source). Aggressive prune.

---

## Top of sheet — what to use this for

If you want a single number with a confidence tag: read the relevant `## §N` section, take the bold answer, ignore the rest.

If you want to verify a claim: jump to `JUSTIFIED_B300_PIPE_CATALOG.md` for the same `§N`.

If you don't trust a number: it's probably already on `REVIEW_CHECKLIST_B300.md`.

---

## §0. Spec card

(In progress — populated as JUSTIFIED entries are filled.)

---

## §1. Pipe topology

(In progress.)

---

## §2. Instruction catalog

(In progress.)

---

(More sections added as the JUSTIFIED catalog progresses.)
