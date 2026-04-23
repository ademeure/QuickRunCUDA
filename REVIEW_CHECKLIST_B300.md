# B300 Catalog — Review Checklist (one-line yes/no per claim)

> **Instructions for user:** Each row is a single claim I am NOT 100% certain of. Mark with `[x]` for "yes, this is correct", `[ ]` for "no, wrong", and add a `// short note` after the line if you want to leave a comment. The structure is grouped by catalog section so you can stop after any group without losing context.
>
> **Format:**
> - `[ ]` claim text — context (numeric value, regime if relevant) — `[ref: B300_PIPE_CATALOG.md:Lxxx]` — `[reason for uncertainty]`
> - One blank line between groups.
>
> **Color key for `[reason for uncertainty]`:**
> - `unverified`: I haven't been able to replicate it yet
> - `clock-mismatch`: number was at a different clock than headline implies
> - `formula`: claim looks like it's computed from spec, not measured
> - `DCE-suspect`: pattern looks vulnerable to compiler eliminating the loop
> - `agent-hearsay`: number came from an LLM swarm without independent re-verify
> - `inconsistent`: same topic reported with different number elsewhere
> - `regime-narrow`: holds only under one combination of (warps, BS, working set) and may not generalize
> - `unit-confusion`: GB vs GiB or wire-rate vs effective ambiguity
> - `superseded-suspect`: I think a later test contradicted this but haven't traced

---

## Group A — Headline FFMA / FP32

(populated as analysis progresses)

## Group B — Memory hierarchy

(populated as analysis progresses)

## Group C — Pipe topology / dispatch ceiling

(populated as analysis progresses)

## Group D — Tensor cores

(populated as analysis progresses)

## Group E — Latency / sync / atomics

(populated as analysis progresses)

## Group F — Power / clock / DVS

(populated as analysis progresses)

## Group G — TMA / mbarrier / cluster

(populated as analysis progresses)

## Group H — Methodology assumptions baked into many measurements

(populated as analysis progresses)
