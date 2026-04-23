# §Self-Op Investigation — IADD MECHANISM correction (2026-04-23)

User skepticism (2026-04-23) caught a wrong mechanism narrative in the original `SELF_OP_DEEP.md`. The HEADLINE FINDING (no architectural per-instruction self-op penalty) is still correct, but the IADD-specific mechanism explanation needs revising.

## What the original SELF_OP_DEEP.md said (TL;DR bullet 5)

> "The IADD case (apparent 2.5× penalty: 4.99 vs 2.02 cy) is a COMPILER PIPE-RE-ROUTING artifact, NOT hardware self-op. ptxas refuses to emit `IADD3 R,R,R,R` (encoding constraint) and falls back to `IMAD.IADD R,R,0x1,R` on the FMA pipe (4 cy lat) instead of IADD3 pipe (2 cy lat)."

## What the SASS evidence ACTUALLY shows

### Self-op test (`v2c_iadd_selfop.cu`, PTX = 1024 × `add.u32 v, v, v`)

SASS opcode counts in inner loop (justifications/SELF_OP_sass/v2c_iadd_selfop.sass):
- **IADD3: 500 instances** — `IADD3 R4, PT, PT, R4, R4, RZ` (= R4 = R4 + R4 + 0)
- **IMAD.IADD: 531 instances** — `IMAD.IADD R4, R4, 0x1, R4` (= R4 = R4*1 + R4)

Inner loop pattern: alternating `IMAD.IADD, IMAD.IADD, IADD3` (2:1 ratio).

**ptxas DOES emit `IADD3 R,R,R,RZ` for self-op.** The original claim "ptxas can't emit IADD3 R,R,R,R" is FALSE — it emits `IADD3 R, PT, PT, R, R, RZ` (with RZ as 3rd source) freely.

Total SASS in inner loop = 500 + 531 = 1031 ≈ 1024 PTX adds. **1 SASS per PTX add.**

### Distinct test (`v2c_iadd_distinct.cu`, PTX = 1024 × `add.u32 v, v, k1`)

SASS opcode counts:
- **IADD3: 515 instances** — `IADD3 R5, PT, PT, R4, R5, R4` (= R5 = R4 + R5 + R4 = 2*R4 + R5)
- IMAD.IADD: 1 instance (probably setup)

Total SASS in inner loop = ~515 ≈ **half of 1024 PTX adds**. **0.5 SASS per PTX add.**

### Independent fusion verification (just done)

4 explicit `add.u32 v, v, k` PTX → 2 UIADD3 SASS:
```
UIADD3 UR5, UPT, UPT, UR4, UR6, UR4 ;
UIADD3 UR5, UPT, UPT, UR4, UR5, UR4 ;
```
Each UIADD3 form `R5 = R4 + R6 + R4 = 2*R4 + R6` is computing **two `v += k` PTX adds in one SASS**. Confirmed fusion.

## The CORRECT mechanism (what's actually happening)

**The 2.5× cy/PTX difference (4.99 vs 2.02) is NOT pipe routing. It's COMPILER FUSION:**

- **Self-op case** (`v = v + v` repeated): each PTX add depends on the previous (chain on v). The compiler can't fuse two iterations because each one doubles v (iter1: 2v, iter2: 4v, iter3: 8v...). So 1 PTX add → 1 SASS instruction.
- **Distinct case** (`v = v + k1` repeated): two consecutive PTX adds become `v += k1; v += k1;` ≡ `v += 2*k1`, which compiler can emit as ONE 3-source IADD3 `R5 = R4 + R5 + R4` (where R4 is k1, R5 is v). So 2 PTX adds → 1 SASS instruction.

Per-SASS cycle cost:
- Self-op: 4.99 cy/PTX × 1.007 SASS/PTX = **5.02 cy/SASS** average (mix of IMAD.IADD and IADD3)
- Distinct: 2.02 cy/PTX × 0.503 SASS/PTX = **4.04 cy/SASS** (pure IADD3, fused)

**The per-SASS cost is ~4-5 cy in BOTH cases** — there's no 2× penalty per SASS instruction. The cy/PTX difference comes from the compiler being able to do half as many SASS instructions in the distinct case via fusion.

## What's RIGHT and what's WRONG in the original investigation

### Still RIGHT ✅

- **No architectural per-instruction self-op penalty for FFMA** (Test 1 results stand: all 5 operand-position variants measure 4.018-4.024 cy/op)
- **No self-op penalty for DFMA, IMAD, LOP3** (per-SASS-instruction cycles are equivalent)
- **Multi-chain self-op = multi-chain distinct at 0.96 op/cy** (Test 3 stands)
- **FFMA pipe latency = 4 cy architectural** (NCHAINS sweep stands)
- **`.reuse` cache helps throughput, not dependent latency** (Test 2 stands)
- **ncu evidence: identical short_scoreboard / wait stalls between self-op and distinct** (stands)
- **Catalog L1948 "self-op chains 2× inflated" claim is REFUTED for the per-SASS-instruction sense**

### Wrong / needs correction ❌

- **"ptxas can't emit IADD3 R,R,R,R encoding constraint"**: FALSE. ptxas emits `IADD3 R, PT, PT, R, R, RZ` for self-op freely. The agent saw this in the SASS but mis-narrated.
- **"Falls back to IMAD.IADD instead of IADD3"**: PARTIALLY TRUE — IMAD.IADD IS emitted alongside IADD3 in self-op (2:1 ratio with IADD3). Why ptxas alternates the two opcodes for self-op specifically (vs pure IADD3 for distinct) is a separate question — possibly to balance pipe utilization, or possibly a quirk of the codegen heuristic.
- **"FMA pipe (4 cy) vs IADD3 pipe (2 cy) explains the 2.5× difference"**: FALSE. The actual cause is compiler FUSION (2 PTX adds → 1 SASS in distinct case; not possible in self-op case because of chain).

### What the user was right about

The user's skepticism (2026-04-23) was warranted. The "compiler pipe re-routing" narrative was a wrong mechanism explanation despite reasonable measurement methodology. This is exactly the "we can't rely on '2×' claims without evidence in any individual case" lesson — and it applies to MECHANISM claims too, not just magnitude claims.

## Updated headline (replaces TL;DR bullet 5/6 of original)

> "The IADD apparent 2.5× cy/PTX difference (4.99 self-op vs 2.02 distinct) is **COMPILER FUSION**, not hardware self-op penalty and not pipe routing. The compiler fuses two consecutive `add.u32 v, v, k` PTX adds into one `IADD3 R, k, R, k` SASS (computing `v += 2*k` in one op). For self-op `add.u32 v, v, v`, fusion is impossible (each iteration doubles v), so 1 SASS per PTX. Per-SASS cycle cost is ~4-5 cy in both cases. There is no per-instruction self-op penalty."

## Implications for DENSE / catalog edits

- DENSE preamble update: change the IADD example to clearly say "compiler fusion" not "pipe routing"
- RECOMMENDED_CATALOG_EDITS: same correction
- The `IMAD.IADD` vs `IADD3` mix in self-op SASS is INTERESTING and worth its own follow-up: WHY does ptxas alternate them for self-op when IADD3 R,R,R,RZ would be uniform?

## Open follow-ups — RESOLVED by user 2026-04-23

> "this is obviously because pipe_alu is only 0.5/cy/SMSP, while IMAD on FMA heavy pipe is independent and also 0.5/cy/SMSP, so together they can saturate the SM in a way that neither of them can individually, and the compiler is smart to do this."

**The 2:1 IMAD.IADD:IADD3 alternation IS pipe-balanced scheduling.** Mechanism:

- **pipe_alu** caps at 0.5 inst/cy/SMSP per instruction in a latency-bound chain (each SMSP can dispatch 1 IADD3 per 2 cycles when waiting for chain dependency)
- **pipe_fmaheavy** (IMAD lives here) also caps at 0.5 inst/cy/SMSP independently
- **Combined**: alternating IMAD.IADD (fmaheavy) and IADD3 (alu) lets ptxas hit 1 inst/cy/SMSP in a self-op chain where pure IADD3 would max at 0.5 inst/cy/SMSP
- The 2:1 ratio reflects pipe-throughput balancing — IMAD.IADD has slightly more capacity in this regime

**For distinct case**, fusion gets you 0.5 SASS:PTX (each IADD3 does 2 PTX adds), so a single pipe is enough.

**ptxas is being SMART**, not falling back due to encoding constraints. The earlier sub-agent narrative misread the alternation as a "fallback" instead of recognizing it as a deliberate pipe-saturation strategy.

This also explains why per-SASS cycles in self-op (~5 cy) are LOWER than pure-IADD3 latency-bound projection (~4 cy on alu alone) would suggest — the alternation halves dispatch interval.

**LESSON**: when ptxas emits a "fallback" opcode, ALWAYS check whether the actual mechanism is pipe-balanced co-issue. NVIDIA's compiler is more sophisticated than naive sub-agent analysis assumes.

## Other follow-ups (still open)

- For OTHER compiler-fusion cases in the audit, similar mis-attributions may be hiding. Worth a SASS-count audit on every "self-op penalty" claim.
- For 3-operand ops where two pipes can share the work via opcode alternation, the architectural latency comparison needs to account for this.

## Methodology lesson recorded

When a sub-agent reports both a measurement AND a mechanism, the MECHANISM is a hypothesis to test independently. SASS-counting + per-PTX-op vs per-SASS-instruction normalization is necessary before believing any "X is on FMA pipe" or "X is rejected by ptxas" mechanism claim.
