# §30B FOLLOWUP — atom→SASS mapping is NOT universal REDG

**Date:** 2026-04-23
**Method:** main-session direct grep across all preserved SASS files in `/root/github/QuickRunCUDA/sass/`
**Triggered by:** rigor pass on existing `30B_atomics.md`

## What §30B claimed (potential overstatement)

The original `justifications/30B_atomics.md` stated as a "KEY FINDING (SASS-verified)":

> "`atom.global.add.u32` compiles to **`REDG.E.ADD.STRONG.GPU`**, NOT `ATOMG`. The compiler uses the REDG (reduction) family **even when a return value is requested**."

This is reproduced in the catalog edit recommendations and DENSE §13.

## What direct SASS evidence shows

Counting opcodes across **20,192 SASS files** in `/root/github/QuickRunCUDA/sass/`:

| Opcode | Total occurrences |
|---|--:|
| ATOM.E (variants) | 318 |
| ATOMG.E (variants) | 4,976 |
| REDG (variants) | 1,869 |
| ATOMS (shared variants) | many |

Per-kernel SASS (sample):

| Kernel SASS file | Atomic opcodes emitted |
|---|---|
| `bench_atom_chip_scope_*.sass` (the §30B chip-throughput test) | **64 × `ATOM.E.ADD.STRONG.GPU`** ← NOT REDG, NOT ATOMG |
| `bench_atom_lat_*.sass` | ATOMG.E.ADD.STRONG.GPU + ATOMG.E.CAS + ATOMG.E.ADD.STRONG.SYS + ATOMS.ADD |
| `bench_atom_chain_1thread_*.sass` | 1 × ATOMS.ADD (shared mem only) |
| `atomics_2942806259.sass` (specific PTX variant 0) | **16 × ATOMG.E.ADD.STRONG.GPU** ← ATOMG, not REDG |
| `atomics_2942806261.sass` (specific PTX variant 2) | **16 × REDG.E.ADD.STRONG.GPU** ← REDG |
| `atomics_2942806260.sass` (specific PTX variant 1) | 16 × ATOMG.E.ADD.STRONG.SYS |
| `atomics_2942806263.sass` (CAS variant) | 16 × ATOMG.E.CAS.STRONG.GPU |
| `atom_global_variants_2942806260.sass` | 1 REDG + 1 ATOMG ← MIXED |

## Inferred pattern (still partial)

The compiler emits ONE of three SASS opcodes for the same source `atomicAdd(...)`:
- **REDG.E.ADD** when the return value is discarded (semantically `red.add` not `atom.add`)
- **ATOMG.E.ADD** when the return value is used and `cuda::atomic` semantics with default scope
- **ATOM.E.ADD** for some scoped or address-pattern variants (haven't fully isolated)

The choice depends on:
- Whether the return value of `atomicAdd` is captured/used
- The PTX scope qualifier (`.cta`, `.gpu`, `.sys`)
- Possibly whether the address is uniform across the warp

## Correction to §30B finding

The original "atom.global.add.u32 → REDG.E.ADD.STRONG.GPU, NOT ATOMG" claim is **PARTIALLY WRONG / OVERSTATED**. It is true that:

- ✅ Some PTX forms of `atom.global.add` DO emit `REDG.E.ADD` (specifically `red.add` PTX without value capture, and possibly `atom.add` without value capture under default scope)
- ✅ The ncu metric implication still holds for those cases (use `lts__t_sectors_op_red` not `_op_atom`)

But it is NOT universally true:
- ❌ `bench_atom_chip_scope` (the chip-wide test that produced the headline 49.1 Gops/s measurement) emits **`ATOM.E.ADD.STRONG.GPU`** — not REDG
- ❌ `bench_atom_lat` emits ATOMG.E variants

## Implications for DENSE / JUSTIFIED

1. **REDG vs ATOMG vs ATOM.E distinction in catalog § / DENSE § needs nuance:**
   - Compiler emits one of three based on return-value usage and scope
   - The "use lts__t_sectors_op_red" advice is correct WHEN the kernel emits REDG, but if the kernel emits ATOMG/ATOM.E, you also need `lts__t_sectors_op_atom`
   - Best practice: capture BOTH ncu metrics and add them

2. **The §30B chip-wide throughput numbers (49.1 / 53.7 / 609 / 221 Gops/s) used `bench_atom_chip_scope` which emits ATOM.E.ADD.STRONG.GPU.** So the §30B numbers themselves are still valid — they were just measured against a DIFFERENT SASS opcode than the agent claimed. The throughput conclusions stand; the SASS-name attribution was wrong.

3. **The `atom.f16/bf16 → ATOM.E.CAS.STRONG.GPU loops` finding IS confirmed** by direct SASS count (REDG.E.ADD.F16x2 for packed; ATOM.E.CAS for scalar). That part of §30B is correct.

## What still needs investigation

- Why does `bench_atom_chip_scope` emit `ATOM.E.ADD.STRONG.GPU` while `bench_atom_clean` (which I would have expected to do the same thing) emits something different?
- What's the exact source-code pattern that makes ptxas choose REDG vs ATOMG vs ATOM.E?
- If a reader uses the §30B catalog edit and encounters ATOMG SASS in their own kernel, will the lts__t_sectors_op_red metric still capture the throughput?

These are open. But the CRITICAL point is: **the universal "atom.global.add → REDG" claim is too strong**. Both DENSE and JUSTIFIED need to be qualified.

## Action items

- [x] DENSE §13 footnote: qualify the "atom→REDG" claim
- [x] JUSTIFIED §30B index entry: link to this followup
- [ ] Future test: run `bench_atom_chip_scope` and `atomics_2942806261` side by side, verify both with `lts__t_sectors_op_atom` AND `lts__t_sectors_op_red`, see which counter catches what
- [ ] Future test: vary source code patterns (capture vs discard, scope qualifier, address pattern) to map source → SASS comprehensively
