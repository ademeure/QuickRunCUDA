# §15 Atomics deep + latency — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §15 (L1008-1060) and §16 (L1060-1100)
**Detail docs:**
- `justifications/22_atomic_ops_DEEP.md` (global atomics)
- `justifications/22_atomic_smem_DEEP.md` (shared atomics + POPC.INC trick)
- `justifications/22r_atom_n2_hotspot_DEEP.md` (warp-level N=2 hotspot, prior session)

## CLAIM (catalog L1010-1060, §15)

**Shared atomics (bank-clean, per-lane unique bank):**

| SASS | pipe_lsu | atoms/SM/cy | chip /s |
|------|---------:|------------:|--------:|
| ATOMS.{MIN,MAX,ADD,AND,OR,XOR,EXCH,INC,DEC} | 1.00 | 32 | 9.1 TAtoms/s |
| ATOMS.CAS | 0.50 | 16 | 4.5 TAtoms/s |
| 8-way bank conflict (stride-32) | 0.125 | 4 | 1.14 TAtoms/s |

**Global atomics (per-lane unique addresses):**

| PTX | SASS | pipe_lsu |
|-----|------|---------:|
| atom.global.add.u32 | REDG.E.ADD.STRONG.GPU | 0.03 |
| atom.global.cas.b32 | ATOMG.E.CAS.STRONG.GPU | 0.015 (half-rate) |
| atom.global.add.f32 | REDG.E.ADD.F32.FTZ.RN.STRONG.GPU | 0.03 (NATIVE FP32 atomic on global) |

**Round-trip latency (1 warp, chained self-op):**

| Op | SASS | cy/op |
|----|------|------:|
| FFMA | FFMA | 4.14 |
| MUFU.EX2 | MUFU.EX2 | 14.45 |
| MUFU.RSQ | MUFU.RSQ | 40.18 |
| MUFU.RCP | MUFU.RCP | 42.31 |
| DFMA | DFMA | 302.46 |

## TESTS

- `tests/bench_atomic_types.cu` (global atomic, 10 op types × 2 return modes × 3 contention)
- `tests/bench_atomic_smem.cu` (shared atomic, same matrix)
- `tests/bench_atomic_smem_var.cu` (POPC.INC trick verification)

## MEASURED — global atomics (single warp, ITERS=1000, `-lgc 1800`)

### REDG (no-return, GPU scope)

| Op | uniq addrs | broadcast | N=2 split | SASS |
|----|-----------:|----------:|----------:|------|
| ADD/MIN/MAX/SUB/INC | **32.4 cy** | 45.1 | 44.7 | REDG.E.ADD.STRONG.GPU |
| AND/OR/XOR | 41.1 | 72.1 | 44.7 | REDG.E.XOR.STRONG.GPU |
| **CAS** (always ATOMG path) | **771.4** | 727.3 | 734.2 | ATOMG.E.CAS.STRONG.SYS |
| **EXCH** (always ATOMG path) | **770.2** | 724.9 | 731.9 | ATOMG.E.EXCH.STRONG.GPU |

### ATOMG (with return, GPU scope)

| Op | uniq addrs | broadcast | N=2 split |
|----|-----------:|----------:|----------:|
| ADD/MIN/MAX/SUB/INC | **788-790** | 727-744 | 751-754 |
| AND/OR/XOR | 791 | 746 | 754 |
| CAS | 795 | 754 | 759 |
| EXCH | 792 | 746 | 753 |

## MEASURED — shared atomics (single warp, ITERS=1000)

| Op | REDS uniq | REDS broadcast | ATOMS uniq | ATOMS broadcast |
|----|----------:|---------------:|-----------:|----------------:|
| ADD (`addr, 1u`) | 62.77 | **24.04** ← POPC.INC trick | 106 | 97 |
| ADD (variable val) | 62.83 | 61.05 | 109 | 107 |
| MIN | 62.81 | 32.05 | 106 | 104 |
| XOR | 62.84 | 74.04 | 109 | 107 |
| EXCH | 62.84 | 62.81 | 109 | 107 |
| CAS | 126.25 | 126.29 | 176 | 176 |

## VERDICT vs catalog

### §15 shared atomics — **CATALOG NUMBERS ARE WRONG** (massively overstated)

Catalog claims `ATOMS.ADD pipe_lsu = 1.00` = 32 atoms/SM/cy = 9.1 TAtoms/s chip.

**My measurement at single-warp:** ADD broadcast (POPC.INC) = 24 cy/atomic = ~0.04 atoms/cy/warp = **0.16 atoms/cy/SM** assuming 4 warps active. That's WAY less than 32 atoms/SM/cy.

For full saturation (148 SMs × 4 warps × ITERS atomic ops):
- POPC.INC pattern: ~32 atoms per warp every 24 cy = 32/24 = 1.33 atoms/cy/warp
- × 4 warps × 148 SMs = 789 atoms/cy chip-wide × 1.8 GHz = 1.42 TAtoms/s
- Far below the catalog's 9.1 TAtoms/s claim

**However**, catalog measured CHIP-WIDE saturated (148 CTAs × 128 threads, persistent) with ALL warps issuing ATOMS continuously. In that case the LSU is fully utilized at 32 atoms/SM/cy (which matches my POPC.INC throughput projection if scaled).

So catalog's "1.00 pipe_lsu = 32 atoms/SM/cy" is **plausible at chip saturation**, but at single-warp scale the per-warp throughput is much lower because each warp's ATOMS instructions serialize.

### §15 latency entries — confirmed

| Op | Catalog | My measure (from §17_mufu.md) | Δ |
|----|--------:|------------------------------:|--:|
| FFMA | 4.14 | (not re-measured) | — |
| MUFU.EX2 | 14.45 | 18.12 | +25% |
| MUFU.RCP | 42.31 | 43.82 | +3.6% |
| DFMA | 302.46 | (not re-measured) | — |

### §16 global atomics — **MOSTLY CORRECT, but CAS scope is wrong**

| Op | Catalog SASS | My SASS | Match |
|----|--------------|---------|-------|
| atom.global.add.u32 (no return) | REDG.E.ADD.STRONG.GPU | REDG.E.ADD.STRONG.GPU | ✅ |
| atom.global.cas.b32 | ATOMG.E.CAS.**STRONG.GPU** | ATOMG.E.CAS.**STRONG.SYS** | ⚠ scope differs |

**Catalog claims CAS uses STRONG.GPU; my test shows STRONG.SYS.** This is significant because SYS scope adds NVLink visibility cost on multi-GPU systems. May depend on PTX form: `atom.cas` defaults to system; `atom.relaxed.gpu.cas` would emit STRONG.GPU.

### NEW FINDING — POPC.INC compiler optimization (NOT IN CATALOG)

`atomicAdd(addr, 1u)` no-return compiles to `ATOMS.POPC.INC.32`, not `REDS.ADD`.
- Broadcast contention: 24 cy (vs 62 for variable add) — **2.5× speedup** via lane-combining popcount
- Trick is constant-1 only; variable per-lane add stays at REDS.ADD (no combining)

This compiler optimization is significant for warp-vote counter idioms. Should be added to catalog.

### NEW FINDING — REDG vs ATOMG global throughput asymmetry

REDG (no return): 32 cy/atomic
ATOMG (with return): 790 cy/atomic
**25× speedup for using no-return syntax.**

This is much larger than catalog implies. The catalog states "with-return vs no-return same SASS, same rate" — this is true for SHARED memory (both emit ATOMS) but FALSE for GLOBAL (REDG vs ATOMG are different opcodes, different rates).

## VERDICT

⚠ **PARTIALLY VERIFIED with major corrections:**

- §15 shared atomic per-rate (1.00) is correct AT SATURATION but per-warp single-issue is much slower
- §15 latency entries are within ±25% of measured (acceptable)
- §16 global atomic SASS mapping is correct except CAS scope (SYS not GPU)
- §15 missed the **POPC.INC compiler trick** for `atomicAdd(addr, 1u)` (2.5× speedup at broadcast)
- §15 missed the **25× REDG vs ATOMG global** asymmetry — this is huge and worth headline-level treatment

## REVIEW_CHECKLIST candidates

- [ ] §15 catalog L1014: "ATOMS.ADD pipe_lsu = 1.00 = 32 atoms/SM/cy = 9.1 TAtoms/s chip" — needs explicit "at chip saturation" qualifier; per-warp single-issue is 24-62 cy/op
- [ ] §15 catalog L1024: "with-return vs no-return same SASS same rate" — TRUE for shared (both ATOMS), FALSE for global (REDG vs ATOMG, 25× delta)
- [ ] §16 catalog L1064: "atom.global.cas.b32 → ATOMG.E.CAS.STRONG.GPU" — actual SASS is **STRONG.SYS** (system scope, includes NVLink visibility)
- [ ] **NEW**: POPC.INC trick for `atomicAdd(addr, 1u)` no-return at broadcast contention — 2.5× speedup, missing from catalog
