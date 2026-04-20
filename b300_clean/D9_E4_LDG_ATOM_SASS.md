# LDG variants & atom/red SASS encoding — V4 / D9 + E4

**Date: 2026-04-20.** Test `bench_ldg_atom_sass.cu`. Compares the SASS
emission for various PTX/CUDA load and atomic intrinsics on B300.

## LDG variants — DIFFERENT encodings

| PTX/CUDA | SASS | Cache scope |
|----------|------|-------------|
| `__ldg(p)` (CUDA intrinsic) | `LDG.E.CONSTANT` | Constant cache (read-only) |
| `ld.global.ca` (PTX) | `LDG.E.STRONG.SM` | SM-strong (cache-all) |
| `ld.global.cg` (PTX) | `LDG.E.STRONG.GPU` | GPU-strong (cache global) |
| `ld.global` (default) | `LDG.E` | No scope qualifier |

**`__ldg` ≠ `ld.global.ca`!** They emit DIFFERENT SASS:
- `__ldg` uses `LDG.E.CONSTANT` — tells the L1/L2 to use the constant
  cache path; optimized for read-only data
- `ld.global.ca` uses `LDG.E.STRONG.SM` — standard cache-all with
  SM-scope ordering

## Practical implication

For READ-ONLY data (model weights, lookup tables, prefilled buffers):
- **Prefer `__ldg`** — uses constant cache path, can be more efficient
- `const __restrict__ T*` parameters trigger compiler to emit `__ldg` automatically
- `ld.global.ca` and plain `ld.global` are NOT equivalent to `__ldg` in SASS

## Performance verified: __ldg vs ld.global.ca = SAME SPEED

Re-measured per `bench_ldg_constant_vs_strong.cu` (16 KB hot region):

| Variant | Time | vs default |
|---------|------|------------|
| `__ldg` (LDG.E.CONSTANT) | 1.216 ms | 1.00× |
| `ld.global.ca` (LDG.E.STRONG.SM) | 1.216 ms | 1.00× |
| `ld.global` (LDG.E) | 1.216 ms | 1.00× |
| **`ld.global.cg` (LDG.E.STRONG.GPU)** | **2.719 ms** | **2.24× SLOWER** |

**Surprise**: despite different SASS, `__ldg` is NOT measurably faster
than `ld.global.ca` for this workload. The "constant cache hint" doesn't
translate to a measurable L1-hot win. The hint may matter only for
specific tensor-core-coupled workloads.

**Big lesson**: avoid `ld.global.cg` (= bypass L1) unless you really
want L2-only access — 2.24× slower at L1-resident workloads.

For READ-WRITE data:
- Use `ld.global` (no qualifier) or `ld.global.ca`
- DO NOT use `__ldg` (inconsistency with later writes)

## Atom/Red variants

| PTX/CUDA | SASS |
|----------|------|
| `atomicAdd(p, v)` (returns old value) | `ATOMG.E.ADD.STRONG.GPU PT, R3, ..., R7` |
| `atom.global.add` (PTX, returns) | `ATOMG.E.ADD.STRONG.GPU PT, R3, ..., R7` |
| `red.global.add` (no return) | `REDG.E.ADD.STRONG.GPU` |
| `red.relaxed.gpu.global.add` | `REDG.E.ADD.STRONG.GPU` (identical) |

**`atom` and `red` use DIFFERENT SASS opcodes:**
- `ATOMG` = atomic-with-return
- `REDG` = atomic-without-return

`red.relaxed.gpu` and `red.global` produce IDENTICAL SASS on B300. The
"relaxed" memory ordering doesn't change the SASS — it would only
matter if surrounded by additional fences (MEMBAR), which neither does
here.

Existing finding (commit `9467cfe`): `red.release.gpu.global` adds a
MEMBAR.ALL.GPU before each red — that's where the ordering enforcement
happens, NOT in the REDG instruction itself.

## Recipe summary

| Want | Use |
|------|-----|
| Read-only data (weights) | `__ldg` or `const __restrict__` parameter |
| Read-write data (cache normally) | `ld.global` (default) |
| Bypass L1 (only L2) | `ld.global.cg` |
| Atomic with return | `atomicAdd` / `atom.global` |
| Atomic without return (faster) | `red.global.add` |
| Released atomic (with global ordering) | DON'T — adds MEMBAR.ALL.GPU = 9× slower |

## Confidence

- **HIGH** for SASS encoding differences (verified via direct SASS dump)
- **HIGH** for `red.relaxed = red.global` SASS-identical
- **MED** for "constant cache more efficient" — depends on workload;
  could verify with ncu `lts__t_sectors_op_read`

## Files

- `tests/bench_ldg_atom_sass.cu` — OP 0-7 (all SASS-only test)
