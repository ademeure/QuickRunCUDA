# Atomic op cost characterization on B300 sm_103a

## Test (`tests/bench_atomic_types.cu`, 1 CTA, 32 threads, ITERS=1000)

Single-warp, no cross-CTA contention. Measures cy/atomic per warp.

## Results

### REDG (no-return, compiler emits REDG.E.X.STRONG.GPU when possible)

| Op | uniq addrs | warp-broadcast | warp N=2 split |
|----|-----------:|---------------:|---------------:|
| ADD | **32.4** | 45.1 | 44.7 |
| MIN/MAX/SUB/INC | **32.4** | 45.1 | 44.7 |
| AND | 41.1 | 72.1 | 44.7 |
| OR  | 41.1 | 72.1 | 44.7 |
| XOR | 41.1 | 72.1 | 44.7 |
| **CAS** | **771.4** | 727.3 | 734.2 |
| **EXCH** | **770.2** | 724.9 | 731.9 |

### ATOMG (with-return, ATOMG.E.X.STRONG.GPU)

| Op | uniq addrs | warp-broadcast | warp N=2 split |
|----|-----------:|---------------:|---------------:|
| All ADD/MIN/MAX/SUB/INC | **788-790** | 727-744 | 751-754 |
| AND/OR/XOR | 791 | 746 | 754 |
| CAS | 795 | 754 | 759 |
| EXCH | 792 | 746 | 753 |

## Key findings

1. **REDG is 16-25× faster than ATOMG** (32 cy vs 790 cy). If you don't
   need the return value, use the no-return syntax — the compiler emits
   REDG.E which can use L2 cache-line combining and avoid the response
   round-trip.

2. **CAS/EXCH always pay the round-trip** (~770 cy). Even with no-return
   syntax, these ops can't be REDG-encoded because their semantics require
   read-compare-write or read-swap. There's no compiler optimization that
   helps here.

3. **AND/OR/XOR REDG is slightly slower** than ADD/MIN/MAX (41 vs 32 cy).
   Possibly handled by a different L2 atomic unit family.

4. **Warp-broadcast contention adds modest cost** for REDG (32 → 45 cy =
   +40%), even though all 32 lanes hit the same address. The L2 atomic
   unit serializes the 32 requests but has fast lane-combining for ADD.

5. **For AND/OR/XOR, broadcast cost is ~2× uniq** (72 vs 41 cy) — the
   bitwise ops don't combine as efficiently.

6. **Warp N=2 split (CONTENTION=2) shows NO penalty** at single-warp scale
   (~44 cy). The §22r 34× hotspot (warp-level N=2 with many CTAs) requires
   high concurrent CTA count to manifest — it's a cross-warp concurrency
   effect at L2 atomic units, not a single-warp pattern.

## Recommendations

1. **Always prefer no-return atomic syntax** if return value isn't used.
   25× speedup is huge.
2. **Replace `atomicAdd(&counter, 1)` with `atomicAdd(&counter, 1u);` (no
   assignment)** — compiler emits REDG, ~32 cy instead of ~790 cy.
3. **For shared counters / histograms**: REDG.ADD is the fastest atomic.
4. **Avoid CAS/EXCH** unless absolutely necessary (lock-free queues etc).
   They cost as much as full ATOMG round-trip.
5. **Lock-free queue alternatives**: instead of atomicCAS-based MPMC queues,
   consider atomicAdd-based slot-counter approaches that can use REDG.

## Caveat: scale matters

This test uses 1 CTA × 32 threads. At full grid (148 CTAs × 128 threads),
contention dynamics change drastically. See `justifications/22r_atom_n2_hotspot_DEEP.md`
for the warp-level N=2 = 34× slowdown at high concurrency.

The relevant rule of thumb:
- **At low concurrency**: REDG is dominated by issue rate (~32 cy/op).
- **At high concurrency**: REDG is dominated by L2 atomic unit contention.
  Multiple distinct addresses (≥8) help; warp-level N∈{2,3,4} hurts due
  to broken intra-warp combining.

---

## SASS verification

| PTX form | SASS emit | Scope |
|----------|-----------|-------|
| `atomicAdd` no-return | **REDG.E.ADD.STRONG.GPU** | GPU |
| `atomicAdd` with return | **ATOMG.E.ADD.STRONG.GPU** | GPU |
| `atomicXor` no-return | **REDG.E.XOR.STRONG.GPU** | GPU |
| `atomicCAS` (any) | **ATOMG.E.CAS.STRONG.SYS** | **SYSTEM** (NVLink-visible) |

### CAS scope insight

`atomicCAS` compiles to `ATOMG.E.CAS.STRONG.SYS` — **system-wide scope**,
not GPU. This is because PTX `atom.cas` defaults to system scope for
strong-ordering guarantees needed for lock-free MPMC patterns to work
across the entire system (multi-GPU NVLink coherence).

The cost difference (CAS ~770 cy vs ADD ATOMG ~790 cy) is small at single-
warp scale because both are dominated by L2 atomic-unit response latency.
But at NVLink-coherent scale (multi-GPU), CAS would pay an additional
NVLink visibility cost similar to `release.sys` (1663 cy intrinsic per
ADDENDUM 9 of cctl_ivall_DEEP).

### Implication

For single-GPU code, you can manually use lower-scope CAS via PTX:
```ptx
atom.global.cas.b32 %0, [%1], %2, %3;       // implicit system scope
atom.relaxed.gpu.global.cas.b32 ...;        // GPU scope (cheaper)
atom.relaxed.cta.global.cas.b32 ...;        // CTA scope
```

The CUDA C++ `atomicCAS` always emits SYS-scope. For tighter scopes, drop
to inline PTX with explicit `.gpu` or `.cta` modifiers — saves potential
NVLink visibility cost in production multi-GPU contexts.
