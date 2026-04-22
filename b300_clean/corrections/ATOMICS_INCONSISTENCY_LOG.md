# Atomics inconsistency log

Each entry: claim — file:line — conflicting source — resolution.

## A1. Pipelined atomic latency: 16 cy vs 43 cy
- **Claim A**: V10_SMEM_ATOMIC.md:54 — `Pipelined throughput = 16 cy/op (V9)`
- **Claim B**: V9_ATOMIC_LATENCY.md:46 — `Pipelined throughput (independent
  atomics): ~43 cy effective at SM`
- **Claim C**: V10_SMEM_ATOMIC.md:59 — `V9 16 cy → SMEM 10 cy`
- **Resolution**: V9 is the source-of-truth. V10_SMEM mis-cited. The "16 cy"
  in V10_SMEM appears nowhere in V9. Memory note repeats V10_SMEM's error.
  → Use **43 cy pipelined** (single-thread, indep ops) and **~10 cy effective**
  for SMEM ATOMS at warp full-rate.

## A2. SMEM atomic peak throughput: 2.27 T vs 4.2 T
- **Claim A (catalog)**: V10_SMEM_ATOMIC.md:10-16 — `2.15-2.27 T atomic/s
  aggregate, 15 G atomic/s per SM`
- **Claim B (memory)**: feedback note `SMEM atomic 4.2 Tops/s no-contention`
- **Resolution**: Memory note unsourced; not present in any clean file.
  Possibly from different op (FP vs INT), different occupancy, or per-clock
  vs per-second confusion. → Use **2.27 T atomic/s** (V10_SMEM, INT32).

## A3. ATOMS pure latency: 107 → 45 cy (memory) vs 4.6 cy (02_shmem)
- **Claim A (memory)**: `ATOMS pure latency 107 → 45 cy (isolated single-thread)`
- **Claim B (catalog)**: 02_shmem.md:162 — `INT32 atomicAdd: 4.6 cy
  no-contention`
- **Resolution**: Numbers measure different things. 4.6 cy is in-warp ATOMS
  throughput; 107/45 may be a cross-clock chain measurement (e.g. 5 µs / 47
  cy at 1860 MHz video clock?). Memory note has no in-tree source. → Re-test
  or strike from memory.

## A4. Atomic peak Gops/s: 449 vs 504 vs 1005
- **Claim A**: B300_TRUE_REFERENCE §5 row — `Stride 4 = 449 Gops/s | peak`
- **Claim B**: 07_atomics §8 — `stride 4B UNROLL=16 = 504 Gops/s`
- **Claim C**: 07_atomics §8 — `True peak: 1005 Gops/s (UNROLL=32)`
- **Claim D**: 07_atomics §7 — `stride 4 = 7.2 Gatomic/s (148×128 thr × 1000)`
- **Resolution**: All correct at different ILP/UNROLL settings. TRUE_REFERENCE
  should cite 1005 Gops/s with `(UNROLL=32, L2-resident)` qualifier and
  retire the bare "449 Gops/s peak" framing. §7 number (7.2) is a
  low-UNROLL artifact in the same file as §8 — confusing.

## A5. L2 atomic units count: ~32 plateau
- **Claim A**: TRUE_REFERENCE row 162 — `L2 atomic units count = ~32`,
  listed as a verified "Counterintuitive finding".
- **Claim B**: ATOMIC_REVERIFY_DEEP.md:53-66 — `"32 L2 atomic units" claim
  is likely wrong on B300 ... ceiling could be MUCH HIGHER than 32`.
- **Resolution**: TRUE_REFERENCE overstates confidence. → Demote to LOW /
  OPEN; per memory's "Dispatch ceiling skepticism" note this kind of
  inferred-from-plateau number is exactly what we should not trust.

## A6. Per-warp anti-pattern slowdown range: 5× vs 12×
- **Claim**: 07_atomics §6 — `5-12× SLOWER than per-CTA`
- **Issue**: Range is too wide (factor of 2.4×) for an "H" confidence row.
  V10_GLOBAL doesn't characterize per-warp explicitly.
- **Resolution**: Mark per-warp slowdown as M (single test) until a
  dedicated sweep narrows the range.

## A7. Combining inflates Gops/s without proportional bytes — guard against
- **Demonstrated**: ATOMIC_LADDER_RIGOROUS — combine=32 takes Gops/s from
  49.7 → 769 (15.5×) while DRAM B/s stays at ~5.5 → 4.0 TB/s (UNCHANGED to
  20% lower). VERSION A reverify pushes Gops to 1230 with DRAM at 80 GB/s.
- **Memory feedback**: `cache-line combining can inflate Gops 8× without
  proportional BW; got '28× ratio' wrong by mixing combined+uncombined`.
- **Rule**: every Gatomic/s number in this catalog must publish (combine,
  WS, L2-resident, DRAM B/s). 07_atomics §6 partially does this; §8 does;
  §7 does NOT. TRUE_REFERENCE §5 does NOT.

## A8. Local atomic L2 round-trip: 164 ns (TRUE_REFERENCE) vs 343 ns (V9)
- **Claim A**: TRUE_REFERENCE row 86 — `Local atomic L2 round-trip = 164 ns`
- **Claim B**: V9_ATOMIC_LATENCY:14-20 — `697 cy / 343 ns @ 2.032 GHz`
- **Claim C**: 07_atomics §1 — `~310 cy near-L2 / ~680 cy far-L2`
- **Resolution**: 164 ns matches **near-L2 round-trip without dependency
  chain forced through return** (~333 cy). 343 ns is **dependency-chained
  RT** (697 cy). Both correct, different definitions. TRUE_REFERENCE row
  needs annotation `(no chain, near-L2)`; otherwise readers will conflate
  it with chained.

## A9. Cross-GPU atomic LOCAL/REMOTE — CONSISTENT
- 07_atomics §11: LOCAL contended 49 G, REMOTE 16 G; LOCAL unique 137 G,
  REMOTE 9 G.
- 12_nvlink_p2p §5a: LOCAL all-contend 49.4 G, REMOTE 16.6 G; unique
  REMOTE 9.2 G.
- Memory: `49 Gatomic/s LOCAL all-contend, 16 Gatomic/s REMOTE`.
- All three sources agree to within 1%. NO inconsistency.

## A10. red.global retraction — CONSISTENT but watch the wording
- 07_atomics §9 and §12: `red.global is 100× SLOWER`.
- TRUE_REFERENCE row 128, 155: `red.release.gpu.global = 614 ns/op,
  9.1× SLOWER than red.global`.
- TRUE_REFERENCE row 154: `NO CCTL.IVALL emitted for any red.global variant`.
- 07_atomics §9 attributes the 100× slowdown to `CCTL.IVALL inserted between
  every instruction`. TRUE_REFERENCE explicitly says CCTL.IVALL does NOT
  exist on B300. → 07_atomics §9 cause-attribution is **likely wrong**;
  the correct cause is MEMBAR.ALL.GPU on the release-ordered variant.
  Need to clarify which red.global variant is being timed (relaxed vs
  release).
