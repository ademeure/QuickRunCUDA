# 07_atomics — CORRECTED REFERENCE (atomics swarm review)

**Source files reviewed:** `07_atomics.md`, `ATOMIC_LADDER_RIGOROUS.md`,
`ATOMIC_REVERIFY_DEEP.md`, `V10_GLOBAL_ATOMIC.md`, `V10_SMEM_ATOMIC.md`,
`V9_ATOMIC_LATENCY.md`, `B300_TRUE_REFERENCE.md`, plus `02_shmem.md` and
`12_nvlink_p2p.md` cross-references.

**Hardware:** B300 SXM6 (sm_103a), 148 SMs, 2.032 GHz boost, 126 MB L2.

---

## 1. Verified single-thread latency (1 warp, dependency chain, hot location)

| Op | Path | Latency | Source | Conf |
|---|---|---:|---|---|
| Global atom.{cta,gpu,sys}.add.u32 (chained) | REDG round-trip | ~697 cy / 343 ns | V9 | H |
| Global atom.relaxed.gpu (per-thread addr, near-L2) | full RT | ~310 cy | 07_atomics §1 | H |
| Global atom.relaxed.gpu (per-thread addr, far-L2) | full RT | ~680 cy | 07_atomics §1 | H |
| Shared atom.relaxed.cta INT32 add | ATOMS | 4.6 cy (no contention), 4.6 cy (32-way) | 02_shmem §SHMEM atomics | H |
| Shared atom.{cluster,gpu}.release/acq_rel | +MEMBAR.ALL.GPU | +260 cy over relaxed | 07_atomics §3 | H |
| Global atom.{cluster,gpu}.release/acq_rel | +MEMBAR.ALL.GPU + ERRBAR + CGAERRBAR | +1040 cy | 07_atomics §3 | H |

**Local atomic L2 round-trip** (B300_TRUE_REFERENCE row 86): 164 ns.
**Cross-GPU via NVLink** (TRUE_REFERENCE row 87): 1,662 ns.

## 2. Verified pipelined throughput (independent atomics, single thread)

| Op (u32) | cy/op | ns | Source |
|---|---:|---:|---|
| atomicInc | 7.9 | 3.9 | 07_atomics §1 |
| atomicDec | 7.0 | 3.4 | 07_atomics §1 |
| atomicAdd / Sub | 15.2 | 7.5 | 07_atomics §1 |
| atomicMin / Max | 15.7 | 7.7 | 07_atomics §1 |
| atomic And/Or/Xor | 23.5 | 11.6 | 07_atomics §1 |
| atomicExch | 49.5 | 24.4 | 07_atomics §1 |
| atomicCAS | 52.5 | 25.9 | 07_atomics §1 |

V10_SMEM `~10 cy effective per atomic at full warp` is consistent with V9
(~16 cy pipelined throughput) within methodology variance.

## 3. Verified throughput (full chip, paired with bytes/s)

| Pattern (atom.global.add.u32) | Gatomic/s | DRAM B/s | Payload B/s | Source |
|---|---:|---:|---:|---|
| Stride 128 B per thread, no combine | 49.7 | 5.52 TB/s | 199 GB/s | ATOMIC_LADDER_RIGOROUS §CASE1 |
| Stride 128 B per thread, uint64 | 49.8 | 5.52 TB/s | 398 GB/s | §CASE2 |
| b128 exch, stride 128 B | 42.2 | 4.64 TB/s | 676 GB/s | §CASE3 |
| Combine=32 (lane=offset, full L2 reuse, WS=32MB) | 1230 | 80 GB/s | 4.93 TB/s | REVERIFY VERSION A |
| Combine=32 (lane=offset, WS=1024MB) | 768 | 4.03 TB/s | 3.07 TB/s | REVERIFY VERSION B |
| Stride 4 B (32 per CL, L2-resident) | 504-1005 | n/a | (peak quoted UNROLL=32) | 07_atomics §8 |

**Universal atomic DRAM ceiling: ~5.5 TB/s** (~75% of HBM raw 7.31).

## 4. SMEM atomic — contention-invariant (V10_SMEM)

| CONTEND | Aggregate atomic/s |
|---:|---:|
| 1 | 2.15 T |
| 2-256 | 2.23-2.27 T |

Per SM: ~15 G atomic/s. **Memory's `4.2 Tops/s no-contention` claim is NOT
present in any reviewed file** — closest verified value is **2.27 T atomic/s**
(V10_SMEM, INT32 ATOMS). See UNRESOLVED.

## 5. Cross-GPU atomic (NVLink, 12_nvlink_p2p §5)

| Pattern | LOCAL | REMOTE |
|---|---:|---:|
| All-contend (warp-uniform) | **49.4 Gatomic/s** | **16.6 Gatomic/s** |
| Unique addresses | 137 G | 9.2 G |
| Single-thread RT | 354 ns | 1,800 ns |

These match memory's "49 G LOCAL / 16 G REMOTE" — consistent across
07_atomics §11 and 12_nvlink_p2p §5a.

## 6. RETIREMENT carried over from 07_atomics §12

- `red.global` is 9.1× SLOWER than `atom.global` (CCTL.IVALL/MEMBAR side-effects).
- `2.7× speedup from coalescing` was misattribution → real 43× from L2-vs-DRAM.
- `acquire.gpu = 780 cy / 17× relaxed` was warp-serialization artifact.
- `atom round-trip 137 cy` was L2 cache latency, not atomic.

---

## RETRACTIONS (originals are wrong / contradict verified data)

1. **B300_TRUE_REFERENCE §5 row "Stride 4 (cache-line combining) | 449 Gops/s | peak"**
   conflicts with `07_atomics.md §8` which states **true peak = 1005 Gops/s
   (UNROLL=32, stride=4B)** and explicitly retires the 372 / 449 number as
   "lower-ILP" measurement. TRUE_REFERENCE should cite 1005 Gops/s with the
   ILP qualifier, OR add caveat that 449 is at default ILP.

2. **B300_TRUE_REFERENCE row "L2 atomic units count = ~32"** is contradicted by
   `ATOMIC_REVERIFY_DEEP.md`: the "32 units" is INFERRED from a stride-sweep
   plateau, not measured; reverify shows VERSION A reaches 20.4 L2 packets/cy
   suggesting ceiling could be MUCH higher. Per memory's "Dispatch ceiling
   skepticism": this should be flagged LOW confidence in TRUE_REFERENCE, not
   listed as a "Counterintuitive finding" with implied verification.

3. **V10_SMEM_ATOMIC §"Comparison to V9"** says
   `V9 found global atomic chained latency = 697 cy ... Pipelined throughput =
   16 cy/op (V9)`. V9_ATOMIC_LATENCY actually says **43 cy** pipelined
   (line 46), not 16 cy. The 16 cy in V10_SMEM appears to be a **typo /
   confusion with the SMEM ~10 cy number**. Memory note "atomic 697 cy
   chained, 16 cy pipelined" repeats this same V10_SMEM error — should be
   **43 cy pipelined** per the underlying V9 measurement.

4. **07_atomics §6 contention table** lists "All threads → A[0]: 27-49 Gops/s"
   while V10_GLOBAL contention=1 gives **50 Gops/s** — consistent at upper
   end. But §6 also says "Per-CTA address: 38-89 Gops/s" while V10_GLOBAL
   does not show this configuration. NOT a contradiction, just under-tested
   in V10.

5. **CLAUDE.md memory note "ATOMS pure latency 107 → 45 cy (isolated single-thread)"**
   does NOT appear in any reviewed file. Closest data: 02_shmem reports INT32
   smem `atomicAdd = 4.6 cy` (single warp, clock64). The "107 → 45" pair is
   not corroborated here — likely refers to a different earlier measurement
   not in the clean catalog. Flag for memory cleanup.

6. **CLAUDE.md memory note "SMEM atomic 4.2 Tops/s no-contention"** is NOT
   reproduced. V10_SMEM measures 2.15-2.27 T atomic/s (close to 2× lower).
   The 4.2 T figure may originate from a per-SM-times-148 calculation, or a
   separate FP-vs-INT distinction. Does not appear in the clean catalog and
   should not be quoted without source.

---

## UNRESOLVED

- **L2 atomic-unit count.** "~32" is a plateau-derived inference; ATOMIC_REVERIFY
  explicitly states the ceiling could be much higher and the catalog claim
  should be LOW confidence. No direct ncu metric available. OPEN.

- **Per-warp address pathology magnitude.** 07_atomics §6 says
  "5-12× SLOWER than per-CTA" for per-warp pattern. Range is wide (5-12×) and
  V10 does not characterize this configuration. OPEN — needs dedicated sweep.

- **Stride sweep range** in 07_atomics §7 (stride=4..256 B) reports 7.2 →
  111 Gatomic/s (15× range). 07_atomics §8 stride=4 reports **504 Gops/s**
  (UNROLL=16), and TRUE_REFERENCE §5 reports **449** (peak). Three
  different "stride=4 peak" numbers (7.2 / 449 / 504 / 1005) coexist
  because of varying UNROLL and L2-residency. Catalog should ALWAYS pair
  stride with (UNROLL, WS, L2-resident?). Currently NOT enforced in §7.

- **Memory note "ATOMS pure latency 107 → 45 cy"** — provenance unknown.
  Should either be re-measured or removed from memory.

- **Memory note "SMEM atomic 4.2 Tops/s"** — provenance unknown; V10
  measured 2.27 T. Either factor-of-~2 difference indicates a config
  diff (e.g., different op type, FP vs INT, full-occupancy vs half) or
  the memory note is stale.

- **VERSION A vs VERSION B (REVERIFY)**: same SASS, 1.6× throughput delta
  purely from stride / L2 reuse. Catalog quotes **single peak numbers** for
  combine=32, but the actual achievable rate depends heavily on access
  pattern. Catalog should publish BOTH endpoints (L2-resident and DRAM-bound).

- **`28× ratio` units mistake (memory feedback)**: not directly observed in
  these atomic files, but ATOMIC_LADDER_RIGOROUS demonstrates exactly the
  failure mode that produced it: combine=32 inflates Gops/s 24× (49.7 → 1230)
  while DRAM bytes/s stays flat at ~5.5 TB/s. Future quotes MUST pair
  Gatomic/s with DRAM B/s and combine factor.
