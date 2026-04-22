# 09 — Memory APIs (CORRECTED — TMA / cp.async section)

Scope of correction: this file ONLY revises the TMA / cp.async / async-copy
material relevant to memory-API throughput. Everything else in
`09_memory_apis.md` (alloc latency, pool, pinned, managed, memset, memcpy,
pointer-attr) is preserved as-is — those sections were already correct.

## NEW canonical TMA / cp.async ladder (B300 SXM6, 2032 MHz)

| Path | BW | % of HBM 7.2 TB/s | Source |
|---|---:|---:|---|
| Plain LDG coalesced | 5.82 TB/s | 81% | V8 I1 |
| Plain STG coalesced | 6.11 TB/s | 85% | V8 I1 |
| `cp.async.ca.shared.global` (LDGSTS) | 6.91 TB/s | 96% | V9_CP_ASYNC_BW.md |
| TMA single-deep read (`cp.async.bulk` 64 KB) | 6.72 TB/s | 92% | V33 |
| TMA write (`cp.async.bulk.tensor`) | 7.17 TB/s | 98% | V34 |
| TMA copy R+W pipelined | 6.21 TB/s combined | 93% of A6 mix peak | V35/V36 |
| **TMA 8-deep PIPELINED read (16 KB tile)** | **7.20 TB/s** | **98.5%** | **V46 — NEW READ SoL** |
| TMA write pipelined 8-deep | 6.34 TB/s | 87% | V47 — NO BENEFIT |
| TMA multicast aggregate (cluster=8, single-deep) | 14.91 TB/s effective | 17% raw HBM × 8-way | V32 |
| TMA multicast pipelined 2-deep | 13.96 TB/s | capped — single engine/cluster | V48 |

## Critical rules

1. **Reads need pipelining; writes are already async fire-and-forget.**
   - Single-deep TMA read tops out at 6.72 TB/s.
   - 8-deep mbarrier-pipelined TMA read hits 7.20 TB/s (98.5% HBM SoL).
   - Pipelining writes gives ZERO benefit (V47: 6.34 vs V34 7.17 TB/s).

2. **NEVER combine `prefetch.L2` with `cp.async.bulk` (TMA).**
   - V42: TMA + prefetch.L2 = **27% SLOWER** than TMA alone.
   - TMA already owns its own DMA path; explicit prefetch instructions block
     forward progress.
   - The V6 I3 finding "prefetch.L2 = 1.58× speedup" applied to OLD
     `cp.async` (LDGSTS), NOT to `cp.async.bulk` / TMA. Do not propagate.

3. **Multicast can NOT be pipelined.**
   - Single multicast engine per cluster.
   - Ceiling = 14.9 TB/s effective at cluster=8 (V32 = V48).
   - Adding pipeline depth (V48 2-deep) hurts slightly.

4. **`cp.async.ca` (LDGSTS) is the best non-TMA read path** at 6.91 TB/s
   (96%) — better than plain LDG (81%) because async loads bypass register
   pressure and L1 bank conflicts.

## Practical recipe (NEW HBM read SoL)

```cuda
// V46 pattern: 8-deep TMA pipeline, 16 KB tiles, mbarrier per stage
// Achieves 7.20 TB/s = 98.5% HBM peak on B300
__shared__ alignas(128) uint8_t tile[8][16384];
__shared__ uint64_t mbar[8];
// for each stage: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
// then mbarrier::test_wait.parity in round-robin → SoL.
```

## RETRACTIONS

| Old claim | Origin | Correction |
|---|---|---|
| "TMA bulk store ≈ 7.5 TB/s estimated" | V9_CP_ASYNC_BW comparison row marked `(est)` | MEASURED 7.17 TB/s (V34) — drop the estimate row |
| "TMA single-deep is HBM SoL at 6.72 TB/s" | V33 | SUPERSEDED by V46 8-deep = 7.20 TB/s |
| "prefetch.L2 helps cp.async (1.58×)" — IF applied to bulk/TMA | V6 I3 (originally about LDGSTS) | WRONG for cp.async.bulk; V42 measured −27%. Restrict V6 I3 claim to plain `cp.async` only |
| "TMA multicast scales with pipeline depth" (implicit) | V32 didn't pipeline | V48 explicitly tested — capped at single engine |
| "cudaMemset uses a special HBM fast-path / DMA" | catalog 8689-8700, EXTENDED_FINDINGS L36 | (already retired in original 09) — kept here for cross-ref |

## UNRESOLVED

- TMA pipeline depth knee: V46 used 8-deep at 16 KB. Would 4-deep × 32 KB or
  16-deep × 8 KB hit higher? Not tested.
- TMA read with `cp.async.bulk.tensor.tile` (2D/3D) vs the 1D variant used in
  V33/V46 — no SoL comparison.
- TMA multicast pipelined depth >2 — V48 only checked 2-deep.
- TMA + cluster=4 vs cluster=8 — V32/V48 only ran cluster=8.
- TMA `wait_group(N)` vs `wait_all` for cp.async (non-bulk) — flagged as
  deferred in V9_CP_ASYNC_BW.md, never resolved.
- TMA on B300 with `OOB` predicate — no test.
- `cp.async.cg` (L2-bypass / cache-global) variant for cp.async — V9 only
  tested `.ca`.

## Confidence

| Claim | Confidence | Verification |
|---|---|---|
| V46 7.20 TB/s 8-deep TMA read = HBM SoL | HIGH | ncu DRAM bytes 620/605 = 102%, 10-rule rigor |
| V34 TMA write 7.17 TB/s | HIGH | V32-V40 doc, ncu-verified |
| V32 multicast 14.9 TB/s effective | HIGH | 18 clusters × 64 KB × 512 iters |
| V48 multicast capped 13.96 TB/s | HIGH | ncu-verified |
| V42 TMA+prefetch.L2 = −27% | HIGH | direct A/B test |
| V9 cp.async.ca 6.91 TB/s | HIGH | ncu DRAM bytes match |
| V47 TMA write pipelining no benefit | HIGH | direct vs V34 |
