# §22j Shared memory bank conflicts — does "random = coalesced"?

**Catalog claim**: "random access = coalesced access" for smem. Counter-intuitive
because classic CUDA lore says bank conflicts serialize 32-way.

**Verdict**: TRUE for single-dword (32-bit) loads; FALSE for wider loads.

## Test setup

- B300 sm_103a, `-lgc 1800`, single warp (32 threads)
- 1024-int smem buffer
- Access pattern: each iter, lane reads `smem[(i & 31) * 32 + idx_base]`
- `idx_base` computed per pattern (seq / stride-2 / stride-16 / random / broadcast)
- Measure cy/load across 10000 unrolled iters

Tests: `tests/bench_smem_pattern.cu` (b32), `tests/bench_smem_wide.cu` (v2/v4).

## Finding 1: Single-dword (b32) loads — NO bank-conflict penalty

| Pattern                    | cy/load |
|----------------------------|--------:|
| Sequential (no conflict)   | 6.69    |
| stride-2 (2-way conflict)  | 6.88    |
| stride-4 (4-way)           | 6.88    |
| stride-8 (8-way)           | 6.88    |
| stride-16 (16-way)         | 6.88    |
| stride-32 (32-way same bank) | 6.69  |
| XORshift random            | 6.94    |
| golden-hash                | 6.88    |
| broadcast                  | 6.69    |
| permutation                | 6.88    |

**All patterns within 4% of each other.** No bank-conflict penalty is visible
for 32-bit LDS on B300. Strided-32 (all 32 lanes hit bank 0) and broadcast
(all lanes hit same address) are actually marginally FASTER, consistent with
broadcast-optimized silicon.

## Finding 2: Wide loads (v2.b32, v4.b32) — bank conflicts ARE real

| Width | Pattern      | cy/load | ratio vs seq |
|-------|--------------|--------:|-------------:|
| b32   | seq          | 6.69    | 1.00× |
| b32   | stride-2     | 6.88    | 1.03× |
| b32   | random       | 6.94    | 1.04× |
| v2.b32| seq          | 6.94    | 1.00× |
| v2.b32| stride-2     | 10.94   | **1.58×** |
| v2.b32| stride-16    | 6.88    | 0.99× |
| v2.b32| broadcast    | 6.88    | 0.99× |
| v2.b32| random       | 11.00   | 1.58× |
| v4.b32| seq          | 11.75   | 1.00× |
| v4.b32| stride-2     | 19.69   | **1.68×** |
| v4.b32| stride-16    | 11.63   | 0.99× |
| v4.b32| broadcast    | 8.44    | 0.72× (fast!) |
| v4.b32| random       | 15.75   | 1.34× |

**Wider loads show real bank-conflict penalties:**
- v2.b32 stride-2 / random = ~1.58× slower than seq
- v4.b32 stride-2 = ~1.68× slower; random = ~1.34× slower

This is because a v4.b32 load per lane spans 4 consecutive banks, so a
warp-wide v4.b32 needs to service 128 bank requests (32 lanes × 4 banks). Any
collision within the same bank serializes.

**Counter-intuitive**: broadcast on v4.b32 is FASTER than sequential (8.44 vs
11.75 cy). Likely because all 32 lanes read the same 4 banks, which the LDS
unit can broadcast as a single transaction.

## Why "random = coalesced" for b32

Hypothesis: B300 LDS for 32-bit loads uses a **hash-accelerated bank
arbitration** where simultaneous requests to the same bank are combined via
broadcast or merged via internal queues. The 32-request arbitration happens in
parallel and all lanes are served in the same cycle regardless of bank pattern.

This is a Hopper+ capability (LDSM multi-bank broadcast primitive) that seems
to extend to regular LDS on Blackwell. The 4-bank-per-lane widening of v2/v4
exceeds the arbitration capacity, so conflicts manifest.

## Catalog implication

The catalog claim "random = coalesced" should be scoped to **32-bit LDS only**.
For v2/v4 wide loads (common in tensor pipelines, `int4` gather), bank
conflicts ARE real and the old rules apply: sequential or permutation-safe
access wins.

## Related: LDSM (matrix load)

Not tested here, but LDSM (tensor shared load) is known to have broadcast
capability that eliminates bank conflicts for registered 8×8 tiles. Worth
verifying separately.

---

## ADDENDUM 2026-04-23 — IMPORTANT CORRECTION

The earlier "random = coalesced" claim was based on a TEST METHODOLOGY ERROR.

### What was wrong

My original tests used patterns like `idx_base = (lane * 32) & 31` which masks ALL lanes to the same index (broadcast pattern), not the bank-conflict pattern that the catalog measured (lane-varying addresses in the same bank).

### Correct test (true bank-conflict pattern)

`idx = ((lane * STRIDE + i) & (SMEM_SIZE-1))` — each lane gets a DIFFERENT address, controlled by STRIDE in u32 units. Stride=32 means lanes 0-31 hit addresses 0, 32, 64, ..., 992 — all in bank 0 (since bank = (addr/4) mod 32).

| STRIDE | cy/load | Slowdown vs stride-1 |
|--------|--------:|---------------------:|
| 1 | 7.01 | 1.0× (no conflict) |
| 2 | 7.20 | 1.03× |
| 4 | 11.13 | **1.59×** (4-way conflict) |
| 8 | 19.13 | **2.73×** (8-way) |
| 16 | 35.13 | **5.01×** (16-way) |
| 32 | 67.13 | **9.6×** (32-way) |

Catalog L1199 claims 13× slowdown at 32-way; I measure 9.6× — qualitatively confirming the catalog finding (bank conflicts are real).

### Correct claim

**Bank conflicts ARE real on B300 for lane-varying addresses in the same bank.** For 32-bit LDS:
- True 32-way conflict = ~10× slowdown
- True 16-way = ~5×
- True 8-way = ~3×
- 4-way = ~1.6×

The "broadcast is fast" finding is a separate phenomenon — when ALL 32 lanes hit the SAME address (not just the same bank), B300 has a broadcast optimization that serves all lanes at no extra cost.

### Updated catalog claim status

Catalog L1191-1200 ("Smem bank conflict cost") is **CONFIRMED with my own test** (within 30% of the catalog's 13× — methodology differs slightly).

The DENSE catalog correction "ld.shared bank-conflict only for v2/v4" was **WRONG** — I should walk that back. Bank conflicts ARE real for 32-bit LDS in proper benchmarks.

Wider loads (v2/v4) have additional pressure (need more banks per lane) but the basic 32-bit bank-conflict mechanism is also active.

### Corrected REVIEW_CHECKLIST entry

The previous "ld.shared bank-conflict-sensitive needs scoping" claim was wrong. Real status: catalog is correct; bank conflicts apply to 32-bit LDS too with the right access pattern.
