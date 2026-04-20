# SHMEM bank conflict / broadcast behavior — V4 / D5

**Date: 2026-04-20.** Test `bench_smem_bank_broadcast.cu`. 1500 MHz,
single warp single block, N_DEPS=8 chain (LDS result → next LDS addr
to defeat hoisting). Times measured by clock64 inside kernel.

## Results: cy/LDS as function of access pattern

| Mode | Pattern (lane k) | cy/ld | × broadcast |
|------|------------------|-------|-------------|
| 0 | All lanes read same `addr[const]` (perfect broadcast) | **13.03** | 1.0× |
| 7 | `addr[lane*33]` (bank-skewed, no conflict) | 14.64 | 1.12× |
| 1 | `addr[lane]` (distinct banks, no conflict) | 14.64 | 1.12× |
| 2 | `addr[lane & 15]` (2-way broadcast, 16 banks) | 14.64 | 1.12× |
| 3 | `addr[lane & 7]` (4-way broadcast, 8 banks) | 14.64 | 1.12× |
| 4 | `addr[lane*2]` (stride-2, no conflict) | 14.89 | 1.14× |
| 6 | `addr[lane*4]` (stride-4 → 4-way conflict) | 18.77 | 1.44× |
| 5 | `addr[lane*32]` (stride-32 → 32-way conflict) | **74.77** | **5.74×** |

## Key findings

1. **Perfect broadcast (1 address) is FASTER than distinct banks**
   (13.03 vs 14.64 cy = 12% saving). Likely a single read + 32-way
   broadcast bus saves 1-2 cy of bank arbitration.

2. **Partial broadcast (2-way, 4-way) costs nothing extra**.
   The bank arbiter recognizes any number of lanes targeting the
   same bank and broadcasts the value once.

3. **Stride-skew (stride-33) defeats conflicts cleanly** — cost is
   identical to distinct (14.64 cy). Confirmed B300 still uses the
   classic 32-bank rotating scheme; relatively-prime strides don't conflict.

4. **Worst-case 32-way conflict costs ~5.7× broadcast** (74 vs 13 cy),
   NOT the naive "32×" expectation. SMEM broadcast bus helps even when
   serializing — the controller may identify groups within the conflict.

5. **Stride-4 (4-way conflict) costs 1.44×** — much less than naive 4×.
   For 4-way: 4 cy serial expected over 1 cy peak ≈ 4×. Observed 1.44×
   suggests partial overlap with chain-latency hiding.

## Practical recipe

For SHMEM-heavy kernels:
- **Default to skewed strides (e.g., 33 instead of 32)** to avoid
  conflicts on transposes and reduction stages.
- **Broadcast loads (e.g., shared scratch) are first-class fast**;
  no need to scatter to distinct banks.
- **N-way broadcast (warps reading same K-element of B in GEMM tiles)
  has zero penalty** vs distinct-bank reads.
- **Worst case 32-way conflict only 5.7×** — annoying but not 32×
  catastrophic; in some kernels acceptable.

## Caveats / scope

- Single-warp single-block test — does NOT measure throughput under
  many warps issuing concurrently. SHMEM aggregate BW (per
  `04_fp32_peak.md` companion data) is 38.5 TB/s.
- N_DEPS=8 chain provides ILP; without chain dep, throughput would
  be different.
- Cycle counts include per-LDS issue overhead (~13-14 cy) plus
  conflict serialization.

## Confidence

- **HIGH** for relative ordering (broadcast < distinct < N-way conflict)
- **HIGH** for skew (stride-33) avoiding conflict
- **MED** for absolute cy/ld magnitudes (single-warp test is
  pessimistic; full-occupancy throughput would saturate at LDS
  pipe peak which is 1-2 cy/inst per SMSP)
- **MED** for "5.7× not 32×" — single warp may show different
  serialization behavior than 32 warps fighting for a single bank

## Files

- `tests/bench_smem_bank_broadcast.cu` — modes 0-7, t=32, b=1
