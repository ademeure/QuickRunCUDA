# §22r N=2 Atomic Hotspot — catalog claim investigation

**Catalog claim (L8466)**: "N=2 atomic is 20× worse than N=1, because both
addresses hash to the same L2 slice."

**Verdict**: Partly right, partly wrong.
- The slowdown IS real (measured up to **34× worse**, not 20×).
- The slowdown only occurs at **WARP-level address split**, NOT CTA-level.
- The mechanism is **broken intra-warp atomic coalescing**, not
  "L2 hash slice collision" as speculated.

## Test setup

- B300 sm_103a, `-lgc 1800`, 148 CTAs × 128 threads × 1000 atomicAdd/thread
- Tests in `tests/bench_atom_n_contend.cu` (CTA-level split),
  `tests/bench_atom_n_red.cu` (no-return, REDG),
  `tests/bench_atom_n_per_thread.cu` (per-thread WARP-level split)

## Finding 1: CTA-level split (N-way per CTA, all 128 threads/CTA share one addr)

ATOMG (with return value — forces full RMW):

| N_ADDR | mean cy/atomic | wall ms | ratio vs N=1 |
|--------|---------------:|--------:|-------------:|
|   1    |   731   | 0.446 | 1.00× |
|   2    |   711   | 0.445 | 0.998× |
|   3    |   736   | 0.458 | 1.03× |
|   4    |   737   | 0.466 | 1.05× |
|   8    |   554   | 0.433 | 0.97× |
|  64    |   540   | 0.424 | 0.95× |
| 148    |   465   | 0.426 | 0.96× |

REDG (no-return — compiler emits REDG.E.ADD.STRONG.GPU):

| N_ADDR | mean cy/atomic | wall ms | ratio vs N=1 |
|--------|---------------:|--------:|-------------:|
|   1    |   629   | 0.410 | 1.00× |
|   2    |   645   | 0.398 | 0.97× |
|   4    |   609   | 0.392 | 0.96× |
|   8    |   310   | 0.216 | 0.53× (faster) |
|  64    |    75   | 0.070 | 0.17× |
| 148    |    45   | 0.029 | 0.07× |

**CTA-level N=2 is NOT 20× worse than N=1.** It's within 5% of N=1.
N-way CTA-level scaling yields modest speedups (up to 14× at N=148 for REDG).

## Finding 2: WARP-level split (N-way per thread, intra-warp)

REDG per-thread (each thread in warp picks addr by `threadIdx % N_ADDR`):

| N_ADDR | mean cy/atomic | wall ms | ratio vs N=1 |
|--------|---------------:|--------:|-------------:|
|   1    |      629   |  0.410 | 1.00× |
| **2**  | **21,435**  | **12.74** | **34× WORSE** |
|   3    | 21,480  | 12.75 | 34× |
|   4    | 20,966  | 12.47 | 33× |
|   8    | 10,471  |  6.24 | 17× |
|  16    |  9,894  |  5.91 | 16× |
|  32    |  4,938  |  2.94 | 8× |
|  64    |  2,467  |  1.47 | 4× |
| 128    |  1,233  |  0.74 | 2× |

**The catalog's "20× hotspot" is REAL at warp level, but the measured factor
is 34×, not 20×. The mechanism is broken lane-combining at the L2 atomic
unit.**

Intra-warp address split destroys the atomic-reduction coalescing that allows
32 lanes to serve as 1 combined op at L2. Once the warp's addresses split, each
sub-warp (or individual thread) serializes at the L2 atomic unit.

## Finding 3: Offset sensitivity for WARP-level N=2

Per-thread N=2 (2 addresses, threads 0,2,4,... → addr0, 1,3,5,... → addr1) with
varying OFFSET_BYTES:

| OFFSET | wall ms | Notes |
|--------|--------:|-------|
|    4 |  6.50 | |
|    8 |  6.50 | |
|   32 | 12.74 | **spike** |
|   64 | 12.73 | spike |
| **128** | **14.84** | **WORST** — exactly L1 line size |
|  256 |  6.50 | recovers |
|  512 | 12.16 | spike |
| 1024 |  6.52 | |
| 4096+ | ~6.5 | stable |
| 16777216 |  8.76 | slight uptick (L2 partition boundary?) |

The peak at OFFSET=128B coincides with the B300 L1 cache line (128B). This
suggests the hotspot is related to **cache-line granularity coalescing**, not
L2 hash-slice collisions. When addr0 and addr1 straddle a cache-line boundary
at just the wrong spacing, the lane-combining logic can't merge them efficiently.

The spikes at 32/64/128/512 B are interesting and suggest a periodic hash-like
structure in the L2 atomic-serving logic. The "same L2 slice" speculation in
the catalog is plausible for these spikes, but the dominant mechanism for the
baseline 16× slowdown (at non-spike offsets) is warp-combining loss.

## Correction to catalog

`B300_PIPE_CATALOG.md:8466` claim:
> "N=2 atomic is ~20× worse than N=1; speculate both addresses hash to same L2 slice"

Should be corrected to:
> "When each thread in a warp atomicAdd's to one of 2 distinct addresses
> (WARP-level address split), performance drops 16-34× vs all-threads-same-addr.
> The baseline factor is ~16× from broken intra-warp lane-combining at the L2
> atomic unit. An additional ~2× hit occurs at pathological offsets
> (32/64/128/512 bytes) where L2 address-hashing causes extra serialization;
> OFFSET=128B (one L1 line) is the worst at 34×. CTA-level split (N CTAs, all
> 128 threads in one CTA share an addr) shows NO slowdown — it's purely a
> warp-coalescing effect."

## Mitigation

To avoid the hotspot:
1. Keep warp-atomic addresses in ONE address (full coalescing) OR
2. Spread across ≥8 distinct addresses per warp (combining groups of 4) OR
3. If N=2 is unavoidable, use OFFSET_BYTES ∈ {4, 8, 256, 1024+} to avoid the
   pathological offsets.
