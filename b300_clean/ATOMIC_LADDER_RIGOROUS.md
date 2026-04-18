# B300 sm_103a HBM-resident atomic FULL UNIT BREAKDOWN
# All measurements: WS=1024 MB (>>L2 60 MB), 1184 blocks × 256 thr = 303104 threads,
#                   OCC=8 (2048 thr/SM = full B300 occupancy), 100 inner iters per kernel call

Each test does N "thread-atomic-instructions" — one atomic call per thread per inner iter.

KEY UNITS (always all 5):
  T = thread-atomics/sec (1 source-level atomicX() call per thread per inner iter)
  W = warp-atomics/sec  = T / 32 (1 SASS instruction per warp issues 1 atomic)
  L = L2 packets/sec    = T / combine_ratio (after warp-level merging at L2 atomic unit)
  P = payload bytes/sec = T × atomic_type_width_bytes (the actual data being added/swapped)
  D = DRAM bytes/sec    = ncu dram__bytes.sum.per_second (measured)

CASE 1: int32 atomicAdd, NO COMBINING (each thread → distinct cache line, stride=128B)
  T = 49.7e9     thread-atomics/sec  (49.7 G atomicAdd calls/sec, summed across all threads)
  W = 1.55e9     warp-atomics/sec
  L = 49.7e9     L2 packets/sec      (combine_ratio=1, so T==L)
  P = 199 GB/s   payload bytes       (49.7e9 × 4B per int32)
  D = 5.52 TB/s  DRAM bytes/sec      (ncu measured)
  
  Per atomic call breakdown:
    payload value = 4 bytes (int32)
    DRAM traffic  = 5.52e12 / 49.7e9 = 111 B/op  (≈ 1 cache line read + ε for write merge)
    L2 ratio      = 1 packet per atomic (no combining)

CASE 2: uint64 atomicAdd, NO COMBINING (stride=128B per thread)
  T = 49.8e9     thread-atomics/sec
  W = 1.56e9     warp-atomics/sec
  L = 49.8e9     L2 packets/sec
  P = 398 GB/s   payload (49.8e9 × 8B per uint64)
  D = 5.52 TB/s  DRAM (ncu)
  
  Per atomic call:
    payload value = 8 bytes
    DRAM traffic  = 111 B/op (same as int32 — DRAM cost is per-cache-line, not per-data-byte)
    P/D ratio     = 7.2% (vs int32 3.6% — 2× because payload is wider for same DRAM cost)

CASE 3: b128 atom.global.exch, NO COMBINING (stride=128B per thread)
  T = 42.2e9     thread-atomics/sec
  W = 1.32e9     warp-atomics/sec
  L = 42.2e9     L2 packets/sec
  P = 676 GB/s   payload (42.2e9 × 16B per b128)
  D = 4.64 TB/s  DRAM (ncu)
  
  Per atomic call:
    payload value = 16 bytes
    DRAM traffic  = 110 B/op (same per-cache-line cost; b128 is slightly slower so DRAM lower)
    P/D ratio     = 14.6% (4× int32 — biggest payload per DRAM byte tested)

CASE 4: int32 atomicAdd, COMBINE=32 (full warp → 1 cache line, lane=offset)
  T = 769e9      thread-atomics/sec
  W = 24.0e9     warp-atomics/sec
  L = 24.0e9     L2 packets/sec      (32 thread-atomics merged → 1 L2 packet per warp)
  P = 3.08 TB/s  payload (769e9 × 4B)
  D = 4.03 TB/s  DRAM
  
  Per warp-atomic-instruction:
    32 thread atomics absorbed by L2 popc-merge → 1 L2 atomic op per warp
    1 L2 op → 1 cache line read + 1 line write to DRAM (eventually)
    DRAM per warp = 4.03e12 / 24.0e9 = 168 B/warp ≈ 1.3× cache line (some L2 reuse)
    P/D ratio = 76.4% (combining lets L2 absorb most traffic; DRAM still hit 4 TB/s)

CASE 5: b128 atom.exch, COMBINE=8 (8 lanes per cache line; 4 unique lines per warp)
  T = 174.3e9    thread-atomics/sec
  W = 5.45e9     warp-atomics/sec    (each warp issues 8 b128 atomics?? actually 32 lanes / 4 lines = 8 lanes/line, all lanes still issue)
  L = 174.3 / 8 = 21.8e9 L2 packets/sec
  P = 2.79 TB/s  payload (174.3e9 × 16B)
  D = 5.51 TB/s  DRAM
  
  Per L2 packet: 256 B DRAM (RMW per cache line)
  Per warp-atomic (32 thread b128s, 4 cache lines): DRAM = 4 × 256 = 1024 B / warp
  Predicted DRAM = W × 1024 B = 5.45e9 × 1024 = 5.58 TB/s ≈ 5.51 measured ✓

UNIVERSAL ATOMIC DRAM CEILING: ~5.5 TB/s = 75% of HBM raw peak (7.31), 82% of mixed-RW peak (6.68).
Atomic forces line-RMW; HBM controller can't push past 5.5 TB/s sustained.

WIDER ATOMIC TYPES → MORE PAYLOAD PER DRAM BYTE:
  int32 c=1:  3.6% payload/DRAM
  uint64 c=1: 7.2% (2x: same per-line cost, 2x payload bytes)
  b128 c=1:   14.6% (4x: same per-line cost, 4x payload bytes)

