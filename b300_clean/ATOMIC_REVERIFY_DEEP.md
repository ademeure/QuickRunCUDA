# B300 atomic — RIGOROUS REVERIFY with comprehensive ncu

## Updated facts
  L2 cache size: **126 MiB** (NOT 60 MB as I'd been assuming)
  Memory bus: 7680 bits = 7.5 × 1024-bit HBM3E stacks (unusual)
  L2 atomic units: catalog says "~32" but INFERRED, not directly measured
  Clock domains: SM 2032/1920, Video/L2 1860 MHz (const), HBM 3996 MHz (const)

## Two "combine=32 atomicAdd" kernels gave wildly different results

VERSION A (small stride, lots of L2 reuse):
  Loop body: addr = (warp_base + i*8192 + lane) & N-1
  Within-warp: 32 lanes = 1 cache line ✓ combine=32
  Iter stride: 32 KB (small)
  WS=32MB: 1230 Gops/s, 4.93 TB/s payload, 4.78 TB/s L1tex, 7.07 TB/s L2tex, 80 GB/s DRAM
  WS=1024MB: 1212 Gops/s, 4.85 TB/s payload, 4.93 TB/s L1tex, 7.46 TB/s L2tex, 82 GB/s DRAM (!!!)

VERSION B (large stride, ~no L2 reuse):
  Loop body: addr = (warp_id + i*N_warps) * 32 + lane
  Iter stride: ~1.21 MB (large)
  WS=32MB: 767 Gops/s, 3.07 TB/s payload, 3.10 TB/s L1tex, 4.74 TB/s L2tex, 858 GB/s DRAM
  WS=1024MB: 768 Gops/s, 3.07 TB/s payload, 2.63 TB/s L1tex, 3.98 TB/s L2tex, 4.03 TB/s DRAM

## Key revelation
VERSION A's "1230 Gops/s payload 4.93 TB/s" sounds amazing — but DRAM is only
80 GB/s! Almost ALL atomic ops are absorbed by L2 cache (warm reuse from small
stride). So this isn't really 4.93 TB/s of work hitting "memory" — it's
4.93 TB/s of work hitting an L2-cached cache line over and over.

VERSION B has the same SASS but 1.6x SLOWER because:
  (a) Per-iter overhead higher (long math vs int math): 624 vs 144 SASS inst
  (b) More importantly: large stride evicts lines → DRAM round-trips needed
      at WS=1024MB, B uses 4 TB/s of DRAM (atomic-RMW limit hit)

## L2 atomic throughput interpretation

VERSION A WS=1024MB:
  Gops/s: 1212
  L2 packets/s (combine=32): 1212/32 = 37.9 G L2 packets/s
  At video clock 1.86 GHz: 37.9/1.86 = 20.4 L2 packets/cy in parallel
  l2tex bytes/cy: 7.46e12/1.86e9 = 4011 B/cy
  Per L2 packet: 4011/20.4 = 197 B (= ~6 sectors of 32B)
  DRAM: 82 GB/s = 44 B/cy (almost zero — warm L2 reuse dominates)

VERSION B WS=1024MB:
  Gops/s: 768
  L2 packets/s: 768/32 = 24 G L2 packets/s
  At video 1.86 GHz: 24/1.86 = 12.9 L2 packets/cy
  l2tex bytes/cy: 3.98e12/1.86e9 = 2140 B/cy
  Per L2 packet: 2140/12.9 = 166 B (= 5 sectors)
  DRAM: 4.03 TB/s = 2167 B/cy = ~17 cache lines per video cycle

## "32 L2 atomic units" claim — likely wrong on B300

VERSION A measures 20-21 L2 packets/cy in parallel (combined; SM-issue-bound)
VERSION B measures 13 L2 packets/cy (combined, larger stride)
Uncombined int32 (earlier): 27 L2 packets/cy

If "32 L2 atomic units" were the architectural ceiling, none of these would
exceed 32. They don't, but VERSION A getting 20.4 with very high L2-residency
suggests the ceiling could be MUCH HIGHER than 32 — we're just bounded by
other factors (SM-issue rate, L2 sub-pipeline, etc.).

We cannot reliably claim a specific L2 atomic unit count from these tests.
The catalog "~32 inferred from plateau" should be marked LOW confidence.

What we CAN say:
  - Combined-warp atomic throughput at SM 2032 MHz ≤ 24-38 G packets/sec
    (depends on access pattern's L2 hit rate)
  - Uncombined atomic at HBM-resident saturates at ~50 Gops/s
  - L2 atomic-unit count is OPEN — 32 is unverified

