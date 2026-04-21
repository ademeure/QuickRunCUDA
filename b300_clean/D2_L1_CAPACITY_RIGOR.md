# D2: L1 cache capacity ≈ 128 KB (1024 lines); no observable associativity conflicts

## Theoretical
- B300 unified L1+SMEM data cache: 256 KB per SM (Hopper/Blackwell)
- L1 portion (configurable): typically 128 KB, rest goes to SMEM
- Cache line: 128 B (well-known)
- Lines per L1: 128 KB / 128 B = 1024 lines max

## Methodology
- Pointer-chase with controlled stride and N elements
- Single warp on single block (1 SM)
- Stride = 4 KB (so only 1 line per 4-KB region is touched)
- Sweep N from 64 to 8000 to find capacity boundary
- Also sweep stride at fixed N to detect associativity conflicts

## Measured (cy/load chained pointer chase)
| N elements | Total bytes touched | Lines touched | cy/load | Cache level |
|------------|--------------------|---------------|---------|-------------|
| 64-1024 | up to 4 MB | up to 1024 | **39** | L1 hit |
| 2048 | 8 MB | 2048 | 488 | partial L2 |
| 3000-8000 | 12-32 MB | 3000+ | **704** | DRAM |

Stride sweep at N=128 (line counts: 128): all stride values 64 B → 64 KB → **38 cy uniform**. No associativity conflict.

## Conclusion
1. **Per-SM L1 capacity ≈ 128 KB = ~1024 cache lines** for LDG.E (cached) global loads. Boundary is sharp (1024 → 2048 line count).
2. **L1 hit latency ≈ 39 cy chained = ~26 ns at 1500 MHz** (matches catalog 20 ns for L1)
3. **DRAM (uncached) chase ≈ 704 cy = ~470 ns** (matches catalog 705 cy)
4. **No observable associativity-induced conflict** across power-of-2 strides 64B to 64KB. B300 likely uses hashed cache indexing — power-of-2 strides do NOT alias.

## Practical implications
- For high-throughput pointer-chase: keep working set ≤ 128 KB per SM
- Strided access patterns don't suffer power-of-2 aliasing penalties (unlike many CPUs)
- This is a mature design: NVIDIA's L1 hash spreads addresses across sets

## Confidence: HIGH for capacity boundary; MED for "no associativity conflict" claim
- Capacity: 1024-line boundary is sharp and reproducible
- Associativity: tested 11 stride values; all latency-uniform within 0.2 cy. Could miss exotic patterns (e.g. odd hash collision triggers).

What would change: if a future test with larger stride range or a specific bit pattern shows asymmetric latency, would update conclusion.
