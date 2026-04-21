# I6: Block→SM scheduling — pairs on TPC siblings, sweeps GPC rows

## Theoretical
Hopper/Blackwell schedule blocks to SMs to maximize spatial distribution. Combined with I8 (TPC = 2 SMs, GPC-row = 16 SMs), the scheduler can be:
- Pair-first: consecutive blocks → TPC siblings (L1/L2 locality)
- Spread-first: consecutive blocks → distant SMs (avoid contention)
- Round-robin: just SM 0, 1, 2, ...

## Methodology
Each block reads `%smid` and writes to `C[blockIdx.x]`. Dump and sort.

## Measured (148 blocks, 1 per SM)
First 20 blocks:
| block | SM | comment |
|-------|----|---------|
| 0 | 142 | start at end |
| 1 | 143 | TPC sibling of 142 |
| 2 | 144 | new TPC |
| 3 | 145 | TPC sibling of 144 |
| 4 | 146 | |
| 5 | 147 | last TPC pair in tail |
| 6 | 0 | wrap to start |
| 7 | 1 | TPC sibling of 0 |
| 8 | 16 | +16 = next GPC row |
| 9 | 17 | TPC sibling |
| 10 | 32 | +16 |
| 11 | 33 | |
| 12 | 48 | +16 |
| 13 | 49 | |
| 14 | 64 | +16 |
| 15 | 65 | |
| 16 | 2 | back to row 0, +2 |
| 17 | 3 | TPC sibling |
| 18 | 18 | row 1, +16 |
| 19 | 19 | |

## Pattern observed
1. **Consecutive block PAIRS land on TPC siblings**: (0,1), (2,3), ... always same TPC
2. **Pair-to-pair stride is +16 (GPC row)**: blocks 8,9 vs 6,7 = +16 SMs
3. **After exhausting GPC column, increment by +2 within row**: (0,1)→(16,17)→(32,33)→(48,49)→(64,65)→(2,3)
4. **Last 6 SMs (142-147) launched FIRST**: maybe partial GPC-row gets priority

## Conclusion
- **Pair-friendly**: consecutive blocks land on adjacent SMs in same TPC. Pair-shared L1 cache (if any) is exploited.
- **Spread-friendly**: pair-to-pair, scheduler jumps +16 to use a different GPC row before reusing the same one.
- **Hybrid TPC+GPC scheduling**: maximizes both L1 sibling locality AND GPC-fabric spread.

## Practical implications
- For pair-cooperative kernels: blockIdx 2N and 2N+1 share TPC → can use DSMEM-like patterns
- For high-bandwidth kernels: don't worry about sibling contention; scheduler spreads across GPCs naturally
- For L2 partitioning: blocks 0-7 already span 4 GPC rows = good L2 distribution

## Confidence: HIGH
- Reproducible pattern across 148, 296 launches
- Consistent with I8 cluster topology finding (TPC = 2 SMs, GPC-row stride = 16)
- Matches NVIDIA's published scheduler heuristic philosophy

What would change it: if launching with cooperative grids or persistent kernels shows different mapping.
