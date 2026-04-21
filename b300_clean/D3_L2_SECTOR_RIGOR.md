# D3: L2 sector size = 32 bytes; sub-sector writes trigger RMW with up to 7× DRAM read amplification

## Theoretical
Modern NVIDIA GPUs (Volta+) use 128-byte L2 cache lines composed of 4× 32-byte sectors. When a thread writes only part of a sector, the L2 controller must fetch the rest of the sector from DRAM, modify it, and write it back. This is "read-modify-write" (RMW) amplification.

Predictions:
- 32B aligned writes (full sector): 0× read amp
- 4B writes (1/8 of sector): up to 8× read amp
- 16B writes (1/2 of sector): up to 2× read amp
- 128B aligned writes (full line, 4 sectors): 0× read amp

## Methodology rigor
- 296 blocks × 128 threads × 1000 iterations
- 256 MB buffer with wrap-around (forces DRAM traffic)
- ncu metrics: `dram__bytes_read.sum`, `dram__bytes_write.sum`
- 6 modes spanning {4B, 16B, 32B, 128B} writes × {32B, 128B} strides

## Measured
| MODE | WSIZE | STRIDE | Intended Wr | DRAM Read | DRAM Write | Read Amp | Write Amp |
|------|-------|--------|-------------|-----------|------------|----------|-----------|
| 0 | 4 | 32 | 151 MB | 1053 MB | 1129 MB | **7.0×** | 7.5× |
| 1 | 4 | 128 | 151 MB | 784 MB | 1029 MB | 5.2× | 6.8× |
| 2 | 16 | 32 | 606 MB | 1049 MB | 1130 MB | 1.7× | 1.9× |
| 3 | 32 | 32 | 1212 MB | 0.6 MB | 1151 MB | **0.0×** | 0.95× |
| 4 | 32 | 128 | 1212 MB | 11 MB | 1033 MB | 0.009× | 0.85× |
| 5 | 128 | 128 | 4849 MB | 60 MB | 4161 MB | 0.012× | 0.86× |

## Conclusion
1. **L2 sector size on B300 = 32 bytes.** Writes < 32B trigger RMW; writes ≥ 32B (aligned) do not.
2. **Sub-sector write penalty:** 4B writes (1/8 sector) cause **7× DRAM read amplification** + ~7.5× write amplification.
3. **Cache line size = 128 bytes** (4 sectors), but RMW fires per-sector, NOT per-line. Writing all 4 sectors of a line is unnecessary as long as each is fully written.
4. **Sweet spot: write 32B-aligned chunks** — uint4 writes (16B) still incur 1.7× amp.

## Practical implications
- For scatter writes: pad to 32B-aligned full sectors (e.g. write `uint4` + `uint4` = 32B together).
- A naive scattered-FP32 write workload hits **<15% effective DRAM BW** vs full-sector writes.
- For 4B atomic-style writes on dense indices, consider buffering 8 values in SMEM and flushing as 32B sector.
- Compaction kernels should accumulate in SMEM until 32B-aligned, then commit.

## Confidence: HIGH
Verified by:
- Exact 7-8× ratio matches 32B/4B = 8× theoretical sector amplification
- Read amp = 0 for 32B-aligned (clean confirmation)
- Cross-mode consistency: MODE 1 less amplified than 0 (sparser stride hits fewer DRAM transactions per byte)

What would change it: if a future test with large block-of-line writes shows different per-line vs per-sector behavior.
