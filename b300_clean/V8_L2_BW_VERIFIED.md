# V8: L2 cache BW = 13.85 TB/s (strided, cache-global) vs L1 30.5 TB/s

## 10-rule rigor walk-through

1. **Theoretical**: L2 BW on Blackwell is not a simple formula — it's bounded by
   L2-to-SM network throughput. Prior catalog measured 13-21 TB/s steady state
   (depends on access pattern and L1/L2 partitioning).

2. **Measured** (3 test configurations):

   | Pattern | Path | L2 sectors | Time | BW delivered |
   |---------|------|------------|------|--------------|
   | Strided, default ld | L1 serves most | 878K (0.4% of L1) | 246 µs | 30.5 TB/s (L1) |
   | Strided, ld.cg (L1 bypass) | L2 direct | 193M | 447 µs | **13.85 TB/s (L2)** |
   | Coalesced, ld.cg | L2 + DRAM mix | 213M + 130 MB DRAM | 652 µs | 10.5 TB/s (L2 output) |

3. **Rule 3**: All L2 measurements < 21 TB/s catalog. Not broken.

4. **Why L1 path shows 30.5 TB/s**: when working set fits in L1 (per-SM 256 KB),
   most loads become L1 hits. The "L2 test" without `.cg` measures L1.
   True L2 BW requires cache-bypass load (`ld.global.cg`) to force L2 traffic.

5. **ncu cross-check**:
   - `lts__t_sectors_srcunit_tex_op_read.sum` gives L2 read sectors directly
   - Multiplied by 32 B = L2 output bytes
   - Divided by kernel time = L2 BW

6. **SASS** (`sass/bench_v8_l2_bw*.sass`): emits `LDG.E.128.CONSTANT` for default,
   `LDG.E.CG.128` for `.cg` variant (cache at L2 only).

7. **Three methods**:
   - ncu lts sectors (authoritative L2 traffic)
   - ncu gpu_time (authoritative timing)
   - Wall clock -T (not used here but aligns with ncu typically)

8. **Conclusive**: with buffer 64 MB (fits in 126 MB L2) and cg loads, L2
   delivers 6.19 GB over 447 µs = 13.85 TB/s. Confirms prior catalog.

9. **Surprise handled**: initial non-cg test showed 30.5 TB/s — but ncu L2
   sectors show only 0.4% of L1 requests. That was L1 BW, not L2. Corrected
   with `.cg` qualifier.

10. **Confidence: HIGH** for the 13.85 TB/s L2 figure.
    Would change if:
    - Different load variant (ld.ca, ld.cv, ld.cs) delivers different BW
    - Buffer size spans full L2 (126 MB) may give higher BW due to prefetcher
    - Atomic operations on L2 have different path (prior catalog shows atomic L2 up to 30 TB/s)

## L2 vs L1 on B300

- **L1** (per-SM, ~256 KB): ~30.5 TB/s aggregate when working set fits
- **L2** (126 MB, partitioned): 13.85 TB/s strided cg, up to ~21 TB/s optimal
- **HBM3E** (287 GB): 7.2 TB/s read peak (V8 I1)
- **PCIe Gen 6** (host): 57.8 GB/s (V8 H1)

BW ladder: L1 (30) : L2 (14-21) : HBM (7) : PCIe (0.06) ratios 4.3 : 2 : 1 : 0.008.

## Implication

For kernels that must squeeze L2 BW:
- Max out L1 first (keep per-SM working set < 256 KB)
- Use `__ldcg` for L1-bypass if L1 thrashing is an issue
- Prefer LDSM / cp.async paths for tensor feeds
- Expect realistic L2 ceiling ~14-21 TB/s (less than L1's 30)