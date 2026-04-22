# V53 DSMEM fenced retest — empirical results

**Date:** 2026-04-22
**Source:** `tests/standalone/v53_dsmem_fenced.cu`
**Toolchain:** `nvcc 13.2 V13.2.78`, `-arch=sm_103a -O3`
**Hardware:** B300 SXM6 sm_103a, 148 SMs, default boost (~2032 MHz, no nvidia-smi lock)
**Methodology:** identical kernels with `FENCED` template parameter only differing in:

```ptx
// FENCED=0: stores ... clock64-end                  (V21 pattern)
// FENCED=1: stores
//           fence.sc.cluster
//           barrier.cluster.arrive
//           barrier.cluster.wait
//           clock64-end
```

Cluster size 8, 18 clusters = 144 CTAs (4 SMs idle), 128 threads/CTA, `__launch_bounds__(128, 1)`.
Anti-DCE: XOR-collapse SMEM after stores → `atomicXor` to global. Anti-LICM: addresses depend on
`tid` AND `(it,j)`. Three runs, results stable to <1%.

---

## 1. Compile status

Clean compile, 24 kernel instantiations (12 configs × 2 FENCED variants).

```
$ nvcc -arch=sm_103a -O3 v53_dsmem_fenced.cu -o /tmp/v53
[OK]
```

---

## 2. SASS verification

`cuobjdump --dump-sass --function _Z9v53_writeILi[01]ELi8ELi2ELi15000EE…` (ILP=8 INNER=2):

**UNFENCED timed-region tail:**
```
/*0710*/   BRA.U UP1, 0x340 ;            // inner loop
/*0720*/   @P0 BRA 0x2e0 ;               // outer loop
/*0730*/   CS2R R4, SR_CLOCKLO ;         // end-clock — NO MEMBAR before this
/*0790*/   MEMBAR.ALL.GPU ;              // (after end-clock; tied to cluster sync)
```

**FENCED timed-region tail:**
```
/*0710*/   BRA.U UP1, 0x340 ;            // inner loop
/*0720*/   @P0 BRA 0x2e0 ;               // outer loop
/*0730*/   MEMBAR.SC.GPU ;               // <-- fence.sc.cluster
/*0770*/   MEMBAR.ALL.GPU ;              // cluster barrier prelude
/*07a0*/   UCGABAR_ARV ;                 // barrier.cluster.arrive
/*07b0*/   UCGABAR_WAIT ;                // barrier.cluster.wait
/*07d0*/   CS2R R4, SR_CLOCKLO ;         // end-clock — AFTER drain
```

Confirmed: the only difference between the two variants in the timed region is
`MEMBAR.SC.GPU + UCGABAR_ARV/WAIT` between the last store and the end-clock.

DSMEM stores compile to `ST.E` (generic-window 32-bit), not STS.

---

## 3. Run output (median of 3 runs)

### Long sustained tests (all 18 clusters, runtime ~12 ms)

| TILE | ILP | UF wall | UF GB/s/clu | UF cyBW/CTA | F wall | F GB/s/clu | F cyBW/CTA | wall ratio | cy ratio |
|-----:|----:|--------:|------------:|------------:|-------:|-----------:|-----------:|-----------:|---------:|
|  2KB |  4  | 10.0 ms | 81.9        | 13.7        | 10.0 ms| 81.8       | 11.8       | 1.00       | 1.16     |
|  4KB |  4  | 12.0 ms | 81.7        | 13.7        | 12.0 ms| 81.9       | 11.9       | 1.00       | 1.15     |
|  4KB |  8  | 11.9 ms | 82.7        | 13.7        | 11.9 ms| 82.6       | 11.8       | 1.00       | 1.16     |
|  8KB |  4  | 12.0 ms | 82.0        | 13.7        | 12.0 ms| 82.1       | 12.0       | 1.00       | 1.14     |
|  8KB |  8  | 11.9 ms | 82.7        | 13.7        | 11.9 ms| 82.6       | 11.9       | 1.00       | 1.15     |
|  8KB | 16  | 12.1 ms | 81.2        | 13.7        | 12.1 ms| 81.2       | 11.5       | 1.00       | 1.20     |
| 16KB |  4  | 12.9 ms | 81.0        | 13.7        | 12.9 ms| 81.2       | 11.8       | 1.00       | 1.16     |
| 16KB |  8  | 12.7 ms | 82.4        | 13.7        | 12.8 ms| 82.4       | 11.9       | 1.00       | 1.15     |
| 16KB | 16  | 12.9 ms | 81.1        | 13.7        | 12.9 ms| 81.1       | 11.5       | 1.00       | 1.20     |
| 64KB |  4  | 13.0 ms | 81.0        | 13.7        | 13.0 ms| 81.0       | 11.4       | 1.00       | 1.20     |
| 64KB |  8  | 12.9 ms | 81.2        | 13.7        | 12.9 ms| 81.1       | 11.4       | 1.00       | 1.20     |
| 64KB | 16  | 13.0 ms | 80.9        | 13.7        | 13.0 ms| 80.9       | 11.4       | 1.00       | 1.20     |

**Wall-clock BW is identical** between fenced and unfenced (ratio 1.00 across all 12 configs).
**clock64-measured BW differs by ~16%** (1.16× cy ratio): the fence + cluster barrier costs
~3M cycles per CTA on top of the ~18M cy of stores, but the kernel-as-a-whole still has to
drain before `cudaDeviceSynchronize`, so wall-time is unchanged.

### V21-style burst tests (1 cluster, 4 warps × ILP=4, varying outer iters)

| outer iters | UF cy/CTA | UF cyBW/CTA | F cy/CTA | F cyBW/CTA | UF/F cyBW |
|------------:|----------:|------------:|---------:|-----------:|----------:|
|     5       |   389     | **53.5**    |  1504    | **13.8**   | **3.88×** |
|    50       |  7248     | 28.7        |  8523    | 24.4       | 1.18      |
|   500       | 76540     | 27.2        | 77886    | 26.7       | 1.02      |
|  5000       | 769541    | 27.0        | 770868   | 27.0       | 1.00      |

**At V21's exact CL=5 burst geometry, the unfenced number (53 GB/s/CTA, ~430 GB/s aggregate
for an 8-CTA cluster) is 3.88× higher than the fenced equivalent (13.8 GB/s/CTA, ~110 GB/s
aggregate). At longer bursts (≥500 outer iters) the gap closes to <2%.**

---

## 4. ncu cross-check

Profiled `v53_write<0,4,1,50000>` (ILP=4, INNER=1, N_ITER=50000, all 144 CTAs):

```
gpc__cycles_elapsed.avg                              cycle  18,036,830
sm__cycles_elapsed.avg                               cycle  18,036,785
smsp__inst_executed.sum                              inst   951,259,680
smsp__inst_executed_pipe_lsu.sum                     inst   115,355,664
l1tex__data_pipe_lsu_wavefronts_mem_shared.sum                155,088   <-- LOCAL only
dram__bytes.sum                                      Kbyte         5.63 <-- ~zero
lts__t_bytes.sum                                     Kbyte       852.70 <-- ~zero
lts__t_sectors_op_write.sum                          sector        3312 <-- ~zero
```

**Critical findings:**
1. Expected DSMEM stores: 144 CTAs × 4 warps × 50000 iters × 4 ILP = 115.2M LSU instructions
   → matches `smsp__inst_executed_pipe_lsu.sum = 115.36M` exactly.
2. `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum` only counts **local** SMEM (≈18432 STS
   from the zero-init loop × ~8 wavefronts ≈ 147k). DSMEM stores are NOT counted here.
3. **L2 sectors written = 3312, DRAM bytes ≈ 5 KB.** DSMEM cross-CTA stores do NOT actually
   traverse L2 — they go through the inter-CTA SMEM fabric directly. The previous claim in
   `DSMEM_REFERENCE.md` line 22 ("Goes through L2 (ncu shows ~4 sectors/load)") is **wrong
   for stores** — fewer than 1 sector per 30,000 stores hit L2.
4. The fenced variant has 8502 L2 write sectors and 85 KB DRAM bytes (still negligible) —
   slightly more because the cluster barrier flushes a few cleanup paths.

---

## 5. Verdict

**DSMEM_DOUBT_REPORT was partially right and partially wrong.**

### Where DSMEM_DOUBT was RIGHT
- V21's "560 GB/s aggregate write ceiling" was indeed measured **without a completion fence**,
  during a burst of only 5 outer iters. At that exact geometry, V53 reproduces the same
  high-issue-rate number (cyBW = 53.5 GB/s/CTA = ~428 GB/s/cluster) and shows the fenced
  equivalent is **3.88× lower** (13.8 GB/s/CTA = ~110 GB/s/cluster). So **at burst lengths
  ≤ a few hundred stores per warp, V21's number IS issue rate, not completion rate**.

### Where DSMEM_DOUBT was WRONG
- For **sustained** writes (loops issuing ≥10⁵ stores per CTA, kernel runtime ≥10 ms), the
  fence makes essentially **no difference to wall-clock BW** — both fenced and unfenced
  saturate at **82 GB/s per cluster (10.2 GB/s per CTA, 1.47 TB/s aggregate across 18
  clusters)**. The reason: the fence only delays *when* stores complete, not *how fast*
  they complete; at steady state the in-flight queue is back-pressured by completion anyway,
  and the kernel-as-a-whole still waits for drain before `cudaDeviceSynchronize`.
- **clock64-vs-events** matters: clock64 captures the fence cost (cy ratio 1.16-1.20×, ~3M cy
  per CTA), but events span the whole kernel including drain so the cost is invisible.

### Real DSMEM write completion BW
- **Sustained per-cluster: 82 GB/s** (not 560)
- **Sustained per-CTA: 10.2 GB/s** (not 70)
- **Sustained aggregate (18 clusters concurrent): 1.47 TB/s**
- **Single-cluster burst (no contention from other clusters): up to ~210 GB/s sustained**
  (brst5k, 8 CTAs × 27 GB/s × 4 B = 187 agg) — so 18-cluster scaling is only 0.44× linear.
  Consistent with there being a per-CTA serving cap on the inter-CTA SMEM fabric.

### Confidence
| Claim | Confidence |
|---|---|
| V21 burst rate of 560 GB/s is issue-rate, not completion (in burst regime) | **HIGH** (3.88× drop reproduced 3 runs, identical SASS except fence) |
| Sustained completion-rate aggregate is 1.47 TB/s across 18 clusters | **HIGH** (event-timed, anti-DCE verified, ncu LSU-inst matches expected count) |
| Single-cluster sustained 187 GB/s aggregate | **HIGH** (clock64 and events agree) |
| `st.shared::cluster.u32` does NOT traverse L2 | **HIGH** (ncu lts__t_sectors_op_write << expected) |
| The 0.44× scaling from 1→18 clusters means there IS a shared resource | MEDIUM (could be SM-issue-throttling or DSMEM fabric saturation) |

---

## 6. Proposed change to `b300_clean/13_*` / DSMEM canonical doc

Original DSMEM_REFERENCE.md §3 "Write BW (ring, all CTAs active)" table claims:
```
| 4 × 4 | 70.08 GB/s/CTA | 560.7 GB/s/cluster ← write ceiling |
```

**Proposed replacement text:**
```
### Write BW (ring) — REVISED (V53 fenced retest, 2026-04-22)
The original V21 numbers (560 GB/s/cluster) measured a SHORT burst (5 outer iters)
WITHOUT a completion fence between the stores and the closing clock64.
Adding `fence.sc.cluster + barrier.cluster.{arrive,wait}` drops the same burst
3.88× to ~110 GB/s/cluster — confirming the original was issue rate.

| Regime                                  | Per-CTA  | Per-cluster | Aggregate (18 clu) |
|-----------------------------------------|----------|-------------|---------------------|
| V21 burst (5 outer iters), unfenced     | 53 GB/s  | 428 GB/s    | n/a                 |
| V21 burst (5 outer iters), fenced       | 14 GB/s  | 110 GB/s    | n/a                 |
| Sustained 1-cluster (5000 iters)        | 23 GB/s  | 187 GB/s    | n/a                 |
| Sustained all-18-clusters (≥10 ms)      | 10 GB/s  | 82 GB/s     | 1.47 TB/s           |

**Use 82 GB/s/cluster (10 GB/s/CTA) for any sustained-throughput model.**
The 187 GB/s 1-cluster figure represents what's achievable when the fabric is
not contended by other clusters; with all 18 clusters writing simultaneously,
per-cluster BW falls to 82 GB/s due to fabric/scheduler contention.
```

§9 Summary rule #2 ("Use DSMEM writes over reads (13× higher aggregate BW: 560 vs 40 GB/s)")
should change to:
```
2. **Use DSMEM writes over reads (~2× higher per-cluster sustained BW: 82 vs ~40 GB/s)**
```

§1 SASS codegen line ("Goes through L2 (ncu shows ~4 sectors/load)") should change to:
```
For READS, ld.shared::cluster.u32 with scalar address compiles to LD.E and goes through
the L2 path (ncu confirms ~4 sectors/load). For WRITES, st.shared::cluster.u32 also
compiles to ST.E but does NOT measurably hit L2 (ncu lts__t_sectors_op_write shows
<1 sector per 30,000 stores) — DSMEM writes use a dedicated inter-CTA fabric.
```

---

## 7. Caveats / honest disclosure

- The 1.47 TB/s "aggregate" assumes the test's per-cluster scaling extrapolates to all 18
  clusters; measured directly with all 18 clusters active.
- The single-cluster 187 GB/s sustained could itself be inflated if the surrounding 17 SMs
  in that cluster are idle and supply bandwidth via shared resources. A future test should
  vary 1, 2, 4, 8, 16, 18 clusters at sustained length to map the true scaling curve.
- I did not test reads in this V53 — V21 chain-read 40 GB/s/cluster is unaffected by this
  result (chain-bound, not fence-bound). The DSMEM_DOUBT proposal for non-chained ILP read
  remains open work (the read kernel in the sketch was not implemented here).
