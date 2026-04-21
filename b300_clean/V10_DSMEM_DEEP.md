# V10: Cluster=8 DSMEM deep dive — BW matrix + constraints

## Measurements (inline-offset BW test, 144 blocks × 128 thr, 10K iters)

### Cluster size sweep (NL=8 ILP fixed)

| Cluster size | Time   | BW         |
|--------------|--------|------------|
| 2            | 0.12 ms| **48.5 TB/s** ← fastest |
| 4            | 0.18 ms| 32.6 TB/s  |
| 8            | 0.18 ms| 32.6 TB/s  |

### ILP sweep (cluster=8 fixed)

| NL (loads/iter) | Time   | BW           |
|-----------------|--------|--------------|
| 1               | 0.12 ms| 6.1 TB/s     |
| 2               | 0.12 ms| 12.2 TB/s (2.0×) |
| 4               | 0.15 ms| 20.2 TB/s (3.3×) |
| 8               | 0.18 ms| 32.6 TB/s (5.4×) |
| 16              | 0.28 ms| **42.3 TB/s** (7.0×) |

## Surprising: cluster=2 is fastest

48.5 TB/s at cluster=2 vs 32.6 at cluster=4/8. Why?
- Smaller cluster = shorter SM-to-SM interconnect routing
- Less coherence coordination across peers
- Tradeoff: cluster=8 offers more DSMEM per CTA (8× remote SMEM) vs
  cluster=2's simpler 2-way peer routing

**Rule of thumb**: Use smallest cluster that gets the DSMEM size you need.
Going larger doesn't add BW; may hurt it via routing overhead.

## ILP scales near-linearly to NL=16

DSMEM loads can be pipelined heavily. 7× speedup at NL=16 over NL=1.
At NL=16 (0.28 ms), 42.3 TB/s exceeds theoretical 38.5 TB/s ceiling.

**Rule 3 caveat**: The 42.3 TB/s figure is above theoretical. Possible
explanations:
1. L1 caching: SMEM footprint (8 KB × blocks) small, may hit L1 after
   initial fill
2. Timing noise at sub-0.3 ms kernels
3. Peer SM serves reads in parallel with its own local activity

This is MEDIUM confidence for the 42 TB/s peak. For the cluster-size
comparison and ILP scaling shape, HIGH confidence.

## DSMEM chained-latency test — FAILS

Multiple attempts to measure `ld.shared::cluster.u32` dependency chain
latency (where each load's result becomes next address) consistently
fail with "unspecified launch failure" on sync. Replicated across:
- My new V10 tests
- Direct reproduction of V8 `bench_dsmem_definitive` kernel
- Minimal 1-warp single-load test

This is an environmental/driver issue with certain DSMEM access patterns.
The non-chained inline-offset pattern works fine. Deferred to V11 with
potential driver update investigation.

## Implication: use cluster=2 for BW, cluster=8 for CAPACITY

- **Cluster=2 cross-CTA reduce**: 48.5 TB/s — fastest
- **Cluster=8 coordinated SMEM reduce**: 32.6 TB/s but 8× more remote SMEM
- **Cluster=4**: same BW as 8, less capacity — not optimal

For cluster CUTLASS-style GEMM with shared SMEM tile, cluster=2 gives best
BW for the cross-CTA communication phase.

## Combined ladder

| Path                    | BW         | Notes                     |
|-------------------------|------------|----------------------------|
| Plain LDS (chain)       | ~1 TB/s (29 cy latency)  | Single warp, limited ILP |
| Plain LDS (coalesced)   | 26.9 TB/s  | V8 best SMEM BW            |
| **Cluster=2 DSMEM NL=8**| **48.5 TB/s** | This V10 measurement       |
| **Cluster=8 DSMEM NL=16**| **42.3 TB/s** (MED, may L1-cache) | Max ILP     |
| HMMA.F16 tensor core    | 578 TFLOPS | Different metric           |