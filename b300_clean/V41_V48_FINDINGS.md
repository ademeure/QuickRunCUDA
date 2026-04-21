# V41-V48 Free-Rein SoL Findings (continuation of V32-V40)

After completing prior tests, this continuation focuses on TMA pipelining,
ALU pipe variants, and SMEM bank conflict reality. All applied 10-rule rigor.

## Headlines

### TMA SoL (V46 = NEW BEST)
**TMA pipelined 8-deep = 7.20 TB/s = 98.5% of HBM peak**

Previous best was V33 at 6.72 TB/s (single-deep). Pipelining gives 2.5×
speedup vs single-deep due to hidden mbarrier wait latency.

| Config | TB/s | %HBM |
|--------|------|------|
| V33 single-deep 64 KB | 6.72 | 92% |
| V34 write 32 KB | 7.17 | 98% |
| V46 read 8-deep 16 KB | **7.20** | **98.5%** |
| V47 write 8-deep | 6.34 | 87% (no benefit) |

**Key**: Reads need pipelining; writes are already async fire-and-forget.

### TMA multicast SoL
V32: 14.9 TB/s effective at 64 KB × 1-deep × 18 clusters.
V48: 13.96 TB/s at 32 KB × 2-deep × 18 clusters (no improvement).
**Multicast can't be pipelined — single engine per cluster.**

## ALU pipe (V41 — MUFU sweep)
EX2 stands alone at 95.8% of 1/(4cy)/SMSP SoL (9.22 Gops/s).
All others (LG2, RCP, RSQRT, SQRT, SIN, COS) at 49% — half rate.
**EX2 is 2× faster than other transcendentals on B300.**

## ALU pipe (V40 — basic ops)
| Op | Glane/s | %SoL | Pipe |
|----|---------|------|------|
| FFMA/FADD/IADD3 | 25-26 | 67% | FMA |
| LOP3/IMUL | 18.7 | 48% | INT-bit (half rate) |
| PRMT | 13.9 | 36% | permute |
| ISETP | 8.4 | 22% | compare |

## SMEM bank conflicts (V44+V45)
Two regimes:
- **Latency-bound (V44)**: 32-way conflict ~2× cost (chain-serial)
- **Throughput-bound (V45)**: 32-way conflict ~1× cost (warp scheduler hides)

For typical CUDA throughput kernels (GEMM/conv): bank conflicts are MUCH cheaper
than the CUDA C programming guide implies (which uses serialization model).

## TMA + prefetch.L2 (V42)
Counter-intuitive: prefetch.L2 + TMA = **27% SLOWER** than no-prefetch.
TMA has its own DMA path; explicit prefetch instructions block forward progress.
**Rule: never combine prefetch.L2 with cp.async.bulk.**

(V6 I3 1.58× speedup was for old cp.async — not bulk.)

## Packed FP cvt (V43, partial)
- e4m3x2 / e5m2x2 (FP8): 17.6 Gelem/s
- bf16x2 / f16x2: 9.05 Gelem/s
- **FP8 cvt 2× faster than BF16 cvt** (output bit-width hypothesis)

CUDA 13.2 BUG: `cvt.rn.satfinite.e2m1x4.f32` rejected on sm_103a despite
being valid in CUDA 12.x. Need PTX syntax migration.

## REDUX/SHFL pipe (V37+V38)
Both SHFL and REDUX share the same shuffle pipe at 1 inst/(4cy)/SMSP.
Peak: ~9.5 Telements/s.
Catalog "redux 4× SHFL" was algorithm-level (replaces tree); raw rate equal.

## Test ladder summary
| Test | What | Peak |
|------|------|------|
| V32 | TMA multicast aggregate | 14.9 TB/s effective |
| V33 | TMA per-CTA read | 6.72 TB/s (92%) |
| V34 | TMA write | 7.17 TB/s (98%) |
| V35-36 | TMA stream copy | 6.21 TB/s R+W (93%) |
| V37 | REDUX peak | 9.09 Telements/s |
| V38 | SHFL peak | 9.48 Telements/s |
| V39 | PRMT peak | 13.9 Glane/s (48%) |
| V40 | ALU pipe ladder | FFMA/FADD/IADD = top tier |
| V41 | MUFU sweep | EX2 9.22, others 4.74 Gops/s |
| V42 | TMA+prefetch | -27% (don't combine) |
| V43 | Packed FP cvt | FP8 2× BF16 |
| V44 | SMEM conflict latency | 32-way ~2× |
| V45 | SMEM conflict throughput | 32-way ~1× (hidden) |
| V46 | TMA pipelined | **7.20 TB/s = 98.5%** ← NEW READ SoL |
| V47 | TMA write pipelined | 6.34 (no benefit) |
| V48 | Multicast pipelined | 13.96 (capped) |

## Methodology gains
- Rule 3 caught: V33 L2-cache (10.84→6.72), V39 LICM (1547%→48%), V44/V45 chain-vs-throughput
- Rule 9 caught: V44 surprising "2-3× conflict" needed V45 verification
- NCU cross-check confirms V46 dram bytes (620/605 = 102%)
