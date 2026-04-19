# NVFP4 cuBLAS Parallel Streams — Recovering Small-Shape MFU

Date: 2026-04-19. Free-rein continuation.
Test: same cuBLAS Lt NVF4 zero-data on N parallel streams concurrently.

## Setup
Built `/tmp/cublas_streams` — modified cuBLAS NVF4 binary that creates N
streams with N independent A/B/C buffers. Launches the same matmul on all
streams in a single iteration (one "wave"), measures aggregate throughput.

NS sweep: 1, 2, 4, 8, 16. Each stream owns its own buffers (no contention).
Workspace 256 MB per stream.

## Boost (zero data) results

| Shape | NS=1 MFU | NS=2 MFU | NS=4-16 MFU | Best gain |
|-------|---------:|---------:|------------:|----------:|
| 2048² K=3072 | 25.7% | 27.9% | 24-27% | +9% |
| **4096² K=3072** | 30.3% | **36.3%** | 34-35% | **+20%** |
| 2048² K=6144 | 35.9% | 38.6% | 36-37% | +8% |
| **4096² K=6144** | 37.6% | **44.6%** | 44-45% | **+19%** |
| 8192² K=3072 | 40.5% | 43.3% | 43% | +7% |
| 8192² K=6144 | 47.8% | 50.4% | 50% | +5% |
| **4096² K=12288** | 43.2% | **50.8%** | 51% | **+18%** |
| 8192² K=12288 | 68.5% | 71.7% | 71% | +5% |

## At -lgc 510 MHz (at-clock MFU vs 3766 TF peak)

| Shape | NS=1 %@510 | NS=2 %@510 | Gain |
|-------|-----------:|-----------:|-----:|
| 2048² K=3072 | 29.8% | 30.9% | +1pp |
| **4096² K=6144** | 43.0% | **51.3%** | **+8.3pp** |
| 8192² K=6144 | 54.8% | 56.9% | +2pp |

Same relative gain (~19%) on 4K² K=6K at boost AND 510 MHz — parallel streams
help equally regardless of clock.

## Findings

1. **NS=2 is the universal sweet spot**. NS=4/8/16 give zero additional benefit.
   Cluster scheduler caps at ~15 max active clusters; one small kernel uses
   ~7 clusters → two parallel streams saturate the 15 slots.
2. **Medium-small shapes** (4K² K=6-12K) benefit most — they had spare cluster
   capacity that NS=2 fills.
3. **Tiny shapes** (2K²) get only ~8% gain — too few tiles even doubled.
4. **Already-saturating shapes** (8K² K=12K) get only ~5% — they're already
   near the 73% MFU ceiling.
5. **Low-clock regime**: parallel streams help less in absolute pp because
   compute pipeline is the gate, not cluster availability. But relative gain
   is similar.

## What about PDL?

PDL (Programmatic Dependent Launch) reduces back-to-back same-stream launch
latency by letting kernel N+1 start initializing while kernel N finishes.
Set per-launch via `cudaLaunchAttributeProgrammaticStreamSerialization`.

**cuBLAS Lt does NOT expose this attribute** — would need to wrap cuLaunchKernel
manually, which is complex.

PDL prior research (commit `af96f51`) showed ~2-5 us savings per launch on
sub-100us kernels. For small-shape NVF4 (~30 us), maybe ~10% wall-clock gain.
Less than the ~20% from parallel streams + simpler.

**Recommendation**: use cublasLtMatmul on 2 parallel streams for small NVF4
shapes — recovers ~20% throughput vs single-stream.

## Confidence

- HIGH on NS=2 sweet spot (consistent across all shapes tested)
- HIGH on per-shape gain magnitudes (multiple replications agree within ±1pp)
- MED on the cause being cluster scheduling (not directly measured ncu max
  active clusters changing with NS)
- MED on PDL prediction (~10% gain estimate based on prior work, not measured
  with cuBLAS specifically)
