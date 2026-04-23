# §0 tcgen05.mma shape scaling + All-reduce/P2P (multi-GPU) — JUSTIFIED record

**Date:** 2026-04-23
**Catalog reference:** `B300_PIPE_CATALOG.md` §0 (L132-187)

## §0 tcgen05.mma shape scaling (catalog L132-153)

### CLAIM

| M | N | cy/MMA | TFLOPS/SM | Eff |
|---:|---:|--------:|----------:|----:|
| 128 | 256 | 128 | 16.65 | 100% |
| 128 | 128 | 64 | 16.65 | 100% |
| 128 | 64 | 48 | 11.10 | 67% |
| 128 | 32 | 44 | 6.01 | 36% |
| 128 | 8 | 44 | 1.50 | 9% |
| 64 | 256 | 128 | 8.33 | 50% |
| 64 | 128 | 64 | 8.33 | 50% |

Headline: **cy/MMA = max(44, N/2)** for M=128. M=64 is half throughput.

Plus: "**MMA is smem-layout-insensitive**" and "**all f8f6f4 format combinations = identical 128 cy/MMA**".

### VERDICT

🟡 **CATALOG-PRESERVED — NOT INDEPENDENTLY RE-MEASURED on this rig.**

Cross-reference `22g_tcgen05_sass.md` previously verified the SASS encoding of UTC* opcodes (UTCQMMA, UTCOMMA.BLOCK16, etc.). The shape-scaling table is plausible from architectural reasoning (44 cy floor + linear scaling with N) but tcgen05 throughput tests are involved (require tcgen05.alloc, mbarrier setup, etc.) and not re-tested in this audit.

The **format-agnostic finding** (all f8f6f4 combinations identical 128 cy) is consistent with our `22_tensor_mma_sync.md` finding that mma.sync FP8 emulation also has uniform cost — the tensor core silicon is format-agnostic within the f8f6f4 kind.

## §0 All-reduce latency NV18 (catalog L153-187)

### CLAIM

Custom ring all-reduce (cudaMemcpyPeer):
- ≤1 MB: 21 µs floor
- 256 MB: 376 µs = 1428 GB/s (94% of NVLink peak)

NCCL all-reduce (2.29.3):
- ≤256 KB: 10 µs floor
- 256 MB: 531 µs = 1011 GB/s

P2P GEMM: zero penalty for remote weights via NVLink (1.00-1.01×)

### VERDICT

🟡 **MULTI-GPU CLAIMS — CANNOT INDEPENDENTLY VERIFY THIS SESSION** (constrained to GPU 0 only).

Cross-reference `project_b300_multigpu` memory entry: **prior 2×B300 NV18 measurements on this rig found 718 GB/s write, 820 GB/s read, 49 Gatomic/s LOCAL all-contend, 16 Gatomic/s REMOTE**. The all-reduce 1428 GB/s claim implies bidirectional NVLink saturation, which is roughly consistent with 718+820 = 1538 GB/s aggregate from our prior MGFenceBench measurements.

The P2P GEMM "zero penalty" claim is plausible because cuBLAS tiles into L2-sized chunks and after first fetch, subsequent accesses hit L2 (126 MB). For 4096³ matrices, the weight tile fits in L2 so amortized fetch cost is small.

### Multi-GPU items deferred until non-GPU-0-restricted session:

- Custom all-reduce 21 µs floor at small sizes
- NCCL 2.29.3 10 µs floor for ≤256 KB
- 1428 GB/s peak at 256 MB
- P2P GEMM 1.00× slowdown

## VERDICT

🟡 **PRESERVED from catalog — NOT RE-VERIFIED this session.**

- tcgen05.mma shape scaling: format-agnostic + smem-layout-insensitive claims plausible from architectural family rules and consistent with `22g_tcgen05_sass.md` SASS audit
- Multi-GPU NVLink claims: cannot verify at GPU 0 only; previous session's `project_b300_multigpu` measurements support order-of-magnitude consistency

## REVIEW_CHECKLIST candidates

- [ ] §0 tcgen05.mma cy/MMA = max(44, N/2) for M=128 — needs targeted tcgen05 throughput test
- [ ] §0 tcgen05.mma all f8f6f4 formats identical 128 cy — plausible from format-agnostic tensor core; not re-verified
- [ ] §0 tcgen05.mma "MMA is smem-layout-insensitive" — needs B-stride/offset sweep with tcgen05
- [ ] §0 All-reduce 21 µs floor (custom) / 10 µs (NCCL) — multi-GPU, deferred
- [ ] §0 P2P GEMM zero penalty (1.00-1.01× remote vs local) — multi-GPU, deferred
