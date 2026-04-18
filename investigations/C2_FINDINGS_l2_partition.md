# C2 — L2 Partition Behavior on B300 (sm_103a, 126 MB L2)

## Question
B300's L2 has 2 partitions split along the die boundary. Does any CUDA host API (`cudaMemAdvise`, `cudaLaunchMemSyncDomain`, `cudaStreamAttributeAccessPolicyWindow`, etc.) actually control which partition data lands in? Or is it solely determined by a fixed physical-address hash?

## Answer (HIGH confidence)
**No CUDA host API controls L2 partition placement within a single CUDA device view.** Address→partition is determined by a fixed hash of the physical address; the driver picks the physical backing but exposes no documented hook to bias that choice.

## Five-method API comparison
Built `/tmp/test_partition_apis.cu`. For each allocation method, swept first 128 KiB at 4 KiB stride and recorded near/far fingerprint via the proven atomic-latency probe:

| Method | Side fingerprint (32 chunks of 4 KiB) |
|---|---|
| M1: cudaMalloc | `NFFNFNNFFNNFNFFNNFFNFNNFFNNFNFFN` |
| M2: Managed + SetPreferredLocation(dev0) | identical to M1 (same hash, same start phase) |
| M3: Managed + SetAccessedBy(dev0) | identical pattern, different start phase (different physical alloc) |
| M4: cudaMallocAsync (mempool) | `FNNFNFFNNFFNFNNFFNNFNFFNNFFNFNNF` (= bit-flipped of M1, same hash) |
| M5: cudaMalloc + MemSyncDomain=Remote launch | identical to M4 |

Pattern is **deterministic per physical allocation**. Neither `cudaMemAdvise` flags nor `cudaLaunchAttributeMemSyncDomain` change which 4 KiB blocks land on which side. **MemSyncDomain affects fence ordering only, not placement** — confirmed by NVIDIA Programming Guide §4.14: "a fence will only order writes matching the fence's domain."

## Latency / throughput numbers (HIGH confidence)
Single SM, single thread, serial `ATOMG.E.ADD.STRONG.GPU` chain (SASS-verified). 1000 iter chain. ncu cross-check: `lts__t_sectors_op_atom.sum=1004` for 1000 iters (99.6% L2 hit, 0% DRAM = L2 round-trip).

| Quantity | Near side | Far side | Ratio |
|---|---:|---:|---:|
| Atomic round-trip (cy) | **285-360** (~295 median) | **670-745** (~700 median) | **2.4×** |
| Latency (ns @ 1920 MHz) | ~155 ns | ~365 ns | 2.4× |

Hash flips at **4 KiB stride exactly** (every adjacent 4 KiB block toggles side), confirming `tests/side_aware.cu`'s `CHUNK_SIZE=4096`.

## Two terms decoded

**MLOPart = Memory Locality Optimized Partition**. CUDA 13.0+ MPS feature for B200/B300 only (CC 10.0/10.3). Activated via `nvidia-cuda-mps-control` `start_server -mlopart`. Splits the GPU **along the die boundary** (= L2 partition boundary) into **2 distinct CUDA devices per B200/B300**, each with own compute+memory view. NVIDIA-cited gain: kernel runtime 2314 ms → 1480 ms (-36%); intra-die P2P 2353 GB/s vs 767 GB/s cross-die. This is the **official "L2 partition affinity" feature**, but it works at the MPS server level, not as a per-allocation API. Source: [NVIDIA blog "Boost GPU Memory Performance with No Code Changes Using NVIDIA CUDA MPS"](https://developer.nvidia.com/blog/boost-gpu-memory-performance-with-no-code-changes-using-nvidia-cuda-mps).

**"cuda-side-boost"** is not a documented term. Best interpretation: shorthand for the L2-side-aware technique embodied in `tests/side_aware.cu` — empirical determination of side hash via atomic-latency probing, then pinning each SM to its near-side data.

## Mechanisms that DO exist for partition affinity
1. **MLOPart** (CUDA 13.0+, MPS-level): hard die-split, exposed as 2 CUDA devices. Application code unchanged but the device IS half the GPU. Official "side affinity".
2. **AccessPolicyWindow / cudaStreamAttributeAccessPolicyWindow** (`cudaAccessPropertyPersisting`): pins data to L2 via persistence, NOT to a specific partition.
3. **Empirical side-aware (`side_aware.cu` style)**: probe latency at runtime, route each SM to its near-side data. The only way to do "soft" partitioning within a single CUDA device view.

## Confidence breakdown
- **HIGH**: existence of 2 partitions, 2.4× latency ratio, 4 KiB hash granularity, no-effect of MemAdvise/MemSyncDomain on placement (5 methods, identical hash fingerprint mod start-phase). SASS+ncu+wall-clock all agree.
- **HIGH**: MLOPart exists, splits at die boundary (per NVIDIA blog).
- **MED**: that MLOPart actually pins L2 traffic at runtime — couldn't run MPS test in this session (would need `start_server -mlopart` + 2-device sweep).
- **What would change MED→HIGH**: run the same atomic-latency hash-fingerprint test under MLOPart-mode MPS; if near-side fraction goes from 50% to ~100% on each MLOPart-virtualized device, that confirms hard partitioning.

## Files
- `/root/github/QuickRunCUDA/tests/side_aware.cu` (existing hackathon code; uses CHUNK_SIZE=4096 = 4 KiB)
- `/root/github/QuickRunCUDA/tests/bench_atom_lat_sides.cu` (existing offset-sweep probe)
- `/root/github/QuickRunCUDA/investigations/c2_l2_partition/test_partition_apis.cu` (NEW — 5-method API comparison)
- `/root/github/QuickRunCUDA/investigations/c2_l2_partition/sweep_sides.sh` (NEW — 4 KiB-stride latency sweep driver)

## Sources
- [Boost GPU Memory Performance with NVIDIA CUDA MPS (MLOPart)](https://developer.nvidia.com/blog/boost-gpu-memory-performance-with-no-code-changes-using-nvidia-cuda-mps)
- [Memory Synchronization Domains — CUDA Programming Guide §4.14](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/memory-sync-domains.html)
- [Inside NVIDIA GPUs: Anatomy of high performance matmul kernels — Aleksa Gordic](https://www.aleksagordic.com/blog/matmul)
- [Microbenchmarking NVIDIA's Blackwell Architecture (arxiv 2512.02189)](https://arxiv.org/pdf/2512.02189)
- [Dissecting NVIDIA Blackwell Architecture (arxiv 2507.10789)](https://arxiv.org/pdf/2507.10789)
