# M12: V7 Final Synthesis (B300 sm_103a)

V7 = 44/50 [x] + 6 deferred to V8 (tcgen05/multicast/NVFP4 advanced).

---

## Top V7 findings (most actionable)

### 1. cp.async DEEP optimizations stack to ~3-4× speedup
- V7 I1: depth=16 → 15× per-cp_async speedup
- V6 I3: prefetch.L2 ahead → 1.58× speedup
- V7 I4: double-buffer → 1.94× speedup
- V7 I2: wait_group N → 6% extra
- **Compound**: deep + prefetch + double-buffer = potential 3-4× over naive

### 2. cuStreamWriteValue is HIDDEN GEM
- V7 M6: 0.45 µs/op (5-6× faster than kernel launch)
- For ultra-low-latency host↔GPU control plane (token decoding, multi-stream coord)

### 3. Persistent kernel patterns (V7 J-series)
- J1: batched persistent at batch=64 = **38 ns/task** (74× per-task speedup)
- J2: work-stealing 16 blocks = 96% efficient; 148 = 52% (atomic contention)
- J4: persistent kernel adds **~0 W power** (static 167 W paid regardless)

### 4. Conditional graphs hierarchy (V7 E)
- WHILE skipped: 28.86 µs (FAST)
- SWITCH 1 case: 241 µs
- IF skipped (V6 D5): 532 µs
- ExecUpdate: 0.26 µs/node (4-16× faster than re-instantiate)

### 5. Architectural facts confirmed
- L2 = 126.5 MB
- HBM single-load = 54 ns
- NO SM power-gating (static constant)
- Boost 2032 MHz holds at 552 W (no throttle)
- Cluster max = 8 (V5 C5)
- DRAM bank conflicts: 64KB stride 29% slower
- SchedHW look-ahead WEAK (compiler is the optimizer)

### 6. Cluster patterns
- DSMEM broadcast 2.4× faster than global (V7 D2)
- All-reduce primitive 487-537 cy (V7 D1)
- Cluster atomic 21% slower than per-CTA (V7 D5)

### 7. Memory hierarchy
- Sequential L2 hit 94%, Random 35% (V7 G2)
- Higher occupancy = HIGHER L1 hit (myth-buster, V7 G3)
- TLB transparent up to 1 GB (V7 G6)
- L2 LRU effective for hot data (V7 G5)

### 8. Numeric cvt
- All narrow FP cvt 5.4 cy (FP4/FP6/FP8/FP16 identical)
- INT8 cvt 2.9 cy (V6 H4)
- satfinite is FREE (V7 H2)
- Only .rn rounding for narrow FP (V7 H1)
- Inf → max (clamp), NaN → NaN-encoded (V7 H3)

### 9. Stream/sync overhead
- Non-blocking stream = default (3.08 µs); regular custom +20% (V6 F4)
- Cluster launch (≥4) FASTER than direct (V6 K2)
- Spin SyncPolicy 4% faster; BlockingSync 70% slower (V7 F2)
- Event chain 3.82 µs/event asymptotic (V6 F3)

### 10. Energy
- Workload-dependent min-energy clock (V7 K2 + M11):
  - FFMA-bound: 510 MHz (16% savings)
  - Memory-bound: 800 MHz (36% savings)
  - **Mixed ML: BOOST 1992 MHz (3× lower than 510!)**
- Single-warp = 75× worse efficiency vs full occupancy (V6 C4 / V7 K4)

---

## Deferred to V8 (hard items)

Mostly require cuTLASS reference for tcgen05.mma full descriptor encoding:

1. **A-series** (7 items): tcgen05.mma working with valid descriptors
   - Needs cute::SmemDescriptor + idesc bit packing
   - cuTLASS source `mma_sm100_umma.hpp` shows 5-arg + 4-tuple format
2. **B-series** (4 items): multicast TMA + tensor descriptor
   - Needs cuTensorMapEncodeTiled + DSMEM mbarrier setup
3. **C-series** (3 items): NVFP4 scalefactor cvt variant
   - cvt.scalefactor for per-block scale (8-element groups)
4. **F3**: cudaMemcpyAsync peer-to-peer between streams (multi-GPU)
5. **L-series V3** (5 items): advanced tooling synthesis
6. **M2**: Dynamic Parallelism + tcgen05 mix

---

## Tools delivered V7

| Tool | Source |
|------|--------|
| Per-pipe latency reference | `utils/warp_latency.sh` (V6 L3) |
| Roofline plotter | `utils/roofline.sh` (V6 L5) |
| Pipe overlap matrix | `utils/overlap_matrix.sh` (V6 L1) |
| Microbench template | `utils/mkbench.sh` (V6 L5) |
| SASS diff | `utils/sass_diff.sh` (V6 L4) |
| Pipe dashboard | `utils/pipe_dashboard.sh` (V6 L3) |

---

## Headlines for ML practitioners

If you're optimizing ML inference on B300:

1. **Use boost clock** — it's lowest energy AND lowest latency for mixed workloads
2. **Maximize occupancy** — no SM power-gating, static cost paid regardless
3. **Use cuStreamWriteValue** for control plane (5-6× faster than kernels)
4. **Use persistent kernels with batching** for token-decoding (38 ns/task at batch=64)
5. **For cuTLASS GEMMs**: prefetch.L2 + cp.async depth=16 + double-buffer = ~3-4×
6. **Cluster launch ≥4** for HMMA-bound (DSMEM affinity + 17% faster launch)
7. **Pass scalars as kernel args** (cmem b0 = 3× faster than global LDG)
8. **Avoid conditional graph IF** (532 µs); use WHILE (28 µs) or SWITCH (241 µs)
9. **Use cudaStreamNonBlocking** flag (regular streams are 20% slower)
10. **128 streams = 108× speedup** for batched inference (saturates here)

---

## V8 candidates (next session)

Beyond V7 deferred items:

1. **Sparsity in tcgen05.mma** (50% sparse weights)
2. **Distributed L2 cache hint** for cross-cluster reuse
3. **Microsecond-scale autotuning** (find optimal block size per kernel)
4. **Cross-process atomic semantics** via IPC
5. **Per-token energy in real LLM inference** (vs synthetic FFMA/memory)
6. **HBM3E channel parallelism** (12 stacks; how does scheduler route?)
7. **PCIe Gen 6 x16 host-device transfer optimal patterns**
8. **NVLink chain-of-tensors patterns** (multi-hop)
