# M13: V8 Synthesis (B300 sm_103a)

V8 = 26/50 [x] + 24 deferred/hard items (tcgen05.mma, multicast TMA, NVFP4
scalefactor, some G/L items needing deeper infrastructure).

---

## Top V8 findings (most actionable)

### 1. `cuStreamBatchMemOp` = 0.11 µs/op — lowest-latency control plane
- Individual cuStreamWriteValue (V7 M6): 0.45 µs
- BatchMemOp 4 ops (V8 E2): **0.11 µs/op** (4× speedup)
- Producer-consumer semaphore (V8 E1): 0.99 µs/pair cross-stream
- Essentially single LDG cycle of latency

### 2. Kernel queue depth = 1024 per stream (user-corrected!)
- V8 M7 initial: claimed unbounded (test was just below limit at 1000)
- V8 M7b corrected: blocking starts at index 1024
- After fill: launch rate matches kernel completion rate
- Workarounds: multiple streams (V6 F1 → 128 streams), cudaGraph, persistent

### 3. UMA cold page-fault = 260× slower
- Cold UMA access: 10.7 ms for 64 MB (6.4 GB/s via fault path)
- Prefetched UMA: 81 µs (immediate, cached on GPU)
- ALWAYS prefetch UMA data before first GPU access

### 4. PCIe Gen 6 x16 ladder
- 1 B: 3.80 µs pinned / 6.20 µs pageable (setup floor)
- 4 MB: pinned peak ~55 GB/s
- ≥16 MB: saturates 57.8 GB/s (90% of ~64 GB/s spec)
- Pageable 33% slower peak (38 vs 58 GB/s)
- Pinned/pageable ratio peaks at **256 KB (2.66× diff)**

### 5. Cross-process zero-copy stack COMPLETE
- V5 I1: IPC handles 55 µs first-touch
- V8 K1: **device-side atomics work across processes** (atomicity preserved)
- V8 K2: Custom memory pools shareable via POSIX FD
- V8 F4: IPC handle scaling 0.73 µs each at 100 handles

### 6. 2:4 sparse HMMA = 2× throughput native
- V8 D1: HMMA.SP.16832.F32.BF16 SASS confirmed
- Same per-instruction cost as dense (15.5 cy)
- Covers 2× K depth → effective 2× throughput for sparse weights
- HIGH confidence (SASS verified, rigor applied)

### 7. NVLink P2P memcpy = 778 GB/s
- 86% of NVLink v7 spec ~900 GB/s per-direction
- 9% faster than kernel-direct write (V5 I2: 714 GB/s)
- Sweet spot ≥16 MB

### 8. Multi-GPU/concurrency overhead table
- P2P memcpy (V8 F1): 778 GB/s NVLink
- H2D + compute overlap (V8 H3): 1.29×
- H2D + D2H bidirectional (V8 H5): 1.59×
- Combined 3-stream (H2D | compute | D2H): near-PCIe-bound

### 9. DRAM tail latency spike
- Avg HBM access 60-80 cy (V6 G3 confirmed)
- Max tail: 1433 cy = 955 ns (23× avg)
- Caused by refresh or bank row activation
- For p99 SLA: budget ~1 µs outliers

### 10. Persistent kernel power (clarified)
- 1 SM × 1 thread spin: 0 W (invisible)
- 148 SMs × 1 thread each: +0.5 W
- 148 SMs × 256 threads SPIN: +7.4 W
- 148 SMs × 256 threads NANOSLEEP: +2.8 W (**62% savings vs spin**)

---

## Architectural facts confirmed/discovered

- **Queue depth = 1024** per stream (not unbounded)
- **HBM stack routing transparent** — no app-level balancing
- **L2 partition routing transparent** (V6 G1)
- **ECC inline** in HBM3E BW spec; 0.2% capacity tax
- **B300 has NO DLA** (data center GPU, not Jetson SoC)
- Cluster launch (V6 K2) is 17% FASTER than direct launch
- Scheduler look-ahead WEAK (compiler is the optimizer)
- Async error check is FREE (3.07 µs = same as no-check)
- Callback doesn't fire after kernel error (stream stuck until reset)
- WHILE conditional graph 18× faster than IF (V7)
- SWITCH 2× faster than IF

---

## ML-optimization cheatsheet

For ML inference on B300:

**Control plane (< 1 µs):**
- `cuStreamBatchMemOp`: 0.11 µs/op
- `cuStreamWriteValue`: 0.45 µs
- Persistent kernel batched (V7 J1): 38 ns/task at batch=64

**Latency-critical:**
- Spin SyncPolicy (V7 F2): 4% faster than default
- Cluster launch ≥4 (V6 K2): 17% faster + DSMEM
- Cluster all-reduce (V7 D1): 487 cy @ CSIZE=4

**Throughput-oriented:**
- cp.async depth=16 + prefetch.L2 + double-buffer (V7 I1+I3+I4): 3-4× stacked
- 2:4 sparse HMMA (V8 D1): 2× throughput for sparse weights
- Boost clock for mixed workloads (V6 C3): 3× lower energy than 510 MHz

**Cross-GPU:**
- NVLink P2P (V8 F1): 778 GB/s
- IPC handles + x-proc atomics + shareable pool: zero-copy multi-process

---

## Deferred to V9

Hard items remaining from V8:
- **A-series** (7 items): tcgen05.mma full descriptor (cuTLASS reference)
- **B-series** (4 items): multicast TMA + tensor descriptor (cute::TMA)
- **C-series** (3 items): NVFP4 scalefactor cvt
- **G-series** (5 items): real LLM inference (needs attention kernel)
- **L1-L4** (4 items): advanced tooling (per-kernel energy, waterfall, etc.)
- **M5, M8** (2 items): NVTX kernel markers, thread affinity

---

## V9 candidates

Beyond deferred:

1. **FlashAttention SoL kernel** (real ML)
2. **MoE routing overhead** characterization
3. **Token-level latency breakdown** (vs synthetic)
4. **Speculative decoding patterns**
5. **HBM3E sub-channel parallelism** (per-stack internal structure)
6. **Compute-communication fusion** (NVSHMEM-style)
7. **Power capping under 1100 W TDP**
8. **NUMA-aware multi-GPU patterns**

---

## V8 commits

All V8 progress committed. Tools added: `utils/workload_classify.sh` (V8 L5).

---

## Source commits
- V8 E2 BatchMemOp: 3a17180
- V8 M7b queue depth: bfaef30
- V8 H4 UMA page-fault: bcb0995
- V8 H1 PCIe sweep: dc0499f
- V8 K1 x-proc atom: 45ef5e1
- V8 K2 shareable pool: 4528be1
- V8 D1 sparse HMMA: 9b7b759
- V8 F1 P2P memcpy: 88ee0cf
- V8 H2b pinned sweep: ea5e153
- V8 J4b persistent power: c72b36b
- V8 L5 classifier: 3601844
