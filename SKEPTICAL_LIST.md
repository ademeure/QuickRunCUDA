# B300 Findings — Skeptical Audit

Dated self-review: stuff I reported but shouldn't have trusted, and gaps still open.

## CONFIRMED WRONG (must retest)

### 1. "HBM peak 5.16 TB/s via ld.global"
- `bench_hbm_peak.cu` uses `offset & 0xFFFFFF` → 16M uint4 slots = **256 MB** range.
- 4736 blocks × 128 threads × 1000 iters re-traverse that 256 MB ~38×.
- B300 L2 ≈ 126 MB, so after first few hundred cycles this is **mostly L2 hits**, not cold HBM.
- The "5.16 TB/s" number is actually a **mixed L2+DRAM** number.

### 2. "TMA peak 6.83 TB/s"
- `bench_tma_peak.cu` block offset = `blockIdx.x × 50 × 8192` = 50 MB per block.
- At 148 blocks, total span = 7.4 GB — but `A` default buffer is 256 MB, so the `(src − A) & 0xFFFFFFFF` wrap potentially reads garbage / unallocated.
- Even the 148-block case only touches 60 MB total, which **fits in L2**. Larger block counts (296, 1480) reporting 12 → 74 TB/s proves the test is L2-limited, not HBM.

### 3. "Single warp can sustain 30+ in-flight loads at 19.7 GB/s"
- `bench_inflight.cu` working set = 31 + 99×32 + 63×1024 = **67 711 u32 = 264 KB**.
- That fits in L1/L2. The "latency hiding" is **L1/L2** latency, not HBM.
- Per-warp "19.7 GB/s" is an L1 number, not an HBM number.
- What I can legitimately claim: a single warp can keep 64 outstanding L1/L2 loads in flight.

### 4. Label mismatch: "FFMA2 peak" is actually scalar FFMA
- `bench_ffma_occ.cu` uses `fma.rn.f32` (scalar), but table header says "FFMA2 Multi-warp Peak".
- Real packed FFMA2 (`fma.rn.f32x2`) is in `bench_ffma2_occ.cu` and was **never re-run at 32 warps** with updated counter.
- Need to either rename the table or run actual FFMA2 to see if packed beats scalar.

## SUSPICIOUS (need SASS / ncu proof before claiming)

### 5. `atom.acquire.cta` = 797 cy (23× relaxed)
- If ld.acquire.cta is only 5% over relaxed, why would atom.acquire.cta be 23×?
- Likely: ptxas emits a pre-fence (e.g. `MEMBAR.SYS` or similar) regardless of `.cta` because of atomic semantics.
- **Verify via SASS.** If the SASS shows a `FENCE.*.GPU` or membar inserted, the `.cta` hint is being ignored.

### 6. `atom.release.cta` = 36 cy "FREE"
- If release.cta is free but release.cluster is 892 cy (26×), that's a large discontinuity.
- Same address, same L2 path — only the fence scope differs. Needs SASS check.

### 7. "Cacheline ping-pong: 4 addrs same cacheline = 1301 cy, 10× worse than 1 addr"
- This measures only CTA0's perspective. With 148 CTAs serializing on 4 hot addresses, CTA0's walltime is (overall_atomics / 4-way_parallelism). Might be a measurement artifact, not actual L2 thrashing.
- Cross-check: ncu `l2_tex_hit_rate`, `l2_atomic_throughput`.

### 8. "HFMA2 = 71.5 TFLOPS = same as FFMA2 (dispatch limit)"
- Measured 125.8 FMAs/cy/SM for HFMA2 vs 126.3 for FFMA scalar — confirms dispatch parity.
- **Interpretation may be right** but 70 TFLOPS is FP32 spec, not FP16. Expect FP16 ≥ 2× FP32 in most arches.
- Either B300 genuinely caps FP16 non-tensor at FP32 rate (a real arch choice), or HFMA2 kernel has a hidden issue.
- **Cross-check**: use ncu `smsp__inst_executed_pipe_fma` to confirm HFMA2 dispatches at half the rate of expected.

### 9. "64 warps/SM verified"
- Saved as memory but actual evidence was indirect. Needs re-probe via `%nwarpid` across all active warps.

### 10. "Predicated execution = ZERO overhead, any predicate value"
- Tested with only 1 warp, 1 CTA. At full occupancy with divergent predicates, expect some cost.
- Need multi-warp, divergent predicate test.

### 11. "Compute-memory overlap: 8 FFMAs FREE when interleaved"
- Tested with 1 warp only. At full occupancy the LSU and ALU may contend. Retest at 32 warps.

## UNTESTED GAPS (genuinely missing)

### Memory / storage
- [ ] **Cold HBM proper** — 4 GB+ working set, stride to cover all channels
- [ ] **HBM BW vs N_blocks curve** — find the inflection point
- [ ] **Host pinned memory (zero-copy) BW + latency** from GPU
- [ ] **L2 replacement policy** — LRU? or FIFO? probe via warmup + eviction
- [ ] **Constant memory latency curve** — 1 access vs chain
- [ ] **Texture path** — does it still exist, is it faster than L1 for 1D/2D?
- [ ] **`ld.global.nc`** — vs `.ca`/`.cg` cold / warm

### Compute
- [ ] **True packed FFMA2 (`fma.rn.f32x2`) at 32 warps** — paired test
- [ ] **BF16 fma** throughput
- [ ] **Double-precision FFMA** throughput (B300 FP64 was removed on sm_103a? verify)
- [ ] **IMAD/IMUL throughput** — assumed = FFMA but not measured
- [ ] **Scalar ALU mix (ADD/AND/XOR/SHF)** full peak

### Tensor
- [ ] **tcgen05.cp** (smem↔TMEM bulk) throughput + latency
- [ ] **Sparse FP8 with proper 2:4 metadata** — gap from 7.44 → 9.3 spec
- [ ] **mxf8f6f4 / mxf4 block-scaled** — if compiler issues resolve

### Scheduler / launch
- [ ] **cudaGraph launch latency** vs individual launch
- [ ] **Cooperative launch (cudaLaunchCooperativeKernel)**
- [ ] **CTA-to-SM placement mapping** (do `blockIdx.x` and SM id correlate?)

### Warp / SM
- [ ] **64 warps/SM** re-probe
- [ ] **Multi-warp predicated execution**
- [ ] **Multi-warp compute-memory overlap**

### Fence / atomic
- [ ] **atom.acquire.cta SASS** — does ptxas respect `.cta`?
- [ ] **Cacheline ping-pong** — ncu cross-check L2 atomic counters
