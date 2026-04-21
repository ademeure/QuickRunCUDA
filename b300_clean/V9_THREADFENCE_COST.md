# V9: __threadfence variants — scope ≈ cost

## Measurements

1000-call chain, single thread. All values include ~23 cy loop overhead.

| Fence                   | Total cy/call | Fence cost (minus loop) | ns @ 2.032 GHz |
|-------------------------|---------------|--------------------------|-----------------|
| baseline (syncwarp)     | 23.00         | 0 (reference)            | 11              |
| __threadfence_block()   | 23.00         | **~0** (essentially free) | 11              |
| __threadfence() (GPU)   | 280.91        | **~258**                 | 138             |
| __threadfence_system()  | 3042.18       | **~3019**                | 1486            |

## Interpretation

- **__threadfence_block()**: essentially free when no contention (CTA-scope
  fence ≈ scoreboard wait, which is cheap for a single thread).
- **__threadfence() (GPU scope)**: ~258 cy = matches L2 round-trip. Forces
  visibility of this thread's writes to the entire GPU.
- **__threadfence_system()**: ~3019 cy = **12× GPU fence**. Forces host
  and peer-GPU visibility via PCIe/NVLink coherence.

## 10-rule rigor

1. **Theoretical**: scope hierarchy (block ⊂ GPU ⊂ system) → costs should
   scale with coherence reach. Confirmed.
2. **Measured**: 23 / 281 / 3042 cy. Each tier ~10× more than the last.
3. Rule 3: all plausible (block=free since single-thread).
4. **Block fence cheap**: intra-CTA memory ordering hardware-fast.
5. **GPU fence ~L2 round-trip**: 258 cy aligns with L2 hit latency (300 cy
   pointer chase). Fence needs all prior ops to drain to L2.
6. **SASS** likely emits `MEMBAR.CTA`, `MEMBAR.GPU`, `MEMBAR.SYS`.
7. Three methods: clock64 + chain-length stable + baseline subtract.
8. Conclusive: tests disambiguated by scope qualifier only; all else equal.
9. Initial "baseline=1991 cy" bug (stale raw from prior run) flagged by rule 9
   — fixed by `rm -f raw` before each config.
10. **Confidence: HIGH**.

## Practical implications

**Use finest-scope fence for what you need:**
- Intra-CTA communication (producer-consumer within block): `__threadfence_block()` FREE
- Global flag visible across GPU (persistent kernel coordination): `__threadfence()` 138 ns
- Host-visible mailbox (CPU polls GPU memory): `__threadfence_system()` 1.5 µs

## Combined with V9 atomic findings

Atomic + fence patterns:
- atomic.cta + fence_block: producer-consumer in CTA — near-zero coherence cost
- atomic.sys alone: implicit system fence → 697 cy latency
- atomic (default scope) WITHOUT fence: no ordering guarantee across warps

**Rule of thumb**: if using scoped atomics (.cta or .gpu), pair with matching
fence for sequential consistency.

## Latency ladder update (complete)

| Op / primitive                 | Cy    | ns @ 2.032 GHz |
|---------------------------------|-------|-----------------|
| FFMA                            | 4     | 2               |
| HMMA.F16                        | 20    | 10              |
| __syncwarp / fence_block        | 23    | 11              |
| SMEM LDS                        | 29    | 14              |
| __syncthreads(4 warps)          | 30    | 15              |
| L1 hit                          | 47    | 23              |
| DFMA                            | 64    | 31              |
| L2 hit                          | ~300  | 148             |
| DRAM                            | ~317  | 156             |
| **__threadfence() (GPU)**       | **281** | **138**       |
| barrier.cluster                 | 370   | 182             |
| global atomic (chained)         | 697   | 343             |
| **__threadfence_system()**      | **3019** | **1486**     |