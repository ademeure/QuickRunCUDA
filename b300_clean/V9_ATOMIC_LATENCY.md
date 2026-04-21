# V9: Atomic latency = 697 cy (scope doesn't affect single-thread latency)

## CORRECTED MEASUREMENT

My initial test claimed atomic.sys = 752 cy vs atom.cta/gpu = 43 cy (17× gap).
**This was WRONG** — apples-to-oranges comparison. Default test chained via
address (v-dependent addr) while scoped test hit fixed addr with no chain →
multiple atomics pipelined, measuring throughput not latency.

## Fair re-measurement (all with value-dependency chain, same address)

Each atomic: `v = atomicAdd(A, v)` so return value chains into next op.

| Scope        | Latency (cy/op) | ns @ 2.032 GHz |
|--------------|------------------|-----------------|
| atom (sys)   | 697.0           | 343             |
| atom.cta     | 697.0           | 343             |
| atom.gpu     | 696.9           | 343             |

**All three identical: ~697 cy per atomic when dependency chained.**

## Interpretation

- **Scope does NOT affect single-thread atomic latency.** Coherence overhead
  only matters when peers actually observe — single-thread serial chain
  doesn't trigger it.
- **697 cy = full L2 atomic round-trip** (read-modify-write-return to SM).
- Without chain dependency, HW pipelines multiple atomics → 43 cy apparent
  "throughput" per op (14× speedup via pipelining).

## When scope DOES matter

Scope affects:
- **Cross-process/GPU coherence**: `.sys` scope waits for host to see
- **Memory ordering**: `.cta` only guarantees visibility within block
- **Cache invalidation**: `.cta` can short-circuit cross-GPU snoops

For PURE latency of a single serial-chain atomic, scope is irrelevant.

For **parallel atomic throughput** (V8 / prior catalog 9475 Gops/s SMEM,
4 TB/s global), scope matters because coherence overhead blocks pipelining.

## Correct atomic performance model

Single-thread latency (dep-chained): **697 cy**
Pipelined throughput (independent atomics): ~**43 cy effective** at SM
Contended throughput (N threads same addr): much higher per-op cost

## 10-rule lesson (rule 9: suspect test before HW)

Rule 9 caught this: initial "17× speedup from scope" was too dramatic.
Re-examining test showed ADDRESS variation in default vs FIXED address
in scoped variant — not a fair comparison. Fixed test shows scope is
latency-neutral for single-thread serial.

## Confidence

**HIGH for corrected 697 cy latency** — reproducible, all scopes identical.
**REJECTED** earlier "17× gap" claim — was measurement artifact.

## Latency ladder update

| Op                   | Latency (cy)  |
|----------------------|---------------|
| SMEM LDS             | 29            |
| L1 hit               | 47            |
| __syncthreads(4 warps)| 30           |
| L2 hit (pointer chase)| ~300         |
| DRAM                 | 317           |
| barrier.cluster      | 370           |
| **global atomic (chained)** | **697** |

Global atomic in serial chain = **2× DRAM latency** — fits "read + atomic unit + write" model.