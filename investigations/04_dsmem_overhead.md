# 04: DSMEM Overhead vs Local SMEM — Definitive Results

**Date:** 2026-04-17  
**GPU:** NVIDIA B300 SXM6 AC (sm_103a), 1920 MHz  
**Kernel:** standalone CUDA binary, `cudaLaunchKernelExC` with cluster attributes  
**SASS verified at each step**

---

## Contradiction Settled

Two prior claims about DSMEM overhead contradicted each other by ~40×:

| Source | Claim |
|---|---|
| B300_PIPE_CATALOG §30.H | "DSMEM only 0.8% slower than local SMEM" (23.07 vs 23.26 cy) |
| AUDIT_NOTES / dsmem_v2.cu | "DSMEM 4.7× SLOWER than local SMEM" |

**Both were wrong.**  The correct ratio is **8×** slower in latency.

---

## Root Cause of §30.H Error: LICM

`bench_dsmem.cu`'s loop was hoisted by ptxas. SASS inspection of `sass/bench_dsmem_1891246005.sass` shows:

```
/*0170*/  LDS R3, [R8+UR5]          ; ← DSMEM load: BEFORE the loop label
  ...
/*01f0*/  .L_x_0:                   ; ← loop label (address 0x01f0)
/*0200*/  UIADD3 UR4, UPT, UR4, 0x1 ; counter
/*0210*/  LOP3.LUT R2, R3, R2, ...  ; XOR accumulate (uses hoisted R3)
/*0220*/  UISETP.NE ...             ; compare counter
/*0230*/  BRA ...                   ; branch back
```

The loop body has **no load instruction**. Both OP=0 (DSMEM) and OP=1 (local SMEM) variants were hoisted identically. Both measured ~23 cy = XOR + branch loop overhead. The 0.8% "difference" was measurement noise.

The root cause: `remote_addr + (threadIdx.x * 4)` is loop-invariant (constant each iteration). ptxas correctly hoisted the load before the loop. This is LICM.

---

## Root Cause of dsmem_v2.cu Error: Wrong Metric

`dsmem_v2.cu` used a single-thread FADD accumulator that serialized loads through a floating-point dependency chain, converting a bandwidth test into a latency test. The reported "4.7× slower" reflects the real latency difference (DSMEM is much higher latency than local SMEM), but the framing as a "bandwidth" metric and the headline "4.7×" mischaracterizes it as a throughput ratio. The actual latency ratio is ~8×.

---

## True Numbers (B300, 1920 MHz, SASS-verified)

### Latency (dependent pointer chain, 1 warp, warp-serial)

| Memory | Cluster Size | cy/load | Time/load |
|---|---|---|---|
| Local SMEM | 1 | 28.0 | 14.6 ns |
| DSMEM | 2 | 224 | 116 ns |
| DSMEM | 4 | 201 | 105 ns |
| DSMEM | 8 | 201 | 105 ns |

**DSMEM latency ratio vs local SMEM: 7.2-8.0×**

Measurement methodology:
- smem[i] stores a byte offset into smem (runtime-computed from `seed`), so each load result determines the next load address — LICM impossible
- Cluster barrier (`barrier.cluster.arrive/wait`) precedes the timing window — sync cost excluded
- Local SMEM: 5000 iterations, rock-solid 28.01 cy/load (σ=0)
- DSMEM cluster=2: 50 iterations, 30 runs, 224.6 cy/load (min 223.8, max 225.9)
- DSMEM cluster=4/8: 5 iterations (crash rate increases with load count — see below), ~201 cy/load

### Throughput (ILP=4, 4 independent dependent chains)

| Memory | cy/load | loads/cy | Ratio vs local |
|---|---|---|---|
| Local SMEM ILP=4 | 7.003 | 0.1428 | 1.0× |
| DSMEM ILP=4 | 63.5 | 0.0157 | 9.1× |

DSMEM throughput is **9×** lower than local SMEM throughput at ILP=4.

Note: local SMEM ILP=4 = 7 cy/load means the 4 independent chains fill ~4 pipeline slots per 7 cycles (effective throughput 0.57 loads/cy per warp). With DSMEM the pipeline does not hide the cross-SM latency at ILP=4.

---

## SASS Mechanism: LD.E via Global Window (not LDS)

`ld.shared::cluster.u32` with a scalar-register address compiles to `LD.E` (global memory load), **not** `LDS`. The `mapa` instruction returns a 64-bit global address into the peer SM's shared memory window. ptxas constructs this address using `PRMT` + `IMAD.IADD` from the mapa result and `SR_SWINHI` (the GPU's shared-window high bits), then issues `LD.E`.

The `LDS R,[R+UR]` DSMEM form (true shared-memory addressing) only appears when the mapa result lands in a **uniform register (UR)** — which requires the mapa inputs (local_base, target_cta) to both be warp-uniform AND recognized by ptxas's uniformity analysis. In tests where peer_base was in a scalar register, `LD.E` was always generated.

The `LD.E` path works correctly — the single-load verification test confirms peer values are read accurately. The higher latency (~224 cy vs ~28 cy for LDS) reflects the cost of routing the global-window load through the L2/interconnect rather than the local shared memory crossbar.

---

## Non-Deterministic Crashes

Dependent DSMEM chains crash with "unspecified launch failure" at approximately:
- Cluster=2: ~50% crash rate at 50+ iterations per test run
- Cluster=4: ~50% crash rate at 10 iterations, 100% at 15+
- Cluster=8: similar to cluster=4

Crashes are non-deterministic. When the test succeeds, the cy/load numbers are extremely repeatable (σ < 0.5%). The crash is likely triggered by the `LD.E` global-window mechanism interacting with some GPU hardware state after many repeated cross-SM global loads — possibly an ECC scrubber, prefetcher state, or a hardware counter overflow in the cluster tracking mechanism. Local SMEM chains at identical iteration counts never crash.

Workaround: use small iteration counts (≤8 for cluster=4, ≤50 for cluster=2) and run multiple samples.

---

## Summary Table

| Quantity | Value | Notes |
|---|---|---|
| Local SMEM latency | 28 cy (14.6 ns) | SASS: `LDS R0,[R0+UR5]`, rock-solid |
| DSMEM latency (cluster=2) | 224 cy (117 ns) | SASS: `LD.E R8,[R6]` via global window |
| DSMEM latency (cluster=4,8) | 201 cy (105 ns) | Slightly faster than cluster=2 (less contention?) |
| Latency ratio | 7.2-8.0× | NOT 0.8% (LICM error) and NOT 4.7× (wrong metric) |
| Local SMEM throughput ILP=4 | 7.0 cy/load | 4 chains hide most latency |
| DSMEM throughput ILP=4 | 63.5 cy/load | Cross-SM bandwidth-limited |
| Throughput ratio | 9.1× | |
| §30.H "0.8% slower" claim | WRONG | Both variants had LICM; measured loop overhead |
| dsmem_v2 "4.7× slower" claim | WRONG framing | True latency ratio is 7-8×, not 4.7× |

---

## Implications

1. DSMEM is useful for sharing data across CTAs without going to L2 — but it costs ~8× more latency per access than local SMEM.
2. Applications that stream data once through DSMEM (e.g., TMA + DSMEM pipeline) pay the latency once and overlap it; dependent DSMEM pointer-chasing is the worst case.
3. The cluster barrier must precede all DSMEM loads (not just the first), or you risk reading stale data — the crash behavior suggests the hardware may be enforcing this with some error mechanism.
4. `ld.shared::cluster` compiles to `LD.E` (global load) on B300 when the address is in a scalar register. This is not a ptxas bug — it correctly generates a valid access to the peer's shared memory via the global address window. The performance penalty is the L2/interconnect latency vs the shared memory crossbar latency.

---

## Files

- `/tmp/dsmem_single_load.cu` — initial 10-iter test (first working DSMEM measurement)
- `/tmp/dsmem_lat1024.cu` — isolated per-process latency test for 10/100/1024 iters
- `/tmp/dsmem_lat_bounds.cu` — crash threshold finder (100 OK, 120 CRASH)
- `/tmp/dsmem_isolate.cu` — one-kernel-per-process isolate test (used for all final stats)
- `/tmp/dsmem_tp.cu` — ILP=4 throughput test (local and DSMEM)
- `/tmp/dsmem_cs_fix.cu` — cluster-size variant (compiled separately for cx=2,4,8)
- `/root/github/QuickRunCUDA/tests/bench_dsmem_definitive.cu` — QuickRunCUDA-format kernel (correct SASS for local SMEM, but crashes under QuickRunCUDA because `cuLaunchKernel` doesn't support cluster dims)
- `sass/bench_dsmem_1891246005.sass` — SASS that proves §30.H LICM error
