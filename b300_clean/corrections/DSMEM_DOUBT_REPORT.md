# DSMEM Doubt Report — adversarial review of DSMEM_REFERENCE / corrections

**Reviewer**: wave-2 doubt agent | **Date**: 2026-04-22 | **Source**: V8/V17/V19/V21 standalones

## V8 retraction is correct (HIGH confidence)

`v8_dsmem_bw.cu` lines 44-65: address `base_u = peer_base + ((tid*4) & MASK)` is invariant
in the loop; offsets are 8 compile-time constants `[base+0..+224]`; `#pragma unroll 1` on
ITERS=50000 with no inter-iter dep on the `r0..r7` results (XOR-collapsed at end). The
*addresses don't change between iterations* and outputs feed only one final write. ptxas
will CSE the 8 LDS to ~8 unique loads or hoist them entirely. Author's V10 retraction
agrees. **The 37 TB/s "97% of peak" was loop overhead.**

## "40 GB/s aggregate read" — plausible from first principles (HIGH)

V21 dependent-chain (`c[k] = ld(peer_base + c[k])`, address chained through result) is
DCE-immune. cy/load=1.08 at ILP=16 → ~1.85 GB/s/thread × 32 threads × 8 CTAs ≈ 470 GB/s
naive, but they report only ~40 GB/s aggregate. Per-CTA measured 5 GB/s × 8 = 40 GB/s. The
discrepancy is real: serialized chain caps at one outstanding load per dep, even with ILP=16
splitting into 16 parallel chains each at ~7 GB/s → 7×8=56 GB/s. Measured 40 GB/s is in the
ballpark. **Plausible, but** the test only stresses pointer-chase. A non-dep ILP test
(addresses derived from `i` not from prior result) might reach 60-80 GB/s. **Mild caveat:
"40 GB/s" is the chain-bound ceiling, not the absolute asymptote.**

## "NO shared bus" — overstated (MEDIUM)

V17 `n_senders<N>` ring uses **1 thread per CTA** with single-issue chained loads.
Per-CTA throughput is ~0.16 loads/ns. 8 CTAs × 0.16 = 1.3 Gload/s aggregate — far below
any plausible bus saturation. Of course you see 1.00× contention; you're 30× under-issued.
**This test cannot rule out a shared bus.** A real bus-contention test would need 8 CTAs
× 4 warps × ILP=16 ring (each ~5 GB/s = 40 GB/s aggregate) and check whether scaling to 8
clusters of 8 still gives per-cluster 40 GB/s. Likely *is* point-to-point per the
architecture, but V17 doesn't prove it.

## "Hot-spot writes UNCAPPED" — wrong framing (MEDIUM-HIGH)

V21 `push_ring_wr` has NO fence between stores and the `clock64` end. The 560 GB/s write
"ceiling" is **issue rate**, not completion. PTX `st.shared::cluster` is fire-and-forget;
the timer ends as soon as the last store enters the queue. Real delivery rate is unbounded
in this measurement. The hot-spot writes claim "no slowdown" is consistent with this — you
can issue indefinitely without the fabric pushing back. The pair-uniform 34 cy "fenced
write latency" (table §2) is more trustworthy because it does include a fence.

## Latency 7.5× ratio — solid (HIGH)

24 cy local vs ~180 cy DSMEM, both via dep-chain. Self-consistent across V12/V15/V16.

## Confidence summary

| Claim | Confidence | Notes |
|---|---|---|
| V8/V10 TB/s DCE'd | **HIGH** | SASS pattern unambiguous |
| ~40 GB/s read aggregate | MED-HIGH | Real but chain-bound |
| ~560 GB/s write | **LOW-MED** | Issue rate, not completion |
| NO shared bus | LOW | Test under-issued by 30× |
| 7.5× latency ratio | HIGH | Cross-test consistent |
| Pair asymmetry 25% | HIGH | V15 8×8 matrix |
| TMA multicast 470 GB/s | HIGH | Cross-pattern |

## Settling test

Build a **fenced ring write** + **non-chained ILP read** kernel that (a) fences after every
N stores and re-times, and (b) uses addresses derived from `i` not loop-carried. If write
BW drops to ~40-100 GB/s under fenced completion, current 560 GB/s is wrong. If non-chained
read hits >>40 GB/s, the read ceiling is chain-imposed not fabric-imposed. Then sweep N=2..8
clusters with full-issue load to falsify "no shared bus".
