# §21 retest — tcgen05.mma "sustained-load throttle" (catalog task #87)

**Status:** ❌ CATALOG CLIFF DOES NOT REPRODUCE; I-cache hypothesis FALSIFIED; root mechanism narrative needs replacement.

**Date:** 2026-04-24. **GPU:** B300 SXM6 sm_103a (GPU 0), default boost (no `-lgc` lock — idle 120 MHz, scales to 2032 MHz under load). **CUDA:** sm_103a NVRTC inline, FP8 `tcgen05.mma kind::f8f6f4` M=128 N=128 K=32, single CTA, 1 warp.

---

## Catalog claim (B300_PIPE_CATALOG.md L7797–7825)

> The "100K iter cliff" finding: peak FP8 throughput drops 60% beyond ~30K continuous MMAs from one warp.
>
> | ITERS  | cy/MMA  | TFLOPS | % peak |
> |--------|---------|--------|--------|
> | 5K     | 128.05  | 4654   | 100% |
> | 30K    | 128.01  | 4655   | 100% (cliff edge) |
> | 50K    | 305.90  | 1949   | 42% |
> | 100K   | 394.16  | 1512   | 32% |
>
> Mechanism (catalog speculation): "dispatch bubbles ... possibly hardware running-average power tracking, tcgen05 internal queue/scheduler limits, sustained-utilization governor"

## User hypothesis tested

Catalog test likely **fully unrolled** the MMA loop. 30K MMAs × 96 byte SASS each = ~2.88 MB of code, far exceeding any reasonable L0/L1 I-cache. So the cliff is **I-cache thrashing**, not a tensor-pipe governor. With `#pragma unroll 1` the body is one MMA inst → I-cache footprint constant → cliff should disappear.

## Test files

- `tests/bench_tcgen05_icache_probe.cu` — runtime-loop kernel; `-H "#define UNROLL N"` selects N=0(=`#pragma unroll 1`) / -1(default) / N>0(`#pragma unroll N`)
- `tests/bench_tcgen05_icache_full_unroll.cu` — `STATIC_ITERS` is a compile-time constant + `#pragma unroll` (full)

## Measurements (FP8, KIND=1, M=128 N=128, single warp, default boost)

| Variant            | iters  | cy/MMA  | cubin bytes | code-bytes/MMA in SASS |
|--------------------|--------|---------|-------------|------------------------|
| `#pragma unroll 1` | 5K     | 67.049  | 28232       | n/a (loop)             |
| `#pragma unroll 1` | 30K    | 67.008  | 28232       | n/a                    |
| `#pragma unroll 1` | 50K    | 67.005  | 28232       | n/a                    |
| `#pragma unroll 1` | 100K   | 67.002  | 28232       | n/a                    |
| `#pragma unroll 8` | 30K    | 64.010  | 32200       | n/a (8×96=768 in body) |
| `#pragma unroll 8` | 50K    | 64.006  | 32200       | n/a                    |
| compiler default   | 30K    | 64.010  | 36488       | n/a                    |
| compiler default   | 50K    | 64.006  | 36488       | n/a                    |
| compiler default   | 100K   | 64.003  | 36488       | n/a                    |
| **FULL UNROLL**    | 30K    | **70.74** | **9 139 120** (9.1 MB) | **96.0 bytes/MMA** (verified) |
| FULL UNROLL†       | 50K    | 64.003  | 32488       | (NVCC silently REFUSED to fully unroll 50K and emitted a loop) |

† NVCC will not honor `#pragma unroll` for 50000 iters even with a compile-time bound — falls back to a 32 KB looped variant. The 30K full-unroll succeeded (9.1 MB cubin verified, 30 000 distinct `UTCQMMA` SASS insts, addr stride 96 B between consecutive MMAs).

## Code-bytes verification (SASS-grep)

```
$ grep -c "UTCQMMA" sass/bench_tcgen05_icache_full_unroll_*.sass    # 9 MB cubin
30000
$ first MMA  /*0910*/   UTCQMMA gdesc[UR4], gdesc[UR6], tmem[UR9], tmem[UR10], idesc[UR11], !UPT ;
$ last MMA   /*2bfab0*/ UTCQMMA gdesc[UR4], gdesc[UR6], tmem[UR9], tmem[UR10], idesc[UR11], UPT ;
$ stride = (0x2bfab0 - 0x0910) / 29999 = 96.0 bytes/MMA   ← exact
```

So FULL-UNROLL 30K = **2.88 MB of straight-line MMA SASS**, vastly exceeding any plausible I-cache (Hopper L0 / L1 I-cache is on the order of 8–32 KB per SMSP). If the I-cache thrash hypothesis were right, full-unroll-30K should be massively slower than unroll=1. **It is only 5.6% slower (70.74 vs 67.0 cy/MMA).**

## Verdict

1. **I-cache hypothesis FALSIFIED.** Even at 90× I-cache footprint (2.88 MB straight-line code), throughput drops only 5.6%, not 2.4× (catalog) or anything resembling a cliff.
2. **The catalog cliff itself does not reproduce on this rig.** Across all three coding styles (unroll=1, unroll=8, compiler-default) and at all iter counts 5K → 100K, FP8 cy/MMA stays in the **64–67** range — flat. The catalog's claimed jump from 128 cy → 305 cy at 30K → 50K is not observed.
3. The **catalog is internally inconsistent**: line 7084 ("Streaming throughput: 67 cy/MMA") matches my measurement; lines 7805–7811 (the "cliff table" at 128 cy baseline + 305 cy at 50K) does not match anything I measure with the documented setup. Most likely the cliff data was collected with a different anti-DCE / sync pattern (e.g. one with mbarrier inside the loop, or a different MMA shape variant than M=128 N=128) and the section was conflated with the 67 cy "true" streaming number.

## Recommended replacement narrative

Catalog L7797–7827 should be **replaced** with:

> **tcgen05.mma sustained throughput (1 warp, 1 CTA, FP8 M=128 N=128 K=32, default boost):**
> 64–67 cy/MMA across 5K → 100K iters, regardless of unroll style.
> No iteration-count cliff observed. The previously reported 128 → 305 cy/MMA "cliff at 30K" failed to reproduce on a clean rig (no leftover procs, default clocks). Probable cause of the original anomaly: residual PROCS / clock thrash at measurement time, or a different MMA shape than documented. Coding style is irrelevant: full-unroll 30K (2.88 MB of straight-line SASS, 90× I-cache size) measures only 5.6% slower than `#pragma unroll 1` — the SM's tensor pipe is **not bottlenecked by I-cache** in this regime.

## What this implies for the catalog

- §21 (task #87) cliff table at L7805–7811 should be marked retracted or moved to a "did not reproduce" appendix.
- The "sustained-utilization governor / dispatch bubble / power tracking" speculation at L7820–7823 should be deleted.
- Catalog L7084 ("Streaming throughput: 67 cy/MMA") is the correct number and stays.
- Confirms catalog's own L7827 practical-implication note ("real GEMM kernels naturally avoid this throttle") — though now the reason is "there is no throttle to avoid", not "load/store work hides it".

## Methodology notes

- All runs preceded by `pkill -9 QuickRunCUDA; sleep 3-5; nvidia-smi -rgc`.
- GPU idle confirmed 0% / 120 MHz before each batch.
- Anti-DCE: `tcgen05.commit.cta_group::1.mbarrier::arrive::one` + `mbarrier.try_wait` after the loop, plus unconditional store of `t1-t0` to `C[0]`.
- Predicate `P` derived from `scaleC` (which flips 0→1 on first iter) ensures every MMA actually executes (no compile-time-dead path).
- Single CTA, 32 threads, `__launch_bounds__(32,1)` — matches catalog's "from one warp" framing.
- Clock state on completion: re-locked with `nvidia-smi -rgc` (returns to default boost).
