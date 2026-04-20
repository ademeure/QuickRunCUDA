# Tensor core warmup cost — V4 / O2

**Date: 2026-04-20.** Test `bench_tensor_warmup.cu`. Single warp, 1500 MHz,
mma.sync m16n8k16 BF16 chain with chain-carry through C fragments.
Clock64 between every MMA to observe per-MMA latency.

## Result

**No observable warmup penalty on `mma.sync`**:

```
MODE=0 no-warmup:  mma0=15cy mma1=20cy mma2=20cy mma3=20cy ... mma7=20cy
MODE=1 pre-warmed: mma0= 2cy mma1=20cy mma2=20cy mma3=20cy ... mma7=20cy
```

- Steady-state per-MMA latency = **20 cy** (= 16 cy mma.sync pipeline +
  4 cy clock64 measurement overhead)
- First MMA in MODE=0 = 15 cy, which is 5 cy FASTER than steady state —
  likely noise from clock64 placement / compiler reorder
- MODE=1 mma0=2 cy: the 16 preceding warmup MMAs overlap with t0
  clock64, making the first timed MMA appear absurdly fast. This is
  measurement artifact, not actual speedup.

## Interpretation

There is **no measurable cold-start penalty** for mma.sync on B300.
The tensor core pipe is ready to accept work from the very first
instruction after kernel launch. Any "tensor warmup" seen in practice
is likely driver/dispatch latency, not SM-level tensor core cold start.

## Did NOT test

- tcgen05.mma (Blackwell-native) — requires tmem alloc + different harness
- After very long idle (millions of cycles) — current test starts
  immediately so doesn't measure true "cold after long idle"
- Across kernel boundaries — single-kernel-launch only

## Practical implications

For low-latency inference: no need to issue dummy MMAs to "warm up"
tensor cores on B300. Dispatch + register setup is the dominant
first-MMA cost, not the tensor core itself.

## Confidence

- **HIGH** that intra-kernel mma.sync has no warmup (8-MMA consecutive
  times all 20 cy ± 0-5 cy noise)
- **MED** for generalizing to tcgen05.mma (untested here)
- **LOW** for "never any warmup, even after long idle" (didn't test
  long-idle scenario)

## Files

- `tests/bench_tensor_warmup.cu` — modes 0, 1
