# clock() vs clock64() vs globaltimer cost — V4 / O7

**Date: 2026-04-20.** Test `bench_clock_vs_clock64.cu`. 1500 MHz,
single warp single block, NC=8 reads chain, anti-DCE via
`acc = acc * 31 + c` (forces real chain dep on each read).

## Result (cy per inner-loop pass)

| Source | cy/op total | cy/op minus baseline |
|--------|-------------|---------------------|
| `mov.u32 %clock` (32-bit clock()) | 16.25 | **4.00** |
| `mov.u64 %clock64` (64-bit clock64()) | 14.375 | **2.125** |
| `mov.u64 %globaltimer` (ns timer) | 14.375 | **2.125** |
| NOP baseline (no clock read) | 12.25 | 0 |

## Surprise: clock() is 2× MORE expensive than clock64()

Despite returning 32-bit (less data), `clock()` costs ~4 cy per read
while `clock64()` and `globaltimer` cost only ~2 cy each — **half the
cost** despite returning 64-bit values.

## SASS evidence

Both compile to `CS2R Rdest, SR_CLOCKLO` (clock(), clock64()) or
`CS2R Rdest, SR_GLOBALTIMERLO` (globaltimer). The CS2R instruction
itself is identical — the difference must be in the post-processing:

- `clock()` returns 32-bit, so the compiler reads CS2R into a 32-bit
  register (R*); but the underlying SR is 64-bit, possibly costing
  extra cycles for the shift/mask
- `clock64()` and `globaltimer` read the full 64-bit value into
  Rlo:Rhi pair, no shift/mask needed
- 64-bit accumulator chain treats all three the same in terms of
  downstream pipe occupancy

## Important: my first measurement was wrong

Initial test reported clock64() at 6.75 cy/op (vs clock() at 45 cy/op).
That was due to **inner-loop DCE** — the compiler removed all 8 inner
clock64() reads because the chain through `acc += c` wasn't strong
enough. Verified via SASS (only 2 CS2R instead of 10).

Fixed with `acc = acc * 31ull + c` — chain dep through multiply forces
the compiler to keep every read.

## Practical recipe

For in-kernel timing:
- **Always prefer `clock64()`** over `clock()` — half the cost AND
  no 32-bit overflow worry (clock() wraps every ~2 sec at 2 GHz)
- **`globaltimer` for ns-resolution wall-clock** (same cost as
  clock64) — useful for cross-warp synchronization measurement
- **Both are essentially "free"** at ~2 cy/read — no need to
  minimize the count for performance

## Confidence

- **HIGH** for the 2.1 cy/read cost (3 trials, anti-DCE verified)
- **HIGH** for clock() costing 2× more than clock64() (consistent
  across multiple runs, SASS-identical CS2R encoding)
- **MED** for "shift/mask explanation" — could also be measurement
  noise around the 2-cy difference; needs SASS instruction-by-instruction
  cycle accounting to be fully sure

## Files

- `tests/bench_clock_vs_clock64.cu` — modes 0/1/2/3
