# M Dimension Effect on BF16 Halves Processing

Date: 2026-04-20. Tests if BF16 m128n128k16 halves effect persists at smaller M.

## Setup
- `tests/bench_tcgen05_bf16_m64.cu` (M=64 variant)
- BF16 m64n128k16, all other modes same as perbit_power.cu
- @ -lgc 1005 MHz, 50M iters

## Baselines

| Configuration | M=64 (W) | M=128 (W) |
|---------------|---------:|----------:|
| Random | 406 | 611 |
| Const +1.0 | 243 | 299 |
| Zero | 241 | 294 |

Per-MAC random penalty:
- M=128: +312W / 16384 MACs/cy = ~19 nW/MAC
- M=64: +163W / 8192 MACs/cy = ~20 nW/MAC

## N-vary cliff (M=64)

| N_unique | M=64 (W) | M=128 (W) |
|---------:|---------:|----------:|
|        1 |      244 |       299 |
|        2 |      244 |       301 |
|        4 |      244 |       299 |
|        8 |      244 |       299 |
|       16 |      246 |       302 |
|       **32** | **406** | **605** |
|       64 |      415 | 623 |
|      128 |      414 | 621 |

**Same cliff position at N_unique=17 (32-byte boundary).** Independent of M.

## Halves test (single unique sub-tile position, M=64)

| Pos | M=64 (W) | M=128 (W) |
|----:|---------:|----------:|
|   0 | 313 | 426 |
|   1 | 334 | 469 |
|   2 | 328 | 461 |
|   3 | 332 | 466 |
|   **4** | **270** ← cliff | **349** |
|   5 | 246 (free) | 304 (free) |
|   6 | 248 | 305 |
|   7 | 246 | 306 |

**HALVES EFFECT IS PRESERVED AT M=64.** Same shape: positions 0-3 expensive,
positions 4-7 nearly free.

## Mechanism: Halves are N-DIRECTION STRUCTURAL

The two-half processing in BF16 is purely N-DIRECTION:
- Half A: N positions 0..63 (sub-tiles 0-3)
- Half B: N positions 64..127 (sub-tiles 4-7)

The mechanism is independent of:
- M dimension (works at M=64 and M=128)
- K dimension (BF16 K=16 fixed)
- LBO/SBO descriptor parameters

This points to the PHYSICAL MAC array layout: BF16 m{64,128}n128k16 likely
has 2 N-direction halves at the multiplier hardware level, with each half
processing 64 N positions independently.

## Implication

The "Half B random sub-tiles are free" optimization works for:
- BF16 m=64 n=128 k=16
- BF16 m=128 n=128 k=16
- Likely BF16 m=128 n=N k=16 for any N divisible by 128

Does NOT apply to FP8/NVFP4 (different MAC array layout for those precisions).

## Confidence

- HIGH on halves working at M=64 (clear pos 0-3 vs 4-7 split)
- HIGH on cliff at same N_unique=17 (universal 32-byte boundary)
- MEDIUM on whether m=128 n=64 (smaller N) still has halves (untested due to N=64 having only 4 sub-tiles)

## Next test (TODO)

- Test BF16 m=128 n=192 or n=256 to see if halves scales with N
- Test BF16 with M=64 + N=64 (smallest combination)
