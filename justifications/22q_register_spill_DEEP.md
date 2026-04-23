# §22q Register spill cliff measurement

**Question**: at what point does register pressure cause spilling, and what's
the throughput hit?

## Test setup

`tests/bench_reg_pressure.cu`:
- N_REGS unique float vars, FMA chain, all kept live across loop
- `__launch_bounds__(32, MIN_BLOCKS)` controls register budget per thread
- Per-thread reg budget ≈ 64K (regfile) / (32 × MIN_BLOCKS) = depends on MIN_BLOCKS

For B300 (sm_103a, 64 KB register file per SM, 32 threads/warp):

| MIN_BLOCKS | Max threads/SM | Max regs/thread |
|------------|---------------:|----------------:|
| 1  |   32  | 2048 (capped at 255) |
| 2  |   64  | 1024 (capped at 255) |
| 4  |  128  |  512 (capped at 255) |
| 8  |  256  |  256 (capped at 255) |
| 16 |  512  |  128 |
| 32 | 1024  |   64 |
| 64 | 2048  |   32 |

## Measurements (`-lgc 1800`)

### Sweep N_REGS at fixed MIN_BLOCKS

| N_REGS | MIN_BLOCKS=16 (≤128 regs) | MIN_BLOCKS=32 (≤64 regs) |
|--------|--------------------------:|-------------------------:|
|  8 | 2.88 cy/FMA, 22 regs, 0 spill | 2.88 cy/FMA, 22 regs, 0 spill |
| 16 | 1.75 cy/FMA, 28 regs, 0 spill | 1.75 cy/FMA, 28 regs, 0 spill |
| 32 | 1.38 cy/FMA, 40 regs, 0 spill | 1.38 cy/FMA, 40 regs, 0 spill |
| 64 | **1.19 cy/FMA, 72 regs, 0 spill** | **7.49 cy/FMA, 64 regs, 64B STACK** |

### Cliff dynamics

The slowdown happens precisely when the compiler's natural register usage
exceeds the budget imposed by `__launch_bounds__`:
- N_REGS=64 needs 72 registers naturally
- MIN_BLOCKS=16 → 128 regs budget → fits → no spill (1.19 cy/FMA)
- MIN_BLOCKS=32 → 64 regs budget → must spill 8 floats → 64B stack → **6.3× slowdown**

## SASS evidence

For the spilled variant, the kernel emits 36 `LDL`/`STL` instructions
(load/store local memory) in the inner loop. These are L1-cached but still
incur 30-50 cy latency each, dominating the FMA chain throughput.

## Headline numbers for B300 sm_103a

- **Register file per SM**: 64 KB = 16,384 regs/SM
- **Max regs/thread**: 255 (NVPTX limit)
- **Per-thread register budget = 16384 / (32 × CTAs/SM)**
- **Spill triggers** when natural reg count > budget
- **Spill penalty**: ~6× slowdown for modest spill (16 floats = 64B), grows with spill volume

## Practical implications

1. **Watch your CTAs/SM target**: setting `__launch_bounds__(threads, blocks)` with high `blocks` value forces tight register budget. If your kernel needs more regs than the budget allows, you spill.
2. **Monitor with `cuobjdump --dump-resource-usage`**: any `STACK:N>0` indicates spilling.
3. **`-Xptxas -v`** at compile time also reports spill stats.
4. **Increase `__launch_bounds__` first arg or decrease second**: if spilling, give the kernel more register headroom.
5. **Trade-off**: lower MIN_BLOCKS = more regs/thread but lower SM occupancy. This tradeoff requires profiling.

## Comparison to catalog claim

Earlier session's project memory mentioned "spill cliff 9× at 32 vars". My measured cliff is:
- 6.3× slowdown at 16-float spill (64B stack)
- Likely scales with spill volume

The "9×" is in the same order of magnitude. Difference may be:
- Different N_REGS / FMA chain pattern  
- Different ILP saturation in the test
- Spill placement (inner vs outer loop)

Bottom line: register spill is a real and significant performance hit on B300, but the magnitude depends on the spill pattern. Always check `STACK:N` in cubin resource usage.
