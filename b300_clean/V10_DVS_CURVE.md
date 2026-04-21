# V10: DVS curve — full power vs clock characterization

## Measurement (FFMA workload, V6 C1 kernel, 148×256, ITERS=3000)

| Clock (MHz) | Time (ms) | Idle (W) | FFMA (W) | Delta (W) | TFLOPS | GFLOPS/W |
|-------------|-----------|----------|----------|-----------|--------|----------|
| 510         | 8701      | 144.0    | 177.8    | 33.8      | 13.7   | 77       |
| 800         | 5530      | 147.2    | 201.5    | 54.3      | 21.6   | 107      |
| 1005        | 4448      | 152.0    | 225.2    | 73.2      | 26.8   | 119      |
| 1200        | 3701      | 157.6    | 254.0    | 96.4      | 32.2   | 127      |
| **1500**    | 2964      | 167.3    | 299.9    | 132.6     | 40.2   | **134**  |
| **1700**    | 2625      | 174.9    | 339.2    | 164.3     | 45.4   | **134**  |
| 1920        | 2312      | 197.7    | 419.5    | 221.8     | 51.5   | 123      |
| 2032 (=1920)| 2314      | 198.4    | 419.0    | 220.6     | 51.5   | 123      |

## Three regimes

1. **< 1005 MHz (idle-dominated)**: efficiency 77-107 GFLOPS/W. Idle power
   doesn't scale down as much as compute, so per-op energy is high.
2. **1005–1700 MHz (sweet spot)**: efficiency 119-134 GFLOPS/W. Linear or
   sublinear power scaling matches throughput growth.
3. **> 1700 MHz (DVS superlinear)**: efficiency drops to 123 GFLOPS/W at 1920.
   V² × f scaling kicks in — extra clock costs disproportional power.

## Confirmation of CLAUDE.md note

`nvidia-smi -lgc 2032` returns **1920 MHz** actual (per CLAUDE.md). The
2032 row in this table shows IDENTICAL time (2312 vs 2314) and power
(419.5 vs 419.0) to the 1920 row — confirming the pin behavior.

## Confirmation of V8 J2

V8 J2 found 1500 MHz = FFMA energy sweet spot. This DVS curve confirms
1500 AND 1700 are tied at 134 GFLOPS/W. Either is the optimal energy
operating point.

## Power scaling analysis

**Delta-power (above idle):**
- 510 → 1005 MHz: +127% clock, +117% delta. Sublinear (good).
- 1005 → 1500: +49% clock, +81% delta. Slightly superlinear.
- 1500 → 1920: +28% clock, +67% delta. Strongly superlinear (DVS kicks in).

The DVS effect is real and well-defined: V² × f scaling above ~1500 MHz.

## Idle power scaling (separate phenomenon)

Idle even varies with clock (no work running):
- 510 MHz: 144 W
- 1920 MHz: 198 W (+37% just for being at high clock)

This is the leakage power growth from higher voltage.

## 10-rule confidence: HIGH

8 clock points, smooth monotonic curves for all metrics. Cross-checked
against V8 J2 sweet spot finding. Verified pin behavior matches CLAUDE.md.

## Practical guidance

For ANY FFMA-dominated kernel:
- **Throughput-critical**: use 1920 MHz (boost), get 51.5 TFLOPS
- **Energy-optimal**: use 1500-1700 MHz, get 40-45 TFLOPS at 134 GFLOPS/W
- **Avoid**: <1005 MHz (idle penalty), >1700 MHz (DVS penalty)

For different workloads, sweet spot may shift:
- DRAM-bound: prior V8 J2 found 1005 MHz best (since BW saturates earlier)
- Tensor cores: needs separate measurement (likely higher sweet spot)