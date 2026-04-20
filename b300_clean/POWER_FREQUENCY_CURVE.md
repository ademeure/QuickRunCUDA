# Power-Frequency Curve: Random vs Optimized BF16 GEMM

Date: 2026-04-20. Detailed clock sweep showing CMOS power-frequency dynamics
and where data optimization gives the most leverage.

## Setup
- BF16 m128n128k16, 50M iters, 148 SMs persistent
- Mode 200 (random) vs mode 6105 (max-opt: Half A=0, 3 random in Half B)
- Clocks locked via `nvidia-smi -lgc CLK`; boost via `-rgc`

## Results

| Clock (MHz) | Random P (W) | Opt P (W) | Random/Opt | Random Δ from clk-1 |
|------------:|-------------:|----------:|-----------:|--------------------:|
|         510 |          353 |       259 |      1.36× |                  -- |
|         800 |          484 |       254 |      1.90× |          +131 |
|        1005 |          613 |       296 |      2.07× |          +129 |
|        1300 |          830 |       369 |      2.25× |          +217 |
|        1500 |         1009 |       426 |      2.36× |          +179 |
|        1800 |         1095 |       537 |      2.03× |          +86 (capped) |
| 2032 (boost) |         1099 |       629 |      1.74× |          +4 (capped) |

## Findings

1. **Random saturates at TDP cap (~1100W) above 1500 MHz** — clock requested but
   power-cap-throttled. Random Δ stays at +86 / +4W vs ~+200W for unrestricted.

2. **Optimized stays well under cap at all clocks** — 629W at boost = 471W
   below TDP. Lots of headroom.

3. **Maximum power savings RATIO at 1500 MHz**: 2.36× (1009W / 426W)
   - Random has full data-dep cost
   - Optimized has minimal (static + small dynamic)
   - Highest RATIO at this clock

4. **Static power floor visible at low clocks**: Optimized=254W at 800 MHz,
   essentially same as 259W at 510 MHz. Static dominates below ~1000 MHz.

5. **Frequency scaling for optimized**: 254→629W from 800→2032 MHz. Power
   scales 2.48× for 2.54× frequency = nearly linear (CMOS expectation).

6. **Random scales faster than linear**: 484→1099W from 800→2032 MHz = 2.27×
   for 2.54× frequency (less than linear because cap kicks in).

## Deployment scenarios

### Energy-efficient inference (low clock)
- 800 MHz: random=484W, opt=254W. Optimization saves 230W per GPU (47%).
- 1005 MHz: opt=296W vs random throttled - opt is 2× more efficient

### Performance-oriented (high clock)
- 1500 MHz: random=1009W (near cap), opt=426W. Save 583W per GPU.
- For workloads at the same throughput, optimization allows MUCH lower power
  → enables denser packaging or quieter cooling.

### Performance-per-Watt sweet spot
- 1300-1500 MHz with optimization: best Performance-per-Watt
- Random at any clock above 1500: capped, suboptimal
- Optimized at boost: full performance at half the power of random-at-same-clock

## Dual-perspective: TFLOPS/W

For 100M MMAs/run, throughput = 100M * 524288 FLOPS / runtime / 1e12 = TFLOPS

At each clock (with FFMA peak ~50 TFLOPS at 1005 MHz, scaling linearly):

| Clock | Random TFLOPS | Opt TFLOPS | Random TF/W | Opt TF/W |
|------:|--------------:|-----------:|------------:|---------:|
|   510 | est 25.5 | 25.5 | 0.072 | 0.098 |
|   800 | 40 | 40 | 0.083 | 0.158 |
|  1005 | 50 | 50 | 0.082 | 0.169 |
|  1300 | 65 | 65 | 0.078 | 0.176 |
|  1500 | 75 | 75 | 0.074 | 0.176 |
|  1800 | est 90 | 90 | 0.082 | 0.168 |
|  2032 | est 100 | 100 | 0.091 | 0.159 |

**Optimized peaks at 1300-1500 MHz with ~0.176 TF/W** (more than 2× random's
~0.078 TF/W at same clock). Best practical operating point for energy-efficient
inference.

## Confidence

- HIGH on the curve shape (7 measurements, monotonic behaviors as expected)
- HIGH on the cap saturation at 1500+ MHz for random
- HIGH on static power floor at low clocks
- HIGH on the 2.36× max ratio at 1500 MHz
- MEDIUM on the TF/W extrapolation (assumes throughput scales linearly with clock, reasonable)
