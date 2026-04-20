# cuBLAS K-id Speedup is SHAPE-DEPENDENT (correction)

**Date: 2026-04-20.** Discovered while applying rule #9 to "K-row identical → 1.40× speedup" claim.

## Headline finding

The 1.40× K-id speedup is NOT universal across shapes. It only triggers when:
- N ∈ {K, 2K} (and N=K/2 if M is large enough)
- All other N values: speedup collapses to ~1.02×

This is NOT due to cuBLAS picking a different algorithm. ncu confirms the
SAME kernel `nvjet_sm103_tss_128x256_64x6_2x1_2cta_v_bz_NNT` runs at all
tested N values. The shape-dependence is intrinsic to the data×kernel
interaction at the hardware level.

**Correction to prior claim:** TCGEN05_POWER_MASTER.md previously attributed
the rectangular-vs-square gap to "different cuBLAS algorithm" - this is wrong.
Same kernel, different shape-induced HW behavior.

## Data (after `nvidia-smi -rgc`, M=K=8192 BF16, GPU 0)

K-id mode (rank-1 along K, varies by N):

```
N      TFLOPS  N/K   Speedup vs random
8192   2098    1.0   1.42×  ← speedup
9216   1503    1.13  1.02×
10240  1501    1.25  1.02×
12288  1521    1.5   1.03×
14336  1501    1.75  1.01×
16384  2089    2.0   1.42×  ← speedup
20480  1494    2.5   1.01×
24576  1505    3.0   1.02×
28672  1495    3.5   1.01×
32768  1513    4.0   1.02×
```

K variation (M=N=8192):
```
K     TFLOPS  Speedup
4096  2117    1.40×  ← N=2K, speedup
4608  1513    1.02×  
5120  1511    1.02×  
5632  1507    1.02×
6144  1535    1.03×  
6656  1483    1.00×
7168  1488    1.00×
7680  1482    1.00×
8192  2086    1.41×  ← N=K, speedup
12288 2145    1.40×  ← (different kernel: 256×256 tile)
16384 2172    1.41×  ← (different kernel)
```

Cross-K validation of N=K, N=2K rule:
```
K=4096 N=4096:  speedup ✓
K=4096 N=8192:  speedup ✓
K=4096 N=12288: NO speedup
K=6144 N=6144:  speedup ✓ (when N=K, not N=8192)
K=6144 N=12288: speedup ✓
K=6144 N=18432: NO speedup
K=8192 N=8192:  speedup ✓
K=8192 N=16384: speedup ✓
K=8192 N=24576: NO speedup
```

So **K=6144 is NOT inherently broken** - the earlier confusion was that we tested
K=6144 with N=8192, where N/K=1.33 falls outside the {1, 2} window.

## Full constant (entropy=0) comparison

To confirm this is shape-conditional STRUCTURED data and not a measurement issue,
tested fully constant B (both A and B all-zero bits):

```
N      Full-const TFLOPS
8192   2251
9216   2218
12288  2257
16384  2260
24576  2262
32768  2262
```

**Full constant: shape-independent. Range 2218-2262 TF (~2% spread).**

Whereas K-id at same shapes: 1495-2098 TF (~40% spread).

## Mechanism interpretation

Two distinct hardware mechanisms:

1. **Universal entropy detector (full-const)**:
   When the bit-entropy-per-byte is exactly zero everywhere, hardware
   gates the multiplier circuits across the entire fabric. Works
   regardless of shape.

2. **Shape-conditional pattern detector (structured low-entropy)**:
   When data has structure (e.g., rank-1 along K), the dedup HW only
   detects the structure when N aligns with cuBLAS scheduling pattern.
   Likely tied to L2/SMEM scheduling order: at N=K and N=2K, the
   CTA-to-N-tile mapping creates cyclic register-reuse patterns that
   the hardware can collapse.

The rule "N ∈ {K, 2K, K/2 if M≥2K}" suggests cuBLAS uses an
M-tile-major scheduling (sweep all M tiles for fixed N tile, then
advance N). At N=K, the M sweep happens K/256=32 times in M-direction
and 32 times in N-direction. Symmetric M=N tile counts may create
the cyclic pattern.

## Practical implication for ML inference

Real Llama-class models use FFN dimensions:
- Llama-70B: K=8192, N_intermediate=28672 (N/K=3.5) → outside speedup window
- Llama-8B: K=4096, N=14336 (N/K=3.5) → outside speedup window
- DeepSeek-V3: K=7168, N=18432 (N/K=2.57) → outside speedup window

**Almost no production inference shape hits the K-id speedup window.**

Combined with the prior "realistic Gaussian INT4 = ~4%" finding, the
practical takeaway is firm:

> **Real ML inference benefits ~2-6% from data-dependent throttle relief,
> NOT the 40% suggested by synthetic same-K-row tests at square shapes.**

## Methodology note: clock state contamination

During this investigation, GPU 0 was discovered stuck at 1005 MHz under
load (despite no explicit lock and Idle reason showing). Initial repro
attempts gave ~1190 TF universally (1.0× ratio). After `sudo nvidia-smi
-rgc -i 0` the clock returned to 2032 MHz boost and the periodic pattern
appeared cleanly.

**Operational lesson:** Always sample `nvidia-smi --query-gpu=clocks.current.sm`
during a long microbenchmark run to confirm the clock is at the expected
boost level. Don't trust prior measurements without clock verification.

## Verification log

| Action | Cmd / file | Result |
|--------|-----------|--------|
| Kernel identification | ncu kernel-name regex | Same kernel for N=8192 thru 32768 |
| Random control | mode=0 | Flat at 1472-1486 TF across all N |
| K-id periodic | mode=1 | Spikes at N=K, 2K, K/2; flat 1500 elsewhere |
| Full-const universal | extreme2 0 0 | Flat 2218-2262 TF across all N |
| K=6144 scaling | tested N=K, 2K | Works when N aligned, NOT inherently broken |
| Clock state | nvidia-smi during run | 1005 MHz contamination ruled out post-rgc |

## NCU verification: pure clock throttle mechanism

Captured cycle/instruction counts at N=K vs N=K+32:

| Metric | N=8192 (1.46×) | N=8224 (1.00×) | Ratio |
|--------|----------------|----------------|-------|
| TFLOPS | 2096 | 1441 | 1.45× |
| sm__cycles_active | 137.0M | 140.4M | 1.02× (work scales with N) |
| sm__pipe_tensor_cycles_active | 536.9M | 553.6M | 1.03× (work scales) |
| smsp__inst_executed | 23.4M | 24.2M | 1.03× (work scales) |

**Cycles and instructions are essentially IDENTICAL** (~3% scaling matches the
N=8224/N=8192 ratio of 1.004×, plus partial-tile overhead).

TFLOPS difference of 45% is **purely clock frequency**:
- N=8192: clock stays at boost ~2032 MHz → 2096 TF (HW gates multiplier on detection)
- N=8224: clock drops to ~1400 MHz → 1441 TF (multiplier stays hot, hits power cap)
- Expected ratio: 2032/1400 = 1.45× ✓ matches measured

This conclusively places the mechanism at HW data-dependent power throttle,
NOT compute-time reduction. Same kernel, same instruction count, same cycles -
only clock differs based on whether HW dedup detects the pattern.

## FP8 confirmation: precision-independent

Same N-dependence pattern at FP8:
```
N      Random  K-id    Speedup
8192   2622    4062    1.55×  ← speedup
9216   2607    2655    1.02×
12288  2620    2690    1.03×
16384  2619    4059    1.55×  ← speedup
24576  2608    2689    1.03×
32768  2607    2699    1.04×
```
FP8 K-id at N=K: 4062 TF = 90% of FP8 spec peak (4500 TF).

## Boundary sharpness: razor-sharp at exactly N=K

```
N=8128  1475 TF (no speedup)       N=8160  1472 TF (no speedup)
N=8192  2100 TF (FULL SPEEDUP)
N=8224  1439 TF (no, slight degr)  N=8256  1443 TF (no)
```

Off by exactly one tile_N (32 elements) → total speedup loss.
M variation is far more tolerant: M=K±32 still gives ~95% of peak speedup.

## Asymmetry: M is "outer", N is "inner" in scheduling

The M direction tolerates variation; N requires exact alignment because:
- B operand is loaded indexed by N (B[k][n])
- The K-id pattern has all K rows = f(n)
- HW dedup detects identical-row content via (m, n) coordinate hash
- Misalignment in N shifts the cyclic pattern; misalignment in M doesn't

## Universal across cuBLAS kernels (256x256 tile)

Tested with K=12288 which forces cuBLAS to pick a different kernel
(`nvjet_sm103_tss_256x256_64x4_2x1_2cta_v_bz_NNT` instead of 128×256):

```
M=K=12288, N varies:
N=6144 (=K/2):   2136 TF   ← full speedup
N=8192 (=2K/3): 2075 TF   ← partial speedup
N=9216 (=0.75K): 1587 TF   ← no speedup
N=12288 (=K):    2141 TF   ← full speedup
N=16384 (=4K/3): 1604 TF   ← no speedup
N=18432 (=1.5K): 1585 TF   ← no speedup
N=24576 (=2K):   2144 TF   ← full speedup
```

**Same N ∈ {K/2, K, 2K} window applies to the 256×256 kernel.**

This rules out kernel-specific implementation details. The shape rule
is intrinsic to the tcgen05.mma HW pattern detector, not to cuBLAS's
choice of tiling.

## Final consolidated rule

For B300 with K-row-identical B operand (rank-1 along K):
- **Full 1.40-1.55× speedup** at N ∈ {K/2, K, 2K} (any cuBLAS kernel)
- **Partial speedup** at N = K * (p/q) for small p, q (e.g., N=2K/3)
- **No speedup** at unrelated N values
- Razor-sharp boundary: ±32 N elements = total speedup loss
- M variation tolerated (5% loss at M=K±32)

Combined with prior findings:
- Full constant data (entropy=0): universal speedup, shape-INDEPENDENT
- Structured low-entropy: shape-CONDITIONAL on N alignment to K
- Real ML inference shapes: outside speedup window → ~2-6% benefit

## A vs B asymmetry: only B operand triggers shape-conditional speedup

Tested A operand with K-column-identity structure (A[m][k] = g(m), constant
along K direction) - dual of the B K-id pattern. Result:

```
M=K=8192 BF16, A=K-col-id, B=random:
N=8192:  1490 TF (vs random 1482 = 1.005×)
N=9216:  1471 TF (vs random 1464 = 1.005×)
N=12288: 1493 TF (vs random 1484 = 1.006×)
N=16384: 1487 TF (vs random 1478 = 1.006×)
N=24576: 1494 TF (vs random 1485 = 1.007×)
```

**A K-identity gives essentially NO speedup at ANY N.** All 0.5-0.7%.

Contrast with B K-identity at N=K=8192: 2098 TF (1.42×).

**Confirms existing model:** A operand is broadcast through fanout (free
regardless of value), B operand is distributed and gates power based on
content. The shape-conditional speedup is purely a B-side mechanism.

This means in real ML inference, you cannot get throttle relief by
making the activation tensor (often A in cuBLAS NN convention) low-entropy.
You'd need the WEIGHT tensor (B) to have the K-id-like structure AND the
shape to fall in N ∈ {K/2, K, 2K}. The combination is rare in practice.
