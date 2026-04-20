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

## Transposition asymmetry: confirms memory-access mechanism

Tested all 4 trans_A × trans_B combinations at M=N=K=8192 (cublas_trans):

```
trans_A trans_B mode  TFLOPS  Speedup
0       0       rand  1493
0       0       Kid   2111    1.41×  ← FULL SPEEDUP
1       0       rand  1495
1       0       Kid   2100    1.40×  ← FULL SPEEDUP
0       1       rand  1502
0       1       Kid   1567    1.04×  ← NO SPEEDUP
1       1       rand  1501
1       1       Kid   1565    1.04×  ← NO SPEEDUP
```

**Transposing B (transB=1) completely kills K-id speedup** even at N=K=8192!
trans_A flag has no effect. The mechanism cares about how B is read from memory.

### Why transposition matters

The fill_krow_id pattern fills `data[k*N + n] = f(n)` (treating data as
row-major K×N). cuBLAS interprets this as:

- **transB=0**: B[k,n] = data[k + n*K]. With K=N, reduces to f(k) for each n.
  Reading B[k,n] for fixed n, varying k: addresses k=0,1,2,..., stride 1.
  All loads return SAME value → HW dedup detects → multiplier gated → boost stays.

- **transB=1**: B[n,k] = data[n + k*N]. With K=N, reduces to f(n) for each k.
  Reading B[n,k] for fixed n, varying k: addresses n, n+N, n+2N, ..., stride N.
  Loads at stride-N return DIFFERENT memory locations (different f values for
  same logical "column" because gather-style access). HW dedup doesn't activate.

### Mechanism: memory-access pattern, not logical data pattern

The HW dedup cache tracks "is current sub-tile bit-identical to recent loads?"
This requires:
1. Memory access pattern that returns same value on consecutive loads
2. Such patterns have specific shape requirements (N ∈ {K/2, K, 2K} maps the
   dedup cache window to either 1 or 2 unique sub-patterns per N-tile)

The N=K rule + transB asymmetry together prove this is a **memory-access-level
power gating mechanism**, not a higher-level pattern detector.

## Practical strict bounds

For a real ML workload to hit this speedup window:
- Weight tensor (B in cuBLAS NN) must be column-constant in memory
- Memory layout must be K-stride-1 (transB=0)
- Shape must be N ∈ {K/2, K, 2K}

The COMBINATION is essentially never satisfied in practice. Confirms that
practical inference benefit is ~2-6%, NOT 40%.

## HW dedup cache depth: 2 slots, but shape-modulated

Tested data with K-period structure (B has 'period' distinct row patterns
repeating along K axis). At M=K=8192:

```
N=K=8192:
  period=1 (K-id):              2104 TF  ← 1.42× FULL SPEEDUP
  period=2 (alternating AB):    2101 TF  ← 1.42× FULL SPEEDUP
  period=3 (cyclic ABC):        1516 TF  ← NO speedup
  period=4 (cyclic ABCD):       1523 TF  ← NO speedup
  period=8 onwards:             ~1480 TF (random baseline)

N=2K=16384:
  period=1:  2082 TF  ← FULL SPEEDUP
  period=2:  1527 TF  ← NO SPEEDUP (cache exhausted by 2 concurrent N-tiles)
  period=3:  1518 TF  ← NO

N=K/2=4096:
  period=1:  2073 TF  ← FULL SPEEDUP
  period=2:  2071 TF  ← FULL SPEEDUP (less concurrent pressure)
```

### Key insights

1. **Dedup cache holds 2 unique sub-patterns max** at single N-tile
   (period=1 and period=2 both work at N=K).

2. **Cache pressure is shape-modulated**: at N=2K, two concurrent N-tile
   workloads compete for the cache slots. Period=1 (single pattern)
   fits in 1 slot leaving room; period=2 (two patterns per N-tile) ×
   2 concurrent N-tiles = 4 slots needed → exhausted.

3. **Period ≥ 3 NEVER works**, regardless of shape.

### Practical implication

For real ML data to trigger throttle relief:
- Weight tensor MUST have ≤ 2 distinct K-row patterns
- AND shape must be in N ∈ {K/2, K} (not even 2K for non-trivial patterns)
- AND transB=0 layout

Real LLM weight matrices have full bit-entropy → period would need to be
the full K (no repetition) = effectively period→∞ → no speedup.

### Implied HW structure

The B-side power-gating circuit appears to maintain a tiny content-addressable
cache (2 entries per byte position?). When current load matches a cached
sub-tile, multiplier circuits stay gated and clock stays at boost. When the
cache misses (3rd unique pattern arrives), all circuits energize and power
draw rises, triggering throttle.

This is consistent with the prior 32-byte sub-tile dedup model, refined:
- Sub-tile granularity: 32 bytes (16 BF16 / 32 FP8 / 64 NVFP4)
- Cache depth per sub-tile: 2 entries
- Replacement: likely LRU
- Activation: persistent within K-row pass; resets between batches

## Universal across K: rule is purely about N/K ratio

Tested K=8704, 10240, 12288 with periods 1/2/3 at N=K, 2K, K/2:

```
Across all 3 K values:
N=K:    period=1 ✓  period=2 ✓  period=3 ✗
N=2K:   period=1 ✓  period=2 ✗  period=3 ✗
N=K/2:  period=1 ✓  period=2 ✓  period=3 ✗
```

**Rule is universal in K** — the cache mechanism is shape-independent at
the K level. Only the N/K ratio matters.

## Final unified mechanism model

```
HW dedup cache architecture (per tcgen05.mma multiplier):
- Per sub-tile position (32 bytes)
- 2 LRU slots (cache depth)
- Likely shared across 2-CTA cluster
- Activates: persistent within K-row sweep
- Resets: between batch groups

Effective slots = 2 / (number of concurrent N-tiles per cluster)
- N ≤ K: 1 N-tile concurrent → 2 slots → period ≤ 2 works
- N = 2K: 2 N-tiles concurrent → 1 slot per N-tile → only period=1 works
- N > 2K: 3+ N-tiles → cache thrash → no speedup
```

This explains ALL observed behavior:
- N=K, K/2: period 1 and 2 work (2 slots available)
- N=2K: only period 1 works (1 slot per concurrent N-tile)
- N=3K, 4K+: nothing works (slots split too thin)
- A K-id no benefit (A side has no dedup mechanism)
- transB=1 no benefit (memory access pattern doesn't match cache architecture)

This is now a HIGH-confidence model with multiple independent verification paths.

## Cache replacement policy: NOT simple LRU

Tested K-row arrangement variants with same 2-pattern set (A, B) at M=K=N=8192:

```
Arrangement          TFLOPS    Speedup   Description
ABAB (chunk=1)       2102 TF   1.42×     ✓ FULL  - immediate alternation
AABB (chunk=2)       1528 TF   1.03×     ✗ NONE  - paired adjacency
AAAABBBB (chunk=4)   1722 TF   1.16×     PARTIAL
8xA-8xB              2019 TF   1.36×     NEAR FULL
32xA-32xB            2077 TF   1.40×     ✓ FULL
64xA-64xB            2082 TF   1.40×     ✓ FULL  - matches K-tile size

ABC (period 3)       1521 TF   1.03×     ✗ NONE
ABCD (period 4)      1521 TF   1.03×     ✗ NONE
AABBCC (chunk=2 of 3 pat) 1517 TF 1.02×  ✗ NONE
```

### Surprising: chunk=2 fails but chunk=1 succeeds

A simple 2-slot LRU should keep both A and B in cache regardless of arrangement.
Yet chunk=2 (AABB) gives essentially no speedup while chunk=1 (ABAB) is full.

**This rules out simple LRU.** Possible mechanisms (could not isolate definitively):

1. **Pattern predictor**: HW detects "alternating period-1" as a special case
   (matches a hardwired pattern recognizer). Other arrangements miss the predictor.

2. **Per-K-tile constancy**: When chunk size ≥ K-tile size (64), each K-tile sees
   one constant pattern → trivial dedup. Below 64, only chunk=1 matches the
   alternation predictor.

3. **Differential gating with timing**: The 2-pattern alternation may match the
   pipeline depth (6 stages) in a way that allows speculative dedup.

The chunk-size-vs-speedup curve is non-monotonic:
chunk: 1 → 2 → 4 → 8 → 32 → 64
ratio: ✓ → ✗ → mid → near-full → ✓ → ✓

This is **strong evidence the dedup HW has multiple detection paths** (period-1
alternation predictor, plus per-K-tile constancy detector). The transitions
between paths create the dip at chunk=2,4.

### Practical implications

For real workloads to trigger speedup, the K-direction structure of weights must:
- Be EXACTLY constant per K-tile (chunk ≥ 64), OR
- Alternate immediately every K-row (pattern AB AB AB...)

Real ML weights satisfy NEITHER → real inference benefit remains ~2-6%.

For SYNTHETIC compression schemes (e.g., quantized weights with shared scales
per K-block of 64), the constant-per-block structure could partially trigger
the dedup. This might be a path for hardware-aware quantization design.

## Detailed chunk-size curve (M=K=N=8192 BF16)

```
chunk    TFLOPS   Speedup
1        2102     1.42×  ✓ FULL (alternation predictor)
2        1525     1.03×  ✗ MIN (worst case)
3        1675     1.13×
4        1728     1.16×
5        1785     1.21×
6        1830     1.24×
7        1844     1.25×
8        2019     1.36×  ← jump (8 divides K-tile)
12       1907     1.29×  ← drop
16       2051     1.39×  ✓ (16 divides 64)
24       1970     1.33×
32       2065     1.40×  ✓ (32 divides 64)
48       1969     1.33×
64       2079     1.40×  ✓ (= K-tile size)
96       1977     1.34×
128      1919     1.30×  ← decline above K-tile
```

### Two mechanisms emerge

**Path 1 — Alternation predictor**: triggered by chunk=1 (immediate ABAB).
A specific HW pattern that recognizes period-1 alternation.

**Path 2 — K-tile constancy**: triggered when chunk size divides K-tile (64).
Powers-of-2 (8, 16, 32, 64) align with K-tile, giving high speedup.
Non-divisors (3, 5, 6, 7, 12, 24, 48) give intermediate speedup.

**Curiosity: chunk > 64 declines.** At chunk=128, two K-tiles per chunk but
between-tile transitions don't get the bonus. At chunk=128, we get 1.30×, less
than chunk=64 (1.40×). Hypothesis: dedup state resets between K-tiles,
making "fresh tile = fresh detection" more efficient than "tile in middle of run".

### Confidence on mechanism breakdown

- HIGH on existence of two paths (chunk=1 outlier + power-of-2 advantage)
- MED on K-tile size (64) hypothesis (matches kernel KSTAGES=64)
- LOW on speculation about why chunk>64 declines

## N-window is the OUTER gate

Tested chunk sizes at N OUTSIDE the speedup window (M=K=8192):

```
N=9216 (out of window):
  chunk=1, 8, 16, 32, 64: ALL 1480-1496 TF (no speedup)
N=24576 (=3K, out of window):
  chunk=1, 8, 16, 32, 64: ALL 1489-1529 TF (no speedup)
```

**No data structure triggers speedup outside the N window.** Confirms:

1. **Outer gate**: N ∈ {K/2, K, 2K} required (memory access pattern)
2. **Inner condition** (only matters when outer satisfied):
   - chunk=1 (alternation), OR
   - chunk divides K-tile size (64), OR
   - period=1 (K-id constant)

Both conditions must hold for speedup. This is now a HIGH-confidence
two-stage gate model.

## Master summary

| Layer | Condition | Mechanism |
|-------|-----------|-----------|
| Memory access | N ∈ {K/2, K, 2K} ∧ transB=0 | Cache lookup pattern matches |
| Data structure | period 1 OR chunk\|64 | Dedup HW activates |
| Result | clock stays at boost | TFLOPS = ceiling |

For real ML inference (Llama N/K=3.5, DeepSeek 2.57): outer gate FAILS.
Speedup not accessible regardless of weight quantization scheme.
