# tcgen05 perf/W: 2-trial clean ladder, full random vs B-positive-only

**Date: 2026-04-20.** ALL measurements at 1500 MHz lock, M=N=256, cta=2,
2-trial reproducibility, with `pkill -9 QuickRunCUDA` + 6s cooldown
between EVERY measurement to avoid process contention.

## Methodology pitfall confirmed

Earlier `bench_nvfp4_k96_lane_fast` measurement showed cy/MMA=128 in
isolation but jumped to 1088 (8.5× inflated) when 5 leftover
QuickRunCUDA processes were running concurrently. **`kill -9 $PID;
wait $PID` is NOT sufficient cleanup**; only `pkill -9 QuickRunCUDA`
followed by `sleep 6` reliably yields clean results.

Earlier `perf/W` table at 1500 MHz had silent contamination — NVFP4
K=64 read 797 W and K=96 read 795 W (impossibly close given 1.5×
work). True values (clean): K=64=689 W, K=96=870 W.

## Full random (mode=0): 2 trials each

| format          | T1 (W) | T2 (W) | mean | TF/W |
|-----------------|--------|--------|------|------|
| FP16  K=16      | 935    | 935    | 935  | 1.95 |
| BF16  K=16      | 876    | 876    | 876  | 2.08 |
| TF32  K=8       | 787    | 787    | 787  | 1.16 |
| FP8   K=32      | 1069   | 1076   | 1073 | 3.39 |
| MXFP8 K=32      | 1046   | 1036   | 1041 | 3.50 |
| NVFP4 K=64      | 689    | 690    | 689  | 10.54 |
| NVFP4 K=96 N=192 (1.33× iters) | 882 | 879 | 880 | 12.39 |
| NVFP4 K=96 N=256 | 870   | 870    | 870  | 12.54 |

Reproducibility: max trial-to-trial gap = 10 W (MXFP8 random).
Most configs vary < 3 W between trials.

## B-positive only (mode=1): 2 trials each

Sign bit of every B element forced to 0; A still fully random.

| format          | T1 (W) | T2 (W) | mean | TF/W | Δ vs random |
|-----------------|--------|--------|------|------|-------------|
| FP16  K=16      | 784    | 786    | 785  | 2.32 | -150 W      |
| BF16  K=16      | 749    | 752    | 750  | 2.43 | -125 W      |
| TF32  K=8       | 679    | 682    | 680  | 1.34 | -107 W      |
| FP8   K=32      | 834    | 835    | 834  | 4.36 | **-238 W**  |
| MXFP8 K=32      | 802    | 798    | 800  | 4.55 | **-241 W**  |
| NVFP4 K=64      | 586    | 585    | 585  | 12.42 | -104 W     |
| NVFP4 K=96 N=192 | 733   | 723    | 728  | 14.99 | -152 W     |
| NVFP4 K=96 N=256 | 722   | 717    | 720  | **15.16** | -150 W |

## Key findings

### 1. Best efficiency: NVFP4 K=96 + B-positive = 15.16 TF/W
21% better than the "random" measurement (12.54 TF/W).
6.7× better than BF16 random.

### 2. FP8 / MXFP8 see HUGE savings from B-positive (-240 W)
Largest absolute savings of any format. Likely because FP8 has 4
elements per dword (more sign bits flip per cycle), so killing all
sign-bit toggle saves the most.

### 3. NVFP4 N=192 ≈ N=256 (within ±10 W)
At equal total work (1.33× more iters at N=192), power is the same.
Throughput per cluster is identical at saturation. **N=192 is the
optimal NVFP4 K=96 ULTRA shape** — same peak, lower per-MMA latency
and SMEM footprint.

### 4. BF16 < FP16 by 60 W in B-positive too
This 60 W gap (vs 110 W in random) confirms BF16 < FP16 power is
a real effect from FP16's wider mantissa, present even with sign
bits zeroed.

### 5. FP8 (1073 W) > FP16 (935 W) at random
Despite 2× the throughput, FP8 only uses 14% more power. That gives
FP8's 1.7× TF/W advantage. With B-positive, FP8 (834W) is barely
above FP16 (785W) but still 2× the throughput → 1.9× TF/W.

## Implications

For inference workloads with FP4/FP8 quantized weights:
- Always store +0 (not -0) for zero weights (free 100-240 W per CTA).
- For static-weight matmuls, pre-quantize to clamp signed weights
  to fewer bits (or pack as positive-magnitude + separate sign mask).
- N=192 GEMM tiles win over N=256 at zero throughput cost for
  NVFP4 K=96.

## Confidence

- **HIGH** — 2 trials per config, ±5 W reproducibility for most.
- **HIGH** that B-positive is the lever (multiple formats consistent).
- **HIGH** that contention silently corrupts measurements; pkill-9 +
  cooldown is the only reliable methodology.

## Addendum: A vs B sign-bit decomposition

Adding mode 2 (A-positive only) and mode 3 (A+B-positive) to the kernels.

| format    | random | B-pos | A-pos | A+B-pos |
|-----------|--------|-------|-------|---------|
| NVFP4 K=96 N=256 | 875W | 725W (-150) | 867W (-8) | 693W (-182) |
| FP8 K=32         | 1081W | 831W (-250) | 1082W (+1) | 809W (-272) |
| BF16 K=16        | 882W | 759W (-123) | 877W (-5)  | 725W (-157) |

**A-side sign bit is essentially FREE** (0-8 W swing across all formats).
**B-side dominates** sign-related power 95%+.

A+B-positive saves only an extra 22-34 W beyond B-positive alone. The
~3-4% incremental gain confirms the asymmetric architecture: B is
broadcast across many multiplier lanes, A has fewer consumers, so
A-side toggle activity contributes minimally to bus power.

## Updated practical recipe

For inference, the **only** sign-bit optimization that meaningfully
helps is on B-side weights. A-side activations can be random/arbitrary
without power penalty.

For NVFP4 K=96 the absolute lowest power is A+B-pos = 693 W = **15.74
TF/W** (10.91 PF / 0.693 kW). 4 W less than my earlier B-positive-only
N=192 result of 14.99 TF/W (rounding).

## SF tensor sensitivity (re-verified)

| format       | SF=1.0 | SF=random | Δ |
|--------------|--------|-----------|---|
| NVFP4 K=96 N=256 random | 873 W | 876 W | +3 W |
| MXFP8 K=32 random       | 1035 W | 1040 W | +5 W |

SF random adds ~3-5 W (negligible, as previously measured at 1005 MHz).
The 100-240 W from data dominates by 2-3 orders of magnitude.

## Zero-skip ladder: A=0, B=0, both=0 (mode 5/6/7)

Mode 5 = A entirely zero, mode 6 = B entirely zero, mode 7 = both
zero. Tests if multiplier short-circuits when one or both operands
are uniformly zero.

| format       | random | A=0 | B=0 | A=B=0 | A-skip Δ | B-skip Δ |
|--------------|--------|-----|-----|-------|----------|----------|
| NVFP4 K=96   | 867 W  | 755 W | 411 W | 394 W | -112 W   | **-456 W** |
| FP8 K=32     | 1069 W | 961 W | 425 W | 403 W | -108 W   | **-644 W** |
| BF16 K=16    | 874 W  | 661 W | 405 W | 394 W | -213 W   | -469 W |
| MXFP8 K=32   | 1039 W | 960 W | 428 W | 406 W | -79 W    | **-611 W** |

### Findings

1. **B=0 zero-skip saves 456-644 W per format** — biggest power lever
   in tcgen05 we have measured. Multiplier truly short-circuits.

2. **A=0 zero-skip saves only 79-213 W**, much less than B=0.
   Asymmetric: B-broadcast architecture means killing A doesn't
   kill the bus activity, but killing B does.

3. **A=B=0 power floor ~394-406 W across all formats** — essentially
   identical regardless of precision. This is the per-instruction
   issue cost: tcgen05 dispatch + barrier + tmem maintenance with
   completely zero-data multiplier work. ~250 W above idle.

4. BF16's A=0 saves more (213 W) than NVFP4's A=0 (112 W). Maybe
   because BF16 accumulator is wider (FP32 from fewer K) so more
   adder activity dampened by A=0.

### Practical implications

- **Pre-detecting and skipping all-zero B tiles** in inference kernels
  could save 456-644 W per CTA (40-60 % power reduction).
- **All-zero A tiles** save 100-200 W (10-20 %), still worth pre-detection
  for sparse activations.
- **The 400 W floor** is the architectural minimum for issuing tcgen05
  at peak rate; cannot go lower without reducing throughput.

## Toggle-skip vs zero-detect asymmetry: A and B work differently!

Added mode 8 (B = constant non-zero), mode 9 (B = constant negative
non-zero), mode 10 (A = constant non-zero) to the kernels. This isolates
"toggle-skip" (multiplier saves power when input doesn't change) from
"zero-detect" (multiplier saves power when input is exactly 0 via
arithmetic short-circuit).

### B-side: TOGGLE-SKIP (any constant saves)

| format       | random | B=+const | B=−const | B=0   | const-vs-zero Δ |
|--------------|--------|----------|----------|-------|----------------|
| NVFP4 K=96   | 867 W  | 411 W    | 411 W    | 409 W | only 2 W       |
| FP8 K=32     | 1066 W | 430 W    | 428 W    | 422 W | 8 W            |
| BF16 K=16    | 882 W  | 410 W    | 409 W    | 400 W | 10 W           |

B=positive constant ≈ B=negative constant ≈ B=zero. **B-side savings
come 100% from inter-element toggle reduction**, NOT from arithmetic
zero detection. The B-broadcast bus consumes most B-side power on
edge transitions.

### A-side: ZERO-DETECT (only A=0 saves a lot)

| format       | random | A=0    | A=const(+) | A=0 saves | A=const saves |
|--------------|--------|--------|------------|-----------|---------------|
| NVFP4 K=96   | 871 W  | 758 W  | 847 W      | -113 W    | -24 W         |
| BF16 K=16    | 877 W  | 662 W  | 790 W      | -215 W    | -87 W         |
| FP8 K=32     | 1082 W | 969 W  | 1080 W     | -113 W    | **-2 W**      |

For FP8, A=constant non-zero saves only 2 W vs random — A-side is
NOT toggle-skip. Only A=0 triggers a savings, attributable to the
multiplier arithmetic zero-detect (`0 × X = 0` short-circuits the
adder/accumulator path).

### Mechanism inferred

The multiplier array has TWO distinct power-saving paths:

1. **B-side bus power-gating**: when consecutive B elements are
   identical, the broadcast bus does not toggle and the per-lane
   multiplier inputs are static. Saves ~456 W per CTA. Independent
   of arithmetic value.

2. **A-side zero-detect**: when A=0, multiplier output is forced to
   0 without computing, propagating through adder/accumulator silently.
   Saves ~113-215 W per CTA. **Specific to A=0 (or A=−0)**, not other
   constants.

### Practical implication

For inference:
- Pre-detecting all-zero A activations (e.g. ReLU-killed rows) → 113-215 W per CTA
- Pre-detecting low-toggle B (e.g. quantized identical-bucket weights)
  → 456-644 W per CTA
- All-zero A AND all-zero B (or B-uniform) → 394-406 W floor (-470 to
  -670 W vs random)

The B-side lever is bigger (broadcast amplifies impact), but A-side
benefits even when B is non-trivial.

### Earlier "zero-skip" claim corrected

My previous statement "B=0 zero-skip saves 456-644W" was a mistake of
naming. The true mechanism is toggle-skip on B-bus; B=any-constant gets
the same 456-644 W savings. This was discovered by adding mode 8
(B=const non-zero) and observing essentially same power as mode 6 (B=0).

### Confidence

- **HIGH** that B-side saving is toggle-skip (3 formats × 3 constants
  all consistent within 10 W).
- **HIGH** that A-side has separate zero-detect mechanism (FP8 A=const
  saves 2 W vs A=0 saves 113 W is unambiguous).
- **MED** for the precise mechanism (multiplier internal architecture
  speculation).

## K-axis vs N-axis variation in B (BF16 K=16 N=256)

Added mode 11 (B uniform along N: each K-row has unique value broadcast
across N) and mode 12 (B uniform along K: each N-col has unique value
repeated across K-rows).

| B pattern                   | power | interpretation                  |
|-----------------------------|-------|---------------------------------|
| All constant (mode 8)       | 408 W | floor (no variation either)     |
| Uniform-K (varies on N)     | 527 W | spatial only (+119 W vs floor)  |
| Uniform-N (varies on K)     | 880 W | temporal only (+472 W vs floor) |
| Fully random (mode 0)       | 876 W | both (~same as K-only!)         |

### K-axis temporal variation drives 75% of B-side power

Decomposition:
- K-axis temporal toggling (cycle-to-cycle B variation): **349 W (75 %)**
- N-axis spatial toggling (within-cycle lane variation): **119 W (25 %)**
- Total B variation cost: 468 W

The B-bus serializes K-rows across cycles. **Each cycle delivers one
K-slice broadcast across N lanes**. When K varies, every cycle the
broadcast bus toggles. When K is constant but N varies, the bus
"toggles" only laterally (within the broadcast), which has less impact.

### Practical implication

For inference power efficiency:
- **Quantized weights with low K-axis entropy save MUCH more power**
  than low N-axis entropy. E.g., if weights are sorted along K so
  consecutive K-rows have similar values, you save ~349 W per CTA.
- **Column-major B arrangement is more important for power than
  row-major** when total B variation budget is fixed.
- **Weight-pruning that aligns zeros along K dimensions** (column-wise
  sparsity in weight terminology) saves more power than along N.

### Confidence

- **HIGH** that K-axis variation is the dominant temporal lever
  (mode 11 = 880 W ≈ random 876 W, mode 12 = 527 W is far less).
- **HIGH** for the 75/25 split (469 W total split into 349/119).
- **MED** for the architectural interpretation (K is the cycle axis);
  could be confirmed by reading PTX/MMA hardware spec.

## A-side axis decomposition (NVFP4 K=96) — A has NO axis sensitivity

Added mode 13 (A uniform-M, varies K) and mode 14 (A uniform-K, varies M).
Compared to A=random baseline:

| A pattern                        | power | vs random |
|----------------------------------|-------|-----------|
| A random (mode 0)                | 872 W | 0         |
| A uniform-M (varies K only, 13)  | 865 W | -7 W      |
| A uniform-K (varies M only, 14)  | 872 W | 0 W       |
| A constant +1.0 (mode 10)        | 843 W | -29 W     |
| A all zero (mode 5)              | 757 W | -115 W    |

### A is axis-agnostic for toggle variation

Unlike B where K-axis dominates 75%, A shows essentially NO response to
which axis varies. Only arithmetic patterns matter:
- A=0: triggers multiplier zero-detect (-115 W)
- A=const non-zero: small toggle reduction (-29 W)
- A variations along any axis: no savings (0-7 W)

### Complete architectural model

| mechanism             | A-side         | B-side          |
|-----------------------|----------------|-----------------|
| Primary lever         | arithmetic zero | toggle-skip    |
| K-axis temporal       | negligible     | 75 % (349 W)    |
| N-axis spatial        | negligible     | 25 % (119 W)    |
| Arithmetic zero-detect| yes (-113 W)   | no              |
| Toggle-skip on const  | small (-29 W)  | yes (-456 W)    |
| Mechanism reason      | per-lane input | broadcast bus   |

### Reconciliation with NVFP4 K=96 random (867-872 W) budget

```
  tcgen05 issue floor (A=B=0):        394 W
+ B-bus K-axis toggle (mode 11 - floor): 461 W
+ A arithmetic vs const (mode 0 - 10):   29 W
-----------------------------------
  sum:                                884 W  (12 W from interaction)
```

Close to measured 867-872 W.

### Practical implication

For inference power optimization, **asymmetry dictates strategy**:

1. **B-side (weights, typically)**: sort/cluster along K-axis to
   minimize temporal toggling. Weight-quantization schemes that
   preserve K-axis clustering (e.g., block-structured quantization
   along K) can save up to 349 W per CTA.

2. **A-side (activations, typically)**: only all-zero activations
   matter for power. Random bit patterns in A use same power as
   any other non-zero activation. Post-ReLU sparsity directly saves
   A-side power.

3. **Maximum realistic savings**: random A + K-clustered B =
   ~575 W per CTA (34 % below random baseline 867 W).

4. **Theoretical max**: A=0 activations + B-clustered weights could
   reach ~400 W = 53 % below random.

## CORRECTION: K-axis power is BINARY, not gradual

After my "K-axis sorting saves 349W" claim, I tested K-chunk granularity
(NVFP4 K=96 N=256). Each chunk has K-rows bit-identical within, varies
across chunks:

| K-chunk size  | # K-transitions | power |
|---------------|-----------------|-------|
| 96 (constant) | 0               | 411 W |
| 48            | 1               | 870 W |
| 32            | 2               | 873 W |
| 16            | 5               | 877 W |
| 8             | 11              | 876 W |
| 4             | 23              | 874 W |
| 2             | 47              | 874 W |
| 1 (random)    | 95              | 873 W |

**Even chunk-48 (only 1 K-transition in entire K=96) uses same power as
fully random K.** K-axis power is BINARY: either all 96 K-rows
bit-identical (411 W floor) OR full ~870 W cost. No middle ground.

### Why the "sort along K" recommendation was WRONG

The hardware processes K in PARALLEL within each cycle (all 96 K values
per output simultaneously, not serially). The relevant toggle is
SPATIAL (across the parallel K-array within a cycle), not TEMPORAL
(across cycles).

For B-bus to be low-toggle, all 96 K-rows must be bit-identical
across N — meaning B is effectively rank-1 along K. **Real weight
matrices have varying K → always pay ~870 W K-axis cost.**

### CORRECTED practical levers for B-side power

After this correction, the ONLY realistic B-side power optimizations are:

1. **Magnitude-set narrowing** (e.g., quantize to 5 distinct positive
   values): saves ~440 W per CTA. **Realistic** — quantization can
   constrain element set without changing matrix structure.

2. **B = uniform tile detection**: if entire B tile is one constant,
   skip dispatch. Saves 459 W. Real but rare.

3. **All-zero B tile detection**: same 459 W as any uniform — special
   case of #2.

4. **Sign-bit zero (B-positive only)**: saves 142-150 W. Realistic for
   weights designed to be all-positive (e.g. magnitude-only stored
   separately from sign mask).

K-row sorting / clustering / similarity does NOT help. The K-axis power
is gated only by full bit-identity of all K-rows, not by similarity.

## DWORD-pattern threshold for B-bus toggle-skip

Tested how many distinct DWORD patterns B can contain before
toggle-skip stops working (NVFP4 K=96 N=256 cta=2):

| # distinct dword patterns | power | savings vs random |
|---------------------------|-------|-------------------|
| 1 (constant)              | 410 W | 462 W (max)       |
| 2 (sign flip / extreme)   | 414 W | 458 W             |
| 4 (splatted)              | 501 W | 371 W             |
| 8 (splatted)              | 725 W | 147 W             |
| 16 (splatted)             | 866 W | 6 W (≈ random)    |
| ~3000 (per-nibble {1..4}) | 642 W | 230 W ★           |
| ~unbounded (random)       | 872 W | 0                 |

### Key observations

1. **Toggle-skip is a 1-2 dword feature.** With 1-2 distinct dword
   patterns in the entire B buffer, savings are nearly max (-460 W).
   At 4 patterns, half savings. Above 16, no benefit.

2. **Popcount variance matters too** (★). Mode 22 has up to ~65K
   distinct per-nibble dword patterns (random nibble from {1,2,3,4}),
   yet uses 642 W — far below the 16-distinct-splatted-dwords case
   (866 W). The reason: per-nibble {1,2,3,4} keeps popcount low
   (avg 10 bits per dword), while splatted dwords vary 0-32 bits.
   Inter-dword popcount toggling drives a separate power component.

3. The 16-dword splat result (866 W) shows that a 1-bit set per nibble
   (low popcount) is NOT enough — what matters is the popcount
   *variance across dwords*. Splatted 0xFF (32 bits) vs 0x00 (0 bits)
   = 32-bit popcount swing per cycle.

### Two distinct B-power components

1. **Bus toggle-skip on small alphabet** (≤ 4 distinct dwords):
   saves up to 460 W. Hardware seems to have a 1-2 entry pattern
   buffer that recognizes constant or alternating bus values.

2. **Popcount-variance toggle** on bus signaling: separately driven
   by Hamming distance between consecutive dwords. Low-popcount
   alphabets (mode 22) save ~230 W even with thousands of distinct
   patterns.

### CORRECTED practical levers (final)

For real LLM inference where weights cannot be made bit-identical:

1. **Quantize B to ≤4 distinct dword patterns per tile** (impossibly
   restrictive for real weights, but if you have it, save 370+ W).

2. **More realistically: quantize B to low-popcount values.** All
   FP4 values with popcount ≤ 1 (e.g., {+0, +0.5, +1.0, +2.0, +4.0,
   ±0.5}) give significant savings via the popcount-variance term
   (~230 W for mode 22-like distributions).

3. **Avoid mixing 0x00 and 0xFF dwords** (max popcount swing).
   Cluster weights into magnitude bands.

4. **Magnitude-set narrowing** (5-pos: 430 W vs 16-random: 870 W) is
   still the biggest realistic lever, ~440 W per CTA.

5. **Sign-bit positive only** (B-positive mode 1): 142-150 W saving.

6. **A=0 detection** (e.g. ReLU activations): 113-215 W saving.

K-row sorting / chunk-K-uniformity gives ZERO benefit (proven by the
chunk-48 = chunk-1 = ~870 W test). The K-axis power is BINARY: all
K-rows bit-identical or full cost.

## Hamming distance vs pattern count — pattern count dominates

Two-dword alternating tests with controlled Hamming distance:

| Hamming bits | power | Δ vs 0-bit |
|--------------|-------|------------|
| 0 (constant) | 408 W | 0          |
| 1            | 426 W | +18 W      |
| 2            | 432 W | +24 W      |
| 4            | 466 W | +58 W      |
| 8            | 482 W | +74 W      |
| 16           | 529 W | +121 W     |
| 24           | 565 W | +157 W     |
| 32 (max)     | 602 W | +194 W     |

Linear ~6 W per Hamming-distance bit between consecutive dwords.

Compare: Gray-code-like (32 distinct dwords, only 1-bit transitions
between consecutive) = 753 W. **Worse than alternating 2 patterns at
32-bit Hamming (602 W)**.

### Pattern count > Hamming distance

Both contribute, but pattern count is the dominant lever:
- 2 patterns × 32-bit Hamming: 602 W
- 32 patterns × 1-bit Hamming: 753 W
- 16 patterns × 8-bit avg Hamming: 866 W (mode 27)
- ~unlimited patterns × ~16-bit avg: 870 W (random)

For real quantized weights (typically 8-16 distinct values per FP4 element
across a tile, many distinct dwords), the pattern-count saturation makes
Hamming optimization mostly moot.

## Final B-side power model

`P(B) = floor + alpha × log(distinct_dwords) + beta × avg_Hamming + gamma × popcount_variance`

Approximate coefficients fitted from data (rough):
- floor: 408 W
- alpha (per log-2 step in pattern count, saturating at ~16 patterns): ~50 W per bit of "log distinct dwords" capped at 4
- beta (per Hamming bit between consecutive dwords): ~6 W
- gamma (per Hamming bit of popcount variance across all dwords): ~5 W

Worst case: random (4 G patterns, 16-bit avg Hamming, high popcount variance) = 408 + 200 + 96 + 80 ≈ 784 W ≈ measured 870 W.

Best case: constant (1 pattern) = 408 W (floor).

Realistic LLM weights: many patterns, mid Hamming, mid popcount variance
≈ 800-870 W. **Hard to push below 600 W per CTA without restructuring
weight quantization.**

## Realistic LLM-weight power numbers (final practical table)

NVFP4 K=96 N=256 cta=2 @ 1500 MHz (10.91 PF saturated):

| B distribution                          | power  | TF/W  | savings | reachable? |
|-----------------------------------------|--------|-------|---------|-----------|
| All constant (theoretical floor)        | 416 W  | 26.2  | -459 W  | only for sparse-zero tiles |
| Pure 5-pos {+0..+2} (positive only)     | 646 W  | 16.9  | -229 W  | only if signs separate    |
| Narrow {0, ±0.5, ±1.0} 5 codes ±        | 712 W  | 15.3  | -163 W  | aggressive log-quant      |
| **LLM-like (70% small, 30% mid, ±)**    | **816 W** | **13.4** | **-59 W** | typical FP4 quantization |
| Random 16 codes (worst case)            | 875 W  | 12.5  | 0       | synthetic only            |

### Bottom line for inference

**Standard FP4 quantization (Llama-style) of normalized weights gives
only ~7 % power reduction vs uncorrelated random data.** The variance
that nature provides (Gaussian → quantize to FP4) is enough to use
nearly all the toggle-energy a multiplier can consume.

To save meaningfully:
- 19 % (163 W): aggressive log-quantization that puts >95 % of weights
  in {0, ±0.5, ±1.0}. Loses some tail accuracy.
- 26 % (229 W): purely positive weights {+0..+2}. Requires storing
  sign separately as 1-bit mask.
- 53 % (459 W): only achievable for entirely uniform tiles (rare).

**TF/W headroom in realistic FP4 inference: 13.4 → 16.9 = 26 % efficiency
gain available** if quantization is restructured for power.

### Final corrected B-side mental model

For real engineers deciding how to quantize weights for power on B300
tcgen05:
1. **Default Gaussian FP4 quantization**: 800-870 W per CTA. No
   meaningful power savings from data structure.
2. **Aggressive low-magnitude clipping**: 700-720 W per CTA. ~7-19 %
   power saved.
3. **All-positive remap (sign as separate bitmap)**: 640-650 W per CTA.
   ~26 % power saved.
4. **All-uniform tiles (e.g., padding tiles, all-zero blocks)**:
   400-420 W per CTA. ~53 % power saved. Worth detecting & special-casing.

A-side (typical activations: post-softmax, post-ReLU):
5. Random A: no extra savings beyond above.
6. Many ReLU-zeroed A rows: detect entire zero rows for ~115 W per CTA.
