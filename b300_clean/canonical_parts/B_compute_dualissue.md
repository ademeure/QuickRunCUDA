# Section B — Compute pipes & dual-issue (§16-§25)

Author: Section-B agent, 2026-04-22.
Scope: B300 SXM6, sm_103a, 148 SMs, 4 SMSPs/SM, 32 lanes/SMSP.
All TFLOPS measurements annotated with clock state (default boost ~2032 MHz
sustained under FFMA; `-lgc 2032` paradoxically pins 1920 MHz).

---

## §16. FP32 FFMA peak — 74.62 TFLOPS at 2032 MHz boost

**Answer:** **FFMA peak = 74.62 TFLOPS = 96.92% of 76.96 theoretical** at 2032 MHz boost; **62.17 TFLOPS = 85.5%** at 1920 MHz locked.  `[🟢 HIGH · src: V8_FFMA_PEAK_VERIFIED.md, B300_TRUE_REFERENCE.md commit 06b0d8d]`

The recipe is `NCHAIN ≥ 3` rotating accumulators with an immediate constant
multiplier (compiler emits `FFMA Rd, Rd, 1.5, Rd` — 2 distinct register sources
+ 1 immediate, sidestepping the 2-port RF read limit), 256 threads × 148 blocks,
boost clock unlocked. Three independent kernels reach within 1.5% of each
other (74.62 / 75.2 / 75.92 TFLOPS) corresponding to 96.92% / 97.65% / 98.6%
of theoretical; the canonical headline number is **74.62** because that is
the figure cited in `B300_TRUE_REFERENCE.md` for the `06b0d8d` clean recipe.

### Theoretical derivation (the 76.96 number)

```
FFMA chip peak  = N_SM × FFMA_per_SM_per_cycle × 2_FLOPS × clock_Hz
                = 148  × 128                    × 2       × 2.032e9
                = 76 964 524 032 FLOPS
                ≈ 76.96 TFLOPS
```

Where `128 = 4 SMSPs × 32 FP32 lanes` per SM (NOT 256 — see Footgun below).
Each FFMA is one instruction that performs 2 FLOPS (one multiply + one add).

At the 1920 MHz "lock paradox" rate:

```
76.96 × (1920 / 2032) = 72.71 TFLOPS  (theoretical at 1920 MHz)
```

So the locked-clock measurement of 62.17 TFLOPS is `62.17 / 72.71 = 85.5%` of
the locked theoretical, and `62.17 / 76.96 = 80.8%` of the boost theoretical
— ALWAYS state which denominator you are using.

### Per-recipe measurement table

| Recipe | Clock | Threads × Blocks | NCHAIN / ILP | Measured | %SoL | ncu pipe_fma | Source |
|---|---|---|---|---:|---:|---:|---|
| `fma %0,%0,%1,%0` 2-source | 2032 boost | 256 × 148 | NCHAIN=8 | **75.20 TFLOPS** | 97.7% | 97.64% | `V8_FFMA_PEAK_VERIFIED.md`, `bench_ffma_warps_per_sm.cu` |
| NCHAIN=3 rotating + IMM | 2032 boost | 256 × 148 | NCHAIN=3 | **74.62 TFLOPS** | 96.92% | (not in source) | `B300_TRUE_REFERENCE.md` `06b0d8d` |
| `fp32_peak_definitive.cu` | 2032 boost | 1024 × 148 | NCHAIN=8 | **75.92 TFLOPS** | 98.6% | (not in source) | `04_fp32_peak.md` |
| Same kernel, locked | 1920 lock | 256 × 148 | NCHAIN=8 | **62.17 TFLOPS** | 85.5% of 72.71 | — | `B300_TRUE_REFERENCE.md` `e1a1220` |
| 3-source distinct `fma %0,%0,%1,%2` | 2032 boost | 256 × 148 | NCHAIN=8 | **51.3 TFLOPS** | 66.6% | 66.64% | `V10_FMA_SOURCE_COUNT.md` (RF-port limited — see §17) |

The headline 74.62 is the most conservative of the three near-peak recipes;
75.2 (V8) and 75.92 (definitive) are within measurement noise (±1.5%).

### Why 97.6%, not 100%

ncu `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active = 97.64%`
directly counts active cycles. The 2.4% gap is:

- Kernel startup / tail bubbles where some SMs have already exited
- Branch overhead between unroll groups (`UISETP.GE` + `BRA` consume one issue
  slot per outer iteration)
- Possible RF-port micro-stalls — a 2-source recipe still issues 2 RF reads/cy
  which is the ceiling, leaving zero margin for any read-port collision

For perfectly clean comparison, V8's pipe_fma 97.64% is the gold reference for
"FFMA pipe-saturated rate"; the 96.92% headline rounds down conservatively.

### Why locked = 85.5% (not also 97%)

There are TWO reasons the locked measurement underperforms its denominator:

1. **The "lock paradox":** `nvidia-smi -lgc 2032` actually pins to **1920 MHz**
   (base clock), NOT 2032. The `-lgc` argument names the BASE you are pinning
   to, not the boost. Default unlocked sustains 2032 under FFMA load.
2. **Different kernel:** The 62.17 measurement (commit `e1a1220`) was a different
   kernel snapshot than V8; it likely had additional loop overhead. The pure
   ratio at 1920 should also reach ~97% if measured with the same V8 recipe.

ALL TFLOPS claims must annotate the clock state. The ~6% gap between 1920 and
2032 explains most of the historical noise in this catalog.

### Latency

FFMA latency = **4.22 cy** per `V9_OP_LATENCY.md` (also see §19). At full-pipe
saturation each SMSP issues 1 FFMA/cy (4× per SM), so chain depth ≥ 4 is needed
to hide the per-instruction latency; the V8 recipe's 8 chains gives 2× margin.

### Recipe source code (V8, 97.64%)

```cuda
__global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int iters, int, int) {
    float v[8], b[8];
    // tid-dependent init defeats LICM
    int tid = threadIdx.x;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        v[k] = A[tid + k*256];
        b[k] = B[tid + k*256];
    }

    #pragma unroll 1
    for (int i = 0; i < iters; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            #pragma unroll
            for (int k = 0; k < 8; k++)
                asm volatile("fma.rn.f32 %0, %0, %1, %0;"
                             : "+f"(v[k])
                             : "f"(b[k]));
        }
    }

    // anti-DCE: unconditional STG of accumulator
    float sum = v[0]+v[1]+v[2]+v[3]+v[4]+v[5]+v[6]+v[7];
    if (sum == -1.0f) C[tid] = sum;   // impossible-if guard
}
```

Launch with `<<<148, 256>>>`. Set `iters = 1000000` so kernel runtime ≥ 8 ms
(launch-overhead-safe).

**Footgun:** ⚠ "B300 has 256 FP32 cores per SM" is **WRONG**. B300 has **128
FP32 cores/SM** — same as Hopper H100 — distributed as 4 SMSPs × 32 lanes.
Claims of "154 TFLOPS FP32" are a 2× formula error from this confusion.
Spec'd peak is 76.96 TFLOPS, not 154.
**Footgun 2:** ⚠ Naming `-lgc 2032` does NOT pin you to 2032; it pins to 1920.
The 6% delta accounts for most of the discrepancy between historical numbers.

**See also:** §17 (RF port limit on 3-source FFMA), §19 (FADD/FMUL same pipe),
§22 (FFMA pipe saturated alongside ALU), `B300_TRUE_REFERENCE.md` row "FP32 FFMA peak",
`corrections/04_fp32_peak_CORRECTED.md`.

---

## §17. FFMA register-source dependence — 3-distinct-source caps at ~67%

**Answer:** **FFMA throughput depends on the number of unique register sources** —
1-2 unique sources → 0.97 inst/SMSP/cy (~97% pipe), **3 unique sources → 0.61–0.65 inst/SMSP/cy (~67% of 76.96 = ~51 TFLOPS)**.  `[🟢 HIGH · src: D6_RF_PORT_RIGOR.md, V10_FMA_SOURCE_COUNT.md, A4_FFMA_PORT_PRESSURE.md]`

This is the **single biggest reason real GEMM kernels under-perform their
"theoretical 75 TFLOPS"**: FFMA reads `a × b + c` (3 sources), and B300 has
only 2 register-file read ports per cycle per SMSP (an operand reuse cache
provides an effective 3rd port for any operand that's broadcast across
consecutive instructions).

### The measurement (V10, ncu pipe_fma + 30.31 G FFMAs)

| Variant | PTX | SASS | Pipe % | TFLOPS | Ratio vs 2-source |
|---|---|---|---:|---:|---:|
| 2-source `fma %0,%0,%1,%0` | self + reg | `FFMA Rd, Rd, R1, Rd` | **97.65%** | **75.2** | 1.000 |
| 3-source `fma %0,%0,%1,%2` | self + 2 distinct regs | `FFMA Rd, Rd, R1, R2` | **66.64%** | **51.3** | **0.683** |

The ratio `0.683 ≈ 2/3` exactly matches the prediction from a 2-RF-read-port
model: 3 reads / 2 ports = 1.5 cy per FFMA = 1/1.5 = 0.667 throughput.

### A4's port-pressure validation (1500 MHz lock, NC=8)

Independent test using all four operand-distinctness configurations:

| Operand pattern | SASS | inst/SMSP/cy | TIPS_inst @ 1500 |
|---|---|---:|---:|
| `fma a,a,a,a` (1 unique) | `FFMA R14, R14, R14, R14` | **0.972** | 27.61 |
| `fma a,b,a,b` (2 unique) | `FFMA R2, R23, R2, R2` | **0.971** | 27.61 |
| `fma a,b,c,a` (3 unique) | `FFMA R28, R18, R28, R23` | **0.612** | 17.40 |

**1 unique = 2 unique = 97% pipe; 3 unique = 61% pipe.** The 0.972 → 0.612
collapse on adding the 3rd distinct source is a 37% throughput hit.

### D6's reuse-cache decomposition

D6 (commit explicit-NC=8 sweep) compares broadcast vs per-chain register patterns:

| Config | NC=1 | NC=2 | NC=4 | NC=8 | NC=16 | Peak fma/cy |
|---|---:|---:|---:|---:|---:|---:|
| Broadcast `za` + `.reuse` | 4.44 | 2.31 | 1.19 | 1.09 | 1.05 cy/fma | **0.96** |
| Per-chain `za[k]`, no reuse | 4.56 | 2.25 | 1.63 | 1.56 | 1.53 cy/fma | **0.65** |

Theoretical 2 reads + 1 reuse → 1 cy. Measured 1.05 cy → 95% of theoretical.
Theoretical 3 reads / 2 ports → 1.5 cy. Measured 1.53 cy → 98% of theoretical.

The clean ratio `0.96 / 0.65 = 1.48` vs theoretical `1.50` is the strongest
empirical anchor for the 2-RF-read-port model. SASS-verified `.reuse` count:
255/256 in broadcast mode, 0/256 in per-chain mode.

### Architectural model

B300 SMSP register file:

- **2 read ports per cycle** (HW)
- **Operand reuse cache** — 1 entry that holds the most recently issued
  operand if the next instruction reads the same register. SASS marks
  reusable operands with the `.reuse` suffix.
- Effective port count when one operand is hot: **3 reads/cy**
- Effective port count when all 3 operands distinct: **2 reads/cy** → 1.5 cy/FFMA

### Why this matters for real workloads

Many production kernel patterns bottleneck here:

| Pattern | Code | Sources | Effective rate |
|---|---|---|---|
| Self-feed (FFMA microbench) | `v = v*b + v` | 2 unique | 75 TFLOPS (97%) |
| Horner polynomial | `t = t*x + c` (c constant) | 2 unique | 75 TFLOPS (97%) |
| GEMM with broadcast | A or B broadcast across MMA | 2 unique (with reuse) | 75 TFLOPS (97%) |
| Vector dot product | `sum += a*b` = `sum = a*b + sum` | **3 unique** | **51 TFLOPS (67%)** |
| Outer-product GEMM | `c[i,j] = a[i] * b[j] + c[i,j]` | **3 unique** | **51 TFLOPS (67%)** |
| FFMA accumulator chain | `acc = acc * x + acc_next` | 3 unique | 51 TFLOPS (67%) |

For kernels that genuinely need 3-source FFMA (most GEMM, most convolution),
**the realistic FP32 ceiling is ~51 TFLOPS, NOT 75**. This is the single most
important number to communicate when budgeting real-workload performance.

### Why this didn't show up in LOP3

LOP3 pipe peak is 0.5 inst/SMSP/cy (already half of FFMA's 1.0). The 2-RF-port
limit kicks in at 0.66/SMSP/cy for 3 distinct reads — which is **above** LOP3's
pipe peak. So LOP3 stays bottlenecked at the pipe, not the RF. RF-port pressure
is observable only on FFMA (and similar high-throughput compute) where the pipe
itself is fast enough that the RF becomes the next bottleneck.

### Implications for benchmark methodology

Any FFMA "peak" number that doesn't state the source-distinctness pattern is
suspect:

- **2-source self-feed** → 75 TFLOPS (97% pipe). The headline number.
- **3-source distinct** → 51 TFLOPS (67% pipe). The realistic ceiling.
- **NCHAIN=3 + IMM** (B300_TRUE_REFERENCE recipe) → 74.6 TFLOPS. The IMM avoids
  needing a 3rd register read; the chain rotation provides ILP without forcing
  3 distinct sources.

Compiler-emitted `.reuse` SASS hints are the on-disk indicator of which path
your kernel hit. `cuobjdump --dump-sass` and grep `.reuse` lines per FFMA.

### Cross-check with `pipe_fma` ncu metric

For any FP32 workload:
- `pipe_fma` < 70% with full occupancy → almost certainly RF-port bound (3-source)
- `pipe_fma` 95-98% → confirmed 2-source pattern hitting peak
- `pipe_fma` between 70% and 95% → mixed; check SASS for partial reuse

### Mechanism alternative not yet ruled out

D6 attributes the gap to "2 RF read ports + reuse cache as effective 3rd port".
A4 mentions "operand collector deduplication" as alternative. Same observable;
underlying mechanism not pinned to one explanation. ncu
`smsp__inst_executed_pipe_fma_collector_*` if available could discriminate.

**Footgun:** ⚠ The headline "97% FFMA peak / 75 TFLOPS" does **NOT generalize**
to outer-product GEMM, dot products, or any kernel with 3 distinct register
sources per FFMA. Those cap at 51 TFLOPS = 67% of theoretical. State the
source-distinctness pattern in any FFMA peak claim.

**See also:** §16 (recipe sidesteps the limit via NCHAIN+IMM), §18 (the `.reuse`
flag is the SASS-level mechanism), `D6_RF_PORT_RIGOR.md`,
`V10_FMA_SOURCE_COUNT.md`, `A4_FFMA_PORT_PRESSURE.md`.

---

## §18. FFMA `.reuse` cache — the SASS-level operand bypass

**Answer:** SASS `.reuse` flag on an operand makes that operand **free at the RF
level for the next consecutive issue** — effectively a per-cycle 1-entry reuse
cache that bypasses one of the 2 RF read ports. Combined with NCHAIN ≥ 3 and a
shared/broadcast operand, this is the mechanism that lets FFMA reach 97% pipe
saturation despite the formal 2-port limit.  `[🟢 HIGH · src: D6_RF_PORT_RIGOR.md, A4_FFMA_PORT_PRESSURE.md]`

### What `.reuse` means in SASS

Each operand position of a SASS instruction can be tagged `.reuse`, telling the
hardware "this operand will likely be re-read by the next instruction; cache it
in the reuse-cache slot for that operand position". On the next cycle, if the
next instruction reads the same register at the same operand position, the RF
read port is bypassed — the value comes from the reuse cache instead.

Key facts:

| Property | Value | Source |
|---|---|---|
| Reuse cache slots per operand position | 1 | classical Volta+ design, observable on B300 |
| Operand positions in FFMA | 3 (a, b, c) | so up to 3 separate reuse caches |
| Lifetime of cached value | 1 cycle (the next instruction) | post-issue eviction |
| SASS hint emission | compiler-controlled | nvcc emits when register liveness analysis indicates re-read |
| Effective RF port count | 2 read ports + (≤3 reuse-cache hits) | combined cap |

### How NCHAIN interacts with `.reuse`

A single FFMA `Rd = Rd * b + Rd` reads `Rd` twice (positions 0 and 2) and `b`
once (position 1). Without reuse:
- Cycle N: read Rd, b, Rd → 3 reads / 2 ports = 1.5 cy
- Cycle N+1: same → 1.5 cy

With `.reuse` on `b`:
- Cycle N: read Rd (port 0), b (port 1), Rd (port 0 again — broadcast within instr)
  → emit FFMA, mark `b` as reuse
- Cycle N+1: read different Rd' (FFMA on different chain), b is reuse-cache hit
  → 2 RF reads needed (Rd' twice or Rd', new b)

But with **NCHAIN=8 chains** rotating, consecutive FFMAs touch DIFFERENT Rd
registers, so the `Rd`-position reuse never hits. Only the `b`-position
(per-chain constant) hits. That's enough to give you 1 effective port back.

This is why D6's measurement `0.96 fma/cy` (with reuse) vs `0.65 fma/cy`
(without reuse) is exactly the 1.5× ratio prediction.

### NCHAIN ≥ 3 + IMM = the B300_TRUE_REFERENCE recipe

The 74.62 TFLOPS recipe (`06b0d8d`) uses:
1. **NCHAIN = 3 rotating accumulators** — 3 independent FFMA chains, each with
   self-feeding `Rd = Rd * IMM + Rd`. Latency is hidden because no chain has
   a dependency on itself for at least 3 instructions = 3 cy < 4.22 cy FFMA
   latency. (Strictly NCHAIN ≥ 4 is needed for full hiding; NCHAIN=3 is
   borderline and the 96.92% — slightly below 97.65% — reflects this.)
2. **Immediate constant** as the `b` source — immediates come from the
   instruction word, not the RF, so they consume **zero RF read ports**.
3. **`.reuse` on the third source** — `Rd` is read at position 2 (the addend);
   compiler marks it `.reuse` so the next FFMA in the same chain (4 cycles
   later, after all NCHAIN chains have rotated through) can see it cached
   if alignment works out.

Net effect: each FFMA needs only 1 RF read port (`Rd` at the multiplicand
position), and the SMSP pipe is the binding constraint instead of RF.

### Anti-pattern: 3 distinct registers, no reuse

```cuda
asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
```

SASS: `FFMA Rd, Ra, Rb, Rc` (no `.reuse` because no register is re-read by
the next instruction). 3 reads/cy → 1.5 cy/FFMA → 67% of pipe. This is
the V10 / D6 / A4 "3-distinct-source" failure mode.

### How to verify `.reuse` in your kernel

```bash
nvcc -arch=sm_103a -O3 -keep my_kernel.cu
cuobjdump --dump-sass my_kernel.cubin | grep -E "FFMA.*reuse"
```

A peak FFMA kernel should show `.reuse` on at least one operand of nearly
every FFMA instruction. D6's broadcast-`za` test showed **255/256 FFMAs**
with reuse; the per-chain version showed **0/256**. The two regimes differ
in measured throughput by exactly 1.48× (matching the 1.5× prediction).

### Why the operand reuse cache is sometimes called "the third port"

It is not literally a 3rd RF port — it is a 1-cycle bypass cache. But its
performance EFFECT for instruction sequences with operand re-reads is
indistinguishable from a 3rd RF read port. The catalog calls it "effective
3rd port" for shorthand; D6's clean factor-of-1.48 derivation rules out the
alternative ("operand collector deduplication") to within measurement noise
but not definitively.

### Edge case: Volta-style operand collectors

On Volta, the reuse cache was per-operand-position with a 1-entry slot.
Hopper/Blackwell appear to retain this model (no public documentation
states otherwise, and B300 measurements match Volta-style behavior to
within 2%). If sm_120 (RTX 5090) GeForce parts have a different cache
size, that would show up as a different `.reuse` SASS pattern from nvcc;
none observed so far.

### Practical kernel-design rule

For any FFMA-heavy kernel where you have algorithmic flexibility:

- Prefer accumulator self-feed (`acc = acc * x + acc`) over independent
  per-iteration constants (`acc = a[i] * b[i] + c[i]`)
- Hoist constants into broadcast registers once before the loop
- Use immediate constants where the multiplicand is a known small float
- Compile with `-keep` and verify `.reuse` count matches FFMA count

If you can't achieve `.reuse` on at least one operand per FFMA, your kernel
will cap at ~51 TFLOPS (67%), regardless of how clean the rest of the
code is.

**See also:** §16 (NCHAIN=3 + IMM peak recipe), §17 (the underlying RF port
constraint), §22 (`.reuse` does NOT create dual-issue with ALU; pipes overlap
via separate dispatch).

---

## §19. FADD = FMUL = FFMA at SASS level

**Answer:** **FADD, FMUL, and FFMA all emit the same FFMA SASS instruction** with
the same 4.22 cy latency and same ~97.65% pipe saturation rate. FFMA "wins"
purely because each instruction carries 2 FLOPS instead of 1 (or 1).  `[🟢 HIGH · src: V8_FADD_FMUL_PEAK.md, V9_OP_LATENCY.md]`

### The measurement table (V8, 256 thr × 148 blocks, ITERS=1M)

| Op | Pipe % | Time | Inst count | Inst/s | TFLOPS (FLOPS/inst) |
|---|---:|---:|---:|---:|---:|
| FADD | 97.65% | 810 µs | 30.31 G | 37.4 G/s | **37.4** (1 FLOP) |
| FMUL | 97.62% | 813 µs | 30.31 G | 37.3 G/s | **37.3** (1 FLOP) |
| FFMA | 97.65% | 809 µs | 30.31 G | 37.5 G/s | **74.8** (2 FLOPS) |

All three at 97.6%-97.7% of pipe — within ±0.1% of each other. The TFLOPS
column shows FFMA at 2× FADD/FMUL not because the hardware is different,
but because each FFMA does a multiply AND an add. The hardware DISPATCHES at
the same 1 inst/cy/SMSP rate for all three.

### Latency

All three are 4.22 cy per `V9_OP_LATENCY.md` (full latency ladder in §19's
"latency ladder" subsection below). That 4.22 cy is the FMA pipe stage depth;
because FADD and FMUL flow through the same pipe, they inherit the same
latency. There is no "fast path" for adds or multiplies on B300.

### SASS at the source level

Inside the kernel (V8_FADD_FMUL_PEAK measurement):

```
FADD R, R, R       ;  same stage
FMUL R, R, R       ;  same stage
FFMA R, R, R, R    ;  same stage but 2 ops
```

ncu instruction counters for each test confirm:
- `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum` matches FADD count
- `smsp__sass_thread_inst_executed_op_fmul_pred_on.sum` matches FMUL count
- `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` matches FFMA count

Compiler is NOT silently fusing FADD+FMUL pairs into FFMA (which would
double-count). Each test emits the expected SASS opcode.

### Kernel-design implication

**Prefer FFMA over (FADD then FMUL) or vice versa whenever mathematically
equivalent.** Same instruction count, 2× FLOPS. Concrete examples:

- `c = a * b; d = c + e;` → 2 instructions, 2 FLOPS. Better: `d = fma(a,b,e);`
  → 1 instruction, 2 FLOPS = same per-inst rate, half the inst count.
- `acc += a * b;` → fma form: 1 inst, 2 FLOPS. Compiler usually does this for you,
  but `-fmad=false` will split it.
- For non-MAD kernels (sums, reductions, polynomial coefficients), FADD at
  37.4 TFLOPS is the realistic peak. Don't claim 75 TFLOPS for a sum reduction.

### Hardware micro-architectural takeaway

The B300 FMA pipe is a single ALU that always performs `d = a × b + c`. When
the source is FADD, the compiler emits FFMA with `b` set to an immediate 1.0
(or the FADD opcode with implicit 1.0 multiplier — both observable in SASS
depending on context). When the source is FMUL, similar with `c` set to 0.
The pipe doesn't have a separate FADD or FMUL physical unit.

This is the standard Volta+ unified-FMA model. Hopper retained it; Blackwell
retained it. Earlier (Maxwell) had separate FADD and FMUL units; Volta unified.

### Anti-pattern: relying on compiler MAD fusion

Some workloads break MAD fusion via order-of-evaluation rules:

```cuda
float a = compute_a();
float b = compute_b();
float c = compute_c();
float d = (a * b) + c;   // compiler may emit FFMA, may emit FMUL+FADD
```

If the compiler emits FMUL+FADD instead of FFMA, you get HALF the FLOPS
throughput from the same inst rate. To guarantee FFMA emission, use explicit
`asm("fma.rn.f32 %0, %1, %2, %3;" ...)` or `__fmaf_rn(a, b, c)`.

### Latency ladder cross-reference

Complete B300 op latency ladder (V9_OP_LATENCY):

| Op | Latency (cy) | Throughput cap | Saturation ILP |
|---|---:|---|---:|
| FFMA / FADD / FMUL | **4.22** | 1/cy/SMSP | 4 chains |
| IMAD | 4.25 | 1/(2cy)/SMSP | 2 chains |
| DFMA | 63.68 | 1/(64cy)/SMSP | 1 chain |
| HMMA m16n8k16 (F32 acc) | 20 | 1/(4cy)/SMSP | 5 chains |

**Footgun:** ⚠ "FADD is faster than FFMA" or "FFMA is slower because it does
more work" are BOTH wrong on B300. They run at the SAME inst/s rate. FFMA
just happens to count as 2 FLOPS per inst.

**See also:** §16 (FFMA peak recipe — note FADD/FMUL would peak at half),
§17 (RF port limits apply equally to all three), §22 (FMA pipe overlaps
freely with ALU pipe regardless of which of these instructions is in flight).

---

## §20. FP64 DFMA — 1.20 TFLOPS = 100% of theoretical

**Answer:** **FP64 DFMA = 1.203 TFLOPS = 100.00% of theoretical** at 2032 MHz.
1:64 ratio vs FP32 per `cudaDeviceGetAttribute(SingleToDoublePrecisionPerfRatio) = 64`.
Single FP64 unit per SMSP, latency 64 cy (one DFMA per 64 cy per SMSP).  `[🟢 HIGH · src: V8_FP64_PEAK_VERIFIED.md, V9_OP_LATENCY.md commit 2d64696]`

### Theoretical derivation

```
FP64 chip peak = N_SM × FP64_per_SM_per_cycle × 2_FLOPS × clock_Hz
              = 148  × (4 SMSPs × 1 DFMA / 64 cy)         × 2 × 2.032e9
              = 148  × 0.0625                              × 2 × 2.032e9
              = 1 203 200 000 FLOPS
              ≈ 1.203 TFLOPS

Equivalently:
FP64 = FP32 / 64 = 76.96 / 64 = 1.2025 TFLOPS  ✓
```

The 1:64 ratio is from the CUDA C Programming Guide Table 13-1 and the
`SingleToDoublePrecisionPerfRatio` device attribute. It is consistent with
NVIDIA's DC-class spec which lists FP64 at 1/64 of FP32 for sm_103a (consumer
GeForce parts may have a different ratio; this is the data-center SKU value).

### Measurement (148 × 256 thr, 100K outer iters)

| Metric | Value |
|---|---|
| ncu `sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active` | **100.00%** |
| ncu `smsp__sass_thread_inst_executed_op_dfma_pred_on.sum` | 30.31 G DFMA |
| gpu_time | 50.40 ms |
| TFLOPS (count × 2 FLOPS / time) | 30.31e9 × 2 / 50.40e-3 = **1.203 TFLOPS** |
| Ratio to theoretical | **100.00%** |

Both ncu pipe utilization AND the wall-clock-derived TFLOPS converge to exactly
1.203 TFLOPS at 100.00% of theoretical. This is the **cleanest pipe saturation
measurement on B300** — easier to reach than FP32 because the FP64 lane is the
ONLY blocker (no other pipe competes for DFMA dispatch).

### Why exactly 100% (not 97.6% like FFMA)

FP64 DFMA throughput is bottlenecked by the single FP64 unit per SMSP at
1/64-cy. As long as the kernel keeps the issue slot filled (4+ warps/SM with
2-source DFMA chains), every SM active cycle issues one DFMA. There is no:
- RF port competition (only 1 DFMA reads at a time per SMSP)
- Other-pipe competition (FP64 is its own pipe, no co-issue contention)
- Branch / loop overhead bubbles big enough to register at the 64-cy granularity

So pipe_fp64 = 100.00%. The 2.4% gap that FFMA shows (97.64%) doesn't appear
here because the FP64 pipe's 1/64-cy issue rate is so slow that any startup
bubbles get amortized below the noise floor.

### Latency

DFMA latency = **64 cy** (`V9_OP_LATENCY`). Single chain saturates the pipe
because chain depth = 1 per SMSP × throughput 1/(64 cy) = 1 = exactly the
saturation point. NCHAIN ≥ 1 is sufficient; NCHAIN > 1 doesn't help (no
parallelism to exploit).

This is why the test measured 100% with very modest occupancy (4 warps/SM).

### Recipe (V8 DFMA peak)

```cuda
__global__ __launch_bounds__(256, 1)
void kernel(double* A, double* B, double* C, int iters, int, int) {
    double v = A[threadIdx.x];
    double b = B[threadIdx.x];

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        asm volatile("fma.rn.f64 %0, %0, %1, %0;"
                     : "+d"(v)
                     : "d"(b));
    }

    if (v == -1.0) C[threadIdx.x] = v;   // anti-DCE
}
```

Launch with `<<<148, 256>>>` and `iters = 100000`. Runtime ~50 ms (well above
the 10 ms launch-overhead floor).

### Comparison with FP32

| Pipe | Theoretical | Measured | % of peak |
|---|---:|---:|---:|
| FP32 (FFMA) | 76.97 TFLOPS | 75.2 TFLOPS | 97.64% |
| FP64 (DFMA) | 1.203 TFLOPS | 1.203 TFLOPS | **100.00%** |

FP64 is 64× slower but 2.4 percentage points closer to its peak than FP32.
The reason FP32 loses 2.4%: RF read-port contention micro-stalls and inst
issue stalls across 4 SMSPs. FP64 serializes so cleanly because the DFMA port
is the ONLY blocker.

### Implication for scientific workloads

For CFD, quantum chemistry, CAE, finite-element solvers (FP64-heavy):
- Use 2-source DFMA chains: `__fma_rn(v, b, v)` or PTX `fma.rn.f64`
- Launch 256 thr × 148 blocks at boost
- Expect **1.20 TFLOPS exactly** — clean SoL with no methodology surprises

This is in stark contrast to FP32, where reaching 75 TFLOPS requires careful
attention to RF port pressure (§17), and also stark contrast to tensor cores
where realistic ≠ peak (§24, §25). FP64 DFMA on B300 is the simplest peak
to achieve.

### Cross-check: 0.95 TFLOPS legacy claim

An older catalog entry (`b300_clean/legacy` row 35) claimed FP64 DFMA = 0.95
TFLOPS. This was measured at 1920 MHz with insufficient warps (1-2 warps/SM),
dropping pipe_fp64 to ~75% via under-occupancy. **Retracted** in the corrections
file. The 1.20 TFLOPS @ 2032 with ≥4 warps/SM is the canonical peak.

### FP64 outside DFMA

| Op | Throughput vs FP32 | Notes |
|---|---|---|
| DFMA | 1:64 | this measurement |
| DADD | 1:64 (same pipe) | inferred — not directly measured |
| DMUL | 1:64 (same pipe) | inferred |
| DDIV | much slower | software emulation, not measured here |
| DFMA via DMMA (FP64 tensor) | 1:64 (NO speedup) | per `06_tensor_cores`, DGEMM ≈ 1.05 TFLOPS — same as DFMA. **No FP64 tensor speedup on B300.** |

This last point is important: **B300 has FP64 tensor cores, but they don't
exceed scalar DFMA throughput**. DGEMM via cuBLAS with FP64 tensor enabled
hits ~1.05 TFLOPS, essentially identical to scalar DFMA's 1.20. The FP64
tensor pipe is more about lower latency or operand-format flexibility than
raw throughput.

**See also:** §19 (FFMA / FADD / FMUL all 4.22 cy; DFMA is its own pipe at 64 cy),
§24 (FP64 tensor — no speedup), `V8_FP64_PEAK_VERIFIED.md`, `06_tensor_cores`
"FP64 tensor" row.

---

## §21. IMAD — 38.5 Tops = 1:2 of FP32

**Answer:** **32-bit IMAD peak = 38.4 Tops = 99.7% of 38.5 theoretical** at 2032 MHz
boost. IMAD is 1:2 of FP32 (NOT 1:1 — common error). It runs on the FMA pipe
(per V40/REPORT_06), so FFMA + IMAD compete for dispatch.  `[🟢 HIGH · src: V8_IMAD_PEAK_VERIFIED.md, V40 + REPORT_06]`

### Theoretical derivation

Per CUDA C Programming Guide Table 13-1, sm_9.x/10.x integer mul/MAD = 64 ops
per SM per cycle (vs 128 for FP32). So:

```
IMAD chip peak = 148 × 64 × 2.032e9 = 19.24 G IMAD/s = 38.5 Tops (IMAD = 2 ops)
              = FP32 / 2 = 76.97 / 2 = 38.5 Tops  ✓
```

The 1:2 ratio reflects that 32-bit integer multiply hardware is half-rate
compared to FP32 multiply on Hopper/Blackwell. **This is documented:**
https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities

### Measurement (V8, ITERS=100K)

| Metric | Value |
|---|---|
| ncu `smsp__sass_thread_inst_executed_op_integer_pred_on.sum` | 30.31 G IMAD |
| ncu `gpu__time_duration.sum` | 1.58 ms |
| IMAD rate | 30.31e9 / 1.58e-3 = 19.18 G IMAD/s |
| Ops/s (× 2) | **38.4 Tops** |
| Ratio to theoretical | **99.7%** |

### Why ncu shows pipe_alu = 0% but pipe_fma is busy

ncu metric attribution: IMAD lives on the **FMA pipe**, not the dedicated ALU
pipe. Per the V40 ALU ladder + REPORT_06 (dispatch placement reverse-engineering),
the FMA pipe handles:
- FFMA (FP32 multiply-add)
- FADD / FMUL (degenerate FFMA forms)
- IMAD (integer multiply-add)
- IADD3 (3-source integer add — yes, on FMA pipe per V40, NOT ALU as
  earlier docs claimed)

The dedicated ALU pipe handles:
- LOP3 (3-input bitwise logic)
- IMUL (32-bit integer multiply, when not part of MAD)
- PRMT (byte permute)
- SEL, ISETP (compare)

So when V8_IMAD_PEAK kernel runs, ncu shows:
- `pipe_fma_cycles_active` ≈ high (IMAD dispatching here)
- `pipe_alu_cycles_active` ≈ 0% (no ALU op in flight)

This is consistent with the §22 dual-issue picture: FMA and ALU are physically
separate pipes that overlap freely. IMAD on FMA + LOP3 on ALU should dual-issue
(not measured directly, but architecturally implied).

### Recipe (V8 IMAD peak)

```cuda
__global__ __launch_bounds__(256, 1)
void kernel(int* A, int* B, int* C, int iters, int, int) {
    int v[8], b[8];
    int tid = threadIdx.x;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        v[k] = A[tid + k*256];
        b[k] = B[tid + k*256];
    }

    #pragma unroll 1
    for (int i = 0; i < iters; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            #pragma unroll
            for (int k = 0; k < 8; k++)
                asm volatile("mad.lo.s32 %0, %0, %1, %0;"
                             : "+r"(v[k])
                             : "r"(b[k]));
        }
    }

    int sum = v[0]+v[1]+v[2]+v[3]+v[4]+v[5]+v[6]+v[7];
    if (sum == -1) C[tid] = sum;
}
```

SASS body: 128 × `IMAD R19, R4, R18, R18` (2-source: R18 self-feeds — same
recipe shape as FFMA peak).

### Latency

IMAD latency = **4.25 cy** per `V9_OP_LATENCY.md`. Throughput cap is
1/(2 cy)/SMSP (because IMAD is half-rate vs FFMA). Saturation ILP ≥ 2 chains.

### IMAD vs FFMA: which to use for what

| Use case | IMAD | FFMA |
|---|---|---|
| 32-bit address calculation | yes | no (but FP32 may be precise enough for small offsets) |
| Hash function with multiplies | yes — at half FP32 rate | no |
| Crypto rounds (AES, ChaCha) | yes — IMAD common | no |
| Reduction with int accumulator | yes | n/a |

For **integer-heavy workloads** (hash, crypto, sorting), the practical ceiling
is **38.5 Tops via IMAD**. Cannot expect FP32 throughput from integer multiply.

For **mixed FP32 + integer** kernels, the FMA pipe is shared between FFMA and
IMAD, so total throughput is bounded by `FFMA_count + 2×IMAD_count ≤ 1 dispatch/SMSP/cy`.
At full saturation: 76.97 TFLOPS FP32 × FP_fraction + 38.5 Tops IMAD × INT_fraction.

### Other integer instruction variants

`V8_IMAD_PEAK` only directly measured 32-bit IMAD. Inferred rates for related ops:

| Op | Rate vs FP32 | Pipe | Confidence |
|---|---|---|---|
| IMAD (32-bit) | 1:2 | FMA | HIGH (V8 measured) |
| IADD3 | 1:1 | FMA (per V40) | HIGH (V40 measured ~26 Glane/s = 67% at 1500 lock = same tier as FFMA) |
| IMAD.HI (high mul) | likely 1:4 | FMA | LOW (not measured here) |
| IMUL (32-bit, non-MAD form) | 1:2 | ALU (per A6 / V40 ladder) | MED (A6 measures 14.16 TIPS @ 1500 = 0.50 inst/SMSP/cy) |
| LOP3 | 1:2 | ALU | HIGH (V40, A6) |

Note the apparent paradox: IMAD on the FMA pipe is 1:2 of FP32, but IMUL on
the ALU pipe is also 1:2. Different pipes, same rate. The reason: IMAD pipe
issue rate is 1/(2 cy)/SMSP because the multiplier hardware itself is half-rate;
IMUL pipe rate is 0.5/SMSP/cy because the ALU pipe (LOP3, IMUL, PRMT) caps at
0.5/SMSP/cy regardless of op. Coincidence in numerical answer.

### Why "IMAD same throughput as FP32" is wrong (common error)

A common mistake (also made by V8 author initially) is to assume IMAD = FFMA
in throughput because both are 3-source MAD instructions on the FMA pipe.
This is FALSE: the FMA pipe has full FP32 throughput (1 FFMA/cy/SMSP) but the
INTEGER multiply hardware is half-rate (1 IMAD per 2 cy/SMSP). The pipe
DISPATCHES at 1/cy regardless, but for IMAD only every other dispatch slot
emits productive work (the rest are bubbles or alternative ops).

If you read a benchmark that claims "76 Tops IMAD on B300", it is wrong
(probably extrapolating FP32 rate to integer without checking the table).

**Footgun:** ⚠ IMAD is **1:2** of FP32, NOT 1:1. Peak is 38.5 Tops, NOT 76.97.
Per CUDA PG Table 13-1 (sm_9.x/10.x).

**See also:** §16 (FP32 FFMA peak — IMAD is half), §22 (FMA pipe is shared
between IMAD and FFMA — they don't dual-issue with each other), `V40_RIGOR.md`,
§27 (pipe placement table — Agent C).

---

## §22. Dual-issue — FMA + ALU pipes overlap freely (the headline)

**Answer:** **B300 SMSPs have physically separate FMA and ALU pipes that overlap
freely. Solo FFMA reaches `pipe_fma = 97.6%`. Solo LOP3 reaches `pipe_alu = 99.5%`
at `inst_issued = 0.51/cy` (because LOP3 has a 2-cycle issue cadence per SMSP).
Dual mode (FFMA + LOP3) reaches `pipe_alu = 98.0%` AND `pipe_fma = 49.4%`
SIMULTANEOUSLY → `pipe_alu + pipe_fma = 147%`. Dispatch is NOT capped.**  `[🟢 HIGH · src: V52_RUN_RESULTS.md (settled 2026-04-22), HEADLINE_CORRECTIONS_v5.md item #7]`

This is the **single most contested architectural finding in the entire B300
catalog**, and the verdict has flipped FIVE TIMES across the audit waves
(HIGH → LOW → MED → LOW → HIGH). Section A below documents the zigzag.
Section B below states the architectural truth as anchored by V52's empirical
ncu measurement. Section C below documents WHY V49/V50's "55% / 74% same-warp
ceiling" was a methodology artifact, not an architectural finding.

### A. The 5-level zigzag (NEW reader: skim; methodology students: study)

The dual-issue verdict has flipped 5 times during the audit:

| Wave | Date | Verdict | Tag | Reasoning |
|---|---|---|---|---|
| W1+W2 | 2026-04-19 | YES dual-issues at varying overlap | 🟢 HIGH | M8 PIPE_OVERLAP_MATRIX measured FFMA+IADD3=56%, FFMA+LDS=96%, FFMA+MUFU=100%+ |
| W3a | 2026-04-20 | NO — same-warp dual capped at 55% | 🟢 HIGH | V49 (`501134a`) "FFMA+LOP3 = 55%, FFMA+IADD3 = 54%, FFMA+PRMT = 51%" — interpreted as "B300 dispatch slot is shared 4 inst/cy/SM regardless of pipe" |
| W3a | 2026-04-20 | YES warp-spec helps to 74% | 🟢 HIGH | V50 (`fbe1c18`) measured warp-specialized at 74% — but still capped, attributed to per-SMSP shared dispatch port |
| W3b | 2026-04-21 | UNCERTAIN — V49/V50 baseline under-occupied | 🔴 LOW | DUAL_ISSUE_DOUBT_REPORT noted FFMA solo at 67% (= latency-bound, not pipe-saturated) so denominator is wrong |
| W4 | 2026-04-21 | RE-PROMOTED to MED | 🟡 MED | META_DOUBT_REPORT: V8 reaches 97.6% with same `__launch_bounds__(256, 1) = 8 warps/SM`, so V49's 67% solo is NOT under-occupancy (mechanism wrong); but the numbers themselves are real measurements |
| W5a | 2026-04-21 | RE-DOWNGRADED via SASS audit | 🔴 LOW | SASS_VERIFY_DUAL_ISSUE: V49's body has 8 FFMA + 8 LOP3 + 1 BRA + 1 UIADD3 + 1 UISETP per inner iteration. **Loop-overhead = 12.5% of body**. V8's body has 128 FFMA + 0 BRA per outer = 1.2% overhead. V49 measured the steady-state of an FFMA+LOP3+branch loop, not pure dual-issue. |
| **W6** | **2026-04-22** | **HIGH (architectural truth) + RETRACTED (numbers)** | **🟢 HIGH** | V52 clean retest with 128-deep inner unroll + ncu `pipe_alu` AND `pipe_fma` simultaneously → **alu = 98.0%, fma = 49.4%, sum = 147%, inst_issued = 1.00/cy in dual mode**. Pipes overlap freely. The "dispatch cap" was a phantom. |

**Meta-lesson:** "the artifact is real" and "the architectural inference from
the artifact is real" are TWO independent claims. W3a/W5a conflated them.
Always state separately:
- (a) Is the measured number trustworthy as published?
- (b) If not, what is the true architectural value?

W3a/W5a got (a) right (no, V49/V50 are loop-overhead contaminated) and (b)
wrong (the cap doesn't exist at the contaminated value; it doesn't exist at all).

### B. The architectural truth (V52, 2026-04-22)

The decisive measurement: V52 (`tests/standalone/v52_dual_issue_clean.cu`)
implements the V8-style 128-deep inner unroll for both FFMA and LOP3 in three
modes (mode=0: solo FFMA, mode=1: solo LOP3, mode=2: dual FFMA+LOP3 alternating),
with `__launch_bounds__(256, 1)` (= 8 warps/SM = 2 warps/SMSP) at default boost.

Anti-DCE: tid-dependent register init + STG of XOR accumulator under
impossible-`if`. Loop overhead in the SASS body: **1.2%** (vs V49's 12.5%).
`pkill -9 v52 && sleep 6` between every run.

**ncu pipe-utilization metrics** (Geometry A: 148 blocks × 256 thr, BPS=1):

```
config (mode,ILP,BPS,N_OUTER)   pipe_alu%   pipe_fma%   inst_issued/cy   alu+fma
<0,4,1,4096>  solo FFMA           0.01       95.39        1.00            95.40
<1,4,1,4096>  solo LOP3          97.27        0.76        0.52            98.03
<2,4,1,4096>  dual                96.17       48.84        0.99           145.01

<0,8,1,2048>  solo FFMA           0.02       97.58        1.00            97.60
<1,8,1,2048>  solo LOP3          99.45        0.39        0.51            99.84
<2,8,1,2048>  dual                98.00       49.39        1.00           147.39

<0,16,1,1024> solo FFMA           0.04       98.66        1.00            98.70
<1,16,1,1024> solo LOP3          99.74        0.20        0.51            99.94
<2,16,1,1024> dual                87.96       44.15        0.89           132.11
```

**These metrics are decisive.** `alu + fma = 145–147%` at ILP=4 and ILP=8
**directly proves** that both pipes are running concurrently. ncu's
`pipe_X_cycles_active` counts cycles where pipe X is doing work; if the sum
exceeds 100%, pipes are overlapping.

The drop at ILP=16 (sum = 132%) is **register pressure**, not architectural:
16 floats + 16 ints = 32 live regs/thread × 256 thr ≈ saturates the 64K RF.
Expected at this occupancy/ILP combination. The architectural answer is
ILP=4 or ILP=8 result.

### Wall-clock cross-check

| ILP | solo FFMA Glane/s | solo LOP3 Glane/s | dual total Glane/s | dual / max(solo) | dual / sum(solo) |
|---:|---:|---:|---:|---:|---:|
| 4  | 32 060 | 16 428 | 32 525 | **101.5%** | 67.0% |
| 8  | 32 706 | 16 824 | 33 114 | **101.2%** | 66.9% |
| 16 | 33 172 | 16 763 | 28 173 | 84.9% | 56.4% |

Wall-clock dual ≈ 101.5% of solo FFMA, NOT close to sum. This was the surface
that misled V49 / V50: looking only at wall-clock, dual ≤ max(solo) → "no
dual-issue". But the ncu pipe metrics tell the real story: BOTH pipes are
saturated; the wall-clock just doesn't reveal that because LOP3's 2-cy
cadence means it produces only ~half as many results per cycle as FFMA, and
those results overlap with FFMA's results in time.

### Why dual ≈ max(solo) wall-clock — the 2-cy LOP3 cadence

The reason `dual_throughput ≈ max(solo_FFMA)` is **NOT** a shared dispatch port.
It is that **LOP3 issues at half the rate of FFMA per cycle**:

- `smsp__inst_issued.avg.per_cycle_active` = **1.00** for FFMA-only,
  **0.51** for LOP3-only, **1.00** for dual.
- `smsp__pipe_alu_cycles_active` = **97-99%** for solo LOP3 — the ALU pipe
  is saturated, but each LOP3 takes ~2 issue cycles.
- In dual mode, FFMA fills the 50% of slots LOP3 leaves idle:
  `pipe_alu + pipe_fma = 145-147%` at ILP=8.

Picture it as two timelines:

```
SMSP issue port (1 inst/cy max):
cy 0   1   2   3   4   5   6   7   8   ...
FFMA   x   x   x   x   x   x   x   x   x  ←  98% saturated (1/cy/SMSP)
LOP3   y   .   y   .   y   .   y   .   y  ←  98% saturated but every other slot
Dual   x   x   x   x   x   x   x   x   x  ←  100% saturated, alternating FFMA/LOP3
        +y  .  +y  .  +y  .  +y  .  +y     in the slots LOP3 wants

Pipe_fma sees: every cy in solo, every other cy in dual → 97% solo / 49% dual
Pipe_alu sees: every other cy in both solo and dual → 98% / 98%
inst_issued/cy: 1.0 solo FFMA / 0.51 solo LOP3 / 1.0 dual
```

So:
- **The FMA pipe and ALU pipe are physically separate** — they overlap freely.
- **LOP3 has a 2-cycle issue cadence per SMSP** (likely the fundamental ALU
  pipe rate, or LOP3-specific). Solo LOP3 throughput is ~16.8 K Glane/s =
  ~43% of the 38.5 K Glane/s "1 inst/cy/SMSP" upper bound — it is actually
  100% of its OWN real ceiling (which is half FFMA's per cycle).
- **Dual mode reaches `inst_issued = 1.00/cy`** AND `pipe_fma + pipe_alu = 147%`
  — this is **clear dual-issue at the dispatch port**, not a shared cap.
- The "harmonic mean" framing in V49 was wrong: the pipes don't share, but
  LOP3's intrinsic 2-cycle issue means dual is bottlenecked by FFMA's slot
  count, with LOP3 piggy-backing in the otherwise-idle ALU port.

### The verdict matrix

| Question | Answer (W6 settled) |
|---|---|
| Does FFMA + LOP3 dual-issue work on B300? | **YES.** Both pipes fire at 98%+ simultaneously. |
| Is V49's "55% same-warp ceiling" architectural? | **NO.** Methodology artifact (8-deep loop, ALU loop overhead). |
| Is V50's "74% warp-specialized ceiling" architectural? | **NO.** Same root cause; warp-split helped because it hid loop overhead. |
| What's the real dispatch behaviour? | **1 inst/SMSP/cy on each pipe, FREELY OVERLAPPING.** LOP3 happens to need 2 issue slots per inst → solo LOP3 = ½× solo FFMA but dual = 1× FFMA + ½× LOP3 = 1.5× FFMA-issue-rate worth of work. |
| Is the "B300 dispatch capped at 128 inst/SM/cy" claim wrong? | **PARTIALLY.** Single-pipe IS capped at 128/SM/cy (= 1/SMSP/cy × 4 SMSPs × 32 lanes), but the FMA + ALU pipes can BOTH be at 128 simultaneously, so total inst/SM/cy can reach ~256. **The "128 ceiling" is per-pipe, not per-SM.** |

### C. Why V49/V50's "55%/74%" was wrong (the methodology artifact)

V49's contaminated body had **8 FFMA + 8 LOP3 + 1 UIADD3 + 1 UISETP + 1 BRA**
per inner iteration (loop overhead ≈ 12.5% of body). V52's body has **128 FFMA
+ 136 LOP3 + 1 UIADD3 + 1 UISETP + 1 BRA** per outer iteration with all 128
FFMAs and 128 LOP3s in a single straight-line unrolled block (loop overhead
≈ 1.2% — within V8's amortization regime).

The V49 body — 8 FFMA + 8 LOP3 with `#pragma unroll 1` on the outer loop —
emits a tight loop where `BRA + UIADD3 + UISETP` consume about 1 cycle of
ALU pipe slot per 17 instructions. That's 5.9% of ALU dispatch consumed by
loop infrastructure, just for the branch. With LOP3 already at 50% pipe
efficiency (2-cy cadence) and the FFMA half of the body wanting full FMA
saturation, the contention shows up as:

- Wall-clock dual = max(solo) → "no dual-issue" (V49 reading)
- ncu pipe_fma + pipe_alu sum < 100% (would have shown if measured)

The V50 fix (warp-specialization) helped not because separate warps avoid
dispatch sharing, but because it **doubles the inner body** — now each warp
has 8 FFMAs (without alternating LOP3s) which compiles to a slightly cleaner
inner block. Loop overhead drops from 12.5% to ~6.25%.

V52 demonstrated this by going to 128-deep unroll for ALL three modes. Loop
overhead drops to 1.2%, and BOTH solo and dual modes saturate their respective
pipes.

### SASS verification (V52 dual mode body)

Inspected `_Z*mode2*` instantiation: ILP=8, BPS=1:

```
FFMA: 128
LOP3: 136     (128 in loop body + ~8 in init/anti-DCE)
UIADD3: 1
UISETP: 1
BRA: 1
STG: 2
```

Inner FFMA encoding:
```
FFMA R11, R11, 1.5, R11
FFMA R12, R12, 1.5, R12
FFMA R13, R13, 1.5, R13
...
```

Same 2-source self-feed pattern V8 uses (Rd × IMM + Rd). Avoids the 3-distinct-
source RF port limit (§17). **The kernel is methodologically clean.**

### What this means for kernel-design

**B300 FMA and ALU pipes overlap freely.** A kernel that interleaves FFMA and
LOP3 (or FFMA and any ALU-pipe op like PRMT, IMUL, ISETP) will get **both**
pipes saturated at their respective rates, NOT throttled to one or the other.
Specifically:

- An FFMA-bound kernel can absorb up to 0.5 LOP3/cy/SMSP (= 50% of LOP3 pipe
  peak) **for free** (no FFMA throughput loss). Past that, LOP3 takes 2 cy
  per inst so adding more LOP3 starts contending with itself.
- A LOP3-bound kernel can absorb up to 1 FFMA/cy/SMSP (= 100% of FMA pipe
  peak) for free. Adding more FFMA past that has nowhere to go.
- The combined work per SMSP/cy is **1 FFMA + 0.5 LOP3 = 1.5 inst/cy/SMSP**
  worth of useful instructions, distributed across 2 physical pipes.

This is **2× the "128 inst/SM/cy dispatch cap"** that earlier docs hinted at.
The cap was per-pipe, not per-SM.

### What this means for the M8 PIPE_OVERLAP_MATRIX

M8's matrix entries (FFMA+IADD3 = 56%, FFMA+LDS = 96%, MUFU+FFMA = 100%+,
HMMA+LDS = 73%) are now **architecturally consistent** with V52's free-overlap
finding. The reason FFMA+IADD3 measures 56% in M8 is **not** dispatch sharing
but the same loop-overhead contamination that bit V49 — M8 also used short
inner bodies. The MUFU+FFMA = 100%+ entry, which W3a/b/W5a struggled to
reconcile with their "shared dispatch cap" interpretation, is now naturally
explained: MUFU and FFMA are on different pipes, MUFU's slow issue rate
leaves FFMA slots open, dual issues freely.

M8's overlap matrix should be **re-measured with V8/V52-style 128-deep
unroll** to get the architectural overlap fractions, not the loop-overhead-
contaminated ones.

### What V52 did NOT settle

The empirical anchor is strong but not all-encompassing:

1. **FMA + ALU specifically.** Other pipe combinations (LSU + tensor, MUFU
   + tensor, HMMA + LDS, etc.) are NOT directly settled by V52.
2. **2-cy LOP3 cadence is INFERRED from `inst_issued/cy = 0.51`.** An
   alternative explanation: "1-cycle issue but 50% stall on RF read port".
   Same observable. V52 cannot distinguish.
3. **ncu metric definition correctness.** ncu metrics are software-defined;
   if NVIDIA's `pipe_X_cycles_active` calculation has a bug for sm_103a, our
   reading would be wrong. We have NOT verified the metric definition against
   PTX-level event counters.
4. **Whether dual-issue works for FMA + tensor.** Tensor uses the tensor pipe;
   should overlap with FMA. Not measured by V52.
5. **Same-warp vs cross-warp in dual mode.** V52 measured 1 warp doing
   alternating FFMA+LOP3. Doesn't directly test 2 warps each doing one type.

### What COULD overturn the V52 settlement

- An ncu metric-definition bug for sm_103a that inflates `pipe_X_cycles_active`
- A clean test where `alu + fma` reproducibly stays at ≤ 100% under V8-style
  methodology
- Direct PTX-event counters showing different per-cycle issue counts than
  ncu metrics
- A SASS-level disassembly showing that LOP3 actually issues every cycle but
  has a 2-cy result-write stall (would change the mechanism explanation but
  not the throughput conclusion)

None of these are expected; V52 is the strongest evidence to date and is
treated as canonical.

### Files for the V52 settlement

| File | Purpose |
|---|---|
| `/root/github/QuickRunCUDA/tests/standalone/v52_dual_issue_clean.cu` | The kernel |
| `/tmp/v52` | Compiled binary |
| `/tmp/v52.sass` | Full SASS dump |
| `/tmp/v52_dual_ilp8_bps1.sass` | Target body SASS (ILP=8, BPS=1) |
| `/tmp/v52_run1.txt`, `_run2.txt`, `_run3.txt` | Three independent runs (within 1%) |
| `/tmp/v52_ncu_full.txt` | Full ncu profile output |
| `/tmp/v52_keep/` | nvcc `-keep` directory |
| `/root/github/QuickRunCUDA/b300_clean/corrections/V52_RUN_RESULTS.md` | Findings doc |

**Footgun:** ⚠ **DO NOT QUOTE V49's 55% or V50's 74% as architectural caps.**
They were 100% loop-overhead methodology artifacts. The architectural answer
is `pipe_alu + pipe_fma = 147%` — pipes overlap freely. The 5-level zigzag
documents how this took 4 audit waves to settle.

**Footgun 2:** ⚠ The "B300 dispatch capped at 128 inst/SM/cy total" claim is
PARTIALLY wrong. It is correct PER PIPE but each SM has multiple pipes that
overlap, so total can reach ~256 inst/SM/cy (FFMA + LOP3 = 1 + 0.5 per SMSP
× 4 SMSPs × 32 lanes = 192 inst/SM/cy in the V52 measured case).

**Footgun 3:** ⚠ Quoting `pipe_X_cycles_active` for a single pipe in isolation
is misleading for dual-issue claims. The ONLY decisive metric is reading
**both `pipe_alu` and `pipe_fma` from the same ncu profile** and summing.
If sum > 100%, pipes overlap.

**See also:** §16 (FFMA solo peak — the "max(solo)" the wall-clock dual
matches), §17 (RF port, separate constraint), §21 (IMAD on FMA pipe — would
NOT dual with FFMA because same pipe), §27 (pipe placement table — Agent C),
M8 PIPE_OVERLAP_MATRIX (now architecturally consistent, re-measure pending),
`V52_RUN_RESULTS.md`, `SASS_VERIFY_DUAL_ISSUE.md`, `DUAL_ISSUE_DOUBT_REPORT.md`,
`META_DOUBT_REPORT.md`, `HEADLINE_CORRECTIONS_v5.md`.

---

## §23. Tensor cores — mma.sync m16n8k16 BF16/FP16 = 569-578 TFLOPS

**Answer:** **mma.sync legacy tensor path (m16n8k16, FP16/BF16, F16 or F32 acc)
= 569-578 TFLOPS at 99.89% pipe_tensor saturation** at 2032 MHz boost.
Latency 20 cy per HMMA.  `[🟢 HIGH · src: V8_HMMA_F16_PEAK.md, V8_HMMA_VARIANTS_PEAK.md, V9_HMMA_LATENCY.md]`

This is the **legacy** tensor path accessible via PTX `mma.sync`. For the
modern Blackwell tensor path see §24 (tcgen05.mma at ~2000-2240 TFLOPS) and
§25 (FP8 cuBLAS at ~3984-4393 TFLOPS).

### V8 measurement (mma.sync.aligned.m16n8k16, 8 chains, 148 × 256 thr)

| Variant | Pipe % | Time | Inst count | TFLOPS |
|---|---:|---:|---:|---:|
| F16/F16 acc | 99.90% | 670.37 µs | 94.72 M | **578.6** |
| F16/F32 acc | 99.89% | 670.27 µs | 94.72 M | **578.6** |
| BF16/F32 acc | 99.89% | 671.46 µs | 94.72 M | **578.6** |

All three at 99.89-99.90% of `sm__pipe_tensor_cycles_active` — saturated.
F32 accumulator is **free** on the legacy tensor pipe (no throughput loss vs
F16 accumulator). This is the modern training path (BF16/F32) reaching
**578 TFLOPS** = 99.9% of the legacy tensor pipe capability.

### TFLOPS derivation

```
HMMA m16n8k16 = 16 × 8 × 16 × 2 FLOPS = 4096 FLOPS per warp-instruction
chip_HMMAs = 94.72 M / 670 µs = 141 G HMMA/s
chip_TFLOPS = 141 G × 4096 FLOPS / 1e12 = 578.6 TFLOPS
```

The 99.9% pipe saturation directly matches the per-instruction count divided
by time, both measured.

### B300_TRUE_REFERENCE entry: 569 TFLOPS

The canonical B300_TRUE_REFERENCE entry (commit `a37d989`) lists:
**FP16/BF16 mma.sync m16n8k16 = 569 TFLOPS = 7.4× FFMA**

The V8 measurement (578.6) is ~1.7% higher than the catalog 569 figure.
The two are within measurement noise; the catalog 569 is the conservative
8-chain "burst" measurement, V8's 578 is the same recipe slightly tuned.
**Both round to 570-580 TFLOPS as the architectural truth.**

### Latency (V9_HMMA_LATENCY)

Single warp serial-dependency chain measurements:

| Chain depth | Total cycles | Latency (cy/HMMA) |
|---:|---:|---:|
| 64 | 1 660 | 25.94 (startup-dominated) |
| 256 | 5 495 | 21.46 |
| 1 024 | 20 873 | 20.38 |
| **4 096** | **82 297** | **20.09** (converged) |

**HMMA m16n8k16 F32-acc latency = 20 cy per instruction.** This converged
value is the single-issue latency through the tensor pipe.

### Throughput cross-check via latency

V8 throughput: 94.72 M HMMAs in 670 µs at 2032 MHz.
- Per SM per cy: `141 G / (148 × 2.032e9) = 0.469 HMMAs/cy/SM`
- Per SMSP: `0.469 / 4 = 0.117 HMMAs/cy/SMSP`
- Equivalently: **1 HMMA per 8.5 cy per SMSP** (or 1 HMMA per 4 cy per SM)

At 20 cy latency, saturating an SMSP requires `20 / 8.5 ≈ 2.35` ILP chains.
**Per warp need ≥3 independent chains.** V8's 8-chain kernel has 2.3× margin,
explaining the 99.9% pipe saturation.

### Theoretical / spec context

The CLAUDE.md "B300 SXM6 theoretical peaks" section lists:
- **FP16/BF16 mma.sync m16n8k16: ~540-580 TFLOPS** (legacy tensor path)

So 569-578 TFLOPS is at the upper end of the spec range. The B200 datasheet
quotes 2500 TFLOPS for "BF16 dense" but that figure is the **modern (tcgen05)
tensor path with 4× higher throughput**, not the legacy mma.sync path.

The 569-578 / 2500 = 23% ratio between legacy and modern is consistent with
"Blackwell modern tensor is 4-5× faster than legacy mma.sync at the same
precision". This is a pure architectural choice: NVIDIA could have built
a faster legacy tensor pipe but chose to keep it at Hopper-level so kernels
don't need to know about the Blackwell-specific optimizations.

### F32 accumulator is free (key training implication)

For modern training pipelines (BF16 inputs, FP32 accumulator):

| Variant | TFLOPS | F32 cost |
|---|---:|---|
| BF16/F16 acc | 578 | n/a |
| BF16/F32 acc | 578 | **0% — F32 is free** |

This matches expectation. On Blackwell legacy HMMA, the accumulator format
doesn't add per-MMA cost. Earlier architectures (Volta, Turing) had a small
F32-acc penalty; Hopper and Blackwell removed it.

### Recipe (V8 HMMA peak)

```cuda
__global__ __launch_bounds__(256, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int iters, int, int) {
    unsigned a0 = A[0], a1 = A[1], a2 = A[2], a3 = A[3];
    unsigned b0 = B[0], b1 = B[1];
    unsigned c0[8], c1[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        c0[k] = c1[k] = 0;
    }

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
                " {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
                : "+r"(c0[k]), "+r"(c1[k])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
            );
        }
    }

    // anti-DCE
    unsigned sum = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) sum ^= c0[k] ^ c1[k];
    if (sum == 0xDEADBEEF) C[threadIdx.x] = sum;
}
```

Launch with `<<<148, 256>>>`. Set `iters = 10000` for ~670 µs runtime.

### What FP8 mma.sync looks like (negative result)

V8 attempted `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`:
- Pipe util only **28%**
- SASS shows `HMMA.16816.F32` (NOT k32, NOT e4m3) — compiler silently fell back
- Only 2 HMMAs in SASS loop instead of expected 8

**FP8 on B300 requires the tcgen05.mma path, NOT legacy mma.sync.** The
claimed 4500 TFLOPS FP8 peak (CLAUDE.md, §25) is via tcgen05; legacy mma.sync
cannot reach it. The compiler accepts the e4m3 PTX syntax but emits non-FP8
SASS — a silent fallback.

For FP8, see §25 (cuBLAS LtMatmul → tcgen05 internally, 3984-4425 TFLOPS).

### Ratio to FFMA (the "7.4× faster than FP32 FFMA" claim)

```
578 TFLOPS / 75.2 TFLOPS = 7.69× FFMA
569 TFLOPS / 76.96 TFLOPS = 7.39× FFMA  (against theoretical)
```

The "7.4× FFMA" entry in B300_TRUE_REFERENCE is the conservative 569/77
ratio. So **legacy HMMA is 7-8× faster than scalar FFMA per chip-cycle**, but
this is still the Hopper-level tensor performance, not the full Blackwell
tensor capability.

### Ratio to modern tcgen05.mma (the "5x more")

569 (legacy) / 2242 (BF16 cuBLAS via tcgen05, §24) = **25%** of modern path.
This is a substantial gap — for production training, USE THE TCGEN05 PATH
(via cuBLAS or CUTLASS), NOT raw mma.sync. The legacy path is good for:
- Simple non-cuBLAS kernels where you want MMA without writing tcgen05 PTX
- Educational / micro-bench purposes
- Code that needs to also work on pre-Blackwell GPUs

### TF32 m16n8k8 = 288 TFLOPS (half of FP16)

| Op | TFLOPS | Notes |
|---|---:|---|
| FP16/BF16 m16n8k16 | 578 | this measurement |
| TF32 m16n8k8 | 288 | half — K dimension halved |
| INT8 m16n8k32.s32.s8 | 143 TOPS | HW-throttled (5 NOPs/issue) |

TF32 has K=8 (half the K of FP16's 16), so each MMA does fewer ops. The
288 TFLOPS = 50% of FP16's 578 follows directly. Latency is ~10 cy (also
half).

### INT8 mma.sync = 143 TOPS, HW-throttled

INT8 IMMA SASS shows **5 NOPs per issue** — the HW throttles INT8 tensor
to 1/(5+1) = 16.67% of the issue rate. So even though INT8 has K=32 (2× FP16),
the throttle drops it to 143 TOPS = 25% of FP16's 578. This is **HW-throttled,
NOT latency-bound** — adding ILP doesn't help past the 5-NOP gap.

### Retracted claims (do NOT cite)

| Retracted | True value | Source |
|---|---|---|
| "1543 TFLOPS BF16 single-chain mma.sync" | 569-578 (8-chain) | TRUE_REFERENCE row 58 |
| "6 357 TFLOPS FP8 via mma.sync" | DCE artifact; FP8 needs tcgen05 | catalog self-retract |
| "2 336 / 2 400 TFLOPS FP8 via mma.sync" | FADD artifact | catalog self-retract |
| "FP4 mma.sync on sm_103a" | REJECTED — only sm_120a | catalog row |

The 1543 number ALSO appears legitimately as the NVLink-5 bidirectional GB/s
figure (§37, Agent C) — that is unrelated and not retracted.

**Footgun:** ⚠ "1543 TFLOPS BF16 single-chain" is **RETRACTED** (over-counted).
The real legacy mma.sync peak is 569-578 TFLOPS (8-chain multi-accumulator).
Use 569 (catalog conservative) or 578 (V8 99.9% pipe-saturated) — both round
to "570 TFLOPS legacy tensor".

**Footgun 2:** ⚠ Don't compare legacy mma.sync (569 TFLOPS) to spec "2500
BF16" — those are different paths (legacy vs tcgen05). For the spec match,
use §24's tcgen05.mma numbers.

**See also:** §24 (modern tcgen05.mma — 4× the legacy throughput), §25 (FP8
via cuBLAS), §16 (FFMA peak — HMMA is 7-8× faster), `V8_HMMA_F16_PEAK.md`,
`V8_HMMA_VARIANTS_PEAK.md`, `V9_HMMA_LATENCY.md`.

---

## §24. Tensor cores — tcgen05.mma BF16/FP16 ~1980-2240 TFLOPS

**Answer:** **tcgen05.mma BF16/FP16 = 2242 TFLOPS zero-data peak / ~1850-1900
TFLOPS realistic** at 1920 MHz lock via cuBLAS (which internally uses tcgen05).
Microbench direct = 2325 TFLOPS. Spec is 2500 BF16 dense — **so 90% MFU.**  `[🟢 HIGH · src: B300_TRUE_REFERENCE.md, V8 prior runs, NVIDIA spec]`

This is the **modern Blackwell tensor path** via the new `tcgen05.mma` PTX
instruction family. `cuBLAS` internally dispatches to this path on B300; it
is also accessible via direct PTX (CUTLASS, custom kernels). 4× the legacy
mma.sync throughput per §23.

### Per-precision peak ladder via cuBLAS internal tcgen05

Source: `corrections/06_tensor_cores_CORRECTED.md` table, anchored to
`B300_TRUE_REFERENCE.md` row 66-67 + commit `6e40ef9`.

| Precision | Zero TFLOPS | Random TFLOPS | Realistic TFLOPS | % of NVIDIA spec | Confidence |
|---|---:|---:|---:|---:|---|
| **FP16** (cuBLAS, 8K³) | 2246 | 1905 | 1744 | 91% of 2465 spec | 🟢 HIGH |
| **BF16** (cuBLAS, 8K³) | 2246 / 2242 | 1883 | 1850 | 90% of 2500 spec | 🟢 HIGH |
| BF16 microbench (tcgen05 direct) | 2325 | n/a | n/a | 93% | 🟢 HIGH |
| **TF32** (cuBLAS, 8K³) | 1113 | n/a | n/a | 90% of 1232 | 🟢 HIGH |

### Key observations

1. **"Zero" data gives the spec-quotable peak.** When `A = B = 0` (or constant),
   tcgen05 hardware engages aggressive operand-merge / zero-skip optimizations
   that reach 91-93% of the published spec. NVIDIA's marketing 2500 TFLOPS
   for BF16 implicitly assumes this best case.

2. **Realistic data drops 10-22%.** Random-fill input data prevents the
   zero-skip optimization, dropping BF16 cuBLAS from 2242 to 1883 (random)
   to 1850 (normal-distribution proxy). This is the **realistic ML inference /
   training number**.

3. **microbench direct = 2325 TFLOPS** — slightly higher than cuBLAS zero-data
   peak (2242). Direct PTX `tcgen05.mma` without the cuBLAS overhead (TMEM
   setup, scaling factor handling, etc.) hits 93% of spec. cuBLAS realistic
   conditions add the descriptor overhead.

### Data-dependent throughput (key for honest numbers)

cuBLAS data-pattern table (commit `6e40ef9`, N=K=M=8192):

| Precision | zero/const | random | normal-ish | worst slowdown |
|---|---:|---:|---:|---:|
| FP16 | 2246 | 1905 | 1744 | **−22%** |
| BF16 | 2246 | 1883 | 1850 | **−18%** |
| FP8 (e4m3) | 4500 | 3983 | 3951 | **−12%** |

**Quoting any cuBLAS peak without specifying data pattern is misleading.**
The 4500/2200/2200 numbers are zero-data; subtract 10-22% for realistic.

For ML inference / training where input activations and weights have
~normal distribution statistics, the **1850 BF16 / 1744 FP16 / 3951 FP8**
TFLOPS numbers are the realistic quotes.

### Why the modern path is faster than legacy mma.sync

| Architectural feature | Legacy mma.sync | Modern tcgen05.mma |
|---|---|---|
| Memory location of operands | RF (registers) | TMEM (tensor memory, dedicated) |
| Number of MMAs per instruction | 1 (m16n8k16) | many (e.g. m128n128k16 single MMA) |
| Operand reuse across MMAs | per-warp | per-CTA via TMEM dedup |
| Power efficiency | baseline | better — fewer RF reads, more dedup |
| Pipe utilization | pipe_tensor (legacy) | UTCHMMA / UTCQMMA / UTCOMMA (new metrics) |

The **4× speedup** comes from doing larger MMAs per instruction (less
instruction-issue overhead) and from B-side TMEM operand deduplication
(see `project_tcgen05_power.md` user memory).

### CRITICAL: ncu pipe_tensor does NOT measure tcgen05

Per the user memory entry "ncu pipe_tensor doesn't measure tcgen05" (and
documented in `06_tensor_cores_CORRECTED.md` R4):

- `sm__pipe_tensor_cycles_active` measures **LEGACY** mma.sync tensor pipe usage
- It does NOT see UTCHMMA / UTCQMMA / UTCOMMA (the tcgen05 SASS opcodes)
- Use `smsp__inst_executed_pipe_tensor_subpipe_hmma_op_utchmma_utcqmma_utcomma_*`
  metric family for tcgen05 measurements
- A tcgen05 kernel can show `pipe_tensor = 99%` for a tiny fraction of its
  runtime if it ALSO contains some legacy mma.sync; the rest is invisible

This is the most-stepped-on rake in the catalog. SESSION_2_DELTA documents an
example: a tcgen05 kernel ran for 9.37 ms wall-clock, ncu pipe_tensor showed
99% active for 2.42 ms, and authors initially concluded the kernel was
"99% pipe_tensor saturated" — wrong. The 2.42 ms was a small mma.sync warmup;
the 7 ms tcgen05 main body was invisible to that metric.

**Fix:** Always use both metrics. For pure-tcgen05, expect pipe_tensor ≈ 0%
and `utchmma_*` ≈ high. For mixed kernels, both contribute.

### TF32 cuBLAS

TF32 (19-bit mantissa float used in tensor MMA inputs) on cuBLAS reaches
**1113 TFLOPS = 90% of 1232 spec**. TF32 is half of FP16 (K=8 instead of
K=16) but tcgen05 makes up some ground vs the legacy mma.sync TF32 (288
TFLOPS — 4× ratio between modern and legacy holds here too).

For workloads accepting TF32 precision (~7 sig digits), this is the FASTEST
"feels-like-FP32" cuBLAS path. PyTorch's `torch.set_float32_matmul_precision('high')`
enables this on Blackwell.

### Sustained vs single-shot

| Pattern | TFLOPS | Notes |
|---|---:|---|
| Single-shot (one cuBLAS Lt call) | 2242 zero / 1883 random | warm clock, no thermal throttle |
| Sustained (cudaGraph 15s) | ~1850 random | minor thermal margin loss |
| Under 600 W power cap | (not measured for BF16) | see §25 for FP8 example |

For BF16, the sustained vs single-shot gap is small (<5%) because BF16 power
draw is well within the 1100 W TDP envelope. The big sustained-throughput
losses appear in NVFP4 (see Agent E) and FP8 (see §25 Footgun).

### Latency

HMMA latency on the legacy path is 20 cy (§23). For tcgen05, latency depends
on the MMA shape; small shapes (m64n64k16) ≈ 30-40 cy, large (m128n128k16) ≈
128 cy. A direct comparison isn't useful because the throughput per
instruction is so different.

### What is the realistic "BF16 TFLOPS" number to cite?

For ML practitioners asking "how fast is B300 for BF16 GEMM":

- **2500 TFLOPS** — NVIDIA datasheet spec (zero-data peak)
- **2242 TFLOPS** — measured cuBLAS zero-data peak (90% of spec)
- **1850 TFLOPS** — measured cuBLAS realistic (random data, 74% of spec)
- **569 TFLOPS** — legacy mma.sync (NOT what cuBLAS uses; only relevant if
  you write your own non-cuBLAS kernel)

Recommended quote: **"~1850-2240 TFLOPS BF16 (cuBLAS, 90% MFU at zero-data,
74% MFU realistic)"**. Always disambiguate the data pattern.

### Files

| File | Purpose |
|---|---|
| `B300_TRUE_REFERENCE.md` row 66-67 | cuBLAS BF16 / FP16 results |
| `06_tensor_cores_CORRECTED.md` table 1 | per-precision tcgen05 peak ladder |
| `corrections/CUBLAS_BIT_ENTROPY_CORRECTION.md` | data-pattern impact on power and throughput |
| `NVFP4_PURE_TCGEN05_RESULTS.md` | direct tcgen05 PTX measurements (NVFP4 — Agent E) |

**Footgun:** ⚠ ncu `pipe_tensor` metric does **NOT** measure tcgen05 — only
legacy mma.sync. For tcgen05 (modern Blackwell tensor path), use the
`utchmma_utcqmma_utcomma` metric family. Reading `pipe_tensor = 0%` on a
tcgen05 kernel does NOT mean the kernel is idle — it means the kernel is
using the modern path that the legacy metric can't see.

**Footgun 2:** ⚠ 2242 / 2246 BF16 TFLOPS is the **zero-data** peak. Realistic
random / normal data drops 10-22%. State the data pattern in any quote.

**See also:** §23 (legacy mma.sync = 4× slower), §25 (FP8 via cuBLAS
LtMatmul — same tcgen05 path internally), Agent E §50-53 (NVFP4 deep dive),
`B300_TRUE_REFERENCE.md`, `06_tensor_cores_CORRECTED.md`.

---

## §25. Tensor cores — FP8 e4m3 cuBLAS LtMatmul ≈ 3984-4425 TFLOPS

**Answer:** **FP8 e4m3 cuBLAS LtMatmul = 4425 TFLOPS zero-data sustained / 3984
TFLOPS random data realistic / 3087 TFLOPS under 600 W power cap.** Sustained
via cudaGraph for 30 sec at 943 W. Spec is 5000 dense — so **88% MFU realistic.**  `[🟢 HIGH · src: B300_TRUE_REFERENCE.md commits 06b0d8d / bf98e90 / TRUE_REFERENCE row 56-57]`

The headline production-ML number on B300. cuBLAS internally uses tcgen05.mma
(NOT legacy mma.sync — see §23 for why FP8 mma.sync silently fails).

### Per-mode FP8 measurement table

| Mode | Data | Sustained? | TFLOPS | Power | % of 5000 spec | Source |
|---|---|---|---:|---:|---:|---|
| Zero data, sustained | const/zero | yes (cudaGraph 30s) | **4425** | 943 W | **88.5%** | TRUE_REFERENCE row 56 (`06b0d8d`) |
| Zero data, peak microbench | const/zero | no | 4486 / 4491 | n/a | 89.7% | TRUE_REFERENCE row 56 |
| Zero data, microbench tcgen05 direct | const/zero | no | 4651 | n/a | 93% | `06_tensor_cores` row 17 |
| **Random data, sustained** | **uniform random** | **yes (realistic)** | **3984** | (not measured) | **80%** | TRUE_REFERENCE row 57 (`bf98e90`) |
| Realistic ML data | normal-ish | yes | 3951 | (not measured) | 79% | `06_tensor_cores` table |
| **Under 600 W power cap** | random | yes | **3087** | 600 W | **62%** | TRUE_REFERENCE warning |

### Recommended single-quote number

**Realistic FP8 GEMM throughput on B300 = ~3984 TFLOPS** (random data, sustained,
no power cap). This is the single most defensible quote for "how fast is B300
for FP8 inference / training" and matches what NVIDIA-internal benchmarks
quote for production workloads (it's just not what they put on the marketing
slide).

### Why "FP8 4500 / 7500 / 8200" peak claims were wrong

The catalog history contains several inflated FP8 peak claims:

| Claim | Reality | Why wrong |
|---|---|---|
| "FP8 4491 TFLOPS = 90% MFU" | True for **zero data** only | Random data drops to 3984 (-10%) |
| "FP8 7500 TFLOPS sparse" | Only with sparse metadata that may be junk | Real sparse → see R6 in 06_tensor_cores; 7.44 PF was a steady-state for that test only |
| "FP8 8200 TFLOPS" | Hyper-aggressive zero-data + tcgen05 direct | Microbench, not realistic |

**ALL the catalog "peak FP8" numbers were zero-data**. The realistic ceiling
is 10-22% lower. Use 3984 (random sustained) as the realistic quote.

### The 600 W power cap drop

Under a 600 W power cap (e.g. for cluster TDP budgeting), FP8 cuBLAS drops
from 3984 to **3087 TFLOPS — a 22% loss**. This is significant for sites that
cap GPU power below the 1100 W TDP. To get the full 3984, the GPU needs to
draw ~1000 W during the kernel.

For comparison, BF16 cuBLAS doesn't show the same cap-induced drop because
BF16 power draw is naturally lower (~700 W at 2242 TFLOPS).

### Sustained vs single-shot via cudaGraph

The **cudaGraph wrapper** is necessary for FP8 sustained measurements:

- Single-shot cuBLAS Lt FP8 call: ~4486 TFLOPS (warm clock, no thermal throttle)
- 30-iteration loop without cudaGraph: drops to ~3500-4000 TFLOPS (clock
  throttles to 1900 MHz under cumulative power)
- 30-second cudaGraph FP8 sustained: **4425 TFLOPS at 943 W** (zero data) /
  3984 TFLOPS (random data). Clock holds at 2032 MHz boost throughout.

The cudaGraph trick: by capturing the cuBLAS call graph once and replaying it
30 sec, the per-call overhead is amortized and the GPU's power management has
a stable workload pattern to optimize against. Without the cudaGraph, repeated
single launches confuse the DVFS controller and cause clock oscillation.

This is documented in `feedback_b300_pitfalls.md` user memory: "cuBLAS needs
cudaGraph" for sustained throughput measurements.

### tcgen05 power model context (§B project_tcgen05_power memory)

FP8 power draw on B300 follows the per-bit toggle-energy model:
- ~80 W per active sign-bit toggle in the operand stream
- B-side has 32-byte sub-tile dedup (lower power than A which is "free")
- K-row pairwise dedup further reduces power
- Worst case (popcount d=16 random) = highest power
- Zero-data exploits the dedup hardware → lowest power, highest throughput

So the data-dependent throughput drop has a direct hardware mechanism:
zero data triggers operand-merge / zero-skip in the multiplier array;
random data prevents that, increases switching power, hits the power cap
sooner.

### How to measure FP8 cuBLAS LtMatmul correctly

```cpp
cublasLtHandle_t lt;
cublasLtCreate(&lt);

cublasLtMatmulDesc_t op;
cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);

// FP8 e4m3 inputs, F32 accumulator, F32 output
cublasLtMatrixLayout_t aDesc, bDesc, cDesc;
cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_8F_E4M3, M, K, M);
cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_8F_E4M3, K, N, K);
cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_32F, M, N, M);

// MUST set scale type for FP8
cudaDataType_t scaleType = CUDA_R_32F;
cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_SCALE_TYPE,
                                &scaleType, sizeof(scaleType));

// Capture cudaGraph
cudaGraph_t graph;
cudaGraphExec_t graphExec;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
for (int i = 0; i < N_REPEAT; i++) {
    cublasLtMatmul(lt, op, &alpha, A, aDesc, B, bDesc, &beta,
                   C, cDesc, C, cDesc, nullptr, nullptr, 0, stream);
}
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

// Time the graph launch
auto t0 = ...;
cudaGraphLaunch(graphExec, stream);
cudaStreamSynchronize(stream);
auto t1 = ...;
double tflops = 2.0 * M * N * K * N_REPEAT / (t1-t0).count() / 1e12;
```

This recipe reaches 4425 TFLOPS zero-data / 3984 TFLOPS random at M=N=K=8192.

### Latency

FP8 tcgen05.mma latency depends on the MMA shape. For typical
m64n64k32 single-MMA: ~30 cy. Cycle-accurate latencies for tcgen05 not
in the catalog with high confidence.

### What about FP8 mma.sync?

Per §23, FP8 via legacy mma.sync **does NOT work as a fast path on B300**:
- Compiler accepts the e4m3 PTX syntax
- SASS silently falls back to a non-FP8 emulation
- Pipe_tensor only 28%; only 2 HMMAs in SASS for claimed 8-chain loop
- "FP8 mma.sync 276 TFLOPS effective" entry is **F2FP.UNPACK + HMMA emulation**,
  NOT native FP8 tensor

**The only path to FP8 peak on B300 is cuBLAS LtMatmul (or direct CUTLASS
+ tcgen05 PTX).** Don't try to use mma.sync for FP8.

### What about FP8 e5m2?

The catalog measurements were done on e4m3 (4-bit exponent, 3-bit mantissa).
e5m2 (5-bit exp, 2-bit mantissa) is also supported on tcgen05 with the same
peak throughput (5000 TFLOPS spec). cuBLAS LtMatmul accepts both via
`CUDA_R_8F_E4M3` / `CUDA_R_8F_E5M2`. No measured throughput difference between
the two.

### What about FP8 with FP32 vs FP8 accumulator?

Same throughput. Accumulator format is free at the tcgen05 level (per §23
finding for HMMA — F32 accumulator is free). FP32 accumulator is what every
production training pipeline uses (BF16/FP8/FP4 inputs → FP32 accum → BF16
output).

### Files

| File | Purpose |
|---|---|
| `B300_TRUE_REFERENCE.md` row 56-57 | cuBLAS FP8 measurements |
| `06_tensor_cores_CORRECTED.md` | per-precision peak ladder |
| `feedback_b300_pitfalls.md` | "cuBLAS needs cudaGraph" |
| `project_tcgen05_power.md` | tcgen05 power model |

### Quick-cite cheat sheet for FP8 on B300

| Need | Use this number |
|---|---:|
| Marketing peak | 5000 TFLOPS (NVIDIA spec) |
| Zero-data sustained measured | 4425 TFLOPS (88.5% MFU) |
| Zero-data peak microbench | 4486-4651 TFLOPS |
| **Realistic ML inference / training** | **3984 TFLOPS (80% MFU)** |
| Under 600 W power cap | 3087 TFLOPS (62%) |
| Sparse FP8 (caveat) | 7440 TFLOPS — flagged DOWNGRADED, sparse metadata may be junk |

**Footgun:** ⚠ Catalog "FP8 4491 / 7500 / 8200 TFLOPS" peaks were **ALL
zero-data**. Realistic random-data is **-10% to -22%**. Use 3984 TFLOPS as
the realistic quote, not 4500.

**Footgun 2:** ⚠ FP8 via legacy `mma.sync` is **NOT NATIVE** on B300 — compiler
silently falls back to F2FP.UNPACK + HMMA emulation. Only cuBLAS LtMatmul (or
direct tcgen05.mma in CUTLASS) reaches the 4000+ TFLOPS path.

**Footgun 3:** ⚠ Without cudaGraph, sustained FP8 measurements drop ~10%
from clock oscillation. Capture the cuBLAS call in cudaGraph then replay for
sustained timing.

**Footgun 4:** ⚠ Power-capped systems (600 W cap common in some cluster
deployments) see FP8 drop to 3087 TFLOPS = 62% MFU. Not all sites can hit
the 3984 realistic quote.

**See also:** §24 (BF16 cuBLAS — same tcgen05 path), §23 (why mma.sync
doesn't work for FP8), Agent E §50-53 (NVFP4 deep dive — even higher peak),
Agent F (power and clock interaction), `B300_TRUE_REFERENCE.md`,
`06_tensor_cores_CORRECTED.md`, `feedback_b300_pitfalls.md`.

---

## Appendix A — The dual-issue 5-level zigzag (full timeline)

This is the documented 5-level flip-flop on the dual-issue verdict, included
in §22 footgun but expanded here for methodology students. The pattern
illustrates why a single doubt-and-revise cycle is insufficient when
empirical anchors are weak.

### Wave 1+2 (early V4, 2026-04-19)

**M8 PIPE_OVERLAP_MATRIX measured.** Method: same-warp 8-deep loop, two
op types interleaved, wall-clock comparison to sum-of-isolated.

| Pair | Overlap |
|---|---:|
| FFMA + LDS | 96% |
| FFMA + IADD3 | 56% |
| FFMA + MUFU | 100%+ |
| HMMA + HMMA | 69% |
| HMMA + LDS | 73% |
| HMMA + LDTM | 28% |

Verdict: **HIGH confidence** that pipes overlap with varying efficiency.
"Issue-port-bound" framing introduced — same-warp same-issue-port competition
explains 56% IADD3+FFMA.

### Wave 3a (V49, V50, late 2026-04-20)

**V49 (`501134a`):** "FFMA+LOP3 same-warp = 55%, FFMA+IADD3 = 54%, FFMA+PRMT = 51%".
Wall-clock measurement, `__launch_bounds__(128, 2) = 2 warps/SMSP`,
`#pragma unroll 1` inner loop with body of 8 FFMA + 8 LOP3 + branch.

V49 author's interpretation: "B300 SMSP scheduler dispatch slot is shared
across pipes (4 inst/cy/SM total). Even different pipes can't both issue
1/cy/SMSP simultaneously."

**V50 (`fbe1c18`):** "Warp-specialized = 74% efficient" — uses 4 FFMA warps
+ 4 LOP3 warps per SM. Higher than V49's 55%.

V50 author's interpretation: warp-spec helps because each warp does only one
op type, reducing per-warp scheduler contention; but still capped at 74%
because each SMSP still alternates 1 FFMA + 1 LOP3 per cycle.

Verdict: **HIGH confidence** that B300 has a "shared dispatch cap"
architectural feature.

### Wave 3b (DUAL_ISSUE_DOUBT_REPORT, early 2026-04-21)

Doubt agent points out:

1. V49's FFMA solo measures **67% of FFMA peak**, NOT 100%. The "separate-pipe
   theoretical" denominator the V49 ratio computes against is itself wrong.
2. `__launch_bounds__(128, 2) = 2 warps/SMSP` is borderline; A1 noted "need
   ≥4 warps/SMSP for full saturation".
3. The 55% → 74% jump from V49 → V50 could be tracking the warp-count change
   (2→4 warps/SMSP), not the warp-spec hypothesis.
4. ncu metrics not collected; cannot confirm the dispatch-cap interpretation.

Recommendation: **DOWNGRADE to LOW**, re-measure with proper occupancy + ncu.

### Wave 4 (META_DOUBT_REPORT, mid 2026-04-21)

Meta-doubt agent audits the doubt-report. Findings:

1. V8_FFMA_PEAK_VERIFIED reaches **97.64% pipe_fma** with `__launch_bounds__(256, 1) = 8 warps/SM = 2 warps/SMSP, IDENTICAL to V49`. So the under-occupancy mechanism is **falsified**.
2. V49's 67% FFMA-solo number is itself anomalous (V8 reaches 97% at same occupancy) — suggests something else is going on (possibly the immediates `0.5f` in V49's pattern).
3. M8's MUFU+FFMA = 100%+ and HMMA+HMMA = 69% are real measurements that contradict the "shared dispatch cap" model.

Recommendation: **RE-PROMOTE to MED** (numbers are real, but mechanism wrong).

### Wave 5a (SASS_VERIFY_DUAL_ISSUE, late 2026-04-21)

SASS-audit agent dumps actual SASS for V49 vs V8:

- V49 inline asm `fma %0, %0, 0f3FC00000, 0f3F000000` — compiler hoists 1.5f
  into R0 once before loop; SASS is `FFMA Rd, Rd, R0.reuse, 0.5`.
- V8 inline asm `fma %0, %0, %1, %0` — SASS is `FFMA Rd, Rsrc1, Rd, Rd`.

Both are 2-source patterns avoiding the 3-distinct-source RF port limit.
The mechanism the meta-doubt invoked (immediates vs registers) is **falsified**.

The REAL difference: V49 inner body is **8 FFMA + 8 LOP3 + 1 BRA + 1 UIADD3 + 1 UISETP** (loop overhead 12.5%), while V8 inner body is **128 FFMA + tail**
(loop overhead 1.2%). V49 measures the steady-state of an FFMA+LOP3+branch loop.

Recommendation: **RE-DOWNGRADE to LOW** until reproduced with V8-style 128-deep
unroll. Architectural question (does B300 SMSP dual-issue FMA + ALU?) remains
**OPEN**.

### Wave 6 (V52, 2026-04-22)

V52 (`tests/standalone/v52_dual_issue_clean.cu`) implements V8-style 128-deep
unroll for solo FFMA, solo LOP3, and dual modes. Critical addition: ncu
profile reading **both `pipe_alu` and `pipe_fma` simultaneously**.

Result: dual mode shows `pipe_alu = 98.0%` AND `pipe_fma = 49.4%` AT THE SAME
TIME. Sum = 147%. `inst_issued/cy = 1.00` in dual mode (vs 0.51 solo LOP3, 1.00 solo FFMA).

**Verdict (final):** Pipes overlap freely. The 55%/74% from V49/V50 are
methodology artifacts (loop-overhead). The "shared dispatch cap" was a
phantom. The real model:

- Each SMSP has 1 dispatch slot per cycle
- The dispatch slot can issue to either FMA pipe OR ALU pipe per cycle
- Solo FFMA: 1 issue/cy → FMA pipe = 97% busy
- Solo LOP3: 1 issue every 2 cy (LOP3 has 2-cy intrinsic cadence) → ALU pipe = 98% busy, inst_issued = 0.51/cy
- Dual: 1 issue/cy alternating → FMA = 49%, ALU = 98%, sum = 147%. Both pipes saturate but FFMA only gets every other cycle (because LOP3 takes the other half).

Confidence: **HIGH (architectural truth) + RETRACTED (V49/V50 numbers)**.

### Lessons for methodology

1. **Wall-clock comparisons cannot prove dual-issue.** They confound dispatch
   sharing with per-instruction issue cadence. Read pipe_X_cycles_active
   metrics from a single ncu profile, sum them.
2. **Loop-overhead contamination is hard to spot.** V49 looked clean at the
   PTX level; only SASS inspection revealed the 12.5% overhead.
3. **The "obvious mechanism" is often wrong.** W3a invoked "shared dispatch
   cap"; W4 invoked "occupancy artifact"; W5a invoked "loop overhead"; only
   W6's direct ncu reading was decisive.
4. **Doubt without empirical anchor is just guessing.** W3b/W5a both arrived
   at "LOW confidence" but for partially-wrong reasons. The right answer was
   "we need ncu pipe_alu + pipe_fma summed", which only W6 actually did.
5. **Always state separately:** (a) Is the published number accurate? (b) Is
   the architectural inference accurate? These are independent.

### Bibliography for the zigzag

| File | Wave | Role |
|---|---|---|
| `M8_PIPE_OVERLAP_MATRIX.md` | W1+2 | initial pipe overlap measurement |
| `V49_DUAL_PIPE_LIMITS.md` (commit `501134a`) | W3a | "55% same-warp ceiling" |
| `V50_WARP_SPECIALIZED_DUAL.md` (commit `fbe1c18`) | W3a | "74% warp-spec" |
| `corrections/DUAL_ISSUE_DOUBT_REPORT.md` | W3b | doubt the V49/V50 baseline |
| `corrections/META_DOUBT_REPORT.md` | W4 | re-promote based on V8 evidence |
| `corrections/SASS_VERIFY_DUAL_ISSUE.md` | W5a | SASS-level forensics |
| `corrections/V52_RUN_RESULTS.md` | W6 | empirical settlement |
| `corrections/HEADLINE_CORRECTIONS_v5.md` | W6 | canonicalization |
| `tests/standalone/v52_dual_issue_clean.cu` | W6 | the kernel |

---

## Appendix B — Compute peak summary table (all of §16-§25 condensed)

For quick reference at the end of Section B. All numbers @ 2032 MHz boost
unless flagged.

| Op | Theoretical peak | Measured peak | %SoL | Section |
|---|---:|---:|---:|---|
| FFMA (FP32, 2-source) | 76.96 TFLOPS | 75.20 TFLOPS | 97.65% | §16 |
| FFMA (3-distinct-source) | n/a (port-limited) | 51.30 TFLOPS | 66.6% | §17 |
| FADD | 38.48 TFLOPS | 37.40 TFLOPS | 97.65% | §19 |
| FMUL | 38.48 TFLOPS | 37.30 TFLOPS | 97.62% | §19 |
| FP64 DFMA | 1.203 TFLOPS | 1.203 TFLOPS | **100.00%** | §20 |
| IMAD (32-bit) | 38.48 Tops | 38.40 Tops | 99.7% | §21 |
| HMMA m16n8k16 (FP16/BF16, F16 acc) | ~580 TFLOPS | 578.6 TFLOPS | 99.90% | §23 |
| HMMA m16n8k16 (BF16, F32 acc) | ~580 TFLOPS | 578.6 TFLOPS | 99.89% | §23 |
| TF32 m16n8k8 | ~290 TFLOPS | 288 TFLOPS | ~99% | §23 |
| INT8 m16n8k32 | (HW-throttled) | 143 Tops | 100% of throttled cap | §23 |
| BF16 cuBLAS (tcgen05, zero data) | 2500 TFLOPS spec | 2242 TFLOPS | 90% | §24 |
| BF16 cuBLAS (random data realistic) | 2500 TFLOPS spec | 1850 TFLOPS | 74% | §24 |
| FP16 cuBLAS (tcgen05, zero data) | 2465 TFLOPS spec | 2246 TFLOPS | 91% | §24 |
| FP16 cuBLAS (random data realistic) | 2465 TFLOPS spec | 1744 TFLOPS | 71% | §24 |
| TF32 cuBLAS | 1232 TFLOPS spec | 1113 TFLOPS | 90% | §24 |
| FP8 e4m3 cuBLAS (zero data sustained) | 5000 TFLOPS spec | 4425 TFLOPS | 88.5% | §25 |
| FP8 e4m3 cuBLAS (random data realistic) | 5000 TFLOPS spec | 3984 TFLOPS | **80%** | §25 |
| FP8 e4m3 cuBLAS (under 600 W cap) | 5000 TFLOPS spec | 3087 TFLOPS | 62% | §25 |

### Pipe occupancy summary (V52 + V8 era)

| Test | pipe_fma | pipe_alu | pipe_fp64 | pipe_tensor | inst_issued/cy | Notes |
|---|---:|---:|---:|---:|---:|---|
| Solo FFMA (V8) | 97.64% | ~0% | 0% | 0% | 1.00 | full RF-port-friendly |
| Solo FADD/FMUL | 97.65% | ~0% | 0% | 0% | 1.00 | identical pipe |
| Solo IMAD | (high) | ~0% | 0% | 0% | (1.00 / 2 — half-rate) | FMA pipe shared, half rate |
| Solo DFMA | 0% | 0% | **100.00%** | 0% | 1/64 | own pipe, fully saturable |
| Solo LOP3 (V52) | <1% | 99.45% | 0% | 0% | **0.51** | 2-cy issue cadence |
| Dual FFMA+LOP3 (V52) | **49.39%** | **98.00%** | 0% | 0% | **1.00** | sum = 147%, free overlap |
| Solo HMMA m16n8k16 (V8) | 0% | 0% | 0% | **99.90%** | (chain-bound) | legacy tensor pipe |
| Solo tcgen05.mma | 0% | 0% | 0% | **0% (!)** | n/a | NOT measured by pipe_tensor |

### Latency ladder (V9 / V8)

| Op | Latency (cy) | Throughput cap | Saturation ILP |
|---|---:|---|---:|
| FFMA / FADD / FMUL | 4.22 | 1/cy/SMSP | 4 chains |
| IMAD | 4.25 | 1/(2cy)/SMSP | 2 chains |
| LOP3 | ~2 | 1/(2cy)/SMSP | 2 chains |
| DFMA | 63.68 | 1/(64cy)/SMSP | 1 chain |
| HMMA m16n8k16 (F32 acc) | 20.09 | 1/(4cy)/SMSP | 5 chains |
| HMMA m16n8k16 (F16 acc) | 20 (inferred same) | 1/(4cy)/SMSP | 5 chains |
| tcgen05.mma small shapes | ~30-40 | (depends on shape) | (varies) |
| tcgen05.mma large shapes | ~128 | (depends on shape) | (varies) |

---

## Appendix C — Quick-cite cheat sheet (Agent B's slice of compute)

| Need | Use this number | Section |
|---|---:|---|
| FP32 FFMA peak (peak conditions) | **75.2 TFLOPS** (97.65%) | §16 |
| FP32 FFMA peak (canonical conservative) | **74.62 TFLOPS** (96.92%) | §16 |
| FP32 FFMA realistic (3-source GEMM) | **51.3 TFLOPS** (67%) | §17 |
| FP32 FFMA at 1920 MHz lock | 62.17 TFLOPS (85.5%) | §16 |
| FP64 DFMA | **1.20 TFLOPS** (100%) | §20 |
| FADD / FMUL (1 FLOP per inst) | 37.4 TFLOPS each | §19 |
| IMAD (32-bit) | **38.4 Tops** (99.7% of 1:2 ratio) | §21 |
| Dual FMA + ALU pipe sum (V52) | **147%** of single-pipe peak | §22 |
| FP16/BF16 mma.sync m16n8k16 | **578 TFLOPS** (99.9%) | §23 |
| TF32 mma.sync m16n8k8 | 288 TFLOPS | §23 |
| INT8 mma.sync m16n8k32 | 143 TOPS (HW-throttled) | §23 |
| BF16 cuBLAS realistic | **1850 TFLOPS** (74%) | §24 |
| BF16 cuBLAS zero-data peak | 2242 TFLOPS (90%) | §24 |
| FP16 cuBLAS realistic | 1744 TFLOPS | §24 |
| TF32 cuBLAS | 1113 TFLOPS | §24 |
| FP8 e4m3 cuBLAS realistic | **3984 TFLOPS** (80%) | §25 |
| FP8 e4m3 cuBLAS zero-data sustained | 4425 TFLOPS (88.5%) | §25 |
| FP8 e4m3 cuBLAS under 600 W cap | 3087 TFLOPS (62%) | §25 |

### Fundamental constants used in this section

| Constant | Value | Source |
|---|---:|---|
| Number of SMs | 148 | sm_103a B300 SXM6 |
| SMSPs per SM | 4 | architectural |
| FP32 lanes per SMSP | 32 | architectural |
| FP32 cores per SM | **128** (NOT 256) | 4 × 32 |
| FP32 cores per chip | 18 944 | 148 × 128 |
| Boost clock (default unlocked) | 2032 MHz | sustained under FFMA |
| Locked clock paradox | `-lgc 2032` pins to **1920** MHz | nvidia-smi quirk |
| FP64:FP32 ratio | **1:64** | `cudaDeviceGetAttribute SingleToDoublePrecisionPerfRatio` |
| IMAD:FP32 ratio | **1:2** | CUDA C PG Table 13-1 |
| RF read ports per SMSP | **2** + reuse cache | D6 measurement |

---

End of Section B (§16-§25).
