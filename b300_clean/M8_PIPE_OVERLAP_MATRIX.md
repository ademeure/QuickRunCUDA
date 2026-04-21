# M8: B300 Single-Warp Pipe Overlap Matrix

Empirical measurements of how pairs of operations overlap when issued from a
single warp (32 threads). All measurements at 1500 MHz locked, ITERS=1000.

---

## The complete matrix

Each cell = `% overlap fraction = (sum_individual - measured_combined) / (sum_individual - max_individual)`.
- **100%** = perfect parallelism (other op completely hidden)
- **0%** = full serialization (combined = sum)
- **>100%** = super-linear (filled scheduler bubbles)

| (1st pipe) \ (2nd pipe) | LSU (LDS) | XU (MUFU) | FMA (FFMA) | ALU (IADD3) | TENSOR (HMMA) | TENSOR (LDTM) |
|--|--|--|--|--|--|--|
| **LSU (LDS)** | -- | -- | **96%** ✓ | -- | 73% | -- |
| **XU (MUFU)** | -- | -- | **100%+** | -- | 71% | -- |
| **FMA (FFMA)** | 96% | 100%+ | -- | 56% | 31% | -- |
| **ALU (IADD3)** | -- | -- | 56% | -- | 32% | -- |
| **TENSOR (HMMA)** | 73% | 71% | 31% | 32% | **69%** | 28% |
| **TENSOR (LDTM)** | -- | -- | -- | -- | 28% | -- |

(empty cells: not measured; pattern symmetric)

---

## Architectural model (refined from V5/V6 evidence)

### Three categories of B300 ops:

**A. Async-queue ops (LSU = LDS/LDG)**
- Issue once into L1TEX queue, doesn't tie up SMSP scheduler
- Completes asynchronously when result returns
- Overlap with anything: **96%+** (essentially free)

**B. Pipe-queued ops (XU = MUFU.RCP/SQRT/EX2)**
- Issue once into xu pipe, then MUFU's internal pipeline executes
- Doesn't tie up SMSP scheduler for full op latency
- Overlap with compute: **71-100%** (very good)

**C. Issue-port-bound compute (FMA/ALU = FFMA/IADD3)**
- Each op consumes 1 cycle of the SMSP scheduler issue port
- Two issue-port-bound ops PARTIALLY parallelize via different pipes (~56%)
- But share scheduler issue ⇒ NOT fully parallel even if pipes are different

**D. Tensor pipe (HMMA + LDTM/STTM)**
- Tensor pipe has its own sequencer + internal pipelining
- 2 HMMA chains in same warp: **69% overlap** (great pipelining)
- HMMA + LDTM: **28%** (both go through tensor pipe sequencer — serialize)
- HMMA + scalar compute: 31-32% (HMMA's long latency leaves bubbles, but
  scalar ops still need scheduler issue which is mostly busy)

---

## Why HMMA + LDS is only 73% (not 96% like FFMA + LDS)

LDS is async-queued (doesn't block scheduler). FFMA + LDS gets 96% because
both are short-latency.

But HMMA has a long pipeline — its INPUT registers are read multi-cycle,
and the result is written multiple cycles later. While HMMA holds those
registers, LDS *can* issue but its writeback may be delayed if it touches
the same register file partition. Hence 73% not 100%.

---

## Why MUFU + FFMA is 100%+ (super-linear)

MUFU.RCP has 15 cy/op latency with bubbles in the issue stream during
pipe wait. FFMA fills those bubbles, so adding FFMA work makes the
scheduler busier WITHOUT increasing total time. Net: combined < max.

---

## Practical kernel-design implications

### For HMMA-bound kernels:
1. **Add async memory ops freely** (LDS, LDG, cp.async): 73-96% overlap
2. **Add transcendentals freely** (RCP, SQRT, EX2): 71% overlap
3. **DO NOT add scalar FFMA/IADD3 expecting parallelism**: only 31-32% overlap
4. **Multi-tile HMMA accumulation**: 69% overlap from independent chains —
   prefer this over scalar mixing

### For FFMA-bound kernels:
1. **Add LDS/LDG freely**: 96% overlap
2. **Add MUFU freely**: 100%+ (fills bubbles!)
3. **Don't add IADD3 expecting full parallel**: 56% overlap

### For RMSNorm-style kernels:
- Variance computation (FFMA reduction) + RCP (single MUFU) = ~100% overlap
- Per V5 G3: RMSNorm 1024 = 392 ns; the RCP cost is essentially FREE

### For LayerNorm:
- Mean (FFMA) + variance (FFMA + RCP) + normalize (FFMA × RCP_result) =
  most ops can pipeline — limit is RAW dep on the RCP result

---

## V6 commits captured in this matrix

| Pair | Overlap | Commit |
|------|---------|--------|
| HMMA + FFMA | 31% | `13f5a16` |
| HMMA + IADD3 | 32% | `aa8b7eb` |
| HMMA + MUFU | 71% | `17cf0d4` |
| HMMA + LDS | 73% | (V5 B6 `e07621d`) |
| HMMA + LDTM | 28% | (V5 B2 `b6648e2`) |
| HMMA + HMMA | 69% | `d7da49c` |
| FFMA + LDS | 96% | `f1b2f4d` |
| FFMA + IADD3 | 56% | `086ed25` |
| FFMA + MUFU | 100%+ | (in MUFU commit) |

---

## Open questions for V7

1. **3-way overlap**: HMMA + LDS + MUFU — does it sum to 73 + 71 = 144%? Or does adding the 3rd hurt?
2. **Cross-warp overlap**: Two warps in same SMSP issuing different pipes
3. **TMEM → RF latency** (LDTM unique issue patterns)
4. **HMMA + cluster-level ops** (cluster.barrier in same warp as HMMA)
5. **Per-SMSP overlap matrix** (does the matrix differ by SMSP?)
