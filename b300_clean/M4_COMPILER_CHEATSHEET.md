# M4 — B300 Compiler Flags Cheatsheet (2026-04-21)

**nvcc / ptxas flags that materially change B300 kernel performance.**
Synthesizes G1-G9 + related findings.

---

## 1. Performance multipliers (must-use)

| Flag | Impact | Source |
|------|--------|--------|
| `-use_fast_math` | **3.28× faster + 19% lower power = 4.05× lower energy** | G2 (#5a5a393) |
| `__forceinline__` (vs __noinline__) | **15.2× faster** for hot helpers | G7 (#21fa032) |
| `-Xptxas=-O3` (vs -O0) | 4.4× faster (.reuse + unrolling) | G1 (#d657f69) |
| `__restrict__` on inputs | 4.4% faster + LDG.E.CONSTANT routing | G4 (#13c2f2d) |

`-use_fast_math` implies:
- `-ftz=true` (subnormal flush)
- `-prec-div=false` / `-prec-sqrt=false` (use approximations)
- Auto-emits FFMA.FTZ, MUFU.RCP/RSQ/EX2/LG2/SIN/COS.FTZ paths

**WARNING:** NVRTC harness in QuickRunCUDA hardcodes `-use_fast_math`. All FFMA
become FFMA.FTZ even when PTX `fma.rn.f32` (no .ftz) is written.

---

## 2. Register budget control

| `__launch_bounds__(N, M)` | regs/thread budget | Spill behavior |
|---------------------------|--------------------|----------------|
| (256, 1) | 256 → 255 cap | 0 spills typically |
| (256, 2) | 128 | 0 spills if ≤128 |
| (256, 4) | 64 | Some spills if kernel needs more |
| (256, 8) | 32 | **15.8× slowdown** if 65 regs needed (G9) |

**Formula:** `regs_per_thread = 65536 / minBlocks / threads_per_block`

**Modern CUDA:** `__launch_bounds__` and `__maxnreg__` are **mutually exclusive**.
Use `minBlocks` to control implicit budget.

**Hard cap:** 255 registers per thread (P3). Above 255 ptxas warns + caps silently.

---

## 3. Cache control

| PTX modifier | SASS opcode | Latency | Notes |
|--------------|-------------|---------|-------|
| `ld.global` (default) | `LDG.E` | 38 cy hit | L1 cached |
| `ld.global.ca` | `LDG.E.STRONG.SM` | 38 cy | L1 (same as default) |
| **`ld.global.cg`** | **`LDG.E.STRONG.GPU`** | **288 cy** | **Bypasses L1 (only true bypass!)** |
| `ld.global.cs` | `LDG.E.EF` | 38 cy | Evict-first hint, still L1 |
| `ld.global.lu` | `LDG.E.LU` | 38 cy | Last-use hint, still L1 |
| `ld.global.nc` | `LDG.E.CONSTANT` | 38 cy | Constant cache path |

**Practical:** only `.cg` makes a real performance difference.
For streaming workloads where L1 pollution wastes capacity, use `.cg`.

`cudaFuncSetCacheConfig(*, cudaFuncCachePrefer*)` is a NO-OP on Hopper+.
Use `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
for SMEM size control.

---

## 4. Architecture target

For B300 (sm_103a / CC 10.3) you MUST use the architecture-specific suffix:
```
nvcc -arch=compute_103a -code=sm_103a       # required for tcgen05
nvcc --generate-code arch=compute_103a,code=lto_103a -dlto  # LTO build
```

**`tcgen05.alloc`, `mma.sync.kind::f8f6f4`, `cluster.barrier`, etc.** all require the
'a' suffix. Without it, ptxas reports "Instruction not supported on .target 'sm_103'".

**G8 finding:** `-dlto` did NOT enable cross-unit inlining in minimal test.
Don't rely on LTO for inlining; put hot helpers in `.cuh` headers + `__forceinline__`.

---

## 5. PTX-to-SASS surprising patterns

### Operations that AUTO-fold (don't waste effort)
- 3-input boolean → 1 LOP3 (M1: ANY 3-input fits in 256 truth tables)
- a OR b OR (a AND c) etc → ptxas emits LOP3.LUT with 8-bit immediate
- mov.b32 / mad*1+0 / add+0 → ELIDED by copy propagation (C2)

### Operations that EMULATE (avoid where possible)
- PTX `vshl/vshr.clamp` → PRMT + SHF (8.4 cy, 1.85× scalar SHF) (K6)
- PTX `vadd4.u32` → 2× LOP3 (14.4 cy)
- PTX `vmin4.u32` → 2× PRMT (**44.6 cy** — very slow!)

### Operations that switch CLUSTER on predication
- Bare `add.u32` → IADD3 (Cluster B, 2.56 cy)
- **`@p add.u32`** → **IMAD.IADD** (Cluster A, 4.94 cy = **2× slower!**) (M2)
- `@p fma.rn.f32` → FFMA stays Cluster A → **FREE** (C10)

**Rule:** Predication on Cluster B ops → moves to Cluster A → 2× cost.
Predication on Cluster A ops → free.

### Cache modifier auto-emission
- `__restrict__` on input → `LDG.E.CONSTANT` (constant cache routing) (G4)
- `__ldg(p)` → `LDG.E.CONSTANT` (D9)
- Default `ld.global` → `LDG.E` (L1 cached, STRONG.SM)

---

## 6. Loop unrolling control

| Pragma | Effect |
|--------|--------|
| (none) | Compiler decides (often unrolls if loop body small) |
| `#pragma unroll` | Force complete unroll (if known trip count) |
| `#pragma unroll N` | Unroll N times |
| `#pragma unroll 1` | Disable unrolling (preserve loop) |

**Best practice for benchmarks:** use `#pragma unroll 16` or `#pragma unroll 32`
for inner loops to amortize the **23 cy/iter empty-loop BRA overhead** (A1 finding).

---

## 7. Build recipes

### Maximum performance NVRTC kernel
```
-use_fast_math
-O3 (default)
__forceinline__ on all device helpers
__restrict__ on all input pointers
__launch_bounds__(threads, minBlocks) tuned for register count
```

### Standalone (offline) for B300
```
nvcc -arch=compute_103a -code=sm_103a -O3 -use_fast_math \
     -Xptxas=-O3 -lineinfo  -keep -keep-dir /tmp/sass \
     your_kernel.cu -o your_kernel
```

### LTO (separate compilation)
```
nvcc -gencode arch=compute_103a,code=lto_103a \
     -gencode arch=compute_103a,code=sm_103a \
     -O3 -rdc=true file1.cu file2.cu -o app
```
**(but G8 finding: LTO didn't actually inline in test; use .cuh headers instead)**

---

## 8. Verifying via SASS

Use `utils/sass_count.sh` (S3) to count opcodes in main loop body:
```
./QuickRunCUDA -f your_kernel.cu ...   # generates sass/
./utils/sass_count.sh sass/your_kernel_<hash>.sass "FFMA"
```

**Common SASS opcodes to watch:**
- `FFMA / FFMA.FTZ` — float multiply-add
- `IMAD / IMAD.IADD / IMAD.MOV` — integer multiply-add (Cluster A)
- `IADD3` — integer add (Cluster B, fastest)
- `LOP3.LUT` — 3-input bool with truth table
- `LDG.E[*scope*]` — global load (default/.STRONG.SM/.STRONG.GPU/etc.)
- `LDS / LDS.128` — shared load
- `STG.E / STS` — global / shared store
- `HMMA.16816.F32` — FP16/BF16 matrix mma
- `FFMA.FTZ R4, R3, R5, R5.reuse` — FFMA with operand reuse cache
- `STL / LDL` — register spill (avoid!)
- `BSYNC` — partial-mask warp barrier
- `BAR.SYNC.DEFER_BLOCKING` — cta barrier
- `barrier.cluster` → `BARC.SYNC` — cluster barrier (~395 cy)

---

For complete commit history, see `M1_V4_DEEP_DIVE_INDEX.md`.
