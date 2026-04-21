# mma.sync FP8 via kind::f8f6f4 is NOT native — it's F2FP + HMMA

## Discovery
On sm_103a (B300), `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32` does NOT compile to a native FP8 mma SASS opcode. ptxas emits:

```
F2FP.F16.E4M3.UNPACK_B   (FP8 → FP16 conversion, one byte at a time)
F2FP.F16.E4M3.UNPACK_B   ...  (12+ unpack instructions for 32 FP8 inputs)
HMMA.16816.F32           (standard FP16 m16n8k16 mma)
HMMA.16816.F32           (second mma for the K=32 worth of data)
```

So K=32 FP8 is implemented as: convert all FP8 → FP16, run TWO HMMA m16n8k16 calls.

## Measured (32-thread warp, single block, varying-operand chain to defeat constant-fold)
| Path | cy/iter | FLOP/cy/warp | Effective TFLOPS (148SM × 4SMSP × 1.5GHz) |
|------|---------|--------------|---------------------------------------------|
| FP8 m16n8k32 kind::f8f6f4 | 70 | 117 | 104 |
| BF16 m16n8k16 (single)    | 31 | 132 | 117 |
| BF16 m16n8k16 ×2 (= K=32) | 51 | 161 | 143 |

**FP8 mma.sync is 1.37× SLOWER than equivalent 2×BF16** for the same K=32 effective FLOPs (8192).

## Implication
- The mma.sync FP8 path on Blackwell adds gratuitous F2FP unpack overhead AND uses the same FP16 HMMA pipe — net: pure loss.
- Native FP8 throughput on B300 is ONLY available via `tcgen05.mma` (the Tensor Memory Generator path).
- The PTX form `kind::f8f6f4` advertises FP8/FP6/FP4 but the SASS reveals it's just an FP8→FP16 syntactic-sugar wrapper around HMMA.
- For any practical FP8 GEMM, do NOT use mma.sync; use cuBLAS / CUTLASS which dispatches to tcgen05.

## Why ptxas chose this path
mma.sync as an instruction class on Blackwell was preserved for backward compat with Hopper/Ada. The native Blackwell tensor instruction is tcgen05.mma. ptxas implements new mma.sync .kind:: variants on top of the legacy HMMA path because the legacy mma.sync warp-level register layout doesn't have a hardware equivalent in tcgen05 (which uses tensor memory not registers).

## Test
`tests/bench_mma_fp8_real.cu` — uses runtime `u2` perturbation + per-iter operand rotation to defeat ptxas constant-fold collapse (which otherwise reduces N×mma to 1×mma + N×FADD).
