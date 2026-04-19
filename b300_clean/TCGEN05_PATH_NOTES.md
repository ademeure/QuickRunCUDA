# tcgen05 on B300 — what we learned this session (2026-04-18/19)

Working tcgen05 path is now CONFIRMED on B300 sm_103a. Three key
discoveries beyond just "it works":

## 1. SASS family for tcgen05

From `cuobjdump --dump-sass` of CUTLASS 01_mma_sm100 example running on B300:

| PTX                          | SASS                                | Purpose                  |
|------------------------------|-------------------------------------|--------------------------|
| `tcgen05.alloc`              | `UTCATOMSWS.FIND_AND_SET.ALIGN`     | Atomic claim TMEM region |
| `tcgen05.mma.kind::f16`      | `UTCHMMA gdesc[UR], gdesc[UR], tmem[UR], tmem[UR], idesc[UR]` | The mma itself |
| `tcgen05.commit` / barrier   | `UTCBAR [UR], URZ`                  | Sync after mma           |
| `tcgen05.dealloc`            | `UTCATOMSWS.AND URZ, UR`            | Release TMEM region      |

Operand format: all use uniform registers (UR) for descriptors. Predicates
gate the instruction (`@UP0`, `@UPT` etc.).

## 2. tcgen05.cp is OPTIONAL for basic GEMM

Critical finding: UTCHMMA can read SMEM directly via SMEM descriptors
(`gdesc[UR]`). NO `tcgen05.cp` (SMEM→TMEM) needed for the input operands!

The CUTLASS minimal example uses:
1. `cooperative_copy<128>(threadIdx.x, gA, sA)` — standard SMEM load
2. `tcgen05.alloc` for output TMEM
3. `tcgen05.mma` reads `sA`/`sB` via descriptors, writes `tD`
4. Standard `tcgen05.ld` reads `tD` to RMEM
5. `tcgen05.dealloc`

So my earlier failed attempts at `tcgen05.cp.shape::128x128b` were unnecessary
— that op is for explicit SMEM→TMEM staging which most GEMMs don't need.

## 3. Compile + runtime recipe (S2 breakthrough)

Discovered this session — historical "alloc/dealloc hung" failure root-caused:

```bash
# Compile flag: explicit compute_103a target (NOT just -arch=sm_103a):
nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 ...
```

Runtime recipe — `.sync.aligned` instructions need ALL warp threads:

```cuda
// CORRECT — all 32 lanes execute
asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
             :: "r"(smem_ptr));
asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
// ... use tmem ...
asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;"
             :: "r"(tmem_addr));
```

```cuda
// WRONG — single-thread guard hangs the warp barrier:
if (threadIdx.x == 0) {
    asm volatile("tcgen05.alloc... .sync.aligned ...");  // 31 lanes hang waiting
}
```

## 4. CUTLASS examples on B300

CUTLASS Blackwell tutorials (`/root/cutlass/examples/cute/tutorial/blackwell/`)
work on B300 sm_103a after a one-line patch:

```diff
- if ((props.major != 10) || (props.major == 10 && props.minor > 1)) {
+ if (0) {  // bypass version check; tcgen05 works on sm_103a
```

`70_blackwell_fp16_gemm.cu` (the full GEMM benchmark) needs deeper
sm_103a-specific template specialization patches — beyond simple version
check. Use cuBLAS for production tcgen05 GEMM benchmarking instead.

## 5. Performance reality

- Small-shape CUTLASS tutorial (FP16 512×1024×256): 126 ms/launch (overhead-bound)
- For real perf: cuBLAS hits 2237 TFLOPS = 90% of NVIDIA spec at 8K³ (per A5)
- cuBLAS's algoId=66 IS the tcgen05 family (per A5 + power data S3)
- Direct PTX tcgen05 only matters for fused custom kernels (FlashAttention etc.)

## What's still open

- tcgen05.cp shape syntax (e.g., for explicit data movement patterns)
- Instruction descriptor (idesc) bitfield format
- 2-CTA cluster mma patterns (cta_group::2)
- TMEM hierarchy: how columns map to physical TMEM banks
- TFLOPS measurement at large M/N/K via timed CUTLASS benchmark
