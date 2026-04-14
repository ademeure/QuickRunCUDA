// Tensor core throughput probe — warp-level MMA.
// Goal: confirm pipe_tensor hosts HMMA/IMMA/QMMA/OMMA and measure peak.

#include <mma.h>

#ifndef OP
#define OP 0
#endif
#ifndef ITERS_INNER
#define ITERS_INNER 32
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    using namespace nvcuda::wmma;

#if OP == 0
    // FP16×FP16 → FP32 accumulator, 16×16×16
    fragment<matrix_a, 16, 16, 16, half, row_major> a;
    fragment<matrix_b, 16, 16, 16, half, col_major> b;
    fragment<accumulator, 16, 16, 16, float> c;
    fill_fragment(a, __float2half(1.0001f));
    fill_fragment(b, __float2half(1.0001f));
    fill_fragment(c, 0.0f);
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ITERS_INNER; j++) mma_sync(c, a, b, c);
    }
    if (c.x[0] == 1.23456f) reinterpret_cast<float*>(C)[blockIdx.x * blockDim.x + threadIdx.x] = c.x[0];
#elif OP == 1
    // BF16×BF16 → FP32, 16×16×16
    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b;
    fragment<accumulator, 16, 16, 16, float> c;
    fill_fragment(a, __float2bfloat16(1.0001f));
    fill_fragment(b, __float2bfloat16(1.0001f));
    fill_fragment(c, 0.0f);
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ITERS_INNER; j++) mma_sync(c, a, b, c);
    }
    if (c.x[0] == 1.23456f) reinterpret_cast<float*>(C)[blockIdx.x * blockDim.x + threadIdx.x] = c.x[0];
#elif OP == 2
    // TF32 16×16×8 — input f32 (19-bit), accum f32
    fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a;
    fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> b;
    fragment<accumulator, 16, 16, 8, float> c;
    fill_fragment(a, 1.0001f);
    fill_fragment(b, 1.0001f);
    fill_fragment(c, 0.0f);
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ITERS_INNER; j++) mma_sync(c, a, b, c);
    }
    if (c.x[0] == 1.23456f) reinterpret_cast<float*>(C)[blockIdx.x * blockDim.x + threadIdx.x] = c.x[0];
#elif OP == 3
    // INT8 × INT8 → INT32, 16×16×16
    fragment<matrix_a, 16, 16, 16, signed char, row_major> a;
    fragment<matrix_b, 16, 16, 16, signed char, col_major> b;
    fragment<accumulator, 16, 16, 16, int> c;
    fill_fragment(a, (signed char)2);
    fill_fragment(b, (signed char)3);
    fill_fragment(c, 0);
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ITERS_INNER; j++) mma_sync(c, a, b, c);
    }
    if (c.x[0] == 0x42424242) reinterpret_cast<int*>(C)[blockIdx.x * blockDim.x + threadIdx.x] = c.x[0];
#endif
}
