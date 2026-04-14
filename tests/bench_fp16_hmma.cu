// FP16 HMMA with FP16 vs FP32 accumulator — cost comparison.

#include <mma.h>

#ifndef OP
#define OP 0
#endif
#ifndef ITERS_INNER
#define ITERS_INNER 32
#endif

extern "C" __global__ __launch_bounds__(128, 2)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    using namespace nvcuda::wmma;

#if OP == 0  // FP16 × FP16 → FP32 (16×16×16)
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
    if (c.x[0] == 1.23456f) ((float*)C)[blockIdx.x * blockDim.x + threadIdx.x] = c.x[0];
#elif OP == 1  // FP16 × FP16 → FP16 (cheaper accum!)
    fragment<matrix_a, 16, 16, 16, half, row_major> a;
    fragment<matrix_b, 16, 16, 16, half, col_major> b;
    fragment<accumulator, 16, 16, 16, half> c;
    fill_fragment(a, __float2half(1.0001f));
    fill_fragment(b, __float2half(1.0001f));
    fill_fragment(c, __float2half(0.0f));
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ITERS_INNER; j++) mma_sync(c, a, b, c);
    }
    if (__half2float(c.x[0]) == 1.23456f) ((float*)C)[blockIdx.x * blockDim.x + threadIdx.x] = __half2float(c.x[0]);
#elif OP == 2  // Inline mma.sync.aligned with FP16 accumulator
    unsigned int a0=threadIdx.x, a1=threadIdx.x*3, a2=threadIdx.x*5, a3=threadIdx.x*7;
    unsigned int b0=threadIdx.x*11, b1=threadIdx.x*13;
    unsigned int d0=0, d1=0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ITERS_INNER; j++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                : "+r"(d0), "+r"(d1)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
        }
    }
    if (d0 == (unsigned)seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = d0+d1;
#endif
}
