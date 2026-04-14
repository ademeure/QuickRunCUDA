// Inline PTX mma.sync for FP8 (QMMA) and FP4 (OMMA) on Blackwell.

#ifndef OP
#define OP 0
#endif
#ifndef ITERS_INNER
#define ITERS_INNER 32
#endif

extern "C" __global__ __launch_bounds__(128, 2)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Each lane holds 4 b32 regs of A, 2 b32 regs of B, 4 f32 C accumulators.
    // For m16n8k32 FP8 / m16n8k64 FP4.
    unsigned int a0 = threadIdx.x | 0x11111111u;
    unsigned int a1 = threadIdx.x | 0x22222222u;
    unsigned int a2 = threadIdx.x | 0x33333333u;
    unsigned int a3 = threadIdx.x | 0x44444444u;
    unsigned int b0 = threadIdx.x | 0x55555555u;
    unsigned int b1 = threadIdx.x | 0x66666666u;
    float c0 = 1.0f, c1 = 1.0f, c2 = 1.0f, c3 = 1.0f;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ITERS_INNER; j++) {
#if OP == 0  // FP8 e4m3 × e4m3 m16n8k32 → f32
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif OP == 1  // FP8 e5m2 × e5m2 m16n8k32 → f32
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif OP == 2  // BF16 m16n8k16 → f32 (for comparison)
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif OP == 3  // FP16 m16n8k16 → f16 (FP16 accumulator — cheaper!)
            unsigned int d0 = c0, d1 = c1;
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                : "+r"(d0), "+r"(d1)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            c0 = __int_as_float(d0); c1 = __int_as_float(d1);
#endif
        }
    }
    if (c0 == 1.23456f) ((float*)C)[blockIdx.x * blockDim.x + threadIdx.x] = c0 + c1 + c2 + c3;
}
